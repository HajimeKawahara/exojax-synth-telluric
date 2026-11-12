"""Compare synthetic opacity against isotope-specific direct ExoJAX Voigt sums."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import generate as generator
import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

from exojax.database.core.broadening import doppler_sigma, gamma_hitran, gamma_natural
from exojax.database.core.line_strength import line_strength
from exojax.opacity.lpf.lpf import xsvector
from exojax.utils.constants import Patm, c

from wavelength import load_wavelength


ROOT = Path(__file__).resolve().parent


def direct_calculator(model, name, grid):
    """Sum all retained lines and all layers without MODIT interpolation."""
    data = model.data[name]
    pressure = jnp.asarray(model.profile["pressure_bar"])
    temperature = jnp.asarray(model.profile["temperature_K"])
    isotope = np.searchsorted(data["partition_isotopes"], data["iso"])
    strengths = []
    for temp in np.asarray(temperature):
        ratios = np.array([np.interp(temp, data["temperatures"], q) /
                           np.interp(296., data["temperatures"], q)
                           for q in data["partition_sums"]])[isotope]
        strengths.append(np.asarray(line_strength(
            temp, np.log(data["sw"]), data["nu"], data["elower"], ratios, 296.)))
    strengths = jnp.asarray(strengths)
    mass = jnp.asarray(data["line_mass"])
    dominant_mass = data["line_mass"][np.argmax(data["sw"])]
    nu = jnp.asarray(data["nu"])
    offset = jnp.asarray(grid)[None, :] - nu[:, None]

    @jax.jit
    def calculate(factors, columns, self_vmr, shifts, isotopes):
        def layer(total, values):
            p, t, strength, column, vmr = values
            width = gamma_hitran(p, t, p * vmr, data["n_air"],
                                 data["gamma_air"] * factors[1],
                                 data["gamma_self"] * factors[2]) + gamma_natural(data["a"])
            sigma = doppler_sigma(nu, t, jnp.where(isotopes, mass, dominant_mass))
            shift = shifts * jnp.asarray(data["delta_air"]) * p * (1. - vmr) / Patm
            xs = xsvector(offset - shift[:, None], sigma, width, strength * factors[0])
            return total + column * xs, None
        value, _ = jax.lax.scan(layer, jnp.zeros(len(grid)),
                                (pressure, temperature, strengths, columns, self_vmr))
        return value
    return calculate


def metrics(reference, prediction):
    difference = np.asarray(prediction) - np.asarray(reference)
    return dict(max_abs=float(np.max(np.abs(difference))),
                rms=float(np.sqrt(np.mean(difference**2))))


def assembled_convergence(model, seed, group, target_nu):
    """Compare complete PHOENIX-times-telluric flux at two native spacings."""
    refined = generator.Telluric(model.dv / 2)
    stars = dict(np.load(ROOT / "cache/phoenix/library.npz"))
    pwv, airmass, rv = 12.37, 2.5, 30.
    far = dict(np.load(ROOT / "cache/far_wings.npz"))
    if int(far["seed"]) != seed or not far["valid_catalog"][group]:
        raise ValueError("Far-wing seed or catalog does not match")
    outputs = []
    for current in (model, refined):
        tau = np.zeros(len(current.grid))
        for k, name in enumerate(generator.SPECIES):
            factors = generator.draw_factors(current.data[name], name, [seed, group, k, 17])
            tau += current.tau(name, factors, pwv)
            if name == "H2O":
                anchors = PchipInterpolator(far["pwv_nodes"], far[name][group], axis=0)(pwv)
            else:
                fraction = (pwv - .05) / 24.95
                anchors = (1-fraction)*far[name][group, 0] + fraction*far[name][group, 1]
            tau += np.interp(current.grid, far["nu"], anchors)
        rest_wave = (1e4/current.grid) / np.sqrt((1+rv/c)/(1-rv/c))
        stellar = np.interp(rest_wave, stars["wave_um"], stars["flux"][group])
        flux = gaussian_filter1d(stellar * np.exp(-airmass*tau),
                                 c/45000./np.sqrt(8*np.log(2))/current.dv,
                                 mode="nearest")
        outputs.append(np.interp(target_nu, current.grid, flux))
    return dict(group=group, pwv_mm=pwv, airmass=airmass, stellar_velocity_kms=rv,
                pixels=len(target_nu), coarse_dv_kms=float(model.dv),
                fine_dv_kms=float(refined.dv), comparison=metrics(outputs[1], outputs[0]),
                far_wings_sha256=generator.sha256(ROOT / "cache/far_wings.npz"),
                caveat="Grid-convergence check; both models use MODIT and the same randomized far wings and molecular priors")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", nargs="+", default=["H2O", "CH4", "N2O"])
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--group", nargs="+", type=int, default=[0, 40])
    parser.add_argument("--dv", type=float, default=.25)
    parser.add_argument("--reference-refinement", type=int, default=2)
    parser.add_argument("--pwv", nargs="+", type=float, default=[12.37])
    parser.add_argument("--assembled", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results/physics_validation.json")
    args = parser.parse_args()
    if args.reference_refinement < 1:
        parser.error("--reference-refinement must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    model = generator.Telluric(args.dv)
    target_nu = 1e4 / load_wavelength()
    resolution_sigma = c / 45000. / np.sqrt(8 * np.log(2)) / model.dv
    report = {
        "generator_sha256": generator.sha256(ROOT / "generate.py"),
        "validator_sha256": generator.sha256(__file__),
        "model_dv_kms": float(model.dv), "gamma_nodes": 48,
        "reference_refinement": args.reference_refinement,
        "seed": args.seed, "group": args.group,
        "reference": "Direct ExoJAX LPF for the 4330-4440 cm^-1 core; isotope-specific Doppler widths and all layers. Far wings enter only the assembled convergence check.",
        "selection": "Raw HITRAN strongest in-order line per species, no observed flux",
        "airmass": 2.5, "resolving_power": 45000., "cases": [],
    }
    tables = {}
    report["pwv_table_paths"] = {}
    for group in args.group:
        table_path = ROOT / f"cache/tau_{model.cache_key}_{args.seed}_{group:03d}_{len(model.grid)}.npz"
        tables[group] = dict(np.load(table_path)) if table_path.exists() else None
        report["pwv_table_paths"][group] = str(table_path.relative_to(ROOT)) if tables[group] is not None else None
    for name in args.species:
        data = model.data[name]
        selected = (data["nu"] >= target_nu.min()+1) & (data["nu"] <= target_nu.max()-1)
        center = (float(data["nu"][np.flatnonzero(selected)[np.argmax(data["sw"][selected])]])
                  if selected.any() else float(np.clip(
                      data["nu"][np.argmin(np.abs(data["nu"] - np.median(target_nu)))],
                      target_nu.min(), target_nu.max())))
        selected_grid = np.flatnonzero(np.abs(model.grid - center) <= .9)
        grid = model.grid[selected_grid]
        reference_grid = np.geomspace(grid[0], grid[-1],
                                      1 + (len(grid) - 1) * args.reference_refinement)
        target = target_nu[np.abs(target_nu - center) <= .65]
        calculator = direct_calculator(model, name, reference_grid)
        k = generator.SPECIES.index(name)

        def sampled(tau, reference=False):
            spectral_grid = reference_grid if reference else grid
            sigma = resolution_sigma * (args.reference_refinement if reference else 1)
            transmission = gaussian_filter1d(np.exp(-2.5 * tau), sigma, mode="nearest")
            return np.interp(target, spectral_grid, transmission)

        for group in args.group:
            factors = generator.draw_factors(data, name, [args.seed, group, k, 17])
            table = tables[group]
            for pwv in args.pwv:
                columns, vmr = model.columns(pwv, name)
                exact = np.asarray(calculator(factors, columns, vmr, 1., True))
                common_mass = np.asarray(calculator(factors, columns, vmr, 1., False))
                unshifted = np.asarray(calculator(factors, columns, vmr, 0., True))
                modit = model.tau(name, factors, pwv)[selected_grid]
                exact_flux = sampled(exact, reference=True)
                case = {
                    "group": group, "species": name, "pwv_mm": pwv, "center_cm1": center,
                    "native_pixels": len(target), "retained_lines": len(data["nu"]),
                    "modit_vs_direct": metrics(exact_flux, sampled(modit)),
                    "common_mass_vs_isotopes": metrics(exact_flux, sampled(common_mass, reference=True)),
                    "pressure_shift_effect": metrics(exact_flux, sampled(unshifted, reference=True)),
                    "direct_transmission_min": float(exact_flux.min()),
                }
                if name != "H2O":
                    case["fixed_self_width_under_abundance_variation"] = {}
                    for scale in (.9, 1.1):
                        varied = np.asarray(calculator(factors, columns*scale, vmr*scale, 1., True))
                        case["fixed_self_width_under_abundance_variation"][str(scale)] = metrics(
                            sampled(varied, reference=True), sampled(exact*scale, reference=True))
                if table is not None:
                    if name == "H2O":
                        approximation = PchipInterpolator(generator.PWV_NODES, table[name], axis=0)(pwv)
                    else:
                        fraction = (pwv - .05) / 24.95
                        approximation = (1 - fraction) * table[name][0] + fraction * table[name][1]
                    case["pwv_interpolation"] = metrics(sampled(modit), sampled(approximation[selected_grid]))
                report["cases"].append(case)
                report["elapsed_seconds"] = time.monotonic() - started
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(case), flush=True)
    if args.assembled:
        report["assembled_grid_convergence"] = assembled_convergence(model, args.seed, args.group[0], target_nu)
        print(json.dumps(report["assembled_grid_convergence"]), flush=True)
    report["strongest_line_max_error"] = max(case["modit_vs_direct"]["max_abs"] for case in report["cases"])
    report["strongest_line_error_below_5e_4"] = report["strongest_line_max_error"] <= 5e-4
    report["elapsed_seconds"] = time.monotonic() - started
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output} after {time.monotonic()-started:.1f} s", flush=True)


if __name__ == "__main__":
    main()

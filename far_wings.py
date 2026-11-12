"""Prepare randomized smooth wings from the remaining available HITRAN lines."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time

import generate
import numpy as np

from prepare_inputs import RAW_FIELDS, uncertainty_cv
from wavelength import load_wavelength


ROOT = Path(__file__).resolve().parent


def load_outside(name):
    """Read public raw lines outside the exact MODIT interval."""
    path = ROOT / f"cache/hitran/{name}.par"
    source = json.loads(path.with_suffix(".json").read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != source["sha256"]:
        raise ValueError(f"HITRAN raw hash mismatch for {name}")
    all_rows = path.read_text().splitlines()
    indices = np.array([i for i, row in enumerate(all_rows)
                        if not 4330. <= float(row[3:15]) <= 4440.])
    rows = [all_rows[i] for i in indices]
    data = {key: np.array([float(row[start:stop]) for row in rows])
            for key, (start, stop) in RAW_FIELDS.items()}
    data["iso"] = np.array([
        int(row[2]) if row[2].isdigit() and row[2] != "0"
        else 10 if row[2] == "0" else ord(row[2])-ord("A")+11 for row in rows])
    data["ierr_codes"] = np.array([[int(c) if c.isdigit() else 0 for c in row[127:133]]
                                  for row in rows], dtype=np.int8)
    for key, column in (("strength", 1), ("gamma_air", 2), ("gamma_self", 3)):
        data[key+"_cv"] = uncertainty_cv(data["ierr_codes"][:, column])
    with np.load(ROOT / f"cache/hitran/{name}.npz") as cached:
        isotope = np.searchsorted(cached["partition_isotopes"], data["iso"])
        temperatures = cached["temperatures"].copy()
        partition = cached["partition_sums"].copy()
    return data, isotope, temperatures, partition, dict(
        raw_sha256=digest, outside_count=len(rows), source_provenance=source,
        raw_index_sha256=hashlib.sha256(indices.tobytes()).hexdigest())


def layer_strength(data, isotope, temperatures, partition, layer_temperature):
    """Use the HITRAN temperature law independently of the numerical grid."""
    result = []
    for temp in layer_temperature:
        qr = np.array([np.interp(temp, temperatures, q) / np.interp(296., temperatures, q)
                       for q in partition])[isotope]
        result.append(data["sw"]/qr * np.exp(-1.4387770*data["elower"]*(1/temp-1/296.))
                      * np.expm1(-1.4387770*data["nu"]/temp)
                      / np.expm1(-1.4387770*data["nu"]/296.))
    return np.array(result)


def wing_basis(data, strengths, profile, name, pwv, anchor):
    """Factor the large-detuning Lorentz tail into line-wise linear terms."""
    water = profile["water_column_per_mm_cm2"] * pwv
    dry = profile["dry_air_column_cm2"] - water * 18.01528/28.9644
    column = water if name == "H2O" else dry * generate.VMR[name]
    self_vmr = column / (water+dry)
    air = np.zeros((len(data["nu"]), len(anchor)))
    own = np.zeros_like(air)
    natural = np.zeros_like(air)
    for p, temp, strength, col, qself in zip(profile["pressure_bar"],
                                           profile["temperature_K"], strengths,
                                           column, self_vmr):
        shift = data["delta_air"] * p * (1-qself) / 1.01325
        separation = anchor[None, :] - (data["nu"]+shift)[:, None]
        common = col * strength[:, None] / (np.pi*separation**2)
        temperature_scale = (296./temp)**data["n_air"]
        air += common * (temperature_scale*data["gamma_air"]*p*(1-qself)/1.01325)[:, None]
        own += common * (temperature_scale*data["gamma_self"]*p*qself/1.01325)[:, None]
        natural += common * (2.6544188e-12*data["a"])[:, None]
    return air, own, natural


def exact_lorentz(data, strengths, profile, name, pwv, anchor, factors):
    """Check the neglected gamma-squared denominator at the same anchors."""
    water = profile["water_column_per_mm_cm2"] * pwv
    dry = profile["dry_air_column_cm2"] - water*18.01528/28.9644
    column = water if name == "H2O" else dry*generate.VMR[name]
    self_vmr = column/(water+dry)
    tau = np.zeros(len(anchor))
    for p, temp, strength, col, qself in zip(profile["pressure_bar"],
                                           profile["temperature_K"], strengths,
                                           column, self_vmr):
        width = (296./temp)**data["n_air"] * p/1.01325 * (
            data["gamma_air"]*factors[1]*(1-qself) + data["gamma_self"]*factors[2]*qself)
        width += 2.6544188e-12*data["a"]
        separation = anchor[None, :] - (data["nu"] + data["delta_air"]*p*(1-qself)/1.01325)[:, None]
        tau += col * np.sum((strength*factors[0]*width)[:, None] /
                           (np.pi*(separation**2+width[:, None]**2)), axis=0)
    return tau


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args()
    started = time.monotonic()
    target = 1e4/load_wavelength()
    anchor = np.linspace(target.min()-2., target.max()+2., 17)
    profile = dict(np.load(ROOT / "cache/standard_atmosphere.npz"))
    manifest = json.loads((ROOT / "cache/phoenix/manifest.json").read_text())
    count = len(manifest["models"])
    valid = np.isin(np.arange(count), generate.eligible_catalogs())
    output = dict(nu=anchor, pwv_nodes=generate.PWV_NODES, seed=args.seed,
                  catalog_ids=np.arange(count), valid_catalog=valid)
    report = {
        "seed": args.seed, "catalog_count": count,
        "unused_catalogs": np.flatnonzero(~valid).tolist(),
        "priors_sha256": generate.sha256(ROOT / "priors.json"),
        "factor_seed": "[seed, catalog_id, species_index_in_generate_SPECIES, 29]",
        "line_order": "Original raw-file order, filtered to nu<4330 or nu>4440",
        "method": "Large-detuning Voigt asymptote S*gamma/(pi*delta_nu**2); temperature, pressure shifts, self broadening and hydrostatic columns included",
        "prior": "Same generate.draw_factors priors, independent RNG substream from MODIT lines",
        "prior_function_sha256": hashlib.sha256(inspect.getsource(generate.draw_factors).encode()).hexdigest(),
        "script_sha256": generate.sha256(__file__),
        "profile_sha256": generate.sha256(ROOT / "cache/standard_atmosphere.npz"),
        "retained_core_interval_cm1": [4330., 4440.],
        "anchor_nu_cm1": anchor.tolist(), "pwv_nodes_mm": generate.PWV_NODES.tolist(),
        "raw_support_cm1": [4050., 6900.],
        "limitations": "Available raw support only; no physical H2O continuum or MT_CKD model. Far wings retain Voigt/Lorentz extrapolation assumptions; this is not a measured continuum.",
        "species": {},
    }
    for k, name in enumerate(generate.SPECIES):
        data, isotope, temperatures, partition, info = load_outside(name)
        strength = layer_strength(data, isotope, temperatures, partition, profile["temperature_K"])
        # Float64 factors make the table reconstructible directly from its seed.
        factors = np.zeros((count, 3, len(data["nu"])))
        for group in np.flatnonzero(valid):
            factors[group] = generate.draw_factors(data, name, [args.seed, int(group), k, 29])
        s_air = np.ascontiguousarray(factors[:, 0]*factors[:, 1])
        s_self = np.ascontiguousarray(factors[:, 0]*factors[:, 2])
        s_only = np.ascontiguousarray(factors[:, 0])
        nodes = generate.PWV_NODES if name == "H2O" else generate.PWV_NODES[[0, -1]]
        values = np.zeros((count, len(nodes), len(anchor)))
        for node, pwv in enumerate(nodes):
            air, own, natural = wing_basis(data, strength, profile, name, pwv, anchor)
            values[:, node] = s_air @ air + s_self @ own + s_only @ natural
        # Validate a stronger-than-typical wet column and a random catalog.
        exact = exact_lorentz(data, strength, profile, name, 25., anchor, factors[0])
        exact_check = np.max(np.abs(np.exp(-2.5*values[0, -1])-np.exp(-2.5*exact)))
        # Check interpolation in wavelength against directly computed midpoints.
        midpoints = (anchor[1:]+anchor[:-1])/2
        mid_air, mid_self, mid_natural = wing_basis(data, strength, profile, name, 25., midpoints)
        mid_tau = s_air[0] @ mid_air + s_self[0] @ mid_self + s_only[0] @ mid_natural
        interpolated = np.interp(midpoints, anchor, values[0, -1])
        interpolation_check = np.max(np.abs(np.exp(-2.5*mid_tau)-np.exp(-2.5*interpolated)))
        info.update(airmass_2p5_max_asymptote_flux_error_group0=float(exact_check),
                    airmass_2p5_max_anchor_interpolation_flux_error_group0=float(interpolation_check),
                    max_vertical_tau=float(values.max()), min_vertical_tau=float(values[valid].min()))
        report["species"][name] = info
        output[name] = values
        print(f"{name}: {len(data['nu'])} outside lines, {values.shape}, asymptote error {exact_check:.3g}, anchor error {interpolation_check:.3g}", flush=True)
    report["elapsed_seconds"] = time.monotonic()-started
    destination = ROOT / "cache/far_wings.npz"
    temporary = destination.with_name("far_wings.partial.npz")
    np.savez_compressed(temporary, **output)
    temporary.replace(destination)
    report["npz_sha256"] = generate.sha256(destination)
    (ROOT / "cache/far_wings.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"Saved {destination} after {time.monotonic()-started:.1f} s", flush=True)


if __name__ == "__main__":
    main()

"""Generate grouped, noiseless PHOENIX x ExoJAX telluric training spectra."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache/matplotlib"))

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

from exojax.database.core.broadening import doppler_sigma, gamma_hitran, gamma_natural
from exojax.database.core.line_strength import line_strength
from exojax.opacity.initspec import init_modit
from exojax.opacity.modit.modit import xsvector_zeroscan
from exojax.utils.constants import Patm, c

from wavelength import LABEL_COLUMNS, load_wavelength

SPECIES = ("H2O", "CO2", "CH4", "N2O", "CO", "O2", "O3")
# Generic dry-air VMRs, independent of the target observation.
VMR = dict(CO2=420e-6, CH4=1.9e-6, N2O=335e-9, CO=100e-9,
           O2=0.20946, O3=40e-9)
PWV_NODES = np.linspace(0.05, 25., 11)
PRIORS = json.loads((ROOT / "priors.json").read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def draw_factors(data, name, seed):
    """Draw alternate molecular constants, shared within one synthetic series."""
    rng = np.random.default_rng(seed)
    factors = []
    for k, field in enumerate(("strength", "gamma_air", "gamma_self")):
        cv = data[field + "_cv"].copy()
        multipliers = PRIORS["finite_code_multiplier"][field]
        multiplier = multipliers.get(name, multipliers["default"])
        # Unknown / open-ended error classes retain the explicit 50% fallback.
        cv[data["ierr_codes"][:, k + 1] >= 4] *= multiplier
        cv[data["ierr_codes"][:, k + 1] < 4] = PRIORS["unknown_uncertainty_cv"]
        sigma = np.sqrt(np.log1p(cv * cv))
        factors.append(np.exp(sigma * rng.normal(size=len(cv)) - sigma**2 / 2))
    return np.asarray(factors)


class Telluric:
    """MODIT layer opacity, with per-line intensities and broadening widths."""

    def __init__(self, dv=0.25):
        n = int(np.ceil(c * np.log(4440. / 4330.) / dv)) + 1
        # Smooth FFT lengths make repeated CPU synthesis inexpensive.
        from scipy.fft import next_fast_len
        n = next_fast_len(n, real=True)
        n += n % 2
        self.grid = np.geomspace(4330., 4440., n)
        self.dv = c * np.log(self.grid[1] / self.grid[0])
        sources = [ROOT / "cache/standard_atmosphere.npz"]
        sources += [ROOT / f"cache/hitran/{name}.npz" for name in SPECIES]
        physics = inspect.getsource(Telluric) + inspect.getsource(draw_factors)
        physics += json.dumps(dict(vmr=VMR, pwv_nodes=PWV_NODES.tolist()))
        physics += json.dumps(PRIORS, sort_keys=True)
        self.cache_key = hashlib.sha256((physics + "".join(map(sha256, sources))).encode()).hexdigest()[:10]
        self.profile = dict(np.load(ROOT / "cache/standard_atmosphere.npz"))
        self.data = {}
        self.compute = {}
        for name in SPECIES:
            d = dict(np.load(ROOT / f"cache/hitran/{name}.npz"))
            self.data[name] = d
            self.compute[name] = self._calculator(d)

    def _calculator(self, d):
        p = self.profile["pressure_bar"]
        t = self.profile["temperature_K"]
        # Keep isotope-dependent strengths/partition sums. The MODIT common
        # Doppler mass uses the dominant isotope; validate after R=45000 LSF.
        mass = d["line_mass"][np.argmax(d["sw"])]
        cont, index, resolution, pm = init_modit(d["nu"], self.grid)
        isotope = np.searchsorted(d["partition_isotopes"], d["iso"])
        strengths = []
        for temp in t:
            ratio = np.array([np.interp(temp, d["temperatures"], q) /
                              np.interp(296., d["temperatures"], q)
                              for q in d["partition_sums"]])[isotope]
            strengths.append(np.asarray(line_strength(temp, np.log(d["sw"]),
                             d["nu"], d["elower"], ratio, 296.)))
        strengths = jnp.asarray(strengths)
        nsigma = np.asarray(doppler_sigma(np.ones(len(t)) * resolution, t, mass))
        nu = jnp.asarray(d["nu"])
        grid = jnp.asarray(self.grid)

        @jax.jit
        def calculate(factors, columns, self_vmr):
            def layer(total, inputs):
                pressure, temperature, sigma, strength, column, qself = inputs
                width = gamma_hitran(pressure, temperature, pressure * qself,
                                     d["n_air"], d["gamma_air"] * factors[1],
                                     d["gamma_self"] * factors[2]) + gamma_natural(d["a"])
                normalized = width * resolution / nu
                # Bracket every actual width; fixed node count avoids recompilation.
                ngamma_grid = jnp.geomspace(jnp.min(normalized)*.99,
                                           jnp.max(normalized)*1.01, 48)
                shifted = nu + jnp.asarray(d["delta_air"]) * pressure * (1-qself) / Patm
                position = jnp.interp(shifted, grid, jnp.arange(len(grid), dtype=float))
                shifted_index = jnp.clip(jnp.floor(position).astype(int), 0, len(grid)-2)
                shifted_cont = position - shifted_index
                xs = xsvector_zeroscan(shifted_cont, shifted_index, resolution, pm, sigma,
                                       normalized, strength * factors[0], grid,
                                       ngamma_grid)
                return total + column * xs, None
            value, _ = jax.lax.scan(layer, jnp.zeros(len(grid)),
                                    (jnp.asarray(p), jnp.asarray(t), jnp.asarray(nsigma),
                                     strengths, columns, self_vmr))
            return value
        return calculate

    def columns(self, pwv, name):
        water = self.profile["water_column_per_mm_cm2"] * pwv
        dry = self.profile["dry_air_column_cm2"] - water * 18.01528 / 28.9644
        total = dry + water
        columns = water if name == "H2O" else dry * VMR[name]
        return columns, columns / total

    def tau(self, name, factors, pwv):
        columns, vmr = self.columns(pwv, name)
        result = np.asarray(self.compute[name](jnp.asarray(factors),
                                              jnp.asarray(columns), jnp.asarray(vmr)))
        if not np.all(np.isfinite(result)) or result.min() < -1e-7:
            raise ValueError(f"Invalid opacity: {name}: {result.min()}")
        return np.maximum(result, 0.)

    def table(self, group, seed):
        path = ROOT / f"cache/tau_{self.cache_key}_{seed}_{group:03d}_{len(self.grid)}.npz"
        if path.exists():
            return dict(np.load(path))
        arrays = {}
        for k, name in enumerate(SPECIES):
            factors = draw_factors(self.data[name], name, [seed, group, k, 17])
            arrays[name + "_factors"] = factors.astype(np.float32)
            # Water self broadening requires a PWV table. Dry columns depend
            # linearly on PWV; their tiny self-width change is checked separately.
            nodes = PWV_NODES if name == "H2O" else PWV_NODES[[0, -1]]
            arrays[name] = np.array([self.tau(name, factors, pwv) for pwv in nodes])
        temporary = path.with_suffix(f".{os.getpid()}.partial.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
        return arrays


def eligible_catalogs():
    """Exclude upstream-interpolated PHOENIX nodes before assigning splits."""
    manifest = json.loads((ROOT / "cache/phoenix/manifest.json").read_text())
    return np.array([i for i, item in enumerate(manifest["models"])
                     if not item.get("interpolated_by_library", False)])


def parameters(n, stars, seed):
    rng = np.random.default_rng(seed)
    nstar = len(stars["teff"])
    # Each row has independent atmosphere/RV draws; a molecular realization
    # stays fixed within a molecular-catalog series and exactly one split.
    eligible = eligible_catalogs()
    series = rng.choice(eligible, n)
    order = np.random.default_rng([seed, 99]).permutation(eligible)
    split = np.zeros(nstar, dtype=np.int8)
    split[order[int(.8*len(order)):int(.9*len(order))]] = 1
    split[order[int(.9*len(order)):]] = 2
    row_split = split[series]
    template = np.empty(n, dtype=int)
    for code in range(3):
        rows = np.flatnonzero(row_split == code)
        template[rows] = rng.choice(eligible[split[eligible] == code], len(rows))
    return dict(series=series, pwv=rng.uniform(.05, 25., n),
                airmass=rng.uniform(1., 2.5, n), rv=rng.normal(0., 30., n),
                abundance=rng.uniform(.9, 1.1, (n, len(SPECIES)-1)),
                split=row_split, template=template)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--dv", type=float, default=.25)
    parser.add_argument("--output", help="Output filename inside this repository")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--prepare", type=int, nargs=2, metavar=("START", "STOP"))
    args = parser.parse_args()
    if args.n < 1 or not np.isfinite(args.dv) or args.dv <= 0:
        parser.error("--n and --dv must be positive")
    started = time.monotonic()
    model = Telluric(args.dv)
    if args.prepare:
        for group in range(*args.prepare):
            if group not in eligible_catalogs():
                continue
            model.table(group, args.seed)
            print(f"Prepared {group}: {time.monotonic()-started:.1f}s", flush=True)
        return
    if args.benchmark:
        for name in SPECIES:
            factors = draw_factors(model.data[name], name, [args.seed, 0])
            for i in range(2):
                t = time.monotonic()
                tau = model.tau(name, factors, 12.)
                print(name, i, time.monotonic()-t, tau.min(), tau.max(), flush=True)
        return
    stars = dict(np.load(ROOT / "cache/phoenix/library.npz"))
    far = dict(np.load(ROOT / "cache/far_wings.npz"))
    if int(far["seed"]) != args.seed or not np.array_equal(far["pwv_nodes"], PWV_NODES):
        raise ValueError("Far-wing cache seed or PWV nodes do not match")
    far_metadata = json.loads((ROOT / "cache/far_wings.json").read_text())
    if far_metadata["priors_sha256"] != sha256(ROOT / "priors.json"):
        raise ValueError("Priors changed; rerun far_wings.py")
    if far_metadata["profile_sha256"] != sha256(ROOT / "cache/standard_atmosphere.npz"):
        raise ValueError("Atmosphere changed; rerun far_wings.py")
    for name in SPECIES:
        if far_metadata["species"][name]["raw_sha256"] != sha256(ROOT / f"cache/hitran/{name}.par"):
            raise ValueError(f"HITRAN {name} changed; rerun far_wings.py")
    if not np.array_equal(np.flatnonzero(far["valid_catalog"]), eligible_catalogs()):
        raise ValueError("PHOENIX eligibility changed; rerun far_wings.py")
    p = parameters(args.n, stars, args.seed)
    wave = load_wavelength()
    path = ROOT / (args.output or f"stellar_telluric_IGRINS_ordidx7_N{args.n}.h5")
    if path.resolve().parent != ROOT:
        raise ValueError("Output must be directly inside this repository")
    if path.exists():
        raise FileExistsError(path)
    partial = path.with_suffix(".partial.h5")
    with h5py.File(partial, "w") as f:
        f.attrs.update(seed=args.seed, complete=False, resolving_power=45000.,
                       grid_dv_kms=model.dv, noise="none", frame="telluric rest; positive stellar RV is redshift",
                       source_wavelength_sha256=hashlib.sha256(wave.tobytes()).hexdigest(),
                       generator_sha256=sha256(__file__), physics_cache_key=model.cache_key,
                       intensity_multiplier=json.dumps(PRIORS["finite_code_multiplier"]["strength"]),
                       air_width_multiplier=json.dumps(PRIORS["finite_code_multiplier"]["gamma_air"]),
                       unknown_uncertainty_cv=PRIORS["unknown_uncertainty_cv"],
                       far_wings="Randomized Lorentz tails from cached HITRAN4050-6900 outside4330-4440 cm-1; no MT_CKD continuum",
                       leakage_scope=PRIORS["leakage_scope"])
        provenance = f.create_group("provenance")
        for name in ("prior_assessment.json", "priors.json", "data/order7_wavelength_um.txt", "cache/phoenix/manifest.json",
                     "cache/standard_atmosphere.json", "cache/far_wings.json", "cache/far_wings.npz"):
            provenance.attrs[name] = sha256(ROOT / name)
        far_group = f.create_group("far_wings")
        for key, value in far.items():
            far_group.create_dataset(key, data=value)
        far_group.attrs["factor_seed"] = "[seed, catalog_id, species_index, 29]; original raw-row order outside4330-4440"
        lines = f.create_group("lines")
        for name, data in model.data.items():
            group = lines.create_group(name)
            for field in ("nu", "sw", "gamma_air", "gamma_self", "iso", "ierr_codes"):
                group.create_dataset(field, data=data[field], compression="gzip")
        f.create_dataset("wavelength", data=wave).attrs["units"] = "micron"
        flux = f.create_dataset("flux", (args.n, len(wave)), dtype="f4",
                                chunks=(min(64,args.n), len(wave)), compression="gzip", shuffle=True)
        labels = f.create_dataset("labels", data=np.column_stack((p["pwv"],p["airmass"],p["series"])).astype("f4"))
        labels.attrs["columns"] = LABEL_COLUMNS
        labels.attrs["units"] = ["mm", "dimensionless", "group_id"]
        params = f.create_group("parameters")
        params.create_dataset("stellar_velocity_kms", data=p["rv"])
        params.create_dataset("template_id", data=p["template"])
        for key in ("teff", "logg", "metallicity"):
            params.create_dataset(key, data=stars[key][p["template"]])
        params.create_dataset("abundance_scale", data=p["abundance"]).attrs["columns"] = list(SPECIES[1:])
        f.create_dataset("split", data=p["split"]).attrs["names"] = ["train", "validation", "test"]
        f.create_dataset("completed", data=np.zeros(args.n, dtype=bool))
        catalogs = f.create_group("line_realizations")
        for group in np.unique(p["series"]):
            rows = np.flatnonzero(p["series"] == group)
            table = model.table(int(group), args.seed)
            g = catalogs.create_group(str(group))
            for name in SPECIES:
                g.create_dataset(name, data=table[name+"_factors"], compression="gzip").attrs["rows"] = ["strength", "gamma_air", "gamma_self"]
            wet = PchipInterpolator(PWV_NODES, table["H2O"], axis=0)(p["pwv"][rows])
            wet_far = PchipInterpolator(PWV_NODES, far["H2O"][group], axis=0)(p["pwv"][rows])
            for j, row in enumerate(rows):
                tau = wet[j] + np.interp(model.grid, far["nu"], wet_far[j])
                u = (p["pwv"][row]-.05)/24.95
                for k, name in enumerate(SPECIES[1:]):
                    background = np.interp(model.grid, far["nu"],
                                           (1-u)*far[name][group,0]+u*far[name][group,1])
                    tau += p["abundance"][row,k] * ((1-u)*table[name][0]+u*table[name][1]+background)
                transmission = np.exp(-p["airmass"][row]*np.maximum(tau,0.))
                beta = p["rv"][row]/c
                rest_wave = (1e4/model.grid) / np.sqrt((1+beta)/(1-beta))
                if rest_wave.min() < stars["wave_um"].min() or rest_wave.max() > stars["wave_um"].max():
                    raise ValueError("PHOENIX crop does not cover Doppler shift")
                stellar = np.interp(rest_wave, stars["wave_um"], stars["flux"][p["template"][row]])
                combined = gaussian_filter1d(stellar * transmission,
                    c / 45000. / np.sqrt(8*np.log(2)) / model.dv, mode="nearest")
                value = np.interp(1e4/wave, model.grid, combined)
                if not np.all(np.isfinite(value)) or np.any(value < 0):
                    raise ValueError("Invalid generated flux")
                flux[row] = value
            f["completed"][rows] = True
            f.flush()
            print(f"Series {group}: {len(rows)} rows; total {f['completed'][:].sum()}/{args.n}; {time.monotonic()-started:.1f}s", flush=True)
        f.attrs["complete"] = True
    partial.replace(path)
    print(path, flush=True)


if __name__ == "__main__":
    main()

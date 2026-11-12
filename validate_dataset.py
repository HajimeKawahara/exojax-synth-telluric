"""Validate the finished synthetic dataset and plot actual generated examples."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache/matplotlib"))

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kstest

from wavelength import LABEL_COLUMNS, load_wavelength

SPECIES = ("H2O", "CO2", "CH4", "N2O", "CO", "O2", "O3")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values):
    return dict(min=float(np.min(values)), max=float(np.max(values)),
                mean=float(np.mean(values)), std=float(np.std(values)),
                quantiles=np.percentile(values, [1, 5, 50, 95, 99]).tolist())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def make_plot(wave, flux, labels, parameters, target):
    fig = plt.figure(figsize=(15, 12), layout="constrained")
    grid = fig.add_gridspec(4, 3)
    example_rows = []
    for row, (name, temperature) in enumerate((("K", 4400), ("G", 5600),
                                              ("F", 6600), ("A", 8600))):
        ax = fig.add_subplot(grid[row, :2])
        candidates = np.flatnonzero(parameters["teff"] == temperature)
        if not len(candidates):
            ax.text(.5, .5, f"No {name}-range example in this sample",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        candidates = candidates[np.argsort(labels[candidates, 0])]
        chosen = candidates[[len(candidates) // 5, 4 * len(candidates) // 5]]
        for index, color in zip(chosen, ("#0072B2", "#D55E00")):
            text = (f"PWV={labels[index, 0]:.2f} mm, X={labels[index, 1]:.2f}, "
                    f"RV={parameters['stellar_velocity_kms'][index]:+.1f} km/s, "
                    f"[M/H]={parameters['metallicity'][index]:+.1f}")
            ax.plot(wave, flux[index], lw=.8, color=color, label=text)
        example_rows.extend(chosen.tolist())
        ax.set(title=f"{name}-range example: Teff={temperature} K",
               ylabel="Normalized flux", xlim=(wave.min(), wave.max()), ylim=(-.02, 1.04))
        ax.legend(fontsize=7, loc="lower left")
        if row == 3:
            ax.set_xlabel("Vacuum wavelength (micron)")
    histograms = [(labels[:, 0], "PWV (mm)", (.05, 25.)),
                  (labels[:, 1], "Airmass", (1., 2.5)),
                  (parameters["stellar_velocity_kms"], "Stellar RV (km/s)", None)]
    for row, (values, title, bounds) in enumerate(histograms):
        ax = fig.add_subplot(grid[row, 2])
        ax.hist(values, bins=32, density=True, color="#0072B2", alpha=.75)
        if bounds:
            ax.axhline(1 / (bounds[1] - bounds[0]), color="black", lw=1)
        else:
            x = np.linspace(-120, 120, 300)
            ax.plot(x, np.exp(-.5 * (x / 30)**2) / (30 * np.sqrt(2 * np.pi)), color="black")
        ax.set(xlabel=title, ylabel="Density")
    ax = fig.add_subplot(grid[3, 2])
    image = ax.hist2d(parameters["teff"], parameters["metallicity"],
                      bins=[np.arange(3900, 10101, 200), np.arange(-1.75, 1., .5)],
                      cmap="Blues")
    fig.colorbar(image[3], ax=ax, label="Rows")
    ax.set(xlabel="Teff (K)", ylabel="[M/H]")
    fig.suptitle(f"IGRINS order 7: {len(flux):,} synthetic spectra; actual generated examples")
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return example_rows


def check_schema(data, reference_wave, expected_n=None):
    """Reject incomplete output, mislabeled axes and altered wavelength grids."""
    require(bool(data.attrs.get("complete", False)), "Dataset is incomplete")
    n = data["flux"].shape[0]
    require(n > 0 and (expected_n is None or n == expected_n), "Unexpected sample count")
    require(data["completed"].shape == (n,) and data["completed"][:].all(),
            "Completion flags are missing or false")
    for key, shape, dtype in (("flux", (n, len(reference_wave)), "float32"),
                              ("labels", (n, 3), "float32"),
                              ("wavelength", reference_wave.shape, "float64")):
        require(data[key].shape == shape and data[key].dtype == np.dtype(dtype),
                f"Schema mismatch: {key}")
    require(list(data["labels"].attrs["columns"]) == LABEL_COLUMNS,
            "Label columns differ from the documented schema")
    require(data["wavelength"].attrs["units"] == "micron", "Invalid wavelength units")
    wave = data["wavelength"][:]
    require(wave.tobytes() == reference_wave.tobytes(), "Wavelengths are not byte-identical")
    require(data.attrs["source_wavelength_sha256"] == hashlib.sha256(wave.tobytes()).hexdigest(),
            "Wavelength hash mismatch")
    return n


def check_split(group, templates, split, eligible):
    """Require separate molecular catalogs and stellar templates in each split."""
    require(split.shape == group.shape and set(split) <= {0, 1, 2}, "Invalid split")
    require(set(group) <= eligible and set(templates) <= eligible,
            "Excluded or unknown catalog/template")
    groups = [set(group[split == k]) for k in range(3)]
    template_groups = [set(templates[split == k]) for k in range(3)]
    require(all(not groups[i] & groups[j] for i in range(3) for j in range(i)),
            "Catalog realization crosses split boundary")
    require(all(not template_groups[i] & template_groups[j]
                for i in range(3) for j in range(i)),
            "PHOENIX template crosses split boundary")
    return groups, template_groups


def validate(path, expected_n=None):
    reference = ROOT / "data/order7_wavelength_um.txt"
    reference_wave = load_wavelength()
    library = dict(np.load(ROOT / "cache/phoenix/library.npz"))
    manifest = json.loads((ROOT / "cache/phoenix/manifest.json").read_text())
    excluded = [i for i, model in enumerate(manifest["models"])
                if model.get("interpolated_by_library", False)]
    eligible = set(range(len(manifest["models"]))) - set(excluded)
    assessment = json.loads((ROOT / "prior_assessment.json").read_text())
    report = dict(validated_utc=datetime.now(timezone.utc).isoformat(), file=path.name)
    with h5py.File(path) as data:
        n = check_schema(data, reference_wave, expected_n)
        wave, flux, labels = (data[key][:] for key in ("wavelength", "flux", "labels"))
        wave_hash = hashlib.sha256(wave.tobytes()).hexdigest()
        require(data.attrs["generator_sha256"] == sha256(ROOT / "generate.py"),
                "Generator changed after dataset generation")
        require(np.isfinite(flux).all() and np.all(flux > 0), "Nonfinite/nonpositive flux")
        require(np.isfinite(labels).all(), "Nonfinite labels")
        hashes = {hashlib.sha256(row.tobytes()).hexdigest() for row in flux}
        require(len(hashes) == n, "Duplicate generated spectra")
        require(np.all(np.ptp(flux, axis=1) > 1e-5), "Flat generated spectrum")
        group = labels[:, 2].astype(int)
        require(np.array_equal(labels[:, 2], group), "Nonintegral series identifiers")
        split = data["split"][:]
        require(list(data["split"].attrs["names"]) == ["train", "validation", "test"],
                "Invalid split names")
        parameters = {key: value[:] for key, value in data["parameters"].items()}
        templates = parameters["template_id"]
        require(templates.shape == (n,), "Invalid PHOENIX template identifiers")
        groups, template_groups = check_split(group, templates, split, eligible)
        if n >= 10000:
            require(set(group) == eligible and set(templates) == eligible,
                    "Not all eligible templates/catalogs were sampled")
            require(all(len(set(templates[group == g])) > 1 for g in set(group)),
                    "A telluric realization is coupled to a single PHOENIX template")
        for key in ("teff", "logg", "metallicity"):
            require(np.array_equal(parameters[key], library[key][templates]), f"Wrong stellar labels: {key}")
        abundance = parameters["abundance_scale"]
        require(abundance.shape == (n, 6) and np.isfinite(abundance).all(),
                "Invalid molecular abundance shape/values")
        require(list(data["parameters/abundance_scale"].attrs["columns"]) == list(SPECIES[1:]),
                "Invalid abundance species order")
        variables = [("pwv_mm", labels[:, 0], .05, 25.),
                     ("airmass", labels[:, 1], 1., 2.5)]
        variables += [(name + "_scale", abundance[:, k], .9, 1.1)
                      for k, name in enumerate(SPECIES[1:])]
        report["parameters"] = {}
        for name, values, lower, upper in variables:
            require(np.all((values >= lower) & (values <= upper)), f"Out-of-range {name}")
            ks = kstest((values - lower) / (upper - lower), "uniform")
            if n >= 100:
                require(ks.pvalue > 1e-6, f"Grossly nonuniform sampling: {name}")
            report["parameters"][name] = dict(stats(values), uniform_ks_p=float(ks.pvalue))
        rv = parameters["stellar_velocity_kms"]
        require(rv.shape == (n,) and np.isfinite(rv).all(), "Invalid RV array")
        ks = kstest(rv / 30., "norm")
        if n >= 100:
            require(ks.pvalue > 1e-6, "RV sampling inconsistent with N(0,30 km/s)")
        report["parameters"]["stellar_velocity_kms"] = dict(stats(rv), normal_ks_p=float(ks.pvalue))
        correlation_input = np.column_stack((labels[:, :2], rv, abundance))
        report["parameters"]["sample_correlation"] = (
            np.corrcoef(correlation_input, rowvar=False).tolist()
            if n >= 3 and np.all(np.std(correlation_input, axis=0) > 0) else None)
        report["distribution_rejection_enabled"] = n >= 100
        for name, digest in data["provenance"].attrs.items():
            require(sha256(ROOT / name) == digest, f"Stored provenance mismatch: {name}")
        with np.load(ROOT / "cache/far_wings.npz") as far:
            stored = data["far_wings"]
            require(set(stored) == set(far.files), "Far-wing cache keys differ")
            for key in far.files:
                value = stored[key][()]
                require(value.shape == far[key].shape and value.dtype == far[key].dtype
                        and value.tobytes() == far[key].tobytes(),
                        f"Stored far-wing array differs from cache: {key}")
            require(int(far["seed"]) == int(data.attrs["seed"]), "Far-wing seed mismatch")
            require(np.array_equal(far["catalog_ids"], np.arange(len(manifest["models"]))),
                    "Invalid far-wing catalog IDs")
            valid = far["valid_catalog"].astype(bool)
            require(valid.shape == (len(manifest["models"]),)
                    and set(np.flatnonzero(valid)) == eligible,
                    "Far-wing validity flags differ from eligible catalogs")
            require(far["nu"].shape == (17,) and np.isfinite(far["nu"]).all()
                    and np.all(np.diff(far["nu"]) > 0)
                    and far["nu"][0] <= np.min(1e4 / wave)
                    and far["nu"][-1] >= np.max(1e4 / wave),
                    "Invalid far-wing wavenumber grid")
            require(np.array_equal(far["pwv_nodes"], np.linspace(.05, 25., 11)),
                    "Invalid far-wing PWV nodes")
            report["far_wings"] = {}
            for species in SPECIES:
                expected_shape = (len(manifest["models"]), 11 if species == "H2O" else 2, 17)
                require(far[species].shape == expected_shape, f"Wrong far-wing shape: {species}")
                values = far[species][valid]
                require(np.isfinite(values).all() and np.all(values >= 0),
                        f"Invalid far-wing opacity: {species}")
                report["far_wings"][species] = stats(values)
            require("29" in stored.attrs["factor_seed"], "Missing far-wing seed convention")
        catalogs = data["line_realizations"]
        require(set(catalogs) == {str(g) for g in set(group)}, "Catalog groups mismatch")
        report["line_factors"] = {}
        for species in SPECIES:
            source = ROOT / f"cache/hitran/{species}.npz"
            expected = assessment["species"][species]
            require(sha256(source) == expected["prepared_npz_sha256"], f"HITRAN cache changed: {species}")
            require(sha256(source.with_suffix(".par")) == expected["raw_source_sha256"],
                    f"Raw HITRAN hash mismatch: {species}")
            with np.load(source) as raw:
                line_count = len(raw["nu"])
                for field in data["lines"][species]:
                    require(np.array_equal(data["lines"][species][field][:], raw[field]),
                            f"Stored HITRAN field differs from cache: {species}/{field}")
            arrays = []
            for name in catalogs:
                factors = catalogs[name][species][:]
                require(factors.shape == (3, line_count), f"Wrong factor dimensions: {name}/{species}")
                require(np.isfinite(factors).all() and np.all(factors > 0), "Invalid line factors")
                require(list(catalogs[name][species].attrs["rows"]) == ["strength", "gamma_air", "gamma_self"],
                        "Invalid factor row labels")
                arrays.append(factors)
            arrays = np.stack(arrays)
            require(len({hashlib.sha256(a.tobytes()).hexdigest() for a in arrays}) == len(groups[0] | groups[1] | groups[2]),
                    f"Duplicate catalog draws: {species}")
            report["line_factors"][species] = {name: stats(arrays[:, k]) for k, name in
                                                 enumerate(("strength", "gamma_air", "gamma_self"))}
        report.update(seed=int(data.attrs["seed"]), shape=list(flux.shape),
                      flux=stats(flux), unique_rows=len(hashes),
                      excluded_templates=excluded, wavelength_sha256=wave_hash,
                      splits={name: dict(rows=int(np.sum(split == k)),
                                         catalogs=sorted(int(g) for g in groups[k]),
                                         templates=sorted(int(t) for t in template_groups[k]))
                              for k, name in enumerate(("train", "validation", "test"))})
    for model in manifest["models"]:
        with np.load(ROOT / "cache/phoenix" / model["file"]) as crop:
            require(hashlib.sha256(crop["flux_lambda"].astype(">f4").tobytes()).hexdigest() == model["crop_sha256"],
                    f"PHOENIX raw crop changed: {model['file']}")
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    plot = results / "dataset_examples.png"
    report["example_rows"] = make_plot(wave, flux, labels, parameters, plot)
    inputs = [reference, ROOT / "cache/phoenix/library.npz", ROOT / "cache/phoenix/manifest.json",
              ROOT / "cache/standard_atmosphere.npz", ROOT / "prior_assessment.json",
              ROOT / "priors.json", ROOT / "cache/far_wings.npz",
              ROOT / "cache/far_wings.json"]
    inputs += sorted(ROOT.glob("*.py"))
    inputs += [ROOT / f"cache/hitran/{species}.npz" for species in SPECIES]
    report["input_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in inputs}
    report["output_sha256"] = sha256(path)
    report["plot"] = str(plot.relative_to(ROOT))
    report["passed"] = True
    report["scope"] = "Structural, sampling, split, variation and input-integrity checks. Numerical forward-model validation is separate; this does not certify real-data generalization or absence of all leakage."
    target = results / "dataset_validation.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(passed=True, rows=len(flux), unique_rows=len(hashes),
                          output_sha256=report["output_sha256"], report=str(target), plot=str(plot)), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", default=ROOT / "stellar_telluric_IGRINS_ordidx7_N10000.h5")
    parser.add_argument("--expected-n", type=int, help="Require this exact sample count")
    args = parser.parse_args()
    validate(args.path, args.expected_n)

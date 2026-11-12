"""Prepare public HITRAN inputs and a target-independent standard atmosphere."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np


ROOT = Path(__file__).resolve().parent
SPECIES = ("H2O", "CO2", "CH4", "N2O", "CO", "O2", "O3")
MOLECULE_IDS = dict(H2O=1, CO2=2, CH4=6, N2O=4, CO=5, O2=7, O3=3)
FINITE_CV = {4: 0.20, 5: 0.10, 6: 0.05, 7: 0.02, 8: 0.01, 9: 0.001}
UNKNOWN_CV = 0.50
RAW_FIELDS = {
    "nu": (3, 15), "sw": (15, 25), "a": (25, 35),
    "gamma_air": (35, 40), "gamma_self": (40, 45),
    "elower": (45, 55), "n_air": (55, 59), "delta_air": (59, 67),
}
TABLE_KEYS = ("temperatures", "partition_isotopes", "partition_sums")


def sha256(path: Path) -> str:
    """Hash the exact input bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def uncertainty_cv(codes: np.ndarray) -> np.ndarray:
    """Map HITRAN classes to explicitly assumed augmentation scales."""
    return np.array([FINITE_CV.get(int(code), UNKNOWN_CV)
                     for code in codes], dtype=np.float64)


def parse_lines(raw: bytes) -> tuple[dict, np.ndarray]:
    """Parse HITRAN records without changing their original row order."""
    rows = raw.decode("ascii").splitlines()
    if not rows or any(len(row) != 160 for row in rows):
        raise ValueError("Expected nonempty HITRAN 160-character records")
    data = {key: np.array([float(row[start:stop]) for row in rows])
            for key, (start, stop) in RAW_FIELDS.items()}
    data["iso"] = np.array([
        int(row[2]) if row[2].isdigit() and row[2] != "0"
        else 10 if row[2] == "0" else ord(row[2]) - ord("A") + 11
        for row in rows
    ])
    codes = np.array([[int(char) if char.isdigit() else 0
                       for char in row[127:133]] for row in rows], dtype=np.int8)
    return data, codes


def download_hitran(source: Path) -> None:
    """Download public line files and build explicitly versioned TIPS tables."""
    import hapi

    source.mkdir(parents=True, exist_ok=True)
    for species in SPECIES:
        molecule = MOLECULE_IDS[species]
        isotopes = [record[0] for key, record in hapi.ISO.items()
                    if key[0] == molecule]
        url = ("https://hitran.org/lbl/api?iso_ids_list="
               + ",".join(map(str, isotopes)) + "&numin=4050&numax=6900")
        raw_path = source / f"{species}.par"
        metadata_path = source / f"{species}.json"
        if not raw_path.exists():
            try:
                with urlopen(url, timeout=180) as response:
                    raw = b"".join(iter(lambda: response.read(1024 * 1024), b""))
            except (HTTPError, URLError) as error:
                raise RuntimeError(
                    f"HITRAN download failed for {species}. Check network access "
                    "and HITRAN access requirements, or use --source with an "
                    "existing public HITRAN .par/.npz/.json cache."
                ) from error
            parse_lines(raw)
            temporary = raw_path.with_suffix(".part")
            temporary.write_bytes(raw)
            temporary.replace(raw_path)
            metadata = {
                "url": url,
                "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "edition": "Live HITRANonline; edition is not pinned by this endpoint",
                "edition_reference": "https://hitran.org/docs/hitran-papers/",
                "sha256": sha256(raw_path),
                "wavenumber_range_cm-1": [4050, 6900],
            }
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        else:
            metadata = json.loads(metadata_path.read_text())
            if sha256(raw_path) != metadata["sha256"]:
                raise ValueError(f"Raw HITRAN provenance hash mismatch: {species}")
        data, _ = parse_lines(raw_path.read_bytes())
        data["line_mass"] = np.array([
            hapi.molecularMass(molecule, int(isotope)) for isotope in data["iso"]
        ])
        temperatures = np.arange(150., 401.)
        partition_isotopes = np.unique(data["iso"])
        partition_sums = np.array([
            hapi.partitionSum(molecule, int(isotope), temperatures.tolist(), version=2021)
            for isotope in partition_isotopes
        ])
        np.savez_compressed(
            source / f"{species}.npz", **data, temperatures=temperatures,
            partition_isotopes=partition_isotopes, partition_sums=partition_sums,
        )
        metadata["partition_sums"] = "HAPI partitionSum(version=2021); TIPS-2021"
        metadata["hapi_version"] = getattr(hapi, "HAPI_VERSION", "not exposed")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Prepared raw {species}: {len(data['nu'])} lines", flush=True)


def prepare_hitran(source: Path, lower: float, upper: float) -> dict:
    """Check every raw line against its numeric cache before cropping."""
    destination = ROOT / "cache" / "hitran"
    destination.mkdir(parents=True, exist_ok=True)
    report = {
        "wavenumber_range_cm1": [lower, upper],
        "line_selection": "All lines in the fixed padded interval; no intensity cutoff",
        "uncertainty_reference": "https://www.hitran.org/docs/uncertainties/",
        "finite_class_upper_edges_as_cv": FINITE_CV,
        "codes_0_to_3_cv": UNKNOWN_CV,
        "distribution": "exp(sqrt(log1p(cv**2))*z - log1p(cv**2)/2), z ~ Normal(0,1)",
        "interpretation": "Positive mean-one multipliers with fractional standard deviation cv",
        "caveat": (
            "HITRAN classes are not calibrated independent Gaussian standard deviations. "
            "Finite upper edges are chosen augmentation scales. Codes 0, 1, and 2 have "
            "no numerical bound; code 3 is >=20% with no upper bound. Their 50% CV is "
            "an explicit conservative modeling assumption, not a measured uncertainty. "
            "Draws describe alternate fixed molecular catalogs, not physical temporal "
            "changes in line constants. No target spectra or fitted corrections are used."
        ),
        "species": {},
    }
    for species in SPECIES:
        raw_path = source / f"{species}.par"
        numeric_path = source / f"{species}.npz"
        metadata_path = source / f"{species}.json"
        metadata = json.loads(metadata_path.read_text())
        raw_hash = sha256(raw_path)
        if raw_hash != metadata["sha256"]:
            raise ValueError(f"Raw HITRAN provenance hash mismatch: {species}")
        if not metadata["url"].startswith("https://hitran.org/lbl/api?"):
            raise ValueError(f"Unrecognized HITRAN provenance: {species}")
        if "2021" not in metadata.get("partition_sums", ""):
            raise ValueError(f"The source cache must document TIPS-2021: {species}")
        parsed, codes = parse_lines(raw_path.read_bytes())
        selected = (parsed["nu"] >= lower) & (parsed["nu"] <= upper)
        with np.load(numeric_path, allow_pickle=False) as saved:
            for key, value in parsed.items():
                np.testing.assert_array_equal(saved[key], value, err_msg=f"{species} {key}")
            output = {key: saved[key].copy() if key in TABLE_KEYS else saved[key][selected]
                      for key in saved.files}
        for key in ("sw", "line_mass", "partition_sums"):
            if not np.all(np.isfinite(output[key]) & (output[key] > 0)):
                raise ValueError(f"Nonpositive or nonfinite {species} {key}")
        output["raw_index"] = np.flatnonzero(selected)
        output["ierr_codes"] = codes[selected]
        species_report = {
            "raw_source_sha256": raw_hash,
            "numeric_source_sha256": sha256(numeric_path),
            "source_metadata_sha256": sha256(metadata_path),
            "source_provenance": metadata,
            "source_count": len(parsed["nu"]),
            "cropped_count": int(selected.sum()),
            "validated_raw_fields": list(parsed),
            "partition_validation": "Source-cache hash and finite positive values; not in raw .par",
            "uncertainty": {},
        }
        for name, column in (("strength", 1), ("gamma_air", 2), ("gamma_self", 3)):
            chosen_codes = codes[selected, column]
            cv = uncertainty_cv(chosen_codes)
            output[f"{name}_cv"] = cv
            unique, counts = np.unique(chosen_codes, return_counts=True)
            species_report["uncertainty"][name] = {
                "code_counts": {str(int(k)): int(v) for k, v in zip(unique, counts)},
                "fallback_count": int(np.count_nonzero(chosen_codes < 4)),
                "cv_min": float(cv.min()) if len(cv) else None,
                "cv_max": float(cv.max()) if len(cv) else None,
                "strength_weighted_rms_cv": float(np.sqrt(np.average(
                    cv**2, weights=output["sw"]))) if len(cv) else None,
            }
        np.savez_compressed(destination / f"{species}.npz", **output)
        shutil.copyfile(raw_path, destination / f"{species}.par")
        shutil.copyfile(metadata_path, destination / f"{species}.json")
        species_report["prepared_npz_sha256"] = sha256(destination / f"{species}.npz")
        report["species"][species] = species_report
        print(f"{species}: verified {len(parsed['nu'])} lines; retained {selected.sum()}", flush=True)
    (ROOT / "prior_assessment.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def standard_temperature_pressure(altitude_km: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the dry US Standard Atmosphere below 51 geopotential km."""
    altitude_km = np.asarray(altitude_km, dtype=float)
    if np.any((altitude_km < 0) | (altitude_km > 51)):
        raise ValueError("Standard atmosphere implemented only from 0 to 51 km")
    boundaries = np.array([0., 11., 20., 32., 47., 51.])
    lapse = np.array([-6.5, 0., 1., 2.8, 0.])
    base_temperature = [288.15]
    base_pressure = [101325.]
    hydrostatic = 9.80665 * 0.0289644 / 8.31432 * 1000.
    for i, rate in enumerate(lapse):
        delta = boundaries[i + 1] - boundaries[i]
        temperature = base_temperature[-1] + rate * delta
        pressure = (base_pressure[-1] * np.exp(-hydrostatic * delta / temperature)
                    if rate == 0 else base_pressure[-1] *
                    (temperature / base_temperature[-1]) ** (-hydrostatic / rate))
        base_temperature.append(temperature)
        base_pressure.append(pressure)
    index = np.minimum(np.searchsorted(boundaries[1:], altitude_km, side="right"), 4)
    rate = lapse[index]
    base_t = np.array(base_temperature)[index]
    delta = altitude_km - boundaries[index]
    temperature = base_t + rate * delta
    pressure = np.empty_like(temperature)
    isothermal = rate == 0
    pressure[isothermal] = np.array(base_pressure)[index[isothermal]] * np.exp(
        -hydrostatic * delta[isothermal] / base_t[isothermal])
    changing = ~isothermal
    pressure[changing] = np.array(base_pressure)[index[changing]] * (
        temperature[changing] / base_t[changing]) ** (-hydrostatic / rate[changing])
    return temperature, pressure / 1e5


def prepare_atmosphere() -> None:
    """Save sixteen generic hydrostatic layers and a normalized water column."""
    edges = np.array([2.7, 3.2, 3.7, 4.2, 4.7, 5.5, 6.5, 8., 10.,
                      11., 14., 17., 20., 25., 32., 40., 50.])
    altitude = (edges[:-1] + edges[1:]) / 2
    temperature, pressure = standard_temperature_pressure(altitude)
    _, pressure_edges = standard_temperature_pressure(edges)
    amu = 1.66053906660e-24
    dry_column = -np.diff(pressure_edges) * 1e6 / (980.665 * 28.9644 * amu)
    water_weights = np.exp(-(edges[:-1] - edges[0]) / 2.) - np.exp(
        -(edges[1:] - edges[0]) / 2.)
    water_per_mm = 0.1 / (18.01528 * amu)
    water_column = water_per_mm * water_weights / water_weights.sum()
    np.testing.assert_allclose(water_column.sum() * 18.01528 * amu * 10., 1.)
    destination = ROOT / "cache"
    destination.mkdir(exist_ok=True)
    np.savez_compressed(
        destination / "standard_atmosphere.npz", layer_edges_km=edges,
        altitude_km=altitude, temperature_K=temperature, pressure_bar=pressure,
        pressure_edges_bar=pressure_edges, dry_air_column_cm2=dry_column,
        water_column_per_mm_cm2=water_column,
    )
    metadata = {
        "temperature_pressure": "US Standard Atmosphere 1976 piecewise hydrostatic profile",
        "reference": "https://ntrs.nasa.gov/api/citations/20050207438/downloads/20050207438.pdf",
        "surface_altitude_km": 2.7,
        "altitude_convention": "Geopotential km; approximate Gemini South elevation",
        "surface_pressure_bar": float(pressure_edges[0]),
        "layers": 16,
        "water_profile": "Exponential number density with assumed 2 km scale height, normalized to 1 mm vertical PWV",
        "water_molecules_per_cm2_per_mm": float(water_per_mm),
        "dry_air_column_note": "Dry reference hydrostatic columns; account for water mass when constructing moist columns",
        "limitations": "Fixed generic profile, no weather variability; atmosphere above 50 km omitted",
        "leakage": "No target dates, measured weather, observed flux, fitted profile, or residuals",
    }
    (destination / "standard_atmosphere.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        help="Reuse a public HITRAN .par/.npz/.json cache instead of downloading")
    parser.add_argument("--lower", type=float, default=4330.)
    parser.add_argument("--upper", type=float, default=4440.)
    args = parser.parse_args()
    if args.lower >= args.upper:
        parser.error("--lower must be smaller than --upper")
    if (args.lower, args.upper) != (4330., 4440.):
        parser.error("This order-7 workflow requires the 4330--4440 cm^-1 core")
    source = args.source
    if source is None:
        source = ROOT / "cache/raw_hitran"
        download_hitran(source)
    prepare_hitran(source, args.lower, args.upper)
    prepare_atmosphere()


if __name__ == "__main__":
    main()

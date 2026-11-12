"""Audit saved non-target-order fits without loading observed spectral arrays."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ALLOWED_ORDERS = {13, 34, 40}


def quantiles(values):
    """Return a small, explicit summary rather than fitted line corrections."""
    return dict(zip(["p50", "p68", "p90", "p95", "max"],
                    np.quantile(values, [.5, .68, .9, .95, 1.]).tolist()))


def main():
    """Check original uncertainty scales and record conservative inflation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Directory of existing non-target-order fit results")
    parser.add_argument("--output", type=Path, default=ROOT / "results/uncertainty_check.json")
    args = parser.parse_args()
    source = args.source.resolve()
    sources = ["line_strength_fit.json", "quadratic_strength_fit.json",
               "quadratic_strength_fit.npz", "line_shape_fit.json",
               "line_shape_interpretation.json"]
    loaded = {name: json.loads((source / name).read_text())
              for name in sources if name.endswith(".json")}
    nominal = loaded["line_strength_fit.json"]
    relaxed = loaded["quadratic_strength_fit.json"]
    shapes = loaded["line_shape_fit.json"]
    interpretation = loaded["line_shape_interpretation.json"]
    for records in [nominal["line_records"], relaxed["line_records"],
                    shapes["shape_records"], interpretation["line_corrections"]]:
        if not {int(row["order"]) for row in records} <= ALLOWED_ORDERS:
            raise ValueError("Calibration records contain a disallowed order")
    # NPZ is lazy: only orders, indices, contrasts and fitted parameters are read.
    # No wavelength, observed-flux, predicted-flux or residual arrays are loaded.
    with np.load(source / "quadratic_strength_fit.npz") as archive:
        if set(archive["orders"]) != ALLOWED_ORDERS:
            raise ValueError("Calibration archive has unexpected orders")
        contrast = archive["line_contrast"]
        z = archive["theta_fixed_contrast10"][archive["line_parameter_indices"]]
    records = relaxed["line_records"]
    uncertainty = np.array([r["uncertainty_upper"] for r in records])
    species = np.array([r["species"] for r in records])
    strength = {}
    for name in ["H2O", "CO2"]:
        strength[name] = {}
        for label, floor in [("all_selected", 0.), ("contrast_above_2_percent", .02)]:
            mask = (species == name) & (contrast >= floor)
            ratio = np.abs(z[mask])
            correction = z[mask] * uncertainty[mask]
            strength[name][label] = {
                "n_lines": int(mask.sum()),
                "absolute_fractional_correction": quantiles(np.abs(correction)),
                "absolute_correction_over_hitran_bin_edge": quantiles(ratio),
                "exceeds_multiplier_count": {str(k): int(np.sum(ratio > k))
                                             for k in [1, 3, 10]},
                "zero_intensity_bound_count": int(np.sum(correction <= -.999)),
            }
    width_records = interpretation["line_corrections"]
    width_factor = np.array([r["corrections"]["joint_prior3"]["width_factor"]
                             for r in width_records])
    width_edge = np.array([r["width_relative_bin_upper"] for r in width_records])
    width_ratio = np.abs(width_factor - 1.) / width_edge
    width_log_ratio = np.abs(np.log(width_factor)) / np.log1p(width_edge)
    cv_strength = relaxed["selection"]["cross_validation"]
    cv_width = shapes["selection"]["cross_validation"]
    result = {
        "calibration_orders": sorted(ALLOWED_ORDERS),
        "excluded_target_order": 7,
        "observed_spectral_arrays_loaded": False,
        "conclusion": "Nominal HITRAN uncertainty edges do not cover the conditional fitted corrections. Wider effective priors help but do not fully explain the IGRINS spectra.",
        "nominal_strength_fit": {
            "n_lines": nominal["n_parameters"],
            "at_original_uncertainty_bound": sum(r["at_cap_count"] for r in nominal["cases"]["primary"]["fit"]),
            "reported_evaluation_rms_percent_before": nominal["baseline"]["evaluation_score"]["rms_percent"],
            "reported_evaluation_rms_percent_after": nominal["cases"]["primary"]["metrics"]["evaluation_score"]["rms_percent"],
        },
        "strength_correction_summary": strength,
        "strength_training_cv": {
            name: {"sse": cv_strength[name]["sse"], "rms_percent": cv_strength[name]["rms_percent"]}
            for name in ["fixed_contrast3", "fixed_contrast10"]
        },
        "H2O_air_width_correction_summary": {
            "n_lines": len(width_factor),
            "absolute_fractional_correction": quantiles(np.abs(width_factor - 1)),
            "absolute_correction_over_hitran_bin_edge": quantiles(width_ratio),
            "absolute_log_correction_over_log1p_bin_edge": quantiles(width_log_ratio),
            "exceeds_fractional_multiplier_count": {str(k): int(np.sum(width_ratio > k)) for k in [1, 3, 10]},
            "factor_0p5_bound_count": int(np.sum(width_factor < .500001)),
        },
        "width_training_cv": {
            name: {"sse": row["sse"], "rms_percent": row["rms_percent"]}
            for name, row in cv_width.items()
        },
        "recommended_finite_code_multiplier": {
            "strength": {"H2O": 10., "CO2": 10., "default": 1.},
            "gamma_air": {"H2O": 3., "default": 1.},
            "gamma_self": {"default": 1.},
        },
        "unquantified_codes": "Codes 0-3 retain the assumed 0.50 fractional scale; multiplying an unmeasured bound by 10 is not justified.",
        "adoption_reason": "Use only aggregate multipliers already compared by training-exposure CV. Strength k=10 wins over k=3 within the fixed-IP family; H2O width k=3 improves over k=1. No observed line-specific corrections or signed mean shifts are transferred.",
        "limitations": [
            "This is calibration on other orders from the same target/night; it is not a strictly independent external validation set.",
            "No order-7 result, observed flux, residual vector, fitted atmospheric profile, stellar correction, IP or wavelength correction enters this audit.",
            "The old fits are conditional on previously fitted nuisance parameters and explored masks. Their CV is not end-to-end independent.",
            "HITRAN error classes are not calibrated Gaussian sigmas; converting inflated edges to lognormal coefficients of variation is an augmentation choice.",
            "Effective strength/width corrections can absorb column, atmospheric-profile, instrumental-profile, blend and stellar-model errors; they are not measured database errors.",
            "The old strength prior depends on original model contrast and approaches k=10 only for strong lines; applying k=10 to all finite-code lines is conservative broadening, not an exact replication.",
            "Widths were fitted only for 35 strong H2O lines in orders 13 and 40. Other species and self widths have no empirical multiplier support here.",
            "Two widths hit the factor-0.5 bound, and some weaker fitted intensities hit zero. Tail quantiles are not calibrated posterior uncertainties.",
            "Even the wider prior leaves spectral mismatch; this audit does not establish that HITRAN errors explain every IGRINS residual or certify order-7 performance.",
        ],
        "source_sha256": {name:
                          hashlib.sha256((source / name).read_bytes()).hexdigest()
                          for name in sources},
    }
    path = args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()

# exojax-synth-telluric

Generate noiseless PHOENIX stellar spectra multiplied by ExoJAX telluric
transmission for WASP-77Ab / IGRINS order 7 ML development. The default run
produces 10,000 spectra on the original 2,048-pixel wavelength grid.

This repository contains code, prior settings, and the wavelength grid only.
Generated spectra, model caches, observed flux, fitted line corrections, and
validation outputs are not distributed. ExoJAX itself is not modified.

## Run

Python 3.12 was used for the original workflow. Run these commands from the
repository root; a fresh cache needs internet access for HITRAN and PHOENIX.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python prepare_inputs.py
python prepare_stars.py
python far_wings.py --seed 20261112
python generate.py --n 10000 --seed 20261112 --dv 0.25
python validate_dataset.py
```

`prepare_inputs.py` downloads public HITRAN lines, prepares TIPS-2021 partition
sums, and constructs a generic standard atmosphere. An existing raw HITRAN
cache can instead be supplied explicitly with `--source /path/to/cache`.
PHOENIX downloads use byte ranges for the required spectral interval. The
HITRAN endpoint is live, so retain the downloaded files and recorded hashes for
exact reproducibility; a later fresh download may differ.

The default output is `stellar_telluric_IGRINS_ordidx7_N10000.h5`. Generation
also creates reusable opacity tables under `cache/`. A `.partial.h5` file is
unfinished; the final file is published locally only after all rows complete.
Existing finished output is not overwritten. Use the same seed for
`far_wings.py` and `generate.py`.

For a smaller run, change `--n` and `--output`, then pass that filename and
`--expected-n` to `validate_dataset.py`. This reduces row assembly but still
requires the fixed stellar library and far-wing preparation.

## Data and ground truth

| Dataset | Shape | Meaning |
|---|---|---|
| `wavelength` | `(2048,)`, float64 | Vacuum microns; bundled instrument grid |
| `flux` | `(N, 2048)`, float32 | Convolved stellar × telluric spectrum |
| `labels` | `(N, 3)`, float32 | `PWV`, `airmass`, `series_index` |
| `parameters/*` | Per row | Stellar labels, RV, template ID, abundance scales |
| `split` | `(N,)` | 0 = train, 1 = validation, 2 = test |
| `line_realizations/*` | Per catalog | Local line strength and width factors |
| `far_wings/*` | Per catalog | Interpolated far-wing optical-depth tables |

Parameter ground truth is saved. Separate stellar-only and telluric-only
ground-truth spectra are **not** saved. They can be reconstructed from the
recorded parameters and cached models. Instrument convolution is applied to
the high-resolution product, so separately convolved components do not
multiply exactly to the combined output.

Use the stored split rather than randomly dividing rows. Stellar templates and
molecular catalog realizations are both disjoint across splits, with an
approximately 80/10/10 allocation. Within each split, templates and catalogs
are sampled independently. `series_index` identifies a fixed molecular
catalog, and `template_id` identifies a PHOENIX model; neither is an ML input
feature. The number of distinct templates and catalogs is smaller than N.

## Model and priors

- PHOENIX: Teff 4,000–10,000 K in 200 K steps, log g = 4.5,
  [M/H] ∈ {−1.5, −1, −0.5, 0, +0.5}, [alpha/M] = 0. Nodes flagged as
  interpolated by the library are excluded. This is an operational AFGK dwarf
  range, not a stellar-population distribution. A model-only pseudo-continuum
  normalizes each template.
- PWV: uniform 0.05–25 mm; airmass: uniform 1–2.5; stellar RV:
  Gaussian mean 0, sigma 30 km/s in the telluric frame. Positive RV is a
  stellar redshift.
- CO2, CH4, N2O, CO, O2, and O3 abundances: independent factors uniform
  0.9–1.1. Water abundance is set by PWV.
- Atmosphere: 16 layers from 2.7 to 50 km, US Standard Atmosphere 1976;
  an assumed 2 km water scale height. No target-night weather is used.
- ExoJAX MODIT includes pressure, self, natural, and Doppler broadening plus
  air-pressure line shifts. The 4330–4440 cm⁻¹ core is supplemented by
  randomized Lorentz tails from other lines in the available 4050–6900 cm⁻¹
  lists. A Gaussian LSF at R = 45,000 follows the stellar × telluric product.

`priors.json` records the line-uncertainty policy. Finite HITRAN uncertainty
codes 4–9 provide fractional scales 0.20, 0.10, 0.05, 0.02, 0.01, and 0.001.
The adopted scale multipliers are 10 for H2O/CO2 line strengths and 3 for H2O
air widths; other finite-code scales use multiplier 1. Unquantified or
open-ended codes 0–3 retain an assumed 0.50 scale. These classes are not
calibrated Gaussian sigmas. Positive mean-one lognormal factors are drawn
independently per line and held fixed across layers and rows in each catalog.

The multipliers were informed by aggregate fits in orders 13, 34, and 40 of
the same target/night. Order-7 observed flux, residuals, fitted line corrections,
stellar corrections, and instrumental corrections were excluded. Thus the
augmentation prior is partly observation-informed, **not fully independent of
the target observation**. No signed or line-specific fitted corrections are
transferred. Effective corrections can absorb atmosphere, blend, stellar, or
LSF errors and do not measure the true HITRAN error distribution.

`check_uncertainties.py --source /path/to/IGRINS/results` optionally repeats
the aggregate audit when those private analysis products are available. It is
not needed for generation; those products are not included here.

## Validation and limits

`validate_dataset.py` checks the schema, parameter distributions, input hashes,
completion flags, duplicate spectra, and split separation, and writes a report
and example plot under `results/`. `validate_physics.py` compares selected
line windows with isotope-specific direct ExoJAX Voigt calculations; its
`--assembled` option compares full-order synthesis at two grid spacings.

For lightweight tests, install `pytest` and run `python -m pytest tests`.
Full generation and physical comparisons require prepared inputs and are more
expensive than these tests.

The model uses a fixed atmospheric profile, fixed stellar gravity, no additional
stellar rotation, and no noise, planet spectrum, blaze, or detector effects.
MODIT uses a dominant-isotope Doppler mass per species. PWV opacity is
interpolated, and dry abundance changes do not recompute their small self-width
contribution. Far-wing support is finite; the Lorentz tails are not a physical
water-continuum or MT_CKD model. Synthetic validation does not establish
real-data generalization or preservation of a planetary signal.

## References

- [HITRAN uncertainty codes](https://www.hitran.org/docs/uncertainties/)
- [HITRAN parameter definitions](https://hitran.org/docs/definitions-and-units/)
- [PHOENIX library](https://phoenix.astro.physik.uni-goettingen.de/spectra/) and
  [Husser et al. (2013)](https://arxiv.org/abs/1303.5632)
- [US Standard Atmosphere 1976](https://ntrs.nasa.gov/citations/20050207438)
- [ExoJAX](https://github.com/HajimeKawahara/exojax)

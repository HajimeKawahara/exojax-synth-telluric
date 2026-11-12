"""Read the fixed IGRINS order-7 wavelength metadata, without any flux data."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
WAVELENGTH_PATH = ROOT / "data/order7_wavelength_um.txt"
LABEL_COLUMNS = ["PWV", "airmass", "series_index"]
WAVELENGTH_SHA256 = "20397d41dc8c9015701c91d3aa0dced1329f5c9f07cb7ebf8060212109b0d962"


def load_wavelength():
    """Return the original 2048 vacuum-micron samples in ascending order."""
    wave = np.loadtxt(WAVELENGTH_PATH, dtype=np.float64)
    if wave.shape != (2048,) or not np.all(np.isfinite(wave)):
        raise ValueError("Invalid order-7 wavelength array")
    if not np.all(np.diff(wave) > 0):
        raise ValueError("Wavelengths must increase monotonically")
    if hashlib.sha256(wave.astype("<f8").tobytes()).hexdigest() != WAVELENGTH_SHA256:
        raise ValueError("Order-7 wavelength checksum mismatch")
    return wave

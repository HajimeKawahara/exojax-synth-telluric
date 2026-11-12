"""Download independent PHOENIX AFGK models using HTTP byte ranges."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time
from urllib.request import Request, urlopen

from astropy.io import fits
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "cache/phoenix"
BASE = "https://ftp.gwdg.de/pub/misc/phoenix/HiResFITS/"
VERSION = "PHOENIX-ACES-AGSS-COND-2011"
WAVE_FILE = f"WAVE_{VERSION}.fits"
WAVE_SHA256 = "9befb317a6d37bf8cf031e9ef1ba8a942f2f13786c953348f437aaffadf60f73"
TEFF = np.arange(4000, 10001, 200)
METALLICITY = [-1.5, -1.0, -0.5, 0.0, 0.5]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def fetch_range(url, first, last):
    """Reject silent full downloads and incomplete range responses."""
    request = Request(url, headers={"Range": f"bytes={first}-{last}"})
    for attempt in range(4):
        try:
            with urlopen(request, timeout=90) as response:
                expected = f"bytes {first}-{last}/"
                if response.status != 206 or not response.headers.get("Content-Range", "").startswith(expected):
                    raise ValueError(f"Server did not honor byte range: {url}")
                data = response.read()
                if len(data) != last - first + 1:
                    raise ValueError(f"Incomplete range response: {url}")
                return data, response.headers.get("ETag")
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


def normalize(wave, flux):
    """Estimate a pseudo-continuum from this synthetic model only."""
    logwave = np.log(wave)
    edges = np.arange(logwave[0], logwave[-1] + 500 / 299792.458, 500 / 299792.458)
    nodes, values = [], []
    for left, right in zip(edges[:-1], edges[1:]):
        selected = (logwave >= left) & (logwave < right)
        if selected.any():
            nodes.append(np.mean(logwave[selected]))
            values.append(np.percentile(flux[selected], 99.5))
    smoothed = gaussian_filter1d(np.log(values), 1, mode="nearest")
    continuum = np.exp(PchipInterpolator(nodes, smoothed)(logwave))
    return (flux / continuum).astype(np.float32)


def prepare(workers=4, wave_source=None):
    DEST.mkdir(parents=True, exist_ok=True)
    wave_path = DEST / WAVE_FILE
    if not wave_path.exists():
        if wave_source is not None:
            if sha256(wave_source.read_bytes()) != WAVE_SHA256:
                raise ValueError("The supplied PHOENIX wavelength file checksum differs.")
            shutil.copyfile(wave_source, wave_path)
        else:
            with urlopen(BASE + WAVE_FILE, timeout=120) as response:
                wave_path.write_bytes(response.read())
    if sha256(wave_path.read_bytes()) != WAVE_SHA256:
        raise ValueError("Shared PHOENIX wavelength file checksum mismatch.")
    all_wave = np.asarray(fits.getdata(wave_path), dtype=float) / 1e4
    start, stop = np.searchsorted(all_wave, [2.24, 2.32])
    wave = all_wave[start:stop]

    def download(parameters):
        teff, metallicity = parameters
        z = "-0.0" if metallicity == 0 else f"{metallicity:+.1f}"
        name = f"lte{teff:05d}-4.50{z}.{VERSION}-HiRes.fits"
        url = BASE + f"{VERSION}/Z{z}/" + name
        target = DEST / (name.removesuffix(".fits") + ".npz")
        metadata_path = target.with_suffix(".json")
        if target.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            with np.load(target) as data:
                if not np.array_equal(data["wave_um"], wave):
                    raise ValueError(f"Cached crop has a different grid: {target}")
                raw = data["flux_lambda"].astype(">f4").tobytes()
                if sha256(raw) != metadata["crop_sha256"]:
                    raise ValueError(f"Cached raw model checksum mismatch: {target}")
                flux = normalize(wave, data["flux_lambda"])
            return flux, metadata
        raw_header, etag = fetch_range(url, 0, 2879)
        header = fits.Header.fromstring(raw_header.decode("ascii"), sep="")
        if "END" not in [raw_header[i:i + 8].decode().strip() for i in range(0, 2880, 80)]:
            raise ValueError("Expected a one-block primary FITS header.")
        if header["NAXIS1"] != len(all_wave) or header["BITPIX"] != -32:
            raise ValueError(f"Unexpected PHOENIX FITS shape or dtype: {url}")
        if not np.allclose([header["PHXTEFF"], header["PHXLOGG"], header["PHXM_H"]], [teff, 4.5, metallicity]):
            raise ValueError(f"PHOENIX labels differ from requested node: {url}")
        first, last = 2880 + 4 * int(start), 2880 + 4 * int(stop) - 1
        raw_flux, flux_etag = fetch_range(url, first, last)
        if etag != flux_etag:
            raise ValueError(f"Remote file changed between requests: {url}")
        flux = np.frombuffer(raw_flux, dtype=">f4").astype(float)
        if not np.all(np.isfinite(flux)) or np.any(flux <= 0):
            raise ValueError(f"Invalid raw model flux: {url}")
        normalized = normalize(wave, flux)
        np.savez_compressed(target, wave_um=wave, flux_lambda=flux.astype(np.float32), flux_normalized=normalized)
        metadata = dict(url=url, teff_K=int(teff), logg=4.5, metallicity=metallicity,
                        alpha=0.0, etag=etag, header_sha256=sha256(raw_header),
                        crop_sha256=sha256(raw_flux), byte_range=[first, last],
                        interpolated_by_library=bool(header.get("INTERPOL", False)),
                        file=target.name)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Downloaded PHOENIX {teff} K, [M/H]={z}", flush=True)
        return normalized, metadata

    parameters = [(int(t), z) for t in TEFF for z in METALLICITY]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        models = list(pool.map(download, parameters))
    flux = np.stack([model[0] for model in models])
    np.savez_compressed(DEST / "library.npz", wave_um=wave, flux=flux,
                        teff=np.array([p[0] for p in parameters]),
                        logg=np.full(len(parameters), 4.5),
                        metallicity=np.array([p[1] for p in parameters]),
                        template_id=np.arange(len(parameters)))
    manifest = dict(source="https://phoenix.astro.physik.uni-goettingen.de/spectra/",
                    paper="https://arxiv.org/abs/1303.5632",
                    grid=VERSION, wavelength_url=BASE + WAVE_FILE,
                    wavelength_sha256=WAVE_SHA256, wavelength_units="vacuum micron",
                    normalization="99.5th percentile per 500 km/s log-wavelength bin; sigma=1 bin Gaussian smoothing of log continuum nodes; PCHIP interpolation. Synthetic model only; a pseudo-continuum, not a physical continuum.",
                    selection="Discrete uniform nodes: Teff 4000..10000 K by 200 K (operational AFGK range), logg=4.5, [M/H] in [-1.5,-1,-0.5,0,0.5], [alpha/M]=0. No interpolation between models.",
                    crop_um=[float(wave[0]), float(wave[-1])], models=[m[1] for m in models])
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {len(models)} independent PHOENIX templates: {flux.shape}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--wave-source", type=Path,
                        help="Reuse a verified raw PHOENIX wavelength FITS file")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    prepare(args.workers, args.wave_source)

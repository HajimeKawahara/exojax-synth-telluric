"""Protect the output schema and the two independent leakage boundaries."""
from __future__ import annotations

import hashlib
import json

import h5py
import numpy as np
import pytest

from validate_dataset import check_schema, check_split
from wavelength import LABEL_COLUMNS, load_wavelength


def test_original_wavelength_bytes_are_preserved():
    wave = load_wavelength()
    assert hashlib.sha256(wave.tobytes()).hexdigest() == (
        "20397d41dc8c9015701c91d3aa0dced1329f5c9f07cb7ebf8060212109b0d962"
    )


@pytest.fixture
def tiny_output(tmp_path):
    wave = load_wavelength()
    with h5py.File(tmp_path / "tiny.h5", "w") as data:
        data.attrs.update(complete=True, source_wavelength_sha256=
                          hashlib.sha256(wave.tobytes()).hexdigest())
        data.create_dataset("flux", data=np.ones((2, len(wave)), dtype="f4"))
        data.create_dataset("labels", data=np.zeros((2, 3), dtype="f4")).attrs[
            "columns"] = LABEL_COLUMNS
        data.create_dataset("wavelength", data=wave).attrs["units"] = "micron"
        data.create_dataset("completed", data=np.ones(2, dtype=bool))
        yield data, wave


def test_small_sample_schema_does_not_require_ten_thousand_rows(tiny_output):
    data, wave = tiny_output
    assert check_schema(data, wave) == 2
    with pytest.raises(ValueError, match="sample count"):
        check_schema(data, wave, expected_n=10000)


@pytest.mark.parametrize("corruption", ["incomplete", "wave", "columns"])
def test_schema_rejects_incomplete_or_misaligned_output(tiny_output, corruption):
    data, wave = tiny_output
    if corruption == "incomplete":
        data["completed"][1] = False
    elif corruption == "wave":
        data["wavelength"][0] = np.nextafter(wave[0], np.inf)
    else:
        data["labels"].attrs["columns"] = ["airmass", "PWV", "series_index"]
    with pytest.raises(ValueError):
        check_schema(data, wave)


@pytest.mark.parametrize("shared", ["catalog", "template"])
def test_catalog_and_template_leaks_are_rejected_independently(shared):
    catalogs, templates = np.array([0, 1]), np.array([2, 3])
    if shared == "catalog":
        catalogs[1] = catalogs[0]
    else:
        templates[1] = templates[0]
    with pytest.raises(ValueError, match="crosses split"):
        check_split(catalogs, templates, np.array([0, 2]), set(range(4)))


def test_sampling_keeps_templates_and_catalogs_disjoint(tmp_path, monkeypatch):
    import generate

    directory = tmp_path / "cache/phoenix"
    directory.mkdir(parents=True)
    models = [{"interpolated_by_library": i == 3} for i in range(20)]
    (directory / "manifest.json").write_text(json.dumps({"models": models}))
    monkeypatch.setattr(generate, "ROOT", tmp_path)
    stars = {"teff": np.linspace(4000, 10000, len(models))}
    parameters = generate.parameters(2000, stars, 401)
    repeated = generate.parameters(2000, stars, 401)
    for key in parameters:
        np.testing.assert_array_equal(parameters[key], repeated[key])
    check_split(parameters["series"], parameters["template"],
                parameters["split"], set(range(20)) - {3})
    assert len(np.unique(parameters["split"])) == 3
    assert np.all((parameters["pwv"] >= .05) & (parameters["pwv"] <= 25))
    assert np.all((parameters["airmass"] >= 1) & (parameters["airmass"] <= 2.5))
    assert np.all((parameters["abundance"] >= .9) & (parameters["abundance"] <= 1.1))
    # Repeated exposure conditions must not force one star per molecular catalog.
    assert any(len(set(parameters["template"][parameters["series"] == group])) > 1
               for group in set(parameters["series"]))

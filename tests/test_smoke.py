from __future__ import annotations

import csv

import h5py
import numpy as np

from hemomesh.baselines.reproduce_suk import log_wss_baseline
from hemomesh.data import SukHDF5Dataset, inspect_database


def _write_tiny_database(path) -> None:
    with h5py.File(path, "w") as handle:
        group = handle.create_group("sample_0000")
        group.create_dataset("pos", data=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]]))
        group.create_dataset("face", data=np.asarray([[0, 1, 2]]))
        group.create_dataset("wss", data=np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        group.create_dataset("pressure", data=np.asarray([[10.0], [9.5], [9.0]]))
        group.create_dataset("inlet_idcs", data=np.asarray([0]))


def test_suk_loader_and_inspection(tmp_path) -> None:
    database = tmp_path / "database.hdf5"
    _write_tiny_database(database)

    dataset = SukHDF5Dataset(database, subset="single")
    sample = dataset[0]
    summary = inspect_database(database, subset="single")

    assert len(dataset) == 1
    assert sample.sample_id == "sample_0000"
    assert sample.pos.shape == (3, 3)
    assert sample.face.shape == (1, 3)
    assert sample.wss.shape == (3, 3)
    assert sample.pressure.shape == (3, 1)
    assert summary["num_samples"] == 1
    assert len(summary["md5"]) == 32


def test_baseline_logging_smoke(tmp_path) -> None:
    truth = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    prediction = truth.copy()
    truth_path = tmp_path / "truth.npy"
    prediction_path = tmp_path / "prediction.npy"
    output_path = tmp_path / "experiments.csv"
    np.save(truth_path, truth)
    np.save(prediction_path, prediction)

    row = log_wss_baseline(
        name="gem_gcn_pretrained",
        subset="single",
        prediction_path=prediction_path,
        truth_path=truth_path,
        output_path=output_path,
    )

    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert row["model"] == "gem_gcn_pretrained"
    assert len(rows) == 1
    assert "wss_approximation_error" in rows[0]["metrics_json"]

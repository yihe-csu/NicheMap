import gzip
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


_PREPROCESS_PATH = Path(__file__).resolve().parents[1] / "nichemap" / "preprocess.py"
_SPEC = importlib.util.spec_from_file_location("nichemap_preprocess_under_test", _PREPROCESS_PATH)
_PREPROCESS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PREPROCESS)
load_xenium_data = _PREPROCESS.load_xenium_data
load_xenium_transcript_data = _PREPROCESS.load_xenium_transcript_data


def _write_transcript_table(path: Path) -> None:
    rows = [
        {
            "transcript_id": 1,
            "cell_id": "cell_a",
            "overlaps_nucleus": 1,
            "feature_name": "Gene1",
            "x_location": 1.0,
            "y_location": 2.0,
            "qv": 40.0,
            "fov_name": "R1",
        },
        {
            "transcript_id": 2,
            "cell_id": "cell_a",
            "overlaps_nucleus": 0,
            "feature_name": "Gene1",
            "x_location": 3.0,
            "y_location": 4.0,
            "qv": 35.0,
            "fov_name": "R1",
        },
        {
            "transcript_id": 3,
            "cell_id": "cell_a",
            "overlaps_nucleus": 1,
            "feature_name": "Gene2",
            "x_location": 5.0,
            "y_location": 6.0,
            "qv": 30.0,
            "fov_name": "R2",
        },
        {
            "transcript_id": 4,
            "cell_id": "cell_b",
            "overlaps_nucleus": 1,
            "feature_name": "Gene1",
            "x_location": 10.0,
            "y_location": 12.0,
            "qv": 25.0,
            "fov_name": "R2",
        },
        {
            "transcript_id": 5,
            "cell_id": "UNASSIGNED",
            "overlaps_nucleus": 0,
            "feature_name": "Gene1",
            "x_location": 100.0,
            "y_location": 100.0,
            "qv": 20.0,
            "fov_name": "R3",
        },
    ]
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        pd.DataFrame(rows).to_csv(fh, index=False)


def test_load_xenium_data_reads_geo_transcript_table_format(tmp_path, capsys):
    transcript_file = tmp_path / "GSM0000000_sample_transcripts.csv.gz"
    _write_transcript_table(transcript_file)

    adata = load_xenium_data(
        base_dir=tmp_path,
        anno_file=tmp_path / "missing_annotation.csv",
        verbose=True,
    )

    assert adata.shape == (2, 2)
    assert adata.obs_names.tolist() == ["cell_a", "cell_b"]
    assert adata.var_names.tolist() == ["Gene1", "Gene2"]
    assert adata[["cell_a"], ["Gene1"]].X[0, 0] == 2
    assert adata[["cell_a"], ["Gene2"]].X[0, 0] == 1
    assert adata[["cell_b"], ["Gene1"]].X[0, 0] == 1
    np.testing.assert_allclose(adata.obsm["spatial"], [[3.0, 4.0], [10.0, 12.0]])
    assert adata.obs.loc["cell_a", "transcript_count"] == 3
    assert adata.obs.loc["cell_a", "n_nucleus_transcripts"] == 2
    assert adata.uns["source_files"]["transcript_file"] == str(transcript_file)
    assert "Annotation file not found" in capsys.readouterr().out


def test_load_xenium_transcript_data_uses_tqdm_instead_of_chunk_prints(
    tmp_path, capsys, monkeypatch
):
    transcript_file = tmp_path / "GSM0000000_sample_transcripts.csv.gz"
    _write_transcript_table(transcript_file)
    tqdm_calls = []

    def fake_tqdm(iterable, **kwargs):
        tqdm_calls.append(kwargs)
        return iterable

    monkeypatch.setattr(_PREPROCESS, "tqdm", fake_tqdm, raising=False)

    adata = load_xenium_transcript_data(
        base_dir=tmp_path,
        transcript_file=transcript_file,
        anno_file=None,
        chunk_size=2,
        verbose=True,
    )

    stdout = capsys.readouterr().out
    assert adata.shape == (2, 2)
    assert len(tqdm_calls) == 1
    assert tqdm_calls[0]["desc"] == "Reading transcripts"
    assert "chunk 1:" not in stdout
    assert "chunk 2:" not in stdout

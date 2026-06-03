"""Tests for dataset extractors registered in scripts/process_public_data.py."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

# scripts/ is not a package; we import via path manipulation.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from process_public_data import DATASET_EXTRACTORS  # noqa: E402


@pytest.fixture
def secom_mini(tmp_path):
    """Synthetic mini SECOM archive: 5 samples × 8 taxa OTU CSV in a zip."""
    archive = tmp_path / "secom_mini.zip"
    otu_csv_text = (
        "taxon,sample1,sample2,sample3,sample4,sample5\n"
        "Bacteroides,10,20,30,40,50\n"
        "Faecalibacterium,5,5,15,25,35\n"
        "Lactobacillus,1,2,3,4,5\n"
        "Bifidobacterium,8,8,8,8,8\n"
        "Akkermansia,0,1,2,3,4\n"
        "Escherichia,3,6,9,12,15\n"
        "Prevotella,2,4,6,8,10\n"
        "Roseburia,7,7,7,7,7\n"
    )
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("secom_otu.csv", otu_csv_text)
    return archive


def test_secom_extractor_writes_counts_taxa_samples(secom_mini, tmp_path):
    out_dir = tmp_path / "secom_out"
    extract = DATASET_EXTRACTORS["secom_v1.0.0"]
    extract(secom_mini, out_dir)
    counts_path = out_dir / "counts.npz"
    taxa_path = out_dir / "taxa.csv"
    samples_path = out_dir / "samples.csv"
    assert counts_path.exists()
    assert taxa_path.exists()
    assert samples_path.exists()
    counts = np.load(counts_path)["counts"]
    assert counts.shape == (5, 8), f"expected (samples=5, taxa=8); got {counts.shape}"
    assert counts.dtype.kind in ("i", "u")
    # Row-1 (sample1) totals: 10+5+1+8+0+3+2+7 = 36
    assert int(counts[0].sum()) == 36
    # Column-0 (Bacteroides) totals: 10+20+30+40+50 = 150
    assert int(counts[:, 0].sum()) == 150
    # taxa.csv first non-header line is Bacteroides
    taxa_lines = taxa_path.read_text().splitlines()
    assert taxa_lines[0] == "taxon"
    assert taxa_lines[1] == "Bacteroides"
    # samples.csv first non-header is sample1
    sample_lines = samples_path.read_text().splitlines()
    assert sample_lines[0] == "sample_id"
    assert sample_lines[1] == "sample1"


@pytest.fixture
def hmp_mini(tmp_path):
    """Tiny BIOM-format v1.0 JSON archive (4 samples × 6 observations).

    BIOM v1.0 is JSON; the biom-format library accepts both v1 (JSON)
    and v2 (HDF5). JSON is the simplest fixture.
    """
    biom = tmp_path / "hmp_mini.biom"
    payload = {
        "id": "mini",
        "format": "Biological Observation Matrix 1.0.0",
        "format_url": "http://biom-format.org",
        "type": "OTU table",
        "generated_by": "test fixture",
        "date": "2026-06-03T00:00:00",
        "matrix_type": "dense",
        "matrix_element_type": "int",
        "shape": [6, 4],
        "rows": [{"id": f"OTU{i}", "metadata": None} for i in range(6)],
        "columns": [{"id": f"S{j}", "metadata": None} for j in range(4)],
        "data": [
            [10, 20, 30, 40],
            [5, 5, 15, 25],
            [1, 2, 3, 4],
            [8, 8, 8, 8],
            [0, 1, 2, 3],
            [3, 6, 9, 12],
        ],
    }
    biom.write_text(json.dumps(payload))
    return biom


def test_hmp_extractor_writes_counts(hmp_mini, tmp_path):
    out_dir = tmp_path / "hmp_out"
    extract = DATASET_EXTRACTORS["hmp_16s"]
    extract(hmp_mini, out_dir)
    counts = np.load(out_dir / "counts.npz")["counts"]
    # BIOM stores observations × samples = (6, 4); extractor transposes
    # to samples × taxa = (4, 6).
    assert counts.shape == (4, 6), f"expected (4, 6), got {counts.shape}"
    # Row-0 (sample S0) totals: 10+5+1+8+0+3 = 27
    assert int(counts[0].sum()) == 27
    # Column-0 (OTU0) totals: 10+20+30+40 = 100
    assert int(counts[:, 0].sum()) == 100
    # taxa / samples files
    taxa_lines = (out_dir / "taxa.csv").read_text().splitlines()
    sample_lines = (out_dir / "samples.csv").read_text().splitlines()
    assert taxa_lines[0] == "taxon"
    assert taxa_lines[1] == "OTU0"
    assert sample_lines[0] == "sample_id"
    assert sample_lines[1] == "S0"

"""Tests for the public-data manifest skeleton."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_manifest_tsv_present_and_has_required_columns():
    manifest = REPO / "data" / "manifest.tsv"
    assert manifest.exists()
    with open(manifest, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        assert set(reader.fieldnames) >= {
            "path",
            "role",
            "provenance",
            "generator",
            "licence",
        }
        rows = list(reader)
    assert len(rows) >= 4
    # Every row points at a path that exists.
    for row in rows:
        target = REPO / row["path"]
        assert target.exists(), f"manifest row {row['path']} missing"


def test_public_data_pointers_record_stable_identifiers():
    secom = (REPO / "data" / "public" / "secom_v1.0.0.md").read_text()
    assert "10.5281/zenodo.6809029" in secom
    assert "10.5281/zenodo.6809028" in secom
    hmp = (REPO / "data" / "public" / "hmp_16s.md").read_text()
    assert "PRJNA43021" in hmp
    assert "PRJNA48489" in hmp


def test_processing_script_advertises_unwired_dataset_with_clear_message(tmp_path: Path):
    archive = tmp_path / "fake.zip"
    archive.write_bytes(b"placeholder")
    out = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "process_public_data.py"),
            "--dataset",
            "secom_v1.0.0",
            "--input",
            str(archive),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 3
    assert "no extractor" in proc.stderr.lower()


def test_processing_script_validates_input_path(tmp_path: Path):
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "process_public_data.py"),
            "--dataset",
            "secom_v1.0.0",
            "--input",
            str(tmp_path / "missing.zip"),
            "--output",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "does not exist" in proc.stderr.lower()

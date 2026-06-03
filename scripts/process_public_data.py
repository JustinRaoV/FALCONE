"""Skeleton public-data processing entrypoint.

This script is the *single* place where downloaded public archives are
turned into processed source-data tables. It is invoked from
``data/public/*.md`` and runs in a fresh checkout once the operator has
already downloaded the archive.

The current implementation is a documented skeleton that:

1. Validates the inputs exist.
2. Verifies any committed SHA-256 record.
3. Records intent and exits — actual extraction code lands when the
   first dataset is wired into a benchmark run.

Adding extraction code for a dataset must include:

* a per-dataset extractor function under ``DATASET_EXTRACTORS``;
* an entry in ``data/manifest.tsv`` for every output file written;
* a ``SOURCE.txt`` next to outputs preserving upstream attribution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np


def _extract_secom_v1_0_0(archive: Path, out_dir: Path) -> None:
    """Extract SECOM v1.0.0 archive (Zenodo 10.5281/zenodo.6809029).

    The archive's OTU CSV is in taxon-first layout (rows = taxa, columns =
    samples). We transpose to samples × taxa for downstream consumers.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        otu_members = [m for m in zf.namelist() if m.endswith("secom_otu.csv")]
        if not otu_members:
            raise FileNotFoundError(
                "secom_otu.csv not found in archive; "
                f"members: {zf.namelist()}"
            )
        otu_member = otu_members[0]
        with zf.open(otu_member) as fh:
            reader = csv.reader(line.decode("utf-8") for line in fh)
            header = next(reader)
            sample_names = header[1:]
            taxa = []
            rows = []
            for row in reader:
                taxa.append(row[0])
                rows.append([int(v) for v in row[1:]])
    counts = np.asarray(rows, dtype=np.int64).T  # samples × taxa
    np.savez_compressed(out_dir / "counts.npz", counts=counts)
    (out_dir / "taxa.csv").write_text("taxon\n" + "\n".join(taxa) + "\n")
    (out_dir / "samples.csv").write_text("sample_id\n" + "\n".join(sample_names) + "\n")


DATASET_EXTRACTORS: dict[str, Callable] = {
    "secom_v1.0.0": _extract_secom_v1_0_0,
}


def _verify_checksum(archive: Path, checksum: Path) -> None:
    if not checksum.exists():
        return
    expected = checksum.read_text().split()[0]
    h = hashlib.sha256()
    with open(archive, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != expected:
        raise SystemExit(
            f"checksum mismatch for {archive}: got {h.hexdigest()}, expected {expected}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        required=True,
        help="Dataset key. Must match a folder under data/public/.",
    )
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    archive = Path(args.input)
    if not archive.exists():
        print(f"error: input archive {archive} does not exist", file=sys.stderr)
        return 2
    checksum_path = archive.with_suffix(archive.suffix + ".sha256")
    _verify_checksum(archive, checksum_path)

    extractor = DATASET_EXTRACTORS.get(args.dataset)
    if extractor is None:
        print(
            f"dataset {args.dataset!r} has no extractor yet. "
            f"Add one under DATASET_EXTRACTORS in {__file__} and update data/manifest.tsv.",
            file=sys.stderr,
        )
        return 3

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    extractor(archive=archive, out_dir=out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

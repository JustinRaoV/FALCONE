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
import hashlib
import os
import sys
from pathlib import Path
from typing import Callable

DATASET_EXTRACTORS: dict[str, Callable] = {
    # populated when a dataset is wired in
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

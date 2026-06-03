# `data/` — Generated Source Data and Public-Data Pointers

This directory holds **only** generated source-data tables, manifest
metadata, and pointers to public-data archives. Raw third-party data is
never committed.

## Contents

```
data/
├── README.md             this file — maps tables to commands and figures
├── manifest.tsv          per-file role / provenance / generator / licence
└── public/               download instructions for public datasets
    ├── secom_v1.0.0.md   Lin, Eggesbo & Peddada (2022) archive
    └── hmp_16s.md        NIH HMP umbrella + 16S rRNA subproject pointers
```

When a benchmark runner produces source-data CSVs, they will land here
with deterministic names and matching `manifest.tsv` rows. Until the
acceptance gates in
[`docs/superpowers/specs/2026-06-02-single-domain-estimator-rebuild-design.md`](../docs/superpowers/specs/2026-06-02-single-domain-estimator-rebuild-design.md)
are evaluated, no such CSVs are produced. This directory therefore stays
empty of generated tables on `main`.

## How a generated table is added

1. Run the benchmark or processing command. The command MUST accept an
   explicit `--output` path under `data/`.
2. Update `manifest.tsv` with one row per generated file. The row
   records `path`, `role`, `provenance`, `generator`, and `licence`.
3. Reference the manifest entry from the figure script that consumes it
   so every numerical claim maps back to a generator.

`manifest.tsv` is the contract. Anything in `data/` that is not in the
manifest is considered orphaned and may be deleted by housekeeping.

## Reproducibility

The generator commands recorded in `manifest.tsv` are designed to be
runnable in a fresh checkout, given the same Python and R toolchain
versions and the same public-data downloads. See `data/public/*.md` for
the public-data download steps.

## Public-data policy

The repository commits:

1. download instructions and stable identifiers,
2. checksum records (SHA-256) for downloaded archives,
3. a processing script,
4. processed source-data tables only when the original licence allows
   redistribution, and
5. subsampling-stability summaries mapped to the public identifier.

The repository does **not** commit raw FASTQ, BIOM tables, OTU tables,
or any other primary data file we do not have explicit redistribution
rights for. If a downstream user wants the raw data, the README in
`data/public/` tells them where to fetch it.

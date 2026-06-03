# SECOM Public Data Archive (v1.0.0)

This file points to the public archive associated with:

> Lin H, Eggesbo M, Peddada SD. *Linear and nonlinear correlation
> estimators unveil undescribed taxa interactions in microbiome data.*
> Nature Communications (2022).
> https://doi.org/10.1038/s41467-022-32243-x

The archive includes the forehead/palm skin microbiome illustration and
the Norwegian Microbiome study illustration used in that paper.

## Stable identifiers

Concept DOI (always points at the latest version):

```
10.5281/zenodo.6809028
```

Version DOI used for reproducibility (v1.0.0):

```
https://doi.org/10.5281/zenodo.6809029
```

## Download

```bash
mkdir -p data/public/secom_v1.0.0
cd data/public/secom_v1.0.0
curl -L -o archive.zip https://zenodo.org/record/6809029/files/SECOM-1.0.0.zip
sha256sum archive.zip > archive.sha256
```

The exact archive file name on Zenodo may evolve with re-uploads; resolve
the version-DOI URL above to find the canonical asset list.

## Checksum policy

`archive.sha256` is committed only after the archive is downloaded for
the first time and verified against the upstream value, if Zenodo
publishes one. Until then this directory contains only this README.

## Processing

```bash
uv run python scripts/process_public_data.py \
    --dataset secom_v1.0.0 \
    --input data/public/secom_v1.0.0/archive.zip \
    --output data/public/secom_v1.0.0/processed
```

The processing script:

1. Verifies `archive.sha256`.
2. Extracts the count tables for the forehead/palm and Norwegian
   subsets.
3. Filters by prevalence and total counts using the same defaults as
   `falcon.preprocessing.prepare_log_composition`.
4. Writes processed source-data tables that the benchmark / figure
   pipeline consumes.

Processed tables MAY be redistributed under the upstream Zenodo licence
(check `archive.zip/LICENCE` after download). The processing script
emits a `LICENCE.txt` alongside any redistributed table so the licence
travels with the data.

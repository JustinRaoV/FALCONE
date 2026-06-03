# NIH Human Microbiome Project — 16S rRNA Diversity

This file points to the public NIH HMP archive used for single-domain
stability evaluation.

## Stable identifiers

Umbrella BioProject:

```
PRJNA43021
```

16S rRNA diversity subproject:

```
PRJNA48489
```

Documentation portal:

```
https://hmpdacc.org/
```

## Download

The NIH HMP releases are mirrored at the HMP Data Analysis and
Coordination Center (HMPDACC). The 16S OTU and metadata tables are
public-domain and large; download them directly from the portal:

```bash
mkdir -p data/public/hmp_16s
cd data/public/hmp_16s
# Resolve the 16S V1-V3 OTU table from PRJNA48489 via the HMPDACC
# portal; record the resolved URL alongside the SHA-256 below.
echo "RESOLVED_URL=$(...)" > url.txt
curl -L -o otu_table.biom "$RESOLVED_URL"
shasum -a 256 otu_table.biom > otu_table.sha256
```

This file deliberately does not pin a specific resolved URL — the
HMPDACC portal occasionally re-mints download links, and committing a
stale URL would mislead reproducers. Operators record the URL they
actually used inside `url.txt` once the download succeeds.

## Processing

```bash
uv run python scripts/process_public_data.py \
    --dataset hmp_16s \
    --input data/public/hmp_16s/otu_table.biom \
    --output data/public/hmp_16s/processed
```

Processing emits:

* `processed/counts.npz` — filtered count matrix
* `processed/metadata.csv` — sample metadata after harmonization
* `processed/manifest_excerpt.tsv` — rows added to top-level
  `data/manifest.tsv` describing the processed tables

The HMP terms of use permit redistribution of derived processed tables;
the processing script writes the upstream attribution string into
`processed/SOURCE.txt`.

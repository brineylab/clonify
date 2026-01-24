# Clonify

Fast and accurate antibody clonal lineage assignment.

## Overview

`clonify` groups antibody sequences into clonal lineages based on:
- CDR3/junction amino acid sequence similarity
- V and J gene usage
- Shared somatic mutations

Supports both unpaired (heavy chain only) and paired (heavy + light chain) sequences.

## Installation

```bash
pip install clonify
```

## Usage

### Basic Usage (Unpaired Sequences)

```python
import polars as pl
from clonify import clonify

# Load your antibody sequence data
df = pl.read_csv("sequences.tsv", separator="\t")

# Cluster sequences
assignments, result_df = clonify(df)

# Save results
result_df.write_csv("clustered.tsv", separator="\t")
```

### Paired Heavy/Light Chain Sequences

For paired sequence data with heavy and light chain information:

```python
from clonify import clonify

# Paired data uses column suffixes to distinguish chains
# Default: ":0" for heavy chain, ":1" for light chain
# Example columns: v_gene:0, j_gene:0, junction_aa:0 (heavy)
#                  v_gene:1, j_gene:1, junction_aa:1 (light)

assignments, result_df = clonify(
    df,
    paired=True,  # Enable paired mode (required)
)
```

Paired mode uses light chain V/J genes for scoring (8-point penalty per mismatch) while partitioning by heavy chain V/J only. This ensures sequences with different light chain gene usage are less likely to cluster together.

#### Custom Column Suffixes

If your data uses different suffixes:

```python
assignments, result_df = clonify(
    df,
    paired=True,
    heavy_suffix="_heavy",  # Custom suffix for heavy chain columns
    light_suffix="_light",  # Custom suffix for light chain columns
)
```

#### Explicit Column Names

You can also specify column names explicitly:

```python
assignments, result_df = clonify(
    df,
    paired=True,
    heavy_vgene_key="v_gene_heavy",
    heavy_jgene_key="j_gene_heavy",
    heavy_cdr3_key="junction_aa_heavy",
    light_vgene_key="v_gene_light",
    light_jgene_key="j_gene_light",
)
```

### Partition Levels

Control how sequences are partitioned for parallel processing:

```python
# Fine-grained partitioning (default) - fastest for large datasets
assignments, result_df = clonify(df, partition_level="vj_gene")

# Partition by V gene only
assignments, result_df = clonify(df, partition_level="v_gene")

# Partition by V gene family (coarsest, 8 partitions)
assignments, result_df = clonify(df, partition_level="v_family")
```

### Command Line

Basic usage:

```bash
clonify run -i sequences.tsv -o clustered.tsv
```

With paired heavy/light chain data:

```bash
# Using default column suffixes (:0 for heavy, :1 for light)
clonify run -i paired_sequences.tsv -o clustered.tsv --paired

# With custom column suffixes
clonify run -i paired_sequences.tsv -o clustered.tsv --paired \
    --heavy-suffix "_heavy" --light-suffix "_light"

# With explicit column names
clonify run -i paired_sequences.tsv -o clustered.tsv --paired \
    --heavy-vgene-key "v_gene_heavy" \
    --heavy-jgene-key "j_gene_heavy" \
    --heavy-cdr3-key "junction_aa_heavy" \
    --light-vgene-key "v_gene_light" \
    --light-jgene-key "j_gene_light"
```

With custom partition level:

```bash
# Use V gene family partitioning (coarser, 8 partitions)
clonify run -i sequences.tsv -o clustered.tsv -p v_family

# Use V+J gene partitioning (finer, default)
clonify run -i sequences.tsv -o clustered.tsv -p vj_gene
```

View all options:

```bash
clonify run --help
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `distance_cutoff` | 0.35 | Maximum dissimilarity for clustering |
| `shared_mutation_bonus` | 0.35 | Bonus for shared mutations |
| `length_penalty_multiplier` | 2.0 | Penalty per unit length difference in CDR3 |
| `partition_level` | "vj_gene" | Partitioning strategy: "v_family", "v_gene", or "vj_gene" |
| `paired` | False | Enable paired heavy/light chain mode |
| `heavy_suffix` | ":0" | Column suffix for heavy chain (paired mode) |
| `light_suffix` | ":1" | Column suffix for light chain (paired mode) |

### Column Auto-Detection

Clonify automatically detects common column names:

| Field | Recognized Names |
|-------|-----------------|
| Sequence ID | `sequence_id`, `seq_id`, `id`, `name` |
| V gene | `v_gene`, `v_call`, `vgene`, `v` |
| J gene | `j_gene`, `j_call`, `jgene`, `j` |
| CDR3/Junction | `junction_aa`, `cdr3_aa`, `cdr3`, `junction`, `junc_aa` |
| Mutations | `v_mutations`, `mutations`, `muts`, `shm` |

In paired mode, these names are searched with the configured suffixes (e.g., `v_gene:0` for heavy chain).

## Algorithm

Clonify uses a multi-stage clustering approach:

1. **Partitioning**: Sequences are partitioned by V/J gene usage (configurable granularity)
2. **Essence Grouping**: Identical (V, J, CDR3) tuples are grouped; for paired sequences, light chain V/J genes also contribute to identity
3. **Megaclustering**: High-weight essences become cluster centers
4. **Hierarchical Clustering**: NN-chain UPGMA within megaclusters
5. **Flat Clustering**: Distance threshold cuts the dendrogram

### Dissimilarity Formula

```
dissimilarity = (edit_distance + v_penalty + j_penalty + light_penalties - mutation_bonus + length_penalty) / min_length
```

- `v_penalty`, `j_penalty`: 8 if genes differ, 0 otherwise
- `light_penalties` (paired mode only): 8 per light chain V/J gene mismatch
- `mutation_bonus`: `shared_mutation_bonus × shared_mutations`
- `length_penalty`: `length_penalty_multiplier × |len1 - len2|`

## License

MIT License

# Clonify

High-performance antibody clonotype clustering.

## Overview

Clonify clusters antibody sequences into clonal lineages based on:
- CDR3/junction amino acid sequence similarity
- V and J gene usage
- Shared somatic mutations

## Installation

```bash
pip install clonify
```

## Usage

### Python API

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

### Command Line

```bash
clonify run -i sequences.tsv -o clustered.tsv
```

### Parameters

- `distance_cutoff` (default: 0.35): Maximum dissimilarity for clustering
- `shared_mutation_bonus` (default: 0.35): Bonus for shared mutations
- `length_penalty_multiplier` (default: 2.0): Penalty per unit length difference

## Algorithm

Clonify uses a multi-stage clustering approach:

1. **Partitioning**: Sequences are partitioned by V gene family
2. **Essence Grouping**: Identical (V, J, CDR3) tuples are grouped
3. **Megaclustering**: High-weight essences become cluster centers
4. **Hierarchical Clustering**: NN-chain UPGMA within megaclusters
5. **Flat Clustering**: Distance threshold cuts the dendrogram

## License

MIT License

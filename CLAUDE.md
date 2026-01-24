# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Clonify is a high-performance antibody clonotype clustering library. It clusters antibody sequences into clonal lineages based on CDR3/junction amino acid sequence similarity, V/J gene usage, and shared somatic mutations. The core algorithm is implemented in Rust with Python bindings via PyO3/maturin.

## Build Commands

```bash
# Install development dependencies
pip install -e ".[dev]"

# Build the Rust extension (development mode)
maturin develop

# Build release version
maturin build --release

# Run all Python tests
pytest

# Run a single test
pytest tests/test_api.py::TestClonify::test_clonify_small_df -v

# Run Rust tests
cargo test --workspace

# Lint Python code
ruff check python tests

# Type check Python
mypy python
```

## Architecture

### Two-Layer Design

The project has a Rust core with Python bindings:

1. **Rust Core** (`crates/clonify-core/`): Pure Rust clustering library
2. **Python Bindings** (`crates/clonify-python/`): PyO3 bindings exposing `_native` module
3. **Python API** (`python/clonify/`): High-level Python interface wrapping the native module

### Clustering Algorithm (Multi-Stage)

The algorithm in `crates/clonify-core/` follows five stages:

1. **Partitioning** (`partition.rs`): Sequences are partitioned by V gene family (1-7 + overflow bucket). Each partition is processed independently.

2. **Essence Grouping** (`essence.rs`): Identical (V, J, CDR3) tuples are grouped into "Essences". Each Essence tracks all mutation variants observed and computes a canonical mutation list via majority voting.

3. **Megaclustering** (`megacluster.rs`, `partition.rs`): High-weight Essences become cluster centers. Essences are assigned to nearest center based on dissimilarity. Centers within MIN_MEGACLUSTER_DISSIMILARITY (0.40) are merged.

4. **Hierarchical Clustering** (`linkage.rs`): NN-chain UPGMA within each megacluster builds a dendrogram.

5. **Flat Clustering** (`flat_cluster.rs`): Distance threshold (default 0.35) cuts the dendrogram to produce final clusters.

### Key Data Types

- **Mutation** (`types.rs`): 16-bit encoded value: `(position << 4) | mutation_code`
- **MutList**: Sorted list of mutations with occurrence weight
- **MutBag**: Accumulator for computing canonical mutations via majority voting
- **EssenceKey**: Tuple of (junction_aa, v_gene_id, j_gene_id)
- **Essence**: Groups sequences with identical EssenceKey, tracks mutation variants

### Dissimilarity Formula

```
dissimilarity = (edit_distance + v_penalty + j_penalty - mutation_bonus + length_penalty) / min_length
```

- `v_penalty`/`j_penalty`: 8 if genes differ, 0 otherwise
- `mutation_bonus`: `mut_value * shared_mutations` (default mut_value=0.35)
- `length_penalty`: `len_penalty * |len1 - len2|` (default len_penalty=2)

### Python API

The main entry point is `clonify()` in `python/clonify/api.py`:
- Accepts polars DataFrame, pandas DataFrame, or file path
- Auto-detects column names via aliases (e.g., "v_gene", "v_call", "vgene")
- Returns `(assignment_dict, result_dataframe)` tuple

Column aliases are defined in `COLUMN_ALIASES` dict for automatic detection.

### Native Module Exports

The `clonify._native` module exposes:
- `ClusterParams`: Configuration class
- `cluster()`: Core clustering function
- `parse_mutations()`: Parse "pos:ref>alt|..." format
- `encode_mutation()`: Encode single mutation
- `hamming_distance()`, `levenshtein_distance()`: Distance functions

## Testing

Test fixtures are in `tests/conftest.py`. Sample data is expected at `data/test_HIV_bnAb_heavies.tsv` (62 sequences).

Key test classes:
- `TestClusterParams`: Parameter validation
- `TestDistanceFunctions`: Distance metric correctness
- `TestMutationParsing`: Mutation string parsing
- `TestClonify`: End-to-end clustering
- `TestDeterminism`: Reproducibility verification

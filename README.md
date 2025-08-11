# clonify

High-performance antibody clonal lineage assignment.

## Installation

clonify is published on PyPI (as `clonify-antibody`) with prebuilt wheels for common platforms:

- macOS (Apple Silicon arm64 and Intel x86_64)
- Linux (manylinux x86_64)

Requires Python 3.8+.

### pip

```bash
python -m pip install -U pip setuptools wheel
python -m pip install clonify-antibody
```

If you want to avoid building from source, you can force a prebuilt wheel only:

```bash
python -m pip install --only-binary=:all: clonify-antibody
```

### uv

```bash
# install into the current environment
uv pip install clonify-antibody

# or create/use a uv-managed virtualenv
uv venv .venv
source .venv/bin/activate
uv pip install clonify-antibody
```

## Compatibility

| OS | Architecture | Python versions (CPython) | Distribution |
| --- | --- | --- | --- |
| macOS | arm64 (Apple Silicon) | 3.8 – 3.12 | Prebuilt wheel |
| macOS | x86_64 (Intel) | 3.8 – 3.12 | Prebuilt wheel |
| Linux (manylinux) | x86_64 | 3.8 – 3.12 | Prebuilt wheel |
| Other/older combos | varies | 3.8+ | Source build via Rust |

Notes:

- Wheels target mainstream CPython versions; newer versions may be added as releases are cut.
- If a wheel isn’t available for your exact combo, the installer will attempt a source build. See Troubleshooting.

### Verify your installation

- CLI: `clonify --help`
- Python:
  
  ```python
  import clonify
  from clonify import clonify as run_clonify
  ```

## Quick start

### CLI

[AIRR-compliant TSV](https://docs.airr-community.org/en/latest/datarep/rearrangements.html) is expected to be input format, using either the unpaired sequence schema or the modified paired sequence schema used by [abstar](https://github.com/brineylab/abstar). Clonify can also accept CSV/TSV/Parquet/NDJSON/JSON files; format is inferred from the extension. The output format is inferred from the extension of the supplied `--output` path; if omitted, CSV is printed to stdout.

```bash
# Minimal: read TSV and write CSV
clonify \
  --input data/airr.tsv \
  --output results/lineages.csv

# With optional JSON mapping of sequence_id -> lineage
clonify \
  --input data/airr.tsv \
  --output results/lineages.parquet \
  --assignments-json results/assignments.json \
  --distance-cutoff 0.35 \
  --shared-mutation-bonus 0.4 \
  --length-penalty-multiplier 2.0
```

### Python API

Using a Polars DataFrame (recommended for native backend):

```python
import polars as pl
from clonify import clonify

# df must include v_gene, j_gene, cdr3, and v_mutations columns (paired inputs are supported)
df = pl.read_csv("data/airr.tsv", separator="\t")

assignments, df_out = clonify(
    df,
    distance_cutoff=0.35,
    shared_mutation_bonus=0.4,
    length_penalty_multiplier=2.0,
)

# Write results
df_out.write_parquet("results/lineages.parquet")
```

You can also pass a file path directly; the format is inferred by extension. If you provide `output_path`, results are written automatically:

```python
from clonify import clonify

assignments, df_out = clonify(
    "data/airr.tsv",
    output_path="results/lineages.csv",  # csv/tsv/parquet/json/ndjson
)
```

## Troubleshooting installation

clonify ships wheels for the platforms listed above. If no compatible wheel is available for your OS/architecture/Python version, the installer will attempt to build from source using Rust (via maturin/pyo3).

### What failure looks like when no wheel is available and Rust is not installed

During `pip install clonify` (or `uv pip install clonify`), you may see messages such as:

- "Building wheel for clonify (pyproject.toml) did not run successfully"
- "error: cargo not found" or "error: can't find Rust compiler"
- "note: maturin failed; ensure Rust toolchain is installed"

This indicates a source build was triggered but a Rust toolchain is missing.

### How to fix

- Install Rust (recommended):
  - `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
  - Restart your shell, then verify: `rustc --version` and `cargo --version`.
- Upgrade build tooling: `python -m pip install -U pip setuptools wheel`.
- macOS build tools (if building from source): `xcode-select --install`.
- Ensure your Python architecture matches your OS:
  - Apple Silicon: Prefer an arm64 Python. Check with `python -c "import platform; print(platform.machine())"` (expect `arm64`). If running under Rosetta (`x86_64`), use an arm64 Python or install the x86_64 wheel in a matching environment.
- Prefer wheels when available: `python -m pip install --only-binary=:all: clonify`.
- If behind a proxy/offline: download the appropriate wheel from PyPI on a connected machine and `pip install path/to/clonify-<version>-<tags>.whl`.

If issues persist, reinstall Rust via rustup and retry with verbose logs:

```bash
python -m pip install -v clonify
```

## Usage

```python
from clonify import clonify
assign_dict, df_out = clonify(df, distance_cutoff=0.35)
```

## Development

- Create a virtual environment (recommended)
- Build editable native extension: `pip install maturin && maturin develop`
- Run tests: `pytest`

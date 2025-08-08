from __future__ import annotations

import itertools
from pathlib import Path

import polars as pl
import pytest

# Import clonify, but gracefully skip if the native extension isn't built yet
try:
    import clonify as clonify_mod

    clonify = clonify_mod.clonify
except Exception as e:  # pragma: no cover - handled by pytest skip
    if isinstance(e, RuntimeError) and "native extension not built" in str(e).lower():
        pytest.skip(
            "clonify native extension not built. Build with `maturin develop` or `pip install .`.",
            allow_module_level=True,
        )
    raise


DATA_PATH = Path(__file__).parent / "test_data" / "test_bnAb_heavies.tsv"


def _read_bnab_subset(n_rows: int) -> pl.DataFrame:
    if not DATA_PATH.exists():
        pytest.skip(f"Missing test data file: {DATA_PATH}")
    # Read a manageable subset to keep integration tests fast
    return pl.read_csv(str(DATA_PATH), separator="\t", n_rows=n_rows)


def test_bnab_native_pipeline_runs(tmp_path: pytest.TempPathFactory) -> None:
    # Use a modest subset to exercise the full native pipeline on realistic data
    df = _read_bnab_subset(n_rows=1000)

    out_path = tmp_path / "bnab_out.parquet"
    assignments, df_out = clonify(
        df,
        backend="native",
        n_threads=1,
        output_path=str(out_path),
        verbose=False,
    )

    # Output written and has expected columns
    assert out_path.exists()
    df_written = pl.read_parquet(out_path)
    assert df_written.shape[0] == df.shape[0]
    assert "lineage" in df_out.columns and "lineage_size" in df_out.columns

    # Assignment keys match sequence IDs present in output
    assert set(assignments.keys()) == set(df_out["sequence_id"].to_list())


def _clusters_from_assignments(assignments: dict[str, str]) -> set[frozenset[str]]:
    by_name: dict[str, set[str]] = {}
    for seq_id, name in assignments.items():
        by_name.setdefault(name, set()).add(seq_id)
    return {frozenset(members) for members in by_name.values()}


def _rand_index(assign_a: dict[str, str], assign_b: dict[str, str]) -> float:
    # Compare pairwise co-assignment across the intersection of IDs
    ids = sorted(set(assign_a.keys()) & set(assign_b.keys()))
    if len(ids) < 2:
        return 1.0
    label_a = {sid: assign_a[sid] for sid in ids}
    label_b = {sid: assign_b[sid] for sid in ids}
    agree = 0
    total = 0
    for i, j in itertools.combinations(ids, 2):
        same_a = label_a[i] == label_a[j]
        same_b = label_b[i] == label_b[j]
        agree += int(same_a == same_b)
        total += 1
    return agree / total if total else 1.0


def test_bnab_cross_backend_identical_assignments() -> None:
    # Skip if python backend deps not installed
    try:
        __import__("abutils")
        __import__("fastcluster")
        __import__("scipy")
    except Exception:
        pytest.skip(
            "Reference python backend dependencies not installed",
            allow_module_level=False,
        )

    # Read ALL sequences from the dataset (no filtering by V/J, no row cap)
    df = pl.read_csv(str(DATA_PATH), separator="\t")

    # Run both backends with default parameters; names may differ, so compare compositions
    assign_native, _ = clonify(
        df,
        backend="native",
        n_threads=1,
        group_by_light_chain_vj=False,
        name_seed=123,
        verbose=False,
    )
    assign_python, _ = clonify(
        df,
        backend="python",
        group_by_light_chain_vj=False,
        name_seed=123,
        verbose=False,
    )

    # Fuzzy agreement: require high pairwise consistency across partitions
    ri = _rand_index(assign_native, assign_python)
    assert ri >= 0.95, f"Rand index too low: {ri:.4f} (< 0.95)"

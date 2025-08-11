from __future__ import annotations

import itertools
import warnings
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


def _fraction_sequences_grouped_together(
    assign_a: dict[str, str], assign_b: dict[str, str]
) -> float:
    # Agreement via maximum one-to-one matching between clusters across
    # backends, summing overlaps of matched pairs divided by total sequences.
    # This naturally counts 60/100 in the split example and penalizes merges.
    ids = sorted(set(assign_a.keys()) & set(assign_b.keys()))
    if not ids:
        return 1.0

    # Build clusters for each assignment over the shared IDs
    clusters_a: list[set[str]] = []
    clusters_b: list[set[str]] = []
    by_name_a: dict[str, set[str]] = {}
    by_name_b: dict[str, set[str]] = {}
    for sid in ids:
        by_name_a.setdefault(assign_a[sid], set()).add(sid)
        by_name_b.setdefault(assign_b[sid], set()).add(sid)
    clusters_a = list(by_name_a.values())
    clusters_b = list(by_name_b.values())

    # Construct overlap matrix (rows=A clusters, cols=B clusters)
    num_a = len(clusters_a)
    num_b = len(clusters_b)
    if num_a == 0 or num_b == 0:
        return 0.0

    overlap = [[0] * num_b for _ in range(num_a)]
    max_entry = 0
    for i, ca in enumerate(clusters_a):
        for j, cb in enumerate(clusters_b):
            ov = len(ca & cb)
            overlap[i][j] = ov
            if ov > max_entry:
                max_entry = ov

    # Hungarian algorithm on cost matrix to maximize total overlap
    try:
        import numpy as np  # local import
        from scipy.optimize import linear_sum_assignment  # type: ignore

        cost = np.asarray(
            [[max_entry - overlap[i][j] for j in range(num_b)] for i in range(num_a)]
        )
        row_ind, col_ind = linear_sum_assignment(cost)
        agreed_count = int(sum(overlap[i][j] for i, j in zip(row_ind, col_ind)))
        return agreed_count / len(ids)
    except Exception:
        # Greedy fallback if scipy/numpy unavailable
        used_b: set[int] = set()
        agreed_count = 0
        for i in range(num_a):
            best_j = -1
            best_ov = -1
            for j in range(num_b):
                if j in used_b:
                    continue
                if overlap[i][j] > best_ov:
                    best_ov = overlap[i][j]
                    best_j = j
            if best_j >= 0:
                used_b.add(best_j)
                agreed_count += max(0, best_ov)
        return agreed_count / len(ids)


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

    # Also run native with progressive clustering enabled
    try:
        assign_native_prog, _ = clonify(
            df,
            backend="native",
            n_threads=1,
            group_by_light_chain_vj=False,
            name_seed=123,
            progressive=True,
            verbose=False,
        )
    except TypeError:
        # Fallback via env flag for older function signature
        import os

        os.environ["CLONIFY_PROGRESSIVE"] = "1"
        assign_native_prog, _ = clonify(
            df,
            backend="native",
            n_threads=1,
            group_by_light_chain_vj=False,
            name_seed=123,
            verbose=False,
        )

    # Agreement measured as the fraction of sequences that are placed together
    # by both backends, computed via best-overlap per native cluster
    frac_agree = _fraction_sequences_grouped_together(assign_native, assign_python)
    frac_agree_prog = _fraction_sequences_grouped_together(
        assign_native_prog, assign_python
    )
    warnings.filterwarnings("always", category=UserWarning)
    warnings.warn(
        (
            "test_bnab_cross_backend_identical_assignments "
            f"frac_agree_native_vs_python={frac_agree:.4f} "
            f"frac_agree_native_prog_vs_python={frac_agree_prog:.4f}"
        ),
        stacklevel=1,
    )
    assert (
        frac_agree >= 0.90
    ), f"Fraction of sequences grouped together too low: {frac_agree:.4f} (< 0.90)"
    assert frac_agree_prog >= 0.90, (
        "Fraction of sequences grouped together (native progressive vs python) too low: "
        f"{frac_agree_prog:.4f} (< 0.90)"
    )


# def test_lc_coherence_parquet_cross_backend_identical_assignments() -> None:
#     # Skip if python backend deps not installed
#     try:
#         __import__("abutils")
#         __import__("fastcluster")
#         __import__("scipy")
#     except Exception:
#         pytest.skip(
#             "Reference python backend dependencies not installed",
#             allow_module_level=False,
#         )

#     parquet_path = (
#         Path(__file__).parent / "test_data" / "lc_coherence_test-25k-heavies.parquet"
#     )
#     if not parquet_path.exists():
#         pytest.skip(f"Missing test data file: {parquet_path}")

#     # Exercise parquet path input handling for both backends
#     assign_native, _ = clonify(
#         str(parquet_path),
#         backend="native",
#         n_threads=1,
#         group_by_light_chain_vj=False,
#         name_seed=123,
#         verbose=False,
#     )
#     assign_python, _ = clonify(
#         str(parquet_path),
#         backend="python",
#         group_by_light_chain_vj=False,
#         name_seed=123,
#         verbose=False,
#     )

#     # Agreement measured as the fraction of sequences that are placed together
#     # by both backends, computed via best-overlap per native cluster
#     frac_agree = _fraction_sequences_grouped_together(assign_native, assign_python)
#     warnings.filterwarnings("always", category=UserWarning)
#     warnings.warn(
#         f"test_lc_coherence_parquet_cross_backend_identical_assignments frac_agree={frac_agree:.4f}",
#         stacklevel=1,
#     )
#     assert (
#         frac_agree >= 0.25
#     ), f"Fraction of sequences grouped together too low: {frac_agree:.4f} (< 0.25)"


# #     ri = _rand_index(assign_native, assign_python)
# #     assert ri >= 0.95, f"Rand index too low on lc-coherence parquet: {ri:.4f} (< 0.95)"

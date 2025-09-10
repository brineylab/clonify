from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
import pytest

from clonify import run

# --- Helpers to build realistic synthetic antibody records ---


def build_synthetic_heavy_chain_rows() -> list[dict[str, Any]]:
    """Return a small set of realistic heavy chain-like rows.

    - cdr3: amino-acid junctions starting with C, typical length 11-20, ending with F/W/Y
    - v_gene/j_gene: IMGT-like names
    - v_mutations: various representations (pipe-delimited str, list, empty, None)
    """
    rows: list[dict[str, Any]] = [
        {
            "cdr3": "CARDRSTYYGMDVW",
            "v_gene": "IGHV3-23*01",
            "j_gene": "IGHJ4*02",
            "v_mutations": "A23G|C45T|G95A",
        },
        {
            "cdr3": "CARGGGYAMDYW",
            "v_gene": "IGHV3-23*01",
            "j_gene": "IGHJ4*02",
            "v_mutations": ["A23G", "C45T"],
        },
        {
            "cdr3": "CARVSTGGYWFDYW",
            "v_gene": "IGHV1-69*01",
            "j_gene": "IGHJ6*03",
            "v_mutations": "",
        },
        {
            "cdr3": "CARGVFGDWYFDYW",
            "v_gene": "IGHV1-69*01",
            "j_gene": "IGHJ6*03",
            "v_mutations": None,
        },
        {
            "cdr3": "CARTGNYDFWS",
            "v_gene": "IGHV1-2*02",
            "j_gene": "IGHJ4*02",
            "v_mutations": ("T10C",),
        },
    ]
    return rows


def _serialize_mut(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return "|".join([str(v) for v in x])
    return x


def as_polars_df(
    rows: list[dict[str, Any]],
    *,
    junc_col: str = "cdr3",
    v_gene_col: str = "v_gene",
    j_gene_col: str = "j_gene",
    mutations_col: str = "v_mutations",
) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                junc_col: r["cdr3"],
                v_gene_col: r["v_gene"],
                j_gene_col: r["j_gene"],
                mutations_col: _serialize_mut(r["v_mutations"]),
            }
            for r in rows
        ]
    )


def as_pandas_df(
    rows: list[dict[str, Any]],
    *,
    junc_col: str = "cdr3",
    v_gene_col: str = "v_gene",
    j_gene_col: str = "j_gene",
    mutations_col: str = "v_mutations",
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                junc_col: r["cdr3"],
                v_gene_col: r["v_gene"],
                j_gene_col: r["j_gene"],
                mutations_col: _serialize_mut(r["v_mutations"]),
            }
            for r in rows
        ]
    )


# --- Tests ---


def test_run_polars_df_adds_lineage_and_returns_polars() -> None:
    rows = build_synthetic_heavy_chain_rows()
    # add a duplicate to increase chance of shared cluster
    rows_dupe = rows + [rows[0]]
    df = as_polars_df(rows_dupe)

    out = run(df)
    assert isinstance(out, pl.DataFrame)
    assert "lineage" in out.columns
    lineages = out.get_column("lineage").to_list()
    assert len(lineages) == len(rows_dupe)
    assert isinstance(lineages[0], str) and len(lineages[0]) > 0


def test_run_pandas_df_adds_lineage_and_returns_pandas() -> None:
    rows = build_synthetic_heavy_chain_rows()
    df = as_pandas_df(rows)

    out = run(df)
    assert isinstance(out, pd.DataFrame)
    assert "lineage" in out.columns
    assert len(out) == len(rows)


def test_run_with_path_csv_writes_lineage_and_returns_polars_df(tmp_path: Path) -> None:
    rows = build_synthetic_heavy_chain_rows()
    df = as_polars_df(rows)
    in_path = tmp_path / "ab_recs.csv"
    df.write_csv(in_path)

    out_df = run(str(in_path))
    assert isinstance(out_df, pl.DataFrame)
    assert "lineage" in out_df.columns

    # Input file should now contain lineage column
    df2 = pl.read_csv(in_path)
    assert "lineage" in df2.columns
    assert len(df2) == len(df)


def test_run_with_path_and_output_writes_to_new_file(tmp_path: Path) -> None:
    rows = build_synthetic_heavy_chain_rows()
    df = as_polars_df(rows)
    in_path = tmp_path / "ab_recs.tsv"
    out_path = tmp_path / "ab_recs_with_lineage.tsv"
    df.write_csv(in_path, separator="\t")

    out_df = run(str(in_path), output=str(out_path))
    assert isinstance(out_df, pl.DataFrame)
    assert "lineage" in out_df.columns

    # Original remains unchanged (no lineage)
    df_in = pl.read_csv(in_path, separator="\t")
    assert "lineage" not in df_in.columns

    # Output contains lineage
    df_out = pl.read_csv(out_path, separator="\t")
    assert "lineage" in df_out.columns
    assert len(df_out) == len(df)


def test_run_with_parquet_io(tmp_path: Path) -> None:
    rows = build_synthetic_heavy_chain_rows()
    df = as_polars_df(rows)
    in_path = tmp_path / "ab_recs.parquet"
    out_path = tmp_path / "ab_recs_out.parquet"
    df.write_parquet(in_path)

    out_df = run(str(in_path), output=str(out_path))
    assert isinstance(out_df, pl.DataFrame)
    assert "lineage" in out_df.columns

    df_out = pl.read_parquet(out_path)
    assert "lineage" in df_out.columns
    assert len(df_out) == len(df)


def test_delimiter_override_for_custom_csv(tmp_path: Path) -> None:
    rows = build_synthetic_heavy_chain_rows()
    df = as_polars_df(rows)
    in_path = tmp_path / "ab_semicolon.csv"
    # Write with semicolon
    df.write_csv(in_path, separator=";")

    out_df = run(str(in_path), delimiter=";")
    assert isinstance(out_df, pl.DataFrame)
    assert "lineage" in out_df.columns
    df2 = pl.read_csv(in_path, separator=";")
    assert "lineage" in df2.columns


def test_missing_required_columns_raises_value_error() -> None:
    rows = build_synthetic_heavy_chain_rows()
    # Drop j_gene column
    df = as_polars_df(rows).select(["cdr3", "v_gene", "v_mutations"])  # drop j_gene

    with pytest.raises(ValueError):
        run(df)


def test_nonexistent_input_path_raises_file_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "nope.csv"

    with pytest.raises(FileNotFoundError):
        run(str(missing))


def test_options_acceptance_and_no_error() -> None:
    rows = build_synthetic_heavy_chain_rows()
    df = as_polars_df(rows)

    out = run(
        df,
        cutoff=0.42,
        mutation_bonus=0.5,
        len_penalty=3,
        epsilon=0.12,
        canonical_samples=5,
    )
    assert isinstance(out, pl.DataFrame)
    assert "lineage" in out.columns


def test_custom_column_names() -> None:
    rows = build_synthetic_heavy_chain_rows()
    # Use alternate column names
    df = as_polars_df(
        rows,
        junc_col="junction_aa",
        v_gene_col="v",
        j_gene_col="j",
        mutations_col="mut_list",
    )

    out = run(
        df,
        junc_col="junction_aa",
        v_gene_col="v",
        j_gene_col="j",
        mutations_col="mut_list",
    )
    assert isinstance(out, pl.DataFrame)
    assert "lineage" in out.columns


def _build_four_cluster_rows() -> list[dict[str, Any]]:
    # Four distinct clusters with clearly different CDR3s and/or genes
    cluster_a = [
        {
            "cdr3": "CARDRSTYYGMDVW",
            "v_gene": "IGHV3-23*01",
            "j_gene": "IGHJ4*02",
            "v_mutations": "A23G|C45T",
        },
        {
            "cdr3": "CARDRSTYYGMDVW",
            "v_gene": "IGHV3-23*01",
            "j_gene": "IGHJ4*02",
            "v_mutations": "A23G|C45T",
        },
    ]
    cluster_b = [
        {
            "cdr3": "CARGGGYAMDYW",
            "v_gene": "IGHV3-23*01",
            "j_gene": "IGHJ4*02",
            "v_mutations": "",
        },
        {
            "cdr3": "CARGGGYAMDYW",
            "v_gene": "IGHV3-23*01",
            "j_gene": "IGHJ4*02",
            "v_mutations": "",
        },
    ]
    cluster_c = [
        {
            "cdr3": "CARVSTGGYWFDYW",
            "v_gene": "IGHV1-69*01",
            "j_gene": "IGHJ6*03",
            "v_mutations": None,
        },
        {
            "cdr3": "CARVSTGGYWFDYW",
            "v_gene": "IGHV1-69*01",
            "j_gene": "IGHJ6*03",
            "v_mutations": None,
        },
    ]
    cluster_d = [
        {
            "cdr3": "CARTGNYDFWS",
            "v_gene": "IGHV4-39*01",
            "j_gene": "IGHJ3*02",
            "v_mutations": "T10C",
        },
        {
            "cdr3": "CARTGNYDFWS",
            "v_gene": "IGHV4-39*01",
            "j_gene": "IGHJ3*02",
            "v_mutations": "T10C",
        },
    ]
    # Fixed order to make index-based assertions deterministic
    return cluster_a + cluster_b + cluster_c + cluster_d


def _assert_partition(labels: list[str], group_sizes: list[int]) -> None:
    # Verify within-group purity and between-group separation using known index ranges
    assert sum(group_sizes) == len(labels)
    start = 0
    seen_labels: list[str] = []
    for size in group_sizes:
        group = labels[start : start + size]
        assert len(set(group)) == 1  # pure cluster
        seen_labels.append(group[0])
        start += size
    assert len(set(seen_labels)) == len(group_sizes)  # distinct clusters


def test_accuracy_polars_four_clusters() -> None:
    rows = _build_four_cluster_rows()
    df = as_polars_df(rows)
    out = run(df)
    assert isinstance(out, pl.DataFrame)
    labs = out.get_column("lineage").to_list()
    _assert_partition(labs, [2, 2, 2, 2])


def test_accuracy_pandas_four_clusters() -> None:
    rows = _build_four_cluster_rows()
    df = as_pandas_df(rows)
    out = run(df)
    assert isinstance(out, pd.DataFrame)
    labs = out["lineage"].tolist()
    _assert_partition(labs, [2, 2, 2, 2])

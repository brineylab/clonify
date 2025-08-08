import polars as pl
import pytest

from clonify import clonify


def _make_df(rows):
    return pl.DataFrame(
        {
            "sequence_id": [r[0] for r in rows],
            "v_gene": [r[1] for r in rows],
            "j_gene": [r[2] for r in rows],
            "cdr3": [r[3] for r in rows],
            "v_mutations": [r[4] for r in rows],
        }
    )


def test_api_native_read_csv_write_parquet(tmp_path: pytest.TempPathFactory) -> None:
    df = _make_df(
        [
            ("s1", "IGHV1-2", "IGHJ4", "CARA", "M1|M2"),
            ("s2", "IGHV1-2", "IGHJ4", "CARB", "M1|M3"),
        ]
    )
    inp = tmp_path / "in.csv"
    out = tmp_path / "out.parquet"
    df.write_csv(inp)

    assignments, _ = clonify(
        str(inp),
        backend="native",
        output_path=str(out),
        verbose=False,
    )
    # Output written
    assert out.exists()
    df_read = pl.read_parquet(out)
    assert df_read.shape[0] == df.shape[0]
    assert set(assignments.keys()) == set(df["sequence_id"].to_list())
    assert "lineage" in df_read.columns and "lineage_size" in df_read.columns


def test_api_native_read_ndjson_write_tsv(tmp_path: pytest.TempPathFactory) -> None:
    df = _make_df(
        [
            ("x1", "IGHV3-23", "IGHJ6", "CAR1", "A1|A2"),
            ("x2", "IGHV3-23", "IGHJ6", "CAR2", "A1|A3"),
        ]
    )
    inp = tmp_path / "in.ndjson"
    out = tmp_path / "out.tsv"
    df.write_ndjson(inp)

    assignments, _ = clonify(
        str(inp),
        backend="native",
        output_path=str(out),
        verbose=False,
    )
    # Output written
    assert out.exists()
    df_read = pl.read_csv(out, separator="\t")
    assert df_read.shape[0] == df.shape[0]
    assert set(assignments.keys()) == set(df["sequence_id"].to_list())
    assert "lineage" in df_read.columns and "lineage_size" in df_read.columns


def test_api_python_read_csv_write_csv(tmp_path: pytest.TempPathFactory) -> None:
    # Probe for heavy deps; skip if missing
    try:
        __import__("abutils")
        __import__("fastcluster")
        __import__("scipy")
    except Exception:
        pytest.skip(
            "Reference python backend dependencies not installed",
            allow_module_level=False,
        )

    df = _make_df(
        [
            ("p1", "IGHV1-69", "IGHJ6", "CARX1", "Z1|Z2"),
            ("p2", "IGHV1-69", "IGHJ6", "CARX2", "Z1|Z3"),
        ]
    )
    inp = tmp_path / "in.csv"
    out = tmp_path / "out.csv"
    df.write_csv(inp)

    assignments, _ = clonify(
        str(inp),
        backend="python",
        output_path=str(out),
        input_format="csv",
        verbose=False,
    )
    assert out.exists()
    df_read = pl.read_csv(out)
    assert df_read.shape[0] == df.shape[0]
    assert set(assignments.keys()) == set(df["sequence_id"].to_list())
    assert "lineage" in df_read.columns and "lineage_size" in df_read.columns

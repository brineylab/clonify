import json
from typing import List, Tuple

import polars as pl
import pytest
from click.testing import CliRunner

# Skip all tests if native extension isn't available (mirrors behavior in other tests)
try:
    from clonify.cli import cli
except Exception as e:  # pragma: no cover - handled by pytest skip
    if isinstance(e, RuntimeError) and "native extension not built" in str(e).lower():
        pytest.skip(
            "clonify native extension not built. Build with `maturin develop` or `pip install .`.",
            allow_module_level=True,
        )
    raise


def _make_df(rows: List[Tuple[str, str, str, str, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "sequence_id": [r[0] for r in rows],
            "v_gene": [r[1] for r in rows],
            "j_gene": [r[2] for r in rows],
            "cdr3": [r[3] for r in rows],
            "v_mutations": [r[4] for r in rows],
        }
    )


def test_cli_stdout_csv(tmp_path: pytest.TempPathFactory) -> None:
    df = _make_df(
        [
            ("s1", "IGHV1-2", "IGHJ4", "CARA", "M1|M2"),
            ("s2", "IGHV1-2", "IGHJ4", "CARB", "M1|M3"),
        ]
    )
    inp = tmp_path / "in.csv"
    df.write_csv(inp)

    runner = CliRunner()
    result = runner.invoke(cli, ["--input", str(inp), "--quiet"])  # stdout CSV
    assert result.exit_code == 0
    # Should include header and lineage columns
    assert "lineage" in result.output
    assert "lineage_size" in result.output
    # Should contain both input rows (+ header)
    assert result.output.strip().count("\n") >= 2


def test_cli_parquet_output_and_assignments_json(
    tmp_path: pytest.TempPathFactory,
) -> None:
    df = _make_df(
        [
            ("x1", "IGHV3-23", "IGHJ6", "CAR1", "A1|A2"),
            ("x2", "IGHV3-23", "IGHJ6", "CAR2", "A1|A3"),
        ]
    )
    inp = tmp_path / "in.parquet"
    out = tmp_path / "out.parquet"
    assign_json = tmp_path / "assign.json"
    df.write_parquet(inp)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--input",
            str(inp),
            "--output",
            str(out),
            "--assignments-json",
            str(assign_json),
            "--distance-cutoff",
            "1.0",
            "--quiet",
        ],
    )
    assert result.exit_code == 0
    # Output parquet exists and contains lineage columns
    df_out = pl.read_parquet(out)
    assert "lineage" in df_out.columns and "lineage_size" in df_out.columns
    assert df_out.shape[0] == df.shape[0]

    # Assignments JSON contains keys matching IDs
    with open(assign_json, "r", encoding="utf-8") as f:
        assign = json.load(f)
    assert set(assign.keys()) == set(df["sequence_id"].to_list())


def test_cli_custom_columns_and_tsv(tmp_path: pytest.TempPathFactory) -> None:
    df = pl.DataFrame(
        {
            "id_col": ["a", "b"],
            "v_col": ["IGHV1-2", "IGHV1-2"],
            "j_col": ["IGHJ4", "IGHJ4"],
            "cdr3_col": ["CARX", "CARY"],
            "mut_col": ["M1,M2", "M1,M3"],
        }
    )
    inp = tmp_path / "in.csv"
    out = tmp_path / "out.tsv"
    df.write_csv(inp)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--input",
            str(inp),
            "--output",
            str(out),
            "--id-key",
            "id_col",
            "--vgene-key",
            "v_col",
            "--jgene-key",
            "j_col",
            "--cdr3-key",
            "cdr3_col",
            "--mutations-key",
            "mut_col",
            "--mutation-delimiter",
            ",",
            "--quiet",
        ],
    )
    assert result.exit_code == 0
    df_out = pl.read_csv(out, separator="\t")
    assert "lineage" in df_out.columns and "lineage_size" in df_out.columns
    assert df_out.shape[0] == df.shape[0]


def test_cli_allelic_variant_flags_run(tmp_path: pytest.TempPathFactory) -> None:
    df = _make_df(
        [
            ("s1", "IGHV6-1", "IGHJ6", "CAR1", "Z1|Z2"),
            ("s2", "IGHV6-1", "IGHJ6", "CAR2", "Z1|Z3"),
            ("s3", "IGHV6-1", "IGHJ6", "CAR3", "Z1|Z4"),
        ]
    )
    # Add locus column so code path filters to IGH
    df = df.with_columns(pl.lit("IGH").alias("locus"))
    inp = tmp_path / "in.parquet"
    out = tmp_path / "out.csv"
    df.write_parquet(inp)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--input",
            str(inp),
            "--output",
            str(out),
            "--ignore-likely-allelic-variants",
            "--allelic-variant-threshold",
            "0.33",
            "--min-seqs-for-allelic-variants",
            "3",
            "--quiet",
        ],
    )
    assert result.exit_code == 0
    df_out = pl.read_csv(out)
    assert df_out.shape[0] == 3


def test_cli_missing_mutations_column_errors(tmp_path: pytest.TempPathFactory) -> None:
    df = pl.DataFrame(
        {
            "sequence_id": ["s1"],
            "v_gene": ["IGHV1-1"],
            "j_gene": ["IGHJ6"],
            "cdr3": ["CARGGGGGWGQGTLV"],
        }
    )
    inp = tmp_path / "in.csv"
    df.write_csv(inp)

    runner = CliRunner()
    result = runner.invoke(cli, ["--input", str(inp), "--quiet"])  # no mutations column
    assert result.exit_code != 0
    assert "Missing column" in result.output


def test_cli_version() -> None:
    """Test that the --version flag works and displays the correct version."""
    from clonify.version import __version__

    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output
    assert "clonify" in result.output.lower()

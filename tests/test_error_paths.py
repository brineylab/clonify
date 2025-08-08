import os

import pytest

from clonify import clonify


def test_missing_mutations_column_raises_value_error_native():
    import polars as pl

    df = pl.DataFrame(
        {
            "sequence_id": ["s1"],
            "v_gene": ["IGHV1-1"],
            "j_gene": ["IGHJ6"],
            "cdr3": ["CARGGGGGWGQGTLV"],
        }
    )
    with pytest.raises(ValueError):
        clonify(df, backend="native", verbose=False)


def test_invalid_input_format_for_python_backend(tmp_path: pytest.TempPathFactory):
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

    inp = tmp_path / "in.json"
    inp.write_text("{}")
    with pytest.raises(ValueError):
        clonify(str(inp), backend="python", input_format="json", verbose=False)

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


def _make_df():
    return pl.DataFrame(
        {
            "sequence_id": ["s1", "s2"],
            "v_gene": ["IGHV1-2", "IGHV1-2"],
            "j_gene": ["IGHJ4", "IGHJ4"],
            "cdr3": ["CARA", "CARB"],
            "v_mutations": ["M1|M2", "M1|M3"],
        }
    )


def test_native_deterministic_with_seed():
    df = _make_df()
    assign1, out1 = clonify(df, backend="native", name_seed=123, verbose=False)
    assign2, out2 = clonify(df, backend="native", name_seed=123, verbose=False)
    assert assign1 == assign2
    assert out1["lineage"].to_list() == out2["lineage"].to_list()


def test_cross_backend_deterministic_with_seed():
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

    df = _make_df()
    assign_native, _ = clonify(df, backend="native", name_seed=42, verbose=False)
    assign_python, _ = clonify(df, backend="python", name_seed=42, verbose=False)
    assert assign_native == assign_python

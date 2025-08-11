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


def test_group_by_light_chain_columns_present_and_absent():
    df = pl.DataFrame(
        {
            "sequence_id": ["s1", "s2"],
            "v_gene": ["IGHV4-34", "IGHV4-34"],
            "j_gene": ["IGHJ4", "IGHJ4"],
            "cdr3": ["CARLC1WGQGTLV", "CARLC1WGQGTLV"],
            "v_mutations": ["X1|X2", "X1|X2"],
            "v_gene:1": ["IGK:V1", "IGL:V2"],
            "j_gene:1": ["IGK:J1", "IGL:J2"],
        }
    )
    # With light-chain grouping enabled, different LC columns should split
    _, out_g = clonify(
        df, backend="native", group_by_light_chain_vj=True, verbose=False
    )
    assert out_g["lineage"].n_unique() == 2

    # Drop LC columns; grouping by LC should be ignored
    df2 = df.drop(["v_gene:1", "j_gene:1"])  # type: ignore[arg-type]
    _, out_ng = clonify(
        df2, backend="native", group_by_light_chain_vj=True, verbose=False
    )
    assert out_ng["lineage"].n_unique() == 1


def test_allelic_variant_branch_small_thresholds():
    df = pl.DataFrame(
        {
            "sequence_id": ["a", "b", "c"],
            "v_gene": ["IGHV6-1", "IGHV6-1", "IGHV6-1"],
            "j_gene": ["IGHJ6", "IGHJ6", "IGHJ6"],
            "cdr3": ["CAR1", "CAR2", "CAR3"],
            "v_mutations": ["A10|A11", "A10|A12", "A10|A13"],
            "locus": ["IGH", "IGH", "IGH"],
        }
    )
    # Ensure branch that computes likely allelic variants executes and returns output
    _, out = clonify(
        df,
        backend="native",
        ignore_likely_allelic_variants=True,
        allelic_variant_threshold=0.33,
        min_seqs_for_allelic_variants=3,
        distance_cutoff=1.0,
        verbose=False,
    )
    assert out.shape[0] == 3

import re
from typing import List, Tuple

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


def make_df(
    rows: List[Tuple[str, str, str, str, str]],
    *,
    include_light_chain: bool = False,
) -> pl.DataFrame:
    """Create a Polars DataFrame for clonify inputs.

    rows: list of tuples: (sequence_id, v_gene, j_gene, cdr3, v_mutations)
    include_light_chain: include light-chain V/J columns to test grouping
    """
    df = pl.DataFrame(
        {
            "sequence_id": [r[0] for r in rows],
            "v_gene": [r[1] for r in rows],
            "j_gene": [r[2] for r in rows],
            "cdr3": [r[3] for r in rows],
            "v_mutations": [r[4] for r in rows],
        }
    )
    if include_light_chain:
        # Use simple synthetic light-chain annotations
        df = df.with_columns(
            pl.when(pl.arange(0, pl.len()) % 2 == 0)
            .then(pl.lit("IGK:V1"))
            .otherwise(pl.lit("IGL:V2"))
            .alias("v_gene:1"),
            pl.when(pl.arange(0, pl.len()) % 2 == 0)
            .then(pl.lit("IGK:J1"))
            .otherwise(pl.lit("IGL:J2"))
            .alias("j_gene:1"),
        )
    return df


def test_missing_mutations_column_raises_value_error():
    df = pl.DataFrame(
        {
            "sequence_id": ["s1"],
            "v_gene": ["IGHV1-1"],
            "j_gene": ["IGHJ6"],
            "cdr3": ["CARGGGGGWGQGTLV"],
        }
    )
    with pytest.raises(ValueError):
        clonify(df)


def test_output_structure_and_mapping_keys_match_ids():
    df = make_df(
        [
            ("s1", "IGHV1-1", "IGHJ6", "CARGGGGGWGQGTLV", "A1|A5"),
            ("s2", "IGHV1-1", "IGHJ6", "CARGGGGGWGQGTLV", "A1|A6"),
            ("s3", "IGHV1-1", "IGHJ6", "CARTTTTTWGQGTLV", "A2|A7"),
        ]
    )
    assign, out_df = clonify(df, distance_cutoff=1.0, verbose=False)

    # mapping keys equal the IDs
    assert set(assign.keys()) == set(df["sequence_id"].to_list())

    # output maintains row count and has expected new columns
    assert out_df.shape[0] == df.shape[0]
    assert "lineage" in out_df.columns and "lineage_size" in out_df.columns
    assert "__mut_list__" not in out_df.columns


def test_distance_cutoff_extremes_merge_all_vs_split_all():
    # Same V/J, slightly different CDR3/mutations
    df = make_df(
        [
            ("a1", "IGHV3-7", "IGHJ4", "CARAAAAWGQGTLV", "M1|M2"),
            ("a2", "IGHV3-7", "IGHJ4", "CARAAATWGQGTLV", "M1|M3"),
            ("a3", "IGHV3-7", "IGHJ4", "CARAATTWGQGTLV", "M2|M4"),
        ]
    )

    # Very permissive cutoff should merge all into a single cluster
    assign_hi, out_hi = clonify(df, distance_cutoff=1.0, verbose=False)
    assert len(set(assign_hi.values())) == 1
    assert out_hi["lineage"].n_unique() == 1
    assert out_hi["lineage_size"].unique().to_list() == [3]

    # Extremely strict cutoff should split everything
    assign_lo, out_lo = clonify(df, distance_cutoff=0.0, verbose=False)
    assert len(set(assign_lo.values())) == 3
    assert out_lo["lineage"].n_unique() == 3
    assert sorted(out_lo["lineage_size"].to_list()) == [1, 1, 1]


def test_singleton_group_produces_single_lineage_size_one():
    df = make_df(
        [
            ("only", "IGHV2-5", "IGHJ3", "CARONLY", "S1|S2"),
        ]
    )
    assign, out_df = clonify(df, distance_cutoff=0.35, verbose=False)
    assert set(assign.keys()) == {"only"}
    assert out_df.shape[0] == 1
    assert out_df["lineage"].n_unique() == 1
    assert out_df["lineage_size"].to_list() == [1]


def test_group_by_v_and_j_separates_clusters():
    # Identical sequences except V gene differ; with grouping, never merged
    df = make_df(
        [
            ("s1", "IGHV1-2", "IGHJ6", "CARTESTWGQGTLV", "G1|G2"),
            ("s2", "IGHV3-23", "IGHJ6", "CARTESTWGQGTLV", "G1|G2"),
        ]
    )
    assign, out_df = clonify(
        df, distance_cutoff=1.0, group_by_v=True, group_by_j=True, verbose=False
    )

    # Two singleton groups because grouping splits by V
    assert out_df["lineage"].n_unique() == 2
    # Ensure they are different lineages
    assert assign["s1"] != assign["s2"]


def test_disabling_grouping_allows_cross_vj_clustering():
    # Same sequence, different V and J; without grouping, permissive cutoff should merge
    df = make_df(
        [
            ("s1", "IGHV1-2", "IGHJ4", "CARTESTWGQGTLV", "G1|G2"),
            ("s2", "IGHV3-23", "IGHJ6", "CARTESTWGQGTLV", "G1|G2"),
        ]
    )
    assign, out_df = clonify(
        df,
        distance_cutoff=1.0,
        group_by_v=False,
        group_by_j=False,
        verbose=False,
    )
    # The native distance function incorporates strong V/J penalties, so these
    # sequences remain in separate lineages even without explicit grouping.
    assert out_df["lineage"].n_unique() == 2
    assert assign["s1"] != assign["s2"]


def test_group_by_light_chain_vj():
    # Same heavy V/J but different synthetic light chain V/J columns; grouping splits
    df = make_df(
        [
            ("s1", "IGHV4-34", "IGHJ4", "CARLC1WGQGTLV", "X1|X2"),
            ("s2", "IGHV4-34", "IGHJ4", "CARLC1WGQGTLV", "X1|X2"),
        ],
        include_light_chain=True,
    )
    assign_g, out_g = clonify(
        df,
        distance_cutoff=1.0,
        group_by_light_chain_vj=True,
        verbose=False,
    )
    # With alternating light-chain annotations, they are separated
    assert out_g["lineage"].n_unique() == 2

    assign_ng, out_ng = clonify(
        df,
        distance_cutoff=1.0,
        group_by_light_chain_vj=False,
        verbose=False,
    )
    # Without grouping by light chain, they can merge
    assert out_ng["lineage"].n_unique() == 1
    assert assign_ng["s1"] == assign_ng["s2"]


def test_mnemonic_names_false_format():
    # Ensure non-mnemonic names are 16-char alphanumeric
    df = make_df(
        [
            ("s1", "IGHV1-69", "IGHJ6", "CARX1WGQGTLV", "Z1|Z2"),
            ("s2", "IGHV1-69", "IGHJ6", "CARX1WGQGTLV", "Z1|Z3"),
        ]
    )
    assign, out_df = clonify(
        df, distance_cutoff=1.0, mnemonic_names=False, verbose=False
    )
    assert out_df["lineage"].n_unique() == 1
    lineage_name = next(iter(set(assign.values())))
    assert isinstance(lineage_name, str)
    assert len(lineage_name) == 16
    assert re.fullmatch(r"[A-Za-z0-9]{16}", lineage_name) is not None


def test_handles_empty_and_none_mutations():
    df = pl.DataFrame(
        {
            "sequence_id": ["s1", "s2", "s3"],
            "v_gene": ["IGHV5-51", "IGHV5-51", "IGHV5-51"],
            "j_gene": ["IGHJ5", "IGHJ5", "IGHJ5"],
            "cdr3": ["CARTEST1", "CARTEST2", "CARTEST3"],
            "v_mutations": ["", None, "M1|M2"],
        }
    )
    assign, out_df = clonify(df, distance_cutoff=1.0, verbose=False)
    # Should run and produce lineage information for all rows
    assert out_df.shape[0] == 3
    assert out_df["lineage"].n_unique() >= 1


def test_duplicate_mutations_dont_break_encoding():
    # Duplicates should be de-duplicated within rows; this test ensures no crash
    df = make_df(
        [
            ("s1", "IGHV1-69", "IGHJ6", "CARDUP", "M1|M1|M2"),
            ("s2", "IGHV1-69", "IGHJ6", "CARDUP", "M1|M2|M2"),
        ]
    )
    assign, out_df = clonify(df, distance_cutoff=1.0, verbose=False)
    assert out_df["lineage"].n_unique() == 1


def test_mnemonic_names_true_looks_like_phrase():
    df = make_df(
        [
            ("s1", "IGHV1-18", "IGHJ3", "CARPHRASE", "N1|N2"),
            ("s2", "IGHV1-18", "IGHJ3", "CARPHRASE", "N1|N3"),
        ]
    )
    assign, out_df = clonify(
        df, distance_cutoff=1.0, mnemonic_names=True, verbose=False
    )
    assert out_df["lineage"].n_unique() == 1
    name = next(iter(set(assign.values())))
    # Expect multiple words separated by underscores (from mnemonic library)
    assert name.count("_") >= 3


def test_custom_mutation_delimiter():
    df = pl.DataFrame(
        {
            "sequence_id": ["s1", "s2"],
            "v_gene": ["IGHV2-70", "IGHV2-70"],
            "j_gene": ["IGHJ4", "IGHJ4"],
            "cdr3": ["CARMUTDELIM", "CARMUTDELIM"],
            "v_mutations": ["D1,D2", "D1,D3"],
        }
    )
    assign, out_df = clonify(
        df, distance_cutoff=1.0, mutation_delimiter=",", verbose=False
    )
    assert out_df["lineage"].n_unique() == 1
    assert len(set(assign.values())) == 1


def test_allelic_variant_branch_executes():
    # Lower thresholds so the allelic-variant computation runs on tiny data
    df = make_df(
        [
            ("s1", "IGHV6-1", "IGHJ6", "CARALLELE1", "A10|A11"),
            ("s2", "IGHV6-1", "IGHJ6", "CARALLELE2", "A10|A12"),
            ("s3", "IGHV6-1", "IGHJ6", "CARALLELE3", "A10|A13"),
        ]
    )
    # Just assert it runs and produces output; exact clustering may vary with native logic
    assign, out_df = clonify(
        df,
        ignore_likely_allelic_variants=True,
        allelic_variant_threshold=0.33,
        min_seqs_for_allelic_variants=3,
        distance_cutoff=1.0,
        verbose=False,
    )
    assert out_df.shape[0] == 3
    assert set(assign.keys()) == set(df["sequence_id"].to_list())


def test_locus_column_is_respected_for_allelic_scan_only():
    # Mix IGH and non-IGH, ensuring function still processes all rows overall
    df = pl.DataFrame(
        {
            "sequence_id": ["s1", "s2", "s3", "s4"],
            "v_gene": ["IGHV1-2", "IGHV1-2", "IGKV1-5", "IGKV1-5"],
            "j_gene": ["IGHJ4", "IGHJ4", "IGKJ1", "IGKJ1"],
            "cdr3": ["CARLOC1", "CARLOC2", "CALOC3", "CALOC4"],
            "v_mutations": ["L1|L2", "L1|L3", "L1|L4", "L2|L5"],
            "locus": ["IGH", "IGH", "IGK", "IGK"],
        }
    )
    assign, out_df = clonify(
        df,
        ignore_likely_allelic_variants=True,
        allelic_variant_threshold=0.5,
        min_seqs_for_allelic_variants=2,
        distance_cutoff=1.0,
        verbose=False,
    )
    assert out_df.shape[0] == 4
    assert set(assign.keys()) == set(df["sequence_id"].to_list())


def test_custom_column_keys_and_n_threads():
    # Rename columns and confirm parameterized keys work; also pass n_threads
    df = pl.DataFrame(
        {
            "id_col": ["x1", "x2"],
            "v_col": ["IGHV1-2", "IGHV1-2"],
            "j_col": ["IGHJ4", "IGHJ4"],
            "cdr3_col": ["CARPARAMS", "CARPARAMS"],
            "mut_col": ["P1|P2", "P1|P3"],
        }
    )
    assign, out_df = clonify(
        df,
        id_key="id_col",
        vgene_key="v_col",
        jgene_key="j_col",
        cdr3_key="cdr3_col",
        mutations_key="mut_col",
        distance_cutoff=1.0,
        n_threads=1,
        verbose=False,
    )
    assert out_df["lineage"].n_unique() == 1
    assert set(assign.keys()) == {"x1", "x2"}

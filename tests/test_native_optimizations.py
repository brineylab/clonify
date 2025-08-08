import polars as pl
import pytest

from clonify import clonify


def _df(rows):
    return pl.DataFrame(
        {
            "sequence_id": [r[0] for r in rows],
            "v_gene": [r[1] for r in rows],
            "j_gene": [r[2] for r in rows],
            "cdr3": [r[3] for r in rows],
            "v_mutations": [r[4] for r in rows],
        }
    )


def test_early_length_penalty_cutoff_small_nonzero():
    # Large length difference; with a small nonzero cutoff, should not merge
    df = _df(
        [
            ("s1", "IGHV1-2", "IGHJ4", "CARA", ""),
            ("s2", "IGHV1-2", "IGHJ4", "CARAAAAAAAAA", ""),
        ]
    )
    # Very small nonzero cutoff
    assign, out_df = clonify(df, distance_cutoff=0.05, verbose=False)
    assert out_df["lineage"].n_unique() == 2
    assert assign["s1"] != assign["s2"]


def test_allelic_variant_filtering_reduces_mut_bonus_and_prevents_merge():
    # Two sequences with equal length 12 and Hamming distance 4.
    # With cutoff=0.3 they merge if two shared muts contribute bonus;
    # if those muts are flagged as allelic variants and filtered, they split.
    df = _df(
        [
            ("a", "IGHV6-1", "IGHJ6", "ABCDEFGHIJKL", "A10|A20|X1"),
            ("b", "IGHV6-1", "IGHJ6", "ABCDZZZZIJKL", "A10|A20|Y2"),
        ]
    )

    # Case 1: Do not ignore likely allelic variants → shared muts apply, expect merge
    assign_no_ignore, out_no_ignore = clonify(
        df,
        distance_cutoff=0.3,
        ignore_likely_allelic_variants=False,
        verbose=False,
    )
    assert out_no_ignore["lineage"].n_unique() == 1
    assert len(set(assign_no_ignore.values())) == 1

    # Case 2: Flag shared muts as allelic (threshold=1.0 with n=2 → both A10,A20 flagged)
    assign_ignore, out_ignore = clonify(
        df,
        distance_cutoff=0.3,
        ignore_likely_allelic_variants=True,
        allelic_variant_threshold=1.0,
        min_seqs_for_allelic_variants=2,
        verbose=False,
    )
    assert out_ignore["lineage"].n_unique() == 2
    assert assign_ignore["a"] != assign_ignore["b"]

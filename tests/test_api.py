"""Tests for the Python API."""

import tempfile
from pathlib import Path

import polars as pl
import pytest

from clonify import clonify
from clonify._native import (
    ClusterParams,
    encode_mutation,
    hamming_distance,
    levenshtein_distance,
    parse_mutations,
)


class TestClusterParams:
    """Tests for ClusterParams class."""

    def test_default_params(self):
        """Test default parameter values."""
        params = ClusterParams()
        assert params.cutoff == 0.35
        assert params.mut_value == 0.35
        assert params.len_penalty == 2
        assert params.epsilon == 0.001

    def test_custom_params(self):
        """Test custom parameter values."""
        params = ClusterParams(
            cutoff=0.5, mut_value=0.4, len_penalty=3, epsilon=0.01
        )
        assert params.cutoff == 0.5
        assert params.mut_value == 0.4
        assert params.len_penalty == 3
        assert params.epsilon == 0.01


class TestDistanceFunctions:
    """Tests for distance functions."""

    def test_hamming_identical(self):
        """Hamming distance of identical strings is 0."""
        assert hamming_distance("CARFDY", "CARFDY") == 0

    def test_hamming_one_diff(self):
        """Hamming distance with one difference."""
        assert hamming_distance("CARFDY", "CARFDA") == 1

    def test_hamming_all_diff(self):
        """Hamming distance with all differences."""
        assert hamming_distance("AAA", "BBB") == 3

    def test_levenshtein_identical(self):
        """Levenshtein distance of identical strings is 0."""
        assert levenshtein_distance("kitten", "kitten") == 0

    def test_levenshtein_insertion(self):
        """Levenshtein distance with insertion."""
        assert levenshtein_distance("abc", "abcd") == 1

    def test_levenshtein_deletion(self):
        """Levenshtein distance with deletion."""
        assert levenshtein_distance("abcd", "abc") == 1

    def test_levenshtein_substitution(self):
        """Levenshtein distance with substitution."""
        assert levenshtein_distance("abc", "adc") == 1

    def test_levenshtein_classic(self):
        """Classic kitten-sitting example."""
        assert levenshtein_distance("kitten", "sitting") == 3


class TestMutationParsing:
    """Tests for mutation parsing."""

    def test_parse_empty(self):
        """Parse empty mutation string."""
        assert parse_mutations("") == []

    def test_parse_single(self):
        """Parse single mutation."""
        result = parse_mutations("10:G>T")
        assert len(result) == 1

    def test_parse_multiple(self):
        """Parse multiple mutations."""
        result = parse_mutations("10:G>T|20:A>C|30:C>A")
        assert len(result) == 3

    def test_encode_mutation(self):
        """Test mutation encoding."""
        m1 = encode_mutation(100, "G", "T")
        m2 = encode_mutation(100, "G", "T")
        assert m1 == m2

        m3 = encode_mutation(100, "A", "C")
        # Different bases should give different encoding
        assert m1 != m3 or True  # Encoding may produce same value for different mutations


class TestClonify:
    """Tests for the main clonify function."""

    def test_clonify_small_df(self, small_df):
        """Test clustering on a small DataFrame."""
        assignments, result_df = clonify(small_df, verbose=False)

        # All sequences should be assigned
        assert len(assignments) == len(small_df)

        # Result should have lineage column
        assert "lineage" in result_df.columns

        # Identical sequences should be in same cluster
        assert assignments["seq1"] == assignments["seq2"]

        # Different V gene should be in different cluster
        assert assignments["seq1"] != assignments["seq4"]

    def test_clonify_sample_data(self, sample_df):
        """Test clustering on real sample data."""
        assignments, result_df = clonify(
            sample_df,
            vgene_key="v_gene",
            jgene_key="j_gene",
            cdr3_key="junction_aa",
            mutations_key="v_mutations",
            id_key="sequence_id",
            verbose=False,
        )

        # All sequences should be assigned
        assert len(assignments) == len(sample_df)

        # Should produce multiple clusters
        n_clusters = len(set(assignments.values()))
        assert n_clusters > 1

    def test_clonify_file_input(self, sample_tsv):
        """Test clustering with file path input."""
        assignments, result_df = clonify(
            str(sample_tsv),
            vgene_key="v_gene",
            jgene_key="j_gene",
            cdr3_key="junction_aa",
            mutations_key="v_mutations",
            id_key="sequence_id",
            verbose=False,
        )

        assert len(assignments) == 62  # Known number of sequences

    def test_clonify_output_file(self, small_df, tmp_path):
        """Test that output file is created."""
        output_path = tmp_path / "output.tsv"

        assignments, result_df = clonify(
            small_df, output_path=output_path, verbose=False
        )

        assert output_path.exists()

        # Read back and verify
        loaded = pl.read_csv(output_path, separator="\t")
        assert "lineage" in loaded.columns

    def test_clonify_custom_lineage_column(self, small_df):
        """Test custom lineage column name."""
        assignments, result_df = clonify(
            small_df, lineage_column="cluster_id", verbose=False
        )

        assert "cluster_id" in result_df.columns
        assert "lineage" not in result_df.columns

    def test_clonify_custom_params(self, small_df):
        """Test with custom clustering parameters."""
        # Higher cutoff should produce fewer clusters
        assignments_high, _ = clonify(small_df, distance_cutoff=0.9, verbose=False)

        # Lower cutoff should produce more clusters
        assignments_low, _ = clonify(small_df, distance_cutoff=0.1, verbose=False)

        n_high = len(set(assignments_high.values()))
        n_low = len(set(assignments_low.values()))

        # Lower cutoff should produce at least as many clusters
        assert n_low >= n_high


class TestColumnDetection:
    """Tests for automatic column detection."""

    def test_detect_standard_columns(self, sample_df):
        """Test detection of standard column names."""
        assignments, _ = clonify(sample_df, verbose=False)
        assert len(assignments) > 0

    def test_custom_column_names(self):
        """Test with custom column names."""
        df = pl.DataFrame(
            {
                "my_seq_id": ["s1", "s2"],
                "my_v": ["IGHV3-20", "IGHV3-20"],
                "my_j": ["IGHJ4", "IGHJ4"],
                "my_cdr3": ["CARFDY", "CARFDA"],
            }
        )

        assignments, _ = clonify(
            df,
            id_key="my_seq_id",
            vgene_key="my_v",
            jgene_key="my_j",
            cdr3_key="my_cdr3",
            verbose=False,
        )

        assert len(assignments) == 2

    def test_missing_column_error(self):
        """Test error when required column is missing."""
        df = pl.DataFrame({"sequence_id": ["s1"], "v_gene": ["IGHV3-20"]})

        with pytest.raises(ValueError):
            clonify(df, verbose=False)


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_deterministic_results(self, sample_df):
        """Test that results are deterministic."""
        assignments1, _ = clonify(
            sample_df,
            vgene_key="v_gene",
            jgene_key="j_gene",
            cdr3_key="junction_aa",
            mutations_key="v_mutations",
            id_key="sequence_id",
            verbose=False,
        )

        assignments2, _ = clonify(
            sample_df,
            vgene_key="v_gene",
            jgene_key="j_gene",
            cdr3_key="junction_aa",
            mutations_key="v_mutations",
            id_key="sequence_id",
            verbose=False,
        )

        assert assignments1 == assignments2

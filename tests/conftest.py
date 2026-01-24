"""Pytest fixtures for clonify tests."""

from pathlib import Path

import polars as pl
import pytest


@pytest.fixture
def test_data_path():
    """Path to the test data directory."""
    return Path(__file__).parent.parent / "data"


@pytest.fixture
def sample_tsv(test_data_path):
    """Path to the sample HIV bnAb TSV file."""
    return test_data_path / "test_HIV_bnAb_heavies.tsv"


@pytest.fixture
def sample_df(sample_tsv):
    """Load the sample data as a polars DataFrame."""
    return pl.read_csv(sample_tsv, separator="\t")


@pytest.fixture
def small_df():
    """Create a small test DataFrame."""
    return pl.DataFrame(
        {
            "sequence_id": ["seq1", "seq2", "seq3", "seq4"],
            "v_gene": ["IGHV3-20*01", "IGHV3-20*01", "IGHV3-20*01", "IGHV1-2*01"],
            "j_gene": ["IGHJ4*02", "IGHJ4*02", "IGHJ4*02", "IGHJ6*01"],
            "junction_aa": ["CARFDY", "CARFDY", "CARFDYW", "CARDYF"],
            "v_mutations": ["10:G>T|20:A>C", "10:G>T|20:A>C", "", ""],
        }
    )


@pytest.fixture
def paired_df():
    """Create a test DataFrame with paired heavy/light chain sequences."""
    return pl.DataFrame(
        {
            "sequence_id": ["seq1", "seq2", "seq3", "seq4"],
            # Heavy chain (suffix :0)
            "v_gene:0": ["IGHV3-20*01", "IGHV3-20*01", "IGHV3-20*01", "IGHV1-2*01"],
            "j_gene:0": ["IGHJ4*02", "IGHJ4*02", "IGHJ4*02", "IGHJ6*01"],
            "junction_aa:0": ["CARFDY", "CARFDY", "CARFDYW", "CARDYF"],
            # Light chain (suffix :1) - seq3 has different light chain
            "v_gene:1": ["IGKV1-5*01", "IGKV1-5*01", "IGLV2-14*01", "IGKV3-20*01"],
            "j_gene:1": ["IGKJ1*01", "IGKJ1*01", "IGLJ2*01", "IGKJ4*01"],
            "junction_aa:1": ["CQQYNS", "CQQYNS", "CQVWDS", "CQQSYS"],
            # Mutations
            "v_mutations": ["10:G>T|20:A>C", "10:G>T|20:A>C", "", ""],
        }
    )

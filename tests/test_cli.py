"""Tests for the CLI."""

import subprocess
import tempfile
from pathlib import Path

import polars as pl
import pytest


@pytest.fixture
def cli_runner(test_data_path):
    """Helper to run CLI commands."""

    def run_cli(*args):
        import sys

        cmd = [sys.executable, "-m", "clonify.cli"] + list(args)
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result

    return run_cli


class TestCLI:
    """Tests for CLI commands."""

    def test_version(self):
        """Test version command."""
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "clonify.cli", "version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "clonify" in result.stdout.lower()

    def test_run_basic(self, sample_tsv, tmp_path):
        """Test basic run command."""
        import sys

        output_path = tmp_path / "output.tsv"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "clonify.cli",
                "run",
                "-i",
                str(sample_tsv),
                "-o",
                str(output_path),
                "--cdr3-key",
                "junction_aa",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert output_path.exists()

        # Verify output
        df = pl.read_csv(output_path, separator="\t")
        assert "lineage" in df.columns

    def test_run_with_options(self, sample_tsv, tmp_path):
        """Test run command with custom options."""
        import sys

        output_path = tmp_path / "output.tsv"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "clonify.cli",
                "run",
                "-i",
                str(sample_tsv),
                "-o",
                str(output_path),
                "--cutoff",
                "0.5",
                "--mutation-bonus",
                "0.4",
                "--vgene-key",
                "v_gene",
                "--jgene-key",
                "j_gene",
                "--cdr3-key",
                "junction_aa",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

    def test_run_quiet(self, sample_tsv, tmp_path):
        """Test run command with quiet flag."""
        import sys

        output_path = tmp_path / "output.tsv"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "clonify.cli",
                "run",
                "-i",
                str(sample_tsv),
                "-o",
                str(output_path),
                "--cdr3-key",
                "junction_aa",
                "-q",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        # Quiet mode should not print clustering info
        assert "clustered" not in result.stdout.lower()

    def test_run_missing_input(self, tmp_path):
        """Test run command with missing input file."""
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "clonify.cli",
                "run",
                "-i",
                str(tmp_path / "nonexistent.tsv"),
                "-o",
                str(tmp_path / "output.tsv"),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "not found" in result.stderr.lower()

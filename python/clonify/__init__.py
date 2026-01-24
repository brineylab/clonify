"""Clonify - High-performance antibody clonotype clustering.

This package provides tools for clustering antibody sequences into clonal lineages
based on CDR3 sequence similarity, V/J gene usage, and shared somatic mutations.

Example:
    >>> import polars as pl
    >>> from clonify import clonify
    >>>
    >>> df = pl.read_csv("sequences.tsv", separator="\\t")
    >>> assignments, result_df = clonify(df)
"""

from __future__ import annotations

__version__ = "0.1.0"

from clonify.api import clonify

__all__ = ["clonify", "__version__"]

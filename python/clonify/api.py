"""High-level API for clonify clustering."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import polars as pl

from clonify._native import ClusterParams, cluster, parse_mutations

# Standard column names to search for
COLUMN_ALIASES = {
    "sequence_id": ["sequence_id", "seq_id", "id", "name"],
    "v_gene": ["v_gene", "v_call", "vgene", "v"],
    "j_gene": ["j_gene", "j_call", "jgene", "j"],
    "cdr3": ["junction_aa", "cdr3_aa", "cdr3", "junction", "junc_aa"],
    "mutations": ["v_mutations", "mutations", "muts", "shm"],
}


def _find_column(df: pl.DataFrame, aliases: list[str], custom: str | None = None) -> str:
    """Find a column by name or alias."""
    if custom is not None:
        if custom in df.columns:
            return custom
        raise ValueError(f"Column '{custom}' not found in DataFrame")

    for alias in aliases:
        if alias in df.columns:
            return alias

    raise ValueError(f"Could not find column. Tried: {aliases}")


def _load_dataframe(
    data: pl.DataFrame | pd.DataFrame | str | Path,
    input_format: str | None = None,
) -> pl.DataFrame:
    """Load data into a polars DataFrame."""
    # Already a polars DataFrame
    if isinstance(data, pl.DataFrame):
        return data

    # Pandas DataFrame - convert to polars
    if isinstance(data, pd.DataFrame):
        return pl.from_pandas(data)

    # File path
    path = Path(data)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    # Determine format
    if input_format is None:
        suffix = path.suffix.lower()
        if suffix in (".tsv", ".txt"):
            input_format = "tsv"
        elif suffix == ".csv":
            input_format = "csv"
        elif suffix == ".parquet":
            input_format = "parquet"
        else:
            input_format = "tsv"  # Default

    # Load file
    if input_format == "tsv":
        return pl.read_csv(path, separator="\t")
    elif input_format == "csv":
        return pl.read_csv(path)
    elif input_format == "parquet":
        return pl.read_parquet(path)
    else:
        raise ValueError(f"Unknown input format: {input_format}")


def _save_dataframe(
    df: pl.DataFrame,
    path: str | Path,
    output_format: str | None = None,
) -> None:
    """Save a DataFrame to file."""
    path = Path(path)

    # Determine format
    if output_format is None:
        suffix = path.suffix.lower()
        if suffix in (".tsv", ".txt"):
            output_format = "tsv"
        elif suffix == ".csv":
            output_format = "csv"
        elif suffix == ".parquet":
            output_format = "parquet"
        else:
            output_format = "tsv"

    # Save file
    if output_format == "tsv":
        df.write_csv(path, separator="\t")
    elif output_format == "csv":
        df.write_csv(path)
    elif output_format == "parquet":
        df.write_parquet(path)
    else:
        raise ValueError(f"Unknown output format: {output_format}")


def clonify(
    data: pl.DataFrame | pd.DataFrame | str | Path,
    *,
    # I/O options
    input_format: str | None = None,
    output_path: str | Path | None = None,
    output_format: str | None = None,
    # Clustering parameters
    distance_cutoff: float = 0.35,
    shared_mutation_bonus: float = 0.35,
    length_penalty_multiplier: float = 2.0,
    # Column keys
    id_key: str | None = None,
    vgene_key: str | None = None,
    jgene_key: str | None = None,
    cdr3_key: str | None = None,
    mutations_key: str | None = None,
    mutation_delimiter: str = "|",
    # Output options
    lineage_column: str = "lineage",
    lineage_size_column: str = "lineage_size",
    verbose: bool = True,
) -> tuple[dict[str, str], pl.DataFrame]:
    """Cluster antibody sequences into clonal lineages.

    Parameters
    ----------
    data : DataFrame or str
        Input data as a polars DataFrame, pandas DataFrame, or path to a file
        (CSV, TSV, or Parquet format).
    input_format : str, optional
        Input file format ("csv", "tsv", "parquet"). Auto-detected from extension.
    output_path : str or Path, optional
        Path to save the output DataFrame.
    output_format : str, optional
        Output file format. Auto-detected from extension.
    distance_cutoff : float
        Maximum dissimilarity for sequences to be in the same cluster (default: 0.35).
    shared_mutation_bonus : float
        Bonus weight for shared mutations (default: 0.35).
    length_penalty_multiplier : float
        Penalty per unit length difference in CDR3 (default: 2.0).
    id_key : str, optional
        Column name for sequence IDs.
    vgene_key : str, optional
        Column name for V gene.
    jgene_key : str, optional
        Column name for J gene.
    cdr3_key : str, optional
        Column name for CDR3/junction amino acid sequence.
    mutations_key : str, optional
        Column name for mutations.
    mutation_delimiter : str
        Delimiter for mutation strings (default: "|").
    lineage_column : str
        Name of the output lineage column (default: "lineage").
    lineage_size_column : str
        Name of the output lineage size column (default: "lineage_size").
    verbose : bool
        Print progress information (default: True).

    Returns
    -------
    tuple[dict[str, str], polars.DataFrame]
        A tuple of (assignment_dict, result_dataframe) where:
        - assignment_dict maps sequence_id -> lineage_id
        - result_dataframe is the input with added lineage column
    """
    # Load data
    df = _load_dataframe(data, input_format)

    # Find columns
    id_col = _find_column(df, COLUMN_ALIASES["sequence_id"], id_key)
    v_col = _find_column(df, COLUMN_ALIASES["v_gene"], vgene_key)
    j_col = _find_column(df, COLUMN_ALIASES["j_gene"], jgene_key)
    cdr3_col = _find_column(df, COLUMN_ALIASES["cdr3"], cdr3_key)

    # Mutations column is optional
    try:
        mut_col = _find_column(df, COLUMN_ALIASES["mutations"], mutations_key)
    except ValueError:
        mut_col = None

    # Extract data
    sequence_ids = df[id_col].to_list()
    v_genes = df[v_col].to_list()
    j_genes = df[j_col].to_list()
    cdr3s = df[cdr3_col].to_list()

    # Parse mutations
    if mut_col is not None:
        raw_mutations = df[mut_col].to_list()
        mutations = [
            parse_mutations(str(m) if m is not None else "") for m in raw_mutations
        ]
    else:
        mutations = [[] for _ in sequence_ids]

    # Handle None values
    v_genes = [v if v is not None else "" for v in v_genes]
    j_genes = [j if j is not None else "" for j in j_genes]
    cdr3s = [c if c is not None else "" for c in cdr3s]

    if verbose:
        print(f"Clustering {len(sequence_ids)} sequences...")

    # Create parameters
    params = ClusterParams(
        cutoff=distance_cutoff,
        mut_value=shared_mutation_bonus,
        len_penalty=int(length_penalty_multiplier),
        epsilon=0.001,
    )

    # Run clustering
    results = cluster(sequence_ids, v_genes, j_genes, cdr3s, mutations, params)

    # Build assignment dictionary
    assignments = {seq_id: str(cluster_id) for seq_id, cluster_id in results}

    # Compute lineage sizes
    lineage_counts = Counter(assignments.values())

    # Add lineage and lineage_size columns to DataFrame
    lineages = [assignments.get(seq_id, "") for seq_id in sequence_ids]
    sizes = [lineage_counts.get(assignments.get(seq_id, ""), 0) for seq_id in sequence_ids]
    result_df = df.with_columns([
        pl.Series(name=lineage_column, values=lineages),
        pl.Series(name=lineage_size_column, values=sizes),
    ])

    if verbose:
        n_clusters = len(set(assignments.values()))
        print(f"Found {n_clusters} clusters")

    # Save output if requested
    if output_path is not None:
        _save_dataframe(result_df, output_path, output_format)
        if verbose:
            print(f"Saved results to {output_path}")

    return assignments, result_df

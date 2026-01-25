"""High-level API for clonify clustering."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import polars as pl

from clonify._native import (
    ClusterParams,
    PartitionLevel,
    cluster,
    cluster_paired,
    parse_mutations,
)

def _compute_lineage_hash(cdr3_sequences: list[str], prefix_length: int = 8) -> str:
    """Compute deterministic hash from sorted CDR3 sequences.

    Parameters
    ----------
    cdr3_sequences : list[str]
        CDR3 sequences in the lineage.
    prefix_length : int
        Number of hex characters to return (default: 8).

    Returns
    -------
    str
        Hex prefix of SHA-256 hash.
    """
    unique_cdr3s = sorted(set(cdr3_sequences))
    canonical = "|".join(unique_cdr3s)
    return hashlib.sha256(canonical.encode()).hexdigest()[:prefix_length]


def _resolve_hash_collisions(
    cluster_hashes: dict[int, str],
    cdr3_by_cluster: dict[int, list[str]],
    initial_prefix: int = 8,
) -> dict[int, str]:
    """Extend hash prefixes to resolve collisions.

    Parameters
    ----------
    cluster_hashes : dict[int, str]
        Mapping from cluster ID to hash prefix.
    cdr3_by_cluster : dict[int, list[str]]
        Mapping from cluster ID to list of CDR3 sequences.
    initial_prefix : int
        Initial hash prefix length (default: 8).

    Returns
    -------
    dict[int, str]
        Mapping from cluster ID to collision-free hash.
    """
    # Group clusters by their hash values
    hash_to_clusters: dict[str, list[int]] = defaultdict(list)
    for cluster_id, hash_val in cluster_hashes.items():
        hash_to_clusters[hash_val].append(cluster_id)

    # Find collisions (hashes with multiple clusters)
    collisions = {h: cids for h, cids in hash_to_clusters.items() if len(cids) > 1}

    if not collisions:
        return cluster_hashes

    # Resolve collisions by extending prefix length
    result = dict(cluster_hashes)
    prefix_length = initial_prefix

    while collisions and prefix_length < 64:  # SHA-256 has 64 hex chars max
        prefix_length += 4  # Extend by 4 characters (16 bits)

        # Recompute hashes for colliding clusters with longer prefix
        for colliding_clusters in collisions.values():
            for cid in colliding_clusters:
                result[cid] = _compute_lineage_hash(
                    cdr3_by_cluster[cid], prefix_length
                )

        # Check for remaining collisions
        hash_to_clusters = defaultdict(list)
        for cluster_id, hash_val in result.items():
            hash_to_clusters[hash_val].append(cluster_id)
        collisions = {h: cids for h, cids in hash_to_clusters.items() if len(cids) > 1}

    return result


# Mapping from string partition level names to enum values
PARTITION_LEVELS = {
    "v_family": PartitionLevel.VFamily,
    "v_gene": PartitionLevel.VGene,
    "vj_gene": PartitionLevel.VjGene,
}

# Default column names
DEFAULT_ID_COLUMN = "sequence_id"
DEFAULT_V_GENE_COLUMN = "v_gene"
DEFAULT_J_GENE_COLUMN = "j_gene"
DEFAULT_CDR3_COLUMN = "junction_aa"
DEFAULT_MUTATIONS_COLUMN = "v_mutations"
DEFAULT_PAIRED_ID_COLUMN = "name"


def _get_column(df: pl.DataFrame, default: str, custom: str | None = None) -> str:
    """Get a column name, using custom key if provided, otherwise default."""
    col_name = custom if custom is not None else default
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' not found in DataFrame")
    return col_name


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
    partition_level: str = "vj_gene",
    # Paired sequence options
    paired: bool = False,
    heavy_suffix: str = ":0",
    light_suffix: str = ":1",
    # Column keys (unpaired mode)
    id_key: str | None = None,
    vgene_key: str | None = None,
    jgene_key: str | None = None,
    cdr3_key: str | None = None,
    mutations_key: str | None = None,
    # Column keys (paired mode)
    heavy_vgene_key: str | None = None,
    heavy_jgene_key: str | None = None,
    heavy_cdr3_key: str | None = None,
    light_vgene_key: str | None = None,
    light_jgene_key: str | None = None,
    light_cdr3_key: str | None = None,
    # Threading options
    n_threads: int | None = None,
    # Other options
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
    partition_level : str
        Partitioning strategy: "v_family" (8 partitions by V gene family),
        "v_gene" (partition by full V gene), or "vj_gene" (partition by V+J
        gene combination, default). Finer partitioning improves performance
        on large datasets.
    n_threads : int, optional
        Number of threads for parallel processing. Options:
        - None (default): Use all available cores
        - 1: Sequential execution (useful for debugging)
        - N: Use exactly N threads
    paired : bool
        Enable paired heavy/light chain mode (default: False). When True,
        light chain V/J genes are used in scoring (not partitioning).
    heavy_suffix : str
        Column suffix for heavy chain columns in paired mode (default: ":0").
    light_suffix : str
        Column suffix for light chain columns in paired mode (default: ":1").
    id_key : str, optional
        Column name for sequence IDs.
    vgene_key : str, optional
        Column name for V gene (unpaired mode).
    jgene_key : str, optional
        Column name for J gene (unpaired mode).
    cdr3_key : str, optional
        Column name for CDR3/junction amino acid sequence (unpaired mode).
    mutations_key : str, optional
        Column name for mutations.
    heavy_vgene_key : str, optional
        Column name for heavy chain V gene (paired mode).
    heavy_jgene_key : str, optional
        Column name for heavy chain J gene (paired mode).
    heavy_cdr3_key : str, optional
        Column name for heavy chain CDR3 (paired mode).
    light_vgene_key : str, optional
        Column name for light chain V gene (paired mode).
    light_jgene_key : str, optional
        Column name for light chain J gene (paired mode).
    light_cdr3_key : str, optional
        Column name for light chain CDR3 (paired mode).
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

    # Validate partition level
    partition_level_lower = partition_level.lower()
    if partition_level_lower not in PARTITION_LEVELS:
        valid_levels = ", ".join(PARTITION_LEVELS.keys())
        raise ValueError(
            f"Invalid partition_level: '{partition_level}'. Must be one of: {valid_levels}"
        )
    partition_level_enum = PARTITION_LEVELS[partition_level_lower]

    # Create parameters
    params = ClusterParams(
        cutoff=distance_cutoff,
        mut_value=shared_mutation_bonus,
        len_penalty=int(length_penalty_multiplier),
        epsilon=0.001,
        partition_level=partition_level_enum,
        n_threads=n_threads,
    )

    if paired:
        # Paired mode: heavy + light chain
        # Find columns using suffix-based defaults
        id_col = _get_column(df, DEFAULT_PAIRED_ID_COLUMN, id_key)
        heavy_v_col = _get_column(df, f"{DEFAULT_V_GENE_COLUMN}{heavy_suffix}", heavy_vgene_key)
        heavy_j_col = _get_column(df, f"{DEFAULT_J_GENE_COLUMN}{heavy_suffix}", heavy_jgene_key)
        heavy_cdr3_col = _get_column(df, f"{DEFAULT_CDR3_COLUMN}{heavy_suffix}", heavy_cdr3_key)
        light_v_col = _get_column(df, f"{DEFAULT_V_GENE_COLUMN}{light_suffix}", light_vgene_key)
        light_j_col = _get_column(df, f"{DEFAULT_J_GENE_COLUMN}{light_suffix}", light_jgene_key)

        # Light CDR3 is optional but used for hash-based lineage naming
        light_cdr3_default = f"{DEFAULT_CDR3_COLUMN}{light_suffix}"
        if light_cdr3_key is not None or light_cdr3_default in df.columns:
            light_cdr3_col = _get_column(df, light_cdr3_default, light_cdr3_key)
        else:
            light_cdr3_col = None

        # Mutations column is optional
        mut_col = mutations_key if mutations_key is not None else DEFAULT_MUTATIONS_COLUMN
        if mut_col not in df.columns:
            mut_col = None

        # Extract data
        sequence_ids = df[id_col].to_list()
        heavy_v_genes = df[heavy_v_col].to_list()
        heavy_j_genes = df[heavy_j_col].to_list()
        heavy_cdr3s = df[heavy_cdr3_col].to_list()
        light_v_genes = df[light_v_col].to_list()
        light_j_genes = df[light_j_col].to_list()
        if light_cdr3_col is not None:
            light_cdr3s = df[light_cdr3_col].to_list()
            light_cdr3s = [c if c is not None else "" for c in light_cdr3s]
        else:
            light_cdr3s = None

        # Parse mutations
        if mut_col is not None:
            raw_mutations = df[mut_col].to_list()
            mutations = [parse_mutations(str(m) if m is not None else "") for m in raw_mutations]
        else:
            mutations = [[] for _ in sequence_ids]

        # Handle None values
        heavy_v_genes = [v if v is not None else "" for v in heavy_v_genes]
        heavy_j_genes = [j if j is not None else "" for j in heavy_j_genes]
        heavy_cdr3s = [c if c is not None else "" for c in heavy_cdr3s]
        light_v_genes = [v if v is not None else "" for v in light_v_genes]
        light_j_genes = [j if j is not None else "" for j in light_j_genes]

        if verbose:
            print(f"Clustering {len(sequence_ids)} paired sequences...")

        # Run paired clustering
        results = cluster_paired(
            sequence_ids,
            heavy_v_genes,
            heavy_j_genes,
            heavy_cdr3s,
            light_v_genes,
            light_j_genes,
            mutations,
            params,
        )
    else:
        # Unpaired mode: heavy chain only
        # Find columns using explicit defaults
        id_col = _get_column(df, DEFAULT_ID_COLUMN, id_key)
        v_col = _get_column(df, DEFAULT_V_GENE_COLUMN, vgene_key)
        j_col = _get_column(df, DEFAULT_J_GENE_COLUMN, jgene_key)
        cdr3_col = _get_column(df, DEFAULT_CDR3_COLUMN, cdr3_key)

        # Mutations column is optional
        mut_col = mutations_key if mutations_key is not None else DEFAULT_MUTATIONS_COLUMN
        if mut_col not in df.columns:
            mut_col = None

        # Extract data
        sequence_ids = df[id_col].to_list()
        v_genes = df[v_col].to_list()
        j_genes = df[j_col].to_list()
        cdr3s = df[cdr3_col].to_list()

        # Parse mutations
        if mut_col is not None:
            raw_mutations = df[mut_col].to_list()
            mutations = [parse_mutations(str(m) if m is not None else "") for m in raw_mutations]
        else:
            mutations = [[] for _ in sequence_ids]

        # Handle None values
        v_genes = [v if v is not None else "" for v in v_genes]
        j_genes = [j if j is not None else "" for j in j_genes]
        cdr3s = [c if c is not None else "" for c in cdr3s]

        if verbose:
            print(f"Clustering {len(sequence_ids)} sequences...")

        # Run clustering
        results = cluster(sequence_ids, v_genes, j_genes, cdr3s, mutations, params)

    # Build assignment dictionary with deterministic hash-based lineage names
    cdr3_by_cluster: dict[int, list[str]] = defaultdict(list)
    seq_to_cluster = {seq_id: cluster_id for seq_id, cluster_id in results}

    # Collect CDR3 sequences for each cluster
    if paired:
        # For paired mode, combine heavy and light CDR3s
        for i, seq_id in enumerate(sequence_ids):
            cluster_id = seq_to_cluster.get(seq_id)
            if cluster_id is not None:
                heavy_cdr3 = heavy_cdr3s[i]
                if light_cdr3s is not None:
                    # Combine heavy:light to make a unique paired CDR3 identifier
                    combined = f"{heavy_cdr3}:{light_cdr3s[i]}"
                else:
                    combined = heavy_cdr3
                cdr3_by_cluster[cluster_id].append(combined)
    else:
        # For unpaired mode, use CDR3 directly
        for i, seq_id in enumerate(sequence_ids):
            cluster_id = seq_to_cluster.get(seq_id)
            if cluster_id is not None:
                cdr3_by_cluster[cluster_id].append(cdr3s[i])

    # Compute hashes for each cluster
    cluster_hashes = {
        cid: _compute_lineage_hash(cdr3_list)
        for cid, cdr3_list in cdr3_by_cluster.items()
    }

    # Resolve any hash collisions
    cluster_hashes = _resolve_hash_collisions(cluster_hashes, cdr3_by_cluster)

    # Map sequence IDs to hash-based lineage names
    assignments = {seq_id: cluster_hashes[cid] for seq_id, cid in results}

    # Compute lineage sizes
    lineage_counts = Counter(assignments.values())

    # Add lineage and lineage_size columns to DataFrame
    lineages = [assignments.get(seq_id, "") for seq_id in sequence_ids]
    sizes = [lineage_counts.get(assignments.get(seq_id, ""), 0) for seq_id in sequence_ids]
    result_df = df.with_columns(
        [
            pl.Series(name=lineage_column, values=lineages),
            pl.Series(name=lineage_size_column, values=sizes),
        ]
    )

    if verbose:
        n_clusters = len(set(assignments.values()))
        print(f"Found {n_clusters} clusters")

    # Save output if requested
    if output_path is not None:
        _save_dataframe(result_df, output_path, output_format)
        if verbose:
            print(f"Saved results to {output_path}")

    return assignments, result_df

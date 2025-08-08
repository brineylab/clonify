from __future__ import annotations

import random
import string
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import polars as pl
from mnemonic import Mnemonic
from natsort import natsorted

try:
    from clonify._native import NativeInputs, average_linkage_cutoff
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "clonify native extension not built. Install with `pip install .` or `uv pip install .` from repo root."
    ) from e


def _split_mutations(mut: str, delimiter: str) -> List[str]:
    """Split a mutation string into a list of mutation codes.

    Empty strings and ``None`` yield an empty list.

    Args:
        mut (str): Raw mutation string (e.g., "A23C|C45T"). May be empty or
            ``None``.
        delimiter (str): Delimiter separating mutations.

    Returns:
        List[str]: Non-empty mutation codes in their original order.
    """
    if mut is None or mut == "":
        return []
    return [m for m in mut.split(delimiter) if m]


def _compute_likely_allelic_variants(
    df: pl.DataFrame,
    vgene_key: str,
    mutations_key: str,
    allelic_variant_threshold: float,
    min_seqs_for_allelic_variants: int,
    verbose: bool,
) -> Dict[str, List[str]]:
    """Identify mutations that are likely germline allelic variants per V gene.

    For each unique V gene in ``df``, the function counts how frequently each
    mutation occurs across sequences assigned to that V gene. Mutations observed
    in at least ``allelic_variant_threshold * num_sequences_for_v`` sequences are
    flagged as likely allelic variants. If ``verbose`` is ``True``, progress and
    summaries are printed.

    Args:
        df (polars.DataFrame): DataFrame with at least ``vgene_key`` and
            ``mutations_key``; per-row mutations must be lists of strings.
        vgene_key (str): Column name with V gene identifiers.
        mutations_key (str): Column name with lists of mutation strings per
            sequence.
        allelic_variant_threshold (float): Fraction threshold; for example,
            ``0.35`` means a mutation must appear in at least 35% of sequences
            for that V gene.
        min_seqs_for_allelic_variants (int): Minimum sequences required for a V
            gene before allelic variants are computed. V genes with fewer
            sequences yield an empty list.
        verbose (bool): Whether to print progress and per-V gene summaries.

    Returns:
        Dict[str, List[str]]: Mapping from V gene to mutation strings that are
        likely allelic variants for that gene.

    Raises:
        KeyError: If ``vgene_key`` or ``mutations_key`` are missing from ``df``.
    """
    unique_vgenes = df[vgene_key].unique().to_list()
    likely: Dict[str, List[str]] = {}
    if verbose:
        print("- identifying mutations that are likely allelic variants...")
    for v in unique_vgenes:
        v_df = df.filter(pl.col(vgene_key) == v)
        if v_df.shape[0] < min_seqs_for_allelic_variants:
            likely[v] = []
            continue
        allele_threshold = v_df.shape[0] * allelic_variant_threshold
        mut_counts: Counter[str] = Counter()
        for muts in v_df[mutations_key]:
            mut_counts.update(muts)
        likely[v] = [m for m, c in mut_counts.items() if c >= allele_threshold]
    if verbose:
        for v, muts in likely.items():
            if muts:
                print(f"    {v}: {', '.join(natsorted(muts))}")
    return likely


def _encode_group_inputs(
    group_df: pl.DataFrame,
    id_key: str,
    vgene_key: str,
    jgene_key: str,
    cdr3_key: str,
    mut_lists_key: str,
    likely_allelic: Dict[str, List[str]],
) -> Tuple[
    List[str], List[int], List[int], List[int], List[int], List[Tuple[int, List[int]]]
]:
    """Encode grouped inputs into integer IDs and ragged arrays for the native backend.

    Transforms a group of sequences into the compact, integer-encoded
    representation expected by the native clustering implementation. Specifically:

    - Maps V and J gene strings to integer IDs.
    - Flattens per-sequence mutation lists into a single vector with a companion
      ``offsets`` array that delimits original sequence boundaries.
    - Converts likely allelic variants per V gene into integer mutation IDs.

    Args:
        group_df (polars.DataFrame): Group of sequences from the full input
            table.
        id_key (str): Column containing unique sequence identifiers.
        vgene_key (str): Column containing V gene identifiers.
        jgene_key (str): Column containing J gene identifiers.
        cdr3_key (str): Column containing CDR3 amino-acid sequences.
        mut_lists_key (str): Column containing lists of mutation strings per
            sequence (already split).
        likely_allelic (Dict[str, List[str]]): Mapping of V gene to mutation
            strings treated as likely allelic variants (i.e., down-weighted or
            ignored by the native algorithm).

    Returns:
        Tuple[List[str], List[int], List[int], List[int], List[int], List[Tuple[int, List[int]]]]:
            - cdr3_list: CDR3 sequences for each row, in order.
            - v_ids: Integer V gene IDs parallel to ``cdr3_list``.
            - j_ids: Integer J gene IDs parallel to ``cdr3_list``.
            - mut_ids_flat: Flattened mutation IDs across all rows.
            - mut_offsets: Offsets delimiting sequences within ``mut_ids_flat``;
              the mutations for row ``i`` are in ``[mut_offsets[i], mut_offsets[i+1])``.
            - v_allelic: For each unique V gene ID, a pair ``(v_id, allelic_mutation_ids)``.

    Raises:
        KeyError: If required columns are missing from ``group_df``.
    """
    v_values = group_df[vgene_key].to_list()
    j_values = group_df[jgene_key].to_list()
    v_map: Dict[str, int] = {}
    j_map: Dict[str, int] = {}
    v_ids: List[int] = []
    j_ids: List[int] = []
    for v in v_values:
        if v not in v_map:
            v_map[v] = len(v_map)
        v_ids.append(v_map[v])
    for j in j_values:
        if j not in j_map:
            j_map[j] = len(j_map)
        j_ids.append(j_map[j])

    mut_to_id: Dict[str, int] = {}
    mut_ids_flat: List[int] = []
    mut_offsets: List[int] = [0]
    for muts in group_df[mut_lists_key]:
        ids = []
        for m in natsorted(muts):
            if m not in mut_to_id:
                mut_to_id[m] = len(mut_to_id)
            ids.append(mut_to_id[m])
        ids_sorted = sorted(set(ids))
        mut_ids_flat.extend(ids_sorted)
        mut_offsets.append(len(mut_ids_flat))

    v_allelic: List[Tuple[int, List[int]]] = []
    for v_str, v_int in v_map.items():
        allelic = likely_allelic.get(v_str, [])
        int_list = sorted({mut_to_id[m] for m in allelic if m in mut_to_id})
        v_allelic.append((v_int, int_list))

    cdr3_list: List[str] = group_df[cdr3_key].to_list()
    return cdr3_list, v_ids, j_ids, mut_ids_flat, mut_offsets, v_allelic


def _assign_cluster_names(
    labels: List[int], ids: List[str], mnemonic_names: bool
) -> Dict[str, str]:
    """Create human-friendly or random cluster names for label groups.

    Given a cluster label for each sequence ``id``, assigns a stable, per-cluster
    name and returns the mapping from sequence ID to cluster name. When
    ``mnemonic_names`` is ``True``, names are generated using mnemonic English
    word lists; otherwise, names are random alphanumeric strings.

    Args:
        labels (List[int]): Cluster labels, one per sequence (parallel to ``ids``).
        ids (List[str]): Sequence identifiers aligned with ``labels``.
        mnemonic_names (bool): If ``True``, generate mnemonic names; otherwise,
            generate random 16-character alphanumeric names.

    Returns:
        Dict[str, str]: Mapping from sequence ID to assigned cluster name.
    """
    mnemo = Mnemonic("english")
    assign: Dict[str, str] = {}
    cluster_ids = sorted(set(labels))
    if mnemonic_names:
        cluster_names = {
            c: "_".join(mnemo.generate(strength=128).split()[:8]) for c in cluster_ids
        }
    else:
        cluster_names = {
            c: "".join(random.choices(string.ascii_letters + string.digits, k=16))
            for c in cluster_ids
        }
    for sid, lab in zip(ids, labels):
        assign[sid] = cluster_names[lab]
    return assign


def clonify(
    df: pl.DataFrame,
    *,
    distance_cutoff: float = 0.35,
    shared_mutation_bonus: float = 0.35,
    length_penalty_multiplier: float | int = 2.0,
    group_by_v: bool = True,
    group_by_j: bool = True,
    group_by_light_chain_vj: bool = True,
    id_key: str = "sequence_id",
    vgene_key: str = "v_gene",
    jgene_key: str = "j_gene",
    cdr3_key: str = "cdr3",
    mutations_key: str = "v_mutations",
    mutation_delimiter: str = "|",
    ignore_likely_allelic_variants: bool = False,
    allelic_variant_threshold: float = 0.35,
    min_seqs_for_allelic_variants: int = 200,
    mnemonic_names: bool = True,
    n_threads: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, str], pl.DataFrame]:
    """Cluster sequences into clonal lineages and return assignments plus an annotated table.

    Clusters immunoglobulin sequences using a native high-performance (Rust)
    backend. The input table is optionally grouped (e.g., by V/J gene and
    inferred light-chain V/J) before clustering. For each group, sequences are
    encoded and passed to an average-linkage routine.

    If ``ignore_likely_allelic_variants`` is ``True``, mutations that appear at
    high frequency for a given V gene are identified and treated as likely
    germline allelic variants (i.e., down-weighted or ignored during clustering).

    The output mapping assigns each sequence ID to a lineage name. The returned
    DataFrame contains all input columns plus two additional columns:

    - ``lineage``: Assigned lineage name for each sequence.
    - ``lineage_size``: Total number of sequences in that lineage.

    Args:
        df (polars.DataFrame): Input with at least ``id_key``, ``vgene_key``,
            ``jgene_key``, ``cdr3_key``, and ``mutations_key``. If a ``"locus"``
            column is present, only rows with ``"IGH"`` are clustered.
        distance_cutoff (float): Maximum clustering distance threshold. Lower
            values yield tighter clusters.
        shared_mutation_bonus (float): Bonus applied when two sequences share a
            mutation.
        length_penalty_multiplier (float | int): Multiplier controlling the
            penalty for CDR3 length differences.
        group_by_v (bool): Whether to group by V gene before clustering.
        group_by_j (bool): Whether to group by J gene before clustering.
        group_by_light_chain_vj (bool): Additionally group by
            ``f"{vgene_key}:1"`` and ``f"{jgene_key}:1"`` if present (commonly
            used for light-chain V/J annotations).
        id_key (str): Column containing unique sequence identifiers.
        vgene_key (str): Column containing V gene identifiers.
        jgene_key (str): Column containing J gene identifiers.
        cdr3_key (str): Column containing CDR3 amino-acid sequences.
        mutations_key (str): Column containing raw mutation strings per
            sequence.
        mutation_delimiter (str): Delimiter used to split ``mutations_key`` into
            individual mutation codes.
        ignore_likely_allelic_variants (bool): If ``True``, detect high-frequency
            mutations per V gene and treat them as likely allelic variants during
            clustering.
        allelic_variant_threshold (float): Fraction of sequences per V gene
            required for a mutation to be considered a likely allelic variant.
        min_seqs_for_allelic_variants (int): Minimum sequences per V gene before
            allelic variants are computed.
        mnemonic_names (bool): If ``True``, use mnemonic word lists for
            human-readable lineage names; otherwise, generate random
            alphanumeric names.
        n_threads (Optional[int]): Number of threads for the native backend.
            ``None`` lets the backend choose a default.
        verbose (bool): Whether to print progress and grouping information.

    Returns:
        Tuple[Dict[str, str], polars.DataFrame]:
            - assignments: Maps each sequence ID (``id_key``) to its lineage name.
            - df_out: The input DataFrame with ``lineage`` and ``lineage_size``
              columns added.

    Raises:
        ValueError: If ``mutations_key`` is missing from ``df``.
        KeyError: If any other required columns are missing when accessed.
    """
    if mutations_key not in df.columns:
        raise ValueError(f"Missing column: {mutations_key}")
    mut_lists = [
        _split_mutations(m, mutation_delimiter) for m in df[mutations_key].to_list()
    ]
    df = df.with_columns(pl.Series(name="__mut_list__", values=mut_lists))

    if "locus" in df.columns:
        filtered_df = df.filter(pl.col("locus") == "IGH")
    else:
        filtered_df = df

    if ignore_likely_allelic_variants:
        likely_allelic = _compute_likely_allelic_variants(
            filtered_df.select([vgene_key, "__mut_list__"]).rename(
                {"__mut_list__": mutations_key}
            ),
            vgene_key,
            mutations_key,
            allelic_variant_threshold,
            min_seqs_for_allelic_variants,
            verbose,
        )
    else:
        likely_allelic = defaultdict(list)

    group_keys: List[str] = []
    if group_by_v:
        group_keys.append(vgene_key)
    if group_by_j:
        group_keys.append(jgene_key)
    if group_by_light_chain_vj and any(c.endswith(":1") for c in df.columns):
        group_keys.extend([f"{vgene_key}:1", f"{jgene_key}:1"])

    if verbose and group_keys:
        pretty = [
            "V gene"
            if k == vgene_key
            else "J gene"
            if k == jgene_key
            else "Light chain V/J genes"
            for k in group_keys
        ]
        print(f"- grouping by {' and '.join(pretty)}")

    if group_keys:
        grouped = df.group_by(group_keys)
        groups = [g[1] for g in grouped]
    else:
        groups = [df]

    assign_total: Dict[str, str] = {}
    out_rows: List[Tuple[str, str]] = []
    for group_df in groups:
        ids: List[str] = group_df[id_key].to_list()
        if len(ids) == 1:
            if mnemonic_names:
                name = "_".join(Mnemonic("english").generate(strength=128).split()[:8])
            else:
                name = "".join(
                    random.choices(string.ascii_letters + string.digits, k=16)
                )
            assign_total[ids[0]] = name
            out_rows.append((ids[0], name))
            continue

        cdr3_list, v_ids, j_ids, mut_ids_flat, mut_offsets, v_allelic = (
            _encode_group_inputs(
                group_df,
                id_key,
                vgene_key,
                jgene_key,
                cdr3_key,
                "__mut_list__",
                likely_allelic,
            )
        )

        native_inp = NativeInputs(
            cdr3_list,
            v_ids,
            j_ids,
            mut_ids_flat,
            mut_offsets,
            v_allelic,
        )

        labels = average_linkage_cutoff(
            native_inp,
            float(shared_mutation_bonus),
            float(length_penalty_multiplier),
            10.0,
            5.0,
            float(distance_cutoff),
            n_threads,
        )
        label_list = list(labels)  # type: ignore[arg-type]
        assign = _assign_cluster_names(label_list, ids, mnemonic_names)
        assign_total.update(assign)
        out_rows.extend((sid, assign[sid]) for sid in ids)

    lineage_size = Counter(name for _, name in out_rows)
    lineage_col = [assign_total[df[id_key][i]] for i in range(df.shape[0])]
    size_col = [lineage_size[lineage_col[i]] for i in range(df.shape[0])]
    df_out = df.with_columns(
        pl.Series(name="lineage", values=lineage_col),
        pl.Series(name="lineage_size", values=size_col),
    ).drop(["__mut_list__"])

    return assign_total, df_out


__all__ = ["clonify"]

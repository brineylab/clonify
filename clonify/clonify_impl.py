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

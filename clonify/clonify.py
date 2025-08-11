from __future__ import annotations

import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple, Union

import polars as pl
from natsort import natsorted

from .naming import assign_names, generate_cluster_name


# ------------------------------
# Helpers used by the Python (reference) implementation
# Moved to module scope so they are picklable under spawn/forkserver
# ------------------------------
def pairwise_distance(
    s1: object,
    s2: object,
    *,
    shared_mutation_bonus: float = 0.35,
    length_penalty_multiplier: float | int = 2,
    vgene_field: str = "v_gene",
    jgene_field: str = "j_gene",
    cdr3_field: str = "cdr3",
    mutations_field: str = "mutations",
    likely_allelic_variants: Optional[Iterable[object]] = None,
    debug: bool = False,
) -> float:
    from rapidfuzz.distance.Levenshtein import (
        distance as levenshtein_distance,  # type: ignore
    )

    germline_penalty = 0
    if s1[vgene_field] != s2[vgene_field]:
        germline_penalty += 10
    if s1[jgene_field] != s2[jgene_field]:
        germline_penalty += 5

    s1_len = len(s1[cdr3_field])
    s2_len = len(s2[cdr3_field])
    length_penalty = abs(s1_len - s2_len) * float(length_penalty_multiplier)
    length = min(s1_len, s2_len)

    if s1_len == s2_len:
        dist = sum([a != b for a, b in zip(s1[cdr3_field], s2[cdr3_field])])
    else:
        dist = levenshtein_distance(s1[cdr3_field], s2[cdr3_field])

    likely_allelic_variants = likely_allelic_variants or []
    mutation_bonus = (
        len(
            set(s1[mutations_field])
            & set(s2[mutations_field]) - set(likely_allelic_variants)
        )
        * shared_mutation_bonus
    )
    score = germline_penalty + (
        (dist + length_penalty - mutation_bonus) / max(length, 1)
    )
    return max(score, 0.001)


def batch_pairwise_distance(seqs, batches, **kwargs):
    distances = []
    for i1, i2 in batches:
        d = pairwise_distance(seqs[i1], seqs[i2], **kwargs)
        distances.append(d)
    return distances


# ------------------------------
# Native (Rust-backed) implementation
# ------------------------------
try:
    from clonify._native import (
        NativeInputs,
        average_linkage_cutoff,
        average_linkage_cutoff_progressive,
    )
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "clonify native extension not built. Install with `pip install .` or `uv pip install .` from repo root."
    ) from e


def _infer_format_from_extension(file_path: str) -> str:
    _, ext = os.path.splitext(file_path.lower())
    if ext in {".csv"}:
        return "csv"
    if ext in {".tsv", ".tab"}:
        return "tsv"
    if ext in {".parquet", ".pq"}:
        return "parquet"
    if ext in {".jsonl", ".ndjson"}:
        return "ndjson"
    if ext in {".json"}:
        return "json"
    raise ValueError(
        f"Unable to infer file format from extension '{ext}'. Specify input_format explicitly."
    )


def _read_dataframe(
    input_path: str,
    input_format: Optional[str],
    *,
    has_header: bool,
    delimiter: Optional[str],
) -> pl.DataFrame:
    fmt = input_format or _infer_format_from_extension(input_path)
    if fmt == "csv":
        return pl.read_csv(
            input_path, has_header=has_header, separator=delimiter or ","
        )
    if fmt == "tsv":
        return pl.read_csv(
            input_path, has_header=has_header, separator=delimiter or "\t"
        )
    if fmt == "parquet":
        return pl.read_parquet(input_path)
    if fmt == "ndjson":
        return pl.read_ndjson(input_path)
    if fmt == "json":
        return pl.read_json(input_path)
    raise ValueError(f"Unsupported input format: {fmt}")


def _write_dataframe(df: pl.DataFrame, output_path: str) -> None:
    fmt = _infer_format_from_extension(output_path)
    if fmt == "csv":
        df.write_csv(output_path)
        return
    if fmt == "tsv":
        df.write_csv(output_path, separator="\t")
        return
    if fmt == "parquet":
        df.write_parquet(output_path)
        return
    if fmt == "ndjson":
        df.write_ndjson(output_path)
        return
    if fmt == "json":
        df.write_json(output_path)
        return
    raise ValueError(f"Unsupported output format derived from path: {output_path}")


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


def clonify_native(
    df: pl.DataFrame,
    *,
    distance_cutoff: float = 0.35,
    shared_mutation_bonus: float = 0.4,
    length_penalty_multiplier: float | int = 2.0,
    group_by_v: bool = True,
    group_by_j: bool = True,
    group_by_light_chain_vj: bool = True,
    id_key: Optional[str] = None,
    vgene_key: Optional[str] = None,
    jgene_key: Optional[str] = None,
    cdr3_key: Optional[str] = None,
    mutations_key: Optional[str] = None,
    locus_key: Optional[str] = None,
    light_vgene_key: Optional[str] = None,
    light_jgene_key: Optional[str] = None,
    mutation_delimiter: str = "|",
    ignore_likely_allelic_variants: bool = False,
    allelic_variant_threshold: float = 0.35,
    min_seqs_for_allelic_variants: int = 200,
    mnemonic_names: bool = True,
    name_seed: Optional[int] = None,
    n_threads: Optional[int] = None,
    progressive: bool = True,
    verbose: bool = True,
) -> Tuple[Dict[str, str], pl.DataFrame]:
    # Resolve dynamic defaults for keys based on schema
    def _choose(candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    resolved_id_key = (
        id_key or _choose(["sequence_id:0", "sequence_id"]) or "sequence_id"
    )
    resolved_v_key = vgene_key or _choose(["v_gene:0", "v_gene"]) or "v_gene"
    resolved_j_key = jgene_key or _choose(["j_gene:0", "j_gene"]) or "j_gene"
    resolved_cdr3_key = cdr3_key or _choose(["cdr3:0", "cdr3"]) or "cdr3"
    resolved_mut_key = (
        mutations_key or _choose(["v_mutations:0", "v_mutations"]) or "v_mutations"
    )
    resolved_locus_key = locus_key or _choose(["locus:0", "locus"])  # may be None
    resolved_lc_v_key = light_vgene_key or _choose(["v_gene:1"])  # may be None
    resolved_lc_j_key = light_jgene_key or _choose(["j_gene:1"])  # may be None

    # Validate required columns exist
    if resolved_mut_key not in df.columns:
        raise ValueError(f"Missing column: {resolved_mut_key}")
    mut_lists = [
        _split_mutations(m, mutation_delimiter) for m in df[resolved_mut_key].to_list()
    ]
    df = df.with_columns(pl.Series(name="__mut_list__", values=mut_lists))

    if resolved_locus_key is not None:
        filtered_df = df.filter(pl.col(resolved_locus_key) == "IGH")
    else:
        filtered_df = df

    if ignore_likely_allelic_variants:
        likely_allelic = _compute_likely_allelic_variants(
            filtered_df.select([resolved_v_key, "__mut_list__"]).rename(
                {"__mut_list__": resolved_mut_key}
            ),
            resolved_v_key,
            resolved_mut_key,
            allelic_variant_threshold,
            min_seqs_for_allelic_variants,
            verbose,
        )
    else:
        likely_allelic = defaultdict(list)

    group_keys: List[str] = []
    if group_by_v:
        group_keys.append(resolved_v_key)
    if group_by_j:
        group_keys.append(resolved_j_key)
    if (
        group_by_light_chain_vj
        and resolved_lc_v_key is not None
        and resolved_lc_j_key is not None
        and resolved_lc_v_key in df.columns
        and resolved_lc_j_key in df.columns
    ):
        group_keys.extend([resolved_lc_v_key, resolved_lc_j_key])

    if verbose and group_keys:
        pretty = [
            "V gene"
            if k == resolved_v_key
            else "J gene"
            if k == resolved_j_key
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

    # progress bar similar to python backend: updates per V/J group
    if verbose:
        print("- assigning lineages:")
        try:
            from tqdm.auto import tqdm  # type: ignore

            group_iter = tqdm(groups)
        except Exception:
            group_iter = groups
    else:
        group_iter = groups

    import os as _os

    env_prog = _os.environ.get("CLONIFY_PROGRESSIVE", "0") in {
        "1",
        "true",
        "TRUE",
        "yes",
        "YES",
    }
    progressive_effective = progressive or env_prog
    for group_df in group_iter:
        ids: List[str] = group_df[resolved_id_key].to_list()
        if len(ids) == 1:
            name = generate_cluster_name(
                ids, mnemonic_names=mnemonic_names, seed=name_seed
            )
            assign_total[ids[0]] = name
            out_rows.append((ids[0], name))
            continue

        cdr3_list, v_ids, j_ids, mut_ids_flat, mut_offsets, v_allelic = (
            _encode_group_inputs(
                group_df,
                resolved_id_key,
                resolved_v_key,
                resolved_j_key,
                resolved_cdr3_key,
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

        if progressive_effective:
            labels = average_linkage_cutoff_progressive(
                native_inp,
                float(shared_mutation_bonus),
                float(length_penalty_multiplier),
                10.0,
                5.0,
                float(distance_cutoff),
                n_threads,
            )
        else:
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
        assign = assign_names(
            label_list, ids, mnemonic_names=mnemonic_names, seed=name_seed
        )
        assign_total.update(assign)
        out_rows.extend((sid, assign[sid]) for sid in ids)

    lineage_size = Counter(name for _, name in out_rows)
    lineage_col = [assign_total[df[resolved_id_key][i]] for i in range(df.shape[0])]
    size_col = [lineage_size[lineage_col[i]] for i in range(df.shape[0])]
    df_out = df.with_columns(
        pl.Series(name="lineage", values=lineage_col),
        pl.Series(name="lineage_size", values=size_col),
    ).drop(["__mut_list__"])

    return assign_total, df_out


def clonify(
    data: Union[pl.DataFrame, str, Iterable[object]],
    *,
    backend: str = "native",
    # Optional file IO parameters (used when `data` is a file path)
    input_format: Optional[str] = None,
    has_header: bool = True,
    delimiter: Optional[str] = None,
    # Optional output path – if provided, results are written to file
    output_path: Optional[str] = None,
    distance_cutoff: float = 0.35,
    shared_mutation_bonus: float = 0.35,
    length_penalty_multiplier: float | int = 2.0,
    group_by_v: bool = True,
    group_by_j: bool = True,
    group_by_light_chain_vj: bool = True,
    id_key: Optional[str] = None,
    vgene_key: Optional[str] = None,
    jgene_key: Optional[str] = None,
    cdr3_key: Optional[str] = None,
    mutations_key: Optional[str] = None,
    locus_key: Optional[str] = None,
    light_vgene_key: Optional[str] = None,
    light_jgene_key: Optional[str] = None,
    mutation_delimiter: str = "|",
    ignore_likely_allelic_variants: bool = False,
    allelic_variant_threshold: float = 0.35,
    min_seqs_for_allelic_variants: int = 200,
    mnemonic_names: bool = True,
    name_seed: Optional[int] = None,
    n_threads: Optional[int] = None,
    progressive: bool = True,
    verbose: bool = True,
) -> Tuple[Dict[str, str], pl.DataFrame]:
    """Public API: dispatch to native (Rust) or reference Python backend.

    - If backend == "native": expects a Polars DataFrame and uses Rust backend.
    - If backend == "python": accepts a Polars DataFrame, file path, or sequence
      iterable; returns results from reference implementation.
    """
    backend_lower = backend.lower()
    if backend_lower == "python":
        # Prepare input for reference backend
        import tempfile
        from pathlib import Path

        if isinstance(data, pl.DataFrame):
            # Serialize to Parquet and pass the file path
            with tempfile.TemporaryDirectory() as tmpdir:
                parq_path = Path(tmpdir) / "input.parquet"
                data.write_parquet(parq_path)
                df_out = clonify_python(
                    str(parq_path),
                    input_fmt="parquet",
                    distance_cutoff=distance_cutoff,
                    shared_mutation_bonus=shared_mutation_bonus,
                    length_penalty_multiplier=length_penalty_multiplier,
                    group_by_v=group_by_v,
                    group_by_j=group_by_j,
                    group_by_light_chain_vj=group_by_light_chain_vj,
                    id_key=(id_key or "sequence_id"),
                    vgene_key=(vgene_key or "v_gene"),
                    jgene_key=(jgene_key or "j_gene"),
                    cdr3_key=(cdr3_key or "cdr3"),
                    mutations_key=(mutations_key or "v_mutations"),
                    mutation_delimiter=mutation_delimiter,
                    ignore_likely_allelic_variants=ignore_likely_allelic_variants,
                    allelic_variant_threshold=allelic_variant_threshold,
                    min_seqs_for_allelic_variants=min_seqs_for_allelic_variants,
                    mnemonic_names=mnemonic_names,
                    name_seed=name_seed,
                    output_fmt="polars",
                    verbose=verbose,
                )
        elif isinstance(data, str):
            # Map input format for the reference backend
            fmt = input_format or _infer_format_from_extension(data)
            if fmt in {"tsv", "tab"}:
                py_fmt = "airr"
            elif fmt in {"csv", "parquet"}:
                py_fmt = fmt
            else:
                raise ValueError(
                    f"Unsupported input format for python backend: {fmt}. Use csv/tsv/parquet."
                )
            df_out = clonify_python(
                data,
                input_fmt=py_fmt,
                distance_cutoff=distance_cutoff,
                shared_mutation_bonus=shared_mutation_bonus,
                length_penalty_multiplier=length_penalty_multiplier,
                group_by_v=group_by_v,
                group_by_j=group_by_j,
                group_by_light_chain_vj=group_by_light_chain_vj,
                id_key=(id_key or "sequence_id"),
                vgene_key=(vgene_key or "v_gene"),
                jgene_key=(jgene_key or "j_gene"),
                cdr3_key=(cdr3_key or "cdr3"),
                mutations_key=(mutations_key or "v_mutations"),
                mutation_delimiter=mutation_delimiter,
                ignore_likely_allelic_variants=ignore_likely_allelic_variants,
                allelic_variant_threshold=allelic_variant_threshold,
                min_seqs_for_allelic_variants=min_seqs_for_allelic_variants,
                mnemonic_names=mnemonic_names,
                name_seed=name_seed,
                output_fmt="polars",
                verbose=verbose,
            )
        else:
            raise TypeError(
                "Python backend expects a file path or Polars DataFrame (will be serialized)."
            )
        assignments = {
            sid: lin
            for sid, lin in zip(
                df_out[(id_key or "sequence_id")].to_list(),
                df_out["lineage"].to_list(),
            )
        }
        if output_path:
            _write_dataframe(df_out, output_path)
        return assignments, df_out

    # Default to native backend
    # Native backend: allow DataFrame directly, or read from path
    if isinstance(data, str):
        df_in = _read_dataframe(
            data, input_format, has_header=has_header, delimiter=delimiter
        )
    elif isinstance(data, pl.DataFrame):
        df_in = data
    else:
        raise TypeError("Native backend expects a Polars DataFrame or a file path.")

    assignments, df_out = clonify_native(
        df_in,
        distance_cutoff=distance_cutoff,
        shared_mutation_bonus=shared_mutation_bonus,
        length_penalty_multiplier=length_penalty_multiplier,
        group_by_v=group_by_v,
        group_by_j=group_by_j,
        group_by_light_chain_vj=group_by_light_chain_vj,
        id_key=id_key,
        vgene_key=vgene_key,
        jgene_key=jgene_key,
        cdr3_key=cdr3_key,
        mutations_key=mutations_key,
        locus_key=locus_key,
        light_vgene_key=light_vgene_key,
        light_jgene_key=light_jgene_key,
        mutation_delimiter=mutation_delimiter,
        ignore_likely_allelic_variants=ignore_likely_allelic_variants,
        allelic_variant_threshold=allelic_variant_threshold,
        min_seqs_for_allelic_variants=min_seqs_for_allelic_variants,
        mnemonic_names=mnemonic_names,
        name_seed=name_seed,
        n_threads=n_threads,
        progressive=progressive,
        verbose=verbose,
    )
    if output_path:
        _write_dataframe(df_out, output_path)
    return assignments, df_out


# ------------------------------
# Python (reference) implementation
# ------------------------------
def clonify_python(
    sequences: Union[str, Iterable[object]],
    output_path: Optional[str] = None,
    distance_cutoff: float = 0.35,
    shared_mutation_bonus: float = 0.35,
    length_penalty_multiplier: Union[int, float] = 2,
    group_by_v: bool = True,
    group_by_j: bool = True,
    group_by_light_chain_vj: bool = True,
    precluster: bool = False,
    preclustering_threshold: float = 0.65,
    id_key: str = "sequence_id",
    vgene_key: str = "v_gene",
    jgene_key: str = "j_gene",
    cdr3_key: str = "cdr3",
    mutations_key: str = "v_mutations",
    preclustering_key: str = "cdr3",
    mutation_delimiter: str = "|",
    ignore_likely_allelic_variants: bool = False,
    allelic_variant_threshold: float = 0.35,
    min_seqs_for_allelic_variants: int = 200,
    lineage_field: str = "lineage",
    lineage_size_field: str = "lineage_size",
    mnemonic_names: bool = True,
    name_seed: Optional[int] = None,
    input_fmt: str = "airr",
    output_fmt: str = "airr",
    temp_directory: Optional[str] = None,
    return_assignment_dict: bool = False,
    batch_size: int = 100000,
    n_processes: int = None,
    verbose: bool = True,
    concise_logging: bool = False,
) -> Union[dict, pl.DataFrame, object, Iterable[object]]:
    # Local imports to avoid heavy dependency cost unless used
    import itertools
    import multiprocessing as mp
    import os
    from collections import Counter as _Counter

    import fastcluster
    from abutils.core.pair import Pair  # type: ignore
    from abutils.core.sequence import Sequence  # type: ignore
    from abutils.io import (  # type: ignore
        from_polars,
        make_dir,
        read_airr,
        read_csv,
        read_parquet,
        to_polars,
    )
    from abutils.tools.cluster import cluster  # type: ignore
    from abutils.utils.utilities import generate_batches  # type: ignore

    # levenshtein_distance is imported in module-scope helper
    from scipy.cluster.hierarchy import fcluster  # type: ignore
    from tqdm.auto import tqdm  # type: ignore

    # set up file paths
    if output_path is not None:
        output_path = os.path.abspath(output_path)
        make_dir(os.path.dirname(output_path))
    if temp_directory is None:
        if output_path is None:
            temp_directory = "/tmp/.clonify_temp"
        else:
            temp_directory = os.path.join(os.path.dirname(output_path), ".clonify_temp")
    make_dir(temp_directory)

    # process input data
    if isinstance(sequences, str):
        input_fmt = input_fmt.lower()
        if input_fmt in ["fasta", "fastq"]:
            import abstar  # type: ignore  # import here to avoid circular import

            sequences = abstar.run(sequences)
        elif input_fmt == "airr":
            sequences = read_airr(sequences)
        elif input_fmt == "parquet":
            sequences = read_parquet(sequences)
        elif input_fmt == "csv":
            sequences = read_csv(sequences)
        else:
            raise ValueError(f"Invalid input format: {input_fmt}")
    df = to_polars(sequences)
    is_paired = isinstance(sequences[0], Pair)

    # filter DataFrame
    fields = [id_key, vgene_key, jgene_key, cdr3_key, mutations_key]
    if precluster:
        fields.append(preclustering_key)
    if is_paired:
        id_key = f"{id_key}:0"
        filtered_df = df.filter(pl.col("locus:0") == "IGH")
        _fields = [f"{f}:0" for f in fields]
        if group_by_light_chain_vj:
            _fields.append(f"{vgene_key}:1")
            _fields.append(f"{jgene_key}:1")
        filtered_df = filtered_df.select(_fields).rename(lambda c: c.replace(":0", ""))
    else:
        filtered_df = df
        if "locus" in df.columns:
            filtered_df = filtered_df.filter(pl.col("locus") == "IGH")
        filtered_df = filtered_df.select(fields)

    # split mutations string into a list
    mutation_lists = [
        [] if m == "" else m.split(mutation_delimiter)
        for m in filtered_df[mutations_key]
    ]
    filtered_df = filtered_df.with_columns(
        pl.Series(name="mutations", values=mutation_lists),
    )

    # identify mutations associated with potential allelic variants
    unique_vgenes = filtered_df[vgene_key].unique().to_list()
    likely_allelic_variants: Dict[str, List[str]] = {}
    if ignore_likely_allelic_variants:
        if verbose:
            print("- identifying mutations that are likely allelic variants...")
        for v in unique_vgenes:
            v_muts: List[str] = []
            v_df = filtered_df.filter(pl.col(vgene_key) == v)
            if v_df.shape[0] < min_seqs_for_allelic_variants:
                likely_allelic_variants[v] = []
                continue
            allele_threshold = v_df.shape[0] * allelic_variant_threshold
            for muts in v_df["mutations"]:
                v_muts.extend(muts)
            mut_counts = _Counter(v_muts)
            likely_allelic_variants[v] = [
                m for m, c in mut_counts.items() if c >= allele_threshold
            ]
        if verbose:
            for v, muts in likely_allelic_variants.items():
                if muts:
                    print(f"    {v}: {', '.join(natsorted(muts))}")
    else:
        for v in unique_vgenes:
            likely_allelic_variants[v] = []

    # group sequences by V/J genes
    if not group_by_v and not group_by_j:
        group_dfs = [filtered_df]
    else:
        group_by = []
        group_by_list = []
        if group_by_v:
            group_by.append(vgene_key)
            group_by_list.append("V gene")
        if group_by_j:
            group_by.append(jgene_key)
            group_by_list.append("J gene")
        if is_paired and group_by_light_chain_vj:
            group_by.append(f"{vgene_key}:1")
            group_by.append(f"{jgene_key}:1")
            group_by_list.append("Light chain V/J genes")
        if verbose:
            print(f"- grouping by {' and '.join(group_by_list)}")
        grouped = filtered_df.group_by(group_by)
        group_dfs = [g[1] for g in grouped]

    # preclustering
    sequence_groups: List[List[Sequence]] = []
    if precluster:
        if verbose:
            print("- preclustering")
        for group_df in group_dfs:
            seqs = from_polars(group_df, sequence_key=cdr3_key)
            clusters = cluster(
                sequences=seqs,
                seq_key=preclustering_key,
                threshold=preclustering_threshold,
            )
            sequence_groups.extend([c.sequences for c in clusters])
    else:
        for group_df in group_dfs:
            sequence_groups.append(from_polars(group_df, sequence_key=cdr3_key))

    # configure multiprocessing
    n_processes = n_processes or mp.cpu_count()
    use_mp = True
    if n_processes == 1:
        use_mp = False
    if all([len(sg) < 1000 for sg in sequence_groups]):
        use_mp = False
    if use_mp:
        pool = mp.Pool(processes=n_processes)

    # assign lineages
    assign_dict: Dict[str, str] = {}
    assign_kwargs = {
        "shared_mutation_bonus": shared_mutation_bonus,
        "length_penalty_multiplier": length_penalty_multiplier,
        "vgene_field": vgene_key,
        "jgene_field": jgene_key,
        "cdr3_field": cdr3_key,
        "mutations_field": "mutations",
    }

    # progress bar
    if verbose:
        print("- assigning lineages:")
        sequence_groups = tqdm(sequence_groups)

    # clonify
    for seqs in sequence_groups:
        if len(seqs) == 1:
            assign_dict[seqs[0].id] = generate_cluster_name(
                [seqs[0].id], mnemonic_names=mnemonic_names, seed=name_seed
            )
            continue

        distances: List[float] = []
        if use_mp:
            async_results = []
            index_iter = itertools.combinations(list(range(len(seqs))), 2)
            for index_batch in generate_batches(index_iter, batch_size):
                async_results.append(
                    pool.apply_async(
                        batch_pairwise_distance,
                        args=(seqs, index_batch),
                        kwds=assign_kwargs,
                    )
                )
            for ar in async_results:
                distances.extend(ar.get())
        else:
            for s1, s2 in itertools.combinations(seqs, 2):
                distances.append(pairwise_distance(s1, s2, **assign_kwargs))

        linkage_matrix = fastcluster.linkage(
            distances,
            method="average",
            preserve_input=False,
        )
        cluster_list = fcluster(
            linkage_matrix,
            distance_cutoff,
            criterion="distance",
        )

        labels = list(cluster_list)
        ids = [s.id for s in seqs]
        assignments = assign_names(
            labels, ids, mnemonic_names=mnemonic_names, seed=name_seed
        )
        assign_dict.update(assignments)

    if use_mp:
        pool.close()
        pool.join()

    # add the lineage name and size to the sequence DataFrame
    lineage_size_dict = _Counter(assign_dict.values())
    lineages = [assign_dict.get(s, None) for s in df[id_key]]
    lineage_sizes = [lineage_size_dict.get(lineage, None) for lineage in lineages]
    df = df.with_columns(
        pl.Series(name=lineage_field, values=lineages),
        pl.Series(name=lineage_size_field, values=lineage_sizes),
    )

    # output
    if return_assignment_dict:
        return assign_dict
    if output_path is not None:
        if output_fmt.lower() == "airr":
            pl.write_csv(df, output_path, separator="\t")
        elif output_fmt.lower() == "parquet":
            pl.write_parquet(df, output_path)
        elif output_fmt.lower() == "csv":
            pl.write_csv(df, output_path)
        else:
            raise ValueError
    if output_fmt.lower() == "polars":
        return df
    if output_fmt.lower == "pandas":
        return df.to_pandas()
    return from_polars(df)


__all__ = [
    "clonify",
    "clonify_native",
    "clonify_python",
]

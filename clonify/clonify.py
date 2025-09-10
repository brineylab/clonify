from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import abcluster
import pandas as pd
import polars as pl

__all__ = ["run"]


def _load_table_from_path(path: str | Path, delimiter: Optional[str] = None) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file does not exist: {p}")

    ext = p.suffix.lower()
    is_tsv_like = ext in {".tsv", ".tab", ".txt"}
    is_parquet = ext in {".parquet", ".pq"}

    if is_parquet:
        return pl.read_parquet(p)
    sep = delimiter or ("\t" if is_tsv_like else ",")
    return pl.read_csv(p, separator=sep)


def _coerce_mutations(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for x in value:
            if x is None:
                continue
            s = str(x).strip()
            if s:
                out.append(s)
        return out
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        parts = [p.strip() for p in s.split("|")]
        return [p for p in parts if p]
    return [str(value)]


def run(
    data: Path | pd.DataFrame | pl.DataFrame,
    *,
    output: Optional[str | Path] = None,
    junc_col: str = "cdr3",
    v_gene_col: str = "v_gene",
    j_gene_col: str = "j_gene",
    mutations_col: str = "v_mutations",
    delimiter: Optional[str] = None,
    cutoff: float = 0.35,
    mutation_bonus: float = 0.35,
    len_penalty: int = 2,
    epsilon: Optional[float] = None,
    canonical_samples: Optional[int] = None,
) -> Any:
    """Cluster antibody-like records into clonal lineages and add a ``lineage`` column.

    This function converts table rows into ``abcluster.Record`` objects, performs
    clustering with ``abcluster.cluster`` using the supplied options, and
    appends a string lineage label for each input row. When a file path is
    provided, the updated table is also written back to disk.

    Args:
        data (Path | str | pandas.DataFrame | polars.DataFrame):
            Input records. If a path-like, the file is read as CSV/TSV (with
            delimiter inference) or Parquet based on extension. If a DataFrame
            is provided, it must already contain the required columns.
        output (Optional[Path | str]):
            Destination to write the updated table when ``data`` is a path. If
            ``None`` (default), the input file is overwritten in place. Ignored
            when ``data`` is a DataFrame.
        junc_col (str):
            Name of the column containing the junction amino-acid sequence
            (typically the CDR3 AA). Default: ``"cdr3"``.
        v_gene_col (str):
            Name of the column containing the V gene identifier
            (e.g., ``"IGHV1-69*01"``). Default: ``"v_gene"``.
        j_gene_col (str):
            Name of the column containing the J gene identifier. Default:
            ``"j_gene"``.
        mutations_col (str):
            Name of the column containing V-segment mutations. Accepted forms
            per row include:
            - pipe-delimited string, e.g., ``"A23G|C45T|G95A"``
            - sequence of strings, e.g., ``["A23G", "C45T"]`` or
              ``("T10C",)``
            - empty string or ``None`` (interpreted as no mutations)
            Values are normalized internally to ``list[str]``. Default:
            ``"v_mutations"``.
        delimiter (Optional[str]):
            Delimiter override when reading/writing CSV/TSV files. If ``None``,
            it is inferred by extension: tab for ``.tsv``/``.tab``/``.txt``,
            comma otherwise. Ignored for Parquet IO. Default: ``None``.
        cutoff (float):
            Base clustering cutoff threshold that governs sequence similarity in
            ``abcluster``. Default: ``0.35``.
        mutation_bonus (float):
            Per-mutation similarity bonus (mapped to
            ``ClusterOptions.mut_value``). Default: ``0.35``.
        len_penalty (int):
            Penalty applied for length differences (mapped to
            ``ClusterOptions.len_penalty``). Default: ``2``.
        epsilon (Optional[float]):
            Optional neighborhood radius used by the clustering backend. If
            ``None``, the backend default is used. Default: ``None``.
        canonical_samples (Optional[int]):
            Optional number of canonical samples considered within clusters. If
            ``None``, the backend default is used. Default: ``None``.

    Returns:
        polars.DataFrame | pandas.DataFrame: The input table augmented with a
        new ``lineage`` column.

        - If ``data`` is a ``pandas.DataFrame``: returns a pandas DataFrame.
        - Otherwise (polars input or path): returns a polars DataFrame.

    Side Effects:
        When ``data`` is a path, writes the updated table with the new
        ``lineage`` column to ``output`` if provided, otherwise overwrites the
        input file in place.

    Raises:
        FileNotFoundError: If ``data`` is a path and the file does not exist.
        TypeError: If ``data`` is neither a DataFrame nor a path-like.
        ValueError: If any required columns are missing from the input frame.

    Examples:
        Basic usage with a polars DataFrame:

        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "cdr3": ["CARDRSTYYGMDVW", "CARGGGYAMDYW"],
        ...     "v_gene": ["IGHV3-23*01", "IGHV3-23*01"],
        ...     "j_gene": ["IGHJ4*02", "IGHJ4*02"],
        ...     "v_mutations": ["A23G|C45T", ""]
        ... })
        >>> out = run(df)
        >>> "lineage" in out.columns
        True

        With a pandas DataFrame:

        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "cdr3": ["CARDRSTYYGMDVW", "CARGGGYAMDYW"],
        ...     "v_gene": ["IGHV3-23*01", "IGHV3-23*01"],
        ...     "j_gene": ["IGHJ4*02", "IGHJ4*02"],
        ...     "v_mutations": [["A23G", "C45T"], None]
        ... })
        >>> out = run(df)
        >>> isinstance(out, pd.DataFrame)
        True

        Reading from a file path and overwriting in place:

        >>> out = run("ab_recs.tsv")  # doctest: +SKIP

        Reading from a CSV with a custom delimiter and writing to a new file:

        >>> out = run(  # doctest: +SKIP
        ...     "ab_recs.csv",
        ...     output="ab_recs_with_lineage.csv",
        ...     delimiter=";",
        ... )

        Using custom column names and advanced options:

        >>> out = run(  # doctest: +SKIP
        ...     df,
        ...     junc_col="junction_aa",
        ...     v_gene_col="v",
        ...     j_gene_col="j",
        ...     mutations_col="mut_list",
        ...     cutoff=0.42,
        ...     mutation_bonus=0.5,
        ...     len_penalty=3,
        ...     epsilon=0.12,
        ...     canonical_samples=5,
        ... )
    """

    input_is_path = isinstance(data, (str, Path))
    input_is_pandas = isinstance(data, pd.DataFrame)
    input_is_polars = isinstance(data, pl.DataFrame)

    # Load if path-like
    if input_is_path:
        df = _load_table_from_path(data, delimiter)
    elif input_is_pandas:
        df = pl.from_pandas(data)
    elif input_is_polars:
        df = data
    else:
        raise TypeError("data must be a pandas/polars DataFrame or a file path")

    # Ensure columns exist (use appropriate frame)
    cols = (junc_col, v_gene_col, j_gene_col, mutations_col)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Build options
    options = abcluster.ClusterOptions()
    options.cutoff = float(cutoff)
    options.mut_value = float(mutation_bonus)
    options.len_penalty = int(len_penalty)
    if epsilon is not None:
        options.epsilon = float(epsilon)
    if canonical_samples is not None:
        options.canonical_samples = int(canonical_samples)

    # Convert rows to abcluster.Record
    records: List[abcluster.Record] = []
    for junc, v, j, muts in df.select(cols).iter_rows():
        rec = abcluster.Record()
        rec.junc = "" if junc is None else str(junc)
        rec.v_gene = "" if v is None else str(v)
        rec.j_gene = "" if j is None else str(j)
        rec.mutations = _coerce_mutations(muts)
        records.append(rec)

    # Cluster
    labels = abcluster.cluster(records, options)

    # Attach lineage column
    df = df.with_columns(pl.Series("lineage", labels))

    if input_is_path:
        # Determine output path and format
        in_path = Path(data)  # type: ignore[arg-type]
        out_path = Path(output) if output is not None else in_path
        ext = out_path.suffix.lower() or in_path.suffix.lower()
        is_tsv_like = ext in {".tsv", ".tab", ".txt"}
        is_parquet = ext in {".parquet", ".pq"}

        if is_parquet:
            df.write_parquet(out_path)
        else:
            sep = delimiter or ("\t" if is_tsv_like else ",")
            df.write_csv(out_path, separator=sep)

    if input_is_pandas:
        return df.to_pandas()

    # Polars DataFrame input: return a new frame with lineage
    return df

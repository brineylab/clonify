from __future__ import annotations

import json
import os
from typing import Optional

import click
import polars as pl

from .clonify_impl import clonify


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
    raise click.ClickException(
        f"Unable to infer file format from extension '{ext}'. Specify --input-format explicitly."
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
    raise click.ClickException(f"Unsupported input format: {fmt}")


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
    raise click.ClickException(
        f"Unsupported output format derived from path: {output_path}"
    )


@click.command(context_settings={"show_default": True})
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    required=True,
    help="Path to input table (CSV/TSV/Parquet/NDJSON/JSON).",
)
@click.option(
    "--input-format",
    type=click.Choice(
        ["csv", "tsv", "parquet", "ndjson", "json"], case_sensitive=False
    ),
    default=None,
    help="Explicitly set input format; otherwise inferred from file extension.",
)
@click.option(
    "--has-header/--no-header",
    default=True,
    help="Whether the input delimited file has a header row (CSV/TSV only).",
)
@click.option(
    "--delimiter",
    type=str,
    default=None,
    help="Custom delimiter for CSV/TSV. If not provided, uses ',' for CSV and '\t' for TSV.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=str),
    required=False,
    help="Where to write the output table. Format inferred from extension.",
)
@click.option(
    "--assignments-json",
    type=click.Path(dir_okay=False, path_type=str),
    required=False,
    help="Optional path to write lineage assignment mapping as JSON.",
)
# clonify() parameters mirrored below
@click.option(
    "--distance-cutoff",
    type=float,
    default=0.35,
    help="Distance cutoff for clustering.",
)
@click.option(
    "--shared-mutation-bonus",
    type=float,
    default=0.35,
    help="Bonus applied for shared mutations.",
)
@click.option(
    "--length-penalty-multiplier",
    type=float,
    default=2.0,
    help="Multiplier for length penalty term.",
)
@click.option(
    "--group-by-v/--no-group-by-v",
    default=True,
    help="Group sequences by V gene before clustering.",
)
@click.option(
    "--group-by-j/--no-group-by-j",
    default=True,
    help="Group sequences by J gene before clustering.",
)
@click.option(
    "--group-by-light-chain-vj/--no-group-by-light-chain-vj",
    default=True,
    help="Also group by light chain V/J genes when present.",
)
@click.option(
    "--id-key",
    type=str,
    default="sequence_id",
    help="Column name for unique sequence IDs.",
)
@click.option("--vgene-key", type=str, default="v_gene", help="Column name for V gene.")
@click.option("--jgene-key", type=str, default="j_gene", help="Column name for J gene.")
@click.option("--cdr3-key", type=str, default="cdr3", help="Column name for CDR3.")
@click.option(
    "--mutations-key",
    type=str,
    default="v_mutations",
    help="Column name for V-region mutations (delimited string).",
)
@click.option(
    "--mutation-delimiter",
    type=str,
    default="|",
    help="Delimiter used within the mutations column.",
)
@click.option(
    "--ignore-likely-allelic-variants/--consider-likely-allelic-variants",
    default=False,
    help="Ignore mutations that are likely allelic variants.",
)
@click.option(
    "--allelic-variant-threshold",
    type=float,
    default=0.35,
    help="Frequency threshold to consider a mutation as likely allelic variant.",
)
@click.option(
    "--min-seqs-for-allelic-variants",
    type=int,
    default=200,
    help="Minimum sequences per V gene to evaluate allelic variants.",
)
@click.option(
    "--mnemonic-names/--random-names",
    default=True,
    help="Generate human-readable mnemonic lineage names instead of random strings.",
)
@click.option(
    "--n-threads",
    type=int,
    default=None,
    help="Number of threads for native clustering. Defaults to library default.",
)
@click.option(
    "--verbose/--quiet",
    default=True,
    help="Print progress information.",
)
def cli(
    input_path: str,
    input_format: Optional[str],
    has_header: bool,
    delimiter: Optional[str],
    output_path: Optional[str],
    assignments_json: Optional[str],
    distance_cutoff: float,
    shared_mutation_bonus: float,
    length_penalty_multiplier: float,
    group_by_v: bool,
    group_by_j: bool,
    group_by_light_chain_vj: bool,
    id_key: str,
    vgene_key: str,
    jgene_key: str,
    cdr3_key: str,
    mutations_key: str,
    mutation_delimiter: str,
    ignore_likely_allelic_variants: bool,
    allelic_variant_threshold: float,
    min_seqs_for_allelic_variants: int,
    mnemonic_names: bool,
    n_threads: Optional[int],
    verbose: bool,
) -> None:
    """Run clonify on an input table and write results to a file or stdout.

    All parameters mirror the Python API and retain the same defaults.
    """
    try:
        df_in = _read_dataframe(
            input_path,
            input_format,
            has_header=has_header,
            delimiter=delimiter,
        )
    except Exception as exc:  # pragma: no cover - IO handling
        raise click.ClickException(str(exc)) from exc

    try:
        assignments, df_out = clonify(
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
            mutation_delimiter=mutation_delimiter,
            ignore_likely_allelic_variants=ignore_likely_allelic_variants,
            allelic_variant_threshold=allelic_variant_threshold,
            min_seqs_for_allelic_variants=min_seqs_for_allelic_variants,
            mnemonic_names=mnemonic_names,
            n_threads=n_threads,
            verbose=verbose,
        )
    except Exception as exc:  # pragma: no cover - surface native errors nicely
        raise click.ClickException(str(exc)) from exc

    if assignments_json is not None:
        try:
            with open(assignments_json, "w", encoding="utf-8") as f:
                json.dump(assignments, f, ensure_ascii=False)
        except Exception as exc:  # pragma: no cover
            raise click.ClickException(
                f"Failed to write assignments JSON to {assignments_json}: {exc}"
            ) from exc

    if output_path:
        try:
            _write_dataframe(df_out, output_path)
        except Exception as exc:  # pragma: no cover
            raise click.ClickException(
                f"Failed to write output table to {output_path}: {exc}"
            ) from exc
    else:
        # Default to CSV on stdout for human-friendliness
        click.echo(df_out.write_csv())


def main() -> None:
    cli(standalone_mode=True)


__all__ = ["cli", "main"]

"""Command-line interface for clonify."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(
    name="clonify",
    help="High-performance antibody clonotype clustering",
    add_completion=False,
    no_args_is_help=True,
)


@app.command()
def run(
    input_path: Path = typer.Option(
        ..., "--input", "-i", help="Input file (TSV, CSV, or Parquet)"
    ),
    output_path: Path | None = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
    distance_cutoff: float = typer.Option(
        0.35, "--cutoff", "-c", help="Distance cutoff for clustering"
    ),
    shared_mutation_bonus: float = typer.Option(
        0.35, "--mutation-bonus", "-m", help="Bonus for shared mutations"
    ),
    length_penalty: float = typer.Option(
        2.0, "--length-penalty", "-l", help="Penalty per unit length difference"
    ),
    partition_level: str = typer.Option(
        "vj_gene",
        "--partition-level",
        "-p",
        help="Partitioning strategy: v_family, v_gene, or vj_gene",
    ),
    # Paired mode options
    paired: bool = typer.Option(
        False, "--paired", help="Enable paired heavy/light chain mode"
    ),
    heavy_suffix: str = typer.Option(
        ":0", "--heavy-suffix", help="Column suffix for heavy chain (paired mode)"
    ),
    light_suffix: str = typer.Option(
        ":1", "--light-suffix", help="Column suffix for light chain (paired mode)"
    ),
    # Column keys (unpaired mode)
    id_key: str | None = typer.Option(
        None, "--id-key", help="Column name for sequence IDs"
    ),
    vgene_key: str | None = typer.Option(
        None, "--vgene-key", help="Column name for V gene (unpaired mode)"
    ),
    jgene_key: str | None = typer.Option(
        None, "--jgene-key", help="Column name for J gene (unpaired mode)"
    ),
    cdr3_key: str | None = typer.Option(
        None, "--cdr3-key", help="Column name for CDR3/junction (unpaired mode)"
    ),
    mutations_key: str | None = typer.Option(
        None, "--mutations-key", help="Column name for mutations"
    ),
    # Column keys (paired mode)
    heavy_vgene_key: str | None = typer.Option(
        None, "--heavy-vgene-key", help="Column name for heavy chain V gene (paired mode)"
    ),
    heavy_jgene_key: str | None = typer.Option(
        None, "--heavy-jgene-key", help="Column name for heavy chain J gene (paired mode)"
    ),
    heavy_cdr3_key: str | None = typer.Option(
        None, "--heavy-cdr3-key", help="Column name for heavy chain CDR3 (paired mode)"
    ),
    light_vgene_key: str | None = typer.Option(
        None, "--light-vgene-key", help="Column name for light chain V gene (paired mode)"
    ),
    light_jgene_key: str | None = typer.Option(
        None, "--light-jgene-key", help="Column name for light chain J gene (paired mode)"
    ),
    # Output options
    lineage_column: str = typer.Option(
        "lineage", "--lineage-col", help="Name of output lineage column"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress output"),
) -> None:
    """Cluster antibody sequences into clonal lineages.

    Supports both unpaired (heavy chain only) and paired (heavy + light chain) sequences.
    Use --paired to enable paired mode for data with both chains.
    """
    from clonify.api import clonify

    if not input_path.exists():
        typer.echo(f"Error: Input file not found: {input_path}", err=True)
        raise typer.Exit(1)

    # Default output path
    if output_path is None:
        output_path = input_path.with_stem(input_path.stem + "_clustered")

    try:
        assignments, _ = clonify(
            str(input_path),
            output_path=str(output_path),
            distance_cutoff=distance_cutoff,
            shared_mutation_bonus=shared_mutation_bonus,
            length_penalty_multiplier=length_penalty,
            partition_level=partition_level,
            paired=paired,
            heavy_suffix=heavy_suffix,
            light_suffix=light_suffix,
            id_key=id_key,
            vgene_key=vgene_key,
            jgene_key=jgene_key,
            cdr3_key=cdr3_key,
            mutations_key=mutations_key,
            heavy_vgene_key=heavy_vgene_key,
            heavy_jgene_key=heavy_jgene_key,
            heavy_cdr3_key=heavy_cdr3_key,
            light_vgene_key=light_vgene_key,
            light_jgene_key=light_jgene_key,
            lineage_column=lineage_column,
            verbose=not quiet,
        )

        if not quiet:
            n_sequences = len(assignments)
            n_clusters = len(set(assignments.values()))
            typer.echo(f"Clustered {n_sequences} sequences into {n_clusters} lineages")
            typer.echo(f"Results saved to: {output_path}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def version() -> None:
    """Print version information."""
    from clonify import __version__

    typer.echo(f"clonify {__version__}")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()

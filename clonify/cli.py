import sys
from typing import Optional

import click

from clonify.clonify import run as run_clonify


@click.group(help="Clonify command-line interface")
def clonify():
    pass


@clonify.command("run")
@click.argument(
    "data",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str),
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, writable=True, path_type=str),
    default=None,
    show_default=True,
    help="Optional output file path. If omitted, overwrites input file. Ignored when input is a DataFrame.",
)
@click.option(
    "--junc-col",
    default="cdr3",
    show_default=True,
    help="Column with junction amino-acid sequence",
)
@click.option(
    "--v-gene-col",
    default="v_gene",
    show_default=True,
    help="Column with V gene identifier",
)
@click.option(
    "--j-gene-col",
    default="j_gene",
    show_default=True,
    help="Column with J gene identifier",
)
@click.option(
    "--mutations-col",
    default="v_mutations",
    show_default=True,
    help="Column with pipe-delimited mutation strings",
)
@click.option(
    "--delimiter",
    default=None,
    show_default=True,
    help="Delimiter override for CSV/TSV input",
)
@click.option(
    "--cutoff",
    type=float,
    default=0.35,
    show_default=True,
    help="Clustering cutoff",
)
@click.option(
    "--mutation-bonus",
    type=float,
    default=0.35,
    show_default=True,
    help="Mutation bonus value",
)
@click.option(
    "--len-penalty",
    type=int,
    default=2,
    show_default=True,
    help="Length penalty",
)
@click.option(
    "--epsilon",
    type=float,
    default=None,
    show_default=True,
    help="DBSCAN epsilon override",
)
@click.option(
    "--canonical-samples",
    type=int,
    default=None,
    show_default=True,
    help="Number of canonical samples",
)
def run(
    data: str,
    junc_col: str,
    v_gene_col: str,
    j_gene_col: str,
    mutations_col: str,
    delimiter: Optional[str],
    cutoff: float,
    mutation_bonus: float,
    len_penalty: int,
    epsilon: Optional[float],
    canonical_samples: Optional[int],
    output: Optional[str],
):
    """Cluster antibody records and print cluster labels to stdout."""
    try:
        labels = run_clonify(
            data=data,
            junc_col=junc_col,
            v_gene_col=v_gene_col,
            j_gene_col=j_gene_col,
            mutations_col=mutations_col,
            delimiter=delimiter,
            cutoff=cutoff,
            mutation_bonus=mutation_bonus,
            len_penalty=len_penalty,
            epsilon=epsilon,
            canonical_samples=canonical_samples,
            output=output,
        )
    except Exception as exc:
        click.secho(f"Error: {exc}", fg="red", err=True)
        sys.exit(1)

    for label in labels:
        click.echo(label)


if __name__ == "__main__":
    clonify()

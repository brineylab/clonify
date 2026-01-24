"""Type stubs for the native Rust module."""

from typing import Optional

class ClusterParams:
    """Parameters for clustering algorithm."""

    def __init__(
        self,
        cutoff: float = 0.35,
        mut_value: float = 0.35,
        len_penalty: int = 2,
        epsilon: float = 0.001,
        min_center_size: Optional[int] = None,
    ) -> None: ...
    @property
    def cutoff(self) -> float: ...
    @property
    def mut_value(self) -> float: ...
    @property
    def len_penalty(self) -> int: ...
    @property
    def epsilon(self) -> float: ...

def cluster(
    sequence_ids: list[str],
    v_genes: list[str],
    j_genes: list[str],
    cdr3s: list[str],
    mutations: list[list[int]],
    params: Optional[ClusterParams] = None,
) -> list[tuple[str, int]]:
    """Cluster antibody sequences into clonal lineages.

    Args:
        sequence_ids: List of sequence identifiers.
        v_genes: List of V gene names.
        j_genes: List of J gene names.
        cdr3s: List of CDR3/junction amino acid sequences.
        mutations: List of mutation lists (each as list of encoded mutations).
        params: Clustering parameters (optional).

    Returns:
        List of (sequence_id, cluster_id) tuples.
    """
    ...

def encode_mutation(position: int, ref_base: str, alt_base: str) -> int:
    """Encode a mutation from position and bases.

    Args:
        position: Position in the sequence.
        ref_base: Reference nucleotide.
        alt_base: Alternate nucleotide.

    Returns:
        Encoded mutation value.
    """
    ...

def parse_mutations(mutation_str: str) -> list[int]:
    """Parse mutation string in format 'pos:ref>alt|pos:ref>alt|...'.

    Args:
        mutation_str: Mutation string to parse.

    Returns:
        List of encoded mutation values.
    """
    ...

def hamming_distance(s1: str, s2: str) -> int:
    """Compute Hamming distance between two equal-length strings.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Number of positions where characters differ.
    """
    ...

def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein distance between two strings.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Minimum edit distance between strings.
    """
    ...

from __future__ import annotations

import hashlib
import random
import string
from typing import Dict, Iterable, List, Optional

from mnemonic import Mnemonic


def _rng_from_members(member_ids: Iterable[str], seed: Optional[int]) -> random.Random:
    """Create a deterministic RNG from cluster member IDs and an optional seed.

    The RNG seed is derived from SHA-256 over: optional seed (as text) and the
    sorted member IDs. This ensures stable names when the same set of IDs and
    seed are used across different backends.
    """
    hasher = hashlib.sha256()
    if seed is not None:
        hasher.update(str(seed).encode("utf-8"))
    for sid in sorted(member_ids):
        hasher.update(b"\x00")
        hasher.update(sid.encode("utf-8"))
    digest = hasher.digest()
    # Use first 8 bytes to form a 64-bit integer seed
    int_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return random.Random(int_seed)


def generate_cluster_name(
    member_ids: List[str], *, mnemonic_names: bool, seed: Optional[int]
) -> str:
    """Generate a deterministic cluster name for a set of member IDs.

    - If mnemonic_names is True: choose 8 words from the BIP-39 English wordlist
      using a deterministic RNG.
    - Otherwise: generate a 16-character alphanumeric string deterministically.
    """
    rng = _rng_from_members(member_ids, seed)
    if mnemonic_names:
        mnemo = Mnemonic("english")
        words = [rng.choice(mnemo.wordlist) for _ in range(8)]
        return "_".join(words)
    alphabet = string.ascii_letters + string.digits
    return "".join(rng.choice(alphabet) for _ in range(16))


def assign_names(
    labels: List[int], ids: List[str], *, mnemonic_names: bool, seed: Optional[int]
) -> Dict[str, str]:
    """Assign deterministic names to clusters defined by labels and ids.

    Returns a mapping of sequence ID -> cluster name.
    """
    # Build mapping: label -> sorted member IDs
    members: Dict[int, List[str]] = {}
    for sid, lab in zip(ids, labels):
        members.setdefault(lab, []).append(sid)
    for lab in members:
        members[lab].sort()

    # Generate a name for each cluster deterministically
    label_to_name: Dict[int, str] = {
        lab: generate_cluster_name(
            members_ids, mnemonic_names=mnemonic_names, seed=seed
        )
        for lab, members_ids in members.items()
    }

    # Build per-sequence assignment mapping
    assignments: Dict[str, str] = {}
    for sid, lab in zip(ids, labels):
        assignments[sid] = label_to_name[lab]
    return assignments


__all__ = ["generate_cluster_name", "assign_names"]

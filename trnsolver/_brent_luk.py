"""Brent-Luk chess-tournament pairing for parallel Jacobi sweeps.

A Jacobi sweep on an n×n symmetric matrix (n even) consists of n-1 rounds.
In each round, n/2 disjoint pivot pairs (p, q) are rotated in parallel. The
Brent-Luk ordering is a classical round-robin tournament that visits every
unordered (i, j) pair with i < j exactly once across the n-1 rounds.

Architectural purpose in trnsolver: if we permute the rows/cols of the
working matrix D (and the columns of V) before each round, the n/2 pairs
can always be laid out at strided positions (0,1), (2,3), ..., (n-2, n-1).
The NKI rotation kernel then operates on fixed strided indices rather than
dynamic ones, so the traced graph is stable and the kernel compiles once
(cached thereafter). See trnsolver/nki/dispatch.py.

The algorithm here is the standard round-robin schedule: fix slot 0; slots
1..n-1 rotate one position clockwise each round. Pairing for round r is
always (slot 2i, slot 2i+1) after rotation.
"""

from __future__ import annotations

import torch


def brent_luk_permutations(n: int) -> torch.Tensor:
    """Return the n-1 round permutations for Brent-Luk pairing.

    Args:
        n: matrix size, must be even and positive.

    Returns:
        Tensor of shape (n-1, n), dtype int64. Row r is the permutation
        applied before round r: after permuting D by row perm[r] and by
        col perm[r], the pivot pairs for round r live at strided positions
        (0,1), (2,3), ..., (n-2, n-1).

    The pairing (perm[r][2i], perm[r][2i+1]) for i in 0..n/2-1 and r in
    0..n-2 enumerates every unordered pair (a, b) with a < b exactly once.
    """
    if n <= 0 or n % 2 != 0:
        raise ValueError(f"brent_luk_permutations requires even n > 0; got n={n}")

    # Classical round-robin: top row and bottom row of n/2 slots each.
    # Slot 0 is fixed; the other n-1 slots rotate one position each round.
    # At round r, the pairing is (top[i], bot[i]) for i in 0..n/2-1.
    half = n // 2
    perms = torch.empty((n - 1, n), dtype=torch.int64)

    # Initial layout:
    #   top = [0, 1, 2, ..., n/2 - 1]
    #   bot = [n-1, n-2, ..., n/2]
    top = list(range(half))
    bot = list(range(n - 1, half - 1, -1))

    for r in range(n - 1):
        # Flatten to a permutation such that permuted-row 2i is top[i] and
        # permuted-row 2i+1 is bot[i]. That is: perm[2i] = top[i], perm[2i+1] = bot[i].
        flat = [0] * n
        for i in range(half):
            flat[2 * i] = top[i]
            flat[2 * i + 1] = bot[i]
        perms[r] = torch.tensor(flat, dtype=torch.int64)

        # Rotate: pull last of bot to head of top, push top[1] down to bot[0],
        # rotate bot right by one. Slot 0 (top[0]) stays fixed.
        new_top = [top[0], bot[0]] + top[1:-1]
        new_bot = bot[1:] + [top[-1]]
        top, bot = new_top, new_bot

    return perms


def verify_pair_coverage(perms: torch.Tensor) -> bool:
    """Validate that the perms cover every unordered pair exactly once.

    Round r contributes the pairs (perms[r][2i], perms[r][2i+1]) for i in
    0..n/2-1. Every (a, b) with a < b must appear in exactly one round.
    """
    n_rounds, n = perms.shape
    assert n_rounds == n - 1, f"expected n-1 rounds, got {n_rounds} for n={n}"

    seen: set[tuple[int, int]] = set()
    for r in range(n_rounds):
        for i in range(n // 2):
            a = int(perms[r, 2 * i].item())
            b = int(perms[r, 2 * i + 1].item())
            if a > b:
                a, b = b, a
            if (a, b) in seen:
                return False
            seen.add((a, b))

    # Expected total: C(n, 2)
    return len(seen) == n * (n - 1) // 2

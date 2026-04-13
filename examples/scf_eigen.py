"""
SCF eigenvalue problem using trnsolver.

Demonstrates the self-consistent field iteration for quantum chemistry:
    1. Build Fock matrix F from density P
    2. Solve generalized eigenproblem FC = SCε
    3. Build new density P from occupied MO coefficients
    4. Check convergence (energy change, density change)

This is the inner loop that runs before DF-MP2 (trnblas).
The eigenvalue solve is the bottleneck for small-to-medium molecules;
for large molecules the Fock build (GEMM-dominated, trnblas) dominates.

Usage:
    python examples/scf_eigen.py --demo
    python examples/scf_eigen.py --nbasis 50 --nocc 10
"""

import argparse
import time

import torch

import trnsolver


def scf_iteration(
    H_core: torch.Tensor,  # (n, n) — core Hamiltonian
    S: torch.Tensor,  # (n, n) — overlap matrix (SPD)
    eri: torch.Tensor,  # (n, n, n, n) — two-electron integrals (simplified)
    nocc: int,  # Number of occupied orbitals
    max_iter: int = 50,
    tol: float = 1e-8,
) -> dict:
    """Run SCF iteration.

    Returns dict with energy, MO coefficients, orbital energies.
    """
    H_core.shape[0]

    # Initial guess: diagonalize H_core
    eps, C = trnsolver.eigh_generalized(H_core, S)
    C_occ = C[:, :nocc]

    # Density matrix
    P = C_occ @ C_occ.T

    energies = []

    for iteration in range(max_iter):
        # Build Fock matrix F = H_core + J - 0.5*K
        # Simplified: J_μν = Σ_λσ P_λσ (μν|λσ)
        # Full 4-index contraction is O(N^4) — in practice use DF
        J = torch.einsum("ls,mnls->mn", P, eri)
        K = torch.einsum("ls,mlns->mn", P, eri)
        F = H_core + J - 0.5 * K

        # Solve FC = SCε
        eps, C = trnsolver.eigh_generalized(F, S)

        # New density from occupied orbitals
        C_occ = C[:, :nocc]
        P_new = C_occ @ C_occ.T

        # Energy: E = Σ_μν P_μν (H_core_μν + F_μν)
        energy = 0.5 * torch.sum(P_new * (H_core + F)).item()
        energies.append(energy)

        # Check convergence
        dP = torch.linalg.norm(P_new - P).item()
        dE = abs(energies[-1] - energies[-2]) if len(energies) > 1 else float("inf")

        P = P_new

        if dP < tol and dE < tol:
            print(f"  SCF converged in {iteration + 1} iterations")
            break

    return {
        "energy": energy,
        "C": C,
        "eps": eps,
        "P": P,
        "iterations": iteration + 1,
        "converged": dP < tol,
    }


def main():
    parser = argparse.ArgumentParser(description="SCF eigenvalue problem")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--nbasis", type=int, default=10)
    parser.add_argument("--nocc", type=int, default=3)
    args = parser.parse_args()

    if args.demo:
        args.nbasis = 10
        args.nocc = 3

    n = args.nbasis
    nocc = args.nocc

    print("SCF calculation:")
    print(f"  Basis functions: {n}")
    print(f"  Occupied MOs:    {nocc}")
    print(f"  Backend:         {trnsolver.get_backend()}")
    print()

    # Generate synthetic Hamiltonian data
    torch.manual_seed(42)

    # Core Hamiltonian (symmetric)
    H = torch.randn(n, n) * 0.1
    H_core = 0.5 * (H + H.T) - 2.0 * torch.eye(n)  # Negative diagonal for bound states

    # Overlap matrix (SPD, close to identity for orthogonal basis)
    S_raw = torch.randn(n, n) * 0.05
    S = torch.eye(n) + 0.5 * (S_raw + S_raw.T)
    S = S @ S.T + 0.1 * torch.eye(n)  # Ensure SPD

    # Two-electron integrals (simplified: random with symmetry)
    eri = torch.randn(n, n, n, n) * 0.01
    # Enforce (μν|λσ) = (νμ|λσ) = (μν|σλ) = (λσ|μν)
    eri = 0.25 * (eri + eri.permute(1, 0, 2, 3) + eri.permute(0, 1, 3, 2) + eri.permute(2, 3, 0, 1))

    t_start = time.perf_counter()
    result = scf_iteration(H_core, S, eri, nocc)
    t_total = time.perf_counter() - t_start

    print(f"\n  E_SCF = {result['energy']:.10f} (synthetic data)")
    print(f"  Converged: {result['converged']} ({result['iterations']} iterations)")
    print(f"  Orbital energies: {result['eps'][:5].numpy()}")
    print(f"  Total: {t_total:.3f}s")


if __name__ == "__main__":
    main()

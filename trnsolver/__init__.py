"""
trnsolver — Linear solvers and eigendecomposition for AWS Trainium via NKI.

Eigenvalue problems, factorizations, and iterative solvers for scientific
computing on Trainium. Complements trnblas (BLAS) and trnfft (FFT).
Part of the trnsci scientific computing suite.
"""

__version__ = "0.20.0"

# Eigenvalue decomposition
from .eigen import eigh, eigh_generalized, eigh_randomized

# Factorizations
from .factor import (
    cholesky,
    inv_spd,
    inv_sqrt_spd,
    inv_sqrt_spd_ns,
    lu,
    pinv,
    qr,
    schur,
    solve,
    solve_spd,
    svd,
)

# Iterative solvers
from .iterative import (
    block_jacobi_preconditioner,
    cg,
    gmres,
    gmres_ir,
    jacobi_preconditioner,
    ssor_preconditioner,
)

# Backend control
from .nki import HAS_NKI, get_backend, set_backend

# Orthogonalizers (overlap matrices with near-linear dependencies)
from .orthogonalize import Orthogonalizer, orthogonalizer

# Occupied-subspace solver for SCF (Chebyshev-filtered subspace iteration)
from .subspace import SubspaceResult, SubspaceState, occupied_subspace

__all__ = [
    # Eigen
    "eigh",
    "eigh_generalized",
    "eigh_randomized",
    "occupied_subspace",
    "SubspaceState",
    "SubspaceResult",
    # Factorization
    "cholesky",
    "lu",
    "qr",
    "solve",
    "solve_spd",
    "inv_spd",
    "inv_sqrt_spd",
    "inv_sqrt_spd_ns",
    "pinv",
    "schur",
    "svd",
    # Orthogonalizers
    "orthogonalizer",
    "Orthogonalizer",
    # Iterative
    "cg",
    "gmres",
    "gmres_ir",
    "jacobi_preconditioner",
    "block_jacobi_preconditioner",
    "ssor_preconditioner",
    # Backend
    "HAS_NKI",
    "set_backend",
    "get_backend",
]

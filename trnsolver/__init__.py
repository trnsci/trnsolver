"""
trnsolver — Linear solvers and eigendecomposition for AWS Trainium via NKI.

Eigenvalue problems, factorizations, and iterative solvers for scientific
computing on Trainium. Complements trnblas (BLAS) and trnfft (FFT).
Part of the trnsci scientific computing suite.
"""

__version__ = "0.4.0"

# Eigenvalue decomposition
from .eigen import eigh, eigh_generalized

# Factorizations
from .factor import cholesky, inv_spd, inv_sqrt_spd, inv_sqrt_spd_ns, lu, qr, solve, solve_spd

# Iterative solvers
from .iterative import cg, gmres, jacobi_preconditioner

# Backend control
from .nki import HAS_NKI, get_backend, set_backend

__all__ = [
    # Eigen
    "eigh",
    "eigh_generalized",
    # Factorization
    "cholesky",
    "lu",
    "qr",
    "solve",
    "solve_spd",
    "inv_spd",
    "inv_sqrt_spd",
    "inv_sqrt_spd_ns",
    # Iterative
    "cg",
    "gmres",
    "jacobi_preconditioner",
    # Backend
    "HAS_NKI",
    "set_backend",
    "get_backend",
]

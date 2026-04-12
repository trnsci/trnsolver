"""
trnsolver — Linear solvers and eigendecomposition for AWS Trainium via NKI.

Eigenvalue problems, factorizations, and iterative solvers for scientific
computing on Trainium. Complements trnblas (BLAS) and trnfft (FFT).
Part of the trn-* scientific computing suite by Playground Logic.
"""

__version__ = "0.1.1"

# Eigenvalue decomposition
from .eigen import eigh, eigh_generalized

# Factorizations
from .factor import cholesky, lu, qr, solve, solve_spd, inv_spd, inv_sqrt_spd

# Iterative solvers
from .iterative import cg, gmres

# Backend control
from .nki import HAS_NKI, set_backend, get_backend

__all__ = [
    # Eigen
    "eigh", "eigh_generalized",
    # Factorization
    "cholesky", "lu", "qr", "solve", "solve_spd", "inv_spd", "inv_sqrt_spd",
    # Iterative
    "cg", "gmres",
    # Backend
    "HAS_NKI", "set_backend", "get_backend",
]

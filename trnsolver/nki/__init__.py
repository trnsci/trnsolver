"""NKI kernel dispatch for Trainium solver acceleration."""

from .dispatch import HAS_NKI, set_backend, get_backend, _use_nki

__all__ = ["HAS_NKI", "set_backend", "get_backend", "_use_nki"]

"""NKI kernel dispatch for Trainium solver acceleration."""

from .dispatch import HAS_NKI, set_backend, get_backend, _use_nki, _REQUIRE_NKI

__all__ = ["HAS_NKI", "set_backend", "get_backend", "_use_nki", "_REQUIRE_NKI"]

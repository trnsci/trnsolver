"""NKI kernel dispatch for Trainium solver acceleration."""

from .dispatch import (
    _REQUIRE_NKI,
    HAS_NKI,
    _use_nki,
    _use_simulator,
    get_backend,
    set_backend,
)

__all__ = [
    "HAS_NKI",
    "set_backend",
    "get_backend",
    "_use_nki",
    "_use_simulator",
    "_REQUIRE_NKI",
]

"""MLIR <-> e-graph <-> MLIR round-trip."""

from .ir import Module, canonicalize, structurally_equal
from .parser import ParseError, parse_module
from .printer import print_module
from .roundtrip import RoundTripResult, roundtrip, roundtrip_text

__all__ = [
    "Module",
    "ParseError",
    "RoundTripResult",
    "canonicalize",
    "parse_module",
    "print_module",
    "roundtrip",
    "roundtrip_text",
    "structurally_equal",
]

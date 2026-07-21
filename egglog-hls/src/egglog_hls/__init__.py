"""egglog-hls: the DialEgg track of EgraphHLS.

A correctness-first MLIR <-> e-graph <-> MLIR round-trip. No transforms:
a program goes in and comes back out unchanged (up to SSA value-numbering),
verified. This is the foundation the rewrite/cost-function work builds on.
"""

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

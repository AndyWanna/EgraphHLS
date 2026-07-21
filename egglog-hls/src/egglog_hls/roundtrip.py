"""Orchestrate and verify the full round-trip.

    MLIR text --parse--> IR --to_egraph--> e-graph --from_egraph--> IR --print--> MLIR text

Verification is two-pronged:
  1. structural identity: the input and the round-tripped output are equal up
     to SSA value-numbering (the notion of "unchanged" MLIR's own printer uses);
  2. fixpoint: parsing the printed output and round-tripping it again is stable.
"""

from __future__ import annotations

from dataclasses import dataclass

from .egraph_bridge import from_egraph, to_egraph
from .ir import Module, canonicalize, diff_summary, structurally_equal
from .parser import parse_module
from .printer import print_module


@dataclass
class RoundTripResult:
    input_text: str
    output_text: str
    identity_ok: bool
    fixpoint_ok: bool
    diff: str

    @property
    def ok(self) -> bool:
        return self.identity_ok and self.fixpoint_ok


def roundtrip(module: Module) -> Module:
    """Push a module through a real e-graph and pull it back out."""
    egraph, term = to_egraph(module)
    return from_egraph(egraph, term)


def roundtrip_text(text: str) -> RoundTripResult:
    source = parse_module(text)
    result = roundtrip(source)

    identity_ok = structurally_equal(source, result)
    # Fixpoint: printing then reparsing the result must not drift.
    reparsed = parse_module(print_module(result))
    fixpoint_ok = structurally_equal(result, reparsed)

    diff = "(identical)" if identity_ok else diff_summary(source, result)
    output_text = print_module(canonicalize(result))
    return RoundTripResult(text, output_text, identity_ok, fixpoint_ok, diff)

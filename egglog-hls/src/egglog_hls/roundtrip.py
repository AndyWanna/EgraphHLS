"""Run the round-trip and check it: identity up to SSA numbering, plus parse/print fixpoint."""

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
    """Push a module through an e-graph and extract it back."""
    egraph, term = to_egraph(module)
    return from_egraph(egraph, term)


def roundtrip_text(text: str) -> RoundTripResult:
    source = parse_module(text)
    result = roundtrip(source)

    identity_ok = structurally_equal(source, result)
    # canonicalize first: raw names may be packed-result refs ("x#0")
    reparsed = parse_module(print_module(canonicalize(result)))
    fixpoint_ok = structurally_equal(result, reparsed)

    diff = "(identical)" if identity_ok else diff_summary(source, result)
    output_text = print_module(canonicalize(result))
    return RoundTripResult(text, output_text, identity_ok, fixpoint_ok, diff)

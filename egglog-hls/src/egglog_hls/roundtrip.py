"""Round-trip pipeline: MLIR -> egglog e-graph -> MLIR.

First milestone is the identity round-trip with NO rewrites applied: lower MLIR
into egglog, immediately extract it back, and confirm the result is equivalent to
the input. Rewrite rules and cost functions come later.

Everything here is a stub until egglog + DialEgg are stood up locally.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EGraph:
    """Placeholder for the egglog e-graph handle.

    Will wrap an actual `egglog.EGraph` once DialEgg's MLIR modeling is wired in.
    """

    # The serialized/loaded egglog program, terms, and the root expression(s)
    # corresponding to the MLIR module. Shape TBD by DialEgg's representation.
    payload: object = None


def lower(mlir_text: str) -> EGraph:
    """MLIR module text -> egglog e-graph.

    Uses DialEgg to model MLIR operations, types, and attributes as egglog terms.
    """
    raise NotImplementedError(
        "lower(): wire up DialEgg's MLIR->egglog modeling. See egglog-hls/README.md."
    )


def raise_(egraph: EGraph) -> str:
    """egglog e-graph -> MLIR module text.

    Extracts a term from the e-graph and renders it back to MLIR. With no rewrites
    applied, this should reproduce the lowered input.
    """
    raise NotImplementedError(
        "raise_(): extract from the e-graph back to MLIR text. See egglog-hls/README.md."
    )


def round_trip(mlir_text: str) -> str:
    """Lower MLIR into the e-graph and immediately raise it back out (no rewrites)."""
    return raise_(lower(mlir_text))

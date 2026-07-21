"""In-memory MLIR IR model for the round-trip pipeline.

The model is intentionally small and dialect-light: an ``Op`` carries the
handful of things needed to reprint MLIR exactly (name, SSA results/operands,
result types, attributes, and nested regions). Op-specific bits that are *not*
SSA values (a memref type on a load, the loop kind, etc.) live in ``meta``.

Invariant that the rest of the pipeline relies on: the only places an SSA
*value name* ever appears are ``Op.results``, ``Op.operands`` and
``Region.args``. ``meta`` never holds a value name. That is what lets
:func:`canonicalize` renumber values uniformly without op-specific knowledge.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from itertools import count


@dataclass
class Region:
    """A single-block region. ``args``/``arg_types`` are the block arguments
    (for scf.for that is the induction variable followed by the iter_args)."""

    args: list[str] = field(default_factory=list)
    arg_types: list[str] = field(default_factory=list)
    ops: list["Op"] = field(default_factory=list)


@dataclass
class Op:
    name: str
    results: list[str] = field(default_factory=list)
    result_types: list[str] = field(default_factory=list)
    operands: list[str] = field(default_factory=list)
    attributes: dict[str, str] = field(default_factory=dict)
    regions: list[Region] = field(default_factory=list)
    meta: dict[str, object] = field(default_factory=dict)


@dataclass
class Func:
    name: str
    arg_names: list[str] = field(default_factory=list)
    arg_types: list[str] = field(default_factory=list)
    result_types: list[str] = field(default_factory=list)
    body: Region = field(default_factory=Region)


@dataclass
class Module:
    funcs: list[Func] = field(default_factory=list)


# --------------------------------------------------------------------------
# Canonicalization
# --------------------------------------------------------------------------
# MLIR's own AsmPrinter renumbers SSA values on every print (%0, %1, ...), so
# "the program came back unchanged" means *structurally identical after
# value-numbering*, not byte-identical text. We define that canonical form here
# and compare canonical modules with plain dataclass equality.


def _rename_region(region: Region, mapping: dict[str, str], counter) -> None:
    # Block args are definitions: assign fresh names first, in order.
    region.args = [_define(a, mapping, counter) for a in region.args]
    for op in region.ops:
        op.operands = [mapping.get(o, o) for o in op.operands]
        op.results = [_define(r, mapping, counter) for r in op.results]
        for sub in op.regions:
            _rename_region(sub, mapping, counter)


def _define(old: str, mapping: dict[str, str], counter) -> str:
    new = str(next(counter))
    mapping[old] = new
    return new


def canonicalize(module: Module) -> Module:
    """Return a deep copy of ``module`` with every SSA value renamed to a
    positional, definition-order name (``"0"``, ``"1"`` ...), per function."""
    out = copy.deepcopy(module)
    for func in out.funcs:
        mapping: dict[str, str] = {}
        counter = count()
        func.arg_names = [_define(a, mapping, counter) for a in func.arg_names]
        _rename_region(func.body, mapping, counter)
    return out


def structurally_equal(a: Module, b: Module) -> bool:
    """True when two modules are identical up to SSA value-numbering."""
    return canonicalize(a) == canonicalize(b)


def diff_summary(a: Module, b: Module) -> str:
    """Human-readable first divergence between two canonical modules, for
    verification failure messages."""
    ca, cb = canonicalize(a), canonicalize(b)
    if ca == cb:
        return "(identical)"
    from .printer import print_module  # local import to avoid a cycle

    la = print_module(ca).splitlines()
    lb = print_module(cb).splitlines()
    for idx, (x, y) in enumerate(zip(la, lb)):
        if x != y:
            return f"first divergence at line {idx + 1}:\n  in : {x.strip()}\n  out: {y.strip()}"
    if len(la) != len(lb):
        return f"different length: {len(la)} vs {len(lb)} lines"
    return "(modules differ but no line-level diff found)"

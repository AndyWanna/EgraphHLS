"""In-memory MLIR IR.

SSA value names only appear in Op.results, Op.operands and Region.args (never in meta),
so canonicalize can renumber them without op-specific knowledge.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from itertools import count


@dataclass
class Region:
    """Single-block region; args are block args (scf.for: induction var, then iter_args)."""

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
    # trailing "attributes {...}", kept verbatim
    attributes_raw: str = ""


@dataclass
class Verbatim:
    """Module-scope op kept as raw text (llvm.mlir.global, llvm.func)."""

    text: str


@dataclass
class Module:
    # functions and verbatim ops, in source order
    funcs: list[Func | Verbatim] = field(default_factory=list)
    # text between "module" and "{"
    attributes_raw: str = ""


# MLIR renumbers SSA values on every print, so "unchanged" means equal after
# value-numbering, not byte-identical.


def _rename_region(region: Region, mapping: dict[str, str], counter) -> None:
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
    """Deep copy with SSA values renamed "0", "1", ... in definition order, per function."""
    out = copy.deepcopy(module)
    for func in out.funcs:
        if isinstance(func, Verbatim):
            continue
        mapping: dict[str, str] = {}
        counter = count()
        func.arg_names = [_define(a, mapping, counter) for a in func.arg_names]
        _rename_region(func.body, mapping, counter)
    return out


def structurally_equal(a: Module, b: Module) -> bool:
    """Equal up to SSA value-numbering."""
    return canonicalize(a) == canonicalize(b)


def diff_summary(a: Module, b: Module) -> str:
    """First line where the two canonical modules differ."""
    ca, cb = canonicalize(a), canonicalize(b)
    if ca == cb:
        return "(identical)"
    from .printer import print_module  # avoid import cycle

    la = print_module(ca).splitlines()
    lb = print_module(cb).splitlines()
    for idx, (x, y) in enumerate(zip(la, lb)):
        if x != y:
            return f"first divergence at line {idx + 1}:\n  in : {x.strip()}\n  out: {y.strip()}"
    if len(la) != len(lb):
        return f"different length: {len(la)} vs {len(lb)} lines"
    return "(modules differ but no line-level diff found)"

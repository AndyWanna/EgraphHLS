"""Approximate FPGA cost per op for extraction (LUT-equivalents, DSP weighted).
The numbers are a starting point to calibrate against Vitis HLS reports."""

from __future__ import annotations

import math

from egglog import DagCostModel
from egglog.deconstruct import get_callable_args, get_callable_fn, get_literal_value

from .egraph_dataflow import iconst, p1, p2, p3

DSP = 100  # LUT-equivalents per DSP slice


def width(ty: str) -> int:
    if ty == "index":
        return 64
    digits = "".join(ch for ch in ty if ch.isdigit())
    return int(digits) if digits else 32


def op_cost(name: str, ty: str, const_rhs: bool) -> int:
    w = width(ty)
    op = name.split(".", 1)[1]
    if op in ("shli", "shrui", "shrsi"):
        return 1 if const_rhs else w * max(1, int(math.log2(w)))
    if op == "muli":
        return DSP * math.ceil(w / 18) ** 2
    if op in ("divsi", "divui", "remsi", "remui", "ceildivsi", "floordivsi"):
        return w * w
    if op in ("index_cast", "index_castui", "extsi", "extui", "trunci", "bitcast"):
        return 0
    if op in ("addf", "subf"):
        return 4 * w + 2 * DSP
    if op == "mulf":
        return w + (3 if w <= 32 else 11) * DSP
    if op == "divf" or name.startswith("math."):
        return 40 * w
    if op in ("sitofp", "uitofp", "fptosi", "fptoui", "extf", "truncf"):
        return 4 * w
    return w


def _marginal(egraph, expr) -> int:
    fn = get_callable_fn(expr)
    if fn not in (p1, p2, p3):
        return 0
    args = get_callable_args(expr)
    name, ty = get_literal_value(args[0]), get_literal_value(args[-1])
    const_rhs = fn is p2 and get_callable_fn(args[4]) is iconst
    return op_cost(name, ty, const_rhs)


hls_cost = DagCostModel(marginal_cost=_marginal, identity=0)

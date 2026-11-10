"""MLIR <-> e-graph boundary on top of egglog.

Every op is one mk_op term with its results, operands, attributes and regions. Each term
embeds its SSA result names, so hash-consing never merges distinct ops.
"""

from __future__ import annotations

import json

from egglog import EGraph, Expr, String, Vec, function

from .ir import Func, Module, Op, Region, Verbatim


# --- e-graph sorts -------------------------------------------------------
class OpTerm(Expr):
    pass


class RegionTerm(Expr):
    pass


class FuncTerm(Expr):
    pass


class ModuleTerm(Expr):
    pass


@function
def mk_op(
    name: String,
    results: Vec[String],
    result_types: Vec[String],
    operands: Vec[String],
    attr_keys: Vec[String],
    attr_vals: Vec[String],
    meta_json: String,
    regions: Vec[RegionTerm],
) -> OpTerm: ...


@function
def mk_region(
    args: Vec[String], arg_types: Vec[String], ops: Vec[OpTerm]
) -> RegionTerm: ...


@function
def mk_func(
    name: String,
    arg_names: Vec[String],
    arg_types: Vec[String],
    result_types: Vec[String],
    body: RegionTerm,
    attrs_raw: String,
) -> FuncTerm: ...


@function
def mk_verbatim(text: String) -> FuncTerm: ...


@function
def mk_module(funcs: Vec[FuncTerm], attrs_raw: String) -> ModuleTerm: ...


# --- Vec helpers ---------------------------------------------------------
def _svec(items: list[str]) -> Vec[String]:
    return Vec[String](*[String(s) for s in items]) if items else Vec[String].empty()


def _vec(elem_type, items: list):
    return Vec[elem_type](*items) if items else Vec[elem_type].empty()


# --- encode: IR -> term --------------------------------------------------
def _encode_op(op: Op) -> OpTerm:
    keys = sorted(op.attributes)
    return mk_op(
        String(op.name),
        _svec(op.results),
        _svec(op.result_types),
        _svec(op.operands),
        _svec(keys),
        _svec([op.attributes[k] for k in keys]),
        String(json.dumps(op.meta, sort_keys=True)),
        _vec(RegionTerm, [_encode_region(r) for r in op.regions]),
    )


def _encode_region(region: Region) -> RegionTerm:
    return mk_region(
        _svec(region.args),
        _svec(region.arg_types),
        _vec(OpTerm, [_encode_op(o) for o in region.ops]),
    )


def _encode_func(func: Func) -> FuncTerm:
    return mk_func(
        String(func.name),
        _svec(func.arg_names),
        _svec(func.arg_types),
        _svec(func.result_types),
        _encode_region(func.body),
        String(func.attributes_raw),
    )


def _encode_module(module: Module) -> ModuleTerm:
    return mk_module(
        _vec(FuncTerm, [mk_verbatim(String(f.text)) if isinstance(f, Verbatim)
                        else _encode_func(f) for f in module.funcs]),
        String(module.attributes_raw),
    )


def to_egraph(module: Module) -> tuple[EGraph, ModuleTerm]:
    """Insert module into a fresh EGraph; returns the graph and the root term."""
    egraph = EGraph()
    term = _encode_module(module)
    egraph.register(term)
    return egraph, term


# --- decode: extracted term -> IR ---------------------------------------
def _callable_name(callable_) -> str:
    """Constructor name of a decl callable (v13: .ident.name; v11: .name / .class_name)."""
    ident = getattr(callable_, "ident", None)
    if ident is not None:
        return ident.name
    return getattr(callable_, "name", None) or getattr(callable_, "class_name")


def _walk(typed_expr):
    """Extracted term -> strings, lists (Vec) and (constructor_name, children) tuples."""
    expr = typed_expr.expr
    if hasattr(expr, "value") and not hasattr(expr, "callable"):  # LitDecl
        return expr.value
    name = _callable_name(expr.callable)
    children = [_walk(a) for a in expr.args]
    if name == "Vec":
        return children
    return (name, children)


def _decode_op(node) -> Op:
    _, (name, results, rtypes, operands, akeys, avals, meta_json, regions) = node
    return Op(
        name=name,
        results=list(results),
        result_types=list(rtypes),
        operands=list(operands),
        attributes=dict(zip(akeys, avals)),
        regions=[_decode_region(r) for r in regions],
        meta=json.loads(meta_json),
    )


def _decode_region(node) -> Region:
    _, (args, arg_types, ops) = node
    return Region(
        args=list(args),
        arg_types=list(arg_types),
        ops=[_decode_op(o) for o in ops],
    )


def _decode_func(node) -> Func:
    _, (name, arg_names, arg_types, result_types, body, attrs_raw) = node
    return Func(
        name=name,
        arg_names=list(arg_names),
        arg_types=list(arg_types),
        result_types=list(result_types),
        body=_decode_region(body),
        attributes_raw=attrs_raw,
    )


def from_egraph(egraph: EGraph, term: ModuleTerm) -> Module:
    extracted = egraph.extract(term)
    node = _walk(extracted.__egg_typed_expr__)
    _, (funcs, attrs_raw) = node
    return Module(funcs=[Verbatim(f[1][0]) if f[0] == "mk_verbatim" else _decode_func(f)
                         for f in funcs], attributes_raw=attrs_raw)

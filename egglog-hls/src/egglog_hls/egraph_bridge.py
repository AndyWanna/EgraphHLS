"""The MLIR <-> e-graph boundary, built on the real ``egglog`` engine.

Encoding is generic and dialect-agnostic (the DialEgg philosophy): every op
becomes one ``mk_op`` term carrying its results, operands, attributes and
nested regions. Because each op term embeds its own SSA result names, the
engine's hash-consing never merges two syntactically distinct ops, so the
round-trip is lossless.

``to_egraph`` builds the term and inserts it into a fresh ``EGraph``;
``from_egraph`` extracts it back and walks the extracted term
(``__egg_typed_expr__``) into the IR model. With no rewrite rules registered
the extracted term equals the inserted one, which is the identity round-trip.
The same extraction path is what will lower a *rewritten* e-graph back to MLIR
once rules are added.
"""

from __future__ import annotations

import json

from egglog import EGraph, Expr, String, Vec, function

from .ir import Func, Module, Op, Region


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
        _vec(FuncTerm, [_encode_func(f) for f in module.funcs]),
        String(module.attributes_raw),
    )


def to_egraph(module: Module) -> tuple[EGraph, ModuleTerm]:
    """Insert ``module`` into a fresh EGraph. Returns the graph and the root
    term handle (needed to extract it back)."""
    egraph = EGraph()
    term = _encode_module(module)
    egraph.register(term)
    return egraph, term


# --- decode: extracted term -> IR ---------------------------------------
def _callable_name(callable_) -> str:
    """Constructor/function name of a decl callable, across egglog versions.
    v13: FunctionRef/InitRef carry ``.ident.name``; v11: FunctionRef has
    ``.name`` and InitRef has ``.class_name``."""
    ident = getattr(callable_, "ident", None)
    if ident is not None:
        return ident.name
    return getattr(callable_, "name", None) or getattr(callable_, "class_name")


def _walk(typed_expr):
    """Turn an extracted egglog TypedExprDecl into plain Python: string leaves
    stay strings, ``Vec`` becomes a list, every other constructor becomes a
    ``(constructor_name, [children...])`` tuple."""
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
    return Module(funcs=[_decode_func(f) for f in funcs], attributes_raw=attrs_raw)

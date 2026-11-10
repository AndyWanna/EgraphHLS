"""Dataflow encoding: pure ops are name-free terms, so equal values share an
e-class and rewrites can match across ops. Effectful ops stay ordered."""

from __future__ import annotations

import itertools
import json

from egglog import EGraph, Expr, String, StringLike, Vec, function, i64, i64Like

from .ir import Func, Module, Op, Region, Verbatim


class Val(Expr):
    pass


class Stmt(Expr):
    pass


class Reg(Expr):
    pass


class Top(Expr):
    pass


class Mod(Expr):
    pass


@function
def arg(region: i64Like, idx: i64Like, ty: StringLike) -> Val: ...
@function
def res(stmt: i64Like, idx: i64Like, ty: StringLike) -> Val: ...
@function
def iconst(v: i64Like, ty: StringLike) -> Val: ...
@function
def const(v: StringLike, ty: StringLike) -> Val: ...
@function
def p1(name: StringLike, extra: StringLike, a: Val, ty: StringLike) -> Val: ...
@function
def p2(name: StringLike, extra: StringLike, a: Val, b: Val, ty: StringLike) -> Val: ...
@function
def p3(name: StringLike, extra: StringLike, a: Val, b: Val, c: Val, ty: StringLike) -> Val: ...
@function
def stmt(id: i64Like, name: StringLike, rtypes: Vec[String], operands: Vec[Val],
         extra: StringLike, regions: Vec[Reg]) -> Stmt: ...
@function
def reg(id: i64Like, arg_types: Vec[String], stmts: Vec[Stmt]) -> Reg: ...
@function
def fn(name: StringLike, arg_types: Vec[String], result_types: Vec[String],
       body: Reg, attrs: StringLike) -> Top: ...
@function
def verbatim(text: StringLike) -> Top: ...
@function
def module(items: Vec[Top], attrs: StringLike) -> Mod: ...


PURE = {1: p1, 2: p2, 3: p3}
# Can trap, so never speculated out of the region that uses them.
TRAPPING = ("div", "rem")
INT_TYPES = ("index", "i")


def _svec(xs):
    return Vec[String](*map(String, xs)) if xs else Vec[String].empty()


def _vec(t, xs):
    return Vec[t](*xs) if xs else Vec[t].empty()


def _extra(op: Op) -> str:
    if not op.attributes and not op.meta:
        return ""
    return json.dumps({"a": op.attributes, "m": op.meta}, sort_keys=True)


def _is_pure(op: Op) -> bool:
    return (op.name.startswith(("arith.", "math.")) and len(op.results) == 1
            and not op.regions and "parts" not in op.meta
            and (op.name == "arith.constant" or len(op.operands) in PURE))


def _const(op: Op) -> Val:
    v, ty = op.attributes["value"], op.result_types[0]
    if ty.startswith(INT_TYPES) and v.lstrip("-").isdigit():
        return iconst(int(v), ty)
    return const(v, ty)


class _Encoder:
    def __init__(self) -> None:
        self.ids = itertools.count()
        self.env: dict[str, Val] = {}

    def region(self, r: Region) -> Reg:
        rid = next(self.ids)
        for i, (a, t) in enumerate(zip(r.args, r.arg_types)):
            self.env[a] = arg(rid, i, t)
        stmts = []
        for op in r.ops:
            if _is_pure(op):
                self.env[op.results[0]] = self.pure(op)
            else:
                stmts.append(self.stmt(op))
        return reg(rid, _svec(r.arg_types), _vec(Stmt, stmts))

    def pure(self, op: Op) -> Val:
        if op.name == "arith.constant":
            return _const(op)
        args = [self.env[o] for o in op.operands]
        return PURE[len(args)](op.name, _extra(op), *args, op.result_types[0])

    def stmt(self, op: Op) -> Stmt:
        sid = next(self.ids)
        operands = [self.env[o] for o in op.operands]
        regions = [self.region(r) for r in op.regions]
        # Ops whose result types we don't parse (opaque) still need a type slot.
        rtypes = op.result_types or [""] * len(op.results)
        for i, (name, ty) in enumerate(zip(op.results, rtypes)):
            self.env[name] = res(sid, i, ty)
        return stmt(sid, op.name, _svec(rtypes), _vec(Val, operands), _extra(op),
                    _vec(Reg, regions))

    def func(self, f: Func) -> Top:
        self.env = {}
        body = Region(f.arg_names, f.arg_types, f.body.ops)
        return fn(f.name, _svec(f.arg_types), _svec(f.result_types), self.region(body),
                  f.attributes_raw)


def encode(m: Module) -> Mod:
    enc = _Encoder()
    items = [verbatim(x.text) if isinstance(x, Verbatim) else enc.func(x) for x in m.funcs]
    return module(_vec(Top, items), m.attributes_raw)


def _name(callable_) -> str:
    ident = getattr(callable_, "ident", None)
    return ident.name if ident is not None else callable_.name


def _walk(expr):
    """Extracted egglog expr -> nested tuples: (ctor, *children); Vec -> tuple."""
    if hasattr(expr, "value") and not hasattr(expr, "callable"):
        return expr.value
    kids = tuple(_walk(a.expr) for a in expr.args)
    name = _name(expr.callable)
    return kids if name == "Vec" else (name, *kids)


class _Frame:
    def __init__(self, region_id: int) -> None:
        self.region_id = region_id
        self.ops: list[Op] = []
        self.names: dict = {}


class _Decoder:
    def __init__(self) -> None:
        self.fresh = (f"v{i}" for i in itertools.count())
        self.frames: list[_Frame] = []
        self.stmt_depth: dict[int, int] = {}

    def lookup(self, v):
        for f in reversed(self.frames):
            if v in f.names:
                return f.names[v]
        return None

    def depth(self, v) -> int:
        """Outermost frame where every leaf of ``v`` is defined."""
        kind = v[0]
        if kind == "arg":
            return next(i for i, f in enumerate(self.frames) if f.region_id == v[1])
        if kind == "res":
            return self.stmt_depth[v[1]]
        if kind in ("iconst", "const"):
            return 0
        d = max((self.depth(c) for c in v[3:-1]), default=0)
        if any(t in v[1] for t in TRAPPING):
            d = len(self.frames) - 1
        return d

    def emit(self, v) -> str:
        found = self.lookup(v)
        if found is not None:
            return found
        if v[0] in ("arg", "res"):
            raise ValueError(f"use of {v[:3]} outside its scope")
        frame = self.frames[self.depth(v)]
        name = next(self.fresh)
        if v[0] in ("iconst", "const"):
            op = Op("arith.constant", [name], [v[2]], [], {"value": str(v[1])})
        else:
            operands = [self.emit(c) for c in v[3:-1]]
            extra = json.loads(v[2]) if v[2] else {"a": {}, "m": {}}
            op = Op(v[1], [name], [v[-1]], operands, extra["a"], meta=extra["m"])
        frame.ops.append(op)
        frame.names[v] = name
        return name

    def region(self, r) -> Region:
        _, rid, arg_types, stmts = r
        frame = _Frame(rid)
        self.frames.append(frame)
        args = []
        for i, t in enumerate(arg_types):
            args.append(next(self.fresh))
            frame.names[("arg", rid, i, t)] = args[-1]
        for s in stmts:
            _, sid, name, rtypes, operands, extra, regions = s
            ins = [self.emit(o) for o in operands]
            subs = [self.region(x) for x in regions]
            outs = [next(self.fresh) for _ in rtypes]
            self.stmt_depth[sid] = len(self.frames) - 1
            for i, (o, t) in enumerate(zip(outs, rtypes)):
                frame.names[("res", sid, i, t)] = o
            ex = json.loads(extra) if extra else {"a": {}, "m": {}}
            frame.ops.append(Op(name, outs, [t for t in rtypes if t], ins, ex["a"], subs,
                                ex["m"]))
        self.frames.pop()
        return Region(args, list(arg_types), frame.ops)


def decode_tree(tree) -> Module:
    _, items, attrs = tree
    out = Module(attributes_raw=attrs)
    for it in items:
        if it[0] == "verbatim":
            out.funcs.append(Verbatim(it[1]))
            continue
        _, name, arg_types, result_types, body, fattrs = it
        b = _Decoder().region(body)
        out.funcs.append(Func(name, b.args, list(arg_types), list(result_types),
                              Region(ops=b.ops), fattrs))
    return out


def decode(term: Mod) -> Module:
    return decode_tree(_walk(term.__egg_typed_expr__.expr))


def optimize(m: Module, rules, cost_model, iters: int = 6) -> tuple[Module, int, int]:
    """Returns (optimized module, cost before, cost after)."""
    egraph = EGraph()
    term = encode(m)
    egraph.register(term)
    _, before = egraph.extract(term, include_cost=True, cost_model=cost_model)
    egraph.register(*rules)
    egraph.run(iters)
    best, after = egraph.extract(term, include_cost=True, cost_model=cost_model)
    return decode(best), before, after


def roundtrip(m: Module, rules=(), iters: int = 10, cost_model=None) -> Module:
    egraph = EGraph()
    term = encode(m)
    egraph.register(term)
    if rules:
        egraph.register(*rules)
        egraph.run(iters)
    return decode(egraph.extract(term, cost_model=cost_model)
                  if cost_model else egraph.extract(term))

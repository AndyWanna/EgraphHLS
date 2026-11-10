"""Reference interpreter for the IR, used to differentially test rewrites."""

from __future__ import annotations

import math
import random
import re

from .ir import Func, Module, Op, Region


class Unsupported(Exception):
    pass


class _Budget(Exception):
    pass


def width(ty: str) -> int:
    if ty == "index":
        return 64
    if ty == "i1":
        return 1
    m = re.fullmatch(r"i(\d+)", ty)
    if not m:
        raise Unsupported(f"type {ty}")
    return int(m.group(1))


def wrap(v: int, ty: str) -> int:
    w = width(ty)
    if w == 1:
        return v & 1
    v &= (1 << w) - 1
    return v - (1 << w) if v >> (w - 1) else v


def unsigned(v: int, ty: str) -> int:
    return v & ((1 << width(ty)) - 1)


def _tdiv(a: int, b: int) -> int:
    if b == 0:
        raise Unsupported("division by zero")
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


INT_BIN = {
    "addi": lambda a, b, t: a + b,
    "subi": lambda a, b, t: a - b,
    "muli": lambda a, b, t: a * b,
    "andi": lambda a, b, t: a & b,
    "ori": lambda a, b, t: a | b,
    "xori": lambda a, b, t: a ^ b,
    "divsi": lambda a, b, t: _tdiv(a, b),
    "remsi": lambda a, b, t: a - b * _tdiv(a, b),
    "divui": lambda a, b, t: _tdiv(unsigned(a, t), unsigned(b, t)),
    "remui": lambda a, b, t: unsigned(a, t) % unsigned(b, t) if b else _tdiv(1, 0),
    "shli": lambda a, b, t: a << b if 0 <= b < width(t) else _tdiv(1, 0),
    "shrui": lambda a, b, t: unsigned(a, t) >> b if 0 <= b < width(t) else _tdiv(1, 0),
    "shrsi": lambda a, b, t: a >> b if 0 <= b < width(t) else _tdiv(1, 0),
    "maxsi": lambda a, b, t: max(a, b),
    "minsi": lambda a, b, t: min(a, b),
}
FLOAT_BIN = {
    "addf": lambda a, b: a + b,
    "subf": lambda a, b: a - b,
    "mulf": lambda a, b: a * b,
    "divf": lambda a, b: a / b if b else math.copysign(math.inf, a) if a else math.nan,
    "maxf": max,
    "minf": min,
}
CMPI = {
    "eq": lambda a, b: a == b, "ne": lambda a, b: a != b,
    "slt": lambda a, b: a < b, "sle": lambda a, b: a <= b,
    "sgt": lambda a, b: a > b, "sge": lambda a, b: a >= b,
}
CMPF = {
    "oeq": lambda a, b: a == b, "one": lambda a, b: a != b,
    "olt": lambda a, b: a < b, "ole": lambda a, b: a <= b,
    "ogt": lambda a, b: a > b, "oge": lambda a, b: a >= b,
    "ueq": lambda a, b: a == b, "une": lambda a, b: a != b,
    "ult": lambda a, b: a < b, "ule": lambda a, b: a <= b,
    "ugt": lambda a, b: a > b, "uge": lambda a, b: a >= b,
}


class Memref:
    """Sparse memory; unwritten cells read a deterministic pseudo-random value."""

    def __init__(self, elem: str, seed: int) -> None:
        self.elem, self.seed, self.cells = elem, seed, {}

    def load(self, idx: tuple):
        if idx not in self.cells:
            rnd = random.Random(hash((self.seed, idx)))
            self.cells[idx] = (rnd.randint(0, 7) if self.elem.startswith("i")
                               else rnd.uniform(-2, 2))
        return self.cells[idx]

    def store(self, idx: tuple, v) -> None:
        self.cells[idx] = v


class View:
    """``polygeist.subindex``: same rank offsets dim 0, lower rank fixes dim 0."""

    def __init__(self, base, i: int, drop: bool) -> None:
        self.base, self.i, self.drop = base, i, drop

    def _at(self, idx: tuple) -> tuple:
        return (self.i, *idx) if self.drop else (idx[0] + self.i, *idx[1:])

    def load(self, idx: tuple):
        return self.base.load(self._at(idx))

    def store(self, idx: tuple, v) -> None:
        self.base.store(self._at(idx), v)


def _rank(memref_ty: str) -> int:
    return len(re.findall(r"(?:\?|\d+)x", memref_ty))


def _elem(memref_ty: str) -> str:
    return memref_ty[memref_ty.index("<") + 1 : memref_ty.rindex(">")].split("x")[-1]


class _Interp:
    def __init__(self, budget: int, funcs: dict) -> None:
        self.budget = budget
        self.funcs = funcs
        self.allocs = 0

    def tick(self) -> None:
        self.budget -= 1
        if self.budget < 0:
            raise _Budget()

    def region(self, r: Region, env: dict, args: list):
        env = dict(env)
        env.update(zip(r.args, args))
        for op in r.ops:
            self.tick()
            out = self.op(op, env)
            if op.name in ("scf.yield", "func.return", "scf.condition"):
                return out
            env.update(zip(op.results, out))
        return []

    def op(self, op: Op, env: dict) -> list:
        n, ins = op.name, [env[o] for o in op.operands]
        ty = op.result_types[0] if op.result_types else ""
        if "parts" in op.meta:
            return self.opaque(op, ins)
        if n == "arith.constant":
            v = op.attributes["value"]
            if v in ("true", "false"):
                return [int(v == "true")]
            return [float(v) if ty.startswith(("f", "b")) else wrap(int(v), ty)]
        short = n.split(".", 1)[1]
        if short in INT_BIN:
            return [wrap(INT_BIN[short](ins[0], ins[1], ty), ty)]
        if short in FLOAT_BIN:
            return [FLOAT_BIN[short](*ins)]
        if short == "negf":
            return [-ins[0]]
        if short in ("cmpi", "cmpf"):
            p, (a, b) = op.attributes["predicate"], ins
            if p in ("ult", "ule", "ugt", "uge") and short == "cmpi":
                t = op.meta["operand_type"]
                a, b, p = unsigned(a, t), unsigned(b, t), "s" + p[1:]
            if short == "cmpf" and (math.isnan(a) or math.isnan(b)):
                return [int(p.startswith("u") or p == "une")]
            return [int((CMPI if short == "cmpi" else CMPF)[p](a, b))]
        if short == "select":
            return [ins[1] if ins[0] else ins[2]]
        if short in ("index_cast", "extsi", "trunci"):
            return [wrap(ins[0], ty)]
        if short in ("index_castui", "extui"):
            return [wrap(unsigned(ins[0], op.meta["src_type"]), ty)]
        if short in ("sitofp",):
            return [float(ins[0])]
        if short in ("fptosi",):
            return [wrap(int(ins[0]), ty)]
        if short in ("extf", "truncf"):
            return [ins[0]]
        if n.startswith("math.") and hasattr(math, short):
            return [getattr(math, short)(*ins)]
        if n == "memref.load":
            return [ins[0].load(tuple(ins[1:]))]
        if n == "memref.store":
            ins[1].store(tuple(ins[2:]), ins[0])
            return []
        if n == "memref.alloc":
            self.allocs += 1
            return [Memref(_elem(ty), -self.allocs)]
        if n in ("scf.yield", "func.return"):
            return ins
        if n == "scf.if":
            r = op.regions[0] if ins[0] else (op.regions[1] if len(op.regions) > 1 else None)
            return self.region(r, env, []) if r else []
        if n == "scf.for":
            lb, ub, step, *carried = ins
            for i in range(lb, ub, step):
                carried = self.region(op.regions[0], env, [i, *carried])
            return carried
        if n == "scf.while":
            vals = ins
            while True:
                cond, *vals = self.region(op.regions[0], env, vals)
                if not cond:
                    return vals
                vals = self.region(op.regions[1], env, vals)
        raise Unsupported(n)

    def opaque(self, op: Op, ins: list) -> list:
        if op.name in ("scf.condition", "memref.cast"):
            return ins
        if op.name == "llvm.mlir.undef":
            return [0]
        parts = op.meta["parts"]
        text = parts[0] + "".join(f"@{i}" + p for i, p in enumerate(parts[1:]))
        if op.name.strip("\"") == "polygeist.subindex":
            src, dst = re.search(r":\s*\((memref<[^>]*>).*->\s*(memref<[^>]*>)", text).groups()
            return [View(ins[0], ins[1], _rank(dst) < _rank(src))]
        call = re.match(r"(?:func\.)?call\s+@([\w$.]+)\(", text)
        if call and call.group(1) in self.funcs:
            f = self.funcs[call.group(1)]
            return self.region(f.body, dict(zip(f.arg_names, ins)), [])
        m = re.match(r"affine\.(load|store)\s+(?:@(\d+),\s*)?@(\d+)\[(.*)\]", text)
        if not m:
            raise Unsupported(op.name)
        idx = []
        for e in m.group(4).split(","):
            sm = re.fullmatch(r"\s*(?:symbol\()?@(\d+)\)?\s*", e)
            if sm:
                idx.append(ins[int(sm.group(1))])
            elif re.fullmatch(r"\s*-?\d+\s*", e):
                idx.append(int(e))
            else:
                raise Unsupported(f"affine index {e}")
        mem = ins[int(m.group(3))]
        if m.group(1) == "load":
            return [mem.load(tuple(idx))]
        mem.store(tuple(idx), ins[int(m.group(2))])
        return []


def run(func: Func, seed: int = 0, budget: int = 50_000_000, funcs: dict | None = None):
    """Run ``func`` on seeded random inputs. Returns (results, memory state)."""
    rnd = random.Random(seed)
    args = []
    for i, t in enumerate(func.arg_types):
        if t.startswith("memref"):
            args.append(Memref(_elem(t), seed * 1000 + i))
        elif t.startswith("f"):
            args.append(rnd.uniform(-2, 2))
        else:
            args.append(wrap(rnd.randint(1, 8), t))
    try:
        out = _Interp(budget, funcs or {}).region(func.body, dict(zip(func.arg_names, args)), [])
    except _Budget:
        raise Unsupported("step budget exceeded") from None
    mem = [sorted(a.cells.items()) for a in args if isinstance(a, Memref)]
    return out, mem


def equivalent(a: Module, b: Module, seeds=range(2), rel_tol: float = 0.0) -> bool:
    fa = {f.name: f for f in a.funcs if isinstance(f, Func)}
    fb = {f.name: f for f in b.funcs if isinstance(f, Func)}
    return fa.keys() == fb.keys() and all(
        _same(run(fa[n], s, funcs=fa), run(fb[n], s, funcs=fb), rel_tol) for n in fa for s in seeds)


def _same(x, y, rel_tol: float) -> bool:
    if rel_tol and isinstance(x, float) and isinstance(y, float):
        return math.isclose(x, y, rel_tol=rel_tol, abs_tol=rel_tol) or repr(x) == repr(y)
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        return len(x) == len(y) and all(_same(p, q, rel_tol) for p, q in zip(x, y))
    return repr(x) == repr(y)

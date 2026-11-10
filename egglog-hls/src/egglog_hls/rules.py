"""Integer rewrite rules over the dataflow encoding."""

from __future__ import annotations

from egglog import String, eq, i64, ne, panic, rewrite, rule, var

from .egraph_dataflow import Val, iconst, p2

INT_TYPES = {"i8": 8, "i16": 16, "i32": 32, "i64": 64, "index": 64}


def _wrap(x, bits: int):
    if bits >= 64:
        return x
    m, h = i64(1 << bits), i64(1 << (bits - 1))
    return ((x + h) % m + m) % m - h


def _fold(name: str, op, ty: str, bits: int):
    a, b, e = var("a", i64), var("b", i64), var("e", String)
    lhs = p2(name, e, iconst(a, ty), iconst(b, ty), ty)
    conds = []
    if bits >= 64:  # i64 overflow would abort egglog; only fold small operands
        lim = i64(1 << 31)
        conds = [a < lim, a > i64(0) - lim, b < lim, b > i64(0) - lim]
    return rewrite(lhs).to(iconst(_wrap(op(a, b), bits), ty), *conds)


def int_rules(types=INT_TYPES):
    x, y, z = var("x", Val), var("y", Val), var("z", Val)
    c = var("c", i64)
    e, e2 = var("e", String), var("e2", String)
    out = []
    for ty, bits in types.items():
        def P(name, a, b, extra=e):
            return p2(f"arith.{name}", extra, a, b, ty)

        zero, one = iconst(0, ty), iconst(1, ty)
        pow2 = [c > i64(0), eq(c & (c - i64(1))).to(i64(0))]
        out += [
            rewrite(P("addi", x, y)).to(P("addi", y, x)),
            rewrite(P("muli", x, y)).to(P("muli", y, x)),
            rewrite(P("addi", x, zero)).to(x),
            rewrite(P("subi", x, zero)).to(x),
            rewrite(P("muli", x, one)).to(x),
            rewrite(P("muli", x, zero)).to(zero),
            rewrite(P("subi", x, x)).to(zero),
            rewrite(P("muli", x, iconst(c, ty))).to(P("shli", x, iconst(c.log2(), ty)), *pow2),
            rewrite(P("divui", x, iconst(c, ty))).to(P("shrui", x, iconst(c.log2(), ty)), *pow2),
            rewrite(P("remui", x, iconst(c, ty))).to(P("andi", x, iconst(c - i64(1), ty)),
                                                     *pow2),
            rewrite(P("addi", P("muli", x, y), P("muli", x, z, e2))).to(
                P("muli", x, P("addi", y, z))),
            _fold("arith.addi", lambda a, b: a + b, ty, bits),
            _fold("arith.subi", lambda a, b: a - b, ty, bits),
            _fold("arith.muli", lambda a, b: a * b, ty, bits),
            _fold("arith.andi", lambda a, b: a & b, ty, bits),
            _fold("arith.ori", lambda a, b: a | b, ty, bits),
            _fold("arith.xori", lambda a, b: a ^ b, ty, bits),
        ]
        # Two different constants in one e-class means an unsound rule fired.
        a, b = var("a", i64), var("b", i64)
        out.append(rule(eq(iconst(a, ty)).to(iconst(b, ty)), ne(a).to(b)).then(
            panic(f"unsound rewrite: two {ty} constants merged")))
    return out

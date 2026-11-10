import pathlib

import pytest
from egglog import String, eq, i64, rewrite, var

from egglog_hls import parse_module, print_module
from egglog_hls import egraph_dataflow as df
from egglog_hls.cost import hls_cost
from egglog_hls.interp import equivalent, run
from egglog_hls.ir import canonicalize, structurally_equal
from egglog_hls.rules import float_rules, int_rules

EXAMPLES = sorted((pathlib.Path(__file__).parent.parent / "examples").glob("*.mlir"))


@pytest.fixture(params=EXAMPLES, ids=[p.stem for p in EXAMPLES])
def module(request):
    return parse_module(request.param.read_text())


def test_egraph_matches_direct_decode(module):
    out = df.roundtrip(module)
    assert structurally_equal(out, df.decode(df.encode(module)))
    assert structurally_equal(out, df.roundtrip(out))
    assert structurally_equal(out, parse_module(print_module(canonicalize(out))))


def test_dataflow_preserves_semantics(module):
    assert equivalent(module, df.roundtrip(module))


def test_rules_preserve_semantics(module):
    rules = int_rules() + float_rules()
    assert equivalent(module, df.roundtrip(module, rules, 6, hls_cost))


SRC = """func.func @f(%a: i32, %b: i32, %c: i32) -> i32 {
  %c8 = arith.constant 8 : i32
  %c0 = arith.constant 0 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  %0 = arith.muli %a, %c8 : i32
  %1 = arith.addi %0, %c0 : i32
  %2 = arith.muli %a, %b : i32
  %3 = arith.muli %a, %c : i32
  %4 = arith.addi %2, %3 : i32
  %5 = arith.addi %c3, %c4 : i32
  %6 = arith.addi %4, %5 : i32
  %7 = arith.addi %1, %6 : i32
  return %7 : i32
}
"""


def test_rules_reduce_multipliers():
    m = parse_module(SRC)
    out = df.roundtrip(m, int_rules(), 8, hls_cost)
    text = print_module(canonicalize(out))
    assert text.count("arith.muli") == 1
    assert "arith.shli" in text and "arith.constant 7 : i32" in text
    assert equivalent(m, out, seeds=range(5))


def test_interp_catches_a_wrong_rule():
    x, c, e = var("x", df.Val), var("c", i64), var("e", String)
    bad = rewrite(df.p2("arith.muli", e, x, df.iconst(c, "i32"), "i32")).to(
        df.p2("arith.shli", e, x, df.iconst(c.log2() - i64(1), "i32"), "i32"),
        c > i64(2), eq(c & (c - i64(1))).to(i64(0)))
    m = parse_module(SRC)
    assert not equivalent(m, df.roundtrip(m, [bad, *int_rules()], 8, hls_cost))


FSRC = """func.func @g(%a: f64, %b: f64, %c: f64) -> f64 {
  %one = arith.constant 1.000000e+00 : f64
  %two = arith.constant 2.000000e+00 : f64
  %0 = arith.mulf %a, %b : f64
  %1 = arith.mulf %a, %c : f64
  %2 = arith.subf %0, %1 : f64
  %3 = arith.mulf %2, %one : f64
  %4 = arith.mulf %3, %two : f64
  return %4 : f64
}
"""


def test_exact_float_rules():
    m = parse_module(FSRC)
    out = df.roundtrip(m, float_rules(), 8, hls_cost)
    assert print_module(canonicalize(out)).count("arith.mulf") == 2
    assert equivalent(m, out, seeds=range(5))


def test_fast_math_factors_multiplies():
    m = parse_module(FSRC)
    out = df.roundtrip(m, float_rules(fast_math=True), 8, hls_cost)
    assert print_module(canonicalize(out)).count("arith.mulf") == 1
    assert equivalent(m, out, seeds=range(5), rel_tol=1e-9)


def test_subindex_view():
    src = """func.func @h(%m: memref<?xf64>, %i: index) -> f64 {
  %c1 = arith.constant 1 : index
  %v = "polygeist.subindex"(%m, %i) : (memref<?xf64>, index) -> memref<?xf64>
  %x = memref.load %v[%c1] : memref<?xf64>
  %j = arith.addi %i, %c1 : index
  %y = memref.load %m[%j] : memref<?xf64>
  %z = arith.subf %x, %y : f64
  return %z : f64
}
"""
    (out,), _ = run(parse_module(src).funcs[0])
    assert out == 0.0

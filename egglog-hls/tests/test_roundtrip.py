"""The core guarantee: every example goes MLIR -> e-graph -> MLIR unchanged."""

from __future__ import annotations

import pathlib

import pytest

from egglog_hls import parse_module, print_module, roundtrip, roundtrip_text
from egglog_hls.ir import structurally_equal

EXAMPLES = sorted((pathlib.Path(__file__).parent.parent / "examples").glob("*.mlir"))
IDS = [p.stem for p in EXAMPLES]


@pytest.fixture(params=EXAMPLES, ids=IDS)
def example_text(request) -> str:
    return request.param.read_text()


def test_roundtrip_is_identity_and_fixpoint(example_text: str) -> None:
    result = roundtrip_text(example_text)
    assert result.identity_ok, result.diff
    assert result.fixpoint_ok


def test_parse_print_are_inverse(example_text: str) -> None:
    module = parse_module(example_text)
    reparsed = parse_module(print_module(module))
    assert structurally_equal(module, reparsed)


def test_yield_form_survives_the_egraph() -> None:
    """The iter_args/yield shape (the ScaleHLS gotcha) must not be lost."""
    text = (pathlib.Path(__file__).parent.parent / "examples" / "matmul_yield.mlir").read_text()
    out = roundtrip(parse_module(text))
    fors = _all_ops(out, "scf.for")
    assert any(len(f.regions[0].args) > 1 for f in fors), "iter_args loop was dropped"
    assert _all_ops(out, "scf.yield"), "scf.yield was dropped"


def _all_ops(module, name):
    found = []

    def walk(region):
        for op in region.ops:
            if op.name == name:
                found.append(op)
            for sub in op.regions:
                walk(sub)

    for func in module.funcs:
        walk(func.body)
    return found


def test_multi_result_prints_as_plain_result_list() -> None:
    # "%0:2 = ..." with "%0#1" uses must come back as legal MLIR: canonical
    # names, never a "#" in a result position.
    text = """func.func @f(%a: i32) -> i32 {
  %0:2 = "test.pair"(%a) : (i32) -> (i32, i32)
  %1 = arith.addi %0#0, %0#1 : i32
  return %1 : i32
}
"""
    result = roundtrip_text(text)
    assert result.ok, result.diff
    assert "#" not in result.output_text
    assert "%1, %2 = \"test.pair\"(%0)" in result.output_text


def test_opaque_op_keeps_string_literals_intact() -> None:
    text = """func.func @f(%a: i32) {
  "test.log"(%a) {fmt = "%d items"} : (i32) -> ()
  return
}
"""
    result = roundtrip_text(text)
    assert result.ok, result.diff
    assert '{fmt = "%d items"}' in result.output_text

"""reg2mem turns the iter_args form into the store form, and the result still
round-trips cleanly through the e-graph."""

from __future__ import annotations

import pathlib

from egglog_hls import parse_module, roundtrip_text
from egglog_hls.normalize import reg2mem

EXAMPLES = pathlib.Path(__file__).parent.parent / "examples"


def _for_loops(module):
    out = []

    def walk(region):
        for op in region.ops:
            if op.name == "scf.for":
                out.append(op)
            for sub in op.regions:
                walk(sub)

    for func in module.funcs:
        walk(func.body)
    return out


def test_reg2mem_removes_iter_args() -> None:
    module = parse_module((EXAMPLES / "matmul_yield.mlir").read_text())
    assert any(len(f.regions[0].args) > 1 for f in _for_loops(module))

    normalized = reg2mem(module)
    assert all(len(f.regions[0].args) == 1 for f in _for_loops(normalized)), \
        "reg2mem left a loop-carried accumulator behind"
    # No scf.yield with operands should remain (accumulator became a store).
    ops = _for_loops(normalized)
    for loop in ops:
        assert not loop.results, "loop still produces a value after reg2mem"


def test_reg2mem_output_still_roundtrips() -> None:
    from egglog_hls.printer import print_module

    module = parse_module((EXAMPLES / "matmul_yield.mlir").read_text())
    normalized_text = print_module(reg2mem(module))
    result = roundtrip_text(normalized_text)
    assert result.ok, result.diff


def test_reg2mem_is_noop_on_store_form() -> None:
    """The already-regular matmul must be left structurally unchanged."""
    from egglog_hls.ir import structurally_equal

    module = parse_module((EXAMPLES / "matmul_regular.mlir").read_text())
    assert structurally_equal(module, reg2mem(module))

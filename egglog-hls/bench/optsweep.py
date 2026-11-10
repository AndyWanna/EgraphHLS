"""Integer rewrites + HLS cost extraction over cgeist outputs, with an
interpreter check. Usage: python bench/optsweep.py out/*.mlir"""
import sys

from egglog_hls import parse_module
from egglog_hls.cost import hls_cost
from egglog_hls.egraph_dataflow import optimize
from egglog_hls.interp import Unsupported, equivalent
from egglog_hls.rules import int_rules

rules = int_rules()
print(f"{'kernel':32} {'before':>9} {'after':>9} {'delta':>7}  check")
for path in sys.argv[1:]:
    m = parse_module(open(path).read())
    out, before, after = optimize(m, rules, hls_cost)
    try:
        check = "same" if equivalent(m, out) else "MISMATCH"
    except Unsupported as e:
        check = f"skipped ({e})"
    name = path.rsplit("/", 1)[-1].removesuffix(".mlir")
    print(f"{name:32} {before:>9} {after:>9} {100 * (before - after) / max(before, 1):6.1f}%  {check}",
          flush=True)

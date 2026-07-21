# egglog-hls

The **DialEgg track** of EgraphHLS: a correctness-first, dialect-agnostic
round-trip between MLIR and an [egglog](https://github.com/egraphs-good/egglog-python)
e-graph.

```
MLIR text --parse--> IR --to_egraph--> e-graph --from_egraph--> IR --print--> MLIR text
```

This is the first task from the June 16 onboarding: *before adding any
transforms, get a program to go in and come back out unchanged, and verify it
across the board.* Rewrites and cost functions build on top of this.

## What works today

- **Real e-graph.** Every module is encoded into `egglog` terms
  (`mk_module`/`mk_func`/`mk_region`/`mk_op`), inserted into an `EGraph`, then
  extracted back and walked into the IR. With no rules registered the extracted
  term equals the inserted one; that is the identity round-trip. The same
  extraction path is what will lower a *rewritten* e-graph back to MLIR later.
- **Both `scf.for` shapes** from Andy's matmul screenshot survive the trip: the
  clean store form (`examples/matmul_regular.mlir`) and the loop-carried
  `iter_args` + `scf.yield` form (`examples/matmul_yield.mlir`).
- **Verified two ways.** `identity`: input and output are equal up to SSA
  value-numbering (the same notion of "unchanged" MLIR's own printer uses).
  `fixpoint`: printing then reparsing the output does not drift.
- **`reg2mem`** normalizes the loop-carried-accumulator pattern into the store
  form (`reg2mem -> optimize -> mem2reg`, the fix for the ScaleHLS gotcha).
  Conservative: loops that do not match the pattern are left untouched.

## Supported MLIR subset

`builtin` module, `func` (func/return/call), `arith` (constant, binary ops,
comparisons), `memref` (alloc/load/store), `scf` (for with/without `iter_args`,
yield). Enough for the affine kernels a correctness-first front end emits, plus
everything in `examples/`. Unrecognized ops raise `ParseError` rather than
being silently dropped.

## Usage

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

# MLIR -> e-graph -> MLIR, verified (exit code != 0 if it does not match)
python -m egglog_hls roundtrip examples/matmul_yield.mlir

# Normalize the iter_args form to the store form
python -m egglog_hls reg2mem examples/matmul_yield.mlir

pytest -q
```

```python
from egglog_hls import roundtrip_text
r = roundtrip_text(open("examples/matmul_yield.mlir").read())
assert r.ok            # identity_ok and fixpoint_ok
print(r.output_text)   # canonical MLIR rebuilt from the e-graph
```

## Layout

| path | role |
|------|------|
| `ir.py` | IR model + canonicalization (SSA value-numbering) + structural equality |
| `parser.py` | MLIR text -> IR |
| `printer.py` | IR -> MLIR text (deterministic, re-parseable) |
| `egraph_bridge.py` | IR <-> `egglog` (encode, insert, extract, decode) |
| `roundtrip.py` | orchestration + verification |
| `normalize.py` | `reg2mem` |
| `cli.py` | `python -m egglog_hls` |

## Not done yet (next milestones)

- **C -> MLIR front end.** The bake-off (ScaleHLS vs ScaleHLS-HIDA over the
  HLSFactory corpus) needs an MLIR/LLVM build and is not wired up here; the
  round-trip is driven off `.mlir` files, which is how correctness is checked
  anyway. This layer slots in ahead of `parse_module`.
- **Dataflow-aware encoding.** Operands are currently encoded as SSA-name
  strings, which is lossless for identity but flat. Rewrites want operands to
  reference their defining e-graph nodes (with explicit effect ordering for the
  side-effecting ops). That richer typed encoding is the next step before any
  equational rewrite rules.
- **`mem2reg`** (the inverse of `reg2mem`, run after optimization).
- Broader dialect coverage (`affine`, `scf.if` results, `linalg`).

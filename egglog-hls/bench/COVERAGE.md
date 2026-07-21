# Frontend coverage bake-off

`C --cgeist--> MLIR --egglog round-trip--> MLIR`, over HLSFactory
(`hlsfactory/hls_dataset_sources`). Run with `bench/bakeoff.py` inside the
toolchain container. Numbers below are the round-trip pass rate (cgeist emitted
a module *and* the e-graph round-trip was identity + fixpoint).

## Results

| suite | round-trip OK | notes |
|-------|---------------|-------|
| **polybench** | **9 / 9** | atax, bicg, gemm, gesummv, k2mm, k3mm, mvt, syr2k, syrk |
| **machsuite** | **3 / 19** | gemm_blocked, gemm_ncubed, stencil2D |

Polybench is the affine-loop-nest sweet spot this track targets, and it is
fully covered. MachSuite is dominated by irregular kernels (graph traversal,
sort, molecular dynamics, FFT) that lower to dialects/constructs the round-trip
does not model yet, and a couple that cgeist itself will not lower.

## What was added to hit polybench 9/9

- unary casts (`arith.index_cast`, `sitofp`, ...): `op %x : src to dst`
- structured `scf.if` (with optional `else` and results)
- unary arith (`arith.negf`) and boolean `arith.constant true/false`
- parser hardened: an unrecognized construct always raises `ParseError`
  (never an `AttributeError`), so the sweep buckets it cleanly
- module- and func-level `attributes {...}` preserved verbatim

## Backlog (ranked by MachSuite coverage impact)

1. **affine dialect** (`affine.for`/`affine.load`/`affine.store`, affine maps) —
   cgeist emits it for several kernels (fft_transpose, sort_radix, stencil3D,
   and behind other gaps). Biggest single lever.
2. **Generic opaque-op fallback** — capture any unrecognized op with its
   result/operand SSA values so it round-trips as an opaque e-graph node
   instead of failing. Would also absorb `polygeist.memref2pointer` and
   `math.*` (e.g. `math.exp`). Needs care so SSA value-numbering stays correct.
3. **multi-result ops** — `%10:2 = scf.for ... -> (t1, t2)` compact syntax and
   `%10#0` result refs (sort_merge).
4. **module-scope `llvm.mlir.global`** and other non-func top-level ops (viterbi).
5. **RT_FAIL on aes_table / aes_tableless** — parses and is identity, but the
   print->reparse fixpoint does not hold. A construct prints in a form that does
   not reparse identically; needs isolating. (Known bug, not just missing coverage.)

## Out of scope (for the affine track)

- **pp4fpgas / rosetta**: C++ with HLS types (`ap_int`, `hls::stream`) — a
  different frontend problem.
- cgeist could not emit a module for `md_grid` / `common` (utility dir).

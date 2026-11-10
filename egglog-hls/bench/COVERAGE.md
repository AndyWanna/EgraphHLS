# Frontend coverage bake-off

`C --cgeist--> MLIR --egglog round-trip--> MLIR`, over HLSFactory
(`hlsfactory/hls_dataset_sources`). Run with `bench/bakeoff.py` inside the
toolchain container. Numbers below are the round-trip pass rate (cgeist emitted
a module *and* the e-graph round-trip was identity + fixpoint).

## Results (2026-11-10)

Each kernel is compiled at its **declared top function** (`set_top` /
`add_files` in HLSFactory's `hls_template.tcl`). Earlier numbers picked the first
function that compiled, which was often a helper (e.g. `rj_xtime` for aes), so
they overstated coverage. OK now also requires `scalehls-opt` to accept the
printed output (the identity + fixpoint checks only compare our parser against
our printer, and missed two real printer bugs).

| suite | round-trip OK | notes |
|-------|---------------|-------|
| **polybench** | **9 / 9** | atax, bicg, gemm, gesummv, k2mm, k3mm, mvt, syr2k, syrk |
| **machsuite** | **13 / 19** | every kernel cgeist can lower round-trips (13 / 13) |

MachSuite failures are all in the frontend (cgeist, LLVM Nov-2022 pin), before
egglog-hls sees anything:

| kernel | cgeist failure |
|--------|----------------|
| aes_table, aes_tableless, viterbi | `'arith.extsi' op operation destroyed but still has uses` (crash) |
| sort_merge | `operand #0 does not dominate this use` (sort.c:38) |
| md_grid | assertion in `ValueCategory::getValue` ("must have shape 1") |
| common | utility dir, no top function |

## How the remaining gaps were closed (2026-11-10)

- **Opaque fallback**: any single-line op we do not model is kept as text
  segments with its SSA uses as holes, so it round-trips and still renumbers.
  Covers `affine.load/store`, `polygeist.*`, `llvm.*`, `scf.condition`, ...
- **Multi-result ops**: `%x:N` expands to N results, uses are `%x#i`.
- **`scf.while`** parsed first-class (before/after regions, `^bb0` args).
- **Module-scope `llvm.mlir.global` / `llvm.func`** kept verbatim.
- `arith.select`, `math.*`; fixed `arith.cmpi` printing the result type and a
  greedy func-header regex that ate `-> (T)`.

## What was added to hit polybench 9/9

- unary casts (`arith.index_cast`, `sitofp`, ...): `op %x : src to dst`
- structured `scf.if` (with optional `else` and results)
- unary arith (`arith.negf`) and boolean `arith.constant true/false`
- parser hardened: an unrecognized construct always raises `ParseError`
  (never an `AttributeError`), so the sweep buckets it cleanly
- module- and func-level `attributes {...}` preserved verbatim

## Backlog

1. **Dataflow e-graph encoding.** The current bridge embeds SSA names in every
   op term, so nothing ever merges and no rewrite can match across ops. The
   round-trip is lossless but cannot optimize. Next milestone.
2. Lower `affine.load/store` with constant / `symbol()` indices to `memref` in
   `normalize.py`, so the e-graph sees one memory dialect.
3. Frontend: try a newer Polygeist for the 5 cgeist crashes.

## Out of scope (for the affine track)

- **pp4fpgas / rosetta**: C++ with HLS types (`ap_int`, `hls::stream`) — a
  different frontend problem.
- cgeist could not emit a module for `md_grid` / `common` (utility dir).

# egglog-hls — status / resume notes

Pick-up doc for the EgraphHLS DialEgg track. Last updated 2026-11-10. Also mirrored in claude-mem (project `github`, observation #33823).

## Where things are

- **Repo:** local clone of `AndyWanna/EgraphHLS` at
  `~/Documents-Local/github/EgraphHLS` (local disk — the iCloud copy under
  `~/Documents/github` hangs on git/read, don't use it).
- **Branch:** `sam/egglog-roundtrip`, 4 commits, **local only, not pushed.**
  (The remote already has an older `sam/roundtrip` with a prior scaffold; new
  work uses a fresh branch to avoid a force-push.)
- **Package:** `egglog-hls/`, Python venv at `egglog-hls/.venv` (egglog 13 on
  the Mac's Python 3.14).

## Current milestone (done)

Correctness-first round-trip on the DialEgg track, verified end to end:

```
C  --cgeist-->  MLIR  --egglog e-graph-->  MLIR      (identity + fixpoint)
```

- Real egglog engine: encode MLIR -> egglog terms -> EGraph -> extract -> decode.
- Both `scf.for` shapes (store form + iter_args/yield) survive.
- `reg2mem` normalizes the loop-carried accumulator to store form.
- **Coverage:** polybench **9/9**, machsuite **13/19** at the declared top
  functions; the 6 misses are cgeist crashes, not ours (see `bench/COVERAGE.md`).
- Every OK output is also validated by `scalehls-opt` (bakeoff `INVALID` bucket).
- 14 tests passing.

## How to resume

Mac side (round-trip + tests):
```bash
cd ~/Documents-Local/github/EgraphHLS/egglog-hls
.venv/bin/python -m pytest -q
.venv/bin/python -m egglog_hls roundtrip examples/matmul_yield.mlir
```

Toolchain side (cgeist / scalehls-opt live in a Docker Desktop container):
```bash
docker start scalehls                 # container persists across restarts
# C -> MLIR:
docker exec scalehls bash -lc '/work/scalehls/polygeist/build/bin/cgeist \
  -S -O0 -function=NAME file.c -o out.mlir'
# coverage sweep (egglog-hls is copied to /work/egglog-hls in the container):
docker exec scalehls bash -lc 'CGEIST=/work/scalehls/polygeist/build/bin/cgeist \
  PYTHONPATH=/work/egglog-hls/src python3 /work/egglog-hls/bench/bakeoff.py \
  /work/HLSFactory/hlsfactory/hls_dataset_sources/polybench'
```
Toolchain tools: `scalehls-opt` + `scalehls-translate` in
`/work/scalehls/build/bin`, `cgeist` in `/work/scalehls/polygeist/build/bin`.
Built Release+asserts (Ubuntu 22.04, clang-14, LLVM Nov-2022 pin). To reclaim
~20 GB: `docker rm -f scalehls && docker volume rm scalehls-data`.

## Next steps (ranked)

1. **Dataflow e-graph encoding** (`egraph_dataflow.py`, beside the old bridge):
   pure ops as name-free terms over a `Val` sort, effectful ops as an ordered
   `Vec[Stmt]`, decode by scoped memoized emit. Identity has to be redefined
   (CSE/reordering now happen). Design in claude-mem / Fable consult 2026-11-10.
2. First rewrite rules: constant fold (bitwidth-aware), x+0 / x*1,
   mul-by-2^k -> shli, udiv-by-2^k -> shrui, a*b + a*c -> a*(b+c).
3. Sam-owned contribution: HLS cost model (DSP for mul, free const shifts,
   width-proportional LUTs for add) + normalization passes; report per-kernel
   cost deltas.
4. Lower constant-index `affine.load/store` to `memref` in `normalize.py`.

## Open decisions (need Sam)

- PR #1 (`sam/egglog-roundtrip`) is open on AndyWanna/EgraphHLS. Coverage work
  is on `sam/affine-and-opaque`, stacked on it.
- Modal credentials for heavy runs: `../.env.modal` (git-excluded, mode 600).
- Post the Slack progress update to Andy (round-trip works + polybench 9/9).
- O-1 angle: keep a differentiated component Sam-owned (normalization passes /
  cost functions) since the DialEgg-vs-custom-Rust framing affects Criterion 5.

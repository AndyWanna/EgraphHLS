# egglog-hls — status / resume notes

Pick-up doc for the EgraphHLS DialEgg track. Last updated during the 2026-07-21
session. Also mirrored in claude-mem (project `github`, observation #33823).

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
- **Coverage:** polybench **9/9**, machsuite **3/19** (see `bench/COVERAGE.md`).
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

## Next steps (ranked, from bench/COVERAGE.md)

1. **Affine dialect** (`affine.for/load/store`, affine maps) — biggest lever for
   machsuite coverage; cgeist emits it for many kernels.
2. **Generic opaque-op fallback** — unknown ops round-trip as opaque e-graph
   nodes (absorbs `polygeist.memref2pointer`, `math.*`). Must keep SSA
   value-numbering correct. Worth a design chat with Andy.
3. **Multi-result ops** — `%10:2 = scf.for ... -> (t1, t2)` + `%10#0` refs.
4. **Module-scope `llvm.mlir.global`** and other non-func top-level ops.
5. **Bug:** aes_table / aes_tableless parse and are identity but fail the
   print->reparse fixpoint. Isolate the construct that prints non-reparseably.

## Open decisions (need Sam)

- Push `sam/egglog-roundtrip` and open a PR to Andy's repo? (Commits authored as
  `sacredvoid`; set a real email before pushing if desired.)
- Post the Slack progress update to Andy (round-trip works + polybench 9/9).
- O-1 angle: keep a differentiated component Sam-owned (normalization passes /
  cost functions) since the DialEgg-vs-custom-Rust framing affects Criterion 5.

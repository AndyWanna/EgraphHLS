# egglog-hls

The **DialEgg track** of EgraphHLS: get HLS designs through an e-graph and back
using [egglog](https://github.com/egraphs-good/egglog) (Python), via
[DialEgg](https://github.com/AzizZayed/dialegg-cgo-artifact), instead of the
hand-rolled Rust `egg` language in `../ehls/`.

This is one of two parallel flows in the repo:

| Directory      | Approach                          | Language    |
|----------------|-----------------------------------|-------------|
| `../ehls/`     | Custom MLIR language on `egg`     | Rust        |
| `egglog-hls/`  | DialEgg / dialect-agnostic egglog | Python      |

Kept separate on purpose so we can compare which path round-trips the most HLS
programs correctly before committing to one.

## Goal (first task, no transforms yet)

Build and verify the full round-trip with **no optimizations**:

```
C/C++  ->  MLIR  ->  e-graph (egglog)  ->  MLIR  ->  C/C++
```

Get a program in and back out, and confirm it is correct. Rewrites and cost
functions come later. Per the Jun 16 call: "before we even add any transforms,
let's just go in and out... make sure it's all correct across the board."

## Pipeline stages

1. **HLS -> MLIR** (front-end bake-off, decided outside this dir):
   - [ScaleHLS](https://github.com/UIUC-ChenLab/scalehls) (primary candidate)
   - [ScaleHLS-HIDA](https://github.com/UIUC-ChenLab/ScaleHLS-HIDA) (newer; mostly extra opt passes)
   - NOT [soda-opt](https://github.com/pnnl/soda-opt) (goes MLIR -> HLS, wrong direction)
   - Test corpus: [HLSFactory `hls_dataset_sources`](https://github.com/sharc-lab/HLSFactory/tree/main/hlsfactory/hls_dataset_sources)
   - Metric for "best": which front-end ingests the most of the corpus consistently.
2. **MLIR -> egglog** (`egglog_hls.lower`): use DialEgg to model MLIR ops/types/attrs in egglog.
3. **egglog -> MLIR** (`egglog_hls.raise_`): extract back to MLIR.
4. **MLIR -> C/C++**: emit / verify against the original.

## Known gotcha: `scf.for` + `yield`

The same loop nest (e.g. matmul `R[i][j] += A[i][k]*B[k][j]`) lowers to MLIR two
ways:
- **Regular form:** nested `scf.for` with a trailing `memref.store`. Easy to optimize.
- **Yield form:** `scf.for` with `yield` + `iter_args` (loop-carried accumulator).
  Semantically identical but irregular; ScaleHLS passes break on it.

Normalization plan: `reg2mem` -> (optimize) -> `mem2reg`. ScaleHLS provides both.
The round-trip must handle both lowering shapes. See `examples/matmul_regular.mlir`
and `examples/matmul_yield.mlir`.

## References

- DialEgg: paper [CGO'25 10.1145/3696443.3708957](https://dl.acm.org/doi/10.1145/3696443.3708957),
  talk [egraphs.org/meeting/2025-08-21-dialegg](https://egraphs.org/meeting/2025-08-21-dialegg),
  artifact [AzizZayed/dialegg-cgo-artifact](https://github.com/AzizZayed/dialegg-cgo-artifact) (archived; may be partial).
- egglog: [egraphs-good/egglog](https://github.com/egraphs-good/egglog)

## Status

Scaffold only. Stages are stubbed (`NotImplementedError`). Next: stand up egglog +
DialEgg locally and implement `lower` / `raise_` for a single MLIR program.

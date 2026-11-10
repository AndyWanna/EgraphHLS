# Frontend coverage

`C --cgeist--> MLIR --egglog round-trip--> MLIR` over HLSFactory, run with `bench/bakeoff.py`. Each kernel is compiled at its declared top function (`set_top`/`add_files` in `hls_template.tcl`). OK means identity + fixpoint hold and `scalehls-opt` accepts the printed output.

| suite | round-trip OK | notes |
|-------|---------------|-------|
| polybench | 9 / 9 | atax, bicg, gemm, gesummv, k2mm, k3mm, mvt, syr2k, syrk |
| machsuite | 13 / 19 | every kernel cgeist can lower round-trips (13 / 13) |

The MachSuite failures are in cgeist (LLVM Nov-2022 pin), before egglog-hls sees anything:

| kernel | cgeist failure |
|--------|----------------|
| aes_table, aes_tableless, viterbi | `'arith.extsi' op operation destroyed but still has uses` (crash) |
| sort_merge | `operand #0 does not dominate this use` (sort.c:38) |
| md_grid | assertion in `ValueCategory::getValue` ("must have shape 1") |
| common | utility dir, no top function |

pp4fpgas and rosetta are C++ with HLS types (`ap_int`, `hls::stream`) and not attempted.

## Integer rewrites

`bench/optsweep.py`: dataflow encoding + `rules.py`, extracted with `cost.py` (rough LUT-equivalents, a DSP counts as 100; not calibrated against Vitis yet). check = original and optimized give identical results under `interp.py` on random inputs.

| kernel | cost before | after | delta | check |
|--------|------------:|------:|------:|-------|
| stencil3D | 16672 | 8032 | -51.8% | same |
| sort_radix | 6867 | 3747 | -45.4% | same |
| gemm_blocked | 3508 | 2548 | -27.4% | same |
| stencil2D | 1792 | 1312 | -26.8% | same |
| gemm_ncubed | 2916 | 2196 | -24.7% | same |
| fft_transpose | 745288 | 702568 | -5.7% | same |
| backprop | 95277 | 91197 | -4.3% | skipped (polygeist.subindex) |
| md_knn | 236172 | 235932 | -0.1% | same |

bfs_bulk, bfs_queue, nw, spmv_crs, spmv_ellpack and all of polybench: no change (polybench is float arithmetic, the rules are integer only), all check same.

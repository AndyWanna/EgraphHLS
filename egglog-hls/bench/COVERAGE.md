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

## Rewrites

`bench/optsweep.py`: dataflow encoding + `rules.py` (integer rules and IEEE-exact float rules), extracted with `cost.py` (rough LUT-equivalents, a DSP counts as 100; not calibrated against Vitis yet). Costs are DAG costs (`greedy-dag` extraction, shared values counted once). check = original and optimized give identical results under `interp.py` on random inputs.

| kernel | cost before | after | delta | check |
|--------|------------:|------:|------:|-------|
| stencil3D | 7824 | 4224 | -46.0% | same |
| sort_radix | 3665 | 1985 | -45.8% | same |
| stencil2D | 1792 | 1312 | -26.8% | same |
| fft_transpose | 181960 | 144880 | -20.4% | same |
| gemm_ncubed | 2516 | 2036 | -19.1% | same |
| gemm_blocked | 2644 | 2164 | -18.2% | same |
| backprop | 89693 | 86433 | -3.6% | same |
| md_knn | 19900 | 19660 | -1.2% | same |

bfs_bulk, bfs_queue, nw, spmv_crs, spmv_ellpack and all of polybench: no change, all check same.

`--fast-math` adds float factoring (`a*b + a*c -> a*(b+c)`) and changes nothing here: polybench bodies have at most one product per sum, and the fft butterflies reuse both products (`x*c + y*c`, `x*c - y*c`), so factoring saves no multiplier.

# egglog-hls

MLIR <-> [egglog](https://github.com/egraphs-good/egglog-python) e-graph, plus a first set of integer rewrites extracted with an HLS cost model.

```
MLIR text --parse--> IR --to_egraph--> e-graph --from_egraph--> IR --print--> MLIR text
```

Supported subset: `func`, `arith`, `math`, `memref`, `scf` (for, if, while, yield). Anything else is kept as opaque text, or raises `ParseError`.

## Usage

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

# round-trip and check (identity + fixpoint), exit code != 0 on mismatch
python -m egglog_hls roundtrip examples/matmul_yield.mlir

# integer rewrites, extracted by HLS cost; --check runs the reference interpreter
python -m egglog_hls optimize --check examples/cgeist_matmul.mlir

# iter_args form -> store form
python -m egglog_hls reg2mem examples/matmul_yield.mlir

pytest -q
```

```python
from egglog_hls import roundtrip_text
r = roundtrip_text(open("examples/matmul_yield.mlir").read())
assert r.ok
print(r.output_text)
```

Two encodings: `egraph_bridge.py` stores ops as text (lossless, no rewrites), `egraph_dataflow.py` makes pure ops name-free terms so rules can match across ops (`rules.py`, costs in `cost.py`).

Results over HLSFactory are in `bench/COVERAGE.md`.

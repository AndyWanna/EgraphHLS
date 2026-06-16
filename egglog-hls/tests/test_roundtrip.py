"""Round-trip identity tests.

The first milestone: lowering MLIR into the e-graph and raising it back out, with
NO rewrites, must reproduce an equivalent module. These are marked xfail until the
DialEgg-backed `lower` / `raise_` are implemented.
"""

from pathlib import Path

import pytest

from egglog_hls import round_trip

EXAMPLES = Path(__file__).parent.parent / "examples"
MLIR_FIXTURES = ["matmul_regular.mlir", "matmul_yield.mlir"]


def _normalize(mlir: str) -> str:
    """Whitespace/comment-insensitive comparison. Replace with a real MLIR-aware
    equivalence check (or `mlir-opt` canonical print) once the pipeline emits MLIR."""
    lines = []
    for line in mlir.splitlines():
        stripped = line.split("//", 1)[0].strip()
        if stripped:
            lines.append(stripped)
    return "\n".join(lines)


@pytest.mark.parametrize("fixture", MLIR_FIXTURES)
@pytest.mark.xfail(reason="lower/raise_ not implemented yet (DialEgg not wired in)")
def test_identity_round_trip(fixture):
    src = (EXAMPLES / fixture).read_text()
    out = round_trip(src)
    assert _normalize(out) == _normalize(src)

#!/usr/bin/env python3
"""Frontend coverage bake-off.

For each kernel directory under ROOT:
  C  --cgeist-->  MLIR  --egglog round-trip-->  MLIR

Auto-detects the top function (filename first, then any function defined in the
primary source, trying each until cgeist emits a non-empty module), round-trips
the MLIR through egglog-hls, and buckets the outcome:

  OK          round-trip identity + fixpoint held
  PARSE_GAP   an MLIR construct egglog-hls does not parse yet (message names it)
  RT_FAIL     parsed, but the round-trip was not identity/fixpoint
  CGEIST_FAIL no function produced a non-empty module (C++/HLS, or no kernel)

Run inside the toolchain container (needs cgeist + egglog-hls importable):

  CGEIST=/work/scalehls/polygeist/build/bin/cgeist \
  PYTHONPATH=/work/egglog-hls/src \
  python3 bakeoff.py /work/HLSFactory/hlsfactory/hls_dataset_sources/polybench
"""
import glob
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from collections import Counter

from egglog_hls import roundtrip_text
from egglog_hls.parser import ParseError

CGEIST = os.environ.get("CGEIST", "/work/scalehls/polygeist/build/bin/cgeist")
# Oracle: real MLIR must accept what we print. Identity + fixpoint only compare
# our parser against our printer, so they cannot catch invalid output.
MLIR_OPT = os.environ.get("MLIR_OPT", "/work/scalehls/build/bin/scalehls-opt")
ROOT = sys.argv[1]
OUT = os.environ.get("BAKEOFF_OUT") or tempfile.mkdtemp(prefix="bakeoff_")
os.makedirs(OUT, exist_ok=True)

FN_DEF = re.compile(r"(?m)^[A-Za-z_][\w\s\*]*?\b([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{")
SKIP_FILE = re.compile(r"(test|tb|bench|_main|support|check|golden)", re.I)
KEYWORDS = {"if", "for", "while", "switch", "return", "sizeof", "main"}


def func_names(cfile):
    txt = open(cfile, errors="ignore").read()
    out = []
    for m in FN_DEF.finditer(txt):
        if m.group(1) not in KEYWORDS and m.group(1) not in out:
            out.append(m.group(1))
    return out


def run_cgeist(cfile, srcdir, func):
    out = os.path.join(OUT, f"{pathlib.Path(cfile).stem}__{func}.mlir")
    cmd = [CGEIST, "-S", "-O0", "-I", srcdir, f"-function={func}", cfile, "-o", out]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return None
    if p.returncode != 0:
        return None
    txt = open(out).read()
    return txt if "func.func" in txt else None


def mlir_rejects(text):
    """None if MLIR_OPT parses+verifies ``text``, else its first error line."""
    p = subprocess.run([MLIR_OPT, "--allow-unregistered-dialect", "-"], input=text,
                       capture_output=True, text=True, timeout=60)
    if p.returncode == 0:
        return None
    return next((l for l in p.stderr.splitlines() if "error" in l), p.stderr)[:80]


def compile_kernel(d):
    srcdir = os.path.join(d, "src")
    if not os.path.isdir(srcdir):
        srcdir = d
    cs = sorted(glob.glob(os.path.join(srcdir, "*.c")))
    primary = [c for c in cs if not SKIP_FILE.search(os.path.basename(c))] or cs
    for c in primary:
        cands = func_names(c)
        base = pathlib.Path(c).stem
        if base not in cands:
            cands.insert(0, base)
        for fn in cands:
            mlir = run_cgeist(c, srcdir, fn)
            if mlir is not None:
                return fn, mlir
    return None, None


def main():
    results = []
    for d in sorted(p for p in glob.glob(os.path.join(ROOT, "*")) if os.path.isdir(p)):
        name = os.path.basename(d)
        fn, mlir = compile_kernel(d)
        if mlir is None:
            results.append((name, "CGEIST_FAIL", "no function emitted a module"))
            continue
        try:
            r = roundtrip_text(mlir)
            if not r.ok:
                results.append((name, "RT_FAIL", r.diff[:80]))
                continue
            err = mlir_rejects(r.output_text)
            results.append((name, "INVALID", err) if err else (name, "OK", f"fn={fn}"))
        except ParseError as e:
            results.append((name, "PARSE_GAP", str(e)[:80]))
        except Exception as e:  # never let one kernel abort the sweep
            results.append((name, "ERROR", f"{type(e).__name__}: {str(e)[:60]}"))

    print(f"{'kernel':22} {'status':12} detail")
    print("-" * 96)
    for n, s, detail in results:
        print(f"{n:22} {s:12} {detail}")
    print("-" * 96)
    c = Counter(s for _, s, _ in results)
    print("SUMMARY:", dict(c), f"| total={len(results)} | OK={c.get('OK', 0)}")


if __name__ == "__main__":
    main()

"""CLI: python -m egglog_hls {roundtrip,reg2mem,optimize} path.mlir [--emit out.mlir]"""

from __future__ import annotations

import argparse
import sys

from .normalize import reg2mem
from .parser import parse_module
from .printer import print_module
from .ir import canonicalize
from .roundtrip import roundtrip_text


def _cmd_roundtrip(args: argparse.Namespace) -> int:
    text = _read(args.path)
    result = roundtrip_text(text)
    if args.emit:
        _write(args.emit, result.output_text)
    elif not args.quiet:
        sys.stdout.write(result.output_text)

    status = "OK" if result.ok else "FAIL"
    sys.stderr.write(
        f"[{status}] {args.path}: identity={result.identity_ok} "
        f"fixpoint={result.fixpoint_ok}\n"
    )
    if not result.ok:
        sys.stderr.write(result.diff + "\n")
    return 0 if result.ok else 1


def _cmd_reg2mem(args: argparse.Namespace) -> int:
    module = reg2mem(parse_module(_read(args.path)))
    out = print_module(canonicalize(module))
    if args.emit:
        _write(args.emit, out)
    else:
        sys.stdout.write(out)
    return 0


def _cmd_optimize(args: argparse.Namespace) -> int:
    from .cost import hls_cost
    from .egraph_dataflow import optimize
    from .interp import Unsupported, equivalent
    from .rules import int_rules

    module = parse_module(_read(args.path))
    out, before, after = optimize(module, int_rules(), hls_cost, args.iters)
    text = print_module(canonicalize(out))
    if args.emit:
        _write(args.emit, text)
    else:
        sys.stdout.write(text)
    sys.stderr.write(f"cost {before} -> {after} ({100 * (before - after) / max(before, 1):.1f}%)\n")
    if args.check:
        try:
            same = equivalent(module, out)
        except Unsupported as e:
            sys.stderr.write(f"check skipped: {e}\n")
            return 0
        sys.stderr.write(f"check: {'equivalent' if same else 'MISMATCH'}\n")
        return 0 if same else 1
    return 0


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="egglog_hls", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    rt = sub.add_parser("roundtrip", help="MLIR -> e-graph -> MLIR, verified")
    rt.add_argument("path")
    rt.add_argument("--emit", help="write output MLIR to this file")
    rt.add_argument("-q", "--quiet", action="store_true", help="suppress MLIR on stdout")
    rt.set_defaults(func=_cmd_roundtrip)

    rm = sub.add_parser("reg2mem", help="normalize scf.for iter_args to store form")
    rm.add_argument("path")
    rm.add_argument("--emit", help="write output MLIR to this file")
    rm.set_defaults(func=_cmd_reg2mem)

    op = sub.add_parser("optimize", help="apply integer rewrites, extract by HLS cost")
    op.add_argument("path")
    op.add_argument("--emit", help="write output MLIR to this file")
    op.add_argument("--iters", type=int, default=6)
    op.add_argument("--check", action="store_true",
                    help="compare against the input with the reference interpreter")
    op.set_defaults(func=_cmd_optimize)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

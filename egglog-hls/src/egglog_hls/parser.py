"""A pragmatic recursive-descent parser for the MLIR subset this pipeline
round-trips: the ``builtin`` module, ``func`` (func/return/call), ``arith``
(constant + binary ops + comparisons), ``memref`` (alloc/load/store) and
``scf`` (for with/without iter_args, if, yield).

It is deliberately not a full MLIR grammar. It targets the constructs a
correctness-first HLS front end (ScaleHLS/HIDA) actually emits for affine
kernels, plus everything in ``examples/``. Anything it does not recognize
raises ``ParseError`` rather than being silently dropped.
"""

from __future__ import annotations

import re

from .ir import Func, Module, Op, Region


class ParseError(Exception):
    pass


_RESULTS_RE = re.compile(r"^\s*(%[\w$.-]+(?:\s*,\s*%[\w$.-]+)*)\s*=\s*(.*)$")
_MODULE_RE = re.compile(r"^module\b\s*(.*?)\s*\{$")
# name, args, and everything between the args ')' and the opening '{'
# (an optional "-> (...)" and/or "attributes {...}").
_FUNC_RE = re.compile(r"^func\.func\s+@([\w$.-]+)\s*\((.*)\)\s*(.*?)\s*\{$")
_FUNC_ATTRS_RE = re.compile(r"attributes\s*\{.*\}")
_FUNC_RET_RE = re.compile(r"->\s*\((.*)\)$|->\s*(.+)$")
_ARG_RE = re.compile(r"%([\w$.-]+)\s*:\s*(.+)")
_FOR_RE = re.compile(
    r"^scf\.for\s+%([\w$.-]+)\s*=\s*%([\w$.-]+)\s+to\s+%([\w$.-]+)\s+step\s+%([\w$.-]+)"
    r"(?:\s+iter_args\((.*?)\))?\s*(?:->\s*\(([^)]*)\))?\s*\{$"
)
_LOAD_RE = re.compile(r"^memref\.load\s+%([\w$.-]+)\[(.*?)\]\s*:\s*(.+)$")
_STORE_RE = re.compile(r"^memref\.store\s+%([\w$.-]+)\s*,\s*%([\w$.-]+)\[(.*?)\]\s*:\s*(.+)$")
_ALLOC_RE = re.compile(r"^memref\.alloc[a]?\(\s*\)\s*:\s*(.+)$")
_CONST_RE = re.compile(r"^arith\.constant\s+(.+?)\s*:\s*(.+)$")
_BINARY_RE = re.compile(r"^(arith\.\w+)\s+%([\w$.-]+)\s*,\s*%([\w$.-]+)\s*:\s*(.+)$")
_CMP_RE = re.compile(r"^(arith\.cmp[if])\s+(\w+)\s*,\s*%([\w$.-]+)\s*,\s*%([\w$.-]+)\s*:\s*(.+)$")
_YIELD_RE = re.compile(r"^scf\.yield(?:\s+(.*?)\s*:\s*(.+))?$")
_RETURN_RE = re.compile(r"^(?:func\.)?return(?:\s+(.*?)\s*:\s*(.+))?$")


def _split_top(text: str, sep: str = ",") -> list[str]:
    """Split on ``sep`` at bracket depth 0 (so ``memref<4x4xf32>`` and nested
    parens stay intact)."""
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch in "<([{":
            depth += 1
        elif ch in ">)]}":
            depth -= 1
        if ch == sep and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _names(csv: str) -> list[str]:
    return [t.strip().lstrip("%") for t in csv.split(",") if t.strip()]


def _elem_type(memref_ty: str) -> str:
    inner = memref_ty[memref_ty.index("<") + 1 : memref_ty.rindex(">")]
    return inner.split("x")[-1]


class _Parser:
    def __init__(self, text: str) -> None:
        self.lines = self._clean(text)
        self.i = 0

    @staticmethod
    def _clean(text: str) -> list[str]:
        out = []
        for raw in text.splitlines():
            line = raw.split("//", 1)[0].rstrip()
            if line.strip():
                out.append(line.strip())
        return out

    def _peek(self) -> str:
        return self.lines[self.i]

    def _next(self) -> str:
        line = self.lines[self.i]
        self.i += 1
        return line

    def parse_module(self) -> Module:
        module = Module()
        mm = _MODULE_RE.match(self._peek())
        if mm:
            module.attributes_raw = mm.group(1)
            self._next()
            while self._peek() != "}":
                module.funcs.append(self._parse_func())
            self._next()  # closing }
        else:  # bare functions, no module wrapper
            while self.i < len(self.lines):
                module.funcs.append(self._parse_func())
        return module

    def _parse_func(self) -> Func:
        header = self._next()
        m = _FUNC_RE.match(header)
        if not m:
            raise ParseError(f"unrecognized func header: {header!r}")
        name, args_s, trailing = m.group(1), m.group(2), m.group(3).strip()
        arg_names, arg_types = [], []
        for a in _split_top(args_s):
            am = _ARG_RE.match(a.strip())
            if not am:
                raise ParseError(f"bad func argument: {a!r}")
            arg_names.append(am.group(1))
            arg_types.append(am.group(2).strip())
        result_types, attrs_raw = self._split_func_trailing(trailing)
        func = Func(name, arg_names, arg_types, result_types, self._parse_region([]))
        func.attributes_raw = attrs_raw
        return func

    @staticmethod
    def _split_func_trailing(trailing: str) -> tuple[list[str], str]:
        """Pull an optional "attributes {...}" and "-> (...)" off the func
        header, in either order, preserving the attributes text verbatim."""
        attrs_raw = ""
        am = _FUNC_ATTRS_RE.search(trailing)
        if am:
            attrs_raw = am.group(0)
            trailing = (trailing[: am.start()] + trailing[am.end() :]).strip()
        result_types: list[str] = []
        rm = _FUNC_RET_RE.match(trailing)
        if rm:
            result_types = _split_top(rm.group(1) or rm.group(2))
        return result_types, attrs_raw

    def _parse_region(self, block_args: list[tuple[str, str]]) -> Region:
        """Parse ops up to the matching ``}``. The opening ``{`` has already
        been consumed by the caller."""
        region = Region(
            args=[a for a, _ in block_args],
            arg_types=[t for _, t in block_args],
        )
        while self._peek() != "}":
            region.ops.append(self._parse_op())
        self._next()  # closing }
        return region

    def _parse_op(self) -> Op:
        line = self._next()
        m = _RESULTS_RE.match(line)
        if m:
            results = _names(m.group(1))
            rhs = m.group(2).strip()
        else:
            results = []
            rhs = line
        return self._dispatch(rhs, results)

    def _dispatch(self, rhs: str, results: list[str]) -> Op:
        if rhs.startswith("scf.for"):
            return self._parse_for(rhs, results)
        if rhs.startswith("arith.constant"):
            cm = _CONST_RE.match(rhs)
            return Op("arith.constant", results, [cm.group(2).strip()], [],
                      {"value": cm.group(1).strip()})
        if rhs.startswith(("arith.cmpi", "arith.cmpf")):
            cm = _CMP_RE.match(rhs)
            return Op(cm.group(1), results, ["i1"], [cm.group(3), cm.group(4)],
                      {"predicate": cm.group(2)})
        if rhs.startswith("arith."):
            bm = _BINARY_RE.match(rhs)
            if not bm:
                raise ParseError(f"unrecognized arith op: {rhs!r}")
            return Op(bm.group(1), results, [bm.group(4).strip()],
                      [bm.group(2), bm.group(3)])
        if rhs.startswith("memref.load"):
            lm = _LOAD_RE.match(rhs)
            mty = lm.group(3).strip()
            operands = [lm.group(1), *_names(lm.group(2))]
            return Op("memref.load", results, [_elem_type(mty)], operands,
                      meta={"memref_type": mty})
        if rhs.startswith("memref.store"):
            sm = _STORE_RE.match(rhs)
            mty = sm.group(4).strip()
            operands = [sm.group(1), sm.group(2), *_names(sm.group(3))]
            return Op("memref.store", [], [], operands, meta={"memref_type": mty})
        if rhs.startswith(("memref.alloc", "memref.alloca")):
            am = _ALLOC_RE.match(rhs)
            return Op("memref.alloc", results, [am.group(1).strip()], [])
        if rhs.startswith("scf.yield"):
            ym = _YIELD_RE.match(rhs)
            operands = _names(ym.group(1)) if ym.group(1) else []
            types = _split_top(ym.group(2)) if ym.group(2) else []
            return Op("scf.yield", [], [], operands, meta={"types": types})
        if rhs.startswith("return") or rhs.startswith("func.return"):
            rm = _RETURN_RE.match(rhs)
            operands = _names(rm.group(1)) if rm.group(1) else []
            types = _split_top(rm.group(2)) if rm.group(2) else []
            return Op("func.return", [], [], operands, meta={"types": types})
        raise ParseError(f"unsupported operation: {rhs!r}")

    def _parse_for(self, rhs: str, results: list[str]) -> Op:
        fm = _FOR_RE.match(rhs)
        if not fm:
            raise ParseError(f"unrecognized scf.for: {rhs!r}")
        iv, lb, ub, step = fm.group(1), fm.group(2), fm.group(3), fm.group(4)
        iter_s, ret_s = fm.group(5), fm.group(6)
        operands = [lb, ub, step]
        block_args = [(iv, "index")]
        result_types = _split_top(ret_s) if ret_s else []
        if iter_s:
            for pair, ty in zip(_split_top(iter_s), result_types):
                pm = re.match(r"%([\w$.-]+)\s*=\s*%([\w$.-]+)", pair.strip())
                if not pm:
                    raise ParseError(f"bad iter_args entry: {pair!r}")
                block_args.append((pm.group(1), ty))  # %acc gets iter type
                operands.append(pm.group(2))           # %init feeds the loop
        region = self._parse_region(block_args)
        return Op("scf.for", results, result_types, operands, regions=[region])


def parse_module(text: str) -> Module:
    return _Parser(text).parse_module()

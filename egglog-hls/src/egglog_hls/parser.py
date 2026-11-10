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

from .ir import Func, Module, Op, Region, Verbatim


class ParseError(Exception):
    pass


# Result list; "%x:N" is the packed form of N results, referenced as %x#i.
_RESULTS_RE = re.compile(r"^\s*(%[\w$.-]+(?::\d+)?(?:\s*,\s*%[\w$.-]+(?::\d+)?)*)\s*=\s*(.*)$")
_WHILE_RE = re.compile(r"^scf\.while\s*\((.*)\)\s*:\s*\((.*)\)\s*->\s*\((.*)\)\s*\{$")
_BB_RE = re.compile(r"^\^bb\d+\((.*)\):$")
# Opaque fallback tokenizer: skip string literals (they may contain "%d"),
# capture every SSA value use.
_OPAQUE_TOK = re.compile(r'"(?:[^"\\]|\\.)*"|%[\w$.-]+(?:#\d+)?')
_MODULE_RE = re.compile(r"^module\b\s*(.*?)\s*\{$")
# name, args, and everything between the args ')' and the opening '{'
# (an optional "-> (...)" and/or "attributes {...}").
_FUNC_RE = re.compile(r"^func\.func\s+@([\w$.-]+)\s*\((.*)\)\s*(.*?)\s*\{$")
_FUNC_ATTRS_RE = re.compile(r"attributes\s*\{.*\}")
_FUNC_RET_RE = re.compile(r"->\s*\((.*)\)$|->\s*(.+)$")
_ARG_RE = re.compile(r"%([\w$.#-]+)\s*:\s*(.+)")
_FOR_RE = re.compile(
    r"^scf\.for\s+%([\w$.#-]+)\s*=\s*%([\w$.#-]+)\s+to\s+%([\w$.#-]+)\s+step\s+%([\w$.#-]+)"
    r"(?:\s+iter_args\((.*?)\))?\s*(?:->\s*\(([^)]*)\))?\s*\{$"
)
_IF_RE = re.compile(r"^scf\.if\s+%([\w$.#-]+)\s*(?:->\s*\(([^)]*)\))?\s*\{$")
_LOAD_RE = re.compile(r"^memref\.load\s+%([\w$.#-]+)\[(.*?)\]\s*:\s*(.+)$")
_STORE_RE = re.compile(r"^memref\.store\s+%([\w$.#-]+)\s*,\s*%([\w$.#-]+)\[(.*?)\]\s*:\s*(.+)$")
_ALLOC_RE = re.compile(r"^memref\.alloc[a]?\(\s*\)\s*:\s*(.+)$")
_CONST_RE = re.compile(r"^arith\.constant\s+(.+?)\s*:\s*(.+)$")
# Same-type n-ary ops: binary arith and arith.select %c, %a, %b : T.
_NARY_RE = re.compile(r"^((?:arith|math)\.\w+)\s+(%[\w$.#-]+(?:\s*,\s*%[\w$.#-]+)+)\s*:\s*(.+)$")
# Unary casts: arith.index_cast / sitofp / trunci / extf / bitcast ...
# form "op %x : <src> to <dst>"
_CAST_RE = re.compile(r"^((?:arith|math)\.\w+)\s+%([\w$.#-]+)\s*:\s*(.+?)\s+to\s+(.+)$")
# Unary same-type ops, e.g. "arith.negf %x : f64".
_UNARY_RE = re.compile(r"^((?:arith|math)\.\w+)\s+%([\w$.#-]+)\s*:\s*(.+)$")
_CMP_RE = re.compile(r"^(arith\.cmp[if])\s+(\w+)\s*,\s*%([\w$.#-]+)\s*,\s*%([\w$.#-]+)\s*:\s*(.+)$")
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


def _match_close(text: str, open_at: int) -> int:
    """Index of the bracket closing the one at ``open_at`` (any bracket kind
    counts toward depth, so types like ``memref<4xf32>`` nest correctly)."""
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] in "<([{":
            depth += 1
        elif text[i] in ">)]}" and not (text[i] == ">" and text[i - 1] == "-"):
            depth -= 1
            if depth == 0:
                return i
    raise ParseError(f"unbalanced brackets: {text!r}")


def _need(regex: re.Pattern, text: str, kind: str) -> re.Match:
    """Match or raise a clean ParseError (never an AttributeError downstream)."""
    m = regex.match(text)
    if m is None:
        raise ParseError(f"unrecognized {kind}: {text!r}")
    return m


def _names(csv: str) -> list[str]:
    return [t.strip().lstrip("%") for t in csv.split(",") if t.strip()]


def _result_names(csv: str) -> list[str]:
    """Expand "%x:3" to ["x#0", "x#1", "x#2"] so each result is its own value
    (uses are spelled %x#i). After canonicalization these get plain names and
    print as an ordinary result list, which is legal MLIR."""
    out = []
    for tok in _names(csv):
        name, _, n = tok.partition(":")
        out.extend([f"{name}#{i}" for i in range(int(n))] if n else [name])
    return out


def _opaque(rhs: str, results: list[str]) -> Op:
    """Generic fallback for a single-line op we do not model: keep its text as
    literal segments with the SSA uses as holes (``meta["parts"]`` has one more
    entry than ``operands``), so canonicalize can still renumber it."""
    parts, operands, last = [], [], 0
    for m in _OPAQUE_TOK.finditer(rhs):
        if m.group(0).startswith("%"):
            parts.append(rhs[last : m.start()])
            operands.append(m.group(0)[1:])
            last = m.end()
    parts.append(rhs[last:])
    name = rhs.split(None, 1)[0].strip('"').split("(", 1)[0]
    return Op(name, results, [], operands, meta={"parts": parts})


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
                module.funcs.append(self._parse_top())
            self._next()  # closing }
        else:  # bare functions, no module wrapper
            while self.i < len(self.lines):
                module.funcs.append(self._parse_top())
        return module

    def _parse_top(self) -> Func | Verbatim:
        if self._peek().startswith("func.func"):
            return self._parse_func()
        if not self._peek().startswith("llvm."):
            raise ParseError(f"unsupported module-scope op: {self._peek()!r}")
        # llvm.mlir.global / llvm.func: keep the text, including any body.
        lines = [self._next()]
        depth = lines[0].count("{") - lines[0].count("}")
        while depth > 0:
            lines.append(self._next())
            depth += lines[-1].count("{") - lines[-1].count("}")
        return Verbatim("\n".join(lines))

    def _parse_func(self) -> Func:
        header = self._next()
        m = _FUNC_RE.match(header)
        if not m:
            raise ParseError(f"unrecognized func header: {header!r}")
        name = m.group(1)
        # Split args from the trailing "-> ..." with a balanced-paren scan:
        # a greedy regex would swallow a parenthesized return type "-> (i8)".
        open_at = header.index("(", m.start(2) - 1)
        close_at = _match_close(header, open_at)
        args_s = header[open_at + 1 : close_at]
        trailing = header[close_at + 1 :].rstrip().removesuffix("{").strip()
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
            results = _result_names(m.group(1))
            rhs = m.group(2).strip()
        else:
            results = []
            rhs = line
        return self._dispatch(rhs, results)

    def _dispatch(self, rhs: str, results: list[str]) -> Op:
        if rhs.startswith("scf.for"):
            return self._parse_for(rhs, results)
        if rhs.startswith("scf.if"):
            return self._parse_if(rhs, results)
        if rhs.startswith("scf.while"):
            return self._parse_while(rhs, results)
        if rhs.startswith("arith.constant"):
            cm = _CONST_RE.match(rhs)
            if cm:
                return Op("arith.constant", results, [cm.group(2).strip()], [],
                          {"value": cm.group(1).strip()})
            bm = re.match(r"arith\.constant\s+(true|false)$", rhs)  # bool: no ": type"
            if bm:
                return Op("arith.constant", results, ["i1"], [], {"value": bm.group(1)})
            raise ParseError(f"unrecognized arith.constant: {rhs!r}")
        if rhs.startswith(("arith.cmpi", "arith.cmpf")):
            cm = _need(_CMP_RE, rhs, "arith.cmp")
            # MLIR prints the *operand* type after ':'; the result is i1.
            return Op(cm.group(1), results, ["i1"], [cm.group(3), cm.group(4)],
                      {"predicate": cm.group(2)},
                      meta={"operand_type": cm.group(5).strip()})
        if rhs.startswith(("arith.", "math.")):
            cm = _CAST_RE.match(rhs)
            if cm:  # unary cast: op %x : src to dst
                return Op(cm.group(1), results, [cm.group(4).strip()],
                          [cm.group(2)], meta={"src_type": cm.group(3).strip()})
            bm = _NARY_RE.match(rhs)
            if bm:
                return Op(bm.group(1), results, [bm.group(3).strip()],
                          _names(bm.group(2)))
            um = _UNARY_RE.match(rhs)  # e.g. arith.negf %x : f64
            if um:
                return Op(um.group(1), results, [um.group(3).strip()], [um.group(2)])
            raise ParseError(f"unrecognized arith op: {rhs!r}")
        if rhs.startswith("memref.load"):
            lm = _need(_LOAD_RE, rhs, "memref.load")
            mty = lm.group(3).strip()
            operands = [lm.group(1), *_names(lm.group(2))]
            return Op("memref.load", results, [_elem_type(mty)], operands,
                      meta={"memref_type": mty})
        if rhs.startswith("memref.store"):
            sm = _need(_STORE_RE, rhs, "memref.store")
            mty = sm.group(4).strip()
            operands = [sm.group(1), sm.group(2), *_names(sm.group(3))]
            return Op("memref.store", [], [], operands, meta={"memref_type": mty})
        if rhs.startswith(("memref.alloc", "memref.alloca")):
            am = _need(_ALLOC_RE, rhs, "memref.alloc")
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
        # Multi-line ops (regions, block labels) cannot go through the fallback:
        # a hole cannot tell a block-arg definition from a use.
        if rhs.endswith("{") or "^bb" in rhs:
            raise ParseError(f"unsupported operation: {rhs!r}")
        return _opaque(rhs, results)

    def _parse_block(self) -> tuple[Region, str]:
        """Read ops until a line that is ``}`` or ``} else {``, consume it,
        and return the region plus which terminator closed it."""
        region = Region()
        while True:
            line = self._peek()
            if line in ("}", "} else {", "} do {"):
                self._next()
                return region, line
            region.ops.append(self._parse_op())

    def _parse_if(self, rhs: str, results: list[str]) -> Op:
        m = _IF_RE.match(rhs)
        if not m:
            raise ParseError(f"unrecognized scf.if: {rhs!r}")
        cond, ret_s = m.group(1), m.group(2)
        result_types = _split_top(ret_s) if ret_s else []
        then_region, term = self._parse_block()
        regions = [then_region]
        if term == "} else {":
            else_region, _ = self._parse_block()
            regions.append(else_region)
        return Op("scf.if", results, result_types, [cond], regions=regions)

    def _parse_while(self, rhs: str, results: list[str]) -> Op:
        """scf.while (%a = %init, ...) : (T...) -> (R...) { before } do {
        ^bb0(%x: R, ...): after }. Before-region args are the %a's."""
        m = _need(_WHILE_RE, rhs, "scf.while")
        init_types = _split_top(m.group(2))
        result_types = _split_top(m.group(3))
        before_args, operands = [], []
        for pair in _split_top(m.group(1)):
            pm = _need(re.compile(r"%([\w$.#-]+)\s*=\s*%([\w$.#-]+)$"), pair, "scf.while init")
            before_args.append(pm.group(1))
            operands.append(pm.group(2))
        before, term = self._parse_block()
        if term != "} do {":
            raise ParseError(f"scf.while without do-region: {rhs!r}")
        before.args, before.arg_types = before_args, init_types
        after_args: list[tuple[str, str]] = []
        bm = _BB_RE.match(self._peek())
        if bm:
            self._next()
            for a in _split_top(bm.group(1)):
                am = _need(_ARG_RE, a, "block argument")
                after_args.append((am.group(1), am.group(2).strip()))
        after = self._parse_region(after_args)
        return Op("scf.while", results, result_types, operands, regions=[before, after])

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
                pm = re.match(r"%([\w$.#-]+)\s*=\s*%([\w$.#-]+)", pair.strip())
                if not pm:
                    raise ParseError(f"bad iter_args entry: {pair!r}")
                block_args.append((pm.group(1), ty))  # %acc gets iter type
                operands.append(pm.group(2))           # %init feeds the loop
        region = self._parse_region(block_args)
        return Op("scf.for", results, result_types, operands, regions=[region])


def parse_module(text: str) -> Module:
    return _Parser(text).parse_module()

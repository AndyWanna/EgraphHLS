"""Print the IR model back to MLIR text.

Output is deterministic and re-parseable: running ``parse_module(print_module(m))``
yields a module structurally equal to ``m`` (see :func:`ir.structurally_equal`).
For a stable, diffable rendering, canonicalize before printing so SSA names are
positional.
"""

from __future__ import annotations

from .ir import Func, Module, Op, Region, Verbatim

_INDENT = "  "


def _v(name: str) -> str:
    return "%" + name


def _lhs(results: list[str]) -> str:
    """Result list. Results parsed from "%x:N" are named x#0..x#N-1; re-pack
    them so the text is legal MLIR (uses stay "%x#i")."""
    if results and "#" in results[0]:
        base = results[0].split("#", 1)[0]
        if results == [f"{base}#{i}" for i in range(len(results))]:
            return f"%{base}:{len(results)}"
    return ", ".join(_v(r) for r in results)


def _print_op(op: Op, depth: int, lines: list[str]) -> None:
    pad = _INDENT * depth
    lhs = _lhs(op.results)
    prefix = f"{lhs} = " if op.results else ""

    if op.name == "scf.for":
        lines.append(pad + prefix + _for_header(op))
        _print_region(op.regions[0], depth, lines)
        lines.append(pad + "}")
        return

    if op.name == "scf.if":
        rets = f" -> ({', '.join(op.result_types)})" if op.result_types else ""
        lines.append(f"{pad}{prefix}scf.if {_v(op.operands[0])}{rets} {{")
        _print_region(op.regions[0], depth, lines)
        if len(op.regions) == 2:
            lines.append(pad + "} else {")
            _print_region(op.regions[1], depth, lines)
        lines.append(pad + "}")
        return

    if op.name == "scf.while":
        before, after = op.regions
        pairs = ", ".join(f"{_v(a)} = {_v(i)}" for a, i in zip(before.args, op.operands))
        lines.append(f"{pad}{prefix}scf.while ({pairs}) : ({', '.join(before.arg_types)})"
                     f" -> ({', '.join(op.result_types)}) {{")
        _print_region(before, depth, lines)
        lines.append(pad + "} do {")
        if after.args:
            bb = ", ".join(f"{_v(a)}: {t}" for a, t in zip(after.args, after.arg_types))
            lines.append(f"{pad}^bb0({bb}):")
        _print_region(after, depth, lines)
        lines.append(pad + "}")
        return

    lines.append(pad + prefix + _print_body(op))


def _for_header(op: Op) -> str:
    region = op.regions[0]
    iv = region.args[0]
    lb, ub, step = op.operands[0], op.operands[1], op.operands[2]
    head = f"scf.for {_v(iv)} = {_v(lb)} to {_v(ub)} step {_v(step)}"
    inits = op.operands[3:]
    accs = region.args[1:]
    if inits:
        pairs = ", ".join(f"{_v(a)} = {_v(i)}" for a, i in zip(accs, inits))
        head += f" iter_args({pairs}) -> ({', '.join(op.result_types)})"
    return head + " {"


def _print_body(op: Op) -> str:
    name = op.name
    if "parts" in op.meta:  # opaque fallback: interleave text and value holes
        parts = op.meta["parts"]
        return parts[0] + "".join(_v(o) + p for o, p in zip(op.operands, parts[1:]))
    if name == "arith.constant":
        val = op.attributes["value"]
        if val in ("true", "false"):  # boolean constants print without a type
            return f"arith.constant {val}"
        return f"arith.constant {val} : {op.result_types[0]}"
    if name in ("arith.cmpi", "arith.cmpf"):
        a, b = op.operands
        return f"{name} {op.attributes['predicate']}, {_v(a)}, {_v(b)} : {op.meta['operand_type']}"
    if name.startswith(("arith.", "math.")) and "src_type" in op.meta:  # unary cast
        return f"{name} {_v(op.operands[0])} : {op.meta['src_type']} to {op.result_types[0]}"
    if name.startswith(("arith.", "math.")):
        operands = ", ".join(_v(o) for o in op.operands)
        return f"{name} {operands} : {op.result_types[0]}"
    if name == "memref.load":
        mem, *idx = op.operands
        idx_s = ", ".join(_v(i) for i in idx)
        return f"memref.load {_v(mem)}[{idx_s}] : {op.meta['memref_type']}"
    if name == "memref.store":
        val, mem, *idx = op.operands
        idx_s = ", ".join(_v(i) for i in idx)
        return f"memref.store {_v(val)}, {_v(mem)}[{idx_s}] : {op.meta['memref_type']}"
    if name == "memref.alloc":
        return f"memref.alloc() : {op.result_types[0]}"
    if name == "scf.yield":
        if not op.operands:
            return "scf.yield"
        vals = ", ".join(_v(o) for o in op.operands)
        return f"scf.yield {vals} : {', '.join(op.meta.get('types', []))}"
    if name == "func.return":
        if not op.operands:
            return "return"
        vals = ", ".join(_v(o) for o in op.operands)
        return f"return {vals} : {', '.join(op.meta.get('types', []))}"
    raise ValueError(f"cannot print op: {op.name}")


def _print_region(region: Region, depth: int, lines: list[str]) -> None:
    for op in region.ops:
        _print_op(op, depth + 1, lines)


def _print_func(func: Func, depth: int, lines: list[str]) -> None:
    pad = _INDENT * depth
    args = ", ".join(f"{_v(n)}: {t}" for n, t in zip(func.arg_names, func.arg_types))
    ret = f" -> ({', '.join(func.result_types)})" if func.result_types else ""
    attrs = f" {func.attributes_raw}" if func.attributes_raw else ""
    lines.append(f"{pad}func.func @{func.name}({args}){ret}{attrs} {{")
    _print_region(func.body, depth, lines)
    lines.append(pad + "}")


def print_module(module: Module) -> str:
    head = f"module {module.attributes_raw} {{" if module.attributes_raw else "module {"
    lines = [head]
    for func in module.funcs:
        if isinstance(func, Verbatim):
            lines.extend(_INDENT + t for t in func.text.splitlines())
        else:
            _print_func(func, 1, lines)
    lines.append("}")
    return "\n".join(lines) + "\n"

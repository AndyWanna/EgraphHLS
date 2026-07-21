"""Print the IR model back to MLIR text.

Output is deterministic and re-parseable: running ``parse_module(print_module(m))``
yields a module structurally equal to ``m`` (see :func:`ir.structurally_equal`).
For a stable, diffable rendering, canonicalize before printing so SSA names are
positional.
"""

from __future__ import annotations

from .ir import Func, Module, Op, Region

_INDENT = "  "


def _v(name: str) -> str:
    return "%" + name


def _print_op(op: Op, depth: int, lines: list[str]) -> None:
    pad = _INDENT * depth
    lhs = ", ".join(_v(r) for r in op.results)
    prefix = f"{lhs} = " if op.results else ""

    if op.name == "scf.for":
        lines.append(pad + prefix + _for_header(op))
        _print_region(op.regions[0], depth, lines)
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
    if name == "arith.constant":
        return f"arith.constant {op.attributes['value']} : {op.result_types[0]}"
    if name in ("arith.cmpi", "arith.cmpf"):
        a, b = op.operands
        return f"{name} {op.attributes['predicate']}, {_v(a)}, {_v(b)} : {op.result_types[0]}"
    if name.startswith("arith."):
        a, b = op.operands
        return f"{name} {_v(a)}, {_v(b)} : {op.result_types[0]}"
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
        _print_func(func, 1, lines)
    lines.append("}")
    return "\n".join(lines) + "\n"

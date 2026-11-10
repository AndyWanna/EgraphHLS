"""reg2mem: demote loop-carried accumulators (scf.for iter_args + scf.yield) to load/store form.

    %init = memref.load %M[I]; %r = scf.for ... iter_args(%acc = %init) { ...; scf.yield %y }; memref.store %r, %M[I]
becomes a load at the top of the body and a store of %y at the end. Loops that do not
match exactly are left alone.
"""

from __future__ import annotations

import copy

from .ir import Func, Module, Op, Region


def _load_key(op: Op) -> tuple | None:
    """(memref, indices, type) of the slot a load reads."""
    if op.name != "memref.load":
        return None
    mem, *idx = op.operands
    return (mem, tuple(idx), op.meta["memref_type"])


def _store_key(op: Op) -> tuple | None:
    if op.name != "memref.store":
        return None
    _val, mem, *idx = op.operands
    return (mem, tuple(idx), op.meta["memref_type"])


def _uses(ops: list[Op], value: str) -> int:
    count = 0
    for op in ops:
        count += op.operands.count(value)
        for region in op.regions:
            count += _uses(region.ops, value)
    return count


def _reg2mem_loop(region: Region, idx: int) -> bool:
    """Rewrite the scf.for at region.ops[idx] in place; False if it does not match."""
    loop = region.ops[idx]
    accs = loop.regions[0].args[1:]          # skip induction var
    inits = loop.operands[3:]
    if not accs:
        return False

    body = loop.regions[0]
    if not body.ops or body.ops[-1].name != "scf.yield":
        return False
    yielded = body.ops[-1].operands
    if len(yielded) != len(accs) or len(loop.results) != len(accs):
        return False

    pre_loads: list[Op] = []
    slots: list[tuple] = []
    for init in inits:
        producer = next((o for o in region.ops[:idx] if init in o.results), None)
        if producer is None or _load_key(producer) is None:
            return False
        # the loop itself is the one allowed use of init
        if _uses(region.ops, init) != 1:
            return False
        pre_loads.append(producer)
        slots.append(_load_key(producer))

    post_stores: list[Op] = []
    for res, slot in zip(loop.results, slots):
        store = next(
            (o for o in region.ops[idx + 1 :]
             if _store_key(o) == slot and o.operands[0] == res),
            None,
        )
        if store is None or _uses(region.ops, res) != 1:
            return False
        post_stores.append(store)

    new_loads = [
        Op("memref.load", [acc], [_elem_of(slot[2])], [slot[0], *slot[1]],
           meta={"memref_type": slot[2]})
        for acc, slot in zip(accs, slots)
    ]
    new_stores = [
        Op("memref.store", [], [], [y, slot[0], *slot[1]],
           meta={"memref_type": slot[2]})
        for y, slot in zip(yielded, slots)
    ]
    body.ops = new_loads + body.ops[:-1] + new_stores

    loop.operands = loop.operands[:3]
    loop.results = []
    loop.result_types = []
    body.args = body.args[:1]
    body.arg_types = body.arg_types[:1]

    remove = set(map(id, pre_loads)) | set(map(id, post_stores))
    region.ops = [o for o in region.ops if id(o) not in remove]
    return True


def _elem_of(memref_ty: str) -> str:
    inner = memref_ty[memref_ty.index("<") + 1 : memref_ty.rindex(">")]
    return inner.split("x")[-1]


def _reg2mem_region(region: Region) -> None:
    for op in region.ops:            # bottom-up: descend first
        for sub in op.regions:
            _reg2mem_region(sub)
    i = 0
    while i < len(region.ops):
        op = region.ops[i]
        if op.name == "scf.for" and len(op.regions[0].args) > 1:
            _reg2mem_loop(region, i)
        i += 1


def reg2mem(module: Module) -> Module:
    """Copy of module with matching loop-carried accumulators demoted to stores."""
    out = copy.deepcopy(module)
    for func in out.funcs:
        if isinstance(func, Func):
            _reg2mem_region(func.body)
    return out

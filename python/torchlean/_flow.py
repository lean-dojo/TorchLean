"""Branches and loops of a capture's internal nodes, lowered to Lean's flow artifact.

An importer builds its internal nodes (those of ``_artifact.lower``) in a region: a list of node
indices and flows, in order. A ``Loop`` is a ``whileLoop`` or a ``scan`` of Lean's ``IR.Flow`` and
a ``Cond`` a ``cond``, each with regions of its own, which read the values of enclosing regions as
they are. ``lower`` writes the artifact that Lean's ``FlowArtifact.ofJson`` reads,
``{"format": ..., "version": ..., "fuel": n, "flow": F}``. Each run of nodes between flows is one
block, an artifact of ``_artifact.lower``, and the state before each block, flow and region end is
the list of values that the rest of the program reads, constants aside, which each block that reads
one writes itself (``_Lowering.region``). ``prune`` drops what gives no value that the program
returns first, so that no block is without outputs. ``flat`` gives the nodes of a program
without a flow that gives a returned value, which ``_artifact.lower`` writes as before flows:
every node of a program without a flow, and the other nodes that give one where its flows give
none (``prune``). ``lower`` is for the rest.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import numbers
from typing import Any

from . import _artifact

# The most iterations that a while loop of an artifact runs each time it is entered, by default.
FUEL = 1 << 16


def natural(fuel: Any) -> int:
    """``fuel`` as the ``int`` that an artifact holds, which ``json`` writes, refused unless it is
    an integer that is not negative, as Lean's ``FlowArtifact.ofJson`` reads it: a
    ``numbers.Integral`` such as a Python or numpy integer, but not a Boolean, nor a tensor or an
    array of one integer."""
    if isinstance(fuel, bool) or not isinstance(fuel, numbers.Integral):
        raise TypeError(f"fuel must be a nonnegative integer, not {type(fuel).__name__}")
    number = int(fuel)
    if number < 0:
        raise ValueError(f"fuel must be a nonnegative integer, not {number}")
    return number


@dataclass
class Loop:
    """A ``whileLoop`` of ``test`` and ``body``, or, where ``test`` is None, a ``scan`` of
    ``length`` iterations of ``body``. In the regions the carried values are ``slots``, which
    start at ``init``; the body gives them ``results`` and the test the Boolean ``holds``. After
    the loop they are ``outs``."""
    slots: list[int]
    init: list[int]
    body: list
    results: list[int]
    outs: list[int]
    test: list | None = None
    holds: int | None = None
    length: int = 0


@dataclass
class Cond:
    """A ``cond`` on the scalar ``index``, JAX's integer branch index or PyTorch's Boolean
    predicate (`_hop.lower`): branch ``k`` is the region ``branches[k]`` with the results
    ``results[k]``, which the selected branch gives ``outs`` after it."""
    index: int
    branches: list[list]
    results: list[list[int]]
    outs: list[int]


def prune(nodes: list, items: list, exit: list[int]) -> list:
    """The region ``items`` without what gives nothing to the values ``exit``: a node whose value
    nothing after it reads, and a flow none of whose results anything after it reads, regions and
    all, as JAX's ``pe.dce_jaxpr`` drops an equation without effects none of whose outputs is used
    (``_default_dce_rule`` and ``_cond_dce_rule``, jax/_src/interpreters/partial_eval.py:1236 and
    jax/_src/lax/control_flow/conditionals.py:823 in JAX 0.11.1). A flow that stays keeps all of
    its results, and its regions are pruned to them, a loop's test to its Boolean. So every region
    of the result gives a value, and every block that ``lower`` writes has an output."""
    live, kept = set(exit), []
    for item in reversed(items):
        if isinstance(item, int):
            if item in live:
                kept.append(item)
                live |= set(nodes[item].get("parents", []))
        elif live & set(item.outs):
            if isinstance(item, Cond):
                item = replace(item, branches=[prune(nodes, branch, results) for branch, results
                                               in zip(item.branches, item.results)])
            else:
                item = replace(item, body=prune(nodes, item.body, item.results),
                               test=None if item.test is None else
                               prune(nodes, item.test, [item.holds]))
            kept.append(item)
            live = (live - set(item.outs)) | _reads(nodes, item)
    kept.reverse()
    return kept


def _reads(nodes: list, flow: Loop | Cond) -> set[int]:
    # The values of enclosing regions that `flow` reads: its head and what its regions read.
    def region(items: list, exit: list[int]) -> set[int]:
        reads = set(exit)
        for item in reversed(items):
            if isinstance(item, int):
                reads.discard(item)
                reads |= set(nodes[item].get("parents", []))
            else:
                reads = (reads - set(item.outs)) | _reads(nodes, item)
        return reads

    if isinstance(flow, Cond):
        return {flow.index}.union(*(region(branch, results)
                                    for branch, results in zip(flow.branches, flow.results)))
    inner = region(flow.body, flow.results)
    if flow.test is not None:
        inner |= region(flow.test, [flow.holds])
    return set(flow.init) | (inner - set(flow.slots))


def flat(nodes: list, region: list, input_ids: list[int],
         output_ids: list[int]) -> tuple[list, dict[int, int]] | None:
    """The nodes that ``_artifact.lower`` writes for the region ``region`` of a whole program,
    from the inputs ``input_ids`` to the outputs ``output_ids``, and the new number of each, or
    ``None`` where a flow of the region gives a returned value, so that ``lower`` writes it. A
    region without flows is every node, numbered as it is; one whose flows give no returned value
    is the inputs and the nodes of the pruned region (``prune``), in their order and renumbered,
    without the nodes of the dropped flows."""
    if all(isinstance(item, int) for item in region):
        return nodes, {index: index for index in range(len(nodes))}
    items = prune(nodes, region, output_ids)
    if not all(isinstance(item, int) for item in items):
        return None
    keep = sorted(set(input_ids) | set(items))
    number = {j: k for k, j in enumerate(keep)}
    return [dict(nodes[j], id=number[j], parents=[number[i] for i in nodes[j].get("parents", [])])
            for j in keep], number


class _Lowering:
    def __init__(self, nodes: list, table: dict):
        self.nodes, self.table = nodes, table
        # The block and the `[step, output]` reference of each value, in the first block holding
        # it, and the number of blocks written, in the order of Lean's `Flow.blocks`.
        self.where: dict[int, tuple[int, list[int]]] = {}
        self.blocks = 0

    def kept(self, values: Any) -> set[int]:
        # The values that a state holds: constants are written where they are read.
        return {j for j in values if self.nodes[j]["kind"] != "const"}

    def live(self, items: list, exit: list[int]) -> set[int]:
        """The values of the state that the region ``items`` reads to give the state ``exit``."""
        live = self.kept(exit)
        for item in reversed(items):
            if isinstance(item, int):
                live.discard(item)
                live |= self.kept(self.nodes[item].get("parents", []))
            else:
                live = self.kept(self.head(item)) | set(self.through(item, live))
        return live

    @staticmethod
    def head(flow: Loop | Cond) -> list[int]:
        # The values that start a flow's state: a loop's carried values or a cond's index.
        return flow.init if isinstance(flow, Loop) else [flow.index]

    def through(self, flow: Loop | Cond, after: set[int]) -> list[int]:
        """The values that pass into ``flow``, after its head in its state: those read after it,
        other than its results, and those of enclosing regions its regions read. A loop's state
        keeps them through every iteration, and a cond's branches give its results and those
        read after it."""
        if isinstance(flow, Loop):
            inner = self.live(flow.body, flow.results)
            if flow.test is not None:
                inner |= self.live(flow.test, [flow.holds])
            inner -= set(flow.slots)
        else:
            inner = set().union(*(self.live(branch, results)
                                  for branch, results in zip(flow.branches, flow.results)))
        return sorted((after - set(flow.outs)) | inner)

    def region(self, items: list, entry: list[int], exit: list[int]) -> list:
        """The flow of the region ``items`` from the state ``entry`` to the state ``exit``: each
        run of nodes one block, the state before a flow its head and the values passing into it,
        and a block between two flows, or a flow and an end of the region, whose states
        differ. The parts are in order in a ``seq`` tree of balanced halves (``_sequence``). A run
        leaves out its constants, which only the blocks that read one write (``block``)."""
        segments: list = [[]]
        for item in items:
            if isinstance(item, int):
                if self.nodes[item]["kind"] != "const":
                    segments[-1].append(item)
            else:
                segments += [item, []]
        passing, leaving, after = {}, {}, self.kept(exit)
        for k in reversed(range(len(segments))):
            if isinstance(segments[k], list):
                after = self.live(segments[k], list(after))
            else:
                passing[k] = self.through(segments[k], after)
                leaving[k] = passing[k] if isinstance(segments[k], Loop) else \
                    sorted(after - set(segments[k].outs))
                after = self.kept(self.head(segments[k])) | set(passing[k])
        parts, state = [], list(entry)
        for k, segment in enumerate(segments):
            if isinstance(segment, list):
                target = exit if k + 1 == len(segments) else \
                    self.head(segments[k + 1]) + passing[k + 1]
                if segment or state != target:
                    parts.append(self.block(segment, state, target))
                state = target
            else:
                parts.append(self.flow(segment, passing[k], leaving[k]))
                state = segment.outs + leaving[k]
        return _sequence(parts)

    def flow(self, flow: Loop | Cond, passing: list[int], leaving: list[int]) -> list:
        if isinstance(flow, Cond):
            return ["cond", *[self.region(branch, passing, results + leaving)
                              for branch, results in zip(flow.branches, flow.results)]]
        frame, exit = flow.slots + passing, flow.results + passing
        if flow.test is None:
            return ["scan", flow.length, self.region(flow.body, frame, exit)]
        return ["while", self.region(flow.test, frame, [flow.holds]),
                self.region(flow.body, frame, exit)]

    def block(self, run: list[int], entry: list[int], exit: list[int]) -> list:
        """The block of the nodes ``run`` from the state ``entry`` to the state ``exit``: one input
        per value of ``entry``, the constants that the nodes and ``exit`` read, each before its
        first reader, and the nodes, renumbered."""
        local: list[dict[str, Any]] = []
        number: dict[int, int] = {}
        for j in entry:
            number.setdefault(j, len(local))
            local.append(dict(id=len(local), kind="input", parents=[],
                              shape=self.nodes[j]["shape"], dtype=self.nodes[j]["dtype"]))

        def read(j: int) -> int:
            if j not in number:
                if self.nodes[j]["kind"] != "const":
                    raise NotImplementedError(f"IR import: node {j} is not in a block's state")
                number[j] = len(local)
                local.append(dict(self.nodes[j], id=len(local)))
            return number[j]

        for i in run:
            parents = [read(j) for j in self.nodes[i].get("parents", [])]
            number[i] = len(local)
            local.append(dict(self.nodes[i], id=len(local), parents=parents))
        outputs = [read(j) for j in exit]
        artifact, refs, _ = _artifact.lower(local, list(range(len(entry))), outputs, self.table)
        for j, k in number.items():
            self.where.setdefault(j, (self.blocks, refs[k]))
        self.blocks += 1
        return ["block", artifact]


def entry(shape: list[int], element: str, code: int | None = None) -> list[dict[str, Any]]:
    """The fields of the registered ``Op.gather`` that reads entry ``i``, an int64 scalar, of a
    tensor of shape ``shape`` at ``element`` along its leading axis, or, given the scatter rule
    ``code``, of the ``Op.scatter`` that writes a value there: one index of the leading axis. A
    scan reads its rows and writes its stacked outputs by these in each importer, PyTorch's
    ``map`` and JAX's ``scan``."""
    fields = [{"shape": list(shape)}, {"shape": []}]
    if code is None:
        return fields + [{"nats": [0]}, {"pairs": []}, {"element": element}]
    return fields + [{"nat": code}, {"nats": [0]}, {"element": element}]

def _sequence(parts: list) -> list:
    """The flow that runs the flows ``parts`` in order, ``["skip"]`` if there are none: a ``seq``
    of the flows of their first and second halves. ``Flow.eval`` runs ``seq first second`` as
    ``first`` and then ``second`` on its result, so every bracketing runs the parts in order, and
    ``Flow.blocks`` lists their blocks in order. Halves nest the parts about log2 of their number
    deep, so a region of many, such as a scan of a cond unrolled, nests a few levels, not as many
    as it has parts, which would pass Python's recursion limit in ``json`` and ``copy``."""
    if not parts:
        return ["skip"]
    if len(parts) == 1:
        return parts[0]
    half = len(parts) // 2
    return ["seq", _sequence(parts[:half]), _sequence(parts[half:])]


def lower(nodes: list, input_ids: list[int], output_ids: list[int], region: list, table: dict,
          fuel: int) -> tuple[dict, dict[int, tuple[int, list[int]]]]:
    """The flow artifact of the region ``region`` of the internal nodes ``nodes``, from the
    inputs ``input_ids`` to the outputs ``output_ids``, under the op table ``table``, each
    ``whileLoop`` running at most ``fuel`` iterations each time it is entered; and the block (by
    the order of Lean's ``Flow.blocks``) and the ``[step, output]`` reference of each value it
    holds, in the first block holding it. The region is pruned first (``prune``)."""
    lowering = _Lowering(nodes, table)
    flow = lowering.region(prune(nodes, region, output_ids), list(input_ids), list(output_ids))
    artifact = {"format": table["format"], "version": table["flow_version"], "fuel": fuel,
                "flow": flow}
    return artifact, lowering.where

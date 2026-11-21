"""The reference outputs of a JAX function, its evaluation one primitive at a time, and the trip
counts of while loops that the importer decides at capture.

``references`` computes the reference fields of the capture document that
``torchlean.jaxpr.export`` writes for the same function and examples, ``evaluate`` the outputs
that eager JAX computes without ``jax.jit``'s rewrites, and ``trips`` the iterations of a
``while`` loop for the importer, from the values that ``known_value`` folds, when ``exact``
admits every primitive that decides them. ``torchlean.jaxpr`` exports the first two, and
imports ``trips`` for its own use. The importer's tables and helpers are read from
``torchlean.jaxpr`` when a function runs, since that module imports this one.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import numpy as np

# The primitives whose integer and Boolean results ``known_value`` folds: shape operations,
# comparisons, selections and the integer arithmetic that XLA computes modulo the word.
_FOLDED = frozenset({"squeeze", "reshape", "broadcast_in_dim", "slice", "concatenate",
                     "convert_element_type", "add", "sub", "mul", "neg", "max", "min", "select_n",
                     "eq", "ne", "lt", "le", "gt", "ge", "and", "or", "not"})
_EXACT = frozenset({"bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
                    "uint64"})
# The primitives that move their operand's words, or flip their sign bits (IEEE 754-2019 5.5.1),
# and compute nothing else, at every type: XLA gives their results word for word, so ``moved``
# admits them for the constant kernel of a convolution.
_MOVED = frozenset({"squeeze", "reshape", "broadcast_in_dim", "transpose", "rev", "slice",
                    "concatenate", "neg", "conj"})
# The most iterations of while loops, nested ones included, that one import unrolls.
_WHILE_TRIPS = 1 << 16


def _closed(body: Any) -> Any:
    """A call body as a closed Jaxpr (an open one is closed with no constants)."""
    from jax.extend.core import ClosedJaxpr
    return body if hasattr(body, "consts") else ClosedJaxpr(body, ())


def call_body(equation: Any) -> Any:
    """The closed Jaxpr of the body of a call equation (``_CALLS``). A high-level primitive's
    (``call_hi_primitive``) is the Jaxpr of its ``expand``, which JAX evaluates and lowers in its
    place."""
    from .jaxpr import _CALLS

    held = equation.params[_CALLS[equation.primitive.name]]
    if not hasattr(held, "expand"):
        return _closed(held)

    def expanded(*flat: Any) -> list[Any]:
        return jax.tree_util.tree_leaves(
            held.expand(*jax.tree_util.tree_unflatten(held.in_tree, flat)))
    return jax.make_jaxpr(expanded)(*held.in_avals_flat)


def _calls_jit(jaxpr: Any) -> bool:
    from .jaxpr import _COMPILED, _TRANSPARENT

    for equation in jaxpr.eqns:
        name = equation.primitive.name
        if name in _COMPILED:
            return True
        if name in _TRANSPARENT and _calls_jit(call_body(equation).jaxpr):
            return True
    return False


def evaluate(closed_jaxpr: Any, *arguments: Any) -> list[Any]:
    """Evaluate a closed Jaxpr one primitive at a time, inside nested ``jax.jit`` calls, the
    branch that each ``cond`` selects and each iteration of a ``scan`` or a ``while`` loop too.

    Each primitive is dispatched on its own, so XLA compiles it as a computation of its own, as
    eager JAX does outside ``jax.jit``, and no rewrite spans two primitives. The captured dtypes
    are preserved even if a local ``jax.enable_x64`` context has ended since tracing.
    """
    with jax.enable_x64():
        return _evaluate(closed_jaxpr, list(arguments), closed_jaxpr.jaxpr.eqns)


def _evaluate(closed_jaxpr: Any, arguments: list[Any], equations: list[Any]) -> list[Any]:
    """The outputs of a closed Jaxpr that ``equations``, all or some of its equations in order,
    compute from ``arguments``, as ``evaluate`` computes them; None for the others."""
    from .jaxpr import _CALLS

    jaxpr = closed_jaxpr.jaxpr
    env: dict[Any, Any] = dict(zip(jaxpr.constvars, closed_jaxpr.consts))
    env.update((var, jax.numpy.asarray(value, dtype=var.aval.dtype) if value is not None else None)
               for var, value in zip(jaxpr.invars, arguments))

    def read(atom: Any) -> Any:
        return (jax.numpy.asarray(atom.val, dtype=atom.aval.dtype) if hasattr(atom, "val")
                else env.get(atom))

    for equation in equations:
        name = equation.primitive.name
        operands = [read(atom) for atom in equation.invars]
        if name in _CALLS:
            outputs = evaluate(call_body(equation), *operands)
        elif name == "cond":
            # The branch that StableHLO's case selects, one primitive at a time.
            branches, index = equation.params["branches"], int(operands[0])
            outputs = evaluate(branches[index if 0 <= index < len(branches) else -1],
                               *operands[1:])
        elif name == "scan":
            # The body once per iteration, one primitive at a time, as lax.scan's reference
            # implementation loops over it.
            params = equation.params
            fixed, carry, scanned = map(list, params["ft_in"].update(operands).unpack())
            pieces: list[list[Any]] = [[] for _ in equation.outvars[len(carry):]]
            for i in reversed(range(params["length"])) if params["reverse"] else \
                    range(params["length"]):
                carry, ys = map(list, params["ft_out"].update(evaluate(
                    _closed(params["jaxpr"]), *fixed, *carry,
                    *[jax.numpy.asarray(x)[i] for x in scanned])).unpack())
                for stacked, y in zip(pieces, ys, strict=True):
                    stacked.append(y)
            outputs = [*carry, *[jax.numpy.stack(stacked[::-1] if params["reverse"] else stacked)
                                 if stacked else jax.numpy.zeros(var.aval.shape, var.aval.dtype)
                                 for stacked, var in zip(pieces, equation.outvars[len(carry):])]]
        elif name == "while":
            # The predicate and the body in turn, one primitive at a time.
            params = equation.params
            split = params["cond_nconsts"]
            start = split + params["body_nconsts"]
            outputs = operands[start:]
            while bool(evaluate(params["cond_jaxpr"], *operands[:split], *outputs)[0]):
                outputs = evaluate(params["body_jaxpr"], *operands[split:start], *outputs)
        else:
            outputs = equation.primitive.bind(*operands, **equation.params)
            if not equation.primitive.multiple_results:
                outputs = [outputs]
        env.update(zip(equation.outvars, outputs))
    return [read(atom) for atom in jaxpr.outvars]


def _reads(jaxpr: Any, wanted: list[bool]) -> tuple[list[bool], list[Any]]:
    """Which inputs of the open Jaxpr ``jaxpr`` its ``wanted`` outputs read, and the equations
    that compute them, in order: an equation is kept when one of its outputs is read, and then it
    reads all its operands."""
    live = {atom for atom, read in zip(jaxpr.outvars, wanted, strict=True)
            if read and not hasattr(atom, "val")}
    kept = []
    for equation in reversed(jaxpr.eqns):
        if any(var in live for var in equation.outvars):
            kept.append(equation)
            live.update(atom for atom in equation.invars if not hasattr(atom, "val"))
    return [var in live for var in jaxpr.invars], kept[::-1]


def foldable(equation: Any) -> bool:
    """Whether ``known_value`` may fold the result of ``equation``: one integer or Boolean result
    of a primitive of ``_FOLDED`` on integer or Boolean operands."""
    return equation.primitive.name in _FOLDED and len(equation.outvars) == 1 and all(
        str(atom.aval.dtype) in _EXACT for atom in [*equation.invars, *equation.outvars])


def moved(equation: Any) -> bool:
    """Whether ``known_value`` may fold the result of ``equation`` into a convolution's kernel
    (``_Builder.fixed``): one result of a primitive of ``_MOVED``, of any type."""
    return equation.primitive.name in _MOVED and len(equation.outvars) == 1


def exact(equation: Any) -> bool:
    """Whether XLA computes the results of ``equation`` as the IR does: ``foldable`` admits it, or
    it is a call whose body's equations all are exact. Integer arithmetic modulo the word,
    comparisons, selections and shape operations have one result in both, while floating
    arithmetic does not: XLA's CPU code flushes subnormals to zero and may divide by multiplying
    by a reciprocal, where the IR rounds as IEEE 754 does. A ``cond``, ``scan`` or ``while`` is
    not exact, so ``trips`` never runs a nested loop."""
    from .jaxpr import _CALLS

    name = equation.primitive.name
    if name in _CALLS:
        return all(exact(inner) for inner in call_body(equation).jaxpr.eqns)
    return foldable(equation)


def known_value(index: int, known: dict[int, Any],
                producers: dict[int, tuple[Any, list[int]]]) -> Any:
    """The value of the importer's node ``index`` at capture, or None when it is not known: a
    constant's (``known``), or the result of an equation of ``producers`` on operand nodes whose
    values are known, which JAX computes on its own and ``known`` keeps. The importer's producers
    are the equations that ``foldable`` admits, and for a convolution's kernel those that ``moved``
    admits too.
    The producers are visited from a stack, operands first, not by recursion, so that a chain as
    long as the counter of a loop of ``_WHILE_TRIPS`` iterations is folded."""
    pending, entered = [index], set()
    while pending:
        node = pending.pop()
        if node in known or node not in producers:
            continue
        equation, parents = producers[node]
        if node not in entered:
            # Visited again once its operands are, which an operand already entered never is.
            entered.add(node)
            pending.append(node)
            pending.extend(parent for parent in parents if parent not in entered)
            continue
        values = [known.get(parent) for parent in parents]
        if all(value is not None for value in values):
            with jax.enable_x64(), jax.ensure_compile_time_eval():
                known[node] = np.asarray(equation.primitive.bind(*values, **equation.params))
    return known.get(index)


def unknowns(index: int, known: dict[int, Any],
             producers: dict[int, tuple[Any, list[int]]]) -> frozenset[int]:
    """The nodes at which ``known_value``'s fold of node ``index`` through ``producers`` stops for
    want of a value: those it reaches that are neither known nor produced, none where the value of
    ``index`` is known. It is known wherever each of them is."""
    stops, pending, seen = set(), [index], set()
    while pending:
        node = pending.pop()
        if node in seen or node in known:
            continue
        seen.add(node)
        if node in producers:
            pending.extend(producers[node][1])
        else:
            stops.add(node)
    return frozenset(stops)


def tested(equation: Any) -> tuple[list[bool], list[Any], list[Any]]:
    """The operands of the ``while`` equation ``equation`` that decide its iterations, as a mask
    over them, with the equations of its predicate and of its body that compute from them. The
    predicate reads some of its constants and carry entries, and the body computes those entries
    from more of its constants and entries, until no more are added."""
    params = equation.params
    test, step = params["cond_jaxpr"], params["body_jaxpr"]
    split = params["cond_nconsts"]
    start = split + params["body_nconsts"]
    # `decided` marks the carry entries the predicate depends on, grown to a fixed point
    # through the body.
    predicate, test_equations = _reads(test.jaxpr, [True])
    decided = predicate[split:]
    while True:
        read, step_equations = _reads(step.jaxpr, decided)
        wider = [a or b for a, b in zip(decided, read[start - split:], strict=True)]
        if wider == decided:
            break
        decided = wider
    return [*predicate[:split], *read[:start - split], *decided], test_equations, step_equations


def trips(equation: Any, known: list[Any], limit: int) -> int | None:
    """The number of iterations that the ``while`` equation ``equation`` runs, counted up to
    ``limit + 1``, when the values of its operands that ``known`` holds (None where a value is not
    known at capture) decide it, or None. Each operand that decides it (``tested``) must be known,
    and each equation of the predicate and of the parts of the body that compute them ``exact``,
    so that the count is the one the IR's arithmetic gives. They run in turn, one primitive at a
    time as in ``evaluate``, until the predicate is false or ``limit``, the import's remaining
    share of ``_WHILE_TRIPS``, is exceeded. A loop whose predicate reads an unknown value or which
    needs an inexact equation is not decided."""
    params = equation.params
    test, step = params["cond_jaxpr"], params["body_jaxpr"]
    split = params["cond_nconsts"]
    start = split + params["body_nconsts"]
    needed, test_equations, step_equations = tested(equation)
    if any(need and value is None for need, value in zip(needed, known, strict=True)):
        return None
    if not all(exact(equation) for equation in [*test_equations, *step_equations]):
        return None
    # Run the loop on those entries alone, the others left unknown.
    values = [value if need else None for need, value in zip(needed, known)]
    state, count = values[start:], 0
    with jax.enable_x64(), jax.ensure_compile_time_eval():
        while bool(_evaluate(test, [*values[:split], *state], test_equations)[0]):
            count += 1
            if count > limit:
                break
            state = _evaluate(step, [*values[split:start], *state], step_equations)
    return count


def _platform(values: list[Any]) -> str:
    platforms = {device.platform for value in values if hasattr(value, "devices")
                 for device in value.devices()}
    return ",".join(sorted(platforms)) if platforms else "unknown"


def references(function: Callable[..., Any], *examples: Any) -> dict[str, Any]:
    """The reference outputs of ``function(*examples)``, with how they were computed.

    ``reference_outputs`` are the outputs' words, ``reference_platform`` the platform of the devices
    that computed them (``"cpu"`` for XLA's CPU backend) and ``jaxlib_version`` the version of
    jaxlib. When ``jax.jit`` compiles any part of the function, or the function holds a ``cond``,
    a ``scan`` or a ``while`` loop, which XLA compiles whole, the result also holds
    ``eager_reference_outputs``:
    the outputs of ``evaluate``, one primitive at a time, which XLA computes without ``jax.jit``'s
    rewrites. Their platform is in ``reference_platform`` too. Merge
    the result into the document that ``export`` writes for the same function and examples.

    The examples and the output leaves are taken as JAX arrays (``canonical``), so a NumPy float64
    leaf is the float32 value the trace computes when 64-bit mode is off. An output whose element
    type or shape is not its traced output's is refused with ``Unsupported``.
    """
    from .jaxpr import canonical

    examples = canonical(examples)
    result = function(*examples)
    return reference_outputs(jax.make_jaxpr(function)(*examples), examples, result)


def reference_outputs(closed: Any, examples: list[Any], result: Any) -> dict[str, Any]:
    """Record the frontend result and, for compiled regions, replay the same captured Jaxpr."""
    import jaxlib

    from .jaxpr import Unsupported, _dtype, _shape, canonical, encode

    outputs = canonical(jax.tree.leaves(result))
    traced = closed.jaxpr.outvars
    if len(outputs) != len(traced):
        raise Unsupported(f"the function returns {len(outputs)} tensors, its trace {len(traced)}")
    for i, (value, var) in enumerate(zip(outputs, traced)):
        if _dtype(value) != _dtype(var.aval) or _shape(value) != _shape(var.aval):
            raise Unsupported(f"reference output {i} is {_dtype(value)} {_shape(value)}, "
                              f"the traced output is {_dtype(var.aval)} {_shape(var.aval)}")
    record: dict[str, Any] = dict(reference_outputs=[encode(value) for value in outputs],
                                  jaxlib_version=jaxlib.__version__)
    computed = list(outputs)
    if _calls_jit(closed.jaxpr):
        eager = canonical(evaluate(closed, *examples))
        record["eager_reference_outputs"] = [encode(value) for value in eager]
        computed += eager
    record["reference_platform"] = _platform(computed)
    return record

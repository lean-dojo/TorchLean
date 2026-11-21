"""The importer's contractions: ``dot_general``, ``ragged_dot_general``, ``scaled_dot`` and
``conv_general_dilated``.

``ContractionImport`` is the part of ``torchlean.jaxpr``'s builder that imports a primitive
contracting its operands over paired axes, with the IR's ``contract`` and ``conv`` (``widened``),
or with products summed by ``wrapping`` or ``reduce_or`` for integer and Boolean operands. The
importer's errors and helpers are read from ``torchlean.jaxpr`` when a primitive is imported, since
that module imports this one.
"""

from __future__ import annotations

from typing import Any

import jax
import numpy as np

from . import _artifact


class ContractionImport:
    """The contraction primitives of ``torchlean.jaxpr``'s builder."""

    def dot(self, left: int, right: int, params: dict[str, Any], shape: list[int]) -> int:
        """One contraction over the paired axes, keeping JAX's output order: batch axes, then
        the free axes of the left and of the right operand. JAX leaves the accumulation type of
        a contraction without ``preferred_element_type`` to the backend; binary16 and bfloat16
        operands are contracted in binary32 and the result rounded once, as XLA's CPU backend
        does (``widened``), not at each step in the operands' format, and float8 operands of a
        float8 result in binary32 rounded once to the format, where XLA's CPU code rounds through
        binary16 first. Operands of another type than the result are converted to it first
        (``promoted``). Integer and Boolean operands are contracted as StableHLO defines
        ``dot_general`` (its spec, "dot_general"): each result entry reduces the products of the
        slices' entries with ``add`` from 0, and for Booleans ``add`` and ``multiply`` are logical
        or and and (its "add" and "multiply"). StableHLO leaves integer overflow
        implementation-defined (its "Errors" section), so wrapping modulo ``2 ^ width``, as in
        ``wrapping``, is the importer's choice. The operands are moved to batch, left free, right
        free and contracted axes, broadcast to that joint shape and multiplied there, and the
        contracted axes summed by ``wrapping``'s tree or ``reduce_or``."""
        from .jaxpr import Unsupported, _is_integer
        precision = params.get("precision")
        defaults = (None, (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
                    jax.lax.DotAlgorithmPreset.DEFAULT)
        if precision not in defaults:
            raise Unsupported("dot_general: an explicit precision or dot algorithm needs an "
                              "operand/accumulator arithmetic rule; the IR contraction does not "
                              f"implement {precision}")
        (lc, rc), (lb, rb) = params["dimension_numbers"]
        lc, rc, lb, rb = map(list, (lc, rc, lb, rb))
        ls, rs = self.shape(left), self.shape(right)
        for axes, rank in ((lb + lc, len(ls)), (rb + rc, len(rs))):
            if len(set(axes)) != len(axes) or any(i < 0 or i >= rank for i in axes):
                raise Unsupported("dot_general: invalid batch or contraction axes")
        if [ls[i] for i in lb] != [rs[i] for i in rb]:
            raise Unsupported("dot_general: batch dimensions differ")
        if [ls[i] for i in lc] != [rs[i] for i in rc]:
            raise Unsupported("dot_general: contraction dimensions differ")
        # The free axes, in operand order.
        lf = [i for i in range(len(ls)) if i not in lb + lc]
        rf = [i for i in range(len(rs)) if i not in rb + rc]
        expected = [ls[i] for i in lb] + [ls[i] for i in lf] + [rs[i] for i in rf]
        if expected != shape:
            raise Unsupported(f"dot_general: expected output {expected}, got {shape}")
        if _is_integer(self.current_dtype) or self.current_dtype == "bool":
            batch, inner = [ls[i] for i in lb], [ls[i] for i in lc]
            free = ([ls[i] for i in lf], [rs[i] for i in rf])
            # Each operand gets size-one axes for the other's free axes, so both broadcast to
            # `joint`, and the trailing contracted axes are summed.
            joint = batch + free[0] + free[1] + inner
            a = self.reshape(self.permute(left, lb + lf + lc),
                             batch + free[0] + [1] * len(rf) + inner)
            b = self.reshape(self.permute(right, rb + rf + rc),
                             batch + [1] * len(lf) + free[1] + inner)
            a, b = self.broadcast(a, joint), self.broadcast(b, joint)
            axes = list(range(len(expected), len(joint)))
            if self.current_dtype == "bool":
                product = self.node("logical_and", [a, b], joint, dtype="bool")
                return self.operation("reduce_or", [product], dict(axes=axes), shape, np.bool_)
            product = self.node("mul_elem", [a, b], joint)
            return self.reshape(self.wrapping("reduce_sum", product, axes), shape)
        return self.widened("contract", [left, right], shape,
                            dims=[list(p) for p in zip(lc, rc)],
                            batch_dims=[list(p) for p in zip(lb, rb)])

    def group_bounds(self, counts: int, sizes: Any) -> list[tuple[int, int]]:
        """The start and end of each group of ``ragged_dot_general``'s group sizes ``n`` of shape
        ``p + [g]``: at each position of ``p``, group ``j`` holds the positions
        ``n_0 + ... + n_(j-1) ≤ i < n_0 + ... + n_j`` of the ragged axis, the sizes summed in
        their integer type as JAX's reference implementation does."""
        kind, prefix = self.nodes[counts]["dtype"], self.shape(counts)[:-1]
        end, bounds, ones = self.scalar(0, sizes, prefix), [], [1] * (len(prefix) + 1)
        for j in range(self.shape(counts)[-1]):
            size = self.reshape(self.slice(counts, [0] * len(prefix) + [j], prefix + [j + 1],
                                           ones), prefix)
            start, end = end, self.node("add", [end, size], prefix, dtype=kind)
            bounds.append((start, end))
        return bounds

    def inside(self, bounds: list[tuple[int, int]], sizes: Any, shape: list[int], axis: int,
               prefix: list[int]) -> list[int]:
        """For each group of ``bounds``, whether each entry of ``shape`` lies in it along
        ``axis``, axis ``i`` of the bounds placed at axis ``prefix[i]`` of ``shape``."""
        order = sorted(range(len(prefix)), key=prefix.__getitem__)
        positions = self.broadcast(self.iota([shape[axis]], 0, sizes), shape, [axis])

        def placed(bound: int) -> int:
            return self.broadcast(self.permute(bound, order), shape, sorted(prefix))
        return [self.node("logical_and", [
            self.node("le", [placed(start), positions], shape, dtype="bool"),
            self.node("lt", [positions, placed(end)], shape, dtype="bool")],
            shape, dtype="bool") for start, end in bounds]

    def ragged_dot(self, parents: list[int], params: dict[str, Any], shape: list[int],
                   dtype: Any, sizes: Any) -> int:
        """``ragged_dot_general`` in its three modes, with the groups of ``group_bounds``. JAX
        documents only the shapes of the three modes, so the meaning of each is TorchLean's reading,
        compared below with JAX's reference implementation (``_ragged_dot_general_impl``). With a
        ragged free axis of the left operand, the form ``lax.ragged_dot`` binds, the result at
        position ``i`` is the contraction with group ``j`` of the right operand: one contraction
        with every group, and one selection per group. A position in no group is ``+0``, and one in
        several (a negative size makes groups overlap) takes the first. JAX's reference
        implementation, which XLA's CPU backend runs, instead contracts with every group, the left
        operand masked to zero outside it, so an infinity or a NaN in one group of the right operand
        makes that column NaN in every group, and overlapping groups add: the import differs from it
        there, by TorchLean's choice of the selection, and refuses neither case. With a ragged
        contracting axis, entry ``j`` of the new leading axis is the contraction with both operands
        set to zero outside group ``j``, as in JAX's reference implementation, so a position adds to
        each group that holds it. With a ragged batch axis the result is ``dot_general``'s and the
        sizes are not read, as in JAX's reference implementation. Group sizes of shape ``p + [g]``
        give each position of the prefix axes their own groups, as JAX's reference implementation
        does: the batch axes, then the left operand's free axes before the ragged one (its
        contracting axes before it, in the order of the contracting dimension numbers, for a ragged
        contracting axis). Sizes of shape ``[g]`` are the same at every position. A group offset is
        refused: JAX 0.11.1 cannot bind an array as that parameter, which must be hashable, and its
        reference implementation raises ``NotImplementedError`` for one. A float8 result is that of
        the operands converted exactly to binary32, contracted, selected and set to zero there, and
        rounded once to the format, as ``widened`` rounds a float8 ``dot_general``. XLA's CPU code
        rounds it twice, through binary16, in each mode, as it rounds a float8 ``dot_general``
        (scripts/checks/ir_capture_jax_contraction.py,
        test_float8_contractions_convert_their_operands)."""
        from .jaxpr import Unsupported, _is_integer
        numbers = params["ragged_dot_dimension_numbers"]
        (lc, rc), (lb, rb) = numbers.dot_dimension_numbers
        ragged, groups = numbers.lhs_ragged_dimensions[0], list(numbers.rhs_group_dimensions)
        left, right, counts = parents
        if params.get("group_offset") is not None:
            raise Unsupported("ragged_dot_general: a group offset is not imported")
        if not _is_integer(self.nodes[counts]["dtype"]):
            raise Unsupported("ragged_dot_general: group sizes must be integers")
        left, right = [self.promoted(i, "ragged_dot_general") for i in (left, right)]
        target = self.current_dtype
        if isinstance(target, dict) and target.get("kind") == "binary":
            self.current_dtype = "float32"
            wide = self.ragged_dot([self.converted(i, "float32") for i in (left, right)] + [counts],
                                   params, shape, np.dtype(np.float32), sizes)
            self.current_dtype = target
            return self.converted(wide, target)
        dims = dict(precision=params.get("precision"),
                    dimension_numbers=numbers.dot_dimension_numbers)
        if ragged in lb:
            return self.dot(left, right, dims, shape)
        ls, rs = self.shape(left), self.shape(right)
        lf = [i for i in range(len(ls)) if i not in [*lb, *lc]]
        rf = [i for i in range(len(rs)) if i not in [*rb, *rc]]
        # The axes of the left operand that the group sizes' leading axes run along.
        before = list(lc[:list(lc).index(ragged)]) if ragged in lc else lf[:lf.index(ragged)]
        each = len(self.shape(counts)) > 1
        prefix = [*lb, *before] if each else []
        if each and self.shape(counts)[:-1] != [ls[i] for i in prefix]:
            raise Unsupported(f"ragged_dot_general: group sizes of shape {self.shape(counts)}")
        bounds = self.group_bounds(counts, sizes)
        if ragged in lc:
            pieces, opposite = [], [*rb, *rc[:len(before)]] if each else []
            for masks in zip(self.inside(bounds, sizes, ls, ragged, prefix),
                             self.inside(bounds, sizes, rs, rc[list(lc).index(ragged)],
                                         opposite)):
                pieces.append(self.reshape(self.dot(*(
                    self.node("where", [mask, i, self.scalar(0, dtype, self.shape(i))],
                              self.shape(i)) for mask, i in zip(masks, (left, right))),
                    dims, shape[1:]), [1] + shape[1:]))
            if len(pieces) < 2:
                return pieces[0] if pieces else self.scalar(0, dtype, shape)
            return self.node("cat", pieces, shape, dim=0)
        # The contraction with every group has the group axis among the right free axes.
        group, axis = len(lb) + len(lf) + rf.index(groups[0]), len(lb) + lf.index(ragged)
        every = shape[:group] + [rs[groups[0]]] + shape[group:]
        product = self.dot(left, right, dims, every)
        cases = []
        placed = [*range(len(lb)), *(len(lb) + lf.index(i) for i in before)] if each else []
        for j, inside in enumerate(self.inside(bounds, sizes, shape, axis, placed)):
            starts, stops = [0] * len(every), list(every)
            starts[group], stops[group] = j, j + 1
            cases.append((inside, self.reshape(self.slice(product, starts, stops,
                                                          [1] * len(every)), shape)))
        result = self.scalar(0, dtype, shape)
        for inside, value in reversed(cases):
            result = self.node("where", [inside, value, result], shape)
        return result

    def scaled_dot(self, parents: list[int], params: dict[str, Any], shape: list[int]) -> int:
        """``scaled_dot`` as JAX defines it, by the decomposition of its ``xla.scaled_dot``
        composite (``jax._src.lax.scaled_dot._scaled_dot_impl``): along each contracting axis,
        entry ``k`` of a scale covers the ``m`` positions from ``k * m`` of its operand, each
        operand and its scale are rounded to bfloat16 and multiplied there, and the products are
        contracted as ``dot_general`` is, converted to a ``preferred_element_type`` other than
        bfloat16 first, as the composite passes that type to ``dot_general`` (``promoted``). A
        float8 operand or scale (``_FLOAT8``) is a value of bfloat16, whose precision and exponent
        range cover each of those formats, so its rounding to bfloat16 is exact."""
        from .jaxpr import Unsupported, _FLOATING, _holds
        if not _holds("bfloat16", self.current_dtype):
            raise Unsupported(f"scaled_dot: a {_artifact._describe(self.current_dtype)} result "
                              "does not hold every bfloat16 product; only exact widenings are "
                              "imported")
        if any(self.nodes[i]["dtype"] not in _FLOATING and not (
                isinstance(self.nodes[i]["dtype"], dict)
                and self.nodes[i]["dtype"].get("kind") == "binary") for i in parents):
            raise Unsupported("scaled_dot: only floating operands and scales are imported")
        (lc, rc), _ = params["dimension_numbers"]
        products = []
        for operand, scale, axes in ((parents[0], parents[2], lc), (parents[1], parents[3], rc)):
            full = self.shape(operand)
            for axis in axes:
                kept = self.shape(scale)
                if kept[axis] != full[axis]:
                    spread = kept[:axis + 1] + [full[axis] // kept[axis]] + kept[axis + 1:]
                    scale = self.reshape(self.broadcast(scale, spread, [
                        d + (d > axis) for d in range(len(kept))]),
                        kept[:axis] + [full[axis]] + kept[axis + 1:])
            products.append(self.node("mul_elem", [
                i if self.nodes[i]["dtype"] == "bfloat16" else self.node(
                    "custom", [i], full, dtype="bfloat16", primitive="cast",
                    rounding="nearestEven")
                for i in (operand, scale)], full, dtype="bfloat16"))
        return self.dot(*[self.promoted(i, "scaled_dot") for i in products], params, shape)

    def convolution(self, parents: list[int], params: dict[str, Any], shape: list[int]) -> int:
        """An IR convolution, batch and feature axes leading.

        The IR node keeps a kernel known at capture (``fixed``) as its payload and takes any
        other kernel as its second operand. Binary16 and bfloat16 operands are convolved in
        binary32 and the result rounded once, as XLA's CPU backend does (``widened``), and float8
        operands of a float8 result in binary32 rounded once to the format, where XLA's CPU code
        rounds through binary16 first. Operands of another type than the result are converted to
        it first, the input and a kernel operand by ``promoted`` and a payload's words exactly,
        as XLA's CPU code converts them, a float8 kernel's too. A padding cell is a zero
        multiplied by its kernel entry, as StableHLO pads the input before its dot products
        (spec.md:2410-2411), so an infinite or NaN entry over padding gives NaN; XLA's CPU code
        skips the padding at some shapes (``TorchLean/NN/IR/README.md``). Other layouts, input
        dilation and batch groups need the general convolution."""
        from .jaxpr import Unsupported, _REDUCED_FLOATS, _binary, _holds
        source, kernel = parents
        numbers = params["dimension_numbers"]
        rank = len(self.shape(source)) - 2
        identity = tuple(range(rank + 2))
        if (tuple(numbers.lhs_spec), tuple(numbers.rhs_spec), tuple(numbers.out_spec)) != \
                (identity, identity, identity):
            raise Unsupported("conv_general_dilated: only the NC..., OI... and NC... layouts "
                              "are imported")
        if (any(int(d) != 1 for d in params["lhs_dilation"])
                or int(params["batch_group_count"]) != 1):
            raise Unsupported("conv_general_dilated: input dilation and batch groups need the "
                              "general convolution")
        if params.get("precision") is not None:
            raise Unsupported("conv_general_dilated: explicit precision needs an arithmetic rule")
        source = self.promoted(source, "conv_general_dilated")
        weights = self.nodes[kernel]
        if weights["kind"] != "const" and (value := self.fixed(kernel)) is not None:
            kernel = self.constant(value, jax.core.ShapedArray(value.shape, value.dtype))
            weights = self.nodes[kernel]
        padding = [(int(a), int(b)) for a, b in params["padding"]]
        if any(a < 0 or b < 0 for a, b in padding):
            raise Unsupported("conv_general_dilated: negative padding needs a slice first")
        groups = int(params["feature_group_count"])
        out_channels, per_group = weights["shape"][0], weights["shape"][1]
        result = self.current_dtype
        # The arithmetic's type, binary32 for a reduced or a float8 result (``widened``).
        kind = "float32" if result in _REDUCED_FLOATS or isinstance(result, dict) and \
            result.get("kind") == "binary" else result
        attrs = dict(spatial_rank=rank, kernel=list(weights["shape"][2:]),
                     stride=[int(s) for s in params["window_strides"]],
                     padding=[a for a, _ in padding], padding_after=[b for _, b in padding],
                     dilation=[int(d) for d in params["rhs_dilation"]], groups=groups,
                     channel_axis=1, input_spatial=self.shape(source)[2:],
                     in_channels=per_group * groups, out_channels=out_channels,
                     bias_bits=[0] * out_channels, bias_dtype=kind)
        if weights["kind"] != "const":
            # JAX's kernel layout, output channels then each group's input channels, is the
            # IR's operand (``_artifact._project``).
            return self.widened("conv", [source, self.promoted(kernel, "conv_general_dilated")],
                                shape, **attrs)
        # A float8 constant keeps only its words (``constant``), and its type is read from the
        # array kept for it.
        float8 = isinstance(weights["dtype"], dict) and weights["dtype"].get("kind") == "binary"
        array = self.known.get(kernel) if float8 else None
        words = np.asarray(weights["bits"], dtype=object).reshape(weights["shape"])
        # The kernel is its words alone, as the artifact reads it (``_artifact._words``). Word 0
        # is +0 in every binary format, the value of the cells outside a group.
        dense_words = np.zeros([out_channels, per_group * groups, *weights["shape"][2:]],
                               dtype=object)
        step = out_channels // groups
        for group in range(groups):
            rows = slice(group * step, (group + 1) * step)
            dense_words[rows, group * per_group:(group + 1) * per_group] = words[rows]
        # The kernel holds its words in the type of the convolution's arithmetic.
        if kind != weights["dtype"]:
            if not ((float8 or _binary(weights["dtype"])) and _binary(kind)
                    and _holds(weights["dtype"], kind)):
                raise Unsupported(f"conv_general_dilated: a {_artifact._describe(kind)} "
                                  f"convolution of a {_artifact._describe(weights['dtype'])} "
                                  "kernel is not an exact widening")
            source_type = array.dtype if float8 else jax.numpy.dtype(weights["dtype"])
            target_type = jax.numpy.dtype(kind)
            unsigned = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}
            dense_words = np.array(dense_words.reshape(-1).tolist(), np.uint64).astype(
                unsigned[source_type.itemsize]).view(source_type).astype(target_type).view(
                unsigned[target_type.itemsize]).reshape(dense_words.shape)
        return self.widened("conv", [source], shape, **attrs,
                            weight_bits=[int(word) for word in dense_words.reshape(-1)],
                            weight_dtype=kind)

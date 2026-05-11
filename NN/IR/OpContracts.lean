/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Graph

/-!
# Operation Contracts

Shared operation contracts for `NN.IR.Graph`.

Several IR passes need to agree on the same small set of “shape contracts”:

- `NN.IR.Infer`: recompute output shapes from op parameters + parent shapes.
- `NN.IR.Check`: expose the documented `Graph.checkShapes` wrapper.
- `NN.IR.Semantics`: evaluate nodes and reject ill-shaped graphs with readable error messages.

The point of this file is to keep shape arithmetic out of individual passes. If an op has nontrivial
shape behavior (concat, matmul, pooling, convolution, LayerNorm flattening, axis moves), define the
contract here first and call it from inference/semantics instead of copying the formula.
-/

@[expose] public section

namespace NN.IR

open Spec

/-!
## Small shape utilities

These helpers are used by multiple IR passes, especially `Infer` and `Semantics`.
-/

namespace ShapeUtil

/-- The output shape of flattening a tensor of shape `s` to a 1D vector. -/
def flattenOutShape (s : Shape) : Shape :=
  .dim (Shape.size s) .scalar

/--
If `s` has rank ≥ 2, return the shape obtained by swapping its first two axes.

Example: `(a, b, rest)` becomes `(b, a, rest)`.
-/
def swapFirstTwoShape? : Shape → Option Shape
  | .dim a (.dim b rest) => some (.dim b (.dim a rest))
  | _ => none

/--
If `s` has shape `(a, b, c)` (rank=3 with scalar base), return `(a, c, b)`.

This is the common “transpose the last two axes” pattern for batched matrices.
-/
def transpose3dLastTwoShape? : Shape → Option Shape
  | .dim a (.dim b (.dim c .scalar)) => some (.dim a (.dim c (.dim b .scalar)))
  | _ => none

end ShapeUtil

namespace OpContracts

/-!
## Generic contract helpers

These functions live outside any particular pass (`Infer`/`Check`/`Semantics`) so they can be
reused without introducing import cycles.
-/

/-- Check that an `axis` is in-bounds for a given shape. -/
def checkAxisValid (axis : Nat) (s : Shape) : Except String Unit := do
  if axis < Shape.rank s then
    pure ()
  else
    throw s!"invalid axis {axis} for rank {Shape.rank s}"

/-- Check that a natural-number op parameter is nonzero. -/
def checkPositive (tag param : String) (n : Nat) : Except String Unit := do
  if n = 0 then
    throw s!"{tag}: {param} must be > 0"
  else
    pure ()

/--
Compute the `(seqLen, embedDim)` pair used to interpret `layernorm axis`.

TorchLean’s IR stores LayerNorm as an `axis : Nat` instead of a full `normalized_shape` tuple.
We interpret this in the same way the PyTorch exporter does:

`normalized_shape = dims.drop axis`

That is, we normalize over the **suffix** of dimensions starting at `axis`. To reuse the current
spec primitive (`Spec.layerNorm`), we flatten the input shape `s` into a 2D view:

* `seqLen`   = product of dimensions *before* `axis` (`dims.take axis`)
* `embedDim` = product of dimensions *from* `axis` onward (`dims.drop axis`)

Then we run 2D last-axis LayerNorm on a `(seqLen × embedDim)` tensor and reshape back.
-/
def layerNorm2DParams (axis : Nat) (s : Shape) : Except String (Nat × Nat) := do
  checkAxisValid axis s
  let dims := Shape.toList s
  let seqLen : Nat := (dims.take axis).foldl (fun acc d => acc * d) 1
  let embedDim : Nat := (dims.drop axis).foldl (fun acc d => acc * d) 1
  pure (seqLen, embedDim)

/--
Check that `axis` refers to the **last** axis of `s`.

This is a convenience predicate for passes/backends that restrict an op to last-axis behavior.
For example, some verification bounds are implemented only for last-axis `softmax`/`layernorm` and
use this check to fail fast with a readable error.
-/
def checkLastAxis (tag : String) (axis : Nat) (s : Shape) : Except String Unit := do
  checkAxisValid axis s
  if axis + 1 = Shape.rank s then
    pure ()
  else
    throw s!"{tag}: only last-axis is supported (axis={axis}, rank={Shape.rank s})"

/--
Compute the inverse of a permutation list.

If `perm` is a permutation of `[0,1,...,r-1]` (where `r = perm.length`), then the inverse `inv`
satisfies `inv[perm[i]] = i`.
-/
def inversePerm (perm : List Nat) : Except String (List Nat) := do
  let r := perm.length
  let mut inv : List (Option Nat) := List.replicate r none

  let rec setOnce (xs : List (Option Nat)) (axis j : Nat) (val : Nat) : Except String (List (Option
    Nat)) := do
    match xs, j with
    | [], _ =>
        throw s!"permute: internal error: index {j} out of range for invLen={r}"
    | none :: rest, 0 =>
        pure (some val :: rest)
    | some _ :: _, 0 =>
        throw s!"permute: duplicate axis {axis} in {repr perm}"
    | x :: rest, Nat.succ k =>
        pure (x :: (← setOnce rest axis k val))

  let rec getOpt : List (Option Nat) → Nat → Option (Option Nat)
    | [], _ => none
    | x :: _, 0 => some x
    | _ :: xs, Nat.succ k => getOpt xs k

  let mut i : Nat := 0
  for p in perm do
    if p < r then
      inv ← setOnce inv p p i
    else
      throw s!"permute: axis {p} out of range for rank {r} in {repr perm}"
    i := i + 1

  let mut outRev : List Nat := []
  for j in [0:r] do
    match getOpt inv j with
    | some (some idx) => outRev := idx :: outRev
    | some none => throw s!"permute: missing axis {j} in {repr perm}"
    | none => throw s!"permute: internal error: missing inv[{j}]"
  pure outRev.reverse

/--
Permutation (0-based axes) that moves `axis` to the **last** position, preserving the relative
order of the other axes.

Example: rank=4 and `axis=1` yields `[0,2,3,1]`.
-/
def permMoveAxisToLast (axis : Nat) (s : Shape) : Except String (List Nat) := do
  checkAxisValid axis s
  let r := Shape.rank s
  pure <| (List.range r).erase axis ++ [axis]

/--
Permutation (0-based axes) that moves `axis` to the **first** position, preserving the relative
order of the other axes.

Example: rank=4 and `axis=2` yields `[2,0,1,3]`.
-/
def permMoveAxisToFront (axis : Nat) (s : Shape) : Except String (List Nat) := do
  checkAxisValid axis s
  let r := Shape.rank s
  pure <| axis :: (List.range r).erase axis

/--
Infer the output shape for `matmul` from the two parent shapes.

Supported cases:
- 2D: `(m×n) · (n×p) → (m×p)`
- limited 3D “batched matmul”: `(b×m×n) · (b×n×p) → (b×m×p)`
-/
def inferMatmulOutShape (a b : Shape) : Except String Shape := do
  match a, b with
  | .dim m (.dim n .scalar), .dim n' (.dim p .scalar) =>
      if _h : n = n' then
        pure (.dim m (.dim p .scalar))
      else
        throw s!"matmul: inner dims mismatch: {n} vs {n'}"
  | .dim batch (.dim m (.dim n .scalar)), .dim batch' (.dim n' (.dim p .scalar)) =>
      if _hb : batch = batch' then
        if _hn : n = n' then
          pure (.dim batch (.dim m (.dim p .scalar)))
        else
          throw s!"matmul: inner dims mismatch: {n} vs {n'}"
      else
        throw s!"matmul: batch dims mismatch: {batch} vs {batch'}"
  | _, _ =>
      throw s!"matmul: unsupported shapes: {repr a} · {repr b}"

/--
Infer the output shape for `concat` from the parent shapes.

All parents must:
- have the same rank,
- agree on every dimension except `axis`, and
- have `axis` in bounds.

The output shape matches the parents except at `axis`, where the dimension is the sum of the input
dimensions.

PyTorch analogy: `torch.cat(xs, dim=axis)` for a list `xs` of tensors.
-/
def inferConcatOutShape (axis : Nat) (parents : List Shape) : Except String Shape := do
  let k := parents.length
  if k < 2 then
    throw s!"concat: expected at least 2 inputs, got {k}"
  let s0 ←
    match parents with
    | [] => throw "concat: internal error"
    | s :: _ => pure s
  checkAxisValid axis s0
  let r0 := Shape.rank s0
  for s in parents do
    if Shape.rank s != r0 then
      throw s!"concat: rank mismatch: expected {r0}, got {Shape.rank s} ({repr s})"

  let rec go (axis : Nat) (shs : List Shape) : Except String Shape := do
    match axis, shs with
    | 0, [] =>
        throw "concat: internal error"
    | 0, s :: rest =>
        match s with
        | .dim n0 tail0 =>
            let mut total : Nat := n0
            for t in rest do
              match t with
              | .dim n tail =>
                  if tail != tail0 then
                    throw <|
                      s!"concat: axis=0 expects matching tail shapes, got {repr tail0} and " ++
                        s!"{repr tail}"
                  total := total + n
              | _ =>
                  throw s!"concat: axis=0 expects rank≥1 inputs, got {repr t}"
            pure (.dim total tail0)
        | _ =>
            throw s!"concat: axis=0 expects rank≥1 inputs, got {repr s}"
    | Nat.succ _, [] =>
        throw "concat: internal error"
    | Nat.succ a, s :: rest =>
        match s with
        | .dim n0 tail0 =>
            let mut tailsRev : List Shape := [tail0]
            for t in rest do
              match t with
              | .dim n tail =>
                  if n != n0 then
                    throw s!"concat: non-axis dim mismatch: expected {n0}, got {n}"
                  tailsRev := tail :: tailsRev
              | _ =>
                  throw s!"concat: axis out of range for input shape {repr t}"
            let outTail ← go a tailsRev.reverse
            pure (.dim n0 outTail)
        | _ =>
            throw s!"concat: axis out of range for input shape {repr s}"

  go axis parents

/-!
## Pooling/Conv2D shape arithmetic (CHW-only)

These formulas mirror the spec/runtime conventions (CHW tensors, no dilation, symmetric padding).
Centralizing them gives inference, evaluation, verification, and export code a shared convention
for convolution and pooling shapes.
-/

/-- Output length for a 1D sliding-window op without padding: `⌊(in - k)/stride⌋ + 1`. -/
def slideOut (inLen k stride : Nat) : Nat :=
  (inLen - k) / stride + 1

/-- Output length for a 1D sliding-window op with symmetric padding: `⌊(in + 2*pad - k)/stride⌋ +
  1`. -/
def slideOutPad (inLen k stride padding : Nat) : Nat :=
  (inLen + 2 * padding - k) / stride + 1

/-- Output shape for CHW pooling without padding. -/
def pool2dCHWOutShape (c inH inW kH kW stride : Nat) : Shape :=
  let outH := slideOut inH kH stride
  let outW := slideOut inW kW stride
  .dim c (.dim outH (.dim outW .scalar))

/-- Output shape for CHW pooling with symmetric padding. -/
def pool2dCHWOutShapePad (c inH inW kH kW stride padding : Nat) : Shape :=
  let outH := slideOutPad inH kH stride padding
  let outW := slideOutPad inW kW stride padding
  .dim c (.dim outH (.dim outW .scalar))

/-- Output shape for CHW conv2d (single-image, no batch dim). -/
def conv2dCHWOutShape (outC inH inW kH kW stride padding : Nat) : Shape :=
  let outH := slideOutPad inH kH stride padding
  let outW := slideOutPad inW kW stride padding
  .dim outC (.dim outH (.dim outW .scalar))

/-- Infer the output shape for CHW pooling without padding, from a parent shape. -/
def inferPool2dCHWOutShape (tag : String) (kH kW stride : Nat) (parent : Shape) : Except String
  Shape := do
  checkPositive tag "kH" kH
  checkPositive tag "kW" kW
  checkPositive tag "stride" stride
  match parent with
  | .dim c (.dim inH (.dim inW .scalar)) =>
      pure (pool2dCHWOutShape c inH inW kH kW stride)
  | s =>
      throw s!"{tag}: expected input shape (C,H,W), got {repr s}"

/-- Infer the output shape for CHW pooling with padding, from a parent shape. -/
def inferPool2dCHWOutShapePad (tag : String) (kH kW stride padding : Nat) (parent : Shape) :
    Except String Shape := do
  checkPositive tag "kH" kH
  checkPositive tag "kW" kW
  checkPositive tag "stride" stride
  match parent with
  | .dim c (.dim inH (.dim inW .scalar)) =>
      pure (pool2dCHWOutShapePad c inH inW kH kW stride padding)
  | s =>
      throw s!"{tag}: expected input shape (C,H,W), got {repr s}"

/-- Infer the output shape for CHW Conv2D, checking the declared `inC` against the parent shape. -/
def inferConv2dCHWOutShape (inC outC kH kW stride padding : Nat) (parent : Shape) : Except String
  Shape := do
  checkPositive "conv2d" "inC" inC
  checkPositive "conv2d" "kH" kH
  checkPositive "conv2d" "kW" kW
  checkPositive "conv2d" "stride" stride
  match parent with
  | .dim inC' (.dim inH (.dim inW .scalar)) =>
      if inC != inC' then
        throw s!"conv2d: inC mismatch: op={inC} vs input={inC'}"
      pure (conv2dCHWOutShape outC inH inW kH kW stride padding)
  | s =>
      throw s!"conv2d: expected input shape (inC,inH,inW), got {repr s}"

end OpContracts

end NN.IR

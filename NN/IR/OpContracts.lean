/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Graph
public import NN.Tensor.Conversion
public import NN.Spec.Layers.Pooling.Spatial

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

open _root_.Spec _root_.TorchLean

/-!
## Small shape utilities

These helpers are used by multiple IR passes, especially `Infer` and `Semantics`.
-/

namespace ShapeUtil

/-- The output shape of flattening a tensor of shape `s` to a 1D vector.

Shape inference and the reference semantics both use this definition. It is a `simp` lemma so
that proofs about either pass see the concrete one-axis shape.
-/
@[simp] def flattenOutShape (s : Shape) : Shape :=
  .dim (Spec.Shape.size s) .scalar

end ShapeUtil

namespace OpContracts

/-!
## Generic contract helpers

These functions live outside any particular pass (`Infer`/`Check`/`Semantics`) so they can be
reused without introducing import cycles.
-/

/-- Check that an `axis` is in-bounds for a given shape. -/
def checkAxisValid (axis : Nat) (s : Shape) : Except String Unit := do
  if axis < Spec.Shape.rank s then
    pure ()
  else
    throw s!"invalid axis {axis} for rank {Spec.Shape.rank s}"

/--
Validate the axis of an axis-dropping reduction (`reduceSum`, `reduceMean`).

The typed reductions `Tensor.reduceSum` and `Tensor.reduceMean` require the reduced axis to have
positive extent, so the shared IR rule is `Spec.Shape.nonemptyAxis?` rather than a bare rank
check: an in-bounds axis of extent zero is rejected by both inference and evaluation. The result
carries the evidence needed by the typed operation.

This is a `simp` definition so proofs that already know the `nonemptyAxis?` verdict can reduce it.
-/
@[simp] def checkReductionAxis (axis : Nat) (s : Shape) :
    Except String (PLift (Shape.NonemptyAxis axis s)) :=
  match Spec.Shape.nonemptyAxis? axis s with
  | some h => .ok h
  | none => .error s!"invalid or empty reduction axis {axis} for shape {repr s}"

/-- Check that a natural-number op parameter is nonzero. -/
def checkPositive (tag param : String) (n : Nat) : Except String Unit := do
  if n = 0 then
    throw s!"{tag}: {param} must be > 0"
  else
    pure ()

/-- Axis permutation that swaps `axis₁` and `axis₂` and leaves every other axis fixed. -/
def transposePerm (rank axis₁ axis₂ : Nat) : Except String (Array Nat) := do
  if axis₁ ≥ rank then
    throw s!"transpose: axis {axis₁} out of range for rank {rank}"
  if axis₂ ≥ rank then
    throw s!"transpose: axis {axis₂} out of range for rank {rank}"
  pure <| (Array.range rank).map fun axis =>
    if axis = axis₁ then axis₂ else if axis = axis₂ then axis₁ else axis

/-- Infer the shape obtained by swapping two arbitrary axes. -/
def inferTransposeOutShape (axis₁ axis₂ : Nat) (parent : Shape) : Except String Shape := do
  let perm ← transposePerm (Spec.Shape.rank parent) axis₁ axis₂
  match Spec.Shape.permute? parent perm.toList with
  | some output => pure output
  | none => throw s!"transpose: internal invalid permutation {repr perm} for {repr parent}"

/--
Compute the matrix dimensions used to interpret `layernorm axis`.

TorchLean’s IR stores LayerNorm as an `axis : Nat` instead of a full `normalized_shape` tuple.
We interpret this in the same way the PyTorch exporter does:

`normalized_shape = dims.drop axis`

That is, we normalize over the **suffix** of dimensions starting at `axis`. To reuse the current
spec primitive (`Spec.layerNorm`), we flatten the input shape `s` into a matrix:

* the row count is the product of dimensions before `axis` (`dims.take axis`);
* the column count is the product of dimensions from `axis` onward (`dims.drop axis`).

LayerNorm then runs over each row and the result is reshaped to `s`. This construction works for
every nonempty tensor rank. The normalized suffix must be nonempty; the leading batch may be empty.
-/
def layerNormMatrixDims (axis : Nat) (s : Shape) : Except String (Nat × Nat) := do
  checkAxisValid axis s
  let dims := Shape.toList s
  let seqLen : Nat := (dims.take axis).foldl (fun acc d => acc * d) 1
  let embedDim : Nat := (dims.drop axis).foldl (fun acc d => acc * d) 1
  checkPositive "layernorm" "embedDim" embedDim
  pure (seqLen, embedDim)

/--
Compute the inverse of a permutation array.

If `perm` is a permutation of `[0,1,...,r-1]` (where `r = perm.length`), then the inverse `inv`
satisfies $\mathrm{inv}[\mathrm{perm}[i]]=i$.
-/
def inversePerm (perm : Array Nat) : Except String (Array Nat) := do
  let r := perm.size
  let mut inv : Array (Option Nat) := Array.replicate r none
  let mut i : Nat := 0
  for p in perm do
    unless p < r do
      throw s!"permute: axis {p} out of range for rank {r} in {repr perm}"
    if inv[p]!.isSome then
      throw s!"permute: duplicate axis {p} in {repr perm}"
    inv := inv.set! p (some i)
    i := i + 1

  let mut out : Array Nat := #[]
  for j in [0:r] do
    match inv[j]! with
    | some idx => out := out.push idx
    | none => throw s!"permute: missing axis {j} in {repr perm}"
  pure out

/--
Permutation (0-based axes) that moves `axis` to the **last** position, preserving the relative
order of the other axes.

Example: rank four with `axis` set to `1` yields `[0,2,3,1]`.
-/
def permMoveAxisToLast (axis : Nat) (s : Shape) : Except String (Array Nat) := do
  checkAxisValid axis s
  let r := Spec.Shape.rank s
  pure <| ((Array.range r).filter (· != axis)).push axis

/--
Permutation (0-based axes) that moves `axis` to the **first** position, preserving the relative
order of the other axes.

Example: rank four with `axis` set to `2` yields `[2,0,1,3]`.
-/
def permMoveAxisToFront (axis : Nat) (s : Shape) : Except String (Array Nat) := do
  checkAxisValid axis s
  let r := Spec.Shape.rank s
  pure <| #[axis] ++ (Array.range r).filter (· != axis)

/--
Geometry of a matrix product after broadcasting batch axes and promoting vectors.

A left vector is viewed as a one-row matrix and a right vector as a one-column matrix.
Their inserted axes are removed from the output; the original operand shapes are retained for
validation and flat indexing.
-/
structure MatmulDims where
  /-- Broadcast batch shape of the output. -/
  leading : Shape
  /-- Row count of the left operand. -/
  rows : Nat
  /-- Contracted extent shared by both operands. -/
  inner : Nat
  /-- Column count of the right operand. -/
  cols : Nat
  /-- Original left operand shape. -/
  leftShape : Shape
  /-- Original right operand shape. -/
  rightShape : Shape
  /-- Batch axes of the left operand, before broadcasting. -/
  leftLeading : Shape := leading
  /-- Batch axes of the right operand, before broadcasting. -/
  rightLeading : Shape := leading
  /-- Whether the left operand was promoted from a vector. -/
  leftVector : Bool := false
  /-- Whether the right operand was promoted from a vector. -/
  rightVector : Bool := false

namespace MatmulDims

/-- Shape of the `matmul` result. -/
@[simp] def outShape (dims : MatmulDims) : Shape :=
  dims.leading.concat
    (Shape.ofList ((if dims.leftVector then [] else [dims.rows]) ++
      (if dims.rightVector then [] else [dims.cols])))

/-- Project a flat output batch coordinate to one operand's unbroadcast batch coordinate. -/
def batchIndex (source target : Shape) (index : Nat) : Nat :=
  let sourceRev := source.toList.reverse
  let targetRev := target.toList.reverse
  ((List.range sourceRev.length).foldl (fun (state : Nat × Nat × Nat) axis =>
    let sourceExtent := sourceRev.getD axis 1
    let targetExtent := targetRev.getD axis 1
    let coordinate := if sourceExtent = 1 then 0 else state.1 % targetExtent
    (state.1 / targetExtent, state.2.1 + coordinate * state.2.2,
      state.2.2 * sourceExtent))
    (index, 0, 1)).2.1

/-- Left scalar used by one output coordinate and one contraction coordinate. -/
def leftIndex (dims : MatmulDims) (output inner : Nat) : Nat :=
  let batch := output / (dims.rows * dims.cols)
  let row := output % (dims.rows * dims.cols) / dims.cols
  batchIndex dims.leftLeading dims.leading batch * (dims.rows * dims.inner) +
    row * dims.inner + inner

/-- Right scalar used by one output coordinate and one contraction coordinate. -/
def rightIndex (dims : MatmulDims) (output inner : Nat) : Nat :=
  let batch := output / (dims.rows * dims.cols)
  let col := output % dims.cols
  batchIndex dims.rightLeading dims.leading batch * (dims.inner * dims.cols) +
    inner * dims.cols + col

end MatmulDims

/-- Broadcast reversed batch dimensions, aligning from the trailing axis.
Singleton axes also broadcast to zero; using `max` here would incorrectly turn zero into one. -/
def broadcastMatmulBatches : List Nat → List Nat → Except String (List Nat)
  | [], bs => pure bs
  | as, [] => pure as
  | a :: as, b :: bs => do
      let rest ← broadcastMatmulBatches as bs
      if a = b then pure (a :: rest)
      else if a = 1 then pure (b :: rest)
      else if b = 1 then pure (a :: rest)
      else throw s!"matmul: incompatible batch dimensions {a} and {b}"

/-- Infer vector promotion and trailing-axis batch broadcasting for non-scalar operands. -/
@[simp] def matmulDims (a b : Shape) : Except String MatmulDims :=
  match a.toList.reverse, b.toList.reverse with
  | n :: m :: leadingRev, p :: n' :: leadingRev' =>
      if _hLeading : leadingRev = leadingRev' then
        if _hInner : n = n' then
          pure
            { leading := Shape.ofList leadingRev.reverse
              rows := m, inner := n, cols := p
              leftShape := a, rightShape := b }
        else
          throw s!"matmul: inner dims mismatch: {n} vs {n'}"
      else if n = n' then do
        let batch ← broadcastMatmulBatches leadingRev leadingRev'
        pure
          { leading := Shape.ofList batch.reverse
            rows := m, inner := n, cols := p
            leftShape := a, rightShape := b
            leftLeading := Shape.ofList leadingRev.reverse
            rightLeading := Shape.ofList leadingRev'.reverse }
      else
        throw s!"matmul: inner dims mismatch: {n} vs {n'}"
  | [n], p :: n' :: leadingRev =>
      if n = n' then
        pure
          { leading := Shape.ofList leadingRev.reverse
            rows := 1, inner := n, cols := p
            leftShape := a, rightShape := b, leftLeading := .scalar, leftVector := true }
      else throw s!"matmul: inner dims mismatch: {n} vs {n'}"
  | n :: m :: leadingRev, [n'] =>
      if n = n' then
        pure
          { leading := Shape.ofList leadingRev.reverse
            rows := m, inner := n, cols := 1
            leftShape := a, rightShape := b, rightLeading := .scalar, rightVector := true }
      else throw s!"matmul: inner dims mismatch: {n} vs {n'}"
  | [n], [n'] =>
      if n = n' then
        pure
          { leading := .scalar, rows := 1, inner := n, cols := 1
            leftShape := a, rightShape := b, leftVector := true, rightVector := true }
      else throw s!"matmul: inner dims mismatch: {n} vs {n'}"
  | _, _ =>
      throw s!"matmul: expected non-scalar inputs, got {repr a} and {repr b}"

/-- Infer the output shape for `matmul` from the two parent shapes (see `matmulDims`). -/
def inferMatmulOutShape (a b : Shape) : Except String Shape :=
  (matmulDims a b).map MatmulDims.outShape

/-- A successful decomposition determines the operand shapes it was computed from. -/
theorem matmulDims_shapes {a b : Shape} {dims : MatmulDims} (h : matmulDims a b = .ok dims) :
    a = dims.leftShape ∧ b = dims.rightShape := by
  cases ha : a.toList.reverse with
  | nil => simp [matmulDims, ha] at h
  | cons n leftTail =>
    cases hb : b.toList.reverse with
    | nil => simp [matmulDims, hb] at h
    | cons p rightTail =>
      cases leftTail with
      | nil =>
        cases rightTail with
        | nil =>
          by_cases hInner : n = p <;>
            simp [matmulDims, ha, hb, hInner] at h
          cases h
          exact ⟨rfl, rfl⟩
        | cons n' rightLeading =>
          by_cases hInner : n = n' <;>
            simp [matmulDims, ha, hb, hInner] at h
          cases h
          exact ⟨rfl, rfl⟩
      | cons m leftLeading =>
        cases rightTail with
        | nil =>
          by_cases hInner : n = p <;>
            simp [matmulDims, ha, hb, hInner] at h
          cases h
          exact ⟨rfl, rfl⟩
        | cons n' rightLeading =>
          by_cases hLeading : leftLeading = rightLeading
          · by_cases hInner : n = n' <;>
              simp [matmulDims, ha, hb, hLeading, hInner] at h
            cases h
            exact ⟨rfl, rfl⟩
          · by_cases hInner : n = n'
            · cases hBatch : broadcastMatmulBatches leftLeading rightLeading with
              | error message =>
                simp [matmulDims, ha, hb, hLeading, hInner, hBatch,
                  Bind.bind, Except.bind] at h
              | ok batch =>
                simp [matmulDims, ha, hb, hLeading, hInner, hBatch,
                  Pure.pure, Except.pure, Bind.bind, Except.bind] at h
                cases h
                exact ⟨rfl, rfl⟩
            · simp [matmulDims, ha, hb, hLeading, hInner] at h

/-- Merge one concat input into the accumulated dimensions. -/
def mergeConcatDims (axis : Nat) : Nat → List Nat → List Nat → Except String (List Nat)
  | _, [], [] => pure []
  | index, expected :: expecteds, actual :: actuals => do
      let rest ← mergeConcatDims axis (index + 1) expecteds actuals
      if index = axis then
        pure ((expected + actual) :: rest)
      else if actual = expected then
        pure (expected :: rest)
      else
        throw s!"concat: non-axis dim mismatch at axis {index}: expected {expected}, got {actual}"
  | _, _, _ => throw "concat: rank mismatch"

/-- Fold compatible concat inputs into the accumulated output dimensions. -/
def foldConcatDims (axis : Nat) : List Nat → List Shape → Except String (List Nat)
  | dimensions, [] => pure dimensions
  | dimensions, shape :: shapes => do
      let dimensions ← mergeConcatDims axis 0 dimensions shape.toList
      foldConcatDims axis dimensions shapes

/--
Infer the output shape for `concat` from an array of parent shapes.

Every parent must have the same rank, the selected axis must exist, and all dimensions other than
that axis must agree. The selected dimensions are summed. Both the axis and the tensor rank are
arbitrary.
-/
def inferConcatOutShape (axis : Nat) (parents : Array Shape) : Except String Shape := do
  match parents.toList with
  | first :: second :: rest =>
      checkAxisValid axis first
      let dimensions ← foldConcatDims axis first.toList (second :: rest)
      pure (Shape.ofList dimensions)
  | _ => throw s!"concat: expected at least 2 inputs, got {parents.size}"

/-!
## Sliding-window shape arithmetic

Convolution and pooling preserve a leading channel axis, but their admissible padding domains are
not identical. The contracts below share validation and traversal while retaining the correct
output formula for each operation family.
-/

/-- Effective kernel width for a dilated window. -/
def effectiveKernel (kernel dilation : Nat) : Nat :=
  Shape.dilatedKernelExtent kernel dilation

/-- Output length for a dilated window with independent low/high padding. -/
def slideOutDilated (input kernel stride dilation paddingBefore paddingAfter : Nat) : Nat :=
  Shape.slidingWindowOutDimDilated input kernel stride dilation paddingBefore paddingAfter

/-- Infer dilated convolution dimensions from one parameter per spatial axis. -/
def inferConvDims (tag : String) (axisNames : List String)
    (inputs kernels strides dilations paddingBefore paddingAfter : List Nat) :
    Except String (List Nat) :=
  go axisNames inputs kernels strides dilations paddingBefore paddingAfter
where
  go : List String → List Nat → List Nat → List Nat → List Nat → List Nat →
      List Nat → Except String (List Nat)
    | [], [], [], [], [], [], [] => pure []
    | axis :: axes, input :: inputs, kernel :: kernels, stride :: strides,
        dilation :: dilations, low :: lows, high :: highs => do
      checkPositive tag s!"{axis} kernel" kernel
      checkPositive tag s!"{axis} stride" stride
      checkPositive tag s!"{axis} dilation" dilation
      let effective := effectiveKernel kernel dilation
      let padded := input + low + high
      if padded < effective then
        throw <|
          s!"{tag}: {axis} window does not fit padded input: input={input}, \
            padding=({low}, {high}), effectiveKernel={effective}"
      let rest ← go axes inputs kernels strides dilations lows highs
      pure (slideOutDilated input kernel stride dilation low high :: rest)
    | _, _, _, _, _, _, _ =>
      throw s!"{tag}: spatial metadata ranks must agree"

/--
Infer pooling output lengths while enforcing the same basic window checks as graph validation.

Pooling uses `poolOutDim`, which assigns an empty output to an empty input axis, an oversized
window, or padding outside the pooling domain. Kernels and strides must still be positive.
-/
def inferPoolingDims (tag : String) (axisNames : List String)
    (inputs kernels strides paddings : List Nat) : Except String (List Nat) :=
  go axisNames inputs kernels strides paddings
where
  go : List String → List Nat → List Nat → List Nat → List Nat → Except String (List Nat)
    | [], [], [], [], [] => pure []
    | axis :: axes, input :: inputs, kernel :: kernels, stride :: strides,
        padding :: paddings => do
        checkPositive tag s!"{axis} kernel" kernel
        checkPositive tag s!"{axis} stride" stride
        let rest ← go axes inputs kernels strides paddings
        pure (poolOutDim input kernel stride padding :: rest)
    | _, _, _, _, _ =>
        throw s!"{tag}: axis-name, input, kernel, stride, and padding ranks must agree"

/-- Validated shape information for pooling over a tensor suffix.

The plan is the common boundary between shape inference, denotational evaluation, and executable
lowering. It carries the exact split and positivity evidence required by the typed pooling
operators, so those passes cannot silently disagree about which axes are spatial. -/
structure PoolPlan (config : WindowConfig) (parent : Shape) where
  /-- Axes preserved before the pooled suffix. -/
  leading : Shape
  /-- Input extents along the pooled axes. -/
  spatial : TorchLean.Tensor Nat [config.spatialRank]
  /-- The prefix and spatial suffix reconstruct the parent shape. -/
  concat_eq : leading.concat (Shape.ofList (Tensor.to spatial (List Nat))) = parent
  /-- Every kernel extent is nonzero. -/
  kernelNonzero : ∀ axis : Fin config.spatialRank, config.kernel.getScalar axis ≠ 0
  /-- Every stride is nonzero. -/
  strideNonzero : ∀ axis : Fin config.spatialRank, config.stride.getScalar axis ≠ 0

namespace PoolPlan

/-- Output shape computed by a validated pooling plan. -/
def outShape {config : WindowConfig} {parent : Shape} (plan : PoolPlan config parent) : Shape :=
  plan.leading.concat <|
    Shape.ofList
      (Tensor.to (Spec.poolOutSpatialPad plan.spatial config.kernel config.stride config.padding)
        (List Nat))

end PoolPlan

/-- Validate and plan pooling over a spatial suffix of arbitrary rank.

Every preceding axis is preserved, so one operation handles unbatched tensors, ordinary batches,
and tensors with several leading batch dimensions.
-/
def planPool (tag : String) (config : WindowConfig) (parent : Shape) :
    Except String (PoolPlan config parent) := do
  checkPositive tag "spatial rank" config.spatialRank
  if hRank : config.spatialRank ≤ parent.rank then
    let split := Shape.splitSuffix parent config.spatialRank hRank
    let spatial : Tensor Nat [config.spatialRank] :=
      Tensor.castShape (Tensor.from split.suffix) (by
        simp [split.suffix_length])
    if hKernel : ∀ axis : Fin config.spatialRank, config.kernel.getScalar axis ≠ 0 then
      if hStride : ∀ axis : Fin config.spatialRank, config.stride.getScalar axis ≠ 0 then
        let leadingRank := split.leading.rank
        let axisNames :=
          (List.range config.spatialRank).map fun axis => s!"axis {leadingRank + axis}"
        let _ ← inferPoolingDims tag axisNames split.suffix
          (Tensor.to config.kernel (List Nat))
          (Tensor.to config.stride (List Nat))
          (Tensor.to config.padding (List Nat))
        pure
          { leading := split.leading
            spatial := spatial
            concat_eq := by
              simp only [spatial, Tensor.to_list_castShape, Tensor.to_list_from_list]
              exact split.concat_eq
            kernelNonzero := hKernel
            strideNonzero := hStride }
      else
        throw s!"{tag}: stride extents must be nonzero"
    else
      throw s!"{tag}: kernel extents must be nonzero"
  else
    throw s!"{tag}: kernel rank {config.spatialRank} exceeds input rank {parent.rank}"

/-- Infer the output shape of an arbitrary-rank pooling operation from its window geometry.

This is the shape rule shared by `Infer.nodeOutShape` and the reference semantics: both call
`planPool` on the same configuration, so the evaluator's pooled tensor has exactly this shape.
-/
def inferWindowOutShape (tag : String) (config : WindowConfig) (parent : Shape) :
    Except String Shape := do
  let plan ← planPool tag config parent
  pure plan.outShape

/-- Infer the output shape of an arbitrary-rank pooling operation. -/
def inferPoolOutShape (tag : String) {spatialRank : Nat}
    (kernels strides paddings : TorchLean.Tensor Nat [spatialRank]) (parent : Shape) :
    Except String Shape :=
  inferWindowOutShape tag
    { spatialRank := spatialRank, kernel := kernels, stride := strides, padding := paddings } parent

/-- Infer the output shape for the full parameterized convolution configuration. -/
def inferConvConfigOutShape (tag : String) (config : ConvConfig) (parent : Shape) :
    Except String Shape := do
  checkPositive tag "inChannels" config.inChannels
  checkPositive tag "outChannels" config.outChannels
  checkPositive tag "groups" config.groups
  if config.inChannels % config.groups != 0 then
    throw s!"{tag}: in_channels={config.inChannels} is not divisible by groups={config.groups}"
  if config.outChannels % config.groups != 0 then
    throw s!"{tag}: out_channels={config.outChannels} is not divisible by groups={config.groups}"
  checkAxisValid config.channelAxis parent
  let dims := parent.toList
  let some actualChannels := dims[config.channelAxis]?
    | throw s!"{tag}: internal missing channel axis {config.channelAxis} in {repr parent}"
  if actualChannels != config.inChannels then
    throw s!"{tag}: input-channel mismatch: op={config.inChannels} vs input={actualChannels}"
  let inputs := dims.drop (config.channelAxis + 1)
  if inputs.length != config.spatialRank then
    throw <|
      s!"{tag}: kernel rank {config.spatialRank} does not match \
        the {inputs.length} spatial axes after channel axis {config.channelAxis}"
  let axisNames :=
    (List.range config.spatialRank).map fun axis => s!"axis {config.channelAxis + 1 + axis}"
  let outputs ← inferConvDims tag axisNames inputs
    (Tensor.to config.kernel (List Nat))
    (Tensor.to config.stride (List Nat))
    (Tensor.to config.dilation (List Nat))
    (Tensor.to config.padding (List Nat))
    (Tensor.to config.paddingAfter (List Nat))
  pure (Shape.ofList (dims.take config.channelAxis ++ config.outChannels :: outputs))

/--
Infer dense, unit-dilation convolution output geometry for an arbitrary spatial rank.

This is the ordinary convolution specialization of `inferConvConfigOutShape`; keeping one checker
prevents the default and parameterized APIs from assigning different shapes to the same operation.
-/
def inferConvOutShape (tag : String) (channelAxis inChannels outChannels : Nat)
    {spatialRank : Nat} (kernels strides paddings : TorchLean.Tensor Nat [spatialRank])
    (parent : Shape) : Except String Shape :=
  inferConvConfigOutShape tag
    { spatialRank := spatialRank
      kernel := kernels
      stride := strides
      padding := paddings
      dilation := Tensor.full [spatialRank] 1
      paddingAfter := paddings
      groups := 1
      channelAxis := channelAxis
      inChannels := inChannels
      outChannels := outChannels }
    parent

/-- Check eval-mode BatchNorm metadata against an arbitrary channel axis. -/
def inferBatchNormEvalOutShape (channelAxis channels : Nat) (parent : Shape) :
    Except String Shape := do
  checkPositive "batch_norm_eval" "channels" channels
  checkAxisValid channelAxis parent
  let some actualChannels := parent.toList[channelAxis]?
    | throw s!"batch_norm_eval: internal missing channel axis {channelAxis} in {repr parent}"
  if channels = actualChannels then
    pure parent
  else
    throw s!"batch_norm_eval: channel mismatch: op={channels} vs input={actualChannels}"

end OpContracts

end NN.IR

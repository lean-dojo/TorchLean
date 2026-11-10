/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.IR.Attr
public import TorchLean.NN.IR.MatmulLayout
public import TorchLean.Tensor.Conversion
public import TorchLean.Tensor.Layers.Pooling

/-!
# Operation Contracts

Shape helpers used by the model recorder and the bound passes: slice dimensions, pooling windows,
convolution geometry and matrix-product layout. The registered operators carry their other shape
conditions in their types, and `Op.decode` checks the corresponding wire attributes.
`TorchLean.NN.IR.Record` uses these helpers, and the CROWN matrix-product rules read the
`IR.MatmulDims` decomposition that `inferMatmulDims` returns.
-/

@[expose] public section

open TorchLean.Shape (rank)

namespace TorchLean.IR

/-!
## Generic contract helpers

The helpers in this section live outside any one user, so they can be reused without an import
cycle.
-/

/-- Check that `axis` is below the rank of `s`. -/
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

/-- Broadcast reversed batch extents, aligning from the trailing axis. A singleton axis broadcasts
to any extent, zero included: `1` against `0` gives `0`, where `max` would give `1`. -/
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
@[simp] def inferMatmulDims (a b : Shape) : Except String MatmulDims :=
  match a.reverse, b.reverse with
  | n :: m :: leadingRev, p :: n' :: leadingRev' =>
      if _hLeading : leadingRev = leadingRev' then
        if _hInner : n = n' then
          pure
            { leading := leadingRev.reverse
              rows := m, inner := n, cols := p
              leftShape := a, rightShape := b }
        else
          throw s!"matmul: inner dims mismatch: {n} vs {n'}"
      else if n = n' then do
        let batch ← broadcastMatmulBatches leadingRev leadingRev'
        pure
          { leading := batch.reverse
            rows := m, inner := n, cols := p
            leftShape := a, rightShape := b
            leftLeading := leadingRev.reverse
            rightLeading := leadingRev'.reverse }
      else
        throw s!"matmul: inner dims mismatch: {n} vs {n'}"
  | [n], p :: n' :: leadingRev =>
      if n = n' then
        pure
          { leading := leadingRev.reverse
            rows := 1, inner := n, cols := p
            leftShape := a, rightShape := b, leftLeading := .scalar, leftVector := true }
      else throw s!"matmul: inner dims mismatch: {n} vs {n'}"
  | n :: m :: leadingRev, [n'] =>
      if n = n' then
        pure
          { leading := leadingRev.reverse
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

/-- Infer the output shape for `matmul` from the two parent shapes (see `inferMatmulDims`). -/
def inferMatmulOutShape (a b : Shape) : Except String Shape :=
  (inferMatmulDims a b).map MatmulDims.outShape

/-- A successful decomposition determines the operand shapes it was computed from. -/
theorem inferMatmulDims_shapes {a b : Shape} {dims : MatmulDims}
    (h : inferMatmulDims a b = .ok dims) :
    a = dims.leftShape ∧ b = dims.rightShape := by
  cases ha : a.reverse with
  | nil => simp [inferMatmulDims, ha] at h
  | cons n leftTail =>
    cases hb : b.reverse with
    | nil => simp [inferMatmulDims, hb] at h
    | cons p rightTail =>
      cases leftTail with
      | nil =>
        cases rightTail with
        | nil =>
          by_cases hInner : n = p <;>
            simp [inferMatmulDims, ha, hb, hInner] at h
          cases h
          exact ⟨rfl, rfl⟩
        | cons n' rightLeading =>
          by_cases hInner : n = n' <;>
            simp [inferMatmulDims, ha, hb, hInner] at h
          cases h
          exact ⟨rfl, rfl⟩
      | cons m leftLeading =>
        cases rightTail with
        | nil =>
          by_cases hInner : n = p <;>
            simp [inferMatmulDims, ha, hb, hInner] at h
          cases h
          exact ⟨rfl, rfl⟩
        | cons n' rightLeading =>
          by_cases hLeading : leftLeading = rightLeading
          · by_cases hInner : n = n' <;>
              simp [inferMatmulDims, ha, hb, hLeading, hInner] at h
            cases h
            exact ⟨rfl, rfl⟩
          · by_cases hInner : n = n'
            · cases hBatch : broadcastMatmulBatches leftLeading rightLeading with
              | error message =>
                simp [inferMatmulDims, ha, hb, hLeading, hInner, hBatch,
                  Bind.bind, Except.bind] at h
              | ok batch =>
                simp [inferMatmulDims, ha, hb, hLeading, hInner, hBatch,
                  Pure.pure, Except.pure, Bind.bind, Except.bind] at h
                cases h
                exact ⟨rfl, rfl⟩
            · simp [inferMatmulDims, ha, hb, hLeading, hInner] at h

/-- A successful decomposition reads the operands as the views `Shape.matmulLeft` and
`Shape.matmulRight` at its vector flags and the result as `Shape.matmulOut`. A vector operand
has extent one in the axis it is promoted with. -/
theorem inferMatmulDims_views {a b : Shape} {dims : MatmulDims}
    (h : inferMatmulDims a b = .ok dims) :
    (dims.leftVector = true → dims.rows = 1) ∧ (dims.rightVector = true → dims.cols = 1) ∧
    a = Shape.matmulLeft dims.leftVector dims.leftLeading dims.rows dims.inner ∧
    b = Shape.matmulRight dims.rightVector dims.rightLeading dims.inner dims.cols ∧
    dims.outShape =
      Shape.matmulOut dims.leftVector dims.rightVector dims.leading dims.rows dims.cols := by
  have ha' : a = a.reverse.reverse := by simp
  have hb' : b = b.reverse.reverse := by simp
  cases ha : a.reverse with
  | nil => simp [inferMatmulDims, ha] at h
  | cons n leftTail =>
    cases hb : b.reverse with
    | nil => simp [inferMatmulDims, hb] at h
    | cons p rightTail =>
      rw [ha] at ha'
      rw [hb] at hb'
      cases leftTail with
      | nil =>
        cases rightTail with
        | nil =>
          by_cases hInner : n = p <;>
            simp [inferMatmulDims, ha, hb, hInner] at h
          cases h
          subst ha' hb'
          simp_all [Shape.matmulLeft, Shape.matmulRight, Shape.matmulOut]
        | cons n' rightLeading =>
          by_cases hInner : n = n' <;>
            simp [inferMatmulDims, ha, hb, hInner] at h
          cases h
          subst ha' hb'
          simp_all [Shape.matmulLeft, Shape.matmulRight, Shape.matmulOut]
      | cons m leftLeading =>
        cases rightTail with
        | nil =>
          by_cases hInner : n = p <;>
            simp [inferMatmulDims, ha, hb, hInner] at h
          cases h
          subst ha' hb'
          simp_all [Shape.matmulLeft, Shape.matmulRight, Shape.matmulOut]
        | cons n' rightLeading =>
          by_cases hLeading : leftLeading = rightLeading
          · by_cases hInner : n = n' <;>
              simp [inferMatmulDims, ha, hb, hLeading, hInner] at h
            cases h
            subst ha' hb'
            simp_all [Shape.matmulLeft, Shape.matmulRight, Shape.matmulOut]
          · by_cases hInner : n = n'
            · cases hBatch : broadcastMatmulBatches leftLeading rightLeading with
              | error message =>
                simp [inferMatmulDims, ha, hb, hLeading, hInner, hBatch,
                  Bind.bind, Except.bind] at h
              | ok batch =>
                simp [inferMatmulDims, ha, hb, hLeading, hInner, hBatch,
                  Pure.pure, Except.pure, Bind.bind, Except.bind] at h
                cases h
                subst ha' hb'
                simp_all [Shape.matmulLeft, Shape.matmulRight, Shape.matmulOut]
            · simp [inferMatmulDims, ha, hb, hLeading, hInner] at h

/-- Two matrices over one leading shape with a shared inner extent decompose at that leading shape,
with neither operand promoted from a vector. -/
theorem inferMatmulDims_concat (L : Shape) (r k c : Nat) :
    inferMatmulDims (L ++ [r, k]) (L ++ [k, c]) =
      .ok { leading := L, rows := r, inner := k, cols := c, leftShape := L ++ [r, k],
            rightShape := L ++ [k, c] } := by
  simp [inferMatmulDims, pure, Except.pure]

/-!
## Sliding-window shape arithmetic

Convolution and pooling preserve a leading channel axis, but their admissible paddings differ.
`inferConvDims` checks the convolution parameters axis by axis and computes the output extents.
`planPool` checks a pooling window, and `PoolPlan.outShape` computes its output shape with
`Pool.outSpatial`.
-/

/-- Infer dilated convolution dimensions from one parameter per spatial axis. -/
def inferConvDims (tag : String) (axisNames : List String)
    (inputs kernels strides dilations paddingBefore paddingAfter : List Nat) :
    Except String (List Nat) :=
  go axisNames inputs kernels strides dilations paddingBefore paddingAfter
where
  /-- The output length of each spatial axis. It fails unless the lists have equal lengths, every
  kernel, stride and dilation is positive and every dilated window fits its padded input. -/
  go : List String → List Nat → List Nat → List Nat → List Nat → List Nat →
      List Nat → Except String (List Nat)
    | [], [], [], [], [], [], [] => pure []
    | axis :: axes, input :: inputs, kernel :: kernels, stride :: strides,
        dilation :: dilations, low :: lows, high :: highs => do
      checkPositive tag s!"{axis} kernel" kernel
      checkPositive tag s!"{axis} stride" stride
      checkPositive tag s!"{axis} dilation" dilation
      let effective := Shape.dilatedKernelExtent kernel dilation
      let padded := input + low + high
      if padded < effective then
        throw <|
          s!"{tag}: {axis} window does not fit padded input: input={input}, \
            padding=({low}, {high}), effectiveKernel={effective}"
      let rest ← go axes inputs kernels strides dilations lows highs
      pure (Shape.slidingWindowOutDimDilated input kernel stride dilation low high :: rest)
    | _, _, _, _, _, _, _ =>
      throw s!"{tag}: spatial metadata ranks must agree"

/-- Validated shape information for pooling over a suffix of a tensor's axes.

It carries the split of the parent shape and the positivity evidence that the typed pooling
operators require. `planPool` builds it, and `inferWindowOutShape` returns its output shape. -/
structure PoolPlan (config : WindowConfig) (parent : Shape) where
  /-- Axes preserved before the pooled suffix. -/
  leading : Shape
  /-- Input extents along the pooled axes. -/
  spatial : Tensor Nat [config.spatialRank]
  /-- The prefix and spatial suffix reconstruct the parent shape. -/
  concat_eq : leading ++ (Tensor.to spatial (List Nat)) = parent
  /-- Every kernel extent is nonzero. -/
  kernel_ne_zero : ∀ axis : Fin config.spatialRank, config.kernel.getScalar axis ≠ 0
  /-- Every stride is nonzero. -/
  stride_ne_zero : ∀ axis : Fin config.spatialRank, config.stride.getScalar axis ≠ 0

namespace PoolPlan

/-- Output shape computed by a validated pooling plan. -/
def outShape {config : WindowConfig} {parent : Shape} (plan : PoolPlan config parent) : Shape :=
  plan.leading ++
    Tensor.to (Pool.outSpatial plan.spatial config.kernel config.stride config.padding)
      (List Nat)

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
        pure
          { leading := split.leading
            spatial := spatial
            concat_eq := by
              simp only [spatial, Tensor.to_list_castShape, Tensor.to_list_from_list]
              exact split.append_eq
            kernel_ne_zero := hKernel
            stride_ne_zero := hStride }
      else
        throw s!"{tag}: stride extents must be nonzero"
    else
      throw s!"{tag}: kernel extents must be nonzero"
  else
    throw s!"{tag}: kernel rank {config.spatialRank} exceeds input rank {parent.rank}"

/-- Infer the output shape of an arbitrary-rank pooling operation from its window geometry: the
output shape of `planPool`'s plan.

The recording of a pooling step (`TorchLean.NN.IR.Record`) checks its window with it.
-/
def inferWindowOutShape (tag : String) (config : WindowConfig) (parent : Shape) :
    Except String Shape := do
  let plan ← planPool tag config parent
  pure plan.outShape

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
  let dims : List Nat := parent
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
  pure (dims.take config.channelAxis ++ config.outChannels :: outputs)

/-!
## Dimension attributes

The recorder's slice helper reads natural-number dimensions through `Shape.axes?`, the same
reader the operator codec uses. It rejects repeated dimensions and dimensions outside the rank.
-/

/-- The distinct dimensions of `s` named by `dims`, in their order. -/
def dimsOf (tag : String) (s : Shape) (dims : Array Nat) : Except String (List (Fin s.rank)) :=
  match s.axes? (dims.map Int.ofNat) with
  | some ds => .ok ds
  | none => .error s!"{tag}: {dims} are not distinct dimensions of {repr s}"

/-- The dimensions `dimsOf` accepts are distinct. -/
theorem nodup_of_dimsOf {tag : String} {s : Shape} {dims : Array Nat}
    {ds : List (Fin s.rank)} (h : dimsOf tag s dims = .ok ds) : ds.Nodup := by
  unfold dimsOf at h
  split at h
  · cases h
    exact Shape.nodup_of_axes?_eq_some ‹_›
  · cases h

/-!
## General data movement
-/

/-- The dimensions of a slice, checked by `Shape.SliceValid`: distinct, one start, stop and
step each, every stop at most its extent and every step positive. -/
def sliceDims (s : Shape) (dims starts stops steps : Array Nat) :
    Except String {ds : List (Fin s.rank) //
      Shape.SliceValid s ds starts.toList stops.toList steps.toList} := do
  let ds ← dimsOf "slice" s dims
  if h : Shape.SliceValid s ds starts.toList stops.toList steps.toList then pure ⟨ds, h⟩
  else throw s!"slice: invalid starts {starts}, stops {stops}, steps {steps} at dimensions \
    {dims} of {repr s}"

end TorchLean.IR

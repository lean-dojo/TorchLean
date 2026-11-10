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
public import TorchLean.Tensor.Contract

/-!
# Operation Contracts

The shape contracts of the operations with a nontrivial shape rule. Each contract returns the
checked attributes in the form the specification operator takes (dimensions as `Fin rank`, with
their validity proofs) together with the output shape. The recording of registered operators
(`TorchLean.NN.IR.Record`) uses them, and the CROWN matrix-product rules read the decomposition
`IR.MatmulDims` that `inferMatmulDims` returns.
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

Operations store dimensions as natural numbers. These contracts read them against the rank of the
shape they act on through the signed readers `Shape.axis?` and `Shape.axes?` that the
importers use, so a natural number below the rank names itself and any other names nothing.
-/

/-- The dimension of `s` named by `dim`, rejecting one at or beyond the rank. -/
def dimOf (tag : String) (s : Shape) (dim : Nat) : Except String (Fin s.rank) :=
  match s.axis? (dim : Int) with
  | some d => .ok d
  | none => .error s!"{tag}: dimension {dim} is out of range for shape {repr s}"

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

/-- One value per dimension of `s`, as a function on its dimensions. -/
def perDim (tag : String) (s : Shape) (values : Array Nat) : Except String (Fin s.rank → Nat) :=
  if h : values.size = s.rank then .ok fun d => values[d.val]'(by rw [h]; exact d.isLt)
  else .error s!"{tag}: expected {s.rank} values, one per dimension of {repr s}, got {values}"

/-- The dimension pairs `(a, b)` with `a` a dimension of `s` and `b` one of `t`. -/
def dimPairsOf (tag : String) (s t : Shape) (pairs : Array (Nat × Nat)) :
    Except String (List (Fin s.rank × Fin t.rank)) :=
  pairs.toList.mapM fun (a, b) => do pure (← dimOf tag s a, ← dimOf tag t b)

/-!
## General data movement
-/

/-- The placement of a broadcast from `s` to `t`: source dimension `i` goes to `dims[i]`, in
increasing order, and each source extent is one or the extent there. -/
def broadcastDims (s t : Shape) (dims : Array Nat) :
    Except String {f : Fin s.rank → Fin t.rank // Shape.BroadcastDims s t f} :=
  if hsize : dims.size = s.rank then
    if hlt : ∀ i : Fin s.rank, dims[i.val]'(by rw [hsize]; exact i.isLt) < t.rank then
      let f : Fin s.rank → Fin t.rank := fun i =>
        ⟨dims[i.val]'(by rw [hsize]; exact i.isLt), hlt i⟩
      if hf : Shape.BroadcastDims s t f then .ok ⟨f, hf⟩
      else .error s!"broadcast: {repr s} does not broadcast to {repr t} at dimensions {dims}"
    else .error s!"broadcast: dimensions {dims} are out of range for {repr t}"
  else .error s!"broadcast: expected one dimension per dimension of {repr s}, got {dims}"

/-- The dimensions of a slice, checked by `Shape.SliceValid`: distinct, one start, stop and
step each, every stop at most its extent and every step positive. -/
def sliceDims (s : Shape) (dims starts stops steps : Array Nat) :
    Except String {ds : List (Fin s.rank) //
      Shape.SliceValid s ds starts.toList stops.toList steps.toList} := do
  let ds ← dimsOf "slice" s dims
  if h : Shape.SliceValid s ds starts.toList stops.toList steps.toList then pure ⟨ds, h⟩
  else throw s!"slice: invalid starts {starts}, stops {stops}, steps {steps} at dimensions \
    {dims} of {repr s}"

/-- The checked index tensor of a gather or scatter along `dims`: the index along `dims[k]` at the
coordinate with flat index `i` is `indices.values[i * dims.length + k]`. -/
def IndexTensor.toTensor (tag : String) (s : Shape) (dims : List (Fin s.rank))
    (indices : IndexTensor) :
    Except String (Tensor (Einops.AxisTuple s.extent dims) indices.shape) :=
  if hsize : indices.values.size = indices.shape.size * dims.length then
    have hlt (i : Fin indices.shape.size) (k : Fin dims.length) :
        i.val * dims.length + k.val < indices.values.size := by
      rw [hsize]
      calc i.val * dims.length + k.val < i.val * dims.length + dims.length := by omega
        _ = (i.val + 1) * dims.length := by rw [Nat.succ_mul]
        _ ≤ indices.shape.size * dims.length := Nat.mul_le_mul_right _ i.isLt
    if hbound : ∀ i : Fin indices.shape.size, ∀ k : Fin dims.length,
        indices.values[i.val * dims.length + k.val]'(hlt i k) < s.extent (dims.get k) then
      .ok (Tensor.Rep.ofFn fun c => fun k =>
        let i : Fin indices.shape.size := Shape.Coord.linearize c
        ⟨indices.values[i.val * dims.length + k.val]'(hlt i k), hbound i k⟩)
    else .error s!"{tag}: an index is out of range for the dimensions of {repr s}"
  else .error s!"{tag}: {indices.values.size} index values for {indices.shape.size} entries of \
    {dims.length} dimensions"

/-- The checked arguments of `Tensor.gather`. -/
structure GatherPlan (s : Shape) (indices : IndexTensor) where
  /-- The indexed dimensions. -/
  dims : List (Fin s.rank)
  /-- Pairs of an operand dimension and an index-tensor dimension read together. -/
  batchDims : List (Fin s.rank × Fin indices.shape.rank)
  /-- The arguments satisfy `Shape.GatherValid`. -/
  valid : Shape.GatherValid s indices.shape dims batchDims
  /-- The typed index tensor. -/
  index : Tensor (Einops.AxisTuple s.extent dims) indices.shape

/-- The shape of a gather: the index-tensor dimensions, then the operand dimensions that are
neither indexed nor batched. -/
def GatherPlan.outShape {s : Shape} {indices : IndexTensor} (plan : GatherPlan s indices) :
    Shape :=
  indices.shape ++ s.eraseAxes (plan.dims ++ plan.batchDims.map Prod.fst)

/-- Check the arguments of a gather from `s` at the indices along `dims` with the batch pairs. -/
def planGather (s : Shape) (dims : Array Nat) (batchDims : Array (Nat × Nat))
    (indices : IndexTensor) : Except String (GatherPlan s indices) := do
  let ds ← dimsOf "gather" s dims
  let bs ← dimPairsOf "gather" s indices.shape batchDims
  if hv : (ds ++ bs.map Prod.fst).Nodup ∧ (bs.map Prod.snd).Nodup ∧
      ∀ p ∈ bs, s.extent p.1 = indices.shape.extent p.2 then
    let index ← IndexTensor.toTensor "gather" s ds indices
    pure ⟨ds, bs, ⟨hv.1, hv.2.1, hv.2.2⟩, index⟩
  else throw s!"gather: dimensions {dims} and batch pairs {batchDims} are not valid for \
    {repr s} and indices of shape {repr indices.shape}"

/-- The concatenated shape of `shapes` along `dim`: every shape has the first one's extents off
`dim`, and the extents along `dim` add. -/
def catShape (dim : Nat) (shapes : Array Shape) : Except String Shape :=
  if hne : 0 < shapes.size then
    if dim < shapes[0].rank then
      if ∀ j : Fin shapes.size, shapes[j] = shapes[0].set dim (shapes[j].getD dim 0) then
        .ok (shapes[0].set dim (∑ j : Fin shapes.size, shapes[j].getD dim 0))
      else .error s!"cat: the inputs differ off dimension {dim}: {repr shapes}"
    else .error s!"cat: dimension {dim} is out of range for {repr shapes[0]}"
  else .error "cat: expected at least one input"

/-!
## Reductions and contractions
-/

/-- The dimensions of a reduction with `op`. A maximum or a minimum starts from the first entry of
each fiber, so it needs every listed dimension nonempty. -/
def reduceDims (op : Tensor.ReduceOp) (s : Shape) (dims : Array Nat) :
    Except String
      {ds : List (Fin s.rank) // (op = .max ∨ op = .min) → ∀ d ∈ ds, 0 < s.extent d} := do
  let ds ← dimsOf "reduce" s dims
  if h : (op = .max ∨ op = .min) → ∀ d ∈ ds, 0 < s.extent d then pure ⟨ds, h⟩
  else throw s!"reduce: a {repr op} over an empty dimension of {repr s}"

/-- The checked pairs of a contraction of `s` with `t`, satisfying `Shape.ContractValid`. -/
def contractDims (s t : Shape) (dims batchDims : Array (Nat × Nat)) :
    Except String
      {p : List (Fin s.rank × Fin t.rank) × List (Fin s.rank × Fin t.rank) //
        Shape.ContractValid s t p.1 p.2} := do
  let ds ← dimPairsOf "contract" s t dims
  let bs ← dimPairsOf "contract" s t batchDims
  if hv : ((bs ++ ds).map Prod.fst).Nodup ∧ ((bs ++ ds).map Prod.snd).Nodup ∧
      ∀ p ∈ bs ++ ds, s.extent p.1 = t.extent p.2 then
    pure ⟨(ds, bs), ⟨hv.1, hv.2.1, hv.2.2⟩⟩
  else throw s!"contract: pairs {dims} and batch pairs {batchDims} are not valid for {repr s} \
    and {repr t}"

end TorchLean.IR

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Layers.Core
public import NN.Runtime.Autograd.Model.Functional.Core

/-!
# TorchLean NN: Attention
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Model

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Proofs.Autograd.Algebra

namespace Layers

namespace Attention

/--
Softmax over the visible keys of each attention row.

The mask uses `true` for a visible key. Gathering those keys before softmax keeps masked scores
out of both the row maximum and its denominator. An entirely masked row stays zero. In particular,
we do not approximate exclusion with a large negative constant, whose behavior depends on the
score scale and cannot represent an empty row.
-/
def probabilities {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {batch heads n : Nat}
    (scores : Ref (m := m) (α := α) [batch, heads, n, n])
    (mask : Option (Tensor Bool [n, n])) :
    m (Ref (m := m) (α := α) [batch, heads, n, n]) := do
  match mask with
  | none => softmaxLast (m := m) (α := α) (s := [batch, heads, n, n]) scores
  | some mask =>
      let empty ← const (m := m) (α := α) (Tensor.zeros (α := α) [batch, heads, n, n])
      (List.finRange n).foldlM (init := empty) fun result query => do
        let visible := (List.finRange n).filter fun key => get2 mask query key
        let indices : Tensor (Fin n) [visible.length] :=
          Tensor.dim fun index => Tensor.scalar (visible[index.val]'index.isLt)
        let indicesRef := Runtime.Autograd.Torch.dataConst (m := m) (α := α) indices
        let row ← select (m := m) (α := α) (s := [batch, heads, n, n]) 2 scores query
        let selected ← indexSelect (m := m) (α := α) (s := [batch, heads, n])
          2 visible.length row indicesRef
        -- An empty gather keeps a zero-gradient edge to the scores without reading their values.
        let normalized ← if visible.isEmpty then pure selected else
          softmaxLast (m := m) (α := α) (s := [batch, heads, visible.length]) selected
        let rowZero ← const (m := m) (α := α)
          (Tensor.zeros (α := α) [batch, heads, n])
        let restored ← scatterAdd (m := m) (α := α) (s := [batch, heads, n])
          2 visible.length rowZero normalized indicesRef
        let source ← reshape (m := m) (α := α)
          (s₁ := [batch, heads, n]) (s₂ := [batch, heads, 1, n]) restored
          (by simp [Shape.size])
        let queryIndex : Tensor (Fin n) [1] := Tensor.ofFn fun _ => query
        scatterAdd (m := m) (α := α) (s := [batch, heads, n, n]) 2 1 result source
          (Runtime.Autograd.Torch.dataConst (m := m) (α := α) queryIndex)

/--
Attention with optional affine projection biases and dropout on softmax probabilities.

The operation order is projection, head splitting, scaled dot products, masked softmax,
probability dropout, value aggregation, and output projection. Dropout therefore removes
individual query/key contributions before values are mixed. The existing fused attention path
remains available to the bias-free constructors without probability dropout.
-/
def forward {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {batch n modelWidth heads headWidth : Nat}
    (queryWeight keyWeight valueWeight :
      Ref (m := m) (α := α) [modelWidth, heads * headWidth])
    (outputWeight : Ref (m := m) (α := α) [heads * headWidth, modelWidth])
    (queryBias keyBias valueBias : Option (Ref (m := m) (α := α) [heads * headWidth]))
    (outputBias : Option (Ref (m := m) (α := α) [modelWidth]))
    (probability : Option (Ref (m := m) (α := α) .scalar))
    (input : Ref (m := m) (α := α) [batch, n, modelWidth])
    (mask : Option (Tensor Bool [n, n])) (seed : Nat) (training : Bool) :
    m (Ref (m := m) (α := α) [batch, n, modelWidth]) := do
  let project := fun
      (weight : Ref (m := m) (α := α) [modelWidth, heads * headWidth])
      (bias : Option (Ref (m := m) (α := α) [heads * headWidth])) => do
    let projected ← matmul (m := m) (α := α) (batchA := [batch]) (batchB := [])
      (batch := [batch]) (mDim := n) (nDim := modelWidth) (pDim := heads * headWidth)
      input weight
    let affine ← match bias with
      | none => pure projected
      | some bias => do
          let broadcast ← broadcastTo (m := m) (α := α)
            (s₁ := [heads * headWidth]) (s₂ := [batch, n, heads * headWidth])
            Shape.BroadcastTo.proof bias
          add (m := m) (α := α) (s := [batch, n, heads * headWidth]) projected broadcast
    let split ← reshape (m := m) (α := α)
      (s₁ := [batch, n, heads * headWidth]) (s₂ := [batch, n, heads, headWidth])
      affine (by simp [Shape.size])
    swapAdjacentAtDepth (m := m) (α := α) (s := [batch, n, heads, headWidth]) 1 split
  let query ← project queryWeight queryBias
  let key ← project keyWeight keyBias
  let value ← project valueWeight valueBias
  let keyTranspose ← swapAdjacentAtDepth (m := m) (α := α)
    (s := [batch, heads, n, headWidth]) 2 key
  let dotProducts ← matmul (m := m) (α := α)
    (batchA := [batch, heads]) (batchB := [batch, heads]) (batch := [batch, heads])
    (mDim := n) (nDim := headWidth) (pDim := n) query keyTranspose
  let scores ← scale (m := m) (α := α) (s := [batch, heads, n, n]) dotProducts
    ((1 : α) / MathFunctions.sqrt (headWidth : α))
  let weights ← probabilities (m := m) (α := α)
    (batch := batch) (heads := heads) (n := n) scores mask
  let weights ← match probability with
    | none => pure weights
    | some probability =>
        F.dropoutRefSeeded (m := m) (α := α) (s := [batch, heads, n, n])
          weights probability seed training
  let attended ← matmul (m := m) (α := α)
    (batchA := [batch, heads]) (batchB := [batch, heads]) (batch := [batch, heads])
    (mDim := n) (nDim := n) (pDim := headWidth) weights value
  let tokenMajor ← swapAdjacentAtDepth (m := m) (α := α)
    (s := [batch, heads, n, headWidth]) 1 attended
  let joined ← reshape (m := m) (α := α)
    (s₁ := [batch, n, heads, headWidth]) (s₂ := [batch, n, heads * headWidth])
    tokenMajor (by simp [Shape.size])
  let output ← matmul (m := m) (α := α) (batchA := [batch]) (batchB := [])
    (batch := [batch]) (mDim := n) (nDim := heads * headWidth) (pDim := modelWidth)
    joined outputWeight
  match outputBias with
  | none => pure output
  | some bias => do
      let broadcast ← broadcastTo (m := m) (α := α)
        (s₁ := [modelWidth]) (s₂ := [batch, n, modelWidth]) Shape.BroadcastTo.proof bias
      add (m := m) (α := α) (s := [batch, n, modelWidth]) output broadcast

/-- Include one state slot only when the corresponding option is enabled. -/
def optionalShapes (enabled : Bool) (shape : Shape) : List Shape :=
  if enabled then [shape] else []

/-- Initial value for an optional state slot, with no placeholder when it is disabled. -/
def optionalState (enabled : Bool) {shape : Shape} (value : Tensor Float shape) :
    TensorPack Float (optionalShapes enabled shape) :=
  match enabled with
  | true => .cons value .nil
  | false => .nil

/-- Native initialization follows the same optional shape list as the ordinary state pack. -/
def optionalInit (enabled : Bool) (shape : Shape) (init : Module.RuntimeInit.FloatInit) :
    Module.RuntimeInit.Plan (optionalShapes enabled shape) :=
  match enabled with
  | true => .cons init .nil
  | false => .nil

/-- Read one optional state slot and return the remaining, still shape-indexed references. -/
def takeOptional {Ref : Shape → Type} (enabled : Bool) (shape : Shape) {rest : List Shape} :
    RefList Ref (optionalShapes enabled shape ++ rest) → Option (Ref shape) × RefList Ref rest :=
  match enabled with
  | true => fun (.cons value remaining) => (some value, remaining)
  | false => fun remaining => (none, remaining)

end Attention

/--
Multi-head self-attention layer for a sequence
`(sequenceLength × modelWidth) → (sequenceLength × modelWidth)`.

This layer packs the four projection matrices `(Wq, Wk, Wv, Wo)` and calls the TorchLean attention
primitive. An optional boolean mask of shape `(sequenceLength × sequenceLength)` can be provided,
for example for causal masking.

PyTorch analogy: `torch.nn.MultiheadAttention(embed_dim=modelWidth, num_heads=headCount)` in
self-attention mode.
-/
def multiHeadAttention
    (batchSize sequenceLength modelWidth headCount headWidth : Nat)
    {sequenceLengthNonzero : sequenceLength ≠ 0}
    (queryWeightSeed keyWeightSeed valueWeightSeed outputWeightSeed : Nat := 0)
    (weightInitialization? : Option Torch.Init.Scheme := none)
    (outputWeightInitialization? : Option Torch.Init.Scheme := none)
    (mask : Option (Tensor Bool [sequenceLength, sequenceLength]) := none) :
    Layer [batchSize, sequenceLength, modelWidth] [batchSize, sequenceLength, modelWidth] :=
  let projectionWidth := headCount * headWidth
  let projectionWeightShape : Shape := [modelWidth, projectionWidth]
  let outputWeightShape : Shape := [projectionWidth, modelWidth]
  let projectionInitialization :=
    weightInitialization?.getD (.xavierUniform modelWidth projectionWidth)
  let outputInitialization :=
    outputWeightInitialization?.orElse (fun _ => weightInitialization?)
      |>.getD (.xavierUniform projectionWidth modelWidth)
  let initialQueryWeight : Tensor Float projectionWeightShape :=
    Torch.Init.tensor projectionInitialization (seed := queryWeightSeed)
  let initialKeyWeight : Tensor Float projectionWeightShape :=
    Torch.Init.tensor projectionInitialization (seed := keyWeightSeed)
  let initialValueWeight : Tensor Float projectionWeightShape :=
    Torch.Init.tensor projectionInitialization (seed := valueWeightSeed)
  let initialOutputWeight : Tensor Float outputWeightShape :=
    Torch.Init.tensor outputInitialization (seed := outputWeightSeed)
  { kind := s!"MultiHeadAttention(heads={headCount}, headWidth={headWidth})"
    stateShapes :=
      [projectionWeightShape, projectionWeightShape, projectionWeightShape, outputWeightShape]
    initState :=
      .cons initialQueryWeight <|
      .cons initialKeyWeight <|
      .cons initialValueWeight <|
      .cons initialOutputWeight .nil
    runtimeInit := some <|
      .cons (Module.RuntimeInit.FloatInit.ofScheme projectionInitialization queryWeightSeed) <|
      .cons (Module.RuntimeInit.FloatInit.ofScheme projectionInitialization keyWeightSeed) <|
      .cons (Module.RuntimeInit.FloatInit.ofScheme projectionInitialization valueWeightSeed) <|
      .cons (Module.RuntimeInit.FloatInit.ofScheme outputInitialization outputWeightSeed) .nil
    requiresGrad := #[true, true, true, true]
    validateConfig := do
      if modelWidth = 0 then
        throw "MultiHeadAttention: model width must be positive"
      if headCount = 0 then
        throw "MultiHeadAttention: head count must be positive"
      if headWidth = 0 then
        throw "MultiHeadAttention: head width must be positive"
      projectionInitialization.validate
      outputInitialization.validate
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun queryWeight keyWeight valueWeight outputWeight input =>
          Runtime.Autograd.Model.multiHeadAttention (m := m) (α := α)
            (leadingShape := [batchSize]) (n := sequenceLength) (numHeads := headCount)
            (dModel := modelWidth) (headDim := headWidth)
            (hN := sequenceLengthNonzero)
            queryWeight keyWeight valueWeight outputWeight input (mask := mask)
  }

/--
Multi-head self-attention with a trainable bias on the final output projection.

The Q/K/V projections remain bias-free.  This is the parameterization used in Karpathy's
educational GPT implementation: the three per-head projections are linear maps without bias,
while the projection applied after concatenating the heads is affine.
-/
def multiHeadAttentionOutputBias
    (batchSize sequenceLength modelWidth headCount headWidth : Nat)
    {sequenceLengthNonzero : sequenceLength ≠ 0}
    (queryWeightSeed keyWeightSeed valueWeightSeed outputWeightSeed : Nat := 0)
    (weightInitialization? : Option Torch.Init.Scheme := none)
    (outputWeightInitialization? : Option Torch.Init.Scheme := none)
    (mask : Option (Tensor Bool [sequenceLength, sequenceLength]) := none) :
    Layer [batchSize, sequenceLength, modelWidth] [batchSize, sequenceLength, modelWidth] :=
  let projectionWidth := headCount * headWidth
  let projectionWeightShape : Shape := [modelWidth, projectionWidth]
  let outputWeightShape : Shape := [projectionWidth, modelWidth]
  let outputBiasShape : Shape := [modelWidth]
  let projectionInitialization :=
    weightInitialization?.getD (.xavierUniform modelWidth projectionWidth)
  let outputInitialization :=
    outputWeightInitialization?.orElse (fun _ => weightInitialization?)
      |>.getD (.xavierUniform projectionWidth modelWidth)
  let initialQueryWeight : Tensor Float projectionWeightShape :=
    Torch.Init.tensor projectionInitialization (seed := queryWeightSeed)
  let initialKeyWeight : Tensor Float projectionWeightShape :=
    Torch.Init.tensor projectionInitialization (seed := keyWeightSeed)
  let initialValueWeight : Tensor Float projectionWeightShape :=
    Torch.Init.tensor projectionInitialization (seed := valueWeightSeed)
  let initialOutputWeight : Tensor Float outputWeightShape :=
    Torch.Init.tensor outputInitialization (seed := outputWeightSeed)
  let initialOutputBias : Tensor Float outputBiasShape := Tensor.zeros (α := Float) outputBiasShape
  { kind := s!"MultiHeadAttention(heads={headCount}, headWidth={headWidth}, outputBias=true)"
    stateShapes :=
      [projectionWeightShape, projectionWeightShape, projectionWeightShape, outputWeightShape,
        outputBiasShape]
    initState :=
      .cons initialQueryWeight <|
      .cons initialKeyWeight <|
      .cons initialValueWeight <|
      .cons initialOutputWeight <|
      .cons initialOutputBias .nil
    runtimeInit := some <|
      .cons (Module.RuntimeInit.FloatInit.ofScheme projectionInitialization queryWeightSeed) <|
      .cons (Module.RuntimeInit.FloatInit.ofScheme projectionInitialization keyWeightSeed) <|
      .cons (Module.RuntimeInit.FloatInit.ofScheme projectionInitialization valueWeightSeed) <|
      .cons (Module.RuntimeInit.FloatInit.ofScheme outputInitialization outputWeightSeed) <|
      .cons .zeros .nil
    requiresGrad := #[true, true, true, true, true]
    validateConfig := do
      if modelWidth = 0 then
        throw "MultiHeadAttention: model width must be positive"
      if headCount = 0 then
        throw "MultiHeadAttention: head count must be positive"
      if headWidth = 0 then
        throw "MultiHeadAttention: head width must be positive"
      projectionInitialization.validate
      outputInitialization.validate
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun queryWeight keyWeight valueWeight outputWeight outputBias input =>
          Runtime.Autograd.Model.multiHeadAttentionOutputBias (m := m) (α := α)
            (leadingShape := [batchSize]) (n := sequenceLength) (numHeads := headCount)
            (dModel := modelWidth) (headDim := headWidth) sequenceLengthNonzero
            queryWeight keyWeight valueWeight outputWeight outputBias input (mask := mask) }

/--
Configurable affine self-attention with optional probability dropout.

The first four state slots remain `queryWeight, keyWeight, valueWeight, outputWeight`.
Enabling input bias appends `queryBias, keyBias, valueBias`; enabling output bias appends
`outputBias`; enabling dropout appends its non-trainable scalar probability. Disabled options
allocate no state slots. All biases start at zero and consume no initialization seeds.
-/
def multiHeadAttentionConfigured
    (batchSize sequenceLength modelWidth headCount headWidth : Nat)
    {sequenceLengthNonzero : sequenceLength ≠ 0}
    (queryWeightSeed keyWeightSeed valueWeightSeed outputWeightSeed : Nat := 0)
    (weightInitialization? : Option Torch.Init.Scheme := none)
    (outputWeightInitialization? : Option Torch.Init.Scheme := none)
    (mask : Option (Tensor Bool [sequenceLength, sequenceLength]) := none)
    (inputBias outputBias : Bool := false)
    (dropout? : Option Float := none) (dropoutSeed : Nat := 0) :
    Layer [batchSize, sequenceLength, modelWidth] [batchSize, sequenceLength, modelWidth] :=
  let base := multiHeadAttention batchSize sequenceLength modelWidth headCount headWidth
    (sequenceLengthNonzero := sequenceLengthNonzero)
    queryWeightSeed keyWeightSeed valueWeightSeed outputWeightSeed
    weightInitialization? outputWeightInitialization? mask
  let projectionShape : Shape := [headCount * headWidth]
  let outputShape : Shape := [modelWidth]
  let extraShapes :=
    Attention.optionalShapes inputBias projectionShape ++
    (Attention.optionalShapes inputBias projectionShape ++
    (Attention.optionalShapes inputBias projectionShape ++
    (Attention.optionalShapes outputBias outputShape ++
    (Attention.optionalShapes dropout?.isSome .scalar ++ []))))
  let projectionBias :=
    Attention.optionalState inputBias (Tensor.zeros (α := Float) projectionShape)
  let extraState : TensorPack Float extraShapes :=
    projectionBias.append <| projectionBias.append <| projectionBias.append <|
      (Attention.optionalState outputBias (Tensor.zeros (α := Float) outputShape)).append <|
      (Attention.optionalState dropout?.isSome (Tensor.scalar (dropout?.getD 0.0))).append .nil
  let projectionBiasInit := Attention.optionalInit inputBias projectionShape .zeros
  let extraInit : Module.RuntimeInit.Plan extraShapes :=
    projectionBiasInit.append <| projectionBiasInit.append <| projectionBiasInit.append <|
      (Attention.optionalInit outputBias outputShape .zeros).append <|
      (Attention.optionalInit dropout?.isSome .scalar
        (.flat (FloatArray.mk #[dropout?.getD 0.0]))).append .nil
  { kind := s!"MultiHeadAttention(heads={headCount}, headWidth={headWidth}, " ++
      s!"inputBias={inputBias}, outputBias={outputBias})"
    stateShapes := base.stateShapes ++ extraShapes
    initState := base.initState.append extraState
    runtimeInit := base.runtimeInit.map (fun plan => plan.append extraInit)
    requiresGrad := #[true, true, true, true] ++
      (if inputBias then #[true, true, true] else #[]) ++
      (if outputBias then #[true] else #[]) ++
      (if dropout?.isSome then #[false] else #[])
    validateConfig := do
      base.validateConfig
      match dropout? with
      | none => pure ()
      | some probability =>
          unless probability.isFinite && 0.0 <= probability && probability <= 1.0 do
            throw <| "MultiHeadAttention: dropout probability must be finite and in [0, 1], " ++
              s!"got {probability}"
    forward := fun mode {α} _ _ => fun {m} _ _ =>
      Runtime.Autograd.Torch.CurriedRef.curry
        (Ref := fun shape => Ref (m := m) (α := α) shape)
        (ss := (base.stateShapes ++ extraShapes) ++ [[batchSize, sequenceLength, modelWidth]])
        (β := m (Ref (m := m) (α := α) [batchSize, sequenceLength, modelWidth]))
        fun arguments => do
          let (state, input) := Runtime.Autograd.Torch.RefList.splitLast arguments
          let .cons queryWeight (.cons keyWeight (.cons valueWeight (.cons outputWeight extra))) :=
            state
          let (queryBias, extra) := Attention.takeOptional inputBias projectionShape extra
          let (keyBias, extra) := Attention.takeOptional inputBias projectionShape extra
          let (valueBias, extra) := Attention.takeOptional inputBias projectionShape extra
          let (outputBias, extra) := Attention.takeOptional outputBias outputShape extra
          let (probability, _) := Attention.takeOptional dropout?.isSome .scalar extra
          Attention.forward (m := m) (α := α)
            (batch := batchSize) (n := sequenceLength) (modelWidth := modelWidth)
            (heads := headCount) (headWidth := headWidth)
            queryWeight keyWeight valueWeight outputWeight queryBias keyBias valueBias
            outputBias probability input mask dropoutSeed (mode == .train) }

end Layers

end Model
end Autograd
end Runtime

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
Attention with optional affine projection biases and dropout on softmax probabilities.

The operation order is projection, head splitting, scaled dot products, masked softmax,
probability dropout, value aggregation, and output projection. Dropout therefore removes
individual query/key contributions before values are mixed.
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
  let scores ← matmul (m := m) (α := α)
    (batchA := [batch, heads]) (batchB := [batch, heads]) (batch := [batch, heads])
    (mDim := n) (nDim := headWidth) (pDim := n) query keyTranspose
    (scale := some ((1 : α) / Spec.attentionScaleDenom headWidth))
  let ones ← const (m := m) (α := α) (Tensor.full [batch, heads, n, n] (1 : α))
  let multiplier ← match probability with
    | none => pure ones
    | some probability =>
        F.dropoutRefSeeded (m := m) (α := α) (s := [batch, heads, n, n])
          ones probability seed training
  let attended ← Torch.softmaxMatmul (m := m) (α := α) (batch := [batch, heads])
    (rows := n) (keys := n) (width := headWidth) scores value multiplier mask
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
Configurable affine self-attention with optional probability dropout.

The first four state slots remain `queryWeight, keyWeight, valueWeight, outputWeight`.
Enabling input bias appends `queryBias, keyBias, valueBias`; enabling output bias appends
`outputBias`; enabling dropout appends its non-trainable scalar probability. Disabled options
allocate no state slots. All biases start at zero and consume no initialization seeds.
-/
def attention
    (batchSize sequenceLength modelWidth headCount headWidth : Nat)
    {sequenceLengthNonzero : sequenceLength ≠ 0}
    (queryWeightSeed keyWeightSeed valueWeightSeed outputWeightSeed : Nat := 0)
    (weightInitialization? : Option Torch.Init.Scheme := none)
    (outputWeightInitialization? : Option Torch.Init.Scheme := none)
    (mask : Option (Tensor Bool [sequenceLength, sequenceLength]) := none)
    (inputBias outputBias : Bool := false)
    (dropout? : Option Float := none) (dropoutSeed : Nat := 0) :
    Layer [batchSize, sequenceLength, modelWidth] [batchSize, sequenceLength, modelWidth] :=
  let projectionWidth := headCount * headWidth
  let projectionWeightShape : Shape := [modelWidth, projectionWidth]
  let outputWeightShape : Shape := [projectionWidth, modelWidth]
  let projectionInitialization :=
    weightInitialization?.getD (.xavierUniform modelWidth projectionWidth)
  let outputInitialization :=
    outputWeightInitialization?.orElse (fun _ => weightInitialization?)
      |>.getD (.xavierUniform projectionWidth modelWidth)
  let weightShapes :=
    [projectionWeightShape, projectionWeightShape, projectionWeightShape, outputWeightShape]
  let weightState : TensorPack Float weightShapes :=
    .cons (Torch.Init.tensor projectionInitialization (seed := queryWeightSeed)) <|
    .cons (Torch.Init.tensor projectionInitialization (seed := keyWeightSeed)) <|
    .cons (Torch.Init.tensor projectionInitialization (seed := valueWeightSeed)) <|
    .cons (Torch.Init.tensor outputInitialization (seed := outputWeightSeed)) .nil
  let weightInit : Module.RuntimeInit.Plan weightShapes :=
    .cons (Module.RuntimeInit.FloatInit.ofScheme projectionInitialization queryWeightSeed) <|
    .cons (Module.RuntimeInit.FloatInit.ofScheme projectionInitialization keyWeightSeed) <|
    .cons (Module.RuntimeInit.FloatInit.ofScheme projectionInitialization valueWeightSeed) <|
    .cons (Module.RuntimeInit.FloatInit.ofScheme outputInitialization outputWeightSeed) .nil
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
    stateShapes := weightShapes ++ extraShapes
    initState := weightState.append extraState
    runtimeInit := some (weightInit.append extraInit)
    requiresGrad := #[true, true, true, true] ++
      (if inputBias then #[true, true, true] else #[]) ++
      (if outputBias then #[true] else #[]) ++
      (if dropout?.isSome then #[false] else #[])
    validateConfig := do
      if modelWidth = 0 then
        throw "Attention: model width must be positive"
      if headCount = 0 then
        throw "Attention: head count must be positive"
      if headWidth = 0 then
        throw "Attention: head width must be positive"
      projectionInitialization.validate
      outputInitialization.validate
      match dropout? with
      | none => pure ()
      | some probability =>
          unless probability.isFinite && 0.0 <= probability && probability <= 1.0 do
            throw <| "MultiHeadAttention: dropout probability must be finite and in [0, 1], " ++
              s!"got {probability}"
    forward := fun mode {α} _ _ => fun {m} _ _ =>
      Runtime.Autograd.Torch.CurriedRef.curry
        (Ref := fun shape => Ref (m := m) (α := α) shape)
        (ss := (weightShapes ++ extraShapes) ++ [[batchSize, sequenceLength, modelWidth]])
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
          if !inputBias && dropout?.isNone then
            Runtime.Autograd.Model.attention (m := m) (α := α)
              (leadingShape := [batchSize]) (n := sequenceLength) (numHeads := headCount)
              (dModel := modelWidth) (headDim := headWidth) (hN := sequenceLengthNonzero)
              queryWeight keyWeight valueWeight outputWeight input (mask := mask)
              (outputBias := outputBias)
          else
            Attention.forward (m := m) (α := α)
              (batch := batchSize) (n := sequenceLength) (modelWidth := modelWidth)
              (heads := headCount) (headWidth := headWidth)
              queryWeight keyWeight valueWeight outputWeight queryBias keyBias valueBias
              outputBias probability input mask dropoutSeed (mode == .train) }

end Layers

end Model
end Autograd
end Runtime

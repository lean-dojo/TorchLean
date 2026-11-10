/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Layer
public import TorchLean.NN.Functional

/-!
# TorchLean NN: Attention

`Layer.attention` is multi-head self-attention with optional projection biases and optional
dropout on the attention probabilities. Without input biases and without dropout its forward pass
is `functional.attention`, and with either of them it is `Attention.forward`.
-/

@[expose] public section

namespace TorchLean
namespace nn.Layer

open TorchLean.Tensor
open Ops

namespace Attention

/--
Attention with optional affine projection biases and dropout on softmax probabilities.

The operation order is projection, head splitting, scaled dot products, masked softmax,
probability dropout, value aggregation, and output projection. Dropout therefore removes
individual query/key contributions before values are mixed. With `dropout? = some p` in training,
the probabilities are multiplied by `functional.dropout` of ones: `1 / (1 - p)` where a key is kept,
`+0` where it is dropped and everywhere when `p = 1`. A kept probability `w` thus becomes the
rounded product of `w` and the rounded `1 / (1 - p)`, one rounding more than `Op.dropout`'s
`w / (1 - p)`, and with `p = 1` a NaN or infinite `w` gives NaN where `Op.dropout` gives `+0`.
-/
def forward {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {batch n modelWidth heads headWidth : Nat}
    (queryWeight keyWeight valueWeight :
      Ref (m := m) (α := α) [modelWidth, heads * headWidth])
    (outputWeight : Ref (m := m) (α := α) [heads * headWidth, modelWidth])
    (queryBias keyBias valueBias : Option (Ref (m := m) (α := α) [heads * headWidth]))
    (outputBias : Option (Ref (m := m) (α := α) [modelWidth]))
    (dropout? : Option Float)
    (input : Ref (m := m) (α := α) [batch, n, modelWidth])
    (mask : Option (Tensor Bool [n, n])) (seed : Nat) (training : Bool) :
    m (Ref (m := m) (α := α) [batch, n, modelWidth]) := do
  let project := fun
      (weight : Ref (m := m) (α := α) [modelWidth, heads * headWidth])
      (bias : Option (Ref (m := m) (α := α) [heads * headWidth])) => do
    let projected ← Ops.apply (m := m) (α := α)
        (.matmul (batchA := [batch]) (batchB := []) (batch := [batch]) (m := n) (n := modelWidth)
          (p := heads * headWidth) false false (fun h => nomatch h) (fun h => nomatch h)
          Shape.BroadcastTo.proof Shape.BroadcastTo.proof) (input, weight, ())
    let affine ← match bias with
      | none => pure projected
      | some bias => do
          let broadcast ← Ops.apply (m := m) (α := α)
              (.broadcast (s := [heads * headWidth]) (t := [batch, n, heads * headWidth])
                (Shape.BroadcastTo.proof).dims (Shape.BroadcastTo.proof).broadcastDims) (bias, ())
          Ops.apply (m := m) (α := α)
              (.add (s := [batch, n, heads * headWidth])) (projected, broadcast, ())
    let split ← Ops.apply (m := m) (α := α)
        (.reshape (s := [batch, n, heads * headWidth]) (t := [batch, n, heads, headWidth])
          (by simp)) (affine, ())
    Ops.apply (m := m) (α := α)
        (.permute (s := [batch, n, heads, headWidth])
          (Shape.adjacentSwap [batch, n, heads, headWidth] 1)
          (Shape.swapAdjacentAtDepth [batch, n, heads, headWidth] 1) (by tensor_shape)) (split, ())
  let query ← project queryWeight queryBias
  let key ← project keyWeight keyBias
  let value ← project valueWeight valueBias
  let ones ← Ops.const (m := m) (α := α) (Tensor.full [batch, heads, n, n] (1 : α))
  let multiplier ← match dropout? with
    | none => pure ones
    | some p => functional.dropout (m := m) (α := α) (s := [batch, heads, n, n]) ones p
                  seed training
  -- `Tensor.attention` with `weights := some multiplier`, in three steps: the key transpose, the
  -- scaled score product and the weighted softmax product.
  let keyTranspose ← Ops.apply (m := m) (α := α)
    (.permute (s := [batch, heads, n, headWidth])
      (Shape.adjacentSwap [batch, heads, n, headWidth] 2)
      (Shape.swapAdjacentAtDepth [batch, heads, n, headWidth] 2) (by tensor_shape)) (key, ())
  let scores ← Ops.apply (m := m) (α := α)
    (.scaledMatmul (batchA := [batch, heads]) (batchB := [batch, heads]) (batch := [batch, heads])
      (m := n) (n := headWidth) (p := n) false false (fun h => nomatch h) (fun h => nomatch h)
      (Shape.CanBroadcastTo.refl _) (Shape.CanBroadcastTo.refl _))
    (query, keyTranspose, .data (Ops.dataConst (m := m) (α := α)
      (Tensor.scalar ((1 : α) / Attention.scaleDenom headWidth))), ())
  let attended ← Ops.apply (m := m) (α := α)
    (.attentionFromScores (batch := [batch, heads]) (rows := n) (keys := n) (width := headWidth)
      mask)
    (scores, value, multiplier, ())
  let tokenMajor ← Ops.apply (m := m) (α := α)
      (.permute (s := [batch, heads, n, headWidth])
        (Shape.adjacentSwap [batch, heads, n, headWidth] 1)
        (Shape.swapAdjacentAtDepth [batch, heads, n, headWidth] 1) (by tensor_shape)) (attended, ())
  let joined ← Ops.apply (m := m) (α := α)
      (.reshape (s := [batch, n, heads, headWidth]) (t := [batch, n, heads * headWidth])
        (by simp)) (tokenMajor, ())
  let output ← Ops.apply (m := m) (α := α)
      (.matmul (batchA := [batch]) (batchB := []) (batch := [batch]) (m := n)
        (n := heads * headWidth) (p := modelWidth) false false (fun h => nomatch h)
        (fun h => nomatch h) Shape.BroadcastTo.proof Shape.BroadcastTo.proof)
      (joined, outputWeight, ())
  match outputBias with
  | none => pure output
  | some bias => do
      let broadcast ← Ops.apply (m := m) (α := α)
          (.broadcast (s := [modelWidth]) (t := [batch, n, modelWidth])
            (Shape.BroadcastTo.proof).dims (Shape.BroadcastTo.proof).broadcastDims) (bias, ())
      Ops.apply (m := m) (α := α) (.add (s := [batch, n, modelWidth]))
          (output, broadcast, ())

/-- Include one state slot only when the corresponding option is enabled. -/
def optionalShapes (enabled : Bool) (shape : Shape) : List Shape :=
  if enabled then [shape] else []

/-- Initial value for an optional state slot, with no placeholder when it is disabled. -/
def optionalState (enabled : Bool) {shape : Shape} (value : Tensor Float shape) :
    TensorPack Float (optionalShapes enabled shape) :=
  match enabled with
  | true => .cons value .nil
  | false => .nil

/--
The runtime initialization plan of an optional state slot, over the shapes of `optionalShapes`:
`init` when the option is enabled and nothing otherwise.
-/
def optionalInit (enabled : Bool) (shape : Shape) (init : Init.Source) :
    Init.Plan (optionalShapes enabled shape) :=
  match enabled with
  | true => .cons init .nil
  | false => .nil

/--
Split one optional slot off a reference list: `some` of its reference when the slot is enabled and
`none` otherwise, together with the references after it.
-/
def takeOptional {Ref : Shape → Type} (enabled : Bool) (shape : Shape) {rest : List Shape} :
    Pack Ref (optionalShapes enabled shape ++ rest) → Option (Ref shape) × Pack Ref rest :=
  match enabled with
  | true => fun (.cons value remaining) => (some value, remaining)
  | false => fun remaining => (none, remaining)

/-- The initializations of the attention weights: `weight?` for the query, key and value weights
and `output?`, then `weight?`, for the output weight, each Xavier uniform by default. -/
def initializations (modelWidth projectionWidth : Nat) (weight? output? : Option Init.Scheme) :
    Init.Scheme × Init.Scheme :=
  (weight?.getD (.xavierUniform modelWidth projectionWidth),
    (output?.orElse fun _ => weight?).getD (.xavierUniform projectionWidth modelWidth))

end Attention

/--
Configurable affine self-attention with optional probability dropout.

The first four state slots are `queryWeight, keyWeight, valueWeight, outputWeight`. With
`inputBias` the slots `queryBias, keyBias, valueBias` follow, and with `outputBias` the slot
`outputBias` comes last. Disabled options allocate no state slots. The dropout probability
`dropout?` is configuration, which `validateConfig` checks, and allocates no state.

The query, key and value weights use `weightInit?`, by default Xavier uniform. The output
weight uses `outputWeightInit?`, then `weightInit?`, then Xavier uniform. All
biases start at zero and consume no initialization seeds.
-/
def attention
    (batchSize sequenceLength modelWidth headCount headWidth : Nat)
    {sequenceLengthNonzero : sequenceLength ≠ 0}
    (queryWeightSeed keyWeightSeed valueWeightSeed outputWeightSeed : Nat := 0)
    (weightInitialization? : Option Init.Scheme := none)
    (outputWeightInitialization? : Option Init.Scheme := none)
    (mask : Option (Tensor Bool [sequenceLength, sequenceLength]) := none)
    (inputBias outputBias : Bool := false)
    (dropout? : Option Float := none) (dropoutSeed : Nat := 0) :
    Layer [batchSize, sequenceLength, modelWidth] [batchSize, sequenceLength, modelWidth] :=
  let projectionWidth := headCount * headWidth
  let projectionWeightShape : Shape := [modelWidth, projectionWidth]
  let outputWeightShape : Shape := [projectionWidth, modelWidth]
  let (projectionInitialization, outputInitialization) :=
    Attention.initializations modelWidth projectionWidth weightInitialization?
      outputWeightInitialization?
  let weightShapes :=
    [projectionWeightShape, projectionWeightShape, projectionWeightShape, outputWeightShape]
  let weightState : TensorPack Float weightShapes :=
    .cons (Init.tensor projectionInitialization (seed := queryWeightSeed)) <|
    .cons (Init.tensor projectionInitialization (seed := keyWeightSeed)) <|
    .cons (Init.tensor projectionInitialization (seed := valueWeightSeed)) <|
    .cons (Init.tensor outputInitialization (seed := outputWeightSeed)) .nil
  let weightInit : Init.Plan weightShapes :=
    .cons (Init.Source.scheme projectionInitialization queryWeightSeed) <|
    .cons (Init.Source.scheme projectionInitialization keyWeightSeed) <|
    .cons (Init.Source.scheme projectionInitialization valueWeightSeed) <|
    .cons (Init.Source.scheme outputInitialization outputWeightSeed) .nil
  let projectionShape : Shape := [headCount * headWidth]
  let outputShape : Shape := [modelWidth]
  let extraShapes :=
    Attention.optionalShapes inputBias projectionShape ++
    (Attention.optionalShapes inputBias projectionShape ++
    (Attention.optionalShapes inputBias projectionShape ++
    (Attention.optionalShapes outputBias outputShape ++ [])))
  let projectionBias :=
    Attention.optionalState inputBias (Tensor.zeros (α := Float) projectionShape)
  let extraState : TensorPack Float extraShapes :=
    projectionBias.append <| projectionBias.append <| projectionBias.append <|
      (Attention.optionalState outputBias (Tensor.zeros (α := Float) outputShape)).append .nil
  let projectionBiasInit := Attention.optionalInit inputBias projectionShape (.scheme .zeros)
  let extraInit : Init.Plan extraShapes :=
    projectionBiasInit.append <| projectionBiasInit.append <| projectionBiasInit.append <|
      (Attention.optionalInit outputBias outputShape (.scheme .zeros)).append .nil
  { kind := s!"MultiHeadAttention(heads={headCount}, headWidth={headWidth}, " ++
      s!"inputBias={inputBias}, outputBias={outputBias})"
    stateShapes := weightShapes ++ extraShapes
    initState := weightState.append extraState
    runtimeInit := some (weightInit.append extraInit)
    requiresGrad := #[true, true, true, true] ++
      (if inputBias then #[true, true, true] else #[]) ++
      (if outputBias then #[true] else #[])
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
          unless functional.inUnitInterval probability do
            throw <| "MultiHeadAttention: dropout probability must be finite and in [0, 1], " ++
              s!"got {probability}"
    forward := fun mode {α} _ _ => fun {m} _ _ =>
      CurriedRef.curry
        (Ref := fun shape => Ref (m := m) (α := α) shape)
        (ss := (weightShapes ++ extraShapes) ++ [[batchSize, sequenceLength, modelWidth]])
        (β := m (Ref (m := m) (α := α) [batchSize, sequenceLength, modelWidth]))
        fun arguments => do
          let (state, input) := Pack.unsnoc arguments
          let .cons queryWeight (.cons keyWeight (.cons valueWeight (.cons outputWeight extra))) :=
            state
          let (queryBias, extra) := Attention.takeOptional inputBias projectionShape extra
          let (keyBias, extra) := Attention.takeOptional inputBias projectionShape extra
          let (valueBias, extra) := Attention.takeOptional inputBias projectionShape extra
          let (outputBias, _) := Attention.takeOptional outputBias outputShape extra
          if !inputBias && dropout?.isNone then
            functional.attention (m := m) (α := α)
              (leadingShape := [batchSize]) (n := sequenceLength) (numHeads := headCount)
              (dModel := modelWidth) (headDim := headWidth) (hN := sequenceLengthNonzero)
              queryWeight keyWeight valueWeight outputWeight input (mask := mask)
              (outputBias := outputBias)
          else
            Attention.forward (m := m) (α := α)
              (batch := batchSize) (n := sequenceLength) (modelWidth := modelWidth)
              (heads := headCount) (headWidth := headWidth)
              queryWeight keyWeight valueWeight outputWeight queryBias keyBias valueBias
              outputBias dropout? input mask dropoutSeed (mode == .train) }

end nn.Layer
end TorchLean

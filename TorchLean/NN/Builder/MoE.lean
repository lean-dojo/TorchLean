/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Builder.Blocks
public import TorchLean.NN.Builder.Sequential
public import TorchLean.NN.Builder

/-!
# Mixture of experts

`nn.moe k router experts` applies `TorchLean.MoE.forward` to every token of a `[tokens, width]`
input. The router scores the experts, the `k` experts with the largest softmax probabilities are
kept, ties going to the lower index, and the outputs of the kept experts are added in increasing
expert index, each weighted by its probability divided by the total probability of the kept
experts. Over `ℝ` this is Mixtral's sparse block (`TorchLean.MoE`).

Every expert runs on every token and the experts that are not kept get gate zero, so all shapes
are static. When the router and the experts act on each token separately, as linear layers and
MLPs over the token axis do, a token's output depends only on that token, whatever else is in its
batch. `nn.moe` accepts any `[tokens, width]` networks, and one that mixes tokens, such as
attention, gives up that independence.

`Layer.moe` is the one layer: the router and the experts are two branches over the same input,
and the experts' outputs come stacked along a leading expert axis. `nn.moe` stacks any experts
with `stackBranches`. `nn.moeMLP` passes an `nn.linear` router and `mlpExperts`, two-layer
MLP experts that keep their weights in single tensors and run with one batched product per layer,
so a step issues a few dozen operations however many experts there are.
-/

@[expose] public section

namespace TorchLean
namespace nn

open Ops

/-- The outputs of `E + 1` branches that read the same input, stacked in branch order along a
leading axis of length `E + 1`. -/
def stackBranches {σ τ : Shape} :
    (E : Nat) → (Fin (E + 1) → Sequential σ τ) → Sequential σ (Shape.dim (E + 1) τ)
  | 0, branches =>
      Sequential.comp (branches 0)
        (Sequential.fromLayer (Layer.op
          (.reshape (t := Shape.dim 1 τ) (Nat.one_mul _).symm)))
  | E + 1, branches =>
      Nat.add_comm 1 (E + 1) ▸ concatBranches
        (Sequential.comp (branches 0)
          (Sequential.fromLayer (Layer.op
            (.reshape (t := Shape.dim 1 τ) (Nat.one_mul _).symm))))
        (stackBranches E fun i => branches i.succ)

/--
The gates of `nn.moe` for every token of `logits`: `MoE.gates k` on each row. The kept
probabilities of a row are summed by one product with a column of ones, a left fold from zero as
`Tensor.sum` is.
-/
def moeGates {α : Type} [TorchLean.Storage α] [Context α] {m : Type → Type} [Monad m]
    [Ops (m := m) (α := α)] {tokens E : Nat} (k : Nat)
    (logits : Ref (m := m) (α := α) [tokens, E + 1]) :
    m (Ref (m := m) (α := α) [tokens, E + 1]) := do
  let probabilities ← Ops.apply (m := m) (α := α)
      (.softmax ⟨1, Shape.AxisInBounds.proof⟩) (logits, ())
  let mask ← Ops.apply (m := m) (α := α)
    (.topKMask ⟨1, Shape.AxisInBounds.proof⟩ k) (probabilities, ())
  let kept ← Ops.apply (m := m) (α := α) .mul (probabilities, mask, ())
  let ones ← Ops.const (m := m) (α := α)
    (TorchLean.Tensor.full [E + 1, 1] (1 : α))
  let total ← Ops.apply (m := m) (α := α)
      (.matmul (batchA := []) (batchB := []) (batch := []) (m := tokens) (n := E + 1) (p := 1) false
        false (fun h => nomatch h) (fun h => nomatch h) Shape.BroadcastTo.proof
        Shape.BroadcastTo.proof) (kept, ones, ())
  let totalB ← Ops.apply (m := m) (α := α)
      (.broadcast (s := [tokens, 1]) (t := [tokens, E + 1]) (Shape.BroadcastTo.proof).dims
        (Shape.BroadcastTo.proof).broadcastDims) (total, ())
  Ops.apply (m := m) (α := α) .div (kept, totalB, ())

/--
The gating and mixing of `Layer.moe`: from the router's logits `[tokens, E + 1]` and the experts'
outputs stacked as `[E + 1, tokens, width]`, compute `MoE.combine` of
`MoE.gates k` for every token, as one batched product of each token's gates with its
expert outputs, summed in increasing expert index from zero.
-/
def moeMix {α : Type} [TorchLean.Storage α] [Context α] {m : Type → Type} [Monad m]
    [Ops (m := m) (α := α)] {tokens width E : Nat} (k : Nat)
    (logits : Ref (m := m) (α := α) [tokens, E + 1])
    (outputs : Ref (m := m) (α := α) [E + 1, tokens, width]) :
    m (Ref (m := m) (α := α) [tokens, width]) := do
  let gates ← nn.moeGates (m := m) (α := α) k logits
  let byToken ← Ops.apply (m := m) (α := α)
      (.permute (s := [E + 1, tokens, width])
        (Shape.adjacentSwap [E + 1, tokens, width] 0)
        (Shape.swapAdjacentAtDepth [E + 1, tokens, width] 0) (by tensor_shape))
      (outputs, ())
  let gateRows ← Ops.apply (m := m) (α := α)
      (.reshape (s := [tokens, E + 1]) (t := [tokens, 1, E + 1])
        (by simp)) (gates, ())
  let mixed ← Ops.apply (m := m) (α := α)
      (.matmul (batchA := [tokens]) (batchB := [tokens]) (batch := [tokens]) (m := 1)
        (n := E + 1) (p := width) false false (fun h => nomatch h) (fun h => nomatch h)
        Shape.BroadcastTo.proof Shape.BroadcastTo.proof) (gateRows, byToken, ())
  Ops.apply (m := m) (α := α)
      (.reshape (s := [tokens, 1, width]) (t := [tokens, width])
        (by simp)) (mixed, ())

/-- The mixture-of-experts layer: `router` scores the experts, `experts` returns their outputs
stacked along a leading expert axis, and `moeMix` gates and mixes them. Its parameters
are the router's followed by the experts'. Validation rejects `k = 0`. -/
def Layer.moe {tokens width outWidth E : Nat} (k : Nat)
    (router : Sequential [tokens, width] [tokens, E + 1])
    (experts : Sequential [tokens, width] [E + 1, tokens, outWidth]) :
    Layer [tokens, width] [tokens, outWidth] :=
  let layer := Layer.combineBranches s!"MoE({E + 1} experts, top {k})" router experts
    fun {α} _ _ {m} _ _ logits outputs => moeMix (m := m) (α := α) k logits outputs
  { layer with
    validateConfig := do
      if k = 0 then throw "MoE: a mixture must keep at least one expert per token"
      layer.validateConfig }

/--
A mixture-of-experts layer over the tokens of a `[tokens, width]` input, as in Mixtral: `router`
scores the experts, the `k` with the largest softmax probabilities are kept (ties to the lower
index), and their outputs are added in increasing expert index, each weighted by its probability
divided by the kept total. `MoE.forward` is the layer on one token. The model is
`Layer.moe` with the experts stacked by `stackBranches`, and its parameters are the router's
followed by each expert's, in order. `k` must be positive.

Example:
```lean
-- Eight experts, two kept per token, on sequences of 16 tokens of width 32.
def model : nn.Builder (nn.Sequential [16, 32] [16, 32]) := do
  let router ← nn.linear 32 8 (batchShape := [16])
  let experts ← (List.range 8).mapM fun _ =>
    nn.Sequential![nn.linear 32 64 (batchShape := [16]), nn.op .relu,
      nn.linear 64 32 (batchShape := [16])]
  pure (nn.moe 2 router fun e => experts.getD e.val (nn.Sequential.id _))
```
-/
def moe {tokens width outWidth E : Nat} (k : Nat)
    (router : Sequential [tokens, width] [tokens, E + 1])
    (experts : Fin (E + 1) → Sequential [tokens, width] [tokens, outWidth]) :
    Sequential [tokens, width] [tokens, outWidth] :=
  nn.Sequential.fromLayer (Layer.moe k router (stackBranches E experts))

/-- `E + 1` two-layer MLP experts of `width`-`hidden`-`width` on every token, their outputs
stacked as `[E + 1, tokens, width]`, with one batched product per layer. Its parameters are the
first weights `[E + 1, width, hidden]` and biases `[E + 1, hidden]`, then the second weights
`[E + 1, hidden, width]` and biases `[E + 1, width]`. -/
def mlpExperts {tokens width E : Nat} (hidden : Nat) (activation : Activation.Kind)
    (firstSeed secondSeed : Nat) : Layer [tokens, width] [E + 1, tokens, width] :=
  let shapes : List Shape :=
    [[E + 1, width, hidden], [E + 1, hidden], [E + 1, hidden, width], [E + 1, width]]
  let first := TorchLean.Init.Scheme.xavierUniform width hidden
  let second := TorchLean.Init.Scheme.xavierUniform hidden width
  let init (scheme : TorchLean.Init.Scheme) (seed : Nat) (s : Shape) :
      TorchLean.Tensor Float s :=
    TorchLean.Init.tensor (shape := s) (scheme := scheme) (seed := seed)
  let plan (scheme : TorchLean.Init.Scheme) (seed : Nat) :=
    Init.Source.scheme scheme seed
  let zero := TorchLean.Init.Scheme.zeros
  { kind := s!"MlpExperts({E + 1} of {width}-{hidden}-{width})"
    stateShapes := shapes
    initState := .cons (init first firstSeed _) (.cons (init zero 0 _)
      (.cons (init second secondSeed _) (.cons (init zero 0 _) .nil)))
    runtimeInit := some (.cons (plan first firstSeed) (.cons (plan zero 0)
      (.cons (plan second secondSeed) (.cons (plan zero 0) .nil))))
    requiresGrad := #[true, true, true, true]
    validateConfig := do
      if hidden = 0 then throw "MoeMlp: the experts need at least one hidden unit"
      if width = 0 then throw "MoeMlp: the token width must be positive"
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun firstWeight firstBias secondWeight secondBias x =>
          ((do
            -- Every expert on every token: `[E + 1, tokens, hidden]`, then
            -- `[E + 1, tokens, width]`.
            let firstProduct : Ref (m := m) (α := α) [E + 1, tokens, hidden] ←
              Ops.apply (m := m) (α := α)
                (.matmul (batchA := []) (batchB := [E + 1]) (batch := [E + 1]) (m := tokens)
                  (n := width) (p := hidden) false false (fun h => nomatch h) (fun h => nomatch h)
                  Shape.BroadcastTo.proof Shape.BroadcastTo.proof) (x, firstWeight, ())
            let firstBias3 ← Ops.apply (m := m) (α := α)
                (.reshape (s := [E + 1, hidden]) (t := [E + 1, 1, hidden])
                  (by simp)) (firstBias, ())
            let preActivation ← functional.addB (m := m) (α := α)
              (t := [E + 1, tokens, hidden]) firstProduct firstBias3
            let hiddenValues ← Ops.apply (m := m) (α := α)
              activation.op (preActivation, ())
            let secondProduct : Ref (m := m) (α := α) [E + 1, tokens, width] ←
              Ops.apply (m := m) (α := α)
                (.matmul (batchA := [E + 1]) (batchB := [E + 1]) (batch := [E + 1]) (m := tokens)
                  (n := hidden) (p := width) false false (fun h => nomatch h) (fun h => nomatch h)
                  Shape.BroadcastTo.proof Shape.BroadcastTo.proof)
                      (hiddenValues, secondWeight, ())
            let secondBias3 ← Ops.apply (m := m) (α := α)
                (.reshape (s := [E + 1, width]) (t := [E + 1, 1, width])
                  (by simp)) (secondBias, ())
            functional.addB (m := m) (α := α)
              (t := [E + 1, tokens, width]) secondProduct secondBias3) :
            m (Ops.Ref (m := m) (α := α) [E + 1, tokens, width])) }

/--
A mixture of `experts` two-layer MLP experts over the tokens of a `[tokens, width]` input, built
from `Layer.moe` with an `nn.linear` router and `mlpExperts`, keeping `k` experts per
token. The layer computes `MoE.forward` on every token, as `nn.moe` with the same router
and experts does, with all experts batched. `experts` and `k` must be positive.

Example:
```lean
-- Eight experts of 32-64-32, two kept per token, on sequences of 16 tokens.
def model : nn.Builder (nn.Sequential [16, 32] [16, 32]) :=
  nn.moeMLP (tokens := 16) (width := 32) (experts := 8) 2 64
```
-/
def moeMLP {tokens width : Nat} (experts k hidden : Nat)
    (activation : Activation.Kind := .relu) :
    Builder (Sequential [tokens, width] [tokens, width]) := do
  let router ← nn.linear width (experts - 1 + 1) (batchShape := [tokens])
  withSeed fun firstSeed => withSeed fun secondSeed =>
    let layer := Layer.moe k router (nn.Sequential.fromLayer
      (mlpExperts (tokens := tokens) (width := width) (E := experts - 1) hidden
        activation firstSeed secondSeed))
    pure <| nn.Sequential.fromLayer
      { layer with
        validateConfig := do
          if experts = 0 then throw "MoeMlp: a mixture needs at least one expert"
          layer.validateConfig }

end nn
end TorchLean

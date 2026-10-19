/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Neural.Blocks
public import NN.API.Neural.Impl
public import NN.API.Seeded

/-!
# Mixture of experts

`nn.moe k router experts` applies `Spec.MoE.forwardSpec` to every token of a `[tokens, width]`
input. The router scores the experts, the `k` experts with the largest softmax probabilities are
kept, ties going to the lower index, and the outputs of the kept experts are added in increasing
expert index, each weighted by its probability divided by the total probability of the kept
experts. Over `ℝ` this is Mixtral's sparse block (`NN.Proofs.Models.MoE`).

Every expert runs on every token and the experts that are not kept get gate zero, so all shapes
are static. When the router and the experts act on each token separately, as linear layers and
MLPs over the token axis do, a token's output depends only on that token, whatever else is in its
batch; `nn.moe` accepts any `[tokens, width]` networks, and one that mixes tokens, such as
attention, gives up that independence.

`nn.moeMlp` is the same layer for the common case where every expert is a two-layer MLP. It
keeps the experts' weights in single tensors and runs them all with one batched product per
layer, so a step issues a few dozen operations instead of several per expert, and is the one to
train with. `nn.moe` takes any experts.
-/

@[expose] public section

namespace TorchLean
namespace nn

open Runtime.Autograd.Torch

/-- The outputs of `E + 1` branches that read the same input, stacked along a new first axis in
branch order. -/
def stackBranches {σ τ : Spec.Shape} :
    (E : Nat) → (Fin (E + 1) → Sequential σ τ) → Sequential σ (τ.prependDim (E + 1))
  | 0, branches =>
      Runtime.Autograd.Model.Layers.Seq.comp (branches 0) (Impl.reshape τ (τ.prependDim 1))
  | E + 1, branches =>
      Nat.add_comm 1 (E + 1) ▸ concatBranches
        (Runtime.Autograd.Model.Layers.Seq.comp (branches 0) (Impl.reshape τ (τ.prependDim 1)))
        (stackBranches E fun i => branches i.succ)

/--
The gating and mixing of `nn.moe`: from the router's logits `[tokens, E + 1]` and the experts'
outputs stacked as `[E + 1, tokens, width]`, compute `Spec.MoE.combineSpec` of
`Spec.MoE.gatesVecSpec k` for every token. The total kept probability and the output are both
summed in increasing expert index, starting from zero.
-/
def Internal.moeMix {α : Type} [TorchLean.Storage α] [Context α] {m : Type → Type} [Monad m]
    [Ops (m := m) (α := α)] {tokens width E : Nat} (k : Nat)
    (logits : Ref (m := m) (α := α) [tokens, E + 1])
    (outputs : Ref (m := m) (α := α) [E + 1, tokens, width]) :
    m (Ref (m := m) (α := α) [tokens, width]) := do
  let probabilities ← softmaxLast (m := m) (α := α) logits
  let mask ← topKMask (m := m) (α := α) k probabilities
  let kept ← mul (m := m) (α := α) probabilities mask
  let mut total ← const (m := m) (α := α) (TorchLean.Tensor.zeros [tokens])
  for e in List.finRange (E + 1) do
    total ← add (m := m) (α := α) total
      (← select (m := m) (α := α) (s := [tokens, E + 1]) 1 kept e)
  let mut mixed ← const (m := m) (α := α) (TorchLean.Tensor.zeros [tokens, width])
  for e in List.finRange (E + 1) do
    let gate ← div (m := m) (α := α)
      (← select (m := m) (α := α) (s := [tokens, E + 1]) 1 kept e) total
    let gateB ← broadcastAfterSum (m := m) (α := α) [tokens, width] 1 gate
    let output ← select (m := m) (α := α) (s := [E + 1, tokens, width]) 0 outputs e
    mixed ← add (m := m) (α := α) mixed (← mul (m := m) (α := α) gateB output)
  pure mixed

/-- The `nn.moe` layer. Its parameters are the router's followed by each expert's, in order. -/
def Layer.moe {tokens width outWidth E : Nat} (k : Nat)
    (router : Sequential [tokens, width] [tokens, E + 1])
    (experts : Fin (E + 1) → Sequential [tokens, width] [tokens, outWidth]) :
    Layer [tokens, width] [tokens, outWidth] :=
  Layer.combineBranches s!"MoE({E + 1} experts, top {k})" router (stackBranches E experts)
    fun {α} _ _ {m} _ _ logits outputs => Internal.moeMix (m := m) (α := α) k logits outputs

/--
A mixture-of-experts layer over the tokens of a `[tokens, width]` input, as in Mixtral: `router`
scores the experts, the `k` with the largest softmax probabilities are kept (ties to the lower
index), and their outputs are added in increasing expert index, each weighted by its probability
divided by the kept total. `Spec.MoE.forwardSpec` is the layer on one token.

Example:
```lean
-- Eight experts, two kept per token, on sequences of 16 tokens of width 32.
def model : nn.Builder (nn.Sequential [16, 32] [16, 32]) := do
  let router ← nn.linear 32 8 (batchShape := [16])
  let experts ← (List.range 8).mapM fun _ =>
    nn.Sequential![nn.linear 32 64 (batchShape := [16]), nn.relu,
      nn.linear 64 32 (batchShape := [16])]
  pure (nn.moe 2 router fun e => experts.getD e.val (nn.Sequential.identity _))
```
-/
def moe {tokens width outWidth E : Nat} (k : Nat)
    (router : Sequential [tokens, width] [tokens, E + 1])
    (experts : Fin (E + 1) → Sequential [tokens, width] [tokens, outWidth]) :
    Sequential [tokens, width] [tokens, outWidth] :=
  nn.Sequential.fromLayer (Layer.moe k router experts)

/-- The experts' activation in `nn.moeMlp`. -/
inductive MoeMlp.Activation where
  | relu | gelu | silu
  deriving Repr, DecidableEq, Inhabited

/--
The gates of `nn.moe` for every token of `logits`: `Spec.MoE.gatesVecSpec k` on each row. The kept
probabilities of a row are summed by one product with a column of ones, a left fold from zero as
`sumSpec` is.
-/
def Internal.moeGates {α : Type} [TorchLean.Storage α] [Context α] {m : Type → Type} [Monad m]
    [Ops (m := m) (α := α)] {tokens E : Nat} (k : Nat)
    (logits : Ref (m := m) (α := α) [tokens, E + 1]) :
    m (Ref (m := m) (α := α) [tokens, E + 1]) := do
  let probabilities ← Runtime.Autograd.Torch.softmaxLast (m := m) (α := α) logits
  let mask ← Runtime.Autograd.Torch.topKMask (m := m) (α := α) k probabilities
  let kept ← Runtime.Autograd.Torch.mul (m := m) (α := α) probabilities mask
  let ones ← Runtime.Autograd.Torch.const (m := m) (α := α)
    (TorchLean.Tensor.full [E + 1, 1] (1 : α))
  let total ← Runtime.Autograd.Torch.matmul (m := m) (α := α) (batchA := []) (batchB := [])
    (batch := []) (mDim := tokens) (nDim := E + 1) (pDim := 1) kept ones
  let totalB ← Runtime.Autograd.Torch.broadcastTo (m := m) (α := α) (s₁ := [tokens, 1])
    (s₂ := [tokens, E + 1]) (Spec.Shape.BroadcastTo.proof) total
  Runtime.Autograd.Torch.div (m := m) (α := α) kept totalB

/-- The `nn.moeMlp` layer with `E + 1` experts. Its parameters are the router's weight
`[E + 1, width]` and bias `[E + 1]`, the experts' first weights `[E + 1, width, hidden]` and biases
`[E + 1, hidden]`, and their second weights `[E + 1, hidden, width]` and biases `[E + 1, width]`. -/
def Layer.moeMlp {tokens width E : Nat} (k hidden : Nat) (activation : MoeMlp.Activation)
    (routerSeed firstSeed secondSeed : Nat) : Layer [tokens, width] [tokens, width] :=
  let shapes : List Spec.Shape :=
    [[E + 1, width], [E + 1], [E + 1, width, hidden], [E + 1, hidden],
      [E + 1, hidden, width], [E + 1, width]]
  let router := Runtime.Autograd.Torch.Init.Scheme.xavierUniform width (E + 1)
  let first := Runtime.Autograd.Torch.Init.Scheme.xavierUniform width hidden
  let second := Runtime.Autograd.Torch.Init.Scheme.xavierUniform hidden width
  let init (scheme : Runtime.Autograd.Torch.Init.Scheme) (seed : Nat) (s : Spec.Shape) :
      TorchLean.Tensor Float s :=
    Runtime.Autograd.Torch.Init.tensor (s := s) (sch := scheme) (seed := seed)
  let plan (scheme : Runtime.Autograd.Torch.Init.Scheme) (seed : Nat) :=
    Runtime.Autograd.Model.Module.RuntimeInit.FloatInit.ofScheme scheme seed
  let zero := Runtime.Autograd.Torch.Init.Scheme.zeros
  { kind := s!"MoeMlp({E + 1} experts of {width}-{hidden}-{width}, top {k})"
    stateShapes := shapes
    initState := .cons (init router routerSeed _) (.cons (init zero 0 _)
      (.cons (init first firstSeed _) (.cons (init zero 0 _)
        (.cons (init second secondSeed _) (.cons (init zero 0 _) .nil)))))
    runtimeInit := some (.cons (plan router routerSeed) (.cons (plan zero 0)
      (.cons (plan first firstSeed) (.cons (plan zero 0)
        (.cons (plan second secondSeed) (.cons (plan zero 0) .nil))))))
    requiresGrad := #[true, true, true, true, true, true]
    validateConfig := do
      if hidden = 0 then throw "MoeMlp: the experts need at least one hidden unit"
      if width = 0 then throw "MoeMlp: the token width must be positive"
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun routerWeight routerBias firstWeight firstBias secondWeight secondBias x =>
          ((do
            -- Router logits `[tokens, E + 1]`.
            let routerT ← Runtime.Autograd.Torch.swapAdjacentAtDepth (m := m) (α := α)
              (s := [E + 1, width]) 0 routerWeight
            let scores ← Runtime.Autograd.Torch.matmul (m := m) (α := α) (batchA := [])
              (batchB := []) (batch := []) (mDim := tokens) (nDim := width) (pDim := E + 1)
              x routerT
            let logits ← Runtime.Autograd.Model.F.addB (m := m) (α := α) (t := [tokens, E + 1])
              scores routerBias
            -- Every expert on every token: `[E + 1, tokens, hidden]`, then
            -- `[E + 1, tokens, width]`.
            let firstProduct ← Runtime.Autograd.Torch.matmul (m := m) (α := α) (batchA := [])
              (batchB := [E + 1]) (batch := [E + 1]) (mDim := tokens) (nDim := width)
              (pDim := hidden) x firstWeight
            let firstBias3 ← Runtime.Autograd.Torch.reshape (m := m) (α := α)
              (s₁ := [E + 1, hidden]) (s₂ := [E + 1, 1, hidden]) firstBias
              (by simp [Spec.Shape.size])
            let preActivation ← Runtime.Autograd.Model.F.addB (m := m) (α := α)
              (t := [E + 1, tokens, hidden]) firstProduct firstBias3
            let hiddenValues ← match activation with
              | .relu => Runtime.Autograd.Torch.relu (m := m) (α := α) preActivation
              | .gelu => Runtime.Autograd.Torch.gelu (m := m) (α := α) preActivation
              | .silu => Runtime.Autograd.Torch.silu (m := m) (α := α) preActivation
            let secondProduct ← Runtime.Autograd.Torch.matmul (m := m) (α := α)
              (batchA := [E + 1]) (batchB := [E + 1]) (batch := [E + 1]) (mDim := tokens)
              (nDim := hidden) (pDim := width) hiddenValues secondWeight
            let secondBias3 ← Runtime.Autograd.Torch.reshape (m := m) (α := α)
              (s₁ := [E + 1, width]) (s₂ := [E + 1, 1, width]) secondBias
              (by simp [Spec.Shape.size])
            let outputs ← Runtime.Autograd.Model.F.addB (m := m) (α := α)
              (t := [E + 1, tokens, width]) secondProduct secondBias3
            -- Mix: `Σₑ gates[t, e] * outputs[e, t]`, one batched product over the tokens.
            let gates ← Internal.moeGates (m := m) (α := α) k logits
            let byToken ← Runtime.Autograd.Torch.swapAdjacentAtDepth (m := m) (α := α)
              (s := [E + 1, tokens, width]) 0 outputs
            let gateRows ← Runtime.Autograd.Torch.reshape (m := m) (α := α)
              (s₁ := [tokens, E + 1]) (s₂ := [tokens, 1, E + 1]) gates (by simp [Spec.Shape.size])
            let mixed ← Runtime.Autograd.Torch.matmul (m := m) (α := α) (batchA := [tokens])
              (batchB := [tokens]) (batch := [tokens]) (mDim := 1) (nDim := E + 1)
              (pDim := width) gateRows byToken
            Runtime.Autograd.Torch.reshape (m := m) (α := α) (s₁ := [tokens, 1, width])
              (s₂ := [tokens, width]) mixed (by simp [Spec.Shape.size])) :
            m (TorchLean.Runtime.ValueRef (m := m) (α := α) [tokens, width])) }

/--
A mixture of `experts` two-layer MLP experts over the tokens of a `[tokens, width]` input, with a
linear router; `k` experts are kept per token. The layer computes `Spec.MoE.forwardSpec` on every
token, as `nn.moe` with the same router and experts does, with all experts batched. `experts` must
be positive.

Example:
```lean
-- Eight experts of 32-64-32, two kept per token, on sequences of 16 tokens.
def model : nn.Builder (nn.Sequential [16, 32] [16, 32]) :=
  nn.moeMlp (tokens := 16) (width := 32) (experts := 8) 2 64
```
-/
def moeMlp {tokens width : Nat} (experts k hidden : Nat)
    (activation : MoeMlp.Activation := .relu) :
    Builder (Sequential [tokens, width] [tokens, width]) :=
  withSeed fun routerSeed => withSeed fun firstSeed => withSeed fun secondSeed =>
    let layer := Layer.moeMlp (tokens := tokens) (width := width) (E := experts - 1) k hidden
      activation routerSeed firstSeed secondSeed
    pure <| nn.Sequential.fromLayer
      { layer with
        validateConfig := do
          if experts = 0 then throw "MoeMlp: a mixture needs at least one expert"
          layer.validateConfig }

end nn
end TorchLean

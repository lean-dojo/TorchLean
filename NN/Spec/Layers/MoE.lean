/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Activation

/-!
# Mixture of experts

A mixture-of-experts layer scores every expert for each token, keeps the `k` best, and adds the
outputs of the kept experts weighted by gates. Implementations leave three choices open that change
the bits of the result, and this file makes each of them definite.

* Ties. Expert `j` *beats* expert `i` when its score is larger, or when neither score is larger
  and `j` has the smaller index. Expert `i` is kept when fewer than `k` experts beat it (`kept`),
  so ties go to the lower index, as in `argmax`. With totally ordered scores (no NaN) this keeps
  `min k E` experts. With NaN among the scores `beats` can cycle: for `[0, NaN, 1]` every expert
  is beaten once, so `k = 1` keeps none and `k = 2` keeps all three. The test needs only the
  comparison `>` and a count, so a device can decide it exactly for every entry at once.
  `torch.topk` documents that the indices of tied elements may vary across invocations.
* Gates. As in Mixtral, the router's logits go through a softmax, the `k` largest probabilities
  are kept, and the kept ones are divided by their sum (`gatesVecSpec`). Over `ℝ` the experts kept
  are those with the largest logits and the gates are the softmax of their logits
  (`NN.Proofs.Models.MoE`). With floats, rounding can make two different logits equally probable,
  so there the probabilities decide.
* Order. The gated outputs are added in increasing expert index, starting from zero
  (`combineSpec`), the order in which Mixtral's sparse block loops over its experts. Every expert
  runs on every token and an expert that is not kept has gate zero, so over `ℝ` this dense form
  equals the sparse form that runs only the kept experts (`sparseForwardSpec`).

`forwardSpec` is the layer on one token. A layer over many tokens applies it to each, so a token's
output does not depend on the other tokens in its batch, and a batch can be split anywhere.
Capacity limits and batch-wide balancing losses break that independence;
`NN.Proofs.Models.MoE` shows what they cost.
-/

@[expose] public section

namespace Spec
namespace MoE

open TorchLean TorchLean.Tensor

variable {α : Type} [Context α]

/-! ## Routing -/

/-- Whether expert `j` beats expert `i` on `scores`: its score is larger, or neither score is
larger and `j` comes first. -/
def beats {E : Nat} (scores : Fin E → α) (j i : Fin E) : Bool :=
  Context.gtBool (scores j) (scores i) || (decide (j < i) && !Context.gtBool (scores i) (scores j))

/-- The number of experts that beat expert `i`. -/
def rank {E : Nat} (scores : Fin E → α) (i : Fin E) : Nat :=
  (Finset.univ.filter fun j => beats scores j i = true).card

/-- Whether expert `i` is one of the `k` experts kept: fewer than `k` experts beat it. -/
def kept {E : Nat} (k : Nat) (scores : Fin E → α) (i : Fin E) : Bool :=
  decide (rank scores i < k)

/-- The experts kept for a token, in increasing index. -/
def keptExperts {E : Nat} (k : Nat) (scores : Fin E → α) : List (Fin E) :=
  (List.finRange E).filter (kept k scores)

variable [TorchLean.Storage α]

/-- One for the entries of a vector of scores that `kept k` keeps and zero for the others. -/
def topKMaskVecSpec {E : Nat} (k : Nat) (scores : Tensor α [E]) : Tensor α [E] :=
  Tensor.dim fun i =>
    Tensor.scalar (if kept k (fun j => scores.getScalar j) i then 1 else 0)

/-- `topKMaskVecSpec` along the last axis of a tensor of any shape. A scalar is a row of one
entry, kept when `0 < k`. -/
def topKMaskSpec (k : Nat) : {s : Shape} → Tensor α s → Tensor α s
  | .scalar, _ => Tensor.scalar (if 0 < k then 1 else 0)
  | .dim _ .scalar, scores => topKMaskVecSpec k scores
  | .dim _ inner, scores =>
      Tensor.dim fun i => topKMaskSpec k (s := inner) (Tensor.unstack scores i)

/-! ## The layer -/

/-- Mixtral's gates for one token: the softmax of the router's logits, kept for the experts
`topKMaskVecSpec` keeps (the `k` largest probabilities when none is NaN) and divided by the sum of
the kept probabilities, and zero for the other experts. With `k = 0` nothing is kept and the
division is `0 / 0`. -/
def gatesVecSpec {E : Nat} (k : Nat) (logits : Tensor α [E]) : Tensor α [E] :=
  let probabilities := Activation.softmaxVecSpec logits
  let keptProbabilities := mulSpec probabilities (topKMaskVecSpec k probabilities)
  divSpec keptProbabilities (replicate (Tensor.scalar (sumSpec keptProbabilities)))

/-- The gated sum of the experts' outputs for one token, `Σₑ gatesₑ * outputsₑ`, added in
increasing expert index starting from zero. -/
def combineSpec {E : Nat} {s : Shape} (gates : Tensor α [E]) (outputs : Fin E → Tensor α s) :
    Tensor α s :=
  (List.finRange E).foldl
    (fun acc e => addSpec acc (mulSpec (replicate (Tensor.scalar (gates.getScalar e))) (outputs e)))
    (replicate (Tensor.scalar 0))

/-- A mixture-of-experts layer on one token `x`: the router's logits give the gates
(`gatesVecSpec`), every expert runs on `x`, and their outputs are combined (`combineSpec`). -/
def forwardSpec {E : Nat} {σ s : Shape} (k : Nat) (router : Tensor α σ → Tensor α [E])
    (experts : Fin E → Tensor α σ → Tensor α s) (x : Tensor α σ) : Tensor α s :=
  combineSpec (gatesVecSpec k (router x)) fun e => experts e x

/-- Mixtral's sparse block on one token: only the kept experts run; each output is multiplied by
its probability divided by the sum of the kept probabilities, and the products are added into zero
in increasing expert index, as the block's loop of `index_add_` over experts does. The sum of the
kept probabilities is taken in increasing index; Mixtral sums them in decreasing order, which gives
the same bits for `k ≤ 2`. -/
def sparseForwardSpec {E : Nat} {σ s : Shape} (k : Nat) (router : Tensor α σ → Tensor α [E])
    (experts : Fin E → Tensor α σ → Tensor α s) (x : Tensor α σ) : Tensor α s :=
  let probabilities := Activation.softmaxVecSpec (router x)
  let p := fun e => probabilities.getScalar e
  let chosen := keptExperts k p
  let total := chosen.foldl (fun acc e => acc + p e) 0
  chosen.foldl
    (fun acc e => addSpec acc (mulSpec (experts e x) (replicate (Tensor.scalar (p e / total)))))
    (replicate (Tensor.scalar 0))

end MoE
end Spec

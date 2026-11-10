/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Layers.Activation
public import TorchLean.Tensor.Layers.Softmax

/-!
# Mixture of experts

A mixture-of-experts layer scores every expert for each token, keeps the `k` best, and adds the
outputs of the kept experts weighted by gates. Implementations leave three choices open that change
the bits of the result, and the definitions below fix each of them.

* Ties. Expert `j` *beats* expert `i` when its score is larger, or when neither score is larger
  and `j` has the smaller index. Expert `i` is kept when fewer than `k` experts beat it (`kept`),
  so ties go to the lower index, as in `argmax`. With totally ordered scores (no NaN) this keeps
  `min k E` experts. With NaN among the scores `beats` can cycle: for `[0, NaN, 1]` every expert
  is beaten once, so `k = 1` keeps none and `k = 2` keeps all three. The test needs only the
  comparison `>` and a count, so a device can decide it exactly for every entry at once.
  `torch.topk` documents that the indices of tied elements may vary across invocations.
* Gates. As in Mixtral, the router's logits go through a softmax, the `k` largest probabilities
  are kept, and the kept ones are divided by their sum (`gates`). Over `ℝ` the experts kept
  are those with the largest logits and the gates are the softmax of their logits
  (`TorchLean.MoE`). With floats, rounding can make two different logits equally probable,
  so there the probabilities decide.
* Order. The gated outputs are added in increasing expert index, starting from zero
  (`combine`), the order in which Mixtral's sparse block loops over its experts. Every expert
  runs on every token and an expert that is not kept has gate zero, so over `ℝ` this dense form
  equals the sparse form that runs only the kept experts (`sparseForward`).

`forward` is the layer on one token. A layer over many tokens applies it to each, so a token's
output does not depend on the other tokens in its batch, and a batch can be split anywhere. The
layer has no capacity limit or batch-wide balancing loss, either of which would break that
independence.

Two papers shaped the layer, and we transcribed both by hand. Shazeer et al., "Outrageously Large
Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (2017, arXiv:1701.06538), gives the
gated sum, eq. (1) of Sec. 2, and the top-`k` softmax gate of eqs. (3) and (5) in Sec. 2.1, whose
noise term (eq. (4)) we leave out. Jiang et al., "Mixtral of Experts" (2024, arXiv:2401.04088),
Sec. 2.1, writes the gate as `Softmax(TopK(x · W_g))`, which agrees with `gates` over `ℝ`. The
steps of `gates` and `sparseForward` are those of the sparse block in Hugging Face Transformers'
`src/transformers/models/mixtral/modeling_mixtral.py`. Over `ℝ`,
`TorchLean.MoE.getScalar_gates_softmax` proves the gate statement above and
`TorchLean.MoE.forward_eq_sparseForward` the dense-equals-sparse one (`TorchLean/NN/MoE.lean`).
`TorchLeanTest/IR/MoEImport.lean` imports a top-2 block written in PyTorch. In binary32 the
imported graph comes within `8 · m / 2^24` of torch's output, for the largest output magnitude
`m`, and in `Float` it computes `forward` bit for bit.
-/

@[expose] public section

namespace TorchLean
namespace MoE

open TorchLean.Tensor

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

variable [Storage α]

/-- One for the entries of a vector of scores that `kept k` keeps and zero for the others. -/
def topKMaskVec {E : Nat} (k : Nat) (scores : Tensor α [E]) : Tensor α [E] :=
  Tensor.dim fun i =>
    Tensor.scalar (if kept k (fun j => scores.getScalar j) i then 1 else 0)

/-- `topKMaskVec` along the last axis of a tensor of any shape. A scalar is a row of one
entry, kept when `0 < k`. -/
def topKMaskInnermost (k : Nat) : {s : Shape} → Tensor α s → Tensor α s :=
  Tensor.lastAxisMap (if 0 < k then 1 else 0) fun _ => topKMaskVec k

/-! ## The layer -/

/-- Mixtral's gates for one token: the softmax of the router's logits, kept for the experts
`topKMaskVec` keeps (the `k` largest probabilities when none is NaN) and divided by the sum of
the kept probabilities, and zero for the other experts. With `k = 0` nothing is kept and the
division is `0 / 0`. -/
def gates {E : Nat} (k : Nat) (logits : Tensor α [E]) : Tensor α [E] :=
  let probabilities := Tensor.softmax logits ⟨0, Nat.one_pos⟩
  let keptProbabilities := probabilities * topKMaskVec k probabilities
  keptProbabilities / Tensor.full _ (Tensor.sum keptProbabilities)

/-- The gated sum of the experts' outputs for one token, `Σₑ gatesₑ * outputsₑ`, added in
increasing expert index starting from zero. -/
def combine {E : Nat} {s : Shape} (gates : Tensor α [E]) (outputs : Fin E → Tensor α s) :
    Tensor α s :=
  (List.finRange E).foldl
    (fun acc e => acc + (Tensor.full _ (gates.getScalar e) * outputs e))
    (Tensor.full _ 0)

/-- A mixture-of-experts layer on one token `x`: the router's logits give the gates
(`gates`), every expert runs on `x`, and their outputs are combined (`combine`). -/
def forward {E : Nat} {σ s : Shape} (k : Nat) (router : Tensor α σ → Tensor α [E])
    (experts : Fin E → Tensor α σ → Tensor α s) (x : Tensor α σ) : Tensor α s :=
  combine (gates k (router x)) fun e => experts e x

/-- Mixtral's sparse block on one token. Only the kept experts run, and each output is multiplied
by its probability divided by the sum of the kept probabilities. The products are added into zero
in increasing expert index, as the block's loop of `index_add_` over experts does. The sum of the
kept probabilities is taken in increasing index. Mixtral sums them in decreasing order, and the two
orders give the same bits for `k ≤ 2`. -/
def sparseForward {E : Nat} {σ s : Shape} (k : Nat) (router : Tensor α σ → Tensor α [E])
    (experts : Fin E → Tensor α σ → Tensor α s) (x : Tensor α σ) : Tensor α s :=
  let probabilities := Tensor.softmax (router x) ⟨0, Nat.one_pos⟩
  let p := fun e => probabilities.getScalar e
  let chosen := keptExperts k p
  let total := chosen.foldl (fun acc e => acc + p e) 0
  chosen.foldl
    (fun acc e => acc + (experts e x * Tensor.full _ (p e / total)))
    (Tensor.full _ 0)

end MoE
end TorchLean

/-! ## The top-k mask along any axis -/

namespace TorchLean.MoE

open Shape TorchLean.Tensor

variable {α : Type} [Context α] [Storage α] {s : Shape}

/-- The top-`k` mask of `x` along `dim`: `topKMaskVec k` on every fiber along `dim`, computed by
moving `dim` last, applying the last-axis mask and moving it back. -/
def topKMask (k : Nat) (x : Tensor α s) (dim : Fin s.rank) : Tensor α s :=
  Tensor.alongAxis dim (fun x => topKMaskInnermost k x) x

/-- An entry of `topKMask k x dim` is the top-`k` mask of the fiber through it, at its index along
`dim`. -/
theorem topKMask_apply (k : Nat) (x : Tensor α s) (dim : Fin s.rank) (c : s.Coord) :
    topKMask k x dim c = (topKMaskVec k (x.fiber dim c)).getScalar
      (Coord.axisEquiv dim (shape := s) c).1 :=
  Tensor.alongAxis_apply (fun x => topKMaskInnermost k x) (fun _ => rfl) dim x c

/-- Along the last axis, `topKMask` is `topKMaskInnermost`. -/
theorem topKMask_last (k : Nat) (x : Tensor α s) (h : s.rank - 1 < s.rank) :
    topKMask k x ⟨s.rank - 1, h⟩ = topKMaskInnermost k x :=
  Tensor.alongAxis_of_last _ (fun x => topKMaskInnermost k x) x
    (show s.rank - 1 + 1 = s.rank by omega)

end TorchLean.MoE

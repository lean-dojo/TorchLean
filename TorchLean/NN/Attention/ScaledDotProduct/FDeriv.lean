/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Attention.ScaledDotProduct
public import TorchLean.NN.Tape.RealGraph.EvalVec
public import TorchLean.NN.Norm.MatrixEntries
public import TorchLean.NN.Attention.HardMask

/-!
# Scaled dot-product attention: tape graph versus specification

`TorchLean.NN.Attention.ScaledDotProduct` builds a tape graph from proven
nodes, and on that graph `Tape.Graph.backpropVec_eq_fderiv` gives
`backprop = (fderiv eval)†`. `TorchLean.Tensor.Layers.Attention` defines
`TorchLean.Attention.scaledDotProduct` on tensors. The file connects the two through these theorems:

- `get_idxOut_evalVec`: for every scale `c`, the output block of the graph evaluation is the
  vectorized specification forward pass `softmax(c * Q Kᵀ) V`.
- `Attention.hasFDerivAt_forwardVec`: the specification forward pass is Fréchet-differentiable, with
  derivative read off the graph.
- `Attention.backprop_eq_fderiv`: at the specification's scale
  `c = 1 / Attention.scaleDenom d`, which is `1/√d` for `d ≠ 0`, the tape reverse pass seeded
  on the output block is the vector-Jacobian product of `TorchLean.Attention.scaledDotProduct`.
- `Attention.backprop_eq_fderiv_allTrueMask`: the same statement for
  the masked code path with the all-true Boolean mask.

Blocks of the evaluated graph are read with the `Tape.GraphData.evalVec` lemmas of
`TorchLean.NN.Tape.RealGraph.EvalVec`, and matrix coordinates through
`Matmul.toVec_idxMN`. Attention is proved in the flat `Vec` world where Mathlib's
calculus lives, and stated about `Tensor ℝ [m, n]`. The first half of the file matches the flat
transpose `MatTranspose.vec` with `permute`. No lemma is needed for `matmul`, whose node computes
`Tensor.matmul` by definition, or for the scaling and softmax nodes, operator arms whose forward
maps are `Tensor.scale` and `Tensor.softmax`. The second half walks the attention graph node by
node.
-/

@[expose] public section

open TorchLean.Shape (size)

namespace TorchLean

open TorchLean.Tensor
open Tape.RealNode
open Matmul

noncomputable section

/-! ## Flattened matrices -/

/-- Flat transposition swaps the row and column index: entry `(i, j)` of `MatTranspose.vec a` is
entry `(j, i)` of `a`. -/
theorem Tape.RealNode.MatTranspose.vec_idxMN {m n : Nat} (a : Vec (Shape.size [m, n])) (i : Fin n)
    (j : Fin m) :
    MatTranspose.vec (m := m) (n := n) a (idxMN (m := n) (n := m) i j) =
      a (idxMN (m := m) (n := n) j i) := by
  have hidx : Fin.cast (MatTranspose.matSize_eq_mul m n).symm
      ((MatTranspose.equiv m n).symm
        (Fin.cast (MatTranspose.matSize_eq_mul n m) (idxMN (m := n) (n := m) i j))) =
      idxMN (m := m) (n := n) j i := by
    apply Fin.ext
    change (idxMN (m := n) (n := m) i j).val / m + n * ((idxMN (m := n) (n := m) i j).val % m) =
      (idxMN (m := m) (n := n) j i).val
    rw [val_idxMN, val_idxMN, Nat.add_mul_div_left _ _ j.pos, Nat.add_mul_mod_self_left,
      Nat.div_eq_of_lt j.isLt, Nat.mod_eq_of_lt j.isLt, Nat.zero_add]
  simp only [MatTranspose.vec, LinearIsometryEquiv.piLpCongrLeft_apply,
    Equiv.piCongrLeft'_apply, finCongr_symm_apply]
  rw [hidx]

/-- Swapping the two axes of a matrix tensor corresponds to the flat transpose.

Attention transposes the key matrix, so without this the `Kᵀ` in `Q Kᵀ` would have to be reasoned
about at the tensor level and at the vector level separately. -/
theorem Tensor.toVec_permute_adjacentSwap {m n : Nat} (A : Tensor ℝ [m, n]) :
    toVec (t := Tensor.permute A (Shape.adjacentSwap [m, n] 0) [n, m]) =
      MatTranspose.vec (m := m) (n := n) (toVec (t := A)) := by
  apply Normalization.vec_ext_idxMN (m := n) (n := m)
  intro i j
  rw [Matmul.toVec_idxMN, get2_transpose, MatTranspose.vec_idxMN,
    Matmul.toVec_idxMN]

/-! ## Attention graph evaluation -/

namespace Attention

open TensorPack (getIdx)

/-- Index of the attention output block in the saved context. -/
def idxOut (m d : Nat) :
    Idx (qkvCtx m d ++ ssScaledDotProduct m d) (.dim m (.dim d .scalar)) :=
  Idx.last (Γ := qkvCtx m d)
    (ss := [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar),
      .dim m (.dim m .scalar)])
    (τ := .dim m (.dim d .scalar))

/-- Query tensor `Q` stored in a vectorized attention context. -/
abbrev ctxQ {m d : Nat} (xV : CtxVec (qkvCtx m d)) : Tensor ℝ [m, d] :=
  ofVec (s := [m, d]) (CtxVec.get (idxQ (m := m) (d := d) (ss := [])) xV)

/-- Key tensor `K` stored in a vectorized attention context. -/
abbrev ctxK {m d : Nat} (xV : CtxVec (qkvCtx m d)) : Tensor ℝ [m, d] :=
  ofVec (s := [m, d]) (CtxVec.get (idxK (m := m) (d := d) (ss := [])) xV)

/-- Value tensor `V` stored in a vectorized attention context. -/
abbrev ctxV {m d : Nat} (xV : CtxVec (qkvCtx m d)) : Tensor ℝ [m, d] :=
  ofVec (s := [m, d]) (CtxVec.get (idxV (m := m) (d := d) (ss := [])) xV)

/-- Tensor-level attention forward pass `softmax(c * Q Kᵀ) V` for an arbitrary scale factor `c`. -/
def scaledDotProductWith {m d : Nat} (c : ℝ) (Q K V : Tensor ℝ [m, d]) : Tensor ℝ [m, d] :=
  Tensor.matmul
    (Tensor.softmax (s := [m, m]) (Tensor.scale (Tensor.matmul Q (Tensor.permute K
        (Shape.adjacentSwap [m, d] 0) [d, m])) c)
      ⟨1, Nat.one_lt_two⟩)
    V

/-- The output block of the attention graph is the vectorized tensor forward pass. -/
theorem get_idxOut_evalVec {m d : Nat} (c : ℝ) (xV : CtxVec (qkvCtx m d)) :
    CtxVec.get (idxOut m d) (Tape.GraphData.evalVec (scaledDotProductGraph c).toData () xV) =
      toVec (scaledDotProductWith c (ctxQ xV) (ctxK xV) (ctxV xV)) := by
  have hOut := Tape.GraphData.get_evalVec_snoc_last (graphProbs m d c).toData
    (nodeOut m d).toNodeData () xV (idxOut m d) rfl
  have hP := Tape.GraphData.get_evalVec_snoc_last (graphScaled m d c).toData
    (nodeProbs m d).toNodeData () xV (Idx.last (Γ := qkvCtx m d)
      (ss := [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)])
      (τ := .dim m (.dim m .scalar))) rfl
  have hS := Tape.GraphData.get_evalVec_snoc_last (graphLogits m d).toData
    (nodeScaled m d c).toNodeData () xV
    (Idx.last (Γ := qkvCtx m d) (ss := [.dim d (.dim m .scalar), .dim m (.dim m .scalar)])
      (τ := .dim m (.dim m .scalar))) rfl
  have hL := Tape.GraphData.get_evalVec_snoc_last (graphKt m d).toData
    (nodeLogits m d).toNodeData () xV
    (Idx.last (Γ := qkvCtx m d) (ss := [.dim d (.dim m .scalar)]) (τ := .dim m (.dim m .scalar)))
    rfl
  have hKt := Tape.GraphData.get_evalVec_snoc_last .nil (nodeKt m d).toNodeData () xV
    (Idx.last (Γ := qkvCtx m d) (ss := []) (τ := .dim d (.dim m .scalar))) rfl
  have hV : CtxVec.get (idxV (m := m) (d := d)
      (ss := [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar),
        .dim m (.dim m .scalar)])) (Tape.GraphData.evalVec (graphProbs m d c).toData () xV) =
      CtxVec.get (idxV (m := m) (d := d) (ss := [])) xV :=
    Tape.GraphData.get_evalVec_input _ () xV _ _ rfl
  have hQ : CtxVec.get (idxQ (m := m) (d := d) (ss := [.dim d (.dim m .scalar)]))
      (Tape.GraphData.evalVec (graphKt m d).toData () xV) =
        CtxVec.get (idxQ (m := m) (d := d) (ss := [])) xV :=
    Tape.GraphData.get_evalVec_input _ () xV _ _ rfl
  have hK : CtxVec.get (idxK (m := m) (d := d) (ss := []))
      (Tape.GraphData.evalVec .nil () xV) =
      CtxVec.get (idxK (m := m) (d := d) (ss := [])) xV :=
    Tape.GraphData.get_evalVec_input _ () xV _ _ rfl
  have hKmat : MatTranspose.vec (m := m) (n := d)
      (CtxVec.get (idxK (m := m) (d := d) (ss := [])) xV) =
      toVec (Tensor.permute (ctxK xV) (Shape.adjacentSwap [m, d] 0) [d, m]) := by
    rw [toVec_permute_adjacentSwap, toVec_ofVec]
  have hProbsF : ∀ yV, (nodeProbs m d).toNodeData.forwardVec () yV =
      toVec (Tensor.softmax (getIdx (TensorPack.ofVec yV) (Idx.last (Γ := qkvCtx m d)
        (ss := [.dim d (.dim m .scalar), .dim m (.dim m .scalar)])
        (τ := .dim m (.dim m .scalar)))) ⟨1, Nat.one_lt_two⟩) := fun _ => rfl
  have hScaledF : ∀ yV, (nodeScaled m d c).toNodeData.forwardVec () yV =
      toVec (Tensor.scale (getIdx (TensorPack.ofVec yV) (Idx.last (Γ := qkvCtx m d)
        (ss := [.dim d (.dim m .scalar)]) (τ := .dim m (.dim m .scalar))))
        (Tensor.scalar c).item) := fun _ => rfl
  rw [scaledDotProductGraph, hOut]
  simp -proj only [nodeOut, Tape.RealNode.matmul, Tape.LawfulNode.forwardVec_ofFn]
  rw [hP, hV, hProbsF, CtxVec.getIdx_unflattenCtx, hS, hScaledF, CtxVec.getIdx_unflattenCtx,
    hL]
  simp -proj only [nodeLogits, Tape.RealNode.matmul, Tape.LawfulNode.forwardVec_ofFn]
  rw [hQ, hKt]
  simp -proj only [nodeKt, Tape.RealNode.matrixTranspose, Tape.LawfulNode.forwardVec_ofFn]
  rw [hK, hKmat]
  simp -proj only [ofVec_toVec, Tensor.item_scalar]
  rfl

/-- The specification forward pass as a function on vectorized contexts. -/
def forwardVec {m d : Nat} (c : ℝ) (xV : CtxVec (qkvCtx m d)) :
    Vec (Shape.size (Shape.dim m (Shape.dim d Shape.scalar))) :=
  toVec (scaledDotProductWith c (ctxQ xV) (ctxK xV) (ctxV xV))

/-- The specification forward pass is the output projection of the graph evaluation. -/
theorem forwardVec_eq_getCLM_evalVec {m d : Nat} (c : ℝ) :
    forwardVec (m := m) (d := d) c =
      fun xV => CtxVec.getCLM (idxOut m d)
        (Tape.GraphData.evalVec (scaledDotProductGraph c).toData () xV) := by
  funext xV
  rw [CtxVec.getCLM_apply, get_idxOut_evalVec]
  rfl

/-- The specification attention forward pass is Fréchet-differentiable, and its derivative is the
output projection of the graph derivative. -/
theorem hasFDerivAt_forwardVec {m d : Nat} (c : ℝ) (xV : CtxVec (qkvCtx m d)) :
    HasFDerivAt (forwardVec (m := m) (d := d) c)
      ((CtxVec.getCLM (idxOut m d)).comp
        (fderiv ℝ
          (Tape.GraphData.evalVec (scaledDotProductGraph (m := m) (d := d) c).toData
            ()) xV)) xV := by
  obtain ⟨D, hD, -⟩ := Tape.GraphData.hasFDerivAt_evalVec_and_jvp
    (scaledDotProductGraph c).toData () xV ((scaledDotProductDGraph c).hg.at xV)
  rw [forwardVec_eq_getCLM_evalVec, hD.fderiv]
  exact (CtxVec.getCLM (idxOut m d)).hasFDerivAt.comp xV hD

/-- Tape backprop seeded on the output block is the VJP of the specification forward pass. -/
theorem backprop_eq_fderiv_forwardVec {m d : Nat} (c : ℝ) (xV : CtxVec (qkvCtx m d))
    (δ : Vec (Shape.size (Shape.dim m (Shape.dim d Shape.scalar)))) :
    Tape.GraphData.backpropVec (scaledDotProductGraph c).toData () xV
      (CtxVec.single (idxOut m d) δ) =
      (fderiv ℝ (forwardVec (m := m) (d := d) c) xV).adjoint δ := by
  rw [(hasFDerivAt_forwardVec c xV).fderiv, ContinuousLinearMap.adjoint_comp,
    ContinuousLinearMap.comp_apply, CtxVec.adjoint_getCLM]
  exact Tape.Graph.backpropVec_eq_fderiv _ () xV _ ((scaledDotProductDGraph c).hg.at xV)

/-- Tape backprop is the vector-Jacobian product of unmasked `scaledDotProduct`. -/
theorem backprop_eq_fderiv {m d : Nat} (hm : m ≠ 0)
    (xV : CtxVec (qkvCtx m d))
    (δ : Vec (Shape.size (Shape.dim m (Shape.dim d Shape.scalar)))) :
    Tape.GraphData.backpropVec
        (scaledDotProductGraph (1 / scaleDenom (α := ℝ) d)).toData () xV
        (CtxVec.single (idxOut m d) δ) =
      (fderiv ℝ (fun xV : CtxVec (qkvCtx m d) =>
        toVec (scaledDotProduct (h1 := hm) (h2 := hm)
          { query := ctxQ xV, key := ctxK xV, value := ctxV xV, mask := none })) xV).adjoint δ :=
  backprop_eq_fderiv_forwardVec _ xV δ

/-- With the all-true Boolean mask, the masked code path of `scaledDotProduct` has
the same vector-Jacobian product, computed by the same tape backprop. -/
theorem backprop_eq_fderiv_allTrueMask {m d : Nat}
    (hm : m ≠ 0) (xV : CtxVec (qkvCtx m d))
    (δ : Vec (Shape.size (Shape.dim m (Shape.dim d Shape.scalar)))) :
    Tape.GraphData.backpropVec
        (scaledDotProductGraph (1 / scaleDenom (α := ℝ) d)).toData () xV
        (CtxVec.single (idxOut m d) δ) =
      (fderiv ℝ (fun xV : CtxVec (qkvCtx m d) =>
        toVec (scaledDotProduct (h1 := hm) (h2 := hm)
          { query := ctxQ xV, key := ctxK xV, value := ctxV xV, mask := some (allTrueMask m m) }))
        xV).adjoint δ := by
  have hfun : (fun xV : CtxVec (qkvCtx m d) =>
      toVec (scaledDotProduct (h1 := hm) (h2 := hm)
        { query := ctxQ xV, key := ctxK xV, value := ctxV xV, mask := some (allTrueMask m m) })) =
      fun xV : CtxVec (qkvCtx m d) =>
        toVec (scaledDotProduct (h1 := hm) (h2 := hm)
          { query := ctxQ xV, key := ctxK xV, value := ctxV xV, mask := none }) := by
    funext xV
    exact congrArg toVec
      (Attention.scaledDotProduct_allTrueMask
        (ctx := { query := ctxQ xV, key := ctxK xV, value := ctxV xV, mask := none }))
  rw [hfun]
  exact backprop_eq_fderiv hm xV δ

end Attention

end
end TorchLean

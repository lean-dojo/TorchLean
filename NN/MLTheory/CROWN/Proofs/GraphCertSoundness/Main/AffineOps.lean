/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.Main.Extraction
public import NN.MLTheory.CROWN.Models.Mlp
public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.IntervalLemmas

/-!
# Certificate Soundness: Affine Nodes

Operator cases `linear` and `matmul` of the IBP certificate induction.  Both reduce to one
box-level lemma about `IBP.linear` with a point bias box; `matmul` instantiates the bias with the
zero vector.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN

namespace CertSoundness

noncomputable section

/-! ### Box-level enclosure -/

/-- `IBP.linear` with a point bias box encloses the affine image of an enclosed value, once the
input box and the value are cast to the weight's input dimension. -/
theorem enclosesBox_ibp_linear {B1 : FlatBox ℝ} {v1 : Val} {m n : Nat}
    (W : Tensor ℝ [m, n]) (b : Tensor ℝ [m])
    (h1 : EnclosesBox B1 v1) (hXin : B1.dim = n) (hxDim : v1.n = n) :
    EnclosesBox
      (toFlatBox (α := ℝ) m
        (NN.MLTheory.CROWN.IBP.linear (α := ℝ) W
          (castBoxDim (α := ℝ) hXin (ofFlatBox (α := ℝ) B1)) (Box.point (α := ℝ) b)))
      ⟨m, Spec.linearSpec (α := ℝ) { weights := W, bias := b }
        (castDimScalar (α := ℝ) hxDim v1.v)⟩ := by
  obtain ⟨hDim, hx⟩ := h1
  obtain ⟨n1, lo, hi⟩ := B1
  obtain ⟨k1, x⟩ := v1
  simp only at hDim hXin hxDim
  subst hDim hXin
  simp only [castDimScalar_self] at hx ⊢
  refine ⟨rfl, ?_⟩
  refine encloses_of_contains _ _
    (NN.MLTheory.CROWN.Theorems.ibp_linear_sound_real W _ _ x b ?_ (Box.contains_point_self b))
  simpa using (contains_castBoxDim_iff rfl _ x).2
    (contains_of_encloses { dim := n1, lo := lo, hi := hi } x hx)

/-! ### Node-level soundness -/

/-- Certificate soundness at a `linear` node. -/
theorem linear_node_encloses {nodes : Array Node} {ps : ParamStore ℝ}
    {cert : Array (Option (FlatBox ℝ))} {inputs : Std.HashMap Nat Val} {vals : Array (Option Val)}
    {k : Nat} {B : FlatBox ℝ} {v : Val}
    (hkKind : (nodes[k]!).kind = .linear)
    (hcertStep : certStepNode? nodes ps cert k = some B)
    (hvalStep : evalNode? nodes ps inputs vals k = some v)
    (hpe : ParentsEnclosed nodes cert vals k) : EnclosesBox B v := by
  simp only [certStepNode?, hkKind] at hcertStep
  simp only [evalNode?, hkKind] at hvalStep
  rcases hparents : NN.IR.unaryParent? (nodes[k]!).parents with _ | p1 <;>
    simp only [hparents, reduceCtorEq] at hcertStep hvalStep
  rcases hgb : getBox? cert p1 with _ | B1 <;> simp only [hgb, reduceCtorEq] at hcertStep
  rcases hlin : ps.linearWB[k]? with _ | p <;>
    simp only [ibpLinear, ibpLinearParams, hlin, reduceCtorEq] at hcertStep hvalStep
  rcases hgv : getVal? vals p1 with _ | v1 <;> simp only [hgv, reduceCtorEq] at hvalStep
  have h1 := parents_enclosed_unary hpe hparents hgb hgv
  obtain ⟨hXin, hcertStep⟩ := dite_eq_some_elim hcertStep
  obtain ⟨hxDim, hvalStep⟩ := dite_eq_some_elim hvalStep
  obtain rfl := Option.some.inj hvalStep
  obtain rfl := Option.some.inj hcertStep
  exact enclosesBox_ibp_linear p.w p.b h1 hXin hxDim

/-- Certificate soundness at a `matmul` node. -/
theorem matmul_node_encloses {nodes : Array Node} {ps : ParamStore ℝ}
    {cert : Array (Option (FlatBox ℝ))} {inputs : Std.HashMap Nat Val} {vals : Array (Option Val)}
    {k : Nat} {B : FlatBox ℝ} {v : Val}
    (hkKind : (nodes[k]!).kind = .matmul)
    (hcertStep : certStepNode? nodes ps cert k = some B)
    (hvalStep : evalNode? nodes ps inputs vals k = some v)
    (hpe : ParentsEnclosed nodes cert vals k) : EnclosesBox B v := by
  simp only [certStepNode?, hkKind] at hcertStep
  simp only [evalNode?, hkKind] at hvalStep
  rcases hparents : NN.IR.unaryParent? (nodes[k]!).parents with _ | p1 <;>
    simp only [hparents, reduceCtorEq] at hcertStep hvalStep
  rcases hgb : getBox? cert p1 with _ | B1 <;> simp only [hgb, reduceCtorEq] at hcertStep
  rcases hmat : ps.matmulW[k]? with _ | p <;>
    simp only [ibpMatmul, hmat, reduceCtorEq] at hcertStep hvalStep
  rcases hgv : getVal? vals p1 with _ | v1 <;> simp only [hgv, reduceCtorEq] at hvalStep
  have h1 := parents_enclosed_unary hpe hparents hgb hgv
  obtain ⟨hXin, hcertStep⟩ := dite_eq_some_elim hcertStep
  obtain ⟨hxDim, hvalStep⟩ := dite_eq_some_elim hvalStep
  obtain rfl := Option.some.inj hvalStep
  obtain rfl := Option.some.inj hcertStep
  exact enclosesBox_ibp_linear p.w (Tensor.full (α := ℝ) (.dim p.m .scalar) 0) h1 hXin hxDim

end

end CertSoundness

end NN.MLTheory.CROWN.Graph

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.Main.Extraction
public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.IntervalLemmas

/-!
# Certificate Soundness: Elementwise Arithmetic Nodes

Operator cases `add`, `sub`, and `mulElem` of the IBP certificate induction.  Each operator has a
box-level lemma (enclosure of the propagated box, with the dimension bookkeeping done once) and a
node-level lemma that reads the parents off the graph and applies it.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN

namespace CertSoundness

noncomputable section

/-! ### Box-level enclosure -/

/-- `boxAdd` encloses the sum of enclosed values with matching dimensions. -/
theorem enclosesBox_boxAdd {B1 B2 : FlatBox ℝ} {v1 v2 : Val}
    (h1 : EnclosesBox B1 v1) (h2 : EnclosesBox B2 v2) (hxy : v1.n = v2.n) :
    EnclosesBox (boxAdd (α := ℝ) B1 B2)
      ⟨v1.n, Tensor.addSpec (α := ℝ) v1.v (castDimScalar (α := ℝ) hxy.symm v2.v)⟩ := by
  obtain ⟨hDim1, hx⟩ := h1
  obtain ⟨hDim2, hy⟩ := h2
  obtain ⟨n1, lo1, hi1⟩ := B1
  obtain ⟨n2, lo2, hi2⟩ := B2
  obtain ⟨m1, x⟩ := v1
  obtain ⟨m2, y⟩ := v2
  simp only at hDim1 hDim2 hxy
  subst hDim1 hDim2 hxy
  simp only [castDimScalar_self] at hx hy ⊢
  rw [NN.MLTheory.CROWN.Graph.Theorems.box_add_on_eq]
  exact ⟨rfl, NN.MLTheory.CROWN.Graph.Theorems.Semantics.box_add_sound (α := ℝ) n1 lo1 hi1 lo2 hi2
    add_le_add x y hx hy⟩

/-- `boxSub` encloses the difference of enclosed values with matching dimensions. -/
theorem enclosesBox_boxSub {B1 B2 : FlatBox ℝ} {v1 v2 : Val}
    (h1 : EnclosesBox B1 v1) (h2 : EnclosesBox B2 v2) (hxy : v1.n = v2.n) :
    EnclosesBox (boxSub (α := ℝ) B1 B2)
      ⟨v1.n, Tensor.subSpec (α := ℝ) v1.v (castDimScalar (α := ℝ) hxy.symm v2.v)⟩ := by
  obtain ⟨hDim1, hx⟩ := h1
  obtain ⟨hDim2, hy⟩ := h2
  obtain ⟨n1, lo1, hi1⟩ := B1
  obtain ⟨n2, lo2, hi2⟩ := B2
  obtain ⟨m1, x⟩ := v1
  obtain ⟨m2, y⟩ := v2
  simp only at hDim1 hDim2 hxy
  subst hDim1 hDim2 hxy
  simp only [castDimScalar_self] at hx hy ⊢
  rw [NN.MLTheory.CROWN.Graph.Theorems.box_sub_on_eq]
  exact ⟨rfl, NN.MLTheory.CROWN.Graph.Theorems.Semantics.box_sub_sound (α := ℝ) n1 lo1 hi1 lo2 hi2
    sub_le_sub x y hx hy⟩

/-- A successful `boxMulElem` encloses the elementwise product of enclosed values. -/
theorem enclosesBox_boxMulElem {B1 B2 Bm : FlatBox ℝ} {v1 v2 : Val}
    (h1 : EnclosesBox B1 v1) (h2 : EnclosesBox B2 v2) (hxy : v1.n = v2.n)
    (hmul : boxMulElem (α := ℝ) B1 B2 = some Bm) :
    EnclosesBox Bm
      ⟨v1.n, Tensor.mulSpec (α := ℝ) v1.v (castDimScalar (α := ℝ) hxy.symm v2.v)⟩ := by
  obtain ⟨hDim1, hx⟩ := h1
  obtain ⟨hDim2, hy⟩ := h2
  obtain ⟨n1, lo1, hi1⟩ := B1
  obtain ⟨n2, lo2, hi2⟩ := B2
  obtain ⟨m1, x⟩ := v1
  obtain ⟨m2, y⟩ := v2
  simp only at hDim1 hDim2 hxy
  subst hDim1 hDim2 hxy
  simp only [castDimScalar_self] at hx hy ⊢
  exact box_mulElem_sound_real n1 lo1 hi1 lo2 hi2 x y hx hy hmul

/-! ### Node-level soundness -/

/-- Certificate soundness at an `add` node. -/
theorem add_node_encloses {nodes : Array Node} {ps : ParamStore ℝ}
    {cert : Array (Option (FlatBox ℝ))} {inputs : Std.HashMap Nat Val} {vals : Array (Option Val)}
    {k : Nat} {B : FlatBox ℝ} {v : Val}
    (hkKind : (nodes[k]!).kind = .add)
    (hcertStep : certStepNode? nodes ps cert k = some B)
    (hvalStep : evalNode? nodes ps inputs vals k = some v)
    (hpe : ParentsEnclosed nodes cert vals k) : EnclosesBox B v := by
  simp only [certStepNode?, hkKind] at hcertStep
  simp only [evalNode?, hkKind] at hvalStep
  rcases hparents : NN.IR.binaryParents? (nodes[k]!).parents with _ | ⟨p1, p2⟩ <;>
    simp only [hparents, reduceCtorEq] at hcertStep hvalStep
  rcases hgb1 : getBox? cert p1 with _ | B1 <;> simp only [hgb1, reduceCtorEq] at hcertStep
  rcases hgb2 : getBox? cert p2 with _ | B2 <;> simp only [hgb2, reduceCtorEq] at hcertStep
  rcases hgv1 : getVal? vals p1 with _ | v1 <;> simp only [hgv1, reduceCtorEq] at hvalStep
  rcases hgv2 : getVal? vals p2 with _ | v2 <;> simp only [hgv2, reduceCtorEq] at hvalStep
  obtain ⟨h1, h2⟩ := parents_enclosed_binary hpe hparents hgb1 hgb2 hgv1 hgv2
  obtain ⟨hxy, hvalStep⟩ := dite_eq_some_elim hvalStep
  obtain ⟨-, hcertStep⟩ := dite_eq_some_elim hcertStep
  obtain rfl := Option.some.inj hvalStep
  obtain rfl := Option.some.inj hcertStep
  exact enclosesBox_boxAdd h1 h2 hxy

/-- Certificate soundness at a `sub` node. -/
theorem sub_node_encloses {nodes : Array Node} {ps : ParamStore ℝ}
    {cert : Array (Option (FlatBox ℝ))} {inputs : Std.HashMap Nat Val} {vals : Array (Option Val)}
    {k : Nat} {B : FlatBox ℝ} {v : Val}
    (hkKind : (nodes[k]!).kind = .sub)
    (hcertStep : certStepNode? nodes ps cert k = some B)
    (hvalStep : evalNode? nodes ps inputs vals k = some v)
    (hpe : ParentsEnclosed nodes cert vals k) : EnclosesBox B v := by
  simp only [certStepNode?, hkKind] at hcertStep
  simp only [evalNode?, hkKind] at hvalStep
  rcases hparents : NN.IR.binaryParents? (nodes[k]!).parents with _ | ⟨p1, p2⟩ <;>
    simp only [hparents, reduceCtorEq] at hcertStep hvalStep
  rcases hgb1 : getBox? cert p1 with _ | B1 <;> simp only [hgb1, reduceCtorEq] at hcertStep
  rcases hgb2 : getBox? cert p2 with _ | B2 <;> simp only [hgb2, reduceCtorEq] at hcertStep
  rcases hgv1 : getVal? vals p1 with _ | v1 <;> simp only [hgv1, reduceCtorEq] at hvalStep
  rcases hgv2 : getVal? vals p2 with _ | v2 <;> simp only [hgv2, reduceCtorEq] at hvalStep
  obtain ⟨h1, h2⟩ := parents_enclosed_binary hpe hparents hgb1 hgb2 hgv1 hgv2
  obtain ⟨hxy, hvalStep⟩ := dite_eq_some_elim hvalStep
  obtain ⟨-, hcertStep⟩ := dite_eq_some_elim hcertStep
  obtain rfl := Option.some.inj hvalStep
  obtain rfl := Option.some.inj hcertStep
  exact enclosesBox_boxSub h1 h2 hxy

/-- Certificate soundness at a `mulElem` node. -/
theorem mulElem_node_encloses {nodes : Array Node} {ps : ParamStore ℝ}
    {cert : Array (Option (FlatBox ℝ))} {inputs : Std.HashMap Nat Val} {vals : Array (Option Val)}
    {k : Nat} {B : FlatBox ℝ} {v : Val}
    (hkKind : (nodes[k]!).kind = .mulElem)
    (hcertStep : certStepNode? nodes ps cert k = some B)
    (hvalStep : evalNode? nodes ps inputs vals k = some v)
    (hpe : ParentsEnclosed nodes cert vals k) : EnclosesBox B v := by
  simp only [certStepNode?, hkKind] at hcertStep
  simp only [evalNode?, hkKind] at hvalStep
  rcases hparents : NN.IR.binaryParents? (nodes[k]!).parents with _ | ⟨p1, p2⟩ <;>
    simp only [hparents, reduceCtorEq] at hcertStep hvalStep
  rcases hgb1 : getBox? cert p1 with _ | B1 <;> simp only [hgb1, reduceCtorEq] at hcertStep
  rcases hgb2 : getBox? cert p2 with _ | B2 <;> simp only [hgb2, reduceCtorEq] at hcertStep
  rcases hgv1 : getVal? vals p1 with _ | v1 <;> simp only [hgv1, reduceCtorEq] at hvalStep
  rcases hgv2 : getVal? vals p2 with _ | v2 <;> simp only [hgv2, reduceCtorEq] at hvalStep
  obtain ⟨h1, h2⟩ := parents_enclosed_binary hpe hparents hgb1 hgb2 hgv1 hgv2
  obtain ⟨hxy, hvalStep⟩ := dite_eq_some_elim hvalStep
  obtain rfl := Option.some.inj hvalStep
  exact enclosesBox_boxMulElem h1 h2 hxy hcertStep

end

end CertSoundness

end NN.MLTheory.CROWN.Graph

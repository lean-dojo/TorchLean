/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.Main.BinaryMatmul
public import NN.MLTheory.CROWN.Models.Mlp
import NN.Tensor.Internal.Laws.Sequence

/-!
# Certificate Soundness: Affine Nodes

Operator cases `linear`, `matmul`, and `concat` of the IBP certificate induction. Linear maps
reduce to `IBP.linear` with a point bias box. Concatenation transports parent enclosures through
the same coordinate bijection used by the executable CROWN passes.
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

/-- Casting an enclosed box and value to their common dimension preserves each scalar bound. -/
private theorem enclosesBox_cast_coordinate {B : FlatBox ℝ} {v : Val} {n : Nat}
    (hB : B.dim = n) (hv : v.n = n) (h : EnclosesBox B v) (i : Fin n) :
    Tensor.getScalar (hB ▸ B.lo) i ≤ Tensor.getScalar (hv ▸ v.v) i ∧
      Tensor.getScalar (hv ▸ v.v) i ≤ Tensor.getScalar (hB ▸ B.hi) i := by
  rcases B with ⟨d, lower, upper⟩
  rcases v with ⟨e, value⟩
  dsimp only at hB hv
  subst d e
  obtain ⟨hDim, h⟩ := h
  simpa only [castDimScalar_self] using h i

/-- Concatenated tensor families preserve the scalar enclosure of every parent occurrence. -/
private theorem enclosesBox_constructed_concat (layout : ConcatLayout)
    (lower upper values : (parent : Fin layout.lengths.length) →
      Tensor ℝ [(layout.parentShape parent).size])
    (h : ∀ parent index,
      Tensor.getScalar (lower parent) index ≤ Tensor.getScalar (values parent) index ∧
        Tensor.getScalar (values parent) index ≤ Tensor.getScalar (upper parent) index) :
    EnclosesBox
      { dim := layout.outputShape.size
        lo := layout.concat lower
        hi := layout.concat upper }
      { n := layout.outputShape.size, v := layout.concat values } := by
  refine ⟨rfl, ?_⟩
  change ∀ index : Fin layout.outputShape.size,
    Tensor.getScalar (layout.concat lower) index ≤
        Tensor.getScalar (layout.concat values) index ∧
      Tensor.getScalar (layout.concat values) index ≤
        Tensor.getScalar (layout.concat upper) index
  exact ConcatLayout.concat_encloses (α := ℝ) layout lower upper values h

/-- The checked flat concat preserves enclosure for every parent occurrence. -/
theorem enclosesBox_concat {layout : ConcatLayout}
    {boxes : Array (FlatBox ℝ)} {values : Array Val} {B : FlatBox ℝ} {v : Val}
    (hboxes : concatFlatBoxes? layout boxes = some B)
    (hvalues : concatFlatValues? layout values = some v)
    (hparents : ∀ (i : Nat) (Bp : FlatBox ℝ) (vp : Val),
      boxes[i]? = some Bp → values[i]? = some vp →
      EnclosesBox Bp vp) : EnclosesBox B v := by
  unfold concatFlatBoxes? at hboxes
  split at hboxes
  · obtain ⟨inputsB, hinputsB, hboxes⟩ := Option.bind_eq_some_iff.mp hboxes
    obtain ⟨hshapeB, hboxes⟩ := dite_eq_some_elim hboxes
    obtain rfl := Option.some.inj hboxes
    unfold concatFlatValues? at hvalues
    split at hvalues
    · obtain ⟨inputsV, hinputsV, hvalues⟩ := Option.bind_eq_some_iff.mp hvalues
      obtain ⟨hshapeV, hvalues⟩ := dite_eq_some_elim hvalues
      obtain rfl := Option.some.inj hvalues
      refine enclosesBox_constructed_concat layout
        (fun parent => hshapeB parent ▸ (inputsB parent).lo)
        (fun parent => hshapeB parent ▸ (inputsB parent).hi)
        (fun parent => hshapeV parent ▸ (inputsV parent).v) ?_
      intro parent index
      exact enclosesBox_cast_coordinate (hshapeB parent) (hshapeV parent)
        (hparents parent.val (inputsB parent) (inputsV parent)
          (Tensor.Internal.sequenceFinM_get_of_eq_some hinputsB parent)
          (Tensor.Internal.sequenceFinM_get_of_eq_some hinputsV parent)) index
    · contradiction
  · contradiction

/-! ### Node-level soundness -/

/-- Certificate soundness for concat on any validated axis and number of parent occurrences. -/
theorem concat_node_encloses {nodes : Array Node} {ps : ParamStore ℝ}
    {cert : Array (Option (FlatBox ℝ))} {inputs : Std.HashMap Nat Val} {vals : Array (Option Val)}
    {k axis : Nat} {B : FlatBox ℝ} {v : Val}
    (hkKind : (nodes[k]!).kind = .concat axis)
    (hcertStep : certStepNode? nodes ps cert k = some B)
    (hvalStep : evalNode? nodes ps inputs vals k = some v)
    (hpe : ParentsEnclosed nodes cert vals k) : EnclosesBox B v := by
  simp only [certStepNode?, hkKind] at hcertStep
  simp only [evalNode?, hkKind] at hvalStep
  cases hlayout : concatNodeLayout? nodes nodes[k]! axis with
  | none => simp [hlayout] at hcertStep
  | some layout =>
      cases hboxes : (nodes[k]!).parents.mapM (getBox? cert) with
      | none => simp [hlayout, hboxes] at hcertStep
      | some boxes =>
          cases hvalues : (nodes[k]!).parents.mapM (getVal? vals) with
          | none => simp [hlayout, hvalues] at hvalStep
          | some values =>
              have hB : concatFlatBoxes? layout boxes = some B := by
                simpa [hlayout, hboxes] using hcertStep
              have hv : concatFlatValues? layout values = some v := by
                simpa [hlayout, hvalues] using hvalStep
              refine enclosesBox_concat hB hv ?_
              intro i Bp vp hBp hvp
              rw [array_mapM_getElem?_of_eq_some hboxes i] at hBp
              rw [array_mapM_getElem?_of_eq_some hvalues i] at hvp
              obtain ⟨parent, hp, hbox⟩ := Option.bind_eq_some_iff.mp hBp
              have hvalue : getVal? vals parent = some vp := by
                simpa [hp] using hvp
              exact hpe parent (Array.mem_of_getElem? hp) Bp vp hbox hvalue

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
  cases hparents : NN.IR.unaryParent? (nodes[k]!).parents with
  | none => exact binary_matmul_node_encloses hkKind hparents hcertStep hvalStep hpe
  | some p1 =>
      simp only [certStepNode?, hkKind] at hcertStep
      simp only [evalNode?, hkKind] at hvalStep
      simp only [hparents] at hcertStep hvalStep
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

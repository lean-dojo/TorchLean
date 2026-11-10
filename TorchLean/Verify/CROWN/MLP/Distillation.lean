/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.MLP

/-!
# Distillation certificates for two-layer MLPs

A student network matches a teacher network up to `eps` on an input box `xB` when
`|T(x)ᵢ - S(x)ᵢ| ≤ eps` for every `x` in `xB` and every output coordinate `i`. We check this over
`ℝ` by bounding both networks with `boundIBP`, subtracting the output boxes with `Graph.boxSub` and
testing that the difference box lies in `[-eps, eps]`. Soundness follows from
`TwoLayerMLP.boundIBP_sound` and `Directed.boxSub_encloses`.

We wrote the check by hand, and it is coarse. The two output boxes come from separate IBP runs, so
the subtraction forgets that both networks read the same `x`. If the student is the teacher and an
output box is `[0, 1]`, the difference box is `[-1, 1]` and the check needs `eps ≥ 1`, where the
true gap is `0`.

Main declarations: `BoxWithinAbs`, `checkBoxWithinAbs`, `checkEquivalence`,
`checkEquivalence_sound`.
-/

@[expose] public section

namespace TorchLean.CROWN.Distillation

open Tensor
open TwoLayerMLP

/-! ## Boxes within `[-eps, eps]` -/

/-- Every coordinate of the box `B` lies in `[-eps, eps]`: `-eps ≤ B.loᵢ` and `B.hiᵢ ≤ eps`. -/
def BoxWithinAbs {n : Nat} (B : Box ℝ (.dim n .scalar)) (eps : ℝ) : Prop :=
  ∀ i : Fin n, (-eps ≤ B.lo.getScalar i) ∧ (B.hi.getScalar i ≤ eps)

/-- Classical `Bool` decision of `BoxWithinAbs B eps`. It is noncomputable because the carrier is
`ℝ`, and it serves as the reference checker that soundness statements refer to. -/
noncomputable def checkBoxWithinAbs {n : Nat} (B : Box ℝ (.dim n .scalar)) (eps : ℝ) : Bool := by
  classical
  exact decide (BoxWithinAbs (n := n) B eps)

/-- `checkBoxWithinAbs B eps` returns `true` exactly when `BoxWithinAbs B eps` holds. -/
theorem checkBoxWithinAbs_spec {n : Nat} {B : Box ℝ (.dim n .scalar)} {eps : ℝ} :
    checkBoxWithinAbs (n := n) B eps = true ↔ BoxWithinAbs (n := n) B eps := by
  classical
  simp [checkBoxWithinAbs, decide_eq_true_eq]

/-! ## Distillation certificate for two-layer MLPs -/

/-- Real-valued reference checker: it returns `true` exactly when the IBP output boxes of `teacher`
and `student` on `xB`, subtracted with `Graph.boxSub`, lie in `[-eps, eps]` in every coordinate. -/
noncomputable def checkEquivalence {inDim hidDim outDim : Nat}
    (teacher student : CROWN.TwoLayerMLP ℝ inDim hidDim outDim)
    (xB : Box ℝ (.dim inDim .scalar))
    (eps : ℝ) : Bool :=
  let tB := Graph.toFlatBox outDim (CROWN.TwoLayerMLP.boundIBP (α := ℝ) teacher xB)
  let sB := Graph.toFlatBox outDim (CROWN.TwoLayerMLP.boundIBP (α := ℝ) student xB)
  checkBoxWithinAbs (Graph.ofFlatBox (Graph.boxSub (α := ℝ) tB sB)) eps

/-- If `checkEquivalence teacher student xB eps = true`, then every `x` in `xB` and every
output coordinate `i` satisfy `|T(x)ᵢ - S(x)ᵢ| ≤ eps`, where `T` and `S` are the two `forward` maps.
-/
theorem checkEquivalence_sound {inDim hidDim outDim : Nat}
    (teacher student : CROWN.TwoLayerMLP ℝ inDim hidDim outDim)
    (xB : Box ℝ (.dim inDim .scalar))
    (eps : ℝ)
    (hcheck : checkEquivalence (inDim := inDim) (hidDim := hidDim) (outDim := outDim)
      teacher student xB eps = true) :
    ∀ x : Tensor ℝ [inDim],
      Box.Contains (α := ℝ) xB x →
      ∀ i : Fin outDim,
        |getScalar (CROWN.TwoLayerMLP.forward (α := ℝ) teacher x -
          CROWN.TwoLayerMLP.forward (α := ℝ) student x) i| ≤ eps := by
  intro x hx i
  have hwithin := checkBoxWithinAbs_spec.mp hcheck
  have ht := TwoLayerMLP.boundIBP_sound (net := teacher) (xB := xB) (x := x) hx
  have hs := TwoLayerMLP.boundIBP_sound (net := student) (xB := xB) (x := x) hx
  -- The output boxes enclose the coordinates of the two outputs.
  have hrow : ∀ net : CROWN.TwoLayerMLP ℝ inDim hidDim outDim,
      Box.Contains (α := ℝ) (boundIBP (α := ℝ) net xB) (forward (α := ℝ) net x) →
      Directed.RowEncloses (Graph.toFlatBox outDim (boundIBP (α := ℝ) net xB))
        outDim fun j => getAtOrZero (forward (α := ℝ) net x) [j] := fun net h => by
    have hc : ∀ j : Fin outDim,
        getScalar (boundIBP (α := ℝ) net xB).lo j ≤ getScalar (forward (α := ℝ) net x) j ∧
          getScalar (forward (α := ℝ) net x) j ≤ getScalar (boundIBP (α := ℝ) net xB).hi j := h
    refine ⟨rfl, fun j => ?_⟩
    simpa only [Graph.toFlatBox, Tensor.getAtOrZero_eq_getScalar, LawfulBoundOps.toReal,
      instLawfulBoundOpsReal, id_eq] using hc j
  -- A box of width `outDim` inside `[-eps, eps]` bounds every coordinate it encloses.
  have hbound : ∀ (B : FlatBox ℝ) (f : Nat → ℝ), Directed.RowEncloses B outDim f →
      BoxWithinAbs (Graph.ofFlatBox B) eps → |f i.val| ≤ eps := by
    rintro ⟨d, lo, hi⟩ f ⟨hd, hf⟩ hW
    obtain rfl : d = outDim := hd
    have hfi := hf i
    have hWi := hW i
    simp only [Graph.ofFlatBox, Tensor.getAtOrZero_eq_getScalar, LawfulBoundOps.toReal, id_eq]
      at hfi hWi
    exact abs_le.mpr ⟨hWi.1.trans hfi.1, hfi.2.trans hWi.2⟩
  -- The difference box encloses the difference of the two outputs.
  simpa only [Tensor.getAtOrZero_eq_getScalar, getScalar_sub] using
    hbound _ _ (Directed.boxSub_encloses (hrow teacher ht) (hrow student hs)) hwithin

end TorchLean.CROWN.Distillation

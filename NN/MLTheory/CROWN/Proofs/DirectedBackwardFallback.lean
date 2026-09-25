/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.DirectedBackwardSoundness

/-!
# Soundness of the output-box fallback

The public objective API can return a constant enclosure when its directed sweep fails. This
file proves the sign-selected directed dot product used on that path.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean NN.IR
open TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph.Internal
open scoped BigOperators

noncomputable section

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

omit [Context α] [BoundOps α] [LawfulBoundOps α] in
private theorem getDimScalarFn_item {n : Nat} (t : Tensor α [n]) (i : Fin n) :
    (getDimScalarFn t i).item = t.getScalar i := by
  simp only [getDimScalarFn, Tensor.getScalar, Spec.get]

/-- Interpret which side of a real objective a directed scalar bounds. -/
def DirectionBound (dir : BackwardDir) (bound : α) (z : ℝ) : Prop :=
  match dir with
  | .lower => value bound ≤ z
  | .upper => z ≤ value bound

/-- The executable sign-selected box dot product bounds its real objective. -/
theorem consumeObjectiveFromBox_encloses (hzero : value (0 : α) = 0)
    (dir : BackwardDir) (obj : FlatTensor α) (box : FlatBox α) (x : Nat → ℝ)
    (hx : RowEncloses box obj.n x) {bound : α}
    (hresult : consumeObjectiveFromBox dir obj box = some bound) :
    DirectionBound dir bound
      (dot obj.n (fun i => value (getAtOrZero obj.v [i])) x) := by
  obtain ⟨n, coefficients⟩ := obj
  obtain ⟨m, lo, hi⟩ := box
  obtain ⟨hdim, hx⟩ := hx
  dsimp only at hdim
  subst m
  let product (direction : BackwardDir) (i : Fin n) : α :=
    let a := coefficients.getScalar i
    let l := lo.getScalar i
    let u := hi.getScalar i
    match direction with
    | .lower => BoundOps.mulDown a (if 0 < a then l else u)
    | .upper => BoundOps.mulUp a (if 0 < a then u else l)
  have hterms (i : Fin n) :
      value (product .lower i) ≤ value (coefficients.getScalar i) * x i.val ∧
        value (coefficients.getScalar i) * x i.val ≤ value (product .upper i) := by
    have hb := hx i
    simp only [read_fin] at hb
    by_cases ha : (0 : α) < coefficients.getScalar i
    · have hp : 0 ≤ value (coefficients.getScalar i) := by
        simpa only [hzero] using ((LawfulBoundOps.lt_iff _ _).mp ha).le
      simp only [product, ha, ↓reduceIte]
      exact ⟨(LawfulBoundOps.mulDown_le _ _).trans
          (mul_le_mul_of_nonneg_left hb.1 hp),
        (mul_le_mul_of_nonneg_left hb.2 hp).trans (LawfulBoundOps.le_mulUp _ _)⟩
    · have hn : value (coefficients.getScalar i) ≤ 0 := by
        rw [← hzero]
        exact le_of_not_gt (fun h => ha ((LawfulBoundOps.lt_iff _ _).mpr h))
      simp only [product, ha, ↓reduceIte]
      exact ⟨(LawfulBoundOps.mulDown_le _ _).trans
          (mul_le_mul_of_nonpos_left hb.2 hn),
        (mul_le_mul_of_nonpos_left hb.1 hn).trans (LawfulBoundOps.le_mulUp _ _)⟩
  have hs := sum_encloses hzero (product .lower) (product .upper)
    (fun i => value (coefficients.getScalar i) * x i.val) hterms
  cases dir <;>
    simp only [consumeObjectiveFromBox, ↓reduceDIte, castDimScalar_self,
      ← Array.foldl_toList, Array.toList_map, List.foldl_map,
      Array.finRange, Array.toList_ofFn, getDimScalarFn_item, decide_eq_true_eq,
      Option.some.injEq] at hresult
  · subst bound
    simpa only [DirectionBound, dot, read_fin, product,
      List.finRange] using hs.1
  · subst bound
    simpa only [DirectionBound, dot, read_fin, product,
      List.finRange] using hs.2

/-- Constant objective forms have exactly their stored real constant as value. -/
theorem constantObjectiveAffine_value (hzero : value (0 : α) = 0)
    (n : Nat) (c : α) (x : Fin n → ℝ) :
    affineValue (constantObjectiveAffine n c) x = value c := by
  simp [affineValue, constantObjectiveAffine, Spec.get2_full, hzero]

/-- A successful output-box fallback returns an affine bound in the requested direction. -/
theorem objectiveFromOutputBox_encloses (hzero : value (0 : α) = 0)
    (dir : BackwardDir) (ibp : Array (Option (FlatBox α))) (output n : Nat)
    (obj : FlatTensor α) (y : Nat → ℝ)
    (hbox : ∀ box, ibp[output]! = some box → RowEncloses box obj.n y)
    {aff : AffineVec α n 1}
    (hresult : objectiveFromOutputBox dir ibp output n obj = some aff)
    (x : Fin n → ℝ) :
    match dir with
    | .lower => affineValue aff x ≤ dot obj.n (fun i => value (getAtOrZero obj.v [i])) y
    | .upper => dot obj.n (fun i => value (getAtOrZero obj.v [i])) y ≤ affineValue aff x := by
  simp only [objectiveFromOutputBox, Option.bind_eq_bind, Option.pure_def] at hresult
  cases hentry : ibp[output]? with
  | none => simp only [hentry] at hresult; cases hresult
  | some entry =>
    simp only [hentry] at hresult
    obtain ⟨box, hb, hresult⟩ := Option.bind_eq_some_iff.mp hresult
    obtain ⟨bound, hc, hresult⟩ := Option.bind_eq_some_iff.mp hresult
    cases Option.some.inj hresult
    rw [hb] at hentry
    have hlookup : ibp[output]! = some box := by
      obtain ⟨hindex, hget⟩ := Array.getElem?_eq_some_iff.mp hentry
      simpa only [getElem!_pos (c := ibp) (i := output) hindex] using hget
    have hs := consumeObjectiveFromBox_encloses hzero dir obj box y (hbox box hlookup) hc
    cases dir <;> simpa only [constantObjectiveAffine_value hzero, DirectionBound] using hs

end

end NN.MLTheory.CROWN.Graph.DirectedBackward

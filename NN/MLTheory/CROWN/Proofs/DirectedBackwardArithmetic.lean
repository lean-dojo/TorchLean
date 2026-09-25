/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine.BackwardObjective
public import NN.MLTheory.CROWN.Proofs.LayerNormDirected

/-!
# Arithmetic of the rounded backward sweep

The backward engine stores intervals for coefficients and for its accumulated constant. These
lemmas interpret the engine's operations over the reals. Ordinary backend addition, subtraction,
and multiplication need not be exact.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph.Internal
open NN.MLTheory.CROWN.IntervalLemmas (intervalMul_encloses)
open scoped BigOperators

noncomputable section

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

/-- Directed accumulation encloses the mathematical sum, including a nonzero initial constant. -/
theorem foldl_encloses {ι : Type} (indices : List ι) (lo hi : ι → α) (f : ι → ℝ)
    (initialLo initialHi : α) (initial : ℝ)
    (hlo : value initialLo ≤ initial) (hhi : initial ≤ value initialHi)
    (hterms : ∀ i ∈ indices, value (lo i) ≤ f i ∧ f i ≤ value (hi i)) :
    value (indices.foldl (fun acc i => BoundOps.addDown acc (lo i)) initialLo) ≤
        initial + (indices.map f).sum ∧
      initial + (indices.map f).sum ≤
        value (indices.foldl (fun acc i => BoundOps.addUp acc (hi i)) initialHi) := by
  induction indices generalizing initialLo initialHi initial with
  | nil => simpa using And.intro hlo hhi
  | cons i indices ih =>
      have ht := hterms i (by simp)
      have hnextLo := (LawfulBoundOps.addDown_le initialLo (lo i)).trans (add_le_add hlo ht.1)
      have hnextHi := (add_le_add hhi ht.2).trans (LawfulBoundOps.le_addUp initialHi (hi i))
      simpa only [List.foldl_cons, List.map_cons, List.sum_cons, add_assoc] using
        ih (BoundOps.addDown initialLo (lo i)) (BoundOps.addUp initialHi (hi i))
          (initial + f i) hnextLo hnextHi (fun j hj => hterms j (by simp [hj]))

/-- A selected slope and its directed correction bound every coefficient and input in their boxes.

For a box crossing zero, the correction uses the negative input endpoint. Its multiplication
reverses the inequality for the coefficient width, hence the opposite direction on subtraction.
-/
theorem coeffAffine_encloses
    {l u al au : α} {x a : ℝ}
    (hl : value l ≤ x) (hu : x ≤ value u)
    (hal : value al ≤ a) (hau : a ≤ value au) :
    value (directedCoeffAffine l u al au).1.1 * x +
        value (directedCoeffAffine l u al au).1.2 ≤ a * x ∧
      a * x ≤ value (directedCoeffAffine l u al au).2.1 * x +
        value (directedCoeffAffine l u al au).2.2 := by
  by_cases hneg : l < (0 : α)
  · have hlneg : value l < 0 := by
      simpa only [(LawfulBoundOps.toReal_zero (α := α))] using (LawfulBoundOps.lt_iff l 0).mp hneg
    by_cases hpos : (0 : α) < u
    · simp only [directedCoeffAffine, hneg, not_true_eq_false, decide_false,
        Bool.false_eq_true, ↓reduceIte, hpos]
      have hwidth := LawfulBoundOps.le_subUp au al
      have hwidth' := LawfulBoundOps.subDown_le al au
      have hcorrLo := (LawfulBoundOps.mulDown_le (BoundOps.subUp au al) l).trans
        (mul_le_mul_of_nonpos_right hwidth hlneg.le)
      have hcorrHi := (mul_le_mul_of_nonpos_right hwidth' hlneg.le).trans
        (LawfulBoundOps.le_mulUp (BoundOps.subDown al au) l)
      have hax := mul_nonneg (sub_nonneg.mpr hal) (sub_nonneg.mpr hl)
      have haw := mul_nonneg (sub_nonneg.mpr hau) (neg_nonneg.mpr hlneg.le)
      have hux := mul_nonneg (sub_nonneg.mpr hau) (sub_nonneg.mpr hl)
      have halw := mul_nonneg (sub_nonneg.mpr hal) (neg_nonneg.mpr hlneg.le)
      constructor <;> nlinarith
    · have hunonpos : value u ≤ 0 := by
        rw [← (LawfulBoundOps.toReal_zero (α := α))]
        exact le_of_not_gt fun h => hpos ((LawfulBoundOps.lt_iff 0 u).mpr h)
      have hx := hu.trans hunonpos
      simpa [directedCoeffAffine, hneg, hpos, (LawfulBoundOps.toReal_zero (α := α))] using
        And.intro (mul_le_mul_of_nonpos_right hau hx) (mul_le_mul_of_nonpos_right hal hx)
  · have hlnonneg : 0 ≤ value l := by
      rw [← (LawfulBoundOps.toReal_zero (α := α))]
      exact le_of_not_gt fun h => hneg ((LawfulBoundOps.lt_iff l 0).mpr h)
    have hx := hlnonneg.trans hl
    simpa [directedCoeffAffine, hneg, (LawfulBoundOps.toReal_zero (α := α))] using
      And.intro (mul_le_mul_of_nonneg_right hal hx) (mul_le_mul_of_nonneg_right hau hx)

omit [BoundOps α] [LawfulBoundOps α] in
/-- A valid flat coordinate read agrees with the typed vector accessor. -/
theorem read_fin {n : Nat} (v : Tensor α [n]) (i : Fin n) :
    getAtOrZero v [i.val] = Tensor.getScalar v i := by
  simp [get_at_or_zero_dim_cons, i.isLt, Tensor.getScalar, Spec.get]

/-- Directed summation over all coordinates encloses the finite real sum. -/
theorem sum_encloses
    {n : Nat} (lo hi : Fin n → α) (f : Fin n → ℝ)
    (hterms : ∀ i, value (lo i) ≤ f i ∧ f i ≤ value (hi i)) :
    value ((List.finRange n).foldl (fun acc i => BoundOps.addDown acc (lo i)) 0) ≤
        ∑ i, f i ∧
      ∑ i, f i ≤
        value ((List.finRange n).foldl (fun acc i => BoundOps.addUp acc (hi i)) 0) := by
  have h := foldl_encloses (List.finRange n) lo hi f 0 0 0
    (by simp [LawfulBoundOps.toReal_zero (α := α)]) (by simp [LawfulBoundOps.toReal_zero (α := α)])
    (fun i _ => hterms i)
  have hsum : ((List.finRange n).map f).sum = ∑ i, f i := by
    rw [List.sum_eq_foldl, List.foldl_map]
    exact List.finRange_foldl_add_eq_finset_sum f
  simpa only [hsum, zero_add] using h

/-- The executable interval dot product encloses the dot product of any two enclosed vectors. -/
theorem dotBox_encloses
    {n : Nat} (aLo aHi bLo bHi : Tensor α [n]) (a b : Fin n → ℝ)
    (ha : ∀ i, value (aLo.getScalar i) ≤ a i ∧ a i ≤ value (aHi.getScalar i))
    (hb : ∀ i, value (bLo.getScalar i) ≤ b i ∧ b i ≤ value (bHi.getScalar i))
    {lo hi : α}
    (hresult : directedDotBox { dim := n, lo := aLo, hi := aHi }
      { dim := n, lo := bLo, hi := bHi } = some (lo, hi)) :
    value lo ≤ ∑ i, a i * b i ∧ (∑ i, a i * b i) ≤ value hi := by
  let term (i : Fin n) := intervalMul
    (aLo.getScalar i) (aHi.getScalar i) (bLo.getScalar i) (bHi.getScalar i)
  have hterms (i : Fin n) : value (term i).1 ≤ a i * b i ∧
      a i * b i ≤ value (term i).2 :=
    intervalMul_encloses (ha i).1 (ha i).2 (hb i).1 (hb i).2
  have h := sum_encloses (fun i => (term i).1) (fun i => (term i).2)
    (fun i => a i * b i) hterms
  simp only [directedDotBox, ↓reduceDIte, castDimScalar_self, read_fin,
    List.foldl_map, Option.some.injEq, Prod.mk.injEq] at hresult
  obtain ⟨rfl, rfl⟩ := hresult
  exact h

private theorem foldl_pair {ι β γ : Type} (indices : List ι)
    (f : β → ι → β) (g : γ → ι → γ) (b : β) (c : γ) :
    indices.foldl (fun acc i => (f acc.1 i, g acc.2 i)) (b, c) =
      (indices.foldl f b, indices.foldl g c) := by
  induction indices generalizing b c with
  | nil => rfl
  | cons i indices ih => exact ih (f b i) (g c i)

/-- Each coefficient produced by the linear transfer encloses the exact transposed matrix
product; its constant contribution encloses the exact bias dot product. -/
theorem linear_encloses
    {m n : Nat} (aLo aHi : Tensor α [m]) (W : Tensor α [m, n]) (b : Tensor α [m])
    (a : Fin m → ℝ)
    (ha : ∀ i, value (aLo.getScalar i) ≤ a i ∧ a i ≤ value (aHi.getScalar i))
    {aX : FlatBox α} {cLo cHi : α}
    (hresult : directedBackwardLinear { dim := m, lo := aLo, hi := aHi } W b =
      some (aX, (cLo, cHi))) :
    aX.dim = n ∧
      (∀ j : Fin n,
        value (getAtOrZero aX.lo [j.val]) ≤ ∑ i, a i * value (Spec.get2 W i j) ∧
          (∑ i, a i * value (Spec.get2 W i j)) ≤ value (getAtOrZero aX.hi [j.val])) ∧
      value cLo ≤ ∑ i, a i * value (b.getScalar i) ∧
        (∑ i, a i * value (b.getScalar i)) ≤ value cHi := by
  have hread (i : Fin m) (j : Fin n) :
      getAtOrZero W [i.val, j.val] = Spec.get2 W i j := by
    simp [get_at_or_zero_dim_cons, i.isLt, j.isLt, Spec.get2, Tensor.getScalar, Spec.get]
  simp only [directedBackwardLinear, ↓reduceDIte, castDimScalar_self] at hresult
  obtain ⟨c, hc, hpair⟩ := Option.map_eq_some_iff.mp hresult
  cases hpair
  refine ⟨rfl, ?_, ?_⟩
  · intro j
    have hterm (i : Fin m) := intervalMul_encloses
      (ha i).1 (ha i).2
      (le_refl (value (Spec.get2 W i j))) (le_refl (value (Spec.get2 W i j)))
    have hsum := sum_encloses
      (fun i => (intervalMul (aLo.getScalar i) (aHi.getScalar i)
        (Spec.get2 W i j) (Spec.get2 W i j)).1)
      (fun i => (intervalMul (aLo.getScalar i) (aHi.getScalar i)
        (Spec.get2 W i j) (Spec.get2 W i j)).2)
      (fun i => a i * value (Spec.get2 W i j)) hterm
    have hfold := foldl_pair (List.finRange m)
      (fun acc i => BoundOps.addDown acc
        (intervalMul (aLo.getScalar i) (aHi.getScalar i)
          (Spec.get2 W i j) (Spec.get2 W i j)).1)
      (fun acc i => BoundOps.addUp acc
        (intervalMul (aLo.getScalar i) (aHi.getScalar i)
          (Spec.get2 W i j) (Spec.get2 W i j)).2) 0 0
    simpa only [read_fin, Tensor.getScalar_dim, hread, hfold] using hsum
  · exact dotBox_encloses aLo aHi b b a (fun i => value (b.getScalar i))
      ha (fun _ => ⟨le_rfl, le_rfl⟩) hc

/-- Interpret a scalar affine form using exact real addition and multiplication. -/
def affineValue {n : Nat} (aff : AffineVec α n 1) (x : Fin n → ℝ) : ℝ :=
  (∑ i, value (Spec.get2 aff.A 0 i) * x i) + value (aff.c.getScalar 0)

/-- The engine's final affine conversion encloses every objective represented by its input
coefficient intervals and accumulated constant. -/
theorem inputAffines_encloses
    {n : Nat} (xLo xHi aLo aHi : Tensor α [n]) (cLo cHi : α)
    (x a : Fin n → ℝ) (c : ℝ)
    (hx : ∀ i, value (xLo.getScalar i) ≤ x i ∧ x i ≤ value (xHi.getScalar i))
    (ha : ∀ i, value (aLo.getScalar i) ≤ a i ∧ a i ≤ value (aHi.getScalar i))
    (hc : value cLo ≤ c ∧ c ≤ value cHi)
    {lower upper : AffineVec α n 1}
    (hresult : directedInputAffines n
      { dim := n, lo := xLo, hi := xHi } { dim := n, lo := aLo, hi := aHi } cLo cHi =
        some (lower, upper)) :
    affineValue lower x ≤ (∑ i, a i * x i) + c ∧
      (∑ i, a i * x i) + c ≤ affineValue upper x := by
  let selected (i : Fin n) :=
    directedCoeffAffine (xLo.getScalar i) (xHi.getScalar i)
      (aLo.getScalar i) (aHi.getScalar i)
  have hselected (i : Fin n) :
      value (selected i).1.1 * x i + value (selected i).1.2 ≤ a i * x i ∧
        a i * x i ≤ value (selected i).2.1 * x i + value (selected i).2.2 :=
    coeffAffine_encloses (hx i).1 (hx i).2 (ha i).1 (ha i).2
  have hcorrection := sum_encloses
    (fun i => (selected i).1.2) (fun i => (selected i).1.2)
    (fun i => value (selected i).1.2) (fun _ => ⟨le_rfl, le_rfl⟩)
  have hcorrection' := sum_encloses
    (fun i => (selected i).2.2) (fun i => (selected i).2.2)
    (fun i => value (selected i).2.2) (fun _ => ⟨le_rfl, le_rfl⟩)
  have hlo := (LawfulBoundOps.addDown_le cLo _).trans (add_le_add hc.1 hcorrection.1)
  have hhi := (add_le_add hc.2 hcorrection'.2).trans (LawfulBoundOps.le_addUp cHi _)
  have hlower := Finset.sum_le_sum (fun i (_ : i ∈ Finset.univ) => (hselected i).1)
  have hupper := Finset.sum_le_sum (fun i (_ : i ∈ Finset.univ) => (hselected i).2)
  simp only [Finset.sum_add_distrib] at hlower hupper
  simp only [directedInputAffines, ↓reduceDIte, castDimScalar_self, read_fin,
    Option.some.injEq, Prod.mk.injEq] at hresult
  obtain ⟨rfl, rfl⟩ := hresult
  simp only [affineValue, Spec.get2_dim, Tensor.getScalar_dim]
  change
    (∑ i, value (selected i).1.1 * x i) +
        value (BoundOps.addDown cLo
          ((List.finRange n).foldl (fun acc i => BoundOps.addDown acc (selected i).1.2) 0)) ≤
      (∑ i, a i * x i) + c ∧
    (∑ i, a i * x i) + c ≤
      (∑ i, value (selected i).2.1 * x i) +
        value (BoundOps.addUp cHi
          ((List.finRange n).foldl (fun acc i => BoundOps.addUp acc (selected i).2.2) 0))
  constructor <;> linarith

end

end NN.MLTheory.CROWN.Graph.DirectedBackward

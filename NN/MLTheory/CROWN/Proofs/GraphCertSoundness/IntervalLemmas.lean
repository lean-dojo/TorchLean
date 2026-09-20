/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.BoundOps.Lawful
public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.Semantics

/-!
# Interval Soundness Lemmas

Scalar interval arithmetic, box-cast lemmas, and point-box facts used by the graph IBP soundness
induction.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN

namespace CertSoundness

noncomputable section

/-!
## Op-level soundness lemmas (enclosure for each supported step)

These lemmas are the building blocks for the final “certificate ⇒ semantics enclosure” theorem.

This proof reuses the following existing components:

* Linear IBP soundness over `ℝ` is already proved in `NN.MLTheory.CROWN.mlp` as
  `NN.MLTheory.CROWN.Theorems.ibp_linear_sound_real`.
* For add/sub/relu on `FlatBox`, the graph file already contains enclosure lemmas in
  `NN.MLTheory.CROWN.Graph.Theorems.Semantics`.
-/

/-- Monotonicity of real ReLU, used by interval enclosure proofs. -/
theorem relu_mono_real : ∀ {a b : ℝ}, a ≤ b →
    Activation.Math.reluSpec (α := ℝ) a ≤ Activation.Math.reluSpec (α := ℝ) b := by
  intro a b hab
  simpa only [Activation.Math.reluSpec_eq_max] using max_le_max hab (le_rfl : (0 : ℝ) ≤ 0)

/-- Addition is monotone in both operands. -/
theorem add_mono_real : ∀ {a b c d : ℝ}, a ≤ b → c ≤ d → a + c ≤ b + d := by
  intro a b c d hab hcd
  exact add_le_add hab hcd

/-- Subtraction is monotone in the minuend and antitone in the subtrahend. -/
theorem sub_mono_real : ∀ {a b c d : ℝ}, a ≤ b → d ≤ c → a - c ≤ b - d := by
  intro a b c d hab hdc
  have hneg : -c ≤ -d := neg_le_neg hdc
  have : a + (-c) ≤ b + (-d) := add_le_add hab hneg
  simpa [sub_eq_add_neg] using this

/-- The runtime `if a < b then a else b` is `min`. -/
theorem if_lt_eq_min (a b : ℝ) :
    (if a < b then a else b) = min a b := by
  by_cases h : a < b
  · simp [h, min_eq_left (le_of_lt h)]
  · have h' : b ≤ a := le_of_not_gt h
    simp [h, min_eq_right h']

/-- The runtime `if a > b then a else b` is `max`.

The implementations branch on a comparison rather than calling `min`/`max`, so these two lemmas are
what let the interval proofs use the Mathlib lattice lemmas at all. -/
theorem if_gt_eq_max (a b : ℝ) :
    (if a > b then a else b) = max a b := by
  by_cases h : a > b
  · simp [h, max_eq_left (le_of_lt h)]
  · have h' : a ≤ b := le_of_not_gt h
    simp [h, max_eq_right h']

/-- Multiplying an interval by a constant: the product lies between the two endpoint products.

Stated with `min`/`max` instead of a case split on the sign of `a`, so the caller never has to know
which endpoint is which. -/
theorem mul_const_bounds {a ly uy y : ℝ} (hy : ly ≤ y) (hy' : y ≤ uy) :
    min (a * ly) (a * uy) ≤ a * y ∧ a * y ≤ max (a * ly) (a * uy) := by
  by_cases ha : 0 ≤ a
  · have hlo : a * ly ≤ a * y := mul_le_mul_of_nonneg_left hy ha
    have hhi : a * y ≤ a * uy := mul_le_mul_of_nonneg_left hy' ha
    refine ⟨le_trans (min_le_left _ _) hlo, le_trans hhi (le_max_right _ _)⟩
  · have ha' : a ≤ 0 := le_of_not_ge ha
    have hlo : a * uy ≤ a * y := mul_le_mul_of_nonpos_left hy' ha'
    have hhi : a * y ≤ a * ly := mul_le_mul_of_nonpos_left hy ha'
    refine ⟨le_trans (min_le_right _ _) hlo, le_trans hhi (le_max_left _ _)⟩

/-- The same, with the interval on the left and the constant on the right. -/
theorem mul_var_bounds {lx ux x y : ℝ} (hx : lx ≤ x) (hx' : x ≤ ux) :
    min (lx * y) (ux * y) ≤ x * y ∧ x * y ≤ max (lx * y) (ux * y) := by
  by_cases hy : 0 ≤ y
  · have hlo : lx * y ≤ x * y := mul_le_mul_of_nonneg_right hx hy
    have hhi : x * y ≤ ux * y := mul_le_mul_of_nonneg_right hx' hy
    refine ⟨le_trans (min_le_left _ _) hlo, le_trans hhi (le_max_right _ _)⟩
  · have hy' : y ≤ 0 := le_of_not_ge hy
    have hlo : ux * y ≤ x * y := mul_le_mul_of_nonpos_right hx' hy'
    have hhi : x * y ≤ lx * y := mul_le_mul_of_nonpos_right hx hy'
    refine ⟨le_trans (min_le_right _ _) hlo, le_trans hhi (le_max_left _ _)⟩

/-- Interval multiplication: the product of two bounded values lies between the min and the max of
the four endpoint products.

This is the classical four-corner rule. All four products are needed because signs can differ, and
taking min and max of the corners is exactly tight for real multiplication. -/
theorem interval_mul_bounds
    {lx ux ly uy x y : ℝ} (hx : lx ≤ x) (hx' : x ≤ ux) (hy : ly ≤ y) (hy' : y ≤ uy) :
    min (min (lx * ly) (lx * uy)) (min (ux * ly) (ux * uy)) ≤ x * y ∧
      x * y ≤ max (max (lx * ly) (lx * uy)) (max (ux * ly) (ux * uy)) := by
  have h_lx : min (lx * ly) (lx * uy) ≤ lx * y ∧ lx * y ≤ max (lx * ly) (lx * uy) :=
    mul_const_bounds (a := lx) hy hy'
  have h_ux : min (ux * ly) (ux * uy) ≤ ux * y ∧ ux * y ≤ max (ux * ly) (ux * uy) :=
    mul_const_bounds (a := ux) hy hy'
  have h_x : min (lx * y) (ux * y) ≤ x * y ∧ x * y ≤ max (lx * y) (ux * y) :=
    mul_var_bounds (lx := lx) (ux := ux) (x := x) (y := y) hx hx'
  -- Lower bound: corners ≤ each endpoint product, hence ≤ min endpoint product, hence ≤ x*y.
  have hC_lx : min (min (lx * ly) (lx * uy)) (min (ux * ly) (ux * uy)) ≤ lx * y := by
    exact le_trans (min_le_left _ _) h_lx.1
  have hC_ux : min (min (lx * ly) (lx * uy)) (min (ux * ly) (ux * uy)) ≤ ux * y := by
    exact le_trans (min_le_right _ _) h_ux.1
  have hC_to_min : min (min (lx * ly) (lx * uy)) (min (ux * ly) (ux * uy)) ≤ min (lx * y) (ux * y)
    :=
    le_min hC_lx hC_ux
  have hlo : min (min (lx * ly) (lx * uy)) (min (ux * ly) (ux * uy)) ≤ x * y :=
    le_trans hC_to_min h_x.1
  -- Upper bound: x*y ≤ max endpoint product ≤ max corner maxes.
  let C : ℝ := max (max (lx * ly) (lx * uy)) (max (ux * ly) (ux * uy))
  have hmax_lx : lx * y ≤ C := le_trans h_lx.2 (le_max_left _ _)
  have hmax_ux : ux * y ≤ C := le_trans h_ux.2 (le_max_right _ _)
  have hmax_to_C : max (lx * y) (ux * y) ≤ C := max_le hmax_lx hmax_ux
  have hhi : x * y ≤ C := le_trans h_x.2 hmax_to_C
  simpa [C] using And.intro hlo hhi

/-! Helpers: our bound propagation uses `BoundOps.min2/max2`, which are defined via `decide (a >
  b)`.
For `ℝ` these coincide with `min/max`. -/

theorem min2_eq_min (a b : ℝ) : NN.MLTheory.CROWN.BoundOps.min2 a b = min a b := by
  by_cases h : a > b
  · have hab : b ≤ a := le_of_lt h
    simp [NN.MLTheory.CROWN.BoundOps.min2, h, min_eq_right hab]
  · have hab : a ≤ b := le_of_not_gt h
    simp [NN.MLTheory.CROWN.BoundOps.min2, h, min_eq_left hab]

/-- The bound-arithmetic `max2` is `max` over `ℝ`. -/
theorem max2_eq_max (a b : ℝ) : NN.MLTheory.CROWN.BoundOps.max2 a b = max a b := by
  by_cases h : a > b
  · have hab : b ≤ a := le_of_lt h
    simp [NN.MLTheory.CROWN.BoundOps.max2, h, max_eq_left hab]
  · have hab : a ≤ b := le_of_not_gt h
    simp [NN.MLTheory.CROWN.BoundOps.max2, h, max_eq_right hab]

/-- Elementwise interval multiplication of two boxes is sound.

Coordinatewise this is `interval_mul_bounds`; the box wrapper adds the dimension check, which is why
the conclusion is about whatever box `boxMulElem` actually returned. -/
theorem box_mulElem_sound_real (n : Nat)
    (lo1 hi1 lo2 hi2 x y : Tensor ℝ [n])
    (hx : encloses { dim := n, lo := lo1, hi := hi1 } x)
    (hy : encloses { dim := n, lo := lo2, hi := hi2 } y) :
    ∀ {B : FlatBox ℝ},
      boxMulElem (α := ℝ)
          { dim := n, lo := lo1, hi := hi1 }
          { dim := n, lo := lo2, hi := hi2 } = some B →
        EnclosesBox B ⟨n, Tensor.mulSpec (α := ℝ) x y⟩ := by
  classical
  intro B hB
  unfold boxMulElem at hB
  simp only [↓reduceDIte] at hB
  rw [← Option.some.inj hB]
  refine ⟨rfl, ?_⟩
  intro i
  have hMul :=
    interval_mul_bounds
      (lx := lo1.getScalar i) (ux := hi1.getScalar i)
      (ly := lo2.getScalar i) (uy := hi2.getScalar i)
      (x := x.getScalar i) (y := y.getScalar i)
      (hx := (hx i).1) (hx' := (hx i).2)
      (hy := (hy i).1) (hy' := (hy i).2)
  simpa [Tensor.mulSpec, min2_eq_min, max2_eq_max, BoundOps.mulDown, BoundOps.mulUp]
    using hMul

/-!
### Casting lemmas (avoid `cases` on `B.dim = v.n`)

`FlatBox` and `FlatTensor` carry their dimensions in dependent types, so it is tempting to
`cases` equalities like `h : B.dim = v.n` to “align” types. In Lean this can easily trigger
dependent elimination failures when the equality mentions fields of dependent records.

Instead, we keep such equalities as *data* and move tensors/boxes across them using
`castDimScalar` / `castBoxDim`. The following small lemmas are proved once (by `cases` on
*fresh* Nat equalities) and then used throughout the main proof without ever `cases`-ing on
`B.dim = v.n` directly.
 -/

theorem castDimScalar_trans {n n' n'' : Nat}
    (h₁ : n = n') (h₂ : n' = n'') (t : Tensor ℝ [n]) :
    castDimScalar (α := ℝ) (Eq.trans h₁ h₂) t
      = castDimScalar (α := ℝ) h₂ (castDimScalar (α := ℝ) h₁ t) := by
  cases h₁
  cases h₂
  rfl

/-- Dimension casts commute with elementwise maps. -/
theorem castDimScalar_map_spec {n n' : Nat}
    (h : n = n') (f : ℝ → ℝ) (t : Tensor ℝ [n]) :
    castDimScalar (α := ℝ) h (Tensor.mapSpec (α := ℝ) f t)
      = Tensor.mapSpec (α := ℝ) f (castDimScalar (α := ℝ) h t) := by
  cases h
  rfl

/-- Dimension casts commute with addition. -/
theorem castDimScalar_add_spec {n n' : Nat}
    (h : n = n') (x y : Tensor ℝ [n]) :
    castDimScalar (α := ℝ) h (Tensor.addSpec (α := ℝ) x y)
      = Tensor.addSpec (α := ℝ) (castDimScalar (α := ℝ) h x) (castDimScalar (α := ℝ) h y) := by
  cases h
  rfl

/-- Dimension casts commute with subtraction. -/
theorem castDimScalar_sub_spec {n n' : Nat}
    (h : n = n') (x y : Tensor ℝ [n]) :
    castDimScalar (α := ℝ) h (Tensor.subSpec (α := ℝ) x y)
      = Tensor.subSpec (α := ℝ) (castDimScalar (α := ℝ) h x) (castDimScalar (α := ℝ) h y) := by
  cases h
  rfl

/-- Dimension casts commute with elementwise multiplication. -/
theorem castDimScalar_mul_spec {n n' : Nat}
    (h : n = n') (x y : Tensor ℝ [n]) :
    castDimScalar (α := ℝ) h (Tensor.mulSpec (α := ℝ) x y)
      = Tensor.mulSpec (α := ℝ) (castDimScalar (α := ℝ) h x) (castDimScalar (α := ℝ) h y) := by
  cases h
  rfl

/-- Casting a box and a point along the same equality does not change containment. -/
theorem contains_castBoxDim_iff {n n' : Nat}
    (h : n = n') (B : Box ℝ (.dim n .scalar)) (x : Tensor ℝ [n]) :
    Box.contains (α := ℝ) (castBoxDim (α := ℝ) h B) (castDimScalar (α := ℝ) h x)
      ↔ Box.contains (α := ℝ) B x := by
  cases h
  simp [castBoxDim, castDimScalar]

/-- Enclosure survives a dimension cast, which is how a box proved for one layer width is reused at
the next one without ever eliminating the equality itself. -/
theorem encloses_castDim {B : FlatBox ℝ} {n' : Nat}
    (h : B.dim = n') (x : Tensor ℝ [B.dim]) :
    encloses B x →
      encloses { dim := n'
                 lo := castDimScalar (α := ℝ) h B.lo
                 hi := castDimScalar (α := ℝ) h B.hi }
        (castDimScalar (α := ℝ) h x) := by
  intro hx
  subst n'
  convert hx using 1 <;>
    simp only [NN.MLTheory.CROWN.Graph.castDimScalar_self]

/-- `Box.contains` implies `encloses` on the flattened box; the two are definitionally the same, and
the lemma exists so proofs can change vocabulary without unfolding. -/
theorem encloses_of_contains {n : Nat}
    (B : Box ℝ (.dim n .scalar)) (x : Tensor ℝ [n]) :
    Box.contains (α := ℝ) B x → encloses (toFlatBox (α := ℝ) n B) x := by
  exact fun hx => hx

/-- The converse direction, for the same reason. -/
theorem contains_of_encloses
    (B : FlatBox ℝ) (x : Tensor ℝ [B.dim]) :
    encloses B x → Box.contains (α := ℝ) (ofFlatBox (α := ℝ) B) x := by
  exact fun hx => hx

/-!
### Point Boxes Always Enclose Their Point

This is used in the `.const` case, where a constant node certifies a point box
`[v,v]` and the semantics returns exactly the same `v`.
-/

theorem encloses_point_self_real {n : Nat} (x : Tensor ℝ [n]) :
    NN.MLTheory.CROWN.Graph.Theorems.Semantics.encloses (α := ℝ) { dim := n, lo := x, hi := x } x :=
      fun _ => ⟨le_rfl, le_rfl⟩

end

end CertSoundness

end NN.MLTheory.CROWN.Graph

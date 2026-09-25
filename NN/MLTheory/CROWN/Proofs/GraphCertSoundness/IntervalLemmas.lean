/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Floats.Interval.RealBounds
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

/-- Interval multiplication: the product of two bounded values lies between the min and the max of
the four endpoint products. This is FloatLib's `mul_bounds_Icc` with the bounds split out. -/
theorem interval_mul_bounds
    {lx ux ly uy x y : ℝ} (hx : lx ≤ x) (hx' : x ≤ ux) (hy : ly ≤ y) (hy' : y ≤ uy) :
    min (min (lx * ly) (lx * uy)) (min (ux * ly) (ux * uy)) ≤ x * y ∧
      x * y ≤ max (max (lx * ly) (lx * uy)) (max (ux * ly) (ux * uy)) :=
  FloatLib.Floats.Interval.mul_bounds_Icc lx ux ly uy x y ⟨hx, hx'⟩ ⟨hy, hy'⟩

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

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

/-- Elementwise interval multiplication of two boxes is sound.

FloatLib bounds each coordinate by the four endpoint products. The box wrapper also checks the
dimensions, so the conclusion concerns the box actually returned by `boxMulElem`. -/
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
    FloatLib.Floats.Interval.mul_bounds_Icc
      (lo1.getScalar i) (hi1.getScalar i) (lo2.getScalar i) (hi2.getScalar i)
      (x.getScalar i) (y.getScalar i) (hx i) (hy i)
  simpa [Tensor.mulSpec, min2_eq_min, max2_eq_max, BoundOps.mulDown, BoundOps.mulUp,
    FloatLib.Floats.Interval.minOfFour, FloatLib.Floats.Interval.maxOfFour]
    using hMul

/-!
### Casting and containment

Boxes and values carry their dimensions in dependent types. Transporting both along the same
equality preserves containment; the box conversion lemmas let operator proofs reuse `Box` bounds
without unfolding the flattened representation.
-/

/-- Casting a box and a point along the same equality does not change containment. -/
theorem contains_castBoxDim_iff {n n' : Nat}
    (h : n = n') (B : Box ℝ (.dim n .scalar)) (x : Tensor ℝ [n]) :
    Box.contains (α := ℝ) (castBoxDim (α := ℝ) h B) (castDimScalar (α := ℝ) h x)
      ↔ Box.contains (α := ℝ) B x := by
  cases h
  simp [castBoxDim, castDimScalar]

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

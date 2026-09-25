/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import NN.MLTheory.CROWN.Extras.FP32
import NN.MLTheory.CROWN.Proofs.DirectedIBPFullSoundness

/-!
# Absolute-value endpoints and seeded random equations

The FP32 proof carrier permits stored real values outside its rounding grid. This regression
checks that the directed abs transfer preserves the positive lower endpoint of a `1/3` singleton
and encloses its exact absolute value. Random equations accept the seeded source values, reject
a fractional mask at probability zero, and require the parent structure checked by IR evaluation.
-/

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean TorchLean.Tensor NN.IR

noncomputable section

private def thirdBox : FlatBox FP32 :=
  { dim := 1
    lo := Tensor.full [1] ⟨(1 : ℝ) / 3⟩
    hi := Tensor.full [1] ⟨(1 : ℝ) / 3⟩ }

/-- A positive FP32 comparison endpoint is preserved even when it is not a grid value. -/
theorem boxAbs_fp32_third_lower :
    LawfulBoundOps.toReal
      ((boxAbs thirdBox).lo.getScalar ⟨0, by simp [boxAbs, thirdBox]⟩) = (1 : ℝ) / 3 := by
  have hnonneg : ¬ (⟨(1 : ℝ) / 3⟩ : FP32) < 0 := by
    rw [LawfulBoundOps.lt_iff, (LawfulBoundOps.toReal_zero (α := FP32))]
    change ¬ (1 : ℝ) / 3 < 0
    norm_num
  simp only [thirdBox, boxAbs, getScalar_ofFn, getScalar_full, hnonneg, ↓reduceIte]
  rfl

/-- The actual FP32 abs box encloses the exact value of the singleton. -/
theorem boxAbs_fp32_third_encloses :
    RowEncloses (boxAbs thirdBox) 1 (fun _ => (1 : ℝ) / 3) := by
  have hinput : RowEncloses thirdBox 1 (fun _ => (1 : ℝ) / 3) := by
    rw [thirdBox, rowEncloses_iff]
    intro i
    simp only [getScalar_full]
    exact ⟨le_rfl, le_rfl⟩
  have h := boxAbs_encloses hinput
  simpa only [abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 1 / 3)] using h

namespace SeededRandomRegression

private def maskNodes (inputShape outShape : Shape) (parents : Array Nat := #[0]) : Array Node :=
  #[{ id := 0, parents := #[], kind := .input, outShape := inputShape },
    { id := 1, parents := parents, kind := .bernoulliMask 17, outShape := outShape }]

private def halfAtZero (id : Nat) (_coordinate : Nat) : ℝ :=
  if id = 0 then 0 else 1 / 2

/-- Unit-interval support alone cannot satisfy the seeded mask equation at probability zero. -/
theorem mask_rejects_half_at_zero :
    ¬ PointwiseRealNodeEquation (maskNodes .scalar .scalar) (fun _ => 1) halfAtZero 1 := by
  norm_num [PointwiseRealNodeEquation, maskNodes, halfAtZero, unaryParent?, Shape.size]

/-- The complete real-node dispatcher also excludes the formerly admitted fractional mask. -/
theorem full_equation_rejects_half_at_zero :
    ¬ RealNodeEquation (maskNodes .scalar .scalar) ({} : ParamStore FP32) #[]
      (fun _ => 1) halfAtZero 1 := by
  simpa [RealNodeEquation, maskNodes] using mask_rejects_half_at_zero

/-- The source's zero mask satisfies the equation for every output shape, including empty ones. -/
theorem zero_mask_equation (s : Shape) :
    PointwiseRealNodeEquation (maskNodes .scalar s)
      (fun id => if id = 0 then 1 else s.size) (fun _ _ => 0) 1 := by
  simp [PointwiseRealNodeEquation, maskNodes, unaryParent?]

/-- The source's unit mask satisfies the equation for every output shape. -/
theorem one_mask_equation (s : Shape) :
    PointwiseRealNodeEquation (maskNodes .scalar s)
      (fun id => if id = 0 then 1 else s.size) (fun _ _ => 1) 1 := by
  simp [PointwiseRealNodeEquation, maskNodes, unaryParent?]

/-- A missing keep-probability parent cannot make the random equation vacuously true. -/
theorem mask_rejects_missing_parent (s : Shape) (dims : Nat → Nat) (v : Nat → Nat → ℝ) :
    ¬ PointwiseRealNodeEquation (maskNodes .scalar s #[]) dims v 1 := by
  simp [PointwiseRealNodeEquation, maskNodes, unaryParent?]

/-- A parent index outside the node array does not denote a source keep probability. -/
theorem mask_rejects_invalid_parent (s : Shape) (dims : Nat → Nat) (v : Nat → Nat → ℝ) :
    ¬ PointwiseRealNodeEquation (maskNodes .scalar s #[2]) dims v 1 := by
  simp [PointwiseRealNodeEquation, maskNodes, unaryParent?]

/-- A one-element vector is not the rank-zero parent required by the source mask operation. -/
theorem mask_rejects_vector_parent (s : Shape) (dims : Nat → Nat) (v : Nat → Nat → ℝ) :
    ¬ PointwiseRealNodeEquation (maskNodes [1] s) dims v 1 := by
  simp [PointwiseRealNodeEquation, maskNodes, unaryParent?]

private def uniformValues (seed : Nat) (s : Shape) (_id coordinate : Nat) : ℝ :=
  if h : coordinate < s.size then
    Spec.Random.uniform (α := ℝ) (Spec.Random.keyOf seed 0)
      (Shape.Coord.unlinearize ⟨coordinate, h⟩)
  else 0

/-- Every seeded uniform tensor satisfies the actual equation at all of its coordinates. -/
theorem uniform_equation (seed : Nat) (s : Shape) :
    PointwiseRealNodeEquation
      #[{ id := 0, parents := #[], kind := .randUniform seed, outShape := s }]
      (fun _ => s.size) (uniformValues seed s) 0 := by
  refine ⟨rfl, rfl, ?_⟩
  intro i
  have hi : i.val < s.size := i.isLt
  simp only [uniformValues, dite_eq_left hi]
  rfl

/-- Uniform generation requires the empty parent list checked by IR evaluation. -/
theorem uniform_rejects_parent (seed : Nat) (s : Shape) (dims : Nat → Nat)
    (v : Nat → Nat → ℝ) :
    ¬ PointwiseRealNodeEquation
      #[{ id := 0, parents := #[0], kind := .randUniform seed, outShape := s }] dims v 0 := by
  simp [PointwiseRealNodeEquation]

end SeededRandomRegression

end

end NN.MLTheory.CROWN.Graph.DirectedBackward

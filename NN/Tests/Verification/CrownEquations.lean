/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import NN.MLTheory.CROWN.Extras.FP32
import NN.MLTheory.CROWN.Proofs.DirectedIBPFullSoundness

/-!
# CROWN equation checks

These compile-time checks distinguish the equations used by the soundness theorems from
weaker, vacuous predicates: seeded random values must match their source, convolution must retain
off-grid coefficients, and LayerNorm needs a nonnegative stabilizer. They do not run in the suite.
-/

namespace NN.Tests.Verification.Crown.Equations

namespace Pointwise

open Spec TorchLean TorchLean.Tensor NN.IR
open NN.MLTheory.CROWN NN.MLTheory.CROWN.Graph NN.MLTheory.CROWN.Graph.DirectedBackward

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

namespace SeededRandomRegression

private def maskNodes (inputShape outShape : Shape) (parents : Array Nat := #[0]) :
    Array NN.IR.Node :=
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

end Pointwise

namespace Convolution

open Spec TorchLean TorchLean.Tensor
open NN.MLTheory.CROWN

noncomputable section

private def oneExtent : Tensor Nat [1] := Tensor.full [1] 1
private def zeroExtent : Tensor Nat [1] := Tensor.full [1] 0

private def thirdLayer : ConvSpec 1 1 1 oneExtent oneExtent zeroExtent FP32 :=
  { kernel := Tensor.full _ ⟨(1 : ℝ) / 3⟩
    bias := Tensor.full [1] 0 }

/-- The actual FP32 affine converter preserves an off-grid singleton weight exactly. -/
theorem convLinearMatrix_fp32_third :
    LawfulBoundOps.toReal
      (getAtOrZero
        (convLinearMatrix (inSpatial := oneExtent) thirdLayer oneExtent zeroExtent 1 .scalar)
        [0, 0]) = (1 : ℝ) / 3 := by
  rfl

/-- Two real kernel coordinates that coincide under zero dilation still contribute twice. -/
theorem convKernelCoefficient_zero_dilation :
    convKernelCoefficient [2] [0] [0] [1] [0] [0] (fun _ => (1 : ℝ) / 3) = 2 / 3 := by
  norm_num [convKernelCoefficient, Spec.Conv.Internal.foldlIndices, List.finRange_succ,
    Spec.Conv.Internal.mkDilatedInputIdx?]

end

end Convolution

namespace Normalization

open Spec TorchLean TorchLean.Tensor
open NN.MLTheory.CROWN
open _root_.Proofs.Autograd.Norm

/-- The exact-real default normalization stabilizer is nonnegative. -/
example :
    0 ≤ LawfulBoundOps.toReal (TorchLean.normalizationEpsilon : ℝ) := by
  change (0 : ℝ) ≤ ((1 / 100000 : ℚ) : ℝ)
  norm_num

example : 0 ≤ LawfulBoundOps.toReal (TorchLean.normalizationEpsilon : FP32) :=
  FP32.normalizationEpsilon_nonneg

/-- A negative stabilizer invalidates the square-root-width bound for the actual real Spec. -/
theorem negative_epsilon_exceeds_uniform_layerNorm_bound :
    let x : Tensor ℝ [1, 2] :=
      Tensor.dim fun _ : Fin 1 => Tensor.ofFn fun j : Fin 2 => if j = 0 then -1 else 1
    Real.sqrt 2 <
      |Spec.get2 (Spec.layerNorm x (Tensor.full [2] 1) (Tensor.full [2] 0)
        (by decide) (by decide) (-3 / 4)) 0 1| := by
  intro x
  have hx (j : Fin 2) : Spec.get2 x 0 j = if j = 0 then -1 else 1 := by
    simp only [x, Spec.get2_eq_apply, Tensor.dim, TorchLean.Tensor.Internal.Rep.stack_apply,
      Tensor.ofFn_apply]
  have hm : rowMeanE x 0 = 0 := by
    norm_num [rowMeanE, Fin.sum_univ_two, hx]
  have hv : rowVarE x 0 = 1 := by
    norm_num [rowVarE, Fin.sum_univ_two, hm, hx]
  rw [get2_layerNorm]
  simp only [hx, hm, hv, Tensor.getScalar_full]
  have hs : Real.sqrt ((1 : ℝ) / 4) = 1 / 2 := by
    rw [show (1 : ℝ) / 4 = (1 / 2) ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]
  norm_num [hs]

end Normalization

end NN.Tests.Verification.Crown.Equations

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Dual.Jet
public import TorchLean.NN.FDeriv.TensorCoordinates

/-!
# Higher derivatives of tensor-valued functions

Tensor coordinates are continuous linear functionals, so they commute with derivatives of every
order. Applied coordinatewise, the scalar dual-number rules then certify tensor computations for
the Euclidean tensor norm, with no restriction on tensor rank or on the differentiation
directions.
-/

@[expose] public section

namespace TorchLean.DualTensor

open Tensor Rep

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- All mixed derivative coefficients of a tensor-valued function along a direction tuple. -/
noncomputable def jet {shape : Shape} {n : Nat} (directions : Fin n → E)
    (f : E → Tensor ℝ shape) (x : E) : Tensor (Dual.Nested ℝ n) shape :=
  Tensor.Rep.ofFn fun i => Dual.jet directions (fun y => f y i) x

/-- Each tensor coordinate carries the corresponding scalar jet. -/
@[simp] theorem jet_apply {shape : Shape} {n : Nat} (directions : Fin n → E)
    (f : E → Tensor ℝ shape) (x : E) (i : shape.Coord) :
    jet directions f x i = Dual.jet directions (fun y => f y i) x :=
  Tensor.Rep.get_ofFn _ i

/-- The jet of a continuous linear tensor map `f` is the seeded tensor with value `f x` and
directions `f (directions k)`. -/
theorem jet_linear {shape : Shape} {n : Nat} (directions : Fin n → E)
    (f : E →L[ℝ] Tensor ℝ shape) (x : E) :
    jet directions f x = Dual.Nested.seedTensor (fun k => f (directions k)) (f x) := by
  apply Tensor.Rep.ext
  intro i
  simp only [jet_apply, Dual.Nested.seedTensor, Tensor.Rep.get_ofFn]
  exact Dual.jet_linear directions ((coordinateCLM i).comp f) x

/-- Runtime seeding is the complete jet of the tensor input, including zero mixed coefficients. -/
theorem jet_id {shape : Shape} {n : Nat} (directions : Fin n → Tensor ℝ shape)
    (input : Tensor ℝ shape) :
    jet directions (fun x => x) input = Dual.Nested.seedTensor directions input :=
  jet_linear directions (ContinuousLinearMap.id ℝ _) input

/-- Each input coordinate `i` has as its jet the scalar seed with value `input i` and directions
`directions k i`. -/
theorem jet_coordinate {shape : Shape} {n : Nat} (directions : Fin n → Tensor ℝ shape)
    (input : Tensor ℝ shape) (i : shape.Coord) :
    Dual.jet directions (fun x => x i) input =
      Dual.Nested.seed (fun k => directions k i) (input i) :=
  Dual.jet_linear directions (coordinateCLM i) input

/-- A tensor held constant has zero derivative coefficients in every direction. -/
theorem jet_const {shape : Shape} {n : Nat} (directions : Fin n → E)
    (value : Tensor ℝ shape) (x : E) :
    jet directions (fun _ => value) x = Tensor.map (Dual.Nested.ofPrimal n) value := by
  apply Tensor.Rep.ext
  intro i
  simp only [jet_apply, Dual.jet_const, Tensor.map_apply]

/-- The jet of the zero tensor is zero at every derivative order. -/
theorem jet_full_zero {shape : Shape} {n : Nat} (directions : Fin n → E) (x : E) :
    jet directions (fun _ => Tensor.full shape (0 : ℝ)) x =
      Tensor.full shape (0 : Dual.Nested ℝ n) := by
  apply Tensor.Rep.ext
  intro i
  simp only [jet_apply, Tensor.full_apply, Dual.jet_const, Dual.Nested.ofPrimal_zero]

/-- If `f` and `g` are `C^n` at `x`, the jet of their pointwise product is the product of their
jets. -/
theorem jet_mul {shape : Shape} {n : Nat} (directions : Fin n → E)
    {f g : E → Tensor ℝ shape} {x : E} (hf : ContDiffAt ℝ n f x) (hg : ContDiffAt ℝ n g x) :
    jet directions (fun y => f y * g y) x =
      jet directions f x * jet directions g x := by
  apply Tensor.Rep.ext
  intro i
  simp only [jet_apply, Tensor.mul_def, Tensor.zipWith_apply]
  exact Dual.jet_mul directions (contDiffAt_coordinate hf i) (contDiffAt_coordinate hg i)

/-- Flat storage indices read the same coefficients as tensor coordinates. -/
theorem jet_getFlat {shape : Shape} {n : Nat} (directions : Fin n → E)
    (f : E → Tensor ℝ shape) (x : E) (i : Fin (Shape.size shape)) :
    (jet directions f x).getFlat i = Dual.jet directions (fun y => (f y).getFlat i) x := by
  simpa only [Tensor.Rep.get, Shape.Coord.linearize_unlinearize] using
    jet_apply directions f x (Shape.Coord.unlinearize i)

/-- If `f` is `C^n` at `x`, the runtime sum of its tensor jet is the jet of the entry sum of `f`,
for empty tensors too. -/
theorem jet_sum_comp {shape : Shape} {n : Nat} (directions : Fin n → E)
    {f : E → Tensor ℝ shape} {x : E} (hf : ContDiffAt ℝ n f x) :
    Dual.jet directions (fun y => Tensor.sum (f y)) x =
      Tensor.sum (jet directions f x) := by
  simp only [Tensor.sum_eq_foldl, Tensor.Rep.foldl_eq_fin_foldl,
    Fin.foldl_eq_foldl_finRange]
  rw [Dual.jet_foldl_add _ directions contDiffAt_const (fun i _ => contDiffAt_getFlat hf i)]
  simp only [Dual.jet_const, Dual.Nested.ofPrimal_zero, jet_getFlat]

/-- Summing a seeded tensor computes the complete jet of the sum function. -/
theorem jet_sum {shape : Shape} {n : Nat} (directions : Fin n → Tensor ℝ shape)
    (x : Tensor ℝ shape) :
    Dual.jet directions (fun y : Tensor ℝ shape => Tensor.sum y) x =
      Tensor.sum (Dual.Nested.seedTensor directions x) := by
  simpa only [jet_id] using
    jet_sum_comp directions (f := fun y : Tensor ℝ shape => y) (x := x) contDiffAt_id

/-- If `f` and `g` are `C^n` at `x`, the dot product of their tensor jets is the jet of their dot
product. -/
theorem jet_dot {shape : Shape} {n : Nat} (directions : Fin n → E)
    {f g : E → Tensor ℝ shape} {x : E} (hf : ContDiffAt ℝ n f x) (hg : ContDiffAt ℝ n g x) :
    Dual.jet directions (fun y => Tensor.dot (f y) (g y)) x =
      Tensor.dot (jet directions f x) (jet directions g x) := by
  have hmul : ContDiffAt ℝ n (fun y => f y * g y) x :=
    contDiffAt_zipWith (fun _ => by fun_prop) hf hg
  simp only [Tensor.dot, jet_sum_comp directions hmul, jet_mul directions hf hg]

/-- If `f` and `g` are `C^n` at `x`, the matrix product of their tensor jets is the jet of their
matrix product. `inner` may be zero, and no entry needs to be positive or nonzero. -/
theorem jet_matmul {n rows inner cols : Nat} (directions : Fin n → E)
    {f : E → Tensor ℝ [rows, inner]} {g : E → Tensor ℝ [inner, cols]}
    {x : E} (hf : ContDiffAt ℝ n f x) (hg : ContDiffAt ℝ n g x) :
    jet directions (fun y => Tensor.matmul (f y) (g y)) x =
      Tensor.matmul (jet directions f x) (jet directions g x) := by
  apply Tensor.Rep.ext
  intro i
  simp only [jet_apply, Tensor.matmul_eq_ofFn, Tensor.Rep.get_ofFn, get2_eq_apply]
  simp (disch := first | fun_prop | (intro k hk; fun_prop)) only
    [Dual.jet_foldl_add, Dual.jet_mul, Dual.jet_const, Dual.Nested.ofPrimal_zero]

/-- Under local smoothness, extracting every tangent gives the tensor's iterated Fréchet
derivative evaluated on the supplied directions, not merely separate coordinate derivatives. -/
theorem tangent_jet {shape : Shape} {n : Nat} (directions : Fin n → E)
    {f : E → Tensor ℝ shape} {x : E} (hf : ContDiffAt ℝ n f x) :
    Dual.Nested.tangentTensor (jet directions f x) =
      iteratedFDeriv ℝ n f x directions := by
  apply Tensor.Rep.ext
  intro i
  simp only [Dual.Nested.tangentTensor, Tensor.map_apply]
  rw [jet_apply, Dual.tangent_jet directions (contDiffAt_coordinate hf i)]
  simpa only [Function.comp_def, coordinateCLM_apply,
    ContinuousLinearMap.compContinuousMultilinearMap_coe] using
      congrArg (fun derivative => derivative directions)
        ((coordinateCLM i).iteratedFDeriv_comp_left hf le_rfl)

end TorchLean.DualTensor

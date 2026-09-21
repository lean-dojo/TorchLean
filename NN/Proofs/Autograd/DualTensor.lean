/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Dual
public import NN.Proofs.Autograd.FDeriv.TensorCoordinates

/-!
# Higher derivatives of tensor-valued functions

Tensor coordinates are continuous linear functionals, so they commute with every derivative
order. This lets the scalar dual-number rules certify tensor computations in the existing
Euclidean tensor norm, with no restriction on tensor rank or differentiation directions.
-/

@[expose] public section

namespace Runtime.Autograd.Model.DualTensor

open Spec TorchLean Proofs.Autograd.TensorCoordinates

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- All mixed derivative coefficients of a tensor-valued function along a direction tuple. -/
noncomputable def jet {shape : Shape} {n : Nat} (directions : Fin n → E)
    (f : E → Tensor ℝ shape) (x : E) : Tensor (Dual.Nested ℝ n) shape :=
  Tensor.Internal.Rep.ofFn fun i => Dual.jet directions (fun y => f y i) x

/-- Each tensor coordinate carries the corresponding scalar jet. -/
@[simp] theorem jet_apply {shape : Shape} {n : Nat} (directions : Fin n → E)
    (f : E → Tensor ℝ shape) (x : E) (i : shape.Coord) :
    jet directions f x i = Dual.jet directions (fun y => f y i) x :=
  Tensor.Internal.Rep.get_ofFn _ i

/-- A continuous linear tensor map carries each direction through unchanged by higher terms. -/
theorem jet_linear {shape : Shape} {n : Nat} (directions : Fin n → E)
    (f : E →L[ℝ] Tensor ℝ shape) (x : E) :
    jet directions f x = Dual.Nested.seedTensor (fun k => f (directions k)) (f x) := by
  apply Tensor.Internal.Rep.ext
  intro i
  simp only [jet_apply, Dual.Nested.seedTensor, Tensor.Internal.Rep.get_ofFn]
  exact Dual.jet_linear directions ((coordinateCLM i).comp f) x

/-- Runtime seeding is the complete jet of the tensor input, including zero mixed coefficients. -/
theorem jet_id {shape : Shape} {n : Nat} (directions : Fin n → Tensor ℝ shape)
    (input : Tensor ℝ shape) :
    jet directions (fun x => x) input = Dual.Nested.seedTensor directions input :=
  jet_linear directions (ContinuousLinearMap.id ℝ _) input

/-- A tensor input coordinate has precisely the value and directions supplied to the runtime. -/
theorem jet_coordinate {shape : Shape} {n : Nat} (directions : Fin n → Tensor ℝ shape)
    (input : Tensor ℝ shape) (i : shape.Coord) :
    Dual.jet directions (fun x => x i) input =
      Dual.Nested.seed (fun k => directions k i) (input i) :=
  Dual.jet_linear directions (coordinateCLM i) input

/-- A tensor held constant has zero derivative coefficients in every direction. -/
theorem jet_const {shape : Shape} {n : Nat} (directions : Fin n → E)
    (value : Tensor ℝ shape) (x : E) :
    jet directions (fun _ => value) x = Tensor.map (Dual.Nested.ofPrimal n) value := by
  apply Tensor.Internal.Rep.ext
  intro i
  simp only [jet_apply, Dual.jet_const, Tensor.map, Tensor.Internal.Rep.map_apply]

/-- Pointwise multiplication of complete jets follows the scalar product rule at every order. -/
theorem jet_mul_at {shape : Shape} {n : Nat} (directions : Fin n → E)
    {f g : E → Tensor ℝ shape} {x : E} (hf : ContDiffAt ℝ n f x) (hg : ContDiffAt ℝ n g x) :
    jet directions (fun y => Tensor.mulSpec (f y) (g y)) x =
      Tensor.mulSpec (jet directions f x) (jet directions g x) := by
  apply Tensor.Internal.Rep.ext
  intro i
  simp only [jet_apply, Tensor.mulSpec, Tensor.map2Spec_apply]
  exact Dual.jet_mul_at directions (contDiffAt_coordinate hf i) (contDiffAt_coordinate hg i)

@[inherit_doc jet_mul_at]
theorem jet_mul {shape : Shape} {n : Nat} (directions : Fin n → E)
    {f g : E → Tensor ℝ shape} (hf : ContDiff ℝ n f) (hg : ContDiff ℝ n g) (x : E) :
    jet directions (fun y => Tensor.mulSpec (f y) (g y)) x =
      Tensor.mulSpec (jet directions f x) (jet directions g x) :=
  jet_mul_at directions hf.contDiffAt hg.contDiffAt

/-- Flat storage indices read the same coefficients as tensor coordinates. -/
theorem jet_getFlat {shape : Shape} {n : Nat} (directions : Fin n → E)
    (f : E → Tensor ℝ shape) (x : E) (i : Fin (Tensor.Internal.Shape.size shape)) :
    (jet directions f x).getFlat i = Dual.jet directions (fun y => (f y).getFlat i) x := by
  simpa only [Tensor.Internal.Rep.get, Tensor.Internal.Coord.linearize_unlinearize] using
    jet_apply directions f x (Tensor.Internal.Coord.unlinearize i)

/-- The runtime sum preserves every derivative coefficient, including for an empty tensor. -/
theorem jet_sumSpec_comp_at {shape : Shape} {n : Nat} (directions : Fin n → E)
    {f : E → Tensor ℝ shape} {x : E} (hf : ContDiffAt ℝ n f x) :
    Dual.jet directions (fun y => Tensor.sumSpec (f y)) x =
      Tensor.sumSpec (jet directions f x) := by
  simp only [Tensor.sumSpec, Tensor.foldlSpec, Tensor.Internal.Rep.foldl_eq_fin_foldl,
    Fin.foldl_eq_foldl_finRange]
  rw [Dual.jet_foldl_add_at _ directions contDiffAt_const (fun i _ => contDiffAt_getFlat hf i)]
  simp only [Dual.jet_const, Dual.Nested.ofPrimal_zero, jet_getFlat]

@[inherit_doc jet_sumSpec_comp_at]
theorem jet_sumSpec_comp {shape : Shape} {n : Nat} (directions : Fin n → E)
    {f : E → Tensor ℝ shape} (hf : ContDiff ℝ n f) (x : E) :
    Dual.jet directions (fun y => Tensor.sumSpec (f y)) x =
      Tensor.sumSpec (jet directions f x) :=
  jet_sumSpec_comp_at directions hf.contDiffAt

/-- Summing a seeded tensor computes the complete jet of the sum function. -/
theorem jet_sumSpec {shape : Shape} {n : Nat} (directions : Fin n → Tensor ℝ shape)
    (x : Tensor ℝ shape) :
    Dual.jet directions (fun y : Tensor ℝ shape => Tensor.sumSpec y) x =
      Tensor.sumSpec (Dual.Nested.seedTensor directions x) := by
  simpa only [jet_id] using
    jet_sumSpec_comp directions (f := fun y : Tensor ℝ shape => y) contDiff_id x

/-- Dot products preserve jets when both operands depend on the input. -/
theorem jet_dotSpec_at {shape : Shape} {n : Nat} (directions : Fin n → E)
    {f g : E → Tensor ℝ shape} {x : E} (hf : ContDiffAt ℝ n f x) (hg : ContDiffAt ℝ n g x) :
    Dual.jet directions (fun y => Tensor.dotSpec (f y) (g y)) x =
      Tensor.dotSpec (jet directions f x) (jet directions g x) := by
  have hmul : ContDiffAt ℝ n (fun y => Tensor.mulSpec (f y) (g y)) x :=
    contDiffAt_map2 (fun _ => by fun_prop) hf hg
  simp only [Tensor.dotSpec, jet_sumSpec_comp_at directions hmul, jet_mul_at directions hf hg]

@[inherit_doc jet_dotSpec_at]
theorem jet_dotSpec {shape : Shape} {n : Nat} (directions : Fin n → E)
    {f g : E → Tensor ℝ shape} (hf : ContDiff ℝ n f) (hg : ContDiff ℝ n g) (x : E) :
    Dual.jet directions (fun y => Tensor.dotSpec (f y) (g y)) x =
      Tensor.dotSpec (jet directions f x) (jet directions g x) :=
  jet_dotSpec_at directions hf.contDiffAt hg.contDiffAt

/-- Matrix contraction propagates all mixed derivatives of both operands.
The shared dimension can be empty; no positivity or nonzero-entry assumptions are needed. -/
theorem jet_matMulSpec_at {n rows inner cols : Nat} (directions : Fin n → E)
    {f : E → Tensor ℝ [rows, inner]} {g : E → Tensor ℝ [inner, cols]}
    {x : E} (hf : ContDiffAt ℝ n f x) (hg : ContDiffAt ℝ n g x) :
    jet directions (fun y => matMulSpec (f y) (g y)) x =
      matMulSpec (jet directions f x) (jet directions g x) := by
  apply Tensor.Internal.Rep.ext
  intro i
  simp only [jet_apply, matMulSpec, Tensor.Internal.Rep.get_ofFn, get2_eq_apply]
  simp (disch := first | fun_prop | (intro k hk; fun_prop)) only
    [Dual.jet_foldl_add_at, Dual.jet_mul_at, Dual.jet_const, Dual.Nested.ofPrimal_zero]

@[inherit_doc jet_matMulSpec_at]
theorem jet_matMulSpec {n rows inner cols : Nat} (directions : Fin n → E)
    {f : E → Tensor ℝ [rows, inner]} {g : E → Tensor ℝ [inner, cols]}
    (hf : ContDiff ℝ n f) (hg : ContDiff ℝ n g) (x : E) :
    jet directions (fun y => matMulSpec (f y) (g y)) x =
      matMulSpec (jet directions f x) (jet directions g x) :=
  jet_matMulSpec_at directions hf.contDiffAt hg.contDiffAt

/-- Local smoothness suffices to identify the extracted tensor with its iterated derivative. -/
theorem tangent_jet_at {shape : Shape} {n : Nat} (directions : Fin n → E)
    {f : E → Tensor ℝ shape} {x : E} (hf : ContDiffAt ℝ n f x) :
    Dual.Nested.tangentTensor (jet directions f x) =
      iteratedFDeriv ℝ n f x directions := by
  apply Tensor.Internal.Rep.ext
  intro i
  simp only [Dual.Nested.tangentTensor, Tensor.map, Tensor.Internal.Rep.map_apply]
  rw [jet_apply, Dual.tangent_jet_at directions (contDiffAt_coordinate hf i)]
  simpa only [Function.comp_def, coordinateCLM_apply,
    ContinuousLinearMap.compContinuousMultilinearMap_coe] using
      congrArg (fun derivative => derivative directions)
        ((coordinateCLM i).iteratedFDeriv_comp_left hf le_rfl)

/-- Extracting every tangent gives the tensor's iterated Fréchet derivative, not merely
separate coordinate derivatives. -/
theorem tangent_jet {shape : Shape} {n : Nat} (directions : Fin n → E)
    {f : E → Tensor ℝ shape} (hf : ContDiff ℝ n f) (x : E) :
    Dual.Nested.tangentTensor (jet directions f x) =
      iteratedFDeriv ℝ n f x directions :=
  tangent_jet_at directions hf.contDiffAt

end Runtime.Autograd.Model.DualTensor

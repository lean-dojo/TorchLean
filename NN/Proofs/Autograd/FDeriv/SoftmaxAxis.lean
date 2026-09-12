/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.FDeriv.SoftmaxSpec
public import NN.Proofs.Tensor.AxisAdjoint

/-!
# Softmax at arbitrary rank

Each output coordinate belongs to one last-axis row. Selecting that row is a linear map,
so the vector softmax derivative applies after row selection at every tensor rank. The
coordinate identities below use the actual max-shifted specification; no uniqueness
assumption on the maximum is needed.

Coordinate functions give the finite-dimensional space on which we differentiate. Constructing
the input tensor and reading the output are explicit in the derivative statement.
-/

@[expose] public section

namespace Proofs.Autograd.SoftmaxAxis

open Spec TorchLean
open TorchLean.Tensor
open scoped BigOperators

noncomputable section

/-- Number of entries in a last-axis row; the scalar convention has one entry. -/
abbrev rowWidth : Shape → Nat
  | .scalar => 1
  | .dim n .scalar => n
  | .dim _ (.dim n s) => rowWidth (.dim n s)

/-- The last coordinate of a tensor coordinate, with index zero for a scalar. -/
def lastCoordinate : {s : Shape} → Shape.Coord s → Fin (rowWidth s)
  | .scalar, _ => (0 : Fin 1)
  | .dim _ .scalar, p => p.1
  | .dim _ (.dim n s), p => lastCoordinate (s := .dim n s) p.2

/-- Replace the last coordinate while keeping all leading coordinates fixed. -/
def rowCoordinate : {s : Shape} → Shape.Coord s → Fin (rowWidth s) → Shape.Coord s
  | .scalar, _, _ => PUnit.unit
  | .dim _ .scalar, _, j => (j, PUnit.unit)
  | .dim _ (.dim _ _), p, j => (p.1, rowCoordinate p.2 j)

/-- Read one last-axis row from a tensor's coordinate function. -/
def rowVector {s : Shape} (x : Shape.Coord s → ℝ) (p : Shape.Coord s) :
    Vec (rowWidth s) :=
  softmaxVecOfFun fun j => x (rowCoordinate p j)

/-- Row selection is linear and continuous on the finite coordinate space. -/
def rowCLM {s : Shape} (p : Shape.Coord s) :
    (Shape.Coord s → ℝ) →L[ℝ] Vec (rowWidth s) := by
  classical
  let linear : (Shape.Coord s → ℝ) →ₗ[ℝ] Vec (rowWidth s) :=
    { toFun := fun x => rowVector x p
      map_add' := by
        intro x y
        ext j
        simp [rowVector]
      map_smul' := by
        intro a x
        ext j
        simp [rowVector] }
  exact ⟨linear, LinearMap.continuous_of_finiteDimensional (f := linear)⟩

/-- Applying the row-selection map reads exactly that row. -/
@[simp] theorem rowCLM_apply {s : Shape} (p : Shape.Coord s) (x : Shape.Coord s → ℝ) :
    rowCLM p x = rowVector x p := rfl

/-- The only coordinate of a one-entry softmax is one. -/
theorem softmaxVec_one (x : Vec 1) (j : Fin 1) : softmaxVec x j = 1 := by
  have hj : j = 0 := Subsingleton.elim _ _
  subst j
  simp [softmaxVec, sumExp, Real.exp_ne_zero]

/-- A rank-one tensor's selected row is its ordinary Euclidean vectorization. -/
theorem rowVector_vector {n : Nat} (x : Tensor ℝ [n]) (p : Shape.Coord [n]) :
    rowVector (fun q => x q) p = getScalarE x := by
  change (softmaxVecOfFun fun j : Fin n => x (j, PUnit.unit)) = getScalarE x
  ext j
  rw [softmaxVecOfFun_apply, getScalarE_ofLp, Tensor.getScalar_eq_apply]

/-- Selecting a row after taking a leading slice selects the same coordinates. -/
theorem rowVector_unstack {n m : Nat} {s : Shape} (x : Tensor ℝ (.dim n (.dim m s)))
    (p : Shape.Coord (.dim n (.dim m s))) :
    rowVector (fun q => x q) p =
      rowVector (fun q => x.unstack p.1 q) p.2 := by
  ext j
  simp only [rowVector, softmaxVecOfFun_apply, rowCoordinate]
  exact (TorchLean.Tensor.Internal.Rep.unstack_apply x p.1 _).symm

/--
The stable innermost specification is the analytic vector softmax on the selected row.

The proof reduces each leading axis structurally, then uses the proved cancellation of the
max shift in the vector kernel. Empty axes require no special numerical convention: they
have no output coordinate at which the equality could fail.
-/
theorem softmaxInnermostSpec_apply {s : Shape} (x : Tensor ℝ s) (p : Shape.Coord s) :
    Activation.Internal.softmaxInnermostSpec x p =
      softmaxVec (rowVector (fun q => x q) p) (lastCoordinate p) := by
  induction s with
  | scalar =>
      simp only [Activation.Internal.softmaxInnermostSpec, Tensor.scalar,
        TorchLean.Tensor.Internal.Rep.get_ofFn, softmaxVec_one]
  | dim n s ih =>
      cases s with
      | scalar =>
          rcases p with ⟨j, ⟨⟩⟩
          rw [rowVector_vector]
          change (Activation.softmaxVecSpec x) (j, PUnit.unit) =
            softmaxVec (getScalarE x) j
          simpa only [getScalarE_ofLp, Tensor.getScalar_eq_apply] using
            congrArg (fun v : Vec n => v j) (getScalarE_softmaxVecSpec x)
      | dim m s =>
          rw [rowVector_unstack]
          change Tensor.dim
            (fun i => Activation.Internal.softmaxInnermostSpec (x.unstack i)) p = _
          simp only [Tensor.dim, TorchLean.Tensor.Internal.Rep.stack_apply]
          exact ih (x.unstack p.1) p.2

/-- Derivative of the innermost specification, assembled from its row derivatives. -/
def innermostDeriv {s : Shape} (x : Shape.Coord s → ℝ) :
    (Shape.Coord s → ℝ) →L[ℝ] (Shape.Coord s → ℝ) :=
  ContinuousLinearMap.pi fun p =>
    (evalCLM (lastCoordinate p)).comp
      ((softmaxDerivCLM (rowVector x p)).comp (rowCLM p))

/-- The max-shifted softmax specification is Fréchet differentiable at every tensor rank. -/
theorem hasFDerivAt_softmaxInnermostSpec {s : Shape} (x : Shape.Coord s → ℝ) :
    HasFDerivAt
      (fun z : Shape.Coord s → ℝ =>
        fun p => Activation.Internal.softmaxInnermostSpec
          (TorchLean.Tensor.Internal.Rep.ofFn z) p)
      (innermostDeriv x) x := by
  have hcoord (p : Shape.Coord s) :
      HasFDerivAt
        (fun z : Shape.Coord s → ℝ =>
          softmaxVec (rowVector z p) (lastCoordinate p))
        ((evalCLM (lastCoordinate p)).comp
          ((softmaxDerivCLM (rowVector x p)).comp (rowCLM p))) x := by
    simpa only [Function.comp_def, rowCLM_apply, evalCLM_apply] using
      (evalCLM (lastCoordinate p)).hasFDerivAt.comp x
        ((hasFDerivAt_softmaxVec (rowCLM p x)).comp x (rowCLM p).hasFDerivAt)
  have h := (hasFDerivAt_pi (𝕜 := ℝ)).mpr hcoord
  have hfun :
      (fun z : Shape.Coord s → ℝ =>
        fun p => Activation.Internal.softmaxInnermostSpec
          (TorchLean.Tensor.Internal.Rep.ofFn z) p) =
      (fun z : Shape.Coord s → ℝ =>
        fun p => softmaxVec (rowVector z p) (lastCoordinate p)) := by
    funext z p
    rw [softmaxInnermostSpec_apply]
    simp only [TorchLean.Tensor.Internal.Rep.get_ofFn]
  rw [hfun]
  exact h

/-- The one-entry softmax has zero differential. -/
theorem softmaxJvp_one (x dx : Vec 1) (j : Fin 1) : softmaxJvp x dx j = 0 := by
  have hj : j = 0 := Subsingleton.elim _ _
  subst j
  simp [softmaxJvp, softmaxVec_one, dotCLM_apply]

/-- The supplied innermost backward rule is the vector JVP on each selected row. -/
theorem softmaxInnermostBackwardSpec_apply {s : Shape}
    (x dx : Tensor ℝ s) (p : Shape.Coord s) :
    Activation.Internal.softmaxInnermostBackwardSpec x dx p =
      softmaxJvp (rowVector (fun q => x q) p) (rowVector (fun q => dx q) p)
        (lastCoordinate p) := by
  induction s with
  | scalar =>
      simp only [Activation.Internal.softmaxInnermostBackwardSpec, Tensor.scalar,
        TorchLean.Tensor.Internal.Rep.get_ofFn, softmaxJvp_one]
  | dim n s ih =>
      cases s with
      | scalar =>
          rcases p with ⟨j, ⟨⟩⟩
          rw [rowVector_vector, rowVector_vector]
          change (Activation.softmaxBackwardSpec 0 x dx) (j, PUnit.unit) =
            softmaxJvp (getScalarE x) (getScalarE dx) j
          simpa only [getScalarE_ofLp, Tensor.getScalar_eq_apply] using
            congrArg (fun v : Vec n => v j) (getScalarE_softmaxBackwardSpec x dx)
      | dim m s =>
          rw [rowVector_unstack, rowVector_unstack]
          change Tensor.dim
            (fun i => Activation.Internal.softmaxInnermostBackwardSpec
              (x.unstack i) (dx.unstack i)) p = _
          simp only [Tensor.dim, TorchLean.Tensor.Internal.Rep.stack_apply]
          exact ih (x.unstack p.1) (dx.unstack p.1) p.2

/-- Applying the analytic derivative computes the actual specification's backward formula. -/
theorem innermostDeriv_apply {s : Shape} (x dx : Shape.Coord s → ℝ) :
    innermostDeriv x dx =
      fun p => Activation.Internal.softmaxInnermostBackwardSpec
        (TorchLean.Tensor.Internal.Rep.ofFn x)
        (TorchLean.Tensor.Internal.Rep.ofFn dx) p := by
  funext p
  rw [softmaxInnermostBackwardSpec_apply]
  simp only [TorchLean.Tensor.Internal.Rep.get_ofFn]
  change (softmaxDerivCLM (rowVector x p) (rowVector dx p)) (lastCoordinate p) = _
  rw [softmaxJvp_eq_deriv]

/-- The actual last-axis softmax derivative is its supplied backward rule at every rank. -/
theorem fderiv_softmaxInnermostSpec {s : Shape} (x dx : Shape.Coord s → ℝ) :
    fderiv ℝ
      (fun z : Shape.Coord s → ℝ =>
        fun p => Activation.Internal.softmaxInnermostSpec
          (TorchLean.Tensor.Internal.Rep.ofFn z) p) x dx =
      fun p => Activation.Internal.softmaxInnermostBackwardSpec
        (TorchLean.Tensor.Internal.Rep.ofFn x)
        (TorchLean.Tensor.Internal.Rep.ofFn dx) p := by
  rw [(hasFDerivAt_softmaxInnermostSpec x).fderiv]
  exact innermostDeriv_apply x dx

/-- The rank-general backward map is self-adjoint under the tensor inner product. -/
theorem dot_softmaxInnermostBackwardSpec {s : Shape} (x dx δ : Tensor ℝ s) :
    Spec.dot (Activation.Internal.softmaxInnermostBackwardSpec x dx) δ =
      Spec.dot dx (Activation.Internal.softmaxInnermostBackwardSpec x δ) := by
  induction s with
  | scalar =>
      simp [Activation.Internal.softmaxInnermostBackwardSpec,
        Spec.dot_eq_tensorAlgebra_dot, TensorAlgebra.dot]
  | dim n s ih =>
      cases s with
      | scalar =>
          change Spec.dot (Activation.softmaxBackwardSpec 0 x dx) δ =
            Spec.dot dx (Activation.softmaxBackwardSpec 0 x δ)
          rw [dot_eq_inner_vec, dot_eq_inner_vec,
            getScalarE_softmaxBackwardSpec, getScalarE_softmaxBackwardSpec]
          exact inner_softmaxJvp_comm (getScalarE x) (getScalarE dx) (getScalarE δ)
      | dim m s =>
          change Spec.dot
            (Tensor.dim fun i =>
              Activation.Internal.softmaxInnermostBackwardSpec (x.unstack i) (dx.unstack i))
            δ = Spec.dot dx
              (Tensor.dim fun i =>
                Activation.Internal.softmaxInnermostBackwardSpec (x.unstack i) (δ.unstack i))
          simp only [dot_eq_sum_unstack, Tensor.unstack_dim]
          refine Finset.sum_congr rfl fun i _ => ?_
          simpa only [dot_eq_sum_unstack] using
            ih (x.unstack i) (dx.unstack i) (δ.unstack i)

end

end Proofs.Autograd.SoftmaxAxis

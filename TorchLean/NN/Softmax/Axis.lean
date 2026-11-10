/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Softmax.Correctness
public import TorchLean.Tensor.Algebra

/-!
# Softmax at arbitrary rank

Each output coordinate belongs to one last-axis row. At every tensor rank the max-shifted
innermost specification equals the vector softmax of the selected row, with no uniqueness
assumption on the maximum. Selecting that row is a linear map (`rowCLM`), so the vector softmax
derivative applies after row selection: `hasFDerivAt_innermost` gives the Fréchet derivative
`innermostDeriv`, `fderiv_innermost` shows that it applies `Softmax.innermostBackward`, and
`dot_innermostBackward` shows that this backward map is self-adjoint.

Coordinate functions give the finite-dimensional space on which we differentiate. Constructing
the input tensor and reading the output are explicit in the derivative statement.
-/

@[expose] public section

namespace TorchLean.Softmax

open Shape
open TorchLean.Tensor
open scoped BigOperators

noncomputable section

/-- Read one last-axis row from a tensor's coordinate function. -/
def rowVector {s : Shape} (x : Shape.Coord s → ℝ) (p : Shape.Coord s) :
    Vec (Shape.rowWidth s) :=
  WithLp.toLp 2 fun j => x (Shape.Coord.updateLast p j)

/-- The only coordinate of a one-entry softmax is one. -/
theorem _root_.TorchLean.Vec.softmax_fin_one (x : Vec 1) (j : Fin 1) : Vec.softmax x j = 1 := by
  have hj : j = 0 := Subsingleton.elim _ _
  subst j
  simp [Vec.softmax, Vec.sumExp, Real.exp_ne_zero]

/-- A selected row is the vectorization of the tensor's row through the coordinate. -/
theorem rowVector_eq_getScalarE {s : Shape} (x : Tensor ℝ s) (p : Shape.Coord s) :
    rowVector (fun q => x q) p = getScalarE (x.row p) := by
  ext j
  rw [rowVector, PiLp.toLp_apply, getScalarE_ofLp, Tensor.getScalar_eq_apply, row_apply]

/-- The stable innermost specification is the analytic vector softmax on the selected row. -/
theorem innermost_apply_row {s : Shape} (x : Tensor ℝ s) (p : Shape.Coord s) :
    innermost x p =
      Vec.softmax (rowVector (fun q => x q) p) (Shape.Coord.getLast p) := by
  rw [innermost_apply]
  split_ifs with hs
  · subst hs
    exact (Vec.softmax_fin_one _ _).symm
  · rw [rowVector_eq_getScalarE]
    simpa only [getScalarE_ofLp] using congrArg (fun v : Vec _ => v (Shape.Coord.getLast p))
      (getScalarE_vec (x.row p))

/-- `rowVector · p`, which reads the last-axis row through `p`, as a continuous linear map on
coordinate functions. -/
def rowCLM {s : Shape} (p : Shape.Coord s) :
    (Shape.Coord s → ℝ) →L[ℝ] Vec (Shape.rowWidth s) := by
  classical
  let linear : (Shape.Coord s → ℝ) →ₗ[ℝ] Vec (Shape.rowWidth s) :=
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

/-- `rowCLM p` maps `x` to `rowVector x p`. -/
@[simp] theorem rowCLM_apply {s : Shape} (p : Shape.Coord s) (x : Shape.Coord s → ℝ) :
    rowCLM p x = rowVector x p := rfl

/-- The derivative of `Softmax.innermost` at `x`. Its coordinate `p` applies the vector softmax
derivative at the row of `x` through `p` to the row of the direction through `p`, read at the
last index of `p`. -/
def innermostDeriv {s : Shape} (x : Shape.Coord s → ℝ) :
    (Shape.Coord s → ℝ) →L[ℝ] (Shape.Coord s → ℝ) :=
  ContinuousLinearMap.pi fun p =>
    (EuclideanSpace.proj (𝕜 := ℝ) (Shape.Coord.getLast p)).comp
      ((Vec.softmaxDerivCLM (rowVector x p)).comp (rowCLM p))

/-- At every tensor rank, the max-shifted softmax `Softmax.innermost`, read through coordinate
functions, has Fréchet derivative `innermostDeriv x` at `x`. -/
theorem hasFDerivAt_innermost {s : Shape} (x : Shape.Coord s → ℝ) :
    HasFDerivAt
      (fun z : Shape.Coord s → ℝ =>
        fun p => innermost (TorchLean.Tensor.Rep.ofFn z) p)
      (innermostDeriv x) x := by
  have hcoord (p : Shape.Coord s) :
      HasFDerivAt
        (fun z : Shape.Coord s → ℝ =>
          Vec.softmax (rowVector z p) (Shape.Coord.getLast p))
        ((EuclideanSpace.proj (𝕜 := ℝ) (Shape.Coord.getLast p)).comp
          ((Vec.softmaxDerivCLM (rowVector x p)).comp (rowCLM p))) x := by
    simpa only [Function.comp_def, rowCLM_apply, EuclideanSpace.coe_proj] using
      (EuclideanSpace.proj (𝕜 := ℝ) (Shape.Coord.getLast p)).hasFDerivAt.comp x
        ((Vec.hasFDerivAt_softmax (rowCLM p x)).comp x (rowCLM p).hasFDerivAt)
  have h := (hasFDerivAt_pi (𝕜 := ℝ)).mpr hcoord
  have hfun :
      (fun z : Shape.Coord s → ℝ =>
        fun p => innermost (TorchLean.Tensor.Rep.ofFn z) p) =
      (fun z : Shape.Coord s → ℝ =>
        fun p => Vec.softmax (rowVector z p) (Shape.Coord.getLast p)) := by
    funext z p
    rw [innermost_apply_row]
    simp only [TorchLean.Tensor.Rep.get_ofFn]
  rw [hfun]
  exact h

/-- On one-entry vectors every entry of `Vec.softmaxJVP x dx` is zero. -/
theorem _root_.TorchLean.Vec.softmaxJVP_fin_one (x dx : Vec 1) (j : Fin 1) :
    Vec.softmaxJVP x dx j = 0 := by
  have hj : j = 0 := Subsingleton.elim _ _
  subst j
  simp [Vec.softmaxJVP, Vec.softmax_fin_one, Vec.inner_eq_sum_mul]

/-- At every coordinate `p`, `Softmax.innermostBackward x dx` is the vector softmax JVP of the
rows of `x` and `dx` through `p`, read at the last index of `p`. -/
theorem innermostBackward_apply_row {s : Shape}
    (x dx : Tensor ℝ s) (p : Shape.Coord s) :
    innermostBackward x dx p =
      Vec.softmaxJVP (rowVector (fun q => x q) p) (rowVector (fun q => dx q) p)
        (Shape.Coord.getLast p) := by
  rw [innermostBackward_apply]
  split_ifs with hs
  · subst hs
    exact (Vec.softmaxJVP_fin_one _ _ _).symm
  · rw [rowVector_eq_getScalarE, rowVector_eq_getScalarE]
    change (Tensor.softmaxBackward (x.row p) (dx.row p) ⟨0, Nat.one_pos⟩).getScalar
      (Shape.Coord.getLast p) = _
    simpa only [getScalarE_ofLp] using congrArg (fun v : Vec _ => v (Shape.Coord.getLast p))
      (getScalarE_softmaxBackward (x.row p) (dx.row p))

/-- `innermostDeriv x` maps `dx` to `Softmax.innermostBackward` of the tensors built from `x` and
`dx`. -/
theorem innermostDeriv_apply {s : Shape} (x dx : Shape.Coord s → ℝ) :
    innermostDeriv x dx =
      fun p => innermostBackward
        (TorchLean.Tensor.Rep.ofFn x)
        (TorchLean.Tensor.Rep.ofFn dx) p := by
  funext p
  rw [innermostBackward_apply_row]
  simp only [TorchLean.Tensor.Rep.get_ofFn]
  change (Vec.softmaxDerivCLM (rowVector x p) (rowVector dx p)) (Shape.Coord.getLast p) = _
  rw [Vec.softmaxJVP_eq_deriv]

/-- At every tensor rank, the Fréchet derivative of `Softmax.innermost` at `x`, applied to `dx`,
is `Softmax.innermostBackward` of the tensors built from `x` and `dx`. -/
theorem fderiv_innermost {s : Shape} (x dx : Shape.Coord s → ℝ) :
    fderiv ℝ
      (fun z : Shape.Coord s → ℝ =>
        fun p => innermost (TorchLean.Tensor.Rep.ofFn z) p) x dx =
      fun p => innermostBackward
        (TorchLean.Tensor.Rep.ofFn x)
        (TorchLean.Tensor.Rep.ofFn dx) p := by
  rw [(hasFDerivAt_innermost x).fderiv]
  exact innermostDeriv_apply x dx

/-- At every tensor rank, `Softmax.innermostBackward x` is self-adjoint for the pairing
`Rep.dot`. -/
theorem dot_innermostBackward {s : Shape} (x dx δ : Tensor ℝ s) :
    Rep.dot (innermostBackward x dx) δ = Rep.dot dx (innermostBackward x δ) := by
  induction s with
  | scalar =>
      simp [innermostBackward, Rep.dot]
  | dim n s ih =>
      cases s with
      | scalar =>
          change Rep.dot (Tensor.softmaxBackward x dx ⟨0, Nat.one_pos⟩) δ =
            Rep.dot dx (Tensor.softmaxBackward x δ ⟨0, Nat.one_pos⟩)
          rw [dot_eq_inner_getScalarE, dot_eq_inner_getScalarE,
            getScalarE_softmaxBackward, getScalarE_softmaxBackward]
          exact Vec.inner_softmaxJVP_comm (getScalarE x) (getScalarE dx) (getScalarE δ)
      | dim m s =>
          cases n with
          | zero => simp [dot_eq_sum_unstack]
          | succ n =>
              change Rep.dot
                (Tensor.dim fun i => innermostBackward (x.unstack i) (dx.unstack i))
                δ = Rep.dot dx
                  (Tensor.dim fun i => innermostBackward (x.unstack i) (δ.unstack i))
              simp only [dot_eq_sum_unstack, Tensor.unstack_dim]
              refine Finset.sum_congr rfl fun i _ => ?_
              simpa only [dot_eq_sum_unstack] using
                ih (x.unstack i) (dx.unstack i) (δ.unstack i)

end

end TorchLean.Softmax

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Softmax.Correctness
public import TorchLean.NN.FDeriv.TensorCoordinates
public import TorchLean.Tensor.Op.Table
public import TorchLean.Tensor.Op.Cert

/-!
# Derivative certificates of softmax and log-softmax along any axis

Softmax along `dim` applies the vector softmax to every fiber of its input along `dim`, and its
pullback applies the vector pullback to every fiber (`Tensor.softmax_apply`,
`Tensor.softmaxBackward_apply`). Take a splitting `e` of the coordinates of a shape into an index
along a fiber and an index `r` of the fiber. If the vector map has derivative `L r` on fiber `r`,
the tensor map has derivative `fiberwiseCLM e L`, and a pullback that acts on each fiber `r` by
an adjoint of `L r` is the adjoint of that derivative (`fiberwise_backward_eq_adjoint`). The arms
of `Op.softmax` and `Op.logSoftmax` are instances at every shape and every axis, with the
splitting `Coord.axisEquiv dim`.
-/

@[expose] public section

open TorchLean.Shape TorchLean.Tensor TorchLean.Vec TorchLean.LogSoftmax TorchLean.Softmax
open scoped RealInnerProductSpace

namespace TorchLean.Op

noncomputable section

variable {s : Shape} {n : Nat} {ι : Type}

/-! ## Maps acting on fibers -/

/-- The fiber at `r` of a real tensor under the splitting `e`, as a Euclidean vector whose entry `j`
is the entry at `e.symm (j, r)`. -/
def fiberCLM (e : s.Coord ≃ Fin n × ι) (r : ι) : Tensor ℝ s →L[ℝ] Vec n :=
  (EuclideanSpace.equiv (𝕜 := ℝ) (ι := Fin n)).symm.toContinuousLinearMap.comp
    (ContinuousLinearMap.pi fun j => Tensor.coordinateCLM (e.symm (j, r)))

/-- Entry `j` of the fiber at `r` is the entry at `e.symm (j, r)`. -/
@[simp] theorem fiberCLM_apply (e : s.Coord ≃ Fin n × ι) (r : ι) (y : Tensor ℝ s) (j : Fin n) :
    fiberCLM e r y j = y (e.symm (j, r)) :=
  rfl

/-- The map that applies `L r` to the fiber at `r` of its argument, for every `r`. -/
def fiberwiseCLM (e : s.Coord ≃ Fin n × ι) (L : ι → Vec n →L[ℝ] Vec n) :
    Tensor ℝ s →L[ℝ] Tensor ℝ s :=
  Tensor.assemble fun c =>
    (EuclideanSpace.proj (𝕜 := ℝ) (e c).1).comp ((L (e c).2).comp (fiberCLM e (e c).2))

/-- The entry of `fiberwiseCLM e L u` at `c` is entry `(e c).1` of `L (e c).2` applied to the
fiber at `(e c).2` of `u`. -/
@[simp] theorem fiberwiseCLM_apply (e : s.Coord ≃ Fin n × ι) (L : ι → Vec n →L[ℝ] Vec n)
    (u : Tensor ℝ s) (c : s.Coord) :
    fiberwiseCLM e L u c = L (e c).2 (fiberCLM e (e c).2 u) (e c).1 := by
  simp [fiberwiseCLM]

/-- The fiber at `r` of `fiberwiseCLM e L u` is `L r` applied to the fiber at `r` of `u`. -/
theorem fiberCLM_fiberwiseCLM (e : s.Coord ≃ Fin n × ι) (L : ι → Vec n →L[ℝ] Vec n) (r : ι)
    (u : Tensor ℝ s) : fiberCLM e r (fiberwiseCLM e L u) = L r (fiberCLM e r u) := by
  ext j
  simp

/-- The inner product of real tensors is the sum over `r` of the inner products of their fibers
at `r`. -/
theorem inner_eq_sum_fiberCLM [Fintype ι] (e : s.Coord ≃ Fin n × ι) (a b : Tensor ℝ s) :
    ⟪a, b⟫ = ∑ r, ⟪fiberCLM e r a, fiberCLM e r b⟫ := by
  rw [Tensor.inner_eq_sum, ← e.symm.sum_comp, Fintype.sum_prod_type_right]
  refine Finset.sum_congr rfl fun r _ => ?_
  simp only [PiLp.inner_apply, fiberCLM_apply, RCLike.inner_apply, conj_trivial, mul_comm]

/-- If `f` has derivative `L r` at the fiber at `r` of `x` for every `r`, a map `F` that applies
`f` to every fiber has derivative `fiberwiseCLM e L` at `x`. -/
theorem fiberwise_hasFDerivAt (e : s.Coord ≃ Fin n × ι) (F : Tensor ℝ s → Tensor ℝ s)
    (f : Vec n → Vec n) (L : ι → Vec n →L[ℝ] Vec n) (x : Tensor ℝ s)
    (hF : ∀ y c, F y c = f (fiberCLM e (e c).2 y) (e c).1)
    (hf : ∀ r, HasFDerivAt f (L r) (fiberCLM e r x)) :
    HasFDerivAt F (fiberwiseCLM e L) x := by
  refine Tensor.hasFDerivAt_of_coordinates F _ x fun c => ?_
  have h := (EuclideanSpace.proj (𝕜 := ℝ) (e c).1).hasFDerivAt.comp x
    ((hf (e c).2).comp x (fiberCLM e (e c).2).hasFDerivAt)
  have hfun :
      (fun y => F y c) = (EuclideanSpace.proj (𝕜 := ℝ) (e c).1) ∘ f ∘ (fiberCLM e (e c).2) := by
    funext y
    rw [hF]
    rfl
  rw [hfun]
  exact h

/-- If `B r` is an adjoint of `L r` for every `r`, the adjoint of `fiberwiseCLM e L` maps `δ` to the
tensor that applies `B r` to the fiber at `r` of `δ`. -/
theorem fiberwiseCLM_adjoint [Fintype ι] (e : s.Coord ≃ Fin n × ι) (L : ι → Vec n →L[ℝ] Vec n)
    (B : ι → Vec n → Vec n) (hB : ∀ r u w, ⟪L r u, w⟫ = ⟪u, B r w⟫) (δ G : Tensor ℝ s)
    (hG : ∀ c, G c = B (e c).2 (fiberCLM e (e c).2 δ) (e c).1) :
    (fiberwiseCLM e L).adjoint δ = G := by
  have hfiber (r : ι) : fiberCLM e r G = B r (fiberCLM e r δ) := by
    ext j
    simp [hG]
  apply ext_inner_left ℝ
  intro u
  rw [ContinuousLinearMap.adjoint_inner_right, inner_eq_sum_fiberCLM e, inner_eq_sum_fiberCLM e]
  refine Finset.sum_congr rfl fun r _ => ?_
  rw [fiberCLM_fiberwiseCLM, hfiber, hB]

/-- If `F` applies `f` to every fiber and `f` has derivative `L r` with adjoint `B r` at fiber `r`
of `x`, then `fiberwiseCLM e L` is the derivative of `F` at `x`, and the tensor `G` that applies
`B r` to fiber `r` of `δ` is the adjoint of that derivative applied to `δ`. -/
theorem fiberwise_backward_eq_adjoint [Fintype ι] (e : s.Coord ≃ Fin n × ι)
    (F : Tensor ℝ s → Tensor ℝ s) (f : Vec n → Vec n) (L : ι → Vec n →L[ℝ] Vec n)
    (B : ι → Vec n → Vec n) (x δ G : Tensor ℝ s)
    (hF : ∀ y c, F y c = f (fiberCLM e (e c).2 y) (e c).1)
    (hf : ∀ r, HasFDerivAt f (L r) (fiberCLM e r x)) (hB : ∀ r u w, ⟪L r u, w⟫ = ⟪u, B r w⟫)
    (hG : ∀ c, G c = B (e c).2 (fiberCLM e (e c).2 δ) (e c).1) :
    HasFDerivAt F (fiberwiseCLM e L) x ∧ G = (fderiv ℝ F x).adjoint δ := by
  have h := fiberwise_hasFDerivAt e F f L x hF hf
  exact ⟨h, by rw [h.fderiv, fiberwiseCLM_adjoint e L B hB δ G hG]⟩

/-! ## Fibers along an axis -/

/-- The fiber of `y` along `dim` through `c`, as a Euclidean vector, is the fiber of `y` at the
other indices of `c` under `Coord.axisEquiv dim`. -/
theorem getScalarE_fiber (y : Tensor ℝ s) (dim : Fin s.rank) (c : s.Coord) :
    getScalarE (y.fiber dim c) = fiberCLM (Coord.axisEquiv dim (shape := s))
      (Coord.axisEquiv dim (shape := s) c).2 y := by
  ext j
  rw [getScalarE_ofLp, getScalar_eq_apply, fiber_apply]
  rfl

/-- The fiber along `dim` through a coordinate of the fiber through `c` is the fiber through
`c`. -/
theorem fiber_axisEquiv_symm {α : Type} [Storage α] (y : Tensor α s)
    (dim : Fin s.rank) (j : Fin (s.axisSize dim)) (c : s.Coord) :
    y.fiber dim ((Coord.axisEquiv dim (shape := s)).symm
      (j, (Coord.axisEquiv dim (shape := s) c).2)) = y.fiber dim c := by
  simp only [Tensor.fiber, Equiv.apply_symm_apply]

/-! ## Softmax -/

/-- The entry of `Tensor.softmax y dim` at `c` is the analytic `Vec.softmax` of the fiber through
`c`, at the index of `c` along `dim`. -/
theorem softmax_apply_vec_softmax (y : Tensor ℝ s) (dim : Fin s.rank) (c : s.Coord) :
    (Tensor.softmax y dim) c =
      Vec.softmax (fiberCLM (Coord.axisEquiv dim (shape := s))
        (Coord.axisEquiv dim (shape := s) c).2 y)
        (Coord.axisEquiv dim (shape := s) c).1 := by
  rw [Tensor.softmax_apply, ← getScalarE_ofLp, Softmax.getScalarE_vec, getScalarE_fiber]

/-- The entry of `Tensor.softmaxBackward x δ dim` at `c` is the analytic `softmaxJVP` of the
fibers of `x` and `δ` through `c`, at the index of `c` along `dim`. -/
theorem softmaxBackward_apply_softmaxJVP (x δ : Tensor ℝ s) (dim : Fin s.rank) (c : s.Coord) :
    Tensor.softmaxBackward x δ dim c =
      softmaxJVP (fiberCLM (Coord.axisEquiv dim (shape := s))
          (Coord.axisEquiv dim (shape := s) c).2 x)
        (fiberCLM (Coord.axisEquiv dim (shape := s))
          (Coord.axisEquiv dim (shape := s) c).2 δ)
        (Coord.axisEquiv dim (shape := s) c).1 := by
  have h := getScalarE_softmaxBackward (x.fiber dim c) (δ.fiber dim c)
  rw [Tensor.softmaxBackward_zero_vec, getScalarE_fiber, getScalarE_fiber] at h
  rw [Tensor.softmaxBackward_apply, ← getScalarE_ofLp, h]

/-- At every shape and axis, the softmax forward map has at `x` the derivative that acts on each
fiber along `dim` by `softmaxDerivCLM` at the matching fiber of `x`, and its pullback is the
adjoint of that derivative. -/
@[op_cert TorchLean.Op.softmax derivative]
theorem softmax_backward_eq_adjoint (dim : Fin s.rank) (x δ : Tensor ℝ s) :
    HasFDerivAt (fun y => (((Op.softmax dim).rule? (α := ℝ) |>.get rfl).forward (y, ())).1)
        (fiberwiseCLM (Coord.axisEquiv dim (shape := s)) fun r =>
          softmaxDerivCLM (fiberCLM (Coord.axisEquiv dim (shape := s)) r x)) x ∧
      (((Op.softmax dim).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.softmax dim).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ :=
  fiberwise_backward_eq_adjoint _ _ Vec.softmax _
    (fun r => softmaxJVP (fiberCLM (Coord.axisEquiv dim (shape := s)) r x)) x δ _
    (fun y c => softmax_apply_vec_softmax y dim c) (fun _ => Vec.hasFDerivAt_softmax _)
    (fun _ u w => by rw [← softmaxJVP_eq_deriv]; exact inner_softmaxJVP_comm _ u w)
    (fun c => softmaxBackward_apply_softmaxJVP x δ dim c)

/-! ## Log-softmax -/

/-- The entry of `Tensor.logSoftmax y dim` at `c` is the analytic `Vec.logSoftmax` of the fiber
through `c`, at the index of `c` along `dim`. -/
theorem logSoftmax_apply_vec_logSoftmax (y : Tensor ℝ s) (dim : Fin s.rank) (c : s.Coord) :
    Tensor.logSoftmax y dim c =
      Vec.logSoftmax (fiberCLM (Coord.axisEquiv dim (shape := s))
        (Coord.axisEquiv dim (shape := s) c).2 y)
        (Coord.axisEquiv dim (shape := s) c).1 := by
  rw [Tensor.logSoftmax_apply, ← getScalarE_ofLp, LogSoftmax.getScalarE_vec, getScalarE_fiber]

/-- The fiber of `Tensor.logSoftmax x dim` along `dim` through `c` is the vector log-softmax of the
fiber of `x` through `c`. -/
theorem fiber_logSoftmax (x : Tensor ℝ s) (dim : Fin s.rank) (c : s.Coord) :
    (Tensor.logSoftmax x dim).fiber dim c =
      LogSoftmax.vec (x.fiber dim c) := by
  apply Tensor.ext_vector
  intro j
  rw [getScalar_eq_apply, fiber_apply, Tensor.logSoftmax_apply, fiber_axisEquiv_symm,
    Equiv.apply_symm_apply]

/-- The entry at `c` of the log-softmax pullback, evaluated on the forward output, is the analytic
`logSoftmaxVJP` of the fibers of `x` and `δ` through `c`, at the index of `c` along `dim`. -/
theorem logSoftmaxBackward_apply_logSoftmaxVJP (x δ : Tensor ℝ s) (dim : Fin s.rank)
    (c : s.Coord) :
    Tensor.logSoftmaxBackward (Tensor.logSoftmax x dim) δ dim c =
      logSoftmaxVJP (fiberCLM (Coord.axisEquiv dim (shape := s))
          (Coord.axisEquiv dim (shape := s) c).2 x)
        (fiberCLM (Coord.axisEquiv dim (shape := s))
          (Coord.axisEquiv dim (shape := s) c).2 δ)
        (Coord.axisEquiv dim (shape := s) c).1 := by
  have h := getScalarE_logSoftmaxBackward (x.fiber dim c) (δ.fiber dim c)
  rw [Tensor.logSoftmaxBackward_zero_vec, Tensor.logSoftmax_zero_vec, getScalarE_fiber,
    getScalarE_fiber] at h
  rw [Tensor.logSoftmaxBackward_apply, fiber_logSoftmax, ← getScalarE_ofLp]
  exact congrArg (fun v : Vec _ => v _) h

/-- At every shape and axis, the log-softmax forward map has at `x` the derivative that acts on
each fiber along `dim` by `logSoftmaxDerivCLM` at the matching fiber of `x`. Its pullback, which
reads the forward output, is the adjoint of that derivative. -/
@[op_cert TorchLean.Op.logSoftmax derivative]
theorem logSoftmax_backward_eq_adjoint (dim : Fin s.rank) (x δ : Tensor ℝ s) :
    HasFDerivAt (fun y => (((Op.logSoftmax dim).rule? (α := ℝ) |>.get rfl).forward (y, ())).1)
        (fiberwiseCLM (Coord.axisEquiv dim (shape := s)) fun r =>
          logSoftmaxDerivCLM (fiberCLM (Coord.axisEquiv dim (shape := s)) r x)) x ∧
      (((Op.logSoftmax dim).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.logSoftmax dim).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ :=
  fiberwise_backward_eq_adjoint _ _ Vec.logSoftmax _
    (fun r => logSoftmaxVJP (fiberCLM (Coord.axisEquiv dim (shape := s)) r x)) x δ _
    (fun y c => logSoftmax_apply_vec_logSoftmax y dim c) (fun _ => Vec.hasFDerivAt_logSoftmax _)
    (fun _ u w => by rw [← logSoftmaxJVP_eq_deriv]; exact inner_logSoftmaxJVP_vjp _ u w)
    (fun c => logSoftmaxBackward_apply_logSoftmaxVJP x δ dim c)

end

end TorchLean.Op

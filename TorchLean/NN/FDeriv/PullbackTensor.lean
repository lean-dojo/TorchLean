/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.FDeriv.Pullback
public import TorchLean.NN.Lipschitz.TensorNorm
public import TorchLean.NN.Op.LayerNorm
public import TorchLean.Tensor.Op.Table.Reduce
public import TorchLean.NN.Op.Pointwise
public import TorchLean.NN.Op.Softmax
public import TorchLean.NN.Tactic.Init
public import TorchLean.Tensor.Movement.BroadcastAdjoint
public import TorchLean.Tensor.Reduce.Real

/-!
# Pullback rules for maps into real tensors

Rules of `HasPullbackAt` for the tensor operations that operator specifications are built from:
entrywise products and quotients, a real function applied at every entry where it has a
derivative, a real function of the entries of two maps, reshaping, casting along a shape
equality, broadcasting, the sum of every entry, a filled tensor and the scalar tensor of a real
number, scaling by a fixed real or a real-valued map, the value of a scalar tensor, an entry read,
and softmax along an axis. With the rules of `TorchLean.NN.FDeriv.Pullback` they let `fun_prop`
build the derivative of an operator's forward map and its pullback from the parts of the
specification. An entrywise map has the slope map `Op.slopeCLM` as derivative, a linear operation
is its own derivative, and each pullback is the adjoint operation read off an inner product
identity.

The `op_deriv_simps` lemmas here give the named entrywise functions over `ℝ` as `Tensor.map` of
the real function, the entry of a finite sum of tensors, and the linearity of the stored
log-softmax pullback in the cotangent.
-/

@[expose] public section

open scoped RealInnerProductSpace

namespace TorchLean

namespace Tensor

variable {s : Shape}

/-- The sum of every entry, as a continuous linear map on real tensors. -/
noncomputable def sumCLM (s : Shape) : Tensor ℝ s →L[ℝ] ℝ :=
  ∑ c : s.Coord, coordinateCLM c

/-- `sumCLM s` applied to `y` is `Tensor.sum y`. -/
@[simp] theorem sumCLM_apply (y : Tensor ℝ s) : sumCLM s y = Tensor.sum y := by
  rw [sumCLM, _root_.sum_apply, sum_eq_coord_sum]
  rfl

/-- Filling a tensor of shape `s` with a real number, as a continuous linear map. -/
noncomputable def fullCLM (s : Shape) : ℝ →L[ℝ] Tensor ℝ s :=
  LinearMap.toContinuousLinearMap
    { toFun := fun a => Tensor.full s a
      map_add' := fun a b => Rep.ext fun c => by simp only [full_apply, Tensor.add_apply]
      map_smul' := fun a b => Rep.ext fun c => by
        simp only [full_apply, Rep.smul_apply, RingHom.id_apply, smul_eq_mul] }

/-- `fullCLM s` applied to `a` is `Tensor.full s a`. -/
@[simp] theorem fullCLM_apply (a : ℝ) : fullCLM s a = Tensor.full s a :=
  rfl

/-- The scalar tensor of a real number, as a continuous linear map. -/
noncomputable def scalarCLM : ℝ →L[ℝ] Tensor ℝ .scalar :=
  LinearMap.toContinuousLinearMap
    { toFun := fun a => Tensor.scalar a
      map_add' := fun a b => Rep.ext fun c => by
        rw [Tensor.add_apply]
        exact (scalar_apply (a + b)).trans (congrArg₂ (· + ·) (scalar_apply a).symm
          (scalar_apply b).symm)
      map_smul' := fun a b => Rep.ext fun c => by
        rw [Rep.smul_apply, RingHom.id_apply, smul_eq_mul]
        exact (scalar_apply (a * b)).trans (congrArg (a * ·) (scalar_apply b).symm) }

/-- `scalarCLM` applied to `a` is `Tensor.scalar a`. -/
@[simp] theorem scalarCLM_apply (a : ℝ) : scalarCLM a = Tensor.scalar a :=
  rfl

/-- Reshaping a real tensor to a shape with as many entries, as a continuous linear map. -/
noncomputable def reshapeCLM (t : Shape) (h : s.size = t.size) : Tensor ℝ s →L[ℝ] Tensor ℝ t :=
  (reshapeₗᵢ t h).toContinuousLinearEquiv.toContinuousLinearMap

/-- `reshapeCLM t h` applied to `y` is `reshape y t h`. -/
@[simp] theorem reshapeCLM_apply (t : Shape) (h : s.size = t.size) (y : Tensor ℝ s) :
    reshapeCLM t h y = reshape y t h :=
  rfl

/-- Casting a real tensor along a shape equality, as a continuous linear map. -/
noncomputable def castShapeCLM {t : Shape} (h : s = t) : Tensor ℝ s →L[ℝ] Tensor ℝ t :=
  LinearMap.toContinuousLinearMap
    { toFun := fun y => castShape y h
      map_add' := fun y z => by subst h; rfl
      map_smul' := fun a y => by subst h; rfl }

/-- `castShapeCLM h` applied to `y` is `castShape y h`. -/
@[simp] theorem castShapeCLM_apply {t : Shape} (h : s = t) (y : Tensor ℝ s) :
    castShapeCLM h y = castShape y h :=
  rfl

/-- The inner product with a filled tensor is the filling value times the sum. -/
theorem inner_full_right (y : Tensor ℝ s) (a : ℝ) :
    ⟪y, Tensor.full s a⟫ = Tensor.sum y * a := by
  rw [inner_eq_sum, sum_eq_coord_sum, Finset.sum_mul]
  exact Finset.sum_congr rfl fun c _ => by rw [full_apply]

/-- Casting along a shape equality preserves the inner product with the cast back. -/
theorem inner_castShape {t : Shape} (h : s = t) (y : Tensor ℝ s) (δ : Tensor ℝ t) :
    ⟪castShape y h, δ⟫ = ⟪y, castShape δ h.symm⟫ := by
  subst h
  rfl

/-- Reshaping preserves the inner product with the reshaped back. -/
theorem inner_reshape {t : Shape} (h : s.size = t.size) (y : Tensor ℝ s) (δ : Tensor ℝ t) :
    ⟪reshape y t h, δ⟫ = ⟪y, reshape δ s h.symm⟫ := by
  have hδ : δ = reshapeₗᵢ t h (reshape δ s h.symm) := (reshape_reshape_self δ h.symm h).symm
  conv_lhs => rw [hδ]
  rw [← reshapeₗᵢ_apply t h y, LinearIsometryEquiv.inner_map_map]

/-- Over `ℝ`, `Tensor.exp` is `Real.exp` at every entry. -/
@[op_deriv_simps] theorem exp_real (x : Tensor ℝ s) : Tensor.exp x = Tensor.map Real.exp x :=
  rfl

/-- Over `ℝ`, `Tensor.log` is `Real.log` at every entry. -/
@[op_deriv_simps] theorem log_real (x : Tensor ℝ s) : Tensor.log x = Tensor.map Real.log x :=
  rfl

/-- Over `ℝ`, `Tensor.sin` is `Real.sin` at every entry. -/
@[op_deriv_simps] theorem sin_real (x : Tensor ℝ s) : Tensor.sin x = Tensor.map Real.sin x :=
  rfl

/-- Over `ℝ`, `Tensor.cos` is `Real.cos` at every entry. -/
@[op_deriv_simps] theorem cos_real (x : Tensor ℝ s) : Tensor.cos x = Tensor.map Real.cos x :=
  rfl

/-- Over `ℝ`, `Tensor.sinh` is `Real.sinh` at every entry. -/
@[op_deriv_simps] theorem sinh_real (x : Tensor ℝ s) : Tensor.sinh x = Tensor.map Real.sinh x :=
  rfl

/-- Over `ℝ`, `Tensor.cosh` is `Real.cosh` at every entry. -/
@[op_deriv_simps] theorem cosh_real (x : Tensor ℝ s) : Tensor.cosh x = Tensor.map Real.cosh x :=
  rfl

/-- Over `ℝ`, `Tensor.tanh` is `Real.tanh` at every entry. -/
@[op_deriv_simps] theorem tanh_real (x : Tensor ℝ s) : Tensor.tanh x = Tensor.map Real.tanh x :=
  rfl

/-- Over `ℝ`, `Tensor.sqrt` with its default policy is `Real.sqrt` at every entry. -/
@[op_deriv_simps] theorem sqrt_real (x : Tensor ℝ s) : Tensor.sqrt x = Tensor.map Real.sqrt x :=
  rfl

/-- The entry at `c` of a finite sum of real tensors is the sum of their entries at `c`. -/
@[op_deriv_simps] theorem Rep.sum_apply {ι : Type*} (u : Finset ι) (f : ι → Tensor ℝ s)
    (c : s.Coord) : (∑ i ∈ u, f i) c = ∑ i ∈ u, f i c := by
  simpa only [coordinateCLM_apply] using _root_.map_sum (coordinateCLM c) f u

end Tensor

namespace Op

variable {s : Shape}

/-- Softmax along `dim` has at `x` the fiberwise derivative of its in-tree certificate, with the
stored softmax pullback at `x` as its adjoint. -/
theorem hasPullbackAt_softmax (dim : Fin s.rank) (x : Tensor ℝ s) :
    HasPullbackAt (fun y => Tensor.softmax y dim)
      (fiberwiseCLM (Shape.Coord.axisEquiv dim (shape := s)) fun r =>
        Vec.softmaxDerivCLM (fiberCLM (Shape.Coord.axisEquiv dim (shape := s)) r x))
      (fun δ => Tensor.softmaxBackward x δ dim) x := by
  have h := fun δ => softmax_backward_eq_adjoint dim x δ
  simp only [Op.rule?_softmax, Option.get_some] at h
  exact HasPullbackAt.of_adjoint (h 0).1 fun δ => by rw [(h δ).2, (h δ).1.fderiv]

/-- The stored log-softmax pullback, read at the forward output, maps a sum of cotangents to the
sum of their pullbacks. -/
@[op_deriv_simps]
theorem logSoftmaxBackward_add (x : Tensor ℝ s) (dim : Fin s.rank) (δ δ' : Tensor ℝ s) :
    Tensor.logSoftmaxBackward (Tensor.logSoftmax x dim) (δ + δ') dim =
      Tensor.logSoftmaxBackward (Tensor.logSoftmax x dim) δ dim +
        Tensor.logSoftmaxBackward (Tensor.logSoftmax x dim) δ' dim := by
  have h := fun δ => (logSoftmax_backward_eq_adjoint dim x δ).2
  simp only [Op.rule?_logSoftmax, Option.get_some] at h
  rw [h, h, h, map_add]

/-- The stored log-softmax pullback, read at the forward output, maps a negated cotangent to the
negated pullback. -/
@[op_deriv_simps]
theorem logSoftmaxBackward_neg (x : Tensor ℝ s) (dim : Fin s.rank) (δ : Tensor ℝ s) :
    Tensor.logSoftmaxBackward (Tensor.logSoftmax x dim) (-δ) dim =
      -Tensor.logSoftmaxBackward (Tensor.logSoftmax x dim) δ dim := by
  have h := fun δ => (logSoftmax_backward_eq_adjoint dim x δ).2
  simp only [Op.rule?_logSoftmax, Option.get_some] at h
  rw [h, h, map_neg]

/-- The stored log-softmax pullback, read at the forward output, maps the zero cotangent to
zero. -/
@[op_deriv_simps]
theorem logSoftmaxBackward_zero (x : Tensor ℝ s) (dim : Fin s.rank) :
    Tensor.logSoftmaxBackward (Tensor.logSoftmax x dim) 0 dim = 0 := by
  have h := fun δ => (logSoftmax_backward_eq_adjoint dim x δ).2
  simp only [Op.rule?_logSoftmax, Option.get_some] at h
  rw [h, map_zero]

end Op

namespace HasPullbackAt

open Tensor Op

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] {s t : Shape} {x : E}
  {f g : E → Tensor ℝ s} {Df Dg : E →L[ℝ] Tensor ℝ s} {bf bg : Tensor ℝ s → E}

/-- A map that applies `φ c` at the entry `c` of `f y`, where `φ c` has derivative `φ' c` at
`f x c`, has the slope map of `φ'` after the derivative of `f`, and pulls back through the
slopes. -/
theorem pointwise (F : Tensor ℝ s → Tensor ℝ s) (φ : s.Coord → ℝ → ℝ) {φ' : s.Coord → ℝ}
    (hF : ∀ y c, F y c = φ c (y c)) (hφ : ∀ c, HasDerivAt (φ c) (φ' c) (f x c))
    (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => F (f y)) ((slopeCLM φ').comp Df) (fun δ => bf (slopeCLM φ' δ)) x :=
  ⟨(pointwise_hasFDerivAt F φ φ' (f x) hF hφ).comp x hf.hasFDerivAt, fun u δ => by
    rw [ContinuousLinearMap.comp_apply, ← ContinuousLinearMap.adjoint_inner_right,
      slopeCLM_adjoint, hf.inner_eq]⟩

/-- If `F y z` applies `φ c` to `(y c, z c)` at every coordinate and `φ c` has the derivative
`d₀ c • fst + d₁ c • snd` at `(f x c, g x c)`, then `fun y => F (f y) (g y)` has the derivative
`(slopeCLM d₀).comp Df + (slopeCLM d₁).comp Dg` at `x` and pulls back through the slope maps. -/
theorem pointwise₂ (F : Tensor ℝ s → Tensor ℝ s → Tensor ℝ s) (φ : s.Coord → ℝ × ℝ → ℝ)
    {d₀ d₁ : s.Coord → ℝ} (hF : ∀ y z c, F y z c = φ c (y c, z c))
    (hφ : ∀ c, HasFDerivAt (φ c)
      (d₀ c • ContinuousLinearMap.fst ℝ ℝ ℝ + d₁ c • ContinuousLinearMap.snd ℝ ℝ ℝ)
      (f x c, g x c))
    (hf : HasPullbackAt f Df bf x) (hg : HasPullbackAt g Dg bg x) :
    HasPullbackAt (fun y => F (f y) (g y)) ((slopeCLM d₀).comp Df + (slopeCLM d₁).comp Dg)
      (fun δ => bf (slopeCLM d₀ δ) + bg (slopeCLM d₁ δ)) x := by
  refine ⟨?_, fun u δ => ?_⟩
  · exact ((pointwise₂_hasFDerivAt (fun p => F p.fst p.snd) φ d₀ d₁ (f x) (g x)
      (fun p c => hF p.fst p.snd c) hφ).comp x (toLp_prod hf hg).hasFDerivAt).congr_fderiv
      (ContinuousLinearMap.ext fun u => rfl)
  · rw [_root_.add_apply, inner_add_left, inner_add_right, ContinuousLinearMap.comp_apply,
      ContinuousLinearMap.comp_apply,
      ← ContinuousLinearMap.adjoint_inner_right (slopeCLM d₀),
      ← ContinuousLinearMap.adjoint_inner_right (slopeCLM d₁), slopeCLM_adjoint,
      slopeCLM_adjoint, hf.inner_eq, hg.inner_eq]

/-- A real function applied at every entry, with derivative `φ' c` at the entry `f x c`. -/
@[fun_prop]
theorem map {φ : ℝ → ℝ} {φ' : s.Coord → ℝ} (hφ : ∀ c, HasDerivAt φ (φ' c) (f x c))
    (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => Tensor.map φ (f y)) ((slopeCLM φ').comp Df)
      (fun δ => bf (slopeCLM φ' δ)) x :=
  pointwise _ (fun _ => φ) (fun _ _ => Tensor.map_apply _ _ _) hφ hf

/-- A binary real function with a fixed right operand `z`, with derivative `φ' c` in its left
argument at `f x c`. -/
@[fun_prop]
theorem zipWith_left {φ : ℝ → ℝ → ℝ} (z : Tensor ℝ s) {φ' : s.Coord → ℝ}
    (hφ : ∀ c, HasDerivAt (fun a => φ a (z c)) (φ' c) (f x c)) (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => Tensor.zipWith φ (f y) z) ((slopeCLM φ').comp Df)
      (fun δ => bf (slopeCLM φ' δ)) x :=
  pointwise _ (fun c a => φ a (z c)) (fun _ _ => zipWith_apply _ _ _ _) hφ hf

/-- A binary real function with a fixed left operand `z`, with derivative `φ' c` in its right
argument at `f x c`. -/
@[fun_prop]
theorem zipWith_right {φ : ℝ → ℝ → ℝ} (z : Tensor ℝ s) {φ' : s.Coord → ℝ}
    (hφ : ∀ c, HasDerivAt (fun a => φ (z c) a) (φ' c) (f x c)) (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => Tensor.zipWith φ z (f y)) ((slopeCLM φ').comp Df)
      (fun δ => bf (slopeCLM φ' δ)) x :=
  pointwise _ (fun c a => φ (z c) a) (fun _ _ => zipWith_apply _ _ _ _) hφ hf

/-- The entrywise product of two maps, by the product rule at every entry. -/
@[fun_prop]
theorem mul (hf : HasPullbackAt f Df bf x) (hg : HasPullbackAt g Dg bg x) :
    HasPullbackAt (fun y => f y * g y)
      ((slopeCLM fun c => g x c).comp Df + (slopeCLM fun c => f x c).comp Dg)
      (fun δ => bf (δ * g x) + bg (δ * f x)) x := by
  refine ⟨?_, fun u δ => ?_⟩
  · have hc : ∀ c, HasFDerivAt (fun y => (f y * g y) c)
        ((g x c) • (coordinateCLM c).comp Df + (f x c) • (coordinateCLM c).comp Dg) x := by
      intro c
      have h := ((coordinateCLM c).hasFDerivAt.comp x hf.hasFDerivAt).mul
        ((coordinateCLM c).hasFDerivAt.comp x hg.hasFDerivAt)
      simp only [Function.comp_def, coordinateCLM_apply] at h
      refine (h.congr_fderiv ?_).congr_of_eventuallyEq (Filter.Eventually.of_forall fun y => ?_)
      · rw [add_comm]
      · simp only [Tensor.mul_apply, Pi.mul_apply]
    refine (hasFDerivAt_of_coordinates _ _ x hc).congr_fderiv ?_
    refine ContinuousLinearMap.ext fun v => Rep.ext fun c => ?_
    simp [slopeCLM_apply, assemble_apply]
  · rw [_root_.add_apply, inner_add_left, inner_add_right, ← hf.inner_eq, ← hg.inner_eq]
    simp only [ContinuousLinearMap.comp_apply, inner_eq_sum, slopeCLM_apply, Tensor.mul_apply]
    congr 1 <;> exact Finset.sum_congr rfl fun c _ => by ring

/-- The entrywise quotient of two maps where every entry of the denominator `g x` is nonzero. -/
@[fun_prop]
theorem div (hg0 : ∀ c, g x c ≠ 0) (hf : HasPullbackAt f Df bf x) (hg : HasPullbackAt g Dg bg x) :
    HasPullbackAt (fun y => f y / g y)
      ((slopeCLM fun c => (g x c)⁻¹).comp Df + (slopeCLM fun c => -(f x c / g x c ^ 2)).comp Dg)
      (fun δ => bf (slopeCLM (fun c => (g x c)⁻¹) δ) +
        bg (slopeCLM (fun c => -(f x c / g x c ^ 2)) δ)) x := by
  have hinv := pointwise (fun y => Tensor.map (·⁻¹) y) (fun _ a => a⁻¹)
    (fun y c => Tensor.map_apply _ _ _) (fun c => hasDerivAt_inv (hg0 c)) hg
  refine ((hf.mul hinv).congr_deriv ?_ fun δ => ?_).congr_of_eventuallyEq
    (Filter.Eventually.of_forall fun y => Rep.ext fun c => ?_)
  · refine ContinuousLinearMap.ext fun v => Rep.ext fun c => ?_
    simp only [_root_.add_apply, ContinuousLinearMap.comp_apply, Tensor.add_apply, slopeCLM_apply,
      Tensor.map_apply]
    ring
  · congr 2 <;> refine Rep.ext fun c => ?_ <;>
      simp only [slopeCLM_apply, Tensor.mul_apply, Tensor.map_apply] <;> ring
  · simp only [Tensor.div_apply, Tensor.mul_apply, Tensor.map_apply, div_eq_mul_inv]

/-- Reshaping to a shape with as many entries, pulled back by reshaping back. -/
@[fun_prop]
theorem reshape (h : s.size = t.size) (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => Tensor.reshape (f y) t h) ((reshapeCLM t h).comp Df)
      (fun δ => bf (Tensor.reshape δ s h.symm)) x := by
  simpa only [reshapeCLM_apply] using
    clm (reshapeCLM t h) (fun δ => Tensor.reshape δ s h.symm) (inner_reshape h) hf

/-- Casting along a shape equality, pulled back by casting back. -/
@[fun_prop]
theorem castShape (h : s = t) (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => Tensor.castShape (f y) h) ((castShapeCLM h).comp Df)
      (fun δ => bf (Tensor.castShape δ h.symm)) x := by
  simpa only [castShapeCLM_apply] using
    clm (castShapeCLM h) (fun δ => Tensor.castShape δ h.symm) (inner_castShape h) hf

/-- Broadcasting along `dims`, pulled back by `reduceBroadcast`. -/
@[fun_prop]
theorem broadcast (dims : Fin s.rank → Fin t.rank) (hb : Shape.BroadcastDims s t dims)
    (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => Tensor.broadcast (f y) t dims hb) ((broadcastCLM t dims hb).comp Df)
      (fun δ => bf (reduceBroadcast dims hb δ)) x := by
  simpa only [broadcastCLM_apply] using
    clm (broadcastCLM t dims hb) (reduceBroadcast dims hb) (inner_broadcast dims hb) hf

/-- The sum of every entry, pulled back by filling. -/
@[fun_prop]
theorem tensor_sum (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => Tensor.sum (f y)) ((sumCLM s).comp Df)
      (fun δ => bf (Tensor.full s δ)) x := by
  simpa only [sumCLM_apply] using clm (sumCLM s) (Tensor.full s) (fun u δ => by
    rw [sumCLM_apply, Real.inner_apply, inner_full_right]) hf

/-- Filling a tensor with a real number, pulled back by the sum. -/
@[fun_prop]
theorem full {f : E → ℝ} {Df : E →L[ℝ] ℝ} {bf : ℝ → E} (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => Tensor.full s (f y)) ((fullCLM s).comp Df)
      (fun δ => bf (Tensor.sum δ)) x := by
  simpa only [fullCLM_apply] using clm (fullCLM s) Tensor.sum (fun u δ => by
    rw [fullCLM_apply, real_inner_comm δ, inner_full_right, Real.inner_apply, mul_comm]) hf

/-- The scalar tensor of a real number, pulled back by reading its entry. -/
@[fun_prop]
theorem scalar {f : E → ℝ} {Df : E →L[ℝ] ℝ} {bf : ℝ → E} (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => Tensor.scalar (f y)) (scalarCLM.comp Df)
      (fun δ => bf (Tensor.item δ)) x := by
  simpa only [scalarCLM_apply] using clm scalarCLM Tensor.item (fun u δ => by
    rw [scalarCLM_apply, inner_eq_sum, Fintype.sum_unique, Real.inner_apply]
    show Tensor.scalar u PUnit.unit * δ PUnit.unit = u * δ PUnit.unit
    rw [scalar_apply]) hf

/-- Scaling every entry by a fixed real `c`, with derivative `c • Df`, pulled back by scaling the
cotangent by `c`. -/
@[fun_prop]
theorem scale_const (c : ℝ) (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => Tensor.scale (f y) c) (c • Df) (fun δ => bf (Tensor.scale δ c)) x := by
  simp only [scale_eq_smul]
  exact ⟨hf.hasFDerivAt.const_smul c, fun u δ => by
    rw [_root_.smul_apply, real_inner_smul_left, ← real_inner_smul_right, hf.inner_eq]⟩

/-- Scaling every entry of a map by a real-valued map `c`, by the product rule, pulled back as
`bc ⟪f x, δ⟫ + bf (Tensor.scale δ (c x))`. -/
@[fun_prop]
theorem scale {c : E → ℝ} {Dc : E →L[ℝ] ℝ} {bc : ℝ → E} (hf : HasPullbackAt f Df bf x)
    (hc : HasPullbackAt c Dc bc x) :
    HasPullbackAt (fun y => Tensor.scale (f y) (c y)) (c x • Df + Dc.smulRight (f x))
      (fun δ => bc ⟪f x, δ⟫ + bf (Tensor.scale δ (c x))) x := by
  simp only [scale_eq_smul]
  exact smul hc hf

/-- The value of a scalar tensor, read as a coordinate, pulled back by the scalar tensor of the
cotangent. -/
@[fun_prop]
theorem item {f : E → Tensor ℝ .scalar} {Df : E →L[ℝ] Tensor ℝ .scalar}
    {bf : Tensor ℝ .scalar → E} (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => Tensor.item (f y)) ((coordinateCLM PUnit.unit).comp Df)
      (fun δ => bf (Tensor.scalar δ)) x :=
  clm (coordinateCLM PUnit.unit) Tensor.scalar (fun u δ => by
    rw [coordinateCLM_apply, inner_eq_sum, Fintype.sum_unique]
    show ⟪u PUnit.unit, δ⟫ = u PUnit.unit * Tensor.scalar δ PUnit.unit
    simp only [scalar_apply, Real.inner_apply, mul_comm]) hf

/-- The entry at `c`, pulled back by the tensor that is the cotangent at `c` and zero
elsewhere. -/
@[fun_prop]
theorem get (c : s.Coord) (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => f y c) ((coordinateCLM c).comp Df)
      (fun δ => bf (Rep.ofFn fun c' => if c' = c then δ else 0)) x :=
  clm (coordinateCLM c) (fun δ => Rep.ofFn fun c' => if c' = c then δ else 0) (fun u δ => by
    simp only [coordinateCLM_apply, inner_eq_sum, Rep.get_ofFn, mul_ite, mul_zero,
      Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte, Real.inner_apply]) hf

/-- A binary real function `φ` of the entries of two maps, where at every coordinate `c` the map
`q ↦ φ q.1 q.2` has the derivative `φ₀ c • fst + φ₁ c • snd` at `(u c, v c)` for `u = f x` and
`v = g x`. The values are given as equations, which the discharger solves after simplifying `f x`
and `g x`, so that `fun_prop` meets the derivative of `φ` at the simplified values. -/
@[fun_prop]
theorem zipWith {φ : ℝ → ℝ → ℝ} {φ₀ φ₁ : s.Coord → ℝ} {u v : Tensor ℝ s} (hu : f x = u)
    (hv : g x = v)
    (hφ : ∀ c, HasFDerivAt (fun q : ℝ × ℝ => φ q.1 q.2)
      (φ₀ c • ContinuousLinearMap.fst ℝ ℝ ℝ + φ₁ c • ContinuousLinearMap.snd ℝ ℝ ℝ) (u c, v c))
    (hf : HasPullbackAt f Df bf x) (hg : HasPullbackAt g Dg bg x) :
    HasPullbackAt (fun y => Tensor.zipWith φ (f y) (g y))
      ((slopeCLM φ₀).comp Df + (slopeCLM φ₁).comp Dg)
      (fun δ => bf (slopeCLM φ₀ δ) + bg (slopeCLM φ₁ δ)) x := by
  subst hu hv
  exact pointwise₂ (Tensor.zipWith φ) (fun _ q => φ q.1 q.2) (fun _ _ _ => zipWith_apply _ _ _ _)
    hφ hf hg

/-- Softmax along `dim` after a map `f` with a pullback: the derivative is the fiberwise softmax
derivative at `f x` after the derivative of `f`, and the pullback applies the stored softmax
pullback at `f x` first. -/
@[fun_prop]
theorem softmax (dim : Fin s.rank) (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => Tensor.softmax (f y) dim)
      ((fiberwiseCLM (Shape.Coord.axisEquiv dim (shape := s)) fun r =>
        Vec.softmaxDerivCLM (fiberCLM (Shape.Coord.axisEquiv dim (shape := s)) r (f x))).comp Df)
      (fun δ => bf (Tensor.softmaxBackward (f x) δ dim)) x :=
  (hasPullbackAt_softmax dim (f x)).comp hf

end HasPullbackAt

end TorchLean

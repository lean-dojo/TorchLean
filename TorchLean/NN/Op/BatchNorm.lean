/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Norm.BatchNorm
public import TorchLean.NN.Op.Certificates
public import TorchLean.Tensor.Op.Table.Normalization
public import TorchLean.NN.Op.Pointwise
public import TorchLean.NN.Tactic.OpDeriv
public import TorchLean.Tensor.Movement.BroadcastAdjoint
public import Mathlib.Analysis.InnerProductSpace.ProdL2

/-!
# Certificates of batch normalization

With the running statistics and epsilon fixed, the forward of `Op.batchNormRunning` is affine in
each of its input, its scale and its bias. Jointly in the three, on their L2 product, its
derivative is the sum of the slope map on the input, the scale tangent map and the broadcast of
the bias. The three backward components form the adjoint of that derivative, and the forward-mode
rule applies it to the tangents (`batchNormRunning_jvp_eq_fderiv`). The scale and bias gradients
are sums over every axis but `dim`, and that sum is the pullback of broadcasting along `dim`
(`reduce_otherAxes_eq_reduceBroadcast`). The real root of a negative number is zero, and real
division by zero is zero, so the denominator needs no hypothesis.

With batch statistics, `Op.batchNorm` normalizes each channel by the mean and variance of its own
entries. Where the spatial shape has a single position or every channel has a nonzero sum of its
variance and epsilon, its forward is differentiable in the input, the scale and the bias together,
and the three backward components are the adjoint of that derivative
(`Op.batchNorm_prod_backward_eq_adjoint`). At every other input it is not differentiable. In the
input alone the condition on each channel weakens to a nonzero sum or a zero scale
(`Op.batchNorm_input_backward_eq_adjoint`). In the scale and the bias, alone or together, the
forward is affine at every input and every epsilon.
-/

@[expose] public section

open scoped RealInnerProductSpace

namespace TorchLean

open TorchLean.Tensor
open Op (slopeCLM)

variable {s : Shape}

/-! ## The scale tangent map -/

/-- The tangent map of `batchNormRunning` in its scale: the broadcast tangent times the normalized
input `normalized`. -/
noncomputable def BatchNorm.runningScaleCLM (dim : Fin s.rank) (normalized : Tensor ℝ s) :
    Tensor ℝ [s.extent dim] →L[ℝ] Tensor ℝ s :=
  LinearMap.toContinuousLinearMap
    { toFun := fun v => normalized * Tensor.broadcast v s _ (s.broadcastDims_axis dim)
      map_add' := fun u v => Rep.ext fun c => by
        simp only [Tensor.mul_apply, broadcast_apply, Tensor.add_apply, mul_add]
      map_smul' := fun a v => Rep.ext fun c => by
        simp only [Tensor.mul_apply, broadcast_apply, Rep.smul_apply, RingHom.id_apply,
          smul_eq_mul]
        ring }

/-- The scale tangent map multiplies the normalized input by the broadcast tangent. -/
@[simp] theorem BatchNorm.runningScaleCLM_apply (dim : Fin s.rank) (normalized : Tensor ℝ s)
    (v : Tensor ℝ [s.extent dim]) :
    runningScaleCLM dim normalized v =
      normalized * Tensor.broadcast v s _ (s.broadcastDims_axis dim) :=
  rfl

/-- The adjoint of the scale tangent map sums the cotangent times the normalized input over every
axis but `dim`. -/
theorem BatchNorm.runningScaleCLM_adjoint_apply (dim : Fin s.rank) (normalized δ : Tensor ℝ s) :
    (runningScaleCLM dim normalized).adjoint δ =
      castShape (Tensor.reduce .sum (δ * normalized) (s.otherAxes [dim]) false)
        (Shape.eraseAxes_otherAxes_singleton s dim) := by
  apply ext_inner_left ℝ
  intro v
  rw [ContinuousLinearMap.adjoint_inner_right, runningScaleCLM_apply,
    ← inner_broadcast_axis, inner_eq_sum, inner_eq_sum]
  exact Finset.sum_congr rfl fun c _ => by simp only [Tensor.mul_apply]; ring

/-! ## Input, scale and bias together -/

/-- The tangent map of `batchNormRunning` in its input, scale and bias together, on their L2
product: the slope map of `spread gamma / std` on the input, plus the scale tangent map of
`normalized` on the scale, plus broadcasting on the bias. -/
noncomputable def BatchNorm.runningCLM (dim : Fin s.rank) (gamma : Tensor ℝ [s.extent dim])
    (normalized std : Tensor ℝ s) :
    WithLp 2 (Tensor ℝ s × WithLp 2 (Tensor ℝ [s.extent dim] × Tensor ℝ [s.extent dim])) →L[ℝ]
      Tensor ℝ s :=
  (slopeCLM fun c => Tensor.broadcast gamma s _ (s.broadcastDims_axis dim) c / std c).comp
      (WithLp.fstL 2 ℝ (Tensor ℝ s)
        (WithLp 2 (Tensor ℝ [s.extent dim] × Tensor ℝ [s.extent dim]))) +
    (runningScaleCLM dim normalized).comp
      ((WithLp.fstL 2 ℝ (Tensor ℝ [s.extent dim]) (Tensor ℝ [s.extent dim])).comp
        (WithLp.sndL 2 ℝ (Tensor ℝ s)
          (WithLp 2 (Tensor ℝ [s.extent dim] × Tensor ℝ [s.extent dim])))) +
    (broadcastCLM s _ (s.broadcastDims_axis dim)).comp
      ((WithLp.sndL 2 ℝ (Tensor ℝ [s.extent dim]) (Tensor ℝ [s.extent dim])).comp
        (WithLp.sndL 2 ℝ (Tensor ℝ s)
          (WithLp 2 (Tensor ℝ [s.extent dim] × Tensor ℝ [s.extent dim]))))

/-- The joint tangent map adds the input, scale and bias tangent maps at their components. -/
theorem BatchNorm.runningCLM_apply (dim : Fin s.rank) (gamma : Tensor ℝ [s.extent dim])
    (normalized std : Tensor ℝ s)
    (p : WithLp 2 (Tensor ℝ s × WithLp 2 (Tensor ℝ [s.extent dim] × Tensor ℝ [s.extent dim]))) :
    runningCLM dim gamma normalized std p =
      slopeCLM (fun c => Tensor.broadcast gamma s _ (s.broadcastDims_axis dim) c / std c) p.fst +
        runningScaleCLM dim normalized p.snd.fst +
        broadcastCLM s _ (s.broadcastDims_axis dim) p.snd.snd :=
  rfl

/-- The adjoint of the joint tangent map collects the adjoints of the three tangent maps. -/
theorem BatchNorm.runningCLM_adjoint_apply (dim : Fin s.rank) (gamma : Tensor ℝ [s.extent dim])
    (normalized std δ : Tensor ℝ s) :
    (runningCLM dim gamma normalized std).adjoint δ =
      WithLp.toLp 2 ((slopeCLM fun c =>
          Tensor.broadcast gamma s _ (s.broadcastDims_axis dim) c / std c).adjoint δ,
        WithLp.toLp 2 ((runningScaleCLM dim normalized).adjoint δ,
          (broadcastCLM s _ (s.broadcastDims_axis dim)).adjoint δ)) := by
  apply ext_inner_left ℝ
  intro q
  rw [ContinuousLinearMap.adjoint_inner_right, runningCLM_apply, inner_add_left,
    inner_add_left, WithLp.prod_inner_apply, WithLp.prod_inner_apply]
  simp only [ContinuousLinearMap.adjoint_inner_right, add_assoc]
  rfl

/-- With the running statistics and epsilon fixed, the forward of `Op.batchNormRunning` has the
joint tangent map as derivative in its input, scale and bias, on their L2 product. -/
theorem Op.batchNormRunning_prod_hasFDerivAt (dim : Fin s.rank) (x : Tensor ℝ s)
    (gamma beta mean var : Tensor ℝ [s.extent dim]) (epsilon : Tensor ℝ []) :
    HasFDerivAt (fun p : WithLp 2 (Tensor ℝ s ×
          WithLp 2 (Tensor ℝ [s.extent dim] × Tensor ℝ [s.extent dim])) =>
        (((Op.batchNormRunning dim).rule? (α := ℝ) |>.get rfl).forward
          (p.fst, p.snd.fst, p.snd.snd, mean, var, epsilon, ())).1)
      (BatchNorm.runningCLM dim gamma
        ((x - Tensor.broadcast mean s _ (s.broadcastDims_axis dim)) /
          Tensor.sqrt (Tensor.broadcast var s _ (s.broadcastDims_axis dim) + full s epsilon.item))
        (Tensor.sqrt (Tensor.broadcast var s _ (s.broadcastDims_axis dim) + full s epsilon.item)))
      (WithLp.toLp 2 (x, WithLp.toLp 2 (gamma, beta))) := by
  set std := Tensor.sqrt (Tensor.broadcast var s _ (s.broadcastDims_axis dim) +
    full s epsilon.item) with hstd
  set spreadMean := Tensor.broadcast mean s _ (s.broadcastDims_axis dim) with hmean
  set L := BatchNorm.runningCLM dim gamma ((x - spreadMean) / std) std with hL
  have hF : ∀ (p : WithLp 2 (Tensor ℝ s ×
        WithLp 2 (Tensor ℝ [s.extent dim] × Tensor ℝ [s.extent dim]))) (c : s.Coord),
      (((Op.batchNormRunning dim).rule? (α := ℝ) |>.get rfl).forward
          (p.fst, p.snd.fst, p.snd.snd, mean, var, epsilon, ())).1 c =
        (p.fst c - spreadMean c) * (std c)⁻¹ * p.snd.fst (Shape.Coord.get c dim, PUnit.unit) +
          p.snd.snd (Shape.Coord.get c dim, PUnit.unit) := by
    intro p c
    show Tensor.batchNorm p.fst dim p.snd.fst p.snd.snd (.running mean var) epsilon.item c = _
    rw [Tensor.batchNorm_running_eq]
    simp only [Tensor.add_apply, Tensor.mul_apply, Tensor.div_apply,
      Rep.hSub_apply, broadcast_axis_apply, getScalar_eq_apply, hstd, hmean,
      div_eq_mul_inv]
  have hc : ∀ c : s.Coord, HasFDerivAt (fun p : WithLp 2 (Tensor ℝ s ×
        WithLp 2 (Tensor ℝ [s.extent dim] × Tensor ℝ [s.extent dim])) =>
        (((Op.batchNormRunning dim).rule? (α := ℝ) |>.get rfl).forward
          (p.fst, p.snd.fst, p.snd.snd, mean, var, epsilon, ())).1 c)
      ((Tensor.coordinateCLM c).comp L)
      (WithLp.toLp 2 (x, WithLp.toLp 2 (gamma, beta))) := by
    intro c
    simp only [hF]
    have hx := (((Tensor.coordinateCLM c).comp
      (WithLp.fstL 2 ℝ (Tensor ℝ s) (WithLp 2 (Tensor ℝ [s.extent dim] ×
        Tensor ℝ [s.extent dim])))).hasFDerivAt
      (x := WithLp.toLp 2 (x, WithLp.toLp 2 (gamma, beta)))).sub_const (spreadMean c)
    have hg := ((Tensor.coordinateCLM
      ((Shape.Coord.get c dim, PUnit.unit) : Shape.Coord [s.extent dim])).comp
      ((WithLp.fstL 2 ℝ (Tensor ℝ [s.extent dim]) (Tensor ℝ [s.extent dim])).comp
        (WithLp.sndL 2 ℝ (Tensor ℝ s) (WithLp 2 (Tensor ℝ [s.extent dim] ×
          Tensor ℝ [s.extent dim]))))).hasFDerivAt
      (x := WithLp.toLp 2 (x, WithLp.toLp 2 (gamma, beta)))
    have hb := ((Tensor.coordinateCLM
      ((Shape.Coord.get c dim, PUnit.unit) : Shape.Coord [s.extent dim])).comp
      ((WithLp.sndL 2 ℝ (Tensor ℝ [s.extent dim]) (Tensor ℝ [s.extent dim])).comp
        (WithLp.sndL 2 ℝ (Tensor ℝ s) (WithLp 2 (Tensor ℝ [s.extent dim] ×
          Tensor ℝ [s.extent dim]))))).hasFDerivAt
      (x := WithLp.toLp 2 (x, WithLp.toLp 2 (gamma, beta)))
    refine (((hx.mul_const (std c)⁻¹).mul hg).add hb).congr_fderiv ?_
    ext dp
    simp only [hL, _root_.add_apply, _root_.smul_apply, ContinuousLinearMap.comp_apply,
      Tensor.coordinateCLM_apply, WithLp.fstL_apply, WithLp.sndL_apply, smul_eq_mul,
      BatchNorm.runningCLM_apply, slopeCLM_apply, BatchNorm.runningScaleCLM_apply,
      broadcastCLM_apply, Tensor.add_apply, Tensor.mul_apply, Tensor.div_apply,
      Rep.hSub_apply, broadcast_axis_apply, getScalar_eq_apply]
    simp only [WithLp.toLp_fst, WithLp.toLp_snd, div_eq_mul_inv]
    ring
  refine (Tensor.hasFDerivAt_of_coordinates _ _ _ hc).congr_fderiv ?_
  refine ContinuousLinearMap.ext fun dp => Rep.ext fun c => ?_
  rw [Tensor.assemble_apply, ContinuousLinearMap.comp_apply,
    Tensor.coordinateCLM_apply]

/-- With the running statistics and epsilon fixed, the joint tangent map is the derivative of
`Op.batchNormRunning` in its input, scale and bias, and the pullback to the three, as one point of
their L2 product, is its adjoint. -/
@[op_cert TorchLean.Op.batchNormRunning derivative 0 1 2]
theorem Op.batchNormRunning_prod_backward_eq_adjoint (dim : Fin s.rank) (x : Tensor ℝ s)
    (gamma beta mean var : Tensor ℝ [s.extent dim]) (epsilon : Tensor ℝ []) (δ : Tensor ℝ s) :
    HasFDerivAt (fun p : WithLp 2 (Tensor ℝ s ×
          WithLp 2 (Tensor ℝ [s.extent dim] × Tensor ℝ [s.extent dim])) =>
        (((Op.batchNormRunning dim).rule? (α := ℝ) |>.get rfl).forward
          (p.fst, p.snd.fst, p.snd.snd, mean, var, epsilon, ())).1)
      (BatchNorm.runningCLM dim gamma
        ((x - Tensor.broadcast mean s _ (s.broadcastDims_axis dim)) /
          Tensor.sqrt (Tensor.broadcast var s _ (s.broadcastDims_axis dim) + full s epsilon.item))
        (Tensor.sqrt (Tensor.broadcast var s _ (s.broadcastDims_axis dim) + full s epsilon.item)))
      (WithLp.toLp 2 (x, WithLp.toLp 2 (gamma, beta))) ∧
    WithLp.toLp 2
        ((((Op.batchNormRunning dim).rule? (α := ℝ) |>.get rfl).backward
          (x, gamma, beta, mean, var, epsilon, ()) (δ, ())).1,
        WithLp.toLp 2
          ((((Op.batchNormRunning dim).rule? (α := ℝ) |>.get rfl).backward
            (x, gamma, beta, mean, var, epsilon, ()) (δ, ())).2.1,
          (((Op.batchNormRunning dim).rule? (α := ℝ) |>.get rfl).backward
            (x, gamma, beta, mean, var, epsilon, ()) (δ, ())).2.2.1)) =
      (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ s ×
          WithLp 2 (Tensor ℝ [s.extent dim] × Tensor ℝ [s.extent dim])) =>
        (((Op.batchNormRunning dim).rule? (α := ℝ) |>.get rfl).forward
          (p.fst, p.snd.fst, p.snd.snd, mean, var, epsilon, ())).1)
        (WithLp.toLp 2 (x, WithLp.toLp 2 (gamma, beta)))).adjoint δ := by
  have h := batchNormRunning_prod_hasFDerivAt dim x gamma beta mean var epsilon
  refine ⟨h, ?_⟩
  have hinput : slopeCLM (fun c => Tensor.broadcast gamma s _ (s.broadcastDims_axis dim) c /
      (Tensor.sqrt (Tensor.broadcast var s _ (s.broadcastDims_axis dim) +
        full s epsilon.item)) c) δ =
      (δ * Tensor.broadcast gamma s _ (s.broadcastDims_axis dim)) /
        Tensor.sqrt (Tensor.broadcast var s _ (s.broadcastDims_axis dim) + full s epsilon.item) :=
    Rep.ext fun c => by
      simp only [slopeCLM_apply, Tensor.mul_apply, Tensor.div_apply]
      ring
  rw [h.fderiv, BatchNorm.runningCLM_adjoint_apply, slopeCLM_adjoint, hinput,
    BatchNorm.runningScaleCLM_adjoint_apply, broadcastCLM_adjoint_apply,
    ← reduce_otherAxes_eq_reduceBroadcast]
  rfl

/-- With the running statistics and epsilon fixed, the forward-mode rule of `Op.batchNormRunning`
applies its joint derivative in input, scale and bias to their tangents. -/
theorem Op.batchNormRunning_jvp_eq_fderiv (dim : Fin s.rank) (x : Tensor ℝ s)
    (gamma beta mean var : Tensor ℝ [s.extent dim]) (epsilon : Tensor ℝ []) (dx : Tensor ℝ s)
    (dgamma dbeta : Tensor ℝ [s.extent dim]) :
    ((((Op.batchNormRunning dim).rule? (α := ℝ) |>.get rfl).jvp?.get rfl)
        (x, gamma, beta, mean, var, epsilon, ()) (dx, dgamma, dbeta, (), (), (), ())).1 =
      fderiv ℝ (fun p : WithLp 2 (Tensor ℝ s ×
          WithLp 2 (Tensor ℝ [s.extent dim] × Tensor ℝ [s.extent dim])) =>
        (((Op.batchNormRunning dim).rule? (α := ℝ) |>.get rfl).forward
          (p.fst, p.snd.fst, p.snd.snd, mean, var, epsilon, ())).1)
        (WithLp.toLp 2 (x, WithLp.toLp 2 (gamma, beta)))
        (WithLp.toLp 2 (dx, WithLp.toLp 2 (dgamma, dbeta))) := by
  rw [(batchNormRunning_prod_hasFDerivAt dim x gamma beta mean var epsilon).fderiv]
  show ((dx * Tensor.broadcast gamma s _ (s.broadcastDims_axis dim)) /
        Tensor.sqrt (Tensor.broadcast var s _ (s.broadcastDims_axis dim) + full s epsilon.item) +
      ((x - Tensor.broadcast mean s _ (s.broadcastDims_axis dim)) /
        Tensor.sqrt (Tensor.broadcast var s _ (s.broadcastDims_axis dim) + full s epsilon.item)) *
        Tensor.broadcast dgamma s _ (s.broadcastDims_axis dim)) +
      Tensor.broadcast dbeta s _ (s.broadcastDims_axis dim) = _
  refine Rep.ext fun c => ?_
  simp only [BatchNorm.runningCLM_apply, slopeCLM_apply, BatchNorm.runningScaleCLM_apply,
    broadcastCLM_apply, WithLp.toLp_fst, WithLp.toLp_snd, Tensor.add_apply,
    Tensor.mul_apply, Tensor.div_apply, Rep.hSub_apply, div_eq_mul_inv]
  ring

/-! ## Batch normalization with batch statistics -/

-- The row broadcasts and row sums below read tensors of shape `[m]` at the shape
-- `[m, n].eraseIdx 1`. Unfolding `List.eraseIdx` in implicit arguments lets `rw` and the pullback
-- rules match them at either shape.
attribute [local implicit_reducible] List.eraseIdx

namespace Normalization

variable {m n : Nat}

/-- Over the reals, the row sums of an `m × n` matrix are the pullback of the row broadcast. -/
theorem rowSum_eq_reduceBroadcast (w : Tensor ℝ [m, n]) :
    rowSum w = reduceBroadcast _ (Shape.eraseIdx_broadcastDims [m, n] 1) w := by
  apply ext_inner_left ℝ
  intro v
  rw [← inner_broadcast, ← dot_eq_inner, ← dot_eq_inner]
  change Rep.dot (s := [m]) v (rowSum w) = Rep.dot (rowBroadcast (n := n) v) w
  rw [dot_vec_eq_sum, dot_mat_eq_sum]
  simp only [getScalar_rowSum, get2_rowBroadcast, Finset.mul_sum]

end Normalization

namespace BatchNorm

open Normalization

variable {channels : Nat} {sSpatial : Shape}

/-- Over the reals, at every epsilon, batch normalization along the leading axis with batch
statistics, flattened to a `[channels, sSpatial.size]` matrix, is the normalized matrix of
`normalizedRows` scaled and shifted row by row by the channel parameters. -/
theorem flat_batchNorm [Shape.WellFormed (.dim channels sSpatial)]
    (x : Tensor ℝ (.dim channels sSpatial)) (gamma beta : Tensor ℝ [channels]) (ε : ℝ) :
    Tensor.reshape (Tensor.batchNorm x ⟨0, Nat.succ_pos _⟩ gamma beta .batch ε) _
        (size_flat channels sSpatial) =
      (normalizedRows (flat x) ε false).1 * rowBroadcast (n := sSpatial.size) gamma +
        rowBroadcast (n := sSpatial.size) beta :=
  Tensor.ext_get2 fun c p => by
    rw [get2_flat_batchNorm, get2_add, get2_mul,
      normalizedRows_fst (size_pos_of_wellFormed_dim channels), get2_div, get2_rowBroadcast,
      get2_rowBroadcast, get2_rowBroadcast, get2_lnCentered, getScalar_lnStd]

/-- Over the reals, at every epsilon, batch normalization along the leading axis with batch
statistics is affine in the scale and the bias: it reshapes the normalized matrix of
`normalizedRows`, scaled and shifted row by row. -/
theorem batchNorm_eq_reshape [Shape.WellFormed (.dim channels sSpatial)]
    (x : Tensor ℝ (.dim channels sSpatial)) (gamma beta : Tensor ℝ [channels]) (ε : ℝ) :
    Tensor.batchNorm x ⟨0, Nat.succ_pos _⟩ gamma beta .batch ε =
      Tensor.reshape ((normalizedRows (flat x) ε false).1 *
          rowBroadcast (n := sSpatial.size) gamma +
        rowBroadcast (n := sSpatial.size) beta) _ (size_flat channels sSpatial).symm := by
  rw [← flat_batchNorm, Tensor.reshape_reshape_self]

/-- The scale gradient of `BatchNorm.backward` is the row sum of the flattened cotangent times the
normalized matrix of `normalizedRows`. -/
theorem backward_scaleGradient
    (x δ : Tensor ℝ (.dim channels sSpatial)) (gamma : Tensor ℝ [channels]) (ε : ℝ) :
    (BatchNorm.backward x gamma δ ε).scaleGradient =
      rowSum (Tensor.reshape δ _ (size_flat channels sSpatial) *
        (normalizedRows (flat x) ε false).1) :=
  rfl

/-- The bias gradient of `BatchNorm.backward` is the row sum of the flattened cotangent. -/
theorem backward_biasGradient
    (x δ : Tensor ℝ (.dim channels sSpatial)) (gamma : Tensor ℝ [channels]) (ε : ℝ) :
    (BatchNorm.backward x gamma δ ε).biasGradient =
      rowSum (Tensor.reshape δ _ (size_flat channels sSpatial)) :=
  rfl

/-- Flattening an input, scale and bias triple on their L2 product to the domain of
`forwardVec`, as a continuous linear map. -/
noncomputable def toDomainCLM :
    WithLp 2 (Tensor ℝ (.dim channels sSpatial) ×
        WithLp 2 (Tensor ℝ [channels] × Tensor ℝ [channels])) →L[ℝ]
      Domain channels sSpatial :=
  ((toVecₗᵢ (.dim channels sSpatial) :
      Tensor ℝ (.dim channels sSpatial) →L[ℝ] Vec (Shape.size (.dim channels sSpatial))).comp
    (WithLp.fstL 2 ℝ _ _)).prod
  (((toVecₗᵢ [channels] : Tensor ℝ [channels] →L[ℝ] Vec (Shape.size [channels])).comp
      ((WithLp.fstL 2 ℝ _ _).comp (WithLp.sndL 2 ℝ _ _))).prod
    ((toVecₗᵢ [channels] : Tensor ℝ [channels] →L[ℝ] Vec (Shape.size [channels])).comp
      ((WithLp.sndL 2 ℝ _ _).comp (WithLp.sndL 2 ℝ _ _))))

/-- `toDomainCLM` flattens each of the three components. -/
@[simp] theorem toDomainCLM_apply (p : WithLp 2 (Tensor ℝ (.dim channels sSpatial) ×
    WithLp 2 (Tensor ℝ [channels] × Tensor ℝ [channels]))) :
    toDomainCLM p = (toVec p.fst, toVec p.snd.fst, toVec p.snd.snd) :=
  rfl

/-- Where the spatial shape has a single position or every channel of `x` has a nonzero sum of its
variance and `ε`, batch normalization along the leading axis with batch statistics, as a map of
the input, the scale and the bias on their L2 product, has the derivative `derivative` read
through the flattenings. Its pullback there is given by the three gradients of
`BatchNorm.backward`. -/
theorem hasPullbackAt_prod [Shape.WellFormed (.dim channels sSpatial)] {ε : ℝ}
    (x : Tensor ℝ (.dim channels sSpatial)) (gamma beta : Tensor ℝ [channels])
    (h : sSpatial.size = 1 ∨
      ∀ c : Fin channels, TorchLean.var (Tensor.get2 (flat x) c) + ε ≠ 0) :
    HasPullbackAt (fun p : WithLp 2 (Tensor ℝ (.dim channels sSpatial) ×
          WithLp 2 (Tensor ℝ [channels] × Tensor ℝ [channels])) =>
        Tensor.batchNorm p.fst ⟨0, Nat.succ_pos _⟩ p.snd.fst p.snd.snd .batch ε)
      (((toVecₗᵢ (.dim channels sSpatial)).symm :
          Vec (Shape.size (.dim channels sSpatial)) →L[ℝ] Tensor ℝ (.dim channels sSpatial)).comp
        ((derivative (matVec (toVec x)) (toVec gamma) ε).comp toDomainCLM))
      (fun δ => WithLp.toLp 2 ((BatchNorm.backward x gamma δ ε).inputGradient,
        WithLp.toLp 2 ((BatchNorm.backward x gamma δ ε).scaleGradient,
          (BatchNorm.backward x gamma δ ε).biasGradient)))
      (WithLp.toLp 2 (x, WithLp.toLp 2 (gamma, beta))) := by
  refine ⟨?_, fun u δ => ?_⟩
  · have hD : HasFDerivAt (forwardVec (channels := channels) (sSpatial := sSpatial) ε)
        (derivative (matVec (toVec x)) (toVec gamma) ε)
        (toDomainCLM (WithLp.toLp 2 (x, WithLp.toLp 2 (gamma, beta)))) :=
      hasFDerivAt_forwardVec x gamma beta h
    refine (((toVecₗᵢ (.dim channels sSpatial)).symm : Vec (Shape.size (.dim channels sSpatial))
        →L[ℝ] Tensor ℝ (.dim channels sSpatial)).hasFDerivAt.comp _
          (hD.comp _ toDomainCLM.hasFDerivAt)).congr_of_eventuallyEq
      (Filter.Eventually.of_forall fun p => ?_)
    simp only [Function.comp_apply, toDomainCLM_apply, forwardVec, LinearIsometryEquiv.coe_coe'',
      toVecₗᵢ_symm_apply, ofVec_toVec]
  · simp only [ContinuousLinearMap.comp_apply, toDomainCLM_apply, LinearIsometryEquiv.coe_coe'']
    rw [derivative_eq_jvp x u.fst gamma u.snd.fst beta u.snd.snd, toVecₗᵢ_symm_apply,
      ofVec_toVec, ← dot_eq_inner, jvp_backward_adjoint x u.fst δ gamma u.snd.fst beta
        u.snd.snd ε]
    simp only [WithLp.prod_inner_apply, dot_eq_inner, add_assoc]
    rfl

/-- Where the spatial shape has a single position or every channel of `x` has a nonzero sum of its
variance and `ε` or a zero scale, batch normalization along the leading axis with batch statistics,
as a map of the input, has the derivative `derivative` on input tangents read through the
flattenings. Its pullback there is the input gradient of `BatchNorm.backward`. -/
theorem hasPullbackAt_input [Shape.WellFormed (.dim channels sSpatial)] {ε : ℝ}
    (x : Tensor ℝ (.dim channels sSpatial)) (gamma beta : Tensor ℝ [channels])
    (h : sSpatial.size = 1 ∨ ∀ c : Fin channels,
      TorchLean.var (Tensor.get2 (flat x) c) + ε ≠ 0 ∨ getScalar gamma c = 0) :
    HasPullbackAt (fun y : Tensor ℝ (.dim channels sSpatial) =>
        Tensor.batchNorm y ⟨0, Nat.succ_pos _⟩ gamma beta .batch ε)
      (((toVecₗᵢ (.dim channels sSpatial)).symm :
          Vec (Shape.size (.dim channels sSpatial)) →L[ℝ] Tensor ℝ (.dim channels sSpatial)).comp
        (((derivative (matVec (toVec x)) (toVec gamma) ε).comp
          (ContinuousLinearMap.inl ℝ _ _)).comp
          (toVecₗᵢ (.dim channels sSpatial) :
            Tensor ℝ (.dim channels sSpatial) →L[ℝ] Vec (Shape.size (.dim channels sSpatial)))))
      (fun δ => (BatchNorm.backward x gamma δ ε).inputGradient) x := by
  have h0 : toVec (0 : Tensor ℝ [channels]) = 0 := map_zero (toVecₗᵢ [channels])
  refine ⟨?_, fun u δ => ?_⟩
  · have hD : HasFDerivAt (fun w => forwardVec (channels := channels) (sSpatial := sSpatial) ε
          (w, toVec gamma, toVec beta))
        ((derivative (matVec (toVec x)) (toVec gamma) ε).comp (ContinuousLinearMap.inl ℝ _ _))
        (toVec x) := by
      rw [forwardVec_eq_closedForm]
      exact hasFDerivAt_closedForm_input (toVec x) (toVec gamma) (toVec beta)
        (h.imp_right fun h c => by simpa only [rowVarE_flat, getScalar_chan] using h c)
    have hV : HasFDerivAt (fun y : Tensor ℝ (.dim channels sSpatial) => toVec y)
        (toVecₗᵢ (.dim channels sSpatial) :
          Tensor ℝ (.dim channels sSpatial) →L[ℝ] Vec (Shape.size (.dim channels sSpatial))) x :=
      ContinuousLinearMap.hasFDerivAt (toVecₗᵢ (.dim channels sSpatial) :
        Tensor ℝ (.dim channels sSpatial) →L[ℝ] Vec (Shape.size (.dim channels sSpatial)))
    refine (((toVecₗᵢ (.dim channels sSpatial)).symm : Vec (Shape.size (.dim channels sSpatial))
        →L[ℝ] Tensor ℝ (.dim channels sSpatial)).hasFDerivAt.comp _
          (hD.comp x hV)).congr_of_eventuallyEq (Filter.Eventually.of_forall fun y => ?_)
    simp only [Function.comp_apply, forwardVec, LinearIsometryEquiv.coe_coe'',
      toVecₗᵢ_symm_apply, ofVec_toVec]
  · have hd : ((derivative (matVec (toVec x)) (toVec gamma) ε).comp
        (ContinuousLinearMap.inl ℝ _ _)) (toVec u) = toVec (jvp x u gamma 0 beta 0 ε) := by
      rw [← derivative_eq_jvp x u gamma 0 beta 0, h0]
      rfl
    simp only [ContinuousLinearMap.comp_apply, LinearIsometryEquiv.coe_coe'',
      toVecₗᵢ_apply] at hd ⊢
    rw [hd, toVecₗᵢ_symm_apply, ofVec_toVec, ← dot_eq_inner (jvp x u gamma 0 beta 0 ε) δ,
      jvp_backward_adjoint x u δ gamma 0 beta 0 ε]
    simp only [dot_eq_inner, inner_zero_left, add_zero]

/-- Where the spatial shape has a single position or every channel of `fx y` has a nonzero sum of
its variance and `ε`, batch normalization along the leading axis with batch statistics of three
maps into the input, the scale and the bias is pulled back by the three gradients of
`BatchNorm.backward` at their values. -/
@[fun_prop]
theorem _root_.TorchLean.HasPullbackAt.batchNorm {E : Type*} [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] [Shape.WellFormed (.dim channels sSpatial)] {ε : ℝ}
    {y : E} {fx : E → Tensor ℝ (.dim channels sSpatial)} {fg fb : E → Tensor ℝ [channels]}
    {Dx : E →L[ℝ] Tensor ℝ (.dim channels sSpatial)} {Dg Db : E →L[ℝ] Tensor ℝ [channels]}
    {bx : Tensor ℝ (.dim channels sSpatial) → E} {bg bb : Tensor ℝ [channels] → E}
    (h : sSpatial.size = 1 ∨
      ∀ c : Fin channels, TorchLean.var (Tensor.get2 (flat (fx y)) c) + ε ≠ 0)
    (hx : HasPullbackAt fx Dx bx y) (hg : HasPullbackAt fg Dg bg y)
    (hb : HasPullbackAt fb Db bb y) :
    HasPullbackAt (fun z => Tensor.batchNorm (fx z) ⟨0, Nat.succ_pos _⟩ (fg z) (fb z) .batch ε)
      ((((toVecₗᵢ (.dim channels sSpatial)).symm :
          Vec (Shape.size (.dim channels sSpatial)) →L[ℝ] Tensor ℝ (.dim channels sSpatial)).comp
        ((derivative (matVec (toVec (fx y))) (toVec (fg y)) ε).comp toDomainCLM)).comp
        ((WithLp.prodContinuousLinearEquiv 2 ℝ _ _).symm.toContinuousLinearMap.comp
          (Dx.prod ((WithLp.prodContinuousLinearEquiv 2 ℝ _ _).symm.toContinuousLinearMap.comp
            (Dg.prod Db)))))
      (fun δ => bx (BatchNorm.backward (fx y) (fg y) δ ε).inputGradient +
        (bg (BatchNorm.backward (fx y) (fg y) δ ε).scaleGradient +
          bb (BatchNorm.backward (fx y) (fg y) δ ε).biasGradient)) y :=
  HasPullbackAt.comp (x := y) (f := fun z => WithLp.toLp 2 (fx z, WithLp.toLp 2 (fg z, fb z)))
    (hasPullbackAt_prod (fx y) (fg y) (fb y) h)
    (HasPullbackAt.toLp_prod hx (HasPullbackAt.toLp_prod hg hb))

/-- For positive `ε`, batch normalization along the leading axis with batch statistics of three
maps into the input, the scale and the bias is pulled back by the three gradients of
`BatchNorm.backward` at their values. -/
@[fun_prop]
theorem _root_.TorchLean.HasPullbackAt.batchNorm_of_pos {E : Type*} [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] [Shape.WellFormed (.dim channels sSpatial)] {ε : ℝ} (hε : 0 < ε)
    {y : E} {fx : E → Tensor ℝ (.dim channels sSpatial)} {fg fb : E → Tensor ℝ [channels]}
    {Dx : E →L[ℝ] Tensor ℝ (.dim channels sSpatial)} {Dg Db : E →L[ℝ] Tensor ℝ [channels]}
    {bx : Tensor ℝ (.dim channels sSpatial) → E} {bg bb : Tensor ℝ [channels] → E}
    (hx : HasPullbackAt fx Dx bx y) (hg : HasPullbackAt fg Dg bg y)
    (hb : HasPullbackAt fb Db bb y) :
    HasPullbackAt (fun z => Tensor.batchNorm (fx z) ⟨0, Nat.succ_pos _⟩ (fg z) (fb z) .batch ε)
      ((((toVecₗᵢ (.dim channels sSpatial)).symm :
          Vec (Shape.size (.dim channels sSpatial)) →L[ℝ] Tensor ℝ (.dim channels sSpatial)).comp
        ((derivative (matVec (toVec (fx y))) (toVec (fg y)) ε).comp toDomainCLM)).comp
        ((WithLp.prodContinuousLinearEquiv 2 ℝ _ _).symm.toContinuousLinearMap.comp
          (Dx.prod ((WithLp.prodContinuousLinearEquiv 2 ℝ _ _).symm.toContinuousLinearMap.comp
            (Dg.prod Db)))))
      (fun δ => bx (BatchNorm.backward (fx y) (fg y) δ ε).inputGradient +
        (bg (BatchNorm.backward (fx y) (fg y) δ ε).scaleGradient +
          bb (BatchNorm.backward (fx y) (fg y) δ ε).biasGradient)) y :=
  HasPullbackAt.batchNorm (Or.inr fun _ => (add_pos_of_nonneg_of_pos (var_nonneg _) hε).ne') hx hg
    hb

end BatchNorm

/-- Where the spatial shape has a single position or every channel has a nonzero sum of its batch
variance and epsilon, the forward of `Op.batchNorm` is differentiable in its input, scale and bias
on their L2 product, and its backward is the adjoint of that derivative. This set is exact, since
at every other input the forward is not differentiable, even where the scale of the failing
channel is zero. -/
@[op_cert TorchLean.Op.batchNorm derivative 0 1 2]
theorem Op.batchNorm_prod_backward_eq_adjoint {channels : Nat} {spatial : Shape}
    (hWellFormed : (Shape.dim channels spatial).Positive)
    (x : Tensor ℝ (.dim channels spatial)) (gamma beta : Tensor ℝ [channels])
    (epsilon : Tensor ℝ [])
    (hrows : spatial.size = 1 ∨
      ∀ c : Fin channels, TorchLean.var (Tensor.get2 (BatchNorm.flat x) c) + epsilon.item ≠ 0)
    (δ : Tensor ℝ (.dim channels spatial)) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ (.dim channels spatial) ×
          WithLp 2 (Tensor ℝ [channels] × Tensor ℝ [channels])) =>
        (((Op.batchNorm hWellFormed).rule? (α := ℝ) |>.get rfl).forward
          (p.fst, p.snd.fst, p.snd.snd, epsilon, ())).1)
      (WithLp.toLp 2 (x, WithLp.toLp 2 (gamma, beta))) ∧
    WithLp.toLp 2
        ((((Op.batchNorm hWellFormed).rule? (α := ℝ) |>.get rfl).backward
          (x, gamma, beta, epsilon, ()) (δ, ())).1,
        WithLp.toLp 2
          ((((Op.batchNorm hWellFormed).rule? (α := ℝ) |>.get rfl).backward
            (x, gamma, beta, epsilon, ()) (δ, ())).2.1,
          (((Op.batchNorm hWellFormed).rule? (α := ℝ) |>.get rfl).backward
            (x, gamma, beta, epsilon, ()) (δ, ())).2.2.1)) =
      (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ (.dim channels spatial) ×
          WithLp 2 (Tensor ℝ [channels] × Tensor ℝ [channels])) =>
        (((Op.batchNorm hWellFormed).rule? (α := ℝ) |>.get rfl).forward
          (p.fst, p.snd.fst, p.snd.snd, epsilon, ())).1)
        (WithLp.toLp 2 (x, WithLp.toLp 2 (gamma, beta)))).adjoint δ := by
  have : Shape.WellFormed (.dim channels spatial) := ⟨hWellFormed⟩
  simp only [Op.rule?_batchNorm, Option.get_some]
  exact HasPullbackAt.differentiableAt_and (BatchNorm.hasPullbackAt_prod x gamma beta hrows) rfl

/-- At every epsilon, the forward of `Op.batchNorm` is differentiable in its scale, and the scale
component of its backward is the adjoint of that derivative. -/
@[op_cert TorchLean.Op.batchNorm derivative 1]
theorem Op.batchNorm_scale_backward_eq_adjoint {channels : Nat} {spatial : Shape}
    (hWellFormed : (Shape.dim channels spatial).Positive)
    (x : Tensor ℝ (.dim channels spatial)) (gamma beta : Tensor ℝ [channels])
    (epsilon : Tensor ℝ []) (δ : Tensor ℝ (.dim channels spatial)) :
    DifferentiableAt ℝ (fun g : Tensor ℝ [channels] =>
        (((Op.batchNorm hWellFormed).rule? (α := ℝ) |>.get rfl).forward
          (x, g, beta, epsilon, ())).1) gamma ∧
    (((Op.batchNorm hWellFormed).rule? (α := ℝ) |>.get rfl).backward
        (x, gamma, beta, epsilon, ()) (δ, ())).2.1 =
      (fderiv ℝ (fun g : Tensor ℝ [channels] =>
        (((Op.batchNorm hWellFormed).rule? (α := ℝ) |>.get rfl).forward
          (x, g, beta, epsilon, ())).1) gamma).adjoint δ := by
  have : Shape.WellFormed (.dim channels spatial) := ⟨hWellFormed⟩
  op_deriv [BatchNorm.batchNorm_eq_reshape, BatchNorm.backward_scaleGradient,
    Normalization.rowSum_eq_reduceBroadcast]

/-- At every epsilon, the forward of `Op.batchNorm` is differentiable in its bias, and the bias
component of its backward is the adjoint of that derivative. -/
@[op_cert TorchLean.Op.batchNorm derivative 2]
theorem Op.batchNorm_bias_backward_eq_adjoint {channels : Nat} {spatial : Shape}
    (hWellFormed : (Shape.dim channels spatial).Positive)
    (x : Tensor ℝ (.dim channels spatial)) (gamma beta : Tensor ℝ [channels])
    (epsilon : Tensor ℝ []) (δ : Tensor ℝ (.dim channels spatial)) :
    DifferentiableAt ℝ (fun b : Tensor ℝ [channels] =>
        (((Op.batchNorm hWellFormed).rule? (α := ℝ) |>.get rfl).forward
          (x, gamma, b, epsilon, ())).1) beta ∧
    (((Op.batchNorm hWellFormed).rule? (α := ℝ) |>.get rfl).backward
        (x, gamma, beta, epsilon, ()) (δ, ())).2.2.1 =
      (fderiv ℝ (fun b : Tensor ℝ [channels] =>
        (((Op.batchNorm hWellFormed).rule? (α := ℝ) |>.get rfl).forward
          (x, gamma, b, epsilon, ())).1) beta).adjoint δ := by
  have : Shape.WellFormed (.dim channels spatial) := ⟨hWellFormed⟩
  op_deriv [BatchNorm.batchNorm_eq_reshape, BatchNorm.backward_biasGradient,
    Normalization.rowSum_eq_reduceBroadcast]

/-- At every input and every epsilon, the forward of `Op.batchNorm` is differentiable in its scale
and bias on their L2 product, and the scale and bias components of its backward are the adjoint of
that derivative. -/
@[op_cert TorchLean.Op.batchNorm derivative 1 2]
theorem Op.batchNorm_scale_bias_backward_eq_adjoint {channels : Nat} {spatial : Shape}
    (hWellFormed : (Shape.dim channels spatial).Positive)
    (x : Tensor ℝ (.dim channels spatial)) (gamma beta : Tensor ℝ [channels])
    (epsilon : Tensor ℝ []) (δ : Tensor ℝ (.dim channels spatial)) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ [channels] × Tensor ℝ [channels]) =>
        (((Op.batchNorm hWellFormed).rule? (α := ℝ) |>.get rfl).forward
          (x, p.fst, p.snd, epsilon, ())).1)
      (WithLp.toLp 2 (gamma, beta)) ∧
    WithLp.toLp 2
        ((((Op.batchNorm hWellFormed).rule? (α := ℝ) |>.get rfl).backward
          (x, gamma, beta, epsilon, ()) (δ, ())).2.1,
        (((Op.batchNorm hWellFormed).rule? (α := ℝ) |>.get rfl).backward
          (x, gamma, beta, epsilon, ()) (δ, ())).2.2.1) =
      (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ [channels] × Tensor ℝ [channels]) =>
        (((Op.batchNorm hWellFormed).rule? (α := ℝ) |>.get rfl).forward
          (x, p.fst, p.snd, epsilon, ())).1)
        (WithLp.toLp 2 (gamma, beta))).adjoint δ := by
  have : Shape.WellFormed (.dim channels spatial) := ⟨hWellFormed⟩
  op_deriv [BatchNorm.batchNorm_eq_reshape, BatchNorm.backward_scaleGradient,
    BatchNorm.backward_biasGradient, Normalization.rowSum_eq_reduceBroadcast]

/-- Where the spatial shape has a single position or every channel has a nonzero sum of its batch
variance and epsilon or a zero scale, the forward of `Op.batchNorm` is differentiable in its input,
and the input component of its backward is the adjoint of that derivative. This set is exact, since
at every other input the forward is not continuous in the input. -/
@[op_cert TorchLean.Op.batchNorm derivative 0]
theorem Op.batchNorm_input_backward_eq_adjoint {channels : Nat} {spatial : Shape}
    (hWellFormed : (Shape.dim channels spatial).Positive)
    (x : Tensor ℝ (.dim channels spatial)) (gamma beta : Tensor ℝ [channels])
    (epsilon : Tensor ℝ [])
    (hrows : spatial.size = 1 ∨ ∀ c : Fin channels,
      TorchLean.var (Tensor.get2 (BatchNorm.flat x) c) + epsilon.item ≠ 0 ∨
        Tensor.getScalar gamma c = 0)
    (δ : Tensor ℝ (.dim channels spatial)) :
    DifferentiableAt ℝ (fun y : Tensor ℝ (.dim channels spatial) =>
        (((Op.batchNorm hWellFormed).rule? (α := ℝ) |>.get rfl).forward
          (y, gamma, beta, epsilon, ())).1) x ∧
    (((Op.batchNorm hWellFormed).rule? (α := ℝ) |>.get rfl).backward
        (x, gamma, beta, epsilon, ()) (δ, ())).1 =
      (fderiv ℝ (fun y : Tensor ℝ (.dim channels spatial) =>
        (((Op.batchNorm hWellFormed).rule? (α := ℝ) |>.get rfl).forward
          (y, gamma, beta, epsilon, ())).1) x).adjoint δ := by
  have : Shape.WellFormed (.dim channels spatial) := ⟨hWellFormed⟩
  simp only [Op.rule?_batchNorm, Option.get_some]
  exact HasPullbackAt.differentiableAt_and (BatchNorm.hasPullbackAt_input x gamma beta hrows) rfl

end TorchLean

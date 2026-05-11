/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape
public import NN.Spec.Layers.Utils

/-!
# Normalization layers (spec layer)

This file collects a few normalization operators used throughout TorchLean's spec/model code.

The common pattern is:

- compute per-axis statistics (mean / variance or RMS),
- normalize with an `epsilon` for numerical stability,
- optionally apply an affine transform (`gamma`, `beta`) like PyTorch does.

## References (papers + PyTorch behavior)

- LayerNorm: Ba et al., "Layer Normalization" (2016): https://arxiv.org/abs/1607.06450
- BatchNorm: Ioffe, Szegedy, "Batch Normalization" (2015): https://arxiv.org/abs/1502.03167
- GroupNorm: Wu, He, "Group Normalization" (2018): https://arxiv.org/abs/1803.08494
- RMSNorm: Zhang, Sennrich, "Root Mean Square Layer Normalization" (2019):
  https://arxiv.org/abs/1910.07467
- WeightNorm: Salimans, Kingma, "Weight Normalization" (2016): https://arxiv.org/abs/1602.07868

- PyTorch LayerNorm: https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
- PyTorch BatchNorm2d: https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
-/

@[expose] public section


namespace Spec
open Tensor
open Numbers

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-- Core normalization routine with explicit broadcast proofs.

This is the shared “math step” behind normalization layers:

`y = ((x - mean) / sqrt(variance + ε)) * gamma + beta`.
-/
def normalizeCore
  (s s_mean s_var s_gamma s_beta : Shape)
  (epsilon : α)
  (x : Tensor α s)
  (mean : Tensor α s_mean)
  (variance : Tensor α s_var)
  (gamma : Tensor α s_gamma)
  (beta : Tensor α s_beta)
  (cb_mean : Shape.CanBroadcastTo s_mean s)
  (cb_var : Shape.CanBroadcastTo s_var s)
  (cb_gamma : Shape.CanBroadcastTo s_gamma s)
  (cb_beta : Shape.CanBroadcastTo s_beta s) : Tensor α s :=

  let mean_broadcast := broadcastTo cb_mean mean
  let variance_broadcast := broadcastTo cb_var variance
  let gamma_broadcast := broadcastTo cb_gamma gamma
  let beta_broadcast := broadcastTo cb_beta beta

  let centered := subSpec x mean_broadcast
  let std := sqrtSpec (addSpec variance_broadcast (fill epsilon s))
  let normalized := divSpec centered std
  addSpec (mulSpec normalized gamma_broadcast) beta_broadcast


/-
  Layer Normalization
  Normalizes along the last dimension
-/
/-- LayerNorm over the last dimension of a `(seqLen, embedDim)` tensor.

Uses `epsilon` (default `Numbers.epsilon`) for numerical stability in the denominator.
-/
def layerNorm {seqLen embedDim : Nat}
  (x : Tensor α (.dim seqLen (.dim embedDim .scalar)))
  (gamma : Tensor α (.dim embedDim .scalar))
  (beta : Tensor α (.dim embedDim .scalar))
  (h_seq_pos : seqLen > 0 := by norm_num)
  (h_embed_pos : embedDim > 0 := by norm_num)
  (epsilon : α := Numbers.epsilon) :
  Tensor α (.dim seqLen (.dim embedDim .scalar)) :=

  -- Compute mean along last dimension (dim = 1)
  let _ : Shape.WellFormed (.dim seqLen (.dim embedDim .scalar)) :=
  ⟨⟨h_seq_pos, ⟨h_embed_pos, trivial⟩⟩⟩

  let s := Shape.dim seqLen (Shape.dim embedDim Shape.scalar)
  let h_rank : Shape.rank s > 0 := by simp [s, Shape.rank]
  let h_valid : Shape.valid_axis_inst (Shape.rank s - 1) s := Shape.validAxisLastAuto h_rank

  let mean := reduceMeanLastGeneralWf x h_rank h_valid

  have h₁ : (Shape.dim seqLen (Shape.dim embedDim Shape.scalar)).rank = 2 := by
    simp [Shape.rank]

  have h4 : Shape.CanBroadcastTo (.dim seqLen .scalar) (.dim seqLen (.dim embedDim .scalar)) := by
    apply Shape.CanBroadcastTo.dim_eq
    apply Shape.CanBroadcastTo.expand_dims
    exact Shape.CanBroadcastTo.scalar_to_any .scalar

  let mean_broadcast := broadcastTo h4 mean
  let centered := subSpec x mean_broadcast

  have inst : Shape.valid_axis_inst (Shape.rank (.dim seqLen (.dim embedDim .scalar)) - 1) (.dim
    seqLen (.dim embedDim .scalar)) := by
    apply Shape.validAxisLastAuto
    simp [h₁]

  let varianceRaw := reduceVarLastGeneral centered inst
  -- Clamp variance to be nonnegative so `std` is always defined/bounded away from 0 even for
  -- approximate numeric contexts (Float/NF) where small negative variance can occur.
  let variance := maxSpec varianceRaw (fill 0 (.dim seqLen .scalar))

  let std := sqrtSpec (addSpec variance (fill epsilon (.dim seqLen .scalar)))

  let std_broadcast := broadcastTo h4 std
  let normalized := divSpec centered std_broadcast

  have h5 : Shape.CanBroadcastTo (.dim embedDim .scalar) (.dim seqLen (.dim embedDim .scalar)) := by
    apply Shape.CanBroadcastTo.expand_dims
    apply Shape.CanBroadcastTo.dim_eq
    apply Shape.CanBroadcastTo.scalar_to_any .scalar

  let gamma_broadcast := broadcastTo h5 gamma
  let beta_broadcast := broadcastTo h5 beta
  let scaled := mulSpec normalized gamma_broadcast
  addSpec scaled beta_broadcast

/-- Backward/VJP for `layerNorm` (returns `(dx, dGamma, dBeta)`). -/
def layerNormBackward
  {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
  (x : Tensor α (.dim seqLen (.dim embedDim .scalar)))
  (gamma : Tensor α (.dim embedDim .scalar))
  (_beta : Tensor α (.dim embedDim .scalar))
  (grad_output : Tensor α (.dim seqLen (.dim embedDim .scalar)))
  (epsilon : α := Numbers.epsilon) :
  (Tensor α (.dim seqLen (.dim embedDim .scalar)) ×  -- ∂L/∂x
   Tensor α (.dim embedDim .scalar) ×                 -- ∂L/∂gamma
   Tensor α (.dim embedDim .scalar)) :=               -- ∂L/∂beta := sum of grad_output

  -- Forward recomputation
  let _ : Shape.WellFormed (.dim seqLen (.dim embedDim .scalar)) :=
  ⟨⟨h_seq_pos, ⟨h_embed_pos, trivial⟩⟩⟩

  let s := Shape.dim seqLen (Shape.dim embedDim Shape.scalar)
  let h_rank : Shape.rank s > 0 := by simp [s, Shape.rank]
  let h_valid : Shape.valid_axis_inst (Shape.rank s - 1) s := Shape.validAxisLastAuto h_rank

  let mean := reduceMeanLastGeneralWf x h_rank h_valid

  have h₁ : (Shape.dim seqLen (Shape.dim embedDim Shape.scalar)).rank = 2 := by
    simp [Shape.rank]

  have h₂ : shapeAfterSum (Shape.dim seqLen (Shape.dim embedDim Shape.scalar)) 1
            = Shape.dim seqLen Shape.scalar := by
    rw [shape_after_sum_dim_1_alt seqLen embedDim]

  have h3 : shapeAfterSum (Shape.dim seqLen (Shape.dim embedDim Shape.scalar)) ((Shape.dim seqLen
    (Shape.dim embedDim Shape.scalar)).rank - 1)
          = Shape.dim seqLen Shape.scalar := by
    rw [h₁]
    rw [h₂]

  have h4 : Shape.CanBroadcastTo (.dim seqLen .scalar) (.dim seqLen (.dim embedDim .scalar)) := by
    apply Shape.CanBroadcastTo.dim_eq
    apply Shape.CanBroadcastTo.expand_dims
    exact Shape.CanBroadcastTo.scalar_to_any .scalar

  let mean_broadcast := broadcastTo h4 mean
  let centered := subSpec x mean_broadcast

  have inst : Shape.valid_axis_inst (Shape.rank (.dim seqLen (.dim embedDim .scalar)) - 1) (.dim
    seqLen (.dim embedDim .scalar)) := by
    apply Shape.validAxisLastAuto
    simp [h₁]

  let varianceRaw := reduceVarLastGeneral centered inst
  let variance := maxSpec varianceRaw (fill 0 (.dim seqLen .scalar))
  let std := sqrtSpec (addSpec variance (fill epsilon (.dim seqLen .scalar)))
  let inv_std := divSpec (fill 1 (.dim seqLen .scalar)) std

  let std_broadcast := broadcastTo h4 std
  let norm := divSpec centered std_broadcast

  have h5 : Shape.CanBroadcastTo (.dim embedDim .scalar) (.dim seqLen (.dim embedDim .scalar)) := by
    apply Shape.CanBroadcastTo.expand_dims
    apply Shape.CanBroadcastTo.dim_eq
    apply Shape.CanBroadcastTo.scalar_to_any .scalar

  -- `gamma` and `beta` have shape `[embedDim]` and are shared across all `seqLen` positions, so
  -- their
  -- gradients sum over the sequence dimension (axis 0).
  letI : Shape.valid_axis_inst 0 (Shape.dim seqLen (Shape.dim embedDim Shape.scalar)) :=
    Shape.validAxisInstZeroAlt2 h_seq_pos

  let grad_beta := reduceSumAuto 0 grad_output
  let grad_gamma := reduceSumAuto 0 (mulSpec grad_output norm)

  -- ∂L/∂x: standard LayerNorm VJP, using per-position statistics over the feature dimension.
  --
  -- Let `N = embedDim`, `xhat = norm`, and `dy = grad_output`.
  -- With `dy_gamma = dy ⊙ gamma`, the closed form is:
  --
  --   dx = inv_std ⊙ ( dy_gamma
  --                    - mean(dy_gamma)
  --                    - xhat ⊙ mean(dy_gamma ⊙ xhat) )
  --
  -- where the `mean` is taken over the last dimension (features) for each sequence position.
  let gamma_broadcast := broadcastTo h5 gamma
  let inv_std_broadcast := broadcastTo h4 inv_std
  let dy_gamma := mulSpec grad_output gamma_broadcast

  let sum_dy_gamma := reduceSumLastGeneral dy_gamma
  -- We interpret `embedDim` as the feature-count `N` in the closed-form LayerNorm VJP.
  --
  -- Note: this relies on the `Context`'s `Coe Nat α` behaving sensibly (in particular, that
  -- `(embedDim : α)` is nonzero when `embedDim > 0`). This holds for TorchLean's shipped backends
  -- (Float/ℝ/IEEE32Exec), but for exotic saturating casts a specialized scalar interface may be
  -- preferable.
  let N : α := (embedDim : α)
  let mean_dy_gamma := divSpec sum_dy_gamma (fill N (.dim seqLen .scalar))

  let sum_dy_gamma_xhat := reduceSumLastGeneral (mulSpec dy_gamma norm)
  let mean_dy_gamma_xhat := divSpec sum_dy_gamma_xhat (fill N (.dim seqLen .scalar))

  let mean_dy_gamma_broadcast := broadcastTo h4 mean_dy_gamma
  let mean_dy_gamma_xhat_broadcast := broadcastTo h4 mean_dy_gamma_xhat

  let grad_x :=
    mulSpec inv_std_broadcast
      (subSpec (subSpec dy_gamma mean_dy_gamma_broadcast) (mulSpec norm
        mean_dy_gamma_xhat_broadcast))

  (grad_x, grad_gamma, grad_beta)

/--
Forward-mode JVP for `layerNorm`.

For each sequence position, LayerNorm is the map
`y = gamma ⊙ xhat + beta` with `xhat = (x - mean(x)) / sqrt(var(x)+eps)`.
The input tangent is normalized by the standard closed form

`dxhat = inv_std ⊙ (dx - mean(dx) - xhat ⊙ mean(dx ⊙ xhat))`,

and affine-parameter tangents contribute `xhat ⊙ dgamma + dbeta`. This is the forward-mode
counterpart of the closed-form VJP above and follows the same clamped-variance convention as the
forward pass.
-/
def layerNormJvp
  {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
  (x tangent : Tensor α (.dim seqLen (.dim embedDim .scalar)))
  (gamma dgamma _beta dbeta : Tensor α (.dim embedDim .scalar))
  (epsilon : α := Numbers.epsilon) :
  Tensor α (.dim seqLen (.dim embedDim .scalar)) :=

  let _ : Shape.WellFormed (.dim seqLen (.dim embedDim .scalar)) :=
  ⟨⟨h_seq_pos, ⟨h_embed_pos, trivial⟩⟩⟩

  let s := Shape.dim seqLen (Shape.dim embedDim Shape.scalar)
  let h_rank : Shape.rank s > 0 := by simp [s, Shape.rank]
  let h_valid : Shape.valid_axis_inst (Shape.rank s - 1) s := Shape.validAxisLastAuto h_rank

  let mean := reduceMeanLastGeneralWf x h_rank h_valid

  have h₁ : (Shape.dim seqLen (Shape.dim embedDim Shape.scalar)).rank = 2 := by
    simp [Shape.rank]

  have h4 : Shape.CanBroadcastTo (.dim seqLen .scalar) (.dim seqLen (.dim embedDim .scalar)) := by
    apply Shape.CanBroadcastTo.dim_eq
    apply Shape.CanBroadcastTo.expand_dims
    exact Shape.CanBroadcastTo.scalar_to_any .scalar

  let mean_broadcast := broadcastTo h4 mean
  let centered := subSpec x mean_broadcast

  have inst : Shape.valid_axis_inst (Shape.rank (.dim seqLen (.dim embedDim .scalar)) - 1) (.dim
    seqLen (.dim embedDim .scalar)) := by
    apply Shape.validAxisLastAuto
    simp [h₁]

  let varianceRaw := reduceVarLastGeneral centered inst
  let variance := maxSpec varianceRaw (fill 0 (.dim seqLen .scalar))
  let std := sqrtSpec (addSpec variance (fill epsilon (.dim seqLen .scalar)))
  let inv_std := divSpec (fill 1 (.dim seqLen .scalar)) std
  let inv_std_broadcast := broadcastTo h4 inv_std
  let norm := mulSpec centered inv_std_broadcast

  let sum_tangent := reduceSumLastGeneral tangent
  let N : α := (embedDim : α)
  let mean_tangent := divSpec sum_tangent (fill N (.dim seqLen .scalar))

  let sum_tangent_norm := reduceSumLastGeneral (mulSpec tangent norm)
  let mean_tangent_norm := divSpec sum_tangent_norm (fill N (.dim seqLen .scalar))

  let mean_tangent_broadcast := broadcastTo h4 mean_tangent
  let mean_tangent_norm_broadcast := broadcastTo h4 mean_tangent_norm

  let dnorm :=
    mulSpec inv_std_broadcast
      (subSpec (subSpec tangent mean_tangent_broadcast)
        (mulSpec norm mean_tangent_norm_broadcast))

  have h5 : Shape.CanBroadcastTo (.dim embedDim .scalar) (.dim seqLen (.dim embedDim .scalar)) := by
    apply Shape.CanBroadcastTo.expand_dims
    apply Shape.CanBroadcastTo.dim_eq
    apply Shape.CanBroadcastTo.scalar_to_any .scalar

  let gamma_broadcast := broadcastTo h5 gamma
  let dgamma_broadcast := broadcastTo h5 dgamma
  let dbeta_broadcast := broadcastTo h5 dbeta
  addSpec (addSpec (mulSpec dnorm gamma_broadcast) (mulSpec norm dgamma_broadcast))
    dbeta_broadcast
/-
  Group Normalization
  Normalizes within groups of channels
-/
 /--
GroupNorm for channel-last tensors `(batch, height, width, channels)`.

PyTorch analogy: `torch.nn.GroupNorm(num_groups=groups, num_channels=channels)` applied per sample.
The mean/variance are computed over *both* the spatial dimensions and the channels within each
group, then an affine transform is applied per channel via `gamma` and `beta`.

This operator is useful in settings where BatchNorm's dependence on batch statistics is awkward
(small batches, verification, or when you want purely per-sample behavior).
-/
def groupNorm
  {batchSize height width channels groups : Nat}
  (x : Tensor α (.dim batchSize (.dim height (.dim width (.dim channels .scalar)))))
  (gamma : Tensor α (.dim channels .scalar))
  (beta : Tensor α (.dim channels .scalar))
  (h_b : batchSize > 0 := by norm_num)
  (h_h : height > 0 := by norm_num)
  (h_w : width > 0 := by norm_num)
  (h_c : channels > 0 := by norm_num)
  (h_g : groups > 0 := by norm_num)
  (h_ge : channels ≥ groups)
  (h_div : channels % groups = 0)
  (epsilon : α := Numbers.epsilon) :
  Tensor α (.dim batchSize (.dim height (.dim width (.dim channels .scalar)))) :=
  let channelsPerGroup := channels / groups
  let hw : Nat := height * width

  have h_hw : hw > 0 := Nat.mul_pos h_h h_w
  have h_cpg_pos : channelsPerGroup > 0 := by
    unfold channelsPerGroup
    apply Nat.div_pos
    exact h_ge
    exact h_g

  have h_ch : channels = groups * channelsPerGroup := by
    rw [← Nat.mod_add_div channels groups, h_div]
    unfold channelsPerGroup
    simp

  -- Reshape to `[batch, groups, channelsPerGroup, height*width]` so:
  -- - statistics are per `(batch, group)`,
  -- - `gamma`/`beta` stay aligned to the channel axis.
  let s_orig := Shape.dim batchSize (.dim height (.dim width (.dim channels .scalar)))
  let s_reshaped := Shape.dim batchSize (.dim groups (.dim channelsPerGroup (.dim hw .scalar)))

  have _wf : Shape.WellFormed s_orig := ⟨⟨h_b, ⟨h_h, ⟨h_w, ⟨h_c, trivial⟩⟩⟩⟩⟩
  have wf_r : Shape.WellFormed s_reshaped := ⟨⟨h_b, ⟨h_g, ⟨h_cpg_pos, ⟨h_hw, trivial⟩⟩⟩⟩⟩

  have h_reshape : Shape.size s_orig = Shape.size s_reshaped := by
    -- batch * height * width * channels = batch * groups * channelsPerGroup * (height*width)
    -- and `channels = groups * channelsPerGroup`.
    simp [s_orig, s_reshaped, Shape.size, hw, h_ch, Nat.mul_left_comm, Nat.mul_comm]

  let x4 := reshapeSpec x h_reshape

  -- Mean over spatial axis, then mean over channels within the group.
  letI : Shape.WellFormed s_reshaped := wf_r
  let rank_x4 : Shape.rank s_reshaped > 0 := by simp [s_reshaped, Shape.rank]
  let ax_last_x4 : Shape.valid_axis_inst (Shape.rank s_reshaped - 1) s_reshaped :=
    Shape.validAxisLastAuto rank_x4
  let mean_hw := reduceMeanLastGeneralWf x4 rank_x4 ax_last_x4

  let s_mean_hw := Shape.dim batchSize (.dim groups (.dim channelsPerGroup .scalar))
  letI : Shape.WellFormed s_mean_hw := ⟨⟨h_b, ⟨h_g, ⟨h_cpg_pos, trivial⟩⟩⟩⟩

  -- Help typeclass search: `reduce_mean_last_general_wf` wants `WellFormed` on the *input* shape.
  -- Here the input is `shape_after_sum s_reshaped (rank-1)`, which simplifies to `s_mean_hw`.
  have h_after_sum :
      shapeAfterSum s_reshaped (Shape.rank s_reshaped - 1) = s_mean_hw := by
    simp [s_reshaped, s_mean_hw, Shape.rank]
  have wf_after_sum :
      Shape.WellFormed (shapeAfterSum s_reshaped (Shape.rank s_reshaped - 1)) := by
    simpa [h_after_sum] using (show Shape.WellFormed s_mean_hw from inferInstance)
  letI : Shape.WellFormed (shapeAfterSum s_reshaped (Shape.rank s_reshaped - 1)) := wf_after_sum

  let rank_mean_hw : Shape.rank s_mean_hw > 0 := by simp [s_mean_hw, Shape.rank]
  let ax_last_mean_hw : Shape.valid_axis_inst (Shape.rank s_mean_hw - 1) s_mean_hw :=
    Shape.validAxisLastAuto rank_mean_hw
  let mean := reduceMeanLastGeneralWf mean_hw rank_mean_hw ax_last_mean_hw

  have cb_mean : Shape.CanBroadcastTo (Shape.dim batchSize (.dim groups .scalar)) s_reshaped := by
    apply Shape.CanBroadcastTo.dim_eq
    apply Shape.CanBroadcastTo.dim_eq
    apply Shape.CanBroadcastTo.expand_dims
    apply Shape.CanBroadcastTo.expand_dims
    exact Shape.CanBroadcastTo.scalar_to_any .scalar

  let mean_b := broadcastTo cb_mean mean
  let centered := subSpec x4 mean_b

  -- Variance: mean of squared centered values across both axes.
  let centered_sq := mulSpec centered centered
  let var_hw := reduceMeanLastGeneralWf centered_sq rank_x4 ax_last_x4
  let variance := reduceMeanLastGeneralWf var_hw rank_mean_hw ax_last_mean_hw
  let variance := maxSpec variance (fill 0 (Shape.dim batchSize (.dim groups .scalar)))
  let std := sqrtSpec (addSpec variance (fill epsilon (Shape.dim batchSize (.dim groups
    .scalar))))
  let std_b := broadcastTo cb_mean std

  let normalized := divSpec centered std_b

  -- Reshape `gamma`/`beta` to `[groups, channelsPerGroup]` and broadcast across batch and spatial.
  have h_param_reshape : Shape.size (Shape.dim channels .scalar) = Shape.size (Shape.dim groups
    (.dim channelsPerGroup .scalar)) := by
    rw [h_ch]
    simp [Shape.size]
  let gamma2 := reshapeSpec gamma h_param_reshape
  let beta2 := reshapeSpec beta h_param_reshape

  have cb_params : Shape.CanBroadcastTo (Shape.dim groups (.dim channelsPerGroup .scalar))
    s_reshaped := by
    apply Shape.CanBroadcastTo.expand_dims
    apply Shape.CanBroadcastTo.dim_eq
    apply Shape.CanBroadcastTo.dim_eq
    apply Shape.CanBroadcastTo.expand_dims
    exact Shape.CanBroadcastTo.scalar_to_any .scalar

  let gamma_b := broadcastTo cb_params gamma2
  let beta_b := broadcastTo cb_params beta2

  let y4 := addSpec (mulSpec normalized gamma_b) beta_b
  reshapeSpec y4 h_reshape.symm

/-
  Normalize along a specific dimension
-/
/--
Normalize along a chosen axis `dim` of a tensor `x`, using per-element affine parameters `gamma`
and `beta` of the same shape as `x`.

This is a "generic building block" that is handy in specs; it is closer to the raw math than to a
single PyTorch module. Most named normalizations (LayerNorm, GroupNorm, BatchNorm) are special
cases of this pattern with a specific choice of axis set and parameter shape.
-/
def normalizeAlongDim
  {s : Shape}
  (x : Tensor α s)
  (gamma : Tensor α s)
  (beta : Tensor α s)
  (dim : Nat)
  (h_valid : Shape.valid_axis_inst dim s)
  (h_wf : Shape.WellFormed s)
  (epsilon : α := Numbers.epsilon)
  : Tensor α s :=

  -- mean shape: shape_after_sum s dimension
  let mean := reduceMeanAuto dim h_valid x

  have h_can_broadcast : Shape.CanBroadcastTo (shapeAfterSum s dim) s :=
    shapeAfterSumBroadcastBack dim h_valid h_wf

  let mean_broadcast := broadcastTo h_can_broadcast mean
  -- center x by subtracting mean (broadcasted)
  let centered := subSpec x mean_broadcast

  -- variance shape: shape_after_sum s dimension (same shape as mean)
  let variance := reduceVarAuto dim h_valid centered

  -- broadcast variance to s for addition of epsilon and sqrt
  let variance_broadcast := broadcastTo h_can_broadcast variance

  -- compute std = sqrt(variance + epsilon)
  let std := sqrtSpec (addSpec variance_broadcast (fill epsilon s))
  -- normalize centered by dividing by std (broadcasted)
  let normalized := divSpec centered std
  -- multiply by gamma (shape s) and add beta (shape s)
  let result := addSpec (mulSpec normalized gamma) beta
  result

/-
  RMS Normalization
  Normalizes using RMS instead of mean/variance
-/
/--
RMSNorm over the last dimension of a `(seqLen, embedDim)` tensor.

Compared to LayerNorm, RMSNorm skips subtracting the mean and normalizes by:

`rms(x) = sqrt(mean(x^2) + eps)`.

This shows up in many Transformer-style models as a cheaper alternative to LayerNorm.
-/
def rmsNorm {seqLen embedDim : Nat}
  (x : Tensor α (.dim seqLen (.dim embedDim .scalar)))
  (gamma : Tensor α (.dim embedDim .scalar))
  (h_seq_pos : seqLen > 0 := by norm_num)
  (h_embed_pos : embedDim > 0 := by norm_num)
  (epsilon : α := Numbers.epsilon) :
  Tensor α (.dim seqLen (.dim embedDim .scalar)) :=
  -- Compute RMS along last dimension
  let squared := squareSpec x

  -- Proofs
  let _ : Shape.WellFormed (.dim seqLen (.dim embedDim .scalar)) :=
  ⟨⟨h_seq_pos, ⟨h_embed_pos, trivial⟩⟩⟩
  let s := Shape.dim seqLen (Shape.dim embedDim Shape.scalar)
  let h_rank : Shape.rank s > 0 := by simp [s, Shape.rank]
  let h_valid : Shape.valid_axis_inst (Shape.rank s - 1) s := Shape.validAxisLastAuto h_rank

  -- Compute mean along last dimension (dim = 1)
  let mean_squared := reduceMeanLastGeneralWf squared h_rank h_valid
  let rms := sqrtSpec (addSpec mean_squared (fill epsilon (.dim seqLen .scalar)))
  -- shape: [seqLen]

  have h_can_broadcast : Shape.CanBroadcastTo (Shape.dim seqLen Shape.scalar) (Shape.dim seqLen
    (.dim embedDim .scalar)) := by
    apply Shape.CanBroadcastTo.dim_eq
    apply Shape.CanBroadcastTo.expand_dims
    apply Shape.CanBroadcastTo.scalar_to_any .scalar

  -- Normalize by RMS
  let rms_broadcast := broadcastTo h_can_broadcast rms
  let normalized := divSpec x rms_broadcast

  have h_gamma_broadcast : Shape.CanBroadcastTo (Shape.dim embedDim Shape.scalar) (Shape.dim seqLen
    (.dim embedDim .scalar)) := by
    apply Shape.CanBroadcastTo.expand_dims
    apply Shape.CanBroadcastTo.dim_eq
    apply Shape.CanBroadcastTo.scalar_to_any .scalar

  -- Scale
  let gamma_broadcast := broadcastTo h_gamma_broadcast gamma
  let result := mulSpec normalized gamma_broadcast
  result

/-
  Weight Normalization
  Normalizes the weight matrix
-/
/--
WeightNorm for a dense weight matrix `(outDim, inDim)`.

This implements the "normalize weight vectors then scale" idea:

- normalize each output row by its L2 norm,
- then rescale by `gamma` (one scalar per output row).

PyTorch analogy: weight normalization is typically applied as a parametrization of a module's
weights rather than as a standalone tensor operator.
-/
def weightNorm {inDim outDim : Nat}
  (weight : Tensor α (.dim outDim (.dim inDim .scalar)))
  (gamma : Tensor α (.dim outDim .scalar))
  (h_out_pos : outDim > 0 := by norm_num)
  (h_in_pos : inDim > 0 := by norm_num)
  (epsilon : α := Numbers.epsilon) :
  Tensor α (.dim outDim (.dim inDim .scalar)) :=

  -- Compute L2 norm of each row
  let squared := squareSpec weight

  -- Register well-formedness and axis

  let s := Shape.dim outDim (Shape.dim inDim Shape.scalar)
  let _ : Shape.WellFormed s := ⟨⟨h_out_pos, ⟨h_in_pos, trivial⟩⟩⟩

  -- `reduce_sum_last_general` infers axis validity via `valid_axis_inst`.
  let h_rank : Shape.rank s > 0 := by simp [s, Shape.rank]
  letI : Shape.valid_axis_inst (Shape.rank s - 1) s := Shape.validAxisLastAuto h_rank

  -- Use reduce_sum_last to compute sum along the last dimension (inDim) for each row
  let rowSums := reduceSumLastGeneral squared
  let rowNorms := sqrtSpec (addSpec rowSums (fill epsilon (.dim outDim .scalar)))
  -- shape: [outDim]

  have h_can_broadcast : Shape.CanBroadcastTo (Shape.dim outDim Shape.scalar) s := by
    apply Shape.CanBroadcastTo.dim_eq
    apply Shape.CanBroadcastTo.expand_dims
    apply Shape.CanBroadcastTo.scalar_to_any .scalar

  -- Normalize weights
  let rowNorms_broadcast := broadcastTo h_can_broadcast rowNorms
  let normalized := divSpec weight rowNorms_broadcast

  -- Scale
  let gamma_broadcast := broadcastTo h_can_broadcast gamma
  let result := mulSpec normalized gamma_broadcast
  result

/-
  Batch Normalization (spec layer)

  TorchLean models *pure* (stateless) BatchNorm operators:

  - `batchNorm`: "training-mode" normalization using statistics computed from the current input
    (TorchLean does not model the running-statistics update),
  - `batchNorm_inference`: inference-time normalization using fixed running mean/variance.
-/

/--
Stateless BatchNorm for channel-first tensors of shape `.dim channels sSpatial`.

This computes per-channel mean/variance over the `sSpatial` axes and applies:

`y = ((x - mean) / sqrt(var + eps)) * gamma + beta`.

PyTorch analogy: `torch.nn.BatchNorm{1,2,3}d` in training mode on an input with batch size `N=1`.
TorchLean does **not** model the running-statistics update here.
-/
def batchNorm
  {channels : Nat} {sSpatial : Shape}
  (x : Tensor α (.dim channels sSpatial))
  (gamma : Tensor α (.dim channels .scalar))
  (beta : Tensor α (.dim channels .scalar))
  (epsilon : α := Numbers.epsilon)
  [Shape.WellFormed (.dim channels sSpatial)] :
  Tensor α (.dim channels sSpatial) :=
  let spatialSize : Nat := Shape.size sSpatial
  let s_flat : Shape := .dim channels (.dim spatialSize .scalar)
  have h_reshape : Shape.size (.dim channels sSpatial) = Shape.size s_flat := by
    simp [s_flat, spatialSize, Shape.size]
  let x2 : Tensor α s_flat := reshapeSpec x h_reshape
  have hwf_x : (Shape.dim channels sSpatial).wellFormed := Shape.WellFormed.proof
  have h_channels : channels > 0 := hwf_x.1
  have h_spatial_wf : sSpatial.wellFormed := hwf_x.2
  have h_spatialSize : spatialSize > 0 := by
    simpa [spatialSize] using Shape.size_pos_of_well_formed (s := sSpatial) h_spatial_wf
  letI : Shape.WellFormed s_flat := ⟨⟨h_channels, ⟨h_spatialSize, trivial⟩⟩⟩
  let h_rank : Shape.rank s_flat > 0 := by simp [s_flat, Shape.rank]
  let h_valid : Shape.valid_axis_inst (Shape.rank s_flat - 1) s_flat :=
    Shape.validAxisLastAuto h_rank
  let mean : Tensor α (.dim channels .scalar) := reduceMeanLastGeneral x2 h_valid
  have cb_flat : Shape.CanBroadcastTo (.dim channels .scalar) s_flat := by
    apply Shape.CanBroadcastTo.dim_eq
    exact Shape.CanBroadcastTo.scalar_to_any (.dim spatialSize .scalar)
  let centered := subSpec x2 (broadcastTo cb_flat mean)
  let centered_sq := mulSpec centered centered
  let varianceRaw : Tensor α (.dim channels .scalar) := reduceMeanLastGeneral centered_sq h_valid
  let variance := maxSpec varianceRaw (fill 0 (.dim channels .scalar))
  have cb : Shape.CanBroadcastTo (.dim channels .scalar) (.dim channels sSpatial) := by
    apply Shape.CanBroadcastTo.dim_eq
    exact Shape.CanBroadcastTo.scalar_to_any sSpatial
  normalizeCore
    (s := .dim channels sSpatial)
    (s_mean := .dim channels .scalar)
    (s_var := .dim channels .scalar)
    (s_gamma := .dim channels .scalar)
    (s_beta := .dim channels .scalar)
    (epsilon := epsilon)
    (x := x)
    (mean := mean)
    (variance := variance)
    (gamma := gamma)
    (beta := beta)
    (cb_mean := cb)
    (cb_var := cb)
    (cb_gamma := cb)
    (cb_beta := cb)

/--
Alias: per-sample normalization over spatial axes ("InstanceNorm-style").

The spec-level `batchNorm*` operators model the `N=1` case (no explicit batch axis and no running
statistics update). Many ML codebases refer to that behavior as *instance normalization*.

These aliases make that intent explicit without changing the existing API surface. -/
def instanceNorm
  {channels : Nat} {sSpatial : Shape}
  (x : Tensor α (.dim channels sSpatial))
  (gamma : Tensor α (.dim channels .scalar))
  (beta : Tensor α (.dim channels .scalar))
  (epsilon : α := Numbers.epsilon)
  [Shape.WellFormed (.dim channels sSpatial)] :
  Tensor α (.dim channels sSpatial) :=
  batchNorm (x := x) (gamma := gamma) (beta := beta) (epsilon := epsilon)

/-- Convenience wrapper: `batchNorm` specialized to a single channel-first image `(C,H,W)`. -/
def batchNorm2d
  {channels height width : Nat}
  (x : MultiChannelImage channels height width α)
  (gamma : Tensor α (.dim channels .scalar))
  (beta : Tensor α (.dim channels .scalar))
  (h_c : channels > 0 := by norm_num)
  (h_h : height > 0 := by norm_num)
  (h_w : width > 0 := by norm_num)
  (epsilon : α := Numbers.epsilon) :
  MultiChannelImage channels height width α :=
  letI : Shape.WellFormed (.dim channels (.dim height (.dim width .scalar))) :=
    ⟨⟨h_c, ⟨h_h, ⟨h_w, trivial⟩⟩⟩⟩
  batchNorm (x := x) (gamma := gamma) (beta := beta) (epsilon := epsilon)

/--
Forward-mode JVP for `batchNorm2d`.

TorchLean's stateless BatchNorm2d computes one set of statistics per channel over the spatial
grid. The input tangent therefore uses the same closed-form normalization differential as
LayerNorm, but with the mean taken over `(height,width)` for each channel:

`dxhat = inv_std * (dx - mean(dx) - xhat * mean(dx*xhat))`.

Affine tangents contribute `xhat * dgamma + dbeta` channel-wise.
-/
def batchNorm2dJvp
  {channels height width : Nat}
  (x tangent : MultiChannelImage channels height width α)
  (gamma dgamma _beta dbeta : Tensor α (.dim channels .scalar))
  (_h_c : channels > 0 := by norm_num)
  (_h_h : height > 0 := by norm_num)
  (_h_w : width > 0 := by norm_num)
  (epsilon : α := Numbers.epsilon) :
  MultiChannelImage channels height width α :=

  let spatial_size := height * width
  let nScalar : Tensor α Shape.scalar := Tensor.scalar (spatial_size : α)

  Tensor.dim (fun c =>
    let channel_data := getAtSpec x c
    let channel_tangent := getAtSpec tangent c
    let channel_gamma := getAtSpec gamma c
    let channel_dgamma := getAtSpec dgamma c
    let channel_dbeta := getAtSpec dbeta c

    let channel_sum :=
      (List.finRange height).foldl (fun acc_h i =>
        (List.finRange width).foldl (fun acc_w j =>
          if h_i : i < height then
            if h_j : j < width then
              addSpec acc_w (getAtSpec (getAtSpec channel_data ⟨i, h_i⟩) ⟨j, h_j⟩)
            else acc_w
          else acc_w
        ) acc_h
      ) (Tensor.scalar 0)
    let mean := divSpec channel_sum nScalar

    let variance_sum :=
      (List.finRange height).foldl (fun acc_h i =>
        (List.finRange width).foldl (fun acc_w j =>
          if h_i : i < height then
            if h_j : j < width then
              let val := getAtSpec (getAtSpec channel_data ⟨i, h_i⟩) ⟨j, h_j⟩
              let diff := subSpec val mean
              addSpec acc_w (mulSpec diff diff)
            else acc_w
          else acc_w
        ) acc_h
      ) (Tensor.scalar 0)

    let varianceRaw := divSpec variance_sum nScalar
    let variance := maxSpec varianceRaw (Tensor.scalar 0)
    let std := sqrtSpec (addSpec variance (Tensor.scalar epsilon))
    let inv_std := divSpec (Tensor.scalar 1) std

    let tangent_sum :=
      (List.finRange height).foldl (fun acc_h i =>
        (List.finRange width).foldl (fun acc_w j =>
          if h_i : i < height then
            if h_j : j < width then
              addSpec acc_w (getAtSpec (getAtSpec channel_tangent ⟨i, h_i⟩) ⟨j, h_j⟩)
            else acc_w
          else acc_w
        ) acc_h
      ) (Tensor.scalar 0)
    let mean_tangent := divSpec tangent_sum nScalar

    let tangent_norm_sum :=
      (List.finRange height).foldl (fun acc_h i =>
        (List.finRange width).foldl (fun acc_w j =>
          if h_i : i < height then
            if h_j : j < width then
              let val := getAtSpec (getAtSpec channel_data ⟨i, h_i⟩) ⟨j, h_j⟩
              let dx := getAtSpec (getAtSpec channel_tangent ⟨i, h_i⟩) ⟨j, h_j⟩
              let xHat := mulSpec (subSpec val mean) inv_std
              addSpec acc_w (mulSpec dx xHat)
            else acc_w
          else acc_w
        ) acc_h
      ) (Tensor.scalar 0)
    let mean_tangent_norm := divSpec tangent_norm_sum nScalar

    Tensor.dim (fun i =>
      Tensor.dim (fun j =>
        let val := getAtSpec (getAtSpec channel_data i) j
        let dx := getAtSpec (getAtSpec channel_tangent i) j
        let xHat := mulSpec (subSpec val mean) inv_std
        let dnorm :=
          mulSpec inv_std
            (subSpec (subSpec dx mean_tangent) (mulSpec xHat mean_tangent_norm))
        addSpec (addSpec (mulSpec dnorm channel_gamma) (mulSpec xHat channel_dgamma))
          channel_dbeta))
  )

/-- Alias for `batchNorm2d` (InstanceNorm-style per-image normalization). -/
def instanceNorm2d
  {channels height width : Nat}
  (x : MultiChannelImage channels height width α)
  (gamma : Tensor α (.dim channels .scalar))
  (beta : Tensor α (.dim channels .scalar))
  (h_c : channels > 0 := by norm_num)
  (h_h : height > 0 := by norm_num)
  (h_w : width > 0 := by norm_num)
  (epsilon : α := Numbers.epsilon) :
  MultiChannelImage channels height width α :=
  batchNorm2d (x := x) (gamma := gamma) (beta := beta) (h_c := h_c) (h_h := h_h) (h_w := h_w)
    (epsilon := epsilon)

/-- Convenience wrapper: `batchNorm` specialized to a `(C, L)` tensor (BatchNorm1d-style). -/
def batchNorm1d
  {channels length : Nat}
  (x : Tensor α (.dim channels (.dim length .scalar)))
  (gamma : Tensor α (.dim channels .scalar))
  (beta : Tensor α (.dim channels .scalar))
  (h_c : channels > 0 := by norm_num)
  (h_l : length > 0 := by norm_num)
  (epsilon : α := Numbers.epsilon) :
  Tensor α (.dim channels (.dim length .scalar)) :=
  letI : Shape.WellFormed (.dim channels (.dim length .scalar)) := ⟨⟨h_c, ⟨h_l, trivial⟩⟩⟩
  batchNorm (x := x) (gamma := gamma) (beta := beta) (epsilon := epsilon)

/-- Alias for `batchNorm1d` (InstanceNorm-style per-sample normalization). -/
def instanceNorm1d
  {channels length : Nat}
  (x : Tensor α (.dim channels (.dim length .scalar)))
  (gamma : Tensor α (.dim channels .scalar))
  (beta : Tensor α (.dim channels .scalar))
  (h_c : channels > 0 := by norm_num)
  (h_l : length > 0 := by norm_num)
  (epsilon : α := Numbers.epsilon) :
  Tensor α (.dim channels (.dim length .scalar)) :=
  batchNorm1d (x := x) (gamma := gamma) (beta := beta) (h_c := h_c) (h_l := h_l) (epsilon :=
    epsilon)

/-- Convenience wrapper: `batchNorm` specialized to a `(C, D, H, W)` tensor (BatchNorm3d-style). -/
def batchNorm3d
  {channels depth height width : Nat}
  (x : Tensor α (.dim channels (.dim depth (.dim height (.dim width .scalar)))))
  (gamma : Tensor α (.dim channels .scalar))
  (beta : Tensor α (.dim channels .scalar))
  (h_c : channels > 0 := by norm_num)
  (h_d : depth > 0 := by norm_num)
  (h_h : height > 0 := by norm_num)
  (h_w : width > 0 := by norm_num)
  (epsilon : α := Numbers.epsilon) :
  Tensor α (.dim channels (.dim depth (.dim height (.dim width .scalar)))) :=
  letI : Shape.WellFormed (.dim channels (.dim depth (.dim height (.dim width .scalar)))) :=
    ⟨⟨h_c, ⟨h_d, ⟨h_h, ⟨h_w, trivial⟩⟩⟩⟩⟩
  batchNorm (x := x) (gamma := gamma) (beta := beta) (epsilon := epsilon)

/-- Alias for `batchNorm3d` (InstanceNorm-style per-sample normalization). -/
def instanceNorm3d
  {channels depth height width : Nat}
  (x : Tensor α (.dim channels (.dim depth (.dim height (.dim width .scalar)))))
  (gamma : Tensor α (.dim channels .scalar))
  (beta : Tensor α (.dim channels .scalar))
  (h_c : channels > 0 := by norm_num)
  (h_d : depth > 0 := by norm_num)
  (h_h : height > 0 := by norm_num)
  (h_w : width > 0 := by norm_num)
  (epsilon : α := Numbers.epsilon) :
  Tensor α (.dim channels (.dim depth (.dim height (.dim width .scalar)))) :=
  batchNorm3d (x := x) (gamma := gamma) (beta := beta) (h_c := h_c) (h_d := h_d) (h_h := h_h)
    (h_w := h_w) (epsilon := epsilon)

-- Batch normalization backward pass for channel-first format
/--
Backward/VJP for `batchNorm2d`.

Returns `(dx, dGamma, dBeta)`. This matches the shape of gradients you expect from a PyTorch-style
BatchNorm2d, but note that our forward is the per-image variant (no explicit batch dimension and no
running statistics).
-/
def batchNorm2dBackward
  {channels height width : Nat}
  (x : MultiChannelImage channels height width α)
  (gamma : Tensor α (.dim channels .scalar))
  (grad_output : MultiChannelImage channels height width α)
  (_h_c : channels > 0 := by norm_num)
  (_h_h : height > 0 := by norm_num)
  (_h_w : width > 0 := by norm_num)
  (epsilon : α := Numbers.epsilon) :
  (MultiChannelImage channels height width α ×  -- ∂L/∂x
   Tensor α (.dim channels .scalar) ×           -- ∂L/∂gamma
   Tensor α (.dim channels .scalar)) :=         -- ∂L/∂beta

  let spatial_size := height * width

  -- Process each channel independently
  let grad_x := Tensor.dim (fun c =>
    let channel_data := getAtSpec x c
    let channel_grad := getAtSpec grad_output c
    let channel_gamma := getAtSpec gamma c

    -- Recompute forward pass statistics (in practice, these would be cached)
    let channel_sum :=
      (List.finRange height).foldl (fun acc_h i =>
        (List.finRange width).foldl (fun acc_w j =>
          if h_i : i < height then
            if h_j : j < width then
              addSpec acc_w (getAtSpec (getAtSpec channel_data ⟨i, h_i⟩) ⟨j, h_j⟩)
            else acc_w
          else acc_w
        ) acc_h
      ) (Tensor.scalar 0)

    let mean := divSpec channel_sum (Tensor.scalar spatial_size)

    let variance_sum :=
      (List.finRange height).foldl (fun acc_h i =>
        (List.finRange width).foldl (fun acc_w j =>
          if h_i : i < height then
            if h_j : j < width then
              let val := getAtSpec (getAtSpec channel_data ⟨i, h_i⟩) ⟨j, h_j⟩
              let diff := subSpec val mean
              addSpec acc_w (mulSpec diff diff)
            else acc_w
          else acc_w
        ) acc_h
      ) (Tensor.scalar 0)

    let varianceRaw := divSpec variance_sum (Tensor.scalar spatial_size)
    let variance := maxSpec varianceRaw (Tensor.scalar 0)
    let std := sqrtSpec (addSpec variance (Tensor.scalar epsilon))
    let inv_std := divSpec (Tensor.scalar 1) std

    -- Standard normalization gradient (equivalent to layer-norm over the spatial grid).
    let dXhat_sum :=
      (List.finRange height).foldl (fun acc_h i =>
        (List.finRange width).foldl (fun acc_w j =>
          if h_i : i < height then
            if h_j : j < width then
              let grad_val := getAtSpec (getAtSpec channel_grad ⟨i, h_i⟩) ⟨j, h_j⟩
              let dXhat := mulSpec grad_val channel_gamma
              addSpec acc_w dXhat
            else acc_w
          else acc_w
        ) acc_h
      ) (Tensor.scalar 0)

    let dXhatXhat_sum :=
      (List.finRange height).foldl (fun acc_h i =>
        (List.finRange width).foldl (fun acc_w j =>
          if h_i : i < height then
            if h_j : j < width then
              let val := getAtSpec (getAtSpec channel_data ⟨i, h_i⟩) ⟨j, h_j⟩
              let grad_val := getAtSpec (getAtSpec channel_grad ⟨i, h_i⟩) ⟨j, h_j⟩
              let centered := subSpec val mean
              let xHat := mulSpec centered inv_std
              let dXhat := mulSpec grad_val channel_gamma
              addSpec acc_w (mulSpec dXhat xHat)
            else acc_w
          else acc_w
        ) acc_h
      ) (Tensor.scalar 0)

    let nScalar : Tensor α Shape.scalar := Tensor.scalar (spatial_size : α)
    let inv_n : Tensor α Shape.scalar := divSpec (Tensor.scalar (1 : α)) nScalar

    Tensor.dim (fun i =>
      Tensor.dim (fun j =>
        let val := getAtSpec (getAtSpec channel_data i) j
        let grad_val := getAtSpec (getAtSpec channel_grad i) j
        let centered := subSpec val mean
        let xHat := mulSpec centered inv_std
        let dXhat := mulSpec grad_val channel_gamma
        let term :=
          subSpec (subSpec (mulSpec nScalar dXhat) dXhat_sum)
            (mulSpec xHat dXhatXhat_sum)
        mulSpec (mulSpec term inv_std) inv_n
      )
    )
  )

  -- Compute gamma gradients (sum over spatial dimensions for each channel)
  let grad_gamma := Tensor.dim (fun c =>
    let channel_data := getAtSpec x c
    let channel_grad := getAtSpec grad_output c

    -- Recompute mean and std for this channel
    let channel_sum :=
      (List.finRange height).foldl (fun acc_h i =>
        (List.finRange width).foldl (fun acc_w j =>
          if h_i : i < height then
            if h_j : j < width then
              addSpec acc_w (getAtSpec (getAtSpec channel_data ⟨i, h_i⟩) ⟨j, h_j⟩)
            else acc_w
          else acc_w
        ) acc_h
      ) (Tensor.scalar 0)

    let mean := divSpec channel_sum (Tensor.scalar spatial_size)

    let variance_sum :=
      (List.finRange height).foldl (fun acc_h i =>
        (List.finRange width).foldl (fun acc_w j =>
          if h_i : i < height then
            if h_j : j < width then
              let val := getAtSpec (getAtSpec channel_data ⟨i, h_i⟩) ⟨j, h_j⟩
              let diff := subSpec val mean
              addSpec acc_w (mulSpec diff diff)
            else acc_w
          else acc_w
        ) acc_h
      ) (Tensor.scalar 0)

    let varianceRaw := divSpec variance_sum (Tensor.scalar spatial_size)
    let variance := maxSpec varianceRaw (Tensor.scalar 0)
    let std := sqrtSpec (addSpec variance (Tensor.scalar epsilon))
    let inv_std := divSpec (Tensor.scalar 1) std

    -- Sum grad_output * normalized_input for this channel
    (List.finRange height).foldl (fun acc_h i =>
      (List.finRange width).foldl (fun acc_w j =>
        if h_i : i < height then
          if h_j : j < width then
            let val := getAtSpec (getAtSpec channel_data ⟨i, h_i⟩) ⟨j, h_j⟩
            let grad_val := getAtSpec (getAtSpec channel_grad ⟨i, h_i⟩) ⟨j, h_j⟩
            let normalized := mulSpec (subSpec val mean) inv_std
            addSpec acc_w (mulSpec grad_val normalized)
          else acc_w
        else acc_w
      ) acc_h
    ) (Tensor.scalar 0)
  )

  -- Compute beta gradients (sum of grad_output for each channel)
  let grad_beta := Tensor.dim (fun c =>
    let channel_grad := getAtSpec grad_output c

    (List.finRange height).foldl (fun acc_h i =>
      (List.finRange width).foldl (fun acc_w j =>
        if h_i : i < height then
          if h_j : j < width then
            addSpec acc_w (getAtSpec (getAtSpec channel_grad ⟨i, h_i⟩) ⟨j, h_j⟩)
          else acc_w
        else acc_w
      ) acc_h
    ) (Tensor.scalar 0)
  )

  (grad_x, grad_gamma, grad_beta)

/-!
## BatchNorm (inference-time, running statistics)

PyTorch distinction:

- *training*: normalize using batch statistics (and update running mean/variance);
- *inference*: normalize using the stored running mean/variance.

TorchLean keeps things pure and explicit: inference-time BatchNorm takes the running statistics
as arguments.
-/

/--
Inference-time BatchNorm for channel-first tensors of shape `.dim channels sSpatial`, using fixed
running statistics.

Formula (per channel `c`):

`y = ((x - μ) / sqrt(σ² + eps)) * γ + β`

This matches the standard evaluation-time behavior of `torch.nn.BatchNorm{1,2,3}d` (no
batch-statistics computation, no running-statistics update).

At inference time, `(μ, σ², γ, β)` are constants, so this is an **affine** map in `x`. See
`NN.Proofs.Analysis.Normalization.batchNorm_inference_eq_mul_add`.
-/
def batchNormInference
  {channels : Nat} {sSpatial : Shape}
  (x : Tensor α (.dim channels sSpatial))
  (runningMean : Tensor α (.dim channels .scalar))
  (runningVar : Tensor α (.dim channels .scalar))
  (gamma : Tensor α (.dim channels .scalar))
  (beta : Tensor α (.dim channels .scalar))
  (epsilon : α := Numbers.epsilon) :
  Tensor α (.dim channels sSpatial) :=
  let s : Shape := .dim channels sSpatial
  have cb : Shape.CanBroadcastTo (.dim channels .scalar) s := by
    apply Shape.CanBroadcastTo.dim_eq
    exact Shape.CanBroadcastTo.scalar_to_any sSpatial
  -- Clamp the variance to stay nonnegative in approximate numeric backends.
  let runningVar := maxSpec runningVar (fill 0 (.dim channels .scalar))
  normalizeCore
    (s := s)
    (s_mean := .dim channels .scalar)
    (s_var := .dim channels .scalar)
    (s_gamma := .dim channels .scalar)
    (s_beta := .dim channels .scalar)
    (epsilon := epsilon)
    (x := x)
    (mean := runningMean)
    (variance := runningVar)
    (gamma := gamma)
    (beta := beta)
    (cb_mean := cb)
    (cb_var := cb)
    (cb_gamma := cb)
    (cb_beta := cb)

/-! Convenience wrapper: `batchNorm_inference` specialized to a single channel-first image `(C,H,W)`. -/
def batchNorm2dInference
  {channels height width : Nat}
  (x : MultiChannelImage channels height width α)
  (runningMean : Tensor α (.dim channels .scalar))
  (runningVar : Tensor α (.dim channels .scalar))
  (gamma : Tensor α (.dim channels .scalar))
  (beta : Tensor α (.dim channels .scalar))
  (epsilon : α := Numbers.epsilon) :
  MultiChannelImage channels height width α :=
  batchNormInference
    (x := x) (runningMean := runningMean) (runningVar := runningVar) (gamma := gamma) (beta := beta)
    (epsilon := epsilon)

/-- Convenience wrapper: `batchNorm_inference` specialized to a `(C, L)` tensor (BatchNorm1d-style). -/
def batchNorm1dInference
  {channels length : Nat}
  (x : Tensor α (.dim channels (.dim length .scalar)))
  (runningMean : Tensor α (.dim channels .scalar))
  (runningVar : Tensor α (.dim channels .scalar))
  (gamma : Tensor α (.dim channels .scalar))
  (beta : Tensor α (.dim channels .scalar))
  (epsilon : α := Numbers.epsilon) :
  Tensor α (.dim channels (.dim length .scalar)) :=
  batchNormInference
    (x := x) (runningMean := runningMean) (runningVar := runningVar) (gamma := gamma) (beta := beta)
    (epsilon := epsilon)

/-- Convenience wrapper: `batchNorm_inference` specialized to a `(C, D, H, W)` tensor (BatchNorm3d-style). -/
def batchNorm3dInference
  {channels depth height width : Nat}
  (x : Tensor α (.dim channels (.dim depth (.dim height (.dim width .scalar)))))
  (runningMean : Tensor α (.dim channels .scalar))
  (runningVar : Tensor α (.dim channels .scalar))
  (gamma : Tensor α (.dim channels .scalar))
  (beta : Tensor α (.dim channels .scalar))
  (epsilon : α := Numbers.epsilon) :
  Tensor α (.dim channels (.dim depth (.dim height (.dim width .scalar)))) :=
  batchNormInference
    (x := x) (runningMean := runningMean) (runningVar := runningVar) (gamma := gamma) (beta := beta)
    (epsilon := epsilon)

end Spec

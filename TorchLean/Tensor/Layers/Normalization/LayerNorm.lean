/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Movement
public import TorchLean.Tensor.Reduce
public import TorchLean.Tensor.Elementwise

/-!
# Normalization layers (spec layer)

Shared pieces of TorchLean's normalization layers, and the JVP and VJP of LayerNorm over the last
axis of a matrix. The forward operators over any axes (`Tensor.layerNorm`, `Tensor.rmsNorm`,
`Tensor.batchNorm`, `Tensor.groupNorm`) are in `TorchLean.Tensor.Layers.Normalization`,
and most of them follow one pattern:

- compute per-axis statistics (mean and variance),
- normalize with an `epsilon` for numerical stability,
- apply an affine transform (`gamma`, `beta`) as PyTorch does.

LayerNorm and BatchNorm with batch statistics divide the centered data by
`Tensor.normalizationScale` before squaring it, so no unscaled square is summed. LayerNorm then
takes the variance of the scaled data, which centers it a second time, and simplifying that second
centering changes floating-point execution.

Reading the definitions as normalization, and their JVPs and VJPs as derivatives, needs a positive
`epsilon` and the laws of real arithmetic, and the scalar-polymorphic definitions do not check
`epsilon`. Over floats the forward, JVP and VJP are separately rounded programs. Their closed forms
are derivatives of the real map, and they need not agree bit for bit with a native backend. We
worked both closed forms out by hand with the chain rule, and `TorchLean.LayerNorm.fderiv_eq_jvp`
and `TorchLean.LayerNorm.adjoint_fderiv_eq_backward` (`TorchLean/NN/Norm/LayerNorm/FDeriv.lean`)
prove them over `ℝ` for positive `epsilon`, `seqLen` and `embedDim`.

## References (papers + PyTorch behavior)

- LayerNorm: Jimmy Lei Ba, Jamie Ryan Kiros and Geoffrey E. Hinton, "Layer Normalization" (2016),
  eqs. (3) and (4) of Sec. 3: https://arxiv.org/abs/1607.06450
- BatchNorm: Sergey Ioffe and Christian Szegedy, "Batch Normalization: Accelerating Deep Network
  Training by Reducing Internal Covariate Shift" (2015), Algorithms 1 and 2:
  https://arxiv.org/abs/1502.03167
- GroupNorm: Yuxin Wu and Kaiming He, "Group Normalization" (2018), eqs. (1), (2), (6) and (7) of
  Sec. 3.1: https://arxiv.org/abs/1803.08494
- RMSNorm: Biao Zhang and Rico Sennrich, "Root Mean Square Layer Normalization", NeurIPS 2019,
  eq. (4) of Sec. 4: https://arxiv.org/abs/1910.07467
- WeightNorm: Tim Salimans and Diederik P. Kingma, "Weight Normalization: A Simple
  Reparameterization to Accelerate Training of Deep Neural Networks" (2016):
  https://arxiv.org/abs/1602.07868

- PyTorch 2.8 LayerNorm: https://docs.pytorch.org/docs/2.8/generated/torch.nn.LayerNorm.html
- PyTorch 2.8 RMSNorm: https://docs.pytorch.org/docs/2.8/generated/torch.nn.RMSNorm.html
- PyTorch 2.8 GroupNorm: https://docs.pytorch.org/docs/2.8/generated/torch.nn.GroupNorm.html
- PyTorch 2.8 BatchNorm modules: https://docs.pytorch.org/docs/2.8/nn.html#normalization-layers
-/

@[expose] public section


namespace TorchLean.Tensor

variable {α : Type} [Storage α] [Context α] {s : Shape}

/-- The scale by which a normalization over `dims` divides the centered fiber `centered` before
squaring it: the IEEE maximum of the fiber's largest magnitude (zero for an empty fiber) and
`sqrt |regularizer|`, with the gradient stopped. -/
def normalizationScale (centered : Tensor α s) (dims : List (Fin s.rank))
    (regularizer : Tensor α (s.reduce dims false)) : Tensor α (s.reduce dims false) :=
  let largest : Tensor α (s.reduce dims false) :=
    if h : ∀ d ∈ dims, 0 < s.extent d then reduce .max (abs centered) dims false fun _ => h
    else full _ 0
  detach (Tensor.maximum largest (Tensor.sqrt (abs regularizer) .clampNegative))

end TorchLean.Tensor

namespace TorchLean
open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]

/-- The gradients of an affine normalization with respect to its input, scale and bias. -/
structure Normalization.Gradients (α : Type) [Storage α]
    (inputShape parameterShape : Shape) where
  /-- Gradient with respect to the input being normalized. -/
  inputGradient : Tensor α inputShape
  /-- Gradient with respect to the learned multiplicative scale. -/
  scaleGradient : Tensor α parameterShape
  /-- Gradient with respect to the learned additive bias. -/
  biasGradient : Tensor α parameterShape
deriving Repr

/-- The normalization `y / sqrt (max (variance + stabilizer) 0) * gamma + beta`, with the root
taken at the statistics' shape and broadcast to `s` along `statsDims`, and `gamma` and `beta`
broadcast to `s` along `gammaDims` and `betaDims`. -/
def Normalization.normalize {s sStats sGamma sBeta : Shape} (y : Tensor α s)
    (variance stabilizer : Tensor α sStats) (gamma : Tensor α sGamma) (beta : Tensor α sBeta)
    {statsDims : Fin sStats.rank → Fin s.rank} (hStats : sStats.BroadcastDims s statsDims)
    {gammaDims : Fin sGamma.rank → Fin s.rank} (hGamma : sGamma.BroadcastDims s gammaDims)
    {betaDims : Fin sBeta.rank → Fin s.rank} (hBeta : sBeta.BroadcastDims s betaDims) :
    Tensor α s :=
  let std := Tensor.sqrt (variance + stabilizer) .clampNegative
  ((y / Tensor.broadcast std s statsDims hStats) * Tensor.broadcast gamma s gammaDims hGamma) +
    Tensor.broadcast beta s betaDims hBeta

omit [DecidableRel ((· > ·) : α → α → Prop)] in
/-- `normalize` with the constant stabilizer `epsilon` is the normalization with the variance
broadcast to `s` before `epsilon` is added and the root is taken. -/
theorem Normalization.normalize_full {s sStats sGamma sBeta : Shape} (y : Tensor α s)
    (variance : Tensor α sStats) (epsilon : α) (gamma : Tensor α sGamma) (beta : Tensor α sBeta)
    {statsDims : Fin sStats.rank → Fin s.rank} (hStats : sStats.BroadcastDims s statsDims)
    {gammaDims : Fin sGamma.rank → Fin s.rank} (hGamma : sGamma.BroadcastDims s gammaDims)
    {betaDims : Fin sBeta.rank → Fin s.rank} (hBeta : sBeta.BroadcastDims s betaDims) :
    normalize y variance (Tensor.full sStats epsilon) gamma beta hStats hGamma hBeta =
      ((y / Tensor.sqrt (Tensor.broadcast variance s statsDims hStats + Tensor.full s epsilon)
          .clampNegative) * Tensor.broadcast gamma s gammaDims hGamma) +
        Tensor.broadcast beta s betaDims hBeta := by
  apply Rep.ext
  intro c
  simp only [normalize, add_def, mul_def, div_def, zipWith_apply, broadcast_apply,
    sqrt_clampNegative_apply, full_apply]

/-- The rows of `x` normalized as `Tensor.layerNorm` and `Tensor.batchNorm` normalize a fiber, with
the inverse standard deviation `1 / std / scale` of each row: the row less its mean, divided by
`scale = Tensor.normalizationScale` of it and by `std = sqrt (max (m + epsilon / scale / scale) 0)`
for the mean `m` of the squared scaled row, recentered first when `recenter`. The row mean sums
the raw row, so a finite row gives NaN when that sum or an entry less the mean overflows. -/
def Normalization.normalizedRows {rows cols : Nat} (x : Tensor α [rows, cols]) (epsilon : α)
    (recenter : Bool) :
    Tensor α [rows, cols] × Tensor α [rows] :=
  let s : Shape := [rows, cols]
  let rowMean := fun t : Tensor α s => Tensor.map (Context.divNat · cols)
    (castShape (reduce .sum t [⟨1, Nat.one_lt_two⟩] false) (Shape.reduce_singleton _ _))
  let spread := fun t : Tensor α [rows] =>
    Tensor.broadcast (s := s.eraseIdx 1) t s _ (s.eraseIdx_broadcastDims 1)
  let centered : Tensor α s := x - spread (rowMean x)
  let regularizer := Tensor.full [rows] epsilon
  let scale : Tensor α [rows] := castShape (Tensor.normalizationScale centered
      [⟨1, Nat.one_lt_two⟩]
      (castShape regularizer (Shape.reduce_singleton s ⟨1, Nat.one_lt_two⟩).symm))
    (Shape.reduce_singleton s ⟨1, Nat.one_lt_two⟩)
  let scaled := centered / spread scale
  let deviation := if recenter then scaled - spread (rowMean scaled) else scaled
  let meanSquare : Tensor α [rows] := rowMean (deviation * deviation)
  let std := Tensor.sqrt (meanSquare + ((regularizer / scale) / scale)) .clampNegative
  (scaled / spread std, (Tensor.full _ 1 / std) / scale)


/-- Backward/VJP of `Tensor.layerNorm` over the last axis of a `(seqLen, embedDim)` matrix, with
named input, scale, and bias gradients, from the forward's normalized rows (`normalizedRows`). -/
def LayerNorm.backward
  {seqLen embedDim : Nat}
  (sequenceLengthPositive : seqLen > 0)
  (embeddingWidthPositive : embedDim > 0)
  (input : Tensor α [seqLen, embedDim])
  (scale : Tensor α [embedDim])
  (outputGradient : Tensor α [seqLen, embedDim])
  (epsilon : α := normalizationEpsilon) :
  Normalization.Gradients α [seqLen, embedDim] [embedDim] :=

  -- No forward activations are passed in, so the normalized rows are recomputed from `input`.
  let _ : Shape.WellFormed (.dim seqLen (.dim embedDim .scalar)) :=
  ⟨⟨sequenceLengthPositive, ⟨embeddingWidthPositive, trivial⟩⟩⟩

  let s : Shape := [seqLen, embedDim]

  let normalized := Normalization.normalizedRows input epsilon true
  let norm := normalized.1
  let invStd := normalized.2

  have h5 : Shape.CanBroadcastTo (.dim embedDim .scalar) (.dim seqLen (.dim embedDim .scalar)) := by
    apply Shape.CanBroadcastTo.expand_dims
    apply Shape.CanBroadcastTo.dim_eq
    exact Shape.CanBroadcastTo.scalar

  -- Affine parameters are shared across positions, so their gradients sum over axis 0.
  let biasGradient := castShape (reduce .sum outputGradient [⟨0, Nat.two_pos⟩] false)
      (Shape.reduce_singleton _ _)
  let scaleGradient := castShape
      (reduce .sum (outputGradient * norm) [⟨0, Nat.two_pos⟩] false)
      (Shape.reduce_singleton _ _)

  -- ∂L/∂x: the standard LayerNorm VJP, with per-position statistics over the feature axis.
  --
  -- Let `N = embedDim`, `xhat = norm` and `dy = outputGradient`.
  -- With `dyGamma = dy ⊙ scale`, the closed form is:
  --
  --   dx = invStd ⊙ ( dyGamma
  --                    - mean(dyGamma)
  --                    - xhat ⊙ mean(dyGamma ⊙ xhat) )
  --
  -- where each `mean` is over the last axis (the features) of one sequence position.
  let scaleBroadcast := Tensor.broadcast scale _ h5.dims h5.broadcastDims
  let invStdBroadcast := Tensor.broadcast invStd s _ (s.eraseIdx_broadcastDims 1)
  let dyGamma := outputGradient * scaleBroadcast

  let sumDyGamma := castShape (reduce .sum dyGamma [⟨1, Nat.one_lt_two⟩] false)
      (Shape.reduce_singleton _ _)
  -- Each mean divides its sum by the feature count `N = embedDim` in a single `Context.divNat`.
  let meanDyGamma := Tensor.map (Context.divNat · embedDim) sumDyGamma

  let sumDyGammaXhat :=
    castShape (reduce .sum (dyGamma * norm) [⟨1, Nat.one_lt_two⟩] false)
        (Shape.reduce_singleton _ _)
  let meanDyGammaXhat := Tensor.map (Context.divNat · embedDim) sumDyGammaXhat

  let meanDyGammaBroadcast :=
    Tensor.broadcast meanDyGamma s _ (s.eraseIdx_broadcastDims 1)
  let meanDyGammaXhatBroadcast :=
    Tensor.broadcast meanDyGammaXhat s _ (s.eraseIdx_broadcastDims 1)

  let inputGradient :=
    invStdBroadcast *
      ((dyGamma - meanDyGammaBroadcast) - (norm *
        meanDyGammaXhatBroadcast))

  { inputGradient, scaleGradient, biasGradient }

/--
Forward-mode JVP of `Tensor.layerNorm` over the last axis of a `(seqLen, embedDim)` matrix.

At each sequence position LayerNorm is the map `y = gamma ⊙ xhat + beta` with
`xhat = (x - mean(x)) / sqrt(var(x)+eps)`. Input tangents go through the standard closed form

`dxhat = invStd ⊙ (dx - mean(dx) - xhat ⊙ mean(dx ⊙ xhat))`,

and affine tangents add `xhat ⊙ dgamma + dbeta`. Its reverse-mode counterpart is
`LayerNorm.backward`, and both take the normalized rows and inverse standard deviations of the
forward from `normalizedRows`.
-/
def LayerNorm.jvp
  {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
  (x tangent : Tensor α [seqLen, embedDim])
  (gamma dgamma _beta dbeta : Tensor α [embedDim])
  (epsilon : α := normalizationEpsilon) :
  Tensor α [seqLen, embedDim] :=

  let _ : Shape.WellFormed (.dim seqLen (.dim embedDim .scalar)) :=
  ⟨⟨h_seq_pos, ⟨h_embed_pos, trivial⟩⟩⟩

  let s : Shape := [seqLen, embedDim]

  let normalized := Normalization.normalizedRows x epsilon true
  let invStd := normalized.2
  let invStdBroadcast := Tensor.broadcast invStd s _ (s.eraseIdx_broadcastDims 1)
  let norm := normalized.1

  let sumTangent := castShape (reduce .sum tangent [⟨1, Nat.one_lt_two⟩] false)
      (Shape.reduce_singleton _ _)
  let meanTangent := Tensor.map (Context.divNat · embedDim) sumTangent

  let sumTangentNorm :=
    castShape (reduce .sum (tangent * norm : Tensor α s) [⟨1, Nat.one_lt_two⟩] false)
        (Shape.reduce_singleton _ _)
  let meanTangentNorm := Tensor.map (Context.divNat · embedDim) sumTangentNorm

  let meanTangentBroadcast := Tensor.broadcast meanTangent s _ (s.eraseIdx_broadcastDims 1)
  let meanTangentNormBroadcast :=
    Tensor.broadcast meanTangentNorm s _ (s.eraseIdx_broadcastDims 1)

  let dnorm :=
    invStdBroadcast *
      ((tangent - meanTangentBroadcast) -
        (norm * meanTangentNormBroadcast))

  have h5 : Shape.CanBroadcastTo (.dim embedDim .scalar) (.dim seqLen (.dim embedDim .scalar)) := by
    apply Shape.CanBroadcastTo.expand_dims
    apply Shape.CanBroadcastTo.dim_eq
    exact Shape.CanBroadcastTo.scalar

  let gammaBroadcast := Tensor.broadcast gamma _ h5.dims h5.broadcastDims
  let dgammaBroadcast := Tensor.broadcast dgamma _ h5.dims h5.broadcastDims
  let dbetaBroadcast := Tensor.broadcast dbeta _ h5.dims h5.broadcastDims
  ((dnorm * gammaBroadcast) + (norm * dgammaBroadcast)) +
    dbetaBroadcast

end TorchLean

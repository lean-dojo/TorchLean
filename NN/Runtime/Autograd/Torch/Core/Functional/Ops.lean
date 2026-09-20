/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape.Reductions
public import NN.Spec.Layers.Conv
public import NN.Spec.Layers.Pooling.Spatial
public import NN.Tensor.Conversion
public import NN.Runtime.Autograd.Torch.Core.Functional.Curried

/-!
# Backend Operation Interface

`Ops` supplies the tensor primitives for models shared by eager and typed graph execution.
`Ref` and `DataRef` select the reference types of the current backend.
-/

@[expose] public section

namespace Runtime.Autograd.Torch

open Spec TorchLean TorchLean.Tensor

/--
Tensor operations shared by eager and typed graph execution.

A model is polymorphic over this class. Eager instances execute and record a tape; typed graph
instances build reusable graph data. Each instance must preserve the corresponding operator's
forward and VJP semantics.
-/
class Ops (m : Type → Type) (α : Type) [Storage α] [Context α] where
  /-- A differentiable tensor handle, indexed by its shape. -/
  Ref : Shape → Type
  /-- Backend representation of non-differentiable tensor data. -/
  DataRef : (β : Type) → [Storage β] → Shape → Type
  /-- Lift fixed non-differentiable data into the backend representation. -/
  dataConst : {β : Type} → [Storage β] → {s : Shape} → Tensor β s → DataRef β s
  /-- Apply a pure transformation to non-differentiable data. -/
  mapData : {β γ : Type} → [Storage β] → [Storage γ] → {s₁ s₂ : Shape} →
      (Tensor β s₁ → Tensor γ s₂) → DataRef β s₁ → DataRef γ s₂
  /--
  Observe a stateful layer's actual forward input and update its persistent buffers.

  Pure interpreters may omit this hook. Runtime interpreters retain the recorded state values for
  differentiation and write updated values only to their persistent, non-trainable storage.
  -/
  updateBuffers? : Option ({ss : List Shape} → {s : Shape} → RefList Ref ss → Ref s →
    (TensorPack α ss → Tensor α s → IO (TensorPack α ss)) → m Unit) := none
  /-- Record a fixed tensor value. -/
  const : {s : Shape} → Tensor α s → m (Ref s)
  /-- Add tensors elementwise. -/
  add : {s : Shape} → Ref s → Ref s → m (Ref s)
  /-- Subtract tensors elementwise. -/
  sub : {s : Shape} → Ref s → Ref s → m (Ref s)
  /-- Multiply tensors elementwise. -/
  mul : {s : Shape} → Ref s → Ref s → m (Ref s)
  /-- Multiply every element by a scalar. -/
  scale : {s : Shape} → Ref s → α → m (Ref s)
  /-- Take the elementwise absolute value. -/
  abs : {s : Shape} → Ref s → m (Ref s)
  /--
  Compute `sqrt(max(x, 0))` elementwise. The selected JVP and VJP are zero for `x ≤ 0`;
  at positive inputs they use `1 / (2 * sqrt(x))`.
  -/
  sqrt : {s : Shape} → Ref s → m (Ref s)
  /--
  Clamp every element between the lower and upper bounds. Input gradients pass through the open
  interval and are zero at both endpoints and outside it. The bounds are fixed scalar settings.
  -/
  clamp : {s : Shape} → Ref s → α → α → m (Ref s)
  /-- Take the elementwise maximum. -/
  max : {s : Shape} → Ref s → Ref s → m (Ref s)
  /-- Take the elementwise minimum. -/
  min : {s : Shape} → Ref s → Ref s → m (Ref s)
  /-- Repeat values along axes allowed by the broadcasting witness. -/
  broadcastTo : {s₁ s₂ : Shape} → Shape.CanBroadcastTo s₁ s₂ → Ref s₁ → m (Ref s₂)
  /-- Change the shape while preserving scalar count and row-major order. -/
  reshape : {s₁ s₂ : Shape} → Ref s₁ → (h : Spec.Shape.size s₁ = Spec.Shape.size s₂) → m (Ref s₂)
  /-- Swap the two adjacent axes starting at `depth`. -/
  swapAdjacentAtDepth {s : Shape} : (depth : Nat) → Ref s → m (Ref (s.swapAdjacentAtDepth depth))
  /-- Sum along `axis` and remove it from the result shape. -/
  reduceSum {s : Shape} (axis : Nat) [Shape.HasNonemptyAxis axis s] [Shape.WellFormed s] :
      Ref s → m (Ref (shapeAfterSum s axis))
  /-- Average along `axis` and remove it from the result shape. -/
  reduceMean {s : Shape} (axis : Nat) [Shape.HasNonemptyAxis axis s] [Shape.WellFormed s] :
      Ref s → m (Ref (shapeAfterSum s axis))
  /-- Select one position along `axis`, removing that axis from the result. -/
  select {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s] :
      Ref s → Fin (Shape.axisSize s axis) → m (Ref (s.eraseAxis axis))
  /-- Gather positions along `axis` in the order given by the indices. -/
  indexSelect {s : Shape} (axis count : Nat) [Shape.AxisInBounds axis s] :
      Ref s → DataRef (Fin (Shape.axisSize s axis)) [count] →
        m (Ref (s.replaceAxis axis count))
  /-- Add source slices into the base at the selected positions along `axis`. -/
  scatterAdd {s : Shape} (axis count : Nat) [Shape.AxisInBounds axis s] :
      Ref s → Ref (s.replaceAxis axis count) →
        DataRef (Fin (Shape.axisSize s axis)) [count] → m (Ref s)
  /-- Multiply matrices after broadcasting their batch prefixes to a common shape. -/
  matmul {batchA batchB batch : Shape} {mDim nDim pDim : Nat}
      [Shape.BroadcastTo batchA batch] [Shape.BroadcastTo batchB batch] :
      Ref (batchA.concat [mDim, nDim]) →
      Ref (batchB.concat [nDim, pDim]) →
      m (Ref (batch.concat [mDim, pDim]))
  /-- Concatenate tensors along their leading axis. -/
  concatLeadingAxis {nDim mDim : Nat} {s : Shape} :
      Ref (s.prependDim nDim) →
      Ref (s.prependDim mDim) →
      m (Ref (s.prependDim (nDim + mDim)))
  /-- Take `len` entries of the leading axis, starting at `start`. -/
  sliceLeadingAxisRange {nDim : Nat} {s : Shape} :
      (start len : Nat) → (h : start + len ≤ nDim) →
      Ref (s.prependDim nDim) → m (Ref (s.prependDim len))
  /-- Apply spatial max pooling to one channels-first sample. -/
  maxPool {d C : Nat}
      {inSpatial kernel stride padding : Tensor Nat [d]} :
      Ref (Shape.ofList (C :: Tensor.to inSpatial (List Nat))) →
      m (Ref (Shape.ofList
        (C :: Tensor.to (Spec.poolOutSpatialPad inSpatial kernel stride padding) (List Nat))))
  /-- Apply spatial average pooling to one channels-first sample, counting padding as zeros. -/
  avgPool {d C : Nat}
      {inSpatial kernel stride padding : Tensor Nat [d]} :
      Ref (Shape.ofList (C :: Tensor.to inSpatial (List Nat))) →
      m (Ref (Shape.ofList
        (C :: Tensor.to (Spec.poolOutSpatialPad inSpatial kernel stride padding) (List Nat))))
  /-- Apply a smooth approximation to spatial max pooling, with sharpness `beta`. -/
  smoothMaxPool {d C : Nat}
      {inSpatial kernel stride padding : Tensor Nat [d]} [DecidableEq α] :
      Ref (Shape.ofList (C :: Tensor.to inSpatial (List Nat))) →
      α →
      m (Ref (Shape.ofList
        (C :: Tensor.to (Spec.poolOutSpatialPad inSpatial kernel stride padding) (List Nat))))
  /-- Replace negative elements with zero. -/
  relu : {s : Shape} → Ref s → m (Ref s)
  /-- Apply the logistic sigmoid elementwise. -/
  sigmoid : {s : Shape} → Ref s → m (Ref s)
  /-- Apply the hyperbolic tangent elementwise. -/
  tanh : {s : Shape} → Ref s → m (Ref s)
  /--
  Apply tanh-approximate GELU as one backend primitive.

  The formula is `0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))`. Keeping it primitive
  avoids building temporary tensors for each term. Backends must match `Activation.geluSpec` and
  `Activation.geluDerivSpec`.
  -/
  gelu : {s : Shape} → Ref s → m (Ref s)
  /-- Apply softmax over the final tensor dimension. -/
  softmaxLast : {s : Shape} → Ref s → m (Ref s)
  /--
  Apply stable log-softmax over the final tensor dimension.

  Use `x - max(x) - log(sum(exp(x - max(x))))`. Arbitrary-axis operations move their selected
  axis to the final position before calling this primitive.
  -/
  logSoftmaxLast : {s : Shape} → Ref s → m (Ref s)
  /-- Apply `log(1 + exp(x))` elementwise. -/
  softplus : {s : Shape} → Ref s → m (Ref s)
  /-- Take the elementwise exponential. -/
  exp : {s : Shape} → Ref s → m (Ref s)
  /--
  Take the elementwise sine of angles in radians. JVPs and VJPs multiply by `cos(x)`, evaluated
  at the original input, so differentiation preserves the derivative's sign across periods.
  -/
  sin : {s : Shape} → Ref s → m (Ref s)
  /-- Take the elementwise cosine of angles in radians, with derivative `-sin(x)`. -/
  cos : {s : Shape} → Ref s → m (Ref s)
  /-- Take the elementwise logarithm. -/
  log : {s : Shape} → Ref s → m (Ref s)
  /-- Take the elementwise reciprocal. -/
  inv : {s : Shape} → Ref s → m (Ref s)
  /-- Keep the value and stop gradients through this reference. -/
  detach : {s : Shape} → Ref s → m (Ref s)
  /--
  Apply `log(softplus(x) + epsilon)` elementwise.

  Positive `epsilon` keeps the logarithm's argument positive, including when a floating-point
  softplus rounds to zero. The derivative is `sigmoid(x) / (softplus(x) + epsilon)`, with
  `epsilon` held fixed.
  -/
  safeLog : {s : Shape} → Ref s → α → m (Ref s)
  /-- Sum every element into a scalar. -/
  sum : {s : Shape} → Ref s → m (Ref Shape.scalar)
  /-- Flatten all axes into one, preserving row-major order. -/
  flatten : {s : Shape} → Ref s → m (Ref [Spec.Shape.size s])
  /-- Apply `weight * input + bias` to one input vector. -/
  linear {inDim outDim : Nat} :
      Ref [outDim, inDim] →
      Ref [outDim] →
      Ref [inDim] →
      m (Ref [outDim])
  /-- Average the squared elementwise difference between prediction and target. -/
  mseLoss : {s : Shape} → Ref s → Ref s → m (Ref Shape.scalar)
  /--
  Normalize each row over its embedding axis, then apply scale and bias.

  The denominator is `sqrt(variance + epsilon)`. Forward and backward use the same `epsilon`;
  gradients are taken with respect to the input, scale, and bias while `epsilon` stays fixed.
  -/
  layerNorm {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
      (x : Ref [seqLen, embedDim]) (gamma beta : Ref [embedDim])
      (epsilon : α := TorchLean.normalizationEpsilon) :
      m (Ref [seqLen, embedDim])
  /--
  Normalize each channel over its spatial axes, then apply scale and bias.

  The denominator is `sqrt(variance + epsilon)`. Forward and backward use the same `epsilon`;
  gradients are taken with respect to the input, scale, and bias while `epsilon` stays fixed.
  -/
  batchNorm {channels : Nat} {sSpatial : Shape}
      (hWellFormed : (sSpatial.prependDim channels).wellFormed)
      (x : Ref (sSpatial.prependDim channels)) (gamma beta : Ref [channels])
      (epsilon : α := TorchLean.normalizationEpsilon) :
      m (Ref (sSpatial.prependDim channels))
  /-- Apply multi-head self-attention to one sequence, using the optional mask. -/
  multiHeadAttention {n numHeads dModel headDim : Nat} (h1 : n ≠ 0) :
      Ref [dModel, numHeads * headDim] →
      Ref [dModel, numHeads * headDim] →
      Ref [dModel, numHeads * headDim] →
      Ref [numHeads * headDim, dModel] →
      Ref [n, dModel] →
      Option (Tensor Bool [n, n]) →
      m (Ref [n, dModel])
  /--
  Multi-head self-attention with an explicit leading batch axis.

  Its mathematical meaning is the leading-axis map of `multiHeadAttention`; implementations may
  execute the samples together, but may not change the mask convention or the per-sample
  forward/VJP semantics.
  -/
  batchedMultiHeadAttention {batch n numHeads dModel headDim : Nat}
      (hBatch : batch ≠ 0) (h1 : n ≠ 0) :
      Ref [dModel, numHeads * headDim] →
      Ref [dModel, numHeads * headDim] →
      Ref [dModel, numHeads * headDim] →
      Ref [numHeads * headDim, dModel] →
      Ref [batch, n, dModel] →
      Option (Tensor Bool [n, n]) →
      m (Ref [batch, n, dModel])
  /-- Apply spatial convolution to one channels-first sample. -/
  conv {d inC outC : Nat}
      {kernel stride padding : Tensor Nat [d]}
      {inSpatial : Tensor Nat [d]} :
      Ref (Shape.ofList (outC :: inC :: Tensor.to kernel (List Nat))) →
      Ref [outC] →
      Ref (Shape.ofList (inC :: Tensor.to inSpatial (List Nat))) →
      m (Ref (Shape.ofList
        (outC :: Tensor.to (Spec.convOutSpatial inSpatial kernel stride padding) (List Nat))))
  /-- Apply spatial transpose convolution to one channels-first sample. -/
  convTranspose {d inC outC : Nat}
      {kernel stride padding : Tensor Nat [d]}
      {inSpatial : Tensor Nat [d]} :
      Ref (Shape.ofList (inC :: outC :: Tensor.to kernel (List Nat))) →
      Ref [outC] →
      Ref (Shape.ofList (inC :: Tensor.to inSpatial (List Nat))) →
      m (Ref (Shape.ofList (outC ::
        Tensor.to (Spec.convTransposeOutSpatial inSpatial kernel stride padding) (List Nat))))
  /--
  Draw a uniform tensor from the seed and the backend's node or call index.

  No `IO` randomness is used, so a fixed graph can replay the draw.
  -/
  randUniform : {s : Shape} → (seed : Nat) → m (Ref s)
  /-- Draw a seeded mask with the scalar reference as its keep probability. -/
  bernoulliMask : {s : Shape} → Ref Shape.scalar → (seed : Nat) → m (Ref s)
  /--
  Optional native packed real transform. Returning `none` leaves the operation to the generic
  differentiable implementation; a backend must not record any nodes before declining the call.
  The output stores real and imaginary components in its final axis.
  -/
  rfft1dNative? : Option ({batch n : Nat} → Ref [batch, n] →
    m (Option (Ref [batch, n / 2 + 1, 2]))) := none
  /-- Optional normalized inverse of the packed real transform, with explicit output length. -/
  irfft1dNative? : Option ({batch n : Nat} → Ref [batch, n / 2 + 1, 2] →
    m (Option (Ref [batch, n]))) := none
  /-- Optional diagonal recurrence with coefficients shared across time and all four VJPs. -/
  selectiveScanDiagNative? : Option ({seqLen state : Nat} →
    Ref [state] → Ref [state] → Ref [seqLen, state] → Ref [state] →
    m (Option (Ref [seqLen, state]))) := none
  /-- Optional diagonal recurrence with token-dependent coefficients and all four VJPs. -/
  selectiveScanDiagVarNative? : Option ({seqLen state : Nat} →
    Ref [seqLen, state] → Ref [seqLen, state] → Ref [seqLen, state] → Ref [state] →
    m (Option (Ref [seqLen, state]))) := none
  /--
  Optional one-sided spectral convolution. Both weight tensors use `[modes, input, output]`;
  the transform is unnormalized and its real inverse divides by `grid`.
  -/
  spectralConv1dRfftNative? : Option ({grid width modes : Nat} →
    Ref [grid, width] → Ref [modes, width, width] → Ref [modes, width, width] →
    m (Option (Ref [grid, width]))) := none

variable {m : Type → Type} {α : Type} [Storage α] [Context α]
    [Ops (m := m) (α := α)]

/-- Differentiable reference type of the current backend. -/
abbrev Ref (s : Shape) : Type :=
  Ops.Ref (m := m) (α := α) s

/-- Backend representation of a non-differentiable tensor. -/
abbrev DataRef (β : Type) [Storage β] (s : Shape) : Type :=
  Ops.DataRef (m := m) (α := α) β s

end Runtime.Autograd.Torch

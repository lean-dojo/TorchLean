/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Program
public import NN.Spec.Layers.Attention
public import NN.Spec.Layers.Loss
public import NN.Spec.Layers.Normalization.BatchNorm
public import NN.Spec.Layers.Pooling

/-!
# SpecEval

Pure (non-`IO`) TorchLean execution for forward models.

This file gives the TorchLean `Program` interface a pure reference interpreter:

- `Ref s` is interpreted as an actual `Tensor α s`,
- operations use their tensor and layer specifications;
- the monad is `Except String`, with errors for invalid pooling parameters and unsupported random
  operations.

This interpreter is not the `Proved.ForwardProgram` evaluator covered by the lowering theorem.
It supports operations outside that proved fragment, including transposed convolution and smooth
max pooling. Raw logarithm uses the scalar specification without the IR evaluator's positivity
check, and log-softmax uses the stable specification rather than taking the logarithm of softmax.
Agreement with the broader `lowerForwardToIR` builder is therefore not established for all programs.
-/

@[expose] public section

namespace NN.Verification.Builtin

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor

/-- Error-reporting monad used by the pure TorchLean spec evaluator. -/
abbrev SpecM := Except String

instance {α : Type} [TorchLean.Storage α] [Context α] :
    Runtime.Autograd.Torch.Ops (m := SpecM) α where
  Ref := fun s => Tensor α s
  DataRef := fun β _ s => Tensor β s

  const := fun {_s} t => pure t
  observe := fun x => pure x.item
  dataConst := fun t => t
  mapData := fun f t => f t

  add := fun {_s} a b => pure (Tensor.addSpec (α := α) a b)
  sub := fun {_s} a b => pure (Tensor.subSpec (α := α) a b)
  mul := fun {_s} a b => pure (Tensor.mulSpec (α := α) a b)
  div := fun {_s} a b => pure (Tensor.divSpec (α := α) a b)
  neg := fun {_s} x => pure (Tensor.negSpec (α := α) x)
  scale := fun {_s} x c => pure (Tensor.scaleSpec (α := α) x c)
  abs := fun {_s} x => pure (Tensor.absSpec (α := α) x)
  sqrt := fun {_s} x => pure (Tensor.sqrtSpec (α := α) x)
  clamp := fun {_s} x lo hi => pure (Tensor.clampSpec (α := α) x lo hi)
  max := fun {_s} a b => pure (Tensor.maxSpec (α := α) a b)
  min := fun {_s} a b => pure (Tensor.minSpec (α := α) a b)

  broadcastTo := fun {s₁ s₂} cb x => pure (Tensor.broadcastTo (α := α) (s₁ := s₁) (s₂ := s₂) cb x)
  reshape := fun {s₁ s₂} x h =>
    pure (Tensor.reshapeSpec (α := α) (source := s₁) (target := s₂) x h)
  swapAdjacentAtDepth := fun {_s} depth x =>
    -- `swapAdjacentAtDepth` at depth 0 corresponds to swapping the first two axes; deeper swaps
    -- recurse through the outer dims.
    pure (Tensor.swapAdjacentAxes (tensor := x) depth)

  reduceSum := fun {s} axis _valid _wf x =>
    let hAxis : Shape.NonemptyAxis axis s := (inferInstance : Shape.HasNonemptyAxis axis s).proof
    pure (Tensor.reduceSum (α := α) (s := s) axis x hAxis)
  reduceMean := fun {s} axis _valid _wf x =>
    let hAxis : Shape.NonemptyAxis axis s := (inferInstance : Shape.HasNonemptyAxis axis s).proof
    pure (Tensor.reduceMean (α := α) (s := s) axis x hAxis)

  select := fun {_s} axis _axisInBounds x index =>
    pure (Tensor.selectSpec axis x index)
  indexSelect := fun {_s} axis _count _axisInBounds x indices =>
    pure (Tensor.indexSelectSpec axis x indices)
  scatterAdd := fun {_s} axis _count _axisInBounds base source indices =>
    pure (Tensor.scatterAddSpec axis base indices source)

  matmul := fun {_batchA _batchB _batch _mDim _nDim _pDim} broadcastA broadcastB a b =>
    pure (Tensor.matmulSpec broadcastA.proof broadcastB.proof a b)
  concat := fun {_nDim _mDim} {_s} a b =>
    pure (Tensor.concatAxisSpec (α := α) .scalar a b)

  slice := fun {_nDim} {_s} start len h x =>
    pure (Spec.sliceRangeSpec x start len h)

  maxPool := fun {d C} {inSpatial kernel stride padding} x =>
    if hKernel : (∀ i : Fin d, kernel.getScalar i ≠ 0) then
      if hStride : (∀ i : Fin d, stride.getScalar i ≠ 0) then
        let layer : Spec.MaxPoolSpec d kernel stride padding hKernel hStride := {}
        pure (Spec.maxPoolSpec (α := α) (d := d) (C := C)
          (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
          (layer := layer) (input := x))
      else
        throw "TorchLeanSpecEval: max_pool invalid stride (some axis has stride=0)"
    else
      throw "TorchLeanSpecEval: max_pool invalid kernel (some axis has size=0)"
  avgPool := fun {d C} {inSpatial kernel stride padding} x =>
    if hKernel : (∀ i : Fin d, kernel.getScalar i ≠ 0) then
      if hStride : (∀ i : Fin d, stride.getScalar i ≠ 0) then
        let layer : Spec.AvgPoolSpec d kernel stride padding hKernel hStride := {}
        pure (Spec.avgPoolSpec (α := α) (d := d) (C := C)
          (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
          (hKernel := hKernel) (layer := layer) (input := x))
      else
        throw "TorchLeanSpecEval: avg_pool invalid stride (some axis has stride=0)"
    else
      throw "TorchLeanSpecEval: avg_pool invalid kernel (some axis has size=0)"
  smoothMaxPool := fun {d C} {inSpatial kernel stride padding}
      [_decidableEq : DecidableEq α] x beta =>
    if hBeta : beta ≠ 0 then
      if hKernel : (∀ i : Fin d, kernel.getScalar i ≠ 0) then
        if hStride : (∀ i : Fin d, stride.getScalar i ≠ 0) then
          let layer : Spec.MaxPoolSpec d kernel stride padding hKernel hStride := {}
          pure (Spec.smoothMaxPoolSpec (α := α) (d := d) (C := C)
            (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
            (layer := layer) (beta := beta) (hBeta := hBeta) (input := x))
        else
          throw "TorchLeanSpecEval: smooth_max_pool invalid stride (some axis has stride=0)"
      else
        throw "TorchLeanSpecEval: smooth_max_pool invalid kernel (some axis has size=0)"
    else
      throw "TorchLeanSpecEval: smooth_max_pool requires nonzero beta"

  relu := fun {_s} x => pure (Activation.reluSpec (α := α) x)
  sigmoid := fun {_s} x => pure (Activation.sigmoidSpec (α := α) x)
  tanh := fun {_s} x => pure (Activation.tanhSpec (α := α) x)
  gelu := fun {_s} x => pure (Activation.geluSpec (α := α) x)
  softmaxLast := fun {s} x =>
    match s with
    | .scalar => pure (Tensor.scalar 1)
    | .dim n inner =>
        let axis := Shape.rank (.dim n inner) - 1
        letI : Shape.AxisInBounds axis (.dim n inner) := ⟨by
          simp [axis, Shape.rank]⟩
        pure (Activation.softmaxSpec (α := α) axis x)
  logSoftmaxLast := fun {s} x =>
    match s with
    | .scalar => pure (Tensor.scalar 0)
    | .dim n inner =>
        let axis := Shape.rank (.dim n inner) - 1
        letI : Shape.AxisInBounds axis (.dim n inner) := ⟨by
          simp [axis, Shape.rank]⟩
        pure (Activation.logSoftmaxSpec (α := α) axis x)
  softplus := fun {_s} x => pure (Activation.softplusSpec (α := α) x)
  exp := fun {_s} x => pure (Tensor.expSpec (α := α) x)
  sin := fun {_s} x => pure (Tensor.sinSpec (α := α) x)
  cos := fun {_s} x => pure (Tensor.cosSpec (α := α) x)
  log := fun {_s} x => pure (Tensor.logSpec (α := α) x)
  inv := fun {_s} x => pure (Tensor.invSpec (α := α) x)
  -- The spec interpreter also runs with dual scalars. Detach must clear their tangents here,
  -- just as IR evaluation does, so later operations cannot propagate the detached tangent.
  detach := fun {_s} x => pure (Tensor.detachSpec (α := α) x)
  safeLog := fun {_s} x ε => pure (Activation.safeLogSpec (α := α) x ε)
  sum := fun {_s} x => pure (Tensor.scalar (Tensor.sumSpec (α := α) x))
  flatten := fun {_s} x => pure (Tensor.flattenSpec (α := α) x)

  linear := fun {inDim outDim} w b x =>
    pure (Tensor.addSpec (α := α)
      (matVecMulSpec (α := α) (m := outDim) (n := inDim) w x) b)
  mseLoss := fun {s} yhat target =>
    pure (Tensor.scalar (Spec.mseSpec (α := α) (s := s) yhat target))

  layerNorm := fun {seqLen embedDim} hSeq hEmb x gamma beta epsilon =>
    pure (Spec.layerNorm (α := α) (seqLen := seqLen) (embedDim := embedDim)
      (x := x) (gamma := gamma) (beta := beta) (h_seq_pos := hSeq) (h_embed_pos := hEmb)
      (epsilon := epsilon))

  batchNorm := fun {channels sSpatial} hWellFormed x gamma beta epsilon =>
    let _ : Shape.WellFormed (.dim channels sSpatial) := ⟨hWellFormed⟩
    pure (Spec.batchNorm (α := α) (channels := channels) (sSpatial := sSpatial)
      (x := x) (gamma := gamma) (beta := beta) (epsilon := epsilon))

  attention := fun {n numHeads dModel headDim batch} _hBatch h1 wq wk wv wo x mask =>
    let layer : Spec.MultiHeadAttention α numHeads dModel headDim :=
      { queryWeight := wq, keyWeight := wk, valueWeight := wv, outputWeight := wo }
    let forward := fun sample => layer.forward n h1 sample mask
    match batch, x with
    | none, sample => pure (forward sample)
    | some _, samples => pure (Tensor.dim (fun i => forward (Tensor.unstack samples i)))

  conv := fun {d inC outC} {kernel stride padding} {inSpatial} w b x =>
    let layer : Spec.ConvSpec d inC outC kernel stride padding α :=
      { kernel := w, bias := b }
    pure (Spec.convSpec (α := α) (d := d) (inC := inC) (outC := outC)
      (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
      (layer := layer) (input := x))
  convTranspose := fun {d inC outC} {kernel stride padding} {inSpatial} w b x =>
    let layer : Spec.ConvTransposeSpec d inC outC kernel stride padding α :=
      { kernel := w, bias := b }
    pure (Spec.convTransposeSpec (α := α) (d := d) (inC := inC) (outC := outC)
      (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
      (layer := layer) (input := x))

  randUniform := fun {_s} _seed =>
    throw <|
      "TorchLeanSpecEval: rand_uniform is not supported in spec backend " ++
        "(needs a deterministic counter)"
  bernoulliMask := fun {_s} _keepProb _seed =>
    throw <|
      "TorchLeanSpecEval: bernoulli_mask is not supported in spec backend " ++
        "(needs a deterministic counter)"

end NN.Verification.Builtin

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.Backend
public import NN.Spec.Core.TensorReductionShape
public import NN.Spec.Layers.Activation
public import NN.Spec.Layers.Attention
public import NN.Spec.Layers.Conv
public import NN.Spec.Layers.Loss
public import NN.Spec.Layers.Normalization
public import NN.Spec.Layers.Pooling

/-!
# SpecEval

Pure (non-`IO`) TorchLean execution for forward models.

This file gives the TorchLean `Program` interface a pure *spec semantics* backend:

- `Ref s` is interpreted as an actual `Tensor α s`,
- each primitive op is interpreted via the corresponding `Spec.*_spec` definition,
- the monad is `Except String`, so unsupported verifier-fragment cases report explicit errors
  instead of silently choosing a meaningless semantics.

This backend is meant as the “reference” semantics when stating compiler-correctness theorems
for `NN.Verification.TorchLean.compileForward1`.
-/

@[expose] public section


namespace NN.Verification.TorchLean

open _root_.Spec
open _root_.Spec.Tensor

/-- Error-reporting monad used by the pure TorchLean spec evaluator. -/
abbrev SpecM := Except String

instance {α : Type} [Context α] [DecidableEq Shape] [Inhabited α] : Runtime.Autograd.Torch.Ops (m :=
  SpecM) α where
  Ref := fun s => Tensor α s

  const := fun {_s} t => pure t

  add := fun {_s} a b => pure (Tensor.addSpec (α := α) a b)
  sub := fun {_s} a b => pure (Tensor.subSpec (α := α) a b)
  mul := fun {_s} a b => pure (Tensor.mulSpec (α := α) a b)
  scale := fun {_s} x c => pure (Tensor.scaleSpec (α := α) x c)
  abs := fun {_s} x => pure (Tensor.absSpec (α := α) x)
  sqrt := fun {_s} x => pure (Tensor.sqrtSpec (α := α) x)
  clamp := fun {_s} x lo hi => pure (Tensor.clampSpec (α := α) x lo hi)
  max := fun {_s} a b => pure (Tensor.maxSpec (α := α) a b)
  min := fun {_s} a b => pure (Tensor.minSpec (α := α) a b)

  broadcastTo := fun {s₁ s₂} cb x => pure (Tensor.broadcastTo (α := α) (s₁ := s₁) (s₂ := s₂) cb x)
  reshape := fun {s₁ s₂} x h => pure (Tensor.reshapeSpec (α := α) (s₁ := s₁) (s₂ := s₂) x h)
  transpose2d := fun {_mDim _nDim} x => pure (Tensor.matrixTransposeSpec (α := α) x)
  transpose3dFirstToLast := fun {_a _b _c} x => pure (Tensor.transpose3DFirstToLastSpec (α
    := α) x)
  transpose3dLastToFirst := fun {_a _b _c} x => pure (Tensor.transpose3DLastToFirstSpec (α
    := α) x)
  transpose3dLastTwo := fun {_a _b _c} x => pure (Tensor.transpose3DLastTwoSpec (α := α) x)
  swapAdjacentAtDepth := fun {_s} depth x =>
    -- `swapAdjacentAtDepth` at depth 0 corresponds to swapping the first two axes; deeper swaps
    -- recurse through the outer dims.
    pure (Tensor.swapAtDepthHelper (tensor := x) depth)

  reduceSum := fun {s} axis _valid _wf x =>
    let hAxis : Shape.valid_axis axis s := (inferInstance : Shape.valid_axis_inst axis s).proof
    let hRed := Shape.proveReducibleAlong axis s hAxis
    pure (Tensor.reduceSum (α := α) (s := s) axis x hRed)
  reduceMean := fun {s} axis _valid _wf x =>
    let hAxis : Shape.valid_axis axis s := (inferInstance : Shape.valid_axis_inst axis s).proof
    let hRed := Shape.proveReducibleAlong axis s hAxis
    pure (Tensor.reduceMean (α := α) (s := s) axis x hRed)

  gatherScalar := fun {_n} x i =>
    pure (getAtSpec x i)
  gatherRow := fun {_rows _cols} x i =>
    pure (getAtSpec x i)

  gatherScalarNat := fun {_n} _x _i => throw
    "TorchLeanSpecEval: gather_scalar_nat not supported in spec backend"
  gatherVecNat := fun {_n _k} _x _idx => throw
    "TorchLeanSpecEval: gather_vec_nat not supported in spec backend"
  gatherRowsNat := fun {_rows _cols _k} _x _idx => throw
    "TorchLeanSpecEval: gather_rows_nat not supported in spec backend"
  scatterAddVec := fun {_n} _x _val _i => throw
    "TorchLeanSpecEval: scatter_add_vec not supported in spec backend"
  scatterAddRow := fun {_rows _cols} _x _row _i => throw
    "TorchLeanSpecEval: scatter_add_row not supported in spec backend"

  matmul := fun {_mDim _nDim _pDim} a b => pure (matMulSpec (α := α) a b)
  bmm := fun {_batch _mDim _nDim _pDim} a b => pure (Tensor.bmmSpec (α := α) a b)

  concatVectors := fun {_nDim _mDim} a b => pure (Tensor.concatVectorsSpec (α := α) a b)
  concatDim0 := fun {_nDim _mDim} {s} a b => pure (Tensor.concatDim0Spec (α := α) (s := s) a b)

  sliceRange0 := fun {_nDim} {_s} _start _len _h _x =>
    throw "TorchLeanSpecEval: slice_range0 not supported in spec backend"

  maxPool := fun {d C} {inSpatial kernel stride padding} {hKernel} x =>
    if hStride : (∀ i : Fin d, stride.get i ≠ 0) then
      let layer : Spec.MaxPoolSpec d kernel stride padding hKernel hStride := {}
      pure (Spec.maxPoolSpec (α := α) (d := d) (C := C)
        (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
        (layer := layer) (input := x))
    else
      throw "TorchLeanSpecEval: max_pool invalid stride (some axis has stride=0)"
  avgPool := fun {d C} {inSpatial kernel stride padding} hKernel x =>
    if hStride : (∀ i : Fin d, stride.get i ≠ 0) then
      let layer : Spec.AvgPoolSpec d kernel stride padding hKernel hStride := {}
      pure (Spec.avgPoolSpec (α := α) (d := d) (C := C)
        (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
        (hKernel := hKernel) (layer := layer) (input := x))
    else
      throw "TorchLeanSpecEval: avg_pool invalid stride (some axis has stride=0)"
  smoothMaxPool := fun {d C} {inSpatial kernel stride padding} {hKernel} x beta =>
    if hStride : (∀ i : Fin d, stride.get i ≠ 0) then
      let layer : Spec.MaxPoolSpec d kernel stride padding hKernel hStride := {}
      pure (Spec.smoothMaxPoolSpec (α := α) (d := d) (C := C)
        (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
        (layer := layer) (beta := beta) (input := x))
    else
      throw "TorchLeanSpecEval: smooth_max_pool invalid stride (some axis has stride=0)"

  maxPool2d := fun {kH kW inH inW inC stride} {h1} {h2} x =>
    if hStride : stride ≠ 0 then
      let layer : Spec.MaxPool2DSpec kH kW stride h1 h2 hStride := {}
      pure (Spec.maxPool2dMultiSpec (α := α) (inC := inC) (inH := inH) (inW := inW) (layer :=
        layer) x)
    else
      throw "TorchLeanSpecEval: max_pool2d invalid stride (stride=0)"
  maxPool2dPad := fun {kH kW inH inW inC stride padding} {h1} {h2} x =>
    if hStride : stride ≠ 0 then
      let layer : Spec.MaxPool2DSpec kH kW stride h1 h2 hStride := {}
      pure (Spec.maxPool2dMultiSpecPad (α := α) (inC := inC) (inH := inH) (inW := inW)
        (stride := stride) (padding := padding) (layer := layer) x)
    else
      throw "TorchLeanSpecEval: max_pool2d_pad invalid stride (stride=0)"
  smoothMaxPool2d := fun {kH kW inH inW inC stride} {h1} {h2} x beta =>
    if hStride : stride ≠ 0 then
      let layer : Spec.MaxPool2DSpec kH kW stride h1 h2 hStride := {}
      pure (Spec.smoothMaxPool2dMultiSpec (α := α) (inC := inC) (inH := inH) (inW := inW) (layer
        := layer) (beta := beta) x)
    else
      throw "TorchLeanSpecEval: smooth_max_pool2d invalid stride (stride=0)"
  avgPool2d := fun {kH kW inH inW inC stride} h1 h2 x =>
    if hStride : stride ≠ 0 then
      let layer : Spec.AvgPool2DSpec kH kW stride h1 h2 hStride := {}
      pure (Spec.avgPool2dMultiSpec (α := α) (inC := inC) (inH := inH) (inW := inW) (h1 := h1)
        (h2 := h2) (layer := layer) x)
    else
      throw "TorchLeanSpecEval: avg_pool2d invalid stride (stride=0)"
  avgPool2dPad := fun {kH kW inH inW inC stride padding} h1 h2 x =>
    if hStride : stride ≠ 0 then
      let layer : Spec.AvgPool2DSpec kH kW stride h1 h2 hStride := {}
      pure (Spec.avgPool2dMultiSpecPad (α := α) (inC := inC) (inH := inH) (inW := inW)
        (stride := stride) (padding := padding) (h1 := h1) (h2 := h2) (layer := layer) x)
    else
      throw "TorchLeanSpecEval: avg_pool2d_pad invalid stride (stride=0)"

  relu := fun {_s} x => pure (Activation.reluSpec (α := α) x)
  sigmoid := fun {_s} x => pure (Activation.sigmoidSpec (α := α) x)
  tanh := fun {_s} x => pure (Activation.tanhSpec (α := α) x)
  softmax := fun {_s} x => pure (Activation.softmaxSpec (α := α) x)
  logSoftmax := fun {_s} x => pure (Activation.logSoftmaxSpec (α := α) x)
  softplus := fun {_s} x => pure (Activation.softplusSpec (α := α) x)
  exp := fun {_s} x => pure (Tensor.expSpec (α := α) x)
  log := fun {_s} x => pure (Tensor.logSpec (α := α) x)
  inv := fun {_s} x => pure (Tensor.invSpec (α := α) x)
  detach := fun {_s} x => pure x
  safeLog := fun {_s} x ε => pure (Activation.safeLogSpec (α := α) x ε)
  sum := fun {_s} x => pure (Tensor.scalar (Tensor.sumSpec (α := α) x))
  flatten := fun {_s} x => pure (Tensor.flattenSpec (α := α) x)

  linear := fun {inDim outDim} w b x =>
    pure (Tensor.addSpec (α := α)
      (matVecMulSpec (α := α) (m := outDim) (n := inDim) w x) b)
  mseLoss := fun {s} yhat target =>
    pure (Tensor.scalar (Spec.mseSpec (α := α) (s := s) yhat target))

  layerNorm := fun {seqLen embedDim} hSeq hEmb x gamma beta =>
    pure (Spec.layerNorm (α := α) (seqLen := seqLen) (embedDim := embedDim)
      (x := x) (gamma := gamma) (beta := beta) (h_seq_pos := hSeq) (h_embed_pos := hEmb))

  batchnormChannelFirst := fun {channels height width} hC hH hW x gamma beta =>
    pure (Spec.batchNorm2d (α := α)
      (channels := channels) (height := height) (width := width)
      (x := x) (gamma := gamma) (beta := beta) (h_c := hC) (h_h := hH) (h_w := hW))

  multiHeadAttention := fun {n numHeads dModel headDim} h1 wq wk wv wo x mask =>
    -- Package the weight matrices into the spec-layer structure.
    let mha : Spec.MultiHeadAttention α numHeads dModel headDim :=
      { Wq := wq, Wk := wk, Wv := wv, Wo := wo }
    pure (Spec.MultiHeadAttention.forward (α := α) (numHeads := numHeads) (dModel := dModel)
      (headDim := headDim)
      (n := n) h1 mha x (mask := mask))

  conv := fun {d inC outC} {kernel stride padding} {inSpatial} {_hInC} {_hKernel} w b x =>
    let layer : Spec.ConvSpec d inC outC kernel stride padding α :=
      { kernel := w, bias := b }
    pure (Spec.convSpec (α := α) (d := d) (inC := inC) (outC := outC)
      (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
      (layer := layer) (input := x))
  convTranspose := fun {d inC outC} {kernel stride padding} {inSpatial} {_hInC} {_hKernel} w b x =>
    let layer : Spec.ConvTransposeSpec d inC outC kernel stride padding α :=
      { kernel := w, bias := b }
    pure (Spec.convTransposeSpec (α := α) (d := d) (inC := inC) (outC := outC)
      (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
      (layer := layer) (input := x))

  conv2d := fun {inC outC kH kW stride padding _inH _inW} {h1} {h2} {h3} kernel bias input =>
    let layer : Spec.Conv2DSpec inC outC kH kW stride padding α h1 h2 h3 :=
      { kernel := kernel, bias := bias }
    pure (Spec.conv2dSpec (α := α) (layer := layer) (input := input))

  convTranspose2d := fun {inC outC kH kW stride padding inH inW} {h1} {h2} {h3} kernel bias input =>
    let h1' : inC > 0 := Nat.pos_of_ne_zero h1
    let layer : Spec.ConvTranspose2DSpec inC outC kH kW stride padding α h1' h2 h3 :=
      { kernel := kernel, bias := bias }
    pure (Spec.convTranspose2dSpec (α := α) (inC := inC) (outC := outC) (kH := kH) (kW := kW)
      (stride := stride) (padding := padding) (inH := inH) (inW := inW) (layer := layer)
      (input := input))

  randUniform := fun {_s} _seed =>
    throw <|
      "TorchLeanSpecEval: rand_uniform is not supported in spec backend " ++
        "(needs a deterministic counter)"
  bernoulliMask := fun {_s} _keepProb _seed =>
    throw <|
      "TorchLeanSpecEval: bernoulli_mask is not supported in spec backend " ++
        "(needs a deterministic counter)"

/-- Convert a parameter `TList` into the spec-eval backend's `RefList` representation. -/
def refListOfTList {α : Type} [Context α] :
    {ss : List Shape} → Runtime.Autograd.Torch.TList α ss → Runtime.Autograd.Torch.RefList (fun s =>
      Tensor α s) ss
  | [], .nil => .nil
  | _s :: ss, .cons t ts => .cons t (refListOfTList (ss := ss) ts)

/-- Spec semantics for `compileForward1`-style models (one distinguished input, last argument). -/
def evalForward1Spec
    {α : Type} [Context α] [DecidableEq Shape] [Inhabited α]
    {paramShapes : List Shape} {inShape outShape : Shape}
    (model : Runtime.Autograd.TorchLean.Program α (paramShapes ++ [inShape]) outShape)
    (params : Runtime.Autograd.Torch.TList α paramShapes)
    (x : Tensor α inShape) : Except String (Tensor α outShape) := do
  let psRefs := refListOfTList (α := α) (ss := paramShapes) params
  let allRefs : Runtime.Autograd.Torch.RefList (fun s => Tensor α s) (paramShapes ++ [inShape]) :=
    Runtime.Autograd.Torch.RefList.append (Ref := fun s => Tensor α s)
      (ss₁ := paramShapes) (ss₂ := [inShape]) psRefs (.cons x .nil)
  Runtime.Autograd.Torch.CurriedRef.uncurry
    (Ref := fun s => Runtime.Autograd.TorchLean.RefTy (m := SpecM) (α := α) s)
    (ss := paramShapes ++ [inShape])
    (model (m := SpecM)) allRefs

end NN.Verification.TorchLean

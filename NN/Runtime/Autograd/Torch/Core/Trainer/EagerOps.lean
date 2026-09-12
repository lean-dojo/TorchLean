/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Functional.Ops
public import NN.Runtime.Autograd.Torch.Core.Ops.Convolution
public import NN.Runtime.Autograd.Torch.Core.Ops.Elementwise
public import NN.Runtime.Autograd.Torch.Core.Ops.Indexing
public import NN.Runtime.Autograd.Torch.Core.Ops.Pooling
public import NN.Runtime.Autograd.Torch.Core.Ops.Spectral
public import NN.Runtime.Autograd.Torch.Core.Trainer.Attention

/-!
# Eager Operations

Run an `Ops` program against the current tape. The instance lives here so evaluation and
recording can use it without importing trainer construction.
-/

@[expose] public section

namespace Runtime.Autograd.Torch

open Spec TorchLean TorchLean.Tensor

/--
Monad used for the eager `Ops` instance: read an `Internal.EagerSession α` and execute in `IO`.

This is the backend that makes `Ops` programs execute immediately by mutating a hidden runtime tape.
-/
abbrev Internal.EagerM (α : Type) [TorchLean.Storage α] :=
  ReaderT (Internal.EagerSession α) IO

/--
`Ops` instance for the eager Torch-style runtime.

This interprets `Ops` primitives by immediately executing them against the hidden mutable tape in
the current `Internal.EagerSession`.
-/
instance {α : Type} [TorchLean.Storage α] [Context α] [TensorTransfer α] :
    Ops (Internal.EagerM α) α where
  Ref := fun s => TensorRef α s
  DataRef := fun β _ s => Tensor β s
  dataConst := fun x => x
  mapData := fun f x => f x
  rfft1dNative? := some fun {batch n} x sess =>
    Internal.EagerSession.rfft1dNative? sess (batch := batch) (n := n) x
  irfft1dNative? := some fun {batch n} x sess =>
    Internal.EagerSession.irfft1dNative? sess (batch := batch) (n := n) x
  selectiveScanDiagNative? := some fun {seqLen state} a b x initial sess =>
    Internal.EagerSession.selectiveScanDiagNative? sess (seqLen := seqLen) (state := state)
      a b x initial
  selectiveScanDiagVarNative? := some fun {seqLen state} a b x initial sess =>
    Internal.EagerSession.selectiveScanDiagVarNative? sess (seqLen := seqLen) (state := state)
      a b x initial
  spectralConv1dRfftNative? := some fun {grid width modes} x realWeight imagWeight sess =>
    Internal.EagerSession.spectralConv1dRfftNative? sess (grid := grid) (width := width)
      (modes := modes) x realWeight imagWeight
  const := fun {s} t => fun sess => Internal.EagerSession.const (α := α) sess (sh := s) t
  add := fun {s} a b => fun sess => Internal.EagerSession.add (α := α) sess (sh := s) a b
  sub := fun {s} a b => fun sess => Internal.EagerSession.sub (α := α) sess (sh := s) a b
  mul := fun {s} a b => fun sess => Internal.EagerSession.mul (α := α) sess (sh := s) a b
  scale := fun {s} x c => fun sess => Internal.EagerSession.scale (α := α) sess (sh := s) x c
  abs := fun {s} x => fun sess => Internal.EagerSession.abs (α := α) sess (sh := s) x
  sqrt := fun {s} x => fun sess => Internal.EagerSession.sqrt (α := α) sess (sh := s) x
  clamp := fun {s} x minVal maxVal => fun sess =>
    Internal.EagerSession.clamp (α := α) sess (sh := s) x minVal maxVal
  max := fun {s} a b => fun sess => Internal.EagerSession.max (α := α) sess (sh := s) a b
  min := fun {s} a b => fun sess => Internal.EagerSession.min (α := α) sess (sh := s) a b
  broadcastTo := fun {s₁ s₂} cb x => fun sess =>
    Internal.EagerSession.broadcastTo (α := α) sess (sh1 := s₁) (sh2 := s₂) cb x
  reshape := fun {s₁ s₂} x h => fun sess =>
    Internal.EagerSession.reshape (α := α) sess (sh1 := s₁) (sh2 := s₂) x h
  swapAdjacentAtDepth := fun {s} depth x => fun sess =>
    Internal.EagerSession.swapAdjacentAtDepth (α := α) sess (sh := s) depth x
  reduceSum := fun {s} axis => fun x => fun sess =>
    Internal.EagerSession.reduceSum (α := α) sess (sh := s) axis x
  reduceMean := fun {s} axis => fun x => fun sess =>
    Internal.EagerSession.reduceMean (α := α) sess (sh := s) axis x
  select := fun {s} axis _axisInBounds x index => fun sess =>
    Internal.EagerSession.select (α := α) sess (shape := s) axis x index
  indexSelect := fun {s} axis count _axisInBounds x indices => fun sess =>
    Internal.EagerSession.indexSelect (α := α) sess (shape := s) axis count x indices
  scatterAdd := fun {s} axis count _axisInBounds base source indices => fun sess =>
    Internal.EagerSession.scatterAdd (α := α) sess (shape := s) axis count base source indices
  matmul := fun {batchA batchB batch : Shape} {mDim nDim pDim : Nat}
      {broadcastA} {broadcastB} a b => fun sess =>
    Internal.EagerSession.matmul (α := α) sess (batchA := batchA) (batchB := batchB)
      (batch := batch) (m := mDim) (n := nDim) (p := pDim)
      (broadcastA := broadcastA) (broadcastB := broadcastB) a b
  concatLeadingAxis := fun {nDim mDim} {s} a b => fun sess =>
    Internal.EagerSession.concatLeadingAxis (α := α) sess (n := nDim) (m := mDim) (sh := s) a b
  sliceLeadingAxisRange := fun {nDim} {s} start len h x => fun sess =>
    Internal.EagerSession.sliceLeadingAxisRange (α := α) sess (n := nDim) (sh := s) x start len h
  maxPool := fun {d C} {inSpatial kernel stride padding} x => fun sess =>
    Internal.EagerSession.maxPool (α := α) sess
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      x
  avgPool := fun {d C} {inSpatial kernel stride padding} x => fun sess =>
    Internal.EagerSession.avgPool (α := α) sess
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      x
  smoothMaxPool := fun {d C} {inSpatial kernel stride padding}
      [_decidableEq : DecidableEq α] x beta => fun sess =>
    Internal.EagerSession.smoothMaxPool (α := α) sess
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      x beta
  relu := fun {s} x => fun sess => Internal.EagerSession.relu (α := α) sess (sh := s) x
  sigmoid := fun {s} x => fun sess => Internal.EagerSession.sigmoid (α := α) sess (sh := s) x
  tanh := fun {s} x => fun sess => Internal.EagerSession.tanh (α := α) sess (sh := s) x
  gelu := fun {s} x => fun sess => Internal.EagerSession.gelu (α := α) sess (sh := s) x
  softmaxLast := fun {s} x => fun sess =>
    Internal.EagerSession.softmaxLast (α := α) sess (sh := s) x
  logSoftmaxLast := fun {s} x => fun sess =>
    Internal.EagerSession.logSoftmaxLast (α := α) sess (sh := s) x
  softplus := fun {s} x => fun sess => Internal.EagerSession.softplus (α := α) sess (sh := s) x
  exp := fun {s} x => fun sess => Internal.EagerSession.exp (α := α) sess (sh := s) x
  sin := fun {s} x => fun sess => Internal.EagerSession.sin (α := α) sess (sh := s) x
  cos := fun {s} x => fun sess => Internal.EagerSession.cos (α := α) sess (sh := s) x
  log := fun {s} x => fun sess => Internal.EagerSession.log (α := α) sess (sh := s) x
  inv := fun {s} x => fun sess => Internal.EagerSession.inv (α := α) sess (sh := s) x
  detach := fun {s} x => fun sess => Internal.EagerSession.detach (α := α) sess (sh := s) x
  safeLog := fun {s} x ε => fun sess => Internal.EagerSession.safeLog (α := α) sess (sh := s) x (ε
    := ε)
  sum := fun {s} x => fun sess => Internal.EagerSession.sum (α := α) sess (sh := s) x
  flatten := fun {s} x => fun sess => Internal.EagerSession.flatten (α := α) sess (sh := s) x
  linear := fun {inDim outDim} w b x => fun sess =>
    Internal.EagerSession.linear (α := α) sess (inDim := inDim) (outDim := outDim) w b x
  mseLoss := fun {s} yhat target => fun sess => Internal.EagerSession.mseLoss (α := α) sess (sh :=
    s) yhat target
  layerNorm := fun {seqLen embedDim} hSeq hEmb x gamma beta epsilon => fun sess =>
    Internal.EagerSession.layerNorm (α := α) sess (seqLen := seqLen) (embedDim := embedDim)
      (h_seq_pos := hSeq) (h_embed_pos := hEmb) x gamma beta (epsilon := epsilon)
  batchNorm := fun {channels sSpatial} hWellFormed x gamma beta epsilon => fun sess =>
    Internal.EagerSession.batchNorm (α := α) sess
      (channels := channels) (sSpatial := sSpatial) hWellFormed x gamma beta (epsilon := epsilon)
  multiHeadAttention := fun {n numHeads dModel headDim} h1 wq wk wv wo x mask => fun sess =>
    Internal.EagerSession.multiHeadAttention (α := α) sess (n := n) (numHeads := numHeads)
      (dModel := dModel) (headDim := headDim) h1 wq wk wv wo x (mask := mask)
  batchedMultiHeadAttention :=
    fun {batch n numHeads dModel headDim} hBatch h1 wq wk wv wo x mask => fun sess =>
      Internal.EagerSession.batchedMultiHeadAttention (α := α) sess
        (batch := batch) (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
        hBatch h1 wq wk wv wo x (mask := mask)
  conv := fun {d inC outC} {kernel stride padding} {inSpatial} w b x => fun sess =>
    Internal.EagerSession.conv (α := α) sess
      (d := d) (inC := inC) (outC := outC)
      (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
      w b x
  convTranspose := fun {d inC outC} {kernel stride padding} {inSpatial} w b x =>
    fun sess =>
      Internal.EagerSession.convTranspose (α := α) sess
        (d := d) (inC := inC) (outC := outC)
        (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
        w b x
  randUniform := fun {s} seed => fun sess =>
    Internal.EagerSession.randUniform (α := α) sess (sh := s) seed
  bernoulliMask := fun {s} keepProb seed => fun sess =>
    Internal.EagerSession.bernoulliMask (α := α) sess (sh := s) keepProb seed

end Runtime.Autograd.Torch

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Functional

/-!
# Torch Trainer Helpers

`Ops` instances, parameter lists, and scalar trainer construction for the Torch-style runtime. This
is the bridge from backend-generic model code to executable training loops.
-/


@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec
open Tensor
open Proofs.Autograd.Algebra

/--
Monad used for the eager `Ops` instance: read an `Internal.EagerSession α` and execute in `IO`.

This is the backend that makes `Ops` programs execute immediately by mutating a hidden runtime tape.
-/
abbrev Internal.EagerM (α : Type) := ReaderT (Internal.EagerSession α) IO

/--
`Ops` instance for the eager Torch-style runtime.

This interprets `Ops` primitives by immediately executing them against the hidden mutable tape in
the current `Internal.EagerSession`.
-/
instance {α : Type} [Context α] [Internal.CudaBridge.TensorConv α] [DecidableEq Shape] :
    Ops (Internal.EagerM α) α where
  Ref := fun s => TensorRef α s
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
  transpose2d := fun {mDim nDim} x => fun sess =>
    Internal.EagerSession.transpose2d (α := α) sess (m := mDim) (n := nDim) x
  transpose3dFirstToLast := fun {a b c} x => fun sess =>
    Internal.EagerSession.transpose3dFirstToLast (α := α) sess (a := a) (b := b) (c := c) x
  transpose3dLastToFirst := fun {a b c} x => fun sess =>
    Internal.EagerSession.transpose3dLastToFirst (α := α) sess (a := a) (b := b) (c := c) x
  transpose3dLastTwo := fun {a b c} x => fun sess =>
    Internal.EagerSession.transpose3dLastTwo (α := α) sess (a := a) (b := b) (c := c) x
  swapAdjacentAtDepth := fun {s} depth x => fun sess =>
    Internal.EagerSession.swapAdjacentAtDepth (α := α) sess (sh := s) depth x
  reduceSum := fun {s} axis => fun x => fun sess =>
    Internal.EagerSession.reduceSum (α := α) sess (sh := s) axis x
  reduceMean := fun {s} axis => fun x => fun sess =>
    Internal.EagerSession.reduceMean (α := α) sess (sh := s) axis x
  gatherScalar := fun {n} x i => fun sess =>
    Internal.EagerSession.gatherScalar (α := α) sess (n := n) x i
  gatherRow := fun {rows cols} x i => fun sess =>
    Internal.EagerSession.gatherRow (α := α) sess (rows := rows) (cols := cols) x i
  gatherScalarNat := fun {n} x i => fun sess =>
    Internal.EagerSession.gatherScalarNat (α := α) sess (n := n) x i
  gatherVecNat := fun {n k} x idx => fun sess =>
    Internal.EagerSession.gatherVecNat (α := α) sess (n := n) (k := k) x idx
  gatherRowsNat := fun {rows cols k} x idx => fun sess =>
    Internal.EagerSession.gatherRowsNat (α := α) sess (rows := rows) (cols := cols) (k := k) x idx
  -- Eager execution can inspect concrete float input values, so it is the backend that validates
  -- token-id batches before embedding lookup or cross entropy.
  tokenIdsFromFloatVec := fun {k} x => fun sess =>
    Internal.EagerSession.tokenIdsFromFloatVec (α := α) sess (k := k) x
  scatterAddVec := fun {n} x v i => fun sess =>
    Internal.EagerSession.scatterAddVec (α := α) sess (n := n) x v i
  scatterAddRow := fun {rows cols} x v i => fun sess =>
    Internal.EagerSession.scatterAddRow (α := α) sess (rows := rows) (cols := cols) x v i
  matmul := fun {mDim nDim pDim} a b => fun sess =>
    Internal.EagerSession.matmul (α := α) sess (m := mDim) (n := nDim) (p := pDim) a b
  bmm := fun {batch mDim nDim pDim} a b => fun sess =>
    Internal.EagerSession.bmm (α := α) sess (batch := batch) (m := mDim) (n := nDim) (p := pDim) a b
  concatVectors := fun {nDim mDim} a b => fun sess =>
    Internal.EagerSession.concatVectors (α := α) sess (n := nDim) (m := mDim) a b
  concatLeadingAxis := fun {nDim mDim} {s} a b => fun sess =>
    Internal.EagerSession.concatLeadingAxis (α := α) sess (n := nDim) (m := mDim) (sh := s) a b
  sliceLeadingAxisRange := fun {nDim} {s} start len h x => fun sess =>
    Internal.EagerSession.sliceLeadingAxisRange (α := α) sess (n := nDim) (sh := s) x start len h
  maxPool := fun {d C} {inSpatial kernel stride padding} {hKernel} x => fun sess =>
    Internal.EagerSession.maxPool (α := α) sess
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      (hKernel := hKernel)
      x
  avgPool := fun {d C} {inSpatial kernel stride padding} hKernel x => fun sess =>
    Internal.EagerSession.avgPool (α := α) sess
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      hKernel
      x
  smoothMaxPool := fun {d C} {inSpatial kernel stride padding} {hKernel} x beta => fun sess =>
    Internal.EagerSession.smoothMaxPool (α := α) sess
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      (hKernel := hKernel)
      x beta
  maxPool2d := fun {kH kW inH inW inC stride} {h1 h2} x => fun sess =>
    Internal.EagerSession.maxPool2d (α := α) sess
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      (h1 := h1) (h2 := h2) x
  maxPool2dPad := fun {kH kW inH inW inC stride padding} {h1 h2} x => fun sess =>
    Internal.EagerSession.maxPool2dPad (α := α) sess
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding :=
        padding)
      (h1 := h1) (h2 := h2) x
  smoothMaxPool2d := fun {kH kW inH inW inC stride} {h1 h2} x beta => fun sess =>
    Internal.EagerSession.smoothMaxPool2d (α := α) sess
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      (h1 := h1) (h2 := h2) x beta
  avgPool2d := fun {kH kW inH inW inC stride} h1 h2 x => fun sess =>
    Internal.EagerSession.avgPool2d (α := α) sess
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      h1 h2 x
  avgPool2dPad := fun {kH kW inH inW inC stride padding} h1 h2 x => fun sess =>
    Internal.EagerSession.avgPool2dPad (α := α) sess
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding :=
        padding)
      h1 h2 x
  relu := fun {s} x => fun sess => Internal.EagerSession.relu (α := α) sess (sh := s) x
  sigmoid := fun {s} x => fun sess => Internal.EagerSession.sigmoid (α := α) sess (sh := s) x
  tanh := fun {s} x => fun sess => Internal.EagerSession.tanh (α := α) sess (sh := s) x
  softmax := fun {s} x => fun sess => Internal.EagerSession.softmax (α := α) sess (sh := s) x
  logSoftmax := fun {s} x => fun sess => Internal.EagerSession.logSoftmax (α := α) sess (sh := s) x
  softplus := fun {s} x => fun sess => Internal.EagerSession.softplus (α := α) sess (sh := s) x
  exp := fun {s} x => fun sess => Internal.EagerSession.exp (α := α) sess (sh := s) x
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
  layerNorm := fun {seqLen embedDim} hSeq hEmb x gamma beta => fun sess =>
    Internal.EagerSession.layerNorm (α := α) sess (seqLen := seqLen) (embedDim := embedDim)
      (h_seq_pos := hSeq) (h_embed_pos := hEmb) x gamma beta
  batchnormChannelFirst := fun {channels height width} hC hH hW x gamma beta => fun sess =>
    Internal.EagerSession.batchnormChannelFirst (α := α) sess
      (channels := channels) (height := height) (width := width) (h_c := hC) (h_h := hH) (h_w := hW)
      x gamma beta
  multiHeadAttention := fun {n numHeads dModel headDim} h1 wq wk wv wo x mask => fun sess =>
    Internal.EagerSession.multiHeadAttention (α := α) sess (n := n) (numHeads := numHeads)
      (dModel := dModel) (headDim := headDim) h1 wq wk wv wo x (mask := mask)
  conv := fun {d inC outC} {kernel stride padding} {inSpatial} {hInC hKernel} w b x => fun sess =>
    Internal.EagerSession.conv (α := α) sess
      (d := d) (inC := inC) (outC := outC)
      (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
      (hInC := hInC) (hKernel := hKernel)
      w b x
  convTranspose := fun {d inC outC} {kernel stride padding} {inSpatial} {hInC hKernel} w b x =>
    fun sess =>
      Internal.EagerSession.convTranspose (α := α) sess
        (d := d) (inC := inC) (outC := outC)
        (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
        (hInC := hInC) (hKernel := hKernel)
        w b x
  conv2d := fun {inC outC kH kW stride padding inH inW} {h1 h2 h3} kernel bias input => fun sess =>
    Internal.EagerSession.conv2d (α := α) sess (inC := inC) (outC := outC) (kH := kH) (kW := kW)
      (stride := stride) (padding := padding) (inH := inH) (inW := inW) (h1 := h1) (h2 := h2) (h3 :=
        h3)
      kernel bias input
  convTranspose2d := fun {inC outC kH kW stride padding inH inW} {h1 h2 h3} kernel bias input =>
    fun sess =>
      Internal.EagerSession.convTranspose2d (α := α) sess
        (inC := inC) (outC := outC) (kH := kH) (kW := kW)
        (stride := stride) (padding := padding) (inH := inH) (inW := inW)
        (h1 := h1) (h2 := h2) (h3 := h3)
        kernel bias input
  randUniform := fun {s} seed => fun sess =>
    Internal.EagerSession.randUniform (α := α) sess (sh := s) seed
  bernoulliMask := fun {s} keepProb seed => fun sess =>
    Internal.EagerSession.bernoulliMask (α := α) sess (sh := s) keepProb seed

/--
`Ops` instance for the compiled graph-building monad `GraphM`.

This interprets `Ops` primitives by *recording* typed IR nodes (rather than executing immediately).
See `Runtime.Autograd.Compiled.GraphM` and `Torch.LinkedSession` for how these graphs are later run.
-/
instance {α : Type} [Context α] [DecidableEq Shape] {Γ : List Shape} :
    Ops (Runtime.Autograd.Compiled.GraphM.M α Γ) α where
  Ref := fun s => Runtime.Autograd.Compiled.GraphM.Var s
  const := fun {s} t => Runtime.Autograd.Compiled.GraphM.const (α := α) (Γ := Γ) (s := s) t
  add := fun {s} a b => Runtime.Autograd.Compiled.GraphM.add (α := α) (Γ := Γ) (s := s) a b
  sub := fun {s} a b => Runtime.Autograd.Compiled.GraphM.sub (α := α) (Γ := Γ) (s := s) a b
  mul := fun {s} a b => Runtime.Autograd.Compiled.GraphM.mul (α := α) (Γ := Γ) (s := s) a b
  scale := fun {s} x c => Runtime.Autograd.Compiled.GraphM.scale (α := α) (Γ := Γ) (s := s) x c
  abs := fun {s} x => Runtime.Autograd.Compiled.GraphM.abs (α := α) (Γ := Γ) (s := s) x
  sqrt := fun {s} x => Runtime.Autograd.Compiled.GraphM.sqrt (α := α) (Γ := Γ) (s := s) x
  clamp := fun {s} x minVal maxVal =>
    Runtime.Autograd.Compiled.GraphM.clamp (α := α) (Γ := Γ) (s := s) x minVal maxVal
  max := fun {s} a b => Runtime.Autograd.Compiled.GraphM.max (α := α) (Γ := Γ) (s := s) a b
  min := fun {s} a b => Runtime.Autograd.Compiled.GraphM.min (α := α) (Γ := Γ) (s := s) a b
  broadcastTo := fun {s₁ s₂} cb x =>
    Runtime.Autograd.Compiled.GraphM.broadcastTo (α := α) (Γ := Γ) (s₁ := s₁) (s₂ := s₂) cb x
  reshape := fun {s₁ s₂} x h =>
    Runtime.Autograd.Compiled.GraphM.reshape (α := α) (Γ := Γ) (s₁ := s₁) (s₂ := s₂) x h
  transpose2d := fun {mDim nDim} x =>
    Runtime.Autograd.Compiled.GraphM.transpose2d (α := α) (Γ := Γ) (m := mDim) (n := nDim) x
  transpose3dFirstToLast := fun {a b c} x =>
    Runtime.Autograd.Compiled.GraphM.transpose3dFirstToLast (α := α) (Γ := Γ) (a := a) (b := b)
      (c := c) x
  transpose3dLastToFirst := fun {a b c} x =>
    Runtime.Autograd.Compiled.GraphM.transpose3dLastToFirst (α := α) (Γ := Γ) (a := a) (b := b)
      (c := c) x
  transpose3dLastTwo := fun {a b c} x =>
    Runtime.Autograd.Compiled.GraphM.transpose3dLastTwo (α := α) (Γ := Γ) (a := a) (b := b) (c :=
      c) x
  swapAdjacentAtDepth := fun {s} depth x =>
    Runtime.Autograd.Compiled.GraphM.swapAdjacentAtDepth (α := α) (Γ := Γ) (s := s) depth x
  reduceSum := fun {s} axis => fun x =>
    Runtime.Autograd.Compiled.GraphM.reduceSum (α := α) (Γ := Γ) (s := s) axis x
  reduceMean := fun {s} axis => fun x =>
    Runtime.Autograd.Compiled.GraphM.reduceMean (α := α) (Γ := Γ) (s := s) axis x
  gatherScalar := fun {n} x i =>
    Runtime.Autograd.Compiled.GraphM.gatherScalar (α := α) (Γ := Γ) (n := n) x i
  gatherRow := fun {rows cols} x i =>
    Runtime.Autograd.Compiled.GraphM.gatherRow (α := α) (Γ := Γ) (rows := rows) (cols := cols) x i
  gatherScalarNat := fun {n} x i =>
    Runtime.Autograd.Compiled.GraphM.gatherScalarNat (α := α) (Γ := Γ) (n := n) x i
  gatherVecNat := fun {n k} x idx =>
    Runtime.Autograd.Compiled.GraphM.gatherVecNat (α := α) (Γ := Γ) (n := n) (k := k) x idx
  gatherRowsNat := fun {rows cols k} x idx =>
    Runtime.Autograd.Compiled.GraphM.gatherRowsNat (α := α) (Γ := Γ) (rows := rows) (cols := cols)
      (k := k) x idx
  -- Dynamic token-id parsing reads runtime tensor contents. The compiled graph builder records
  -- symbolic tensor operations, so this adapter must stay on the eager path for now.
  tokenIdsFromFloatVec := fun {_k} _x =>
    throw "compiled GraphM: tokenIdsFromFloatVec requires eager backend (dynamic token ids)"
  scatterAddVec := fun {n} x v i =>
    Runtime.Autograd.Compiled.GraphM.scatterAddVec (α := α) (Γ := Γ) (n := n) x v i
  scatterAddRow := fun {rows cols} x v i =>
    Runtime.Autograd.Compiled.GraphM.scatterAddRow (α := α) (Γ := Γ) (rows := rows) (cols := cols)
      x v i
  matmul := fun {mDim nDim pDim} a b =>
    Runtime.Autograd.Compiled.GraphM.matmul (α := α) (Γ := Γ) (m := mDim) (n := nDim) (p := pDim) a
      b
  bmm := fun {batch mDim nDim pDim} a b =>
    Runtime.Autograd.Compiled.GraphM.bmm (α := α) (Γ := Γ) (batch := batch) (m := mDim) (n := nDim)
      (p := pDim) a b
  concatVectors := fun {nDim mDim} a b =>
    Runtime.Autograd.Compiled.GraphM.concatVectors (α := α) (Γ := Γ) (n := nDim) (m := mDim) a b
  concatLeadingAxis := fun {nDim mDim} {s} a b =>
    Runtime.Autograd.Compiled.GraphM.concatLeadingAxis (α := α) (Γ := Γ) (n := nDim) (m := mDim) (s := s)
      a b
  sliceLeadingAxisRange := fun {nDim} {s} start len h x =>
    Runtime.Autograd.Compiled.GraphM.sliceLeadingAxisRange (α := α) (Γ := Γ) (n := nDim) (s := s) x start len
      h
  maxPool := fun {d C} {inSpatial kernel stride padding} {hKernel} x =>
    Runtime.Autograd.Compiled.GraphM.maxPool (α := α) (Γ := Γ)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      (hKernel := hKernel)
      x
  avgPool := fun {d C} {inSpatial kernel stride padding} hKernel x =>
    Runtime.Autograd.Compiled.GraphM.avgPool (α := α) (Γ := Γ)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      (hKernel := hKernel)
      x
  smoothMaxPool := fun {d C} {inSpatial kernel stride padding} {hKernel} x beta =>
    Runtime.Autograd.Compiled.GraphM.smoothMaxPool (α := α) (Γ := Γ)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      (hKernel := hKernel)
      x beta
  maxPool2d := fun {kH kW inH inW inC stride} {h1 h2} x =>
    Runtime.Autograd.Compiled.GraphM.maxPool2d (α := α) (Γ := Γ)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      (h1 := h1) (h2 := h2) x
  maxPool2dPad := fun {kH kW inH inW inC stride padding} {h1 h2} x =>
    Runtime.Autograd.Compiled.GraphM.maxPool2dPad (α := α) (Γ := Γ)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding :=
        padding)
      (h1 := h1) (h2 := h2) x
  smoothMaxPool2d := fun {kH kW inH inW inC stride} {h1 h2} x beta =>
    Runtime.Autograd.Compiled.GraphM.smoothMaxPool2d (α := α) (Γ := Γ)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      (h1 := h1) (h2 := h2) x beta
  avgPool2d := fun {kH kW inH inW inC stride} h1 h2 x =>
    Runtime.Autograd.Compiled.GraphM.avgPool2d (α := α) (Γ := Γ)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      h1 h2 x
  avgPool2dPad := fun {kH kW inH inW inC stride padding} h1 h2 x =>
    Runtime.Autograd.Compiled.GraphM.avgPool2dPad (α := α) (Γ := Γ)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding :=
        padding)
      h1 h2 x
  relu := fun {s} x => Runtime.Autograd.Compiled.GraphM.relu (α := α) (Γ := Γ) (s := s) x
  sigmoid := fun {s} x => Runtime.Autograd.Compiled.GraphM.sigmoid (α := α) (Γ := Γ) (s := s) x
  tanh := fun {s} x => Runtime.Autograd.Compiled.GraphM.tanh (α := α) (Γ := Γ) (s := s) x
  softmax := fun {s} x => Runtime.Autograd.Compiled.GraphM.softmax (α := α) (Γ := Γ) (s := s) x
  logSoftmax := fun {s} x => Runtime.Autograd.Compiled.GraphM.logSoftmax (α := α) (Γ := Γ) (s := s)
    x
  softplus := fun {s} x => Runtime.Autograd.Compiled.GraphM.softplus (α := α) (Γ := Γ) (s := s) x
  exp := fun {s} x => Runtime.Autograd.Compiled.GraphM.exp (α := α) (Γ := Γ) (s := s) x
  log := fun {s} x => Runtime.Autograd.Compiled.GraphM.log (α := α) (Γ := Γ) (s := s) x
  inv := fun {s} x => Runtime.Autograd.Compiled.GraphM.inv (α := α) (Γ := Γ) (s := s) x
  detach := fun {s} x => Runtime.Autograd.Compiled.GraphM.detach (α := α) (Γ := Γ) (s := s) x
  safeLog := fun {s} x ε => Runtime.Autograd.Compiled.GraphM.safeLog (α := α) (Γ := Γ) (s := s) x
    (ε := ε)
  sum := fun {s} x => Runtime.Autograd.Compiled.GraphM.sum (α := α) (Γ := Γ) (s := s) x
  flatten := fun {s} x => Runtime.Autograd.Compiled.GraphM.flatten (α := α) (Γ := Γ) (s := s) x
  linear := fun {inDim outDim} w b x =>
    Runtime.Autograd.Compiled.GraphM.linear (α := α) (Γ := Γ) (inDim := inDim) (outDim := outDim) w
      b x
  mseLoss := fun {s} yhat target =>
    Runtime.Autograd.Compiled.GraphM.mseLoss (α := α) (Γ := Γ) (s := s) yhat target
  layerNorm := fun {seqLen embedDim} hSeq hEmb x gamma beta =>
    Runtime.Autograd.Compiled.GraphM.layerNorm (α := α) (Γ := Γ) (seqLen := seqLen) (embedDim :=
      embedDim)
      (h_seq_pos := hSeq) (h_embed_pos := hEmb) x gamma beta
  batchnormChannelFirst := fun {channels height width} hC hH hW x gamma beta =>
    Runtime.Autograd.Compiled.GraphM.batchnormChannelFirst (α := α) (Γ := Γ)
      (channels := channels) (height := height) (width := width) (h_c := hC) (h_h := hH) (h_w := hW)
      x gamma beta
  multiHeadAttention := fun {n numHeads dModel headDim} h1 wq wk wv wo x mask =>
    Runtime.Autograd.Compiled.GraphM.multiHeadAttention (α := α) (Γ := Γ) (n := n) (numHeads :=
      numHeads)
      (dModel := dModel) (headDim := headDim) h1 wq wk wv wo x (mask := mask)
  conv := fun {d inC outC} {kernel stride padding} {inSpatial} {hInC hKernel} w b x =>
    Runtime.Autograd.Compiled.GraphM.conv (α := α) (Γ := Γ)
      (d := d) (inC := inC) (outC := outC)
      (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
      (hInC := hInC) (hKernel := hKernel)
      w b x
  convTranspose := fun {d inC outC} {kernel stride padding} {inSpatial} {hInC hKernel} w b x =>
    Runtime.Autograd.Compiled.GraphM.convTranspose (α := α) (Γ := Γ)
      (d := d) (inC := inC) (outC := outC)
      (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
      (hInC := hInC) (hKernel := hKernel)
      w b x
  conv2d := fun {inC outC kH kW stride padding inH inW} {h1 h2 h3} kernel bias input =>
    Runtime.Autograd.Compiled.GraphM.conv2d (α := α) (Γ := Γ) (inC := inC) (outC := outC) (kH := kH)
      (kW := kW)
      (stride := stride) (padding := padding) (inH := inH) (inW := inW) (h1 := h1) (h2 := h2) (h3 :=
        h3)
      kernel bias input
  convTranspose2d := fun {inC outC kH kW stride padding inH inW} {h1 h2 h3} kernel bias input =>
    Runtime.Autograd.Compiled.GraphM.convTranspose2d (α := α) (Γ := Γ)
      (inC := inC) (outC := outC) (kH := kH) (kW := kW)
      (stride := stride) (padding := padding) (inH := inH) (inW := inW)
      (h1 := h1) (h2 := h2) (h3 := h3)
      kernel bias input
  randUniform := fun {s} seed => do
    Runtime.Autograd.Compiled.GraphM.randUniform (α := α) (Γ := Γ) (s := s) (seed := seed)
  bernoulliMask := fun {s} keepProb seed => do
    Runtime.Autograd.Compiled.GraphM.bernoulliMask (α := α) (Γ := Γ) (s := s) keepProb (seed :=
      seed)

/--
Heterogeneous list of trainable parameters, indexed by a list of shapes.

This is the Torch front-end analogue of "a parameter vector" (like `model.parameters()` in PyTorch),
but with shapes tracked at the type level.
-/
inductive ParamList (α : Type) : List Shape → Type where
  | nil : ParamList α []
  | cons {s : Shape} {ss : List Shape} : Param α s → ParamList α ss → ParamList α (s :: ss)

namespace ParamList

/--
Materialize the SGD update `v - lr * g` in a single traversal.

This is used by `sgdStep_fast` as a runtime-performance optimization to avoid building deep thunk
chains when training for many steps.
-/
  def subScaleMaterialize {α : Type} [Sub α] [Mul α] :
    {s : Shape} → Tensor α s → Tensor α s → α → Tensor α s
  | .scalar, .scalar v, .scalar g, lr =>
      Tensor.scalar (v - (lr * g))
  | .dim n s', .dim fv, .dim fg, lr =>
      let arr : Array (Tensor α s') := Array.ofFn (fun i : Fin n => subScaleMaterialize (s := s')
        (fv i) (fg i) lr)
      Tensor.dim (fun i =>
        let hn : arr.size = n := by
          simp [arr]
        let hi : i.1 < arr.size :=
          Eq.ndrec (motive := fun m => i.1 < m) i.2 hn.symm
        arr[i.1]'hi)

/--
Allocate a fresh `ParamList` from an initial `TList` of parameter tensors.

Each tensor becomes an `IO.Ref` so it can be updated by optimizer steps.
-/
def ofTList {α : Type} {ss : List Shape} (xs : TList α ss) : IO (ParamList α ss) := do
  match xs with
  | .nil => pure .nil
  | .cons x xs =>
      let r ← IO.mkRef x
      let cudaValue ← IO.mkRef (none : Option Runtime.Autograd.Cuda.AnyBuffer)
      let hostCurrent ← IO.mkRef true
      let p : Param α _ := { value := r, cudaValue := cudaValue, hostCurrent := hostCurrent }
      let ps ← ofTList (α := α) xs
      pure (.cons p ps)

/--
Allocate a fresh `ParamList` from an initial `TList` of parameter tensors, with explicit
`requiresGrad` flags.

Returns an error when the flag list length does not match the parameter shape list length.
-/
def ofTListWithRequiresGrad {α : Type} :
    {ss : List Shape} → TList α ss → List Bool → IO (ParamList α ss)
  | [], .nil, [] => pure .nil
  | _s :: ss, .cons x xs, rg :: rgs => do
      let r ← IO.mkRef x
      let cudaValue ← IO.mkRef (none : Option Runtime.Autograd.Cuda.AnyBuffer)
      let hostCurrent ← IO.mkRef true
      let p : Param α _ :=
        { value := r, cudaValue := cudaValue, hostCurrent := hostCurrent, requiresGrad := rg }
      let ps ← ofTListWithRequiresGrad (α := α) (ss := ss) xs rgs
      pure (.cons p ps)
  | [], .nil, _ =>
      throw <| IO.userError "torch: requiresGrad list longer than parameter list"
  | _ :: _, .cons _ _, [] =>
      throw <| IO.userError "torch: requiresGrad list shorter than parameter list"

/-- Read the current parameter values as a `TList` aligned with the shape list. -/
def values {α : Type} : {ss : List Shape} → ParamList α ss → IO (TList α ss)
  | [], .nil => pure .nil
  | _s :: ss, .cons p ps => do
      let v ← p.value.get
      let vs ← values (α := α) (ss := ss) ps
      pure (.cons v vs)

/-- Read parameter values, synchronizing CUDA-resident mirrors first when necessary. -/
def valuesSynced {α : Type} [Internal.CudaBridge.TensorConv α] [DecidableEq Shape] :
    {ss : List Shape} → ParamList α ss → IO (TList α ss)
  | [], .nil => pure .nil
  | _s :: ss, .cons p ps => do
      Internal.syncParamCudaToHost (α := α) (sh := _s) p
      let v ← p.value.get
      let vs ← valuesSynced (α := α) (ss := ss) ps
      pure (.cons v vs)

/-- Overwrite the current parameter values from a `TList` aligned with the shape list. -/
def setValues {α : Type} : {ss : List Shape} → ParamList α ss → TList α ss → IO Unit
  | [], .nil, .nil => pure ()
  | _s :: ss, .cons p ps, .cons v vs => do
      Internal.setParamHostValue (α := α) (sh := _s) p v
      setValues (α := α) (ss := ss) ps vs

/--
Apply an SGD step `p := p - lr * g` to each parameter that has `requiresGrad = true`.

`gs` must be aligned with the parameter shapes.
-/
def sgdStep {α : Type} [Context α] : {ss : List Shape} → ParamList α ss → (lr : α) → TList α ss → IO
  Unit
  | [], .nil, _lr, .nil => pure ()
  | _s :: ss, .cons p ps, lr, .cons g gs => do
      if p.requiresGrad then
        let v ← p.value.get
        let updated : Tensor α _s :=
          -- Materialize `v - lr * g` in one pass so long runs do not retain expression closures.
          subScaleMaterialize (α := α) (s := _s) v g lr
        Internal.setParamHostValue (α := α) (sh := _s) p updated
      sgdStep (α := α) (ss := ss) ps lr gs

end ParamList

/--
Bundle a scalar-loss training loop for a fixed parameter pack and input signature.

This is the low-level trainer object used by module-backed execution:
- `forward` computes a scalar loss,
- `backward` computes gradients w.r.t. parameters,
- `step` applies an optimizer update (typically SGD),
- `getParams` reads current parameter values.
-/
structure ScalarTrainer (α : Type) (paramShapes inputShapes : List Shape) where
  /-- Mutable trainable parameter pack. -/
  params : ParamList α paramShapes
  /-- Compute the scalar loss for a curried input pack. -/
  forward : Curried.Fn α inputShapes (IO (Tensor α Shape.scalar))
  /-- Compute gradients aligned with `paramShapes` for a curried input pack. -/
  backward : Curried.Fn α inputShapes (IO (TList α paramShapes))
  /-- Apply one SGD-style update for a curried input pack. -/
  step : α → Curried.Fn α inputShapes (IO Unit)
  /--
  Optional Adam update path.

  In eager CUDA mode this is a device-gradient/device-moment update path.  Other backends expose
  `none` and should use the generic optimizer wrappers.
  -/
  adamStep? : Option (α → α → α → α → Curried.Fn α inputShapes (IO Unit)) := none
  /--
  Optional AdamW update path.

  In eager CUDA mode this is a device-gradient/device-moment update path with decoupled weight
  decay. Other backends expose `none` and should use the generic optimizer wrappers.
  -/
  adamWStep? : Option (α → α → α → α → α → Curried.Fn α inputShapes (IO Unit)) := none
  /-- Read current parameter values, synchronizing device mirrors if needed. -/
  getParams : IO (TList α paramShapes)

namespace Internal

/--
Extract gradients (as a typed `TList`) for a list of eager `TensorRef`s from a dense gradient array.
-/
def gradsOfRefs {α : Type} [DecidableEq Shape] :
    {ss : List Shape} → Array (Runtime.AnyTensor α) → RefList (TensorRef α) ss → IO (TList α ss)
  | [], _grads, .nil => pure .nil
  | s :: ss, grads, .cons r rs => do
      let g ← Internal.EagerSession.grad (α := α) (sh := s) grads r
      let gs ← gradsOfRefs (α := α) (ss := ss) grads rs
      pure (.cons g gs)

/--
Record all parameters as tape leaves in an eager session, returning their corresponding
  `TensorRef`s.

This is the eager analogue of "using" a parameter pack during a forward pass.
-/
def useParams {α : Type} [CudaBridge.TensorConv α] [DecidableEq Shape] :
    {ss : List Shape} → ParamList α ss → EagerM α (RefList (TensorRef α) ss)
  | [], .nil => pure .nil
  | s :: ss, .cons p ps => fun sess => do
      let r ← Internal.EagerSession.use (α := α) (sh := s) sess p
      let rs ← useParams (α := α) (ss := ss) ps sess
      pure (.cons r rs)

/--
Record all input tensors as tape leaves in an eager session, returning their corresponding
  `TensorRef`s.
-/
def useInputs {α : Type} [CudaBridge.TensorConv α] [DecidableEq Shape] :
    {ss : List Shape} → TList α ss → EagerM α (RefList (TensorRef α) ss)
  | [], .nil => pure .nil
  | s :: ss, .cons x xs => fun sess => do
      let r ← Internal.EagerSession.input (α := α) (sh := s) sess x
      let rs ← useInputs (α := α) (ss := ss) xs sess
      pure (.cons r rs)

end Internal

/--
Build a `ScalarTrainer` from an initial parameter pack and a backend-generic loss definition.

`loss` is written once against the `Ops` interface over a concatenated context
`paramShapes ++ inputShapes`. Depending on `opts.backend`, we either:
- compile the loss once (compiled backend), or
- execute it eagerly by building a runtime tape each step (eager backend).
-/
def scalarTrainer {α : Type} [Context α] [Internal.CudaBridge.TensorConv α] [DecidableEq Shape]
    {paramShapes inputShapes : List Shape} (opts : Options := {})
    (initRequiresGrad : List Bool := List.replicate paramShapes.length true)
    (loss :
      ∀ {m : Type → Type}, [Monad m] → [Ops (m := m) (α := α)] →
        CurriedRef (fun s => Ops.Ref (m := m) (α := α) s) (paramShapes ++ inputShapes)
          (m (Ops.Ref (m := m) (α := α) Shape.scalar))) :
    Curried.Fn α paramShapes (IO (ScalarTrainer α paramShapes inputShapes)) :=
    Curried.curry (α := α) (ss := paramShapes) (β := IO (ScalarTrainer α paramShapes inputShapes))
    (fun initParams => do
    let ps ← ParamList.ofTListWithRequiresGrad (α := α) initParams initRequiresGrad
    match opts.backend with
    | .compiled =>
        if opts.device != .cpu then
          throw <| IO.userError
            s!"torch compiled backend currently supports device `cpu`; requested `{opts.deviceName}`"
        let Γ : List Shape := paramShapes ++ inputShapes
        let build : Runtime.Autograd.Compiled.GraphM.M α Γ (Runtime.Autograd.Compiled.GraphM.Var
          Shape.scalar) := do
          let vs ← Runtime.Autograd.Compiled.GraphM.args (α := α) (Γ := Γ)
          CurriedRef.applyVarList (Γ := Γ) (β := Runtime.Autograd.Compiled.GraphM.M α Γ
            (Runtime.Autograd.Compiled.GraphM.Var Shape.scalar))
            (loss (m := Runtime.Autograd.Compiled.GraphM.M α Γ)) vs
        let compiled ← okOrThrow (compileScalar (α := α) (Γ := Γ) build)
        let ssFull : List Shape := compiled.ssPrev ++ [Shape.scalar]
        let fullGraph : Proofs.Autograd.Algebra.GraphData α Unit Γ ssFull :=
          .snoc (ss := compiled.ssPrev) (τ := Shape.scalar) compiled.gPrev compiled.node
        let outId : Nat := Runtime.Autograd.Compiled.outId (Γ := Γ) (ss := ssFull)

        let getScalarFromTape (t : Runtime.Autograd.Tape α) : IO (Tensor α Shape.scalar) := do
          let any ← match t.getValue? outId with
            | some v => pure v
            | none => throw <| IO.userError "torch.compile: missing output value in compiled tape"
          if h : any.s = Shape.scalar then
            pure (Tensor.castShape any.t h)
          else
            throw <| IO.userError
              s!"torch.compile: output shape mismatch (expected scalar, got {Shape.pretty any.s})"

        let rec gradsPrefix :
            {ss : List Shape} → Array (Runtime.AnyTensor α) → Nat → IO (TList α ss)
          | [], _grads, _off => pure .nil
          | s :: ss, grads, off => do
              let any ← match grads[off]? with
                | some v => pure v
                | none => throw <| IO.userError "torch.compile: gradient array too small"
              if h : any.s = s then
                let g : Tensor α s := Tensor.castShape any.t h
                let gs ← gradsPrefix (ss := ss) grads (off + 1)
                pure (.cons g gs)
              else
                throw <| IO.userError <|
                  s!"torch.compile: gradient shape mismatch at idx={off} (expected "
                    ++ s!"{Shape.pretty s}, got "
                    ++ s!"{Shape.pretty any.s})"

        let forward : Curried.Fn α inputShapes (IO (Tensor α Shape.scalar)) :=
          Curried.curry (α := α) (ss := inputShapes) (β := IO (Tensor α Shape.scalar)) (fun xs => do
            let pv ← ParamList.values (α := α) ps
            let args := Proofs.Autograd.Algebra.TList.append (α := α) (ss₁ := paramShapes) (ss₂ :=
              inputShapes) pv xs
            let (tape, _ctx) := Runtime.Autograd.Compiled.compile (α := α) (Γ := Γ) (ss := ssFull)
              fullGraph args
            getScalarFromTape tape)
        let backward : Curried.Fn α inputShapes (IO (TList α paramShapes)) :=
          Curried.curry (α := α) (ss := inputShapes) (β := IO (TList α paramShapes)) (fun xs => do
            let pv ← ParamList.values (α := α) ps
            let args := Proofs.Autograd.Algebra.TList.append (α := α) (ss₁ := paramShapes) (ss₂ :=
              inputShapes) pv xs
            let (tape, _ctx) := Runtime.Autograd.Compiled.compile (α := α) (Γ := Γ) (ss := ssFull)
              fullGraph args
            let grads ← okOrThrow (Runtime.Autograd.Compiled.backwardDenseAllFromOutput (α := α) (Γ
              := Γ) (ss := ssFull) tape)
            gradsPrefix (ss := paramShapes) grads 0)
        let step (lr : α) : Curried.Fn α inputShapes (IO Unit) :=
          Curried.curry (α := α) (ss := inputShapes) (β := IO Unit) (fun xs => do
            let g ← Curried.uncurry (α := α) (ss := inputShapes) (β := IO (TList α paramShapes))
              backward xs
            ParamList.sgdStep (α := α) (ss := paramShapes) ps lr g)
        pure
          { params := ps
            forward := forward
            backward := backward
            step := step
            adamStep? := none
            adamWStep? := none
            getParams := ParamList.values (α := α) (ss := paramShapes) ps }
    | .eager =>
        let sess ← Internal.EagerSession.new (α := α) opts
        let adamStateRef ← IO.mkRef (Std.HashMap.emptyWithCapacity : Internal.EagerSession.CudaAdamState)
        let lossEager := loss (m := Internal.EagerM α)
        let forward : Curried.Fn α inputShapes (IO (Tensor α Shape.scalar)) :=
          Curried.curry (α := α) (ss := inputShapes) (β := IO (Tensor α Shape.scalar)) (fun xs => do
            sess.resetTape
            let lossRef ← (do
              let pRefs ← Internal.useParams (α := α) (ss := paramShapes) ps
              let xRefs ← Internal.useInputs (α := α) (ss := inputShapes) xs
              let allRefs := RefList.append (ss₁ := paramShapes) (ss₂ := inputShapes) pRefs xRefs
              CurriedRef.uncurry (ss := paramShapes ++ inputShapes) (lossEager) allRefs) |>.run sess
            Internal.EagerSession.getValue (α := α) sess (sh := Shape.scalar) lossRef)
        let backward : Curried.Fn α inputShapes (IO (TList α paramShapes)) :=
          Curried.curry (α := α) (ss := inputShapes) (β := IO (TList α paramShapes)) (fun xs => do
            sess.resetTape
            let (lossRef, pRefs) ← (do
              let pRefs ← Internal.useParams (α := α) (ss := paramShapes) ps
              let xRefs ← Internal.useInputs (α := α) (ss := inputShapes) xs
              let allRefs := RefList.append (ss₁ := paramShapes) (ss₂ := inputShapes) pRefs xRefs
              let lossRef ← CurriedRef.uncurry (ss := paramShapes ++ inputShapes) (lossEager) allRefs
              pure (lossRef, pRefs)) |>.run sess
            let grads ← Internal.EagerSession.backwardScalarDenseAll (α := α) sess lossRef
            Internal.gradsOfRefs (α := α) (ss := paramShapes) grads pRefs)
        let step (lr : α) : Curried.Fn α inputShapes (IO Unit) :=
          Curried.curry (α := α) (ss := inputShapes) (β := IO Unit) (fun xs => do
            if opts.usesCuda then
              -- One eager training step in a tape-scoped device-memory bracket: `withStep` opens with
              -- `resetTape`, and on exit releases every tape-owned temporary and retires the snapshot,
              -- so the device working set stays flat across steps. See `EagerSession.withStep`.
              sess.withStep do
                let lossRef ← (do
                  let pRefs ← Internal.useParams (α := α) (ss := paramShapes) ps
                  let xRefs ← Internal.useInputs (α := α) (ss := inputShapes) xs
                  let allRefs := RefList.append (ss₁ := paramShapes) (ss₂ := inputShapes) pRefs xRefs
                  CurriedRef.uncurry (ss := paramShapes ++ inputShapes) (lossEager) allRefs) |>.run sess
                let gradsDev ← Internal.EagerSession.backwardScalarParamGradsCuda (α := α) sess lossRef
                Internal.EagerSession.sgdStepAllCudaMap (α := α) sess lr gradsDev
                Internal.EagerSession.releaseCudaGradMap gradsDev
            else
              let g ← Curried.uncurry (α := α) (ss := inputShapes) (β := IO (TList α paramShapes))
                backward xs
              ParamList.sgdStep (α := α) (ss := paramShapes) ps lr g)
        let adamStep? : Option (α → α → α → α → Curried.Fn α inputShapes (IO Unit)) :=
          if opts.usesCuda then
            some (fun lr beta1 beta2 epsilon =>
              Curried.curry (α := α) (ss := inputShapes) (β := IO Unit) (fun xs => do
                sess.withStep do
                  let lossRef ← (do
                    let pRefs ← Internal.useParams (α := α) (ss := paramShapes) ps
                    let xRefs ← Internal.useInputs (α := α) (ss := inputShapes) xs
                    let allRefs := RefList.append (ss₁ := paramShapes) (ss₂ := inputShapes) pRefs xRefs
                    CurriedRef.uncurry (ss := paramShapes ++ inputShapes) (lossEager) allRefs) |>.run sess
                  let gradsDev ← Internal.EagerSession.backwardScalarParamGradsCuda (α := α) sess lossRef
                  Internal.EagerSession.adamStepAllCudaMap (α := α) sess adamStateRef lr beta1 beta2
                    epsilon gradsDev
                  Internal.EagerSession.releaseCudaGradMap gradsDev))
          else
            none
        let adamWStep? : Option (α → α → α → α → α → Curried.Fn α inputShapes (IO Unit)) :=
          if opts.usesCuda then
            some (fun lr weightDecay beta1 beta2 epsilon =>
              Curried.curry (α := α) (ss := inputShapes) (β := IO Unit) (fun xs => do
                sess.withStep do
                  let lossRef ← (do
                  let pRefs ← Internal.useParams (α := α) (ss := paramShapes) ps
                  let xRefs ← Internal.useInputs (α := α) (ss := inputShapes) xs
                  let allRefs := RefList.append (ss₁ := paramShapes) (ss₂ := inputShapes) pRefs xRefs
                  CurriedRef.uncurry (ss := paramShapes ++ inputShapes) (lossEager) allRefs) |>.run sess
                  let gradsDev ← Internal.EagerSession.backwardScalarParamGradsCuda (α := α) sess lossRef
                  Internal.EagerSession.adamWStepAllCudaMap (α := α) sess adamStateRef lr weightDecay
                    beta1 beta2 epsilon gradsDev
                  Internal.EagerSession.releaseCudaGradMap gradsDev))
          else
            none
        pure
          { params := ps
            forward := forward
            backward := backward
            step := step
            adamStep? := adamStep?
            adamWStep? := adamWStep?
            getParams := ParamList.valuesSynced (α := α) (ss := paramShapes) ps })
end Torch
end Autograd
end Runtime

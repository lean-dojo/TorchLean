/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Functional.Ops
public import NN.Runtime.Autograd.TypedGraph.GraphM.Convolution
public import NN.Runtime.Autograd.TypedGraph.GraphM.Neural
public import NN.Runtime.Autograd.TypedGraph.GraphM.Pooling

/-!
# Graph Operations

Interpret `Ops` programs as typed graph nodes. This instance is shared by graph evaluation
and training; it does not allocate trainer state.
-/

@[expose] public section

namespace Runtime.Autograd.Torch

open Spec TorchLean
open TorchLean.Tensor
open Proofs.Autograd.Algebra

/--
`Ops` instance for the typed graph builder monad `GraphM`.

This interprets `Ops` primitives by recording typed SSA nodes rather than executing them
immediately. `Runtime.Autograd.TypedGraph.GraphM` builds the graph data;
`Runtime.Autograd.Torch.TypedGraph` packages it for repeated execution.
-/
instance {α Δ : Type} [TorchLean.Storage α] [Context α] {Γ : List Shape} :
    Ops (Runtime.Autograd.TypedGraph.GraphM.MWith α Δ Γ) α where
  Ref := fun s => Runtime.Autograd.TypedGraph.GraphM.Var s
  DataRef := fun β _ s => Δ → Tensor β s
  dataConst := fun x _ => x
  mapData := fun f x d => f (x d)
  const := fun {s} t => Runtime.Autograd.TypedGraph.GraphM.const (α := α) (Γ := Γ) (s := s) t
  add := fun {s} a b => Runtime.Autograd.TypedGraph.GraphM.add (α := α) (Γ := Γ) (s := s) a b
  sub := fun {s} a b => Runtime.Autograd.TypedGraph.GraphM.sub (α := α) (Γ := Γ) (s := s) a b
  mul := fun {s} a b => Runtime.Autograd.TypedGraph.GraphM.mul (α := α) (Γ := Γ) (s := s) a b
  scale := fun {s} x c => Runtime.Autograd.TypedGraph.GraphM.scale (α := α) (Γ := Γ) (s := s) x c
  abs := fun {s} x => Runtime.Autograd.TypedGraph.GraphM.abs (α := α) (Γ := Γ) (s := s) x
  sqrt := fun {s} x => Runtime.Autograd.TypedGraph.GraphM.sqrt (α := α) (Γ := Γ) (s := s) x
  clamp := fun {s} x minVal maxVal =>
    Runtime.Autograd.TypedGraph.GraphM.clamp (α := α) (Γ := Γ) (s := s) x minVal maxVal
  max := fun {s} a b => Runtime.Autograd.TypedGraph.GraphM.max (α := α) (Γ := Γ) (s := s) a b
  min := fun {s} a b => Runtime.Autograd.TypedGraph.GraphM.min (α := α) (Γ := Γ) (s := s) a b
  broadcastTo := fun {s₁ s₂} cb x =>
    Runtime.Autograd.TypedGraph.GraphM.broadcastTo (α := α) (Γ := Γ) (s₁ := s₁) (s₂ := s₂) cb x
  reshape := fun {s₁ s₂} x h =>
    Runtime.Autograd.TypedGraph.GraphM.reshape (α := α) (Γ := Γ) (s₁ := s₁) (s₂ := s₂) x h
  swapAdjacentAtDepth := fun {s} depth x =>
    Runtime.Autograd.TypedGraph.GraphM.swapAdjacentAtDepth (α := α) (Γ := Γ) (s := s) depth x
  reduceSum := fun {s} axis => fun x =>
    Runtime.Autograd.TypedGraph.GraphM.reduceSum (α := α) (Γ := Γ) (s := s) axis x
  reduceMean := fun {s} axis => fun x =>
    Runtime.Autograd.TypedGraph.GraphM.reduceMean (α := α) (Γ := Γ) (s := s) axis x
  select := fun {s} axis _axisInBounds x index =>
    Runtime.Autograd.TypedGraph.GraphM.select (α := α) (Γ := Γ) (s := s) axis x index
  indexSelect := fun {s} axis count _axisInBounds x indices =>
    Runtime.Autograd.TypedGraph.GraphM.indexSelect
      (α := α) (Γ := Γ) (s := s) axis count x indices
  scatterAdd := fun {s} axis count _axisInBounds base source indices =>
    Runtime.Autograd.TypedGraph.GraphM.scatterAdd
      (α := α) (Γ := Γ) (s := s) axis count base source indices
  matmul := fun {batchA batchB batch : Shape} {mDim nDim pDim : Nat}
      {broadcastA} {broadcastB} a b =>
    Runtime.Autograd.TypedGraph.GraphM.matmul (α := α) (Γ := Γ)
      (batchA := batchA) (batchB := batchB) (batch := batch)
      (m := mDim) (n := nDim) (p := pDim)
      (broadcastA := broadcastA) (broadcastB := broadcastB) a b
  concatLeadingAxis := fun {nDim mDim} {s} a b =>
    Runtime.Autograd.TypedGraph.GraphM.concatLeadingAxis (α := α) (Γ := Γ) (n := nDim) (m := mDim)
      (s := s) a b
  sliceLeadingAxisRange := fun {nDim} {s} start len h x =>
    Runtime.Autograd.TypedGraph.GraphM.sliceLeadingAxisRange (α := α) (Γ := Γ) (n := nDim) (s := s)
      x start len h
  maxPool := fun {d C} {inSpatial kernel stride padding} x =>
    Runtime.Autograd.TypedGraph.GraphM.maxPool (α := α) (Γ := Γ)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      x
  avgPool := fun {d C} {inSpatial kernel stride padding} x =>
    Runtime.Autograd.TypedGraph.GraphM.avgPool (α := α) (Γ := Γ)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      x
  smoothMaxPool := fun {d C} {inSpatial kernel stride padding}
      [_decidableEq : DecidableEq α] x beta =>
    Runtime.Autograd.TypedGraph.GraphM.smoothMaxPool (α := α) (Γ := Γ)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      x beta
  relu := fun {s} x => Runtime.Autograd.TypedGraph.GraphM.relu (α := α) (Γ := Γ) (s := s) x
  sigmoid := fun {s} x => Runtime.Autograd.TypedGraph.GraphM.sigmoid (α := α) (Γ := Γ) (s := s) x
  tanh := fun {s} x => Runtime.Autograd.TypedGraph.GraphM.tanh (α := α) (Γ := Γ) (s := s) x
  gelu := fun {s} x => Runtime.Autograd.TypedGraph.GraphM.gelu (α := α) (Γ := Γ) (s := s) x
  softmaxLast := fun {s} x =>
    Runtime.Autograd.TypedGraph.GraphM.softmaxLast (α := α) (Γ := Γ) (s := s) x
  logSoftmaxLast := fun {s} x =>
    Runtime.Autograd.TypedGraph.GraphM.logSoftmaxLast (α := α) (Γ := Γ) (s := s) x
  softplus := fun {s} x => Runtime.Autograd.TypedGraph.GraphM.softplus (α := α) (Γ := Γ) (s := s) x
  exp := fun {s} x => Runtime.Autograd.TypedGraph.GraphM.exp (α := α) (Γ := Γ) (s := s) x
  sin := fun {s} x => Runtime.Autograd.TypedGraph.GraphM.sin (α := α) (Γ := Γ) (s := s) x
  cos := fun {s} x => Runtime.Autograd.TypedGraph.GraphM.cos (α := α) (Γ := Γ) (s := s) x
  log := fun {s} x => Runtime.Autograd.TypedGraph.GraphM.log (α := α) (Γ := Γ) (s := s) x
  inv := fun {s} x => Runtime.Autograd.TypedGraph.GraphM.inv (α := α) (Γ := Γ) (s := s) x
  detach := fun {s} x => Runtime.Autograd.TypedGraph.GraphM.detach (α := α) (Γ := Γ) (s := s) x
  safeLog := fun {s} x ε => Runtime.Autograd.TypedGraph.GraphM.safeLog (α := α) (Γ := Γ) (s := s) x
    (ε := ε)
  sum := fun {s} x => Runtime.Autograd.TypedGraph.GraphM.sum (α := α) (Γ := Γ) (s := s) x
  flatten := fun {s} x => Runtime.Autograd.TypedGraph.GraphM.flatten (α := α) (Γ := Γ) (s := s) x
  linear := fun {inDim outDim} w b x =>
    Runtime.Autograd.TypedGraph.GraphM.linear (α := α) (Γ := Γ) (inDim := inDim) (outDim := outDim)
      w b x
  mseLoss := fun {s} yhat target =>
    Runtime.Autograd.TypedGraph.GraphM.mseLoss (α := α) (Γ := Γ) (s := s) yhat target
  layerNorm := fun {seqLen embedDim} hSeq hEmb x gamma beta epsilon =>
    Runtime.Autograd.TypedGraph.GraphM.layerNorm (α := α) (Γ := Γ) (seqLen := seqLen) (embedDim :=
      embedDim)
      (h_seq_pos := hSeq) (h_embed_pos := hEmb) x gamma beta (epsilon := epsilon)
  batchNorm := fun {channels sSpatial} hWellFormed x gamma beta epsilon =>
    Runtime.Autograd.TypedGraph.GraphM.batchNorm (α := α) (Γ := Γ)
      (channels := channels) (sSpatial := sSpatial) hWellFormed x gamma beta (epsilon := epsilon)
  multiHeadAttention := fun {n numHeads dModel headDim} h1 wq wk wv wo x mask =>
    Runtime.Autograd.TypedGraph.GraphM.multiHeadAttention (α := α) (Γ := Γ) (n := n) (numHeads :=
      numHeads)
      (dModel := dModel) (headDim := headDim) h1 wq wk wv wo x (mask := mask)
  batchedMultiHeadAttention :=
    fun {batch n numHeads dModel headDim} _hBatch h1 wq wk wv wo x mask =>
      Runtime.Autograd.TypedGraph.GraphM.batchedMultiHeadAttention (α := α) (Γ := Γ)
        (batch := batch) (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
        h1 wq wk wv wo x (mask := mask)
  conv := fun {d inC outC} {kernel stride padding} {inSpatial} w b x =>
    Runtime.Autograd.TypedGraph.GraphM.conv (α := α) (Γ := Γ)
      (d := d) (inC := inC) (outC := outC)
      (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
      w b x
  convTranspose := fun {d inC outC} {kernel stride padding} {inSpatial} w b x =>
    Runtime.Autograd.TypedGraph.GraphM.convTranspose (α := α) (Γ := Γ)
      (d := d) (inC := inC) (outC := outC)
      (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
      w b x
  randUniform := fun {s} seed => do
    Runtime.Autograd.TypedGraph.GraphM.randUniform (α := α) (Γ := Γ) (s := s) (seed := seed)
  bernoulliMask := fun {s} keepProb seed => do
    Runtime.Autograd.TypedGraph.GraphM.bernoulliMask (α := α) (Γ := Γ) (s := s) keepProb (seed :=
      seed)

end Runtime.Autograd.Torch

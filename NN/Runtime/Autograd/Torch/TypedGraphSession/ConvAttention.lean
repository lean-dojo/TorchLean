/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.TypedGraphSession.GraphOps
public import NN.Runtime.Autograd.TypedGraph.GraphM.Convolution
public import NN.Runtime.Autograd.TypedGraph.GraphM.Neural

/-!
# Typed Graph Session: Convolution and Attention Operations
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Internal

namespace TypedGraphSession

/--
N-D convolution for channels-first tensors `(inC, spatial...)` (no batch axis).

Kernel layout is `(outC, inC, kernelSpatial...)`, bias is `(outC)`.

PyTorch comparison: `torch.nn.functional.conv{d}d` specialized to a single sample.
-/
def conv {α : Type} [TorchLean.Storage α] (s : TypedGraphSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {d inC outC : Nat}
  {kernel stride padding : TorchLean.Tensor Nat [d]}
  {inSpatial : TorchLean.Tensor Nat [d]}
  (w : TensorRef α (Shape.ofList (outC :: inC :: Tensor.to kernel (List Nat))))
  (b : TensorRef α [outC])
  (x : TensorRef α (Shape.ofList (inC :: Tensor.to inSpatial (List Nat)))) :
  IO (TensorRef α
    (Shape.ofList (outC ::
      Tensor.to (Spec.convOutSpatial inSpatial kernel stride padding) (List Nat)))) :=
  commitGraphM (α := α) s
    (β := TensorRef α
      (Shape.ofList (outC ::
        Tensor.to (Spec.convOutSpatial inSpatial kernel stride padding) (List Nat))))
    (refs := #[w.identity?, b.identity?, x.identity?])
    (fun {Γ} {ss} xv nat g => do
      let (v, st') ← runGraphM (α := α) (Γ := Γ)
        (Runtime.Autograd.TypedGraph.GraphM.conv (α := α) (Γ := Γ)
          (d := d) (inC := inC) (outC := outC)
          (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
          { id := w.id } { id := b.id } { id := x.id })
        ss g
      let ⟨ss', g'⟩ := st'
      let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
      pure ({ id := v.id }, st1))

/--
N-D transpose convolution for channels-first tensors `(inC, spatial...)` (no batch axis).

Kernel layout is `(inC, outC, kernelSpatial...)` (PyTorch convention), bias is `(outC)`.

PyTorch comparison: `torch.nn.functional.conv_transpose{d}d` specialized to a single sample.
-/
def convTranspose {α : Type} [TorchLean.Storage α]
    (s : TypedGraphSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {d inC outC : Nat}
  {kernel stride padding : TorchLean.Tensor Nat [d]}
  {inSpatial : TorchLean.Tensor Nat [d]}
  (w : TensorRef α (Shape.ofList (inC :: outC :: Tensor.to kernel (List Nat))))
  (b : TensorRef α [outC])
  (x : TensorRef α (Shape.ofList (inC :: Tensor.to inSpatial (List Nat)))) :
  IO (TensorRef α
    (Shape.ofList (outC ::
      Tensor.to (Spec.convTransposeOutSpatial inSpatial kernel stride padding) (List Nat)))) :=
  commitGraphM (α := α) s
    (β := TensorRef α
      (Shape.ofList (outC ::
        Tensor.to (Spec.convTransposeOutSpatial inSpatial kernel stride padding) (List Nat))))
    (refs := #[w.identity?, b.identity?, x.identity?])
    (fun {Γ} {ss} xv nat g => do
      let (v, st') ← runGraphM (α := α) (Γ := Γ)
        (Runtime.Autograd.TypedGraph.GraphM.convTranspose (α := α) (Γ := Γ)
          (d := d) (inC := inC) (outC := outC)
          (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
          { id := w.id } { id := b.id } { id := x.id })
        ss g
      let ⟨ss', g'⟩ := st'
      let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
      pure ({ id := v.id }, st1))

/--
Multi-head self-attention.

This is a shape-specialized attention primitive used by transformer-style examples:
- input `x` has shape `(n, dModel)`
- `wq`, `wk`, `wv` map `dModel → numHeads*headDim`
- `wo` maps `numHeads*headDim → dModel`
- optional `mask` is a boolean `(n,n)` attention mask

PyTorch comparison: similar to `torch.nn.MultiheadAttention` / scaled dot-product attention, but
encoded in a fully typed graph for lowering and later semantic analysis.
-/
def multiHeadAttention {α : Type} [TorchLean.Storage α]
    (s : TypedGraphSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (wq : TensorRef α [dModel, numHeads * headDim])
  (wk : TensorRef α [dModel, numHeads * headDim])
  (wv : TensorRef α [dModel, numHeads * headDim])
  (wo : TensorRef α [numHeads * headDim, dModel])
  (x : TensorRef α [n, dModel])
  (mask : Option (Tensor Bool [n, n]) := none) :
  IO (TensorRef α [n, dModel]) :=
  commitGraphM (α := α) s (β := TensorRef α [n, dModel])
      (refs := #[wq.identity?, wk.identity?, wv.identity?, wo.identity?, x.identity?])
      (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.multiHeadAttention (α := α) (Γ := Γ)
        (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim) (h1 := h1)
        { id := wq.id } { id := wk.id } { id := wv.id } { id := wo.id } { id := x.id } (mask :=
          mask))
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))
end TypedGraphSession

end Internal

end Torch
end Autograd
end Runtime

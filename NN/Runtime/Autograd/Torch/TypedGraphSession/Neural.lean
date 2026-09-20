/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.TypedGraphSession.GraphOps
public import NN.Runtime.Autograd.TypedGraph.GraphM.Neural

/-!
# Typed Graph Session: Neural-Network Operations
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
Record elementwise logistic sigmoid.

PyTorch comparison: `torch.sigmoid(x)`.
-/
def sigmoid {α : Type} [TorchLean.Storage α]
    (s : TypedGraphSession α) [Context α] {sh : Shape}
  (x : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (refs := #[x.identity?])
      (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.sigmoid (α := α) (Γ := Γ) (s := sh) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record elementwise hyperbolic tangent.

PyTorch comparison: `torch.tanh(x)`.
-/
def tanh {α : Type} [TorchLean.Storage α]
    (s : TypedGraphSession α) [Context α] {sh : Shape}
  (x : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (refs := #[x.identity?])
      (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.tanh (α := α) (Γ := Γ) (s := sh) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/-- Record softmax along an explicitly selected tensor dimension. -/
def softmax {α : Type} [TorchLean.Storage α]
    (s : TypedGraphSession α) [Context α]
    {sh : Shape} (axis : Nat) [Shape.AxisInBounds axis sh]
    (x : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (refs := #[x.identity?])
      (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.softmax
        (α := α) (Γ := Γ) (s := sh) axis { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/-- Record stable log-softmax along an explicitly selected tensor dimension. -/
def logSoftmax {α : Type} [TorchLean.Storage α]
    (s : TypedGraphSession α) [Context α]
    {sh : Shape} (axis : Nat) [Shape.AxisInBounds axis sh]
    (x : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (refs := #[x.identity?])
      (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.logSoftmax
        (α := α) (Γ := Γ) (s := sh) axis { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record elementwise softplus.

PyTorch comparison: `torch.nn.functional.softplus(x)`.
-/
def softplus {α : Type} [TorchLean.Storage α]
    (s : TypedGraphSession α) [Context α] {sh : Shape}
  (x : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (refs := #[x.identity?])
      (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.softplus (α := α) (Γ := Γ) (s := sh) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record elementwise exponential.

PyTorch comparison: `torch.exp(x)`.
-/
def exp {α : Type} [TorchLean.Storage α]
    (s : TypedGraphSession α) [Context α] {sh : Shape}
  (x : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (refs := #[x.identity?])
      (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.exp (α := α) (Γ := Γ) (s := sh) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/-- Record sine of angles in radians, retaining its JVP and VJP in the typed graph. -/
def sin {α : Type} [TorchLean.Storage α]
    (s : TypedGraphSession α) [Context α] {sh : Shape}
    (x : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (refs := #[x.identity?])
      (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.sin (α := α) (Γ := Γ) (s := sh) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/-- Record cosine of angles in radians, with derivative `-sin(x)` at the original input. -/
def cos {α : Type} [TorchLean.Storage α]
    (s : TypedGraphSession α) [Context α] {sh : Shape}
    (x : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (refs := #[x.identity?])
      (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.cos (α := α) (Γ := Γ) (s := sh) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record elementwise natural logarithm.

PyTorch comparison: `torch.log(x)`.
-/
def log {α : Type} [TorchLean.Storage α]
    (s : TypedGraphSession α) [Context α] {sh : Shape}
  (x : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (refs := #[x.identity?])
      (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.log (α := α) (Γ := Γ) (s := sh) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record elementwise log with epsilon guard.

This is intended for numerically stable losses; it corresponds approximately to `log(max(x, ε))`.
PyTorch comparison: `torch.log(torch.clamp(x, min=ε))`.
-/
def safeLog {α : Type} [TorchLean.Storage α]
    (s : TypedGraphSession α) [Context α] {sh : Shape}
  (x : TensorRef α sh) (ε : α := Context.defaultEpsilon) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (refs := #[x.identity?])
      (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.safeLog (α := α) (Γ := Γ) (s := sh) { id := x.id } (ε :=
        ε))
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Sum-reduce all elements to a scalar.

PyTorch comparison: `x.sum()`.
-/
def sum {α : Type} [TorchLean.Storage α]
    (s : TypedGraphSession α) [Context α] {sh : Shape}
  (x : TensorRef α sh) : IO (TensorRef α Shape.scalar) :=
  commitGraphM (α := α) s (β := TensorRef α Shape.scalar) (refs := #[x.identity?])
      (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.sum (α := α) (Γ := Γ) (s := sh) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record a fully-connected linear layer: `y = w • x + b`.

Type-level shapes enforce `w : (outDim, inDim)`, `b : (outDim,)`, and `x : (inDim,)`.
PyTorch comparison: `torch.nn.functional.linear(x, weight=w, bias=b)` (with the same weight layout).
-/
def linear {α : Type} [TorchLean.Storage α]
    (s : TypedGraphSession α) [Add α] [Mul α] [Zero α]
  {inDim outDim : Nat}
  (w : TensorRef α [outDim, inDim])
  (b : TensorRef α [outDim])
  (x : TensorRef α [inDim]) : IO (TensorRef α [outDim]) :=
  commitGraphM (α := α) s (β := TensorRef α [outDim])
      (refs := #[w.identity?, b.identity?, x.identity?]) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.linear (α := α) (Γ := Γ)
        (inDim := inDim) (outDim := outDim) { id := w.id } { id := b.id } { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Mean-squared-error loss returning a scalar.

PyTorch comparison: `torch.nn.functional.mse_loss(yhat, target, reduction="mean")`.
-/
def mseLoss {α : Type} [TorchLean.Storage α] (s : TypedGraphSession α)
  [Add α] [Sub α] [Mul α] [Div α] [Zero α] [One α] [NatCast α]
  {sh : Shape} (yhat target : TensorRef α sh) : IO (TensorRef α Shape.scalar) :=
  commitGraphM (α := α) s (β := TensorRef α Shape.scalar)
      (refs := #[yhat.identity?, target.identity?]) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.mseLoss (α := α) (Γ := Γ) (s := sh) { id := yhat.id } { id
        := target.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Layer normalization over the trailing embedding dimension.

This variant is specialized to 2D tensors of shape `(seqLen, embedDim)` and expects positive
dimensions for numerical stability and well-formedness.
PyTorch comparison: `torch.nn.LayerNorm(embedDim)` (applied per token), or
`torch.nn.functional.layer_norm`.
-/
def layerNorm {α : Type} [TorchLean.Storage α] (s : TypedGraphSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
  (x : TensorRef α [seqLen, embedDim])
  (gamma : TensorRef α [embedDim])
  (beta : TensorRef α [embedDim])
  (epsilon : α := TorchLean.normalizationEpsilon) : IO (TensorRef α [seqLen, embedDim]) :=
  commitGraphM (α := α) s (β := TensorRef α [seqLen, embedDim])
      (refs := #[x.identity?, gamma.identity?, beta.identity?]) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.layerNorm (α := α) (Γ := Γ)
        (seqLen := seqLen) (embedDim := embedDim) (h_seq_pos := h_seq_pos) (h_embed_pos :=
          h_embed_pos)
        { id := x.id } { id := gamma.id } { id := beta.id } (epsilon := epsilon))
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/-- Batch normalization over every spatial axis of a channel-first tensor. -/
def batchNorm {α : Type} [TorchLean.Storage α] (s : TypedGraphSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {channels : Nat} {sSpatial : Shape}
  (hWellFormed : (Shape.dim channels sSpatial).wellFormed)
  (x : TensorRef α (.dim channels sSpatial))
  (gamma : TensorRef α [channels])
  (beta : TensorRef α [channels])
  (epsilon : α := TorchLean.normalizationEpsilon) :
  IO (TensorRef α (.dim channels sSpatial)) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim channels sSpatial))
    (refs := #[x.identity?, gamma.identity?, beta.identity?])
    (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.batchNorm (α := α) (Γ := Γ)
        (channels := channels) (sSpatial := sSpatial) hWellFormed
        { id := x.id } { id := gamma.id } { id := beta.id } (epsilon := epsilon))
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))
end TypedGraphSession

end Internal

end Torch
end Autograd
end Runtime

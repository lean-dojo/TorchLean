/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.LeadingAxis
public import NN.Runtime.Autograd.TypedGraph.GraphM.Elementwise
public import NN.Runtime.Autograd.TypedGraph.GraphM.ShapeIndex
public import NN.Spec.Layers.Attention
public import NN.Spec.Layers.Normalization.BatchNorm

/-!
# GraphM Neural Layers

Normalization and attention builders for typed graphs.

Normalization records the supplied `epsilon` without validating it or substituting
`Context.defaultEpsilon`. The default `TorchLean.normalizationEpsilon` can round to zero in tiny
formats. Pass an explicit representable positive, finite `epsilon`; a zero value can produce NaNs
when normalizing a constant input.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace TypedGraph
namespace GraphM

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Proofs.Autograd.Algebra
-- Typed context indices come from `NN.Proofs.Autograd.Tape.Util.Idx`, the one place
-- `Idx` and `getIdx` are defined.
open Proofs (Idx getIdx)

/--
Layer normalization (sequence-first), producing the same shape as the input.

PyTorch comparison: `torch.nn.LayerNorm` / `torch.nn.functional.layer_norm` (modulo exact layout).

Forward-mode status: implemented by `Spec.layerNormJvp`, including parameter tangents for
`gamma` and `beta`.
-/
def layerNorm {α : Type} {Δ : Type} [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {Γ : List Shape} {seqLen embedDim : Nat}
  (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
  (x : Var (.dim seqLen (.dim embedDim .scalar)))
  (gamma : Var (.dim embedDim .scalar))
  (beta : Var (.dim embedDim .scalar))
  (epsilon : α := TorchLean.normalizationEpsilon) :
  MWith α Δ Γ (Var (.dim seqLen (.dim embedDim .scalar))) := do
  let ⟨ss, g, _⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let ig ← liftM (mkIdx (_α := α) (Γ := Γ) ss gamma)
  let ib ← liftM (mkIdx (_α := α) (Γ := Γ) ss beta)
  let node : NodeData α Δ (Γ ++ ss) (.dim seqLen (.dim embedDim .scalar)) :=
    NodeData.ofLocalCompact (fun lookup => (lookup.read ix, lookup.read ig, lookup.read ib))
      (forward := fun ctx _d =>
        Spec.layerNorm (α := α) (seqLen := seqLen) (embedDim := embedDim)
          (x := ctx.1)
          (gamma := ctx.2.1)
          (beta := ctx.2.2)
          (h_seq_pos := h_seq_pos) (h_embed_pos := h_embed_pos) (epsilon := epsilon))
      (jvp := fun ctx dctx _d =>
        let xv := ctx.1
        let gv := ctx.2.1
        let bv := ctx.2.2
        let dx := dctx.1
        let dg := dctx.2.1
        let db := dctx.2.2
        Spec.layerNormJvp (α := α) (seqLen := seqLen) (embedDim := embedDim)
          (h_seq_pos := h_seq_pos) (h_embed_pos := h_embed_pos)
          (x := xv) (tangent := dx) (gamma := gv) (dgamma := dg) (_beta := bv) (dbeta := db)
          (epsilon := epsilon))
      (vjp := fun ctx _d dLdy =>
        let xv := ctx.1
        let gv := ctx.2.1
        let gradients :=
          Spec.layerNormBackward (α := α) (seqLen := seqLen) (embedDim := embedDim)
            (sequenceLengthPositive := h_seq_pos)
            (embeddingWidthPositive := h_embed_pos)
            (input := xv) (scale := gv) (outputGradient := dLdy) (epsilon := epsilon)
        let z0 :=
          Contributions.add (α := α) (shapes := Γ ++ ss)
            (Contributions.single (α := α) (Γ := Γ ++ ss)
              (s := .dim seqLen (.dim embedDim .scalar)) ix gradients.inputGradient)
            (Contributions.single (α := α) (Γ := Γ ++ ss)
              (s := .dim embedDim .scalar) ig gradients.scaleGradient)
        Contributions.add (α := α) (shapes := Γ ++ ss) z0
          (Contributions.single (α := α) (Γ := Γ ++ ss)
            (s := .dim embedDim .scalar) ib gradients.biasGradient))
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := (.dim seqLen (.dim embedDim .scalar))) g node

/-- Batch normalization over every spatial axis of a channel-first tensor. -/
def batchNorm {α : Type} {Δ : Type} [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] {Γ : List Shape} {channels : Nat} {sSpatial : Shape}
  (hWellFormed : (Shape.dim channels sSpatial).wellFormed)
  (x : Var (.dim channels sSpatial))
  (gamma : Var (.dim channels .scalar))
  (beta : Var (.dim channels .scalar))
  (epsilon : α := TorchLean.normalizationEpsilon) :
  MWith α Δ Γ (Var (.dim channels sSpatial)) := do
  let _ : Shape.WellFormed (.dim channels sSpatial) := ⟨hWellFormed⟩
  let ⟨ss, g, _⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let ig ← liftM (mkIdx (_α := α) (Γ := Γ) ss gamma)
  let ib ← liftM (mkIdx (_α := α) (Γ := Γ) ss beta)
  let outS : Shape := .dim channels sSpatial
  let node : NodeData α Δ (Γ ++ ss) outS :=
    NodeData.ofLocalCompact (fun lookup => (lookup.read ix, lookup.read ig, lookup.read ib))
      (forward := fun ctx _d =>
        Spec.batchNorm (α := α) (channels := channels) (sSpatial := sSpatial)
          (x := ctx.1)
          (gamma := ctx.2.1)
          (beta := ctx.2.2) (epsilon := epsilon))
      (jvp := fun ctx dctx _d =>
        let xv := ctx.1
        let gv := ctx.2.1
        let bv := ctx.2.2
        let dx := dctx.1
        let dg := dctx.2.1
        let db := dctx.2.2
        Spec.batchNormJvp (α := α) (channels := channels) (sSpatial := sSpatial)
          (x := xv) (tangent := dx) (gamma := gv) (dgamma := dg) (_beta := bv) (dbeta := db)
          (epsilon := epsilon))
      (vjp := fun ctx _d dLdy =>
        let xv := ctx.1
        let gv := ctx.2.1
        let gradients :=
          Spec.batchNormBackward (α := α) (channels := channels) (sSpatial := sSpatial)
            (x := xv) (gamma := gv) (gradOutput := dLdy) (epsilon := epsilon)
        let z0 :=
          Contributions.add (α := α) (shapes := Γ ++ ss)
            (Contributions.single (α := α) (Γ := Γ ++ ss)
              (s := outS) ix gradients.inputGradient)
            (Contributions.single (α := α) (Γ := Γ ++ ss)
              (s := .dim channels .scalar) ig gradients.scaleGradient)
        Contributions.add (α := α) (shapes := Γ ++ ss) z0
          (Contributions.single (α := α) (Γ := Γ ++ ss)
            (s := .dim channels .scalar) ib gradients.biasGradient))
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
Multi-head attention primitive (shape-specialized).

PyTorch comparison: `torch.nn.MultiheadAttention` / scaled dot-product attention.

Forward-mode status: implemented by `Spec.multiHeadAttentionJvp`, including tangents for the
input and all four projection matrices.
-/
def multiHeadAttention {α : Type} {Δ : Type} [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {Γ : List Shape} {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (wq : Var (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wk : Var (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wv : Var (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wo : Var (.dim (numHeads * headDim) (.dim dModel .scalar)))
  (x : Var (.dim n (.dim dModel .scalar)))
  (mask : Option (Tensor Bool [n, n]) := none) :
  MWith α Δ Γ (Var (.dim n (.dim dModel .scalar))) := do
  let ⟨ss, g, _⟩ ← get
  let iwq ← liftM (mkIdx (_α := α) (Γ := Γ) ss wq)
  let iwk ← liftM (mkIdx (_α := α) (Γ := Γ) ss wk)
  let iwv ← liftM (mkIdx (_α := α) (Γ := Γ) ss wv)
  let iwo ← liftM (mkIdx (_α := α) (Γ := Γ) ss wo)
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) (.dim n (.dim dModel .scalar)) :=
    NodeData.ofLocalCompact (fun lookup =>
      (lookup.read iwq, lookup.read iwk, lookup.read iwv, lookup.read iwo, lookup.read ix))
      (forward := fun ctx _d =>
        let mha : Spec.MultiHeadAttention α numHeads dModel headDim :=
          { queryWeight := ctx.1
            keyWeight := ctx.2.1
            valueWeight := ctx.2.2.1
            outputWeight := ctx.2.2.2.1 }
        Spec.MultiHeadAttention.forward (α := α) (n := n) (h1 := h1)
          (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
          (mha := mha) (x := ctx.2.2.2.2) (mask := mask))
      (jvp := fun ctx dctx _d =>
        let mha : Spec.MultiHeadAttention α numHeads dModel headDim :=
          { queryWeight := ctx.1
            keyWeight := ctx.2.1
            valueWeight := ctx.2.2.1
            outputWeight := ctx.2.2.2.1 }
        let dmha : Spec.MultiHeadAttention α numHeads dModel headDim :=
          { queryWeight := dctx.1
            keyWeight := dctx.2.1
            valueWeight := dctx.2.2.1
            outputWeight := dctx.2.2.2.1 }
        Spec.multiHeadAttentionJvp (α := α) (h1 := h1)
          (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
          (mha := mha) (dmha := dmha)
          (x := ctx.2.2.2.2)
          (dx := dctx.2.2.2.2)
          (mask := mask))
      (vjp := fun ctx _d dLdy =>
        let mha : Spec.MultiHeadAttention α numHeads dModel headDim :=
          { queryWeight := ctx.1
            keyWeight := ctx.2.1
            valueWeight := ctx.2.2.1
            outputWeight := ctx.2.2.2.1 }
        let xv := ctx.2.2.2.2
        let gradients :=
          Spec.multiHeadAttentionBackward (α := α) (h1 := h1)
            (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
            (mha := mha) (x := xv) (mask := mask) (gradOutput := dLdy)
        let dWq := gradients.parameters.queryWeight
        let dWk := gradients.parameters.keyWeight
        let dWv := gradients.parameters.valueWeight
        let dWo := gradients.parameters.outputWeight
        let dx := gradients.input
        let z0 :=
          Contributions.add (α := α) (shapes := Γ ++ ss)
            (Contributions.single (α := α) (Γ := Γ ++ ss)
              (s := .dim dModel (.dim (numHeads * headDim) .scalar)) iwq dWq)
            (Contributions.single (α := α) (Γ := Γ ++ ss)
              (s := .dim dModel (.dim (numHeads * headDim) .scalar)) iwk dWk)
        let z1 :=
          Contributions.add (α := α) (shapes := Γ ++ ss) z0
            (Contributions.single (α := α) (Γ := Γ ++ ss)
              (s := .dim dModel (.dim (numHeads * headDim) .scalar)) iwv dWv)
        let z2 :=
          Contributions.add (α := α) (shapes := Γ ++ ss) z1
            (Contributions.single (α := α) (Γ := Γ ++ ss)
              (s := .dim (numHeads * headDim) (.dim dModel .scalar)) iwo dWo)
        Contributions.add (α := α) (shapes := Γ ++ ss) z2
          (Contributions.single (α := α) (Γ := Γ ++ ss) (s := .dim n (.dim dModel .scalar)) ix dx))
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := (.dim n (.dim dModel .scalar))) g node

/--
Leading-axis map of the proved multi-head-attention node.

The typed graph intentionally records the per-sample nodes. This keeps its forward, JVP, and
VJP definitions inherited directly from `multiHeadAttention`, while an eager device backend may
execute the same map as one batched contraction.
-/
def batchedMultiHeadAttention {α : Type} {Δ : Type} [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {Γ : List Shape} {batch n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (wq : Var (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wk : Var (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wv : Var (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wo : Var (.dim (numHeads * headDim) (.dim dModel .scalar)))
  (x : Var (.dim batch (.dim n (.dim dModel .scalar))))
  (mask : Option (Tensor Bool [n, n]) := none) :
  MWith α Δ Γ (Var (.dim batch (.dim n (.dim dModel .scalar)))) :=
  Runtime.Autograd.mapOuterAxisWith
    (const (α := α) (Δ := Δ) (Γ := Γ) <| Tensor.dim (fun i : Fin 0 => Fin.elim0 i))
    (fun x start len h =>
      sliceLeadingAxisRange (α := α) (Δ := Δ) (Γ := Γ) x start len h)
    (fun x h => reshape (α := α) (Δ := Δ) (Γ := Γ) x h)
    (fun x y => concatLeadingAxis (α := α) (Δ := Δ) (Γ := Γ) x y)
    (fun sample => multiHeadAttention (α := α) (Δ := Δ) (Γ := Γ)
      h1 wq wk wv wo sample mask)
    x

end GraphM
end TypedGraph
end Autograd
end Runtime

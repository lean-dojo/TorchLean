/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/


module

public import NN.Runtime.Autograd.Engine.Core.Base
public import NN.Spec.Core.TensorReductionShape.ConcatSlice
public import NN.Spec.Layers.Linear

/-!
Linear-algebra operations for the eager engine.

The definitions here cover matrix products, batched products, affine layers, and the corresponding
runtime graph nodes shared by CPU and CUDA-backed execution.
-/

@[expose] public section

namespace Runtime
namespace Autograd

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Tape

/--
Fully-connected linear layer `y = W x + b` (matvec).

Type-level shapes enforce `W : (outDim, inDim)`, `x : (inDim,)`, `b : (outDim,)`.
PyTorch comparison: `torch.nn.functional.linear`.
-/
def linear {α : Type} [TorchLean.Storage α] [Add α] [Mul α] [Zero α]
  {inDim outDim : Nat}
  (t : Tape α) (wId bId xId : Nat) : Result (Tape α × Nat) := do
  let W ← requireValue (α:=α) (t:=t) (s:=.dim outDim (.dim inDim .scalar)) wId
  let b ← requireValue (α:=α) (t:=t) (s:=.dim outDim .scalar) bId
  let x ← requireValue (α:=α) (t:=t) (s:=.dim inDim .scalar) xId
  let layer : Spec.LinearSpec α inDim outDim := { weights := W, bias := b }
  let y := Spec.linearSpec (α:=α) layer x
  let node : Node α :=
    { name := some "linear"
      value := Spec.SomeTensor.ofTensor y
      requiresGrad :=
        (t.getNode? wId).any (·.requiresGrad) ||
        (t.getNode? bId).any (·.requiresGrad) ||
        (t.getNode? xId).any (·.requiresGrad)
      parents := #[wId, bId, xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := .dim outDim .scalar) dLdyAny
        let dW := Spec.linearWeightsDerivSpec (α:=α) x dLdy
        let db := Spec.linearBiasDerivSpec (α:=α) (dW) dLdy x
        let dx := Spec.linearInputDerivSpec (α:=α) W dLdy
        pure #[
          (wId, Spec.SomeTensor.ofTensor dW),
          (bId, Spec.SomeTensor.ofTensor db),
          (xId, Spec.SomeTensor.ofTensor dx)
        ]
    }
  pure (t.addNode node)

/--
Matrix-rank multiplication with explicit batch-prefix broadcasting.

`a` has shape `batchA ++ [m, n]`, `b` has shape `batchB ++ [n, p]`, and the result has
shape `batch ++ [m, p]`. The empty-prefix defaults preserve ordinary 2D matrix multiplication.
PyTorch comparison: `torch.matmul(a, b)` for operands of rank at least two.
-/
def matmul {α : Type} [TorchLean.Storage α] [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {m n p : Nat} (t : Tape α) (aId bId : Nat)
  (batchA : Shape := .scalar) (batchB : Shape := .scalar) (batch : Shape := .scalar)
  [broadcastA : Shape.BroadcastTo batchA batch]
  [broadcastB : Shape.BroadcastTo batchB batch] : Result (Tape α × Nat) := do
  let a ← requireValue (α := α) (t := t) (s := batchA.concat [m, n]) aId
  let b ← requireValue (α := α) (t := t) (s := batchB.concat [n, p]) bId
  let y := TorchLean.Tensor.matmulSpec broadcastA.proof broadcastB.proof a b
  let node : Node α :=
    { name := some "matmul"
      value := Spec.SomeTensor.ofTensor y
      requiresGrad :=
        (t.getNode? aId).any (·.requiresGrad) ||
        (t.getNode? bId).any (·.requiresGrad)
      parents := #[aId, bId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := batch.concat [m, p]) dLdyAny
        let (dA, dB) :=
          TorchLean.Tensor.matmulBackwardSpec broadcastA.proof broadcastB.proof a b dLdy
        pure #[(aId, Spec.SomeTensor.ofTensor dA), (bId, Spec.SomeTensor.ofTensor dB)]
    }
  pure (t.addNode node)

/--
Concatenate two tensors along dimension 0.

PyTorch comparison: `torch.cat([a, b], dim=0)`.
-/
def concatLeadingAxis {α : Type} [TorchLean.Storage α]
  {n m : Nat} {s : Shape} (t : Tape α) (aId bId : Nat) : Result (Tape α × Nat) := do
  let a ← requireValue (α := α) (t := t) (s := .dim n s) aId
  let b ← requireValue (α := α) (t := t) (s := .dim m s) bId
  let y := TorchLean.Tensor.concatAxisSpec .scalar (α := α) (n := n) (m := m) (suffix := s) a b
  let node : Node α :=
    { name := some "concat_leading_axis"
      value := Spec.SomeTensor.ofTensor y
      requiresGrad :=
        (t.getNode? aId).any (·.requiresGrad) ||
        (t.getNode? bId).any (·.requiresGrad)
      parents := #[aId, bId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := .dim (n + m) s) dLdyAny
        let dA := Spec.sliceRangeSpec (α := α) (n := n + m) (shape := s) dLdy 0 n
          (by simp)
        let dB := Spec.sliceRangeSpec (α := α) (n := n + m) (shape := s) dLdy n m
          (by simp)
        pure #[(aId, Spec.SomeTensor.ofTensor dA), (bId, Spec.SomeTensor.ofTensor dB)]
    }
  pure (t.addNode node)

/--
Slice along dimension 0: `x[start : start+len]`.

The proof argument `h` enforces bounds.
PyTorch comparison: `x[start:start+len]` on tensors with a leading dimension.
-/
def sliceLeadingAxisRange {α : Type} [TorchLean.Storage α] [Zero α]
  {n : Nat} {s : Shape} (t : Tape α) (xId : Nat) (start len : Nat) (h : start + len ≤ n) :
  Result (Tape α × Nat) :=
  unary (α := α) (t := t) (σ := .dim n s) (τ := .dim len s)
    "slice_leading_axis_range" xId
    (forward := fun x => Spec.sliceRangeSpec (α := α) (n := n) (shape := s) x start len h)
    (backward := fun _x dLdz =>
      TorchLean.Tensor.sliceAxisRangeBackwardSpec (α := α) (s := .dim n s) 0 start len h dLdz)

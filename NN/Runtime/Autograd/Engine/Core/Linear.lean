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
@[inline] def linear {α : Type} [TorchLean.Storage α] [Add α] [Mul α] [Zero α]
  {inDim outDim : Nat}
  (t : Tape α) (wId bId xId : Nat) : Result (Tape α × Nat) := do
  let W ← requireValue (α:=α) (t:=t) (s:=.dim outDim (.dim inDim .scalar)) wId
  let b ← requireValue (α:=α) (t:=t) (s:=.dim outDim .scalar) bId
  let x ← requireValue (α:=α) (t:=t) (s:=.dim inDim .scalar) xId
  let layer : Spec.LinearSpec α inDim outDim := { weights := W, bias := b }
  let y := Spec.linearSpec (α:=α) layer x
  let weightGrad := (t.getNode? wId).any (·.requiresGrad)
  let biasGrad := (t.getNode? bId).any (·.requiresGrad)
  let inputGrad := (t.getNode? xId).any (·.requiresGrad)
  let node : Node α :=
    { name := some "linear"
      value := Spec.SomeTensor.ofTensor y
      requiresGrad := weightGrad || biasGrad || inputGrad
      parents := #[wId, bId, xId]
      -- Gradient accumulation drops contributions to parents that do not require gradients, so
      -- those products are skipped here. A frozen weight or a data input costs nothing.
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := .dim outDim .scalar) dLdyAny
        let mut contributions : Array (Nat × Spec.SomeTensor α) := Array.mkEmpty 3
        if weightGrad then
          contributions := contributions.push
            (wId, Spec.SomeTensor.ofTensor (Spec.linearWeightsDerivSpec (α := α) x dLdy))
        if biasGrad then
          contributions := contributions.push (bId, Spec.SomeTensor.ofTensor dLdy)
        if inputGrad then
          contributions := contributions.push
            (xId, Spec.SomeTensor.ofTensor (Spec.linearInputDerivSpec (α := α) W dLdy))
        pure contributions
    }
  pure (t.addNode node)

/--
Matrix-rank multiplication with explicit batch-prefix broadcasting.

`a` has shape `batchA ++ [m, n]`, `b` has shape `batchB ++ [n, p]`, and the result has
shape `batch ++ [m, p]`. The empty-prefix defaults preserve ordinary 2D matrix multiplication.
PyTorch comparison: `torch.matmul(a, b)` for operands of rank at least two.
-/
@[inline] def matmul {α : Type} [TorchLean.Storage α] [Add α] [Mul α] [Zero α]
  {m n p : Nat} (t : Tape α) (aId bId : Nat)
  (batchA : Shape := .scalar) (batchB : Shape := .scalar) (batch : Shape := .scalar)
  [broadcastA : Shape.BroadcastTo batchA batch]
  [broadcastB : Shape.BroadcastTo batchB batch] : Result (Tape α × Nat) :=
  binary (α := α) (t := t) (σ₁ := batchA.concat [m, n]) (σ₂ := batchB.concat [n, p])
    (τ := batch.concat [m, p]) "matmul" aId bId
    (forward := TorchLean.Tensor.matmulSpec broadcastA.proof broadcastB.proof)
    (backward := TorchLean.Tensor.matmulBackwardSpec broadcastA.proof broadcastB.proof)

/--
Concatenate two tensors along dimension 0.

PyTorch comparison: `torch.cat([a, b], dim=0)`.
-/
@[inline] def concat {α : Type} [TorchLean.Storage α]
  {n m : Nat} {s : Shape} (t : Tape α) (aId bId : Nat) : Result (Tape α × Nat) :=
  binary (α := α) (t := t) (σ₁ := .dim n s) (σ₂ := .dim m s) (τ := .dim (n + m) s)
    "concat_leading_axis" aId bId
    (forward := TorchLean.Tensor.concatAxisSpec .scalar)
    (backward := fun _a _b dLdy =>
      (Spec.sliceRangeSpec dLdy 0 n (by simp), Spec.sliceRangeSpec dLdy n m (by simp)))

/--
Slice along dimension 0: `x[start : start+len]`.

The proof argument `h` enforces bounds.
PyTorch comparison: `x[start:start+len]` on tensors with a leading dimension.
-/
@[inline] def slice {α : Type} [TorchLean.Storage α] [Zero α]
  {n : Nat} {s : Shape} (t : Tape α) (xId : Nat) (start len : Nat) (h : start + len ≤ n) :
  Result (Tape α × Nat) :=
  unary (α := α) (t := t) (σ := .dim n s) (τ := .dim len s)
    "slice" xId
    (forward := fun x => Spec.sliceRangeSpec (α := α) (n := n) (shape := s) x start len h)
    (backward := fun _x dLdz =>
      TorchLean.Tensor.sliceAxisRangeBackwardSpec (α := α) (s := .dim n s) 0 start len h dLdz)

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

CUDA helpers: shape/broadcast metadata derived from `Spec.Shape` proofs.

In particular, CUDA broadcast kernels operate on explicit runtime arrays:
- `inDims  : Array Nat` (outermost-first)
- `outDims : Array Nat` (outermost-first)
- `axisMap : Array Nat` of length `outDims.size`

The `axisMap` encoding matches `csrc/cuda/kernels/torchlean_cuda_kernels.cu`:
- `axisMap[j] = 0` means output axis `j` is an inserted/broadcast axis (input coordinate is `0`)
- `axisMap[j] = inAxis+1` maps output axis `j` to input axis `inAxis` (0-based), with the `+1`
  sentinel so `0` can be reserved for inserted axes.

This module provides a total function producing `axisMap` from a `Shape.CanBroadcastTo` proof.
-/

module

public import NN.Spec.Core.Shape

@[expose] public section

namespace Runtime
namespace Autograd
namespace Cuda

open Spec

namespace Broadcast

/-!
### `axisMap` generation

The recursion follows the structure of `Shape.CanBroadcastTo`:
- `expand_dims` inserts a new *outer* axis in the output, so we prepend `0`.
- `dim_eq` / `dim_1_to_n` align an output axis with an input axis, so we prepend `1` (maps to input
  axis 0) and shift all tail mappings by `+1` (because tail input axes are one level deeper).
-/

def shiftInputAxes (m : Array Nat) : Array Nat :=
  m.map (fun v => if v == 0 then 0 else v + 1)

/-- Generate the CUDA `axisMap` array from a `Shape.CanBroadcastTo` proof. -/
def axisMap : {s₁ s₂ : Shape} → Shape.CanBroadcastTo s₁ s₂ → Array Nat
  | _, _, .scalar_to_any s₂ =>
      Array.replicate (Shape.rank s₂) 0
  | _, _, .dim_eq tail =>
      #[1] ++ shiftInputAxes (axisMap tail)
  | _, _, .dim_1_to_n tail =>
      #[1] ++ shiftInputAxes (axisMap tail)
  | _, _, .expand_dims tail =>
      #[0] ++ axisMap tail

/-- Convenience bundle for CUDA broadcast kernels: `(inDims, outDims, axisMap)`. -/
def broadcastArgs {s₁ s₂ : Shape} (cb : Shape.CanBroadcastTo s₁ s₂) :
    Array Nat × Array Nat × Array Nat :=
  (Shape.toArray s₁, Shape.toArray s₂, axisMap cb)

end Broadcast

end Cuda
end Autograd
end Runtime

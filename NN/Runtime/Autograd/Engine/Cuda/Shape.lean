/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

CUDA helpers: shape/broadcast metadata derived from `Spec.Shape` proofs.

In particular, CUDA broadcast kernels operate on explicit runtime arrays:
- `inDims  : Array Nat` (outermost-first)
- `outDims : Array Nat` (outermost-first)
- `axisMap : Array Nat` of length `outDims.size`

The `axisMap` encoding matches `csrc/libtorch/kernels.cpp`:
- `axisMap[j] = 0` means output axis `j` is an inserted/broadcast axis (input coordinate is `0`)
- `axisMap[j] = inAxis+1` maps output axis `j` to input axis `inAxis` (0-based), with the `+1`
  sentinel so `0` can be reserved for inserted axes.

This module provides a total function producing `axisMap` from a `Shape.CanBroadcastTo` proof.
-/

module

public import NN.Spec.Core.TensorReductionShape.Reductions

@[expose] public section

namespace Runtime
namespace Autograd
namespace Cuda

open Spec TorchLean

namespace Broadcast

/-! ### `axisMap` generation -/

/-- Shift every nonzero entry of a partial axis map up by one, making room for a leading axis. -/
def shiftInputAxes (m : Array Nat) : Array Nat :=
  m.map (fun v => if v == 0 then 0 else v + 1)

/-- Generate the CUDA `axisMap` for right-aligned broadcasting.

The proof establishes compatibility, but the map itself is determined solely by the two ranks:
missing leading axes map to zero, and every remaining output axis maps to its aligned input axis. -/
def axisMap {s₁ s₂ : Shape} (_cb : Shape.CanBroadcastTo s₁ s₂) : Array Nat :=
  Array.replicate (Shape.rank s₂ - Shape.rank s₁) 0 ++
    (Array.range (Shape.rank s₁)).map (fun i => i + 1)

/-- CUDA axis map that restores the axis removed by `shapeAfterSum`. -/
def afterSumAxisMap : (s : Shape) → (axis : Nat) → Array Nat
  | .scalar, _ => #[]
  | .dim _ inner, 0 => #[0] ++ (Array.range (Shape.rank inner)).map (fun i => i + 1)
  | .dim _ inner, Nat.succ axis => #[1] ++ shiftInputAxes (afterSumAxisMap inner axis)

/-- CUDA metadata for `TorchLean.Tensor.broadcastAfterSum`. -/
def afterSumArgs (s : Shape) (axis : Nat) : Array Nat × Array Nat × Array Nat :=
  (Shape.toArray (TorchLean.Tensor.shapeAfterSum s axis), Shape.toArray s, afterSumAxisMap s axis)

end Broadcast

end Cuda
end Autograd
end Runtime

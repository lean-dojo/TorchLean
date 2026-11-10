/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

CUDA helpers: broadcast metadata for `Shape` values.

CUDA broadcast kernels take three runtime arrays:
- `inDims  : Array Nat` (outermost first)
- `outDims : Array Nat` (outermost first)
- `axisMap : Array Nat` of length `outDims.size`

The `axisMap` encoding matches `csrc/libtorch/torchlean.cpp`:
- `axisMap[j] = 0` marks output axis `j` as an inserted axis.
- `axisMap[j] = inAxis+1` maps output axis `j` to input axis `inAxis` (0-based), and the `+1`
  keeps `0` free for inserted axes.

The native side checks that each input axis appears once, with extent `1` or the extent of its
output axis. `Broadcast.axisMap` builds the map from a `Shape.CanBroadcastTo` proof.
-/

module

public import TorchLean.Tensor.Reduce

@[expose] public section

open TorchLean.Shape (rank)

namespace TorchLean
namespace LibTorch

namespace Broadcast

/-! ### `axisMap` generation -/

/-- Generate the CUDA `axisMap` for right-aligned broadcasting.

The proof establishes compatibility, but the map itself is determined solely by the two ranks:
missing leading axes map to zero, and every remaining output axis maps to its aligned input axis. -/
def axisMap {s₁ s₂ : Shape} (_cb : Shape.CanBroadcastTo s₁ s₂) : Array Nat :=
  Array.replicate (Shape.rank s₂ - Shape.rank s₁) 0 ++
    (Array.range (Shape.rank s₁)).map (fun i => i + 1)

/-- CUDA axis map that restores the axis removed by `List.eraseIdx`.

An out-of-range axis gives the identity map, matching `List.eraseIdx`, which leaves the shape
unchanged. Building each entry from its index avoids copying and shifting the suffix at every
level of the shape. -/
def afterSumAxisMap (s : Shape) (axis : Nat) : Array Nat :=
  Array.ofFn fun i : Fin (Shape.rank s) =>
    if i.val < axis then i.val + 1 else if i.val = axis then 0 else i.val

/-- The `(inDims, outDims, axisMap)` arguments of `Buffer.broadcastTo` that broadcast a sum over
`axis` back to `s`. -/
def afterSumArgs (s : Shape) (axis : Nat) : Array Nat × Array Nat × Array Nat :=
  ((s.eraseIdx axis).toArray, s.toArray, afterSumAxisMap s axis)

end Broadcast

end LibTorch
end TorchLean

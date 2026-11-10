/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Builder.Conv -- shake: keep

/-!
# Pooling

The check `TorchLean.Window.validatePool` of a pooling layer. A pooling layer is described by a
`TorchLean.Window` alone, and its output grid is `TorchLean.Window.poolOutputSpatial`.
-/

@[expose] public section

namespace TorchLean.Window

/-- Check a pooling layer: a positive channel count, a nonempty input grid, positive kernel and
stride entries, and a nonempty output grid under `poolOutputSpatial`. -/
def validatePool {d : Nat} (window : Window d)
    (channels : Nat) (input : Tensor Nat [d])
    (kind : String := "Pooling") : Except String Unit := do
  if channels = 0 then
    throw s!"{kind}: channel count must be positive"
  nn.Layer.validateSpatial kind input window.kernelSize window.stride
  if (window.poolOutputSpatial input).prod = 0 then
    throw s!"{kind}: geometry produced an empty spatial grid"

end TorchLean.Window

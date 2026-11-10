/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Layers.Conv

@[expose] public section


namespace TorchLean
open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

/-!
# Spatial Pooling

The general pooling operators are `TorchLean.Tensor.pool`, `poolBackward`, `smoothMaxPool` and
`adaptivePool` (`TorchLean.Tensor.Window`), over any axes of any shape. This module
defines the output extents of spatial pooling: `Pool.outSpatial`, which applies
`Shape.poolExtent` to every axis, and `Window.poolOutputSpatial`, the pooling rule of a `Window`.

Conventions:
- Every axis of the pooled tensor is spatial.
- `kernel`, `stride`, and `padding` are per-axis vectors (`Tensor Nat [d]`).
- Padding is symmetric.

The extents are the floor formula of `torch.nn.MaxPool2d` (PyTorch 2.8 docs),
`⌊(H_in + 2 padding - dilation (kernel - 1) - 1) / stride + 1⌋`, written by hand at unit dilation.
The bound of half the kernel on the padding is the one the `torch.nn.AvgPool2d` page asks for.
-/

/-!
### Output shapes
-/

/-- Output grid of spatial pooling with equal padding on both sides of each axis: `Shape.poolExtent`
in floor mode with unit dilation. An empty input axis or kernel, a zero stride, padding beyond half
the kernel, or a window longer than the padded axis gives an empty output axis. -/
def Pool.outSpatial {d : Nat} (inSpatial kernel stride padding : Tensor Nat [d]) :
    Tensor Nat [d] :=
  Tensor.ofFn (fun i =>
    Shape.poolExtent (inSpatial.getScalar i) (kernel.getScalar i) (stride.getScalar i)
      (padding.getScalar i) (padding.getScalar i) 1 false)

/-- Each output extent of a pooling layer is `Shape.poolExtent` of its axis. -/
theorem Pool.outSpatial_apply {d : Nat} (inSpatial kernel stride padding : Tensor Nat [d])
    (axis : Shape.Coord [d]) :
    outSpatial inSpatial kernel stride padding axis =
      Shape.poolExtent (inSpatial.getScalar axis.1) (kernel.getScalar axis.1)
        (stride.getScalar axis.1) (padding.getScalar axis.1) (padding.getScalar axis.1) 1 false :=
  Tensor.ofFn_apply _ axis

/-- Output grid of a pooling layer with this window (`Pool.outSpatial`). -/
def Window.poolOutputSpatial {d : Nat} (window : Window d) (input : Tensor Nat [d]) :
    Tensor Nat [d] :=
  Pool.outSpatial input window.kernelSize window.stride window.padding

end TorchLean

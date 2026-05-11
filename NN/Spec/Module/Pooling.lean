/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Pooling
public import NN.Spec.Module.SpecModule

/-!
# Pooling module wrappers

These wrappers expose pooling specs as `NNModuleSpec`s.

Conventions:

- Channel-first images use shape `(C, H, W)` at the spec level (`.dim C (.dim H (.dim W .scalar))`).
- `MaxPool2DModuleSpec` applies the spatial max-pool independently per channel.
- `AvgPool2DModuleSpec` is provided for a single-channel 2D tensor; multi-channel usage typically
  maps it per channel in the same way as max-pool.

If you want a PyTorch mapping: `nn.MaxPool2d` / `nn.AvgPool2d` on a single `(C,H,W)` image (no
  batch).
-/

@[expose] public section


namespace Spec
open Tensor
open ModSpec

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

-- MaxPool2D module specification wrapper
/-- MaxPool2D wrapper (channel-first, pool applied per channel). -/
def MaxPool2DModuleSpec {kH kW stride inH inW inC: Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  {hStride : stride ≠ 0}
  (m : MaxPool2DSpec kH kW stride h1 h2 hStride) :
  NNModuleSpec α
    (.dim inC (.dim inH (.dim inW .scalar)))
    (.dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1) .scalar))) :=
{ forward := fun x =>
    -- Apply pooling to each channel independently.
    Tensor.dim (fun c => maxPool2dSpec m (getAtSpec x c)),
  kind := "MaxPool2D",
  export_func := {
  toPyTorch := s!"nn.MaxPool2d(kernel_size=({kH}, {kW}), stride={stride})",
  dimensions := (inC, inC)  -- MaxPool preserves channel count
} }

-- AvgPool2D module specification wrapper
/-- AvgPool2D wrapper (2D tensor). -/
def AvgPool2DModuleSpec {kH kW stride inH inW : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  {hStride : stride ≠ 0}
  (m : AvgPool2DSpec kH kW stride h1 h2 hStride) :
  NNModuleSpec α
    (.dim inH (.dim inW .scalar))
    (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1) .scalar)) :=
{ forward := fun x => avgPool2dSpec (layer := m) x, kind := "AvgPool2D", export_func := {
  toPyTorch := s!"nn.AvgPool2d(kernel_size=({kH}, {kW}), stride={stride})",
  dimensions := (inH, (inH - kH) / stride + 1)  -- Height dimensions
} }

end Spec

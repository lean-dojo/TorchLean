/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Neural.Layers.Convolution -- shake: keep

/-!
# Pooling

Arbitrary-rank pooling configuration.
-/

@[expose] public section

namespace TorchLean
namespace nn

/-- Configuration shared by arbitrary-dimensional pooling layers. -/
structure Pooling.Config (d : Nat) where
  /-- Window extent along each spatial axis. -/
  kernelSize : Tensor Nat [d]
  /-- Step along each spatial axis. -/
  stride : Tensor Nat [d] := Tensor.ones [d]
  /-- Symmetric padding along each spatial axis. -/
  padding : Tensor Nat [d] := Tensor.zeros [d]

/-- Output grid produced from an input grid by this pooling configuration. -/
def Pooling.Config.outputSpatial {d : Nat} (config : Pooling.Config d)
    (input : Tensor Nat [d]) : Tensor Nat [d] :=
  Spec.poolOutSpatialPad input
    config.kernelSize
    config.stride
    config.padding

namespace Pooling.Config

/-- Validate channel width and pooling geometry. -/
def validate {d : Nat} (config : Pooling.Config d)
    (channels : Nat) (input : Tensor Nat [d])
    (kind : String := "Pooling") : Except String Unit := do
  if channels = 0 then
    throw s!"{kind}: channel count must be positive"
  if input.prod = 0 then
    throw s!"{kind}: input spatial dimensions must be positive"
  if !decide (∀ axis : Fin d, config.kernelSize.getScalar axis ≠ 0) then
    throw s!"{kind}: kernel size entries must be positive"
  if !decide (∀ axis : Fin d, config.stride.getScalar axis ≠ 0) then
    throw s!"{kind}: stride entries must be positive"
  if (config.outputSpatial input).prod = 0 then
    throw s!"{kind}: geometry produced an empty spatial grid"

end Pooling.Config

end nn
end TorchLean

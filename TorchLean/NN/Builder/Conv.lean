/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Builder.Layer -- shake: keep

/-!
# Convolution

The configuration record of convolution and transpose convolution layers of any spatial rank.
`Conv.Config` extends `TorchLean.Window` with an output width and a weight initialization.
`nn.conv` computes its output grid with `TorchLean.Window.convOutputSpatial` and `nn.convTranspose`
with `TorchLean.Window.transposedOutputSpatial`. `Conv.Config.validate` checks a convolution,
and a transpose convolution passes its own grid to the same checks, `validateConvolution`.
`TorchLean.Window.convolution` adds a width to a window.
-/

@[expose] public section

namespace TorchLean
namespace nn

/-- Configuration of an arbitrary-dimensional convolution or transpose convolution layer. -/
structure Conv.Config (d : Nat) extends Window d where
  /-- Number of output channels. Must be positive when the layer is validated. -/
  outChannels : Nat
  /-- Initialization scheme for the kernel weights. -/
  weightInit : Init.Scheme := .uniform (-0.1) 0.1

/--
Checks shared by both convolution directions: both channel counts, every input extent and every
kernel and stride entry must be positive, the grid `output` that the caller computes must be
nonempty, and `initialization` must pass validation.
-/
def validateConvolution {d : Nat}
    (inputChannels outputChannels : Nat) (input kernelSize stride output : Tensor Nat [d])
    (initialization : Init.Scheme) (kind : String) : Except String Unit := do
  if inputChannels = 0 then
    throw s!"{kind}: input channel count must be positive"
  if outputChannels = 0 then
    throw s!"{kind}: output channel count must be positive"
  Layer.validateSpatial kind input kernelSize stride
  if output.prod = 0 then
    throw s!"{kind}: geometry produced an empty spatial grid"
  initialization.validate

namespace Conv.Config

/-- Check the channel counts, the forward output grid `config.convOutputSpatial input` and the
kernel initialization with `validateConvolution`. -/
def validate {d : Nat} (config : Conv.Config d)
    (inputChannels : Nat) (input : Tensor Nat [d])
    (kind : String := "Conv") : Except String Unit :=
  validateConvolution inputChannels config.outChannels input
    config.kernelSize config.stride (config.convOutputSpatial input)
    config.weightInit kind

end Conv.Config

end nn
end TorchLean

namespace TorchLean.Window

/-- The convolution configuration with this window and `outChannels` output channels. -/
def convolution {d : Nat} (window : Window d) (outChannels : Nat) :
    nn.Conv.Config d :=
  { toWindow := window, outChannels }

/-- A convolution built from a window keeps that window. -/
@[simp] theorem convolution_toWindow {d : Nat} (window : Window d) (outChannels : Nat) :
    (window.convolution outChannels).toWindow = window :=
  rfl

/-- A convolution built from a window keeps the requested output width. -/
@[simp] theorem convolution_outChannels {d : Nat} (window : Window d) (outChannels : Nat) :
    (window.convolution outChannels).outChannels = outChannels :=
  rfl

end TorchLean.Window

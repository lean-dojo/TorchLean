/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Neural.Builders -- shake: keep

/-!
# Convolution

Arbitrary-rank convolution geometry and configuration records.
-/

@[expose] public section

namespace TorchLean
namespace nn

/-- Kernel, stride, and padding shared by convolutional layers with different channel widths. -/
structure Convolution.Geometry (d : Nat) where
  /-- Kernel extent along each spatial axis. -/
  kernelSize : Tensor Nat [d]
  /-- Step along each spatial axis. -/
  stride : Tensor Nat [d] := Tensor.ones [d]
  /-- Symmetric zero-padding along each spatial axis. -/
  padding : Tensor Nat [d] := Tensor.zeros [d]

namespace Convolution.Geometry

/-- Output grid produced by this convolution geometry. -/
def outputSpatial {d : Nat} (geometry : Convolution.Geometry d)
    (input : Tensor Nat [d]) : Tensor Nat [d] :=
  Spec.convOutSpatial input
    geometry.kernelSize
    geometry.stride
    geometry.padding

/-- Output grid produced when this geometry is used for transpose convolution. -/
def transposedOutputSpatial {d : Nat} (geometry : Convolution.Geometry d)
    (input : Tensor Nat [d]) : Tensor Nat [d] :=
  Spec.convTransposeOutSpatial input
    geometry.kernelSize
    geometry.stride
    geometry.padding

/--
Unit-stride geometry with an odd kernel along each axis and padding equal to the kernel radius.
-/
def samePadding {d : Nat} (radius : Tensor Nat [d]) : Convolution.Geometry d :=
  { kernelSize := radius.map (fun extent => 2 * extent + 1)
    stride := Tensor.ones [d]
    padding := radius }

/-- Same-padding geometry preserves every input extent. -/
theorem outputSpatial_samePadding {d : Nat} (input radius : Tensor Nat [d]) :
    (samePadding radius).outputSpatial input = input := by
  rw [outputSpatial]
  exact Spec.convOutSpatial_same input radius

end Convolution.Geometry

/-- Configuration shared by arbitrary-dimensional convolution layers. -/
structure Convolution.Config (d : Nat) where
  /-- Number of output channels. Must be positive when the layer is validated. -/
  outChannels : Nat
  /-- Kernel extent along each spatial axis. -/
  kernelSize : Tensor Nat [d]
  /-- Step along each spatial axis. -/
  stride : Tensor Nat [d] := Tensor.ones [d]
  /-- Symmetric zero-padding along each spatial axis. -/
  padding : Tensor Nat [d] := Tensor.zeros [d]
  /-- Initialization scheme for the kernel weights. -/
  weightInitialization : Init.Scheme := .uniform (-0.1) 0.1

/-- Output grid produced from an input grid by this convolution configuration. -/
def Convolution.Config.outputSpatial {d : Nat} (config : Convolution.Config d)
    (input : Tensor Nat [d]) : Tensor Nat [d] :=
  Spec.convOutSpatial input
    config.kernelSize
    config.stride
    config.padding

/-- Shared convolution checks using the output geometry computed by the caller. -/
def Internal.validateConvolution {d : Nat}
    (inputChannels outputChannels : Nat) (input kernelSize stride output : Tensor Nat [d])
    (initialization : Init.Scheme) (kind : String) : Except String Unit := do
  if inputChannels = 0 then
    throw s!"{kind}: input channel count must be positive"
  if outputChannels = 0 then
    throw s!"{kind}: output channel count must be positive"
  if input.prod = 0 then
    throw s!"{kind}: input spatial dimensions must be positive"
  if !decide (∀ axis : Fin d, kernelSize.getScalar axis ≠ 0) then
    throw s!"{kind}: kernel size entries must be positive"
  if !decide (∀ axis : Fin d, stride.getScalar axis ≠ 0) then
    throw s!"{kind}: stride entries must be positive"
  if output.prod = 0 then
    throw s!"{kind}: geometry produced an empty spatial grid"
  initialization.validate

namespace Convolution.Config

/-- Validate channel widths, spatial geometry, and kernel initialization. -/
def validate {d : Nat} (config : Convolution.Config d)
    (inputChannels : Nat) (input : Tensor Nat [d])
    (kind : String := "Conv") : Except String Unit :=
  Internal.validateConvolution inputChannels config.outChannels input
    config.kernelSize config.stride (config.outputSpatial input) config.weightInitialization kind

end Convolution.Config

/-- Build a convolution configuration by adding an output-channel width to shared geometry. -/
def Convolution.Geometry.convolution {d : Nat} (geometry : Convolution.Geometry d)
    (outChannels : Nat) : Convolution.Config d :=
  { outChannels
    kernelSize := geometry.kernelSize
    stride := geometry.stride
    padding := geometry.padding }

/-- Adding an output width to a geometry keeps that width. -/
@[simp] theorem Convolution.Geometry.convolution_outChannels {d : Nat}
    (geometry : Convolution.Geometry d)
    (outChannels : Nat) :
    (geometry.convolution outChannels).outChannels = outChannels :=
  rfl

/-- The spatial grid is the geometry's, so a shape proof can be discharged from the geometry alone
without unfolding the configuration the builder produced. -/
@[simp] theorem Convolution.Geometry.convolution_outputSpatial {d : Nat}
    (geometry : Convolution.Geometry d)
    (outChannels : Nat) (input : Tensor Nat [d]) :
    (geometry.convolution outChannels).outputSpatial input = geometry.outputSpatial input :=
  rfl

/-- Configuration shared by arbitrary-dimensional transposed-convolution layers. -/
structure TransposedConvolution.Config (d : Nat) where
  /-- Number of output channels. Must be positive when the layer is validated. -/
  outChannels : Nat
  /-- Kernel extent along each spatial axis. -/
  kernelSize : Tensor Nat [d]
  /-- Step along each spatial axis. -/
  stride : Tensor Nat [d] := Tensor.ones [d]
  /-- Symmetric zero-padding along each spatial axis. -/
  padding : Tensor Nat [d] := Tensor.zeros [d]
  /-- Initialization scheme for the kernel weights. -/
  weightInitialization : Init.Scheme := .uniform (-0.1) 0.1

/-- Output grid produced from an input grid by this transpose-convolution configuration. -/
def TransposedConvolution.Config.outputSpatial {d : Nat}
    (config : TransposedConvolution.Config d)
    (input : Tensor Nat [d]) : Tensor Nat [d] :=
  Spec.convTransposeOutSpatial input
    config.kernelSize
    config.stride
    config.padding

namespace TransposedConvolution.Config

/-- Validate channel widths, spatial geometry, and kernel initialization. -/
def validate {d : Nat} (config : TransposedConvolution.Config d)
    (inputChannels : Nat) (input : Tensor Nat [d])
    (kind : String := "ConvTranspose") : Except String Unit :=
  Internal.validateConvolution inputChannels config.outChannels input
    config.kernelSize config.stride (config.outputSpatial input) config.weightInitialization kind

end TransposedConvolution.Config

/-- Build a transpose-convolution configuration from shared geometry. -/
def Convolution.Geometry.transposedConvolution {d : Nat}
    (geometry : Convolution.Geometry d)
    (outChannels : Nat) : TransposedConvolution.Config d :=
  { outChannels
    kernelSize := geometry.kernelSize
    stride := geometry.stride
    padding := geometry.padding }

/-- Same for the transpose direction: the requested output width survives. -/
@[simp] theorem Convolution.Geometry.transposedConvolution_outChannels {d : Nat}
    (geometry : Convolution.Geometry d) (outChannels : Nat) :
    (geometry.transposedConvolution outChannels).outChannels = outChannels :=
  rfl

/-- Reusing one geometry for both directions is only sound if each direction keeps its own output
rule, and it does: the transpose configuration's grid is `Geometry.transposedOutputSpatial`, never
the forward `Geometry.outputSpatial`. -/
@[simp] theorem Convolution.Geometry.transposedConvolution_outputSpatial {d : Nat}
    (geometry : Convolution.Geometry d) (outChannels : Nat) (input : Tensor Nat [d]) :
    (geometry.transposedConvolution outChannels).outputSpatial input =
      geometry.transposedOutputSpatial input :=
  rfl

end nn
end TorchLean

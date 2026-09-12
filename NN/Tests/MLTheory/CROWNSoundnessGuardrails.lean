/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import Mathlib.Tactic.FinCases
public import NN.MLTheory.CROWN.Graph.Engine
public import NN.Tests.MLTheory.Utils
public import NN.Tests.Utils
public import NN.Tensor

/-!
# CROWN Soundness Guardrails

Check the boundary between supported IR configurations and missing transfer rules. Unsupported
convolution geometry and mismatched LayerNorm payload shapes must return no bounds. Dense
convolution and last-axis affine LayerNorm retain their value bounds; LayerNorm payload
derivatives remain unresolved.
-/

@[expose] public section

namespace NN.Tests.MLTheory.CROWNSoundnessGuardrails

open Spec TorchLean
open NN.IR
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph

open Tests.Utils (assertBoundAt assertNoBoundAt)
open NN.Tests.MLTheory.Utils (pointFlatBox)

def checkConvolutionGuards : IO Unit := do
  let inputSpatial : Tensor Nat [1] := [4]
  let kernel : Tensor Nat [1] := [2]
  let stride : Tensor Nat [1] := [1]
  let padding : Tensor Nat [1] := [1]
  let dilationOne : Tensor Nat [1] := [1]
  let dilationTwo : Tensor Nat [1] := [2]
  let paddingAfterZero : Tensor Nat [1] := [0]
  let weights : Tensor Float [2, 2, 2] := Tensor.full (α := Float) [2, 2, 2] 1
  let bias : Tensor Float [2] := Tensor.full (α := Float) [2] 0
  have hKernel : ∀ i : Fin 1, kernel.getScalar i ≠ 0 := by
    intro i
    fin_cases i
    simp [kernel]
  have hStride : ∀ i : Fin 1, stride.getScalar i ≠ 0 := by
    intro i
    fin_cases i
    simp [stride]
  let base : ConvParams Float :=
    { spatialRank := 1
      inChannels := 2
      outChannels := 2
      kernel := kernel
      stride := stride
      padding := padding
      dilation := dilationOne
      paddingAfter := padding
      groups := 1
      inputSpatial := inputSpatial
      kernelNonzero := hKernel
      strideNonzero := hStride
      spec := { kernel := by simpa [kernel] using weights, bias := bias } }
  let baseConfig : ConvConfig :=
    { spatialRank := 1
      kernel := kernel
      stride := stride
      padding := padding
      dilation := dilationOne
      paddingAfter := padding
      groups := 1
      channelAxis := 0
      inChannels := 2
      outChannels := 2 }
  let input : Tensor Float [2, 4] := Tensor.full (α := Float) [2, 4] 1
  let inputBox := pointFlatBox input
  let graphFor (config : ConvConfig) (outShape : Shape) : NN.IR.Graph :=
    { nodes :=
        #[ { id := 0, parents := #[], kind := .input, outShape := [2, 4] }
         , { id := 1, parents := #[0], kind := .conv config, outShape := outShape } ] }
  let storeFor (params : ConvParams Float) : ParamStore Float :=
    { inputBoxes := Std.HashMap.emptyWithCapacity.insert 0 inputBox
      convCfg := Std.HashMap.emptyWithCapacity.insert 1 params }
  let supportedGraph := graphFor baseConfig [2, 5]
  let supportedStore := storeFor base
  unless crownGraphSemanticsSupported supportedGraph supportedStore do
    throw <| IO.userError "supported dense convolution was rejected"
  assertBoundAt "supported dense convolution"
    (runIBP supportedGraph supportedStore) 1

  let groupedParams := { base with groups := 2 }
  let groupedConfig := { baseConfig with groups := 2 }
  let groupedGraph := graphFor groupedConfig [2, 5]
  let groupedStore := storeFor groupedParams
  unless !crownGraphSemanticsSupported groupedGraph groupedStore do
    throw <| IO.userError "grouped convolution was accepted"
  assertNoBoundAt "grouped convolution" (runIBP groupedGraph groupedStore) 1

  let dilatedParams := { base with dilation := dilationTwo }
  let dilatedConfig := { baseConfig with dilation := dilationTwo }
  let dilatedGraph := graphFor dilatedConfig [2, 4]
  let dilatedStore := storeFor dilatedParams
  unless !crownGraphSemanticsSupported dilatedGraph dilatedStore do
    throw <| IO.userError "dilated convolution was accepted"
  assertNoBoundAt "dilated convolution" (runIBP dilatedGraph dilatedStore) 1

  let asymmetricParams := { base with paddingAfter := paddingAfterZero }
  let asymmetricConfig := { baseConfig with paddingAfter := paddingAfterZero }
  let asymmetricGraph := graphFor asymmetricConfig [2, 4]
  let asymmetricStore := storeFor asymmetricParams
  unless !crownGraphSemanticsSupported asymmetricGraph asymmetricStore do
    throw <| IO.userError "asymmetrically padded convolution was accepted"
  let ibp := runIBP asymmetricGraph asymmetricStore
  assertNoBoundAt "asymmetrically padded convolution" ibp 1
  let ctx : AffineCtx := { inputId := 0, inputDim := 8 }
  assertNoBoundAt "asymmetric convolution CROWN"
    (runCROWN asymmetricGraph asymmetricStore ctx ibp) 1

/--
Keep affine LayerNorm value bounds separate from unsupported shapes and derivative rules.

For the constant input, each normalized row is zero and the bias makes every output three.
The backward objective sums all four entries, so its bounds must contain twelve. A payload with
the same number of entries but a different shape must still be rejected, even if a caller
supplies a precomputed output box.
-/
def checkLayerNormPayloadGuard : IO Unit := do
  let input : Tensor Float [2, 2] := Tensor.full (α := Float) [2, 2] 1
  let params : LayerNormParams Float :=
    { normalizedShape := [2]
      gamma := Tensor.full (α := Float) [2] 2
      beta := Tensor.full (α := Float) [2] 3
      eps := 0.25 }
  let graph : NN.IR.Graph :=
    { nodes :=
        #[ { id := 0, parents := #[], kind := .input, outShape := [2, 2] }
         , { id := 1, parents := #[0], kind := .layernorm 1, outShape := [2, 2] } ] }
  let store : ParamStore Float :=
    { inputBoxes := Std.HashMap.emptyWithCapacity.insert 0 (pointFlatBox input)
      layerNorm := Std.HashMap.emptyWithCapacity.insert 1 params }
  unless crownGraphSemanticsSupported graph store do
    throw <| IO.userError "last-axis LayerNorm payload was rejected"
  let ibp := runIBP graph store
  assertBoundAt "affine LayerNorm IBP" ibp 1
  let ctx : AffineCtx := { inputId := 0, inputDim := 4 }
  assertBoundAt "affine LayerNorm affine bounds" (runAffine graph store ctx ibp) 1
  assertBoundAt "affine LayerNorm CROWN" (runCROWN graph store ctx ibp) 1
  let objective : FlatTensor Float := { n := 4, v := Tensor.full (α := Float) [4] 1 }
  let .ok objectiveBox := backwardObjectiveBox? graph store ctx ibp (pointFlatBox input) 1 objective
    | throw <| IO.userError "affine LayerNorm backward objective was rejected"
  let lower := Tensor.to objectiveBox.lo (Array Float)
  let upper := Tensor.to objectiveBox.hi (Array Float)
  unless lower.size == 1 && upper.size == 1 && lower[0]! <= 12 && 12 <= upper[0]! do
    throw <| IO.userError "affine LayerNorm objective bounds omitted the bias"
  let derivatives := runDirectionalDerivative graph store ibp (pointFlatBox input)
  assertBoundAt "LayerNorm input derivative seed" derivatives 0
  assertNoBoundAt "LayerNorm payload first derivative" derivatives 1
  assertNoBoundAt "LayerNorm payload second derivative"
    (runMixedSecondDerivative graph store ibp derivatives derivatives) 1

  let wrongShape : LayerNormParams Float :=
    { normalizedShape := [1, 2]
      gamma := Tensor.full (α := Float) [1, 2] 2
      beta := Tensor.full (α := Float) [1, 2] 3
      eps := params.eps }
  let invalidStore := { store with layerNorm := store.layerNorm.insert 1 wrongShape }
  unless !crownGraphSemanticsSupported graph invalidStore do
    throw <| IO.userError "mismatched LayerNorm payload shape was accepted"
  let invalidIbp := runIBP graph invalidStore
  assertNoBoundAt "mismatched LayerNorm IBP" invalidIbp 1
  assertNoBoundAt "mismatched LayerNorm affine"
    (runAffine graph invalidStore ctx invalidIbp) 1
  assertNoBoundAt "mismatched LayerNorm CROWN"
    (runCROWN graph invalidStore ctx invalidIbp) 1
  let forgedIbp :=
    #[some (pointFlatBox input), some (FlatBox.ofTensor (Tensor.full (α := Float) [4] 0))]
  match runCROWNBackwardObjective graph invalidStore ctx forgedIbp 1 objective with
  | none => pure ()
  | some _ => throw <| IO.userError "mismatched LayerNorm backward CROWN was accepted"

def run : IO Unit := do
  checkConvolutionGuards
  checkLayerNormPayloadGuard

end NN.Tests.MLTheory.CROWNSoundnessGuardrails

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Models.Mlp
public import NN.Tests.MLTheory.Utils
public import NN.Tests.Utils

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

/-- Coefficient cancellation must not turn a lower logit into a certified winner. -/
def checkAffineCancellation : IO Unit := do
  let graph : NN.IR.Graph := ⟨#[
    { id := 0, parents := #[], kind := .input, outShape := [1] },
    { id := 1, parents := #[0], kind := .linear, outShape := [3] },
    { id := 2, parents := #[1], kind := .linear, outShape := [2] }]⟩
  let w1 : Tensor Float32 [3, 1] := Tensor.full [3, 1] 1
  let b1 : Tensor Float32 [3] := Tensor.full [3] 0
  let w2 : Tensor Float32 [2, 3] := Tensor.matrix fun i j =>
    if i.val = 1 then 0 else
    if j.val = 0 then 16777216 else if j.val = 1 then 1 else -16777216
  let b2 : Tensor Float32 [2] := [0, 0.05]
  let input : Tensor Float32 [1] := [0.1]
  let box : FlatBox Float32 := { dim := 1, lo := input, hi := input }
  let params : ParamStore Float32 :=
    { inputBoxes := ({} : Std.HashMap Nat (FlatBox Float32)).insert 0 box
      linearWB := ({} : Std.HashMap Nat (LinParams Float32)).insert 1
        { m := 3, n := 1, w := w1, b := b1 } |>.insert 2
        { m := 2, n := 3, w := w2, b := b2 } }
  let .ok output := outputBoxCROWN? graph params box 0 2 1
    | throw <| IO.userError "CROWN failed on the cancellation graph"
  -- The stored weights sum to one in real arithmetic; execution rounds its first logit to 0.125.
  unless getAtOrZero output.lo [0] ≤ 0.1 && 0.125 ≤ getAtOrZero output.hi [0] do
    throw <| IO.userError "CROWN lost coefficient-rounding error"
  unless !(getAtOrZero output.lo [1] > getAtOrZero output.hi [0]) do
    throw <| IO.userError "CROWN certified the wrong class after coefficient cancellation"
  let net : TwoLayerMLP Float32 1 3 2 :=
    { hiddenWeight := w1, hiddenBias := b1, outputWeight := w2, outputBias := b2 }
  let mlpBounds := boundAffineCrown net (Box.point input)
  unless mlpBounds.lo.getScalar 0 ≤ 0.1 && 0.125 ≤ mlpBounds.hi.getScalar 0 do
    throw <| IO.userError "standalone MLP CROWN lost coefficient-rounding error"

/-- Negative weights must consume a lower parent bound, even in the upper-only API. -/
def checkUpperAffineSign : IO Unit := do
  let graph : NN.IR.Graph := ⟨#[
    { id := 0, parents := #[], kind := .input, outShape := [1] },
    { id := 1, parents := #[0], kind := .relu, outShape := [1] },
    { id := 2, parents := #[1], kind := .linear, outShape := [1] }]⟩
  let params : ParamStore Float :=
    { inputBoxes := ({} : Std.HashMap Nat (FlatBox Float)).insert 0
        { dim := 1, lo := [-1], hi := [1] }
      linearWB := ({} : Std.HashMap Nat (LinParams Float)).insert 2
        { m := 1, n := 1, w := [[-1]], b := [0] } }
  let affines := runAffine graph params { inputId := 0, inputDim := 1 } (runIBP graph params)
  let some affine := affines[2]!
    | throw <| IO.userError "upper affine bound missing for negative ReLU"
  unless 0 ≤ getAtOrZero affine.aff.c [0] do
    throw <| IO.userError "upper affine bound excludes -ReLU(0) = 0"

def run : IO Unit := do
  checkConvolutionGuards
  checkLayerNormPayloadGuard
  checkAffineCancellation
  checkUpperAffineSign

end NN.Tests.MLTheory.CROWNSoundnessGuardrails

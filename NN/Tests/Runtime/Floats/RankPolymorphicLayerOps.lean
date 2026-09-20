/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Layers.Core
public import NN.Runtime.PyTorch.Export.IRPyTorch
public import NN.Runtime.PyTorch.Import.TorchExport
public import NN.Tests.MLTheory.Utils
public import NN.Tests.Utils

/-!
# Rank-Polymorphic Layer Operations

Focused checks for public layer helpers whose semantics do not depend on tensor rank.
-/

@[expose] public section

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Tests
namespace Floats
namespace RankPolymorphicLayerOps

open Tests.Utils (assertApprox assertBoundAt assertNoBoundAt)
open NN.Tests.MLTheory.Utils (pointFlatBox)

/-- Check grouped, dilated convolution with asymmetric padding. -/
def checkGroupedDilatedConvolution : IO Unit := do
  let inputSpatial : Tensor Nat [1] := [4]
  let kernel : Tensor Nat [1] := [2]
  let stride : Tensor Nat [1] := [1]
  let dilation : Tensor Nat [1] := [2]
  let paddingBefore : Tensor Nat [1] := [1]
  let paddingAfter : Tensor Nat [1] := [0]
  let input : Tensor Float [4, 4] :=
    [[1, 2, 3, 4], [10, 20, 30, 40], [100, 200, 300, 400], [1000, 2000, 3000, 4000]]
  let weights : Tensor Float [4, 4, 2] :=
    Tensor.dim fun outputChannel =>
      Tensor.dim fun inputChannel =>
        Tensor.dim fun _ =>
          Tensor.scalar <| if outputChannel.val = inputChannel.val then 1 else 0
  let bias : Tensor Float [4] := [1, 2, 3, 4]
  have hKernel : ∀ i : Fin 1, kernel.getScalar i ≠ 0 := by
    intro i
    fin_cases i
    simp [kernel]
  have hStride : ∀ i : Fin 1, stride.getScalar i ≠ 0 := by
    intro i
    fin_cases i
    simp [stride]
  let params : NN.IR.ConvParams Float :=
    { spatialRank := 1
      inChannels := 4
      outChannels := 4
      kernel := kernel
      stride := stride
      padding := paddingBefore
      dilation := dilation
      paddingAfter := paddingAfter
      groups := 2
      inputSpatial := inputSpatial
      kernelNonzero := hKernel
      strideNonzero := hStride
      spec := { kernel := by simpa [kernel] using weights, bias := bias } }
  let config : NN.IR.ConvConfig :=
    { spatialRank := 1
      kernel := kernel
      stride := stride
      padding := paddingBefore
      dilation := dilation
      paddingAfter := paddingAfter
      groups := 2
      channelAxis := 0
      inChannels := 4
      outChannels := 4 }
  let graph : NN.IR.Graph :=
    { nodes :=
        #[ { id := 0, parents := #[], kind := .input, outShape := [4, 4] }
         , { id := 1, parents := #[0], kind := .conv config, outShape := [4, 3] } ] }
  let payload : NN.IR.Payload Float :=
    { conv? := fun id => if id = 1 then some params else none }
  let result ←
    match NN.IR.Graph.denote (α := Float) graph payload (Spec.SomeTensor.ofTensor input) 1 with
    | .ok value => pure value
    | .error message => throw <| IO.userError message
  let expected : Array Float := #[3, 5, 7, 22, 42, 62, 203, 403, 603, 2004, 4004, 6004]
  for (value, target) in (Tensor.to result.tensor (Array Float)).zip expected do
    assertApprox "grouped/dilated/asymmetric convolution" value target (tol := 1e-6)
  let verifierParams : NN.MLTheory.CROWN.Graph.ParamStore Float :=
    { inputBoxes := Std.HashMap.emptyWithCapacity.insert 0 (pointFlatBox input)
      convCfg := Std.HashMap.emptyWithCapacity.insert 1 params }
  unless !NN.MLTheory.CROWN.Graph.crownGraphSemanticsSupported graph verifierParams do
    throw <| IO.userError "grouped and dilated convolution was unexpectedly accepted by CROWN"
  let ibp := NN.MLTheory.CROWN.Graph.runIBP graph verifierParams
  assertNoBoundAt "grouped and dilated convolution IBP rejection" ibp 1
  let ctx : NN.MLTheory.CROWN.Graph.AffineCtx := { inputId := 0, inputDim := 16 }
  let affine := NN.MLTheory.CROWN.Graph.runAffine graph verifierParams ctx ibp
  assertNoBoundAt "grouped and dilated convolution affine rejection" affine 1
  let crown := NN.MLTheory.CROWN.Graph.runCROWN graph verifierParams ctx ibp
  assertNoBoundAt "grouped and dilated convolution CROWN rejection" crown 1
  let forgedIbp :=
    #[some (pointFlatBox input), some (pointFlatBox result.tensor)]
  let objective : NN.MLTheory.CROWN.Graph.FlatTensor Float :=
    { n := 12, v := Tensor.full (α := Float) [12] 1 }
  match NN.MLTheory.CROWN.Graph.runCROWNBackwardObjective graph verifierParams ctx forgedIbp 1
      objective with
  | none => pure ()
  | some _ =>
      throw <| IO.userError "grouped and dilated convolution backward CROWN was not rejected"
  let denseParams : NN.IR.ConvParams Float :=
    { params with
      dilation := [1]
      paddingAfter := paddingBefore
      groups := 1 }
  let denseConfig : NN.IR.ConvConfig :=
    { config with
      dilation := [1]
      paddingAfter := paddingBefore
      groups := 1 }
  let denseGraph : NN.IR.Graph :=
    { nodes :=
        #[ { id := 0, parents := #[], kind := .input, outShape := [4, 4] }
         , { id := 1, parents := #[0], kind := .conv denseConfig, outShape := [4, 5] } ] }
  let denseStore : NN.MLTheory.CROWN.Graph.ParamStore Float :=
    { inputBoxes := Std.HashMap.emptyWithCapacity.insert 0 (pointFlatBox input)
      convCfg := Std.HashMap.emptyWithCapacity.insert 1 denseParams }
  unless NN.MLTheory.CROWN.Graph.crownGraphSemanticsSupported denseGraph denseStore do
    throw <| IO.userError "dense unit-dilation symmetric convolution was rejected by CROWN"
  assertBoundAt "supported dense convolution enclosure"
    (NN.MLTheory.CROWN.Graph.runIBP denseGraph denseStore) 1
  let python ←
    match Export.IRPyTorch.emit graph verifierParams 0 1 with
    | .ok code => pure code
    | .error message => throw <| IO.userError message
  unless python.contains "F.pad(_x, (1, 0))" && python.contains "dilation=(2,)" &&
      python.contains "groups=2" do
    throw <| IO.userError
      "grouped and dilated convolution geometry was not preserved by IR-to-PyTorch export"

/--
Evaluate affine LayerNorm with a non-default epsilon and check its value bounds.

Both rows have mean-centered values `[-1, 1]` and variance one. The expected entries therefore
come from `gamma * [-1, 1] / sqrt(1.25) + beta`. IBP must contain the evaluated entries, and the
backward objective must contain their sum.
-/
def checkAffineLayerNormPayload : IO Unit := do
  let shape : Shape := [2, 2]
  let graph : NN.IR.Graph :=
    { nodes :=
        #[ { id := 0, parents := #[], kind := .input, outShape := shape }
         , { id := 1, parents := #[0], kind := .layernorm 1, outShape := shape } ] }
  let params : NN.IR.LayerNormParams Float :=
    { normalizedShape := [2]
      gamma := (Tensor.from #[2.0, 3.0] : Tensor Float [2])
      beta := (Tensor.from #[1.0, -1.0] : Tensor Float [2])
      eps := 0.25 }
  let payload : NN.IR.Payload Float :=
    { layerNorm? := fun id => if id = 1 then some params else none }
  let input : Tensor Float [2, 2] := [[1, 3], [2, 4]]
  let result ←
    match NN.IR.Graph.denote (α := Float) graph payload (Spec.SomeTensor.ofTensor input) 1 with
    | .ok value => pure value
    | .error message => throw <| IO.userError message
  let expected : Array Float := #[-0.7888543819998317, 1.6832815729997477,
    -0.7888543819998317, 1.6832815729997477]
  let values := Tensor.to result.tensor (Array Float)
  for (value, target) in values.zip expected do
    assertApprox "affine LayerNorm payload epsilon" value target (tol := 1e-6)
  let verifierParams : NN.MLTheory.CROWN.Graph.ParamStore Float :=
    { inputBoxes := Std.HashMap.emptyWithCapacity.insert 0 (pointFlatBox input)
      layerNorm := Std.HashMap.emptyWithCapacity.insert 1 params }
  unless NN.MLTheory.CROWN.Graph.crownGraphSemanticsSupported graph verifierParams do
    throw <| IO.userError "affine LayerNorm payload was rejected by CROWN"
  let ibp := NN.MLTheory.CROWN.Graph.runIBP graph verifierParams
  let some bounds := ibp[1]?.join
    | throw <| IO.userError "affine LayerNorm has no IBP bound"
  let lower := Tensor.to bounds.lo (Array Float)
  let upper := Tensor.to bounds.hi (Array Float)
  unless lower.size == values.size && upper.size == values.size do
    throw <| IO.userError "affine LayerNorm IBP bound has the wrong shape"
  for i in [:values.size] do
    unless lower[i]! <= values[i]! && values[i]! <= upper[i]! do
      throw <| IO.userError s!"affine LayerNorm escaped its IBP bound at coordinate {i}"
  let ctx : NN.MLTheory.CROWN.Graph.AffineCtx := { inputId := 0, inputDim := 4 }
  let affine := NN.MLTheory.CROWN.Graph.runAffine graph verifierParams ctx ibp
  assertBoundAt "affine LayerNorm affine bounds" affine 1
  let crown := NN.MLTheory.CROWN.Graph.runCROWN graph verifierParams ctx ibp
  assertBoundAt "affine LayerNorm CROWN" crown 1
  let objective : NN.MLTheory.CROWN.Graph.FlatTensor Float :=
    { n := 4, v := Tensor.full (α := Float) [4] 1 }
  let .ok objectiveBox := NN.MLTheory.CROWN.Graph.backwardObjectiveBox? graph verifierParams
      ctx ibp (pointFlatBox input) 1 objective
    | throw <| IO.userError "affine LayerNorm backward objective was rejected"
  let objectiveLo := Tensor.to objectiveBox.lo (Array Float)
  let objectiveHi := Tensor.to objectiveBox.hi (Array Float)
  let expectedSum := values.foldl (· + ·) 0
  unless objectiveLo.size == 1 && objectiveHi.size == 1 &&
      objectiveLo[0]! <= expectedSum && expectedSum <= objectiveHi[0]! do
    throw <| IO.userError "affine LayerNorm sum escaped its backward objective bound"

/-- Show the non-leading-axis row-major concat ordering and reject it in every CROWN pass. -/
def checkNonLeadingConcatRejected : IO Unit := do
  let shape : Shape := [2, 2]
  let outShape : Shape := [2, 4]
  let graph : NN.IR.Graph :=
    { nodes :=
        #[ { id := 0, parents := #[], kind := .input, outShape := shape }
         , { id := 1, parents := #[], kind := .input, outShape := shape }
         , { id := 2, parents := #[0, 1], kind := .concat 1, outShape := outShape } ] }
  let input : Tensor Float [2, 2] := [[1, 2], [3, 4]]
  let result ←
    match NN.IR.Graph.denote (α := Float) graph {} (Spec.SomeTensor.ofTensor input) 2 with
    | .ok value => pure value
    | .error message => throw <| IO.userError message
  let expected : Array Float := #[1, 2, 1, 2, 3, 4, 3, 4]
  for (value, target) in (Tensor.to result.tensor (Array Float)).zip expected do
    assertApprox "axis-1 concat row-major ordering" value target (tol := 1e-6)
  let inputBox := pointFlatBox input
  let verifierParams : NN.MLTheory.CROWN.Graph.ParamStore Float :=
    { inputBoxes :=
        (Std.HashMap.emptyWithCapacity.insert 0 inputBox).insert 1 inputBox }
  unless !NN.MLTheory.CROWN.Graph.crownGraphSemanticsSupported graph verifierParams do
    throw <| IO.userError "axis-1 concat was unexpectedly accepted by CROWN"
  let ibp := NN.MLTheory.CROWN.Graph.runIBP graph verifierParams
  assertNoBoundAt "axis-1 concat IBP rejection" ibp 2
  let ctx : NN.MLTheory.CROWN.Graph.AffineCtx := { inputId := 0, inputDim := 4 }
  let affine := NN.MLTheory.CROWN.Graph.runAffine graph verifierParams ctx ibp
  assertNoBoundAt "axis-1 concat affine rejection" affine 2
  let crown := NN.MLTheory.CROWN.Graph.runCROWN graph verifierParams ctx ibp
  assertNoBoundAt "axis-1 concat CROWN rejection" crown 2
  let forgedIbp :=
    #[some inputBox, some inputBox, some (pointFlatBox result.tensor)]
  let objective : NN.MLTheory.CROWN.Graph.FlatTensor Float :=
    { n := 8, v := Tensor.full (α := Float) [8] 1 }
  match NN.MLTheory.CROWN.Graph.runCROWNBackwardObjective graph verifierParams ctx forgedIbp 2
      objective with
  | none => pure ()
  | some _ => throw <| IO.userError "axis-1 concat backward CROWN was not rejected"

/-- Parse and execute the affine LayerNorm payload emitted by the PyTorch graph bridge. -/
def checkImportedAffineLayerNorm : IO Unit := do
  let json ←
    match Lean.Json.parse <|
      "{\"format\":\"torchlean.ir.v1\",\"input_id\":0,\"output_ids\":[1]," ++
      "\"nodes\":[{\"id\":0,\"kind\":\"input\",\"parents\":[],\"shape\":[2,2]}," ++
      "{\"id\":1,\"kind\":\"layernorm\",\"parents\":[0],\"shape\":[2,2]," ++
      "\"axis\":1,\"eps\":0.25,\"gamma\":[2.0,3.0],\"beta\":[1.0,-1.0]}]}" with
    | .ok value => pure value
    | .error message => throw <| IO.userError message
  let captured ←
    match Import.PyTorch.TorchExport.parseGraph json with
    | .ok value => pure value
    | .error message => throw <| IO.userError message
  let payload ←
    match Import.PyTorch.TorchExport.parsePayload json with
    | .ok value => pure value
    | .error message => throw <| IO.userError message
  let input : Tensor Float [2, 2] := [[1, 3], [2, 4]]
  let result ←
    match NN.IR.Graph.denote (α := Float) captured.graph payload
        (Spec.SomeTensor.ofTensor input) captured.outputIds[0]! with
    | .ok value => pure value
    | .error message => throw <| IO.userError message
  let expected : Array Float := #[-0.7888543819998317, 1.6832815729997477,
    -0.7888543819998317, 1.6832815729997477]
  for (value, target) in (Tensor.to result.tensor (Array Float)).zip expected do
    assertApprox "imported affine LayerNorm epsilon" value target (tol := 1e-6)

/-- Execute affine eval-mode BatchNorm with an explicit non-default epsilon. -/
def checkAffineBatchNormPayload : IO Unit := do
  let shape : Shape := [1, 2]
  let graph : NN.IR.Graph :=
    { nodes :=
        #[ { id := 0, parents := #[], kind := .input, outShape := shape }
         , { id := 1, parents := #[0], kind := .batchNormEval 1 2, outShape := shape } ] }
  let params : NN.IR.BatchNormEvalParams Float :=
    { c := 2
      gamma := [2, 3]
      beta := [(1 : Float), (-1 : Float)]
      mean := [1, 2]
      var := [3, 2]
      eps := 0.5 }
  let payload : NN.IR.Payload Float :=
    { batchNormEval? := fun id => if id = 1 then some params else none }
  let input : Tensor Float [1, 2] := [[2, 4]]
  let result ←
    match NN.IR.Graph.denote (α := Float) graph payload (Spec.SomeTensor.ofTensor input) 1 with
    | .ok value => pure value
    | .error message => throw <| IO.userError message
  let expected : Array Float := #[2.0690449676496976, 2.794733192202055]
  for (value, target) in (Tensor.to result.tensor (Array Float)).zip expected do
    assertApprox "affine BatchNorm payload epsilon" value target (tol := 1e-6)

/-- Exercise arbitrary-shape running statistics and arbitrary-leading linear derivatives. -/
def run : IO Unit := do
  checkGroupedDilatedConvolution
  checkAffineLayerNormPayload
  checkNonLeadingConcatRejected
  checkImportedAffineLayerNorm
  checkAffineBatchNormPayload
  let running : Tensor Float [2, 2] :=
    Tensor.dim fun i => Tensor.dim fun j => Tensor.scalar ((4 * i.val + 2 * j.val : Nat) : Float)
  let batch : Tensor Float [2, 2] :=
    Tensor.dim fun i =>
      Tensor.dim fun j => Tensor.scalar ((4 * i.val + 2 * j.val + 2 : Nat) : Float)
  let momentum : Tensor Float .scalar := Tensor.scalar 0.25
  let updated :=
    Runtime.Autograd.Model.Layers.updateRunning running batch momentum
  for (actual, expected) in (Tensor.to updated (Array Float)).zip #[0.5, 2.5, 4.5, 6.5] do
    assertApprox "arbitrary-shape running-stat update" actual expected (tol := 1e-6)

  let biased : Tensor Float [2, 2] :=
    Tensor.dim fun i => Tensor.dim fun j => Tensor.scalar ((2 * i.val + j.val + 1 : Nat) : Float)
  let corrected :=
    Runtime.Autograd.Model.Layers.unbiasedRunningVariance biased 4
  for (actual, expected) in (Tensor.to corrected (Array Float)).zip #[4 / 3, 8 / 3, 4, 16 / 3] do
    assertApprox "arbitrary-shape running-variance correction" actual expected (tol := 1e-6)

  let leading : Shape := .dim 2 (.dim 2 .scalar)
  let weights : Tensor Float [1, 2] :=
    Tensor.dim fun _ => Tensor.dim fun j => Tensor.scalar ((j.val + 1 : Nat) : Float)
  let input : Tensor Float (leading.appendDim 2) := by
    simpa [leading, Shape.appendDim_eq_concat] using
      (Tensor.dim fun i => Tensor.dim fun j => Tensor.dim fun k =>
        Tensor.scalar ((4 * i.val + 2 * j.val + k.val + 1 : Nat) : Float) :
        Tensor Float [2, 2, 2])
  let gradOutput : Tensor Float (leading.appendDim 1) := by
    simpa [leading, Shape.appendDim_eq_concat] using
      (Tensor.dim fun _ => Tensor.dim fun _ => Tensor.dim fun _ => Tensor.scalar 1 :
        Tensor Float [2, 2, 1])
  let gradients :=
    Spec.linearDerivSpec (leading := leading) (by decide) weights input gradOutput
  for (actual, expected) in
      (Tensor.to gradients.weightGradient (Array Float)).zip #[16, 20] do
    assertApprox "arbitrary-leading linear weight gradient" actual expected (tol := 1e-6)
  for (actual, expected) in (Tensor.to gradients.biasGradient (Array Float)).zip #[4] do
    assertApprox "arbitrary-leading linear bias gradient" actual expected (tol := 1e-6)
  for (actual, expected) in
      (Tensor.to gradients.inputGradient (Array Float)).zip #[1, 2, 1, 2, 1, 2, 1, 2] do
    assertApprox "arbitrary-leading linear input gradient" actual expected (tol := 1e-6)

  IO.println "rank_polymorphic_layer_ops: ok"

end RankPolymorphicLayerOps
end Floats
end Tests

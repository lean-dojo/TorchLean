/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TypedGraph.GraphM
public import NN.API.Neural.Execution
public import NN.API.Optim
public import NN.Runtime.Autograd.Train
public import NN.Spec.Models.Mlp
public import NN.Tests.Runtime.Floats.Utils
public import NN.Tests.Runtime.TypedGraphScalingRegression
import NN.Kernel
import NN.API.Autograd.Function
import NN.API.Precision

/-!
# Consolidated Float Runtime Autograd Tests

Runtime checks for dynamic-tape gradients, typed-graph output references and derivative rules,
disconnected nonfinite nodes, and native optimizer/cache behavior.
-/

@[expose] public section


/-! ## MLP tape gradients -/

open Spec TorchLean
open TorchLean.Tensor
open Examples

namespace Tests
namespace Floats
namespace AutogradEngine

open Runtime.Autograd

abbrev inDim  := 2
abbrev hidDim := 3
abbrev outDim := 1

abbrev tag : String := "autograd_engine_test"

open Tests.Utils (ParamIds)

/-! ### Fixed inputs and parameters -/
def hiddenWeight : Tensor Float [hidDim, inDim] :=
  (Tensor.from #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).reshape [hidDim, inDim] (by dsimp; decide)

def hiddenBias : Tensor Float [hidDim] :=
  (Tensor.from #[0.1, 0.2, 0.3]).reshape [hidDim] (by dsimp; decide)

def outputWeight : Tensor Float [outDim, hidDim] :=
  (Tensor.from #[0.7, 0.8, 0.9]).reshape [outDim, hidDim] (by dsimp; decide)

def outputBias : Tensor Float [outDim] :=
  (Tensor.from #[0.4]).reshape [outDim] (by dsimp; decide)

def x : Tensor Float [inDim] :=
  (Tensor.from #[0.5, 0.8]).reshape [inDim] (by dsimp; decide)

def dLdy : Tensor Float [outDim] :=
  (Tensor.from #[1.0]).reshape [outDim] (by dsimp; decide)

def hiddenLayer : Spec.LinearSpec Float inDim hidDim :=
  { weights := hiddenWeight, bias := hiddenBias }
def outputLayer : Spec.LinearSpec Float hidDim outDim :=
  { weights := outputWeight, bias := outputBias }

def expected :=
  Examples.mlpBackward hiddenLayer outputLayer x dLdy

/-! ### Comparison with the hand-derived backward pass -/
def checkMlpGrads :
  Runtime.Autograd.Result Bool := do
  let t0 : Tape Float := Tape.empty

  let m : TapeM Float _ := do
    let hiddenWeightId ← Train.TapeM.param hiddenWeight (name := some "hiddenWeight")
    let hiddenBiasId ← Train.TapeM.param hiddenBias (name := some "hiddenBias")
    let outputWeightId ← Train.TapeM.param outputWeight (name := some "outputWeight")
    let outputBiasId ← Train.TapeM.param outputBias (name := some "outputBias")
    let xId ← Train.TapeM.const x (name := some "x")

    let z1Id ← TapeM.linear (inDim:=inDim) (outDim:=hidDim) hiddenWeightId hiddenBiasId xId
    let a1Id ← TapeM.relu (s := [hidDim]) z1Id
    let yId ← TapeM.linear (inDim:=hidDim) (outDim:=outDim) outputWeightId outputBiasId a1Id

    let t ← get
    let grads ← liftM (Tape.backward (t:=t) yId (Spec.SomeTensor.ofTensor dLdy))

    let ids : ParamIds :=
      { hiddenWeightId := hiddenWeightId, hiddenBiasId := hiddenBiasId,
        outputWeightId := outputWeightId, outputBiasId := outputBiasId }
    pure (ids, grads)

  let ((ids, grads), _) ← TapeM.run t0 m

  let (dW1_exp, db1_exp, dW2_exp, db2_exp, _dX_exp) := expected

  let dW1_dyn ← Train.requireGradTensor (tag := tag)
    (s := [hidDim, inDim]) grads ids.hiddenWeightId
  let db1_dyn ← Train.requireGradTensor (tag := tag)
    (s := [hidDim]) grads ids.hiddenBiasId
  let dW2_dyn ← Train.requireGradTensor (tag := tag)
    (s := [outDim, hidDim]) grads ids.outputWeightId
  let db2_dyn ← Train.requireGradTensor (tag := tag)
    (s := [outDim]) grads ids.outputBiasId

  -- Compare the coordinates directly; formatted tensors can hide small gradient errors.
  let close {s : Shape} (actual expected : Tensor Float s) : Bool :=
    let left := actual.to (Array Float)
    let right := expected.to (Array Float)
    left.size == right.size && (left.zip right).all fun (x, y) =>
      x.isFinite && y.isFinite && decide (Float.abs (x - y) ≤ 1e-12)
  pure (close dW1_dyn dW1_exp && close db1_dyn db1_exp &&
    close dW2_dyn dW2_exp && close db2_dyn db2_exp)

def run : IO Unit := do
  match checkMlpGrads with
  | .ok true => IO.println "autograd_engine_test (Float): OK"
  | .ok false => throw <| IO.userError "autograd_engine_test (Float): FAILED"
  | .error msg => throw <| IO.userError s!"autograd_engine_test (Float): {msg}"

end AutogradEngine
end Floats
end Tests


/-!
CPU LayerNorm tape execution, including analytic checks for all three gradients.
-/

open Spec TorchLean
open TorchLean.Tensor

namespace Tests
namespace Floats
namespace AutogradLayerNorm

open Runtime.Autograd

abbrev seqLen := 2
abbrev embedDim := 3

def x : Tensor Float [seqLen, embedDim] :=
  (Tensor.from #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).reshape [seqLen, embedDim] (by dsimp; decide)

def gamma : Tensor Float [embedDim] :=
  (Tensor.from #[1.0, 0.9, 1.1]).reshape [embedDim] (by dsimp; decide)

def beta : Tensor Float [embedDim] :=
  (Tensor.from #[0.0, 0.1, -0.1]).reshape [embedDim] (by dsimp; decide)

def checkLayerNormGrads :
  Runtime.Autograd.Result
    (Float × Tensor Float [seqLen, embedDim] × Tensor Float [embedDim] ×
      Tensor Float [embedDim]) := do
  let t0 : Tape Float := Tape.empty
  let m : TapeM Float _ := do
    let xId ← Train.TapeM.param x (name := some "x")
    let gammaId ← Train.TapeM.param gamma (name := some "gamma")
    let betaId ← Train.TapeM.param beta (name := some "beta")
    let yId ← TapeM.layerNorm (seqLen := seqLen) (embedDim := embedDim) (by decide) (by decide) xId
      gammaId betaId
    let lossId ← TapeM.sum (s := [seqLen, embedDim]) yId
    let t ← get
    let lossVal ← liftM (Train.requireScalarValue (tag := "layer_norm") t lossId)
    let grads ← liftM (Tape.backwardScalar (t := t) lossId)
    pure (xId, gammaId, betaId, lossVal, grads)

  let ((xId, gammaId, betaId, lossVal, grads), _) ← TapeM.run t0 m

  let dX ← Train.requireGradTensor (tag := "layer_norm")
    (s := [seqLen, embedDim]) grads xId
  let dGamma ← Train.requireGradTensor (tag := "layer_norm")
    (s := [embedDim]) grads gammaId
  let dBeta ← Train.requireGradTensor (tag := "layer_norm")
    (s := [embedDim]) grads betaId

  pure (lossVal, dX, dGamma, dBeta)

def run : IO Unit := do
  match checkLayerNormGrads with
  | .error msg => throw <| IO.userError s!"autograd_layernorm_test (Float): {msg}"
  | .ok (loss, dX, dGamma, dBeta) =>
    Tests.Utils.assertFinite "LayerNorm loss" loss
    for value in Tensor.to dX (Array Float) do
      Tests.Utils.assertFinite "LayerNorm input gradient" value
    for value in Tensor.to dGamma (Array Float) do
      Tests.Utils.assertFinite "LayerNorm scale gradient" value
    for value in Tensor.to dBeta (Array Float) do
      Tests.Utils.assertFinite "LayerNorm bias gradient" value
    -- Both rows have centered coordinates [-0.1, 0, 0.1] and variance 1/150.
    -- For the summed output, dx_i = (gamma_i - mean gamma
    --   - centered_i * mean(gamma * centered) / (variance + epsilon)) / stddev.
    let variance : Float := 1 / 150 + TorchLean.normalizationEpsilon
    let stddev := Float.sqrt variance
    let first := (0.1 / 300 / variance) / stddev
    let middle := -0.1 / stddev
    let last := (0.1 - 0.1 / 300 / variance) / stddev
    Utils.assertArrayApprox "LayerNorm input gradient"
      (Tensor.to dX (Array Float)) #[first, middle, last, first, middle, last] 1e-10
    Utils.assertArrayApprox "LayerNorm scale gradient"
      (Tensor.to dGamma (Array Float)) #[-0.2 / stddev, 0, 0.2 / stddev] 1e-10
    Utils.assertArrayApprox "LayerNorm bias gradient" (Tensor.to dBeta (Array Float)) #[2, 2, 2] 0
    Tests.Utils.assertApprox "LayerNorm summed output" loss (0.02 / stddev) 1e-10
    IO.println "autograd_layernorm_test (Float): OK"

end AutogradLayerNorm
end Floats
end Tests

/-!
CPU convolution tape execution with exact kernel/bias gradients for one two-dimensional window.
-/

open Spec TorchLean
open TorchLean.Tensor

namespace Tests
namespace Floats
namespace AutogradConv

open Runtime.Autograd

abbrev inC := 1
abbrev outC := 1
abbrev kH := 2
abbrev kW := 2
abbrev stride := 1
abbrev padding := 0
abbrev inH := 2
abbrev inW := 2

def outH : Nat := Spec.Shape.slidingWindowOutDim inH kH stride padding
def outW : Nat := Spec.Shape.slidingWindowOutDim inW kW stride padding

def kernel : Tensor Float [outC, inC, kH, kW] :=
  (Tensor.from #[0.2, -0.1, 0.3, 0.4]).reshape [outC, inC, kH, kW] (by dsimp; decide)

def bias : Tensor Float [outC] :=
  (Tensor.from #[0.05]).reshape [outC] (by dsimp; decide)

def input : Tensor Float [inC, inH, inW] :=
  (Tensor.from #[1.0, 2.0, 3.0, 4.0]).reshape [inC, inH, inW] (by dsimp; decide)

def checkConvGrads :
  Runtime.Autograd.Result
    (Tensor Float [outC, inC, kH, kW] × Tensor Float [outC]) := do
  let t0 : Tape Float := Tape.empty
  let m : TapeM Float _ := do
    let kId ← Train.TapeM.param kernel (name := some "kernel")
    let bId ← Train.TapeM.param bias (name := some "bias")
    let xId ← Train.TapeM.const input (name := some "input")
    let yId ← TapeM.conv (d := 2) (inC := inC) (outC := outC)
      (kernel := [kH, kW]) (stride := [stride, stride])
      (padding := [padding, padding]) (inSpatial := [inH, inW]) kId bId xId
    let lossId ← TapeM.sum (s := [outC, outH, outW]) yId
    let t ← get
    let grads ← liftM (Tape.backwardScalar (t := t) lossId)
    pure (kId, bId, grads)

  let ((kId, bId, grads), _) ← TapeM.run t0 m
  let dK ← Train.requireGradTensor (tag := "conv")
    (s := [outC, inC, kH, kW]) grads kId
  let dB ← Train.requireGradTensor (tag := "conv")
    (s := [outC]) grads bId
  pure (dK, dB)

def run : IO Unit := do
  match checkConvGrads with
  | .error msg => throw <| IO.userError s!"autograd_conv_test (Float): {msg}"
  | .ok (dK, dB) =>
    for value in Tensor.to dK (Array Float) do
      Tests.Utils.assertFinite "convolution kernel gradient" value
    for value in Tensor.to dB (Array Float) do
      Tests.Utils.assertFinite "convolution bias gradient" value
    -- A single unpadded window gives dK = input and dB = 1 for the summed output.
    Utils.assertArrayApprox "convolution kernel gradient" (Tensor.to dK (Array Float))
      #[1, 2, 3, 4] 0
    Utils.assertArrayApprox "convolution bias gradient" (Tensor.to dB (Array Float)) #[1] 0
    IO.println "autograd_conv_test (Float): OK"

end AutogradConv
end Floats
end Tests

/-! ## Typed graph log-softmax JVP -/

namespace Tests
namespace Floats
namespace TypedGraphLogSoftmaxJvp

open Spec TorchLean
open TorchLean.Tensor

/-- Check that typed graph log-softmax uses its JVP rather than its distinct reverse-mode VJP. -/
def run : IO Unit := do
  let vectorShape : Shape := [2]
  let build :
      Runtime.Autograd.TypedGraph.GraphM.M Float [vectorShape]
        (Runtime.Autograd.TypedGraph.GraphM.Var vectorShape) := do
    let x ← Runtime.Autograd.TypedGraph.GraphM.arg
      (α := Float) (Γ := [vectorShape]) 0 vectorShape
    Runtime.Autograd.TypedGraph.GraphM.logSoftmax 0 x
  let graph ←
    match Runtime.Autograd.Torch.lowerToTypedGraph
        (α := Float) (Γ := [vectorShape]) (τ := vectorShape) build with
    | .ok c => pure c
    | .error e => throw <| IO.userError s!"typed graph log-softmax JVP: lowering failed: {e}"
  let logits : Tensor Float vectorShape :=
    (Tensor.from #[0.0, Float.log 2.0]).reshape [2] (by dsimp; decide)
  let tangent : Tensor Float vectorShape := (Tensor.from #[1.0, 0.0]).reshape [2] (by dsimp; decide)
  let inputs : TorchLean.TensorPack Float [vectorShape] := .cons logits .nil
  let tangents : TorchLean.TensorPack Float [vectorShape] := .cons tangent .nil
  let got := Runtime.Autograd.Torch.TypedGraph.jvp graph inputs tangents
  let got0 := Tensor.getScalar got ⟨0, by decide⟩
  let got1 := Tensor.getScalar got ⟨1, by decide⟩
  unless Float.abs (got0 - 2.0 / 3.0) ≤ 1e-5 &&
      Float.abs (got1 - (-1.0 / 3.0)) ≤ 1e-5 do
    throw <| IO.userError s!"typed graph log-softmax JVP: got {pretty got}, expected [2/3, -1/3]"
  IO.println "typed_graph_log_softmax_jvp_test (Float): OK"

end TypedGraphLogSoftmaxJvp
end Floats
end Tests

/-! ## Typed graph output references -/

namespace Tests
namespace Floats
namespace TypedGraphOutputReference

open Spec TorchLean
open TorchLean.Tensor

/--
Typed graph lowering accepts an input as the output, even when no node is recorded or later nodes
are not selected as the result. Forward, JVP, and VJP must all follow that same output reference.
-/
def run : IO Unit := do
  let identityBuild :
      Runtime.Autograd.TypedGraph.GraphM.M Float [Shape.scalar]
        (Runtime.Autograd.TypedGraph.GraphM.Var Shape.scalar) := do
    Runtime.Autograd.TypedGraph.GraphM.arg
      (α := Float) (Γ := [Shape.scalar]) 0 Shape.scalar
  let identity ←
    match Runtime.Autograd.Torch.lowerToTypedGraph
        (α := Float) (Γ := [Shape.scalar]) (τ := Shape.scalar) identityBuild with
    | .ok graph => pure graph
    | .error e => throw <| IO.userError s!"typed graph identity lowering failed: {e}"
  unless identity.nodeShapes.isEmpty do
    throw <| IO.userError "typed graph identity lowering unexpectedly recorded a node"

  let earlierOutputBuild :
      Runtime.Autograd.TypedGraph.GraphM.M Float [Shape.scalar]
        (Runtime.Autograd.TypedGraph.GraphM.Var Shape.scalar) := do
    let x ← Runtime.Autograd.TypedGraph.GraphM.arg
      (α := Float) (Γ := [Shape.scalar]) 0 Shape.scalar
    let _unused ← Runtime.Autograd.TypedGraph.GraphM.add x x
    pure x
  let earlierOutput ←
    match Runtime.Autograd.Torch.lowerToTypedGraph
        (α := Float) (Γ := [Shape.scalar]) (τ := Shape.scalar) earlierOutputBuild with
    | .ok graph => pure graph
    | .error e => throw <| IO.userError s!"typed graph earlier-output lowering failed: {e}"
  unless earlierOutput.nodeShapes.length == 1 do
    throw <| IO.userError "typed graph earlier-output lowering lost the unused recorded node"

  let inputs : TorchLean.TensorPack Float [Shape.scalar] :=
    .cons (Tensor.scalar 3.0) .nil
  let tangents : TorchLean.TensorPack Float [Shape.scalar] :=
    .cons (Tensor.scalar 2.0) .nil
  let checkGraph (label : String)
      (graph : Runtime.Autograd.Torch.TypedGraph Float [Shape.scalar] Shape.scalar) : IO Unit := do
    let output := Tensor.item (Runtime.Autograd.Torch.TypedGraph.forward graph inputs)
    let tangent := Tensor.item (Runtime.Autograd.Torch.TypedGraph.jvp graph inputs tangents)
    let gradients := Runtime.Autograd.Torch.TypedGraph.vjp
      graph inputs (Tensor.scalar 5.0)
    let gradient := match gradients with
      | .cons grad .nil => Tensor.item grad
    unless output == 3.0 && tangent == 2.0 && gradient == 5.0 do
      throw <| IO.userError
        s!"{label}: got forward={output}, jvp={tangent}, vjp={gradient}; expected 3, 2, 5"
  checkGraph "typed graph identity output" identity
  checkGraph "typed graph earlier output" earlierOutput

  let publicModel : TorchLean.nn.TypedGraphModel [] Shape.scalar Shape.scalar Float := identity
  let noParams : TorchLean.nn.State Float [] := TorchLean.nn.State.empty
  let publicOutput := Tensor.item <|
    TorchLean.nn.TypedGraphModel.forward publicModel noParams (Tensor.scalar 3.0)
  let publicTangent := Tensor.item <|
    TorchLean.nn.TypedGraphModel.jvp publicModel noParams noParams
      (Tensor.scalar 3.0) (Tensor.scalar 2.0)
  let (_, publicInputGradient) :=
    TorchLean.nn.TypedGraphModel.vjp publicModel noParams
      (Tensor.scalar 3.0) (Tensor.scalar 5.0)
  let publicInputGradient := Tensor.item publicInputGradient
  unless publicOutput == 3.0 && publicTangent == 2.0 &&
      publicInputGradient == 5.0 do
    throw <| IO.userError <|
      s!"typed graph public API: got forward={publicOutput}, jvp={publicTangent}, " ++
      s!"vjp={publicInputGradient}; expected 3, 2, 5"
  IO.println "typed_graph_output_reference_test (Float): OK"

end TypedGraphOutputReference
end Floats
end Tests

/-! ## Typed graph smooth-max parameter checks -/

namespace Tests
namespace Floats
namespace TypedGraphSmoothMaxDomain

open Spec TorchLean

/-- Typed `GraphM` rejects an undefined zero inverse temperature while building the graph. -/
def run : IO Unit := do
  let inputShape : Shape := [1, 1, 2]
  let spatial : TorchLean.Tensor Nat [2] := [1, 2]
  let kernel : TorchLean.Tensor Nat [2] := [1, 2]
  let stride : TorchLean.Tensor Nat [2] := [1, 1]
  let padding : TorchLean.Tensor Nat [2] := [0, 0]
  let outputShape : Shape :=
    Shape.ofList
      (1 :: Tensor.to (Spec.poolOutSpatialPad spatial kernel stride padding) (List Nat))
  let build :
      Runtime.Autograd.TypedGraph.GraphM.M Float [inputShape]
        (Runtime.Autograd.TypedGraph.GraphM.Var outputShape) := do
    let x ← Runtime.Autograd.TypedGraph.GraphM.arg
      (α := Float) (Γ := [inputShape]) 0 inputShape
    Runtime.Autograd.TypedGraph.GraphM.smoothMaxPool
      (d := 2) (C := 1) (inSpatial := spatial) (kernel := kernel)
      (stride := stride) (padding := padding) x 0.0
  match Runtime.Autograd.Torch.lowerToTypedGraph
      (α := Float) (Γ := [inputShape]) (τ := outputShape) build with
  | .error _ => IO.println "typed_graph_smooth_max_domain_test (Float): OK"
  | .ok _ => throw <| IO.userError "typed graph smooth-max accepted zero beta"

end TypedGraphSmoothMaxDomain
end Floats
end Tests

/-! ## Dense gradients for disconnected nodes -/

namespace Tests
namespace Floats
namespace DisconnectedDenseGradient

open Spec TorchLean
open TorchLean.Tensor
open Runtime.Autograd

/-- A disconnected reciprocal at zero must not turn an unrelated leaf gradient into `NaN`. -/
def run : IO Unit := do
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) (Tensor.scalar 0.0) (name := some "x")
  let (t2, outId) := Tape.leaf (t := t1) (Tensor.scalar 3.0) (name := some "output")
  let (t3, invId) ← IO.ofExcept <|
    Tape.inv (α := Float) (t := t2) (s := Shape.scalar) xId
  let grads ← IO.ofExcept <|
    Tape.backwardDenseAll (t := t3) outId (Spec.SomeTensor.ofTensor (Tensor.scalar 1.0))
  unless grads.size = t3.nodes.size do
    throw <| IO.userError "disconnected dense gradient: result length mismatch"
  let checkFiniteZero (label : String) (id : Nat) : IO Unit := do
    let grad ← match grads[id]? with
      | some grad => pure grad
      | none => throw <| IO.userError s!"{label}: gradient id out of bounds"
    if h : grad.shape = Shape.scalar then
      let value := Tensor.item (grad.cast h)
      unless value.isFinite && value == 0.0 do
        throw <| IO.userError s!"{label}: expected finite zero, got {value}"
    else
      throw <| IO.userError s!"{label}: expected a scalar gradient"
  checkFiniteZero "disconnected reciprocal input gradient" xId
  checkFiniteZero "disconnected reciprocal output gradient" invId
  IO.println "disconnected_dense_gradient_test (Float): OK"

end DisconnectedDenseGradient
end Floats
end Tests

/-! ## Optimizer and scheduler edge-case regressions -/

namespace Tests
namespace Floats
namespace OptimizerNumerics

open Runtime.Autograd

/-- Finite approximate equality used by the optimizer numerical regressions. -/
def close (x y : Float) (tol : Float := 1e-5) : Bool :=
  x.isFinite && y.isFinite && (x - y).abs ≤ tol

/-- Construct one scalar parameter for a compact optimizer test. -/
def scalarParameter (id : Nat) (value : Float) : Train.Parameter Float :=
  Train.Parameter.create id (Tensor.scalar value)

/-- Construct a one-entry scalar gradient map. -/
def scalarGradient (id : Nat) (value : Float) : Std.HashMap Nat (Spec.SomeTensor Float) :=
  ({} : Std.HashMap Nat (Spec.SomeTensor Float)).insert id
    (Spec.SomeTensor.ofTensor (Tensor.scalar value))

/-- Read a scalar parameter while preserving the runtime's error reporting. -/
def scalarParameterValue (tag : String) (parameters : Train.ParameterTable Float) (id : Nat) :
    Runtime.Autograd.Result Float := do
  let value ← Train.ParameterTable.get (tag := tag) (s := .scalar) parameters id
  pure value.item

/-- Adam bias correction advances only when that particular parameter receives a gradient. -/
def checkSparseAdamSteps : Runtime.Autograd.Result Bool := do
  let initialOptimizerState : Train.OptimizerState Float :=
    { algorithm := .adam
      parameterGroups :=
        #[{ parameterIds := #[0, 1]
            learningRate := 0.1
            beta1 := 0.9
            beta2 := 0.999
            epsilon := 1e-8 }] }
  let initialParameters : Train.ParameterTable Float :=
    #[scalarParameter 0 1.0, scalarParameter 1 1.0]
  let firstStep ← Train.Optimizer.step initialOptimizerState initialParameters
    (scalarGradient 0 1.0)
  let secondStep ← Train.Optimizer.step firstStep.optimizerState firstStep.parameters
    (scalarGradient 1 1.0)
  let firstParameter ← scalarParameterValue "sparse Adam" secondStep.parameters 0
  let secondParameter ← scalarParameterValue "sparse Adam" secondStep.parameters 1
  let restored := Train.OptimizerState.restore secondStep.optimizerState.snapshot
  pure <|
    close firstParameter 0.9 && close secondParameter 0.9 &&
      secondStep.optimizerState.stepCount == 2 &&
      secondStep.optimizerState.parameterStepCount? 0 == some 1 &&
      secondStep.optimizerState.parameterStepCount? 1 == some 1 &&
      restored.parameterStepCount? 0 == some 1 &&
      restored.parameterStepCount? 1 == some 1 &&
      restored.parameterGroups.any (fun group =>
        (group.adamPowers.get? 0).map (·.1) == some 1 &&
          (group.adamPowers.get? 1).map (·.1) == some 1)

/-- Recomputing powers after clearing the cache preserves a changed beta's dual tangent.
This fixture clears the cache explicitly; it does not test automatic invalidation. -/
def checkAdamDualCoefficientChange : Bool :=
  let beta : Model.Dual Float := ⟨0.9, 0.0⟩
  let betaWithTangent : Model.Dual Float := ⟨0.9, 1.0⟩
  let group : Train.ParameterGroup (Model.Dual Float) :=
    { parameterIds := #[0], learningRate := ⟨0.01, 0.0⟩, beta1 := beta }
  let cachedPowers :=
    group.adamPowers.insert 0 ⟨5, Optim.AdamPowers.compute group.beta1 group.beta2 5⟩
  let cached := { group with adamPowers := cachedPowers }
  let changed := { cached with beta1 := betaWithTangent, adamPowers := ∅ }
  let powers := Train.Optimizer.Internal.adamPowers changed 0 5
  let expected := Optim.scalarPowNat betaWithTangent 5
  beta == betaWithTangent &&
    powers.first.re.toBits == expected.re.toBits &&
    powers.first.du.toBits == expected.du.toBits &&
    powers.first.du != 0.0

/-- Compare Adam moments and counters without relying on scalar Boolean equality. -/
def sameAdamBuffers (first second : Train.OptimizerState Float) (id : Nat) : Bool :=
  match first.parameterStates.get? id, second.parameterStates.get? id with
  | none, none => true
  | some ⟨s₁, .adam t₁ m₁ v₁⟩, some ⟨s₂, .adam t₂ m₂ v₂⟩ =>
      if h₁ : s₁ = .scalar then
        if h₂ : s₂ = .scalar then
          t₁ == t₂ &&
            (Tensor.castShape m₁ h₁).item.toBits == (Tensor.castShape m₂ h₂).item.toBits &&
            (Tensor.castShape v₁ h₁).item.toBits == (Tensor.castShape v₂ h₂).item.toBits
        else false
      else false
  | _, _ => false

/--
Cached Adam/AdamW agree bit for bit with reconstructed powers after sparse updates, snapshots,
coefficient changes, and a restored parameter-local counter.
-/
def checkAdamCacheLifecycle (algorithm : Train.OptimizerAlgorithm) :
    Runtime.Autograd.Result Bool := do
  let mut optimizerState : Train.OptimizerState Float :=
    { algorithm := algorithm
      parameterGroups :=
        #[{ parameterIds := #[0, 1]
            learningRate := 0.01
            weightDecay := 0.02
            beta1 := 0.9
            beta2 := 0.999
            epsilon := 1e-8 }] }
  let mut parameters : Train.ParameterTable Float :=
    #[scalarParameter 0 1.0, scalarParameter 1 (-0.5)]
  for step in [:64] do
    if step == 8 then
      let some group := optimizerState.parameterGroups[0]?
        | return false
      let nextGroup := { group with learningRate := 0.005 }
      if group.adamPowers.size == 0 || nextGroup.adamPowers.size != group.adamPowers.size then
        return false
      optimizerState := { optimizerState with parameterGroups := #[nextGroup] }
    if step == 16 then
      optimizerState := Train.OptimizerState.restore optimizerState.snapshot
    if step == 24 then
      optimizerState := { optimizerState with
        parameterGroups := optimizerState.parameterGroups.map fun group =>
          { group with beta1 := 0.8, beta2 := 0.99, adamPowers := ∅ } }
    if step == 32 || step == 40 then
      if let some ⟨shape, .adam _ firstMoment secondMoment⟩ :=
          optimizerState.parameterStates.get? 0 then
        optimizerState := { optimizerState with
          parameterStates := optimizerState.parameterStates.insert 0
            ⟨shape, .adam (if step == 32 then 7 else 53) firstMoment secondMoment⟩ }
    if step == 48 then
      let some group := optimizerState.parameterGroups[0]?
        | return false
      optimizerState := { optimizerState with parameterGroups :=
        #[{ group with parameterIds := #[1] },
          { group with parameterIds := #[0], beta1 := 0.7, adamPowers := ∅ }] }
    let gradient := scalarGradient (step % 2) (if step % 3 == 0 then -0.25 else 0.5)
    let referenceState := { optimizerState with
      parameterGroups := optimizerState.parameterGroups.map fun group =>
        { group with adamPowers := ∅ } }
    let reference ← Train.Optimizer.step referenceState parameters gradient
    let cached ← Train.Optimizer.step optimizerState parameters gradient
    let updatedId := step % 2
    let some group := cached.optimizerState.parameterGroups.find?
        (fun group => group.parameterIds.contains updatedId)
      | return false
    let some ⟨cachedStep, _⟩ := group.adamPowers.get? updatedId
      | return false
    if cached.optimizerState.parameterStepCount? updatedId != some cachedStep then
      return false
    for id in [:2] do
      let expected ← scalarParameterValue "Adam reconstructed powers" reference.parameters id
      let actual ← scalarParameterValue "Adam cached powers" cached.parameters id
      if actual.toBits != expected.toBits ||
          !sameAdamBuffers cached.optimizerState reference.optimizerState id then
        return false
    optimizerState := cached.optimizerState
    parameters := cached.parameters
  pure true

/-- Momentum dampening does not scale the first buffer, matching the standard SGD convention. -/
def checkMomentumInitialization : Runtime.Autograd.Result Bool := do
  let initialOptimizerState : Train.OptimizerState Float :=
    { algorithm := .momentum
      parameterGroups :=
        #[{ parameterIds := #[0]
            learningRate := 0.1
            momentum := 0.9
            dampening := 0.5 }] }
  let initialParameters : Train.ParameterTable Float := #[scalarParameter 0 1.0]
  let firstStep ← Train.Optimizer.step initialOptimizerState initialParameters
    (scalarGradient 0 2.0)
  let first ← scalarParameterValue "momentum initialization" firstStep.parameters 0
  let secondStep ← Train.Optimizer.step firstStep.optimizerState firstStep.parameters
    (scalarGradient 0 2.0)
  let second ← scalarParameterValue "momentum initialization" secondStep.parameters 0
  pure (close first 0.8 && close second 0.52)

/-- Adadelta's update accumulator stores the unscaled update, independently of the learning rate. -/
def checkAdadeltaAccumulator : Bool :=
  let parameters : Tensor Float .scalar := Tensor.scalar 10.0
  let gradients : Tensor Float .scalar := Tensor.scalar 2.0
  let state := Optim.Adadelta.init 0.5 0.0 1.0 parameters
  let result := Optim.Adadelta.update state parameters gradients
  close result.optimizerState.squaredUpdateAverage.item 0.8 &&
    close result.parameters.item (10.0 - 1.0 / Float.sqrt 5.0)

open TorchLean.Tensor in
/-- Adadelta's fused RMS terms match the reference `sqrt (average + epsilon)` expression. -/
def checkAdadeltaFastPath : Bool :=
  let parameters : Tensor Float [3] := [1.0, -2.0, 0.5]
  let gradients : Tensor Float [3] := [0.3, -0.7, 0.0]
  let initial := Optim.Adadelta.init 1.0 0.9 1e-6 parameters
  let first := Optim.Adadelta.update initial parameters gradients
  let state := first.optimizerState
  let result := Optim.Adadelta.update state first.parameters gradients
  let epsilon := Tensor.full [3] state.epsilon
  let nextAverage :=
    addSpec (scaleSpec state.squaredGradientAverage state.rho)
      (scaleSpec (squareSpec gradients) (1 - state.rho))
  let ratio :=
    divSpec (sqrtSpec (addSpec state.squaredUpdateAverage epsilon))
      (sqrtSpec (addSpec nextAverage epsilon))
  let reference :=
    subSpec first.parameters
      (scaleSpec (mulSpec ratio gradients) state.learningRate)
  (List.finRange 3).all fun i =>
    (result.parameters.getScalar i).toBits == (reference.getScalar i).toBits

/-- Warmup-cosine decay remains at zero after its finite schedule has ended. -/
def checkWarmupCosineStops : Bool :=
  let base : Optim.Scheduler.WarmupCosine Float :=
    Optim.Scheduler.WarmupCosine.create 1.0 2 10
  let atEnd := { base with currentStep := 10 }
  let afterEnd := { base with currentStep := 20 }
  atEnd.current == 0.0 && afterEnd.current == 0.0

/-- Native one-cycle ends at `initial_lr / final_div_factor`, PyTorch's `OneCycleLR` endpoint. -/
def checkOneCycleEndpoints : Bool :=
  let base : Optim.Scheduler.OneCycle Float :=
    Optim.Scheduler.OneCycle.create 1.0 10 25.0 0.3 1.0e4
  let finished := { base with currentStep := 10 }
  base.current == 1.0 / 25.0 && finished.current == (1.0 / 25.0) / 1.0e4

/-- Public optimizer configurations reject domains that make their updates undefined. -/
def checkPublicOptimizerValidation : Bool :=
  (TorchLean.optim.adam { learningRate := 1e-3 }).validate.isOk &&
    (!(TorchLean.optim.adam { learningRate := 1e-3, beta1 := 1.0 }).validate.isOk) &&
    (!(TorchLean.optim.adamW { learningRate := 1e-3, weightDecay := -0.1 }).validate.isOk) &&
    (!(TorchLean.optim.rmsProp { learningRate := 1e-3, epsilon := 0.0 }).validate.isOk) &&
    (!(TorchLean.optim.sgd
      { learningRate := 0.1, momentum := Float.ofBits 0x7ff8000000000000 }).validate.isOk)

/-- Run the optimizer and scheduler edge-case regressions. -/
def run : IO Unit := do
  match checkSparseAdamSteps with
  | .error msg => throw <| IO.userError s!"optimizer numerics (sparse Adam): {msg}"
  | .ok false => throw <| IO.userError "optimizer numerics (sparse Adam): FAILED"
  | .ok true => pure ()
  unless checkAdamDualCoefficientChange do
    throw <| IO.userError "optimizer numerics (Adam dual coefficient change): FAILED"
  for algorithm in [Train.OptimizerAlgorithm.adam, .adamw] do
    match checkAdamCacheLifecycle algorithm with
    | .error msg => throw <| IO.userError s!"optimizer numerics (Adam power cache): {msg}"
    | .ok false => throw <| IO.userError "optimizer numerics (Adam power cache): FAILED"
    | .ok true => pure ()
  match checkMomentumInitialization with
  | .error msg => throw <| IO.userError s!"optimizer numerics (momentum): {msg}"
  | .ok false => throw <| IO.userError "optimizer numerics (momentum): FAILED"
  | .ok true => pure ()
  unless checkAdadeltaAccumulator do
    throw <| IO.userError "optimizer numerics (Adadelta accumulator): FAILED"
  unless checkAdadeltaFastPath do
    throw <| IO.userError "optimizer numerics (Adadelta fast path): FAILED"
  unless checkWarmupCosineStops do
    throw <| IO.userError "optimizer numerics (warmup cosine): FAILED"
  unless checkOneCycleEndpoints do
    throw <| IO.userError "optimizer numerics (one-cycle endpoints): FAILED"
  unless checkPublicOptimizerValidation do
    throw <| IO.userError "optimizer configuration validation: FAILED"
  IO.println "optimizer and scheduler edge cases (Float): OK"

end OptimizerNumerics
end Floats
end Tests

namespace Tests
namespace Floats

/-- A shared scalar intermediate must stay connected to the existing derivative programs. -/
private def customPolynomial {shape : Shape} : autograd.Function shape shape :=
  fun x => do
    let y ← Runtime.mul x x
    Runtime.add (← Runtime.mul y y) (← Runtime.const (Tensor.full shape 1))

private def polynomial {α : Type} [Mul α] [Add α] [One α] (x : α) : α :=
  let y := x * x
  y * y + 1

private def energy (x : Tensor Float [2]) : Float := Tensor.sum (x * x)

private def weights : Tensor Float [2, 1] := [[2], [3]]

private def project (x : Tensor Float [1, 2]) : Tensor Float [1, 1] := x.matmul weights

private def customObjective {shape : Shape} : autograd.Function shape [] := fun x => do
  TorchLean.Runtime.sum (← customPolynomial x)

private abbrev Wide := FloatLib.Floats.ExecFloat.Binary 15 112

/-- Frontend wiring is checked separately from the primitive derivative theorems: shared uses,
reverse/forward mode, nested differentiation and wide scalar storage all use the same program. -/
private def checkCustomAutograd : IO Unit := do
  let input : Tensor Float [2] := Tensor.ofFn fun i => Float.ofNat (i.val + 1)
  let gradient ← autograd.grad customObjective input
  let forward ← autograd.jacfwd customPolynomial input
  let reverse ← autograd.jacrev customPolynomial input
  let hessian ← autograd.hessian customObjective input
  let recorded ← polynomial.run input (grad := true)
  let recordedGradient ← recorded.backward
  unless recorded.value[0] == (2 : Float) && recorded.value[1] == (17 : Float) &&
      recordedGradient[0] == gradient[0] && recordedGradient[1] == gradient[1] do
    throw <| IO.userError "custom calculation: ordinary function disagrees with recorded operations"
  let consumed ← try
      discard <| recorded.backward
      pure false
    catch _ => pure true
  unless consumed do throw <| IO.userError "custom calculation: consumed recording was reused"
  let closed ← polynomial.run input (grad := true)
  closed.close
  closed.close
  let released ← try
      discard <| closed.backward
      pure false
    catch _ => pure true
  unless released do throw <| IO.userError "custom calculation: closed recording was reused"
  let loss ← energy.run input (grad := true)
  let lossGradient ← loss.backward
  unless loss.value.item == (5 : Float) &&
      lossGradient[0] == 2 && lossGradient[1] == 4 do
    throw <| IO.userError "custom calculation: whole-tensor reduction lost its derivative"
  let projected ← project.run ([[1, 2]] : Tensor Float [1, 2]) (grad := true)
  let projectedGradient ← projected.backward (Tensor.full [1, 1] 2)
  unless projected.value[0][0] == 8 &&
      projectedGradient[0][0] == 4 && projectedGradient[0][1] == 6 do
    throw <| IO.userError "custom calculation: matrix product or explicit cotangent disagrees"
  unless gradient[0] == 4 && gradient[1] == 32 &&
      forward[0][0] == 4 && forward[1][1] == 32 &&
      forward[0][1] == 0 && forward[1][0] == 0 &&
      forward[0][0] == reverse[0][0] && forward[1][1] == reverse[1][1] &&
      reverse[0][1] == 0 && reverse[1][0] == 0 &&
      hessian[0][0] == 12 && hessian[1][1] == 48 &&
      hessian[0][1] == 0 && hessian[1][0] == 0 do
    throw <| IO.userError "custom calculation: recorded derivative programs disagree"
  let wide : Tensor Wide [1] := Tensor.full [1] 2
  let wideGradient ← autograd.grad customObjective wide
  unless FloatLib.Floats.ExecFloat.Binary.toRat? wideGradient[0] == some 32 do
    throw <| IO.userError "custom calculation: binary128 gradient changed precision"
  let small : Rat := 1 / (2 ^ 100 : Nat)
  let precise : Tensor Wide [1] :=
    Tensor.full [1] (Rat.cast (1 + small))
  let preciseGradient ← autograd.grad customObjective precise
  unless FloatLib.Floats.ExecFloat.Binary.toRat? preciseGradient[0] == some (4 + 12 * small) do
    throw <| IO.userError "custom calculation: gradient lost digits beyond binary64"
  -- CPU recording also retains digits that would disappear in a binary32 transfer.
  let native : Tensor Float [1] := Tensor.full [1] (Float.ofBits 0x3ff0000000001000)
  let nativeResult ← (fun (x : Float) => x * x).run native (grad := true)
  let nativeGradient ← nativeResult.backward
  unless nativeResult.device == cpu &&
      nativeResult.value[0].toBits == (native[0] * native[0]).toBits &&
      nativeGradient[0].toBits == (2 * native[0]).toBits do
    throw <| IO.userError "custom calculation: binary64 recording narrowed to binary32"
  -- Ordinary CPU recording retains the same wide digits as the typed derivative above.
  -- Configured GPU recording is checked separately in the CUDA suite.
  let recordedWide ← polynomial.run precise (grad := true)
  try
    unless recordedWide.device == cpu do
      throw <| IO.userError "custom calculation: CPU recording selected a different device"
    let gradient ← recordedWide.backward
    unless FloatLib.Floats.ExecFloat.Binary.toRat? gradient[0] == some (4 + 12 * small) do
      throw <| IO.userError "custom calculation: CPU recording narrowed its gradient"
  finally
    recordedWide.close

@[no_expose] def runAllAutogradTests : IO Unit := do
  IO.println "=== Runtime autograd test suite (Float) ==="
  AutogradEngine.run
  AutogradLayerNorm.run
  AutogradConv.run
  TypedGraphLogSoftmaxJvp.run
  TypedGraphOutputReference.run
  TypedGraphSmoothMaxDomain.run
  DisconnectedDenseGradient.run
  TypedGraphScalingRegression.run
  OptimizerNumerics.run
  checkCustomAutograd
  IO.println "=== Autograd test suite completed ==="

end Floats
end Tests

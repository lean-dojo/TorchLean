/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Lean.Data.Json
public import NN.API
public import NN.Core.ExternalProcess
public import NN.Runtime.Autograd.Model.Norm
public import NN.Runtime.RL.Core
public import NN.Spec.Generative.Diffusion.PFODE
public import NN.Spec.Layers.Loss
public import NN.Spec.Models.Gmm
public import NN.Spec.Models.Hmm
public import NN.Tests.Runtime.Floats.Utils
public import Std
public import NN.IR.OpContracts
public import NN.Spec.Layers.Dropout

/-!
# TorchLeanOpsCheck

Runtime checks for TorchLean operator wrappers over the float runtime.

The file is intentionally fixture-driven: each helper builds one small tensor example, runs it
through the relevant execution path or external backend boundary, and compares the result against
another TorchLean path, a closed form, or PyTorch when it is available.
-/

@[expose] public section

open Lean
open Spec TorchLean
open TorchLean TorchLean.Tensor
open Tests.Utils
open Tests.Floats.Utils

namespace Tests
namespace Floats
namespace TorchLeanOpsCheck

/-! ## Shared fixtures -/

/-- BatchNorm parity fixture: batch size. -/
abbrev bnN : Nat := 2
/-- BatchNorm parity fixture: channel count. -/
abbrev bnC : Nat := 2
/-- BatchNorm parity fixture: image height. -/
abbrev bnH : Nat := 2
/-- BatchNorm parity fixture: image width. -/
abbrev bnW : Nat := 2
/-- NCHW shape used by the BatchNorm runtime and PyTorch parity checks. -/
abbrev bnShape : Shape := [bnN, bnC, bnH, bnW]

/-- Scratch directory for small Python parity scripts emitted by this test module. -/
def workDir : System.FilePath :=
  TorchLean.External.Process.artifactWorkDir "ops_check"

/-- Path for the generated BatchNorm parity script. -/
def batchNormParityScriptPath : System.FilePath :=
  workDir / "batchnorm_parity.py"

/-- Evaluate softmax along a statically checked tensor dimension in the eager runtime. -/
def evalSoftmaxAxis {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (x : Tensor Float s) : IO (Tensor Float s) := do
  let sess ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float)
  let action : Runtime.Autograd.Torch.Internal.EagerM Float (Tensor Float s) := do
    let xRef ← Runtime.Autograd.Torch.Ops.const
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float) (s := s) x
    let yRef ← Runtime.Autograd.Model.F.softmax
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float) (s := s) axis xRef
    let sess ← read
    liftM <| Runtime.Autograd.Torch.Internal.EagerSession.getValue
      (α := Float) (sh := s) sess yRef
  action sess

/-- Evaluate softmax and its VJP along a statically checked tensor dimension. -/
def evalSoftmaxAxisWithGradient {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (x upstream : Tensor Float s) : IO (Tensor Float s × Tensor Float s) := do
  let sess ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float)
  let xRef ← Runtime.Autograd.Torch.Internal.EagerSession.input
    (α := Float) (sh := s) sess x (name := some "softmax_input") (requiresGrad := true)
  let action : Runtime.Autograd.Torch.Internal.EagerM Float
      (Runtime.Autograd.Torch.TensorRef Float s) :=
    Runtime.Autograd.Model.F.softmax
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float) (s := s) axis xRef
  let yRef ← action sess
  let y ← Runtime.Autograd.Torch.Internal.EagerSession.getValue
    (α := Float) (sh := s) sess yRef
  let grads ← Runtime.Autograd.Torch.Internal.EagerSession.backwardDenseAll
    (α := Float) (sh := s) sess yRef upstream
  let dx ← Runtime.Autograd.Torch.Internal.EagerSession.grad grads xRef
  pure (y, dx)

/-- Evaluate log-softmax and its VJP along a statically checked tensor dimension. -/
def evalLogSoftmaxAxisWithGradient {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (x upstream : Tensor Float s) : IO (Tensor Float s × Tensor Float s) := do
  let sess ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float)
  let xRef ← Runtime.Autograd.Torch.Internal.EagerSession.input
    (α := Float) (sh := s) sess x (name := some "log_softmax_input") (requiresGrad := true)
  let action : Runtime.Autograd.Torch.Internal.EagerM Float
      (Runtime.Autograd.Torch.TensorRef Float s) :=
    Runtime.Autograd.Model.F.logSoftmax
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float) (s := s) axis xRef
  let yRef ← action sess
  let y ← Runtime.Autograd.Torch.Internal.EagerSession.getValue
    (α := Float) (sh := s) sess yRef
  let grads ← Runtime.Autograd.Torch.Internal.EagerSession.backwardDenseAll
    (α := Float) (sh := s) sess yRef upstream
  let dx ← Runtime.Autograd.Torch.Internal.EagerSession.grad grads xRef
  pure (y, dx)

/-- Check outer, final, and interior softmax dimensions against explicit numerical fixtures. -/
def checkSoftmaxDimensions : IO Unit := do
  let matrix : Tensor Float [2, 3] :=
    (Tensor.from (#[1, 2, 3, 4, 5, 6] : Array Float)).reshape [2, 3] (by dsimp; decide)
  let alongRows ← evalSoftmaxAxis 0 matrix
  let alongColumns ← evalSoftmaxAxis 1 matrix
  let rowExpected := #[0.047425873, 0.047425873, 0.047425873,
    0.952574127, 0.952574127, 0.952574127]
  let columnExpected := #[0.090030573, 0.244728471, 0.665240956,
    0.090030573, 0.244728471, 0.665240956]
  for (actual, expected) in (Tensor.to alongRows (Array Float)).zip rowExpected do
    assertApprox "softmax axis 0" actual expected 1e-5
  for (actual, expected) in (Tensor.to alongColumns (Array Float)).zip columnExpected do
    assertApprox "softmax axis 1" actual expected 1e-5

  let cube : Tensor Float [2, 2, 2] :=
    (Tensor.from (#[0, 2, 1, 4, 3, 8, 7, 9] : Array Float)).reshape [2, 2, 2] (by dsimp; decide)
  let alongMiddle ← evalSoftmaxAxis 1 cube
  let middleExpected := #[0.268941421, 0.119202922, 0.731058579, 0.880797078,
    0.017986210, 0.268941421, 0.982013790, 0.731058579]
  for (actual, expected) in (Tensor.to alongMiddle (Array Float)).zip middleExpected do
    assertApprox "softmax interior axis" actual expected 1e-5

  let upstream : Tensor Float [2, 2, 2] :=
    (Tensor.from (#[1, -2, 3, 4, -1, 2, 5, -3] : Array Float)).reshape
      [2, 2, 2] (by dsimp; decide)
  let (_, gradient) ← evalSoftmaxAxisWithGradient 1 cube upstream
  let expectedGradient :=
    Activation.softmaxBackwardSpec (α := Float) (s := [2, 2, 2])
      1 cube upstream
  for (actual, expected) in
      (Tensor.to gradient (Array Float)).zip (Tensor.to expectedGradient (Array Float)) do
    assertApprox "softmax interior-axis gradient" actual expected 1e-5

  let (logProbabilities, logGradient) ←
    evalLogSoftmaxAxisWithGradient 1 cube upstream
  let expectedLogProbabilities :=
    Activation.logSoftmaxSpec (α := Float) (s := [2, 2, 2])
      1 cube
  let expectedLogGradient :=
    Activation.logSoftmaxBackwardSpec
      (α := Float) (s := [2, 2, 2])
      1 expectedLogProbabilities upstream
  for (actual, expected) in
      (Tensor.to logProbabilities (Array Float)).zip
        (Tensor.to expectedLogProbabilities (Array Float)) do
    assertApprox "log-softmax interior axis" actual expected 1e-5
  for (actual, expected) in
      (Tensor.to logGradient (Array Float)).zip (Tensor.to expectedLogGradient (Array Float)) do
    assertApprox "log-softmax interior-axis gradient" actual expected 1e-5

/-- Check that classification metrics support outer and inner class axes. -/
def checkClassificationAxes : IO Unit := do
  let logits : Tensor Float [2, 3] :=
    (Tensor.from (#[1, 5, 2, 9, 4, 3] : Array Float)).reshape [2, 3] (by dsimp; decide)
  unless TorchLean.Metrics.argmaxAxis? 1 logits = #[some 1, some 0] do
    throw <| IO.userError "argmax along the inner class axis returned incorrect indices"
  unless TorchLean.Metrics.argmaxAxis? 0 logits = #[some 1, some 0, some 1] do
    throw <| IO.userError "argmax along the outer class axis returned incorrect indices"

  let rowTargets : Tensor Float [2, 3] :=
    (Tensor.from (#[0, 1, 0, 1, 0, 0] : Array Float)).reshape [2, 3] (by dsimp; decide)
  unless TorchLean.Metrics.accuracyOneHotAxis 1 logits rowTargets =
      ({ correct := 2, total := 2 } : TorchLean.Metrics.Accuracy) do
    throw <| IO.userError "one-hot accuracy along the inner class axis was incorrect"

  let columnTargets : Tensor Float [2, 3] :=
    (Tensor.from (#[0, 1, 0, 1, 0, 1] : Array Float)).reshape [2, 3] (by dsimp; decide)
  unless TorchLean.Metrics.accuracyOneHotAxis 0 logits columnTargets =
      ({ correct := 3, total := 3 } : TorchLean.Metrics.Accuracy) do
    throw <| IO.userError "one-hot accuracy along the outer class axis was incorrect"

/-- Evaluate weighted MSE through the eager runtime. -/
def evalWeightedMse
    (prediction target weights : Tensor Float [4]) : IO Float := do
  let sess ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float)
  let action : Runtime.Autograd.Torch.Internal.EagerM Float (Tensor Float .scalar) := do
    let predictionRef ← Runtime.Autograd.Torch.Ops.const
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float)
      (s := [4]) prediction
    let targetRef ← Runtime.Autograd.Torch.Ops.const
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float)
      (s := [4]) target
    let weightsRef ← Runtime.Autograd.Torch.Ops.const
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float)
      (s := [4]) weights
    let lossRef ← TorchLean.Loss.mseWeighted
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float)
      (s := [4]) predictionRef targetRef weightsRef
    let sess ← read
    liftM <| Runtime.Autograd.Torch.Internal.EagerSession.getValue
      (α := Float) (sh := .scalar) sess lossRef
  pure <| Tensor.item (← action sess)

/-- Check masking and normalization semantics for weighted MSE. -/
def checkWeightedMse : IO Unit := do
  let target : Tensor Float [4] := [0, 0, 0, 0]
  let weights : Tensor Float [4] := [0.5, 0, 0.5, 0]
  let first : Tensor Float [4] := [1, 100, 3, 200]
  let changedExcluded : Tensor Float [4] := [1, -500, 3, 900]
  let firstLoss ← evalWeightedMse first target weights
  let changedLoss ← evalWeightedMse changedExcluded target weights
  assertApprox "weighted MSE excludes zero-weight coordinates" firstLoss changedLoss 1e-6
  assertApprox "normalized weighted MSE" firstLoss 5 1e-6

/-- Evaluate a two-row weighted integer-label cross entropy through the eager runtime. -/
def evalWeightedRowCrossEntropy
    (logits : Tensor Float [2, 2])
    (weights : Tensor Float [2]) : IO Float := do
  let targets : Tensor (Fin 2) [2] :=
    Tensor.ofFn fun i => if i = 0 then ⟨0, by decide⟩ else ⟨1, by decide⟩
  let sess ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float)
  let action : Runtime.Autograd.Torch.Internal.EagerM Float (Tensor Float .scalar) := do
    let logitsRef ← Runtime.Autograd.Torch.Ops.const
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float)
      (s := [2, 2]) logits
    let weightsRef ← Runtime.Autograd.Torch.Ops.const
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float)
      (s := [2]) weights
    let logitsIndexed : Runtime.Autograd.Model.RefTy
        (Runtime.Autograd.Torch.Internal.EagerM Float) Float
        (Shape.concat [2] [2]) := by simpa using logitsRef
    let targetsIndexed : Tensor (Fin 2) (Shape.concat [2] Shape.scalar) := by
      simpa using targets
    let weightsIndexed : Runtime.Autograd.Model.RefTy
        (Runtime.Autograd.Torch.Internal.EagerM Float) Float
        (Shape.concat [2] Shape.scalar) := by simpa using weightsRef
    let lossRef ← TorchLean.Loss.crossEntropyWeighted
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float)
      1 rfl logitsIndexed targetsIndexed weightsIndexed
    let sess ← read
    liftM <| Runtime.Autograd.Torch.Internal.EagerSession.getValue
      (α := Float) (sh := .scalar) sess lossRef
  pure <| Tensor.item (← action sess)

/--
Check weighted cross entropy as a runtime operation.

The first assertion changes only a zero-weight row and therefore must leave the loss unchanged. The
second checks linearity in the explicit row weights. Together they catch accidental mean reduction,
mask misalignment, and a weights tensor that is ignored by the runtime.
-/
def checkWeightedRowCrossEntropy : IO Unit := do
  let logitsA : Tensor Float [2, 2] :=
    (Tensor.from (#[0, 0, 20, -20] : Array Float)).reshape [2, 2] (by dsimp; decide)
  let logitsB : Tensor Float [2, 2] :=
    (Tensor.from (#[0, 0, -20, 20] : Array Float)).reshape [2, 2] (by dsimp; decide)
  let firstOnly : Tensor Float [2] :=
    (Tensor.from (#[1, 0] : Array Float)).reshape [2] (by dsimp; decide)
  let secondOnly : Tensor Float [2] :=
    (Tensor.from (#[0, 1] : Array Float)).reshape [2] (by dsimp; decide)
  let mixture : Tensor Float [2] := (Tensor.from #[0.25, 0.75]).reshape [2] (by dsimp; decide)
  let firstA ← evalWeightedRowCrossEntropy logitsA firstOnly
  let firstB ← evalWeightedRowCrossEntropy logitsB firstOnly
  assertApprox "zero-weight row is excluded" firstA firstB 1e-5
  let second ← evalWeightedRowCrossEntropy logitsA secondOnly
  let mixed ← evalWeightedRowCrossEntropy logitsA mixture
  assertApprox "weighted row loss is linear in row weights"
    mixed (0.25 * firstA + 0.75 * second) 1e-4

/-- Check indexed cross entropy with both leading and trailing coordinates around the class axis. -/
def checkMiddleAxisCrossEntropy : IO Unit := do
  let logits : Tensor Float [2, 2, 2] :=
    (Tensor.from (#[10, -10, -10, 10, -10, 10, 10, -10] : Array Float)).reshape
      [2, 2, 2] (by dsimp; decide)
  let targets : Tensor (Fin 2) [2, 2] :=
    TorchLean.Tensor.stack 0 fun outer => Tensor.ofFn fun inner =>
      if outer.val = inner.val then ⟨0, by decide⟩ else ⟨1, by decide⟩
  let sess ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float)
  let action : Runtime.Autograd.Torch.Internal.EagerM Float (Tensor Float .scalar) := do
    let logitsRef ← Runtime.Autograd.Torch.Ops.const
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float)
      (s := [2, 2, 2]) logits
    let logitsIndexed : Runtime.Autograd.Model.RefTy
        (Runtime.Autograd.Torch.Internal.EagerM Float) Float
        (Shape.concat [2] ([2, 2] : Shape)) := by simpa using logitsRef
    let targetsIndexed : Tensor (Fin 2) (Shape.concat [2] [2]) := by
      simpa using targets
    let lossRef ← TorchLean.Loss.crossEntropy
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float)
      1 rfl logitsIndexed targetsIndexed
    let sess ← read
    liftM <| Runtime.Autograd.Torch.Internal.EagerSession.getValue
      (α := Float) (sh := .scalar) sess lossRef
  assertApprox "cross entropy preserves a suffix after the class axis"
    (Tensor.item (← action sess)) 0 1e-5

/-- Check that tied token lookup removes exactly one independent affine vocabulary head. -/
def checkTiedTokenEmbeddingParameterCount : IO Unit := do
  let countElements (shapes : List Shape) : Nat :=
    shapes.foldl (fun count shape => count + Spec.Shape.size shape) 0
  let config : TorchLean.nn.models.CausalTransformer.Config :=
    { sequenceLength := 2
      vocabularySize := 7
      headCount := 1
      headWidth := 4
      feedForwardWidth := 8
      layerCount := 1 }
  let untiedModel := TorchLean.nn.build 11 <|
    TorchLean.nn.models.CausalTransformer.indexed config [1]
  let tiedModel := TorchLean.nn.build 11 <|
    TorchLean.nn.models.CausalTransformer.tied config [1]
  let untiedCount := countElements <|
    untiedModel.stateShapes
  let tiedCount := countElements <|
    tiedModel.stateShapes
  let independentHeadCount :=
    config.vocabularySize * config.modelWidth + config.vocabularySize
  unless untiedCount = tiedCount + independentHeadCount do
    throw <| IO.userError <|
      s!"tied token embedding: untied={untiedCount}, tied={tiedCount}, " ++
        s!"expected difference={independentHeadCount}"

/-- Run a tied-token model through one loss and backward pass. -/
def checkTiedTokenEmbeddingBackward : IO Unit := do
  let config : TorchLean.nn.models.CausalTransformer.Config :=
    { sequenceLength := 2
      vocabularySize := 7
      headCount := 1
      headWidth := 4
      feedForwardWidth := 8
      layerCount := 1 }
  let model := TorchLean.nn.build 19 <|
    TorchLean.nn.models.CausalTransformer.tied config [1]
  let definition := TorchLean.nn.models.CausalTransformer.objective config model
  let module ← TorchLean.Module.instantiate
    definition { execution := .eager } (α := Float)
  let inputs : Tensor (Fin config.vocabularySize) [1, 2] :=
    TorchLean.Tensor.stack 0 fun _ => Tensor.ofFn fun i =>
      if i.val = 0 then ⟨0, by decide⟩ else ⟨1, by decide⟩
  let targets : Tensor (Fin config.vocabularySize) [1, 2] :=
    TorchLean.Tensor.stack 0 fun _ => Tensor.ofFn fun i =>
      if i.val = 0 then ⟨1, by decide⟩ else ⟨2, by decide⟩
  let (gradient, loss) ←
    TorchLean.Module.Objective.grad module
      TorchLean.Arguments.empty
      (TorchLean.Arguments.empty |>.push inputs |>.push targets)
      (value := true)
  assertFinite "tied token embedding loss" (Tensor.item loss)
  let firstState : Fin model.stateShapes.length ←
    if h : 0 < model.stateShapes.length then
      pure ⟨0, h⟩
    else
      throw <| IO.userError "tied token model has no shared embedding state"
  let sharedGradient := gradient.get firstState
  let values := Tensor.to sharedGradient (Array Float)
  for value in values do
    assertFinite "tied token embedding gradient" value
  unless values.any (fun value => Float.abs value > 1.0e-8) do
    throw <| IO.userError
      "tied token embedding produced a zero gradient for its shared lookup/projection matrix"

/--
Check that self-attention can initialize its residual output projection independently of Q/K/V.

Deep Transformers commonly use a smaller initializer for the projection written back to the
residual stream. Distinct constant schemes make this a direct wiring check: Q/K/V must contain
ones, while the output projection must contain zeros.
-/
def checkAttentionOutputProjectionInitializer : IO Unit := do
  let layer :=
    Runtime.Autograd.Model.Layers.multiHeadAttention
      1 1 2 1 2 (sequenceLengthNonzero := by decide)
      (weightInitialization? := some .ones)
      (outputWeightInitialization? := some .zeros)
  let (wq, wk, wv, wo) :=
    match layer.initState with
    | .cons wq (.cons wk (.cons wv (.cons wo .nil))) => (wq, wk, wv, wo)
  for i in List.finRange 2 do
    for j in List.finRange 2 do
      assertApprox s!"attention Q initializer[{i.val},{j.val}]" (matVal wq i j) 1 1e-7
      assertApprox s!"attention K initializer[{i.val},{j.val}]" (matVal wk i j) 1 1e-7
      assertApprox s!"attention V initializer[{i.val},{j.val}]" (matVal wv i j) 1 1e-7
      assertApprox s!"attention output initializer[{i.val},{j.val}]" (matVal wo i j) 0 1e-7

/-- Typed graph execution preserves leaf gradient flags and leaves frozen parameters unchanged. -/
def checkTypedGraphLeafMetadata : IO Unit := do
  let sess ← Runtime.Autograd.Model.Session.new (α := Float)
    (options := { execution := .typedGraph })
  let frozen ← Runtime.Autograd.Model.Session.param sess
    (Tensor.scalar 2.0) (name := some "frozen") (requiresGrad := some false)
  let trainable ← Runtime.Autograd.Model.Session.param sess
    (Tensor.scalar 3.0) (name := some "trainable") (requiresGrad := some true)
  let frozenRef ← Runtime.Autograd.Model.Session.use sess frozen
  let trainableRef ← Runtime.Autograd.Model.Session.use sess trainable
  let sum ← Runtime.Autograd.Model.Session.add sess frozenRef trainableRef
  let loss ← Runtime.Autograd.Model.Session.mul sess sum sum
  let grads ← Runtime.Autograd.Model.Session.backwardScalarDenseAll sess loss
  let frozenGrad ← Runtime.Autograd.Model.Session.grad sess grads frozenRef
  let trainableGrad ← Runtime.Autograd.Model.Session.grad sess grads trainableRef
  assertApprox "typed graph frozen gradient" (Tensor.item frozenGrad) 0 1e-7
  assertApprox "typed graph trainable gradient" (Tensor.item trainableGrad) 10 1e-7
  Runtime.Autograd.Model.Session.sgdStepAll sess 0.1 grads
  assertApprox "typed graph frozen parameter" (Tensor.item (← frozen.value.get)) 2 1e-7
  assertApprox "typed graph updated parameter" (Tensor.item (← trainable.value.get)) 2 1e-7

/-! ## Eager/typed graph operator parity -/

/-- Evaluate public session softmax and log-softmax along a three-axis tensor's middle axis. -/
def evalSessionSoftmaxFixture (execution : Runtime.Autograd.Torch.ExecutionMode) :
    IO (Tensor Float [2, 2, 2] ×
      Tensor Float [2, 2, 2]) := do
  let shape : Shape := [2, 2, 2]
  let input : Tensor Float shape :=
    (Tensor.from (#[0, 2, 1, 4, 3, 8, 7, 9] : Array Float)).reshape [2, 2, 2] (by dsimp; decide)
  let sess ← Runtime.Autograd.Model.Session.new (α := Float)
    (options := { execution := execution })
  let inputRef ← Runtime.Autograd.Model.Session.const sess (sh := shape) input
  let probabilitiesRef ←
    Runtime.Autograd.Model.Session.softmax sess (sh := shape) 1 inputRef
  let logProbabilitiesRef ←
    Runtime.Autograd.Model.Session.logSoftmax sess (sh := shape) 1 inputRef
  let probabilities ←
    Runtime.Autograd.Model.Session.getValue sess (sh := shape) probabilitiesRef
  let logProbabilities ←
    Runtime.Autograd.Model.Session.getValue sess (sh := shape) logProbabilitiesRef
  pure (probabilities, logProbabilities)

/-- Evaluate a fixed 2x3 by 3x2 matrix product in the selected execution mode. -/
def evalMatmulFixture (execution : Runtime.Autograd.Torch.ExecutionMode) :
    IO (Tensor Float [2, 2]) := do
  let sess ← Runtime.Autograd.Model.Session.new (α := Float)
    (options := { execution := execution })
  let a : Tensor Float [2, 3] :=
    Tensor.generate [2, 3] fun coordinate =>
      Float.ofNat (coordinate.getD 0 0 + 2 * coordinate.getD 1 0 + 1)
  let b : Tensor Float [3, 2] :=
    Tensor.generate [3, 2] fun coordinate =>
      Float.ofNat (3 * coordinate.getD 0 0 + coordinate.getD 1 0 + 1)
  let aR ← Runtime.Autograd.Model.Session.const sess (sh := [2, 3]) a
  let bR ← Runtime.Autograd.Model.Session.const sess (sh := [3, 2]) b
  let cR ← Runtime.Autograd.Model.Session.matmul sess
    (batchA := Shape.scalar) (batchB := Shape.scalar) (batch := Shape.scalar)
    (m := 2) (n := 3) (p := 2) aR bR
  Runtime.Autograd.Model.Session.getValue sess (sh := [2, 2]) cR

/-- Evaluate a fixed length-2 plus length-3 vector concatenation in the selected execution mode. -/
def evalConcatFixture (execution : Runtime.Autograd.Torch.ExecutionMode) :
    IO (Tensor Float [5]) := do
  let sess ← Runtime.Autograd.Model.Session.new (α := Float)
    (options := { execution := execution })
  let a : Tensor Float [2] := Tensor.ofFn fun i => Float.ofNat (i.val + 1)
  let b : Tensor Float [3] := Tensor.ofFn fun i => 10.0 + Float.ofNat i.val
  let aR ← Runtime.Autograd.Model.Session.const sess (sh := [2]) a
  let bR ← Runtime.Autograd.Model.Session.const sess (sh := [3]) b
  let cR ← Runtime.Autograd.Model.Session.concatLeadingAxis sess
    (n := 2) (m := 3) (sh := .scalar) aR bR
  Runtime.Autograd.Model.Session.getValue sess (sh := [5]) cR

/-- Evaluate a two-spatial-axis max-pooling example through the generic pooling API. -/
def evalMaxPoolFixture (execution : Runtime.Autograd.Torch.ExecutionMode) :
    IO (Tensor Float [1, 2, 2]) := do
  let sess ← Runtime.Autograd.Model.Session.new (α := Float)
    (options := { execution := execution })
  let spatial : TorchLean.Tensor Nat [2] := [4, 4]
  let kernel : TorchLean.Tensor Nat [2] := [2, 2]
  let stride : TorchLean.Tensor Nat [2] := [2, 2]
  let padding : TorchLean.Tensor Nat [2] := [0, 0]
  let x : Tensor Float [1, 4, 4] :=
    Tensor.generate [1, 4, 4] fun coordinate =>
      Float.ofNat (coordinate.getD 1 0 * 10 + coordinate.getD 2 0)
  have hInputShape : Shape.ofList (1 :: Tensor.to spatial (List Nat)) = [1, 4, 4] := by
    simp [spatial]
  let x' : Tensor Float (Shape.ofList (1 :: Tensor.to spatial (List Nat))) :=
    hInputShape.symm ▸ x
  let xR ← Runtime.Autograd.Model.Session.const sess x'
  let yR ← Runtime.Autograd.Model.Session.maxPool sess
    (d := 2) (channels := 1) (spatial := spatial)
    (kernel := kernel) (stride := stride) (padding := padding) xR
  let y ← Runtime.Autograd.Model.Session.getValue sess yR
  have hOutputSpatial :
      Spec.poolOutSpatialPad spatial kernel stride padding =
        ([2, 2] : TorchLean.Tensor Nat [2]) := by
    apply TorchLean.Tensor.ext_vector
    intro i
    fin_cases i <;>
      norm_num [spatial, kernel, stride, padding, Spec.poolOutSpatialPad, Spec.poolOutDim,
        Shape.slidingWindowOutDim]
  have hShape :
      Shape.ofList
          (1 :: Tensor.to (Spec.poolOutSpatialPad spatial kernel stride padding) (List Nat)) =
        [1, 2, 2] := by
    rw [hOutputSpatial]
    rfl
  return Tensor.castShape y hShape

/-- Evaluate a two-spatial-axis average-pooling example through the generic pooling API. -/
def evalAvgPoolFixture (execution : Runtime.Autograd.Torch.ExecutionMode) :
    IO (Tensor Float [1, 2, 2]) := do
  let sess ← Runtime.Autograd.Model.Session.new (α := Float)
    (options := { execution := execution })
  let spatial : TorchLean.Tensor Nat [2] := [4, 4]
  let kernel : TorchLean.Tensor Nat [2] := [2, 2]
  let stride : TorchLean.Tensor Nat [2] := [2, 2]
  let padding : TorchLean.Tensor Nat [2] := [0, 0]
  let x : Tensor Float [1, 4, 4] :=
    Tensor.generate [1, 4, 4] fun coordinate =>
      Float.ofNat (coordinate.getD 1 0 * 10 + coordinate.getD 2 0)
  have hInputShape : Shape.ofList (1 :: Tensor.to spatial (List Nat)) = [1, 4, 4] := by
    simp [spatial]
  let x' : Tensor Float (Shape.ofList (1 :: Tensor.to spatial (List Nat))) :=
    hInputShape.symm ▸ x
  let xR ← Runtime.Autograd.Model.Session.const sess x'
  let yR ← Runtime.Autograd.Model.Session.avgPool sess
    (d := 2) (channels := 1) (spatial := spatial)
    (kernel := kernel) (stride := stride) (padding := padding) xR
  let y ← Runtime.Autograd.Model.Session.getValue sess yR
  have hOutputSpatial :
      Spec.poolOutSpatialPad spatial kernel stride padding =
        ([2, 2] : TorchLean.Tensor Nat [2]) := by
    apply TorchLean.Tensor.ext_vector
    intro i
    fin_cases i <;>
      norm_num [spatial, kernel, stride, padding, Spec.poolOutSpatialPad, Spec.poolOutDim,
        Shape.slidingWindowOutDim]
  have hShape :
      Shape.ofList
          (1 :: Tensor.to (Spec.poolOutSpatialPad spatial kernel stride padding) (List Nat)) =
        [1, 2, 2] := by
    rw [hOutputSpatial]
    rfl
  return Tensor.castShape y hShape

/-! ## BatchNorm fixture and PyTorch parity -/

/-- NCHW input with different signs per channel, chosen so mean/variance are easy to inspect. -/
def bnInput : Tensor Float bnShape :=
  Tensor.generate [bnN, bnC, bnH, bnW] fun coordinate =>
    let n := coordinate.getD 0 0
    let c := coordinate.getD 1 0
    let h := coordinate.getD 2 0
    let w := coordinate.getD 3 0
    let base := Float.ofNat (n * 8 + c * 4 + h * 2 + w + 1)
    if c = 0 then base else -base

/-- BatchNorm scale parameter for the two channels. -/
def bnGamma : Tensor Float [bnC] :=
  [1.0, 0.5]

/-- BatchNorm shift parameter for the two channels. -/
def bnBeta : Tensor Float [bnC] :=
  [0.0, 0.1]

/-- Running mean used by the eval-mode BatchNorm fixture. -/
def bnMean : Tensor Float [bnC] :=
  [2.0, -3.0]

/-- Running variance used by the eval-mode BatchNorm fixture. -/
def bnVar : Tensor Float [bnC] :=
  [4.0, 9.0]

/--
Run training-mode BatchNorm and return the output together with the computed channel statistics.

This checks the TorchLean runtime wrapper directly, not only the exported graph path.
-/
def evalBatchNormTrain :
    IO (Tensor Float bnShape × Tensor Float [bnC] × Tensor Float [bnC]) :=
    do
  let sess ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float)
  let action : Runtime.Autograd.Torch.Internal.EagerM Float
      (Tensor Float bnShape × Tensor Float [bnC] × Tensor Float [bnC]) :=
    do
      let xR ← Runtime.Autograd.Torch.Ops.const
        (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float) (s := bnShape) bnInput
      let gR ← Runtime.Autograd.Torch.Ops.const
        (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float) (s := [bnC])
        bnGamma
      let bR ← Runtime.Autograd.Torch.Ops.const
        (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float) (s := [bnC])
        bnBeta
      let (yR, meanR, varR) ← Runtime.Autograd.Model.Norm.batchNormTrainStats
        (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float)
        (batch := bnN) (channels := bnC) (spatial := [bnH, bnW])
        (by decide) xR gR bR
      let sess ← read
      let y ← liftM <| Runtime.Autograd.Torch.Internal.EagerSession.getValue
        (α := Float) (sh := bnShape) sess yR
      let mean ← liftM <| Runtime.Autograd.Torch.Internal.EagerSession.getValue
        (α := Float) (sh := [bnC]) sess meanR
      let var ← liftM <| Runtime.Autograd.Torch.Internal.EagerSession.getValue
        (α := Float) (sh := [bnC]) sess varR
      pure (y, mean, var)
  action sess

/-- Run eval-mode BatchNorm using fixed running statistics. -/
def evalBatchNormEval :
    IO (Tensor Float bnShape) := do
  let sess ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float)
  let action : Runtime.Autograd.Torch.Internal.EagerM Float (Tensor Float bnShape) := do
    let xR ← Runtime.Autograd.Torch.Ops.const
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float) (s := bnShape) bnInput
    let gR ← Runtime.Autograd.Torch.Ops.const
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float) (s := [bnC])
      bnGamma
    let bR ← Runtime.Autograd.Torch.Ops.const
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float) (s := [bnC])
      bnBeta
    let mR ← Runtime.Autograd.Torch.Ops.const
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float) (s := [bnC])
      bnMean
    let vR ← Runtime.Autograd.Torch.Ops.const
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float) (s := [bnC])
      bnVar
    let yR ← Runtime.Autograd.Model.Norm.batchNormEval
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float)
      (batch := bnN) (channels := bnC) (spatial := [bnH, bnW])
      (by decide) xR gR bR mR vR
    let sess ← read
    liftM <| Runtime.Autograd.Torch.Internal.EagerSession.getValue
      (α := Float) (sh := bnShape) sess yR
  action sess

/--
Closed-form BatchNorm expression.

Training and evaluation share this affine formula and differ only in which statistics reach it:
batch mean/variance in training mode, the fixed running statistics in evaluation mode. Both call
sites below pass their own pair, so one helper is enough.
-/
def expectedBatchNormAffine (x gamma beta mean var : Float) : Float :=
  ((x - mean) / Float.sqrt (var + TorchLean.normalizationEpsilon)) * gamma + beta

/-- Flatten a tensor in TorchLean's canonical row-major order for parity comparisons. -/
def flattenRowMajor {s : Shape} (t : Tensor Float s) : Array Float :=
  Tensor.to t (Array Float)

/-- Small Python program used to compare TorchLean BatchNorm against PyTorch. -/
def batchNormParityScript : String :=
  String.intercalate "\n"
    [ "import json"
    , "import torch"
    , "import torch.nn.functional as F"
    , ""
    , "x = torch.tensor(["
    , "    [[[1., 2.], [3., 4.]], [[-5., -6.], [-7., -8.]]],"
    , "    [[[9., 10.], [11., 12.]], [[-13., -14.], [-15., -16.]]],"
    , "], dtype=torch.float32)"
    , "gamma = torch.tensor([1.0, 0.5], dtype=torch.float32)"
    , "beta = torch.tensor([0.0, 0.1], dtype=torch.float32)"
    , "running_mean = torch.tensor([2.0, -3.0], dtype=torch.float32)"
    , "running_var = torch.tensor([4.0, 9.0], dtype=torch.float32)"
    , "eps = 1e-5"
    , "mean = x.mean(dim=(0, 2, 3))"
    , "var = x.var(dim=(0, 2, 3), unbiased=False)"
    , "train = F.batch_norm(x, None, None, gamma, beta, training=True, eps=eps)"
    , "eval = F.batch_norm(x, running_mean, running_var, gamma, beta, training=False, eps=eps)"
    , "print(json.dumps({"
    , "    'mean': mean.flatten().tolist(),"
    , "    'var': var.flatten().tolist(),"
    , "    'train': train.flatten().tolist(),"
    , "    'eval': eval.flatten().tolist(),"
    , "}))"
    ]

/--
Compare TorchLean BatchNorm output and statistics against PyTorch, when PyTorch is installed.

The fallback skip is deliberate: the Lean-side closed-form checks still run on machines without a
Python/PyTorch environment.
-/
def checkBatchNormAgainstPyTorch
    (trainY evalY : Tensor Float bnShape)
    (mean var : Tensor Float [bnC]) : IO Unit := do
  if !(← pythonHasTorch) then
    IO.println "torchlean_ops_check: PyTorch BatchNorm parity skipped (`torch` not installed)"
    return ()
  IO.FS.createDirAll workDir
  IO.FS.writeFile batchNormParityScriptPath batchNormParityScript
  let out ← TorchLean.External.Process.runStdoutChecked
    (ctx := "torchlean_ops_check: batchnorm pytorch parity")
    (cmd := "python3")
    (args := #[batchNormParityScriptPath.toString])
    (cwd := some ".")
  let pyJson ←
    match Json.parse out with
    | .ok j => pure j
    | .error e =>
        throw (IO.userError s!"torchlean_ops_check: bad BatchNorm parity JSON: {e}\n{out}")
  let readField key := do
    match jsonFloatArrayField pyJson key with
    | .ok xs => pure xs
    | .error e => throw (IO.userError s!"torchlean_ops_check: {e}")
  assertArrayApprox "batchnorm_nchw pytorch mean"
    #[vecVal mean ⟨0, by decide⟩, vecVal mean ⟨1, by decide⟩] (← readField "mean")
  assertArrayApprox "batchnorm_nchw pytorch var"
    #[vecVal var ⟨0, by decide⟩, vecVal var ⟨1, by decide⟩] (← readField "var")
  assertArrayApprox "batchnorm pytorch train" (flattenRowMajor trainY) (← readField "train")
  assertArrayApprox "batchnorm pytorch eval" (flattenRowMajor evalY) (← readField "eval")

/--
Run the full BatchNorm check: closed-form expectations first, then optional PyTorch parity.
-/
def checkBatchNorm : IO Unit := do
  let (trainY, mean, var) ← evalBatchNormTrain
  let evalY ← evalBatchNormEval

  assertApprox "batchnorm_nchw mean[0] expected" (vecVal mean ⟨0, by decide⟩) 6.5
  assertApprox "batchnorm_nchw mean[1] expected" (vecVal mean ⟨1, by decide⟩) (-10.5)
  assertApprox "batchnorm_nchw var[0] expected" (vecVal var ⟨0, by decide⟩) 17.25
  assertApprox "batchnorm_nchw var[1] expected" (vecVal var ⟨1, by decide⟩) 17.25

  for n in List.finRange bnN do
    for c in List.finRange bnC do
      for h in List.finRange bnH do
        for w in List.finRange bnW do
          let x := tensorVal bnInput [n, c, h, w]
          let gamma := vecVal bnGamma c
          let beta := vecVal bnBeta c
          -- Training normalizes with the statistics of this very batch, ...
          let trainExpected :=
            expectedBatchNormAffine x gamma beta (vecVal mean c) (vecVal var c)
          -- ... while evaluation normalizes with the stored running statistics.
          let evalExpected :=
            expectedBatchNormAffine x gamma beta (vecVal bnMean c) (vecVal bnVar c)
          assertApprox s!"batchnorm_nchw train[{n.val},{c.val},{h.val},{w.val}] expected"
            (tensorVal trainY [n, c, h, w]) trainExpected 1e-5
          assertApprox s!"batchnorm_nchw eval[{n.val},{c.val},{h.val},{w.val}] expected"
            (tensorVal evalY [n, c, h, w]) evalExpected 1e-5

  checkBatchNormAgainstPyTorch trainY evalY mean var

/-! ## Loss and attention specification regressions -/

/-- Check the cosine gradient away from clipping kinks against the forward loss. -/
def checkCosineDerivative : IO Unit := do
  let aligned : Tensor Float [1] := [2]
  let alignedGradient := Spec.cosineSimilarityDerivSpec aligned aligned
  assertApprox "cosine aligned nonunit gradient" (vecVal alignedGradient ⟨0, by decide⟩) 0
  for scale in (#[0.5, 2, 5] : Array Float) do
    for targetValues in (#[#[1.0, -2.0], #[0.01, 0.02]] : Array (Array Float)) do
      let values : Array Float := #[3 * scale, 4 * scale]
      let predicted : Tensor Float [2] := [values[0]!, values[1]!]
      let target : Tensor Float [2] := [targetValues[0]!, targetValues[1]!]
      let gradient := Spec.cosineSimilarityDerivSpec predicted target (epsilon := 0.1)
      let step : Float := 1.0e-5
      for i in [0:2] do
        let plus := values.modify i (· + step)
        let minus := values.modify i (· - step)
        let plusTensor : Tensor Float [2] := [plus[0]!, plus[1]!]
        let minusTensor : Tensor Float [2] := [minus[0]!, minus[1]!]
        let difference :=
          (Spec.cosineSimilaritySpec plusTensor target (epsilon := 0.1) -
            Spec.cosineSimilaritySpec minusTensor target (epsilon := 0.1)) / (2 * step)
        assertApprox s!"cosine finite difference scale={scale} component={i}"
          (Tensor.to gradient (Array Float))[i]! difference 1e-6

/-- Check reduction axes and the selected derivatives at clipped loss branches. -/
def checkLossSemantics : IO Unit := do
  checkCosineDerivative
  let logits : Tensor Float [2, 3] :=
    (Tensor.from (#[0, 0, 0, 0, 0, 0] : Array Float)).reshape [2, 3] (by dsimp; decide)
  let target : Tensor Float [2, 3] :=
    (Tensor.from (#[1, 0, 0, 0, 1, 0] : Array Float)).reshape [2, 3] (by dsimp; decide)
  let logitsLoss := Spec.crossEntropyLogitsSpec 1 logits target
  assertApprox "cross entropy averages samples, not classes" logitsLoss (Float.log 3) 1e-5

  let probabilities : Tensor Float [3] :=
    (Tensor.from #[0.05, 0.5, 0.95]).reshape [3] (by dsimp; decide)
  let distribution : Tensor Float [3] :=
    (Tensor.from (#[1, 1, 1] : Array Float)).reshape [3] (by dsimp; decide)
  let probabilityGrad :=
    Spec.crossEntropyDerivSpec 0 probabilities distribution (epsilon := 0.1)
  assertApprox "cross entropy lower clipped branch" (vecVal probabilityGrad ⟨0, by decide⟩) 0
  assertApprox "cross entropy interior branch" (vecVal probabilityGrad ⟨1, by decide⟩) (-2)
  assertApprox "cross entropy upper clipped branch" (vecVal probabilityGrad ⟨2, by decide⟩) 0

  assertApprox "BCE lower clipped branch"
    (Spec.binaryCrossEntropyDerivSpec 0.05 1 (epsilon := 0.1)) 0
  assertApprox "BCE interior branch"
    (Spec.binaryCrossEntropyDerivSpec 0.5 1 (epsilon := 0.1)) (-2)
  assertApprox "BCE upper clipped branch"
    (Spec.binaryCrossEntropyDerivSpec 0.95 1 (epsilon := 0.1)) 0

  let huberPrediction : Tensor Float [2] :=
    (Tensor.from (#[1, 3] : Array Float)).reshape [2] (by dsimp; decide)
  let huberTarget : Tensor Float [2] :=
    (Tensor.from (#[0, 0] : Array Float)).reshape [2] (by dsimp; decide)
  assertApprox "Huber delta=2 forward"
    (Spec.huberSpec huberPrediction huberTarget (delta := 2)) 2.25
  let huberGrad := Spec.huberDerivSpec huberPrediction huberTarget (delta := 2)
  assertApprox "Huber delta=2 quadratic gradient"
    (vecVal huberGrad ⟨0, by decide⟩) 0.5
  assertApprox "Huber delta=2 linear gradient"
    (vecVal huberGrad ⟨1, by decide⟩) 1
  assertApprox "RL Huber uses the same delta convention"
    (Runtime.RL.Core.huberLoss (α := Float) 3 0 2) 4

  let shortPrediction : Tensor Float [2] := (Tensor.from #[0.05, 0]).reshape [2] (by dsimp; decide)
  let unitTarget : Tensor Float [2] :=
    (Tensor.from (#[1, 0] : Array Float)).reshape [2] (by dsimp; decide)
  let cosineGrad :=
    Spec.cosineSimilarityDerivSpec shortPrediction unitTarget (epsilon := 0.1)
  assertApprox "cosine epsilon branch[0]" (vecVal cosineGrad ⟨0, by decide⟩) (-10)
  assertApprox "cosine epsilon branch[1]" (vecVal cosineGrad ⟨1, by decide⟩) 0

  assertApprox "zero-feature attention scale"
    (Spec.attentionScaleDenom (α := Float) 0) 1

def checkCorrectedMathematicalSpecs : IO Unit := do
  let matrix : Tensor Float [2, 2] :=
    (Tensor.from (#[0, 2, 4, 6] : Array Float)).reshape [2, 2] (by dsimp; decide)
  assertApprox "tensor-wide population variance"
    (TorchLean.Tensor.varianceSpec matrix) 5

  -- Each GroupNorm group contains one channel and both spatial positions. This fixture checks that
  -- the generic spatial suffix is reduced independently for each channel group.
  let groupNormInput : Tensor Float [1, 2, 1, 2] :=
    (Tensor.from (#[1, 10, 3, 14] : Array Float)).reshape [1, 2, 1, 2] (by dsimp; decide)
  let groupNormScale : Tensor Float [2] :=
    (Tensor.from (#[1, 1] : Array Float)).reshape [2] (by dsimp; decide)
  let groupNormBias : Tensor Float [2] :=
    (Tensor.from (#[0, 0] : Array Float)).reshape [2] (by dsimp; decide)
  let groupNormOutput := Spec.groupNorm (groups := 2)
    groupNormInput groupNormScale groupNormBias (hGroupsLe := by decide) (hDiv := by decide)
  assertArrayApprox "GroupNorm channel grouping" (Tensor.to groupNormOutput (Array Float))
    #[-0.9999998, 0.9999998, -0.99999994, 0.99999994] 1e-5

  let clustered : Tensor Float [2] :=
    (Tensor.from #[1.0e12, 1.0e12 + 1]).reshape [2] (by dsimp; decide)
  let clusteredVariance := TorchLean.Tensor.reduceVar 0 clustered Spec.Shape.NonemptyAxis.zero
  assertApprox "centered population variance"
    (scalarVal clusteredVariance) 0.25 1e-8

  assertApprox "softplus large positive input"
    (Activation.Math.softplusSpec (1000 : Float)) 1000 1e-10

  let emptySpatial : TorchLean.Tensor Nat [1] := [0]
  let unitKernel : TorchLean.Tensor Nat [1] := [1]
  let unitStride : TorchLean.Tensor Nat [1] := [1]
  let zeroPadding : TorchLean.Tensor Nat [1] := [0]
  unless (Spec.poolOutSpatialPad emptySpatial unitKernel unitStride zeroPadding).getScalar 0 = 0 do
    throw <| IO.userError "pooling emitted a fully padded window for an empty input axis"
  let emptyPoolShape ← match NN.IR.OpContracts.inferPoolOutShape "max_pool"
      unitKernel unitStride zeroPadding (Shape.ofList [1, 0]) with
    | .ok shape => pure shape
    | .error message => throw <| IO.userError message
  unless emptyPoolShape = Shape.ofList [1, 0] do
    throw <| IO.userError "IR pooling rejected the specification's empty-axis semantics"
  let oversizedKernel : TorchLean.Tensor Nat [1] := [3]
  let oversizedPoolShape ← match NN.IR.OpContracts.inferPoolOutShape "max_pool"
      oversizedKernel unitStride zeroPadding (Shape.ofList [1, 1]) with
    | .ok shape => pure shape
    | .error message => throw <| IO.userError message
  unless oversizedPoolShape = Shape.ofList [1, 0] do
    throw <| IO.userError "IR pooling rejected the specification's oversized-window semantics"

  let schedule : Generative.Diffusion.VPLinearSchedule Float :=
    { beta0 := 1, beta1 := 2 }
  let epsModel : Generative.Diffusion.EpsModel Float [1] :=
    { eps := fun _ _ => (Tensor.from (#[1] : Array Float)).reshape [1] (by dsimp; decide) }
  let state : Tensor Float [1] := (Tensor.from (#[2] : Array Float)).reshape [1] (by dsimp; decide)
  let t : Float := 0.75
  let rhs := Generative.Diffusion.pfOdeRhs schedule epsModel state t
  let beta := schedule.beta t
  let sigma := schedule.sigma t
  let expectedRhs :=
    (-1 / 2) * beta * 2 +
      (1 / 2) * Generative.Diffusion.safeDiv beta sigma
  assertApprox "probability-flow ODE coefficient"
    (vecVal rhs ⟨0, by decide⟩) expectedRhs

  let dt : Float := -0.5
  let afterT1 := Generative.Diffusion.eulerStep
    (Generative.Diffusion.pfOdeRhs schedule epsModel) state 1 dt
  let expectedSample := Generative.Diffusion.eulerStep
    (Generative.Diffusion.pfOdeRhs schedule epsModel) afterT1 0.5 dt
  let sampled := Generative.Diffusion.pfOdeSampleEuler schedule epsModel 2 state
  assertApprox "probability-flow Euler time order"
    (vecVal sampled ⟨0, by decide⟩) (vecVal expectedSample ⟨0, by decide⟩)

  let impossibleHmm : Spec.HMMSpec Float 1 1 :=
    { initial := (Tensor.from (#[1] : Array Float)).reshape [1] (by dsimp; decide)
      transition := (Tensor.from (#[1] : Array Float)).reshape [1, 1] (by dsimp; decide)
      emission := (Tensor.from (#[0] : Array Float)).reshape [1, 1] (by dsimp; decide) }
  let observations : Spec.ObservationSeq 1 1 :=
    TorchLean.Tensor.ofFn fun _ => ⟨0, by decide⟩
  assertApprox "impossible HMM observation has zero likelihood"
    (Spec.hmmForwardSpec impossibleHmm observations) 0
  unless (Spec.hmmLogLikelihoodSpec impossibleHmm observations).isNone do
    throw <| IO.userError "impossible HMM observation received a finite log-likelihood"

  let singular : Tensor Float [1, 1] :=
    (Tensor.from (#[0] : Array Float)).reshape [1, 1] (by dsimp; decide)
  unless (Spec.inverseSpec? singular).isNone do
    throw <| IO.userError "singular matrix inverse returned a value"
  let invalidGmm : Spec.GMMSpec Float 1 1 :=
    { weights := (Tensor.from (#[1] : Array Float)).reshape [1] (by dsimp; decide)
      means := (Tensor.from (#[0] : Array Float)).reshape [1, 1] (by dsimp; decide)
      covariances := (Tensor.from (#[0] : Array Float)).reshape [1, 1, 1] (by dsimp; decide) }
  unless (Spec.gmmForwardSpec invalidGmm
      ((Tensor.from (#[0] : Array Float)).reshape [1] (by dsimp; decide))).isNone do
    throw <| IO.userError "GMM accepted a singular covariance"
  let unnormalizedGmm : Spec.GMMSpec Float 2 1 :=
    { weights := (Tensor.from #[0.6, 0.6]).reshape [2] (by dsimp; decide)
      means := (Tensor.from (#[0, 1] : Array Float)).reshape [2, 1] (by dsimp; decide)
      covariances := (Tensor.from (#[1, 1] : Array Float)).reshape [2, 1, 1] (by dsimp; decide) }
  unless (Spec.gmmForwardSpec unnormalizedGmm
      ((Tensor.from (#[0] : Array Float)).reshape [1] (by dsimp; decide))).isNone do
    throw <| IO.userError "GMM accepted mixture weights that do not sum to one"
  let negativeDefiniteGmm : Spec.GMMSpec Float 1 2 :=
    { weights := (Tensor.from (#[1] : Array Float)).reshape [1] (by dsimp; decide)
      means := (Tensor.from (#[0, 0] : Array Float)).reshape [1, 2] (by dsimp; decide)
      covariances := (Tensor.from (#[-1, 0, 0, -1] : Array Float)).reshape
        [1, 2, 2] (by dsimp; decide) }
  unless (Spec.gmmForwardSpec negativeDefiniteGmm
      ((Tensor.from (#[0, 0] : Array Float)).reshape [2] (by dsimp; decide))).isNone do
    throw <| IO.userError "GMM accepted a negative-definite covariance with positive determinant"
  let validGmm : Spec.GMMSpec Float 1 2 :=
    { weights := (Tensor.from (#[1] : Array Float)).reshape [1] (by dsimp; decide)
      means := (Tensor.from (#[0, 0] : Array Float)).reshape [1, 2] (by dsimp; decide)
      covariances := (Tensor.from #[2, 0.5, 0.5, 1]).reshape [1, 2, 2] (by dsimp; decide) }
  unless (Spec.gmmForwardSpec validGmm
      ((Tensor.from (#[0, 0] : Array Float)).reshape [2] (by dsimp; decide))).isSome do
    throw <| IO.userError "GMM rejected a symmetric positive-definite covariance"

/-- Entrypoint called by the curated float runtime suite. -/
def run : IO Unit := do
  IO.println "torchlean_ops_check: begin"
  checkSoftmaxDimensions
  checkClassificationAxes
  checkWeightedMse
  checkWeightedRowCrossEntropy
  checkMiddleAxisCrossEntropy
  checkTiedTokenEmbeddingParameterCount
  checkTiedTokenEmbeddingBackward
  checkAttentionOutputProjectionInitializer
  checkTypedGraphLeafMetadata

  let softmaxInput : Tensor Float [2, 2, 2] :=
    (Tensor.from (#[0, 2, 1, 4, 3, 8, 7, 9] : Array Float)).reshape [2, 2, 2] (by dsimp; decide)
  let expectedSoftmax := Activation.softmaxSpec (α := Float) 1 softmaxInput
  let expectedLogSoftmax := Activation.logSoftmaxSpec (α := Float) 1 softmaxInput
  for execution in [.eager, .typedGraph] do
    let executionName := match execution with
      | .eager => "eager"
      | .typedGraph => "typed_graph"
    let (actualSoftmax, actualLogSoftmax) ← evalSessionSoftmaxFixture execution
    for (actual, expected) in
        (Tensor.to actualSoftmax (Array Float)).zip (Tensor.to expectedSoftmax (Array Float)) do
      assertApprox s!"session softmax axis 1 ({executionName})" actual expected 1e-5
    for (actual, expected) in
        (Tensor.to actualLogSoftmax (Array Float)).zip
          (Tensor.to expectedLogSoftmax (Array Float)) do
      assertApprox s!"session log-softmax axis 1 ({executionName})" actual expected 1e-5

  let mmE ← evalMatmulFixture .eager
  let mmC ← evalMatmulFixture .typedGraph
  for i in List.finRange 2 do
    for j in List.finRange 2 do
      assertApprox s!"matmul[{i.val},{j.val}] eager/typed-graph" (matVal mmE i j)
        (matVal mmC i j) 1e-5

  let cvE ← evalConcatFixture .eager
  let cvC ← evalConcatFixture .typedGraph
  for i in List.finRange 5 do
    assertApprox s!"concat[{i.val}] eager/typed-graph" (vecVal cvE i) (vecVal cvC i) 1e-5

  let mpE ← evalMaxPoolFixture .eager
  let mpC ← evalMaxPoolFixture .typedGraph
  for hi in List.finRange 2 do
    for wi in List.finRange 2 do
      assertApprox s!"max_pool[{hi.val},{wi.val}] eager/typed-graph"
        (tensorVal mpE [0, hi, wi])
        (tensorVal mpC [0, hi, wi])
        1e-5

  let apE ← evalAvgPoolFixture .eager
  let apC ← evalAvgPoolFixture .typedGraph
  for hi in List.finRange 2 do
    for wi in List.finRange 2 do
      assertApprox s!"avg_pool[{hi.val},{wi.val}] eager/typed-graph"
        (tensorVal apE [0, hi, wi])
        (tensorVal apC [0, hi, wi])
        1e-5

  checkBatchNorm
  checkLossSemantics
  checkCorrectedMathematicalSpecs

  IO.println "torchlean_ops_check: ok"

end TorchLeanOpsCheck
end Floats
end Tests

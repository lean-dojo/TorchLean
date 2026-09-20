/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API

/-!
# Public Application Surface Contract

This module intentionally imports only `NN.API`. The examples below exercise the ordinary tensor,
model, data, and trainer workflow, while the declarations at the end pin one canonical entry point
for every application-facing subsystem. If a public name moves or falls out of the umbrella, the
test library stops compiling.
-/

@[expose] public section

namespace NN.Tests.API.PublicSurface

open TorchLean
open TorchLean.Tensor

def vector : Tensor Float [3] := [0.5, 1.5, 2.5]

def bytes : Tensor UInt8 [3] := [1, 2, 3]

def promoted : Tensor Float [3] := bytes + vector

def matrix : Tensor Float [2, 3] :=
  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

def right : Tensor Float [3, 2] :=
  [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

def transposed : Tensor Float [3, 2] :=
  rearrange matrix "row column -> column row"

def expanded : Tensor Float [2, 2, 3] :=
  expand matrix "row column -> batch row column" with batch := 2

def rowSums : Tensor Float [2] :=
  reduce matrix "row column -> row" by sum

def product : Tensor Float [2, 2] :=
  einsum matrix, right
    "row contracted, contracted column -> row column"

def dimensions : List (String × Nat) :=
  parse_shape matrix "row column"

def converted : Array Float :=
  Tensor.to (Tensor.from (#[1.0, 2.0, 3.0] : Array Float)) (Array Float)

def reshaped : Tensor Float [3, 2] :=
  matrix.reshape [3, 2]

def loaded (path : System.FilePath) : IO (Tensor Float [2, 3]) :=
  Tensor.load path

def arguments : Arguments Float [[3], [2]] :=
  (Arguments.empty.push vector).push rowSums

def state : nn.State Float [[3], [2]] :=
  (nn.State.empty.push vector).push rowSums

def model : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![
    nn.linear 2 4,
    nn.relu,
    nn.linear 4 1
  ]

def trainer : Trainer [2] [1] :=
  Trainer.new model
    { objective := .meanSquaredError
      optimizer := optim.adam { learningRate := 0.01 }
      seed := 7 }

def tokens : Tensor Nat [8] :=
  text.Tokenizer.byte.encodeFixed 8 "torchlean"

def initialized : Tensor Float [2, 3] :=
  Init.xavierUniform 7

/-- Checkpoint operations preserve both model and data storage dictionaries. -/
def moduleCheckpoint {α β : Type} [Storage α] [Storage β] [Context α]
    [Checkpoint.Checkpointable α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (objective : Module.Objective α β stateShapes inputShapes dataInputShapes)
    (path : System.FilePath) : IO Unit := do
  Checkpoint.save objective path
  Checkpoint.load objective path

/-- Native binary64 storage must remain usable through the polymorphic checkpoint API. -/
def floatModuleCheckpoint
    (objective : Module.Objective Float Float [[1]] [[1]]) (path : System.FilePath) : IO Unit :=
  moduleCheckpoint objective path

/-- Backend-owned optimizer checkpoints also accept native binary32 modules. -/
def float32OptimizerCheckpoint
    (objective : Module.Objective Float32 Float [[1]] [[1]]) (path : System.FilePath) :
    IO Unit := do
  Checkpoint.Optimizer.save objective path
  Checkpoint.Optimizer.load objective path

#check Tensor.cast
#check Tensor.at
#check Tensor.set
#check Tensor.modify
#check Tensor.qr
#check Tensor.cholesky
#check Tensor.solveRidge
#check Arguments.split
#check nn.State.split
#check nn.lowerToTypedGraph
#check nn.conv
#check nn.multiHeadAttention
#check nn.transformerEncoderBlock
#check nn.heads.classifier
#check nn.models.cnn
#check nn.models.resnet
#check nn.models.unet
#check nn.models.vit
#check nn.models.fno
#check nn.models.kan
#check nn.models.rnn
#check nn.models.gru
#check nn.models.lstm
#check nn.models.Mamba.languageModel
#check nn.models.Generative.autoencoder
#check nn.models.Diffusion.NoisePredictor.basic
#check nn.models.PPO.actor
#check Data.fromSamples
#check Data.batch
#check Data.randomSplit
#check Trainer.Session.step
#check Trainer.Session.stepBatch
#check Checkpoint.State.save
#check autograd.grad
#check autograd.vjp
#check autograd.model.grad
#check autograd.model.vjp
#check optim.adamW
#check Loss.oneHotCrossEntropy
#check Metrics.argmaxAxis?
#check Runtime.ExecutionMode.parse
#check Runtime.Device.parse
#check rand.manualSeed
#check text.chooseNextToken
#check text.GPT2BPE.load
#check ssl.BlockMask.apply
#check rl.replay.empty
#check rl.policy.ppoLoss

def expect (label : String) (condition : Bool) : IO Unit := do
  unless condition do
    throw <| IO.userError s!"public API contract failed: {label}"

/--
Differentiate a builder's output against a supplied cotangent through the public objective API.

The input is included as one more trainable state entry, after the model's parameters and buffers.
That lets the same eager or typed objective return both parameter and input VJPs. The cotangent
remains an ordinary argument, and the scalar objective is `sum(output * cotangent)`.
-/
def normalizationObjective (model : nn.Sequential [1, 2] [1, 2]) :
    Module.ObjectiveDefinition Unit (nn.stateShapes model ++ [[1, 2]]) [[1, 2]] :=
  let definition := nn.Objective.fromLoss model
    (fun {α} _ _ {m} _ _ output cotangent =>
      show m (Runtime.Autograd.Model.RefTy (m := m) (α := α) []) from do
        let product ← Runtime.mul (m := m) (α := α) output cotangent
        Runtime.sum (m := m) (α := α) product)
  { initState := TensorPack.append definition.initState
      (TensorPack.singleton ([[-0.25, 0.25]] : Tensor Float [1, 2]))
    requiresGrad := (nn.requiresGrad model).push true
    validate := nn.validate model
    loss := fun {α} _ _ {m} _ _ => by
      simpa only [List.append_assoc, List.cons_append, List.nil_append] using
        (definition.loss (α := α) (m := m)) }

/-- Expected state and input VJPs for one public normalization builder. -/
structure NormalizationCase where
  label : String
  model : nn.Sequential [1, 2] [1, 2]
  stateVjp : Array (Array Float)
  inputVjp : Array Float

/--
Small normalization examples with independently checked PyTorch values.

For `x = [-1/4, 1/4]` and `eps = 1/4`, both the variance and mean square are `1/16`,
so the initial unit scale gives `[-1/√5, 1/√5]`. The cotangent `[2, -1]` has a nonzero
mean: LayerNorm removes that component from the input VJP, while RMSNorm keeps it.
The numbers below were also evaluated with PyTorch 2.13 in float64. Affine options change
which parameters exist; with their initial scale one and bias zero, the input VJP stays the same.
-/
def normalizationCases (eps : Rat) : List NormalizationCase :=
  let centered := #[2.1466252583997982, -2.1466252583997982]
  let uncentered := #[3.041052449399714, -1.2521980673998823]
  let elementScale := #[-0.8944271909999159, -0.4472135954999579]
  let channelScale := #[-1.3416407864998738]
  [ ⟨"LayerNorm", nn.build 0 (nn.layerNorm [1] (width := 2) (eps := eps)),
      #[elementScale, #[2.0, -1.0]], centered⟩
  , ⟨"LayerNorm without bias",
      nn.build 0 (nn.layerNorm [1] (width := 2) (eps := eps) (bias := false)),
      #[elementScale], centered⟩
  , ⟨"LayerNorm without affine",
      nn.build 0 (nn.layerNorm [1] (width := 2) (eps := eps) (affine := false)),
      #[], centered⟩
  , ⟨"RMSNorm", nn.build 0 (nn.rmsNorm [1] (width := 2) (eps := eps)),
      #[elementScale], uncentered⟩
  , ⟨"RMSNorm without affine",
      nn.build 0 (nn.rmsNorm [1] (width := 2) (eps := eps) (affine := false)),
      #[], uncentered⟩
  , ⟨"InstanceNorm", nn.build 0 (nn.instanceNorm [2] (channels := 1) (eps := eps)),
      #[channelScale, #[1.0]], centered⟩
  , ⟨"InstanceNorm without bias",
      nn.build 0 (nn.instanceNorm [2] (channels := 1) (eps := eps) (bias := false)),
      #[channelScale], centered⟩
  , ⟨"InstanceNorm without affine",
      nn.build 0 (nn.instanceNorm [2] (channels := 1) (eps := eps) (affine := false)),
      #[], centered⟩
  , ⟨"GroupNorm", nn.build 0 (nn.groupNorm [2] 1 (channels := 1) (eps := eps)),
      #[channelScale, #[1.0]], centered⟩
  , ⟨"GroupNorm without bias",
      nn.build 0 (nn.groupNorm [2] 1 (channels := 1) (eps := eps) (bias := false)),
      #[channelScale], centered⟩
  , ⟨"GroupNorm without affine",
      nn.build 0 (nn.groupNorm [2] 1 (channels := 1) (eps := eps) (affine := false)),
      #[], centered⟩
  , ⟨"BatchNorm", nn.build 0 (nn.batchNorm [2] (channels := 1) (eps := eps)),
      #[channelScale, #[1.0], #[0.0], #[0.0], #[0.0]], centered⟩ ]

/-- Compare every finite oracle value, including the shape's total element count. -/
def expectNormalizationValues (label : String) (actual expected : Array Float) : IO Unit := do
  expect s!"{label}: element count" (actual.size == expected.size)
  for (value, target) in actual.toList.zip expected.toList do
    expect s!"{label}: expected {target}, got {value}"
      (value.isFinite && Float.abs (value - target) ≤ 1e-9)

/--
Check parameter omission, custom epsilon, and gradients through the public builders.

BatchNorm retains its five state entries; its three buffers have zero objective gradients.
The other builders have only the scale and bias entries requested by their options.
-/
def checkNormalizationBuilders : IO Unit := do
  let input : Tensor Float [1, 2] := [[-0.25, 0.25]]
  let cotangent : Tensor Float [1, 2] := [[2.0, -1.0]]
  for testCase in normalizationCases (1 / 4) do
    expect s!"{testCase.label}: parameter and buffer count"
      ((nn.stateShapes testCase.model).length == testCase.stateVjp.size)
    for execution in [Runtime.ExecutionMode.eager, .typedGraph] do
      let config : Runtime.Config := { execution := execution, device := .cpu }
      let label := s!"{testCase.label} ({reprStr execution})"
      let module ← nn.Module.instantiate testCase.model config (α := Float)
      let output ← module.forward input
      expectNormalizationValues s!"{label}: forward" (output.to (Array Float))
        #[-0.4472135954999579, 0.4472135954999579]
      let objective ← Module.instantiate (normalizationObjective testCase.model) config (α := Float)
      let (gradients, value) ← objective.grad
        (Arguments.empty.push cotangent) Arguments.empty (value := true)
      expectNormalizationValues s!"{label}: output dotted with cotangent"
        (value.to (Array Float)) #[-1.3416407864998738]
      let parts := gradients.split
        (leftShapes := nn.stateShapes testCase.model) (rightShapes := [[1, 2]])
      expectNormalizationValues s!"{label}: input VJP"
        ((parts.right.get 0).to (Array Float)) testCase.inputVjp
      for index in List.finRange (nn.stateShapes testCase.model).length do
        expectNormalizationValues s!"{label}: state VJP {index.val}"
          ((parts.left.get index).to (Array Float)) testCase.stateVjp[index.val]!
  for eps in [(0 : Rat), -(1 / 4 : Rat)] do
    for testCase in normalizationCases eps do
      expect s!"{testCase.label}: nonpositive epsilon fails validation"
        (!(nn.validate testCase.model).isOk)
      let rejected ← try
        let _ ← nn.Module.instantiate testCase.model (α := Float)
        pure false
      catch error =>
        pure (error.toString.contains "epsilon must be positive")
      expect s!"{testCase.label}: nonpositive epsilon fails instantiation" rejected

def run : IO Unit := do
  expect "mixed tensor arithmetic"
    (Tensor.to promoted (Array Float) == #[1.5, 3.5, 5.5])
  expect "shape-pattern reduction"
    (Tensor.to rowSums (Array Float) == #[6.0, 15.0])
  expect "general conversion"
    (converted == #[1.0, 2.0, 3.0])
  expect "text tensors"
    (tokens.to (Array Nat) == #[116, 111, 114, 99, 104, 108, 101, 97])
  checkNormalizationBuilders
  IO.println "  public application surface: passed"

end NN.Tests.API.PublicSurface

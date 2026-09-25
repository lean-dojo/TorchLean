/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Models.Generative
public import NN.GraphSpec.Primitives.Spatial
public import NN.GraphSpec.ToSequential
public import NN.Spec.Module.GruModels
public import NN.Spec.Module.LstmModels
public import NN.Spec.Module.RnnModels
public import NN.Tests.Utils

/-!
# Model Boundary Regressions

Empty recurrent sequences preserve their initial states. Classifier gradients, convolution
validation, and discriminator initialization must agree with the model actually being executed.
-/

@[expose] public section

namespace NN.Tests.API.ModelEdgeCases

open TorchLean

def expect (label : String) (condition : Bool) : IO Unit := do
  unless condition do
    throw <| IO.userError s!"model boundary regression: {label}"

def expectValues (label : String) (actual expected : Array Float) : IO Unit := do
  expect s!"{label}: finite values" (actual.all Float.isFinite && expected.all Float.isFinite)
  expect s!"{label}: identical bits" (actual.map Float.toBits == expected.map Float.toBits)

def expectTensor {shape : Shape} (label : String) (actual expected : Tensor Float shape) :
    IO Unit :=
  expectValues label (actual.to (Array Float)) (expected.to (Array Float))

def expectValidation (label : String) (message? : Option String)
    (result : Except String Unit) : IO Unit :=
  match message?, result with
  | none, .ok () => pure ()
  | some expected, .error actual => expect label (actual == expected)
  | _, _ => throw <| IO.userError s!"model boundary regression: {label}: unexpected validation"

def referenceTensor (shape : Shape) (offset : Float) : Tensor Float shape :=
  Tensor.generateFlat shape fun index => offset + (index % 7).toFloat / 31

def referenceHead : Spec.LinearSpec Float 3 1 :=
  { weights := [[1.0, -0.25, 0.5]], bias := [0.2] }

def referenceRnn : Spec.RNNSpec Float 2 3 :=
  { weights := referenceTensor _ 0.03, bias := referenceTensor _ (-0.02) }

def referenceGru : Spec.GRUSpec Float 2 3 :=
  { resetWeight := referenceTensor _ 0.01, resetBias := referenceTensor _ 0.02
    updateWeight := referenceTensor _ (-0.03), updateBias := referenceTensor _ 0.04
    candidateWeight := referenceTensor _ 0.05, candidateBias := referenceTensor _ (-0.06) }

def checkRnnSequence : IO Unit := do
  let model : Spec.Rnn.Model Float 2 3 1 :=
    { rnn := referenceRnn, outputLayer := referenceHead }
  let stack : Spec.Rnn.StackedModel Float 2 3 1 :=
    { layers := .cons referenceRnn .nil, outputLayer := referenceHead }
  let initial := referenceTensor [3] 0.3
  let (emptyOutput, emptyState) := model.forwardSequence (Tensor.zeros [0, 2]) initial
  expectTensor "RNN empty output" emptyOutput (Tensor.zeros [0, 1])
  expectTensor "RNN empty final state" emptyState initial
  let (stackOutput, stackState) := stack.forward (Tensor.zeros [0, 2]) (initial, ())
  expectTensor "RNN empty singleton stack output" stackOutput emptyOutput
  expectTensor "RNN empty singleton stack state" stackState.1 initial
  expectTensor "RNN empty module" (model.toModule.forward (Tensor.zeros [0, 2]))
    (Tensor.zeros [0, 1])
  let input := referenceTensor [2, 2] (-0.1)
  let hidden := Spec.rnnSequenceSpec referenceRnn input initial
  let (output, finalState) := model.forwardSequence input initial
  expectTensor "RNN nonempty output" output <|
    Tensor.mapLeading [2] (Spec.linearSpec referenceHead) hidden
  expectTensor "RNN nonempty final state" finalState (Tensor.get hidden 1)
  let (stackOutput, stackState) := stack.forward input (initial, ())
  expectTensor "RNN nonempty singleton stack output" stackOutput output
  expectTensor "RNN nonempty singleton stack state" stackState.1 finalState
  expectTensor "RNN nonempty module" (model.toModule.forward input) <|
    Tensor.mapLeading [2] (Spec.linearSpec referenceHead)
      (Spec.rnnSequenceSpec referenceRnn input (Tensor.zeros [3]))

def checkGruSequence : IO Unit := do
  let model : Spec.Gru.Model Float 2 3 1 :=
    { gru := referenceGru, outputLayer := referenceHead }
  let stack : Spec.Gru.StackedModel Float 2 3 1 :=
    { layers := .cons referenceGru .nil, outputLayer := referenceHead }
  let initial := referenceTensor [3] 0.3
  let (emptyOutput, emptyState) := model.forwardSequence (Tensor.zeros [0, 2]) initial
  expectTensor "GRU empty output" emptyOutput (Tensor.zeros [0, 1])
  expectTensor "GRU empty final state" emptyState initial
  let (stackOutput, stackState) := stack.forward (Tensor.zeros [0, 2]) (initial, ())
  expectTensor "GRU empty singleton stack output" stackOutput emptyOutput
  expectTensor "GRU empty singleton stack state" stackState.1 initial
  expectTensor "GRU empty module" (model.toModule.forward (Tensor.zeros [0, 2]))
    (Tensor.zeros [0, 1])
  let input := referenceTensor [2, 2] (-0.1)
  let hidden := Spec.gruSequenceSpec referenceGru input initial
  let (output, finalState) := model.forwardSequence input initial
  expectTensor "GRU nonempty output" output <|
    Tensor.mapLeading [2] (Spec.linearSpec referenceHead) hidden
  expectTensor "GRU nonempty final state" finalState (Tensor.get hidden 1)
  let (stackOutput, stackState) := stack.forward input (initial, ())
  expectTensor "GRU nonempty singleton stack output" stackOutput output
  expectTensor "GRU nonempty singleton stack state" stackState.1 finalState
  expectTensor "GRU nonempty module" (model.toModule.forward input) <|
    Tensor.mapLeading [2] (Spec.linearSpec referenceHead)
      (Spec.gruSequenceSpec referenceGru input (Tensor.zeros [3]))

def referenceLstm : Spec.LSTMSpec Float 1 1 :=
  { forgetWeight := [[0.1, -0.2]], forgetBias := [0.05]
    inputWeight := [[-0.3, 0.1]], inputBias := [-0.04]
    candidateWeight := [[0.2, 0.3]], candidateBias := [0.07]
    outputWeight := [[-0.1, 0.2]], outputBias := [0.03] }

def referenceClassifier : Spec.Lstm.Classifier Float 1 1 1 :=
  { lstm := referenceLstm, classifier := { weights := [[3.0]], bias := [0.0] } }

def checkEmptyLstmClassifier : IO Unit := do
  let input : Tensor Float [0, 1] := Tensor.zeros _
  let initial : Spec.LSTMState Float 1 := { hidden := [2.0], cell := [-0.75] }
  expectTensor "empty LSTM classifier forward"
    (referenceClassifier.forward input initial) ([6.0] : Tensor Float [1])
  let (gradients, inputGradient, lossGradients) :=
    referenceClassifier.backward input initial ([1.0] : Tensor Float [1])
  expectTensor "empty LSTM head weight gradient" gradients.classifier.weightGradient
    ([[2.0]] : Tensor Float [1, 1])
  expectTensor "empty LSTM head bias gradient" gradients.classifier.biasGradient
    ([1.0] : Tensor Float [1])
  expectTensor "empty LSTM initial hidden gradient" lossGradients.hidden
    ([3.0] : Tensor Float [1])
  expectTensor "empty LSTM initial cell gradient" lossGradients.cell (Tensor.zeros [1])
  expectTensor "empty LSTM input gradient" inputGradient (Tensor.zeros [0, 1])
  for weight in [gradients.cell.forgetWeight, gradients.cell.inputWeight,
      gradients.cell.candidateWeight, gradients.cell.outputWeight] do
    expectTensor "empty LSTM gate weight gradient" weight (Tensor.zeros [1, 2])
  for bias in [gradients.cell.forgetBias, gradients.cell.inputBias,
      gradients.cell.candidateBias, gradients.cell.outputBias] do
    expectTensor "empty LSTM gate bias gradient" bias (Tensor.zeros [1])

/-- Check every coordinate against a central difference of the actual forward objective. -/
def checkGradient {shape : Shape} (label : String) (value gradient : Tensor Float shape)
    (objective : Tensor Float shape → Float) : IO Unit := do
  let values := value.to (Array Float)
  let gradients := gradient.to (Array Float)
  let epsilon := 1e-5
  for index in [:values.size] do
    let shifted := fun delta =>
      Tensor.generateFlat shape fun coordinate =>
        values[coordinate]! + if coordinate == index then delta else 0.0
    let difference := (objective (shifted epsilon) - objective (shifted (-epsilon))) /
      (2.0 * epsilon)
    _root_.Tests.Utils.assertApprox s!"{label}[{index}]" gradients[index]! difference 1e-6

def checkNonemptyLstmClassifier : IO Unit := do
  let model := referenceClassifier
  let input : Tensor Float [2, 1] := [[0.25], [-0.5]]
  let initial : Spec.LSTMState Float 1 := { hidden := [2.0], cell := [-0.75] }
  let upstream : Tensor Float [1] := [1.25]
  let (gradients, inputGradient, lossGradients) := model.backward input initial upstream
  let objective := fun (candidate : Spec.Lstm.Classifier Float 1 1 1)
      (xs : Tensor Float [2, 1]) (state : Spec.LSTMState Float 1) =>
    1.25 * (candidate.forward xs state).getScalar 0
  checkGradient "LSTM classifier head weight" model.classifier.weights
    gradients.classifier.weightGradient fun value =>
      objective { model with classifier := { model.classifier with weights := value } }
        input initial
  checkGradient "LSTM classifier head bias" model.classifier.bias
    gradients.classifier.biasGradient fun value =>
      objective { model with classifier := { model.classifier with bias := value } } input initial
  checkGradient "LSTM classifier inputs" input inputGradient fun value =>
    objective model value initial
  checkGradient "LSTM classifier initial hidden" initial.hidden lossGradients.hidden fun value =>
    objective model input { initial with hidden := value }
  checkGradient "LSTM classifier initial cell" initial.cell lossGradients.cell fun value =>
    objective model input { initial with cell := value }
  checkGradient "LSTM forget weight" model.lstm.forgetWeight
    gradients.cell.forgetWeight fun value =>
      objective { model with lstm := { model.lstm with forgetWeight := value } } input initial
  checkGradient "LSTM forget bias" model.lstm.forgetBias gradients.cell.forgetBias fun value =>
    objective { model with lstm := { model.lstm with forgetBias := value } } input initial
  checkGradient "LSTM input weight" model.lstm.inputWeight gradients.cell.inputWeight fun value =>
    objective { model with lstm := { model.lstm with inputWeight := value } } input initial
  checkGradient "LSTM input bias" model.lstm.inputBias gradients.cell.inputBias fun value =>
    objective { model with lstm := { model.lstm with inputBias := value } } input initial
  checkGradient "LSTM candidate weight" model.lstm.candidateWeight
    gradients.cell.candidateWeight fun value =>
      objective { model with lstm := { model.lstm with candidateWeight := value } } input initial
  checkGradient "LSTM candidate bias" model.lstm.candidateBias
    gradients.cell.candidateBias fun value =>
      objective { model with lstm := { model.lstm with candidateBias := value } } input initial
  checkGradient "LSTM output weight" model.lstm.outputWeight
    gradients.cell.outputWeight fun value =>
      objective { model with lstm := { model.lstm with outputWeight := value } } input initial
  checkGradient "LSTM output bias" model.lstm.outputBias gradients.cell.outputBias fun value =>
    objective { model with lstm := { model.lstm with outputBias := value } } input initial

def stateValues {source target : Shape} (model : nn.Sequential source target) :
    Array (Array Float) :=
  Array.ofFn fun index : Fin (nn.stateShapes model).length =>
    ((nn.initialState model).get index).to (Array Float)

def initializerSeed? :
    Runtime.Autograd.Model.Module.RuntimeInit.FloatInit → Option Nat
  | .uniform _ _ seed
  | .normal _ _ seed
  | .xavierUniform _ _ seed
  | .kaimingUniform _ seed => some seed
  | .zeros | .ones | .flat _ => none

def initializationSeeds? {source target : Shape} (model : nn.Sequential source target) :
    Option (Array (Option Nat)) :=
  (nn.runtimeInit? model).map fun plan => plan.toArray.map initializerSeed?

def checkDiscriminator : IO Unit := do
  let config : nn.models.Generative.Config :=
    { dataWidth := 4, hiddenWidth := 3, latentWidth := 0 }
  let (zeroLatent, zeroStream) :=
    (nn.models.Generative.discriminator config (batchShape := [])) (rand.SeedStream.init 37)
  let (oneLatent, oneStream) :=
    (nn.models.Generative.discriminator { config with latentWidth := 1 } (batchShape := []))
      (rand.SeedStream.init 37)
  expectValidation "zero-latent discriminator" none (nn.validate zeroLatent)
  expectValidation "positive-latent discriminator" none (nn.validate oneLatent)
  let expectedShapes : List Shape := [[3, 4], [3], [3, 3], [3], [1, 3], [1]]
  expect "discriminator parameter shapes"
    (nn.stateShapes zeroLatent == expectedShapes && nn.stateShapes oneLatent == expectedShapes)
  let zeroValues := stateValues zeroLatent
  let oneValues := stateValues oneLatent
  expect "discriminator parameter count" (zeroValues.size == 6 && oneValues.size == 6)
  for index in [:zeroValues.size] do
    expectValues s!"discriminator parameter {index}" zeroValues[index]! oneValues[index]!
  expect "discriminator allocates three weight seeds"
    (zeroStream.counter == 3 && oneStream.counter == 3)
  let seeds := some #[some (rand.nextSeed 37 0), none, some (rand.nextSeed 37 1), none,
    some (rand.nextSeed 37 2), none]
  expect "zero-latent discriminator seed order" (initializationSeeds? zeroLatent == seeds)
  expect "positive-latent discriminator seed order" (initializationSeeds? oneLatent == seeds)
  expectValidation "whole generative config still requires latent width"
    (some "Generative: latent width must be positive") config.validate
  let (generator, generatorStream) :=
    (nn.models.Generative.generator config (batchShape := [])) (rand.SeedStream.init 37)
  let (autoencoder, autoencoderStream) :=
    (nn.models.Generative.autoencoder config (batchShape := [])) (rand.SeedStream.init 37)
  expectValidation "generator still requires latent width"
    (some "Generator: latent width must be positive") (nn.validate generator)
  expectValidation "autoencoder still requires latent width"
    (some "Autoencoder: latent width must be positive") (nn.validate autoencoder)
  expect "invalid latent widths consume no generator or autoencoder seeds"
    (generatorStream.counter == 0 && autoencoderStream.counter == 0)
  for (invalid, message) in [
      ({ config with dataWidth := 0 }, "Discriminator: data width must be positive"),
      ({ config with hiddenWidth := 0 }, "Discriminator: hidden width must be positive")] do
    let (model, stream) :=
      (nn.models.Generative.discriminator invalid (batchShape := [])) (rand.SeedStream.init 37)
    expectValidation "discriminator validates used widths" (some message) (nn.validate model)
    expect "invalid discriminator consumes no seeds" (stream.counter == 0)

def checkConvValidation {rank : Nat} (label : String) (inputChannels outputChannels : Nat)
    (kernel stride padding spatial : Tensor Nat [rank]) (message? : Option String) : IO Unit := do
  let ordinary := nn.build 37 <| nn.conv (inputChannels := inputChannels) spatial
    { outChannels := outputChannels, kernelSize := kernel, stride, padding }
  expectValidation s!"{label}: ordinary convolution" message? (nn.validate ordinary)
  let chain := GraphSpec.Chain.conv inputChannels outputChannels kernel stride padding spatial
  match GraphSpec.ToSequential.toSeq chain with
  | .error message => throw <| IO.userError s!"{label}: convolution lowering failed: {message}"
  | .ok model => expectValidation s!"{label}: GraphSpec convolution" message? model.validate

def checkConvolution : IO Unit := do
  checkConvValidation "zero stride" 1 1 ([1] : Tensor Nat [1]) [0] [0] [4]
    (some "Conv: stride entries must be positive")
  checkConvValidation "zero kernel" 1 1 ([0] : Tensor Nat [1]) [1] [0] [4]
    (some "Conv: kernel size entries must be positive")
  checkConvValidation "empty input with nonempty padded output" 1 1
    ([1] : Tensor Nat [1]) [1] [1] [0]
    (some "Conv: input spatial dimensions must be positive")
  checkConvValidation "collapsed output" 1 1 ([3] : Tensor Nat [1]) [1] [0] [1]
    (some "Conv: geometry produced an empty spatial grid")
  checkConvValidation "zero input channels" 0 1 ([1] : Tensor Nat [1]) [1] [0] [4]
    (some "Conv: input channel count must be positive")
  checkConvValidation "zero output channels" 1 0 ([1] : Tensor Nat [1]) [1] [0] [4]
    (some "Conv: output channel count must be positive")
  checkConvValidation "zero stride on second axis" 1 1
    ([1, 1] : Tensor Nat [2]) [1, 0] [0, 0] [4, 4]
    (some "Conv: stride entries must be positive")
  checkConvValidation "valid padded convolution" 1 2 ([3] : Tensor Nat [1]) [1] [1] [4] none
  let chain := GraphSpec.Chain.conv 1 2 ([3] : Tensor Nat [1]) [1] [1] [4]
  for occurrence in [0, 3] do
    match GraphSpec.ToSequential.Internal.toSeqFrom chain occurrence with
    | .error message => throw <| IO.userError s!"valid convolution lowering failed: {message}"
    | .ok (model, nextOccurrence) =>
        expectValidation "initialized GraphSpec convolution" none model.validate
        expect "GraphSpec convolution advances occurrence index" (nextOccurrence == occurrence + 1)
        expect "GraphSpec convolution parameter order"
          (nn.stateShapes model == ([ [2, 1, 3], [2] ] : List Shape))
        expect "GraphSpec convolution trainability" (nn.requiresGrad model == #[true, true])
        let values := stateValues model
        expect "GraphSpec convolution parameter count" (values.size == 2)
        expectValues "GraphSpec convolution kernel initialization" values[0]! <|
          (Runtime.Autograd.Torch.Init.tensor (s := [2, 1, 3])
            (.uniform (-0.1) 0.1) occurrence).to (Array Float)
        expectValues "GraphSpec convolution bias initialization" values[1]! #[0.0, 0.0]
        expect "GraphSpec convolution runtime seed order"
          (initializationSeeds? model == some #[some occurrence, none])

/-- Reading past a flat initializer payload reports an error instead of panicking. -/
def checkFlatInitializerIndex : IO Unit := do
  let init := Runtime.Autograd.Model.Module.RuntimeInit.FloatInit.flat (FloatArray.mk #[2.5])
  expect "flat initializer in-range sample"
    (match Runtime.Autograd.Model.Module.RuntimeInit.sampleAt init 0 with
      | .ok value => value == 2.5
      | .error _ => false)
  expect "flat initializer out-of-range sample"
    (match Runtime.Autograd.Model.Module.RuntimeInit.sampleAt init 1 with
      | .ok _ => false
      | .error message => message.contains "outside")

def run : IO Unit := do
  checkEmptyLstmClassifier
  checkNonemptyLstmClassifier
  checkRnnSequence
  checkGruSequence
  checkDiscriminator
  checkConvolution
  checkFlatInitializerIndex
  IO.println "  Model boundary regressions: passed"

end NN.Tests.API.ModelEdgeCases

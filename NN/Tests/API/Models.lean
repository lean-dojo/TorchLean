/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Models
public import NN.API.Module
public import NN.API.Neural.Execution
public import NN.API.Neural.Summary

/-!
# Public Model Execution Tests

CPU forward checks for public architectures, with complete output and VJP regressions for
configurable model composition and deterministic initialization.
-/

@[expose] public section

namespace NN.Tests.API.Models

open TorchLean

def expect (label : String) (condition : Bool) : IO Unit := do
  unless condition do
    throw <| IO.userError s!"public model execution failed: {label}"

def runModel {input output : Spec.Shape}
    (label : String) (model : nn.Sequential input output) : IO Unit := do
  expect s!"{label} validates" (nn.validate model).isOk
  let module ← nn.Module.instantiate model (α := Float)
  let result ← module.forward (mode := some .eval) (Tensor.zeros input)
  let values := Tensor.to result (Array Float)
  expect s!"{label} returns the exact output size" (values.size == output.size)
  expect s!"{label} returns finite values" (values.all Float.isFinite)

def checkModuleModeOverride : IO Unit := do
  let model : nn.Sequential [2] [2] :=
    nn.build 0 nn.relu
  let module ← nn.Module.instantiate model (α := Float)
  let input : Tensor Float [2] := [-1.0, 1.0]
  expect "ordinary module starts in training mode" (← module.isTraining)
  let _ ← module.forward (mode := some .eval) input
  expect "ordinary predict preserves training mode" (← module.isTraining)
  module.eval
  let _ ← module.forward (mode := some .eval) input
  expect "ordinary predict preserves evaluation mode" (!(← module.isTraining))

def indexedModel : nn.IndexedModel [2] [2, 2] (Fin 3) :=
  (nn.build 0 (nn.embedding 3 2)).model [2]

def checkIndexedModeOverride : IO Unit := do
  let module ← nn.IndexedModule.instantiate indexedModel (α := Float)
  let input : Tensor (Fin 3) [2] := [0, 2]
  expect "indexed module starts in training mode" (← module.isTraining)
  let _ ← module.forward (mode := some .eval) input
  expect "indexed predict preserves training mode" (← module.isTraining)
  module.eval
  let _ ← module.forward (mode := some .eval) input
  expect "indexed predict preserves evaluation mode" (!(← module.isTraining))

def cnnConfig : nn.models.CNN.Config 1 :=
  { inputChannels := 1
    spatial := [4]
    stages :=
      [{ block :=
           { convolution :=
               { outChannels := 2
                 kernelSize := [3]
                 padding := [1] } }
         pooling :=
           { kernelSize := [2]
             stride := [2] } }]
    classCount := 3 }

def cnn : nn.Sequential (cnnConfig.inputShape [2]) (cnnConfig.outputShape [2]) :=
  nn.build 1 (nn.models.cnn cnnConfig [2])

def resnetConfig : nn.models.ResNet.Config 1 :=
  let convolution : nn.Convolution.Config 1 :=
    { outChannels := 2, kernelSize := [3], padding := [1] }
  { inputChannels := 1
    spatial := [4]
    stem := convolution
    stages := List.replicate 2 { first := convolution, second := convolution }
    classCount := 3 }

def resnet : nn.Sequential (resnetConfig.inputShape [2]) (resnetConfig.outputShape [2]) :=
  nn.build 2 (nn.models.resnet resnetConfig [2])

def vitEncoderConfig (pooling : nn.models.ViT.Pooling) : nn.models.ViT.EncoderConfig 1 :=
  { inputChannels := 1
    spatial := [2]
    patchEmbedding :=
      { outChannels := 2
        kernelSize := [1] }
    headCount := 1
    headWidth := 2
    feedForwardWidth := 3
    layerCount := 1
    pooling }

def vitConfig (pooling : nn.models.ViT.Pooling) : nn.models.ViT.Config 1 :=
  (vitEncoderConfig pooling).classifier 3

def meanVit : nn.Sequential
    ((vitConfig .mean).inputShape [2]) ((vitConfig .mean).outputShape [2]) :=
  nn.build 3 (nn.models.vit (vitConfig .mean) [2])

def classVit : nn.Sequential ((vitConfig .cls).inputShape [2]) ((vitConfig .cls).outputShape [2]) :=
  nn.build 4 (nn.models.vit (vitConfig .cls) [2])

def maskedPatchConfig : nn.models.ViT.MaskedPatchReconstructor.Config 1 :=
  { encoder := vitEncoderConfig .mean
    reconstructionWidth := 3 }

def maskedPatchReconstructor :
    nn.Sequential (maskedPatchConfig.encoder.inputShape [2]) (maskedPatchConfig.outputShape [2]) :=
  nn.build 5 (nn.models.ViT.maskedPatchReconstructor maskedPatchConfig [2])

def kanConfig : nn.models.KAN.Config :=
  { inputWidth := 2
    hiddenWidths := [3]
    outputWidth := 1
    edge :=
      nn.models.KAN.PiecewiseLinear.edgeFamily
        { gridSize := 3, inputScale := 2 } }

def kan : nn.Sequential (kanConfig.inputShape [2]) (kanConfig.outputShape [2]) :=
  nn.build 6 (nn.models.kan kanConfig [2])

def fnoConfig : nn.models.FNO.Config 1 :=
  { spatial := [2]
    modes := [1]
    width := 1
    layerCount := 1
    activation := .gelu }

def fno : nn.Sequential (fnoConfig.inputShape [2]) (fnoConfig.outputShape [2]) :=
  nn.build 7 (nn.models.fno fnoConfig [2])

def recurrentConfig : nn.models.Recurrent.Config :=
  { sequenceLength := 2
    inputWidth := 2
    hiddenWidths := [2]
    outputWidth := 1 }

def rnn : nn.Sequential (recurrentConfig.inputShape [2]) (recurrentConfig.outputShape [2]) :=
  nn.build 8 (nn.models.rnn recurrentConfig [2])

def gru : nn.Sequential (recurrentConfig.inputShape [2]) (recurrentConfig.outputShape [2]) :=
  nn.build 9 (nn.models.gru recurrentConfig [2])

def lstm : nn.Sequential (recurrentConfig.inputShape [2]) (recurrentConfig.outputShape [2]) :=
  nn.build 10 (nn.models.lstm recurrentConfig [2])

def mambaConfig : nn.models.Mamba.Config :=
  { vocabularySize := 2
    modelWidths := [2] }

def mamba :
    nn.Sequential (mambaConfig.inputShape 2 [2]) (mambaConfig.outputShape 2 [2]) :=
  nn.build 11 (nn.models.Mamba.languageModel mambaConfig 2 [2])

def causalTransformerConfig : nn.models.CausalTransformer.Config :=
  { sequenceLength := 3
    vocabularySize := 4
    headCount := 1
    headWidth := 4
    feedForwardWidth := 8
    layerCount := 1 }

local instance : NeZero causalTransformerConfig.vocabularySize := ⟨by decide⟩

def causalTransformer :
    nn.Sequential
      (causalTransformerConfig.vocabularyShape [1])
      (causalTransformerConfig.vocabularyShape [1]) :=
  nn.build 12 (nn.models.CausalTransformer.oneHot causalTransformerConfig [1])

def causalInput (second third : Fin causalTransformerConfig.vocabularySize) :
    Tensor Float (causalTransformerConfig.vocabularyShape [1]) :=
  let tokens : Tensor (Fin causalTransformerConfig.vocabularySize) [1, 3] :=
    Tensor.stack 0 fun _ =>
      Tensor.ofFn fun position =>
        if position.val = 0 then
          Fin.ofNat causalTransformerConfig.vocabularySize 0
        else if position.val = 1 then
          second
        else
          third
  by
    simpa [causalTransformerConfig, nn.models.CausalTransformer.Config.vocabularyShape] using
      TorchLean.Tensor.oneHotIndices
        (α := Float) causalTransformerConfig.vocabularySize tokens

def expectApprox (label : String) (actual expected tolerance : Float) : IO Unit := do
  unless Float.abs (actual - expected) ≤ tolerance do
    throw <| IO.userError
      s!"public model execution failed: {label} (expected {expected}, got {actual})"

def checkCausalTransformerFutureIsolation : IO Unit := do
  let module ← nn.Module.instantiate causalTransformer (α := Float)
  let first ← module.forward (mode := some .eval)
    (causalInput
      (Fin.ofNat causalTransformerConfig.vocabularySize 1)
      (Fin.ofNat causalTransformerConfig.vocabularySize 2))
  let changedFuture ← module.forward (mode := some .eval)
    (causalInput
      (Fin.ofNat causalTransformerConfig.vocabularySize 3)
      (Fin.ofNat causalTransformerConfig.vocabularySize 3))
  for token in [0:causalTransformerConfig.vocabularySize] do
    let firstValue :=
      first.at? #[0, 0, token] |>.getD (0.0 / 0.0)
    let changedValue :=
      changedFuture.at? #[0, 0, token] |>.getD (0.0 / 0.0)
    expectApprox s!"causal position 0, token {token}"
      changedValue firstValue 1e-6

def generativeConfig : nn.models.Generative.Config :=
  { dataWidth := 3
    hiddenWidth := 2
    latentWidth := 2 }

def autoencoder :
    nn.Sequential (generativeConfig.dataShape [2]) (generativeConfig.dataShape [2]) :=
  nn.build 12 (nn.models.Generative.autoencoder generativeConfig [2])

def generator :
    nn.Sequential (generativeConfig.latentShape [2]) (generativeConfig.dataShape [2]) :=
  nn.build 13 (nn.models.Generative.generator generativeConfig [2])

def discriminator :
    nn.Sequential (generativeConfig.dataShape [2]) (generativeConfig.scoreShape [2]) :=
  nn.build 14 (nn.models.Generative.discriminator generativeConfig [2])

def diffusionConfig : nn.models.Diffusion.NoisePredictor.Config 1 :=
  { dataChannels := 1
    spatial := [4]
    hiddenChannels := 2 }

def basicDiffusion :
    nn.Sequential (diffusionConfig.inputShape [2]) (diffusionConfig.outputShape [2]) :=
  nn.build 15 (nn.models.Diffusion.NoisePredictor.basic diffusionConfig [2])

def residualDiffusion :
    nn.Sequential (diffusionConfig.inputShape [2]) (diffusionConfig.outputShape [2]) :=
  nn.build 16 (nn.models.Diffusion.NoisePredictor.residual diffusionConfig [2])

def ppoConfig : nn.models.PPO.Config :=
  { observationWidth := 2
    hiddenWidth := 2
    actionCount := 3 }

def actor : nn.Sequential (ppoConfig.inputShape [2]) (ppoConfig.actorOutputShape [2]) :=
  nn.build 17 (nn.models.PPO.actor ppoConfig [2])

def critic : nn.Sequential (ppoConfig.inputShape [2]) (ppoConfig.criticOutputShape [2]) :=
  nn.build 18 (nn.models.PPO.critic ppoConfig [2])

/-- Compare finite tensor entries by their binary64 representation, including the sign of zero. -/
def expectTensorBits {shape : Shape} (label : String) (actual expected : Tensor Float shape) :
    IO Unit := do
  let actual := actual.to (Array Float)
  let expected := expected.to (Array Float)
  expect s!"{label}: finite values" (actual.all Float.isFinite && expected.all Float.isFinite)
  expect s!"{label}: identical bits" (actual.map Float.toBits == expected.map Float.toBits)

/-- Compare explicit layer wiring bit for bit, including every state and input VJP. -/
def checkWiring {input output : Shape} (label : String)
    (actual reference : nn.Sequential input output) : IO Unit := do
  expect s!"{label}: validates" (nn.validate actual).isOk
  expect s!"{label}: reference validates" (nn.validate reference).isOk
  expect s!"{label}: trainable entries" (nn.requiresGrad actual == nn.requiresGrad reference)
  if same : nn.stateShapes actual = nn.stateShapes reference then
    let state : nn.State Float (nn.stateShapes actual) := nn.initialState actual
    let referenceState : nn.State Float (nn.stateShapes reference) := nn.initialState reference
    let alignedState := referenceState.cast same.symm
    for index in List.finRange (nn.stateShapes actual).length do
      expectTensorBits s!"{label}: initializer {index.val}"
        (state.get index) (alignedState.get index)
    let inputValue : Tensor Float input :=
      Tensor.generateFlat _ fun i => (i % 11).toFloat / 7 - 0.4
    let cotangent : Tensor Float output :=
      Tensor.generateFlat _ fun i => (i % 5).toFloat / 3 - 0.2
    let graph ← nn.lowerToTypedGraph (α := Float) actual
    let referenceGraph ← nn.lowerToTypedGraph (α := Float) reference
    let actualOutput := nn.TypedGraphModel.forward graph state inputValue
    let expectedOutput := nn.TypedGraphModel.forward referenceGraph referenceState inputValue
    expectTensorBits s!"{label}: complete output" actualOutput expectedOutput
    let (gradient, inputGradient) := nn.TypedGraphModel.vjp graph state inputValue cotangent
    let (referenceGradient, referenceInputGradient) :=
      nn.TypedGraphModel.vjp referenceGraph referenceState inputValue cotangent
    expectTensorBits s!"{label}: complete input VJP" inputGradient referenceInputGradient
    let referenceGradient := referenceGradient.cast same.symm
    for index in List.finRange (nn.stateShapes actual).length do
      expectTensorBits s!"{label}: complete parameter VJP {index.val}"
        (gradient.get index) (referenceGradient.get index)
  else
    throw <| IO.userError s!"{label}: model state layout differs from explicit wiring"

/-- Stage lists preserve singleton wiring and carry independently chosen spatial geometry. -/
def checkCnnStages : IO Unit := do
  let reference : nn.Sequential (cnnConfig.inputShape [2]) (cnnConfig.outputShape [2]) :=
    nn.build 1 <| nn.Sequential![
      nn.conv ([4] : Tensor Nat [1])
        { outChannels := 2, kernelSize := [3], padding := [1] }
        (batchShape := [2]) (inputChannels := 1),
      nn.relu,
      nn.maxPool ([4] : Tensor Nat [1]) { kernelSize := [2], stride := [2] }
        (batchShape := [2]) (channels := 2),
      nn.flattenAfter [2] (shape := [2, 2]),
      nn.linear 4 3 (batchShape := [2])
    ]
  checkWiring "CNN singleton" cnn reference
  let config : nn.models.CNN.Config 1 :=
    { inputChannels := 1, spatial := [8], classCount := 2
      stages :=
        [{ block := { convolution :=
             { outChannels := 2, kernelSize := [3], stride := [2], padding := [1] } }
           pooling := { kernelSize := [1] } },
         { block := { convolution := { outChannels := 3, kernelSize := [1] } }
           pooling := { kernelSize := [2], stride := [2] } }] }
  let reference : nn.Sequential (config.inputShape [2, 1]) (config.outputShape [2, 1]) :=
    nn.build 21 <| nn.Sequential![
      nn.conv ([8] : Tensor Nat [1])
        { outChannels := 2, kernelSize := [3], stride := [2], padding := [1] }
        (batchShape := [2, 1]) (inputChannels := 1),
      nn.relu,
      nn.maxPool ([4] : Tensor Nat [1]) { kernelSize := [1] }
        (batchShape := [2, 1]) (channels := 2),
      nn.conv ([4] : Tensor Nat [1]) { outChannels := 3, kernelSize := [1] }
        (batchShape := [2, 1]) (inputChannels := 2),
      nn.relu,
      nn.maxPool ([4] : Tensor Nat [1]) { kernelSize := [2], stride := [2] }
        (batchShape := [2, 1]) (channels := 3),
      nn.flattenAfter [2, 1] (shape := [3, 2]),
      nn.linear 6 2 (batchShape := [2, 1])
    ]
  checkWiring "CNN independent stages" (nn.build 21 (nn.models.cnn config [2, 1])) reference
  let headOnly : nn.models.CNN.Config 1 := { config with stages := [] }
  let reference : nn.Sequential (headOnly.inputShape [2, 1]) (headOnly.outputShape [2, 1]) :=
    nn.build 21 <| nn.Sequential![
      nn.flattenAfter [2, 1] (shape := [1, 8]), nn.linear 8 2 (batchShape := [2, 1])
    ]
  checkWiring "CNN head only" (nn.build 21 (nn.models.cnn headOnly [2, 1])) reference

/-- Recurrent depth and width choices preserve layer order and all parameter sensitivities. -/
def checkRecurrentStages : IO Unit := do
  checkWiring "RNN singleton" rnn <| nn.build 8 <| nn.Sequential![
    nn.rnn 2 2 2 (batchShape := [2]), nn.linear 2 1 (batchShape := [2, 2])
  ]
  checkWiring "GRU singleton" gru <| nn.build 9 <| nn.Sequential![
    nn.gru 2 2 2 (batchShape := [2]), nn.linear 2 1 (batchShape := [2, 2])
  ]
  checkWiring "LSTM singleton" lstm <| nn.build 10 <| nn.Sequential![
    nn.lstm 2 2 2 (batchShape := [2]), nn.linear 2 1 (batchShape := [2, 2])
  ]
  let config := { recurrentConfig with hiddenWidths := [3, 1, 2] }
  checkWiring "RNN independent widths" (nn.build 22 (nn.models.rnn config [2, 1])) <|
    nn.build 22 <| nn.Sequential![
      nn.rnn 2 2 3 (batchShape := [2, 1]), nn.rnn 2 3 1 (batchShape := [2, 1]),
      nn.rnn 2 1 2 (batchShape := [2, 1]), nn.linear 2 1 (batchShape := [2, 1, 2])
    ]
  checkWiring "GRU independent widths" (nn.build 22 (nn.models.gru config [2, 1])) <|
    nn.build 22 <| nn.Sequential![
      nn.gru 2 2 3 (batchShape := [2, 1]), nn.gru 2 3 1 (batchShape := [2, 1]),
      nn.gru 2 1 2 (batchShape := [2, 1]), nn.linear 2 1 (batchShape := [2, 1, 2])
    ]
  checkWiring "LSTM independent widths" (nn.build 22 (nn.models.lstm config [2, 1])) <|
    nn.build 22 <| nn.Sequential![
      nn.lstm 2 2 3 (batchShape := [2, 1]), nn.lstm 2 3 1 (batchShape := [2, 1]),
      nn.lstm 2 1 2 (batchShape := [2, 1]), nn.linear 2 1 (batchShape := [2, 1, 2])
    ]
  for sequenceLength in [0, 2] do
    let config : nn.models.Recurrent.Config :=
      { recurrentConfig with sequenceLength, hiddenWidths := [] }
    let head : nn.Sequential (config.inputShape [2, 1]) (config.outputShape [2, 1]) := by
      simpa [nn.models.Recurrent.Config.inputShape, nn.models.Recurrent.Config.outputShape,
        config, recurrentConfig] using
        (nn.build 23 (nn.linear 2 1 (batchShape := [2, 1, sequenceLength])))
    checkWiring s!"RNN head only, length {sequenceLength}"
      (nn.build 23 (nn.models.rnn config [2, 1])) head
    checkWiring s!"GRU head only, length {sequenceLength}"
      (nn.build 23 (nn.models.gru config [2, 1] (convention := .resetAfter))) head
    checkWiring s!"LSTM head only, length {sequenceLength}"
      (nn.build 23 (nn.models.lstm config [2, 1])) head

/-- Mamba stacks share core options; a head-only model never inspects those options. -/
def checkMambaStages : IO Unit := do
  checkWiring "Mamba singleton" mamba <| nn.build 11 <| nn.Sequential![
    nn.mamba 2 2 2 (batchShape := [2]), nn.linear 2 2 (batchShape := [2, 2])
  ]
  let config : nn.models.Mamba.Config :=
    { vocabularySize := 2, modelWidths := [3, 2]
      expansion := 1, stateWidth := 2, kernelWidth := 2 }
  checkWiring "Mamba independent widths"
    (nn.build 24 (nn.models.Mamba.languageModel config 2 [2, 1])) <|
    nn.build 24 <| nn.Sequential![
      nn.mamba 2 2 3 (batchShape := [2, 1]) (options := config.options),
      nn.mamba 2 3 2 (batchShape := [2, 1]) (options := config.options),
      nn.linear 2 2 (batchShape := [2, 1, 2])
    ]
  let headOnly : nn.models.Mamba.Config :=
    { vocabularySize := 2, expansion := 0, stateWidth := 0, kernelWidth := 0 }
  for sequenceLength in [0, 2] do
    let head : nn.Sequential
        (headOnly.inputShape sequenceLength [2, 1])
        (headOnly.outputShape sequenceLength [2, 1]) := by
      simpa [nn.models.Mamba.Config.inputShape, nn.models.Mamba.Config.outputShape, headOnly] using
        (nn.build 25 (nn.linear 2 2 (batchShape := [2, 1, sequenceLength])))
    checkWiring s!"Mamba head only, length {sequenceLength}"
      (nn.build 25 (nn.models.Mamba.languageModel headOnly sequenceLength [2, 1])) head

/-- Identity stages preserve the existing two-block classifier wiring. -/
def checkResNetIdentityStages : IO Unit := do
  let convolution : nn.Convolution.Config 1 :=
    { outChannels := 2, kernelSize := [3], padding := [1] }
  have spatialShape : Tensor.to ([4] : Tensor Nat [1]) Shape = [4] := by decide
  have outputShape :
      Shape.concat [2]
          (((convolution.outputSpatial ([4] : Tensor Nat [1])).to Shape).prependDim
            convolution.outChannels) = [2, 2, 4] := by
    decide
  let stem : nn.Builder (nn.Sequential [2, 1, 4] [2, 2, 4]) := by
    simpa only [outputShape, spatialShape] using
      (nn.conv ([4] : Tensor Nat [1]) convolution (batchShape := [2]) (inputChannels := 1))
  let hidden : nn.Builder (nn.Sequential [2, 2, 4] [2, 2, 4]) := by
    simpa only [outputShape, spatialShape] using
      (nn.conv ([4] : Tensor Nat [1]) convolution (batchShape := [2]) (inputChannels := 2))
  let residualBranch : nn.Builder (nn.Sequential [2, 2, 4] [2, 2, 4]) := do
    let branch ← nn.Sequential![hidden, nn.relu, hidden]
    pure (nn.residual branch)
  let reference : nn.Sequential (resnetConfig.inputShape [2]) (resnetConfig.outputShape [2]) :=
    nn.build 2 <| nn.Sequential![
      stem, nn.relu, residualBranch, nn.relu, residualBranch, nn.relu,
      nn.globalAvgPool ([4] : Tensor Nat [1]) (batchShape := [2]) (channels := 2),
      nn.linear 2 3 (batchShape := [2])
    ]
  checkWiring "ResNet identity stages" resnet reference

/-- Projected stages keep main-branch-first addition and parameter order during downsampling. -/
def checkResNetProjectedStages : IO Unit := do
  let convolution : nn.Convolution.Config 1 :=
    { outChannels := 2, kernelSize := [3], padding := [1] }
  have spatialShape : Tensor.to ([4] : Tensor Nat [1]) Shape = [4] := by decide
  have downsampledShape : Tensor.to ([2] : Tensor Nat [1]) Shape = [2] := by decide
  let stem : nn.Builder (nn.Sequential [2, 1, 1, 4] [2, 1, 2, 4]) := by
    have outputShape :
        Shape.concat [2, 1]
            (((convolution.outputSpatial ([4] : Tensor Nat [1])).to Shape).prependDim
              convolution.outChannels) = [2, 1, 2, 4] := by
      decide
    simpa only [outputShape, spatialShape] using
      (nn.conv ([4] : Tensor Nat [1]) convolution (batchShape := [2, 1]) (inputChannels := 1))
  let first : nn.Convolution.Config 1 :=
    { outChannels := 3, kernelSize := [3], stride := [2], padding := [1] }
  let second : nn.Convolution.Config 1 :=
    { outChannels := 3, kernelSize := [3], padding := [1] }
  let shortcut : nn.Convolution.Config 1 :=
    { outChannels := 3, kernelSize := [1], stride := [2] }
  let config : nn.models.ResNet.Config 1 :=
    { inputChannels := 1, spatial := [4], stem := convolution, classCount := 2
      stages := [{ first, second, shortcut := .projection shortcut }] }
  let downsample : nn.Builder (nn.Sequential [2, 1, 2, 4] [2, 1, 3, 2]) := by
    have outputShape :
        Shape.concat [2, 1]
            (((first.outputSpatial ([4] : Tensor Nat [1])).to Shape).prependDim
              first.outChannels) = [2, 1, 3, 2] := by
      decide
    simpa only [outputShape, spatialShape] using
      (nn.conv ([4] : Tensor Nat [1]) first (batchShape := [2, 1]) (inputChannels := 2))
  let hidden : nn.Builder (nn.Sequential [2, 1, 3, 2] [2, 1, 3, 2]) := by
    have outputShape :
        Shape.concat [2, 1]
            (((second.outputSpatial ([2] : Tensor Nat [1])).to Shape).prependDim
              second.outChannels) = [2, 1, 3, 2] := by
      decide
    simpa only [outputShape, downsampledShape] using
      (nn.conv ([2] : Tensor Nat [1]) second (batchShape := [2, 1]) (inputChannels := 3))
  let projectedBranch : nn.Builder (nn.Sequential [2, 1, 2, 4] [2, 1, 3, 2]) := do
    let branch ← nn.Sequential![downsample, nn.relu, hidden]
    let projected : nn.Sequential [2, 1, 2, 4] [2, 1, 3, 2] ← by
      have outputShape :
          Shape.concat [2, 1]
              (((shortcut.outputSpatial ([4] : Tensor Nat [1])).to Shape).prependDim
                shortcut.outChannels) = [2, 1, 3, 2] := by
        decide
      simpa only [outputShape, spatialShape] using
        (nn.conv ([4] : Tensor Nat [1]) shortcut
          (batchShape := [2, 1]) (inputChannels := 2))
    pure (nn.addBranches branch projected)
  let reference : nn.Sequential (config.inputShape [2, 1]) (config.outputShape [2, 1]) :=
    nn.build 26 <| nn.Sequential![
      stem, nn.relu, projectedBranch, nn.relu,
      nn.globalAvgPool ([2] : Tensor Nat [1]) (batchShape := [2, 1]) (channels := 3),
      nn.linear 3 2 (batchShape := [2, 1])
    ]
  checkWiring "ResNet projected downsampling"
    (nn.build 26 (nn.models.resnet config [2, 1])) reference
  let noStages : nn.models.ResNet.Config 1 := { config with stages := [] }
  let reference : nn.Sequential (noStages.inputShape [2, 1]) (noStages.outputShape [2, 1]) :=
    nn.build 26 <| nn.Sequential![
      stem, nn.relu,
      nn.globalAvgPool ([4] : Tensor Nat [1]) (batchShape := [2, 1]) (channels := 2),
      nn.linear 2 2 (batchShape := [2, 1])
    ]
  checkWiring "ResNet empty stages" (nn.build 26 (nn.models.resnet noStages [2, 1])) reference

def run : IO Unit := do
  checkCnnStages
  checkRecurrentStages
  checkMambaStages
  checkResNetIdentityStages
  checkResNetProjectedStages
  checkModuleModeOverride
  checkIndexedModeOverride
  runModel "CNN" cnn
  runModel "ResNet" resnet
  runModel "mean-pooled ViT" meanVit
  runModel "class-token ViT" classVit
  runModel "masked patch reconstructor" maskedPatchReconstructor
  runModel "KAN" kan
  runModel "FNO" fno
  runModel "RNN" rnn
  runModel "GRU" gru
  runModel "LSTM" lstm
  runModel "Mamba" mamba
  runModel "causal Transformer" causalTransformer
  checkCausalTransformerFutureIsolation
  runModel "autoencoder" autoencoder
  runModel "generator" generator
  runModel "discriminator" discriminator
  let autoencoderSummary ←
    match nn.summary autoencoder with
    | .ok summary => pure summary
    | .error message => throw <| IO.userError message
  let generatorSummary ←
    match nn.summary generator with
    | .ok summary => pure summary
    | .error message => throw <| IO.userError message
  let discriminatorSummary ←
    match nn.summary discriminator with
    | .ok summary => pure summary
    | .error message => throw <| IO.userError message
  expect "generic autoencoder leaves its output activation to the caller"
    (!(autoencoderSummary.layers.any fun layer => layer.kind == "Sigmoid"))
  expect "generic generator leaves its output activation to the caller"
    (!(generatorSummary.layers.any fun layer => layer.kind == "Sigmoid"))
  expect "generic discriminator returns logits"
    (!(discriminatorSummary.layers.any fun layer => layer.kind == "Sigmoid"))
  runModel "basic diffusion predictor" basicDiffusion
  runModel "residual diffusion predictor" residualDiffusion
  runModel "PPO actor" actor
  runModel "PPO critic" critic

end NN.Tests.API.Models

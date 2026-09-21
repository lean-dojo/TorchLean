/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Models
public import NN.API.Module
public import NN.API.Neural.Summary

/-!
# Public Model Execution Tests

Compact CPU forward checks for every ordinary public architecture family.
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
  let result ← module.predict (Tensor.zeros input)
  let values := Tensor.to result (Array Float)
  expect s!"{label} returns the exact output size" (values.size == output.size)
  expect s!"{label} returns finite values" (values.all Float.isFinite)

def checkModulePredictPreservesMode : IO Unit := do
  let model : nn.Sequential [2] [2] :=
    nn.build 0 nn.relu
  let module ← nn.Module.instantiate model (α := Float)
  let input : Tensor Float [2] := [-1.0, 1.0]
  expect "ordinary module starts in training mode" (← module.isTraining)
  let _ ← module.predict input
  expect "ordinary predict preserves training mode" (← module.isTraining)
  module.eval
  let _ ← module.predict input
  expect "ordinary predict preserves evaluation mode" (!(← module.isTraining))

def indexedModel : nn.IndexedModel [2] [2, 2] (Fin 3) :=
  (nn.build 0 (nn.embedding 3 2)).model [2]

def checkIndexedPredictPreservesMode : IO Unit := do
  let module ← nn.IndexedModule.instantiate indexedModel (α := Float)
  let input : Tensor (Fin 3) [2] := [0, 2]
  expect "indexed module starts in training mode" (← module.isTraining)
  let _ ← module.predict input
  expect "indexed predict preserves training mode" (← module.isTraining)
  module.eval
  let _ ← module.predict input
  expect "indexed predict preserves evaluation mode" (!(← module.isTraining))

def cnnConfig : nn.models.CNN.Config 1 :=
  { inputChannels := 1
    spatial := [4]
    convolution :=
      { outChannels := 2
        kernelSize := [3]
        padding := [1] }
    pooling :=
      { kernelSize := [2]
        stride := [2] }
    classCount := 3 }

def cnn : nn.Sequential (cnnConfig.inputShape [2]) (cnnConfig.outputShape [2]) :=
  nn.build 1 (nn.models.cnn cnnConfig [2])

def resnetConfig : nn.models.ResNet.Config 1 :=
  { inputChannels := 1
    spatial := [4]
    hiddenChannels := 2
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
    hiddenWidth := 2
    outputWidth := 1 }

def rnn : nn.Sequential (recurrentConfig.inputShape [2]) (recurrentConfig.outputShape [2]) :=
  nn.build 8 (nn.models.rnn recurrentConfig [2])

def gru : nn.Sequential (recurrentConfig.inputShape [2]) (recurrentConfig.outputShape [2]) :=
  nn.build 9 (nn.models.gru recurrentConfig [2])

def lstm : nn.Sequential (recurrentConfig.inputShape [2]) (recurrentConfig.outputShape [2]) :=
  nn.build 10 (nn.models.lstm recurrentConfig [2])

def mambaConfig : nn.models.Mamba.Config :=
  { vocabularySize := 2
    modelWidth := 2 }

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
  let first ← module.predict
    (causalInput
      (Fin.ofNat causalTransformerConfig.vocabularySize 1)
      (Fin.ofNat causalTransformerConfig.vocabularySize 2))
  let changedFuture ← module.predict
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

def run : IO Unit := do
  checkModulePredictPreservesMode
  checkIndexedPredictPreservesMode
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

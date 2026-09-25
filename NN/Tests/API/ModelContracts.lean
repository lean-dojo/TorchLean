/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Models.FNO
public import NN.API.Models.Cnn
public import NN.API.Models.ResNet
public import NN.API.Models.Diffusion
public import NN.API.Models.KAN
public import NN.API.Models.SelfSupervised
public import NN.API.Models.Unet
public import NN.API.Autograd.Model
public import NN.API.Neural.Execution
public import NN.API.Trainer.Core
public import NN.Spec.Module.RnnModels
public import NN.Spec.Module.GruModels
public import NN.Spec.Module.LstmModels

/-!
# Public Model Contract Tests

Regression checks for model state and endpoint behavior that are easy to lose during builder
refactors.
-/

@[expose] public section

namespace NN.Tests.API.ModelContracts

open TorchLean

def expect (label : String) (condition : Bool) : IO Unit := do
  unless condition do
    throw <| IO.userError s!"public model contract failed: {label}"

def vitEncoderConfig (pooling : nn.models.ViT.Pooling) : nn.models.ViT.EncoderConfig 1 :=
  { inputChannels := 1
    spatial := [2]
    patchEmbedding :=
      { outChannels := 3
        kernelSize := [1] }
    headCount := 1
    headWidth := 3
    feedForwardWidth := 4
    layerCount := 0
    pooling }

def vitConfig (pooling : nn.models.ViT.Pooling) : nn.models.ViT.Config 1 :=
  (vitEncoderConfig pooling).classifier 2

def meanVit :=
  nn.build 7 (nn.models.vitEncoder (vitEncoderConfig .mean))

def clsVit :=
  nn.build 7 (nn.models.vitEncoder (vitEncoderConfig .cls))

def emptyPatchVitConfig (pooling : nn.models.ViT.Pooling) :
    nn.models.ViT.EncoderConfig 1 :=
  { vitEncoderConfig pooling with spatial := [0] }

def emptyMeanVit :=
  nn.build 7 (nn.models.vitEncoder (emptyPatchVitConfig .mean))

def emptyClassVit :=
  nn.build 7 (nn.models.vitEncoder (emptyPatchVitConfig .cls))

def paddedEmptyVitConfig : nn.models.ViT.EncoderConfig 1 :=
  { vitEncoderConfig .mean with
    spatial := [0]
    patchEmbedding :=
      { outChannels := 3
        kernelSize := [1]
        padding := [1] } }

def paddedEmptyVit :=
  nn.build 7 (nn.models.vitEncoder paddedEmptyVitConfig)

def emptyInputCnnConfig : nn.models.CNN.Config 1 :=
  { inputChannels := 1
    spatial := [0]
    stages :=
      [{ block :=
           { convolution :=
               { outChannels := 2
                 kernelSize := [1]
                 padding := [1] } }
         pooling :=
           { kernelSize := [1] } }]
    classCount := 2 }

def emptyInputCnn :=
  nn.build 7 (nn.models.cnn emptyInputCnnConfig)

def emptyAfterConvCnnConfig : nn.models.CNN.Config 1 :=
  { inputChannels := 1
    spatial := [1]
    stages :=
      [{ block :=
           { convolution :=
               { outChannels := 2
                 kernelSize := [3] } }
         pooling :=
           { kernelSize := [1] } }]
    classCount := 2 }

def emptyAfterConvCnn :=
  nn.build 7 (nn.models.cnn emptyAfterConvCnnConfig)

def emptyAfterPoolCnnConfig : nn.models.CNN.Config 1 :=
  { inputChannels := 1
    spatial := [2]
    stages :=
      [{ block :=
           { convolution :=
               { outChannels := 2
                 kernelSize := [1] } }
         pooling :=
           { kernelSize := [3] } }]
    classCount := 2 }

def emptyAfterPoolCnn :=
  nn.build 7 (nn.models.cnn emptyAfterPoolCnnConfig)

def emptyInputConv :=
  nn.build 7
    (nn.conv (inputChannels := 1) ([0] : Tensor Nat [1])
      { outChannels := 2
        kernelSize := [1]
        padding := [1] })

def emptyOutputConv :=
  nn.build 7
    (nn.conv (inputChannels := 1) ([1] : Tensor Nat [1])
      { outChannels := 2
        kernelSize := [3] })

def emptyOutputConvTranspose :=
  nn.build 7
    (nn.convTranspose (inputChannels := 1) ([1] : Tensor Nat [1])
      { outChannels := 2
        kernelSize := [1]
        padding := [1] })

def zeroChannelMaxPool :=
  nn.build 7
    (nn.maxPool (channels := 0) ([2] : Tensor Nat [1])
      { kernelSize := [1] })

def emptyInputMaxPool :=
  nn.build 7
    (nn.maxPool (channels := 1) ([0] : Tensor Nat [1])
      { kernelSize := [1] })

def emptyOutputAvgPool :=
  nn.build 7
    (nn.avgPool (channels := 1) ([1] : Tensor Nat [1])
      { kernelSize := [3] })

def emptyInputResNetConfig : nn.models.ResNet.Config 1 :=
  let convolution : nn.Convolution.Config 1 :=
    { outChannels := 2, kernelSize := [3], padding := [1] }
  { inputChannels := 1
    spatial := [0]
    stem := convolution
    stages := List.replicate 2 { first := convolution, second := convolution }
    classCount := 2 }

def emptyInputResNet :=
  nn.build 7 (nn.models.resnet emptyInputResNetConfig)

def emptyDiffusionConfig : nn.models.Diffusion.NoisePredictor.Config 1 :=
  { dataChannels := 1
    spatial := [0]
    hiddenChannels := 2 }

def emptyBasicDiffusion :=
  nn.build 7 (nn.models.Diffusion.NoisePredictor.basic emptyDiffusionConfig)

def emptyResidualDiffusion :=
  nn.build 7 (nn.models.Diffusion.NoisePredictor.residual emptyDiffusionConfig)

def zeroGridFnoConfig : nn.models.FNO.Config 1 :=
  { spatial := [0]
    modes := [0]
    width := 1
    layerCount := 1 }

def zeroGridFno :=
  nn.build 7 (nn.models.fno zeroGridFnoConfig)

def zeroWidthFnoConfig : nn.models.FNO.Config 1 :=
  { spatial := [1]
    modes := [1]
    width := 0
    layerCount := 1 }

def zeroWidthFno :=
  nn.build 7 (nn.models.fno zeroWidthFnoConfig)

def unetConfig : nn.models.UNet.Config 1 :=
  { inputChannels := 1
    baseChannels := 2
    outputChannels := 1
    spatial := [4]
    pooling :=
      { kernelSize := [2]
        stride := [2] }
    upsampling :=
      { kernelSize := [2]
        stride := [2] } }

def unet : nn.Sequential (unetConfig.inputShape) (unetConfig.outputShape) :=
  nn.build 19 (nn.models.unet unetConfig)

def batchedUnet : nn.Sequential (unetConfig.inputShape [2]) (unetConfig.outputShape [2]) :=
  nn.build 19 (nn.models.unet unetConfig [2])

def paddedEmptyUnetConfig : nn.models.UNet.Config 1 :=
  { unetConfig with
    spatial := [0]
    pooling :=
      { kernelSize := [1]
        stride := [1]
        padding := [1] }
    upsampling :=
      { kernelSize := [1]
        stride := [1]
        padding := [1] } }

def paddedEmptyUnet :=
  nn.build 19 (nn.models.unet paddedEmptyUnetConfig)

/- These definitions intentionally use a config-derived shape. They are compile-time regression
checks that public axis APIs accept model shapes without proof arguments or list conversion. -/
abbrev classifierOutput : Spec.Shape :=
  (vitConfig .mean).outputShape [4]

def classifierSoftmax : nn.Sequential classifierOutput classifierOutput :=
  nn.build 7 (nn.softmax (shape := classifierOutput) 1)

def classifierObjective : Trainer.Objective classifierOutput :=
  .oneHotCrossEntropy 1

def classifierLoss :
    autograd.model.Loss classifierOutput classifierOutput :=
  autograd.model.Loss.oneHotCrossEntropy 1

def invalidClassifierLoss :
    autograd.model.Loss classifierOutput classifierOutput :=
  autograd.model.Loss.oneHotCrossEntropy classifierOutput.rank

def fullDropout : nn.Sequential [3] [3] :=
  nn.build 11 (nn.dropout (shape := [3]) 1.0)

def zeroDropout : nn.Sequential [3] [3] :=
  nn.build 11 (nn.dropout (shape := [3]) 0.0)

def negativeDropout : nn.Sequential [3] [3] :=
  nn.build 11 (nn.dropout (shape := [3]) (-0.1))

def oversizedDropout : nn.Sequential [3] [3] :=
  nn.build 11 (nn.dropout (shape := [3]) 1.1)

def nanDropout : nn.Sequential [3] [3] :=
  nn.build 11 (nn.dropout (shape := [3]) (Float.ofBits 0x7ff8000000000000))

def infiniteDropout : nn.Sequential [3] [3] :=
  nn.build 11 (nn.dropout (shape := [3]) (Float.ofBits 0x7ff0000000000000))

def nestedInvalidDropout : nn.Sequential [3] [3] :=
  nn.residual negativeDropout

def zeroMomentumBatchNorm : nn.Sequential [1, 1] [1, 1] :=
  nn.build 23 (nn.batchNorm ([1] : Tensor Nat [1]) (channels := 1)
    (momentum := 0.0))

def fullMomentumBatchNorm : nn.Sequential [1, 1] [1, 1] :=
  nn.build 23 (nn.batchNorm ([1] : Tensor Nat [1]) (channels := 1)
    (momentum := 1.0))

def negativeMomentumBatchNorm : nn.Sequential [1, 1] [1, 1] :=
  nn.build 23 (nn.batchNorm ([1] : Tensor Nat [1]) (channels := 1)
    (momentum := -0.1))

def oversizedMomentumBatchNorm : nn.Sequential [1, 1] [1, 1] :=
  nn.build 23 (nn.batchNorm ([1] : Tensor Nat [1]) (channels := 1)
    (momentum := 1.1))

def nanMomentumBatchNorm : nn.Sequential [1, 1] [1, 1] :=
  nn.build 23 (nn.batchNorm ([1] : Tensor Nat [1]) (channels := 1)
    (momentum := Float.ofBits 0x7ff8000000000000))

def invalidSoftmax : nn.Sequential [2, 3] [2, 3] :=
  nn.build 7 (nn.softmax (shape := [2, 3]) 2)

def invalidReshape : nn.Sequential [2, 2] [3] :=
  nn.build 7 (nn.reshape [2, 2] [3])

def zeroWidthLayerNorm : nn.Sequential [0] [0] :=
  nn.build 7 (nn.layerNorm (width := 0))

def zeroWidthRmsNorm : nn.Sequential [0] [0] :=
  nn.build 7 (nn.rmsNorm (width := 0))

def zeroChannelBatchNorm : nn.Sequential [0, 1] [0, 1] :=
  nn.build 7 (nn.batchNorm ([1] : Tensor Nat [1]) (channels := 0))

def emptySpatialInstanceNorm : nn.Sequential [1, 0] [1, 0] :=
  nn.build 7 (nn.instanceNorm ([0] : Tensor Nat [1]) (channels := 1))

def zeroGroupNorm : nn.Sequential [2, 1] [2, 1] :=
  nn.build 7 (nn.groupNorm ([1] : Tensor Nat [1]) 0 (channels := 2))

def nondivisibleGroupNorm : nn.Sequential [3, 1] [3, 1] :=
  nn.build 7 (nn.groupNorm ([1] : Tensor Nat [1]) 2 (channels := 3))

def attentionConfig : nn.MultiHeadAttention.Config :=
  { headCount := 1, headWidth := 4 }

def zeroSequenceAttention : nn.Sequential [0, 4] [0, 4] :=
  nn.build 7
    (nn.multiHeadAttention (sequenceLength := 0) (modelWidth := 4) attentionConfig)

def zeroModelWidthAttention : nn.Sequential [2, 0] [2, 0] :=
  nn.build 7
    (nn.multiHeadAttention (sequenceLength := 2) (modelWidth := 0) attentionConfig)

def zeroHeadCountAttention : nn.Sequential [2, 4] [2, 4] :=
  nn.build 7
    (nn.multiHeadAttention (sequenceLength := 2) (modelWidth := 4)
      { headCount := 0, headWidth := 4 })

def zeroHeadWidthBiasedAttention : nn.Sequential [2, 4] [2, 4] :=
  nn.build 7
    (nn.multiHeadAttention (sequenceLength := 2) (modelWidth := 4)
      { headCount := 1, headWidth := 0, outputBias := true })

def transformerConfig : nn.TransformerEncoder.Stack.Config :=
  { layerCount := 0
    block :=
      { headCount := 1
        headWidth := 4
        feedForwardWidth := 4 } }

def zeroSequenceTransformer : nn.Sequential [0, 4] [0, 4] :=
  nn.build 7
    (nn.transformerEncoderStack
      (sequenceLength := 0) (modelWidth := 4) transformerConfig)

def zeroWidthTransformer : nn.Sequential [2, 0] [2, 0] :=
  nn.build 7
    (nn.transformerEncoderStack
      (sequenceLength := 2) (modelWidth := 0) transformerConfig)

def zeroHeadCountTransformerConfig : nn.TransformerEncoder.Stack.Config :=
  { layerCount := 0
    block :=
      { headCount := 0
        headWidth := 4
        feedForwardWidth := 4 } }

def zeroHeadCountTransformer : nn.Sequential [2, 4] [2, 4] :=
  nn.build 7
    (nn.transformerEncoderStack
      (sequenceLength := 2) (modelWidth := 4) zeroHeadCountTransformerConfig)

def zeroHeadWidthTransformerConfig : nn.TransformerEncoder.Stack.Config :=
  { layerCount := 0
    block :=
      { headCount := 1
        headWidth := 0
        feedForwardWidth := 4 } }

def zeroHeadWidthTransformer : nn.Sequential [2, 4] [2, 4] :=
  nn.build 7
    (nn.transformerEncoderStack
      (sequenceLength := 2) (modelWidth := 4) zeroHeadWidthTransformerConfig)

def zeroFeedForwardTransformerConfig : nn.TransformerEncoder.Stack.Config :=
  { layerCount := 0
    block :=
      { headCount := 1
        headWidth := 4
        feedForwardWidth := 0 } }

def zeroFeedForwardTransformer : nn.Sequential [2, 4] [2, 4] :=
  nn.build 7
    (nn.transformerEncoderStack
      (sequenceLength := 2) (modelWidth := 4) zeroFeedForwardTransformerConfig)

def invalidDropoutTransformerConfig : nn.TransformerEncoder.Stack.Config :=
  { layerCount := 0
    block :=
      { headCount := 1
        headWidth := 4
        feedForwardWidth := 4
        dropout? := some (-0.1) } }

def invalidDropoutTransformer : nn.Sequential [2, 4] [2, 4] :=
  nn.build 7
    (nn.transformerEncoderStack
      (sequenceLength := 2) (modelWidth := 4) invalidDropoutTransformerConfig)

def invalidInitializationTransformerConfig : nn.TransformerEncoder.Stack.Config :=
  { layerCount := 0
    block :=
      { headCount := 1
        headWidth := 4
        feedForwardWidth := 4
        weightInitialization? := some (.normal 0.0 (-1.0)) } }

def invalidInitializationTransformer : nn.Sequential [2, 4] [2, 4] :=
  nn.build 7
    (nn.transformerEncoderStack
      (sequenceLength := 2) (modelWidth := 4) invalidInitializationTransformerConfig)

def zeroInputLinear : nn.Sequential [0] [2] :=
  nn.build 7 (nn.linear 0 2)

def zeroOutputLinear : nn.Sequential [2] [0] :=
  nn.build 7 (nn.linear 2 0)

def zeroSequenceRnn : nn.Sequential [0, 2] [0, 3] :=
  nn.build 7 (nn.rnn 0 2 3)

def zeroInputGru : nn.Sequential [2, 0] [2, 3] :=
  nn.build 7 (nn.gru 2 0 3)

def zeroHiddenMamba : nn.Sequential [2, 3] [2, 0] :=
  nn.build 7 (nn.mamba 2 3 0)

def zeroSequenceLstm : nn.Sequential [0, 2] [0, 3] :=
  nn.build 7 (nn.lstm 0 2 3)

def invalidShallowMlp : nn.Sequential [2] [1] :=
  nn.build 7
    (nn.mlp 2 1
      { hiddenWidths := []
        dropout? := some (-0.1) })

def reversedUniformLinear : nn.Sequential [2] [2] :=
  nn.build 29 (nn.linear 2 2
    (config := { weightInitialization? := some (.uniform 1.0 (-1.0)) }))

def negativeStdLinear : nn.Sequential [2] [2] :=
  nn.build 29 (nn.linear 2 2
    (config := { biasInitialization := .normal 0.0 (-1.0) }))

def invalidEmbeddingTable : nn.Embedding 2 3 :=
  nn.build 31 (nn.embedding 2 3
    { weightInitialization := .normal 0.0 (-1.0) })

def invalidEmbedding : nn.IndexedModel [2] [2, 3] (Fin 2) :=
  invalidEmbeddingTable.model [2]

def validEmbeddingTable : nn.Embedding 2 3 :=
  nn.build 31 (nn.embedding 2 3)

def validEmbedding : nn.IndexedModel [2] [2, 3] (Fin 2) :=
  validEmbeddingTable.model [2]

def zeroVocabularyEmbeddingTable : nn.Embedding 0 3 :=
  nn.build 31 (nn.embedding 0 3)

def zeroVocabularyEmbedding : nn.IndexedModel [2] [2, 3] (Fin 0) :=
  zeroVocabularyEmbeddingTable.model [2]

def zeroWidthEmbeddingTable : nn.Embedding 2 0 :=
  nn.build 31 (nn.embedding 2 0)

def zeroWidthEmbedding : nn.IndexedModel [2] [2, 0] (Fin 2) :=
  zeroWidthEmbeddingTable.model [2]

def zeroVocabularyOneHotEmbedding : nn.Sequential [0] [3] :=
  nn.build 31 (nn.oneHotEmbedding 0 3)

def zeroWidthOneHotEmbedding : nn.Sequential [2] [0] :=
  nn.build 31 (nn.oneHotEmbedding 2 0)

def sinusoidalEncoding : nn.Sequential [3, 5] [3, 5] :=
  nn.build 13 (nn.sinusoidalPositionalEncoding
    (sequenceLength := 3) (embeddingWidth := 5) (config := { startPosition := 7 }))

def rotaryEncoding : nn.Sequential [3, 4] [3, 4] :=
  nn.build 17 (nn.rope
    (sequenceLength := 3) (headWidth := 4) (config := { startPosition := 5 }))

def oddWidthRotaryEncoding : nn.Sequential [2, 3] [2, 3] :=
  nn.build 17 (nn.rope
    (sequenceLength := 2) (headWidth := 3) (config := { startPosition := 5 }))

def zeroSequenceLearnedPosition : nn.Sequential [0, 4] [0, 4] :=
  nn.build 17 (nn.learnedPositionalEmbedding
    (sequenceLength := 0) (embeddingWidth := 4))

def zeroWidthLearnedPosition : nn.Sequential [2, 0] [2, 0] :=
  nn.build 17 (nn.learnedPositionalEmbedding
    (sequenceLength := 2) (embeddingWidth := 0))

def zeroSequenceSinusoidalPosition : nn.Sequential [0, 4] [0, 4] :=
  nn.build 17 (nn.sinusoidalPositionalEncoding
    (sequenceLength := 0) (embeddingWidth := 4))

def zeroWidthSinusoidalPosition : nn.Sequential [2, 0] [2, 0] :=
  nn.build 17 (nn.sinusoidalPositionalEncoding
    (sequenceLength := 2) (embeddingWidth := 0))

def zeroSequenceRope : nn.Sequential [0, 4] [0, 4] :=
  nn.build 17 (nn.rope (sequenceLength := 0) (headWidth := 4))

def zeroWidthRope : nn.Sequential [2, 0] [2, 0] :=
  nn.build 17 (nn.rope (sequenceLength := 2) (headWidth := 0))

def zeroChannelGlobalPool : nn.Sequential [0, 2] [0] :=
  nn.build 17 (nn.globalAvgPool ([2] : Tensor Nat [1]) (channels := 0))

def malformedTrainability : nn.Sequential [1] [1] :=
  nn.Sequential.fromLayer
    { Runtime.Autograd.Model.Layers.relu (s := [1]) with
      kind := "MalformedTrainability"
      requiresGrad := #[true]
    }

def malformedIndexed : nn.IndexedModel [2] [2, 3] (Fin 2) :=
  nn.IndexedModel.Internal.create
    validEmbedding.stateShapes
    validEmbedding.initialState
    (fun mode => nn.IndexedModel.Internal.program validEmbedding mode)
    (kind := "MalformedIndexed")
    (trainableMask := #[])

/-- Invalid builders must fail before their placeholder forward can produce derivatives. -/
def checkInvalidAutograd : IO Unit := do
  let model := nn.build 0 (nn.softmax (shape := [2]) 9)
  let state := autograd.model.initialState model (α := Float)
  let input : Tensor Float [2] := [1.0, 2.0]
  for action in [
      (do let _ ← autograd.model.vjp model state input input; pure ()),
      (do let _ ← autograd.model.jacrev model state input; pure ())] do
    let rejected ← try
      action
      pure false
    catch error =>
      pure (error.toString.contains "out of bounds")
    expect "autograd rejected invalid model before differentiation" rejected

/-- Parameter HVPs preserve the coupled weight and bias curvature. -/
def checkHigherDerivatives : IO Unit := do
  let model := nn.build 0 (nn.linear 1 1)
  let state : autograd.model.State model Float := autograd.model.initialState model
  let direction : autograd.model.State model Float := autograd.model.fullState model 1.0
  let curvature ← autograd.model.hvp model autograd.model.Loss.mse
    state ([2.0] : Tensor Float [1]) ([0.0] : Tensor Float [1]) direction
  let weight := curvature.get ⟨0, by decide⟩
  let bias := curvature.get ⟨1, by decide⟩
  expect "parameter Hessian weight direction" (Tensor.to weight (Array Float) == #[12.0])
  expect "parameter Hessian bias direction" (Tensor.to bias (Array Float) == #[6.0])

/-- Deterministic, nonconstant reference parameters and states. -/
def referenceTensor (shape : Shape) (offset : Float) : Tensor Float shape :=
  Tensor.generateFlat shape fun index => offset + (index % 7).toFloat / 31

def referenceRnn (input hidden : Nat) : Spec.RNNSpec Float input hidden :=
  { weights := referenceTensor _ 0.03, bias := referenceTensor _ (-0.02) }

def referenceGru (input hidden : Nat) : Spec.GRUSpec Float input hidden :=
  { resetWeight := referenceTensor _ 0.01, resetBias := referenceTensor _ 0.02
    updateWeight := referenceTensor _ (-0.03), updateBias := referenceTensor _ 0.04
    candidateWeight := referenceTensor _ 0.05, candidateBias := referenceTensor _ (-0.06) }

def referenceLstm (input hidden : Nat) : Spec.LSTMSpec Float input hidden :=
  { forgetWeight := referenceTensor _ 0.01, forgetBias := referenceTensor _ 0.02
    inputWeight := referenceTensor _ (-0.03), inputBias := referenceTensor _ 0.04
    candidateWeight := referenceTensor _ 0.05, candidateBias := referenceTensor _ (-0.06)
    outputWeight := referenceTensor _ 0.07, outputBias := referenceTensor _ 0.08 }

def referenceHead : Spec.LinearSpec Float 2 1 :=
  { weights := [[1.0, -0.25]], bias := [0.2] }

def expectTensor {shape : Shape} (label : String) (actual expected : Tensor Float shape) :
    IO Unit := do
  let actual := actual.to (Array Float)
  let expected := expected.to (Array Float)
  expect s!"{label}: finite values" (actual.all Float.isFinite && expected.all Float.isFinite)
  expect s!"{label}: identical bits" (actual.map Float.toBits == expected.map Float.toBits)

/-- Compare typed RNN stacks against explicit cell composition and state selection. -/
def checkRnnReferenceStack : IO Unit := do
  let first := referenceRnn 2 3
  let second := referenceRnn 3 1
  let third := referenceRnn 1 2
  let model : Spec.Rnn.StackedModel Float 2 2 1 :=
    { layers := .cons first (.cons second (.cons third .nil)), outputLayer := referenceHead }
  let input := referenceTensor [2, 2] 0.2
  let h₁ := referenceTensor [3] 0.1
  let h₂ := referenceTensor [1] (-0.2)
  let h₃ := referenceTensor [2] 0.3
  let firstOutput := Spec.rnnSequenceSpec first input h₁
  let secondOutput := Spec.rnnSequenceSpec second firstOutput h₂
  let thirdOutput := Spec.rnnSequenceSpec third secondOutput h₃
  let (output, finalStates) := model.forward input (h₁, (h₂, (h₃, ())))
  expectTensor "RNN independent widths: complete output" output <|
    Tensor.mapLeading [2] (Spec.linearSpec referenceHead) thirdOutput
  expectTensor "RNN first final state" finalStates.1 (Tensor.get firstOutput 1)
  expectTensor "RNN second final state" finalStates.2.1 (Tensor.get secondOutput 1)
  expectTensor "RNN third final state" finalStates.2.2.1 (Tensor.get thirdOutput 1)
  let (emptyOutput, emptyStates) :=
    model.forward (Tensor.zeros [0, 2]) (h₁, (h₂, (h₃, ())))
  expectTensor "RNN empty output" emptyOutput (Tensor.zeros [0, 1])
  expectTensor "RNN empty first state" emptyStates.1 h₁
  expectTensor "RNN empty second state" emptyStates.2.1 h₂
  expectTensor "RNN empty third state" emptyStates.2.2.1 h₃
  expectTensor "RNN chain uses the same zero-state stack"
    ((Spec.Rnn.stacked model.layers referenceHead).forward input)
    (model.forward input (Tensor.zeros _, (Tensor.zeros _, (Tensor.zeros _, ())))).1
  let oldCell := referenceRnn 2 2
  let oldChain : Spec.Module.Chain Float [2, 2] [2, 1] :=
    .comp (.single (Spec.Module.rnn oldCell))
      (.comp (.single (Spec.Module.rnn oldCell))
        (.single (Spec.Module.liftLeading (Spec.Module.linear referenceHead))))
  expectTensor "RNN two-layer chain preserves existing output"
    ((Spec.Rnn.stacked (.cons oldCell (.cons oldCell .nil)) referenceHead).forward input)
    (oldChain.forward input)
  let headOnly : Spec.Rnn.StackedModel Float 2 2 1 :=
    { layers := .nil, outputLayer := referenceHead }
  for sequenceLength in [0, 2] do
    let input := referenceTensor [sequenceLength, 2] 0.2
    expectTensor "RNN head-only reference" (headOnly.forward input ()).1 <|
      Tensor.mapLeading [sequenceLength] (Spec.linearSpec referenceHead) input

/-- GRU stacks preserve nonzero initial states at every independently sized layer. -/
def checkGruReferenceStack : IO Unit := do
  let first := referenceGru 2 3
  let second := referenceGru 3 1
  let third := referenceGru 1 2
  let model : Spec.Gru.StackedModel Float 2 2 1 :=
    { layers := .cons first (.cons second (.cons third .nil)), outputLayer := referenceHead }
  let input := referenceTensor [2, 2] 0.2
  let h₁ := referenceTensor [3] 0.1
  let h₂ := referenceTensor [1] (-0.2)
  let h₃ := referenceTensor [2] 0.3
  let firstOutput := Spec.gruSequenceSpec first input h₁
  let secondOutput := Spec.gruSequenceSpec second firstOutput h₂
  let thirdOutput := Spec.gruSequenceSpec third secondOutput h₃
  let (output, finalStates) := model.forward input (h₁, (h₂, (h₃, ())))
  expectTensor "GRU independent widths: complete output" output <|
    Tensor.mapLeading [2] (Spec.linearSpec referenceHead) thirdOutput
  expectTensor "GRU first final state" finalStates.1 (Tensor.get firstOutput 1)
  expectTensor "GRU second final state" finalStates.2.1 (Tensor.get secondOutput 1)
  expectTensor "GRU third final state" finalStates.2.2.1 (Tensor.get thirdOutput 1)
  let (emptyOutput, emptyStates) :=
    model.forward (Tensor.zeros [0, 2]) (h₁, (h₂, (h₃, ())))
  expectTensor "GRU empty output" emptyOutput (Tensor.zeros [0, 1])
  expectTensor "GRU empty first state" emptyStates.1 h₁
  expectTensor "GRU empty second state" emptyStates.2.1 h₂
  expectTensor "GRU empty third state" emptyStates.2.2.1 h₃
  expectTensor "GRU chain uses the same zero-state stack"
    ((Spec.Gru.stacked model.layers referenceHead).forward input)
    (model.forward input (Tensor.zeros _, (Tensor.zeros _, (Tensor.zeros _, ())))).1
  let oldCell := referenceGru 2 2
  let oldChain : Spec.Module.Chain Float [2, 2] [2, 1] :=
    .comp (.single (Spec.Module.gru oldCell))
      (.comp (.single (Spec.Module.gru oldCell))
        (.single (Spec.Module.liftLeading (Spec.Module.linear referenceHead))))
  expectTensor "GRU two-layer chain preserves existing output"
    ((Spec.Gru.stacked (.cons oldCell (.cons oldCell .nil)) referenceHead).forward input)
    (oldChain.forward input)
  let headOnly : Spec.Gru.StackedModel Float 2 2 1 :=
    { layers := .nil, outputLayer := referenceHead }
  for sequenceLength in [0, 2] do
    let input := referenceTensor [sequenceLength, 2] 0.2
    expectTensor "GRU head-only reference" (headOnly.forward input ()).1 <|
      Tensor.mapLeading [sequenceLength] (Spec.linearSpec referenceHead) input

def expectLstmState {width : Nat} (label : String)
    (actual expected : Spec.LSTMState Float width) : IO Unit := do
  expectTensor s!"{label}: hidden" actual.hidden expected.hidden
  expectTensor s!"{label}: cell" actual.cell expected.cell

/-- LSTM stacks retain both state components, including through empty sequences. -/
def checkLstmReferenceStack : IO Unit := do
  let first := referenceLstm 2 3
  let second := referenceLstm 3 1
  let third := referenceLstm 1 2
  let model : Spec.Lstm.StackedModel Float 2 2 1 :=
    { layers := .cons first (.cons second (.cons third .nil)), outputLayer := referenceHead }
  let input := referenceTensor [2, 2] 0.2
  let h₁ : Spec.LSTMState Float 3 :=
    { hidden := referenceTensor _ 0.1, cell := referenceTensor _ (-0.1) }
  let h₂ : Spec.LSTMState Float 1 :=
    { hidden := referenceTensor _ (-0.2), cell := referenceTensor _ 0.2 }
  let h₃ : Spec.LSTMState Float 2 :=
    { hidden := referenceTensor _ 0.3, cell := referenceTensor _ (-0.3) }
  let (firstOutput, firstState) := Spec.lstmSequenceSpec first input h₁
  let (secondOutput, secondState) := Spec.lstmSequenceSpec second firstOutput h₂
  let (thirdOutput, thirdState) := Spec.lstmSequenceSpec third secondOutput h₃
  let (output, finalStates) := model.forward input (h₁, (h₂, (h₃, ())))
  expectTensor "LSTM independent widths: complete output" output <|
    Tensor.mapLeading [2] (Spec.linearSpec referenceHead) thirdOutput
  expectLstmState "LSTM first final state" finalStates.1 firstState
  expectLstmState "LSTM second final state" finalStates.2.1 secondState
  expectLstmState "LSTM third final state" finalStates.2.2.1 thirdState
  let (emptyOutput, emptyStates) :=
    model.forward (Tensor.zeros [0, 2]) (h₁, (h₂, (h₃, ())))
  expectTensor "LSTM empty output" emptyOutput (Tensor.zeros [0, 1])
  expectLstmState "LSTM empty first state" emptyStates.1 h₁
  expectLstmState "LSTM empty second state" emptyStates.2.1 h₂
  expectLstmState "LSTM empty third state" emptyStates.2.2.1 h₃
  let zeroState (width : Nat) : Spec.LSTMState Float width :=
    { hidden := Tensor.zeros _, cell := Tensor.zeros _ }
  expectTensor "LSTM chain uses the same zero-state stack"
    ((Spec.Lstm.stacked model.layers referenceHead).forward input)
    (model.forward input (zeroState 3, (zeroState 1, (zeroState 2, ())))).1
  let oldCell := referenceLstm 2 2
  let oldChain : Spec.Module.Chain Float [2, 2] [2, 1] :=
    .comp (.single (Spec.Module.lstm oldCell))
      (.comp (.single (Spec.Module.lstm oldCell))
        (.single (Spec.Module.liftLeading (Spec.Module.linear referenceHead))))
  expectTensor "LSTM two-layer chain preserves existing output"
    ((Spec.Lstm.stacked (.cons oldCell (.cons oldCell .nil)) referenceHead).forward input)
    (oldChain.forward input)
  let headOnly : Spec.Lstm.StackedModel Float 2 2 1 :=
    { layers := .nil, outputLayer := referenceHead }
  for sequenceLength in [0, 2] do
    let input := referenceTensor [sequenceLength, 2] 0.2
    expectTensor "LSTM head-only reference" (headOnly.forward input ()).1 <|
      Tensor.mapLeading [sequenceLength] (Spec.linearSpec referenceHead) input

def run : IO Unit := do
  checkRnnReferenceStack
  checkGruReferenceStack
  checkLstmReferenceStack
  checkHigherDerivatives
  checkInvalidAutograd
  let meanVitSummary ←
    match nn.summary meanVit with
    | .ok summary => pure summary
    | .error message => throw <| IO.userError message
  expect "ViT encoder exposes its patch conversion under the ViT namespace"
    (meanVitSummary.layers.any fun layer => layer.kind == "ViT.PatchesToTokens")
  expect "ViT encoder ends with LayerNorm"
    (match meanVitSummary.layers[meanVitSummary.layers.size - 1]? with
     | some layer => layer.kind == "LayerNorm"
     | none => false)
  expect "mean-pooled ViT has no class-token parameter"
    (!(nn.stateShapes meanVit).contains [1, 3])
  expect "class-pooled ViT owns one shared class-token parameter"
    ((nn.stateShapes clsVit).contains [1, 3])
  expect "class-token parameter adds exactly one state entry"
    ((nn.stateShapes clsVit).length == (nn.stateShapes meanVit).length + 1)
  expect "every compact ViT state entry is trainable"
    ((nn.requiresGrad clsVit).all id)
  expect "mean-pooled ViT rejects an empty patch grid"
    (!(nn.validate emptyMeanVit).isOk)
  expect "class-token ViT rejects an empty patch grid"
    (!(nn.validate emptyClassVit).isOk)
  expect "ViT rejects an empty input grid even when padding would create patches"
    (!(nn.validate paddedEmptyVit).isOk)
  expect "CNN rejects an empty input grid even when padding would create features"
    (!(nn.validate emptyInputCnn).isOk)
  expect "CNN rejects convolution geometry that collapses the spatial grid"
    (!(nn.validate emptyAfterConvCnn).isOk)
  expect "CNN rejects pooling geometry that collapses the spatial grid"
    (!(nn.validate emptyAfterPoolCnn).isOk)
  expect "convolution rejects an empty input grid even when padding creates output"
    (!(nn.validate emptyInputConv).isOk)
  expect "convolution rejects geometry that produces an empty output grid"
    (!(nn.validate emptyOutputConv).isOk)
  expect "transpose convolution rejects geometry that produces an empty output grid"
    (!(nn.validate emptyOutputConvTranspose).isOk)
  expect "max pooling rejects zero channels"
    (!(nn.validate zeroChannelMaxPool).isOk)
  expect "max pooling rejects an empty input grid"
    (!(nn.validate emptyInputMaxPool).isOk)
  expect "average pooling rejects geometry that produces an empty output grid"
    (!(nn.validate emptyOutputAvgPool).isOk)
  expect "ResNet rejects an empty input grid"
    (!(nn.validate emptyInputResNet).isOk)
  expect "basic diffusion predictor rejects an empty input grid"
    (!(nn.validate emptyBasicDiffusion).isOk)
  expect "residual diffusion predictor rejects an empty input grid"
    (!(nn.validate emptyResidualDiffusion).isOk)
  expect "FNO rejects an empty spatial grid"
    (!(nn.validate zeroGridFno).isOk)
  expect "FNO rejects zero channel width"
    (!(nn.validate zeroWidthFno).isOk)
  expect "valid autograd loss axes pass validation" classifierLoss.validate.isOk
  expect "invalid autograd loss axes are rejected"
    (!invalidClassifierLoss.validate.isOk)

  expect "U-Net validates before execution" (nn.validate unet).isOk
  expect "U-Net rejects an empty input grid even when pooling padding creates output"
    (!(nn.validate paddedEmptyUnet).isOk)
  expect "U-Net has eight affine layers"
    ((nn.stateShapes unet).length == 16)
  expect "every U-Net parameter is trainable"
    ((nn.requiresGrad unet).size == 16 && (nn.requiresGrad unet).all id)

  let unetGraph ← nn.lowerToTypedGraph (α := Float) unet
  let unetInput : Tensor Float (unetConfig.inputShape) :=
    Tensor.zeros unetConfig.inputShape
  let unetOutput :=
    nn.TypedGraphModel.forward unetGraph (nn.initialState unet) unetInput
  let unetValues := Tensor.to unetOutput (Array Float)
  expect "U-Net forward preserves its configured boundary"
    (unetValues.size == 4 && unetValues.all Float.isFinite)

  let batchedUnetGraph ← nn.lowerToTypedGraph (α := Float) batchedUnet
  let batchedUnetInput : Tensor Float (unetConfig.inputShape [2]) :=
    Tensor.zeros (unetConfig.inputShape [2])
  let batchedUnetOutput :=
    nn.TypedGraphModel.forward batchedUnetGraph
      (nn.initialState batchedUnet) batchedUnetInput
  let batchedUnetValues := Tensor.to batchedUnetOutput (Array Float)
  expect "U-Net maps independently across leading axes"
    (batchedUnetValues.size == 8 && batchedUnetValues.all Float.isFinite)

  expect "valid dropout endpoints pass model validation"
    ((nn.validate zeroDropout).isOk && (nn.validate fullDropout).isOk)
  expect "negative dropout is rejected" (!(nn.validate negativeDropout).isOk)
  expect "dropout above one is rejected" (!(nn.validate oversizedDropout).isOk)
  expect "NaN dropout is rejected" (!(nn.validate nanDropout).isOk)
  expect "infinite dropout is rejected" (!(nn.validate infiniteDropout).isOk)
  expect "nested invalid dropout is rejected" (!(nn.validate nestedInvalidDropout).isOk)

  expect "BatchNorm momentum endpoints validate"
    ((nn.validate zeroMomentumBatchNorm).isOk && (nn.validate fullMomentumBatchNorm).isOk)
  expect "negative BatchNorm momentum is rejected"
    (!(nn.validate negativeMomentumBatchNorm).isOk)
  expect "BatchNorm momentum above one is rejected"
    (!(nn.validate oversizedMomentumBatchNorm).isOk)
  expect "NaN BatchNorm momentum is rejected"
    (!(nn.validate nanMomentumBatchNorm).isOk)

  expect "out-of-bounds softmax axes are rejected"
    (!(nn.validate invalidSoftmax).isOk)
  expect "reshape is proof-free and rejects unequal element counts"
    (!(nn.validate invalidReshape).isOk)
  expect "invalid model placeholders do not allocate fake state"
    ((nn.stateShapes invalidReshape).isEmpty && (nn.requiresGrad invalidReshape).isEmpty)
  expect "zero-width LayerNorm is rejected"
    (!(nn.validate zeroWidthLayerNorm).isOk)
  expect "zero-width RMSNorm is rejected"
    (!(nn.validate zeroWidthRmsNorm).isOk)
  expect "BatchNorm rejects zero channels"
    (!(nn.validate zeroChannelBatchNorm).isOk)
  expect "InstanceNorm rejects empty spatial geometry"
    (!(nn.validate emptySpatialInstanceNorm).isOk)
  expect "GroupNorm rejects zero groups"
    (!(nn.validate zeroGroupNorm).isOk)
  expect "GroupNorm rejects nondivisible channels"
    (!(nn.validate nondivisibleGroupNorm).isOk)
  expect "attention rejects an empty sequence"
    (!(nn.validate zeroSequenceAttention).isOk)
  expect "attention rejects zero model width"
    (!(nn.validate zeroModelWidthAttention).isOk)
  expect "attention rejects zero head count"
    (!(nn.validate zeroHeadCountAttention).isOk)
  expect "biased attention rejects zero head width"
    (!(nn.validate zeroHeadWidthBiasedAttention).isOk)
  expect "an empty Transformer stack still validates sequence length"
    (!(nn.validate zeroSequenceTransformer).isOk)
  expect "an empty Transformer stack still validates model width"
    (!(nn.validate zeroWidthTransformer).isOk)
  expect "an empty Transformer stack still validates head count"
    (!(nn.validate zeroHeadCountTransformer).isOk)
  expect "an empty Transformer stack still validates head width"
    (!(nn.validate zeroHeadWidthTransformer).isOk)
  expect "an empty Transformer stack still validates feed-forward width"
    (!(nn.validate zeroFeedForwardTransformer).isOk)
  expect "an empty Transformer stack still validates dropout"
    (!(nn.validate invalidDropoutTransformer).isOk)
  expect "an empty Transformer stack still validates initialization"
    (!(nn.validate invalidInitializationTransformer).isOk)
  expect "Linear rejects zero input width"
    (!(nn.validate zeroInputLinear).isOk)
  expect "Linear rejects zero output width"
    (!(nn.validate zeroOutputLinear).isOk)
  expect "RNN rejects an empty sequence"
    (!(nn.validate zeroSequenceRnn).isOk)
  expect "GRU rejects zero input width"
    (!(nn.validate zeroInputGru).isOk)
  expect "Mamba rejects zero hidden width"
    (!(nn.validate zeroHiddenMamba).isOk)
  expect "LSTM rejects an empty sequence"
    (!(nn.validate zeroSequenceLstm).isOk)
  expect "an MLP validates dropout even when it has no hidden layer"
    (!(nn.validate invalidShallowMlp).isOk)
  expect "learned positions reject an empty sequence"
    (!(nn.validate zeroSequenceLearnedPosition).isOk)
  expect "learned positions reject zero embedding width"
    (!(nn.validate zeroWidthLearnedPosition).isOk)
  expect "sinusoidal positions reject an empty sequence"
    (!(nn.validate zeroSequenceSinusoidalPosition).isOk)
  expect "sinusoidal positions reject zero embedding width"
    (!(nn.validate zeroWidthSinusoidalPosition).isOk)
  expect "invalid sinusoidal positions do not allocate fake buffers"
    ((nn.stateShapes zeroSequenceSinusoidalPosition).isEmpty &&
      (nn.requiresGrad zeroSequenceSinusoidalPosition).isEmpty)
  expect "RoPE rejects an empty sequence"
    (!(nn.validate zeroSequenceRope).isOk)
  expect "RoPE rejects zero head width"
    (!(nn.validate zeroWidthRope).isOk)
  expect "invalid RoPE does not allocate fake buffers"
    ((nn.stateShapes zeroSequenceRope).isEmpty &&
      (nn.requiresGrad zeroSequenceRope).isEmpty)
  expect "global average pooling rejects zero channels"
    (!(nn.validate zeroChannelGlobalPool).isOk)
  expect "reversed uniform initialization is rejected"
    (!(nn.validate reversedUniformLinear).isOk)
  expect "negative normal standard deviation is rejected"
    (!(nn.validate negativeStdLinear).isOk)
  expect "indexed embedding initialization is validated"
    (!invalidEmbedding.validate.isOk)
  expect "invalid indexed embedding initialization allocates no parameter state"
    (invalidEmbedding.stateShapes.isEmpty && invalidEmbedding.requiresGrad.isEmpty)
  expect "indexed embeddings reject an empty vocabulary"
    (!zeroVocabularyEmbedding.validate.isOk)
  expect "empty-vocabulary embeddings allocate no parameter state"
    (zeroVocabularyEmbedding.stateShapes.isEmpty &&
      zeroVocabularyEmbedding.requiresGrad.isEmpty)
  expect "indexed embeddings reject zero embedding width"
    (!zeroWidthEmbedding.validate.isOk)
  expect "zero-width embeddings allocate no parameter state"
    (zeroWidthEmbedding.stateShapes.isEmpty &&
      zeroWidthEmbedding.requiresGrad.isEmpty)
  expect "one-hot embeddings reject an empty vocabulary"
    (!(nn.validate zeroVocabularyOneHotEmbedding).isOk)
  expect "one-hot embeddings reject zero embedding width"
    (!(nn.validate zeroWidthOneHotEmbedding).isOk)
  expect "layer validation rejects a mismatched trainability mask"
    (!(nn.validate malformedTrainability).isOk)
  expect "model summaries reject a mismatched trainability mask"
    (!(nn.summary malformedTrainability).isOk)
  expect "indexed-model validation rejects a mismatched trainability mask"
    (!malformedIndexed.validate.isOk)

  let moduleRejected ←
    try
      let _ ← nn.Module.instantiate negativeDropout (α := Float)
      pure false
    catch _ =>
      pure true
  expect "module instantiation validates dropout" moduleRejected

  let initializerRejected ←
    try
      let _ ← nn.Module.instantiate reversedUniformLinear (α := Float)
      pure false
    catch _ =>
      pure true
  expect "module instantiation validates initializer plans" initializerRejected

  let objectiveRejected ←
    try
      let definition := nn.Objective.mse negativeDropout
      let _ ← Module.instantiate definition (α := Float)
      pure false
    catch _ =>
      pure true
  expect "objective instantiation validates dropout" objectiveRejected

  let objectiveMaskRejected ←
    try
      let definition :=
        { nn.Objective.mse zeroDropout with requiresGrad := #[] }
      let _ ← Module.instantiate definition (α := Float)
      pure false
    catch _ =>
      pure true
  expect "objective instantiation validates its trainability mask" objectiveMaskRejected

  let loweringRejected ←
    try
      let _ ← nn.lowerToTypedGraph (α := Float) negativeDropout
      pure false
    catch _ =>
      pure true
  expect "typed-graph lowering validates dropout" loweringRejected

  let input : Tensor Float [3] := [2.0, -4.0, 8.0]
  let fullGraph ← nn.lowerToTypedGraph (α := Float) fullDropout (mode := .train)
  let fullOutput :=
    nn.TypedGraphModel.forward fullGraph (nn.initialState fullDropout) input
  let fullValues := Tensor.to fullOutput (Array Float)
  expect "p = 1 training dropout returns finite zeros"
    (fullValues.all fun value => value.isFinite && value == 0.0)

  let zeroGraph ← nn.lowerToTypedGraph (α := Float) zeroDropout (mode := .train)
  let zeroOutput :=
    nn.TypedGraphModel.forward zeroGraph (nn.initialState zeroDropout) input
  expect "p = 0 training dropout is identity"
    (Tensor.to zeroOutput (Array Float) == Tensor.to input (Array Float))

  let evalGraph ← nn.lowerToTypedGraph (α := Float) fullDropout (mode := .eval)
  let evalOutput :=
    nn.TypedGraphModel.forward evalGraph (nn.initialState fullDropout) input
  expect "evaluation dropout is identity at every probability"
    (Tensor.to evalOutput (Array Float) == Tensor.to input (Array Float))

  match nn.runtimeInit? sinusoidalEncoding with
  | none =>
      throw <| IO.userError
        "public model contract failed: sinusoidal encoding has no runtime plan"
  | some plan =>
      let initializers := plan.toArray
      expect "sinusoidal encoding runtime plan has one buffer" (initializers.size == 1)
      match initializers[0]? with
      | some (Runtime.Autograd.Model.Module.RuntimeInit.FloatInit.flat values) =>
          expect "sinusoidal encoding has one exact runtime buffer" (values.size == 15)
      | _ =>
          throw <| IO.userError
            "public model contract failed: sinusoidal encoding buffer is not exact"
  expect "sinusoidal encoding buffer is non-trainable"
    (nn.requiresGrad sinusoidalEncoding == #[false])

  match nn.runtimeInit? rotaryEncoding with
  | none =>
      throw <| IO.userError
        "public model contract failed: RoPE has no runtime plan"
  | some plan =>
      let initializers := plan.toArray
      expect "RoPE runtime plan has two buffers" (initializers.size == 2)
      match initializers[0]?, initializers[1]? with
      | some (Runtime.Autograd.Model.Module.RuntimeInit.FloatInit.flat cosValues),
          some (Runtime.Autograd.Model.Module.RuntimeInit.FloatInit.flat sinValues) => do
          expect "RoPE cosine runtime buffer has the full table" (cosValues.size == 12)
          expect "RoPE sine runtime buffer has the full table" (sinValues.size == 12)
      | _, _ =>
          throw <| IO.userError
            "public model contract failed: RoPE buffers are not exact"
  expect "RoPE buffers are non-trainable"
    (nn.requiresGrad rotaryEncoding == #[false, false])

  let oddRopeInput : Tensor Float [2, 3] :=
    (Tensor.from #[1.0, 2.0, 3.0, -4.0, 5.0, -6.0]).reshape [2, 3] (by decide)
  let oddRopeGraph ← nn.lowerToTypedGraph (α := Float) oddWidthRotaryEncoding
  let oddRopeOutput :=
    nn.TypedGraphModel.forward oddRopeGraph
      (nn.initialState oddWidthRotaryEncoding) oddRopeInput
  let oddRopeInputValues := Tensor.to oddRopeInput (Array Float)
  let oddRopeOutputValues := Tensor.to oddRopeOutput (Array Float)
  expect "odd-width RoPE preserves each final unpaired coordinate"
    (oddRopeOutputValues[2]? == oddRopeInputValues[2]? &&
      oddRopeOutputValues[5]? == oddRopeInputValues[5]?)

  IO.println "  public model contracts: passed"

end NN.Tests.API.ModelContracts

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Models.CausalTransformer.Runtime
public import NN.API.Models.Cnn
public import NN.API.Models.Diffusion
public import NN.API.Models.FNO
public import NN.API.Models.Generative
public import NN.API.Models.KAN
public import NN.API.Models.Mamba
public import NN.API.Models.PPO
public import NN.API.Models.Recurrent
public import NN.API.Models.ResNet
public import NN.API.Models.SelfSupervised
public import NN.API.Models.Unet
public import NN.API.Models.Vit
public import NN.API.Neural.Transformer

/-!
# Builder Seed API Tests

Regression checks for public model constructors whose initialization must advance `nn.Builder`.
-/

@[expose] public section

namespace NN.Tests.API.BuilderSeeds

open TorchLean

universe u

def expectCounter (tag : String) (expected actual : Nat) : IO Unit := do
  unless actual == expected do
    throw <| IO.userError
      s!"builder seed check failed: {tag} (expected {expected}, got {actual})"

def expectError {α : Type} (tag expected : String) (result : Except String α) : IO Unit := do
  match result with
  | .error message =>
      unless message == expected do
        throw <| IO.userError
          s!"builder seed check failed: {tag} (expected `{expected}`, got `{message}`)"
  | .ok _ =>
      throw <| IO.userError
        s!"builder seed check failed: {tag} (expected an error)"

def counterAfter {α : Type u} (builder : nn.Builder α) (seed : Nat) : Nat :=
  (builder (rand.SeedStream.init seed)).2.counter

def initializerSeed? :
    Runtime.Autograd.Model.Module.RuntimeInit.FloatInit → Option Nat
  | .uniform _ _ seed
  | .normal _ _ seed
  | .xavierUniform _ _ seed
  | .kaimingUniform _ seed => some seed
  | .zeros
  | .ones
  | .flat _ => none

def stochasticInitializerSeeds {source target : Spec.Shape}
    (model : nn.Sequential source target) : Except String (Array Nat) :=
  match nn.runtimeInit? model with
  | none => .error "model does not provide a complete runtime initialization plan"
  | some plan => .ok (plan.toArray.filterMap initializerSeed?)

def expectedStreamSeeds (baseSeed count : Nat) : Array Nat :=
  Array.ofFn fun index : Fin count =>
    rand.nextSeed baseSeed index.val

def expectedTransformerSeeds (baseSeed layerCount keysPerLayer : Nat) : Array Nat :=
  (List.range layerCount).flatMap (fun layerIndex =>
    (List.range 6).map fun parameterIndex =>
      rand.nextSeed baseSeed (keysPerLayer * layerIndex + parameterIndex))
    |>.toArray

def expectSeeds (tag : String) (expected : Array Nat)
    (actual : Except String (Array Nat)) : IO Unit := do
  let actual ←
    match actual with
    | .ok seeds => pure seeds
    | .error message => throw <| IO.userError s!"builder seed check failed: {tag}: {message}"
  unless actual == expected do
    throw <| IO.userError
      s!"builder seed check failed: {tag} (expected {expected}, got {actual})"

def fnoConfig : nn.models.FNO.Config 1 :=
  { spatial := Tensor.full [1] 2
    modes := Tensor.full [1] 1
    width := 1
    layerCount := 0 }

def convConfig : nn.Convolution.Config 1 :=
  { outChannels := 1
    kernelSize := [1] }

def convBlockConfig (dropout? : Option Float := none) : nn.ConvBlock.Config 1 :=
  { convolution := convConfig
    dropout? }

def poolConfig : nn.Pooling.Config 1 :=
  { kernelSize := [1] }

def convPoolBlockConfig (dropout? : Option Float := none) : nn.ConvPoolBlock.Config 1 :=
  { block := convBlockConfig dropout?
    pooling := poolConfig }

def generativeConfig : nn.models.Generative.Config :=
  { dataWidth := 4, hiddenWidth := 3, latentWidth := 2 }

def mambaConfig : nn.models.Mamba.Config :=
  { vocabularySize := 4, modelWidth := 2 }

def recurrentConfig : nn.models.Recurrent.Config :=
  { sequenceLength := 3
    inputWidth := 4
    hiddenWidth := 2
    outputWidth := 5 }

def cnnConfig : nn.models.CNN.Config 1 :=
  { inputChannels := 1
    spatial := [4]
    convolution :=
      { outChannels := 2
        kernelSize := [1] }
    pooling :=
      { kernelSize := [1] }
    classCount := 3 }

def resnetConfig : nn.models.ResNet.Config 1 :=
  { inputChannels := 1
    spatial := [4]
    hiddenChannels := 2
    classCount := 3 }

def diffusionConfig : nn.models.Diffusion.NoisePredictor.Config 1 :=
  { dataChannels := 1
    spatial := [4]
    hiddenChannels := 2 }

def unetConfig : nn.models.UNet.Config 1 :=
  { inputChannels := 1
    baseChannels := 2
    outputChannels := 1
    spatial := [4]
    pooling :=
      { kernelSize := [2] }
    upsampling :=
      { kernelSize := [2] } }

def vitEncoderConfig : nn.models.ViT.EncoderConfig 1 :=
  { inputChannels := 1
    spatial := [4]
    patchEmbedding :=
      { outChannels := 2
        kernelSize := [2]
        stride := [2] }
    headCount := 1
    headWidth := 2
    feedForwardWidth := 4 }

def vitConfig : nn.models.ViT.Config 1 :=
  vitEncoderConfig.classifier 3

def maskedAutoencoderConfig : nn.models.ViT.MaskedAutoencoder.Config 1 :=
  { encoder := vitEncoderConfig
    reconstructionWidth := 4 }

def kanConfig : nn.models.KAN.Config :=
  { inputWidth := 2
    hiddenWidths := [3]
    outputWidth := 1 }

def ppoConfig : nn.models.PPO.Config :=
  { observationWidth := 2
    hiddenWidth := 3
    actionCount := 2 }

def causalConfig : nn.models.CausalTransformer.Config :=
  { sequenceLength := 3
    vocabularySize := 5
    headCount := 1
    headWidth := 2
    feedForwardWidth := 4
    layerCount := 1 }

def transformerConfig : nn.TransformerEncoder.Stack.Config :=
  { layerCount := 12
    block :=
      { headCount := 2
        headWidth := 2
        feedForwardWidth := 7
        dropout? := some 0.1 } }

def run : IO Unit := do
  expectCounter "classification head seeds only its random weight" 1 <|
    counterAfter
      (nn.heads.classifier (featureShape := [2, 3]) 4)
      11
  expectCounter "regression head seeds only its random weight" 1 <|
    counterAfter
      (nn.heads.regressor (featureShape := [2, 3]) 4)
      11
  expectCounter "default linear seeds its weight but not its zero bias" 1 <|
    counterAfter (nn.linear 2 3) 11
  expectCounter "deterministic linear initialization consumes no keys" 0 <|
    counterAfter
      (nn.linear 2 3
        (config := { weightInitialization? := some .zeros }))
      11
  expectCounter "stochastic linear weight and bias consume two keys" 2 <|
    counterAfter
      (nn.linear 2 3
        (config := { biasInitialization := .normal 0.0 1.0 }))
      11
  expectCounter "invalid linear dimensions consume no keys" 0 <|
    counterAfter (nn.linear 0 3) 11
  expectCounter "invalid convolution geometry consumes no keys" 0 <|
    counterAfter
      (nn.conv (inputChannels := 1) ([0] : Tensor Nat [1]) convConfig)
      11
  expectCounter "invalid attention dimensions consume no keys" 0 <|
    counterAfter
      (nn.multiHeadAttention
        (sequenceLength := 3) (modelWidth := 2)
        { headCount := 0, headWidth := 2 })
      11
  expectCounter "invalid learned positions consume no keys" 0 <|
    counterAfter
      (nn.learnedPositionalEmbedding
        (sequenceLength := 0) (embeddingWidth := 2))
      11
  expectCounter "zero-probability dropout is deterministic" 0 <|
    counterAfter (nn.dropout (shape := [3]) 0.0) 11
  expectCounter "full-probability dropout is deterministic" 0 <|
    counterAfter (nn.dropout (shape := [3]) 1.0) 11
  expectCounter "invalid dropout consumes no keys" 0 <|
    counterAfter (nn.dropout (shape := [3]) 1.5) 11
  expectError "dropout validation uses the public layer name"
    "Dropout: probability must be finite and in [0, 1], got 1.500000" <|
      nn.validate <| nn.build 11 <| nn.dropout (shape := [3]) 1.5
  expectCounter "deterministic embedding initialization consumes no keys" 0 <|
    counterAfter
      (nn.embedding 2 3 { weightInitialization := .zeros })
      11
  expectCounter "invalid embedding initialization consumes no keys" 0 <|
    counterAfter
      (nn.embedding 2 3
        { weightInitialization := .normal 0.0 (-1.0) })
      11
  expectCounter "FNO with no operator blocks seeds lift and projection" 2 <|
    counterAfter (nn.models.fno fnoConfig) 11
  expectCounter "invalid FNO configuration consumes no keys" 0 <|
    counterAfter (nn.models.fno { fnoConfig with width := 0 }) 11
  let fnoTwelve := { fnoConfig with layerCount := 12 }
  let (fnoModel, fnoStream) :=
    (nn.models.fno fnoTwelve (batchShape := [])) (rand.SeedStream.init 17)
  expectCounter "FNO allocates every stochastic parameter key" 38 fnoStream.counter
  expectSeeds "FNO initializer keys exactly match its builder stream"
    (expectedStreamSeeds 17 38) (stochasticInitializerSeeds fnoModel)
  let (transformerModel, transformerStream) :=
    (nn.transformerEncoderStack
      transformerConfig (sequenceLength := 3) (modelWidth := 4)
      (mask := none) (batchShape := []))
      (rand.SeedStream.init 29)
  expectCounter "Transformer allocates six parameter and two dropout keys per block"
    (8 * transformerConfig.layerCount) transformerStream.counter
  expectSeeds "Transformer parameter keys exactly match its builder stream"
    (expectedTransformerSeeds 29 transformerConfig.layerCount 8)
    (stochasticInitializerSeeds transformerModel)
  let endpointTransformer :=
    { transformerConfig with
      layerCount := 1
      block := { transformerConfig.block with dropout? := some 1.0 } }
  expectCounter "endpoint Transformer dropout consumes no mask keys" 6 <|
    counterAfter
      (nn.transformerEncoderStack
        endpointTransformer (sequenceLength := 3) (modelWidth := 4))
      11
  expectCounter "convolution seeds its random kernel but not its zero bias" 1 <|
    counterAfter (nn.conv (inputChannels := 1) (Tensor.full [1] 2) convConfig) 11
  expectCounter "convolution block without dropout seeds only its kernel" 1 <|
    counterAfter
      (nn.convBlock (inputChannels := 1)
        (Tensor.full [1] 2) (convBlockConfig none))
      11
  expectCounter "convolution block with dropout adds one mask key" 2 <|
    counterAfter
      (nn.convBlock (inputChannels := 1)
        (Tensor.full [1] 2) (convBlockConfig (some 0.1)))
      11
  expectCounter "pooled convolution block without dropout seeds only its kernel" 1 <|
    counterAfter
      (nn.convPoolBlock (inputChannels := 1)
        (Tensor.full [1] 2) (convPoolBlockConfig none))
      11
  expectCounter "pooled convolution block with dropout adds one mask key" 2 <|
    counterAfter
      (nn.convPoolBlock (inputChannels := 1)
        (Tensor.full [1] 2) (convPoolBlockConfig (some 0.1)))
      11
  expectCounter "RMSNorm unit scale is deterministic" 0 <|
    counterAfter (nn.rmsNorm (width := 2)) 11
  expectCounter "autoencoder seeds four linear weights" 4 <|
    counterAfter (nn.models.Generative.autoencoder generativeConfig) 11
  expectCounter "invalid autoencoder configuration consumes no keys" 0 <|
    counterAfter
      (nn.models.Generative.autoencoder
        { generativeConfig with hiddenWidth := 0 })
      11
  expectError "autoencoder validation names the public constructor"
    "Autoencoder: hidden width must be positive" <|
      nn.validate <| nn.build 11 <|
        nn.models.Generative.autoencoder
          { generativeConfig with hiddenWidth := 0 }
  expectError "generator validation names the public constructor"
    "Generator: latent width must be positive" <|
      nn.validate <| nn.build 11 <|
        nn.models.Generative.generator
          { generativeConfig with latentWidth := 0 }
  expectError "discriminator validation names the public constructor"
    "Discriminator: data width must be positive" <|
      nn.validate <| nn.build 11 <|
        nn.models.Generative.discriminator
          { generativeConfig with dataWidth := 0 }
  expectCounter "RNN model seeds its core and output weight" 2 <|
    counterAfter (nn.models.rnn recurrentConfig) 11
  expectCounter "invalid recurrent configuration consumes no keys" 0 <|
    counterAfter
      (nn.models.rnn { recurrentConfig with outputWidth := 0 })
      11
  expectError "RNN validation names the public constructor"
    "RNN: output width must be positive" <|
      nn.validate <| nn.build 11 <|
        nn.models.rnn { recurrentConfig with outputWidth := 0 }
  expectCounter "GRU model seeds its core and output projection" 4 <|
    counterAfter (nn.models.gru recurrentConfig) 11
  expectError "GRU validation names the public constructor"
    "GRU: input width must be positive" <|
      nn.validate <| nn.build 11 <|
        nn.models.gru { recurrentConfig with inputWidth := 0 }
  expectCounter "LSTM model seeds its core and output projection" 5 <|
    counterAfter (nn.models.lstm recurrentConfig) 11
  expectError "LSTM validation names the public constructor"
    "LSTM: hidden width must be positive" <|
      nn.validate <| nn.build 11 <|
        nn.models.lstm { recurrentConfig with hiddenWidth := 0 }
  expectCounter "Mamba language model seeds its core and output projection" 4 <|
    counterAfter (nn.models.Mamba.languageModel mambaConfig 3) 11
  expectCounter "zero-width Mamba configuration consumes no keys" 0 <|
    counterAfter
      (nn.models.Mamba.languageModel { mambaConfig with modelWidth := 0 } 3)
      11
  expectCounter "zero-vocabulary Mamba configuration consumes no keys" 0 <|
    counterAfter
      (nn.models.Mamba.languageModel { mambaConfig with vocabularySize := 0 } 3)
      11
  expectCounter "zero-length Mamba configuration consumes no keys" 0 <|
    counterAfter (nn.models.Mamba.languageModel mambaConfig 0) 11
  expectError "Mamba vocabulary validation names the public field"
    "Mamba: vocabulary size must be positive" <|
      nn.models.Mamba.Config.validate { mambaConfig with vocabularySize := 0 } 3
  expectError "Mamba width validation names the public field"
    "Mamba: model width must be positive" <|
      nn.models.Mamba.Config.validate { mambaConfig with modelWidth := 0 } 3
  expectCounter "MLP seeds one weight per affine layer" 3 <|
    counterAfter
      (nn.mlp 2 5
        { hiddenWidths := [3, 4] })
      11
  expectCounter "MLP dropout adds one mask key per hidden stage" 5 <|
    counterAfter
      (nn.mlp 2 5
        { hiddenWidths := [3, 4], dropout? := some 0.1 })
      11
  expectCounter "invalid CNN configuration consumes no keys" 0 <|
    counterAfter
      (nn.models.cnn { cnnConfig with classCount := 0 })
      11
  expectError "CNN validation does not leak the convolution layer"
    "CNN: output channel count must be positive" <|
      nn.models.CNN.Config.validate
        { cnnConfig with convolution := { cnnConfig.convolution with outChannels := 0 } }
  expectCounter "invalid ResNet configuration consumes no keys" 0 <|
    counterAfter
      (nn.models.resnet { resnetConfig with classCount := 0 })
      11
  expectCounter "invalid basic diffusion configuration consumes no keys" 0 <|
    counterAfter
      (nn.models.Diffusion.NoisePredictor.basic
        { diffusionConfig with hiddenChannels := 0 })
      11
  expectCounter "invalid residual diffusion configuration consumes no keys" 0 <|
    counterAfter
      (nn.models.Diffusion.NoisePredictor.residual
        { diffusionConfig with hiddenChannels := 0 })
      11
  expectError "diffusion predictor validation uses its public namespace"
    "Diffusion.NoisePredictor: hidden channel count must be positive" <|
      nn.models.Diffusion.NoisePredictor.Config.validate
        { diffusionConfig with hiddenChannels := 0 }
  expectCounter "invalid U-Net configuration consumes no keys" 0 <|
    counterAfter
      (nn.models.unet { unetConfig with outputChannels := 0 })
      11
  expectError "U-Net validation does not leak the pooling layer"
    "UNet: kernel size entries must be positive" <|
      nn.models.UNet.Config.validate
        { unetConfig with pooling := { unetConfig.pooling with kernelSize := [0] } }
  expectCounter "invalid ViT encoder configuration consumes no keys" 0 <|
    counterAfter
      (nn.models.vitEncoder { vitEncoderConfig with headCount := 0 })
      11
  expectError "ViT validation names the public model"
    "ViT: head count must be positive" <|
      nn.models.ViT.EncoderConfig.validate
        { vitEncoderConfig with headCount := 0 }
  expectError "ViT validation does not leak the patch convolution"
    "ViT: output channel count must be positive" <|
      nn.models.ViT.EncoderConfig.validate
        { vitEncoderConfig with
          patchEmbedding := { vitEncoderConfig.patchEmbedding with outChannels := 0 } }
  expectCounter "invalid ViT dropout consumes no keys" 0 <|
    counterAfter
      (nn.models.vitEncoder
        { vitEncoderConfig with dropout? := some 1.5 })
      11
  expectError "ViT validates its dropout before construction"
    "ViT: dropout probability must be finite and in [0, 1], got 1.500000" <|
      nn.models.ViT.EncoderConfig.validate
        { vitEncoderConfig with dropout? := some 1.5 }
  expectCounter "invalid ViT classifier configuration consumes no keys" 0 <|
    counterAfter
      (nn.models.vit { vitConfig with classCount := 0 })
      11
  expectCounter "invalid masked autoencoder configuration consumes no keys" 0 <|
    counterAfter
      (nn.models.ViT.maskedAutoencoder
        { maskedAutoencoderConfig with reconstructionWidth := 0 })
      11
  expectError "masked autoencoder validation names the public model"
    "ViT.MaskedAutoencoder: head count must be positive" <|
      nn.models.ViT.MaskedAutoencoder.Config.validate
        { maskedAutoencoderConfig with
          encoder := { maskedAutoencoderConfig.encoder with headCount := 0 } }
  expectCounter "invalid KAN configuration consumes no keys" 0 <|
    counterAfter
      (nn.models.kan { kanConfig with hiddenWidths := [3, 0] })
      11
  expectError "piecewise-linear KAN validation uses its public namespace"
    "KAN.PiecewiseLinear: input width must be positive" <|
      nn.validate <|
        nn.models.KAN.PiecewiseLinear.layer { gridSize := 2 } 0
  expectCounter "invalid PPO actor configuration consumes no keys" 0 <|
    counterAfter
      (nn.models.PPO.actor { ppoConfig with actionCount := 0 })
      11
  expectError "PPO actor validation names the public constructor"
    "PPO.actor: action count must be positive" <|
      nn.validate <| nn.build 11 <|
        nn.models.PPO.actor { ppoConfig with actionCount := 0 }
  expectCounter "PPO critic ignores the actor-only action count" 2 <|
    counterAfter
      (nn.models.PPO.critic { ppoConfig with actionCount := 0 })
      11
  expectCounter "invalid PPO critic dimensions consume no keys" 0 <|
    counterAfter
      (nn.models.PPO.critic { ppoConfig with hiddenWidth := 0 })
      11
  expectError "PPO critic validation names the public constructor"
    "PPO.critic: hidden width must be positive" <|
      nn.validate <| nn.build 11 <|
        nn.models.PPO.critic { ppoConfig with hiddenWidth := 0 }
  let invalidCausal := { causalConfig with feedForwardWidth := 0 }
  expectError "causal validation names the public model"
    "CausalTransformer: feed-forward width must be positive" <|
      nn.models.CausalTransformer.Config.validate invalidCausal
  expectCounter "invalid causal hidden stack consumes no keys" 0 <|
    counterAfter (nn.models.CausalTransformer.hidden invalidCausal) 11
  expectCounter "invalid one-hot causal model consumes no keys" 0 <|
    counterAfter (nn.models.CausalTransformer.oneHot invalidCausal) 11
  expectCounter "invalid indexed causal model consumes no keys" 0 <|
    counterAfter (nn.models.CausalTransformer.indexed invalidCausal) 11
  expectCounter "invalid tied causal model consumes no keys" 0 <|
    counterAfter (nn.models.CausalTransformer.tied invalidCausal) 11

end NN.Tests.API.BuilderSeeds

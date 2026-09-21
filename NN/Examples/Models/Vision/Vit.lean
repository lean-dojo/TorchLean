/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Real-data CUDA example:
  python3 scripts/datasets/download_example_data.py --cifar10
  lake -R -K cuda=true exe torchlean vit --device cuda --n-total 1 --steps 1

This is a real-data ViT-style CIFAR-10 minibatch run:
- patch embedding via the generic convolution operation over two spatial axes,
- reshape + transpose to tokens,
- two Transformer encoder blocks,
- learned class-token pooling and a linear head.
-/

module

public import NN.Examples.Models.Common.RealData

/-!
# ViT-Style Real-Data Example

Runnable `torchlean vit` example. It trains a compact ViT-style image classifier on a
prepared CIFAR-10 minibatch: patch embedding by convolution, token reshape, transformer block, and
linear head.

The reusable model wiring lives behind the public `TorchLean.nn.models.vit` constructor. The command
adds CIFAR loader construction and the step-limited training loop.

```bash
python3 scripts/datasets/download_example_data.py --cifar10
lake -R -K cuda=true exe torchlean vit --device cuda --n-total 1 --steps 1
```

This command is a small runtime check. Larger image-token runs belong in runtime profiling work,
not the default quick path.
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Vision.Vit

/-- CLI subcommand name used in terminal banners and parser errors. -/
def exeName : String := "vit"

/-- Default JSON loss-curve path for this command. -/
def defaultLogPath : System.FilePath := Support.trainLogPath "vit"

/--
Static minibatch size for the ViT example.

The batch axis is part of the checked model type, so changing this value changes the input and
output shapes at compile time.
-/
def batchSize : Nat := 1

/-- CIFAR image channels. -/
def inputChannels : Nat := RealData.cifarChannels

/-- Height of the CIFAR crop used by this runnable ViT command. -/
def cropHeight : Nat := 4

/-- Width of the CIFAR crop used by this runnable ViT command. -/
def cropWidth : Nat := 4

/-- Patch height used by the convolutional patch embedding. -/
def patchHeight : Nat := 2

/-- Patch width used by the convolutional patch embedding. -/
def patchWidth : Nat := 2

/-- Patch stride; equal to patch size here, so patches do not overlap. -/
def stride : Nat := 2

/-- No zero-padding for the patch embedding. -/
def padding : Nat := 0

/--
Transformer feature width. The `4×4` crop yields four `2×2` image patches, plus the learned class
token used by the classifier.
-/
def modelWidth : Nat := 4

/-- CIFAR class count, hence the output-logit width. -/
def classCount : Nat := RealData.cifarClasses

/-- Number of attention heads in each encoder block. -/
def attentionHeads : Nat := 2

/-- Per-head feature width; `attentionHeads * attentionHeadWidth = modelWidth`. -/
def attentionHeadWidth : Nat := 2

/-- Feed-forward hidden width inside the encoder block. -/
def feedForwardWidth : Nat := 8

/-- Number of Transformer encoder blocks. -/
def transformerLayers : Nat := 2

/-- Shared ViT configuration used by shapes and the reusable public model constructor. -/
abbrev modelConfig : nn.models.ViT.Config 2 :=
  { inputChannels := inputChannels
    spatial := [cropHeight, cropWidth]
    patchEmbedding :=
      { outChannels := modelWidth
        kernelSize := [patchHeight, patchWidth]
        stride := [stride, stride]
        padding := [padding, padding] }
    classCount := classCount
    headCount := attentionHeads
    headWidth := attentionHeadWidth
    feedForwardWidth := feedForwardWidth
    layerCount := transformerLayers
    pooling := .cls }

/-- Batch shape used by this training example. -/
abbrev batch : Shape := [batchSize]

/-- Batched image shape derived from `modelConfig`. -/
abbrev input : Shape := modelConfig.inputShape batch

/-- Batched classifier output derived from `modelConfig`. -/
abbrev output : Shape := modelConfig.outputShape batch

/--
Compact ViT-style classifier from the public model API.

The constructor builds patch embedding, token reshape, positional embeddings, the configured
encoder stack, token pooling, and the classifier head.
-/
def model : nn.Builder (nn.Sequential input output) :=
  nn.models.vit modelConfig batch

/-- Train the CIFAR ViT with the public `Trainer` surface. -/
def train (runtime : Runtime.Config) (flags : RealData.CifarModelTrainFlags) :
    IO Trainer.Report := do
  let batches ←
    RealData.loadCifarBatches exeName batchSize flags.data.nRows flags.data.seed
      flags.data.xPath flags.data.yPath
  let batches ← batches.mapM fun sample =>
    CLI.orThrow exeName <|
      RealData.cropCifarBatch batchSize cropHeight cropWidth sample
  let trainer :=
    Trainer.new model <|
      Trainer.RunConfig.forObjective
        (Trainer.RunConfig.fromRuntime runtime
          { optimizer := optim.adam { learningRate := flags.training.learningRate } })
        (.oneHotCrossEntropy 1)
        (seed := flags.data.seed)
  let trained ← trainer.train
    (Data.fromSamples batches)
    (flags.training.trainOptions
      (logTitle := "ViT CIFAR training")
      (logNotes := RealData.cifarClassifierNotes batchSize flags))
  pure trained.report

/-- CLI entrypoint for CIFAR ViT training on the selected runtime device. -/
def main (args : List String) : IO UInt32 :=
  TrainCommand.classificationNpy exeName args
    (fun rest => RealData.CifarModelTrainFlags.parse exeName rest defaultLogPath 1 1e-3)
    (Support.bannerWithDevice exeName "ViT CIFAR training")
    train

end NN.Examples.Models.Vision.Vit

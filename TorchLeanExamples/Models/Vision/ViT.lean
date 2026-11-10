/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLeanExamples.Models.RealData
public import TorchLean.NN.Models.ViT
public import TorchLeanExamples.Models.Training

/-!
# ViT-Style Real-Data Example

The `torchlean vit` command trains a compact ViT-style image classifier on prepared CIFAR-10
minibatches, cropped to their top-left 4 × 4 pixels. The model embeds patches by convolution,
reshapes them into tokens, runs two Transformer blocks, pools the class token and applies a linear
head.

The public `TorchLean.nn.vit` constructor builds the model, and the command adds the CIFAR
loader and a training loop for a set number of steps.

```bash
python3 scripts/datasets/download_example_data.py --cifar10
scripts/lake.sh -Kcuda=true exe torchlean vit --device cuda --n-total 1 --steps 1
```

The command is a small runtime check. Larger image-token runs belong to runtime profiling work.
-/

@[expose] public section

open TorchLean.nn.ViT.Pooling

namespace TorchLean.Examples.ViT

/-- CLI subcommand name used in terminal banners and parser errors. -/
def exeName : String := "vit"

/-- Default JSON loss-curve path for this command. -/
def defaultLogPath : System.FilePath := Training.artifactPath "vit"

/--
Static minibatch size for the ViT example.

The batch axis is part of the checked model type, so changing this value changes the input and
output shapes at compile time.
-/
def batchSize : Nat := 1

/-- CIFAR image channels. -/
def inputChannels : Nat := RealData.cifarChannels

/-- Height of the top-left CIFAR crop the ViT trains on. -/
def cropHeight : Nat := 4

/-- Width of the top-left CIFAR crop the ViT trains on. -/
def cropWidth : Nat := 4

/-- Patch height used by the convolutional patch embedding. -/
def patchHeight : Nat := 2

/-- Patch width used by the convolutional patch embedding. -/
def patchWidth : Nat := 2

/-- Patch stride, equal to the patch size here so that patches do not overlap. -/
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

/-- Per-head feature width, chosen so that `attentionHeads * attentionHeadWidth = modelWidth`. -/
def attentionHeadWidth : Nat := 2

/-- Feed-forward hidden width inside each encoder block. -/
def feedForwardWidth : Nat := 8

/-- Number of Transformer encoder blocks. -/
def transformerLayers : Nat := 2

/-- ViT configuration with class-token pooling, read by the shapes and by `nn.vit`. -/
abbrev modelConfig : nn.ViT.Config 2 :=
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
    pooling := cls }

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
  nn.vit modelConfig batch

/-- Train the ViT with Adam on one-hot cross-entropy over shuffled, cropped CIFAR minibatches. -/
def train (runtime : Session.Config) (flags : Training.Options NPY.Options) :
    IO Trainer.Report :=
  RealData.trainCifarClassifier batchSize cropHeight cropWidth exeName
    "ViT CIFAR training" model runtime flags

/-- CIFAR ViT entrypoint for the selected device, by default one step at learning rate `1e-3`. -/
def main (args : List String) : IO UInt32 :=
  NPY.train exeName args
    (fun rest => Training.Options.parse exeName rest defaultLogPath 1 1e-3
      (parseData := RealData.parseCifar))
    (Training.banner exeName "ViT CIFAR training")
    train (fun result => result.printSummary) (target := "class-label")

end TorchLean.Examples.ViT

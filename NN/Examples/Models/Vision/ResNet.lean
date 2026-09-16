/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Run:
  python3 scripts/datasets/download_example_data.py --cifar10
  lake -R -K cuda=true exe torchlean resnet --device cuda --n-total 1 --steps 1
-/

module

public import NN.Examples.Models.Common.RealData

/-!
# Residual Classifier Training Example

This command trains the public rank-polymorphic residual classifier on a small CIFAR-10 crop. The
model contains a convolutional stem, two residual branches, global average pooling over all spatial
axes, and a linear classifier. The compact crop keeps the command useful as a runtime check while
still exercising the same residual composition and pooling code used by larger configurations.
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Vision.ResNet

/-- CLI subcommand name used in terminal banners and parser errors. -/
def exeName : String := "resnet"

/-- Default JSON training-log path. -/
def defaultLogPath : System.FilePath := Support.trainLogPath "resnet"

/-- Static minibatch size carried by the checked model shape. -/
def batchSize : Nat := 1

/-- CIFAR input channels. -/
def inputChannels : Nat := RealData.cifarChannels

/-- Height of the compact CIFAR crop. -/
def cropHeight : Nat := 8

/-- Width of the compact CIFAR crop. -/
def cropWidth : Nat := 8

/-- Channel width of the residual trunk. -/
def hiddenChannels : Nat := 4

/--
Complete residual-classifier architecture.

`spatial := [8, 8]` fixes the checked input grid, while `kernelRadius := [1, 1]` selects a `3 x 3`
same-padding kernel. The model constructor derives all intermediate and output tensor types from
this one value.
-/
abbrev modelConfig : nn.models.ResNet.Config 2 :=
  { inputChannels := inputChannels
    spatial := [cropHeight, cropWidth]
    hiddenChannels := hiddenChannels
    kernelRadius := [1, 1]
    classCount := RealData.cifarClasses }

/-- Batched channel-first input type derived from `modelConfig`. -/
abbrev input : Shape := modelConfig.input [batchSize]

/-- One row of class logits per input sample, also derived from `modelConfig`. -/
abbrev output : Shape := modelConfig.output [batchSize]

/-- Residual classifier from the public model API. -/
def model : nn.Builder (nn.Sequential input output) :=
  nn.models.resnet modelConfig [batchSize]

/-- Train the residual classifier with the public classification trainer. -/
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
      (logTitle := "ResNet CIFAR training")
      (logNotes := RealData.cifarClassifierNotes batchSize flags
        #[s!"spatial={cropHeight}x{cropWidth}",
          s!"hiddenChannels={hiddenChannels}"]))
  pure trained.report

/-- CLI entrypoint for the CIFAR residual-classifier training path. -/
def main (args : List String) : IO UInt32 :=
  TrainCommand.classificationNpy exeName args
    (fun rest => RealData.CifarModelTrainFlags.parse exeName rest defaultLogPath 1 1e-3)
    (Support.bannerWithDevice exeName "ResNet CIFAR training")
    train

end NN.Examples.Models.Vision.ResNet

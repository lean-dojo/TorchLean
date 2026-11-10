/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Run:
  python3 scripts/datasets/download_example_data.py --cifar10
  scripts/lake.sh -Kcuda=true exe torchlean resnet --device cuda --n-total 1 --steps 1
-/

module

public import TorchLeanExamples.Models.RealData
public import TorchLean.NN.Models.ResNet
public import TorchLeanExamples.Models.Training

/-!
# Residual Classifier Training Example

The `resnet` command trains the public rank-polymorphic residual classifier on an 8 × 8 CIFAR-10
crop. The model has a convolutional stem, two residual stages, global average pooling over all
spatial axes and a linear classifier. The small crop keeps the command fast enough for a runtime
check, and it runs the same residual composition and pooling code as larger configurations.
-/

@[expose] public section

namespace TorchLean.Examples.ResNet

/-- CLI subcommand name used in terminal banners and parser errors. -/
def exeName : String := "resnet"

/-- Default JSON training-log path. -/
def defaultLogPath : System.FilePath := Training.artifactPath "resnet"

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

`spatial := [8, 8]` fixes the checked input grid. The stem and both residual stages use `3 x 3`
same-padding convolutions. Each stage has an identity shortcut, so its branch preserves the grid
and channel width. The constructor derives the intermediate and output types from this geometry.
-/
abbrev modelConfig : nn.ResNet.Config 2 :=
  let convolution : nn.Conv.Config 2 :=
    { outChannels := hiddenChannels, kernelSize := [3, 3], padding := [1, 1] }
  { inputChannels := inputChannels
    spatial := [cropHeight, cropWidth]
    stem := convolution
    stages := List.replicate 2 { first := convolution, second := convolution }
    classCount := RealData.cifarClasses }

/-- Batched channel-first input type derived from `modelConfig`. -/
abbrev input : Shape := modelConfig.inputShape [batchSize]

/-- One row of class logits per input sample, also derived from `modelConfig`. -/
abbrev output : Shape := modelConfig.outputShape [batchSize]

/-- Residual classifier from the public model API. -/
def model : nn.Builder (nn.Sequential input output) :=
  nn.resnet modelConfig [batchSize]

/-- Train the residual classifier with Adam on one-hot cross-entropy over cropped CIFAR batches. -/
def train (runtime : Session.Config) (flags : Training.Options NPY.Options) :
    IO Trainer.Report :=
  RealData.trainCifarClassifier batchSize cropHeight cropWidth exeName
    "ResNet CIFAR training" model runtime flags
    #[s!"spatial={cropHeight}x{cropWidth}", s!"hiddenChannels={hiddenChannels}"]

/-- CIFAR ResNet entrypoint for the selected device, by default one step at learning rate `1e-3`. -/
def main (args : List String) : IO UInt32 :=
  NPY.train exeName args
    (fun rest => Training.Options.parse exeName rest defaultLogPath 1 1e-3
      (parseData := RealData.parseCifar))
    (Training.banner exeName "ResNet CIFAR training")
    train (fun result => result.printSummary) (target := "class-label")

end TorchLean.Examples.ResNet

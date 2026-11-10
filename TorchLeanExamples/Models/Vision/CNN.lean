/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLeanExamples.Models.RealData
public import TorchLean.NN.Models.CNN
public import TorchLeanExamples.Models.Training

/-!
# CNN Training Example

The `torchlean cnn` command trains a small convolutional classifier on prepared CIFAR-10
minibatches, cropped to their top-left 8 × 8 pixels.

The public `TorchLean.nn.cnn` constructor builds the model, and the command adds what
surrounds it: CLI parsing, dataset selection, training for a set number of steps and the TrainLog
file.

```bash
python3 scripts/datasets/download_example_data.py --cifar10
scripts/lake.sh -Kcuda=true exe torchlean cnn --device cuda --n-total 1 --steps 1
```
-/

@[expose] public section

namespace TorchLean.Examples.CNN

/-- CLI subcommand name used in terminal banners and parser errors. -/
def exeName : String := "cnn"

/-- Default JSON loss-curve path for this command. -/
def defaultLogPath : System.FilePath := Training.artifactPath "cnn"

/--
Minibatch size of the CIFAR run.

The model owns the batch axis, so this value appears in the input and output shapes and in the
call to `RealData.trainCifarClassifier`.
-/
def batchSize : Nat := 1

/-- CIFAR image channels. -/
def inputChannels : Nat := RealData.cifarChannels

/-- Height of the top-left CIFAR crop the CNN trains on. -/
def cropHeight : Nat := 8

/-- Width of the top-left CIFAR crop the CNN trains on. -/
def cropWidth : Nat := 8

/-- CIFAR class count, hence the output-logit width. -/
def classCount : Nat := RealData.cifarClasses

/--
CNN configuration read by the model and both shapes. Its one stage applies a `[3, 3]` convolution
with stride 2 and padding 1 into four channels, ReLU, and `[2, 2]` max pooling with stride 2.
-/
abbrev modelConfig : nn.CNN.Config 2 :=
  { inputChannels := inputChannels
    spatial := [cropHeight, cropWidth]
    stages :=
      [{ block := { convolution :=
           { outChannels := 4
             kernelSize := [3, 3]
             stride := [2, 2]
             padding := [1, 1] } }
         pooling := { kernelSize := [2, 2], stride := [2, 2] } }]
    classCount := classCount }

/-- Input shape `[1, 3, 8, 8]`: a minibatch of cropped CIFAR images in channel-first layout. -/
abbrev input : Shape := modelConfig.inputShape [batchSize]

/-- Output shape `[1, 10]`: one row of class logits per image. -/
abbrev output : Shape := modelConfig.outputShape [batchSize]

/--
Small convolutional classifier from the public model API.

The command picks the CIFAR paths and the runtime flags, and the model is an ordinary
`nn.Sequential` value.
-/
def model : nn.Builder (nn.Sequential input output) :=
  nn.cnn modelConfig [batchSize]

/-- Train the CNN with Adam on one-hot cross-entropy over shuffled, cropped CIFAR minibatches. -/
def train (runtime : Session.Config) (flags : Training.Options NPY.Options) :
    IO Trainer.Report :=
  RealData.trainCifarClassifier batchSize cropHeight cropWidth exeName
    "CNN training" model runtime flags

/-- CIFAR CNN entrypoint for the selected device, by default one step at learning rate `1e-3`. -/
def main (args : List String) : IO UInt32 :=
  NPY.train exeName args
    (fun rest => Training.Options.parse exeName rest defaultLogPath 1 1e-3
      (parseData := RealData.parseCifar))
    (Training.banner exeName "CNN training")
    train (fun result => result.printSummary) (target := "class-label")

end TorchLean.Examples.CNN

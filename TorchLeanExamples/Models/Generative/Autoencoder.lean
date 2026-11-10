/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Run:
  python3 scripts/datasets/download_example_data.py --cifar10
  scripts/lake.sh -Kcuda=true exe torchlean autoencoder --device cuda --steps 1 --n-total 1
-/

module

public import TorchLeanExamples.Models.RealData
public import TorchLeanExamples.Models.Training

/-!
# Autoencoder CIFAR Example

Trains a dense `16 → 8 → 4 → 8 → 16` autoencoder, with ReLU between the layers and a sigmoid at the
end, to reproduce its input. The input is the first 16 values of one flattened, channel-first
CIFAR-10 image (the left half of the first channel's top row), and the target is the same vector.

The command uses Adam and mean squared error through the public `Trainer`, then prints a training
summary and writes the selected `TrainLog` JSON. It writes no reconstructed image files.
-/

@[expose] public section

open TorchLean.Trainer.Objective

namespace TorchLean.Examples.Autoencoder

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "autoencoder"

/-- Default JSON loss-curve path for this command. -/
def defaultLogPath : System.FilePath := Training.artifactPath "autoencoder"

/--
Data, hidden and latent widths (16, 8 and 4), shared by the model and the data loader.
-/
abbrev modelConfig : nn.Generative.Config :=
  { dataWidth := 16, hiddenWidth := 8, latentWidth := 4 }

/-- Number of image vectors loaded for each training sample. -/
def batchSize : Nat := 1

/-- Input shape `[batchSize, 16]`: a batch of 16-value CIFAR image prefixes. -/
abbrev input := modelConfig.dataShape [batchSize]

/-- Target shape, equal to `input` because the target is the input itself. -/
abbrev output := modelConfig.dataShape [batchSize]

/--
Trainable dense autoencoder: the public `nn.Generative.autoencoder` backbone with a sigmoid
on its output. The command chooses the dataset, optimizer, runtime options and log path.
-/
def model : nn.Builder (nn.Sequential input output) :=
  nn.Sequential![
    nn.Generative.autoencoder modelConfig [batchSize],
    nn.op .sigmoid
  ]

/-- One-sample dataset whose input and target are the same 16-value CIFAR vector. -/
def data (flags : Training.Options NPY.Options) (seed : Nat) :
    Trainer.Dataset input output :=
  RealData.cifarFeatureDataset batchSize modelConfig exeName
    (fun tensor ↦ { input := tensor, target := tensor })
    flags.data.xPath flags.data.yPath flags.data.nRows seed

/-- Check that the CIFAR-10 image and label files exist, then train the autoencoder with Adam at
the `--lr` learning rate and mean squared error. -/
def train (runtime : Session.Config) (flags : Training.Options NPY.Options) :
    IO (Trainer.Result input output) := do
  Data.requirePairedFiles exeName
    "CIFAR-10 images" flags.data.xPath
    "CIFAR-10 labels" flags.data.yPath
    RealData.missingCifarHint
  let trainer :=
    Trainer.create model <|
      { runtime with
          optimizer := optim.adam { lr := flags.training.lr }
          objective := mse }
  trainer.train
    (data flags runtime.seed)
    (flags.training.trainOptions
      (logTitle := "Autoencoder CIFAR reconstruction")
      (logNotes := RealData.trainingNotes "cifar10" batchSize flags))

/--
Entry point for CIFAR reconstruction. It loads one real CIFAR minibatch, builds the reconstruction
sample `x -> x`, trains for `--steps` steps at learning rate `--lr` (by default 10 and `1e-3`),
prints the training summary and writes the training log.
-/
def main (args : List String) : IO UInt32 :=
  NPY.train exeName args
    (fun rest => Training.Options.parse exeName rest defaultLogPath 10 1e-3
      (parseData := RealData.parseCifar))
    (Training.banner exeName "CIFAR vector reconstruction")
    train (fun result => result.printSummary)

end TorchLean.Examples.Autoencoder

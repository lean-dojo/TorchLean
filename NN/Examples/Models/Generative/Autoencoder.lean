/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Run:
  python3 scripts/datasets/download_example_data.py --cifar10
  lake -R -K cuda=true exe torchlean autoencoder --device cuda --steps 1 --n-total 1
-/

module

public import NN.Examples.Models.Common.RealData

/-!
# Autoencoder CIFAR Example

Trains a dense `16 → 8 → 4 → 8 → 16` autoencoder with a final sigmoid. The input is the first
16 values of one flattened, channel-first CIFAR-10 image; the target is that same vector.
This is a compact reconstruction exercise, not a full-image autoencoder.

The command uses Adam and mean squared error through the public `Trainer`, then prints a training
summary and writes the selected `TrainLog` JSON. It does not export reconstructed image files.
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Generative.Autoencoder

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "autoencoder"

/-- Default JSON loss-curve path for this command. -/
def defaultLogPath : System.FilePath := Support.trainLogPath "autoencoder"

/--
Dense autoencoder dimensions shared by the model and data boundary.
-/
abbrev modelConfig : nn.models.Generative.Config :=
  { dataWidth := 16, hiddenWidth := 8, latentWidth := 4 }

/-- Number of image vectors loaded for each training sample. -/
def batchSize : Nat := 1

/-- Input shape: a batch of flattened CIFAR image vectors. -/
abbrev input := modelConfig.data [batchSize]

/-- Target shape: the same flattened image-vector batch, because this is reconstruction. -/
abbrev output := modelConfig.data [batchSize]

/--
Trainable dense autoencoder.

The architecture is defined in the public model API. The command chooses the dataset, optimizer,
runtime options, and logging path.
-/
def model : nn.Builder (nn.Sequential input output) :=
  nn.Sequential![
    nn.models.Generative.autoencoder modelConfig [batchSize],
    nn.sigmoid
  ]

/-- Public singleton dataset for compact CIFAR reconstruction. -/
def data (flags : RealData.CifarModelTrainFlags) : Trainer.Dataset input output :=
  RealData.cifarFeatureDataset batchSize modelConfig exeName
    (fun tensor ↦ { input := tensor, target := tensor })
    flags.data.xPath flags.data.yPath flags.data.nRows flags.data.seed

/-- Train the compact autoencoder with the public `Trainer` surface. -/
def train (runtime : Runtime.Config) (flags : RealData.CifarModelTrainFlags) :
    IO (Trainer.Result input output) := do
  Data.requirePairedFiles exeName
    "CIFAR-10 images" flags.data.xPath
    "CIFAR-10 labels" flags.data.yPath
    RealData.missingCifarHint
  let trainer :=
    Trainer.new model <|
      Trainer.RunConfig.forObjective
        (Trainer.RunConfig.fromRuntime runtime
          { optimizer := optim.adam { learningRate := flags.training.learningRate } })
        .meanSquaredError
        (seed := flags.data.seed)
  trainer.train
    (data flags)
    (flags.training.trainOptions
      (logTitle := "Autoencoder CIFAR reconstruction")
      (logNotes := RealData.cifarClassifierNotes batchSize flags))

/--
Executable entrypoint for CIFAR reconstruction.

The command loads one real CIFAR minibatch, builds the supervised reconstruction sample `x -> x`,
trains the autoencoder for `--steps`, and writes the standard TorchLean training summary/log.
-/
def main (args : List String) : IO UInt32 :=
  TrainCommand.regressionNpy exeName args
    (fun rest => RealData.CifarModelTrainFlags.parse exeName rest defaultLogPath 10 1e-3)
    (Support.bannerWithDevice exeName "CIFAR vector reconstruction")
    train

end NN.Examples.Models.Generative.Autoencoder

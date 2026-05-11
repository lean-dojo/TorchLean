/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Core

/-!
# Real Dataset Paths

Small tutorial fixtures live in `NN.Examples.Data.ToyPaths`.

This module names the default paths used by examples that train on externally
downloaded datasets prepared by `scripts/datasets/download_example_data.py`.

The files are intentionally not committed: CIFAR-10 and text corpora are real datasets with their
own licenses/citation expectations, and CI should not download hundreds of megabytes implicitly.
-/

@[expose] public section

namespace NN
namespace Examples
namespace Data
namespace RealPaths

/-- Default root for user-downloaded real datasets. -/
def defaultDataDir : System.FilePath :=
  "data/real"

/-- Parse an optional `--data-dir PATH` flag for real-data examples. -/
def takeDataDir (args : List String) (default : System.FilePath := defaultDataDir) :
    Except String (System.FilePath × List String) := do
  let (dir?, rest) ← NN.API.CLI.takePathFlagOnce args "data-dir"
  pure (dir?.getD default, rest)

/-- Directory containing prepared CIFAR-10 `.npy` arrays. -/
def cifar10Dir (dataDir : System.FilePath := defaultDataDir) : System.FilePath :=
  dataDir / "cifar10"

/-- Prepared CIFAR-10 training images, shape `(N, 3, 32, 32)`, float32 in `[0, 1]`. -/
def cifar10TrainX (dataDir : System.FilePath := defaultDataDir) : System.FilePath :=
  cifar10Dir dataDir / "cifar10_train_X.npy"

/-- Prepared CIFAR-10 training labels, shape `(N,)`, float32 integer labels `0..9`. -/
def cifar10TrainY (dataDir : System.FilePath := defaultDataDir) : System.FilePath :=
  cifar10Dir dataDir / "cifar10_train_y.npy"

/-- Prepared CIFAR-10 test images, shape `(N, 3, 32, 32)`, float32 in `[0, 1]`. -/
def cifar10TestX (dataDir : System.FilePath := defaultDataDir) : System.FilePath :=
  cifar10Dir dataDir / "cifar10_test_X.npy"

/-- Prepared CIFAR-10 test labels, shape `(N,)`, float32 integer labels `0..9`. -/
def cifar10TestY (dataDir : System.FilePath := defaultDataDir) : System.FilePath :=
  cifar10Dir dataDir / "cifar10_test_y.npy"

/--
Directory containing a user-prepared ImageNet-style 64x64 subset.

TorchLean does not redistribute ImageNet.  Users point `scripts/datasets/torchlean_data_convert.py
image-folder` at their own ImageNet/ILSVRC, ImageNet-compatible, or Tiny-ImageNet-style directory
tree and write the converted arrays here.
-/
def imagenet64Dir (dataDir : System.FilePath := defaultDataDir) : System.FilePath :=
  dataDir / "imagenet64"

/-- Prepared ImageNet-style training images, shape `(N, 3, 64, 64)`, float32 in `[0, 1]`. -/
def imagenet64TrainX (dataDir : System.FilePath := defaultDataDir) : System.FilePath :=
  imagenet64Dir dataDir / "imagenet64_train_X.npy"

/--
Prepared ImageNet-style training labels, shape `(N,)`, float32 integer class ids.

The converter assigns ids by sorted subdirectory name when `--labels-from-dirs` is used.
-/
def imagenet64TrainY (dataDir : System.FilePath := defaultDataDir) : System.FilePath :=
  imagenet64Dir dataDir / "imagenet64_train_y.npy"

/-- Directory containing prepared UCI household-power forecasting windows. -/
def householdPowerDir (dataDir : System.FilePath := defaultDataDir) : System.FilePath :=
  dataDir / "household_power"

/--
Prepared UCI household-power inputs, shape `(N, 24, 1)`, float32 normalized to `[0, 1]`.

Each row is a 24-hour window of hourly mean `Global_active_power`.
-/
def householdPowerX (dataDir : System.FilePath := defaultDataDir) : System.FilePath :=
  householdPowerDir dataDir / "household_power_X.npy"

/--
Prepared UCI household-power targets, shape `(N, 24, 1)`, float32 normalized to `[0, 1]`.

Each target row is the corresponding input window shifted by one hour.
-/
def householdPowerY (dataDir : System.FilePath := defaultDataDir) : System.FilePath :=
  householdPowerDir dataDir / "household_power_Y.npy"

/-- Directory containing the prepared UCI Auto MPG tabular regression CSV. -/
def autoMpgDir (dataDir : System.FilePath := defaultDataDir) : System.FilePath :=
  dataDir / "auto_mpg"

/-- Prepared UCI Auto MPG CSV with normalized columns `x1..x7,y`. -/
def autoMpgCsv (dataDir : System.FilePath := defaultDataDir) : System.FilePath :=
  autoMpgDir dataDir / "auto_mpg.csv"

/-- Directory containing downloaded text corpora. -/
def textDir (dataDir : System.FilePath := defaultDataDir) : System.FilePath :=
  dataDir / "text"

/-- Karpathy tiny-shakespeare corpus. -/
def tinyShakespeare (dataDir : System.FilePath := defaultDataDir) : System.FilePath :=
  textDir dataDir / "tiny_shakespeare.txt"

/-- TinyStories validation split, useful for small local language-model smoke training. -/
def tinyStoriesValid (dataDir : System.FilePath := defaultDataDir) : System.FilePath :=
  textDir dataDir / "tinystories_valid.txt"

end RealPaths
end Data
end Examples
end NN

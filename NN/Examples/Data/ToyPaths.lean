/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Core

/-!
# Generated Tutorial Dataset Paths

TorchLean tutorials can generate a few small deterministic datasets under `NN/Examples/Data/`.
This module centralizes:

- the default data directory,
- the file paths of the generated sample datasets, and
- a small CLI convenience for overriding the directory via `--data-dir`.

These helpers are small and predictable: they exist so tutorial code does not hardcode paths (or
reimplement the same flag parsing) in many places.
-/

@[expose] public section

namespace NN
namespace Examples
namespace Data
namespace ToyPaths

/-- Default relative directory containing generated tutorial datasets. -/
def defaultDataDir : System.FilePath :=
  "NN/Examples/Data"

/--
Parse an optional `--data-dir PATH` flag (defaults to `defaultDataDir`).

This is used by tutorials that load local generated CSV/NPY files from disk.
-/
def takeDataDir (args : List String) (default : System.FilePath := defaultDataDir) :
    Except String (System.FilePath × List String) := do
  let (dir?, rest) ← NN.API.CLI.takePathFlagOnce args "data-dir"
  pure (dir?.getD default, rest)

/-- `toy_regression.csv` (2D regression). -/
def regressionCsv (dataDir : System.FilePath := defaultDataDir) : System.FilePath :=
  dataDir / "toy_regression.csv"

/-- `toy_regression_X.npy` (shape 25x2). -/
def regressionXNpy (dataDir : System.FilePath := defaultDataDir) : System.FilePath :=
  dataDir / "toy_regression_X.npy"

/-- `toy_regression_y.npy` (shape 25x1). -/
def regressionYNpy (dataDir : System.FilePath := defaultDataDir) : System.FilePath :=
  dataDir / "toy_regression_y.npy"

/-- `toy_cifar10like_X.npy` (shape 200x3x32x32). -/
def cifar10likeXNpy (dataDir : System.FilePath := defaultDataDir) : System.FilePath :=
  dataDir / "toy_cifar10like_X.npy"

/-- `toy_cifar10like_y.npy` (shape 200). -/
def cifar10likeYNpy (dataDir : System.FilePath := defaultDataDir) : System.FilePath :=
  dataDir / "toy_cifar10like_y.npy"

end ToyPaths
end Data
end Examples
end NN

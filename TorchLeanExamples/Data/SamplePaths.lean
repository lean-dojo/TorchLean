/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# Generated Tutorial Dataset Paths

Default directory and file names of the deterministic sample datasets that
`TorchLeanExamples/Data/generate_small_data.py` writes into `TorchLeanExamples/Data/`. Examples that
read them take `--data-dir` through the shared CLI parser to point at another directory.
-/

@[expose] public section

namespace TorchLean
namespace Examples
namespace SampleData

/--
Default relative directory containing generated tutorial datasets.

Tutorials that accept `--data-dir` read it with
`TorchLean.CLI.takePathFlag args "data-dir" (default := defaultDataDir)`.
-/
def defaultDataDir : System.FilePath :=
  "TorchLeanExamples/Data"

/-- `small_regression.csv` (2D regression). -/
def regressionCSV (dataDir : System.FilePath := defaultDataDir) : System.FilePath :=
  dataDir / "small_regression.csv"

/-- `small_regression_X.npy` (shape 25x2). -/
def regressionX (dataDir : System.FilePath := defaultDataDir) : System.FilePath :=
  dataDir / "small_regression_X.npy"

/-- `small_regression_y.npy` (shape 25x1). -/
def regressionY (dataDir : System.FilePath := defaultDataDir) : System.FilePath :=
  dataDir / "small_regression_y.npy"

/-- `small_cifar10like_X.npy` (shape 200x3x32x32). -/
def cifar10LikeX (dataDir : System.FilePath := defaultDataDir) : System.FilePath :=
  dataDir / "small_cifar10like_X.npy"

/-- `small_cifar10like_y.npy` (shape 200). -/
def cifar10LikeY (dataDir : System.FilePath := defaultDataDir) : System.FilePath :=
  dataDir / "small_cifar10like_y.npy"

end SampleData
end Examples
end TorchLean

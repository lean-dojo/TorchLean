/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Data.Training

/-!
# Compiled Docstring Examples: Datasets

Every `Example:` block in a `TorchLean.Data` docstring appears here verbatim, one namespace per
entry point. Nothing in this file runs: the guarantee we want is that the snippet a reader copies
out of the docstring still elaborates, and typechecking gives us exactly that.

`repo_lint.py` compares the fenced `lean` block inside each `Example:` against the namespace body
carrying the matching `doc-example:` marker, so the two cannot drift apart. Adding an example means
writing it here first, running `python3 scripts/checks/repo_lint.py --sync-doc-examples`, and
letting that command copy the checked text into the docstring.
-/

@[expose] public section

namespace NN.Tests.API.DocExamples.Data

open TorchLean

-- doc-example: NN/API/Data/Training.lean :: def fromTensors
namespace FromTensors

-- Four samples of a two-feature regression problem. The leading `4` counts samples.
def inputs : Tensor Float [4, 2] :=
  [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

def targets : Tensor Float [4, 1] := [[0.0], [1.0], [1.0], [0.0]]

-- The sample axis leaves the type; what stays is the shape of one sample.
def dataset : Trainer.Dataset [2] [1] := Data.fromTensors inputs targets

end FromTensors

-- doc-example: NN/API/Data/Training.lean :: def generate
namespace Generate

-- Samples built in Lean instead of read from disk. `α` stays abstract, so one builder serves a
-- `.native` run and an `.ieee` run.
def dataset : Trainer.Dataset [1] [1] :=
  Data.generate fun {_α} _ _ _ =>
    Array.ofFn (n := 8) fun (i : Fin 8) =>
      let x := i.val.toFloat
      { input := Tensor.map Runtime.ofFloat ([x] : Tensor Float [1])
        target := Tensor.map Runtime.ofFloat ([2.0 * x] : Tensor Float [1]) }

end Generate

-- doc-example: NN/API/Data/Training.lean :: def fromSamples
namespace FromSamples

def samples : Array (Sample.Supervised Float [2] [1]) :=
  #[{ input := [0.0, 1.0], target := [1.0] },
    { input := [1.0, 1.0], target := [0.0] }]

def dataset : Trainer.Dataset [2] [1] := Data.fromSamples samples

end FromSamples

-- doc-example: NN/API/Data/Training.lean :: def fromSample
namespace FromSample

-- One sample is enough to smoke-test a model end to end.
def dataset : Trainer.Dataset [2] [1] :=
  Data.fromSample { input := [1.25, -2.5], target := [3.75] }

end FromSample

-- doc-example: NN/API/Data/Training.lean :: def defer
namespace Defer

-- The file is read when the trainer materializes the dataset, not while this definition is
-- elaborated, so building the module never touches the disk.
def dataset (path : System.FilePath) : Trainer.Dataset [2] [1] :=
  Data.defer do
    let input : Tensor Float [2] ← Tensor.load path
    pure { input := input, target := [3.75] }

end Defer

-- doc-example: NN/API/Data/Training.lean :: def batch
namespace Batch

-- Sixteen single samples become four batched samples, and the batch axis lands in front of both
-- shapes so a batched model accepts the result unchanged.
def batched (dataset : Trainer.Dataset [2] [1]) :
    Trainer.Dataset [4, 2] [4, 1] :=
  Data.batch 4 dataset (shuffle := true) (seed := 0)

end Batch

-- doc-example: NN/API/Data/Training.lean :: def randomSplit
namespace RandomSplit

-- Deterministic given the seed, so the held-out samples are the same on every machine.
def split (dataset : Trainer.Dataset [2] [1]) : Data.DatasetSplit [2] [1] :=
  Data.randomSplit 3 dataset (seed := 17)

def heldOut (dataset : Trainer.Dataset [2] [1]) : Trainer.Dataset [2] [1] :=
  (split dataset).test

end RandomSplit

-- doc-example: NN/API/Data/Training.lean :: def fromCsv
namespace FromCsv

-- Three columns per row: two features then one target, delivered as batches of eight.
def dataset : Trainer.Dataset [8, 2] [8, 1] :=
  Data.fromCsv "data/table.csv" (batchSize := 8) (inputWidth := 2) (targetWidth := 1)

end FromCsv

-- doc-example: NN/API/Data/Training.lean :: def fromSupervisedSource
namespace FromSupervisedSource

def source : Data.SupervisedSource :=
  Data.SupervisedSource.fromFiles "data/inputs.npy" "data/targets.npy"
    (sampleCount := 64) (input := [16]) (target := [1])

def dataset : Trainer.Dataset [16] [1] := Data.fromSupervisedSource source

end FromSupervisedSource

-- doc-example: NN/API/Data/Training.lean :: def fromLabeledSource
namespace FromLabeledSource

def source : Data.LabeledSource :=
  Data.LabeledSource.fromFiles "data/images.npy" "data/labels.npy"
    (sampleCount := 64) (input := [3, 8, 8]) (classCount := 10)

-- Labels are one-hot encoded on load, so the target shape is `[classCount]`.
def dataset : Trainer.Dataset [3, 8, 8] [10] := Data.fromLabeledSource source

end FromLabeledSource

-- doc-example: NN/API/Data/Sources.lean :: def requireFile
namespace RequireFile

-- A missing fixture should fail with a sentence the reader can act on.
def checkFixtures : IO Unit :=
  Data.requireFile "cifar10_images" "input tensor" "data/cifar_x.npy"
    (hint := "run: python3 NN/Examples/Data/generate_small_data.py")

end RequireFile

-- doc-example: NN/API/Data/Sources.lean :: def fromFiles (inputPath targetPath
namespace SupervisedSourceFromFiles

-- Shapes are per sample; `sampleCount` says how many leading rows of the files this run uses.
def source : Data.SupervisedSource :=
  Data.SupervisedSource.fromFiles
    (inputPath := "data/inputs.npy") (targetPath := "data/targets.npy")
    (sampleCount := 64) (input := [16]) (target := [1])

end SupervisedSourceFromFiles

-- doc-example: NN/API/Data/Loaders.lean :: def collateSupervised
namespace CollateSupervised

-- Three samples in, one batched sample out, with the batch size checked against the array length.
def batched (samples : Array (Sample.Supervised Float [2] [1])) :
    Except String (Sample.Batch Float 3 [2] [1]) :=
  Data.collateSupervised 3 samples

end CollateSupervised

end NN.Tests.API.DocExamples.Data

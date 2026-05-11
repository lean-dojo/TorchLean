/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN
public import NN.API.Samples.Bands

/-!
# Simple CNN training example

This is the small image-classification counterpart to `SimpleMlpTrain.lean`.

It remains in `Quickstart` as a next-step file for users who have finished the core five-file path
(`TensorBasics`, `Widgets`, `AutogradBasics`, `SimpleMlpTrain`, `Proofs`). For broader maintained
vision models, prefer `NN/Examples/Models`.

It uses the same public training API:

1. define a `SeqTask`,
2. build samples,
3. call `fit`,
4. inspect loss / accuracy.

The only difference is the model and the sample shape.

Check this tutorial module directly:

- `lake build NN.Examples.Quickstart.SimpleCnnTrain`

For the maintained command-line CNN trainer, use `NN/Examples/Models/Vision/Cnn.lean`:

- `python3 scripts/datasets/download_example_data.py --cifar10`
- `lake exe torchlean cnn --cpu --n-total 20 --steps 1`

Optional flags:

- `--epochs E`
- `--batch N`

Public API used here:

- `nn.conv`, `nn.relu`, `nn.flattenBatch`, `nn.linear`
- `train.classificationOneHot`
- `Data.labeled`, `Data.batchLoader`
- `train.fitLoaderWith` + `train.Callbacks`

Reader note:

- the model type here is already batched: `Images batch 1 4 4 -> batch × Vec 2`;
- `Semantics.Scalar α` / `Runtime.Scalar α` mean the same thing as in the other tutorials:
  `α` must both support the model's math and be executable as a runtime backend.

See `NN/Examples/Quickstart/README.md` for the shared conventions in this folder.
-/

@[expose] public section


namespace NN.Examples.Quickstart.SimpleCNNTrain

open Spec
open Tensor
open NN.Tensor
open NN.API

def mkModel {batch : Nat} :
    nn.M (nn.Sequential (Shape.Images batch 1 4 4) (shape![batch, 2])) :=
  -- Explicit, PyTorch-like layer stacking (batched `N×C×H×W` path):
  --
  --   Conv2d(1 -> 3, k=2x2, stride=1, padding=0)
  --   -> ReLU
  --   -> Flatten(start_dim=1) -> Linear(_, 2)
  let outC : Nat := 3
  let outH : Nat := (4 - 2) / 1 + 1
  let outW : Nat := (4 - 2) / 1 + 1
  let featInner : Shape := Shape.Image outC outH outW
  let featSize : Nat := Spec.Shape.size featInner
  nn.sequential![
    nn.conv (n := batch) (inC := 1) (inH := 4) (inW := 4)
      { outC := outC, kH := 2, kW := 2, stride := 1, padding := 0 },
    nn.relu,
    nn.flattenBatch,
    nn.linear featSize 2 (pfx := NN.Tensor.Shape.Vec batch)
  ]

def runOnce {batch : Nat} (task : train.Task (Shape.Images batch 1 4 4) (shape![batch, 2]))
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α] [API.Runtime.Scalar α]
    (runner : train.Runner α task) (epochs : Nat := 20) (seed : Nat := 0) : IO Unit := do
  let samplesF := API.Samples.Bands.trainCHWFloat
  let probes := API.Samples.Bands.probesCHW (α := α) API.Runtime.ofFloat
  let dataset : Data.Dataset (TorchLean.TList α [Shape.CHW 1 4 4, Shape.Vec 2]) :=
    Data.labeled (α := α) (σ := Shape.CHW 1 4 4) 2 samplesF
  let loader := Data.batchLoader dataset batch (shuffle := true) (seed := seed) (dropLast := true)
  let batchedDs ← API.Common.orThrow "SimpleCNNTrain" <| Data.BatchLoader.batchDataset loader

  IO.println "model = Conv2d(1, 3, 2x2) -> ReLU -> Flatten -> Linear(_, 2)"
  IO.println s!"dataset size = {dataset.size}"

  let opt := optim.adam 0.03
  let cfg := { (train.epochs epochs (optimizer := opt)) with logEvery := 0 }
  let hooks : train.Callbacks α :=
    (train.onTrainStart do
      train.withMode runner .eval do
        train.Report.reportLossAccuracyOneHotBatched (task := task) runner batchedDs "before"
        train.Report.reportClassProbesBatchedFromSingle
          (task := task) (runner := runner) probes
          "predictions(before)" (includeLogits := true))
    ++ train.logLossEvery 5
    ++ (train.onEpochEnd (fun ev =>
      train.withMode runner .eval do
        train.Report.reportLossAccuracyOneHotBatched (task := task) runner batchedDs
          s!"epoch {ev.epoch + 1}"))
    ++ (train.onTrainEnd (fun _ =>
      train.withMode runner .eval do
        train.Report.reportLossAccuracyOneHotBatched (task := task) runner batchedDs "after"
        train.Report.reportClassProbesBatchedFromSingle
          (task := task) (runner := runner) probes
          "predictions(after)" (includeLogits := true)))

  IO.println s!"training: Adam(lr=0.03), epochs={epochs}, batch_size={batch}, shuffle=true"
  let (_report, _loader') ← train.fitLoaderWith (task := task) runner cfg loader hooks

def main (args : List String) : IO Unit := do
  IO.println "== Quickstart next step: simple CNN training =="
  let args := API.CLI.dropDashDash args
  let (seed, args) ← API.Common.orThrow "SimpleCNNTrain" <| API.CLI.takeSeed args 0
  let (eb, args) ← API.Common.orThrow "SimpleCNNTrain" <| API.CLI.takeEpochBatch args 20 2
  if eb.batch = 0 then
    throw <| IO.userError "SimpleCNNTrain: --batch must be > 0"

  let task : train.Task (Shape.Images eb.batch 1 4 4) (shape![eb.batch, 2]) :=
    train.classificationOneHot (nn.build seed (mkModel (batch := eb.batch)))

  train.run task args (fun {α} _ _ _ _ runner rest => do
    API.Common.orThrow "SimpleCNNTrain" <| API.CLI.requireNoArgs rest
    runOnce (batch := eb.batch) (task := task) (α := α) runner (epochs := eb.epochs) (seed := seed))

end NN.Examples.Quickstart.SimpleCNNTrain

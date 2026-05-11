/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN
public import NN.API.Samples.Bands

/-!
# ResNet BasicBlock training example (small CHW)

This next-step file demonstrates the public `API.nn.blocks.resnetBasicBlock` builder on a small 4×4
"band" dataset (vertical vs horizontal bars).

It is intentionally not part of the first quickstart path. It is here as a compact bridge from the
basic MLP/CNN tutorials to the maintained model-zoo ResNet example under `NN/Examples/Models`.

Note: this particular example uses BatchNorm blocks, so the model is built for a fixed batch size.
Most other tutorials in this folder use batch-free models and let `API.train.runLoader*` lift them.

Check this tutorial module directly:

- `lake build NN.Examples.Quickstart.ResnetBasicblockTrain`

For the maintained command-line ResNet trainer, use `NN/Examples/Models/Vision/Resnet.lean`:

- `python3 scripts/datasets/download_example_data.py --cifar10`
- `lake exe torchlean resnet --cpu --n-total 20 --steps 1`

Optional flags:

- `--epochs E`
- `--quick` (skip expensive before/after evaluation reports; intended for CI/smoke runs)

Public API used here:

- `nn.conv`, `nn.batchNorm`, `nn.resnetBasicBlock`
- `nn.flattenBatch`, `nn.linear`
- `Data.batchLoader`
- `train.fitLoaderWith` + `train.Callbacks`

Reader note:

- this is the one tutorial here whose model is *already batched* in its type because BatchNorm uses
  cross-batch statistics;
- so the task shape is fixed to the chosen `--batch` size and the loader uses `drop_last=true`;
- `Semantics.Scalar α` / `Runtime.Scalar α` still mean "the chosen scalar backend supports the math"
  and "the backend is executable".

See `NN/Examples/Quickstart/README.md` for the shared conventions in this folder.

Why this tutorial exists:

- it shows how TorchLean exposes residual CNN blocks, not just plain sequential stacks;
- it gives users one clean example of fixed-batch training, which matters once BatchNorm enters the
  model;
- it stays small enough to read in one sitting while still looking recognizably "ResNet-like".
-/

@[expose] public section


namespace NN.Examples.Quickstart.ResNetBasicBlockTrain

open Spec
open Tensor
open NN.Tensor
open NN.API

def mkModel (n : Nat) [NeZero n] :
    nn.M (nn.Sequential (Shape.Images n 1 4 4) (shape![n, 2])) :=
  -- A compact ResNet-style CNN:
  --   Conv3x3(1 -> 8) + BN + ReLU
  --   BasicBlock(8 -> 8)
  --   BasicBlock(8 -> 16, downsample)
  --   Flatten(start_dim=1) -> Linear(_, 2)
  let conv : nn.Conv :=
    { outC := 8
      kH := 3, kW := 3
      stride := 1, padding := 1 }
  -- Conv output: `H×W` stays 4×4 here, but we keep the formula so the model composition remains
  -- definitional if the input/config changes.
  let h1 : Nat := (4 + 2 * conv.padding - conv.kH) / conv.stride + 1
  let w1 : Nat := (4 + 2 * conv.padding - conv.kW) / conv.stride + 1
  let h2 : Nat := nn.blocks.down2 h1
  let w2 : Nat := nn.blocks.down2 w1
  let featInner : Shape := Shape.Image 16 h2 w2
  let featSize : Nat := Spec.Shape.size featInner
  nn.sequential![
    nn.conv (n := n) (inC := 1) (inH := 4) (inW := 4) conv,
    nn.batchNorm (n := n) (c := 8) (h := h1) (w := w1),
    nn.relu,
    nn.resnetBasicBlock (n := n) (inC := 8) (h := h1) (w := w1)
      { outC := 8, downsample := false, activation := .relu },
    nn.resnetBasicBlock (n := n) (inC := 8) (h := h1) (w := w1)
      { outC := 16, downsample := true, activation := .relu },
    nn.flattenBatch,
    nn.linear featSize 2 (pfx := Shape.Vec n)
  ]

/--
Run one training session on the small band-image dataset.

This is intentionally separated from `main` so the tutorial keeps a clean “build task, choose
backend, then train” structure.
-/
def runOnce {batch : Nat} (task : train.Task (Shape.Images batch 1 4 4) (shape![batch, 2]))
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α] [API.Runtime.Scalar α]
  (runner : train.Runner α task)
  (epochs : Nat := 20)
    (seed : Nat := 0)
    (quick : Bool := false) : IO Unit := do
  let samplesF := API.Samples.Bands.trainCHWFloat
  let probes := API.Samples.Bands.probesCHW (α := α) API.Runtime.ofFloat
  let dataset : Data.Dataset (TorchLean.TList α [Shape.CHW 1 4 4, Shape.Vec 2]) :=
    Data.labeled (α := α) (σ := Shape.CHW 1 4 4) 2 samplesF
  let loader := Data.batchLoader dataset batch (shuffle := true) (seed := seed) (dropLast := true)

  IO.println "model = Conv+BN+ReLU -> BasicBlock -> BasicBlock(downsample) -> Flatten -> Linear(2)"
  IO.println s!"dataset size = {dataset.size}"

  let opt := optim.adam 0.03
  let cfg := { (train.epochs epochs (optimizer := opt)) with logEvery := 0 }
  let hooks : train.Callbacks α ←
    if quick then
      pure <| train.logLossEvery (α := α) 1
    else
      let batchedDs ← API.Common.orThrow "ResNetBasicBlockTrain" <| Data.BatchLoader.batchDataset
        loader
      -- The richer path mirrors how a user would inspect a real image classifier:
      -- check accuracy and a few probe predictions before training, during training, and after.
      pure <|
        (train.onTrainStart do
          train.withMode runner .eval do
            train.Report.reportLossAccuracyOneHotBatched (task := task) runner batchedDs "before"
            train.Report.reportClassProbesBatchedFromSingle
              (task := task) (runner := runner) probes "predictions(before)")
        ++ train.logLossEvery (α := α) 1
        ++ (train.onEpochEnd (fun ev =>
          train.withMode runner .eval do
            train.Report.reportLossAccuracyOneHotBatched (task := task) runner batchedDs
              s!"epoch {ev.epoch + 1}"))
        ++ (train.onTrainEnd (fun _ =>
          train.withMode runner .eval do
            train.Report.reportLossAccuracyOneHotBatched (task := task) runner batchedDs "after"
            train.Report.reportClassProbesBatchedFromSingle
              (task := task) (runner := runner) probes "predictions(after)"))

  let (_report, _loader') ← train.fitLoaderWith (task := task) runner cfg loader hooks

def main (args : List String) : IO Unit := do
  IO.println "== Quickstart next step: ResNet BasicBlock training (small CHW) =="
  let args0 := API.CLI.dropDashDash args
  let quick := args0.contains "--quick"
  let args := args0.filter (fun a => a != "--quick")
  let (seed, args) ← API.Common.orThrow "ResNetBasicBlockTrain" <| API.CLI.takeSeed args 0
  let (eb, args) ← API.Common.orThrow "ResNetBasicBlockTrain" <| API.CLI.takeEpochBatch args 20 2
  if h : eb.batch = 0 then
    throw <| IO.userError "ResNetBasicBlockTrain: --batch must be > 0"
  else
    -- BatchNorm builders require a nonzero batch size; the CLI enforces it.
    letI : NeZero eb.batch := ⟨h⟩

    let task : train.Task (Shape.Images eb.batch 1 4 4) (shape![eb.batch, 2]) :=
      train.classificationOneHot (nn.build seed (mkModel (n := eb.batch)))

    train.run task args (fun {α} _ _ _ _ runner rest => do
      API.Common.orThrow "ResNetBasicBlockTrain" <| API.CLI.requireNoArgs rest
      runOnce (batch := eb.batch) (task := task) (α := α) runner (epochs := eb.epochs) (seed :=
        seed) (quick := quick))

end NN.Examples.Quickstart.ResNetBasicBlockTrain

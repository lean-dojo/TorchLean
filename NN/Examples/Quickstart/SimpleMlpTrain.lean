/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN

/-!
# Simple MLP training example (regression)

This is a focused end-to-end example of training a small MLP in TorchLean.

It mirrors the simplest PyTorch workflow:

1. build a small synthetic dataset (in-memory),
2. define an MLP (`Linear -> ReLU -> Linear`),
3. train with Adam,
4. report loss before/after, plus a few probe predictions.

Run:

- `lake exe torchlean quickstart_mlp`
- `lake exe torchlean quickstart_mlp --steps 200 --dtype float --backend eager`

Optional flags (tutorial-specific):

- `--seed S` (model init + any shuffling)
- `--steps N`
-/

@[expose] public section


namespace NN.Examples.Quickstart.SimpleMLPTrain

open Spec
open Tensor
open NN.Tensor
open NN.API

def inDim : Nat := 2
def outDim : Nat := 1

/-- A small 2-layer MLP `2 -> 8 -> 1`. -/
def mkModel : nn.M (nn.Sequential (Shape.Vec inDim) (Shape.Vec outDim)) :=
  nn.sequential![
    nn.linear inDim 8 (pfx := Spec.Shape.scalar),
    nn.relu,
    nn.linear 8 outDim (pfx := Spec.Shape.scalar)
  ]

/-- Regression task: model + MSE loss. -/
def task (seed : Nat) :=
  train.regression (nn.build seed mkModel)

/--
Small piecewise-linear regression target:

`y = 0.8 * relu(x1 + x2) - 0.4 * relu(x2 - x1) + 0.2`.

This is easy for a compact ReLU MLP to fit, which keeps the quickstart dependable.
-/
def target {α : Type} [Semantics.Scalar α] [Runtime.Scalar α] (x1 x2 : α) : α :=
  (0.8 * Semantics.relu (x1 + x2)) - (0.4 * Semantics.relu (x2 - x1)) + 0.2

/--
Build the training dataset at the runtime-selected scalar type `α`.

We write the sample coordinates as `Float` literals first because they are convenient to read,
and `Data.supervisedDim0F` lifts them into the chosen runtime scalar backend.
-/
def buildDataset {α : Type} [Semantics.Scalar α] [Runtime.Scalar α] :
    _root_.Runtime.Autograd.Train.Dataset (sample.Supervised α (Shape.Vec inDim) (Shape.Vec outDim))
      :=
  -- TensorDataset-style path (PyTorch analogue: `TensorDataset(X, Y)`).
  --
  -- We build a small coordinate grid as a batched tensor `X : (n,2)`, then compute
  -- the corresponding batched targets `Y : (n,1)`.
  let X : Spec.Tensor Float (shape![5 * 5, inDim]) :=
    API.Samples.grid2Square (-1.0) 1.0 5
  let Y : Spec.Tensor Float (shape![5 * 5, outDim]) :=
    API.Samples.regression2to1Float X (fun x1 x2 => target (α := Float) x1 x2)
  Data.supervisedDim0F (α := α) X Y

def main (args : List String) : IO Unit := do
  let args := API.CLI.dropDashDash args
  let (seed, args) ← API.Common.orThrow "SimpleMLPTrain" <| API.CLI.takeSeed args 0
  let (steps?, args) ← API.Common.orThrow "SimpleMLPTrain" <| API.CLI.takeNatFlagOnce args "steps"

  let steps := steps?.getD 200
  if steps = 0 then
    throw <| IO.userError "SimpleMLPTrain: --steps must be > 0"

  let task := task (seed := seed)

  IO.println "== Quickstart: simple MLP training =="
  IO.println s!"seed  = {seed}"
  IO.println s!"steps = {steps}"
  IO.println "model = MLP(2 -> 8 -> 1)"

  train.run task args (fun {α} _ _ _ _ runner rest => do
    API.Common.orThrow "SimpleMLPTrain" <| API.CLI.requireNoArgs rest

    let dataset := buildDataset (α := α)
    IO.println s!"dataset size = {dataset.size}"

    let reportProbes := fun (title : String) => do
      IO.println title
      for (name, x1, x2) in [("center", 0.0, 0.0), ("heldout", 0.25, -0.75)] do
        let x : Spec.Tensor α (Shape.Vec inDim) := API.Samples.vec2 (α := α) API.Runtime.ofFloat x1
          x2
        let yhat ← train.predict (task := task) runner x
        IO.println
          s!"  {name}: x=({x1},{x2})  target={target (α := Float) x1 x2}  pred={Spec.pretty yhat}"

    -- Before training.
    train.Report.reportMeanLoss (task := task) runner dataset "before"
    reportProbes "predictions(before)"

    -- Train.
    let cfg := train.steps steps (optimizer := optim.adam 0.03) (logEvery := 25)
    let _report ← train.fitDataset (task := task) runner cfg dataset

    -- After training.
    train.evalMode (task := task) runner
    train.Report.reportMeanLoss (task := task) runner dataset "after"
    reportProbes "predictions(after)"
    pure ())

end NN.Examples.Quickstart.SimpleMLPTrain

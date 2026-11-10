/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Trainer.Train
public import TorchLean.Verify.Trainer
public import TorchLean.NN.Data.Dataset
public import TorchLean.NN.Module.Command
public import TorchLean.NN.Trainer.Constructor

/-!
# Train a Classifier, Then Verify Robustness

The complete high-level workflow: build and train a classifier, then verify that its class cannot
change inside an L-infinity input ball. `trained.verify` does the graph lowering and the bound pass
itself.

Training:
- build a two-layer ReLU classifier (2 → 8 → 2)
- train it with one-hot cross-entropy (SGD at learning rate 0.1, 120 steps)

Verification:
- call `trained.verify` over the L∞ ball of radius 0.10 around the center
- run the default `crown` method over the trained parameters, directed CROWN on the rounded
  scalars this workflow uses
- report typed output bounds, the worst-case class margin, and the certification result

Run:
  `scripts/lake.sh exe verify -- torchlean-mlp-workflow`
  `scripts/lake.sh exe verify -- torchlean-mlp-workflow --arithmetic ieee`
-/

@[expose] public section


namespace TorchLean.Examples.MLPTrainVerifyWorkflow

open Trainer.Objective

/-- Eight two-dimensional training inputs, separated by the sign of the first coordinate. -/
def inputs : Tensor Float [8, 2] :=
  [[2.0, 0.0],
   [1.5, 0.5],
   [1.5, -0.5],
   [1.0, 0.0],
   [-2.0, 0.0],
   [-1.5, 0.5],
   [-1.5, -0.5],
   [-1.0, 0.0]]

/-- One-hot labels: positive first coordinate is class zero, negative is class one. -/
def labels : Tensor Float [8, 2] :=
  [[1.0, 0.0],
   [1.0, 0.0],
   [1.0, 0.0],
   [1.0, 0.0],
   [0.0, 1.0],
   [0.0, 1.0],
   [0.0, 1.0],
   [0.0, 1.0]]

/-- The workflow model `Linear(2 → 8)`, ReLU, `Linear(8 → 2)`, built from initialization seed 0. -/
def model : nn.Sequential [2] [2] :=
  nn.build 0 <| nn.Sequential![nn.linear 2 8, nn.op .relu, nn.linear 8 2]

/--
Train `model` on the eight labelled points under the scalar backend `α`, then check that class 0
stays the top label on the ℓ∞ ball of radius `0.1` around `(1.5, 0)`.

The trained result owns its parameter snapshot, so `trained.verify` checks the parameters that
training produced.
-/
def run {α : Type} [Storage α] [Context α] [ToString α]
    [FromFloat α] (options : Session.Config) : IO Unit := do
  let dataset := Data.fromTensors inputs labels
  let trainer := Trainer.create model <|
    { options with
        optimizer := optim.sgd { lr := 0.1 }
        objective := oneHotCrossEntropy 0
        seed := 0 }
  IO.println "== Train and verify classifier (2 → 8 → 2) =="
  IO.println
    s!"Training on device={options.device.cliName}"
  let trained ← trainer.train dataset { steps := 120 }
  IO.println s!"avg_loss(on samples)={trained.report.loss.after}"
  let center : Tensor Float [2] := [1.5, 0.0]
  let prediction ← trained.predict center
  IO.println s!"prediction at center={Tensor.pretty prediction}"
  IO.println "Checking class 0 throughout ||x - center||_∞ ≤ 0.10"
  let report ← trained.verify center
    (radius := 0.10)
    (property := .topLabel 0)
  report.printSummary

/--
CLI entry point of `scripts/lake.sh exe verify -- torchlean-mlp-workflow`. It takes only the
runtime flags and rejects `--device cuda`, because the verifier lowers parameters held as Lean
tensors.
-/
def main (args : List String) : IO Unit := do
  Module.withSelectedRuntime args
    (fun {α} _ _ _ _ _cast options rest => do
      CLI.requireNoArgs "torchlean-mlp-workflow" rest
      if options.usesCUDA then
        throw <| IO.userError
          ("torchlean-mlp-workflow: CUDA eager training is not used here; this workflow keeps " ++
            "trained parameters as Lean tensors so the verifier can lower and check them. " ++
            "Use the model-training examples for CUDA runtime training, or run this verifier " ++
            "workflow without --device cuda.")
      run (α := α) options)

end TorchLean.Examples.MLPTrainVerifyWorkflow

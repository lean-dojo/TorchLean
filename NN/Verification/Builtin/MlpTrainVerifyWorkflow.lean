/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Verification
public import NN.API.Data.Training
public import NN.API.Module.Command
public import NN.API.Trainer.Constructor

/-!
# Train a Classifier, Then Verify Robustness

This is the complete high-level workflow: build and train a classifier, then verify that its class
cannot change inside an L-infinity input ball. Graph lowering and the bound pass remain inside the
normal `trained.verify` operation.

Training:
- build a two-layer ReLU classifier
- train it with one-hot cross-entropy

Verification:
- call `trained.verify` with `norm := .inf`
- run fixed-relaxation α-CROWN with IBP-inferred ReLU phases over the trained parameters (the
  API calls this algorithm `.alphaBetaCrown`; it has no β multipliers or branch splits)
- report typed output bounds, the worst-case class margin, and the certification result

Run:
  `scripts/lake.sh exe verify -- torchlean-mlp-workflow`
  `scripts/lake.sh exe verify -- torchlean-mlp-workflow --arithmetic ieee`
-/

@[expose] public section


namespace NN.Verification.Builtin.MlpTrainVerifyWorkflow

open Spec TorchLean

/-- Linearly separable two-dimensional training inputs. -/
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

/-- Deterministically instantiate the workflow model from initialization seed zero. -/
def model : nn.Sequential [2] [2] :=
  nn.build 0 <| nn.Sequential![nn.linear 2 8, nn.relu, nn.linear 8 2]

/--
Run training and verification under a chosen scalar backend `α`.

The trained result owns the trained parameters. The robustness call therefore checks the model that
was actually trained.
-/
def run {α : Type} [Storage α] [Context α] [ToString α]
    [Runtime.FromFloat α] (options : Runtime.Config) : IO Unit := do
  let dataset := Data.fromTensors inputs labels
  let trainer := Trainer.new model <|
    Trainer.RunConfig.forObjective
      (Trainer.RunConfig.fromRuntime options { optimizer := optim.sgd { learningRate := 0.1 } })
      (.oneHotCrossEntropy 0)

  IO.println "== Train and verify classifier (2 → 8 → 2) =="
  IO.println
    s!"Training with execution={reprStr options.execution}, device={options.device.cliName}"
  let trained ← trainer.train dataset { steps := 120 }
  IO.println s!"avg_loss(on samples)={trained.report.loss.after}"
  let center : Tensor Float [2] := [1.5, 0.0]
  let prediction ← trained.predict center
  IO.println s!"prediction at center={Spec.pretty prediction}"
  IO.println "Checking class 0 throughout ||x - center||_∞ ≤ 0.10"
  let report ← trained.verify center
    (radius := 0.10)
    (norm := .inf)
    (property := .topLabel 0)
  report.printSummary

/--
CLI entry point for the native TorchLean MLP workflow.

This is wired into `scripts/lake.sh exe verify -- torchlean-mlp-workflow`.
-/
def main (args : List String) : IO Unit := do
  let args :=
    if CLI.hasFlagValue args "execution" then
      args
    else
      "--execution=typed-graph" :: args
  Module.withSelectedRuntime args
    (fun {α} _ _ _ _ _cast options rest => do
      CLI.requireNoArgs "torchlean-mlp-workflow" rest
      if options.usesCuda then
        throw <| IO.userError
          ("torchlean-mlp-workflow: CUDA eager training is not used here; this workflow keeps " ++
            "trained parameters as Lean tensors so the verifier can lower and check them. " ++
            "Use the model-training examples for CUDA runtime training, or run this verifier " ++
            "workflow without --device cuda.")
      run (α := α) options)

end NN.Verification.Builtin.MlpTrainVerifyWorkflow

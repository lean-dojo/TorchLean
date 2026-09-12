/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded
public import NN.API.Trainer.Constructor
public import NN.API.Trainer.Train.Loop
public import NN.API.Data.Training

/-!
# Compiled Docstring Examples: Training

Every `Example:` block in a `TorchLean.Trainer` or `TorchLean.optim` docstring appears here
verbatim, one namespace per entry point. Nothing in this file runs: the guarantee we want is that
the snippet a reader copies out of the docstring still elaborates, and typechecking gives us
exactly that.

Snippets take the trainer and the dataset as parameters rather than building fixtures, so each one
stands on its own in the docstring where it lands.

`repo_lint.py` compares the fenced `lean` block inside each `Example:` against the namespace body
carrying the matching `doc-example:` marker, so the two cannot drift apart.
-/

@[expose] public section

namespace NN.Tests.API.DocExamples.Training

open TorchLean

-- doc-example: NN/API/Trainer/Constructor.lean :: def new
namespace New

-- A trainer pairs a model with the loss, the optimizer, and the runtime it trains under.
def model : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![nn.linear 2 8, nn.relu, nn.linear 8 1]

def trainer : TorchLean.Trainer [2] [1] :=
  Trainer.new model
    { objective := .meanSquaredError
      optimizer := optim.adam { learningRate := 0.03 }
      seed := 7 }

-- An already-built model is accepted too, and then the seed has nothing left to decide.
def fromBuiltModel : TorchLean.Trainer [2] [1] :=
  Trainer.new (nn.build 7 model)

end New

-- doc-example: NN/API/Trainer/Core.lean :: structure RunConfig
namespace RunSettings

-- The defaults train on the CPU in Lean's `Float32`. `.ieee` swaps in the bit-level reference
-- semantics, which is the setting to reach for when a result looks like a rounding artifact.
def settings : Trainer.RunConfig :=
  { optimizer := optim.sgd { learningRate := 0.01, momentum := 0.9 }
    arithmetic := .native
    execution := .eager
    device := .cpu }

end RunSettings

-- doc-example: NN/API/Trainer/Core.lean :: inductive Objective
namespace Objectives

-- Regression scores a prediction against a target tensor.
def regression : Trainer.Objective [1] := .meanSquaredError

-- Classification needs the axis the logits live on, here the only axis of a ten-class output.
def classification : Trainer.Objective [10] := .oneHotCrossEntropy 0

end Objectives

-- doc-example: NN/API/Trainer/Run.lean :: structure TrainOptions
namespace Options

-- `steps` has no default on purpose: how long to train is not a library's decision.
def options : Trainer.TrainOptions :=
  { steps := 200
    logEvery := 25
    saveCheckpoint? := some "checkpoints/mlp.state" }

end Options

-- doc-example: NN/API/Trainer/Run.lean :: def tensor
namespace Probes

-- Probes print a named prediction before and after the run, so training shows its movement
-- without a separate evaluation script.
def probes : Array (Trainer.Probe [2]) :=
  #[Trainer.Probe.tensor "heldout" [0.25, -0.75] (inputText := "x = (0.25, -0.75)")]

end Probes

-- doc-example: NN/API/Trainer/Train/Loop.lean :: def train
namespace Train

-- `trained` owns the parameters the run produced; `trainer` still describes the untrained model.
def run (trainer : TorchLean.Trainer [2] [1])
    (data : Trainer.Dataset [2] [1]) : IO Unit := do
  let trained ← trainer.train data { steps := 200, logEvery := 25 }
  trained.printSummary
  let prediction ← trained.predict [0.25, -0.75]
  IO.println s!"trained(heldout) = {reprStr prediction}"

end Train

-- doc-example: NN/API/Trainer/Train/Loop.lean :: def predict
namespace Predict

-- This is inference with the freshly initialized parameters, which is what makes it a useful
-- baseline to print before training starts.
def baseline (trainer : TorchLean.Trainer [2] [1]) : IO (Tensor Float [1]) :=
  trainer.predict [0.25, -0.75]

end Predict

-- doc-example: NN/API/Trainer/Train/Loop.lean :: def load
namespace Load

-- Reads back what `trained.save` wrote and re-measures the report on `data`, so both reported
-- losses are the restored model's loss and the step count is `0`.
def restore (trainer : TorchLean.Trainer [2] [1])
    (data : Trainer.Dataset [2] [1]) : IO (Trainer.Result [2] [1]) :=
  trainer.load "checkpoints/mlp.state" data

end Load

-- doc-example: NN/API/Trainer/Train/Loop.lean :: def trainStream
namespace TrainStream

-- No fixed dataset here: a sample is generated per step, and the loss curve is measured on one
-- held-out sample every `curveEvery` updates.
def sampleAt (step : Nat) : Sample.Supervised Float [1] [1] :=
  let x := step.toFloat * 0.01
  { input := [x], target := [2.0 * x] }

def run (trainer : TorchLean.Trainer [1] [1]) (runtime : Runtime.Config) :
    IO (Trainer.StreamResult [1] [1]) :=
  trainer.trainStream runtime sampleAt
    { input := [0.5], target := [1.0] }
    { steps := 500, logEvery := 100 }
    (curveEvery := 50)

end TrainStream

-- doc-example: NN/API/Trainer/Session.lean :: def «open»
namespace Open

-- A session holds the instantiated model and the bound optimizer. `trainer.train` is exactly this
-- call, a loop of `step`, and a `finish`, so opening a session is how you take that loop over.
def stepOnce (trainer : TorchLean.Trainer [2] [1]) : IO Float := do
  let session ← trainer.open
  session.step { input := [1.0, 0.0], target := [1.0] }

end Open

-- doc-example: NN/API/Trainer/Session.lean :: opaque step
namespace Step

-- One update per call, returning that sample's loss. Batching several samples into a single
-- update is `stepBatch`, not repeated `step` calls.
def descend (trainer : TorchLean.Trainer [2] [1]) : IO (Trainer.Result [2] [1]) := do
  let session ← trainer.open
  let sample : Sample.Supervised Float [2] [1] := { input := [1.0, 0.0], target := [1.0] }
  let before ← session.loss sample
  for _ in List.range 100 do
    let _ ← session.step sample
  let after ← session.loss sample
  session.finish { before := before, after := after }

end Step

-- doc-example: NN/API/Trainer/Session.lean :: opaque finish
namespace Finish

-- Keep a model snapshot with the losses measured for it. The session can continue training
-- without changing this result.
def package {trainer : TorchLean.Trainer [2] [1]}
    (session : Trainer.Session trainer) (before after : Float) :
    IO (Trainer.Result [2] [1]) :=
  session.finish { before := before, after := after }

end Finish

-- doc-example: NN/API/Optim/Config.lean :: def sgd
namespace Sgd

-- `torch.optim.SGD(params, lr=0.01)`, then the same with heavy-ball momentum.
def plain : optim.Optimizer := optim.sgd { learningRate := 0.01 }

def withMomentum : optim.Optimizer :=
  optim.sgd { learningRate := 0.01, momentum := 0.9 }

end Sgd

-- doc-example: NN/API/Optim/Config.lean :: def adam
namespace Adam

-- `torch.optim.Adam(params, lr=1e-3)`: same default moments, same stabilizer
-- (Kingma and Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015).
def optimizer : optim.Optimizer := optim.adam { learningRate := 1e-3 }

end Adam

-- doc-example: NN/API/Optim/Config.lean :: def adamW
namespace AdamW

-- Decoupled weight decay, so the penalty does not travel through the adaptive moments
-- (Loshchilov and Hutter, "Decoupled Weight Decay Regularization", ICLR 2019).
def optimizer : optim.Optimizer :=
  optim.adamW { learningRate := 1e-3, weightDecay := 0.01 }

end AdamW

end NN.Tests.API.DocExamples.Training

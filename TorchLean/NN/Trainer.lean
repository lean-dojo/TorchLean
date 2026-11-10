/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Sequential.Summary
public import TorchLean.NN.Models.FNO.Block -- shake: keep
public import TorchLean.NN.Models.Recurrent.Stack -- shake: keep
public import TorchLean.NN.Models.VQVAE.Autograd -- shake: keep
public import TorchLean.NN.Module -- shake: keep
public import TorchLean.NN.Trainer.Summary -- shake: keep

/-!
# Trainer

Create a trainer from a checked model, choose its loss and optimizer, then train or predict:

```lean
let trainer := Trainer.create model
  { objective := .mse
    optimizer := optim.adam { lr := 0.03 } }
let y0 ← trainer.predict x
let trained ← trainer.train data { steps := 200, samplesPerStep := 16, logEvery := 25 }
trained.printSummary
trained.save "model.state"
```

`trainer.train` and `trainer.open` accept `Tensor Float` values and execute in the selected
binary32 scalar: `Float32` for `native` or FloatLib's `ExecFloat.Binary 8 23` for `ieee`.
Results record that selection and read the binary32 values back to `Float`.

For a different scalar, `trainer.openTyped (α := α)` keeps inputs, parameters, predictions, and
losses in `α` throughout a CPU session. Supply `initialState?` with typed state when initialization
must preserve digits beyond the model's stored seeded `Float` values. Both opening paths return
a `Trainer.Session` for programs that drive the optimizer loop themselves.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

namespace TorchLean

namespace Trainer

universe u

/--
Runtime, device, arithmetic, and optimizer settings for a trainer or one training call.

The runtime fields (`device`, `seed`, `backendProfile?`, `showBackend`) are those of
`Session.Config`, which this record extends, so `{ runtime with optimizer := ... }` builds one
from a runtime configuration.

Example:
```lean
-- The defaults train on the CPU in Lean's `Float32`. `ieee` swaps in the bit-level reference
-- semantics, which is the setting to reach for when a result looks like a rounding artifact.
def settings : Trainer.RunConfig :=
  { optimizer := optim.sgd { lr := 0.01, momentum := 0.9 }
    arithmetic := .native
    device := .cpu }
```
-/
structure RunConfig extends Session.Config where
  /-- Optimizer used unless a training call supplies another run configuration. -/
  optimizer : optim.Optimizer := optim.sgd { lr := 0.01 }
  /--
  Arithmetic semantics selected by `trainer.open` and the managed training methods.

  `native` trains in Lean's `Float32` and `ieee` in FloatLib's `ExecFloat.Binary 8 23`. Both are
  binary32, and `Tensor Float` values are converted at the boundary. These methods reject
  `complex`, while `trainer.openTyped` runs in the scalar it is given.
  -/
  arithmetic : Arithmetic := .native

/--
Loss used to train a model with output shape `output`.

The objective decides only how a `(prediction, target)` pair becomes a scalar, so the output shape
is its only index.

Example:
```lean
-- Regression scores a prediction against a target tensor.
def regression : Trainer.Objective [1] := .mse

-- Classification needs the axis the logits live on, here the only axis of a ten-class output.
def classification : Trainer.Objective [10] := .oneHotCrossEntropy 0
```
-/
inductive Objective (output : Shape) where
  /-- Mean-squared-error supervised regression. -/
  | mse (reduction : Loss.Reduction := .mean)
  /-- One-hot cross entropy over a class or structured logit tensor. -/
  | oneHotCrossEntropy (axis : Nat)
      (reduction : Loss.Reduction := .mean)
  /-- A checked TorchLean loss program supplied by the caller. -/
  | custom
      (loss : ∀ {α : Type}, [TorchLean.Storage α] → [Context α] →
        Program α
          [output, output] ([] : Shape))

namespace Objective

/--
Lower an objective for a checked model into the runtime module definition that the trainer
instantiates.

Training mode is the default, and `nn.Mode.eval` builds the graph that stateful layers use for
inference.
-/
def definition {σ τ : Shape} (objective : Objective τ)
    (model : TorchLean.nn.Sequential σ τ) (mode : nn.Mode := .train) :
    ObjectiveDef Unit (nn.Sequential.stateShapes model) [σ, τ] :=
  match objective with
  | .mse reduction =>
      nn.Objective.mse
        (model := model) (reduction := reduction) (mode := mode)
  | .oneHotCrossEntropy axis reduction =>
      nn.Objective.oneHotCrossEntropy
        (model := model) axis (reduction := reduction) (mode := mode)
  | .custom loss =>
      nn.Objective.fromLoss model loss mode

end Objective

/--
Model-independent options accepted by `Trainer.create`: the run settings of `RunConfig` and the
training objective.

Managed training runs in the binary32 scalar that `arithmetic` selects and exchanges `Float`
values with the caller, while `trainer.openTyped` keeps the scalar it is given in the session and
its finished result.
-/
structure Config (input output : Shape) extends RunConfig where
  /-- Training objective attached to this trainer. -/
  objective : Objective output := .mse

end Trainer

/--
A checked model together with its loss, runtime settings and initialization seed.

`Trainer.create` builds one, and dot notation gives prediction, training and model inspection.
-/
structure Trainer (input output : Shape) where
  /-- Checked TorchLean model. -/
  model : TorchLean.nn.Sequential input output
  /-- Supervised objective used by `train`. -/
  objective : Trainer.Objective output
  /-- Runtime, backend-contract, and optimizer choices carried by this trainer. When
  `Trainer.create` received a `TorchLean.nn.Builder`, its `seed` initialized the model, and changing
  the seed afterwards does not rebuild it. -/
  runtime : Trainer.RunConfig := {}

namespace Trainer

/-- Structured checked-model summary for this trainer. -/
def summary {σ τ : Shape}
    (trainer : TorchLean.Trainer σ τ) : Except String nn.ModelSummary :=
  nn.summary trainer.model

/-- Print the checked-model summary under a caller-chosen heading. -/
def printSummary {σ τ : Shape}
    (trainer : TorchLean.Trainer σ τ)
    (label : String := "model") : IO Unit := do
  IO.println s!"{label}:"
  let details ← IO.ofExcept trainer.summary
  IO.println details

end Trainer

end TorchLean

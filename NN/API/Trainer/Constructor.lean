/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Trainer.Core

/-!
# Trainer Construction

Constructors for `TorchLean.Trainer`.
-/

@[expose] public section

namespace TorchLean

namespace Trainer

universe u

/--
Values accepted by `Trainer.new`.

This class lets `Trainer.new` accept either a seedable model builder or an already-built checked
model:

```lean
Trainer.new modelBuilder ...
Trainer.new alreadyBuiltModel ...
```

The seed is consumed only by the builder case. Already-built models pass through unchanged.
-/
class ToModel (Model : Type u)
    (input output : outParam Shape) where
  /-- Materialize the model, using the seed only when the value still needs initialization. -/
  build : Nat → Model → TorchLean.nn.Sequential input output

instance {σ τ : Shape} :
    ToModel (TorchLean.nn.Sequential σ τ) σ τ where
  build _ model := model

instance {σ τ : Shape} :
    ToModel (TorchLean.nn.Builder (TorchLean.nn.Sequential σ τ)) σ τ where
  build seed model := TorchLean.nn.build seed model

/--
Build a trainer from a sequential model or seedable model builder.

Example:
```lean
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
```
-/
def new {Model : Type u} {σ τ : Shape}
    [ToModel Model σ τ] (model : Model)
    (config : Config σ τ := {}) :
    TorchLean.Trainer σ τ :=
  let builtModel := ToModel.build config.seed model
  { model := builtModel
    objective := config.objective
    runtime := config.toRunConfig
    seed := config.seed }

end Trainer

end TorchLean

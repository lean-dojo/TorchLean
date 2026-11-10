/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Trainer

/-!
# Trainer Construction

`Trainer.create` builds a `TorchLean.Trainer` from a checked model or a seedable model builder, and
the `ToModel` class lets it take either.
-/

@[expose] public section

namespace TorchLean

namespace Trainer

universe u

/--
Values accepted by `Trainer.create`: a seedable model builder or an already-built checked model.

```lean
Trainer.create modelBuilder ...
Trainer.create alreadyBuiltModel ...
```

Only the builder uses the seed, and an already-built model passes through unchanged.
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
Build a trainer from a sequential model or from a seedable model builder, which `config.seed`
initializes.

Example:
```lean
-- A trainer pairs a model with the loss, the optimizer, and the runtime it trains under.
def model : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![nn.linear 2 8, nn.op .relu, nn.linear 8 1]

def trainer : TorchLean.Trainer [2] [1] :=
  Trainer.create model
    { objective := mse
      optimizer := optim.adam { lr := 0.03 }
      seed := 7 }

-- An already-built model is accepted too, and then the seed has nothing left to decide.
def fromBuiltModel : TorchLean.Trainer [2] [1] :=
  Trainer.create (nn.build 7 model)
```
-/
def create {Model : Type u} {σ τ : Shape}
    [ToModel Model σ τ] (model : Model)
    (config : Config σ τ := {}) :
    TorchLean.Trainer σ τ :=
  let builtModel := ToModel.build config.seed model
  { model := builtModel
    objective := config.objective
    runtime := config.toRunConfig }

end Trainer

end TorchLean

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

-- This module supplies public namespace exports used by downstream consumers. Import shaking
-- cannot see those downstream lookups, so keep the marked imports.
module -- shake: keep-downstream

public import NN.API.Arithmetic
public import NN.API.Optim.Config -- shake: keep
public import NN.Runtime.Autograd.Model.Optim -- shake: keep
public import NN.Tensor -- shake: keep

/-!
# Optimizers

Optimizer configuration records and runtime optimizer constructors.

The default trainer config exposes self-contained core update rules for SGD, momentum SGD,
AdaGrad, RMSProp, Adam, AdamW, and Adadelta. Runtime-only extension points live here too:

- Muon is an optimizer, but `optim.muon.optimizer` requires an explicit orthogonalization backend.
  The identity backend is available for proofs and fallback behavior.
- GaLore is exposed as gradient-projection machinery around a base update.  The public name is
  therefore `optim.galore.sgd`, which says exactly which update rule owns the state.
-/

@[expose] public section

namespace TorchLean

namespace optim

/--
Check the supplied configuration and its representation in the selected runtime scalar.

`Float32` and IEEE32 reject overflow, positive stabilizers rounded to zero, and averaging
coefficients rounded to one. `Float` retains binary64 domains. Custom scalar instances may supply
`Runtime.FromFloat.roundForValidation`; its default retains the input-domain checks.
-/
def Optimizer.validateFor {α : Type} [Runtime.FromFloat α]
    (optimizer : Optimizer) : Except String Unit := do
  optimizer.validate
  let converted := Optimizer.Internal.mapScalars
    (Runtime.FromFloat.roundForValidation (α := α)) optimizer
  match converted.validate with
  | .ok () => pure ()
  | .error message => throw s!"{message} after conversion to the runtime scalar"

/-- Optimizer algorithms accepted by model-training commands. -/
inductive Algorithm where
  | sgd
  | adaGrad
  | rmsProp
  | adam
  | adamW
  | adaDelta
deriving DecidableEq, Repr

namespace Algorithm

/-- Parse an optimizer name used by `--optim`. -/
def parse (value : String) : Except String Algorithm :=
  match value with
  | "sgd" => pure .sgd
  | "adagrad" => pure .adaGrad
  | "rmsprop" => pure .rmsProp
  | "adam" => pure .adam
  | "adamw" => pure .adamW
  | "adadelta" => pure .adaDelta
  | _ =>
      throw s!"bad --optim {value}; expected sgd, adagrad, rmsprop, adam, adamw, or adadelta"

/-- Name written to logs and summaries. -/
def displayName : Algorithm → String
  | .sgd => "SGD"
  | .adaGrad => "AdaGrad"
  | .rmsProp => "RMSProp"
  | .adam => "Adam"
  | .adamW => "AdamW"
  | .adaDelta => "Adadelta"

/-- Configure the selected algorithm with its command-line learning rate and defaults. -/
def configure (algorithm : Algorithm) (learningRate : Float) : Optimizer :=
  match algorithm with
  | .sgd => optim.sgd { learningRate }
  | .adaGrad => optim.adaGrad { learningRate }
  | .rmsProp => optim.rmsProp { learningRate }
  | .adam => optim.adam { learningRate, beta2 := 0.95 }
  | .adamW => optim.adamW { learningRate, weightDecay := 0.1, beta2 := 0.95 }
  | .adaDelta => optim.adaDelta { learningRate }

end Algorithm

namespace muon

export Optim.Muon (Orthogonalizer)

/--
Runtime Muon-style optimizer for module-level training.

Muon is public at the runtime layer because a meaningful Muon run needs an orthogonalization
backend. The default identity backend supports proofs and fallback behavior; production Muon should
pass a matrix-shaped orthogonalizer.
-/
def optimizer {α : Type} [TorchLean.Storage α] [Context α]
    (learningRate momentum : α)
    (orthogonalizer : {s : Shape} → Orthogonalizer α s :=
      fun {s} => Optim.Muon.identityOrthogonalizer (α := α) (s := s))
    {paramShapes : List Shape} :
    Runtime.Autograd.Model.Optim.Optimizer α paramShapes :=
  Runtime.Autograd.Model.Optim.muon
    (α := α) learningRate momentum orthogonalizer (paramShapes := paramShapes)

end muon

namespace galore

export Optim.GaLore (Projector)

/--
Projected-SGD runtime constructor for GaLore-style gradient projection.

This is a projection strategy wrapped around an SGD update.  Full GaLore also needs a policy that
constructs and refreshes low-rank projectors for matrix parameters; this constructor exposes the
verified update boundary once a same-shape projector is supplied.
-/
def sgd {α : Type} [TorchLean.Storage α] [Context α]
    (learningRate : α)
    (projector : {s : Shape} → Projector α s s :=
      fun {s} => Optim.GaLore.identityProjector (α := α) (s := s))
    {paramShapes : List Shape} :
    Runtime.Autograd.Model.Optim.Optimizer α paramShapes :=
  Runtime.Autograd.Model.Optim.projectedSGD
    (α := α) learningRate projector (paramShapes := paramShapes)

end galore

end optim


end TorchLean

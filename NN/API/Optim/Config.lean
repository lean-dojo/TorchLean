/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# Optimizer Configuration

Optimizer algorithms and hyperparameters shared by the public training and runtime APIs.
-/

@[expose] public section

namespace TorchLean
namespace optim

namespace Optimizer.Internal

/--
Closed optimizer representation used only when lowering the public configuration to a runtime.
-/
inductive View where
  | sgd (learningRate : Float) (momentum : Float)
  | adaGrad (learningRate : Float) (epsilon : Float)
  | rmsProp (learningRate : Float) (decay : Float) (epsilon : Float)
  | adam (learningRate : Float) (beta1 : Float) (beta2 : Float) (epsilon : Float)
  | adamW (learningRate : Float) (weightDecay : Float)
      (beta1 : Float) (beta2 : Float) (epsilon : Float)
  | adaDelta (learningRate : Float) (rho : Float) (epsilon : Float)
deriving Repr

end Optimizer.Internal

/--
An optimizer configuration accepted by the trainer and manual-module APIs.

Construct values with `optim.sgd`, `optim.adam`, and the other record-based helpers. The sealed
representation prevents positional runtime constructors from leaking into user code.
-/
structure Optimizer where
  private mk ::
  private representation : Optimizer.Internal.View

/-- Public SGD optimizer configuration. -/
structure SGD.Config where
  /-- Learning rate. -/
  learningRate : Float
  /-- Momentum coefficient. -/
  momentum : Float := 0.0
deriving Repr

/-- Public AdaGrad optimizer configuration. -/
structure AdaGrad.Config where
  /-- Learning rate. -/
  learningRate : Float
  /-- Numerical stabilizer. -/
  epsilon : Float := 1e-10
deriving Repr

/-- Public RMSProp optimizer configuration. -/
structure RMSProp.Config where
  /-- Learning rate. -/
  learningRate : Float
  /-- Decay coefficient for the running average of squared gradients. -/
  decay : Float := 0.99
  /-- Numerical stabilizer. -/
  epsilon : Float := 1e-8
deriving Repr

/-- Public Adam optimizer configuration. -/
structure Adam.Config where
  /-- Learning rate. -/
  learningRate : Float
  /-- First moment coefficient. -/
  beta1 : Float := 0.9
  /-- Second moment coefficient. -/
  beta2 : Float := 0.999
  /-- Numerical stabilizer. -/
  epsilon : Float := 1e-8
deriving Repr

/-- Public AdamW optimizer configuration. -/
structure AdamW.Config where
  /-- Learning rate. -/
  learningRate : Float
  /-- First moment coefficient. -/
  beta1 : Float := 0.9
  /-- Second moment coefficient. -/
  beta2 : Float := 0.999
  /-- Numerical stabilizer. -/
  epsilon : Float := 1e-8
  /-- Decoupled weight decay. -/
  weightDecay : Float := 0.01
deriving Repr

/-- Public Adadelta optimizer configuration. -/
structure AdaDelta.Config where
  /-- Learning rate. -/
  learningRate : Float := 1.0
  /-- Decay coefficient for gradient/update accumulators. -/
  rho : Float := 0.9
  /-- Numerical stabilizer. -/
  epsilon : Float := 1e-6
deriving Repr

namespace Optimizer.Internal

/-- Build the sealed public optimizer value at the API boundary. -/
opaque create (representation : View) : Optimizer :=
  ⟨representation⟩

/-- Reveal an optimizer only at the runtime-lowering boundary. -/
opaque view (optimizer : Optimizer) : View :=
  match optimizer with
  | ⟨representation⟩ => representation

/--
Multiply by ten until the magnitude reaches one, reporting how many tens that took.

`budget` is what makes the recursion obviously terminating. Four hundred steps is past the smallest
subnormal `Float`, so a finite nonzero input never runs out of it.
-/
def tensToUnit (value : Float) (budget : Nat) (digits : Nat := 0) : Float × Nat :=
  match budget with
  | 0 => (value, digits)
  | remaining + 1 =>
      if 1.0 ≤ value.abs then
        (value, digits)
      else
        tensToUnit (value * 10.0) remaining (digits + 1)

/--
Render a hyperparameter so that reading the text back gives the same number.

`toString` on `Float` always prints six digits after the decimal point, so Adam's default `epsilon`
of `1e-8` comes out as `0.000000`. For `describe` that is not cosmetic: the string it returns is
supposed to be the syntax that rebuilds the same optimizer, and an epsilon of zero rebuilds a
different one, one whose adaptive denominator has lost its guard. Values that survive fixed-point
printing are left exactly as `toString` gives them; anything smaller is scaled into exponent form,
which Lean reads back as a float literal. Six significant digits is the resolution either way, so
this is a legible description rather than a bit-exact serialization.
-/
def formatScalar (value : Float) : String :=
  if !value.isFinite || value == 0.0 || 1e-4 ≤ value.abs then
    toString value
  else
    let (mantissa, digits) := tensToUnit value 400
    s!"{mantissa}e-{digits}"

end Optimizer.Internal

/--
SGD optimizer config, optionally with momentum.

Example:
```lean
-- `torch.optim.SGD(params, lr=0.01)`, then the same with heavy-ball momentum.
def plain : optim.Optimizer := optim.sgd { learningRate := 0.01 }

def withMomentum : optim.Optimizer :=
  optim.sgd { learningRate := 0.01, momentum := 0.9 }
```
-/
def sgd (config : SGD.Config) : Optimizer :=
  Optimizer.Internal.create (.sgd config.learningRate config.momentum)

/-- AdaGrad optimizer config, written `optim.adaGrad { learningRate := 0.05 }`. -/
def adaGrad (config : AdaGrad.Config) : Optimizer :=
  Optimizer.Internal.create (.adaGrad config.learningRate config.epsilon)

/-- RMSProp optimizer config, written `optim.rmsProp { learningRate := 1e-3 }`. -/
def rmsProp (config : RMSProp.Config) : Optimizer :=
  Optimizer.Internal.create (.rmsProp config.learningRate config.decay config.epsilon)

/--
Adam optimizer config, written `optim.adam { learningRate := 1e-3 }`.

Example:
```lean
-- `torch.optim.Adam(params, lr=1e-3)`: same default moments, same stabilizer
-- (Kingma and Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015).
def optimizer : optim.Optimizer := optim.adam { learningRate := 1e-3 }
```
-/
def adam (config : Adam.Config) : Optimizer :=
  Optimizer.Internal.create
    (.adam config.learningRate config.beta1 config.beta2 config.epsilon)

/--
AdamW optimizer config, written `optim.adamW { learningRate := 1e-3 }`.

Example:
```lean
-- Decoupled weight decay, so the penalty does not travel through the adaptive moments
-- (Loshchilov and Hutter, "Decoupled Weight Decay Regularization", ICLR 2019).
def optimizer : optim.Optimizer :=
  optim.adamW { learningRate := 1e-3, weightDecay := 0.01 }
```
-/
def adamW (config : AdamW.Config) : Optimizer :=
  Optimizer.Internal.create
    (.adamW config.learningRate config.weightDecay config.beta1 config.beta2 config.epsilon)

/-- AdaDelta optimizer config, written `optim.adaDelta {}`. -/
def adaDelta (config : AdaDelta.Config) : Optimizer :=
  Optimizer.Internal.create (.adaDelta config.learningRate config.rho config.epsilon)

namespace Optimizer

/-- Render an optimizer in the same record-based syntax used to construct it. Every scalar goes
through `Internal.formatScalar`, so a small stabilizer such as Adam's `epsilon` stays readable
instead of printing as `0.000000`. -/
def describe (optimizer : Optimizer) : String :=
  let render := Internal.formatScalar
  match Internal.view optimizer with
  | .sgd learningRate momentum =>
      s!"optim.sgd \{ learningRate := {render learningRate}, momentum := {render momentum} }"
  | .adaGrad learningRate epsilon =>
      s!"optim.adaGrad \{ learningRate := {render learningRate}, epsilon := {render epsilon} }"
  | .rmsProp learningRate decay epsilon =>
      s!"optim.rmsProp \{ learningRate := {render learningRate}, decay := {render decay}, "
        ++ s!"epsilon := {render epsilon} }"
  | .adam learningRate beta1 beta2 epsilon =>
      s!"optim.adam \{ learningRate := {render learningRate}, beta1 := {render beta1}, "
        ++ s!"beta2 := {render beta2}, epsilon := {render epsilon} }"
  | .adamW learningRate weightDecay beta1 beta2 epsilon =>
      s!"optim.adamW \{ learningRate := {render learningRate}, "
        ++ s!"weightDecay := {render weightDecay}, beta1 := {render beta1}, "
        ++ s!"beta2 := {render beta2}, epsilon := {render epsilon} }"
  | .adaDelta learningRate rho epsilon =>
      s!"optim.adaDelta \{ learningRate := {render learningRate}, rho := {render rho}, "
        ++ s!"epsilon := {render epsilon} }"

instance : ToString Optimizer where
  toString := describe

instance : Repr Optimizer where
  reprPrec optimizer _ := Std.Format.text optimizer.describe

/-- Return the base learning rate encoded in an optimizer configuration. -/
def learningRate (optimizer : Optimizer) : Float :=
  match Internal.view optimizer with
  | .sgd learningRate _ => learningRate
  | .adaGrad learningRate _ => learningRate
  | .rmsProp learningRate _ _ => learningRate
  | .adam learningRate _ _ _ => learningRate
  | .adamW learningRate _ _ _ _ => learningRate
  | .adaDelta learningRate _ _ => learningRate

namespace Internal

/--
Reject a hyperparameter that is not a finite number at or above zero.

`isFinite` is the load-bearing half of the test: `0.0 <= value` alone would accept `+∞`, and an
infinite learning rate or weight decay poisons every later update instead of failing where it was
configured.
-/
def requireFiniteNonnegative (name : String) (value : Float) : Except String Unit := do
  unless value.isFinite && 0.0 <= value do
    throw s!"optimizer: {name} must be finite and nonnegative"

/--
Reject a coefficient that is not a finite number in `[0, 1)`.

Exponential-average coefficients such as Adam's `beta1` and `beta2` belong in the half-open
interval: at exactly `1.0` the running average never forgets its initial value, so the optimizer
would ignore the gradient forever rather than merely converge slowly.
-/
def requireUnitInterval (name : String) (value : Float) : Except String Unit := do
  unless value.isFinite && 0.0 <= value && value < 1.0 do
    throw s!"optimizer: {name} must be finite and satisfy 0 <= {name} < 1"

/-- Reject a hyperparameter that is not a finite number strictly above zero. This is the check for
quantities that end up in a denominator, such as Adam's `epsilon`. -/
def requireFinitePositive (name : String) (value : Float) : Except String Unit := do
  unless value.isFinite && 0.0 < value do
    throw s!"optimizer: {name} must be finite and positive"

end Internal

/--
Check the numerical domain of an optimizer configuration before allocating optimizer state.

The checks rule out undefined bias corrections and non-finite updates. They are shared by the
trainer, manual-module, and reinforcement-learning entry points.
-/
def validate (optimizer : Optimizer) : Except String Unit :=
  match Internal.view optimizer with
  | .sgd learningRate momentum => do
      Internal.requireFiniteNonnegative "learning rate" learningRate
      Internal.requireUnitInterval "momentum" momentum
  | .adaGrad learningRate epsilon => do
      Internal.requireFiniteNonnegative "learning rate" learningRate
      Internal.requireFinitePositive "epsilon" epsilon
  | .rmsProp learningRate decay epsilon => do
      Internal.requireFiniteNonnegative "learning rate" learningRate
      Internal.requireUnitInterval "decay" decay
      Internal.requireFinitePositive "epsilon" epsilon
  | .adam learningRate beta1 beta2 epsilon => do
      Internal.requireFiniteNonnegative "learning rate" learningRate
      Internal.requireUnitInterval "beta1" beta1
      Internal.requireUnitInterval "beta2" beta2
      Internal.requireFinitePositive "epsilon" epsilon
  | .adamW learningRate weightDecay beta1 beta2 epsilon => do
      Internal.requireFiniteNonnegative "learning rate" learningRate
      Internal.requireFiniteNonnegative "weight decay" weightDecay
      Internal.requireUnitInterval "beta1" beta1
      Internal.requireUnitInterval "beta2" beta2
      Internal.requireFinitePositive "epsilon" epsilon
  | .adaDelta learningRate rho epsilon => do
      Internal.requireFiniteNonnegative "learning rate" learningRate
      Internal.requireUnitInterval "rho" rho
      Internal.requireFinitePositive "epsilon" epsilon

/-- Transform every scalar in a configuration for validation after conversion. -/
def Internal.mapScalars (cast : Float → Float) (optimizer : Optimizer) : Optimizer :=
  Internal.create <| match Internal.view optimizer with
  | .sgd learningRate momentum => .sgd (cast learningRate) (cast momentum)
  | .adaGrad learningRate epsilon => .adaGrad (cast learningRate) (cast epsilon)
  | .rmsProp learningRate decay epsilon =>
      .rmsProp (cast learningRate) (cast decay) (cast epsilon)
  | .adam learningRate beta1 beta2 epsilon =>
      .adam (cast learningRate) (cast beta1) (cast beta2) (cast epsilon)
  | .adamW learningRate weightDecay beta1 beta2 epsilon =>
      .adamW (cast learningRate) (cast weightDecay) (cast beta1) (cast beta2) (cast epsilon)
  | .adaDelta learningRate rho epsilon =>
      .adaDelta (cast learningRate) (cast rho) (cast epsilon)

/--
Check both the supplied configuration and its binary32 representation.

Training must reject coefficients that round to one, stabilizers that round to zero, and finite
binary64 rates that overflow binary32. `validate` remains available for binary64 callers.
-/
def validateFloat32 (optimizer : Optimizer) : Except String Unit := do
  optimizer.validate
  match (Internal.mapScalars (fun value => value.toFloat32.toFloat) optimizer).validate with
  | .ok () => pure ()
  | .error message => throw s!"{message} after conversion to binary32"

end Optimizer
end optim
end TorchLean

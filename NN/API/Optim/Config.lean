/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Floats.Formats.BinaryInterchange.Conversion.Text.DecimalFormatting
public import FloatLib.Floats.Formats.BinaryInterchange.Format.Catalog

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
Render a hyperparameter as a Lean expression, preserving every finite binary64 bit.

FloatLib rounds the exact value to seventeen significant decimal digits. The candidate is decoded
with Lean's scientific-literal decoder and `OfScientific Float`; it is used only if the resulting
bits match. Otherwise an explicit `Float.ofBits` expression retains the original finite value.
Negative zero and nonfinite values use `Float.ofBits` directly. NaNs follow Lean's canonicalization.
-/
def formatScalar (value : Float) : String :=
  let bits := value.toBits
  let exact := s!"(Float.ofBits {bits.toNat})"
  if !value.isFinite || bits == 0x8000000000000000 then
    exact
  else
    let model :=
      FloatLib.Floats.Formats.BinaryInterchange.Model.ofNatBits (fmt := .binary64) bits.toNat
    let text := model.formatScientific 16
    let negative := text.startsWith "-"
    let magnitude := if negative then (text.drop 1).toString else text
    match Lean.Syntax.decodeScientificLitVal? magnitude with
    | some (mantissa, sign, exponent) =>
        let parsed : Float := OfScientific.ofScientific mantissa sign exponent
        let parsed := if negative then -parsed else parsed
        if parsed.toBits == bits then text else exact
    | none => exact

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

/-- Render an optimizer in the record-based syntax used to construct it. Finite fields retain
their binary64 bits through `Internal.formatScalar`, including coefficients close to one and
subnormal stabilizers. Negative zero and nonfinite fields use explicit `Float.ofBits` expressions;
NaNs follow Lean's canonicalization. -/
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

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
  | sgd (lr : Float) (momentum : Float)
  | adaGrad (lr : Float) (epsilon : Float)
  | rmsProp (lr : Float) (decay : Float) (epsilon : Float)
  | adam (lr : Float) (beta1 : Float) (beta2 : Float) (epsilon : Float)
  | adamW (lr : Float) (weightDecay : Float)
      (beta1 : Float) (beta2 : Float) (epsilon : Float)
  | adaDelta (lr : Float) (rho : Float) (epsilon : Float)
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
  lr : Float
  /-- Momentum coefficient. -/
  momentum : Float := 0.0
deriving Repr

/-- Public AdaGrad optimizer configuration. -/
structure AdaGrad.Config where
  /-- Learning rate. -/
  lr : Float
  /-- Numerical stabilizer. -/
  epsilon : Float := 1e-10
deriving Repr

/-- Public RMSProp optimizer configuration. -/
structure RMSProp.Config where
  /-- Learning rate. -/
  lr : Float
  /-- Decay coefficient for the running average of squared gradients. -/
  decay : Float := 0.99
  /-- Numerical stabilizer. -/
  epsilon : Float := 1e-8
deriving Repr

/-- Public Adam optimizer configuration. -/
structure Adam.Config where
  /-- Learning rate. -/
  lr : Float
  /-- First moment coefficient. -/
  beta1 : Float := 0.9
  /-- Second moment coefficient. -/
  beta2 : Float := 0.999
  /-- Numerical stabilizer. -/
  epsilon : Float := 1e-8
deriving Repr

/-- Public AdamW optimizer configuration. -/
structure AdamW.Config extends Adam.Config where
  /-- Decoupled weight decay. -/
  weightDecay : Float := 0.01
deriving Repr

/-- Public Adadelta optimizer configuration. -/
structure AdaDelta.Config where
  /-- Learning rate. -/
  lr : Float := 1.0
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
def plain : optim.Optimizer := optim.sgd { lr := 0.01 }

def withMomentum : optim.Optimizer :=
  optim.sgd { lr := 0.01, momentum := 0.9 }
```
-/
def sgd (config : SGD.Config) : Optimizer :=
  Optimizer.Internal.create (.sgd config.lr config.momentum)

/-- AdaGrad optimizer config, written `optim.adaGrad { lr := 0.05 }`. -/
def adaGrad (config : AdaGrad.Config) : Optimizer :=
  Optimizer.Internal.create (.adaGrad config.lr config.epsilon)

/-- RMSProp optimizer config, written `optim.rmsProp { lr := 1e-3 }`. -/
def rmsProp (config : RMSProp.Config) : Optimizer :=
  Optimizer.Internal.create (.rmsProp config.lr config.decay config.epsilon)

/--
Adam optimizer config, written `optim.adam { lr := 1e-3 }`.

Example:
```lean
-- `torch.optim.Adam(params, lr=1e-3)`: same default moments, same stabilizer
-- (Kingma and Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015).
def optimizer : optim.Optimizer := optim.adam { lr := 1e-3 }
```
-/
def adam (config : Adam.Config) : Optimizer :=
  Optimizer.Internal.create
    (.adam config.lr config.beta1 config.beta2 config.epsilon)

/--
AdamW optimizer config, written `optim.adamW { lr := 1e-3 }`.

Example:
```lean
-- Decoupled weight decay, so the penalty does not travel through the adaptive moments
-- (Loshchilov and Hutter, "Decoupled Weight Decay Regularization", ICLR 2019).
def optimizer : optim.Optimizer :=
  optim.adamW { lr := 1e-3, weightDecay := 0.01 }
```
-/
def adamW (config : AdamW.Config) : Optimizer :=
  Optimizer.Internal.create
    (.adamW config.lr config.weightDecay config.beta1 config.beta2 config.epsilon)

/-- AdaDelta optimizer config, written `optim.adaDelta {}`. -/
def adaDelta (config : AdaDelta.Config) : Optimizer :=
  Optimizer.Internal.create (.adaDelta config.lr config.rho config.epsilon)

namespace Optimizer

/-- Render an optimizer in the record-based syntax used to construct it. Finite fields retain
their binary64 bits through `Internal.formatScalar`, including coefficients close to one and
subnormal stabilizers. Negative zero and nonfinite fields use explicit `Float.ofBits` expressions;
NaNs follow Lean's canonicalization. -/
def describe (optimizer : Optimizer) : String :=
  let render := Internal.formatScalar
  match Internal.view optimizer with
  | .sgd lr momentum =>
      s!"optim.sgd \{ lr := {render lr}, momentum := {render momentum} }"
  | .adaGrad lr epsilon =>
      s!"optim.adaGrad \{ lr := {render lr}, epsilon := {render epsilon} }"
  | .rmsProp lr decay epsilon =>
      s!"optim.rmsProp \{ lr := {render lr}, decay := {render decay}, "
        ++ s!"epsilon := {render epsilon} }"
  | .adam lr beta1 beta2 epsilon =>
      s!"optim.adam \{ lr := {render lr}, beta1 := {render beta1}, "
        ++ s!"beta2 := {render beta2}, epsilon := {render epsilon} }"
  | .adamW lr weightDecay beta1 beta2 epsilon =>
      s!"optim.adamW \{ lr := {render lr}, "
        ++ s!"weightDecay := {render weightDecay}, beta1 := {render beta1}, "
        ++ s!"beta2 := {render beta2}, epsilon := {render epsilon} }"
  | .adaDelta lr rho epsilon =>
      s!"optim.adaDelta \{ lr := {render lr}, rho := {render rho}, "
        ++ s!"epsilon := {render epsilon} }"

instance : ToString Optimizer where
  toString := describe

instance : Repr Optimizer where
  reprPrec optimizer _ := Std.Format.text optimizer.describe

/-- Return the base learning rate encoded in an optimizer configuration. -/
def lr (optimizer : Optimizer) : Float :=
  match Internal.view optimizer with
  | .sgd lr _ => lr
  | .adaGrad lr _ => lr
  | .rmsProp lr _ _ => lr
  | .adam lr _ _ _ => lr
  | .adamW lr _ _ _ _ => lr
  | .adaDelta lr _ _ => lr

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

The checks require finite hyperparameters, nonnegative rates and weight decay, positive
stabilizers, and averaging coefficients below one. They are shared by the trainer, manual-module,
and reinforcement-learning entry points. They do not bound parameters or gradients, so valid
hyperparameters alone do not guarantee finite updates.
-/
def validate (optimizer : Optimizer) : Except String Unit :=
  match Internal.view optimizer with
  | .sgd lr momentum => do
      Internal.requireFiniteNonnegative "learning rate" lr
      Internal.requireUnitInterval "momentum" momentum
  | .adaGrad lr epsilon => do
      Internal.requireFiniteNonnegative "learning rate" lr
      Internal.requireFinitePositive "epsilon" epsilon
  | .rmsProp lr decay epsilon => do
      Internal.requireFiniteNonnegative "learning rate" lr
      Internal.requireUnitInterval "decay" decay
      Internal.requireFinitePositive "epsilon" epsilon
  | .adam lr beta1 beta2 epsilon => do
      Internal.requireFiniteNonnegative "learning rate" lr
      Internal.requireUnitInterval "beta1" beta1
      Internal.requireUnitInterval "beta2" beta2
      Internal.requireFinitePositive "epsilon" epsilon
  | .adamW lr weightDecay beta1 beta2 epsilon => do
      Internal.requireFiniteNonnegative "learning rate" lr
      Internal.requireFiniteNonnegative "weight decay" weightDecay
      Internal.requireUnitInterval "beta1" beta1
      Internal.requireUnitInterval "beta2" beta2
      Internal.requireFinitePositive "epsilon" epsilon
  | .adaDelta lr rho epsilon => do
      Internal.requireFiniteNonnegative "learning rate" lr
      Internal.requireUnitInterval "rho" rho
      Internal.requireFinitePositive "epsilon" epsilon

/-- Transform every scalar in a configuration for validation after conversion. -/
def Internal.mapScalars (cast : Float → Float) (optimizer : Optimizer) : Optimizer :=
  Internal.create <| match Internal.view optimizer with
  | .sgd lr momentum => .sgd (cast lr) (cast momentum)
  | .adaGrad lr epsilon => .adaGrad (cast lr) (cast epsilon)
  | .rmsProp lr decay epsilon =>
      .rmsProp (cast lr) (cast decay) (cast epsilon)
  | .adam lr beta1 beta2 epsilon =>
      .adam (cast lr) (cast beta1) (cast beta2) (cast epsilon)
  | .adamW lr weightDecay beta1 beta2 epsilon =>
      .adamW (cast lr) (cast weightDecay) (cast beta1) (cast beta2) (cast epsilon)
  | .adaDelta lr rho epsilon =>
      .adaDelta (cast lr) (cast rho) (cast epsilon)

end Optimizer
end optim
end TorchLean

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Floats.Formats.BinaryInterchange.Conversion.Text.Formatting
public import FloatLib.Floats.Formats.BinaryInterchange.Format.Catalog
public import TorchLean.Tensor.Elementwise

/-!
# Optimizer Configuration

Optimizer algorithms and hyperparameters shared by the public training and runtime APIs.

We copied the defaults by hand from the PyTorch 2.8 signatures of the six `torch.optim` classes:
- SGD `momentum=0`,
- Adagrad `eps=1e-10`,
- RMSprop `alpha=0.99`, our `decay`, and `eps=1e-08`,
- Adam and AdamW `betas=(0.9, 0.999)` and `eps=1e-08`, with AdamW's `weight_decay=1e-2`,
- Adadelta `lr=1.0`, `rho=0.9` and `eps=1e-06`.

Adadelta is the one config with a default `lr`. Torch's other defaults, `1e-3` for SGD, Adam and
AdamW and `1e-2` for Adagrad and RMSprop, are left to the caller, who has to write one. Our
RMSProp `decay` is torch's 0.99, where Hinton's lecture 6e slide uses 0.9.
-/

@[expose] public section

namespace TorchLean
namespace optim

namespace Optimizer

/--
Closed optimizer representation behind `Optimizer`. `describe`, `lr`, `validate`,
`Optimizer.mapScalars` and `Module.withConfiguredOptimizer` match on it.
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

end Optimizer

/--
An optimizer configuration accepted by the trainer and manual-module APIs.

Construct values with `optim.sgd`, `optim.adam`, and the other record-based helpers. The
constructor and the field are private, and `Optimizer.create` and `Optimizer.toView` convert to and
from the positional `Optimizer.View`.
-/
structure Optimizer where
  private mk ::
  private representation : Optimizer.View

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

namespace Optimizer

/-- Build the sealed public optimizer value at the API boundary. -/
@[no_expose] def create (representation : View) : Optimizer :=
  ⟨representation⟩

/-- Reveal an optimizer only at the runtime-lowering boundary. -/
@[no_expose] def toView (optimizer : Optimizer) : View :=
  match optimizer with
  | ⟨representation⟩ => representation

/-- Revealing a freshly sealed optimizer returns its representation. -/
@[simp] theorem toView_create (representation : View) : toView (create representation) =
    representation := by
  rfl

/-- Sealing the representation of an optimizer reconstructs that optimizer. -/
@[simp] theorem create_toView (optimizer : Optimizer) : create (toView optimizer) = optimizer := by
  cases optimizer
  rfl

/--
Render a hyperparameter as a Lean expression that preserves every finite binary64 bit.

FloatLib rounds the exact value to seventeen significant decimal digits, and the text is kept when
Lean's scientific-literal decoder and `OfScientific Float` read it back to the same bits. Otherwise
a `Float.ofBits` expression carries the original finite value. Negative zero and nonfinite values
always use `Float.ofBits`, and NaNs follow Lean's canonicalization.
-/
def formatScalar (value : Float) : String :=
  let bits := value.toBits
  let exact := s!"(Float.ofBits {bits.toNat})"
  if !value.isFinite || bits == 0x8000000000000000 then
    exact
  else
    let model :=
      FloatLib.Floats.Formats.BinaryInterchange.Model.ofNatBits (fmt := .binary64) bits.toNat
    let text := model.format (style := .scientific 16)
    let negative := text.startsWith "-"
    let magnitude := if negative then (text.drop 1).toString else text
    match Lean.Syntax.decodeScientificLitVal? magnitude with
    | some (mantissa, sign, exponent) =>
        let parsed : Float := OfScientific.ofScientific mantissa sign exponent
        let parsed := if negative then -parsed else parsed
        if parsed.toBits == bits then text else exact
    | none => exact

end Optimizer

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
  Optimizer.create (.sgd config.lr config.momentum)

/-- AdaGrad optimizer config, written `optim.adaGrad { lr := 0.05 }`. -/
def adaGrad (config : AdaGrad.Config) : Optimizer :=
  Optimizer.create (.adaGrad config.lr config.epsilon)

/-- RMSProp optimizer config, written `optim.rmsProp { lr := 1e-3 }`. -/
def rmsProp (config : RMSProp.Config) : Optimizer :=
  Optimizer.create (.rmsProp config.lr config.decay config.epsilon)

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
  Optimizer.create
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
  Optimizer.create
    (.adamW config.lr config.weightDecay config.beta1 config.beta2 config.epsilon)

/-- AdaDelta optimizer config, written `optim.adaDelta {}`. -/
def adaDelta (config : AdaDelta.Config) : Optimizer :=
  Optimizer.create (.adaDelta config.lr config.rho config.epsilon)

namespace Optimizer

/-- Render an optimizer in the record-based syntax that constructs it. Every field goes through
`formatScalar`, so finite fields keep their binary64 bits, coefficients close to one and
subnormal stabilizers included. Negative zero and nonfinite fields become `Float.ofBits`
expressions, and NaNs follow Lean's canonicalization. -/
def describe (optimizer : Optimizer) : String :=
  let render := formatScalar
  match toView optimizer with
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
  match toView optimizer with
  | .sgd lr _ => lr
  | .adaGrad lr _ => lr
  | .rmsProp lr _ _ => lr
  | .adam lr _ _ _ => lr
  | .adamW lr _ _ _ _ => lr
  | .adaDelta lr _ _ => lr

/--
Reject a coefficient that is not a finite number in `[0, 1)`.

At `1.0` the coefficient stops decaying, with a different effect in each optimizer. The momentum
buffer sums every past gradient, and RMSProp's squared average stays at its initial zero, so its
step is `lr * g / epsilon`. Adadelta's two averages stay at zero and its step reduces to `lr * g`.
Adam's and AdamW's moments stay at zero while the bias correction `1 - beta^t` is zero, so the
corrected moments divide `0` by `0`.
-/
def requireUnitInterval (name : String) (value : Float) : Except String Unit := do
  unless value.isFinite && 0.0 <= value && value < 1.0 do
    throw s!"optimizer: {name} must be finite and satisfy 0 <= {name} < 1"

/--
Check the numerical domain of an optimizer configuration before any optimizer state is allocated.

The checks require finite hyperparameters, nonnegative rates and weight decay, positive
stabilizers, and averaging coefficients in `[0, 1)`. They are shared by the trainer, manual-module,
and reinforcement-learning entry points. They do not bound parameters or gradients, so valid
hyperparameters alone do not guarantee finite updates.

Torch 2.8 checks less: its constructors accept SGD `momentum=1.5`, RMSprop `alpha=1.0`, Adadelta
`rho=1.0`, an infinite `lr` and `eps=0`, and all of these fail here. We want positive stabilizers
because at `epsilon = 0` a zero gradient entry gives `0 / 0` on the first step of AdaGrad, RMSProp,
Adam, AdamW and Adadelta.
-/
def validate (optimizer : Optimizer) : Except String Unit :=
  match toView optimizer with
  | .sgd lr momentum => do
      Context.requireWithin "optimizer: learning rate" (some 0) none lr
      requireUnitInterval "momentum" momentum
  | .adaGrad lr epsilon => do
      Context.requireWithin "optimizer: learning rate" (some 0) none lr
      Context.requireWithin "optimizer: epsilon" (some 0) none epsilon (closed := false)
  | .rmsProp lr decay epsilon => do
      Context.requireWithin "optimizer: learning rate" (some 0) none lr
      requireUnitInterval "decay" decay
      Context.requireWithin "optimizer: epsilon" (some 0) none epsilon (closed := false)
  | .adam lr beta1 beta2 epsilon => do
      Context.requireWithin "optimizer: learning rate" (some 0) none lr
      requireUnitInterval "beta1" beta1
      requireUnitInterval "beta2" beta2
      Context.requireWithin "optimizer: epsilon" (some 0) none epsilon (closed := false)
  | .adamW lr weightDecay beta1 beta2 epsilon => do
      Context.requireWithin "optimizer: learning rate" (some 0) none lr
      Context.requireWithin "optimizer: weight decay" (some 0) none weightDecay
      requireUnitInterval "beta1" beta1
      requireUnitInterval "beta2" beta2
      Context.requireWithin "optimizer: epsilon" (some 0) none epsilon (closed := false)
  | .adaDelta lr rho epsilon => do
      Context.requireWithin "optimizer: learning rate" (some 0) none lr
      requireUnitInterval "rho" rho
      Context.requireWithin "optimizer: epsilon" (some 0) none epsilon (closed := false)

/-- Transform every scalar in a configuration for validation after conversion. -/
def mapScalars (cast : Float → Float) (optimizer : Optimizer) : Optimizer :=
  Optimizer.create <| match Optimizer.toView optimizer with
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

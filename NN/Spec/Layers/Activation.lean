/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape

/-!
# Activation functions (spec layer)

This module is TorchLean's "activation toolbox": **pure** mathematical definitions of common
nonlinearities and their derivatives.

Design intent:

- Scalar definitions live under `Activation.Math` (functions `α → α`).
- Tensor-level definitions are almost always the scalar function mapped pointwise via `map_spec`.
- Where the math is inherently *non-pointwise* (notably `softmax`), we provide a shape-aware
  implementation plus an explicit backward/VJP.

PyTorch mental model:

- Scalar `Activation.Math.*` corresponds to the formulas behind `torch.nn.functional.*`.
- Tensor-level `Activation.*_spec` corresponds to applying that nonlinearity elementwise.
- `softmaxSpec` here is the real last-axis softmax on tensors (like `torch.softmax(x, dim=-1)`),
  implemented recursively over outer dimensions.

Notes on scalar polymorphism:

TorchLean tries hard not to bake "Float everywhere" into the spec. All definitions are written
against a `Context α` plus the exact algebra/analysis typeclasses they need. That is what lets
the same layer definitions instantiate over:

- `Float` for fast runtime execution,
- exact/reasoning scalars for proofs,
- interval-like scalars for verification.

References / analogies (stable entry points):

- PyTorch activations: https://pytorch.org/docs/stable/nn.functional.html
- PyTorch `torch.softmax`: https://pytorch.org/docs/stable/generated/torch.softmax.html
- ReLU: Vinod Nair and Geoffrey Hinton,
  "Rectified Linear Units Improve Restricted Boltzmann Machines" (ICML 2010)
- ELU: Djork-Arne Clevert et al.,
  "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)" (ICLR 2016)
- GELU: Dan Hendrycks and Kevin Gimpel, "Gaussian Error Linear Units (GELUs)" (arXiv:1606.08415)
- Swish / SiLU: Prajit Ramachandran et al., "Searching for Activation Functions" (arXiv:1710.05941)
-/

@[expose] public section


open Spec
open Tensor

namespace Activation
namespace Math

variable {α : Type} [Context α]

/-! ## Scalar activations -/

/-- ReLU: `relu(x) = max(x, 0)`.

PyTorch analogy: `torch.nn.functional.relu`.

This is the simplest nonlinearity we use throughout TorchLean because it stays meaningful across
many scalar backends (including ones that do not support `exp/log`).
-/
def reluSpec {α : Type} [Zero α] [Max α] (x : α) : α :=
  Max.max x 0

/-- A standard subgradient choice for ReLU:

`d/dx relu(x) = 1` if `x > 0`, and `0` otherwise.

PyTorch analogy: autograd picks a subgradient at `x = 0`; our spec commits to a concrete one to
make "the derivative" a pure function.

The `DecidableRel (· > ·)` constraint reflects that this definition literally branches on `x > 0`.
-/
def reluDerivSpec {α : Type} [Zero α] [One α] [LT α] [DecidableRel ((· > ·) : α → α → Prop)] (x :
  α) : α :=
  if x > 0 then 1 else 0

/-- Logistic sigmoid:

`sigmoid(x) = 1 / (1 + exp(-x))`.

PyTorch analogy: `torch.nn.functional.sigmoid` (or `torch.sigmoid`).
-/
def sigmoidSpec (x : α) : α :=
  1 / (1 + MathFunctions.exp (-x))

/-- Derivative of sigmoid:

`sigmoid'(x) = σ(x) * (1 - σ(x))`.

We write it this way (in terms of `σ(x)`) because that is the form used in most AD systems and it
avoids re-expanding the exponential expression.
-/
def sigmoidDerivSpec (x : α) : α :=
  let s := sigmoidSpec x
  s * (1 - s)

/-- Hyperbolic tangent: `tanh(x)`.  PyTorch analogy: `torch.tanh`. -/
def tanhSpec (x : α) : α :=
  MathFunctions.tanh x

/-- Derivative of tanh:

`tanh'(x) = 1 - tanh(x)^2`.
-/
def tanhDerivSpec (x : α) : α :=
  1 - ((MathFunctions.tanh x) * (MathFunctions.tanh x))

/-- Leaky ReLU:

`leaky_relu(x; α) = x` if `x > 0`, else `α * x`.

PyTorch analogy: `torch.nn.functional.leaky_relu` with `negative_slope = α`.
-/
def leakyReluSpec {α : Type} [Zero α] [Mul α] [LT α] [DecidableRel ((· > ·) : α → α → Prop)] (x :
  α) (αₗ : α) : α :=
  if x > 0 then x else αₗ * x

/-- Derivative of leaky ReLU:

`d/dx leaky_relu(x; α) = 1` if `x > 0`, else `α`.
-/
def leakyReluDerivSpec {α : Type} [Zero α] [One α] [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
  (x : α) (αₗ : α) : α :=
  if x > 0 then 1 else αₗ

/-- Sinh: `sinh(x)`. -/
def sinhSpec (x : α) : α :=
  MathFunctions.sinh x

/-- Sinh derivative: `cosh(x)`. -/
def sinhDerivSpec (x : α) : α :=
  MathFunctions.cosh x

/-- Cosh: `cosh(x)`. -/
def coshSpec (x : α) : α :=
  MathFunctions.cosh x

/-- Cosh derivative: `sinh(x)`. -/
def coshDerivSpec (x : α) : α :=
  MathFunctions.sinh x

/-- Logistic form written as `exp(x)/(exp(x)+1)`.

This is mathematically the same sigmoid function as `sigmoidSpec`; we keep it as `logisticSpec`
because several scalar approximation proofs reason about this `exp(x)` numerator form directly.

Important naming choice: this is **not** called scalar softmax. A one-entry softmax is always `1`;
the real softmax API in TorchLean is the tensor-level `Activation.softmaxSpec` below.
-/
def logisticSpec (x : α) : α :=
  MathFunctions.exp x / (MathFunctions.exp x + 1)

/-- Derivative of `logisticSpec`, expressed in output form. -/
def logisticDerivSpec (x : α) : α :=
  logisticSpec x * (1 - logisticSpec x)

/-- ELU (Exponential Linear Unit):

`elu(x; α) = x` if `x > 0`, else `α * (exp(x) - 1)`.

PyTorch analogy: `torch.nn.functional.elu` with `alpha = α`.
-/
def eluSpec {α : Type} [Zero α] [One α] [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
  [MathFunctions α] [Sub α] [Mul α] (x : α) (alpha : α) : α :=
  if x > 0 then x else alpha * (MathFunctions.exp x - 1)

/-- Derivative of ELU:

`elu'(x; α) = 1` if `x > 0`, else `α * exp(x)`.
-/
def eluDerivSpec {α : Type} [Zero α] [One α] [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
  [MathFunctions α] [Mul α] (x : α) (alpha : α) : α :=
  if x > 0 then 1 else alpha * MathFunctions.exp x

/-- GELU (approximate): the common tanh-based approximation used in many Transformer codebases.

PyTorch analogy: `torch.nn.functional.gelu(x, approximate="tanh")`.
-/
def geluSpec {α : Type} [MathFunctions α] [OfScientific α] [Add α] [Mul α] [Div α] [Sub α] [OfNat α
  1] (x : α) : α :=
  let two : α := (1 : α) + (1 : α)
  let pi : α := MathFunctions.pi
  let sqrt_two_over_pi := MathFunctions.sqrt (two / pi)
  let coeff : α := 0.044715
  x * ((1 : α) + MathFunctions.tanh (sqrt_two_over_pi * (x + coeff * x * x * x))) / two

/-- GELU derivative for the tanh-based approximation. -/
def geluDerivSpec {α : Type} [MathFunctions α] [OfScientific α] [Add α] [Mul α] [Div α] [Sub α]
  [OfNat α 1] (x : α) : α :=
  let two : α := (1 : α) + (1 : α)
  let three : α := (1 : α) + (1 : α) + (1 : α)
  let pi : α := MathFunctions.pi
  let sqrt_two_over_pi := MathFunctions.sqrt (two / pi)
  let coeff : α := 0.044715
  let tanh_term := MathFunctions.tanh (sqrt_two_over_pi * (x + coeff * x * x * x))
  let sech_term := (1 : α) - tanh_term * tanh_term
  let inner_deriv := sqrt_two_over_pi * ((1 : α) + three * coeff * x * x)
  ((1 : α) + tanh_term + x * sech_term * inner_deriv) / two

/-- Swish / SiLU:

`swish(x) = x * sigmoid(x)`.

PyTorch analogy: `torch.nn.functional.silu`.
-/
def swishSpec (x : α) : α :=
  x * sigmoidSpec x

/-- Derivative of Swish / SiLU.

Written in terms of `sigmoid(x)` for the same reason as `sigmoidDerivSpec`: this is the form
used by AD systems and is convenient to reuse in proofs.
-/
def swishDerivSpec (x : α) : α :=
  let s := sigmoidSpec x
  s + x * s * (1 - s)

/-- Softplus:

`softplus(x) = log(1 + exp(x))`.

PyTorch analogy: `torch.nn.functional.softplus`.
-/
def softplusSpec (x : α) : α :=
  MathFunctions.log (1 + MathFunctions.exp x)

/-- Derivative of softplus:

`softplus'(x) = sigmoid(x)`.
-/
def softplusDerivSpec (x : α) : α :=
  sigmoidSpec x

/-- A smooth log surrogate:

`safe_log(x; ε) = log(softplus(x) + ε)`.

We use this when we want something "log-like" without having to carry side conditions about the
input being strictly positive.
-/
def safeLogSpec (x : α) (ε : α := Numbers.epsilon) : α :=
  MathFunctions.log (softplusSpec x + ε)

/-- Derivative of `safe_log_spec`. -/
def safeLogDerivSpec (x : α) (ε : α := Numbers.epsilon) : α :=
  softplusDerivSpec x / (softplusSpec x + ε)

/-- A smooth absolute value surrogate:

`smooth_abs(x; ε) = sqrt(x^2 + ε)`.

Useful when you want an `abs`-like shape but keep differentiability at `0`.
-/
def smoothAbsSpec (x : α) (ε : α := Numbers.epsilon) : α :=
  MathFunctions.sqrt (x * x + ε)

/-- Derivative of `smooth_abs_spec`. -/
def smoothAbsDerivSpec (x : α) (ε : α := Numbers.epsilon) : α :=
  x / smoothAbsSpec x ε

end Math

variable {α : Type} [Context α]

/-- Tensor-level tanh (pointwise).

PyTorch analogy: `torch.tanh(t)` or `torch.nn.functional.tanh(t)` applied elementwise.
-/
def tanhSpec {s : Shape} : Tensor α s → Tensor α s :=
  mapSpec Activation.Math.tanhSpec

/-- Tensor-level ReLU (pointwise). -/
def reluSpec {α : Type} [Zero α] [Max α] {s : Shape} (t : Tensor α s) : Tensor α s :=
  mapSpec Activation.Math.reluSpec t

/-- Tensor-level sigmoid (pointwise). -/
def sigmoidSpec {s : Shape} (t : Tensor α s) : Tensor α s :=
  mapSpec Activation.Math.sigmoidSpec t

/-- Tensor-level ReLU derivative (pointwise), using the scalar subgradient choice in
`Activation.Math.relu_deriv_spec`. -/
def reluDerivSpec {α : Type} [Zero α] [One α] [LT α] [DecidableRel ((· > ·) : α → α → Prop)] {s :
  Shape} (t : Tensor α s) : Tensor α s :=
  mapSpec Activation.Math.reluDerivSpec t

/-- Tensor-level sigmoid derivative (pointwise). -/
def sigmoidDerivSpec {s : Shape} (t : Tensor α s) : Tensor α s :=
  mapSpec Activation.Math.sigmoidDerivSpec t

/--
Derivative of sigmoid when the sigmoid output has already been computed.

Recurrent layers save gate activations during the forward pass, so their backward specs should use
this shared helper instead of re-defining `s * (1 - s)` locally.
-/
def sigmoidOutputDerivSpec {s : Shape} (sigmoidOutput : Tensor α s) : Tensor α s :=
  mulSpec sigmoidOutput (subSpec (fill 1 s) sigmoidOutput)

/-- Tensor-level tanh derivative (pointwise). -/
def tanhDerivSpec {s : Shape} (t : Tensor α s) : Tensor α s :=
  mapSpec Activation.Math.tanhDerivSpec t

/-!
## Proper (last‑axis) softmax on tensors

These are the shape‑aware softmax definitions used in attention / classification layers.
They recurse over outer dimensions and apply a numerically‑stable softmax to the last axis.
-/

/-- Softmax on a length-`n` vector.

This is the "real" softmax, not the scalar logistic helper in `Activation.Math.logisticSpec`.

Numerical stability:

We implement the standard stabilized form `softmax(x) = exp(x - m) / Σ exp(x - m)` where
`m = max_i x_i`. Subtracting the max avoids overflow in typical floating-point backends, and it is
also a nice canonical form to reference in proofs.
-/
def softmaxVecSpec {n : Nat} (t : Tensor α (.dim n .scalar)) : Tensor α (.dim n .scalar) :=
  match n with
  | 0 => t
  | Nat.succ n' =>
      -- We pick the max by seeding the fold with the first element, so we do not need a sentinel
      -- value (and the definition stays meaningful beyond plain floating-point).
      let maxT : Tensor α .scalar :=
        match t with
        | Tensor.dim f =>
            let first : α := match f ⟨0, Nat.succ_pos n'⟩ with | Tensor.scalar v => v
            let maxVal : α :=
              (List.finRange (Nat.succ n')).foldl
                (fun acc i =>
                  match f i with
                  | Tensor.scalar v => max acc v)
                first
            Tensor.scalar maxVal
      let shifted : Tensor α (.dim (Nat.succ n') .scalar) := subSpec t (replicate maxT)
      let ex := expSpec shifted
      let denom : α :=
        match ex with
        | Tensor.dim g =>
            (List.finRange (Nat.succ n')).foldl
              (fun acc i => acc + match g i with | Tensor.scalar v => v)
              0
      divSpec ex (replicate (Tensor.scalar denom))

/-- Softmax along the last axis (recurses over outer dimensions).

PyTorch analogy: `torch.softmax(x, dim=-1)`.

For `s = .scalar` we return `1` (there is only one coordinate). For higher-rank tensors we keep
the outer structure and apply `softmax_vec_spec` at the last axis.
-/
def softmaxSpec : {s : Shape} → Tensor α s → Tensor α s
  | .scalar, _ => Tensor.scalar 1
  | .dim n .scalar, t => softmaxVecSpec (α := α) (n := n) t
  | .dim n inner, Tensor.dim f =>
      Tensor.dim (fun i : Fin n => softmaxSpec (s := inner) (f i))

/-- Backward/VJP for last-axis softmax.

If `y = softmax(x)` and we are given an upstream gradient `dL/dy`, then for each last-axis slice:

`dL/dx = y ⊙ (dL/dy - ⟨dL/dy, y⟩)`

This is the standard Jacobian-vector product for softmax, written in a way that avoids materializing
the full `n×n` Jacobian.
-/
def softmaxBackwardSpec : {s : Shape} → Tensor α s → Tensor α s → Tensor α s
  | .scalar, _x, _dY => Tensor.scalar 0
  | .dim n .scalar, x, dY =>
      let y := softmaxVecSpec (α := α) (n := n) x
      let s : α := sumSpec (mulSpec dY y)
      mulSpec y (subSpec dY (replicate (Tensor.scalar s)))
  | .dim n inner, Tensor.dim xF, Tensor.dim dF =>
      Tensor.dim (fun i : Fin n => softmaxBackwardSpec (s := inner) (xF i) (dF i))

/-
## Log-softmax (stable)

`log_softmax` is often the numerically-preferred form for cross-entropy on logits:

`CE(p, logits) = -mean_i p_i * log_softmax(logits)_i`.

We define it with the same max-shift trick as `softmax_vec_spec`, but return log-probabilities
directly to avoid ever computing `log(0)` when `exp` underflows.
-/

/-- Log-softmax on a length-`n` vector. -/
def logSoftmaxVecSpec {n : Nat} (t : Tensor α (.dim n .scalar)) : Tensor α (.dim n .scalar) :=
  match n with
  | 0 => t
  | Nat.succ n' =>
      let maxT : Tensor α .scalar :=
        match t with
        | Tensor.dim f =>
            let first : α := match f ⟨0, Nat.succ_pos n'⟩ with | Tensor.scalar v => v
            let maxVal : α :=
              (List.finRange (Nat.succ n')).foldl
                (fun acc i =>
                  match f i with
                  | Tensor.scalar v => max acc v)
                first
            Tensor.scalar maxVal
      let shifted : Tensor α (.dim (Nat.succ n') .scalar) := subSpec t (replicate maxT)
      let ex := expSpec shifted
      let denom : α :=
        match ex with
        | Tensor.dim g =>
            (List.finRange (Nat.succ n')).foldl
              (fun acc i => acc + match g i with | Tensor.scalar v => v)
              0
      let logDenom : α := MathFunctions.log denom
      subSpec shifted (replicate (Tensor.scalar logDenom))

/-- Log-softmax along the last axis (recurses over outer dimensions). -/
def logSoftmaxSpec : {s : Shape} → Tensor α s → Tensor α s
  | .scalar, _ => Tensor.scalar 0
  | .dim n .scalar, t => logSoftmaxVecSpec (α := α) (n := n) t
  | .dim n inner, Tensor.dim f =>
      Tensor.dim (fun i : Fin n => logSoftmaxSpec (s := inner) (f i))

/-- Backward/VJP for last-axis log-softmax.

If `y = log_softmax(x)`, then `softmax(x) = exp(y)` and the vector-Jacobian product is

`dL/dx = dL/dy - softmax(x) * sum(dL/dy)`.

This is the same formula used by PyTorch's stable `log_softmax` backward path.  We take the
already-computed output `y` rather than the logits `x`, so runtime backends can avoid recomputing
the max-shifted forward pass during backprop.
-/
def logSoftmaxBackwardSpec : {s : Shape} → Tensor α s → Tensor α s → Tensor α s
  | .scalar, _y, _dY => Tensor.scalar 0
  | .dim _n .scalar, y, dY =>
      let probs := expSpec y
      let rowSum : α := sumSpec dY
      subSpec dY (mulSpec probs (replicate (Tensor.scalar rowSum)))
  | .dim n inner, Tensor.dim yF, Tensor.dim dF =>
      Tensor.dim (fun i : Fin n => logSoftmaxBackwardSpec (s := inner) (yF i) (dF i))

/-- Tensor-level leaky ReLU (pointwise).  PyTorch analogy: `torch.nn.functional.leaky_relu`. -/
def leakyReluSpec {α : Type} [Zero α] [Mul α] [LT α] [DecidableRel ((· > ·) : α → α → Prop)] {s :
  Shape} (t : Tensor α s) (αₗ : α) : Tensor α s :=
  mapSpec (Activation.Math.leakyReluSpec αₗ) t

/-- Tensor-level derivative of leaky ReLU (pointwise). -/
def leakyReluDerivSpec {α : Type} [Zero α] [One α] [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
  {s : Shape} (t : Tensor α s) (αₗ : α) : Tensor α s :=
  mapSpec (Activation.Math.leakyReluDerivSpec αₗ) t

/-- Tensor-level ELU (pointwise).  PyTorch analogy: `torch.nn.functional.elu`. -/
def eluSpec {α : Type} [Zero α] [One α] [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
  [MathFunctions α] [Sub α] [Mul α] {s : Shape} (t : Tensor α s) (alpha : α) : Tensor α s :=
  mapSpec (Activation.Math.eluSpec alpha) t

/-- Tensor-level derivative of ELU (pointwise). -/
def eluDerivSpec {α : Type} [Zero α] [One α] [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
  [MathFunctions α] [Mul α] {s : Shape} (t : Tensor α s) (alpha : α) : Tensor α s :=
  mapSpec (Activation.Math.eluDerivSpec alpha) t

/-- Tensor-level GELU (approximate, pointwise).  PyTorch analogy: `gelu(..., approximate="tanh")`.
  -/
def geluSpec {α : Type} [MathFunctions α] [OfScientific α] [Add α] [Mul α] [Div α] [Sub α] [OfNat α
  1] {s : Shape} (t : Tensor α s) : Tensor α s :=
  mapSpec Activation.Math.geluSpec t

/-- Tensor-level derivative of tanh-approx GELU (pointwise). -/
def geluDerivSpec {α : Type} [MathFunctions α] [OfScientific α] [Add α] [Mul α] [Div α] [Sub α]
  [OfNat α 1] {s : Shape} (t : Tensor α s) : Tensor α s :=
  mapSpec Activation.Math.geluDerivSpec t

/-- Tensor-level Swish / SiLU (pointwise). -/
def swishSpec {s : Shape} (t : Tensor α s) : Tensor α s :=
  mapSpec Activation.Math.swishSpec t

/-- Tensor-level derivative of Swish / SiLU (pointwise). -/
def swishDerivSpec {s : Shape} (t : Tensor α s) : Tensor α s :=
  mapSpec Activation.Math.swishDerivSpec t

/-- Tensor-level softplus (pointwise). -/
def softplusSpec {s : Shape} (t : Tensor α s) : Tensor α s :=
  mapSpec Activation.Math.softplusSpec t

/-- Tensor-level derivative of softplus (pointwise). -/
def softplusDerivSpec {s : Shape} (t : Tensor α s) : Tensor α s :=
  mapSpec Activation.Math.softplusDerivSpec t

/-- Tensor-level `safe_log_spec` (pointwise). -/
def safeLogSpec {s : Shape} (t : Tensor α s) (ε : α := Numbers.epsilon) : Tensor α s :=
  mapSpec (fun x => Activation.Math.safeLogSpec (α := α) x ε) t

/-- Tensor-level derivative of `safe_log_spec` (pointwise). -/
def safeLogDerivSpec {s : Shape} (t : Tensor α s) (ε : α := Numbers.epsilon) : Tensor α s :=
  mapSpec (fun x => Activation.Math.safeLogDerivSpec (α := α) x ε) t

/-- Tensor-level `smooth_abs_spec` (pointwise). -/
def smoothAbsSpec {s : Shape} (t : Tensor α s) (ε : α := Numbers.epsilon) : Tensor α s :=
  mapSpec (fun x => Activation.Math.smoothAbsSpec (α := α) x ε) t

/-- Tensor-level derivative of `smooth_abs_spec` (pointwise). -/
def smoothAbsDerivSpec {s : Shape} (t : Tensor α s) (ε : α := Numbers.epsilon) : Tensor α s :=
  mapSpec (fun x => Activation.Math.smoothAbsDerivSpec (α := α) x ε) t

-- Generic activation gradient computation
-- Applies the chain rule: ∂L/∂x = ∂L/∂f(x) * f'(x)
/-- A generic pointwise activation VJP helper.

Given:

- `f'` (as a tensor-level derivative function),
- the forward input `x`,
- and an upstream gradient `dL/df(x)`,

this returns `dL/dx` by the chain rule:

`dL/dx = dL/df(x) ⊙ f'(x)`.

This matches how most PyTorch elementwise ops behave in backward: multiply upstream gradients by
the pointwise derivative mask/value.
-/
def activationGradientSpec {s : Shape}
  (activation_deriv : Tensor α s → Tensor α s)
  (input : Tensor α s)
  (grad_output : Tensor α s) :
  Tensor α s :=
  mulSpec grad_output (activation_deriv input)

end Activation

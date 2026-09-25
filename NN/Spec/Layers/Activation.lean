/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape.LinearAlgebra

/-!
# Activation Specifications

Scalar activation functions and their chosen derivatives live in `Activation.Math`. Tensor
operations map those definitions pointwise, except for shape-dependent operations such as softmax
and log-softmax. The definitions are polymorphic over the scalar `Context`, allowing the same layer
specification to be interpreted over runtime floats, exact scalars, or verification domains.

The formulas and conventions follow these references:

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


open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Activation

/-- Activation functions with a parameter-free pointwise interpretation.

This type is shared by model specifications and public model builders. Keeping the choice in the
specification layer prevents configuration strings from silently selecting the wrong semantics.
-/
inductive Kind where
  /-- Rectified linear unit, `max(0, x)`. -/
  | relu
  /-- GELU in its tanh approximation (`Math.geluSpec`). This is PyTorch's
  `nn.GELU(approximate='tanh')`, not the default erf-based `nn.GELU()`. -/
  | gelu
  /-- SiLU/Swish, `x * sigmoid(x)`. -/
  | silu
  /-- Hyperbolic tangent. -/
  | tanh
  /-- Logistic sigmoid. -/
  | sigmoid
deriving Repr, DecidableEq

namespace Math

variable {α : Type} [TorchLean.Storage α] [Context α]

/-! ## Scalar activations -/

/-- ReLU: $\operatorname{ReLU}(x)=\max(x,0)$.

PyTorch analogy: `torch.nn.functional.relu`.

This is the simplest nonlinearity we use throughout TorchLean because it stays meaningful across
many scalar backends (including ones that do not support `exp/log`).

At zero we return the scalar zero directly. This matters for dual numbers: their equality compares
primals, so this branch clears every tangent at the kink, matching `reluDerivSpec`. Elementwise
maximum has a different convention and splits a tie equally. Every other input retains the
backend's `max` result, including a floating-point NaN, which does not compare equal to zero.
-/
def reluSpec {α : Type} [Zero α] [Max α] [BEq α] (x : α) : α :=
  if x == 0 then 0 else Max.max x 0

/-- The explicit zero test preserves maximum on ordered scalars with ordinary equality. -/
@[simp] theorem relu_zero_branch_eq_max {α : Type} [LinearOrder α] [Zero α] (x : α) :
    (if x == 0 then (0 : α) else max x 0) = max x 0 := by
  by_cases hx : x = 0 <;> simp [hx]

/-- On an ordered scalar with ordinary equality, ReLU is the usual maximum with zero. -/
@[simp] theorem reluSpec_eq_max {α : Type} [LinearOrder α] [Zero α] (x : α) :
    reluSpec x = max x 0 :=
  relu_zero_branch_eq_max x

/-- A standard subgradient choice for ReLU:

$\frac{d}{dx}\operatorname{ReLU}(x)=1$ if $x>0$, and $0$ otherwise.

PyTorch analogy: autograd picks a subgradient at $x=0$; our spec commits to a concrete one to
make "the derivative" a pure function.

The `DecidableRel (· > ·)` constraint reflects that this definition branches on $x>0$.
-/
def reluDerivSpec {α : Type} [Zero α] [One α] [LT α] [DecidableRel ((· > ·) : α → α → Prop)] (x :
  α) : α :=
  if x > 0 then 1 else 0

/-- Logistic sigmoid, evaluated with a nonpositive exponential argument:

$\operatorname{sigmoid}(x)=1/(1+\exp(-x))$.

For positive inputs we use this expression directly. For zero and negative inputs we use the
equivalent ratio $\exp(x)/(1+\exp(x))$. Both denominators lie between `1` and `2` over the reals.
In floating-point arithmetic this avoids forming `exp(-x)` when `x` is a large negative number.
The same branch also matters for dual numbers: differentiating a quotient with an infinite
denominator can produce `NaN`, even when the sigmoid value has rounded to zero.

PyTorch analogy: `torch.nn.functional.sigmoid` (or `torch.sigmoid`).
-/
def sigmoidSpec (x : α) : α :=
  if x > 0 then
    1 / (1 + MathFunctions.exp (-x))
  else
    let z := MathFunctions.exp x
    z / (1 + z)

/-- Derivative of sigmoid:

$\operatorname{sigmoid}'(x)=\sigma(x)(1-\sigma(x))$.

The sigmoid value uses the stable branch above, so negative tails retain their small positive
derivatives until the exponential itself underflows. On the positive side, the derivative becomes
zero once the sigmoid value rounds to one; this is the output-based derivative used by the graph
VJP as well.
-/
def sigmoidDerivSpec (x : α) : α :=
  let s := sigmoidSpec x
  s * (1 - s)

/-- Hyperbolic tangent: `tanh(x)`.  PyTorch analogy: `torch.tanh`. -/
def tanhSpec (x : α) : α :=
  MathFunctions.tanh x

/-- Derivative of tanh:

$\tanh'(x)=1-\tanh^2(x)$.
-/
def tanhDerivSpec (x : α) : α :=
  1 - ((MathFunctions.tanh x) * (MathFunctions.tanh x))

/-- Leaky ReLU:

$\operatorname{leaky\_relu}(x;\alpha)=x$ if $x>0$, else $\alpha x$.

PyTorch analogy: `torch.nn.functional.leaky_relu` with `negative_slope = α`.
-/
def leakyReluSpec {α : Type} [Zero α] [Mul α] [LT α] [DecidableRel ((· > ·) : α → α → Prop)] (x :
  α) (αₗ : α) : α :=
  if x > 0 then x else αₗ * x

/-- Derivative of leaky ReLU:

$\frac{d}{dx}\operatorname{leaky\_relu}(x;\alpha)=1$ if $x>0$, else $\alpha$.
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

/-- Logistic form written as $\exp(x)/(\exp(x)+1)$.

This is mathematically the same sigmoid function as `sigmoidSpec`; we keep it as `logisticSpec`
because several scalar approximation proofs reason about this `exp(x)` numerator form directly.

Important naming choice: this is **not** called scalar softmax. A one-entry softmax is always `1`;
the real softmax API in TorchLean is the axis-parametric tensor operation `Activation.softmaxSpec`.
-/
def logisticSpec (x : α) : α :=
  MathFunctions.exp x / (MathFunctions.exp x + 1)

/-- Derivative of `logisticSpec`, expressed in output form. -/
def logisticDerivSpec (x : α) : α :=
  logisticSpec x * (1 - logisticSpec x)

/-- ELU (Exponential Linear Unit):

$\operatorname{ELU}(x;\alpha)=x$ if $x>0$, else $\alpha(\exp(x)-1)$.

PyTorch analogy: `torch.nn.functional.elu` with `alpha = α`.
-/
def eluSpec {α : Type} [Zero α] [One α] [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
  [MathFunctions α] [Sub α] [Mul α] (x : α) (alpha : α) : α :=
  if x > 0 then x else alpha * (MathFunctions.exp x - 1)

/-- Derivative of ELU:

$\operatorname{ELU}'(x;\alpha)=1$ if $x>0$, else $\alpha\exp(x)$.
-/
def eluDerivSpec {α : Type} [Zero α] [One α] [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
  [MathFunctions α] [Mul α] (x : α) (alpha : α) : α :=
  if x > 0 then 1 else alpha * MathFunctions.exp x

/-- The rational coefficient `44715 / 1000000` in the standard tanh approximation to GELU. -/
def geluTanhCoeff {α : Type} [TorchLean.Storage α] [Context α] : α :=
  ((44715 : Nat) : α) / ((1000000 : Nat) : α)

/-- GELU (approximate): the common tanh-based approximation used in many Transformer codebases.

PyTorch analogy: `torch.nn.functional.gelu(x, approximate="tanh")`. This is not PyTorch's default
`nn.GELU()`, which uses the exact erf form; TorchLean has no erf primitive, so only the tanh
approximation is specified.
-/
def geluSpec {α : Type} [TorchLean.Storage α] [Context α] (x : α) : α :=
  let two : α := (1 : α) + (1 : α)
  let pi : α := MathFunctions.pi
  let sqrtTwoOverPi := MathFunctions.sqrt (two / pi)
  let coeff : α := geluTanhCoeff
  x * ((1 : α) + MathFunctions.tanh (sqrtTwoOverPi * (x + coeff * x * x * x))) / two

/-- GELU derivative for the tanh-based approximation. -/
def geluDerivSpec {α : Type} [TorchLean.Storage α] [Context α] (x : α) : α :=
  let two : α := (1 : α) + (1 : α)
  let three : α := (1 : α) + (1 : α) + (1 : α)
  let pi : α := MathFunctions.pi
  let sqrtTwoOverPi := MathFunctions.sqrt (two / pi)
  let coeff : α := geluTanhCoeff
  let tanhTerm := MathFunctions.tanh (sqrtTwoOverPi * (x + coeff * x * x * x))
  let sechTerm := (1 : α) - tanhTerm * tanhTerm
  let innerDeriv := sqrtTwoOverPi * ((1 : α) + three * coeff * x * x)
  ((1 : α) + tanhTerm + x * sechTerm * innerDeriv) / two

/-- Swish / SiLU:

$\operatorname{swish}(x)=x\operatorname{sigmoid}(x)$.

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

/-- Softplus, evaluated without a large positive exponential:

$\operatorname{softplus}(x)=\log(1+\exp(x))$.

The positive branch uses the equivalent expression $x+\log(1+\exp(-x))$; this keeps finite
floating-point inputs finite when `exp(x)` itself would overflow. The operation remains the
one-argument, unit-scale softplus used throughout TorchLean.

PyTorch analogy: `torch.nn.functional.softplus`.
-/
def softplusSpec (x : α) : α :=
  if x > 0 then
    x + MathFunctions.log (1 + MathFunctions.exp (-x))
  else
    MathFunctions.log (1 + MathFunctions.exp x)

/-- Derivative of softplus:

$\operatorname{softplus}'(x)=\operatorname{sigmoid}(x)$.

Using the same stable sigmoid keeps the derivative finite in both tails. Evaluating this formula
over dual scalars also propagates the softplus second derivative without a large exponential.
-/
def softplusDerivSpec (x : α) : α :=
  sigmoidSpec x

/-- A smooth log surrogate:

$\operatorname{safe\_log}(x;\varepsilon)
=\log(\operatorname{softplus}(x)+\varepsilon)$.

We use this when we want something "log-like" without having to carry side conditions about the
input being strictly positive.
-/
def safeLogSpec (x : α) (ε : α := Context.defaultEpsilon) : α :=
  MathFunctions.log (softplusSpec x + ε)

/-- Derivative of `safeLogSpec`. -/
def safeLogDerivSpec (x : α) (ε : α := Context.defaultEpsilon) : α :=
  softplusDerivSpec x / (softplusSpec x + ε)

/-- A smooth absolute value surrogate:

$\operatorname{smooth\_abs}(x;\varepsilon)=\sqrt{x^2+\varepsilon}$.

Useful when you want an `abs`-like shape but keep differentiability at `0`.
-/
def smoothAbsSpec (x : α) (ε : α := Context.defaultEpsilon) : α :=
  MathFunctions.sqrt (x * x + ε)

/-- Derivative of `smoothAbsSpec`. -/
def smoothAbsDerivSpec (x : α) (ε : α := Context.defaultEpsilon) : α :=
  x / smoothAbsSpec x ε

end Math

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- Tensor-level tanh (pointwise).

PyTorch analogy: `torch.tanh(t)` or `torch.nn.functional.tanh(t)` applied elementwise.
-/
def tanhSpec {s : Shape} : Tensor α s → Tensor α s :=
  mapSpec Activation.Math.tanhSpec

/-- Tensor-level ReLU (pointwise). -/
def reluSpec {α : Type} [TorchLean.Storage α] [Zero α] [Max α] [BEq α]
    {s : Shape} (t : Tensor α s) : Tensor α s :=
  mapSpec Activation.Math.reluSpec t

/-- Tensor-level sigmoid (pointwise). -/
def sigmoidSpec {s : Shape} (t : Tensor α s) : Tensor α s :=
  mapSpec Activation.Math.sigmoidSpec t

/-- Tensor-level ReLU derivative (pointwise), using the scalar subgradient choice in
`Activation.Math.reluDerivSpec`. -/
def reluDerivSpec {α : Type} [TorchLean.Storage α] [Zero α] [One α] [LT α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (t : Tensor α s) : Tensor α s :=
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
  mulSpec sigmoidOutput (subSpec (Tensor.full s 1) sigmoidOutput)

/-- Tensor-level tanh derivative (pointwise). -/
def tanhDerivSpec {s : Shape} (t : Tensor α s) : Tensor α s :=
  mapSpec Activation.Math.tanhDerivSpec t

/-- Apply a parameter-free pointwise activation to a tensor. -/
def Kind.applySpec {s : Shape} : Kind → Tensor α s → Tensor α s
  | .relu => reluSpec
  | .gelu => mapSpec Math.geluSpec
  | .silu => mapSpec Math.swishSpec
  | .tanh => tanhSpec
  | .sigmoid => sigmoidSpec

/-- Apply the derivative selected by a parameter-free pointwise activation. -/
def Kind.derivSpec {s : Shape} : Kind → Tensor α s → Tensor α s
  | .relu => reluDerivSpec
  | .gelu => mapSpec Math.geluDerivSpec
  | .silu => mapSpec Math.swishDerivSpec
  | .tanh => tanhDerivSpec
  | .sigmoid => sigmoidDerivSpec

/-!
## Softmax on tensors

These are the shape‑aware softmax definitions used in attention / classification layers.
They recurse over outer dimensions and apply a numerically‑stable softmax to the last axis.
-/

/-- Maximum entry of a nonempty vector, returned as a scalar tensor.

The fold is seeded by the first coordinate rather than by a numeric sentinel. Consequently the
result is one of the input coordinates for every linearly ordered scalar type. Softmax and
log-softmax share this definition so their range-reduction convention cannot drift apart.
-/
def maxVecSpec {n : Nat} (t : Tensor α [Nat.succ n]) : Tensor α .scalar :=
  let first : α := t.getScalar ⟨0, Nat.succ_pos n⟩
  let maximum : α :=
    (List.finRange (Nat.succ n)).foldl
      (fun acc i => max acc (t.getScalar i))
      first
  Tensor.scalar maximum

/-- Max-shifted exponentials shared by stable softmax and log-softmax. -/
def maxShiftedExpVecSpec {n : Nat}
    (t : Tensor α [Nat.succ n]) : Tensor α [Nat.succ n] :=
  expSpec (subSpec t (replicate (maxVecSpec t)))

/-- Softmax on a length-`n` vector.

This is the "real" softmax, not the scalar logistic helper in `Activation.Math.logisticSpec`.

Numerical stability:

We implement the standard stabilized form
$\operatorname{softmax}(x)_i=\exp(x_i-m)/\sum_j\exp(x_j-m)$, where
$m=\max_i x_i$. Subtracting the max avoids overflow in typical floating-point backends, and it is
also a nice canonical form to reference in proofs.
-/
def softmaxVecSpec {n : Nat} (t : Tensor α [n]) : Tensor α [n] :=
  match n with
  | 0 => t
  | Nat.succ _ =>
      let ex := maxShiftedExpVecSpec t
      let denom : α := sumSpec ex
      divSpec ex (replicate (Tensor.scalar denom))

/-- Softmax along the last axis (recurses over outer dimensions).

PyTorch analogy: `torch.softmax(x, dim=-1)`.

For `s = .scalar` we return `1` (there is only one coordinate). For higher-rank tensors we keep
the outer structure and apply `softmaxVecSpec` at the last axis.
-/
def Internal.softmaxInnermostSpec : {s : Shape} → Tensor α s → Tensor α s
  | .scalar, _ => Tensor.scalar 1
  | .dim n .scalar, t => softmaxVecSpec (α := α) (n := n) t
  | .dim n inner, tensor =>
      Tensor.dim fun i : Fin n =>
        Internal.softmaxInnermostSpec (s := inner) (Tensor.unstack tensor i)

/-- Last-axis softmax on a matrix acts independently on each row. -/
@[simp] theorem unstack_softmaxInnermostSpec_matrix {rows columns : Nat}
    (tensor : Tensor α [rows, columns]) (row : Fin rows) :
    Tensor.unstack (Internal.softmaxInnermostSpec tensor) row =
      softmaxVecSpec (Tensor.unstack tensor row) := by
  rw [show Internal.softmaxInnermostSpec tensor =
    Tensor.dim (fun i => softmaxVecSpec (Tensor.unstack tensor i)) by rfl]
  simp

/-- Backward/VJP for last-axis softmax.

If $y=\operatorname{softmax}(x)$ and we are given an upstream gradient $\partial L/\partial y$,
then for each last-axis slice:

$$
\frac{\partial L}{\partial x}
=y\odot\left(
  \frac{\partial L}{\partial y}
  -\left\langle\frac{\partial L}{\partial y},y\right\rangle
\right).
$$

This is the standard Jacobian-vector product for softmax, written in a way that avoids materializing
the full `n×n` Jacobian.
-/
def Internal.softmaxInnermostBackwardSpec : {s : Shape} → Tensor α s → Tensor α s → Tensor α s
  | .scalar, _x, _dY => Tensor.scalar 0
  | .dim n .scalar, x, dY =>
      let y := softmaxVecSpec (α := α) (n := n) x
      let s : α := sumSpec (mulSpec dY y)
      mulSpec y (subSpec dY (replicate (Tensor.scalar s)))
  | .dim n inner, x, dY =>
      Tensor.dim fun i : Fin n =>
        Internal.softmaxInnermostBackwardSpec (s := inner)
          (Tensor.unstack x i) (Tensor.unstack dY i)

/-- Numerically stable softmax along any tensor dimension.

The selected dimension is moved to the innermost position, where a private kernel computes each
one-dimensional slice, and is then restored. This definition covers outer and interior dimensions
without imposing a memory-layout convention on the mathematical tensor.
-/
def softmaxSpec {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (x : Tensor α s) : Tensor α s :=
  let swaps := Shape.moveAxisToInnermostSwaps s.rank axis
  let moved := Tensor.permuteByAdjacentSwaps x swaps
  let y := Internal.softmaxInnermostSpec moved
  let restored := Tensor.permuteByAdjacentSwaps y swaps.reverse
  Shape.applyAdjacentSwaps_reverse s swaps ▸ restored

/-- Backward/VJP for softmax along any tensor dimension. -/
def softmaxBackwardSpec {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (x dY : Tensor α s) : Tensor α s :=
  let swaps := Shape.moveAxisToInnermostSwaps s.rank axis
  let movedX := Tensor.permuteByAdjacentSwaps x swaps
  let movedDY := Tensor.permuteByAdjacentSwaps dY swaps
  let dX := Internal.softmaxInnermostBackwardSpec movedX movedDY
  let restored := Tensor.permuteByAdjacentSwaps dX swaps.reverse
  Shape.applyAdjacentSwaps_reverse s swaps ▸ restored

/-
## Log-softmax (stable)

`logSoftmax` is often the numerically-preferred form for cross-entropy on logits:

`CE(p, logits) = -mean_i p_i * log_softmax(logits)_i`.

We define it with the same max-shift trick as `softmaxVecSpec`, but return log-probabilities
directly to avoid ever computing `log(0)` when `exp` underflows.
-/

/-- Log-softmax on a length-`n` vector. -/
def logSoftmaxVecSpec {n : Nat} (t : Tensor α [n]) : Tensor α [n] :=
  match n with
  | 0 => t
  | Nat.succ n' =>
      let maxT : Tensor α .scalar := maxVecSpec t
      let shifted : Tensor α [Nat.succ n'] := subSpec t (replicate maxT)
      let ex := maxShiftedExpVecSpec t
      let denom : α := sumSpec ex
      let logDenom : α := MathFunctions.log denom
      subSpec shifted (replicate (Tensor.scalar logDenom))

/-- Log-softmax along the last axis (recurses over outer dimensions). -/
def Internal.logSoftmaxInnermostSpec : {s : Shape} → Tensor α s → Tensor α s
  | .scalar, _ => Tensor.scalar 0
  | .dim n .scalar, t => logSoftmaxVecSpec (α := α) (n := n) t
  | .dim n inner, tensor =>
      Tensor.dim fun i : Fin n =>
        Internal.logSoftmaxInnermostSpec (s := inner) (Tensor.unstack tensor i)

/-- Forward-mode JVP for last-axis log-softmax.

If $y=\operatorname{logsoftmax}(x)$, then each last-axis slice has directional derivative

$dy=dx-\operatorname{replicate}(\langle\exp(y),dx\rangle)$.

Unlike the VJP below, the subtracted scalar is replicated uniformly across the slice; the
softmax probabilities occur only inside the dot product. Taking the already-computed output `y`
also avoids recomputing the stable forward pass.
-/
def Internal.logSoftmaxInnermostJvpSpec : {s : Shape} → Tensor α s → Tensor α s → Tensor α s
  | .scalar, _y, _dx => Tensor.scalar 0
  | .dim _n .scalar, y, dx =>
      let probs := expSpec y
      let directionalMean : α := dotSpec probs dx
      subSpec dx (replicate (Tensor.scalar directionalMean))
  | .dim n inner, y, dx =>
      Tensor.dim fun i : Fin n =>
        Internal.logSoftmaxInnermostJvpSpec (s := inner)
          (Tensor.unstack y i) (Tensor.unstack dx i)

/-- Backward/VJP for last-axis log-softmax.

If $y=\operatorname{logsoftmax}(x)$, then $\operatorname{softmax}(x)=\exp(y)$ and the
vector-Jacobian product is

$$
\frac{\partial L}{\partial x}
=\frac{\partial L}{\partial y}
 -\operatorname{softmax}(x)\sum_i\frac{\partial L}{\partial y_i}.
$$

This is the same formula used by PyTorch's stable `log_softmax` backward path.  We take the
already-computed output `y` rather than the logits `x`, so runtime backends can avoid recomputing
the max-shifted forward pass during backprop.
-/
def Internal.logSoftmaxInnermostBackwardSpec :
    {s : Shape} → Tensor α s → Tensor α s → Tensor α s
  | .scalar, _y, _dY => Tensor.scalar 0
  | .dim _n .scalar, y, dY =>
      let probs := expSpec y
      let rowSum : α := sumSpec dY
      subSpec dY (mulSpec probs (replicate (Tensor.scalar rowSum)))
  | .dim n inner, y, dY =>
      Tensor.dim fun i : Fin n =>
        Internal.logSoftmaxInnermostBackwardSpec (s := inner)
          (Tensor.unstack y i) (Tensor.unstack dY i)

/-- Numerically stable log-softmax along any tensor dimension. -/
def logSoftmaxSpec {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (x : Tensor α s) : Tensor α s :=
  let swaps := Shape.moveAxisToInnermostSwaps s.rank axis
  let moved := Tensor.permuteByAdjacentSwaps x swaps
  let y := Internal.logSoftmaxInnermostSpec moved
  let restored := Tensor.permuteByAdjacentSwaps y swaps.reverse
  Shape.applyAdjacentSwaps_reverse s swaps ▸ restored

/-- Forward-mode derivative of log-softmax along any tensor dimension. -/
def logSoftmaxJvpSpec {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (y dx : Tensor α s) : Tensor α s :=
  let swaps := Shape.moveAxisToInnermostSwaps s.rank axis
  let movedY := Tensor.permuteByAdjacentSwaps y swaps
  let movedDX := Tensor.permuteByAdjacentSwaps dx swaps
  let dy := Internal.logSoftmaxInnermostJvpSpec movedY movedDX
  let restored := Tensor.permuteByAdjacentSwaps dy swaps.reverse
  Shape.applyAdjacentSwaps_reverse s swaps ▸ restored

/-- Backward/VJP for log-softmax along any tensor dimension. -/
def logSoftmaxBackwardSpec {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (y dY : Tensor α s) : Tensor α s :=
  let swaps := Shape.moveAxisToInnermostSwaps s.rank axis
  let movedY := Tensor.permuteByAdjacentSwaps y swaps
  let movedDY := Tensor.permuteByAdjacentSwaps dY swaps
  let dX := Internal.logSoftmaxInnermostBackwardSpec movedY movedDY
  let restored := Tensor.permuteByAdjacentSwaps dX swaps.reverse
  Shape.applyAdjacentSwaps_reverse s swaps ▸ restored

/-- Tensor-level leaky ReLU (pointwise).  PyTorch analogy: `torch.nn.functional.leaky_relu`. -/
def leakyReluSpec {α : Type} [TorchLean.Storage α] [Zero α] [Mul α] [LT α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (t : Tensor α s) (αₗ : α) : Tensor α s :=
  mapSpec (Activation.Math.leakyReluSpec αₗ) t

/-- Tensor-level derivative of leaky ReLU (pointwise). -/
def leakyReluDerivSpec {α : Type} [TorchLean.Storage α] [Zero α] [One α] [LT α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (t : Tensor α s) (αₗ : α) : Tensor α s :=
  mapSpec (Activation.Math.leakyReluDerivSpec αₗ) t

/-- Tensor-level ELU (pointwise).  PyTorch analogy: `torch.nn.functional.elu`. -/
def eluSpec {α : Type} [TorchLean.Storage α] [Zero α] [One α] [LT α]
    [DecidableRel ((· > ·) : α → α → Prop)] [MathFunctions α] [Sub α] [Mul α]
    {s : Shape} (t : Tensor α s) (alpha : α) : Tensor α s :=
  mapSpec (Activation.Math.eluSpec alpha) t

/-- Tensor-level derivative of ELU (pointwise). -/
def eluDerivSpec {α : Type} [TorchLean.Storage α] [Zero α] [One α] [LT α]
    [DecidableRel ((· > ·) : α → α → Prop)] [MathFunctions α] [Mul α]
    {s : Shape} (t : Tensor α s) (alpha : α) : Tensor α s :=
  mapSpec (Activation.Math.eluDerivSpec alpha) t

/-- Tensor-level GELU (approximate, pointwise). PyTorch analogy: `gelu(..., approximate="tanh")`. -/
def geluSpec {α : Type} [TorchLean.Storage α] [Context α] {s : Shape} (t : Tensor α s) :
    Tensor α s :=
  mapSpec Activation.Math.geluSpec t

/-- Tensor-level derivative of tanh-approx GELU (pointwise). -/
def geluDerivSpec {α : Type} [TorchLean.Storage α] [Context α] {s : Shape} (t : Tensor α s) :
    Tensor α s :=
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

/-- Tensor-level `safeLogSpec` (pointwise). -/
def safeLogSpec {s : Shape} (t : Tensor α s) (ε : α := Context.defaultEpsilon) : Tensor α s :=
  mapSpec (fun x => Activation.Math.safeLogSpec (α := α) x ε) t

/-- Tensor-level derivative of `safeLogSpec` (pointwise). -/
def safeLogDerivSpec {s : Shape} (t : Tensor α s) (ε : α := Context.defaultEpsilon) : Tensor α s :=
  mapSpec (fun x => Activation.Math.safeLogDerivSpec (α := α) x ε) t

/-- Tensor-level `smoothAbsSpec` (pointwise). -/
def smoothAbsSpec {s : Shape} (t : Tensor α s) (ε : α := Context.defaultEpsilon) : Tensor α s :=
  mapSpec (fun x => Activation.Math.smoothAbsSpec (α := α) x ε) t

/-- Tensor-level derivative of `smoothAbsSpec` (pointwise). -/
def smoothAbsDerivSpec {s : Shape} (t : Tensor α s) (ε : α := Context.defaultEpsilon) :
    Tensor α s :=
  mapSpec (fun x => Activation.Math.smoothAbsDerivSpec (α := α) x ε) t

/-- A generic pointwise activation VJP helper.

Given:

- `f'` (as a tensor-level derivative function),
- the forward input `x`,
- and an upstream gradient $\partial L/\partial f(x)$,

this returns $\partial L/\partial x$ by the chain rule:

$\frac{\partial L}{\partial x}
=\frac{\partial L}{\partial f(x)}\odot f'(x)$.

This matches how most PyTorch elementwise ops behave in backward: multiply upstream gradients by
the pointwise derivative mask/value.
-/
def vjpSpec {s : Shape}
  (activationDeriv : Tensor α s → Tensor α s)
  (input : Tensor α s)
  (gradOutput : Tensor α s) :
  Tensor α s :=
  mulSpec gradOutput (activationDeriv input)

end Activation

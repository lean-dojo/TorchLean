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
- Blanchard, Higham and Higham, "Accurate Computation of the Log-Sum-Exp and Softmax Functions":
  https://arxiv.org/abs/1909.03469
- Barron, Gnanapragasam and Tschannen, "A Fix for Softmax and Cross-Entropy":
  https://arxiv.org/abs/2610.31642
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
  Context.sigmoid x

/-- Derivative of sigmoid:

$\operatorname{sigmoid}'(x)=\sigma(x)(1-\sigma(x))$.

We evaluate the equivalent exponential ratio directly, rather than subtracting a rounded sigmoid
value from one. This retains representable derivatives on both tails.
-/
def sigmoidDerivSpec (x : α) : α :=
  TorchLean.Numeric.sigmoidDeriv x

/-- Hyperbolic tangent: `tanh(x)`.  PyTorch analogy: `torch.tanh`. -/
def tanhSpec (x : α) : α :=
  MathFunctions.tanh x

/-- Derivative of tanh:

$\tanh'(x)=1-\tanh^2(x)$.

The runtime squares a scaled reciprocal cosh instead of subtracting rounded squares from one.
-/
def tanhDerivSpec (x : α) : α :=
  TorchLean.Numeric.tanhDeriv x

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

The negative branch uses the scalar context's compensated exponential difference, so small inputs
do not disappear when `exp x` rounds to one. Differentiation carriers supply its chain rule.
-/
def eluSpec {α : Type} [Context α] (x : α) (alpha : α) : α :=
  if x > 0 then x else alpha * Context.expm1 x

/-- Derivative of ELU:

$\operatorname{ELU}'(x;\alpha)=1$ if $x>0$, else $\alpha\exp(x)$.
-/
def eluDerivSpec {α : Type} [Zero α] [One α] [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
  [MathFunctions α] [Mul α] (x : α) (alpha : α) : α :=
  if x > 0 then 1 else alpha * MathFunctions.exp x

/-- The GELU tanh coefficient, rounded from the exact rational `44715 / 1000000`.

Casting the fraction once avoids overflowing its denominator in small floating-point formats.
-/
def geluTanhCoeff {α : Type} [RatCast α] : α :=
  Rat.cast (44715 / 1000000 : Rat)

/-- GELU (approximate): the common tanh-based approximation used in many Transformer codebases.

PyTorch analogy: `torch.nn.functional.gelu(x, approximate="tanh")`. This is not PyTorch's default
`nn.GELU()`, which uses the exact erf form. This activation selects the tanh approximation;
the separate pointwise `erf` operation depends on the scalar context's supported operations.

We evaluate the gate as `sigmoid (2 * inner)`, the same real function as `(1 + tanh inner) / 2`.
This avoids cancelling the negative tail when tanh rounds to minus one; it is not a different
approximation to the Gaussian CDF.
-/
def geluSpec {α : Type} [TorchLean.Storage α] [Context α] (x : α) : α :=
  let two : α := (1 : α) + (1 : α)
  let pi : α := MathFunctions.pi
  let sqrtTwoOverPi := MathFunctions.sqrt (two / pi)
  let coeff : α := geluTanhCoeff
  let inner := sqrtTwoOverPi * (x + coeff * x * x * x)
  let gate := sigmoidSpec (inner + inner)
  x * gate

/-- GELU derivative for the tanh-based approximation. -/
def geluDerivSpec {α : Type} [TorchLean.Storage α] [Context α] (x : α) : α :=
  let two : α := (1 : α) + (1 : α)
  let three : α := (1 : α) + (1 : α) + (1 : α)
  let pi : α := MathFunctions.pi
  let sqrtTwoOverPi := MathFunctions.sqrt (two / pi)
  let coeff : α := geluTanhCoeff
  let inner := sqrtTwoOverPi * (x + coeff * x * x * x)
  let sechTerm := tanhDerivSpec inner
  let innerDeriv := sqrtTwoOverPi * ((1 : α) + three * coeff * x * x)
  let correction := if sechTerm == 0 then 0 else x * sechTerm * innerDeriv
  (two * sigmoidSpec (inner + inner) + correction) / two

/-- Swish / SiLU:

$\operatorname{swish}(x)=x\operatorname{sigmoid}(x)$.

PyTorch analogy: `torch.nn.functional.silu`.

On the negative branch, split the exponential and multiply by `x` before its second small
factor. The product can remain representable after `exp x`, and hence sigmoid itself, underflows.
-/
def swishSpec (x : α) : α :=
  if x > 0 then x * sigmoidSpec x
  else
    let r := MathFunctions.exp (x / 2)
    ((x * r) * r) / (1 + r * r)

/-- Derivative of Swish / SiLU.

The negative branch groups `x + denominator` before multiplying the split exponential factors.
Forming sigmoid and its slope separately could underflow both before multiplying by `x`.
-/
def swishDerivSpec (x : α) : α :=
  if x > 0 then sigmoidSpec x + x * sigmoidDerivSpec x
  else
    let r := MathFunctions.exp (x / 2)
    let denominator := 1 + r * r
    (((x + denominator) * r) * r) / (denominator * denominator)

/-- `log(1 + x)` with compensation for rounding in the addition.
When `1 + x` rounds to one, return `x` rather than losing the small term. Equality, not order,
selects this case so the same formula also works for complex scalars. This is not a
correct-rounding guarantee for the backend's logarithm. -/
def log1pSpec (x : α) : α :=
  Context.log1p x

/-- `exp(x) - 1`, retaining small arguments when the exponential rounds to one.

Near zero, the logarithmic correction accounts for rounding in `exp(x)`. Outside the unit
neighborhood we use direct subtraction, avoiding a logarithm's branch ambiguity for complex
scalars. The zero-exponential branch handles underflow. As with `log1pSpec`, accuracy still
depends on the scalar implementation of `exp` and `log`.
-/
def expm1Spec (x : α) : α :=
  Context.expm1 x

/-- Softplus, evaluated without a large positive exponential:

$\operatorname{softplus}(x)=\log(1+\exp(x))$.

The positive branch uses the equivalent expression $x+\log(1+\exp(-x))$; this keeps finite
floating-point inputs finite when `exp(x)` itself would overflow. The operation remains the
one-argument, unit-scale softplus used throughout TorchLean.

PyTorch analogy: `torch.nn.functional.softplus`.
-/
def softplusSpec (x : α) : α :=
  if x > 0 then
    x + log1pSpec (MathFunctions.exp (-x))
  else
    log1pSpec (MathFunctions.exp x)

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
  let scale := max (MathFunctions.abs x) (MathFunctions.sqrt (MathFunctions.abs ε))
  if scale == 0 || !(scale / scale == scale / scale) then
    MathFunctions.sqrt (x * x + ε)
  else
    let normalized := x / scale
    scale * MathFunctions.sqrt (normalized * normalized + (ε / scale) / scale)

/-- Derivative of `smoothAbsSpec`. -/
def smoothAbsDerivSpec (x : α) (ε : α := Context.defaultEpsilon) : α :=
  x / smoothAbsSpec x ε

end Math

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- Tensor-level tanh (pointwise).

PyTorch analogy: `torch.tanh(t)` or `torch.nn.functional.tanh(t)` applied elementwise.
-/
def tanhSpec {s : Shape} : Tensor α s → Tensor α s :=
  Tensor.map Activation.Math.tanhSpec

/-- Tensor-level ReLU (pointwise). -/
def reluSpec {α : Type} [TorchLean.Storage α] [Zero α] [Max α] [BEq α]
    {s : Shape} (t : Tensor α s) : Tensor α s :=
  Tensor.map Activation.Math.reluSpec t

/-- Tensor-level sigmoid (pointwise). -/
def sigmoidSpec {s : Shape} (t : Tensor α s) : Tensor α s :=
  Tensor.map Activation.Math.sigmoidSpec t

/-- Tensor-level ReLU derivative (pointwise), using the scalar subgradient choice in
`Activation.Math.reluDerivSpec`. -/
def reluDerivSpec {α : Type} [TorchLean.Storage α] [Zero α] [One α] [LT α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (t : Tensor α s) : Tensor α s :=
  Tensor.map Activation.Math.reluDerivSpec t

/-- Tensor-level sigmoid derivative (pointwise). -/
def sigmoidDerivSpec {s : Shape} (t : Tensor α s) : Tensor α s :=
  Tensor.map Activation.Math.sigmoidDerivSpec t

/-- Apply sigmoid's pointwise pullback while retaining the incoming gradient in range repairs. -/
def sigmoidBackwardSpec {s : Shape} (x seed : Tensor α s) : Tensor α s :=
  map2Spec TorchLean.Numeric.sigmoidBackward x seed

/--
Derivative of sigmoid when the sigmoid output has already been computed.

This output-only formula cannot recover a small slope once the activation has rounded to one.
Use `sigmoidDerivSpec` on the preactivation when that value is available.
-/
def sigmoidOutputDerivSpec {α : Type} [TorchLean.Storage α] [One α] [Sub α] [Mul α]
    {s : Shape} (sigmoidOutput : Tensor α s) : Tensor α s :=
  mulSpec sigmoidOutput (subSpec (Tensor.full s 1) sigmoidOutput)

/-- Derivative of tanh from its cached output. At rounded endpoints this may lose a representable
slope; `tanhDerivSpec` uses the preactivation instead. -/
def tanhOutputDerivSpec {α : Type} [TorchLean.Storage α] [One α] [Sub α] [Mul α]
    {s : Shape} (tanhOutput : Tensor α s) : Tensor α s :=
  subSpec (Tensor.full s 1) (mulSpec tanhOutput tanhOutput)

/-- Tensor-level tanh derivative (pointwise). -/
def tanhDerivSpec {s : Shape} (t : Tensor α s) : Tensor α s :=
  Tensor.map Activation.Math.tanhDerivSpec t

/-- Apply tanh's pointwise pullback without separately rounding away a recoverable local slope. -/
def tanhBackwardSpec {s : Shape} (x seed : Tensor α s) : Tensor α s :=
  map2Spec TorchLean.Numeric.tanhBackward x seed

/-- Apply a parameter-free pointwise activation to a tensor. -/
def Kind.applySpec {s : Shape} : Kind → Tensor α s → Tensor α s
  | .relu => reluSpec
  | .gelu => Tensor.map Math.geluSpec
  | .silu => Tensor.map Math.swishSpec
  | .tanh => tanhSpec
  | .sigmoid => sigmoidSpec

/-- Apply the derivative selected by a parameter-free pointwise activation. -/
def Kind.derivSpec {s : Shape} : Kind → Tensor α s → Tensor α s
  | .relu => reluDerivSpec
  | .gelu => Tensor.map Math.geluDerivSpec
  | .silu => Tensor.map Math.swishDerivSpec
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

/-- Normalize finite weights without requiring their sum to fit in the output format.

The ordinary left fold is retained unless it loses range or swallows a nonzero term. In that
case, the complete sum and quotients are replayed over finite rational coefficient trees before
rounding each output. This also retains derivatives carried by nested scalars. Nonfinite inputs
and carriers without coefficient conversion keep the ordinary formula; a genuine zero sum is
not repaired. Information lost while constructing the weights cannot be recovered here.
-/
def Internal.normalizeWeights {n : Nat} (values : Tensor α [n]) : Tensor α [n] :=
  let ops := (inferInstance : Numeric.QuotientArithmetic α)
  if !ops.supported then
    divSpec values (replicate (Tensor.scalar (sumSpec values)))
  else
    let (denominator, stalled) := foldlSpec (fun (accumulator, stalled) value =>
      let next := accumulator + value
      (next, stalled || (value != 0 && next == accumulator))) (0, false) values
    let ordinary := divSpec values (replicate (Tensor.scalar denominator))
    let (_, flags) := ops.addWithFlags denominator 0
    if !stalled && flags == 0 && denominator != 0 then ordinary
    else
      let repaired : Option (Tensor α [n]) := do
        let zero ← ops.encode (0 : α)
        let total ← foldlSpec (fun accumulated value => do
          let total ← accumulated
          let term ← ops.encode value
          pure (Numeric.QuotientCoefficients.add total term)) (some zero) values
        if total.primal == 0 then none
        else
          pure (Tensor.ofFn fun i =>
            let original := ordinary.getScalar i
            match ops.encode (values.getScalar i) with
            | none => original
            | some value =>
                let coefficients := Numeric.QuotientCoefficients.div value total
                let candidate := (ops.decode coefficients).getD original
                if coefficients.primal == 0 && original == 0 then
                  ops.copyPrimal original candidate
                else candidate)
      repaired.getD ordinary

/-- Softmax on a length-`n` vector.

This is the "real" softmax, not the scalar logistic helper in `Activation.Math.logisticSpec`.

Numerical stability:

We implement the standard stabilized form
$\operatorname{softmax}(x)_i=\exp(x_i-m)/\sum_j\exp(x_j-m)$, where
$m=\max_i x_i$. Subtracting the maximum bounds the exponentials by one. The shared normalizer
also handles an overflowing or stagnating denominator when finite coefficient conversion is
available; shifting alone does not protect that reduction in low precision.
-/
def softmaxVecSpec {n : Nat} (t : Tensor α [n]) : Tensor α [n] :=
  match n with
  | 0 => t
  | Nat.succ _ =>
      Internal.normalizeWeights (maxShiftedExpVecSpec t)

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

/-- Form a centered softmax pullback from finite, unrounded gradient coefficients.

Attention can supply the original contraction instead of encoding an already overflowed tensor.
The caller retains its ordinary result if the original inputs cannot be encoded. -/
@[noinline] def Internal.softmaxPullbackCoefficients? {n : Nat}
    (weights : Tensor α [n])
    (gradient : Fin n →
      Option (Numeric.QuotientCoefficients (Numeric.QuotientArithmetic.depth α)))
    (origin : Option (Numeric.QuotientCoefficients (Numeric.QuotientArithmetic.depth α))) :
    Option (Fin n → Option
      (Numeric.QuotientCoefficients (Numeric.QuotientArithmetic.depth α))) := do
  let ops := (inferInstance : Numeric.QuotientArithmetic α)
  let origin ← origin
  let zero ← ops.encode (0 : α)
  let total ← (List.finRange n).foldl (fun accumulated i => do
    let total ← accumulated
    let weight ← ops.encode (weights.getScalar i)
    let upstream ← gradient i
    pure (Numeric.QuotientCoefficients.add total
      (Numeric.QuotientCoefficients.mul
        (Numeric.QuotientCoefficients.sub upstream origin) weight))) (some zero)
  pure fun i => do
    let weight ← ops.encode (weights.getScalar i)
    let upstream ← gradient i
    let difference := Numeric.QuotientCoefficients.sub
      (Numeric.QuotientCoefficients.sub upstream origin) total
    pure (Numeric.QuotientCoefficients.mul weight difference)

/-- Decode a centered softmax pullback only after combining all finite coefficients. -/
@[noinline] def Internal.replaySoftmaxPullback? {n : Nat}
    (weights ordinary : Tensor α [n])
    (gradient : Fin n →
      Option (Numeric.QuotientCoefficients (Numeric.QuotientArithmetic.depth α)))
    (origin : Option (Numeric.QuotientCoefficients (Numeric.QuotientArithmetic.depth α))) :
    Option (Tensor α [n]) := do
  let ops := (inferInstance : Numeric.QuotientArithmetic α)
  let result ← Internal.softmaxPullbackCoefficients? weights gradient origin
  pure (Tensor.ofFn fun i =>
    match result i with
    | some coefficients =>
        let original := ordinary.getScalar i
        let candidate := (ops.decode coefficients).getD original
        if coefficients.primal == 0 && original == 0 then
          ops.copyPrimal original candidate
        else candidate
    | none => ordinary.getScalar i)

/-- Apply a weighted, centered softmax Jacobian to one row.

Finite input coefficients are replayed only when the ordinary result loses range. In particular,
the difference of two finite incoming gradients may overflow even when every final component
fits. Replaying the whole contraction avoids both that intermediate and spurious `0 * ∞` at
blocked weights. Nonfinite inputs and unsupported carriers retain the ordinary formula.
-/
def Internal.softmaxPullback {n : Nat} (weights gradient : Tensor α [n])
    (anchor : α) : Tensor α [n] :=
  let centered := subSpec gradient (replicate (Tensor.scalar anchor))
  let mean := sumSpec (mulSpec centered weights)
  let ordinary := mulSpec weights (subSpec centered (replicate (Tensor.scalar mean)))
  let ops := (inferInstance : Numeric.QuotientArithmetic α)
  if !ops.supported then ordinary
  else if !(List.finRange n).any (fun i =>
      let (_, flags) := ops.addWithFlags (ordinary.getScalar i) 0
      flags != 0) then ordinary
  else
    (Internal.replaySoftmaxPullback? weights ordinary
      (fun i => ops.encode (gradient.getScalar i)) (ops.encode anchor)).getD ordinary

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

Before taking the weighted mean, subtract the upstream value at a maximal logit from every
upstream coordinate. This leaves the real VJP unchanged, avoids materializing a Jacobian, and
retains small derivatives when the dominant probability has rounded to one. Constant upstream
vectors become exactly zero before either multiplication or reduction.
-/
def Internal.softmaxInnermostBackwardSpec : {s : Shape} → Tensor α s → Tensor α s → Tensor α s
  | .scalar, _x, _dY => Tensor.scalar 0
  | .dim 0 .scalar, _x, dY => dY
  | .dim (Nat.succ n) .scalar, x, dY =>
      let pivot : Fin (Nat.succ n) :=
        Fin.cast (by simp [Shape.size]) (Tensor.argmax (by simp [Shape.size]) x)
      let y := softmaxVecSpec x
      Internal.softmaxPullback y dY (dY.getScalar pivot)
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

/-- Log-softmax on a length-`n` vector.

One maximal entry supplies the unit term of the normalizer. We sum the other exponentials
separately and use `log1pSpec`, so a small tail is not lost in an addition to one. Exactly one
entry is omitted, including when several entries tie for the maximum.
-/
def logSoftmaxVecSpec {n : Nat} (t : Tensor α [n]) : Tensor α [n] :=
  match n with
  | 0 => t
  | Nat.succ n' =>
      let pivot : Fin (Nat.succ n') :=
        Fin.cast (by simp [Shape.size]) (Tensor.argmax (by simp [Shape.size]) t)
      let shifted := subSpec t (replicate (Tensor.scalar (t.getScalar pivot)))
      let tail : Tensor α [Nat.succ n'] := Tensor.ofFn fun i =>
        if i = pivot then 0 else MathFunctions.exp (shifted.getScalar i)
      let logDenom := Math.log1pSpec (sumSpec tail)
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
also avoids recomputing the stable forward pass. The pivot's contribution is separated with
`expm1Spec`, retaining its small complementary probability in forward mode as well.
-/
def Internal.logSoftmaxInnermostJvpSpec : {s : Shape} → Tensor α s → Tensor α s → Tensor α s
  | .scalar, _y, _dx => Tensor.scalar 0
  | .dim 0 .scalar, _y, dx => dx
  | .dim (Nat.succ n) .scalar, y, dx =>
      let pivot : Fin (Nat.succ n) :=
        Fin.cast (by simp [Shape.size]) (Tensor.argmax (by simp [Shape.size]) y)
      let anchor := dx.getScalar pivot
      let others : Tensor α [Nat.succ n] := Tensor.ofFn fun i =>
        if i = pivot then 0 else MathFunctions.exp (y.getScalar i) * dx.getScalar i
      let correction := Math.expm1Spec (y.getScalar pivot) * anchor + sumSpec others
      subSpec (subSpec dx (replicate (Tensor.scalar anchor)))
        (replicate (Tensor.scalar correction))
  | .dim n inner, y, dx =>
      Tensor.dim fun i : Fin n =>
        Internal.logSoftmaxInnermostJvpSpec (s := inner)
          (Tensor.unstack y i) (Tensor.unstack dx i)

/-- Vector log-softmax VJP, keeping the pivot's small complementary probability.

For the largest log-probability, `1 - exp(y)` is evaluated as `-expm1(y)`. Its own upstream
coordinate is separated from the remaining sum before multiplying. Thus a one-hot loss does
not subtract two values that have already rounded to one. The other coordinates use the usual
linear-time VJP. This formula also retains the meaning of arbitrary, unnormalized upstream values.
-/
def Internal.logSoftmaxVecBackwardSpec {n : Nat} (y dY : Tensor α [n]) : Tensor α [n] :=
  match n with
  | 0 => dY
  | Nat.succ n' =>
      let pivot : Fin (Nat.succ n') :=
        Fin.cast (by simp [Shape.size]) (Tensor.argmax (by simp [Shape.size]) y)
      let others : Tensor α [Nat.succ n'] := Tensor.ofFn fun i =>
        if i = pivot then 0 else dY.getScalar i
      let rowSum := sumSpec dY
      let otherSum := sumSpec others
      Tensor.ofFn fun i =>
        let p := MathFunctions.exp (y.getScalar i)
        if i = pivot then
          -(Math.expm1Spec (y.getScalar i)) * dY.getScalar i - p * otherSum
        else dY.getScalar i - p * rowSum

/-- Backward/VJP for last-axis log-softmax.

If $y=\operatorname{logsoftmax}(x)$, then $\operatorname{softmax}(x)=\exp(y)$ and the
vector-Jacobian product is

$$
\frac{\partial L}{\partial x}
=\frac{\partial L}{\partial y}
 -\operatorname{softmax}(x)\sum_i\frac{\partial L}{\partial y_i}.
$$

We take the already-computed output `y` rather than the logits `x`. The vector implementation
rearranges the pivot coordinate to avoid cancellation when its probability is close to one.
-/
def Internal.logSoftmaxInnermostBackwardSpec :
    {s : Shape} → Tensor α s → Tensor α s → Tensor α s
  | .scalar, _y, _dY => Tensor.scalar 0
  | .dim _n .scalar, y, dY => Internal.logSoftmaxVecBackwardSpec y dY
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
  Tensor.map (fun x => Activation.Math.leakyReluSpec x αₗ) t

/-- Tensor-level derivative of leaky ReLU (pointwise). -/
def leakyReluDerivSpec {α : Type} [TorchLean.Storage α] [Zero α] [One α] [LT α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (t : Tensor α s) (αₗ : α) : Tensor α s :=
  Tensor.map (fun x => Activation.Math.leakyReluDerivSpec x αₗ) t

/-- Tensor-level ELU (pointwise).  PyTorch analogy: `torch.nn.functional.elu`. -/
def eluSpec {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (t : Tensor α s) (alpha : α) : Tensor α s :=
  Tensor.map (fun x => Activation.Math.eluSpec x alpha) t

/-- Tensor-level derivative of ELU (pointwise). -/
def eluDerivSpec {α : Type} [TorchLean.Storage α] [Zero α] [One α] [LT α]
    [DecidableRel ((· > ·) : α → α → Prop)] [MathFunctions α] [Mul α]
    {s : Shape} (t : Tensor α s) (alpha : α) : Tensor α s :=
  Tensor.map (fun x => Activation.Math.eluDerivSpec x alpha) t

/-- Tensor-level GELU (approximate, pointwise). PyTorch analogy: `gelu(..., approximate="tanh")`. -/
def geluSpec {α : Type} [TorchLean.Storage α] [Context α] {s : Shape} (t : Tensor α s) :
    Tensor α s :=
  Tensor.map Activation.Math.geluSpec t

/-- Tensor-level derivative of tanh-approx GELU (pointwise). -/
def geluDerivSpec {α : Type} [TorchLean.Storage α] [Context α] {s : Shape} (t : Tensor α s) :
    Tensor α s :=
  Tensor.map Activation.Math.geluDerivSpec t

/-- Tensor-level Swish / SiLU (pointwise). -/
def swishSpec {s : Shape} (t : Tensor α s) : Tensor α s :=
  Tensor.map Activation.Math.swishSpec t

/-- Tensor-level derivative of Swish / SiLU (pointwise). -/
def swishDerivSpec {s : Shape} (t : Tensor α s) : Tensor α s :=
  Tensor.map Activation.Math.swishDerivSpec t

/-- Tensor-level softplus (pointwise). -/
def softplusSpec {s : Shape} (t : Tensor α s) : Tensor α s :=
  Tensor.map Activation.Math.softplusSpec t

/-- Tensor-level derivative of softplus (pointwise). -/
def softplusDerivSpec {s : Shape} (t : Tensor α s) : Tensor α s :=
  Tensor.map Activation.Math.softplusDerivSpec t

/-- Tensor-level `safeLogSpec` (pointwise). -/
def safeLogSpec {s : Shape} (t : Tensor α s) (ε : α := Context.defaultEpsilon) : Tensor α s :=
  Tensor.map (fun x => Activation.Math.safeLogSpec (α := α) x ε) t

/-- Tensor-level derivative of `safeLogSpec` (pointwise). -/
def safeLogDerivSpec {s : Shape} (t : Tensor α s) (ε : α := Context.defaultEpsilon) : Tensor α s :=
  Tensor.map (fun x => Activation.Math.safeLogDerivSpec (α := α) x ε) t

/-- Tensor-level `smoothAbsSpec` (pointwise). -/
def smoothAbsSpec {s : Shape} (t : Tensor α s) (ε : α := Context.defaultEpsilon) : Tensor α s :=
  Tensor.map (fun x => Activation.Math.smoothAbsSpec (α := α) x ε) t

/-- Tensor-level derivative of `smoothAbsSpec` (pointwise). -/
def smoothAbsDerivSpec {s : Shape} (t : Tensor α s) (ε : α := Context.defaultEpsilon) :
    Tensor α s :=
  Tensor.map (fun x => Activation.Math.smoothAbsDerivSpec (α := α) x ε) t

end Activation

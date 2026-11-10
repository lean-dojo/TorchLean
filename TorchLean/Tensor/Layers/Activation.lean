/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.ArgMax
public import TorchLean.Tensor.Fiber
public import TorchLean.Tensor.AdjacentSwap

/-!
# Activation Specifications

Scalar activation functions and their chosen derivatives live in `TorchLean.Activation`. The sigmoid
and `tanh` are the scalar operations `Context.sigmoid` and `MathFunctions.tanh`, with slopes
`TorchLean.Numeric.sigmoidDeriv` and `TorchLean.Numeric.tanhDeriv`. Tensor operations map those
definitions pointwise, except for shape-dependent operations such as softmax and log-softmax. The
definitions are polymorphic over the scalar `Context`, so one layer specification can be read over
runtime floats, exact scalars or verification domains.

We wrote each activation by hand from its paper's formula, at torch's documented semantics. Two
torch parameters with defaults are plain arguments here, with none: the slope of `leakyReLU`
(torch's `negative_slope=0.01`) and the `alpha` of `elu` (torch's `1.0`). Over ℝ `softplus` is
torch's `log (1 + exp x)` at `beta=1`. For `x > 0` we evaluate `x + log1p (exp (-x))`, and the two
can round differently: with glibc's `exp` and `log1p` they differ in the last place at `x = 0.01` in
binary64. We keep no `threshold`. Above `x = 20` torch returns `x`, so in binary64 `softplus 25` is
`25.000000000013888` where torch gives `25`. In binary32 the tail is under half an ulp of every
`x > 20`, and the two agree. Softmax is eq. (1.4) of Blanchard, Higham and Higham with the shift
`a = max x`. Log-softmax subtracts their log-sum-exp (4.1), which takes `log1p` of the shifted
exponentials with the maximum left out. Barron, Gnanapragasam and Tschannen's eq. (5) also leaves
the maximum out but applies a softplus to the log-sum-exp of the others, which matches ours over ℝ
at a finite maximum.

Two checks in `TorchLeanTest/Interop/IRPyTorch.lean` run the module that `TorchLean.PyTorch.emit`
writes. In binary64, `torch.nn.functional.gelu` with `approximate='tanh'` gives the words of
`geluTanh` at `±0`, a NaN at a NaN, and values within a relative `1e-12` of ours at `±1` and `±30`.
In binary64 and binary32, `torch.relu` gives our words at `+0`, `±1` and `±∞` and a NaN at a NaN.
At `-0` it keeps the sign, so the emitted module adds `+0.0` to reach the `+0` of `relu`.

The formulas and conventions follow these references:

- PyTorch 2.8 docs: `torch.nn.functional` (https://docs.pytorch.org/docs/2.8/nn.functional.html)
  and `torch.softmax` (https://docs.pytorch.org/docs/2.8/generated/torch.softmax.html)
- Blanchard, Higham and Higham, "Accurate Computation of the Log-Sum-Exp and Softmax Functions"
  (2019, arXiv:1909.03469), Sec. 1 and 4, Algorithm 4.1: https://arxiv.org/abs/1909.03469
- Barron, Gnanapragasam and Tschannen, "A Fix for Softmax and Cross-Entropy" (2026,
  arXiv:2610.31642), Sec. 2: https://arxiv.org/abs/2610.31642
- ReLU: Vinod Nair and Geoffrey Hinton, "Rectified Linear Units Improve Restricted Boltzmann
  Machines", Proc. 27th ICML, 2010
- ELU: Djork-Arné Clevert, Thomas Unterthiner and Sepp Hochreiter, "Fast and Accurate Deep Network
  Learning by Exponential Linear Units (ELUs)", ICLR 2016, arXiv:1511.07289
- GELU: Dan Hendrycks and Kevin Gimpel, "Gaussian Error Linear Units (GELUs)" (2016,
  arXiv:1606.08415), Sec. 2, which also names the SiLU `x σ(x)`
- Swish / SiLU: Prajit Ramachandran, Barret Zoph and Quoc V. Le, "Searching for Activation
  Functions" (2017, arXiv:1710.05941), and Stefan Elfwing, Eiji Uchibe and Kenji Doya,
  "Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement
  Learning" (2017, arXiv:1702.03118)
-/

@[expose] public section


open TorchLean.Shape
open TorchLean.Tensor
open FloatLib.Numerics (MathFunctions)

namespace TorchLean.Activation

/-- Activation functions with a parameter-free pointwise interpretation.

Model specifications and the public model builders share this type. Keeping the choice in the
specification layer stops a configuration string from silently selecting the wrong semantics.
-/
inductive Kind where
  /-- Rectified linear unit, `max(0, x)`. -/
  | relu
  /-- GELU in its tanh approximation (`geluTanh`), which is PyTorch's
  `nn.GELU(approximate='tanh')`. The default `nn.GELU()` uses erf. -/
  | geluTanh
  /-- SiLU/Swish, `x * sigmoid(x)`. -/
  | silu
  /-- Hyperbolic tangent. -/
  | tanh
  /-- Logistic sigmoid. -/
  | sigmoid
deriving Repr, DecidableEq

variable {α : Type} [Storage α] [Context α]

/-! ## Scalar activations -/

/-- ReLU: $\operatorname{ReLU}(x)=\max(x,0)$.

PyTorch analogy: `torch.nn.functional.relu`.

It needs only zero, `max` and equality, so it makes sense over many scalar backends, including
ones without `exp` or `log`.

An input equal to zero returns the scalar zero. That matters for dual numbers: their equality
compares primals, so the branch clears every tangent at the kink, matching `reluDeriv`.
Elementwise maximum has its own convention and splits a tie equally. Every other input gets the
backend's `max x 0`, including a floating-point NaN, which does not compare equal to zero.
-/
def relu {α : Type} [Zero α] [Max α] [BEq α] (x : α) : α :=
  if x == 0 then 0 else Max.max x 0

/-- The zero branch of `relu` agrees with `max x 0` on a linearly ordered scalar. -/
@[simp] theorem relu_zero_branch_eq_max {α : Type} [LinearOrder α] [Zero α] (x : α) :
    (if x == 0 then (0 : α) else max x 0) = max x 0 := by
  by_cases hx : x = 0 <;> simp [hx]

/-- On an ordered scalar with ordinary equality, ReLU is the usual maximum with zero. -/
@[simp] theorem relu_eq_max {α : Type} [LinearOrder α] [Zero α] (x : α) :
    relu x = max x 0 :=
  relu_zero_branch_eq_max x

/-- A standard subgradient choice for ReLU:

$\frac{d}{dx}\operatorname{ReLU}(x)=1$ if $x>0$, and $0$ otherwise.

PyTorch's autograd also picks a subgradient at $x=0$. The spec commits to this one, so "the
derivative" is a pure function.
-/
def reluDeriv {α : Type} [Zero α] [One α] [LT α] [DecidableRel ((· > ·) : α → α → Prop)] (x :
  α) : α :=
  if x > 0 then 1 else 0

/-- Leaky ReLU:

$\operatorname{leaky\_relu}(x;\alpha)=x$ if $x>0$, else $\alpha x$.

PyTorch analogy: `torch.nn.functional.leaky_relu` with `negative_slope = α`.
-/
def leakyReLU {α : Type} [Zero α] [Mul α] [LT α] [DecidableRel ((· > ·) : α → α → Prop)] (x :
  α) (αₗ : α) : α :=
  if x > 0 then x else αₗ * x

/-- Derivative of leaky ReLU:

$\frac{d}{dx}\operatorname{leaky\_relu}(x;\alpha)=1$ if $x>0$, else $\alpha$.
-/
def leakyReLUDeriv {α : Type} [Zero α] [One α] [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
  (x : α) (αₗ : α) : α :=
  if x > 0 then 1 else αₗ

/-- ELU (Exponential Linear Unit):

$\operatorname{ELU}(x;\alpha)=x$ if $x>0$, else $\alpha(\exp(x)-1)$.

PyTorch analogy: `torch.nn.functional.elu` with `alpha = α`.

The negative branch uses the scalar context's `expm1`, so small inputs do not disappear when
`exp x` rounds to one. Differentiation carriers supply its chain rule.
-/
def elu {α : Type} [Context α] (x : α) (alpha : α) : α :=
  if x > 0 then x else alpha * Context.expm1 x

/-- Derivative of ELU:

$\operatorname{ELU}'(x;\alpha)=1$ if $x>0$, else $\alpha\exp(x)$.
-/
def eluDeriv {α : Type} [Zero α] [One α] [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
  [MathFunctions α] [Mul α] (x : α) (alpha : α) : α :=
  if x > 0 then 1 else alpha * MathFunctions.exp x

/-- The constant `√(2 / π)` of the tanh GELU, computed in the scalar type: exact over the reals,
and in a binary format the square root of the rounded quotient of `2` by the rounded `π`. In
binary64 that is the binary64 value nearest `√(2 / π)`, and in binary32 it is one unit in the last
place below it. -/
def geluBeta {α : Type} [Context α] : α :=
  MathFunctions.sqrt (2 / MathFunctions.pi)

/-- The constant `0.044715` of the tanh GELU, the exact fraction rounded once. Casting the fraction
once avoids overflowing its denominator in small floating-point formats. -/
def geluKappa {α : Type} [RatCast α] : α :=
  Rat.cast (44715 / 1000000)

/-- GELU in its tanh approximation, `torch.nn.functional.gelu(x, approximate="tanh")`:
`0.5 x (1 + tanh (β (x + κ x³)))` with `β = geluBeta` and `κ = geluKappa`, one operation after
another in the order of PyTorch's CPU kernel at capability level `DEFAULT`.

PyTorch's default `nn.GELU()` is the exact erf form, `Tensor.gelu`. Over the reals this is the tanh
approximation of Hendrycks and Gimpel with its constants `√(2 / π)` and `0.044715`
(`TorchLean.Activation.geluTanh_real`). PyTorch rounds `√(2 / π)` once to binary64 and then to the
computation type. Once `tanh` rounds to `-1`, a finite negative input gives `-0`, as PyTorch's
order does. At `±∞` the result is the limit, `+∞` or `-0`. At `-∞` PyTorch's order computes
`-∞ · 0`, a NaN.
-/
def geluTanh {α : Type} [Storage α] [Context α] (x : α) : α :=
  if Context.isInf x then (if x > 0 then x else -0) else
  let inner := geluBeta * (x + geluKappa * (x * x * x))
  (Rat.cast (1 / 2) : α) * x * (1 + MathFunctions.tanh inner)

/-- The derivative of `geluTanh`, `(1 + tanh u + x (1 - tanh² u) u') / 2` with
`u = β (x + κ x³)`: the gate `1 + tanh u` is evaluated as `2 sigmoid (2 u)`, and the second term
is dropped where `TorchLean.Numeric.tanhDeriv u` is zero. At `±∞` it is the limit, `1` or
`-0`, which the slope approaches from below at `-∞`. -/
def geluTanhDeriv {α : Type} [Storage α] [Context α] (x : α) : α :=
  if Context.isInf x then (if x > 0 then 1 else -0) else
  let two : α := (1 : α) + (1 : α)
  let three : α := (1 : α) + (1 : α) + (1 : α)
  let inner := geluBeta * (x + geluKappa * (x * x * x))
  let sechTerm := Numeric.tanhDeriv inner
  let innerDeriv := geluBeta * ((1 : α) + three * geluKappa * x * x)
  let correction := if sechTerm == 0 then 0 else x * sechTerm * innerDeriv
  (two * Context.sigmoid (inner + inner) + correction) / two

/-- Swish / SiLU:

$\operatorname{swish}(x)=x\operatorname{sigmoid}(x)$.

PyTorch analogy: `torch.nn.functional.silu`.

The negative branch splits the exponential as `r * r` with `r = exp (x / 2)` and multiplies `x` by
one factor at a time. The product can then be representable where `exp x`, and with it the
sigmoid, underflows. At `±∞` the result is the limit, `+∞` or `-0`. At `-∞` the negative branch
would compute `-∞ · 0`, a NaN.
-/
def silu (x : α) : α :=
  if Context.isInf x then (if x > 0 then x else -0) else
  if x > 0 then x * Context.sigmoid x
  else
    let r := MathFunctions.exp (x / 2)
    ((x * r) * r) / (1 + r * r)

/-- Derivative of Swish / SiLU.

The negative branch forms `x + denominator` before multiplying by the split exponential `r * r`,
because forming sigmoid and its slope separately could underflow both before the product with
`x`. At `±∞` the slope is the limit, `1` or `-0` (approached from below at `-∞`). At either
infinity its branch would compute `∞ · 0`, a NaN.
-/
def siluDeriv (x : α) : α :=
  if Context.isInf x then (if x > 0 then 1 else -0) else
  if x > 0 then Context.sigmoid x + x * Numeric.sigmoidDeriv x
  else
    let r := MathFunctions.exp (x / 2)
    let denominator := 1 + r * r
    (((x + denominator) * r) * r) / (denominator * denominator)

/-- Softplus, evaluated without a large positive exponential:

$\operatorname{softplus}(x)=\log(1+\exp(x))$.

The positive branch uses the equivalent expression $x+\log(1+\exp(-x))$, which keeps a finite
floating-point input finite when `exp(x)` would overflow. TorchLean uses this one-argument,
unit-scale softplus throughout.

PyTorch analogy: `torch.nn.functional.softplus`.
-/
def softplus (x : α) : α :=
  if x > 0 then
    x + Context.log1p (MathFunctions.exp (-x))
  else
    Context.log1p (MathFunctions.exp x)

/-- A smooth log surrogate:

$\operatorname{safe\_log}(x;\varepsilon)=\log(\operatorname{softplus}(x)+\varepsilon)$.

For `ε > 0` the argument of the logarithm is positive at every `x`, so a log-like term needs no
side condition that its input be strictly positive.
-/
def safeLog (x : α) (ε : α := Context.defaultEpsilon) : α :=
  MathFunctions.log (softplus x + ε)

/-- Derivative of `safeLog`. -/
def safeLogDeriv (x : α) (ε : α := Context.defaultEpsilon) : α :=
  Context.sigmoid x / (softplus x + ε)

/-- A smooth absolute value surrogate:

$\operatorname{smooth\_abs}(x;\varepsilon)=\sqrt{x^2+\varepsilon}$.

For `ε > 0` it has the shape of `abs` and is differentiable at `0`. We factor out the scale
`max |x| (sqrt |ε|)` before squaring, so a large finite `x` does not overflow `x * x`.
-/
def smoothAbs (x : α) (ε : α := Context.defaultEpsilon) : α :=
  let scale := max (MathFunctions.abs x) (MathFunctions.sqrt (MathFunctions.abs ε))
  if scale == 0 || !(scale / scale == scale / scale) then
    MathFunctions.sqrt (x * x + ε)
  else
    let normalized := x / scale
    scale * MathFunctions.sqrt (normalized * normalized + (ε / scale) / scale)

/-- Derivative of `smoothAbs`. -/
def smoothAbsDeriv (x : α) (ε : α := Context.defaultEpsilon) : α :=
  x / smoothAbs x ε

end TorchLean.Activation

namespace TorchLean.Tensor

/-- Rectified linear activation at every coordinate (`torch.relu`), `Activation.relu`. -/
def relu {α : Type} [TorchLean.Storage α] [Zero α] [Max α] [BEq α]
    {s : Shape} (t : Tensor α s) : Tensor α s :=
  map Activation.relu t

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- The logistic sigmoid at every coordinate (`torch.sigmoid`), `Context.sigmoid`. -/
def sigmoid {s : Shape} (t : Tensor α s) : Tensor α s :=
  map Context.sigmoid t

/-- The hyperbolic tangent at every coordinate (`torch.tanh`), `MathFunctions.tanh`. -/
def tanh {s : Shape} (t : Tensor α s) : Tensor α s :=
  map MathFunctions.tanh t

end TorchLean.Tensor

namespace TorchLean

variable {α : Type} [Storage α] [Context α]

/-- Sigmoid's pointwise pullback, `TorchLean.Numeric.sigmoidBackward` at every coordinate: the
slope at `x` times the incoming gradient `seed`, without first losing the slope's small tail. -/
def Tensor.sigmoidBackward {s : Shape} (x seed : Tensor α s) : Tensor α s :=
  zipWith Numeric.sigmoidBackward x seed

/-- Tanh's pointwise pullback, `TorchLean.Numeric.tanhBackward` at every coordinate: the slope
at `x` times the incoming gradient `seed`, with intermediate range loss repaired. -/
def Tensor.tanhBackward {s : Shape} (x seed : Tensor α s) : Tensor α s :=
  zipWith Numeric.tanhBackward x seed

/-- Apply a parameter-free pointwise activation to a tensor. -/
def Activation.Kind.apply {s : Shape} : Kind → Tensor α s → Tensor α s
  | .relu => Tensor.relu
  | .geluTanh => Tensor.map Activation.geluTanh
  | .silu => Tensor.map Activation.silu
  | .tanh => Tensor.tanh
  | .sigmoid => Tensor.sigmoid

/-- Apply the derivative selected by a parameter-free pointwise activation. -/
def Activation.Kind.deriv {s : Shape} : Kind → Tensor α s → Tensor α s
  | .relu => Tensor.map reluDeriv
  | .geluTanh => Tensor.map geluTanhDeriv
  | .silu => Tensor.map siluDeriv
  | .tanh => Tensor.map Numeric.tanhDeriv
  | .sigmoid => Tensor.map Numeric.sigmoidDeriv

/-!
## Softmax on tensors

These are the kernels behind `TorchLean.Tensor.softmax` and `TorchLean.Tensor.logSoftmax`
(`TorchLean.Tensor.Layers.Softmax`): the vector kernels act on one fiber, and the last-axis
kernels recurse over the outer dimensions and apply them to every row.
-/

/-- The index of the first strict maximum of a vector: log-softmax and `logsumexp` shift by its
entry, and the softmax VJP and the log-softmax JVP and VJP single out its coordinate. One left fold
from index `0` moves to a strictly larger entry, so a NaN at index `0` is kept and a later NaN is
passed over. On a vector whose entries equal themselves it is `Tensor.argmax`
(`Softmax.pivot_eq_argmax`). -/
def Softmax.pivot {n : Nat} (t : Tensor α [n]) (h : 0 < n := by simp) : Fin n :=
  (List.finRange n).foldl (fun best j => if t.getScalar j > t.getScalar best then j else best)
    ⟨0, h⟩

/-- On a vector whose entries equal themselves, `Softmax.pivot` is `Tensor.argmax`. -/
theorem Softmax.pivot_eq_argmax {n : Nat} (t : Tensor α [n]) (h : 0 < n)
    (hself : ∀ j, (t.getScalar j == t.getScalar j) = true) :
    Softmax.pivot t h = (Tensor.argmax t ⟨0, Nat.one_pos⟩ false h).item := by
  rw [Tensor.item_argmax_vector, Softmax.pivot]
  simp only [hself, Bool.not_true, Bool.false_or, Bool.true_and, decide_eq_true_eq]

/-- The maximum of a nonempty vector as a scalar tensor: the fold of the IEEE `Context.maximum`
seeded by the first coordinate, which is `reduce .max` over the one axis
(`Softmax.maxVec_eq_reduce`). The stable softmax kernel `vec` shifts by it,
while log-softmax and `logsumexp` shift by the entry at `pivot`. -/
def Softmax.maxVec {n : Nat} (t : Tensor α [Nat.succ n]) : Tensor α .scalar :=
  let first : α := t.getScalar ⟨0, Nat.succ_pos n⟩
  let maximum : α :=
    (List.finRange (Nat.succ n)).foldl
      (fun acc i => Context.maximum acc (t.getScalar i))
      first
  Tensor.scalar maximum

/-- The exponentials of a vector shifted by its maximum `maxVec`, the weights that the stable
softmax kernel `vec` normalizes. -/
def Softmax.maxShiftedExpVec {n : Nat}
    (t : Tensor α [Nat.succ n]) : Tensor α [Nat.succ n] :=
  Tensor.exp (t - Tensor.full _ (maxVec t).item)

/-- Normalize finite weights without requiring their sum to fit in the output format.

We keep the ordinary left fold unless it loses range or swallows a nonzero term. In that case the
whole sum and the quotients are replayed over finite rational coefficient trees before each output
is rounded, which also keeps the derivatives that nested scalars carry. Nonfinite inputs and
carriers without coefficient conversion use the ordinary formula, and a sum that is exactly zero
is not repaired. Information lost in forming the weights cannot be recovered here.
-/
def Softmax.normalizeWeights {n : Nat} (values : Tensor α [n]) : Tensor α [n] :=
  let ops := (inferInstance : Numeric.QuotientArithmetic α)
  if !ops.supported then
    values / Tensor.full _ (Tensor.sum values)
  else
    let (denominator, stalled) := Tensor.foldl (fun (accumulator, stalled) value =>
      let next := accumulator + value
      (next, stalled || (value != 0 && next == accumulator))) (0, false) values
    let ordinary := values / Tensor.full _ denominator
    let (_, flags) := ops.addWithFlags denominator 0
    if !stalled && flags == 0 && denominator != 0 then ordinary
    else
      let repaired : Option (Tensor α [n]) := do
        let zero ← ops.encode (0 : α)
        let total ← Tensor.foldl (fun accumulated value => do
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

/-- Softmax of one fiber, a length-`n` vector: the kernel that `Tensor.softmax` applies along
its axis.

Numerical stability:

The definition is the standard stabilized form
$\operatorname{softmax}(x)_i=\exp(x_i-m)/\sum_j\exp(x_j-m)$, where $m=\max_i x_i$. Subtracting the
maximum bounds the exponentials by one, but in low precision their sum can overflow or stop growing.
When finite coefficient conversion is available, the shared normalizer `Softmax.normalizeWeights`
repairs that denominator too.
-/
def Softmax.vec {n : Nat} (t : Tensor α [n]) : Tensor α [n] :=
  match n with
  | 0 => t
  | Nat.succ _ =>
      Softmax.normalizeWeights (maxShiftedExpVec t)

/-- Softmax along the last axis (recurses over outer dimensions).

PyTorch analogy: `torch.softmax(x, dim=-1)`.

For `s = .scalar` we return `1` (there is only one coordinate). For higher-rank tensors we keep
the outer structure and apply `Softmax.vec` at the last axis.
-/
def Softmax.innermost : {s : Shape} → Tensor α s → Tensor α s :=
  Tensor.lastAxisMap 1 fun _ => Softmax.vec

/-- Last-axis softmax at a coordinate: the softmax of the row through it, `1` on a scalar. -/
theorem Softmax.innermost_apply {s : Shape} (x : Tensor α s) (p : s.Coord) :
    Softmax.innermost x p =
      if s = .scalar then 1
      else (Softmax.vec (x.row p)).getScalar (Shape.Coord.getLast p) :=
  Tensor.lastAxisMap_apply 1 (fun _ => Softmax.vec) x p

/-- Last-axis softmax on a matrix acts independently on each row. -/
@[simp] theorem Softmax.unstack_innermost_matrix {rows columns : Nat}
    (tensor : Tensor α [rows, columns]) (row : Fin rows) :
    Tensor.unstack (innermost tensor) row =
      vec (Tensor.unstack tensor row) := by
  rw [show innermost tensor =
    Tensor.dim (fun i => vec (Tensor.unstack tensor i)) by rfl]
  simp

/-- Form a centered softmax pullback from finite, unrounded gradient coefficients.

Attention can pass the coefficients of its original contraction, so an overflowed tensor need not
be encoded. If the original inputs cannot be encoded, the result is `none` and the caller keeps
its ordinary result. -/
@[noinline] def Softmax.pullbackCoefficients? {n : Nat}
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
@[noinline] def Softmax.replayPullback? {n : Nat}
    (weights ordinary : Tensor α [n])
    (gradient : Fin n →
      Option (Numeric.QuotientCoefficients (Numeric.QuotientArithmetic.depth α)))
    (origin : Option (Numeric.QuotientCoefficients (Numeric.QuotientArithmetic.depth α))) :
    Option (Tensor α [n]) := do
  let ops := (inferInstance : Numeric.QuotientArithmetic α)
  let result ← Softmax.pullbackCoefficients? weights gradient origin
  pure (Tensor.ofFn fun i =>
    match result i with
    | some coefficients =>
        let original := ordinary.getScalar i
        let candidate := (ops.decode coefficients).getD original
        if coefficients.primal == 0 && original == 0 then
          ops.copyPrimal original candidate
        else candidate
    | none => ordinary.getScalar i)

/-- The softmax pullback of one row, `weights ⊙ (c - ⟨c, weights⟩)` with the centered gradient
`c = gradient - anchor`.

We replay the finite input coefficients only when this ordinary result loses range. The difference
of two finite incoming gradients can overflow even when every final component fits, and replaying
the whole contraction avoids that intermediate as well as a spurious `0 * ∞` at a blocked weight.
Nonfinite inputs and unsupported carriers use the ordinary formula.
-/
def Softmax.pullback {n : Nat} (weights gradient : Tensor α [n])
    (anchor : α) : Tensor α [n] :=
  let centered := gradient - Tensor.full _ anchor
  let mean := Tensor.sum (centered * weights)
  let ordinary := weights * (centered - Tensor.full _ mean)
  let ops := (inferInstance : Numeric.QuotientArithmetic α)
  if !ops.supported then ordinary
  else if !(List.finRange n).any (fun i =>
      let (_, flags) := ops.addWithFlags (ordinary.getScalar i) 0
      flags != 0) then ordinary
  else
    (Softmax.replayPullback? weights ordinary
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

Before taking the weighted mean we subtract the upstream value at a maximal logit
(`Softmax.pivot`) from every upstream coordinate. Over the reals the VJP is unchanged, because
the probabilities sum to one, and small derivatives survive when the dominant probability has
rounded to one. No Jacobian is materialized, and a constant upstream vector becomes exactly zero
before any multiplication or reduction.
-/
def Softmax.innermostBackward : {s : Shape} → Tensor α s → Tensor α s → Tensor α s
  | .scalar, _x, _dY => Tensor.scalar 0
  | .dim 0 .scalar, _x, dY => dY
  | .dim (Nat.succ n) .scalar, x, dY =>
      let pivot : Fin (Nat.succ n) := Softmax.pivot x
      let y := Softmax.vec x
      Softmax.pullback y dY (dY.getScalar pivot)
  | .dim n inner, x, dY =>
      Tensor.dim fun i : Fin n =>
        innermostBackward (s := inner)
          (Tensor.unstack x i) (Tensor.unstack dY i)

/-- The last-axis softmax VJP at a coordinate: the VJP of the rows through it, `0` on a scalar. -/
theorem Softmax.innermostBackward_apply {s : Shape} (x dY : Tensor α s)
    (p : s.Coord) :
    Softmax.innermostBackward x dY p =
      if s = .scalar then 0
      else (Softmax.innermostBackward (x.row p) (dY.row p)).getScalar
        (Shape.Coord.getLast p) :=
  rowwise_apply Softmax.innermostBackward 0 (fun _ _ => rfl) (fun _ _ => rfl) x dY p

/-
## Log-softmax (stable)

`logSoftmax` is often the numerically preferred form for cross-entropy on logits:

`CE(p, logits) = -sum_i p_i * log_softmax(logits)_i`.

The definition uses the max-shift trick of `Softmax.vec`, shifting by the entry at the
first strict maximum `Softmax.pivot`, and computes the log-probabilities from the shifted logits,
so an underflowing `exp` never leads to `log(0)`.
-/

/-- Log-softmax of one fiber, a length-`n` vector: the kernel that `Tensor.logSoftmax` applies
along its axis.

One maximal entry, the pivot, supplies the unit term of the normalizer. We sum the exponentials
of the other shifted entries and apply `Context.log1p`, so a small tail is not lost in an addition
to one, and exactly one entry is left out even when several tie for the maximum. An infinite pivot
makes the normalizer the pivot's own shift `∞ - ∞`, a NaN in a binary format, so every entry is a
NaN, as in PyTorch. Otherwise the normalizer is the `log1p` of the tail.
-/
def LogSoftmax.vec {n : Nat} (t : Tensor α [n]) : Tensor α [n] :=
  match n with
  | 0 => t
  | Nat.succ n' =>
      let pivot : Fin (Nat.succ n') := Softmax.pivot t
      let shifted := t - Tensor.full _ (t.getScalar pivot)
      let tail : Tensor α [Nat.succ n'] := Tensor.ofFn fun i =>
        if i = pivot then 0 else MathFunctions.exp (shifted.getScalar i)
      let logDenom := if Context.isInf (t.getScalar pivot) then shifted.getScalar pivot
        else Context.log1p (Tensor.sum tail)
      shifted - Tensor.full _ logDenom

/-- Log-softmax along the last axis (recurses over outer dimensions). -/
def LogSoftmax.innermost : {s : Shape} → Tensor α s → Tensor α s :=
  Tensor.lastAxisMap 0 fun _ => LogSoftmax.vec

/-- Last-axis log-softmax at a coordinate: the log-softmax of the row through it, `0` on a
scalar. -/
theorem LogSoftmax.innermost_apply {s : Shape} (x : Tensor α s) (p : s.Coord) :
    LogSoftmax.innermost x p =
      if s = .scalar then 0
      else (LogSoftmax.vec (x.row p)).getScalar (Shape.Coord.getLast p) :=
  Tensor.lastAxisMap_apply 0 (fun _ => LogSoftmax.vec) x p

/-- Forward-mode JVP for last-axis log-softmax.

If $y=\operatorname{logsoftmax}(x)$, then each last-axis slice has directional derivative

$dy=dx-\langle\exp(y),dx\rangle\,\mathbf{1}$.

The subtracted scalar is the same at every coordinate of the slice, so the softmax probabilities
occur only inside the dot product. Taking the computed output `y` avoids running the stable
forward pass again. We split off the pivot's term with `Context.expm1`, which keeps its small
complementary probability in forward mode as well.
-/
def LogSoftmax.innermostJVP : {s : Shape} → Tensor α s → Tensor α s → Tensor α s
  | .scalar, _y, _dx => Tensor.scalar 0
  | .dim 0 .scalar, _y, dx => dx
  | .dim (Nat.succ n) .scalar, y, dx =>
      let pivot : Fin (Nat.succ n) := Softmax.pivot y
      let anchor := dx.getScalar pivot
      let others : Tensor α [Nat.succ n] := Tensor.ofFn fun i =>
        if i = pivot then 0 else MathFunctions.exp (y.getScalar i) * dx.getScalar i
      let correction := Context.expm1 (y.getScalar pivot) * anchor + Tensor.sum others
      (dx - Tensor.full _ anchor) -
        Tensor.full _ correction
  | .dim n inner, y, dx =>
      Tensor.dim fun i : Fin n =>
        innermostJVP (s := inner)
          (Tensor.unstack y i) (Tensor.unstack dx i)

/-- The last-axis log-softmax JVP at a coordinate: the JVP of the rows through it, `0` on a
scalar. -/
theorem LogSoftmax.innermostJVP_apply {s : Shape} (y dx : Tensor α s)
    (p : s.Coord) :
    LogSoftmax.innermostJVP y dx p =
      if s = .scalar then 0
      else (LogSoftmax.innermostJVP (y.row p) (dx.row p)).getScalar
        (Shape.Coord.getLast p) :=
  rowwise_apply LogSoftmax.innermostJVP 0 (fun _ _ => rfl) (fun _ _ => rfl) y dx p

/-- Vector log-softmax VJP, keeping the pivot's small complementary probability.

At the largest log-probability we evaluate `1 - exp(y)` as `-expm1(y)` and split that
coordinate's own upstream value from the sum of the others before multiplying, so a one-hot loss
never subtracts two values that have already rounded to one. The other coordinates use the usual
linear-time VJP, and the formula holds for any upstream vector, normalized or not.
-/
def LogSoftmax.vecBackward {n : Nat} (y dY : Tensor α [n]) : Tensor α [n] :=
  match n with
  | 0 => dY
  | Nat.succ n' =>
      let pivot : Fin (Nat.succ n') := Softmax.pivot y
      let others : Tensor α [Nat.succ n'] := Tensor.ofFn fun i =>
        if i = pivot then 0 else dY.getScalar i
      let rowSum := Tensor.sum dY
      let otherSum := Tensor.sum others
      Tensor.ofFn fun i =>
        let p := MathFunctions.exp (y.getScalar i)
        if i = pivot then
          -(Context.expm1 (y.getScalar i)) * dY.getScalar i - p * otherSum
        else dY.getScalar i - p * rowSum

/-- Backward/VJP for last-axis log-softmax.

If $y=\operatorname{logsoftmax}(x)$, then $\operatorname{softmax}(x)=\exp(y)$ and the
vector-Jacobian product is

$$
\frac{\partial L}{\partial x}
=\frac{\partial L}{\partial y}
 -\operatorname{softmax}(x)\sum_i\frac{\partial L}{\partial y_i}.
$$

The first argument is the output `y`, which the forward pass has already computed. The vector
kernel `LogSoftmax.vecBackward` rearranges the pivot coordinate to avoid
cancellation when its probability is close to one.
-/
def LogSoftmax.innermostBackward :
    {s : Shape} → Tensor α s → Tensor α s → Tensor α s
  | .scalar, _y, _dY => Tensor.scalar 0
  | .dim _n .scalar, y, dY => LogSoftmax.vecBackward y dY
  | .dim n inner, y, dY =>
      Tensor.dim fun i : Fin n =>
        innermostBackward (s := inner)
          (Tensor.unstack y i) (Tensor.unstack dY i)

/-- The last-axis log-softmax VJP at a coordinate: the VJP of the rows through it, `0` on a
scalar. -/
theorem LogSoftmax.innermostBackward_apply {s : Shape} (y dY : Tensor α s)
    (p : s.Coord) :
    LogSoftmax.innermostBackward y dY p =
      if s = .scalar then 0
      else (LogSoftmax.vecBackward (y.row p) (dY.row p)).getScalar
        (Shape.Coord.getLast p) :=
  rowwise_apply LogSoftmax.innermostBackward 0 (fun _ _ => rfl) (fun _ _ => rfl)
    y dY p

end TorchLean

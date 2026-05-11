/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Core
public import NN.Spec.Core.Context
public import NN.Spec.Core.Tensor

/-!
# Supplemental Activation Operators

This file defines additional activation functions and transfer rules intended for CROWN/IBP-style
bound propagation:

- Leaky ReLU: `f(x) = max(αx, x)` (typically `0 < α < 1`)
- ELU: `f(x) = x` if `x > 0`, else `α (exp(x) - 1)`
- Softplus: `f(x) = log(1 + exp(x))`
- Mish: `f(x) = x * tanh(softplus(x))`
- SiLU/Swish: `f(x) = x * sigmoid(x)`

## Soundness status

A few helpers in this file use low-order Taylor approximations (`exp_approx`, `log_approx`,
`tanh_approx`, `sigmoid_approx`). These are executable approximations, not theorem-backed
enclosures of the true transcendental functions. Bounds derived from them should only be used under
an explicit checker/oracle assumption, not as part of the unconditional CROWN soundness core.

If you need enclosure-sound bounds for transcendentals, prefer validated enclosure backends (e.g.
Arb-backed intervals in `NN/Floats`) or restrict to the operator subset covered by the main
soundness theorems.

## References

Activations:
- ELU: Clevert, Unterthiner, Hochreiter, "Fast and Accurate Deep Network Learning by Exponential
  Linear Units (ELUs)", ICLR 2016: https://arxiv.org/abs/1511.07289
- Swish/SiLU: Ramachandran, Zoph, Le, "Searching for Activation Functions", 2017:
  https://arxiv.org/abs/1710.05941
- Mish: Misra, "Mish: A Self Regularized Non-Monotonic Activation Function", 2019:
  https://arxiv.org/abs/1908.08681

Bound propagation context:
- CROWN: Zhang et al., "Efficient Neural Network Robustness Certification with General Activation
  Functions", 2018: https://arxiv.org/abs/1811.00866
- auto_LiRPA: Xu et al., "Automatic Perturbation Analysis for Scalable Certified Robustness and
  Beyond", NeurIPS 2020: https://arxiv.org/abs/2002.12920
-/

@[expose] public section


namespace NN.MLTheory.CROWN.Operators.Activations

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN

variable {α : Type} [Context α]

/-! ## Approximation Helpers -/

/-- Low-order Taylor approximation of `exp`. This is executable, not an enclosure theorem. -/
def expApprox (x : α) : α :=
  -- Taylor: exp(x) ≈ 1 + x + x²/2 + x³/6
  let x2 := x * x
  let x3 := x2 * x
  Numbers.one + x + x2 * Numbers.pointfive + x3 / (Numbers.one + Numbers.one + Numbers.one +
    Numbers.one + Numbers.one + Numbers.one)

/-- Low-order approximation of `log(1 + y)` around `y = 0`; not an enclosure theorem. -/
def logApprox (x : α) : α :=
  -- log(1 + y) ≈ y - y²/2 + y³/3 for small y
  let y := x - Numbers.one
  y - y * y * Numbers.pointfive + y * y * y / Numbers.three

/-- Executable `softplus` approximation computed via `logApprox (1 + expApprox x)`. -/
def softplusApprox (x : α) : α :=
  -- softplus(x) ≈ log(1 + exp(x))
  let ex := expApprox x
  logApprox (Numbers.one + ex)

/-- Executable `sigmoid` approximation computed via `expApprox`. -/
def sigmoidApprox (x : α) : α :=
  Numbers.one / (Numbers.one + expApprox (-x))

/-- Low-order approximation of `tanh`; not an enclosure theorem. -/
def tanhApprox (x : α) : α :=
  let x2 := x * x
  let x3 := x2 * x
  x - x3 / Numbers.three

/-! ### Leaky ReLU -/

/-- Leaky ReLU: f(x) = x if x ≥ 0, αx if x < 0. -/
def leakyRelu (negSlope : α) (x : α) : α :=
  if x > Numbers.zero then x else negSlope * x

/-- IBP for Leaky ReLU on scalars. -/
def ibpLeakyReluScalar (negSlope : α) (l u : α) : α × α :=
  -- Both branches are linear, just apply to endpoints
  let fl := leakyRelu negSlope l
  let fu := leakyRelu negSlope u
  -- For the usual positive slope, the function is monotone increasing.
  if negSlope > Numbers.zero then
    (fl, fu)
  else
    -- If negative slope, need min/max
    (if fl < fu then fl else fu, if fl > fu then fl else fu)

/-- IBP for Leaky ReLU on boxes. -/
def ibpLeakyRelu (n : Nat) (negSlope : α) (B : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  match B.lo, B.hi with
  | .dim lo, .dim hi =>
    let outLo := Tensor.dim (fun i =>
      match lo i, hi i with
      | .scalar l, .scalar u =>
        let (bl, _) := ibpLeakyReluScalar negSlope l u
        Tensor.scalar bl)
    let outHi := Tensor.dim (fun i =>
      match lo i, hi i with
      | .scalar l, .scalar u =>
        let (_, bh) := ibpLeakyReluScalar negSlope l u
        Tensor.scalar bh)
    { lo := outLo, hi := outHi }

/-- Affine bounds for Leaky ReLU.
    Unlike ReLU, both branches are linear so relaxation is easier. -/
def affLeakyRelu (negSlope : α) (l u : α) : α × α × α × α :=
  if l > Numbers.zero then
    -- Positive region: y = x
    (Numbers.one, Numbers.zero, Numbers.one, Numbers.zero)
  else if u < Numbers.zero then
    -- Negative region: y = αx
    (negSlope, Numbers.zero, negSlope, Numbers.zero)
  else
    -- Crossing region: both branches active
    -- Use triangle relaxation similar to ReLU
    let slope_upper := (u - negSlope * l) / (u - l)
    let bias_upper := u * (Numbers.one - slope_upper)
    -- Lower bound: use minimum slope (α if α < 1, else 1)
    let slope_lower := if negSlope < Numbers.one then negSlope else Numbers.one
    (slope_lower, Numbers.zero, slope_upper, bias_upper)

/-- Derivative bounds for Leaky ReLU (for Lyapunov). -/
def derivLeakyRelu (negSlope : α) (l u : α) : α × α :=
  if l > Numbers.zero then
    (Numbers.one, Numbers.one)
  else if u < Numbers.zero then
    (negSlope, negSlope)
  else
    -- Crossing: derivative is either α or 1
    (if negSlope < Numbers.one then negSlope else Numbers.one,
     if negSlope > Numbers.one then negSlope else Numbers.one)

/-! ### ELU (Exponential Linear Unit) -/

/-- ELU: f(x) = x if x > 0, α(exp(x) - 1) if x ≤ 0. -/
def elu (scale : α) (x : α) : α :=
  if x > Numbers.zero then x
  else scale * (expApprox x - Numbers.one)

/-- IBP for ELU on scalars. -/
def ibpEluScalar (scale : α) (l u : α) : α × α :=
  if l > Numbers.zero then
    -- Pure positive region
    (l, u)
  else if u < Numbers.zero then
    -- Pure negative region: α(exp(x) - 1), monotone
    (scale * (expApprox l - Numbers.one), scale * (expApprox u - Numbers.one))
  else
    -- Crossing: min at most negative, max at most positive
    let negMin := scale * (expApprox l - Numbers.one)
    (negMin, u)

/-- IBP for ELU on boxes. -/
def ibpElu (n : Nat) (scale : α) (B : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  match B.lo, B.hi with
  | .dim lo, .dim hi =>
    let outLo := Tensor.dim (fun i =>
      match lo i, hi i with
      | .scalar l, .scalar u =>
        Tensor.scalar (ibpEluScalar scale l u).1)
    let outHi := Tensor.dim (fun i =>
      match lo i, hi i with
      | .scalar l, .scalar u =>
        Tensor.scalar (ibpEluScalar scale l u).2)
    { lo := outLo, hi := outHi }

/-- Affine bounds for ELU.
    Positive branch is linear, negative branch is nonlinear (exp). -/
def affElu (scale : α) (l u : α) : α × α × α × α :=
  if l > Numbers.zero then
    (Numbers.one, Numbers.zero, Numbers.one, Numbers.zero)
  else if u < Numbers.zero then
    -- Negative region: use secant for bounds
    let fl := scale * (expApprox l - Numbers.one)
    let fu := scale * (expApprox u - Numbers.one)
    let slope := (fu - fl) / (u - l)
    (slope, fl - slope * l, slope, fl - slope * l)
  else
    -- Crossing region
    let fl := scale * (expApprox l - Numbers.one)
    -- Upper: linear for positive, secant for negative part
    let slope_upper := (u - fl) / (u - l)
    let bias_upper := u - slope_upper * u
    -- Lower: conservative
    let slope_lower := scale * expApprox l  -- derivative at l
    (slope_lower, fl - slope_lower * l, slope_upper, bias_upper)

/-- Derivative bounds for ELU. -/
def derivElu (scale : α) (l u : α) : α × α :=
  if l > Numbers.zero then
    (Numbers.one, Numbers.one)
  else if u < Numbers.zero then
    -- Derivative: α·exp(x), monotone increasing
    (scale * expApprox l, scale * expApprox u)
  else
    -- Crossing: min derivative in negative region, max is 1
    (scale * expApprox l, Numbers.one)

/-! ### Softplus -/

/-- Softplus: f(x) = log(1 + exp(x)).
    Using approximation since Numbers.log may not be available. -/
def softplus (x : α) : α :=
  -- For numerical stability: log(1 + exp(x)) ≈ x for large x
  -- Use approximation
  softplusApprox x

/-- IBP for Softplus. Softplus is monotone increasing. -/
def ibpSoftplusScalar (l u : α) : α × α :=
  (softplusApprox l, softplusApprox u)

/-- IBP for Softplus on boxes. -/
def ibpSoftplus (n : Nat) (B : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  match B.lo, B.hi with
  | .dim lo, .dim hi =>
    let outLo := Tensor.dim (fun i =>
      match lo i with
      | .scalar l => Tensor.scalar (softplusApprox l))
    let outHi := Tensor.dim (fun i =>
      match hi i with
      | .scalar u => Tensor.scalar (softplusApprox u))
    { lo := outLo, hi := outHi }

/-! ### SiLU/Swish -/

/-- SiLU/Swish: f(x) = x · sigmoid(x). -/
def silu (x : α) : α :=
  x * sigmoidApprox x

/-- IBP for SiLU. Not monotone, has global minimum around x ≈ -1.28. -/
def ibpSiluScalar (l u : α) : α × α :=
  let fl := silu l
  let fu := silu u
  -- SiLU minimum is at x ≈ -1.28, value ≈ -0.28
  let critX := -(Numbers.one + Numbers.pointfive * Numbers.pointfive)  -- approx -1.25
  let critVal := critX * sigmoidApprox critX
  if l > critX then
    -- Right of minimum, monotone increasing
    (fl, fu)
  else if u < critX then
    -- Left of minimum, monotone decreasing
    (fu, fl)
  else
    -- Spans minimum
    (critVal, if fl > fu then fl else fu)

/-- IBP for SiLU on boxes. -/
def ibpSilu (n : Nat) (B : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  match B.lo, B.hi with
  | .dim lo, .dim hi =>
    let outLo := Tensor.dim (fun i =>
      match lo i, hi i with
      | .scalar l, .scalar u =>
        Tensor.scalar (ibpSiluScalar l u).1)
    let outHi := Tensor.dim (fun i =>
      match lo i, hi i with
      | .scalar l, .scalar u =>
        Tensor.scalar (ibpSiluScalar l u).2)
    { lo := outLo, hi := outHi }

/-! ### Mish -/

/-- Mish: f(x) = x · tanh(softplus(x)). -/
def mish (x : α) : α :=
  x * tanhApprox (softplusApprox x)

/-- IBP for Mish. Similar behavior to SiLU. -/
def ibpMishScalar (l u : α) : α × α :=
  let fl := mish l
  let fu := mish u
  -- Mish minimum is around x ≈ -1.22
  let critX := -(Numbers.one + Numbers.pointfive * Numbers.pointfive)
  let critVal := mish critX
  if l > critX then
    (fl, fu)
  else if u < critX then
    (fu, fl)
  else
    (critVal, if fl > fu then fl else fu)

/-- IBP for Mish on boxes. -/
def ibpMish (n : Nat) (B : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  match B.lo, B.hi with
  | .dim lo, .dim hi =>
    let outLo := Tensor.dim (fun i =>
      match lo i, hi i with
      | .scalar l, .scalar u =>
        Tensor.scalar (ibpMishScalar l u).1)
    let outHi := Tensor.dim (fun i =>
      match lo i, hi i with
      | .scalar l, .scalar u =>
        Tensor.scalar (ibpMishScalar l u).2)
    { lo := outLo, hi := outHi }

namespace Theorems

/-- Leaky ReLU definition structure. -/
theorem leaky_relu_def (negSlope x : α) :
    leakyRelu negSlope x = if x > Numbers.zero then x else negSlope * x := by
  rfl

/-- ELU definition structure. -/
theorem elu_def (scale x : α) :
    elu scale x = if x > Numbers.zero then x else scale * (expApprox x - Numbers.one) := by
  rfl

/-- Softplus definition structure. -/
theorem softplus_def (x : α) :
    softplus x = softplusApprox x := by
  rfl

/-- IBP for leaky ReLU returns a pair. -/
theorem ibp_leaky_relu_scalar_pair (negSlope l u : α) :
    ∃ lo hi : α, ibpLeakyReluScalar negSlope l u = (lo, hi) := by
  exact ⟨(ibpLeakyReluScalar negSlope l u).1, (ibpLeakyReluScalar negSlope l u).2, rfl⟩

/-- IBP for ELU returns a pair. -/
theorem ibp_elu_scalar_pair (scale l u : α) :
    ∃ lo hi : α, ibpEluScalar scale l u = (lo, hi) := by
  exact ⟨(ibpEluScalar scale l u).1, (ibpEluScalar scale l u).2, rfl⟩

end Theorems

end NN.MLTheory.CROWN.Operators.Activations

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Core

/-!
# Leaky-ReLU Bounds

Interval, affine, and derivative transfer rules for
`leakyRelu negSlope x = if x > 0 then x else negSlope * x`.

The formulas cover positive, zero, and negative branch slopes. In particular, when a non-positive
slope crosses zero, the interval rule includes the value at the kink instead of considering only
the two endpoints.

Reference: Zhang et al., "Efficient Neural Network Robustness Certification with General
Activation Functions", NeurIPS 2018, arXiv:1811.00866.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Operators.Activations

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- Leaky ReLU with slope `negSlope` on the non-positive branch. -/
def leakyRelu (negSlope : α) (x : α) : α :=
  if x > 0 then x else negSlope * x

/-- Minimum of two values using the executable scalar order. -/
def min2 (x y : α) : α :=
  if x < y then x else y

/-- Maximum of two values using the executable scalar order. -/
def max2 (x y : α) : α :=
  if x > y then x else y

/-- Exact endpoint-and-kink interval propagation for scalar Leaky ReLU. -/
def ibpLeakyReluScalar (negSlope : α) (lo hi : α) : α × α :=
  let flo := leakyRelu negSlope lo
  let fhi := leakyRelu negSlope hi
  if negSlope > 0 then
    (flo, fhi)
  else if (!(lo > 0)) && (!(hi < 0)) then
    (min2 (min2 flo fhi) 0, max2 (max2 flo fhi) 0)
  else
    (min2 flo fhi, max2 flo fhi)

/-- Apply `ibpLeakyReluScalar` coordinatewise to a vector box. -/
def ibpLeakyRelu (n : Nat) (negSlope : α) (box : Box α (.dim n .scalar)) :
    Box α (.dim n .scalar) :=
  { lo := Tensor.dim fun i =>
      Tensor.scalar (ibpLeakyReluScalar negSlope
        (box.lo.getScalar i) (box.hi.getScalar i)).1
    hi := Tensor.dim fun i =>
      Tensor.scalar (ibpLeakyReluScalar negSlope
        (box.lo.getScalar i) (box.hi.getScalar i)).2 }

/--
Lower and upper affine forms for Leaky ReLU on `[lo, hi]`, returned as
`(lowerSlope, lowerBias, upperSlope, upperBias)`.

For a crossing interval the function is convex when `negSlope ≤ 1` and concave when
`negSlope > 1`; the secant and branch support exchange roles accordingly.
-/
def affLeakyRelu (negSlope : α) (lo hi : α) : α × α × α × α :=
  if lo > 0 then
    (1, 0, 1, 0)
  else if hi < 0 then
    (negSlope, 0, negSlope, 0)
  else if !(hi > lo) then
    (0, 0, 0, 0)
  else
    let secantSlope := (hi - negSlope * lo) / (hi - lo)
    let secantBias := hi * (1 - secantSlope)
    if negSlope > 1 then
      (secantSlope, secantBias, negSlope, 0)
    else
      (negSlope, 0, secantSlope, secantBias)

/-- Range of the two branch derivatives over an interval. -/
def derivLeakyRelu (negSlope : α) (lo hi : α) : α × α :=
  if lo > 0 then
    (1, 1)
  else if hi < 0 then
    (negSlope, negSlope)
  else
    (if negSlope < 1 then negSlope else 1,
     if negSlope > 1 then negSlope else 1)

end NN.MLTheory.CROWN.Operators.Activations

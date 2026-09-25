/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Core
public import NN.Spec.Layers.Activation

/-!
# Runtime CROWN operators

Executable helper operators for the graph-based CROWN/IBP engine.

This file keeps runtime certificate replay separate from proof imports:
- Some proof layer CROWN modules import `Mathlib` and large theorem developments. Native
  executables that only replay certificates should not pay that import cost.
- The graph verifier and executable certificate checks only need a compact set of computational
  definitions: ReLU relaxations plus interval rules for a few scalar activations.

These definitions live under `NN.MLTheory.CROWN.Runtime.Ops`, with no direct Mathlib dependency.
The proof modules can cite these functions, but this file itself stays focused on the runtime
support code used for fast certificate replay.

References (bound propagation background):
- Zhang et al., "Efficient Neural Network Robustness Certification with General Activation
  Functions" (CROWN), 2018: https://arxiv.org/abs/1811.00866
- Singh et al., "An Abstract Domain for Certifying Neural Networks" (DeepPoly), POPL 2019.
-/

@[expose] public section


namespace NN.MLTheory.CROWN.Runtime.Ops

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- Parameters of a per-neuron affine relaxation `y = slope * x + bias` used in CROWN/DeepPoly. -/
structure ReLURelax (α : Type) where
  /-- Linear coefficient. -/
  slope : α
  /-- Constant offset. -/
  bias  : α

namespace ReLU

/--
Upper (over-approx) affine relaxation for ReLU on an interval `[l,u]`.

Returns parameters `(slope, bias)` for a line `y = slope * x + bias` that upper-bounds `relu x`
for all `x ∈ [l,u]`.
-/
def relaxScalar (l u : α) : ReLURelax α :=
  if u > 0 then
    if l > 0 then
      { slope := 1, bias := 0 }
    else
      let denom := (u - l)
      let αs := u / denom
      let β := -αs * l
      { slope := αs, bias := β }
  else
    { slope := 0, bias := 0 }

/-!
Lower (under-approx) relaxation for ReLU.

For crossing bounds `l < 0 < u`, basic CROWN/DeepPoly chooses either:
- `y ≥ 0` (slope 0), or
- `y ≥ x` (slope 1),
based on which side of 0 is “wider”. This is the non-α-optimized lower relaxation.
-/
def relaxScalarLower (l u : α) : ReLURelax α :=
  if u > 0 then
    if l > 0 then
      { slope := 1, bias := 0 }
    else
      -- crossing: choose either y ≥ 0 or y ≥ x
      let slope :=
        if u > (-l) then 1 else 0
      { slope := slope, bias := 0 }
  else
    { slope := 0, bias := 0 }

/-- Apply `relaxScalar` componentwise to vector lower and upper bound tensors. -/
def relaxVector {n : Nat} (lo hi : Tensor α [n]) :
    Tensor (ReLURelax α) [n] :=
  Tensor.dim (fun i =>
    Tensor.scalar (relaxScalar (lo.getScalar i) (hi.getScalar i)))

/-- Apply `relaxScalarLower` componentwise to vector lower and upper bound tensors. -/
def relaxVectorLower {n : Nat} (lo hi : Tensor α [n]) :
    Tensor (ReLURelax α) [n] :=
  Tensor.dim (fun i =>
    Tensor.scalar (relaxScalarLower (lo.getScalar i) (hi.getScalar i)))

/--
Propagate an affine form through ReLU using a per-neuron relaxation.

Given `y ≈ A*x + c` and per-output relaxations `(slopeᵢ, biasᵢ)`, produces the affine form
`y' ≈ diag(slope) * (A*x + c) + bias`.
-/
def propagateAffine {inDim hidDim : Nat}
  (relax : Tensor (ReLURelax α) [hidDim])
  (aff : AffineVec α inDim hidDim) : AffineVec α inDim hidDim :=
  let A' := Tensor.dim (fun i =>
    let rp := relax.getScalar i
    Tensor.dim (fun j => Tensor.scalar (get2 aff.A i j * rp.slope)))
  let c' := Tensor.dim (fun i =>
    let rp := relax.getScalar i
    Tensor.scalar (rp.slope * aff.c.getScalar i + rp.bias))
  { A := A', c := c' }

end ReLU

namespace IBP

/-- Generic elementwise bound propagation for monotone activations (min/max of endpoints). -/
def mapMinmax {s : Shape} (f : α → α) (xB : Box α s) : Box α s :=
  let outLo := Tensor.map2Spec (fun l u =>
    let fl := f l
    let fu := f u
    if fl > fu then fu else fl) xB.lo xB.hi
  let outHi := Tensor.map2Spec (fun l u =>
    let fl := f l
    let fu := f u
    if fl > fu then fl else fu) xB.lo xB.hi
  { lo := outLo, hi := outHi }

/-- Interval bound propagation for `sigmoid` (monotone, so min/max of endpoints). -/
def sigmoid {s : Shape} (xB : Box α s) : Box α s :=
  mapMinmax Activation.Math.sigmoidSpec xB

/-- Interval bound propagation for `tanh` (monotone, so min/max of endpoints). -/
def tanh {s : Shape} (xB : Box α s) : Box α s :=
  mapMinmax Activation.Math.tanhSpec xB

/--
Conservative IBP for `sin` using a 1-Lipschitz enclosure:

$$
\sin([l,u])\subseteq[\sin(m)-r,\sin(m)+r]\cap[-1,1],
\qquad m=\frac{l+u}{2},\quad r=\frac{u-l}{2}.
$$

This avoids periodic case splits (no `floor/ceil` in `Context α`) while remaining sound.
-/
def sin {s : Shape} (xB : Box α s) : Box α s :=
  let outLo := Tensor.map2Spec (fun l u =>
    let m := (l + u) / 2
    let r := (u - l) / 2
    let base := MathFunctions.sin m
    max (-1) (base - r)) xB.lo xB.hi
  let outHi := Tensor.map2Spec (fun l u =>
    let m := (l + u) / 2
    let r := (u - l) / 2
    let base := MathFunctions.sin m
    min 1 (base + r)) xB.lo xB.hi
  { lo := outLo, hi := outHi }

/-- Same 1-Lipschitz enclosure as `IBP.sin`, but for `cos`. -/
def cos {s : Shape} (xB : Box α s) : Box α s :=
  let outLo := Tensor.map2Spec (fun l u =>
    let m := (l + u) / 2
    let r := (u - l) / 2
    let base := MathFunctions.cos m
    max (-1) (base - r)) xB.lo xB.hi
  let outHi := Tensor.map2Spec (fun l u =>
    let m := (l + u) / 2
    let r := (u - l) / 2
    let base := MathFunctions.cos m
    min 1 (base + r)) xB.lo xB.hi
  { lo := outLo, hi := outHi }

end IBP

end NN.MLTheory.CROWN.Runtime.Ops

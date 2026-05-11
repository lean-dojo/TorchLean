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
- Some proof-facing CROWN modules import `Mathlib` and large theorem developments. Native
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

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN

variable {α : Type} [Context α]

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
based on which side of 0 is “wider”. This keeps the relaxation sound without α-optimization.
-/
def relaxScalarLower (l u : α) : ReLURelax α :=
  if u > 0 then
    if l > 0 then
      { slope := 1, bias := 0 }
    else
      -- crossing: choose either y ≥ 0 or y ≥ x
      let slope :=
        if u > (-l) then Numbers.one else Numbers.zero
      { slope := slope, bias := 0 }
  else
    { slope := 0, bias := 0 }

/-- Vectorized `relax_scalar`, applied componentwise to `lo`/`hi`. -/
def relaxVector {n : Nat} (lo hi : Tensor α (.dim n .scalar)) :
    Tensor (ReLURelax α) (.dim n .scalar) :=
  match lo, hi with
  | Tensor.dim l, Tensor.dim u =>
    Tensor.dim (fun i => match l i, u i with
      | Tensor.scalar li, Tensor.scalar ui => Tensor.scalar (relaxScalar li ui))

/-- Vectorized `relax_scalar_lower`, applied componentwise to `lo`/`hi`. -/
def relaxVectorLower {n : Nat} (lo hi : Tensor α (.dim n .scalar)) :
    Tensor (ReLURelax α) (.dim n .scalar) :=
  match lo, hi with
  | Tensor.dim l, Tensor.dim u =>
    Tensor.dim (fun i => match l i, u i with
      | Tensor.scalar li, Tensor.scalar ui => Tensor.scalar (relaxScalarLower li ui))

/--
Propagate an affine form through ReLU using a per-neuron relaxation.

Given `y ≈ A*x + c` and per-output relaxations `(slopeᵢ, biasᵢ)`, produces the affine form
`y' ≈ diag(slope) * (A*x + c) + bias`.
-/
def propagateAffine {inDim hidDim : Nat}
  (relax : Tensor (ReLURelax α) (.dim hidDim .scalar))
  (aff : AffineVec α inDim hidDim) : AffineVec α inDim hidDim :=
  match relax, aff.A, aff.c with
  | Tensor.dim r, Tensor.dim rows, Tensor.dim bias =>
    let A' := Tensor.dim (fun i =>
      match rows i, r i with
      | Tensor.dim cols, Tensor.scalar rp =>
        Tensor.dim (fun j =>
          match cols j with
          | Tensor.scalar aij => Tensor.scalar (aij * rp.slope)))
    let c' := Tensor.dim (fun i =>
      match bias i, r i with
      | Tensor.scalar ci, Tensor.scalar rp => Tensor.scalar (rp.slope * ci + rp.bias))
    { A := A', c := c' }

end ReLU

namespace IBP

/-- Generic elementwise bound propagation for monotone activations (min/max of endpoints). -/
def mapMinmax {n : Nat} (f : α → α) (xB : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  match xB.lo, xB.hi with
  | Tensor.dim lo, Tensor.dim hi =>
    let outLo := Tensor.dim (fun i =>
      match lo i, hi i with
      | Tensor.scalar l, Tensor.scalar u =>
        let fl := f l; let fu := f u
        Tensor.scalar (if fl > fu then fu else fl))
    let outHi := Tensor.dim (fun i =>
      match lo i, hi i with
      | Tensor.scalar l, Tensor.scalar u =>
        let fl := f l; let fu := f u
        Tensor.scalar (if fl > fu then fl else fu))
    { lo := outLo, hi := outHi }

/-- Interval bound propagation for `sigmoid` (monotone, so min/max of endpoints). -/
def sigmoid {n : Nat} (xB : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  mapMinmax Activation.Math.sigmoidSpec xB

/-- Interval bound propagation for `tanh` (monotone, so min/max of endpoints). -/
def tanh {n : Nat} (xB : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  mapMinmax Activation.Math.tanhSpec xB

/--
Conservative IBP for `sin` using a 1-Lipschitz enclosure:
  sin([l,u]) ⊆ [sin(m)-r, sin(m)+r] ∩ [-1,1],  m=(l+u)/2, r=(u-l)/2.

This avoids periodic case splits (no `floor/ceil` in `Context α`) while remaining sound.
-/
def sin {n : Nat} (xB : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  match xB.lo, xB.hi with
  | Tensor.dim lo, Tensor.dim hi =>
    let outLo := Tensor.dim (fun i =>
      match lo i, hi i with
      | Tensor.scalar l, Tensor.scalar u =>
        let m := (l + u) / Numbers.two
        let r := (u - l) / Numbers.two
        let base := MathFunctions.sin m
        let rawLo := base - r
        Tensor.scalar (max Numbers.neg_one rawLo))
    let outHi := Tensor.dim (fun i =>
      match lo i, hi i with
      | Tensor.scalar l, Tensor.scalar u =>
        let m := (l + u) / Numbers.two
        let r := (u - l) / Numbers.two
        let base := MathFunctions.sin m
        let rawHi := base + r
        Tensor.scalar (min Numbers.one rawHi))
    { lo := outLo, hi := outHi }

/-- Same 1-Lipschitz enclosure as `IBP.sin`, but for `cos`. -/
def cos {n : Nat} (xB : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  match xB.lo, xB.hi with
  | Tensor.dim lo, Tensor.dim hi =>
    let outLo := Tensor.dim (fun i =>
      match lo i, hi i with
      | Tensor.scalar l, Tensor.scalar u =>
        let m := (l + u) / Numbers.two
        let r := (u - l) / Numbers.two
        let base := MathFunctions.cos m
        let rawLo := base - r
        Tensor.scalar (max Numbers.neg_one rawLo))
    let outHi := Tensor.dim (fun i =>
      match lo i, hi i with
      | Tensor.scalar l, Tensor.scalar u =>
        let m := (l + u) / Numbers.two
        let r := (u - l) / Numbers.two
        let base := MathFunctions.cos m
        let rawHi := base + r
        Tensor.scalar (min Numbers.one rawHi))
    { lo := outLo, hi := outHi }

end IBP

end NN.MLTheory.CROWN.Runtime.Ops

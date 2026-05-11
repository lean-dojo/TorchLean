/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Algebra.Soundness
public import NN.Spec.Core.Utils

import Mathlib.Algebra.Order.Algebra

/-!
# Dual

Dual numbers for forward-mode differentiation.

This is a lightweight runtime-oriented `Dual α` scalar that can be used to compute:
- Jacobian-vector products by running a function once with dual inputs, and
- Hessian-vector products by running reverse-mode over dual scalars (forward-over-reverse).

Notes:
- This is an *execution* facility. It is not connected to the `fderiv` proof layer.
- Non-smooth primitives (`abs`, `max`, `min`) use a subgradient-like choice based on the primal.
- `Pow` uses the standard `x^y` rule (`d (x^y) = x^y * (y' * log x + y * x'/x)`), which is
  only mathematically justified on domains where that expression is defined.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace TorchLean

open Spec
open Tensor

/--
A dual number `x + ε·dx` used for forward-mode automatic differentiation.

`re` is the primal value and `du` is the tangent (directional derivative). When you run a
computation once on dual inputs, the resulting `du` component computes a Jacobian-vector product.
-/
structure Dual (α : Type) where
  /-- re. -/
  re : α
  /-- du. -/
  du : α
deriving Repr

namespace Dual

/-- Project the primal component of a dual number. -/
@[simp] def primal {α : Type} (x : Dual α) : α := x.re
/-- Project the tangent component of a dual number. -/
@[simp] def tangent {α : Type} (x : Dual α) : α := x.du

/-- Embed a primal value as a dual number with zero tangent. -/
def ofPrimal {α : Type} [Zero α] (x : α) : Dual α := ⟨x, 0⟩
/-- Embed a tangent value as a dual number with zero primal. -/
def ofTangent {α : Type} [Zero α] (dx : α) : Dual α := ⟨0, dx⟩
/-- Convenience constructor using the `(primal, tangent)` order. -/
def mk' {α : Type} (x dx : α) : Dual α := ⟨x, dx⟩

end Dual

namespace Dual

/-- `Dual` is inhabited by `default` primal value with zero tangent. -/
instance {α : Type} [Inhabited α] [Zero α] : Inhabited (Dual α) :=
  ⟨⟨default, 0⟩⟩

/-- Zero dual number: `0 + ε·0`. -/
instance {α : Type} [Zero α] : Zero (Dual α) where
  zero := ⟨0, 0⟩
/-- One dual number: `1 + ε·0`. -/
instance {α : Type} [One α] [Zero α] : One (Dual α) where
  one := ⟨1, 0⟩
/-- Negation is componentwise (`-(x + ε·dx) = (-x) + ε·(-dx)`). -/
instance {α : Type} [Neg α] : Neg (Dual α) where
  neg x := ⟨-x.re, -x.du⟩

/-- Addition is componentwise, so tangents add linearly. -/
instance {α : Type} [Add α] : Add (Dual α) where
  add x y := ⟨x.re + y.re, x.du + y.du⟩
/-- Subtraction is componentwise, so tangents subtract linearly. -/
instance {α : Type} [Sub α] : Sub (Dual α) where
  sub x y := ⟨x.re - y.re, x.du - y.du⟩

/-- Multiplication uses the product rule: `(x·y)' = x'·y + x·y'`. -/
instance {α : Type} [Mul α] [Add α] : Mul (Dual α) :=
  ⟨fun x y => ⟨x.re * y.re, x.du * y.re + x.re * y.du⟩⟩

/-- Division uses the quotient rule: `(x / y)' = (x'·y - x·y') / y^2`. -/
instance {α : Type} [Div α] [Mul α] [Sub α] [Add α] : Div (Dual α) :=
  ⟨fun x y =>
    -- (x / y)' = (x' y - x y') / y^2
    let denom := y.re * y.re
    ⟨x.re / y.re, (x.du * y.re - x.re * y.du) / denom⟩⟩

/-- Boolean equality compares primals only (tangents are treated as metadata). -/
instance {α : Type} [BEq α] : BEq (Dual α) where
  beq x y := x.re == y.re
/-- Strict order compares primals only. -/
instance {α : Type} [LT α] : LT (Dual α) where
  lt x y := x.re < y.re
/-- Non-strict order compares primals only. -/
instance {α : Type} [LE α] : LE (Dual α) where
  le x y := x.re ≤ y.re

/-- `max` chooses the branch by primal value (subgradient-style for tangents). -/
instance {α : Type} [Context α] : Max (Dual α) where
  max x y := if x.re > y.re then x else y

/-- `min` chooses the branch by primal value (subgradient-style for tangents). -/
instance {α : Type} [Context α] : Min (Dual α) where
  min x y := if x.re < y.re then x else y

/-- Coerce naturals into dual numbers with zero tangent. -/
instance {α : Type} [Context α] : Coe Nat (Dual α) where
  coe n := ⟨(n : α), 0⟩

/-- Lift TorchLean's numeric literals (`Numbers`) into dual numbers with zero tangent. -/
instance {α : Type} [Context α] : Numbers (Dual α) where
  neg_point_five := ⟨Numbers.neg_point_five, 0⟩
  neg_one := ⟨Numbers.neg_one, 0⟩
  pointone := ⟨Numbers.pointone, 0⟩
  pointfive := ⟨Numbers.pointfive, 0⟩
  one := ⟨Numbers.one, 0⟩
  zero := ⟨Numbers.zero, 0⟩
  two := ⟨Numbers.two, 0⟩
  three := ⟨Numbers.three, 0⟩
  four := ⟨Numbers.four, 0⟩
  five := ⟨Numbers.five, 0⟩
  ten := ⟨Numbers.ten, 0⟩
  log10 := ⟨Numbers.log10, 0⟩
  log10000 := ⟨Numbers.log10000, 0⟩
  epsilon := ⟨Numbers.epsilon, 0⟩
  neg_thousand := ⟨Numbers.neg_thousand, 0⟩

/-- Forward-mode chain rule implementations for `MathFunctions` over dual numbers. -/
instance {α : Type} [Context α] : MathFunctions (Dual α) where
  exp x :=
    let ex := MathFunctions.exp x.re
    ⟨ex, ex * x.du⟩
  tanh x :=
    let th := MathFunctions.tanh x.re
    -- d/dx tanh = 1 - tanh^2
    ⟨th, (1 - th * th) * x.du⟩
  cosh x :=
    let ch := MathFunctions.cosh x.re
    ⟨ch, MathFunctions.sinh x.re * x.du⟩
  sinh x :=
    let sh := MathFunctions.sinh x.re
    ⟨sh, MathFunctions.cosh x.re * x.du⟩
  sqrt x :=
    let r := MathFunctions.sqrt x.re
    ⟨r, x.du / (((2 : Nat) : α) * r)⟩
  abs x :=
    if x.re < 0 then ⟨MathFunctions.abs x.re, -x.du⟩ else ⟨MathFunctions.abs x.re, x.du⟩
  log x :=
    ⟨MathFunctions.log x.re, x.du / x.re⟩
  pi := ⟨MathFunctions.pi, 0⟩
  cos x :=
    ⟨MathFunctions.cos x.re, -(MathFunctions.sin x.re) * x.du⟩
  sin x :=
    ⟨MathFunctions.sin x.re, (MathFunctions.cos x.re) * x.du⟩

/--
Power rule for `x^y` over dual numbers.

We use the standard identity `d(x^y) = x^y * (y' * log x + y * x'/x)`, which is mathematically
justified only on domains where the right-hand side is defined.
-/
instance {α : Type} [Context α] : Pow (Dual α) (Dual α) where
  pow x y :=
    -- d (x^y) = x^y * (y' * log x + y * x'/x)
    let r : α := x.re ^ y.re
    let dr : α := r * (y.du * MathFunctions.log x.re + y.re * (x.du / x.re))
    ⟨r, dr⟩

/-- Lift a scalar `Context` to dual numbers by deciding comparisons on primals. -/
instance {α : Type} [Context α] : Context (Dual α) where
  decidable_gt := fun x y => (Context.decidable_gt) x.re y.re

end Dual

namespace DualTensor

/-- Map a tensor of primals to a tensor of duals with zero tangents. -/
def ofPrimal {α : Type} [Zero α] : ∀ {s : Shape}, Tensor α s → Tensor (Dual α) s :=
  fun {_s} t => Spec.mapTensor (fun a => Dual.ofPrimal a) t

/-- Lift a shape-indexed tensor list (`TList`) to dual numbers with zero tangents. -/
def ofPrimalTList {α : Type} [Zero α] :
    {ss : List Shape} → _root_.Proofs.Autograd.Algebra.TList α ss →
      _root_.Proofs.Autograd.Algebra.TList (Dual α) ss
  | [], .nil => .nil
  | _ :: ss, .cons x xs => .cons (ofPrimal (s := _) x) (ofPrimalTList (ss := ss) xs)

/--
Combine a primal tensor and a tangent tensor into a dual tensor.

This is the tensor-level analogue of `Dual.mk'`.
-/
def withTangents {α : Type} [Context α] :
    {s : Shape} → Tensor α s → Tensor α s → Tensor (Dual α) s
  | .scalar, .scalar x, .scalar dx => .scalar ⟨x, dx⟩
  | .dim _n s, .dim xs, .dim dxs => .dim (fun i => withTangents (s := s) (xs i) (dxs i))

/-- Tensor-list version of `withTangents`. -/
def withTangentsTList {α : Type} [Context α] :
    {ss : List Shape} →
      _root_.Proofs.Autograd.Algebra.TList α ss →
      _root_.Proofs.Autograd.Algebra.TList α ss →
      _root_.Proofs.Autograd.Algebra.TList (Dual α) ss
  | [], .nil, .nil => .nil
  | _ :: ss, .cons x xs, .cons dx dxs =>
      .cons (withTangents (s := _) x dx) (withTangentsTList (ss := ss) xs dxs)

/-- Project the tangent part of a dual tensor. -/
def tangent {α : Type} : ∀ {s : Shape}, Tensor (Dual α) s → Tensor α s
  | .scalar, .scalar x => .scalar x.du
  | .dim _n s, .dim xs => .dim (fun i => tangent (s := s) (xs i))

/-- Tensor-list version of `tangent`. -/
def tangentTList {α : Type} :
    {ss : List Shape} → _root_.Proofs.Autograd.Algebra.TList (Dual α) ss →
      _root_.Proofs.Autograd.Algebra.TList α ss
  | [], .nil => .nil
  | _ :: ss, .cons x xs => .cons (tangent (s := _) x) (tangentTList (ss := ss) xs)

end DualTensor

end TorchLean
end Autograd
end Runtime

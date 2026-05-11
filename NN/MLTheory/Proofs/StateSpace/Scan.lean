/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.SelectiveScan
import Mathlib.Algebra.Ring.Basic

/-!
# Proofs for affine selective scan

The Mamba/S4 scan theorem is an algebra theorem about affine maps.  A sequential recurrent update
and a parallel prefix scan are equivalent because affine transition composition is associative:

`(a₂,b₂) ∘ (a₁,b₁) = (a₂*a₁, a₂*b₁ + b₂)`.

The tensor/CUDA implementation is allowed to choose an efficient scan schedule, but the mathematical
contract is this file: prefix summaries denote the same state as the left-to-right recurrence.
-/

@[expose] public section

namespace NN
namespace MLTheory
namespace StateSpace

open _root_.Spec
namespace ScalarAffineTransition

variable {α : Type}

/-- Composing scalar affine transitions agrees with function composition. -/
@[simp] theorem compose_apply [Semiring α] (t₂ t₁ : _root_.Spec.ScalarAffineTransition α)
    (h : α) :
    (_root_.Spec.ScalarAffineTransition.compose t₂ t₁).apply h = t₂.apply (t₁.apply h) := by
  cases t₁
  cases t₂
  simp [_root_.Spec.ScalarAffineTransition.compose, _root_.Spec.ScalarAffineTransition.apply,
    mul_add, mul_assoc, add_assoc]

/-- The identity transition is a left identity for composition. -/
@[simp] theorem compose_id_left [Semiring α] (t : _root_.Spec.ScalarAffineTransition α) :
    _root_.Spec.ScalarAffineTransition.compose _root_.Spec.ScalarAffineTransition.id t = t := by
  cases t
  simp [_root_.Spec.ScalarAffineTransition.compose, _root_.Spec.ScalarAffineTransition.id]

/-- The identity transition is a right identity for composition. -/
@[simp] theorem compose_id_right [Semiring α] (t : _root_.Spec.ScalarAffineTransition α) :
    _root_.Spec.ScalarAffineTransition.compose t _root_.Spec.ScalarAffineTransition.id = t := by
  cases t
  simp [_root_.Spec.ScalarAffineTransition.compose, _root_.Spec.ScalarAffineTransition.id]

/-- Scalar affine transition composition is associative. -/
@[simp] theorem compose_assoc [Semiring α]
    (t₃ t₂ t₁ : _root_.Spec.ScalarAffineTransition α) :
    _root_.Spec.ScalarAffineTransition.compose
        (_root_.Spec.ScalarAffineTransition.compose t₃ t₂) t₁ =
      _root_.Spec.ScalarAffineTransition.compose t₃
        (_root_.Spec.ScalarAffineTransition.compose t₂ t₁) := by
  cases t₁
  cases t₂
  cases t₃
  simp [_root_.Spec.ScalarAffineTransition.compose, mul_add, mul_assoc, add_assoc]

/-- Zero is a fixed point of a homogeneous scalar transition. -/
@[simp] theorem homogeneous_zero_fixed [Semiring α] (a : α) :
    (_root_.Spec.ScalarAffineTransition.apply { a := a, b := 0 } 0) = 0 := by
  simp [_root_.Spec.ScalarAffineTransition.apply]

end ScalarAffineTransition

namespace DiagonalTransition

variable {α : Type}

/-- Applying a diagonal transition is exactly the scalar affine update in each channel. -/
@[simp] theorem apply_vecGet [Add α] [Mul α] {stateDim : Nat}
    (tr : _root_.Spec.DiagonalTransition α stateDim)
    (h : _root_.Spec.Tensor α (.dim stateDim .scalar))
    (i : Fin stateDim) :
    _root_.Spec.Tensor.vecGet (tr.apply h) i =
      _root_.Spec.Tensor.vecGet tr.a i * _root_.Spec.Tensor.vecGet h i +
        _root_.Spec.Tensor.vecGet tr.b i := by
  cases tr with
  | mk a b =>
      cases a with
      | dim fa =>
          cases b with
          | dim fb =>
              cases h with
              | dim fh =>
                  cases hfa : fa i with
                  | scalar ai =>
                  cases hfh : fh i with
                  | scalar hi =>
                  cases hfb : fb i with
                  | scalar bi =>
                  change
                    _root_.Spec.Tensor.toScalar
                      (_root_.Spec.get
                        (_root_.Spec.Tensor.addSpec
                          (_root_.Spec.Tensor.mulSpec (Tensor.dim fa) (Tensor.dim fh))
                          (Tensor.dim fb)) i) =
                    (fa i).toScalar * (fh i).toScalar + (fb i).toScalar
                  simp [_root_.Spec.get, _root_.Spec.getAtSpec,
                    _root_.Spec.Tensor.addSpec, _root_.Spec.Tensor.mulSpec,
                    _root_.Spec.Tensor.map2Spec, _root_.Spec.Tensor.toScalar,
                    hfa, hfh, hfb]

/--
Composing diagonal transitions agrees channelwise with composing the corresponding scalar affine
maps.  This is the exact algebraic invariant used by the variable-coefficient selective-scan
kernel: each flattened state lane is an independent affine scan.
-/
@[simp] theorem compose_apply_vecGet [Semiring α] {stateDim : Nat}
    (t₂ t₁ : _root_.Spec.DiagonalTransition α stateDim)
    (h : _root_.Spec.Tensor α (.dim stateDim .scalar))
    (i : Fin stateDim) :
    _root_.Spec.Tensor.vecGet ((_root_.Spec.DiagonalTransition.compose t₂ t₁).apply h) i =
      _root_.Spec.Tensor.vecGet (t₂.apply (t₁.apply h)) i := by
  cases t₁ with
  | mk a₁ b₁ =>
      cases t₂ with
      | mk a₂ b₂ =>
          cases a₁ with
          | dim fa₁ =>
              cases b₁ with
              | dim fb₁ =>
                  cases a₂ with
                  | dim fa₂ =>
                      cases b₂ with
                      | dim fb₂ =>
                          cases h with
                          | dim fh =>
                              cases hfa₁ : fa₁ i with
                              | scalar a₁ =>
                              cases hfb₁ : fb₁ i with
                              | scalar b₁ =>
                              cases hfa₂ : fa₂ i with
                              | scalar a₂ =>
                              cases hfb₂ : fb₂ i with
                              | scalar b₂ =>
                              cases hfh : fh i with
                              | scalar hi =>
                              change
                                _root_.Spec.Tensor.toScalar
                                  (_root_.Spec.get
                                    (_root_.Spec.Tensor.addSpec
                                      (_root_.Spec.Tensor.mulSpec
                                        (_root_.Spec.Tensor.mulSpec
                                          (Tensor.dim fa₂) (Tensor.dim fa₁))
                                        (Tensor.dim fh))
                                      (_root_.Spec.Tensor.addSpec
                                        (_root_.Spec.Tensor.mulSpec
                                          (Tensor.dim fa₂) (Tensor.dim fb₁))
                                        (Tensor.dim fb₂))) i) =
                                _root_.Spec.Tensor.toScalar
                                  (_root_.Spec.get
                                    (_root_.Spec.Tensor.addSpec
                                      (_root_.Spec.Tensor.mulSpec
                                        (Tensor.dim fa₂)
                                        (_root_.Spec.Tensor.addSpec
                                          (_root_.Spec.Tensor.mulSpec
                                            (Tensor.dim fa₁) (Tensor.dim fh))
                                          (Tensor.dim fb₁)))
                                      (Tensor.dim fb₂)) i)
                              simp [_root_.Spec.get, _root_.Spec.getAtSpec,
                                _root_.Spec.Tensor.addSpec, _root_.Spec.Tensor.mulSpec,
                                _root_.Spec.Tensor.map2Spec,
                                _root_.Spec.Tensor.toScalar,
                                hfa₁, hfb₁, hfa₂, hfb₂, hfh,
                                mul_add, mul_assoc, add_assoc]

end DiagonalTransition

/-- Running appended transition lists is the same as running the first list, then the second. -/
theorem runScalarAffine_append {α : Type} [Semiring α] (h0 : α)
    (xs ys : List (_root_.Spec.ScalarAffineTransition α)) :
    _root_.Spec.runScalarAffine h0 (xs ++ ys) =
      _root_.Spec.runScalarAffine (_root_.Spec.runScalarAffine h0 xs) ys := by
  induction xs generalizing h0 with
  | nil =>
      simp [_root_.Spec.runScalarAffine]
  | cons tr rest ih =>
      simp [_root_.Spec.runScalarAffine, ih]

/-- Running a singleton transition list is the same as applying that transition. -/
@[simp] theorem runScalarAffine_singleton {α : Type} [Mul α] [Add α] (h0 : α)
    (tr : _root_.Spec.ScalarAffineTransition α) :
    _root_.Spec.runScalarAffine h0 [tr] = tr.apply h0 := by
  rfl

/--
The affine summary of a transition list denotes the same state as the sequential recurrence.

This is the core parallel-scan correctness theorem: a tree/parallel scan may compute the summary,
but applying that summary to the initial state is extensionally identical to recurrent execution.
-/
theorem summarizeScalarAffine_apply_eq_run {α : Type} [Semiring α] (h0 : α)
    (trs : List (_root_.Spec.ScalarAffineTransition α)) :
    (_root_.Spec.summarizeScalarAffine trs).apply h0 =
      _root_.Spec.runScalarAffine h0 trs := by
  induction trs generalizing h0 with
  | nil =>
      simp [_root_.Spec.summarizeScalarAffine, _root_.Spec.runScalarAffine,
        _root_.Spec.ScalarAffineTransition.id,
        _root_.Spec.ScalarAffineTransition.apply]
  | cons tr rest ih =>
      simp [_root_.Spec.summarizeScalarAffine, _root_.Spec.runScalarAffine, ih]

/--
Prefix summaries compose across list append in the expected order, at the level of denotation.

The order matters: `xs ++ ys` means "run `xs`, then run `ys`", so the summary is
`summary(ys) ∘ summary(xs)`.
-/
theorem summarizeScalarAffine_append_apply {α : Type} [Semiring α] (h0 : α)
    (xs ys : List (_root_.Spec.ScalarAffineTransition α)) :
    (_root_.Spec.summarizeScalarAffine (xs ++ ys)).apply h0 =
      (_root_.Spec.ScalarAffineTransition.compose
        (_root_.Spec.summarizeScalarAffine ys)
        (_root_.Spec.summarizeScalarAffine xs)).apply h0 := by
  rw [summarizeScalarAffine_apply_eq_run]
  rw [ScalarAffineTransition.compose_apply]
  rw [summarizeScalarAffine_apply_eq_run, summarizeScalarAffine_apply_eq_run]
  exact runScalarAffine_append h0 xs ys

/--
The affine summary of an appended list factors as `summary(ys) ∘ summary(xs)`.

Unlike `summarizeScalarAffine_append_apply`, this is equality of the summary transitions
themselves.  It is the algebraic contract needed by parallel prefix-scan implementations.
-/
theorem summarizeScalarAffine_append {α : Type} [Semiring α]
    (xs ys : List (_root_.Spec.ScalarAffineTransition α)) :
    _root_.Spec.summarizeScalarAffine (xs ++ ys) =
      _root_.Spec.ScalarAffineTransition.compose
        (_root_.Spec.summarizeScalarAffine ys)
        (_root_.Spec.summarizeScalarAffine xs) := by
  induction xs with
  | nil =>
      simp [_root_.Spec.summarizeScalarAffine]
  | cons tr rest ih =>
      simp [_root_.Spec.summarizeScalarAffine, ih,
        ScalarAffineTransition.compose_assoc]

/-- The scan output list has exactly one state per transition. -/
@[simp] theorem scalarAffineScan_length {α : Type} [Mul α] [Add α] (h0 : α)
    (trs : List (_root_.Spec.ScalarAffineTransition α)) :
    (_root_.Spec.scalarAffineScan h0 trs).length = trs.length := by
  induction trs generalizing h0 with
  | nil =>
      simp [_root_.Spec.scalarAffineScan]
  | cons tr rest ih =>
      simp [_root_.Spec.scalarAffineScan, ih]

/--
Scanning appended transition lists is equivalent to scanning the prefix, then continuing the scan
from the recurrent state reached after the prefix.
-/
theorem scalarAffineScan_append {α : Type} [Mul α] [Add α] (h0 : α)
    (xs ys : List (_root_.Spec.ScalarAffineTransition α)) :
    _root_.Spec.scalarAffineScan h0 (xs ++ ys) =
      _root_.Spec.scalarAffineScan h0 xs ++
        _root_.Spec.scalarAffineScan (_root_.Spec.runScalarAffine h0 xs) ys := by
  induction xs generalizing h0 with
  | nil =>
      simp [_root_.Spec.scalarAffineScan, _root_.Spec.runScalarAffine]
  | cons tr rest ih =>
      simp [_root_.Spec.scalarAffineScan, _root_.Spec.runScalarAffine, ih]

/-- The diagonal tensor scan output list has exactly one state per transition. -/
@[simp] theorem diagonalSelectiveScan_length {α : Type} [Add α] [Mul α] {stateDim : Nat}
    (h0 : _root_.Spec.Tensor α (.dim stateDim .scalar))
    (trs : List (_root_.Spec.DiagonalTransition α stateDim)) :
    (_root_.Spec.diagonalSelectiveScan h0 trs).length = trs.length := by
  induction trs generalizing h0 with
  | nil =>
      simp [_root_.Spec.diagonalSelectiveScan]
  | cons tr rest ih =>
      simp [_root_.Spec.diagonalSelectiveScan, ih]

/-- Running appended diagonal transition lists factors through the recurrent state after the prefix. -/
theorem runDiagonalTransitions_append {α : Type} [Add α] [Mul α] {stateDim : Nat}
    (h0 : _root_.Spec.Tensor α (.dim stateDim .scalar))
    (xs ys : List (_root_.Spec.DiagonalTransition α stateDim)) :
    _root_.Spec.runDiagonalTransitions h0 (xs ++ ys) =
      _root_.Spec.runDiagonalTransitions (_root_.Spec.runDiagonalTransitions h0 xs) ys := by
  induction xs generalizing h0 with
  | nil =>
      simp [_root_.Spec.runDiagonalTransitions]
  | cons tr rest ih =>
      simp [_root_.Spec.runDiagonalTransitions, ih]

/--
The diagonal selective-scan output over an appended sequence factors into:

1. the scan outputs for the prefix, followed by
2. the scan outputs for the suffix started from the recurrent state reached after the prefix.

This is the sequence-level contract behind causal/chunked Mamba execution: scanning a longer
sequence cannot change the already-emitted prefix states.
-/
theorem diagonalSelectiveScan_append {α : Type} [Add α] [Mul α] {stateDim : Nat}
    (h0 : _root_.Spec.Tensor α (.dim stateDim .scalar))
    (xs ys : List (_root_.Spec.DiagonalTransition α stateDim)) :
    _root_.Spec.diagonalSelectiveScan h0 (xs ++ ys) =
      _root_.Spec.diagonalSelectiveScan h0 xs ++
        _root_.Spec.diagonalSelectiveScan (_root_.Spec.runDiagonalTransitions h0 xs) ys := by
  induction xs generalizing h0 with
  | nil =>
      simp [_root_.Spec.diagonalSelectiveScan, _root_.Spec.runDiagonalTransitions]
  | cons tr rest ih =>
      simp [_root_.Spec.diagonalSelectiveScan, _root_.Spec.runDiagonalTransitions, ih]

/--
A homogeneous affine transition over `ℝ` is Lipschitz with factor `ρ` whenever `|a| ≤ ρ`.

This is the one-channel stability lemma used to lift diagonal SSMs into contraction proofs.
-/
theorem abs_homogeneous_apply_le (a ρ h : ℝ) (ha : |a| ≤ ρ) :
    |(_root_.Spec.ScalarAffineTransition.apply { a := a, b := 0 } h)| ≤ ρ * |h| := by
  calc
    |(_root_.Spec.ScalarAffineTransition.apply { a := a, b := 0 } h)|
        = |a * h| := by simp [_root_.Spec.ScalarAffineTransition.apply]
    _ = |a| * |h| := by rw [abs_mul]
    _ ≤ ρ * |h| := mul_le_mul_of_nonneg_right ha (abs_nonneg h)

/-- A homogeneous scalar transition with `|a| ≤ 1` is non-expansive. -/
theorem abs_homogeneous_apply_le_self (a h : ℝ) (ha : |a| ≤ 1) :
    |(_root_.Spec.ScalarAffineTransition.apply { a := a, b := 0 } h)| ≤ |h| := by
  calc
    |(_root_.Spec.ScalarAffineTransition.apply { a := a, b := 0 } h)| ≤ 1 * |h| :=
      abs_homogeneous_apply_le a 1 h ha
    _ = |h| := by simp

end StateSpace
end MLTheory
end NN

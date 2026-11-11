/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.NN.Op.Layout
public import TorchLean.NN.Op.Pointwise
public import TorchLean.NN.Op.Select
meta import TorchLean.Tensor.Op.Cert

/-!
# `op_cert` registers a derivative certificate only in the canonical form

`@[op_cert Op.<operator> derivative]` registers a theorem only when its statement is
`TorchLean.DerivCert f x δ v` for the operator's forward `f` in the listed inputs, a point `x` of
them, a cotangent `δ` of its value outputs and its pullback `v` of `δ`, with domain premises only
(`TorchLean.Tensor.Op.Cert`). Each theorem below is true, mentions the operator's rule, and is
rejected with the error shown: the review's tautology `backward = backward`, a statement about the
forward alone, the adjoint equation of a constant map, the certificate of `exp` at the point `0`,
the certificate of `neg` with the cotangent at the input, the certificate of `pad` in its input
registered at its padding value, the same certificate without its differentiability, the
certificate of `select` registered at its Boolean condition too, the certificate of `relu` with a
premise that is `False`, mentions the cotangent, fixes the input or is refuted by `simp_all`, the
certificate of `relu` under a variable of an empty type, also when the shape is built from it, the
certificate of `exp` under a variable whose type fixes the cotangent and the input, and the
certificates of `reshape` whose proof argument is false or built from the premise `False`. A
`run_meta` checks that none of them is registered.

The positive controls register under new names the certificate of `relu` stated with `DerivCert`,
the certificate of `exp`, which gives its derivative, the certificate of `cat`, and the
certificate of `pad` in its input obtained from the joint one by `DerivCert.comp`. The `run_meta`
checks what the check found in them and in the library's certificates of `cat` and `pad`.
-/

@[expose] public section

namespace TorchLeanTest.OpCert

open TorchLean

variable {s : Shape}

/-! ## Statements that are rejected -/

/--
error: op_cert: TorchLeanTest.OpCert.reluBackwardRefl is not a derivative certificate of
TorchLean.Op.relu at the inputs [0]: its statement is not `DerivCert f x δ v` with `f` the
operator's forward in these inputs, `x` a point of them, `δ` a cotangent of its value outputs and
`v` its pullback of `δ` at these inputs
-/
#guard_msgs in
/-- The review's tautology: the backward of `relu` is itself. -/
@[op_cert TorchLean.Op.relu derivative]
theorem reluBackwardRefl :
    ((Op.relu (s := s)).rule? (α := ℝ) |>.get rfl).backward =
      ((Op.relu (s := s)).rule? (α := ℝ) |>.get rfl).backward := rfl

/--
error: op_cert: TorchLeanTest.OpCert.reluForwardRefl is not a derivative certificate of
TorchLean.Op.relu at the inputs [0]: its statement is not `DerivCert f x δ v` with `f` the
operator's forward in these inputs, `x` a point of them, `δ` a cotangent of its value outputs and
`v` its pullback of `δ` at these inputs
-/
#guard_msgs in
/-- The forward of `relu` is itself. -/
@[op_cert TorchLean.Op.relu derivative]
theorem reluForwardRefl :
    ((Op.relu (s := s)).rule? (α := ℝ) |>.get rfl).forward =
      ((Op.relu (s := s)).rule? (α := ℝ) |>.get rfl).forward := rfl

/--
error: op_cert: TorchLeanTest.OpCert.reluConstant is not a derivative certificate of
TorchLean.Op.relu at the inputs [0]: its statement is not `DerivCert f x δ v` with `f` the
operator's forward in these inputs, `x` a point of them, `δ` a cotangent of its value outputs and
`v` its pullback of `δ` at these inputs
-/
#guard_msgs in
/-- The constant map with the value of `relu` at `x` is differentiable, and the adjoint of its
derivative sends every cotangent to zero. -/
@[op_cert TorchLean.Op.relu derivative]
theorem reluConstant (x δ : Tensor ℝ s) :
    DifferentiableAt ℝ (fun _ : Tensor ℝ s =>
        (((Op.relu (s := s)).rule? (α := ℝ) |>.get rfl).forward (x, ())).1) x ∧
      0 = (fderiv ℝ (fun _ : Tensor ℝ s =>
        (((Op.relu (s := s)).rule? (α := ℝ) |>.get rfl).forward (x, ())).1) x).adjoint δ :=
  ⟨differentiableAt_const _, by simp⟩

/--
error: op_cert: TorchLeanTest.OpCert.expAtZero is not a derivative certificate of TorchLean.Op.exp
at the inputs [0]: input 0 is 0, not a variable of the statement
-/
#guard_msgs in
/-- The certificate of `exp` at the point `0`. -/
@[op_cert TorchLean.Op.exp derivative]
theorem expAtZero (δ : Tensor ℝ s) :
    DifferentiableAt ℝ (fun y =>
        (((Op.exp (s := s)).rule? (α := ℝ) |>.get rfl).forward (y, ())).1) 0 ∧
      (((Op.exp (s := s)).rule? (α := ℝ) |>.get rfl).backward (0, ()) (δ, ())).1 =
        (fderiv ℝ (fun y =>
          (((Op.exp (s := s)).rule? (α := ℝ) |>.get rfl).forward (y, ())).1) 0).adjoint δ :=
  ⟨(Op.exp_backward_eq_adjoint 0 δ).1.differentiableAt, (Op.exp_backward_eq_adjoint 0 δ).2⟩

/--
error: op_cert: TorchLeanTest.OpCert.negAtInput is not a derivative certificate of TorchLean.Op.neg
at the inputs [0]: the cotangent is x, a variable already used for another input
-/
#guard_msgs in
/-- The certificate of `neg` with the cotangent at the input. -/
@[op_cert TorchLean.Op.neg derivative]
theorem negAtInput (x : Tensor ℝ s) : type_of% (Op.neg_backward_eq_adjoint x x) :=
  Op.neg_backward_eq_adjoint x x

/--
error: op_cert: TorchLeanTest.OpCert.padInputAtValue is not a derivative certificate of
TorchLean.Op.pad at the inputs [1]: its statement is not `DerivCert f x δ v` with `f` the
operator's forward in these inputs, `x` a point of them, `δ` a cotangent of its value outputs and
`v` its pullback of `δ` at these inputs
-/
#guard_msgs in
/-- The certificate of `pad` in its input, registered at its padding value. -/
@[op_cert TorchLean.Op.pad derivative 1]
theorem padInputAtValue (low high interior : Fin s.rank → Nat) (value : Tensor ℝ [])
    (x : Tensor ℝ s) (δ : Tensor ℝ (s.pad low high interior)) :
    type_of% (Op.pad_backward_eq_adjoint low high interior value x δ) :=
  Op.pad_backward_eq_adjoint low high interior value x δ

/--
error: op_cert: TorchLeanTest.OpCert.padInputAdjoint is not a derivative certificate of
TorchLean.Op.pad at the inputs [0]: its statement is not `DerivCert f x δ v` with `f` the
operator's forward in these inputs, `x` a point of them, `δ` a cotangent of its value outputs and
`v` its pullback of `δ` at these inputs
-/
#guard_msgs in
/-- The pullback of `pad` to its input is the adjoint of the derivative, without the
differentiability of the forward. -/
@[op_cert TorchLean.Op.pad derivative 0]
theorem padInputAdjoint (low high interior : Fin s.rank → Nat) (value : Tensor ℝ [])
    (x : Tensor ℝ s) (δ : Tensor ℝ (s.pad low high interior)) :
    type_of% (Op.pad_backward_eq_adjoint low high interior value x δ).2 :=
  (Op.pad_backward_eq_adjoint low high interior value x δ).2

/--
error: op_cert: TorchLeanTest.OpCert.selectAtCondition is not a derivative certificate of
TorchLean.Op.select at the inputs [0, 1, 2]: input 0 is a Port.Kind.bool port, not a value port
-/
#guard_msgs in
/-- The certificate of `select` in its two value inputs, registered at its Boolean condition
too. -/
@[op_cert TorchLean.Op.select derivative 0 1 2]
theorem selectAtCondition (cond : Tensor Bool s) (x y δ : Tensor ℝ s) :
    type_of% (select_prod_backward_eq_adjoint cond x y δ) :=
  select_prod_backward_eq_adjoint cond x y δ

/--
error: op_cert: TorchLeanTest.OpCert.reluFalse is not a derivative certificate of TorchLean.Op.relu
at the inputs [0]: the premise h is rejected by rule (c): it is False
-/
#guard_msgs in
/-- The certificate of `relu` under the premise `False`. -/
@[op_cert TorchLean.Op.relu derivative]
theorem reluFalse (x δ : Tensor ℝ s) (h : False) :
    type_of% (Op.relu_backward_eq_adjoint x δ h.elim) :=
  Op.relu_backward_eq_adjoint x δ h.elim

/--
error: op_cert: TorchLeanTest.OpCert.reluCotangent is not a derivative certificate of
TorchLean.Op.relu at the inputs [0]: the premise _hδ is rejected by rule (a): it mentions the
cotangent
-/
#guard_msgs in
/-- The certificate of `relu` under a premise on the cotangent. -/
@[op_cert TorchLean.Op.relu derivative]
theorem reluCotangent (x δ : Tensor ℝ s) (hx : ∀ c, x c ≠ 0) (_hδ : δ = 0) :
    type_of% (Op.relu_backward_eq_adjoint x δ hx) :=
  Op.relu_backward_eq_adjoint x δ hx

/--
error: op_cert: TorchLeanTest.OpCert.reluFixed is not a derivative certificate of TorchLean.Op.relu
at the inputs [0]: the premise _hx₀ is rejected by rule (c): it fixes a varying input
-/
#guard_msgs in
/-- The certificate of `relu` under a premise that fixes the input. -/
@[op_cert TorchLean.Op.relu derivative]
theorem reluFixed (x δ : Tensor ℝ s) (hx : ∀ c, x c ≠ 0) (_hx₀ : x = 0) :
    type_of% (Op.relu_backward_eq_adjoint x δ hx) :=
  Op.relu_backward_eq_adjoint x δ hx

/--
error: op_cert: TorchLeanTest.OpCert.reluRefuted is not a derivative certificate of
TorchLean.Op.relu at the inputs [0]: the premises [hx, h] are rejected by rule (d): `first |
decide | simp_all | omega` refutes them
-/
#guard_msgs in
/-- The certificate of `relu` under a premise that has no solution. -/
@[op_cert TorchLean.Op.relu derivative]
theorem reluRefuted (x δ : Tensor ℝ s) (hx : ∀ c, x c ≠ 0) (h : s.rank < 0) :
    type_of% (Op.relu_backward_eq_adjoint x δ hx) :=
  absurd h (Nat.not_lt_zero _)

/--
error: op_cert: TorchLeanTest.OpCert.reluEmpty is not a derivative certificate of TorchLean.Op.relu
at the inputs [0]: the binder k is not a proposition, an argument of the operator, an input or the
cotangent
-/
#guard_msgs in
/-- The canonical proposition of `relu` at every input, under a variable of an empty type. -/
@[op_cert TorchLean.Op.relu derivative]
theorem reluEmpty (x δ : Tensor ℝ s) (k : Empty) :
    DerivCert (fun y => (((Op.relu (s := s)).rule? (α := ℝ) |>.get rfl).forward (y, ())).1) x δ
      (((Op.relu (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 :=
  k.elim

/--
error: op_cert: TorchLeanTest.OpCert.reluEmptyShape is not a derivative certificate of
TorchLean.Op.relu at the inputs [0]: the binder k is not a proposition, an argument of the
operator, an input or the cotangent
-/
#guard_msgs in
/-- The canonical proposition of `relu` at a shape built from a variable of an empty type. -/
@[op_cert TorchLean.Op.relu derivative]
theorem reluEmptyShape (k : Empty) (x δ : Tensor ℝ (k.elim : Shape)) :
    DerivCert (fun y => (((Op.relu (s := k.elim)).rule? (α := ℝ) |>.get rfl).forward (y, ())).1)
      x δ (((Op.relu (s := k.elim)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 :=
  k.elim

/--
error: op_cert: TorchLeanTest.OpCert.expPLift is not a derivative certificate of TorchLean.Op.exp
at the inputs [0]: the binder _k is not a proposition, an argument of the operator, an input or
the cotangent
-/
#guard_msgs in
/-- The certificate of `exp` under a variable whose type fixes the cotangent and the input. -/
@[op_cert TorchLean.Op.exp derivative]
theorem expPLift (x δ : Tensor ℝ s) (_k : PLift (δ = 0 ∧ x = 0)) :
    type_of% (Op.exp_backward_eq_adjoint x δ) :=
  Op.exp_backward_eq_adjoint x δ

/--
error: op_cert: TorchLeanTest.OpCert.reshapeFalse is not a derivative certificate of
TorchLean.Op.reshape at the inputs [0]: the premise h is rejected by rule (b): it mentions no
variable
-/
#guard_msgs in
/-- The canonical proposition of `reshape` from `[1]` to `[2]`, whose proof argument is false. -/
@[op_cert TorchLean.Op.reshape derivative]
theorem reshapeFalse (h : Shape.size [1] = Shape.size [2]) (x : Tensor ℝ [1])
    (δ : Tensor ℝ [2]) :
    DerivCert (fun y => (((Op.reshape (s := [1]) (t := [2]) h).rule? (α := ℝ) |>.get rfl).forward
        (y, ())).1) x δ
      (((Op.reshape (s := [1]) (t := [2]) h).rule? (α := ℝ) |>.get rfl).backward
        (x, ()) (δ, ())).1 :=
  absurd h (by decide)

/--
error: op_cert: TorchLeanTest.OpCert.reshapeVacuous is not a derivative certificate of
TorchLean.Op.reshape at the inputs [0]: the premise bad is rejected by rule (c): it is False
-/
#guard_msgs in
/-- The certificate of `reshape` whose proof argument is built from the premise `False`. -/
@[op_cert TorchLean.Op.reshape derivative]
theorem reshapeVacuous (a b : Shape) (bad : False) (x : Tensor ℝ a) (δ : Tensor ℝ b) :
    type_of% (reshape_backward_eq_adjoint (False.elim bad) x δ) :=
  False.elim bad

/-! ## Statements that are registered -/

/-- Away from zero, `relu` is differentiable and its pullback is the adjoint of the derivative,
stated with `DerivCert`. -/
@[op_cert TorchLean.Op.relu derivative]
theorem reluCert (x δ : Tensor ℝ s) (hx : ∀ c, x c ≠ 0) :
    DerivCert (fun y => (((Op.relu (s := s)).rule? (α := ℝ) |>.get rfl).forward (y, ())).1) x δ
      (((Op.relu (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 :=
  ⟨(Op.relu_backward_eq_adjoint x δ hx).1.differentiableAt,
    (Op.relu_backward_eq_adjoint x δ hx).2⟩

/-- The certificate of `exp`, which gives its derivative. -/
@[op_cert TorchLean.Op.exp derivative]
theorem expCert (x δ : Tensor ℝ s) : type_of% (Op.exp_backward_eq_adjoint x δ) :=
  Op.exp_backward_eq_adjoint x δ

/-- The certificate of `cat` along `dim`, jointly in its inputs. -/
@[op_cert TorchLean.Op.cat derivative]
theorem catCert (dim : Fin s.rank) (sizes : List Nat)
    (xs : Values ℝ (sizes.map fun n => .value (s.set dim n)))
    (δ : Tensor ℝ (s.set dim sizes.sum)) :
    type_of% (cat_backward_eq_adjoint dim sizes xs δ) :=
  cat_backward_eq_adjoint dim sizes xs δ

/-- The forward of `pad` is differentiable in its input, the padding value fixed, and its pullback
to the input is the adjoint of the derivative: the joint certificate precomposed with
`u ↦ (u, value)` by `DerivCert.comp`. -/
@[op_cert TorchLean.Op.pad derivative 0]
theorem padInputCert (low high interior : Fin s.rank → Nat) (value : Tensor ℝ [])
    (x : Tensor ℝ s) (δ : Tensor ℝ (s.pad low high interior)) :
    DerivCert (fun y => (((Op.pad (s := s) low high interior).rule? (α := ℝ) |>.get rfl).forward
        (y, value, ())).1) x δ
      (((Op.pad (s := s) low high interior).rule? (α := ℝ) |>.get rfl).backward
        (x, value, ()) (δ, ())).1 := by
  have h := DerivCert.comp (g := fun u => WithLp.toLp 2 (u, value)) (y := x)
    (pad_prod_backward_eq_adjoint low high interior x value δ)
    ((HasPullbackAt.id x).toLp_prod (HasPullbackAt.const value x))
  rw [add_zero] at h
  exact h

/-! ## What was registered

None of the rejected theorems is registered, and the check found the expected varying positions,
domain premises and form in the controls and in the library's certificates of `cat` and `pad`. -/

open Lean in
run_meta do
  let env ← getEnv
  for name in [``reluBackwardRefl, ``reluForwardRefl, ``reluConstant, ``expAtZero,
      ``negAtInput, ``padInputAtValue, ``padInputAdjoint, ``selectAtCondition, ``reluFalse,
      ``reluCotangent, ``reluFixed, ``reluRefuted, ``reluEmpty, ``reluEmptyShape, ``expPLift,
      ``reshapeFalse, ``reshapeVacuous] do
    if (Op.certExtension.getState env).any (·.certificate == name) ||
        (Op.derivEntry? env name).isSome then
      throwError "{name} is registered"
  for (name, varying, premises, strong) in [(``reluCert, [0], [`hx], false),
      (``expCert, [0], [], true), (``catCert, [], [], false), (``padInputCert, [0], [], false),
      (``cat_backward_eq_adjoint, [], [], false),
      (``Op.pad_backward_eq_adjoint, [0], [], false),
      (``Op.pad_value_backward_eq_adjoint, [1], [], false)] do
    let some entry := Op.derivEntry? env name | throwError "{name} is not registered"
    unless entry.varying == varying && entry.premises == premises && entry.strong == strong do
      throwError "{name}: the check found the positions {toString entry.varying}, the premises \
        {toString entry.premises} and the strong form {entry.strong}"

end TorchLeanTest.OpCert

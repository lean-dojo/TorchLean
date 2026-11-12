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
them, a cotangent `δ` of its value outputs and its pullback `v` of `δ`, under premises that pass
the rules of `TorchLean.Tensor.Op.Cert`. Each theorem below is true, mentions the operator's rule,
and is rejected with the error shown. The last check confirms that none of them is registered and
that the check finds the expected varying positions, domain premises and form in the library's
certificates of `relu`, `exp`, `cat` and `pad`.
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
/-- The backward of `relu` equals itself. -/
@[op_cert TorchLean.Op.relu derivative]
theorem reluBackwardRefl :
    ((Op.relu (s := s)).rule? (α := ℝ) |>.get rfl).backward =
      ((Op.relu (s := s)).rule? (α := ℝ) |>.get rfl).backward := rfl

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
TorchLean.Op.relu at the inputs [0]: the premises [hx, h₁, h₂] are rejected by rule (d):
`simp_all` or `omega` refutes them
-/
#guard_msgs in
/-- The certificate of `relu` under two premises on the rank that contradict each other. -/
@[op_cert TorchLean.Op.relu derivative]
theorem reluRefuted (x δ : Tensor ℝ s) (hx : ∀ c, x c ≠ 0) (h₁ : s.rank < 3) (h₂ : 5 < s.rank) :
    type_of% (Op.relu_backward_eq_adjoint x δ hx) :=
  absurd h₂ (by omega)

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

/-! ## What is registered -/

open Lean in
run_meta do
  let env ← getEnv
  for name in [``reluBackwardRefl, ``expAtZero, ``negAtInput, ``padInputAdjoint,
      ``selectAtCondition, ``reluCotangent, ``reluFixed, ``reluRefuted, ``reluEmptyShape,
      ``expPLift, ``reshapeFalse, ``reshapeVacuous] do
    if (Op.certExtension.getState env).any (·.certificate == name) ||
        (Op.derivEntry? env name).isSome then
      throwError "{name} is registered"
  for (name, varying, premises, strong) in [(``Op.relu_backward_eq_adjoint, [0], [`hx], true),
      (``Op.exp_backward_eq_adjoint, [0], [], true), (``cat_backward_eq_adjoint, [], [], false),
      (``Op.pad_backward_eq_adjoint, [0], [], false),
      (``Op.pad_value_backward_eq_adjoint, [1], [], false)] do
    let some entry := Op.derivEntry? env name | throwError "{name} is not registered"
    unless entry.varying == varying && entry.premises == premises && entry.strong == strong do
      throwError "{name}: the check found the positions {toString entry.varying}, the premises \
        {toString entry.premises} and the strong form {entry.strong}"

end TorchLeanTest.OpCert

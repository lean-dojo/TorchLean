/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.DAG.Syntax
import Mathlib.Algebra.Order.Algebra

/-!
# Pure semantics of GraphSpec DAGs

This module interprets typed DAG terms and blocks in `TorchLean.TensorPack` environments and
proves that evaluation respects renaming, substitution, inlining, and block composition.
-/

@[expose] public section

namespace NN
namespace GraphSpec
namespace DAG

open Spec TorchLean
open TorchLean.Tensor

namespace Env

open Runtime.Autograd.Torch

/-- Typed environment lookup for pure tensors. -/
def tget {α : Type} [TorchLean.Storage α] :
    {Γ : List Shape} → {s : Shape} → TorchLean.TensorPack α Γ → Var Γ s →
      TorchLean.Tensor α s
  | _ :: _, _, .cons x _, .head => x
  | _ :: _, _, .cons _ xs, .tail i => tget xs i

/-- Looking up a programmatically selected variable agrees with typed-list lookup. -/
@[simp] theorem tget_ofFin {α : Type} [TorchLean.Storage α] {Γ : List Shape}
    (env : TorchLean.TensorPack α Γ)
    (i : Fin Γ.length) :
    tget env (Var.ofFin i) = TorchLean.TensorPack.get env i := by
  induction Γ with
  | nil => exact Fin.elim0 i
  | cons shape Γ ih =>
      cases env with
      | cons value rest =>
          refine Fin.cases ?_ ?_ i
          · rfl
          · intro j
            change tget rest (Var.ofFin j) = TorchLean.TensorPack.get rest j
            exact ih rest j

/-- Appending a value does not change the meaning of an existing variable. -/
@[simp] theorem tget_append_weakenRight {α : Type} [TorchLean.Storage α]
    {Γ : List Shape} {s t : Shape}
    (env : TorchLean.TensorPack α Γ) (value : TorchLean.Tensor α t) (i : Var Γ s) :
    tget (TorchLean.TensorPack.append env (.cons value .nil))
        (Var.weakenRight i) = tget env i := by
  induction i with
  | head => cases env; rfl
  | tail i ih =>
      cases env with
      | cons _ rest => exact ih rest

/-- Looking up a variable embedded from the left reads the original left environment. -/
@[simp] theorem tget_append_inLeft {α : Type} [TorchLean.Storage α]
    {Γ Δ : List Shape} {s : Shape}
    (left : TorchLean.TensorPack α Γ) (right : TorchLean.TensorPack α Δ) (v : Var Γ s) :
    tget (TorchLean.TensorPack.append left right) (Var.inLeft Δ v) = tget left v := by
  induction v with
  | head => cases left; rfl
  | tail v ih =>
      cases left with
      | cons _ rest => exact ih rest

/-- Looking up a variable embedded from the right reads the appended right environment. -/
@[simp] theorem tget_append_inRight {α : Type} [TorchLean.Storage α]
    {Γ Δ : List Shape} {s : Shape}
    (left : TorchLean.TensorPack α Γ) (right : TorchLean.TensorPack α Δ) (v : Var Δ s) :
    tget (TorchLean.TensorPack.append left right) (Var.inRight Γ v) = tget right v := by
  induction Γ with
  | nil => cases left; rfl
  | cons shape Γ ih =>
      cases left with
      | cons _ rest => exact ih rest

/-- Looking up the final variable returns the value most recently appended to an environment. -/
@[simp] theorem tget_append_last {α : Type} [TorchLean.Storage α]
    {Γ : List Shape} {s : Shape}
    (env : TorchLean.TensorPack α Γ) (value : TorchLean.Tensor α s) :
    tget (TorchLean.TensorPack.append env (.cons value .nil)) (Var.last Γ) = value := by
  induction Γ with
  | nil => cases env; rfl
  | cons _ Γ ih =>
      cases env with
      | cons _ rest => exact ih rest


end Env

namespace Term

open Runtime.Autograd.Torch

/-! ### Spec interpreter -/

mutual
  /-- Evaluate a typed argument list by evaluating each component term under the same environment.
    -/
  def evalArgs
      {Γ : List Shape} {ins : List Shape}
      {α : Type 0} [TorchLean.Storage α] [Context α]
      (env : TorchLean.TensorPack α Γ) :
      Args Γ ins → TorchLean.TensorPack α ins
    | .nil => .nil
    | .cons t ts => .cons (eval (Γ := Γ) (α := α) env t) (evalArgs (Γ := Γ) (α := α) env ts)

  /--
  Pure evaluation of a DAG term.

  This is the “math-first” semantics: we interpret a term as a pure function on tensors.
  No monads, no mutation, no autograd tape, just the Spec definitions of primitives.

  The key runtime discipline is the environment discipline:

  - `var` reads from `env`,
  - `op` evaluates its arguments and feeds them to the primitive’s `specFwd`,
  - `let1` evaluates the bound term once and extends `env` for the body.
  -/
  def eval
      {Γ : List Shape} {τ : Shape}
      {α : Type 0} [TorchLean.Storage α] [Context α]
      (env : TorchLean.TensorPack α Γ) :
      Term Γ τ → TorchLean.Tensor α τ
    | .var i => Env.tget (α := α) env i
    | .cast t h =>
        match h with
        | rfl => eval (Γ := Γ) (α := α) env t
    | .castEnv t h =>
        -- `h : Γ_src = Γ_tgt` comes from term construction/lowering. For evaluation we want to
        -- *rewrite the term* to the current environment `Γ_tgt`, not rewrite the environment
        -- to the source `Γ_src`, so we match on `h.symm`.
        match h.symm with
        | rfl => eval (Γ := Γ) (α := α) env t
    | .op p args =>
        let xs := evalArgs (Γ := Γ) (ins := _) (α := α) env args
        p.specFwd (α := α) xs
    | .let1 (σ := σ) t body =>
        let v := eval (Γ := Γ) (α := α) env t
        let env' : TorchLean.TensorPack α (Γ ++ [σ]) :=
          TorchLean.TensorPack.append (α := α) (ss₁ := Γ) (ss₂ := [σ]) env (.cons v .nil)
        eval (Γ := Γ ++ [σ]) (α := α) env' body
end

/-- Evaluating an operation node first evaluates its typed arguments, then applies the
primitive's pure semantics. -/
@[simp] theorem eval_op {Γ ins : List Shape} {τ : Shape} {α : Type 0} [TorchLean.Storage α]
    [Context α]
    (env : TorchLean.TensorPack α Γ) (primitive : PrimOp ins τ) (args : Args Γ ins) :
    eval env (.op primitive args) = primitive.specFwd (evalArgs env args) := by
  rfl

/-- Evaluating an output-shape cast transports the value along the same shape equality. -/
@[simp] theorem eval_cast {Γ : List Shape} {σ τ : Shape} {α : Type 0} [TorchLean.Storage α]
    [Context α]
    (env : TorchLean.TensorPack α Γ) (term : Term Γ σ) (h : σ = τ) :
    eval env (.cast term h) = h ▸ eval env term := by
  cases h
  rfl

/-! ### Semantics of variable renaming -/

end Term

namespace Env

/-- A variable renaming preserves an environment when every renamed lookup has the same value. -/
def RenamingSound {α : Type} [TorchLean.Storage α] {Γ Δ : List Shape}
    (envΓ : TorchLean.TensorPack α Γ)
    (envΔ : TorchLean.TensorPack α Δ)
    (ρ : {s : Shape} → Var Γ s → Var Δ s) : Prop :=
  ∀ {s : Shape} (v : Var Γ s), tget envΔ (ρ v) = tget envΓ v

/-- A sound renaming remains sound when the same value is appended to both environments. -/
theorem RenamingSound.lift_right {α : Type} [TorchLean.Storage α]
    {Γ Δ : List Shape} {t : Shape}
    {envΓ : TorchLean.TensorPack α Γ}
    {envΔ : TorchLean.TensorPack α Δ}
    {ρ : {s : Shape} → Var Γ s → Var Δ s}
    (hρ : RenamingSound envΓ envΔ ρ) (value : TorchLean.Tensor α t) :
    RenamingSound
      (TorchLean.TensorPack.append envΓ (.cons value .nil))
      (TorchLean.TensorPack.append envΔ (.cons value .nil))
      (Var.liftRight ρ) := by
  unfold RenamingSound
  intro s v
  induction Γ with
  | nil =>
      cases envΓ
      cases v with
      | head => exact tget_append_last envΔ value
      | tail impossible => nomatch impossible
  | cons shape Γ ih =>
      cases envΓ with
      | cons head tail =>
          cases v with
          | head =>
              change
                tget (TorchLean.TensorPack.append envΔ (.cons value .nil))
                    (Var.weakenRight (ρ .head)) =
                  tget
                    (TorchLean.TensorPack.append (.cons head tail)
                      (.cons value .nil)) .head
              calc
                _ = tget envΔ (ρ .head) := tget_append_weakenRight envΔ value (ρ .head)
                _ = tget (.cons head tail) .head := hρ .head
                _ = _ := (tget_append_weakenRight (.cons head tail) value .head).symm
          | tail v =>
              change
                tget (TorchLean.TensorPack.append envΔ (.cons value .nil))
                    (Var.liftRight (fun w => ρ (.tail w)) v) =
                  tget (TorchLean.TensorPack.append tail (.cons value .nil)) v
              exact ih (envΓ := tail) (ρ := fun w => ρ (.tail w))
                (fun w => hρ (.tail w)) v

end Env

namespace Term

mutual
  private def complexity {Γ : List Shape} {s : Shape} : Term Γ s → Nat
    | .var _ => 1
    | .cast term _ => complexity term + 1
    | .castEnv term _ => complexity term + 1
    | .op _ args => argsComplexity args + 1
    | .let1 value body => complexity value + complexity body + 1

  private def argsComplexity {Γ ins : List Shape} : Args Γ ins → Nat
    | .nil => 1
    | .cons term rest => complexity term + argsComplexity rest + 1
end

mutual
  /-- Pure evaluation commutes with a sound renaming of an operation's arguments. -/
  theorem evalArgs_rename
      {Γ Δ ins : List Shape} {α : Type 0} [TorchLean.Storage α] [Context α]
      (envΓ : TorchLean.TensorPack α Γ)
      (envΔ : TorchLean.TensorPack α Δ)
      (ρ : {s : Shape} → Var Γ s → Var Δ s)
      (hρ : Env.RenamingSound envΓ envΔ ρ) :
      (args : Args Γ ins) →
        evalArgs envΔ (Args.rename ρ args) = evalArgs envΓ args
    | .nil => rfl
    | .cons term rest => by
        simp only [Args.rename, evalArgs]
        rw [eval_rename envΓ envΔ ρ hρ term,
          evalArgs_rename envΓ envΔ ρ hρ rest]
  termination_by args => argsComplexity args

  decreasing_by all_goals (simp only [argsComplexity]; omega)

  /-- Pure evaluation commutes with any variable renaming that preserves environment lookup. -/
  theorem eval_rename
      {Γ Δ : List Shape} {s : Shape} {α : Type 0} [TorchLean.Storage α] [Context α]
      (envΓ : TorchLean.TensorPack α Γ)
      (envΔ : TorchLean.TensorPack α Δ)
      (ρ : {t : Shape} → Var Γ t → Var Δ t)
      (hρ : Env.RenamingSound envΓ envΔ ρ) :
      (term : Term Γ s) →
        eval envΔ (Term.rename ρ term) = eval envΓ term
    | .var v => hρ v
    | .cast term rfl => eval_rename envΓ envΔ ρ hρ term
    | .castEnv term rfl => eval_rename envΓ envΔ ρ hρ term
    | .op primitive args => by
        simp only [Term.rename, eval]
        rw [evalArgs_rename envΓ envΔ ρ hρ args]
    | .let1 value body => by
        simp only [Term.rename, eval]
        rw [eval_rename envΓ envΔ ρ hρ value]
        exact eval_rename
          (TorchLean.TensorPack.append envΓ
            (.cons (eval envΓ value) .nil))
          (TorchLean.TensorPack.append envΔ
            (.cons (eval envΓ value) .nil))
          (Var.liftRight ρ) (hρ.lift_right (eval envΓ value)) body
  termination_by term => complexity term
  decreasing_by all_goals (simp only [complexity]; omega)
end

/-- Renaming arguments into the left side of an appended environment preserves their values. -/
@[simp] theorem evalArgs_rename_inLeft
    {Γ Δ ins : List Shape} {α : Type 0} [TorchLean.Storage α] [Context α]
    (left : TorchLean.TensorPack α Γ)
    (right : TorchLean.TensorPack α Δ) (args : Args Γ ins) :
    evalArgs (TorchLean.TensorPack.append left right)
        (Args.rename (Var.inLeft Δ) args) = evalArgs left args := by
  apply evalArgs_rename left (TorchLean.TensorPack.append left right)
  intro shape position
  exact Env.tget_append_inLeft left right position

/-- Renaming arguments into the right side of an appended environment preserves their values. -/
@[simp] theorem evalArgs_rename_inRight
    {Γ Δ ins : List Shape} {α : Type 0} [TorchLean.Storage α] [Context α]
    (left : TorchLean.TensorPack α Γ)
    (right : TorchLean.TensorPack α Δ) (args : Args Δ ins) :
    evalArgs (TorchLean.TensorPack.append left right)
        (Args.rename (Var.inRight Γ) args) = evalArgs right args := by
  apply evalArgs_rename right (TorchLean.TensorPack.append left right)
  intro shape position
  exact Env.tget_append_inRight left right position

/-- Renaming a term into the left side of an appended environment preserves its value. -/
@[simp] theorem eval_rename_inLeft
    {Γ Δ : List Shape} {s : Shape} {α : Type 0} [TorchLean.Storage α] [Context α]
    (left : TorchLean.TensorPack α Γ)
    (right : TorchLean.TensorPack α Δ) (term : Term Γ s) :
    eval (TorchLean.TensorPack.append left right)
        (Term.rename (Var.inLeft Δ) term) = eval left term := by
  apply eval_rename left (TorchLean.TensorPack.append left right)
  intro shape position
  exact Env.tget_append_inLeft left right position

/-- Appending an arbitrary typed environment does not change a weakened term's value. -/
@[simp] theorem eval_weakenAppend
    {Γ Δ : List Shape} {s : Shape} {α : Type 0} [TorchLean.Storage α] [Context α]
    (left : TorchLean.TensorPack α Γ)
    (right : TorchLean.TensorPack α Δ) (term : Term Γ s) :
    eval (TorchLean.TensorPack.append left right)
        (Term.weakenAppend Δ term) = eval left term := by
  exact eval_rename_inLeft left right term

/-- Renaming a term into the right side of an appended environment preserves its value. -/
@[simp] theorem eval_rename_inRight
    {Γ Δ : List Shape} {s : Shape} {α : Type 0} [TorchLean.Storage α] [Context α]
    (left : TorchLean.TensorPack α Γ)
    (right : TorchLean.TensorPack α Δ) (term : Term Δ s) :
    eval (TorchLean.TensorPack.append left right)
        (Term.rename (Var.inRight Γ) term) = eval right term := by
  apply eval_rename right (TorchLean.TensorPack.append left right)
  intro shape position
  exact Env.tget_append_inRight left right position

/-- Appending an unrelated value does not change a term's pure meaning. -/
@[simp] theorem eval_weakenRight
    {Γ : List Shape} {s t : Shape} {α : Type 0} [TorchLean.Storage α] [Context α]
    (env : TorchLean.TensorPack α Γ) (value : TorchLean.Tensor α t)
    (term : Term Γ s) :
    eval (TorchLean.TensorPack.append env (.cons value .nil))
      (Term.weakenRight term) = eval env term := by
  apply eval_rename env
    (TorchLean.TensorPack.append env (.cons value .nil)) Var.weakenRight
  unfold Env.RenamingSound
  intro shape v
  exact Env.tget_append_weakenRight env value v

/-- The final variable in an extended environment denotes the value that was just appended. -/
@[simp] theorem eval_var_last_append
    {Γ : List Shape} {t : Shape} {α : Type 0} [TorchLean.Storage α] [Context α]
    (env : TorchLean.TensorPack α Γ) (value : TorchLean.Tensor α t) :
    eval (TorchLean.TensorPack.append env (.cons value .nil))
      (Term.var (Var.last Γ)) = value := by
  simpa only [eval] using Env.tget_append_last env value

end Term

namespace Env

/-- A term substitution represents an environment when every assigned term evaluates to the
value stored at the corresponding source variable. -/
def SubstitutionSound {α : Type} [TorchLean.Storage α] [Context α] {Γ Δ : List Shape}
    (envΓ : TorchLean.TensorPack α Γ)
    (envΔ : TorchLean.TensorPack α Δ)
    (σ : Substitution Γ Δ) : Prop :=
  ∀ {s : Shape} (v : Var Γ s), Term.eval envΔ (σ v) = Env.tget envΓ v

/-- A sound substitution remains sound across the value introduced by a `let` binding. -/
theorem SubstitutionSound.lift_right {α : Type} [TorchLean.Storage α] [Context α]
    {Γ Δ : List Shape} {t : Shape}
    {envΓ : TorchLean.TensorPack α Γ}
    {envΔ : TorchLean.TensorPack α Δ}
    {σ : Substitution Γ Δ} (hσ : SubstitutionSound envΓ envΔ σ)
    (value : TorchLean.Tensor α t) :
    SubstitutionSound
      (TorchLean.TensorPack.append envΓ (.cons value .nil))
      (TorchLean.TensorPack.append envΔ (.cons value .nil))
      (Substitution.liftRight σ) := by
  intro s v
  induction Γ with
  | nil =>
      cases envΓ
      cases v with
      | head =>
          simp only [Substitution.liftRight, Term.eval]
          exact Env.tget_append_last envΔ value
      | tail impossible => nomatch impossible
  | cons shape Γ ih =>
      cases envΓ with
      | cons head tail =>
          cases v with
          | head =>
              simp only [Substitution.liftRight]
              rw [Term.eval_weakenRight]
              exact hσ .head
          | tail v =>
              simp only [Substitution.liftRight]
              exact ih (envΓ := tail) (σ := fun w => σ (.tail w))
                (fun w => hσ (.tail w)) v

end Env

namespace Term

mutual
  /-- Pure evaluation commutes with a sound substitution of operation arguments. -/
  theorem evalArgs_substitute
      {Γ Δ inputs : List Shape} {α : Type 0} [TorchLean.Storage α] [Context α]
      (envΓ : TorchLean.TensorPack α Γ)
      (envΔ : TorchLean.TensorPack α Δ)
      (σ : Substitution Γ Δ) (hσ : Env.SubstitutionSound envΓ envΔ σ) :
      (args : Args Γ inputs) →
        evalArgs envΔ (Args.substitute σ args) = evalArgs envΓ args
    | .nil => rfl
    | .cons term rest => by
        simp only [Args.substitute, evalArgs]
        rw [eval_substitute envΓ envΔ σ hσ term,
          evalArgs_substitute envΓ envΔ σ hσ rest]
  termination_by args => argsComplexity args
  decreasing_by all_goals (simp only [argsComplexity]; omega)

  /-- Pure evaluation commutes with any term substitution representing the source environment. -/
  theorem eval_substitute
      {Γ Δ : List Shape} {s : Shape} {α : Type 0} [TorchLean.Storage α] [Context α]
      (envΓ : TorchLean.TensorPack α Γ)
      (envΔ : TorchLean.TensorPack α Δ)
      (σ : Substitution Γ Δ) (hσ : Env.SubstitutionSound envΓ envΔ σ) :
      (term : Term Γ s) →
        eval envΔ (Term.substitute σ term) = eval envΓ term
    | .var v => hσ v
    | .cast term rfl => eval_substitute envΓ envΔ σ hσ term
    | .castEnv term rfl => eval_substitute envΓ envΔ σ hσ term
    | .op primitive args => by
        simp only [Term.substitute, eval]
        rw [evalArgs_substitute envΓ envΔ σ hσ args]
    | .let1 value body => by
        simp only [Term.substitute, eval]
        rw [eval_substitute envΓ envΔ σ hσ value]
        exact eval_substitute
          (TorchLean.TensorPack.append envΓ
            (.cons (eval envΓ value) .nil))
          (TorchLean.TensorPack.append envΔ
            (.cons (eval envΓ value) .nil))
          (Substitution.liftRight σ) (hσ.lift_right (eval envΓ value)) body
  termination_by term => complexity term
  decreasing_by all_goals (simp only [complexity]; omega)
end

/-- Evaluating an argument selected by a typed variable agrees with lookup in the evaluated
argument environment. -/
@[simp] theorem eval_get {Γ Δ : List Shape} {s : Shape} {α : Type 0} [TorchLean.Storage α]
    [Context α]
    (env : TorchLean.TensorPack α Δ) (args : Args Δ Γ) (v : Var Γ s) :
    eval env (Args.get args v) = Env.tget (evalArgs env args) v := by
  induction v with
  | head => cases args; rfl
  | tail v ih =>
      cases args with
      | cons _ rest => exact ih rest

/-- Pure evaluation of an inlined term equals evaluation of the original term under the supplied
argument values. -/
theorem eval_instantiate
    {Γ Δ : List Shape} {s : Shape} {α : Type 0} [TorchLean.Storage α] [Context α]
    (env : TorchLean.TensorPack α Δ) (args : Args Δ Γ) (term : Term Γ s) :
    eval env (term.instantiate args) = eval (evalArgs env args) term := by
  apply eval_substitute (evalArgs env args) env (Args.get args)
  intro shape v
  exact eval_get env args v

/-- Evaluating concatenated graph arguments concatenates their tensor values in the same order. -/
theorem evalArgs_append
    {Γ left right : List Shape} {α : Type 0} [TorchLean.Storage α] [Context α]
    (env : TorchLean.TensorPack α Γ) :
    (leftArgs : Args Γ left) → (rightArgs : Args Γ right) →
      evalArgs env (Args.append leftArgs rightArgs) =
        TorchLean.TensorPack.append
          (evalArgs env leftArgs) (evalArgs env rightArgs)
  | .nil, _ => rfl
  | .cons term rest, rightArgs => by
      simp only [Args.append, evalArgs, TorchLean.TensorPack.append]
      rw [evalArgs_append env rest rightArgs]

/-- Evaluating a statically split argument list agrees with splitting its evaluated values. -/
theorem evalArgs_splitAppend
    {Γ left right : List Shape} {α : Type 0} [TorchLean.Storage α] [Context α]
    (env : TorchLean.TensorPack α Γ) (args : Args Γ (left ++ right)) :
    let argumentParts := Args.splitAppend args
    let valueParts := TorchLean.TensorPack.split (evalArgs env args)
    (evalArgs env argumentParts.1, evalArgs env argumentParts.2) = valueParts := by
  induction left with
  | nil => rfl
  | cons shape left ih =>
      cases args with
      | cons term rest =>
          simpa only [Args.splitAppend, evalArgs,
            TorchLean.TensorPack.split] using
            congrArg
              (fun parts =>
                (TorchLean.TensorPack.cons (eval env term) parts.1, parts.2))
              (ih rest)

/-- Evaluating the left part of a typed argument split returns the corresponding value prefix. -/
theorem evalArgs_splitAppend_fst
    {Γ left right : List Shape} {α : Type 0} [TorchLean.Storage α] [Context α]
    (env : TorchLean.TensorPack α Γ) (args : Args Γ (left ++ right)) :
    evalArgs env (Args.splitAppend args).1 =
      (TorchLean.TensorPack.split (evalArgs env args)).1 := by
  exact congrArg Prod.fst (evalArgs_splitAppend env args)

/-- Evaluating the right part of a typed argument split returns the corresponding value suffix. -/
theorem evalArgs_splitAppend_snd
    {Γ left right : List Shape} {α : Type 0} [TorchLean.Storage α] [Context α]
    (env : TorchLean.TensorPack α Γ) (args : Args Γ (left ++ right)) :
    evalArgs env (Args.splitAppend args).2 =
      (TorchLean.TensorPack.split (evalArgs env args)).2 := by
  exact congrArg Prod.snd (evalArgs_splitAppend env args)

/-- Prepending an unrelated value does not change a term's pure meaning. -/
@[simp] theorem eval_weakenLeft
    {Γ : List Shape} {s t : Shape} {α : Type 0} [TorchLean.Storage α] [Context α]
    (env : TorchLean.TensorPack α Γ) (value : TorchLean.Tensor α t)
    (term : Term Γ s) :
    eval (.cons value env) (Term.weakenLeft term) = eval env term := by
  apply eval_rename env (.cons value env) (fun v => .tail v)
  unfold Env.RenamingSound
  intro shape v
  rfl

/-- Prepending an unrelated value does not change a typed argument list's pure meaning. -/
@[simp] theorem evalArgs_weakenLeft
    {Γ ins : List Shape} {t : Shape} {α : Type 0} [TorchLean.Storage α] [Context α]
    (env : TorchLean.TensorPack α Γ) (value : TorchLean.Tensor α t) :
    (args : Args Γ ins) →
      evalArgs (.cons value env) (Args.weakenLeft args) = evalArgs env args
  | .nil => rfl
  | .cons term rest => by
      simp only [Args.weakenLeft, evalArgs, eval_weakenLeft,
        evalArgs_weakenLeft env value rest]
termination_by args => argsComplexity args
decreasing_by (simp only [argsComplexity]; omega)

/-- Evaluating all variables of an environment returns that environment in order. -/
@[simp] theorem evalArgs_vars
    {Γ : List Shape} {α : Type 0} [TorchLean.Storage α] [Context α]
    (env : TorchLean.TensorPack α Γ) :
    evalArgs env (Args.vars Γ) = env := by
  induction Γ with
  | nil => cases env; rfl
  | cons shape Γ ih =>
      cases env with
      | cons value rest =>
          simp only [Args.vars, evalArgs, eval, Env.tget, evalArgs_weakenLeft]
          rw [ih rest]

end Term

namespace Block

open Runtime.Autograd.Torch

/-- Reinterpret a block's result list along an equality of output shapes. -/
def castOutputs {Γ outputs outputs' : List Shape} (h : outputs = outputs') :
    Block Γ outputs → Block Γ outputs'
  | block => h ▸ block

/-- Evaluate a multi-output block, preserving sharing introduced by `let1`. -/
def eval {Γ outs : List Shape} {α : Type 0} [TorchLean.Storage α] [Context α]
    (env : TorchLean.TensorPack α Γ) : Block Γ outs → TorchLean.TensorPack α outs
  | .ret results => Term.evalArgs (Γ := Γ) (α := α) env results
  | .let1 (σ := σ) value body =>
      let result := Term.eval (Γ := Γ) (α := α) env value
      let env' : TorchLean.TensorPack α (Γ ++ [σ]) :=
        TorchLean.TensorPack.append
          (α := α) (ss₁ := Γ) (ss₂ := [σ]) env (.cons result .nil)
      eval (Γ := Γ ++ [σ]) (α := α) env' body

/-- Casting a block's output shapes casts its evaluated typed result by the same equality. -/
@[simp] theorem eval_castOutputs
    {Γ outputs outputs' : List Shape} {α : Type 0} [TorchLean.Storage α] [Context α]
    (env : TorchLean.TensorPack α Γ) (h : outputs = outputs') (block : Block Γ outputs) :
    eval env (castOutputs h block) = h ▸ eval env block := by
  subst outputs'
  rfl

/-- Substituting terms into a block preserves its pure multi-output semantics whenever the
substitution denotes the original environment. -/
theorem eval_substitute {Γ Δ outs : List Shape} {α : Type 0} [TorchLean.Storage α] [Context α]
    (envΓ : TorchLean.TensorPack α Γ) (envΔ : TorchLean.TensorPack α Δ)
    (σ : Substitution Γ Δ) (hσ : Env.SubstitutionSound envΓ envΔ σ) :
    (block : Block Γ outs) → eval envΔ (block.substitute σ) = eval envΓ block
  | .ret results => by
      simp only [Block.substitute, eval]
      exact Term.evalArgs_substitute envΓ envΔ σ hσ results
  | .let1 value body => by
      simp only [Block.substitute, eval]
      rw [Term.eval_substitute envΓ envΔ σ hσ value]
      exact eval_substitute
        (TorchLean.TensorPack.append envΓ
          (.cons (Term.eval envΓ value) .nil))
        (TorchLean.TensorPack.append envΔ
          (.cons (Term.eval envΓ value) .nil))
        (Substitution.liftRight σ) (hσ.lift_right (Term.eval envΓ value)) body

/-- Evaluating an inlined block is the same as evaluating its original body under the supplied
typed argument values. -/
theorem eval_instantiate
    {Γ Δ outs : List Shape} {α : Type 0} [TorchLean.Storage α] [Context α]
    (env : TorchLean.TensorPack α Δ) (args : Args Δ Γ) (block : Block Γ outs) :
    eval env (block.instantiate args) = eval (Term.evalArgs env args) block := by
  apply eval_substitute (Term.evalArgs env args) env (Args.get args)
  intro shape termVar
  exact Term.eval_get env args termVar

/-- Pure evaluation of block composition is ordinary typed environment extension. -/
theorem eval_andThenWithRenaming
    {Γ middle outputs : List Shape} {α : Type 0} [TorchLean.Storage α] [Context α]
    (envΓ : TorchLean.TensorPack α Γ) (second : Block (Γ ++ middle) outputs) :
    ∀ {Δ : List Shape} (first : Block Δ middle) (envΔ : TorchLean.TensorPack α Δ)
      (ρ : {s : Shape} → Var Γ s → Var Δ s),
      Env.RenamingSound envΓ envΔ ρ →
      eval envΔ (andThenWithRenaming ρ first second) =
        eval (TorchLean.TensorPack.append envΓ (eval envΔ first)) second
  | _, .ret results, envΔ, ρ, hρ => by
      simp only [andThenWithRenaming, eval_instantiate, Term.evalArgs_append]
      rw [Term.evalArgs_rename envΓ envΔ ρ hρ, Term.evalArgs_vars]
      rfl
  | _, .let1 value body, envΔ, ρ, hρ => by
      let result := Term.eval envΔ value
      let envΔ' := TorchLean.TensorPack.append envΔ (.cons result .nil)
      have hρ' : Env.RenamingSound envΓ envΔ' (fun v => Var.weakenRight (ρ v)) := by
        intro shape v
        exact (Env.tget_append_weakenRight envΔ result (ρ v)).trans (hρ v)
      simpa only [andThenWithRenaming, eval, result, envΔ'] using
        eval_andThenWithRenaming envΓ second body envΔ'
          (fun v => Var.weakenRight (ρ v)) hρ'

/-- Composing two blocks evaluates the first once and appends its typed outputs for the second. -/
theorem eval_andThen {Γ middle outputs : List Shape} {α : Type 0} [TorchLean.Storage α] [Context α]
    (env : TorchLean.TensorPack α Γ) (first : Block Γ middle)
    (second : Block (Γ ++ middle) outputs) :
    eval env (first.andThen second) =
      eval (TorchLean.TensorPack.append env (eval env first)) second := by
  unfold andThen
  apply eval_andThenWithRenaming env second first env (fun v => v)
  intro shape v
  rfl


end Block
end DAG
end GraphSpec
end NN

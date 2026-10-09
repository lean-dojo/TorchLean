/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import Mathlib.Init

/-!
# Computations inside custom tensor operations

Tensor graphs describe dependencies between operations. A kernel expression describes the
calculation of one output element inside an operation. Variables are typed, inputs are read-only,
branches are lazy, and folds accumulate in increasing index order. Arithmetic is supplied
explicitly; no field laws or floating-point reassociation are assumed.

Indices use unsigned 64-bit arithmetic rather than silently compiling unbounded natural numbers to
machine integers. The frontend must reject constants outside that range. Tensor reads return errors
instead of inventing a value for an invalid address.
-/

@[expose] public section

namespace NN.Kernel

/-- The three runtime value types supported inside a kernel. -/
inductive Ty where
  | scalar
  | index
  | predicate
  deriving DecidableEq, Repr

/-- Scalar arithmetic remains a parameter of the semantics. -/
abbrev Ty.Value (α : Type) : Ty → Type
  | .scalar => α
  | .index => UInt64
  | .predicate => Bool

/-- A variable can only refer to a value of its declared type. -/
inductive Var : List Ty → Ty → Type where
  | zero {t : Ty} {Γ : List Ty} : Var (t :: Γ) t
  | succ {Γ : List Ty} {t s : Ty} : Var Γ t → Var (s :: Γ) t

/-- Values of the variables currently in scope. -/
structure Env (α : Type) (Γ : List Ty) where
  get : {t : Ty} → Var Γ t → t.Value α

/-- Introduce a local binding without changing the values of older variables. -/
def Env.push {α : Type} {Γ : List Ty} {t : Ty}
    (value : t.Value α) (env : Env α Γ) : Env α (t :: Γ) :=
  ⟨fun v => match v with
    | .zero => value
    | .succ v => env.get v⟩

/-- Floating-point operations are kept separate: multiplication followed by addition is not FMA. -/
inductive ScalarOp where
  | add | sub | mul | div
  deriving Repr, DecidableEq

/-- Index operations follow Lean's `UInt64` semantics, including wraparound. -/
inductive IndexOp where
  | add | sub | mul | div | mod
  deriving Repr, DecidableEq

/-- Comparisons produce a predicate, not an integer mask. -/
inductive Compare where
  | eq | lt | le
  deriving Repr, DecidableEq

/-- An arithmetic implementation fixes scalar operations and comparisons for the whole kernel. -/
structure Arithmetic (α : Type) where
  binary : ScalarOp → α → α → α
  neg : α → α
  compare : Compare → α → α → Bool

/-- Evaluation failures retain the offending input, address, or unsupported output size. -/
inductive Error where
  | input (operand : Nat)
  | bounds (operand : Nat) (index : UInt64) (size : Nat)
  | size (count : Nat)
  deriving Repr, DecidableEq

/-- Input access is shared by reference execution and lowered execution. -/
abbrev Reader (α : Type) := Nat → UInt64 → Except Error α

/-- A typed expression with explicitly ordered arithmetic and bounded sequential folds.

The fold body binds the accumulator first and the iteration index second. A count of zero leaves
the initial value unchanged. Only the selected conditional branch is evaluated.
-/
inductive Expr (α : Type) : List Ty → Ty → Type where
  | scalar {Γ : List Ty} (value : α) : Expr α Γ .scalar
  | index {Γ : List Ty} (value : UInt64) : Expr α Γ .index
  | predicate {Γ : List Ty} (value : Bool) : Expr α Γ .predicate
  | var {Γ : List Ty} {t : Ty} (v : Var Γ t) : Expr α Γ t
  | load {Γ : List Ty} (operand : Nat) (index : Expr α Γ .index) : Expr α Γ .scalar
  | binary {Γ : List Ty} (op : ScalarOp) (x y : Expr α Γ .scalar) : Expr α Γ .scalar
  | neg {Γ : List Ty} (x : Expr α Γ .scalar) : Expr α Γ .scalar
  | indexBinary {Γ : List Ty} (op : IndexOp) (x y : Expr α Γ .index) : Expr α Γ .index
  | compare {Γ : List Ty} (op : Compare) (x y : Expr α Γ .scalar) : Expr α Γ .predicate
  | indexCompare {Γ : List Ty} (op : Compare) (x y : Expr α Γ .index) : Expr α Γ .predicate
  | cond {Γ : List Ty} {t : Ty} (p : Expr α Γ .predicate) (yes no : Expr α Γ t) : Expr α Γ t
  | letIn {Γ : List Ty} {s t : Ty} (value : Expr α Γ s) (body : Expr α (s :: Γ) t) : Expr α Γ t
  | fold {Γ : List Ty} (count : Expr α Γ .index) (initial : Expr α Γ .scalar)
      (body : Expr α (.scalar :: .index :: Γ) .scalar) : Expr α Γ .scalar

/-- Unsigned division and remainder by zero retain Lean's specified behavior. -/
def IndexOp.eval : IndexOp → UInt64 → UInt64 → UInt64
  | .add, x, y => x + y
  | .sub, x, y => x - y
  | .mul, x, y => x * y
  | .div, x, y => x / y
  | .mod, x, y => x % y

/-- Unsigned comparisons do not reinterpret indices as signed integers. -/
def Compare.index : Compare → UInt64 → UInt64 → Bool
  | .eq, x, y => x == y
  | .lt, x, y => x < y
  | .le, x, y => x ≤ y

/-- Execute precisely `count` iterations, stopping at the first failed input read. -/
def iterate {α : Type} (step : UInt64 → α → Except Error α) :
    Nat → UInt64 → α → Except Error α
  | 0, _, acc => .ok acc
  | n + 1, i, acc => do
      let acc ← step i acc
      iterate step n (i + 1) acc

/-- Equality of step functions preserves the full fold, including its failure behavior. -/
@[congr] theorem iterate_congr {α : Type} {f g : UInt64 → α → Except Error α}
    (h : ∀ i acc, f i acc = g i acc) (n : Nat) (i : UInt64) (acc : α) :
    iterate f n i acc = iterate g n i acc := by
  have : f = g := funext fun i => funext (h i)
  subst g
  rfl

/-- Splitting off the last step preserves the order of all reads and arithmetic.
The index conversion is modular, just like the unsigned counter used by `iterate`. -/
theorem iterate_succ {α : Type} (step : UInt64 → α → Except Error α)
    (n : Nat) (i : UInt64) (acc : α) :
    iterate step (n + 1) i acc = (do
      let value ← iterate step n i acc
      step (i + n.toUInt64) value) := by
  induction n generalizing i acc with
  | zero =>
      change (step i acc >>= fun value => Except.ok value) = step (i + 0) acc
      rw [UInt64.add_zero]
      simp only [Bind.bind, Except.bind]
      cases step i acc <;> rfl
  | succ n ih =>
      change (step i acc >>= fun value => iterate step (n + 1) (i + 1) value) =
        ((step i acc >>= fun value => iterate step n (i + 1) value) >>=
          fun value => step (i + (n + 1).toUInt64) value)
      cases h : step i acc with
      | error e => simp [Bind.bind, Except.bind]
      | ok value =>
          simp only [Bind.bind, Except.bind]
          rw [ih]
          have hIndex : (i + 1) + n.toUInt64 = i + (n + 1).toUInt64 := by
            simp only [Nat.toUInt64, UInt64.ofNat_add, UInt64.ofNat_one]
            rw [UInt64.add_assoc, UInt64.add_comm 1]
          rw [hIndex]
          rfl

/-- A recursive calculation can use the loop backend when its equations give a scalar recurrence.
This includes failing input reads: neither later steps nor the final result mask an earlier error.
No laws about the scalar arithmetic, and no reassociation of operations, are assumed. -/
theorem iterate_eq_recurrence {α : Type} (f : Nat → Except Error α)
    (step : UInt64 → α → Except Error α) (initial : Except Error α)
    (hzero : f 0 = initial)
    (hsucc : ∀ n, f (n + 1) = (do
      let previous ← f n
      step n.toUInt64 previous)) (n : Nat) :
    (do let value ← initial; iterate step n 0 value) = f n := by
  induction n with
  | zero => cases initial <;> simp [iterate, hzero, Bind.bind, Except.bind]
  | succ n ih =>
      rw [hsucc, ← ih]
      cases initial <;> simp [iterate_succ, Bind.bind, Except.bind]

/-- Reference semantics: operands are evaluated left to right and folds are never reassociated. -/
def Expr.eval {α : Type} (arithmetic : Arithmetic α) (read : Reader α)
    {Γ : List Ty} {t : Ty} (env : Env α Γ) : Expr α Γ t → Except Error (t.Value α)
  | .scalar x => .ok x
  | .index i => .ok i
  | .predicate p => .ok p
  | .var v => .ok (env.get v)
  | .load operand i => do read operand (← i.eval arithmetic read env)
  | .binary op x y => do
      let x ← x.eval arithmetic read env
      let y ← y.eval arithmetic read env
      return arithmetic.binary op x y
  | .neg x => do return arithmetic.neg (← x.eval arithmetic read env)
  | .indexBinary op x y => do
      let x ← x.eval arithmetic read env
      let y ← y.eval arithmetic read env
      return op.eval x y
  | .compare op x y => do
      let x ← x.eval arithmetic read env
      let y ← y.eval arithmetic read env
      return arithmetic.compare op x y
  | .indexCompare op x y => do
      let x ← x.eval arithmetic read env
      let y ← y.eval arithmetic read env
      return op.index x y
  | .cond p yes no => do
      let p ← p.eval arithmetic read env
      bif p then yes.eval arithmetic read env else no.eval arithmetic read env
  | .letIn value body => do
      let value ← value.eval arithmetic read env
      body.eval arithmetic read (env.push value)
  | .fold count initial body => do
      let count ← count.eval arithmetic read env
      let initial ← initial.eval arithmetic read env
      iterate (fun i acc => body.eval arithmetic read
        (Env.push (t := .scalar) acc (Env.push (t := .index) i env)))
        count.toNat 0 initial

/-- A type-preserving change of variable addresses. -/
abbrev Renaming (Γ Δ : List Ty) := {t : Ty} → Var Γ t → Var Δ t

/-- Extend a renaming under a binder, keeping the newly bound variable fixed. -/
def Renaming.lift {Γ Δ : List Ty} {s : Ty} (ρ : Renaming Γ Δ) :
    Renaming (s :: Γ) (s :: Δ)
  | _, .zero => .zero
  | _, .succ v => .succ (ρ v)

/-- Rename local variables without changing the expression's operation order. -/
def Expr.rename {α : Type} {Γ Δ : List Ty} {t : Ty} (ρ : Renaming Γ Δ) :
    Expr α Γ t → Expr α Δ t
  | .scalar x => .scalar x
  | .index i => .index i
  | .predicate p => .predicate p
  | .var v => .var (ρ v)
  | .load operand i => .load operand (i.rename ρ)
  | .binary op x y => .binary op (x.rename ρ) (y.rename ρ)
  | .neg x => .neg (x.rename ρ)
  | .indexBinary op x y => .indexBinary op (x.rename ρ) (y.rename ρ)
  | .compare op x y => .compare op (x.rename ρ) (y.rename ρ)
  | .indexCompare op x y => .indexCompare op (x.rename ρ) (y.rename ρ)
  | .cond p yes no => .cond (p.rename ρ) (yes.rename ρ) (no.rename ρ)
  | .letIn value body => .letIn (value.rename ρ) (body.rename ρ.lift)
  | .fold count initial body =>
      .fold (count.rename ρ) (initial.rename ρ)
        (body.rename (Renaming.lift (s := .scalar) (Renaming.lift (s := .index) ρ)))

/-- Pull an environment back along variable addresses. -/
def Env.pull {α : Type} {Γ Δ : List Ty} (ρ : Renaming Γ Δ) (env : Env α Δ) : Env α Γ :=
  ⟨fun v => env.get (ρ v)⟩

theorem Env.pull_push {α : Type} {Γ Δ : List Ty} {s : Ty}
    (ρ : Renaming Γ Δ) (env : Env α Δ) (value : s.Value α) :
    (Env.pull (Renaming.lift ρ) (Env.push value env) : Env α (s :: Γ)) =
      (Env.push value (Env.pull ρ env) : Env α (s :: Γ)) := by
  apply congrArg (@Env.mk α (s :: Γ))
  funext t v
  cases v <;> rfl

private theorem Env.pull_fold {α : Type} {Γ Δ : List Ty}
    (ρ : Renaming Γ Δ) (env : Env α Δ) (i : UInt64) (acc : α) :
    (Env.pull (Renaming.lift (s := .scalar) (Renaming.lift (s := .index) ρ))
      (Env.push (t := .scalar) acc (Env.push (t := .index) i env)) :
        Env α (.scalar :: .index :: Γ)) =
      (Env.push (t := .scalar) acc (Env.push (t := .index) i (Env.pull ρ env)) :
        Env α (.scalar :: .index :: Γ)) := by
  apply congrArg (@Env.mk α (.scalar :: .index :: Γ))
  funext t v
  cases v with
  | zero => rfl
  | succ v => cases v <;> rfl

/-- Moving local variables is valid for arbitrary scalar arithmetic and failing input reads. -/
theorem Expr.eval_rename {α : Type} {Γ : List Ty} {t : Ty} (expr : Expr α Γ t)
    (arithmetic : Arithmetic α) (read : Reader α) {Δ : List Ty}
    (ρ : Renaming Γ Δ) (env : Env α Δ) :
    (expr.rename ρ).eval arithmetic read env = expr.eval arithmetic read (env.pull ρ) := by
  induction expr generalizing Δ with
  | scalar | index | predicate | var => rfl
  | load operand i ih => simp only [rename, eval, ih]
  | binary op x y hx hy => simp only [rename, eval, hx, hy]
  | neg x hx => simp only [rename, eval, hx]
  | indexBinary op x y hx hy => simp only [rename, eval, hx, hy]
  | compare op x y hx hy => simp only [rename, eval, hx, hy]
  | indexCompare op x y hx hy => simp only [rename, eval, hx, hy]
  | cond p yes no hp hy hn => simp only [rename, eval, hp, hy, hn]
  | letIn value body hv hb => simp only [rename, eval, hv, hb, Env.pull_push]
  | fold count initial body hc hi hb =>
      simp only [rename, eval, hc, hi]
      congr 1
      funext n
      congr 1
      funext acc
      apply iterate_congr
      intro i value
      rw [hb]
      exact congrArg (fun env => body.eval arithmetic read env) (Env.pull_fold ρ env i value)

end NN.Kernel

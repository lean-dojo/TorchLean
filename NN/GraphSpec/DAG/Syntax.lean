/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Program

/-!
# Typed GraphSpec DAG syntax

This module defines the shape-indexed variables, terms, argument lists, substitutions, and
multi-result blocks used by the canonical GraphSpec DAG representation.
-/

@[expose] public section

namespace NN
namespace GraphSpec
namespace DAG

open Spec TorchLean
open TorchLean.Tensor

/-! ## Primitives (arbitrary arity) -/

/--
An n-ary primitive operation.

Compared to the sequential `GraphSpec.Primitive`, a `PrimOp` here is parameter-free: parameters are
just ordinary inputs in the environment. This is what makes the DAG language flexible: a “layer”
is expressed by `let1`-binding its parameters and then using them as inputs to ops.

Type indices:
- `ins : List Shape` is the ordered list of input tensor shapes the op expects.
- `τ : Shape` is the output tensor shape.
 -/
structure PrimOp (ins : List Shape) (τ : Shape) where
  /-- Debug name for error messages / inspection. -/
  name : String
  /-- Pure reference semantics (`ins` arguments packed as a typed list). -/
  specFwd : ∀ {α : Type 0}, [TorchLean.Storage α] → [Context α] →
    TorchLean.TensorPack α ins → TorchLean.Tensor α τ
  /-- Executable TorchLean program with arguments of shapes `ins`. -/
  program :
    ∀ {α : Type 0}, [TorchLean.Storage α] → [Context α] → Runtime.Autograd.Model.Program α ins τ

/-! ## Typed variables -/

/-- A de Bruijn variable whose result shape is part of its type.

Unlike a bare `Fin Γ.length`, `Var Γ s` records that the selected entry of `Γ` has shape `s`.
Environment lookup therefore reduces structurally, without dependent casts through `List.get`.
`ofFin` remains available to programmatic lowerings that discover positions dynamically.
-/
inductive Var : (Γ : List Shape) → Shape → Type where
  /-- The first value in a nonempty environment. -/
  | head {s : Shape} {Γ : List Shape} : Var (s :: Γ) s
  /-- A variable inherited from the tail of an environment. -/
  | tail {Γ : List Shape} {s t : Shape} : Var Γ t → Var (s :: Γ) t

namespace Var

/-- Convert a numeric environment position to a shape-indexed variable. -/
def ofFin {Γ : List Shape} (i : Fin Γ.length) : Var Γ (Γ.get i) :=
  match Γ with
  | .nil => nomatch i
  | .cons _ _ => Fin.cases .head (fun j => .tail (ofFin j)) i

/-- Preserve a variable when one value is appended to its environment. -/
def weakenRight {Γ : List Shape} {s t : Shape} : Var Γ s → Var (Γ ++ [t]) s
  | .head => .head
  | .tail v => .tail (weakenRight v)

/-- Embed a variable from the left side of an appended environment. -/
def inLeft {Γ : List Shape} (right : List Shape) {s : Shape} : Var Γ s → Var (Γ ++ right) s
  | .head => .head
  | .tail v => .tail (inLeft right v)

/-- Embed a variable from the right side of an appended environment. -/
def inRight (left : List Shape) {Γ : List Shape} {s : Shape} : Var Γ s → Var (left ++ Γ) s :=
  match left with
  | .nil => fun v => v
  | .cons _ rest => fun v => .tail (inRight rest v)

/-- The final variable in an environment extended by one value. -/
def last (Γ : List Shape) {s : Shape} : Var (Γ ++ [s]) s :=
  match Γ with
  | .nil => .head
  | .cons _ rest => .tail (last rest)

/-- Extend a shape-preserving variable renaming across one value appended to both environments. -/
def liftRight {Γ Δ : List Shape} {t : Shape}
    (ρ : {s : Shape} → Var Γ s → Var Δ s) {s : Shape}
    (v : Var (Γ ++ [t]) s) : Var (Δ ++ [t]) s :=
  match Γ with
  | .nil =>
      match v with
      | .head => last Δ
      | .tail impossible => nomatch impossible
  | .cons _ _ =>
      match v with
      | .head => weakenRight (ρ .head)
      | .tail inherited => liftRight (fun v => ρ (.tail v)) inherited

end Var

/-! ## DAG terms + arguments (mutual) -/

mutual
  /--
  A well-typed DAG term.

  Read this as: “under environment `Γ`, this term produces a tensor of shape `τ`”.
  -/
  inductive Term : (Γ : List Shape) → Shape → Type 1 where
    /-- Variable read (shape-indexed de Bruijn position in the environment). -/
    | var {Γ : List Shape} {s : Shape} (i : Var Γ s) : Term Γ s
    /--
    Cast a term’s *output shape* along a propositional equality.

    This is an internal hygiene tool: when we build terms programmatically (e.g. by lowering a
    higher-level syntax into DAG form), we often end up with goals like “`Γ.get i = τ`” that are
    true but not definitional.

    Using a `cast` node keeps the term in constructor form (so evaluators/compilers can still
    pattern match), and pushes the non-definitional equality into the semantics where it can be
    handled by `cases h`.
     -/
    | cast {Γ : List Shape} {σ τ : Shape} : Term Γ σ → σ = τ → Term Γ τ
    /--
    Cast a term’s *environment* along a propositional equality.

    This is useful when normalizing list-association/parenthesization choices in `Γ` without
    changing meaning.
     -/
    | castEnv {Γ Γ' : List Shape} {τ : Shape} : Term Γ τ → Γ = Γ' → Term Γ' τ
    /-- Apply an n-ary primitive op to n arguments. -/
    | op {Γ : List Shape} {ins : List Shape} {τ : Shape} :
        PrimOp ins τ → Args Γ ins → Term Γ τ
    /-- Let-bind a single intermediate value, extending the environment. -/
    | let1 {Γ : List Shape} {σ τ : Shape} : Term Γ σ → Term (Γ ++ [σ]) τ → Term Γ τ

  /--
  A typed list of argument terms.

  `Args Γ [s₁, …, sₙ]` is a tuple of `n` terms, each well-typed under the same environment `Γ`,
  with corresponding shapes `s₁, …, sₙ`.
  -/
  inductive Args : (Γ : List Shape) → List Shape → Type 1 where
    | nil {Γ} : Args Γ []
    | cons {Γ} {s : Shape} {ss : List Shape} : Term Γ s → Args Γ ss → Args Γ (s :: ss)
end

mutual
  /-- Rename every free variable in a typed operation-argument list. -/
  def Args.rename {Γ Δ : List Shape} {ss : List Shape}
      (ρ : {t : Shape} → Var Γ t → Var Δ t) : Args Γ ss → Args Δ ss
    | .nil => .nil
    | .cons term rest => .cons (Term.rename ρ term) (Args.rename ρ rest)

  /-- Rename every free variable of a term while preserving its tensor shape. -/
  def Term.rename {Γ Δ : List Shape} {s : Shape}
      (ρ : {t : Shape} → Var Γ t → Var Δ t) : Term Γ s → Term Δ s
    | .var i => .var (ρ i)
    | .cast term h => .cast (Term.rename ρ term) h
    | .castEnv term h =>
        match h.symm with
        | rfl => Term.rename ρ term
    | .op primitive args => .op primitive (Args.rename ρ args)
    | .let1 value body =>
        .let1 (Term.rename ρ value) (Term.rename (Var.liftRight ρ) body)

  /-- Preserve a term when an unrelated value is prepended to its environment. -/
  def Term.weakenLeft {Γ : List Shape} {s t : Shape} : Term Γ s → Term (t :: Γ) s
    | term => Term.rename (fun v => .tail v) term

  /-- Preserve typed operation arguments when a value is prepended to their environment. -/
  def Args.weakenLeft {Γ : List Shape} {ss : List Shape} {t : Shape} :
      Args Γ ss → Args (t :: Γ) ss
    | .nil => .nil
    | .cons term rest => .cons (Term.weakenLeft term) (Args.weakenLeft rest)
end

namespace Term

/-- Preserve a term when an unrelated value is appended to its environment. -/
def weakenRight {Γ : List Shape} {s t : Shape} (term : Term Γ s) : Term (Γ ++ [t]) s :=
  Term.rename Var.weakenRight term

/-- Preserve a term when an arbitrary typed environment is appended. -/
def weakenAppend {Γ : List Shape} (right : List Shape) {s : Shape}
    (term : Term Γ s) : Term (Γ ++ right) s :=
  Term.rename (Var.inLeft right) term

end Term

namespace Args

/-- Select the term assigned to a shape-indexed variable. -/
def get {Γ Δ : List Shape} {s : Shape} : Args Δ Γ → Var Γ s → Term Δ s
  | .cons term _, .head => term
  | .cons _ rest, .tail v => get rest v

/-- Concatenate two typed argument lists living in the same graph environment. -/
def append {Γ left right : List Shape} :
    Args Γ left → Args Γ right → Args Γ (left ++ right)
  | .nil, rightArgs => rightArgs
  | .cons term rest, rightArgs => .cons term (append rest rightArgs)

/-- Split arguments at a type-level list boundary.

This is the argument-list counterpart of `TensorPack.splitAppend`.  It is useful when a model owns a
concatenated parameter ABI but its implementation is assembled recursively from smaller models:
each component receives exactly the terms belonging to its part of the ABI, with every tensor
shape retained by the type checker.
-/
def splitAppend {Γ : List Shape} : {left right : List Shape} →
    Args Γ (left ++ right) → Args Γ left × Args Γ right
  | .nil, _right, args => (.nil, args)
  | .cons _shape left, right, .cons term rest =>
      let parts := splitAppend (left := left) (right := right) rest
      (.cons term parts.1, parts.2)

/-- Splitting arguments immediately after concatenating them recovers both original lists. -/
@[simp] theorem splitAppend_append {Γ left right : List Shape}
    (leftArgs : Args Γ left) (rightArgs : Args Γ right) :
    splitAppend (append leftArgs rightArgs) = (leftArgs, rightArgs) := by
  induction left with
  | nil => cases leftArgs; rfl
  | cons shape left ih =>
      cases leftArgs with
      | cons term rest =>
          simp only [append, splitAppend]
          rw [ih rest]

/-- Concatenating both parts of a split recovers the original typed argument list. -/
theorem append_splitAppend {Γ left right : List Shape}
    (args : Args Γ (left ++ right)) :
    append (splitAppend args).1 (splitAppend args).2 = args := by
  induction left with
  | nil => rfl
  | cons shape left ih =>
      cases args with
      | cons term rest =>
          simp only [splitAppend, append]
          rw [ih rest]

/-- View every entry of a typed environment as a term in that same environment.

The result preserves the order and shape indices of `Γ`.  Large graph definitions can therefore
pattern-match once on `vars Γ` instead of selecting every input by a numeric index and separately
proving that the selected position has the expected shape. -/
def vars : (Γ : List Shape) → Args Γ Γ
  | .nil => .nil
  | .cons _ rest => .cons (.var .head) (Args.weakenLeft (vars rest))

/-- Selecting from renamed arguments renames the selected term. -/
@[simp] theorem get_rename {Γ Δ ss : List Shape} {s : Shape}
    (ρ : {t : Shape} → Var Γ t → Var Δ t) (args : Args Γ ss) (position : Var ss s) :
    Args.get (Args.rename ρ args) position = Term.rename ρ (Args.get args position) := by
  induction position with
  | head => cases args; rfl
  | tail position ih =>
      cases args with
      | cons _ rest => exact ih rest

/-- Lookup commutes with embedding an argument list below one new graph variable. -/
@[simp] theorem get_weakenLeft {Γ ss : List Shape} {s t : Shape}
    (args : Args Γ ss) (position : Var ss s) :
    Args.get (Args.weakenLeft (t := t) args) position =
      Term.weakenLeft (t := t) (Args.get args position) := by
  induction position with
  | head => cases args; rfl
  | tail position ih =>
      cases args with
      | cons _ rest => exact ih rest

/-- Selecting a variable from the complete environment returns that variable as a term. -/
@[simp] theorem get_vars {Γ : List Shape} {s : Shape} (position : Var Γ s) :
    Args.get (vars Γ) position = Term.var position := by
  induction position with
  | head => rfl
  | tail position ih =>
      simp only [vars, get, get_weakenLeft, ih, Term.weakenLeft, Term.rename]

end Args

/-! ## Typed substitution

Substitution is the operation used to inline one graph into another.  A substitution assigns a
well-typed term in `Δ` to every variable in `Γ`; applying it replaces the free variables of a term
without changing the term's result shape.  The let-binding case lifts the assignment across the
new value appended by the binder.
-/

/-- A shape-preserving assignment of terms to every variable in an environment. -/
abbrev Substitution (Γ Δ : List Shape) := {s : Shape} → Var Γ s → Term Δ s

namespace Substitution

/-- Extend a substitution across one value appended to both environments. -/
def liftRight {Γ Δ : List Shape} {t : Shape}
    (σ : Substitution Γ Δ) : Substitution (Γ ++ [t]) (Δ ++ [t]) :=
  match Γ with
  | .nil => fun v =>
      match v with
      | .head => .var (Var.last Δ)
      | .tail impossible => nomatch impossible
  | .cons _ rest => fun v =>
      match v with
      | .head => Term.weakenRight (σ .head)
      | .tail inherited =>
          liftRight (Γ := rest) (fun v => σ (.tail v)) inherited
termination_by Γ

end Substitution

mutual
  /-- Replace every free variable in a typed operation-argument list. -/
  def Args.substitute {Γ Δ shapes : List Shape} (σ : Substitution Γ Δ) :
      Args Γ shapes → Args Δ shapes
    | .nil => .nil
    | .cons term rest => .cons (Term.substitute σ term) (Args.substitute σ rest)

  /-- Replace every free variable in a term by its assigned term. -/
  def Term.substitute {Γ Δ : List Shape} {s : Shape} (σ : Substitution Γ Δ) :
      Term Γ s → Term Δ s
    | .var v => σ v
    | .cast term equality => .cast (Term.substitute σ term) equality
    | .castEnv term equality =>
        match equality.symm with
        | rfl => Term.substitute σ term
    | .op primitive args => .op primitive (Args.substitute σ args)
    | .let1 value body =>
        .let1 (Term.substitute σ value)
          (Term.substitute (Substitution.liftRight σ) body)
end

namespace Term

/-- Inline a term by supplying one typed argument term for each free variable. -/
def instantiate {Γ Δ : List Shape} {s : Shape}
    (arguments : Args Δ Γ) (term : Term Γ s) : Term Δ s :=
  term.substitute (Args.get arguments)

end Term

/-- A multi-result DAG block with shared A-normal-form bindings.

`Block.let1` computes an intermediate once and makes it available to every eventual result. This
is the representation needed by recurrent steps that return both a new state and an output derived
from that state.
-/
inductive Block : (Γ : List Shape) → List Shape → Type 1 where
  /-- Return one term for each output shape. -/
  | ret {Γ outs} : Args Γ outs → Block Γ outs
  /-- Compute one shared intermediate and append it to the environment. -/
  | let1 {Γ outs σ} : Term Γ σ → Block (Γ ++ [σ]) outs → Block Γ outs

namespace Block

/-- Replace every free variable in a multi-result block. -/
def substitute {Γ Δ outputs : List Shape} (σ : Substitution Γ Δ) :
    Block Γ outputs → Block Δ outputs
  | .ret results => .ret (Args.substitute σ results)
  | .let1 value body =>
      .let1 (Term.substitute σ value)
        (Block.substitute (Substitution.liftRight σ) body)

/-- Inline a multi-result block by supplying all of its free variables. -/
def instantiate {Γ Δ outputs : List Shape}
    (arguments : Args Δ Γ) (block : Block Γ outputs) : Block Δ outputs :=
  block.substitute (Args.get arguments)

/-- Compose two multi-output blocks.

The second block sees the original environment followed by every result of the first block. Any
`let1` bindings inside the first block remain shared. This is the typed DAG analogue of binding a
tuple-valued computation and is the basic operation needed to compose recurrent cells, residual
branches, and encoder-decoder stages.
-/
def andThenWithRenaming {Γ Δ middle outputs : List Shape}
    (ρ : {s : Shape} → Var Γ s → Var Δ s)
    (first : Block Δ middle) (second : Block (Γ ++ middle) outputs) : Block Δ outputs :=
  match first with
  | .ret results =>
      second.instantiate (Args.append (Args.rename ρ (Args.vars Γ)) results)
  | .let1 value body =>
      .let1 value <| andThenWithRenaming (fun v => Var.weakenRight (ρ v)) body second
termination_by sizeOf first
decreasing_by simp_wf

/-- Feed every output of `first` to `second`, preserving the original environment. -/
def andThen {Γ middle outputs : List Shape}
    (first : Block Γ middle) (second : Block (Γ ++ middle) outputs) : Block Γ outputs :=
  andThenWithRenaming (fun v => v) first second

end Block
end DAG
end GraphSpec
end NN

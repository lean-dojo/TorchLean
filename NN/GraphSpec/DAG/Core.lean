/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.Backend
import Mathlib.Algebra.Order.Algebra

/-!
# Canonical GraphSpec DAG

This file defines the canonical general GraphSpec representation: typed SSA/DAG terms.

`NN.GraphSpec.Core` gives a *sequential* authoring language (`Graph` with `>>>`) for chain-like
architectures. That syntax lowers into this DAG language. Many modern architectures are not purely
sequential:

- **skip connections** (ResNets),
- **shared subcomputations** (reusing the same feature tensor multiple times),
- **multi-input ops** (e.g. `add`, concatenation, attention blocks, …).

`GraphSpec.DAG` is the general version: a small SSA/A-normal-form term language whose terms denote
DAG-shaped computation graphs.

## Core idea: typed environments

We track an environment `Γ : List Shape` at the type level. A term `Term Γ τ` means:

- “if you provide values for every shape in `Γ` (in order),”
- “this term computes a tensor of shape `τ`.”

This is similar in spirit to a simply-typed lambda calculus with de Bruijn variables, except:

- there are no lambdas (only `let`-bindings), so the graph is acyclic by construction,
- primitive ops have arbitrary arity `ins : List Shape`,
- and `Args Γ ins` is a typed list of input subterms for an op.

The constructors are:

- `var`   : read a variable from the environment (`Fin Γ.length` index),
- `cast` / `castEnv` : explicit casts to handle non-definitional equalities in `Shape` / `Γ`,
- `op`    : apply an n-ary primitive to n arguments,
- `let1`  : bind an intermediate result and extend the environment (`Γ ++ [σ]`).

## Mathematical semantics (intended)

For a fixed scalar type `α` with `[Context α]`, we interpret an environment as a typed list of
tensors `TList α Γ`. Then:

- `Term.eval : TList α Γ → Term Γ τ → Tensor α τ` is the *pure* reference semantics.
- `Term.compile` produces a backend-generic TorchLean program that computes the same graph, but in
  the executable (monadic, reference-based) runtime world.

## Small example (residual add)

The residual pattern “main path + skip path” is the canonical example that needs DAG syntax:

```
y   = linear(W,b,x)
out = relu(y + x)
```

Here `x` is used twice, so a pure chain representation would need duplication. With `let1`,
sharing is explicit: compute `y` once, then reuse it.

See `NN/GraphSpec/Models/ResidualLinear.lean` for a complete, well-typed instance.

## References / citations

Conceptual background (stable, standard references):

- SSA form: Cytron et al. (1991), “Efficiently Computing Static Single Assignment Form…”.
- A-normal form / let-normal form (used to enforce DAG structure).
- Automatic differentiation survey: Baydin et al. (2018), “Automatic Differentiation in Machine
  Learning: a Survey”.
- Residual networks: He et al. (2016), “Deep Residual Learning for Image Recognition”.
-/

@[expose] public section


namespace NN
namespace GraphSpec
namespace DAG

open Spec
open Tensor
open NN.Tensor

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
  specFwd : ∀ {α : Type 0}, [Context α] → Runtime.Autograd.Torch.TList α ins → Tensor α τ
  /-- Executable TorchLean program with arguments of shapes `ins`. -/
  torchProgram :
    ∀ {α : Type 0}, [Context α] → [DecidableEq Shape] →
      Runtime.Autograd.TorchLean.Program α ins τ

/-! ## DAG terms + arguments (mutual) -/

mutual
  /--
  A well-typed DAG term.

  Read this as: “under environment `Γ`, this term produces a tensor of shape `τ`”.
  -/
  inductive Term : (Γ : List Shape) → Shape → Type 1 where
    /-- Variable read (de Bruijn index into the environment). -/
    | var {Γ : List Shape} (i : Fin Γ.length) : Term Γ (Γ.get i)
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

namespace Env

open Runtime.Autograd.Torch

/--
Typed environment lookup for pure tensors.

This is the underlying “variable semantics” for `Term.eval`.
 -/
def tget {α : Type} : {ss : List Shape} → TList α ss → (i : Fin ss.length) → Tensor α (ss.get i)
  | [], .nil, i => nomatch i
  | s :: ss, .cons x xs, i =>
      -- We use `Fin.cases` rather than pattern-matching on `⟨0, _⟩` / `⟨Nat.succ _, _⟩`.
      --
      -- This makes evaluation/`simp` behave well even when indices are produced by numeric notation
      -- (which may elaborate via `OfNat` rather than literally as `Fin.mk`).
      Fin.cases
        (motive := fun i : Fin (s :: ss).length => Tensor α ((s :: ss).get i))
        (by simpa using x)
        (fun j => by
          -- `List.get` at a successor index reduces definitionally to the tail.
          simpa using tget (ss := ss) xs j)
        i

/--
Typed environment lookup for backend references.

This is the underlying “variable semantics” for `Term.compile`.
 -/
def rget {Ref : Shape → Type} : {ss : List Shape} → Runtime.Autograd.Torch.RefList Ref ss →
    (i : Fin ss.length) → Ref (ss.get i)
  | [], .nil, i => nomatch i
  | s :: ss, .cons x xs, i =>
      Fin.cases
        (motive := fun i : Fin (s :: ss).length => Ref ((s :: ss).get i))
        (by simpa using x)
        (fun j => by
          simpa using rget (ss := ss) xs j)
        i

end Env

namespace Term

open Runtime.Autograd.Torch

/-! ### Spec interpreter -/

mutual
  /-- Evaluate a typed argument list by evaluating each component term under the same environment.
    -/
  def evalArgs
      {Γ : List Shape} {ins : List Shape}
      {α : Type 0} [Context α]
      (env : TList α Γ) :
      Args Γ ins → TList α ins
    | .nil => .nil
    | .cons t ts => .cons (eval (Γ := Γ) (α := α) env t) (evalArgs (Γ := Γ) (α := α) env ts)

  /--
  Pure evaluation of a DAG term.

  This is the “math-first” semantics: we interpret a term as a pure function on tensors.
  No monads, no mutation, no autograd tape — just the Spec definitions of primitives.

  The key runtime discipline is the environment discipline:

  - `var` reads from `env`,
  - `op` evaluates its arguments and feeds them to the primitive’s `specFwd`,
  - `let1` evaluates the bound term once and extends `env` for the body.
  -/
  def eval
      {Γ : List Shape} {τ : Shape}
      {α : Type 0} [Context α]
      (env : TList α Γ) :
      Term Γ τ → Tensor α τ
    | .var i => Env.tget (α := α) (ss := Γ) env i
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
        let env' : TList α (Γ ++ [σ]) :=
          Proofs.Autograd.Algebra.TList.append (α := α) (ss₁ := Γ) (ss₂ := [σ]) env (.cons v .nil)
        eval (Γ := Γ ++ [σ]) (α := α) env' body
end

/-! ### TorchLean compiler -/

/--
`RefT` is the backend’s reference type for tensors of a given shape.

In the executable runtime, primitives operate on *references* (allocated tensors) inside a monad.
This matches how typical deep-learning runtimes model device placement, mutation, and autograd.
 -/
abbrev RefT (m : Type → Type) (α : Type 0) [Context α] [DecidableEq Shape]
    [Runtime.Autograd.Torch.Ops (m := m) (α := α)] (s : Shape) : Type :=
  Runtime.Autograd.Torch.Ops.Ref (m := m) (α := α) s

mutual
  /-- Compile a typed argument list by compiling each component term under the same environment. -/
  def compileArgs
      {Γ ins : List Shape}
      {α : Type 0} [Context α] [DecidableEq Shape]
      {m : Type → Type} [Monad m] [Runtime.Autograd.Torch.Ops (m := m) (α := α)]
      (env : Runtime.Autograd.Torch.RefList (RefT (m := m) (α := α)) Γ) :
      Args Γ ins → m (Runtime.Autograd.Torch.RefList (RefT (m := m) (α := α)) ins)
    | .nil => pure .nil
    | .cons t ts => do
        let r ← compile (Γ := Γ) (α := α) (m := m) env t
        let rs ← compileArgs (Γ := Γ) (ins := _) (α := α) (m := m) env ts
        pure (.cons r rs)

  /--
  Compile a typed `Term Γ τ` into the backend monad `m`, producing a reference to a tensor of shape
    `τ`.

  This is the "executable" counterpart of `Term.eval`: instead of returning a pure `Tensor`, we
  emit backend ops (`Runtime.Autograd.Torch.Ops`) that allocate tensors and apply primitives.
  -/
  def compile
      {Γ : List Shape} {τ : Shape}
      {α : Type 0} [Context α] [DecidableEq Shape]
      {m : Type → Type} [Monad m] [Runtime.Autograd.Torch.Ops (m := m) (α := α)]
      (env : Runtime.Autograd.Torch.RefList (RefT (m := m) (α := α)) Γ) :
      Term Γ τ → m (RefT (m := m) (α := α) τ)
    | .var i => pure (Env.rget (Ref := RefT (m := m) (α := α)) (ss := Γ) env i)
    | .cast t h =>
        match h with
        | rfl => compile (Γ := Γ) (α := α) (m := m) env t
    | .castEnv t h =>
        -- Same structure as `eval`: rewrite the term to the current environment.
        match h.symm with
        | rfl => compile (Γ := Γ) (α := α) (m := m) env t
    | .op (ins := ins) p args => do
        let rs ← compileArgs (Γ := Γ) (ins := ins) (α := α) (m := m) env args
        Runtime.Autograd.Torch.CurriedRef.uncurry (ss := ins)
          (Ref := RefT (m := m) (α := α))
          (p.torchProgram (α := α)) rs
    | .let1 (σ := σ) t body => do
        let v ← compile (Γ := Γ) (α := α) (m := m) env t
        let env' :=
          Runtime.Autograd.Torch.RefList.append
            (Ref := RefT (m := m) (α := α))
            (ss₁ := Γ) (ss₂ := [σ]) env (.cons v .nil)
        compile (Γ := Γ ++ [σ]) (α := α) (m := m) env' body
end

end Term

/-! ## Model wrapper -/

/--
A small “model” wrapper around DAG terms.

This mirrors the sequential `Graph` surface:

- `ps` are parameter tensor shapes (tracked at the type level),
- `ins` are the shapes of *non-parameter inputs* (e.g. data tensors),
- `τ` is the output shape.

The model body is a `Term (ps ++ ins) τ`, i.e. it expects an environment that starts with
parameters and then contains the actual inputs.
 -/
structure Model (ps ins : List Shape) (τ : Shape) where
  /-- init Params. -/
  initParams : Runtime.Autograd.Torch.TList Float ps
  /-- body. -/
  body : Term (ps ++ ins) τ

namespace Model

open Runtime.Autograd.Torch

/--
Pure forward semantics of a DAG model.

We build the full environment `Γ = ps ++ ins` by appending the parameter list and the input list,
then evaluate the body using `Term.eval`.
 -/
def specFwd {ps ins : List Shape} {τ : Shape} (m : Model ps ins τ)
    {α : Type 0} [Context α]
    (params : TList α ps) (xs : TList α ins) : Tensor α τ :=
  let env := Proofs.Autograd.Algebra.TList.append (α := α) (ss₁ := ps) (ss₂ := ins) params xs
  Term.eval (Γ := ps ++ ins) (α := α) env m.body

/--
Compile a DAG model to a backend-generic TorchLean program.

The resulting program expects arguments in the order `ps ++ ins` (parameters first, then inputs),
matching the environment discipline used by `specFwd`.
 -/
def torchProgram {ps ins : List Shape} {τ : Shape} (m : Model ps ins τ)
    {α : Type 0} [Context α] [DecidableEq Shape] :
    Runtime.Autograd.TorchLean.Program α (ps ++ ins) τ :=
  fun {μ} _ _ =>
    Runtime.Autograd.Torch.CurriedRef.curry
      (Ref := Term.RefT (m := μ) (α := α))
      (ss := ps ++ ins)
      (β := μ (Term.RefT (m := μ) (α := α) τ))
      (fun args => Term.compile (Γ := ps ++ ins) (α := α) (m := μ) args m.body)

end Model

end DAG
end GraphSpec
end NN

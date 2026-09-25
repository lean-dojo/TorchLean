/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.DAG.Lowering
public import NN.GraphSpec.DAG.Semantics

/-!
# GraphSpec DAG model wrappers

This module packages typed terms and blocks as parameterized single- and multi-output models,
with pure forward semantics, inlining theorems, and executable program conversion.
-/

@[expose] public section

namespace NN
namespace GraphSpec
namespace DAG

open Spec TorchLean
open TorchLean.Tensor

/-! ## Model wrapper -/

/--
A small “model” wrapper around DAG terms.

This mirrors the sequential `Chain` surface:

- `ps` are parameter tensor shapes (tracked at the type level),
- `ins` are the shapes of *non-parameter inputs* (e.g. data tensors),
- `τ` is the output shape.

The model body is a `Term (ps ++ ins) τ`, i.e. it expects an environment that starts with
parameters and then contains the actual inputs.
 -/
structure Model (ps ins : List Shape) (τ : Shape) where
  /-- Initial parameter values, one tensor per shape in `ps`. Shipping the initialization with the
  model means a `Model` is runnable on its own, with no separate setup step. -/
  initParams : TorchLean.TensorPack Float ps
  /-- The computation itself, as a term over the environment `ps ++ ins`: parameters first, then
  data inputs. That fixed ordering is what lets `initParams` be typed by `ps` alone. -/
  body : Term (ps ++ ins) τ

namespace Model

open Runtime.Autograd.Torch

/-- Inline a model body into a larger graph using explicit parameter and input terms.

The result is still an ordinary DAG term: no primitive boundary is introduced, and subsequent
lowering, differentiation, or numerical analysis can inspect every operation of the model.
Each parameter and input term is copied into every use in the body, so pass variables (or
`let1`-bound values) when an argument is expensive or used more than once.
-/
def inline {Γ ps ins : List Shape} {τ : Shape} (model : Model ps ins τ)
    (params : Args Γ ps) (inputs : Args Γ ins) : Term Γ τ :=
  model.body.instantiate (Args.append params inputs)

/--
Pure forward semantics of a DAG model.

We build the full environment `Γ = ps ++ ins` by appending the parameter list and the input list,
then evaluate the body using `Term.eval`.
 -/
def specFwd {ps ins : List Shape} {τ : Shape} (m : Model ps ins τ)
    {α : Type 0} [TorchLean.Storage α] [Context α]
    (params : TorchLean.TensorPack α ps) (xs : TorchLean.TensorPack α ins) : TorchLean.Tensor α τ :=
  let env := TorchLean.TensorPack.append (α := α) (ss₁ := ps) (ss₂ := ins) params xs
  Term.eval (Γ := ps ++ ins) (α := α) env m.body

/-- Inlining a model into a larger DAG preserves the model's pure forward semantics. -/
theorem eval_inline {Γ ps ins : List Shape} {τ : Shape}
    (model : Model ps ins τ) {α : Type 0} [TorchLean.Storage α] [Context α]
    (env : TorchLean.TensorPack α Γ) (params : Args Γ ps) (inputs : Args Γ ins) :
    Term.eval env (model.inline params inputs) =
      model.specFwd (Term.evalArgs env params) (Term.evalArgs env inputs) := by
  rw [inline, Term.eval_instantiate, Term.evalArgs_append]
  rfl

/--
Lower a DAG model to an execution-polymorphic TorchLean program.

The resulting program expects arguments in the order `ps ++ ins` (parameters first, then inputs),
matching the environment discipline used by `specFwd`.
 -/
def toProgram {ps ins : List Shape} {τ : Shape} (m : Model ps ins τ)
    {α : Type 0} [TorchLean.Storage α] [Context α] :
    Runtime.Autograd.Model.Program α (ps ++ ins) τ :=
  fun {μ} _ _ =>
    Runtime.Autograd.Torch.CurriedRef.curry
      (Ref := Runtime.Autograd.Model.RefTy (m := μ) (α := α))
      (ss := ps ++ ins)
      (β := μ (Runtime.Autograd.Model.RefTy (m := μ) (α := α) τ))
      (fun args => Term.lower (Γ := ps ++ ins) (α := α) (m := μ) args m.body)

end Model

/-! ## Models with several outputs -/

/-- A typed DAG model that returns several tensors.

Recurrent layers commonly return both an updated state and an observable output. Keeping these as
a typed list preserves their individual shapes and avoids flattening unrelated tensors into an
untyped buffer. The block body uses shared `let1` bindings, so an updated state can be computed once
and returned alongside values derived from it.
-/
structure MultiModel (ps ins outs : List Shape) where
  /-- Default parameters, in the same ABI order used by `body`. -/
  initParams : TorchLean.TensorPack Float ps
  /-- Shared computation ending in one well-typed term for every output shape. -/
  body : Block (ps ++ ins) outs

namespace MultiModel

open Runtime.Autograd.Torch

/-- Inline a multi-output model into a larger graph.

Shared `let` bindings in the original block remain shared after substitution, so recurrent state
updates are not duplicated when both the state and a derived output are returned. The parameter
and input terms are copied into every use, as in `Model.inline`.
-/
def inline {Γ ps ins outs : List Shape} (model : MultiModel ps ins outs)
    (params : Args Γ ps) (inputs : Args Γ ins) : Block Γ outs :=
  model.body.instantiate (Args.append params inputs)

/-- Pure reference semantics of a multi-output DAG model. -/
def specFwd {ps ins outs : List Shape} (m : MultiModel ps ins outs)
    {α : Type 0} [TorchLean.Storage α] [Context α] (params : TorchLean.TensorPack α ps)
    (xs : TorchLean.TensorPack α ins) : TorchLean.TensorPack α outs :=
  let env := TorchLean.TensorPack.append
    (α := α) (ss₁ := ps) (ss₂ := ins) params xs
  Block.eval (Γ := ps ++ ins) (α := α) env m.body

/-- Inlining a multi-output model preserves every output and every shared intermediate in its
pure reference semantics. -/
theorem eval_inline {Γ ps ins outs : List Shape} (model : MultiModel ps ins outs)
    {α : Type 0} [TorchLean.Storage α] [Context α] (env : TorchLean.TensorPack α Γ)
    (params : Args Γ ps) (inputs : Args Γ ins) :
    Block.eval env (model.inline params inputs) =
      model.specFwd (Term.evalArgs env params) (Term.evalArgs env inputs) := by
  rw [inline, Block.eval_instantiate, Term.evalArgs_append]
  rfl

/-- An execution-polymorphic program returning several shape-indexed tensor references. -/
abbrev MultiOutputProgram (α : Type 0) [TorchLean.Storage α] [Context α]
    (ins outs : List Shape) : Type 1 :=
  ∀ {μ : Type → Type}, [Monad μ] → [Runtime.Autograd.Torch.Ops (m := μ) (α := α)] →
    CurriedRef (fun s => Runtime.Autograd.Model.RefTy (m := μ) (α := α) s) ins
      (μ (RefList (Runtime.Autograd.Model.RefTy (m := μ) (α := α)) outs))

/-- Lower every result of a multi-output model for the selected TorchLean execution target. -/
def toProgram {ps ins outs : List Shape} (m : MultiModel ps ins outs)
    {α : Type 0} [TorchLean.Storage α] [Context α] : MultiOutputProgram α (ps ++ ins) outs :=
  fun {μ} _ _ =>
    CurriedRef.curry
      (Ref := Runtime.Autograd.Model.RefTy (m := μ) (α := α))
      (ss := ps ++ ins)
      (β := μ (RefList (Runtime.Autograd.Model.RefTy (m := μ) (α := α)) outs))
      (fun args => Block.lower (Γ := ps ++ ins) (α := α) (μ := μ) args m.body)

end MultiModel
end DAG
end GraphSpec
end NN

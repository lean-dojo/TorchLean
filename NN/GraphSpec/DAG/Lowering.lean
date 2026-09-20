/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.DAG.Syntax

/-!
# Executable lowering of GraphSpec DAGs

This module lowers typed DAG terms and multi-result blocks into execution-polymorphic TorchLean
programs over backend tensor references.
-/

@[expose] public section

namespace NN
namespace GraphSpec
namespace DAG

open Spec TorchLean
open TorchLean.Tensor

namespace Env

open Runtime.Autograd.Torch

/--
Typed environment lookup for backend references.

This is the underlying “variable semantics” for `Term.lower`.
 -/
def rget {Ref : Shape → Type} : {Γ : List Shape} → {s : Shape} →
    Runtime.Autograd.Torch.RefList Ref Γ → Var Γ s → Ref s
  | _ :: _, _, .cons x _, .head => x
  | _ :: _, _, .cons _ xs, .tail i => rget xs i

end Env

namespace Term

open Runtime.Autograd.Torch

/-! ### Lowering to TorchLean programs -/

mutual
  /-- Lower a typed argument list by lowering each component term under the same environment. -/
  def lowerArgs
      {Γ ins : List Shape}
      {α : Type 0} [TorchLean.Storage α] [Context α]
      {m : Type → Type} [Monad m] [Runtime.Autograd.Torch.Ops (m := m) (α := α)]
      (env : Runtime.Autograd.Torch.RefList (Runtime.Autograd.Model.RefTy (m := m) (α := α)) Γ) :
      Args Γ ins →
        m (Runtime.Autograd.Torch.RefList (Runtime.Autograd.Model.RefTy (m := m) (α := α)) ins)
    | .nil => pure .nil
    | .cons t ts => do
        let r ← lower (Γ := Γ) (α := α) (m := m) env t
        let rs ← lowerArgs (Γ := Γ) (ins := _) (α := α) (m := m) env ts
        pure (.cons r rs)

  /--
  Lower a typed `Term Γ τ` into the execution monad `m`, producing a reference to a tensor of shape
    `τ`.

  This is the "executable" counterpart of `Term.eval`: instead of returning a pure
  `TorchLean.Tensor`, we emit runtime operations (`Runtime.Autograd.Torch.Ops`) that allocate
  tensors and apply primitives.
  -/
  def lower
      {Γ : List Shape} {τ : Shape}
      {α : Type 0} [TorchLean.Storage α] [Context α]
      {m : Type → Type} [Monad m] [Runtime.Autograd.Torch.Ops (m := m) (α := α)]
      (env : Runtime.Autograd.Torch.RefList (Runtime.Autograd.Model.RefTy (m := m) (α := α)) Γ) :
      Term Γ τ → m (Runtime.Autograd.Model.RefTy (m := m) (α := α) τ)
    | .var i => pure (Env.rget (Ref := Runtime.Autograd.Model.RefTy (m := m) (α := α)) env i)
    | .cast t h =>
        match h with
        | rfl => lower (Γ := Γ) (α := α) (m := m) env t
    | .castEnv t h =>
        -- Same structure as `eval`: rewrite the term to the current environment.
        match h.symm with
        | rfl => lower (Γ := Γ) (α := α) (m := m) env t
    | .op (ins := ins) p args => do
        let rs ← lowerArgs (Γ := Γ) (ins := ins) (α := α) (m := m) env args
        Runtime.Autograd.Torch.CurriedRef.uncurry (ss := ins)
          (Ref := Runtime.Autograd.Model.RefTy (m := m) (α := α))
          (p.program (α := α)) rs
    | .let1 (σ := σ) t body => do
        let v ← lower (Γ := Γ) (α := α) (m := m) env t
        let env' :=
          Runtime.Autograd.Torch.RefList.append
            (Ref := Runtime.Autograd.Model.RefTy (m := m) (α := α))
            (ss₁ := Γ) (ss₂ := [σ]) env (.cons v .nil)
        lower (Γ := Γ ++ [σ]) (α := α) (m := m) env' body
end

end Term

namespace Block

open Runtime.Autograd.Torch

/-- Lower a multi-output block for an arbitrary TorchLean execution target. -/
def lower {Γ outs : List Shape} {α : Type 0} [TorchLean.Storage α] [Context α]
    {μ : Type → Type} [Monad μ] [Runtime.Autograd.Torch.Ops (m := μ) (α := α)]
    (env : RefList (Runtime.Autograd.Model.RefTy (m := μ) (α := α)) Γ) :
    Block Γ outs → μ (RefList (Runtime.Autograd.Model.RefTy (m := μ) (α := α)) outs)
  | .ret results => Term.lowerArgs (Γ := Γ) (α := α) (m := μ) env results
  | .let1 (σ := σ) value body => do
      let result ← Term.lower (Γ := Γ) (α := α) (m := μ) env value
      let env' := RefList.append
        (Ref := Runtime.Autograd.Model.RefTy (m := μ) (α := α))
        (ss₁ := Γ) (ss₂ := [σ]) env (.cons result .nil)
      lower (Γ := Γ ++ [σ]) (α := α) (μ := μ) env' body

end Block
end DAG
end GraphSpec
end NN

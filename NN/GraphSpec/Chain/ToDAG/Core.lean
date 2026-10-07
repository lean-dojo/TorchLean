/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.Chain.Syntax
public import NN.GraphSpec.DAG.Syntax

/-!
# Structural conversion of sequential GraphSpec chains to DAG terms

This module embeds each sequential primitive as a DAG primitive and lowers composition to
explicit SSA-style let bindings.
-/

@[expose] public section

namespace NN
namespace GraphSpec

open Spec TorchLean
open TorchLean.Tensor

/-! ## Lowering: sequential chain → DAG term -/

/-!
GraphSpec has two surface syntaxes:

- `NN.GraphSpec.Chain`: a *sequential* DSL (`Chain` + `>>>`), ideal for pure pipelines.
- `NN.GraphSpec.DAG`: a *general* SSA/A-normal-form term language, ideal for sharing/skip
  connections.

The DAG term language is GraphSpec’s “general graph” core: it is the representation that can
express sharing and skip connections.

`Chain` exists because it is the clearest way to write pipelines, and it has its own
direct Spec semantics (`Interp.spec`) and program translation (`Chain.toProgram`).

This lowering is still useful whenever you want to *embed* a sequential pipeline into the DAG world
(e.g. to reuse DAG-only tooling, or to keep a single GraphSpec example surface that can export DAG
models).

The declarations below provide a structural lowering:

- `Chain.toDAGTerm` produces a `DAG.Term (ps ++ [σ]) τ`, i.e. a DAG term whose environment starts
  with the parameter list `ps` and ends with the (single) data input `σ`.
Notes:
- The lowering is *purely structural*: it introduces `let1` binders between stages to make the
  sequential flow explicit in SSA form.
- Each sequential `Primitive ps σ τ` is embedded as a DAG primitive op with inputs `ps ++ [σ]`.
  This embedding is generic: any custom GraphSpec primitive automatically becomes usable in the
  DAG world.
-/

namespace LowerToDAG

/-!
### Lowering internals

The definitions below (`primCall` and `toTerm`) implement the structural lowering. The
principal entry point is `Chain.toDAGTerm`.
-/

/-! ### Primitive embedding: `Primitive` → `DAG.PrimOp` -/

/--
Embed a sequential GraphSpec primitive as a DAG primitive op.

The resulting op has input shapes `ps ++ [σ]` (parameters followed by the data input).
 -/
def Primitive.toDAGPrimOp {ps : List Shape} {σ τ : Shape} (p : Primitive ps σ τ) :
    DAG.PrimOp (ps ++ [σ]) τ :=
  { name := p.name
    specFwd := fun {α} _storage _ctx xs =>
      let (params, xs') :=
        TorchLean.TensorPack.split
          (α := α) (ss₁ := ps) (ss₂ := [σ]) xs
      match xs' with
      | .cons x .nil => p.specFwd (α := α) params x
    program := fun {α} _storage _ctx => p.program (α := α)
  }

/-! ### Building well-typed DAG arguments for a primitive call -/


/--
Lower a unary `Primitive` application into the DAG term language.

Parameters are read from the middle `ps` segment of the ambient environment, in the same order as
the primitive's parameter ABI, and the final data input is supplied by `x`.
-/
def primCall
    {pre ps post extra : List Shape} {σ τ : Shape}
    (p : Primitive ps σ τ)
    (x : DAG.Term ((pre ++ ps ++ post) ++ extra) σ) :
    DAG.Term ((pre ++ ps ++ post) ++ extra) τ := by
  let Γ : List Shape := (pre ++ ps ++ post) ++ extra
  let op : DAG.PrimOp (ps ++ [σ]) τ := Primitive.toDAGPrimOp (ps := ps) (σ := σ) (τ := τ) p
  let paramsArgs : DAG.Args Γ ps :=
    DAG.Args.rename
      (fun v => DAG.Var.inLeft extra (DAG.Var.inLeft post (DAG.Var.inRight pre v)))
      (DAG.Args.vars ps)
  let args : DAG.Args Γ (ps ++ [σ]) :=
    DAG.Args.append paramsArgs (.cons x .nil)
  exact DAG.Term.op op args

/-! ### Chain lowering -/

/-- Lower a sequential `Chain` to an SSA-style `DAG.Term`, with parameters read from the
  environment. -/
def toTerm
    {pre ps post extra : List Shape} {σ τ : Shape}
    (g : Chain ps σ τ)
    (x : DAG.Term ((pre ++ ps ++ post) ++ extra) σ) :
    DAG.Term ((pre ++ ps ++ post) ++ extra) τ := by
  let Γ : List Shape := (pre ++ ps ++ post) ++ extra
  match g with
  | .id _ =>
      exact x
  | .prim p =>
      exact primCall p x
  | .seq (ps₁ := ps₁) (ps₂ := ps₂) (σ := σ) (τ := τm) (υ := τ) g₁ g₂ =>
      -- Outer env: `(pre ++ (ps₁ ++ ps₂) ++ post) ++ extra`
      let Γ0 : List Shape := (pre ++ (ps₁ ++ ps₂) ++ post) ++ extra
      -- Left subgraph sees post = ps₂ ++ post.
      have hΓ1 :
          (pre ++ ps₁ ++ (ps₂ ++ post)) ++ extra
          =
          Γ0 := by
        simp [Γ0, List.append_assoc]
      let t₁ : DAG.Term Γ0 τm :=
        DAG.Term.castEnv
          (toTerm (pre := pre) (ps := ps₁) (post := ps₂ ++ post) (extra := extra) g₁
            (by
              have x0 : DAG.Term Γ0 σ := by
                simpa [Γ, Γ0] using x
              exact
                DAG.Term.castEnv x0 (by simp [Γ0, List.append_assoc])))
          hΓ1
      -- `let1`-bind and translate the right subgraph.
      let bodyEnv : List Shape := Γ0 ++ [τm]
      let boundVar : DAG.Term bodyEnv τm :=
        DAG.Term.var (DAG.Var.last Γ0)
      -- Translate `g₂` under its own parenthesization, then cast back to `bodyEnv`.
      let rhsEnv : List Shape := ((pre ++ ps₁) ++ ps₂ ++ post) ++ (extra ++ [τm])
      have hRhs : rhsEnv = bodyEnv := by
        simp [rhsEnv, bodyEnv, Γ0, List.append_assoc]
      let boundVar' : DAG.Term rhsEnv τm :=
        DAG.Term.castEnv boundVar hRhs.symm
      let t₂' : DAG.Term rhsEnv τ :=
        toTerm (pre := pre ++ ps₁) (ps := ps₂) (post := post) (extra := extra ++ [τm]) g₂ boundVar'
      let t₂ : DAG.Term bodyEnv τ :=
        DAG.Term.castEnv t₂' hRhs
      let out : DAG.Term Γ0 τ := DAG.Term.let1 t₁ t₂
      exact out

/-! ### Public API -/

/--
Lower a sequential `Chain` to a DAG term with environment `ps ++ [σ]`.
 -/
def Chain.toDAGTerm {ps : List Shape} {σ τ : Shape} (g : Chain ps σ τ) :
    DAG.Term (ps ++ [σ]) τ :=
  let x : DAG.Term (([] ++ ps ++ []) ++ [σ]) σ :=
    DAG.Term.var (DAG.Var.last ([] ++ ps ++ []))
  -- `toTerm`’s environment is definitional `([] ++ ps ++ [] ++ [σ])`; normalize to `ps ++ [σ]`.
  by
    simpa [List.nil_append, List.append_nil, List.append_assoc] using
      (toTerm (pre := []) (ps := ps) (post := []) (extra := [σ]) g x)

end LowerToDAG

end GraphSpec
end NN

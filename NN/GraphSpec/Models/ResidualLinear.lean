/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.DAG.Model
public import NN.GraphSpec.DAG.Primitives.Core

/-!
# Residual Linear Block

A small DAG model for `x ↦ ReLU(Wx + b + x)`. The weight and bias have shapes
`[d, d]` and `[d]`; both start at zero, so the initial model computes `ReLU(x)`.

The input variable appears in both the linear branch and the skip branch. A `let1`
binds the linear result before the addition. Reusing an environment variable does
not recompute a preceding input expression.

Read this alongside `NN.GraphSpec.DAG.Syntax` and `NN.GraphSpec.DAG.Semantics` for typed variables
and `let1` semantics.
-/

@[expose] public section


namespace NN
namespace GraphSpec
namespace Models

open Spec TorchLean
open TorchLean.Tensor
open NN.GraphSpec.DAG

/--
Parameter ABI for the residual block.

The layout is exactly:

- `W : Tensor α [d, d]`
- `b : Tensor α [d]`

The parameter-free skip path reuses the input `x`.
-/
abbrev ResidualLinearParams (d : Nat) : List Shape :=
  [[d, d], [d]]

/--
Residual linear block in DAG form.

In ordinary math notation, this is

$$
x\mapsto\operatorname{ReLU}(Wx+b+x).
$$

The same input variable is used by the linear operation and the residual addition.
-/
def residualLinear (d : Nat) :
    DAG.Model (ps := ResidualLinearParams d) (ins := [[d]]) (τ := [d]) :=
  let Γ : List Shape :=
    [[d, d], [d], [d]]
  let w : DAG.Term Γ [d, d] :=
    DAG.Term.var (Γ := Γ) .head
  let b : DAG.Term Γ [d] :=
    DAG.Term.var (Γ := Γ) (.tail .head)
  let x : DAG.Term Γ [d] :=
    DAG.Term.var (Γ := Γ) (.tail (.tail .head))
  let y : DAG.Term Γ [d] :=
    DAG.Term.op (Γ := Γ) (DAG.PrimOp.linear (inDim := d) (outDim := d))
      (DAG.Args.cons w (DAG.Args.cons b (DAG.Args.cons x (DAG.Args.nil))))
  { initParams :=
      -- Deterministic, simple init: all zeros.
      let W0 : TorchLean.Tensor Float [d, d] := Tensor.zeros (α := Float) [d, d]
      let b0 : TorchLean.Tensor Float [d] := Tensor.zeros (α := Float) [d]
      .cons W0 (.cons b0 .nil)
    body :=
      DAG.Term.let1 y <|
        let Γ' : List Shape :=
          [[d, d], [d], [d], [d]]
        let yv : DAG.Term Γ' [d] :=
          DAG.Term.var (Γ := Γ') (.tail (.tail (.tail .head)))
        let add : DAG.Term Γ' [d] :=
          DAG.Term.op (Γ := Γ') (DAG.PrimOp.add (s := [d]))
            (DAG.Args.cons yv
              (DAG.Args.cons
                (DAG.Term.var (Γ := Γ') (.tail (.tail .head)))
                (DAG.Args.nil)))
        DAG.Term.op (Γ := Γ') (DAG.PrimOp.relu (s := [d])) (DAG.Args.cons add
          (DAG.Args.nil))
  }

end Models
end GraphSpec
end NN

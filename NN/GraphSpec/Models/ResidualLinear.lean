/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.DAG

/-!
# Residual Linear Block

This is the smallest “ResNet-like” example in the directory, and it is intentionally chosen to be
easy to read.

It shows the structural reason we need the DAG IR without dragging in convolution arithmetic:

```
y   = Linear(x)
out = ReLU(y + x)
```

Because `x` is consumed by both the main path and the skip path, a pure chain would have to
recompute the input path or hide sharing inside a special-purpose combinator. In `GraphSpec.DAG` we
express the sharing directly with `let1`.

This file is best read as a “hello world” for DAG-authored GraphSpec examples:

- one explicit parameter ABI,
- one shared intermediate,
- one multi-input primitive (`add`),
- one final nonlinearity.

References / citations:
- He et al. (2016), “Deep Residual Learning for Image Recognition” (ResNets).
- `NN.GraphSpec.DAG.Core` for the term language and semantics.
-/

@[expose] public section


namespace NN
namespace GraphSpec
namespace Models

open Spec
open Tensor
open NN.Tensor
open NN.GraphSpec.DAG

/--
Parameter ABI for the residual block.

The layout is exactly:

- `W : Mat d d`
- `b : Vec d`

The skip path is parameter-free; it simply reuses the input `x`.
-/
abbrev ResidualLinearParams (d : Nat) : List Shape :=
  [Shape.Mat d d, Shape.Vec d]

/--
Residual linear block in DAG form.

In ordinary math notation, this is

`x ↦ relu((W x + b) + x)`.

This is a good first DAG example because the only genuinely DAG-specific feature is sharing the
input between the main branch and the skip branch.
-/
def residualLinear (d : Nat) :
    DAG.Model (ps := ResidualLinearParams d) (ins := [Shape.Vec d]) (τ := Shape.Vec d) :=
  let ps : List Shape := ResidualLinearParams d
  let Γ : List Shape := ps ++ [Shape.Vec d]
  let w : DAG.Term Γ (Shape.Mat d d) :=
    DAG.Term.var (Γ := Γ) ⟨0, by simp [Γ, ps]⟩
  let b : DAG.Term Γ (Shape.Vec d) :=
    DAG.Term.var (Γ := Γ) ⟨1, by simp [Γ, ps]⟩
  let x : DAG.Term Γ (Shape.Vec d) :=
    DAG.Term.var (Γ := Γ) ⟨ps.length, by simp [Γ, ps, ResidualLinearParams]⟩
  let y : DAG.Term Γ (Shape.Vec d) :=
    DAG.Term.op (Γ := Γ) (DAG.PrimOp.linear (inDim := d) (outDim := d))
      (DAG.Args.cons w (DAG.Args.cons b (DAG.Args.cons x (DAG.Args.nil))))
  { initParams :=
      -- Deterministic, simple init: all zeros.
      let W0 : Tensor Float (Shape.Mat d d) := Spec.zeros (α := Float) (Shape.Mat d d)
      let b0 : Tensor Float (Shape.Vec d) := Spec.zeros (α := Float) (Shape.Vec d)
      .cons W0 (.cons b0 .nil)
    body :=
      DAG.Term.let1 y <|
        let Γ' := Γ ++ [Shape.Vec d]
        let yv : DAG.Term Γ' (Shape.Vec d) :=
          DAG.Term.var (Γ := Γ') ⟨Γ.length, by simp [Γ', Γ, ps, ResidualLinearParams]⟩
        let add : DAG.Term Γ' (Shape.Vec d) :=
          DAG.Term.op (Γ := Γ') (DAG.PrimOp.add (s := Shape.Vec d))
            (DAG.Args.cons yv
              (DAG.Args.cons
                (DAG.Term.var (Γ := Γ') ⟨ps.length, by simp [Γ', Γ, ps, ResidualLinearParams]⟩)
                (DAG.Args.nil)))
        DAG.Term.op (Γ := Γ') (DAG.PrimOp.relu (s := Shape.Vec d)) (DAG.Args.cons add
          (DAG.Args.nil))
  }

end Models
end GraphSpec
end NN

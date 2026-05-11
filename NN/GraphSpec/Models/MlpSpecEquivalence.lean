/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.Models.Mlp
public import NN.Spec.Models.Mlp

/-!
# MLP Spec Equivalence

This is the first model-level alignment theorem for GraphSpec.

We prove that interpreting the GraphSpec MLP architecture (`NN.GraphSpec.Models.mlp`) via the
GraphSpec Spec-interpreter (`NN.GraphSpec.Interp.spec`) agrees with the existing *hand-written*
Spec reference implementation (`NN.Spec.Models.Mlp.Examples.mlp_forward`), after packaging the
typed parameter list into two `LinearSpec`s in the obvious way.

Why this matters:

- It proves that `Primitive.linear` and sequential composition (`>>>`) compute the intended Spec
  formula for a concrete model.
- It gives a template for additional equivalence proofs where we compare a GraphSpec architecture to an
  existing hand-written Spec reference implementation.
- It anchors the intended meaning of the *parameter ABI* for the sequential DSL: when you compose
  graphs with `>>>`, the type-level parameter list concatenates, so each model has a canonical
  “parameter order” that refactors can be checked against.

Related context (informal pointers):

- Many projects formalize neural-network semantics in a proof assistant, but the combination of
  (1) a typed architecture DSL, (2) a pure “Spec” semantics, and (3) a compilation path to an
  executable runtime is still relatively uncommon.
- For comparison, see e.g.:
  - Brucker & Stell (2025), “Formalizing Neural Networks” (Isabelle/HOL; relates two network
    representations and supports importing models from TensorFlow.js),
  - Aleksandrov & Völlinger (2023), “Formalizing Piecewise Affine Activation Functions of Neural
    Networks in Coq” (formal layer semantics for verification).
-/

@[expose] public section


namespace NN
namespace GraphSpec
namespace Models

open Spec
open Tensor
open NN.Tensor

open Runtime.Autograd.Torch (TList)

/-- Parameter ABI for the 2-layer MLP `inDim → hidDim → outDim`: `(W₁,b₁,W₂,b₂)`. -/
abbrev MLPParams (inDim hidDim outDim : Nat) : List Shape :=
  [ Shape.Mat hidDim inDim, Shape.Vec hidDim
  , Shape.Mat outDim hidDim, Shape.Vec outDim ]

/--
**Theorem (GraphSpec MLP agrees with Spec reference).**

Fix dimensions `inDim → hidDim → outDim`. Let `params` be the 4-tensor parameter list
`(W₁, b₁, W₂, b₂)` and `x` an input vector.

Then the GraphSpec interpreter applied to the GraphSpec MLP graph computes exactly the same tensor
as the reference `Examples.mlp_forward` from `NN.Spec.Models.Mlp`, after interpreting the parameter
list as two `LinearSpec`s.

Informally, both sides compute the same explicit formula:

```
z₁ = W₁ · x + b₁
a₁ = relu(z₁)
out = W₂ · a₁ + b₂
```

where the dot/plus are the `Spec.linear_spec` and `Activation.relu_spec` operations already used by
the Spec model.
-/
theorem mlp_interp_eq_spec_mlp_forward
    {α : Type} [Context α]
    {inDim hidDim outDim : Nat}
    (params : TList α (MLPParams inDim hidDim outDim))
    (x : Tensor α (NN.Tensor.Shape.Vec inDim)) :
    Interp.spec (mlp (inDim := inDim) (hidDim := hidDim) (outDim := outDim)) params x
    =
    let (w1, b1, w2, b2) :=
      match params with
      | .cons w1 (.cons b1 (.cons w2 (.cons b2 .nil))) => (w1, b1, w2, b2)
    let l1 : Spec.LinearSpec α inDim hidDim := { weights := w1, bias := b1 }
    let l2 : Spec.LinearSpec α hidDim outDim := { weights := w2, bias := b2 }
    Examples.mlpForward (α := α) l1 l2 x := by
  cases params with
  | cons w1 params =>
    cases params with
    | cons b1 params =>
      cases params with
      | cons w2 params =>
        cases params with
        | cons b2 params =>
          cases params with
          | nil =>
            let l1 : Spec.LinearSpec α inDim hidDim := { weights := w1, bias := b1 }
            let l2 : Spec.LinearSpec α hidDim outDim := { weights := w2, bias := b2 }
            have hsplit1 :
                (Interp.splitParams
                    (ps₁ := [Shape.Mat hidDim inDim, Shape.Vec hidDim])
                    (ps₂ := [Shape.Mat outDim hidDim, Shape.Vec outDim])
                    (.cons w1 (.cons b1 (.cons w2 (.cons b2 .nil))))).1
                  =
                (.cons w1 (.cons b1 .nil) :
                  TList α [Shape.Mat hidDim inDim, Shape.Vec hidDim]) := by
              rfl
            have hsplit2 :
                (Interp.splitParams
                    (ps₁ := [Shape.Mat hidDim inDim, Shape.Vec hidDim])
                    (ps₂ := [Shape.Mat outDim hidDim, Shape.Vec outDim])
                    (.cons w1 (.cons b1 (.cons w2 (.cons b2 .nil))))).2
                  =
                (.cons w2 (.cons b2 .nil) :
                  TList α [Shape.Mat outDim hidDim, Shape.Vec outDim]) := by
              rfl
            have hspec :
                Interp.spec (mlp (inDim := inDim) (hidDim := hidDim) (outDim := outDim))
                  (.cons w1 (.cons b1 (.cons w2 (.cons b2 .nil)))) x
                  =
                Spec.linearSpec (α := α) l2
                  (Activation.reluSpec (Spec.linearSpec (α := α) l1 x)) := by
              unfold Interp.spec mlp
              simp [Graph.linear, Graph.relu, Primitive.linear, Primitive.relu]
              cases hsplit1
              cases hsplit2
              rfl
            have hR := Examples.mlp_spec_forward_eq (α := α) l1 l2 x
            simpa [Examples.mlpForward] using hspec.trans hR.symm

end Models
end GraphSpec
end NN

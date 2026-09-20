/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.Models.Mlp
public import NN.Spec.Models.Mlp
public import NN.GraphSpec.Chain.Semantics

/-!
# MLP Spec Equivalence

`mlp_interp_eq_spec_mlp_forward` identifies the GraphSpec chain interpreter with the
reference two-layer MLP: `Linear → ReLU → Linear`. Both receive the same weights and
biases in the order `(W₁, b₁, W₂, b₂)`.

This theorem compares pure spec evaluations. It does not assert that DAG lowering,
autograd, or a native backend preserves those values.
-/

@[expose] public section


namespace NN
namespace GraphSpec
namespace Models

open Spec TorchLean
open TorchLean.Tensor

/-- Parameter ABI for the 2-layer MLP: `(W₁, b₁, W₂, b₂)`. -/
abbrev MLPParams (inputWidth hiddenWidth outputWidth : Nat) : List Shape :=
  [[hiddenWidth, inputWidth], [hiddenWidth], [outputWidth, hiddenWidth], [outputWidth]]

/--
**Theorem (GraphSpec MLP agrees with Spec reference).**

Fix widths `inputWidth → hiddenWidth → outputWidth`. Let `params` be the 4-tensor parameter list
`(W₁, b₁, W₂, b₂)` and `x` an input vector.

Then the GraphSpec interpreter applied to the GraphSpec MLP graph computes exactly the same tensor
as the reference `Examples.mlpForward` from `NN.Spec.Models.Mlp`, after interpreting the parameter
list as two `LinearSpec`s.

Informally, both sides compute the same explicit formula:

$$
\begin{aligned}
z_1 &= W_1x+b_1,\\
a_1 &= \operatorname{ReLU}(z_1),\\
\mathrm{out} &= W_2a_1+b_2.
\end{aligned}
$$

where the dot/plus are the `Spec.linearSpec` and `Activation.reluSpec` operations already used by
the Spec model.
-/
theorem mlp_interp_eq_spec_mlp_forward
    {α : Type} [TorchLean.Storage α] [Context α]
    {inputWidth hiddenWidth outputWidth : Nat}
    (params : TorchLean.TensorPack α
      (MLPParams inputWidth hiddenWidth outputWidth))
    (x : TorchLean.Tensor α [inputWidth]) :
    Interp.spec
      (mlp
        (inputWidth := inputWidth)
        (hiddenWidth := hiddenWidth)
        (outputWidth := outputWidth))
      params x
    =
    let (w1, b1, w2, b2) :=
      match params with
      | .cons w1 (.cons b1 (.cons w2 (.cons b2 .nil))) => (w1, b1, w2, b2)
    let l1 : Spec.LinearSpec α inputWidth hiddenWidth := { weights := w1, bias := b1 }
    let l2 : Spec.LinearSpec α hiddenWidth outputWidth := { weights := w2, bias := b2 }
    Examples.mlpForward (α := α) l1 l2 x := by
  cases params with
  | cons w1 ps =>
    cases ps with
    | cons b1 ps =>
      cases ps with
      | cons w2 ps =>
        cases ps with
        | cons b2 ps =>
          cases ps
          rfl

end Models
end GraphSpec
end NN

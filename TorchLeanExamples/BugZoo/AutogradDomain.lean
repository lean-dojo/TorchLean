/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.VJPOp
public import TorchLean.Tensor.Elementwise
public import TorchLean.Tensor.Movement
public import TorchLean.Tensor.Layers.Dropout
public import TorchLean.Tensor.Layers.Embedding
public import TorchLean.Tensor.Layers.Linear
public import TorchLean.Tensor.Layers.Loss
public import TorchLean.Tensor.Layers.Loss.Axis
public import TorchLean.Tensor.Layers.Softmax

/-!
# BugZoo: autograd domains before masks

PyTorch's autograd notes document a sharp footgun. If a program computes $x/0$ and masks the bad
value afterward, the forward loss can look masked while the backward graph contains the undefined
division, and the documented example gives a `nan` gradient for the masked-out entry:

https://docs.pytorch.org/docs/main/notes/autograd.html#division-by-zero-in-autograd

What TorchLean offers here is a graph-level contract. The safe-domain choice is a spec node of
its own: use `safeDiv` in the recorded computation, then mask or weight the resulting tensor.
Proofs and importers downstream can then see the protected division in the graph.

Here `safeDiv` divides by `denominator + epsilon` and does not clamp the denominator away from
zero, so a denominator equal to `-epsilon` makes that sum zero. The theorems unfold the formula
and nothing more: finite forward values and correct gradients for arbitrary inputs or scalar
instances need domain and backend-conformance evidence.
-/

@[expose] public section

namespace TorchLean.Examples.BugZoo.AutogradDomain

open TorchLean.Tensor

/--
Divide by the epsilon-shifted denominator with `Tensor.safeDiv`, then apply the numeric
contribution mask. Meaningful values need `denominator + epsilon` to be nonzero.
-/
def maskAfterSafeDiv {s : Shape}
    {α : Type} [Storage α] [Context α]
    (mask numerator denominator : Tensor α s) : Tensor α s :=
  mask * Tensor.safeDiv numerator denominator

/--
Raw division followed by a numeric mask. A zero mask does not remove an undefined division.
-/
def maskAfterDiv {s : Shape}
    {α : Type} [Storage α] [Context α]
    (mask numerator denominator : Tensor α s) : Tensor α s :=
  mask * (numerator / denominator)

/--
`maskAfterSafeDiv` unfolds to the mask times `a / (b + Context.defaultEpsilon)` coordinatewise, so
the shifted denominator is visible in the spec before masking.
-/
theorem maskAfterSafeDiv_eq {s : Shape}
    {α : Type} [Storage α] [Context α]
    (mask numerator denominator : Tensor α s) :
    maskAfterSafeDiv mask numerator denominator =
      mask *
        (Tensor.zipWith (fun a b => a / (b + Context.defaultEpsilon))
          numerator denominator) := by
  rfl

/-- `maskAfterDiv` unfolds to the mask times the raw quotient `numerator / denominator`. -/
theorem maskAfterDiv_eq {s : Shape}
    {α : Type} [Storage α] [Context α]
    (mask numerator denominator : Tensor α s) :
    maskAfterDiv mask numerator denominator =
      mask * (numerator / denominator) := by
  rfl

end TorchLean.Examples.BugZoo.AutogradDomain

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Autograd.Ops

/-!
# BugZoo: autograd domains before masks

PyTorch's own autograd notes document a sharp footgun: if a program computes $x/0$ and only
masks the bad value afterward, the forward loss can look masked while the backward graph still
contains the undefined division. The documented example gives a `nan` gradient for the masked-out
entry:

https://docs.pytorch.org/docs/main/notes/autograd.html#division-by-zero-in-autograd

TorchLean's useful claim here is the graph-level contract. The safe-domain choice is an explicit
spec node: use `safedivSpec` in the computation that is recorded, then mask or weight the resulting
tensor. Downstream proofs and importers can then see the protected division directly in the graph
shape.

Here `safedivSpec` means division by `denominator + epsilon`; it does not clamp the denominator
away from zero. A denominator equal to `-epsilon` still makes that sum zero. The unfolding theorem
below identifies the formula, but does not establish finite forward values or correct gradients
for arbitrary inputs or scalar instances. Those require domain and backend-conformance evidence.
-/

@[expose] public section

open TorchLean

namespace NN.Examples.BugZoo.AutogradDomain

open TorchLean.Tensor

/--
Shift the denominator by epsilon before applying the numeric contribution mask.
The shifted denominator must still be nonzero.
-/
def maskAfterSafeDiv {s : Spec.Shape}
    {α : Type} [Storage α] [Context α]
    (mask numerator denominator : Tensor α s) : Tensor α s :=
  Tensor.mulSpec mask (Tensor.safedivSpec numerator denominator)

/--
Raw division followed by a numeric mask. A zero mask does not remove an undefined division.
-/
def unsafeDivThenMask {s : Spec.Shape}
    {α : Type} [Storage α] [Context α]
    (mask numerator denominator : Tensor α s) : Tensor α s :=
  Tensor.mulSpec mask (Tensor.divSpec numerator denominator)

/--
The shifted denominator is visible in the specification before masking.
-/
theorem maskAfterSafeDiv_uses_epsilon_denominator {s : Spec.Shape}
    {α : Type} [Storage α] [Context α]
    (mask numerator denominator : Tensor α s) :
    maskAfterSafeDiv mask numerator denominator =
      Tensor.mulSpec mask
        (Tensor.map2Spec (fun a b => a / (b + Context.defaultEpsilon))
          numerator denominator) := by
  rfl

/-- The contrast graph really is a raw division followed by masking. -/
theorem unsafeDivThenMask_unfold {s : Spec.Shape}
    {α : Type} [Storage α] [Context α]
    (mask numerator denominator : Tensor α s) :
    unsafeDivThenMask mask numerator denominator =
      Tensor.mulSpec mask (Tensor.divSpec numerator denominator) := by
  rfl

end NN.Examples.BugZoo.AutogradDomain

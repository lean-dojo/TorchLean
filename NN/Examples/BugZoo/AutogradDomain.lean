/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Autograd.Ops

/-!
# BugZoo: autograd domains before masks

PyTorch's own autograd notes document a sharp footgun: if a program computes `x / 0` and only
masks the bad value afterward, the forward loss can look masked while the backward graph still
contains the undefined division. The documented example gives a `nan` gradient for the masked-out
entry:

https://docs.pytorch.org/docs/main/notes/autograd.html#division-by-zero-in-autograd

TorchLean's useful claim here is deliberately narrow. We do not prove every user graph is finite.
Instead, the safe-domain choice is an explicit spec node: use `safedivSpec` in the computation that
is recorded, then mask or weight the resulting tensor.
-/

@[expose] public section

namespace NN.Examples.BugZoo.AutogradDomain

open Spec
open Spec.Tensor

/--
The safe pattern records an epsilon-protected division in the graph before the mask is applied.

This mirrors PyTorch's recommendation to mask before division or otherwise avoid recording an
undefined division. The mask is represented as a numeric weight tensor because this file is about
the algebraic graph boundary, not boolean indexing.
-/
def maskAfterSafeDiv {s : Spec.Shape}
    {α : Type} [Context α]
    (mask numerator denominator : Spec.Tensor α s) : Spec.Tensor α s :=
  Spec.Tensor.mulSpec mask (Spec.Tensor.safedivSpec numerator denominator)

/--
The risky shape of the graph: division is recorded first, and the mask is applied afterward.

This definition is intentionally present as a contrast class. It can still be useful at runtime
when a denominator is externally known to be safe, but that safety is not visible in the graph
shape itself.
-/
def unsafeDivThenMask {s : Spec.Shape}
    {α : Type} [Context α]
    (mask numerator denominator : Spec.Tensor α s) : Spec.Tensor α s :=
  Spec.Tensor.mulSpec mask (Spec.Tensor.divSpec numerator denominator)

/--
The safe-domain contract expands to division by `denominator + epsilon`, followed by the mask.

This is the checked TorchLean hook: downstream proofs and importers can distinguish the protected
graph from the "divide first, mask later" graph.
-/
theorem maskAfterSafeDiv_uses_epsilon_denominator {s : Spec.Shape}
    {α : Type} [Context α]
    (mask numerator denominator : Spec.Tensor α s) :
    maskAfterSafeDiv mask numerator denominator =
      Spec.Tensor.mulSpec mask
        (Spec.Tensor.map2Spec (fun a b => a / (b + Numbers.epsilon)) numerator denominator) := by
  rfl

/-- The contrast graph really is a raw division followed by masking. -/
theorem unsafeDivThenMask_unfold {s : Spec.Shape}
    {α : Type} [Context α]
    (mask numerator denominator : Spec.Tensor α s) :
    unsafeDivThenMask mask numerator denominator =
      Spec.Tensor.mulSpec mask (Spec.Tensor.divSpec numerator denominator) := by
  rfl

end NN.Examples.BugZoo.AutogradDomain

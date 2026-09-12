/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Context
public import NN.Tensor.Internal.Representation.Storage -- shake: keep

/-!
# BugZoo: ignored labels are a reduction contract

PyTorch issue #75181 reported `CrossEntropyLoss(ignore_index=...)` returning `nan` for an all-
ignored target case:

https://github.com/pytorch/pytorch/issues/75181

Represent ignored labels by explicit zero contributions and choose the empty-reduction policy.
-/

@[expose] public section

open TorchLean

namespace NN.Examples.BugZoo.IgnoredLabelLoss

/-- A per-example loss contributes exactly when its label is active. -/
def labelContribution {α : Type} [Zero α] (active : Bool) (loss : α) : α :=
  if active then loss else 0

/-- Ignored labels contribute no scalar loss. -/
@[simp] theorem ignored_label_contributes_zero {α : Type} [Zero α] (loss : α) :
    labelContribution false loss = 0 := by
  rfl

/-- Active labels contribute their ordinary scalar loss. -/
@[simp] theorem active_label_contributes_loss {α : Type} [Zero α] (loss : α) :
    labelContribution true loss = loss := by
  rfl

/--
One explicit empty-reduction policy: divide by an epsilon-shifted active count.

The type of `activeCount` does not enforce a nonnegative integer count, and `Context` alone gives
no positivity law for epsilon. Finiteness therefore depends on the chosen scalar instance and
valid inputs; the theorem below only unfolds the chosen formula.
-/
def safeMaskedMean {α : Type} [Storage α] [Context α] (total activeCount : α) : α :=
  total / (activeCount + Context.defaultEpsilon)

/-- The denominator policy for `safeMaskedMean` is visible in the definition. -/
theorem safeMaskedMean_uses_epsilon_denominator {α : Type} [Storage α] [Context α]
    (total activeCount : α) :
    safeMaskedMean total activeCount = total / (activeCount + Context.defaultEpsilon) := by
  rfl

end NN.Examples.BugZoo.IgnoredLabelLoss

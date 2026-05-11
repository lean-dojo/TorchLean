/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Context

/-!
# BugZoo: ignored labels are a reduction contract

PyTorch issue #75181 reported `CrossEntropyLoss(ignore_index=...)` returning `nan` for an all-
ignored target case:

https://github.com/pytorch/pytorch/issues/75181

The formal lesson is not "TorchLean has PyTorch's full label-indexed loss kernel." It is simpler:
ignored labels should be represented as an explicit contribution mask, and the empty-active-label
reduction policy should be stated in the spec rather than left as backend behavior.
-/

@[expose] public section

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

Real training code may choose a different policy, such as returning zero for an empty batch. The
important thing is that the policy is named and checkable instead of hidden inside a backend loss
kernel.
-/
def safeMaskedMean {α : Type} [Context α] (total activeCount : α) : α :=
  total / (activeCount + Numbers.epsilon)

/-- The denominator policy for `safeMaskedMean` is visible in the definition. -/
theorem safeMaskedMean_uses_epsilon_denominator {α : Type} [Context α]
    (total activeCount : α) :
    safeMaskedMean total activeCount = total / (activeCount + Numbers.epsilon) := by
  rfl

end NN.Examples.BugZoo.IgnoredLabelLoss

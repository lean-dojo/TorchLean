/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Context
public import TorchLean.Tensor.Storage -- shake: keep

/-!
# BugZoo: ignored labels are a reduction contract

PyTorch issue #75181 reported `CrossEntropyLoss(ignore_index=...)` returning `nan` when every
target label is ignored:

https://github.com/pytorch/pytorch/issues/75181

When no label is active, an averaged loss divides zero by zero. Each ignored label here contributes
zero, and the empty-reduction policy is a definition of its own, `safeMaskedMean`.
-/

@[expose] public section

namespace TorchLean.Examples.BugZoo.IgnoredLabel

/-- A per-example loss contributes exactly when its label is active. -/
def contribution {α : Type} [Zero α] (active : Bool) (loss : α) : α :=
  if active then loss else 0

/-- Ignored labels contribute no scalar loss. -/
@[simp] theorem contribution_false {α : Type} [Zero α] (loss : α) :
    contribution false loss = 0 := by
  rfl

/-- Active labels contribute their ordinary scalar loss. -/
@[simp] theorem contribution_true {α : Type} [Zero α] (loss : α) :
    contribution true loss = loss := by
  rfl

/--
One empty-reduction policy: divide the total by the active count shifted by epsilon.

Nothing in the type makes `activeCount` a nonnegative integer, and `Context` alone gives no
positivity law for epsilon. Whether the result is finite depends on the scalar instance and the
inputs, and the theorem that follows only unfolds the formula.
-/
def safeMaskedMean {α : Type} [Storage α] [Context α] (total activeCount : α) : α :=
  total / (activeCount + Context.defaultEpsilon)

/-- `safeMaskedMean` unfolds to `total / (activeCount + Context.defaultEpsilon)`. -/
theorem safeMaskedMean_eq {α : Type} [Storage α] [Context α]
    (total activeCount : α) :
    safeMaskedMean total activeCount = total / (activeCount + Context.defaultEpsilon) := by
  rfl

end TorchLean.Examples.BugZoo.IgnoredLabel

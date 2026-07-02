/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.List.Basic
public import NN.Proofs.Utils.List
public import NN.Spec.RL.Core

/-!
# RL Core Proofs

Small structural theorems about TorchLean's pure RL helper functions in `NN.Spec.RL.Core`.

The emphasis here is on *shape/structure* properties (mostly list lengths and truncation
behavior). These facts are often used to justify that derived quantities (discounted returns,
GAE advantages, etc.) align with the rollout data they came from.

These are kept modest but useful:

- discounted-return helpers preserve list length,
- GAE preserves list length,
- `returnsFromAdvantages` truncates to the shorter list.

References:

- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018),
  Sections 3-6 (returns, discounted backups, TD ideas):
  http://incompleteideas.net/book/the-book-2nd.html
- Schulman et al., “High-Dimensional Continuous Control Using Generalized Advantage Estimation”
  (2016): https://arxiv.org/abs/1506.02438
-/

@[expose] public section

namespace Proofs
namespace RL
namespace Core

/-- `discountedReturnsFrom` produces exactly one return per input reward. -/
theorem discountedReturnsFrom_length {α : Type} [Zero α] [Add α] [Mul α]
    (gamma : α) (rewards : List α) (bootstrap : α) :
    (Spec.RL.discountedReturnsFrom (α := α) gamma rewards bootstrap).length = rewards.length := by
  unfold Spec.RL.discountedReturnsFrom
  simpa using
    (List.foldl_cons_snd_length (l := rewards.reverse)
      (step := fun acc reward => reward + gamma * acc)
      (accScalar := bootstrap) (accList := ([] : List α)))

/-- `discountedReturns` preserves list length. -/
theorem discountedReturns_length {α : Type} [Zero α] [Add α] [Mul α]
    (gamma : α) (rewards : List α) :
    (Spec.RL.discountedReturns (α := α) gamma rewards).length = rewards.length := by
  simpa [Spec.RL.discountedReturns] using
    discountedReturnsFrom_length (α := α) gamma rewards 0

/-- `discountedReturnsDone` returns one value per paired reward/done entry. -/
theorem discountedReturnsDone_length_eq_min {α : Type} [Zero α] [One α] [Add α] [Mul α]
    (gamma : α) (rewards : List α) (dones : List Bool) (bootstrap : α) :
    (Spec.RL.discountedReturnsDone (α := α) gamma rewards dones bootstrap).length =
      min rewards.length dones.length := by
  unfold Spec.RL.discountedReturnsDone
  simpa [List.length_zip] using
    (List.foldl_cons_snd_length (l := (List.zip rewards dones).reverse)
      (step := fun acc step => Spec.RL.discountedBackup (α := α) step.1 gamma acc step.2)
      (accScalar := bootstrap) (accList := ([] : List α)))

/-- When reward and done lists have the same length, `discountedReturnsDone` preserves that length. -/
theorem discountedReturnsDone_length_of_eqLength {α : Type} [Zero α] [One α] [Add α] [Mul α]
    (gamma : α) (rewards : List α) (dones : List Bool) (bootstrap : α)
    (h : rewards.length = dones.length) :
    (Spec.RL.discountedReturnsDone (α := α) gamma rewards dones bootstrap).length =
      rewards.length := by
  rw [discountedReturnsDone_length_eq_min (α := α) (gamma := gamma) (rewards := rewards)
    (dones := dones) (bootstrap := bootstrap)]
  simp [h]

/-- Generalized-advantage-estimation returns one advantage per input step. -/
theorem generalizedAdvantageEstimation_length {α : Type}
    [Zero α] [One α] [Add α] [Mul α] [Sub α]
    (gamma lam : α) (steps : List (Spec.RL.AdvantageStep α)) :
    (Spec.RL.generalizedAdvantageEstimation (α := α) gamma lam steps).length = steps.length := by
  unfold Spec.RL.generalizedAdvantageEstimation
  simpa using
    (List.foldl_cons_snd_length (l := steps.reverse)
      (step := fun acc step =>
        let mask := Spec.RL.continueMask (α := α) step.done
        let delta := step.reward + gamma * mask * step.nextValue - step.value
        delta + gamma * lam * mask * acc)
      (accScalar := 0) (accList := ([] : List α)))

/-- `returnsFromAdvantages` truncates to the shorter of the two input lists. -/
theorem returnsFromAdvantages_length {α : Type} [Add α]
    (advantages values : List α) :
    (Spec.RL.returnsFromAdvantages (α := α) advantages values).length =
      min advantages.length values.length := by
  induction advantages generalizing values with
  | nil =>
      cases values <;> simp [Spec.RL.returnsFromAdvantages]
  | cons a as ih =>
      cases values <;> simp [Spec.RL.returnsFromAdvantages, ih]

end Core
end RL
end Proofs

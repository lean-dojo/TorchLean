/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.List.Basic
public import NN.Spec.RL.Core

/-!
# RL Core Proofs

Small structural theorems about TorchLean's pure RL helper functions in `NN.Spec.RL.Core`.

The emphasis here is on *shape/structure* properties (mostly list lengths and truncation
behaviour). These facts are often used to justify that derived quantities (discounted returns,
GAE advantages, etc.) align with the rollout data they came from.

These are intentionally modest but useful:

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

private theorem discountedReturnsFrom_fold_length {α : Type} [Zero α] [Add α] [Mul α]
    (gamma : α) (xs : List α) (accScalar : α) (accList : List α) :
    (((xs.foldl
      (fun (acc : α × List α) reward =>
        let g := reward + gamma * acc.1
        (g, g :: acc.2))
      (accScalar, accList)).2).length = accList.length + xs.length) := by
  induction xs generalizing accScalar accList with
  | nil =>
      simp
  | cons x xs ih =>
      simp [List.foldl, ih, Nat.add_left_comm, Nat.add_comm]

/-- `discountedReturnsFrom` produces exactly one return per input reward. -/
theorem discountedReturnsFrom_length {α : Type} [Zero α] [Add α] [Mul α]
    (gamma : α) (rewards : List α) (bootstrap : α) :
    (Spec.RL.discountedReturnsFrom (α := α) gamma rewards bootstrap).length = rewards.length := by
  unfold Spec.RL.discountedReturnsFrom
  simpa using
    (discountedReturnsFrom_fold_length (gamma := gamma) (xs := rewards.reverse) (accScalar := bootstrap)
      (accList := ([] : List α)))

/-- `discountedReturns` preserves list length. -/
theorem discountedReturns_length {α : Type} [Zero α] [Add α] [Mul α]
    (gamma : α) (rewards : List α) :
    (Spec.RL.discountedReturns (α := α) gamma rewards).length = rewards.length := by
  simpa [Spec.RL.discountedReturns] using
    discountedReturnsFrom_length (α := α) gamma rewards 0

private theorem discountedReturnsDone_fold_length {α : Type} [Zero α] [One α] [Add α] [Mul α]
    (gamma : α) (xs : List (α × Bool)) (accScalar : α) (accList : List α) :
    (((xs.foldl
      (fun (acc : α × List α) step =>
        let g := Spec.RL.discountedBackup (α := α) step.1 gamma acc.1 step.2
        (g, g :: acc.2))
      (accScalar, accList)).2).length = accList.length + xs.length) := by
  induction xs generalizing accScalar accList with
  | nil =>
      simp
  | cons x xs ih =>
      simp [List.foldl, ih, Nat.add_left_comm, Nat.add_comm]

/-- `discountedReturnsDone` returns one value per paired reward/done entry. -/
theorem discountedReturnsDone_length_eq_min {α : Type} [Zero α] [One α] [Add α] [Mul α]
    (gamma : α) (rewards : List α) (dones : List Bool) (bootstrap : α) :
    (Spec.RL.discountedReturnsDone (α := α) gamma rewards dones bootstrap).length =
      min rewards.length dones.length := by
  unfold Spec.RL.discountedReturnsDone
  simpa [List.length_zip] using
    (discountedReturnsDone_fold_length (gamma := gamma) (xs := (List.zip rewards dones).reverse)
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

private theorem gae_fold_length {α : Type} [Zero α] [One α] [Add α] [Mul α] [Sub α]
    (gamma lam : α) (xs : List (Spec.RL.AdvantageStep α)) (accAdv : α) (accList : List α) :
    (((xs.foldl
      (fun (acc : α × List α) step =>
        let mask := Spec.RL.continueMask (α := α) step.done
        let delta := step.reward + gamma * mask * step.nextValue - step.value
        let adv := delta + gamma * lam * mask * acc.1
        (adv, adv :: acc.2))
      (accAdv, accList)).2).length = accList.length + xs.length) := by
  induction xs generalizing accAdv accList with
  | nil =>
      simp
  | cons x xs ih =>
      simp [List.foldl, ih, Nat.add_assoc, Nat.add_comm]

/-- Generalized-advantage-estimation returns one advantage per input step. -/
theorem generalizedAdvantageEstimation_length {α : Type}
    [Zero α] [One α] [Add α] [Mul α] [Sub α]
    (gamma lam : α) (steps : List (Spec.RL.AdvantageStep α)) :
    (Spec.RL.generalizedAdvantageEstimation (α := α) gamma lam steps).length = steps.length := by
  unfold Spec.RL.generalizedAdvantageEstimation
  simpa using
    (gae_fold_length (gamma := gamma) (lam := lam) (xs := steps.reverse) (accAdv := 0)
      (accList := ([] : List α)))

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

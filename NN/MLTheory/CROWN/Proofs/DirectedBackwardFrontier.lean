/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.DirectedBackwardState

/-!
# The objective remaining at a backward-sweep frontier

Processed rows remain in the executable coefficient array. The invariant sums only unprocessed
nodes and the designated input, so those stale rows are never counted twice.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph.Internal
open scoped BigOperators

noncomputable section

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

/-- A finite real inner product, indexed by natural-number coordinates. -/
def dot (n : Nat) (a x : Nat → ℝ) : ℝ := ∑ i : Fin n, a i.val * x i.val

/-- The designated input remains live after its position has passed in the reverse sweep. -/
def pending (input k : Nat) : Finset Nat := insert input (Finset.range k)

/-- The exact objective represented by the rows that have not yet been discharged. -/
def frontierValue (dims : Nat → Nat) (v : Nat → Nat → ℝ) (input k : Nat)
    (f : Nat → Nat → ℝ) (c : ℝ) : ℝ :=
  (∑ id ∈ pending input k, dot (dims id) (f id) (v id)) + c

/-- The executable state encloses coefficients of one exact remaining objective. -/
def Represents (dims : Nat → Nat) (v : Nat → Nat → ℝ) (input k : Nat)
    (st : DirectedBackwardState α) (z : ℝ) : Prop :=
  ∃ f c, StateEncloses dims st f c ∧ frontierValue dims v input k f c = z

/-- Updating a live parent adds exactly its contribution to the pending objective. -/
theorem frontier_add {dims : Nat → Nat} {v : Nat → Nat → ℝ} {input k pid : Nat}
    (hp : pid ∈ pending input k) (f : Nat → Nat → ℝ) (a : Nat → ℝ) (c : ℝ) :
    frontierValue dims v input k
      (fun id i => if id = pid then f id i + a i else f id i) c =
      frontierValue dims v input k f c + dot (dims pid) a (v pid) := by
  have hrow (id : Nat) :
      dot (dims id) (fun i => if id = pid then f id i + a i else f id i) (v id) =
        dot (dims id) (f id) (v id) +
          if id = pid then dot (dims pid) a (v pid) else 0 := by
    by_cases h : id = pid
    · subst id
      simp [dot, add_mul, Finset.sum_add_distrib]
    · simp [h, dot]
  simp only [frontierValue, hrow, Finset.sum_add_distrib]
  simp only [Finset.sum_ite_eq', hp, ↓reduceIte]
  ring

/-- Dropping a non-input frontier row subtracts precisely that row's objective. -/
theorem frontier_succ {dims : Nat → Nat} {v : Nat → Nat → ℝ} {input k : Nat}
    (hk : k ≠ input) (f : Nat → Nat → ℝ) (c : ℝ) :
    frontierValue dims v input (k + 1) f c =
      frontierValue dims v input k f c + dot (dims k) (f k) (v k) := by
  have hset : pending input (k + 1) = insert k (pending input k) := by
    simp only [pending, Finset.range_add_one, Finset.insert_comm]
  have hnot : k ∉ pending input k := by simp [pending, hk]
  rw [frontierValue, hset, Finset.sum_insert hnot]
  unfold frontierValue
  ring

/-- Passing the designated input does not remove its objective from the invariant. -/
theorem frontier_input {dims : Nat → Nat} {v : Nat → Nat → ℝ} (input : Nat)
    (f : Nat → Nat → ℝ) (c : ℝ) :
    frontierValue dims v input (input + 1) f c =
      frontierValue dims v input input f c := by
  simp [frontierValue, pending, Finset.range_add_one]

/-- The final frontier contains only the designated input and the accumulated constant. -/
theorem frontier_zero (dims : Nat → Nat) (v : Nat → Nat → ℝ) (input : Nat)
    (f : Nat → Nat → ℝ) (c : ℝ) :
    frontierValue dims v input 0 f c = dot (dims input) (f input) (v input) + c := by
  simp [frontierValue, pending]

/-- Adding an enclosed coefficient contribution preserves the exact objective relation. -/
theorem represents_add {dims : Nat → Nat} {v : Nat → Nat → ℝ} {input k : Nat}
    {st : DirectedBackwardState α} {z : ℝ}
    (h : Represents dims v input k st z) (pid : Nat)
    (hp : pid < st.coeffs.size) (hlive : pid ∈ pending input k)
    (box : FlatBox α) (a : Nat → ℝ) (ha : RowEncloses box (dims pid) a) :
    Represents dims v input k (addDirectedCoeff st pid box)
      (z + dot (dims pid) a (v pid)) := by
  obtain ⟨f, c, hs, hz⟩ := h
  refine ⟨_, c, (addCoeff_encloses hs pid hp box a ha).1, ?_⟩
  rw [frontier_add hlive, hz]

/-- Adding an enclosed constant changes the represented objective by its exact real value. -/
theorem represents_addConstant {dims : Nat → Nat} {v : Nat → Nat → ℝ} {input k : Nat}
    {st : DirectedBackwardState α} {z : ℝ}
    (h : Represents dims v input k st z) (lo hi : α) (d : ℝ)
    (hd : value lo ≤ d ∧ d ≤ value hi) :
    Represents dims v input k (addDirectedConstant st lo hi) (z + d) := by
  obtain ⟨f, c, hs, hz⟩ := h
  refine ⟨f, c + d, addConstant_encloses hs lo hi d hd, ?_⟩
  simp only [frontierValue] at hz ⊢
  linarith

/-- An absent non-input row contributes zero when its position passes the frontier. -/
theorem represents_drop_none {dims : Nat → Nat} {v : Nat → Nat → ℝ} {input k : Nat}
    {st : DirectedBackwardState α} {z : ℝ}
    (h : Represents dims v input (k + 1) st z) (hk : k ≠ input)
    (hsize : k < st.coeffs.size) (hnone : st.coeffs[k]! = none) :
    Represents dims v input k st z := by
  obtain ⟨f, c, hs, hz⟩ := h
  have hf := hs.1 k hsize
  simp only [hnone] at hf
  have hdot : dot (dims k) (f k) (v k) = 0 := by simp [dot, hf]
  exact ⟨f, c, hs, by simpa only [frontier_succ hk, hdot, add_zero] using hz⟩

/-- A present row supplies the exact coefficient consumed by the next backward step. -/
theorem represents_drop_some {dims : Nat → Nat} {v : Nat → Nat → ℝ} {input k : Nat}
    {st : DirectedBackwardState α} {z : ℝ}
    (h : Represents dims v input (k + 1) st z) (hk : k ≠ input)
    (hsize : k < st.coeffs.size) {box : FlatBox α} (hsome : st.coeffs[k]! = some box) :
    ∃ a, RowEncloses box (dims k) a ∧
      Represents dims v input k st (z - dot (dims k) a (v k)) := by
  obtain ⟨f, c, hs, hz⟩ := h
  have hf := hs.1 k hsize
  simp only [hsome] at hf
  refine ⟨f k, hf, f, c, hs, ?_⟩
  rw [frontier_succ hk] at hz
  linarith

end

end NN.MLTheory.CROWN.Graph.DirectedBackward

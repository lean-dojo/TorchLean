/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Context.Real
public import NN.Spec.Core.Random

/-!
# Support of seeded random tensors

Each tensor coordinate uses the source generator's key and row-major index. Over the real
numbers, uniform draws lie in `[0, 1)`. Masks contain only zero and one, and their probability
endpoints give constant masks. These are properties of the seeded Spec operations.
-/

@[expose] public section

namespace Spec.Random

open TorchLean

/-- Reducing the generator word modulo a positive denominator gives a valid draw. -/
theorem sampleNat_lt (key : UInt64) (linearIndex : Nat) {denom : Nat} (hdenom : 0 < denom) :
    sampleNat key linearIndex denom < denom :=
  Nat.mod_lt _ hdenom

/-- Converting a natural draw to a real quotient gives a nonnegative value. -/
theorem sampleUnit_nonneg (u denom : Nat) : 0 ≤ sampleUnit (α := ℝ) u denom :=
  div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)

/-- A valid draw is strictly below one when the quotient is evaluated in the reals. -/
theorem sampleUnit_lt_one {u denom : Nat} (hu : u < denom) :
    sampleUnit (α := ℝ) u denom < 1 := by
  have hdenom : (0 : ℝ) < denom := Nat.cast_pos.mpr (Nat.zero_lt_of_lt hu)
  exact (div_lt_one hdenom).mpr (Nat.cast_lt.mpr hu)

/-- Every branch of the source keep decision returns either zero or one. -/
theorem keepBit_eq_zero_or_eq_one {α : Type} [Context α] (keepProb : α) (u denom : Nat) :
    keepBit keepProb u denom = 0 ∨ keepBit keepProb u denom = 1 := by
  unfold keepBit
  split
  · split
    · split <;> simp
    · simp
  · simp

/-- A zero keep probability drops every draw. -/
@[simp] theorem keepBit_zero (u denom : Nat) : keepBit (0 : ℝ) u denom = 0 := by
  simp [keepBit]

/-- A unit keep probability keeps every draw, including at the sampling boundary. -/
@[simp] theorem keepBit_one (u denom : Nat) : keepBit (1 : ℝ) u denom = 1 := by
  simp [keepBit]

namespace Internal

/-- The recursive uniform generator uses the offset plus the coordinate's row-major index. -/
theorem uniform_apply {α : Type} [Storage α] [Context α] (key : UInt64) {s : Shape}
    (offset : Nat) (c : s.Coord) :
    uniform (α := α) key offset c =
      sampleUnit (sampleNat key (offset + (Shape.Coord.linearize c).val)) (2 ^ 32) := by
  induction s generalizing offset with
  | scalar =>
    cases c
    simp [uniform, Shape.Coord.linearize, Tensor.Internal.Coord.linearize]
  | dim n s ih =>
    obtain ⟨i, c⟩ := c
    rw [uniform]
    simp only [Tensor.dim, Tensor.Internal.Rep.stack_apply]
    rw [ih]
    simp only [Shape.Coord.linearize, Tensor.Internal.Coord.linearize_cons_val,
      Shape.internalSize_eq, Nat.mul_comm, Nat.add_comm, Nat.add_assoc]

/-- The recursive mask generator uses the offset plus the coordinate's row-major index. -/
theorem mask_apply {α : Type} [Storage α] [Context α] (key : UInt64) (keepProb : α)
    {s : Shape} (offset : Nat) (c : s.Coord) :
    mask key keepProb offset c =
      keepBit keepProb (sampleNat key (offset + (Shape.Coord.linearize c).val)) (2 ^ 32) := by
  induction s generalizing offset with
  | scalar =>
    cases c
    simp [mask, Shape.Coord.linearize, Tensor.Internal.Coord.linearize]
  | dim n s ih =>
    obtain ⟨i, c⟩ := c
    rw [mask]
    simp only [Tensor.dim, Tensor.Internal.Rep.stack_apply]
    rw [ih]
    simp only [Shape.Coord.linearize, Tensor.Internal.Coord.linearize_cons_val,
      Shape.internalSize_eq, Nat.mul_comm, Nat.add_comm, Nat.add_assoc]

end Internal

/-- A uniform tensor coordinate is the source draw at its row-major index. -/
theorem uniform_apply {α : Type} [Storage α] [Context α] (key : UInt64) {s : Shape}
    (c : s.Coord) :
    uniform (α := α) key c =
      sampleUnit (sampleNat key (Shape.Coord.linearize c).val) (2 ^ 32) := by
  simpa only [uniform, Nat.zero_add] using Internal.uniform_apply (α := α) key 0 c

/-- Every seeded real uniform coordinate belongs to the half-open unit interval. -/
theorem uniform_mem_Ico (key : UInt64) {s : Shape} (c : s.Coord) :
    uniform (α := ℝ) key c ∈ Set.Ico (0 : ℝ) 1 := by
  rw [uniform_apply]
  exact ⟨sampleUnit_nonneg _ _, sampleUnit_lt_one (sampleNat_lt _ _ (by positivity))⟩

/-- A mask coordinate is the source keep decision at its row-major index. -/
theorem mask_apply {α : Type} [Storage α] [Context α] (key : UInt64) (keepProb : α)
    {s : Shape} (c : s.Coord) :
    mask key keepProb c =
      keepBit keepProb (sampleNat key (Shape.Coord.linearize c).val) (2 ^ 32) := by
  simpa only [mask, Nat.zero_add] using Internal.mask_apply key keepProb 0 c

/-- Every seeded mask coordinate is binary, for every keep probability. -/
theorem mask_eq_zero_or_eq_one {α : Type} [Storage α] [Context α]
    (key : UInt64) (keepProb : α) {s : Shape} (c : s.Coord) :
    mask key keepProb c = 0 ∨ mask key keepProb c = 1 := by
  rw [mask_apply]
  exact keepBit_eq_zero_or_eq_one _ _ _

/-- Every seeded real mask coordinate lies in the closed unit interval. -/
theorem mask_mem_Icc (key : UInt64) (keepProb : ℝ) {s : Shape} (c : s.Coord) :
    mask key keepProb c ∈ Set.Icc (0 : ℝ) 1 := by
  rcases mask_eq_zero_or_eq_one key keepProb c with h | h <;> simp [h]

/-- A seeded real mask at keep probability zero drops every coordinate. -/
@[simp] theorem mask_zero_apply (key : UInt64) {s : Shape} (c : s.Coord) :
    mask key (0 : ℝ) c = 0 := by
  simp only [mask_apply, keepBit_zero]

/-- A seeded real mask at keep probability one keeps every coordinate. -/
@[simp] theorem mask_one_apply (key : UInt64) {s : Shape} (c : s.Coord) :
    mask key (1 : ℝ) c = 1 := by
  simp only [mask_apply, keepBit_one]

end Spec.Random

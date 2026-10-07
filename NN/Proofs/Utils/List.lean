/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.List.Fold
public import Mathlib.Algebra.BigOperators.Group.Finset.Defs
public import Mathlib.Algebra.Ring.Defs
public import Mathlib.Data.Fintype.Basic

/-!
# List Utils

Small list-fold lemmas used throughout TorchLean proof files.  They are generic, not NN-specific,
so tensor, runtime-approximation, and RL proofs can share the same fold facts instead of carrying
local copies.
-/

@[expose] public section

namespace List

/--
`foldl max` upper bound helper.

If the initial accumulator and every `f i` are `≤ eps`, then the folded maximum is also `≤ eps`.
-/
theorem foldl_max_le_of_le {ι β : Type} [LinearOrder β] (l : List ι) (f : ι → β) {acc eps : β}
    (hacc : acc ≤ eps) (hf : ∀ i ∈ l, f i ≤ eps) :
    l.foldl (fun a i => max a (f i)) acc ≤ eps := by
  exact List.foldlRecOn (motive := fun a => a ≤ eps) l _ hacc
    fun _ h i hi => max_le h (hf i hi)

/--
Lower bound helper for `foldl max`.

The folded maximum is always at least as large as the initial accumulator.
-/
theorem le_foldl_max_init {ι β : Type} [LinearOrder β] (l : List ι) (f : ι → β) (acc : β) :
    acc ≤ l.foldl (fun a i => max a (f i)) acc := by
  exact List.foldlRecOn (motive := fun a => acc ≤ a) l _ (le_refl acc)
    fun a h i _ => h.trans (le_max_left a (f i))

/--
Membership helper for `foldl max`.

If `i ∈ l`, then `f i` is `≤` the folded maximum.
-/
theorem le_foldl_max_of_mem {ι β : Type} [LinearOrder β] (l : List ι) (f : ι → β) {acc : β} {i : ι}
    (hi : i ∈ l) :
    f i ≤ l.foldl (fun a j => max a (f j)) acc := by
  induction l generalizing acc with
  | nil =>
      cases hi
  | cons hd tl ih =>
      rcases (List.mem_cons.1 hi) with rfl | hiTail
      · have h1 : f i ≤ max acc (f i) := le_max_right _ _
        have h2 : max acc (f i) ≤ tl.foldl (fun a j => max a (f j)) (max acc (f i)) :=
          le_foldl_max_init tl f (max acc (f i))
        simpa [List.foldl] using le_trans h1 h2
      · simpa [List.foldl] using ih (acc := max acc (f hd)) hiTail

/-- A finite left fold of `max` returns either its initial accumulator or one of the values it
visited. This complements the order bounds above with the attainment fact needed by stable
softmax: the computed maximum is an actual input coordinate, so one max-shifted exponential is
exactly `exp 0 = 1`. -/
theorem foldl_max_eq_init_or_mem {ι β : Type} [LinearOrder β] (l : List ι) (f : ι → β) (acc : β) :
    l.foldl (fun a i => max a (f i)) acc = acc ∨
      ∃ i ∈ l, l.foldl (fun a j => max a (f j)) acc = f i := by
  induction l generalizing acc with
  | nil => simp
  | cons hd tl ih =>
      rcases ih (acc := max acc (f hd)) with hinit | ⟨i, hi, hvalue⟩
      · by_cases h : acc ≤ f hd
        · right
          refine ⟨hd, by simp, ?_⟩
          simpa [List.foldl, max_eq_right h] using hinit
        · left
          have hle : f hd ≤ acc := le_of_not_ge h
          simpa [List.foldl, max_eq_left hle] using hinit
      · right
        refine ⟨i, by simp [hi], ?_⟩
        simpa [List.foldl] using hvalue

-- Left-fold addition lemmas used to normalize finite sums in later proofs.

/--
Pointwise congruence for left folds of the shape `acc + f x`.

Many tensor and autograd proofs use executable folds for sums; this lemma lets them rewrite the
per-coordinate summand without re-proving a list induction locally.
-/
theorem foldl_add_congr {α β : Type} [Add α] (l : List β) (f g : β → α) (a : α)
    (h : ∀ x, f x = g x) :
    l.foldl (fun s x => s + f x) a = l.foldl (fun s x => s + g x) a := by
  rw [funext h]

/--
Turn `foldl (fun a x => a + f x) acc` into `acc + foldl (fun a x => a + f x) 0`.

This is the standard "peel off the initial accumulator" lemma for left folds over `+`.
-/
theorem foldl_add_init {α β : Type*} [AddMonoid β] (l : List α) (f : α → β) (acc : β) :
    l.foldl (fun a x => a + f x) acc = acc + l.foldl (fun a x => a + f x) 0 := by
  simpa only [List.foldl_map, add_zero] using
    (List.foldl_assoc (op := (· + ·)) (ha := ⟨add_assoc⟩)
      (l := l.map f) (a₁ := acc) (a₂ := 0))

/--
Distribute a fold of `g1 x + g2 x` into the sum of two folds.

This is the list-level form of “finite sum distributes over addition”.
-/
theorem foldl_add_distrib2 {α β : Type} [AddCommMonoid α] (l : List β) (g1 g2 : β → α) :
    ∀ a1 a2,
      l.foldl (fun s x => s + (g1 x + g2 x)) (a1 + a2) =
        l.foldl (fun s x => s + g1 x) a1 + l.foldl (fun s x => s + g2 x) a2 := by
  intro a1 a2
  induction l generalizing a1 a2 with
  | nil =>
      simp
  | cons hd tl ih =>
      simp [List.foldl]
      have hacc :
          (a1 + a2) + (g1 hd + g2 hd) = (a1 + g1 hd) + (a2 + g2 hd) := by
        simp [add_left_comm, add_comm]
      simpa [hacc, add_assoc, add_left_comm, add_comm] using
        (ih (a1 := a1 + g1 hd) (a2 := a2 + g2 hd))

/-- Pull multiplication by a fixed scalar through an additive left fold. -/
theorem foldl_add_mul_right {α β : Type} [Semiring α] (l : List β) (g : β → α) (a k : α) :
    l.foldl (fun acc x => acc + g x * k) (a * k) =
      (l.foldl (fun acc x => acc + g x) a) * k := by
  exact List.foldl_hom (fun x => x * k) (fun x y => (add_mul x (g y) k).symm)

/--
Rewrite the canonical `List.finRange` addition fold into a `Finset.univ` sum.

Specs use `List.foldl` because it computes well; proofs usually want `Finset.sum` so standard
big-operator lemmas apply.
-/
theorem finRange_foldl_add_eq_finset_sum {β : Type} [AddCommMonoid β] {n : Nat} (f : Fin n → β) :
    (List.finRange n).foldl (fun s i => s + f i) 0 = (Finset.univ : Finset (Fin n)).sum f := by
  classical
  calc
    _ = ((List.finRange n).map f).sum := by
      rw [List.sum_eq_foldl, List.foldl_map]
    _ = _ := by
      simp only [Finset.sum, Finset.val_univ_fin, Multiset.map_coe, Multiset.sum_coe]

/--
Accumulator form of `finRange_foldl_add_eq_finset_sum`.

This avoids re-proving "peel off the initial accumulator, then rewrite the zero fold" in large
finite-sum proofs.
-/
theorem finRange_foldl_add_acc {β : Type} [AddCommMonoid β] {n : Nat} (f : Fin n → β) (acc : β) :
    (List.finRange n).foldl (fun s i => s + f i) acc =
      acc + (Finset.univ : Finset (Fin n)).sum f := by
  rw [foldl_add_init, finRange_foldl_add_eq_finset_sum]

end List

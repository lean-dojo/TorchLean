/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Algebra.Order.Group.MinMax
public import Mathlib.Data.List.FinRange

/-!
# List Utils

Small list-fold lemmas used throughout TorchLean proof files.

These are intentionally generic (not NN-specific) and live in a single place so large proofs do not
accumulate many near-duplicate "foldl max" / "foldl add" helper lemmas.
-/

@[expose] public section

namespace List

/--
`foldl max` upper bound helper.

If the initial accumulator and every `f i` are `≤ eps`, then the folded maximum is also `≤ eps`.
-/
lemma foldl_max_le_of_le {ι β : Type} [LinearOrder β] (l : List ι) (f : ι → β) {acc eps : β}
    (hacc : acc ≤ eps) (hf : ∀ i ∈ l, f i ≤ eps) :
    l.foldl (fun a i => max a (f i)) acc ≤ eps := by
  induction l generalizing acc with
  | nil =>
      simpa using hacc
  | cons a l ih =>
      have ha : f a ≤ eps := hf a (by simp)
      have hacc' : max acc (f a) ≤ eps := max_le hacc ha
      have hf' : ∀ i ∈ l, f i ≤ eps := by
        intro i hi
        exact hf i (by simp [hi])
      simpa [List.foldl] using ih (acc := max acc (f a)) hacc' hf'

/--
Lower bound helper for `foldl max`.

The folded maximum is always at least as large as the initial accumulator.
-/
lemma le_foldl_max_init {ι β : Type} [LinearOrder β] (l : List ι) (f : ι → β) (acc : β) :
    acc ≤ l.foldl (fun a i => max a (f i)) acc := by
  induction l generalizing acc with
  | nil =>
      simp
  | cons a l ih =>
      have h1 : acc ≤ max acc (f a) := le_max_left _ _
      have h2 : max acc (f a) ≤ l.foldl (fun a i => max a (f i)) (max acc (f a)) := by
        simpa using (ih (acc := max acc (f a)))
      simpa [List.foldl] using le_trans h1 h2

/--
Membership helper for `foldl max`.

If `i ∈ l`, then `f i` is `≤` the folded maximum.
-/
lemma le_foldl_max_of_mem {ι β : Type} [LinearOrder β] (l : List ι) (f : ι → β) {acc : β} {i : ι}
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

-- ---------------------------------------------------------------------------
-- Addition folds
-- ---------------------------------------------------------------------------

/--
Turn `foldl (fun a x => a + f x) acc` into `acc + foldl (fun a x => a + f x) 0`.

This is the standard "peel off the initial accumulator" lemma for left folds over `+`.
-/
lemma foldl_add_init {α β : Type} [AddMonoid β] (l : List α) (f : α → β) (acc : β) :
    l.foldl (fun a x => a + f x) acc = acc + l.foldl (fun a x => a + f x) 0 := by
  induction l generalizing acc with
  | nil =>
      simp
  | cons x xs ih =>
      -- Use the IH twice: once for `acc + f x`, once for `f x`.
      have h1 :
          xs.foldl (fun a y => a + f y) (acc + f x) =
            (acc + f x) + xs.foldl (fun a y => a + f y) 0 := ih (acc := acc + f x)
      have h2 :
          xs.foldl (fun a y => a + f y) (f x) =
            (f x) + xs.foldl (fun a y => a + f y) 0 := ih (acc := f x)
      simp [List.foldl, h1, h2, add_assoc]

lemma add_foldl_add0 {α β : Type} [AddMonoid β] (l : List α) (f : α → β) (acc : β) :
    acc + l.foldl (fun a x => a + f x) 0 = l.foldl (fun a x => a + f x) acc := by
  simpa using (foldl_add_init (l := l) (f := f) (acc := acc)).symm

end List


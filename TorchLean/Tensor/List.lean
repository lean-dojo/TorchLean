/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.List.Fold
public import Mathlib.Algebra.BigOperators.Group.Finset.Defs
public import Mathlib.Data.Fintype.Basic
import all Batteries.Data.List.Basic

/-!
# Lemmas about list folds, `mapM`, chunks and array prefixes

Generic lemmas about left folds over lists. A fold with `max` is bounded by any common upper bound,
dominates its initial accumulator and every visited value, and returns one of these values. A fold
with `+` over `List.finRange n` equals its initial accumulator plus the `Finset` sum.

An optional `mapM` over a list `l` keeps the length when it succeeds, and it returns `l'` when `l'`
has the length of `l` and the mapped function gives each entry of `l'` at its position. A successful
`mapM` in `Except` gives a successful optional `mapM`, with the converted results, for every reader
that agrees with it after the conversion. An array whose list is a prefix of another array's list
has the same entries at its indices, and `x` is entry `a.size` of every array whose list has the
list of `a.push x` as a prefix.

For a nonempty `l`, Batteries' `l.toChunks n` is `[l]` when `n = 0` and otherwise the first `n`
entries followed by the chunks of the rest. The chunks concatenate to `l`, and for `n > 0` each has
at most `n` entries and there are `⌈l.length / n⌉` of them.
-/

@[expose] public section

namespace List

/--
A left fold of `max` is at most `eps` when its initial accumulator and every `f i` are.
-/
theorem foldl_max_le_of_le {ι β : Type} [LinearOrder β] (l : List ι) (f : ι → β) {acc eps : β}
    (hacc : acc ≤ eps) (hf : ∀ i ∈ l, f i ≤ eps) :
    l.foldl (fun a i => max a (f i)) acc ≤ eps := by
  exact List.foldlRecOn (motive := fun a => a ≤ eps) l _ hacc
    fun _ h i hi => max_le h (hf i hi)

/--
A left fold of `max` is at least its initial accumulator.
-/
theorem le_foldl_max_init {ι β : Type} [LinearOrder β] (l : List ι) (f : ι → β) (acc : β) :
    acc ≤ l.foldl (fun a i => max a (f i)) acc := by
  exact List.foldlRecOn (motive := fun a => acc ≤ a) l _ (le_refl acc)
    fun a h i _ => h.trans (le_max_left a (f i))

/--
A left fold of `max` over `l` is at least `f i` for every `i ∈ l`.
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

/-- A left fold of `max` returns its initial accumulator or one of the values `f i` it visits.
When the accumulator is itself one of the values, as the first logit is in stable softmax, the
maximum is attained, so one max-shifted exponential is `exp 0 = 1`. -/
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

/-- In an additive commutative monoid, a left fold of `s + f i` over `List.finRange n` from `acc`
equals `acc + ∑ i, f i`. -/
theorem finRange_foldl_add_eq_add_sum {β : Type} [AddCommMonoid β] {n : Nat}
    (f : Fin n → β) (acc : β) :
    (List.finRange n).foldl (fun s i => s + f i) acc =
      acc + (Finset.univ : Finset (Fin n)).sum f := by
  simpa only [List.foldl_map, add_zero, List.sum_eq_foldl, Finset.sum, Finset.val_univ_fin,
    Multiset.map_coe, Multiset.sum_coe] using
    List.foldl_assoc (op := (· + ·)) (ha := ⟨add_assoc⟩) (l := (List.finRange n).map f)
      (a₁ := acc) (a₂ := 0)

/-- A successful list `mapM` in `Except` is a successful optional `mapM` of any reader that agrees
with it, after a conversion, on the successes of the list's entries. -/
theorem mapM_option_of_except {ε β γ δ : Type} {check : β → Except ε γ}
    {read : β → Option δ} {convert : γ → δ} :
    ∀ {l : List β} {r : List γ}, (∀ x ∈ l, ∀ y, check x = .ok y → read x = some (convert y)) →
      l.mapM check = .ok r → l.mapM read = some (r.map convert)
  | [], r, _, h => by
      simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at h
      subst h; rfl
  | x :: xs, r, hread, h => by
      rw [List.mapM_cons] at h
      cases hx : check x with
      | error e => simp [hx, bind, Except.bind] at h
      | ok y =>
        cases hxs : xs.mapM check with
        | error e => simp [hx, hxs, bind, Except.bind] at h
        | ok ys =>
          simp only [hx, hxs, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
          subst h
          rw [List.mapM_cons, hread x (by simp) y hx,
            mapM_option_of_except (fun z hz => hread z (by simp [hz])) hxs]
          rfl

/-- A successful optional `mapM` keeps the length. -/
theorem mapM_some_length {β γ : Type} {f : β → Option γ} :
    ∀ {l : List β} {l' : List γ}, l.mapM f = some l' → l'.length = l.length
  | [], l', h => by
      simp only [List.mapM_nil, pure, Option.some.injEq] at h
      subst h; rfl
  | x :: xs, l', h => by
      simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, pure,
        Option.some.injEq] at h
      obtain ⟨y, -, ys, hys, rfl⟩ := h
      simp [mapM_some_length hys]

/-- An optional `mapM` succeeds with `l'` when it gives each entry of `l'` at its position. -/
theorem mapM_some_of_getElem {β γ : Type} {f : β → Option γ} :
    ∀ {l : List β} {l' : List γ}, l.length = l'.length →
      (∀ m (hm : m < l.length), f l[m] = l'[m]?) → l.mapM f = some l'
  | [], [], _, _ => rfl
  | [], _ :: _, hlen, _ => by simp at hlen
  | _ :: _, [], hlen, _ => by simp at hlen
  | x :: xs, y :: ys, hlen, h => by
      have h0 : f x = some y := h 0 (by simp)
      have hrest := mapM_some_of_getElem (f := f) (l := xs) (l' := ys) (by simpa using hlen)
        (fun m hm => h (m + 1) (by simp; omega))
      rw [List.mapM_cons, h0, hrest]
      rfl

/-- The loop of `toChunks n` with the chunk `acc₁` in progress and the finished chunks `acc₂` fills
`acc₁` up to `n` entries and continues with the chunks of the rest. -/
private theorem toChunks_go {β : Type} {n : Nat} (hn : 0 < n) :
    ∀ (l : List β) (acc₁ : Array β) (acc₂ : Array (List β)), 0 < acc₁.size → acc₁.size ≤ n →
      List.toChunks.go n l acc₁ acc₂ =
        acc₂.toList ++ (acc₁.toList ++ l.take (n - acc₁.size)) ::
          List.toChunks n (l.drop (n - acc₁.size))
  | [], acc₁, acc₂, _, _ => by simp [List.toChunks.go, List.toChunks]
  | x :: xs, acc₁, acc₂, h0, hle => by
      obtain ⟨m, rfl⟩ : ∃ m, n = m + 1 := ⟨n - 1, by omega⟩
      unfold List.toChunks.go
      by_cases hsize : acc₁.size = m + 1
      · rw [ite_eq_left (by simp [hsize]), toChunks_go hn xs _ _ (by simp) (by simp)]
        simp [hsize, List.toChunks, toChunks_go hn xs #[x] #[] (by simp) (by simp)]
      · rw [ite_eq_right (by simpa using hsize), toChunks_go hn xs _ _ (by simp) (by simp; omega)]
        have hk : m + 1 - acc₁.size = (m + 1 - (acc₁.size + 1)) + 1 := by omega
        simp [hk]

/-- `l.toChunks n` is empty for the empty list, `[l]` for `n = 0`, and otherwise the first `n`
entries followed by the chunks of the rest. -/
theorem toChunks_eq_ite {β : Type} (n : Nat) (l : List β) :
    l.toChunks n =
      if l = [] then [] else if n = 0 then [l] else l.take n :: (l.drop n).toChunks n := by
  match n, l with
  | _, [] => simp [List.toChunks]
  | 0, x :: xs => simp [List.toChunks]
  | m + 1, x :: xs =>
      simp only [List.toChunks, reduceCtorEq, ite_false, Nat.add_one_ne_zero]
      rw [toChunks_go (Nat.succ_pos m) xs #[x] #[] (by simp) (by simp)]
      simp [List.toChunks]

/-- The chunks concatenate to the list. -/
theorem flatten_toChunks {β : Type} (n : Nat) (l : List β) :
    (l.toChunks n).flatten = l := by
  induction l using WellFounded.induction (r := InvImage (· < ·) List.length)
    (InvImage.wf _ Nat.lt_wfRel.wf) with
  | h l ih =>
    rw [toChunks_eq_ite]
    split_ifs with hl hn
    · simp [hl]
    · simp
    · have : 0 < l.length := List.length_pos_iff.mpr hl
      rw [List.flatten_cons, ih (l.drop n) (by simp only [InvImage, List.length_drop]; omega),
        List.take_append_drop]

/-- For `n > 0` every chunk has at most `n` entries. -/
theorem length_le_of_mem_toChunks {β : Type} {n : Nat} (hn : 0 < n) (l : List β) :
    ∀ chunk ∈ l.toChunks n, chunk.length ≤ n := by
  induction l using WellFounded.induction (r := InvImage (· < ·) List.length)
    (InvImage.wf _ Nat.lt_wfRel.wf) with
  | h l ih =>
    rw [toChunks_eq_ite]
    split_ifs with hl hn'
    · simp
    · omega
    · intro chunk hchunk
      rcases List.mem_cons.mp hchunk with rfl | hchunk
      · simp
      · have : 0 < l.length := List.length_pos_iff.mpr hl
        exact ih (l.drop n) (by simp only [InvImage, List.length_drop]; omega) chunk hchunk

/-- For `n > 0` a list of `m` entries has `⌈m / n⌉` chunks. -/
theorem length_toChunks {β : Type} {n : Nat} (hn : 0 < n) (l : List β) :
    (l.toChunks n).length = (l.length + n - 1) / n := by
  induction l using WellFounded.induction (r := InvImage (· < ·) List.length)
    (InvImage.wf _ Nat.lt_wfRel.wf) with
  | h l ih =>
    rw [toChunks_eq_ite]
    split_ifs with hl hn'
    · subst hl
      simp only [List.length_nil, List.length_nil, Nat.zero_add]
      exact (Nat.div_eq_of_lt (by omega)).symm
    · omega
    · have hpos : 0 < l.length := List.length_pos_iff.mpr hl
      rw [List.length_cons, ih (l.drop n) (by simp only [InvImage, List.length_drop]; omega),
        List.length_drop]
      rcases Nat.lt_or_ge l.length n with hlt | hge
      · rw [Nat.sub_eq_zero_of_le hlt.le, Nat.zero_add, Nat.div_eq_of_lt (by omega)]
        exact (Nat.div_eq_of_lt_le (by omega) (by omega)).symm
      · have : l.length + n - 1 = l.length - n + n - 1 + n := by omega
        rw [this, Nat.add_div_right _ hn]

end List

namespace Array

/-- An entry of an array whose list is a prefix of another's is the other's entry. -/
theorem getElem?_of_toList_prefix {β : Type} {a b : Array β} (h : a.toList <+: b.toList)
    {j : Nat} {x : β} (hx : a[j]? = some x) : b[j]? = some x := by
  have hj : j < a.toList.length := by simpa using (Array.getElem?_eq_some_iff.mp hx).1
  obtain ⟨t, ht⟩ := h
  rw [← Array.getElem?_toList] at hx ⊢
  rw [← ht, List.getElem?_append_left hj]
  exact hx

/-- The entry an array push appends is the entry at its index in any extension. -/
theorem getElem?_of_push_toList_prefix {β : Type} {a b : Array β} {x : β}
    (h : (a.push x).toList <+: b.toList) : b[a.size]? = some x := by
  obtain ⟨t, ht⟩ := h
  rw [← Array.getElem?_toList, ← ht]
  simp

end Array

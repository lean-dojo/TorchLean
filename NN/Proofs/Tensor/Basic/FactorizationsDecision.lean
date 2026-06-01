/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Basic.FactorizationsVariational
public import NN.Proofs.Tensor.Basic.FactorizationsReconstruction

/-!
# CHD discovery decision layer (`decision.py`, `_GraphDiscoveryMain.py`)

[`FactorizationsVariational`](./FactorizationsVariational.lean) proved that CHD's `noise` level is a
spectral fraction in `[0,1]`. This file closes the gap up to the *graph-structure decisions* CHD makes
from those numbers — the outer discovery loop. Each is a deterministic comparison over finite data; the
executable specs (`Spec.argMinFn`, `Spec.kernelChooserFn`, …) mirror the Python verbatim, and the
theorems here establish their selection guarantees:

* **`argMinFn_le` / `argMaxFn_le`** — the fold-based `np.argmin`/`np.argmax` really return the index of a
  least / greatest element (the activation prune step, the mode chooser).
* **`kernelChooserFn_eq_some` / `kernelChooserFn_eq_none`** — `MinNoiseKernelChooser` is *sound and
  complete*: it returns `some s` with `s` valid and of least `noise` among valid kernels exactly when a
  valid kernel exists, and `none` otherwise. The `noise ≤ 1` precondition that makes the `2` sentinel
  work is exactly the verified `varNoiseFn_le_one`.
* **`modeChooserFn_ge`** — `MaxIncrementModeChooser` returns the iteration of largest `noise` increment.
* **`allPrunedFn_iff`** — the stopping test `np.all(active_modes == 0)` holds iff every ancestor is
  pruned.

Scope honesty: everything is exact over `ℝ`. The comparisons in the specs go through the `Context` order
test (`gtBool`/`ltBool`); `gtBool_true_iff` (from `FactorizationsReconstruction`) bridges them to the
real `<`, after which the selection proofs are pure order theory over `Fin (n+1)`.
-/

@[expose] public section

namespace Spec.Factorization

open Spec.Factorization.Reconstruction

variable {n : Nat}

/-! ## Bridge: the `Context` order tests over `ℝ` -/

/-- Over `ℝ`, `gtBool x y` is the decidable `y < x`. -/
theorem gtBool_eq_decide (x y : ℝ) : Context.gtBool x y = decide (y < x) := by
  by_cases h : y < x
  · have h1 : Context.gtBool x y = true := gtBool_true_iff.mpr h
    rw [h1]; simp [h]
  · have h1 : Context.gtBool x y = false := by
      cases hc : Context.gtBool x y with
      | false => rfl
      | true => exact absurd (gtBool_true_iff.mp hc) h
    rw [h1]; simp [h]

/-- Over `ℝ`, `ltBool x y` is the decidable `x < y`. -/
theorem ltBool_eq_decide (x y : ℝ) : Spec.ltBool x y = decide (x < y) := by
  rw [Spec.ltBool, gtBool_eq_decide]

/-! ## A generic fold-selection lemma

Both `argMinFn` and `argMaxFn` are `List.foldl`s of the shape "keep the running best, swap in `j` when
the `Bool` test `cmp (key j) (key best)` fires". The next lemma proves such a fold returns a `le`-best
index over `init :: l`, for any preorder `le` whose strict part is decided by `cmp`. Instantiating
`le := (· ≤ ·)` gives the argmax guarantee; `le := (· ≥ ·)` gives argmin. -/

private theorem foldl_select {m : Nat} (key : Fin m → ℝ) (cmp : ℝ → ℝ → Bool)
    (le : ℝ → ℝ → Prop) (hrefl : ∀ x, le x x)
    (htrans : ∀ x y z, le x y → le y z → le x z)
    (htrue : ∀ x y, cmp x y = true → le y x) (hfalse : ∀ x y, cmp x y = false → le x y)
    (init : Fin m) (l : List (Fin m)) :
    le (key init)
        (key (l.foldl (fun best j => if cmp (key j) (key best) then j else best) init))
      ∧ ∀ j ∈ l,
        le (key j)
          (key (l.foldl (fun best j => if cmp (key j) (key best) then j else best) init)) := by
  induction l generalizing init with
  | nil => exact ⟨hrefl _, by simp⟩
  | cons j₀ t ih =>
    rw [List.foldl_cons]
    set best' := (if cmp (key j₀) (key init) then j₀ else init) with hb
    have hstep_init : le (key init) (key best') := by
      by_cases hcmp : cmp (key j₀) (key init) = true
      · rw [hb, if_pos hcmp]; exact htrue _ _ hcmp
      · rw [hb, if_neg hcmp]; exact hrefl _
    have hstep_j0 : le (key j₀) (key best') := by
      by_cases hcmp : cmp (key j₀) (key init) = true
      · rw [hb, if_pos hcmp]; exact hrefl _
      · rw [hb, if_neg hcmp]
        rw [Bool.not_eq_true] at hcmp
        exact hfalse _ _ hcmp
    obtain ⟨hm, hc⟩ := ih best'
    refine ⟨htrans _ _ _ hstep_init hm, ?_⟩
    intro j hj
    rcases List.mem_cons.mp hj with rfl | hj'
    · exact htrans _ _ _ hstep_j0 hm
    · exact hc j hj'

/-! ## `argmin` / `argmax` -/

/-- **`argMinFn` returns the index of a least element.** -/
theorem argMinFn_le (a : Fin (n + 1) → ℝ) (j : Fin (n + 1)) :
    a (Spec.argMinFn a) ≤ a j := by
  have h := foldl_select (key := a) (cmp := Spec.ltBool) (le := fun p q => q ≤ p)
    (fun x => le_refl x) (fun x y z hxy hyz => le_trans hyz hxy)
    (fun x y hh => by rw [ltBool_eq_decide] at hh; exact (of_decide_eq_true hh).le)
    (fun x y hh => by rw [ltBool_eq_decide] at hh; exact not_lt.mp (of_decide_eq_false hh))
    (0 : Fin (n + 1)) (List.finRange (n + 1))
  exact h.2 j (List.mem_finRange j)

/-- **`argMaxFn` returns the index of a greatest element.** -/
theorem argMaxFn_le (a : Fin (n + 1) → ℝ) (j : Fin (n + 1)) :
    a j ≤ a (Spec.argMaxFn a) := by
  have h := foldl_select (key := a) (cmp := Context.gtBool) (le := fun p q => p ≤ q)
    (fun x => le_refl x) (fun x y z => le_trans)
    (fun x y hh => by rw [gtBool_eq_decide] at hh; exact (of_decide_eq_true hh).le)
    (fun x y hh => by rw [gtBool_eq_decide] at hh; exact not_lt.mp (of_decide_eq_false hh))
    (0 : Fin (n + 1)) (List.finRange (n + 1))
  exact h.2 j (List.mem_finRange j)

/-! ## `MinNoiseKernelChooser` -/

/-- **`MinNoiseKernelChooser` is sound and complete (some branch).** If some kernel is valid
(`noise < Z_low`) and all noises respect the ceiling `noise ≤ 1` (the verified `varNoiseFn_le_one`),
the chooser returns `some s` with `s` itself valid and of least `noise` among all valid kernels. -/
theorem kernelChooserFn_eq_some {noises Zlows : Fin (n + 1) → ℝ}
    (hbound : ∀ i, noises i ≤ 1) {v : Fin (n + 1)} (hv : noises v < Zlows v) :
    ∃ s, Spec.kernelChooserFn noises Zlows = some s ∧ noises s < Zlows s
      ∧ ∀ j, noises j < Zlows j → noises s ≤ noises j := by
  -- the `np.where`-replaced key (valid ↦ noise, invalid ↦ the `2` sentinel `1 + 1`)
  set key : Fin (n + 1) → ℝ :=
    (fun i => if Spec.ltBool (noises i) (Zlows i) then noises i else (1 : ℝ) + 1) with hkeydef
  have hkv : ∀ i, noises i < Zlows i → key i = noises i := by
    intro i hi
    show (if Spec.ltBool (noises i) (Zlows i) then noises i else (1 : ℝ) + 1) = noises i
    rw [ltBool_eq_decide]; simp [hi]
  have hkinv : ∀ i, ¬ noises i < Zlows i → key i = (1 : ℝ) + 1 := by
    intro i hi
    show (if Spec.ltBool (noises i) (Zlows i) then noises i else (1 : ℝ) + 1) = (1 : ℝ) + 1
    rw [ltBool_eq_decide]; simp [hi]
  set s := Spec.argMinFn key with hs
  have hle : ∀ j, key s ≤ key j := fun j => argMinFn_le key j
  -- the chosen `s` is valid: otherwise `key s = 2 ≤ key v = noises v ≤ 1`, impossible
  have hsvalid : noises s < Zlows s := by
    by_contra hns
    have hchain := hle v
    rw [hkinv s hns, hkv v hv] at hchain
    have := le_trans hchain (hbound v)
    norm_num at this
  refine ⟨s, ?_, hsvalid, ?_⟩
  · show (if Spec.ltBool (noises s) (Zlows s) then some s else none) = some s
    have hbt : Spec.ltBool (noises s) (Zlows s) = true := by rw [ltBool_eq_decide]; simp [hsvalid]
    rw [if_pos hbt]
  · intro j hj
    have hchain := hle j
    rwa [hkv s hsvalid, hkv j hj] at hchain

/-- **`MinNoiseKernelChooser` is sound and complete (none branch).** If no kernel is valid, the chooser
returns `none` — CHD's "no ancestor" verdict. -/
theorem kernelChooserFn_eq_none {noises Zlows : Fin (n + 1) → ℝ}
    (hno : ∀ i, ¬ noises i < Zlows i) : Spec.kernelChooserFn noises Zlows = none := by
  set key : Fin (n + 1) → ℝ :=
    (fun i => if Spec.ltBool (noises i) (Zlows i) then noises i else (1 : ℝ) + 1) with hkeydef
  set s := Spec.argMinFn key with hs
  show (if Spec.ltBool (noises s) (Zlows s) then some s else none) = none
  have hbf : Spec.ltBool (noises s) (Zlows s) = true → False := by
    rw [ltBool_eq_decide]; intro h; exact (hno s) (of_decide_eq_true h)
  rw [if_neg hbf]

/-! ## `MaxIncrementModeChooser` -/

/-- **`MaxIncrementModeChooser` returns the iteration of largest `noise` increment.** -/
theorem modeChooserFn_ge (noises : Fin (n + 1) → ℝ) (j : Fin (n + 1)) :
    Spec.modeIncrementFn noises j ≤ Spec.modeIncrementFn noises (Spec.modeChooserFn noises) := by
  rw [Spec.modeChooserFn]
  exact argMaxFn_le (Spec.modeIncrementFn noises) j

/-! ## The stopping rule -/

/-- **The stopping test `np.all(active_modes == 0)` holds iff every ancestor is pruned.** -/
theorem allPrunedFn_iff {k : Nat} (m : Fin k → ℝ) :
    Spec.allPrunedFn m = true ↔ ∀ i, m i = 0 := by
  rw [Spec.allPrunedFn, List.all_eq_true]
  have key : ∀ i : Fin k,
      ((!Context.gtBool (m i) 0 && !Context.gtBool 0 (m i)) = true) ↔ m i = 0 := by
    intro i
    rw [gtBool_eq_decide, gtBool_eq_decide, ← decide_not, ← decide_not, Bool.and_eq_true,
      decide_eq_true_eq, decide_eq_true_eq]
    constructor
    · rintro ⟨h1, h2⟩; exact le_antisymm (not_lt.mp h1) (not_lt.mp h2)
    · intro h; rw [h]; exact ⟨lt_irrefl 0, lt_irrefl 0⟩
  constructor
  · intro h i; exact (key i).mp (h i (List.mem_finRange i))
  · intro h i _; exact (key i).mpr (h i)

/-! ## CHD `Z_test`: the null-distribution significance thresholds

`Z_test` (`interpolatory.py`) builds the null distribution of the `noise` statistic under random data,
sorts the per-sample noises, and reports the 5th/95th percentiles as `Z_low`/`Z_high`. The numerical
heart — the *value* of each sample's noise — is the **same** `varNoiseFn` whose `[0,1]` bound we already
proved (`varNoiseFn_nonneg`/`_le_one`). So the percentiles inherit that bound, and `Z_low ≤ Z_high`
because a 5th percentile never exceeds a 95th — pure order-statistic monotonicity over the sorted list.

The order statistic `kthSmallestFn` sorts with the `Context` comparator `leBool`; over `ℝ` that is the
real `≤` (`leBool_eq_le`), letting Mathlib's `sortedLE_mergeSort` supply sortedness. -/

/-- Over `ℝ`, the `Context` comparator `leBool x y` is the decidable `x ≤ y`. -/
theorem leBool_eq_decide (x y : ℝ) : Spec.leBool x y = decide (x ≤ y) := by
  rw [Spec.leBool, ltBool_eq_decide, ← decide_not, decide_eq_decide]
  exact not_lt

/-- Over `ℝ`, the `leBool` sort key *is* the decided `(· ≤ ·)`, so `kthSmallestFn` sorts with the
real order (matching Mathlib's `sortedLE_mergeSort`). -/
private theorem leBool_eq_le : (Spec.leBool : ℝ → ℝ → Bool) = (fun x y => decide (x ≤ y)) := by
  funext x y; exact leBool_eq_decide x y

/-- `getD` at an in-range index is the corresponding `getElem` (the `0` fallback is unused). -/
private theorem getD_zero_eq {L : List ℝ} {i : Nat} (h : i < L.length) : L.getD i 0 = L[i] := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h, Option.getD_some]

/-- `kthSmallestFn` over `ℝ` is the `k`-th entry of the list sorted by the *real* order. -/
theorem kthSmallestFn_eq_sorted_getD {N : Nat} (a : Fin N → ℝ) (k : Nat) :
    Spec.kthSmallestFn a k = (((List.finRange N).map a).mergeSort (· ≤ ·)).getD k 0 := by
  rw [Spec.kthSmallestFn, leBool_eq_le]

/-! ### Order-statistic facts -/

/-- **`kthSmallestFn` is one of the family's values** (for an in-range `k`): sorting permutes, so the
selected entry came from `a`. -/
theorem kthSmallestFn_mem {N : Nat} (a : Fin N → ℝ) {k : Nat} (hk : k < N) :
    ∃ i, Spec.kthSmallestFn a k = a i := by
  have hlen : (((List.finRange N).map a).mergeSort (· ≤ ·)).length = N := by
    rw [List.length_mergeSort, List.length_map, List.length_finRange]
  have hk' : k < (((List.finRange N).map a).mergeSort (· ≤ ·)).length := by rw [hlen]; exact hk
  have hmem : Spec.kthSmallestFn a k ∈ ((List.finRange N).map a).mergeSort (· ≤ ·) := by
    rw [kthSmallestFn_eq_sorted_getD, getD_zero_eq hk']
    exact List.getElem_mem hk'
  rw [List.mem_mergeSort, List.mem_map] at hmem
  obtain ⟨i, _, hi⟩ := hmem
  exact ⟨i, hi.symm⟩

/-- **An in-range order statistic is `≥ 0`** when every value is. -/
theorem kthSmallestFn_nonneg {N : Nat} (a : Fin N → ℝ) (hpos : ∀ i, 0 ≤ a i) {k : Nat}
    (hk : k < N) : 0 ≤ Spec.kthSmallestFn a k := by
  obtain ⟨i, hi⟩ := kthSmallestFn_mem a hk; rw [hi]; exact hpos i

/-- **An in-range order statistic is `≤ 1`** when every value is. -/
theorem kthSmallestFn_le_one {N : Nat} (a : Fin N → ℝ) (hle : ∀ i, a i ≤ 1) {k : Nat}
    (hk : k < N) : Spec.kthSmallestFn a k ≤ 1 := by
  obtain ⟨i, hi⟩ := kthSmallestFn_mem a hk; rw [hi]; exact hle i

/-- **Order statistics are monotone in their rank** (`k ≤ k' → kₜₕ ≤ k'ₜₕ`): the underlying list is
sorted ascending, so later indices hold larger values. This is exactly why `Z_low ≤ Z_high`. -/
theorem kthSmallestFn_mono {N : Nat} (a : Fin N → ℝ) {k k' : Nat} (hkk : k ≤ k') (hk' : k' < N) :
    Spec.kthSmallestFn a k ≤ Spec.kthSmallestFn a k' := by
  have hlen : (((List.finRange N).map a).mergeSort (· ≤ ·)).length = N := by
    rw [List.length_mergeSort, List.length_map, List.length_finRange]
  have hkL : k < (((List.finRange N).map a).mergeSort (· ≤ ·)).length := by
    rw [hlen]; exact lt_of_le_of_lt hkk hk'
  have hk'L : k' < (((List.finRange N).map a).mergeSort (· ≤ ·)).length := by rw [hlen]; exact hk'
  rw [kthSmallestFn_eq_sorted_getD, kthSmallestFn_eq_sorted_getD, getD_zero_eq hkL, getD_zero_eq hk'L]
  exact List.sortedLE_mergeSort.getElem_le_getElem_of_le hkk

/-! ### The percentile indices -/

/-- The 5th-percentile index is in range for a nonempty sample. -/
theorem zLowIdx_lt {N : Nat} (hN : 0 < N) : Spec.zLowIdx N < N := by
  rw [Spec.zLowIdx]; exact Nat.div_lt_self hN (by norm_num)

/-- The 95th-percentile index is in range for a nonempty sample. -/
theorem zHighIdx_lt {N : Nat} (hN : 0 < N) : Spec.zHighIdx N < N := by
  rw [Spec.zHighIdx, Nat.div_lt_iff_lt_mul (by norm_num : (0 : Nat) < 20)]
  nlinarith [hN]

/-- The 5th-percentile index never exceeds the 95th. -/
theorem zLowIdx_le_zHighIdx (N : Nat) : Spec.zLowIdx N ≤ Spec.zHighIdx N := by
  rw [Spec.zLowIdx, Spec.zHighIdx]
  exact Nat.div_le_div_right (Nat.le_mul_of_pos_left N (by norm_num))

/-! ### Each null sample's noise inherits the `[0,1]` bound -/

/-- **Every `Z_test` null sample is a genuine fraction** (`0 ≤ noise`): it is `varNoiseFn` of the
projected draw, and `varNoiseFn_nonneg` already bounds that. -/
theorem sampleNoisesFn_nonneg {n N : Nat} {Λ : Fin n → ℝ} (hΛ : ∀ i, 0 ≤ Λ i) {γ : ℝ}
    (hγ : 0 < γ) (V : Fin n → Fin n → ℝ) (samples : Fin N → Fin n → ℝ) (j : Fin N) :
    0 ≤ Spec.sampleNoisesFn Λ V γ samples j := by
  rw [Spec.sampleNoisesFn]; exact varNoiseFn_nonneg hΛ hγ _

/-- **Every `Z_test` null sample is `≤ 1`** (`varNoiseFn_le_one`). -/
theorem sampleNoisesFn_le_one {n N : Nat} {Λ : Fin n → ℝ} (hΛ : ∀ i, 0 ≤ Λ i) {γ : ℝ}
    (hγ : 0 < γ) (V : Fin n → Fin n → ℝ) (samples : Fin N → Fin n → ℝ) (j : Fin N) :
    Spec.sampleNoisesFn Λ V γ samples j ≤ 1 := by
  rw [Spec.sampleNoisesFn]; exact varNoiseFn_le_one hΛ hγ _

/-! ### `Z_low` / `Z_high` are well-posed thresholds -/

/-- **`Z_low` is a genuine fraction in `[0,1]` (lower bound).** -/
theorem zLowFn_nonneg {n N : Nat} {Λ : Fin n → ℝ} (hΛ : ∀ i, 0 ≤ Λ i) {γ : ℝ} (hγ : 0 < γ)
    (V : Fin n → Fin n → ℝ) (samples : Fin N → Fin n → ℝ) (hN : 0 < N) :
    0 ≤ Spec.zLowFn Λ V γ samples := by
  rw [Spec.zLowFn]
  exact kthSmallestFn_nonneg _ (fun j => sampleNoisesFn_nonneg hΛ hγ V samples j) (zLowIdx_lt hN)

/-- **`Z_low ≤ 1`.** -/
theorem zLowFn_le_one {n N : Nat} {Λ : Fin n → ℝ} (hΛ : ∀ i, 0 ≤ Λ i) {γ : ℝ} (hγ : 0 < γ)
    (V : Fin n → Fin n → ℝ) (samples : Fin N → Fin n → ℝ) (hN : 0 < N) :
    Spec.zLowFn Λ V γ samples ≤ 1 := by
  rw [Spec.zLowFn]
  exact kthSmallestFn_le_one _ (fun j => sampleNoisesFn_le_one hΛ hγ V samples j) (zLowIdx_lt hN)

/-- **`Z_high` is a genuine fraction in `[0,1]` (lower bound).** -/
theorem zHighFn_nonneg {n N : Nat} {Λ : Fin n → ℝ} (hΛ : ∀ i, 0 ≤ Λ i) {γ : ℝ} (hγ : 0 < γ)
    (V : Fin n → Fin n → ℝ) (samples : Fin N → Fin n → ℝ) (hN : 0 < N) :
    0 ≤ Spec.zHighFn Λ V γ samples := by
  rw [Spec.zHighFn]
  exact kthSmallestFn_nonneg _ (fun j => sampleNoisesFn_nonneg hΛ hγ V samples j) (zHighIdx_lt hN)

/-- **`Z_high ≤ 1`.** -/
theorem zHighFn_le_one {n N : Nat} {Λ : Fin n → ℝ} (hΛ : ∀ i, 0 ≤ Λ i) {γ : ℝ} (hγ : 0 < γ)
    (V : Fin n → Fin n → ℝ) (samples : Fin N → Fin n → ℝ) (hN : 0 < N) :
    Spec.zHighFn Λ V γ samples ≤ 1 := by
  rw [Spec.zHighFn]
  exact kthSmallestFn_le_one _ (fun j => sampleNoisesFn_le_one hΛ hγ V samples j) (zHighIdx_lt hN)

/-- **`Z_low ≤ Z_high`.** The lower percentile of the null distribution never exceeds the upper one —
the order-statistic monotonicity over the shared sorted noises. The test `Z_low ≤ noise ≤ Z_high` it
implies (the "no anomaly" window of `_GraphDiscoveryMain.py`) is therefore non-degenerate. -/
theorem zLowFn_le_zHighFn {n N : Nat} (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (γ : ℝ)
    (samples : Fin N → Fin n → ℝ) (hN : 0 < N) :
    Spec.zLowFn Λ V γ samples ≤ Spec.zHighFn Λ V γ samples := by
  rw [Spec.zLowFn, Spec.zHighFn]
  exact kthSmallestFn_mono _ (zLowIdx_le_zHighIdx N) (zHighIdx_lt hN)

/-! ### Tying the `Z_test` verdict back to the kernel chooser -/

/-- **A significant edge is never anomalously noisy.** If the observed `noise` clears the lower tail
(`noise < Z_low`), it also sits below the upper tail (`noise < Z_high`), because `Z_low ≤ Z_high`. -/
theorem zSignificant_lt_zHighFn {n N : Nat} (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (γ : ℝ)
    (samples : Fin N → Fin n → ℝ) (hN : 0 < N) {obs : ℝ}
    (hsig : obs < Spec.zLowFn Λ V γ samples) : obs < Spec.zHighFn Λ V γ samples :=
  lt_of_lt_of_le hsig (zLowFn_le_zHighFn Λ V γ samples hN)

/-- **The `Z_test` decision feeds the kernel chooser.** When the observed `noise` of the data clears the
`Z_low` threshold (`zSignificantFn = true`), the single-kernel `MinNoiseKernelChooser` admits the edge —
returns `some 0`. This connects the statistical layer (`Z_test`) to the discovery decision layer
(`kernelChooserFn`, proved sound/complete above); the `noise ≤ 1` ceiling the chooser needs is the
verified `varNoiseFn_le_one`. -/
theorem zTest_admits_edge {n N : Nat} {Λ : Fin n → ℝ} (hΛ : ∀ i, 0 ≤ Λ i) {γ : ℝ} (hγ : 0 < γ)
    (V : Fin n → Fin n → ℝ) (samples : Fin N → Fin n → ℝ) (ga : Fin n → ℝ)
    (hsig : Spec.zSignificantFn (Spec.varNoiseFn Λ γ (Spec.projFn V ga))
      (Spec.zLowFn Λ V γ samples) = true) :
    Spec.kernelChooserFn (fun _ : Fin 1 => Spec.varNoiseFn Λ γ (Spec.projFn V ga))
      (fun _ : Fin 1 => Spec.zLowFn Λ V γ samples) = some 0 := by
  have hlt : Spec.varNoiseFn Λ γ (Spec.projFn V ga) < Spec.zLowFn Λ V γ samples := by
    rw [Spec.zSignificantFn, ltBool_eq_decide] at hsig; exact of_decide_eq_true hsig
  have hb : ∀ i : Fin 1, (fun _ : Fin 1 => Spec.varNoiseFn Λ γ (Spec.projFn V ga)) i ≤ 1 :=
    fun _ => varNoiseFn_le_one hΛ hγ _
  obtain ⟨s, hs, _, _⟩ := kernelChooserFn_eq_some
    (noises := fun _ : Fin 1 => Spec.varNoiseFn Λ γ (Spec.projFn V ga))
    (Zlows := fun _ : Fin 1 => Spec.zLowFn Λ V γ samples) hb (v := 0) hlt
  rw [hs]; exact congrArg some (Fin.fin_one_eq_zero s)

end Spec.Factorization

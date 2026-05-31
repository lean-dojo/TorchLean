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

end Spec.Factorization

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Models.Hopfield.Energy

/-!
# Hopfield cyclic sweep progress (tie-handling)

The update convention `s[u] := (θ[u] ≤ net[u])` sends ties to `+1`, and the convergence argument
has to handle those ties. With symmetric weights and zero diagonal no coordinate update raises the
energy, but at a tie `net = θ` the state can change while the energy stays flat. Such a change
flips the unit from `-1` to `+1` and raises the number of `+1` units (`pluses`) by one, so energy
and `pluses` together form a lexicographic progress measure.

`cycleUpdate_progress` states it for one full cyclic sweep `cycleUpdate`: a sweep that changes the
state either lowers the energy strictly, or keeps it and raises `pluses` strictly.
-/

@[expose] public section


namespace TorchLean.Hopfield

open scoped BigOperators

variable {n : Nat}

/-- One full cyclic sweep, updating the units of `Fin n` one at a time in increasing order. -/
noncomputable def cycleUpdate (p : Params ℝ n) (s : State n) : State n :=
  (List.finRange n).foldl (fun s u => updateAt (α := ℝ) p s u) s

section

variable (p : Params ℝ n)

private theorem energy_foldl_le
    (hsym : (Matrix.of p.W).IsSymm) (hdiag : (Matrix.of p.W).diag = 0) :
    ∀ l : List (Fin n), ∀ s : State n,
      energy (α := ℝ) p (l.foldl (fun s u => updateAt (α := ℝ) p s u) s)
        ≤
      energy (α := ℝ) p s := by
  classical
  intro l
  induction l with
  | nil =>
      intro s
      simp
  | cons u l IH =>
      intro s
      have h1 :
          energy (α := ℝ) p (updateAt (α := ℝ) p s u) ≤ energy (α := ℝ) p s :=
        energy_updateAt_le (n := n) p hsym hdiag s u
      have h2 :
          energy (α := ℝ) p (l.foldl (fun s u => updateAt (α := ℝ) p s u) (updateAt (α := ℝ) p s u))
            ≤
          energy (α := ℝ) p (updateAt (α := ℝ) p s u) :=
        IH (s := updateAt (α := ℝ) p s u)
      simpa [List.foldl] using le_trans h2 h1

/-- For symmetric weights with zero diagonal, a full sweep never increases the energy: it composes
the single-unit bound `energy_updateAt_le`. -/
theorem energy_cycleUpdate_le
    (hsym : (Matrix.of p.W).IsSymm) (hdiag : (Matrix.of p.W).diag = 0)
    (s : State n) :
    energy (α := ℝ) p (cycleUpdate (n := n) p s) ≤ energy (α := ℝ) p s := by
  classical
  simpa [cycleUpdate] using energy_foldl_le (n := n) (p := p) hsym hdiag (List.finRange n) s

private theorem pluses_updateAt_gt_of_ne
    (hsym : (Matrix.of p.W).IsSymm) (hdiag : (Matrix.of p.W).diag = 0)
    (s : State n) (u : Fin n)
    (hE : energy (α := ℝ) p (updateAt (α := ℝ) p s u) = energy (α := ℝ) p s)
    (hchange : updateAt (α := ℝ) p s u ≠ s) :
    pluses (n := n) (updateAt (α := ℝ) p s u) > pluses (n := n) s := by
  classical
  -- If `net ≠ θ` and the state changes, energy strictly decreases, contradicting `hE`.
  by_cases hnet : net (α := ℝ) p s u = p.θ u
  · have hdec : decide (p.θ u ≤ net (α := ℝ) p s u) = true := by simp [hnet]
    -- At a tie the update sets `u` to `true`, so a change means `s u = false`.
    have hsuf : s u = false := by
      simpa [hdec] using apply_eq_not_of_updateAt_ne p s u hchange
    have hps :
        pluses (n := n) (updateAt (α := ℝ) p s u) = pluses (n := n) s + 1 :=
      pluses_updateAt_eq_succ (p := p) (s := s) (u := u) hsuf hdec
    exact lt_of_lt_of_eq (Nat.lt_succ_self _) hps.symm
  · have hlt :=
      energy_updateAt_lt_of_net_ne (n := n) p hsym hdiag s u hchange (by simpa [eq_comm] using
        hnet)
    exact False.elim (hlt.ne hE)

private theorem pluses_updateAt_ge
    (hsym : (Matrix.of p.W).IsSymm) (hdiag : (Matrix.of p.W).diag = 0)
    (s : State n) (u : Fin n)
    (hE : energy (α := ℝ) p (updateAt (α := ℝ) p s u) = energy (α := ℝ) p s) :
    pluses (n := n) (updateAt (α := ℝ) p s u) ≥ pluses (n := n) s := by
  classical
  by_cases hchange : updateAt (α := ℝ) p s u = s
  · exact ge_of_eq (congrArg (pluses (n := n)) hchange)
  · exact le_of_lt (pluses_updateAt_gt_of_ne (n := n) (p := p) hsym hdiag s u hE
    hchange)

private theorem pluses_foldl_ge
    (hsym : (Matrix.of p.W).IsSymm) (hdiag : (Matrix.of p.W).diag = 0) :
    ∀ l : List (Fin n), ∀ s : State n,
      (energy (α := ℝ) p (l.foldl (fun s u => updateAt (α := ℝ) p s u) s) = energy (α := ℝ) p s) →
      pluses (n := n) (l.foldl (fun s u => updateAt (α := ℝ) p s u) s) ≥ pluses (n := n) s := by
  classical
  intro l
  induction l with
  | nil =>
      intro s hE
      simp
  | cons u l IH =>
      intro s hE
      set s1 : State n := updateAt (α := ℝ) p s u with hs1
      set sf : State n := l.foldl (fun s u => updateAt (α := ℝ) p s u) s1 with hsf
      have hle1 : energy (α := ℝ) p s1 ≤ energy (α := ℝ) p s :=
        by simpa [hs1] using energy_updateAt_le (n := n) p hsym hdiag s u
      have hleTail : energy (α := ℝ) p sf ≤ energy (α := ℝ) p s1 := by
        simpa [hsf] using energy_foldl_le (n := n) (p := p) hsym hdiag l s1
      have hEsf : energy (α := ℝ) p sf = energy (α := ℝ) p s := by
        simpa [hsf, hs1, List.foldl] using hE
      have hge1 : energy (α := ℝ) p s ≤ energy (α := ℝ) p s1 := by
        simpa [hEsf] using hleTail
      have hE1 : energy (α := ℝ) p s1 = energy (α := ℝ) p s :=
        le_antisymm hle1 hge1
      have hErest : energy (α := ℝ) p sf = energy (α := ℝ) p s1 := by
        simp [hEsf, hE1]
      have hGe1 : pluses (n := n) s1 ≥ pluses (n := n) s :=
        by
          simpa [hs1] using pluses_updateAt_ge (n := n) (p := p) hsym hdiag s u hE1
      have hGeRest : pluses (n := n) sf ≥ pluses (n := n) s1 :=
        IH (s := s1) hErest
      exact le_trans hGe1 hGeRest

private theorem pluses_foldl_gt_of_ne
    (hsym : (Matrix.of p.W).IsSymm) (hdiag : (Matrix.of p.W).diag = 0) :
    ∀ l : List (Fin n), ∀ s : State n,
      (energy (α := ℝ) p (l.foldl (fun s u => updateAt (α := ℝ) p s u) s) = energy (α := ℝ) p s) →
      (l.foldl (fun s u => updateAt (α := ℝ) p s u) s ≠ s) →
      pluses (n := n) (l.foldl (fun s u => updateAt (α := ℝ) p s u) s) > pluses (n := n) s := by
  classical
  intro l
  induction l with
  | nil =>
      intro s hE hne
      cases hne rfl
  | cons u l IH =>
      intro s hE hne
      set s1 : State n := updateAt (α := ℝ) p s u with hs1
      set sf : State n := l.foldl (fun s u => updateAt (α := ℝ) p s u) s1 with hsf
      -- A `foldl` over `u :: l` first steps to `s1`, so `hs1` and `hsf` rewrite it to `sf`.
      have hfold : (u :: l).foldl (fun s u => updateAt (α := ℝ) p s u) s = sf :=
        (congrArg (fun t => l.foldl (fun s u => updateAt (α := ℝ) p s u) t) hs1.symm).trans
          hsf.symm
      have hne_sf : sf ≠ s := by
        intro hEq
        apply hne
        exact hfold.trans hEq
      have hle1 : energy (α := ℝ) p s1 ≤ energy (α := ℝ) p s :=
        by simpa [hs1] using energy_updateAt_le (n := n) p hsym hdiag s u
      have hleTail : energy (α := ℝ) p sf ≤ energy (α := ℝ) p s1 := by
        simpa [hsf] using energy_foldl_le (n := n) (p := p) hsym hdiag l s1
      have hEsf : energy (α := ℝ) p sf = energy (α := ℝ) p s := by
        exact (congrArg (energy (α := ℝ) p) hfold.symm).trans hE
      have hge1 : energy (α := ℝ) p s ≤ energy (α := ℝ) p s1 := by
        simpa [hEsf] using hleTail
      have hE1 : energy (α := ℝ) p s1 = energy (α := ℝ) p s :=
        le_antisymm hle1 hge1
      have hErest : energy (α := ℝ) p sf = energy (α := ℝ) p s1 := by
        simp [hEsf, hE1]
      by_cases hHead : s1 = s
      · have hTailNe : l.foldl (fun s u => updateAt (α := ℝ) p s u) s1 ≠ s1 := by
          simpa only [hsf, hHead] using hne_sf
        have hTailLt := IH s1 hErest hTailNe
        simpa only [hfold, hsf, hHead] using hTailLt
      · have hHeadLt : pluses (n := n) s1 > pluses (n := n) s :=
          by
            have huNe : updateAt (α := ℝ) p s u ≠ s := by
              -- `updateAt ... = s1`, so `updateAt ... ≠ s` follows from `hHead`.
              simpa [hs1] using hHead
            simpa [hs1] using
              pluses_updateAt_gt_of_ne (n := n) (p := p) hsym hdiag s u hE1 huNe
        have hTailGe : pluses (n := n) sf ≥ pluses (n := n) s1 :=
          pluses_foldl_ge (p := p) hsym hdiag l s1 hErest
        exact lt_of_lt_of_le hHeadLt hTailGe

/-- With symmetric weights and zero diagonal, every sweep that changes the state makes progress:
either the energy strictly drops, or it stays equal and the number of active units strictly rises.

`TorchLean.NN.Models.Hopfield.Convergence` turns this lexicographic measure into termination.
Energy alone is not enough, because a unit sitting exactly at its threshold can flip without
changing the energy. -/
theorem cycleUpdate_progress
    (hsym : (Matrix.of p.W).IsSymm) (hdiag : (Matrix.of p.W).diag = 0)
    (s : State n) (hchange : cycleUpdate (n := n) p s ≠ s) :
    energy (α := ℝ) p (cycleUpdate (n := n) p s) < energy (α := ℝ) p s
      ∨
    (energy (α := ℝ) p (cycleUpdate (n := n) p s) = energy (α := ℝ) p s
      ∧ pluses (n := n) (cycleUpdate (n := n) p s) > pluses (n := n) s) := by
  classical
  have hle := energy_cycleUpdate_le (n := n) (p := p) hsym hdiag s
  by_cases hEq : energy (α := ℝ) p (cycleUpdate (n := n) p s) = energy (α := ℝ) p s
  · have hP : pluses (n := n) (cycleUpdate (n := n) p s) > pluses (n := n) s := by
      simpa [cycleUpdate] using
        pluses_foldl_gt_of_ne (p := p) hsym hdiag (List.finRange n) s hEq hchange
    exact Or.inr ⟨hEq, hP⟩
  · exact Or.inl (lt_of_le_of_ne hle hEq)

end

end TorchLean.Hopfield

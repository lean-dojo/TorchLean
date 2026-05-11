/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.Proofs.Hopfield.Energy

/-!
# Hopfield cyclic sweep progress (tie-handling)

This file proves the key “tie-handling” lemma needed for paper-style Hopfield global-dynamics
claims under the update convention `s[u] := (θ[u] ≤ net[u])` (“ties go to +1”).

Energy is non-increasing under each coordinate update, but in the tie case `net = θ` the energy
can stay constant while the state changes. We show that in this tie case, the number of `+1`s
(`pluses`) strictly increases. This yields a lexicographic progress measure.

We package the statement for a **full cyclic sweep** over coordinates:

* Either energy strictly decreases, or energy is unchanged and `pluses` strictly increases.
-/

@[expose] public section


namespace NN.MLTheory.Proofs.Hopfield

open scoped BigOperators
open Spec

open Spec.Hopfield

variable {n : Nat}

noncomputable def cycleUpdate (p : Params ℝ n) (s : State n) : State n :=
  (List.finRange n).foldl (fun s u => updateAt (α := ℝ) p s u) s

section

variable (p : Params ℝ n)

private lemma energy_foldl_le
    (hsym : SymmetricW (n := n) p) (hdiag : DiagonalZero (n := n) p) :
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

theorem energy_cycleUpdate_le
    (hsym : SymmetricW (n := n) p) (hdiag : DiagonalZero (n := n) p)
    (s : State n) :
    energy (α := ℝ) p (cycleUpdate (n := n) p s) ≤ energy (α := ℝ) p s := by
  classical
  simpa [cycleUpdate] using energy_foldl_le (n := n) (p := p) hsym hdiag (List.finRange n) s

private lemma pluses_updateAt_gt_of_energy_eq_of_ne
    (hsym : SymmetricW (n := n) p) (hdiag : DiagonalZero (n := n) p)
    (s : State n) (u : Fin n)
    (hE : energy (α := ℝ) p (updateAt (α := ℝ) p s u) = energy (α := ℝ) p s)
    (hchange : updateAt (α := ℝ) p s u ≠ s) :
    pluses (n := n) (updateAt (α := ℝ) p s u) > pluses (n := n) s := by
  classical
  -- If `net ≠ θ` and the state changes, energy strictly decreases, contradicting `hE`.
  by_cases hnet : net (α := ℝ) p s u = p.θ u
  · have hdec : decide (p.θ u ≤ net (α := ℝ) p s u) = true := by simp [hnet]
    -- The update sets `u := true`; changing implies the old bit was `false`.
    have hsuf : s u = false := by
      have hsu : updateAt (α := ℝ) p s u u ≠ s u := by
        intro hEq
        apply hchange
        funext i
        by_cases hi : i = u
        · subst hi; simpa using hEq
        · simp [updateAt_apply_ne (p := p) (s := s) (u := u) (v := i) hi]
      have hup : updateAt (α := ℝ) p s u u = true := by
        have hu'' :
            updateAt (α := ℝ) p s u u = decide (p.θ u ≤ net (α := ℝ) p s u) := by
          simp [updateAt]
        exact hu''.trans hdec
      cases hsu0 : s u <;> try rfl
      exfalso
      exact hsu (by simpa [hsu0] using hup)
    have hps :
        pluses (n := n) (updateAt (α := ℝ) p s u) = pluses (n := n) s + 1 :=
      pluses_updateAt_eq_succ_of_set_true (p := p) (s := s) (u := u) hsuf hdec
    -- Rewrite the goal using `hps`.
    have : pluses (n := n) (updateAt (α := ℝ) p s u) > pluses (n := n) s := by
      rw [hps]
      simp
    exact this
  · have hlt :=
      energy_updateAt_lt_of_change_of_ne (n := n) p hsym hdiag s u hchange (by simpa [eq_comm] using
        hnet)
    exact False.elim (hlt.ne hE)

private lemma pluses_updateAt_ge_of_energy_eq
    (hsym : SymmetricW (n := n) p) (hdiag : DiagonalZero (n := n) p)
    (s : State n) (u : Fin n)
    (hE : energy (α := ℝ) p (updateAt (α := ℝ) p s u) = energy (α := ℝ) p s) :
    pluses (n := n) (updateAt (α := ℝ) p s u) ≥ pluses (n := n) s := by
  classical
  by_cases hchange : updateAt (α := ℝ) p s u = s
  · -- Avoid `simp` here: `updateAt` is a simp lemma that unfolds to `Function.update`,
    -- which would prevent rewriting by `hchange`.
    exact ge_of_eq (by
      simpa using congrArg (pluses (n := n)) hchange)
  · exact le_of_lt (pluses_updateAt_gt_of_energy_eq_of_ne (n := n) (p := p) hsym hdiag s u hE
    hchange)

private lemma pluses_foldl_ge_of_energy_eq
    (hsym : SymmetricW (n := n) p) (hdiag : DiagonalZero (n := n) p) :
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
          simpa [hs1] using pluses_updateAt_ge_of_energy_eq (n := n) (p := p) hsym hdiag s u hE1
      have hGeRest : pluses (n := n) sf ≥ pluses (n := n) s1 :=
        IH (s := s1) hErest
      exact le_trans hGe1 hGeRest

private lemma pluses_foldl_gt_of_energy_eq_of_ne
    (hsym : SymmetricW (n := n) p) (hdiag : DiagonalZero (n := n) p) :
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
      have hfold : (u :: l).foldl (fun s u => updateAt (α := ℝ) p s u) s = sf := by
        -- `foldl` on a `cons` and rewrite the initial accumulator using `hs1`/`hsf`.
        have h0 :
            (u :: l).foldl (fun s u => updateAt (α := ℝ) p s u) s =
              l.foldl (fun s u => updateAt (α := ℝ) p s u) (updateAt (α := ℝ) p s u) := by
          rfl
        have h1 :
            l.foldl (fun s u => updateAt (α := ℝ) p s u) (updateAt (α := ℝ) p s u) =
              l.foldl (fun s u => updateAt (α := ℝ) p s u) s1 := by
          exact congrArg (fun t => l.foldl (fun s u => updateAt (α := ℝ) p s u) t) hs1.symm
        exact h0.trans (h1.trans hsf.symm)
      have hne_sf : sf ≠ s := by
        intro hEq
        apply hne
        simpa [hfold] using hEq
      have hle1 : energy (α := ℝ) p s1 ≤ energy (α := ℝ) p s :=
        by simpa [hs1] using energy_updateAt_le (n := n) p hsym hdiag s u
      have hleTail : energy (α := ℝ) p sf ≤ energy (α := ℝ) p s1 := by
        simpa [hsf] using energy_foldl_le (n := n) (p := p) hsym hdiag l s1
      have hEsf : energy (α := ℝ) p sf = energy (α := ℝ) p s := by
        simpa [hfold] using hE
      have hge1 : energy (α := ℝ) p s ≤ energy (α := ℝ) p s1 := by
        simpa [hEsf] using hleTail
      have hE1 : energy (α := ℝ) p s1 = energy (α := ℝ) p s :=
        le_antisymm hle1 hge1
      have hErest : energy (α := ℝ) p sf = energy (α := ℝ) p s1 := by
        simp [hEsf, hE1]
      by_cases hHead : s1 = s
      · have hTailNe : l.foldl (fun s u => updateAt (α := ℝ) p s u) s ≠ s := by
          intro hEq
          apply hne_sf
          -- `sf = l.foldl _ s1`, and `s1 = s`, so `sf = l.foldl _ s = s`.
          calc
            sf = l.foldl (fun s u => updateAt (α := ℝ) p s u) s1 := hsf
            _ = l.foldl (fun s u => updateAt (α := ℝ) p s u) s := by simp [hHead]
            _ = s := hEq
        have hE' :
            energy (α := ℝ) p (l.foldl (fun s u => updateAt (α := ℝ) p s u) s) = energy (α := ℝ) p s
              := by
          -- Rewrite `hEsf : energy sf = energy s` along `sf = foldl _ s`.
          have : energy (α := ℝ) p sf = energy (α := ℝ) p s := hEsf
          -- Replace `sf` by the tail fold using `hsf` and `hHead`.
          calc
            energy (α := ℝ) p (l.foldl (fun s u => updateAt (α := ℝ) p s u) s)
                = energy (α := ℝ) p (l.foldl (fun s u => updateAt (α := ℝ) p s u) s1) := by
                  simp [hHead]
            _ = energy (α := ℝ) p sf := by simp [hsf]
            _ = energy (α := ℝ) p s := this
        have hTailLt :
            pluses (n := n) (l.foldl (fun s u => updateAt (α := ℝ) p s u) s) >
              pluses (n := n) s :=
          IH s hE' hTailNe
        -- Head update is a no-op, so the whole fold is the tail fold.
        have huNoop : updateAt (α := ℝ) p s u = s := by
          -- `updateAt ... = s1 = s`.
          exact hs1.symm.trans hHead
        have hfull :
            (u :: l).foldl (fun s u => updateAt (α := ℝ) p s u) s =
              l.foldl (fun s u => updateAt (α := ℝ) p s u) s := by
          have h0 :
              (u :: l).foldl (fun s u => updateAt (α := ℝ) p s u) s =
                l.foldl (fun s u => updateAt (α := ℝ) p s u) (updateAt (α := ℝ) p s u) := by
            rfl
          have h1 :
              l.foldl (fun s u => updateAt (α := ℝ) p s u) (updateAt (α := ℝ) p s u) =
                l.foldl (fun s u => updateAt (α := ℝ) p s u) s := by
            exact congrArg (fun t => l.foldl (fun s u => updateAt (α := ℝ) p s u) t) huNoop
          exact h0.trans h1
        -- Avoid `simp` here: `updateAt` has simp lemmas that unfold to `Function.update`.
        have hpl :
            pluses (n := n) (l.foldl (fun s u => updateAt (α := ℝ) p s u) s) =
              pluses (n := n) ((u :: l).foldl (fun s u => updateAt (α := ℝ) p s u) s) :=
          congrArg (pluses (n := n)) hfull.symm
        exact lt_of_lt_of_eq hTailLt hpl
      · have hHeadLt : pluses (n := n) s1 > pluses (n := n) s :=
          by
            have huNe : updateAt (α := ℝ) p s u ≠ s := by
              -- `updateAt ... = s1`, so `updateAt ... ≠ s` follows from `hHead`.
              simpa [hs1] using hHead
            simpa [hs1] using
              pluses_updateAt_gt_of_energy_eq_of_ne (n := n) (p := p) hsym hdiag s u hE1 huNe
        have hTailGe : pluses (n := n) sf ≥ pluses (n := n) s1 :=
          pluses_foldl_ge_of_energy_eq (p := p) hsym hdiag l s1 hErest
        exact lt_of_lt_of_le hHeadLt hTailGe

theorem cycleUpdate_progress
    (hsym : SymmetricW (n := n) p) (hdiag : DiagonalZero (n := n) p)
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
        pluses_foldl_gt_of_energy_eq_of_ne (p := p) hsym hdiag (List.finRange n) s hEq hchange
    exact Or.inr ⟨hEq, hP⟩
  · exact Or.inl (lt_of_le_of_ne hle hEq)

end

end NN.MLTheory.Proofs.Hopfield

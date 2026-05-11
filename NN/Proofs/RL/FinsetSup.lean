/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.Real.Basic
public import Mathlib.Data.Finset.Lattice.Fold

/-!
# Finset Suprema Helpers

This file collects small lemmas about `Finset.sup'` that are useful in RL proofs.

We keep these helpers in their own module so RL proof files (finite MDPs, stochastic MDPs, etc.)
do not each re-prove the same `sup'`-algebra facts locally.

References:
- Puterman, *Markov Decision Processes* (1994), discounted dynamic programming chapter
  (the “sup is Lipschitz” step in Bellman optimality contraction proofs).
- Bertsekas, *Dynamic Programming and Optimal Control*, Vol. 1 (contraction/monotonicity arguments).
- mathlib docs for the underlying finset order-theory API:
  https://leanprover-community.github.io/mathlib4_docs/Mathlib/Data/Finset/Lattice/Fold.html
-/

@[expose] public section

namespace Proofs
namespace RL

/-- If `f i ≤ g i + c` for all `i ∈ s`, then `sup f ≤ sup g + c` over the same nonempty finset. -/
theorem sup'_le_add_const
    {ι : Type}
    (s : Finset ι) (hs : s.Nonempty)
    (f g : ι → ℝ) (c : ℝ)
    (hfg : ∀ i ∈ s, f i ≤ g i + c) :
    s.sup' hs f ≤ s.sup' hs g + c := by
  refine Finset.sup'_le hs f ?_
  intro i hi
  have hsup : g i + c ≤ s.sup' hs g + c := by
    -- `Finset.le_sup'` gives `g i ≤ sup g`; we add `c` to both sides.
    simpa [add_comm, add_left_comm, add_assoc] using add_le_add_left (Finset.le_sup' g hi) c
  exact (hfg i hi).trans hsup

end RL
end Proofs

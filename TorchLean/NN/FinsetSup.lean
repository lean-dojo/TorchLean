/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Basic.Real.Basic
public import Mathlib.Data.Finset.Lattice.Fold
import Mathlib.Algebra.Order.Group.Finset

/-!
# Finset Suprema Helpers

Real functions within `c` of each other on a nonempty finset have suprema within `c`. The RL proofs
use this step to show that Bellman optimality is Lipschitz in the value function.

References:
- Puterman, *Markov Decision Processes* (1994), discounted dynamic programming chapter
  (the “sup is Lipschitz” step in Bellman optimality contraction proofs).
- Bertsekas, *Dynamic Programming and Optimal Control*, Vol. 1 (contraction/monotonicity arguments).
- mathlib docs for the underlying finset order-theory API:
  https://leanprover-community.github.io/mathlib4_docs/Mathlib/Data/Finset/Lattice/Fold.html
-/

@[expose] public section

namespace TorchLean
namespace RL

/-- If `f` and `g` differ by at most `c` everywhere on `s`, so do their suprema. -/
theorem abs_sup'_sub_sup'_le
    {ι : Type}
    (s : Finset ι) (hs : s.Nonempty)
    (f g : ι → ℝ) (c : ℝ)
    (hfg : ∀ i ∈ s, |f i - g i| ≤ c) :
    |s.sup' hs f - s.sup' hs g| ≤ c := by
  refine abs_sub_le_iff.mpr ⟨sub_le_iff_le_add'.mpr ?_, sub_le_iff_le_add'.mpr ?_⟩
  · refine (Finset.sup'_mono_fun fun i hi => ?_).trans_eq (Finset.sup'_add s g c hs).symm
    exact sub_le_iff_le_add'.mp (abs_sub_le_iff.mp (hfg i hi)).1
  · refine (Finset.sup'_mono_fun fun i hi => ?_).trans_eq (Finset.sup'_add s f c hs).symm
    exact sub_le_iff_le_add'.mp (abs_sub_le_iff.mp (hfg i hi)).2

end RL
end TorchLean

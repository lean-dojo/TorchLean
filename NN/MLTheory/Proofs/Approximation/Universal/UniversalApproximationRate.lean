/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.Proofs.Approximation.Universal.UniversalApproximation

/-!
# Universal approximation (1D, explicit rate)

This file strengthens `relu_universal_approximation_Icc` by choosing an explicit hidden width in
terms of the Lipschitz constant $L$, the interval length $b-a$, and the target accuracy
$\varepsilon$.

The bound is the standard $O(1/\mathrm{hidDim})$ rate from piecewise-linear interpolation:
we pick

$$
\mathrm{hidDim}=\left\lceil\frac{2L(b-a)}{\varepsilon}\right\rceil+1,
$$

which guarantees uniform approximation error below $\varepsilon$ on $[a,b]$.

Mathematically, this is the quantitative sibling of the constructive one-dimensional ReLU
universal approximation proof in `UniversalApproximation`: sample a Lipschitz function on a
uniform grid, interpolate linearly by hinge functions, and choose the grid fine enough that the
Lipschitz modulus controls the interpolation error.  The style is classical approximation theory
(Pinkus) and agrees with the first-order rate used in modern ReLU-network approximation analyses
such as Yarotsky's quantitative bounds.
-/

@[expose] public section

namespace NN.MLTheory.Proofs.UniversalApproximation

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor
open Examples

noncomputable section

/-- Explicit hidden width for the 1D Lipschitz ReLU approximation construction. -/
def reluApproximationWidth (L a b ε : ℝ) : ℕ :=
  Nat.ceil (2 * L * (b - a) / ε) + 1

/-- The explicit ReLU approximation width is always positive. -/
theorem relu_approximation_width_pos (L a b ε : ℝ) : 0 < reluApproximationWidth L a b ε := by
  simp [reluApproximationWidth]

/--
The chosen width makes the mesh-size error term smaller than the target accuracy.

This is the arithmetic heart of the explicit-rate theorem: the ceiling construction ensures
$N>2L(b-a)/\varepsilon$, hence $2L(b-a)/N<\varepsilon$.
-/
theorem two_mul_mul_sub_div_relu_approximation_width_lt {L a b ε : ℝ} (hε : 0 < ε) :
    (2 * L * (b - a)) / (reluApproximationWidth L a b ε : ℝ) < ε := by
  classical
  let N : ℕ := reluApproximationWidth L a b ε
  have hNpos_nat : 0 < N := relu_approximation_width_pos L a b ε
  have hNpos : 0 < (N : ℝ) := by exact_mod_cast hNpos_nat
  have hr_lt : (2 * L * (b - a) / ε : ℝ) < (N : ℝ) := by
    have hr_le :
        (2 * L * (b - a) / ε : ℝ) ≤ (Nat.ceil (2 * L * (b - a) / ε) : ℝ) :=
      Nat.le_ceil _
    have : (2 * L * (b - a) / ε : ℝ) < (Nat.ceil (2 * L * (b - a) / ε) : ℝ) + 1 := by
      linarith
    simpa [N, reluApproximationWidth, Nat.cast_add, Nat.cast_one, add_assoc] using this
  have hmul : ε * (2 * L * (b - a) / ε) < ε * (N : ℝ) := mul_lt_mul_of_pos_left hr_lt hε
  have hεne : (ε : ℝ) ≠ 0 := ne_of_gt hε
  have hleft : ε * (2 * L * (b - a) / ε) = 2 * L * (b - a) := by
    calc
      ε * (2 * L * (b - a) / ε) = ε * (2 * L * (b - a)) / ε := by
        simp [mul_div_assoc']
      _ = 2 * L * (b - a) := by
        simpa using (mul_div_cancel_left₀ (2 * L * (b - a)) hεne)
  have hnum : 2 * L * (b - a) < ε * (N : ℝ) := by
    simpa [hleft] using hmul
  exact (div_lt_iff₀ hNpos).2 (by simpa [mul_comm, mul_assoc] using hnum)

/--
Universal approximation (1D, hinge form) with an explicit width choice.

This is a quantitative variant of `relu_universal_approximation_Icc_hinge` where the hidden width
is fixed to `reluApproximationWidth L a b ε`.
-/
theorem relu_universal_approximation_Icc_hinge_rate {f : ℝ → ℝ} {a b L : ℝ}
    (h_ab : a < b) (hL : 0 < L)
    (h_lip : ∀ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, |f x - f y| ≤ L * |x - y|) :
    ∀ ε > 0,
      ∃ (t : Fin (reluApproximationWidth L a b ε) → ℝ)
        (c : Fin (reluApproximationWidth L a b ε) → ℝ),
        ∀ x ∈ Set.Icc a b,
          |f x - hingeFun (reluApproximationWidth L a b ε) t c (f a) x| < ε := by
  intro ε hε
  apply relu_hinge_approximation_Icc_of_mesh h_ab hL h_lip
    (relu_approximation_width_pos L a b ε) hε
  simpa [mul_div_assoc', mul_assoc] using
    (two_mul_mul_sub_div_relu_approximation_width_lt (L := L) (a := a) (b := b) hε)

/--
Universal approximation (1D, explicit rate) for a 2-layer ReLU MLP.

This is the MLP-packaged version of `relu_universal_approximation_Icc_hinge_rate`.
-/
theorem relu_universal_approximation_Icc_rate {f : ℝ → ℝ} {a b L : ℝ}
    (h_ab : a < b) (hL : 0 < L)
    (h_lip : ∀ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, |f x - f y| ≤ L * |x - y|) :
    ∀ ε > 0,
      ∃ (l1 : LinearSpec ℝ 1 (reluApproximationWidth L a b ε))
        (l2 : LinearSpec ℝ (reluApproximationWidth L a b ε) 1),
        ∀ x ∈ Set.Icc a b,
          |f x - mlpEvalScalar (reluApproximationWidth L a b ε) l1 l2 x| < ε := by
  intro ε hε
  classical
  rcases
      relu_universal_approximation_Icc_hinge_rate (f := f) (a := a) (b := b) (L := L)
        h_ab hL h_lip ε hε with
    ⟨t, c, happx⟩
  refine ⟨hingeLayer1 (reluApproximationWidth L a b ε) t,
    hingeLayer2 (reluApproximationWidth L a b ε) c (f a),
    ?_⟩
  intro x hx
  have hnet :
      mlpEvalScalar (reluApproximationWidth L a b ε)
          (hingeLayer1 (reluApproximationWidth L a b ε) t)
          (hingeLayer2 (reluApproximationWidth L a b ε) c (f a)) x =
        hingeFun (reluApproximationWidth L a b ε) t c (f a) x := by
    simpa using (mlp_eval_scalar_hinge (reluApproximationWidth L a b ε) t c (f a) x)
  simpa [hnet] using happx x hx

end

end NN.MLTheory.Proofs.UniversalApproximation

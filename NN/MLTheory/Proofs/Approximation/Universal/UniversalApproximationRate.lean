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
    (relu_approximation_width_pos L a b ε)
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

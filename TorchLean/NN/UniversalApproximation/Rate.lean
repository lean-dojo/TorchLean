/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.UniversalApproximation

/-!
# Universal approximation (1D, explicit rate)

`relu_Icc` asserts that some hidden width works. The theorems here fix the
width in terms of the Lipschitz constant $L$, the interval length $b-a$ and the target accuracy
$\varepsilon$:

$$
\mathrm{hidDim}=\left\lceil\frac{2L(b-a)}{\varepsilon}\right\rceil+1.
$$

The bound is the standard $O(1/\mathrm{hidDim})$ rate of piecewise-linear interpolation. With
$\mathrm{hidDim}$ uniform cells the error is at most $2L(b-a)/\mathrm{hidDim}$, and this width
pushes it below $\varepsilon$ uniformly on $[a,b]$.

The construction is the one-dimensional proof of
`TorchLean.NN.UniversalApproximation`: sample the Lipschitz
function on a uniform grid, interpolate linearly by hinge functions, and choose the grid fine enough
that the Lipschitz modulus controls the interpolation error. The style is classical approximation
theory (Pinkus) and agrees with the first-order rate used in modern ReLU-network approximation
analyses such as Yarotsky's quantitative bounds.
-/

@[expose] public section

namespace TorchLean.UniversalApproximation

open _root_.TorchLean.Tensor

noncomputable section

/--
Universal approximation (1D, hinge form) at a fixed width: under the hypotheses of
`relu_Icc_hinge`, a hinge network with `width L a b ε`
hidden units and bias $f(a)$ is within $\varepsilon$ of $f$ on $[a,b]$.
-/
theorem relu_Icc_hinge_rate {f : ℝ → ℝ} {a b L : ℝ}
    (h_ab : a < b) (hL : 0 < L)
    (h_lip : ∀ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, |f x - f y| ≤ L * |x - y|) :
    ∀ ε > 0,
      ∃ (t : Fin (width L a b ε) → ℝ)
        (c : Fin (width L a b ε) → ℝ),
        ∀ x ∈ Set.Icc a b,
          |f x - hingeFun t c (f a) x| < ε := by
  intro ε hε
  apply relu_Icc_hinge_of_mesh h_ab hL h_lip
    (width_pos L a b ε)
  simpa [mul_div_assoc', mul_assoc] using
    (div_width_lt (L := L) (a := a) (b := b) hε)

/--
Universal approximation (1D, rate form) for a 2-layer ReLU MLP: the hidden width
`width L a b ε` reaches accuracy $\varepsilon$ on $[a,b]$.

The two layers are `hingeLayer1` and `hingeLayer2` carrying the network from
`relu_Icc_hinge_rate`.
-/
theorem relu_Icc_rate {f : ℝ → ℝ} {a b L : ℝ}
    (h_ab : a < b) (hL : 0 < L)
    (h_lip : ∀ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, |f x - f y| ≤ L * |x - y|) :
    ∀ ε > 0,
      ∃ (l1 : Linear ℝ 1 (width L a b ε))
        (l2 : Linear ℝ (width L a b ε) 1),
        ∀ x ∈ Set.Icc a b,
          |f x - mlpEvalScalar (width L a b ε) l1 l2 x| < ε := by
  intro ε hε
  obtain ⟨t, c, happx⟩ :=
    relu_Icc_hinge_rate (f := f) (a := a) (b := b) (L := L)
      h_ab hL h_lip ε hε
  refine ⟨hingeLayer1 (width L a b ε) t,
    hingeLayer2 (width L a b ε) c (f a), fun x hx => ?_⟩
  rw [mlpEvalScalar_hinge]
  exact happx x hx

end

end TorchLean.UniversalApproximation

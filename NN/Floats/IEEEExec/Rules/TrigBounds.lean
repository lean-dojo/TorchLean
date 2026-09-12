/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.Complex.Trigonometric

/-!
## Simple Taylor remainder bounds for `Real.sin` / `Real.cos`

`IEEE32Exec.Trig.sinCosTaylorSmall` uses a reduced-input Taylor kernel (degree 13/12). The theorems
below are coarse analytic facts about separate degree-three/two baselines; they are not an accuracy
contract for the executable kernel. For many proofs these baseline bounds are still useful on
$|x|\le 1$.

Mathlib provides clean, reusable bounds for the first nontrivial truncations:

- `Real.sin_bound`: $|\sin x-(x-x^3/6)|\le |x|^5/100$ for $|x|\le 1$;
- `Real.cos_bound`: $|\cos x-(1-x^2/2)|\le |x|^4(5/96)$ for $|x|\le 1$.

These are coarse but robust and are often sufficient as a local ingredient in larger numerical
proofs. Tighter bounds for higher-degree truncations can be added later if needed.
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754
namespace IEEE32Exec

noncomputable section

/-- Quadratic Taylor approximation $1-x^2/2$ used as a baseline for `cos`. -/
def cosTaylor2 (x : ℝ) : ℝ :=
  1 - x ^ 2 / 2

/-- Cubic Taylor approximation $x-x^3/6$ used as a baseline for `sin`. -/
def sinTaylor3 (x : ℝ) : ℝ :=
  x - x ^ 3 / 6

/-- Degree-three Taylor error for sine on `|x| ≤ 1`, as `|x|⁵ / 100`.

This is Mathlib `Real.sin_bound` restated in the shape the enclosure proofs consume, so the interval
rules never have to unfold `sinTaylor3`. -/
theorem abs_sin_sub_sinTaylor3_le (x : ℝ) (hx : |x| ≤ 1) :
    |Real.sin x - sinTaylor3 x| ≤ |x| ^ 5 / 100 := by
  simpa [sinTaylor3] using (Real.sin_bound (x := x) hx)

/-- Degree-two Taylor error for cosine on `|x| ≤ 1`, from Mathlib `Real.cos_bound`. -/
theorem abs_cos_sub_cosTaylor2_le (x : ℝ) (hx : |x| ≤ 1) :
    |Real.cos x - cosTaylor2 x| ≤ |x| ^ 4 * (5 / 96) := by
  simpa [cosTaylor2] using (Real.cos_bound (x := x) hx)

end

end IEEE32Exec
end TorchLean.Floats.IEEE754

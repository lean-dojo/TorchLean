/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Algebra.BigOperators.Group.Finset.Basic
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Algebra.Order.Ring.Abs
public import Mathlib.Analysis.SpecialFunctions.Pow.Real
public import NN.MLTheory.LearningTheory.Stability.Core
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# 1D ridge regression: replace-one uniform stability (squared loss)

This file is a self-contained, fully formalized worked example:

- we define the *closed-form* 1D ridge-regression estimator, and
- we prove a deterministic **replace-one uniform stability** bound for the squared loss, under
  bounded inputs.

## Proof outline (informal)

At a high level, the uniform stability proof follows the standard “strongly convex ERM is stable”
template, specialized to the 1D closed-form ridge solution:

1. Express `ŵ(S)` and `ŵ(S')` as ratios of sums `sumXY / (sumXX + λ*N)`.
2. Bound how much the numerator `sumXY` and denominator `sumXX` can change when one example is
   replaced (via a simple finite-sum perturbation lemma).
3. Bound the change in the reciprocal of the denominator, hence bound `|ŵ(S) - ŵ(S')|`.
4. Translate a bound on `|w-w'|` into a bound on the loss change for the squared loss
   `(w*x - y)^2` by factoring a difference of squares.

## Ridge regression in 1D (math)

Each example is a pair `(x,y) ∈ ℝ×ℝ`. For a dataset `S` of size `N`, ridge regression (with
regularization parameter `λ > 0`) minimizes the regularized squared loss

`(1/N) * ∑ᵢ (w*xᵢ - yᵢ)² + λ*w²`.

In 1D, the minimizer has the familiar closed form

`ŵ(S) = (∑ᵢ xᵢ yᵢ) / (∑ᵢ xᵢ² + λ*N)`.

In this file we set `N = n+1` (so indices are `Fin (n+1)`), because “remove-at” and “replace-at”
operations are most convenient in that convention in our `Dataset` library.

## Datasets as tensors

In `Stability.Core`, a dataset `Dataset N Z` is a length-`N` spec tensor (`Spec.Vec N Z`).
We use `Dataset.get S i` to access the `i`-th example.

## Stability statement (informal)

Let `S'` be `S` with one example replaced. If inputs are bounded by `|x| ≤ X` and `|y| ≤ Y`, then
for any test point `z` we bound

`|ℓ(ŵ(S), z) - ℓ(ŵ(S'), z)|`

where `ℓ(w,(x,y)) = (w*x - y)²`.

The final bound is explicit (a rational expression in `X,Y,λ,N`) and matches the expected scaling
for strongly convex regularized ERM: it is `O(1/(λ*N))` up to problem-dependent constants.

This is intended as a small, fully proved example that can be cited in documentation/papers.

## References / citations (informal pointers)

- Ridge/Tikhonov regularization: Tikhonov (1963); Hoerl & Kennard (1970), “Ridge Regression: Biased
  Estimation…”.
- Stability and generalization: Bousquet & Elisseeff (2002), “Stability and Generalization”.
- Stability for regularized ERM / strong convexity: Shalev-Shwartz et al. (2010), “Learnability,
  Stability and Uniform Convergence”.
- For additional viewpoints on stability and generalization, see also: Poggio, Rifkin, Mukherjee &
  Niyogi (2004), “General conditions for predictivity in learning theory”.
-/

@[expose] public section


noncomputable section

open scoped BigOperators

namespace NN.MLTheory.LearningTheory.Stability.RidgeRegression1D

variable {n : Nat}

/-! ## Bounded examples -/

/--
An example `(x,y)` together with bounds `|x| ≤ X` and `|y| ≤ Y`.

This lets us state stability bounds as theorems with explicit constants in terms of `X` and `Y`.
-/
def BoundedExample (X Y : ℝ) : Type :=
  {p : ℝ × ℝ // |p.1| ≤ X ∧ |p.2| ≤ Y}

namespace BoundedExample

variable {X Y : ℝ}

/-!
We keep `BoundedExample` as a subtype so bounds are carried as hypotheses in the type and can be
reused uniformly throughout the proof (instead of repeating assumptions).
-/

@[simp] def x (z : BoundedExample X Y) : ℝ := z.1.1
/-- The `y` coordinate of a bounded example. -/
@[simp] def y (z : BoundedExample X Y) : ℝ := z.1.2

/-- The `x` coordinate satisfies the declared bound `|x| ≤ X`. -/
theorem abs_x_le (z : BoundedExample X Y) : |z.x| ≤ X := z.2.1
/-- The `y` coordinate satisfies the declared bound `|y| ≤ Y`. -/
theorem abs_y_le (z : BoundedExample X Y) : |z.y| ≤ Y := z.2.2

/-- The declared bound `X` is nonnegative (because `|x| ≤ X`). -/
theorem X_nonneg (z : BoundedExample X Y) : 0 ≤ X :=
  le_trans (abs_nonneg z.x) z.abs_x_le

/-- The declared bound `Y` is nonnegative (because `|y| ≤ Y`). -/
theorem Y_nonneg (z : BoundedExample X Y) : 0 ≤ Y :=
  le_trans (abs_nonneg z.y) z.abs_y_le

end BoundedExample

/-! ## Sums and estimator -/

section

variable {X Y : ℝ}

/-- Sum of squares `∑ xᵢ²`. -/
def sumXX (S : Dataset (n + 1) (BoundedExample X Y)) : ℝ :=
  ∑ i ∈ (Finset.univ : Finset (Fin (n + 1))), (Dataset.get S i).x ^ 2

/-- Cross-term sum `∑ xᵢ yᵢ`. -/
def sumXY (S : Dataset (n + 1) (BoundedExample X Y)) : ℝ :=
  ∑ i ∈ (Finset.univ : Finset (Fin (n + 1))), (Dataset.get S i).x * (Dataset.get S i).y

/--
Closed-form 1D ridge fit.

`ridgeFit1D λ S = (∑ xᵢ yᵢ) / (∑ xᵢ² + λ*N)` where `N = n+1`.
-/
def ridgeFit1D (lam : ℝ) (S : Dataset (n + 1) (BoundedExample X Y)) : ℝ :=
  let N : ℝ := ((n + 1 : Nat) : ℝ)
  (sumXY (n := n) S) / (sumXX (n := n) S + lam * N)

/-- Squared loss `ℓ(w,(x,y)) = (w*x - y)²`. -/
def sqLoss (w : ℝ) (z : BoundedExample X Y) : ℝ :=
  (w * z.x - z.y) ^ 2

end

/-! ## Generic “sum changes at one index” lemma -/

section

variable {X Y : ℝ}

/--
If you replace a single element of a dataset, then the change in a sum over the dataset can be
written as a single-term difference.

This is a standard “finite sum perturbation” identity and is the main combinatorial input needed
to control `sumXX` and `sumXY` under replace-one.
-/
private lemma sum_replaceAt_sub {Z : Type} (φ : Z → ℝ)
    (S : Dataset (n + 1) Z) (i : Fin (n + 1)) (z' : Z) :
    (∑ j ∈ (Finset.univ : Finset (Fin (n + 1))), φ (Dataset.get S j)) -
        (∑ j ∈ (Finset.univ : Finset (Fin (n + 1))), φ (Dataset.get (replaceAt S i z') j)) =
      φ (Dataset.get S i) - φ z' := by
  classical
  set s : Finset (Fin (n + 1)) := (Finset.univ : Finset (Fin (n + 1)))
  have hi : i ∈ s := by simp [s]
  let f : Fin (n + 1) → ℝ := fun j => φ (Dataset.get S j)
  have hS' : (fun j => φ (Dataset.get (replaceAt S i z') j)) = Function.update f i (φ z') := by
    funext j
    by_cases h : j = i
    · subst h
      simp [replaceAt, Dataset.get, Dataset.toFn, Dataset.ofFn, Function.update]
    · simp [replaceAt, f, Function.update, h]
  have hS : (fun j => φ (Dataset.get S j)) = Function.update f i (f i) := by
    funext j
    by_cases h : j = i
    · subst h; simp [f, Function.update]
    · simp [f, Function.update, h]
  have hsumS :
      (∑ j ∈ s, φ (Dataset.get S j)) = (∑ j ∈ s, Function.update f i (f i) j) := by
    simp [hS]
  have hsumS' :
      (∑ j ∈ s, φ (Dataset.get (replaceAt S i z') j)) = (∑ j ∈ s, Function.update f i (φ z') j) :=
        by
    simp [hS']
  -- Now apply `Finset.sum_update_of_mem` to both sums and cancel the common remainder.
  have hUpd1 : (∑ j ∈ s, Function.update f i (f i) j) = f i + ∑ j ∈ s \ {i}, f j := by
    simpa using (Finset.sum_update_of_mem (s := s) (i := i) hi f (f i))
  have hUpd2 : (∑ j ∈ s, Function.update f i (φ z') j) = φ z' + ∑ j ∈ s \ {i}, f j := by
    simpa using (Finset.sum_update_of_mem (s := s) (i := i) hi f (φ z'))
  have hsub :
      (∑ j ∈ s, φ (Dataset.get S j)) - (∑ j ∈ s, φ (Dataset.get (replaceAt S i z') j)) =
        (f i + ∑ j ∈ s \ {i}, f j) - (φ z' + ∑ j ∈ s \ {i}, f j) := by
    -- rewrite both sums via the update lemmas
    rw [hsumS, hsumS', hUpd1, hUpd2]
  calc
    (∑ j ∈ s, φ (Dataset.get S j)) - (∑ j ∈ s, φ (Dataset.get (replaceAt S i z') j))
        = (f i + ∑ j ∈ s \ {i}, f j) - (φ z' + ∑ j ∈ s \ {i}, f j) := hsub
    _ = f i - φ z' := by ring
    _ = φ (Dataset.get S i) - φ z' := by rfl

end

/-! ## Ridge stability proof -/

namespace Ridge1D

variable {X Y lam : ℝ}

/-!
Everything below is “analysis lemmas” that culminate in the final uniform stability theorem.
The section exposes the headline theorem while keeping intermediate constants and algebraic bounds
local to the proof.
-/

def N : ℝ := ((n + 1 : Nat) : ℝ)

/-! `N = n+1` is positive as a real number. -/
lemma N_pos : 0 < N (n := n) := by
  simpa [N] using (Nat.cast_pos.mpr (Nat.succ_pos n))

/-! `sumXX` is nonnegative (it is a sum of squares). -/
private lemma sumXX_nonneg (S : Dataset (n + 1) (BoundedExample X Y)) :
    0 ≤ sumXX (n := n) S := by
  classical
  refine Finset.sum_nonneg ?_
  intro i hi
  have : 0 ≤ (Dataset.get S i).x ^ 2 := by nlinarith
  simpa using this

/-!
The ridge denominator `sumXX(S) + λ*N` is positive when `λ > 0`.

This ensures the closed-form ratio is well-defined and lets us use order properties of division.
-/
private lemma denom_pos (hlam : 0 < lam) (S : Dataset (n + 1) (BoundedExample X Y)) :
    0 < sumXX (n := n) S + lam * N (n := n) := by
  have h1 : 0 ≤ sumXX (n := n) S := sumXX_nonneg (n := n) (X := X) (Y := Y) S
  have h2 : 0 < lam * N (n := n) := mul_pos hlam (N_pos (n := n))
  linarith

/-!
Lower bound on the ridge denominator: `λ*N ≤ sumXX(S) + λ*N`.

We use this to replace the (dataset-dependent) denominator with a uniform lower bound.
-/
private lemma denom_lower (S : Dataset (n + 1) (BoundedExample X Y)) :
    lam * N (n := n) ≤ sumXX (n := n) S + lam * N (n := n) := by
  have h1 : 0 ≤ sumXX (n := n) S := sumXX_nonneg (n := n) (X := X) (Y := Y) S
  linarith

/-!
Absolute bound on the cross-term sum `sumXY`.

This is a simple consequence of the bounds `|x| ≤ X` and `|y| ≤ Y`.
-/
private lemma abs_sumXY_le (S : Dataset (n + 1) (BoundedExample X Y)) :
    |sumXY (n := n) S| ≤ N (n := n) * X * Y := by
  classical
  have hterm :
      ∀ i : Fin (n + 1), |(Dataset.get S i).x * (Dataset.get S i).y| ≤ X * Y := by
    intro i
    have hx : |(Dataset.get S i).x| ≤ X := (Dataset.get S i).abs_x_le
    have hy : |(Dataset.get S i).y| ≤ Y := (Dataset.get S i).abs_y_le
    have hX0 : 0 ≤ X := (Dataset.get S i).X_nonneg
    have hY0 : 0 ≤ Y := (Dataset.get S i).Y_nonneg
    calc
      |(Dataset.get S i).x * (Dataset.get S i).y| =
          |(Dataset.get S i).x| * |(Dataset.get S i).y| := by simp [abs_mul]
      _ ≤ X * Y := by
            exact mul_le_mul hx hy (abs_nonneg _) hX0
  calc
    |sumXY (n := n) S|
        ≤ ∑ i ∈ (Finset.univ : Finset (Fin (n + 1))), |(Dataset.get S i).x * (Dataset.get S i).y| :=
          by
            simpa [sumXY] using
              (Finset.abs_sum_le_sum_abs (s := (Finset.univ : Finset (Fin (n + 1))))
                (f := fun i => (Dataset.get S i).x * (Dataset.get S i).y))
    _ ≤ ∑ _i ∈ (Finset.univ : Finset (Fin (n + 1))), (X * Y) := by
          refine Finset.sum_le_sum ?_
          intro i hi
          simpa using hterm i
    _ = ((n + 1 : Nat) : ℝ) * (X * Y) := by
          simp [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
    _ = N (n := n) * X * Y := by
          simp [N]
          ring_nf

/-!
Replacing one example changes `sumXY` by at most `2*X*Y`.

This is the “numerator perturbation” bound for the ridge closed form.
-/
private lemma abs_sumXY_sub_replaceAt_le (S : Dataset (n + 1) (BoundedExample X Y)) (i : Fin (n +
  1))
    (z' : BoundedExample X Y) :
    |sumXY (n := n) S - sumXY (n := n) (replaceAt S i z')| ≤ 2 * X * Y := by
  classical
  have hdiff :
      sumXY (n := n) S - sumXY (n := n) (replaceAt S i z') =
        (Dataset.get S i).x * (Dataset.get S i).y - z'.x * z'.y := by
    -- Use the generic sum-change lemma with `φ(z) = x*z`.
    have := (sum_replaceAt_sub (φ := fun z : BoundedExample X Y => z.x * z.y) S i z')
    simpa [sumXY] using this
  have hAi : |(Dataset.get S i).x * (Dataset.get S i).y| ≤ X * Y := by
    have hx : |(Dataset.get S i).x| ≤ X := (Dataset.get S i).abs_x_le
    have hy : |(Dataset.get S i).y| ≤ Y := (Dataset.get S i).abs_y_le
    have hX0 : 0 ≤ X := (Dataset.get S i).X_nonneg
    calc
      |(Dataset.get S i).x * (Dataset.get S i).y| =
          |(Dataset.get S i).x| * |(Dataset.get S i).y| := by simp [abs_mul]
      _ ≤ X * Y := by exact mul_le_mul hx hy (abs_nonneg _) hX0
  have hA' : |z'.x * z'.y| ≤ X * Y := by
    have hx : |z'.x| ≤ X := z'.abs_x_le
    have hy : |z'.y| ≤ Y := z'.abs_y_le
    have hX0 : 0 ≤ X := z'.X_nonneg
    calc
      |z'.x * z'.y| = |z'.x| * |z'.y| := by simp [abs_mul]
      _ ≤ X * Y := by exact mul_le_mul hx hy (abs_nonneg _) hX0
  calc
    |sumXY (n := n) S - sumXY (n := n) (replaceAt S i z')|
        = |(Dataset.get S i).x * (Dataset.get S i).y - z'.x * z'.y| := by simp [hdiff]
    _ ≤ |(Dataset.get S i).x * (Dataset.get S i).y| + |z'.x * z'.y| := by
          simpa [sub_eq_add_neg] using abs_add_le ((Dataset.get S i).x * (Dataset.get S i).y)
            (-(z'.x * z'.y))
    _ ≤ X * Y + X * Y := by nlinarith [hAi, hA']
    _ = 2 * X * Y := by ring

/-!
Replacing one example changes `sumXX` by at most `2*X^2`.

This is the “denominator perturbation” bound for the ridge closed form.
-/
private lemma abs_sumXX_sub_replaceAt_le (S : Dataset (n + 1) (BoundedExample X Y)) (i : Fin (n +
  1))
    (z' : BoundedExample X Y) :
    |sumXX (n := n) S - sumXX (n := n) (replaceAt S i z')| ≤ 2 * X ^ 2 := by
  classical
  have hdiff :
      sumXX (n := n) S - sumXX (n := n) (replaceAt S i z') =
        (Dataset.get S i).x ^ 2 - z'.x ^ 2 := by
    have := (sum_replaceAt_sub (φ := fun z : BoundedExample X Y => z.x ^ 2) S i z')
    simpa [sumXX] using this
  have hsq_i : (Dataset.get S i).x ^ 2 ≤ X ^ 2 := by
    have hx : |(Dataset.get S i).x| ≤ X := (Dataset.get S i).abs_x_le
    have hX0 : 0 ≤ X := (Dataset.get S i).X_nonneg
    have habs_sq : |(Dataset.get S i).x| ^ 2 ≤ X ^ 2 := by
      have : |(Dataset.get S i).x| * |(Dataset.get S i).x| ≤ X * X :=
        mul_le_mul hx hx (abs_nonneg _) hX0
      simpa [pow_two] using this
    -- convert `|x|^2` to `x^2`
    simpa [sq_abs] using habs_sq
  have hsq' : z'.x ^ 2 ≤ X ^ 2 := by
    have hx : |z'.x| ≤ X := z'.abs_x_le
    have hX0 : 0 ≤ X := z'.X_nonneg
    have habs_sq : |z'.x| ^ 2 ≤ X ^ 2 := by
      have : |z'.x| * |z'.x| ≤ X * X :=
        mul_le_mul hx hx (abs_nonneg _) hX0
      simpa [pow_two] using this
    simpa [sq_abs] using habs_sq
  calc
    |sumXX (n := n) S - sumXX (n := n) (replaceAt S i z')|
        = |(Dataset.get S i).x ^ 2 - z'.x ^ 2| := by simp [hdiff]
    _ ≤ (Dataset.get S i).x ^ 2 + z'.x ^ 2 := by
          have h := abs_add_le ((Dataset.get S i).x ^ 2) (-(z'.x ^ 2))
          calc
            |(Dataset.get S i).x ^ 2 - z'.x ^ 2|
                = |(Dataset.get S i).x ^ 2 + -(z'.x ^ 2)| := by simp [sub_eq_add_neg]
            _ ≤ |(Dataset.get S i).x ^ 2| + |-(z'.x ^ 2)| := h
            _ = (Dataset.get S i).x ^ 2 + z'.x ^ 2 := by simp
    _ ≤ 2 * X ^ 2 := by nlinarith [hsq_i, hsq']

/-!
Bound the magnitude of the fitted ridge weight.

This is a quick, coarse bound of the form `|ŵ(S)| ≤ (X*Y)/λ`.
-/
private lemma abs_w_le (hlam : 0 < lam) (S : Dataset (n + 1) (BoundedExample X Y)) :
    |ridgeFit1D (n := n) (X := X) (Y := Y) lam S| ≤ (X * Y) / lam := by
  classical
  set D : ℝ := sumXX (n := n) (X := X) (Y := Y) S + lam * N (n := n)
  have hDpos : 0 < D := by
    simpa [D] using (denom_pos (n := n) (X := X) (Y := Y) (lam := lam) hlam S)
  have hDge : lam * N (n := n) ≤ D := by
    simpa [D] using (denom_lower (n := n) (X := X) (Y := Y) (lam := lam) S)
  have hlamNpos : 0 < lam * N (n := n) := mul_pos hlam (N_pos (n := n))
  have hB : |sumXY (n := n) (X := X) (Y := Y) S| ≤ N (n := n) * X * Y :=
    abs_sumXY_le (n := n) (X := X) (Y := Y) S
  have habsD : |D| = D := abs_of_pos hDpos
  have hfit : ridgeFit1D (n := n) (X := X) (Y := Y) lam S = (sumXY (n := n) (X := X) (Y := Y) S) / D
    := by
    simp [ridgeFit1D, D, N]
  have hpos_num : 0 ≤ |sumXY (n := n) (X := X) (Y := Y) S| := abs_nonneg _
  calc
      |ridgeFit1D (n := n) (X := X) (Y := Y) lam S|
          = |sumXY (n := n) (X := X) (Y := Y) S| / D := by
              -- use `D>0` to remove `|D|` in `abs_div`
              simp [hfit, abs_div, habsD]
    _ ≤ |sumXY (n := n) (X := X) (Y := Y) S| / (lam * N (n := n)) := by
          exact div_le_div_of_nonneg_left hpos_num hlamNpos hDge
    _ ≤ (N (n := n) * X * Y) / (lam * N (n := n)) := by
          exact div_le_div_of_nonneg_right hB (le_of_lt hlamNpos)
    _ = (X * Y) / lam := by
          have hNne : (N (n := n)) ≠ 0 := ne_of_gt (N_pos (n := n))
          field_simp [N, hNne, (ne_of_gt hlam)]

/-!
Bound the residual `|ŵ(S)*x - y|` at a test point.

This is another coarse bound used at the very end when bounding the loss change via
`(e-e')*(e+e')` for `e = w*x-y`.
-/
private lemma abs_residual_le (hlam : 0 < lam) (S : Dataset (n + 1) (BoundedExample X Y)) (z :
  BoundedExample X Y) :
    |ridgeFit1D (n := n) (X := X) (Y := Y) lam S * z.x - z.y| ≤
      Y * (lam + X ^ 2) / lam := by
  have hw : |ridgeFit1D (n := n) (X := X) (Y := Y) lam S| ≤ (X * Y) / lam :=
    abs_w_le (n := n) (X := X) (Y := Y) (lam := lam) hlam S
  have hx : |z.x| ≤ X := z.abs_x_le
  have hy : |z.y| ≤ Y := z.abs_y_le
  calc
    |ridgeFit1D (n := n) (X := X) (Y := Y) lam S * z.x - z.y|
        ≤ |ridgeFit1D (n := n) (X := X) (Y := Y) lam S * z.x| + |z.y| := by
            simpa [sub_eq_add_neg] using abs_add_le (ridgeFit1D (n := n) (X := X) (Y := Y) lam S *
              z.x) (-z.y)
    _ = |ridgeFit1D (n := n) (X := X) (Y := Y) lam S| * |z.x| + |z.y| := by
          simp [abs_mul]
    _ ≤ ((X * Y) / lam) * X + Y := by
          have hXYlam_nonneg : 0 ≤ (X * Y) / lam := by
            have hX0 : 0 ≤ X := z.X_nonneg
            have hY0 : 0 ≤ Y := z.Y_nonneg
            exact div_nonneg (mul_nonneg hX0 hY0) (le_of_lt hlam)
          have hmul : |ridgeFit1D (n := n) (X := X) (Y := Y) lam S| * |z.x| ≤ ((X * Y) / lam) * X :=
            by
            exact mul_le_mul hw hx (abs_nonneg _) hXYlam_nonneg
          exact add_le_add hmul hy
    _ = Y * (lam + X ^ 2) / lam := by
          field_simp [(ne_of_gt hlam)]
          ring

/-!
## Main theorem: deterministic replace-one uniform stability

The next theorem is the headline result of this file. We increase the heartbeat limit because the
proof contains a fair amount of `simp`/`ring`/`field_simp`-style arithmetic bookkeeping.
-/

set_option maxHeartbeats 1000000 in
/--
**Uniform stability of 1D ridge regression (bounded inputs, squared loss).**

Assume `λ > 0`. Then the ridge estimator `ridgeFit1D λ` is uniformly stable (in the replace-one
sense) for the squared loss, with an explicit bound `β` that scales like `1/(λ * N)` where
`N = n+1`.

The stability notion used here is `UniformStableReplace` from `Stability.Core`.
-/
theorem ridgeFit1D_sqLoss_uniformStableReplace (hlam : 0 < lam) :
    UniformStableReplace (Z := BoundedExample X Y) (H := ℝ)
      (A := fun S => ridgeFit1D (n := n) (X := X) (Y := Y) lam S)
      (ℓ := fun w z => sqLoss (X := X) (Y := Y) w z)
      (β := 4 * X ^ 2 * Y ^ 2 * (lam + X ^ 2) ^ 2 / (lam ^ 3 * N (n := n))) := by
  classical
  intro S i z z'
  set S' : Dataset (n + 1) (BoundedExample X Y) := replaceAt S i z'
  set w : ℝ := ridgeFit1D (n := n) (X := X) (Y := Y) lam S
  set w' : ℝ := ridgeFit1D (n := n) (X := X) (Y := Y) lam S'
  set D : ℝ := sumXX (n := n) (X := X) (Y := Y) S + lam * N (n := n)
  set D' : ℝ := sumXX (n := n) (X := X) (Y := Y) S' + lam * N (n := n)
  have hDpos : 0 < D := by
    simpa [D] using (denom_pos (n := n) (X := X) (Y := Y) (lam := lam) hlam S)
  have hD'pos : 0 < D' := by
    simpa [D'] using (denom_pos (n := n) (X := X) (Y := Y) (lam := lam) hlam S')
  have hDge : lam * N (n := n) ≤ D := by
    simpa [D] using (denom_lower (n := n) (X := X) (Y := Y) (lam := lam) S)
  have hD'ge : lam * N (n := n) ≤ D' := by
    simpa [D'] using (denom_lower (n := n) (X := X) (Y := Y) (lam := lam) S')
  have hlamNpos : 0 < lam * N (n := n) := mul_pos hlam (N_pos (n := n))
  have hBdiff : |sumXY (n := n) (X := X) (Y := Y) S - sumXY (n := n) (X := X) (Y := Y) S'| ≤ 2 * X *
    Y := by
    simpa [S'] using abs_sumXY_sub_replaceAt_le (n := n) (X := X) (Y := Y) S i z'
  have hB' : |sumXY (n := n) (X := X) (Y := Y) S'| ≤ N (n := n) * X * Y :=
    abs_sumXY_le (n := n) (X := X) (Y := Y) S'

  have hw_def : w = (sumXY (n := n) (X := X) (Y := Y) S) / D := by
    simp [w, ridgeFit1D, D, N]
  have hw'_def : w' = (sumXY (n := n) (X := X) (Y := Y) S') / D' := by
    simp [w', ridgeFit1D, D', N, S']

  -- Bound |1/D - 1/D'|
  have hA_diff : |sumXX (n := n) (X := X) (Y := Y) S - sumXX (n := n) (X := X) (Y := Y) S'| ≤ 2 * X
    ^ 2 := by
    simpa [S'] using abs_sumXX_sub_replaceAt_le (n := n) (X := X) (Y := Y) S i z'
  have hInv :
      |(1 / D) - (1 / D')| ≤ (2 * X ^ 2) / (lam ^ 2 * (N (n := n)) ^ 2) := by
    have hDne : D ≠ 0 := ne_of_gt hDpos
    have hD'ne : D' ≠ 0 := ne_of_gt hD'pos
    have hInvEq : (1 / D) - (1 / D') = (D' - D) / (D * D') := by
      simpa [one_div] using (inv_sub_inv (a := D) (b := D') hDne hD'ne)
    have hposDD' : 0 < D * D' := mul_pos hDpos hD'pos
    have hDD'ge : (lam * N (n := n)) ^ 2 ≤ D * D' := by
      nlinarith [hDge, hD'ge, le_of_lt hDpos, le_of_lt hD'pos]
    have hDdiff : |D' - D| ≤ 2 * X ^ 2 := by
      have : D' - D = sumXX (n := n) (X := X) (Y := Y) S' - sumXX (n := n) (X := X) (Y := Y) S := by
        simp [D, D']
      have : |D' - D| = |sumXX (n := n) (X := X) (Y := Y) S - sumXX (n := n) (X := X) (Y := Y) S'|
        := by
        simp [this, abs_sub_comm]
      simpa [this] using hA_diff
    have hn : 0 ≤ 2 * X ^ 2 := by nlinarith
    calc
      |(1 / D) - (1 / D')|
          = |(D' - D) / (D * D')| := by
              simpa using congrArg abs hInvEq
      _ = |D' - D| / (D * D') := by simp [abs_div, abs_of_pos hposDD']
      _ ≤ (2 * X ^ 2) / (D * D') := by
            exact div_le_div_of_nonneg_right hDdiff (le_of_lt hposDD')
      _ ≤ (2 * X ^ 2) / ((lam * N (n := n)) ^ 2) := by
            exact div_le_div_of_nonneg_left hn (sq_pos_of_pos hlamNpos) hDD'ge
      _ = (2 * X ^ 2) / (lam ^ 2 * (N (n := n)) ^ 2) := by ring_nf

  -- Bound |w - w'|.
  have hw_diff :
      |w - w'| ≤ (2 * X * Y * (lam + X ^ 2)) / (lam ^ 2 * N (n := n)) := by
    have habsD : |D| = D := abs_of_pos hDpos
    have hterm1 :
        |(sumXY (n := n) (X := X) (Y := Y) S - sumXY (n := n) (X := X) (Y := Y) S') / D|
          ≤ (2 * X * Y) / (lam * N (n := n)) := by
      have h0 : 0 ≤ |sumXY (n := n) (X := X) (Y := Y) S - sumXY (n := n) (X := X) (Y := Y) S'| :=
        abs_nonneg _
      have : |(sumXY (n := n) (X := X) (Y := Y) S - sumXY (n := n) (X := X) (Y := Y) S') / D|
          = |sumXY (n := n) (X := X) (Y := Y) S - sumXY (n := n) (X := X) (Y := Y) S'| / D := by
            simp [abs_div, habsD]
      calc
        |(sumXY (n := n) (X := X) (Y := Y) S - sumXY (n := n) (X := X) (Y := Y) S') / D|
            = |sumXY (n := n) (X := X) (Y := Y) S - sumXY (n := n) (X := X) (Y := Y) S'| / D := this
        _ ≤ |sumXY (n := n) (X := X) (Y := Y) S - sumXY (n := n) (X := X) (Y := Y) S'|
              / (lam * N (n := n)) := by
              exact div_le_div_of_nonneg_left h0 hlamNpos hDge
        _ ≤ (2 * X * Y) / (lam * N (n := n)) := by
              exact div_le_div_of_nonneg_right hBdiff (le_of_lt hlamNpos)
    have hterm2 :
        |sumXY (n := n) (X := X) (Y := Y) S'| * |(1 / D) - (1 / D')|
          ≤ (2 * X ^ 3 * Y) / (lam ^ 2 * N (n := n)) := by
      have hX0 : 0 ≤ X := (Dataset.get S' 0).X_nonneg
      have hY0 : 0 ≤ Y := (Dataset.get S' 0).Y_nonneg
      have hN0 : 0 ≤ N (n := n) := le_of_lt (N_pos (n := n))
      have hB'' : |sumXY (n := n) (X := X) (Y := Y) S'| ≤ (N (n := n) * X * Y) := hB'
      have hB0 : 0 ≤ N (n := n) * X * Y := by
        have : 0 ≤ N (n := n) * X := mul_nonneg hN0 hX0
        simpa [mul_assoc] using (mul_nonneg this hY0)
      have hMul :
          |sumXY (n := n) (X := X) (Y := Y) S'| * |(1 / D) - (1 / D')|
            ≤ (N (n := n) * X * Y) * ((2 * X ^ 2) / (lam ^ 2 * (N (n := n)) ^ 2)) := by
        exact mul_le_mul hB'' hInv (abs_nonneg _) hB0
      have hNne : (N (n := n)) ≠ 0 := ne_of_gt (N_pos (n := n))
      -- simplify the right-hand side
      have hSimp :
          (N (n := n) * X * Y) * ((2 * X ^ 2) / (lam ^ 2 * (N (n := n)) ^ 2))
            = (2 * X ^ 3 * Y) / (lam ^ 2 * N (n := n)) := by
        field_simp [hNne]
      simpa [hSimp] using hMul
    have hsplit :
        |(sumXY (n := n) (X := X) (Y := Y) S) / D - (sumXY (n := n) (X := X) (Y := Y) S') / D'|
          ≤ (2 * X * Y) / (lam * N (n := n)) + (2 * X ^ 3 * Y) / (lam ^ 2 * N (n := n)) := by
      have htri :=
        abs_sub_le ((sumXY (n := n) (X := X) (Y := Y) S) / D)
          ((sumXY (n := n) (X := X) (Y := Y) S') / D)
          ((sumXY (n := n) (X := X) (Y := Y) S') / D')
      have h1 :
          |(sumXY (n := n) (X := X) (Y := Y) S) / D - (sumXY (n := n) (X := X) (Y := Y) S') / D|
            = |(sumXY (n := n) (X := X) (Y := Y) S - sumXY (n := n) (X := X) (Y := Y) S') / D| := by
              ring_nf
      have h2 :
          |(sumXY (n := n) (X := X) (Y := Y) S') / D - (sumXY (n := n) (X := X) (Y := Y) S') / D'|
            = |sumXY (n := n) (X := X) (Y := Y) S'| * |(1 / D) - (1 / D')| := by
        have hEq :
            (sumXY (n := n) (X := X) (Y := Y) S') / D - (sumXY (n := n) (X := X) (Y := Y) S') / D'
              = (sumXY (n := n) (X := X) (Y := Y) S') * ((1 / D) - (1 / D')) := by
          simp [div_eq_mul_inv]
          ring_nf
        calc
          |(sumXY (n := n) (X := X) (Y := Y) S') / D - (sumXY (n := n) (X := X) (Y := Y) S') / D'|
              = |(sumXY (n := n) (X := X) (Y := Y) S') * ((1 / D) - (1 / D'))| := by
                  simp [hEq]
          _ = |sumXY (n := n) (X := X) (Y := Y) S'| * |(1 / D) - (1 / D')| := by simp [abs_mul]
      have htri' :
          |(sumXY (n := n) (X := X) (Y := Y) S) / D - (sumXY (n := n) (X := X) (Y := Y) S') / D'|
            ≤ |(sumXY (n := n) (X := X) (Y := Y) S - sumXY (n := n) (X := X) (Y := Y) S') / D|
              + (|sumXY (n := n) (X := X) (Y := Y) S'| * |(1 / D) - (1 / D')|) := by
              simpa [h1, h2] using htri
      nlinarith [htri', hterm1, hterm2]
    have hw_eq :
        |w - w'| = |(sumXY (n := n) (X := X) (Y := Y) S) / D - (sumXY (n := n) (X := X) (Y := Y) S')
          / D'| := by
      simp [w, w', hw_def, hw'_def]
    have hsimp :
        (2 * X * Y) / (lam * N (n := n)) + (2 * X ^ 3 * Y) / (lam ^ 2 * N (n := n))
          = (2 * X * Y * (lam + X ^ 2)) / (lam ^ 2 * N (n := n)) := by
      have hNne : (N (n := n)) ≠ 0 := ne_of_gt (N_pos (n := n))
      field_simp [hNne, (ne_of_gt hlam)]
    calc
      |w - w'|
          = |(sumXY (n := n) (X := X) (Y := Y) S) / D - (sumXY (n := n) (X := X) (Y := Y) S') / D'|
            := hw_eq
      _ ≤ (2 * X * Y) / (lam * N (n := n)) + (2 * X ^ 3 * Y) / (lam ^ 2 * N (n := n)) := hsplit
      _ = (2 * X * Y * (lam + X ^ 2)) / (lam ^ 2 * N (n := n)) := hsimp

  -- Convert `|w-w'|` to a loss difference bound.
  set e : ℝ := w * z.x - z.y
  set e' : ℝ := w' * z.x - z.y
  have hres : |e| ≤ Y * (lam + X ^ 2) / lam := by
    simpa [e, w] using abs_residual_le (n := n) (X := X) (Y := Y) (lam := lam) hlam S z
  have hres' : |e'| ≤ Y * (lam + X ^ 2) / lam := by
    simpa [e', w', S'] using abs_residual_le (n := n) (X := X) (Y := Y) (lam := lam) hlam S' z
  have hx : |z.x| ≤ X := z.abs_x_le
  have he_diff : |e - e'| ≤ X * |w - w'| := by
    have : e - e' = (w - w') * z.x := by
      simp [e, e', sub_eq_add_neg]
      ring_nf
    calc
      |e - e'| = |(w - w') * z.x| := by simp [this]
      _ = |w - w'| * |z.x| := by simp [abs_mul]
      _ ≤ |w - w'| * X := by
            exact mul_le_mul_of_nonneg_left hx (abs_nonneg (w - w'))
      _ = X * |w - w'| := by ring
  have he_sum : |e + e'| ≤ 2 * (Y * (lam + X ^ 2) / lam) := by
    have : |e + e'| ≤ |e| + |e'| := by simpa using abs_add_le e e'
    nlinarith [this, hres, hres']
  have hloss :
      |sqLoss (X := X) (Y := Y) w z - sqLoss (X := X) (Y := Y) w' z|
        = |e - e'| * |e + e'| := by
    have hEq : sqLoss (X := X) (Y := Y) w z - sqLoss (X := X) (Y := Y) w' z = (e - e') * (e + e') :=
      by
      simp [sqLoss, e, e', pow_two]
      ring_nf
    calc
      |sqLoss (X := X) (Y := Y) w z - sqLoss (X := X) (Y := Y) w' z|
          = |(e - e') * (e + e')| := by simp [hEq]
      _ = |e - e'| * |e + e'| := by simp [abs_mul]
  calc
    |sqLoss (X := X) (Y := Y) w z - sqLoss (X := X) (Y := Y) w' z|
        = |e - e'| * |e + e'| := hloss
    _ ≤ (X * |w - w'|) * (2 * (Y * (lam + X ^ 2) / lam)) := by
          have hX0 : 0 ≤ X := z.X_nonneg
          have hBw0 : 0 ≤ X * |w - w'| := mul_nonneg hX0 (abs_nonneg _)
          have hProd :
              |e - e'| * |e + e'| ≤ (X * |w - w'|) * (2 * (Y * (lam + X ^ 2) / lam)) := by
            exact mul_le_mul he_diff he_sum (abs_nonneg _) hBw0
          exact hProd
    _ ≤ (X * ((2 * X * Y * (lam + X ^ 2)) / (lam ^ 2 * N (n := n))))
          * (2 * (Y * (lam + X ^ 2) / lam)) := by
            have hX0 : 0 ≤ X := z.X_nonneg
            have hC0 : 0 ≤ 2 * (Y * (lam + X ^ 2) / lam) := by
              have : 0 ≤ Y * (lam + X ^ 2) / lam := by
                have hY0 : 0 ≤ Y := z.Y_nonneg
                have : 0 ≤ lam + X ^ 2 := by nlinarith
                exact div_nonneg (mul_nonneg hY0 this) (le_of_lt hlam)
              nlinarith
            exact mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hw_diff hX0) hC0
    _ = 4 * X ^ 2 * Y ^ 2 * (lam + X ^ 2) ^ 2 / (lam ^ 3 * N (n := n)) := by
          have hNne : (N (n := n)) ≠ 0 := ne_of_gt (N_pos (n := n))
          field_simp [hNne, (ne_of_gt hlam)]
          ring_nf

end Ridge1D

end NN.MLTheory.LearningTheory.Stability.RidgeRegression1D

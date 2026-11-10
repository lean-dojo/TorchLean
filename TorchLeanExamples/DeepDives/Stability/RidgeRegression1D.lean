/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.LearningTheory.Stability
public import TorchLean.Tensor.Context.Real
public import TorchLean.Tensor.ConfiguredBinary
public import TorchLean.Verify.Floats.IEEE754.Expressions
import TorchLean.Tensor.List

/-!
# 1D ridge regression: replace-one uniform stability (squared loss)

The example has three parts:

- `fit`, the *closed-form* 1D ridge-regression estimator over any scalar `Context α`, with a
  deterministic **replace-one uniform stability** bound over the reals for the squared loss under
  bounded inputs,
- `Ridge.toReal_fit`, which relates the fit in any configured binary format to rounding
  after each operation, and
- `Ridge.Sample`, a two-point binary32 dataset for `#eval`.

## Proof outline (informal)

At a high level, the uniform stability proof follows the standard “strongly convex ERM is stable”
template, specialized to the 1D closed-form ridge solution:

1. Express $\widehat w(S)$ and $\widehat w(S')$ as ratios of sums
   $\mathrm{sumXY}/(\mathrm{sumXX}+\lambda N)$.
2. Bound how much the numerator `sumXY` and denominator `sumXX` can change when one example is
   replaced (via a simple finite-sum perturbation lemma).
3. Bound the change in the reciprocal of the denominator, hence bound
   $|\widehat w(S)-\widehat w(S')|$.
4. Translate a bound on $|w-w'|$ into a bound on the loss change for the squared loss
   $(wx-y)^2$ by factoring a difference of squares.

## Ridge regression in 1D (math)

Each example is a pair $(x,y)\in\mathbb R\times\mathbb R$. For a dataset $S$ of size $N$, ridge
regression with regularization parameter $\lambda>0$ minimizes

$$
\frac1N\sum_i(wx_i-y_i)^2+\lambda w^2.
$$

In 1D, the minimizer has the familiar closed form

$$
\widehat w(S)=\frac{\sum_i x_i y_i}{\sum_i x_i^2+\lambda N}.
$$

Here $N=n+1$, so indices are `Fin (n+1)`. The convention keeps $N$ positive, which the denominator
bounds need, and it matches “remove-at” in the `Dataset` library, which takes a dataset of size
$n+1$.

## Datasets as tensors

In `TorchLean.NN.LearningTheory.Stability`, a dataset `Dataset N Z` is a tensor of shape `[N]`
(`TorchLean.Tensor Z [N]`). `Dataset.toFn S i` is the `i`-th example.

## Stability statement (informal)

Let $S'$ be $S$ with one example replaced. If inputs satisfy $|x|\le X$ and $|y|\le Y$, then
for any test point $z$ the theorem bounds

$|\ell(\widehat w(S),z)-\ell(\widehat w(S'),z)|$,

where $\ell(w,(x,y))=(wx-y)^2$.

The final bound is $4X^2Y^2(\lambda+X^2)^2/(\lambda^3 N)$. It scales as $1/N$ for fixed
$X,Y,\lambda>0$. Its dependence on $\lambda$ includes an inverse-cubic term, so it is not a
uniform $O(1/(\lambda N))$ estimate as $\lambda$ tends to zero.

## Floating-point formats

`fit` adds with `Fin.foldl`, so the summation order is fixed, and in a floating-point format
every multiplication, addition and division rounds in that format.
`Ridge.eval_fitExpr` shows that the expression `Ridge.fitExpr` evaluates to the fit in any
configured binary format, and `Ridge.toReal_fit` shows that, in an IEEE
format and when the evaluation stays finite, the real value of the fit equals the
round-after-each-operation semantics of `Ridge.fitExpr`. The example dataset runs in binary32.
`abs_toReal_ridgeFit_sub_ridgeFit_le` bounds the distance from the real value of the fit to the
real fit of the decoded data by the forward error bound `IEEE754.errorBound` of `Ridge.fitExpr`.

## References / citations (informal pointers)

- Ridge/Tikhonov regularization: Tikhonov (1963) and Hoerl & Kennard (1970), “Ridge Regression:
  Biased Estimation…”.
- Stability and generalization: Bousquet & Elisseeff (2002), “Stability and Generalization”.
- Stability for regularized ERM / strong convexity: Shalev-Shwartz et al. (2010), “Learnability,
  Stability and Uniform Convergence”.
- For additional viewpoints on stability and generalization, see also: Poggio, Rifkin, Mukherjee &
  Niyogi (2004), “General conditions for predictivity in learning theory”.
-/

@[expose] public section

open TorchLean.LearningTheory Dataset

noncomputable section

open scoped BigOperators

namespace TorchLean.Examples.Ridge

variable {n : Nat}

/-!
## The fit over any scalar

We add with `Fin.foldl` rather than `Finset.sum`, because a `Context α` need not be a commutative
monoid and in a floating-point format the order of the additions changes the rounded result. Over
`ℝ`, `sumXX_eq_sum` and `sumXY_eq_sum` give the `Finset` sums.
-/

section Fit

variable {α : Type} [Context α]

/-- The sum of the products `x * x` over the pairs `(x, y)` of `S`, added from left to right. -/
def sumXX (S : Dataset (n + 1) (α × α)) : α :=
  Fin.foldl (n + 1) (fun acc i => acc + (Dataset.toFn S i).1 * (Dataset.toFn S i).1) 0

/-- The sum of the products `x * y` over the pairs `(x, y)` of `S`, added from left to right. -/
def sumXY (S : Dataset (n + 1) (α × α)) : α :=
  Fin.foldl (n + 1) (fun acc i => acc + (Dataset.toFn S i).1 * (Dataset.toFn S i).2) 0

/--
The closed-form 1D ridge fit $\operatorname{ridgeFit}(\lambda,S)
=\frac{\sum_i x_i y_i}{\sum_i x_i^2+\lambda N}$ with $N=n+1$, computed in `α`.
-/
def fit (lam : α) (S : Dataset (n + 1) (α × α)) : α :=
  sumXY S / (sumXX S + lam * ((n + 1 : Nat) : α))

end Fit

/-! ## Bounded examples -/

/--
An example $(x,y)$ together with bounds $|x|\le X$ and $|y|\le Y$.

With the bounds attached, the stability theorems state their constants in terms of `X` and `Y`.
The bounds are carried in the subtype, so every lemma below can use them without repeating
hypotheses.
-/
def BoundedExample (X Y : ℝ) : Type :=
  {p : ℝ × ℝ // |p.1| ≤ X ∧ |p.2| ≤ Y}

namespace BoundedExample

variable {X Y : ℝ}

/-- The `x` coordinate of a bounded example. -/
@[simp] def x (z : BoundedExample X Y) : ℝ := z.1.1
/-- The `y` coordinate of a bounded example. -/
@[simp] def y (z : BoundedExample X Y) : ℝ := z.1.2

/-- The `x` coordinate satisfies the declared bound $|x|\le X$. -/
theorem abs_x_le (z : BoundedExample X Y) : |z.x| ≤ X := z.2.1
/-- The `y` coordinate satisfies the declared bound $|y|\le Y$. -/
theorem abs_y_le (z : BoundedExample X Y) : |z.y| ≤ Y := z.2.2

/-- The declared bound `X` is nonnegative because $|x|\le X$. -/
theorem X_nonneg (z : BoundedExample X Y) : 0 ≤ X :=
  le_trans (abs_nonneg z.x) z.abs_x_le

/-- The declared bound `Y` is nonnegative because $|y|\le Y$. -/
theorem Y_nonneg (z : BoundedExample X Y) : 0 ≤ Y :=
  le_trans (abs_nonneg z.y) z.abs_y_le

/-- The cross term of a bounded example satisfies $|xy|\le XY$. -/
theorem abs_x_mul_y_le (z : BoundedExample X Y) : |z.x * z.y| ≤ X * Y := by
  rw [abs_mul]
  exact mul_le_mul z.abs_x_le z.abs_y_le (abs_nonneg _) z.X_nonneg

/-- The squared input of a bounded example satisfies $x^2\le X^2$. -/
theorem sq_x_le (z : BoundedExample X Y) : z.x ^ 2 ≤ X ^ 2 := by
  rw [← sq_abs z.x]
  exact pow_le_pow_left₀ (abs_nonneg _) z.abs_x_le 2

end BoundedExample

/-! ## The real estimator -/

/-- Over `ℝ`, `sumXX S` is the `Finset` sum of the squared inputs. -/
theorem sumXX_eq_sum (S : Dataset (n + 1) (ℝ × ℝ)) :
    sumXX S = ∑ i, (Dataset.toFn S i).1 ^ 2 := by
  simp only [sq]
  rw [sumXX, Fin.foldl_eq_finRange_foldl]
  exact ((Fin.sum_univ_def _).trans (List.sum_eq_foldl.trans List.foldl_map)).symm

/-- Over `ℝ`, `sumXY S` is the `Finset` sum of the products `x * y`. -/
theorem sumXY_eq_sum (S : Dataset (n + 1) (ℝ × ℝ)) :
    sumXY S = ∑ i, (Dataset.toFn S i).1 * (Dataset.toFn S i).2 := by
  rw [sumXY, Fin.foldl_eq_finRange_foldl]
  exact ((Fin.sum_univ_def _).trans (List.sum_eq_foldl.trans List.foldl_map)).symm

section

variable {X Y : ℝ}

/-- The `(x, y)` pairs of a dataset of bounded examples. -/
def pairs {m : Nat} (S : Dataset m (BoundedExample X Y)) : Dataset m (ℝ × ℝ) :=
  Dataset.ofFn fun i => (Dataset.toFn S i).1

/-- The `i`-th pair of `pairs S` is the pair of the `i`-th example of `S`. -/
theorem toFn_pairs {m : Nat} (S : Dataset m (BoundedExample X Y)) (i : Fin m) :
    Dataset.toFn (pairs S) i = (Dataset.toFn S i).1 := by
  rw [pairs, Dataset.toFn_ofFn_apply]

/-- The ridge fit `fit` over `ℝ` of the pairs of a dataset of bounded examples. -/
def estimator (lam : ℝ) (S : Dataset (n + 1) (BoundedExample X Y)) : ℝ :=
  fit lam (pairs S)

/-- `estimator lam S` is the ratio $\frac{\sum_i x_i y_i}{\sum_i x_i^2+\lambda N}$ of `Finset`
sums, with $N=n+1$. -/
theorem estimator_eq (lam : ℝ) (S : Dataset (n + 1) (BoundedExample X Y)) :
    estimator lam S = (∑ i, (Dataset.toFn S i).x * (Dataset.toFn S i).y) /
      (∑ i, (Dataset.toFn S i).x ^ 2 + lam * ((n + 1 : Nat) : ℝ)) := by
  show sumXY (pairs S) / (sumXX (pairs S) + lam * ((n + 1 : Nat) : ℝ)) = _
  simp only [sumXY_eq_sum, sumXX_eq_sum, toFn_pairs, BoundedExample.x, BoundedExample.y]

/-- Squared loss $\ell(w,(x,y))=(wx-y)^2$. -/
def sqLoss (w : ℝ) (z : BoundedExample X Y) : ℝ :=
  (w * z.x - z.y) ^ 2

end

/-! ## Generic “sum changes at one index” lemma -/

section

/--
Replacing a single element of a dataset changes a sum over the dataset by a single-term
difference.

The finite-sum perturbation identity is the combinatorial input needed to control `sumXX` and
`sumXY` under replace-one.
-/
private theorem sum_replaceAt_sub {Z : Type} (φ : Z → ℝ)
    (S : Dataset (n + 1) Z) (i : Fin (n + 1)) (z' : Z) :
    (∑ j ∈ (Finset.univ : Finset (Fin (n + 1))), φ (Dataset.toFn S j)) -
        (∑ j ∈ (Finset.univ : Finset (Fin (n + 1))), φ (Dataset.toFn (replaceAt S i z') j)) =
      φ (Dataset.toFn S i) - φ z' := by
  -- Only the summand at `i` survives the difference, since `toFn_replaceAt` is the identity
  -- elsewhere.
  rw [← Finset.sum_sub_distrib, Finset.sum_eq_single i
    (fun j _ hj => by rw [toFn_replaceAt, ite_eq_right hj, sub_self]) (by simp)]
  rw [toFn_replaceAt, ite_eq_left rfl]

end

/-! ## Ridge stability proof -/

variable {X Y lam : ℝ}

/-!
The analysis lemmas below culminate in the final uniform stability theorem. The section exposes
that theorem and keeps the intermediate algebraic bounds private to the proof.
-/

/-- The sample size `n + 1` as a real number, so the averaging denominators stay readable. -/
def sampleSize : ℝ := ((n + 1 : Nat) : ℝ)

/-- $N=n+1$ is positive as a real number. -/
theorem sampleSize_pos : 0 < sampleSize (n := n) := by
  simpa [sampleSize] using (Nat.cast_pos.mpr (Nat.succ_pos n))

/-- `sumXX` is nonnegative (it is a sum of squares). -/
private theorem sumXX_nonneg (S : Dataset (n + 1) (BoundedExample X Y)) :
    0 ≤ sumXX (pairs S) := by
  rw [sumXX_eq_sum]
  exact Finset.sum_nonneg fun _ _ => sq_nonneg _

/--
The ridge denominator $\operatorname{sumXX}(S)+\lambda N$ is positive when $\lambda>0$.

A positive denominator keeps the closed-form ratio clear of Lean's convention $x/0=0$ and lets the
proofs use order properties of division.
-/
private theorem denom_pos (hlam : 0 < lam) (S : Dataset (n + 1) (BoundedExample X Y)) :
    0 < sumXX (pairs S) + lam * sampleSize (n := n) := by
  have h1 : 0 ≤ sumXX (pairs S) := sumXX_nonneg (n := n) (X := X) (Y := Y) S
  have h2 : 0 < lam * sampleSize (n := n) := mul_pos hlam (sampleSize_pos (n := n))
  linarith

/--
Lower bound on the ridge denominator: $\lambda N\le\operatorname{sumXX}(S)+\lambda N$.

The proofs below use it to replace the dataset-dependent denominator with a uniform lower bound.
-/
private theorem denom_lower (S : Dataset (n + 1) (BoundedExample X Y)) :
    lam * sampleSize (n := n) ≤ sumXX (pairs S) + lam * sampleSize (n := n) := by
  have h1 : 0 ≤ sumXX (pairs S) := sumXX_nonneg (n := n) (X := X) (Y := Y) S
  linarith

/-- Absolute bound on the cross-term sum `sumXY`, from the bounds $|x|\le X$ and $|y|\le Y$. -/
private theorem abs_sumXY_le (S : Dataset (n + 1) (BoundedExample X Y)) :
    |sumXY (pairs S)| ≤ sampleSize (n := n) * X * Y := by
  classical
  calc
    |sumXY (pairs S)|
        ≤ ∑ i ∈ (Finset.univ : Finset (Fin (n + 1))),
            |(Dataset.toFn S i).x * (Dataset.toFn S i).y| :=
          by
            simp only [sumXY_eq_sum, toFn_pairs]
            exact Finset.abs_sum_le_sum_abs (s := (Finset.univ : Finset (Fin (n + 1))))
              (f := fun i => (Dataset.toFn S i).x * (Dataset.toFn S i).y)
    _ ≤ ∑ _i ∈ (Finset.univ : Finset (Fin (n + 1))), (X * Y) :=
          Finset.sum_le_sum fun i _ => (Dataset.toFn S i).abs_x_mul_y_le
    _ = ((n + 1 : Nat) : ℝ) * (X * Y) := by
          simp [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
    _ = sampleSize (n := n) * X * Y := by
          simp [sampleSize]
          ring_nf

/--
Replacing one example changes `sumXY` by at most $2XY$: the numerator perturbation bound for the
ridge closed form.
-/
private theorem abs_sumXY_sub_replaceAt_le (S : Dataset (n + 1) (BoundedExample X Y))
    (i : Fin (n + 1)) (z' : BoundedExample X Y) :
    |sumXY (pairs S) - sumXY (pairs (replaceAt S i z'))| ≤ 2 * X * Y := by
  classical
  have hdiff :
      sumXY (pairs S) - sumXY (pairs (replaceAt S i z')) =
        (Dataset.toFn S i).x * (Dataset.toFn S i).y - z'.x * z'.y := by
    simp only [sumXY_eq_sum, toFn_pairs]
    exact sum_replaceAt_sub (φ := fun z : BoundedExample X Y => z.x * z.y) S i z'
  calc
    |sumXY (pairs S) - sumXY (pairs (replaceAt S i z'))|
        = |(Dataset.toFn S i).x * (Dataset.toFn S i).y - z'.x * z'.y| := by rw [hdiff]
    _ ≤ |(Dataset.toFn S i).x * (Dataset.toFn S i).y| + |z'.x * z'.y| := abs_sub _ _
    _ ≤ X * Y + X * Y := add_le_add (Dataset.toFn S i).abs_x_mul_y_le z'.abs_x_mul_y_le
    _ = 2 * X * Y := by ring

/--
Replacing one example changes `sumXX` by at most $2X^2$: the denominator perturbation bound for
the ridge closed form.
-/
private theorem abs_sumXX_sub_replaceAt_le (S : Dataset (n + 1) (BoundedExample X Y))
    (i : Fin (n + 1)) (z' : BoundedExample X Y) :
    |sumXX (pairs S) - sumXX (pairs (replaceAt S i z'))| ≤ 2 * X ^ 2 := by
  classical
  have hdiff :
      sumXX (pairs S) - sumXX (pairs (replaceAt S i z')) =
        (Dataset.toFn S i).x ^ 2 - z'.x ^ 2 := by
    simp only [sumXX_eq_sum, toFn_pairs]
    exact sum_replaceAt_sub (φ := fun z : BoundedExample X Y => z.x ^ 2) S i z'
  calc
    |sumXX (pairs S) - sumXX (pairs (replaceAt S i z'))|
        = |(Dataset.toFn S i).x ^ 2 - z'.x ^ 2| := by rw [hdiff]
    _ ≤ |(Dataset.toFn S i).x ^ 2| + |z'.x ^ 2| := abs_sub _ _
    _ = (Dataset.toFn S i).x ^ 2 + z'.x ^ 2 := by simp
    _ ≤ X ^ 2 + X ^ 2 := add_le_add (Dataset.toFn S i).sq_x_le z'.sq_x_le
    _ = 2 * X ^ 2 := by ring

/--
Bound the magnitude of the fitted ridge weight.

The bound $|\widehat w(S)|\le XY/\lambda$ is coarse, and it is all that the residual bound below
needs.
-/
private theorem abs_w_le (hlam : 0 < lam) (S : Dataset (n + 1) (BoundedExample X Y)) :
    |estimator (n := n) (X := X) (Y := Y) lam S| ≤ (X * Y) / lam := by
  classical
  set D : ℝ := sumXX (pairs S) + lam * sampleSize (n := n)
  have hDpos : 0 < D := by
    simpa [D] using (denom_pos (n := n) (X := X) (Y := Y) (lam := lam) hlam S)
  have hDge : lam * sampleSize (n := n) ≤ D := by
    simpa [D] using (denom_lower (n := n) (X := X) (Y := Y) (lam := lam) S)
  have hlamNpos : 0 < lam * sampleSize (n := n) := mul_pos hlam (sampleSize_pos (n := n))
  have hB : |sumXY (pairs S)| ≤ sampleSize (n := n) * X * Y :=
    abs_sumXY_le (n := n) (X := X) (Y := Y) S
  have habsD : |D| = D := abs_of_pos hDpos
  have hfit :
      estimator (n := n) (X := X) (Y := Y) lam S = sumXY (pairs S) / D := rfl
  have hpos_num : 0 ≤ |sumXY (pairs S)| := abs_nonneg _
  calc
      |estimator (n := n) (X := X) (Y := Y) lam S|
          = |sumXY (pairs S)| / D := by
              -- `D > 0` removes `|D|` after `abs_div`.
              simp [hfit, abs_div, habsD]
    _ ≤ |sumXY (pairs S)| / (lam * sampleSize (n := n)) := by
          exact div_le_div_of_nonneg_left hpos_num hlamNpos hDge
    _ ≤ (sampleSize (n := n) * X * Y) / (lam * sampleSize (n := n)) := by
          exact div_le_div_of_nonneg_right hB (le_of_lt hlamNpos)
    _ = (X * Y) / lam := by
          have hNne : (sampleSize (n := n)) ≠ 0 := ne_of_gt (sampleSize_pos (n := n))
          field_simp [sampleSize, hNne, (ne_of_gt hlam)]

/--
Bound the residual $|\widehat w(S)x-y|$ at a test point.

The bound is coarse as well. The main proof uses it at the very end, when it bounds the loss change
through $(e-e')(e+e')$ for $e=wx-y$.
-/
private theorem abs_residual_le (hlam : 0 < lam) (S : Dataset (n + 1) (BoundedExample X Y))
    (z : BoundedExample X Y) :
    |estimator (n := n) (X := X) (Y := Y) lam S * z.x - z.y| ≤ Y * (lam + X ^ 2) / lam := by
  have hw : |estimator (n := n) (X := X) (Y := Y) lam S| ≤ (X * Y) / lam :=
    abs_w_le (n := n) (X := X) (Y := Y) (lam := lam) hlam S
  have hXYlam_nonneg : 0 ≤ (X * Y) / lam :=
    div_nonneg (mul_nonneg z.X_nonneg z.Y_nonneg) (le_of_lt hlam)
  calc
    |estimator (n := n) (X := X) (Y := Y) lam S * z.x - z.y|
        ≤ |estimator (n := n) (X := X) (Y := Y) lam S * z.x| + |z.y| := abs_sub _ _
    _ = |estimator (n := n) (X := X) (Y := Y) lam S| * |z.x| + |z.y| := by
          rw [abs_mul]
    _ ≤ ((X * Y) / lam) * X + Y :=
          add_le_add (mul_le_mul hw z.abs_x_le (abs_nonneg _) hXYlam_nonneg) z.abs_y_le
    _ = Y * (lam + X ^ 2) / lam := by
          field_simp [(ne_of_gt hlam)]
          ring

/-!
## Main theorem: deterministic replace-one uniform stability

The next theorem is the headline result. Its proof combines the parameter-sensitivity and
prediction-loss bounds established above.
-/

/--
**Uniform stability of 1D ridge regression (bounded inputs, squared loss).**

Assume $\lambda>0$. Then the ridge estimator `estimator λ` is uniformly stable in the replace-one
sense for the squared loss, with bound
$\beta=4X^2Y^2(\lambda+X^2)^2/(\lambda^3 N)$, where $N=n+1$.
Training, replacement, and test examples all carry the same bounds $|x|\le X$, $|y|\le Y$.

The stability notion used here is `UniformStableReplace` from
`TorchLean.NN.LearningTheory.Stability`.
-/
theorem uniformStableReplace_estimator (hlam : 0 < lam) :
    UniformStableReplace (Z := BoundedExample X Y) (H := ℝ)
      (A := fun S => estimator (n := n) (X := X) (Y := Y) lam S)
      (ℓ := fun w z => sqLoss (X := X) (Y := Y) w z)
      (β := 4 * X ^ 2 * Y ^ 2 * (lam + X ^ 2) ^ 2 / (lam ^ 3 * sampleSize (n := n))) := by
  classical
  intro S i z z'
  set S' : Dataset (n + 1) (BoundedExample X Y) := replaceAt S i z'
  set w : ℝ := estimator (n := n) (X := X) (Y := Y) lam S
  set w' : ℝ := estimator (n := n) (X := X) (Y := Y) lam S'
  set D : ℝ := sumXX (pairs S) + lam * sampleSize (n := n)
  set D' : ℝ := sumXX (pairs S') + lam * sampleSize (n := n)
  have hDpos : 0 < D := by
    simpa [D] using (denom_pos (n := n) (X := X) (Y := Y) (lam := lam) hlam S)
  have hD'pos : 0 < D' := by
    simpa [D'] using (denom_pos (n := n) (X := X) (Y := Y) (lam := lam) hlam S')
  have hDge : lam * sampleSize (n := n) ≤ D := by
    simpa [D] using (denom_lower (n := n) (X := X) (Y := Y) (lam := lam) S)
  have hD'ge : lam * sampleSize (n := n) ≤ D' := by
    simpa [D'] using (denom_lower (n := n) (X := X) (Y := Y) (lam := lam) S')
  have hlamNpos : 0 < lam * sampleSize (n := n) := mul_pos hlam (sampleSize_pos (n := n))
  have hBdiff :
      |sumXY (pairs S) - sumXY (pairs S')| ≤ 2 * X * Y := by
    simpa [S'] using abs_sumXY_sub_replaceAt_le (n := n) (X := X) (Y := Y) S i z'
  have hB' : |sumXY (pairs S')| ≤ sampleSize (n := n) * X * Y :=
    abs_sumXY_le (n := n) (X := X) (Y := Y) S'

  have hw_def : w = (sumXY (pairs S)) / D := rfl
  have hw'_def : w' = (sumXY (pairs S')) / D' := rfl

  -- Bound |1/D - 1/D'|
  have hA_diff :
      |sumXX (pairs S) - sumXX (pairs S')| ≤ 2 * X ^ 2 := by
    simpa [S'] using abs_sumXX_sub_replaceAt_le (n := n) (X := X) (Y := Y) S i z'
  have hInv :
      |(1 / D) - (1 / D')| ≤ (2 * X ^ 2) / (lam ^ 2 * (sampleSize (n := n)) ^ 2) := by
    have hDne : D ≠ 0 := ne_of_gt hDpos
    have hD'ne : D' ≠ 0 := ne_of_gt hD'pos
    have hInvEq : (1 / D) - (1 / D') = (D' - D) / (D * D') := by
      simpa [one_div] using (inv_sub_inv (a := D) (b := D') hDne hD'ne)
    have hposDD' : 0 < D * D' := mul_pos hDpos hD'pos
    have hDD'ge : (lam * sampleSize (n := n)) ^ 2 ≤ D * D' := by
      nlinarith [hDge, hD'ge, le_of_lt hDpos, le_of_lt hD'pos]
    have hDdiff : |D' - D| ≤ 2 * X ^ 2 := by
      have hsub :
          D' - D = sumXX (pairs S') - sumXX (pairs S) := by
        simp [D, D']
      rw [hsub, abs_sub_comm]
      exact hA_diff
    have hn : 0 ≤ 2 * X ^ 2 := by positivity
    calc
      |(1 / D) - (1 / D')|
          = |(D' - D) / (D * D')| := by
              simpa using congrArg abs hInvEq
      _ = |D' - D| / (D * D') := by simp [abs_div, abs_of_pos hposDD']
      _ ≤ (2 * X ^ 2) / (D * D') := by
            exact div_le_div_of_nonneg_right hDdiff (le_of_lt hposDD')
      _ ≤ (2 * X ^ 2) / ((lam * sampleSize (n := n)) ^ 2) := by
            exact div_le_div_of_nonneg_left hn (sq_pos_of_pos hlamNpos) hDD'ge
      _ = (2 * X ^ 2) / (lam ^ 2 * (sampleSize (n := n)) ^ 2) := by ring_nf

  -- Bound |w - w'|.
  have hw_diff :
      |w - w'| ≤ (2 * X * Y * (lam + X ^ 2)) / (lam ^ 2 * sampleSize (n := n)) := by
    have habsD : |D| = D := abs_of_pos hDpos
    have hterm1 :
        |(sumXY (pairs S) - sumXY (pairs S')) / D|
          ≤ (2 * X * Y) / (lam * sampleSize (n := n)) := by
      have h0 : 0 ≤ |sumXY (pairs S) - sumXY (pairs S')| :=
        abs_nonneg _
      have : |(sumXY (pairs S) - sumXY (pairs S')) / D|
          = |sumXY (pairs S) - sumXY (pairs S')| / D := by
            simp [abs_div, habsD]
      calc
        |(sumXY (pairs S) - sumXY (pairs S')) / D|
            = |sumXY (pairs S) - sumXY (pairs S')| / D := this
        _ ≤ |sumXY (pairs S) - sumXY (pairs S')|
              / (lam * sampleSize (n := n)) := by
              exact div_le_div_of_nonneg_left h0 hlamNpos hDge
        _ ≤ (2 * X * Y) / (lam * sampleSize (n := n)) := by
              exact div_le_div_of_nonneg_right hBdiff (le_of_lt hlamNpos)
    have hterm2 :
        |sumXY (pairs S')| * |(1 / D) - (1 / D')|
          ≤ (2 * X ^ 3 * Y) / (lam ^ 2 * sampleSize (n := n)) := by
      have hN0 : 0 ≤ sampleSize (n := n) := le_of_lt (sampleSize_pos (n := n))
      have hB0 : 0 ≤ sampleSize (n := n) * X * Y :=
        mul_nonneg (mul_nonneg hN0 z.X_nonneg) z.Y_nonneg
      have hMul :
          |sumXY (pairs S')| * |(1 / D) - (1 / D')|
            ≤ (sampleSize (n := n) * X * Y) *
              ((2 * X ^ 2) / (lam ^ 2 * (sampleSize (n := n)) ^ 2)) := by
        exact mul_le_mul hB' hInv (abs_nonneg _) hB0
      have hNne : (sampleSize (n := n)) ≠ 0 := ne_of_gt (sampleSize_pos (n := n))
      have hSimp :
          (sampleSize (n := n) * X * Y) * ((2 * X ^ 2) / (lam ^ 2 * (sampleSize (n := n)) ^ 2))
            = (2 * X ^ 3 * Y) / (lam ^ 2 * sampleSize (n := n)) := by
        field_simp [hNne]
      simpa [hSimp] using hMul
    have hsplit :
        |(sumXY (pairs S)) / D - (sumXY (pairs S')) / D'|
          ≤ (2 * X * Y) / (lam * sampleSize (n := n)) +
            (2 * X ^ 3 * Y) / (lam ^ 2 * sampleSize (n := n)) := by
      have htri :=
        abs_sub_le ((sumXY (pairs S)) / D)
          ((sumXY (pairs S')) / D)
          ((sumXY (pairs S')) / D')
      have h1 :
          |(sumXY (pairs S)) / D - (sumXY (pairs S')) / D|
            = |(sumXY (pairs S) - sumXY (pairs S')) / D| := by
              ring_nf
      have h2 :
          |(sumXY (pairs S')) / D - (sumXY (pairs S')) / D'|
            = |sumXY (pairs S')| * |(1 / D) - (1 / D')| := by
        have hEq :
            (sumXY (pairs S')) / D - (sumXY (pairs S')) / D'
              = (sumXY (pairs S')) * ((1 / D) - (1 / D')) := by
          simp [div_eq_mul_inv]
          ring_nf
        calc
          |(sumXY (pairs S')) / D - (sumXY (pairs S')) / D'|
              = |(sumXY (pairs S')) * ((1 / D) - (1 / D'))| := by
                  simp [hEq]
          _ = |sumXY (pairs S')| * |(1 / D) - (1 / D')| := by simp [abs_mul]
      have htri' :
          |(sumXY (pairs S)) / D - (sumXY (pairs S')) / D'|
            ≤ |(sumXY (pairs S) - sumXY (pairs S')) / D|
              + (|sumXY (pairs S')| * |(1 / D) - (1 / D')|) := by
              simpa [h1, h2] using htri
      nlinarith [htri', hterm1, hterm2]
    have hw_eq :
        |w - w'| =
          |(sumXY (pairs S)) / D -
            (sumXY (pairs S')) / D'| := by
      simp [w, w', hw_def, hw'_def]
    have hsimp :
        (2 * X * Y) / (lam * sampleSize (n := n)) +
        (2 * X ^ 3 * Y) / (lam ^ 2 * sampleSize (n := n))
          = (2 * X * Y * (lam + X ^ 2)) / (lam ^ 2 * sampleSize (n := n)) := by
      have hNne : (sampleSize (n := n)) ≠ 0 := ne_of_gt (sampleSize_pos (n := n))
      field_simp [hNne, (ne_of_gt hlam)]
    calc
      |w - w'|
          = |(sumXY (pairs S)) / D - (sumXY (pairs S')) / D'|
            := hw_eq
      _ ≤ (2 * X * Y) / (lam * sampleSize (n := n)) +
        (2 * X ^ 3 * Y) / (lam ^ 2 * sampleSize (n := n)) := hsplit
      _ = (2 * X * Y * (lam + X ^ 2)) / (lam ^ 2 * sampleSize (n := n)) := hsimp

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
          have hBw0 : 0 ≤ X * |w - w'| := mul_nonneg z.X_nonneg (abs_nonneg _)
          exact mul_le_mul he_diff he_sum (abs_nonneg _) hBw0
    _ ≤ (X * ((2 * X * Y * (lam + X ^ 2)) / (lam ^ 2 * sampleSize (n := n))))
          * (2 * (Y * (lam + X ^ 2) / lam)) := by
            have hC0 : 0 ≤ 2 * (Y * (lam + X ^ 2) / lam) := by
              have : 0 ≤ Y * (lam + X ^ 2) / lam :=
                div_nonneg (mul_nonneg z.Y_nonneg (add_nonneg hlam.le (sq_nonneg X)))
                  (le_of_lt hlam)
              nlinarith
            exact mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hw_diff z.X_nonneg) hC0
    _ = 4 * X ^ 2 * Y ^ 2 * (lam + X ^ 2) ^ 2 / (lam ^ 3 * sampleSize (n := n)) := by
          have hNne : (sampleSize (n := n)) ≠ 0 := ne_of_gt (sampleSize_pos (n := n))
          field_simp [hNne, (ne_of_gt hlam)]
          ring_nf

end TorchLean.Examples.Ridge

end

open FloatLib.Floats (ExecFloat Binary32)

namespace TorchLean.Examples.Ridge

open IEEE754

variable {n : Nat}

/-! ## A tensor-flavored wrapper (feature vector of length 1) -/

/-- Shape of the feature tensor in the `Vec1` packaging: a one-dimensional tensor of length $1$. -/
abbrev xShape : Shape := .dim 1 .scalar

/--
An example where the input feature is packaged as a length-$1$ tensor, together with a scalar label.

The layout is closer to typical ML “(feature vector, label)” pairs and eases reuse of tensor
utilities elsewhere in TorchLean.
-/
abbrev VectorExample : Type :=
  Tensor Binary32 xShape × Binary32

/--
The single feature coordinate (entry $0$) of a length-$1$ feature tensor.
-/
def VectorExample.x (z : VectorExample) : Binary32 :=
  z.1.getScalar ⟨0, by decide⟩

/-- Label coordinate `y` of a `VectorExample` pair. -/
@[simp] abbrev VectorExample.y (z : VectorExample) : Binary32 := z.2

/--
The binary32 ridge fit `fit` of a dataset whose inputs are length-`1` tensors, repacked into
the `(x, y)` pairs that `fit` reads.
-/
def fitVec (lam : Binary32) (S : Dataset (n + 1) VectorExample) :
  Binary32 :=
  fit lam <|
    Dataset.ofFn (n := n + 1) (Z := Binary32 × Binary32) (fun i =>
      let zi := Dataset.toFn S i
      (VectorExample.x zi, zi.y))

/-! ## The expression bridge in any configured binary format -/

section

open FloatLib.Floats.Formats.BinaryInterchange

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- The expression adding the products `x * x` of the pairs of `S` from left to right. -/
def sumXXExpr (S : Dataset (n + 1) (Value × Value)) : IEEE754.Expr Value :=
  Fin.foldl (n + 1) (fun acc i =>
    .add acc (.mul (.const (Dataset.toFn S i).1) (.const (Dataset.toFn S i).1))) (.const 0)

/-- The expression adding the products `x * y` of the pairs of `S` from left to right. -/
def sumXYExpr (S : Dataset (n + 1) (Value × Value)) : IEEE754.Expr Value :=
  Fin.foldl (n + 1) (fun acc i =>
    .add acc (.mul (.const (Dataset.toFn S i).1) (.const (Dataset.toFn S i).2))) (.const 0)

/-- The expression of the ridge fit `fit lam S`, with its operations in the same order. -/
def fitExpr (lam : Value) (S : Dataset (n + 1) (Value × Value)) : IEEE754.Expr Value :=
  .div (sumXYExpr S)
    (.add (sumXXExpr S) (.mul (.const lam) (.const ((n + 1 : Nat) : Value))))

private theorem eval_sum (env : Nat → Value) (m : Nat) (terms : Fin m → IEEE754.Expr Value)
    (initial : IEEE754.Expr Value) :
    Expr.eval env (Fin.foldl m (fun acc i => .add acc (terms i)) initial) =
      Fin.foldl m (fun acc i => acc + Expr.eval env (terms i))
        (Expr.eval env initial) := by
  induction m generalizing initial with
  | zero => simp
  | succ m ih =>
    simp only [Fin.foldl_succ]
    exact ih (fun i => terms i.succ) (.add initial (terms 0))

/-- The expression `fitExpr lam S` evaluates to the fit `fit lam S`. -/
theorem eval_fitExpr (lam : Value) (S : Dataset (n + 1) (Value × Value)) :
    Expr.eval (fun _ => 0) (fitExpr lam S) = fit lam S := by
  simp only [fitExpr, Expr.eval, sumXXExpr, sumXYExpr, eval_sum, fit, sumXX,
    sumXY, ExecFloat.add, ExecFloat.mul, ExecFloat.div]
  rfl

/-- The real value of `fitExpr lam S` when each operation computes in `ℝ` and rounds to the
format of `Value` (`Expr.evalReal`). -/
noncomputable def fitReal (lam : Value) (S : Dataset (n + 1) (Value × Value)) : ℝ :=
  Expr.evalReal (fun _ => (ExecFloat.Binary.toModel (0 : Value)).toReal) (fitExpr lam S)

/-- In an IEEE format, the real value of the fit `fit lam S` is `fitReal lam S` when
every intermediate result of `fitExpr lam S` is finite and its divisor is nonzero (`hfin`). -/
theorem toReal_fit (lam : Value) (S : Dataset (n + 1) (Value × Value))
    {d : FloatLib.Numerics.Dyadic}
    (hfin : IEEE754.FiniteEval (fun _ => 0) (fitExpr lam S) d)
    (hformat : format.isIEEE = true := by ieee_format) :
    (ExecFloat.Binary.toModel (fit lam S)).toReal = fitReal lam S := by
  have h := Expr.toReal_eval (env := fun _ => (0 : Value))
    (e := fitExpr lam S) (d := d) hfin hformat
  rw [eval_fitExpr] at h
  rw [fitReal]
  exact h

/-! ## The distance to the real fit -/

private theorem evalExact_sum (env : Nat → ℝ) (m : Nat) (terms : Fin m → IEEE754.Expr Value)
    (initial : IEEE754.Expr Value) :
    IEEE754.evalExact env (Fin.foldl m (fun acc i => .add acc (terms i)) initial) =
      Fin.foldl m (fun acc i => acc + IEEE754.evalExact env (terms i))
        (IEEE754.evalExact env initial) := by
  induction m generalizing initial with
  | zero => simp
  | succ m ih =>
    simp only [Fin.foldl_succ]
    exact ih (fun i => terms i.succ) (.add initial (terms 0))

private theorem divisorsExceedError_sum (env : Nat → ℝ) (m : Nat)
    (terms : Fin m → IEEE754.Expr Value) (initial : IEEE754.Expr Value)
    (hterms : ∀ i, IEEE754.DivisorsExceedError env (terms i))
    (hinitial : IEEE754.DivisorsExceedError env initial) :
    IEEE754.DivisorsExceedError env (Fin.foldl m (fun acc i => .add acc (terms i)) initial) := by
  induction m generalizing initial with
  | zero => simpa only [Fin.foldl_zero] using hinitial
  | succ m ih =>
    simp only [Fin.foldl_succ]
    exact ih (fun i => terms i.succ) (.add initial (terms 0)) (fun i => hterms i.succ)
      ⟨hinitial, hterms 0⟩

private theorem add_step_le {u η T t E e S c q : ℝ} (hu : 0 < u) (hT : |T| ≤ S) (ht : |t| ≤ c)
    (hE : E ≤ (q * (1 + u) - 1) * S + (2 + u) * (q - 1) / u * η) (he : e ≤ u * c + η) :
    u * |T + t| + ((1 + u) * (E + e) + η) ≤
      (q * (1 + u) * (1 + u) - 1) * S + ((1 + u) * (1 + u) - 1) * c +
        (2 + u) * (q * (1 + u) - 1) / u * η := by
  have hu0 : u ≠ 0 := hu.ne'
  have key : (2 + u) * (q * (1 + u) - 1) / u * η =
      (1 + u) * ((2 + u) * (q - 1) / u * η) + (2 + u) * η := by
    field_simp
    ring
  have h1 : u * |T + t| ≤ u * (S + c) :=
    mul_le_mul_of_nonneg_left ((abs_add_le _ _).trans (add_le_add hT ht)) hu.le
  have h2 : (1 + u) * (E + e) ≤
      (1 + u) * ((q * (1 + u) - 1) * S + (2 + u) * (q - 1) / u * η + (u * c + η)) :=
    mul_le_mul_of_nonneg_left (add_le_add hE he) (by linarith)
  rw [key]
  linarith

private theorem errorBound_sum_le (env : Nat → ℝ) (m : Nat)
    (terms : Fin m → IEEE754.Expr Value) (c : Fin m → ℝ)
    (hc : ∀ i, |IEEE754.evalExact env (terms i)| ≤ c i)
    (he : ∀ i, IEEE754.errorBound env (terms i) ≤
      Model.unitRoundoffAt format * c i + Model.underflowErrorAt format) :
    |IEEE754.evalExact env (Fin.foldl m (fun acc i => .add acc (terms i)) (.const 0))| ≤
        ∑ i, c i ∧
      IEEE754.errorBound env (Fin.foldl m (fun acc i => .add acc (terms i)) (.const 0)) ≤
        ((1 + Model.unitRoundoffAt format) ^ (m + 1) - 1) * ∑ i, c i +
          (2 + Model.unitRoundoffAt format) * ((1 + Model.unitRoundoffAt format) ^ m - 1) /
            Model.unitRoundoffAt format * Model.underflowErrorAt format := by
  have hu := Model.unitRoundoffAt_pos format
  induction m with
  | zero =>
    have h0 : (ExecFloat.Binary.toModel (0 : Value)).toReal = 0 := ExecFloat.Binary.toReal_zero
    simp only [Fin.foldl_zero, IEEE754.evalExact, IEEE754.errorBound, h0, abs_zero,
      Fin.sum_univ_zero, mul_zero, pow_zero, sub_self, zero_div, zero_mul, add_zero, le_refl,
      and_self]
  | succ m ih =>
    obtain ⟨ih1, ih2⟩ := ih (fun i => terms i.castSucc) (fun i => c i.castSucc)
      (fun i => hc i.castSucc) (fun i => he i.castSucc)
    simp only [Fin.foldl_succ_last, Fin.sum_univ_castSucc, IEEE754.evalExact,
      IEEE754.errorBound]
    have hl := hc (Fin.last m)
    have hcl : 0 ≤ c (Fin.last m) := (abs_nonneg _).trans hl
    refine ⟨(abs_add_le _ _).trans (add_le_add ih1 hl), ?_⟩
    have hq : 1 ≤ (1 + Model.unitRoundoffAt format) ^ m := one_le_pow₀ (by linarith)
    refine (add_step_le (q := (1 + Model.unitRoundoffAt format) ^ m) hu ih1 hl
      (ih2.trans_eq (by ring)) (he (Fin.last m))).trans ?_
    have e1 : (1 + Model.unitRoundoffAt format) ^ (m + 1 + 1) =
        (1 + Model.unitRoundoffAt format) ^ m * (1 + Model.unitRoundoffAt format) *
          (1 + Model.unitRoundoffAt format) := by ring
    have e2 : (1 + Model.unitRoundoffAt format) ^ (m + 1) =
        (1 + Model.unitRoundoffAt format) ^ m * (1 + Model.unitRoundoffAt format) := by ring
    rw [e1, e2]
    nlinarith [mul_nonneg (mul_nonneg (sub_nonneg.2 hq)
      (mul_self_nonneg (1 + Model.unitRoundoffAt format))) hcl]

private theorem errorBound_div_le (env : Nat → ℝ) (a b : IEEE754.Expr Value) {A EA D ED : ℝ}
    (ha : |IEEE754.evalExact env a| ≤ A) (hea : IEEE754.errorBound env a ≤ EA)
    (hb : IEEE754.evalExact env b = D) (heb : IEEE754.errorBound env b ≤ ED)
    (hED : 0 ≤ ED) (hlt : ED < |D|) (hEA : 0 ≤ EA) :
    IEEE754.errorBound env (.div a b) ≤
      Model.unitRoundoffAt format * (A / |D|) +
        ((1 + Model.unitRoundoffAt format) * ((EA + A / |D| * ED) / (|D| - ED)) +
          Model.underflowErrorAt format) := by
  have hu := Model.unitRoundoffAt_pos format
  have hD : 0 < |D| := hED.trans_lt hlt
  have hA : 0 ≤ A := (abs_nonneg _).trans ha
  simp only [IEEE754.errorBound]
  rw [hb]
  have hq : |IEEE754.evalExact env a / D| ≤ A / |D| := by
    rw [abs_div]
    exact div_le_div_of_nonneg_right ha hD.le
  have hnum : IEEE754.errorBound env a +
      |IEEE754.evalExact env a / D| * IEEE754.errorBound env b ≤ EA + A / |D| * ED :=
    add_le_add hea ((mul_le_mul_of_nonneg_left heb (abs_nonneg _)).trans
      (mul_le_mul_of_nonneg_right hq hED))
  have hfrac : (IEEE754.errorBound env a +
      |IEEE754.evalExact env a / D| * IEEE754.errorBound env b) /
        (|D| - IEEE754.errorBound env b) ≤ (EA + A / |D| * ED) / (|D| - ED) :=
    (div_le_div_of_nonneg_right hnum (by linarith)).trans
      (div_le_div_of_nonneg_left (add_nonneg hEA (mul_nonneg (div_nonneg hA hD.le) hED))
        (by linarith) (by linarith))
  have h1 := mul_le_mul_of_nonneg_left hq hu.le
  have h2 := mul_le_mul_of_nonneg_left hfrac
    (by linarith : (0 : ℝ) ≤ 1 + Model.unitRoundoffAt format)
  linarith

private theorem evalExact_sumXXExpr (S : Dataset (n + 1) (Value × Value)) :
    IEEE754.evalExact (fun _ => 0) (sumXXExpr S) =
      sumXX (Dataset.ofFn fun i => ((ExecFloat.Binary.toModel (Dataset.toFn S i).1).toReal,
        (ExecFloat.Binary.toModel (Dataset.toFn S i).2).toReal)) := by
  have h0 : (ExecFloat.Binary.toModel (0 : Value)).toReal = 0 := ExecFloat.Binary.toReal_zero
  simp only [sumXXExpr, evalExact_sum, IEEE754.evalExact, h0, sumXX, Dataset.toFn_ofFn_apply]

private theorem evalExact_sumXYExpr (S : Dataset (n + 1) (Value × Value)) :
    IEEE754.evalExact (fun _ => 0) (sumXYExpr S) =
      sumXY (Dataset.ofFn fun i => ((ExecFloat.Binary.toModel (Dataset.toFn S i).1).toReal,
        (ExecFloat.Binary.toModel (Dataset.toFn S i).2).toReal)) := by
  have h0 : (ExecFloat.Binary.toModel (0 : Value)).toReal = 0 := ExecFloat.Binary.toReal_zero
  simp only [sumXYExpr, evalExact_sum, IEEE754.evalExact, h0, sumXY, Dataset.toFn_ofFn_apply]

private theorem evalExact_ridgeExpr (lam : Value) (S : Dataset (n + 1) (Value × Value))
    (hN : (ExecFloat.Binary.toModel ((n + 1 : Nat) : Value)).toReal = ((n + 1 : Nat) : ℝ)) :
    IEEE754.evalExact (fun _ => 0) (fitExpr lam S) =
      fit (ExecFloat.Binary.toModel lam).toReal
        (Dataset.ofFn fun i => ((ExecFloat.Binary.toModel (Dataset.toFn S i).1).toReal,
          (ExecFloat.Binary.toModel (Dataset.toFn S i).2).toReal)) := by
  simp only [fitExpr, IEEE754.evalExact, evalExact_sumXXExpr, evalExact_sumXYExpr, hN]
  rfl

/-- In an IEEE format, when `fitExpr lam S` evaluates finitely (`hfin`), every divisor exceeds
its error bound (`hdiv`) and the count `n + 1` converts exactly (`hN`), the real value of the fit
`fit lam S` is within `IEEE754.errorBound (fun _ => 0) (fitExpr lam S)` of `fit`
over `ℝ` at the real values of `lam` and of the pairs of `S`.
`ExecFloat.Binary.toReal_ofRat_natCast_of_le` gives `hN` when the conversion of `n + 1` is finite
and `n + 1 ≤ 2 ^ (format.fracWidth + 1)`. -/
theorem abs_toReal_ridgeFit_sub_ridgeFit_le (lam : Value) (S : Dataset (n + 1) (Value × Value))
    {d : FloatLib.Numerics.Dyadic}
    (hfin : IEEE754.FiniteEval (fun _ => 0) (fitExpr lam S) d)
    (hdiv : IEEE754.DivisorsExceedError (fun _ => 0) (fitExpr lam S))
    (hN : (ExecFloat.Binary.toModel ((n + 1 : Nat) : Value)).toReal = ((n + 1 : Nat) : ℝ))
    (hformat : format.isIEEE = true := by ieee_format) :
    |(ExecFloat.Binary.toModel (fit lam S)).toReal -
        fit (ExecFloat.Binary.toModel lam).toReal
          (Dataset.ofFn fun i => ((ExecFloat.Binary.toModel (Dataset.toFn S i).1).toReal,
            (ExecFloat.Binary.toModel (Dataset.toFn S i).2).toReal))| ≤
      IEEE754.errorBound (fun _ => 0) (fitExpr lam S) := by
  have h0 : (ExecFloat.Binary.toModel (0 : Value)).toReal = 0 := ExecFloat.Binary.toReal_zero
  have h := IEEE754.abs_toReal_eval_sub_evalExact_le (fun _ => (0 : Value)) hfin
    (by simpa only [h0] using hdiv) hformat
  simp only [h0, eval_fitExpr] at h
  rwa [evalExact_ridgeExpr lam S hN] at h

private theorem errorBound_den_le (lam : Value) (S : Dataset (n + 1) (Value × Value))
    (hN : (ExecFloat.Binary.toModel ((n + 1 : Nat) : Value)).toReal = ((n + 1 : Nat) : ℝ)) :
    IEEE754.evalExact (fun _ => 0)
        (.add (sumXXExpr S) (.mul (.const lam) (.const ((n + 1 : Nat) : Value)))) =
      ∑ i, (ExecFloat.Binary.toModel (Dataset.toFn S i).1).toReal ^ 2 +
        (ExecFloat.Binary.toModel lam).toReal * ((n + 1 : Nat) : ℝ) ∧
    IEEE754.errorBound (fun _ => 0)
        (.add (sumXXExpr S) (.mul (.const lam) (.const ((n + 1 : Nat) : Value)))) ≤
      ((1 + Model.unitRoundoffAt format) ^ (n + 3) - 1) *
          ∑ i, (ExecFloat.Binary.toModel (Dataset.toFn S i).1).toReal ^ 2 +
        ((1 + Model.unitRoundoffAt format) ^ 2 - 1) *
          |(ExecFloat.Binary.toModel lam).toReal * ((n + 1 : Nat) : ℝ)| +
        (2 + Model.unitRoundoffAt format) * ((1 + Model.unitRoundoffAt format) ^ (n + 2) - 1) /
          Model.unitRoundoffAt format * Model.underflowErrorAt format := by
  have hu := Model.unitRoundoffAt_pos format
  have hsum := errorBound_sum_le (fun _ => 0) (n + 1)
    (fun i => .mul (.const (Dataset.toFn S i).1) (.const (Dataset.toFn S i).1))
    (fun i => (ExecFloat.Binary.toModel (Dataset.toFn S i).1).toReal ^ 2)
    (fun i => le_of_eq (by simp only [IEEE754.evalExact]; rw [abs_mul_self, sq]))
    (fun i => le_of_eq (by
      simp only [IEEE754.errorBound, IEEE754.evalExact, mul_zero, add_zero, zero_add]
      rw [abs_mul_self, sq]))
  have hB : IEEE754.evalExact (fun _ => 0) (sumXXExpr S) =
      ∑ i, (ExecFloat.Binary.toModel (Dataset.toFn S i).1).toReal ^ 2 := by
    rw [evalExact_sumXXExpr, sumXX_eq_sum]
    simp only [Dataset.toFn_ofFn_apply]
  refine ⟨by simp only [IEEE754.evalExact, hB, hN], ?_⟩
  simp only [IEEE754.errorBound, IEEE754.evalExact, mul_zero, add_zero, zero_add, hN]
  refine (add_step_le (q := (1 + Model.unitRoundoffAt format) ^ (n + 1)) hu hsum.1 le_rfl
    (hsum.2.trans_eq (by ring)) le_rfl).trans_eq ?_
  ring

/-- When the count `n + 1` converts exactly (`hN`) and the exact denominator
`|∑ x_i ^ 2 + lam * (n + 1)|` of the real values exceeds the closed-form bound on its error on the
left of `h`, every divisor of `fitExpr lam S` exceeds its error bound. -/
theorem divisorsExceedError_ridgeExpr (lam : Value) (S : Dataset (n + 1) (Value × Value))
    (hN : (ExecFloat.Binary.toModel ((n + 1 : Nat) : Value)).toReal = ((n + 1 : Nat) : ℝ))
    (h : ((1 + Model.unitRoundoffAt format) ^ (n + 3) - 1) *
          ∑ i, (ExecFloat.Binary.toModel (Dataset.toFn S i).1).toReal ^ 2 +
        ((1 + Model.unitRoundoffAt format) ^ 2 - 1) *
          |(ExecFloat.Binary.toModel lam).toReal * ((n + 1 : Nat) : ℝ)| +
        (2 + Model.unitRoundoffAt format) * ((1 + Model.unitRoundoffAt format) ^ (n + 2) - 1) /
          Model.unitRoundoffAt format * Model.underflowErrorAt format <
      |∑ i, (ExecFloat.Binary.toModel (Dataset.toFn S i).1).toReal ^ 2 +
        (ExecFloat.Binary.toModel lam).toReal * ((n + 1 : Nat) : ℝ)|) :
    IEEE754.DivisorsExceedError (fun _ => 0) (fitExpr lam S) := by
  obtain ⟨hD, hED⟩ := errorBound_den_le lam S hN
  have hXX : IEEE754.DivisorsExceedError (fun _ => 0) (sumXXExpr S) :=
    divisorsExceedError_sum _ _ _ _ (fun _ => ⟨trivial, trivial⟩) trivial
  have hXY : IEEE754.DivisorsExceedError (fun _ => 0) (sumXYExpr S) :=
    divisorsExceedError_sum _ _ _ _ (fun _ => ⟨trivial, trivial⟩) trivial
  refine ⟨hXY, ⟨hXX, trivial, trivial⟩, ?_⟩
  rw [hD]
  exact hED.trans_lt h

/-- Under `hN` and `h` of `divisorsExceedError_ridgeExpr` and `∑ |x_i * y_i| ≤ C` (`hC`), the error
bound of `fitExpr lam S` is at most
`u * (C / |D|) + ((1 + u) * ((EA + C / |D| * ED) / (|D| - ED)) + η)`. Here `D` is the exact
denominator, `ED` the left side of `h`,
`EA = ((1 + u) ^ (n + 2) - 1) * C + (2 + u) * ((1 + u) ^ (n + 1) - 1) / u * η` bounds the
numerator's error, `u = Model.unitRoundoffAt format` and `η = Model.underflowErrorAt format`. -/
theorem errorBound_ridgeExpr_le (lam : Value) (S : Dataset (n + 1) (Value × Value)) {C : ℝ}
    (hC : ∑ i, |(ExecFloat.Binary.toModel (Dataset.toFn S i).1).toReal *
      (ExecFloat.Binary.toModel (Dataset.toFn S i).2).toReal| ≤ C)
    (hN : (ExecFloat.Binary.toModel ((n + 1 : Nat) : Value)).toReal = ((n + 1 : Nat) : ℝ))
    (h : ((1 + Model.unitRoundoffAt format) ^ (n + 3) - 1) *
          ∑ i, (ExecFloat.Binary.toModel (Dataset.toFn S i).1).toReal ^ 2 +
        ((1 + Model.unitRoundoffAt format) ^ 2 - 1) *
          |(ExecFloat.Binary.toModel lam).toReal * ((n + 1 : Nat) : ℝ)| +
        (2 + Model.unitRoundoffAt format) * ((1 + Model.unitRoundoffAt format) ^ (n + 2) - 1) /
          Model.unitRoundoffAt format * Model.underflowErrorAt format <
      |∑ i, (ExecFloat.Binary.toModel (Dataset.toFn S i).1).toReal ^ 2 +
        (ExecFloat.Binary.toModel lam).toReal * ((n + 1 : Nat) : ℝ)|) :
    IEEE754.errorBound (fun _ => 0) (fitExpr lam S) ≤
      Model.unitRoundoffAt format *
          (C / |∑ i, (ExecFloat.Binary.toModel (Dataset.toFn S i).1).toReal ^ 2 +
            (ExecFloat.Binary.toModel lam).toReal * ((n + 1 : Nat) : ℝ)|) +
        ((1 + Model.unitRoundoffAt format) *
            ((((1 + Model.unitRoundoffAt format) ^ (n + 2) - 1) * C +
                (2 + Model.unitRoundoffAt format) *
                    ((1 + Model.unitRoundoffAt format) ^ (n + 1) - 1) /
                  Model.unitRoundoffAt format * Model.underflowErrorAt format +
              C / |∑ i, (ExecFloat.Binary.toModel (Dataset.toFn S i).1).toReal ^ 2 +
                  (ExecFloat.Binary.toModel lam).toReal * ((n + 1 : Nat) : ℝ)| *
                (((1 + Model.unitRoundoffAt format) ^ (n + 3) - 1) *
                    ∑ i, (ExecFloat.Binary.toModel (Dataset.toFn S i).1).toReal ^ 2 +
                  ((1 + Model.unitRoundoffAt format) ^ 2 - 1) *
                    |(ExecFloat.Binary.toModel lam).toReal * ((n + 1 : Nat) : ℝ)| +
                  (2 + Model.unitRoundoffAt format) *
                      ((1 + Model.unitRoundoffAt format) ^ (n + 2) - 1) /
                    Model.unitRoundoffAt format * Model.underflowErrorAt format)) /
              (|∑ i, (ExecFloat.Binary.toModel (Dataset.toFn S i).1).toReal ^ 2 +
                  (ExecFloat.Binary.toModel lam).toReal * ((n + 1 : Nat) : ℝ)| -
                (((1 + Model.unitRoundoffAt format) ^ (n + 3) - 1) *
                    ∑ i, (ExecFloat.Binary.toModel (Dataset.toFn S i).1).toReal ^ 2 +
                  ((1 + Model.unitRoundoffAt format) ^ 2 - 1) *
                    |(ExecFloat.Binary.toModel lam).toReal * ((n + 1 : Nat) : ℝ)| +
                  (2 + Model.unitRoundoffAt format) *
                      ((1 + Model.unitRoundoffAt format) ^ (n + 2) - 1) /
                    Model.unitRoundoffAt format * Model.underflowErrorAt format))) +
          Model.underflowErrorAt format) := by
  have hu := Model.unitRoundoffAt_pos format
  have hη := Model.underflowErrorAt_pos format
  obtain ⟨hD, hED⟩ := errorBound_den_le lam S hN
  have hsum := errorBound_sum_le (fun _ => 0) (n + 1)
    (fun i => .mul (.const (Dataset.toFn S i).1) (.const (Dataset.toFn S i).2))
    (fun i => |(ExecFloat.Binary.toModel (Dataset.toFn S i).1).toReal *
      (ExecFloat.Binary.toModel (Dataset.toFn S i).2).toReal|)
    (fun i => le_of_eq (by simp only [IEEE754.evalExact]))
    (fun i => le_of_eq (by
      simp only [IEEE754.errorBound, IEEE754.evalExact, mul_zero, add_zero, zero_add]))
  have hq : ∀ k : ℕ, 0 ≤ (1 + Model.unitRoundoffAt format) ^ k - 1 :=
    fun k => sub_nonneg.2 (one_le_pow₀ (by linarith))
  have hK : ∀ k : ℕ, 0 ≤ (2 + Model.unitRoundoffAt format) *
      ((1 + Model.unitRoundoffAt format) ^ k - 1) / Model.unitRoundoffAt format *
        Model.underflowErrorAt format :=
    fun k => mul_nonneg (div_nonneg (mul_nonneg (by linarith) (hq k)) hu.le) hη.le
  have hC0 : 0 ≤ C := (Finset.sum_nonneg fun i _ => abs_nonneg _).trans hC
  have hB0 : 0 ≤ ∑ i, (ExecFloat.Binary.toModel (Dataset.toFn S i).1).toReal ^ 2 :=
    Finset.sum_nonneg fun i _ => sq_nonneg _
  rw [fitExpr]
  exact errorBound_div_le _ _ _ (hsum.1.trans hC)
    (hsum.2.trans (add_le_add (mul_le_mul_of_nonneg_left hC (hq _)) le_rfl)) hD hED
    (add_nonneg (add_nonneg (mul_nonneg (hq _) hB0) (mul_nonneg (hq 2) (abs_nonneg _))) (hK _)) h
    (add_nonneg (mul_nonneg (hq _) hC0) (hK _))

end

/-! ## A two-point dataset -/

namespace Sample

/-- Pack a scalar `x` into a length-1 vector tensor (shape `xShape`). -/
def mkVec1 (x : Binary32) : Tensor Binary32 xShape :=
  Tensor.dim (fun _ : Fin 1 => Tensor.scalar x)

/-- The two examples $(x, y) = (1, 2)$ and $(3, 4)$, so $N=2$ (and $n=1$ in the $N=n+1$
convention). -/
def dataset : Dataset 2 VectorExample :=
  Dataset.ofFn (n := 2) (Z := VectorExample) (fun i =>
    Fin.cases
      (mkVec1 (1 : Binary32), (2 : Binary32))
      (fun _ => (mkVec1 (3 : Binary32), (4 : Binary32)))
      i)

/-- The regularization parameter $\lambda = 1$ used with `S`. -/
def lam : Binary32 := (1 : Binary32)

/-- The ridge weight that `fitVec` computes from `lam` and `S` in binary32. Every
intermediate is a small integer, so the final division, $14/12$, is the one step that rounds. -/
def wHat : Binary32 :=
  fitVec (n := 1) lam dataset

end Sample

end TorchLean.Examples.Ridge

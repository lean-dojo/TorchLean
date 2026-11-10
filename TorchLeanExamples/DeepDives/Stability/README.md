# Uniform stability of 1D ridge regression

`RidgeRegression1D.lean` (module `TorchLeanExamples.DeepDives.Stability.RidgeRegression1D`,
namespace `TorchLean.Examples.Ridge`) defines the closed-form fit of
one-dimensional ridge regression over any scalar `Context α` and proves a replace-one uniform
stability bound for it over the reals. It also relates the fit to rounding after each operation in
any configured binary format and evaluates it in binary32 on a two-point dataset.

## The fit

For a dataset `S` of `N = n + 1` pairs `(x, y)` in `α`, `Ridge.fit lam S` is
`sumXY S / (sumXX S + lam * N)`. The sums `sumXX` and `sumXY` add the products `x * x` and `x * y`
from left to right with `Fin.foldl`. Over `ℝ`, `sumXX_eq_sum` and `sumXY_eq_sum` give the
`Finset` sums.

## The real estimator

A `BoundedExample X Y` is a pair `(x, y)` with `|x| ≤ X` and `|y| ≤ Y`. `Ridge.estimator lam S` is
`Ridge.fit` over `ℝ` on the pairs of `S`, and `Ridge.estimator_eq` writes it as
`(Σ xᵢ yᵢ) / (Σ xᵢ² + lam * N)`, which minimizes `(1/N) Σ (w xᵢ - yᵢ)² + lam * w²`.
`Ridge.uniformStableReplace_estimator` proves, for `0 < lam`, that this fit is
`UniformStableReplace` (from `TorchLean.NN.LearningTheory.Stability`) for the
squared loss `sqLoss w (x, y) = (w x - y)²`, with

```text
β = 4 X² Y² (lam + X²)² / (lam³ N)
```

Replacing any one training example by another bounded example changes the loss at any bounded
test example by at most β. For fixed `X`, `Y` and `lam` the bound falls like 1/N. Its `1/lam³`
factor means it is not a uniform O(1/(lam N)) estimate as `lam` goes to 0.

## Floating-point formats

`Ridge.fit` adds in a fixed order, and in a floating-point format every multiplication, addition
and division rounds in that format. For any configured binary format, `Ridge.eval_fitExpr` in
`Examples.Ridge` shows that the expression `Ridge.fitExpr` evaluates to the fit. In an IEEE
format, `Ridge.toReal_fit` shows that the real value of the fit equals
`Ridge.fitReal`, the evaluation of `Ridge.fitExpr` over the reals with rounding to the format after
each operation. Its hypothesis `hfin` asks that every intermediate result be finite and
every divisor nonzero. `Ridge.abs_toReal_ridgeFit_sub_ridgeFit_le` bounds the distance from the
real value of the fit to `Ridge.fit` over `ℝ` at the real values of `lam` and the data by the
forward error bound `IEEE754.errorBound` of `Ridge.fitExpr`. Besides `hfin`, it asks that every
divisor exceed its error bound (`hdiv`) and that the count `n + 1` convert without rounding (`hN`).

## The two-point dataset

`Ridge.Sample` fits the examples (1, 2) and (3, 4) in binary32, each input stored as
a length-1 tensor, with `lam = 1`: `wHat = Ridge.fitVec (n := 1) lam S`. The sums 10 and 14
and the denominator 12 are exact in binary32, so `wHat` is the real fit 14/12 = 7/6 rounded once.

## Build

```bash
scripts/lake.sh build TorchLeanExamples.DeepDives.Stability.RidgeRegression1D
```

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.IEEE32.Expressions
public import NN.MLTheory.LearningTheory.Stability.Core

/-!
# 1D ridge regression under `ExecFloat.Binary 8 23`: core definitions

This module contains the **reusable definitions** used for the executable float32 ridge regression
development:

- an `(x,y)` example type over `ExecFloat.Binary 8 23`,
- fold-order-sensitive floating-point sums (`Fin.foldl`), and
- an executable ridge regression implementation together with an FP32-style expression spec and a
  bridge lemma (under a finiteness assumption).

For a higher-level overview and “why this exists”, see the umbrella module
`NN.MLTheory.LearningTheory.Stability.RidgeRegression1D.IEEE32Exec`.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)

namespace NN.MLTheory.LearningTheory.Stability.RidgeRegression1D.IEEE32Exec

open TorchLean.Floats.IEEE754

variable {n : Nat}

/-! ## Example types -/

/--
An example $(x,y)$ where both coordinates are `ExecFloat.Binary 8 23` numbers.

This mirrors the real-valued pair $(x,y)\in\mathbb{R}\times\mathbb{R}$ used in
`NN.MLTheory.LearningTheory.Stability.RidgeRegression1D.Real`.
-/
abbrev Example : Type :=
  (ExecFloat.Binary 8 23) × (ExecFloat.Binary 8 23)

/-- Feature coordinate `x` of an `Example` pair. -/
@[simp] abbrev Example.x (z : Example) : ExecFloat.Binary 8 23 := z.1
/-- Label coordinate `y` of an `Example` pair. -/
@[simp] abbrev Example.y (z : Example) : ExecFloat.Binary 8 23 := z.2

/-!
## IEEE32Exec implementation (executable)

We implement sums using `Fin.foldl` instead of `Finset.sum` because `ExecFloat.Binary 8 23` does not
satisfy
the commutative-monoid laws required by `Finset.sum` (NaN payload propagation breaks algebraic
equalities).

Even if exceptional values never occur, evaluation order still matters for floats due to rounding.
-/

/-- Executable sum $\sum_i x_i^2$ (with IEEE-754 rounding after every multiplication and
addition). -/
def sumXX (S : Dataset (n + 1) Example) : ExecFloat.Binary 8 23 :=
  Fin.foldl (n + 1) (fun acc i => acc + (Dataset.get S i).x * (Dataset.get S i).x) 0

/-- Executable sum $\sum_i x_i y_i$ (with IEEE-754 rounding after every multiplication and
addition). -/
def sumXY (S : Dataset (n + 1) Example) : ExecFloat.Binary 8 23 :=
  Fin.foldl (n + 1) (fun acc i => acc + (Dataset.get S i).x * (Dataset.get S i).y) 0

/--
Executable ridge regression (1D) using the fold-based sums.

This is the direct “what we would run” implementation (subject to IEEE-754 behavior).
-/
def ridgeFit1DExec (lam : ExecFloat.Binary 8 23) (S : Dataset (n + 1) Example) :
  ExecFloat.Binary 8 23 :=
  let N : ExecFloat.Binary 8 23 := OfNat.ofNat (n + 1)
  (sumXY (n := n) S) / (sumXX (n := n) S + lam * N)

/-! ## A tensor-flavored wrapper (feature vector of length 1) -/

/-- Shape of the feature tensor in the `Vec1` packaging: a one-dimensional tensor of length $1$. -/
abbrev XShape : Spec.Shape := .dim 1 .scalar

/--
An example where the input feature is packaged as a length-$1$ tensor, together with a scalar label.

This is closer to typical ML “(feature vector, label)” layouts and makes it easier to reuse tensor
utilities elsewhere in TorchLean.
-/
abbrev VectorExample : Type :=
  TorchLean.Tensor (ExecFloat.Binary 8 23) XShape × (ExecFloat.Binary 8 23)

/--
Extract the single feature coordinate (entry $0$) from a length-$1$ feature tensor.
-/
def VectorExample.x0 (z : VectorExample) : ExecFloat.Binary 8 23 :=
  z.1.getScalar ⟨0, by decide⟩

/-- Label coordinate `y` of an `VectorExample` pair. -/
@[simp] abbrev VectorExample.y (z : VectorExample) : ExecFloat.Binary 8 23 := z.2

/--
Ridge regression where the dataset stores inputs as length-`1` tensors.

This is just a packaging conversion into the scalar-pair dataset expected by `ridgeFit1DExec`.
-/
def ridgeFit1DExecVec1 (lam : ExecFloat.Binary 8 23) (S : Dataset (n + 1) VectorExample) :
  ExecFloat.Binary 8 23 :=
  ridgeFit1DExec (n := n) lam <|
    Dataset.ofFn (n := n + 1) (Z := Example) (fun i =>
      let zi := Dataset.get S i
      (VectorExample.x0 zi, zi.y))

/-! ## FP32 (“round-after-each-primitive”) spec via the existing expression bridge -/

namespace RidgeIEEEBridge

/-- Expression for $\sum_i x_i^2$ over the dataset. -/
def sumXXExpr (S : Dataset (n + 1) Example) : IEEE32Exec.Expr :=
  Fin.foldl (n + 1) (fun acc i =>
    .add acc (.mul (.const (Dataset.get S i).x) (.const (Dataset.get S i).x)))
    (.const (0 : ExecFloat.Binary 8 23))

/-- Expression for $\sum_i x_i y_i$ over the dataset. -/
def sumXYExpr (S : Dataset (n + 1) Example) : IEEE32Exec.Expr :=
  Fin.foldl (n + 1) (fun acc i =>
    .add acc (.mul (.const (Dataset.get S i).x) (.const (Dataset.get S i).y)))
    (.const (0 : ExecFloat.Binary 8 23))

/--
Closed expression computing the ridge-regression slope
$\beta=(\sum_i x_i y_i)/(\sum_i x_i^2+\lambda N)$.
-/
def ridgeExpr (lam : ExecFloat.Binary 8 23) (S : Dataset (n + 1) Example) : IEEE32Exec.Expr :=
  let N : ExecFloat.Binary 8 23 := OfNat.ofNat (n + 1)
  .div
    (sumXYExpr (n := n) S)
    (.add (sumXXExpr (n := n) S) (.mul (.const lam) (.const N)))

private theorem eval_sum (env : Nat → ExecFloat.Binary 8 23) (m : Nat)
    (terms : Fin m → IEEE32Exec.Expr) (initial : IEEE32Exec.Expr) :
    IEEE32Exec.evalRuntime env (Fin.foldl m (fun acc i => .add acc (terms i)) initial) =
      Fin.foldl m (fun acc i => acc + IEEE32Exec.evalRuntime env (terms i))
        (IEEE32Exec.evalRuntime env initial) := by
  induction m generalizing initial with
  | zero => simp
  | succ m ih =>
    simp only [Fin.foldl_succ]
    exact ih (fun i => terms i.succ) (.add initial (terms 0))

/-- The expression evaluates to the direct fit with the same ordered binary32 operations. -/
theorem eval_ridgeExpr (lam : ExecFloat.Binary 8 23) (S : Dataset (n + 1) Example) :
    IEEE32Exec.evalRuntime (fun _ => 0) (ridgeExpr lam S) = ridgeFit1DExec lam S := by
  simp only [ridgeExpr, IEEE32Exec.evalRuntime, sumXXExpr, sumXYExpr, eval_sum,
    ridgeFit1DExec, sumXX, sumXY, ExecFloat.add, ExecFloat.mul, ExecFloat.div]
  rfl

/--
Evaluate `ridgeExpr` using the FP32-style spec semantics.

This returns a real number that corresponds to interpreting each float primitive as:

1. compute in $\mathbb{R}$,
2. round to float32, then
3. coerce back to $\mathbb{R}$ via `toReal`.
-/
noncomputable def ridgeFitSpec (lam : ExecFloat.Binary 8 23) (S : Dataset (n + 1) Example) : ℝ :=
  IEEE32Exec.evalSpec (fun _ => (ExecFloat.Binary.toModel (0 : ExecFloat.Binary 8 23)).toReal)
    (ridgeExpr (n := n) lam S)

/--
Bridge lemma: `toReal` of the executable IEEE evaluator agrees with the FP32-expression semantics,
provided evaluation stays finite (no NaN/Inf/div-by-zero along the way).
-/
theorem ridgeFit_toReal_eq_spec_of_finiteEval
    (lam : ExecFloat.Binary 8 23) (S : Dataset (n + 1) Example)
    {d : FloatLib.Numerics.Dyadic}
    (hfin : IEEE32Exec.FiniteEval (fun _ => 0) (ridgeExpr (n := n) lam S) d) :
    (ExecFloat.Binary.toModel (ridgeFit1DExec (n := n) lam S)).toReal =
      ridgeFitSpec (n := n) lam S := by
  simpa only [ridgeFitSpec, eval_ridgeExpr] using
    (IEEE32Exec.toReal_evalRuntime_eq_evalSpec (env := fun _ => (0 : ExecFloat.Binary 8 23))
      (e := ridgeExpr (n := n) lam S) (d := d) hfin)

end RidgeIEEEBridge

end NN.MLTheory.LearningTheory.Stability.RidgeRegression1D.IEEE32Exec

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.BridgeFP32Expr
public import NN.MLTheory.LearningTheory.Stability.Core

/-!
# 1D ridge regression under `IEEE32Exec`: core definitions

This module contains the **reusable definitions** used for the executable float32 ridge regression
development:

- an `(x,y)` example type over `IEEE32Exec`,
- fold-order-sensitive floating-point sums (`Fin.foldl`), and
- an executable ridge regression implementation together with an FP32-style expression spec and a
  bridge lemma (under a finiteness assumption).

For a higher-level overview and “why this exists”, see the umbrella module
`NN.MLTheory.LearningTheory.Stability.RidgeRegression1D.IEEE32Exec`.
-/

@[expose] public section


noncomputable section

open scoped BigOperators

namespace NN.MLTheory.LearningTheory.Stability.RidgeRegression1D.IEEE32Exec

open TorchLean.Floats
open TorchLean.Floats.IEEE754

variable {n : Nat}

/-! ## Example types -/

/--
An example `(x,y)` where both coordinates are `IEEE32Exec` numbers.

This mirrors the real-valued pair `(x,y) : ℝ×ℝ` used in
`NN.MLTheory.LearningTheory.Stability.RidgeRegression1D.Real`.
-/
abbrev ExampleIEEE32 : Type :=
  IEEE32Exec × IEEE32Exec

/-- Feature coordinate `x` of an `ExampleIEEE32` pair. -/
@[simp] abbrev ExampleIEEE32.x (z : ExampleIEEE32) : IEEE32Exec := z.1
/-- Label coordinate `y` of an `ExampleIEEE32` pair. -/
@[simp] abbrev ExampleIEEE32.y (z : ExampleIEEE32) : IEEE32Exec := z.2

/-!
## IEEE32Exec implementation (executable)

We implement sums using `Fin.foldl` instead of `Finset.sum` because `IEEE32Exec` does not satisfy
the commutative-monoid laws required by `Finset.sum` (NaN payload propagation breaks algebraic
equalities).

Even if exceptional values never occur, evaluation order still matters for floats due to rounding.
-/

/-- Sum `f 0 + f 1 + ... + f (m-1)` using a left fold (order matters for floats). -/
def sumFin (m : Nat) (f : Fin m → IEEE32Exec) : IEEE32Exec :=
  Fin.foldl m (fun acc i => acc + f i) 0

/-- Executable sum `∑ xᵢ²` (with IEEE-754 rounding at each multiplication/addition). -/
def sumXX (S : Dataset (n + 1) ExampleIEEE32) : IEEE32Exec :=
  sumFin (n + 1) (fun i => (Dataset.get S i).x * (Dataset.get S i).x)

/-- Executable sum `∑ xᵢ yᵢ` (with IEEE-754 rounding at each multiplication/addition). -/
def sumXY (S : Dataset (n + 1) ExampleIEEE32) : IEEE32Exec :=
  sumFin (n + 1) (fun i => (Dataset.get S i).x * (Dataset.get S i).y)

/--
Executable ridge regression (1D) using the fold-based sums.

This is the direct “what we would run” implementation (subject to IEEE-754 behavior).
-/
def ridgeFit1DExec (lam : IEEE32Exec) (S : Dataset (n + 1) ExampleIEEE32) : IEEE32Exec :=
  let N : IEEE32Exec := (n + 1 : Nat)
  (sumXY (n := n) S) / (sumXX (n := n) S + lam * N)

/-! ## A tensor-flavored wrapper (feature vector of length 1) -/

/-- Shape of the feature tensor in the `Vec1` packaging: a 1D tensor of length `1`. -/
abbrev XShape : Spec.Shape := .dim 1 .scalar

/--
An example where the input feature is packaged as a length-`1` tensor, together with a scalar label.

This is closer to typical ML “(feature vector, label)” layouts and makes it easier to reuse tensor
utilities elsewhere in TorchLean.
-/
abbrev ExampleIEEE32Vec1 : Type :=
  Spec.Tensor IEEE32Exec XShape × IEEE32Exec

/--
Extract the single feature coordinate (the `0`-th entry) from a length-`1` feature tensor.
-/
def ExampleIEEE32Vec1.x0 (z : ExampleIEEE32Vec1) : IEEE32Exec :=
  match z.1 with
  | .dim f =>
      (f ⟨0, by decide⟩).toScalar

/-- Label coordinate `y` of an `ExampleIEEE32Vec1` pair. -/
@[simp] abbrev ExampleIEEE32Vec1.y (z : ExampleIEEE32Vec1) : IEEE32Exec := z.2

/--
Ridge regression where the dataset stores inputs as length-`1` tensors.

This is just a packaging conversion into the scalar-pair dataset expected by `ridgeFit1D_exec`.
-/
def ridgeFit1DExecVec1 (lam : IEEE32Exec) (S : Dataset (n + 1) ExampleIEEE32Vec1) : IEEE32Exec :=
  ridgeFit1DExec (n := n) lam <|
    Dataset.ofFn (n := n + 1) (Z := ExampleIEEE32) (fun i =>
      let zi := Dataset.get S i
      (ExampleIEEE32Vec1.x0 zi, zi.y))

/-! ## FP32 (“round-after-each-primitive”) spec via the existing expression bridge -/

namespace RidgeIEEEBridge

open IEEE32Exec

/-- Expression for the term `x*x` for a single example. -/
def termXXExpr (z : ExampleIEEE32) : IEEE32Exec.Expr :=
  .mul (.const z.x) (.const z.x)

/-- Expression for the term `x*y` for a single example. -/
def termXYExpr (z : ExampleIEEE32) : IEEE32Exec.Expr :=
  .mul (.const z.x) (.const z.y)

/-- Expression for `∑ xᵢ^2` over the dataset. -/
def sumXXExpr (S : Dataset (n + 1) ExampleIEEE32) : IEEE32Exec.Expr :=
  Fin.foldl (n + 1) (fun acc i => .add acc (termXXExpr (Dataset.get S i))) (.const (0 : IEEE32Exec))

/-- Expression for `∑ xᵢ*yᵢ` over the dataset. -/
def sumXYExpr (S : Dataset (n + 1) ExampleIEEE32) : IEEE32Exec.Expr :=
  Fin.foldl (n + 1) (fun acc i => .add acc (termXYExpr (Dataset.get S i))) (.const (0 : IEEE32Exec))

/-- Closed expression computing the ridge-regression slope `β = (∑ xᵢ yᵢ) / (∑ xᵢ^2 + λ N)`. -/
def ridgeExpr (lam : IEEE32Exec) (S : Dataset (n + 1) ExampleIEEE32) : IEEE32Exec.Expr :=
  let N : IEEE32Exec := (n + 1 : Nat)
  .div
    (sumXYExpr (n := n) S)
    (.add (sumXXExpr (n := n) S) (.mul (.const lam) (.const N)))

/--
Execute `ridgeExpr` using the bit-level IEEE runtime evaluator.

We use the constant environment `fun _ => 0` because `ridgeExpr` is closed (it contains no
variables).
-/
def ridgeFit1DExecExpr (lam : IEEE32Exec) (S : Dataset (n + 1) ExampleIEEE32) : IEEE32Exec :=
  IEEE32Exec.evalRuntime (fun _ => 0) (ridgeExpr (n := n) lam S)

/--
Evaluate `ridgeExpr` using the FP32-style spec semantics.

This returns a real number that corresponds to interpreting each float primitive as:

1. compute in `ℝ`,
2. round to float32, then
3. coerce back to `ℝ` via `toReal`.
-/
def ridgeFit1DFp32Spec (lam : IEEE32Exec) (S : Dataset (n + 1) ExampleIEEE32) : ℝ :=
  IEEE32Exec.evalSpec (fun _ => IEEE32Exec.toReal (0 : IEEE32Exec)) (ridgeExpr (n := n) lam S)

/--
Bridge lemma: `toReal` of the executable IEEE evaluator agrees with the FP32-expression semantics,
provided evaluation stays finite (no NaN/Inf/div-by-zero along the way).
-/
theorem ridgeFit1D_execExpr_toReal_eq_fp32Spec_of_finiteEval
    (lam : IEEE32Exec) (S : Dataset (n + 1) ExampleIEEE32)
    {d : IEEE32Exec.Dyadic}
    (hfin : IEEE32Exec.FiniteEval (fun _ => 0) (ridgeExpr (n := n) lam S) d) :
    IEEE32Exec.toReal (ridgeFit1DExecExpr (n := n) lam S) = ridgeFit1DFp32Spec (n := n) lam S :=
      by
  simpa [ridgeFit1DExecExpr, ridgeFit1DFp32Spec] using
    (IEEE32Exec.toReal_evalRuntime_eq_evalSpec (env := fun _ => (0 : IEEE32Exec))
      (e := ridgeExpr (n := n) lam S) (d := d) hfin)

end RidgeIEEEBridge

end NN.MLTheory.LearningTheory.Stability.RidgeRegression1D.IEEE32Exec

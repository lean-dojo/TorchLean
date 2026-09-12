/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public meta import Aesop.BuiltinRules
public import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic.NormNum.Inv
import Mathlib.Tactic.NormNum.Pow
import Mathlib.Tactic.Positivity.Finset
meta import Mathlib.Tactic.Simps.Basic
meta import Mathlib.Tactic.ToAdditive
public import NN.Tensor.Internal.Check.Transform
public import NN.Tensor.Internal.Laws.Equivalence.Index
public meta import NN.Tensor.Internal.Elab.Common -- shake: keep
public import NN.Tensor.Internal.Laws.Equivalence -- shake: keep

/-!
# Certified row-major index specialization

Checked operations express source indices through compact row-major programs.
This module partially evaluates those programs while elaborating a concrete
operation and returns an equality certificate for the specialized arithmetic.
-/

public meta section

namespace TorchLean.Tensor.Internal.Elab.Impl

open Lean
open Lean.Meta

/--
Transport one loop-index bound from a checked shape to its concrete native
counter bound before arithmetic certification.
-/
def nativeLoopIndexBound
    (index hNativeBound hIndexBound : Expr) : MetaM Expr := do
  let nativeBound ← inferType hNativeBound >>= fun type => do
    unless type.isAppOfArity ``Eq 3 do
      throwError "internal error: a native loop bound is not an equality"
    pure type.getAppArgs[1]!
  let boundPredicate ←
    withLocalDeclD `bound (mkConst ``Nat) fun bound => do
      mkLambdaFVars #[bound] (← mkLT index bound)
  let hBoundPredicate ←
    mkAppM ``congrArg #[boundPredicate, hNativeBound]
  let nativeIndexBound ←
    mkAppM ``Eq.mpr #[hBoundPredicate, hIndexBound]
  withTransparency .all <|
    mkExpectedTypeHint nativeIndexBound (← mkLT index nativeBound)

/--
Build the shared simplifier for compact row-major index programs.

Callers may provide operation-specific definitions such as the compact
reduction index. Coordinate semantics and proof fields remain opaque.
-/
def rowMajorIndexSimpContext
    (extraDeclarations : Array Name := #[]) : MetaM Simp.Context := do
  let mut simpTheorems ← getSimpTheorems
  let declarations := #[
      ``rearrangeLinearIndex,
      ``Rearrangement.Impl.rowMajorCoordinates,
      ``Rearrangement.Impl.rowMajorIndex,
      ``Check.TransformPlan.normalized,
      ``Check.NormalizedTransform.inputAxes,
      ``Check.NormalizedTransform.outputAxes,
      ``Check.PartialAxisLengths.seed,
      ``Check.PartialAxisLengths.set,
      ``Check.SupplementaryLengths.lookup?] ++ extraDeclarations
  for declaration in declarations do
    if let some equations ← getEqnsFor? declaration then
      for equation in equations do
        simpTheorems ← simpTheorems.addConst equation
    else
      simpTheorems := simpTheorems.addDeclToUnfoldCore declaration
  Simp.mkContext
    (config := { zeta := true, failIfUnchanged := false })
    (simpTheorems := #[simpTheorems])
    (congrTheorems := ← getSimpCongrTheorems)

/--
Partially evaluate a compact row-major index.

The returned proof is oriented from the specialized expression to the
original program, ready to compose with its semantic correctness theorem.
-/
def compileRowMajorIndex
    (index : Expr) (extraDeclarations : Array Name := #[]) :
    MetaM (Expr × Expr) := do
  let unfolded ← withTransparency .all <| whnf index
  let hUnfolded ← withTransparency .all <|
    mkExpectedTypeHint (← mkEqRefl unfolded) (← mkEq unfolded index)
  let context ← rowMajorIndexSimpContext extraDeclarations
  let (simplified, _) ← simp unfolded context #[← Simp.getSimprocs]
  let hUnfoldedSimplified ← simplified.getProof' unfolded
  let hSimplifiedUnfolded ← mkAppM ``Eq.symm #[hUnfoldedSimplified]
  let hSimplifiedIndex ←
    mkAppM ``Eq.trans #[hSimplifiedUnfolded, hUnfolded]
  return (simplified.expr, hSimplifiedIndex)

end TorchLean.Tensor.Internal.Elab.Impl

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.Lowering.Reduce.Index
public meta import TorchLean.Tensor.Einops.Elab.Index
public meta import TorchLean.Tensor.Einops.Elab.Einsum.Index -- shake: keep
public import TorchLean.Tensor.Einops.Elab.Index

/-!
# Certified native reduction indices

Concrete reductions decode their output and fiber counters separately, and
`compileReductionLogicalIndex` compiles the two counters into the source
tensor's row-major index without the extra combined-coordinate decode of the
compact semantic law.
-/

public meta section

namespace TorchLean.Einops.Elab

open Lean
open Lean.Meta
open Lean.Elab.Term

/--
Compile the logical input index of a concrete reduction.

The executable expression is the partial evaluation of the general separated
row-major index. The returned equality composes with the checked reduction law,
so no generated arithmetic proof depends on a tensor's rank or concrete
dimensions.
-/
def compileReductionLogicalIndex
    (checked : Expr)
    (outputFin fiberFin outputValue fiberValue : Expr) :
    TermElabM (Expr × Expr × Expr) := do
  let value ← mkAppM ``Transform.value #[checked]
  let normalized ← mkAppM ``TransformPlan.normalized #[value]
  let axisLength ← mkAppM ``TransformPlan.axisLength #[value]
  let inputAxes ←
    mkAppM ``NormalForm.inputAxes #[normalized]
  let outputAxes ←
    mkAppM ``NormalForm.outputAxes #[normalized]
  let reducedAxes ←
    mkAppM ``Transform.reducedAxes #[checked]
  let separatedValue ←
    mkAppM ``Lowering.Reduction.separatedIndex #[
      axisLength, inputAxes, outputAxes, reducedAxes,
      outputValue, fiberValue]
  let (compiledValue, hCompiledSeparated) ←
    compileRowMajorIndex separatedValue
      #[``Lowering.Reduction.separatedIndex,
        ``Transform.reducedAxes]
  let hSource ←
    mkAppM ``Lowering.Reduction.inputAxes_subset_append #[
      checked]
  let hSeparatedCompact ←
    mkAppM ``Lowering.Reduction.separatedIndex_eq #[
      axisLength, inputAxes, outputAxes, reducedAxes,
      hSource, outputFin, fiberFin]
  let hCompiledCompact ←
    mkAppM ``Eq.trans #[hCompiledSeparated, hSeparatedCompact]
  let outputCoordinate ← mkAppM ``Shape.Coord.unlinearize #[outputFin]
  let semanticIndex ←
    mkAppM ``Lowering.Reduction.inputFlatIndex #[
      checked, outputCoordinate, fiberFin]
  let hSemanticCompact ←
    mkAppM ``Lowering.Reduction.inputFlatIndex_val #[
      checked, outputFin, fiberFin]
  let hCompactSemantic ← mkAppM ``Eq.symm #[hSemanticCompact]
  let hCompiledValue ←
    mkAppM ``Eq.trans #[hCompiledCompact, hCompactSemantic]
  return (compiledValue, semanticIndex, hCompiledValue)

end TorchLean.Einops.Elab

/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Laws.ReductionIndex
public meta import NN.Tensor.Internal.Elab.Native.Index
public meta import NN.Tensor.Internal.Elab.Einsum.Index -- shake: keep
public import NN.Tensor.Internal.Elab.Native.Index

/-!
# Certified native reduction indices

Concrete reductions decode their output and fiber counters separately. This
module compiles those counters directly into the source tensor's row-major
index, avoiding the extra combined-coordinate decode used by the compact
semantic law.
-/

public meta section

namespace TorchLean.Tensor.Internal.Elab.Impl

open Lean
open Lean.Meta
open Lean.Elab.Term

/--
Compile the logical input index of a concrete reduction.

The executable expression is the partial evaluation of the general separated
row-major index. Its theorem composes with the checked reduction law, so no
generated arithmetic proof depends on a tensor's rank or concrete dimensions.
-/
def compileReductionLogicalIndex
    (checked : Expr) (_checkedValue : Check.CheckedTransform)
    (outputFin fiberFin outputValue fiberValue : Expr) :
    TermElabM (Expr × Expr × Expr) := do
  let value ← mkAppM ``Check.CheckedTransform.value #[checked]
  let normalized ← mkAppM ``Check.TransformPlan.normalized #[value]
  let axisLength ← mkAppM ``Check.TransformPlan.axisLength #[value]
  let inputAxes ←
    mkAppM ``Check.NormalizedTransform.inputAxes #[normalized]
  let outputAxes ←
    mkAppM ``Check.NormalizedTransform.outputAxes #[normalized]
  let reducedAxes ←
    mkAppM ``Check.CheckedTransform.reducedAxes #[checked]
  let separatedValue ←
    mkAppM ``Lowering.Reduce.Impl.separatedRearrangeLinearIndex #[
      axisLength, inputAxes, outputAxes, reducedAxes,
      outputValue, fiberValue]
  let (compiledValue, hCompiledSeparated) ←
    compileRowMajorIndex separatedValue
      #[``Lowering.Reduce.Impl.separatedRearrangeLinearIndex,
        ``Check.CheckedTransform.reducedAxes]
  let hSource ←
    mkAppM ``Lowering.Reduce.Impl.inputAxes_subset_output_append_reduced #[
      checked]
  let hSeparatedCompact ←
    mkAppM ``Lowering.Reduce.Impl.separatedRearrangeLinearIndex_eq #[
      axisLength, inputAxes, outputAxes, reducedAxes,
      hSource, outputFin, fiberFin]
  let hCompiledCompact ←
    mkAppM ``Eq.trans #[hCompiledSeparated, hSeparatedCompact]
  let outputCoordinate ← mkAppM ``Coord.unlinearize #[outputFin]
  let semanticIndex ←
    mkAppM ``Lowering.Reduce.Impl.reductionInputFlatIndex #[
      checked, outputCoordinate, fiberFin]
  let hSemanticCompact ←
    mkAppM ``Lowering.Reduce.Impl.reductionInputFlatIndex_val #[
      checked, outputFin, fiberFin]
  let hCompactSemantic ← mkAppM ``Eq.symm #[hSemanticCompact]
  let hCompiledValue ←
    mkAppM ``Eq.trans #[hCompiledCompact, hCompactSemantic]
  return (compiledValue, semanticIndex, hCompiledValue)

end TorchLean.Tensor.Internal.Elab.Impl

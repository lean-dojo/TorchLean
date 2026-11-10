/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public meta import TorchLean.Tensor.Einops.Elab.Einsum
public meta import TorchLean.Tensor.Einops.Elab.Pack
public meta import TorchLean.Tensor.Literal
public meta import TorchLean.Tensor.Einops.Elab.Transform
public import Mathlib.LinearAlgebra.Matrix.Trace
public import TorchLean.Tensor.Literal
public import Std.Tactic.BVDecide.Normalize.Prop
import TorchLean.Tensor.Einops.Elab.Syntax
public import Mathlib.Algebra.Order.Algebra
public import Mathlib.Analysis.SpecialFunctions.Pow.Real
public import TorchLean.Tensor.Einops.Elab.Einsum.Index
public import TorchLean.Tensor.Einops.Elab.Einsum.Symbolic
public import TorchLean.Tensor.Native.Tile
public import TorchLean.Tensor.Einops.Equivalence.Lowering
public import TorchLean.Tensor.Axes
public import TorchLean.Tensor.LinearAlgebra.Real

/-!
# Mathlib matrix correspondence

A rank-two tensor has coordinates
`Fin rows × (Fin columns × PUnit)`, while a mathlib matrix is a curried
function `Fin rows → Fin columns → α`. `Rep.matrixEquiv`, defined in
`TorchLean.Tensor.LinearAlgebra.Real`, identifies these
representations without changing entry order.

The theorems connect symbolic einops expressions for transpose, matrix product, trace and row
and column sums to the corresponding mathlib matrix operations. Their dimensions are arbitrary
natural numbers, including zero.
-/

@[expose] public section

namespace TorchLean.Tensor.Rep

universe u

section

open Shape (Coord)
open Einops (AxisTuple)

variable {α : Type u} [Storage α] {rows columns : Nat}

/--
Swapping the two axes of a rank-two tensor with `rearrange` is mathlib's `Matrix.transpose`.
-/
theorem matrixEquiv_rearrange_transpose
    (tensor : Rep α [rows, columns]) :
    matrixEquiv α columns rows
        (rearrange tensor "row column -> column row") =
      Matrix.transpose (matrixEquiv α rows columns tensor) := by
  have linearizeMatrixCoordinate
      {outer inner : Nat} (outerCoordinate : Fin outer)
      (innerCoordinate : Fin inner) :
      (Coord.linearize (s := [outer, inner])
        (outerCoordinate, innerCoordinate, PUnit.unit)).val =
        innerCoordinate.val + inner * outerCoordinate.val := by
    have outerStep :=
      Coord.linearize_cons_val (s := [inner]) outerCoordinate
        (innerCoordinate, PUnit.unit)
    have innerStep :=
      Coord.linearize_cons_val (s := []) innerCoordinate PUnit.unit
    calc
      (Coord.linearize (s := [outer, inner])
          (outerCoordinate, innerCoordinate, PUnit.unit)).val =
          (Coord.linearize (s := [inner])
              (innerCoordinate, PUnit.unit)).val +
            inner * outerCoordinate.val := by
              simpa only [Shape.size_cons, Shape.size_nil, Nat.mul_one] using outerStep
      _ =
          ((Coord.linearize (s := []) PUnit.unit).val +
              innerCoordinate.val) +
            inner * outerCoordinate.val := by
              exact congrArg (fun value =>
                value + inner * outerCoordinate.val) <| by
                  simpa only [Shape.size_nil, Nat.one_mul] using innerStep
      _ = innerCoordinate.val + inner * outerCoordinate.val := by
        have scalarIndexBound :
            (Coord.linearize (s := []) PUnit.unit).val < 1 :=
          (Coord.linearize (s := []) PUnit.unit).isLt
        omega
  ext column row
  change
    (rearrange tensor "row column -> column row")
        (column, row, PUnit.unit) =
      tensor (row, column, PUnit.unit)
  apply Einops.Lowering.rearrange_apply_of_linearIndex
  simp (config := { zeta := true }) only [Einops.PartialAxisLengths.set,
    Einops.PartialAxisLengths.seed, List.lookup_nil, Einops.NormalForm.inputAxes,
    Einops.TransformPlan.normalized, List.flatten_cons, List.flatten_nil,
    List.append_nil, List.cons_append, List.nil_append,
    Einops.NormalForm.outputAxes, Shape.size_cons, Shape.size_nil]
  have outputLinearize :
      (Coord.linearize (s := [columns, rows])
        (column, row, PUnit.unit)).val =
        row.val + rows * column.val :=
    linearizeMatrixCoordinate column row
  have inputLinearize :
      (Coord.linearize (s := [rows, columns])
        (row, column, PUnit.unit)).val =
        column.val + columns * row.val :=
    linearizeMatrixCoordinate row column
  rw [outputLinearize, inputLinearize]
  apply Einops.Rearrangement.linearIndex_swap
  · decide
  · exact row.isLt

variable {R : Type u} [Storage R] [AddCommMonoid R] [MulOneClass R]
variable {contracted : Nat}

/--
The two-operand einsum contraction `row contracted, contracted column -> row column` is mathlib
matrix multiplication, for arbitrary natural-number dimensions.
-/
theorem matrixEquiv_einsum_mul
    (leftTensor : Rep R [rows, contracted])
    (rightTensor : Rep R [contracted, columns]) :
    matrixEquiv R rows columns
        (einsum leftTensor, rightTensor
          "row contracted, contracted column -> row column") =
      matrixEquiv R rows contracted leftTensor *
        matrixEquiv R contracted columns rightTensor := by
  rw [Einops.Lowering.einsumKernel_correct]
  ext row column
  simp only [matrixEquiv_apply, Matrix.mul_apply]
  let contractedCoordinateEquiv :
      AxisTuple
          (fun axis =>
            if axis = Einops.EinsumAxis.named "row" then rows
            else if axis = Einops.EinsumAxis.named "contracted" then contracted
            else if axis = Einops.EinsumAxis.named "column" then columns
            else 1)
          [Einops.EinsumAxis.named "contracted"] ≃
        Fin contracted :=
    (Equiv.piCongrRight fun index : Fin 1 => by
      have hIndex : index = 0 := Subsingleton.elim _ _
      subst index
      apply finCongr
      simp [
        show Einops.EinsumAxis.named "contracted" ≠
            Einops.EinsumAxis.named "row" by decide]).trans <|
      Equiv.piUnique (fun _ : Fin 1 => Fin contracted)
  refine (Einops.Semantics.denoteEinsum_apply _ _ _ ?_ ?_ ?_).trans ?_
  · intro contractedCoordinate
    exact
      (row, contractedCoordinateEquiv contractedCoordinate,
        column, PUnit.unit)
  · intro contractedCoordinate
    change (row, column, PUnit.unit) = (row, column, PUnit.unit)
    rfl
  · intro contractedCoordinate
    apply contractedCoordinateEquiv.injective
    rfl
  · refine Fintype.sum_equiv contractedCoordinateEquiv _ _ ?_
    intro contractedCoordinate
    with_unfolding_all
      grw (transparency := all) [Einops.Semantics.einsumProduct_get_ordered]
      simp only [List.ofFn_succ, List.ofFn_zero, List.foldl_cons,
        List.foldl_nil, Fin.cases_zero, Fin.cases_succ, one_mul]
      congr 1
      · congr 1
        unfold Einops.Einsum.inputOfGlobal
        change
          Coord.broadcast [rows, contracted] [rows, contracted] _ _ =
            (row, contractedCoordinateEquiv contractedCoordinate,
              PUnit.unit)
        grw (transparency := all) [Coord.broadcast_self]
        rfl
      · congr 1
        unfold Einops.Einsum.inputOfGlobal
        change
          Coord.broadcast [contracted, columns] [contracted, columns] _ _ =
            (contractedCoordinateEquiv contractedCoordinate,
              column, PUnit.unit)
        grw (transparency := all) [Coord.broadcast_self]
        rfl

/--
Repeating and contracting one matrix label computes the mathlib trace.
-/
theorem einsum_diagonal_trace {dimension : Nat}
    (matrixTensor : Rep R [dimension, dimension]) :
    (einsum matrixTensor "diagonal diagonal ->") PUnit.unit =
      Matrix.trace (matrixEquiv R dimension dimension matrixTensor) := by
  rw [Einops.Lowering.einsumKernel_correct]
  simp only [Matrix.trace, Matrix.diag_apply]
  let diagonalCoordinateEquiv :
      AxisTuple
          (fun axis =>
            if axis = Einops.EinsumAxis.named "diagonal" then dimension
            else 1)
          [Einops.EinsumAxis.named "diagonal"] ≃
        Fin dimension :=
    (Equiv.piCongrRight fun index : Fin 1 => by
      have hIndex : index = 0 := Subsingleton.elim _ _
      subst index
      apply finCongr
      simp).trans <|
      Equiv.piUnique (fun _ : Fin 1 => Fin dimension)
  refine (Einops.Semantics.denoteEinsum_apply _ _ _ ?_ ?_ ?_).trans ?_
  · intro diagonalCoordinate
    exact (diagonalCoordinateEquiv diagonalCoordinate, PUnit.unit)
  · intro diagonalCoordinate
    rfl
  · intro diagonalCoordinate
    apply diagonalCoordinateEquiv.injective
    rfl
  · refine Fintype.sum_equiv diagonalCoordinateEquiv _ _ ?_
    intro diagonalCoordinate
    with_unfolding_all
      grw (transparency := all) [Einops.Semantics.einsumProduct_get_ordered]
      simp only [List.ofFn_succ, List.ofFn_zero, List.foldl_cons,
        List.foldl_nil, Fin.cases_zero, one_mul, matrixEquiv_apply]
      congr 1
      unfold Einops.Einsum.inputOfGlobal
      change
        Coord.broadcast [dimension, dimension] [dimension, dimension] _ _ =
          (diagonalCoordinateEquiv diagonalCoordinate,
            diagonalCoordinateEquiv diagonalCoordinate, PUnit.unit)
      grw (transparency := all) [Coord.broadcast_self]
      rfl

/--
Omitting the column label produces the finite sum of each matrix row.
-/
theorem einsum_row_sums (matrixTensor : Rep R [rows, columns])
    (row : Fin rows) :
    (einsum matrixTensor "row column -> row") (row, PUnit.unit) =
      ∑ column, matrixEquiv R rows columns matrixTensor row column := by
  rw [Einops.Lowering.einsumKernel_correct]
  simp only [matrixEquiv_apply]
  let columnCoordinateEquiv :
      AxisTuple
          (fun axis =>
            if axis = Einops.EinsumAxis.named "row" then rows
            else if axis = Einops.EinsumAxis.named "column" then columns
            else 1)
          [Einops.EinsumAxis.named "column"] ≃
        Fin columns :=
    (Equiv.piCongrRight fun index : Fin 1 => by
      have hIndex : index = 0 := Subsingleton.elim _ _
      subst index
      apply finCongr
      simp [
        show Einops.EinsumAxis.named "column" ≠
            Einops.EinsumAxis.named "row" by decide]).trans <|
      Equiv.piUnique (fun _ : Fin 1 => Fin columns)
  refine (Einops.Semantics.denoteEinsum_apply _ _ _ ?_ ?_ ?_).trans ?_
  · intro columnCoordinate
    exact (row, columnCoordinateEquiv columnCoordinate, PUnit.unit)
  · intro columnCoordinate
    rfl
  · intro columnCoordinate
    apply columnCoordinateEquiv.injective
    rfl
  · refine Fintype.sum_equiv columnCoordinateEquiv _ _ ?_
    intro columnCoordinate
    with_unfolding_all
      grw (transparency := all) [Einops.Semantics.einsumProduct_get_ordered]
      simp only [List.ofFn_succ, List.ofFn_zero, List.foldl_cons,
        List.foldl_nil, Fin.cases_zero, one_mul]
      congr 1
      unfold Einops.Einsum.inputOfGlobal
      change
        Coord.broadcast [rows, columns] [rows, columns] _ _ =
          (row, columnCoordinateEquiv columnCoordinate, PUnit.unit)
      grw (transparency := all) [Coord.broadcast_self]
      rfl

/--
Omitting the row label produces the finite sum of each matrix column.
-/
theorem einsum_column_sums (matrixTensor : Rep R [rows, columns])
    (column : Fin columns) :
    (einsum matrixTensor "row column -> column") (column, PUnit.unit) =
      ∑ row, matrixEquiv R rows columns matrixTensor row column := by
  rw [Einops.Lowering.einsumKernel_correct]
  simp only [matrixEquiv_apply]
  let rowCoordinateEquiv :
      AxisTuple
          (fun axis =>
            if axis = Einops.EinsumAxis.named "row" then rows
            else if axis = Einops.EinsumAxis.named "column" then columns
            else 1)
          [Einops.EinsumAxis.named "row"] ≃
        Fin rows :=
    (Equiv.piCongrRight fun index : Fin 1 => by
      have hIndex : index = 0 := Subsingleton.elim _ _
      subst index
      apply finCongr
      simp).trans <|
      Equiv.piUnique (fun _ : Fin 1 => Fin rows)
  refine (Einops.Semantics.denoteEinsum_apply _ _ _ ?_ ?_ ?_).trans ?_
  · intro rowCoordinate
    exact (rowCoordinateEquiv rowCoordinate, column, PUnit.unit)
  · intro rowCoordinate
    rfl
  · intro rowCoordinate
    apply rowCoordinateEquiv.injective
    rfl
  · refine Fintype.sum_equiv rowCoordinateEquiv _ _ ?_
    intro rowCoordinate
    with_unfolding_all
      grw (transparency := all) [Einops.Semantics.einsumProduct_get_ordered]
      simp only [List.ofFn_succ, List.ofFn_zero, List.foldl_cons,
        List.foldl_nil, Fin.cases_zero, one_mul]
      congr 1
      unfold Einops.Einsum.inputOfGlobal
      change
        Coord.broadcast [rows, columns] [rows, columns] _ _ =
          (rowCoordinateEquiv rowCoordinate, column, PUnit.unit)
      grw (transparency := all) [Coord.broadcast_self]
      rfl

end

end TorchLean.Tensor.Rep

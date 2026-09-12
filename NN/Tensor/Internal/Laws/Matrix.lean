/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Elab
public import NN.Tensor.Internal.Laws.Equivalence
public import Mathlib.Data.Matrix.Mul
public import Mathlib.LinearAlgebra.Matrix.Trace

/-!
# Mathlib matrix correspondence

A rank-two TorchLean.Tensor.Internal tensor has coordinates
`Fin rows × (Fin columns × PUnit)`, while a mathlib matrix is a curried
function `Fin rows → Fin columns → α`. `Rep.matrixEquiv` identifies these
representations without changing entry order.

The theorems in this module connect general symbolic einops expressions to
the corresponding mathlib matrix operations. Their dimensions are arbitrary
natural numbers, including zero.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal

open TorchLean.Tensor

universe u

namespace Rep

/--
The canonical equivalence between a rank-two coordinate tensor and a mathlib
matrix with the same row and column index types.
-/
def matrixEquiv (α : Type u) [Storage α] (rows columns : Nat) :
    Rep α [rows, columns] ≃ Matrix (Fin rows) (Fin columns) α where
  toFun tensor row column :=
    tensor (row, column, PUnit.unit)
  invFun matrix :=
    Rep.ofFn fun coordinate =>
      matrix coordinate.1 coordinate.2.1
  left_inv tensor := by
    ext coordinate
    rcases coordinate with ⟨row, column, scalarCoordinate⟩
    cases scalarCoordinate
    exact Rep.get_ofFn
      (fun coordinate : Coord [rows, columns] =>
        tensor (coordinate.1, coordinate.2.1, PUnit.unit))
      (row, column, PUnit.unit)
  right_inv matrix := by
    ext row column
    exact Rep.get_ofFn
      (fun coordinate : Coord [rows, columns] =>
        matrix coordinate.1 coordinate.2.1)
      (row, column, PUnit.unit)

variable {α : Type u} [Storage α] {rows columns : Nat}

/-- The matrix view reads the same row and column entry as the tensor view. -/
@[simp] theorem matrixEquiv_apply (tensor : Rep α [rows, columns])
    (row : Fin rows) (column : Fin columns) :
    matrixEquiv α rows columns tensor row column =
      tensor (row, column, PUnit.unit) :=
  rfl

/-- The inverse matrix view rebuilds the corresponding rank-two tensor entry. -/
@[simp] theorem matrixEquiv_symm_apply
    (matrix : Matrix (Fin rows) (Fin columns) α)
    (coordinate : Coord [rows, columns]) :
    (matrixEquiv α rows columns).symm matrix coordinate =
      matrix coordinate.1 coordinate.2.1 :=
  Rep.get_ofFn _ _

/--
The elementary rank-two axis swap is exactly mathlib matrix transpose.
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
              simpa only [Shape.size, Nat.mul_one] using outerStep
      _ =
          ((Coord.linearize (s := []) PUnit.unit).val +
              innerCoordinate.val) +
            inner * outerCoordinate.val := by
              exact congrArg (fun value =>
                value + inner * outerCoordinate.val) <| by
                  simpa only [Shape.size, Nat.one_mul] using innerStep
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
  apply Lowering.rearrangeTensor_apply_eq_of_linearIndex_eq
  simp (config := { zeta := true }) only [Check.PartialAxisLengths.set,
    Check.PartialAxisLengths.seed, Check.SupplementaryLengths.lookup?,
    List.find?_nil, Option.map_none, Check.NormalizedTransform.inputAxes,
    Check.TransformPlan.normalized, List.flatten_cons, List.flatten_nil,
    List.append_nil, List.cons_append, List.nil_append,
    Check.NormalizedTransform.outputAxes, Shape.size_cons, Shape.size_nil]
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
  apply rearrangeLinearIndex_swap
  · decide
  · exact row.isLt

variable {R : Type u} [Storage R] [AddCommMonoid R] [Monoid R]
variable {contracted : Nat}

/--
Symbolic matrix multiplication is the ordinary two-operand einsum
contraction, for arbitrary natural-number dimensions.
-/
theorem matrixEquiv_einsum_mul
    (leftTensor : Rep R [rows, contracted])
    (rightTensor : Rep R [contracted, columns]) :
    matrixEquiv R rows columns
        (einsum leftTensor, rightTensor
          "row contracted, contracted column -> row column") =
      matrixEquiv R rows contracted leftTensor *
        matrixEquiv R contracted columns rightTensor := by
  rw [Lowering.einsumTensorKernel_correct]
  ext row column
  simp only [matrixEquiv_apply, Matrix.mul_apply]
  let contractedCoordinateEquiv :
      AxisTuple
          (fun axis =>
            if axis = Check.EinsumAxis.named "row" then rows
            else if axis = Check.EinsumAxis.named "contracted" then contracted
            else if axis = Check.EinsumAxis.named "column" then columns
            else 1)
          [Check.EinsumAxis.named "contracted"] ≃
        Fin contracted :=
    (Equiv.piCongrRight fun index : Fin 1 => by
      have hIndex : index = 0 := Subsingleton.elim _ _
      subst index
      apply finCongr
      simp [
        show Check.EinsumAxis.named "contracted" ≠
            Check.EinsumAxis.named "row" by decide]).trans <|
      Equiv.piUnique (fun _ : Fin 1 => Fin contracted)
  refine (Semantics.denoteEinsum_apply_reconstructed _ _ _ ?_ ?_ ?_).trans ?_
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
      grw (transparency := all) [Semantics.einsumProductTensor_get]
      simp only [List.ofFn_succ, List.ofFn_zero, List.prod_cons,
        List.prod_nil, Fin.cases_zero, Fin.cases_succ, mul_one]
      congr 1
      · congr 1
        unfold Check.CheckedEinsum.inputCoordinateOfGlobal
        change
          Coord.broadcast [rows, contracted] [rows, contracted] _ _ =
            (row, contractedCoordinateEquiv contractedCoordinate,
              PUnit.unit)
        grw (transparency := all) [Coord.broadcast_self]
        rfl
      · congr 1
        unfold Check.CheckedEinsum.inputCoordinateOfGlobal
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
  rw [Lowering.einsumTensorKernel_correct]
  simp only [Matrix.trace, Matrix.diag_apply]
  let diagonalCoordinateEquiv :
      AxisTuple
          (fun axis =>
            if axis = Check.EinsumAxis.named "diagonal" then dimension
            else 1)
          [Check.EinsumAxis.named "diagonal"] ≃
        Fin dimension :=
    (Equiv.piCongrRight fun index : Fin 1 => by
      have hIndex : index = 0 := Subsingleton.elim _ _
      subst index
      apply finCongr
      simp).trans <|
      Equiv.piUnique (fun _ : Fin 1 => Fin dimension)
  refine (Semantics.denoteEinsum_apply_reconstructed _ _ _ ?_ ?_ ?_).trans ?_
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
      grw (transparency := all) [Semantics.einsumProductTensor_get]
      simp only [List.ofFn_succ, List.ofFn_zero, List.prod_cons,
        List.prod_nil, Fin.cases_zero, mul_one, matrixEquiv_apply]
      congr 1
      unfold Check.CheckedEinsum.inputCoordinateOfGlobal
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
  rw [Lowering.einsumTensorKernel_correct]
  simp only [matrixEquiv_apply]
  let columnCoordinateEquiv :
      AxisTuple
          (fun axis =>
            if axis = Check.EinsumAxis.named "row" then rows
            else if axis = Check.EinsumAxis.named "column" then columns
            else 1)
          [Check.EinsumAxis.named "column"] ≃
        Fin columns :=
    (Equiv.piCongrRight fun index : Fin 1 => by
      have hIndex : index = 0 := Subsingleton.elim _ _
      subst index
      apply finCongr
      simp [
        show Check.EinsumAxis.named "column" ≠
            Check.EinsumAxis.named "row" by decide]).trans <|
      Equiv.piUnique (fun _ : Fin 1 => Fin columns)
  refine (Semantics.denoteEinsum_apply_reconstructed _ _ _ ?_ ?_ ?_).trans ?_
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
      grw (transparency := all) [Semantics.einsumProductTensor_get]
      simp only [List.ofFn_succ, List.ofFn_zero, List.prod_cons,
        List.prod_nil, Fin.cases_zero, mul_one]
      congr 1
      unfold Check.CheckedEinsum.inputCoordinateOfGlobal
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
  rw [Lowering.einsumTensorKernel_correct]
  simp only [matrixEquiv_apply]
  let rowCoordinateEquiv :
      AxisTuple
          (fun axis =>
            if axis = Check.EinsumAxis.named "row" then rows
            else if axis = Check.EinsumAxis.named "column" then columns
            else 1)
          [Check.EinsumAxis.named "row"] ≃
        Fin rows :=
    (Equiv.piCongrRight fun index : Fin 1 => by
      have hIndex : index = 0 := Subsingleton.elim _ _
      subst index
      apply finCongr
      simp).trans <|
      Equiv.piUnique (fun _ : Fin 1 => Fin rows)
  refine (Semantics.denoteEinsum_apply_reconstructed _ _ _ ?_ ?_ ?_).trans ?_
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
      grw (transparency := all) [Semantics.einsumProductTensor_get]
      simp only [List.ofFn_succ, List.ofFn_zero, List.prod_cons,
        List.prod_nil, Fin.cases_zero, mul_one]
      congr 1
      unfold Check.CheckedEinsum.inputCoordinateOfGlobal
      change
        Coord.broadcast [rows, columns] [rows, columns] _ _ =
          (rowCoordinateEquiv rowCoordinate, column, PUnit.unit)
      grw (transparency := all) [Coord.broadcast_self]
      rfl

end Rep

end TorchLean.Tensor.Internal

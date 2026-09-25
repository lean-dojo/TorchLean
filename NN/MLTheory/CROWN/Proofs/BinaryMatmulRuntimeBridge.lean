/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.Semantics
public import NN.Tensor.Internal.Laws.RowMajor
public import NN.Proofs.Tensor.Basic.Core

/-!
# Binary matmul and flat runtime values

The typed matrix kernel and the broadcast evaluator contract the same scalar coordinates in
the same order. These lemmas connect shaped runtime values to the flat certificate semantics.
-/

public section

namespace NN.IR.OpContracts.MatmulDims

private theorem radix_fold_invariant (extents : List Nat) (index offset stride : Nat) :
    let result := extents.foldl (fun (state : Nat × Nat × Nat) extent =>
      (state.1 / extent, state.2.1 + (state.1 % extent) * state.2.2,
        state.2.2 * extent)) (index, offset, stride)
    result.1 = index / extents.prod ∧
      result.2.1 + result.1 * result.2.2 = offset + index * stride := by
  induction extents generalizing index offset stride with
  | nil => simp
  | cons extent extents ih =>
      obtain ⟨hquotient, hvalue⟩ :=
        ih (index / extent) (offset + (index % extent) * stride) (stride * extent)
      refine ⟨?_, ?_⟩
      · simpa only [List.foldl_cons, List.prod_cons, Nat.div_div_eq_div_mul] using hquotient
      · dsimp only [List.foldl_cons]
        rw [hvalue]
        have hindex := congrArg (· * stride) (Nat.mod_add_div index extent)
        nlinarith

/-- Projecting an in-bounds batch index to the same batch shape preserves it. -/
theorem batchIndex_self (shape : Spec.Shape) (index : Nat) (hi : index < shape.size) :
    batchIndex shape shape index = index := by
  let extents := shape.toList.reverse
  have hsize : extents.prod = shape.size := by
    simp [extents, Spec.Shape.toList, Spec.Shape.size_eq_prod]
  have hlist : (List.range extents.length).map (fun axis => extents.getD axis 1) = extents := by
    apply List.ext_getElem
    · simp
    · intro axis hleft hright
      simp [hright]
  have hdigit (extent value : Nat) :
      (if extent = 1 then 0 else value % extent) = value % extent := by
    by_cases h : extent = 1 <;> simp [h, Nat.mod_one]
  have hfold :
      batchIndex shape shape index =
        (extents.foldl (fun (state : Nat × Nat × Nat) extent =>
          (state.1 / extent, state.2.1 + (state.1 % extent) * state.2.2,
            state.2.2 * extent)) (index, 0, 1)).2.1 := by
    unfold batchIndex
    simp only [hdigit]
    conv_rhs => rw [← hlist, List.foldl_map]
  obtain ⟨hquotient, hvalue⟩ := radix_fold_invariant extents index 0 1
  rw [hsize, Nat.div_eq_of_lt hi] at hquotient
  rw [hquotient] at hvalue
  simpa only [hfold, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.mul_one] using hvalue

end NN.IR.OpContracts.MatmulDims

namespace NN.IR.OpContracts

/-- Successful normalization records each operand's unbroadcast matrix or vector shape. -/
theorem matmulDims_operand_shapes {a b : Spec.Shape} {dims : MatmulDims}
    (h : matmulDims a b = .ok dims) :
    dims.leftShape =
        (if dims.leftVector then [dims.inner]
          else dims.leftLeading.concat [dims.rows, dims.inner]) ∧
      dims.rightShape =
        (if dims.rightVector then [dims.inner]
          else dims.rightLeading.concat [dims.inner, dims.cols]) := by
  cases ha : a.toList.reverse with
  | nil => simp [matmulDims, ha] at h
  | cons n leftTail =>
    cases hb : b.toList.reverse with
    | nil => simp [matmulDims, hb] at h
    | cons p rightTail =>
      cases leftTail with
      | nil =>
        cases rightTail with
        | nil =>
          by_cases hInner : n = p <;>
            simp [matmulDims, ha, hb, hInner] at h
          cases h
          constructor
          · simpa [Spec.Shape.toList, Spec.Shape.ofList, Spec.Shape.concat_eq_append,
              List.reverse_cons, List.append_assoc, hInner] using congrArg List.reverse ha
          · simpa [Spec.Shape.toList, Spec.Shape.ofList, Spec.Shape.concat_eq_append,
              List.reverse_cons, List.append_assoc, hInner] using congrArg List.reverse hb
        | cons n' rightLeading =>
          by_cases hInner : n = n' <;>
            simp [matmulDims, ha, hb, hInner] at h
          cases h
          constructor
          · simpa [Spec.Shape.toList, Spec.Shape.ofList, Spec.Shape.concat_eq_append,
              List.reverse_cons, List.append_assoc, hInner] using congrArg List.reverse ha
          · simpa [Spec.Shape.toList, Spec.Shape.ofList, Spec.Shape.concat_eq_append,
              List.reverse_cons, List.append_assoc, hInner] using congrArg List.reverse hb
      | cons m leftLeading =>
        cases rightTail with
        | nil =>
          by_cases hInner : n = p <;>
            simp [matmulDims, ha, hb, hInner] at h
          cases h
          constructor
          · simpa [Spec.Shape.toList, Spec.Shape.ofList, Spec.Shape.concat_eq_append,
              List.reverse_cons, List.append_assoc, hInner] using congrArg List.reverse ha
          · simpa [Spec.Shape.toList, Spec.Shape.ofList, Spec.Shape.concat_eq_append,
              List.reverse_cons, List.append_assoc, hInner] using congrArg List.reverse hb
        | cons n' rightLeading =>
          by_cases hLeading : leftLeading = rightLeading
          · by_cases hInner : n = n' <;>
              simp [matmulDims, ha, hb, hLeading, hInner] at h
            cases h
            constructor
            · simpa [Spec.Shape.toList, Spec.Shape.ofList, Spec.Shape.concat_eq_append,
                List.reverse_cons, List.append_assoc, hInner, hLeading] using
                congrArg List.reverse ha
            · simpa [Spec.Shape.toList, Spec.Shape.ofList, Spec.Shape.concat_eq_append,
                List.reverse_cons, List.append_assoc, hInner, hLeading] using
                congrArg List.reverse hb
          · by_cases hInner : n = n'
            · cases hBatch : broadcastMatmulBatches leftLeading rightLeading with
              | error message =>
                simp [matmulDims, ha, hb, hLeading, hInner, hBatch,
                  Bind.bind, Except.bind] at h
              | ok batch =>
                simp [matmulDims, ha, hb, hLeading, hInner, hBatch,
                  Pure.pure, Except.pure, Bind.bind, Except.bind] at h
                cases h
                constructor
                · simpa [Spec.Shape.toList, Spec.Shape.ofList, Spec.Shape.concat_eq_append,
                    List.reverse_cons, List.append_assoc, hInner] using congrArg List.reverse ha
                · simpa [Spec.Shape.toList, Spec.Shape.ofList, Spec.Shape.concat_eq_append,
                    List.reverse_cons, List.append_assoc, hInner] using congrArg List.reverse hb
            · simp [matmulDims, ha, hb, hLeading, hInner] at h

/-- The typed matrix kernel can be selected only when both batch layouts are unchanged. -/
theorem matmulDims_kernel_batches {a b : Spec.Shape} {dims : MatmulDims}
    (h : matmulDims a b = .ok dims)
    (hleft : dims.leftShape = dims.leading.concat [dims.rows, dims.inner])
    (hright : dims.rightShape = dims.leading.concat [dims.inner, dims.cols]) :
    dims.leftLeading = dims.leading ∧ dims.rightLeading = dims.leading := by
  obtain ⟨hl, hr⟩ := matmulDims_operand_shapes h
  constructor
  · cases hv : dims.leftVector with
    | false =>
        simp only [hv, Bool.false_eq_true, ↓reduceIte] at hl
        rw [hleft] at hl
        exact (List.append_cancel_right (by
          simpa only [Spec.Shape.concat_eq_append] using hl)).symm
    | true =>
        simp only [hv, ↓reduceIte] at hl
        have hlength := congrArg List.length (hl.symm.trans hleft)
        simp only [Spec.Shape.concat_eq_append, List.length_append, List.length_cons,
          List.length_nil] at hlength
        omega
  · cases hv : dims.rightVector with
    | false =>
        simp only [hv, Bool.false_eq_true, ↓reduceIte] at hr
        rw [hright] at hr
        exact (List.append_cancel_right (by
          simpa only [Spec.Shape.concat_eq_append] using hr)).symm
    | true =>
        simp only [hv, ↓reduceIte] at hr
        have hlength := congrArg List.length (hr.symm.trans hright)
        simp only [Spec.Shape.concat_eq_append, List.length_append, List.length_cons,
          List.length_nil] at hlength
        omega

end NN.IR.OpContracts

namespace NN.IR.Graph

open Spec TorchLean TorchLean.Tensor
open TorchLean.Tensor.Internal (Coord)

private theorem flatten_read_coordinate {α : Type} [Storage α] [Zero α]
    {shape : Shape} (value : Tensor α shape) (coordinate : shape.Coord) :
    getAtOrZero (Tensor.flattenSpec value) [(Coord.linearize coordinate).val] =
      value coordinate := by
  have hi : (Coord.linearize coordinate).val < shape.size := by
    simpa only [← Shape.internalSize_eq] using (Coord.linearize coordinate).isLt
  rw [get_at_or_zero_dim_cons, dite_eq_left hi, get_at_or_zero_scalar_nil]
  change Tensor.getScalar (Tensor.flattenSpec value)
    ⟨(Coord.linearize coordinate).val, hi⟩ = value coordinate
  exact Spec.getScalar_flattenSpec_linearize value coordinate

private theorem flatten_read_cast {α : Type} [Storage α] [Zero α]
    {source target : Shape} (value : Tensor α source) (h : source = target) (index : Nat) :
    getAtOrZero (Tensor.flattenSpec (Tensor.castShape value h)) [index] =
      getAtOrZero (Tensor.flattenSpec value) [index] := by
  cases h
  rfl

private theorem flat_read_fin {α : Type} [Storage α] [Zero α]
    {n : Nat} (value : Tensor α [n]) (index : Fin n) :
    getAtOrZero value [index.val] = Tensor.getScalar value index := by
  simp [get_at_or_zero_dim_cons, index.isLt, Tensor.getScalar, Spec.get]

private theorem matrix_coordinate_index (leading : Shape) {rows cols : Nat}
    (batch : leading.Coord) (row : Fin rows) (col : Fin cols) :
    (Coord.linearize ((Coord.appendEquiv leading [rows, cols]).symm
      (batch, row, col, PUnit.unit))).val =
        (Coord.linearize batch).val * (rows * cols) + row.val * cols + col.val := by
  rw [Coord.linearize_appendEquiv_symm_val,
    Coord.linearize_cons_val (s := [cols]) row (col, PUnit.unit),
    Tensor.vectorCoordinate_linearize_val col]
  simp only [Tensor.Internal.Shape.size_cons, Tensor.Internal.Shape.size_nil, Nat.mul_one]
  ring

private theorem matrix_output_indices (batch rows cols row col : Nat)
    (hrow : row < rows) (hcol : col < cols) :
    let output := batch * (rows * cols) + row * cols + col
    output / (rows * cols) = batch ∧
      output % (rows * cols) / cols = row ∧ output % cols = col := by
  have hsmall : row * cols + col < rows * cols := by
    have h := Nat.mul_le_mul_right cols (Nat.succ_le_of_lt hrow)
    nlinarith
  have hout : batch * (rows * cols) + row * cols + col =
      (row * cols + col) + (rows * cols) * batch := by ring
  dsimp only
  constructor
  · rw [hout, Nat.add_mul_div_left _ _ (Nat.zero_lt_of_lt hsmall),
      Nat.div_eq_of_lt hsmall, Nat.zero_add]
  constructor
  · rw [hout, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hsmall,
      show row * cols + col = col + cols * row by ring,
      Nat.add_mul_div_left _ _ (Nat.zero_lt_of_lt hcol), Nat.div_eq_of_lt hcol,
      Nat.zero_add]
  · rw [show batch * (rows * cols) + row * cols + col =
        col + cols * (row + rows * batch) by ring,
      Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hcol]

private theorem cast_cons_apply {α : Type} [Storage α] {source target : Shape} {n : Nat}
    (value : Tensor α (.dim n source)) (h : Shape.dim n source = Shape.dim n target)
    (head : Fin n) (tail : target.Coord) :
    Tensor.castShape value h (head, tail) =
      Tensor.castShape (Tensor.unstack value head) (List.cons.inj h).2 tail := by
  obtain rfl := (List.cons.inj h).2
  simp [Tensor.unstack]

private theorem matmulLeading_coordinate {α : Type} [Storage α] [Context α]
    (leading : Shape) {rows inner cols : Nat}
    (left : Tensor α (leading.concat [rows, inner]))
    (right : Tensor α (leading.concat [inner, cols]))
    (batch : leading.Coord) (row : Fin rows) (col : Fin cols) :
    Tensor.castShape (matmulLeading leading left right)
        (Shape.concat_eq_append leading [rows, cols])
        ((Coord.appendEquiv leading [rows, cols]).symm (batch, row, col, PUnit.unit)) =
      (List.finRange inner).foldl (fun acc k =>
        acc +
          Tensor.castShape left (Shape.concat_eq_append leading [rows, inner])
            ((Coord.appendEquiv leading [rows, inner]).symm
            (batch, row, k, PUnit.unit)) *
          Tensor.castShape right (Shape.concat_eq_append leading [inner, cols])
            ((Coord.appendEquiv leading [inner, cols]).symm
            (batch, k, col, PUnit.unit))) 0 := by
  induction leading with
  | scalar =>
      cases batch
      simp [matmulLeading, Tensor.zipEach, Coord.appendEquiv, Spec.matMulSpec,
        Spec.get2, Tensor.getScalar_eq_apply, Spec.get, Tensor.unstack]
  | dim length leading ih =>
      obtain ⟨head, batch⟩ := batch
      simpa [matmulLeading, Tensor.zipEach, Coord.appendEquiv,
        cast_cons_apply, Tensor.dim, Tensor.unstack] using
        ih (Tensor.unstack left head) (Tensor.unstack right head) batch

private theorem matmulLeading_flat_read {α : Type} [Storage α] [Context α]
    (leading : Shape) {rows inner cols : Nat}
    (left : Tensor α (leading.concat [rows, inner]))
    (right : Tensor α (leading.concat [inner, cols]))
    (output : Nat) (houtput : output < (leading.concat [rows, cols]).size) :
    getAtOrZero (Tensor.flattenSpec (matmulLeading leading left right)) [output] =
      (List.range inner).foldl (fun acc k =>
        acc +
          getAtOrZero (Tensor.flattenSpec left)
            [OpContracts.MatmulDims.batchIndex leading leading (output / (rows * cols)) *
              (rows * inner) + (output % (rows * cols) / cols) * inner + k] *
          getAtOrZero (Tensor.flattenSpec right)
            [OpContracts.MatmulDims.batchIndex leading leading (output / (rows * cols)) *
              (inner * cols) + k * cols + output % cols]) 0 := by
  let coordinate : (leading ++ [rows, cols]).Coord :=
    Shape.Coord.unlinearize
      ⟨output, by simpa only [Shape.concat_eq_append] using houtput⟩
  have hlinear : (Coord.linearize coordinate).val = output :=
    congrArg Fin.val (Shape.Coord.linearize_unlinearize
      (⟨output, by simpa only [Shape.concat_eq_append] using houtput⟩ :
        Fin (leading ++ [rows, cols]).size))
  obtain ⟨⟨batch, row, col, ⟨⟩⟩, hcoordinate⟩ :=
    (Coord.appendEquiv leading [rows, cols]).symm.surjective coordinate
  have hout : (Coord.linearize batch).val * (rows * cols) + row.val * cols + col.val =
      output := by
    rw [← matrix_coordinate_index leading batch row col, hcoordinate]
    exact hlinear
  obtain ⟨hbatch, hrow, hcol⟩ :=
    matrix_output_indices (Coord.linearize batch).val rows cols row.val col.val
      row.isLt col.isLt
  rw [hout] at hbatch hrow hcol
  have hproject := OpContracts.MatmulDims.batchIndex_self leading
    (Coord.linearize batch).val
    (by simpa only [← Shape.internalSize_eq] using (Coord.linearize batch).isLt)
  have hread := flatten_read_coordinate
    (Tensor.castShape (matmulLeading leading left right)
      (Shape.concat_eq_append leading [rows, cols]))
    ((Coord.appendEquiv leading [rows, cols]).symm (batch, row, col, PUnit.unit))
  rw [flatten_read_cast, matrix_coordinate_index, hout] at hread
  rw [hread, matmulLeading_coordinate, hbatch, hrow, hcol, hproject,
    ← List.map_coe_finRange_eq_range, List.foldl_map]
  apply congrArg (fun f => (List.finRange inner).foldl f 0)
  funext acc k
  have hl := flatten_read_coordinate
    (Tensor.castShape left (Shape.concat_eq_append leading [rows, inner]))
    ((Coord.appendEquiv leading [rows, inner]).symm (batch, row, k, PUnit.unit))
  have hr := flatten_read_coordinate
    (Tensor.castShape right (Shape.concat_eq_append leading [inner, cols]))
    ((Coord.appendEquiv leading [inner, cols]).symm (batch, k, col, PUnit.unit))
  rw [flatten_read_cast, matrix_coordinate_index] at hl hr
  rw [hl, hr]

/-- Every checked matmul layout has the same flat value in the typed and flat evaluators.
This includes arbitrary batch broadcasting, vector promotion, and empty dimensions. -/
theorem flattenSpec_matmulWithDims {α : Type} [Storage α] [Context α]
    {leftShape rightShape : Shape} {dims : OpContracts.MatmulDims}
    (hDims : OpContracts.matmulDims leftShape rightShape = .ok dims)
    (left : Tensor α dims.leftShape) (right : Tensor α dims.rightShape) :
    Tensor.flattenSpec (matmulWithDims dims left right) =
      matmulFlat dims (Tensor.flattenSpec left) (Tensor.flattenSpec right) := by
  apply Tensor.ext_vector
  intro output
  rw [← flat_read_fin (Tensor.flattenSpec (matmulWithDims dims left right)) output]
  unfold matmulWithDims
  split
  · rename_i h
    obtain ⟨hleft, hright⟩ := OpContracts.matmulDims_kernel_batches hDims h.1 h.2.1
    rw [flatten_read_cast, matmulLeading_flat_read _ _ _ output.val
      (by simpa only [← h.2.2] using output.isLt)]
    simp only [matmulFlat, Tensor.getScalar_ofFn, OpContracts.MatmulDims.leftIndex,
      OpContracts.MatmulDims.rightIndex, hleft, hright, flatten_read_cast]
  · rw [Tensor.flattenSpec_unflattenSpec, flat_read_fin]

end NN.IR.Graph

namespace NN.MLTheory.CROWN.Graph.CertSoundness

open Spec TorchLean TorchLean.Tensor

noncomputable section

/-- The binary certificate evaluator reproduces a checked shaped runtime matrix product. -/
theorem evalBinaryMatmul?_matmulWithDims
    {leftShape rightShape : Shape} {dims : NN.IR.OpContracts.MatmulDims}
    (hDims : NN.IR.OpContracts.matmulDims leftShape rightShape = .ok dims)
    (left : Tensor ℝ dims.leftShape) (right : Tensor ℝ dims.rightShape) :
    evalBinaryMatmul? leftShape rightShape
        ⟨dims.leftShape.size, Tensor.flattenSpec left⟩
        ⟨dims.rightShape.size, Tensor.flattenSpec right⟩ =
      some ⟨dims.outShape.size,
        Tensor.flattenSpec (NN.IR.Graph.matmulWithDims dims left right)⟩ := by
  obtain ⟨hleft, hright⟩ := NN.IR.OpContracts.matmulDims_shapes hDims
  simp only [evalBinaryMatmul?, hDims, Except.toOption]
  rw [ite_eq_left ⟨congrArg Shape.size hleft.symm, congrArg Shape.size hright.symm⟩,
    NN.IR.Graph.flattenSpec_matmulWithDims hDims]

end

end NN.MLTheory.CROWN.Graph.CertSoundness

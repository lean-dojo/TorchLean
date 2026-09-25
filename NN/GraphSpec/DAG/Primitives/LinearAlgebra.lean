/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.DAG.Syntax

/-!
# DAG Linear Algebra Primitives

Typed matrix products, vector-matrix products, and full sums for general computation graphs.
-/

@[expose] public section

namespace NN
namespace GraphSpec
namespace DAG

open _root_.Spec _root_.TorchLean
open TorchLean.Tensor
open _root_.TorchLean.Tensor

namespace PrimOp

/-- Multiply matrices after broadcasting their batch prefixes to a common shape. -/
def matmul (batchA batchB batch : Shape) (mDim nDim pDim : Nat)
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch) :
    PrimOp
      [batchA.concat [mDim, nDim], batchB.concat [nDim, pDim]]
      (batch.concat [mDim, pDim]) :=
  { name := s!"matmul({mDim},{nDim},{pDim})"
    specFwd := fun {α} _ _ xs =>
      match xs with
      | .cons a (.cons b .nil) =>
          _root_.TorchLean.Tensor.matmulSpec broadcastA broadcastB a b
    program := fun {α} _ _ =>
      fun {m} _ _ => fun a b => by
        letI : Shape.BroadcastTo batchA batch := ⟨broadcastA⟩
        letI : Shape.BroadcastTo batchB batch := ⟨broadcastB⟩
        exact
        Runtime.Autograd.Model.matmul (m := m) (α := α)
          (batchA := batchA) (batchB := batchB) (batch := batch)
          (mDim := mDim) (nDim := nDim) (pDim := pDim) a b }

/-- Pure evaluation of matrix multiplication with explicitly broadcast batch prefixes. -/
@[simp] theorem matmul_specFwd {batchA batchB batch : Shape} {mDim nDim pDim : Nat}
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch)
    {α : Type} [TorchLean.Storage α] [Context α]
    (left : _root_.TorchLean.Tensor α (batchA.concat [mDim, nDim]))
    (right : _root_.TorchLean.Tensor α (batchB.concat [nDim, pDim])) :
    (matmul batchA batchB batch mDim nDim pDim broadcastA broadcastB).specFwd
        (.cons left (.cons right .nil)) =
      _root_.TorchLean.Tensor.matmulSpec broadcastA broadcastB left right := by
  rfl

namespace Internal

/-- Multiply vectors and matrices pointwise over a common batch shape. -/
def vecMatCommonBatchSpec {α : Type} [TorchLean.Storage α] [Context α] {rows columns : Nat}
    (batch : Shape) (vectors : _root_.TorchLean.Tensor α (batch.concat [rows]))
    (matrices : _root_.TorchLean.Tensor α (batch.concat [rows, columns])) :
    _root_.TorchLean.Tensor α (batch.concat [columns]) :=
  _root_.TorchLean.Tensor.zipEach batch [columns] _root_.Spec.vecMatMulSpec vectors matrices

/-- With no batch axes left, batched vector-matrix product is the plain one. -/
@[simp] theorem vecMatCommonBatchSpec_scalar {α : Type} [TorchLean.Storage α] [Context α]
    {rows columns : Nat} (vector : _root_.TorchLean.Tensor α [rows])
    (matrix : _root_.TorchLean.Tensor α [rows, columns]) :
    vecMatCommonBatchSpec .scalar vector matrix =
      _root_.Spec.vecMatMulSpec vector matrix := by
  rfl

/-- A batch axis distributes over the vector-matrix product slicewise. -/
@[simp] theorem vecMatCommonBatchSpec_dim {α : Type} [TorchLean.Storage α] [Context α]
    {count rows columns : Nat} {rest : Shape}
    (vectors : Fin count → _root_.TorchLean.Tensor α (rest.concat [rows]))
    (matrices : Fin count → _root_.TorchLean.Tensor α (rest.concat [rows, columns])) :
    vecMatCommonBatchSpec (.dim count rest) (.dim vectors) (.dim matrices) =
      .dim (fun index => vecMatCommonBatchSpec rest (vectors index) (matrices index)) := by
  simp [vecMatCommonBatchSpec, _root_.TorchLean.Tensor.zipEach]

end Internal

/-- Multiply vectors by matrices after broadcasting their batch prefixes to a common shape. -/
def broadcastVecMat (vectorBatch matrixBatch batch : Shape) (rows columns : Nat)
    (broadcastVector : Shape.CanBroadcastTo vectorBatch batch)
    (broadcastMatrix : Shape.CanBroadcastTo matrixBatch batch) :
    PrimOp
      [vectorBatch.concat [rows], matrixBatch.concat [rows, columns]]
      (batch.concat [columns]) :=
  { name := s!"vecMat({rows},{columns})"
    specFwd := fun {α} _ _ xs =>
      match xs with
      | .cons vector (.cons matrix .nil) =>
          let commonVector := _root_.TorchLean.Tensor.broadcastTo
            (_root_.TorchLean.Tensor.LinearAlgebra.Internal.extendBroadcastSuffix [rows]
              broadcastVector) vector
          let commonMatrix := _root_.TorchLean.Tensor.broadcastTo
            (_root_.TorchLean.Tensor.LinearAlgebra.Internal.extendBroadcastSuffix [rows, columns]
              broadcastMatrix)
            matrix
          Internal.vecMatCommonBatchSpec batch commonVector commonMatrix
    program := fun {α} _ _ =>
      fun {m} _ _ => fun vector matrix => by
        letI : Shape.BroadcastTo vectorBatch batch := ⟨broadcastVector⟩
        letI : Shape.BroadcastTo matrixBatch batch := ⟨broadcastMatrix⟩
        exact (do
          let rowVector ← Runtime.Autograd.Model.reshape (m := m) (α := α)
            (s₂ := vectorBatch.concat [1, rows]) vector
            (by simp [_root_.Spec.Shape.size_concat, _root_.Spec.Shape.size])
          let product ← Runtime.Autograd.Model.matmul (m := m) (α := α)
            (batchA := vectorBatch) (batchB := matrixBatch) (batch := batch)
            (mDim := 1) (nDim := rows) (pDim := columns)
            rowVector matrix
          Runtime.Autograd.Model.reshape (m := m) (α := α)
            (s₂ := batch.concat [columns]) product
            (by simp [_root_.Spec.Shape.size_concat, _root_.Spec.Shape.size]) :
          m (Runtime.Autograd.Model.RefTy (m := m) (α := α)
            (batch.concat [columns]))) }

/-- Pure evaluation of broadcasted vector–matrix multiplication. -/
@[simp] theorem broadcastVecMat_specFwd
    {vectorBatch matrixBatch batch : Shape} {rows columns : Nat}
    (broadcastVector : Shape.CanBroadcastTo vectorBatch batch)
    (broadcastMatrix : Shape.CanBroadcastTo matrixBatch batch)
    {α : Type} [TorchLean.Storage α] [Context α]
    (vector : _root_.TorchLean.Tensor α (vectorBatch.concat [rows]))
    (matrix : _root_.TorchLean.Tensor α (matrixBatch.concat [rows, columns])) :
    (broadcastVecMat vectorBatch matrixBatch batch rows columns
      broadcastVector broadcastMatrix).specFwd (.cons vector (.cons matrix .nil)) =
      Internal.vecMatCommonBatchSpec batch
        (_root_.TorchLean.Tensor.broadcastTo
          (_root_.TorchLean.Tensor.LinearAlgebra.Internal.extendBroadcastSuffix [rows]
            broadcastVector) vector)
        (_root_.TorchLean.Tensor.broadcastTo
          (_root_.TorchLean.Tensor.LinearAlgebra.Internal.extendBroadcastSuffix [rows, columns]
            broadcastMatrix)
          matrix) := by
  rfl

/-- Sum every scalar entry of a tensor. -/
def sum (s : Shape) : PrimOp [s] .scalar :=
  { name := "sum"
    specFwd := fun {_α} _storage _ctx xs =>
      match xs with
      | .cons input .nil => .scalar (_root_.TorchLean.Tensor.sumSpec input)
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input =>
        Runtime.Autograd.Model.sum (m := m) (α := α) (s := s) input }


end PrimOp

end DAG
end GraphSpec
end NN

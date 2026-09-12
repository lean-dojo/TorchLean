/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.DAG.Syntax

/-!
# DAG Linear Algebra Primitives

Typed matrix, vector, batched contraction, and scaling operations for general computation graphs.
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
def vecMatCommonBatchSpec {α : Type} [TorchLean.Storage α] [Context α] {rows columns : Nat} :
    (batch : Shape) →
      _root_.TorchLean.Tensor α (batch.concat [rows]) →
      _root_.TorchLean.Tensor α (batch.concat [rows, columns]) →
      _root_.TorchLean.Tensor α (batch.concat [columns])
  | .scalar, vector, matrix => _root_.Spec.vecMatMulSpec vector matrix
  | .dim _ rest, vectors, matrices =>
      .dim fun index =>
        vecMatCommonBatchSpec rest (_root_.Spec.get vectors index)
          (_root_.Spec.get matrices index)

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
  simp [vecMatCommonBatchSpec]

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

/-- Dot one shared vector with every vector in a batch. -/
def batchedSharedDot (batch width : Nat) :
    PrimOp [[width], [batch, width]] [batch] :=
  { name := s!"batchedSharedDot({batch},{width})"
    specFwd := fun {α} _ _ xs =>
      match xs with
      | .cons v (.cons vectors .nil) =>
          .dim fun i => .scalar
            (_root_.TorchLean.Tensor.dotSpec (α := α) v (_root_.Spec.get vectors i))
    program := fun {α} _ _ =>
      fun {m} _ _ => fun vector vectors =>
        (do
          let column ← Runtime.Autograd.Model.reshape (m := m) (α := α)
            (s₂ := [width, 1]) vector
            (by simp [_root_.Spec.Shape.size])
          Runtime.Autograd.Model.matmul (m := m) (α := α)
            (batchA := .scalar) (batchB := .scalar) (batch := .scalar)
            (mDim := batch) (nDim := width) (pDim := 1) vectors column >>= fun result =>
          Runtime.Autograd.Model.reshape (m := m) (α := α)
            (s₂ := [batch]) result (by simp [_root_.Spec.Shape.size]) :
          m (Runtime.Autograd.Model.RefTy (m := m) (α := α) [batch])) }

/-- Apply a depthwise weighted reduction independently to every element of a batch.

Both inputs use the layout `batch × width × channels`.  For each batch index and channel, the
result is the sum over `width` of the pointwise product.  The executable program implements that
equation with elementwise multiplication, an axis swap, and a leading-axis reduction; exposing the
operation here avoids making every caller reproduce that layout plumbing.
-/
def batchedDepthwiseWeightedSum (batch width channels : Nat)
    (hBatch : 0 < batch) (hWidth : 0 < width) (hChannels : 0 < channels) :
    PrimOp
      [[batch, width, channels], [batch, width, channels]]
      [batch, channels] :=
  letI : NeZero batch := ⟨Nat.ne_of_gt hBatch⟩
  letI : NeZero width := ⟨Nat.ne_of_gt hWidth⟩
  letI : NeZero channels := ⟨Nat.ne_of_gt hChannels⟩
  letI : _root_.Spec.Shape.HasNonemptyAxis 0
      [width, channels] :=
    _root_.Spec.Shape.hasNonemptyAxisZeroOfPos hWidth
  letI : _root_.Spec.Shape.HasNonemptyAxis 0
      [width, batch, channels] :=
    _root_.Spec.Shape.hasNonemptyAxisZeroOfPos hWidth
  { name := s!"batchedDepthwiseWeightedSum({batch},{width},{channels})"
    specFwd := fun {α} _ _ xs =>
      match xs with
      | .cons values (.cons weights .nil) =>
          .dim fun i => _root_.TorchLean.Tensor.reduceSum (α := α) 0
            (_root_.TorchLean.Tensor.mulSpec
              (_root_.Spec.get values i) (_root_.Spec.get weights i))
            (_root_.Spec.Shape.hasNonemptyAxisZeroOfPos hWidth).proof
    program := fun {α} _ _ =>
      fun {m} _ _ => fun values weights =>
        (do
          let weighted ← Runtime.Autograd.Model.mul (m := m) (α := α)
            (s := [batch, width, channels]) values weights
          let tapsFirst : Runtime.Autograd.Model.RefTy (m := m) (α := α)
              [width, batch, channels] ←
            Runtime.Autograd.Model.swapAdjacentAtDepth (m := m) (α := α) 0 weighted
          Runtime.Autograd.Model.reduceSum (m := m) (α := α) 0 tapsFirst :
          m (Runtime.Autograd.Model.RefTy (m := m) (α := α)
            [batch, channels])) }

/-- Pure evaluation of a batched depthwise weighted sum. -/
@[simp] theorem batchedDepthwiseWeightedSum_specFwd
    {batch width channels : Nat}
    (hBatch : 0 < batch) (hWidth : 0 < width) (hChannels : 0 < channels)
    {α : Type} [TorchLean.Storage α] [Context α]
    (values weights :
      _root_.TorchLean.Tensor α [batch, width, channels]) :
    (batchedDepthwiseWeightedSum batch width channels hBatch hWidth hChannels).specFwd
        (.cons values (.cons weights .nil)) =
      .dim (fun i => _root_.TorchLean.Tensor.reduceSum 0
        (_root_.TorchLean.Tensor.mulSpec (_root_.Spec.get values i) (_root_.Spec.get weights i))
        (_root_.Spec.Shape.hasNonemptyAxisZeroOfPos hWidth).proof) := by
  rfl


/-- Form one outer product for every pair of vectors in a batch. -/
def batchedOuter (batch rows columns : Nat) :
    PrimOp
      [[batch, rows], [batch, columns]]
      [batch, rows, columns] :=
  { name := s!"batchedOuter({batch},{rows},{columns})"
    specFwd := fun {α} _ _ xs =>
      match xs with
      | .cons left (.cons right .nil) =>
          .dim fun i => _root_.Spec.outerProductSpec (α := α)
            (_root_.Spec.get left i) (_root_.Spec.get right i)
    program := fun {α} _ _ =>
      fun {m} _ _ => fun left right =>
        (do
          let columns' ← Runtime.Autograd.Model.reshape (m := m) (α := α)
            (s₂ := [batch, rows, 1]) left
            (by simp [_root_.Spec.Shape.size])
          let rows' ← Runtime.Autograd.Model.reshape (m := m) (α := α)
            (s₂ := [batch, 1, columns]) right
            (by simp [_root_.Spec.Shape.size])
          Runtime.Autograd.Model.matmul (m := m) (α := α)
            (batchA := [batch]) (batchB := [batch]) (batch := [batch])
            (mDim := rows) (nDim := 1) (pDim := columns)
            columns' rows' :
          m (Runtime.Autograd.Model.RefTy (m := m) (α := α)
            [batch, rows, columns])) }

/-- Pure evaluation of pairwise batched outer products. -/
@[simp] theorem batchedOuter_specFwd {batch rows columns : Nat}
    {α : Type} [TorchLean.Storage α] [Context α]
    (left : _root_.TorchLean.Tensor α [batch, rows])
    (right : _root_.TorchLean.Tensor α [batch, columns]) :
    (batchedOuter batch rows columns).specFwd (.cons left (.cons right .nil)) =
      .dim (fun i => _root_.Spec.outerProductSpec
        (_root_.Spec.get left i) (_root_.Spec.get right i)) := by
  rfl

/-- Scale every matrix row by the corresponding coordinate of a batched vector. -/
def batchedRowScale (batch rows columns : Nat) :
    PrimOp [[batch, rows], [batch, rows, columns]] [batch, rows, columns] :=
  { name := s!"batchedRowScale({batch},{rows},{columns})"
    specFwd := fun {α} _ _ xs =>
      match xs with
      | .cons scales (.cons matrices .nil) =>
          .dim fun i => .dim fun row => .dim fun column => .scalar <|
            _root_.TorchLean.Tensor.getScalar (_root_.Spec.get scales i) row *
              _root_.Spec.get2 (_root_.Spec.get matrices i) row column
    program := fun {α} _ _ =>
      fun {m} _ _ => fun scales matrices =>
        (do
          let columns' ← Runtime.Autograd.Model.reshape (m := m) (α := α)
            (s₂ := [batch, rows, 1]) scales
            (by simp [_root_.Spec.Shape.size])
          let expanded ← Runtime.Autograd.Model.broadcastTo (m := m) (α := α)
            (s₂ := [batch, rows, columns])
            (_root_.Spec.Shape.CanBroadcastTo.dim_eq
              (_root_.Spec.Shape.CanBroadcastTo.dim_eq
                (_root_.Spec.Shape.CanBroadcastTo.dim_1_to_n
                  _root_.Spec.Shape.CanBroadcastTo.scalar))) columns'
          Runtime.Autograd.Model.mul (m := m) (α := α) expanded matrices :
          m (Runtime.Autograd.Model.RefTy (m := m) (α := α)
            [batch, rows, columns])) }

/-- Pure evaluation of batched row scaling. -/
@[simp] theorem batchedRowScale_specFwd {batch rows columns : Nat}
    {α : Type} [TorchLean.Storage α] [Context α]
    (scales : _root_.TorchLean.Tensor α [batch, rows])
    (matrices : _root_.TorchLean.Tensor α [batch, rows, columns]) :
    (batchedRowScale batch rows columns).specFwd (.cons scales (.cons matrices .nil)) =
      .dim (fun i => .dim (fun row => .dim (fun column => .scalar
        (_root_.TorchLean.Tensor.getScalar (_root_.Spec.get scales i) row *
          _root_.Spec.get2 (_root_.Spec.get matrices i) row column)))) := by
  rfl

/-- Scale every tensor in a batch by its corresponding scalar coefficient.

The trailing tensor shape is unrestricted. This one primitive therefore covers vectors, matrices,
and higher-rank values without introducing rank-specific operation names. -/
def batchedScale (batch : Nat) (elementShape : Shape) :
    PrimOp [[batch], .dim batch elementShape] (.dim batch elementShape) :=
  { name := s!"batchedScale({batch})"
    specFwd := fun {_} _storage _ctx xs =>
      match xs with
      | .cons coefficients (.cons values .nil) =>
          .dim fun i => _root_.TorchLean.Tensor.mulSpec
            (_root_.TorchLean.Tensor.full elementShape
              (_root_.TorchLean.Tensor.item (_root_.Spec.get coefficients i)))
            (_root_.Spec.get values i)
    program := fun {α} _ _ =>
      fun {m} _ _ => fun coefficients values =>
        (do
          let coefficientShape : Shape :=
            .dim batch (_root_.Spec.Shape.singletonAxes elementShape)
          let coefficients' ← Runtime.Autograd.Model.reshape (m := m) (α := α)
            (s₁ := [batch]) (s₂ := coefficientShape) coefficients (by
              simp [coefficientShape, _root_.Spec.Shape.size])
          let _ : _root_.Spec.Shape.SameRank
              (_root_.Spec.Shape.singletonAxes elementShape) elementShape :=
            ⟨_root_.Spec.Shape.rank_singletonAxes elementShape⟩
          let expanded ← Runtime.Autograd.Model.broadcastTo (m := m) (α := α)
            (s₁ := coefficientShape) (s₂ := .dim batch elementShape)
            (_root_.Spec.Shape.CanBroadcastTo.dim_eq
              (_root_.Spec.Shape.CanBroadcastTo.singletonAxes elementShape)) coefficients'
          Runtime.Autograd.Model.mul (m := m) (α := α) expanded values :
          m (Runtime.Autograd.Model.RefTy (m := m) (α := α)
            (.dim batch elementShape))) }

/-- Pure evaluation of one scalar coefficient per batched tensor. -/
@[simp] theorem batchedScale_specFwd {batch : Nat} {elementShape : Shape}
    {α : Type} [TorchLean.Storage α] [Context α]
    (coefficients : _root_.TorchLean.Tensor α [batch])
    (values : _root_.TorchLean.Tensor α (.dim batch elementShape)) :
    (batchedScale batch elementShape).specFwd (.cons coefficients (.cons values .nil)) =
      .dim (fun i => _root_.TorchLean.Tensor.mulSpec
        (_root_.TorchLean.Tensor.full elementShape
          (_root_.TorchLean.Tensor.item (_root_.Spec.get coefficients i)))
        (_root_.Spec.get values i)) := by
  rfl


/-- Multiply each row of a matrix by the corresponding vector coordinate. -/
def rowScale (rows columns : Nat) :
    PrimOp
      [[rows], [rows, columns]]
      [rows, columns] :=
  { name := s!"rowScale({rows},{columns})"
    specFwd := fun {α} _ _ xs =>
      match xs with
      | .cons scales (.cons mtx .nil) =>
          _root_.TorchLean.Tensor.dim fun row => _root_.TorchLean.Tensor.dim fun column =>
            _root_.TorchLean.Tensor.scalar <|
              _root_.TorchLean.Tensor.getScalar scales row * _root_.Spec.get2 mtx row column
    program := fun {α} _ _ =>
      fun {m} _ _ => fun scales matrix =>
        (do
          let column ← Runtime.Autograd.Model.reshape (m := m) (α := α)
            (s₂ := [rows, 1]) scales (by simp [Spec.Shape.size])
          let expanded ← Runtime.Autograd.Model.broadcastTo (m := m) (α := α)
            (s₂ := [rows, columns])
            (_root_.Spec.Shape.CanBroadcastTo.dim_eq
              (_root_.Spec.Shape.CanBroadcastTo.dim_1_to_n
                _root_.Spec.Shape.CanBroadcastTo.scalar)) column
          Runtime.Autograd.Model.mul (m := m) (α := α) expanded matrix :
          m (Runtime.Autograd.Model.RefTy (m := m) (α := α)
            [rows, columns])) }

/-- Form the outer product of two vectors. -/
def outer (rows columns : Nat) :
    PrimOp
      [[rows], [columns]]
      [rows, columns] :=
  { name := s!"outer({rows},{columns})"
    specFwd := fun {α} _ _ xs =>
      match xs with
      | .cons left (.cons right .nil) => _root_.Spec.outerProductSpec (α := α) left right
    program := fun {α} _ _ =>
      fun {m} _ _ => fun left right =>
        (do
          let leftColumn ← Runtime.Autograd.Model.reshape (m := m) (α := α)
            (s₂ := [rows, 1]) left (by simp [Spec.Shape.size])
          let rightRow ← Runtime.Autograd.Model.reshape (m := m) (α := α)
            (s₂ := [1, columns]) right (by simp [Spec.Shape.size])
          Runtime.Autograd.Model.matmul (m := m) (α := α)
            (batchA := .scalar) (batchB := .scalar) (batch := .scalar)
            (mDim := rows) (nDim := 1) (pDim := columns) leftColumn rightRow :
          m (Runtime.Autograd.Model.RefTy (m := m) (α := α)
            [rows, columns])) }

/-- Multiply a tensor by a scalar supplied as a graph input. -/
def scalarMul (s : Shape) : PrimOp [.scalar, s] s :=
  { name := "scalarMul"
    specFwd := fun {_α} _storage _ctx xs =>
      match xs with
      | .cons coefficient (.cons input .nil) =>
          _root_.TorchLean.Tensor.mulSpec
            (_root_.TorchLean.Tensor.full s coefficient.item) input
    program := fun {α} _ _ =>
      fun {m} _ _ => fun scalar input =>
        (do
          let expanded ← Runtime.Autograd.Model.broadcastTo (m := m) (α := α)
            (_root_.Spec.Shape.CanBroadcastTo.scalarTo s) scalar
          Runtime.Autograd.Model.mul (m := m) (α := α) expanded input :
          m (Runtime.Autograd.Model.RefTy (m := m) (α := α) s)) }

/-- Scalar multiplication depends only on the value carried by its scalar-shaped input. -/
@[simp] theorem scalarMul_specFwd {α : Type} [TorchLean.Storage α] [Context α] {s : Shape}
    (coefficient : _root_.TorchLean.Tensor α .scalar) (input : _root_.TorchLean.Tensor α s) :
    (scalarMul s).specFwd (.cons coefficient (.cons input .nil)) =
      _root_.TorchLean.Tensor.mapSpec (fun value => coefficient.item * value) input := by
  exact _root_.TorchLean.Tensor.mulSpec_full_left coefficient.item input


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

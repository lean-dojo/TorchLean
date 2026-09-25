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


namespace Internal

/-- Apply a binary tensor operation independently over matching leading axes. -/
def mapLeading₂ {α : Type} [TorchLean.Storage α] {s t u : Shape}
    (f : Tensor α s → Tensor α t → Tensor α u) :
    (leading : Shape) → Tensor α (leading.concat s) → Tensor α (leading.concat t) →
      Tensor α (leading.concat u)
  | .scalar, x, y => f x y
  | .dim _ rest, x, y =>
      .dim fun i => mapLeading₂ f rest (_root_.Spec.get x i) (_root_.Spec.get y i)

/-- Equal leading axes preserve a broadcast between suffixes of equal rank. -/
theorem prepend_broadcast {s t : Shape} [same : Shape.SameRank s t]
    (leading : Shape) (h : Shape.CanBroadcastTo s t) :
    Shape.CanBroadcastTo (leading.concat s) (leading.concat t) := by
  induction leading with
  | scalar => exact h
  | dim n rest ih =>
      let : Shape.SameRank (rest.concat s) (rest.concat t) :=
        ⟨by simp only [Tensor.LinearAlgebra.Internal.rank_concat, same.rank_eq]⟩
      exact Shape.CanBroadcastTo.dim_eq ih

end Internal

/-- Form outer products independently over an arbitrary common leading shape. -/
def outer (rows columns : Nat) (leadingShape : Shape := .scalar) :
    PrimOp
      [leadingShape.concat [rows], leadingShape.concat [columns]]
      (leadingShape.concat [rows, columns]) :=
  { name := s!"outer({rows},{columns})"
    specFwd := fun {α} _ _ xs =>
      match xs with
      | .cons left (.cons right .nil) =>
          Internal.mapLeading₂ (_root_.Spec.outerProductSpec (α := α)) leadingShape left right
    program := fun {α} _ _ =>
      fun {m} _ _ => fun left right => by
        letI : Shape.BroadcastTo leadingShape leadingShape :=
          ⟨Shape.CanBroadcastTo.refl leadingShape⟩
        exact (do
          let leftColumn ← Runtime.Autograd.Model.reshape (m := m) (α := α)
            (s₂ := leadingShape.concat [rows, 1]) left
            (by simp [Shape.size_concat, Shape.size])
          let rightRow ← Runtime.Autograd.Model.reshape (m := m) (α := α)
            (s₂ := leadingShape.concat [1, columns]) right
            (by simp [Shape.size_concat, Shape.size])
          Runtime.Autograd.Model.matmul (m := m) (α := α)
            (batchA := leadingShape) (batchB := leadingShape) (batch := leadingShape)
            (mDim := rows) (nDim := 1) (pDim := columns) leftColumn rightRow :
          m (Runtime.Autograd.Model.RefTy (m := m) (α := α)
            (leadingShape.concat [rows, columns]))) }

/-- With no leading axes, the primitive forms the ordinary vector outer product. -/
@[simp] theorem outer_specFwd {rows columns : Nat}
    {α : Type} [TorchLean.Storage α] [Context α]
    (left : Tensor α [rows]) (right : Tensor α [columns]) :
    (outer rows columns).specFwd (.cons left (.cons right .nil)) =
      _root_.Spec.outerProductSpec left right := by
  rfl

/-- One leading axis forms a separate outer product for each pair of vectors. -/
@[simp] theorem outer_specFwd_dim {batch rows columns : Nat}
    {α : Type} [TorchLean.Storage α] [Context α]
    (left : Tensor α [batch, rows]) (right : Tensor α [batch, columns]) :
    (outer rows columns [batch]).specFwd (.cons left (.cons right .nil)) =
      .dim (fun i => _root_.Spec.outerProductSpec
        (_root_.Spec.get left i) (_root_.Spec.get right i)) := by
  rfl

/-- Scale each matrix row by its vector coordinate over an arbitrary common leading shape. -/
def rowScale (rows columns : Nat) (leadingShape : Shape := .scalar) :
    PrimOp
      [leadingShape.concat [rows], leadingShape.concat [rows, columns]]
      (leadingShape.concat [rows, columns]) :=
  { name := s!"rowScale({rows},{columns})"
    specFwd := fun {_} _ _ xs =>
      match xs with
      | .cons scales (.cons matrices .nil) =>
          Internal.mapLeading₂
            (fun scale matrix => .dim fun row => .dim fun column => .scalar
              (Tensor.getScalar scale row * _root_.Spec.get2 matrix row column))
            leadingShape scales matrices
    program := fun {α} _ _ =>
      fun {m} _ _ => fun scales matrices =>
        (do
          let column ← Runtime.Autograd.Model.reshape (m := m) (α := α)
            (s₂ := leadingShape.concat [rows, 1]) scales
            (by simp [Shape.size_concat, Shape.size])
          let expanded ← Runtime.Autograd.Model.broadcastTo (m := m) (α := α)
            (s₂ := leadingShape.concat [rows, columns])
            (Internal.prepend_broadcast leadingShape
              (Shape.CanBroadcastTo.dim_eq
                (Shape.CanBroadcastTo.dim_1_to_n Shape.CanBroadcastTo.scalar))) column
          Runtime.Autograd.Model.mul (m := m) (α := α) expanded matrices :
          m (Runtime.Autograd.Model.RefTy (m := m) (α := α)
            (leadingShape.concat [rows, columns]))) }

/-- One leading axis scales each row of each matrix independently. -/
@[simp] theorem rowScale_specFwd_dim {batch rows columns : Nat}
    {α : Type} [TorchLean.Storage α] [Context α]
    (scales : Tensor α [batch, rows]) (matrices : Tensor α [batch, rows, columns]) :
    (rowScale rows columns [batch]).specFwd (.cons scales (.cons matrices .nil)) =
      .dim (fun i => .dim (fun row => .dim (fun column => .scalar
        (Tensor.getScalar (_root_.Spec.get scales i) row *
          _root_.Spec.get2 (_root_.Spec.get matrices i) row column)))) := by
  rfl

/-- Multiply each trailing tensor by the scalar at its coordinate in the leading shape. -/
def scalarMul (s : Shape) (leadingShape : Shape := .scalar) :
    PrimOp [leadingShape, leadingShape.concat s] (leadingShape.concat s) :=
  { name := "scalarMul"
    specFwd := fun {_} _ _ xs =>
      match xs with
      | .cons coefficients (.cons values .nil) =>
          Internal.mapLeading₂
            (fun coefficient value => Tensor.mulSpec (Tensor.full s coefficient.item) value)
            leadingShape (coefficients.castShape (Shape.concat_scalar leadingShape).symm) values
    program := fun {α} _ _ =>
      fun {m} _ _ => fun coefficients values =>
        (do
          let coefficientShape := leadingShape.concat (Shape.singletonAxes s)
          let coefficients' ← Runtime.Autograd.Model.reshape (m := m) (α := α)
            (s₂ := coefficientShape) coefficients (by simp [coefficientShape, Shape.size_concat])
          let _ : Shape.SameRank (Shape.singletonAxes s) s := ⟨Shape.rank_singletonAxes s⟩
          let expanded ← Runtime.Autograd.Model.broadcastTo (m := m) (α := α)
            (s₂ := leadingShape.concat s)
            (Internal.prepend_broadcast leadingShape (Shape.CanBroadcastTo.singletonAxes s))
            coefficients'
          Runtime.Autograd.Model.mul (m := m) (α := α) expanded values :
          m (Runtime.Autograd.Model.RefTy (m := m) (α := α) (leadingShape.concat s))) }

/-- Scalar multiplication depends only on the value carried by its scalar-shaped input. -/
@[simp] theorem scalarMul_specFwd {α : Type} [TorchLean.Storage α] [Context α] {s : Shape}
    (coefficient : Tensor α .scalar) (input : Tensor α s) :
    (scalarMul s).specFwd (.cons coefficient (.cons input .nil)) =
      Tensor.mapSpec (fun value => coefficient.item * value) input := by
  exact Tensor.mulSpec_full_left coefficient.item input

/-- One leading axis assigns one scalar coefficient to each trailing tensor. -/
@[simp] theorem scalarMul_specFwd_dim {batch : Nat} {elementShape : Shape}
    {α : Type} [TorchLean.Storage α] [Context α]
    (coefficients : Tensor α [batch]) (values : Tensor α (.dim batch elementShape)) :
    (scalarMul elementShape [batch]).specFwd (.cons coefficients (.cons values .nil)) =
      .dim (fun i => Tensor.mulSpec
        (Tensor.full elementShape (Tensor.item (_root_.Spec.get coefficients i)))
        (_root_.Spec.get values i)) := by
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

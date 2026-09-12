/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape.Broadcasting
public import NN.Spec.Core.TensorReductionShape.Reductions
public import NN.Spec.Core.Tensor.Linalg

@[expose] public section


open Spec TorchLean

namespace TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]

/-!
# Linear Algebra Helpers

Rank-polymorphic axis permutations, broadcasted matmul, and shape matching.
-/

/-- Swap adjacent tensor axes at `depth` and `depth + 1`. -/
def swapAdjacentAxes {β : Type} [TorchLean.Storage β]
    {shape : Shape} (tensor : Tensor β shape) (depth : Nat) :
    Tensor β (shape.swapAdjacentAtDepth depth) :=
  match depth, shape with
  | 0, .dim _ (.dim _ _) =>
      Tensor.dim fun j =>
        Tensor.dim fun i =>
          Tensor.unstack (Tensor.unstack tensor i) j
  | depth + 1, .dim _ _ =>
      Tensor.dim fun i => swapAdjacentAxes (Tensor.unstack tensor i) depth
  | depth, .scalar => by
      cases depth <;> exact tensor
  | 0, .dim _ .scalar => tensor

/-- Apply adjacent-axis swaps while retaining the resulting shape in the return type. -/
def permuteByAdjacentSwaps {β : Type} [TorchLean.Storage β]
    {s : Shape} (tensor : Tensor β s) :
    (depths : List Nat) → Tensor β (s.applyAdjacentSwaps depths)
  | [] => tensor
  | depth :: depths =>
      permuteByAdjacentSwaps (swapAdjacentAxes tensor depth) depths

/-- Swapping at depth zero exchanges the two leading axes. -/
theorem swapAdjacentAxes_zero {β : Type} [TorchLean.Storage β]
    {m n : Nat} {s : Shape}
    (tensor : Tensor β (.dim m (.dim n s))) :
    swapAdjacentAxes tensor 0 =
      .dim (fun j => .dim (fun i => Spec.get (Spec.get tensor i) j)) := by
  rfl

namespace LinearAlgebra
namespace Internal

/-- Concatenated shapes add their ranks. -/
theorem rank_concat (left right : Shape) :
    (left.concat right).rank = left.rank + right.rank := by
  induction left with
  | scalar => simp [Shape.rank]
  | dim _ tail ih =>
      simp only [Shape.rank, ih]
      grind

/-- Appending the same suffix preserves equal ranks. -/
theorem sameRank_concat_right (left right suffix : Shape) (same : Shape.SameRank left right) :
    Shape.SameRank (left.concat suffix) (right.concat suffix) :=
  ⟨by simp only [rank_concat, same.rank_eq]⟩

/-- Extend prefix-broadcast evidence across a fixed non-broadcasted tensor suffix. -/
theorem extendBroadcastSuffix {source target : Shape} (suffix : Shape)
    (broadcast : Shape.CanBroadcastTo source target) :
    Shape.CanBroadcastTo (source.concat suffix) (target.concat suffix) := by
  induction target generalizing source with
  | scalar =>
      cases source with
      | scalar => exact Shape.CanBroadcastTo.refl suffix
      | dim _ _ => exact absurd broadcast Shape.not_canBroadcastTo_dim_scalar
  | dim n target ih =>
      cases source with
      | scalar => exact (ih (Shape.canBroadcastTo_scalar_dim.mp broadcast)).expand_dims
      | dim m source =>
          by_cases hRank : source.rank = target.rank
          · obtain ⟨hHead, hTail⟩ := (Shape.canBroadcastTo_dim_dim_of_rank_eq hRank).mp broadcast
            exact (Shape.canBroadcastTo_dim_dim_of_rank_eq
              (by simp only [rank_concat, hRank])).mpr ⟨hHead, ih hTail⟩
          · exact (ih ((Shape.canBroadcastTo_dim_dim_of_rank_ne hRank).mp broadcast)).expand_dims

/-- A batch prefix followed by a matrix has the size of the flattened batch matrix. -/
theorem flattenBatchMatrix_size (batch : Shape) (m n : Nat) :
    (batch.concat [m, n]).size = Shape.size [batch.size, m, n] := by
  simp [Shape.size_concat, Shape.size]

/-- Multiply matrices that already share a common batch prefix. -/
def matmulCommonBatchSpec {α : Type} [TorchLean.Storage α]
    [Add α] [Mul α] [Zero α]
    {batch : Shape} {m n p : Nat}
    (A : Tensor α (batch.concat [m, n])) (B : Tensor α (batch.concat [n, p])) :
    Tensor α (batch.concat [m, p]) :=
  match batch with
  | .scalar => matMulSpec A B
  | .dim _ _ =>
      Tensor.dim fun index =>
        matmulCommonBatchSpec
          (Tensor.unstack A index) (Tensor.unstack B index)

/-- With no batch axes left, batched matmul is plain matmul. -/
@[simp] theorem matmulCommonBatchSpec_scalar {α : Type}
    [TorchLean.Storage α]
    [Add α] [Mul α] [Zero α] {m n p : Nat}
    (left : Tensor α [m, n]) (right : Tensor α [n, p]) :
    matmulCommonBatchSpec (batch := .scalar) left right = matMulSpec left right := by
  rfl

/-- Batched matmul over an outer axis is the batched matmul of each pair of slices. -/
@[simp] theorem matmulCommonBatchSpec_dim {α : Type}
    [TorchLean.Storage α]
    [Add α] [Mul α] [Zero α]
    {count m n p : Nat} {rest : Shape}
    (left : Fin count → Tensor α (rest.concat [m, n]))
    (right : Fin count → Tensor α (rest.concat [n, p])) :
    matmulCommonBatchSpec (batch := .dim count rest) (.dim left) (.dim right) =
      .dim (fun index => matmulCommonBatchSpec (left index) (right index)) := by
  simp [matmulCommonBatchSpec]

end Internal
end LinearAlgebra

/-- Matrix-rank matmul with explicit broadcasting of both batch prefixes.

`A` has shape `batchA ++ [m, n]`, `B` has shape `batchB ++ [n, p]`, and both batch
prefixes broadcast to `batch`. The result has shape `batch ++ [m, p]`. -/
def matmulSpec {α : Type} [TorchLean.Storage α]
    [Add α] [Mul α] [Zero α]
    {batchA batchB batch : Shape} {m n p : Nat}
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch)
    (A : Tensor α (batchA.concat [m, n])) (B : Tensor α (batchB.concat [n, p])) :
    Tensor α (batch.concat [m, p]) :=
  let commonA := broadcastTo (LinearAlgebra.Internal.extendBroadcastSuffix [m, n] broadcastA) A
  let commonB := broadcastTo (LinearAlgebra.Internal.extendBroadcastSuffix [n, p] broadcastB) B
  LinearAlgebra.Internal.matmulCommonBatchSpec commonA commonB

/-- Reverse-mode derivatives for matrix-rank matmul with broadcasted batch prefixes. -/
def matmulBackwardSpec {α : Type} [TorchLean.Storage α]
    [Add α] [Mul α] [Zero α]
    {batchA batchB batch : Shape} {m n p : Nat}
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch)
    (A : Tensor α (batchA.concat [m, n])) (B : Tensor α (batchB.concat [n, p]))
    (dC : Tensor α (batch.concat [m, p])) :
    Tensor α (batchA.concat [m, n]) × Tensor α (batchB.concat [n, p]) :=
  let commonA := broadcastTo (LinearAlgebra.Internal.extendBroadcastSuffix [m, n] broadcastA) A
  let commonB := broadcastTo (LinearAlgebra.Internal.extendBroadcastSuffix [n, p] broadcastB) B
  let commonBT : Tensor α (batch.concat [p, n]) := by
    simpa using swapAdjacentAxes commonB batch.rank
  let commonAT : Tensor α (batch.concat [n, m]) := by
    simpa using swapAdjacentAxes commonA batch.rank
  let dCommonA := LinearAlgebra.Internal.matmulCommonBatchSpec dC commonBT
  let dCommonB := LinearAlgebra.Internal.matmulCommonBatchSpec commonAT dC
  let dA :=
    reduceFromBroadcastTo
      (LinearAlgebra.Internal.extendBroadcastSuffix [m, n] broadcastA) dCommonA
  let dB :=
    reduceFromBroadcastTo
      (LinearAlgebra.Internal.extendBroadcastSuffix [n, p] broadcastB) dCommonB
  (dA, dB)

end TorchLean.Tensor

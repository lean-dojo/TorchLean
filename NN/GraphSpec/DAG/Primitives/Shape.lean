/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.DAG.Syntax
public import NN.Runtime.Autograd.Model.Functional.Einsum

/-!
# DAG Shape and Reduction Primitives

Reductions, transposes, gathers, broadcasting, concatenation, slicing, and shape-preserving views.
-/

@[expose] public section

namespace NN
namespace GraphSpec
namespace DAG

open _root_.Spec _root_.TorchLean
open TorchLean.Tensor
open _root_.TorchLean.Tensor

namespace PrimOp

/-- Sum a statically nonempty tensor axis. -/
def reduceSum (s : Shape) (axis : Nat) [_root_.Spec.Shape.HasNonemptyAxis axis s]
    [_root_.Spec.Shape.WellFormed s] : PrimOp [s] (_root_.TorchLean.Tensor.shapeAfterSum s axis) :=
  { name := s!"reduceSum(axis={axis})"
    specFwd := fun {α} _ _ xs =>
      match xs with
      | .cons input .nil =>
          _root_.TorchLean.Tensor.reduceSum (α := α) axis input
            (inferInstance : _root_.Spec.Shape.HasNonemptyAxis axis s).proof
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input =>
        Runtime.Autograd.Model.reduceSum (m := m) (α := α) axis input }

/-- Pure evaluation of an axis sum. -/
@[simp] theorem reduceSum_specFwd {s : Shape} {axis : Nat}
    [_root_.Spec.Shape.HasNonemptyAxis axis s] [_root_.Spec.Shape.WellFormed s]
    {α : Type} [TorchLean.Storage α] [Context α] (input : _root_.TorchLean.Tensor α s) :
    (reduceSum s axis).specFwd (.cons input .nil) =
      _root_.TorchLean.Tensor.reduceSum (α := α) axis input
        (inferInstance : _root_.Spec.Shape.HasNonemptyAxis axis s).proof := by
  rfl

/-- Average a statically nonempty tensor axis. -/
def reduceMean (s : Shape) (axis : Nat) [_root_.Spec.Shape.HasNonemptyAxis axis s]
    [_root_.Spec.Shape.WellFormed s] : PrimOp [s] (_root_.TorchLean.Tensor.shapeAfterSum s axis) :=
  { name := s!"reduceMean(axis={axis})"
    specFwd := fun {α} _ _ xs =>
      match xs with
      | .cons input .nil =>
          _root_.TorchLean.Tensor.reduceMean (α := α) axis input
            (inferInstance : _root_.Spec.Shape.HasNonemptyAxis axis s).proof
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input =>
        Runtime.Autograd.Model.reduceMean (m := m) (α := α) axis input }

/-- Pure evaluation of an axis mean. -/
@[simp] theorem reduceMean_specFwd {s : Shape} {axis : Nat}
    [_root_.Spec.Shape.HasNonemptyAxis axis s] [_root_.Spec.Shape.WellFormed s]
    {α : Type} [TorchLean.Storage α] [Context α] (input : _root_.TorchLean.Tensor α s) :
    (reduceMean s axis).specFwd (.cons input .nil) =
      _root_.TorchLean.Tensor.reduceMean (α := α) axis input
        (inferInstance : _root_.Spec.Shape.HasNonemptyAxis axis s).proof := by
  rfl

/-- Swap adjacent tensor axes at a statically chosen depth. -/
def swapAdjacentAtDepth (s : Shape) (depth : Nat) :
    PrimOp [s] (s.swapAdjacentAtDepth depth) :=
  { name := s!"swapAdjacentAtDepth({depth})"
    specFwd := fun {_α} _storage _ctx xs =>
      match xs with
      | .cons input .nil => _root_.TorchLean.Tensor.swapAdjacentAxes input depth
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input =>
        Runtime.Autograd.Model.swapAdjacentAtDepth (m := m) (α := α) depth input }

/-- Pure evaluation of an adjacent-axis swap. -/
@[simp] theorem swapAdjacentAtDepth_specFwd {α : Type} [TorchLean.Storage α] [Context α]
    (s : Shape) (depth : Nat) (input : _root_.TorchLean.Tensor α s) :
    (swapAdjacentAtDepth s depth).specFwd (.cons input .nil) =
      _root_.TorchLean.Tensor.swapAdjacentAxes input depth := by
  rfl

/-- Select one bounded coordinate along an arbitrary tensor axis. -/
def select (shape : Shape) (axis : Nat) [Shape.AxisInBounds axis shape]
    (index : Fin (shape.axisSize axis)) : PrimOp [shape] (shape.eraseAxis axis) :=
  { name := s!"select(axis={axis},index={index.val})"
    specFwd := fun {_α} _storage _ctx xs =>
      match xs with
      | .cons input .nil => _root_.TorchLean.Tensor.selectSpec axis input index
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input =>
        Runtime.Autograd.Model.select (m := m) (α := α) axis input index }

/-- Expand singleton or missing dimensions according to a checked broadcasting derivation. -/
def broadcast {source target : Shape} (proof : source.CanBroadcastTo target) :
    PrimOp [source] target :=
  { name := "broadcast"
    specFwd := fun {α} _ _ xs =>
      match xs with
      | .cons input .nil => _root_.TorchLean.Tensor.broadcastTo (α := α) proof input
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input =>
        Runtime.Autograd.Model.broadcastTo (m := m) (α := α) proof input }

/-- Moving a replacement extent from the front to `axis` realizes `replaceAxis`. -/
theorem applyAdjacentSwaps_range_eq_replaceAxis
    (shape : Shape) (axis extent : Nat) (hAxis : axis < shape.rank) :
    (Shape.dim extent (shape.eraseAxis axis)).applyAdjacentSwaps (List.range axis) =
      shape.replaceAxis axis extent := by
  have range_succ_eq_zero_cons_map_succ (n : Nat) :
      List.range (n + 1) = 0 :: (List.range n).map Nat.succ := by
    induction n with
    | zero => rfl
    | succ n ih =>
        calc
          List.range (n.succ + 1) = List.range (n + 1) ++ [n + 1] := by
            rw [show n.succ + 1 = (n + 1) + 1 by grind, List.range_succ]
          _ = (0 :: (List.range n).map Nat.succ) ++ [n + 1] := by rw [ih]
          _ = 0 :: (List.range (n + 1)).map Nat.succ := by
            rw [List.range_succ, List.map_append]
            rfl
  have applyAdjacentSwaps_dim_map_succ
      (outer : Nat) (s : Shape) (depths : List Nat) :
      (Shape.dim outer s).applyAdjacentSwaps (depths.map Nat.succ) =
        Shape.dim outer (s.applyAdjacentSwaps depths) := by
    induction depths generalizing s with
    | nil => rfl
    | cons depth depths ih =>
        simp only [List.map_cons, Shape.applyAdjacentSwaps,
          Shape.swapAdjacentAtDepth, ih]
  induction shape generalizing axis with
  | scalar => simp [Shape.rank] at hAxis
  | dim outer rest ih =>
      cases axis with
      | zero => rfl
      | succ axis =>
          have hInner : axis < rest.rank := by
            simp only [Shape.rank] at hAxis
            grind
          rw [range_succ_eq_zero_cons_map_succ]
          simp only [Shape.eraseAxis, Shape.applyAdjacentSwaps,
            Shape.swapAdjacentAtDepth, applyAdjacentSwaps_dim_map_succ]
          rw [ih axis hInner]
          rfl

/-- Replacing an in-bounds axis by its existing extent leaves the shape unchanged. -/
@[simp] theorem replaceAxis_axisSize (shape : Shape) (axis : Nat)
    [hAxis : Shape.AxisInBounds axis shape] :
    shape.replaceAxis axis (shape.axisSize axis) = shape := by
  induction shape generalizing axis with
  | scalar => exact (Nat.not_lt_zero axis hAxis.proof).elim
  | dim outer rest ih =>
      cases axis with
      | zero => rfl
      | succ axis =>
          have innerAxis : Shape.AxisInBounds axis rest :=
            ⟨by
              have := hAxis.proof
              simp only [Shape.rank] at this
              grind⟩
          simp only [Shape.replaceAxis,
            @Shape.axisSize_succ outer rest axis innerAxis hAxis]
          rw [@ih axis innerAxis]

/-- Concatenate tensors along an arbitrary statically valid axis. -/
def concatAxisSpec {α : Type} [TorchLean.Storage α]
    (shape : Shape) (axis left right : Nat)
    [Shape.AxisInBounds axis shape]
    (a : _root_.TorchLean.Tensor α (shape.replaceAxis axis left))
    (b : _root_.TorchLean.Tensor α (shape.replaceAxis axis right)) :
    _root_.TorchLean.Tensor α (shape.replaceAxis axis (left + right)) :=
  let swaps := List.range axis
  have axisReplacement (extent : Nat) :
      (Shape.dim extent (shape.eraseAxis axis)).applyAdjacentSwaps swaps =
        shape.replaceAxis axis extent :=
    applyAdjacentSwaps_range_eq_replaceAxis shape axis extent
      (inferInstance : Shape.AxisInBounds axis shape).proof
  let moveToFront (extent : Nat)
      (input : _root_.TorchLean.Tensor α (shape.replaceAxis axis extent)) :
      _root_.TorchLean.Tensor α (.dim extent (shape.eraseAxis axis)) :=
    let input' : _root_.TorchLean.Tensor α
        ((Shape.dim extent (shape.eraseAxis axis)).applyAdjacentSwaps swaps) :=
      (axisReplacement extent).symm ▸ input
    let moved := _root_.TorchLean.Tensor.permuteByAdjacentSwaps input' swaps.reverse
    Shape.applyAdjacentSwaps_reverse (.dim extent (shape.eraseAxis axis)) swaps ▸ moved
  let outputFront := _root_.TorchLean.Tensor.concatAxisSpec .scalar
    (moveToFront left a) (moveToFront right b)
  axisReplacement (left + right) ▸
    _root_.TorchLean.Tensor.permuteByAdjacentSwaps outputFront swaps

/-- Concatenate along `axis`; `shape` supplies every unchanged axis extent. -/
def concatAxis (shape : Shape) (axis left right : Nat)
    [Shape.AxisInBounds axis shape] :
    PrimOp [shape.replaceAxis axis left, shape.replaceAxis axis right]
      (shape.replaceAxis axis (left + right)) :=
  { name := s!"concat(axis={axis},left={left},right={right})"
    specFwd := fun {α} _ _ xs =>
      match xs with
      | .cons a (.cons b .nil) => concatAxisSpec (α := α) shape axis left right a b
    program := fun {α} _ _ =>
      fun {m} _ _ => fun a b =>
        let run : m (Runtime.Autograd.Model.RefTy (m := m) (α := α)
            (shape.replaceAxis axis (left + right))) := do
          let swaps := List.range axis
          have axisReplacement (extent : Nat) :
              (Shape.dim extent (shape.eraseAxis axis)).applyAdjacentSwaps swaps =
                shape.replaceAxis axis extent :=
            applyAdjacentSwaps_range_eq_replaceAxis shape axis extent
              (inferInstance : Shape.AxisInBounds axis shape).proof
          let moveToFront :
              (extent : Nat) →
                Runtime.Autograd.Model.RefTy (m := m) (α := α)
                  (shape.replaceAxis axis extent) →
                m (Runtime.Autograd.Model.RefTy (m := m) (α := α)
                  (.dim extent (shape.eraseAxis axis))) :=
            fun extent input => do
              let input' : Runtime.Autograd.Model.RefTy (m := m) (α := α)
                  ((Shape.dim extent (shape.eraseAxis axis)).applyAdjacentSwaps swaps) :=
                (axisReplacement extent).symm ▸ input
              let moved ← Runtime.Autograd.Model.F.Einsum.permuteBySwapsTyped
                (m := m) (α := α) input' swaps.reverse
              pure (Shape.applyAdjacentSwaps_reverse
                (.dim extent (shape.eraseAxis axis)) swaps ▸ moved)
          let moveFromFront :
              (extent : Nat) →
                Runtime.Autograd.Model.RefTy (m := m) (α := α)
                  (.dim extent (shape.eraseAxis axis)) →
                m (Runtime.Autograd.Model.RefTy (m := m) (α := α)
                  (shape.replaceAxis axis extent)) :=
            fun extent input => do
              let moved ← Runtime.Autograd.Model.F.Einsum.permuteBySwapsTyped
                (m := m) (α := α) input swaps
              pure (axisReplacement extent ▸ moved)
          let aFront ← moveToFront left a
          let bFront ← moveToFront right b
          let outputFront ← Runtime.Autograd.Model.concatLeadingAxis
            (m := m) (α := α) (nDim := left) (mDim := right)
            (s := shape.eraseAxis axis) aFront bFront
          moveFromFront (left + right) outputFront
        run }

/-- The pure meaning of arbitrary-axis concatenation. -/
@[simp] theorem concatAxis_specFwd {shape : Shape} {axis left right : Nat}
    [Shape.AxisInBounds axis shape] {α : Type} [TorchLean.Storage α] [Context α]
    (a : _root_.TorchLean.Tensor α (shape.replaceAxis axis left))
    (b : _root_.TorchLean.Tensor α (shape.replaceAxis axis right)) :
    (concatAxis shape axis left right).specFwd (.cons a (.cons b .nil)) =
      concatAxisSpec shape axis left right a b := by
  rfl

/-- Select a checked contiguous range along an arbitrary statically valid axis. -/
def sliceAxisRange (shape : Shape) (axis start length : Nat)
    [Shape.AxisInBounds axis shape]
    (hRange : start + length ≤ shape.axisSize axis) :
    PrimOp [shape] (shape.replaceAxis axis length) :=
  { name := s!"slice(axis={axis},start={start},length={length})"
    specFwd := fun {α} _ _ xs =>
      match xs with
      | .cons input .nil =>
          _root_.TorchLean.Tensor.sliceAxisRangeSpec (α := α)
            axis input start length hRange
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input =>
        let run : m (Runtime.Autograd.Model.RefTy (m := m) (α := α)
            (shape.replaceAxis axis length)) := do
          let swaps := List.range axis
          have axisReplacement (extent : Nat) :
              (Shape.dim extent (shape.eraseAxis axis)).applyAdjacentSwaps swaps =
                shape.replaceAxis axis extent :=
            applyAdjacentSwaps_range_eq_replaceAxis shape axis extent
              (inferInstance : Shape.AxisInBounds axis shape).proof
          let moveToFront :
              (extent : Nat) →
                Runtime.Autograd.Model.RefTy (m := m) (α := α)
                  (shape.replaceAxis axis extent) →
                m (Runtime.Autograd.Model.RefTy (m := m) (α := α)
                  (.dim extent (shape.eraseAxis axis))) :=
            fun extent input => do
              let input' : Runtime.Autograd.Model.RefTy (m := m) (α := α)
                  ((Shape.dim extent (shape.eraseAxis axis)).applyAdjacentSwaps swaps) :=
                (axisReplacement extent).symm ▸ input
              let moved ← Runtime.Autograd.Model.F.Einsum.permuteBySwapsTyped
                (m := m) (α := α) input' swaps.reverse
              pure (Shape.applyAdjacentSwaps_reverse
                (.dim extent (shape.eraseAxis axis)) swaps ▸ moved)
          let moveFromFront :
              (extent : Nat) →
                Runtime.Autograd.Model.RefTy (m := m) (α := α)
                  (.dim extent (shape.eraseAxis axis)) →
                m (Runtime.Autograd.Model.RefTy (m := m) (α := α)
                  (shape.replaceAxis axis extent)) :=
            fun extent input => do
              let moved ← Runtime.Autograd.Model.F.Einsum.permuteBySwapsTyped
                (m := m) (α := α) input swaps
              pure (axisReplacement extent ▸ moved)
          let total := shape.axisSize axis
          let input' : Runtime.Autograd.Model.RefTy (m := m) (α := α)
              (shape.replaceAxis axis total) :=
            (replaceAxis_axisSize shape axis).symm ▸ input
          let inputFront ← moveToFront total input'
          let outputFront ← Runtime.Autograd.Model.sliceLeadingAxisRange
            (m := m) (α := α) (nDim := total) (s := shape.eraseAxis axis)
            start length hRange inputFront
          moveFromFront length outputFront
        run }

/-- The pure meaning of a checked arbitrary-axis contiguous slice. -/
@[simp] theorem sliceAxisRange_specFwd {shape : Shape} {axis start length : Nat}
    [Shape.AxisInBounds axis shape]
    (hRange : start + length ≤ shape.axisSize axis)
    {α : Type} [TorchLean.Storage α] [Context α] (input : _root_.TorchLean.Tensor α shape) :
    (sliceAxisRange shape axis start length hRange).specFwd (.cons input .nil) =
      _root_.TorchLean.Tensor.sliceAxisRangeSpec axis input start length hRange := by
  rfl


/-- Reshape a tensor while preserving its statically known number of entries. -/
def reshape (source target : Shape) (hSize : source.size = target.size) :
    PrimOp [source] target :=
  { name := "reshape"
    specFwd := fun {α} _ _ xs =>
      match xs with
      | .cons input .nil => _root_.TorchLean.Tensor.reshapeSpec (α := α) input hSize
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input =>
        Runtime.Autograd.Model.reshape (m := m) (α := α) input hSize }

/-- Pure evaluation of a shape-preserving view. -/
@[simp] theorem reshape_specFwd {source target : Shape}
    (hSize : source.size = target.size) {α : Type} [TorchLean.Storage α] [Context α]
    (input : _root_.TorchLean.Tensor α source) :
    (reshape source target hSize).specFwd (.cons input .nil) =
      _root_.TorchLean.Tensor.reshapeSpec input hSize := by
  rfl


end PrimOp

end DAG
end GraphSpec
end NN

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Util.Idx
public import NN.Proofs.Autograd.Runtime.ShapeErasure

/-!
Indexed tensor lookup and an array context for executing algebraic tapes. The correspondence
certificate is erased at runtime; node preparation reads just the tensors that the node saves.
-/

@[expose] public section

namespace Proofs.Autograd.Algebra

open Spec TorchLean

/-- Tensor selection without committing a node to a linked pack representation. -/
@[ext] structure TensorLookup (α : Type) [Storage α] (shapes : List Shape) where
  /-- Select the tensor with the shape certified by its index. -/
  read : {shape : Shape} → Idx shapes shape → Tensor α shape

/-- Present a linked tensor pack through the selection interface. -/
def TensorLookup.ofPack {α : Type} [Storage α] {shapes : List Shape}
    (xs : TorchLean.TensorPack α shapes) : TensorLookup α shapes :=
  ⟨fun idx => getIdx xs idx⟩

namespace TensorLookup

variable {α : Type} [Storage α]

/-- Materialize a reader at an explicit typed boundary. -/
def toPack : {shapes : List Shape} → TensorLookup α shapes → TorchLean.TensorPack α shapes
  | [], _ => .nil
  | _ :: shapes, lookup =>
      .cons (lookup.read ⟨0, rfl⟩)
        (toPack (shapes := shapes) ⟨fun idx => lookup.read ⟨idx.i.succ, idx.h⟩⟩)

@[simp] theorem getIdx_toPack {shapes : List Shape} (lookup : TensorLookup α shapes)
    {shape : Shape} (idx : Idx shapes shape) :
    getIdx lookup.toPack idx = lookup.read idx := by
  induction shapes with
  | nil => exact Fin.elim0 idx.i
  | cons head shapes ih =>
      obtain ⟨i, h⟩ := idx
      cases i using Fin.cases with
      | zero => cases h; rfl
      | succ i =>
          exact ih ⟨fun idx => lookup.read ⟨idx.i.succ, idx.h⟩⟩ ⟨i, h⟩

@[simp] theorem toPack_ofPack {shapes : List Shape} (xs : TorchLean.TensorPack α shapes) :
    (ofPack xs).toPack = xs := by
  apply TorchLean.TensorPack.ext_getIdx
  intro shape idx
  simp only [getIdx_toPack, ofPack]

@[simp] theorem ofPack_toPack {shapes : List Shape} (lookup : TensorLookup α shapes) :
    ofPack lookup.toPack = lookup := by
  apply TensorLookup.ext
  funext shape idx
  exact getIdx_toPack lookup idx

/-- Equal context positions certify equal tensor shapes. -/
theorem shape_eq_of_index_eq {shapes : List Shape} {source target : Shape}
    (left : Idx shapes source) (right : Idx shapes target)
    (same : left.i.val = right.i.val) : source = target :=
  left.h.symm.trans ((congrArg shapes.get (Fin.ext same)).trans right.h)

/-- Replace one typed entry of a reader. -/
def set {shapes : List Shape} (lookup : TensorLookup α shapes)
    {shape : Shape} (idx : Idx shapes shape) (value : Tensor α shape) :
    TensorLookup α shapes :=
  ⟨fun other =>
    if same : idx.i.val = other.i.val then
      Tensor.castShape value (shape_eq_of_index_eq idx other same)
    else
      lookup.read other⟩

@[simp] theorem read_set_self {shapes : List Shape} (lookup : TensorLookup α shapes)
    {shape : Shape} (idx : Idx shapes shape) (value : Tensor α shape) :
    (lookup.set idx value).read idx = value := by
  simp only [set, dite_true]
  rfl

@[simp] theorem read_set_other {shapes : List Shape} (lookup : TensorLookup α shapes)
    {shape otherShape : Shape} (idx : Idx shapes shape) (value : Tensor α shape)
    (other : Idx shapes otherShape) (different : idx.i.val ≠ other.i.val) :
    (lookup.set idx value).read other = lookup.read other := by
  simp only [set, dite_eq_right different]

/-- Erasing shapes removes the transport in a reader update. -/
theorem ofTensor_read_set {shapes : List Shape} (lookup : TensorLookup α shapes)
    {shape otherShape : Shape} (idx : Idx shapes shape) (value : Tensor α shape)
    (other : Idx shapes otherShape) :
    Spec.SomeTensor.ofTensor ((lookup.set idx value).read other) =
      if idx.i.val = other.i.val then Spec.SomeTensor.ofTensor value
      else Spec.SomeTensor.ofTensor (lookup.read other) := by
  by_cases same : idx.i.val = other.i.val
  · simp only [set, dite_eq_left same, ite_eq_left same, Spec.SomeTensor.ofTensor_castShape]
  · simp only [set, dite_eq_right same, ite_eq_right same]

/-- Updating from another reader copies exactly that reader's value at the selected position. -/
theorem read_set_from {shapes : List Shape} (lookup source : TensorLookup α shapes)
    {shape otherShape : Shape} (idx : Idx shapes shape) (other : Idx shapes otherShape) :
    (lookup.set idx (source.read idx)).read other =
      if other.i.val = idx.i.val then source.read other else lookup.read other := by
  by_cases same : other.i.val = idx.i.val
  · have shapes := shape_eq_of_index_eq idx other same.symm
    cases shapes
    have positions : idx = other := by
      cases idx
      cases other
      simp only [Idx.mk.injEq]
      exact Fin.ext same.symm
    subst other
    simp only [read_set_self, ite_true]
  · rw [read_set_other _ _ _ _ (Ne.symm same), ite_eq_right same]

end TensorLookup

/-- Runtime tensor references with an erased certificate of their shapes and order. -/
@[ext] structure TensorContext (α : Type) [Storage α] (shapes : List Shape) where
  /-- Inputs followed by node values in evaluation order. -/
  values : Array (Spec.SomeTensor α)
  /-- Every runtime entry has the shape at the same position in the typed context. -/
  valid : ∃ xs : TorchLean.TensorPack α shapes, values = xs.toShapeErasedArray

namespace TensorContext

variable {α : Type} [Storage α] {shapes : List Shape}

/-- Convert input tensors once, before traversing the graph. -/
def ofPack (xs : TorchLean.TensorPack α shapes) : TensorContext α shapes :=
  ⟨xs.toShapeErasedArray, xs, rfl⟩

/-- Runtime and typed contexts have the same number of entries. -/
theorem size_values (ctx : TensorContext α shapes) : ctx.values.size = shapes.length := by
  obtain ⟨xs, h⟩ := ctx.valid
  simp [h]

private theorem shape_get (ctx : TensorContext α shapes) {shape : Shape}
    (idx : Idx shapes shape) :
    (ctx.values[idx.i.val]'(by rw [size_values]; exact idx.i.isLt)).shape = shape := by
  obtain ⟨xs, h⟩ := ctx.valid
  simpa only [h, TorchLean.TensorPack.get_toShapeErasedArray,
    Spec.SomeTensor.shape_ofTensor] using idx.h

/-- Read a previously checked position in constant time. -/
@[no_expose] def lookup (ctx : TensorContext α shapes) : TensorLookup α shapes :=
  ⟨fun idx =>
    (ctx.values[idx.i.val]'(by rw [size_values]; exact idx.i.isLt)).cast (shape_get ctx idx)⟩

/-- Array lookup agrees with the original typed pack lookup. -/
@[simp] theorem lookup_ofPack (xs : TorchLean.TensorPack α shapes) :
    (ofPack xs).lookup = TensorLookup.ofPack xs := by
  apply TensorLookup.ext
  funext shape idx
  rcases idx with ⟨i, h⟩
  simp only [lookup, ofPack, TensorLookup.ofPack, TorchLean.TensorPack.get_toShapeErasedArray]
  cases h
  rfl

@[simp] theorem ofTensor_lookup (ctx : TensorContext α shapes) {shape : Shape}
    (idx : Idx shapes shape) :
    Spec.SomeTensor.ofTensor (ctx.lookup.read idx) =
      ctx.values[idx.i.val]'(by rw [size_values]; exact idx.i.isLt) := by
  simp only [lookup, Spec.SomeTensor.ofTensor_cast]

/-- Update one tensor reference without copying the context when it is uniquely owned. -/
def set (ctx : TensorContext α shapes) {shape : Shape} (idx : Idx shapes shape)
    (value : Tensor α shape) : TensorContext α shapes :=
  ⟨ctx.values.set idx.i.val (Spec.SomeTensor.ofTensor value)
      (h := by rw [size_values]; exact idx.i.isLt), by
    refine ⟨(ctx.lookup.set idx value).toPack, ?_⟩
    apply Array.ext
    · simp only [Array.size_set, TorchLean.TensorPack.size_toShapeErasedArray, size_values]
    · intro i leftBound rightBound
      have bound : i < ctx.values.size := by
        simpa only [Array.size_set] using leftBound
      let other : Idx shapes (shapes.get ⟨i, by
        simpa only [TorchLean.TensorPack.size_toShapeErasedArray] using rightBound⟩) :=
        ⟨⟨i, by simpa only [TorchLean.TensorPack.size_toShapeErasedArray] using rightBound⟩, rfl⟩
      rw [Array.getElem_set,
        TorchLean.TensorPack.get_toShapeErasedArray _ other.i]
      change (if idx.i.val = i then Spec.SomeTensor.ofTensor value else ctx.values[i]) =
        Spec.SomeTensor.ofTensor (getIdx (ctx.lookup.set idx value).toPack other)
      rw [TensorLookup.getIdx_toPack, TensorLookup.ofTensor_read_set, ofTensor_lookup]
      ⟩

@[simp] theorem lookup_set (ctx : TensorContext α shapes) {shape : Shape}
    (idx : Idx shapes shape) (value : Tensor α shape) :
    (ctx.set idx value).lookup = ctx.lookup.set idx value := by
  apply TensorLookup.ext
  funext otherShape other
  have same :
      Spec.SomeTensor.ofTensor ((ctx.set idx value).lookup.read other) =
        Spec.SomeTensor.ofTensor ((ctx.lookup.set idx value).read other) := by
    rw [ofTensor_lookup, TensorLookup.ofTensor_read_set, ofTensor_lookup]
    simp only [set, Array.getElem_set]
  simpa only [Spec.SomeTensor.ofTensor, Spec.SomeTensor.mk.injEq, heq_eq_eq, true_and] using same

/-- Transport the type index without changing the runtime array. -/
def cast {other : List Shape} (h : shapes = other) (ctx : TensorContext α shapes) :
    TensorContext α other :=
  h ▸ ctx

@[simp] theorem cast_ofPack {other : List Shape} (h : shapes = other)
    (xs : TorchLean.TensorPack α shapes) :
    cast h (ofPack xs) = ofPack (TorchLean.TensorPack.cast h xs) := by
  cases h
  rfl

/-- Append a tensor reference; prepared nodes must release their reader before this push. -/
def push {shape : Shape} (ctx : TensorContext α shapes) (x : Tensor α shape) :
    TensorContext α (shapes ++ [shape]) :=
  ⟨ctx.values.push (Spec.SomeTensor.ofTensor x), by
    obtain ⟨xs, h⟩ := ctx.valid
    exact ⟨xs.snoc x, by simp [h]⟩⟩

@[simp] theorem push_ofPack {shape : Shape} (xs : TorchLean.TensorPack α shapes)
    (x : Tensor α shape) :
    (ofPack xs).push x = ofPack (xs.snoc x) := by
  apply TensorContext.ext
  simp [push, ofPack]

private theorem shape_back {shape : Shape} (ctx : TensorContext α (shapes ++ [shape])) :
    (ctx.values.back (by rw [size_values]; simp)).shape = shape := by
  obtain ⟨xs, same⟩ := ctx.valid
  have values : ctx.values = xs.unsnoc.1.toShapeErasedArray.push
      (Spec.SomeTensor.ofTensor xs.unsnoc.2) := by
    rw [same, ← TorchLean.TensorPack.toShapeErasedArray_snoc, TorchLean.TensorPack.snoc_unsnoc]
  simp only [values, Array.back, Array.size_push, Nat.add_sub_cancel,
    Array.getElem_push_eq, Spec.SomeTensor.shape_ofTensor]

/-- Remove the last reference in constant time, preserving the typed prefix. -/
@[no_expose] def pop {shape : Shape} (ctx : TensorContext α (shapes ++ [shape])) :
    TensorContext α shapes × Tensor α shape :=
  (⟨ctx.values.pop, by
    obtain ⟨xs, same⟩ := ctx.valid
    refine ⟨xs.unsnoc.1, ?_⟩
    rw [same, ← TorchLean.TensorPack.snoc_unsnoc xs,
      TorchLean.TensorPack.toShapeErasedArray_snoc, Array.pop_push]
    simp only [TorchLean.TensorPack.unsnoc_snoc]⟩,
   (ctx.values.back (by rw [size_values]; simp)).cast (shape_back ctx))

@[simp] theorem pop_ofPack_snoc {shape : Shape} (xs : TorchLean.TensorPack α shapes)
    (value : Tensor α shape) :
    pop (ofPack (xs.snoc value)) = (ofPack xs, value) := by
  apply Prod.ext
  · apply TensorContext.ext
    simp only [pop, ofPack, TorchLean.TensorPack.toShapeErasedArray_snoc, Array.pop_push]
  · simp only [pop, ofPack, TorchLean.TensorPack.toShapeErasedArray_snoc,
      Array.back, Array.size_push, Nat.add_sub_cancel, Array.getElem_push_eq]
    rfl

@[simp] theorem pop_ofPack {shape : Shape}
    (xs : TorchLean.TensorPack α (shapes ++ [shape])) :
    pop (ofPack xs) = (ofPack xs.unsnoc.1, xs.unsnoc.2) := by
  conv_lhs => rw [← TorchLean.TensorPack.snoc_unsnoc xs]
  exact pop_ofPack_snoc _ _

private theorem decode_ok (ctx : TensorContext α shapes) :
    ∃ xs, TorchLean.TensorPack.ofShapeErasedArray ctx.values (shapes := shapes) = .ok xs := by
  obtain ⟨xs, h⟩ := ctx.valid
  exact ⟨xs, by simp [h]⟩

/-- Recover the public typed result once at the execution boundary. -/
def toPack (ctx : TensorContext α shapes) : TorchLean.TensorPack α shapes :=
  match h : TorchLean.TensorPack.ofShapeErasedArray ctx.values (shapes := shapes) with
  | .ok xs => xs
  | .error _ => False.elim (by
      obtain ⟨xs, hx⟩ := decode_ok ctx
      rw [h] at hx
      contradiction)

@[simp] theorem toPack_ofPack (xs : TorchLean.TensorPack α shapes) :
    (ofPack xs).toPack = xs := by
  unfold toPack
  split
  next ys h =>
    simpa only [ofPack, TorchLean.TensorPack.ofShapeErasedArray_toShapeErasedArray,
      Except.ok.injEq] using h.symm
  next msg h =>
    simp only [ofPack, TorchLean.TensorPack.ofShapeErasedArray_toShapeErasedArray] at h
    cases h

@[simp] theorem ofPack_toPack (ctx : TensorContext α shapes) :
    ofPack ctx.toPack = ctx := by
  obtain ⟨xs, values⟩ := ctx.valid
  have same : ctx = ofPack xs := TensorContext.ext values
  rw [same, toPack_ofPack]

@[simp] theorem lookup_toPack (ctx : TensorContext α shapes) :
    TensorLookup.ofPack ctx.toPack = ctx.lookup := by
  rw [← lookup_ofPack, ofPack_toPack]

@[simp] theorem toPack_pop {shape : Shape} (ctx : TensorContext α (shapes ++ [shape])) :
    ((ctx.pop).1.toPack, (ctx.pop).2) = ctx.toPack.unsnoc := by
  conv_lhs => rw [← ofPack_toPack ctx]
  simp only [pop_ofPack, toPack_ofPack]

@[simp] theorem toPack_cast {other : List Shape} (same : shapes = other)
    (ctx : TensorContext α shapes) :
    (ctx.cast same).toPack = TorchLean.TensorPack.cast same ctx.toPack := by
  cases same
  rfl

@[simp] theorem toPack_push {shape : Shape} (ctx : TensorContext α shapes)
    (value : Tensor α shape) :
    (ctx.push value).toPack = ctx.toPack.snoc value := by
  conv_lhs => rw [← ofPack_toPack ctx]
  simp only [push_ofPack, toPack_ofPack]

end TensorContext

end Proofs.Autograd.Algebra

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Util.Idx
public import NN.Proofs.Autograd.Runtime.ShapeErasure

/-!
# Forward Execution Contexts

Node closures read tensors through a shape-indexed reader. The logical evaluator supplies a
`TensorPack`; the compiled evaluator supplies an array whose correspondence with such a pack is
proved. Array reads do not traverse the list of preceding node values. Appending is amortized
constant time when the array is uniquely owned; retaining a context or reader can require copying
the shared array on a later append.

The correspondence is a proposition, so no pack or shape-list traversal is retained at runtime.
Conversion back to a pack happens once, at the typed evaluation boundary.
-/

@[expose] public section

namespace Runtime.Autograd.IRExec

open Spec TorchLean
open Proofs (Idx)

/-- Shape-indexed tensor selection, independent of how the context stores its values. -/
@[ext] structure TensorReader (α : Type) [Storage α] (shapes : List Shape) where
  /-- Read the tensor at a previously checked position and shape. -/
  read : {shape : Shape} → Idx shapes shape → Tensor α shape

namespace TensorReader

variable {α : Type} [Storage α] {shapes : List Shape}

/-- Present a typed pack through the same selection interface as an array context. -/
def ofPack (xs : TensorPack α shapes) : TensorReader α shapes :=
  ⟨fun index => Proofs.getIdx xs index⟩

/-- Existing typed contexts can supply a node's reader without changing tensor values. -/
instance : Coe (TensorPack α shapes) (TensorReader α shapes) := ⟨ofPack⟩

end TensorReader

/-- Select a tensor from a forward context using the index checked during lowering. -/
def readTensor {α : Type} [Storage α] {Γ : List Shape} {s : Shape}
    (xs : TensorReader α Γ) (idx : Idx Γ s) : Tensor α s :=
  xs.read idx

/-- Reading a pack through its reader agrees with the original typed lookup. -/
@[simp] theorem readTensor_ofPack {α : Type} [Storage α] {Γ : List Shape} {s : Shape}
    (xs : TensorPack α Γ) (idx : Idx Γ s) :
    readTensor (xs := xs) idx = Proofs.getIdx xs idx := rfl

namespace Internal

/-- Runtime values together with their erased proof of shape and position correspondence. -/
@[ext] structure ContextArray (α : Type) [Storage α] (shapes : List Shape) where
  /-- Values in the same order as the input and successive SSA nodes. -/
  values : Array (Spec.SomeTensor α)
  /-- The array is the shape erasure of a well-typed context. -/
  valid : ∃ xs : TensorPack α shapes, values = xs.toShapeErasedArray

namespace ContextArray

variable {α : Type} [Storage α] {shapes : List Shape}

private def appendPack : {ss : List Shape} →
    TensorPack α ss → Array (Spec.SomeTensor α) → Array (Spec.SomeTensor α)
  | [], .nil, acc => acc
  | _ :: _, .cons x xs, acc => appendPack xs (acc.push (Spec.SomeTensor.ofTensor x))

private theorem appendPack_eq {ss : List Shape} (xs : TensorPack α ss)
    (acc : Array (Spec.SomeTensor α)) :
    appendPack xs acc = acc ++ xs.toShapeErasedArray := by
  induction xs generalizing acc with
  | nil => simp [appendPack, TensorPack.toShapeErasedArray]
  | cons x xs ih =>
      simp [appendPack, ih, TensorPack.toShapeErasedArray]

/-- Copy the tensor references from a pack into an array in one pass. -/
@[no_expose] def ofPack (xs : TensorPack α shapes) : ContextArray α shapes :=
  ⟨appendPack xs #[], xs, by simp [appendPack_eq]⟩

/-- The one-pass pack conversion has the ordinary shape-erasure semantics. -/
@[simp] theorem values_ofPack (xs : TensorPack α shapes) :
    (ofPack xs).values = xs.toShapeErasedArray := by
  simp [ofPack, appendPack_eq]

/-- An array context contains exactly one value per declared shape. -/
theorem size_values (ctx : ContextArray α shapes) : ctx.values.size = shapes.length := by
  obtain ⟨xs, h⟩ := ctx.valid
  simp [h]

private theorem shape_get (ctx : ContextArray α shapes) {shape : Shape}
    (idx : Idx shapes shape) :
    (ctx.values[idx.i.val]'(by rw [size_values]; exact idx.i.isLt)).shape = shape := by
  obtain ⟨xs, h⟩ := ctx.valid
  simpa only [h, TensorPack.get_toShapeErasedArray, Spec.SomeTensor.shape_ofTensor] using idx.h

/-- Constant-time array selection with a shape cast justified by the context invariant. -/
@[no_expose] def reader (ctx : ContextArray α shapes) : TensorReader α shapes :=
  ⟨fun idx =>
    (ctx.values[idx.i.val]'(by rw [size_values]; exact idx.i.isLt)).cast (shape_get ctx idx)⟩

/-- Array and pack readers select the same tensor at every typed index. -/
@[simp] theorem reader_ofPack (xs : TensorPack α shapes) :
    (ofPack xs).reader = TensorReader.ofPack xs := by
  apply TensorReader.ext
  funext shape idx
  rcases idx with ⟨i, h⟩
  change Spec.SomeTensor.cast _ _ = Tensor.castShape (xs.get i) h
  simp only [values_ofPack, TensorPack.get_toShapeErasedArray]
  cases h
  rfl

/-- Transport only the type index; the runtime array is unchanged. -/
def cast {other : List Shape} (h : shapes = other) (ctx : ContextArray α shapes) :
    ContextArray α other :=
  h ▸ ctx

/-- Casting an array context commutes with casting its typed pack. -/
@[simp] theorem cast_ofPack {other : List Shape} (h : shapes = other)
    (xs : TensorPack α shapes) :
    cast h (ofPack xs) = ofPack (TensorPack.cast h xs) := by
  cases h
  rfl

/--
Append a tensor reference, amortized constant time when the array is uniquely owned.

Retaining the previous context or a reader of it can make this append copy the shared array.
-/
def push {shape : Shape} (ctx : ContextArray α shapes) (x : Tensor α shape) :
    ContextArray α (shapes ++ [shape]) :=
  ⟨ctx.values.push (Spec.SomeTensor.ofTensor x), by
    obtain ⟨xs, h⟩ := ctx.valid
    exact ⟨xs.snoc x, by simp [h]⟩⟩

/-- Array append has the same typed meaning as pack append. -/
@[simp] theorem push_ofPack {shape : Shape} (xs : TensorPack α shapes) (x : Tensor α shape) :
    (ofPack xs).push x = ofPack (xs.snoc x) := by
  apply ContextArray.ext
  simp [push]

private theorem decode_ok (ctx : ContextArray α shapes) :
    ∃ xs, TensorPack.ofShapeErasedArray ctx.values (shapes := shapes) = .ok xs := by
  obtain ⟨xs, h⟩ := ctx.valid
  exact ⟨xs, by simp [h]⟩

/-- Recover the typed result once; an invalid shape branch is excluded by `valid`. -/
def toPack (ctx : ContextArray α shapes) : TensorPack α shapes :=
  match h : TensorPack.ofShapeErasedArray ctx.values (shapes := shapes) with
  | .ok xs => xs
  | .error _ => False.elim (by
      obtain ⟨xs, hx⟩ := decode_ok ctx
      rw [h] at hx
      contradiction)

/-- Converting a well-typed pack to the runtime array and back loses no information. -/
@[simp] theorem toPack_ofPack (xs : TensorPack α shapes) :
    (ofPack xs).toPack = xs := by
  unfold toPack
  split
  next ys h =>
    simpa only [values_ofPack, TensorPack.ofShapeErasedArray_toShapeErasedArray,
      Except.ok.injEq] using h.symm
  next msg h =>
    simp only [values_ofPack, TensorPack.ofShapeErasedArray_toShapeErasedArray] at h
    cases h

end ContextArray
end Internal
end Runtime.Autograd.IRExec

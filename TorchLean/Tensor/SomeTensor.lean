/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.AdjacentSwap

/-!
# Shape-Erased Tensors

`SomeTensor` stores a `TorchLean.Tensor` together with the shape that indexes its type. It erases
the shape of a tensor over a fixed element type. `TorchLean.Op.Value` erases a whole port, its
element sort and its shape, and `TorchLean.IR.someTensor?` reads a value at a value port as a
`SomeTensor`. Runtime collections use `SomeTensor` when they must contain tensors of different
shapes, including autograd tapes, graph interpreters, and certificate checkers.

Backend resources have records of their own. The CUDA tape keeps an opaque device buffer with its
runtime shape (`SomeBuffer`), and operation-polymorphic programs package references owned by a
monad.
-/

@[expose] public section

namespace TorchLean

/-- A tensor paired with the shape that indexes its type. -/
structure SomeTensor (α : Type) [Storage α] where
  /-- The runtime shape of the tensor. -/
  shape : Shape
  /-- The tensor value, indexed by its stored shape. -/
  tensor : Tensor α shape

namespace SomeTensor

variable {α : Type} [Storage α]

/-- Package a statically shaped tensor for shape-erased storage. -/
@[simp] def ofTensor {shape : Shape} (tensor : Tensor α shape) : SomeTensor α :=
  ⟨shape, tensor⟩

/-- Packing a tensor records its static shape as the runtime shape. -/
@[simp] theorem shape_ofTensor {shape : Shape} (tensor : Tensor α shape) :
    (ofTensor tensor).shape = shape :=
  rfl

/-- Packing and unpacking a tensor returns it unchanged, since the recorded shape matches. -/
@[simp] theorem tensor_ofTensor {shape : Shape} (tensor : Tensor α shape) :
    (ofTensor tensor).tensor = tensor :=
  rfl

/-- The stored tensor at type `Tensor α shape`, given a proof that its runtime shape is `shape`. -/
def cast {shape : Shape} (value : SomeTensor α) (h : value.shape = shape) :
    Tensor α shape :=
  Tensor.castShape value.tensor h

/-- Casting to the shape already recorded returns the stored tensor. -/
@[simp] theorem cast_self (value : SomeTensor α) (h : value.shape = value.shape) :
    value.cast h = value.tensor := by
  exact Tensor.castShape_self value.tensor h

/-- Reading a cast tensor at a coordinate reads the stored tensor at the transported
coordinate. -/
theorem cast_apply {shape : Shape} (value : SomeTensor α) (h : value.shape = shape)
    (index : Shape.Coord shape) :
    value.cast h index = value.tensor (_root_.cast (congrArg Shape.Coord h.symm) index) := by
  cases value
  cases h
  rfl

/-- Reading a value equal to `⟨s, x⟩` reads `x` at the coordinate transported along the shape
equality. -/
theorem tensor_apply_of_eq {value : SomeTensor α} {s : Shape} {x : Tensor α s}
    (h : value = ⟨s, x⟩) (c : value.shape.Coord) :
    value.tensor c = x (Shape.Coord.reindexEq (by rw [h]) c) := by
  subst h
  rw [Shape.Coord.reindexEq_self]

/-- Repacking a cast tensor recovers the original value. -/
@[simp] theorem ofTensor_cast (value : SomeTensor α) {shape : Shape}
    (h : value.shape = shape) : ofTensor (value.cast h) = value := by
  subst shape
  rw [cast_self]
  cases value
  rfl

/-- Erasing a tensor's shape after transport recovers the original shape-erased value. -/
@[simp] theorem ofTensor_castShape {shape shape' : Shape}
    (tensor : Tensor α shape) (h : shape = shape') :
    ofTensor (Tensor.castShape tensor h) = ofTensor tensor := by
  cases h
  rfl

/-- Swap the axes at depths `depth` and `depth + 1`, and store the result with its swapped shape. -/
def swapAdjacentAtDepth (value : SomeTensor α) (depth : Nat) : SomeTensor α :=
  match value with
  | ⟨shape, tensor⟩ =>
      ⟨shape.swapAdjacentAtDepth depth,
          Tensor.permute tensor (shape.adjacentSwap depth) (shape.swapAdjacentAtDepth depth)⟩

end SomeTensor
end TorchLean

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape.LinearAlgebra

/-!
# Shape-Erased Tensors

`SomeTensor` stores a `TorchLean.Tensor` together with the shape that indexes its type. It is the
sole general shape-erasure boundary for tensors in TorchLean. Runtime collections use it when they
must contain tensors of different shapes, including autograd tapes, graph interpreters, and
certificate checkers.

Backend-specific resources are not alternative tensor wrappers. For example, the CUDA tape keeps
an opaque device buffer together with runtime shape and allocation metadata; operation-polymorphic
programs similarly package references owned by a monad rather than tensor values.
-/

@[expose] public section

open TorchLean

namespace Spec

/-- A tensor paired with the shape that indexes its type. -/
structure SomeTensor (α : Type) [TorchLean.Storage α] where
  /-- The runtime shape of the tensor. -/
  shape : Shape
  /-- The tensor value, indexed by its stored shape. -/
  tensor : Tensor α shape

namespace SomeTensor

variable {α : Type} [TorchLean.Storage α]

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

/-- Cast the stored tensor after checking its runtime shape. -/
def cast {shape : Shape} (value : SomeTensor α) (h : value.shape = shape) :
    Tensor α shape :=
  Tensor.castShape value.tensor h

/-- Casting to the shape already recorded returns the stored tensor. -/
@[simp] theorem cast_self (value : SomeTensor α) (h : value.shape = value.shape) :
    value.cast h = value.tensor := by
  rw [Subsingleton.elim h rfl]
  rfl

/-- Repacking a tensor after a successful shape cast recovers the original value. -/
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

/-- Swap two adjacent axes at `depth`, retaining the resulting shape in the package. -/
def swapAdjacentAtDepth (value : SomeTensor α) (depth : Nat) : SomeTensor α :=
  match value with
  | ⟨shape, tensor⟩ =>
      ⟨shape.swapAdjacentAtDepth depth, Tensor.swapAdjacentAxes (tensor := tensor) depth⟩

end SomeTensor
end Spec

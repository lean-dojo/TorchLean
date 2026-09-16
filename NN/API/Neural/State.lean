/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor.Pack
public import NN.Tensor -- shake: keep

/-!
# Model State

`nn.State α shapes` is the public, shape-indexed container for model parameters and persistent
buffers. Its recursive tensor-pack representation stays behind the runtime boundary; application
code reads and replaces entries through named operations.
-/

public section

namespace TorchLean
namespace nn

/--
Model parameters and persistent buffers in their statically known order.

The element type is shared while every tensor keeps its own shape from `shapes`.
-/
structure State (α : Type) [TorchLean.Storage α] (shapes : List Shape) where
  private mk ::
  private tensors : TorchLean.TensorPack α shapes

namespace State

variable {α β γ : Type}
  [TorchLean.Storage α] [TorchLean.Storage β] [TorchLean.Storage γ]
  {shapes leftShapes rightShapes : List Shape}

@[expose] section

namespace Internal

/-- Wrap the runtime tensor-pack representation at an implementation boundary. -/
@[no_expose] def fromTensorPack
    (tensors : TorchLean.TensorPack α shapes) : State α shapes :=
  ⟨tensors⟩

/-- Reveal the runtime tensor-pack representation at an implementation boundary. -/
@[no_expose] def toTensorPack
    (state : State α shapes) : TorchLean.TensorPack α shapes :=
  match state with
  | ⟨tensors⟩ => tensors

/-- Revealing a freshly wrapped tensor pack returns the original pack. -/
@[simp] theorem toTensorPack_fromTensorPack (tensors : TorchLean.TensorPack α shapes) :
    toTensorPack (fromTensorPack tensors) = tensors := by
  rfl

/-- Wrapping the representation of a state reconstructs that state. -/
@[simp] theorem fromTensorPack_toTensorPack (state : State α shapes) :
    fromTensorPack (toTensorPack state) = state := by
  cases state
  rfl

/-- Replace one tensor in the runtime representation. -/
def replaceTensor {ss : List Shape}
    (tensors : TorchLean.TensorPack α ss)
    (index : Fin ss.length)
    (value : Tensor α (ss.get index)) :
    TorchLean.TensorPack α ss :=
  match tensors with
  | .nil => nomatch index
  | .cons first rest =>
      match index with
      | ⟨0, _⟩ => .cons value rest
      | ⟨Nat.succ offset, isValid⟩ =>
          .cons first <|
            replaceTensor rest
              ⟨offset, Nat.lt_of_succ_lt_succ isValid⟩ value

end Internal

/-- Two states are equal when their tensor-pack representations are equal. -/
@[ext] theorem ext {left right : State α shapes}
    (h : Internal.toTensorPack left = Internal.toTensorPack right) :
    left = right := by
  rw [← Internal.fromTensorPack_toTensorPack left,
    ← Internal.fromTensorPack_toTensorPack right, h]

/-- Empty model state. -/
def empty : State α [] :=
  Internal.fromTensorPack TorchLean.TensorPack.empty

/-- Construct state whose every tensor contains `value`. -/
def full (value : α) : State α shapes :=
  Internal.fromTensorPack (TorchLean.TensorPack.fill value)

/-- Construct all-zero state. -/
def zeros [Zero α] : State α shapes :=
  Internal.fromTensorPack TorchLean.TensorPack.zero

/-- Read one state tensor; its result shape is determined by the index. -/
def get (state : State α shapes) (index : Fin shapes.length) :
    Tensor α (shapes.get index) :=
  TorchLean.TensorPack.get (Internal.toTensorPack state) index

/-- Replace one state tensor with another tensor of exactly the required shape. -/
def set (state : State α shapes) (index : Fin shapes.length)
    (value : Tensor α (shapes.get index)) : State α shapes :=
  Internal.fromTensorPack <|
    Internal.replaceTensor (Internal.toTensorPack state) index value

/-- Apply a shape-preserving conversion to every state tensor. -/
def map (state : State α shapes)
    (f : ∀ {shape : Shape}, Tensor α shape → Tensor β shape) :
    State β shapes :=
  Internal.fromTensorPack <|
    TorchLean.TensorPack.map f (Internal.toTensorPack state)

/-- Combine corresponding state tensors with a shape-preserving operation. -/
def zipWith (first : State α shapes) (second : State β shapes)
    (f : ∀ {shape : Shape},
      Tensor α shape → Tensor β shape → Tensor γ shape) :
    State γ shapes :=
  Internal.fromTensorPack <|
    TorchLean.TensorPack.zipWith f
      (Internal.toTensorPack first) (Internal.toTensorPack second)

/-- Concatenate two model states while retaining the combined shape layout in the type. -/
def append (first : State α leftShapes) (second : State α rightShapes) :
    State α (leftShapes ++ rightShapes) :=
  Internal.fromTensorPack <|
    TorchLean.TensorPack.append
      (Internal.toTensorPack first) (Internal.toTensorPack second)

/-- Add one tensor at the end of a state, extending its statically known layout. -/
def push {shape : Shape} (state : State α shapes) (tensor : Tensor α shape) :
    State α (shapes ++ [shape]) :=
  state.append <|
    Internal.fromTensorPack (TorchLean.TensorPack.singleton tensor)

instance [Repr α] : Repr (State α shapes) where
  reprPrec state precedence :=
    reprPrec (Internal.toTensorPack state) precedence

/-- Named result of splitting state at a statically known shape-list boundary. -/
structure Partition (α : Type) [TorchLean.Storage α]
    (leftShapes rightShapes : List Shape) where
  /-- State before the split boundary. -/
  left : State α leftShapes
  /-- State after the split boundary. -/
  right : State α rightShapes
deriving Repr

/-- Split state at a statically known shape-list boundary. -/
def split (state : State α (leftShapes ++ rightShapes)) :
    Partition α leftShapes rightShapes :=
  let (left, right) := TorchLean.TensorPack.split
    (ss₁ := leftShapes) (ss₂ := rightShapes)
    (Internal.toTensorPack state)
  { left := Internal.fromTensorPack left
    right := Internal.fromTensorPack right }

/-- Transport state along an equality between its statically known shape layouts. -/
def cast (state : State α leftShapes) (sameShapes : leftShapes = rightShapes) :
    State α rightShapes :=
  Internal.fromTensorPack <|
    TorchLean.TensorPack.cast sameShapes (Internal.toTensorPack state)

end
end State
end nn
end TorchLean

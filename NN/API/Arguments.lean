/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor.Pack
public import NN.Tensor -- shake: keep

/-!
# Typed Program Arguments

`Arguments α shapes` is the public container for an ordered set of tensors supplied to a
multi-input program. Each tensor retains its own statically known shape while the tensor-pack
representation remains behind `Arguments.Internal`.
-/

public section

namespace TorchLean

/-- Ordered tensor arguments for a program whose input shapes are known statically. -/
structure Arguments (α : Type) [TorchLean.Storage α] (shapes : List Shape) where
  private mk ::
  private tensors : TorchLean.TensorPack α shapes

namespace Arguments

variable {α β γ : Type}
  [TorchLean.Storage α] [TorchLean.Storage β] [TorchLean.Storage γ]
  {shapes leftShapes rightShapes : List Shape}

@[expose] section

namespace Internal

/-- Wrap the tensor-pack representation at an implementation boundary. -/
@[no_expose] def fromTensorPack
    (tensors : TorchLean.TensorPack α shapes) : Arguments α shapes :=
  ⟨tensors⟩

/-- Reveal the tensor-pack representation at an implementation boundary. -/
@[no_expose] def toTensorPack
    (arguments : Arguments α shapes) : TorchLean.TensorPack α shapes :=
  match arguments with
  | ⟨tensors⟩ => tensors

/-- Revealing a freshly wrapped tensor pack returns the original pack. -/
@[simp] theorem toTensorPack_fromTensorPack (tensors : TorchLean.TensorPack α shapes) :
    toTensorPack (fromTensorPack tensors) = tensors := by
  rfl

/-- Wrapping the representation of arguments reconstructs those arguments. -/
@[simp] theorem fromTensorPack_toTensorPack (arguments : Arguments α shapes) :
    fromTensorPack (toTensorPack arguments) = arguments := by
  cases arguments
  rfl

end Internal

/-- Two argument sequences are equal when their tensor-pack representations are equal. -/
@[ext] theorem ext {left right : Arguments α shapes}
    (h : Internal.toTensorPack left = Internal.toTensorPack right) :
    left = right := by
  rw [← Internal.fromTensorPack_toTensorPack left,
    ← Internal.fromTensorPack_toTensorPack right, h]

/-- No program arguments. -/
def empty : Arguments α [] :=
  Internal.fromTensorPack TorchLean.TensorPack.empty

/-- Add one tensor at the end of an argument sequence. -/
def push {shape : Shape} (arguments : Arguments α shapes) (tensor : Tensor α shape) :
    Arguments α (shapes ++ [shape]) :=
  Internal.fromTensorPack <|
    TorchLean.TensorPack.snoc (Internal.toTensorPack arguments) tensor

/-- Read one argument; its result shape is determined by the index. -/
def get (arguments : Arguments α shapes) (index : Fin shapes.length) :
    Tensor α (shapes.get index) :=
  TorchLean.TensorPack.get (Internal.toTensorPack arguments) index

/-- Apply a shape-preserving conversion to every argument. -/
def map (arguments : Arguments α shapes)
    (f : ∀ {shape : Shape}, Tensor α shape → Tensor β shape) :
    Arguments β shapes :=
  Internal.fromTensorPack <|
    TorchLean.TensorPack.map f (Internal.toTensorPack arguments)

/-- Combine corresponding arguments with a shape-preserving operation. -/
def zipWith (first : Arguments α shapes) (second : Arguments β shapes)
    (f : ∀ {shape : Shape},
      Tensor α shape → Tensor β shape → Tensor γ shape) :
    Arguments γ shapes :=
  Internal.fromTensorPack <|
    TorchLean.TensorPack.zipWith f
      (Internal.toTensorPack first) (Internal.toTensorPack second)

/-- Concatenate two ordered argument sequences. -/
def append (first : Arguments α leftShapes) (second : Arguments α rightShapes) :
    Arguments α (leftShapes ++ rightShapes) :=
  Internal.fromTensorPack <|
    TorchLean.TensorPack.append
      (Internal.toTensorPack first) (Internal.toTensorPack second)

instance [Repr α] : Repr (Arguments α shapes) where
  reprPrec arguments precedence :=
    reprPrec (Internal.toTensorPack arguments) precedence

/-- Named result of splitting arguments at a statically known shape-list boundary. -/
structure Partition (α : Type) [TorchLean.Storage α]
    (leftShapes rightShapes : List Shape) where
  /-- Arguments before the split boundary. -/
  left : Arguments α leftShapes
  /-- Arguments after the split boundary. -/
  right : Arguments α rightShapes
deriving Repr

/-- Split arguments at a statically known shape-list boundary. -/
def split (arguments : Arguments α (leftShapes ++ rightShapes)) :
    Partition α leftShapes rightShapes :=
  let (left, right) := TorchLean.TensorPack.split
    (ss₁ := leftShapes) (ss₂ := rightShapes)
    (Internal.toTensorPack arguments)
  { left := Internal.fromTensorPack left
    right := Internal.fromTensorPack right }

end
end Arguments
end TorchLean

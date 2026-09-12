/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.SomeTensor
public import NN.Tensor.Pack

/-!
# Tensor Shape-Erasure Boundary

Conversions between statically shape-indexed `TensorPack` values and arrays of existentially
packaged `Spec.SomeTensor` values. Runtime tapes and heterogeneous external data use the erased
representation; typed computation should recover a `TensorPack` immediately after crossing that
boundary.
-/

@[expose] public section

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace TorchLean.TensorPack

variable {α : Type} [Storage α]

/-- Erase the individual tensor shapes in a typed pack into a runtime array. -/
def toShapeErasedArray : {ss : List Shape} →
    TensorPack α ss → Array (Spec.SomeTensor α)
  | [], .nil => #[]
  | _ :: ss, .cons tensor tensors =>
      #[Spec.SomeTensor.ofTensor tensor] ++ toShapeErasedArray (ss := ss) tensors

/-- Tag each shape-erased tensor with its runtime index, beginning at `start`. -/
def toIndexedShapeErasedArray : {ss : List Shape} →
    TensorPack α ss → Nat → Array (Nat × Spec.SomeTensor α)
  | [], .nil, _ => #[]
  | _ :: ss, .cons tensor tensors, index =>
      #[(index, Spec.SomeTensor.ofTensor tensor)] ++
        toIndexedShapeErasedArray (ss := ss) tensors (index + 1)

/--
Recover a statically shape-indexed pack from a prefix of a shape-erased runtime array.

`start` selects the first array entry to consume. The conversion checks every stored shape and
fails if the array is too short or an entry has the wrong shape. Entries after the requested pack
are intentionally ignored, which supports recovering a typed prefix of a larger runtime context.
-/
def ofShapeErasedArray (values : Array (Spec.SomeTensor α))
    (start : Nat := 0) : {shapes : List Shape} → Except String (TensorPack α shapes)
  | [] => pure .nil
  | shape :: shapes => do
      let value ← match values[start]? with
        | some value => pure value
        | none => throw s!"tensor pack: shape-erased array is missing entry {start}"
      if h : value.shape = shape then
        pure (.cons (value.cast h) (← ofShapeErasedArray values (start + 1) (shapes := shapes)))
      else
        throw <|
          s!"tensor pack: shape mismatch at entry {start} (expected {Shape.pretty shape}, got " ++
            s!"{Shape.pretty value.shape})"

end TorchLean.TensorPack

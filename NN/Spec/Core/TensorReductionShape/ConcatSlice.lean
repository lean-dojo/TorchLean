/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor.Internal.Representation.Segment
public import NN.Spec.Core.TensorReductionShape.ShapeChange -- shake: keep
public import NN.Spec.Core.Context -- shake: keep

@[expose] public section


open Spec TorchLean

namespace TorchLean.Tensor

/-!
# Concatenation and Slicing

Concatenation, slicing, singleton dimensions, and layout transforms.
-/

/-- Concatenate at the axis following an arbitrary leading shape. -/
def concatAxisSpec {α : Type} [TorchLean.Storage α]
    (leading : Shape) {n m : Nat} {suffix : Shape}
    (left : Tensor α (leading.concat (.dim n suffix)))
    (right : Tensor α (leading.concat (.dim m suffix))) :
    Tensor α (leading.concat (.dim (n + m) suffix)) := by
  exact TorchLean.Tensor.Internal.Rep.castShape (by simp [Shape.concat_eq_append]) <|
    TorchLean.Tensor.Internal.Rep.concatenateAxis
      (leftLength := n) (rightLength := m) leading.toList suffix.toList
      (TorchLean.Tensor.Internal.Rep.castShape (by simp [Shape.concat_eq_append]) left)
      (TorchLean.Tensor.Internal.Rep.castShape (by simp [Shape.concat_eq_append]) right)

/-- Insert a singleton dimension at any valid axis. -/
def unsqueezeSpec {α : Type} [TorchLean.Storage α]
    {shape : Shape} (tensor : Tensor α shape)
    (axis : Nat) (hAxis : axis ≤ shape.rank) : Tensor α (shape.insertAxis axis 1) :=
  match axis, shape with
  | 0, .scalar => Tensor.dim fun _ => tensor
  | 0, .dim _ _ => Tensor.dim fun _ => tensor
  | axis + 1, .dim _ rest =>
      Tensor.dim fun i =>
        unsqueezeSpec (Tensor.unstack tensor i) axis (by
          simp only [Shape.rank] at hAxis
          omega)
  | axis + 1, .scalar =>
      False.elim (Nat.not_succ_le_zero axis hAxis)
termination_by axis

end TorchLean.Tensor

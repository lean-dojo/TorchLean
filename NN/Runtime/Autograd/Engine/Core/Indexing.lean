/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/


module

public import NN.Runtime.Autograd.Engine.Core.Base

/-!
# Core Tape Indexing Operations

This file implements gather and scatter-style tape nodes. The forward rules expose typed indexing
operations, and the backward rules route upstream gradients back to the selected source coordinates.
-/

@[expose] public section

namespace Runtime
namespace Autograd

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Tape

/-- Select one bounded coordinate from any tensor axis. -/
@[inline] def select {α : Type} [TorchLean.Storage α] [Zero α]
    {s : Shape} (t : Tape α) (xId : Nat) (axis : Nat)
    [Shape.AxisInBounds axis s] (index : Fin (Shape.axisSize s axis)) :
    Result (Tape α × Nat) := do
  let x ← requireValue (α := α) (t := t) (s := s) xId
  let y := Tensor.selectSpec axis x index
  let node : Node α :=
    { name := some s!"select(axis={axis}, index={index.val})"
      value := Spec.SomeTensor.ofTensor y
      requiresGrad := (t.getNode? xId).any (·.requiresGrad)
      parents := #[xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s.eraseAxis axis) dLdyAny
        let dx := Tensor.selectBackwardSpec axis index dLdy
        pure #[(xId, Spec.SomeTensor.ofTensor dx)] }
  pure (t.addNode node)
/-- Select several bounded coordinates from any tensor axis. -/
@[inline] def indexSelect {α : Type} [TorchLean.Storage α] [Add α] [Zero α]
    {s : Shape} (t : Tape α) (xId : Nat) (axis count : Nat)
    [Shape.AxisInBounds axis s]
    (indices : Tensor (Fin (Shape.axisSize s axis)) [count]) : Result (Tape α × Nat) := do
  let x ← requireValue (α := α) (t := t) (s := s) xId
  let y := Tensor.indexSelectSpec axis x indices
  let node : Node α :=
    { name := some s!"index_select(axis={axis})"
      value := Spec.SomeTensor.ofTensor y
      requiresGrad := (t.getNode? xId).any (·.requiresGrad)
      parents := #[xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s.replaceAxis axis count) dLdyAny
        let zero := Tensor.full s (0 : α)
        let dx := Tensor.scatterAddSpec axis zero indices dLdy
        pure #[(xId, Spec.SomeTensor.ofTensor dx)] }
  pure (t.addNode node)

/-- Add indexed source slices into any tensor axis. -/
@[inline] def scatterAdd {α : Type} [TorchLean.Storage α] [Add α] [Zero α]
    {s : Shape} (t : Tape α) (baseId sourceId : Nat) (axis count : Nat)
    [Shape.AxisInBounds axis s]
    (indices : Tensor (Fin (Shape.axisSize s axis)) [count]) : Result (Tape α × Nat) := do
  let base ← requireValue (α := α) (t := t) (s := s) baseId
  let source ← requireValue (α := α) (t := t) (s := s.replaceAxis axis count) sourceId
  let y := Tensor.scatterAddSpec axis base indices source
  let node : Node α :=
    { name := some s!"scatter_add(axis={axis})"
      value := Spec.SomeTensor.ofTensor y
      requiresGrad :=
        (t.getNode? baseId).any (·.requiresGrad) ||
        (t.getNode? sourceId).any (·.requiresGrad)
      parents := #[baseId, sourceId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        let dSource := Tensor.indexSelectSpec axis dLdy indices
        pure #[
          (baseId, Spec.SomeTensor.ofTensor dLdy),
          (sourceId, Spec.SomeTensor.ofTensor dSource)] }
  pure (t.addNode node)

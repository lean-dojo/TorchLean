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
    Result (Tape α × Nat) :=
  unary (α := α) (t := t) (σ := s) (τ := s.eraseAxis axis)
    s!"select(axis={axis}, index={index.val})" xId
    (forward := fun x => Tensor.selectSpec axis x index)
    (backward := fun _x dLdy => Tensor.selectBackwardSpec axis index dLdy)

/-- Select several bounded coordinates from any tensor axis. -/
@[inline] def indexSelect {α : Type} [TorchLean.Storage α] [Add α] [Zero α]
    {s : Shape} (t : Tape α) (xId : Nat) (axis count : Nat)
    [Shape.AxisInBounds axis s]
    (indices : Tensor (Fin (Shape.axisSize s axis)) [count]) : Result (Tape α × Nat) :=
  unary (α := α) (t := t) (σ := s) (τ := s.replaceAxis axis count)
    s!"index_select(axis={axis})" xId
    (forward := fun x => Tensor.indexSelectSpec axis x indices)
    (backward := fun _x dLdy =>
      let zero := Tensor.full s (0 : α)
      Tensor.scatterAddSpec axis zero indices dLdy)

/-- Add indexed source slices into any tensor axis. -/
@[inline] def scatterAdd {α : Type} [TorchLean.Storage α] [Add α] [Zero α]
    {s : Shape} (t : Tape α) (baseId sourceId : Nat) (axis count : Nat)
    [Shape.AxisInBounds axis s]
    (indices : Tensor (Fin (Shape.axisSize s axis)) [count]) : Result (Tape α × Nat) :=
  binary (α := α) (t := t) (σ₁ := s) (σ₂ := s.replaceAxis axis count) (τ := s)
    s!"scatter_add(axis={axis})" baseId sourceId
    (forward := fun base source => Tensor.scatterAddSpec axis base indices source)
    (backward := fun _base _source dLdy =>
      (dLdy, Tensor.indexSelectSpec axis dLdy indices))

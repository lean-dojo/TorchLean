/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Dual

/-!
# Nested forward-mode inputs

Each layer of `Dual.Nested α n` carries one independent differentiation direction. The last
direction is outermost: for two directions the input is `⟨⟨x, v⟩, ⟨w, 0⟩⟩`. The first listed
direction is therefore innermost. Keeping this order matters for backends whose arithmetic
is not associative.

Seeding and extraction work for every runtime scalar context. Their exact-real derivative
interpretation is proved separately in `NN.Proofs.Autograd.Dual`.
-/

@[expose] public section

namespace Runtime.Autograd.Model.Dual

open Spec TorchLean

/-- The runtime dual carrier nested once for each differentiation direction. -/
def Nested (α : Type) : Nat → Type
  | 0 => α
  | n + 1 => Dual (Nested α n)

/-- Preserve the base backend's storage at order zero; dual layers use array storage. -/
instance instStorageNested {α : Type} [Storage α] (n : Nat) : Storage (Nested α n) :=
  match n with
  | 0 => inferInstanceAs (Storage α)
  | _ + 1 => inferInstanceAs (Storage (Dual _))

/-- Reuse the scalar backend's arithmetic at every level of nesting. -/
instance instContextNested {α : Type} [Storage α] [Context α] (n : Nat) :
    Context (Nested α n) :=
  match n with
  | 0 => inferInstanceAs (Context α)
  | n + 1 =>
    @Dual.instContextOfStorage (Nested α n) (instStorageNested n) (instContextNested n)

namespace Nested

/-- Embed a constant in all primal slots and set its derivative coefficients to zero. -/
def ofPrimal {α : Type} [Storage α] [Context α] : (n : Nat) → α → Nested α n
  | 0, value => value
  | n + 1, value => Dual.ofPrimal (ofPrimal n value)

/-- Zero gradient buffers have the same representation at every nesting depth. -/
@[simp] theorem ofPrimal_zero {α : Type} [Storage α] [Context α] (n : Nat) :
    ofPrimal n (0 : α) = (0 : Nested α n) := by
  induction n with
  | zero => rfl
  | succ n ih =>
      change Dual.mk (ofPrimal n 0) 0 = Dual.mk 0 0
      rw [ih]

/-- Seed constant directions; the last direction occupies the outermost dual layer. -/
def seed {α : Type} [Storage α] [Context α] : {n : Nat} → (Fin n → α) → α → Nested α n
  | 0, _, value => value
  | n + 1, directions, value =>
    ⟨seed (Fin.init directions) value, ofPrimal n (directions (Fin.last n))⟩

/-- Extract the coefficient containing every differentiation direction exactly once. -/
def tangent {α : Type} : {n : Nat} → Nested α n → α
  | 0, value => value
  | _ + 1, value => tangent value.du

/-- Zero directions give the same coefficients as a constant embedding. -/
@[simp] theorem seed_zero {α : Type} [Storage α] [Context α] (n : Nat) (x : α) :
    seed (fun _ : Fin n => 0) x = ofPrimal n x := by
  induction n with
  | zero => rfl
  | succ n ih =>
    change Dual.mk (seed (fun _ : Fin n => 0) x) (ofPrimal n 0) =
      Dual.mk (ofPrimal n x) 0
    rw [ih, ofPrimal_zero]

/-- Seed a direction tuple of arbitrary tensor shape, coordinate by coordinate. -/
def seedTensor {α : Type} [Storage α] [Context α] {shape : Shape} {n : Nat}
    (directions : Fin n → Tensor α shape) (input : Tensor α shape) :
    Tensor (Nested α n) shape :=
  Tensor.Internal.Rep.ofFn fun i => seed (fun k => directions k i) (input i)

/-- Extract the mixed derivative coefficient at every output coordinate. -/
def tangentTensor {α : Type} [Storage α] {shape : Shape} {n : Nat}
    (output : Tensor (Nested α n) shape) : Tensor α shape :=
  Tensor.map tangent output

/-- Seeding preserves the supplied value and directions at every tensor coordinate. -/
@[simp] theorem seedTensor_apply {α : Type} [Storage α] [Context α] {shape : Shape} {n : Nat}
    (directions : Fin n → Tensor α shape) (input : Tensor α shape) (i : shape.Coord) :
    seedTensor directions input i = seed (fun k => directions k i) (input i) :=
  Tensor.Internal.Rep.get_ofFn _ i

/-- A tensor with zero directions is held constant at every nesting depth. -/
@[simp] theorem seedTensor_zero {α : Type} [Storage α] [Context α] {shape : Shape}
    (n : Nat) (x : Tensor α shape) :
    seedTensor (fun _ : Fin n => Tensor.zeros shape) x = Tensor.map (ofPrimal n) x := by
  apply Tensor.Internal.Rep.ext
  intro i
  simp only [seedTensor_apply, Tensor.map, Tensor.Internal.Rep.map_apply, Tensor.zeros,
    Tensor.full_apply, seed_zero]

/-- Tensor extraction reads exactly the scalar coefficient at each coordinate. -/
@[simp] theorem tangentTensor_apply {α : Type} [Storage α] {shape : Shape} {n : Nat}
    (output : Tensor (Nested α n) shape) (i : shape.Coord) :
    tangentTensor output i = tangent (output i) :=
  Tensor.Internal.Rep.map_apply _ _ i

/-- Adding a direction uses the existing one-step tensor seeding operation. -/
theorem seedTensor_succ {α : Type} [Storage α] [Context α] {shape : Shape} {n : Nat}
    (directions : Fin (n + 1) → Tensor α shape) (input : Tensor α shape) :
    seedTensor directions input =
      DualTensor.withTangents (seedTensor (Fin.init directions) input)
        (Tensor.map (ofPrimal n) (directions (Fin.last n))) := by
  apply Tensor.Internal.Rep.ext
  intro i
  change (seedTensor directions input i : Dual (Nested α n)) =
    (DualTensor.withTangents (seedTensor (Fin.init directions) input)
      (Tensor.map (ofPrimal n) (directions (Fin.last n)))) i
  simp only [DualTensor.withTangents, Tensor.map2Spec_apply, seedTensor_apply, seed,
    Tensor.map, Tensor.Internal.Rep.map_apply, Dual.mk']
  rfl

end Nested

end Runtime.Autograd.Model.Dual

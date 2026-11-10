/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Dual

/-!
# Nested forward-mode inputs

Each layer of `Dual.Nested α n` carries one independent differentiation direction, with the last
direction outermost and the first innermost: for directions `v` and `w` the input is
`⟨⟨x, v⟩, ⟨w, 0⟩⟩`. The order matters on backends whose arithmetic is not associative, such as
floating point.

Seeding and extraction work in every runtime scalar context, and `TorchLean.NN.Dual.Jet`
proves what they mean as exact real derivatives.
-/

@[expose] public section

namespace TorchLean.Dual

/-- The runtime dual carrier nested once for each differentiation direction. -/
def Nested (α : Type) : Nat → Type
  | 0 => α
  | n + 1 => Dual (Nested α n)

/-- Keep the base backend's storage at order zero. Dual layers use array storage. -/
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

/--
Embed `value` as a constant: it fills the innermost primal coefficient, and every derivative
coefficient is zero.
-/
def ofPrimal {α : Type} [Storage α] [Context α] : (n : Nat) → α → Nested α n
  | 0, value => value
  | n + 1, value => Dual.ofPrimal (ofPrimal n value)

/-- Embedding `0` gives the zero of `Nested α n`, so zero buffers agree at every nesting depth. -/
@[simp] theorem ofPrimal_zero {α : Type} [Storage α] [Context α] (n : Nat) :
    ofPrimal n (0 : α) = (0 : Nested α n) := by
  induction n with
  | zero => rfl
  | succ n ih =>
      change Dual.mk (ofPrimal n 0) 0 = Dual.mk 0 0
      rw [ih]

/-- Seed `value` with one constant direction per layer, the last one in the outermost layer. -/
def seed {α : Type} [Storage α] [Context α] : {n : Nat} → (Fin n → α) → α → Nested α n
  | 0, _, value => value
  | n + 1, directions, value =>
    ⟨seed (Fin.init directions) value, ofPrimal n (directions (Fin.last n))⟩

/-- Read the coefficient that contains every differentiation direction exactly once. -/
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
  Tensor.Rep.ofFn fun i => seed (fun k => directions k i) (input i)

/-- Extract the mixed derivative coefficient at every output coordinate. -/
def tangentTensor {α : Type} [Storage α] {shape : Shape} {n : Nat}
    (output : Tensor (Nested α n) shape) : Tensor α shape :=
  Tensor.map tangent output

/--
At each coordinate, `seedTensor` is `seed` applied to that coordinate of the directions and of the
input.
-/
@[simp] theorem seedTensor_apply {α : Type} [Storage α] [Context α] {shape : Shape} {n : Nat}
    (directions : Fin n → Tensor α shape) (input : Tensor α shape) (i : shape.Coord) :
    seedTensor directions input i = seed (fun k => directions k i) (input i) :=
  Tensor.Rep.get_ofFn _ i

/-- Seeding a tensor with zero directions embeds every coordinate with `ofPrimal`. -/
@[simp] theorem seedTensor_zero {α : Type} [Storage α] [Context α] {shape : Shape}
    (n : Nat) (x : Tensor α shape) :
    seedTensor (fun _ : Fin n => Tensor.zeros shape) x = Tensor.map (ofPrimal n) x := by
  apply Tensor.Rep.ext
  intro i
  simp only [seedTensor_apply, Tensor.map_apply, Tensor.zeros,
    Tensor.full_apply, seed_zero]

/-- `tangentTensor` reads `tangent` of the output entry at each coordinate. -/
@[simp] theorem tangentTensor_apply {α : Type} [Storage α] {shape : Shape} {n : Nat}
    (output : Tensor (Nested α n) shape) (i : shape.Coord) :
    tangentTensor output i = tangent (output i) :=
  Tensor.map_apply _ _ i

/-- Seeding `n + 1` directions pairs the seeding of the first `n` with the last as tangent. -/
theorem seedTensor_succ {α : Type} [Storage α] [Context α] {shape : Shape} {n : Nat}
    (directions : Fin (n + 1) → Tensor α shape) (input : Tensor α shape) :
    seedTensor directions input =
      DualTensor.withTangents (seedTensor (Fin.init directions) input)
        (Tensor.map (ofPrimal n) (directions (Fin.last n))) := by
  apply Tensor.Rep.ext
  intro i
  change (seedTensor directions input i : Dual (Nested α n)) =
    (DualTensor.withTangents (seedTensor (Fin.init directions) input)
      (Tensor.map (ofPrimal n) (directions (Fin.last n)))) i
  simp only [DualTensor.withTangents, Tensor.zipWith_apply, seedTensor_apply, seed,
    Tensor.map_apply]
  rfl

end Nested

end TorchLean.Dual

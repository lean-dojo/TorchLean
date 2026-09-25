/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Algebra.Context

/-!
A VJP can describe a uniform contribution with finitely many exceptions. Its dense program remains
available as a compatibility adapter. The certificate preserves the scalar program at every entry;
in particular, adding two contributions does not discard zeros or combine repeated parents early.
-/

@[expose] public section

namespace Proofs.Autograd.Algebra

open Spec TorchLean

/-- A typed position whose shape is available without traversing the context's shape list. -/
abbrev SomeIdx (shapes : List Shape) := (shape : Shape) × Idx shapes shape

namespace TensorLookup

variable {α : Type} [Storage α] {shapes : List Shape}

/-- A reader whose tensors all contain the same scalar. -/
def fill (value : α) : TensorLookup α shapes :=
  ⟨fun {shape} _ => Tensor.full shape value⟩

/-- Pointwise addition retains left-to-right scalar argument order. -/
def add [Add α] (left right : TensorLookup α shapes) : TensorLookup α shapes :=
  ⟨fun idx => Tensor.addSpec (left.read idx) (right.read idx)⟩

@[simp] theorem ofPack_fill (value : α) :
    ofPack (TorchLean.TensorPack.fill value (ss := shapes)) = fill value := by
  apply TensorLookup.ext
  funext shape idx
  obtain ⟨i, rfl⟩ := idx
  induction shapes with
  | nil => exact Fin.elim0 i
  | cons head shapes ih =>
      cases i using Fin.cases with
      | zero => rfl
      | succ i => exact ih i

@[simp] theorem ofPack_zero [Zero α] :
    ofPack (TorchLean.TensorPack.zero (α := α) (ss := shapes)) = fill 0 := by
  have same : TorchLean.TensorPack.zero (α := α) (ss := shapes) =
      TorchLean.TensorPack.fill 0 := by
    induction shapes with
    | nil => rfl
    | cons shape shapes ih => simp only [TorchLean.TensorPack.zero, TorchLean.TensorPack.fill, ih]
  rw [same, ofPack_fill]

@[simp] theorem ofPack_add [Add α] (left right : TorchLean.TensorPack α shapes) :
    ofPack (TorchLean.TensorPack.add left right) = add (ofPack left) (ofPack right) := by
  apply TensorLookup.ext
  funext shape idx
  obtain ⟨i, rfl⟩ := idx
  induction shapes with
  | nil => exact Fin.elim0 i
  | cons head shapes ih =>
      cases left with
      | cons left lefts =>
        cases right with
        | cons right rights =>
          cases i using Fin.cases with
          | zero => rfl
          | succ i => exact ih lefts rights i

end TensorLookup

/-- A dense VJP program with a certified compact description of its entries. -/
structure Contributions (α : Type) [Storage α] (shapes : List Shape) where
  /-- The original dense program, evaluated only by the compatibility adapter. -/
  dense : Unit → TorchLean.TensorPack α shapes
  /-- Constant-time lookup for the small expression formed by one node's VJP. -/
  lookup : TensorLookup α shapes
  /-- Scalar present in tensors at positions outside `support`. -/
  uniform : α
  /-- Positions where the contribution can differ from the uniform tensor. -/
  support : List (SomeIdx shapes)
  /-- The compact reader computes exactly the dense VJP. -/
  correct : TensorLookup.ofPack (dense ()) = lookup
  /-- Every omitted position really is uniform, without assuming any law about zero. -/
  outside : ∀ {shape} (idx : Idx shapes shape),
    (∀ entry ∈ support, entry.2.i.val ≠ idx.i.val) →
      lookup.read idx = Tensor.full shape uniform

namespace Contributions

variable {α : Type} [Storage α] {shapes : List Shape}

/-- A uniform contribution; no parent positions need individual updates. -/
def fill (value : α) : Contributions α shapes :=
  { dense := fun _ => TorchLean.TensorPack.fill value
    lookup := TensorLookup.fill value
    uniform := value
    support := []
    correct := TensorLookup.ofPack_fill value
    outside := fun _ _ => rfl }

/-- The original all-zero pack, with its explicit uniform contribution retained. -/
def zero [Zero α] : Contributions α shapes :=
  { dense := fun _ => TorchLean.TensorPack.zero
    lookup := TensorLookup.fill 0
    uniform := 0
    support := []
    correct := TensorLookup.ofPack_zero
    outside := fun _ _ => rfl }

private theorem add_full [Add α] (shape : Shape) (left right : α) :
    Tensor.addSpec (Tensor.full shape left) (Tensor.full shape right) =
      Tensor.full shape (left + right) := by
  apply Tensor.Internal.Rep.ext
  intro coordinate
  simp [Tensor.addSpec]

/--
Add two VJP descriptions in the original order. Duplicate support entries intentionally remain:
the reader evaluates the complete local sum before the global gradient receives one contribution.
-/
def add [Add α] (left right : Contributions α shapes) : Contributions α shapes :=
  { dense := fun _ => TorchLean.TensorPack.add (left.dense ()) (right.dense ())
    lookup := TensorLookup.add left.lookup right.lookup
    uniform := left.uniform + right.uniform
    support := left.support ++ right.support
    correct := by rw [TensorLookup.ofPack_add, left.correct, right.correct]
    outside := by
      intro shape idx absent
      change Tensor.addSpec (left.lookup.read idx) (right.lookup.read idx) = _
      rw [left.outside idx (fun entry member =>
        absent entry (List.mem_append_left _ member))]
      rw [right.outside idx (fun entry member =>
        absent entry (List.mem_append_right _ member))]
      exact add_full shape left.uniform right.uniform }

end Contributions

end Proofs.Autograd.Algebra

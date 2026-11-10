/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Op.Table.Rule

/-!
# Equation lemmas of the comparisons and logical operators

Each lemma here states one arm of `Op.rule?` for a comparison (`eq`, `ne`, `lt`, `le`, `gt` and
`ge`, at the scalar and at integer words), a predicate (`isNaN`, `isInf`, `isFinite`) or a logical
operator on Boolean tensors.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)
open TorchLean.Shape (size size_nil)

namespace TorchLean

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

namespace Op

/-- The rule of `Op.eq`. -/
theorem rule?_eq [Op.Capabilities α] {s : Shape} :
    (Op.eq (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.eq xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) } :=
  rfl

/-- The rule of `Op.eq` at integer words. -/
theorem rule?_eq_integer [Op.Capabilities α] {signed : Bool} {width : Nat} {s : Shape} :
    (Op.eq (s := s) (.integer signed width)).rule? (α := α) = some
      { forward := fun xs => (Tensor.zipWith (fun x y => decide (x = y)) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) } :=
  rfl

/-- The rule of `Op.ne`. -/
theorem rule?_ne [Op.Capabilities α] {s : Shape} :
    (Op.ne (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.ne xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) } :=
  rfl

/-- The rule of `Op.ne` at integer words. -/
theorem rule?_ne_integer [Op.Capabilities α] {signed : Bool} {width : Nat} {s : Shape} :
    (Op.ne (s := s) (.integer signed width)).rule? (α := α) = some
      { forward := fun xs => (Tensor.zipWith (fun x y => decide (x ≠ y)) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) } :=
  rfl

/-- The rule of `Op.lt`. -/
theorem rule?_lt [Op.Capabilities α] {s : Shape} :
    (Op.lt (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.lt xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) } :=
  rfl

/-- The rule of `Op.lt` at integer words, in the type's order. -/
theorem rule?_lt_integer [Op.Capabilities α] {signed : Bool} {width : Nat} {s : Shape} :
    (Op.lt (s := s) (.integer signed width)).rule? (α := α) = some
      { forward := fun xs => (Tensor.zipWith (Word.lt signed) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) } :=
  rfl

/-- The rule of `Op.le`. -/
theorem rule?_le [Op.Capabilities α] {s : Shape} :
    (Op.le (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.le xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) } :=
  rfl

/-- The rule of `Op.le` at integer words, in the type's order. -/
theorem rule?_le_integer [Op.Capabilities α] {signed : Bool} {width : Nat} {s : Shape} :
    (Op.le (s := s) (.integer signed width)).rule? (α := α) = some
      { forward := fun xs => (Tensor.zipWith (Word.le signed) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) } :=
  rfl

/-- The rule of `Op.gt`. -/
theorem rule?_gt [Op.Capabilities α] {s : Shape} :
    (Op.gt (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.gt xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) } :=
  rfl

/-- The rule of `Op.gt` at integer words, in the type's order. -/
theorem rule?_gt_integer [Op.Capabilities α] {signed : Bool} {width : Nat} {s : Shape} :
    (Op.gt (s := s) (.integer signed width)).rule? (α := α) = some
      { forward := fun xs => (Tensor.zipWith (fun x y => Word.lt signed y x) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) } :=
  rfl

/-- The rule of `Op.ge`. -/
theorem rule?_ge [Op.Capabilities α] {s : Shape} :
    (Op.ge (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.ge xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) } :=
  rfl

/-- The rule of `Op.ge` at integer words, in the type's order. -/
theorem rule?_ge_integer [Op.Capabilities α] {signed : Bool} {width : Nat} {s : Shape} :
    (Op.ge (s := s) (.integer signed width)).rule? (α := α) = some
      { forward := fun xs => (Tensor.zipWith (fun x y => Word.le signed y x) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) } :=
  rfl

/-- The rule of `Op.isNaN`. -/
theorem rule?_isNaN [Op.Capabilities α] {s : Shape} :
    (Op.isNaN (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.isNaN xs.1, ())
        backward := fun _ _ => ((), ()) } :=
  rfl

/-- The rule of `Op.isInf`. -/
theorem rule?_isInf [Op.Capabilities α] {s : Shape} :
    (Op.isInf (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.isInf xs.1, ())
        backward := fun _ _ => ((), ()) } :=
  rfl

/-- The rule of `Op.isFinite`. -/
theorem rule?_isFinite [Op.Capabilities α] {s : Shape} :
    (Op.isFinite (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.isFinite xs.1, ())
        backward := fun _ _ => ((), ()) } :=
  rfl

/-- The rule of `Op.logicalNot`. -/
theorem rule?_logicalNot [Op.Capabilities α] {s : Shape} :
    (Op.logicalNot (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.logicalNot xs.1, ())
        backward := fun _ _ => ((), ()) } :=
  rfl

/-- The rule of `Op.logicalAnd`. -/
theorem rule?_logicalAnd [Op.Capabilities α] {s : Shape} :
    (Op.logicalAnd (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.logicalAnd xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) } :=
  rfl

/-- The rule of `Op.logicalOr`. -/
theorem rule?_logicalOr [Op.Capabilities α] {s : Shape} :
    (Op.logicalOr (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.logicalOr xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) } :=
  rfl

/-- The rule of `Op.logicalXor`. -/
theorem rule?_logicalXor [Op.Capabilities α] {s : Shape} :
    (Op.logicalXor (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.logicalXor xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) } :=
  rfl

end Op

end TorchLean

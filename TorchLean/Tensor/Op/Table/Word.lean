/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Op.Table.Rule

/-!
# Equation lemmas of the word operators and conversions

Each lemma here states one arm of `Op.rule?` for an operator on integer words (the bitwise
operators, the shifts, `populationCount`, `countLeadingZeros` and `mulhi`) or a conversion
(`bitcast`, `cast`, `encodeIndex`, `decodeIndex`). `Op.rule?_bitcast_eq_none` and
`Op.rule?_decodeIndex_eq_none` state where a conversion has no rule.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)
open TorchLean.Shape (size size_nil)

namespace TorchLean

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

namespace Op

/-- The rule of `Op.bitwiseAnd`. -/
theorem rule?_bitwiseAnd [Op.Capabilities α] {signed : Bool} {width : Nat} {s : Shape} :
    (Op.bitwiseAnd (signed := signed) (width := width) (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.zipWith (· &&& ·) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) } :=
  rfl

/-- The rule of `Op.bitwiseOr`. -/
theorem rule?_bitwiseOr [Op.Capabilities α] {signed : Bool} {width : Nat} {s : Shape} :
    (Op.bitwiseOr (signed := signed) (width := width) (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.zipWith (· ||| ·) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) } :=
  rfl

/-- The rule of `Op.bitwiseXor`. -/
theorem rule?_bitwiseXor [Op.Capabilities α] {signed : Bool} {width : Nat} {s : Shape} :
    (Op.bitwiseXor (signed := signed) (width := width) (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.zipWith (· ^^^ ·) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) } :=
  rfl

/-- The rule of `Op.bitwiseLeftShift`. -/
theorem rule?_bitwiseLeftShift [Op.Capabilities α] {signed : Bool} {width : Nat} {s : Shape} :
    (Op.bitwiseLeftShift (signed := signed) (width := width) (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.zipWith Word.shiftLeft xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) } :=
  rfl

/-- The rule of `Op.shiftRightArithmetic`. -/
theorem rule?_shiftRightArithmetic [Op.Capabilities α] {signed : Bool} {width : Nat} {s : Shape} :
    (Op.shiftRightArithmetic (signed := signed) (width := width) (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.zipWith Word.shiftRightArithmetic xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) } :=
  rfl

/-- The rule of `Op.shiftRightLogical`. -/
theorem rule?_shiftRightLogical [Op.Capabilities α] {signed : Bool} {width : Nat} {s : Shape} :
    (Op.shiftRightLogical (signed := signed) (width := width) (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.zipWith Word.shiftRightLogical xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) } :=
  rfl

/-- The rule of `Op.bitwiseNot`. -/
theorem rule?_bitwiseNot [Op.Capabilities α] {signed : Bool} {width : Nat} {s : Shape} :
    (Op.bitwiseNot (signed := signed) (width := width) (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.map (~~~ ·) xs.1, ())
        backward := fun _ _ => ((), ()) } :=
  rfl

/-- The rule of `Op.populationCount`. -/
theorem rule?_populationCount [Op.Capabilities α] {signed : Bool} {width : Nat} {s : Shape} :
    (Op.populationCount (signed := signed) (width := width) (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.map BitVec.cpop xs.1, ())
        backward := fun _ _ => ((), ()) } :=
  rfl

/-- The rule of `Op.countLeadingZeros`. -/
theorem rule?_countLeadingZeros [Op.Capabilities α] {signed : Bool} {width : Nat} {s : Shape} :
    (Op.countLeadingZeros (signed := signed) (width := width) (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.map BitVec.clz xs.1, ())
        backward := fun _ _ => ((), ()) } :=
  rfl

/-- The rule of `Op.mulhi`. -/
theorem rule?_mulhi [Op.Capabilities α] {signed : Bool} {width : Nat} {s : Shape} :
    (Op.mulhi (signed := signed) (width := width) (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.zipWith (Word.mulhi signed) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) } :=
  rfl

/-- The rule of `Op.bitcast` between integer words: every word unchanged. -/
theorem rule?_bitcast_integer [Op.Capabilities α] {s : Shape} {width : Nat} {source target : Bool} :
    (Op.bitcast (s := s) width (.integer source) (.integer target)).rule? (α := α) = some
      { forward := fun xs => (xs.1, ())
        backward := fun _ _ => ((), ()) } :=
  rfl

/-- The rule of `Op.bitcast` from the scalar to integer words, where the scalar's words have
`width` bits. -/
theorem rule?_bitcast_scalar_integer [Bits α] {s : Shape} {width : Nat} {signed : Bool}
    (h : Numeric.Word.width α = width) :
    (Op.bitcast (s := s) width .scalar (.integer signed)).rule? (α := α) = some
      { forward := fun xs => (Tensor.map (fun (x : α) => (Numeric.Word.toBits x).cast h) xs.1, ())
        backward := fun _ _ => ((), ()) } := by
  subst h; exact dite_eq_left rfl

/-- The rule of `Op.bitcast` from integer words to the scalar, where the scalar's words have
`width` bits. -/
theorem rule?_bitcast_integer_scalar [Bits α] {s : Shape} {width : Nat} {signed : Bool}
    (h : Numeric.Word.width α = width) :
    (Op.bitcast (s := s) width (.integer signed) .scalar).rule? (α := α) = some
      { forward := fun xs =>
          (Tensor.map (fun (b : BitVec width) => Numeric.Word.ofBits (α := α) (b.cast h.symm))
            xs.1, ())
        backward := fun _ _ => ((), ())
        jvp? := some fun _ _ => (Tensor.full _ 0, ()) } := by
  subst h; exact dite_eq_left rfl

/-- The rule of `Op.bitcast` from the scalar to itself, where its words have `width` bits: every
value unchanged. -/
theorem rule?_bitcast_scalar [Bits α] {s : Shape} {width : Nat}
    (h : Numeric.Word.width α = width) :
    (Op.bitcast (s := s) width .scalar .scalar).rule? (α := α) = some
      { forward := fun xs => (xs.1, ())
        backward := fun _ _ => ((), ())
        jvp? := some fun _ _ => (Tensor.full _ 0, ()) } :=
  ite_eq_left h

/-- `Op.bitcast` with a scalar side has no rule where the scalar's words do not have `width`
bits. -/
theorem rule?_bitcast_eq_none [Bits α] {s : Shape} {width : Nat}
    (h : Numeric.Word.width α ≠ width)
    {source target : Op.Word} (hside : source = .scalar ∨ target = .scalar) :
    (Op.bitcast (s := s) width source target).rule? (α := α) = none := by
  rcases hside with rfl | rfl
  · cases target
    · exact ite_eq_right h
    · exact dite_eq_right h
  · cases source
    · exact ite_eq_right h
    · exact dite_eq_right h

/-- The rule of `Op.cast` from Booleans. -/
theorem rule?_cast_bool [Op.Capabilities α] {s : Shape} :
    (Op.cast (s := s) .bool).rule? (α := α) = some
      { forward := fun xs => (Tensor.map (fun b => if b then (1 : α) else 0) xs.1, ())
        backward := fun _ _ => ((), ())
        jvp? := some fun _ _ => (Tensor.full _ 0, ()) } :=
  rfl

/-- The rule of `Op.cast` from indices. -/
theorem rule?_cast_index [Op.Capabilities α] {s : Shape} (bound : Nat) :
    (Op.cast (s := s) (.index bound)).rule? (α := α) = some
      { forward := fun xs => (Tensor.map (fun j => ((j.val : Nat) : α)) xs.1, ())
        backward := fun _ _ => ((), ())
        jvp? := some fun _ _ => (Tensor.full _ 0, ()) } :=
  rfl

/-- The rule of `Op.encodeIndex`: the words of the components, after the check that each fits the
integer type. -/
theorem rule?_encodeIndex [Op.Capabilities α] {s : Shape} {bounds : List Nat} {signed : Bool}
    {width : Nat} :
    (Op.encodeIndex (s := s) bounds signed width).rule? (α := α) = some
      { forward := fun xs => (Op.indexWords width xs.1, ())
        backward := fun _ _ => ((), ())
        validate := fun xs => Op.checkIndexWords signed width xs.1 } :=
  rfl

/-- The rule of `Op.decodeIndex` where its output has a value: the tuples of the words, after the
check that each component lies in `[0, bound)`. -/
theorem rule?_decodeIndex [Op.Capabilities α] {s : Shape} {bounds : List Nat} {signed : Bool}
    {width : Nat} (h : (∀ b ∈ bounds, 0 < b) ∨ s.size = 0) :
    (Op.decodeIndex (s := s) bounds signed width).rule? (α := α) = some
      { forward := fun xs => (Op.wordIndicesOr h signed xs.1, ())
        backward := fun _ _ => ((), ())
        validate := fun xs => (Op.wordIndices signed xs.1).map fun _ => () } :=
  dite_eq_left h

/-- `Op.decodeIndex` has no rule where a bound is zero and its output has entries: no tuple
exists there. -/
theorem rule?_decodeIndex_eq_none [Op.Capabilities α] {s : Shape} {bounds : List Nat}
    {signed : Bool} {width : Nat} (h : ¬((∀ b ∈ bounds, 0 < b) ∨ s.size = 0)) :
    (Op.decodeIndex (s := s) bounds signed width).rule? (α := α) = none :=
  dite_eq_right h

end Op

end TorchLean

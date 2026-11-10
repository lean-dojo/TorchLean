/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Op.Table.Word

/-!
# Bitcast round trips

The scalar-side `Op.bitcast` rules of `TorchLean.Op.rule?`, at the width of the scalar's words, are
inverse to each other: integer words through the scalar and back are unchanged
(`bitcast_integer_scalar_integer`), and so are scalars through integer words and back
(`bitcast_scalar_integer_scalar`).
-/

@[expose] public section

namespace TorchLean.Op

/-- Integer words through the scalar and back, by the scalar-side `Op.bitcast` rules at the width of
the scalar's words, are unchanged. -/
theorem bitcast_integer_scalar_integer {α : Type} [Storage α] [Context α] [Bits α] {s : Shape}
    {width : Nat} {source target : Bool} (h : Numeric.Word.width α = width)
    (xs : Tensor (BitVec width) s) :
    (((Op.bitcast (s := s) width (.integer source) .scalar).rule? (α := α)).bind fun toScalar =>
      ((Op.bitcast (s := s) width .scalar (.integer target)).rule? (α := α)).map fun toWord =>
        (toWord.forward ((toScalar.forward (xs, ())).1, ())).1) = some xs := by
  rw [Op.rule?_bitcast_integer_scalar h, Op.rule?_bitcast_scalar_integer h]
  subst h
  simp only [Option.bind_some, Option.map_some, Option.some.injEq,
    Tensor.Rep.map_map]
  have hid : ((fun x : α => (Numeric.Word.toBits x).cast rfl) ∘
      fun b : BitVec (Numeric.Word.width α) => Numeric.Word.ofBits (α := α) (b.cast rfl)) = id :=
    funext fun b => by simp [Bits.toBits_ofBits]
  rw [hid, Tensor.Rep.map_id]

/-- Scalars through integer words and back, by the scalar-side `Op.bitcast` rules at the width of
the scalar's words, are unchanged. -/
theorem bitcast_scalar_integer_scalar {α : Type} [Storage α] [Context α] [Bits α] {s : Shape}
    {width : Nat} {source target : Bool} (h : Numeric.Word.width α = width) (xs : Tensor α s) :
    (((Op.bitcast (s := s) width .scalar (.integer source)).rule? (α := α)).bind fun toWord =>
      ((Op.bitcast (s := s) width (.integer target) .scalar).rule? (α := α)).map fun toScalar =>
        (toScalar.forward ((toWord.forward (xs, ())).1, ())).1) = some xs := by
  rw [Op.rule?_bitcast_scalar_integer h, Op.rule?_bitcast_integer_scalar h]
  subst h
  simp only [Option.bind_some, Option.map_some, Option.some.injEq,
    Tensor.Rep.map_map]
  have hid : ((fun b : BitVec (Numeric.Word.width α) => Numeric.Word.ofBits (α := α) (b.cast rfl)) ∘
      fun x : α => (Numeric.Word.toBits x).cast rfl) = id :=
    funext fun x => by simp [Numeric.Word.ofBits_toBits]
  rw [hid, Tensor.Rep.map_id]

end TorchLean.Op

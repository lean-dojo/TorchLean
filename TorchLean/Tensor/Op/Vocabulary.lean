/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Op.Table.Linear
public import TorchLean.Tensor.Op.Table.Word

/-!
# Which operators have rules, and their output ports

Every operator has a rule at a scalar with every capability but `Cbrt`, `NextAfter` and `Bits`,
with seven exceptions (`rule?_isSome_or_custom`). They are `custom`, `cbrt`, `nextafter`, a
`bitcast` with a scalar side, a `decodeIndex` with a zero bound and entries, an integer `div`
without rounding and a Boolean `scatter` other than `replace`. With those three classes `cbrt`,
`nextafter` and every `bitcast` at the scalar's word width have rules
(`rule?_isSome_cbrt_nextafter_bitcast`). Every operator has one or two output ports
(`outputs_length`).
-/

@[expose] public section

namespace TorchLean.Op

/-- Every operator but `custom`, `cbrt`, `nextafter`, a `bitcast` with a scalar side, a
`decodeIndex` with a zero bound and entries, an integer `div` without rounding and a Boolean
`scatter` other than `replace` has a rule at a scalar with every capability but `Cbrt`,
`NextAfter` and `Bits`. -/
theorem rule?_isSome_or_custom {α : Type} [Storage α] [Context α] [Rounding α] [Erf α]
    [InverseTrig α] [DecidableEq α] {ins outs : List Port} (op : Op ins outs) :
    (op.rule? (α := α)).isSome ∨ (∃ name inputs output, HEq op (Op.custom name inputs output)) ∨
      (∃ s, HEq op (Op.cbrt (s := s))) ∨ (∃ s, HEq op (Op.nextafter (s := s))) ∨
      (∃ s width source target, (source = .scalar ∨ target = .scalar) ∧
        HEq op (Op.bitcast (s := s) width source target)) ∨
      (∃ s bounds signed width, ¬((∀ b ∈ bounds, 0 < b) ∨ s.size = 0) ∧
        HEq op (Op.decodeIndex (s := s) bounds signed width)) ∨
      (∃ s signed width, HEq op (Op.div (s := s) .none (.integer signed width))) ∨
      ∃ s t rule dims h, rule ≠ .replace ∧
        HEq op (Op.scatter (s := s) (t := t) rule dims h .bool) := by
  cases op
  case custom name inputs output => exact Or.inr (Or.inl ⟨name, inputs, output, HEq.rfl⟩)
  case cbrt s => exact Or.inr (Or.inr (Or.inl ⟨s, HEq.rfl⟩))
  case nextafter s => exact Or.inr (Or.inr (Or.inr (Or.inl ⟨s, HEq.rfl⟩)))
  case bitcast s width source target =>
    cases source
    · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl
        ⟨s, width, .scalar, target, Or.inl rfl, HEq.rfl⟩))))
    · cases target
      · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl
          ⟨s, width, _, .scalar, Or.inr rfl, HEq.rfl⟩))))
      · exact Or.inl rfl
  case decodeIndex s bounds signed width =>
    by_cases h : (∀ b ∈ bounds, 0 < b) ∨ s.size = 0
    · exact Or.inl ((congrArg Option.isSome (Op.rule?_decodeIndex (α := α) (s := s)
        (signed := signed) (width := width) h)).trans rfl)
    · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl
        ⟨s, bounds, signed, width, h, HEq.rfl⟩)))))
  case linear => exact Or.inl Op.rule?_linear_isSome
  case div rounding e =>
    cases rounding <;> cases e <;> first
      | exact Or.inl rfl
      | exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨_, _, _, HEq.rfl⟩))))))
  case scatter rule dims h e =>
    cases e <;> try exact Or.inl rfl
    cases rule <;> first
      | exact Or.inl rfl
      | exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr
          ⟨_, _, _, _, _, by decide, HEq.rfl⟩))))))
  case gelu approximate => cases approximate <;> exact Or.inl rfl
  case cast source => cases source <;> exact Or.inl rfl
  case sqrt policy => cases policy <;> exact Or.inl rfl
  all_goals first
    | exact Or.inl rfl
    | cases ‹Op.NumericElement› <;> exact Or.inl rfl
    | cases ‹Op.Element› <;> exact Or.inl rfl

/-- At a scalar with `Cbrt`, `NextAfter` and `Bits`, `cbrt`, `nextafter` and every `bitcast` at the
scalar's word width have rules. -/
theorem rule?_isSome_cbrt_nextafter_bitcast {α : Type} [Storage α] [Context α] [Cbrt α]
    [NextAfter α] [Bits α] (s : Shape) (source target : Op.Word) :
    ((Op.cbrt (s := s)).rule? (α := α)).isSome ∧ ((Op.nextafter (s := s)).rule? (α := α)).isSome ∧
      ((Op.bitcast (s := s) (Numeric.Word.width α) source target).rule? (α := α)).isSome := by
  refine ⟨rfl, rfl, ?_⟩
  cases source <;> cases target
  · exact (congrArg Option.isSome (Op.rule?_bitcast_scalar (α := α) (s := s) rfl)).trans rfl
  · exact (congrArg Option.isSome
      (Op.rule?_bitcast_scalar_integer (α := α) (s := s) rfl)).trans rfl
  · exact (congrArg Option.isSome
      (Op.rule?_bitcast_integer_scalar (α := α) (s := s) rfl)).trans rfl
  · rfl

/-- Every operator has one or two output ports. -/
theorem outputs_length {ins outs : List Port} (op : Op ins outs) :
    outs.length = 1 ∨ outs.length = 2 := by
  cases op <;> simp

end TorchLean.Op

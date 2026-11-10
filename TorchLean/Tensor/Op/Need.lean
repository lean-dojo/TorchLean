/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Op.Capability
public import TorchLean.Tensor.Op.Codec
public import TorchLean.Tensor.Op.Table.Rule

/-!
# The capability an operator needs, and random keys

`Op.need op` is the optional capability of the scalar (`Op.Capabilities`) that `Op.rule?` reads for
`op`: `Rounding` for the roundings and for the rounded divisions, `fmod` and `remainder` at the
scalar, `Erf` for `erf` and the exact GELU, `InverseTrig` for the inverse trigonometric functions
and `atan2`, `Cbrt` for `cbrt`, `NextAfter` for `nextafter`, `Bits` of the operator's width for a
`bitcast` with a scalar side, and none for every other operator. An instance of
`Op.Need.Supplied α n` is a capability record that holds `n`, found from `Rounding α`, `Erf α`,
`InverseTrig α`, `Cbrt α`, `NextAfter α` or `Bits α`, so `Op.Supported α op` is
synthesized at a call site that names the operator exactly when the scalar has what it needs. The
eager CPU tape and the typed graph run `Op.rule?` with that record, the eager CUDA tape runs the
operator's LibTorch entry, and the verifier recorder reads no record: it lowers only operators whose
lowering reads no optional capability. `Op.isSome_rule?_iff` states that the classification agrees
with the whole table: for every operator with `Op.hasRule` (all but `custom`, a `decodeIndex` with a
zero bound and entries, an integer true division and a Boolean scatter other than a replacement),
`Op.rule?` has a rule at a capability record exactly when the record holds the capability the
operator needs.

`Op.withKey? op` replaces the key of a random operator (`randUniform`, `randn`, `bernoulliMask`)
and is `none` for every other operator. The replacement keeps the capability the operator needs
(`Op.need_withKey`) and its wire tag (`Op.encode_withKey_tag`).
-/

@[expose] public section

namespace TorchLean

/-- An optional capability of the scalar that the rule of an operator reads. -/
inductive Op.Need where
  /-- No capability. -/
  | none
  /-- The integral roundings, `Rounding`. -/
  | rounding
  /-- The error function, `Erf`. -/
  | erf
  /-- The inverse trigonometric functions, `InverseTrig`. -/
  | inverseTrig
  /-- The cube root, `Cbrt`. -/
  | cbrt
  /-- C's `nextafter`, `NextAfter`. -/
  | nextafter
  /-- The interchange word of `width` bits: `Bits α` with `Numeric.Word.width α = width`. -/
  | bits (width : Nat)
  deriving DecidableEq, Repr, Inhabited

/-- The capability `Op.rule?` reads for `op`, `.none` for an operator that reads none. -/
@[reducible] def Op.need : {ins outs : List Port} → Op ins outs → Op.Need
  | _, _, .div .trunc | _, _, .div .floor => .rounding
  | _, _, .floor | _, _, .ceil | _, _, .round | _, _, .trunc | _, _, .frac => .rounding
  | _, _, .fmod | _, _, .remainder => .rounding
  | _, _, .atan | _, _, .asin | _, _, .acos | _, _, .atan2 => .inverseTrig
  | _, _, .erf | _, _, .gelu .none => .erf
  | _, _, .cbrt => .cbrt
  | _, _, .nextafter => .nextafter
  | _, _, .bitcast width .scalar _ | _, _, .bitcast width _ .scalar => .bits width
  | _, _, _ => .none

/-- Whether the capability record `caps` holds the capability `n`. -/
def Op.Need.Available {α : Type} (n : Op.Need) (caps : Op.Capabilities α) : Prop :=
  match n with
  | .none => True
  | .rounding => caps.rounding?.isSome = true
  | .erf => caps.erf?.isSome = true
  | .inverseTrig => caps.inverseTrig?.isSome = true
  | .cbrt => caps.cbrt?.isSome = true
  | .nextafter => caps.nextafter?.isSome = true
  | .bits width => (caps.bits?.any fun b => decide (b.width = width)) = true

/-- Whether a capability record holds a capability is decided from its options. -/
instance Op.Need.decidableAvailable {α : Type} (n : Op.Need) (caps : Op.Capabilities α) :
    Decidable (n.Available caps) :=
  match n with
  | .none => inferInstanceAs (Decidable True)
  | .rounding => inferInstanceAs (Decidable (caps.rounding?.isSome = true))
  | .erf => inferInstanceAs (Decidable (caps.erf?.isSome = true))
  | .inverseTrig => inferInstanceAs (Decidable (caps.inverseTrig?.isSome = true))
  | .cbrt => inferInstanceAs (Decidable (caps.cbrt?.isSome = true))
  | .nextafter => inferInstanceAs (Decidable (caps.nextafter?.isSome = true))
  | .bits width => inferInstanceAs
      (Decidable ((caps.bits?.any fun b => decide (b.width = width)) = true))

/-- A capability record of the scalar `α` that holds the capability `n`. The eager CPU tape and
the typed graph run `Op.rule?` with `caps`, and the eager CUDA tape and the verifier recorder never
read it. -/
class Op.Need.Supplied (α : Type) (n : Op.Need) where
  /-- The capability record. -/
  caps : Op.Capabilities α
  /-- The record holds `n`. -/
  available : n.Available caps

/-- The record without capabilities, for an operator that needs none. -/
instance Op.Need.Supplied.empty {α : Type} : Op.Need.Supplied α .none :=
  ⟨⟨Option.none, Option.none, Option.none, Option.none, Option.none, Option.none⟩,
    trivial⟩

/-- The record holding the scalar's `Rounding`. -/
instance Op.Need.Supplied.rounding {α : Type} [instance_ : Rounding α] :
    Op.Need.Supplied α .rounding :=
  ⟨Op.Capabilities.ofCapability, rfl⟩

/-- The record holding the scalar's `Erf`. -/
instance Op.Need.Supplied.erf {α : Type} [instance_ : Erf α] : Op.Need.Supplied α .erf :=
  ⟨Op.Capabilities.ofCapability, rfl⟩

/-- The record holding the scalar's `InverseTrig`. -/
instance Op.Need.Supplied.inverseTrig {α : Type} [instance_ : InverseTrig α] :
    Op.Need.Supplied α .inverseTrig :=
  ⟨Op.Capabilities.ofCapability, rfl⟩

/-- The record holding the scalar's `Cbrt`. -/
instance Op.Need.Supplied.cbrt {α : Type} [instance_ : Cbrt α] : Op.Need.Supplied α .cbrt :=
  ⟨Op.Capabilities.ofCapability, rfl⟩

/-- The record holding the scalar's `NextAfter`. -/
instance Op.Need.Supplied.nextafter {α : Type} [instance_ : NextAfter α] :
    Op.Need.Supplied α .nextafter :=
  ⟨Op.Capabilities.ofCapability, rfl⟩

/-- The record holding the scalar's `Bits`, for the width of its words. -/
instance Op.Need.Supplied.bits {α : Type} [instance_ : Bits α] :
    Op.Need.Supplied α (.bits (Numeric.Word.width α)) :=
  ⟨Op.Capabilities.ofCapability, decide_eq_true rfl⟩

/-- A capability record holding what `op` needs at the scalar `α`. -/
abbrev Op.Supported (α : Type) {ins outs : List Port} (op : Op ins outs) :=
  Op.Need.Supplied α op.need

/-- Whether `Op.rule?` can give `op` a rule: not for a custom node, whose rule is the checked
program of its node, nor for a `decodeIndex` with a zero bound and entries, whose output has no
value, nor for an integer true division or a Boolean scatter other than a replacement, which have no
meaning. -/
def Op.hasRule : {ins outs : List Port} → Op ins outs → Bool
  | _, _, .custom .. => false
  | _, _, @Op.decodeIndex s bounds _ _ => decide ((∀ b ∈ bounds, 0 < b) ∨ s.size = 0)
  | _, _, .div .none (.integer ..) => false
  | _, _, .scatter rule _ _ .bool => decide (rule = .replace)
  | _, _, _ => true

/-- The operator with its random key replaced, for `randUniform`, `randn` and `bernoulliMask`, and
`none` for an operator without a key. -/
def Op.withKey? : {ins outs : List Port} → Op ins outs → Option (UInt64 → Op ins outs)
  | _, _, .randUniform _ => some fun key => .randUniform key
  | _, _, .randn _ => some fun key => .randn key
  | _, _, .bernoulliMask _ => some fun key => .bernoulliMask key
  | _, _, _ => Option.none

namespace Op

variable {α : Type} [Storage α] [Context α]

/-- The capability classification agrees with the table: for every operator with
`op.hasRule = true`, `Op.rule?` at the capability record `caps` has a rule exactly when `caps` holds
what the operator needs. -/
theorem isSome_rule?_iff (caps : Op.Capabilities α) {ins outs : List Port} (op : Op ins outs)
    (h : op.hasRule = true) :
    (@Op.rule? α _ _ caps ins outs op).isSome = true ↔ op.need.Available caps := by
  obtain ⟨rounding, erf, inverseTrig, cbrt, nextafter, bits⟩ := caps
  cases op
  case div mode e => cases mode <;> cases e <;> first
    | exact iff_of_true rfl trivial
    | cases rounding <;> rfl
    | simp [Op.hasRule] at h
  case gelu mode => cases mode <;> first | exact iff_of_true rfl trivial | cases erf <;> rfl
  case linear lead _ _ => exact iff_of_true (by cases lead <;> rfl) trivial
  case cast source => exact iff_of_true (by cases source <;> rfl) trivial
  case sqrt policy => exact iff_of_true (by cases policy <;> rfl) trivial
  case custom => cases h
  case decodeIndex => exact iff_of_true (Option.isSome_dite.2 (of_decide_eq_true h)) trivial
  case scatter r _ _ e => cases e <;> cases r <;> first
    | exact iff_of_true rfl trivial
    | simp [Op.hasRule] at h
  case bitcast width source target =>
    cases source <;> cases target <;> first
      | exact iff_of_true rfl trivial
      | cases bits with
        | none => rfl
        | some b => exact Option.isSome_dite.trans decide_eq_true_iff.symm
  all_goals (try cases ‹Op.Element›) <;> (try cases ‹Op.NumericElement›) <;> first
    | exact iff_of_true rfl trivial
    | cases rounding <;> rfl
    | cases erf <;> rfl
    | cases inverseTrig <;> rfl
    | cases cbrt <;> rfl
    | cases nextafter <;> rfl

/-- Replacing the key of a random operator keeps the capability it needs. -/
theorem need_withKey {ins outs : List Port} {op : Op ins outs} {f : UInt64 → Op ins outs}
    (h : op.withKey? = some f) (key : UInt64) : (f key).need = op.need := by
  cases op <;> cases h <;> rfl

/-- Replacing the key of a random operator keeps its wire tag. -/
theorem encode_withKey_tag {ins outs : List Port} {op : Op ins outs} {f : UInt64 → Op ins outs}
    (h : op.withKey? = some f) (key : UInt64) :
    (Op.encode ⟨ins, outs, f key⟩).tag = (Op.encode ⟨ins, outs, op⟩).tag := by
  cases op <;> cases h <;> rfl

end Op

end TorchLean

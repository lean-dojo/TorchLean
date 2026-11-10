/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Init.Data.BitVec.Lemmas
public import TorchLean.NN.IR.DType
public import TorchLean.Tensor.ConfiguredBinary
public import TorchLean.Tensor.Word
public import TorchLean.Tensor.Op.Capability
public import TorchLean.Tensor.Op.Port
public import FloatLib.Floats.ExecFloat.Conversion.Core
public import FloatLib.Floats.Formats.BinaryInterchange.Conversion.Cast.Proof
public import FloatLib.Floats.Formats.BinaryInterchange.Conversion.Cast.Runtime
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Value.CoreProof

/-!
# Element types of the reference runtime

The carriers, words and conversions of the IR's element types, below the reference runtime of
typed programs. The binary carriers are FloatLib's configured formats (`Binary.Scalar`).
`Binary.cast` is FloatLib's exact-source binary conversion: it rounds the source's interchange word
once in the destination format and keeps its IEEE exception flags, and `Binary.cast_value` proves
that it returns FloatLib's specified directed rounding, exceptional encodings included. Neither
constants nor conversions pass through binary64.

The reference carriers (`Reference.Carrier`) add Booleans as `Bool` and integers of any positive
width as Lean's `BitVec`. Their signedness is part of the `DType`, not of a second carrier.
Integers and Booleans convert to binary formats by rounding their exact value once. A binary value
converts to an integer type by rounding its finite value to an integer in the cast's direction
(toward zero is C++'s `static_cast`, which PyTorch's conversions use), and fails at a NaN, an
infinity or an integer the type cannot hold. PyTorch's conversion is undefined at those values,
except to uint8, which `c10::convert` reaches through int64 (the PyTorch capture writes that route
as two casts). Integer types convert among themselves modulo `2 ^ width`, as `static_cast` does,
Booleans to integers as zero and one, and integers and binary values to Booleans as `x != 0`.

Element words are unsigned, including signed integer encodings, and Boolean words are zero or
one. Loading rejects excessive words, invalid descriptors and incompatible tensor sizes, and
`Reference.read_write` proves that encoding a decoded word gives the word back. The reference
runtime `TorchLean.IR.Program.Machine.reference` takes the storage, arithmetic and capabilities of
each element type from `Reference.storage`, `Reference.arithmetic` and `Reference.capabilities`.
-/

@[expose] public section

namespace TorchLean.IR

open FloatLib.Floats
open FloatLib.Floats.Formats.BinaryInterchange

namespace Binary

/-- FloatLib's configured representation for the complete validated format descriptor. -/
abbrev Scalar (format : FloatFormat) :=
  ExecFloat.Binary format.expWidth format.fracWidth format.encoding format.exponentBias
    format.expWidth_ge_two format.fracWidth_pos format.exponentBias_pos
    format.exponentBias_le_maxFinite

/-- FloatLib's exact-source binary conversion, retaining its IEEE exception flags. -/
def cast (source target : FloatFormat) (rounding : Model.IEEERoundingMode)
    (value : Scalar source) : Scalar target × ExecFloat.ConversionStatus :=
  let outcome := Model.cast source target (ExecFloat.Binary.toModel value)
    (mode := rounding) (status := true)
  (ExecFloat.Binary.ofModel outcome.value,
    { inexact := outcome.status.inexact
      overflow := outcome.status.overflow
      underflow := outcome.status.underflow
      invalid := outcome.status.invalid })

/-- The value of the binary cast is FloatLib's directed conversion `Model.cast` of the value of its
argument. -/
theorem cast_value (source target : FloatFormat) (rounding : Model.IEEERoundingMode)
    (value : Scalar source) :
    ExecFloat.Binary.toModel (cast source target rounding value).1 =
      Model.cast source target (ExecFloat.Binary.toModel value) (mode := rounding) := by
  simp only [cast, ExecFloat.Binary.toModel_ofModel, Model.value_cast]

end Binary

namespace Reference

/-- Configured binary values, Boolean values, and arbitrary-width integer bit vectors. -/
abbrev Carrier : DType → Type
  | .binary format => Binary.Scalar format
  | .bool => Bool
  | .integer _ (some width) => BitVec width
  | _ => Empty

/-- Whether `dtype` is valid and the reference runtime represents it: a binary format, the
Booleans or an integer type of fixed width. -/
def supported (dtype : DType) : Bool :=
  dtype.valid && match dtype with
    | .binary _ | .bool | .integer _ (some _) => true
    | _ => false

/-- Decode one element word. A word with more bits than the element type has is refused. -/
def read : (dtype : DType) → Nat → Except String (Carrier dtype)
  | .binary format, word =>
      if word < 2 ^ format.bitWidth then pure (ExecFloat.Binary.ofNatBits word)
      else throw "reference IR: word exceeds its binary format"
  | .bool, word =>
      if word < 2 then pure (word != 0)
      else throw "reference IR: Boolean word must be zero or one"
  | .integer signed (some width), word => do
      (DType.integer signed (some width)).check
      if h : word < 2 ^ width then pure (BitVec.ofNatLT word h)
      else throw "reference IR: word exceeds its integer width"
  | dtype, _ => throw s!"reference IR: no word representation for {dtype}"

/-- Encode one element as its unsigned word: the bits of a binary value, zero or one for a
Boolean, and the bit pattern of an integer. -/
def write : (dtype : DType) → Carrier dtype → Except String Nat
  | .binary _, value => .ok (ExecFloat.Binary.toNatBits value)
  | .bool, value => .ok (if value then 1 else 0)
  | .integer signed (some width), value => do
      (DType.integer signed (some width)).check
      pure value.toNat
  | dtype, _ => .error s!"reference IR: no word representation for {dtype}"

/-- Round an exact integer once to a binary format, reporting inexactness and overflow. -/
def integerToBinary (target : FloatFormat) (rounding : Model.IEEERoundingMode) (value : Int) :
    Binary.Scalar target × ExecFloat.ConversionStatus :=
  let rounded := Model.ofRat target (value : Rat) (mode := rounding)
  let status := Model.dyadicRoundingStatus target rounding
    (FloatLib.Numerics.Dyadic.ofScaledInt value 0) rounded
  (ExecFloat.Binary.ofModel rounded,
    { inexact := status.inexact
      overflow := status.overflow
      underflow := status.underflow
      invalid := status.invalid })

/-- The element of the integer type of `width` bits whose value is `n`, or `none` when `n` does
not fit. Nothing is reduced modulo `2 ^ width`: at a positive width the element's value is `n`
(`toInteger_ofInteger`). -/
def ofInteger (signed : Bool) (width : Nat) (n : Int) : Option (BitVec width) :=
  if Word.fits signed width n then some (BitVec.ofInt width n) else none

/-- `ofInteger` fails exactly where `n` does not fit. -/
theorem ofInteger_eq_none_iff {signed : Bool} {width : Nat} {n : Int} :
    ofInteger signed width n = none ↔ Word.fits signed width n = false := by
  unfold ofInteger
  split <;> simp_all

/-- The element `ofInteger` returns has the value `n`, at every positive width. -/
theorem toInteger_ofInteger {signed : Bool} {width : Nat} {n : Int} {value : BitVec width}
    (hwidth : 0 < width) (h : ofInteger signed width n = some value) :
    Word.toInteger signed value = n := by
  unfold ofInteger at h
  split at h
  · rename_i hfits
    cases h
    cases signed
    · simp only [Word.fits, decide_eq_true_eq, Bool.false_eq_true, ↓reduceIte] at hfits
      simp only [Word.toInteger, Bool.false_eq_true, ↓reduceIte, BitVec.toNat_ofInt]
      rw [Int.emod_eq_of_lt hfits.1 (by exact_mod_cast hfits.2)]
      exact Int.toNat_of_nonneg hfits.1
    · simp only [Word.fits, decide_eq_true_eq, ↓reduceIte] at hfits
      simp only [Word.toInteger, ↓reduceIte, BitVec.toInt_ofInt]
      have hpow : ((2 ^ width : Nat) : Int) = 2 * 2 ^ (width - 1) := by
        rw [Nat.cast_pow, Nat.cast_ofNat, ← pow_succ']
        congr 1
        omega
      exact Int.bmod_eq_of_le_mul_two (by omega) (by omega)
  · cases h

/-- Every element is the element of its value, at every width. -/
theorem ofInteger_toInteger {signed : Bool} {width : Nat} (value : BitVec width) :
    ofInteger signed width (Word.toInteger signed value) = some value := by
  cases signed
  · have hfits : Word.fits false width (value.toNat : Int) = true := by
      simp only [Word.fits, Bool.false_eq_true, ↓reduceIte, decide_eq_true_eq]
      exact ⟨by omega, by exact_mod_cast value.isLt⟩
    simp [ofInteger, Word.toInteger, hfits]
  · have hfits : Word.fits true width value.toInt = true := by
      simp only [Word.fits, ↓reduceIte, decide_eq_true_eq]
      exact ⟨by exact_mod_cast BitVec.le_toInt value, by exact_mod_cast BitVec.toInt_lt⟩
    simp [ofInteger, Word.toInteger, hfits]

/-- A binary value as an element of an integer type: its exact value rounded to an integer in the
direction `rounding` (`Model.roundDyadicToInt`), with the inexact flag where the value is no
integer. It is an error at a NaN, an infinity or an integer the type cannot hold. -/
def binaryToInteger {format : FloatFormat} (signed : Bool) (width : Nat)
    (rounding : Model.IEEERoundingMode) (value : Binary.Scalar format) :
    Except String (BitVec width × ExecFloat.ConversionStatus) :=
  match Model.toDyadic? (ExecFloat.Binary.toModel value) with
  | none => .error "reference IR: a NaN or an infinity has no integer value"
  | some d =>
      let n := Model.roundDyadicToInt rounding d
      match ofInteger signed width n with
      | some word => .ok (word, { inexact := !Model.dyadicIsIntegral d })
      | none => .error s!"reference IR: the integer {n} does not fit the target type"

/-- Conversion between element types of the reference runtime, with its status flags. Binary
formats convert by FloatLib's directed binary conversion, integers and Booleans to binary formats
by rounding their exact value once, binary values to integers in the cast's direction
(`binaryToInteger`), integer types among themselves modulo `2 ^ width` (the `wrapped` flag),
Booleans to integers as zero and one, and integers and binary values to Booleans as `x != 0`.

Any other conversion needs identical source and target types. A binary conversion goes through
FloatLib even when the formats are equal, because exceptional-value handling and flags are part of
the conversion semantics.
-/
def cast (source target : DType) (rounding : Model.IEEERoundingMode)
    (value : Carrier source) : Except String (Carrier target × ExecFloat.ConversionStatus) :=
  match source, target with
  | .binary src, .binary dst => pure (Binary.cast src dst rounding value)
  | .integer signed (some width), .binary dst => do
      (DType.integer signed (some width)).check
      pure (integerToBinary dst rounding (if signed then value.toInt else value.toNat))
  | .bool, .binary dst => pure (integerToBinary dst rounding (if value then 1 else 0))
  | .binary _, .integer signed (some width) => do
      (DType.integer signed (some width)).check
      binaryToInteger signed width rounding value
  | .bool, .integer signed (some width) => do
      (DType.integer signed (some width)).check
      match ofInteger signed width (if value then 1 else 0) with
      | some word => pure (word, {})
      | none => throw "reference IR: one does not fit the target type"
  | .integer signed (some width), .integer signed' (some width') => do
      (DType.integer signed (some width)).check
      (DType.integer signed' (some width')).check
      let n := Word.toInteger signed value
      pure (BitVec.ofInt width' n, { wrapped := !Word.fits signed' width' n })
  | .integer signed (some width), .bool => do
      (DType.integer signed (some width)).check
      pure (value != 0, {})
  | .binary _, .bool => pure (Model.toRat? (ExecFloat.Binary.toModel value) != some 0, {})
  | source, target =>
      if supported source && supported target then
        if h : source = target then by
          cases h
          exact .ok (value, {})
        else .error s!"reference IR: conversion {source} → {target} is not implemented"
      else .error s!"reference IR: no representation for {source} or {target}"

/-- Encoding the element decoded from a word gives back that word. -/
theorem read_write (dtype : DType) (word : Nat) (value : Carrier dtype)
    (h : read dtype word = .ok value) : write dtype value = .ok word := by
  cases dtype with
  | binary format =>
      simp only [read] at h
      split at h
      · cases h
        simp only [write, ExecFloat.Binary.toNatBits_ofNatBits_of_lt word ‹_›]
      · cases h
  | bool =>
      simp only [read] at h
      split at h
      · have hw : word = 0 ∨ word = 1 := by omega
        rcases hw with rfl | rfl <;> simp_all [write, Pure.pure, Except.pure]
      · cases h
  | integer signed width =>
      cases width with
      | none => cases value
      | some width =>
          simp only [read, Bind.bind, Except.bind] at h
          cases hc : DType.check (.integer signed (some width)) with
          | error error => simp [hc] at h
          | ok checked =>
              cases checked
              simp only [hc] at h
              split at h
              · cases h
                simp only [write, hc, Bind.bind, Except.bind, Pure.pure, Except.pure,
                  BitVec.toNat_ofNatLT]
              · cases h
  | context | rational | real | decimal _ | posit _ | complex _ | named _ _ => cases value

/-- A successful conversion of a binary value to an integer type of positive width returns the
element whose value is the binary value's exact dyadic value rounded to an integer in the cast's
direction (`Model.roundDyadicToInt`). -/
theorem toInteger_cast_binary {format : FloatFormat} {signed : Bool} {width : Nat}
    {rounding : Model.IEEERoundingMode} {value : Binary.Scalar format} {word : BitVec width}
    {flags : ExecFloat.ConversionStatus} (hwidth : 0 < width)
    (h : cast (.binary format) (.integer signed (some width)) rounding value = .ok (word, flags)) :
    ∃ d, Model.toDyadic? (ExecFloat.Binary.toModel value) = some d ∧
      Word.toInteger signed word = Model.roundDyadicToInt rounding d := by
  have hcast : cast (.binary format) (.integer signed (some width)) rounding value =
      ((DType.integer signed (some width)).check >>= fun _ =>
        binaryToInteger signed width rounding value) := rfl
  rw [hcast] at h
  simp only [Bind.bind, Except.bind] at h
  cases hc : DType.check (.integer signed (some width)) with
  | error e => simp [hc] at h
  | ok u =>
      simp only [hc, binaryToInteger] at h
      cases hq : Model.toDyadic? (ExecFloat.Binary.toModel value) with
      | none => simp [hq] at h
      | some d =>
          simp only [hq] at h
          cases ho : ofInteger signed width (Model.roundDyadicToInt rounding d) with
          | none => simp [ho] at h
          | some w =>
              simp only [ho, Except.ok.injEq, Prod.mk.injEq] at h
              obtain ⟨rfl, -⟩ := h
              exact ⟨d, rfl, toInteger_ofInteger hwidth ho⟩

/-- The binary part of reference conversion satisfies the same FloatLib specification. -/
theorem cast_binary_value (source target : FloatFormat) (rounding : Model.IEEERoundingMode)
    (value : Binary.Scalar source) (converted : Binary.Scalar target)
    (flags : ExecFloat.ConversionStatus)
    (h : cast (.binary source) (.binary target) rounding value = .ok (converted, flags)) :
    ExecFloat.Binary.toModel converted =
      Model.cast source target (ExecFloat.Binary.toModel value) (mode := rounding) := by
  have hcast : cast (.binary source) (.binary target) rounding value =
      pure (Binary.cast source target rounding value) := rfl
  rw [hcast] at h
  have hpair : Binary.cast source target rounding value = (converted, flags) := Except.ok.inj h
  have hv := Binary.cast_value source target rounding value
  rw [hpair] at hv
  exact hv

/-- The storage of each element type: the configured formats' storage, arrays elsewhere. -/
@[instance_reducible]
def storage : (d : DType) → Storage (Carrier d)
  | .binary format => inferInstanceAs (Storage (Binary.Scalar format))
  | other => instArrayStorage (Carrier other)

/-- The arithmetic of the configured binary formats, and `none` at every other element type. -/
def arithmetic : (d : DType) → Option (Context (Carrier d))
  | .binary format => some (inferInstanceAs (Context (Binary.Scalar format)))
  | _ => none

/-- The optional capabilities of the configured binary formats, all absent at every other type. -/
@[instance_reducible]
def capabilities : (d : DType) → Op.Capabilities (Carrier d)
  | .binary format => inferInstanceAs (Op.Capabilities (Binary.Scalar format))
  | _ => ⟨none, none, none, none, none, none⟩

end Reference

end TorchLean.IR

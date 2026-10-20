/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Init.Data.BitVec.Lemmas
public import NN.IR.BinaryProofs

/-!
# Typed reference execution

Binary arithmetic reuses FloatLib at the declared precision. Boolean and fixed-width integer
values can also pass through the shared storage-only tensor operations. Integers of any positive
width use Lean's `BitVec`; their signedness remains in `DType`, rather than in a second carrier.
Integers also implement PyTorch's integer pointwise operators: two's-complement negation,
truncating and flooring division, and powers with wrapping multiplication. Other integer
arithmetic is not inferred from the representation. Integers and Booleans convert to binary
formats by rounding their exact value once; no other cross-type conversion is implemented.

Element words are unsigned, including signed integer encodings. Boolean words are canonically
zero or one. Loading rejects excessive words, invalid descriptors and incompatible tensor sizes;
`Reference.read_write` proves that decoding a valid word does not change its encoding.

This interpreter is a software reference, not a proof of XLA or LibTorch execution. Other element
representations and operations still need implementations before they can run.
-/

@[expose] public section

namespace NN.IR.Reference

open TorchLean FloatLib.Floats
open FloatLib.Floats.Formats.BinaryInterchange

/-- Configured binary values, Boolean values, and arbitrary-width integer bit vectors. -/
abbrev scalar : DType → Type
  | .binary format => Binary.Scalar format
  | .bool => Bool
  | .integer _ (some width) => BitVec width
  | _ => Empty

/-- Whether this reference implements a valid element representation. -/
def supported (dtype : DType) : Bool :=
  dtype.valid && match dtype with
    | .binary _ | .bool | .integer _ (some _) => true
    | _ => false

/-- Decode one complete element word, rejecting rather than truncating excessive bits. -/
def read : (dtype : DType) → Nat → Except String (scalar dtype)
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

/-- Encode one element without interpreting signed integer words as host floats. -/
def write : (dtype : DType) → scalar dtype → Except String Nat
  | .binary _, value => .ok (ExecFloat.Binary.toNatBits value)
  | .bool, value => .ok (if value then 1 else 0)
  | .integer signed (some width), value => do
      (DType.integer signed (some width)).check
      pure value.toNat
  | dtype, _ => .error s!"reference IR: no word representation for {dtype}"

/-- Round an exact integer once to a binary format, reporting inexactness and overflow. -/
def integerToBinary (target : FloatFormat) (rounding : Model.IEEERoundingMode) (value : Int) :
    Binary.Scalar target × ExecFloat.ConversionStatus :=
  let rounded := Model.roundRatQWithRounding target rounding value
  (ExecFloat.Binary.ofModel rounded,
    { inexact := Model.toRat? rounded != some (value : Rat)
      overflow := Model.isInf rounded })

/-- Reuse directed binary conversion; integers and Booleans round their exact value once.

Other nonbinary conversions require identical types. Binary conversions still use
FloatLib when the formats are equal, since exceptional-value handling and flags belong to the
conversion semantics, not to a tensor relabeling.
-/
def cast (source target : DType) (rounding : Model.IEEERoundingMode)
    (value : scalar source) : Except String (scalar target × ExecFloat.ConversionStatus) :=
  match source, target with
  | .binary src, .binary dst => Binary.cast (.binary src) (.binary dst) rounding value
  | .integer signed (some width), .binary dst => do
      (DType.integer signed (some width)).check
      pure (integerToBinary dst rounding (if signed then value.toInt else value.toNat))
  | .bool, .binary dst => pure (integerToBinary dst rounding (if value then 1 else 0))
  | source, target =>
      if supported source && supported target then
        if h : source = target then by
          cases h
          exact .ok (value, {})
        else .error s!"reference IR: conversion {source} → {target} is not implemented"
      else .error s!"reference IR: no representation for {source} or {target}"

/-- A successfully decoded element retains its entire original word on encoding. -/
theorem read_write (dtype : DType) (word : Nat) (value : scalar dtype)
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

/-- Two's-complement power by repeated squaring, the loop of PyTorch's `powi_impl`. -/
def powWrapping {width : Nat} (base : BitVec width) (exponent : Nat) : BitVec width :=
  if _h : exponent = 0 then 1
  else
    let rest := powWrapping (base * base) (exponent / 2)
    if exponent % 2 = 1 then base * rest else rest
termination_by exponent
decreasing_by omega

/--
PyTorch's integer meanings of the pointwise operators.

Negation and powers wrap in two's complement. Division by zero is an error, as PyTorch raises
`RuntimeError: ZeroDivisionError`. The most negative value divided by `-1` wraps to itself, which
PyTorch returns in flooring division and in truncating division up to 16 bits; its truncating
division of 32- and 64-bit integers traps there with `SIGFPE`. A negative exponent gives `1` for
base `1`, `±1` for base `-1` and `0` otherwise, as PyTorch's `powi` does for tensor exponents.
`clamp_min` returns the bound where the operand is below it, as eager PyTorch does for a scalar
bound. True division, `rsqrt`, `erf` and GELU are absent: PyTorch promotes integer operands to
floating point for them, so a frontend inserts explicit casts.
-/
def integerPointwise (signed : Bool) (width : Nat) : Spec.PointwiseTable (BitVec width)
  | .neg => some (Spec.PointwiseFn.unary fun x => .ok (-x))
  | .div .trunc => some (Spec.PointwiseFn.binary fun x y =>
      if y = 0 then .error "div: integer division by zero (PyTorch raises ZeroDivisionError)"
      else .ok (if signed then x.sdiv y else x.udiv y))
  | .div .floor => some (Spec.PointwiseFn.binary fun x y =>
      if y = 0 then .error "div: integer division by zero (PyTorch raises ZeroDivisionError)"
      else .ok (if signed then BitVec.ofInt width (x.toInt.fdiv y.toInt) else x.udiv y))
  | .pow => some (Spec.PointwiseFn.binary fun x y =>
      if signed && y.toInt < 0 then
        .ok (if x = 1 then 1 else if x = -1 then (if y.toInt % 2 = 0 then 1 else -1) else 0)
      else .ok (powWrapping x y.toNat))
  | .clampMin => some (Spec.PointwiseFn.binary fun x bound =>
      .ok (if (if signed then x.slt bound else x.ult bound) then bound else x))
  | .div .none | .rsqrt | .erf | .gelu _ => none

/-- Pointwise operators of element types without an arithmetic dictionary. -/
def pointwise : (dtype : DType) → Spec.PointwiseTable (scalar dtype)
  | .integer signed (some width) => integerPointwise signed width
  | _ => fun _ => none

/-- The binary part of reference conversion satisfies the same FloatLib specification. -/
theorem cast_binary_value (source target : FloatFormat) (rounding : Model.IEEERoundingMode)
    (value : Binary.Scalar source) (converted : Binary.Scalar target)
    (flags : ExecFloat.ConversionStatus)
    (h : cast (.binary source) (.binary target) rounding value = .ok (converted, flags)) :
    ExecFloat.Binary.toModel converted =
      Model.castWithRounding source target (ExecFloat.Binary.toModel value) rounding :=
  Binary.cast_value source target rounding value converted flags h

end NN.IR.Reference

namespace NN.IR.Interpreter

open TorchLean

/-- Binary numerical evaluation and storage-only Boolean/arbitrary-width integer evaluation. -/
def reference : Interpreter where
  Scalar := Reference.scalar
  storage := fun dtype => match dtype with
    | .binary format => inferInstanceAs (Storage (Binary.Scalar format))
    | other => instArrayStorage (Reference.scalar other)
  supported := Reference.supported
  arithmetic := fun dtype => match dtype with
    | .binary format => some (inferInstanceAs (Context (Binary.Scalar format)))
    | _ => none
  cast := Reference.cast
  pointwise := Reference.pointwise

end NN.IR.Interpreter

namespace NN.IR.Reference

open TorchLean

/-- Load a typed tensor from complete element words, checking its type, size and word ranges. -/
def load (dtype : DType) (shape : Spec.Shape) (words : Array Nat) :
    Except String (Interpreter.Value Interpreter.reference) := do
  letI : Storage (Interpreter.reference.Scalar dtype) := Interpreter.reference.storage dtype
  do
  dtype.check
  unless supported dtype do
    throw s!"reference IR: no representation for {dtype}"
  unless words.size == Spec.Shape.size shape do
    throw "reference IR: word count does not match the tensor shape"
  let tensor ← @Tensor.generateFlatM (Except String) inferInstance
    (Interpreter.reference.Scalar dtype) (Interpreter.reference.storage dtype) shape
    (fun index => read dtype words[index]!)
  pure ⟨dtype, @Spec.SomeTensor.ofTensor (Interpreter.reference.Scalar dtype)
    (Interpreter.reference.storage dtype) shape tensor⟩

/-- Complete element words of a typed tensor, retaining binary special values and integer bits. -/
def words (value : Interpreter.Value Interpreter.reference) : Except String (Array Nat) := do
  letI : Storage (Interpreter.reference.Scalar value.dtype) :=
    Interpreter.reference.storage value.dtype
  do
  value.dtype.check
  unless supported value.dtype do
    throw s!"reference IR: no representation for {value.dtype}"
  let data := @Tensor.Internal.Rep.data (Interpreter.reference.Scalar value.dtype)
    (Interpreter.reference.storage value.dtype) value.tensor.shape value.tensor.tensor
  data.mapM (write value.dtype)

end NN.IR.Reference

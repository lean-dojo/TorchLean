/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Interpreter
public import NN.Spec.Core.FloatInstances
public import FloatLib.Floats.Formats.BinaryInterchange.Conversion.Cast.Runtime

/-!
# Mixed binary precision

The shared interpreter can execute each numerical node at its declared FloatLib
precision. Casts decode the source's exact interchange word and round in the
destination format. Neither constants nor conversions pass through binary64.

This is CPU software execution of the IR, not a reproduction of XLA or LibTorch
kernel schedules. In particular, reduction order belongs to the IR semantics.
Elementary functions use the approximations supplied by FloatLib's `Context`.
Pointwise operators use the dictionary's `Context.pointwise` entries; FloatLib has
no error function, so `erf` and the exact GELU are reported as unsupported.
-/

@[expose] public section

namespace NN.IR

open TorchLean
open FloatLib.Floats
open FloatLib.Floats.Formats.BinaryInterchange

namespace Binary

/-- FloatLib's configured representation for the complete validated format descriptor. -/
abbrev Scalar (format : FloatFormat) :=
  ExecFloat.Binary format.expWidth format.fracWidth format.encoding format.exponentBias
    format.expWidth_ge_two format.fracWidth_pos format.exponentBias_pos
    format.exponentBias_le_maxFinite

/-- Unsupported descriptors have no values in this interpreter. -/
def scalar : DType → Type
  | .binary format => Scalar format
  | _ => Empty

/-- FloatLib's exact-source binary conversion, retaining its IEEE exception flags. -/
def cast (source target : DType) (rounding : Model.IEEERoundingMode)
    (value : scalar source) : Except String (scalar target × ExecFloat.ConversionStatus) :=
  match source, target with
  | .binary src, .binary dst =>
      let outcome := Model.cast src dst (ExecFloat.Binary.toModel value)
        (mode := rounding) (status := true)
      .ok (ExecFloat.Binary.ofModel outcome.value,
        { inexact := outcome.status.inexact
          overflow := outcome.status.overflow
          underflow := outcome.status.underflow
          invalid := outcome.status.invalid })
  | _, _ => .error s!"binary IR: conversion {source} → {target} is not implemented"

/-- Load a shaped binary tensor from complete words, rejecting truncation or padding. -/
def tensor (format : FloatFormat) (shape : Spec.Shape) (words : Array Nat) :
    Except String (Tensor (Scalar format) shape) := do
  unless words.size == Spec.Shape.size shape do
    throw "binary IR: interchange word count does not match the tensor shape"
  unless words.all (· < 2 ^ format.bitWidth) do
    throw "binary IR: interchange word does not fit the declared format"
  pure <| Tensor.generateFlat shape fun index => ExecFloat.Binary.ofNatBits words[index]!

end Binary

namespace Interpreter

/-- Execute numerical nodes using configurable FloatLib binary arithmetic on CPU. -/
def binary : Interpreter where
  Scalar := Binary.scalar
  storage := fun dtype => match dtype with
    | .binary format => inferInstanceAs (Storage (Binary.Scalar format))
    | other => instArrayStorage (Binary.scalar other)
  supported := fun dtype => match dtype with | .binary _ => true | _ => false
  arithmetic := fun dtype => match dtype with
    | .binary format => some (inferInstanceAs (Context (Binary.Scalar format)))
    | _ => none
  cast := Binary.cast

namespace Payload

/-- Constant-only payload for numerical primitives; other parameter stores remain explicit. -/
def binary (constants : Nat → Option (Value Interpreter.binary) := fun _ => none) :
    Payload Interpreter.binary where
  constant? := constants
  homogeneous := fun dtype => letI := Interpreter.binary.storage dtype; {}
  call := fun name _ _ => .error s!"binary IR: no implementation registered for {name}"

end Payload

namespace Value

/-- Load exact interchange words, rejecting truncation and shape mismatches. -/
def binary (format : FloatFormat) (shape : Spec.Shape) (words : Array Nat) :
    Except String (Value Interpreter.binary) :=
  letI := Interpreter.binary.storage (.binary format)
  do
    let tensor ← Binary.tensor format shape words
    pure ⟨.binary format, Spec.SomeTensor.ofTensor tensor⟩

/-- Exact interchange words of a binary tensor, including signed zeros and NaN payloads. -/
def words (value : Value Interpreter.binary) : Except String (Array Nat) :=
  match value with
  | ⟨.binary format, tensor⟩ =>
      letI := Interpreter.binary.storage (.binary format)
      .ok (tensor.tensor.data.map ExecFloat.Binary.toNatBits)
  | ⟨dtype, _⟩ => .error s!"binary IR: {dtype} is not a binary tensor"

end Value

end Interpreter

end NN.IR

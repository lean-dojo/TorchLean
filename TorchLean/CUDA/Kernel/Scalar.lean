/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.CUDA.Kernel.Emit
public import TorchLean.Tensor.Constructors
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Value.CoreProof
public import TorchLean.Tensor.Bytes

/-!
# Exact scalar encodings for custom computations

A `Scalar` instance fixes a type's `Precision`, the arithmetic it uses on the device and a
lossless encoding into unsigned words. Native floats run on hardware arithmetic. Configured binary
values travel as complete interchange words, and the device picks native operations or software
arithmetic from the format, so wide values are never narrowed through binary64. `Scalar.pack` and
`Scalar.unpack` move tensors to and from little-endian bytes.

The round-trip law `TorchLean.Numeric.Word.ofBits_toBits` is about Lean values. Moving the bytes
and computing with them on the device is native code.
-/

@[expose] public section

namespace TorchLean.Kernel

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange

/-- Arithmetic from the type's own `+`, `-`, `*`, `/`, negation, `==`, `<` and `≤` and its
`IndexCast` conversions, applied in the original evaluation order without algebraic
reassociation. -/
def Arithmetic.ofOps {α : Type} [Add α] [Sub α] [Mul α] [Div α] [Neg α]
    [BEq α] [LT α] [LE α] [DecidableLT α] [DecidableLE α] [IndexCast α] : Arithmetic α where
  binary
    | .add, x, y => x + y
    | .sub, x, y => x - y
    | .mul, x, y => x * y
    | .div, x, y => x / y
  neg := fun x => -x
  compare
    | .eq, x, y => x == y
    | .lt, x, y => x < y
    | .le, x, y => x ≤ y
  toIndex := IndexCast.toIndex
  ofIndex := IndexCast.ofIndex

/-- The arithmetic representation of a scalar type: a native CUDA format or a configured binary
format. It does not depend on the execution device. -/
inductive Precision where
  /-- A native CUDA format, `float` or `double`. -/
  | native (format : CUDA.Format)
  /-- A configured binary format, emitted by `emitBinary`. -/
  | binary (format : FloatFormat)

/-- A scalar type of custom computations: its words, the representation the device computes in
and the operations of that representation. -/
class Scalar (α : Type) extends Numeric.Word α where
  /-- The arithmetic representation of the scalar type. -/
  precision : Precision
  /-- Operations implemented by the selected GPU representation. -/
  arithmetic : Arithmetic α

/-- `Float32` computes in native binary32 with its own operations. -/
instance instScalarFloat32 : Scalar Float32 where
  precision := .native .binary32
  arithmetic := Arithmetic.ofOps

/-- `Float` computes in native binary64 with its own operations. -/
instance instScalarFloat : Scalar Float where
  precision := .native .binary64
  arithmetic := Arithmetic.ofOps

/-- The index of a configured binary value: its exact value truncated toward zero
(`ExecFloat.Binary.toInt?`) and saturated to the `UInt64` range, `0` at a NaN and below zero and
`2^64 - 1` at a positive infinity, as `Float32.toUInt64` converts. -/
def configuredToIndex {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]
    (x : ExecFloat (Configured.Family format code plan)) : UInt64 :=
  match ExecFloat.Binary.toInt? x with
  | some n =>
      if n ≤ 0 then 0 else if n < 2 ^ 64 then n.toNat.toUInt64 else UInt64.ofNat (2 ^ 64 - 1)
  | none =>
      let m := ExecFloat.Binary.toModel x
      if m.isNaN || m.signBit then 0 else UInt64.ofNat (2 ^ 64 - 1)

/-- The configured binary value of an index: the index rounded once to nearest even in the
format (the model's `NatCast`), with the format's overflow convention. -/
def configuredOfIndex {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]
    (i : UInt64) : ExecFloat (Configured.Family format code plan) :=
  ExecFloat.Binary.ofModel ((i.toNat : Nat) : Model format)

/-- A configured binary format converts to and from indices by `configuredToIndex` and
`configuredOfIndex`. -/
instance instIndexCastConfigured {format : FloatFormat} {plan : Configured.StoragePlan format}
    {code : Type} [ExecFloat.ModelCodec plan (Model format) code] :
    IndexCast (ExecFloat (Configured.Family format code plan)) where
  toIndex := configuredToIndex
  ofIndex := configuredOfIndex

/-- A configured binary format computes in that format with its own operations. -/
instance instScalarConfigured {format : FloatFormat} {plan : Configured.StoragePlan format}
    {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code] :
    Scalar (ExecFloat (Configured.Family format code plan)) where
  precision := .binary format
  arithmetic := Arithmetic.ofOps

/-- Bytes per complete value: 4, 8, or the configured bit width rounded up to whole bytes. -/
def Precision.bytes : Precision → Nat
  | .native .binary32 => 4
  | .native .binary64 => 8
  | .binary format => (format.bitWidth + 7) / 8

/-- Whether custom GPU arithmetic supports the format: every native format, and configured binary
formats with at most 30 exponent bits and 4096 fraction bits. Only the format is inspected, so the
check says nothing about an available GPU or free memory. -/
def Precision.supportsGPU : Precision → Bool
  | .native _ => true
  | .binary format => format.expWidth ≤ 30 && format.fracWidth ≤ 4096

/-- Serialize a tensor's complete scalar words in row-major order, little-endian within each word.
An encoding wider than `precision.bytes` bytes is an error. -/
def Scalar.pack {α : Type} [Storage α] {shape : Shape} (scalar : Scalar α)
    (tensor : Tensor α shape) : Except String ByteArray := do
  let mut bytes := ByteArray.empty
  for h : i in [:shape.size] do
    let bits := (scalar.toBits (Tensor.Rep.getFlat tensor ⟨i, h.upper⟩)).toNat
    unless bits < 2 ^ (8 * scalar.precision.bytes) do
      throw "custom computation: input encoding exceeds the scalar byte width"
    bytes := Bytes.pushWordLE bytes scalar.precision.bytes bits
  return bytes

/-- Decode a tensor of `shape` from little-endian words, after checking that the byte count is
`shape.size * precision.bytes`. -/
def Scalar.unpack {α : Type} [Storage α] (scalar : Scalar α) (shape : Shape)
    (bytes : ByteArray) : Except String (Tensor α shape) := do
  let width := scalar.precision.bytes
  unless bytes.size == shape.size * width do
    throw s!"custom computation: expected {shape.size * width} bytes, received {bytes.size}"
  return Tensor.generateFlat shape fun i =>
    scalar.ofNatBits <| Bytes.wordFromLE bytes (i * width) width

end TorchLean.Kernel

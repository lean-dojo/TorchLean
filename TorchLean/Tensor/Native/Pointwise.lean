/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Rep.Pointwise -- shake: keep

/-!
# Certified native pointwise kernels

This module names the four promoting arithmetic operations (`BinaryOp`) and gives each its scalar
operation (`BinaryOp.Scalar`). All four run through one C kernel per operand format: packed `Float`
with packed `Float`, packed bytes with packed `Float`, and packed `Float` with packed bytes.

The proof-visible bodies are `Array.zipWith` of the `Float` operation, so they truncate to the
shorter input, and tensor callers have equal lengths by their shapes. The `_data` and `_apply`
theorems prove the array and coordinate semantics of these bodies. Compiled code calls
`torchlean_float_array_binary`, `torchlean_byte_float_array_binary` and
`torchlean_float_byte_array_binary`, and their agreement with the bodies is an FFI trust boundary.
-/

@[expose] public section

namespace TorchLean.Tensor

/--
The binary arithmetic operations with promoting pointwise tensor kernels. The C kernels read
them by constructor index, from `add` as 0 to `div` as 3.
-/
inductive BinaryOp where
  /-- Addition. -/
  | add
  /-- Subtraction. -/
  | sub
  /-- Multiplication. -/
  | mul
  /-- Division. -/
  | div

/-- The scalar operation that the pointwise tensor operation `op` applies to elements of `γ`. -/
class BinaryOp.Scalar (op : BinaryOp) (γ : Type) where
  /-- Apply `op` to two scalars. -/
  eval : γ → γ → γ

/-- The scalar operation of `BinaryOp.add` is `+`. -/
instance {γ : Type} [Add γ] : BinaryOp.Scalar .add γ := ⟨(· + ·)⟩

/-- `simp` rewrites the scalar operation of `BinaryOp.add` to `+`. -/
@[simp] theorem BinaryOp.Scalar.eval_add {γ : Type} [Add γ] :
    (BinaryOp.Scalar.eval .add : γ → γ → γ) = (· + ·) := rfl

/-- The scalar operation of `BinaryOp.sub` is `-`. -/
instance {γ : Type} [Sub γ] : BinaryOp.Scalar .sub γ := ⟨(· - ·)⟩

/-- `simp` rewrites the scalar operation of `BinaryOp.sub` to `-`. -/
@[simp] theorem BinaryOp.Scalar.eval_sub {γ : Type} [Sub γ] :
    (BinaryOp.Scalar.eval .sub : γ → γ → γ) = (· - ·) := rfl

/-- The scalar operation of `BinaryOp.mul` is `*`. -/
instance {γ : Type} [Mul γ] : BinaryOp.Scalar .mul γ := ⟨(· * ·)⟩

/-- `simp` rewrites the scalar operation of `BinaryOp.mul` to `*`. -/
@[simp] theorem BinaryOp.Scalar.eval_mul {γ : Type} [Mul γ] :
    (BinaryOp.Scalar.eval .mul : γ → γ → γ) = (· * ·) := rfl

/-- The scalar operation of `BinaryOp.div` is `/`. -/
instance {γ : Type} [Div γ] : BinaryOp.Scalar .div γ := ⟨(· / ·)⟩

/-- `simp` rewrites the scalar operation of `BinaryOp.div` to `/`. -/
@[simp] theorem BinaryOp.Scalar.eval_div {γ : Type} [Div γ] :
    (BinaryOp.Scalar.eval .div : γ → γ → γ) = (· / ·) := rfl

end TorchLean.Tensor

namespace TorchLean.Native

open Tensor (Rep BinaryOp)
open Shape (Coord)

/-- The `Float` operation of `op`, read from its `BinaryOp.Scalar` instance. -/
@[reducible, simp] def floatBinary : BinaryOp → Float → Float → Float
  | .add => BinaryOp.Scalar.eval .add
  | .sub => BinaryOp.Scalar.eval .sub
  | .mul => BinaryOp.Scalar.eval .mul
  | .div => BinaryOp.Scalar.eval .div

/--
Apply `op` elementwise to two packed `Float` buffers, truncating to the shorter one.

Compiled code calls `torchlean_float_array_binary`, which selects `op` once and runs one IEEE
binary64 loop.
-/
@[extern "torchlean_float_array_binary"]
def floatBufferBinary (op : BinaryOp) (left right : @& FloatArray) : FloatArray :=
  ⟨Array.zipWith (floatBinary op) left.data right.data⟩

/-- Apply `op` to two packed `Float` tensors in one preallocated native loop. -/
def binary (op : BinaryOp) {shape : Shape}
    (left right : @Rep Float shape instFloatStorage) :
    @Rep Float shape instFloatStorage where
  buffer := floatBufferBinary op left.buffer right.buffer
  size_eq := by
    rw [← instFloatStorage.toArray_size]
    change (Array.zipWith (floatBinary op) left.data right.data).size = _
    simp

/-- The array of `binary op left right` is `Array.zipWith (floatBinary op)` of the inputs. -/
@[simp] theorem binary_data (op : BinaryOp) {shape : Shape}
    (left right : @Rep Float shape instFloatStorage) :
    (binary op left right).data =
      Array.zipWith (floatBinary op) left.data right.data := by
  change (floatBufferBinary op left.buffer right.buffer).data =
    Array.zipWith (floatBinary op) left.buffer.data right.buffer.data
  rfl

/-- Each coordinate of `binary op left right` is `floatBinary op` of the input coordinates. -/
@[simp, grind =] theorem binary_apply (op : BinaryOp) {shape : Shape}
    (left right : @Rep Float shape instFloatStorage) (coordinate : Coord shape) :
    binary op left right coordinate =
      floatBinary op (left coordinate) (right coordinate) := by
  let index := Coord.linearize coordinate
  change (binary op left right).getFlat index =
    floatBinary op (left.getFlat index) (right.getFlat index)
  rw [← Rep.data_getFlat (binary op left right) index,
    ← Rep.data_getFlat left index, ← Rep.data_getFlat right index]
  simp

/--
Apply `op` elementwise to packed bytes, converted by the exact `UInt8.toFloat`, and packed `Float`s,
so each entry has the bits of the promoted `Float` operation.

The `_data` and `_apply` theorems of `promoteLeft` prove the semantics of this body, and
compiled code calls the trusted C loop `torchlean_byte_float_array_binary`.
-/
@[extern "torchlean_byte_float_array_binary"]
def byteFloatBufferBinary (op : BinaryOp) (left : @& ByteArray) (right : @& FloatArray) :
    FloatArray :=
  ⟨Array.zipWith (fun x y => floatBinary op x.toFloat y) left.data right.data⟩

/--
Apply `op` elementwise to packed `Float`s and packed bytes converted by `UInt8.toFloat`.

The theorems of `promoteRight` prove the semantics of this body, and compiled code calls the
trusted C loop `torchlean_float_byte_array_binary`.
-/
@[extern "torchlean_float_byte_array_binary"]
def floatByteBufferBinary (op : BinaryOp) (left : @& FloatArray) (right : @& ByteArray) :
    FloatArray :=
  ⟨Array.zipWith (fun x y => floatBinary op x y.toFloat) left.data right.data⟩

/-- Apply `op` to a byte tensor, converted to `Float`, and a packed `Float` tensor in one loop. -/
def promoteLeft (op : BinaryOp) {shape : Shape}
    (left : @Rep UInt8 shape instUInt8Storage) (right : @Rep Float shape instFloatStorage) :
    @Rep Float shape instFloatStorage where
  buffer := byteFloatBufferBinary op left.buffer right.buffer
  size_eq := by
    rw [← instFloatStorage.toArray_size]
    change (Array.zipWith (fun x y => floatBinary op x.toFloat y) left.data right.data).size = _
    simp

/-- Apply `op` to a packed `Float` tensor and a byte tensor, converted to `Float`, in one loop. -/
def promoteRight (op : BinaryOp) {shape : Shape}
    (left : @Rep Float shape instFloatStorage) (right : @Rep UInt8 shape instUInt8Storage) :
    @Rep Float shape instFloatStorage where
  buffer := floatByteBufferBinary op left.buffer right.buffer
  size_eq := by
    rw [← instFloatStorage.toArray_size]
    change (Array.zipWith (fun x y => floatBinary op x y.toFloat) left.data right.data).size = _
    simp

/-- The array of `promoteLeft op left right` is the `zipWith` of the converted bytes. -/
@[simp] theorem promoteLeft_data (op : BinaryOp) {shape : Shape}
    (left : @Rep UInt8 shape instUInt8Storage) (right : @Rep Float shape instFloatStorage) :
    (promoteLeft op left right).data =
      Array.zipWith (fun x y => floatBinary op x.toFloat y) left.data right.data := by
  change (byteFloatBufferBinary op left.buffer right.buffer).data =
    Array.zipWith (fun x y => floatBinary op x.toFloat y) left.buffer.data right.buffer.data
  rfl

/-- The array of `promoteRight op left right` is the `zipWith` of the converted bytes. -/
@[simp] theorem promoteRight_data (op : BinaryOp) {shape : Shape}
    (left : @Rep Float shape instFloatStorage) (right : @Rep UInt8 shape instUInt8Storage) :
    (promoteRight op left right).data =
      Array.zipWith (fun x y => floatBinary op x y.toFloat) left.data right.data := by
  change (floatByteBufferBinary op left.buffer right.buffer).data =
    Array.zipWith (fun x y => floatBinary op x y.toFloat) left.buffer.data right.buffer.data
  rfl

/-- Each coordinate of `promoteLeft op left right` is `floatBinary op` after conversion. -/
@[simp, grind =] theorem promoteLeft_apply (op : BinaryOp) {shape : Shape}
    (left : @Rep UInt8 shape instUInt8Storage) (right : @Rep Float shape instFloatStorage)
    (coordinate : Coord shape) :
    promoteLeft op left right coordinate =
      floatBinary op (left coordinate).toFloat (right coordinate) := by
  let index := Coord.linearize coordinate
  change (promoteLeft op left right).getFlat index =
    floatBinary op (left.getFlat index).toFloat (right.getFlat index)
  rw [← Rep.data_getFlat (promoteLeft op left right) index,
    ← Rep.data_getFlat left index, ← Rep.data_getFlat right index]
  simp

/-- Each coordinate of `promoteRight op left right` is `floatBinary op` after conversion. -/
@[simp, grind =] theorem promoteRight_apply (op : BinaryOp) {shape : Shape}
    (left : @Rep Float shape instFloatStorage) (right : @Rep UInt8 shape instUInt8Storage)
    (coordinate : Coord shape) :
    promoteRight op left right coordinate =
      floatBinary op (left coordinate) (right coordinate).toFloat := by
  let index := Coord.linearize coordinate
  change (promoteRight op left right).getFlat index =
    floatBinary op (left.getFlat index) (right.getFlat index).toFloat
  rw [← Rep.data_getFlat (promoteRight op left right) index,
    ← Rep.data_getFlat left index, ← Rep.data_getFlat right index]
  simp

end TorchLean.Native

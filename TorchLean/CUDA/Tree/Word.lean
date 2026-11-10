/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.CUDA.Kernel.Scalar

/-!
# Scalars held as their encodings

Lean's `Float` and `Float32` hold a single NaN. In the logical model every NaN is the same value,
and `Float.toBits` gives it one encoding, so a `Float` cannot carry a NaN payload or a signaling
NaN, and an arena of them cannot either. `FloatBits` and `Float32Bits` hold binary64 and binary32
values as their encodings, a `UInt64` and a `UInt32`, so every encoding is a different value
(`FloatBits.toBits_ofBits`) and every word read back is the word written
(`FloatBits.toNat_toBits_ofNatBits`). With them the round trips of
`TorchLean.Kernel.Arena.Columns.ofBlocks_words_scalars` and
`TorchLean.Kernel.Arena.decode_ofBlocks_encode`, which hold for every scalar type, return each
word as it was, NaN payloads included. So do the device blocks of `Arena.upload` and
`Arena.Device.download`, and kernels that only move scalars, such as the nodes of an unfold that
reads them from an input.

We decode only at the edge, with `FloatBits.toFloat` and, for a whole arena,
`Arena.map FloatBits.toFloat`. Arithmetic on encodings decodes its operands, computes with the
operations of `Float` and encodes the result (`Arithmetic.transfer`), so decoding commutes with
every operation (`Arithmetic.decode_transfer_binary`). A NaN that arithmetic makes is Lean's one
NaN on the host and the GPU's NaN on the device, which compute in `double` and `float`, so the
results of arithmetic agree with the host up to the encoding of NaNs, as for `Float`.
-/

@[expose] public section

namespace TorchLean.Kernel

variable {α β : Type}

/-- The arithmetic `A` of `β` carried to `α` along `decode` and `encode`: an operation decodes
its operands, computes with `A` and encodes the result, and a comparison compares the decoded
operands. -/
def Arithmetic.transfer (A : Arithmetic β) (decode : α → β) (encode : β → α) : Arithmetic α where
  binary op x y := encode (A.binary op (decode x) (decode y))
  neg x := encode (A.neg (decode x))
  compare op x y := A.compare op (decode x) (decode y)

/-- When `decode` undoes `encode`, decoding the result of an operation of
`Arithmetic.transfer A decode encode` is the operation of `A` on the decoded operands. -/
theorem Arithmetic.decode_transfer_binary {A : Arithmetic β} {decode : α → β} {encode : β → α}
    (h : ∀ y, decode (encode y) = y) (op : ScalarOp) (x y : α) :
    decode ((A.transfer decode encode).binary op x y) = A.binary op (decode x) (decode y) :=
  h _

/-- When `decode` undoes `encode`, decoding a negation of `Arithmetic.transfer A decode encode` is
the negation of `A` of the decoded operand. -/
theorem Arithmetic.decode_transfer_neg {A : Arithmetic β} {decode : α → β} {encode : β → α}
    (h : ∀ y, decode (encode y) = y) (x : α) :
    decode ((A.transfer decode encode).neg x) = A.neg (decode x) :=
  h _

/-- A binary64 value held as its encoding, so that a NaN keeps its sign, its quiet bit and its
payload. -/
structure FloatBits where
  /-- The binary64 encoding. -/
  bits : UInt64
  deriving DecidableEq, Inhabited, Repr

namespace FloatBits

/-- The `Float` of an encoding. Every NaN encoding gives Lean's one NaN. -/
def toFloat (x : FloatBits) : Float :=
  Float.ofBits x.bits

/-- The encoding of a `Float`, as `Float.toBits` gives it. -/
def ofFloat (x : Float) : FloatBits :=
  ⟨x.toBits⟩

/-- Decoding the encoding of a `Float` gives the `Float`. -/
@[simp] theorem toFloat_ofFloat (x : Float) : (ofFloat x).toFloat = x :=
  Numeric.Word.ofBits_toBits x

/-- The words of `FloatBits` are its encodings, with `ofNatBits`, `succ` and `pred` in `UInt64`
arithmetic. -/
instance : Numeric.Word FloatBits where
  width := 64
  toBits x := x.bits.toBitVec
  ofBits b := ⟨⟨b⟩⟩
  ofBits_toBits _ := rfl
  ofNatBits n := ⟨n.toUInt64⟩
  ofNatBits_eq _ := rfl
  succ x := ⟨x.bits + 1⟩
  succ_eq _ := rfl
  pred x := ⟨x.bits - 1⟩
  pred_eq _ := rfl

/-- Every 64-bit word is the encoding of the value it gives, so no two encodings are the same
value, NaNs included. -/
theorem toBits_ofBits (b : BitVec 64) :
    Numeric.Word.toBits (Numeric.Word.ofBits (α := FloatBits) b) = b :=
  rfl

/-- A word read with `Numeric.Word.ofNatBits` is held as it was read: the encoding of
`ofNatBits n` is `n` modulo `2 ^ 64`. -/
theorem toNat_toBits_ofNatBits (n : Nat) :
    (Numeric.Word.toBits (Numeric.Word.ofNatBits n : FloatBits)).toNat = n % 2 ^ 64 :=
  BitVec.toNat_ofNat n 64

/-- Binary64 encodings compute as `double` on the device and in the arithmetic of `Float` on the
host, decoded with `toFloat` and encoded with `ofFloat`. -/
instance : Scalar FloatBits where
  precision := .native .binary64
  arithmetic := Arithmetic.transfer Arithmetic.ofOps toFloat ofFloat

end FloatBits

/-- A binary32 value held as its encoding, so that a NaN keeps its sign, its quiet bit and its
payload. -/
structure Float32Bits where
  /-- The binary32 encoding. -/
  bits : UInt32
  deriving DecidableEq, Inhabited, Repr

namespace Float32Bits

/-- The `Float32` of an encoding. Every NaN encoding gives Lean's one NaN. -/
def toFloat32 (x : Float32Bits) : Float32 :=
  Float32.ofBits x.bits

/-- The encoding of a `Float32`, as `Float32.toBits` gives it. -/
def ofFloat32 (x : Float32) : Float32Bits :=
  ⟨x.toBits⟩

/-- Decoding the encoding of a `Float32` gives the `Float32`. -/
@[simp] theorem toFloat32_ofFloat32 (x : Float32) : (ofFloat32 x).toFloat32 = x :=
  Numeric.Word.ofBits_toBits x

/-- The words of `Float32Bits` are its encodings, with `ofNatBits`, `succ` and `pred` in `UInt32`
arithmetic. -/
instance : Numeric.Word Float32Bits where
  width := 32
  toBits x := x.bits.toBitVec
  ofBits b := ⟨⟨b⟩⟩
  ofBits_toBits _ := rfl
  ofNatBits n := ⟨n.toUInt32⟩
  ofNatBits_eq _ := rfl
  succ x := ⟨x.bits + 1⟩
  succ_eq _ := rfl
  pred x := ⟨x.bits - 1⟩
  pred_eq _ := rfl

/-- Every 32-bit word is the encoding of the value it gives, so no two encodings are the same
value, NaNs included. -/
theorem toBits_ofBits (b : BitVec 32) :
    Numeric.Word.toBits (Numeric.Word.ofBits (α := Float32Bits) b) = b :=
  rfl

/-- A word read with `Numeric.Word.ofNatBits` is held as it was read: the encoding of
`ofNatBits n` is `n` modulo `2 ^ 32`. -/
theorem toNat_toBits_ofNatBits (n : Nat) :
    (Numeric.Word.toBits (Numeric.Word.ofNatBits n : Float32Bits)).toNat = n % 2 ^ 32 :=
  BitVec.toNat_ofNat n 32

/-- Binary32 encodings compute as `float` on the device and in the arithmetic of `Float32` on the
host, decoded with `toFloat32` and encoded with `ofFloat32`. -/
instance : Scalar Float32Bits where
  precision := .native .binary32
  arithmetic := Arithmetic.transfer Arithmetic.ofOps toFloat32 ofFloat32

end Float32Bits

end TorchLean.Kernel

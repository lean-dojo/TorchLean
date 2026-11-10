/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Floats.Formats.IEEE754.Native
public import TorchLean.CUDA.LibTorch.Trusted
public import TorchLean.Tensor.Datatype
public import TorchLean.CUDA.Kernel.Scalar

/-!
# Runtime Tensor Transfer

Backend-neutral conversion between executable tensors and the host `Float` representation used at
native runtime boundaries. `Float` and `Float32` map to native float64 and float32 tensors. A scalar
with a configured encoding (`Kernel.Scalar`) moves as its packed words, and any other scalar
stays on CPU.
-/

@[expose] public section

namespace TorchLean

open FloatLib.Floats
open FloatLib.Floats.Formats.BinaryInterchange (FloatFormat)

/--
Conversion between an executable scalar type and the host `Float` representation used at native
runtime boundaries.

Native transfer and host inspection are separate operations. A scalar can let `readFloatTensor`
show every element as a `Float` while `toFloatTensor` refuses a native transfer, as the configured
binary32 instance does.
-/
class TensorTransfer (α : Type) [Storage α] where
  /-- Native dtype, when this carrier can use the native tensor runtime. The default `none` keeps
  custom scalar types on CPU, so they are never narrowed through `Float`. Device availability is
  checked separately, and the dtype does not certify native arithmetic. -/
  dtype? : Option LibTorch.Dtype := none
  /-- Lossless configured encoding for custom arithmetic on the tape. A byte encoding alone does not
  make a scalar usable by native tensor kernels or optimizers. -/
  encoding? : Option (Kernel.Scalar α) := none
  /-- Encode a tensor in the host representation used by native runtimes. -/
  toFloatTensor : {s : Shape} → Tensor α s → IO (Tensor Float s)
  /-- Decode a tensor received from a native runtime without changing its shape. -/
  ofFloatTensor : {s : Shape} → Tensor Float s → IO (Tensor α s)
  /-- Convert one scalar to the host representation used by native runtimes. -/
  toFloat : α → IO Float
  /-- Read a tensor as `Float` values for host-side inspection, separate from native transfer. -/
  readFloatTensor : {s : Shape} → Tensor α s → IO (Tensor Float s) := toFloatTensor

/-- Whether a scalar has a native tape representation. Device availability is checked separately. -/
def TensorTransfer.supportsGPU {α : Type} [Storage α] [TensorTransfer α] : Bool :=
  (TensorTransfer.dtype? (α := α)).isSome

/-- Whether custom arithmetic can retain this scalar's configured words on the GPU tape. -/
def TensorTransfer.supportsEncodedGPU {α : Type} [Storage α] [TensorTransfer α] : Bool :=
  match TensorTransfer.encoding? (α := α) with
  | some scalar => match scalar.precision with
    | .binary _ => scalar.precision.supportsGPU
    | .native _ => false
  | none => false

/-- Native dtype required at a transfer boundary. A carrier without one must stay on CPU. -/
def TensorTransfer.dtype {α : Type} [Storage α] [TensorTransfer α] :
    IO LibTorch.Dtype :=
  match TensorTransfer.dtype? (α := α) with
  | some dtype => pure dtype
  | none => throw <| IO.userError
      "torch: this scalar has no native tensor representation; run it on CPU"

/-- `Float` retains binary64 on the native tape, without an intermediate binary32 conversion. -/
instance (priority := 1000) : TensorTransfer Float where
  dtype? := some .float64
  toFloatTensor := fun tensor => pure tensor
  ofFloatTensor := fun tensor => pure tensor
  toFloat := pure

/-- Native binary32 transfers preserve the runtime's float32 wire representation. -/
instance (priority := 1000) : TensorTransfer Float32 where
  dtype? := some .float32
  toFloatTensor := fun tensor => pure (Tensor.map Float32.toFloat tensor)
  ofFloatTensor := fun tensor => pure (Tensor.map Float.toFloat32 tensor)
  toFloat := fun x => pure x.toFloat

/--
Host-side conversion for a scalar with a configured encoding (`Kernel.Scalar`), which `encoding?`
records. Transfer between configured words and native tensors throws in both directions.

`toFloat` and `readFloatTensor` read back through `ExecFloat.Binary.toFloat`, exact except for NaN
payloads, for a binary format whose words have the format's width, with `fracWidth` and
`maxNormalExponent` at most binary64's and `minSubnormalExponent` at least binary64's. Other
readback throws.
-/
instance (priority := 100) (α : Type) [Storage α] [Kernel.Scalar α] :
    TensorTransfer α :=
  -- One check of the format serves every element of a tensor.
  let read? : Option (α → Float) :=
    match Kernel.Scalar.precision (α := α) with
    | .binary format =>
        if h : Numeric.Word.width α = format.bitWidth then
          if format.fracWidth ≤ FloatFormat.binary64.fracWidth ∧
              FloatFormat.binary64.minSubnormalExponent ≤ format.minSubnormalExponent ∧
              format.maxNormalExponent ≤ FloatFormat.binary64.maxNormalExponent then
            some fun x => ExecFloat.Binary.toFloat (format := format) (plan := .wide)
              (ExecFloat.Binary.ofModel ⟨(Numeric.Word.toBits x).cast h⟩)
          else none
        else none
    | .native _ => none
  { encoding? := some inferInstance
    toFloatTensor := fun {_s} _ =>
      throw <| IO.userError "torch: configured words cannot be converted to a native tensor"
    ofFloatTensor := fun {_s} _ =>
      throw <| IO.userError "torch: native tensors cannot supply configured words"
    toFloat := fun x => match read? with
      | some read => pure (read x)
      | none => throw <| IO.userError "torch: inspect this scalar in its own format"
    readFloatTensor := fun tensor => match read? with
      | some read => pure (Tensor.map read tensor)
      | none => throw <| IO.userError "torch: inspect this scalar in its own format" }

/--
CPU-preserving fallback for scalar types without a native tensor representation.

Add a higher-priority `TensorTransfer α` instance when a scalar type has an honest host `Float`
encoding. CPU execution does not invoke these failing transfer operations.
-/
instance (priority := 10) (α : Type) [Storage α] : TensorTransfer α where
  toFloatTensor := fun {_s} _ =>
    throw <| IO.userError <|
      "torch: this scalar type cannot be converted to a native tensor; " ++
        "use Float for native execution or run this scalar on CPU"
  ofFloatTensor := fun {_s} _ =>
    throw <| IO.userError <|
      "torch: native tensors cannot be converted to this scalar type; " ++
        "use Float for native execution or run this scalar on CPU"
  toFloat := fun _ =>
    throw <| IO.userError <|
      "torch: this scalar type cannot be converted to a native scalar; " ++
        "use Float for native execution or run this scalar on CPU"

end TorchLean

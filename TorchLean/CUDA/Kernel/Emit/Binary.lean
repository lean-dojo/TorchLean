/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.CUDA.Kernel.Scalar

/-!
# Configured binary CUDA emission

`emitBinary` renders an expression for a configured binary format. Lowering is the same as for
native arithmetic, and every configured format shares one device template, `TLBinary` in
`csrc/libtorch/binary.h`. The template uses native instructions where the complete format and
the GPU architecture allow it and integer-limb arithmetic otherwise. The format descriptor selects
the exceptional-value and overflow conventions. The template is hand-written foreign code bundled
with the native bridge, and it is trusted: no FloatLib proof covers it.
-/

@[expose] public section

namespace TorchLean.CUDA

open FloatLib.Floats.Formats.BinaryInterchange
open Kernel

/-- The dialect of a configured binary format: literals as exact little-endian bytes and the four
operations as the template's round-to-nearest-even arithmetic. -/
@[no_expose] def Dialect.configured {α : Type} (format : FloatFormat) (encode : α → Nat) :
    Dialect α :=
  let bytes := (format.bitWidth + 7) / 8
  { scalar := "scalar"
    literal := fun x => do
      let bits := encode x
      unless bits < 2 ^ format.bitWidth do
        throw "custom computation: literal exceeds the configured binary width"
      let digits := (Bytes.pushWordLE .empty bytes bits).toList.map fun byte =>
        toString byte.toNat
      return "scalar{{" ++ String.intercalate ", " digits ++ "}}"
    binary := fun op => match op with
      | .add => "tl_add" | .sub => "tl_sub" | .mul => "tl_mul" | .div => "tl_div" }

/-- The include, the template instance and the four operations that precede a kernel of a
configured binary format. Formats over 30 exponent bits or 4096 fraction bits are rejected. -/
@[no_expose] def binaryPrelude (format : FloatFormat) : Except String String := do
  unless (Precision.binary format).supportsGPU do
    throw "custom computation: GPU software arithmetic supports at most 30 exponent bits and \
      4096 fraction bits; use CPU for larger formats"
  let encoding := match format.encoding with
    | .ieee => 0 | .finiteMaxNaN => 1 | .finiteUnsignedZero => 2 | .finite => 3
  return "#include \"torchlean_binary.cuh\"\n" ++
    s!"\nusing scalar = TLBinary<{format.expWidth}, {format.fracWidth}, \
      {format.exponentBias}LL, {encoding}>;\n" ++
    "__device__ scalar tl_add(scalar x, scalar y) { return x + y; }\n" ++
    "__device__ scalar tl_sub(scalar x, scalar y) { return x - y; }\n" ++
    "__device__ scalar tl_mul(scalar x, scalar y) { return x * y; }\n" ++
    "__device__ scalar tl_div(scalar x, scalar y) { return x / y; }\n"

/-- Render CUDA source for a configured binary format: the prelude of `binaryPrelude`, which
rejects the formats the template does not support, then the body in the dialect of
`Dialect.configured` through the shared renderer. -/
@[no_expose] def emitBinary {α : Type} (format : FloatFormat) (encode : α → Nat) (inputs : Nat)
    (expr : Expr α [.index] .scalar) : Except String String := do
  let prelude ← binaryPrelude format
  let ((statement, result), _) := (lower Names.output (Target.lower expr)).run 0
  let body ← Statement.renderWith (Dialect.configured format encode) inputs statement
  return prelude ++ entrypoint "scalar" inputs body result

end TorchLean.CUDA

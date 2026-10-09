/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Kernel.Scalar

/-!
# Configured binary CUDA emission

Configured binary formats share one device implementation, with native operations selected by
the complete format and GPU architecture and integer-limb arithmetic otherwise. Expression lowering
is the same as for native arithmetic; device arithmetic itself is an explicit foreign-code trust
boundary. The descriptor selects exceptional-value and overflow conventions. Device arithmetic is
bundled with the native bridge in `csrc/libtorch/binary.h`, not generated from FloatLib proofs.
-/

@[expose] public section

namespace NN.Kernel.Cuda

open FloatLib.Floats.Formats.BinaryInterchange

/-- Emit exact byte literals and configured nearest-even arithmetic using the shared renderer. -/
@[no_expose] def binarySource {α : Type} (format : FloatFormat) (encode : α → Nat) (inputs : Nat)
    (expr : Expr α [.index] .scalar) : Except String String := do
  unless (Precision.binary format).supportsGpu do
    throw "custom computation: GPU software arithmetic supports at most 30 exponent bits and \
      4096 fraction bits; use CPU for larger formats"
  let bytes := (format.bitWidth + 7) / 8
  let dialect : Internal.Dialect α :=
    { scalar := "scalar"
      literal := fun x => do
        let bits := encode x
        unless bits < 2 ^ format.bitWidth do
          throw "custom computation: literal exceeds the configured binary width"
        let digits := (List.range bytes).map fun i => toString ((bits >>> (8 * i)) % 256)
        return "scalar{{" ++ String.intercalate ", " digits ++ "}}"
      binary := fun op => match op with
        | .add => "tl_add" | .sub => "tl_sub" | .mul => "tl_mul" | .div => "tl_div" }
  let ((statement, result), _) := (Internal.lower Internal.Names.output (Target.lower expr)).run 0
  let body ← Internal.renderWith dialect inputs statement
  let encoding := match format.encoding with
    | .ieee => 0 | .finiteMaxNaN => 1 | .finiteUnsignedZero => 2 | .finite => 3
  return "#include \"torchlean_binary.cuh\"\n" ++
    s!"\nusing scalar = TLBinary<{format.expWidth}, {format.fracWidth}, \
      {format.exponentBias}LL, {encoding}>;\n" ++
    "__device__ scalar tl_add(scalar x, scalar y) { return x + y; }\n" ++
    "__device__ scalar tl_sub(scalar x, scalar y) { return x - y; }\n" ++
    "__device__ scalar tl_mul(scalar x, scalar y) { return x * y; }\n" ++
    "__device__ scalar tl_div(scalar x, scalar y) { return x / y; }\n" ++
    Internal.entrypoint "scalar" inputs body result

end NN.Kernel.Cuda

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.CUDA.Kernel.Emit
public import TorchLean.CUDA.Kernel.Emit.Binary
public import TorchLean.CUDA.LibTorch.Buffer

/-!
# Native execution of generated custom operations

`runBytes` and `runBuffers` hand CUDA source to two `@[extern]` entry points,
`torchlean_kernel_run_bytes` and `torchlean_kernel_run_buffer`. The source is always generated
from a typed expression, never caller-supplied text. The native bridge compiles it with NVRTC
under a fixed floating-point policy, caches a bounded number of modules per process (keyed by
source and context) and launches on LibTorch's current stream. Input buffers and the module stay
alive until the bounds record has been checked.

Both entry points are trusted foreign code, and no Lean theorem covers NVRTC or device execution.
LibTorch keeps ownership of tensor storage and of its own numerical operations. CPU-only builds
link the same symbols, which return an ordinary IO error instead of running CUDA.
-/

@[expose] public section

namespace TorchLean.Kernel

open LibTorch

@[never_extract, extern "torchlean_kernel_run_buffer"]
private opaque executeBuffer (source : @& String) (dtype : Dtype)
    (inputs : @& Array Buffer) (count : UInt64) : IO Buffer

@[never_extract, extern "torchlean_kernel_run_bytes"]
private opaque executeBytes (source : @& String) (format : UInt32) (width : UInt64)
    (inputs : @& Array ByteArray) (count : UInt64) : IO ByteArray

/-- Run `expr` at `count` output indices over inputs of complete scalar encodings and return the
encoded outputs. Wide binary values never pass through native floats. -/
@[no_expose] def runBytes {α : Type} (scalar : Scalar α)
    (expr : Expr α [.index] .scalar) (inputs : Array ByteArray) (count : UInt64) :
    IO ByteArray := do
  let (source, tag) ← match scalar.precision with
    | .native format => do
        let source ← IO.ofExcept (CUDA.emit format (fun x => (scalar.toBits x).toNat.toUInt64)
          inputs.size expr)
        pure (source.text, if format == .binary32 then (0 : UInt32) else 1)
    | .binary format => do
        let source ← IO.ofExcept
          (CUDA.emitBinary format (fun x => (scalar.toBits x).toNat) inputs.size expr)
        pure (source, (2 : UInt32))
  executeBytes source tag scalar.precision.bytes.toUInt64 inputs count

/-- Run `expr`, generated for `format`, on resident buffers of that format and return a fresh
output buffer.

Inputs are borrowed and never mutated. Bounds errors and compilation failures are returned before
an output buffer is exposed.
-/
@[no_expose] def runBuffers {α : Type} (format : CUDA.Format) (bits : α → UInt64)
    (expr : Expr α [.index] .scalar) (inputs : Array Buffer) (count : UInt64) : IO Buffer := do
  let source ← IO.ofExcept (CUDA.emit format bits inputs.size expr)
  executeBuffer source.text (match format with | .binary32 => .float32 | .binary64 => .float64)
    inputs count

end TorchLean.Kernel

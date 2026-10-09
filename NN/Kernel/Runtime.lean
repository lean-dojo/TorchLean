/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Kernel.Cuda
public import NN.Kernel.Cuda.Binary
public import NN.Runtime.Autograd.Engine.LibTorch.Buffer

/-!
# Native execution of generated custom operations

CUDA source is generated from typed expressions, not accepted as caller-provided text. NVRTC
compiles it with the native bridge's fixed floating-point policy. The bridge caches a
bounded number of modules per process, keys them by source and context, and launches on LibTorch's
current stream. Input buffers and the module remain alive until the bounds record is checked.

This is an explicit foreign-code boundary, not a Lean theorem about NVRTC or device execution.
LibTorch continues to own tensor storage and its established numerical operations. CPU-only
builds resolve the same extern symbols but report an ordinary IO error instead of executing CUDA.
-/

@[expose] public section

namespace NN.Kernel.Internal

open _root_.Runtime.Autograd.LibTorch

@[never_extract, extern "torchlean_kernel_run_buffer"]
private opaque executeBuffer (source : @& String) (inputs : @& Array Buffer)
    (count : UInt64) : IO Buffer

@[never_extract, extern "torchlean_kernel_run_bytes"]
private opaque executeBytes (source : @& String) (format : UInt32) (width : UInt64)
    (inputs : @& Array ByteArray) (count : UInt64) : IO ByteArray

/-- Execute with complete scalar encodings. Wide binary values never pass through native floats. -/
@[no_expose] def runBytes {α : Type} (scalar : Scalar α)
    (expr : Expr α [.index] .scalar) (inputs : Array ByteArray) (count : UInt64) :
    IO ByteArray := do
  let (source, tag) ← match scalar.precision with
    | .native format => do
        let source ← IO.ofExcept (Cuda.source format (fun x => (scalar.encode x).toUInt64)
          inputs.size expr)
        pure (source.text, if format == .binary32 then (0 : UInt32) else 1)
    | .binary format => do
        let source ← IO.ofExcept (Cuda.binarySource format scalar.encode inputs.size expr)
        pure (source, (2 : UInt32))
  executeBytes source tag scalar.precision.bytes.toUInt64 inputs count

/-- Run a generated operation on existing resident binary32 buffers.

Inputs are borrowed and never mutated. Bounds errors and compilation failures are returned before
an output buffer is exposed. This helper is binary32-specific; other precisions use tensor
execution.
-/
@[no_expose] def runBuffers {α : Type} (format : Cuda.Format) (bits : α → UInt64)
    (expr : Expr α [.index] .scalar) (inputs : Array Buffer) (count : UInt64) : IO Buffer := do
  if format != .binary32 then
    throw (IO.userError "kernel: this operation requires binary32; use tensor execution otherwise")
  let source ← IO.ofExcept (Cuda.source format bits inputs.size expr)
  executeBuffer source.text inputs count

end NN.Kernel.Internal

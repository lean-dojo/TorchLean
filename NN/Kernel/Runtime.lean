/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Kernel.Cuda
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

@[never_extract, extern "torchlean_kernel_run_host"]
private opaque executeHost (source : @& String) (format : UInt32)
    (inputs : @& Array FloatArray) (count : UInt64) : IO FloatArray

/-- Run a generated operation on existing resident binary32 buffers.

Inputs are borrowed and never mutated. Bounds errors and compilation failures are returned before
an output buffer is exposed. Binary64 requires the host-array interface because the existing
resident-buffer ABI stores binary32 values.
-/
@[no_expose] def runBuffers {α : Type} (format : Cuda.Format) (bits : α → UInt64)
    (expr : Expr α [.index] .scalar) (inputs : Array Buffer) (count : UInt64) : IO Buffer := do
  if format != .binary32 then
    throw (IO.userError "kernel: resident buffers store binary32; use runHost for binary64")
  let source ← IO.ofExcept (Cuda.source format bits inputs.size expr)
  executeBuffer source.text inputs count

/-- Upload inputs, execute a generated operation, and download the result.

The selected format controls computation on the device. Host arrays contain Lean `Float` values;
binary32 uploads round them once, while binary64 preserves them. Input reads and all generated
arithmetic subsequently use the selected device format. This interface does not promise a theorem
about that external execution.
-/
@[no_expose] def runHost {α : Type} (format : Cuda.Format) (bits : α → UInt64)
    (expr : Expr α [.index] .scalar) (inputs : Array FloatArray)
    (count : UInt64) : IO FloatArray := do
  let source ← IO.ofExcept (Cuda.source format bits inputs.size expr)
  let format : UInt32 := match format with | .binary32 => 0 | .binary64 => 1
  executeHost source.text format inputs count

end NN.Kernel.Internal

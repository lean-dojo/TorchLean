/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

The opaque CUDA buffer type and the map of the native sources behind TorchLean's FFI.
-/

module

/-!
# Trusted CUDA Runtime Boundary

The opaque CUDA buffer type of the native runtime. Buffers are created only by the FFI allocation
and copy functions. The nonemptiness witness below exists so that Lean accepts extern functions
returning `Buffer`. It is not a default CUDA allocation, and no code should use it as one.

DocGen documents only Lean modules, so the map from native translation units to the Lean modules
that call them lives here, in the module that forms the trust boundary.

## Trust boundary

The CUDA backend crosses a C++/LibTorch FFI boundary, and Lean does not prove the compiled native
implementation correct. The trusted pieces include LibTorch and ATen, their CUDA libraries,
compiler and runtime, GPU hardware, and Lean's external-object ABI and finalizers.

TorchLean keeps its own tape and selected local VJPs, while ATen computes tensor values and local
backward operations with autograd recording disabled. Proof-facing kernel specifications, float32
agreement hypotheses, and graph semantics are Lean definitions. Runtime regression and numerical
parity tests give evidence for one build on the inputs they run.

## Native source groups

- `csrc/libtorch/torchlean_libtorch.h`
  Shared boxed-buffer ABI, size checks, device guards, and the no-autograd call boundary.
  Lean modules: `TorchLean.CUDA.LibTorch.Trusted` and `TorchLean.CUDA.LibTorch.Buffer`.

- `csrc/libtorch/torchlean.cpp`
  Shared implementation of storage ownership, allocation, transfers, seeded random values,
  runtime controls, tensor operations, and direct calls to ATen backward operators.
  It implements elementwise operations, reductions, indexing, matrix multiplication,
  normalization, Fourier transforms and scans, and it compiles and launches generated kernels
  through NVRTC.
  Lean modules: `TorchLean.CUDA.LibTorch.Buffer`, `TorchLean.CUDA.LibTorch.Kernels`,
  `TorchLean.CUDA.LibTorch.Controls` and `TorchLean.CUDA.Kernel.Launch`.
  Lean composes attention in `TorchLean.NN.LibTorch.Attention` and the multi-head node of
  `TorchLean.LibTorch.Kernel.ofOp` keeps Q/K/V and probability buffers on its tape, so the native
  buffer has no attention context. LibTorch owns the CUDA allocator, and TorchLean's payload
  counters track logical ownership. Native buffers carry binary32 or binary64 according to the
  executable scalar type.

- `csrc/libtorch/conv_pool.cpp`
  Grouped and transposed convolutions and max, average and smooth max pools through ATen, each
  with its own backward call.
  Lean modules: `TorchLean.CUDA.LibTorch.ConvPool` and `TorchLean.CUDA.LibTorch.Kernels`.

- `csrc/libtorch/binary.h`
  Device addition, multiplication and division in configured binary formats, which NVRTC receives
  in memory as the header `torchlean_binary.cuh`.
  Lean module: `TorchLean.CUDA.Kernel.Emit.Binary`, whose generated kernels include that header.

- `csrc/libtorch/operations.h`
  Common operation list that generates the LibTorch exports and the unavailable-build signatures.

- `csrc/libtorch/unavailable.c`
  The same symbols for builds without LibTorch. The runtime status is `.notLinked`, settings read as
  off and memory statistics as zero, IO constructors return an error, and pure buffer operations
  abort with a message to rebuild with `-K cuda=true`. User CUDA sessions are rejected before any of
  these calls runs.

- `csrc/libtorch/tests/elementwise/regression.cpp`
  A C++ regression executable for the production backend's elementwise C ABI, built against the
  backend's SDK. No Lean module calls it.

- `csrc/cpu/torchlean_tensor.c`
  Packed CPU kernels over `FloatArray` and `Array` storage: slice append, two-dimensional
  transpose, pointwise arithmetic, and addition of packed bytes to packed doubles. Weak boxed
  wrappers let Lean's evaluator find them in the package's dynamic library.
  Lean modules: `TorchLean.Tensor.Storage`,
  `TorchLean.Tensor.Native.Transpose` and
  `TorchLean.Tensor.Native.Pointwise`.

- `csrc/runtime/allocator.cpp` and `csrc/runtime/lean_libc_compat.h`
  Lean's small-object allocation entry points, compiled in one translation unit with mimalloc and
  linked on Linux unless `-K sharedRuntime=true`. The header keeps the allocator's option parser
  on the pre-C23 `strtol` ABI of Lean's linker sysroot. No Lean module calls them.

The GPU random stream computes the SplitMix64 draws of `TorchLean.Random.sampleNat` with ATen
integer operations. The CUDA stress tests compare its Bernoulli masks exactly with
`TorchLean.Random.keepBit` and its uniform buffers with `d / 2^32` within `1e-7`.

-/

@[expose] public section

namespace TorchLean
namespace LibTorch

/-- Storage format of a tape buffer. Native buffers hold binary32 or binary64, and an encoded
buffer keeps its scalars in their own format, with the complete format descriptor on its handle. -/
inductive Dtype where
  /-- Native binary32 scalars. -/
  | float32
  /-- Native binary64 scalars. -/
  | float64
  /-- Scalars in a configured format, with the format descriptor on the handle. -/
  | encoded
  deriving Repr, DecidableEq, BEq, Nonempty

/-- Bytes per native scalar. Encoded buffers give `0`: their word width comes from the format
descriptor, and they cannot use the native checkpoint framing. -/
def Dtype.bytes : Dtype → Nat
  | .float32 => 4
  | .float64 => 8
  | .encoded => 0

/--
Opaque handle to a contiguous, dtype-carrying CUDA buffer, the `torchlean_cuda_buffer` declared in
`csrc/libtorch/torchlean_libtorch.h`. Builds without `-K cuda=true` cannot create one.
-/
opaque BufferPointed : NonemptyType.{0}

/--
Lean type of the native CUDA buffer handles.

The `NonemptyType` wrapper is Lean's standard representation of an external resource. It gives
extern declarations a nonempty result type, and compiled code reference-counts the handles like any
Lean object.
Only the native buffer constructors create a value.
-/
def Buffer : Type := BufferPointed.val

/-- `Buffer` is nonempty, as its extern declarations need. -/
instance : Nonempty Buffer := BufferPointed.property

end LibTorch
end TorchLean

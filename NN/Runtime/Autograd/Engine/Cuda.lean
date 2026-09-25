/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Buffer
public import NN.Runtime.Autograd.Engine.Cuda.ConvPool
public import NN.Runtime.Autograd.Engine.Cuda.Convert
public import NN.Runtime.Autograd.Engine.Cuda.DGemm
public import NN.Runtime.Autograd.Engine.Cuda.Float32Contract
public import NN.Runtime.Autograd.Engine.Cuda.Fno1dRfft
public import NN.Runtime.Autograd.Engine.Cuda.KernelSpec
public import NN.Runtime.Autograd.Engine.Cuda.Kernels
public import NN.Runtime.Autograd.Engine.Cuda.LibTorch
public import NN.Runtime.Autograd.Engine.Cuda.Ops
public import NN.Runtime.Autograd.Engine.Cuda.Shape
public import NN.Runtime.Autograd.Engine.Cuda.Tape
public import NN.Runtime.Autograd.Engine.Cuda.Trusted

/-!
# CUDA engine for eager execution

This umbrella collects the CUDA side of TorchLean's eager autograd engine.

The modules separate native execution from the tape and proof-facing contracts:

- `Trusted` and `Buffer` expose the opaque FFI buffer type and allocation/copy primitives.
- `LibTorch` exposes precision, determinism, SDP, device, allocator, and version controls.
- `Kernels`, `ConvPool`, and `DGemm` declare LibTorch CUDA and CPU-stub entrypoints.
- `Tape` and `Ops` build the CUDA reverse-mode tape over those buffers.
- `Float32Contract` and `KernelSpec` state the proof layer reference contracts for native bits.

The LibTorch bridge executes without recording a LibTorch autograd graph. TorchLean keeps tape
traversal and selected local VJPs. Lean proves the pure specs and graph-level connections; runtime
controls and tests do not prove the compiled native implementation.
-/

@[expose] public section

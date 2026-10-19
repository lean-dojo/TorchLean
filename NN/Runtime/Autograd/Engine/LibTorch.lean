/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.LibTorch.Buffer
public import NN.Runtime.Autograd.Engine.LibTorch.ConvPool
public import NN.Runtime.Autograd.Engine.LibTorch.Convert
public import NN.Runtime.Autograd.Engine.LibTorch.DGemm
public import NN.Runtime.Autograd.Engine.LibTorch.Float32Contract
public import NN.Runtime.Autograd.Engine.LibTorch.KernelSpec
public import NN.Runtime.Autograd.Engine.LibTorch.Kernels
public import NN.Runtime.Autograd.Engine.LibTorch.Controls
public import NN.Runtime.Autograd.Engine.LibTorch.DeviceInfo
public import NN.Runtime.Autograd.Engine.LibTorch.Ops
public import NN.Runtime.Autograd.Engine.LibTorch.Shape
public import NN.Runtime.Autograd.Engine.LibTorch.Tape
public import NN.Runtime.Autograd.Engine.LibTorch.Trusted

/-!
# LibTorch backend for eager execution

This umbrella collects TorchLean's LibTorch adapter, currently targeting CUDA devices.

The modules separate native execution from the tape and proof-facing contracts:

- `Trusted` and `Buffer` expose the opaque FFI buffer type and allocation/copy primitives.
- `Controls` exposes precision, determinism, SDP, device, allocator, and version controls.
- `Kernels`, `ConvPool`, and `DGemm` declare the LibTorch CUDA entrypoints.
- `Tape` and `Ops` build the CUDA reverse-mode tape over those buffers.
- `Float32Contract` and `KernelSpec` state the proof layer reference contracts for native bits.

The LibTorch bridge executes without recording a LibTorch autograd graph. TorchLean keeps tape
traversal and selected local VJPs. Lean proves the pure specs and graph-level connections; runtime
controls and tests do not prove the compiled native implementation.
-/

@[expose] public section

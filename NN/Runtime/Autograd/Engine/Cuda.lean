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
public import NN.Runtime.Autograd.Engine.Cuda.Fno1dRfftFused
public import NN.Runtime.Autograd.Engine.Cuda.KernelSpec
public import NN.Runtime.Autograd.Engine.Cuda.Kernels
public import NN.Runtime.Autograd.Engine.Cuda.NativeSources
public import NN.Runtime.Autograd.Engine.Cuda.Ops
public import NN.Runtime.Autograd.Engine.Cuda.Shape
public import NN.Runtime.Autograd.Engine.Cuda.Tape
public import NN.Runtime.Autograd.Engine.Cuda.TexTable
public import NN.Runtime.Autograd.Engine.Cuda.Trusted

/-!
# CUDA engine for eager execution

This umbrella collects the CUDA side of TorchLean's eager autograd engine.

The modules here are split by trust boundary by trust boundary:

- `Trusted` and `Buffer` expose the opaque FFI buffer type and allocation/copy primitives.
- `Kernels`, `ConvPool`, and `DGemm` declare native CUDA/CPU-stub kernel entrypoints.
- `Tape` and `Ops` build the CUDA reverse-mode tape over those buffers.
- `Float32Contract` and `KernelSpec` state the proof layer reference contracts for native bits.
- `NativeSources` documents which C/CUDA files implement the external symbols.

The compiled CUDA binary is a native trust boundary. Lean proves the pure specs and graph-level
connections around this boundary, while runtime tests validate the native implementation path.
-/

@[expose] public section

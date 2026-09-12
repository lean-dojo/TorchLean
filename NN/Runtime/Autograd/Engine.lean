/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Core
public import NN.Runtime.Autograd.Engine.Cuda
public import NN.Runtime.Autograd.Engine.FastKernels
public import NN.Runtime.Autograd.Engine.TapeM

/-!
# Autograd engine

This is the public umbrella for TorchLean's low-level eager autograd engine.

- `Engine.Core` is the pure CPU tape over shape-erased `Spec.SomeTensor` values.
- `Engine.TapeM` is a `StateT` layer around the pure tape.
- `Engine.FastKernels` provides opt-in runtime kernels for hot CPU/GPU paths.
- `Engine.Cuda` collects the CUDA float32 tape, FFI kernels, and proof layer native contracts.

Higher-level APIs should usually import `NN.Runtime.Autograd.Torch` or
`NN.Runtime.Autograd.Model`; this module is for code that works directly at the tape
engine boundary.
-/

@[expose] public section

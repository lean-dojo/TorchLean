/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Tape
public import NN.Runtime.Autograd.Engine.Core
public import NN.Runtime.Autograd.Engine.LibTorch

/-!
# Autograd engine

This is the public umbrella for TorchLean's low-level eager autograd engine.

- `Engine.Tape` is the tape: a grow-only array of nodes with local VJPs, and the reverse sweep
  that accumulates their contributions. It is generic in the type of the values it stores.
- `Engine.Core` is the CPU side: the tape over shape-erased `Spec.SomeTensor` values and the CPU
  operations.
- `Engine.LibTorch` is the CUDA side: the same tape over device buffers, the CUDA operations, FFI
  kernels, and proof layer native contracts.

Higher-level APIs should usually import `NN.Runtime.Autograd.Torch` or
`NN.Runtime.Autograd.Model`; this module is for code that works directly at the tape
engine boundary.
-/

@[expose] public section

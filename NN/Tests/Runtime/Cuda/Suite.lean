/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tests.Runtime.ParameterAliases
public import NN.Tests.Runtime.Cuda.Softmax
public import NN.Tests.Runtime.Cuda.Elementwise
public import NN.Tests.Runtime.Cuda.Attention
public import NN.Tests.Runtime.Cuda.SelectiveScan
public import NN.Tests.Runtime.Cuda.Fft
public import NN.Tests.Runtime.Cuda.Stress
public import NN.Tests.Runtime.Cuda.Trainer

/-!
# Suite

Focused LibTorch CUDA runtime checks.

Checks of numerical primitives, tape wiring, buffer ownership, shape validation, and training
through the LibTorch CUDA path. Comparisons with CPU execution and explicit reference values cover
the exercised cases; they do not prove correctness of the SDK's kernels for every input.
-/

@[expose] public section

namespace Tests
namespace Cuda

/-- Unified CUDA test entrypoint (called by `NN/Tests/Suite.lean`). -/
def run : IO Unit := do
  NN.Tests.Runtime.ParameterAliases.runCuda
  IO.println "=== Runtime CUDA checks ==="
  Softmax.run
  Elementwise.run
  Attention.run
  SelectiveScan.run
  Fft.run
  Stress.run
  Trainer.run
  IO.println "=== CUDA checks completed ==="

end Cuda
end Tests

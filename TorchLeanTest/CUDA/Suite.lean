/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLeanTest.CUDA.ParameterAliases
public import TorchLeanTest.CUDA.Numerics
public import TorchLeanTest.CUDA.Softmax
public import TorchLeanTest.CUDA.Elementwise
public import TorchLeanTest.CUDA.Attention
public import TorchLeanTest.CUDA.SelectiveScan
public import TorchLeanTest.CUDA.FFT
public import TorchLeanTest.CUDA.Stress
public import TorchLeanTest.CUDA.ReduceTree
public import TorchLeanTest.CUDA.Indexing
public import TorchLeanTest.CUDA.Graph
public import TorchLeanTest.CUDA.Trainer

/-!
# Suite

One entry point runs the LibTorch CUDA runtime checks of numerical primitives, tape wiring, buffer
ownership, shape validation and training. They compare CUDA results with CPU execution and with
reference values, so the SDK's kernels are tested on the exercised cases only.
-/

@[expose] public section

namespace TorchLean.Test
namespace CUDA

/-- Runs the CUDA checks of `ParameterAliases` and then those of each CUDA module imported above.
`TorchLeanTest/Suite.lean` calls it when LibTorch is linked and a CUDA device is visible. -/
def run : IO Unit := do
  ParameterAliases.runCUDA
  IO.println "=== Runtime CUDA checks ==="
  Numerics.run
  Softmax.run
  Elementwise.run
  Attention.run
  SelectiveScan.run
  FFT.run
  Stress.run
  ReduceTree.run
  Indexing.run
  Graph.run
  Trainer.run
  IO.println "=== CUDA checks completed ==="

end CUDA
end TorchLean.Test

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tests.Runtime.Cuda.Softmax
public import NN.Tests.Runtime.Cuda.Elementwise
public import NN.Tests.Runtime.Cuda.LayerNorm
public import NN.Tests.Runtime.Cuda.BatchNorm
public import NN.Tests.Runtime.Cuda.Attention
public import NN.Tests.Runtime.Cuda.ConvPool
public import NN.Tests.Runtime.Cuda.ConvTranspose
public import NN.Tests.Runtime.Cuda.GatherScatter
public import NN.Tests.Runtime.Cuda.DeterministicReductions
public import NN.Tests.Runtime.Cuda.SelectiveScan
public import NN.Tests.Runtime.Cuda.PositionalEncoding
public import NN.Tests.Runtime.Cuda.Matmul
public import NN.Tests.Runtime.Cuda.Fft
public import NN.Tests.Runtime.Cuda.ViewsBroadcastReduce
public import NN.Tests.Runtime.Cuda.LinearMseConcatSliceGather
public import NN.Tests.Runtime.Cuda.TexTable
public import NN.Tests.Runtime.Cuda.Stress

/-!
# Suite

CUDA kernel-coverage regression tests.

These tests are deliberately small and deterministic. CUDA correctness lives at the native trust
boundary; this suite compares the Lean side eager/autograd behavior with the CUDA FFI path so
memory, shape, and numerical regressions are caught during regression testing.
-/

@[expose] public section

namespace Tests
namespace Cuda

/-- Unified CUDA test entrypoint (called by `NN/Tests/Suite.lean`). -/
def run : IO Unit := do
  IO.println "=== Runtime CUDA kernel coverage suite ==="
  Softmax.run
  Elementwise.run
  LayerNorm.run
  BatchNorm.run
  Attention.run
  ConvPool.run
  ConvTranspose.run
  GatherScatter.run
  DeterministicReductions.run
  SelectiveScan.run
  PositionalEncoding.run
  Matmul.run
  Fft.run
  ViewsBroadcastReduce.run
  LinearMseConcatSliceGather.run
  TexTable.run
  Stress.run
  IO.println "=== CUDA kernel coverage suite completed ==="

end Cuda
end Tests

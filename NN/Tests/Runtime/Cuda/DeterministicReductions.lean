/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Buffer
public import NN.Runtime.Autograd.Engine.Cuda.Kernels
public import NN.Runtime.Autograd.Engine.Cuda.ConvPool
public import NN.Tests.Runtime.Cuda.Utils
public import Std

/-!
# CUDA Deterministic Reductions Mode

These tests exercise TorchLean's opt-in "deterministic reductions" mode.

Goal: when deterministic mode is enabled, kernels that would otherwise accumulate using `atomicAdd`
must become bit-stable across runs (same input, same output, exact float equality).

The tests are written so they run:
- with CUDA enabled (`lake test -K cuda=true`) using real device buffers, and
- with CUDA disabled (default) via the CPU stub implementations.
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace DeterministicReductions

open Runtime.Autograd.Cuda

-- Exact numerical buffer comparison, shared with `Stress` through `Cuda.Utils`.
-- Signed-zero preservation is checked separately through `toBits`.
open Tests.Cuda.Utils (assertFloatArrayEq)

def runScatterAddTwice : IO Unit := do
  IO.println "== deterministic scatter_add: exact repeatability =="

  -- Enable deterministic mode via Lean side API.
  Buffer.setDeterministicReductions true

  -- Construct a values buffer with a large leading element + many ones, and scatter-add all
  -- updates into a single output index. This is a worst-case accumulator for non-associativity.
  let k : UInt32 := 4096
  let one : UInt32 := 1
  let n : UInt32 := 1

  let x ← Buffer.zerosIO n
  let big := Buffer.full one 1.0e8
  let ones := Buffer.full (k - one) 1.0
  let values := Buffer.concatBuffers big ones one (k - one)

  let idx : Array Nat := Array.replicate k.toNat 0
  let y1 := Buffer.scatterAdd x values n idx k
  let y2 := Buffer.scatterAdd x values n idx k

  assertFloatArrayEq "scatterAdd deterministic run1 vs run2"
    (Buffer.toFloatArray y1) (Buffer.toFloatArray y2)

/-- Deterministic scatter starts its left fold at the base value, including its signed zero. -/
def runScatterAddBaseOrder : IO Unit := do
  Buffer.setDeterministicReductions true
  let base := Buffer.ofFloatArray (FloatArray.mk #[1.0e8, -0.0])
  let values := Buffer.ofFloatArray (FloatArray.mk #[-1.0e8, 1.0])
  let result := Buffer.scatterAdd base values 2 #[0, 0] 2
  assertFloatArrayEq "scatterAdd includes base before updates"
    (Buffer.toFloatArray result) (FloatArray.mk #[1.0, -0.0])
  unless ((Buffer.toFloatArray result).get! 1).toBits == (-0.0 : Float).toBits do
    throw <| IO.userError "scatterAdd changed an untouched negative zero"
  let rows := Buffer.ofFloatArray (FloatArray.mk #[1.0e8, 1.0e8])
  let rowValues := Buffer.ofFloatArray (FloatArray.mk #[-1.0e8, -1.0e8, 1.0, 1.0])
  let rowResult := Buffer.scatterAddRows rows rowValues 1 2 #[0, 0] 2
  assertFloatArrayEq "scatterAddRows includes base before updates"
    (Buffer.toFloatArray rowResult) (FloatArray.mk #[1.0, 1.0])

def outDim (inDim k stride padding : Nat) : Nat :=
  Spec.Shape.slidingWindowOutDim inDim k stride padding

def runAvgPoolBwdTwice : IO Unit := do
  IO.println "== deterministic avg_pool backward: exact repeatability =="

  Buffer.setDeterministicReductions true

  -- A small overlapping-window case (stride=1) so the backward pass needs accumulation.
  let inC : UInt32 := 1
  let inSpatial : Array Nat := #[17, 17]
  let kernel : Array Nat := #[3, 3]
  let stride : Array Nat := #[1, 1]
  let padding : Array Nat := #[1, 1]

  let outH : Nat := outDim 17 3 1 1
  let outW : Nat := outDim 17 3 1 1
  let outElems : UInt32 := UInt32.ofNat (inC.toNat * outH * outW)

  let gradOutput ← Buffer.randUniformIO outElems 12345
  let y1 := torchleanAvgPoolBwdCuda gradOutput inSpatial kernel stride padding inC
  let y2 := torchleanAvgPoolBwdCuda gradOutput inSpatial kernel stride padding inC

  assertFloatArrayEq "avg_pool backward deterministic run1 vs run2"
    (Buffer.toFloatArray y1) (Buffer.toFloatArray y2)

/-- Entry point called by the CUDA runtime suite. -/
def run : IO Unit := do
  IO.println "== CUDA deterministic reductions =="
  runScatterAddTwice
  runScatterAddBaseOrder
  runAvgPoolBwdTwice
  Buffer.setDeterministicReductions false
  IO.println "== CUDA deterministic reductions: OK =="

end DeterministicReductions
end Cuda
end Tests

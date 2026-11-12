/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.CUDA.LibTorch.Buffer
public import TorchLean.CUDA.LibTorch.Controls
public import TorchLean.CUDA.LibTorch.Kernels

/-!
# CUDA rounding and cross-entropy underflow

`Buffer.axpy` rounds `a + c * b` once because its generated kernel calls `fma`, and the
deterministic `Buffer.reduceSumAxis` folds each column in order because its generated kernel sums
each column from `+0` in one thread, row by row, in the input's format.

With `e` the spacing at one, `a = -(1 + 2e)` and `b = c = 1 + e` give `e * e` under one rounding
and `0` when `c * b` rounds first. With `p` the precision, `2^p`, 4095 ones and `-2^p` fold to `0`
in order, since each `2^p + 1` is a tie that rounds to `2^p`, while a reversed or regrouped fold,
or one in a wider format, keeps some of the ones.

Cross-entropy keeps a nonzero loss and target gradient when each non-target exponential
underflows but their combined mass is representable. The binary32 and binary64 cases check the
least subnormal exactly, since an absolute tolerance would admit zero.
-/

@[expose] public section

namespace TorchLean.Test
namespace CUDA
namespace Numerics

open LibTorch

/-- Download the one value of `buffer` and release the buffer. -/
def readOne (label : String) (buffer : Buffer) : IO Float := do
  let host ← Buffer.toFloatArrayIO buffer
  discard <| Buffer.releaseIO buffer
  unless host.size == 1 do
    throw <| IO.userError s!"{label}: native result has {host.size} values"
  pure host[0]!

/-- Check both fixtures in `dtype`, whose binary format `name` has `precision` significand bits. -/
def checkDtype (dtype : Dtype) (name : String) (precision : Nat) : IO Unit := do
  let spacing := Float.scaleB 1.0 (1 - (precision : Int))
  let top := Float.scaleB 1.0 precision
  let a ← Buffer.ofFloatArrayIO (FloatArray.mk #[-(1.0 + 2.0 * spacing)]) dtype
  let b ← Buffer.ofFloatArrayIO (FloatArray.mk #[1.0 + spacing]) dtype
  let fused ← readOne s!"axpy {name}" (Buffer.axpy a b (1.0 + spacing))
  discard <| Buffer.releaseIO a
  discard <| Buffer.releaseIO b
  unless fused == spacing * spacing do
    throw <| IO.userError
      s!"numerics: Buffer.axpy rounds the {name} multiply-add twice"
  let values := (#[top] ++ Array.replicate 4095 1.0).push (-top)
  let data ← Buffer.ofFloatArrayIO (FloatArray.mk values) dtype
  let folded ← readOne s!"reduceSumAxis {name}" (Buffer.reduceSumAxis data #[values.size] 0)
  discard <| Buffer.releaseIO data
  unless folded == 0.0 do
    throw <| IO.userError s!"numerics: Buffer.reduceSumAxis does not fold {name} columns in order"

/-- A two-level cross-entropy normalizer retains the least subnormal at logits `[0, -gap, -gap]`
with target class `0`, in both the loss and its target-logit pullback. -/
def checkCrossEntropyUnderflow (dtype : Dtype) (gap least : Float) : IO Unit := do
  let logits ← Buffer.ofFloatArrayIO (FloatArray.mk #[0, -gap, -gap]) dtype
  let target ← Buffer.ofFloatArrayIO (FloatArray.mk #[1, 0, 0]) dtype
  let seed ← Buffer.ofFloatArrayIO (FloatArray.mk #[1]) dtype
  let loss ← readOne "cross-entropy underflow"
    (Buffer.crossEntropy logits target #[3] 0 false)
  let (dx, dt) := Buffer.crossEntropyBackward logits target seed #[3] 0 false
  let gradient ← Buffer.toFloatArrayIO dx
  for buffer in [logits, target, seed, dx, dt] do
    discard <| Buffer.releaseIO buffer
  unless loss == least do
    throw <| IO.userError "numerics: cross-entropy lost the combined subnormal tail"
  unless gradient.size == 3 && gradient[0]! == -least do
    throw <| IO.userError "numerics: cross-entropy lost the subnormal target gradient"

/-- The fixtures in binary32 and binary64, with deterministic algorithms enabled and the previous
setting restored. -/
def run : IO Unit := do
  let previous ← getDeterministic
  setDeterministic true
  unless ← getDeterministic do
    throw <| IO.userError "numerics: deterministic algorithms could not be enabled"
  try
    checkDtype .float32 "binary32" 24
    checkDtype .float64 "binary64" 53
    checkCrossEntropyUnderflow .float32 104 (Float.scaleB 1.0 (-149))
    checkCrossEntropyUnderflow .float64 745.5 (Float.scaleB 1.0 (-1074))
    IO.println "  numerics: fused arithmetic, ordered reductions and cross-entropy underflow passed"
  finally
    setDeterministic previous

end Numerics
end CUDA
end TorchLean.Test

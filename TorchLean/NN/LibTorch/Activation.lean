/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.LibTorch.Tape
public import TorchLean.CUDA.LibTorch.Kernels

/-!
# CUDA Tape Operations: Shared Helpers
-/

@[expose] public section

namespace TorchLean
namespace LibTorch

open TorchLean.Tensor

namespace Tape

/-- Reject shape metadata containing a dimension outside the CUDA `UInt32` ABI. -/
def validateU32Dimensions (opName : String) (dims : Array Nat) : Result Unit := do
  for dim in dims do
    if (Bytes.toUInt32? dim).isNone then
      throw s!"autograd: cuda: {opName}: dimension does not fit in UInt32"

/-- Numerically stable softplus: `max(x,0) + log1p(exp(-abs(x)))`. -/
def softplusBuf (x : Buffer) (n : UInt32) : Buffer :=
  let zeros := Buffer.full n 0.0 (Buffer.dtype x)
  let max0 := Buffer.max x zeros
  let absx := Buffer.abs x
  let negAbs := Buffer.scale absx (-1.0)
  let expNegAbs := Buffer.exp negAbs
  let logTerm := Buffer.log1p expNegAbs
  let y := Buffer.add max0 logTerm
  Buffer.releaseThen zeros <| Buffer.releaseThen max0 <|
    Buffer.releaseThen absx <| Buffer.releaseThen negAbs <| Buffer.releaseThen expNegAbs <|
      Buffer.releaseThen logTerm y

/--
Row-wise stable softmax.

The returned `WithWorkspace` owns the buffers used to compute the stable formula. Backward needs
only the output, so callers can release the workspace as soon as they have consumed the forward
result.
-/
def rowSoftmaxForward (x : Buffer) (rows cols : UInt32) : Buffer.WithWorkspace :=
  let rowMax := Buffer.reduceMaxByRow x rows cols
  let maxB := Buffer.broadcastVecToCols rowMax rows cols
  let shifted := Buffer.sub x maxB
  let ex := Buffer.exp shifted
  let rowSum := Buffer.reduceSumByRow ex rows cols
  let sumB := Buffer.broadcastVecToCols rowSum rows cols
  let y := Buffer.div ex sumB
  { value := y, workspace := #[rowMax, maxB, shifted, ex, rowSum, sumB] }

/-- Row-wise softmax VJP, centering the upstream values at a maximal probability before
taking their weighted mean, which retains small derivatives at saturated outputs. -/
def rowSoftmaxBackward (y dLdy : Buffer) (rows cols : UInt32) : Buffer :=
  let pivot := Buffer.argmaxMaskByRow y rows cols
  let selected := Buffer.mask dLdy pivot
  let anchor := Buffer.reduceSumByRow selected rows cols
  let anchorB := Buffer.broadcastVecToCols anchor rows cols
  let upstream := Buffer.sub dLdy anchorB
  let dy_y := Buffer.mul upstream y
  let dot := Buffer.reduceSumByRow dy_y rows cols
  let dotB := Buffer.broadcastVecToCols dot rows cols
  let centered := Buffer.sub upstream dotB
  Buffer.releaseManyThen #[pivot, selected, anchor, anchorB, upstream, dy_y, dot, dotB, centered]
    (Buffer.mul y centered)

/--
Row-wise stable log-softmax.

One maximal exponential supplies the exact unit term. The other exponentials are summed
separately before `log1p`, retaining small tails that would disappear in `log(1 + tail)`.
As with `rowSoftmaxForward`, backward needs only the output, so callers can release the workspace
once the forward result is consumed.
-/
def rowLogSoftmaxForward (x : Buffer) (rows cols : UInt32) : Buffer.WithWorkspace :=
  let rowMax := Buffer.reduceMaxByRow x rows cols
  let maxB := Buffer.broadcastVecToCols rowMax rows cols
  let shifted := Buffer.sub x maxB
  let ex := Buffer.exp shifted
  let pivot := Buffer.argmaxMaskByRow x rows cols
  let tail := Buffer.sub ex pivot
  let rowSum := Buffer.reduceSumByRow tail rows cols
  let logSum := Buffer.log1p rowSum
  let logSumB := Buffer.broadcastVecToCols logSum rows cols
  let y := Buffer.sub shifted logSumB
  { value := y, workspace := #[rowMax, maxB, shifted, ex, pivot, tail, rowSum, logSum, logSumB] }

/-- Row-wise log-softmax VJP with TorchLean's own backward formula. It separates the pivot
contribution and uses `expm1` there to avoid subtracting two rounded unit probabilities. -/
def rowLogSoftmaxBackward (y dLdy : Buffer) (rows cols : UInt32) : Buffer :=
  let pivot := Buffer.argmaxMaskByRow y rows cols
  let ones := Buffer.full (rows * cols) 1 (Buffer.dtype y)
  let rest := Buffer.sub ones pivot
  let others := Buffer.mask dLdy rest
  let otherSum := Buffer.reduceSumByRow others rows cols
  let otherSumB := Buffer.broadcastVecToCols otherSum rows cols
  let probs := Buffer.exp y
  let rowSum := Buffer.reduceSumByRow dLdy rows cols
  let sumB := Buffer.broadcastVecToCols rowSum rows cols
  let scaled := Buffer.mul probs sumB
  let regular := Buffer.sub dLdy scaled
  let regularMasked := Buffer.mask regular rest
  let complement := Buffer.expm1 y
  let diagonal := Buffer.mul complement dLdy
  let negativeDiagonal := Buffer.neg diagonal
  let offDiagonal := Buffer.mul probs otherSumB
  let special := Buffer.sub negativeDiagonal offDiagonal
  let specialMasked := Buffer.mask special pivot
  Buffer.releaseManyThen
    #[pivot, ones, rest, others, otherSum, otherSumB, probs, rowSum, sumB, scaled,
      regular, regularMasked, complement, diagonal, negativeDiagonal, offDiagonal,
      special, specialMasked] (Buffer.add regularMasked specialMasked)
end Tape

end LibTorch
end TorchLean

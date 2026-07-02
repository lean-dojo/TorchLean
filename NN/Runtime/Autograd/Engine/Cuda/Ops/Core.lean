/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Tape
public import NN.Runtime.Autograd.Engine.Cuda.Kernels
public import NN.Runtime.Autograd.Engine.Cuda.ConvPool
public import NN.Runtime.Autograd.Engine.Cuda.Convert
public import NN.Runtime.Autograd.Engine.Cuda.Shape
public import NN.Spec.Core.TensorReductionShape

/-!
# CUDA Tape Operations: Shared Helpers
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Cuda

open Spec
open Tensor

namespace Tape

/-!
## Small helpers
-/

/-- Checked `Nat → UInt32` conversion for CUDA boundaries. Errors if `n ≥ 2^32`. -/
def u32 (n : Nat) : Result UInt32 :=
  AnyBuffer.natToU32Checked n

/-- Number of elements in a runtime shape, checked for the `UInt32` CUDA ABI. -/
def numelU32 (s : Shape) : Result UInt32 :=
  AnyBuffer.numelU32 s

/--
Fold all leading axes into a row count and keep the last axis as the column count.

CUDA softmax/log-softmax kernels are 2D row kernels. This helper gives the shared convention used
for vectors, matrices, and higher-rank tensors: softmax is always along the last axis.
-/
def foldRowsColsLastAxis (s : Shape) : Result (UInt32 × UInt32) := do
  match s.toList.reverse with
  | [] =>
      throw "autograd: softmax: scalar input is not supported"
  | cols :: restRev =>
      let rowsFold : Nat := restRev.foldl (init := 1) (fun acc d => acc * d)
      if cols = 0 then
        throw "autograd: softmax: last dimension is 0"
      else if rowsFold = 0 then
        throw "autograd: softmax: folded leading dimension is 0"
      else
        pure (← u32 rowsFold, ← u32 cols)

/-- Broadcast a scalar CUDA buffer to `outShape`. Used by scalar reductions during backprop. -/
def broadcastScalarToShape (g : Buffer) (outShape : Shape) : Buffer :=
  let outDims := outShape.toArray
  let axisMap := Array.replicate outDims.size 0
  Buffer.broadcastTo g #[] outDims axisMap

/-- Logistic sigmoid implemented from primitive CUDA elementwise ops. -/
def sigmoidBuf (x : Buffer) (n : UInt32) : Buffer :=
  let ones := Buffer.full n 1.0
  let negx := Buffer.scale x (-1.0)
  let ex := Buffer.exp negx
  let denom := Buffer.add ones ex
  let y := Buffer.div ones denom
  Buffer.releaseThen ones <| Buffer.releaseThen negx <|
    Buffer.releaseThen ex <| Buffer.releaseThen denom y

/-- Hyperbolic tangent implemented as `(exp(2x)-1)/(exp(2x)+1)`. -/
def tanhBuf (x : Buffer) (n : UInt32) : Buffer :=
  let ones := Buffer.full n 1.0
  let twoX := Buffer.scale x 2.0
  let e2x := Buffer.exp twoX
  let num := Buffer.sub e2x ones
  let den := Buffer.add e2x ones
  let y := Buffer.div num den
  Buffer.releaseThen ones <| Buffer.releaseThen twoX <| Buffer.releaseThen e2x <|
    Buffer.releaseThen num <| Buffer.releaseThen den y

/-- Numerically stable softplus: `max(x,0) + log(1 + exp(-abs(x)))`. -/
def softplusBuf (x : Buffer) (n : UInt32) : Buffer :=
  let zeros := Buffer.full n 0.0
  let ones := Buffer.full n 1.0
  let max0 := Buffer.max x zeros
  let absx := Buffer.abs x
  let negAbs := Buffer.scale absx (-1.0)
  let expNegAbs := Buffer.exp negAbs
  let onePlusExp := Buffer.add ones expNegAbs
  let logTerm := Buffer.log onePlusExp
  let y := Buffer.add max0 logTerm
  Buffer.releaseThen zeros <| Buffer.releaseThen ones <| Buffer.releaseThen max0 <|
    Buffer.releaseThen absx <| Buffer.releaseThen negAbs <| Buffer.releaseThen expNegAbs <|
      Buffer.releaseThen onePlusExp <| Buffer.releaseThen logTerm y

/--
Row-wise stable softmax.

The returned `WithWorkspace` records the buffers used to compute the stable formula. The tape keeps
those buffers only as long as the node may need them for backprop, then releases them explicitly.
-/
def rowSoftmaxForward (x : Buffer) (rows cols : UInt32) : Buffer.WithWorkspace :=
  let rowMax := Buffer.reduceMaxAxis1 x rows cols
  let maxB := Buffer.broadcastVecToCols rowMax rows cols
  let shifted := Buffer.sub x maxB
  let ex := Buffer.exp shifted
  let rowSum := Buffer.reduceSumAxis1 ex rows cols
  let sumB := Buffer.broadcastVecToCols rowSum rows cols
  let y := Buffer.div ex sumB
  { value := y, workspace := [rowMax, maxB, shifted, ex, rowSum, sumB] }

private def rowSoftmaxFwd (x : Buffer) (rows cols : UInt32) : Buffer :=
  (rowSoftmaxForward x rows cols).value

/--
Row-wise hard-masked softmax.

`mask` is a `{0,1}` buffer with the same `(rows, cols)` shape as `x`; blocked entries contribute
literal zero numerator. This matches `Spec.hardMaskedSoftmaxSpec`, not a finite additive sentinel.
-/
def rowHardMaskedSoftmaxForward (x mask : Buffer) (rows cols : UInt32) : Buffer.WithWorkspace :=
  let rowMax := Buffer.reduceMaxAxis1 x rows cols
  let maxB := Buffer.broadcastVecToCols rowMax rows cols
  let shifted := Buffer.sub x maxB
  let ex := Buffer.exp shifted
  let exMasked := Buffer.mul ex mask
  let rowSum := Buffer.reduceSumAxis1 exMasked rows cols
  let sumB := Buffer.broadcastVecToCols rowSum rows cols
  let y := Buffer.div exMasked sumB
  { value := y, workspace := [rowMax, maxB, shifted, ex, exMasked, rowSum, sumB] }

/-- Row-wise softmax VJP: `dX = y * (dY - sum(dY*y, axis=1))`. -/
def rowSoftmaxBwd (y dLdy : Buffer) (rows cols : UInt32) : Buffer :=
  -- JVP/VJP: dX = y * (dY - sum(dY*y, axis=1)).
  let dy_y := Buffer.mul dLdy y
  let dot := Buffer.reduceSumAxis1 dy_y rows cols
  let dotB := Buffer.broadcastVecToCols dot rows cols
  let centered := Buffer.sub dLdy dotB
  Buffer.releaseThen dy_y <| Buffer.releaseThen dot <| Buffer.releaseThen dotB <|
    Buffer.releaseThen centered <| Buffer.mul y centered

/--
Row-wise stable log-softmax.

This computes `x - rowMax - log(sum(exp(x-rowMax)))` directly, avoiding the less stable
`log(softmax(x))` route. As with softmax, the returned workspace buffers belong to the tape node
until the backward pass has finished.
-/
def rowLogSoftmaxForward (x : Buffer) (rows cols : UInt32) : Buffer.WithWorkspace :=
  let rowMax := Buffer.reduceMaxAxis1 x rows cols
  let maxB := Buffer.broadcastVecToCols rowMax rows cols
  let shifted := Buffer.sub x maxB
  let ex := Buffer.exp shifted
  let rowSum := Buffer.reduceSumAxis1 ex rows cols
  let logSum := Buffer.log rowSum
  let logSumB := Buffer.broadcastVecToCols logSum rows cols
  let y := Buffer.sub shifted logSumB
  { value := y, workspace := [rowMax, maxB, shifted, ex, rowSum, logSum, logSumB] }

private def rowLogSoftmaxFwd (x : Buffer) (rows cols : UInt32) : Buffer :=
  (rowLogSoftmaxForward x rows cols).value

/-- Row-wise log-softmax VJP: `dX = dY - exp(y) * sum(dY, axis=1)`. -/
def rowLogSoftmaxBwd (y dLdy : Buffer) (rows cols : UInt32) : Buffer :=
  -- VJP: dX = dY - exp(logSoftmax(X)) * sum(dY, axis=1).
  let probs := Buffer.exp y
  let rowSum := Buffer.reduceSumAxis1 dLdy rows cols
  let sumB := Buffer.broadcastVecToCols rowSum rows cols
  let scaled := Buffer.mul probs sumB
  Buffer.releaseThen probs <| Buffer.releaseThen rowSum <| Buffer.releaseThen sumB <|
    Buffer.releaseThen scaled <| Buffer.sub dLdy scaled
end Tape

end Cuda
end Autograd
end Runtime

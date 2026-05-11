/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

CUDA tape op coverage.

This module extends `Runtime.Autograd.Cuda.Tape` with the op surface used by the Torch-style eager
front-end (mirroring the CPU tape surface in `NN.Runtime.Autograd.Engine.Core`), but specialized to
CUDA `Buffer` values (float32) plus runtime `Spec.Shape` metadata.

Implementation notes:
- We only use existing CUDA externs from:
  * `Engine/Cuda/Buffer.lean` (elementwise + basic reductions)
  * `NN.Runtime.Autograd.Engine.Cuda.Kernels` (reshape helpers, axis reductions, gather/scatter,
    bmm, transpose)
  * `Engine/Cuda/ConvPool.lean` (conv2d + pooling)
- Many higher-level ops (softmax, layer_norm, attention) are implemented as compositions of these
  primitives. This keeps the Lean surface stable while preserving a clear kernel-fusion boundary.
- Masked attention is optional; see notes in `multi_head_attention` for how to add it
  without CPU compute.
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Tape
public import NN.Runtime.Autograd.Engine.Cuda.Kernels
public import NN.Runtime.Autograd.Engine.Cuda.ConvPool
public import NN.Runtime.Autograd.Engine.Cuda.Convert
public import NN.Runtime.Autograd.Engine.Cuda.Shape
public import NN.Spec.Core.TensorReductionShape

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
  Buffer.div ones denom

/-- Hyperbolic tangent implemented as `(exp(2x)-1)/(exp(2x)+1)`. -/
def tanhBuf (x : Buffer) (n : UInt32) : Buffer :=
  let ones := Buffer.full n 1.0
  let twoX := Buffer.scale x 2.0
  let e2x := Buffer.exp twoX
  let num := Buffer.sub e2x ones
  let den := Buffer.add e2x ones
  Buffer.div num den

/-- Numerically stable softplus: `max(x,0) + log(1 + exp(-abs(x)))`. -/
def softplusBuf (x : Buffer) (n : UInt32) : Buffer :=
  -- Stable softplus: max(x,0) + log(1 + exp(-abs(x)))
  let zeros := Buffer.full n 0.0
  let ones := Buffer.full n 1.0
  let max0 := Buffer.max x zeros
  let absx := Buffer.abs x
  let negAbs := Buffer.scale absx (-1.0)
  let expNegAbs := Buffer.exp negAbs
  let logTerm := Buffer.log (Buffer.add ones expNegAbs)
  Buffer.add max0 logTerm

/--
Row-wise stable softmax plus scratch buffers that should be released after the backward pass.

The output is the first component. The scratch list records temporary buffers created while
computing the stable formula so long training runs do not wait for native finalizers.
-/
def softmax2dFwdWithCleanup (x : Buffer) (rows cols : UInt32) : Buffer × List Buffer :=
  -- Numerically stable row-wise softmax.
  let rowMax := Buffer.reduceMaxAxis1 x rows cols
  let maxB := Buffer.broadcastVecToCols rowMax rows cols
  let shifted := Buffer.sub x maxB
  let ex := Buffer.exp shifted
  let rowSum := Buffer.reduceSumAxis1 ex rows cols
  let sumB := Buffer.broadcastVecToCols rowSum rows cols
  let y := Buffer.div ex sumB
  (y, [rowMax, maxB, shifted, ex, rowSum, sumB])

def softmax2dFwd (x : Buffer) (rows cols : UInt32) : Buffer :=
  (softmax2dFwdWithCleanup x rows cols).1

/-- Row-wise softmax VJP: `dX = y * (dY - sum(dY*y, axis=1))`. -/
def softmax2dBwd (y dLdy : Buffer) (rows cols : UInt32) : Buffer :=
  -- JVP/VJP: dX = y * (dY - sum(dY*y, axis=1)).
  let dy_y := Buffer.mul dLdy y
  let dot := Buffer.reduceSumAxis1 dy_y rows cols
  let dotB := Buffer.broadcastVecToCols dot rows cols
  let centered := Buffer.sub dLdy dotB
  Buffer.mul y centered

/--
Row-wise stable log-softmax plus scratch buffers that should be released after backprop.

This computes `x - rowMax - log(sum(exp(x-rowMax)))` directly, avoiding the less stable
`log(softmax(x))` route.
-/
def logSoftmax2dFwdWithCleanup (x : Buffer) (rows cols : UInt32) : Buffer × List Buffer :=
  -- Stable row-wise log-softmax:
  --   x - rowMax - log(sum(exp(x - rowMax))).
  -- This avoids the `softmax` then `log` underflow path and matches PyTorch's intended primitive.
  let rowMax := Buffer.reduceMaxAxis1 x rows cols
  let maxB := Buffer.broadcastVecToCols rowMax rows cols
  let shifted := Buffer.sub x maxB
  let ex := Buffer.exp shifted
  let rowSum := Buffer.reduceSumAxis1 ex rows cols
  let logSum := Buffer.log rowSum
  let logSumB := Buffer.broadcastVecToCols logSum rows cols
  let y := Buffer.sub shifted logSumB
  (y, [rowMax, maxB, shifted, ex, rowSum, logSum, logSumB])

def logSoftmax2dFwd (x : Buffer) (rows cols : UInt32) : Buffer :=
  (logSoftmax2dFwdWithCleanup x rows cols).1

/-- Row-wise log-softmax VJP: `dX = dY - exp(y) * sum(dY, axis=1)`. -/
def logSoftmax2dBwd (y dLdy : Buffer) (rows cols : UInt32) : Buffer :=
  -- VJP: dX = dY - exp(logSoftmax(X)) * sum(dY, axis=1).
  let probs := Buffer.exp y
  let rowSum := Buffer.reduceSumAxis1 dLdy rows cols
  let sumB := Buffer.broadcastVecToCols rowSum rows cols
  Buffer.sub dLdy (Buffer.mul probs sumB)

/-!
## Elementwise ops
-/

/-- Elementwise addition. -/
def add {s : Shape} (t : Tape) (aId bId : Nat) : Result (Tape × Nat) :=
  binary (t := t) "add" aId bId s s s
    (forward := Buffer.add)
    (backward := fun _a _b dLdy =>
      let da := Buffer.copy dLdy
      let zeros := Buffer.zeros (Buffer.size dLdy)
      let dbRaw := Buffer.axpy zeros dLdy 1.0
      let db := Buffer.releaseThen zeros dbRaw
      (da, db))

/-- Elementwise subtraction. -/
def sub {s : Shape} (t : Tape) (aId bId : Nat) : Result (Tape × Nat) :=
  binary (t := t) "sub" aId bId s s s
    (forward := Buffer.sub)
    (backward := fun _a _b dLdy => (Buffer.copy dLdy, Buffer.scale dLdy (-1.0)))

/-- Elementwise multiplication. -/
def mul {s : Shape} (t : Tape) (aId bId : Nat) : Result (Tape × Nat) :=
  binary (t := t) "mul" aId bId s s s
    (forward := Buffer.mul)
    (backward := fun a b dLdy => (Buffer.mul dLdy b, Buffer.mul dLdy a))

/-- Multiply by a scalar constant. -/
def scale {s : Shape} (t : Tape) (xId : Nat) (c : Float) : Result (Tape × Nat) :=
  unary (t := t) "scale" xId s s
    (forward := fun x => Buffer.scale x c)
    (backward := fun _x dLdy => Buffer.scale dLdy c)

/-- Elementwise abs. -/
def abs {s : Shape} (t : Tape) (xId : Nat) : Result (Tape × Nat) :=
  unary (t := t) "abs" xId s s
    (forward := Buffer.abs)
    (backward := fun x dLdy => Buffer.absBwd x dLdy)

/-- Elementwise sqrt. -/
def sqrt {s : Shape} (t : Tape) (xId : Nat) : Result (Tape × Nat) :=
  unary (t := t) "sqrt" xId s s
    (forward := Buffer.sqrt)
    (backward := fun x dLdy => Buffer.sqrtBwd x dLdy)

/-- Clamp each element to `[lo, hi]`. -/
def clamp {s : Shape} (t : Tape) (xId : Nat) (lo hi : Float) : Result (Tape × Nat) :=
  unary (t := t) "clamp" xId s s
    (forward := fun x => Buffer.clamp x lo hi)
    (backward := fun x dLdy => Buffer.clampBwd x dLdy lo hi)

/-- Elementwise max. -/
def max {s : Shape} (t : Tape) (aId bId : Nat) : Result (Tape × Nat) :=
  binary (t := t) "max" aId bId s s s
    (forward := Buffer.max)
    (backward := fun a b dLdy => Buffer.maxBwd a b dLdy)

/-- Elementwise min. -/
def min {s : Shape} (t : Tape) (aId bId : Nat) : Result (Tape × Nat) :=
  binary (t := t) "min" aId bId s s s
    (forward := Buffer.min)
    (backward := fun a b dLdy => Buffer.minBwd a b dLdy)

/-- Elementwise division. -/
def div {s : Shape} (t : Tape) (aId bId : Nat) : Result (Tape × Nat) :=
  binary (t := t) "div" aId bId s s s
    (forward := Buffer.div)
    (backward := fun a b dLdy =>
      let da := Buffer.div dLdy b
      let b2 := Buffer.mul b b
      let aOverB2 := Buffer.div a b2
      let db := Buffer.scale (Buffer.mul dLdy aOverB2) (-1.0)
      (da, db))

/-- Elementwise ReLU. -/
def relu {s : Shape} (t : Tape) (xId : Nat) : Result (Tape × Nat) :=
  unary (t := t) "relu" xId s s
    (forward := Buffer.relu)
    (backward := fun x dLdy => Buffer.reluBwd x dLdy)

/-- Elementwise exp. -/
def exp {s : Shape} (t : Tape) (xId : Nat) : Result (Tape × Nat) :=
  unary (t := t) "exp" xId s s
    (forward := Buffer.exp)
    (backward := fun x dLdy => Buffer.mul dLdy (Buffer.exp x))

/-- Elementwise log. -/
def log {s : Shape} (t : Tape) (xId : Nat) : Result (Tape × Nat) :=
  unary (t := t) "log" xId s s
    (forward := Buffer.log)
    (backward := fun x dLdy => Buffer.mul dLdy (Buffer.inv x))

/-- Elementwise reciprocal `1/x`. -/
def inv {s : Shape} (t : Tape) (xId : Nat) : Result (Tape × Nat) :=
  unary (t := t) "inv" xId s s
    (forward := Buffer.inv)
    (backward := fun x dLdy =>
      let invx := Buffer.inv x
      let invx2 := Buffer.mul invx invx
      Buffer.scale (Buffer.mul dLdy invx2) (-1.0))

/--
Elementwise "safe log" that protects against `log(0)` by adding a small `ε` internally.

Spec semantics: `log(softplus(x) + ε)`.
-/
def safeLog {s : Shape} (t : Tape) (xId : Nat) (ε : Float) : Result (Tape × Nat) := do
  let n ← numelU32 s
  unary (t := t) "safe_log" xId s s
    (forward := fun x =>
      let epsBuf := Buffer.full n ε
      let sp := softplusBuf x n
      Buffer.log (Buffer.add sp epsBuf))
    (backward := fun x dLdy =>
      let epsBuf := Buffer.full n ε
      let sp := softplusBuf x n
      let denom := Buffer.add sp epsBuf
      let sig := sigmoidBuf x n
      let dlog := Buffer.div sig denom
      Buffer.mul dLdy dlog)

/-- Elementwise sigmoid (logistic). -/
def sigmoid {s : Shape} (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  let n ← numelU32 s
  unary (t := t) "sigmoid" xId s s
    (forward := fun x => sigmoidBuf x n)
    (backward := fun x dLdy =>
      let y := sigmoidBuf x n
      let ones := Buffer.full n 1.0
      let dy := Buffer.mul y (Buffer.sub ones y)
      Buffer.mul dLdy dy)

/-- Elementwise tanh. -/
def tanh {s : Shape} (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  let n ← numelU32 s
  unary (t := t) "tanh" xId s s
    (forward := fun x => tanhBuf x n)
    (backward := fun x dLdy =>
      let y := tanhBuf x n
      let ones := Buffer.full n 1.0
      let dy := Buffer.sub ones (Buffer.mul y y)
      Buffer.mul dLdy dy)

/-- Elementwise softplus. -/
def softplus {s : Shape} (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  let n ← numelU32 s
  unary (t := t) "softplus" xId s s
    (forward := fun x => softplusBuf x n)
    (backward := fun x dLdy =>
      let dy := sigmoidBuf x n
      Buffer.mul dLdy dy)

/-!
## Reductions / views
-/

/-- Reduce-sum of all entries, producing a scalar. -/
def sum {s : Shape} (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  let x ← requireValue (t := t) xId s
  let y := Buffer.reduceSum x
  let node : Node :=
    { name := some "sum"
      value := { s := Shape.scalar, buf := y }
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny Shape.scalar
        let dx := broadcastScalarToShape dLdy.buf s
        pure [(xId, { s := s, buf := dx })] }
  pure (t.addNode node)

/-- Flatten `s` into a 1D vector of length `Shape.size s`. -/
def flatten {s : Shape} (t : Tape) (xId : Nat) : Result (Tape × Nat) :=
  unary (t := t) "flatten" xId s (.dim (Shape.size s) .scalar)
    (forward := fun x => x)
    (backward := fun _x dLdy => Buffer.copy dLdy)

/--
Reshape a buffer while preserving number of elements.

This is a no-copy view operation: it reuses the same contiguous buffer.
-/
def reshape {s₁ s₂ : Shape} (t : Tape) (xId : Nat) (_h : Shape.size s₁ = Shape.size s₂) :
    Result (Tape × Nat) :=
  unary (t := t) "reshape" xId s₁ s₂
    (forward := fun x => x)
    (backward := fun _x dLdy => Buffer.copy dLdy)

/-- Transpose a 2D buffer. -/
def transpose2d {m n : Nat} (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  let m32 ← u32 m
  let n32 ← u32 n
  unary (t := t) "transpose2d" xId (.dim m (.dim n .scalar)) (.dim n (.dim m .scalar))
    (forward := fun x => Buffer.transpose2d x m32 n32)
    (backward := fun _x dLdy => Buffer.transpose2d dLdy n32 m32)

/--
Swap adjacent axes at a given depth in an N-D buffer.

If `depth` is out of range, this is treated as the identity (matches the spec-layer helper).
-/
def swapAdjacentAtDepth {s : Shape} (t : Tape) (depth : Nat) (xId : Nat) : Result (Tape × Nat) := do
  let depth32 ← u32 depth
  let dimsIn : Array Nat := Shape.toArray s
  let outShape : Shape := s.swapAdjacentAtDepth depth
  let dimsOut : Array Nat := Shape.toArray outShape
  let validDepth := depth + 1 < Shape.rank s
  unary (t := t) "swapAdjacentAtDepth" xId s outShape
    (forward := fun x =>
      if validDepth then
        Buffer.swapAdjacentAtDepth x dimsIn depth32
      else
        x)
    (backward := fun _x dLdy =>
      if validDepth then
        Buffer.swapAdjacentAtDepth dLdy dimsOut depth32
      else
        dLdy)

/-- Permute a 3D tensor `(a,b,c) → (b,c,a)`. -/
def transpose3dFirstToLast {a b c : Nat} (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  let s₀ : Shape := .dim a (.dim b (.dim c .scalar))
  let (t1, id1) ← swapAdjacentAtDepth (t := t) (s := s₀) 0 xId
  let s₁ : Shape := .dim b (.dim a (.dim c .scalar))
  swapAdjacentAtDepth (t := t1) (s := s₁) 1 id1

/-- Permute a 3D tensor `(a,b,c) → (c,a,b)`. -/
def transpose3dLastToFirst {a b c : Nat} (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  let s₀ : Shape := .dim a (.dim b (.dim c .scalar))
  let (t1, id1) ← swapAdjacentAtDepth (t := t) (s := s₀) 1 xId
  let s₁ : Shape := .dim a (.dim c (.dim b .scalar))
  swapAdjacentAtDepth (t := t1) (s := s₁) 0 id1

/-- Swap the last two axes of a 3D tensor `(a,b,c) → (a,c,b)`. -/
def transpose3dLastTwo {a b c : Nat} (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  let s : Shape := .dim a (.dim b (.dim c .scalar))
  swapAdjacentAtDepth (t := t) (s := s) 1 xId

/--
Broadcast `x : s₁` to `s₂`.

Forward: `broadcastTo`.
Backward: sum-reduce broadcasted axes (`reduceFromBroadcastTo`).
-/
def broadcastTo {s₁ s₂ : Shape} (t : Tape) (cb : Shape.CanBroadcastTo s₁ s₂) (xId : Nat) :
    Result (Tape × Nat) := do
  let inDims := Shape.toArray s₁
  let outDims := Shape.toArray s₂
  let axisMap := Broadcast.axisMap cb
  unary (t := t) "broadcastTo" xId s₁ s₂
    (forward := fun x => Buffer.broadcastTo x inDims outDims axisMap)
    (backward := fun _x dLdy => Buffer.reduceFromBroadcastTo dLdy inDims outDims axisMap)

/-- Reduce-sum along `axis`. -/
def reduceSum {s : Shape} (axis : Nat) [valid : Shape.valid_axis_inst axis s] [wf : Shape.WellFormed s]
    (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  let axis32 ← u32 axis
  let dims : Array Nat := Shape.toArray s
  let outShape : Shape := shapeAfterSum s axis
  unary (t := t) s!"reduce_sum(axis={axis})" xId s outShape
    (forward := fun x => Buffer.reduceSumAxis x dims axis32)
    (backward := fun _x dLdy =>
      let cb := shapeAfterSumBroadcastBack (s := s) axis valid wf
      let (inDims, outDims, axisMap) := Broadcast.broadcastArgs cb
      Buffer.broadcastTo dLdy inDims outDims axisMap)

/-- Reduce-mean along `axis`. -/
def reduceMean {s : Shape} (axis : Nat) [valid : Shape.valid_axis_inst axis s] [wf : Shape.WellFormed s]
    (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  let axis32 ← u32 axis
  let dims : Array Nat := Shape.toArray s
  let outShape : Shape := shapeAfterSum s axis
  unary (t := t) s!"reduce_mean(axis={axis})" xId s outShape
    (forward := fun x =>
      let sum := Buffer.reduceSumAxis x dims axis32
      let denomNat :=
        match getDimSize s axis with
        | some n => n
        | none => 1
      Buffer.scale sum (1.0 / (Float.ofNat denomNat)))
    (backward := fun _x dLdy =>
      let cb := shapeAfterSumBroadcastBack (s := s) axis valid wf
      let (inDims, outDims, axisMap) := Broadcast.broadcastArgs cb
      let dLdx := Buffer.broadcastTo dLdy inDims outDims axisMap
      let denomNat :=
        match getDimSize s axis with
        | some n => n
        | none => 1
      Buffer.scale dLdx (1.0 / (Float.ofNat denomNat)))

/-!
## Linear algebra
-/

/-- 2D matrix multiply. -/
def matmul {m n p : Nat} (t : Tape) (aId bId : Nat) : Result (Tape × Nat) := do
  let m32 ← u32 m
  let n32 ← u32 n
  let p32 ← u32 p
  let one32 : UInt32 := 1
  let σ₁ : Shape := .dim m (.dim n .scalar)
  let σ₂ : Shape := .dim n (.dim p .scalar)
  let τ : Shape := .dim m (.dim p .scalar)
  binary (t := t) "matmul" aId bId σ₁ σ₂ τ
    (forward := fun a b => Buffer.bmm a b one32 m32 n32 p32)
    (backward := fun a b dLdy =>
      let bT := Buffer.transpose2d b n32 p32
      let aT := Buffer.transpose2d a m32 n32
      let dA := Buffer.bmm dLdy bT one32 m32 p32 n32
      let dB := Buffer.bmm aT dLdy one32 n32 m32 p32
      (dA, dB))

/-- Batched matrix multiply. -/
def bmm {batch m n p : Nat} (t : Tape) (aId bId : Nat) : Result (Tape × Nat) := do
  let b32 ← u32 batch
  let m32 ← u32 m
  let n32 ← u32 n
  let p32 ← u32 p
  let σ₁ : Shape := .dim batch (.dim m (.dim n .scalar))
  let σ₂ : Shape := .dim batch (.dim n (.dim p .scalar))
  let τ : Shape := .dim batch (.dim m (.dim p .scalar))
  let dimsA : Array Nat := #[batch, m, n]
  let dimsB : Array Nat := #[batch, n, p]
  binary (t := t) "bmm" aId bId σ₁ σ₂ τ
    (forward := fun a b => Buffer.bmm a b b32 m32 n32 p32)
    (backward := fun a b dLdy =>
      let depthLast : UInt32 := 1
      let bT := Buffer.swapAdjacentAtDepth b dimsB depthLast   -- (batch,p,n)
      let aT := Buffer.swapAdjacentAtDepth a dimsA depthLast   -- (batch,n,m)
      let dA := Buffer.bmm dLdy bT b32 m32 p32 n32
      -- dB = aᵀ @ dLdy  (batch,n,m) * (batch,m,p) -> (batch,n,p)
      let dB := Buffer.bmm aT dLdy b32 n32 m32 p32
      (dA, dB))

/--
Fused real-FFT spectral convolution used by the CUDA FNO1D path.

Shapes:
- `x : (grid, width)`,
- `wRe, wIm : (modes, width, width)`,
- output `y : (grid, width)`.

The low-level buffer primitive owns the numerical contract and VJP:
`rfft(x)` is unnormalized, the inverse is normalized, and the backward kernels include the
half-spectrum adjoint factors for real FFTs. This tape node simply records those three parent
dependencies and checks the runtime shapes before calling the native kernels.
-/
def spectralConv1dRfft {grid width modes : Nat}
    (t : Tape) (xId wReId wImId : Nat) : Result (Tape × Nat) := do
  if grid = 0 then
    throw "autograd: spectralConv1dRfft: grid must be positive"
  if width = 0 then
    throw "autograd: spectralConv1dRfft: width must be positive"
  if modes > grid / 2 + 1 then
    throw "autograd: spectralConv1dRfft: modes exceeds rfft frequency count"
  let grid32 ← u32 grid
  let width32 ← u32 width
  let modes32 ← u32 modes
  let xShape : Shape := .dim grid (.dim width .scalar)
  let wShape : Shape := .dim modes (.dim width (.dim width .scalar))
  let x ← requireValue (t := t) xId xShape
  let wRe ← requireValue (t := t) wReId wShape
  let wIm ← requireValue (t := t) wImId wShape
  let y := Buffer.spectralConv1dRfftFwd x wRe wIm grid32 width32 modes32
  let node : Node :=
    { name := some "spectralConv1dRfft"
      value := { s := xShape, buf := y }
      requires_grad := true
      parents := [xId, wReId, wImId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny xShape
        let dx := Buffer.spectralConv1dRfftBwdX x wRe wIm dLdy.buf grid32 width32 modes32
        let dWRe := Buffer.spectralConv1dRfftBwdWRe x wRe wIm dLdy.buf grid32 width32 modes32
        let dWIm := Buffer.spectralConv1dRfftBwdWIm x wRe wIm dLdy.buf grid32 width32 modes32
        pure
          [ (xId, { s := xShape, buf := dx })
          , (wReId, { s := wShape, buf := dWRe })
          , (wImId, { s := wShape, buf := dWIm }) ] }
  pure (t.addNode node)

/-!
## Linear layer / losses
-/

/-- Linear layer: `y = W·x + b` with `W : (outDim,inDim)`, `x : inDim`, `b : outDim`. -/
def linear {outDim inDim : Nat} (t : Tape) (wId bId xId : Nat) : Result (Tape × Nat) := do
  let out32 ← u32 outDim
  let in32 ← u32 inDim
  let one32 : UInt32 := 1
  let wBuf ← requireValue (t := t) wId (.dim outDim (.dim inDim .scalar))
  let bBuf ← requireValue (t := t) bId (.dim outDim .scalar)
  let xBuf ← requireValue (t := t) xId (.dim inDim .scalar)
  let wx := Buffer.bmm wBuf xBuf one32 out32 in32 one32
  let yBuf := Buffer.add wx bBuf
  let node : Node :=
    { name := some "linear"
      value := { s := .dim outDim .scalar, buf := yBuf }
      requires_grad := true
      parents := [wId, bId, xId]
      cleanup := [wx]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny (.dim outDim .scalar)
        let g := dLdy.buf
        let dW := Buffer.bmm g xBuf one32 out32 one32 in32
        let db := Buffer.copy g
        let wT := Buffer.transpose2d wBuf out32 in32
        let dx := Buffer.bmm wT g one32 in32 out32 one32
        pure
          [ (wId, { s := .dim outDim (.dim inDim .scalar), buf := dW })
          , (bId, { s := .dim outDim .scalar, buf := db })
          , (xId, { s := .dim inDim .scalar, buf := dx }) ] }
  pure (t.addNode node)

/-- Mean-squared-error loss with `"mean"` reduction (single scalar output). -/
def mseLoss {s : Shape} (t : Tape) (yhatId targetId : Nat) : Result (Tape × Nat) := do
  let yhat ← requireValue (t := t) yhatId s
  let target ← requireValue (t := t) targetId s
  let diff := Buffer.sub yhat target
  let squared := Buffer.mul diff diff
  let sum := Buffer.reduceSum squared
  let denom : Float := Float.ofNat (Shape.size s)
  let mean := Buffer.scale sum (1.0 / denom)
  let node : Node :=
    { name := some "mse_loss"
      value := { s := Shape.scalar, buf := mean }
      requires_grad := true
      parents := [yhatId, targetId]
      cleanup := [diff, squared, sum]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny Shape.scalar
        let gBroad := broadcastScalarToShape dLdy.buf s
        let scaleConst : Float := 2.0 / denom
        let diffGrad := Buffer.mul diff gBroad
        let dYhat := Buffer.scale diffGrad scaleConst
        let dTarget := Buffer.releaseThen gBroad <|
          Buffer.releaseThen diffGrad <|
            Buffer.scale dYhat (-1.0)
        pure [
          (yhatId, { s := s, buf := dYhat }),
          (targetId, { s := s, buf := dTarget })
        ] }
  pure (t.addNode node)

/-!
## Concat / slice (1D)
-/

/-- Concatenate two 1D buffers. -/
def concat1d {n m : Nat} (t : Tape) (aId bId : Nat) : Result (Tape × Nat) := do
  let n32 ← u32 n
  let m32 ← u32 m
  let σ₁ : Shape := .dim n .scalar
  let σ₂ : Shape := .dim m .scalar
  let τ : Shape := .dim (n + m) .scalar
  let nm32 ← u32 (n + m)
  let startN32 ← u32 n
  binary (t := t) "concat1d" aId bId σ₁ σ₂ τ
    (forward := fun a b => Buffer.concat1d a b n32 m32)
    (backward := fun _a _b dLdy =>
      let dA := Buffer.slice1d dLdy nm32 0 n32
      let dB := Buffer.slice1d dLdy nm32 startN32 m32
      (dA, dB))

/-- Concatenate two 1D tensors (CPU tape name). -/
def concatVectors {n m : Nat} (t : Tape) (aId bId : Nat) : Result (Tape × Nat) := do
  let n32 ← u32 n
  let m32 ← u32 m
  let nm32 ← u32 (n + m)
  let startN32 ← u32 n
  binary (t := t) "concat_vectors" aId bId (.dim n .scalar) (.dim m .scalar) (.dim (n + m) .scalar)
    (forward := fun a b => Buffer.concat1d a b n32 m32)
    (backward := fun _a _b dLdy =>
      let dA := Buffer.slice1d dLdy nm32 0 n32
      let dB := Buffer.slice1d dLdy nm32 startN32 m32
      (dA, dB))

/-- Slice a 1D buffer. -/
def slice1d {n start len : Nat} (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  if start + len ≤ n then
    let n32 ← u32 n
    let start32 ← u32 start
    let len32 ← u32 len
    let outShape : Shape := .dim len .scalar
    let x ← requireValue (t := t) xId (.dim n .scalar)
    let y := Buffer.slice1d x n32 start32 len32
    let node : Node :=
      { name := some s!"slice1d[{start}:{start+len}]"
        value := { s := outShape, buf := y }
        requires_grad := true
        parents := [xId]
        backward := fun dLdyAny => do
          let dLdy ← requireGrad dLdyAny outShape
          let pre := Buffer.zeros start32
          let postLen : Nat := n - start - len
          let post32 ← u32 postLen
          let post := Buffer.zeros post32
          let tmp := Buffer.concat1d pre dLdy.buf start32 len32
          let startLen32 ← u32 (start + len)
          let dx := Buffer.releaseThen pre <|
            Buffer.releaseThen post <|
              Buffer.releaseThen tmp <|
                Buffer.concat1d tmp post startLen32 post32
          pure [(xId, { s := .dim n .scalar, buf := dx })] }
    pure (t.addNode node)
  else
    throw "autograd: slice1d: start+len out of bounds"

/-!
## Concat / slice along dim 0
-/

/-- Concatenate along dim 0 for tensors with leading dimension (CPU tape name). -/
def concatDim0 {n m : Nat} {s : Shape} (t : Tape) (aId bId : Nat) : Result (Tape × Nat) := do
  let inner : Nat := Shape.size s
  let nLen : Nat := n * inner
  let mLen : Nat := m * inner
  let nLen32 ← u32 nLen
  let mLen32 ← u32 mLen
  let nmLen32 ← u32 (nLen + mLen)
  binary (t := t) "concat_dim0" aId bId (.dim n s) (.dim m s) (.dim (n + m) s)
    (forward := fun a b => Buffer.concat1d a b nLen32 mLen32)
    (backward := fun _a _b dLdy =>
      let dA := Buffer.slice1d dLdy nmLen32 0 nLen32
      let dB := Buffer.slice1d dLdy nmLen32 nLen32 mLen32
      (dA, dB))

/-- Slice along dim 0: `x[start:start+len]` (CPU tape name). -/
def sliceRange0 {n : Nat} {s : Shape} (t : Tape) (xId : Nat) (start len : Nat)
    (_h : len + start ≤ n) : Result (Tape × Nat) := do
  let inner : Nat := Shape.size s
  let nTot : Nat := n * inner
  let startOff : Nat := start * inner
  let lenTot : Nat := len * inner
  let rightTot : Nat := nTot - (startOff + lenTot)
  let nTot32 ← u32 nTot
  let start32 ← u32 startOff
  let len32 ← u32 lenTot
  let right32 ← u32 rightTot
  let midLen32 ← u32 (startOff + lenTot)
  unary (t := t) "slice_range0" xId (.dim n s) (.dim len s)
    (forward := fun x => Buffer.slice1d x nTot32 start32 len32)
    (backward := fun _x dLdy =>
      let left := Buffer.zeros start32
      let right := Buffer.zeros right32
      let tmp := Buffer.concat1d left dLdy start32 len32
      Buffer.releaseThen left <|
        Buffer.releaseThen right <|
          Buffer.releaseThen tmp <|
            Buffer.concat1d tmp right midLen32 right32)

/-!
## Gather / scatter (host Nat indices)

Indices are non-differentiable and remain on the host. Kernels totalize out-of-bounds indices as
documented in `NN.Runtime.Autograd.Engine.Cuda.Kernels`.
-/

/-- Gather a scalar from a 1D vector using a compile-time index. -/
def gatherScalar {n : Nat} (t : Tape) (xId : Nat) (i : Fin n) : Result (Tape × Nat) := do
  let n32 ← u32 n
  let one32 : UInt32 := 1
  let indices : Array Nat := #[i.val]
  let x ← requireValue (t := t) xId (.dim n .scalar)
  let y := Buffer.gatherVec x n32 indices one32
  let node : Node :=
    { name := some s!"gather_scalar[{i.val}]"
      value := { s := Shape.scalar, buf := y }
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny Shape.scalar
        let zeros := Buffer.zeros n32
        let dx := Buffer.releaseThen zeros <|
          Buffer.scatterAdd zeros dLdy.buf n32 indices one32
        pure [(xId, { s := .dim n .scalar, buf := dx })] }
  pure (t.addNode node)

/-- Gather a row from a 2D matrix using a compile-time index. -/
def gatherRow {rows cols : Nat} (t : Tape) (xId : Nat) (i : Fin rows) : Result (Tape × Nat) := do
  let rows32 ← u32 rows
  let cols32 ← u32 cols
  let one32 : UInt32 := 1
  let i32 ← u32 i.val
  let indices : Array Nat := #[i.val]
  let x ← requireValue (t := t) xId (.dim rows (.dim cols .scalar))
  let y := Buffer.gatherRows x rows32 cols32 indices one32
  let node : Node :=
    { name := some s!"gather_row[{i.val}]"
      value := { s := .dim cols .scalar, buf := y }
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny (.dim cols .scalar)
        let zerosLen ← u32 (rows * cols)
        let zeros := Buffer.zeros zerosLen
        let dx := Buffer.releaseThen zeros <|
          Buffer.scatterAddRow zeros dLdy.buf rows32 cols32 i32
        pure [(xId, { s := .dim rows (.dim cols .scalar), buf := dx })] }
  pure (t.addNode node)

/-- Gather a scalar from a 1D vector using a runtime `Nat` index (totalized by the kernel). -/
def gatherScalarNat {n : Nat} (t : Tape) (xId : Nat) (i : Nat) : Result (Tape × Nat) := do
  let n32 ← u32 n
  let one32 : UInt32 := 1
  let indices : Array Nat := #[i]
  let x ← requireValue (t := t) xId (.dim n .scalar)
  let y := Buffer.gatherVec x n32 indices one32
  let node : Node :=
    { name := some s!"gather_scalar_nat[{i}]"
      value := { s := Shape.scalar, buf := y }
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny Shape.scalar
        let zeros := Buffer.zeros n32
        let dx := Buffer.releaseThen zeros <|
          Buffer.scatterAdd zeros dLdy.buf n32 indices one32
        pure [(xId, { s := .dim n .scalar, buf := dx })] }
  pure (t.addNode node)

def natTensorToIndexArray {k : Nat} (idx : Tensor Nat (.dim k .scalar)) : Array Nat :=
  match idx with
  | .dim f =>
      Array.ofFn (fun i : Fin k =>
        match f i with
        | .scalar n => n)

/-- Gather `k` scalars from a length-`n` vector. -/
def gatherVecNat {n k : Nat} (t : Tape) (xId : Nat) (idx : Tensor Nat (.dim k .scalar)) :
    Result (Tape × Nat) := do
  let n32 ← u32 n
  let k32 ← u32 k
  let indices := natTensorToIndexArray (k := k) idx
  let x ← requireValue (t := t) xId (.dim n .scalar)
  let y := Buffer.gatherVec x n32 indices k32
  let node : Node :=
    { name := some "gather_vec_nat"
      value := { s := .dim k .scalar, buf := y }
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny (.dim k .scalar)
        -- Scatter-add the gathered gradient back into the length-`n` input.
        let zeros := Buffer.zeros n32
        let dx := Buffer.releaseThen zeros <|
          Buffer.scatterAdd zeros dLdy.buf n32 indices k32
        pure [(xId, { s := .dim n .scalar, buf := dx })] }
  pure (t.addNode node)

/-- Gather `k` rows from a `(rows, cols)` matrix (row-major). -/
def gatherRowsNat {rows cols k : Nat} (t : Tape) (xId : Nat)
    (idx : Tensor Nat (.dim k .scalar)) :
    Result (Tape × Nat) := do
  let rows32 ← u32 rows
  let cols32 ← u32 cols
  let k32 ← u32 k
  let indices := natTensorToIndexArray (k := k) idx
  let x ← requireValue (t := t) xId (.dim rows (.dim cols .scalar))
  let y := Buffer.gatherRows x rows32 cols32 indices k32
  let node : Node :=
    { name := some "gather_rows_nat"
      value := { s := .dim k (.dim cols .scalar), buf := y }
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny (.dim k (.dim cols .scalar))
        -- Scatter-add the gathered row gradients back into the `(rows, cols)` input.
        let zerosLen ← u32 (rows * cols)
        let zeros := Buffer.zeros zerosLen
        let dx := Buffer.releaseThen zeros <|
          Buffer.scatterAddRows zeros dLdy.buf rows32 cols32 indices k32
        pure [(xId, { s := .dim rows (.dim cols .scalar), buf := dx })] }
  pure (t.addNode node)

/-- Scatter-add into a vector: `out = x` with `out[i] += v`. -/
def scatterAddVec {n : Nat} (t : Tape) (xId vId : Nat) (i : Fin n) : Result (Tape × Nat) := do
  let n32 ← u32 n
  let one32 : UInt32 := 1
  let x ← requireValue (t := t) xId (.dim n .scalar)
  let v ← requireValue (t := t) vId Shape.scalar
  let indices : Array Nat := #[i.val]
  let y := Buffer.scatterAdd x v n32 indices one32
  let node : Node :=
    { name := some s!"scatter_add_vec[{i.val}]"
      value := { s := .dim n .scalar, buf := y }
      requires_grad := true
      parents := [xId, vId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny (.dim n .scalar)
        let dv1 := Buffer.gatherVec dLdy.buf n32 indices one32
        let dx := Buffer.copy dLdy.buf
        -- `gatherVec` returns length-1; reinterpret as scalar (same numel).
        pure [
          (xId, { s := .dim n .scalar, buf := dx }),
          (vId, { s := Shape.scalar, buf := dv1 })
        ] }
  pure (t.addNode node)

/-- Scatter-add into a matrix row: `out = x` with `out[i,:] += v`. -/
def scatterAddRow {rows cols : Nat} (t : Tape) (xId vId : Nat) (i : Fin rows) :
    Result (Tape × Nat) := do
  let rows32 ← u32 rows
  let cols32 ← u32 cols
  let one32 : UInt32 := 1
  let i32 ← u32 i.val
  let x ← requireValue (t := t) xId (.dim rows (.dim cols .scalar))
  let v ← requireValue (t := t) vId (.dim cols .scalar)
  let y := Buffer.scatterAddRow x v rows32 cols32 i32
  let node : Node :=
    { name := some s!"scatter_add_row[{i.val}]"
      value := { s := .dim rows (.dim cols .scalar), buf := y }
      requires_grad := true
      parents := [xId, vId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny (.dim rows (.dim cols .scalar))
        let indices : Array Nat := #[i.val]
        let dv1 := Buffer.gatherRows dLdy.buf rows32 cols32 indices one32
        let dx := Buffer.copy dLdy.buf
        -- `gatherRows` returns (1,cols) laid out as length `cols`; reinterpret as vector.
        pure [
          (xId, { s := .dim rows (.dim cols .scalar), buf := dx }),
          (vId, { s := .dim cols .scalar, buf := dv1 })
        ] }
  pure (t.addNode node)

/-!
## Conv2D + pooling (ConvPool FFI)
-/

/-- Conv2D forward/backward via ConvPool FFI (single image, channels-first). -/
def conv2d
  {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (t : Tape) (kernelId biasId inputId : Nat) : Result (Tape × Nat) := do
  have _ := h1
  have _ := h2
  have _ := h3
  let inC32 ← u32 inC
  let outC32 ← u32 outC
  let kH32 ← u32 kH
  let kW32 ← u32 kW
  let stride32 ← u32 stride
  let pad32 ← u32 padding
  let inH32 ← u32 inH
  let inW32 ← u32 inW
  let kernel ← requireValue (t := t) kernelId (.dim outC (.dim inC (.dim kH (.dim kW .scalar))))
  let bias ← requireValue (t := t) biasId (.dim outC .scalar)
  let input ← requireValue (t := t) inputId (.dim inC (.dim inH (.dim inW .scalar)))
  let outH : Nat := (inH + 2 * padding - kH) / stride + 1
  let outW : Nat := (inW + 2 * padding - kW) / stride + 1
  let y := torchleanConv2dFwdCuda input kernel bias inC32 inH32 inW32 outC32 kH32 kW32 stride32
    pad32
  let outShape : Shape := .dim outC (.dim outH (.dim outW .scalar))
  let node : Node :=
    { name := some "conv2d"
      value := { s := outShape, buf := y }
      requires_grad := true
      parents := [kernelId, biasId, inputId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        let (dKernel, dBias, dInput) :=
          torchleanConv2dBwdCuda input kernel dLdy.buf
            inC32 inH32 inW32 outC32 kH32 kW32 stride32 pad32
        pure [
          (kernelId, { s := .dim outC (.dim inC (.dim kH (.dim kW .scalar))), buf := dKernel }),
          (biasId, { s := .dim outC .scalar, buf := dBias }),
          (inputId, { s := .dim inC (.dim inH (.dim inW .scalar)), buf := dInput })
        ] }
  pure (t.addNode node)

/-!
### ConvTranspose2D (ConvPool FFI)
-/

/-- ConvTranspose2D forward/backward via ConvPool FFI (single image, channels-first). -/
def convTranspose2d
  {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (t : Tape) (kernelId biasId inputId : Nat) : Result (Tape × Nat) := do
  have _ := h1
  have _ := h2
  have _ := h3
  let inC32 ← u32 inC
  let outC32 ← u32 outC
  let kH32 ← u32 kH
  let kW32 ← u32 kW
  let stride32 ← u32 stride
  let pad32 ← u32 padding
  let inH32 ← u32 inH
  let inW32 ← u32 inW
  let kernel ← requireValue (t := t) kernelId (.dim inC (.dim outC (.dim kH (.dim kW .scalar))))
  let bias ← requireValue (t := t) biasId (.dim outC .scalar)
  let input ← requireValue (t := t) inputId (.dim inC (.dim inH (.dim inW .scalar)))
  let outH : Nat := (inH - 1) * stride - 2 * padding + kH
  let outW : Nat := (inW - 1) * stride - 2 * padding + kW
  let y :=
    torchleanConvTranspose2dFwdCuda input kernel bias inC32 inH32 inW32 outC32 kH32 kW32 stride32
      pad32
  let outShape : Shape := .dim outC (.dim outH (.dim outW .scalar))
  let node : Node :=
    { name := some "conv_transpose2d"
      value := { s := outShape, buf := y }
      requires_grad := true
      parents := [kernelId, biasId, inputId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        let (dKernel, dBias, dInput) :=
          torchleanConvTranspose2dBwdCuda input kernel dLdy.buf
            inC32 inH32 inW32 outC32 kH32 kW32 stride32 pad32
        pure [
          (kernelId, { s := .dim inC (.dim outC (.dim kH (.dim kW .scalar))), buf := dKernel }),
          (biasId, { s := .dim outC .scalar, buf := dBias }),
          (inputId, { s := .dim inC (.dim inH (.dim inW .scalar)), buf := dInput })
        ] }
  pure (t.addNode node)

/-!
### Generic naming wrappers

The CUDA tape exposes `conv`/`max_pool`/`avg_pool`/`smooth_max_pool` using the same names as the
CPU tape. These dispatch to the ConvPool CUDA FFI entrypoints that take per-axis parameters as
`Array Nat` (rank ≤ 8).

The `*2d*` wrappers remain as concise convenience names for the common rank-2 case.
-/

/-- N-D convolution (CUDA) via ConvPool FFI (rank ≤ 8). -/
def conv
  {d inC outC : Nat}
  {kernel stride padding : Vector Nat d}
  {inSpatial : Vector Nat d}
  (t : Tape) (kernelId biasId inputId : Nat)
  (hInC : inC ≠ 0)
  (hKernel : ∀ i : Fin d, kernel.get i ≠ 0) :
  Result (Tape × Nat) := do
  have _ := hInC
  have _ := hKernel
  if d = 0 then
    throw "autograd: cuda: conv: d=0 is not supported"
  if d > 8 then
    throw "autograd: cuda: conv: rank too large (max 8)"
  if !decide (∀ i : Fin d, stride.get i ≠ 0) then
    throw "autograd: cuda: conv: stride must be > 0"

  let inC32 ← u32 inC
  let outC32 ← u32 outC
  let inSpatialArr : Array Nat := Array.ofFn (fun i : Fin d => inSpatial.get i)
  let kernelSpatialArr : Array Nat := Array.ofFn (fun i : Fin d => kernel.get i)
  let strideArr : Array Nat := Array.ofFn (fun i : Fin d => stride.get i)
  let paddingArr : Array Nat := Array.ofFn (fun i : Fin d => padding.get i)

  let kernelShape : Shape :=
    Shape.ofList (outC :: inC :: kernel.toList)
  let inputShape : Shape :=
    Shape.ofList (inC :: inSpatial.toList)
  let outSpatial : Vector Nat d :=
    Vector.ofFn (fun a =>
      (inSpatial.get a + 2 * padding.get a - kernel.get a) / stride.get a + 1)
  let outShape : Shape :=
    Shape.ofList (outC :: outSpatial.toList)

  let kernelBuf ← requireValue (t := t) kernelId kernelShape
  let biasBuf ← requireValue (t := t) biasId (.dim outC .scalar)
  let inputBuf ← requireValue (t := t) inputId inputShape

  let y :=
    torchleanConvFwdCuda inputBuf kernelBuf biasBuf
      inSpatialArr kernelSpatialArr strideArr paddingArr
      inC32 outC32

  let node : Node :=
    { name := some "conv"
      value := { s := outShape, buf := y }
      requires_grad := true
      parents := [kernelId, biasId, inputId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        let (dKernel, dBias, dInput) :=
          torchleanConvBwdCuda inputBuf kernelBuf dLdy.buf
            inSpatialArr kernelSpatialArr strideArr paddingArr
            inC32 outC32
        pure [
          (kernelId, { s := kernelShape, buf := dKernel }),
          (biasId, { s := .dim outC .scalar, buf := dBias }),
          (inputId, { s := inputShape, buf := dInput })
        ] }
  pure (t.addNode node)

/-- N-D transposed convolution (CUDA) via ConvPool FFI (rank ≤ 8). -/
def convTranspose
  {d inC outC : Nat}
  {kernel stride padding : Vector Nat d}
  {inSpatial : Vector Nat d}
  (t : Tape) (kernelId biasId inputId : Nat)
  (hInC : inC ≠ 0)
  (hKernel : ∀ i : Fin d, kernel.get i ≠ 0) :
  Result (Tape × Nat) := do
  have _ := hInC
  have _ := hKernel
  if d = 0 then
    throw "autograd: cuda: conv_transpose: d=0 is not supported"
  if d > 8 then
    throw "autograd: cuda: conv_transpose: rank too large (max 8)"
  if !decide (∀ i : Fin d, stride.get i ≠ 0) then
    throw "autograd: cuda: conv_transpose: stride must be > 0"

  let inC32 ← u32 inC
  let outC32 ← u32 outC
  let inSpatialArr : Array Nat := Array.ofFn (fun i : Fin d => inSpatial.get i)
  let kernelSpatialArr : Array Nat := Array.ofFn (fun i : Fin d => kernel.get i)
  let strideArr : Array Nat := Array.ofFn (fun i : Fin d => stride.get i)
  let paddingArr : Array Nat := Array.ofFn (fun i : Fin d => padding.get i)

  -- NOTE: for transposed conv, kernel layout is `(inC, outC, kernelSpatial...)`.
  let kernelShape : Shape :=
    Shape.ofList (inC :: outC :: kernel.toList)
  let inputShape : Shape :=
    Shape.ofList (inC :: inSpatial.toList)
  let outSpatial : Vector Nat d :=
    Vector.ofFn (fun a =>
      (inSpatial.get a - 1) * stride.get a - 2 * padding.get a + kernel.get a)
  let outShape : Shape :=
    Shape.ofList (outC :: outSpatial.toList)

  let kernelBuf ← requireValue (t := t) kernelId kernelShape
  let biasBuf ← requireValue (t := t) biasId (.dim outC .scalar)
  let inputBuf ← requireValue (t := t) inputId inputShape

  let y :=
    torchleanConvTransposeFwdCuda inputBuf kernelBuf biasBuf
      inSpatialArr kernelSpatialArr strideArr paddingArr
      inC32 outC32

  let node : Node :=
    { name := some "conv_transpose"
      value := { s := outShape, buf := y }
      requires_grad := true
      parents := [kernelId, biasId, inputId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        let (dKernel, dBias, dInput) :=
          torchleanConvTransposeBwdCuda inputBuf kernelBuf dLdy.buf
            inSpatialArr kernelSpatialArr strideArr paddingArr
            inC32 outC32
        pure [
          (kernelId, { s := kernelShape, buf := dKernel }),
          (biasId, { s := .dim outC .scalar, buf := dBias }),
          (inputId, { s := inputShape, buf := dInput })
        ] }
  pure (t.addNode node)

/-- MaxPool2D via ConvPool FFI (no padding). -/
def maxPool2d {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  have _ := h1
  have _ := h2
  let inC32 ← u32 inC
  let inH32 ← u32 inH
  let inW32 ← u32 inW
  let kH32 ← u32 kH
  let kW32 ← u32 kW
  let stride32 ← u32 stride
  let pad32 : UInt32 := 0
  let x ← requireValue (t := t) xId (.dim inC (.dim inH (.dim inW .scalar)))
  let outH : Nat := (inH - kH) / stride + 1
  let outW : Nat := (inW - kW) / stride + 1
  let outShape : Shape := .dim inC (.dim outH (.dim outW .scalar))
  let y := torchleanMaxPool2dFwdCuda x inC32 inH32 inW32 kH32 kW32 stride32 pad32
  let node : Node :=
    { name := some "max_pool2d"
      value := { s := outShape, buf := y }
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        let dx := torchleanMaxPool2dBwdCuda x dLdy.buf inC32 inH32 inW32 kH32 kW32 stride32 pad32
        pure [(xId, { s := .dim inC (.dim inH (.dim inW .scalar)), buf := dx })] }
  pure (t.addNode node)

/-- MaxPool2D via ConvPool FFI (with symmetric padding). -/
def maxPool2dPad {kH kW inH inW inC stride padding : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  have _ := h1
  have _ := h2
  let inC32 ← u32 inC
  let inH32 ← u32 inH
  let inW32 ← u32 inW
  let kH32 ← u32 kH
  let kW32 ← u32 kW
  let stride32 ← u32 stride
  let pad32 ← u32 padding
  let x ← requireValue (t := t) xId (.dim inC (.dim inH (.dim inW .scalar)))
  let outH : Nat := (inH + 2 * padding - kH) / stride + 1
  let outW : Nat := (inW + 2 * padding - kW) / stride + 1
  let outShape : Shape := .dim inC (.dim outH (.dim outW .scalar))
  let y := torchleanMaxPool2dFwdCuda x inC32 inH32 inW32 kH32 kW32 stride32 pad32
  let node : Node :=
    { name := some "max_pool2d_pad"
      value := { s := outShape, buf := y }
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        let dx := torchleanMaxPool2dBwdCuda x dLdy.buf inC32 inH32 inW32 kH32 kW32 stride32 pad32
        pure [(xId, { s := .dim inC (.dim inH (.dim inW .scalar)), buf := dx })] }
  pure (t.addNode node)

/-- N-D max pooling (CUDA) via ConvPool FFI (rank ≤ 8). -/
def maxPool
    {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  have _ := hKernel
  if d = 0 then
    throw "autograd: cuda: max_pool: d=0 is not supported"
  if d > 8 then
    throw "autograd: cuda: max_pool: rank too large (max 8)"
  if !decide (∀ i : Fin d, stride.get i ≠ 0) then
    throw "autograd: cuda: max_pool: stride must be > 0"

  let inC32 ← u32 C
  let inSpatialArr : Array Nat := Array.ofFn (fun i : Fin d => inSpatial.get i)
  let kernelArr : Array Nat := Array.ofFn (fun i : Fin d => kernel.get i)
  let strideArr : Array Nat := Array.ofFn (fun i : Fin d => stride.get i)
  let paddingArr : Array Nat := Array.ofFn (fun i : Fin d => padding.get i)

  let inputShape : Shape :=
    Shape.ofList (C :: inSpatial.toList)
  let outSpatial : Vector Nat d :=
    Vector.ofFn (fun a =>
      (inSpatial.get a + 2 * padding.get a - kernel.get a) / stride.get a + 1)
  let outShape : Shape :=
    Shape.ofList (C :: outSpatial.toList)

  let xBuf ← requireValue (t := t) xId inputShape
  let y :=
    torchleanMaxPoolFwdCuda xBuf inSpatialArr kernelArr strideArr paddingArr inC32

  let node : Node :=
    { name := some "max_pool"
      value := { s := outShape, buf := y }
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        let dx :=
          torchleanMaxPoolBwdCuda xBuf dLdy.buf
            inSpatialArr kernelArr strideArr paddingArr inC32
        pure [(xId, { s := inputShape, buf := dx })] }
  pure (t.addNode node)

/-- Smooth max-pool2d (log-sum-exp surrogate) via ConvPool FFI (no padding). -/
def smoothMaxPool2d {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (t : Tape) (xId : Nat) (beta : Float) : Result (Tape × Nat) := do
  if beta == 0.0 then
    throw "autograd: cuda: smooth_max_pool2d: beta must be nonzero"
  have _ := h1
  have _ := h2
  let inC32 ← u32 inC
  let inH32 ← u32 inH
  let inW32 ← u32 inW
  let kH32 ← u32 kH
  let kW32 ← u32 kW
  let stride32 ← u32 stride
  let pad32 : UInt32 := 0
  let x ← requireValue (t := t) xId (.dim inC (.dim inH (.dim inW .scalar)))
  let outH : Nat := (inH - kH) / stride + 1
  let outW : Nat := (inW - kW) / stride + 1
  let outShape : Shape := .dim inC (.dim outH (.dim outW .scalar))
  let y := torchleanSmoothMaxPool2dFwdCuda x beta inC32 inH32 inW32 kH32 kW32 stride32 pad32
  let node : Node :=
    { name := some "smooth_max_pool2d"
      value := { s := outShape, buf := y }
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        let dx :=
          torchleanSmoothMaxPool2dBwdCuda x dLdy.buf beta inC32 inH32 inW32 kH32 kW32 stride32 pad32
        pure [(xId, { s := .dim inC (.dim inH (.dim inW .scalar)), buf := dx })] }
  pure (t.addNode node)

/-- Smooth max-pool2d (log-sum-exp surrogate) via ConvPool FFI (with symmetric padding). -/
def smoothMaxPool2dPad {kH kW inH inW inC stride padding : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (t : Tape) (xId : Nat) (beta : Float) : Result (Tape × Nat) := do
  if beta == 0.0 then
    throw "autograd: cuda: smooth_max_pool2d_pad: beta must be nonzero"
  have _ := h1
  have _ := h2
  let inC32 ← u32 inC
  let inH32 ← u32 inH
  let inW32 ← u32 inW
  let kH32 ← u32 kH
  let kW32 ← u32 kW
  let stride32 ← u32 stride
  let pad32 ← u32 padding
  let x ← requireValue (t := t) xId (.dim inC (.dim inH (.dim inW .scalar)))
  let outH : Nat := (inH + 2 * padding - kH) / stride + 1
  let outW : Nat := (inW + 2 * padding - kW) / stride + 1
  let outShape : Shape := .dim inC (.dim outH (.dim outW .scalar))
  let y := torchleanSmoothMaxPool2dFwdCuda x beta inC32 inH32 inW32 kH32 kW32 stride32 pad32
  let node : Node :=
    { name := some "smooth_max_pool2d_pad"
      value := { s := outShape, buf := y }
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        let dx :=
          torchleanSmoothMaxPool2dBwdCuda x dLdy.buf beta inC32 inH32 inW32 kH32 kW32 stride32 pad32
        pure [(xId, { s := .dim inC (.dim inH (.dim inW .scalar)), buf := dx })] }
  pure (t.addNode node)

/-- N-D smooth max pooling (CUDA) via ConvPool FFI (rank ≤ 8). -/
def smoothMaxPool
    {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    (t : Tape) (xId : Nat) (beta : Float) : Result (Tape × Nat) := do
  if beta == 0.0 then
    throw "autograd: cuda: smooth_max_pool: beta must be nonzero"
  have _ := hKernel
  if d = 0 then
    throw "autograd: cuda: smooth_max_pool: d=0 is not supported"
  if d > 8 then
    throw "autograd: cuda: smooth_max_pool: rank too large (max 8)"
  if !decide (∀ i : Fin d, stride.get i ≠ 0) then
    throw "autograd: cuda: smooth_max_pool: stride must be > 0"

  let inC32 ← u32 C
  let inSpatialArr : Array Nat := Array.ofFn (fun i : Fin d => inSpatial.get i)
  let kernelArr : Array Nat := Array.ofFn (fun i : Fin d => kernel.get i)
  let strideArr : Array Nat := Array.ofFn (fun i : Fin d => stride.get i)
  let paddingArr : Array Nat := Array.ofFn (fun i : Fin d => padding.get i)

  let inputShape : Shape :=
    Shape.ofList (C :: inSpatial.toList)
  let outSpatial : Vector Nat d :=
    Vector.ofFn (fun a =>
      (inSpatial.get a + 2 * padding.get a - kernel.get a) / stride.get a + 1)
  let outShape : Shape :=
    Shape.ofList (C :: outSpatial.toList)

  let xBuf ← requireValue (t := t) xId inputShape
  let y :=
    torchleanSmoothMaxPoolFwdCuda xBuf beta
      inSpatialArr kernelArr strideArr paddingArr
      inC32

  let node : Node :=
    { name := some "smooth_max_pool"
      value := { s := outShape, buf := y }
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        let dx :=
          torchleanSmoothMaxPoolBwdCuda xBuf dLdy.buf beta
            inSpatialArr kernelArr strideArr paddingArr
            inC32
        pure [(xId, { s := inputShape, buf := dx })] }
  pure (t.addNode node)

/-- AvgPool2D via ConvPool FFI (no padding). -/
def avgPool2d {kH kW inH inW inC stride : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0)
  (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  have _ := h1
  have _ := h2
  let inC32 ← u32 inC
  let inH32 ← u32 inH
  let inW32 ← u32 inW
  let kH32 ← u32 kH
  let kW32 ← u32 kW
  let stride32 ← u32 stride
  let pad32 : UInt32 := 0
  let x ← requireValue (t := t) xId (.dim inC (.dim inH (.dim inW .scalar)))
  let outH : Nat := (inH - kH) / stride + 1
  let outW : Nat := (inW - kW) / stride + 1
  let outShape : Shape := .dim inC (.dim outH (.dim outW .scalar))
  let y := torchleanAvgPool2dFwdCuda x inC32 inH32 inW32 kH32 kW32 stride32 pad32
  let node : Node :=
    { name := some "avg_pool2d"
      value := { s := outShape, buf := y }
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        let dx := torchleanAvgPool2dBwdCuda dLdy.buf inC32 inH32 inW32 kH32 kW32 stride32 pad32
        pure [(xId, { s := .dim inC (.dim inH (.dim inW .scalar)), buf := dx })] }
  pure (t.addNode node)

/-- AvgPool2D via ConvPool FFI (with symmetric padding). -/
def avgPool2dPad {kH kW inH inW inC stride padding : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0)
  (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  have _ := h1
  have _ := h2
  let inC32 ← u32 inC
  let inH32 ← u32 inH
  let inW32 ← u32 inW
  let kH32 ← u32 kH
  let kW32 ← u32 kW
  let stride32 ← u32 stride
  let pad32 ← u32 padding
  let x ← requireValue (t := t) xId (.dim inC (.dim inH (.dim inW .scalar)))
  let outH : Nat := (inH + 2 * padding - kH) / stride + 1
  let outW : Nat := (inW + 2 * padding - kW) / stride + 1
  let outShape : Shape := .dim inC (.dim outH (.dim outW .scalar))
  let y := torchleanAvgPool2dFwdCuda x inC32 inH32 inW32 kH32 kW32 stride32 pad32
  let node : Node :=
    { name := some "avg_pool2d_pad"
      value := { s := outShape, buf := y }
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        let dx := torchleanAvgPool2dBwdCuda dLdy.buf inC32 inH32 inW32 kH32 kW32 stride32 pad32
        pure [(xId, { s := .dim inC (.dim inH (.dim inW .scalar)), buf := dx })] }
  pure (t.addNode node)

/-- N-D average pooling (CUDA) via ConvPool FFI (rank ≤ 8). -/
def avgPool
    {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
    (hKernel : ∀ i : Fin d, kernel.get i ≠ 0)
    (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  have _ := hKernel
  if d = 0 then
    throw "autograd: cuda: avg_pool: d=0 is not supported"
  if d > 8 then
    throw "autograd: cuda: avg_pool: rank too large (max 8)"
  if !decide (∀ i : Fin d, stride.get i ≠ 0) then
    throw "autograd: cuda: avg_pool: stride must be > 0"

  let inC32 ← u32 C
  let inSpatialArr : Array Nat := Array.ofFn (fun i : Fin d => inSpatial.get i)
  let kernelArr : Array Nat := Array.ofFn (fun i : Fin d => kernel.get i)
  let strideArr : Array Nat := Array.ofFn (fun i : Fin d => stride.get i)
  let paddingArr : Array Nat := Array.ofFn (fun i : Fin d => padding.get i)

  let inputShape : Shape :=
    Shape.ofList (C :: inSpatial.toList)
  let outSpatial : Vector Nat d :=
    Vector.ofFn (fun a =>
      (inSpatial.get a + 2 * padding.get a - kernel.get a) / stride.get a + 1)
  let outShape : Shape :=
    Shape.ofList (C :: outSpatial.toList)

  let xBuf ← requireValue (t := t) xId inputShape
  let y :=
    torchleanAvgPoolFwdCuda xBuf
      inSpatialArr kernelArr strideArr paddingArr
      inC32

  let node : Node :=
    { name := some "avg_pool"
      value := { s := outShape, buf := y }
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        let dx :=
          torchleanAvgPoolBwdCuda dLdy.buf
            inSpatialArr kernelArr strideArr paddingArr
            inC32
        pure [(xId, { s := inputShape, buf := dx })] }
  pure (t.addNode node)

/-!
## Normalization
-/

/--
LayerNorm over the last dimension for `(seqLen, embedDim)` buffers.

This implementation uses the standard stable formulas and is expressed in terms of existing
CUDA kernels (axis reductions + broadcasts + pointwise ops).
-/
def layerNorm {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
  (t : Tape) (xId gammaId betaId : Nat) : Result (Tape × Nat) := do
  have _ := h_seq_pos
  have _ := h_embed_pos
  let rows32 ← u32 seqLen
  let cols32 ← u32 embedDim
  let x ← requireValue (t := t) xId (.dim seqLen (.dim embedDim .scalar))
  let gamma ← requireValue (t := t) gammaId (.dim embedDim .scalar)
  let beta ← requireValue (t := t) betaId (.dim embedDim .scalar)
  -- Forward intermediates.
  let sum1 := Buffer.reduceSumAxis1 x rows32 cols32
  let invCols : Float := 1.0 / Float.ofNat embedDim
  let mean := Buffer.scale sum1 invCols                           -- (rows)
  let meanB := Buffer.broadcastVecToCols mean rows32 cols32        -- (rows,cols)
  let centered := Buffer.sub x meanB
  let centered2 := Buffer.mul centered centered
  let varSum := Buffer.reduceSumAxis1 centered2 rows32 cols32
  let var := Buffer.scale varSum invCols                           -- (rows)
  let eps : Float := Numbers.epsilon
  let epsVec := Buffer.full rows32 eps
  let varEps := Buffer.add var epsVec
  let std := Buffer.sqrt varEps                                    -- (rows)
  let stdB := Buffer.broadcastVecToCols std rows32 cols32
  let xHat := Buffer.div centered stdB
  let gammaB := Buffer.broadcastVecToRows gamma rows32 cols32
  let betaB := Buffer.broadcastVecToRows beta rows32 cols32
  let xHatGamma := Buffer.mul xHat gammaB
  let y := Buffer.add xHatGamma betaB
  let outShape : Shape := .dim seqLen (.dim embedDim .scalar)
  let node : Node :=
    { name := some "layer_norm"
      value := { s := outShape, buf := y }
      requires_grad := true
      parents := [xId, gammaId, betaId]
      cleanup :=
        [ sum1, mean, meanB, centered, centered2, varSum, var, epsVec, varEps
        , std, stdB, xHat, gammaB, betaB, xHatGamma ]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        -- dBeta / dGamma (sum over seqLen axis).
        let dBeta := Buffer.reduceSumAxis0 dLdy.buf rows32 cols32
        let dGamma := Buffer.reduceSumAxis0 (Buffer.mul dLdy.buf xHat) rows32 cols32
        -- dX
        let dXhat := Buffer.mul dLdy.buf gammaB
        let sumDXhat := Buffer.reduceSumAxis1 dXhat rows32 cols32         -- (rows)
        let sumDXhatXhat := Buffer.reduceSumAxis1 (Buffer.mul dXhat xHat) rows32 cols32
        let sum1B := Buffer.broadcastVecToCols sumDXhat rows32 cols32
        let sum2B := Buffer.broadcastVecToCols sumDXhatXhat rows32 cols32
        let term :=
          Buffer.sub (Buffer.sub (Buffer.scale dXhat (Float.ofNat embedDim)) sum1B)
            (Buffer.mul xHat sum2B)
        let invStd := Buffer.inv std
        let invStdB := Buffer.broadcastVecToCols invStd rows32 cols32
        let dx := Buffer.scale (Buffer.mul term invStdB) invCols
        pure [
          (xId, { s := outShape, buf := dx }),
          (gammaId, { s := .dim embedDim .scalar, buf := dGamma }),
          (betaId, { s := .dim embedDim .scalar, buf := dBeta })
        ] }
  pure (t.addNode node)

/--
BatchNorm for a single channel-first image `(C,H,W)` (no batch axis).

We normalize per-channel across the spatial dimension `H*W`, reusing the same math as layer-norm
by treating the buffer as a `(channels, height*width)` matrix.
-/
def batchnormChannelFirst
  {channels height width : Nat} (h_c : channels > 0) (h_h : height > 0) (h_w : width > 0)
  (t : Tape) (xId gammaId betaId : Nat) : Result (Tape × Nat) := do
  have _ := h_c
  have _ := h_h
  have _ := h_w
  let rows32 ← u32 channels
  let cols : Nat := height * width
  if cols = 0 then
    throw "autograd: batchnorm_channel_first: height*width = 0"
  let cols32 ← u32 cols
  let xShape : Shape := .dim channels (.dim height (.dim width .scalar))
  let x ← requireValue (t := t) xId xShape
  let gamma ← requireValue (t := t) gammaId (.dim channels .scalar)
  let beta ← requireValue (t := t) betaId (.dim channels .scalar)
  -- Treat as a zero-copy (channels, cols) view; the underlying layout already matches.
  let sum1 := Buffer.reduceSumAxis1 x rows32 cols32
  let invCols : Float := 1.0 / Float.ofNat cols
  let mean := Buffer.scale sum1 invCols
  let meanB := Buffer.broadcastVecToCols mean rows32 cols32
  let centered := Buffer.sub x meanB
  let centered2 := Buffer.mul centered centered
  let varSum := Buffer.reduceSumAxis1 centered2 rows32 cols32
  let var := Buffer.scale varSum invCols
  let eps : Float := Numbers.epsilon
  let epsVec := Buffer.full rows32 eps
  let varEps := Buffer.add var epsVec
  let std := Buffer.sqrt varEps
  let stdB := Buffer.broadcastVecToCols std rows32 cols32
  let xHat := Buffer.div centered stdB
  let gammaB := Buffer.broadcastVecToCols gamma rows32 cols32
  let betaB := Buffer.broadcastVecToCols beta rows32 cols32
  let xHatGamma := Buffer.mul xHat gammaB
  let y := Buffer.add xHatGamma betaB
  let node : Node :=
    { name := some "batchnorm_channel_first"
      value := { s := xShape, buf := y }
      requires_grad := true
      parents := [xId, gammaId, betaId]
      cleanup :=
        [ sum1, mean, meanB, centered, centered2, varSum, var, epsVec, varEps
        , std, stdB, xHat, gammaB, betaB, xHatGamma ]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny xShape
        -- dBeta / dGamma sum over spatial dimension (axis=1 of the folded matrix).
        let dBeta := Buffer.reduceSumAxis1 dLdy.buf rows32 cols32
        let dGamma := Buffer.reduceSumAxis1 (Buffer.mul dLdy.buf xHat) rows32 cols32
        -- dX
        let dXhat := Buffer.mul dLdy.buf gammaB
        let sumDXhat := Buffer.reduceSumAxis1 dXhat rows32 cols32
        let sumDXhatXhat := Buffer.reduceSumAxis1 (Buffer.mul dXhat xHat) rows32 cols32
        let sum1B := Buffer.broadcastVecToCols sumDXhat rows32 cols32
        let sum2B := Buffer.broadcastVecToCols sumDXhatXhat rows32 cols32
        let term :=
          Buffer.sub (Buffer.sub (Buffer.scale dXhat (Float.ofNat cols)) sum1B)
            (Buffer.mul xHat sum2B)
        let invStd := Buffer.inv std
        let invStdB := Buffer.broadcastVecToCols invStd rows32 cols32
        let dx := Buffer.scale (Buffer.mul term invStdB) invCols
        pure [
          (xId, { s := xShape, buf := dx }),
          (gammaId, { s := .dim channels .scalar, buf := dGamma }),
          (betaId, { s := .dim channels .scalar, buf := dBeta })
        ] }
  pure (t.addNode node)

/-!
## Softmax (last axis, row folding)

We implement softmax along the last axis by folding all leading dimensions into one `rows` axis.
This covers:
- 2D softmax (`(rows, cols)`),
- 3D batched softmax (`(batch, rows, cols)`) by folding `batch*rows` into `rows`.
-/

def softmax {s : Shape} (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  match s with
  | .scalar =>
      let _x ← requireValue (t := t) xId Shape.scalar
      let one32 : UInt32 := 1
      let one := Buffer.full one32 1.0
      let node : Node :=
        { name := some "softmax"
          value := { s := Shape.scalar, buf := one }
          requires_grad := true
          parents := [xId]
          backward := fun dLdyAny => do
            let _ ← requireGrad dLdyAny Shape.scalar
            let dx := Buffer.zeros one32
            pure [(xId, { s := Shape.scalar, buf := dx })] }
      pure (t.addNode node)
  | _ =>
      let (rows32, cols32) ← foldRowsColsLastAxis s
      let x ← requireValue (t := t) xId s
      let (y, cleanup) := softmax2dFwdWithCleanup x rows32 cols32
      let node : Node :=
        { name := some "softmax"
          value := { s := s, buf := y }
          requires_grad := true
          parents := [xId]
          cleanup := cleanup
          backward := fun dLdyAny => do
            let dLdy ← requireGrad dLdyAny s
            let dx := softmax2dBwd y dLdy.buf rows32 cols32
            pure [(xId, { s := s, buf := dx })] }
      pure (t.addNode node)

/-- Stable log-softmax along the last axis, implemented directly on CUDA buffers. -/
def logSoftmax {s : Shape} (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  match s with
  | .scalar =>
      let _x ← requireValue (t := t) xId Shape.scalar
      let one32 : UInt32 := 1
      let zero := Buffer.zeros one32
      let node : Node :=
        { name := some "log_softmax"
          value := { s := Shape.scalar, buf := zero }
          requires_grad := true
          parents := [xId]
          backward := fun dLdyAny => do
            let _ ← requireGrad dLdyAny Shape.scalar
            let dx := Buffer.zeros one32
            pure [(xId, { s := Shape.scalar, buf := dx })] }
      pure (t.addNode node)
  | _ =>
      let (rows32, cols32) ← foldRowsColsLastAxis s
      let x ← requireValue (t := t) xId s
      let (y, cleanup) := logSoftmax2dFwdWithCleanup x rows32 cols32
      let node : Node :=
        { name := some "log_softmax"
          value := { s := s, buf := y }
          requires_grad := true
          parents := [xId]
          cleanup := cleanup
          backward := fun dLdyAny => do
            let dLdy ← requireGrad dLdyAny s
            let dx := logSoftmax2dBwd y dLdy.buf rows32 cols32
            pure [(xId, { s := s, buf := dx })] }
      pure (t.addNode node)

/-!
## Multi-head self-attention

Forward structure matches `Spec.MultiHeadAttention.forward`:
1. `Q = x @ Wq`, `K = x @ Wk`, `V = x @ Wv`
2. reshape to heads `(numHeads, n, headDim)`
3. attention per head (batched): `softmax(Q Kᵀ / sqrt(headDim)) @ V`
4. combine heads, then output projection `@ Wo`

Masking:
- If `mask` is provided, we upload it as a float32 `{0,1}` matrix and apply the same semantics as
  the spec: masked logits are replaced by `-1000.0` before softmax, and gradients through masked
  entries are zeroed.
- This incurs a host-to-device copy for the mask (since the mask is a host `Tensor Bool`).
-/

def multiHeadAttention
  {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (t : Tape) (wqId wkId wvId woId xId : Nat)
  (mask : Option (Tensor Bool (.dim n (.dim n .scalar))) := none)
  (useFlash : Bool := true) :
  Result (Tape × Nat) := do
  have _ := h1
  let one32 : UInt32 := 1
  let depth0 : UInt32 := 0
  let depth1 : UInt32 := 1
  let n32 ← u32 n
  let h32 ← u32 numHeads
  let dModel32 ← u32 dModel
  let head32 ← u32 headDim
  let projDim : Nat := numHeads * headDim
  let proj32 ← u32 projDim
  let wq ← requireValue (t := t) wqId (.dim dModel (.dim projDim .scalar))
  let wk ← requireValue (t := t) wkId (.dim dModel (.dim projDim .scalar))
  let wv ← requireValue (t := t) wvId (.dim dModel (.dim projDim .scalar))
  let wo ← requireValue (t := t) woId (.dim projDim (.dim dModel .scalar))
  let x ← requireValue (t := t) xId (.dim n (.dim dModel .scalar))
  -- Projections: (n,dModel) @ (dModel,projDim) -> (n,projDim)
  let Q := Buffer.bmm x wq one32 n32 dModel32 proj32
  let K := Buffer.bmm x wk one32 n32 dModel32 proj32
  let V := Buffer.bmm x wv one32 n32 dModel32 proj32
  -- Split heads:
  --   (n, projDim) views as (n, numHeads, headDim) with the same underlying layout,
  --   then we swap to (numHeads, n, headDim) so each head is the batch axis for `bmm`.
  let dimsView : Array Nat := #[n, numHeads, headDim]
  let dimsHead : Array Nat := #[numHeads, n, headDim]
  let Qh := Buffer.swapAdjacentAtDepth Q dimsView depth0  -- (numHeads,n,headDim)
  let Kh := Buffer.swapAdjacentAtDepth K dimsView depth0  -- (numHeads,n,headDim)
  let Vh := Buffer.swapAdjacentAtDepth V dimsView depth0  -- (numHeads,n,headDim)
  let scale : Float := 1.0 / Float.sqrt (Float.ofNat headDim)
  -- Optional mask: `mask[i,j]=true` means allowed; false positions are set to a large negative
  -- constant before softmax.
  let (maskB, hasMask) : Buffer × UInt32 :=
    match mask with
    | none => (Buffer.zeros 0, 0)
    | some m =>
        let mF := Buffer.ofFloatArray (Convert.flattenBoolMask (s := .dim n (.dim n .scalar)) m)
        let inDims : Array Nat := #[n, n]
        let outDims : Array Nat := #[numHeads, n, n]
        let axisMap : Array Nat := #[0, 1, 2]
        (Buffer.broadcastTo mF inDims outDims axisMap, 1)
  -- Fused native attention over split heads. This replaces the composed
  -- `scores -> mask -> softmax -> bmm` path while keeping the same spec contract.
  let (outHeads, attnCleanup) ←
    if useFlash then
      let outHeads := Buffer.flashAttentionFwd Qh Kh Vh maskB hasMask h32 n32 head32 scale
      pure (outHeads, ([] : List Buffer))
    else
      let KhT := Buffer.swapAdjacentAtDepth Kh dimsHead depth1
      let scores := Buffer.bmm Qh KhT h32 n32 head32 n32
      let scaled0 := Buffer.scale scores scale
      let (scaled, maskCleanup) ←
        match mask with
        | none => pure (scaled0, ([] : List Buffer))
        | some _ => do
            let total : Nat := numHeads * n * n
            let total32 ← u32 total
            let ones := Buffer.full total32 1.0
            let invMask := Buffer.sub ones maskB
            let fill := Buffer.full total32 (-1000.0)
            let scaledMask := Buffer.mul scaled0 maskB
            let fillMask := Buffer.mul fill invMask
            let scaled := Buffer.add scaledMask fillMask
            pure (scaled, [ones, invMask, fill, scaledMask, fillMask])
      let rowsFold32 ← u32 (numHeads * n)
      let (attn, softmaxCleanup) := softmax2dFwdWithCleanup scaled rowsFold32 n32
      let outHeads := Buffer.bmm attn Vh h32 n32 n32 head32
      pure (outHeads, [KhT, scores, scaled0, scaled, attn] ++ maskCleanup ++ softmaxCleanup)
  -- combine heads: swap to (n,numHeads,headDim), then reshape to (n,projDim)
  let swapped := Buffer.swapAdjacentAtDepth outHeads dimsHead depth0  -- (n,numHeads,headDim)
  let concat := swapped  -- view as (n,projDim)
  -- output projection: (n,projDim) @ (projDim,dModel) -> (n,dModel)
  let y := Buffer.bmm concat wo one32 n32 proj32 dModel32
  let outShape : Shape := .dim n (.dim dModel .scalar)
  let node : Node :=
    { name := some "multi_head_attention"
      value := { s := outShape, buf := y }
      requires_grad := true
      parents := [wqId, wkId, wvId, woId, xId]
      cleanup := [Q, K, V, Qh, Kh, Vh, maskB, outHeads, swapped] ++ attnCleanup
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        -- Backprop through output projection: y = concat @ wo
        let woT := Buffer.transpose2d wo proj32 dModel32
        let dConcat := Buffer.bmm dLdy.buf woT one32 n32 dModel32 proj32
        let concatT := Buffer.transpose2d concat n32 proj32
        let dWo := Buffer.bmm concatT dLdy.buf one32 proj32 n32 dModel32
        -- Backprop combine-heads: reshape/view then swap back
        let dSwapped := dConcat -- view (n,numHeads,headDim)
        let dOutHeads := Buffer.swapAdjacentAtDepth dSwapped dimsView depth0 -- (numHeads,n,headDim)
        let (dQh, dKh, dVh) ←
          if useFlash then
            -- Fused VJP for the native attention primitive. The kernels recompute the row-wise
            -- softmax summaries instead of materializing/storing the full attention matrix.
            pure
              ( Buffer.flashAttentionBwdQ Qh Kh Vh maskB dOutHeads hasMask h32 n32 head32 scale
              , Buffer.flashAttentionBwdK Qh Kh Vh maskB dOutHeads hasMask h32 n32 head32 scale
              , Buffer.flashAttentionBwdV Qh Kh Vh maskB dOutHeads hasMask h32 n32 head32 scale )
          else
            let dimsAttn : Array Nat := #[numHeads, n, n]
            let KhT := Buffer.swapAdjacentAtDepth Kh dimsHead depth1
            let scores := Buffer.bmm Qh KhT h32 n32 head32 n32
            let scaled0 := Buffer.scale scores scale
            let scaled ←
              match mask with
              | none => pure scaled0
              | some _ => do
                  let total : Nat := numHeads * n * n
                  let total32 ← u32 total
                  let ones := Buffer.full total32 1.0
                  let invMask := Buffer.sub ones maskB
                  let fill := Buffer.full total32 (-1000.0)
                  pure <| Buffer.add (Buffer.mul scaled0 maskB) (Buffer.mul fill invMask)
            let rowsFold32 ← u32 (numHeads * n)
            let attn := softmax2dFwd scaled rowsFold32 n32
            let VhT := Buffer.swapAdjacentAtDepth Vh dimsHead depth1
            let dAttn := Buffer.bmm dOutHeads VhT h32 n32 head32 n32
            let attnT := Buffer.swapAdjacentAtDepth attn dimsAttn depth1
            let dVh := Buffer.bmm attnT dOutHeads h32 n32 n32 head32
            let dScaled := softmax2dBwd attn dAttn rowsFold32 n32
            let dScoresMasked := Buffer.scale dScaled scale
            let dScores :=
              match mask with
              | none => dScoresMasked
              | some _ => Buffer.mul dScoresMasked maskB
            let dQh := Buffer.bmm dScores Kh h32 n32 n32 head32
            let dScoresT := Buffer.swapAdjacentAtDepth dScores dimsAttn depth1
            let dKh := Buffer.bmm dScoresT Qh h32 n32 n32 head32
            pure (dQh, dKh, dVh)
        -- Backprop split-head permutations: swap back to (n,numHeads,headDim), then view as (n,projDim).
        let dQ := Buffer.swapAdjacentAtDepth dQh dimsHead depth0
        let dK := Buffer.swapAdjacentAtDepth dKh dimsHead depth0
        let dV := Buffer.swapAdjacentAtDepth dVh dimsHead depth0
        -- Backprop projections Q = x @ wq etc.
        let wqT := Buffer.transpose2d wq dModel32 proj32
        let wkT := Buffer.transpose2d wk dModel32 proj32
        let wvT := Buffer.transpose2d wv dModel32 proj32
        let dxQ := Buffer.bmm dQ wqT one32 n32 proj32 dModel32
        let dxK := Buffer.bmm dK wkT one32 n32 proj32 dModel32
        let dxV := Buffer.bmm dV wvT one32 n32 proj32 dModel32
        let dx := Buffer.add (Buffer.add dxQ dxK) dxV
        let xT := Buffer.transpose2d x n32 dModel32
        let dWq := Buffer.bmm xT dQ one32 dModel32 n32 proj32
        let dWk := Buffer.bmm xT dK one32 dModel32 n32 proj32
        let dWv := Buffer.bmm xT dV one32 dModel32 n32 proj32
        pure [
          (xId,  { s := .dim n (.dim dModel .scalar), buf := dx }),
          (wqId, { s := .dim dModel (.dim projDim .scalar), buf := dWq }),
          (wkId, { s := .dim dModel (.dim projDim .scalar), buf := dWk }),
          (wvId, { s := .dim dModel (.dim projDim .scalar), buf := dWv }),
          (woId, { s := .dim projDim (.dim dModel .scalar), buf := dWo })
        ] }
  pure (t.addNode node)

end Tape

end Cuda
end Autograd
end Runtime

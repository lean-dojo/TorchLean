/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Tape
public import NN.Runtime.Autograd.Engine.Cuda.Convert

public import NN.Runtime.Autograd.Engine.Cuda.Ops

/-!
# CUDA FNO1D (real RFFT fused path)

This file provides a CUDA-only forward + VJP wrapper for a small real-valued FNO1D model whose
spectral convolution is implemented by the fused cuFFT-backed primitive `Tape.spectralConv1dRfft`.

Why this is not a `TorchLean.NN.LayerDef`:
- `LayerDef` is backend-polymorphic and runs through the `Torch.Ops` interface.
- The fused `spectralConv1dRfft` op is implemented only for the CUDA tape backend.

This module is meant to be called by runnable examples that want the performance path, while the
portable reference path lives in `NN.Runtime.Autograd.TorchLean.Fno1d`.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Cuda

open Spec
open Tensor

namespace Fno1dRfftFused

/-- Runtime vector shape abbreviation used by the small fused FNO wrapper. -/
abbrev vec (n : Nat) : Shape := .dim n .scalar

/-- Runtime matrix shape abbreviation used by the small fused FNO wrapper. -/
abbrev mat (m n : Nat) : Shape := .dim m (.dim n .scalar)

/--
Trainable parameter plus Adam moment buffers.

All three arrays use the same row-major layout for `shape`. The value array is uploaded to CUDA
when building a tape; the moment arrays stay on the host because this small wrapper performs Adam
updates in Lean after downloading gradients.
-/
structure Param where
  /-- Runtime tensor shape for `value`, `m`, and `v`. -/
  shape : Shape
  /-- Current parameter values in row-major order. -/
  value : FloatArray
  /-- Adam first-moment accumulator. -/
  m : FloatArray
  /-- Adam second-moment accumulator. -/
  v : FloatArray

/-- Output of one fused-FNO tape construction. -/
structure Forward where
  /-- The completed CUDA tape. -/
  tape : Tape
  /-- Node id of the prediction tensor. -/
  predId : Nat
  /-- Optional scalar loss node id, present only when a target was supplied. -/
  lossId? : Option Nat
  /-- Tape node ids for parameters, in the same order as the parameter array. -/
  paramIds : Array Nat

/-- Minimal Adam state carried across fused-FNO training steps. -/
structure AdamState where
  /-- Step counter (1-based in the Adam bias correction formulas). -/
  step : Nat := 0
  /-- Cached `beta1^step` for bias correction (starts at 1). -/
  beta1Pow : Float := 1.0
  /-- Cached `beta2^step` for bias correction (starts at 1). -/
  beta2Pow : Float := 1.0

def zerosArray (n : Nat) : FloatArray :=
  FloatArray.mk (Array.mk (List.replicate n 0.0))

/--
One step of the small deterministic LCG used for fused-FNO parameter initialization.

This is intentionally local to the fused CUDA example path so the engine layer does not depend on
the higher-level `Torch.Init` helper namespace.
-/
def lcgNext (s : Nat) : Nat :=
  (1664525 * s + 1013904223) % 4294967296

/-- Deterministic pseudo-random number in `[0, 1)` derived from `seed` and a scalar index. -/
def rand01 (seed idx : Nat) : Float :=
  let rec go : Nat → Nat → Nat
    | 0, s => lcgNext s
    | Nat.succ n, s => go n (lcgNext s)
  (Float.ofNat (go idx seed)) / 4294967296.0

/-- Deterministic uniform sample in `[lo, hi)` for a scalar index. -/
def uniformAt (seed idx : Nat) (lo hi : Float) : Float :=
  lo + rand01 seed idx * (hi - lo)

/-- Initialize a row-major parameter array with deterministic uniform samples. -/
def initFloatArray (shape : Shape) (seed : Nat) (lo hi : Float) : FloatArray :=
  FloatArray.mk <| Array.ofFn (n := Shape.size shape) (fun i =>
    uniformAt seed i.val lo hi)

/-- Initialize a trainable parameter and zero Adam moments. -/
def initParam (shape : Shape) (seed : Nat) (lo hi : Float) : Param :=
  { shape := shape
    value := initFloatArray shape seed lo hi
    m := zerosArray (Shape.size shape)
    v := zerosArray (Shape.size shape) }

/-- Initialize a bias-like parameter at zero with zero Adam moments. -/
def initBias (shape : Shape) : Param :=
  { shape := shape
    value := zerosArray (Shape.size shape)
    m := zerosArray (Shape.size shape)
    v := zerosArray (Shape.size shape) }

/--
Initialize parameters for the fused FNO1D model:

- input lift: `W_in : (1,width)`, `b_in : (width)`
- blocks: `(wRe,wIm) : (modes,width,width)`, `wSkip : (width,width)`, `bSkip : (width)`
- output proj: `W_out : (width,1)`, `b_out : (1)`
-/
def initParams (grid width modes blocks : Nat) (seed : Nat) : Array Param := Id.run do
  let _ := grid
  let spectralShape : Shape := .dim modes (.dim width (.dim width .scalar))
  let wSkipShape : Shape := mat width width
  let bSkipShape : Shape := vec width
  let mut ps : Array Param := #[]
  ps := ps.push (initParam (mat 1 width) (seed + 1) (-0.04) 0.04)
  ps := ps.push (initBias (vec width))
  for b in [0:blocks] do
    let base := seed + 100 + 31 * b
    ps := ps.push (initParam spectralShape (base + 0) (-0.04) 0.04)
    ps := ps.push (initParam spectralShape (base + 1) (-0.04) 0.04)
    ps := ps.push (initParam wSkipShape (base + 2) (-0.04) 0.04)
    ps := ps.push (initBias bSkipShape)
  ps := ps.push (initParam (mat width 1) (seed + 1000) (-0.04) 0.04)
  ps := ps.push (initBias (vec 1))
  pure ps

/-- Fetch a parameter with an error message that points to the fused-FNO wrapper. -/
def getParam (ps : Array Param) (i : Nat) : Result Param :=
  match ps[i]? with
  | some p => pure p
  | none => throw s!"autograd: fused-fno: parameter index out of bounds: {i}"

/-- Upload parameter `i` as a gradient-requiring CUDA tape leaf and record its node id. -/
def addParamLeaf (t : Tape) (ps : Array Param) (paramIds : Array Nat) (i : Nat) :
    Result (Tape × Array Nat × Nat) := do
  let p ← getParam ps i
  let (t', id) := t.leaf { s := p.shape, buf := Buffer.ofFloatArray p.value } (some s!"param{i}")
  pure (t', paramIds.push id, id)

/-- Broadcast a vector of length `cols` across `grid` rows. -/
def broadcastVecToMat (t : Tape) (grid cols : Nat) (xId : Nat) : Result (Tape × Nat) :=
  Tape.broadcastTo (t := t) (s₁ := vec cols) (s₂ := mat grid cols) Shape.BroadcastTo.proof xId

/--
Build a CUDA tape that computes prediction (and optionally MSE loss) for the fused real-RFFT FNO.

Inputs:
- `x : (grid)` (interpreted as `(grid,1)`),
- optional `target : (grid)`.
-/
def forward (grid width modes blocks : Nat)
    (ps : Array Param)
    (x : Tensor Float (vec grid))
    (target? : Option (Tensor Float (vec grid))) :
    Result Forward := do
  let xMatShape : Shape := mat grid 1
  let yMatShape : Shape := mat grid 1
  let hiddenShape : Shape := mat grid width
  let paramIds0 : Array Nat := #[]
  let (t0, xId) := Tape.empty.leaf
    { s := xMatShape, buf := Buffer.ofFloatArray (Convert.flattenFloat (s := vec grid) x) }
    (some "x") false

  let (t1, paramIds1, wInId) ← addParamLeaf t0 ps paramIds0 0
  let (t2, paramIds2, bInId) ← addParamLeaf t1 ps paramIds1 1
  let mut paramIds := paramIds2
  let (t3, h0Id) ← Tape.matmul (t := t2) (m := grid) (n := 1) (p := width) xId wInId
  let (t4, bInBId) ← broadcastVecToMat (grid := grid) (cols := width) t3 bInId
  let (t5, hId0) ← Tape.add (t := t4) (s := hiddenShape) h0Id bInBId

  let mut t := t5
  let mut hId := hId0
  for b in [0:blocks] do
    let base := 2 + 4 * b
    let (tA, idsA, wReId) ← addParamLeaf t ps paramIds base
    t := tA; paramIds := idsA
    let (tB, idsB, wImId) ← addParamLeaf t ps paramIds (base + 1)
    t := tB; paramIds := idsB
    let (tC, idsC, wSkipId) ← addParamLeaf t ps paramIds (base + 2)
    t := tC; paramIds := idsC
    let (tD, idsD, bSkipId) ← addParamLeaf t ps paramIds (base + 3)
    t := tD; paramIds := idsD
    let (tSpec, ySpecId) ← Tape.spectralConv1dRfft (t := t) (grid := grid) (width := width) (modes := modes)
      hId wReId wImId
    let (tSkip0, ySkip0Id) ← Tape.matmul (t := tSpec) (m := grid) (n := width) (p := width) hId wSkipId
    let (tBias, bSkipBId) ← broadcastVecToMat (grid := grid) (cols := width) tSkip0 bSkipId
    let (tSkip, ySkipId) ← Tape.add (t := tBias) (s := hiddenShape) ySkip0Id bSkipBId
    let (tSum, yId) ← Tape.add (t := tSkip) (s := hiddenShape) ySpecId ySkipId
    let (tRelu, yReluId) ← Tape.relu (t := tSum) (s := hiddenShape) yId
    t := tRelu
    hId := yReluId

  let outBase := 2 + 4 * blocks
  let (tOutW, idsOutW, wOutId) ← addParamLeaf t ps paramIds outBase
  t := tOutW; paramIds := idsOutW
  let (tOutB, idsOutB, bOutId) ← addParamLeaf t ps paramIds (outBase + 1)
  t := tOutB; paramIds := idsOutB
  let (tPred0, pred0Id) ← Tape.matmul (t := t) (m := grid) (n := width) (p := 1) hId wOutId
  let (tPredB, bOutBId) ← Tape.broadcastTo (t := tPred0) (s₁ := vec 1) (s₂ := yMatShape)
    Shape.BroadcastTo.proof bOutId
  let (tPred, predId) ← Tape.add (t := tPredB) (s := yMatShape) pred0Id bOutBId
  match target? with
  | none =>
      pure { tape := tPred, predId := predId, lossId? := none, paramIds := paramIds }
  | some y =>
      let (tTarget, targetId) := tPred.leaf
        { s := yMatShape, buf := Buffer.ofFloatArray (Convert.flattenFloat (s := vec grid) y) }
        (some "target") false
      let (tLoss, lossId) ← Tape.mseLoss (t := tTarget) (s := yMatShape) predId targetId
      pure { tape := tLoss, predId := predId, lossId? := some lossId, paramIds := paramIds }

/-- Download a scalar CUDA tape value to host `Float`. -/
def scalarFromTape (t : Tape) (id : Nat) : Result Float := do
  let b ← Tape.requireValue (t := t) id Shape.scalar
  let a : FloatArray := Buffer.toFloatArray b
  pure (a.get! 0)

/-- Download a `(grid,1)` prediction matrix as a length-`grid` tensor. -/
def predFromTape (grid : Nat) (t : Tape) (id : Nat) : Result (Tensor Float (vec grid)) := do
  let b ← Tape.requireValue (t := t) id (mat grid 1)
  match Convert.unflattenFloat? (s := vec grid) (Buffer.toFloatArray b) with
  | some y => pure y
  | none => throw "autograd: fused-fno: prediction shape mismatch"

/-- Mean MSE loss over a host-side list of `(input,target)` samples. -/
def meanLoss (grid width modes blocks : Nat)
    (ps : Array Param) (samples : List (Tensor Float (vec grid) × Tensor Float (vec grid))) :
    Result Float := do
  if samples.isEmpty then
    pure (0.0 / 0.0)
  else
    let mut acc := 0.0
    for (x, y) in samples do
      let fw ← forward (grid := grid) (width := width) (modes := modes) (blocks := blocks) ps x (some y)
      let lossId ← match fw.lossId? with
        | some id => pure id
        | none => throw "autograd: fused-fno: internal missing loss id"
      acc := acc + (← scalarFromTape fw.tape lossId)
    pure (acc / Float.ofNat samples.length)

/--
Host-side Adam update for one flattened parameter array.

Bias correction factors are passed in already computed as `1 - beta₁^t` and `1 - beta₂^t`.
-/
def adamUpdateArrayBiasCorrected
    (value m v grad : FloatArray)
    (lr beta1 beta2 eps : Float)
    (biasCorr1 biasCorr2 : Float) :
    FloatArray × FloatArray × FloatArray := Id.run do
  let mut value' : Array Float := #[]
  let mut m' : Array Float := #[]
  let mut v' : Array Float := #[]
  for i in [:value.size] do
    let g := grad.get! i
    let mi := beta1 * m.get! i + (1.0 - beta1) * g
    let vi := beta2 * v.get! i + (1.0 - beta2) * g * g
    let mHat := mi / biasCorr1
    let vHat := vi / biasCorr2
    value' := value'.push (value.get! i - lr * mHat / (Float.sqrt vHat + eps))
    m' := m'.push mi
    v' := v'.push vi
  pure (FloatArray.mk value', FloatArray.mk m', FloatArray.mk v')

/--
Run reverse-mode on the fused-FNO tape and update every recorded parameter with Adam.

Gradients are computed on CUDA buffers and downloaded to host arrays before the update. This keeps
the wrapper simple and explicit; high-throughput optimizer kernels should live in a separate CUDA
optimizer layer rather than being hidden inside this model helper.
-/
def updateParamsAdam
    (ps : Array Param) (fw : Forward) (lr : Float) (st : AdamState)
    (beta1 : Float := 0.9) (beta2 : Float := 0.999) (eps : Float := 1e-8) :
    Result (Array Param × AdamState) := do
  let lossId ← match fw.lossId? with
    | some id => pure id
    | none => throw "autograd: fused-fno: internal missing loss id"
  let grads ← Tape.backwardDenseAll (t := fw.tape) lossId
    { s := Shape.scalar, buf := Buffer.full 1 1.0 }

  -- Advance bias correction state.
  let st' : AdamState :=
    { step := st.step + 1
      beta1Pow := st.beta1Pow * beta1
      beta2Pow := st.beta2Pow * beta2 }
  let biasCorr1 := 1.0 - st'.beta1Pow
  let biasCorr2 := 1.0 - st'.beta2Pow

  let mut out := ps
  for i in [:ps.size] do
    let p ← getParam out i
    let nodeId ← match fw.paramIds[i]? with
      | some id => pure id
      | none => throw "autograd: fused-fno: internal missing param id"
    let gAny ← match grads[nodeId]? with
      | some g => pure g
      | none => throw "autograd: fused-fno: internal missing grad"
    if _h : gAny.s = p.shape then
      let grad := Buffer.toFloatArray gAny.buf
      let (value', m', v') :=
        adamUpdateArrayBiasCorrected p.value p.m p.v grad lr beta1 beta2 eps biasCorr1 biasCorr2
      if hi : i < out.size then
        out := out.set i { p with value := value', m := m', v := v' } hi
      else
        throw "autograd: fused-fno: internal update index invalid"
    else
      throw "autograd: fused-fno: gradient shape mismatch"
  pure (out, st')

end Fno1dRfftFused

end Cuda
end Autograd
end Runtime

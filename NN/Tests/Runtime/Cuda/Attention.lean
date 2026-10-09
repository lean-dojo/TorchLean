/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.LibTorch.Ops
public import NN.Tensor
public import NN.Tests.Runtime.Cuda.Utils

/-!
# CUDA Kernel Coverage: Multi-Head Attention

Compares CPU eager tape vs CUDA eager tape for `multi_head_attention` (forward + backward).

The case stays small so float64/float32 roundoff differences stay limited.
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace Attention

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Runtime.Autograd

abbrev n : Nat := 2
abbrev numHeads : Nat := 2
abbrev dModel : Nat := 4
abbrev headDim : Nat := 2

theorem n_ne_zero : n ≠ 0 := by decide

abbrev projDim : Nat := numHeads * headDim

def wq : Tensor Float [dModel, projDim] :=
  (Tensor.from #[
    0.01, 0.02, 0.03, 0.04,
    0.05, 0.06, 0.07, 0.08,
    0.09, 0.10, 0.11, 0.12,
    0.13, 0.14, 0.15, 0.16
  ]).reshape [dModel, projDim] (by dsimp; decide)

def wk : Tensor Float [dModel, projDim] :=
  (Tensor.from #[
    0.02, 0.01, 0.04, 0.03,
    0.06, 0.05, 0.08, 0.07,
    0.10, 0.09, 0.12, 0.11,
    0.14, 0.13, 0.16, 0.15
  ]).reshape [dModel, projDim] (by dsimp; decide)

def wv : Tensor Float [dModel, projDim] :=
  (Tensor.from #[
    0.03, 0.00, 0.01, 0.02,
    0.00, 0.03, 0.02, 0.01,
    0.01, 0.02, 0.03, 0.00,
    0.02, 0.01, 0.00, 0.03
  ]).reshape [dModel, projDim] (by dsimp; decide)

def wo : Tensor Float [projDim, dModel] :=
  (Tensor.from #[
    0.05, 0.00, 0.01, 0.02,
    0.00, 0.05, 0.02, 0.01,
    0.01, 0.02, 0.05, 0.00,
    0.02, 0.01, 0.00, 0.05
  ]).reshape [projDim, dModel] (by dsimp; decide)

def x : Tensor Float [n, dModel] :=
  (Tensor.from #[
    0.10, -0.20, 0.05, 0.30,
    -0.05, 0.25, -0.10, 0.15
  ]).reshape [n, dModel] (by dsimp; decide)

def mask : Tensor Bool [n, n] :=
  (Tensor.from #[
    true,  true,
    false, true
  ]).reshape [n, n] (by dsimp; decide)

/-- Evaluate a checked buffer operation at this point in a test's ownership sequence. -/
@[no_expose] def checked {α : Type} (action : Unit → Except String α) : IO α := do
  let result ← IO.lazyPure action
  IO.ofExcept result

/-- Saved probabilities and borrowed Q/K/V support repeated VJPs after output release.
A fully blocked row contributes zero even when its cotangent is NaN. -/
def checkSavedBuffers : IO Unit := do
  let before ← Runtime.Autograd.LibTorch.Buffer.memory
  let q ← Runtime.Autograd.LibTorch.Buffer.zerosIO 2
  let k ← Runtime.Autograd.LibTorch.Buffer.ofFloatArrayIO <| FloatArray.mk #[1.0, -1.0]
  let v ← Runtime.Autograd.LibTorch.Buffer.ofFloatArrayIO <| FloatArray.mk #[2.0, 4.0]
  let allowed ← Runtime.Autograd.LibTorch.Buffer.ofFloatArrayIO <|
    FloatArray.mk #[1.0, 1.0, 0.0, 0.0]
  let (output, probabilities) ← checked fun _ =>
    Runtime.Autograd.LibTorch.Buffer.attentionForward q k v (some allowed) 1 2 1 1.0
  discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO allowed
  Utils.assertFloatArrayApprox "attention forward"
    (← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO output) (FloatArray.mk #[3.0, 0.0]) 1e-5
  Utils.assertFloatArrayApprox "attention saved probabilities"
    (← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO probabilities)
    (FloatArray.mk #[0.5, 0.5, 0.0, 0.0]) 1e-5
  discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO output
  for blockedSeed in #[7.0, 0.0 / 0.0] do
    let seed ← Runtime.Autograd.LibTorch.Buffer.ofFloatArrayIO <|
      FloatArray.mk #[1.0, blockedSeed]
    let (dq, dk, dv) ← checked fun _ =>
      Runtime.Autograd.LibTorch.Buffer.attentionBackward q k v probabilities seed 1 2 1 1.0
    Utils.assertFloatArrayApprox "attention dQ ignores blocked cotangent"
      (← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO dq) (FloatArray.mk #[-1.0, 0.0]) 1e-5
    Utils.assertFloatArrayApprox "attention dK ignores blocked cotangent"
      (← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO dk) (FloatArray.mk #[0.0, 0.0]) 1e-5
    Utils.assertFloatArrayApprox "attention dV ignores blocked cotangent"
      (← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO dv) (FloatArray.mk #[0.5, 0.5]) 1e-5
    for buffer in #[seed, dq, dk, dv] do
      discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO buffer
  -- A blocked query must be removed before dSᵀ Q; multiplying it by zero still yields NaN.
  let allowed ← Runtime.Autograd.LibTorch.Buffer.ofFloatArrayIO <|
    FloatArray.mk #[1.0, 1.0, 0.0, 0.0]
  let seed ← Runtime.Autograd.LibTorch.Buffer.ofFloatArrayIO <| FloatArray.mk #[1.0, 7.0]
  for value in #[0.0 / 0.0, 1.0 / 0.0, -1.0 / 0.0] do
    let blockedQ ← Runtime.Autograd.LibTorch.Buffer.ofFloatArrayIO <| FloatArray.mk #[0.0, value]
    let (output, saved) ← checked fun _ =>
      Runtime.Autograd.LibTorch.Buffer.attentionForward blockedQ k v (some allowed) 1 2 1 1.0
    Utils.assertFloatArrayApprox "attention blocks nonfinite queries"
      (← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO output) (FloatArray.mk #[3.0, 0.0]) 0.0
    discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO output
    let (dq, dk, dv) ← checked fun _ =>
      Runtime.Autograd.LibTorch.Buffer.attentionBackward blockedQ k v saved seed 1 2 1 1.0
    Utils.assertFloatArrayApprox "attention dK ignores blocked query"
      (← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO dk) (FloatArray.mk #[0.0, 0.0]) 0.0
    Utils.assertFloatArrayApprox "attention blocked-query dQ"
      (← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO dq) (FloatArray.mk #[-1.0, 0.0]) 0.0
    Utils.assertFloatArrayApprox "attention blocked-query dV"
      (← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO dv) (FloatArray.mk #[0.5, 0.5]) 0.0
    for buffer in #[blockedQ, saved, dq, dk, dv] do
      discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO buffer
  for buffer in #[allowed, seed] do
    discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO buffer
  for buffer in #[q, k, v, probabilities] do
    discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO buffer
  -- A blocked row stays zero even when the value product contains 0 * NaN or 0 * infinity.
  let zero ← Runtime.Autograd.LibTorch.Buffer.zerosIO 1
  let one ← Runtime.Autograd.LibTorch.Buffer.fullIO 1 1.0
  for value in #[0.0 / 0.0, 1.0 / 0.0, -1.0 / 0.0] do
    let nonfinite ← Runtime.Autograd.LibTorch.Buffer.fullIO 1 value
    let (blocked, saved) ← checked fun _ =>
      Runtime.Autograd.LibTorch.Buffer.attentionForward zero zero nonfinite (some zero) 1 1 1 1.0
    Utils.assertFloatArrayApprox "attention blocks nonfinite values"
      (← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO blocked) (FloatArray.mk #[0.0]) 0.0
    for buffer in #[blocked, saved] do
      discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO buffer
    -- A nonfinite key also needs a final dQ mask after contraction, in both scaling branches.
    for scale in #[1.0, 0.5] do
      let (blocked, saved) ← checked fun _ =>
        Runtime.Autograd.LibTorch.Buffer.attentionForward zero nonfinite one (some zero) 1 1 1 scale
      Utils.assertFloatArrayApprox "attention blocks nonfinite keys"
        (← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO blocked) (FloatArray.mk #[0.0]) 0.0
      discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO blocked
      let (dq, dk, dv) ← checked fun _ =>
        Runtime.Autograd.LibTorch.Buffer.attentionBackward zero nonfinite one saved one 1 1 1 scale
      for (label, gradient) in #[("dQ", dq), ("dK", dk), ("dV", dv)] do
        Utils.assertFloatArrayApprox s!"attention blocked nonfinite-key {label}"
          (← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO gradient) (FloatArray.mk #[0.0]) 0.0
        discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO gradient
      discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO saved
    discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO nonfinite
  -- Allowed NaN and either infinity are not fully blocked rows, even at zero scale.
  for value in #[0.0 / 0.0, 1.0 / 0.0, -1.0 / 0.0] do
    let nonfinite ← Runtime.Autograd.LibTorch.Buffer.fullIO 1 value
    for (label, query, key) in #[("K", one, nonfinite), ("Q", nonfinite, one)] do
      for scale in #[1.0, 0.0] do
        let (invalid, invalidSaved) ← checked fun _ =>
          Runtime.Autograd.LibTorch.Buffer.attentionForward query key one (some one) 1 1 1 scale
        unless ((← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO invalid).get! 0).isNaN do
          throw <| IO.userError
            s!"attention hid allowed nonfinite arithmetic ({label}={value}, scale={scale})"
        for buffer in #[invalid, invalidSaved] do
          discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO buffer
    discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO nonfinite
  for buffer in #[zero, one] do
    discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO buffer
  let empty ← Runtime.Autograd.LibTorch.Buffer.zerosIO 0
  for shape in (#[(0, 2, 1), (1, 0, 1), (1, 2, 0)] :
      Array (UInt32 × UInt32 × UInt32)) do
    let (batch, rows, cols) := shape
    let (emptyOutput, emptyProbabilities) ← checked fun _ =>
      Runtime.Autograd.LibTorch.Buffer.attentionForward empty empty empty none batch rows cols 1.0
    unless (← Runtime.Autograd.LibTorch.Buffer.sizeIO emptyProbabilities) == batch * rows * rows do
      throw <| IO.userError "empty attention returned probabilities with the wrong size"
    let (dq, dk, dv) ← checked fun _ =>
      Runtime.Autograd.LibTorch.Buffer.attentionBackward
        empty empty empty emptyProbabilities empty batch rows cols 1.0
    for result in #[emptyOutput, dq, dk, dv] do
      unless (← Runtime.Autograd.LibTorch.Buffer.sizeIO result) == 0 do
        throw <| IO.userError "attention returned a nonempty output/gradient for an empty shape"
      discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO result
    discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO emptyProbabilities
  discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO empty
  let after ← Runtime.Autograd.LibTorch.Buffer.memory
  unless after.liveBytes == before.liveBytes do
    throw <| IO.userError "attention saved-buffer checks retained tensor payloads"

/-- Discard masked overflow and avoid intermediate overflow for shrinking and growing scales. -/
def checkFiniteInputRegressions : IO Unit := do
  let before ← Runtime.Autograd.LibTorch.Buffer.memory
  let q ← Runtime.Autograd.LibTorch.Buffer.ofFloatArrayIO <| FloatArray.mk #[1e20, 1e20]
  let k ← Runtime.Autograd.LibTorch.Buffer.ofFloatArrayIO <| FloatArray.mk #[0.0, 1e20]
  let v ← Runtime.Autograd.LibTorch.Buffer.ofFloatArrayIO <| FloatArray.mk #[2.0, 3.0]
  let mask ← Runtime.Autograd.LibTorch.Buffer.ofFloatArrayIO <|
    FloatArray.mk #[1.0, 0.0, 1.0, 0.0]
  let (output, probabilities) ← checked fun _ =>
    Runtime.Autograd.LibTorch.Buffer.attentionForward q k v (some mask) 1 2 1 1.0
  Utils.assertFloatArrayApprox "attention discards masked overflow"
    (← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO output) (FloatArray.mk #[2.0, 2.0]) 0.0
  for buffer in #[q, k, v, mask, output, probabilities] do
    discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO buffer
  let q ← Runtime.Autograd.LibTorch.Buffer.zerosIO 1
  let k ← Runtime.Autograd.LibTorch.Buffer.ofFloatArrayIO <|
    FloatArray.mk #[Float.ofNat (2 ^ 100)]
  let v ← Runtime.Autograd.LibTorch.Buffer.fullIO 1 1.0
  let (output, probabilities) ← checked fun _ =>
    Runtime.Autograd.LibTorch.Buffer.attentionForward q k v none 1 1 1 (Float.ofNat (2 ^ 60))
  Utils.assertFloatArrayApprox "attention scales the dot product"
    (← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO output) (FloatArray.mk #[1.0]) 0.0
  for buffer in #[q, k, v, output, probabilities] do
    discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO buffer
  -- The raw dot product overflows, but the default head-dimension scale makes it finite.
  let q ← Runtime.Autograd.LibTorch.Buffer.fullIO 4 1.4e19
  let k ← Runtime.Autograd.LibTorch.Buffer.fullIO 4 1.4e19
  let v ← Runtime.Autograd.LibTorch.Buffer.ofFloatArrayIO <| FloatArray.mk #[2.0, 2.0, 4.0, 4.0]
  let seed ← Runtime.Autograd.LibTorch.Buffer.fullIO 4 1.0
  for scale in #[1.0 / Float.sqrt 2.0, -1.0 / Float.sqrt 2.0] do
    let (output, probabilities) ← checked fun _ =>
      Runtime.Autograd.LibTorch.Buffer.attentionForward q k v none 1 2 2 scale
    Utils.assertFloatArrayApprox "attention avoids raw dot-product overflow"
      (← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO output)
      (FloatArray.mk #[3.0, 3.0, 3.0, 3.0]) 0.0
    let (dq, dk, dv) ← checked fun _ =>
      Runtime.Autograd.LibTorch.Buffer.attentionBackward q k v probabilities seed 1 2 2 scale
    Utils.assertFloatArrayApprox "attention large-input dQ"
      (← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO dq)
      (FloatArray.mk #[0.0, 0.0, 0.0, 0.0]) 0.0
    -- Compare after rescaling to keep the tolerance relative to these large finite derivatives.
    let normalizedDK := (← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO dk).data.map (· / 1.4e19)
    Utils.assertFloatArrayApprox "attention large-input dK" (FloatArray.mk normalizedDK)
      (FloatArray.mk #[-2.0 * scale, -2.0 * scale, 2.0 * scale, 2.0 * scale]) 1e-5
    Utils.assertFloatArrayApprox "attention large-input dV"
      (← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO dv)
      (FloatArray.mk #[1.0, 1.0, 1.0, 1.0]) 0.0
    for buffer in #[output, probabilities, dq, dk, dv] do
      discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO buffer
  for buffer in #[q, k, v, seed] do
    discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO buffer
  -- Split the Float64 scale before narrowing the factors to float32.
  -- A tiny nonzero scale therefore need not vanish when its square root is representable.
  let q ← Runtime.Autograd.LibTorch.Buffer.fullIO 2 1e30
  let k ← Runtime.Autograd.LibTorch.Buffer.ofFloatArrayIO <| FloatArray.mk #[0.0, 1e30]
  let v ← Runtime.Autograd.LibTorch.Buffer.ofFloatArrayIO <| FloatArray.mk #[2.0, 4.0]
  for (scale, expected) in #[(1e-46, 4.0), (-1e-46, 2.0), (0.0, 3.0)] do
    let (output, probabilities) ← checked fun _ =>
      Runtime.Autograd.LibTorch.Buffer.attentionForward q k v none 1 2 1 scale
    Utils.assertFloatArrayApprox "attention retains tiny split scale"
      (← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO output)
      (FloatArray.mk #[expected, expected]) 0.0
    for buffer in #[output, probabilities] do
      discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO buffer
  for buffer in #[q, k, v] do
    discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO buffer
  let after ← Runtime.Autograd.LibTorch.Buffer.memory
  unless after.liveBytes == before.liveBytes do
    throw <| IO.userError "attention scaling checks retained tensor payloads"

def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: multi_head_attention ==="
  checkSavedBuffers
  checkFiniteInputRegressions

  let layoutInput : Tensor Float [2, 4] :=
    (Tensor.from (#[0, 1, 2, 3, 4, 5, 6, 7] : Array Float)).reshape [2, 4] (by dsimp; decide)
  let split := Spec.splitHeadsSpec layoutInput 2 2 (by decide)
  let expectedSplit : Tensor Float [2, 2, 2] :=
    (Tensor.from (#[0, 1, 4, 5, 2, 3, 6, 7] : Array Float)).reshape [2, 2, 2] (by dsimp; decide)
  Utils.assertTensorApprox "split-head row-major permutation" split expectedSplit (tol := 0)

  let specScores : Tensor Float [2, 2] :=
    (Tensor.from #[1000.0, -1000.0, 3.0, 4.0]).reshape [2, 2] (by dsimp; decide)
  let specMask : Tensor Bool [2, 2] :=
    (Tensor.from #[false, true, false, false]).reshape [2, 2] (by dsimp; decide)
  let specOut := Spec.hardMaskedSoftmaxSpec specScores specMask
  Utils.assertTensorApprox "hard-masked softmax spec"
    specOut ((Tensor.from #[0.0, 1.0, 0.0, 0.0]).reshape [2, 2] (by dsimp; decide))

  -- A blocked extreme score must not influence stabilization. The second row checks the explicit
  -- all-blocked convention used by hard-masked attention.
  let extremeScores := Runtime.Autograd.LibTorch.Buffer.ofFloatArray <|
    FloatArray.mk #[1000.0, -1000.0, 3.0, 4.0]
  let extremeMask := Runtime.Autograd.LibTorch.Buffer.ofFloatArray <|
    FloatArray.mk #[0.0, 1.0, 0.0, 0.0]
  let extremeOut := Runtime.Autograd.LibTorch.Buffer.hardMaskedSoftmaxByRow
    extremeScores extremeMask 2 2
  let extremeHost := Runtime.Autograd.LibTorch.Buffer.toFloatArray extremeOut
  Utils.assertApprox "hard mask ignores blocked row maximum[0]" (extremeHost.get! 0) 0.0
    (tol := 1e-3)
  Utils.assertApprox "hard mask preserves allowed probability[1]" (extremeHost.get! 1) 1.0
    (tol := 1e-3)
  Utils.assertApprox "all-blocked hard mask row[0]" (extremeHost.get! 2) 0.0 (tol := 1e-3)
  Utils.assertApprox "all-blocked hard mask row[1]" (extremeHost.get! 3) 0.0 (tol := 1e-3)
  discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO extremeScores
  discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO extremeMask
  discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO extremeOut

  let outShape : Shape := [n, dModel]

  -- CPU tape
  let t0 : Tape Float := Tape.empty
  let (t1, wqId) := Tape.leaf (t := t0) wq (name := some "wq")
  let (t2, wkId) := Tape.leaf (t := t1) wk (name := some "wk")
  let (t3, wvId) := Tape.leaf (t := t2) wv (name := some "wv")
  let (t4, woId) := Tape.leaf (t := t3) wo (name := some "wo")
  let (t5, xId) := Tape.leaf (t := t4) x (name := some "x")
  let (t6, yId) ← IO.ofExcept
    (Tape.attention (α := Float) (t := t5)
      (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
      (h1 := n_ne_zero) wqId wkId wvId woId xId (mask := some mask))
  let yCpu ← Utils.cpuValue (s := outShape) t6 yId
  let seedCpu : Spec.SomeTensor Float :=
    Spec.SomeTensor.ofTensor (Tensor.full outShape (1.0 : Float))
  let gradsCpu ← IO.ofExcept (Tape.backwardDenseAll (α := Float) (t := t6) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := outShape) gradsCpu xId
  let dWqCpu ← Utils.cpuGrad (s := [dModel, projDim]) gradsCpu wqId
  let dWkCpu ← Utils.cpuGrad (s := [dModel, projDim]) gradsCpu wkId
  let dWvCpu ← Utils.cpuGrad (s := [dModel, projDim]) gradsCpu wvId
  let dWoCpu ← Utils.cpuGrad (s := [projDim, dModel]) gradsCpu woId

  -- CUDA tape
  let t0c : Runtime.Autograd.LibTorch.Tape := Runtime.Autograd.LibTorch.Tape.empty
  let (t1c, wqIdc) := Runtime.Autograd.LibTorch.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer wq)
    (name := some "wq")
  let (t2c, wkIdc) := Runtime.Autograd.LibTorch.Tape.leaf (t := t1c) (Utils.tensorToAnyBuffer wk)
    (name := some "wk")
  let (t3c, wvIdc) := Runtime.Autograd.LibTorch.Tape.leaf (t := t2c) (Utils.tensorToAnyBuffer wv)
    (name := some "wv")
  let (t4c, woIdc) := Runtime.Autograd.LibTorch.Tape.leaf (t := t3c) (Utils.tensorToAnyBuffer wo)
    (name := some "wo")
  let (t5c, xIdc) := Runtime.Autograd.LibTorch.Tape.leaf (t := t4c) (Utils.tensorToAnyBuffer x)
    (name := some "x")
  let directResult ← Runtime.Autograd.LibTorch.Tape.attention (t := t5c)
      (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
      (h1 := n_ne_zero) wqIdc wkIdc wvIdc woIdc xIdc (mask := some mask)
  let (t6c, yIdc) ← IO.ofExcept directResult
  let yCuda ← Utils.cudaValue (s := outShape) t6c yIdc
  let seedCuda : Runtime.Autograd.LibTorch.AnyBuffer :=
    { s := outShape,
      buf := Runtime.Autograd.LibTorch.Buffer.full (UInt32.ofNat (Spec.Shape.size outShape)) 1.0 }
  let gradsCuda ← IO.ofExcept
    (Runtime.Autograd.LibTorch.Tape.backwardDenseAll (t := t6c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := outShape) gradsCuda xIdc
  let dWqCuda ← Utils.cudaGrad (s := [dModel, projDim]) gradsCuda wqIdc
  let dWkCuda ← Utils.cudaGrad (s := [dModel, projDim]) gradsCuda wkIdc
  let dWvCuda ← Utils.cudaGrad (s := [dModel, projDim]) gradsCuda wvIdc
  let dWoCuda ← Utils.cudaGrad (s := [projDim, dModel]) gradsCuda woIdc

  -- Distinct samples are essential here: duplicated samples cannot expose a permutation that
  -- accidentally exchanges the batch and head axes.
  let xSecond : Tensor Float [n, dModel] :=
    (Tensor.from #[
      1.0, 2.0, 3.0, 1.0,
      -2.0, 1.0, 1.0, 3.0
    ]).reshape [n, dModel] (by dsimp; decide)
  let batchIdentity : Tensor Float [dModel, dModel] :=
    (Tensor.from #[
      1.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 1.0
    ]).reshape [dModel, dModel] (by dsimp; decide)
  let xFirst : Tensor Float [n, dModel] :=
    (Tensor.from #[
      4.0, 0.0, 1.0, 0.0,
      0.0, 4.0, 0.0, 1.0
    ]).reshape [n, dModel] (by dsimp; decide)
  let xBatch : Tensor Float [2, n, dModel] :=
    TorchLean.Tensor.stack 0 fun i => if i.val = 0 then xFirst else xSecond
  let batchShape : Shape := [2, n, dModel]
  let tb0 : Runtime.Autograd.LibTorch.Tape := Runtime.Autograd.LibTorch.Tape.empty
  let (tb1, bwq) :=
    Runtime.Autograd.LibTorch.Tape.leaf (t := tb0) (Utils.tensorToAnyBuffer batchIdentity)
  let (tb2, bwk) :=
    Runtime.Autograd.LibTorch.Tape.leaf (t := tb1) (Utils.tensorToAnyBuffer batchIdentity)
  let (tb3, bwv) :=
    Runtime.Autograd.LibTorch.Tape.leaf (t := tb2) (Utils.tensorToAnyBuffer batchIdentity)
  let (tb4, bwo) :=
    Runtime.Autograd.LibTorch.Tape.leaf (t := tb3) (Utils.tensorToAnyBuffer batchIdentity)
  let (tb5, bx) := Runtime.Autograd.LibTorch.Tape.leaf (t := tb4) (Utils.tensorToAnyBuffer xBatch)
  let batchResult ← Runtime.Autograd.LibTorch.Tape.attention (t := tb5)
    (batch := some 2) (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
    (hBatch := by decide) n_ne_zero bwq bwk bwv bwo bx (mask := some mask)
  let (tb6, byId) ← IO.ofExcept batchResult
  let yBatch ← Utils.cudaValue (s := batchShape) tb6 byId
  let batchSeed : Runtime.Autograd.LibTorch.AnyBuffer :=
    { s := batchShape,
      buf := Runtime.Autograd.LibTorch.Buffer.full
        (UInt32.ofNat (Spec.Shape.size batchShape)) 1.0 }
  let batchGrads ← IO.ofExcept
    (Runtime.Autograd.LibTorch.Tape.backwardDenseAll (t := tb6) byId batchSeed)
  let dxBatch ← Utils.cudaGrad (s := batchShape) batchGrads bx
  let dWqBatch ← Utils.cudaGrad (s := [dModel, projDim]) batchGrads bwq
  let dWkBatch ← Utils.cudaGrad (s := [dModel, projDim]) batchGrads bwk
  let dWvBatch ← Utils.cudaGrad (s := [dModel, projDim]) batchGrads bwv
  let dWoBatch ← Utils.cudaGrad (s := [projDim, dModel]) batchGrads bwo

  -- Compare the batch with two independent CPU tapes, summing shared weight gradients.
  let cpuSample := fun (input : Tensor Float [n, dModel]) => do
    let (tc1, cq) := Tape.leaf (t := Tape.empty) batchIdentity
    let (tc2, ck) := Tape.leaf (t := tc1) batchIdentity
    let (tc3, cv) := Tape.leaf (t := tc2) batchIdentity
    let (tc4, co) := Tape.leaf (t := tc3) batchIdentity
    let (tc5, cx) := Tape.leaf (t := tc4) input
    let (tc6, cy) ← IO.ofExcept <|
      Tape.attention (t := tc5) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim)
        n_ne_zero cq ck cv co cx (some mask)
    let output ← Utils.cpuValue (s := outShape) tc6 cy
    let gradients ← IO.ofExcept <|
      Tape.backwardDenseAll tc6 cy (Spec.SomeTensor.ofTensor (Tensor.full outShape (1.0 : Float)))
    let dx ← Utils.cpuGrad (s := outShape) gradients cx
    let dq ← Utils.cpuGrad (s := [dModel, projDim]) gradients cq
    let dk ← Utils.cpuGrad (s := [dModel, projDim]) gradients ck
    let dv ← Utils.cpuGrad (s := [dModel, projDim]) gradients cv
    let dw ← Utils.cpuGrad (s := [projDim, dModel]) gradients co
    pure (output, dx, dq, dk, dv, dw)
  let (y0, dx0, dq0, dk0, dv0, dw0) ← cpuSample xFirst
  let (y1, dx1, dq1, dk1, dv1, dw1) ← cpuSample xSecond
  let expectedY : Tensor Float batchShape :=
    Tensor.stack 0 fun i => if i.val = 0 then y0 else y1
  let expectedDx : Tensor Float batchShape :=
    Tensor.stack 0 fun i => if i.val = 0 then dx0 else dx1
  Utils.assertTensorApprox "batched mha forward" yBatch expectedY (tol := 2e-2)
  Utils.assertTensorApprox "batched mha dx" dxBatch expectedDx (tol := 2e-2)
  Utils.assertTensorApprox "batched mha dWq" dWqBatch (Tensor.add dq0 dq1) (tol := 2e-2)
  Utils.assertTensorApprox "batched mha dWk" dWkBatch (Tensor.add dk0 dk1) (tol := 2e-2)
  Utils.assertTensorApprox "batched mha dWv" dWvBatch (Tensor.add dv0 dv1) (tol := 2e-2)
  Utils.assertTensorApprox "batched mha dWo" dWoBatch (Tensor.add dw0 dw1) (tol := 2e-2)

  Utils.assertTensorApprox (s := outShape) "mha forward" yCuda yCpu (tol := 2e-2)
  Utils.assertTensorApprox (s := outShape) "mha dx" dxCuda dxCpu (tol := 2e-2)
  Utils.assertTensorApprox (s := [dModel, projDim]) "mha dWq" dWqCuda dWqCpu (tol := 2e-2)
  Utils.assertTensorApprox (s := [dModel, projDim]) "mha dWk" dWkCuda dWkCpu (tol := 2e-2)
  Utils.assertTensorApprox (s := [dModel, projDim]) "mha dWv" dWvCuda dWvCpu (tol := 2e-2)
  Utils.assertTensorApprox (s := [projDim, dModel]) "mha dWo" dWoCuda dWoCpu (tol := 2e-2)

end Attention
end Cuda
end Tests

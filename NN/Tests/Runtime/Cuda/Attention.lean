/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Ops
public import NN.Tensor
public import NN.Tests.Runtime.Cuda.Utils

/-!
# CUDA Kernel Coverage: Multi-Head Attention

Compares CPU eager tape vs CUDA eager tape for `multi_head_attention` (forward + backward).

The case stays small so stub-mode remains lightweight and float64/float32 roundoff differences stay
limited.
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

/-- Evaluate a checked native call at this point in a test's ownership/configuration sequence. -/
@[no_expose] def checked {α : Type} (action : Unit → Except String α) : IO α := do
  let result ← IO.lazyPure action
  Utils.okOrThrow result

@[no_expose] def rejectsPairedBackward
    (output seed : Runtime.Autograd.Cuda.Buffer) : IO Bool := do
  try
    discard <| checked fun _ => Runtime.Autograd.Cuda.Buffer.libTorchAttentionBwd output seed
    pure false
  catch _ =>
    pure true

@[no_expose] def checkDeterministicPolicy
    (output seed : Runtime.Autograd.Cuda.Buffer) : IO Unit := do
  match Runtime.Autograd.Cuda.Buffer.runtimeStatus with
  | .cpuStub => pure ()
  | _ =>
    let deterministic ← Runtime.Autograd.Cuda.LibTorch.getDeterministic
    let benchmark ← Runtime.Autograd.Cuda.LibTorch.getCuDNNBenchmark
    try
      Runtime.Autograd.Cuda.LibTorch.setDeterministic (!deterministic)
      unless ← rejectsPairedBackward output seed do
        throw <| IO.userError "paired attention accepted a changed deterministic policy"
    finally
      Runtime.Autograd.Cuda.LibTorch.setDeterministic deterministic
      Runtime.Autograd.Cuda.LibTorch.setCuDNNBenchmark benchmark

/-- The paired ABI must work in the portable stub as well as the LibTorch build. -/
def checkPairedBuffers : IO Unit := do
  let q ← Runtime.Autograd.Cuda.Buffer.zerosIO 2
  let k ← Runtime.Autograd.Cuda.Buffer.ofFloatArrayIO <| FloatArray.mk #[1.0, -1.0]
  let v ← Runtime.Autograd.Cuda.Buffer.ofFloatArrayIO <| FloatArray.mk #[2.0, 4.0]
  let allowed ← Runtime.Autograd.Cuda.Buffer.ofFloatArrayIO <|
    FloatArray.mk #[1.0, 1.0, 0.0, 0.0]
  let seed ← Runtime.Autograd.Cuda.Buffer.ofFloatArrayIO <| FloatArray.mk #[1.0, 7.0]
  let output ← checked fun _ =>
    Runtime.Autograd.Cuda.Buffer.libTorchAttentionFwd q k v allowed 1 1 2 1 1.0
  for input in #[q, k, v, allowed] do
    discard <| Runtime.Autograd.Cuda.Buffer.releaseIO input
  let ys ← Runtime.Autograd.Cuda.Buffer.toFloatArrayIO output
  Utils.assertFloatArrayApprox "paired attention forward after input release"
    ys (FloatArray.mk #[3.0, 0.0]) 1e-5
  for _ in [:2] do
    let (dq, dk, dv) ← checked fun _ =>
      Runtime.Autograd.Cuda.Buffer.libTorchAttentionBwd output seed
    Utils.assertFloatArrayApprox "paired attention dQ"
      (← Runtime.Autograd.Cuda.Buffer.toFloatArrayIO dq) (FloatArray.mk #[-1.0, 0.0]) 1e-5
    Utils.assertFloatArrayApprox "paired attention dK"
      (← Runtime.Autograd.Cuda.Buffer.toFloatArrayIO dk) (FloatArray.mk #[0.0, 0.0]) 1e-5
    Utils.assertFloatArrayApprox "paired attention dV ignores blocked cotangent"
      (← Runtime.Autograd.Cuda.Buffer.toFloatArrayIO dv) (FloatArray.mk #[0.5, 0.5]) 1e-5
    for gradient in #[dq, dk, dv] do
      discard <| Runtime.Autograd.Cuda.Buffer.releaseIO gradient
  checkDeterministicPolicy output seed
  discard <| Runtime.Autograd.Cuda.Buffer.releaseIO output
  unless ← rejectsPairedBackward output seed do
    throw <| IO.userError "paired attention accepted a released forward buffer"
  discard <| Runtime.Autograd.Cuda.Buffer.releaseIO seed
  let empty ← Runtime.Autograd.Cuda.Buffer.zerosIO 0
  for shape in (#[(0, 2, 1), (1, 0, 1), (1, 2, 0)] :
      Array (UInt32 × UInt32 × UInt32)) do
    let (batch, rows, cols) := shape
    let emptyOutput ← checked fun _ =>
      Runtime.Autograd.Cuda.Buffer.libTorchAttentionFwd
        empty empty empty empty 0 batch rows cols 1.0
    let (dq, dk, dv) ← checked fun _ =>
      Runtime.Autograd.Cuda.Buffer.libTorchAttentionBwd emptyOutput empty
    for result in #[emptyOutput, dq, dk, dv] do
      unless Runtime.Autograd.Cuda.Buffer.size result == 0 do
        throw <| IO.userError "paired attention returned a nonempty result for an empty shape"
      discard <| Runtime.Autograd.Cuda.Buffer.releaseIO result
  discard <| Runtime.Autograd.Cuda.Buffer.releaseIO empty

def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: multi_head_attention ==="
  checkPairedBuffers

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
  -- all-blocked convention used by both composed and fused hard-masked attention.
  let extremeScores := Runtime.Autograd.Cuda.Buffer.ofFloatArray <|
    FloatArray.mk #[1000.0, -1000.0, 3.0, 4.0]
  let extremeMask := Runtime.Autograd.Cuda.Buffer.ofFloatArray <|
    FloatArray.mk #[0.0, 1.0, 0.0, 0.0]
  let extremeOut := Runtime.Autograd.Cuda.Buffer.hardMaskedSoftmaxByRow
    extremeScores extremeMask 2 2
  let extremeHost := Runtime.Autograd.Cuda.Buffer.toFloatArray extremeOut
  Utils.assertApprox "hard mask ignores blocked row maximum[0]" (extremeHost.get! 0) 0.0
    (tol := 1e-3)
  Utils.assertApprox "hard mask preserves allowed probability[1]" (extremeHost.get! 1) 1.0
    (tol := 1e-3)
  Utils.assertApprox "all-blocked hard mask row[0]" (extremeHost.get! 2) 0.0 (tol := 1e-3)
  Utils.assertApprox "all-blocked hard mask row[1]" (extremeHost.get! 3) 0.0 (tol := 1e-3)
  discard <| Runtime.Autograd.Cuda.Buffer.releaseIO extremeScores
  discard <| Runtime.Autograd.Cuda.Buffer.releaseIO extremeMask
  discard <| Runtime.Autograd.Cuda.Buffer.releaseIO extremeOut

  let outShape : Shape := [n, dModel]

  -- CPU tape
  let t0 : Tape Float := Tape.empty
  let (t1, wqId) := Tape.leaf (t := t0) wq (name := some "wq")
  let (t2, wkId) := Tape.leaf (t := t1) wk (name := some "wk")
  let (t3, wvId) := Tape.leaf (t := t2) wv (name := some "wv")
  let (t4, woId) := Tape.leaf (t := t3) wo (name := some "wo")
  let (t5, xId) := Tape.leaf (t := t4) x (name := some "x")
  let (t6, yId) ← Utils.okOrThrow
    (Tape.multiHeadAttention (α := Float) (t := t5)
      (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
      (h1 := n_ne_zero) wqId wkId wvId woId xId (mask := some mask))
  let yCpu ← Utils.cpuValue (s := outShape) t6 yId
  let seedCpu : Spec.SomeTensor Float :=
    Spec.SomeTensor.ofTensor (Tensor.full outShape (1.0 : Float))
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t6) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := outShape) gradsCpu xId
  let dWqCpu ← Utils.cpuGrad (s := [dModel, projDim]) gradsCpu wqId
  let dWkCpu ← Utils.cpuGrad (s := [dModel, projDim]) gradsCpu wkId
  let dWvCpu ← Utils.cpuGrad (s := [dModel, projDim]) gradsCpu wvId
  let dWoCpu ← Utils.cpuGrad (s := [projDim, dModel]) gradsCpu woId

  -- CUDA tape
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, wqIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer wq)
    (name := some "wq")
  let (t2c, wkIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t1c) (Utils.tensorToAnyBuffer wk)
    (name := some "wk")
  let (t3c, wvIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t2c) (Utils.tensorToAnyBuffer wv)
    (name := some "wv")
  let (t4c, woIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t3c) (Utils.tensorToAnyBuffer wo)
    (name := some "wo")
  let (t5c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t4c) (Utils.tensorToAnyBuffer x)
    (name := some "x")
  let directResult ← Runtime.Autograd.Cuda.Tape.multiHeadAttention (t := t5c)
      (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
      (h1 := n_ne_zero) wqIdc wkIdc wvIdc woIdc xIdc (mask := some mask)
  let (t6c, yIdc) ← Utils.okOrThrow directResult
  let yCuda ← Utils.cudaValue (s := outShape) t6c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := outShape,
      buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Spec.Shape.size outShape)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t6c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := outShape) gradsCuda xIdc
  let dWqCuda ← Utils.cudaGrad (s := [dModel, projDim]) gradsCuda wqIdc
  let dWkCuda ← Utils.cudaGrad (s := [dModel, projDim]) gradsCuda wkIdc
  let dWvCuda ← Utils.cudaGrad (s := [dModel, projDim]) gradsCuda wvIdc
  let dWoCuda ← Utils.cudaGrad (s := [projDim, dModel]) gradsCuda woIdc

  -- Composition is selected explicitly for comparison with the default paired ATen path
  -- and the independent CPU tape.
  let t0s : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1s, wqIds) := Runtime.Autograd.Cuda.Tape.leaf (t := t0s) (Utils.tensorToAnyBuffer wq)
    (name := some "wq")
  let (t2s, wkIds) := Runtime.Autograd.Cuda.Tape.leaf (t := t1s) (Utils.tensorToAnyBuffer wk)
    (name := some "wk")
  let (t3s, wvIds) := Runtime.Autograd.Cuda.Tape.leaf (t := t2s) (Utils.tensorToAnyBuffer wv)
    (name := some "wv")
  let (t4s, woIds) := Runtime.Autograd.Cuda.Tape.leaf (t := t3s) (Utils.tensorToAnyBuffer wo)
    (name := some "wo")
  let (t5s, xIds) := Runtime.Autograd.Cuda.Tape.leaf (t := t4s) (Utils.tensorToAnyBuffer x)
    (name := some "x")
  let composedResult ← Runtime.Autograd.Cuda.Tape.multiHeadAttention (t := t5s)
      (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
      (h1 := n_ne_zero) wqIds wkIds wvIds woIds xIds (mask := some mask)
      (attentionCapsule := NN.Backend.Attention.torchLeanComposed)
  let (t6s, yIds) ← Utils.okOrThrow composedResult
  let yCudaComposed ← Utils.cudaValue (s := outShape) t6s yIds
  let seedComposed : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := outShape,
      buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Spec.Shape.size outShape)) 1.0 }
  let gradsComposed ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t6s) yIds seedComposed)
  let dxCudaComposed ← Utils.cudaGrad (s := outShape) gradsComposed xIds
  let dWqCudaComposed ← Utils.cudaGrad (s := [dModel, projDim]) gradsComposed wqIds
  let dWkCudaComposed ← Utils.cudaGrad (s := [dModel, projDim]) gradsComposed wkIds
  let dWvCudaComposed ← Utils.cudaGrad (s := [dModel, projDim]) gradsComposed wvIds
  let dWoCudaComposed ← Utils.cudaGrad (s := [projDim, dModel]) gradsComposed woIds

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
  let tb0 : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (tb1, bwq) :=
    Runtime.Autograd.Cuda.Tape.leaf (t := tb0) (Utils.tensorToAnyBuffer batchIdentity)
  let (tb2, bwk) :=
    Runtime.Autograd.Cuda.Tape.leaf (t := tb1) (Utils.tensorToAnyBuffer batchIdentity)
  let (tb3, bwv) :=
    Runtime.Autograd.Cuda.Tape.leaf (t := tb2) (Utils.tensorToAnyBuffer batchIdentity)
  let (tb4, bwo) :=
    Runtime.Autograd.Cuda.Tape.leaf (t := tb3) (Utils.tensorToAnyBuffer batchIdentity)
  let (tb5, bx) := Runtime.Autograd.Cuda.Tape.leaf (t := tb4) (Utils.tensorToAnyBuffer xBatch)
  let batchResult ← Runtime.Autograd.Cuda.Tape.batchedMultiHeadAttention (t := tb5)
    (batch := 2) (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
    (by decide) n_ne_zero bwq bwk bwv bwo bx (mask := some mask)
    (attentionCapsule := NN.Backend.Attention.torchLeanComposed)
  let (tb6, byId) ← Utils.okOrThrow batchResult
  let yBatch ← Utils.cudaValue (s := batchShape) tb6 byId
  let batchSeed : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := batchShape,
      buf := Runtime.Autograd.Cuda.Buffer.full
        (UInt32.ofNat (Spec.Shape.size batchShape)) 1.0 }
  let batchGrads ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := tb6) byId batchSeed)
  let dxBatch ← Utils.cudaGrad (s := batchShape) batchGrads bx
  let dWqBatch ← Utils.cudaGrad (s := [dModel, projDim]) batchGrads bwq
  let dWkBatch ← Utils.cudaGrad (s := [dModel, projDim]) batchGrads bwk
  let dWvBatch ← Utils.cudaGrad (s := [dModel, projDim]) batchGrads bwv
  let dWoBatch ← Utils.cudaGrad (s := [projDim, dModel]) batchGrads bwo

  -- The default direct ATen pair implements the same batched operation.
  -- Comparing distinct samples catches layout mistakes in the composed BMM path and its VJP.
  let tn0 : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (tn1, nwq) :=
    Runtime.Autograd.Cuda.Tape.leaf (t := tn0) (Utils.tensorToAnyBuffer batchIdentity)
  let (tn2, nwk) :=
    Runtime.Autograd.Cuda.Tape.leaf (t := tn1) (Utils.tensorToAnyBuffer batchIdentity)
  let (tn3, nwv) :=
    Runtime.Autograd.Cuda.Tape.leaf (t := tn2) (Utils.tensorToAnyBuffer batchIdentity)
  let (tn4, nwo) :=
    Runtime.Autograd.Cuda.Tape.leaf (t := tn3) (Utils.tensorToAnyBuffer batchIdentity)
  let (tn5, nx) := Runtime.Autograd.Cuda.Tape.leaf (t := tn4) (Utils.tensorToAnyBuffer xBatch)
  let directBatchResult ← Runtime.Autograd.Cuda.Tape.batchedMultiHeadAttention (t := tn5)
    (batch := 2) (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
    (by decide) n_ne_zero nwq nwk nwv nwo nx (mask := some mask)
  let (tn6, nyId) ← Utils.okOrThrow directBatchResult
  let yBatchDirect ← Utils.cudaValue (s := batchShape) tn6 nyId
  let directBatchSeed : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := batchShape,
      buf := Runtime.Autograd.Cuda.Buffer.full
        (UInt32.ofNat (Spec.Shape.size batchShape)) 1.0 }
  let directBatchGrads ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := tn6) nyId directBatchSeed)
  let dxBatchDirect ← Utils.cudaGrad (s := batchShape) directBatchGrads nx
  let dWqBatchDirect ← Utils.cudaGrad (s := [dModel, projDim]) directBatchGrads nwq
  let dWkBatchDirect ← Utils.cudaGrad (s := [dModel, projDim]) directBatchGrads nwk
  let dWvBatchDirect ← Utils.cudaGrad (s := [dModel, projDim]) directBatchGrads nwv
  let dWoBatchDirect ← Utils.cudaGrad (s := [projDim, dModel]) directBatchGrads nwo
  Utils.assertTensorApprox "batched mha forward" yBatch yBatchDirect (tol := 1e-4)
  Utils.assertTensorApprox "batched mha dx" dxBatch dxBatchDirect (tol := 1e-4)
  Utils.assertTensorApprox "batched mha dWq" dWqBatch dWqBatchDirect (tol := 1e-4)
  Utils.assertTensorApprox "batched mha dWk" dWkBatch dWkBatchDirect (tol := 1e-4)
  Utils.assertTensorApprox "batched mha dWv" dWvBatch dWvBatchDirect (tol := 1e-4)
  Utils.assertTensorApprox "batched mha dWo" dWoBatch dWoBatchDirect (tol := 1e-4)

  -- Attention is numerically "busy" (exp/softmax + multiple matmuls). Use a slightly looser tol.
  Utils.assertTensorApprox (s := outShape) "direct vs composed mha forward" yCuda yCudaComposed
    (tol := 2e-2)
  Utils.assertTensorApprox (s := outShape) "direct vs composed mha dx" dxCuda dxCudaComposed
    (tol := 2e-2)
  Utils.assertTensorApprox (s := [dModel, projDim]) "direct vs composed mha dWq"
    dWqCuda dWqCudaComposed (tol := 2e-2)
  Utils.assertTensorApprox (s := [dModel, projDim]) "direct vs composed mha dWk"
    dWkCuda dWkCudaComposed (tol := 2e-2)
  Utils.assertTensorApprox (s := [dModel, projDim]) "direct vs composed mha dWv"
    dWvCuda dWvCudaComposed (tol := 2e-2)
  Utils.assertTensorApprox (s := [projDim, dModel]) "direct vs composed mha dWo"
    dWoCuda dWoCudaComposed (tol := 2e-2)

  Utils.assertTensorApprox (s := outShape) "mha forward" yCuda yCpu (tol := 2e-2)
  Utils.assertTensorApprox (s := outShape) "mha dx" dxCuda dxCpu (tol := 2e-2)
  Utils.assertTensorApprox (s := [dModel, projDim]) "mha dWq" dWqCuda dWqCpu (tol := 2e-2)
  Utils.assertTensorApprox (s := [dModel, projDim]) "mha dWk" dWkCuda dWkCpu (tol := 2e-2)
  Utils.assertTensorApprox (s := [dModel, projDim]) "mha dWv" dWvCuda dWvCpu (tol := 2e-2)
  Utils.assertTensorApprox (s := [projDim, dModel]) "mha dWo" dWoCuda dWoCpu (tol := 2e-2)

end Attention
end Cuda
end Tests

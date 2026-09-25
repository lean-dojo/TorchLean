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
# CUDA Kernel Coverage: BatchNorm

Compares CPU eager tape and CUDA eager tape for arbitrary-rank spatial BatchNorm.
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace BatchNorm

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Runtime.Autograd

abbrev channels : Nat := 2
abbrev height : Nat := 2
abbrev width : Nat := 2

theorem channels_pos : channels > 0 := by decide
theorem height_pos : height > 0 := by decide
theorem width_pos : width > 0 := by decide

abbrev spatial : Shape := [height, width]

theorem input_wellFormed : (spatial.prependDim channels).wellFormed := by
  exact ⟨channels_pos, ⟨height_pos, ⟨width_pos, trivial⟩⟩⟩

def x : Tensor Float [channels, height, width] :=
  (Tensor.from #[
    -- channel 0
    1.0, 2.0,
    3.0, 4.0,
    -- channel 1
    -0.5, 0.5,
    1.5, -1.0
  ]).reshape [channels, height, width] (by dsimp; decide)

def gamma : Tensor Float [channels] :=
  (Tensor.from #[1.0, 0.5]).reshape [channels] (by dsimp; decide)

def beta : Tensor Float [channels] :=
  (Tensor.from #[0.0, 0.1]).reshape [channels] (by dsimp; decide)

def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: batch_norm ==="

  let outShape : Shape := [channels, height, width]
  let upstream : Tensor Float outShape :=
    (Tensor.from #[1.0, -2.0, 0.5, 3.0, -0.25, 1.5, -1.0, 0.75]).reshape
      [channels, height, width] (by dsimp; decide)

  -- CPU tape
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) x (name := some "x")
  let (t2, gId) := Tape.leaf (t := t1) gamma (name := some "gamma")
  let (t3, bId) := Tape.leaf (t := t2) beta (name := some "beta")
  let (t4, yId) ← Utils.okOrThrow
    (Tape.batchNorm (α := Float) (t := t3) (channels := channels) (sSpatial := spatial)
      input_wellFormed xId gId bId)
  let yCpu ← Utils.cpuValue (s := outShape) t4 yId
  let seedCpu : Spec.SomeTensor Float :=
    Spec.SomeTensor.ofTensor upstream
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t4) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := outShape) gradsCpu xId
  let dGammaCpu ← Utils.cpuGrad (s := [channels]) gradsCpu gId
  let dBetaCpu ← Utils.cpuGrad (s := [channels]) gradsCpu bId

  -- CUDA tape
  let baseline ← Runtime.Autograd.Cuda.Buffer.allocatorStats
  let xBuffer ← Runtime.Autograd.Cuda.Buffer.ofFloatArrayIO
    (Runtime.Autograd.Cuda.Convert.flattenFloat x)
  let gammaBuffer ← Runtime.Autograd.Cuda.Buffer.ofFloatArrayIO
    (Runtime.Autograd.Cuda.Convert.flattenFloat gamma)
  let betaBuffer ← Runtime.Autograd.Cuda.Buffer.ofFloatArrayIO
    (Runtime.Autograd.Cuda.Convert.flattenFloat beta)
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c)
    { s := outShape, buf := xBuffer }
    (name := some "x")
  let (t2c, gIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t1c)
    { s := [channels], buf := gammaBuffer }
    (name := some "gamma")
  let (t3c, bIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t2c)
    { s := [channels], buf := betaBuffer }
    (name := some "beta")
  let result ← IO.lazyPure fun _ =>
    Runtime.Autograd.Cuda.Tape.batchNorm (t := t3c) (channels := channels)
      (spatial := spatial) input_wellFormed xIdc gIdc bIdc
  let (t4c, yIdc) ← Utils.okOrThrow result
  let forward ← Runtime.Autograd.Cuda.Buffer.allocatorStats
  -- Inputs, output, normalized input, broadcast gamma, and one channel-sized standard deviation.
  let expectedBytes := 4 * (4 * Shape.size outShape + 3 * channels)
  unless forward.liveBytes.toNat == baseline.liveBytes.toNat + expectedBytes do
    throw <| IO.userError
      s!"batchnorm: retained forward scratch ({forward.liveBytes} live bytes)"
  let yCuda ← Utils.cudaValue (s := outShape) t4c yIdc
  Utils.assertTensorApprox (s := outShape) "batchnorm forward" yCuda yCpu (tol := 5e-3)
  for pass in [0:3] do
    let seedBuffer ← Runtime.Autograd.Cuda.Buffer.ofFloatArrayIO
      (Runtime.Autograd.Cuda.Convert.flattenFloat upstream)
    let gradients ← Runtime.Autograd.Cuda.Tape.backwardSparse t4c yIdc
      { s := outShape, buf := seedBuffer } (fun id => id == xIdc || id == gIdc || id == bIdc)
    for (id, shape, expected) in #[
        (xIdc, outShape, Runtime.Autograd.Cuda.Convert.flattenFloat dxCpu),
        (gIdc, [channels], Runtime.Autograd.Cuda.Convert.flattenFloat dGammaCpu),
        (bIdc, [channels], Runtime.Autograd.Cuda.Convert.flattenFloat dBetaCpu)] do
      let some gradient := gradients.get? id
        | throw <| IO.userError s!"batchnorm: missing gradient {id} on pass {pass}"
      unless gradient.s == shape do
        throw <| IO.userError "batchnorm: gradient shape mismatch"
      Utils.assertFloatArrayApprox s!"batchnorm gradient {id}, pass {pass}"
        (← Runtime.Autograd.Cuda.Buffer.toFloatArrayIO gradient.buf) expected (tol := 5e-3)
    Runtime.Autograd.Cuda.Tape.releaseSparseGrads gradients
    let after ← Runtime.Autograd.Cuda.Buffer.allocatorStats
    unless after.liveBytes == forward.liveBytes do
      throw <| IO.userError "batchnorm: backward retained temporary payloads"
  for node in t4c.nodes do
    discard <| Runtime.Autograd.Cuda.Buffer.releaseIO node.value.buf
    for buffer in node.cleanup do
      discard <| Runtime.Autograd.Cuda.Buffer.releaseIO buffer
  let retired ← Runtime.Autograd.Cuda.Buffer.allocatorStats
  unless retired.liveBytes == baseline.liveBytes do
    throw <| IO.userError "batchnorm: tape retirement retained payloads"

end BatchNorm
end Cuda
end Tests

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.LibTorch.Ops
public import NN.Tensor
public import NN.Tests.Runtime.Cuda.Utils
import NN.Kernel

/-!
# CUDA Kernel Coverage: Elementwise Ops

Focused activation value/VJP regressions and a composite forward/backward test cover
(`add/sub/mul/scale/abs/sqrt/clamp/max/min/relu/sigmoid/tanh/gelu/softplus/exp/log/inv/safe_log`)
plus `sum`. Generated custom computations also exercise NVRTC, scalar precision, lazy branches,
sequential accumulation, empty tensors and checked native reads.
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace Elementwise

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Runtime.Autograd
open scoped NN.Kernel

def assertActivationValue (label : String) (got expected : Float)
    (rtol : Float := 3e-6) : IO Unit := do
  if expected.isNaN then
    unless got.isNaN do
      throw <| IO.userError s!"{label}: expected NaN, got {got}"
  else if expected == 0.0 || expected.isInf then
    unless got.toBits == expected.toBits do
      throw <| IO.userError
        s!"{label}: expected bits {expected.toBits}, got {got.toBits}"
  else
    -- No absolute floor: replacing a tiny representable result with zero must fail.
    unless !got.isNaN && !got.isInf && (got - expected).abs ≤ rtol * expected.abs do
      throw <| IO.userError s!"{label}: got {got}, expected {expected} (rtol {rtol})"

def assertActivationArray (label : String) (got expected : FloatArray)
    (rtol : Float := 3e-6) : IO Unit := do
  unless got.size == expected.size do
    throw <| IO.userError s!"{label}: size {got.size}, expected {expected.size}"
  for i in [:expected.size] do
    assertActivationValue s!"{label}[{i}]" (got.get! i) (expected.get! i) rtol

def checkActivation
    (label : String)
    (forward : Runtime.Autograd.LibTorch.Buffer → Runtime.Autograd.LibTorch.Buffer)
    (record : (s : Shape) → Runtime.Autograd.LibTorch.Tape → Nat →
      Result (Runtime.Autograd.LibTorch.Tape × Nat))
    (derivative : Float32 → Float32)
    (inputs expected seeds : Array Float) : IO Unit := do
  unless inputs.size == expected.size && inputs.size == seeds.size do
    throw <| IO.userError s!"{label}: inconsistent test data"
  let s : Shape := [inputs.size]
  let input ← Runtime.Autograd.LibTorch.Buffer.ofFloatArrayIO (FloatArray.mk inputs)
  let original ← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO input
  let direct ← IO.lazyPure fun _ => forward input
  assertActivationArray s!"{label} buffer"
    (← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO direct) (FloatArray.mk expected)
  discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO direct
  assertActivationArray s!"{label} borrowed input"
    (← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO input) original

  let (t1, xId) := Runtime.Autograd.LibTorch.Tape.leaf
    Runtime.Autograd.LibTorch.Tape.empty { s := s, buf := input }
  let result ← IO.lazyPure fun _ => record s t1 xId
  let (t2, yId) ← IO.ofExcept result
  let some node := t2.getNode? yId
    | throw <| IO.userError s!"{label}: missing activation node"
  unless t2.nodes.size == 2 && node.parents == #[xId] &&
      node.requiresGrad && node.ownsValue do
    throw <| IO.userError s!"{label}: expected one owned, differentiable unary node"
  assertActivationArray s!"{label} tape"
    (← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO node.value.buf) (FloatArray.mk expected)
  let expectedGrad := FloatArray.mk <| expected.mapIdx fun i y =>
    (seeds[i]!.toFloat32 * derivative y.toFloat32).toFloat
  let retained ← Runtime.Autograd.LibTorch.Buffer.allocatorStats
  for pass in [:2] do
    let seed ← Runtime.Autograd.LibTorch.Buffer.ofFloatArrayIO (FloatArray.mk seeds)
    -- Exercise the recorded closure before accumulation, including signed-zero cotangents.
    let localResult ← IO.lazyPure fun _ => node.backward { s := s, buf := seed }
    let contributions ← IO.ofExcept localResult
    let some (parentId, contribution) := contributions[0]?
      | throw <| IO.userError s!"{label}: missing VJP contribution"
    unless contributions.size == 1 && parentId == xId && contribution.s == s do
      throw <| IO.userError s!"{label}: malformed VJP contribution"
    assertActivationArray s!"{label} local VJP, pass {pass}"
      (← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO contribution.buf) expectedGrad
        (rtol := 3e-5)
    discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO contribution.buf
    assertActivationArray s!"{label} borrowed cotangent"
      (← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO seed) (FloatArray.mk seeds)
    let grads ← Runtime.Autograd.LibTorch.Tape.backwardSparse t2 yId
      { s := s, buf := seed } (fun id => id == xId)
    let some grad := grads.get? xId
      | throw <| IO.userError s!"{label}: missing tape gradient"
    assertActivationArray s!"{label} tape VJP, pass {pass}"
      (← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO grad.buf) expectedGrad (rtol := 3e-5)
    Runtime.Autograd.LibTorch.Tape.releaseSparseGrads grads
    assertActivationArray s!"{label} input after backward"
      (← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO input) original
    assertActivationArray s!"{label} output after backward"
      (← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO node.value.buf) (FloatArray.mk expected)
    let after ← Runtime.Autograd.LibTorch.Buffer.allocatorStats
    unless after.liveBytes == retained.liveBytes do
      throw <| IO.userError s!"{label}: backward retained temporary payloads on pass {pass}"
  discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO node.value.buf
  discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO input

/-- Direct activations retain tiny values, selected nonfinite behavior, and TorchLean's VJPs. -/
def runActivationNumerics : IO Unit := do
  IO.println "== direct tanh/sigmoid numerical regressions =="
  let inf := Float.ofBits 0x7ff0000000000000
  let nan := Float.ofBits 0x7ff8000000000000
  let tiny : Array Float :=
    #[1e-30, -1e-30, 1e-12, -1e-12, 1e-8, -1e-8, 1e-5, -1e-5]
  let central : Array Float := #[-3.0, -1.0, -0.5, 0.5, 1.0, 3.0]
  let inputs := (tiny ++ #[0.0, -0.0] ++ central ++ #[-100.0, 100.0, -inf, inf, nan]).map
    (fun (x : Float) => x.toFloat32.toFloat)
  let seedPattern : Array Float := #[2.0, -0.5, 1.5, -3.0, 0.25]
  let seeds := inputs.mapIdx fun i _ => seedPattern[i % seedPattern.size]!
  let tanhValues := inputs.map fun x => (MathFunctions.tanh x).toFloat32.toFloat
  let tanhNode := fun s t id => Runtime.Autograd.LibTorch.Tape.tanh (s := s) t id
  let sigmoidNode := fun s t id => Runtime.Autograd.LibTorch.Tape.sigmoid (s := s) t id
  let tanhDerivative := fun (y : Float32) => 1 - y * y
  let sigmoidDerivative := fun (y : Float32) => y * (1 - y)
  checkActivation "tanh tiny/center/tails" Runtime.Autograd.LibTorch.Buffer.tanh tanhNode
    tanhDerivative inputs tanhValues seeds

  -- Test normal negative-tail values separately from ATen's saturated float32 tail.
  let sigmoidInputs := inputs ++ #[-80.0, -20.0, 20.0, 80.0]
  let sigmoidValues := sigmoidInputs.map fun x =>
    if x == -100.0 then 0.0 else (Activation.Math.sigmoidSpec x).toFloat32.toFloat
  let sigmoidSeeds := sigmoidInputs.mapIdx fun i _ => seedPattern[i % seedPattern.size]!
  checkActivation "sigmoid tiny/center/tails" Runtime.Autograd.LibTorch.Buffer.sigmoid sigmoidNode
    sigmoidDerivative sigmoidInputs sigmoidValues sigmoidSeeds

  -- Infinite cotangents at saturated outputs still multiply by zero and produce NaN.
  let specialInputs : Array Float :=
    #[0.0, -0.0, 1.0, -1.0, 100.0, -100.0, inf, -inf, nan, 0.5, -100.0]
  let specialSeeds : Array Float :=
    #[-0.0, 0.0, inf, -inf, inf, -inf, -0.0, -2.0, 0.0, nan, nan]
  checkActivation "tanh nonfinite VJP" Runtime.Autograd.LibTorch.Buffer.tanh tanhNode tanhDerivative
    specialInputs (specialInputs.map fun x => (MathFunctions.tanh x).toFloat32.toFloat)
    specialSeeds
  checkActivation "sigmoid nonfinite VJP" Runtime.Autograd.LibTorch.Buffer.sigmoid sigmoidNode
    sigmoidDerivative specialInputs
    (specialInputs.map fun x =>
      if x == -100.0 then 0.0 else (Activation.Math.sigmoidSpec x).toFloat32.toFloat)
    specialSeeds
  checkActivation "tanh empty" Runtime.Autograd.LibTorch.Buffer.tanh tanhNode
    tanhDerivative #[] #[] #[]
  checkActivation "sigmoid empty" Runtime.Autograd.LibTorch.Buffer.sigmoid sigmoidNode
    sigmoidDerivative #[] #[] #[]

private def square := fun (x : Float32) => x * x
private def positiveSquare := fun (x : Float32) => if x < 0 then 0 else x * x
private def squareWide := fun (x : Float) => x * x

/-- Native execution checks complement source-equivalence proofs at the NVRTC/FFI boundary. -/
private def checkCustomComputations : IO Unit := do
  let input : Tensor Float32 [513] := Tensor.ofFn fun i => Float32.ofNat (i.val % 17) - 8
  let squared ← square.run input (device := gpu)
  let activated ← positiveSquare.run input (device := gpu)
  for i in List.finRange 513 do
    unless squared[i].toBits == (square input[i]).toBits &&
        activated[i].toBits == (positiveSquare input[i]).toBits do
      throw <| IO.userError s!"custom binary32: incorrect result at {i}"
  let wide : Tensor Float [2] := [1.0000000000000002, -3.5]
  let squaredWide ← squareWide.run wide (device := gpu)
  for i in List.finRange 2 do
    unless squaredWide[i].toBits == (squareWide wide[i]).toBits do
      throw <| IO.userError s!"custom binary64: incorrect result at {i}"
  let _ ← square.run (Tensor.zeros [0]) (device := gpu)
  let rows : Tensor Float32 [2, 3] := [[16777216, 1, -16777216], [1, 2, 3]]
  let rowSum := Program.of
    (fun (read : NN.Kernel.Reader Float32) (row : UInt64) =>
      NN.Kernel.iterate (fun column acc => do
        let x ← read 0 (row * 3 + column)
        pure (acc + x)) 3 0 0)
  let sums ← rowSum.run (Arguments.empty.push rows) [2] (device := gpu)
  unless sums[0] == 0 && sums[1] == 6 do
    throw <| IO.userError "custom fold: changed sequential addition order"
  let guarded := Program.of
    (fun (read : NN.Kernel.Reader Float32) (i : UInt64) =>
      if i == 0 then pure 7 else read 0 i)
  let noInput : Tensor Float32 [0] := Tensor.zeros [0]
  let guardedOutput ← guarded.run (Arguments.empty.push noInput) [1] (device := gpu)
  unless guardedOutput[0] == 7 do
    throw <| IO.userError "custom conditional: incorrect branch"
  let rejected ← try
    let _ ← guarded.run (Arguments.empty.push noInput) [2] (device := gpu)
    pure false
  catch _ => pure true
  unless rejected do
    throw <| IO.userError "custom read: accepted an out-of-bounds native access"
  IO.println "  custom computations: binary32/binary64, branches, folds and bounds passed"

open Runtime.Autograd.LibTorch in
/-- Check the C ABI's scalar cast and multiplication order against separate ATen calls. -/
private def checkScaledProductExponential : IO Unit := do
  IO.println "=== scaledProdExp composition parity ==="
  -- Varied, signed, finite fixtures, kept in a range where `exp` stays finite.
  let xs : FloatArray := FloatArray.mk
    #[0.10, -0.20, 0.35, -0.50, 0.75, -0.90, 0.00, 1.00, -1.00, 0.42]
  let ys : FloatArray := FloatArray.mk
    #[0.90, -0.75, 0.50, -0.30, 0.15, -0.05, 1.00, -1.00, 0.25, -0.60]
  let n : UInt32 := xs.size.toUInt32
  let x := Buffer.ofFloatArray xs
  let y := Buffer.ofFloatArray ys
  -- Scalars spanning sign and magnitude, including the constant-one case `c = 0`.
  for c in (#[-2.0, 0.5, 3.25, -0.125, 1.0, 0.0] : Array Float) do
    let actual := Buffer.scaledProdExp x y c
    let composed := Buffer.exp (Buffer.mul (Buffer.mul (Buffer.full n c) x) y)
    let af := Buffer.toFloatArray actual
    let ac := Buffer.toFloatArray composed
    if af.size != ac.size then
      throw <| IO.userError s!"scaledProdExp c={c}: size mismatch ({af.size} vs {ac.size})"
    let mut mism : Nat := 0
    let mut maxDiff : Float := 0.0
    for i in [:af.size] do
      let vf := af.get! i
      let vc := ac.get! i
      -- Bit-level comparison: `toBits` distinguishes results that `==` would call equal.
      if vf.toBits != vc.toBits then mism := mism + 1
      let d := Float.abs (vf - vc)
      if d > maxDiff then maxDiff := d
    if mism != 0 then
      throw <| IO.userError <|
        s!"scaledProdExp c={c}: {mism}/{af.size} elements differ from composed exp((c·x)·y) " ++
          s!"(max |Δ|={maxDiff})"
  IO.println "  scaledProdExp bit-identical to composed exp((c·x)·y) over all fixtures ✓"

/-- Exercise native elementwise VJPs, generated computations, and composed buffer operations. -/
@[no_expose] def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: elementwise ==="
  runActivationNumerics
  checkScaledProductExponential
  checkCustomComputations

  let s : Shape := [5]
  let a : Tensor Float s :=
    (Tensor.from #[0.10, -0.20, 0.30, -0.15, 0.05]).reshape [5] (by dsimp; decide)
  let b : Tensor Float s :=
    (Tensor.from #[0.20,  0.10, -0.25, 0.40, -0.05]).reshape [5] (by dsimp; decide)

  let scaleC : Float := 0.3
  let clampLo : Float := 1e-3
  let clampHi : Float := 10.0
  let eps : Float := 1e-6

  -- CPU tape
  let t0 : Tape Float := Tape.empty
  let (t1, aId) := Tape.leaf (t := t0) a (name := some "a")
  let (t2, bId) := Tape.leaf (t := t1) b (name := some "b")
  let (t3, u1) ← IO.ofExcept (Tape.add (α := Float) (t := t2) (s := s) aId bId)
  let (t4, u2) ← IO.ofExcept (Tape.scale (α := Float) (t := t3) (s := s) aId scaleC)
  let (t5, u3) ← IO.ofExcept (Tape.sub (α := Float) (t := t4) (s := s) u1 u2)
  let (t6, u4) ← IO.ofExcept (Tape.mul (α := Float) (t := t5) (s := s) u3 bId)
  let (t7, u5) ← IO.ofExcept (Tape.max (α := Float) (t := t6) (s := s) u4 aId)
  let (t8, u6) ← IO.ofExcept (Tape.min (α := Float) (t := t7) (s := s) u5 bId)
  let (t9, u7) ← IO.ofExcept (Tape.relu (α := Float) (t := t8) (s := s) u6)
  let (t10, u8) ← IO.ofExcept (Tape.sigmoid (α := Float) (t := t9) (s := s) u7)
  let (t11, u9) ← IO.ofExcept (Tape.tanh (α := Float) (t := t10) (s := s) u8)
  let (t12, u10) ← IO.ofExcept (Tape.softplus (α := Float) (t := t11) (s := s) u9)
  let (t13, u11) ← IO.ofExcept (Tape.exp (α := Float) (t := t12) (s := s) u10)
  let (t14, u12) ← IO.ofExcept (Tape.abs (α := Float) (t := t13) (s := s) u11)
  let (t15, u13) ← IO.ofExcept (Tape.clamp (α := Float) (t := t14) (s := s) u12 clampLo clampHi)
  let (t16, u14) ← IO.ofExcept (Tape.sqrt (α := Float) (t := t15) (s := s) u13)
  let (t17, u15) ← IO.ofExcept (Tape.inv (α := Float) (t := t16) (s := s) u14)
  let (t18, u16) ← IO.ofExcept (Tape.log (α := Float) (t := t17) (s := s) u14)
  let (t19, u17) ← IO.ofExcept (Tape.safeLog (α := Float) (t := t18) (s := s) u14 (ε := eps))
  let (t20, u18) ← IO.ofExcept (Tape.add (α := Float) (t := t19) (s := s) u15 u16)
  let (t21, u19) ← IO.ofExcept (Tape.add (α := Float) (t := t20) (s := s) u18 u17)
  let (t22, outId) ← IO.ofExcept (Tape.sum (α := Float) (t := t21) (s := s) u19)

  let outCpu ← Utils.cpuValue (s := Shape.scalar) t22 outId
  let seedCpu : Spec.SomeTensor Float := Spec.SomeTensor.ofTensor (Tensor.scalar 1.0)
  let gradsCpu ← IO.ofExcept (Tape.backwardDenseAll (α := Float) (t := t22) outId seedCpu)
  let dA_cpu ← Utils.cpuGrad (s := s) gradsCpu aId
  let dB_cpu ← Utils.cpuGrad (s := s) gradsCpu bId

  -- CUDA tape
  let t0c : Runtime.Autograd.LibTorch.Tape := Runtime.Autograd.LibTorch.Tape.empty
  let (t1c, aIdc) :=
    Runtime.Autograd.LibTorch.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer a) (name := some "a")
  let (t2c, bIdc) :=
    Runtime.Autograd.LibTorch.Tape.leaf (t := t1c) (Utils.tensorToAnyBuffer b) (name := some "b")
  let (t3c, u1c) ← IO.ofExcept (Runtime.Autograd.LibTorch.Tape.add (t := t2c) (s := s)
    aIdc bIdc)
  let (t4c, u2c) ← IO.ofExcept
    (Runtime.Autograd.LibTorch.Tape.scale (t := t3c) (s := s) aIdc scaleC)
  let (t5c, u3c) ← IO.ofExcept (Runtime.Autograd.LibTorch.Tape.sub (t := t4c) (s := s) u1c u2c)
  let (t6c, u4c) ← IO.ofExcept (Runtime.Autograd.LibTorch.Tape.mul (t := t5c) (s := s) u3c bIdc)
  let (t7c, u5c) ← IO.ofExcept (Runtime.Autograd.LibTorch.Tape.max (t := t6c) (s := s) u4c aIdc)
  let (t8c, u6c) ← IO.ofExcept (Runtime.Autograd.LibTorch.Tape.min (t := t7c) (s := s) u5c bIdc)
  let (t9c, u7c) ← IO.ofExcept (Runtime.Autograd.LibTorch.Tape.relu (t := t8c) (s := s) u6c)
  let (t10c, u8c) ← IO.ofExcept (Runtime.Autograd.LibTorch.Tape.sigmoid (t := t9c) (s := s) u7c)
  let (t11c, u9c) ← IO.ofExcept (Runtime.Autograd.LibTorch.Tape.tanh (t := t10c) (s := s) u8c)
  let (t12c, u10c) ← IO.ofExcept (Runtime.Autograd.LibTorch.Tape.softplus (t := t11c)
    (s := s) u9c)
  let (t13c, u11c) ← IO.ofExcept (Runtime.Autograd.LibTorch.Tape.exp (t := t12c) (s := s) u10c)
  let (t14c, u12c) ← IO.ofExcept (Runtime.Autograd.LibTorch.Tape.abs (t := t13c) (s := s) u11c)
  let (t15c, u13c) ← IO.ofExcept
    (Runtime.Autograd.LibTorch.Tape.clamp (t := t14c) (s := s) u12c clampLo clampHi)
  let (t16c, u14c) ← IO.ofExcept (Runtime.Autograd.LibTorch.Tape.sqrt (t := t15c) (s := s) u13c)
  let (t17c, u15c) ← IO.ofExcept (Runtime.Autograd.LibTorch.Tape.inv (t := t16c) (s := s) u14c)
  let (t18c, u16c) ← IO.ofExcept (Runtime.Autograd.LibTorch.Tape.log (t := t17c) (s := s) u14c)
  let (t19c, u17c) ← IO.ofExcept
    (Runtime.Autograd.LibTorch.Tape.safeLog (t := t18c) (s := s) u14c eps)
  let (t20c, u18c) ← IO.ofExcept (Runtime.Autograd.LibTorch.Tape.add (t := t19c) (s :=
    s) u15c u16c)
  let (t21c, u19c) ← IO.ofExcept (Runtime.Autograd.LibTorch.Tape.add (t := t20c) (s :=
    s) u18c u17c)
  let (t22c, outIdc) ← IO.ofExcept (Runtime.Autograd.LibTorch.Tape.sum (t := t21c) (s
    := s) u19c)

  let outCuda ← Utils.cudaValue (s := Shape.scalar) t22c outIdc
  let seedCuda : Runtime.Autograd.LibTorch.AnyBuffer :=
    { s := Shape.scalar, buf := Runtime.Autograd.LibTorch.Buffer.full 1 1.0 }
  let gradsCuda ← IO.ofExcept
    (Runtime.Autograd.LibTorch.Tape.backwardDenseAll (t := t22c) outIdc seedCuda)
  let dA_cuda ← Utils.cudaGrad (s := s) gradsCuda aIdc
  let dB_cuda ← Utils.cudaGrad (s := s) gradsCuda bIdc

  Utils.assertTensorApprox (s := Shape.scalar) "elementwise forward" outCuda outCpu (tol := 2e-3)
  Utils.assertTensorApprox (s := s) "elementwise backward dA" dA_cuda dA_cpu (tol := 2e-3)
  Utils.assertTensorApprox (s := s) "elementwise backward dB" dB_cuda dB_cpu (tol := 2e-3)

  -- GELU is one semantic tape node and one pointwise kernel in each direction. Check both against
  -- the spec-backed CPU tape over the nonlinear center and saturated tails.
  let geluShape : Shape := [7]
  let geluInput : Tensor Float geluShape :=
    (Tensor.from #[-10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0]).reshape [7] (by dsimp; decide)
  let geluCpu0 : Tape Float := Tape.empty
  let (geluCpu1, geluCpuInputId) :=
    Tape.leaf (t := geluCpu0) geluInput (name := some "gelu input")
  let (geluCpu2, geluCpuOutputId) ← IO.ofExcept <|
    Tape.gelu (α := Float) (t := geluCpu1) (s := geluShape) geluCpuInputId
  let geluCpuOutput ← Utils.cpuValue (s := geluShape) geluCpu2 geluCpuOutputId
  let geluSeedCpu : Spec.SomeTensor Float :=
    Spec.SomeTensor.ofTensor (Tensor.full (α := Float) geluShape 1.0)
  let geluCpuGrads ← IO.ofExcept <|
    Tape.backwardDenseAll (α := Float) (t := geluCpu2) geluCpuOutputId geluSeedCpu
  let geluCpuGrad ← Utils.cpuGrad (s := geluShape) geluCpuGrads geluCpuInputId

  let geluCuda0 : Runtime.Autograd.LibTorch.Tape := Runtime.Autograd.LibTorch.Tape.empty
  let (geluCuda1, geluCudaInputId) :=
    Runtime.Autograd.LibTorch.Tape.leaf
      (t := geluCuda0) (Utils.tensorToAnyBuffer geluInput) (name := some "gelu input")
  let (geluCuda2, geluCudaOutputId) ← IO.ofExcept <|
    Runtime.Autograd.LibTorch.Tape.gelu (t := geluCuda1) (s := geluShape) geluCudaInputId
  let geluCudaOutput ← Utils.cudaValue (s := geluShape) geluCuda2 geluCudaOutputId
  let geluSeedCuda : Runtime.Autograd.LibTorch.AnyBuffer :=
    { s := geluShape, buf := Runtime.Autograd.LibTorch.Buffer.full 7 1.0 }
  let geluCudaGrads ← IO.ofExcept <|
    Runtime.Autograd.LibTorch.Tape.backwardDenseAll
      (t := geluCuda2) geluCudaOutputId geluSeedCuda
  let geluCudaGrad ← Utils.cudaGrad (s := geluShape) geluCudaGrads geluCudaInputId

  Utils.assertTensorApprox
    (s := geluShape) "fused GELU forward" geluCudaOutput geluCpuOutput (tol := 3e-5)
  Utils.assertTensorApprox
    (s := geluShape) "fused GELU backward" geluCudaGrad geluCpuGrad (tol := 3e-5)

  -- The fused optimizer primitive must preserve the staged AdamW computation it replaces.
  let optimShape : Shape := [4]
  let params : Tensor Float optimShape :=
    (Tensor.from #[1.0, -2.0, 0.5, 4.0]).reshape [4] (by dsimp; decide)
  let gradient : Tensor Float optimShape :=
    (Tensor.from #[0.2, -0.1, 0.4, -0.3]).reshape [4] (by dsimp; decide)
  let firstMoment : Tensor Float optimShape :=
    (Tensor.from #[0.01, -0.02, 0.03, -0.04]).reshape [4] (by dsimp; decide)
  let secondMoment : Tensor Float optimShape :=
    (Tensor.from #[0.2, 0.1, 0.4, 0.3]).reshape [4] (by dsimp; decide)
  let paramsBuf := Utils.tensorToBuffer params
  let gradientBuf := Utils.tensorToBuffer gradient
  let firstMomentBuf := Utils.tensorToBuffer firstMoment
  let secondMomentBuf := Utils.tensorToBuffer secondMoment
  let beta1 : Float := 0.9
  let beta2 : Float := 0.999
  let oneMinusBeta1 : Float := 1.0 - beta1
  let oneMinusBeta2 : Float := 1.0 - beta2
  let firstMomentCorrection : Float := 1.0 / (1.0 - beta1)
  let secondMomentCorrection : Float := 1.0 / (1.0 - beta2)
  let epsilon : Float := 1e-8
  let learningRate : Float := 3e-4
  let weightDecay : Float := 0.1

  let mScaled := Runtime.Autograd.LibTorch.Buffer.scale firstMomentBuf beta1
  let expectedM := Runtime.Autograd.LibTorch.Buffer.axpy mScaled gradientBuf oneMinusBeta1
  let gradientSquared := Runtime.Autograd.LibTorch.Buffer.mul gradientBuf gradientBuf
  let vScaled := Runtime.Autograd.LibTorch.Buffer.scale secondMomentBuf beta2
  let expectedV := Runtime.Autograd.LibTorch.Buffer.axpy vScaled gradientSquared oneMinusBeta2
  let mHat := Runtime.Autograd.LibTorch.Buffer.scale expectedM firstMomentCorrection
  let vHat := Runtime.Autograd.LibTorch.Buffer.scale expectedV secondMomentCorrection
  let sqrtVHat := Runtime.Autograd.LibTorch.Buffer.sqrt vHat
  let epsilonBuf := Runtime.Autograd.LibTorch.Buffer.full 4 epsilon
  let denominator := Runtime.Autograd.LibTorch.Buffer.add sqrtVHat epsilonBuf
  let normalizedUpdate := Runtime.Autograd.LibTorch.Buffer.div mHat denominator
  let decayedParams :=
    Runtime.Autograd.LibTorch.Buffer.axpy paramsBuf paramsBuf (-(learningRate * weightDecay))
  let expectedParams :=
    Runtime.Autograd.LibTorch.Buffer.axpy decayedParams normalizedUpdate (-learningRate)
  let (gotParams, gotM, gotV) := Runtime.Autograd.LibTorch.Buffer.adamStep
    paramsBuf gradientBuf firstMomentBuf secondMomentBuf
    beta1 oneMinusBeta1 beta2 oneMinusBeta2
    firstMomentCorrection secondMomentCorrection epsilon
    (-(learningRate * weightDecay)) (-learningRate)

  Utils.assertTensorApprox (s := optimShape) "fused AdamW parameters"
    (← Utils.bufferToTensor gotParams) (← Utils.bufferToTensor expectedParams) (tol := 1e-6)
  Utils.assertTensorApprox (s := optimShape) "fused AdamW first moment"
    (← Utils.bufferToTensor gotM) (← Utils.bufferToTensor expectedM) (tol := 1e-6)
  Utils.assertTensorApprox (s := optimShape) "fused AdamW second moment"
    (← Utils.bufferToTensor gotV) (← Utils.bufferToTensor expectedV) (tol := 1e-6)

end Elementwise
end Cuda
end Tests

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.CUDA.LibTorch.Kernels
public import TorchLeanTest.Harness.CUDA

/-!
# CUDA kernel coverage: diagonal selective scan

The buffer primitives behind the Mamba/SSM runtime path, `Buffer.selectiveScanDiag` and
`Buffer.selectiveScanDiagBackward` with coefficients per channel and their `Var` forms with
coefficients per time step, against hand-computed values of the recurrence
`h[t,j] = A[j] * h[t-1,j] + B[j] * X[t,j]`. An empty sequence gives an empty output and zero
gradients of `A`, `B` and `h0`, and an empty state gives an empty output. The checks run under
`scripts/lake.sh -Kcuda=true test`.
-/

@[expose] public section

namespace TorchLean.Test
namespace CUDA
namespace SelectiveScan

open LibTorch

-- Shared buffer assertions check tolerances and reject nonfinite values.
open Harness (assertFloatArrayApprox)

/-- Checks `Buffer.selectiveScanDiag`, `Buffer.selectiveScanDiagBackward` and their `Var` forms
against hand-computed values within `1e-5`, the empty sequence and empty state, the order of the
float32 and float64 sums, and the signs of the gradients at `dY = -0`. -/
def run : IO Unit := do
  IO.println "== CUDA selective_scan_diag_fwd =="

  let A ← Buffer.ofFloatArrayIO (FloatArray.mk #[0.5, 0.25])
  let B ← Buffer.ofFloatArrayIO (FloatArray.mk #[1.0, 2.0])
  let h0 ← Buffer.ofFloatArrayIO (FloatArray.mk #[1.0, -1.0])
  let X ← Buffer.ofFloatArrayIO (FloatArray.mk #[
    2.0, 1.0,
    4.0, -2.0,
    0.0, 3.0
  ])

  let out := Buffer.selectiveScanDiag A B X h0 3 2
  let expected := FloatArray.mk #[
    2.5, 1.75,
    5.25, -3.5625,
    2.625, 5.109375
  ]
  assertFloatArrayApprox "selectiveScanDiagFwd" (Buffer.toFloatArray out) expected (tol := 1e-5)

  let dY ← Buffer.ofFloatArrayIO (FloatArray.mk #[
    1.0, 1.0,
    1.0, 1.0,
    1.0, 1.0
  ])
  let (dA, dB, dX, dH0) := Buffer.selectiveScanDiagBackward A B X h0 out dY 3 2
  assertFloatArrayApprox "selectiveScanDiagBwd.dA" (Buffer.toFloatArray dA)
    (FloatArray.mk #[10.75, -2.6875]) (tol := 1e-5)
  assertFloatArrayApprox "selectiveScanDiagBwd.dB" (Buffer.toFloatArray dB)
    (FloatArray.mk #[9.5, 1.8125]) (tol := 1e-5)
  assertFloatArrayApprox "selectiveScanDiagBwd.dX" (Buffer.toFloatArray dX)
    (FloatArray.mk #[
      1.75, 2.625,
      1.5, 2.5,
      1.0, 2.0
    ]) (tol := 1e-5)
  assertFloatArrayApprox "selectiveScanDiagBwd.dH0" (Buffer.toFloatArray dH0)
    (FloatArray.mk #[0.875, 0.328125]) (tol := 1e-5)

  let emptyX ← Buffer.ofFloatArrayIO (FloatArray.mk #[])
  let emptyOut := Buffer.selectiveScanDiag A B emptyX h0 0 2
  if Buffer.size emptyOut != 0 then
    throw <| IO.userError
      s!"selectiveScanDiagFwd empty seq: got size {Buffer.size emptyOut}, expected 0"

  let emptyParam ← Buffer.ofFloatArrayIO (FloatArray.mk #[])
  let zeroStateOut := Buffer.selectiveScanDiag emptyParam emptyParam emptyX emptyParam 3 0
  if Buffer.size zeroStateOut != 0 then
    throw <| IO.userError
      s!"selectiveScanDiagFwd zero state: got size {Buffer.size zeroStateOut}, expected 0"

  let Avar ← Buffer.ofFloatArrayIO (FloatArray.mk #[
    0.5, 0.25,
    0.1, 0.75,
    -1.0, 0.2
  ])
  let Bvar ← Buffer.ofFloatArrayIO (FloatArray.mk #[
    1.0, 2.0,
    0.5, -1.0,
    0.25, 0.5
  ])
  let Xvar ← Buffer.ofFloatArrayIO (FloatArray.mk #[
    2.0, 1.0,
    4.0, -2.0,
    0.0, 3.0
  ])
  let outVar := Buffer.selectiveScanDiagVar Avar Bvar Xvar h0 3 2
  let expectedVar := FloatArray.mk #[
    2.5, 1.75,
    2.25, 3.3125,
    -2.25, 2.1625
  ]
  assertFloatArrayApprox "selectiveScanDiagVarFwd" (Buffer.toFloatArray outVar) expectedVar
    (tol := 1e-5)

  -- Nonuniform cotangents expose time-index shifts in the variable-coefficient VJP.
  let dyVar ← Buffer.ofFloatArrayIO (FloatArray.mk #[1.0, -2.0, 3.0, 4.0, -1.0, 0.5])
  let (daVar, dbVar, dxVar, dhVar) :=
    Buffer.selectiveScanDiagVarBackward Avar Bvar Xvar h0 outVar dyVar 3 2
  assertFloatArrayApprox "selectiveScanDiagVarBwd.dA" (Buffer.toFloatArray daVar)
    (FloatArray.mk #[1.4, -1.075, 10.0, 7.175, -2.25, 1.65625]) (tol := 1e-5)
  assertFloatArrayApprox "selectiveScanDiagVarBwd.dB" (Buffer.toFloatArray dbVar)
    (FloatArray.mk #[2.8, 1.075, 16.0, -8.2, 0.0, 1.5]) (tol := 1e-5)
  assertFloatArrayApprox "selectiveScanDiagVarBwd.dX" (Buffer.toFloatArray dxVar)
    (FloatArray.mk #[1.4, 2.15, 2.0, -4.1, -0.25, 0.25]) (tol := 1e-5)
  assertFloatArrayApprox "selectiveScanDiagVarBwd.dH0" (Buffer.toFloatArray dhVar)
    (FloatArray.mk #[0.7, 0.26875]) (tol := 1e-5)

  let (daEmpty, dbEmpty, dxEmpty, dhEmpty) :=
    Buffer.selectiveScanDiagBackward A B emptyX h0 emptyOut emptyX 0 2
  for (label, buffer, expected) in [
      ("dA", daEmpty, FloatArray.mk #[0.0, 0.0]),
      ("dB", dbEmpty, FloatArray.mk #[0.0, 0.0]),
      ("dX", dxEmpty, FloatArray.mk #[]),
      ("dH0", dhEmpty, FloatArray.mk #[0.0, 0.0])] do
    assertFloatArrayApprox s!"selectiveScanDiagBwd.empty.{label}"
      (Buffer.toFloatArray buffer) expected (tol := 0.0)

  -- Left to right, `big + -big` cancels before `1` is added. A doubling scan adds `-big + 1`
  -- first, which rounds to `-big`, and returns `0` at the last step.
  for (dtype, big) in [(Dtype.float32, 67108864.0), (Dtype.float64, 36028797018963968.0)] do
    let one ← Buffer.ofFloatArrayIO (FloatArray.mk #[1.0]) dtype
    let zero ← Buffer.ofFloatArrayIO (FloatArray.mk #[0.0]) dtype
    let inputs ← Buffer.ofFloatArrayIO (FloatArray.mk #[big, -big, 1.0]) dtype
    let states := Buffer.toFloatArray (Buffer.selectiveScanDiag one one inputs zero 3 1)
    unless states.data.map Float.toBits == #[big, 0.0, 1.0].map Float.toBits do
      throw <| IO.userError s!"selectiveScanDiagFwd ({repr dtype}) order: got {states.data}"

  -- With `A = 0` each adjoint is its cotangent, so the shared `dA` adds `[1, 1, big]` and the
  -- shared `dB` adds `[1, big, -big]`. From `+0` in increasing `t` these give `big + 2` and `+0`.
  -- In reverse time `big + 1` rounds to `big` twice, and `-big + big` cancels before `1` is added.
  for (dtype, big, total) in [(Dtype.float32, 16777216.0, 16777218.0),
      (Dtype.float64, 9007199254740992.0, 9007199254740994.0)] do
    let zero ← Buffer.ofFloatArrayIO (FloatArray.mk #[0.0]) dtype
    let one ← Buffer.ofFloatArrayIO (FloatArray.mk #[1.0]) dtype
    let inputs ← Buffer.ofFloatArrayIO (FloatArray.mk #[1.0, big, -big]) dtype
    let ones ← Buffer.ofFloatArrayIO (FloatArray.mk #[1.0, 1.0, 1.0]) dtype
    let states := Buffer.selectiveScanDiag zero one inputs one 3 1
    let (gradA, gradB, _, _) := Buffer.selectiveScanDiagBackward zero one inputs one states ones 3 1
    let sums :=
      ((Buffer.toFloatArray gradA).data ++ (Buffer.toFloatArray gradB).data).map Float.toBits
    unless sums == #[total, 0.0].map Float.toBits do
      throw <| IO.userError s!"selectiveScanDiagBwd ({repr dtype}) shared order: got bits {sums}"

  -- The last adjoint is the last cotangent itself, so `dY = [-0]` reaches `dX` and `dH0` as
  -- `-0`. The shared `dA` and `dB` add it to `+0` and give `+0`.
  let one ← Buffer.ofFloatArrayIO (FloatArray.mk #[1.0])
  let negativeZero ← Buffer.ofFloatArrayIO (FloatArray.mk #[-0.0])
  let single := Buffer.selectiveScanDiag one one one one 1 1
  let (sharedA, sharedB, sharedX, sharedH0) :=
    Buffer.selectiveScanDiagBackward one one one one single negativeZero 1 1
  let shared := #[sharedA, sharedB, sharedX, sharedH0].map fun b =>
    (Buffer.toFloatArray b).data.map Float.toBits
  unless shared == #[#[0], #[0], #[0x8000000000000000], #[0x8000000000000000]] do
    throw <| IO.userError s!"selectiveScanDiagBwd at dY = -0: got bits {shared}"
  let singleVar := Buffer.selectiveScanDiagVar one one one one 1 1
  let (stepA, stepB, stepX, stepH0) :=
    Buffer.selectiveScanDiagVarBackward one one one one singleVar negativeZero 1 1
  let perStep := #[stepA, stepB, stepX, stepH0].map fun b =>
    (Buffer.toFloatArray b).data.map Float.toBits
  unless perStep == #[#[0x8000000000000000], #[0x8000000000000000], #[0x8000000000000000],
      #[0x8000000000000000]] do
    throw <| IO.userError s!"selectiveScanDiagVarBwd at dY = -0: got bits {perStep}"

  IO.println "== CUDA selective_scan_diag_fwd: OK =="

end SelectiveScan
end CUDA
end TorchLean.Test

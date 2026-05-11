/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Buffer
public import NN.Runtime.Autograd.Engine.Cuda.Kernels

/-!
# CUDA kernel coverage: diagonal selective scan

This exercises the low-level buffer primitive backing the first Mamba/SSM runtime path.  The test
runs both with real CUDA (`lake test -K cuda=true`) and with the CPU stub backend.
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace SelectiveScan

open Runtime.Autograd.Cuda

def floatArrayOfList (xs : List Float) : FloatArray :=
  xs.foldl (fun acc x => acc.push x) (FloatArray.emptyWithCapacity xs.length)

def assertFloatArrayApprox (msg : String) (a b : FloatArray) (tol : Float := 1e-5) : IO Unit := do
  if a.size != b.size then
    throw <| IO.userError s!"{msg}: size mismatch ({a.size} vs {b.size})"
  for i in [:a.size] do
    let x := a.get! i
    let y := b.get! i
    let d := if x > y then x - y else y - x
    if d > tol then
      throw <| IO.userError s!"{msg}[{i}]: got {x}, expected {y}, |diff|={d}"

def run : IO Unit := do
  IO.println "== CUDA selective_scan_diag_fwd =="

  let A := Buffer.ofFloatArray (floatArrayOfList [0.5, 0.25])
  let B := Buffer.ofFloatArray (floatArrayOfList [1.0, 2.0])
  let h0 := Buffer.ofFloatArray (floatArrayOfList [1.0, -1.0])
  let X := Buffer.ofFloatArray (floatArrayOfList [
    2.0, 1.0,
    4.0, -2.0,
    0.0, 3.0
  ])

  let out := Buffer.selectiveScanDiagFwd A B X h0 3 2
  let expected := floatArrayOfList [
    2.5, 1.75,
    5.25, -3.5625,
    2.625, 5.109375
  ]
  assertFloatArrayApprox "selectiveScanDiagFwd" (Buffer.toFloatArray out) expected

  let dY := Buffer.ofFloatArray (floatArrayOfList [
    1.0, 1.0,
    1.0, 1.0,
    1.0, 1.0
  ])
  let (dA, dB, dX, dH0) := Buffer.selectiveScanDiagBwd A B X h0 out dY 3 2
  assertFloatArrayApprox "selectiveScanDiagBwd.dA" (Buffer.toFloatArray dA)
    (floatArrayOfList [10.75, -2.6875])
  assertFloatArrayApprox "selectiveScanDiagBwd.dB" (Buffer.toFloatArray dB)
    (floatArrayOfList [9.5, 1.8125])
  assertFloatArrayApprox "selectiveScanDiagBwd.dX" (Buffer.toFloatArray dX)
    (floatArrayOfList [
      1.75, 2.625,
      1.5, 2.5,
      1.0, 2.0
    ])
  assertFloatArrayApprox "selectiveScanDiagBwd.dH0" (Buffer.toFloatArray dH0)
    (floatArrayOfList [0.875, 0.328125])

  let emptyX := Buffer.ofFloatArray (floatArrayOfList [])
  let emptyOut := Buffer.selectiveScanDiagFwd A B emptyX h0 0 2
  if Buffer.size emptyOut != 0 then
    throw <| IO.userError s!"selectiveScanDiagFwd empty seq: got size {Buffer.size emptyOut}, expected 0"

  let emptyParam := Buffer.ofFloatArray (floatArrayOfList [])
  let zeroStateOut := Buffer.selectiveScanDiagFwd emptyParam emptyParam emptyX emptyParam 3 0
  if Buffer.size zeroStateOut != 0 then
    throw <| IO.userError s!"selectiveScanDiagFwd zero state: got size {Buffer.size zeroStateOut}, expected 0"

  let Avar := Buffer.ofFloatArray (floatArrayOfList [
    0.5, 0.25,
    0.1, 0.75,
    -1.0, 0.2
  ])
  let Bvar := Buffer.ofFloatArray (floatArrayOfList [
    1.0, 2.0,
    0.5, -1.0,
    0.25, 0.5
  ])
  let Xvar := Buffer.ofFloatArray (floatArrayOfList [
    2.0, 1.0,
    4.0, -2.0,
    0.0, 3.0
  ])
  let outVar := Buffer.selectiveScanDiagVarFwd Avar Bvar Xvar h0 3 2
  let expectedVar := floatArrayOfList [
    2.5, 1.75,
    2.25, 3.3125,
    -2.25, 2.1625
  ]
  assertFloatArrayApprox "selectiveScanDiagVarFwd" (Buffer.toFloatArray outVar) expectedVar

  IO.println "== CUDA selective_scan_diag_fwd: OK =="

end SelectiveScan
end Cuda
end Tests

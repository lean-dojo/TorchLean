/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Buffer
public import NN.Runtime.Autograd.Engine.Cuda.Kernels
public import NN.Tests.Runtime.Cuda.Utils

/-!
# CUDA kernel coverage: diagonal selective scan

This checks the low-level buffer primitive backing the first Mamba/SSM runtime path. The test runs
both with real CUDA (`lake test -K cuda=true`) and with the CPU stub backend.
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace SelectiveScan

open Runtime.Autograd.Cuda

/-!
The float-buffer assertions and the `floatArray` literal wrapper come from `Cuda.Utils`, where the
rest of this directory already gets them. This file used to define its own `assertFloatArrayApprox`
that open-coded the absolute difference. The shared one goes through `Utils.assertApprox`, so these
checks now also reject `NaN` and infinities: with `d := NaN` the old `d > tol` test was false, which
means a kernel returning `NaN` used to pass this suite silently.
-/
open Tests.Cuda.Utils (floatArray assertFloatArrayApprox)

def run : IO Unit := do
  IO.println "== CUDA selective_scan_diag_fwd =="

  let A := Buffer.ofFloatArray (floatArray #[0.5, 0.25])
  let B := Buffer.ofFloatArray (floatArray #[1.0, 2.0])
  let h0 := Buffer.ofFloatArray (floatArray #[1.0, -1.0])
  let X := Buffer.ofFloatArray (floatArray #[
    2.0, 1.0,
    4.0, -2.0,
    0.0, 3.0
  ])

  let out := Buffer.selectiveScanDiagFwd A B X h0 3 2
  let expected := floatArray #[
    2.5, 1.75,
    5.25, -3.5625,
    2.625, 5.109375
  ]
  assertFloatArrayApprox "selectiveScanDiagFwd" (Buffer.toFloatArray out) expected (tol := 1e-5)

  let dY := Buffer.ofFloatArray (floatArray #[
    1.0, 1.0,
    1.0, 1.0,
    1.0, 1.0
  ])
  let (dA, dB, dX, dH0) := Buffer.selectiveScanDiagBwd A B X h0 out dY 3 2
  assertFloatArrayApprox "selectiveScanDiagBwd.dA" (Buffer.toFloatArray dA)
    (floatArray #[10.75, -2.6875]) (tol := 1e-5)
  assertFloatArrayApprox "selectiveScanDiagBwd.dB" (Buffer.toFloatArray dB)
    (floatArray #[9.5, 1.8125]) (tol := 1e-5)
  assertFloatArrayApprox "selectiveScanDiagBwd.dX" (Buffer.toFloatArray dX)
    (floatArray #[
      1.75, 2.625,
      1.5, 2.5,
      1.0, 2.0
    ]) (tol := 1e-5)
  assertFloatArrayApprox "selectiveScanDiagBwd.dH0" (Buffer.toFloatArray dH0)
    (floatArray #[0.875, 0.328125]) (tol := 1e-5)

  let emptyX := Buffer.ofFloatArray (floatArray #[])
  let emptyOut := Buffer.selectiveScanDiagFwd A B emptyX h0 0 2
  if Buffer.size emptyOut != 0 then
    throw <| IO.userError
      s!"selectiveScanDiagFwd empty seq: got size {Buffer.size emptyOut}, expected 0"

  let emptyParam := Buffer.ofFloatArray (floatArray #[])
  let zeroStateOut := Buffer.selectiveScanDiagFwd emptyParam emptyParam emptyX emptyParam 3 0
  if Buffer.size zeroStateOut != 0 then
    throw <| IO.userError
      s!"selectiveScanDiagFwd zero state: got size {Buffer.size zeroStateOut}, expected 0"

  let Avar := Buffer.ofFloatArray (floatArray #[
    0.5, 0.25,
    0.1, 0.75,
    -1.0, 0.2
  ])
  let Bvar := Buffer.ofFloatArray (floatArray #[
    1.0, 2.0,
    0.5, -1.0,
    0.25, 0.5
  ])
  let Xvar := Buffer.ofFloatArray (floatArray #[
    2.0, 1.0,
    4.0, -2.0,
    0.0, 3.0
  ])
  let outVar := Buffer.selectiveScanDiagVarFwd Avar Bvar Xvar h0 3 2
  let expectedVar := floatArray #[
    2.5, 1.75,
    2.25, 3.3125,
    -2.25, 2.1625
  ]
  assertFloatArrayApprox "selectiveScanDiagVarFwd" (Buffer.toFloatArray outVar) expectedVar
    (tol := 1e-5)

  IO.println "== CUDA selective_scan_diag_fwd: OK =="

end SelectiveScan
end Cuda
end Tests

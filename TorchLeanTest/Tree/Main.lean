/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

import TorchLean.CUDA.LibTorch.Buffer
import TorchLeanTest.Tree.Suite

/-!
Entry point for `scripts/lake.sh exe tree_tests`: the host checks of tree arenas, derived
signatures, folds, folds compiled from functions, hylomorphisms, unfolds, scalars held as their
encodings and unary functions as kernel tables, then the device checks when LibTorch and a device
are available, or the checks of builds without LibTorch. With
`TORCHLEAN_REQUIRE_CUDA=1`, a build or machine without a usable CUDA device is an error. On the host
bridge the device checks are skipped: tree kernels are generated CUDA, and `Hylo.run` and
`Unfold.run` take their host references there.
-/

/-- Run the host checks of tree arenas, derived signatures, folds, folds compiled from functions,
hylomorphisms, unfolds, scalars held as their encodings and unary functions as kernel tables, and
the device checks the build supports. -/
public def main : IO Unit := do
  if (← IO.getEnv "TORCHLEAN_REQUIRE_CUDA") == some "1" then
    TorchLean.LibTorch.Buffer.requireCudaDevice
  TorchLean.Test.Tree.run
  match ← TorchLean.LibTorch.Buffer.RuntimeStatus.query with
  | .notLinked => TorchLean.Test.Tree.runUnlinked
  | .nativeAvailable =>
    if (← TorchLean.LibTorch.deviceKind) == .cuda then TorchLean.Test.Tree.runCUDA
    else IO.println "  tree device checks: skipped (the bridge runs on the host)"
  | .nativeUnavailable =>
    throw <| IO.userError "TorchLean was built with CUDA, but no usable CUDA device is visible"

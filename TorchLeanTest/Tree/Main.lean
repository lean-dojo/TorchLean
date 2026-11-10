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
`TORCHLEAN_REQUIRE_CUDA=1`, a build or machine without a usable device is an error.
-/

/-- Run the host checks of tree arenas, derived signatures, folds, folds compiled from functions,
hylomorphisms, unfolds, scalars held as their encodings and unary functions as kernel tables, and
the device checks the build supports. -/
public def main : IO Unit := do
  if (← IO.getEnv "TORCHLEAN_REQUIRE_CUDA") == some "1" then
    TorchLean.LibTorch.Buffer.requireNativeRuntime
  TorchLean.Test.Tree.run
  match ← TorchLean.LibTorch.Buffer.RuntimeStatus.query with
  | .notLinked => TorchLean.Test.Tree.runUnlinked
  | .nativeAvailable => TorchLean.Test.Tree.runCUDA
  | .nativeUnavailable =>
    throw <| IO.userError "TorchLean was built with CUDA, but no usable CUDA device is visible"

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLeanTest.Tree.Derive
public import TorchLeanTest.Tree.Elab
public import TorchLeanTest.Tree.Hylo
public import TorchLeanTest.Tree.Table
public import TorchLeanTest.Tree.Unfold
public import TorchLeanTest.Tree.Word

/-!
# Tree checks

The checks of tree arenas, derived signatures, folds, folds compiled from functions,
hylomorphisms, unfolds, scalars held as their encodings and unary functions as kernel tables, in
three entry points: the host checks, the device checks for builds with LibTorch and a visible
device, and the checks of builds without LibTorch.
-/

@[expose] public section

namespace TorchLean.Test.Tree

/-- The host checks of tree arenas, derived signatures, folds, folds compiled from functions,
hylomorphisms, unfolds, scalars held as their encodings and unary functions as kernel tables. -/
def run : IO Unit := do
  Arenas.run
  Derived.run
  Folds.run
  Elaborated.run
  Hylos.run
  Unfolds.run
  Words.run
  Tables.run

/-- The device checks of tree arenas, derived signatures, folds, folds compiled from functions,
hylomorphisms, unfolds, scalars held as their encodings and unary functions as kernel tables. -/
def runCUDA : IO Unit := do
  Arenas.runCUDA
  Derived.runCUDA
  Folds.runCUDA
  Elaborated.runCUDA
  Hylos.runCUDA
  Unfolds.runCUDA
  Words.runCUDA
  Tables.runCUDA

/-- The checks of tree arenas, derived signatures, hylomorphisms, unfolds and kernel tables in
builds without LibTorch. -/
def runUnlinked : IO Unit := do
  Arenas.runUnlinked
  Derived.runUnlinked
  Hylos.runUnlinked
  Unfolds.runUnlinked
  Tables.runUnlinked

end TorchLean.Test.Tree

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.LibTorch.Ops.Core

/-!
# Native diagonal scan tape nodes

Both coefficient layouts return all states after their updates. Their backward closures retain
the forward states and accumulate into all four parents. The variable-coefficient kernel returns
one gradient per token; the shared-coefficient kernel sums those contributions across time.
The native implementation uses a parallel affine prefix scan for eligible inputs, with a
sequential recurrence fallback for expansive or nonfinite inputs and nonfinite parallel results.
Reassociation can change floating-point rounding; the capsule leaves reduction order unspecified.
-/

@[expose] public section

namespace Runtime.Autograd.LibTorch.Tape

open Spec

namespace Internal

/-- Common recording and shape checks for the two diagonal-scan coefficient layouts. -/
@[inline] def diagonalScan (variableCoefficients : Bool) {seqLen state : Nat}
    (t : Tape SomeBuffer) (aId bId xId initialId : Nat) : Result (Tape SomeBuffer × Nat) := do
  let time32 ← SomeBuffer.natToU32Checked seqLen
  let state32 ← SomeBuffer.natToU32Checked state
  let sequenceShape : Shape := [seqLen, state]
  let coefficientShape : Shape := if variableCoefficients then sequenceShape else [state]
  let a ← requireValue t aId coefficientShape
  let b ← requireValue t bId coefficientShape
  let x ← requireValue t xId sequenceShape
  let initial ← requireValue t initialId [state]
  let output := if variableCoefficients
    then Buffer.selectiveScanDiagVarFwd a b x initial time32 state32
    else Buffer.selectiveScanDiagFwd a b x initial time32 state32
  let node : Node SomeBuffer :=
    { name := some (if variableCoefficients then "selectiveScanDiagVar" else "selectiveScanDiag")
      value := { s := sequenceShape, buf := output }
      requiresGrad := (t.getNode? aId).any (·.requiresGrad) ||
        (t.getNode? bId).any (·.requiresGrad) ||
        (t.getNode? xId).any (·.requiresGrad) ||
        (t.getNode? initialId).any (·.requiresGrad)
      parents := #[aId, bId, xId, initialId]
      backward := fun gradient => do
        let checked ← requireGrad gradient sequenceShape
        let (da, db, dx, dInitial) := if variableCoefficients
          then Buffer.selectiveScanDiagVarBwd a b x initial output checked.buf time32 state32
          else Buffer.selectiveScanDiagBwd a b x initial output checked.buf time32 state32
        pure #[
          (aId, { s := coefficientShape, buf := da }),
          (bId, { s := coefficientShape, buf := db }),
          (xId, { s := sequenceShape, buf := dx }),
          (initialId, { s := [state], buf := dInitial })] }
  return t.addNode node

end Internal

/-- Record the recurrence with coefficient vectors shared across time. -/
@[inline] def selectiveScanDiag {seqLen state : Nat} (t : Tape SomeBuffer) (aId bId xId initialId
    : Nat) :
    Result (Tape SomeBuffer × Nat) :=
  Internal.diagonalScan false (seqLen := seqLen) (state := state) t aId bId xId initialId

/-- Record the recurrence with independent coefficient rows for every token. -/
@[inline] def selectiveScanDiagVar {seqLen state : Nat} (t : Tape SomeBuffer) (aId bId xId initialId
    : Nat) :
    Result (Tape SomeBuffer × Nat) :=
  Internal.diagonalScan true (seqLen := seqLen) (state := state) t aId bId xId initialId

end Runtime.Autograd.LibTorch.Tape

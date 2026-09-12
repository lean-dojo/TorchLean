/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Ops.Core

/-!
# Native diagonal scan tape nodes

Both coefficient layouts return all states after their updates. Their backward closures retain
the forward states and accumulate into all four parents. The variable-coefficient kernel returns
one gradient per token; the shared-coefficient kernel sums those contributions across time.
Each native thread owns one state channel and traverses time in order.
-/

@[expose] public section

namespace Runtime.Autograd.Cuda.Tape

open Spec

namespace Internal

/-- Common recording and shape checks for the two diagonal-scan coefficient layouts. -/
def diagonalScan (variableCoefficients : Bool) {seqLen state : Nat}
    (t : Tape) (aId bId xId initialId : Nat) : Result (Tape × Nat) := do
  let time32 ← AnyBuffer.natToU32Checked seqLen
  let state32 ← AnyBuffer.natToU32Checked state
  let sequenceShape : Shape := [seqLen, state]
  let coefficientShape : Shape := if variableCoefficients then sequenceShape else [state]
  let a ← t.requireValue aId coefficientShape
  let b ← t.requireValue bId coefficientShape
  let x ← t.requireValue xId sequenceShape
  let initial ← t.requireValue initialId [state]
  let output := if variableCoefficients
    then Buffer.selectiveScanDiagVarFwd a b x initial time32 state32
    else Buffer.selectiveScanDiagFwd a b x initial time32 state32
  let node : Node :=
    { name := some (if variableCoefficients then "selectiveScanDiagVar" else "selectiveScanDiag")
      value := { s := sequenceShape, buf := output }
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
def selectiveScanDiag {seqLen state : Nat} (t : Tape) (aId bId xId initialId : Nat) :
    Result (Tape × Nat) :=
  Internal.diagonalScan false (seqLen := seqLen) (state := state) t aId bId xId initialId

/-- Record the recurrence with independent coefficient rows for every token. -/
def selectiveScanDiagVar {seqLen state : Nat} (t : Tape) (aId bId xId initialId : Nat) :
    Result (Tape × Nat) :=
  Internal.diagonalScan true (seqLen := seqLen) (state := state) t aId bId xId initialId

end Runtime.Autograd.Cuda.Tape

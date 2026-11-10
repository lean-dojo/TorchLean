/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.Report
public import TorchLean.NN.Trainer.Results
public import TorchLean.NN.Sequential.Execution -- shake: keep
public import TorchLean.Verify.CROWN.Arithmetic -- shake: keep

/-!
# Running `trained.verify`

The public `trained.verify` operation, `Trainer.Result.verify`, and the code behind it. The result's
lowered model goes to `Verification.forLowered`, which checks the center and radius, encloses the
requested ball in a box of the model's scalar type, runs the chosen algorithm and returns the output
box as a validated `Verification.Report`. On a rounded scalar `alphaCROWN` runs directed CROWN, and
the report names `crown`, the algorithm that ran.

The bounds hold for the recorded verifier graph. Native slices and selections copy source entries
unchanged, but no finite-precision theorem identifies the whole lowered graph with the source model,
so applying a report to the original execution needs that correspondence argument.

A call reads `trained.verify center (radius := r)`, and the input region is the L∞ ball of radius
`r` around `center`. Verifier graphs are built directly with `TorchLean.Verify.Lowering`.
-/

@[expose] public section

namespace TorchLean

namespace Verification

/-- Read a box of the runtime scalar back as the host-`Float` box that a `Report` carries. -/
def readBounds {α : Type} [TorchLean.Storage α] [Context α]
    [TensorTransfer α]
    (box : CROWN.FlatBox α) : IO (CROWN.FlatBox Float) := do
  let lo ← TensorTransfer.readFloatTensor box.lo
  let hi ← TensorTransfer.readFloatTensor box.hi
  pure { dim := box.dim, lo, hi }

/--
Enclose the requested binary64 ball, then convert its endpoints to the runtime scalar.

Converting the center and the radius first and subtracting afterwards could shrink the region,
most of all when the subtraction cancels most of the center. Instead the binary64 box is rounded
outward, its lower endpoints convert toward `-∞` and its upper endpoints toward `+∞`, so the result
contains the binary64 ball.
-/
def inputBoxFromFloat {α : Type} [TorchLean.Storage α] [Context α]
    [FromFloat α] {σ : Shape}
    (center : Tensor Float σ) (radius : Float) : CROWN.FlatBox α :=
  let box := CROWN.FlatBox.lInfBall center radius
  { dim := box.dim
    lo := Tensor.map (fun x => ofFloat (α := α) x (mode := .down)) box.lo
    hi := Tensor.map (fun x => ofFloat (α := α) x (mode := .up)) box.hi }

/--
Verify the lowering `lowered` of a trained model over the L∞ ball of radius `radius` around
`center`.

The bounds are computed on the lowered graph's import as a graph of registered operators
(`IR.LoweredModel.toImported`, `CROWN.Op`), so applying them to the source model needs a
separate correspondence argument. The report names the algorithm that ran: `alphaCROWN` on a
rounded scalar runs directed CROWN, so the report says `crown`.
-/
def forLowered {σ : Shape} {α : Type}
    [TorchLean.Storage α] [Context α]
    [FromFloat α] [TensorTransfer α]
    [CROWN.BoundOps α]
    [CROWN.NonlinearBoundOps α]
    (lowered : Thunk (Except String (IR.Imported α))) :
    (center : Tensor Float σ) →
    (radius : Float) →
    (property : Property) →
    (algorithm : Algorithm) →
    IO Report :=
  fun centerFloat radius property algorithm => do
    IO.ofExcept (validateRadius radius)
    IO.ofExcept (validateCenter centerFloat)

    let imported ← IO.ofExcept lowered.get
    let inputBox := inputBoxFromFloat (α := α) centerFloat radius
    unless CROWN.Op.validInputs imported.graph [inputBox] do
      throw <| IO.userError
        s!"verification center of {σ.size} entries does not fit the graph inputs"
    let outputBox ←
    match algorithm with
      | .ibp =>
          IO.ofExcept (imported.outputBox? (imported.runIBP inputBox))
      | .crown =>
          IO.ofExcept (imported.outputBoxCROWN? inputBox)
      | .alphaCROWN =>
          IO.ofExcept (CROWN.Op.outputBoxAlphaCROWN? imported.graph imported.payload inputBox 0)

    let ran : Algorithm :=
      if algorithm == .alphaCROWN &&
          !CROWN.BoundOps.exactReassociation (α := α) then
        .crown
      else
        algorithm
    let bounds ← readBounds outputBox
    IO.ofExcept <| Report.fromBounds radius bounds
      (property := property)
      (algorithm := ran)

end Verification

/--
Verify the trained model over the L∞ ball of radius `radius` around `center`.

`property` and `algorithm` have defaults, so checking output bounds needs only `center` and
`radius`. It throws a user error when the result carries no lowered model, as results of
`openTyped` sessions do.
-/
opaque Trainer.Result.verify {α : Type} [Storage α] {σ τ : Shape}
    (result : Trainer.Result σ τ α)
    (center : Tensor Float σ)
    (radius : Float)
    (property : Verification.Property := .outputBounds)
    (algorithm : Verification.Algorithm := .crown) :
    IO Verification.Report :=
  match Trainer.Result.lowered? result with
  | some (.native lowered) =>
      Verification.forLowered lowered center radius property algorithm
  | some (.ieee lowered) =>
      Verification.forLowered lowered center radius property algorithm
  | none => throw <| IO.userError "verification is unavailable for this result's scalar/runtime"

end TorchLean

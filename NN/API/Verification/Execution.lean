/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Verification.Core
public import NN.API.Trainer.Core
public import NN.MLTheory.CROWN.Cert.AlphaBetaCROWN
public import NN.API.Neural.Execution -- shake: keep
public import NN.Verification.Builtin.Lowering -- shake: keep

/-!
# Verification Execution

Implementation support for the public `trained.verify` operation. The public named arguments are
lowered into a verifier input region and returned as a validated `Verification.Report`.

Applications should import `NN.API.Verification` and call
`trained.verify center (radius := r) (norm := .inf)`. Direct verifier graph construction remains
in `NN.API.Verification.Lowering`.
-/

@[expose] public section

namespace TorchLean

namespace Verification

namespace Internal

/-- Convert an internal flat interval into the public host-Float bounds representation. -/
def readBounds {α : Type} [TorchLean.Storage α] [Context α]
    [Runtime.TensorTransfer α]
    (box : NN.MLTheory.CROWN.FlatBox α) : IO Bounds := do
  let lower ← Runtime.readFloatTensor box.lo
  let upper ← Runtime.readFloatTensor box.hi
  pure
    { size := box.dim, lower, upper }

/--
Build the verification closure retained by an ordinary trained result.

The closure captures the completed parameter snapshot, so subsequent updates to the training
session cannot change the model being verified.
-/
def forState {σ τ : Shape} {α : Type}
    [TorchLean.Storage α] [Context α]
    [Runtime.FromFloat α] [Runtime.TensorTransfer α]
    [NN.MLTheory.CROWN.BoundOps α]
    [NN.MLTheory.CROWN.NonlinearBoundOps α]
    (trainer : TorchLean.Trainer σ τ)
    (modelState : nn.State α (nn.stateShapes trainer.model)) :
    (center : Tensor Float σ) →
    (radius : Float) →
    (norm : Norm) →
    (property : Property) →
    (algorithm : Algorithm) →
    IO Report :=
  fun centerFloat radius norm property algorithm => do
    match validateRadius radius with
    | .ok () => pure ()
    | .error message => throw <| IO.userError message

    match norm with
    | .inf => pure ()
    | .one =>
        throw <| IO.userError
          "L1 verification is not implemented by the current box-based verifier; use norm := .inf"
    | .two =>
        throw <| IO.userError
          "L2 verification is not implemented by the current box-based verifier; use norm := .inf"

    let lowered ←
      match NN.Verification.Builtin.lowerForwardToIR
          (TorchLean.nn.forward trainer.model (α := α))
          (nn.State.Internal.toTensorPack modelState) with
      | .ok result => pure result
      | .error message => throw <| IO.userError message

    let center := Tensor.map (Runtime.ofFloat (α := α)) centerFloat
    let regionRadius := Runtime.ofFloat (α := α) radius
    let inputBox := NN.Verification.Builtin.lInfBall center regionRadius
    let parameters := lowered.seedInputBox inputBox
    let outputBox ←
    match algorithm with
      | .ibp =>
          lowered.outputBoxOrThrow (lowered.runIBP parameters)
      | .crown =>
          lowered.outputBoxCROWNOrThrow parameters inputBox
      | .alphaBetaCrown =>
          let inputDim ←
            match lowered.inputDim? with
            | .ok dim => pure dim
            | .error message => throw <| IO.userError message
          match NN.MLTheory.CROWN.Cert.outputBoxAlphaBetaCROWN?
              (α := α) lowered.graph parameters inputBox
              lowered.inputId lowered.outputId inputDim with
          | .ok box => pure box
          | .error message => throw <| IO.userError message

    let bounds ← readBounds outputBox
    match Report.fromBounds radius bounds
        (norm := norm)
        (property := property)
        (algorithm := algorithm) with
    | .ok report => pure report
    | .error message => throw <| IO.userError message

end Internal

end Verification

end TorchLean

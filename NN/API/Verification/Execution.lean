/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Verification.Core
public import NN.API.Trainer.Core
public import NN.MLTheory.CROWN.Cert.PhasedCROWN
public import NN.API.Neural.Execution -- shake: keep
public import NN.Verification.Builtin.Lowering -- shake: keep

/-!
# Verification Execution

Implementation support for the public `trained.verify` operation. The public named arguments are
lowered into a verifier input region and returned as a validated `Verification.Report`.
The bounds concern the recorded verifier graph. The broad lowering does not attach a
finite-precision source-equivalence theorem: one-hot slicing and leading-axis selection can
change signed zeros and exceptional values, including in multi-sample attention. Applying a
report to the original execution needs that separate correspondence argument.

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
Enclose the requested binary64 ball before converting its endpoints to the runtime scalar.

Rounding the center and radius separately can shrink the region, especially when subtraction
cancels most of the center. The directed conversion requires a faithful `roundForValidation`
hook, supplied by both arithmetic backends used by ordinary trained results.
-/
def inputBoxFromFloat {α : Type} [TorchLean.Storage α] [Context α]
    [Runtime.FromFloat α] {σ : Shape}
    (center : Tensor Float σ) (radius : Float) : NN.MLTheory.CROWN.FlatBox α :=
  let box := NN.Verification.Builtin.lInfBall center radius
  { dim := box.dim
    lo := Tensor.map (Runtime.ofFloatDirected (α := α) false) box.lo
    hi := Tensor.map (Runtime.ofFloatDirected (α := α) true) box.hi }

/--
Build the verification closure retained by an ordinary trained result.

The closure captures the completed parameter snapshot, so subsequent updates to the training
session cannot change the model being verified.
The returned report describes the lowered graph; it does not resolve the source-correspondence
restriction on its one-hot slicing and selection operations. It names the algorithm that ran:
`.alphaCrown` on a rounded scalar runs directed CROWN, so the report says `.crown`.
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
    IO.ofExcept (validateRadius radius)

    match norm with
    | .inf => pure ()
    | .one =>
        throw <| IO.userError
          "L1 verification is not implemented by the current box-based verifier; use norm := .inf"
    | .two =>
        throw <| IO.userError
          "L2 verification is not implemented by the current box-based verifier; use norm := .inf"

    let lowered ← IO.ofExcept <|
      NN.Verification.Builtin.lowerForwardToIR
        (TorchLean.nn.forward trainer.model (α := α))
        (nn.State.Internal.toTensorPack modelState)

    let inputBox := inputBoxFromFloat (α := α) centerFloat radius
    let parameters := lowered.seedInputBox inputBox
    let outputBox ←
    match algorithm with
      | .ibp =>
          IO.ofExcept (lowered.outputBox? (lowered.runIBP parameters))
      | .crown =>
          IO.ofExcept (lowered.outputBoxCROWN? parameters inputBox)
      | .alphaCrown =>
          let inputDim ← IO.ofExcept lowered.inputDim?
          IO.ofExcept <| NN.MLTheory.CROWN.Cert.outputBoxPhasedCROWN?
            (α := α) lowered.graph parameters inputBox
            lowered.inputId lowered.outputId inputDim

    let ran : Algorithm :=
      if algorithm == .alphaCrown &&
          !NN.MLTheory.CROWN.BoundOps.supportsExactAffineReassociation (α := α) then
        .crown
      else
        algorithm
    let bounds ← readBounds outputBox
    IO.ofExcept <| Report.fromBounds radius bounds
      (norm := norm)
      (property := property)
      (algorithm := ran)

end Internal

end Verification

end TorchLean

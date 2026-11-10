/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Sample
public import TorchLean.Verify.CROWN.Op.ExecCROWN
public import TorchLeanExamples.Verification.Lyapunov.Sampling
public import TorchLean.NN.Autodiff
public import TorchLean.Verify.CROWN.Arithmetic

/-!
# Lowered Loss Analysis for Two-Stage Lyapunov Workflows

The executable operations of the all-in-Lean pipeline: projected gradient ascent on the lowered
loss, and a report that runs IBP and CROWN on that loss over an input box. The report prints the
IBP and CROWN bounds and decides nothing.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


open TorchLean.Tensor

namespace TorchLean.Examples.Lyapunov

open CROWN

local notation "Scalar" => FloatLib.Floats.Binary32

/-- Shapes supplied to the lowered scalar loss: parameters followed by one state vector. -/
abbrev lossInputs (width : Nat) : List Shape := paramShapes width ++ [xShape]

/--
One projected gradient-ascent step on the state input of a lowered scalar loss: the state moves to
`state + stepSize • ∇ₓ loss`, clamped coordinatewise to `[-radius, radius]`.

The parameters are held fixed, and their gradients are discarded.
-/
def projectedGradientAscentStep
    (width : Nat)
    (lossGraph : TypedGraph Scalar (lossInputs width) [])
    (parameters : TensorPack Scalar (paramShapes width))
    (state : Tensor Scalar xShape)
    (stepSize radius : Scalar) : Tensor Scalar xShape :=
  let arguments : TensorPack Scalar (lossInputs width) :=
    Pack.append
      (ss₁ := paramShapes width) (ss₂ := [xShape])
      parameters
      (Pack.singleton state)
  let allGradients : TensorPack Scalar (lossInputs width) :=
    TypedGraph.WithData.vjp lossGraph arguments ()
      (Tensor.scalar (1 : Scalar))
  let stateGradients : TensorPack Scalar [xShape] :=
    let (_, stateGradients) := Pack.split
      (ss₁ := paramShapes width) (ss₂ := [xShape]) allGradients
    stateGradients
  let gradient : Tensor Scalar xShape := Pack.get stateGradients ⟨0, by decide⟩
  let updated := state + Tensor.scale gradient stepSize
  clampStateTensor (-radius) radius updated

/--
Lower the Lyapunov loss at `parameters` to the verifier IR. `noinline` keeps its large polymorphic
program out of the callers' code.
-/
@[noinline] def lowerLossToIR
    (width : Nat)
    (parameters : TensorPack Scalar (Lyapunov.paramShapes width)) :
    Except String (IR.LoweredModel Scalar) :=
  IR.lowerForward
    (α := Scalar) (paramShapes := Lyapunov.paramShapes width)
    (inShape := xShape) (outShape := [])
    (lossProgram width (β := Scalar))
    parameters

/-- Run IBP on the loss graph `G` with payload `P` from the box `inputBox` and print the lower
and upper bounds of the output box. -/
@[noinline] def reportIBP (G : IR.Graph) (P : IR.Payload Scalar)
    (inputBox : FlatBox Scalar) : IO Unit := do
  let outputBounds ← IO.ofExcept (Op.outputBox? G (Op.runIBP G P (Op.inputSeeds G [inputBox])) 0)
  IO.println s!"[IBP] loss lo = {pretty outputBounds.lo}"
  IO.println s!"[IBP] loss hi = {pretty outputBounds.hi}"

/-- Run CROWN on the loss graph `G` with payload `P` from the box `inputBox` and print the lower and
upper loss bounds, or CROWN's error message. -/
@[noinline] def reportCROWN (G : IR.Graph) (P : IR.Payload Scalar)
    (inputBox : FlatBox Scalar) : IO Unit := do
  match Op.outputBoxCROWN? G P inputBox 0 with
  | .ok bounds =>
      IO.println s!"[CROWN] loss lo = {pretty bounds.lo}"
      IO.println s!"[CROWN] loss hi = {pretty bounds.hi}"
  | .error message =>
      IO.println s!"[CROWN] {message}"

/--
Lower the shared scalar loss and report IBP and CROWN on the ℓ∞ box of radius `epsilon` around the
origin.
-/
def reportLossBox
    (width : Nat)
    (parameters : TensorPack Scalar (paramShapes width))
    (epsilon : Scalar) : IO Unit := do
  IO.println "Stage 2 report: IBP and CROWN bounds of the scalar loss over a small box"
  let lowered ← IO.ofExcept (lowerLossToIR width parameters)

  IO.println s!"lowered IR nodes: {lowered.graph.nodes.size}"

  let origin : Tensor Scalar xShape := Tensor.zeros (α := Scalar) xShape
  let inputBox : FlatBox Scalar := FlatBox.lInfBall (α := Scalar) origin epsilon
  let imported := lowered.toImported

  reportIBP imported.graph imported.payload inputBox
  reportCROWN imported.graph imported.payload inputBox

end TorchLean.Examples.Lyapunov

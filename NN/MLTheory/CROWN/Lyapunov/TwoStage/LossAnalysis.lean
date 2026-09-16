/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Neural.State
public import NN.API.Sample
public import NN.MLTheory.CROWN.Graph
public import NN.MLTheory.CROWN.Lyapunov.TwoStage.Execution
public import NN.Runtime.Autograd.Model.Autodiff
public import NN.Verification.Builtin.Lowering

/-!
# Lowered Loss Analysis for Two-Stage Lyapunov Workflows

The executable operations shared by the hybrid and all-in-Lean pipelines: projected gradient
ascent on the lowered loss and an IBP/CROWN check of that loss over an input box.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace NN.MLTheory.CROWN.Lyapunov.TwoStage.LossAnalysis

open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN.Lyapunov.TwoStage

/-- Executable float32 semantics used by both lowered two-stage pipelines. -/
abbrev Scalar : Type := (ExecFloat.Binary 8 23)

/-- Shapes supplied to the lowered scalar loss: parameters followed by one state vector. -/
abbrev LossInputs (width : Nat) : List Shape := Core.paramShapes width ++ [Core.xShape]

/--
Take one projected gradient-ascent step on the state input of a lowered scalar loss.

Only the state gradient is retained; parameters remain fixed. The result is projected back into
the coordinate box `[-radius, radius]`.
-/
def projectedGradientStep
    (width : Nat)
    (lossGraph : Runtime.Autograd.Torch.TypedScalarGraph Scalar (LossInputs width))
    (parameters : TorchLean.nn.State Scalar (Core.paramShapes width))
    (state : Tensor Scalar Core.xShape)
    (stepSize radius : Scalar) : Tensor Scalar Core.xShape :=
  let arguments : TorchLean.TensorPack Scalar (LossInputs width) :=
    TorchLean.TensorPack.append (α := Scalar)
      (ss₁ := Core.paramShapes width) (ss₂ := [Core.xShape])
      (TorchLean.nn.State.Internal.toTensorPack parameters)
      (TorchLean.TensorPack.singleton state)
  let allGradients : TorchLean.TensorPack Scalar (LossInputs width) :=
    Runtime.Autograd.Torch.TypedScalarGraph.backward
      (α := Scalar) (Γ := LossInputs width) lossGraph arguments
  let stateGradients : TorchLean.TensorPack Scalar [Core.xShape] :=
    let (_, stateGradients) := TorchLean.TensorPack.split (α := Scalar)
      (ss₁ := Core.paramShapes width) (ss₂ := [Core.xShape]) allGradients
    stateGradients
  let gradient := TorchLean.TensorPack.get stateGradients ⟨0, by decide⟩
  let updated := Tensor.addSpec state (Tensor.scaleSpec gradient stepSize)
  Execution.clampStateTensor (-radius) radius updated

/-- Lower the Lyapunov loss while keeping its large polymorphic program out of callers' code. -/
@[noinline] def lowerLossToIR
    (width : Nat)
    (parameters : TorchLean.nn.State Scalar (Core.paramShapes width)) :
    Except String (NN.Verification.Builtin.LoweredIR Scalar) :=
  NN.Verification.Builtin.lowerForwardToIR
    (α := Scalar) (paramShapes := Core.paramShapes width)
    (inShape := Core.xShape) (outShape := [])
    (Core.lossProgram width (β := Scalar))
    (TorchLean.nn.State.Internal.toTensorPack parameters)

/-- Run and print the IBP bound for an already lowered loss graph. -/
@[noinline] def reportIBP
    (lowered : NN.Verification.Builtin.LoweredIR Scalar)
    (parameterStore : ParamStore Scalar) : IO Unit := do
  let intervalBounds := runIBP (α := Scalar) lowered.graph parameterStore
  let outputBounds ← lowered.outputBoxOrThrow intervalBounds
  IO.println s!"[IBP] scalar loss box dim={outputBounds.dim}"

/-- Run and print the CROWN bound for an already lowered loss graph. -/
@[noinline] def reportCROWN
    (lowered : NN.Verification.Builtin.LoweredIR Scalar)
    (inputBox : FlatBox Scalar) (parameterStore : ParamStore Scalar) : IO Unit := do
  match outputBoxCROWN? lowered.graph parameterStore inputBox
      lowered.inputId lowered.outputId Core.xDim with
  | .ok bounds =>
      IO.println s!"[CROWN] loss lo = {pretty bounds.lo}"
      IO.println s!"[CROWN] loss hi = {pretty bounds.hi}"
  | .error message =>
      IO.println s!"[CROWN] {message}"

/--
Lower the shared scalar loss and report its IBP and CROWN bounds on an origin-centered box.
-/
def checkLossBox
    (width : Nat)
    (parameters : TorchLean.nn.State Scalar (Core.paramShapes width))
    (epsilon : Scalar) : IO Unit := do
  IO.println "Stage 2 check: IBP + CROWN on the scalar loss over a small box"
  let lowered ←
    match lowerLossToIR width parameters with
    | .ok result => pure result
    | .error error => throw <| IO.userError error

  IO.println s!"lowered IR nodes: {lowered.graph.nodes.size}"

  let origin : Tensor Scalar Core.xShape := Tensor.zeros (α := Scalar) Core.xShape
  let inputBox : FlatBox Scalar := FlatBox.lInfBall (α := Scalar) origin epsilon
  let parameterStore : ParamStore Scalar := lowered.seedInputBox inputBox

  reportIBP lowered parameterStore
  reportCROWN lowered inputBox parameterStore

end NN.MLTheory.CROWN.Lyapunov.TwoStage.LossAnalysis

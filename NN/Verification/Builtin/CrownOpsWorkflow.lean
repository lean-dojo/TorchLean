/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.API.Verification.Lowering

/-!
# TorchLean CROWN Ops Workflow

Running CROWN end-to-end on small TorchLean graphs.

We lower TorchLean programs to the verifier IR (`NN.IR.Graph`), then run:
- IBP (`runIBP`)
- CROWN output bounds (`outputBoxCROWN?`)
- objective-dependent backward CROWN (`backwardObjectiveBox?`)

The workflow gives compact runtime checks for:
- `softmax` (vector)
- `mse_loss` (vector → scalar)

On the CLI's rounded backends, both CROWN queries use directed backward propagation and may fall
back to coarse IBP bounds. A successful run does not establish tighter bounds or a model theorem.

For attention + `layer_norm`, see
  `NN/Verification/Builtin/TransformerIBPWorkflow.lean`.

Run:
  `lake exe verify -- torchlean-crown-ops`
  `lake exe verify -- torchlean-crown-ops --arithmetic ieee`
-/

@[expose] public section


namespace NN.Verification.Builtin.CrownOpsWorkflow

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor
open _root_.TorchLean

open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN

/-- Input dimension for the softmax workflow model. -/
def softmaxInDim : Nat := 2
/-- Output dimension for the softmax workflow model. -/
def softmaxOutDim : Nat := 3

/-- Input shape for the softmax workflow model. -/
def softmaxXShape : Spec.Shape := [softmaxInDim]
/-- Output shape for the softmax workflow model. -/
def softmaxYShape : Spec.Shape := [softmaxOutDim]

/-- TorchLean model: `Linear -> Softmax`. -/
def softmaxModel : nn.Sequential [softmaxInDim] [softmaxOutDim] :=
  nn.build 0 <|
    nn.Sequential![
      nn.linear softmaxInDim softmaxOutDim,
      nn.softmax (shape := [softmaxOutDim]) 0
    ]

/-- Parameter shapes for `softmaxModel`. -/
def softmaxParamShapes : List Spec.Shape := nn.stateShapes softmaxModel

/-- Directed lower bound on the softmax margin $p_0 - p_1$. -/
def softmaxMargin {α : Type} [TorchLean.Storage α] [Context α] [BoundOps α]
    (lo hi : TorchLean.Tensor α softmaxYShape) : α :=
  let lo0 := _root_.TorchLean.Tensor.getScalar lo ⟨0, by decide⟩
  let hi1 := _root_.TorchLean.Tensor.getScalar hi ⟨1, by decide⟩
  BoundOps.subDown lo0 hi1

/--
Run the softmax workflow under a chosen scalar backend `α`.

This lowers the TorchLean model to verifier IR and prints IBP/CROWN bounds.
-/
def runSoftmax {α : Type} [TorchLean.Storage α] [_root_.Context α] [ToString α]
    [Runtime.FromFloat α] [BoundOps α] [NonlinearBoundOps α] : IO Unit := do
  IO.println "== Workflow 1: linear -> softmax (vector) =="
  let cast : Float → α := Runtime.ofFloat

  let params : nn.State α softmaxParamShapes :=
    nn.State.empty
      |>.push
        (TorchLean.Tensor.map cast <|
          (Tensor.from (#[1.0, -0.5, 0.2, 0.7, -0.3, 0.1] : Array Float)).reshape
            [3, 2] (by dsimp; decide))
      |>.push
        (TorchLean.Tensor.map cast <|
          (Tensor.from (#[0.1, -0.2, 0.0] : Array Float)).reshape [3] (by dsimp; decide))

  let lowered ←
    match Verification.lowerProgramToIR (α := α)
          (nn.forward softmaxModel (α := α)) params with
    | .ok c => pure c
    | .error e => throw <| IO.userError e

  IO.println s!"lowered IR nodes: {lowered.graph.nodes.size}"

  let x0 : TorchLean.Tensor α softmaxXShape :=
    TorchLean.Tensor.map cast
      ((Tensor.from (#[0.2, -0.1] : Array Float)).reshape [2] (by dsimp; decide))
  let eps : α := Runtime.ofFloat 0.05
  let xB : FlatBox α := NN.Verification.Builtin.lInfBall (α := α) x0 eps
  let ps : ParamStore α := lowered.seedInputBox xB

  -- IBP
  let ibp := lowered.runIBP ps
  let outB ← lowered.outputBoxOrThrow ibp
  if hDim : outB.dim = softmaxOutDim then
    let loY : TorchLean.Tensor α softmaxYShape := by
      simpa [softmaxYShape] using outB.loAsDim hDim
    let hiY : TorchLean.Tensor α softmaxYShape := by
      simpa [softmaxYShape] using outB.hiAsDim hDim
    IO.println s!"[IBP] p lo = {pretty loY}"
    IO.println s!"[IBP] p hi = {pretty hiY}"
    IO.println s!"[IBP] margin(p0 - p1) = {softmaxMargin (α := α) loY hiY}"
  else
    throw <| IO.userError s!"[IBP] unexpected output dim {outB.dim} (expected {softmaxOutDim})"

  -- CROWN output bounds.
  let outC ← lowered.outputBoxCROWNOrThrow ps xB
  if hOut : outC.dim = softmaxOutDim then
    let loY : TorchLean.Tensor α softmaxYShape := by
      simpa [softmaxYShape] using outC.loAsDim hOut
    let hiY : TorchLean.Tensor α softmaxYShape := by
      simpa [softmaxYShape] using outC.hiAsDim hOut
    IO.println s!"[CROWN] p lo = {pretty loY}"
    IO.println s!"[CROWN] p hi = {pretty hiY}"
    IO.println s!"[CROWN] margin(p0 - p1) = {softmaxMargin (α := α) loY hiY}"
  else
    throw <| IO.userError s!"[CROWN] unexpected output dim {outC.dim} (expected {softmaxOutDim})"

  -- Backward/dual CROWN for the margin objective: p0 - p1.
  let objV : TorchLean.Tensor α [softmaxOutDim] :=
    TorchLean.Tensor.map cast
      ((Tensor.from (#[1.0, -1.0, 0.0] : Array Float)).reshape [3] (by dsimp; decide))
  let obj : FlatTensor α := { n := softmaxOutDim, v := objV }
  let outObjective ←
    match lowered.backwardObjectiveBox? ps ibp xB obj with
    | .ok result => pure result
    | .error msg => throw <| IO.userError s!"[CROWN-backward] {msg}"
  if outObjective.dim != 1 then
    throw <| IO.userError
      s!"[CROWN-backward] unexpected output dim {outObjective.dim} (expected 1)"
  let loM : α := getAtOrZero outObjective.lo [0]
  let hiM : α := getAtOrZero outObjective.hi [0]
  IO.println s!"[CROWN-backward] margin lo = {loM}"
  IO.println s!"[CROWN-backward] margin hi = {hiM}"

/-- Input dimension for the MSE-loss workflow model. -/
def mseInDim : Nat := 2
/-- Output dimension for the MSE-loss workflow model. -/
def mseOutDim : Nat := 2

/-- Weight shape for the MSE-loss workflow's linear layer. -/
def mseWShape : Spec.Shape := [mseOutDim, mseInDim]
/-- Bias shape for the MSE-loss workflow's linear layer. -/
def mseBShape : Spec.Shape := [mseOutDim]
/-- Input shape for the MSE-loss workflow. -/
def mseXShape : Spec.Shape := [mseInDim]
/-- Output shape for the MSE-loss workflow. -/
def mseYShape : Spec.Shape := [mseOutDim]

/-- Parameter shapes for the MSE-loss workflow (`[W,b,target]`). -/
def mseParamShapes : List Spec.Shape := [mseWShape, mseBShape, mseYShape]

/-- TorchLean forward program computing
$\widehat{y}=\operatorname{linear}(x)$ and
$\operatorname{mse\_loss}(\widehat{y},\mathrm{target})$, returning a scalar. -/
def mseLossModel {α : Type} [TorchLean.Storage α] [Context α] :
    _root_.Runtime.Autograd.Model.Program α (mseParamShapes ++ [mseXShape])
      [] :=
  fun {m} _ _ =>
    fun w b target x =>
      (do
        let yhat ← Runtime.linear
          (m := m) (α := α)
          (batchShape := []) (inputWidth := mseInDim) (outputWidth := mseOutDim) w b x
        Runtime.mseLoss (m := m) (α := α) (s := mseYShape) yhat target
        : m (Runtime.ValueRef (m := m) (α := α) []))

/--
Run the MSE-loss workflow under a chosen scalar backend `α`.

This lowers the TorchLean forward computation to verifier IR and prints IBP/CROWN bounds for the
scalar loss.
-/
def runMSE {α : Type} [TorchLean.Storage α] [_root_.Context α] [ToString α]
    [Runtime.FromFloat α] [BoundOps α] [NonlinearBoundOps α] : IO Unit := do
  IO.println "== Workflow 2: linear -> mse_loss (scalar) =="
  let cast : Float → α := Runtime.ofFloat

  let params : nn.State α mseParamShapes :=
    nn.State.empty
      |>.push
        (TorchLean.Tensor.map cast <|
          (Tensor.from (#[0.4, -0.3, 1.2, 0.1] : Array Float)).reshape
            [2, 2] (by dsimp; decide))
      |>.push
        (TorchLean.Tensor.map cast <|
          (Tensor.from (#[0.05, -0.02] : Array Float)).reshape [2] (by dsimp; decide))
      |>.push
        (TorchLean.Tensor.map cast <|
          (Tensor.from (#[0.0, 1.0] : Array Float)).reshape [2] (by dsimp; decide))

  let lowered ←
    match Verification.lowerProgramToIR (α := α)
          (mseLossModel (α := α)) params with
    | .ok c => pure c
    | .error e => throw <| IO.userError e

  IO.println s!"lowered IR nodes: {lowered.graph.nodes.size}"

  let x0 : TorchLean.Tensor α mseXShape :=
    TorchLean.Tensor.map cast
      ((Tensor.from (#[0.3, -0.4] : Array Float)).reshape [2] (by dsimp; decide))
  let eps : α := Runtime.ofFloat 0.05
  let xB : FlatBox α := NN.Verification.Builtin.lInfBall (α := α) x0 eps
  let ps : ParamStore α := lowered.seedInputBox xB

  -- IBP
  let ibp := lowered.runIBP ps
  let outB ← lowered.outputBoxOrThrow ibp
  if outB.dim != 1 then
    throw <| IO.userError s!"[IBP] unexpected output dim {outB.dim} (expected 1)"
  IO.println s!"[IBP] loss lo = {pretty outB.lo}"
  IO.println s!"[IBP] loss hi = {pretty outB.hi}"

  -- CROWN output bounds on the scalar loss.
  let outC ← lowered.outputBoxCROWNOrThrow ps xB
  if outC.dim != 1 then
    throw <| IO.userError s!"[CROWN] unexpected output dim {outC.dim} (expected 1)"
  IO.println s!"[CROWN] loss lo = {pretty outC.lo}"
  IO.println s!"[CROWN] loss hi = {pretty outC.hi}"

  -- Backward/dual CROWN for the loss objective itself (obj = 1).
  let obj : FlatTensor α := { n := 1, v := Tensor.full [1] 1 }
  let outObjective ←
    match lowered.backwardObjectiveBox? ps ibp xB obj with
    | .ok result => pure result
    | .error msg => throw <| IO.userError s!"[CROWN-backward] {msg}"
  if outObjective.dim != 1 then
    throw <| IO.userError
      s!"[CROWN-backward] unexpected output dim {outObjective.dim} (expected 1)"
  IO.println s!"[CROWN-backward] loss lo = {pretty outObjective.lo}"
  IO.println s!"[CROWN-backward] loss hi = {pretty outObjective.hi}"

/-- Run all CROWN-ops workflows (softmax + mse_loss) under a chosen scalar backend `α`. -/
def runOnce {α : Type} [TorchLean.Storage α] [_root_.Context α] [ToString α] [Runtime.FromFloat α]
    [BoundOps α] [NonlinearBoundOps α] : IO Unit := do
  runSoftmax (α := α)
  IO.println ""
  runMSE (α := α)

/--
CLI entry point for the CROWN-ops workflow.

This is wired into `lake exe verify -- torchlean-crown-ops`.
Missing bounds or unexpected output dimensions fail the command.
-/
def main (args : List String) : IO Unit :=
  NN.Verification.Builtin.runWithBoundArithmetic
    "TorchLean → IR → IBP + CROWN (ops: softmax/mse_loss)" args
    (@runOnce)

end NN.Verification.Builtin.CrownOpsWorkflow

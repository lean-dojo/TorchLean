/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.API.Verification.Lowering

/-!
# TorchLean Transformer IBP Workflow

Small end-to-end workflow:

TorchLean (MHA + LayerNorm + MSE) → lower to `NN.IR.Graph` → run:
- IBP (`runIBP`)
- basic CROWN forward bounds (`runCROWN`)
- objective-dependent backward/dual CROWN (`runCROWNBackwardObjective`)

Run:
  `lake exe verify -- torchlean-transformer-ibp`
  `lake exe verify -- torchlean-transformer-ibp --with-crown`
  `lake exe verify -- torchlean-transformer-ibp --arithmetic ieee`
-/

@[expose] public section


namespace NN.Verification.Builtin.TransformerIBPWorkflow

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor
open _root_.TorchLean

open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN

/-- Sequence length for the transformer verification example. -/
def n : Nat := 2
/-- Model embedding dimension. -/
def dModel : Nat := 2
/-- Number of attention heads. -/
def numHeads : Nat := 1
/-- Per-head embedding dimension. -/
def headDim : Nat := 2
/-- Batch size for the transformer verification example. -/
def batch : Nat := 1

/-- Input shape `(batch × n × dModel)`. -/
def xShape : Spec.Shape := [batch, n, dModel]
/-- Projection weight shape for Q/K/V: `(dModel × (numHeads*headDim))`. -/
def wProjShape : Spec.Shape := [dModel, numHeads * headDim]
/-- Output projection weight shape: `((numHeads*headDim) × dModel)`. -/
def wOShape : Spec.Shape := [numHeads * headDim, dModel]
/-- LayerNorm scale parameter shape, matching the feature dimension. -/
def gammaShape : Spec.Shape := [dModel]
/-- LayerNorm beta shape, matching the feature dimension. -/
def betaShape : Spec.Shape := [dModel]
/-- MSE target shape (matches the model output shape). -/
def targetShape : Spec.Shape := xShape

/-- Parameter shapes list for `modelLoss` (`Wq,Wk,Wv,Wo,gamma,beta,target`). -/
def paramShapes : List Spec.Shape :=
  [wProjShape, wProjShape, wProjShape, wOShape, gammaShape, betaShape, targetShape]

/-- TorchLean program: `mha -> layer_norm -> mse_loss`, returning a scalar loss. -/
def modelLoss {α : Type} [TorchLean.Storage α] [Context α] :
    _root_.Runtime.Autograd.Model.Program α (paramShapes ++ [xShape])
      [] :=
  fun {m} _ _ =>
    fun wq wk wv wo gamma beta target x =>
      (do
        let y ← _root_.Runtime.Autograd.Model.multiHeadAttention (m := m) (α := α)
          (leadingShape := [batch]) (n := n) (numHeads := numHeads) (dModel := dModel)
          (headDim := headDim)
          (hN := by decide) wq wk wv wo x (mask := none)
        let yLn ← _root_.Runtime.Autograd.Model.layerNorm (m := m) (α := α)
          (leading := [batch, n]) (width := dModel) (hWidth := by decide) y gamma beta
        Runtime.mseLoss (m := m) (α := α) (s := xShape) yLn target
        : m (Runtime.ValueRef (m := m) (α := α) []))

/-- Runtime-selected typed runner used by the CLI entrypoint. -/
def runMain {α : Type} [TorchLean.Storage α] [_root_.Context α] [ToString α] [Runtime.FromFloat α]
    [BoundOps α] [NonlinearBoundOps α] (withCrown : Bool) : IO Unit := do
  let cast : Float → α := Runtime.ofFloat
  let identity : Tensor α [2, 2] :=
    TorchLean.Tensor.map cast <|
      (Tensor.from (#[1.0, 0.0, 0.0, 1.0] : Array Float)).reshape
        [2, 2] (by dsimp; decide)
  let params : nn.State α paramShapes :=
    nn.State.empty
      |>.push identity
      |>.push identity
      |>.push identity
      |>.push identity
      |>.push
        (TorchLean.Tensor.map cast <|
          (Tensor.from (#[1.0, 1.0] : Array Float)).reshape [2] (by dsimp; decide))
      |>.push
        (TorchLean.Tensor.map cast <|
          (Tensor.from (#[0.0, 0.0] : Array Float)).reshape [2] (by dsimp; decide))
      |>.push
        (TorchLean.Tensor.map cast <|
          (Tensor.from (#[0.0, 0.0, 0.0, 0.0] : Array Float)).reshape
            [1, 2, 2] (by dsimp; decide))

  let lowered ←
    match Verification.lowerProgramToIR (α := α)
          (modelLoss (α := α)) params with
    | .ok c => pure c
    | .error e => throw <| IO.userError e

  IO.println s!"lowered IR nodes: {lowered.graph.nodes.size}"

  let x0 : TorchLean.Tensor α xShape :=
    TorchLean.Tensor.map cast
      ((Tensor.from (#[0.2, -0.3, 0.7, 0.1] : Array Float)).reshape [1, 2, 2] (by dsimp; decide))
  let eps : α := Runtime.ofFloat 0.05
  let xB : FlatBox α := NN.Verification.Builtin.lInfBall (α := α) x0 eps
  let ps : ParamStore α := lowered.seedInputBox xB

  let boxes := lowered.runIBP ps
  let outB ← lowered.outputBoxOrThrow boxes
  IO.println s!"[IBP] loss lo: {pretty outB.lo}"
  IO.println s!"[IBP] loss hi: {pretty outB.hi}"

  if !withCrown then
    IO.println ("[CROWN] skipped for the default runtime-check path; " ++
      "pass --with-crown for the heavier transformer CROWN run")
    return ()

  IO.println ("[CROWN] running transformer-scale forward CROWN; " ++
    "this experimental path can take minutes")
  let inputDim := Spec.Shape.size xShape
  match lowered.outputBoxCROWN? ps xB with
  | .ok outC =>
      if hOut : outC.dim = 1 then
          IO.println s!"[CROWN] loss lo: {pretty outC.lo}"
          IO.println s!"[CROWN] loss hi: {pretty outC.hi}"
      else
        IO.println s!"[CROWN] unexpected output dim {outC.dim} (expected 1)"
  | .error msg =>
      IO.println s!"[CROWN] {msg}"

  IO.println "[CROWN-backward] running objective-dependent backward CROWN"
  let obj : FlatTensor α := { n := 1, v := Tensor.full [1] 1 }
  match lowered.backwardObjectiveBox? ps boxes xB obj with
  | .ok outC =>
      IO.println s!"[CROWN-backward] loss lo: {pretty outC.lo}"
      IO.println s!"[CROWN-backward] loss hi: {pretty outC.hi}"
  | .error msg =>
      IO.println s!"[CROWN-backward] {msg}"

/-- Runtime-selected typed runner for the default IBP-only path. -/
def runMainDefault {α : Type} [TorchLean.Storage α]
    [_root_.Context α] [ToString α]
    [Runtime.FromFloat α] [BoundOps α] [NonlinearBoundOps α] : IO Unit :=
  runMain (α := α) false

/-- Runtime-selected typed runner for the heavier IBP+CROWN path. -/
def runMainWithCrown {α : Type} [TorchLean.Storage α]
    [_root_.Context α] [ToString α]
    [Runtime.FromFloat α] [BoundOps α] [NonlinearBoundOps α] : IO Unit :=
  runMain (α := α) true

/--
CLI entry point for the transformer-IBP workflow.

This is wired into `lake exe verify -- torchlean-transformer-ibp`.

By default this command is a fast validation check: lower the TorchLean transformer fragment to the
verification IR and run IBP on the scalar loss. Pass `--with-crown` to also run the experimental
transformer-scale CROWN passes. The separate `torchlean-crown-ops` command keeps CROWN itself in the
standard check suite on compact graphs, while this file focuses on the heavier attention/layer-norm
front-end path.
-/
def main (args : List String) : IO Unit := do
  let parsedWithCrown : Bool × List String ←
    match CLI.takeBoolFlag args "with-crown" with
    | .ok parsed => pure parsed
    | .error msg => throw <| IO.userError msg
  let withCrown : Bool := parsedWithCrown.1
  let restArgs : List String := parsedWithCrown.2
  if withCrown then
    NN.Verification.Builtin.runWithBoundArithmetic
      "TorchLean (MHA+LayerNorm+MSE) → IR → IBP" restArgs
      (@runMainWithCrown)
  else
    NN.Verification.Builtin.runWithBoundArithmetic
      "TorchLean (MHA+LayerNorm+MSE) → IR → IBP" restArgs
      (@runMainDefault)

end NN.Verification.Builtin.TransformerIBPWorkflow

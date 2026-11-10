/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL.PPO.ActorCritic
public import TorchLean.Verify.Lowering
public import TorchLeanExamples.Verification.BoundArithmetic
public import TorchLean.NN.Data.Source
public import TorchLean.NN.Builder
public import TorchLean.NN.Training.Reporting

/-!
# TorchLean Transformer IBP Workflow

A small end-to-end workflow: the TorchLean program (MHA + LayerNorm + MSE) is lowered to a graph of
registered operators, imported with its payload (`IR.LoweredModel.toImported`) and bounded by
- IBP on the operator graph (`TorchLean.CROWN.Op.runIBP`)
- optional CROWN output bounds on the operator graph (`TorchLean.CROWN.Op.outputBoxCROWN?`)
- optional objective-dependent backward CROWN on the operator graph
  (`TorchLean.CROWN.Op.backwardObjectiveBox?`) from its IBP boxes

The example has one batch, two tokens, two features and one attention head. On the CLI's rounded
backends, both CROWN queries propagate backward with directed rounding and may fall back to IBP
bounds. The mean squared error at the end has no backward rule, so both CROWN bounds come from the
loss's IBP box. The run is a runtime integration check and certifies nothing about the model.

Run:
  `lake exe verify -- torchlean-transformer-ibp`
  `lake exe verify -- torchlean-transformer-ibp --with-crown`
  `lake exe verify -- torchlean-transformer-ibp --arithmetic ieee`
-/

@[expose] public section

namespace TorchLean.Examples.TransformerIBPWorkflow

open Tensor

open CROWN.Graph
open CROWN

/-- Sequence length for the transformer verification example. -/
def seqLen : Nat := 2
/-- Model embedding dimension. -/
def dModel : Nat := 2
/-- Number of attention heads. -/
def numHeads : Nat := 1
/-- Per-head embedding dimension. -/
def headDim : Nat := 2
/-- Batch size for the transformer verification example. -/
def batch : Nat := 1

/-- Input shape `(batch × seqLen × dModel)`. -/
def xShape : Shape := [batch, seqLen, dModel]
/-- Projection weight shape for Q/K/V: `(dModel × (numHeads*headDim))`. -/
def wProjShape : Shape := [dModel, numHeads * headDim]
/-- Output projection weight shape: `((numHeads*headDim) × dModel)`. -/
def wOShape : Shape := [numHeads * headDim, dModel]
/-- LayerNorm scale parameter shape, matching the feature dimension. -/
def gammaShape : Shape := [dModel]
/-- LayerNorm beta shape, matching the feature dimension. -/
def betaShape : Shape := [dModel]
/-- MSE target shape (matches the model output shape). -/
def targetShape : Shape := xShape

/-- Parameter shapes list for `modelLoss` (`Wq,Wk,Wv,Wo,gamma,beta,target`). -/
def paramShapes : List Shape :=
  [wProjShape, wProjShape, wProjShape, wOShape, gammaShape, betaShape, targetShape]

/-- TorchLean program: `mha -> layer_norm -> mse_loss`, returning a scalar loss. -/
def modelLoss {α : Type} [Storage α] [Context α] :
    Program α (paramShapes ++ [xShape])
      [] :=
  fun {m} _ _ =>
    fun wq wk wv wo gamma beta target x =>
      (do
        let y ← nn.functional.attention (m := m) (α := α)
          (leadingShape := [batch]) (n := seqLen) (numHeads := numHeads) (dModel := dModel)
          (headDim := headDim)
          (hN := by decide) wq wk wv wo x (mask := none)
        let yLn ← nn.functional.layerNorm (m := m) (α := α)
          (leading := [batch, seqLen]) (width := dModel) (hWidth := by decide) y gamma beta
        Ops.apply (m := m) (α := α) (.mseLoss (s := xShape))
          (yLn, target, ())
        : m (Ops.Ref (m := m) (α := α) []))

/-- Lower `modelLoss` under the scalar backend `α`, with identity projections, LayerNorm scale 1
and bias 0 and a zero target, and print IBP bounds on the loss over the ℓ∞ ball of radius `0.05`
around the input `[[0.2, -0.3], [0.7, 0.1]]`. With `withCrown` it also prints the CROWN and
backward CROWN bounds. -/
def run {α : Type} [Storage α] [Context α] [ToString α] [FromFloat α]
    [BoundOps α] [NonlinearBoundOps α] (withCrown : Bool) : IO Unit := do
  let cast : Float → α := ofFloat
  let identity : Tensor α [2, 2] :=
    Tensor.map cast <|
      (Tensor.from (#[1.0, 0.0, 0.0, 1.0] : Array Float)).reshape
        [2, 2] (by dsimp; decide)
  let params : TensorPack α paramShapes :=
    Pack.nil
      |>.snoc identity
      |>.snoc identity
      |>.snoc identity
      |>.snoc identity
      |>.snoc
        (Tensor.map cast <|
          (Tensor.from (#[1.0, 1.0] : Array Float)).reshape [2] (by dsimp; decide))
      |>.snoc
        (Tensor.map cast <|
          (Tensor.from (#[0.0, 0.0] : Array Float)).reshape [2] (by dsimp; decide))
      |>.snoc
        (Tensor.map cast <|
          (Tensor.from (#[0.0, 0.0, 0.0, 0.0] : Array Float)).reshape
            [1, 2, 2] (by dsimp; decide))

  let lowered ← IO.ofExcept <|
    IR.lowerForward (α := α) (modelLoss (α := α)) params

  IO.println s!"lowered IR nodes: {lowered.graph.nodes.size}"
  let imported := lowered.toImported

  let x0 : Tensor α xShape :=
    Tensor.map cast
      ((Tensor.from (#[0.2, -0.3, 0.7, 0.1] : Array Float)).reshape [1, 2, 2] (by dsimp; decide))
  let eps : α := ofFloat 0.05
  let xB : FlatBox α := FlatBox.lInfBall (α := α) x0 eps

  let boxes := imported.runIBP xB
  let outB ← IO.ofExcept (imported.outputBox? boxes)
  if outB.dim != 1 then
    throw <| IO.userError s!"[IBP] unexpected output dim {outB.dim} (expected 1)"
  IO.println s!"[IBP] loss lo: {pretty outB.lo}"
  IO.println s!"[IBP] loss hi: {pretty outB.hi}"

  if !withCrown then
    IO.println ("[CROWN] skipped for the default runtime-check path; " ++
      "pass --with-crown to also compute CROWN output and objective bounds")
    return ()

  IO.println "[CROWN] computing output bounds for the attention/LayerNorm loss"
  let outC ← IO.ofExcept (imported.outputBoxCROWN? xB)
  if outC.dim != 1 then
    throw <| IO.userError s!"[CROWN] unexpected output dim {outC.dim} (expected 1)"
  IO.println s!"[CROWN] loss lo: {pretty outC.lo}"
  IO.println s!"[CROWN] loss hi: {pretty outC.hi}"

  IO.println "[CROWN-backward] running objective-dependent backward CROWN"
  let obj : FlatTensor α := { n := 1, v := Tensor.full [1] 1 }
  let outObjective ←
    match imported.backwardObjectiveBox? boxes xB obj with
    | .ok result => pure result
    | .error msg => throw <| IO.userError s!"[CROWN-backward] {msg}"
  if outObjective.dim != 1 then
    throw <| IO.userError
      s!"[CROWN-backward] unexpected output dim {outObjective.dim} (expected 1)"
  IO.println s!"[CROWN-backward] loss lo: {pretty outObjective.lo}"
  IO.println s!"[CROWN-backward] loss hi: {pretty outObjective.hi}"

/--
CLI entry point of `lake exe verify -- torchlean-transformer-ibp`. By default it runs IBP on the
scalar loss of the attention/LayerNorm example, `--with-crown` adds the CROWN output and objective
bounds, and `--arithmetic` picks the scalar backend (native `Float32` by default). Missing bounds
or unexpected output dimensions fail the command.

The linear/softmax and linear/MSE examples belong to the separate `torchlean-crown-ops` command.
-/
def main (args : List String) : IO Unit := do
  let (withCrown, restArgs) ← IO.ofExcept (CLI.takeBoolFlag args "with-crown")
  let title := if withCrown then "TorchLean (MHA+LayerNorm+MSE) → IR → IBP + CROWN"
    else "TorchLean (MHA+LayerNorm+MSE) → IR → IBP"
  BoundArithmetic.run title restArgs
    (fun {α} _ _ _ _ _ _ => run (α := α) withCrown)

end TorchLean.Examples.TransformerIBPWorkflow

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.API.Verification.Lowering

/-!
# TorchLean IBP Workflow

Small end-to-end workflow:

TorchLean forward model → lower to `NN.IR.Graph` → run Lean IBP (`runIBP`).

Run:
  `lake exe verify -- torchlean-ibp`
  `lake exe verify -- torchlean-ibp --arithmetic ieee`
-/

@[expose] public section


namespace NN.Verification.Builtin.IBPWorkflow

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor
open _root_.TorchLean

open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN

/-- Input dimension for the small MLP in this workflow. -/
def inDim : Nat := 2
/-- Hidden width for the small MLP in this workflow. -/
def hidDim : Nat := 3
/-- Output dimension for the small MLP in this workflow. -/
def outDim : Nat := 1

/-- Input shape for the workflow model. -/
def xShape : Spec.Shape := [inDim]
/-- Output shape for the workflow model. -/
def yShape : Spec.Shape := [outDim]

/-- TorchLean model used in the workflow (a 2-layer ReLU MLP). -/
def mkModel : nn.Builder (nn.Sequential xShape yShape) :=
  nn.Sequential![
    nn.linear inDim hidDim,
    nn.relu,
    nn.linear hidDim outDim
  ]

/-- Deterministically instantiate `mkModel` from initialization seed zero. -/
def model : nn.Sequential xShape yShape :=
  nn.build 0 mkModel

/-- Parameter shapes for `model`. -/
def paramShapes : List Spec.Shape := nn.stateShapes model

/-- Runtime-selected typed runner used by the CLI entrypoint. -/
def runMain {α : Type} [TorchLean.Storage α] [_root_.Context α] [ToString α]
    [Runtime.FromFloat α] [BoundOps α] [NonlinearBoundOps α] : IO Unit := do
  let cast : Float → α := Runtime.ofFloat
  let params : nn.State α paramShapes :=
    nn.State.empty
      |>.push
        (TorchLean.Tensor.map cast <|
          (Tensor.from (#[0.1, 0.2, 0.3, 0.4, 0.5, 0.6] : Array Float)).reshape
            [3, 2] (by dsimp; decide))
      |>.push
        (TorchLean.Tensor.map cast <|
          (Tensor.from (#[0.1, 0.2, 0.3] : Array Float)).reshape [3] (by dsimp; decide))
      |>.push
        (TorchLean.Tensor.map cast <|
          (Tensor.from (#[0.7, 0.8, 0.9] : Array Float)).reshape [1, 3] (by dsimp; decide))
      |>.push
        (TorchLean.Tensor.map cast <|
          (Tensor.from (#[0.4] : Array Float)).reshape [1] (by dsimp; decide))

  let lowered ←
    match Verification.lowerForwardToIR (α := α) model params with
    | .ok c => pure c
    | .error e => throw <| IO.userError e

  IO.println s!"lowered IR nodes: {lowered.graph.nodes.size}"

  let x0 : TorchLean.Tensor α xShape :=
    TorchLean.Tensor.map cast
      ((Tensor.from (#[0.5, 0.8] : Array Float)).reshape [2] (by dsimp; decide))
  let eps : α := Runtime.ofFloat 0.1
  let xB : FlatBox α := NN.Verification.Builtin.lInfBall (α := α) x0 eps
  let ps : ParamStore α := lowered.seedInputBox xB

  let boxes := lowered.runIBP ps
  let outB ← lowered.outputBoxOrThrow boxes
  IO.println s!"output box lo: {pretty outB.lo}"
  IO.println s!"output box hi: {pretty outB.hi}"

/--
CLI entry point for the TorchLean → IR → IBP workflow.

This is wired into `lake exe verify -- torchlean-ibp`.
-/
def main (args : List String) : IO Unit := do
  NN.Verification.Builtin.runWithBoundArithmetic "TorchLean → IR → IBP (small MLP)" args
    (@runMain)

end NN.Verification.Builtin.IBPWorkflow

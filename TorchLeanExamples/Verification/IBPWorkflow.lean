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
# TorchLean IBP Workflow

A small end-to-end workflow: the TorchLean forward model is lowered to a graph of registered
operators, imported with its payload (`IR.LoweredModel.toImported`) and bounded by Lean IBP
(`TorchLean.CROWN.Op.runIBP`).

Run:
  `lake exe verify -- torchlean-ibp`
  `lake exe verify -- torchlean-ibp --arithmetic ieee`
-/

@[expose] public section


namespace TorchLean.Examples.IBPWorkflow

open Tensor

open CROWN

/-- Input dimension for the small MLP in this workflow. -/
def inDim : Nat := 2
/-- Hidden width for the small MLP in this workflow. -/
def hidDim : Nat := 3
/-- Output dimension for the small MLP in this workflow. -/
def outDim : Nat := 1

/-- Input shape for the workflow model. -/
def xShape : Shape := [inDim]
/-- Output shape for the workflow model. -/
def yShape : Shape := [outDim]

/-- The workflow's model: `Linear(2 → 3)`, ReLU, `Linear(3 → 1)`. -/
def mkModel : nn.Builder (nn.Sequential xShape yShape) :=
  nn.Sequential![
    nn.linear inDim hidDim,
    nn.op .relu,
    nn.linear hidDim outDim
  ]

/-- Deterministically instantiate `mkModel` from initialization seed zero. -/
def model : nn.Sequential xShape yShape :=
  nn.build 0 mkModel

/-- Parameter shapes for `model`. -/
def paramShapes : List Shape := nn.Sequential.stateShapes model

/-- Lower `model` with fixed weights to verifier IR under the scalar backend `α` and print the IBP
output box for the ℓ∞ ball of radius `0.1` around `(0.5, 0.8)`. -/
def run {α : Type} [Storage α] [Context α] [ToString α]
    [FromFloat α] [BoundOps α] [NonlinearBoundOps α] : IO Unit := do
  let cast : Float → α := ofFloat
  let params : TensorPack α paramShapes :=
    Pack.nil
      |>.snoc
        (Tensor.map cast <|
          (Tensor.from (#[0.1, 0.2, 0.3, 0.4, 0.5, 0.6] : Array Float)).reshape
            [3, 2] (by dsimp; decide))
      |>.snoc
        (Tensor.map cast <|
          (Tensor.from (#[0.1, 0.2, 0.3] : Array Float)).reshape [3] (by dsimp; decide))
      |>.snoc
        (Tensor.map cast <|
          (Tensor.from (#[0.7, 0.8, 0.9] : Array Float)).reshape [1, 3] (by dsimp; decide))
      |>.snoc
        (Tensor.map cast <|
          (Tensor.from (#[0.4] : Array Float)).reshape [1] (by dsimp; decide))

  let lowered ← IO.ofExcept (Verification.lowerForwardToIR (α := α) model params)

  IO.println s!"lowered IR nodes: {lowered.graph.nodes.size}"
  let imported := lowered.toImported

  let x0 : Tensor α xShape :=
    Tensor.map cast
      ((Tensor.from (#[0.5, 0.8] : Array Float)).reshape [2] (by dsimp; decide))
  let eps : α := ofFloat 0.1
  let xB : FlatBox α := FlatBox.lInfBall (α := α) x0 eps

  let boxes := imported.runIBP xB
  let outB ← IO.ofExcept (imported.outputBox? boxes)
  IO.println s!"output box lo: {pretty outB.lo}"
  IO.println s!"output box hi: {pretty outB.hi}"

/--
CLI entry point of `lake exe verify -- torchlean-ibp`. Its one flag, `--arithmetic`, picks
`Float32` (`native`, the default) or `ExecFloat.Binary 8 23` (`ieee`).
-/
def main (args : List String) : IO Unit := do
  BoundArithmetic.run "TorchLean → IR → IBP (small MLP)" args
    (@run)

end TorchLean.Examples.IBPWorkflow

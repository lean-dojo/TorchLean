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
# TorchLean CROWN Ops Workflow

Running CROWN end-to-end on small TorchLean graphs.

We lower each TorchLean program to a graph of registered operators, import it with its payload
(`IR.LoweredModel.toImported`) and run:
- IBP on the operator graph (`TorchLean.CROWN.Op.runIBP`)
- CROWN output bounds on the operator graph (`TorchLean.CROWN.Op.outputBoxCROWN?`)
- objective-dependent backward CROWN on the operator graph
  (`TorchLean.CROWN.Op.backwardObjectiveBox?`) from its IBP boxes

The two programs are compact runtime checks for `softmax` (vector) and `mse_loss` (vector →
scalar).

On the CLI's rounded backends, both CROWN queries propagate backward with directed rounding. Where
an affine bound fails they fall back to the IBP box, and every result is met with that box, so the
CROWN bounds are never looser than IBP's but need not be tighter. A run bounds one input box per
program and proves no theorem about either model.

For attention + `layer_norm`, see
  `TorchLeanExamples/Verification/TransformerIBPWorkflow.lean`.

Run:
  `scripts/lake.sh exe verify -- torchlean-crown-ops`
  `scripts/lake.sh exe verify -- torchlean-crown-ops --arithmetic ieee`
-/

@[expose] public section


namespace TorchLean.Examples.CROWNOpsWorkflow

open Tensor nn

open CROWN.Graph
open CROWN

namespace Softmax

/-- TorchLean model: `Linear -> Softmax`. -/
def model : nn.Sequential [2] [3] :=
  nn.build 0 <|
    nn.Sequential![
      nn.linear 2 3,
      nn.op (.softmax (s := [3]) ⟨0, by decide⟩)
    ]

/-- Print the softmax probability bounds of `box` under `tag`, with the margin lower bound
`lo(p₀) - hi(p₁)` rounded down. Fails unless `box` has dimension 3, the softmax output size. -/
def print {α : Type} [Storage α] [Context α] [ToString α]
    [BoundOps α] (tag : String) (box : FlatBox α) : IO Unit := do
  if hDim : box.dim = 3 then
    let loY : Tensor α [3] := box.loAsDim hDim
    let hiY : Tensor α [3] := box.hiAsDim hDim
    IO.println s!"[{tag}] p lo = {pretty loY}"
    IO.println s!"[{tag}] p hi = {pretty hiY}"
    let margin := BoundOps.subDown
      (Tensor.getScalar loY ⟨0, by decide⟩) (Tensor.getScalar hiY ⟨1, by decide⟩)
    IO.println s!"[{tag}] margin(p0 - p1) = {margin}"
  else
    throw <| IO.userError s!"[{tag}] unexpected output dim {box.dim} (expected 3)"

/--
Run the softmax workflow under the scalar backend `α`: lower `model` with fixed weights to verifier
IR and print its IBP and CROWN bounds. The input box is the ℓ∞ ball of radius `0.05` around
`(0.2, -0.1)`, and backward CROWN also bounds the margin `p₀ - p₁`.
-/
def run {α : Type} [Storage α] [Context α] [ToString α]
    [FromFloat α] [BoundOps α] [NonlinearBoundOps α] : IO Unit := do
  IO.println "== Workflow 1: linear -> softmax (vector) =="
  let cast : Float → α := ofFloat

  let params : TensorPack α (nn.Sequential.stateShapes model) :=
    Pack.nil
      |>.snoc
        (Tensor.map cast ([[1.0, -0.5], [0.2, 0.7], [-0.3, 0.1]] : Tensor Float [3, 2]))
      |>.snoc
        (Tensor.map cast ([0.1, -0.2, 0.0] : Tensor Float [3]))

  let lowered ← IO.ofExcept <|
    IR.lowerForward (α := α) (nn.Sequential.forward model (α := α)) params

  IO.println s!"lowered IR nodes: {lowered.graph.nodes.size}"
  let imported := lowered.toImported

  let x0 : Tensor α [2] := Tensor.map cast ([0.2, -0.1] : Tensor Float [2])
  let eps : α := ofFloat 0.05
  let xB : FlatBox α := FlatBox.lInfBall (α := α) x0 eps

  -- IBP
  let boxes := imported.runIBP xB
  let outB ← IO.ofExcept (imported.outputBox? boxes)
  print "IBP" outB

  -- CROWN output bounds.
  let outC ← IO.ofExcept (imported.outputBoxCROWN? xB)
  print "CROWN" outC

  -- Backward/dual CROWN for the margin objective: p0 - p1.
  let objV : Tensor α [3] := Tensor.map cast ([1.0, -1.0, 0.0] : Tensor Float [3])
  let obj : FlatTensor α := { n := 3, v := objV }
  let outObjective ←
    match imported.backwardObjectiveBox? boxes xB obj with
    | .ok result => pure result
    | .error msg => throw <| IO.userError s!"[CROWN-backward] {msg}"
  if outObjective.dim != 1 then
    throw <| IO.userError
      s!"[CROWN-backward] unexpected output dim {outObjective.dim} (expected 1)"
  let loM : α := getAtOrZero outObjective.lo [0]
  let hiM : α := getAtOrZero outObjective.hi [0]
  IO.println s!"[CROWN-backward] margin lo = {loM}"
  IO.println s!"[CROWN-backward] margin hi = {hiM}"

end Softmax

namespace Loss

/-- TorchLean forward program computing
$\widehat{y}=\operatorname{linear}(x)$ and
$\operatorname{mse\_loss}(\widehat{y},\mathrm{target})$, returning a scalar. Its inputs are the
weight, the bias, the target and `x`, in that order. -/
def model {α : Type} [Storage α] [Context α] :
    Program α [[2, 2], [2], [2], [2]] [] :=
  fun {m} _ _ =>
    fun w b target x =>
      (do
        let yhat ← functional.linear
          (m := m) (α := α)
          (leadingShape := []) (inDim := 2) (outDim := 2) w b x
        Ops.apply (m := m) (α := α) (.mseLoss (s := [2]))
          (yhat, target, ())
        : m (Ops.Ref (m := m) (α := α) []))

/--
Run the MSE-loss workflow under the scalar backend `α`: lower `model` with fixed weights and target
`(0, 1)`, then print IBP, CROWN and backward CROWN bounds on the loss over the ℓ∞ ball of radius
`0.05` around `(0.3, -0.4)`.
-/
def run {α : Type} [Storage α] [Context α] [ToString α]
    [FromFloat α] [BoundOps α] [NonlinearBoundOps α] : IO Unit := do
  IO.println "== Workflow 2: linear -> mse_loss (scalar) =="
  let cast : Float → α := ofFloat

  let params : TensorPack α [[2, 2], [2], [2]] :=
    Pack.nil
      |>.snoc
        (Tensor.map cast ([[0.4, -0.3], [1.2, 0.1]] : Tensor Float [2, 2]))
      |>.snoc
        (Tensor.map cast ([0.05, -0.02] : Tensor Float [2]))
      |>.snoc
        (Tensor.map cast ([0.0, 1.0] : Tensor Float [2]))

  let lowered ← IO.ofExcept <|
    IR.lowerForward (α := α) (model (α := α)) params

  IO.println s!"lowered IR nodes: {lowered.graph.nodes.size}"
  let imported := lowered.toImported

  let x0 : Tensor α [2] := Tensor.map cast ([0.3, -0.4] : Tensor Float [2])
  let eps : α := ofFloat 0.05
  let xB : FlatBox α := FlatBox.lInfBall (α := α) x0 eps

  -- IBP
  let boxes := imported.runIBP xB
  let outB ← IO.ofExcept (imported.outputBox? boxes)
  if outB.dim != 1 then
    throw <| IO.userError s!"[IBP] unexpected output dim {outB.dim} (expected 1)"
  IO.println s!"[IBP] loss lo = {pretty outB.lo}"
  IO.println s!"[IBP] loss hi = {pretty outB.hi}"

  -- CROWN output bounds on the scalar loss.
  let outC ← IO.ofExcept (imported.outputBoxCROWN? xB)
  if outC.dim != 1 then
    throw <| IO.userError s!"[CROWN] unexpected output dim {outC.dim} (expected 1)"
  IO.println s!"[CROWN] loss lo = {pretty outC.lo}"
  IO.println s!"[CROWN] loss hi = {pretty outC.hi}"

  -- Backward/dual CROWN for the loss objective itself (obj = 1).
  let obj : FlatTensor α := { n := 1, v := Tensor.full [1] 1 }
  let outObjective ←
    match imported.backwardObjectiveBox? boxes xB obj with
    | .ok result => pure result
    | .error msg => throw <| IO.userError s!"[CROWN-backward] {msg}"
  if outObjective.dim != 1 then
    throw <| IO.userError
      s!"[CROWN-backward] unexpected output dim {outObjective.dim} (expected 1)"
  IO.println s!"[CROWN-backward] loss lo = {pretty outObjective.lo}"
  IO.println s!"[CROWN-backward] loss hi = {pretty outObjective.hi}"

end Loss

/-- Run both CROWN examples under a chosen scalar backend `α`. -/
def run {α : Type} [Storage α] [Context α] [ToString α] [FromFloat α]
    [BoundOps α] [NonlinearBoundOps α] : IO Unit := do
  Softmax.run (α := α)
  IO.println ""
  Loss.run (α := α)

/--
CLI entry point of `scripts/lake.sh exe verify -- torchlean-crown-ops`. Its one flag,
`--arithmetic`, picks `Float32` (`native`, the default) or `ExecFloat.Binary 8 23` (`ieee`).
Missing bounds or unexpected output dimensions fail the command.
-/
def main (args : List String) : IO Unit :=
  BoundArithmetic.run
    "TorchLean → IR → IBP + CROWN (ops: softmax/mse_loss)" args
    (@run)

end TorchLean.Examples.CROWNOpsWorkflow

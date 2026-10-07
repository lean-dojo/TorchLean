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
  `scripts/lake.sh exe verify -- torchlean-crown-ops`
  `scripts/lake.sh exe verify -- torchlean-crown-ops --arithmetic ieee`
-/

@[expose] public section


namespace NN.Verification.Builtin.CrownOpsWorkflow

open _root_.Spec _root_.TorchLean

open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN

namespace Softmax

/-- TorchLean model: `Linear -> Softmax`. -/
def model : nn.Sequential [2] [3] :=
  nn.build 0 <|
    nn.Sequential![
      nn.linear 2 3,
      nn.softmax (shape := [3]) 0
    ]

/-- Print the softmax probability bounds of a verifier box under `tag`, together with the directed
margin `p₀ - p₁`. Fails when the box does not have the softmax output dimension. -/
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
Run the softmax workflow under a chosen scalar backend `α`.

This lowers the TorchLean model to verifier IR and prints IBP/CROWN bounds.
-/
def run {α : Type} [Storage α] [Context α] [ToString α]
    [Runtime.FromFloat α] [BoundOps α] [NonlinearBoundOps α] : IO Unit := do
  IO.println "== Workflow 1: linear -> softmax (vector) =="
  let cast : Float → α := Runtime.ofFloat

  let params : nn.State α (nn.stateShapes model) :=
    nn.State.empty
      |>.push
        (Tensor.map cast ([[1.0, -0.5], [0.2, 0.7], [-0.3, 0.1]] : Tensor Float [3, 2]))
      |>.push
        (Tensor.map cast ([0.1, -0.2, 0.0] : Tensor Float [3]))

  let lowered ← IO.ofExcept <|
    Verification.lowerProgramToIR (α := α) (nn.forward model (α := α)) params

  IO.println s!"lowered IR nodes: {lowered.graph.nodes.size}"

  let x0 : Tensor α [2] := Tensor.map cast ([0.2, -0.1] : Tensor Float [2])
  let eps : α := Runtime.ofFloat 0.05
  let xB : FlatBox α := NN.Verification.Builtin.lInfBall (α := α) x0 eps
  let ps : ParamStore α := lowered.seedInputBox xB

  -- IBP
  let ibp := lowered.runIBP ps
  let outB ← IO.ofExcept (lowered.outputBox? ibp)
  print "IBP" outB

  -- CROWN output bounds.
  let outC ← IO.ofExcept (lowered.outputBoxCROWN? ps xB)
  print "CROWN" outC

  -- Backward/dual CROWN for the margin objective: p0 - p1.
  let objV : Tensor α [3] := Tensor.map cast ([1.0, -1.0, 0.0] : Tensor Float [3])
  let obj : FlatTensor α := { n := 3, v := objV }
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

end Softmax

namespace Loss

/-- TorchLean forward program computing
$\widehat{y}=\operatorname{linear}(x)$ and
$\operatorname{mse\_loss}(\widehat{y},\mathrm{target})$, returning a scalar. -/
def model {α : Type} [Storage α] [Context α] :
    _root_.Runtime.Autograd.Model.Program α [[2, 2], [2], [2], [2]] [] :=
  fun {m} _ _ =>
    fun w b target x =>
      (do
        let yhat ← Runtime.linear
          (m := m) (α := α)
          (batchShape := []) (inputWidth := 2) (outputWidth := 2) w b x
        Runtime.mseLoss (m := m) (α := α) (s := [2]) yhat target
        : m (Runtime.ValueRef (m := m) (α := α) []))

/--
Run the MSE-loss workflow under a chosen scalar backend `α`.

This lowers the TorchLean forward computation to verifier IR and prints IBP/CROWN bounds for the
scalar loss.
-/
def run {α : Type} [Storage α] [Context α] [ToString α]
    [Runtime.FromFloat α] [BoundOps α] [NonlinearBoundOps α] : IO Unit := do
  IO.println "== Workflow 2: linear -> mse_loss (scalar) =="
  let cast : Float → α := Runtime.ofFloat

  let params : nn.State α [[2, 2], [2], [2]] :=
    nn.State.empty
      |>.push
        (Tensor.map cast ([[0.4, -0.3], [1.2, 0.1]] : Tensor Float [2, 2]))
      |>.push
        (Tensor.map cast ([0.05, -0.02] : Tensor Float [2]))
      |>.push
        (Tensor.map cast ([0.0, 1.0] : Tensor Float [2]))

  let lowered ← IO.ofExcept <|
    Verification.lowerProgramToIR (α := α) (model (α := α)) params

  IO.println s!"lowered IR nodes: {lowered.graph.nodes.size}"

  let x0 : Tensor α [2] := Tensor.map cast ([0.3, -0.4] : Tensor Float [2])
  let eps : α := Runtime.ofFloat 0.05
  let xB : FlatBox α := NN.Verification.Builtin.lInfBall (α := α) x0 eps
  let ps : ParamStore α := lowered.seedInputBox xB

  -- IBP
  let ibp := lowered.runIBP ps
  let outB ← IO.ofExcept (lowered.outputBox? ibp)
  if outB.dim != 1 then
    throw <| IO.userError s!"[IBP] unexpected output dim {outB.dim} (expected 1)"
  IO.println s!"[IBP] loss lo = {pretty outB.lo}"
  IO.println s!"[IBP] loss hi = {pretty outB.hi}"

  -- CROWN output bounds on the scalar loss.
  let outC ← IO.ofExcept (lowered.outputBoxCROWN? ps xB)
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

end Loss

/-- Run both CROWN examples under a chosen scalar backend `α`. -/
def run {α : Type} [Storage α] [Context α] [ToString α] [Runtime.FromFloat α]
    [BoundOps α] [NonlinearBoundOps α] : IO Unit := do
  Softmax.run (α := α)
  IO.println ""
  Loss.run (α := α)

/--
CLI entry point for the CROWN-ops workflow.

This is wired into `scripts/lake.sh exe verify -- torchlean-crown-ops`.
Missing bounds or unexpected output dimensions fail the command.
-/
def main (args : List String) : IO Unit :=
  NN.Verification.Builtin.runWithBoundArithmetic
    "TorchLean → IR → IBP + CROWN (ops: softmax/mse_loss)" args
    (@run)

end NN.Verification.Builtin.CrownOpsWorkflow

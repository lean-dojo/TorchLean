/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN
public import NN.Verification.TorchLean.Compile

/-!
# TorchLean IBP

Small end-to-end workflow:

TorchLean forward model → compile to `NN.IR.Graph` → run Lean IBP (`runIBP`).

Run:
  `lake exe verify -- torchlean-ibp`
  `lake exe verify -- torchlean-ibp --dtype ieee754exec`
  `lake exe verify -- torchlean-ibp --dtype float32`
-/

@[expose] public section


namespace NN.Examples.Verification.TorchLean.TorchLeanIBP

open _root_.Spec
open _root_.Spec.Tensor
open NN.API

open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN

/-- Input dimension for the small MLP in this workflow. -/
def inDim : Nat := 2
/-- Hidden width for the small MLP in this workflow. -/
def hidDim : Nat := 3
/-- Output dimension for the small MLP in this workflow. -/
def outDim : Nat := 1

/-- Input shape for the workflow model. -/
def xShape : Shape := Shape.Vec inDim
/-- Output shape for the workflow model. -/
def yShape : Shape := Shape.Vec outDim

/-- TorchLean model used in the workflow (a 2-layer ReLU MLP). -/
def mkModel : nn.M (nn.Sequential xShape yShape) :=
  nn.sequential![
    nn.linear inDim hidDim (pfx := Spec.Shape.scalar),
    nn.relu,
    nn.linear hidDim outDim (pfx := Spec.Shape.scalar)
  ]

def model : nn.Sequential xShape yShape :=
  nn.build 0 mkModel

/-- Parameter shapes for `model`. -/
def paramShapes : List Shape := nn.paramShapes model

/--
CLI entry point for the TorchLean → IR → IBP workflow.

This is wired into `lake exe verify -- torchlean-ibp`.
-/
def main (args : List String) : IO Unit := do
  NN.API.Common.runWithRuntimeDType "TorchLean → IR → IBP (small MLP)" args
    (fun {α} _ _ _ _ => do
      let cast : Float → α := Runtime.ofFloat
      let params : tlist.TList α paramShapes :=
        tlist!
          (NN.Tensor.tensorNDOfLenEq (α := α) [3, 2]
            [cast 0.1, cast 0.2, cast 0.3, cast 0.4, cast 0.5, cast 0.6] (by rfl)),
          (NN.Tensor.tensorNDOfLenEq (α := α) [3]
            [cast 0.1, cast 0.2, cast 0.3] (by rfl)),
          (NN.Tensor.tensorNDOfLenEq (α := α) [1, 3]
            [cast 0.7, cast 0.8, cast 0.9] (by rfl)),
          (NN.Tensor.tensorNDOfLenEq (α := α) [1]
            [cast 0.4] (by rfl))

      let compiled ←
        match NN.Verification.TorchLean.compileForward1
              (α := α) (paramShapes := paramShapes) (inShape := xShape) (outShape := yShape)
              (nn.program (model := model) (α := α)) params with
        | .ok c => pure c
        | .error e => throw <| IO.userError e

      IO.println s!"compiled IR nodes: {compiled.graph.nodes.size}"

      let x0 : Tensor α xShape :=
        NN.Tensor.tensorNDOfLenEq (α := α) [2] [cast 0.5, cast 0.8] (by rfl)
      let eps : α := Runtime.ofFloat 0.1
      let rad : Tensor α xShape := Spec.fill eps xShape

      let xB : FlatBox α :=
        { dim := inDim
          lo := Tensor.subSpec x0 rad
          hi := Tensor.addSpec x0 rad }

      let ps : ParamStore α :=
        { compiled.ps with inputBoxes := compiled.ps.inputBoxes.insert compiled.inputId xB }

      let boxes := runIBP (α := α) compiled.graph ps
      let some outB := boxes[compiled.outputId]!
        | throw <| IO.userError "IBP produced no output box"
      IO.println s!"output box lo: {pretty outB.lo}"
      IO.println s!"output box hi: {pretty outB.hi}"
    )

end NN.Examples.Verification.TorchLean.TorchLeanIBP

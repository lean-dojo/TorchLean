/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN
public import NN.Verification.TorchLean.Compile

/-!
# TorchLean transformer IBP

Small end-to-end workflow:

TorchLean (MHA + LayerNorm + MSE) → compile to `NN.IR.Graph` → run:
- IBP (`runIBP`)
- basic CROWN forward bounds (`runCROWN`)
- objective-dependent backward/dual CROWN (`runCROWNBackwardObjective`)

Run:
  `lake exe verify -- torchlean-transformer-ibp`
  `lake exe verify -- torchlean-transformer-ibp --with-crown`
  `lake exe verify -- torchlean-transformer-ibp --dtype ieee754exec`
-/

@[expose] public section


namespace NN.Examples.Verification.TorchLean.TorchLeanTransformerIBP

open _root_.Spec
open _root_.Spec.Tensor
open NN.API

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
def xShape : Shape := .dim batch (NN.Tensor.Shape.Mat n dModel)
/-- Projection weight shape for Q/K/V: `(dModel × (numHeads*headDim))`. -/
def wProjShape : Shape := NN.Tensor.Shape.Mat dModel (numHeads * headDim)
/-- Output projection weight shape: `((numHeads*headDim) × dModel)`. -/
def wOShape : Shape := NN.Tensor.Shape.Mat (numHeads * headDim) dModel
/-- LayerNorm gamma shape. -/
def gammaShape : Shape := NN.Tensor.Shape.Vec dModel
/-- LayerNorm beta shape. -/
def betaShape : Shape := NN.Tensor.Shape.Vec dModel
/-- MSE target shape (matches the model output shape). -/
def targetShape : Shape := xShape

/-- Parameter shapes list for `modelLoss` (`Wq,Wk,Wv,Wo,gamma,beta,target`). -/
def paramShapes : List Shape :=
  [wProjShape, wProjShape, wProjShape, wOShape, gammaShape, betaShape, targetShape]

/-- TorchLean program: `mha -> layer_norm -> mse_loss`, returning a scalar loss. -/
def modelLoss {α : Type} [Context α] [DecidableEq Shape] :
    TorchLean.Program α (paramShapes ++ [xShape]) Shape.scalar :=
  fun {m} _ _ =>
    fun wq wk wv wo gamma beta target x =>
      (do
        let y ← TorchLean.multiHeadAttention (m := m) (α := α)
          (batch := batch) (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
          (h1 := by decide) wq wk wv wo x (mask := none)
        let yLn ← TorchLean.layerNorm (m := m) (α := α)
          (batch := batch) (seqLen := n) (embedDim := dModel) (h_seq_pos := by decide)
          (h_embed_pos := by decide)
          y gamma beta
        TorchLean.mseLoss (m := m) (α := α) (s := xShape) yLn target
        : m (TorchLean.RefTy (m := m) (α := α) Shape.scalar))

/--
CLI entry point for the transformer-IBP workflow.

This is wired into `lake exe verify -- torchlean-transformer-ibp`.

By default this command is a fast validation check: compile the TorchLean transformer fragment to the
verification IR and run IBP on the scalar loss. Pass `--with-crown` to also run the experimental
transformer-scale CROWN passes. The separate `torchlean-crown-ops` command keeps CROWN itself in the
standard check suite on compact graphs, while this file focuses on the heavier attention/layer-norm
front-end path.
-/
def main (args : List String) : IO Unit := do
  let (withCrown, args) ←
    match NN.API.CLI.takeBoolFlagOnce args "with-crown" with
    | .ok parsed => pure parsed
    | .error msg => throw <| IO.userError msg
  NN.API.Common.runWithRuntimeDType "TorchLean (MHA+LayerNorm+MSE) → IR → IBP" args
    (fun {α} _ _ _ _ => do
      let cast : Float → α := Runtime.ofFloat
      let params : tlist.TList α paramShapes :=
        tlist!
          (NN.Tensor.tensorNDOfLenEq (α := α) [2, 2]
            [cast 1.0, cast 0.0, cast 0.0, cast 1.0] (by rfl)),
          (NN.Tensor.tensorNDOfLenEq (α := α) [2, 2]
            [cast 1.0, cast 0.0, cast 0.0, cast 1.0] (by rfl)),
          (NN.Tensor.tensorNDOfLenEq (α := α) [2, 2]
            [cast 1.0, cast 0.0, cast 0.0, cast 1.0] (by rfl)),
          (NN.Tensor.tensorNDOfLenEq (α := α) [2, 2]
            [cast 1.0, cast 0.0, cast 0.0, cast 1.0] (by rfl)),
          (NN.Tensor.tensorNDOfLenEq (α := α) [2]
            [cast 1.0, cast 1.0] (by rfl)),
          (NN.Tensor.tensorNDOfLenEq (α := α) [2]
            [cast 0.0, cast 0.0] (by rfl)),
          (NN.Tensor.tensorNDOfLenEq (α := α) [1, 2, 2]
            [cast 0.0, cast 0.0, cast 0.0, cast 0.0] (by rfl))

      let compiled ←
        match NN.Verification.TorchLean.compileForward1
              (α := α) (paramShapes := paramShapes) (inShape := xShape) (outShape := Shape.scalar)
              (modelLoss (α := α)) params with
        | .ok c => pure c
        | .error e => throw <| IO.userError e

      IO.println s!"compiled IR nodes: {compiled.graph.nodes.size}"

      let x0 : Tensor α xShape :=
        NN.Tensor.tensorNDOfLenEq (α := α) [1, 2, 2]
          [cast 0.2, cast (-0.3), cast 0.7, cast 0.1] (by rfl)
      let eps : α := Runtime.ofFloat 0.05
      let rad : Tensor α xShape := Spec.fill (α := α) eps xShape

      let loM := Tensor.subSpec x0 rad
      let hiM := Tensor.addSpec x0 rad
      let lo := Tensor.flattenSpec (α := α) loM
      let hi := Tensor.flattenSpec (α := α) hiM

      let xB : FlatBox α := { dim := Spec.Shape.size xShape, lo := lo, hi := hi }
      let ps : ParamStore α :=
        { compiled.ps with inputBoxes := compiled.ps.inputBoxes.insert compiled.inputId xB }

      let boxes := runIBP (α := α) compiled.graph ps
      let some outB := boxes[compiled.outputId]!
        | throw <| IO.userError "IBP produced no output box"
      IO.println s!"[IBP] loss lo: {pretty outB.lo}"
      IO.println s!"[IBP] loss hi: {pretty outB.hi}"

      if !withCrown then
        IO.println "[CROWN] skipped for the default smoke path; pass --with-crown for the heavier transformer CROWN run"
        return ()

      -- Basic CROWN forward bounds on the scalar loss (w.r.t. the input node).
      let inputDim := Spec.Shape.size xShape
      let ctx : AffineCtx := { inputId := compiled.inputId, inputDim := inputDim }
      let crown := runCROWN (α := α) compiled.graph ps ctx boxes
      match crown[compiled.outputId]! with
      | none =>
          IO.println "[CROWN] no affine bounds for loss"
      | some outAff =>
          if hIn : outAff.inDim = inputDim then
            if hOut : outAff.outDim = 1 then
              let xBox : Box α (.dim outAff.inDim .scalar) :=
                { lo := Tensor.castVecDim (α := α) (n := inputDim) (m := outAff.inDim) hIn.symm xB.lo
                  hi := Tensor.castVecDim (α := α) (n := inputDim) (m := outAff.inDim) hIn.symm xB.hi }
              let outLo := NN.MLTheory.CROWN.AffineVec.evalOnBox (α := α) outAff.loAff xBox
              let outHi := NN.MLTheory.CROWN.AffineVec.evalOnBox (α := α) outAff.hiAff xBox
              IO.println s!"[CROWN] loss lo: {pretty outLo.lo}"
              IO.println s!"[CROWN] loss hi: {pretty outHi.hi}"
            else
              IO.println s!"[CROWN] unexpected output dim {outAff.outDim} (expected 1)"
          else
            IO.println s!"[CROWN] unexpected input dim {outAff.inDim} (expected {inputDim})"

      -- Backward/dual CROWN for the objective `loss` itself (obj = 1).
      let obj : FlatVec α := { n := 1, v := Spec.fill (α := α) Numbers.one (.dim 1 .scalar) }
      match runCROWNBackwardObjective (α := α) compiled.graph ps ctx boxes compiled.outputId obj with
      | none =>
          IO.println "[CROWN-backward] no affine bounds for objective"
      | some objAff =>
          if hIn : objAff.inDim = inputDim then
            if hOut : objAff.outDim = 1 then
              let xBox : Box α (.dim objAff.inDim .scalar) :=
                { lo := Tensor.castVecDim (α := α) (n := inputDim) (m := objAff.inDim) hIn.symm xB.lo
                  hi := Tensor.castVecDim (α := α) (n := inputDim) (m := objAff.inDim) hIn.symm xB.hi }
              let outLo := NN.MLTheory.CROWN.AffineVec.evalOnBox (α := α) objAff.loAff xBox
              let outHi := NN.MLTheory.CROWN.AffineVec.evalOnBox (α := α) objAff.hiAff xBox
              IO.println s!"[CROWN-backward] loss lo: {pretty outLo.lo}"
              IO.println s!"[CROWN-backward] loss hi: {pretty outHi.hi}"
            else
              IO.println s!"[CROWN-backward] unexpected output dim {objAff.outDim} (expected 1)"
          else
            IO.println s!"[CROWN-backward] unexpected input dim {objAff.inDim} (expected {inputDim})"
    )

end NN.Examples.Verification.TorchLean.TorchLeanTransformerIBP

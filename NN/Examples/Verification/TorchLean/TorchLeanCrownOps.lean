/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN
public import NN.Verification.TorchLean.Compile

/-!
# TorchLean CROWN ops

Running CROWN end-to-end on small TorchLean graphs.

We compile TorchLean programs to the verifier IR (`NN.IR.Graph`), then run:
- IBP (`runIBP`)
- basic CROWN forward bounds (`runCROWN`)
- objective-dependent backward/dual CROWN (`runCROWNBackwardObjective`)

The goal is to provide compact, fast workflows that exercise the nonlinear ops added to CROWN:
- `softmax` (vector)
- `mse_loss` (vector → scalar)

For attention + `layer_norm`, see
  `NN/Examples/Verification/TorchLean/TorchLeanTransformerIBP.lean`.

Run:
  `lake exe verify -- torchlean-crown-ops`
  `lake exe verify -- torchlean-crown-ops --dtype ieee754exec`
-/

@[expose] public section


namespace NN.Examples.Verification.TorchLean.TorchLeanCrownOps

open _root_.Spec
open _root_.Spec.Tensor
open NN.API

open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN

/-- Input dimension for the softmax workflow model. -/
def softmaxInDim : Nat := 2
/-- Output dimension for the softmax workflow model. -/
def softmaxOutDim : Nat := 3

/-- Input shape for the softmax workflow model. -/
def softmaxXShape : Shape := .dim softmaxInDim .scalar
/-- Output shape for the softmax workflow model. -/
def softmaxYShape : Shape := .dim softmaxOutDim .scalar

/-- TorchLean model: `Linear -> Softmax`. -/
def softmaxModel : nn.Sequential softmaxXShape softmaxYShape :=
  nn.build 0 <|
    nn.sequential![
      nn.linear softmaxInDim softmaxOutDim (pfx := Spec.Shape.scalar),
      nn.softmax
    ]

/-- Parameter shapes for `softmaxModel`. -/
def softmaxParamShapes : List Shape := nn.paramShapes softmaxModel

/-- Example margin functional on softmax outputs (`lo0 - hi1`). -/
def softmaxMargin {α : Type} [Context α]
    (lo hi : Tensor α softmaxYShape) : α :=
  let lo0 := Tensor.vecGet lo fin0!
  let hi1 := Tensor.vecGet hi fin1!
  lo0 - hi1

/--
Run the softmax workflow under a chosen scalar backend `α`.

This compiles the TorchLean model to the verifier IR and prints IBP/CROWN bounds.
-/
def runSoftmax {α : Type} [Semantics.Scalar α] [DecidableEq Shape] [ToString α]
    [Runtime.Scalar α] : IO Unit := do
  IO.println "== Workflow 1: linear -> softmax (vector) =="
  let cast : Float → α := Runtime.ofFloat

  let params : tlist.TList α softmaxParamShapes :=
    tlist!
      (NN.Tensor.tensorNDOfLenEq (α := α) [3, 2]
        [ cast 1.0, cast (-0.5)
        , cast 0.2, cast 0.7
        , cast (-0.3), cast 0.1
        ] (by rfl)),
      (NN.Tensor.tensorNDOfLenEq (α := α) [3] [cast 0.1, cast (-0.2), cast 0.0] (by rfl))

  let compiled ←
    match NN.Verification.TorchLean.compileForward1
          (α := α) (paramShapes := softmaxParamShapes) (inShape := softmaxXShape) (outShape :=
            softmaxYShape)
          (nn.program (model := softmaxModel) (α := α)) params with
    | .ok c => pure c
    | .error e => throw <| IO.userError e

  IO.println s!"compiled IR nodes: {compiled.graph.nodes.size}"

  let x0 : Tensor α softmaxXShape :=
    NN.Tensor.tensorNDOfLenEq (α := α) [2] [cast 0.2, cast (-0.1)] (by rfl)
  let eps : α := Runtime.ofFloat 0.05
  let rad : Tensor α softmaxXShape := Spec.fill (α := α) eps softmaxXShape
  let xB : FlatBox α :=
    { dim := softmaxInDim
      lo := Tensor.subSpec x0 rad
      hi := Tensor.addSpec x0 rad }

  let ps : ParamStore α :=
    { compiled.ps with inputBoxes := compiled.ps.inputBoxes.insert compiled.inputId xB }

  -- IBP
  let ibp := runIBP (α := α) compiled.graph ps
  let some outB := ibp[compiled.outputId]! | throw <| IO.userError "IBP produced no output box"
  if hDim : outB.dim = softmaxOutDim then
    let loY : Tensor α softmaxYShape := by
      simpa [softmaxYShape] using
        Tensor.castVecDim (α := α) (n := outB.dim) (m := softmaxOutDim) hDim outB.lo
    let hiY : Tensor α softmaxYShape := by
      simpa [softmaxYShape] using
        Tensor.castVecDim (α := α) (n := outB.dim) (m := softmaxOutDim) hDim outB.hi
    IO.println s!"[IBP] p lo = {pretty loY}"
    IO.println s!"[IBP] p hi = {pretty hiY}"
    IO.println s!"[IBP] margin(p0 - p1) = {softmaxMargin (α := α) loY hiY}"
  else
    IO.println s!"[IBP] unexpected output dim {outB.dim} (expected {softmaxOutDim})"

  -- CROWN (forward, affine lower+upper)
  let ctx : AffineCtx := { inputId := compiled.inputId, inputDim := softmaxInDim }
  let crown := runCROWN (α := α) compiled.graph ps ctx ibp
  match crown[compiled.outputId]! with
  | none =>
      IO.println "[CROWN] no affine bounds for softmax output"
  | some outAff =>
      if hIn : outAff.inDim = softmaxInDim then
        if hOut : outAff.outDim = softmaxOutDim then
          let xBox : Box α (.dim outAff.inDim .scalar) :=
            { lo := Tensor.castVecDim (α := α) (n := softmaxInDim) (m := outAff.inDim) hIn.symm xB.lo
              hi := Tensor.castVecDim (α := α) (n := softmaxInDim) (m := outAff.inDim) hIn.symm xB.hi }
          let outLo := NN.MLTheory.CROWN.AffineVec.evalOnBox (α := α) outAff.loAff xBox
          let outHi := NN.MLTheory.CROWN.AffineVec.evalOnBox (α := α) outAff.hiAff xBox
          let loY : Tensor α softmaxYShape := by
            simpa [softmaxYShape] using
              Tensor.castVecDim (α := α) (n := outAff.outDim) (m := softmaxOutDim) hOut outLo.lo
          let hiY : Tensor α softmaxYShape := by
            simpa [softmaxYShape] using
              Tensor.castVecDim (α := α) (n := outAff.outDim) (m := softmaxOutDim) hOut outHi.hi
          IO.println s!"[CROWN] p lo = {pretty loY}"
          IO.println s!"[CROWN] p hi = {pretty hiY}"
          IO.println s!"[CROWN] margin(p0 - p1) = {softmaxMargin (α := α) loY hiY}"
        else
          IO.println s!"[CROWN] unexpected output dim {outAff.outDim} (expected {softmaxOutDim})"
      else
        IO.println s!"[CROWN] unexpected input dim {outAff.inDim} (expected {softmaxInDim})"

  -- Backward/dual CROWN for the margin objective: p0 - p1.
  let objV : Tensor α (.dim softmaxOutDim .scalar) :=
    NN.Tensor.tensorNDOfLenEq (α := α) [3] [cast 1.0, cast (-1.0), cast 0.0] (by rfl)
  let obj : FlatVec α := { n := softmaxOutDim, v := objV }
  match runCROWNBackwardObjective (α := α) compiled.graph ps ctx ibp compiled.outputId obj with
  | none =>
      IO.println "[CROWN-backward] no affine bounds for margin objective"
  | some objAff =>
      if hIn : objAff.inDim = softmaxInDim then
        if hOut : objAff.outDim = 1 then
          let xBox : Box α (.dim objAff.inDim .scalar) :=
            { lo := Tensor.castVecDim (α := α) (n := softmaxInDim) (m := objAff.inDim) hIn.symm xB.lo
              hi := Tensor.castVecDim (α := α) (n := softmaxInDim) (m := objAff.inDim) hIn.symm xB.hi }
          let outLo := NN.MLTheory.CROWN.AffineVec.evalOnBox (α := α) objAff.loAff xBox
          let outHi := NN.MLTheory.CROWN.AffineVec.evalOnBox (α := α) objAff.hiAff xBox
          let loM : α := getAtOrZero outLo.lo [0]
          let hiM : α := getAtOrZero outHi.hi [0]
          IO.println s!"[CROWN-backward] margin lo = {loM}"
          IO.println s!"[CROWN-backward] margin hi = {hiM}"
        else
          IO.println s!"[CROWN-backward] unexpected objective dim {objAff.outDim} (expected 1)"
      else
        IO.println
          s!"[CROWN-backward] unexpected input dim {objAff.inDim} (expected {softmaxInDim})"

/-- Input dimension for the MSE-loss workflow model. -/
def mseInDim : Nat := 2
/-- Output dimension for the MSE-loss workflow model. -/
def mseOutDim : Nat := 2

/-- Weight shape for the MSE-loss workflow's linear layer. -/
def mseWShape : Shape := .dim mseOutDim (.dim mseInDim .scalar)
/-- Bias shape for the MSE-loss workflow's linear layer. -/
def mseBShape : Shape := .dim mseOutDim .scalar
/-- Input shape for the MSE-loss workflow. -/
def mseXShape : Shape := .dim mseInDim .scalar
/-- Output shape for the MSE-loss workflow. -/
def mseYShape : Shape := .dim mseOutDim .scalar

/-- Parameter shapes for the MSE-loss workflow program (`[W,b,target]`). -/
def mseParamShapes : List Shape := [mseWShape, mseBShape, mseYShape]

/-- TorchLean program: `yhat = linear(x); mse_loss(yhat, target)` returning a scalar. -/
def mseLossModel {α : Type} [Context α] [DecidableEq Shape] :
    TorchLean.Program α (mseParamShapes ++ [mseXShape]) Shape.scalar :=
  fun {m} _ _ =>
    fun w b target x =>
      (do
        let yhat ← TorchLean.linear (m := m) (α := α) (inDim := mseInDim) (outDim := mseOutDim) w
          b x
        TorchLean.mseLoss (m := m) (α := α) (s := mseYShape) yhat target
        : m (TorchLean.RefTy (m := m) (α := α) Shape.scalar))

/--
Run the MSE-loss workflow under a chosen scalar backend `α`.

This compiles the TorchLean program to the verifier IR and prints IBP/CROWN bounds for the scalar
  loss.
-/
def runMSE {α : Type} [Semantics.Scalar α] [DecidableEq Shape] [ToString α]
    [Runtime.Scalar α] : IO Unit := do
  IO.println "== Workflow 2: linear -> mse_loss (scalar) =="
  let cast : Float → α := Runtime.ofFloat

  let params : tlist.TList α mseParamShapes :=
    tlist!
      (NN.Tensor.tensorNDOfLenEq (α := α) [2, 2]
        [cast 0.4, cast (-0.3), cast 1.2, cast 0.1] (by rfl)),
      (NN.Tensor.tensorNDOfLenEq (α := α) [2] [cast 0.05, cast (-0.02)] (by rfl)),
      (NN.Tensor.tensorNDOfLenEq (α := α) [2] [cast 0.0, cast 1.0] (by rfl))

  let compiled ←
    match NN.Verification.TorchLean.compileForward1
          (α := α) (paramShapes := mseParamShapes) (inShape := mseXShape) (outShape :=
            Shape.scalar)
          (mseLossModel (α := α)) params with
    | .ok c => pure c
    | .error e => throw <| IO.userError e

  IO.println s!"compiled IR nodes: {compiled.graph.nodes.size}"

  let x0 : Tensor α mseXShape :=
    NN.Tensor.tensorNDOfLenEq (α := α) [2] [cast 0.3, cast (-0.4)] (by rfl)
  let eps : α := Runtime.ofFloat 0.05
  let rad : Tensor α mseXShape := Spec.fill (α := α) eps mseXShape
  let xB : FlatBox α :=
    { dim := mseInDim
      lo := Tensor.subSpec x0 rad
      hi := Tensor.addSpec x0 rad }

  let ps : ParamStore α :=
    { compiled.ps with inputBoxes := compiled.ps.inputBoxes.insert compiled.inputId xB }

  let ctx : AffineCtx := { inputId := compiled.inputId, inputDim := mseInDim }

  -- IBP
  let ibp := runIBP (α := α) compiled.graph ps
  let some outB := ibp[compiled.outputId]! | throw <| IO.userError "IBP produced no output box"
  IO.println s!"[IBP] loss lo = {pretty outB.lo}"
  IO.println s!"[IBP] loss hi = {pretty outB.hi}"

  -- CROWN forward bounds on the scalar loss.
  let crown := runCROWN (α := α) compiled.graph ps ctx ibp
  match crown[compiled.outputId]! with
  | none =>
      IO.println "[CROWN] no affine bounds for mse_loss"
  | some outAff =>
      if hIn : outAff.inDim = mseInDim then
        if _hOut : outAff.outDim = 1 then
          let xBox : Box α (.dim outAff.inDim .scalar) :=
            { lo := Tensor.castVecDim (α := α) (n := mseInDim) (m := outAff.inDim) hIn.symm xB.lo
              hi := Tensor.castVecDim (α := α) (n := mseInDim) (m := outAff.inDim) hIn.symm xB.hi }
          let outLo := NN.MLTheory.CROWN.AffineVec.evalOnBox (α := α) outAff.loAff xBox
          let outHi := NN.MLTheory.CROWN.AffineVec.evalOnBox (α := α) outAff.hiAff xBox
          IO.println s!"[CROWN] loss lo = {pretty outLo.lo}"
          IO.println s!"[CROWN] loss hi = {pretty outHi.hi}"
        else
          IO.println s!"[CROWN] unexpected output dim {outAff.outDim} (expected 1)"
      else
        IO.println s!"[CROWN] unexpected input dim {outAff.inDim} (expected {mseInDim})"

  -- Backward/dual CROWN for the loss objective itself (obj = 1).
  let obj : FlatVec α := { n := 1, v := Spec.fill (α := α) Numbers.one (.dim 1 .scalar) }
  match runCROWNBackwardObjective (α := α) compiled.graph ps ctx ibp compiled.outputId obj with
  | none =>
      IO.println "[CROWN-backward] no affine bounds for loss objective"
  | some objAff =>
      if hIn : objAff.inDim = mseInDim then
        if _hOut : objAff.outDim = 1 then
          let xBox : Box α (.dim objAff.inDim .scalar) :=
            { lo := Tensor.castVecDim (α := α) (n := mseInDim) (m := objAff.inDim) hIn.symm xB.lo
              hi := Tensor.castVecDim (α := α) (n := mseInDim) (m := objAff.inDim) hIn.symm xB.hi }
          let outLo := NN.MLTheory.CROWN.AffineVec.evalOnBox (α := α) objAff.loAff xBox
          let outHi := NN.MLTheory.CROWN.AffineVec.evalOnBox (α := α) objAff.hiAff xBox
          IO.println s!"[CROWN-backward] loss lo = {pretty outLo.lo}"
          IO.println s!"[CROWN-backward] loss hi = {pretty outHi.hi}"
        else
          IO.println s!"[CROWN-backward] unexpected objective dim {objAff.outDim} (expected 1)"
      else
        IO.println s!"[CROWN-backward] unexpected input dim {objAff.inDim} (expected {mseInDim})"

/-- Run all CROWN-ops workflows (softmax + mse_loss) under a chosen scalar backend `α`. -/
def runOnce {α : Type} [Semantics.Scalar α] [DecidableEq Shape] [ToString α]
    [Runtime.Scalar α] : IO Unit := do
  runSoftmax (α := α)
  IO.println ""
  runMSE (α := α)

/--
CLI entry point for the CROWN-ops workflow.

This is wired into `lake exe verify -- torchlean-crown-ops`.
-/
def main (args : List String) : IO Unit :=
  NN.API.Common.mainWithRuntimeDType "TorchLean → IR → IBP + CROWN (ops: softmax/mse_loss)" args
    (fun {α} _ _ _ _ => runOnce (α := α))

end NN.Examples.Verification.TorchLean.TorchLeanCrownOps

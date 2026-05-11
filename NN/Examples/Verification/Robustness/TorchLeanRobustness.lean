/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN
public import NN.MLTheory.CROWN.Core
public import NN.MLTheory.CROWN.Graph
public import NN.Verification.TorchLean.Compile

/-!
# TorchLean robustness

End-to-end robustness certification for a TorchLean model.

We build a compact 2-class classifier in TorchLean, compile it to the verifier IR, and certify a
margin condition on an `ℓ∞` input box using:

- IBP (`runIBP`)
- a simple CROWN/affine pass (`runAffine` + `AffineVec.eval_on_box`)

Spec we certify (binary logits):
  `∀ x ∈ [x0-ε, x0+ε], logit0(x) > logit1(x)`

Run:
  `lake exe verify -- torchlean-robustness`
  `lake exe verify -- torchlean-robustness --float32-mode ieee754exec`
-/

@[expose] public section


namespace NN.Examples.Verification.Robustness.TorchLeanRobustness

open _root_.Spec
open _root_.Spec.Tensor
open NN.API

open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN

/-- Input dimension for the TorchLean robustness example. -/
def inDim : Nat := 2
/-- Hidden width for the TorchLean robustness example. -/
def hidDim : Nat := 1
/-- Number of output logits/classes. -/
def outDim : Nat := 2

/-- First-layer weight shape. -/
def W1Shape : Shape := .dim hidDim (.dim inDim .scalar)
/-- First-layer bias shape. -/
def b1Shape : Shape := .dim hidDim .scalar
/-- Second-layer weight shape. -/
def W2Shape : Shape := .dim outDim (.dim hidDim .scalar)
/-- Second-layer bias shape. -/
def b2Shape : Shape := .dim outDim .scalar
/-- Input shape. -/
def xShape : Shape := .dim inDim .scalar
/-- Output/logit shape. -/
def yShape : Shape := .dim outDim .scalar

/-- Parameter shapes list used by the compiled TorchLean program (`[W1,b1,W2,b2]`). -/
def paramShapes : List Shape := [W1Shape, b1Shape, W2Shape, b2Shape]

/-- Compute a (conservative) margin lower bound `lo0 - hi1` from logit bounds. -/
def margin {α : Type} [Context α]
    (lo hi : Tensor α yShape) : α :=
  let lo0 := Tensor.vecGet lo fin0!
  let hi1 := Tensor.vecGet hi fin1!
  lo0 - hi1

/-- Decide if the output bounds certify `logit0 > logit1`. -/
def certifiedMargin {α : Type} [Context α]
    (lo hi : Tensor α yShape) : Bool :=
  let lo0 := Tensor.vecGet lo fin0!
  let hi1 := Tensor.vecGet hi fin1!
  Context.gtBool lo0 hi1

/-- TorchLean program for a 2-layer ReLU MLP producing two logits. -/
def classifier {α : Type} [Context α] [DecidableEq Shape] :
    TorchLean.Program α (paramShapes ++ [xShape]) yShape :=
  fun {m} _ _ =>
    fun w1 b1 w2 b2 x =>
      (do
        let z1 ← TorchLean.linear (m := m) (α := α) (inDim := inDim) (outDim := hidDim) w1 b1 x
        let h ← TorchLean.relu (m := m) (α := α) (s := b1Shape) z1
        TorchLean.linear (m := m) (α := α) (inDim := hidDim) (outDim := outDim) w2 b2 h
        : m (TorchLean.RefTy (m := m) (α := α) yShape))

/--
Run the robustness check once under a chosen scalar backend `α`.

This compiles the TorchLean program to the verifier IR, then computes output bounds with IBP and an
affine/CROWN-style pass.
-/
def runOnce {α : Type} [Semantics.Scalar α] [DecidableEq Shape] [ToString α]
    [Runtime.Scalar α] : IO Unit := do
  let cast : Float → α := Runtime.ofFloat
  -- These in-source constants are intentional: this workflow is a compact TorchLean-native
  -- verifier path check, not a data-backed benchmark. Data-backed robustness uses
  -- `NN.Verification.Robustness.Digits`, which loads weights and examples from JSON assets.
  --
  -- The chosen weights keep the hidden pre-activation positive over the whole ε-box, so the ReLU
  -- stays linear and the expected certified margin is easy to inspect by hand.
  let W1 : Tensor α W1Shape :=
    NN.Tensor.tensorNDOfLenEq (α := α) [1, 2] [cast 1.0, cast 1.0] (by rfl)
  let b1 : Tensor α b1Shape :=
    NN.Tensor.tensorNDOfLenEq (α := α) [1] [cast 0.0] (by rfl)
  let W2 : Tensor α W2Shape :=
    NN.Tensor.tensorNDOfLenEq (α := α) [2, 1] [cast 1.0, cast (-1.0)] (by rfl)
  let b2 : Tensor α b2Shape :=
    NN.Tensor.tensorNDOfLenEq (α := α) [2] [cast 0.0, cast 0.0] (by rfl)

  let params : tlist.TList α paramShapes :=
    tlist! W1, b1, W2, b2

  let compiled ←
    match NN.Verification.TorchLean.compileForward1
          (α := α) (paramShapes := paramShapes) (inShape := xShape) (outShape := yShape)
          (classifier (α := α)) params with
    | .ok c => pure c
    | .error e => throw <| IO.userError e

  let x0 : Tensor α xShape :=
    NN.Tensor.tensorNDOfLenEq (α := α) [2] [cast 1.0, cast 1.0] (by rfl)
  let eps : α := Runtime.ofFloat 0.1
  let rad : Tensor α xShape := Spec.fill (α := α) eps xShape

  let xB : FlatBox α :=
    { dim := inDim
      lo := Tensor.subSpec x0 rad
      hi := Tensor.addSpec x0 rad }

  let ps : ParamStore α :=
    { compiled.ps with inputBoxes := compiled.ps.inputBoxes.insert compiled.inputId xB }

  IO.println s!"compiled IR nodes: {compiled.graph.nodes.size}"
  IO.println s!"x0 = {pretty x0}, eps = {eps}"

  -- IBP
  let ibp := runIBP (α := α) compiled.graph ps
  let some outB := ibp[compiled.outputId]! | throw <| IO.userError "IBP produced no output box"
  IO.println s!"[IBP] logits lo = {pretty outB.lo}"
  IO.println s!"[IBP] logits hi = {pretty outB.hi}"
  if h : outB.dim = outDim then
    let loY : Tensor α yShape := by
      simpa [yShape] using Tensor.castVecDim (α := α) (n := outB.dim) (m := outDim) h outB.lo
    let hiY : Tensor α yShape := by
      simpa [yShape] using Tensor.castVecDim (α := α) (n := outB.dim) (m := outDim) h outB.hi
    IO.println s!"[IBP] margin(lo0 - hi1) = {margin (α := α) loY hiY}"
    IO.println s!"[IBP] certified? {certifiedMargin (α := α) loY hiY}"
  else
    IO.println s!"[IBP] unexpected output dim {outB.dim} (expected {outDim}); skipping margin check"

  -- CROWN / affine (w.r.t. the input node)
  let ctx : AffineCtx := { inputId := compiled.inputId, inputDim := inDim }
  let affs := runAffine (α := α) compiled.graph ps ctx ibp
  match affs[compiled.outputId]! with
  | none =>
      IO.println "[CROWN] no affine form for output (unexpected for linear/relu/linear)"
  | some outAff =>
      if hIn : outAff.inDim = inDim then
        if hOut : outAff.outDim = outDim then
          let xBox : Box α (.dim outAff.inDim .scalar) :=
            { lo := Tensor.castVecDim (α := α) (n := inDim) (m := outAff.inDim) hIn.symm xB.lo
              hi := Tensor.castVecDim (α := α) (n := inDim) (m := outAff.inDim) hIn.symm xB.hi }
          let outC := NN.MLTheory.CROWN.AffineVec.evalOnBox (α := α) outAff.aff xBox
          let loY : Tensor α yShape := by
            simpa [yShape] using
              Tensor.castVecDim (α := α) (n := outAff.outDim) (m := outDim) hOut outC.lo
          let hiY : Tensor α yShape := by
            simpa [yShape] using
              Tensor.castVecDim (α := α) (n := outAff.outDim) (m := outDim) hOut outC.hi
          IO.println s!"[CROWN] logits lo = {pretty loY}"
          IO.println s!"[CROWN] logits hi = {pretty hiY}"
          IO.println s!"[CROWN] margin(lo0 - hi1) = {margin (α := α) loY hiY}"
          IO.println s!"[CROWN] certified? {certifiedMargin (α := α) loY hiY}"
        else
          IO.println <|
            (s!"[CROWN] unexpected output dim {outAff.outDim} (expected {outDim}); " ++
              s!"skipping margin check")
      else
        IO.println
          s!"[CROWN] unexpected input dim {outAff.inDim} (expected {inDim}); skipping affine eval"

  -- Backward/dual CROWN (objective-dependent) for the margin: logit0 - logit1.
  let objV : Tensor α (.dim outDim .scalar) :=
    NN.Tensor.tensorNDOfLenEq (α := α) [2] [cast 1.0, cast (-1.0)] (by rfl)
  let obj : FlatVec α := { n := outDim, v := objV }
  match runCROWNBackwardObjective (α := α) compiled.graph ps ctx ibp compiled.outputId obj with
  | none =>
      IO.println "[CROWN-backward] no affine bounds for margin objective"
  | some objAff =>
      if hIn : objAff.inDim = inDim then
        if hOut : objAff.outDim = 1 then
          let xBox : Box α (.dim objAff.inDim .scalar) :=
            { lo := Tensor.castVecDim (α := α) (n := inDim) (m := objAff.inDim) hIn.symm xB.lo
              hi := Tensor.castVecDim (α := α) (n := inDim) (m := objAff.inDim) hIn.symm xB.hi }
          let outLo := NN.MLTheory.CROWN.AffineVec.evalOnBox (α := α) objAff.loAff xBox
          let outHi := NN.MLTheory.CROWN.AffineVec.evalOnBox (α := α) objAff.hiAff xBox
          let loM : α := getAtOrZero outLo.lo [0]
          let hiM : α := getAtOrZero outHi.hi [0]
          IO.println s!"[CROWN-backward] margin lo = {loM}"
          IO.println s!"[CROWN-backward] margin hi = {hiM}"
          IO.println s!"[CROWN-backward] certified? {Context.gtBool loM (0 : α)}"
        else
          IO.println s!"[CROWN-backward] unexpected objective dim {objAff.outDim} (expected 1)"
      else
        IO.println s!"[CROWN-backward] unexpected input dim {objAff.inDim} (expected {inDim})"

/--
CLI entry point for the TorchLean robustness workflow.

This is wired into `lake exe verify -- torchlean-robustness`.
-/
def main (args : List String) : IO Unit :=
  NN.API.Common.mainWithRuntimeDType "TorchLean → IR → IBP/CROWN robustness" args
    (fun {α} _ _ _ _ => runOnce (α := α))

end NN.Examples.Verification.Robustness.TorchLeanRobustness

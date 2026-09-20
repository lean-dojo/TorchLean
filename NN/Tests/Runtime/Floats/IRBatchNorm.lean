/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine
public import NN.Runtime.Autograd.IRExec
public import NN.Runtime.PyTorch.Export.IRPyTorch
public import NN.Tests.Runtime.Floats.Utils

/-!
# IR BatchNorm Checks

Regression checks for the first-class eval-mode BatchNorm IR node.  This fixture chooses channel
axis `1` in a rank-four tensor; the IR operation itself accepts any valid channel axis.
-/

@[expose] public section

namespace Tests
namespace Floats
namespace IRBatchNorm

open Spec TorchLean
open TorchLean TorchLean.Tensor
open NN.IR
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open Tests.Utils
open Tests.Floats.Utils

abbrev n : Nat := 2
abbrev c : Nat := 2
abbrev h : Nat := 2
abbrev w : Nat := 2
abbrev inputShape : Shape := [n, c, h, w]

def input : Tensor Float inputShape :=
  Tensor.generate [n, c, h, w] fun coordinate =>
    let ni := coordinate.getD 0 0
    let ci := coordinate.getD 1 0
    let hi := coordinate.getD 2 0
    let wi := coordinate.getD 3 0
    let base := Float.ofNat (ni * 8 + ci * 4 + hi * 2 + wi + 1)
    if ci = 0 then base else -base

def gamma : Tensor Float [c] := [1.0, 0.5]
def beta : Tensor Float [c] := [0.0, 0.1]
def mean : Tensor Float [c] := [2.0, -3.0]
def var : Tensor Float [c] := [4.0, 9.0]
def bnEps : Float := 1e-5

def graph : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := #[], kind := .input, outShape := inputShape },
      { id := 1, parents := #[0], kind := .batchNormEval 1 c, outShape := inputShape }
    ] }

def payload : Payload Float :=
  { batchNormEval? := fun id =>
      if id = 1 then
        some { c := c, gamma := gamma, beta := beta, mean := mean, var := var, eps := bnEps }
      else
        none }

def expected (x gamma beta mean var eps : Float) : Float :=
  ((x - mean) / Float.sqrt (max var 0.0 + eps)) * gamma + beta

def flatVal {n : Nat} (t : Tensor Float [n]) (i : Fin n) : Float :=
  (get t i).item

def expectSome {α : Type} (label : String) : Option α → IO α
  | some x => pure x
  | none => throw (IO.userError s!"ir_batchnorm: expected {label}")

def verifierBox : FlatBox Float :=
  { dim := Spec.Shape.size inputShape
    lo := Tensor.flattenSpec input
    hi := Tensor.flattenSpec input }

def verifierParams : ParamStore Float :=
  { inputBoxes := (Std.HashMap.emptyWithCapacity.insert 0 verifierBox)
    batchNormEval :=
      (Std.HashMap.emptyWithCapacity.insert 1
        { c := c, gamma := gamma, beta := beta, mean := mean, var := var, eps := bnEps }) }

def unitObjective : FlatTensor Float :=
  { n := Spec.Shape.size inputShape
    v := Tensor.ofFn fun i => if decide (i.val = 0) then 1.0 else 0.0 }

def expectInputShape (label : String) (v : Spec.SomeTensor Float) :
    IO (Tensor Float inputShape) := do
  match v with
  | ⟨s, t⟩ =>
      if hs : s = inputShape then
        pure (hs ▸ t)
      else
        throw (IO.userError s!"{label}: expected {repr inputShape}, got {repr s}")

def checkTensor (label : String) (y : Tensor Float inputShape) : IO Unit := do
  for ni in List.finRange n do
    for ci in List.finRange c do
      for hi in List.finRange h do
        for wi in List.finRange w do
          let want :=
            expected (tensorVal input [ni, ci, hi, wi]) (vecVal gamma ci) (vecVal beta ci)
              (vecVal mean ci) (vecVal var ci) bnEps
          assertApprox s!"{label}[{ni.val},{ci.val},{hi.val},{wi.val}]"
            (tensorVal y [ni, ci, hi, wi]) want 1e-5

def run : IO Unit := do
  IO.println "ir_batchnorm: begin"
  match Graph.checkShapes graph with
  | Except.ok () => pure ()
  | Except.error e => throw (IO.userError s!"ir_batchnorm: graph shape check failed: {e}")

  let yDenote ←
    match Graph.denote (α := Float) (g := graph) (payload := payload)
        (input := Spec.SomeTensor.mk (α := Float) inputShape input) (outputId := 1) with
    | .ok v => expectInputShape "ir_batchnorm denote" v
    | .error e => throw (IO.userError s!"ir_batchnorm: denote failed: {e}")
  checkTensor "ir_batchnorm denote" yDenote

  let exec ←
    match Runtime.Autograd.IRExec.lowerToForwardGraph (α := Float) graph payload with
    | Except.ok e => pure e
    | Except.error e => throw (IO.userError s!"ir_batchnorm: lowering failed: {e}")
  let inputExec : Tensor Float exec.inShape ←
    if hIn : exec.inShape = inputShape then
      pure (hIn.symm ▸ input)
    else
      throw (IO.userError s!"ir_batchnorm: typed graph input shape mismatch: {repr exec.inShape}")
  let vals := Runtime.Autograd.IRExec.ForwardGraph.denoteAll (α := Float) exec inputExec
  match vals[1]? with
  | some v =>
      let y ← expectInputShape "ir_batchnorm typed graph" v
      checkTensor "ir_batchnorm typed graph" y
  | none =>
      throw (IO.userError "ir_batchnorm: typed graph output node missing")

  let ibp := runIBP (α := Float) graph verifierParams
  let yB ← expectSome "IBP BatchNorm box" ibp[1]!
  let want0 := expected (tensorVal input [0, 0, 0, 0]) (vecVal gamma 0) (vecVal beta 0)
    (vecVal mean 0) (vecVal var 0) bnEps
  if hdim : yB.dim = Spec.Shape.size inputShape then
    let lo : Tensor Float [Spec.Shape.size inputShape] :=
      NN.MLTheory.CROWN.Graph.castDimScalar (α := Float) hdim yB.lo
    let hi : Tensor Float [Spec.Shape.size inputShape] :=
      NN.MLTheory.CROWN.Graph.castDimScalar (α := Float) hdim yB.hi
    assertApprox "ir_batchnorm ibp lo[0]" (flatVal lo ⟨0, by decide⟩) want0 1e-5
    assertApprox "ir_batchnorm ibp hi[0]" (flatVal hi ⟨0, by decide⟩) want0 1e-5
  else
    throw (IO.userError s!"ir_batchnorm: IBP output dimension mismatch: {yB.dim}")

  let ctx : AffineCtx := { inputId := 0, inputDim := Spec.Shape.size inputShape }
  let aff := runAffine (α := Float) graph verifierParams ctx ibp
  let _ ← expectSome "affine BatchNorm transfer" aff[1]!
  let crown := runCROWN (α := Float) graph verifierParams ctx ibp
  let _ ← expectSome "CROWN BatchNorm transfer" crown[1]!
  let backward ← expectSome "backward CROWN BatchNorm transfer"
    (runCROWNBackwardObjective (α := Float) graph verifierParams ctx ibp 1 unitObjective)
  let loCoeffs := Tensor.flattenSpec backward.loAff.A
  let hiCoeffs := Tensor.flattenSpec backward.hiAff.A
  for i in List.finRange
      (Spec.Shape.size [backward.outDim, backward.inDim]) do
    assertApprox s!"ir_batchnorm backward lower coefficient[{i.val}]"
      (flatVal loCoeffs i) 0.0 0.0
    assertApprox s!"ir_batchnorm backward upper coefficient[{i.val}]"
      (flatVal hiCoeffs i) 0.0 0.0

  let pyCode ←
    match Export.IRPyTorch.emit graph verifierParams 0 1 with
    | .ok code => pure code
    | .error e => throw (IO.userError s!"ir_batchnorm: IR->PyTorch export failed: {e}")
  unless pyCode.contains "_eps = 0.000010" do
    throw (IO.userError "ir_batchnorm: IR->PyTorch export did not preserve BatchNorm eps")

  IO.println "ir_batchnorm: ok"

end IRBatchNorm
end Floats
end Tests

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tests.Runtime.Floats.Utils
public import NN.Verification.Builtin.ExecutableLowering
public import NN.API.Seeded
public import NN.MLTheory.CROWN.Graph.Engine.Derivatives -- shake: keep
public import NN.MLTheory.CROWN.Graph.Engine.Affine

/-!
# TorchLeanIRExecEquivCheck

Runtime check: IR denotation agrees with the executable `IRExec` bridge.

We lower a small TorchLean model to `NN.IR.Graph` with its payload, then lower that IR to an
`ForwardGraph` and check that both evaluators produce the same output tensor.
-/

@[expose] public section


open Spec TorchLean
open TorchLean TorchLean.Tensor
open Tests.Utils
open Tests.Floats.Utils

namespace Tests
namespace Floats
namespace TorchLeanIRExecEquivCheck

private def checkValues (label : String) (actual expected : Array (Spec.SomeTensor Float)) :
    IO Unit := do
  unless actual.size == expected.size do
    throw <| IO.userError s!"{label}: value-table lengths differ"
  for i in [:expected.size] do
    match actual[i]?, expected[i]? with
    | some a, some b =>
        unless a.shape == b.shape do
          throw <| IO.userError s!"{label}: shape mismatch at node {i}"
        let av := a.tensor.data
        let bv := b.tensor.data
        unless av.size == bv.size do
          throw <| IO.userError s!"{label}: tensor lengths differ at node {i}"
        for j in [:bv.size] do
          unless (av[j]!.isNaN && bv[j]!.isNaN) || av[j]!.toBits == bv[j]!.toBits do
            throw <| IO.userError s!"{label}: value mismatch at node {i}, entry {j}"
    | _, _ => throw <| IO.userError s!"{label}: missing node {i}"

/-- Array execution keeps every intermediate shape and value, including shared and older parents. -/
@[no_expose] def checkForwardContextValues : IO Unit := do
  let vector : Shape := [2]
  let graph : NN.IR.Graph :=
    { nodes := #[
        { id := 0, parents := #[], kind := .input, outShape := vector },
        { id := 1, parents := #[0], kind := .relu, outShape := vector },
        { id := 2, parents := #[1], kind := .sum, outShape := .scalar },
        { id := 3, parents := #[2], kind := .broadcastTo .scalar vector, outShape := vector },
        { id := 4, parents := #[0, 3], kind := .add, outShape := vector },
        { id := 5, parents := #[4], kind := .sum, outShape := .scalar },
        { id := 6, parents := #[2, 5], kind := .add, outShape := .scalar },
        { id := 7, parents := #[0], kind := .detach, outShape := vector }
      ] }
  let x : Tensor Float vector := [-3.0, 2.0]
  let expected ←
    match NN.IR.Graph.denoteAll graph {} (Spec.SomeTensor.ofTensor x) with
    | .ok values => pure values
    | .error error => throw <| IO.userError error
  let exec ←
    match Runtime.Autograd.IRExec.lowerToForwardGraph (α := Float) graph {} with
    | .ok exec => pure exec
    | .error error => throw <| IO.userError error
  if h : vector = exec.inShape then
    let input := Tensor.castShape x h
    checkValues "mixed forward contexts" (exec.denoteAll input) expected
    checkValues "typed forward contexts" (exec.eval input).toShapeErasedArray expected
  else
    throw <| IO.userError "mixed forward contexts: wrong input shape"

/-- The executable raw logarithm stays total; only the dynamic IR evaluator rejects its domain. -/
@[no_expose] def checkRawLogExecutionBoundary : IO Unit := do
  let graph : NN.IR.Graph :=
    { nodes := #[
        { id := 0, parents := #[], kind := .input, outShape := .scalar },
        { id := 1, parents := #[0], kind := .log, outShape := .scalar }
      ] }
  let exec ←
    match Runtime.Autograd.IRExec.lowerToForwardGraph (α := Float) graph {} with
    | .ok exec => pure exec
    | .error error => throw <| IO.userError error
  if h : Shape.scalar = exec.inShape then
    for value in [-1.0, 0.0, 1.0, 2.0] do
      let x := Tensor.scalar value
      let expected := #[Spec.SomeTensor.ofTensor x,
        Spec.SomeTensor.ofTensor (Tensor.logSpec x)]
      checkValues "raw log context" (exec.denoteAll (Tensor.castShape x h)) expected
      if value ≤ 0.0 then
        match NN.IR.Graph.denoteAll graph {} (Spec.SomeTensor.ofTensor x) with
        | .error _ => pure ()
        | .ok _ => throw <| IO.userError "raw log IR unexpectedly accepted a nonpositive input"
  else
    throw <| IO.userError "raw log contexts: wrong input shape"

/-- Hard-mask IBP keeps blocked entries exact and avoids uncertified transcendental rounding. -/
def checkHardMaskedSoftmaxIbpBoundary : IO Unit := do
  let logitsLo : Tensor Float [3] := [-2.0, 0.0, 1.0]
  let logitsHi : Tensor Float [3] := [3.0, 4.0, 5.0]
  let mixedMask : Tensor Bool [3] := [true, false, true]
  let singletonMask : Tensor Bool [3] := [false, true, false]
  let (mixedLo, mixedHi) :=
    NN.MLTheory.CROWN.Graph.ibpHardMaskedSoftmaxLastTensor logitsLo logitsHi mixedMask
  let (singletonLo, singletonHi) :=
    NN.MLTheory.CROWN.Graph.ibpHardMaskedSoftmaxLastTensor logitsLo logitsHi singletonMask
  let checkTensor (label : String) (actual expected : Tensor Float [3]) : IO Unit :=
    for i in List.finRange 3 do
      assertApprox s!"{label}[{i.val}]" (vecVal actual i) (vecVal expected i) 0.0
  let mixedExpectedLo : Tensor Float [3] := [0.0, 0.0, 0.0]
  let mixedExpectedHi : Tensor Float [3] := [1.0, 0.0, 1.0]
  let singletonExpected : Tensor Float [3] := [0.0, 1.0, 0.0]
  checkTensor "mixed hard-mask lower" mixedLo mixedExpectedLo
  checkTensor "mixed hard-mask upper" mixedHi mixedExpectedHi
  checkTensor "singleton hard-mask lower" singletonLo singletonExpected
  checkTensor "singleton hard-mask upper" singletonHi singletonExpected

/-- Higher-rank softmax values are bounded row-wise, but its derivative pass is vector-only. -/
def checkSoftmaxDerivativeShapeGuard : IO Unit := do
  let matrixShape : Shape := [2, 2]
  let graph : NN.IR.Graph :=
    { nodes := #[
        { id := 0, parents := #[], kind := .input, outShape := matrixShape },
        { id := 1, parents := #[0], kind := .softmax 1, outShape := matrixShape }
      ] }
  let flat : Tensor Float [4] := [0.0, 0.0, 0.0, 0.0]
  let inputBox : NN.MLTheory.CROWN.FlatBox Float := { dim := 4, lo := flat, hi := flat }
  let params : NN.MLTheory.CROWN.Graph.ParamStore Float :=
    { inputBoxes := Std.HashMap.emptyWithCapacity.insert 0 inputBox }
  let values := NN.MLTheory.CROWN.Graph.runIBP (g := graph) (ps := params)
  unless values[1]!.isSome do
    throw <| IO.userError "matrix softmax value IBP unexpectedly failed"
  let first := NN.MLTheory.CROWN.Graph.runScalarDerivative graph params values
  let directional :=
    NN.MLTheory.CROWN.Graph.runDirectionalDerivative graph params values inputBox
  let second := NN.MLTheory.CROWN.Graph.runScalarSecondDerivative graph params values first
  unless first[1]!.isNone && directional[1]!.isNone && second[1]!.isNone do
    throw <| IO.userError "matrix softmax used the vector-only derivative transfer rule"

/-- Finite-precision graph checks fail closed when no directed nonlinear enclosure is available. -/
def checkNonlinearBoundCapabilities : IO Unit := do
  let shape : Shape := [2]
  let inputBox : NN.MLTheory.CROWN.FlatBox Float :=
    { dim := 2, lo := [1.0, 4.0], hi := [2.0, 9.0] }
  let params : NN.MLTheory.CROWN.Graph.ParamStore Float :=
    { inputBoxes := Std.HashMap.emptyWithCapacity.insert 0 inputBox }
  let unaryGraph (kind : NN.IR.OpKind) : NN.IR.Graph :=
    { nodes := #[
        { id := 0, parents := #[], kind := .input, outShape := shape },
        { id := 1, parents := #[0], kind := kind, outShape := shape }
      ] }

  let expBoxes := NN.MLTheory.CROWN.Graph.runIBP (unaryGraph .exp) params
  unless expBoxes[1]!.isNone do
    throw <| IO.userError "Float exp IBP accepted an uncertified host transcendental"

  let sqrtBoxes := NN.MLTheory.CROWN.Graph.runIBP (unaryGraph .sqrt) params
  unless sqrtBoxes[1]!.isSome do
    throw <| IO.userError "Float sqrt IBP rejected its outward-widened enclosure"

  let softmaxGraph := unaryGraph (.softmax 0)
  let softmaxBoxes := NN.MLTheory.CROWN.Graph.runIBP softmaxGraph params
  unless softmaxBoxes[1]!.isSome do
    throw <| IO.userError "Float softmax IBP failed to return its codomain enclosure"
  let softmaxDeriv := NN.MLTheory.CROWN.Graph.runScalarDerivative softmaxGraph params softmaxBoxes
  unless softmaxDeriv[1]!.isNone do
    throw <| IO.userError "Float softmax derivative used exact-scalar coupled arithmetic"

  let reluGraph := unaryGraph .relu
  let reluBoxes := NN.MLTheory.CROWN.Graph.runIBP reluGraph params
  let affine := NN.MLTheory.CROWN.Graph.runAffine reluGraph params
    { inputId := 0, inputDim := 2 } reluBoxes
  match affine[1]! with
  | none => throw <| IO.userError "ReLU affine fallback was not produced"
  | some upper =>
    if hIn : upper.inDim = 2 then
      if hOut : upper.outDim = 2 then
        let affIn := NN.MLTheory.CROWN.Graph.castAffineIn (α := Float) hIn upper.aff
        let aff22 := NN.MLTheory.CROWN.Graph.castAffineOut (α := Float) hOut affIn
        assertApprox "ReLU constant affine coefficient"
          (matVal aff22.A ⟨0, by decide⟩ ⟨0, by decide⟩) 0.0 0.0
        let endpoint := vecVal aff22.c ⟨0, by decide⟩
        unless 2.0 ≤ endpoint && endpoint < 2.000001 do
          throw <| IO.userError "ReLU affine endpoint lost its directed enclosure"
      else
        throw <| IO.userError s!"ReLU affine output dimension was {upper.outDim}, expected 2"
    else
      throw <| IO.userError s!"ReLU affine input dimension was {upper.inDim}, expected 2"

/-- Rounded CROWN keeps an affine coefficient through a linear node using outward arithmetic. -/
def checkDirectedBackwardLinear : IO Unit := do
  let shape : Shape := [1]
  let graph : NN.IR.Graph :=
    { nodes := #[
        { id := 0, parents := #[], kind := .input, outShape := shape },
        { id := 1, parents := #[0], kind := .linear, outShape := shape }
      ] }
  let inputBox : NN.MLTheory.CROWN.FlatBox Float :=
    { dim := 1, lo := [-1.0], hi := [1.0] }
  let params : NN.MLTheory.CROWN.Graph.ParamStore Float :=
    { inputBoxes := Std.HashMap.emptyWithCapacity.insert 0 inputBox
      linearWB := Std.HashMap.emptyWithCapacity.insert 1
        { m := 1, n := 1, w := [[2.0]], b := [0.0] } }
  let ibp := NN.MLTheory.CROWN.Graph.runIBP graph params
  let objective : NN.MLTheory.CROWN.Graph.FlatTensor Float := { n := 1, v := [1.0] }
  let ctx : NN.MLTheory.CROWN.Graph.AffineCtx := { inputId := 0, inputDim := 1 }
  let some bounds :=
      NN.MLTheory.CROWN.Graph.runCROWNBackwardObjective graph params ctx ibp 1 objective
    | throw <| IO.userError "directed backward CROWN failed on a linear graph"
  let lowerCoeff := getAtOrZero bounds.loAff.A [0, 0]
  let upperCoeff := getAtOrZero bounds.hiAff.A [0, 0]
  if lowerCoeff == 0.0 || upperCoeff == 0.0 then
    throw <| IO.userError "directed backward CROWN discarded the linear input coefficient"
  let objectiveBox ←
    match NN.MLTheory.CROWN.Graph.evalBackwardObjectiveBox? bounds inputBox 1 with
    | .ok box => pure box
    | .error e => throw <| IO.userError e
  let some outputBox := ibp[1]!
    | throw <| IO.userError "linear IBP did not produce an output box"
  unless getAtOrZero objectiveBox.lo [0] ≤ getAtOrZero outputBox.lo [0] do
    throw <| IO.userError "directed backward lower bound does not enclose linear IBP"
  unless getAtOrZero outputBox.hi [0] ≤ getAtOrZero objectiveBox.hi [0] do
    throw <| IO.userError "directed backward upper bound does not enclose linear IBP"

/--
Check that batched attention lowers to the same operation as the mathematical attention spec.

The token and head dimensions are deliberately different. This catches an invalid direct reshape
from `(tokens, heads * headDim)` to `(heads, tokens, headDim)`, which can otherwise look plausible
in shape-only tests.
-/
def checkBatchedAttentionLowering : IO Unit := do
  let batch : Nat := 2
  let n : Nat := 3
  let numHeads : Nat := 2
  let headDim : Nat := 2
  let dModel : Nat := numHeads * headDim
  have hBatch : batch ≠ 0 := by decide
  have hSeq : n ≠ 0 := by decide

  let weightShape : Shape := [dModel, dModel]
  let inputShape : Shape := [batch, n, dModel]
  let sampleShape : Shape := [n, dModel]

  let matrix (scale offset : Float) : Tensor Float weightShape :=
    Tensor.generate [dModel, dModel] fun coordinate =>
      offset + scale * Float.ofNat
        (1 + coordinate.getD 0 0 * dModel + coordinate.getD 1 0)
  let sample (offset : Float) : Tensor Float sampleShape :=
    Tensor.generate [n, dModel] fun coordinate =>
      offset + 0.05 * Float.ofNat
        (1 + coordinate.getD 0 0 * dModel + coordinate.getD 1 0)

  let wq := matrix 0.013 (-0.09)
  let wk := matrix (-0.017) 0.11
  let wv := matrix 0.019 (-0.04)
  let wo := matrix 0.023 0.02
  let x : Tensor Float inputShape :=
    TorchLean.Tensor.stack 0 fun b => sample (0.3 * Float.ofNat b.val)

  let paramShapes : List Shape := [weightShape, weightShape, weightShape, weightShape]
  let params : TorchLean.TensorPack Float paramShapes :=
    .cons wq (.cons wk (.cons wv (.cons wo .nil)))
  let mha : Spec.MultiHeadAttention Float numHeads dModel headDim :=
    { queryWeight := wq, keyWeight := wk, valueWeight := wv, outputWeight := wo }
  let checkMask
      (label : String)
      (mask : Option (Tensor Bool [n, n]) := none) : IO Unit := do
    let prog :
        Runtime.Autograd.Model.Program Float (paramShapes ++ [inputShape]) inputShape :=
      fun {m} _ _ wqR wkR wvR woR xR =>
        Runtime.Autograd.Torch.batchedMultiHeadAttention
          (m := m) (α := Float) (batch := batch) (n := n)
          (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
          hBatch hSeq wqR wkR wvR woR xR mask

    let lowered ←
      match NN.Verification.Builtin.lowerForwardToIR
          (α := Float) (paramShapes := paramShapes) (inShape := inputShape)
          (outShape := inputShape) prog params with
      | .error e =>
          throw <| IO.userError s!"{label} attention lowering failed: {e}"
      | .ok c => pure c
    let payload : NN.IR.Payload Float :=
      NN.Verification.Builtin.payloadOfParamStore (α := Float) lowered.ps
    if mask.isSome then
      let hasHardMask := lowered.graph.nodes.any fun node =>
        match node.kind with
        | .hardMaskedSoftmax _ => true
        | _ => false
      unless hasHardMask do
        throw <| IO.userError
          s!"{label} attention lowering did not emit a hard-masked-softmax node"
    let yIR : Tensor Float inputShape ←
      match NN.IR.Graph.denote (α := Float) (g := lowered.graph) (payload := payload)
          (input := Spec.SomeTensor.mk (α := Float) inputShape x)
          (outputId := lowered.outputId) with
      | .error e => throw <| IO.userError s!"{label} attention lowering denote failed: {e}"
      | .ok out =>
          match NN.IR.Graph.expectShape (α := Float) (expected := inputShape) out with
          | .ok y => pure y
          | .error e => throw <| IO.userError s!"{label} attention lowering output mismatch: {e}"

    let ySpec : Tensor Float inputShape :=
      TorchLean.Tensor.stack 0 fun b =>
        Spec.MultiHeadAttention.forward (α := Float) (n := n) (h1 := hSeq)
          (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
          mha (sample (0.3 * Float.ofNat b.val)) mask

    let yIRFlat := Tensor.flattenSpec yIR
    let ySpecFlat := Tensor.flattenSpec ySpec
    for i in List.finRange (Spec.Shape.size inputShape) do
      assertApprox s!"{label} attention lowering[{i.val}]"
        (vecVal yIRFlat i) (vecVal ySpecFlat i) 2e-5

    -- An exact input box must remain evaluable by IBP.  In particular, a fully blocked mask may
    -- not leave an inverse whose interval contains zero.
    let inputBox := NN.Verification.Builtin.lInfBall (α := Float) x 0.0
    let boxes := lowered.runIBP (lowered.seedInputBox inputBox)
    let _ ← lowered.outputBoxOrThrow boxes

  checkMask "unmasked"
  checkMask "fully blocked" (some (Spec.allFalseMask n n))

def run : IO Unit := do
  IO.println "torchlean_ir_exec_equiv_check: begin"
  checkForwardContextValues
  checkRawLogExecutionBoundary
  checkHardMaskedSoftmaxIbpBoundary
  checkSoftmaxDerivativeShapeGuard
  checkNonlinearBoundCapabilities
  checkDirectedBackwardLinear

  let inputWidth : Nat := 2
  let hiddenWidth : Nat := 3
  let outputWidth : Nat := 1
  let xShape : Shape := [inputWidth]
  let yShape : Shape := [outputWidth]

  -- A small deterministic TorchLean MLP built through the public model API.
  let model := nn.build 0 <|
    nn.mlp inputWidth outputWidth { hiddenWidths := [hiddenWidth] }

  let paramShapes := Runtime.Autograd.Model.Layers.Seq.stateShapes model
  let params : TorchLean.TensorPack Float paramShapes :=
    Runtime.Autograd.Model.Layers.Seq.initState (m := model)

  -- One input vector.
  let x : Tensor Float xShape :=
    Tensor.ofFn fun i => [0.5, 0.8][i.val]!

  -- TorchLean forward computation for the model.
  let prog :
      Runtime.Autograd.Model.Program Float (paramShapes ++ [xShape]) yShape :=
    Runtime.Autograd.Model.Layers.Seq.forward model (α := Float)

  -- Lower to IR and executable typed graph data.
  let (c, exec) ←
    match NN.Verification.Builtin.lowerForwardExecutable
        (α := Float) (paramShapes := paramShapes) (inShape := xShape) (outShape := yShape) prog
          params with
    | .error e => throw <| IO.userError s!"torchlean_ir_exec_equiv_check: lowering failed: {e}"
    | .ok r => pure r

  let payload : NN.IR.Payload Float :=
    NN.Verification.Builtin.payloadOfParamStore (α := Float) c.ps

  -- Cast the test input into the executable graph's expected input shape.
  let xExec : Tensor Float exec.inShape ←
    if hIn : exec.inShape = xShape then
      pure <| Tensor.castShape x (Eq.symm hIn)
    else
      let msg :=
        s!"torchlean_ir_exec_equiv_check: exec input shape mismatch: got {repr exec.inShape}" ++
          s!", expected {repr xShape}"
      throw <| IO.userError msg

  -- IR denotation at the lowered output node.
  let yIR : Tensor Float yShape ←
    match NN.IR.Graph.denote (α := Float) (g := c.graph) (payload := payload)
        (input := Spec.SomeTensor.mk (α := Float) xShape x) (outputId := c.outputId) with
    | .error e => throw <| IO.userError s!"torchlean_ir_exec_equiv_check: IR denote failed: {e}"
    | .ok out =>
        match NN.IR.Graph.expectShape (α := Float) (expected := yShape) out with
        | .ok t => pure t
        | .error e =>
            throw <| IO.userError s!"torchlean_ir_exec_equiv_check: IR output shape mismatch: {e}"

  -- Executable `GraphData` evaluation, then read the IR output id from the full value table.
  let execVals := Runtime.Autograd.IRExec.ForwardGraph.denoteAll (α := Float) exec xExec
  let yExec : Tensor Float yShape ←
    match execVals[c.outputId]? with
    | none =>
        let msg :=
          "torchlean_ir_exec_equiv_check: exec outputId out of bounds: " ++
            s!"{c.outputId}"
        throw <| IO.userError msg
    | some out =>
        match NN.IR.Graph.expectShape (α := Float) (expected := yShape) out with
        | .ok t => pure t
        | .error e =>
            throw <| IO.userError s!"torchlean_ir_exec_equiv_check: exec output shape mismatch: {e}"

  for i in List.finRange outputWidth do
    assertApprox s!"ir/exec forward[{i.val}]" (vecVal yIR i) (vecVal yExec i) 1e-6

  checkBatchedAttentionLowering

  IO.println "torchlean_ir_exec_equiv_check: ok"

end TorchLeanIRExecEquivCheck
end Floats
end Tests

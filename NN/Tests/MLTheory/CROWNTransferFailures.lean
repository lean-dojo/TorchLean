/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tests.MLTheory.Utils
public import NN.Verification.Cert.IBPCert

/-!
# Interval transfer regressions

These checks exercise actual convolution nodes, failed parent transfers, malformed matrix payloads,
and concatenation across independent graph inputs.
-/

public section

namespace NN.Tests.MLTheory.CROWNTransferFailures

open Spec TorchLean
open NN.IR NN.MLTheory.CROWN NN.MLTheory.CROWN.Graph

private def require (condition : Bool) (message : String) : IO Unit :=
  unless condition do throw <| IO.userError s!"CROWN transfer: {message}"

/-- An exactly representable dyadic sum must remain inside convolution's directed bounds. -/
def checkConvolution (α : Type) [Storage α] [Context α] [BoundOps α]
    [NonlinearBoundOps α] (small : α) (label : String) : IO Unit := do
  let kernel : Tensor Nat [1] := [129]
  let stride : Tensor Nat [1] := [1]
  let input : Tensor α [1, 129] :=
    Tensor.dim fun _ => Tensor.ofFn fun i => if i.val = 0 then 1 else small
  let parameters : ConvParams α :=
    { spatialRank := 1
      inChannels := 1
      outChannels := 1
      kernel
      stride
      padding := [0]
      dilation := [1]
      paddingAfter := [0]
      groups := 1
      inputSpatial := [129]
      kernelNonzero := by intro i; fin_cases i; simp [kernel]
      strideNonzero := by intro i; fin_cases i; simp [stride]
      spec :=
        { kernel := by simpa [kernel] using (Tensor.full (α := α) [1, 1, 129] 1)
          bias := [0] } }
  let config : ConvConfig :=
    { spatialRank := 1, kernel, stride, padding := [0], dilation := [1], paddingAfter := [0]
      groups := 1, channelAxis := 0, inChannels := 1, outChannels := 1 }
  let graph : NN.IR.Graph := { nodes := #[
    { id := 0, kind := .input, parents := #[], outShape := [1, 129] },
    { id := 1, kind := .conv config, parents := #[0], outShape := [1, 1] }] }
  let ps : ParamStore α :=
    { inputBoxes := Std.HashMap.emptyWithCapacity.insert 0
        (FlatBox.ofTensor (Tensor.flattenSpec input))
      convCfg := Std.HashMap.emptyWithCapacity.insert 1 parameters }
  let some result := (runIBP graph ps)[1]?.join
    | throw <| IO.userError s!"CROWN transfer: {label} convolution has no bound"
  let exact := (1 : α) + 128 * small
  require (!(decide (getAtOrZero result.lo [0] > exact)) &&
      !(decide (exact > getAtOrZero result.hi [0])))
    s!"{label} convolution lost small terms"

/-- A rejected LayerNorm transfer stays unresolved through every dependent branch. -/
def checkFailedParents : IO Unit := do
  let shape : Shape := [2]
  let input : Tensor Float [2] := [-1, 1]
  let parameters : LayerNormParams Float :=
    { normalizedShape := shape, gamma := [0, 0], beta := [3, 3], eps := 0 }
  let initialNodes : Array NN.IR.Node := #[
    { id := 0, kind := .input, parents := #[], outShape := shape },
    { id := 1, kind := .layernorm 0, parents := #[0], outShape := shape }]
  let ps : ParamStore Float :=
    { inputBoxes := Std.HashMap.emptyWithCapacity.insert 0 (Utils.pointFlatBox input)
      layerNorm := Std.HashMap.emptyWithCapacity.insert 1 parameters }
  let consumers : Array (NN.IR.OpKind × Array Nat × Shape) := #[
    (.sum, #[1], []), (.relu, #[1], shape), (.abs, #[1], shape),
    (.tanh, #[1], shape), (.sigmoid, #[1], shape), (.exp, #[1], shape),
    (.sin, #[1], shape), (.cos, #[1], shape), (.sqrt, #[1], shape),
    (.inv, #[1], shape), (.log, #[1], shape), (.softplus, #[1], shape),
    (.detach, #[1], shape), (.add, #[1, 0], shape), (.sub, #[0, 1], shape),
    (.mulElem, #[1, 0], shape), (.safeLog, #[0, 1], shape),
    (.concat 0, #[0, 1], [4])]
  for (kind, parents, outShape) in consumers do
    let graph : NN.IR.Graph := { nodes := initialNodes.push { id := 2, kind, parents, outShape } }
    let boxes := runIBP graph ps
    require ((boxes[1]?.join).isNone && (boxes[2]?.join).isNone)
      "failed parent produced a descendant bound"
  let graph : NN.IR.Graph :=
    { nodes := initialNodes.push { id := 2, kind := .sum, parents := #[1], outShape := [] } }
  let payload : Payload Float :=
    { layerNorm? := fun id => if id = 1 then some parameters else none }
  let .ok value := NN.IR.Graph.denote graph payload (SomeTensor.ofTensor input) 2
    | throw <| IO.userError "CROWN transfer: epsilon-zero reference is undefined"
  require (getAtOrZero value.tensor [] == 6) "epsilon-zero reference value changed"
  let rejected ← try
    let _ ← NN.Verification.IBPCert.check graph ps 2 "/unused-failed-transfer.json"
    pure false
  catch error => pure (error.toString.contains "no output box")
  require rejected "failed transfer reached certificate comparison"
  for kind in [NN.IR.OpKind.tanh, .sigmoid, .sin, .cos] do
    for parents in [#[], #[0, 0], #[99]] do
      let malformed : NN.IR.Node := { id := 2, kind, parents, outShape := shape }
      require
        ((ibpStepNodeAt? initialNodes ps #[some (Utils.pointFlatBox input)] 2 malformed).isNone)
        "malformed nonlinear parents produced a bound"

/-- A reduction must not conceal a stored matrix's inconsistent output dimension. -/
def checkMatmulDimensions : IO Unit := do
  let graph : NN.IR.Graph := { nodes := #[
    { id := 0, kind := .input, parents := #[], outShape := [1] },
    { id := 1, kind := .matmul, parents := #[0], outShape := [1] },
    { id := 2, kind := .sum, parents := #[1], outShape := [] }] }
  let ps : ParamStore Float :=
    { inputBoxes := Std.HashMap.emptyWithCapacity.insert 0
        (Utils.pointFlatBox ([1] : Tensor Float [1]))
      matmulW := Std.HashMap.emptyWithCapacity.insert 1 { m := 2, n := 1, w := [[1], [2]] } }
  let malformedStores := [
    ps,
    { ps with matmulW :=
        Std.HashMap.emptyWithCapacity.insert 1 { m := 1, n := 2, w := [[1, 2]] } },
    { ps with matmulW := Std.HashMap.emptyWithCapacity }]
  let ctx : AffineCtx := { inputId := 0, inputDim := 1 }
  let input := Utils.pointFlatBox ([1] : Tensor Float [1])
  for store in malformedStores do
    require (!crownGraphSemanticsSupported graph store) "malformed unary matrix payload accepted"
    let boxes := runIBP graph store
    require (boxes.all Option.isNone) "malformed unary matrix produced bounds"
    require ((runCROWN graph store ctx boxes).all Option.isNone)
      "malformed unary matrix produced nodewise CROWN bounds"
    require ((runAffine graph store ctx boxes).all Option.isNone)
      "malformed unary matrix produced an affine projection"
    require ((runCROWNBackwardObjective graph store ctx boxes 2 { n := 1, v := [1] }).isNone)
      "malformed unary matrix produced a backward objective"
    require (!(outputBoxCROWN? graph store input 0 2 1).isOk)
      "malformed unary matrix produced output bounds"
  let missingParent := { graph with
    nodes := graph.nodes.set! 1 { id := 1, kind := .matmul, parents := #[99], outShape := [2] } }
  require (!crownGraphSemanticsSupported missingParent ps) "missing unary matrix parent accepted"
  let valid := { graph with
    nodes := graph.nodes.set! 1 { id := 1, kind := .matmul, parents := #[0], outShape := [2] } }
  require (crownGraphSemanticsSupported valid ps) "valid unary matrix payload rejected"
  let some result := (runIBP valid ps)[2]?.join
    | throw <| IO.userError "CROWN transfer: valid unary matrix has no bound"
  require (getAtOrZero result.lo [0] ≤ 3 && 3 ≤ getAtOrZero result.hi [0])
    "valid unary matrix lost its sum"

/-- Every graph input contributes its interval when affine bounds select one input variable. -/
def checkIndependentConcat : IO Unit := do
  let graph : NN.IR.Graph := { nodes := #[
    { id := 0, kind := .input, parents := #[], outShape := [1] },
    { id := 1, kind := .input, parents := #[], outShape := [1] },
    { id := 2, kind := .concat 0, parents := #[0, 1, 1], outShape := [3] }] }
  let first : FlatBox Float := { dim := 1, lo := [1], hi := [2] }
  let second : FlatBox Float := { dim := 1, lo := [3], hi := [4] }
  let ps := ({} : ParamStore Float).seedInputBox 0 first |>.seedInputBox 1 second
  let boxes := runIBP graph ps
  let ctx : AffineCtx := { inputId := 0, inputDim := 1 }
  let crown := runCROWN graph ps ctx boxes
  let some bound := crown[2]?.join
    | throw <| IO.userError "CROWN transfer: independent concat has no CROWN bound"
  require (bound.inDim == 1 && bound.outDim == 3) "independent concat dimensions changed"
  require (((runAffine graph ps ctx boxes)[2]?.join).isSome)
    "independent concat has no upper affine bound"
  let .ok output := evalCROWNOutputBox? crown first 2 1
    | throw <| IO.userError "CROWN transfer: independent concat cannot be evaluated"
  for x in [1, 2] do
    for y in [3, 4] do
      for (index, expected) in [(0, x), (1, y), (2, y)] do
        require (getAtOrZero output.lo [index] ≤ expected &&
          expected ≤ getAtOrZero output.hi [index]) "independent concat lost an endpoint"

/-- A grouped convolution carries both terms of an upstream square's directional derivatives. -/
def checkConvolutionDerivatives : IO Unit := do
  let kernel : Tensor Nat [1] := [3]
  let stride : Tensor Nat [1] := [1]
  let parameters : ConvParams Float :=
    { spatialRank := 1, inChannels := 4, outChannels := 4, kernel, stride
      padding := [1], dilation := [2], paddingAfter := [2], groups := 2, inputSpatial := [8]
      kernelNonzero := by intro i; fin_cases i; simp [kernel]
      strideNonzero := by intro i; fin_cases i; simp [stride]
      spec :=
        { kernel := by simpa [kernel] using (Tensor.full (α := Float) [4, 4, 3] 1)
          bias := [7, 7, 7, 7] } }
  let configuration : ConvConfig :=
    { spatialRank := 1, inChannels := 4, outChannels := 4, kernel, stride
      padding := [1], dilation := [2], paddingAfter := [2], groups := 2, channelAxis := 1 }
  let graph : NN.IR.Graph := { nodes := #[
    { id := 0, kind := .input, parents := #[], outShape := [2, 4, 8] },
    { id := 1, kind := .mulElem, parents := #[0, 0], outShape := [2, 4, 8] },
    { id := 2, kind := .conv configuration, parents := #[1], outShape := [2, 4, 7] }] }
  let point (value : Float) : FlatBox Float :=
    Utils.pointFlatBox (Tensor.full (α := Float) [2, 4, 8] value)
  let ps : ParamStore Float :=
    { inputBoxes := Std.HashMap.emptyWithCapacity.insert 0 (point 1)
      convCfg := Std.HashMap.emptyWithCapacity.insert 2 parameters }
  let values := runIBP graph ps
  let left := runDirectionalDerivative graph ps values (point 2)
  let right := runDirectionalDerivative graph ps values (point 3)
  let second := runMixedSecondDerivative graph ps values left right
  let some firstBox := left[2]?.join
    | throw <| IO.userError "CROWN transfer: grouped convolution has no first derivative"
  let some secondBox := second[2]?.join
    | throw <| IO.userError "CROWN transfer: grouped convolution has no mixed derivative"
  require (firstBox.dim == 56 && secondBox.dim == 56) "convolution derivative shape changed"
  for index in [0:56] do
    let outputPosition := index % 7
    let validTaps := ([0, 1, 2] : List Nat).filter fun k =>
      1 ≤ outputPosition + 2 * k && outputPosition + 2 * k < 9
    let count : Float := 2 * validTaps.length.toFloat
    let firstExpected := 4 * count
    let secondExpected := 12 * count
    require (getAtOrZero firstBox.lo [index] ≤ firstExpected &&
        firstExpected ≤ getAtOrZero firstBox.hi [index])
      "convolution first derivative lost its direction or retained the bias"
    require (getAtOrZero secondBox.lo [index] ≤ secondExpected &&
        secondExpected ≤ getAtOrZero secondBox.hi [index])
      "convolution mixed derivative lost its upstream bilinear term"

def run : IO Unit := do
  checkConvolution Float (1 / 18014398509481984) "Float"
  checkConvolution Float32 (1 / 67108864) "Float32"
  checkFailedParents
  checkMatmulDimensions
  checkIndependentConcat
  checkConvolutionDerivatives
  IO.println "CROWN transfer: rounding, failure, shape and independent-input checks passed"

end NN.Tests.MLTheory.CROWNTransferFailures

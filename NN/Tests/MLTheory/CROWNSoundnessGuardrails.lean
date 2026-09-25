/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Models.Mlp
public import NN.MLTheory.CROWN.Proofs.DirectedBackwardEvaluation
public import NN.Tests.MLTheory.DirectedConvSemanticsRegression
public import NN.Tests.MLTheory.DirectedIBPNormalization
public import NN.Tests.MLTheory.DirectedIBPPointwise
public import NN.MLTheory.CROWN.Extras.FP32
public import NN.Tests.MLTheory.Utils
public import NN.Tests.Utils

/-!
# CROWN Soundness Guardrails

Check the boundary between valid IR configurations and malformed payloads. Invalid convolution
geometry and mismatched LayerNorm payload shapes must return no bounds. Valid convolution and
affine LayerNorm retain their value and derivative bounds. A rounded backend must reject imported
ReLU slopes it cannot use, and the rounded IBP soundness theorem must apply to `FP32`.
-/

@[expose] public section

namespace NN.Tests.MLTheory.CROWNSoundnessGuardrails

open Spec TorchLean
open NN.IR
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph

open Tests.Utils (assertBoundAt assertNoBoundAt)
open NN.Tests.MLTheory.Utils (pointFlatBox)

def checkConvolutionGuards : IO Unit := do
  let inputSpatial : Tensor Nat [1] := [4]
  let kernel : Tensor Nat [1] := [2]
  let stride : Tensor Nat [1] := [1]
  let padding : Tensor Nat [1] := [1]
  let dilationOne : Tensor Nat [1] := [1]
  let dilationZero : Tensor Nat [1] := [0]
  let paddingAfterZero : Tensor Nat [1] := [0]
  let weights : Tensor Float [2, 2, 2] := Tensor.full (α := Float) [2, 2, 2] 1
  let bias : Tensor Float [2] := Tensor.full (α := Float) [2] 0
  have hKernel : ∀ i : Fin 1, kernel.getScalar i ≠ 0 := by
    intro i
    fin_cases i
    simp [kernel]
  have hStride : ∀ i : Fin 1, stride.getScalar i ≠ 0 := by
    intro i
    fin_cases i
    simp [stride]
  let base : ConvParams Float :=
    { spatialRank := 1
      inChannels := 2
      outChannels := 2
      kernel := kernel
      stride := stride
      padding := padding
      dilation := dilationOne
      paddingAfter := padding
      groups := 1
      inputSpatial := inputSpatial
      kernelNonzero := hKernel
      strideNonzero := hStride
      spec := { kernel := by simpa [kernel] using weights, bias := bias } }
  let baseConfig : ConvConfig :=
    { spatialRank := 1
      kernel := kernel
      stride := stride
      padding := padding
      dilation := dilationOne
      paddingAfter := padding
      groups := 1
      channelAxis := 0
      inChannels := 2
      outChannels := 2 }
  let input : Tensor Float [2, 4] := Tensor.full (α := Float) [2, 4] 1
  let inputBox := pointFlatBox input
  let graphFor (config : ConvConfig) (outShape : Shape) : NN.IR.Graph :=
    { nodes :=
        #[ { id := 0, parents := #[], kind := .input, outShape := [2, 4] }
         , { id := 1, parents := #[0], kind := .conv config, outShape := outShape } ] }
  let storeFor (params : ConvParams Float) : ParamStore Float :=
    { inputBoxes := Std.HashMap.emptyWithCapacity.insert 0 inputBox
      convCfg := Std.HashMap.emptyWithCapacity.insert 1 params }
  let supportedGraph := graphFor baseConfig [2, 5]
  let supportedStore := storeFor base
  unless crownGraphSemanticsSupported supportedGraph supportedStore do
    throw <| IO.userError "supported dense convolution was rejected"
  assertBoundAt "supported dense convolution"
    (runIBP supportedGraph supportedStore) 1

  let groupedParams := { base with groups := 0 }
  let groupedConfig := { baseConfig with groups := 0 }
  let groupedGraph := graphFor groupedConfig [2, 5]
  let groupedStore := storeFor groupedParams
  unless !crownGraphSemanticsSupported groupedGraph groupedStore do
    throw <| IO.userError "convolution with zero groups was accepted"
  assertNoBoundAt "convolution with zero groups" (runIBP groupedGraph groupedStore) 1

  let dilatedParams := { base with dilation := dilationZero }
  let dilatedConfig := { baseConfig with dilation := dilationZero }
  let dilatedGraph := graphFor dilatedConfig [2, 6]
  let dilatedStore := storeFor dilatedParams
  unless !crownGraphSemanticsSupported dilatedGraph dilatedStore do
    throw <| IO.userError "convolution with zero dilation was accepted"
  assertNoBoundAt "convolution with zero dilation" (runIBP dilatedGraph dilatedStore) 1

  let asymmetricParams := { base with paddingAfter := paddingAfterZero }
  let asymmetricConfig := { baseConfig with paddingAfter := paddingAfterZero }
  let asymmetricGraph := graphFor asymmetricConfig [2, 5]
  let asymmetricStore := storeFor asymmetricParams
  unless !crownGraphSemanticsSupported asymmetricGraph asymmetricStore do
    throw <| IO.userError "asymmetric convolution with a mismatched output shape was accepted"
  let ibp := runIBP asymmetricGraph asymmetricStore
  assertNoBoundAt "asymmetric convolution output shape" ibp 1
  let ctx : AffineCtx := { inputId := 0, inputDim := 8 }
  assertNoBoundAt "asymmetric convolution output shape CROWN"
    (runCROWN asymmetricGraph asymmetricStore ctx ibp) 1

/--
Check affine LayerNorm values and derivatives while rejecting inconsistent payload shapes.

For the constant input, each normalized row is zero and the bias makes every output three.
The backward objective sums all four entries, so its bounds must contain twelve. A payload with
the same number of entries but a different shape must still be rejected, even if a caller
supplies a precomputed output box.
-/
def checkLayerNormPayloadGuard : IO Unit := do
  let input : Tensor Float [2, 2] := Tensor.full (α := Float) [2, 2] 1
  let params : LayerNormParams Float :=
    { normalizedShape := [2]
      gamma := Tensor.full (α := Float) [2] 2
      beta := Tensor.full (α := Float) [2] 3
      eps := 0.25 }
  let graph : NN.IR.Graph :=
    { nodes :=
        #[ { id := 0, parents := #[], kind := .input, outShape := [2, 2] }
         , { id := 1, parents := #[0], kind := .layernorm 1, outShape := [2, 2] } ] }
  let store : ParamStore Float :=
    { inputBoxes := Std.HashMap.emptyWithCapacity.insert 0 (pointFlatBox input)
      layerNorm := Std.HashMap.emptyWithCapacity.insert 1 params }
  unless crownGraphSemanticsSupported graph store do
    throw <| IO.userError "last-axis LayerNorm payload was rejected"
  let ibp := runIBP graph store
  assertBoundAt "affine LayerNorm IBP" ibp 1
  let ctx : AffineCtx := { inputId := 0, inputDim := 4 }
  assertBoundAt "affine LayerNorm affine bounds" (runAffine graph store ctx ibp) 1
  assertBoundAt "affine LayerNorm CROWN" (runCROWN graph store ctx ibp) 1
  let objective : FlatTensor Float := { n := 4, v := Tensor.full (α := Float) [4] 1 }
  let .ok objectiveBox := backwardObjectiveBox? graph store ctx ibp (pointFlatBox input) 1 objective
    | throw <| IO.userError "affine LayerNorm backward objective was rejected"
  let lower := Tensor.to objectiveBox.lo (Array Float)
  let upper := Tensor.to objectiveBox.hi (Array Float)
  unless lower.size == 1 && upper.size == 1 && lower[0]! <= 12 && 12 <= upper[0]! do
    throw <| IO.userError "affine LayerNorm objective bounds omitted the bias"
  let derivatives := runDirectionalDerivative graph store ibp (pointFlatBox input)
  assertBoundAt "LayerNorm input derivative seed" derivatives 0
  assertBoundAt "LayerNorm payload first derivative" derivatives 1
  assertBoundAt "LayerNorm payload second derivative"
    (runMixedSecondDerivative graph store ibp derivatives derivatives) 1

  let wrongShape : LayerNormParams Float :=
    { normalizedShape := [1, 2]
      gamma := Tensor.full (α := Float) [1, 2] 2
      beta := Tensor.full (α := Float) [1, 2] 3
      eps := params.eps }
  let invalidStore := { store with layerNorm := store.layerNorm.insert 1 wrongShape }
  unless !crownGraphSemanticsSupported graph invalidStore do
    throw <| IO.userError "mismatched LayerNorm payload shape was accepted"
  let invalidIbp := runIBP graph invalidStore
  assertNoBoundAt "mismatched LayerNorm IBP" invalidIbp 1
  assertNoBoundAt "mismatched LayerNorm affine"
    (runAffine graph invalidStore ctx invalidIbp) 1
  assertNoBoundAt "mismatched LayerNorm CROWN"
    (runCROWN graph invalidStore ctx invalidIbp) 1
  let forgedIbp :=
    #[some (pointFlatBox input), some (FlatBox.ofTensor (Tensor.full (α := Float) [4] 0))]
  match runCROWNBackwardObjective graph invalidStore ctx forgedIbp 1 objective with
  | none => pure ()
  | some _ => throw <| IO.userError "mismatched LayerNorm backward CROWN was accepted"

/-- Coefficient cancellation must not turn a lower logit into a certified winner. -/
def checkAffineCancellation : IO Unit := do
  let graph : NN.IR.Graph := ⟨#[
    { id := 0, parents := #[], kind := .input, outShape := [1] },
    { id := 1, parents := #[0], kind := .linear, outShape := [3] },
    { id := 2, parents := #[1], kind := .linear, outShape := [2] }]⟩
  let w1 : Tensor Float32 [3, 1] := Tensor.full [3, 1] 1
  let b1 : Tensor Float32 [3] := Tensor.full [3] 0
  let w2 : Tensor Float32 [2, 3] := Tensor.matrix fun i j =>
    if i.val = 1 then 0 else
    if j.val = 0 then 16777216 else if j.val = 1 then 1 else -16777216
  let b2 : Tensor Float32 [2] := [0, 0.05]
  let input : Tensor Float32 [1] := [0.1]
  let box : FlatBox Float32 := { dim := 1, lo := input, hi := input }
  let params : ParamStore Float32 :=
    { inputBoxes := ({} : Std.HashMap Nat (FlatBox Float32)).insert 0 box
      linearWB := ({} : Std.HashMap Nat (LinParams Float32)).insert 1
        { m := 3, n := 1, w := w1, b := b1 } |>.insert 2
        { m := 2, n := 3, w := w2, b := b2 } }
  let .ok output := outputBoxCROWN? graph params box 0 2 1
    | throw <| IO.userError "CROWN failed on the cancellation graph"
  -- The stored weights sum to one in real arithmetic; execution rounds its first logit to 0.125.
  unless getAtOrZero output.lo [0] ≤ 0.1 && 0.125 ≤ getAtOrZero output.hi [0] do
    throw <| IO.userError "CROWN lost coefficient-rounding error"
  unless !(getAtOrZero output.lo [1] > getAtOrZero output.hi [0]) do
    throw <| IO.userError "CROWN certified the wrong class after coefficient cancellation"
  let net : TwoLayerMLP Float32 1 3 2 :=
    { hiddenWeight := w1, hiddenBias := b1, outputWeight := w2, outputBias := b2 }
  let mlpBounds := boundAffineCrown net (Box.point input)
  unless mlpBounds.lo.getScalar 0 ≤ 0.1 && 0.125 ≤ mlpBounds.hi.getScalar 0 do
    throw <| IO.userError "standalone MLP CROWN lost coefficient-rounding error"

/-- Negative weights must consume a lower parent bound, even in the upper-only API. -/
def checkUpperAffineSign : IO Unit := do
  let graph : NN.IR.Graph := ⟨#[
    { id := 0, parents := #[], kind := .input, outShape := [1] },
    { id := 1, parents := #[0], kind := .relu, outShape := [1] },
    { id := 2, parents := #[1], kind := .linear, outShape := [1] }]⟩
  let params : ParamStore Float :=
    { inputBoxes := ({} : Std.HashMap Nat (FlatBox Float)).insert 0
        { dim := 1, lo := [-1], hi := [1] }
      linearWB := ({} : Std.HashMap Nat (LinParams Float)).insert 2
        { m := 1, n := 1, w := [[-1]], b := [0] } }
  let affines := runAffine graph params { inputId := 0, inputDim := 1 } (runIBP graph params)
  let some affine := affines[2]!
    | throw <| IO.userError "upper affine bound missing for negative ReLU"
  unless 0 ≤ getAtOrZero affine.aff.c [0] do
    throw <| IO.userError "upper affine bound excludes -ReLU(0) = 0"

/-- A rounded backend has no ReLU relaxation in its directed pass, so it refuses imported slopes. -/
def checkRoundedReluAlphaRejected : IO Unit := do
  let graph : NN.IR.Graph := ⟨#[
    { id := 0, parents := #[], kind := .input, outShape := [1] },
    { id := 1, parents := #[0], kind := .relu, outShape := [1] },
    { id := 2, parents := #[1], kind := .linear, outShape := [1] }]⟩
  let params : ParamStore Float :=
    { inputBoxes := ({} : Std.HashMap Nat (FlatBox Float)).insert 0
        { dim := 1, lo := [-1], hi := [1] }
      linearWB := ({} : Std.HashMap Nat (LinParams Float)).insert 2
        { m := 1, n := 1, w := [[1]], b := [0] } }
  let ctx : AffineCtx := { inputId := 0, inputDim := 1 }
  let ibp := runIBP graph params
  let obj : FlatTensor Float := { n := 1, v := [1] }
  let slopes : Array (Option (FlatTensor Float)) := #[none, some { n := 1, v := [0.5] }, none]
  if (runCROWNBackwardObjectiveLowerWithReluAlpha graph params ctx ibp 2 obj slopes).isSome then
    throw <| IO.userError "rounded CROWN accepted ReLU slopes it does not use"
  if (runCROWNBackwardObjectiveLowerWithReluAlpha graph params ctx ibp 2 obj
      #[none, none, none]).isNone then
    throw <| IO.userError "rounded CROWN failed without ReLU slopes"

/-- The original core-family classifier keeps its meaning alongside the unrestricted theorem. -/
def checkIBPForwardSupport : IO Unit := do
  let covered : Array NN.IR.Node := #[
    { id := 0, parents := #[], kind := .input, outShape := [2] },
    { id := 1, parents := #[0], kind := .linear, outShape := [2] },
    { id := 2, parents := #[1], kind := .relu, outShape := [2] },
    { id := 3, parents := #[2], kind := .sum, outShape := [1] }]
  unless DirectedBackward.ibpForwardSupported covered do
    throw <| IO.userError "rounded IBP support check rejected a linear/ReLU/sum graph"
  let binaryMatmul : NN.IR.Node := { id := 2, parents := #[0, 1], kind := .matmul, outShape := [1] }
  if DirectedBackward.ibpForwardSupported (covered.push binaryMatmul) then
    throw <| IO.userError "rounded IBP support check accepted binary matmul"

/-- The original core-family theorem remains available at the `FP32` model. -/
example (g : NN.IR.Graph) (ps : ParamStore FP32) (dims : Nat → Nat) (v : Nat → Nat → ℝ)
    (hparent : ∀ id, id < g.nodes.size → ∀ p ∈ g.nodes[id]!.parents, p < id)
    (hsupported : DirectedBackward.ibpForwardSupported g.nodes = true)
    (hinputs : DirectedBackward.InputsInBoxes g.nodes ps dims v)
    (hequation : ∀ id, id < g.nodes.size →
      DirectedBackward.NodeEquation g.nodes ps (runIBP g ps) dims v id)
    (id : Nat) (hid : id < g.nodes.size) (box : FlatBox FP32)
    (hbox : (runIBP g ps)[id]! = some box) :
    DirectedBackward.RowEncloses box (dims id) (v id) :=
  DirectedBackward.runIBP_encloses g ps hparent hsupported hinputs hequation id hid box hbox

/-- Every operation is covered at FP32 without an intermediate-enclosure hypothesis or a
restriction to the original core family. The fixed epsilon premise is proved for the format. -/
example (g : NN.IR.Graph) (ps : ParamStore FP32) (dims : Nat → Nat) (v : Nat → Nat → ℝ)
    (hparent : ∀ id, id < g.nodes.size → ∀ p ∈ g.nodes[id]!.parents, p < id)
    (hinputs : DirectedBackward.InputsInBoxes g.nodes ps dims v)
    (hequation : ∀ id, id < g.nodes.size →
      DirectedBackward.RealNodeEquation g.nodes ps (runIBP g ps) dims v id)
    (id : Nat) (hid : id < g.nodes.size) (box : FlatBox FP32)
    (hbox : (runIBP g ps)[id]! = some box) :
    DirectedBackward.RowEncloses box (dims id) (v id) :=
  DirectedBackward.runIBP_encloses_all g ps DirectedBackward.normalizationEpsilon_nonneg_fp32
    hparent hinputs hequation id hid box hbox

/-- The complete objective workflow specializes to FP32 with its actual arithmetic instances. -/
example {g : NN.IR.Graph} {ps : ParamStore FP32} {ctx : AffineCtx}
    {dims : Nat → Nat} {v : Nat → Nat → ℝ}
    (input_lt : ctx.inputId < g.nodes.size)
    (input_dim : dims ctx.inputId = ctx.inputDim)
    (input_kind : g.nodes[ctx.inputId]!.kind = .input)
    (node_id : ∀ id, id < g.nodes.size → g.nodes[id]!.id = id)
    (parent_lt : ∀ id, id < g.nodes.size → ∀ p ∈ g.nodes[id]!.parents, p < id)
    (hinputs : DirectedBackward.InputsInBoxes g.nodes ps dims v)
    (equation : ∀ id, id < g.nodes.size →
      DirectedBackward.RealNodeEquation g.nodes ps (runIBP g ps) dims v id)
    (xB : FlatBox FP32) (hx : DirectedBackward.RowEncloses xB ctx.inputDim (v ctx.inputId))
    (output : Nat) (houtput : output < g.nodes.size) (obj : FlatTensor FP32)
    (hdim : obj.n = dims output) {result : FlatBox FP32}
    (hresult : backwardObjectiveBox? g ps ctx (runIBP g ps) xB output obj = .ok result) :
    DirectedBackward.RowEncloses result 1
      (fun _ => DirectedBackward.dot (dims output)
        (fun i => LawfulBoundOps.toReal (getAtOrZero obj.v [i])) (v output)) :=
  DirectedBackward.backwardObjectiveBox_encloses_runIBP_all rfl
    input_lt input_dim input_kind node_id parent_lt
    DirectedBackward.normalizationEpsilon_nonneg_fp32 hinputs equation
    xB hx output houtput obj hdim hresult

def run : IO Unit := do
  checkConvolutionGuards
  checkLayerNormPayloadGuard
  checkAffineCancellation
  checkUpperAffineSign
  checkRoundedReluAlphaRejected
  checkIBPForwardSupport

end NN.Tests.MLTheory.CROWNSoundnessGuardrails

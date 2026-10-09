/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Extras.FP32
public import NN.MLTheory.CROWN.Models.Mlp
public import NN.MLTheory.CROWN.Proofs.DirectedBackwardEvaluation
public import NN.Tests.Utils
public import NN.Tests.Verification.CrownEquations
public import NN.Verification.Cert.IBPCert
public import NN.Verification.Robustness.TopLabel

/-!
# CROWN runtime checks

Exercise executable boundaries the real-arithmetic theorems alone do not check: malformed
payload rejection, directed rounding, failed-parent propagation and classification verdicts.
The imported equation checks also instantiate the soundness assumptions with actual FP32 values.
-/

public section

namespace NN.Tests.Verification.Crown

open Spec TorchLean NN.MLTheory.CROWN

/-- A concrete tensor as a flattened singleton box for the runtime transfers. -/
private def pointFlatBox {α : Type} [Storage α] [Context α] {s : Shape} (value : Tensor α s) :
    FlatBox α :=
  FlatBox.ofTensor (Tensor.flattenSpec value)

namespace Bounds

open Spec TorchLean
open NN.IR
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph

open Tests.Utils (assertBoundAt assertNoBoundAt)

private def checkConvolutionGuards : IO Unit := do
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
      spec := { kernel := Tensor.castShape weights (by simp [kernel]), bias := bias } }
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
  if crownGraphSemanticsSupported groupedGraph groupedStore then
    throw <| IO.userError "convolution with zero groups was accepted"
  assertNoBoundAt "convolution with zero groups" (runIBP groupedGraph groupedStore) 1

  let dilatedParams := { base with dilation := dilationZero }
  let dilatedConfig := { baseConfig with dilation := dilationZero }
  let dilatedGraph := graphFor dilatedConfig [2, 6]
  let dilatedStore := storeFor dilatedParams
  if crownGraphSemanticsSupported dilatedGraph dilatedStore then
    throw <| IO.userError "convolution with zero dilation was accepted"
  assertNoBoundAt "convolution with zero dilation" (runIBP dilatedGraph dilatedStore) 1

  let asymmetricParams := { base with paddingAfter := paddingAfterZero }
  let asymmetricConfig := { baseConfig with paddingAfter := paddingAfterZero }
  let asymmetricGraph := graphFor asymmetricConfig [2, 5]
  let asymmetricStore := storeFor asymmetricParams
  if crownGraphSemanticsSupported asymmetricGraph asymmetricStore then
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
private def checkLayerNormPayloadGuard : IO Unit := do
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
  if crownGraphSemanticsSupported graph invalidStore then
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
private def checkAffineCancellation : IO Unit := do
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
  if getAtOrZero output.lo [1] > getAtOrZero output.hi [0] then
    throw <| IO.userError "CROWN certified the wrong class after coefficient cancellation"
  let net : TwoLayerMLP Float32 1 3 2 :=
    { hiddenWeight := w1, hiddenBias := b1, outputWeight := w2, outputBias := b2 }
  let mlpBounds := boundAffineCrown net (Box.point input)
  unless mlpBounds.lo.getScalar 0 ≤ 0.1 && 0.125 ≤ mlpBounds.hi.getScalar 0 do
    throw <| IO.userError "standalone MLP CROWN lost coefficient-rounding error"

/-- Negative weights must consume a lower parent bound, even in the upper-only API. -/
private def checkUpperAffineSign : IO Unit := do
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
private def checkRoundedReluAlphaRejected : IO Unit := do
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
private def checkIBPForwardSupport : IO Unit := do
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

namespace SumEquationRegression

open DirectedBackward

/-- Summing two ones cannot give zero, even when no IBP parent row exists. -/
theorem real_sum_rejects_wrong_value_without_row :
    ¬ RealNodeEquation
      #[{ id := 0, parents := #[], kind := .input, outShape := [2] },
        { id := 1, parents := #[0], kind := .sum, outShape := .scalar }]
      ({} : ParamStore FP32) #[] (fun id => if id = 0 then 2 else 1)
      (fun id _ => if id = 0 then 1 else 0) 1 := by
  norm_num [RealNodeEquation, unaryParent?]

/-- The actual tensor sum satisfies the equation for every shape and any cached IBP rows. -/
theorem actual_tensor_sum_equation {s : Shape} (t : Tensor ℝ s)
    (ps : ParamStore FP32) (ibp : Array (Option (FlatBox FP32))) :
    RealNodeEquation
      #[{ id := 0, parents := #[], kind := .input, outShape := s },
        { id := 1, parents := #[0], kind := .sum, outShape := .scalar }]
      ps ibp (fun id => if id = 0 then s.size else 1)
      (fun id coordinate => if id = 0 then
        if h : coordinate < s.size then t (Shape.Coord.unlinearize ⟨coordinate, h⟩) else 0
      else Tensor.sumSpec t) 1 := by
  change ∀ p, unaryParent? #[0] = some p →
    1 = 1 ∧ Tensor.sumSpec t =
      ∑ i : Fin (if p = 0 then s.size else 1),
        if p = 0 then
          if h : i.val < s.size then t (Shape.Coord.unlinearize ⟨i.val, h⟩) else 0
        else Tensor.sumSpec t
  intro p hp
  have hp0 : p = 0 := by simpa [unaryParent?] using hp.symm
  subst p
  constructor
  · rfl
  · change Tensor.sumSpec t = ∑ i : Fin s.size,
      if h : i.val < s.size then t (Shape.Coord.unlinearize ⟨i.val, h⟩) else 0
    have hsum : Tensor.sumSpec t =
        ∑ i : Fin s.size, t (Shape.Coord.unlinearize i) := by
      rw [Spec.sum_spec_eq_coord_sum]
      exact ((Shape.Coord.equivFin s).symm.sum_comp (fun c => t c)).symm
    rw [hsum]
    apply Finset.sum_congr rfl
    intro i _
    simp only [dite_eq_left i.isLt]

end SumEquationRegression

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
  DirectedBackward.runIBP_encloses_all g ps
    FP32.normalizationEpsilon_nonneg
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
    FP32.normalizationEpsilon_nonneg hinputs equation
    xB hx output houtput obj hdim hresult

/-! ## Classification verdicts

The graph computes `1 - relu x` and the constant `0.5`. On `[-2, 1]`, optimizing only the upper
affine form falsely certifies the first logit. This checks the complete bound-to-verdict path.
-/

namespace LabelBounds

open NN.MLTheory.CROWN NN.MLTheory.CROWN.Graph
open NN.Verification.Robustness

private def graph : NN.IR.Graph := ⟨#[
  { id := 0, parents := #[], kind := .input, outShape := [1] },
  { id := 1, parents := #[0], kind := .relu, outShape := [1] },
  { id := 2, parents := #[1], kind := .linear, outShape := [2] }]⟩

private def box (lo hi : Float) : FlatBox Float :=
  { dim := 1, lo := [lo], hi := [hi] }

private def params (xB : FlatBox Float) : ParamStore Float :=
  { inputBoxes := ({} : Std.HashMap Nat (FlatBox Float)).insert 0 xB
    linearWB := ({} : Std.HashMap Nat (LinParams Float)).insert 2
      { m := 2, n := 1, w := [[-1], [0]], b := [1, 0.5] } }

private def verdict (xB : FlatBox Float) (label : Nat) : IO Bool :=
  match outputBoxCROWN? graph (params xB) xB 0 2 xB.dim with
  | .ok bounds => pure (TopLabel.check bounds.lo bounds.hi label)
  | .error e => throw <| IO.userError s!"CROWN label bounds verdict failed: {e}"

def run : IO Unit := do
  let crossing := box (-2) 1
  -- The upper form alone would give the first logit the point range [1, 1].
  let affines := runAffine graph (params crossing) { inputId := 0, inputDim := 1 }
    (runIBP graph (params crossing))
  let some upper := affines[2]!
    | throw <| IO.userError "CROWN label bounds: upper affine form missing"
  let separated :=
    if hIn : crossing.dim = upper.inDim then
      let upperOnly := upper.evalOnFlatBox crossing hIn
      if h0 : 0 < upper.outDim then upperOnly.lo.getScalar ⟨0, h0⟩ > 0.5 else false
    else false
  unless separated do
    throw <| IO.userError "CROWN label bounds: the test graph no longer separates the two forms"
  if ← verdict crossing 0 then
    throw <| IO.userError "CROWN label bounds certified label 0, which loses at x = 1"
  if ← verdict crossing 1 then
    throw <| IO.userError "CROWN label bounds certified label 1, which loses at x = -2"
  -- On x ∈ [-2, -1] the first logit is exactly 1 and does win.
  unless ← verdict (box (-2) (-1)) 0 do
    throw <| IO.userError "CROWN label bounds rejected label 0 on a box where it always wins"
  IO.println "CROWN label bounds verdict: ok"


end LabelBounds

/-- Check rejection and rounding behavior in the executable CROWN pipeline. -/

def run : IO Unit := do
  checkConvolutionGuards
  checkLayerNormPayloadGuard
  checkAffineCancellation
  LabelBounds.run
  checkUpperAffineSign
  checkRoundedReluAlphaRejected
  checkIBPForwardSupport

end Bounds

namespace Transfers

open Spec TorchLean
open NN.IR NN.MLTheory.CROWN NN.MLTheory.CROWN.Graph

private def require (condition : Bool) (message : String) : IO Unit :=
  unless condition do throw <| IO.userError s!"CROWN transfer: {message}"

/-- An exactly representable dyadic sum must remain inside convolution's directed bounds. -/
private def checkConvolution (α : Type) [Storage α] [Context α] [BoundOps α]
    [NonlinearBoundOps α] [DecidableLE α] (small : α) (label : String) : IO Unit := do
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
        { kernel := Tensor.castShape (Tensor.full (α := α) [1, 1, 129] 1) (by simp [kernel])
          bias := [0] } }
  let config : ConvConfig :=
    { spatialRank := 1, kernel, stride, padding := [0], dilation := [1], paddingAfter := [0]
      groups := 1, channelAxis := 0, inChannels := 1, outChannels := 1 }
  let graph : NN.IR.Graph := { nodes := #[
    { id := 0, kind := .input, parents := #[], outShape := [1, 129] },
    { id := 1, kind := .conv config, parents := #[0], outShape := [1, 1] }] }
  let ps : ParamStore α :=
    { inputBoxes := Std.HashMap.emptyWithCapacity.insert 0 (pointFlatBox input)
      convCfg := Std.HashMap.emptyWithCapacity.insert 1 parameters }
  let some result := (runIBP graph ps)[1]?.join
    | throw <| IO.userError s!"CROWN transfer: {label} convolution has no bound"
  let exact := (1 : α) + 128 * small
  require (decide (getAtOrZero result.lo [0] ≤ exact) &&
      decide (exact ≤ getAtOrZero result.hi [0]))
    s!"{label} convolution lost small terms"

/-- A rejected LayerNorm transfer stays unresolved through every dependent branch. -/
private def checkFailedParents : IO Unit := do
  let shape : Shape := [2]
  let input : Tensor Float [2] := [-1, 1]
  let parameters : LayerNormParams Float :=
    { normalizedShape := shape, gamma := [0, 0], beta := [3, 3], eps := 0 }
  let initialNodes : Array NN.IR.Node := #[
    { id := 0, kind := .input, parents := #[], outShape := shape },
    { id := 1, kind := .layernorm 0, parents := #[0], outShape := shape }]
  let ps : ParamStore Float :=
    { inputBoxes := Std.HashMap.emptyWithCapacity.insert 0 (pointFlatBox input)
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
    let _ ← NN.Verification.Cert.IBPCert.check graph ps 2 "/unused-failed-transfer.json"
    pure false
  catch error => pure (error.toString.contains "no output box")
  require rejected "failed transfer reached certificate comparison"
  for kind in [NN.IR.OpKind.tanh, .sigmoid, .sin, .cos] do
    for parents in [#[], #[0, 0], #[99]] do
      let malformed : NN.IR.Node := { id := 2, kind, parents, outShape := shape }
      require
        ((ibpStepNodeAt? initialNodes ps #[some (pointFlatBox input)] 2 malformed).isNone)
        "malformed nonlinear parents produced a bound"

/-- A reduction must not conceal a stored matrix's inconsistent output dimension. -/
private def checkMatmulDimensions : IO Unit := do
  let graph : NN.IR.Graph := { nodes := #[
    { id := 0, kind := .input, parents := #[], outShape := [1] },
    { id := 1, kind := .matmul, parents := #[0], outShape := [1] },
    { id := 2, kind := .sum, parents := #[1], outShape := [] }] }
  let ps : ParamStore Float :=
    { inputBoxes := Std.HashMap.emptyWithCapacity.insert 0
        (pointFlatBox ([1] : Tensor Float [1]))
      matmulW := Std.HashMap.emptyWithCapacity.insert 1 { m := 2, n := 1, w := [[1], [2]] } }
  let malformedStores := [
    ps,
    { ps with matmulW :=
        Std.HashMap.emptyWithCapacity.insert 1 { m := 1, n := 2, w := [[1, 2]] } },
    { ps with matmulW := Std.HashMap.emptyWithCapacity }]
  let ctx : AffineCtx := { inputId := 0, inputDim := 1 }
  let input := pointFlatBox ([1] : Tensor Float [1])
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
private def checkIndependentConcat : IO Unit := do
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
private def checkConvolutionDerivatives : IO Unit := do
  let kernel : Tensor Nat [1] := [3]
  let stride : Tensor Nat [1] := [1]
  let parameters : ConvParams Float :=
    { spatialRank := 1, inChannels := 4, outChannels := 4, kernel, stride
      padding := [1], dilation := [2], paddingAfter := [2], groups := 2, inputSpatial := [8]
      kernelNonzero := by intro i; fin_cases i; simp [kernel]
      strideNonzero := by intro i; fin_cases i; simp [stride]
      spec :=
        { kernel := Tensor.castShape (Tensor.full (α := Float) [4, 4, 3] 1) (by simp [kernel])
          bias := [7, 7, 7, 7] } }
  let configuration : ConvConfig :=
    { spatialRank := 1, inChannels := 4, outChannels := 4, kernel, stride
      padding := [1], dilation := [2], paddingAfter := [2], groups := 2, channelAxis := 1 }
  let graph : NN.IR.Graph := { nodes := #[
    { id := 0, kind := .input, parents := #[], outShape := [2, 4, 8] },
    { id := 1, kind := .mulElem, parents := #[0, 0], outShape := [2, 4, 8] },
    { id := 2, kind := .conv configuration, parents := #[1], outShape := [2, 4, 7] }] }
  let point (value : Float) : FlatBox Float :=
    pointFlatBox (Tensor.full (α := Float) [2, 4, 8] value)
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

/-- Exercise transfers whose malformed inputs or rounded accumulations defeat real-only tests. -/
def run : IO Unit := do
  checkConvolution Float (1 / 18014398509481984) "Float"
  checkConvolution Float32 (1 / 67108864) "Float32"
  checkFailedParents
  checkMatmulDimensions
  checkIndependentConcat
  checkConvolutionDerivatives
  IO.println "CROWN transfer: rounding, failure, shape and independent-input checks passed"

end Transfers

/-- Check rejection, rounding and verdicts in the executable CROWN pipeline. -/
def run : IO Unit := do
  Bounds.run
  Transfers.run

end NN.Tests.Verification.Crown

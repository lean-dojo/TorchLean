/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.DirectedIBPFullBackward

/-!
# Directed evaluation of backward affine bounds

The returned affine functions have a real interpretation. Their executable interval evaluator
rounds each product, sum, and final constant outwards before returning a scalar box.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean NN.IR
open TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph.Internal
open scoped BigOperators

noncomputable section

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

/-- Evaluating a pair of real affine enclosures on an enclosing input box retains enclosure. -/
theorem affineBoundsEval_encloses
    (bounds : FlatAffineBounds α) (box : FlatBox α) (x y : Nat → ℝ)
    (hx : RowEncloses box bounds.inDim x) (hy : AffineRowsEnclose bounds x y) :
    RowEncloses
      { dim := bounds.outDim
        lo := (bounds.evalOnFlatBox box hx.1).lo
        hi := (bounds.evalOnFlatBox box hx.1).hi }
      bounds.outDim y := by
  obtain ⟨n, lo, hi⟩ := box
  have hd := hx.1
  change n = bounds.inDim at hd
  subst n
  have hinput (j : Fin bounds.inDim) :
      value (lo.getScalar j) ≤ x j.val ∧ x j.val ≤ value (hi.getScalar j) := by
    simpa only [read_fin] using hx.2 j
  refine ⟨rfl, ?_⟩
  intro i
  have hl := affineEvalOnBox_encloses bounds.loAff
    { lo := lo, hi := hi } (fun j => x j.val) hinput i
  have hu := affineEvalOnBox_encloses bounds.hiAff
    { lo := lo, hi := hi } (fun j => x j.val) hinput i
  simpa only [read_fin, FlatAffineBounds.evalOnFlatBox, FlatBox.getScalarBox,
    FlatBox.loAsDim, FlatBox.hiAsDim, Tensor.cast_shape_rfl] using
      And.intro (hl.1.trans (hy i).1) ((hy i).2.trans hu.2)

/-- The public scalar-box evaluator encloses every point enclosed by its affine argument. -/
theorem evalBackwardObjectiveBox_encloses
    (bounds : FlatAffineBounds α) (xB : FlatBox α) (inputDim : Nat)
    (x y : Nat → ℝ) (hx : RowEncloses xB bounds.inDim x)
    (hy : AffineRowsEnclose bounds x y) {result : FlatBox α}
    (hresult : evalBackwardObjectiveBox? bounds xB inputDim = .ok result) :
    RowEncloses result 1 y := by
  by_cases hi : bounds.inDim = inputDim
  · by_cases hxDim : xB.dim = inputDim
    · by_cases ho : bounds.outDim = 1
      · simp only [evalBackwardObjectiveBox?, hi, hxDim, ho, ↓reduceDIte] at hresult
        have he := Except.ok.inj hresult
        subst result
        have h := affineBoundsEval_encloses bounds xB x y hx hy
        cases bounds with
        | mk n m lower upper =>
            dsimp only at ho
            subst m
            simpa only [FlatAffineBounds.evalOnFlatBoxAsDim, Tensor.cast_shape_rfl] using h
      · simp [evalBackwardObjectiveBox?, hi, hxDim, ho] at hresult
    · simp [evalBackwardObjectiveBox?, hi, hxDim] at hresult
  · simp [evalBackwardObjectiveBox?, hi] at hresult

/-- The rounded objective workflow returns an interval containing its real output objective,
including all directed arithmetic in both propagation and final evaluation, provided the forward
IBP boxes in `point` are sound. `backwardObjectiveBox_encloses_runIBP_all` derives those enclosures
from the input boxes and the real node equations. -/
theorem backwardObjectiveBox_encloses
    (hrounded : BoundOps.supportsExactAffineReassociation (α := α) = false)
    {g : Graph} {ps : ParamStore α} {ibp : Array (Option (FlatBox α))}
    {ctx : AffineCtx} {dims : Nat → Nat} {v : Nat → Nat → ℝ}
    (point : GraphPoint g.nodes ps ibp ctx dims v)
    (xB : FlatBox α) (hx : RowEncloses xB ctx.inputDim (v ctx.inputId))
    (output : Nat) (houtput : output < g.nodes.size) (obj : FlatTensor α)
    (hdim : obj.n = dims output) {result : FlatBox α}
    (hresult : backwardObjectiveBox? g ps ctx ibp xB output obj = .ok result) :
    RowEncloses result 1
      (fun _ => dot (dims output) (fun i => value (getAtOrZero obj.v [i])) (v output)) := by
  unfold backwardObjectiveBox? at hresult
  cases hb : runCROWNBackwardObjective g ps ctx ibp output obj with
  | none => simp [hb] at hresult
  | some bounds =>
      simp only [hb] at hresult
      have hs := runCROWNBackwardObjective_encloses hrounded point
        output houtput obj hdim hb
      exact evalBackwardObjectiveBox_encloses bounds xB ctx.inputDim _ _
        (by simpa only [hs.1] using hx) hs.2.2 hresult


/-- The rounded objective workflow, run on the boxes of `runIBP`, returns an interval containing
its real output objective. Forward IBP, the backward sweep, and the final evaluation are all
covered, for graphs whose node kinds pass `ibpForwardSupported`. -/
theorem backwardObjectiveBox_encloses_runIBP [NonlinearBoundOps α] [LawfulNonlinearBoundOps α]
    (hrounded : BoundOps.supportsExactAffineReassociation (α := α) = false)
    {g : Graph} {ps : ParamStore α} {ctx : AffineCtx} {dims : Nat → Nat} {v : Nat → Nat → ℝ}
    (input_lt : ctx.inputId < g.nodes.size)
    (input_dim : dims ctx.inputId = ctx.inputDim)
    (input_kind : g.nodes[ctx.inputId]!.kind = .input)
    (node_id : ∀ id, id < g.nodes.size → g.nodes[id]!.id = id)
    (parent_lt : ∀ id, id < g.nodes.size → ∀ p ∈ g.nodes[id]!.parents, p < id)
    (hsupported : ibpForwardSupported g.nodes = true)
    (hinputs : InputsInBoxes g.nodes ps dims v)
    (equation : ∀ id, id < g.nodes.size → NodeEquation g.nodes ps (runIBP g ps) dims v id)
    (xB : FlatBox α) (hx : RowEncloses xB ctx.inputDim (v ctx.inputId))
    (output : Nat) (houtput : output < g.nodes.size) (obj : FlatTensor α)
    (hdim : obj.n = dims output) {result : FlatBox α}
    (hresult : backwardObjectiveBox? g ps ctx (runIBP g ps) xB output obj = .ok result) :
    RowEncloses result 1
      (fun _ => dot (dims output) (fun i => value (getAtOrZero obj.v [i])) (v output)) :=
  backwardObjectiveBox_encloses hrounded
    (GraphPoint.ofRunIBP input_lt input_dim input_kind node_id parent_lt hsupported hinputs
      equation) xB hx output houtput obj hdim hresult

/-- Forward IBP, rounded backward propagation, and final interval evaluation enclose the real
objective for every graph operation. Intermediate bounds are proved from the input boxes. -/
theorem backwardObjectiveBox_encloses_runIBP_all
    [NonlinearBoundOps α] [LawfulNonlinearBoundOps α] [LawfulMinBoundOps α]
    (hrounded : BoundOps.supportsExactAffineReassociation (α := α) = false)
    {g : Graph} {ps : ParamStore α} {ctx : AffineCtx} {dims : Nat → Nat} {v : Nat → Nat → ℝ}
    (input_lt : ctx.inputId < g.nodes.size)
    (input_dim : dims ctx.inputId = ctx.inputDim)
    (input_kind : g.nodes[ctx.inputId]!.kind = .input)
    (node_id : ∀ id, id < g.nodes.size → g.nodes[id]!.id = id)
    (parent_lt : ∀ id, id < g.nodes.size → ∀ p ∈ g.nodes[id]!.parents, p < id)
    (hepsilon : 0 ≤ value (TorchLean.normalizationEpsilon : α))
    (hinputs : InputsInBoxes g.nodes ps dims v)
    (equation : ∀ id, id < g.nodes.size →
      RealNodeEquation g.nodes ps (runIBP g ps) dims v id)
    (xB : FlatBox α) (hx : RowEncloses xB ctx.inputDim (v ctx.inputId))
    (output : Nat) (houtput : output < g.nodes.size) (obj : FlatTensor α)
    (hdim : obj.n = dims output) {result : FlatBox α}
    (hresult : backwardObjectiveBox? g ps ctx (runIBP g ps) xB output obj = .ok result) :
    RowEncloses result 1
      (fun _ => dot (dims output) (fun i => value (getAtOrZero obj.v [i])) (v output)) :=
  backwardObjectiveBox_encloses hrounded
    (GraphPoint.ofRunIBPAll input_lt input_dim input_kind node_id parent_lt hepsilon hinputs
      equation) xB hx output houtput obj hdim hresult

end

end NN.MLTheory.CROWN.Graph.DirectedBackward

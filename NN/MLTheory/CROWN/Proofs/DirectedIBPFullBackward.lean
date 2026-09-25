/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.DirectedConvSemantics
public import NN.MLTheory.CROWN.Proofs.DirectedIBPFullSoundness
public import NN.MLTheory.CROWN.Proofs.DirectedIBPStructuralSemantics

/-!
# Rounded CROWN soundness from the complete forward pass

The actual real equations used by the full IBP proof also imply the equations used by the
backward sweep. Convolution coefficients preserve the stored weights, and structural operations
use the same checked coordinate maps in both directions.

`GraphPoint.ofRunIBPAll` derives every intermediate enclosure from the input boxes. The resulting
CROWN theorems have no operation-family restriction and require no second equation for a node.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean NN.IR
open TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph.Internal

noncomputable section

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

/-- Real node semantics supplies each equation consumed by the backward sweep. -/
theorem RealNodeEquation.nodeEquation
    {nodes : Array Node} {ps : ParamStore α} {ibp : Array (Option (FlatBox α))}
    {dims : Nat → Nat} {v : Nat → Nat → ℝ} {id : Nat}
    (heq : RealNodeEquation nodes ps ibp dims v id)
    (hparents : ∀ p ∈ nodes[id]!.parents, p < nodes.size) :
    NodeEquation nodes ps ibp dims v id := by
  cases hk : nodes[id]!.kind <;> simp only [RealNodeEquation, hk] at heq
  case conv configuration => exact heq.nodeEquation hk
  case matmul => exact heq.1
  case concat | transpose | permute | broadcastTo | reduceSum | reduceMean | maxPool | avgPool =>
    exact StructuralRealNodeEquation.nodeEquation
      (by simp only [ibpStructuralSupportedNode, hk]) heq hparents
  case softmax | hardMaskedSoftmax | layernorm | batchNormEval
      | safeLog | mseLoss | randUniform | bernoulliMask =>
    simp only [NodeEquation, hk]
  all_goals exact heq

variable [NonlinearBoundOps α] [LawfulNonlinearBoundOps α] [LawfulMinBoundOps α]

/-- Build a backward graph point from input enclosures and the actual real node equations. -/
theorem GraphPoint.ofRunIBPAll {g : Graph} {ps : ParamStore α} {ctx : AffineCtx}
    {dims : Nat → Nat} {v : Nat → Nat → ℝ}
    (input_lt : ctx.inputId < g.nodes.size)
    (input_dim : dims ctx.inputId = ctx.inputDim)
    (input_kind : g.nodes[ctx.inputId]!.kind = .input)
    (node_id : ∀ id, id < g.nodes.size → g.nodes[id]!.id = id)
    (parent_lt : ∀ id, id < g.nodes.size → ∀ p ∈ g.nodes[id]!.parents, p < id)
    (hepsilon : 0 ≤ value (TorchLean.normalizationEpsilon : α))
    (hinputs : InputsInBoxes g.nodes ps dims v)
    (equation : ∀ id, id < g.nodes.size →
      RealNodeEquation g.nodes ps (runIBP g ps) dims v id) :
    GraphPoint g.nodes ps (runIBP g ps) ctx dims v where
  input_lt := input_lt
  input_dim := input_dim
  input_kind := input_kind
  node_id := node_id
  parent_lt := parent_lt
  ibp_encloses := runIBP_encloses_all g ps hepsilon parent_lt hinputs equation
  equation := fun id hid =>
    (equation id hid).nodeEquation fun p hp => lt_trans (parent_lt id hid p hp) hid

/-- A successful rounded backward objective encloses the real objective after the full IBP pass.
All intermediate enclosures follow from the input boxes and real node equations. -/
theorem runCROWNBackwardObjective_encloses_runIBP_all
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
    (output : Nat) (houtput : output < g.nodes.size) (obj : FlatTensor α)
    (hdim : obj.n = dims output) {bounds : FlatAffineBounds α}
    (hresult : runCROWNBackwardObjective g ps ctx (runIBP g ps) output obj = some bounds) :
    bounds.inDim = ctx.inputDim ∧ bounds.outDim = 1 ∧
      AffineRowsEnclose bounds (v ctx.inputId)
        (fun _ => dot (dims output) (fun i => value (getAtOrZero obj.v [i])) (v output)) :=
  runCROWNBackwardObjective_encloses hrounded
    (GraphPoint.ofRunIBPAll input_lt input_dim input_kind node_id parent_lt hepsilon hinputs
      equation) output houtput obj hdim hresult

/-- Nodewise rounded CROWN bounds enclose the real output after the full forward pass. -/
theorem directedNodeBounds_encloses_runIBP_all
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
    (output : Nat) (houtput : output < g.nodes.size)
    (hdim : dims output = g.nodes[output]!.outShape.size)
    {bounds : FlatAffineBounds α}
    (hresult : directedNodeBounds? g ps ctx (runIBP g ps) output = some bounds) :
    bounds.inDim = ctx.inputDim ∧ bounds.outDim = dims output ∧
      AffineRowsEnclose bounds (v ctx.inputId) (v output) :=
  directedNodeBounds_encloses
    (GraphPoint.ofRunIBPAll input_lt input_dim input_kind node_id parent_lt hepsilon hinputs
      equation) output houtput hdim hresult

end

end NN.MLTheory.CROWN.Graph.DirectedBackward

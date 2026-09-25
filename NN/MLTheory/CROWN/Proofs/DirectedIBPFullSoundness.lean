/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.DirectedIBPSoundness
public import NN.MLTheory.CROWN.Proofs.DirectedIBPConvolution
public import NN.MLTheory.CROWN.Proofs.DirectedIBPPointwise
public import NN.MLTheory.CROWN.Proofs.DirectedIBPStructural
public import NN.MLTheory.CROWN.Proofs.DirectedIBPNormalization

/-!
# Rounded forward soundness for every graph operation

`RealNodeEquation` describes the real operation at each node. Tensor operations use their actual
coordinate maps, reductions, normalization formulas, and interpreted stored parameters. Random
nodes use the seeded real Spec operations. Full-tensor sum equations hold independently of IBP
row availability.

`runIBP_encloses_all` proves enclosure for every successful entry of the executable pass. It has
no operation-family restriction and assumes no enclosure for intermediate nodes. Invalid shapes,
missing parameters, and unavailable arithmetic transfers can still return `none`.
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

/-- The real semantics of every graph operation.

The spatial convolution and structural equations stand on their own. In particular, they do not
assume a second affine equation about coefficients computed by the verifier. Unary stored-weight
and binary tensor matrix multiplication share an operation tag; their parent counts distinguish
the two equations. Stored `linear` nodes use vector affine equations; the public IR bridge for
`linear` requires vector-shaped parents.
-/
def RealNodeEquation (nodes : Array Node) (ps : ParamStore α)
    (ibp : Array (Option (FlatBox α))) (dims : Nat → Nat) (v : Nat → Nat → ℝ)
    (id : Nat) : Prop :=
  match nodes[id]!.kind with
  | .sum =>
      ∀ p, unaryParent? nodes[id]!.parents = some p →
        dims id = 1 ∧ v id 0 = ∑ i : Fin (dims p), v p i.val
  | .conv configuration => ConvolutionNodeEquation nodes ps dims v id configuration
  | .matmul =>
      NodeEquation nodes ps ibp dims v id ∧ BinaryMatmulNodeEquation nodes dims v id
  | .concat _ | .transpose .. | .permute _ | .broadcastTo ..
  | .reduceSum _ | .reduceMean _ | .maxPool _ | .avgPool _ =>
      StructuralRealNodeEquation nodes dims v id
  | .softmax _ | .hardMaskedSoftmax _ | .layernorm _ | .batchNormEval .. =>
      NormalizationNodeEquation nodes ps dims v id
  | .safeLog | .mseLoss | .randUniform _ | .bernoulliMask _ =>
      PointwiseRealNodeEquation nodes dims v id
  | _ => NodeEquation nodes ps ibp dims v id

variable [NonlinearBoundOps α] [LawfulNonlinearBoundOps α] [LawfulMinBoundOps α]

/-- Every successful IBP transfer encloses the corresponding real operation.

The epsilon premise concerns the backend's fixed default normalization constant. Stored
normalization payloads retain the executable positivity checks on their own epsilon.
-/
theorem ibpStepNodeAt?_all_encloses
    (hepsilon : 0 ≤ value (TorchLean.normalizationEpsilon : α))
    {nodes : Array Node} {ps : ParamStore α} {boxes ibp : Array (Option (FlatBox α))}
    {dims : Nat → Nat} {v : Nat → Nat → ℝ} {id : Nat}
    (hinput : nodes[id]!.kind = .input →
      ∀ box, ps.inputBoxes[id]? = some box → RowEncloses box (dims id) (v id))
    (heq : RealNodeEquation nodes ps ibp dims v id)
    (hagree : ∀ p ∈ nodes[id]!.parents, (boxes[p]?).join = ibp[p]!)
    (henc : ∀ p ∈ nodes[id]!.parents, ∀ box, ibp[p]! = some box →
      RowEncloses box (dims p) (v p))
    {box : FlatBox α} (hstep : ibpStepNodeAt? nodes ps boxes id nodes[id]! = some box) :
    RowEncloses box (dims id) (v id) := by
  have hget : ∀ p ∈ nodes[id]!.parents, ∀ B, (boxes[p]?).join = some B →
      RowEncloses B (dims p) (v p) := fun p hp B hB =>
    henc p hp B ((hagree p hp).symm.trans hB)
  cases hk : nodes[id]!.kind <;> simp only [RealNodeEquation, hk] at heq
  case sum =>
    apply ibpStepNodeAt?_encloses
      (by simp only [ibpForwardSupportedNode, hk]) hinput ?_ hagree henc hstep
    simp only [NodeEquation, hk]
    intro p hp B hB
    obtain ⟨h1, hv⟩ := heq p hp
    exact ⟨h1, (henc p (mem_of_unaryParent?_eq_some hp) B hB).1.symm, hv⟩
  case matmul =>
    rcases ibpStep_matmul_parents hk hstep with hunary | ⟨p, q, hparents⟩
    · exact ibpStepNodeAt?_encloses
        (by simp only [ibpForwardSupportedNode, hk, hunary, beq_self_eq_true])
        hinput heq.1 hagree henc hstep
    · exact ibpStep_binaryMatmul_encloses hk hparents heq.2 hget hstep
  case conv configuration =>
    exact ibpStep_convolution_encloses hk heq hget hstep
  case concat | transpose | permute | broadcastTo | reduceSum | reduceMean | maxPool | avgPool =>
    exact ibpStepNodeAt?_structural_encloses
      (by simp only [ibpStructuralSupportedNode, hk]) heq hagree henc hstep
  case softmax | hardMaskedSoftmax | layernorm | batchNormEval =>
    exact ibpStepNodeAt?_normalization_encloses hepsilon
      (by simp only [normalizationNodeKind, hk]) heq hagree henc hstep
  case abs | maxElem | minElem | softplus =>
    exact ibpStepNodeAt?_pointwise_encloses
      (by simp only [ibpPointwiseSupportedNode, hk]) heq
      (by simp only [PointwiseRealNodeEquation, hk]) hagree henc hstep
  case safeLog | mseLoss | randUniform | bernoulliMask =>
    exact ibpStepNodeAt?_pointwise_encloses
      (by simp only [ibpPointwiseSupportedNode, hk])
      (by simp only [NodeEquation, hk]) heq hagree henc hstep
  all_goals
    exact ibpStepNodeAt?_encloses
      (by simp only [ibpForwardSupportedNode, hk]) hinput heq hagree henc hstep

/-- The complete rounded forward pass encloses every real node value for which it returns a box.

Inputs lie in their seeded boxes, parents precede consumers, and nodes obey their actual real
equations. The proof supplies all intermediate enclosures by induction, including convolution,
binary matrix multiplication, structural operations, normalization, and random realizations.
-/
theorem runIBP_encloses_all (g : Graph) (ps : ParamStore α)
    (hepsilon : 0 ≤ value (TorchLean.normalizationEpsilon : α))
    {dims : Nat → Nat} {v : Nat → Nat → ℝ}
    (hparent : ∀ id, id < g.nodes.size → ∀ p ∈ g.nodes[id]!.parents, p < id)
    (hinputs : InputsInBoxes g.nodes ps dims v)
    (hequation : ∀ id, id < g.nodes.size →
      RealNodeEquation g.nodes ps (runIBP g ps) dims v id) :
    ∀ id, id < g.nodes.size → ∀ box, (runIBP g ps)[id]! = some box →
      RowEncloses box (dims id) (v id) := by
  apply runIBP_encloses_of_step g ps hparent
  intro id hid boxes hagree henc box hstep
  exact ibpStepNodeAt?_all_encloses hepsilon (hinputs id hid) (hequation id hid)
    hagree henc hstep

end

end NN.MLTheory.CROWN.Graph.DirectedBackward

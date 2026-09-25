/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.DirectedIBPBasic
public import NN.MLTheory.CROWN.Proofs.DirectedIBPSoftmax
public import NN.MLTheory.CROWN.Proofs.DirectedIBPLayerNorm
public import NN.MLTheory.CROWN.Proofs.DirectedIBPBatchNorm
import all NN.MLTheory.CROWN.Graph.Engine.Base

/-!
# Rounded forward normalization steps

The node equations below describe the real tensor specifications. Stored affine parameters,
running statistics, and epsilon are interpreted through `LawfulBoundOps.toReal`. The step theorem
then proves enclosure by the actual executable transfer, using the usual parent agreement and
parent enclosure hypotheses.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean TorchLean.Tensor NN.IR
open NN.MLTheory.CROWN.Graph.Internal

noncomputable section

/-- The four normalization operations covered by this forward-step theorem. -/
def normalizationNodeKind : OpKind → Bool
  | .softmax _ | .hardMaskedSoftmax _ | .layernorm _ | .batchNormEval _ _ => true
  | _ => false

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

/-- Real tensor equations for the normalization family, independently of any interval transfer.
The successful shape and payload resolutions name the same mathematical operations as IR
evaluation. No equation mentions a returned box or assumes an enclosure. -/
def NormalizationNodeEquation (nodes : Array Node) (ps : ParamStore α)
    (dims : Nat → Nat) (v : Nat → Nat → ℝ) (id : Nat) : Prop :=
  let node := nodes[id]!
  let s := node.outShape
  match node.kind with
  | .softmax axis =>
      ∀ p, unaryParent? node.parents = some p →
        dims p = s.size ∧ dims id = s.size ∧
          ∃ haxis : Shape.AxisInBounds axis s,
            letI := haxis
            ∀ c : s.Coord, v id (Shape.Coord.linearize c).val =
              Activation.softmaxSpec axis (tensorOfFlatValues s (v p)) c
  | .hardMaskedSoftmax mask =>
      ∀ p, unaryParent? node.parents = some p →
        dims p = s.size ∧ dims id = s.size ∧
          ∀ decoded : Tensor Bool mask.shape,
            (HardMask.toTensor? mask).toOption = some decoded →
              ∀ hshape : mask.shape = s, ∀ c : s.Coord,
                v id (Shape.Coord.linearize c).val =
                  Spec.hardMaskedSoftmaxSpec (tensorOfFlatValues s (v p)) (hshape ▸ decoded) c
  | .layernorm axis =>
      ∀ p, unaryParent? node.parents = some p →
        dims p = s.size ∧ dims id = s.size ∧
          LayerNormRealEquation s axis ps.layerNorm[id]? (v p) (v id)
  | .batchNormEval axis _ =>
      ∀ p, unaryParent? node.parents = some p →
        ∀ parent, nodes[p]? = some parent →
          ∀ config, ps.batchNormEval[id]? = some config →
            dims p = parent.outShape.size ∧ dims id = parent.outShape.size ∧
              BatchNormRealEquation parent.outShape axis config (v p) (v id)
  | _ => True

variable [NonlinearBoundOps α] [LawfulNonlinearBoundOps α]

/-- Successful normalization branches enclose their real node equations under the same parent
agreement and parent enclosure interface as the generic forward-step theorem. -/
theorem ibpStepNodeAt?_normalization_encloses
    (hepsilon : 0 ≤ value (TorchLean.normalizationEpsilon : α))
    {nodes : Array Node} {ps : ParamStore α} {boxes ibp : Array (Option (FlatBox α))}
    {dims : Nat → Nat} {v : Nat → Nat → ℝ} {id : Nat}
    (hfamily : normalizationNodeKind nodes[id]!.kind = true)
    (heq : NormalizationNodeEquation nodes ps dims v id)
    (hagree : ∀ p ∈ nodes[id]!.parents, (boxes[p]?).join = ibp[p]!)
    (henc : ∀ p ∈ nodes[id]!.parents, ∀ box, ibp[p]! = some box →
      RowEncloses box (dims p) (v p))
    {box : FlatBox α} (hstep : ibpStepNodeAt? nodes ps boxes id nodes[id]! = some box) :
    RowEncloses box (dims id) (v id) := by
  have hunary {p : Nat} (hp : unaryParent? nodes[id]!.parents = some p)
      (B : FlatBox α) (hB : (boxes[p]?).join = some B) :
      RowEncloses B (dims p) (v p) :=
    henc p (mem_of_unaryParent?_eq_some hp) B
      ((hagree p (mem_of_unaryParent?_eq_some hp)).symm.trans hB)
  cases hk : nodes[id]!.kind <;>
    simp only [normalizationNodeKind, hk, Bool.false_eq_true] at hfamily
  all_goals
    simp only [NormalizationNodeEquation, hk] at heq
    rw [ibpStepNodeAt?, hk] at hstep
    dsimp only at hstep
  case softmax axis =>
    split at hstep
    · contradiction
    · obtain ⟨p, input, hp, hinput, hout⟩ := unary_lookup hstep
      obtain ⟨_, hd, haxis, hv⟩ := heq p hp
      let _ := haxis
      split at hout
      · rename_i hdim
        have houtEq := Option.some.inj hout
        cases houtEq
        rw [hd, hdim]
        exact ibpSoftmaxRange_encloses axis (tensorOfFlatValues _ (v p)) hv
      · contradiction
  case hardMaskedSoftmax mask =>
    obtain ⟨p, input, hp, hinput, hout⟩ := unary_lookup hstep
    obtain ⟨_, hd, hv⟩ := heq p hp
    split at hout
    · rename_i hdim
      split at hout
      · rename_i hshape
        obtain ⟨decoded, hdecoded, hout⟩ := Option.bind_eq_some_iff.mp hout
        have houtEq := Option.some.inj hout
        cases houtEq
        rw [hd, rowEncloses_flatten_iff]
        intro c
        rw [hv decoded hdecoded hshape c]
        exact ibpHardMaskedSoftmaxLastTensor_encloses (α := α) _ _
          (tensorOfFlatValues _ (v p)) (hshape ▸ decoded) c
      · contradiction
    · contradiction
  case layernorm axis =>
    split at hstep
    · contradiction
    · obtain ⟨p, input, hp, hinput, hout⟩ := unary_lookup hstep
      obtain ⟨hdp, hd, hv⟩ := heq p hp
      have hx := hunary hp input hinput
      rw [hdp] at hx
      rw [hd]
      split at hout
      · cases hparameters : ps.layerNorm[id]? with
        | none =>
            simp only [hparameters] at hout hv
            exact ibpLayerNormBox?_encloses hepsilon hx hv hout
        | some parameters =>
            simp only [hparameters] at hout hv
            exact ibpLayerNormPayloadBox?_encloses hx hv hout
      · contradiction
  case batchNormEval axis channels =>
    obtain ⟨p, hp, hstep⟩ := Option.bind_eq_some_iff.mp hstep
    obtain ⟨parent, hparent, hstep⟩ := Option.bind_eq_some_iff.mp hstep
    obtain ⟨config, hconfig, hstep⟩ := Option.bind_eq_some_iff.mp hstep
    obtain ⟨_, _, hstep⟩ := Option.bind_eq_some_iff.mp hstep
    split at hstep
    · contradiction
    · obtain ⟨input, hinput, hstep⟩ := Option.bind_eq_some_iff.mp hstep
      obtain ⟨hdp, hd, hv⟩ := heq p hp parent hparent config hconfig
      have hx := hunary hp input hinput
      rw [hdp] at hx
      rw [hd]
      exact ibpBatchNormEval?_encloses hx hv hstep

end

end NN.MLTheory.CROWN.Graph.DirectedBackward

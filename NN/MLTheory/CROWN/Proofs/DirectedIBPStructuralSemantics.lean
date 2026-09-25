/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.DirectedIBPStructural
public import NN.MLTheory.CROWN.Proofs.DirectedIBPAxisSemantics

/-!
# Structural real semantics and the backward interface

The checked backward concat layout preserves the forward layout. Transpose and permutation use
the coordinate bijection of the exact tensor evaluator. These facts derive the legacy backward
equations from the actual real semantics, without a second semantic assumption.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph.Internal

noncomputable section

variable {α : Type} [Storage α] [Context α]

/-- A successful backward concat check preserves the forward layout and occurrence count. -/
theorem concatBackwardLayout?_forward
    {nodes : Array Node} {ibp : Array (Option (FlatBox α))}
    {node : Node} {axis : Nat} {layout : ConcatLayout}
    (h : concatBackwardLayout? nodes ibp node axis = some layout) :
    concatNodeLayout? nodes node axis = some layout ∧
      node.parents.size = layout.lengths.length := by
  unfold concatBackwardLayout? at h
  obtain ⟨candidate, hforward, h⟩ := Option.bind_eq_some_iff.mp h
  split at h
  next hsize =>
    obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
    split at h
    · obtain rfl := Option.some.inj h
      exact ⟨hforward, by simpa only [beq_iff_eq] using hsize⟩
    · contradiction
  next => contradiction

variable [BoundOps α] [LawfulBoundOps α]

theorem permutationEquation_of_eval
    {dims : Nat → Nat} {v : Nat → Nat → ℝ} {id p : Nat}
    {s t : Shape} {forward : Array Nat}
    (hdp : dims p = s.size) (hd : dims id = t.size)
    (heval : NN.IR.Graph.permuteSomeTensor ⟨s, realTensor s (v p)⟩ forward =
      .ok ⟨t, realTensor t (v id)⟩) :
    PermutationEquation dims v id p t forward := by
  unfold PermutationEquation
  rw [hdp, hd]
  exact permuteSomeTensor_flatEquation heval

/-- The exact forward concat equation supplies the backward concat node equation. -/
theorem StructuralRealNodeEquation.concat_nodeEquation
    {nodes : Array Node} {ps : ParamStore α} {ibp : Array (Option (FlatBox α))}
    {dims : Nat → Nat} {v : Nat → Nat → ℝ} {id axis : Nat}
    (h : StructuralRealNodeEquation nodes dims v id)
    (hkind : nodes[id]!.kind = .concat axis) :
    NodeEquation nodes ps ibp dims v id := by
  simp only [StructuralRealNodeEquation, hkind] at h
  simp only [NodeEquation, hkind]
  intro layout hlayout
  obtain ⟨hforward, hsize⟩ := concatBackwardLayout?_forward hlayout
  obtain ⟨hd, hcoords⟩ := h layout hforward
  have hparent (p : Fin layout.lengths.length) :
      nodes[id]!.parents[p.val]? = some nodes[id]!.parents[p.val]! := by
    have hp : p.val < nodes[id]!.parents.size := by omega
    simp [hp]
  exact ⟨hd, hsize, fun p => (hcoords p _ (hparent p)).1,
    fun p => (hcoords p _ (hparent p)).2⟩

/-- Exact real permutation semantics supplies the backward permutation equation. -/
theorem StructuralRealNodeEquation.permute_nodeEquation
    {nodes : Array Node} {ps : ParamStore α} {ibp : Array (Option (FlatBox α))}
    {dims : Nat → Nat} {v : Nat → Nat → ℝ} {id : Nat} {forward : Array Nat}
    (h : StructuralRealNodeEquation nodes dims v id)
    (hkind : nodes[id]!.kind = .permute forward)
    (hparents : ∀ p ∈ nodes[id]!.parents, p < nodes.size) :
    NodeEquation nodes ps ibp dims v id := by
  simp only [StructuralRealNodeEquation, hkind] at h
  simp only [NodeEquation, hkind]
  intro p hp
  have hvalid := hparents p (NN.IR.mem_of_unaryParent?_eq_some hp)
  have hlookup : nodes[p]? = some nodes[p]! := by simp [hvalid]
  obtain ⟨hdp, hd, heval⟩ := h p nodes[p]! hp hlookup
  exact permutationEquation_of_eval hdp hd heval

/-- Exact real transpose semantics uses the same checked axis permutation as the backward pass. -/
theorem StructuralRealNodeEquation.transpose_nodeEquation
    {nodes : Array Node} {ps : ParamStore α} {ibp : Array (Option (FlatBox α))}
    {dims : Nat → Nat} {v : Nat → Nat → ℝ} {id a b : Nat}
    (h : StructuralRealNodeEquation nodes dims v id)
    (hkind : nodes[id]!.kind = .transpose a b)
    (hparents : ∀ p ∈ nodes[id]!.parents, p < nodes.size) :
    NodeEquation nodes ps ibp dims v id := by
  simp only [StructuralRealNodeEquation, hkind] at h
  simp only [NodeEquation, hkind]
  intro p hp forward hforward
  have hvalid := hparents p (NN.IR.mem_of_unaryParent?_eq_some hp)
  have hlookup : nodes[p]? = some nodes[p]! := by simp [hvalid]
  obtain ⟨hdp, hd, heval⟩ := h p nodes[p]! hp hlookup
  cases hperm : NN.IR.OpContracts.transposePerm nodes[p]!.outShape.rank a b with
  | error err => simp only [hperm, Except.toOption, reduceCtorEq] at hforward
  | ok perm =>
      have heq : perm = forward :=
        Option.some.inj (by simpa only [hperm, Except.toOption] using hforward)
      subst forward
      simp only [transposeTensor, hperm, Bind.bind, Except.bind] at heval
      exact permutationEquation_of_eval hdp hd heval

/--
One real structural semantics supplies the backward node equations. Valid parent indices are
ordinary graph topology evidence; no interval enclosure or redundant value equation is assumed.
-/
theorem StructuralRealNodeEquation.nodeEquation
    {nodes : Array Node} {ps : ParamStore α} {ibp : Array (Option (FlatBox α))}
    {dims : Nat → Nat} {v : Nat → Nat → ℝ} {id : Nat}
    (hsupported : ibpStructuralSupportedNode nodes[id]! = true)
    (heq : StructuralRealNodeEquation nodes dims v id)
    (hparents : ∀ p ∈ nodes[id]!.parents, p < nodes.size) :
    NodeEquation nodes ps ibp dims v id := by
  cases hkind : nodes[id]!.kind <;>
    simp only [ibpStructuralSupportedNode, hkind, Bool.false_eq_true] at hsupported
  all_goals first
    | exact heq.concat_nodeEquation hkind
    | exact heq.permute_nodeEquation hkind hparents
    | exact heq.transpose_nodeEquation hkind hparents
    | simp only [NodeEquation, hkind]

end

end NN.MLTheory.CROWN.Graph.DirectedBackward

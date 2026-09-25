/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.DirectedIBPConcat
public import NN.MLTheory.CROWN.Proofs.DirectedIBPReduction
public import NN.MLTheory.CROWN.Proofs.DirectedIBPAvgPool

/-!
# Rounded forward IBP for structural operations

The equations below use exact tensor operations on the real parent coordinates. The family
predicate selects operation kinds; the executable transfers retain their full shape, axis,
window, and arithmetic guards.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean NN.IR
open TorchLean.Tensor
open NN.MLTheory.CROWN

noncomputable section

/-- The structural, axis-reduction, and pooling operation family, at every configuration. -/
def ibpStructuralSupportedNode (node : Node) : Bool :=
  match node.kind with
  | .concat _ | .transpose .. | .permute _ | .broadcastTo ..
  | .reduceSum _ | .reduceMean _ | .maxPool _ | .avgPool _ => true
  | _ => false

/-- A unary tensor evaluator acts on the real parent and returns the declared real output. -/
def ShapedUnaryRealEquation (nodes : Array Node) (node : Node)
    (dims : Nat → Nat) (v : Nat → Nat → ℝ) (id : Nat)
    (op : SomeTensor ℝ → Except String (SomeTensor ℝ)) : Prop :=
  ∀ p parent, unaryParent? node.parents = some p → nodes[p]? = some parent →
    dims p = parent.outShape.size ∧ dims id = node.outShape.size ∧
      op ⟨parent.outShape, realTensor parent.outShape (v p)⟩ =
        .ok ⟨node.outShape, realTensor node.outShape (v id)⟩

/--
Exact real equations for this family. Broadcast and reduction use the output shape actually
computed by their transfers; concat uses the checked occurrence-wise layout. No equation
mentions interval endpoints or assumes enclosure of the output.
-/
def StructuralRealNodeEquation (nodes : Array Node) (dims : Nat → Nat)
    (v : Nat → Nat → ℝ) (id : Nat) : Prop :=
  let node := nodes[id]!
  match node.kind with
  | .concat axis =>
      ∀ layout, concatNodeLayout? nodes node axis = some layout →
        ConcatRealEquation node layout dims v id
  | .transpose axis₁ axis₂ =>
      ShapedUnaryRealEquation nodes node dims v id (transposeTensor axis₁ axis₂)
  | .permute perm =>
      ShapedUnaryRealEquation nodes node dims v id
        (fun x => NN.IR.Graph.permuteSomeTensor x perm)
  | .maxPool config =>
      ShapedUnaryRealEquation nodes node dims v id (NN.IR.Graph.evalMaxPool config)
  | .avgPool config =>
      ShapedUnaryRealEquation nodes node dims v id (NN.IR.Graph.evalAvgPool config)
  | .broadcastTo source target =>
      ∀ p, unaryParent? node.parents = some p →
        dims p = source.size ∧ dims id = target.size ∧
          ∀ h : Shape.CanBroadcastTo source target, ∀ i : Fin target.size,
            v id i.val =
              tensorValues (Tensor.broadcastTo h (realTensor source (v p))) i.val
  | .reduceSum axis =>
      ∀ p parent, unaryParent? node.parents = some p → nodes[p]? = some parent →
        dims p = parent.outShape.size ∧
          dims id = (Tensor.shapeAfterSum parent.outShape axis).size ∧
            ∀ i : Fin (Tensor.shapeAfterSum parent.outShape axis).size,
              v id i.val = realAxisSum parent.outShape axis (v p) i.val
  | .reduceMean axis =>
      ∀ p parent, unaryParent? node.parents = some p → nodes[p]? = some parent →
        dims p = parent.outShape.size ∧
          dims id = (Tensor.shapeAfterSum parent.outShape axis).size ∧
            ∀ i : Fin (Tensor.shapeAfterSum parent.outShape axis).size,
              v id i.val = realAxisMean parent.outShape axis (v p) i.val
  | _ => True

private theorem shapedUnary_lookup {β γ : Type}
    {nodes : Array Node} {parents : Array Nat} {boxes : Array (Option β)}
    {k : Node → β → Option γ} {out : γ}
    (h : (do
      let p ← unaryParent? parents
      let parent ← nodes[p]?
      let input ← (boxes[p]?).join
      k parent input) = some out) :
    ∃ p parent input, unaryParent? parents = some p ∧ nodes[p]? = some parent ∧
      (boxes[p]?).join = some input ∧ k parent input = some out := by
  obtain ⟨p, hp, h⟩ := Option.bind_eq_some_iff.mp h
  obtain ⟨parent, hparent, h⟩ := Option.bind_eq_some_iff.mp h
  obtain ⟨input, hinput, h⟩ := Option.bind_eq_some_iff.mp h
  exact ⟨p, parent, input, hp, hparent, hinput, h⟩

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

private theorem shapedUnary_monotone_step
    {nodes : Array Node} {node : Node} {boxes : Array (Option (FlatBox α))}
    {dims : Nat → Nat} {v : Nat → Nat → ℝ} {id : Nat}
    {op : SomeTensor α → Except String (SomeTensor α)}
    {exactOp : SomeTensor ℝ → Except String (SomeTensor ℝ)}
    (hop : TensorTransferEncloses op exactOp)
    (heq : ShapedUnaryRealEquation nodes node dims v id exactOp)
    (hparents : ∀ p ∈ node.parents, ∀ B, (boxes[p]?).join = some B →
      RowEncloses B (dims p) (v p))
    {box : FlatBox α}
    (hstep : (do
      let p ← unaryParent? node.parents
      let parent ← nodes[p]?
      ibpMonotoneSomeTensor? parent.outShape node.outShape op (← (boxes[p]?).join)) =
        some box) :
    RowEncloses box (dims id) (v id) := by
  obtain ⟨p, parent, B, hp, hparent, hB, hstep⟩ := shapedUnary_lookup hstep
  obtain ⟨hdp, hd, hy⟩ := heq p parent hp hparent
  have hin := hparents p (mem_of_unaryParent?_eq_some hp) B hB
  rw [hdp] at hin
  rw [hd]
  exact (ibpMonotoneSomeTensor?_encloses hop hin hy hstep).congr
    (fun i => by simp only [tensorValues_fin, getScalar_flatten_realTensor])

variable [NonlinearBoundOps α] [LawfulNonlinearBoundOps α]

/--
One structural IBP step encloses its exact real value whenever the parent boxes enclose their
real values. The parent agreement interface is the same as for the basic forward step theorem.
All executable guards remain in `hstep`, including empty-axis and finite-arithmetic checks.
-/
theorem ibpStepNodeAt?_structural_encloses
    {nodes : Array Node} {ps : ParamStore α} {boxes ibp : Array (Option (FlatBox α))}
    {dims : Nat → Nat} {v : Nat → Nat → ℝ} {id : Nat}
    (hsupported : ibpStructuralSupportedNode nodes[id]! = true)
    (heq : StructuralRealNodeEquation nodes dims v id)
    (hagree : ∀ p ∈ nodes[id]!.parents, (boxes[p]?).join = ibp[p]!)
    (henc : ∀ p ∈ nodes[id]!.parents, ∀ box, ibp[p]! = some box →
      RowEncloses box (dims p) (v p))
    {box : FlatBox α} (hstep : ibpStepNodeAt? nodes ps boxes id nodes[id]! = some box) :
    RowEncloses box (dims id) (v id) := by
  have hget : ∀ p ∈ nodes[id]!.parents, ∀ B, (boxes[p]?).join = some B →
      RowEncloses B (dims p) (v p) := fun p hp B hB =>
    henc p hp B ((hagree p hp).symm.trans hB)
  have hunary {p : Nat} (hp : unaryParent? nodes[id]!.parents = some p) :=
    hget p (mem_of_unaryParent?_eq_some hp)
  unfold StructuralRealNodeEquation at heq
  rw [ibpStepNodeAt?] at hstep
  cases hk : (nodes[id]!).kind <;>
    simp only [ibpStructuralSupportedNode, hk, Bool.false_eq_true] at hsupported <;>
    simp only [hk] at heq <;>
    rw [hk] at hstep <;> dsimp only at hstep
  case concat axis =>
    exact concatNodeBoxes?_encloses heq hget hstep
  case transpose axis₁ axis₂ =>
    exact shapedUnary_monotone_step (transposeTensor_encloses axis₁ axis₂) heq hget hstep
  case permute perm =>
    exact shapedUnary_monotone_step (permuteSomeTensor_encloses perm) heq hget hstep
  case maxPool config =>
    exact shapedUnary_monotone_step (evalMaxPool_encloses config) heq hget hstep
  case avgPool config =>
    obtain ⟨p, parent, B, hp, hparent, hB, hbox⟩ :=
      shapedUnary_lookup (k := fun parent => ibpAvgPool? config parent.outShape nodes[id]!.outShape)
        hstep
    obtain ⟨hdp, hd, hy⟩ := heq p parent hp hparent
    have hin := hunary hp B hB
    rw [hdp] at hin
    rw [hd]
    exact (ibpAvgPool?_encloses hin hy hbox).congr
      (fun i => by simp only [tensorValues_fin, getScalar_flatten_realTensor])
  case broadcastTo source target =>
    obtain ⟨p, B, hp, hB, hbox⟩ :=
      unary_lookup (k := ibpBroadcastTo source target) hstep
    obtain ⟨hdp, hd, hv⟩ := heq p hp
    have hin := hunary hp B hB
    rw [hdp] at hin
    rw [hd]
    have hb := ibpBroadcastTo_canBroadcast hbox
    exact (ibpBroadcastTo_encloses hin hb hbox).congr (hv hb)
  case reduceSum axis =>
    obtain ⟨p, parent, B, hp, hparent, hB, hbox⟩ :=
      shapedUnary_lookup (k := fun parent B => ibpReduceSumAxis axis B parent.outShape) hstep
    obtain ⟨hdp, hd, hv⟩ := heq p parent hp hparent
    have hin := hunary hp B hB
    rw [hdp] at hin
    rw [hd]
    exact (ibpReduceSumAxis_encloses hin hbox).congr hv
  case reduceMean axis =>
    obtain ⟨p, parent, B, hp, hparent, hB, hbox⟩ :=
      shapedUnary_lookup (k := fun parent B => ibpReduceMeanAxis axis B parent.outShape) hstep
    obtain ⟨hdp, hd, hv⟩ := heq p parent hp hparent
    have hin := hunary hp B hB
    rw [hdp] at hin
    rw [hd]
    exact (ibpReduceMeanAxis_encloses hin hbox).congr hv

end

end NN.MLTheory.CROWN.Graph.DirectedBackward

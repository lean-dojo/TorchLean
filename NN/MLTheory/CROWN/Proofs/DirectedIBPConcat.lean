/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.DirectedIBPTensor
public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.Main.Extraction
import NN.Tensor.Internal.Laws.Sequence

/-!
# Directed enclosure for arbitrary-axis concatenation

Concatenation transports endpoint bounds through the existing layout's coordinate equivalence.
Parent occurrences remain distinct, including when a graph lists the same parent more than once.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph.CertSoundness

noncomputable section

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

/-- Exact concat coordinates in the same layout used by the executable transfer. -/
def realConcat (layout : ConcatLayout)
    (f : Fin layout.lengths.length → Nat → ℝ) (i : Nat) : ℝ :=
  getAtOrZero
    (layout.concat fun parent => Tensor.ofFn fun j => f parent j.val) [i]

private theorem rowEncloses_cast_coordinate {B : FlatBox α} {n : Nat} {f : Nat → ℝ}
    (hB : RowEncloses B n f) (hd : B.dim = n) (i : Fin n) :
    value ((hd ▸ B.lo).getScalar i) ≤ f i.val ∧
      f i.val ≤ value ((hd ▸ B.hi).getScalar i) := by
  obtain ⟨d, lo, hi⟩ := B
  dsimp only at hd
  subst d
  exact rowEncloses_iff.mp hB i

/-- Checked concatenation encloses the exact value supplied by each parent occurrence. -/
theorem concatFlatBoxes?_encloses {layout : ConcatLayout}
    {boxes : Array (FlatBox α)} {box : FlatBox α}
    {f : Fin layout.lengths.length → Nat → ℝ}
    (hbox : concatFlatBoxes? layout boxes = some box)
    (hparents : ∀ (p : Fin layout.lengths.length) (B : FlatBox α),
      boxes[p.val]? = some B → RowEncloses B (layout.parentShape p).size (f p)) :
    RowEncloses box layout.outputShape.size (realConcat layout f) := by
  unfold concatFlatBoxes? at hbox
  split at hbox
  · obtain ⟨inputs, hinputs, hbox⟩ := Option.bind_eq_some_iff.mp hbox
    obtain ⟨hd, hbox⟩ := dite_eq_some_elim hbox
    obtain rfl := Option.some.inj hbox
    rw [rowEncloses_iff]
    intro i
    simp only [realConcat, read_fin, ConcatLayout.concat, getScalar_ofFn]
    exact rowEncloses_cast_coordinate
      (hparents (layout.flatEquiv.symm i).1 _
        (Tensor.Internal.sequenceFinM_get_of_eq_some hinputs (layout.flatEquiv.symm i).1))
      (hd _) (layout.flatEquiv.symm i).2
  · contradiction

omit [BoundOps α] [LawfulBoundOps α] in
/-- The real concat equation names the source of each output scalar, independently of boxes. -/
def ConcatRealEquation (node : Node) (layout : ConcatLayout)
    (dims : Nat → Nat) (v : Nat → Nat → ℝ) (id : Nat) : Prop :=
  dims id = layout.outputShape.size ∧
    ∀ (occurrence : Fin layout.lengths.length) (parent : Nat),
      node.parents[occurrence.val]? = some parent →
        dims parent = (layout.parentShape occurrence).size ∧
          ∀ i : Fin (layout.parentShape occurrence).size,
            v id (layout.flatEquiv ⟨occurrence, i⟩).val = v parent i.val

/-- Node concat soundness uses the checked layout and the existing occurrence-wise array lemma. -/
theorem concatNodeBoxes?_encloses {nodes : Array Node}
    {boxes : Array (Option (FlatBox α))} {node : Node} {axis id : Nat}
    {dims : Nat → Nat} {v : Nat → Nat → ℝ} {box : FlatBox α}
    (heq : ∀ layout, concatNodeLayout? nodes node axis = some layout →
      ConcatRealEquation node layout dims v id)
    (hparents : ∀ p ∈ node.parents, ∀ B, (boxes[p]?).join = some B →
      RowEncloses B (dims p) (v p))
    (hbox : concatNodeBoxes? nodes boxes node axis = some box) :
    RowEncloses box (dims id) (v id) := by
  unfold concatNodeBoxes? at hbox
  obtain ⟨layout, hlayout, hbox⟩ := Option.bind_eq_some_iff.mp hbox
  obtain ⟨inputs, hinputs, hbox⟩ := Option.bind_eq_some_iff.mp hbox
  obtain ⟨hout, hcoords⟩ := heq layout hlayout
  have hentry {p : Fin layout.lengths.length} {B : FlatBox α}
      (hB : inputs[p.val]? = some B) :
      ∃ parent, node.parents[p.val]? = some parent ∧
        (boxes[parent]?).join = some B := by
    rw [array_mapM_getElem?_of_eq_some hinputs p.val] at hB
    exact Option.bind_eq_some_iff.mp hB
  have h := concatFlatBoxes?_encloses
    (f := fun p => v (node.parents[p.val]!)) hbox (by
      intro p B hB
      obtain ⟨parent, hp, hb⟩ := hentry hB
      have hd := (hcoords p parent hp).1
      simpa only [getElem!_def, hp, ← hd] using
        hparents parent (Array.mem_of_getElem? hp) B hb)
  rw [hout]
  apply h.congr
  intro i
  let source := layout.flatEquiv.symm i
  have hexists : ∃ B, inputs[source.1.val]? = some B := by
    unfold concatFlatBoxes? at hbox
    split at hbox
    · obtain ⟨entries, hentries, _⟩ := Option.bind_eq_some_iff.mp hbox
      exact ⟨entries source.1, Tensor.Internal.sequenceFinM_get_of_eq_some hentries source.1⟩
    · contradiction
  obtain ⟨B, hB⟩ := hexists
  obtain ⟨parent, hp, _⟩ := hentry hB
  have hc := (hcoords source.1 parent hp).2 source.2
  simpa only [realConcat, read_fin, ConcatLayout.concat, getScalar_ofFn,
    source, Sigma.eta, Equiv.apply_symm_apply, getElem!_def, hp] using hc

end

end NN.MLTheory.CROWN.Graph.DirectedBackward

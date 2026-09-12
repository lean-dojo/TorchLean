/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.CertificateStep
public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.Semantics

/-!
# Shared Extraction Lemmas for the Certificate Induction

The per-operator soundness lemmas all start the same way: read the parent ids off the node,
show the parent boxes and values exist because the node produced `some`, and fetch the parent
enclosure from the induction hypothesis.  This file packages the induction hypothesis as
`ParentsEnclosed` and provides the small lookup lemmas each operator case needs.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN

namespace CertSoundness

noncomputable section

/-- Induction hypothesis at node `k`: every parent whose certificate box and semantic value both
exist is enclosed.  Phrased through `getBox?`/`getVal?` so operator cases never need array sizes
or the topological order. -/
def ParentsEnclosed (nodes : Array Node) (cert : Array (Option (FlatBox ℝ)))
    (vals : Array (Option Val)) (k : Nat) : Prop :=
  ∀ p : Nat, p ∈ (nodes[k]!).parents →
    ∀ (Bp : FlatBox ℝ) (vp : Val),
      getBox? cert p = some Bp → getVal? vals p = some vp → EnclosesBox Bp vp

/-- A successful safe box lookup is an ordinary array read. -/
theorem getElem!_of_getBox?_eq_some {cert : Array (Option (FlatBox ℝ))} {p : Nat}
    {B : FlatBox ℝ} (h : getBox? cert p = some B) : cert[p]! = some B := by
  unfold getBox? at h
  split at h
  · exact h
  · exact absurd h (by simp)

/-- A successful safe value lookup is an ordinary array read. -/
theorem getElem!_of_getVal?_eq_some {vals : Array (Option Val)} {p : Nat} {v : Val}
    (h : getVal? vals p = some v) : vals[p]! = some v := by
  unfold getVal? at h
  split at h
  · exact h
  · exact absurd h (by simp)

/-- Parent enclosure for the single parent of a unary node. -/
theorem parents_enclosed_unary {nodes : Array Node} {cert : Array (Option (FlatBox ℝ))}
    {vals : Array (Option Val)} {k p1 : Nat} {B1 : FlatBox ℝ} {v1 : Val}
    (hpe : ParentsEnclosed nodes cert vals k)
    (hparents : NN.IR.unaryParent? (nodes[k]!).parents = some p1)
    (hgb : getBox? cert p1 = some B1) (hgv : getVal? vals p1 = some v1) :
    EnclosesBox B1 v1 :=
  hpe p1 (NN.IR.mem_of_unaryParent?_eq_some hparents) B1 v1 hgb hgv

/-- Parent enclosure for both parents of a binary node. -/
theorem parents_enclosed_binary {nodes : Array Node} {cert : Array (Option (FlatBox ℝ))}
    {vals : Array (Option Val)} {k p1 p2 : Nat} {B1 B2 : FlatBox ℝ} {v1 v2 : Val}
    (hpe : ParentsEnclosed nodes cert vals k)
    (hparents : NN.IR.binaryParents? (nodes[k]!).parents = some (p1, p2))
    (hgb1 : getBox? cert p1 = some B1) (hgb2 : getBox? cert p2 = some B2)
    (hgv1 : getVal? vals p1 = some v1) (hgv2 : getVal? vals p2 = some v2) :
    EnclosesBox B1 v1 ∧ EnclosesBox B2 v2 :=
  ⟨hpe p1 (NN.IR.fst_mem_of_binaryParents?_eq_some hparents) B1 v1 hgb1 hgv1,
    hpe p2 (NN.IR.snd_mem_of_binaryParents?_eq_some hparents) B2 v2 hgb2 hgv2⟩

/-- A dependent `if` that produced `some` took its positive branch. -/
theorem dite_eq_some_elim {p : Prop} [Decidable p] {β : Type} {f : p → Option β} {b : β}
    (h : (if hp : p then f hp else none) = some b) : ∃ hp : p, f hp = some b := by
  split at h
  · exact ⟨_, h⟩
  · cases h

end

end CertSoundness

end NN.MLTheory.CROWN.Graph

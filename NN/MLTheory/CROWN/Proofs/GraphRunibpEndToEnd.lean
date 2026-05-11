/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness

/-!
# End-to-end IBP soundness (graph dialect, over `ℝ`)

`NN.MLTheory.CROWN.Proofs.GraphCertSoundness` proves:

> If a per-node IBP certificate is locally consistent (`CertLocalOK`) and the value semantics is
> locally consistent (`SemLocalOK`), then each certified box encloses the corresponding value.

This file supplies a concrete, total evaluator and a concrete, total IBP propagation and proves
they satisfy the local-consistency predicates under the `TopoSorted` assumption (parents have
smaller ids). Combining these results yields an end-to-end theorem.
-/

@[expose] public section


namespace NN.MLTheory.CROWN.Graph

open _root_.Spec
open _root_.Spec.Tensor

namespace CertSoundness

noncomputable section

/-!
## Array helper lemmas (`getElem!` after `setIfInBounds`)
-/

private lemma getElem!_setIfInBounds_ne {α : Type} [Inhabited α]
    (xs : Array α) (i : Nat) (a : α) (j : Nat)
    (hj : j < xs.size) (hij : i ≠ j) :
    (xs.setIfInBounds i a)[j]! = xs[j]! := by
  have hj' : j < (xs.setIfInBounds i a).size := by simpa using hj
  calc
    (xs.setIfInBounds i a)[j]! = (xs.setIfInBounds i a)[j]'hj' := by
      simpa using (getElem!_pos (c := xs.setIfInBounds i a) (i := j) hj')
    _ = xs[j]'hj := by
      simpa using (Array.getElem_setIfInBounds_ne (xs := xs) (i := i) (a := a) (j := j) hj hij)
    _ = xs[j]! := by
      simpa using (getElem!_pos (c := xs) (i := j) hj).symm

private lemma getElem!_setIfInBounds_self {α : Type} [Inhabited α]
    (xs : Array α) (i : Nat) (a : α) (hi : i < xs.size) :
    (xs.setIfInBounds i a)[i]! = a := by
  have hi' : i < (xs.setIfInBounds i a).size := by simpa using hi
  calc
    (xs.setIfInBounds i a)[i]! = (xs.setIfInBounds i a)[i]'hi' := by
      simpa using (getElem!_pos (c := xs.setIfInBounds i a) (i := i) hi')
    _ = a := by
      simp [Array.getElem_setIfInBounds_self]

/-!
## Total evaluators (Nat-recursive, prefix semantics)

We define fold-by-id evaluators using `Nat.rec` rather than `List.foldl` to keep proofs small and
stable. The resulting arrays coincide with the intended “evaluate in node-id order” semantics.
-/

def evalGraphPrefix (g : Graph) (ps : ParamStore ℝ) (inputs : Std.HashMap Nat Val) :
    Nat → Array (Option Val)
  | 0 => Array.replicate g.nodes.size none
  | n + 1 =>
      let acc := evalGraphPrefix g ps inputs n
      acc.set! n (evalNode? g.nodes ps inputs acc n)

/-- Evaluate all nodes of `g` in id order, returning the final value array. -/
def evalGraphRec (g : Graph) (ps : ParamStore ℝ) (inputs : Std.HashMap Nat Val) : Array (Option Val)
  :=
  evalGraphPrefix g ps inputs g.nodes.size

/-- Prefix evaluator for the safe IBP checker step (`certStepNode?`). -/
def runIBPPrefix (g : Graph) (ps : ParamStore ℝ) : Nat → Array (Option (FlatBox ℝ))
  | 0 => Array.replicate g.nodes.size none
  | n + 1 =>
      let acc := runIBPPrefix g ps n
      acc.set! n (certStepNode? g.nodes ps acc n)

/-- Run the safe IBP checker step across the full graph, producing a per-node certificate array. -/
def runIBP? (g : Graph) (ps : ParamStore ℝ) : Array (Option (FlatBox ℝ)) :=
  runIBPPrefix g ps g.nodes.size

/-! Prefix size facts. -/

private lemma evalGraphPrefix_size (g : Graph) (ps : ParamStore ℝ) (inputs : Std.HashMap Nat Val) :
    ∀ n, (evalGraphPrefix g ps inputs n).size = g.nodes.size := by
  intro n; induction n with
  | zero => simp [evalGraphPrefix]
  | succ n ih =>
      simp [evalGraphPrefix, ih, Array.set!_eq_setIfInBounds]

private lemma runIBPPrefix_size (g : Graph) (ps : ParamStore ℝ) :
    ∀ n, (runIBPPrefix g ps n).size = g.nodes.size := by
  intro n; induction n with
  | zero => simp [runIBPPrefix]
  | succ n ih =>
      simp [runIBPPrefix, ih, Array.set!_eq_setIfInBounds]

/-! Prefix stability: later writes do not change earlier entries. -/

private lemma evalGraphPrefix_succ_get_of_lt
    (g : Graph) (ps : ParamStore ℝ) (inputs : Std.HashMap Nat Val)
    {n i : Nat} (hi : i < n) :
    (evalGraphPrefix g ps inputs (n + 1))[i]! = (evalGraphPrefix g ps inputs n)[i]! := by
  classical
  let acc := evalGraphPrefix g ps inputs n
  have haccSz : acc.size = g.nodes.size := by
    simpa [acc] using evalGraphPrefix_size (g := g) (ps := ps) (inputs := inputs) n
  by_cases hn : n < g.nodes.size
  · have hiAcc : i < acc.size := by
      -- `i < n < nodes.size = acc.size`
      have : i < g.nodes.size := lt_of_lt_of_le hi (Nat.le_of_lt hn)
      simpa [haccSz] using this
    have hne : n ≠ i := Nat.ne_of_gt hi
    have hstep :
        (acc.set! n (evalNode? g.nodes ps inputs acc n))[i]! = acc[i]! := by
      simpa [Array.set!_eq_setIfInBounds] using
        (getElem!_setIfInBounds_ne (xs := acc) (i := n)
          (a := evalNode? g.nodes ps inputs acc n) (j := i) hiAcc hne)
    simpa [evalGraphPrefix, acc, hn] using hstep
  · have hnle : g.nodes.size ≤ n := Nat.le_of_not_gt hn
    have haccLe : acc.size ≤ n := by simpa [haccSz] using hnle
    -- out-of-bounds write is a no-op
    simp [evalGraphPrefix, acc, Array.set!_eq_setIfInBounds, Array.setIfInBounds_eq_of_size_le
      haccLe]

private lemma runIBPPrefix_succ_get_of_lt
    (g : Graph) (ps : ParamStore ℝ)
    {n i : Nat} (hi : i < n) :
    (runIBPPrefix g ps (n + 1))[i]! = (runIBPPrefix g ps n)[i]! := by
  classical
  let acc := runIBPPrefix g ps n
  have haccSz : acc.size = g.nodes.size := by
    simpa [acc] using runIBPPrefix_size (g := g) (ps := ps) n
  by_cases hn : n < g.nodes.size
  · have hiAcc : i < acc.size := by
      have : i < g.nodes.size := lt_of_lt_of_le hi (Nat.le_of_lt hn)
      simpa [haccSz] using this
    have hne : n ≠ i := Nat.ne_of_gt hi
    have hstep :
        (acc.set! n (certStepNode? g.nodes ps acc n))[i]! = acc[i]! := by
      simpa [Array.set!_eq_setIfInBounds] using
        (getElem!_setIfInBounds_ne (xs := acc) (i := n)
          (a := certStepNode? g.nodes ps acc n) (j := i) hiAcc hne)
    simpa [runIBPPrefix, acc, hn] using hstep
  · have hnle : g.nodes.size ≤ n := Nat.le_of_not_gt hn
    have haccLe : acc.size ≤ n := by simpa [haccSz] using hnle
    simp [runIBPPrefix, acc, Array.set!_eq_setIfInBounds, Array.setIfInBounds_eq_of_size_le haccLe]

private lemma evalGraphPrefix_get_of_lt
    (g : Graph) (ps : ParamStore ℝ) (inputs : Std.HashMap Nat Val)
    {k n i : Nat} (hkn : k ≤ n) (hi : i < k) :
    (evalGraphPrefix g ps inputs n)[i]! = (evalGraphPrefix g ps inputs k)[i]! := by
  induction n generalizing k with
  | zero =>
      have hk : k = 0 := Nat.eq_zero_of_le_zero hkn
      subst hk
      cases hi
  | succ n ih =>
      rcases Nat.lt_or_eq_of_le hkn with hklt | rfl
      · have hkn' : k ≤ n := Nat.le_of_lt_succ hklt
        have hin : i < n := lt_of_lt_of_le hi hkn'
        exact (evalGraphPrefix_succ_get_of_lt (g := g) (ps := ps) (inputs := inputs) (n := n) (i :=
          i) hin) ▸
          ih (hkn := hkn') hi
      · rfl

private lemma runIBPPrefix_get_of_lt
    (g : Graph) (ps : ParamStore ℝ)
    {k n i : Nat} (hkn : k ≤ n) (hi : i < k) :
    (runIBPPrefix g ps n)[i]! = (runIBPPrefix g ps k)[i]! := by
  induction n generalizing k with
  | zero =>
      have hk : k = 0 := Nat.eq_zero_of_le_zero hkn
      subst hk
      cases hi
  | succ n ih =>
      rcases Nat.lt_or_eq_of_le hkn with hklt | rfl
      · have hkn' : k ≤ n := Nat.le_of_lt_succ hklt
        have hin : i < n := lt_of_lt_of_le hi hkn'
        exact (runIBPPrefix_succ_get_of_lt (g := g) (ps := ps) (n := n) (i := i) hin) ▸
          ih (hkn := hkn') hi
      · rfl

/-!
## Congruence of the step functions under `TopoSorted`

If two arrays agree on all parent ids of node `id`, then the step function at `id` evaluates to the
same result.
-/

private lemma evalNode?_congr_of_parents
    (nodes : Array Node) (ps : ParamStore ℝ) (inputs : Std.HashMap Nat Val)
    (vals₁ vals₂ : Array (Option Val))
    {id : Nat} (hid : id < nodes.size)
    (hsize₁ : vals₁.size = nodes.size)
    (hsize₂ : vals₂.size = nodes.size)
    (hsupp : match (nodes[id]!).kind with
      | .input | .const _ | .detach
      | .add | .sub | .mul_elem | .relu
      | .linear | .matmul
      | .tanh | .sigmoid | .sin | .cos => True
      | _ => False)
    (hparsLt : ∀ p, p ∈ (nodes[id]!).parents → p < id)
    (hpar : ∀ p, p ∈ (nodes[id]!).parents → vals₁[p]! = vals₂[p]!) :
    evalNode? nodes ps inputs vals₁ id = evalNode? nodes ps inputs vals₂ id := by
  classical
  cases hk : (nodes[id]!).kind with
  | input =>
      simp [evalNode?, hk]
  | const valueShape =>
      simp [evalNode?, hk]
  | detach =>
      cases hp : (nodes[id]!).parents with
      | nil => simp [evalNode?, hk, hp]
      | cons p rest =>
          have hpLt : p < id := hparsLt p (by simp [hp])
          have hpNodes : p < nodes.size := lt_trans hpLt hid
          have hp1 : p < vals₁.size := by simpa [hsize₁] using hpNodes
          have hp2 : p < vals₂.size := by simpa [hsize₂] using hpNodes
          have hpEq : vals₁[p]! = vals₂[p]! := hpar p (by simp [hp])
          have hget : getVal? vals₁ p = getVal? vals₂ p := by
            simpa [getVal?, hp1, hp2] using hpEq
          simpa [evalNode?, hk, hp] using hget
  | add =>
      cases hp : (nodes[id]!).parents with
      | nil => simp [evalNode?, hk, hp]
      | cons p1 rest =>
        cases rest with
        | nil => simp [evalNode?, hk, hp]
        | cons p2 tail =>
            have hp1Lt : p1 < id := hparsLt p1 (by simp [hp])
            have hp2Lt : p2 < id := hparsLt p2 (by simp [hp])
            have hp1Nodes : p1 < nodes.size := lt_trans hp1Lt hid
            have hp2Nodes : p2 < nodes.size := lt_trans hp2Lt hid
            have hp1a : p1 < vals₁.size := by simpa [hsize₁] using hp1Nodes
            have hp1b : p1 < vals₂.size := by simpa [hsize₂] using hp1Nodes
            have hp2a : p2 < vals₁.size := by simpa [hsize₁] using hp2Nodes
            have hp2b : p2 < vals₂.size := by simpa [hsize₂] using hp2Nodes
            have heq1 : vals₁[p1]! = vals₂[p1]! := hpar p1 (by simp [hp])
            have heq2 : vals₁[p2]! = vals₂[p2]! := hpar p2 (by simp [hp])
            have hget1 : getVal? vals₁ p1 = getVal? vals₂ p1 := by
              simpa [getVal?, hp1a, hp1b] using heq1
            have hget2 : getVal? vals₁ p2 = getVal? vals₂ p2 := by
              simpa [getVal?, hp2a, hp2b] using heq2
            simp [evalNode?, hk, hp, hget1, hget2]
  | sub =>
      cases hp : (nodes[id]!).parents with
      | nil => simp [evalNode?, hk, hp]
      | cons p1 rest =>
        cases rest with
        | nil => simp [evalNode?, hk, hp]
        | cons p2 tail =>
            have hp1Lt : p1 < id := hparsLt p1 (by simp [hp])
            have hp2Lt : p2 < id := hparsLt p2 (by simp [hp])
            have hp1Nodes : p1 < nodes.size := lt_trans hp1Lt hid
            have hp2Nodes : p2 < nodes.size := lt_trans hp2Lt hid
            have hp1a : p1 < vals₁.size := by simpa [hsize₁] using hp1Nodes
            have hp1b : p1 < vals₂.size := by simpa [hsize₂] using hp1Nodes
            have hp2a : p2 < vals₁.size := by simpa [hsize₁] using hp2Nodes
            have hp2b : p2 < vals₂.size := by simpa [hsize₂] using hp2Nodes
            have heq1 : vals₁[p1]! = vals₂[p1]! := hpar p1 (by simp [hp])
            have heq2 : vals₁[p2]! = vals₂[p2]! := hpar p2 (by simp [hp])
            have hget1 : getVal? vals₁ p1 = getVal? vals₂ p1 := by
              simpa [getVal?, hp1a, hp1b] using heq1
            have hget2 : getVal? vals₁ p2 = getVal? vals₂ p2 := by
              simpa [getVal?, hp2a, hp2b] using heq2
            simp [evalNode?, hk, hp, hget1, hget2]
  | mul_elem =>
      cases hp : (nodes[id]!).parents with
      | nil => simp [evalNode?, hk, hp]
      | cons p1 rest =>
        cases rest with
        | nil => simp [evalNode?, hk, hp]
        | cons p2 tail =>
            have hp1Lt : p1 < id := hparsLt p1 (by simp [hp])
            have hp2Lt : p2 < id := hparsLt p2 (by simp [hp])
            have hp1Nodes : p1 < nodes.size := lt_trans hp1Lt hid
            have hp2Nodes : p2 < nodes.size := lt_trans hp2Lt hid
            have hp1a : p1 < vals₁.size := by simpa [hsize₁] using hp1Nodes
            have hp1b : p1 < vals₂.size := by simpa [hsize₂] using hp1Nodes
            have hp2a : p2 < vals₁.size := by simpa [hsize₁] using hp2Nodes
            have hp2b : p2 < vals₂.size := by simpa [hsize₂] using hp2Nodes
            have heq1 : vals₁[p1]! = vals₂[p1]! := hpar p1 (by simp [hp])
            have heq2 : vals₁[p2]! = vals₂[p2]! := hpar p2 (by simp [hp])
            have hget1 : getVal? vals₁ p1 = getVal? vals₂ p1 := by
              simpa [getVal?, hp1a, hp1b] using heq1
            have hget2 : getVal? vals₁ p2 = getVal? vals₂ p2 := by
              simpa [getVal?, hp2a, hp2b] using heq2
            simp [evalNode?, hk, hp, hget1, hget2]
  | relu =>
      cases hp : (nodes[id]!).parents with
      | nil => simp [evalNode?, hk, hp]
      | cons p rest =>
          have hpLt : p < id := hparsLt p (by simp [hp])
          have hpNodes : p < nodes.size := lt_trans hpLt hid
          have hp1 : p < vals₁.size := by simpa [hsize₁] using hpNodes
          have hp2 : p < vals₂.size := by simpa [hsize₂] using hpNodes
          have hpEq : vals₁[p]! = vals₂[p]! := hpar p (by simp [hp])
          have hget : getVal? vals₁ p = getVal? vals₂ p := by
            simpa [getVal?, hp1, hp2] using hpEq
          simp [evalNode?, hk, hp, hget]
  | tanh =>
      cases hp : (nodes[id]!).parents with
      | nil => simp [evalNode?, hk, hp]
      | cons p rest =>
          have hpLt : p < id := hparsLt p (by simp [hp])
          have hpNodes : p < nodes.size := lt_trans hpLt hid
          have hp1 : p < vals₁.size := by simpa [hsize₁] using hpNodes
          have hp2 : p < vals₂.size := by simpa [hsize₂] using hpNodes
          have hpEq : vals₁[p]! = vals₂[p]! := hpar p (by simp [hp])
          have hget : getVal? vals₁ p = getVal? vals₂ p := by
            simpa [getVal?, hp1, hp2] using hpEq
          simp [evalNode?, hk, hp, hget]
  | sigmoid =>
      cases hp : (nodes[id]!).parents with
      | nil => simp [evalNode?, hk, hp]
      | cons p rest =>
          have hpLt : p < id := hparsLt p (by simp [hp])
          have hpNodes : p < nodes.size := lt_trans hpLt hid
          have hp1 : p < vals₁.size := by simpa [hsize₁] using hpNodes
          have hp2 : p < vals₂.size := by simpa [hsize₂] using hpNodes
          have hpEq : vals₁[p]! = vals₂[p]! := hpar p (by simp [hp])
          have hget : getVal? vals₁ p = getVal? vals₂ p := by
            simpa [getVal?, hp1, hp2] using hpEq
          simp [evalNode?, hk, hp, hget]
  | sin =>
      cases hp : (nodes[id]!).parents with
      | nil => simp [evalNode?, hk, hp]
      | cons p rest =>
          have hpLt : p < id := hparsLt p (by simp [hp])
          have hpNodes : p < nodes.size := lt_trans hpLt hid
          have hp1 : p < vals₁.size := by simpa [hsize₁] using hpNodes
          have hp2 : p < vals₂.size := by simpa [hsize₂] using hpNodes
          have hpEq : vals₁[p]! = vals₂[p]! := hpar p (by simp [hp])
          have hget : getVal? vals₁ p = getVal? vals₂ p := by
            simpa [getVal?, hp1, hp2] using hpEq
          simp [evalNode?, hk, hp, hget]
  | cos =>
      cases hp : (nodes[id]!).parents with
      | nil => simp [evalNode?, hk, hp]
      | cons p rest =>
          have hpLt : p < id := hparsLt p (by simp [hp])
          have hpNodes : p < nodes.size := lt_trans hpLt hid
          have hp1 : p < vals₁.size := by simpa [hsize₁] using hpNodes
          have hp2 : p < vals₂.size := by simpa [hsize₂] using hpNodes
          have hpEq : vals₁[p]! = vals₂[p]! := hpar p (by simp [hp])
          have hget : getVal? vals₁ p = getVal? vals₂ p := by
            simpa [getVal?, hp1, hp2] using hpEq
          simp [evalNode?, hk, hp, hget]
  | linear =>
      cases hp : (nodes[id]!).parents with
      | nil => simp [evalNode?, hk, hp]
      | cons p rest =>
          have hpLt : p < id := hparsLt p (by simp [hp])
          have hpNodes : p < nodes.size := lt_trans hpLt hid
          have hp1 : p < vals₁.size := by simpa [hsize₁] using hpNodes
          have hp2 : p < vals₂.size := by simpa [hsize₂] using hpNodes
          have hpEq : vals₁[p]! = vals₂[p]! := hpar p (by simp [hp])
          have hget : getVal? vals₁ p = getVal? vals₂ p := by
            simpa [getVal?, hp1, hp2] using hpEq
          simp [evalNode?, hk, hp, hget]
  | matmul =>
      cases hp : (nodes[id]!).parents with
      | nil => simp [evalNode?, hk, hp]
      | cons p rest =>
          have hpLt : p < id := hparsLt p (by simp [hp])
          have hpNodes : p < nodes.size := lt_trans hpLt hid
          have hp1 : p < vals₁.size := by simpa [hsize₁] using hpNodes
          have hp2 : p < vals₂.size := by simpa [hsize₂] using hpNodes
          have hpEq : vals₁[p]! = vals₂[p]! := hpar p (by simp [hp])
          have hget : getVal? vals₁ p = getVal? vals₂ p := by
            simpa [getVal?, hp1, hp2] using hpEq
          simp [evalNode?, hk, hp, hget]
  | _ =>
      have : False := by
        simp [hk] at hsupp
      exact False.elim this

private lemma certStepNode?_congr_of_parents
    (nodes : Array Node) (ps : ParamStore ℝ)
    (cert₁ cert₂ : Array (Option (FlatBox ℝ)))
    {id : Nat} (hid : id < nodes.size)
    (hsize₁ : cert₁.size = nodes.size)
    (hsize₂ : cert₂.size = nodes.size)
    (hparsLt : ∀ p, p ∈ (nodes[id]!).parents → p < id)
    (hpar : ∀ p, p ∈ (nodes[id]!).parents → cert₁[p]! = cert₂[p]!) :
    certStepNode? nodes ps cert₁ id = certStepNode? nodes ps cert₂ id := by
  classical
  -- local helper for parent boxes
  have hboxOfParent (p : Nat) (hp : p ∈ (nodes[id]!).parents) : getBox? cert₁ p = getBox? cert₂ p :=
    by
    have hpLt : p < id := hparsLt p hp
    have hpNodes : p < nodes.size := lt_trans hpLt hid
    have hp1 : p < cert₁.size := by simpa [hsize₁] using hpNodes
    have hp2 : p < cert₂.size := by simpa [hsize₂] using hpNodes
    have hpEq : cert₁[p]! = cert₂[p]! := hpar p hp
    simpa [getBox?, hp1, hp2] using hpEq
  cases hk : (nodes[id]!).kind with
  | input =>
      simp [certStepNode?, hk]
  | const valueShape =>
      simp [certStepNode?, hk]
  | detach =>
      cases hp : (nodes[id]!).parents with
      | nil => simp [certStepNode?, hk, hp]
      | cons p rest =>
          have hbox : getBox? cert₁ p = getBox? cert₂ p := hboxOfParent p (by simp [hp])
          simpa [certStepNode?, hk, hp] using hbox
  | add =>
      cases hp : (nodes[id]!).parents with
      | nil => simp [certStepNode?, hk, hp]
      | cons p1 rest =>
        cases rest with
        | nil => simp [certStepNode?, hk, hp]
        | cons p2 tail =>
            have hbox1 : getBox? cert₁ p1 = getBox? cert₂ p1 := hboxOfParent p1 (by simp [hp])
            have hbox2 : getBox? cert₁ p2 = getBox? cert₂ p2 := hboxOfParent p2 (by simp [hp])
            simp [certStepNode?, hk, hp, hbox1, hbox2]
  | sub =>
      cases hp : (nodes[id]!).parents with
      | nil => simp [certStepNode?, hk, hp]
      | cons p1 rest =>
        cases rest with
        | nil => simp [certStepNode?, hk, hp]
        | cons p2 tail =>
            have hbox1 : getBox? cert₁ p1 = getBox? cert₂ p1 := hboxOfParent p1 (by simp [hp])
            have hbox2 : getBox? cert₁ p2 = getBox? cert₂ p2 := hboxOfParent p2 (by simp [hp])
            simp [certStepNode?, hk, hp, hbox1, hbox2]
  | mul_elem =>
      cases hp : (nodes[id]!).parents with
      | nil => simp [certStepNode?, hk, hp]
      | cons p1 rest =>
        cases rest with
        | nil => simp [certStepNode?, hk, hp]
        | cons p2 tail =>
            have hbox1 : getBox? cert₁ p1 = getBox? cert₂ p1 := hboxOfParent p1 (by simp [hp])
            have hbox2 : getBox? cert₁ p2 = getBox? cert₂ p2 := hboxOfParent p2 (by simp [hp])
            simp [certStepNode?, hk, hp, hbox1, hbox2]
  | relu =>
      cases hp : (nodes[id]!).parents with
      | nil => simp [certStepNode?, hk, hp]
      | cons p rest =>
          have hbox : getBox? cert₁ p = getBox? cert₂ p := hboxOfParent p (by simp [hp])
          simp [certStepNode?, hk, hp, hbox]
  | tanh =>
      cases hp : (nodes[id]!).parents with
      | nil => simp [certStepNode?, hk, hp]
      | cons p rest =>
          have hbox : getBox? cert₁ p = getBox? cert₂ p := hboxOfParent p (by simp [hp])
          simp [certStepNode?, hk, hp, hbox]
  | sigmoid =>
      cases hp : (nodes[id]!).parents with
      | nil => simp [certStepNode?, hk, hp]
      | cons p rest =>
          have hbox : getBox? cert₁ p = getBox? cert₂ p := hboxOfParent p (by simp [hp])
          simp [certStepNode?, hk, hp, hbox]
  | sin =>
      cases hp : (nodes[id]!).parents with
      | nil => simp [certStepNode?, hk, hp]
      | cons p rest =>
          have hbox : getBox? cert₁ p = getBox? cert₂ p := hboxOfParent p (by simp [hp])
          simp [certStepNode?, hk, hp, hbox]
  | cos =>
      cases hp : (nodes[id]!).parents with
      | nil => simp [certStepNode?, hk, hp]
      | cons p rest =>
          have hbox : getBox? cert₁ p = getBox? cert₂ p := hboxOfParent p (by simp [hp])
          simp [certStepNode?, hk, hp, hbox]
  | linear =>
      cases hp : (nodes[id]!).parents with
      | nil => simp [certStepNode?, hk, hp]
      | cons p rest =>
          have hbox : getBox? cert₁ p = getBox? cert₂ p := hboxOfParent p (by simp [hp])
          simp [certStepNode?, hk, hp, hbox]
  | matmul =>
      cases hp : (nodes[id]!).parents with
      | nil => simp [certStepNode?, hk, hp]
      | cons p rest =>
          have hbox : getBox? cert₁ p = getBox? cert₂ p := hboxOfParent p (by simp [hp])
          simp [certStepNode?, hk, hp, hbox]
  | _ =>
      simp [certStepNode?, hk]

/-!
## Local consistency of the total evaluators
-/

theorem evalGraphRec_SemLocalOK (g : Graph) (ps : ParamStore ℝ) (inputs : Std.HashMap Nat Val)
    (htopo : TopoSorted g) (hsupp : Supported g) :
    SemLocalOK g ps inputs (evalGraphRec g ps inputs) := by
  classical
  refine ⟨by simpa [evalGraphRec] using (evalGraphPrefix_size (g := g) (ps := ps) (inputs := inputs)
    g.nodes.size), ?_⟩
  intro id hid
  -- `evalGraphPrefix (id+1)` sets index `id`.
  have hidSz : id < (evalGraphPrefix g ps inputs id).size := by
    -- size is always `nodes.size`
    simpa [evalGraphPrefix_size (g := g) (ps := ps) (inputs := inputs)] using hid
  have hstep :
      (evalGraphPrefix g ps inputs (id + 1))[id]!
        = evalNode? g.nodes ps inputs (evalGraphPrefix g ps inputs id) id := by
    -- unfold the `id+1` step and compute the `id` lookup
    simp [evalGraphPrefix, Array.set!_eq_setIfInBounds,
      getElem!_setIfInBounds_self (xs := evalGraphPrefix g ps inputs id) (i := id)
        (a := evalNode? g.nodes ps inputs (evalGraphPrefix g ps inputs id) id) hidSz]
  have hstable :
      (evalGraphRec g ps inputs)[id]!
        = (evalGraphPrefix g ps inputs (id + 1))[id]! := by
    have : id + 1 ≤ g.nodes.size := Nat.succ_le_of_lt hid
    -- stability lemma gives `prefix size` equals `prefix (id+1)` at index `id`
    have := evalGraphPrefix_get_of_lt (g := g) (ps := ps) (inputs := inputs)
      (k := id + 1) (n := g.nodes.size) (i := id)
      (hkn := this) (hi := Nat.lt_succ_self id)
    simpa [evalGraphRec] using this
  have hset :
      (evalGraphRec g ps inputs)[id]!
        = evalNode? g.nodes ps inputs (evalGraphPrefix g ps inputs id) id := by
    exact Eq.trans hstable hstep
  have hpar :
      ∀ p, p ∈ (g.nodes[id]!).parents →
        (evalGraphPrefix g ps inputs id)[p]!
          = (evalGraphRec g ps inputs)[p]! := by
    intro p hp
    have hpLt : p < id := htopo id hid p hp
    have hpLe : id ≤ g.nodes.size := Nat.le_of_lt hid
    have := evalGraphPrefix_get_of_lt (g := g) (ps := ps) (inputs := inputs)
      (k := id) (n := g.nodes.size) (i := p)
      (hkn := hpLe) (hi := hpLt)
    simpa [evalGraphRec] using this.symm
  have hsize₁ : (evalGraphPrefix g ps inputs id).size = g.nodes.size :=
    evalGraphPrefix_size (g := g) (ps := ps) (inputs := inputs) id
  have hsize₂ : (evalGraphRec g ps inputs).size = g.nodes.size := by
    simpa [evalGraphRec] using evalGraphPrefix_size (g := g) (ps := ps) (inputs := inputs)
      g.nodes.size
  have hnode :
      evalNode? g.nodes ps inputs (evalGraphPrefix g ps inputs id) id
        = evalNode? g.nodes ps inputs (evalGraphRec g ps inputs) id := by
    -- use congruence: parents (< id) agree between prefix and final
    have hparsLt : ∀ p, p ∈ (g.nodes[id]!).parents → p < id := by
      intro p hp; exact htopo id hid p hp
    -- `Supported` ensures we only hit supported constructors at this node id
    have hs : match (g.nodes[id]!).kind with
        | .input | .const _ | .detach
        | .add | .sub | .mul_elem | .relu
        | .linear | .matmul
        | .tanh | .sigmoid | .sin | .cos => True
        | _ => False := hsupp id hid
    simpa using (evalNode?_congr_of_parents (nodes := g.nodes) (ps := ps) (inputs := inputs)
      (vals₁ := evalGraphPrefix g ps inputs id)
      (vals₂ := evalGraphRec g ps inputs)
      (hid := hid) (hsize₁ := hsize₁) (hsize₂ := hsize₂) (hsupp := hs) (hparsLt := hparsLt) hpar)
  -- finish by rewriting the full-array step to the prefix step
  calc
    (evalGraphRec g ps inputs)[id]! = evalNode? g.nodes ps inputs (evalGraphPrefix g ps inputs id)
      id := hset
    _ = evalNode? g.nodes ps inputs (evalGraphRec g ps inputs) id := hnode

/-- Under topological order, the certificate produced by `runIBP?` satisfies `CertLocalOK`. -/
theorem runIBP?_CertLocalOK (g : Graph) (ps : ParamStore ℝ)
    (htopo : TopoSorted g) :
    CertLocalOK (g := g) (ps := ps) (runIBP? g ps) := by
  classical
  refine ⟨by simpa [runIBP?] using (runIBPPrefix_size (g := g) (ps := ps) g.nodes.size), ?_⟩
  intro id hid
  have hidSz : id < (runIBPPrefix g ps id).size := by
    simpa [runIBPPrefix_size (g := g) (ps := ps)] using hid
  have hstep :
      (runIBPPrefix g ps (id + 1))[id]!
        = certStepNode? g.nodes ps (runIBPPrefix g ps id) id := by
    simp [runIBPPrefix, Array.set!_eq_setIfInBounds,
      getElem!_setIfInBounds_self (xs := runIBPPrefix g ps id) (i := id)
        (a := certStepNode? g.nodes ps (runIBPPrefix g ps id) id) hidSz]
  have hstable :
      (runIBP? g ps)[id]!
        = (runIBPPrefix g ps (id + 1))[id]! := by
    have : id + 1 ≤ g.nodes.size := Nat.succ_le_of_lt hid
    have := runIBPPrefix_get_of_lt (g := g) (ps := ps)
      (k := id + 1) (n := g.nodes.size) (i := id)
      (hkn := this) (hi := Nat.lt_succ_self id)
    simpa [runIBP?] using this
  have hset :
      (runIBP? g ps)[id]!
        = certStepNode? g.nodes ps (runIBPPrefix g ps id) id := by
    exact Eq.trans hstable hstep
  have hpar :
      ∀ p, p ∈ (g.nodes[id]!).parents →
        (runIBPPrefix g ps id)[p]!
          = (runIBP? g ps)[p]! := by
    intro p hp
    have hpLt : p < id := htopo id hid p hp
    have hpLe : id ≤ g.nodes.size := Nat.le_of_lt hid
    have := runIBPPrefix_get_of_lt (g := g) (ps := ps)
      (k := id) (n := g.nodes.size) (i := p)
      (hkn := hpLe) (hi := hpLt)
    simpa [runIBP?] using this.symm
  have hsize₁ : (runIBPPrefix g ps id).size = g.nodes.size :=
    runIBPPrefix_size (g := g) (ps := ps) id
  have hsize₂ : (runIBP? g ps).size = g.nodes.size := by
    simpa [runIBP?] using runIBPPrefix_size (g := g) (ps := ps) g.nodes.size
  have hparsLt : ∀ p, p ∈ (g.nodes[id]!).parents → p < id := by
    intro p hp; exact htopo id hid p hp
  have hnode :
      certStepNode? g.nodes ps (runIBPPrefix g ps id) id
        = certStepNode? g.nodes ps (runIBP? g ps) id := by
    simpa using (certStepNode?_congr_of_parents (nodes := g.nodes) (ps := ps)
      (cert₁ := runIBPPrefix g ps id)
      (cert₂ := runIBP? g ps)
      (hid := hid) (hsize₁ := hsize₁) (hsize₂ := hsize₂) (hparsLt := hparsLt) hpar)
  simp [hset, hnode]

/-!
## End-to-end theorem
-/

theorem runIBP?_encloses_evalGraphRec
    (g : Graph) (ps : ParamStore ℝ)
    (inputs : Std.HashMap Nat Val)
    (htopo : TopoSorted g)
    (hsupp : Supported g)
    (hinputs : InputsEnclosed g ps inputs) :
    ∀ id : Nat, id < g.nodes.size →
      match (runIBP? g ps)[id]!, (evalGraphRec g ps inputs)[id]! with
      | some B, some v => EnclosesBox B v
      | _, _ => True := by
  have hcert : CertLocalOK (g := g) (ps := ps) (runIBP? g ps) :=
    runIBP?_CertLocalOK (g := g) (ps := ps) htopo
  have hsem : SemLocalOK (g := g) (ps := ps) (inputs := inputs) (evalGraphRec g ps inputs) :=
    evalGraphRec_SemLocalOK (g := g) (ps := ps) (inputs := inputs) htopo hsupp
  exact cert_encloses_semantics
    (g := g) (ps := ps)
    (cert := runIBP? g ps)
    (inputs := inputs)
    (vals := evalGraphRec g ps inputs)
    (htopo := htopo)
    (hsupp := hsupp)
    (hcert := hcert)
    (hsem := hsem)
    (hinputs := hinputs)

end

end CertSoundness

end NN.MLTheory.CROWN.Graph

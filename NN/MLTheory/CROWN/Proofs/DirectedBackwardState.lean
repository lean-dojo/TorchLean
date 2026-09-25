/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.DirectedBackwardArithmetic

/-!
# Coefficient and constant invariants for backward CROWN

An absent coefficient represents zero. A present coefficient box must have the node's width and
enclose its exact real coefficient vector. This convention makes repeated parent occurrences
ordinary additions, including cancellation between the two parents of subtraction.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph.Internal
open scoped BigOperators

noncomputable section

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

/-- A box has the expected width and encloses a real coefficient at each valid coordinate. -/
def RowEncloses (box : FlatBox α) (n : Nat) (f : Nat → ℝ) : Prop :=
  box.dim = n ∧ ∀ i : Fin n,
    value (getAtOrZero box.lo [i.val]) ≤ f i.val ∧
      f i.val ≤ value (getAtOrZero box.hi [i.val])

/-- The coefficient table encloses an exact real coefficient for every node. -/
def CoefficientsEnclose (dims : Nat → Nat) (table : Array (Option (FlatBox α)))
    (f : Nat → Nat → ℝ) : Prop :=
  ∀ id, id < table.size →
    match table[id]! with
    | none => ∀ i : Fin (dims id), f id i.val = 0
    | some box => RowEncloses box (dims id) (f id)

/-- Both coefficient intervals and the accumulated constant contain an exact objective. -/
def StateEncloses (dims : Nat → Nat) (st : DirectedBackwardState α)
    (f : Nat → Nat → ℝ) (c : ℝ) : Prop :=
  CoefficientsEnclose dims st.coeffs f ∧ value st.cstLo ≤ c ∧ c ≤ value st.cstHi

private theorem read_set {β : Type} [Inhabited β] (xs : Array β)
    (i j : Nat) (a : β) (hj : j < xs.size) :
    (xs.set! i a)[j]! = if i = j then a else xs[j]! := by
  rw [Array.set!_eq_setIfInBounds]
  have hj' : j < (xs.setIfInBounds i a).size := by simpa using hj
  rw [getElem!_pos (c := xs.setIfInBounds i a) (i := j) hj']
  by_cases hij : i = j
  · subst i
    simp [Array.getElem_setIfInBounds_self]
  · rw [Array.getElem_setIfInBounds_ne hj hij, ite_eq_right hij,
      getElem!_pos (c := xs) (i := j) hj]

/-- Updating one coefficient row adds its exact contribution, even when the row was absent. -/
theorem addCoeff_encloses {dims : Nat → Nat} {st : DirectedBackwardState α}
    {f : Nat → Nat → ℝ} {c : ℝ} (hstate : StateEncloses dims st f c)
    (pid : Nat) (hp : pid < st.coeffs.size) (box : FlatBox α) (a : Nat → ℝ)
    (hbox : RowEncloses box (dims pid) a) :
    StateEncloses dims (addDirectedCoeff st pid box)
      (fun id i => if id = pid then f id i + a i else f id i) c ∧
      (addDirectedCoeff st pid box).failed = st.failed ∧
      (addDirectedCoeff st pid box).coeffs.size = st.coeffs.size := by
  obtain ⟨hcoeffs, hc⟩ := hstate
  have hpcoeff := hcoeffs pid hp
  cases hentry : st.coeffs[pid]! with
  | none =>
      simp only [hentry] at hpcoeff
      simp only [addDirectedCoeff, hentry]
      refine ⟨⟨?_, hc⟩, True.intro, Array.size_set! ..⟩
      intro id hid
      have hid' : id < st.coeffs.size := by simpa using hid
      rw [read_set _ _ _ _ hid']
      by_cases heq : id = pid
      · subst id
        simp only [↓reduceIte]
        refine ⟨hbox.1, ?_⟩
        intro i
        simpa only [hpcoeff i, zero_add] using hbox.2 i
      · rw [ite_eq_right (Ne.symm heq)]
        simpa only [heq, ↓reduceIte] using hcoeffs id hid'
  | some previous =>
      simp only [hentry] at hpcoeff
      have hdim : previous.dim = box.dim := hpcoeff.1.trans hbox.1.symm
      simp only [addDirectedCoeff, hentry, dite_eq_left hdim]
      refine ⟨⟨?_, hc⟩, True.intro, Array.size_set! ..⟩
      intro id hid
      have hid' : id < st.coeffs.size := by simpa using hid
      rw [read_set _ _ _ _ hid']
      by_cases heq : id = pid
      · subst id
        simp only [↓reduceIte]
        refine ⟨hpcoeff.1, ?_⟩
        intro i
        obtain ⟨previousDim, previousLo, previousHi⟩ := previous
        obtain ⟨boxDim, boxLo, boxHi⟩ := box
        dsimp only at hdim hpcoeff hbox ⊢
        subst boxDim
        have hn := hpcoeff.1
        change previousDim = dims pid at hn
        subst previousDim
        simpa only [castDimScalar_self, read_fin, Tensor.getScalar_map2Spec] using
          And.intro
            ((LawfulBoundOps.addDown_le (previousLo.getScalar i) (boxLo.getScalar i)).trans
              (add_le_add (by simpa only [read_fin] using (hpcoeff.2 i).1)
                (by simpa only [read_fin] using (hbox.2 i).1)))
            ((add_le_add (by simpa only [read_fin] using (hpcoeff.2 i).2)
              (by simpa only [read_fin] using (hbox.2 i).2)).trans
              (LawfulBoundOps.le_addUp (previousHi.getScalar i) (boxHi.getScalar i)))
      · rw [ite_eq_right (Ne.symm heq)]
        simpa only [heq, ↓reduceIte] using hcoeffs id hid'

/-- Accumulating a constant interval preserves enclosure of the exact shifted constant. -/
theorem addConstant_encloses {dims : Nat → Nat} {st : DirectedBackwardState α}
    {f : Nat → Nat → ℝ} {c : ℝ} (hstate : StateEncloses dims st f c)
    (lo hi : α) (d : ℝ) (hd : value lo ≤ d ∧ d ≤ value hi) :
    StateEncloses dims (addDirectedConstant st lo hi) f (c + d) :=
  ⟨hstate.1,
    (LawfulBoundOps.addDown_le st.cstLo lo).trans (add_le_add hstate.2.1 hd.1),
    (add_le_add hstate.2.2 hd.2).trans (LawfulBoundOps.le_addUp st.cstHi hi)⟩

/-- Negating a coefficient exchanges the endpoints and encloses its exact additive inverse. -/
theorem negateCoeff_encloses
    {box : FlatBox α} {n : Nat} {a : Nat → ℝ} (hbox : RowEncloses box n a) :
    RowEncloses (negateDirectedCoeff box) n (fun i => -a i) := by
  obtain ⟨dim, lo, hi⟩ := box
  obtain ⟨hdim, hbox⟩ := hbox
  dsimp only at hdim
  subst dim
  refine ⟨rfl, ?_⟩
  intro i
  have hlo := LawfulBoundOps.subDown_le (0 : α) (hi.getScalar i)
  have hhi := LawfulBoundOps.le_subUp (0 : α) (lo.getScalar i)
  have hb := hbox i
  simp only [read_fin] at hb
  simp only [negateDirectedCoeff, read_fin, Tensor.getScalar_mapSpec]
  rw [(LawfulBoundOps.toReal_zero (α := α)), zero_sub] at hlo hhi
  exact ⟨hlo.trans (neg_le_neg hb.2), (neg_le_neg hb.1).trans hhi⟩

omit [LawfulBoundOps α] in
/-- Coefficient accumulation never changes the number of graph rows, even on failure. -/
@[simp] theorem addCoeff_size (st : DirectedBackwardState α) (pid : Nat) (box : FlatBox α) :
    (addDirectedCoeff st pid box).coeffs.size = st.coeffs.size := by
  unfold addDirectedCoeff
  split
  · exact Array.size_set! ..
  · split
    · exact Array.size_set! ..
    · rfl

omit [LawfulBoundOps α] in
/-- Discharging a node only changes the constant or the failure flag. -/
@[simp] theorem consume_size (st : DirectedBackwardState α) (aB xB : FlatBox α) :
    (consumeDirectedObjective st aB xB).coeffs.size = st.coeffs.size := by
  unfold consumeDirectedObjective
  split <;> rfl

omit [LawfulBoundOps α] in
/-- A failed state remains failed while accumulating any later coefficient. -/
theorem addCoeff_failed (st : DirectedBackwardState α) (pid : Nat) (box : FlatBox α)
    (h : st.failed = true) : (addDirectedCoeff st pid box).failed = true := by
  unfold addDirectedCoeff
  split
  · exact h
  · split
    · exact h
    · rfl

/-- The initial table encloses the output objective and zero at every other node. -/
theorem initial_encloses (dims : Nat → Nat)
    (size output : Nat) (obj : FlatTensor α)
    (hdim : obj.n = dims output) :
    StateEncloses dims
      { coeffs := (Array.replicate size none).set! output (some (pointCoeffBox obj))
        cstLo := 0, cstHi := 0 }
      (fun id i => if id = output then value (getAtOrZero obj.v [i]) else 0) 0 := by
  refine ⟨?_, by simp [(LawfulBoundOps.toReal_zero (α := α))]⟩
  intro id hid
  have hid' : id < (Array.replicate size (none : Option (FlatBox α))).size := by
    simpa using hid
  rw [read_set _ _ _ _ hid']
  by_cases heq : id = output
  · subst id
    simp only [↓reduceIte]
    exact ⟨hdim, fun _ => ⟨le_rfl, le_rfl⟩⟩
  · simp [heq, Ne.symm heq, show id < size by simpa using hid']

end

end NN.MLTheory.CROWN.Graph.DirectedBackward

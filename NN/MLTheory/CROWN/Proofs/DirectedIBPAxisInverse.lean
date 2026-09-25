/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import all NN.MLTheory.CROWN.Proofs.GraphConcatInversePermutation
public import NN.MLTheory.CROWN.Proofs.DirectedIBPAxisPlan
public import Mathlib.Data.Fintype.EquivFin

/-!
# Checked inverse axis arrays

A valid axis array represents a finite permutation. The executable inverse-table builder returns
its inverse, so applying that builder twice recovers the original axis array.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph

open Spec

noncomputable section

def axisArray {rank : Nat} (e : Equiv.Perm (Fin rank)) : Array Nat :=
  Array.ofFn fun i => (e i).val

@[simp] theorem axisArray_size {rank : Nat} (e : Equiv.Perm (Fin rank)) :
    (axisArray e).size = rank := by simp [axisArray]

@[simp] theorem axisArray_getElem {rank : Nat} (e : Equiv.Perm (Fin rank))
    (i : Nat) (hi : i < (axisArray e).size) :
    (axisArray e)[i] = (e ⟨i, by simpa using hi⟩).val := by
  simp [axisArray]

@[simp] theorem axisArray_getElem! {rank : Nat} (e : Equiv.Perm (Fin rank))
    (i : Fin rank) :
    (axisArray e)[i.val]! = (e i).val := by simp [axisArray, i.isLt]

/-- Read a natural-number list in the axis order encoded by a finite permutation. -/
def axisMap {rank : Nat} (e : Equiv.Perm (Fin rank)) (xs : List Nat) : List Nat :=
  (axisArray e).toList.map (fun i => xs[i]!)

@[simp] theorem axisMap_length {rank : Nat} (e : Equiv.Perm (Fin rank)) (xs : List Nat) :
    (axisMap e xs).length = rank := by simp [axisMap]

@[simp] theorem axisMap_getElem! {rank : Nat} (e : Equiv.Perm (Fin rank))
    (xs : List Nat) (i : Fin rank) :
    (axisMap e xs)[i.val]! = xs[(e i).val]! := by
  simp [axisMap, axisArray, List.map_ofFn, i.isLt]

theorem axisMap_symm {rank : Nat} (e : Equiv.Perm (Fin rank))
    (xs : List Nat) (hlen : xs.length = rank) :
    axisMap e.symm (axisMap e xs) = xs := by
  apply List.ext_getElem (by simp [hlen])
  intro i hi hix
  have hi' : i < rank := by simpa using hi
  have h := axisMap_getElem! e.symm (axisMap e xs) ⟨i, hi'⟩
  rw [axisMap_getElem! e xs (e.symm ⟨i, hi'⟩), Equiv.apply_symm_apply] at h
  simpa only [getElem!_pos (axisMap e.symm (axisMap e xs)) i hi,
    getElem!_pos xs i hix] using h

/-- An axis list obtained by swaps has an ordinary finite permutation representation. -/
theorem exists_axisArray {perm : Array Nat} {rank : Nat}
    (hperm : perm.toList.Perm (List.range rank)) :
    ∃ e : Equiv.Perm (Fin rank), perm = axisArray e := by
  have hsize : perm.size = rank := by simpa using hperm.length_eq
  have hbound (i : Fin rank) : perm[i.val]! < rank := by
    have hi : i.val < perm.size := by simp [hsize]
    have hm : perm[i.val]! ∈ perm.toList := by
      simp only [getElem!_pos perm i.val hi]
      exact List.getElem_mem (by simpa using hi)
    exact List.mem_range.mp (hperm.mem_iff.mp hm)
  let f : Fin rank → Fin rank := fun i => ⟨perm[i.val]!, hbound i⟩
  have hf : Function.Injective f := by
    intro a b hab
    apply Fin.ext
    have ha : a.val < perm.size := by simp [hsize]
    have hb : b.val < perm.size := by simp [hsize]
    have hal : a.val < perm.toList.length := by simpa using ha
    have hbl : b.val < perm.toList.length := by simpa using hb
    have hab' : perm.toList[a.val] = perm.toList[b.val] := by
      have := congrArg Fin.val hab
      simpa only [f, Array.getElem_toList,
        getElem!_pos perm a.val ha, getElem!_pos perm b.val hb] using this
    exact (hperm.nodup_iff.mpr (List.nodup_range)).getElem_inj_iff.mp hab'
  let e := Equiv.ofBijective f ((Finite.injective_iff_bijective).mp hf)
  refine ⟨e, ?_⟩
  apply Array.ext
  · simpa using hsize
  · intro i hi₁ hi₂
    simp only [axisArray_getElem, e, Equiv.ofBijective_apply, f]
    exact (getElem!_pos perm i hi₁).symm

/-- The checked inverse-table algorithm computes the inverse finite permutation. -/
theorem inversePerm_axisArray {rank : Nat} (e : Equiv.Perm (Fin rank)) :
    NN.IR.OpContracts.inversePerm (axisArray e) = .ok (axisArray e.symm) := by
  let inverse : Nat → Nat := fun i => if h : i < rank then (e.symm ⟨i, h⟩).val else 0
  have hinjective : ∀ a < (axisArray e).size, ∀ b < (axisArray e).size,
      inverse a = inverse b → a = b := by
    intro a ha b hb hab
    have ha' : a < rank := by simpa using ha
    have hb' : b < rank := by simpa using hb
    simp only [inverse, ha', hb', dite_true] at hab
    exact congrArg Fin.val (e.symm.injective (Fin.ext hab))
  have hbound (i : Nat) (hi : i < (axisArray e).size) :
      (axisArray e)[i] < (axisArray e).size := by
    simpa only [axisArray_getElem, axisArray_size] using
      (e ⟨i, by simpa using hi⟩).isLt
  have hindex (i : Nat) (hi : i < (axisArray e).size) :
      inverse (axisArray e)[i] = i := by
    simp only [axisArray_getElem]
    simp [inverse, (e ⟨i, by simpa using hi⟩).isLt]
  have hinverse (i : Nat) (hi : i < (axisArray e).size) :
      inverse i < (axisArray e).size := by
    have hi' : i < rank := by simpa using hi
    simp only [inverse, hi', dite_true, axisArray_size]
    exact (e.symm ⟨i, hi'⟩).isLt
  rw [inversePerm_eq_of_inverse (axisArray e) inverse
    hinjective hbound hindex hinverse]
  congr 1
  apply Array.ext
  · simp
  · intro i hi₁ hi₂
    have hi : i < rank := by simpa using hi₂
    simp [inverse, hi]

private theorem mapM_getElem?_eq {axes dims out : List Nat}
    (h : axes.mapM (fun i => dims[i]?) = some out) :
    out = axes.map (fun i => dims[i]!) := by
  induction axes generalizing out with
  | nil => simpa using h.symm
  | cons a axes ih =>
      simp only [List.mapM_cons, bind, Option.bind] at h
      cases ha : dims[a]? with
      | none => simp only [ha, reduceCtorEq] at h
      | some x =>
          simp only [ha] at h
          obtain ⟨tail, htail, heq⟩ := Option.bind_eq_some_iff.mp h
          obtain rfl := Option.some.inj heq
          rw [List.map_cons, ← ih htail]
          simp only [getElem!_def, ha]

/-- A successful shape check has the declared rank and reads precisely its requested axes. -/
theorem permuteShape_properties {s t : Shape} {perm : Array Nat}
    (h : Shape.permute? s perm.toList = some t) :
    perm.size = s.rank ∧ perm.toList.Nodup ∧
      t = perm.toList.map (fun i => s[i]!) := by
  have hlen : perm.size = s.rank := by
    by_contra hn
    simp [Shape.permute?, hn] at h
  have hnodup : perm.toList.Nodup := by
    by_contra hn
    simp [Shape.permute?, hlen, hn] at h
  simp only [Shape.permute?, Array.length_toList, hlen, bne_self_eq_false,
    Bool.false_eq_true, ite_false, hnodup, decide_true, Bool.not_true] at h
  obtain ⟨out, hout, heq⟩ := Option.map_eq_some_iff.mp h
  obtain rfl := heq
  exact ⟨hlen, hnodup, mapM_getElem?_eq hout⟩

end

end NN.MLTheory.CROWN.Graph

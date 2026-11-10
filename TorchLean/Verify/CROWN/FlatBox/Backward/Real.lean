/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.FlatBox.Backward
public import TorchLean.Verify.CROWN.FlatBox.Directed
public import TorchLean.Tensor.ShapeChange.Flatten
public import TorchLean.Tensor.List
public import TorchLean.Tensor.Rep.Traversal.Sequence

/-!
# The directed backward state and its real reading

The backward CROWN proofs on operator graphs share these lemmas, which concern tensors, flat boxes
and affine forms only (no graph). They give the real reading of the directed backward state (`dot`,
`pending`, `frontierValue`, `Represents`, `StateEncloses`), show that adding, consuming and
dropping coefficients keep it, and prove the transposed linear, slice, broadcast and
concatenation coefficient rules. The slice rule rests on the coordinate laws of checked slices
(`Tensor.sliceAxisRangeCoordinate_injective`, `Tensor.index_sliceAxisRangeCoordinate`) and on the
pullback of a slice at the source coordinates it reads and at the others
(`Tensor.sliceBackward_apply_sliceAxisRangeCoordinate`, `Tensor.sliceBackward_apply_eq_zero`). The
rest cover the reverse fold over the graph positions (`reverseSweep_preserves`), the real equations
of copies, pointwise binary operations and linear layers (`CopyEquation`, `BinaryEquation`,
`LinearEquation`), and the evaluation of affine bounds over an input box
(`affineBoundsEval_encloses`, `evalObjectiveBox_encloses`).
-/

@[expose] public section

open TorchLean.Shape.Coord (linearize linearize_unlinearize unlinearize)
open TorchLean.Shape.Coord (unlinearize_linearize)
open TorchLean.Shape (rank_cons)

namespace TorchLean.Tensor

/-- Distinct output coordinates of a checked slice read distinct source coordinates. -/
theorem sliceAxisRangeCoordinate_injective :
    {shape : Shape} → (dim : Fin shape.rank) →
      (start count : Nat) → (hRange : start + count ≤ Shape.axisSize shape dim.val) →
      Function.Injective (sliceAxisRangeCoordinate dim start count hRange)
  | .dim n rest, ⟨0, _⟩, start, count, hRange => by
      rintro ⟨i, c⟩ ⟨j, d⟩ h
      apply Prod.ext
      · apply Fin.ext
        have hIndex := congrArg (fun p : (Shape.dim n rest).Coord => p.1.val) h
        change start + i.val = start + j.val at hIndex
        change i.val = j.val
        exact Nat.add_left_cancel hIndex
      · exact congrArg (fun p : (Shape.dim n rest).Coord => p.2) h
  | .dim n rest, ⟨axis + 1, hAxis⟩, start, count, hRange => by
      rintro ⟨i, c⟩ ⟨j, d⟩ h
      apply Prod.ext
      · exact congrArg (fun p : (Shape.dim n rest).Coord => p.1) h
      · exact sliceAxisRangeCoordinate_injective (shape := rest)
          ⟨axis, by simp only [Shape.rank_cons] at hAxis; omega⟩ start count hRange
          (congrArg (fun p : (Shape.dim n rest).Coord => p.2) h)

/-- The source coordinate of a slice output coordinate has the output index plus the start along
the slice axis, and the output index along every other axis. -/
theorem index_sliceAxisRangeCoordinate : {shape : Shape} → (dim : Fin shape.rank) →
    (start count : Nat) → (hRange : start + count ≤ Shape.axisSize shape dim.val) →
    (c : Shape.Coord (shape.set dim.val count)) → (k : Nat) →
    Shape.Coord.index (sliceAxisRangeCoordinate dim start count hRange c) k =
      if k = dim.val then start + Shape.Coord.index c k else Shape.Coord.index c k
  | .dim n rest, ⟨0, _⟩, start, count, hRange, ⟨i, c⟩, k => by
      cases k with
      | zero => rfl
      | succ k => exact (ite_eq_right (Nat.succ_ne_zero k)).symm
  | .dim n rest, ⟨a + 1, ha⟩, start, count, hRange, ⟨i, c⟩, k => by
      cases k with
      | zero => exact (ite_eq_right (show ¬(0 = a + 1) by omega)).symm
      | succ k =>
        refine (index_sliceAxisRangeCoordinate (shape := rest)
          ⟨a, by simp only [Shape.rank_cons] at ha; omega⟩ start count hRange c k).trans ?_
        by_cases hk : k = a
        · rw [ite_eq_left hk, ite_eq_left (show k + 1 = a + 1 by rw [hk])]
          rfl
        · rw [ite_eq_right hk, ite_eq_right (show ¬(k + 1 = a + 1) by omega)]
          rfl

/-- The pullback of a slice along one axis with step one reads the upstream gradient at each
source coordinate the slice reads. -/
theorem sliceBackward_apply_sliceAxisRangeCoordinate {α : Type} [Storage α] [Zero α]
    {shape : Shape} (dim : Fin shape.rank) (start count : Nat)
    (hRange : start + count ≤ Shape.axisSize shape dim.val)
    (value : Tensor α (shape.set dim.val count))
    (c : Shape.Coord (shape.set dim.val count)) :
    sliceBackward [dim] [start] [start + count] [1]
        (castShape value (Shape.slice_singleton shape dim start count).symm)
        (sliceAxisRangeCoordinate dim start count hRange c) = value c := by
  have hidx := index_sliceAxisRangeCoordinate dim start count hRange c
  have hshape := Shape.slice_singleton shape dim start count
  simp only [sliceBackward, Rep.get_ofFn]
  rw [dite_eq_left, castShape_apply]
  · congr 1
    refine Shape.Coord.ext_index fun k hk => ?_
    rw [Shape.rank_set] at hk
    rw [Shape.Coord.index_cast hshape, Shape.Coord.index_ofFun,
      dite_eq_left (by rwa [Shape.rank_slice])]
    simp only [Shape.valueAt_singleton, Shape.Coord.val_get, Fin.val_cast, ite_self, Nat.div_one,
      hidx, Fin.ext_iff]
    split_ifs <;> omega
  · intro d
    have hlt := (Shape.Coord.get (_root_.cast (congrArg Shape.Coord hshape.symm) c) d).isLt
    rw [Shape.Coord.val_get, Shape.Coord.index_cast hshape.symm] at hlt
    simp only [Shape.valueAt_singleton, Shape.Coord.val_get, Fin.val_cast, ite_self, Nat.div_one,
      Nat.mod_one, hidx, Fin.ext_iff, true_and]
    split_ifs <;> omega

/-- The pullback of a slice along one axis with step one is zero at every source coordinate the
slice does not read. -/
theorem sliceBackward_apply_eq_zero {α : Type} [Storage α] [Zero α] {shape : Shape}
    (dim : Fin shape.rank) (start count : Nat)
    (hRange : start + count ≤ Shape.axisSize shape dim.val)
    (value : Tensor α (shape.slice [dim] [start] [start + count] [1])) (c : shape.Coord)
    (hc : c ∉ Set.range (sliceAxisRangeCoordinate dim start count hRange)) :
    sliceBackward [dim] [start] [start + count] [1] value c = 0 := by
  simp only [sliceBackward, Rep.get_ofFn]
  rw [dite_eq_right]
  intro hin
  apply hc
  refine ⟨_root_.cast (congrArg Shape.Coord (Shape.slice_singleton shape dim start count))
    (Shape.Coord.ofFun fun d => ⟨_, (hin d).2.2⟩), ?_⟩
  have hdim := (hin (dim.cast (shape.rank_slice [dim] [start] [start + count] [1]).symm)).1
  simp only [Shape.valueAt_singleton, Shape.Coord.val_get, Fin.cast_cast, Fin.cast_eq_self,
    ite_true] at hdim
  refine Shape.Coord.ext_index fun k hk => ?_
  rw [index_sliceAxisRangeCoordinate,
    Shape.Coord.index_cast (Shape.slice_singleton shape dim start count),
    Shape.Coord.index_ofFun, dite_eq_left (by rwa [Shape.rank_slice])]
  simp only [Shape.valueAt_singleton, Shape.Coord.val_get, Fin.val_cast, ite_self, Nat.div_one,
    Fin.ext_iff]
  split_ifs <;> (try subst_vars) <;> omega

end TorchLean.Tensor

namespace TorchLean.CROWN.Directed
open Tensor
open Graph
open Graph (intervalMul_encloses)
open scoped BigOperators
noncomputable section

open Shape (Coord)

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

/-- The slopes and corrections of `CROWN.Directed.coeffAffine` bound the product `a * x` of a
coefficient and an input in their boxes from below and above, as affine functions of `x`.

For an input box that crosses zero, the correction multiplies the coefficient width by the negative
input endpoint `l`. That multiplication reverses the inequality on the width, hence the opposite
rounding direction on the subtraction.
-/
theorem coeffAffine_encloses
    {l u al au : α} {x a : ℝ}
    (hl : value l ≤ x) (hu : x ≤ value u)
    (hal : value al ≤ a) (hau : a ≤ value au) :
    value (coeffAffine l u al au).1.1 * x +
        value (coeffAffine l u al au).1.2 ≤ a * x ∧
      a * x ≤ value (coeffAffine l u al au).2.1 * x +
        value (coeffAffine l u al au).2.2 := by
  by_cases hneg : l < (0 : α)
  · have hlneg : value l < 0 := by
      simpa only [(LawfulBoundOps.toReal_zero (α := α))] using (LawfulBoundOps.lt_iff l 0).mp hneg
    by_cases hpos : (0 : α) < u
    · simp only [coeffAffine, hneg, not_true_eq_false, decide_false,
        Bool.false_eq_true, ↓reduceIte, hpos]
      have hwidth := LawfulBoundOps.le_subUp au al
      have hwidth' := LawfulBoundOps.subDown_le al au
      have hcorrLo := (LawfulBoundOps.mulDown_le (BoundOps.subUp au al) l).trans
        (mul_le_mul_of_nonpos_right hwidth hlneg.le)
      have hcorrHi := (mul_le_mul_of_nonpos_right hwidth' hlneg.le).trans
        (LawfulBoundOps.le_mulUp (BoundOps.subDown al au) l)
      have hax := mul_nonneg (sub_nonneg.mpr hal) (sub_nonneg.mpr hl)
      have haw := mul_nonneg (sub_nonneg.mpr hau) (neg_nonneg.mpr hlneg.le)
      have hux := mul_nonneg (sub_nonneg.mpr hau) (sub_nonneg.mpr hl)
      have halw := mul_nonneg (sub_nonneg.mpr hal) (neg_nonneg.mpr hlneg.le)
      constructor <;> nlinarith
    · have hunonpos : value u ≤ 0 := by
        rw [← (LawfulBoundOps.toReal_zero (α := α))]
        exact le_of_not_gt fun h => hpos ((LawfulBoundOps.lt_iff 0 u).mpr h)
      have hx := hu.trans hunonpos
      simpa [coeffAffine, hneg, hpos, (LawfulBoundOps.toReal_zero (α := α))] using
        And.intro (mul_le_mul_of_nonpos_right hau hx) (mul_le_mul_of_nonpos_right hal hx)
  · have hlnonneg : 0 ≤ value l := by
      rw [← (LawfulBoundOps.toReal_zero (α := α))]
      exact le_of_not_gt fun h => hneg ((LawfulBoundOps.lt_iff l 0).mpr h)
    have hx := hlnonneg.trans hl
    simpa [coeffAffine, hneg, (LawfulBoundOps.toReal_zero (α := α))] using
      And.intro (mul_le_mul_of_nonneg_right hal hx) (mul_le_mul_of_nonneg_right hau hx)

/-- The interval dot product `CROWN.Directed.dotBox` of two boxes encloses the dot product of any
two vectors they enclose. -/
theorem dotBox_encloses
    {n : Nat} (aLo aHi bLo bHi : Tensor α [n]) (a b : Fin n → ℝ)
    (ha : ∀ i, value (aLo.getScalar i) ≤ a i ∧ a i ≤ value (aHi.getScalar i))
    (hb : ∀ i, value (bLo.getScalar i) ≤ b i ∧ b i ≤ value (bHi.getScalar i))
    {lo hi : α}
    (hresult : dotBox { dim := n, lo := aLo, hi := aHi }
      { dim := n, lo := bLo, hi := bHi } = some (lo, hi)) :
    value lo ≤ ∑ i, a i * b i ∧ (∑ i, a i * b i) ≤ value hi := by
  let term (i : Fin n) := intervalMul
    (aLo.getScalar i) (aHi.getScalar i) (bLo.getScalar i) (bHi.getScalar i)
  have hterms (i : Fin n) : value (term i).1 ≤ a i * b i ∧
      a i * b i ≤ value (term i).2 :=
    intervalMul_encloses (ha i).1 (ha i).2 (hb i).1 (hb i).2
  have h := sum_encloses (fun i => (term i).1) (fun i => (term i).2)
    (fun i => a i * b i) hterms
  let lower (acc : α) (i : Fin n) := BoundOps.addDown acc (term i).1
  let upper (acc : α) (i : Fin n) := BoundOps.addUp acc (term i).2
  have hfold := List.foldl_hom₂ (List.finRange n) Prod.mk lower upper
    (fun acc i => (lower acc.1 i, upper acc.2 i)) 0 0 (fun _ _ _ => rfl)
  dsimp only [lower, upper, term] at hfold
  simp only [dotBox, ↓reduceDIte, castDimScalar_self, getAtOrZero_eq_getScalar,
    Fin.foldl_eq_finRange_foldl, hfold, Option.some.injEq, Prod.mk.injEq] at hresult
  obtain ⟨rfl, rfl⟩ := hresult
  exact h

/-- The transposed linear map `CROWN.Directed.linear` sends a coefficient box that encloses `a`
to a box of width `n` whose entry `j` encloses the pairing of `a` with column `j` of `W`, and to a
constant interval that encloses the pairing of `a` with `b`. -/
theorem linear_encloses
    {m n : Nat} (aLo aHi : Tensor α [m]) (W : Tensor α [m, n]) (b : Tensor α [m])
    (a : Fin m → ℝ)
    (ha : ∀ i, value (aLo.getScalar i) ≤ a i ∧ a i ≤ value (aHi.getScalar i))
    {aX : FlatBox α} {cLo cHi : α}
    (hresult : linear { dim := m, lo := aLo, hi := aHi } W b =
      some (aX, (cLo, cHi))) :
    aX.dim = n ∧
      (∀ j : Fin n,
        value (getAtOrZero aX.lo [j.val]) ≤ ∑ i, a i * value (Tensor.get2 W i j) ∧
          (∑ i, a i * value (Tensor.get2 W i j)) ≤ value (getAtOrZero aX.hi [j.val])) ∧
      value cLo ≤ ∑ i, a i * value (b.getScalar i) ∧
        (∑ i, a i * value (b.getScalar i)) ≤ value cHi := by
  simp only [linear, ↓reduceDIte, castDimScalar_self] at hresult
  obtain ⟨c, hc, hpair⟩ := Option.map_eq_some_iff.mp hresult
  cases hpair
  refine ⟨rfl, ?_, ?_⟩
  · intro j
    let term (i : Fin m) := intervalMul (aLo.getScalar i) (aHi.getScalar i)
      (Tensor.get2 W i j) (Tensor.get2 W i j)
    have hterm (i : Fin m) := intervalMul_encloses
      (ha i).1 (ha i).2
      (le_refl (value (Tensor.get2 W i j))) (le_refl (value (Tensor.get2 W i j)))
    have hsum := sum_encloses (fun i => (term i).1) (fun i => (term i).2)
      (fun i => a i * value (Tensor.get2 W i j)) hterm
    let lower (acc : α) (i : Fin m) := BoundOps.addDown acc (term i).1
    let upper (acc : α) (i : Fin m) := BoundOps.addUp acc (term i).2
    have hfold := List.foldl_hom₂ (List.finRange m) Prod.mk lower upper
      (fun acc i => (lower acc.1 i, upper acc.2 i)) 0 0 (fun _ _ _ => rfl)
    dsimp only [lower, upper, term] at hfold
    simpa only [getAtOrZero_eq_getScalar, Tensor.getScalar_dim, getAtOrZero_eq_get2, hfold] using
      hsum
  · exact dotBox_encloses aLo aHi b b a (fun i => value (b.getScalar i))
      ha (fun _ => ⟨le_rfl, le_rfl⟩) hc

/-- The real value of a one-row affine form at `x`: its row paired with `x`, plus its constant. -/
def affineValue {n : Nat} (aff : Linear α n 1) (x : Fin n → ℝ) : ℝ :=
  (∑ i, value (Tensor.get2 aff.weights 0 i) * x i) + value (aff.bias.getScalar 0)

/-- The lower and upper forms that `CROWN.Directed.inputAffines` builds bound `a · x + c` from below
and above, for every input `x`, coefficients `a` and constant `c` that its input box, its
coefficient box and its constant interval enclose. -/
theorem inputAffines_encloses
    {n : Nat} (xLo xHi aLo aHi : Tensor α [n]) (cLo cHi : α)
    (x a : Fin n → ℝ) (c : ℝ)
    (hx : ∀ i, value (xLo.getScalar i) ≤ x i ∧ x i ≤ value (xHi.getScalar i))
    (ha : ∀ i, value (aLo.getScalar i) ≤ a i ∧ a i ≤ value (aHi.getScalar i))
    (hc : value cLo ≤ c ∧ c ≤ value cHi)
    {lower upper : Linear α n 1}
    (hresult : inputAffines n
      { dim := n, lo := xLo, hi := xHi } { dim := n, lo := aLo, hi := aHi } cLo cHi =
        some (lower, upper)) :
    affineValue lower x ≤ (∑ i, a i * x i) + c ∧
      (∑ i, a i * x i) + c ≤ affineValue upper x := by
  let selected (i : Fin n) :=
    coeffAffine (xLo.getScalar i) (xHi.getScalar i)
      (aLo.getScalar i) (aHi.getScalar i)
  have hselected (i : Fin n) :
      value (selected i).1.1 * x i + value (selected i).1.2 ≤ a i * x i ∧
        a i * x i ≤ value (selected i).2.1 * x i + value (selected i).2.2 :=
    coeffAffine_encloses (hx i).1 (hx i).2 (ha i).1 (ha i).2
  have hcorrection := sum_encloses
    (fun i => (selected i).1.2) (fun i => (selected i).1.2)
    (fun i => value (selected i).1.2) (fun _ => ⟨le_rfl, le_rfl⟩)
  have hcorrection' := sum_encloses
    (fun i => (selected i).2.2) (fun i => (selected i).2.2)
    (fun i => value (selected i).2.2) (fun _ => ⟨le_rfl, le_rfl⟩)
  have hlo := (LawfulBoundOps.addDown_le cLo _).trans (add_le_add hc.1 hcorrection.1)
  have hhi := (add_le_add hc.2 hcorrection'.2).trans (LawfulBoundOps.le_addUp cHi _)
  have hlower := Finset.sum_le_sum (fun i (_ : i ∈ Finset.univ) => (hselected i).1)
  have hupper := Finset.sum_le_sum (fun i (_ : i ∈ Finset.univ) => (hselected i).2)
  simp only [Finset.sum_add_distrib] at hlower hupper
  simp only [inputAffines, ↓reduceDIte, castDimScalar_self, getAtOrZero_eq_getScalar,
    Option.some.injEq, Prod.mk.injEq] at hresult
  obtain ⟨rfl, rfl⟩ := hresult
  simp only [affineValue, Tensor.get2_dim, Tensor.getScalar_dim]
  change
    (∑ i, value (selected i).1.1 * x i) +
        value (BoundOps.addDown cLo
          ((List.finRange n).foldl (fun acc i => BoundOps.addDown acc (selected i).1.2) 0)) ≤
      (∑ i, a i * x i) + c ∧
    (∑ i, a i * x i) + c ≤
      (∑ i, value (selected i).2.1 * x i) +
        value (BoundOps.addUp cHi
          ((List.finRange n).foldl (fun acc i => BoundOps.addUp acc (selected i).2.2) 0))
  constructor <;> linarith

/-- A coefficient table encloses the real coefficients `f`: the box of each node encloses its
coefficients, and a node without a box has zero coefficients. -/
def CoefficientsEnclose (dims : Nat → Nat) (table : Array (Option (FlatBox α)))
    (f : Nat → Nat → ℝ) : Prop :=
  ∀ id, id < table.size →
    match table[id]! with
    | none => ∀ i : Fin (dims id), f id i.val = 0
    | some box => RowEncloses box (dims id) (f id)

/-- A state encloses the real coefficients `f` and the real constant `c` when its table encloses
`f` and its constant interval holds `c`. -/
def StateEncloses (dims : Nat → Nat) (st : State α)
    (f : Nat → Nat → ℝ) (c : ℝ) : Prop :=
  CoefficientsEnclose dims st.coeffs f ∧ value st.cstLo ≤ c ∧ c ≤ value st.cstHi

private theorem read_set {β : Type} [Inhabited β] (xs : Array β)
    (i j : Nat) (a : β) (hj : j < xs.size) :
    (xs.set! i a)[j]! = if i = j then a else xs[j]! := by
  by_cases hij : i = j
  · subst i
    simp only [Array.getElem!_set!_self xs j a hj, ite_true]
  · rw [Array.getElem!_set!_ne xs i j a hij, ite_eq_right hij]

/-- Adding a box that encloses `a` at row `pid` adds `a` to the coefficients of that row, whether
or not the row had a box, and keeps the failure flag and the size of the table. -/
theorem addCoeff_encloses {dims : Nat → Nat} {st : State α}
    {f : Nat → Nat → ℝ} {c : ℝ} (hstate : StateEncloses dims st f c)
    (pid : Nat) (hp : pid < st.coeffs.size) (box : FlatBox α) (a : Nat → ℝ)
    (hbox : RowEncloses box (dims pid) a) :
    StateEncloses dims (addCoeff st pid box)
      (fun id i => if id = pid then f id i + a i else f id i) c ∧
      (addCoeff st pid box).failed = st.failed ∧
      (addCoeff st pid box).coeffs.size = st.coeffs.size := by
  obtain ⟨hcoeffs, hc⟩ := hstate
  have hpcoeff := hcoeffs pid hp
  cases hentry : st.coeffs[pid]! with
  | none =>
      simp only [hentry] at hpcoeff
      simp only [addCoeff, hp, ↓reduceIte, hentry]
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
      simp only [addCoeff, hp, ↓reduceIte, hentry, dite_eq_left hdim]
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
        simpa only [castDimScalar_self, Tensor.getAtOrZero_eq_getScalar,
          Tensor.getScalar_zipWith] using
          And.intro
            ((LawfulBoundOps.addDown_le (previousLo.getScalar i) (boxLo.getScalar i)).trans
              (add_le_add (by simpa only [Tensor.getAtOrZero_eq_getScalar] using (hpcoeff.2 i).1)
                (by simpa only [Tensor.getAtOrZero_eq_getScalar] using (hbox.2 i).1)))
            ((add_le_add (by simpa only [Tensor.getAtOrZero_eq_getScalar] using (hpcoeff.2 i).2)
              (by simpa only [Tensor.getAtOrZero_eq_getScalar] using (hbox.2 i).2)).trans
              (LawfulBoundOps.le_addUp (previousHi.getScalar i) (boxHi.getScalar i)))
      · rw [ite_eq_right (Ne.symm heq)]
        simpa only [heq, ↓reduceIte] using hcoeffs id hid'

/-- Adding an interval that holds `d` to the constant adds `d` to the enclosed constant. -/
theorem addConstant_encloses {dims : Nat → Nat} {st : State α}
    {f : Nat → Nat → ℝ} {c : ℝ} (hstate : StateEncloses dims st f c)
    (lo hi : α) (d : ℝ) (hd : value lo ≤ d ∧ d ≤ value hi) :
    StateEncloses dims (addConstant st lo hi) f (c + d) :=
  ⟨hstate.1,
    (LawfulBoundOps.addDown_le st.cstLo lo).trans (add_le_add hstate.2.1 hd.1),
    (add_le_add hstate.2.2 hd.2).trans (LawfulBoundOps.le_addUp st.cstHi hi)⟩

omit [LawfulBoundOps α] in
/-- Adding a coefficient keeps the table's size, also when a width mismatch or an out-of-range node
fails the state. -/
@[simp] theorem addCoeff_size (st : State α) (pid : Nat) (box : FlatBox α) :
    (addCoeff st pid box).coeffs.size = st.coeffs.size := by
  unfold addCoeff
  split
  · split
    · exact Array.size_set! ..
    · split
      · exact Array.size_set! ..
      · rfl
  · rfl

omit [LawfulBoundOps α] in
/-- A fold of coefficient additions keeps the size of the table. -/
@[simp] theorem addParents_size (parents : Array (Nat × FlatBox α))
    (st : State α) :
    (parents.foldl (fun state entry => addCoeff state entry.1 entry.2) st).coeffs.size =
      st.coeffs.size := by
  apply Array.foldl_induction
    (fun (_ : Nat) (state : State α) => state.coeffs.size = st.coeffs.size) rfl
  intro i state h
  simpa only [addCoeff_size] using h

/-- A property `P` that each step below `size` takes from position `k + 1` to `k` holds at `0`
after the reverse fold over `List.finRange n` with `n ≤ size`, from a state where it holds at
`n`. -/
theorem reverseSweep_preserves {β : Type} (step : β → Nat → β) (P : Nat → β → Prop)
    (size : Nat) (hstep : ∀ k, k < size → ∀ st, P (k + 1) st → P k (step st k)) :
    ∀ n, n ≤ size → ∀ st, P n st →
      P 0 ((List.finRange n).reverse.foldl (fun state i => step state i.val) st) := by
  intro n
  induction n with
  | zero =>
      intro _ st h
      exact h
  | succ n ih =>
      intro hn st h
      have hs := hstep n (by omega) st h
      simpa only [List.finRange_succ_last, List.reverse_append, List.reverse_cons,
        List.reverse_nil, List.cons_append, List.nil_append, List.foldl_cons,
        ← List.map_reverse, List.foldl_map,
        Fin.val_last, Fin.val_castSucc] using ih (by omega) (step st n) hs

omit [LawfulBoundOps α] in
/-- Discharging a node only changes the constant or the failure flag. -/
@[simp] theorem consume_size (st : State α) (aB xB : FlatBox α) :
    (consume st aB xB).coeffs.size = st.coeffs.size := by
  unfold consume
  split <;> rfl

omit [LawfulBoundOps α] in
/-- Adding a coefficient to a failed state gives a failed state. -/
theorem addCoeff_failed (st : State α) (pid : Nat) (box : FlatBox α)
    (h : st.failed = true) : (addCoeff st pid box).failed = true := by
  unfold addCoeff
  split
  · split
    · exact h
    · split
      · exact h
      · rfl
  · rfl

/-- The initial state encloses the reals of the objective at the output node, zero at every other
node and the constant `0`. -/
theorem initial_encloses (dims : Nat → Nat)
    (size output : Nat) (obj : FlatTensor α)
    (hdim : obj.n = dims output) :
    StateEncloses dims
      { coeffs := (Array.replicate size none).set! output (some (FlatBox.ofTensor obj.v))
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

/-- The real inner product of the first `n` coordinates of `a` and `x`. -/
def dot (n : Nat) (a x : Nat → ℝ) : ℝ := (a ∘ Fin.val : Fin n → ℝ) ⬝ᵥ (x ∘ Fin.val)

/-- The pending nodes at position `k` of the reverse sweep: the nodes below `k`, and the designated
input even when its index is at least `k`. -/
def pending (input k : Nat) : Finset Nat := insert input (Finset.range k)

/-- The real objective of the coefficients `f` and the constant `c` over the pending nodes: the
pairings of `f` with `v` at those nodes, plus `c`. -/
def frontierValue (dims : Nat → Nat) (v : Nat → Nat → ℝ) (input k : Nat)
    (f : Nat → Nat → ℝ) (c : ℝ) : ℝ :=
  (∑ id ∈ pending input k, dot (dims id) (f id) (v id)) + c

/-- A state represents `z` at position `k` when it encloses real coefficients and a real constant
whose objective over the pending nodes is `z`. -/
def Represents (dims : Nat → Nat) (v : Nat → Nat → ℝ) (input k : Nat)
    (st : State α) (z : ℝ) : Prop :=
  ∃ f c, StateEncloses dims st f c ∧ frontierValue dims v input k f c = z

/-- Adding `a` to the coefficients of a pending node `pid` adds the pairing of `a` with the node's
coordinates to the objective. -/
theorem frontier_add {dims : Nat → Nat} {v : Nat → Nat → ℝ} {input k pid : Nat}
    (hp : pid ∈ pending input k) (f : Nat → Nat → ℝ) (a : Nat → ℝ) (c : ℝ) :
    frontierValue dims v input k
      (fun id i => if id = pid then f id i + a i else f id i) c =
      frontierValue dims v input k f c + dot (dims pid) a (v pid) := by
  have hrow (id : Nat) :
      dot (dims id) (fun i => if id = pid then f id i + a i else f id i) (v id) =
        dot (dims id) (f id) (v id) +
          if id = pid then dot (dims pid) a (v pid) else 0 := by
    by_cases h : id = pid
    · subst id
      simp [dot, dotProduct, add_mul, Finset.sum_add_distrib]
    · simp [h, dot, dotProduct]
  simp only [frontierValue, hrow, Finset.sum_add_distrib]
  simp only [Finset.sum_ite_eq', hp, ↓reduceIte]
  ring

/-- At a node `k` other than the designated input, the objective over the pending nodes at `k + 1`
is the one at `k` plus the pairing of row `k` with the node's coordinates. -/
theorem frontier_succ {dims : Nat → Nat} {v : Nat → Nat → ℝ} {input k : Nat}
    (hk : k ≠ input) (f : Nat → Nat → ℝ) (c : ℝ) :
    frontierValue dims v input (k + 1) f c =
      frontierValue dims v input k f c + dot (dims k) (f k) (v k) := by
  have hset : pending input (k + 1) = insert k (pending input k) := by
    simp only [pending, Finset.range_add_one, Finset.insert_comm]
  have hnot : k ∉ pending input k := by simp [pending, hk]
  rw [frontierValue, hset, Finset.sum_insert hnot]
  unfold frontierValue
  ring

/-- The objective over the pending nodes is the same at `input + 1` and at `input`, since the
designated input is pending at both. -/
theorem frontier_input {dims : Nat → Nat} {v : Nat → Nat → ℝ} (input : Nat)
    (f : Nat → Nat → ℝ) (c : ℝ) :
    frontierValue dims v input (input + 1) f c =
      frontierValue dims v input input f c := by
  simp [frontierValue, pending, Finset.range_add_one]

/-- At position `0` the objective over the pending nodes is the pairing at the designated input
plus the constant. -/
theorem frontier_zero (dims : Nat → Nat) (v : Nat → Nat → ℝ) (input : Nat)
    (f : Nat → Nat → ℝ) (c : ℝ) :
    frontierValue dims v input 0 f c = dot (dims input) (f input) (v input) + c := by
  simp [frontierValue, pending]

/-- Adding a box that encloses `a` at a pending node `pid` adds the pairing of `a` with the node's
coordinates to the represented value. -/
theorem represents_add {dims : Nat → Nat} {v : Nat → Nat → ℝ} {input k : Nat}
    {st : State α} {z : ℝ}
    (h : Represents dims v input k st z) (pid : Nat)
    (hp : pid < st.coeffs.size) (hlive : pid ∈ pending input k)
    (box : FlatBox α) (a : Nat → ℝ) (ha : RowEncloses box (dims pid) a) :
    Represents dims v input k (addCoeff st pid box)
      (z + dot (dims pid) a (v pid)) := by
  obtain ⟨f, c, hs, hz⟩ := h
  refine ⟨_, c, (addCoeff_encloses hs pid hp box a ha).1, ?_⟩
  rw [frontier_add hlive, hz]

/-- Adding an interval that holds `d` to the constant adds `d` to the represented value. -/
theorem represents_addConstant {dims : Nat → Nat} {v : Nat → Nat → ℝ} {input k : Nat}
    {st : State α} {z : ℝ}
    (h : Represents dims v input k st z) (lo hi : α) (d : ℝ)
    (hd : value lo ≤ d ∧ d ≤ value hi) :
    Represents dims v input k (addConstant st lo hi) (z + d) := by
  obtain ⟨f, c, hs, hz⟩ := h
  refine ⟨f, c + d, addConstant_encloses hs lo hi d hd, ?_⟩
  simp only [frontierValue] at hz ⊢
  linarith

/-- Passing a node other than the designated input that has no coefficient box keeps the
represented value. -/
theorem represents_drop_none {dims : Nat → Nat} {v : Nat → Nat → ℝ} {input k : Nat}
    {st : State α} {z : ℝ}
    (h : Represents dims v input (k + 1) st z) (hk : k ≠ input)
    (hsize : k < st.coeffs.size) (hnone : st.coeffs[k]! = none) :
    Represents dims v input k st z := by
  obtain ⟨f, c, hs, hz⟩ := h
  have hf := hs.1 k hsize
  simp only [hnone] at hf
  have hdot : dot (dims k) (f k) (v k) = 0 := by simp [dot, dotProduct, hf]
  exact ⟨f, c, hs, by simpa only [frontier_succ hk, hdot, add_zero] using hz⟩

/-- When the row of a node `k` other than the designated input has a box, the box encloses some
`a`, and at position `k` the state represents `z` less the pairing of `a` with the coordinates of
`k`. -/
theorem represents_drop_some {dims : Nat → Nat} {v : Nat → Nat → ℝ} {input k : Nat}
    {st : State α} {z : ℝ}
    (h : Represents dims v input (k + 1) st z) (hk : k ≠ input)
    (hsize : k < st.coeffs.size) {box : FlatBox α} (hsome : st.coeffs[k]! = some box) :
    ∃ a, RowEncloses box (dims k) a ∧
      Represents dims v input k st (z - dot (dims k) a (v k)) := by
  obtain ⟨f, c, hs, hz⟩ := h
  have hf := hs.1 k hsize
  simp only [hsome] at hf
  refine ⟨f k, hf, f, c, hs, ?_⟩
  rw [frontier_succ hk] at hz
  linarith

/-- The interval dot product `CROWN.Directed.dotBox` of two boxes that enclose `a` and `x` at width
`n` encloses `dot n a x`. -/
theorem dotBox_row_encloses
    {n : Nat} {aB xB : FlatBox α} {a x : Nat → ℝ}
    (ha : RowEncloses aB n a) (hx : RowEncloses xB n x) {lo hi : α}
    (hresult : dotBox aB xB = some (lo, hi)) :
    value lo ≤ dot n a x ∧ dot n a x ≤ value hi := by
  obtain ⟨adim, alo, ahi⟩ := aB
  obtain ⟨xdim, xlo, xhi⟩ := xB
  obtain ⟨haDim, ha⟩ := ha
  obtain ⟨hxDim, hx⟩ := hx
  dsimp only at haDim hxDim
  subst adim
  subst xdim
  exact dotBox_encloses alo ahi xlo xhi (fun i => a i.val) (fun i => x i.val)
    (by simpa only [getAtOrZero_eq_getScalar] using ha)
    (by simpa only [getAtOrZero_eq_getScalar] using hx) hresult

/-- Discharging a coefficient box that encloses `a` against a box that encloses `x` fails, or adds
`dot n a x` to the represented value through the constant. -/
theorem represents_consume
    {dims : Nat → Nat} {v : Nat → Nat → ℝ} {input k n : Nat}
    {st : State α} {z : ℝ}
    (h : Represents dims v input k st z) {aB xB : FlatBox α} {a x : Nat → ℝ}
    (ha : RowEncloses aB n a) (hx : RowEncloses xB n x) :
    (consume st aB xB).failed = true ∨
      Represents dims v input k (consume st aB xB) (z + dot n a x) := by
  cases hresult : dotBox aB xB with
  | none =>
      left
      simp [consume, hresult, State.fail]
  | some bounds =>
      rcases bounds with ⟨lo, hi⟩
      right
      simp only [consume, hresult]
      exact represents_addConstant h lo hi (dot n a x)
        (dotBox_row_encloses ha hx hresult)

/-- Entry `j` of the transposed product of `W` with the coefficients `a`, at the reals of `W`. -/
def transposedCoeff {m n : Nat} (W : Tensor α [m, n]) (a : Nat → ℝ) (j : Nat) : ℝ :=
  ∑ i : Fin m, a i.val * value (getAtOrZero W [i.val, j])

/-- The pairing of the coefficients `a` with the reals of the bias `b`. -/
def biasDot {m : Nat} (b : Tensor α [m]) (a : Nat → ℝ) : ℝ :=
  ∑ i : Fin m, a i.val * value (b.getScalar i)

/-- The transposed linear map of a box that encloses `a` gives a box that encloses
`transposedCoeff W a` and a constant interval that holds `biasDot b a`. -/
theorem linear_row_encloses
    {m n : Nat} {aB : FlatBox α} (W : Tensor α [m, n]) (b : Tensor α [m])
    {a : Nat → ℝ} (ha : RowEncloses aB m a)
    {aX : FlatBox α} {lo hi : α}
    (hresult : linear aB W b = some (aX, (lo, hi))) :
    RowEncloses aX n (transposedCoeff W a) ∧
      value lo ≤ biasDot b a ∧ biasDot b a ≤ value hi := by
  obtain ⟨adim, alo, ahi⟩ := aB
  obtain ⟨hdim, ha⟩ := ha
  dsimp only at hdim
  subst adim
  have h := linear_encloses alo ahi W b (fun i => a i.val)
    (by simpa only [getAtOrZero_eq_getScalar] using ha) hresult
  refine ⟨⟨h.1, ?_⟩, h.2.2⟩
  intro j
  simpa only [transposedCoeff, getAtOrZero_eq_get2] using h.2.1 j

/-- If `y` is `W x + b` at the reals of `W` and `b`, pairing `a` with `y` gives the pairing of
`transposedCoeff W a` with `x` plus `biasDot b a`. -/
theorem dot_linear {m n : Nat} (W : Tensor α [m, n]) (b : Tensor α [m])
    (a x y : Nat → ℝ)
    (hy : ∀ i : Fin m, y i.val =
      (∑ j : Fin n, value (Tensor.get2 W i j) * x j.val) + value (b.getScalar i)) :
    dot m a y = dot n (transposedCoeff W a) x + biasDot b a := by
  simp only [dot, dotProduct, Function.comp_apply, hy, mul_add, Finset.sum_add_distrib,
    Finset.mul_sum, transposedCoeff, Finset.sum_mul, getAtOrZero_eq_get2, biasDot]
  congr 1
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro j _
  apply Finset.sum_congr rfl
  intro i _
  ring

/-- A successful transposed linear map at a pending node `pid` adds the pairing of `a` with the
layer's output `y` to the represented value, whether or not `pid` has a coefficient box. It keeps
the failure flag and the size of the table. -/
theorem represents_linear
    {dims : Nat → Nat} {v : Nat → Nat → ℝ} {input k m n : Nat}
    {st : State α} {z : ℝ}
    (h : Represents dims v input k st z) (pid : Nat)
    (hp : pid < st.coeffs.size) (hlive : pid ∈ pending input k) (hdim : dims pid = n)
    {aB : FlatBox α} {a y : Nat → ℝ} (ha : RowEncloses aB m a)
    (W : Tensor α [m, n]) (b : Tensor α [m])
    (hy : ∀ i : Fin m, y i.val =
      (∑ j : Fin n, value (Tensor.get2 W i j) * v pid j.val) + value (b.getScalar i))
    {aX : FlatBox α} {lo hi : α}
    (hresult : linear aB W b = some (aX, (lo, hi))) :
    Represents dims v input k (addConstant (addCoeff st pid aX) lo hi)
        (z + dot m a y) ∧
      (addConstant (addCoeff st pid aX) lo hi).failed = st.failed ∧
      (addConstant (addCoeff st pid aX) lo hi).coeffs.size = st.coeffs.size := by
  have he := linear_row_encloses W b ha hresult
  have hr : RowEncloses aX (dims pid) (transposedCoeff W a) := by
    simpa only [hdim] using he.1
  have hadd := represents_add h pid hp hlive aX (transposedCoeff W a) hr
  have hc := represents_addConstant hadd lo hi (biasDot b a) he.2
  obtain ⟨f, c, hs, _⟩ := h
  have hstate := addCoeff_encloses hs pid hp aX (transposedCoeff W a) hr
  refine ⟨?_, hstate.2.1, hstate.2.2⟩
  simpa only [hdim, dot_linear W b a (v pid) y hy, add_assoc] using hc

/-- The real equation of a linear node `id` with input `pid`, weight `W` and bias `b`: the node
has width `m`, its input width `n`, and coordinate `i` is row `i` of `W` paired with the input plus
entry `i` of `b`. -/
def LinearEquation (dims : Nat → Nat) (v : Nat → Nat → ℝ) (id pid : Nat)
    {m n : Nat} (W : Tensor α [m, n]) (b : Tensor α [m]) : Prop :=
  dims id = m ∧ dims pid = n ∧ ∀ i : Fin m, v id i.val =
    (∑ j : Fin n, value (Tensor.get2 W i j) * v pid j.val) + value (b.getScalar i)

/-- The real equation of a copy `id` of node `pid`: the two nodes have equal widths and equal
coordinates. -/
def CopyEquation (dims : Nat → Nat) (v : Nat → Nat → ℝ) (id pid : Nat) : Prop :=
  dims pid = dims id ∧ ∀ i : Fin (dims id), v id i.val = v pid i.val

/-- The real equation of a node `id` that applies `f` entry by entry to nodes `p` and `q` of its
width. -/
def BinaryEquation (dims : Nat → Nat) (v : Nat → Nat → ℝ) (id p q : Nat)
    (f : ℝ → ℝ → ℝ) : Prop :=
  dims p = dims id ∧ dims q = dims id ∧
    ∀ i : Fin (dims id), v id i.val = f (v p i.val) (v q i.val)

/-- The finite vector `f` read at every natural index, with zero from `n` on, which lets a
coefficient row of the graph's table be built from `f` without dependent casts. -/
def extendFin {n : Nat} (f : Fin n → ℝ) (i : Nat) : ℝ :=
  if hi : i < n then f ⟨i, hi⟩ else 0

/-- The extension agrees with the finite vector at every valid coordinate. -/
@[simp] theorem extendFin_val {n : Nat} (f : Fin n → ℝ) (i : Fin n) :
    extendFin f i.val = f i := by simp [extendFin, i.isLt]

/-- Distinct output entries of a slice layout read distinct source entries. -/
theorem slice_flatIndex_injective (layout : SliceLayout) :
    Function.Injective layout.flatIndex := by
  intro i j h
  have hcoord := congrArg (fun k : Fin layout.sourceShape.size => Coord.unlinearize k) h
  simp only [SliceLayout.flatIndex, Coord.unlinearize_linearize] at hcoord
  have hout := Tensor.sliceAxisRangeCoordinate_injective
    ⟨layout.axis, layout.axisInBounds.proof⟩ layout.start layout.length
      layout.start_add_length_le hcoord
  simpa only [Coord.linearize_unlinearize] using
    congrArg (fun c : layout.outputShape.Coord => Coord.linearize c) hout

/-- The scatter of a slice layout puts entry `i` of the coefficient at the source entry
`flatIndex i` that output entry `i` reads. -/
theorem slice_scatter_selected {β : Type} [Storage β] [Zero β] (layout : SliceLayout)
    (coefficient : Tensor β [layout.outputShape.size]) (i : Fin layout.outputShape.size) :
    (layout.scatter coefficient).getScalar (layout.flatIndex i) = coefficient.getScalar i := by
  unfold SliceLayout.scatter SliceLayout.flatIndex
  rw [Tensor.getScalar_reshape_toFin,
    Tensor.sliceBackward_apply_sliceAxisRangeCoordinate]
  rw [← Tensor.getScalar_reshape_toFin, Tensor.reshape_reshape_self,
    Coord.linearize_unlinearize]

/-- The scatter of a slice layout is zero at every source entry the slice does not read. -/
theorem slice_scatter_outside {β : Type} [Storage β] [Zero β] (layout : SliceLayout)
    (coefficient : Tensor β [layout.outputShape.size]) (i : Fin layout.sourceShape.size)
    (hOutside : i ∉ Set.range layout.flatIndex) :
    (layout.scatter coefficient).getScalar i = 0 := by
  have hCoordinate : Coord.unlinearize i ∉
      Set.range (Tensor.sliceAxisRangeCoordinate ⟨layout.axis, layout.axisInBounds.proof⟩
        layout.start layout.length layout.start_add_length_le) := by
    rintro ⟨c, hc⟩
    apply hOutside
    refine ⟨Coord.linearize c, ?_⟩
    simpa only [SliceLayout.flatIndex, Coord.unlinearize_linearize,
      Coord.linearize_unlinearize] using
      congrArg (fun c : layout.sourceShape.Coord => Coord.linearize c) hc
  unfold SliceLayout.scatter
  rw [← Coord.linearize_unlinearize i, Tensor.getScalar_reshape_toFin]
  exact Tensor.sliceBackward_apply_eq_zero (α := β) ⟨layout.axis, layout.axisInBounds.proof⟩
    layout.start layout.length layout.start_add_length_le _ _ hCoordinate

/-- Pairing `a` with the slice `y` of `source` is pairing the scatter of `a`, zero off the slice,
with `source`. -/
theorem dot_slice (layout : SliceLayout) (a y source : Nat → ℝ)
    (hy : ∀ i : Fin layout.outputShape.size, y i.val = source (layout.flatIndex i).val) :
    dot layout.outputShape.size a y =
      dot layout.sourceShape.size
        (extendFin fun i =>
          (layout.scatter (Tensor.ofFn fun j => a j.val)).getScalar i) source := by
  simp only [dot, dotProduct, Function.comp_apply]
  apply Fintype.sum_of_injective layout.flatIndex (slice_flatIndex_injective layout)
  · intro i hi
    simp only [extendFin_val, slice_scatter_outside layout _ i hi, zero_mul]
  · intro i
    simp only [extendFin_val, slice_scatter_selected, Tensor.getScalar_ofFn, hy i]

/-- The scatter `scatterSliceBoxes?` of a box that encloses `a` encloses the scatter of `a`,
which is zero off the slice. -/
theorem scatterSliceBoxes?_encloses (layout : SliceLayout)
    {input output : FlatBox α} {a : Nat → ℝ}
    (ha : RowEncloses input layout.outputShape.size a)
    (houtput : scatterSliceBoxes? layout input = some output) :
    RowEncloses output layout.sourceShape.size
      (extendFin fun i =>
        (layout.scatter (Tensor.ofFn fun j => a j.val)).getScalar i) := by
  obtain ⟨dim, lo, hi⟩ := input
  obtain ⟨hdim, ha⟩ := ha
  dsimp only at hdim
  subst dim
  simp only [scatterSliceBoxes?, ↓reduceDIte, Option.some.injEq] at houtput
  subst output
  refine ⟨rfl, ?_⟩
  intro i
  simp only [Tensor.getAtOrZero_eq_getScalar, extendFin_val]
  by_cases hInside : i ∈ Set.range layout.flatIndex
  · obtain ⟨j, rfl⟩ := hInside
    simpa only [slice_scatter_selected, Tensor.getScalar_ofFn,
      Tensor.getAtOrZero_eq_getScalar] using ha j
  · simp only [slice_scatter_outside layout _ i hInside, LawfulBoundOps.toReal_zero,
      le_refl, and_self]

/-- Adding the scatter of a box that encloses `a` at the pending parent `p` of a slice node `id`
adds the pairing of `a` with the coordinates of `id` to the represented value. -/
theorem represents_slice {dims : Nat → Nat} {v : Nat → Nat → ℝ} {input k id p : Nat}
    {st : State α} {z : ℝ}
    (h : Represents dims v input k st z) (hp : p < st.coeffs.size)
    (hlive : p ∈ pending input k) (layout : SliceLayout)
    (hdim : dims p = layout.sourceShape.size)
    {aB aX : FlatBox α} {a : Nat → ℝ}
    (ha : RowEncloses aB layout.outputShape.size a)
    (hy : ∀ i : Fin layout.outputShape.size, v id i.val = v p (layout.flatIndex i).val)
    (hresult : scatterSliceBoxes? layout aB = some aX) :
    Represents dims v input k (addCoeff st p aX)
      (z + dot layout.outputShape.size a (v id)) := by
  have hbox := scatterSliceBoxes?_encloses layout ha hresult
  have hbox' : RowEncloses aX (dims p)
      (extendFin fun i =>
        (layout.scatter (Tensor.ofFn fun j => a j.val)).getScalar i) := by
    simpa only [hdim] using hbox
  have hrep := represents_add h p hp hlive aX _ hbox'
  rw [hdim, ← dot_slice layout a (v id) (v p) hy] at hrep
  exact hrep

/-- Adding a box that encloses `a` at the pending parent `pid` of a copy `id` adds the pairing of
`a` with the coordinates of `id` to the represented value. -/
theorem represents_copy {dims : Nat → Nat} {v : Nat → Nat → ℝ} {input k id pid : Nat}
    {st : State α} {z : ℝ}
    (h : Represents dims v input k st z) (hp : pid < st.coeffs.size)
    (hlive : pid ∈ pending input k) {aB : FlatBox α} {a : Nat → ℝ}
    (ha : RowEncloses aB (dims id) a) (heq : CopyEquation dims v id pid) :
    Represents dims v input k (addCoeff st pid aB)
      (z + dot (dims id) a (v id)) := by
  have hb : RowEncloses aB (dims pid) a := by simpa only [heq.1] using ha
  have hd : dot (dims pid) a (v pid) = dot (dims id) a (v id) := by
    rw [heq.1]
    simp only [dot, dotProduct, Function.comp_apply, heq.2]
  simpa only [hd] using represents_add h pid hp hlive aB a hb

/-- Adding a box that encloses `a` at both pending parents `p` and `q` of a sum `id` adds the
pairing of `a` with the coordinates of `id` to the represented value. -/
theorem represents_addParents {dims : Nat → Nat} {v : Nat → Nat → ℝ}
    {input k id p q : Nat} {st : State α} {z : ℝ}
    (h : Represents dims v input k st z) (hp : p < st.coeffs.size)
    (hq : q < st.coeffs.size) (hplive : p ∈ pending input k) (hqlive : q ∈ pending input k)
    {aB : FlatBox α} {a : Nat → ℝ} (ha : RowEncloses aB (dims id) a)
    (hdp : dims p = dims id) (hdq : dims q = dims id)
    (hy : ∀ i : Fin (dims id), v id i.val = v p i.val + v q i.val) :
    Represents dims v input k (addCoeff (addCoeff st p aB) q aB)
      (z + dot (dims id) a (v id)) := by
  have hap : RowEncloses aB (dims p) a := by simpa only [hdp] using ha
  have haq : RowEncloses aB (dims q) a := by simpa only [hdq] using ha
  have h₁ := represents_add h p hp hplive aB a hap
  have h₂ := represents_add h₁ q (by simpa using hq) hqlive aB a haq
  have hd : dot (dims id) a (v id) =
      dot (dims p) a (v p) + dot (dims q) a (v q) := by
    rw [hdp, hdq]
    simp only [dot, dotProduct, Function.comp_apply, hy, mul_add, Finset.sum_add_distrib]
  simpa only [hd, add_assoc] using h₂

/-- Adding a box that encloses `a` at the pending parent `p` of a difference `id` and its negation
at the pending parent `q` adds the pairing of `a` with the coordinates of `id` to the represented
value. The parents may coincide, and the rounded box need not be a point. -/
theorem represents_subParents
    {dims : Nat → Nat} {v : Nat → Nat → ℝ} {input k id p q : Nat}
    {st : State α} {z : ℝ}
    (h : Represents dims v input k st z) (hp : p < st.coeffs.size)
    (hq : q < st.coeffs.size) (hplive : p ∈ pending input k) (hqlive : q ∈ pending input k)
    {aB : FlatBox α} {a : Nat → ℝ} (ha : RowEncloses aB (dims id) a)
    (hdp : dims p = dims id) (hdq : dims q = dims id)
    (hy : ∀ i : Fin (dims id), v id i.val = v p i.val - v q i.val) :
    Represents dims v input k
      (addCoeff (addCoeff st p aB) q (boxNeg aB))
      (z + dot (dims id) a (v id)) := by
  have hap : RowEncloses aB (dims p) a := by simpa only [hdp] using ha
  have haq : RowEncloses (boxNeg aB) (dims q) (fun i => -a i) := by
    simpa only [hdq] using boxNeg_encloses ha
  have h₁ := represents_add h p hp hplive aB a hap
  have h₂ := represents_add h₁ q (by simpa using hq) hqlive
    (boxNeg aB) (fun i => -a i) haq
  have hd : dot (dims id) a (v id) =
      dot (dims p) a (v p) + dot (dims q) (fun i => -a i) (v q) := by
    rw [hdp, hdq]
    simp only [dot, dotProduct, Function.comp_apply, hy, mul_sub, Finset.sum_sub_distrib,
      neg_mul, Finset.sum_neg_distrib]
    ring
  simpa only [hd, add_assoc] using h₂

/-- Repeating a one-entry box that holds `a` over `n` entries gives a box that encloses `n` copies
of `a`, the coefficients that a sum over every axis passes to its parent. -/
theorem broadcast_encloses {n : Nat} (lo hi : Tensor α [1]) (a : ℝ)
    (ha : value (lo.getScalar 0) ≤ a ∧ a ≤ value (hi.getScalar 0)) :
    RowEncloses
      { dim := n
        lo := Tensor.full (α := α) (.dim n .scalar) (getAtOrZero lo [0])
        hi := Tensor.full (α := α) (.dim n .scalar) (getAtOrZero hi [0]) }
      n (fun _ => a) := by
  refine ⟨rfl, ?_⟩
  intro i
  simp only [Tensor.getAtOrZero_eq_getScalar, Tensor.getScalar_full]
  rw [show getAtOrZero lo [0] = lo.getScalar 0 from Tensor.getAtOrZero_eq_getScalar lo 0,
    show getAtOrZero hi [0] = hi.getScalar 0 from Tensor.getAtOrZero_eq_getScalar hi 0]
  exact ha

/-- Gathering the entries of a box that encloses `a` along a map `index` of indices gives a box
that encloses `a` gathered along `index`. -/
theorem gather_encloses {m n : Nat} (lo hi : Tensor α [m]) (a : Nat → ℝ)
    (ha : RowEncloses { dim := m, lo := lo, hi := hi } m a)
    (index : Fin n → Fin m) :
    RowEncloses
      { dim := n
        lo := Tensor.ofFn (fun i => lo.getScalar (index i))
        hi := Tensor.ofFn (fun i => hi.getScalar (index i)) }
      n (extendFin (fun i => a (index i).val)) := by
  refine ⟨rfl, ?_⟩
  intro i
  simpa only [Tensor.getAtOrZero_eq_getScalar, Tensor.getScalar_ofFn, extendFin_val] using
    ha.2 (index i)

/-- The inner product over the row-major entries of a shape is the sum over its coordinates. -/
theorem dot_eq_sum_coord (t : Shape) (a x : Nat → ℝ) :
    dot t.size a x =
      ∑ c : t.Coord, a (Coord.linearize c).val * x (Coord.linearize c).val :=
  (Fintype.sum_bijective _ (Function.bijective_iff_has_inverse.mpr ⟨Coord.unlinearize,
    Coord.unlinearize_linearize, Coord.linearize_unlinearize⟩) _ _ fun _ => rfl).symm

/-- Permuting the axes keeps the pairing of coefficients with values: coefficients `a` paired with
`permute x σ` are their permutation back by `Shape.unpermute` paired with `x`. -/
theorem sum_mul_permute {s out : Shape} (σ : Equiv.Perm (Fin s.rank)) (hout : out = s.permute σ)
    (a : Tensor ℝ out) (x : Tensor ℝ s) :
    ∑ c, a c * Tensor.permute x σ out hout c =
      ∑ c, Tensor.permute a (s.unpermute σ hout) s (s.permute_unpermute σ hout).symm c * x c := by
  have key : ∀ {f : out.Coord → s.Coord} {g : s.Coord → out.Coord}, (∀ c, g (f c) = c) →
      (∀ d, f (g d) = d) → ∑ c, a c * x (f c) = ∑ d, a (g d) * x d := fun hfg hgf =>
    Fintype.sum_bijective _ (Function.bijective_iff_has_inverse.mpr ⟨_, hfg, hgf⟩) _ _
      fun c => by rw [hfg]
  simp only [Tensor.permute_apply]
  subst hout
  refine key (fun c => Shape.Coord.ext_index fun k hk => ?_)
    (fun d => Shape.Coord.ext_index fun k hk => ?_) <;>
  · rw [Shape.Coord.index_ofFun, dite_eq_left hk]
    simp only [Fin.val_cast, Shape.Coord.val_get, Shape.Coord.index_reindexEq,
      Shape.Coord.index_ofFun]
    simp [Shape.unpermute, Equiv.permCongr_def]

/-- The sweep invariant at position `k`: the state has failed, or its table has size `size` and it
represents `z`. -/
def SweepInvariant (dims : Nat → Nat) (v : Nat → Nat → ℝ) (input size k : Nat)
    (z : ℝ) (st : State α) : Prop :=
  st.failed = true ∨ st.coeffs.size = size ∧ Represents dims v input k st z

/-- The initial state, with the box of the objective at the output node and the constant `0`,
satisfies the sweep invariant at position `size` for the value `obj · v output`. -/
theorem initial_represents
    (dims : Nat → Nat) (v : Nat → Nat → ℝ) (input size output : Nat)
    (houtput : output < size) (obj : FlatTensor α) (hdim : obj.n = dims output) :
    SweepInvariant dims v input size size
      (dot (dims output) (fun i => value (getAtOrZero obj.v [i])) (v output))
      { coeffs := (Array.replicate size none).set! output (some (FlatBox.ofTensor obj.v))
        cstLo := 0, cstHi := 0 } := by
  right
  refine ⟨by simp, _, 0, initial_encloses dims size output obj hdim, ?_⟩
  have hmem : output ∈ pending input size := by simp [pending, houtput]
  have hrow (id : Nat) :
      dot (dims id)
        (fun i => if id = output then value (getAtOrZero obj.v [i]) else 0) (v id) =
        if id = output then
          dot (dims output) (fun i => value (getAtOrZero obj.v [i])) (v output) else 0 := by
    by_cases h : id = output
    · subst id
      simp
    · simp [h, dot, dotProduct]
  simp only [frontierValue, hrow, Finset.sum_ite_eq', hmem, ↓reduceIte, add_zero]

/-- The forms that `CROWN.Directed.inputAffines` builds from boxes of width `n` bound
`dot n a x + c` from below and above, for every `x`, `a` and `c` that the input box, the coefficient
box and the constant interval enclose. -/
theorem inputAffines_row_encloses
    {n : Nat} {xB aB : FlatBox α} {x a : Nat → ℝ} {c : ℝ} {lo hi : α}
    (hx : RowEncloses xB n x) (ha : RowEncloses aB n a)
    (hc : value lo ≤ c ∧ c ≤ value hi) {lower upper : Linear α n 1}
    (hresult : inputAffines n xB aB lo hi = some (lower, upper)) :
    affineValue lower (fun i => x i.val) ≤ dot n a x + c ∧
      dot n a x + c ≤ affineValue upper (fun i => x i.val) := by
  obtain ⟨xdim, xlo, xhi⟩ := xB
  obtain ⟨adim, alo, ahi⟩ := aB
  obtain ⟨hxDim, hx⟩ := hx
  obtain ⟨haDim, ha⟩ := ha
  dsimp only at hxDim haDim
  subst xdim
  subst adim
  exact inputAffines_encloses xlo xhi alo ahi lo hi
    (fun i => x i.val) (fun i => a i.val) c
    (by simpa only [Tensor.getAtOrZero_eq_getScalar] using hx)
    (by simpa only [Tensor.getAtOrZero_eq_getScalar] using ha) hc hresult

/-- The input's row, read with a zero box of width `n` as default, encloses the input's
coefficients. -/
theorem inputRow_encloses
    {dims : Nat → Nat} {st : State α} {f : Nat → Nat → ℝ} {c : ℝ}
    (h : StateEncloses dims st f c) (input n : Nat)
    (hinput : input < st.coeffs.size) (hdim : dims input = n) :
    RowEncloses
      (st.coeffs[input]!.getD
        { dim := n
          lo := Tensor.full (α := α) (.dim n .scalar) 0
          hi := Tensor.full (α := α) (.dim n .scalar) 0 })
      n (f input) := by
  have hf := h.1 input hinput
  rw [hdim] at hf
  cases he : st.coeffs[input]! with
  | none =>
      simp only [he, Option.getD_none] at hf ⊢
      refine ⟨rfl, ?_⟩
      intro i
      simp only [Tensor.getAtOrZero_eq_getScalar, Tensor.getScalar_full,
        LawfulBoundOps.toReal_zero (α := α), hf i,
        le_refl, and_self]
  | some box =>
      simpa only [he, Option.getD_some] using hf

private theorem zip_finRange {β : Type} (parents : Array Nat) (n : Nat)
    (hsize : parents.size = n) (f : Fin n → β) :
    parents.zip ((Array.finRange n).map f) =
      (Array.finRange n).map (fun i => (parents[i.val]!, f i)) := by
  apply Array.ext
  · simp [hsize]
  · intro i hi hj
    simp only [Array.size_zip, Array.size_map, Array.size_finRange, hsize, Nat.min_self] at hi
    simp [getElem!_pos (c := parents) (i := i) (by omega)]

/-- A fold that adds a box enclosing `a i` at the pending node `parent i` for each `i` of a list
adds the sum of the pairings of `a i` with the coordinates of `parent i` to the represented
value. -/
theorem represents_fold {ι : Type} (indices : List ι) (parent : ι → Nat)
    (box : ι → FlatBox α) (a : ι → Nat → ℝ)
    {dims : Nat → Nat} {v : Nat → Nat → ℝ} {input k : Nat}
    {st : State α} {z : ℝ}
    (h : Represents dims v input k st z)
    (hparent : ∀ i, parent i < st.coeffs.size)
    (hlive : ∀ i, parent i ∈ pending input k)
    (hbox : ∀ i, RowEncloses (box i) (dims (parent i)) (a i)) :
    Represents dims v input k
      (indices.foldl (fun state i => addCoeff state (parent i) (box i)) st)
      (z + (indices.map (fun i => dot (dims (parent i)) (a i) (v (parent i)))).sum) := by
  induction indices generalizing st z with
  | nil => simpa using h
  | cons i indices ih =>
      have hadd := represents_add h (parent i) (hparent i) (hlive i) (box i) (a i) (hbox i)
      have hrest := ih hadd (fun j => by simpa using hparent j)
      simpa only [List.foldl_cons, List.map_cons, List.sum_cons, add_assoc] using hrest

/-- Pairing `a` with a concatenation `y` of the parents' coordinates is the sum, over the blocks of
the layout, of the block of `a` paired with the block's parent, one term per block even when a
parent repeats. -/
theorem dot_concat (layout : ConcatLayout) (parents : Array Nat)
    (a y : Nat → ℝ) (v : Nat → Nat → ℝ)
    (hy : ∀ (parent : Fin layout.lengths.length) (i : Fin (layout.parentShape parent).size),
      y (layout.flatEquiv ⟨parent, i⟩).val = v (parents[parent.val]!) i.val) :
    dot layout.outputShape.size a y =
      ∑ parent : Fin layout.lengths.length,
        dot (layout.parentShape parent).size
          (extendFin (fun i => a (layout.flatEquiv ⟨parent, i⟩).val))
          (v (parents[parent.val]!)) := by
  simp only [dot, dotProduct, Function.comp_apply]
  symm
  rw [← Fintype.sum_sigma']
  apply Fintype.sum_equiv layout.flatEquiv
  intro source
  rcases source with ⟨parent, i⟩
  simp only [extendFin_val, hy]

/-- Splitting a box that encloses `a` into the blocks of a concatenation
(`CROWN.Directed.splitCoeff`) and adding each block at its pending parent adds the pairing of `a`
with the concatenation `y` to the represented value. -/
theorem represents_concat {dims : Nat → Nat} {v : Nat → Nat → ℝ} {input k : Nat}
    {st : State α} {z : ℝ}
    (h : Represents dims v input k st z)
    (layout : ConcatLayout) (parents : Array Nat) (hsize : parents.size = layout.lengths.length)
    (hparent : ∀ p ∈ parents, p < st.coeffs.size)
    (hlive : ∀ p ∈ parents, p ∈ pending input k)
    (hdim : ∀ parent : Fin layout.lengths.length,
      dims (parents[parent.val]!) = (layout.parentShape parent).size)
    {aB : FlatBox α} {a y : Nat → ℝ}
    (ha : RowEncloses aB layout.outputShape.size a)
    (hy : ∀ (parent : Fin layout.lengths.length) (i : Fin (layout.parentShape parent).size),
      y (layout.flatEquiv ⟨parent, i⟩).val = v (parents[parent.val]!) i.val)
    {coefficients : Array (FlatBox α)}
    (hresult : splitCoeff layout aB = some coefficients) :
    Represents dims v input k
      ((parents.zip coefficients).foldl
        (fun state entry => addCoeff state entry.1 entry.2) st)
      (z + dot layout.outputShape.size a y) := by
  obtain ⟨adim, alo, ahi⟩ := aB
  obtain ⟨haDim, ha⟩ := ha
  dsimp only at haDim
  subst adim
  simp only [splitCoeff, ↓reduceDIte, castDimScalar_self,
    Option.some.injEq] at hresult
  subst coefficients
  let coefficient (parent : Fin layout.lengths.length) : FlatBox α :=
    { dim := (layout.parentShape parent).size
      lo := layout.split alo parent
      hi := layout.split ahi parent }
  let exactCoefficient (parent : Fin layout.lengths.length) : Nat → ℝ :=
    extendFin (fun i => a (layout.flatEquiv ⟨parent, i⟩).val)
  have hmem (parent : Fin layout.lengths.length) : parents[parent.val]! ∈ parents := by
    have hp : parent.val < parents.size := by omega
    rw [getElem!_pos (c := parents) (i := parent.val) hp]
    exact Array.getElem_mem hp
  have hb (parent : Fin layout.lengths.length) :
      RowEncloses (coefficient parent) (dims (parents[parent.val]!)) (exactCoefficient parent) := by
    rw [hdim parent]
    exact gather_encloses alo ahi a ⟨rfl, ha⟩ (fun i => layout.flatEquiv ⟨parent, i⟩)
  have hf := represents_fold (List.finRange layout.lengths.length)
    (fun parent => parents[parent.val]!) coefficient exactCoefficient h
    (fun parent => hparent _ (hmem parent)) (fun parent => hlive _ (hmem parent)) hb
  have hsum :
      ((List.finRange layout.lengths.length).map
        (fun parent => dot (dims (parents[parent.val]!))
          (exactCoefficient parent) (v (parents[parent.val]!)))).sum =
        dot layout.outputShape.size a y := by
    rw [← Fin.sum_univ_def]
    simp only [hdim]
    exact (dot_concat layout parents a y v hy).symm
  rw [hsum] at hf
  change Represents dims v input k
    ((parents.zip ((Array.finRange layout.lengths.length).map coefficient)).foldl
      (fun state entry => addCoeff state entry.1 entry.2) st)
    (z + dot layout.outputShape.size a y)
  rw [zip_finRange parents _ hsize coefficient, ← Array.foldl_toList, Array.toList_map,
    List.foldl_map]
  simpa only [Array.finRange, Array.toList_ofFn, List.finRange] using hf

/-- `bound` bounds `z` in direction `dir`: from below at `.lower`, from above at `.upper`. -/
def DirectionBound (dir : BackwardDir) (bound : α) (z : ℝ) : Prop :=
  match dir with
  | .lower => value bound ≤ z
  | .upper => z ≤ value bound

/-- The bound of `CROWN.Directed.consumeBox`, which pairs each entry of the objective with the end
of the box its sign selects, bounds `obj · x` in direction `dir` for every `x` the box encloses. -/
theorem consumeBox_encloses
    (dir : BackwardDir) (obj : FlatTensor α) (box : FlatBox α) (x : Nat → ℝ)
    (hx : RowEncloses box obj.n x) {bound : α}
    (hresult : consumeBox dir obj box = some bound) :
    DirectionBound dir bound
      (dot obj.n (fun i => value (getAtOrZero obj.v [i])) x) := by
  obtain ⟨n, coefficients⟩ := obj
  obtain ⟨m, lo, hi⟩ := box
  obtain ⟨hdim, hx⟩ := hx
  dsimp only at hdim
  subst m
  let product (direction : BackwardDir) (i : Fin n) : α :=
    let a := coefficients.getScalar i
    let l := lo.getScalar i
    let u := hi.getScalar i
    match direction with
    | .lower => BoundOps.mulDown a (if 0 < a then l else u)
    | .upper => BoundOps.mulUp a (if 0 < a then u else l)
  have hterms (i : Fin n) :
      value (product .lower i) ≤ value (coefficients.getScalar i) * x i.val ∧
        value (coefficients.getScalar i) * x i.val ≤ value (product .upper i) := by
    have hb := hx i
    simp only [Tensor.getAtOrZero_eq_getScalar] at hb
    by_cases ha : (0 : α) < coefficients.getScalar i
    · have hp : 0 ≤ value (coefficients.getScalar i) := by
        simpa only [LawfulBoundOps.toReal_zero (α := α)] using
          ((LawfulBoundOps.lt_iff _ _).mp ha).le
      simp only [product, ha, ↓reduceIte]
      exact ⟨(LawfulBoundOps.mulDown_le _ _).trans
          (mul_le_mul_of_nonneg_left hb.1 hp),
        (mul_le_mul_of_nonneg_left hb.2 hp).trans (LawfulBoundOps.le_mulUp _ _)⟩
    · have hn : value (coefficients.getScalar i) ≤ 0 := by
        rw [← (LawfulBoundOps.toReal_zero (α := α))]
        exact le_of_not_gt (fun h => ha ((LawfulBoundOps.lt_iff _ _).mpr h))
      simp only [product, ha, ↓reduceIte]
      exact ⟨(LawfulBoundOps.mulDown_le _ _).trans
          (mul_le_mul_of_nonpos_left hb.2 hn),
        (mul_le_mul_of_nonpos_left hb.1 hn).trans (LawfulBoundOps.le_mulUp _ _)⟩
  have hs := sum_encloses (product .lower) (product .upper)
    (fun i => value (coefficients.getScalar i) * x i.val) hterms
  cases dir <;>
    simp only [consumeBox, ↓reduceDIte, castDimScalar_self,
      Fin.foldl_eq_finRange_foldl, decide_eq_true_eq,
      Option.some.injEq] at hresult
  · subst bound
    simpa only [DirectionBound, dot, dotProduct, Function.comp_apply,
      Tensor.getAtOrZero_eq_getScalar, product, List.finRange] using hs.1
  · subst bound
    simpa only [DirectionBound, dot, dotProduct, Function.comp_apply,
      Tensor.getAtOrZero_eq_getScalar, product, List.finRange] using hs.2

/-- The constant form `CROWN.Directed.constantAffine n c` has the real of `c` as its value at every
point. -/
theorem constantAffine_value
    (n : Nat) (c : α) (x : Fin n → ℝ) :
    affineValue (constantAffine n c) x = value c := by
  simp [affineValue, constantAffine, Tensor.get2_full, (LawfulBoundOps.toReal_zero (α := α))]

/-- The constant form that `CROWN.Directed.outputBoxBound` builds from the box of the output node
bounds `obj · y` in direction `dir` at every point, when that box encloses `y`. -/
theorem outputBoxBound_encloses
    (dir : BackwardDir) (ibp : Array (Option (FlatBox α))) (output n : Nat)
    (obj : FlatTensor α) (y : Nat → ℝ)
    (hbox : ∀ box, ibp[output]! = some box → RowEncloses box obj.n y)
    {aff : Linear α n 1}
    (hresult : outputBoxBound dir ibp output n obj = some aff)
    (x : Fin n → ℝ) :
    match dir with
    | .lower => affineValue aff x ≤ dot obj.n (fun i => value (getAtOrZero obj.v [i])) y
    | .upper => dot obj.n (fun i => value (getAtOrZero obj.v [i])) y ≤ affineValue aff x := by
  simp only [outputBoxBound, Option.bind_eq_bind, Option.pure_def] at hresult
  cases hentry : ibp[output]? with
  | none => simp only [hentry] at hresult; cases hresult
  | some entry =>
    simp only [hentry] at hresult
    obtain ⟨box, hb, hresult⟩ := Option.bind_eq_some_iff.mp hresult
    obtain ⟨bound, hc, hresult⟩ := Option.bind_eq_some_iff.mp hresult
    cases Option.some.inj hresult
    rw [hb] at hentry
    have hlookup : ibp[output]! = some box := by
      obtain ⟨hindex, hget⟩ := Array.getElem?_eq_some_iff.mp hentry
      simpa only [getElem!_pos (c := ibp) (i := output) hindex] using hget
    have hs := consumeBox_encloses dir obj box y (hbox box hlookup) hc
    cases dir <;> simpa only [constantAffine_value, DirectionBound] using hs

/-- The affine forms of `bounds` enclose `y` at `x`: each coordinate of `y` lies between the values
of the lower and upper rows at `x`. -/
def AffineRowsEnclose (bounds : FlatAffineBounds α) (x y : Nat → ℝ) : Prop :=
  ∀ i : Fin bounds.outDim,
    (∑ j : Fin bounds.inDim, value (Tensor.get2 bounds.loAff.weights i j) * x j.val) +
        value (bounds.loAff.bias.getScalar i) ≤ y i.val ∧
      y i.val ≤
        (∑ j : Fin bounds.inDim, value (Tensor.get2 bounds.hiAff.weights i j) * x j.val) +
          value (bounds.hiAff.bias.getScalar i)

/-- A lower and an upper one-row form whose values at `x` bound `z` give affine bounds with one
output row that enclose `z` at `x`. -/
theorem scalarBounds_enclose {n : Nat} {lower upper : Linear α n 1}
    {x : Nat → ℝ} {z : ℝ}
    (h : affineValue lower (fun i => x i.val) ≤ z ∧
      z ≤ affineValue upper (fun i => x i.val)) :
    AffineRowsEnclose { inDim := n, outDim := 1, loAff := lower, hiAff := upper }
      x (fun _ => z) := by
  intro i
  have hi : i = 0 := Subsingleton.elim _ _
  subst i
  exact h

/-- The constant bounds `boundsConst` of a box that encloses `y` enclose `y` at every point. -/
theorem boundsConst_encloses
    (n : Nat) (box : FlatBox α) (x y : Nat → ℝ)
    (hy : RowEncloses box box.dim y) :
    AffineRowsEnclose (boundsConst n box.dim box.lo box.hi) x y := by
  dsimp only [AffineRowsEnclose, boundsConst]
  intro i
  simpa only [Tensor.get2_full, LawfulBoundOps.toReal_zero (α := α), zero_mul,
    Finset.sum_const_zero, zero_add, Tensor.getAtOrZero_eq_getScalar] using hy.2 i

/-- Evaluating affine bounds that enclose `y` at `x` on a box that encloses `x` gives a box that
encloses `y`. -/
theorem affineBoundsEval_encloses
    (bounds : FlatAffineBounds α) (box : FlatBox α) (x y : Nat → ℝ)
    (hx : RowEncloses box bounds.inDim x) (hy : AffineRowsEnclose bounds x y) :
    RowEncloses
      { dim := bounds.outDim
        lo := (bounds.evalOnFlatBox box hx.1).lo
        hi := (bounds.evalOnFlatBox box hx.1).hi }
      bounds.outDim y := by
  obtain ⟨n, lo, hi⟩ := box
  have hd := hx.1
  change n = bounds.inDim at hd
  subst n
  have hinput (j : Fin bounds.inDim) :
      value (lo.getScalar j) ≤ x j.val ∧ x j.val ≤ value (hi.getScalar j) := by
    simpa only [Tensor.getAtOrZero_eq_getScalar] using hx.2 j
  refine ⟨rfl, ?_⟩
  intro i
  have hl := affineEvalOnBox_encloses bounds.loAff
    { lo := lo, hi := hi } (fun j => x j.val) hinput i
  have hu := affineEvalOnBox_encloses bounds.hiAff
    { lo := lo, hi := hi } (fun j => x j.val) hinput i
  simpa only [Tensor.getAtOrZero_eq_getScalar, FlatAffineBounds.evalOnFlatBox,
    Linear.evalOnFlatBox,
    FlatBox.getScalarBox, FlatBox.loAsDim, FlatBox.hiAsDim, Tensor.castShape_rfl] using
      And.intro (hl.1.trans (hy i).1) ((hy i).2.trans hu.2)

/-- The box that `evalObjectiveBox?` gives from affine bounds that enclose `y` at `x` and an
input box that encloses `x` encloses `y`, as a box of width one. -/
theorem evalObjectiveBox_encloses
    (bounds : FlatAffineBounds α) (xB : FlatBox α) (inputDim : Nat)
    (x y : Nat → ℝ) (hx : RowEncloses xB bounds.inDim x)
    (hy : AffineRowsEnclose bounds x y) {result : FlatBox α}
    (hresult : evalObjectiveBox? bounds xB inputDim = .ok result) :
    RowEncloses result 1 y := by
  by_cases hi : bounds.inDim = inputDim
  · by_cases hxDim : xB.dim = inputDim
    · by_cases ho : bounds.outDim = 1
      · simp only [evalObjectiveBox?, hi, hxDim, ho, ↓reduceDIte] at hresult
        have he := Except.ok.inj hresult
        subst result
        have h := affineBoundsEval_encloses bounds xB x y hx hy
        cases bounds with
        | mk n m lower upper =>
            dsimp only at ho
            subst m
            simpa only [FlatAffineBounds.evalOnFlatBoxAsDim, Tensor.castShape_rfl] using h
      · simp [evalObjectiveBox?, hi, hxDim, ho] at hresult
    · simp [evalObjectiveBox?, hi, hxDim] at hresult
  · simp [evalObjectiveBox?, hi] at hresult

end
end TorchLean.CROWN.Directed

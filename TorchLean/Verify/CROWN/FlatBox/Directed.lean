/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.FlatBox.Transfer
public import TorchLean.Verify.CROWN.BoundOps.Interval
public import TorchLean.Verify.CROWN.BoundOps.Lawful
public import TorchLean.Tensor.List
public import TorchLean.Tensor.Numeric.Real

/-!
# Directed enclosure of the flat-box transfers

`RowEncloses box n f` says that a box of width `n` encloses the real coordinates `f` once its
directed endpoints are read as reals (`LawfulBoundOps.toReal`). The `_encloses` lemmas show that
the transfers of `TorchLean.Verify.CROWN.FlatBox.Transfer` keep this enclosure: arithmetic,
negation, ReLU, the unary enclosures, SiLU, directed sums (`sum_encloses`, `boxSum_encloses`), the
affine layer (`ibpLinearParams_encloses`) and the checked concatenation
(`concatFlatBoxes?_encloses`). The soundness proof of the rounded IBP pass in `CROWN.Op.Directed`
applies them at each node.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean.CROWN.Directed

open Tensor
open Graph (intervalMul_encloses)
open LawfulBoundOps (toReal_minimum toReal_maximum)
open scoped BigOperators

noncomputable section

open Graph

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

/-- `box` has width `n`, and for each `i < n` its endpoints at `i`, read through `toReal`, bound
`f i` from below and above. -/
def RowEncloses (box : FlatBox α) (n : Nat) (f : Nat → ℝ) : Prop :=
  box.dim = n ∧ ∀ i : Fin n,
    value (getAtOrZero box.lo [i.val]) ≤ f i.val ∧
      f i.val ≤ value (getAtOrZero box.hi [i.val])

/-- Enclosure depends only on the coordinates below the box width. -/
theorem RowEncloses.congr {box : FlatBox α} {n : Nat} {f g : Nat → ℝ}
    (h : RowEncloses box n f) (hfg : ∀ i : Fin n, g i.val = f i.val) : RowEncloses box n g :=
  ⟨h.1, fun i => by simpa only [hfg i] using h.2 i⟩

/-- The ReLU of an endpoint denotes the real ReLU of its value. -/
theorem value_relu (a : α) : value (Activation.relu a) = max (value a) 0 := by
  unfold Activation.relu
  split
  next h =>
    rw [LawfulBoundOps.toReal_eq_of_beq h, (LawfulBoundOps.toReal_zero (α := α)), max_self]
  next => rw [LawfulBoundOps.toReal_max, (LawfulBoundOps.toReal_zero (α := α))]

/-- `RowEncloses` at the width of the box, stated with `getScalar` in place of
`Tensor.getAtOrZero`. -/
theorem rowEncloses_iff {d : Nat} {lo hi : Tensor α [d]} {f : Nat → ℝ} :
    RowEncloses { dim := d, lo := lo, hi := hi } d f ↔
      ∀ i : Fin d, value (lo.getScalar i) ≤ f i.val ∧ f i.val ≤ value (hi.getScalar i) := by
  simp only [RowEncloses, Tensor.getAtOrZero_eq_getScalar, true_and]

/-- Interval addition encloses the sum of enclosed coordinates. -/
theorem boxAdd_encloses {x y : FlatBox α} {n : Nat} {f g : Nat → ℝ}
    (hx : RowEncloses x n f) (hy : RowEncloses y n g) :
    RowEncloses (boxAdd x y) n (fun i => f i + g i) := by
  obtain ⟨dx, lx, ux⟩ := x
  obtain ⟨dy, ly, uy⟩ := y
  have hdx := hx.1
  have hdy := hy.1
  change dx = n at hdx
  change dy = n at hdy
  subst hdx hdy
  rw [rowEncloses_iff] at hx hy
  simp only [boxAdd, dite_true]
  rw [rowEncloses_iff]
  intro i
  simp only [getScalar_zipWith]
  exact ⟨(LawfulBoundOps.addDown_le _ _).trans (add_le_add (hx i).1 (hy i).1),
    (add_le_add (hx i).2 (hy i).2).trans (LawfulBoundOps.le_addUp _ _)⟩

/-- Interval subtraction encloses the difference of enclosed coordinates. -/
theorem boxSub_encloses {x y : FlatBox α} {n : Nat} {f g : Nat → ℝ}
    (hx : RowEncloses x n f) (hy : RowEncloses y n g) :
    RowEncloses (boxSub x y) n (fun i => f i - g i) := by
  obtain ⟨dx, lx, ux⟩ := x
  obtain ⟨dy, ly, uy⟩ := y
  have hdx := hx.1
  have hdy := hy.1
  change dx = n at hdx
  change dy = n at hdy
  subst hdx hdy
  rw [rowEncloses_iff] at hx hy
  simp only [boxSub, dite_true]
  rw [rowEncloses_iff]
  intro i
  simp only [getScalar_zipWith]
  exact ⟨(LawfulBoundOps.subDown_le _ _).trans (sub_le_sub (hx i).1 (hy i).2),
    (sub_le_sub (hx i).2 (hy i).1).trans (LawfulBoundOps.le_subUp _ _)⟩

/-- Directed negation encloses the negation of enclosed coordinates. -/
theorem boxNeg_encloses {x : FlatBox α} {n : Nat} {f : Nat → ℝ}
    (hx : RowEncloses x n f) :
    RowEncloses (boxNeg x) n (fun i => -f i) := by
  obtain ⟨dx, lx, ux⟩ := x
  have hdx := hx.1
  change dx = n at hdx
  subst hdx
  rw [rowEncloses_iff] at hx
  simp only [boxNeg]
  rw [rowEncloses_iff]
  intro i
  simp only [getScalar_map]
  have hlo := LawfulBoundOps.subDown_le (0 : α) (ux.getScalar i)
  have hhi := LawfulBoundOps.le_subUp (0 : α) (lx.getScalar i)
  rw [(LawfulBoundOps.toReal_zero (α := α)), zero_sub] at hlo hhi
  exact ⟨hlo.trans (neg_le_neg (hx i).2), (neg_le_neg (hx i).1).trans hhi⟩

/-- The four-corner product box encloses the product of enclosed coordinates. -/
theorem boxMulElem_encloses {x y box : FlatBox α} {n : Nat} {f g : Nat → ℝ}
    (hx : RowEncloses x n f) (hy : RowEncloses y n g) (hbox : boxMulElem x y = some box) :
    RowEncloses box n (fun i => f i * g i) := by
  obtain ⟨dx, lx, ux⟩ := x
  obtain ⟨dy, ly, uy⟩ := y
  have hdx := hx.1
  have hdy := hy.1
  change dx = n at hdx
  change dy = n at hdy
  subst hdx hdy
  rw [rowEncloses_iff] at hx hy
  simp only [boxMulElem, dite_true, Option.some.injEq] at hbox
  subst hbox
  rw [rowEncloses_iff]
  intro i
  have h := intervalMul_encloses (hx i).1 (hx i).2 (hy i).1 (hy i).2
  simpa only [intervalMul, getScalar_ofFn] using h

/-- Endpoint ReLU encloses the real ReLU of enclosed coordinates. -/
theorem boxReLU_encloses {x : FlatBox α} {n : Nat} {f : Nat → ℝ}
    (hx : RowEncloses x n f) :
    RowEncloses (boxReLU x) n (fun i => max (f i) 0) := by
  obtain ⟨dx, lx, ux⟩ := x
  have hdx := hx.1
  change dx = n at hdx
  subst hdx
  rw [rowEncloses_iff] at hx
  simp only [boxReLU]
  rw [rowEncloses_iff]
  intro i
  simp only [getScalar_map, value_relu]
  exact ⟨max_le_max (hx i).1 le_rfl, max_le_max (hx i).2 le_rfl⟩

/-- If `enclose` satisfies `UnaryEnclosure F` and `boxUnaryEnclosure? enclose`, which applies it
coordinatewise, returns a box, that box encloses the image of `f` under `F`. -/
theorem boxUnaryEnclosure?_encloses [NonlinearBoundOps α]
    {F : ℝ → ℝ} {enclose : α → α → Option (α × α)} (hF : UnaryEnclosure F enclose)
    {x box : FlatBox α} {n : Nat} {f : Nat → ℝ}
    (hx : RowEncloses x n f) (hbox : boxUnaryEnclosure? enclose x = some box) :
    RowEncloses box n (fun i => F (f i)) := by
  obtain ⟨dx, lx, ux⟩ := x
  have hdx := hx.1
  change dx = n at hdx
  subst hdx
  rw [rowEncloses_iff] at hx
  unfold boxUnaryEnclosure? at hbox
  cases hb : Tensor.sequenceFinM fun i => enclose (lx.getScalar i) (ux.getScalar i) with
  | none => simp only [hb, Option.bind_eq_bind, Option.bind_none, reduceCtorEq] at hbox
  | some bounds =>
      simp only [hb, Option.bind_eq_bind, Option.bind_some, Option.pure_def,
        Option.some.injEq] at hbox
      subst hbox
      rw [rowEncloses_iff]
      intro i
      have hi := Tensor.sequenceFinM_get_of_eq_some hb i
      simpa only [getScalar_ofFn] using hF hi (hx i).1 (hx i).2

/-- The product of the input box and its sigmoid box (`ibpSiLU?`) encloses real SiLU, although
SiLU is not monotone. -/
theorem ibpSiLU?_encloses [NonlinearBoundOps α] [LawfulNonlinearBoundOps α]
    {x box : FlatBox α} {n : Nat} {f : Nat → ℝ}
    (hx : RowEncloses x n f) (hbox : ibpSiLU? x = some box) :
    RowEncloses box n (fun i => f i * Real.sigmoid (f i)) := by
  unfold ibpSiLU? at hbox
  cases hg : boxUnaryEnclosure? NonlinearBoundOps.sigmoidBounds x with
  | none => simp [hg] at hbox
  | some gate =>
      simp only [hg, Option.bind_eq_bind, Option.bind_some] at hbox
      exact boxMulElem_encloses hx
        (boxUnaryEnclosure?_encloses LawfulNonlinearBoundOps.sigmoidBounds_enclosure hx hg) hbox

/-- `divBounds` with the numerator interval `[1, 1]` encloses the reciprocal. -/
theorem unaryEnclosure_inv [NonlinearBoundOps α] [LawfulNonlinearBoundOps α] :
    UnaryEnclosure (α := α) (·⁻¹) (fun lo hi => NonlinearBoundOps.divBounds 1 1 lo hi) := by
  intro lo hi outLo outHi x h hlo hhi
  have hone : value (1 : α) = 1 := LawfulBoundOps.toReal_one
  have := LawfulNonlinearBoundOps.divBounds_enclosure (α := α) h
    (le_of_eq hone) (le_of_eq hone.symm) hlo hhi
  simpa only [one_div] using this

/-! ## Sums and affine rows -/

/-- If `initialLo` and `initialHi` bound `initial`, and `lo i` and `hi i` bound `f i` at each
index, then folding `addDown` and `addUp` over `indices` from these endpoints bounds `initial`
plus the sum of the `f i`. -/
theorem foldl_encloses {ι : Type} (indices : List ι) (lo hi : ι → α) (f : ι → ℝ)
    (initialLo initialHi : α) (initial : ℝ)
    (hlo : value initialLo ≤ initial) (hhi : initial ≤ value initialHi)
    (hterms : ∀ i ∈ indices, value (lo i) ≤ f i ∧ f i ≤ value (hi i)) :
    value (indices.foldl (fun acc i => BoundOps.addDown acc (lo i)) initialLo) ≤
        initial + (indices.map f).sum ∧
      initial + (indices.map f).sum ≤
        value (indices.foldl (fun acc i => BoundOps.addUp acc (hi i)) initialHi) := by
  induction indices generalizing initialLo initialHi initial with
  | nil => simpa using And.intro hlo hhi
  | cons i indices ih =>
      have ht := hterms i (by simp)
      have hnextLo := (LawfulBoundOps.addDown_le initialLo (lo i)).trans (add_le_add hlo ht.1)
      have hnextHi := (add_le_add hhi ht.2).trans (LawfulBoundOps.le_addUp initialHi (hi i))
      simpa only [List.foldl_cons, List.map_cons, List.sum_cons, add_assoc] using
        ih (BoundOps.addDown initialLo (lo i)) (BoundOps.addUp initialHi (hi i))
          (initial + f i) hnextLo hnextHi (fun j hj => hterms j (by simp [hj]))

/-- Directed summation over all coordinates encloses the finite real sum. -/
theorem sum_encloses
    {n : Nat} (lo hi : Fin n → α) (f : Fin n → ℝ)
    (hterms : ∀ i, value (lo i) ≤ f i ∧ f i ≤ value (hi i)) :
    value ((List.finRange n).foldl (fun acc i => BoundOps.addDown acc (lo i)) 0) ≤
        ∑ i, f i ∧
      ∑ i, f i ≤
        value ((List.finRange n).foldl (fun acc i => BoundOps.addUp acc (hi i)) 0) := by
  have h := foldl_encloses (List.finRange n) lo hi f 0 0 0
    (by simp [LawfulBoundOps.toReal_zero (α := α)]) (by simp [LawfulBoundOps.toReal_zero (α := α)])
    (fun i _ => hterms i)
  have hsum : ((List.finRange n).map f).sum = ∑ i, f i := (Fin.sum_univ_def f).symm
  simpa only [hsum, zero_add] using h

/-- For a real point `x` in `box`, row `i` of `aff.evalOnBox box` bounds row `i` of the affine
form at `x`, with its coefficients and offset read through `toReal`. -/
theorem affineEvalOnBox_encloses
    {n m : Nat} (aff : Linear α n m) (box : Box α (.dim n .scalar))
    (x : Fin n → ℝ)
    (hx : ∀ j, value (box.lo.getScalar j) ≤ x j ∧ x j ≤ value (box.hi.getScalar j))
    (i : Fin m) :
    value ((aff.evalOnBox box).lo.getScalar i) ≤
        (∑ j, value (Tensor.get2 aff.weights i j) * x j) + value (aff.bias.getScalar i) ∧
      (∑ j, value (Tensor.get2 aff.weights i j) * x j) + value (aff.bias.getScalar i) ≤
        value ((aff.evalOnBox box).hi.getScalar i) := by
  let lower (j : Fin n) := Context.minimum
    (BoundOps.mulDown (Tensor.get2 aff.weights i j) (box.lo.getScalar j))
    (BoundOps.mulDown (Tensor.get2 aff.weights i j) (box.hi.getScalar j))
  let upper (j : Fin n) := Context.maximum
    (BoundOps.mulUp (Tensor.get2 aff.weights i j) (box.lo.getScalar j))
    (BoundOps.mulUp (Tensor.get2 aff.weights i j) (box.hi.getScalar j))
  have ht (j : Fin n) :
      value (lower j) ≤ value (Tensor.get2 aff.weights i j) * x j ∧
        value (Tensor.get2 aff.weights i j) * x j ≤ value (upper j) := by
    have h := intervalMul_encloses
      (le_refl (value (Tensor.get2 aff.weights i j)))
        (le_refl (value (Tensor.get2 aff.weights i j)))
      (hx j).1 (hx j).2
    simpa only [intervalMul, toReal_minimum, toReal_maximum, min_self, max_self,
      lower, upper] using h
  have hs := sum_encloses lower upper
    (fun j => value (Tensor.get2 aff.weights i j) * x j) ht
  simp only [Linear.evalOnBox, IBP.linear, Box.point, Tensor.getScalar_dim]
  exact ⟨(LawfulBoundOps.addDown_le _ _).trans (add_le_add hs.1 le_rfl),
    (add_le_add hs.2 le_rfl).trans (LawfulBoundOps.le_addUp _ _)⟩

/-- `boxSum` encloses the sum of all enclosed coordinates in a box of width one. -/
theorem boxSum_encloses {x : FlatBox α} {f : Nat → ℝ} (hx : RowEncloses x x.dim f)
    {g : Nat → ℝ} (hg : g 0 = ∑ i : Fin x.dim, f i.val) :
    RowEncloses (boxSum x) 1 g := by
  obtain ⟨d, lx, ux⟩ := x
  rw [rowEncloses_iff] at hx
  have hs := sum_encloses (fun i => lx.getScalar i) (fun i => ux.getScalar i)
    (fun i : Fin d => f i.val) hx
  simp only [boxSum]
  rw [rowEncloses_iff]
  intro i
  have hi : i = 0 := Subsingleton.elim _ _
  subst hi
  simpa only [getScalar_full, Fin.val_zero, hg] using hs

/-- The linear transfer encloses the real affine image of an enclosed input. -/
theorem ibpLinearParams_encloses {p : SomeLinear α} {x box : FlatBox α}
    {f g : Nat → ℝ} (hx : RowEncloses x p.inDim f)
    (hg : ∀ i : Fin p.outDim,
      g i.val = (∑ j : Fin p.inDim, value (Tensor.get2 p.spec.weights i j) * f j.val) +
        value (p.spec.bias.getScalar i))
    (hbox : ibpLinearParams p x = some box) :
    RowEncloses box p.outDim g := by
  obtain ⟨n, m, ⟨W, b⟩⟩ := p
  obtain ⟨dx, lx, ux⟩ := x
  have hdx := hx.1
  change dx = n at hdx
  subst hdx
  rw [rowEncloses_iff] at hx
  simp only [ibpLinearParams, dite_true, Option.some.injEq] at hbox
  subst hbox
  change RowEncloses { dim := m, lo := _, hi := _ } m g
  rw [rowEncloses_iff]
  intro i
  have h := affineEvalOnBox_encloses (⟨W, b⟩ : Linear α dx m) ⟨lx, ux⟩
    (fun j => f j.val) hx i
  rw [hg i]
  exact h

/-- The real coordinates of a concatenation: the parents' coordinates `f parent` placed by
`ConcatLayout.concat`, the layout the executable transfer uses. -/
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

/-- If `concatFlatBoxes?` succeeds and each parent's box encloses that parent's coordinates, the
result encloses their concatenation `realConcat layout f`. -/
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
    obtain ⟨hd, hbox⟩ := Option.dite_none_right_eq_some.mp hbox
    obtain rfl := Option.some.inj hbox
    rw [rowEncloses_iff]
    intro i
    simp only [realConcat, Tensor.getAtOrZero_eq_getScalar, ConcatLayout.concat, getScalar_ofFn]
    exact rowEncloses_cast_coordinate
      (hparents (layout.flatEquiv.symm i).1 _
        (Tensor.sequenceFinM_get_of_eq_some hinputs (layout.flatEquiv.symm i).1))
      (hd _) (layout.flatEquiv.symm i).2
  · contradiction

/-- Reading a matrix entry commutes with reading the matrix through `toReal`. -/
theorem get2_map_toReal {m n : Nat} (W : Tensor α [m, n]) (i : Fin m) (j : Fin n) :
    Tensor.get2 (Tensor.map value W) i j = value (Tensor.get2 W i j) := by
  simp [Tensor.get2, Tensor.getScalar, Tensor.item]

/-- Over `ℝ`, `MathFunctions.sqrt (max z 0)` is `Real.sqrt z`. -/
theorem _root_.TorchLean.MathFunctions.sqrt_max_zero (z : ℝ) :
    MathFunctions.sqrt (max z 0) = Real.sqrt z := by
  rw [MathFunctions.sqrt_real]
  rcases le_total z 0 with h | h
  · rw [max_eq_right h, Real.sqrt_zero, Real.sqrt_eq_zero'.mpr h]
  · rw [max_eq_left h]

end

end TorchLean.CROWN.Directed

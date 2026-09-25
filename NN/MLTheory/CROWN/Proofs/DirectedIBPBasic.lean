/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.DirectedBackwardNodeBounds

/-!
# Directed forward interval transfers

Scalar, box, and parent-lookup lemmas shared by the rounded forward IBP proofs.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean NN.IR
open TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph.Internal
open NN.MLTheory.CROWN.IntervalLemmas (intervalMul_encloses value_min2 value_max2)
open scoped BigOperators

noncomputable section

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

/-! ## Scalar and box transfers -/

/-- Directed affine interval evaluation encloses each real affine row. -/
theorem affineEvalOnBox_encloses
    {n m : Nat} (aff : AffineVec α n m) (box : Box α (.dim n .scalar))
    (x : Fin n → ℝ)
    (hx : ∀ j, value (box.lo.getScalar j) ≤ x j ∧ x j ≤ value (box.hi.getScalar j))
    (i : Fin m) :
    value ((aff.evalOnBox box).lo.getScalar i) ≤
        (∑ j, value (Spec.get2 aff.A i j) * x j) + value (aff.c.getScalar i) ∧
      (∑ j, value (Spec.get2 aff.A i j) * x j) + value (aff.c.getScalar i) ≤
        value ((aff.evalOnBox box).hi.getScalar i) := by
  let lower (j : Fin n) := BoundOps.min2
    (BoundOps.mulDown (Spec.get2 aff.A i j) (box.lo.getScalar j))
    (BoundOps.mulDown (Spec.get2 aff.A i j) (box.hi.getScalar j))
  let upper (j : Fin n) := BoundOps.max2
    (BoundOps.mulUp (Spec.get2 aff.A i j) (box.lo.getScalar j))
    (BoundOps.mulUp (Spec.get2 aff.A i j) (box.hi.getScalar j))
  have ht (j : Fin n) :
      value (lower j) ≤ value (Spec.get2 aff.A i j) * x j ∧
        value (Spec.get2 aff.A i j) * x j ≤ value (upper j) := by
    have h := intervalMul_encloses
      (le_refl (value (Spec.get2 aff.A i j))) (le_refl (value (Spec.get2 aff.A i j)))
      (hx j).1 (hx j).2
    simpa only [intervalMul, value_min2, value_max2, min_self, max_self,
      lower, upper] using h
  have hs := sum_encloses lower upper
    (fun j => value (Spec.get2 aff.A i j) * x j) ht
  simp only [AffineVec.evalOnBox, Tensor.getScalar_dim]
  exact ⟨(LawfulBoundOps.addDown_le _ _).trans (add_le_add hs.1 le_rfl),
    (add_le_add hs.2 le_rfl).trans (LawfulBoundOps.le_addUp _ _)⟩

/-- The ReLU of an endpoint denotes the real ReLU of its value. -/
theorem value_relu (a : α) : value (Activation.Math.reluSpec a) = max (value a) 0 := by
  unfold Activation.Math.reluSpec
  split
  next h =>
    rw [LawfulBoundOps.toReal_eq_of_beq h, (LawfulBoundOps.toReal_zero (α := α)), max_self]
  next => rw [LawfulBoundOps.toReal_max, (LawfulBoundOps.toReal_zero (α := α))]

/-- Coordinate bounds of a flat box, read through the typed accessor. -/
theorem rowEncloses_iff {d : Nat} {lo hi : Tensor α [d]} {f : Nat → ℝ} :
    RowEncloses { dim := d, lo := lo, hi := hi } d f ↔
      ∀ i : Fin d, value (lo.getScalar i) ≤ f i.val ∧ f i.val ≤ value (hi.getScalar i) := by
  simp only [RowEncloses, read_fin, true_and]

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
  simp only [getScalar_map2Spec]
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
  simp only [getScalar_map2Spec]
  exact ⟨(LawfulBoundOps.subDown_le _ _).trans (sub_le_sub (hx i).1 (hy i).2),
    (sub_le_sub (hx i).2 (hy i).1).trans (LawfulBoundOps.le_subUp _ _)⟩

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
theorem boxRelu_encloses {x : FlatBox α} {n : Nat} {f : Nat → ℝ}
    (hx : RowEncloses x n f) :
    RowEncloses (boxRelu x) n (fun i => max (f i) 0) := by
  obtain ⟨dx, lx, ux⟩ := x
  have hdx := hx.1
  change dx = n at hdx
  subst hdx
  rw [rowEncloses_iff] at hx
  simp only [boxRelu]
  rw [rowEncloses_iff]
  intro i
  simp only [getScalar_mapSpec, value_relu]
  exact ⟨max_le_max (hx i).1 le_rfl, max_le_max (hx i).2 le_rfl⟩

/-- A lawful scalar enclosure, applied coordinatewise, encloses the pointwise image. -/
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
  cases hb : traverseFin fun i => enclose (lx.getScalar i) (ux.getScalar i) with
  | none => simp only [hb, Option.bind_eq_bind, Option.bind_none, reduceCtorEq] at hbox
  | some bounds =>
      simp only [hb, Option.bind_eq_bind, Option.bind_some, Option.pure_def,
        Option.some.injEq] at hbox
      subst hbox
      rw [rowEncloses_iff]
      intro i
      have hi := traverseFin_eq_some_iff.mp hb i
      simpa only [getScalar_ofFn] using hF hi (hx i).1 (hx i).2

/-- Reciprocal bounds are the division law with numerator `1`. -/
theorem unaryEnclosure_inv [NonlinearBoundOps α] [LawfulNonlinearBoundOps α] :
    UnaryEnclosure (α := α) (·⁻¹) (fun lo hi => NonlinearBoundOps.divBounds 1 1 lo hi) := by
  intro lo hi outLo outHi x h hlo hhi
  have hone : value (1 : α) = 1 := LawfulBoundOps.toReal_one
  have := LawfulNonlinearBoundOps.divBounds_enclosure (α := α) h
    (le_of_eq hone) (le_of_eq hone.symm) hlo hhi
  simpa only [one_div] using this

/-- Directed summation of all coordinates encloses the real sum. -/
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
theorem ibpLinearParams_encloses {p : LinParams α} {x box : FlatBox α}
    {f g : Nat → ℝ} (hx : RowEncloses x p.n f)
    (hg : ∀ i : Fin p.m,
      g i.val = (∑ j : Fin p.n, value (Spec.get2 p.w i j) * f j.val) + value (p.b.getScalar i))
    (hbox : ibpLinearParams p x = some box) :
    RowEncloses box p.m g := by
  obtain ⟨m, n, W, b⟩ := p
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
  have h := affineEvalOnBox_encloses (⟨W, b⟩ : AffineVec α dx m) ⟨lx, ux⟩
    (fun j => f j.val) hx i
  rw [hg i]
  exact h

/-- The bias-free matmul transfer encloses the real matrix-vector product. -/
theorem ibpMatmul_encloses {id : Nat} {ps : ParamStore α} {config : MatParams α}
    (hconfig : ps.matmulW[id]? = some config) {x box : FlatBox α}
    {f g : Nat → ℝ} (hx : RowEncloses x config.n f)
    (hg : ∀ i : Fin config.m,
      g i.val = (∑ j : Fin config.n, value (Spec.get2 config.w i j) * f j.val) +
        value ((Tensor.full (α := α) (.dim config.m .scalar) 0).getScalar i))
    (hbox : ibpMatmul id ps x = some box) :
    RowEncloses box config.m g := by
  simp only [ibpMatmul, hconfig] at hbox
  obtain ⟨m, n, W⟩ := config
  obtain ⟨dx, lx, ux⟩ := x
  have hdx := hx.1
  change dx = n at hdx
  subst hdx
  rw [rowEncloses_iff] at hx
  simp only [dite_true, Option.some.injEq] at hbox
  subst hbox
  change RowEncloses { dim := m, lo := _, hi := _ } m g
  rw [rowEncloses_iff]
  intro i
  have h := affineEvalOnBox_encloses
    (⟨W, Tensor.full (α := α) (.dim m .scalar) 0⟩ : AffineVec α dx m) ⟨lx, ux⟩
    (fun j => f j.val) hx i
  rw [hg i]
  exact h

/-- Enclosure depends only on the coordinates below the box width. -/
theorem RowEncloses.congr {box : FlatBox α} {n : Nat} {f g : Nat → ℝ}
    (h : RowEncloses box n f) (hfg : ∀ i : Fin n, g i.val = f i.val) : RowEncloses box n g :=
  ⟨h.1, fun i => by simpa only [hfg i] using h.2 i⟩

/-! ## One forward step -/

theorem unary_step {dims : Nat → Nat} {v : Nat → Nat → ℝ} {id p : Nat} {F : ℝ → ℝ}
    {transfer : FlatBox α → Option (FlatBox α)}
    (htransfer : ∀ {B box : FlatBox α} {n : Nat} {f : Nat → ℝ},
      RowEncloses B n f → transfer B = some box → RowEncloses box n (fun i => F (f i)))
    (heq : UnaryEquation dims v id p F) {B box : FlatBox α}
    (hB : RowEncloses B (dims p) (v p)) (hbox : transfer B = some box) :
    RowEncloses box (dims id) (v id) := by
  obtain ⟨hd, hv⟩ := heq
  rw [hd] at hB
  exact (htransfer hB hbox).congr hv

omit [BoundOps α] [LawfulBoundOps α] in
theorem unary_lookup {parents : Array Nat} {boxes : Array (Option (FlatBox α))}
    {k : FlatBox α → Option (FlatBox α)} {box : FlatBox α}
    (h : (do let p ← unaryParent? parents; k (← (boxes[p]?).join)) = some box) :
    ∃ p B, unaryParent? parents = some p ∧ (boxes[p]?).join = some B ∧ k B = some box := by
  cases hp : unaryParent? parents with
  | none => simp [hp] at h
  | some p =>
      cases hb : (boxes[p]?).join with
      | none => simp [hp, hb] at h
      | some B => exact ⟨p, B, rfl, hb, by simpa [hp, hb] using h⟩

omit [BoundOps α] [LawfulBoundOps α] in
theorem binary_lookup {parents : Array Nat} {boxes : Array (Option (FlatBox α))}
    {k : FlatBox α → FlatBox α → Option (FlatBox α)} {box : FlatBox α}
    (h : (do
      let (p, q) ← binaryParents? parents
      let x ← (boxes[p]?).join
      let y ← (boxes[q]?).join
      k x y) = some box) :
    ∃ p q X Y, binaryParents? parents = some (p, q) ∧ (boxes[p]?).join = some X ∧
      (boxes[q]?).join = some Y ∧ k X Y = some box := by
  cases hpq : binaryParents? parents with
  | none => simp [hpq] at h
  | some pq =>
      obtain ⟨p, q⟩ := pq
      cases hx : (boxes[p]?).join with
      | none => simp [hpq, hx] at h
      | some X =>
          cases hy : (boxes[q]?).join with
          | none => simp [hpq, hx, hy] at h
          | some Y => exact ⟨p, q, X, Y, rfl, hx, hy, by simpa [hpq, hx, hy] using h⟩


end

end NN.MLTheory.CROWN.Graph.DirectedBackward

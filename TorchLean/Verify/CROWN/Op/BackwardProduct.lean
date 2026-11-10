/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.Op.Backward

/-!
# Soundness of the McCormick product rules and of the CROWN queries on operator graphs

At a lawful scalar, each backward rule of direction `dir` bounds the real contribution of its node
in that direction (`backwardRule_sound`). The products `mul`, `matmul` and `scaledMatmul` bound each
term `a·(x·y)` linearly in one operand when the other's interval is a point. Otherwise a McCormick
plane bounds the term when the coefficient interval has one sign, and the interval product of the
operands bounds it when that interval changes sign (`productTerm_sound`). Every other operator falls
through to `directedRule_sound`. The planes are the bounds Shi et al. (2020, arXiv:2002.06622,
Sec. 3.3) give for `z = x·y`, and `productTerm_sound` gets them from `(x - lx)(y - ly) ≥ 0` and
`(x - lx)(uy - y) ≥ 0`, each multiplied by the coefficient or its negation, whichever is
nonnegative.

The CROWN queries of `Op.ExecCROWN` are sound at a supplied real point of the graph (`GraphPoint`)
whose every operator node satisfies its real equations (`Equation`, `ProductEquation`), whether or
not the query reaches the node or passes it by a fallback. The point and the equations are premises
here, and `Op.BackwardRun` derives them from a successful run of the graph. The queries are the
objective forms (`objectiveForms_encloses`, `backwardObjective?_encloses`), the affine bounds of a
matrix objective (`affineBounds?_encloses`), the bounds of a node (`nodeBounds?_encloses`) and the
boxes of a graph output (`backwardObjectiveBox?_encloses`, `outputBoxCROWN?_encloses`).
-/

@[expose] public section

open TorchLean.Shape (rank_eq_length size_append)

namespace TorchLean.CROWN.Op

open Tensor
open Graph
open CROWN.Directed

section MatmulLayoutBounds

section MatmulIndex
open Shape (Coord)

/-- Over a batch shape `S` that broadcasts to `T`, `batchIndex` sends each index of `T` to an index
of `S`. -/
theorem _root_.TorchLean.IR.MatmulDims.batchIndex_lt {S T : Shape}
    (h : Shape.CanBroadcastTo S T) {idx : Nat}
    (hidx : idx < T.size) : IR.MatmulDims.batchIndex S T idx < S.size := by
  obtain ⟨c, hc⟩ := Coord.linearize_surjective (s := T) ⟨idx, hidx⟩
  have hle := h.rank_le
  simp only [Shape.rank_eq_length] at hle
  let p : S.Coord := Shape.Coord.ofFun fun a =>
    if hi : S.extent a = 1 then ⟨0, by omega⟩
    else Fin.cast ((h.broadcastDims.extent_eq a).resolve_left hi).symm
      (Shape.Coord.get c (h.dims a))
  have hp : ∀ a < S.length, Shape.Coord.index p a =
      if S.getD a 1 = 1 then 0 else Shape.Coord.index c (a + (T.length - S.length)) := by
    intro a ha
    have ha' : a < S.rank := by rw [Shape.rank_eq_length]; exact ha
    have hext : S.extent ⟨a, ha'⟩ = S.getD a 1 := by
      simp [Shape.extent, List.getD_eq_getElem?_getD, ha]
    rw [Shape.Coord.index_ofFun, dite_eq_left ha', ← hext]
    split_ifs
    · rfl
    · rw [Fin.val_cast, Shape.Coord.val_get]
      simp [Shape.CanBroadcastTo.dims, Shape.rank_eq_length]
  have hval : idx = (Coord.linearize c).val := by rw [hc]
  rw [hval, IR.MatmulDims.batchIndex_linearize S h c p hp]
  exact (Coord.linearize p).isLt

end MatmulIndex

/-- The result of a matrix product layout has width `b.size * (m * p)`. -/
theorem matmulLayout_outShape_size {lv rv : Bool} {bA bB b : Shape} {m n p : Nat}
    (hl : lv = true → m = 1) (hr : rv = true → p = 1) :
    (matmulLayout lv rv bA bB b m n p).outShape.size = b.size * (m * p) := by
  cases lv <;> cases rv <;>
    simp_all [matmulLayout, Shape.size_append]

/-- The left operand of a matrix product layout has width `bA.size * (m * n)`. -/
theorem matmulLayout_leftShape_size {lv rv : Bool} {bA bB b : Shape} {m n p : Nat}
    (hl : lv = true → m = 1) :
    (matmulLayout lv rv bA bB b m n p).leftShape.size = bA.size * (m * n) := by
  simp only [matmulLayout]
  rw [Shape.size_matmulLeft hl, Shape.size_append]
  simp

/-- The right operand of a matrix product layout has width `bB.size * (n * p)`. -/
theorem matmulLayout_rightShape_size {lv rv : Bool} {bA bB b : Shape} {m n p : Nat}
    (hr : rv = true → p = 1) :
    (matmulLayout lv rv bA bB b m n p).rightShape.size = bB.size * (n * p) := by
  simp only [matmulLayout]
  rw [Shape.size_matmulRight hr, Shape.size_append]
  simp

/-- When `bA` broadcasts to `b`, a matrix product layout reads a left operand coordinate in range
at every result coordinate `o` and contraction index `t < n`. -/
theorem matmulLayout_leftIndex_lt {lv rv : Bool} {bA bB b : Shape} {m n p : Nat}
    (hl : lv = true → m = 1) (hr : rv = true → p = 1) (hA : Shape.CanBroadcastTo bA b)
    {o t : Nat} (ho : o < (matmulLayout lv rv bA bB b m n p).outShape.size) (ht : t < n) :
    (matmulLayout lv rv bA bB b m n p).leftIndex o t <
      (matmulLayout lv rv bA bB b m n p).leftShape.size := by
  rw [matmulLayout_outShape_size hl hr] at ho
  rw [matmulLayout_leftShape_size hl]
  simp only [IR.MatmulDims.leftIndex, matmulLayout]
  have hmp : 0 < m * p := by
    rcases Nat.eq_zero_or_pos (m * p) with h0 | h0
    · rw [h0, Nat.mul_zero] at ho; omega
    · exact h0
  have hp0 : 0 < p := Nat.pos_of_mul_pos_left hmp
  have hB : o / (m * p) < b.size := (Nat.div_lt_iff_lt_mul hmp).2 ho
  have hbi := IR.MatmulDims.batchIndex_lt hA hB
  have hrow : o % (m * p) / p < m :=
    (Nat.div_lt_iff_lt_mul hp0).2 (Nat.mod_lt _ hmp)
  generalize IR.MatmulDims.batchIndex bA b (o / (m * p)) = bi at hbi ⊢
  generalize o % (m * p) / p = row at hrow ⊢
  have h1 : row * n + t < m * n := by
    have : (row + 1) * n ≤ m * n := Nat.mul_le_mul_right n hrow
    rw [Nat.add_mul, Nat.one_mul] at this
    omega
  have h2 : (bi + 1) * (m * n) ≤ bA.size * (m * n) := Nat.mul_le_mul_right _ hbi
  rw [Nat.add_mul, Nat.one_mul] at h2
  omega

/-- When `bB` broadcasts to `b`, a matrix product layout reads a right operand coordinate in range
at every result coordinate `o` and contraction index `t < n`. -/
theorem matmulLayout_rightIndex_lt {lv rv : Bool} {bA bB b : Shape} {m n p : Nat}
    (hl : lv = true → m = 1) (hr : rv = true → p = 1) (hB : Shape.CanBroadcastTo bB b)
    {o t : Nat} (ho : o < (matmulLayout lv rv bA bB b m n p).outShape.size) (ht : t < n) :
    (matmulLayout lv rv bA bB b m n p).rightIndex o t <
      (matmulLayout lv rv bA bB b m n p).rightShape.size := by
  rw [matmulLayout_outShape_size hl hr] at ho
  rw [matmulLayout_rightShape_size hr]
  simp only [IR.MatmulDims.rightIndex, matmulLayout]
  have hmp : 0 < m * p := by
    rcases Nat.eq_zero_or_pos (m * p) with h0 | h0
    · rw [h0, Nat.mul_zero] at ho; omega
    · exact h0
  have hp0 : 0 < p := Nat.pos_of_mul_pos_left hmp
  have hb : o / (m * p) < b.size := (Nat.div_lt_iff_lt_mul hmp).2 ho
  have hbi := IR.MatmulDims.batchIndex_lt hB hb
  have hcol : o % p < p := Nat.mod_lt _ hp0
  generalize IR.MatmulDims.batchIndex bB b (o / (m * p)) = bi at hbi ⊢
  generalize o % p = col at hcol ⊢
  have h1 : t * p + col < n * p := by
    have : (t + 1) * p ≤ n * p := Nat.mul_le_mul_right p ht
    rw [Nat.add_mul, Nat.one_mul] at this
    omega
  have h2 : (bi + 1) * (n * p) ≤ bB.size * (n * p) := Nat.mul_le_mul_right _ hbi
  rw [Nat.add_mul, Nat.one_mul] at h2
  omega

end MatmulLayoutBounds

section Lawful

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

/-- Adding `d` on the left of both values keeps a bound in direction `dir`. -/
theorem DirBound.add_left {dir : BackwardDir} {a b : ℝ} (h : DirBound dir a b) (d : ℝ) :
    DirBound dir (d + a) (d + b) := by
  cases dir <;> simp only [DirBound] at h ⊢ <;> linarith

/-- An accumulator of product terms bounds `z` in direction `dir` at operand coordinates `x`
(`nx` entries) and `y` (`ny` entries) when its tables and its constant enclose real coefficients
and a real constant whose value at `x` and `y` bounds `z` in that direction. -/
def AccBound (dir : BackwardDir) (acc : PartialCoeffs α × PartialCoeffs α × (α × α))
    (nx ny : Nat) (x y : Nat → ℝ) (z : ℝ) : Prop :=
  ∃ fx fy cr, PartialEncloses acc.1 nx fx ∧ PartialEncloses acc.2.1 ny fy ∧
    (value acc.2.2.1 ≤ cr ∧ cr ≤ value acc.2.2.2) ∧
    DirBound dir (Directed.dot nx fx x + Directed.dot ny fy y + cr) z

/-- Empty tables with a constant interval that holds `cr` bound `cr` in either direction. -/
theorem accBound_replicate (dir : BackwardDir) (nx ny : Nat) (x y : Nat → ℝ) {c : α × α} {cr : ℝ}
    (hc : value c.1 ≤ cr ∧ cr ≤ value c.2) :
    AccBound dir (Array.replicate nx none, Array.replicate ny none, c) nx ny x y cr :=
  ⟨_, _, cr, partialEncloses_replicate nx, partialEncloses_replicate ny, hc, by
    simp only [Directed.dot, Function.comp_def, zero_dotProduct', zero_add]
    exact DirBound.refl dir cr⟩

/-- The interval `negProduct e f` holds `-(e * f)`, read as reals. -/
theorem negProduct_encloses (e f : α) :
    value (negProduct e f).1 ≤ -(value e * value f) ∧
      -(value e * value f) ≤ value (negProduct e f).2 := by
  simp only [negProduct]
  constructor
  · linarith [LawfulBoundOps.subDown_le (0 : α) (BoundOps.mulUp e f),
      LawfulBoundOps.le_mulUp e f, LawfulBoundOps.toReal_zero (α := α)]
  · linarith [LawfulBoundOps.le_subUp (0 : α) (BoundOps.mulDown e f),
      LawfulBoundOps.mulDown_le e f, LawfulBoundOps.toReal_zero (α := α)]

/-- Intervals that hold `cr` and `d` add, rounding outward, to an interval that holds `cr + d`. -/
theorem addPair_encloses {c v : α × α} {cr d : ℝ} (hc : value c.1 ≤ cr ∧ cr ≤ value c.2)
    (hv : value v.1 ≤ d ∧ d ≤ value v.2) :
    value (BoundOps.addDown c.1 v.1) ≤ cr + d ∧ cr + d ≤ value (BoundOps.addUp c.2 v.2) :=
  ⟨by linarith [LawfulBoundOps.addDown_le c.1 v.1],
    by linarith [LawfulBoundOps.le_addUp c.2 v.2]⟩

/-- A value that is not below zero is nonnegative as a real. -/
theorem toReal_nonneg_of_not_lt {a : α} (h : ¬ a < 0) : 0 ≤ value a := by
  rw [LawfulBoundOps.lt_iff, LawfulBoundOps.toReal_zero] at h
  exact not_lt.mp h

/-- A value that is not above zero is nonpositive as a real. -/
theorem toReal_nonpos_of_not_lt {a : α} (h : ¬ 0 < a) : value a ≤ 0 := by
  rw [LawfulBoundOps.lt_iff, LawfulBoundOps.toReal_zero] at h
  exact not_lt.mp h

/-- One product term keeps the bound: with its coefficient's interval holding `ar` and the operand
boxes holding `x i` and `y j`, the accumulator after the term bounds `z + ar · x i · y j` in
direction `dir`. -/
theorem productTerm_sound (dir : BackwardDir)
    {acc : PartialCoeffs α × PartialCoeffs α × (α × α)} {nx ny : Nat} {x y : Nat → ℝ}
    {z : ℝ} (h : AccBound dir acc nx ny x y z) {a : α × α} {ar : ℝ}
    (ha : value a.1 ≤ ar ∧ ar ≤ value a.2) {i j : Nat} (hi : i < nx) (hj : j < ny)
    {lx ux ly uy : α} (hx : value lx ≤ x i ∧ x i ≤ value ux)
    (hy : value ly ≤ y j ∧ y j ≤ value uy) :
    AccBound dir (productTerm dir acc a i j lx ux ly uy) nx ny x y
      (z + ar * (x i * y j)) := by
  obtain ⟨cx, cy, c⟩ := acc
  obtain ⟨fx, fy, cr, hfx, hfy, hc, hb⟩ := h
  have hb' := hb.add_right (ar * (x i * y j))
  unfold productTerm
  dsimp only
  split_ifs with h1 h2 h3 h4
  · -- `y` is a point.
    have hyy : y j = value ly := by
      simp only [Bool.not_eq_true', decide_eq_false_iff_not] at h1
      rw [LawfulBoundOps.lt_iff] at h1
      linarith [hy.1, hy.2]
    have hm := Graph.intervalMul_encloses ha.1 ha.2 (le_refl (value ly))
      (le_refl (value ly))
    refine ⟨Function.update fx i (fx i + ar * value ly), fy, cr, accum_encloses hfx hi hm, hfy,
      hc, ?_⟩
    convert hb' using 1
    rw [dot_bump hi, hyy]
    ring
  · -- `x` is a point.
    have hxx : x i = value lx := by
      simp only [Bool.not_eq_true', decide_eq_false_iff_not] at h2
      rw [LawfulBoundOps.lt_iff] at h2
      linarith [hx.1, hx.2]
    have hm := Graph.intervalMul_encloses ha.1 ha.2 (le_refl (value lx))
      (le_refl (value lx))
    refine ⟨fx, Function.update fy j (fy j + ar * value lx), cr, hfx, accum_encloses hfy hj hm,
      hc, ?_⟩
    convert hb' using 1
    rw [dot_bump hj, hxx]
    ring
  · -- One sign, the plane through `(lx, ly)`.
    dsimp only
    have hmx := Graph.intervalMul_encloses ha.1 ha.2 (le_refl (value ly))
      (le_refl (value ly))
    have hmy := Graph.intervalMul_encloses ha.1 ha.2 (le_refl (value lx))
      (le_refl (value lx))
    have hr := negProduct_encloses lx ly
    have hmc := Graph.intervalMul_encloses ha.1 ha.2 hr.1 hr.2
    refine ⟨Function.update fx i (fx i + ar * value ly),
      Function.update fy j (fy j + ar * value lx),
      cr + ar * -(value lx * value ly), accum_encloses hfx hi hmx, accum_encloses hfy hj hmy,
      addPair_encloses hc hmc, ?_⟩
    rw [dot_bump hi, dot_bump hj]
    simp only [Bool.or_eq_true, Bool.not_eq_true', decide_eq_false_iff_not] at h3
    cases dir
    · simp only [Bool.not_eq_true', decide_eq_false_iff_not] at h4
      have har : 0 ≤ ar := (toReal_nonneg_of_not_lt h4).trans ha.1
      simp only [DirBound] at hb ⊢
      nlinarith [mul_nonneg (mul_nonneg har (sub_nonneg.2 hx.1)) (sub_nonneg.2 hy.1)]
    · simp only [Bool.not_not, decide_eq_true_eq] at h4
      have har : ar ≤ 0 := ha.2.trans (toReal_nonpos_of_not_lt (h3.resolve_left (not_not.2 h4)))
      simp only [DirBound] at hb ⊢
      nlinarith [mul_nonneg (mul_nonneg (neg_nonneg.2 har) (sub_nonneg.2 hx.1))
        (sub_nonneg.2 hy.1)]
  · -- One sign, the plane through `(lx, uy)`.
    dsimp only
    have hmx := Graph.intervalMul_encloses ha.1 ha.2 (le_refl (value uy))
      (le_refl (value uy))
    have hmy := Graph.intervalMul_encloses ha.1 ha.2 (le_refl (value lx))
      (le_refl (value lx))
    have hr := negProduct_encloses lx uy
    have hmc := Graph.intervalMul_encloses ha.1 ha.2 hr.1 hr.2
    refine ⟨Function.update fx i (fx i + ar * value uy),
      Function.update fy j (fy j + ar * value lx),
      cr + ar * -(value lx * value uy), accum_encloses hfx hi hmx, accum_encloses hfy hj hmy,
      addPair_encloses hc hmc, ?_⟩
    rw [dot_bump hi, dot_bump hj]
    simp only [Bool.or_eq_true, Bool.not_eq_true', decide_eq_false_iff_not] at h3
    cases dir
    · simp only [Bool.not_eq_true', decide_eq_false_iff_not, not_not] at h4
      have har : ar ≤ 0 := ha.2.trans (toReal_nonpos_of_not_lt (h3.resolve_left (not_not.2 h4)))
      simp only [DirBound] at hb ⊢
      nlinarith [mul_nonneg (mul_nonneg (neg_nonneg.2 har) (sub_nonneg.2 hx.1))
        (sub_nonneg.2 hy.2)]
    · simp only [Bool.not_not, decide_eq_true_eq] at h4
      have har : 0 ≤ ar := (toReal_nonneg_of_not_lt h4).trans ha.1
      simp only [DirBound] at hb ⊢
      nlinarith [mul_nonneg (mul_nonneg har (sub_nonneg.2 hx.1)) (sub_nonneg.2 hy.2)]
  · -- Both signs: the term against the operand boxes.
    have hz := Graph.intervalMul_encloses hx.1 hx.2 hy.1 hy.2
    have hmc := Graph.intervalMul_encloses ha.1 ha.2 hz.1 hz.2
    refine ⟨fx, fy, cr + ar * (x i * y j), hfx, hfy, addPair_encloses hc hmc, ?_⟩
    convert hb' using 1
    ring

/-- A fold of steps that each add their term `g e` to the bounded value adds the terms' sum. -/
theorem foldl_accBound {β : Type} (dir : BackwardDir) {nx ny : Nat} {x y : Nat → ℝ}
    (step : PartialCoeffs α × PartialCoeffs α × (α × α) → β →
      PartialCoeffs α × PartialCoeffs α × (α × α)) (g : β → ℝ) (l : List β)
    (hstep : ∀ e ∈ l, ∀ acc z, AccBound dir acc nx ny x y z →
      AccBound dir (step acc e) nx ny x y (z + g e)) :
    ∀ acc z, AccBound dir acc nx ny x y z →
      AccBound dir (l.foldl step acc) nx ny x y (z + (l.map g).sum) := by
  induction l with
  | nil => intro acc z h; simpa using h
  | cons e l ih =>
    intro acc z h
    have h' := ih (fun e' he' => hstep e' (List.mem_cons_of_mem e he')) _ _
      (hstep e (List.mem_cons_self ..) acc z h)
    simpa only [List.foldl_cons, List.map_cons, List.sum_cons, add_assoc] using h'

/-- Applying an accumulator that bounds `w` at the coordinates of nodes `p` and `q` (earlier than
`k`) to a state that represents `z` gives a state whose represented value bounds `z + w`. -/
theorem applyProduct_sound (dir : BackwardDir) {dims : Nat → Nat} {v : Nat → Nat → ℝ}
    {input k : Nat} {st : State α} {z : ℝ} (h : Represents dims v input k st z)
    (hk : k < st.coeffs.size) {p q : Nat} (hp : p < k) (hq : q < k) {nx ny : Nat}
    (hnx : nx = dims p) (hny : ny = dims q)
    {acc : PartialCoeffs α × PartialCoeffs α × (α × α)} {w : ℝ}
    (hacc : AccBound dir acc nx ny (v p) (v q) w) :
    DirInvariant dir dims v input st.coeffs.size k (z + w)
      (applyProduct st p q nx ny acc) := by
  subst hnx hny
  obtain ⟨cx, cy, c⟩ := acc
  obtain ⟨fx, fy, cr, hfx, hfy, hc, hb⟩ := hacc
  have hb' : DirBound dir (z +
      (Directed.dot (dims p) fx (v p) + Directed.dot (dims q) fy (v q) + cr)) (z + w) :=
    hb.add_left z
  refine ⟨_, hb', Or.inr ?_⟩
  unfold applyProduct
  dsimp only
  rcases hB1 : coeffBox (dims p) cx with _ | B1 <;>
    rcases hB2 : coeffBox (dims q) cy with _ | B2 <;> dsimp only
  · rw [coeffBox_none hfx hB1, coeffBox_none hfy hB2, zero_add, zero_add]
    exact ⟨rfl, represents_addConstant h c.1 c.2 cr hc⟩
  · rw [coeffBox_none hfx hB1, zero_add]
    refine ⟨addCoeff_size .., ?_⟩
    have h2 := represents_add h q (by omega) (mem_pending_of_lt hq) B2 fy
      (coeffBox_encloses hfy hB2)
    simpa only [add_assoc] using represents_addConstant h2 c.1 c.2 cr hc
  · rw [coeffBox_none hfy hB2, add_zero]
    refine ⟨addCoeff_size .., ?_⟩
    have h1 := represents_add h p (by omega) (mem_pending_of_lt hp) B1 fx
      (coeffBox_encloses hfx hB1)
    have := represents_addConstant h1 c.1 c.2 cr hc
    rwa [add_assoc] at this
  · refine ⟨(addCoeff_size ..).trans (addCoeff_size ..), ?_⟩
    have h1 := represents_add h p (by omega) (mem_pending_of_lt hp) B1 fx
      (coeffBox_encloses hfx hB1)
    have h2 := represents_add h1 q (by simp; omega) (mem_pending_of_lt hq) B2 fy
      (coeffBox_encloses hfy hB2)
    have := represents_addConstant h2 c.1 c.2 cr hc
    simpa only [add_assoc] using this

/-- The real equation of a matrix product node `id` at the layout `D`: each result coordinate is
the sum over the contraction of the products of the operand coordinates the layout pairs. -/
def MatmulEquation (dims : Nat → Nat) (v : Nat → Nat → ℝ) (id p q : Nat)
    (D : IR.MatmulDims) : Prop :=
  ∀ o : Fin (dims id), v id o.val =
    ∑ t : Fin D.inner, v p (D.leftIndex o.val t.val) * v q (D.rightIndex o.val t.val)

/-- From an accumulator that bounds `z`, the product terms of a matrix product at layout `D` bound
`z` plus the sum over result coordinates of the coefficient `ar` times the contraction. Each
coefficient interval must hold its `ar` (a missing coefficient has `ar` zero), every pair the layout
reads must be in range and the boxes must enclose the operands. -/
theorem matmulTerms_sound (dir : BackwardDir) {D : IR.MatmulDims}
    {a : Fin D.outShape.size → Option (α × α)} {L R : FlatBox α} {nx ny : Nat}
    {x y : Nat → ℝ} (ar : Nat → ℝ)
    (ha : ∀ o : Fin D.outShape.size, ∀ ao, a o = some ao →
      value ao.1 ≤ ar o.val ∧ ar o.val ≤ value ao.2)
    (hnone : ∀ o : Fin D.outShape.size, a o = none → ar o.val = 0)
    (hidx : ∀ o < D.outShape.size, ∀ t < D.inner,
      D.leftIndex o t < nx ∧ D.rightIndex o t < ny)
    (hL : RowEncloses L nx x) (hR : RowEncloses R ny y)
    {acc : PartialCoeffs α × PartialCoeffs α × (α × α)} {z : ℝ}
    (hacc : AccBound dir acc nx ny x y z) :
    AccBound dir (matmulTerms dir D a L R acc) nx ny x y
      (z + ∑ o : Fin D.outShape.size, ar o.val *
        ∑ t : Fin D.inner, x (D.leftIndex o.val t.val) * y (D.rightIndex o.val t.val)) := by
  unfold matmulTerms
  rw [Fin.sum_univ_def]
  refine foldl_accBound dir _ (fun o : Fin D.outShape.size => ar o.val *
    ∑ t : Fin D.inner, x (D.leftIndex o.val t.val) * y (D.rightIndex o.val t.val)) _ ?_ _ _ hacc
  intro o _ acc' z' h'
  dsimp only
  rcases hao : a o with _ | ao
  · dsimp only
    rw [hnone o hao, zero_mul, add_zero]
    exact h'
  · dsimp only
    rw [Finset.mul_sum, show (∑ t : Fin D.inner, ar o.val *
        (x (D.leftIndex o.val t.val) * y (D.rightIndex o.val t.val))) =
        ((List.range D.inner).map fun t => ar o.val *
          (x (D.leftIndex o.val t) * y (D.rightIndex o.val t))).sum from
        (by rw [Fin.sum_univ_def, ← List.map_coe_finRange_eq_range, List.map_map]; rfl)]
    refine foldl_accBound dir _ _ _ ?_ _ _ h'
    intro t ht acc'' z'' h''
    rw [List.mem_range] at ht
    obtain ⟨hi, hj⟩ := hidx o.val o.isLt t ht
    exact productTerm_sound dir h'' (ha o ao hao) hi hj (hL.2 ⟨_, hL.1 ▸ hi⟩)
      (hR.2 ⟨_, hR.1 ▸ hj⟩)

/-- When the boxes enclose the operands and every coordinate the layout reads is in range, the
interval matrix product at layout `D` encloses the contractions. -/
theorem binaryMatmulBox_rowEncloses (D : IR.MatmulDims) {L R : FlatBox α}
    {nx ny : Nat} {x y : Nat → ℝ}
    (hidx : ∀ o < D.outShape.size, ∀ t < D.inner,
      D.leftIndex o t < nx ∧ D.rightIndex o t < ny)
    (hL : RowEncloses L nx x) (hR : RowEncloses R ny y) :
    RowEncloses (binaryMatmulBox D L R) D.outShape.size
      (fun o => ∑ t : Fin D.inner, x (D.leftIndex o t.val) * y (D.rightIndex o t.val)) := by
  refine ⟨rfl, fun o => ?_⟩
  simp only [binaryMatmulBox]
  rw [getAtOrZero_eq_getScalar, getAtOrZero_eq_getScalar, Tensor.getScalar_ofFn,
    Tensor.getScalar_ofFn, show (∑ t : Fin D.inner,
        x (D.leftIndex o.val t.val) * y (D.rightIndex o.val t.val)) =
        ((List.range D.inner).map fun t => x (D.leftIndex o.val t) * y (D.rightIndex o.val t)).sum
      from by rw [Fin.sum_univ_def, ← List.map_coe_finRange_eq_range, List.map_map]; rfl]
  have h := foldl_encloses (List.range D.inner)
    (fun t => (intervalMul (getAtOrZero L.lo [D.leftIndex o.val t])
      (getAtOrZero L.hi [D.leftIndex o.val t]) (getAtOrZero R.lo [D.rightIndex o.val t])
      (getAtOrZero R.hi [D.rightIndex o.val t])).1)
    (fun t => (intervalMul (getAtOrZero L.lo [D.leftIndex o.val t])
      (getAtOrZero L.hi [D.leftIndex o.val t]) (getAtOrZero R.lo [D.rightIndex o.val t])
      (getAtOrZero R.hi [D.rightIndex o.val t])).2)
    (fun t => x (D.leftIndex o.val t) * y (D.rightIndex o.val t)) 0 0 0
    (by simp [LawfulBoundOps.toReal_zero]) (by simp [LawfulBoundOps.toReal_zero]) ?_
  · -- The directed pair fold of the box is the pair of the directed folds of its endpoints.
    have hp : ∀ (l : List Nat) (p : Nat → α × α) (acc : α × α),
        (l.foldl (fun a t => (BoundOps.addDown a.1 (p t).1, BoundOps.addUp a.2 (p t).2)) acc).1 =
          l.foldl (fun a t => BoundOps.addDown a (p t).1) acc.1 ∧
        (l.foldl (fun a t => (BoundOps.addDown a.1 (p t).1, BoundOps.addUp a.2 (p t).2)) acc).2 =
          l.foldl (fun a t => BoundOps.addUp a (p t).2) acc.2 :=
      fun _ _ _ => ⟨Eq.symm (List.foldl_hom Prod.fst fun _ _ => rfl),
        Eq.symm (List.foldl_hom Prod.snd fun _ _ => rfl)⟩
    rw [← (hp _ _ (0, 0)).1, ← (hp _ _ (0, 0)).2] at h
    simpa only [zero_add] using h
  · intro t ht
    rw [List.mem_range] at ht
    obtain ⟨hi, hj⟩ := hidx o.val o.isLt t ht
    exact CROWN.Graph.intervalMul_encloses (hL.2 ⟨_, hi⟩).1
      (hL.2 ⟨_, hi⟩).2 (hR.2 ⟨_, hj⟩).1 (hR.2 ⟨_, hj⟩).2

/-- The real equation of a scaled matrix product node `id` at the layout `D`: each result
coordinate is the contraction of `MatmulEquation` times the scale, the first coordinate of `r`. -/
def ScaledMatmulEquation (dims : Nat → Nat) (v : Nat → Nat → ℝ) (id p q r : Nat)
    (D : IR.MatmulDims) : Prop :=
  ∀ o : Fin (dims id), v id o.val =
    (∑ t : Fin D.inner, v p (D.leftIndex o.val t.val) * v q (D.rightIndex o.val t.val)) * v r 0

/-- The second half of the scaled matrix product rule. Let `F` be an accumulator of the products
`w_o·s` that bounds `w`, where `w_o` are the contractions at layout `D` and `s` is the scale at
`r`. Adding its coefficient of the scale at `r` (when it has one), then the matrix product's terms
at its coefficients of `w`, started from its constant, gives a state whose represented value bounds
`z + w`. -/
theorem scaledMatmul_tail (dir : BackwardDir) {dims : Nat → Nat} {v : Nat → Nat → ℝ}
    {input k : Nat} {st : State α} {z : ℝ} (h : Represents dims v input k st z)
    (hk : k < st.coeffs.size) {p q r : Nat} (hp : p < k) (hq : q < k) (hr : r < k)
    (h1r : dims r = 1) {D : IR.MatmulDims} {L R : FlatBox α}
    (hidx : ∀ o < D.outShape.size, ∀ t < D.inner,
      D.leftIndex o t < L.dim ∧ D.rightIndex o t < R.dim)
    (hL : RowEncloses L L.dim (v p)) (hR : RowEncloses R R.dim (v q)) (hLd : L.dim = dims p)
    (hRd : R.dim = dims q) {N : Nat} (hN : N = D.outShape.size)
    {F : PartialCoeffs α × PartialCoeffs α × (α × α)} {w : ℝ}
    (hF : AccBound dir F N 1 (fun o => ∑ t : Fin D.inner,
      v p (D.leftIndex o t.val) * v q (D.rightIndex o t.val)) (v r) w)
    {st₁ : State α}
    (hst₁ : (coeffBox 1 F.2.1 = none ∧ st₁ = st) ∨
      ∃ B, coeffBox 1 F.2.1 = some B ∧ st₁ = addCoeff st r B) :
    DirInvariant dir dims v input st.coeffs.size k (z + w)
      (applyProduct st₁ p q L.dim R.dim
        (matmulTerms dir D (fun o => (F.1[o.val]?).join) L R
          (Array.replicate L.dim none, Array.replicate R.dim none, F.2.2))) := by
  subst hN
  obtain ⟨fx, fs, cr, hfx, hfs, hc, hb⟩ := hF
  have hst : st₁.coeffs.size = st.coeffs.size ∧
      Represents dims v input k st₁ (z + Directed.dot 1 fs (v r)) := by
    rcases hst₁ with ⟨hB, rfl⟩ | ⟨B, hB, rfl⟩
    · exact ⟨rfl, by rw [coeffBox_none hfs hB, add_zero]; exact h⟩
    · refine ⟨addCoeff_size .., ?_⟩
      have := represents_add h r (by omega) (mem_pending_of_lt hr) B fs
        (h1r ▸ coeffBox_encloses hfs hB)
      rwa [h1r] at this
  obtain ⟨hsz, h₁⟩ := hst
  have hacc := matmulTerms_sound dir (D := D) (a := fun o => (F.1[o.val]?).join) fx ?_ ?_ hidx hL
    hR (accBound_replicate dir L.dim R.dim (v p) (v q) hc)
  · have hfin := applyProduct_sound dir h₁ (hsz ▸ hk) hp hq hLd hRd hacc
    rw [hsz] at hfin
    refine DirInvariant.mono ?_ hfin
    have e : z + Directed.dot 1 fs (v r) + (cr + ∑ o : Fin D.outShape.size, fx o.val *
        ∑ t : Fin D.inner, v p (D.leftIndex o.val t.val) * v q (D.rightIndex o.val t.val)) =
        z + (Directed.dot D.outShape.size fx (fun o => ∑ t : Fin D.inner,
          v p (D.leftIndex o t.val) * v q (D.rightIndex o t.val)) + Directed.dot 1 fs (v r) + cr) :=
            by
      simp only [Directed.dot, dotProduct, Function.comp_apply]
      ring
    rw [e]
    exact hb.add_left z
  · intro o ao hao
    have := hfx.2 o.val o.isLt
    simpa [partialLo, partialHi, hao] using this
  · intro o hao
    have := hfx.2 o.val o.isLt
    simp only [partialLo, partialHi, hao, Option.map_none, Option.getD_none,
      LawfulBoundOps.toReal_zero] at this
    exact le_antisymm this.2 this.1

/-- The real equations the product rules use at operator node `id`: an entrywise product of two
references multiplies them entry by entry, a matrix product is `MatmulEquation` and a scaled
matrix product `ScaledMatmulEquation` at its layout. Every other operator needs none. -/
def ProductEquation (dims : Nat → Nat) (v : Nat → Nat → ℝ) (id : Nat) :
    IR.Operator → List IR.Ref → Prop
  | ⟨_, _, .mul⟩, [r₁, r₂] => BinaryEquation dims v id r₁.node r₂.node (· * ·)
  | ⟨_, _, .matmul (batchA := bA) (batchB := bB) (batch := b) (m := m) (n := n) (p := p) lv rv
      _ _ _ _⟩, [r₁, r₂] =>
      MatmulEquation dims v id r₁.node r₂.node (matmulLayout lv rv bA bB b m n p)
  | ⟨_, _, .scaledMatmul (batchA := bA) (batchB := bB) (batch := b) (m := m) (n := n) (p := p) lv
      rv _ _ _ _⟩, [r₁, r₂, r₃] =>
      ScaledMatmulEquation dims v id r₁.node r₂.node r₃.node (matmulLayout lv rv bA bB b m n p)
  | _, _ => True

/-- The backward rules of direction `dir` are sound at operator node `k` when its real equations
hold: the entrywise, matrix and scaled matrix products by their McCormick terms, every other
operator by the shared rules (`directedRule_sound`). -/
theorem backwardRule_sound [Op.Capabilities α] (dir : BackwardDir)
    {G : IR.Graph} {P : IR.Payload α} {ibp : Array (Option (FlatBox α))} {ctx : AffineCtx}
    {dims : Nat → Nat} {v : Nat → Nat → ℝ} (hG : GraphPoint G P ibp ctx dims v) {input k : Nat}
    (heq : ∀ o refs, G.nodes[k]? = some (.op o refs) →
      Equation G P dims v k o refs.toList)
    (hprod : ∀ o refs, G.nodes[k]? = some (.op o refs) →
      ProductEquation dims v k o refs.toList) :
    RuleSound dir G (backwardRule dir G P ibp) dims v input k := by
  intro o refs hnode hall st z aY a hk h ha
  have hd := directedRule_sound dir hG (input := input) heq o refs hnode hall st z aY a hk h ha
  obtain ⟨hlt, -⟩ := Array.getElem?_eq_some_iff.mp hnode
  have hpar : ∀ r ∈ refs.toList, r.node < k := fun r hr =>
    hG.topo k hlt o refs hnode r (Array.mem_toList_iff.mp hr)
  have hpe := hprod o refs hnode
  generalize refs.toList = l at hpar hpe hd
  obtain ⟨ins, outs, op⟩ := o
  cases op <;> (try cases ‹Op.NumericElement›) <;> (try cases ‹Op.Element›)
  case mul s =>
    rcases l with _ | ⟨r₁, _ | ⟨r₂, _ | ⟨r₃, l⟩⟩⟩
    all_goals try exact hd
    have hp := hpar r₁ (by simp)
    have hq := hpar r₂ (by simp)
    unfold backwardRule
    dsimp only
    split
    · next X Y hX hY =>
      have hXe := box?_encloses hG hX
      have hYe := box?_encloses hG hY
      split
      · next hdim =>
        obtain ⟨hdk, hdx, hdy⟩ := hdim
        have hak : aY.dim = dims k := ha.1
        refine DirInvariant.mono ?_ (applyProduct_sound dir h hk hp hq hXe.1 hYe.1
          (foldl_accBound dir _ (fun i => a i * (v r₁.node i * v r₂.node i)) _ ?_ _ _
            (accBound_replicate dir _ _ _ _ (cr := 0) ⟨?_, ?_⟩)))
        · rw [zero_add, ← List.map_coe_finRange_eq_range, List.map_map, ← Fin.sum_univ_def, hak]
          simp only [Function.comp_apply]
          have : (∑ i : Fin (dims k), a i.val * (v r₁.node i.val * v r₂.node i.val)) =
              Directed.dot (dims k) a (v k) := by
            simp only [Directed.dot, dotProduct, Function.comp_apply]
            refine Finset.sum_congr rfl fun i _ => ?_
            rw [hpe.2.2 i]
          rw [this]
          exact DirBound.refl dir _
        · intro i hi acc z' hacc
          rw [List.mem_range] at hi
          have hik : i < dims k := hak ▸ hi
          have hiX : i < X.dim := by rw [hdx, ← hdk]; exact hi
          have hiY : i < Y.dim := by rw [hdy, ← hdk]; exact hi
          exact productTerm_sound dir hacc (ha.2 ⟨i, hik⟩) hiX hiY
            (hXe.2 ⟨i, hXe.1 ▸ hiX⟩) (hYe.2 ⟨i, hYe.1 ▸ hiY⟩)
        · simp [LawfulBoundOps.toReal_zero]
        · simp [LawfulBoundOps.toReal_zero]
      · exact DirInvariant.of_eq (Or.inl rfl)
    · exact DirInvariant.of_eq (Or.inl rfl)
  case matmul bA bB b m n p lv rv hl hr hA hB =>
    rcases l with _ | ⟨r₁, _ | ⟨r₂, _ | ⟨r₃, l⟩⟩⟩
    all_goals try exact hd
    have hp := hpar r₁ (by simp)
    have hq := hpar r₂ (by simp)
    unfold backwardRule
    dsimp only
    split
    · next L R hL hR =>
      have hLe := box?_encloses hG hL
      have hRe := box?_encloses hG hR
      split
      · next hdim =>
        obtain ⟨hdk, hdl, hdr⟩ := hdim
        have hak : aY.dim = dims k := ha.1
        refine DirInvariant.mono ?_ (applyProduct_sound dir h hk hp hq hLe.1 hRe.1
          (matmulTerms_sound dir a ?_ ?_ ?_ (by rw [hLe.1]; exact hLe) (by rw [hRe.1]; exact hRe)
            (accBound_replicate dir _ _ _ _ (cr := 0) ⟨?_, ?_⟩)))
        · rw [zero_add]
          have hN : (matmulLayout lv rv bA bB b m n p).outShape.size = dims k := hdk.symm.trans hak
          rw [hN]
          simp only [Directed.dot, dotProduct, Function.comp_apply]
          have hpe' : MatmulEquation dims v k r₁.node r₂.node
              (matmulLayout lv rv bA bB b m n p) := hpe
          simp only [MatmulEquation] at hpe'
          simp only [← hpe']
          exact DirBound.refl dir _
        · intro o ao hao
          cases hao
          exact ha.2 ⟨o.val, by rw [← hak, hdk]; exact o.isLt⟩
        · intro o hao
          simp at hao
        · intro o ho t ht
          exact ⟨hdl ▸ matmulLayout_leftIndex_lt hl hr hA ho ht,
            hdr ▸ matmulLayout_rightIndex_lt hl hr hB ho ht⟩
        · simp [LawfulBoundOps.toReal_zero]
        · simp [LawfulBoundOps.toReal_zero]
      · exact DirInvariant.of_eq (Or.inl rfl)
    · exact DirInvariant.of_eq (Or.inl rfl)
  case scaledMatmul bA bB b m n p lv rv hl hr hA hB =>
    rcases l with _ | ⟨r₁, _ | ⟨r₂, _ | ⟨r₃, _ | ⟨r₄, l⟩⟩⟩⟩
    all_goals try exact hd
    have hp := hpar r₁ (by simp)
    have hq := hpar r₂ (by simp)
    have hs := hpar r₃ (by simp)
    unfold backwardRule
    dsimp only
    split
    · next L R S hL hR hS =>
      have hLe := box?_encloses hG hL
      have hRe := box?_encloses hG hR
      have hSe := box?_encloses hG hS
      split
      · next hdim =>
        obtain ⟨hdk, hdl, hdr, hds⟩ := hdim
        have hak : aY.dim = dims k := ha.1
        have h1r : dims r₃.node = 1 := hSe.1.symm.trans hds
        have hidx : ∀ o < (matmulLayout lv rv bA bB b m n p).outShape.size,
            ∀ t < (matmulLayout lv rv bA bB b m n p).inner,
              (matmulLayout lv rv bA bB b m n p).leftIndex o t < L.dim ∧
                (matmulLayout lv rv bA bB b m n p).rightIndex o t < R.dim :=
          fun o ho t ht => ⟨hdl ▸ matmulLayout_leftIndex_lt hl hr hA ho ht,
            hdr ▸ matmulLayout_rightIndex_lt hl hr hB ho ht⟩
        have hL' : RowEncloses L L.dim (v r₁.node) := by rw [hLe.1]; exact hLe
        have hR' : RowEncloses R R.dim (v r₂.node) := by rw [hRe.1]; exact hRe
        have hW := binaryMatmulBox_rowEncloses _ hidx hL' hR'
        have hpe' : ScaledMatmulEquation dims v k r₁.node r₂.node r₃.node
            (matmulLayout lv rv bA bB b m n p) := hpe
        simp only [ScaledMatmulEquation] at hpe'
        split
        rotate_left
        all_goals
          refine DirInvariant.mono ?_ (scaledMatmul_tail dir h hk hp hq hs h1r hidx hL' hR' hLe.1
            hRe.1 hdk (foldl_accBound dir _ (fun o => a o *
              ((∑ t : Fin (matmulLayout lv rv bA bB b m n p).inner,
                v r₁.node ((matmulLayout lv rv bA bB b m n p).leftIndex o t.val) *
                v r₂.node ((matmulLayout lv rv bA bB b m n p).rightIndex o t.val)) *
                  v r₃.node 0)) _ ?_ _ _
              (accBound_replicate dir _ _ _ _ (cr := 0) ⟨?_, ?_⟩)) ?_)
          · rw [zero_add, ← List.map_coe_finRange_eq_range, List.map_map, ← Fin.sum_univ_def]
            simp only [Function.comp_apply]
            have : (∑ i : Fin aY.dim, a i.val *
                ((∑ t : Fin (matmulLayout lv rv bA bB b m n p).inner,
                  v r₁.node ((matmulLayout lv rv bA bB b m n p).leftIndex i.val t.val) *
                  v r₂.node ((matmulLayout lv rv bA bB b m n p).rightIndex i.val t.val)) *
                    v r₃.node 0)) = Directed.dot (dims k) a (v k) := by
              rw [hak]
              simp only [Directed.dot, dotProduct, Function.comp_apply]
              refine Finset.sum_congr rfl fun o _ => ?_
              rw [hpe' o]
            rw [this]
            exact DirBound.refl dir _
          · intro o ho acc' z' h'
            rw [List.mem_range] at ho
            exact productTerm_sound dir h' (ha.2 ⟨o, hak ▸ ho⟩) ho Nat.zero_lt_one
              (hW.2 ⟨o, hdk ▸ ho⟩) (hSe.2 ⟨0, h1r ▸ Nat.zero_lt_one⟩)
          · simp [LawfulBoundOps.toReal_zero]
          · simp [LawfulBoundOps.toReal_zero]
        · next hB => exact Or.inl ⟨hB, rfl⟩
        · next B hB => exact Or.inr ⟨B, hB, rfl⟩
      · exact DirInvariant.of_eq (Or.inl rfl)
    · exact DirInvariant.of_eq (Or.inl rfl)
  all_goals exact hd

omit [LawfulBoundOps α] in
/-- The tighter of two forms is one of them. -/
theorem tighterForm_eq_or [Op.Capabilities α] (lower : Bool)
    (xB? : Option (FlatBox α)) {n : Nat} (a b : Linear α n 1) :
    tighterForm lower xB? a b = a ∨ tighterForm lower xB? a b = b := by
  unfold tighterForm
  split
  · split <;> split <;> simp
  · exact Or.inl rfl

/-- The forms of `objectiveForms` bound the objective `obj · v out` at a real point of the
graph whose operator nodes satisfy their real equations, for any input box: the lower form from
below and the upper form from above, when there is one. -/
theorem objectiveForms_encloses [Op.Capabilities α] {G : IR.Graph}
    {P : IR.Payload α} {ibp : Array (Option (FlatBox α))} {ctx : AffineCtx}
    {dims : Nat → Nat} {v : Nat → Nat → ℝ} (hG : GraphPoint G P ibp ctx dims v)
    (heq : ∀ k o refs, G.nodes[k]? = some (.op o refs) →
      Equation G P dims v k o refs.toList)
    (hprod : ∀ k o refs, G.nodes[k]? = some (.op o refs) →
      ProductEquation dims v k o refs.toList)
    {out : Nat} {obj : FlatTensor α} (hobj : obj.n = dims out) (xB? : Option (FlatBox α)) :
    (∀ l, (objectiveForms G P ctx ibp out obj xB?).1 = some l →
      affineValue l (fun i => v ctx.inputId i.val) ≤ objectiveValue dims v out obj) ∧
    (∀ u, (objectiveForms G P ctx ibp out obj xB?).2 = some u →
      objectiveValue dims v out obj ≤ affineValue u (fun i => v ctx.inputId i.val)) := by
  have hs : ∀ l u, sweep (directedStep G P ibp ctx) G ctx ibp out obj = some (l, u) →
      affineValue l (fun i => v ctx.inputId i.val) ≤ objectiveValue dims v out obj ∧
        objectiveValue dims v out obj ≤ affineValue u (fun i => v ctx.inputId i.val) :=
    fun _ _ hsw => directedSweep_encloses hG heq hobj hsw
  have hb : ∀ dir l u, sweep (backwardStep dir G P ibp ctx) G ctx ibp out obj = some (l, u) →
      DirBound dir (affineValue (dirForm dir l u) (fun i => v ctx.inputId i.val))
        (objectiveValue dims v out obj) :=
    fun dir _ _ hsw => sweep_bound dir hG
      (fun k _ => backwardRule_sound dir hG (heq k) (hprod k)) hobj hsw
  unfold objectiveForms
  dsimp only
  generalize hl : sweep (backwardStep .lower G P ibp ctx) G ctx ibp out obj = SL
  generalize hsh : sweep (directedStep G P ibp ctx) G ctx ibp out obj = SS
  generalize hu : sweep (backwardStep .upper G P ibp ctx) G ctx ibp out obj = SU
  constructor
  · intro l h
    rcases SL with _ | ⟨l₁, u₁⟩ <;> rcases SS with _ | ⟨l₂, u₂⟩ <;>
      simp only [Option.map_some, Option.map_none, Option.some.injEq, reduceCtorEq] at h
    · subst h
      exact (hs _ _ hsh).1
    · subst h
      exact hb .lower _ _ hl
    · subst h
      rcases tighterForm_eq_or true xB? l₁ l₂ with e | e <;> rw [e]
      · exact hb .lower _ _ hl
      · exact (hs _ _ hsh).1
  · intro u h
    rcases SU with _ | ⟨l₁, u₁⟩ <;> rcases SS with _ | ⟨l₂, u₂⟩ <;>
      simp only [Option.map_some, Option.map_none, Option.some.injEq, reduceCtorEq] at h
    · subst h
      exact (hs _ _ hsh).2
    · subst h
      exact hb .upper _ _ hu
    · subst h
      rcases tighterForm_eq_or false xB? u₁ u₂ with e | e <;> rw [e]
      · exact hb .upper _ _ hu
      · exact (hs _ _ hsh).2

/-- The lower and upper forms of `backwardObjective?` bound the objective `obj · v out` at a real
point of the graph whose operator nodes satisfy their real equations (`objectiveForms_encloses` on
the input's box). -/
theorem backwardObjective?_encloses [Op.Capabilities α] {G : IR.Graph}
    {P : IR.Payload α} {ibp : Array (Option (FlatBox α))} {ctx : AffineCtx}
    {dims : Nat → Nat} {v : Nat → Nat → ℝ} (hG : GraphPoint G P ibp ctx dims v)
    (heq : ∀ k o refs, G.nodes[k]? = some (.op o refs) →
      Equation G P dims v k o refs.toList)
    (hprod : ∀ k o refs, G.nodes[k]? = some (.op o refs) →
      ProductEquation dims v k o refs.toList)
    {out : Nat} {obj : FlatTensor α} (hobj : obj.n = dims out)
    {lower upper : Linear α ctx.inputDim 1}
    (h : backwardObjective? G P ctx ibp out obj = some (lower, upper)) :
    affineValue lower (fun i => v ctx.inputId i.val) ≤ objectiveValue dims v out obj ∧
      objectiveValue dims v out obj ≤ affineValue upper (fun i => v ctx.inputId i.val) := by
  have hf := objectiveForms_encloses hG heq hprod hobj (ibp[ctx.inputId]?).join
  unfold backwardObjective? at h
  split at h
  · rename_i l u he
    obtain ⟨rfl, rfl⟩ : l = lower ∧ u = upper := by simpa using h
    rw [he] at hf
    exact ⟨hf.1 _ rfl, hf.2 _ rfl⟩
  · cases h

/-- The rows of `affineBounds?` enclose the rows of `C · v out` at a real point of the graph whose
operator nodes satisfy their real equations, when `outSize?` gives the width of node `out`: row `r`
bounds `∑ j, C r j * v out j`. -/
theorem affineBounds?_encloses [Op.Capabilities α] {G : IR.Graph}
    {P : IR.Payload α} {ibp : Array (Option (FlatBox α))} {ctx : AffineCtx}
    {dims : Nat → Nat} {v : Nat → Nat → ℝ} (hG : GraphPoint G P ibp ctx dims v)
    (heq : ∀ k o refs, G.nodes[k]? = some (.op o refs) →
      Equation G P dims v k o refs.toList)
    (hprod : ∀ k o refs, G.nodes[k]? = some (.op o refs) →
      ProductEquation dims v k o refs.toList)
    {out : Nat} (hdim : outSize? G out = some (dims out)) {k n : Nat} (C : Tensor α [k, n])
    {bounds : FlatAffineBounds α} (h : affineBounds? G P ctx ibp out C = some bounds) :
    bounds.inDim = ctx.inputDim ∧ bounds.outDim = k ∧ n = dims out ∧
      AffineRowsEnclose bounds (v ctx.inputId)
        (fun r => Directed.dot n (fun j => value (getAtOrZero C [r, j])) (v out)) := by
  by_cases hn : n = dims out
  · subst hn
    unfold affineBounds? at h
    simp only [hdim, Option.bind_eq_bind, Option.bind_some, guard, ↓reduceIte,
      Option.pure_def] at h
    obtain ⟨rows, hrows, hb⟩ := Option.bind_eq_some_iff.mp h
    cases hb
    refine ⟨rfl, rfl, rfl, fun r => ?_⟩
    have hs := backwardObjective?_encloses hG heq hprod rfl
      (Tensor.sequenceFinM_get_of_eq_some hrows r)
    simpa [objectiveValue, Directed.dot, dotProduct, affineValue,
      Tensor.getAtOrZero_eq_getScalar,
      Tensor.getAtOrZero_eq_get2, Tensor.matrix, Tensor.get2] using hs
  · unfold affineBounds? at h
    simp [hdim, guard, hn] at h

/-- The rows of `nodeBounds?` enclose the coordinates of node `out` at a real point of the graph
whose operator nodes satisfy their real equations, when `outSize?` gives the node's width: the rows
of the identity objective, or the constant rows of the node's box when a sweep fails. -/
theorem nodeBounds?_encloses [Op.Capabilities α] {G : IR.Graph}
    {P : IR.Payload α} {ibp : Array (Option (FlatBox α))} {ctx : AffineCtx}
    {dims : Nat → Nat} {v : Nat → Nat → ℝ} (hG : GraphPoint G P ibp ctx dims v)
    (heq : ∀ k o refs, G.nodes[k]? = some (.op o refs) →
      Equation G P dims v k o refs.toList)
    (hprod : ∀ k o refs, G.nodes[k]? = some (.op o refs) →
      ProductEquation dims v k o refs.toList)
    {out : Nat} (hdim : outSize? G out = some (dims out)) {bounds : FlatAffineBounds α}
    (h : nodeBounds? G P ctx ibp out = some bounds) :
    bounds.inDim = ctx.inputDim ∧ bounds.outDim = dims out ∧
      AffineRowsEnclose bounds (v ctx.inputId) (v out) := by
  unfold nodeBounds? at h
  simp only [hdim, Option.bind_eq_bind, Option.bind_some] at h
  split at h
  · rename_i bounds' hb
    cases h
    unfold affineBounds? at hb
    simp only [hdim, Option.bind_eq_bind, Option.bind_some, guard, ↓reduceIte,
      Option.pure_def] at hb
    obtain ⟨rows, hrows, hb⟩ := Option.bind_eq_some_iff.mp hb
    cases hb
    refine ⟨rfl, rfl, fun i => ?_⟩
    have hs := backwardObjective?_encloses hG heq hprod rfl
      (Tensor.sequenceFinM_get_of_eq_some hrows i)
    simpa [objectiveValue, Directed.dot, dotProduct, affineValue,
      Tensor.getAtOrZero_eq_getScalar,
      apply_ite, apply_dite, LawfulBoundOps.toReal_zero (α := α),
      LawfulBoundOps.toReal_one (α := α), Tensor.matrix, Tensor.get2] using hs
  · obtain ⟨B, hB, hb⟩ := Option.bind_eq_some_iff.mp h
    cases hb
    have hy := hG.ibp_encloses out B hB
    exact ⟨rfl, hy.1, boundsConst_encloses ctx.inputDim B (v ctx.inputId) (v out)
      (by simpa only [hy.1] using hy)⟩

/-- The box of `backwardObjectiveBox?` at graph output `i` encloses the objective `obj · v out` of
its node `out` at a real point of the graph whose operator nodes satisfy their real equations, when
the box `xB` encloses the point's input. -/
theorem backwardObjectiveBox?_encloses [Op.Capabilities α] [NonlinearBoundOps α]
    {G : IR.Graph} {P : IR.Payload α} {ibp : Array (Option (FlatBox α))} {xB : FlatBox α}
    {i out : Nat} {ctx : AffineCtx} (hctx : outputContext G xB i = .ok (out, ctx))
    {dims : Nat → Nat} {v : Nat → Nat → ℝ} (hG : GraphPoint G P ibp ctx dims v)
    (heq : ∀ k o refs, G.nodes[k]? = some (.op o refs) →
      Equation G P dims v k o refs.toList)
    (hprod : ∀ k o refs, G.nodes[k]? = some (.op o refs) →
      ProductEquation dims v k o refs.toList)
    (hx : RowEncloses xB ctx.inputDim (v ctx.inputId)) {obj : FlatTensor α}
    (hobj : obj.n = dims out) {result : FlatBox α}
    (h : backwardObjectiveBox? G P ibp xB i obj = .ok result) :
    RowEncloses result 1 (fun _ => objectiveValue dims v out obj) := by
  have hf := objectiveForms_encloses hG heq hprod hobj (some xB)
  have hbox : ∀ B, ibp[out]! = some B → RowEncloses B obj.n (v out) := by
    intro B hB
    rw [hobj]
    by_cases hlt : out < ibp.size
    · rw [getElem!_pos ibp out hlt] at hB
      exact hG.ibp_encloses out B (by rw [getElem?_pos ibp out hlt, hB]; rfl)
    · rw [getElem!_neg ibp out hlt] at hB
      cases hB
  have hL : ∀ l, ((objectiveForms G P ctx ibp out obj (some xB)).1 <|>
      outputBoxBound .lower ibp out ctx.inputDim obj) = some l →
      affineValue l (fun i => v ctx.inputId i.val) ≤ objectiveValue dims v out obj := by
    intro l hl
    cases hF : (objectiveForms G P ctx ibp out obj (some xB)).1 with
    | some l' =>
      rw [hF] at hl
      cases hl
      exact hf.1 _ hF
    | none =>
      rw [hF] at hl
      have := outputBoxBound_encloses .lower ibp out ctx.inputDim obj (v out) hbox
        hl (fun i => v ctx.inputId i.val)
      simpa only [objectiveValue, ← hobj] using this
  have hU : ∀ u, ((objectiveForms G P ctx ibp out obj (some xB)).2 <|>
      outputBoxBound .upper ibp out ctx.inputDim obj) = some u →
      objectiveValue dims v out obj ≤ affineValue u (fun i => v ctx.inputId i.val) := by
    intro u hu
    cases hF : (objectiveForms G P ctx ibp out obj (some xB)).2 with
    | some u' =>
      rw [hF] at hu
      cases hu
      exact hf.2 _ hF
    | none =>
      rw [hF] at hu
      have := outputBoxBound_encloses .upper ibp out ctx.inputDim obj (v out) hbox
        hu (fun i => v ctx.inputId i.val)
      simpa only [objectiveValue, ← hobj] using this
  unfold backwardObjectiveBox? at h
  rw [hctx] at h
  simp only [bind, Except.bind] at h
  split at h
  · rename_i loAff hiAff hlo hhi
    split at h
    · cases h
    · rename_i B hB
      split at h
      · rename_i I hI
        split at h
        · rename_i l u hl hu
          split at h
          · simp only [pure, Except.pure, Except.ok.injEq] at h
            subst h
            have hz : RowEncloses B 1 (fun _ => objectiveValue dims v out obj) :=
              evalObjectiveBox_encloses _ xB ctx.inputDim (v ctx.inputId) _ hx
                (scalarBounds_enclose ⟨hL _ hlo, hU _ hhi⟩) hB
            have hc : RowEncloses I obj.n (v out) := by
              rw [hobj]
              exact hG.ibp_encloses out I hI
            have hcl := consumeBox_encloses .lower obj I (v out) hc hl
            have hcu := consumeBox_encloses .upper obj I (v out) hc hu
            refine meetBox_encloses hz ⟨rfl, fun j => ?_⟩
            rw [objectiveValue, ← hobj]
            simpa only [DirectionBound, Tensor.getAtOrZero_eq_getScalar,
              Tensor.getScalar_full] using
              And.intro hcl hcu
          · cases h
        · cases h
      · cases h
  · split at h
    · cases h
    · rename_i heq'
      cases heq'

/-- The CROWN box of graph output `i` encloses the coordinates of its node `out` at a real point of
the graph whose operator nodes satisfy their real equations, when the interval boxes of the run
enclose the point, the box `xB` encloses its input and `outSize?` gives the node's width. -/
theorem outputBoxCROWN?_encloses [Op.Capabilities α] [NonlinearBoundOps α]
    {G : IR.Graph} {P : IR.Payload α} {xB : FlatBox α} {i out : Nat} {ctx : AffineCtx}
    (hctx : outputContext G xB i = .ok (out, ctx)) {dims : Nat → Nat} {v : Nat → Nat → ℝ}
    (hG : GraphPoint G P (runIBP G P (inputSeeds G [xB])) ctx dims v)
    (heq : ∀ k o refs, G.nodes[k]? = some (.op o refs) →
      Equation G P dims v k o refs.toList)
    (hprod : ∀ k o refs, G.nodes[k]? = some (.op o refs) →
      ProductEquation dims v k o refs.toList)
    (hdim : outSize? G out = some (dims out)) (hx : RowEncloses xB ctx.inputDim (v ctx.inputId))
    {result : FlatBox α} (h : outputBoxCROWN? G P xB i = .ok result) :
    RowEncloses result (dims out) (v out) := by
  unfold outputBoxCROWN? at h
  rw [hctx] at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  · rename_i B hB
    unfold evalCROWNOutputBox? at hB
    simp only [bind, Except.bind, Array.set!_eq_setIfInBounds,
      Array.getElem?_setIfInBounds_self, Array.size_replicate] at hB
    by_cases hlt : out < G.nodes.size
    · simp only [hlt, ↓reduceIte] at hB
      generalize hNB : nodeBounds? G P ctx (runIBP G P (inputSeeds G [xB])) out = NB at hB
      rcases NB with _ | bounds
      · simp only [throw, throwThe, MonadExceptOf.throw, reduceCtorEq] at hB
      · simp only [pure, Except.pure] at hB
        obtain ⟨hin, hout, hrows⟩ := nodeBounds?_encloses hG heq hprod hdim hNB
        rw [dite_eq_left hin, dite_eq_left hx.1] at hB
        obtain rfl := Except.ok.inj hB
        have hE := affineBoundsEval_encloses bounds xB (v ctx.inputId) (v out)
          (by rw [hin]; exact hx) hrows
        unfold meetWith at h
        split at h
        · rename_i I hI
          split at h
          · obtain rfl := Except.ok.inj h
            rw [← hout]
            exact meetBox_encloses hE (by rw [hout]; exact hG.ibp_encloses out I hI)
          · simp only [throw, throwThe, MonadExceptOf.throw, reduceCtorEq] at h
        · simp only [throw, throwThe, MonadExceptOf.throw, reduceCtorEq] at h
    · simp only [hlt, ↓reduceIte, throw, throwThe, MonadExceptOf.throw, reduceCtorEq] at hB

end Lawful

end TorchLean.CROWN.Op

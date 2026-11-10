/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.Op.DirectedBasic
import TorchLean.Verify.CROWN.BoundOps.Interval
import TorchLean.NN.Softmax

/-!
# Enclosures of the rounded IBP boxes at nonlinear endpoint operations

The lemmas here are stated at a scalar whose nonlinear endpoint operations are lawful as well
(`LawfulNonlinearBoundOps`). The boxes of the exponential, logarithmic, hyperbolic and
trigonometric transfers enclose the real function values (`boxSafeLog?_encloses`,
`coshBounds_enclosure`, `arsinhBounds_enclosure`, `tanBounds_enclosure`), and log softmax's box
encloses the real log probabilities (`logSoftmaxEntry_encloses`). Mean squared error divides an
enclosure of the sum of the squared differences by an enclosure of the exact coordinate count
(`Directed.boxMean?_encloses`), batch norm's box encloses the real normalization
(`batchNormBounds_encloses`), and a layer norm's box, the directed box of each row met with the
uniform range of each entry, encloses the real layer norm (`layerNormBox?_encloses`,
`layerNormRange_encloses`).
-/

@[expose] public section

namespace TorchLean.CROWN.Op.Directed

open Tensor
open Graph
open CROWN.Directed

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

noncomputable section

variable [NonlinearBoundOps α] [LawfulNonlinearBoundOps α]

/-- Both safeLog paths enclose `log (softplus x + epsilon)`, including its varying epsilon.

The executable positivity check supplies the logarithm's domain. If the logarithm transfer
declines, directed reciprocal bounds give `1 - 1/z ≤ log z ≤ z - 1`.
-/
theorem _root_.TorchLean.CROWN.Directed.safeLogBounds_enclosure :
    BinaryEnclosure (α := α)
      (fun x epsilon => Real.log (Real.log (1 + Real.exp x) + epsilon))
      NonlinearBoundOps.safeLogBounds := by
  intro lo hi epsilonLo epsilonHi outLo outHi x epsilon hout hxlo hxhi helo hehi
  unfold NonlinearBoundOps.safeLogBounds at hout
  obtain ⟨⟨softLo, softHi⟩, hsoft, hout⟩ := Option.bind_eq_some_iff.mp hout
  dsimp only at hout
  have hsoftBounds := softplusBounds_enclosure hsoft hxlo hxhi
  have hlo : value (BoundOps.addDown softLo epsilonLo) ≤
      Real.log (1 + Real.exp x) + epsilon :=
    (LawfulBoundOps.addDown_le _ _).trans (add_le_add hsoftBounds.1 helo)
  have hhi : Real.log (1 + Real.exp x) + epsilon ≤
      value (BoundOps.addUp softHi epsilonHi) :=
    (add_le_add hsoftBounds.2 hehi).trans (LawfulBoundOps.le_addUp _ _)
  split at hout
  next hpos =>
    have hposReal : 0 < value (BoundOps.addDown softLo epsilonLo) := by
      simpa only [(LawfulBoundOps.toReal_zero (α := α))] using
        (LawfulBoundOps.lt_iff 0 _).mp hpos
    have hz := hposReal.trans_le hlo
    cases hlog : NonlinearBoundOps.logBounds
        (BoundOps.addDown softLo epsilonLo) (BoundOps.addUp softHi epsilonHi) with
    | some bounds =>
        simp only [hlog, Option.pure_def, Option.some.injEq] at hout
        subst bounds
        exact LawfulNonlinearBoundOps.logBounds_enclosure hlog hlo hhi
    | none =>
        simp only [hlog] at hout
        obtain ⟨⟨reciprocalLo, reciprocalHi⟩, hdiv, hout⟩ :=
          Option.bind_eq_some_iff.mp hout
        obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj hout)
        have hreciprocal := LawfulNonlinearBoundOps.divBounds_enclosure (x := (1 : ℝ)) hdiv
          (by simp [LawfulBoundOps.toReal_one (α := α)])
          (by simp [LawfulBoundOps.toReal_one (α := α)]) hlo hhi
        constructor
        · calc
            value (BoundOps.subDown 1 reciprocalHi) ≤ 1 - value reciprocalHi := by
              simpa only [(LawfulBoundOps.toReal_one (α := α))] using
                LawfulBoundOps.subDown_le (1 : α) reciprocalHi
            _ ≤ 1 - 1 / (Real.log (1 + Real.exp x) + epsilon) :=
              sub_le_sub_left hreciprocal.2 1
            _ ≤ Real.log (Real.log (1 + Real.exp x) + epsilon) := by
              simpa only [one_div] using Real.one_sub_inv_le_log_of_pos hz
        · calc
            Real.log (Real.log (1 + Real.exp x) + epsilon) ≤
                Real.log (1 + Real.exp x) + epsilon - 1 :=
              Real.log_le_sub_one_of_pos hz
            _ ≤ value (BoundOps.addUp softHi epsilonHi) - 1 := sub_le_sub_right hhi 1
            _ ≤ value (BoundOps.subUp (BoundOps.addUp softHi epsilonHi) 1) := by
              simpa only [(LawfulBoundOps.toReal_one (α := α))] using
                LawfulBoundOps.le_subUp (BoundOps.addUp softHi epsilonHi) (1 : α)
  next => contradiction

/-- SafeLog uses the same enclosed scalar epsilon at every input coordinate. -/
theorem _root_.TorchLean.CROWN.Directed.boxSafeLog?_encloses
    {x epsilon box : FlatBox α} {n : Nat} {f e : Nat → ℝ}
    (hx : RowEncloses x n f) (he : RowEncloses epsilon 1 e)
    (hbox : boxSafeLog? x epsilon = some box) :
    RowEncloses box n (fun i => Real.log (Real.log (1 + Real.exp (f i)) + e 0)) := by
  obtain ⟨d, lo, hi⟩ := epsilon
  have hd := he.1
  change d = 1 at hd
  subst hd
  rw [rowEncloses_iff] at he
  simp only [boxSafeLog?, dite_true] at hbox
  apply boxUnaryEnclosure?_encloses (F := fun x => Real.log (Real.log (1 + Real.exp x) + e 0))
    ?_ hx hbox
  intro lo' hi' outLo outHi x' hout hxlo hxhi
  exact safeLogBounds_enclosure hout hxlo hxhi (he 0).1 (he 0).2

/-- `eluBounds a` encloses `x` above 0 and `a (exp x - 1)` at most 0, for any sign of
`a`. -/
theorem eluBounds_enclosure (a : α) :
    UnaryEnclosure (fun x => if 0 < x then x else value a * (Real.exp x - 1))
      (eluBounds a) := by
  intro lo hi outLo outHi x h hlo hhi
  have hpos : ∀ e : α, 0 < e ↔ 0 < value e := fun e => by
    rw [LawfulBoundOps.lt_iff, LawfulBoundOps.toReal_zero]
  unfold eluBounds at h
  by_cases h₀ : 0 < lo
  · simp only [h₀, ↓reduceIte, Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    have hx : 0 < x := ((hpos lo).mp h₀).trans_le hlo
    simp only [hx, ↓reduceIte]
    exact ⟨hlo, hhi⟩
  simp only [h₀, ↓reduceIte, Option.bind_eq_bind] at h
  obtain ⟨e, he, h⟩ := Option.bind_eq_some_iff.mp h
  -- the negative branch, on `[lo, min hi 0]`
  have hneg : x ≤ 0 →
      value (intervalMul (BoundOps.subDown e.1 1) (BoundOps.subUp e.2 1) a a).1 ≤
        value a * (Real.exp x - 1) ∧ value a * (Real.exp x - 1) ≤
          value (intervalMul (BoundOps.subDown e.1 1) (BoundOps.subUp e.2 1) a a).2 := by
    intro hx0
    have hxm : x ≤ value (Context.minimum hi 0) := by
      rw [LawfulBoundOps.toReal_minimum, LawfulBoundOps.toReal_zero]
      exact le_min hhi hx0
    have hex := LawfulNonlinearBoundOps.expBounds_enclosure (α := α) he hlo hxm
    have hd := LawfulBoundOps.subDown_le e.1 1
    have hu := LawfulBoundOps.le_subUp e.2 1
    rw [LawfulBoundOps.toReal_one] at hd hu
    have hl : value (BoundOps.subDown e.1 1) ≤ Real.exp x - 1 := by linarith [hex.1]
    have hh : Real.exp x - 1 ≤ value (BoundOps.subUp e.2 1) := by linarith [hex.2]
    rw [mul_comm (value a)]
    exact Graph.intervalMul_encloses hl hh le_rfl le_rfl
  split_ifs at h with h₁
  · simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    simp only [LawfulBoundOps.toReal_minimum, LawfulBoundOps.toReal_maximum,
      LawfulBoundOps.toReal_zero]
    split_ifs with hx
    · exact ⟨(min_le_right _ _).trans hx.le, hhi.trans (le_max_right _ _)⟩
    · have hn := hneg (not_lt.mp hx)
      exact ⟨(min_le_left _ _).trans hn.1, hn.2.trans (le_max_left _ _)⟩
  · simp only [Option.some.injEq] at h
    rw [h] at hneg
    have hx : ¬ 0 < x := fun hx => h₁ ((hpos hi).mpr (hx.trans_le hhi))
    simp only [hx, ↓reduceIte]
    exact hneg (not_lt.mp hx)

/-- `halfBounds` encloses `x / 2`. -/
theorem halfBounds_enclosure : UnaryEnclosure (fun x => x / 2) (halfBounds (α := α)) := by
  intro lo hi outLo outHi x h hlo hhi
  have hd := LawfulBoundOps.addDown_le (1 : α) 1
  have hu := LawfulBoundOps.le_addUp (1 : α) 1
  rw [LawfulBoundOps.toReal_one] at hd hu
  exact LawfulNonlinearBoundOps.divBounds_enclosure (α := α) h hlo hhi
    (show value (BoundOps.addDown (1 : α) 1) ≤ (2 : ℝ) by linarith)
    (show (2 : ℝ) ≤ value (BoundOps.addUp (1 : α) 1) by linarith)

/-- `expNegBounds` encloses `exp (-x)`. -/
theorem expNegBounds_enclosure :
    UnaryEnclosure (fun x => Real.exp (-x)) (expNegBounds (α := α)) := by
  intro lo hi outLo outHi x h hlo hhi
  have hd := LawfulBoundOps.subDown_le (0 : α) hi
  have hu := LawfulBoundOps.le_subUp (0 : α) lo
  rw [LawfulBoundOps.toReal_zero] at hd hu
  exact LawfulNonlinearBoundOps.expBounds_enclosure (α := α) h (by linarith) (by linarith)

/-- `sinhBounds` encloses `sinh`. -/
theorem sinhBounds_enclosure : UnaryEnclosure Real.sinh (sinhBounds (α := α)) := by
  intro lo hi outLo outHi x h hlo hhi
  unfold sinhBounds at h
  simp only [Option.bind_eq_bind] at h
  obtain ⟨e, he, h⟩ := Option.bind_eq_some_iff.mp h
  obtain ⟨f, hf, h⟩ := Option.bind_eq_some_iff.mp h
  have hex := LawfulNonlinearBoundOps.expBounds_enclosure (α := α) he hlo hhi
  have hfx : value f.1 ≤ Real.exp (-x) ∧ Real.exp (-x) ≤ value f.2 :=
    expNegBounds_enclosure (α := α) hf hlo hhi
  have hd := LawfulBoundOps.subDown_le e.1 f.2
  have hu := LawfulBoundOps.le_subUp e.2 f.1
  have hx : value outLo ≤ (Real.exp x - Real.exp (-x)) / 2 ∧
      (Real.exp x - Real.exp (-x)) / 2 ≤ value outHi :=
    halfBounds_enclosure (α := α) h (by linarith [hex.1, hfx.2]) (by linarith [hex.2, hfx.1])
  rwa [Real.sinh_eq]

/-- `twoCoshAt a` bounds `2 cosh a`. -/
theorem twoCoshAt_mem {a l u : α} (h : twoCoshAt a = some (l, u)) :
    value l ≤ 2 * Real.cosh (value a) ∧ 2 * Real.cosh (value a) ≤ value u := by
  unfold twoCoshAt at h
  simp only [Option.bind_eq_bind] at h
  obtain ⟨e, he, h⟩ := Option.bind_eq_some_iff.mp h
  obtain ⟨f, hf, h⟩ := Option.bind_eq_some_iff.mp h
  simp only [Option.some.injEq, Prod.mk.injEq] at h
  obtain ⟨rfl, rfl⟩ := h
  have hex := LawfulNonlinearBoundOps.expBounds_enclosure (α := α) he le_rfl le_rfl
  have hfx : value f.1 ≤ Real.exp (-value a) ∧ Real.exp (-value a) ≤ value f.2 :=
    expNegBounds_enclosure (α := α) hf le_rfl le_rfl
  have hd := LawfulBoundOps.addDown_le e.1 f.1
  have hu := LawfulBoundOps.le_addUp e.2 f.2
  rw [Real.cosh_eq]
  constructor <;> linarith [hex.1, hex.2, hfx.1, hfx.2]

/-- `coshBounds` encloses `cosh`. -/
theorem coshBounds_enclosure : UnaryEnclosure Real.cosh (coshBounds (α := α)) := by
  intro lo hi outLo outHi x h hlo hhi
  unfold coshBounds at h
  simp only [Option.bind_eq_bind] at h
  obtain ⟨l, hl, h⟩ := Option.bind_eq_some_iff.mp h
  obtain ⟨u, hu, h⟩ := Option.bind_eq_some_iff.mp h
  have hL := twoCoshAt_mem (α := α) hl
  have hU := twoCoshAt_mem (α := α) hu
  have hz : ∀ e : α, (0 < e ↔ 0 < value e) ∧ (e < 0 ↔ value e < 0) := fun e => by
    rw [LawfulBoundOps.lt_iff (α := α) 0 e, LawfulBoundOps.lt_iff (α := α) e 0,
      LawfulBoundOps.toReal_zero]
    exact ⟨.rfl, .rfl⟩
  have hup : 2 * Real.cosh x ≤ value (Context.maximum l.2 u.2) := by
    rw [LawfulBoundOps.toReal_maximum]
    have habs : |x| ≤ max |value lo| |value hi| := abs_le_max_abs_abs hlo hhi
    rcases le_total |value lo| |value hi| with hle | hle
    · have hc : Real.cosh x ≤ Real.cosh (value hi) :=
        Real.cosh_le_cosh.mpr (by rwa [max_eq_right hle] at habs)
      exact le_max_of_le_right (by linarith [hU.2])
    · have hc : Real.cosh x ≤ Real.cosh (value lo) :=
        Real.cosh_le_cosh.mpr (by rwa [max_eq_left hle] at habs)
      exact le_max_of_le_left (by linarith [hL.2])
  have hlow : value (if 0 < lo then l.1 else if hi < 0 then u.1 else BoundOps.addDown 1 1) ≤
      2 * Real.cosh x := by
    split_ifs with h₀ h₁
    · have h0 := (hz lo).1.mp h₀
      have hc : Real.cosh (value lo) ≤ Real.cosh x := Real.cosh_le_cosh.mpr (by
        rw [abs_of_pos h0, abs_of_pos (h0.trans_le hlo)]; exact hlo)
      linarith [hL.1]
    · have h1 := (hz hi).2.mp h₁
      have hc : Real.cosh (value hi) ≤ Real.cosh x := Real.cosh_le_cosh.mpr (by
        rw [abs_of_neg h1, abs_of_neg (hhi.trans_lt h1)]; linarith)
      linarith [hU.1]
    · have hd := LawfulBoundOps.addDown_le (1 : α) 1
      rw [LawfulBoundOps.toReal_one] at hd
      linarith [Real.one_le_cosh x]
  have hx : value outLo ≤ 2 * Real.cosh x / 2 ∧ 2 * Real.cosh x / 2 ≤ value outHi :=
    halfBounds_enclosure (α := α) h hlow hup
  constructor <;> linarith [hx.1, hx.2]

/-- `logSoftmaxBounds lo hi k` encloses entry `k` of the log-softmax of `t` when `lo`
and `hi` enclose `t` at every index: at R that entry is `t k - log (∑ j, exp (t j))`
(`LogSoftmax.getScalar_vec_eq_sub_log`), which is
`-log (∑ j, if j = k then 1 else exp (t j - t k))`. -/
theorem logSoftmaxBounds_encloses {n : Nat} {lo hi : Fin n → α} {k : Fin n} {b : α × α}
    {t : Tensor ℝ [n]} (ht : ∀ j, value (lo j) ≤ t.getScalar j ∧ t.getScalar j ≤ value (hi j))
    (h : logSoftmaxBounds lo hi k = some b) :
    value b.1 ≤ (LogSoftmax.vec t).getScalar k ∧
      (LogSoftmax.vec t).getScalar k ≤ value b.2 := by
  cases n with
  | zero => exact k.elim0
  | succ n =>
  set x : Fin (n + 1) → ℝ := fun j => t.getScalar j with hxdef
  have hshift : x k - Real.log (∑ j, Real.exp (x j)) =
      -Real.log (∑ j, if j = k then 1 else Real.exp (x j - x k)) := by
    have hs : (∑ j, if j = k then (1 : ℝ) else Real.exp (x j - x k)) =
        ∑ j, Real.exp (x j - x k) :=
      Finset.sum_congr rfl fun j _ => by split_ifs with hj <;> simp [hj]
    have hpos : 0 < ∑ j, Real.exp (x j - x k) :=
      Finset.sum_pos (fun j _ => Real.exp_pos _) ⟨k, Finset.mem_univ _⟩
    have hfac : (∑ j, Real.exp (x j)) = Real.exp (x k) * ∑ j, Real.exp (x j - x k) := by
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun j _ => by rw [← Real.exp_add]; ring_nf
    rw [hs, hfac, Real.log_mul (Real.exp_pos _).ne' hpos.ne', Real.log_exp]; ring
  unfold logSoftmaxBounds at h
  obtain ⟨e, he, h⟩ := Option.bind_eq_some_iff.mp h
  obtain ⟨l, hl, h⟩ := Option.bind_eq_some_iff.mp h
  obtain rfl := Option.some.inj h
  have hterm : ∀ j, value (e j).1 ≤ (if j = k then 1 else Real.exp (x j - x k)) ∧
      (if j = k then 1 else Real.exp (x j - x k)) ≤ value (e j).2 := by
    intro j
    have hj := Tensor.sequenceFinM_get_of_eq_some he j
    by_cases hjk : j = k
    · simp only [hjk, ↓reduceIte, Option.some.injEq] at hj ⊢
      simp [← hj, LawfulBoundOps.toReal_one (α := α)]
    · simp only [hjk, ↓reduceIte] at hj ⊢
      have hd := LawfulBoundOps.subDown_le (lo j) (hi k)
      have hu := LawfulBoundOps.le_subUp (hi j) (lo k)
      exact LawfulNonlinearBoundOps.expBounds_enclosure (α := α) hj
        (by linarith [(ht j).1, (ht k).2]) (by linarith [(ht j).2, (ht k).1])
  have hs := sum_encloses (fun j => (e j).1) (fun j => (e j).2)
    (fun j => if j = k then 1 else Real.exp (x j - x k)) hterm
  have hL := LawfulNonlinearBoundOps.logBounds_enclosure (α := α) hl hs.1 hs.2
  have hd := LawfulBoundOps.subDown_le (0 : α) l.2
  have hu := LawfulBoundOps.le_subUp (0 : α) l.1
  rw [LawfulBoundOps.toReal_zero] at hd hu
  rw [LogSoftmax.getScalar_vec_eq_sub_log]
  change value _ ≤ x k - Real.log (∑ j, Real.exp (x j)) ∧
    x k - Real.log (∑ j, Real.exp (x j)) ≤ value _
  rw [hshift]
  exact ⟨by linarith [hL.2], by linarith [hL.1]⟩

/-- `logSoftmaxEntry dim lo hi c` encloses entry `c` of the log-softmax of `x` along `dim`
when `lo` and `hi` enclose `x` at every coordinate. -/
theorem logSoftmaxEntry_encloses {s : Shape} (dim : Fin s.rank) {lo hi : Tensor α s}
    {x : Tensor ℝ s} (hx : ∀ c, value (lo c) ≤ x c ∧ x c ≤ value (hi c)) (c : s.Coord)
    {b : α × α} (h : logSoftmaxEntry dim lo hi c = some b) :
    value b.1 ≤ Tensor.logSoftmax x dim c ∧ Tensor.logSoftmax x dim c ≤ value b.2 := by
  rw [Tensor.logSoftmax_apply]
  refine logSoftmaxBounds_encloses (fun j => ?_) h
  simp only [getScalar_eq_apply, fiber_apply]
  exact hx _

/-- `expm1Bounds` encloses `exp x - 1`. -/
theorem expm1Bounds_enclosure :
    UnaryEnclosure (fun x => Real.exp x - 1) (expm1Bounds (α := α)) := by
  intro lo hi outLo outHi x h hlo hhi
  unfold expm1Bounds at h
  simp only [Option.bind_eq_bind] at h
  obtain ⟨e, he, h⟩ := Option.bind_eq_some_iff.mp h
  simp only [Option.some.injEq, Prod.mk.injEq] at h
  obtain ⟨rfl, rfl⟩ := h
  have hex := LawfulNonlinearBoundOps.expBounds_enclosure (α := α) he hlo hhi
  have hd := LawfulBoundOps.subDown_le e.1 (1 : α)
  have hu := LawfulBoundOps.le_subUp e.2 (1 : α)
  rw [LawfulBoundOps.toReal_one] at hd hu
  exact ⟨by linarith [hex.1], by linarith [hex.2]⟩

/-- `log1pBounds` encloses `log (1 + x)`. -/
theorem log1pBounds_enclosure :
    UnaryEnclosure (fun x => Real.log (1 + x)) (log1pBounds (α := α)) := by
  intro lo hi outLo outHi x h hlo hhi
  have hd := LawfulBoundOps.addDown_le (1 : α) lo
  have hu := LawfulBoundOps.le_addUp (1 : α) hi
  rw [LawfulBoundOps.toReal_one] at hd hu
  exact LawfulNonlinearBoundOps.logBounds_enclosure (α := α) h (by linarith) (by linarith)

/-- `rsqrtBounds` encloses `1 / sqrt x`. -/
theorem rsqrtBounds_enclosure :
    UnaryEnclosure (fun x => 1 / Real.sqrt x) (rsqrtBounds (α := α)) := by
  intro lo hi outLo outHi x h hlo hhi
  unfold rsqrtBounds at h
  simp only [Option.bind_eq_bind] at h
  obtain ⟨r, hr, h⟩ := Option.bind_eq_some_iff.mp h
  have hrx := LawfulNonlinearBoundOps.sqrtBounds_enclosure (α := α) hr hlo hhi
  have h1 : value (1 : α) = 1 := LawfulBoundOps.toReal_one
  exact LawfulNonlinearBoundOps.divBounds_enclosure (α := α) h h1.le h1.ge hrx.1 hrx.2

/-- `tanBounds` encloses `tan`. -/
theorem tanBounds_enclosure : UnaryEnclosure Real.tan (tanBounds (α := α)) := by
  intro lo hi outLo outHi x h hlo hhi
  unfold tanBounds at h
  simp only [Option.bind_eq_bind] at h
  obtain ⟨s, hs, h⟩ := Option.bind_eq_some_iff.mp h
  obtain ⟨c, hc, h⟩ := Option.bind_eq_some_iff.mp h
  have hsx := LawfulNonlinearBoundOps.sinBounds_enclosure (α := α) hs hlo hhi
  have hcx := LawfulNonlinearBoundOps.cosBounds_enclosure (α := α) hc hlo hhi
  rw [Real.tan_eq_sin_div_cos]
  exact LawfulNonlinearBoundOps.divBounds_enclosure (α := α) h hsx.1 hsx.2 hcx.1 hcx.2

/-- `logbBounds b` encloses `logb b`. -/
theorem logbBounds_enclosure (b : Nat) :
    UnaryEnclosure (Real.logb b) (logbBounds (α := α) b) := by
  intro lo hi outLo outHi x h hlo hhi
  unfold logbBounds at h
  simp only [Option.bind_eq_bind] at h
  obtain ⟨l, hl, h⟩ := Option.bind_eq_some_iff.mp h
  obtain ⟨d, hd, h⟩ := Option.bind_eq_some_iff.mp h
  have hlx := LawfulNonlinearBoundOps.logBounds_enclosure (α := α) hl hlo hhi
  have hb := countBounds_encloses (α := α) b
  have hdb := LawfulNonlinearBoundOps.logBounds_enclosure (α := α) hd hb.1 hb.2
  exact LawfulNonlinearBoundOps.divBounds_enclosure (α := α) h hlx.1 hlx.2 hdb.1 hdb.2

/-- `exp2Bounds` encloses `2 ^ x`. -/
theorem exp2Bounds_enclosure :
    UnaryEnclosure (fun x => (2 : ℝ) ^ x) (exp2Bounds (α := α)) := by
  intro lo hi outLo outHi x h hlo hhi
  unfold exp2Bounds at h
  simp only [Option.bind_eq_bind] at h
  obtain ⟨d, hd, h⟩ := Option.bind_eq_some_iff.mp h
  have hb := countBounds_encloses (α := α) 2
  simp only [Nat.cast_ofNat] at hb
  have hdb := LawfulNonlinearBoundOps.logBounds_enclosure (α := α) hd hb.1 hb.2
  have hp := Graph.intervalMul_encloses (α := α) hdb.1 hdb.2 hlo hhi
  beta_reduce
  rw [Real.rpow_def_of_pos (by norm_num : (0 : ℝ) < 2)]
  exact LawfulNonlinearBoundOps.expBounds_enclosure (α := α) h hp.1 hp.2

/-- `arsinhAt a` bounds `arsinh a`. -/
theorem arsinhAt_mem {a l u : α} (h : arsinhAt a = some (l, u)) :
    value l ≤ Real.arsinh (value a) ∧ Real.arsinh (value a) ≤ value u := by
  unfold arsinhAt at h
  simp only [Option.bind_eq_bind] at h
  obtain ⟨r, hr, h⟩ := Option.bind_eq_some_iff.mp h
  have hd := LawfulBoundOps.addDown_le (1 : α) (BoundOps.mulDown a a)
  have hu := LawfulBoundOps.le_addUp (1 : α) (BoundOps.mulUp a a)
  have hmd := LawfulBoundOps.mulDown_le a a
  have hmu := LawfulBoundOps.le_mulUp a a
  rw [LawfulBoundOps.toReal_one] at hd hu
  have hrx := LawfulNonlinearBoundOps.sqrtBounds_enclosure (α := α) hr
    (x := 1 + value a ^ 2) (by rw [sq]; linarith) (by rw [sq]; linarith)
  have hsd := LawfulBoundOps.addDown_le a r.1
  have hsu := LawfulBoundOps.le_addUp a r.2
  rw [Real.arsinh]
  exact LawfulNonlinearBoundOps.logBounds_enclosure (α := α) h (by linarith [hrx.1])
    (by linarith [hrx.2])

/-- `arsinhBounds` encloses `arsinh`. -/
theorem arsinhBounds_enclosure :
    UnaryEnclosure Real.arsinh (arsinhBounds (α := α)) := by
  intro lo hi outLo outHi x h hlo hhi
  unfold arsinhBounds at h
  simp only [Option.bind_eq_bind] at h
  obtain ⟨l, hl, h⟩ := Option.bind_eq_some_iff.mp h
  obtain ⟨u, hu, h⟩ := Option.bind_eq_some_iff.mp h
  simp only [Option.some.injEq, Prod.mk.injEq] at h
  obtain ⟨rfl, rfl⟩ := h
  exact ⟨(arsinhAt_mem hl).1.trans (Real.arsinh_le_arsinh.mpr hlo),
    (Real.arsinh_le_arsinh.mpr hhi).trans (arsinhAt_mem hu).2⟩

/-- `hypotBounds` encloses `hypot` at `ℝ`. -/
theorem hypotBounds_enclosure :
    BinaryEnclosure (Context.hypot (α := ℝ)) (hypotBounds (α := α)) := by
  intro aLo aHi bLo bHi outLo outHi x y h hal hau hbl hbu
  rw [Context.hypot_real]
  have hp := Graph.intervalMul_encloses (α := α) hal hau hal hau
  have hq := Graph.intervalMul_encloses (α := α) hbl hbu hbl hbu
  have hd := LawfulBoundOps.addDown_le (Context.maximum (0 : α) (intervalMul aLo aHi aLo aHi).1)
    (Context.maximum 0 (intervalMul bLo bHi bLo bHi).1)
  have hu := LawfulBoundOps.le_addUp (intervalMul aLo aHi aLo aHi).2
    (intervalMul bLo bHi bLo bHi).2
  simp only [LawfulBoundOps.toReal_maximum, LawfulBoundOps.toReal_zero] at hd
  have hx2 : 0 ≤ x * x := mul_self_nonneg x
  have hy2 : 0 ≤ y * y := mul_self_nonneg y
  exact LawfulNonlinearBoundOps.sqrtBounds_enclosure (α := α) h
    (by linarith [max_le hx2 hp.1, max_le hy2 hq.1]) (by linarith [hp.2, hq.2])

/-- A box of `batchNormDenominator?` encloses `sqrt (max var 0 + epsilon)`. -/
theorem batchNormDenominator?_encloses {mean var gamma beta epsilon : α} {d : α × α}
    (h : batchNormDenominator? mean var gamma beta epsilon = some d) :
    value d.1 ≤ Real.sqrt (max (value var) 0 + value epsilon) ∧
      Real.sqrt (max (value var) 0 + value epsilon) ≤ value d.2 := by
  simp only [batchNormDenominator?, Option.bind_eq_bind] at h
  obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
  obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
  obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
  obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
  obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
  obtain ⟨r, hr, h⟩ := Option.bind_eq_some_iff.mp h
  obtain ⟨d', hd', h⟩ := Option.bind_eq_some_iff.mp h
  obtain ⟨d₀, hd₀, hcheck⟩ := Option.bind_eq_some_iff.mp hd'
  obtain rfl := finiteBounds?_eq_of_eq_some hr
  obtain rfl := finiteBounds?_eq_of_eq_some hcheck
  have hd : d₀ = d := by
    split at h
    · exact Option.some.inj h
    · contradiction
  subst hd
  have hv : value (Context.maximum var 0) = max (value var) 0 := by
    rw [LawfulBoundOps.toReal_maximum, LawfulBoundOps.toReal_zero (α := α)]
  have hdl : value (BoundOps.addDown (Context.maximum var 0) epsilon) ≤
      max (value var) 0 + value epsilon := by
    rw [← hv]; exact LawfulBoundOps.addDown_le _ _
  have hdu : max (value var) 0 + value epsilon ≤
      value (BoundOps.addUp (Context.maximum var 0) epsilon) := by
    rw [← hv]; exact LawfulBoundOps.le_addUp _ _
  exact LawfulNonlinearBoundOps.sqrtBounds_enclosure hd₀ hdl hdu

/-- `batchNormBounds` encloses `(x - mean) / r * gamma + beta` when its endpoints
enclose `x` and the endpoints of `d` enclose `r`. -/
theorem batchNormBounds_encloses {lo hi mean gamma beta : α} {d : α × α} {x r : ℝ}
    (hx : value lo ≤ x ∧ x ≤ value hi) (hr : value d.1 ≤ r ∧ r ≤ value d.2) {b : α × α}
    (h : batchNormBounds lo hi mean gamma beta d = some b) :
    value b.1 ≤ (x - value mean) / r * value gamma + value beta ∧
      (x - value mean) / r * value gamma + value beta ≤ value b.2 := by
  simp only [batchNormBounds, Option.bind_eq_bind] at h
  obtain ⟨_, -, h₁⟩ := Option.bind_eq_some_iff.mp h
  obtain ⟨c, hc, h₂⟩ := Option.bind_eq_some_iff.mp h₁
  obtain ⟨q, hq, h₃⟩ := Option.bind_eq_some_iff.mp h₂
  obtain ⟨p, hp, h₄⟩ := Option.bind_eq_some_iff.mp h₃
  obtain ⟨q₀, hq₀, hqc⟩ := Option.bind_eq_some_iff.mp hq
  clear h h₁ h₂ h₃ hq
  obtain rfl := finiteBounds?_eq_of_eq_some hc
  obtain rfl := finiteBounds?_eq_of_eq_some hqc
  obtain rfl := finiteBounds?_eq_of_eq_some hp
  obtain rfl := finiteBounds?_eq_of_eq_some h₄
  have hql : value (BoundOps.subDown lo mean) ≤ x - value mean :=
    (LawfulBoundOps.subDown_le _ _).trans (sub_le_sub hx.1 le_rfl)
  have hqu : x - value mean ≤ value (BoundOps.subUp hi mean) :=
    (sub_le_sub hx.2 le_rfl).trans (LawfulBoundOps.le_subUp _ _)
  have hqq := LawfulNonlinearBoundOps.divBounds_enclosure hq₀ hql hqu hr.1 hr.2
  have hm := Graph.intervalMul_encloses hqq.1 hqq.2 (le_refl (value gamma))
    (le_refl (value gamma))
  exact ⟨(LawfulBoundOps.addDown_le _ _).trans (add_le_add hm.1 le_rfl),
    (add_le_add hm.2 le_rfl).trans (LawfulBoundOps.le_addUp _ _)⟩

/-- A successful mean transfer divides the enclosed sum by the exact coordinate count. -/
theorem _root_.TorchLean.CROWN.Directed.boxMean?_encloses {x box : FlatBox α}
    {n : Nat} {f g : Nat → ℝ} (hx : RowEncloses x n f) (hg : g 0 = (∑ i : Fin n, f i.val) / n)
    (hbox : boxMean? x = some box) :
    RowEncloses box 1 g := by
  obtain ⟨d, lo, hi⟩ := x
  have hd := hx.1
  change d = n at hd
  subst d
  rw [rowEncloses_iff] at hx
  unfold boxMean? at hbox
  obtain ⟨⟨outLo, outHi⟩, hmean, hbox⟩ := Option.bind_eq_some_iff.mp hbox
  have hn : 0 < n := by
    by_contra hn
    have hn : n = 0 := Nat.eq_zero_of_not_pos hn
    subst hn
    simp [rowMean?] at hmean
  obtain rfl := Option.some.inj hbox
  rw [rowEncloses_iff]
  intro i
  have hi : i = 0 := Subsingleton.elim _ _
  subst hi
  simpa only [getScalar_full, Fin.val_zero, hg] using
    Graph.rowMean?_encloses hn _ (fun i => f i.val) hx hmean

/-- The layer norm over the last axis of a real matrix, with the real values of the rounded scale,
bias and a nonnegative epsilon, lies within `|gamma| * radius` of the bias, for the radius that
`layerNormAbsBound` gives the row width. -/
theorem layerNorm_abs_sub_le {rows w : Nat} (hw : 0 < w)
    {gamma beta : Tensor α [w]} {epsilon radius : α} (hepsilon : 0 ≤ value epsilon)
    (hradius : NonlinearBoundOps.layerNormAbsBound (α := α) w = some radius)
    (x : Tensor ℝ [rows, w]) (i : Fin rows) (j : Fin w) :
    |Tensor.get2 (Tensor.layerNorm x [⟨1, Nat.one_lt_two⟩] (Tensor.map value gamma)
        (Tensor.map value beta) (value epsilon)) i j - value (beta.getScalar j)| ≤
      |value (gamma.getScalar j)| * value radius := by
  have hl := Normalization.get2_layerNorm hw (value epsilon) x
    (Tensor.map value gamma) (Tensor.map value beta) i j
  erw [hl]
  rw [getScalar_map, getScalar_map, add_sub_cancel_right, abs_mul]
  exact (mul_le_mul_of_nonneg_right
    ((Graph.layerNorm_abs_le_sqrt hw x _ hepsilon i j).trans
      (LawfulNonlinearBoundOps.layerNormAbsBound_sound hradius)) (abs_nonneg _)).trans_eq
    (mul_comm _ _)

/-- An entry of the layer norm over the last axis of a real matrix, with the real values of the
rounded scale, bias and a nonnegative epsilon, lies in the range `layerNormRange` gives
its column. -/
theorem layerNormRange_encloses {rows w : Nat} (hw : 0 < w) {gamma beta : Tensor α [w]}
    {epsilon l u : α} (hepsilon : 0 ≤ value epsilon) (x : Tensor ℝ [rows, w]) (i : Fin rows)
    (j : Fin w)
    (h : layerNormRange w (gamma.getScalar j) (beta.getScalar j) = some (l, u)) :
    value l ≤ Tensor.get2 (Tensor.layerNorm x [⟨1, Nat.one_lt_two⟩] (Tensor.map value gamma)
        (Tensor.map value beta) (value epsilon)) i j ∧
      Tensor.get2 (Tensor.layerNorm x [⟨1, Nat.one_lt_two⟩] (Tensor.map value gamma)
        (Tensor.map value beta) (value epsilon)) i j ≤ value u := by
  unfold layerNormRange at h
  by_cases hw1 : w ≤ 1
  · rw [ite_eq_left hw1] at h
    obtain rfl : w = 1 := by omega
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj h)
    have h0 := Fintype.sum_balance (Tensor.get2 x i)
    simp only [Fintype.sum_subsingleton _ j, Fintype.balance_apply] at h0
    have hl := Normalization.get2_layerNorm hw (value epsilon) x
      (Tensor.map value gamma) (Tensor.map value beta) i j
    erw [hl]
    rw [h0, zero_div, zero_mul, zero_add, getScalar_map]
    exact ⟨le_rfl, le_rfl⟩
  · rw [ite_eq_right hw1] at h
    obtain ⟨r, hr, h⟩ := Option.map_eq_some_iff.mp h
    have habs := abs_sub_le_iff.mp
      (layerNorm_abs_sub_le (gamma := gamma) (beta := beta) hw hepsilon hr x i j)
    split_ifs at h with hgb
    · obtain ⟨hg, hb⟩ := Bool.and_eq_true_iff.mp hgb
      rw [LawfulBoundOps.toReal_eq_of_beq hg, LawfulBoundOps.toReal_eq_of_beq hb,
        LawfulBoundOps.toReal_one, LawfulBoundOps.toReal_zero, abs_one, one_mul, sub_zero,
        zero_sub] at habs
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj h
      refine ⟨(LawfulBoundOps.subDown_le _ _).trans ?_, habs.1⟩
      rw [LawfulBoundOps.toReal_zero]
      linarith [habs.2]
    · obtain ⟨rfl, rfl⟩ := Prod.mk.inj h
      have hs : |value (gamma.getScalar j)| * value r ≤ value (BoundOps.mulUp
          (Graph.absBound (gamma.getScalar j) (gamma.getScalar j)) r) := by
        refine (mul_le_mul_of_nonneg_right ?_ ((Real.sqrt_nonneg _).trans
          (LawfulNonlinearBoundOps.layerNormAbsBound_sound hr))).trans
          (LawfulBoundOps.le_mulUp _ _)
        exact abs_le_absBound le_rfl le_rfl
      exact ⟨(LawfulBoundOps.subDown_le _ _).trans (by linarith [habs.2]),
        le_trans (by linarith [habs.1]) (LawfulBoundOps.le_addUp _ _)⟩

omit [LawfulNonlinearBoundOps α] in
/-- A successful `layerNormRow?` has a positive epsilon, read as a real. -/
theorem _root_.TorchLean.CROWN.Graph.layerNormRow?_epsilon_pos
    {n : Nat} {lo hi gamma beta : Tensor α [n]} {epsilon : α} {out : Tensor α [n] × Tensor α [n]}
    (h : layerNormRow? lo hi gamma beta epsilon = some out) : 0 < value epsilon := by
  unfold layerNormRow? at h
  obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
  split at h
  · contradiction
  · rename_i hepsilon
    rw [← LawfulBoundOps.toReal_zero (α := α), ← LawfulBoundOps.lt_iff]
    simpa using hepsilon

/-- `layerNormBox?` encloses the layer norm over the last axis of a real matrix, with the
real values of the rounded scale, bias and epsilon, when its input box encloses the matrix. -/
theorem layerNormBox?_encloses {rows w : Nat} (hw : 0 < w) {gamma beta : Tensor α [w]}
    {epsilon : α} {Bx B : FlatBox α} {x : Tensor ℝ [rows, w]}
    (hx : Encloses Bx (Graph.flatOfSome ⟨[rows, w], x⟩))
    (h : layerNormBox? rows w gamma beta epsilon Bx = some B) :
    Encloses B (Graph.flatOfSome ⟨[rows, w], Tensor.layerNorm x [⟨1, Nat.one_lt_two⟩]
      (Tensor.map value gamma) (Tensor.map value beta) (value epsilon)⟩) := by
  unfold layerNormBox? at h
  dsimp only at h
  split at h
  · rename_i hd
    simp only [Option.bind_eq_bind] at h
    obtain ⟨bounds, hbounds, h⟩ := Option.bind_eq_some_iff.mp h
    obtain rfl := Option.some.inj h
    have hreal : realTensor [rows, w]
        (fun i => getAtOrZero (Graph.flatOfSome ⟨[rows, w], x⟩).v [i]) = x := by
      apply Tensor.Rep.ext
      intro c
      rw [realTensor_apply]
      exact Tensor.getAtOrZero_reshape_toFin x c
    have hin := tensorEncloses_ibpUnflatten hd hx
    rw [hreal] at hin
    have hmap (t : Tensor α [w]) :
        Tensor.map value t = Tensor.ofFn fun k => value (t.getScalar k) :=
      Tensor.ext_vector fun k => by rw [getScalar_map, getScalar_ofFn]
    have e : Tensor.layerNorm x [⟨1, Nat.one_lt_two⟩] (Tensor.map value gamma)
        (Tensor.map value beta) (value epsilon) = Tensor.layerNorm x [⟨1, Nat.one_lt_two⟩]
          (Tensor.ofFn fun k : Fin w => value (gamma.getScalar k))
          (Tensor.ofFn fun k : Fin w => value (beta.getScalar k)) (value epsilon) :=
      congrArg₂ (fun g b => Tensor.layerNorm x [⟨1, Nat.one_lt_two⟩] g b (value epsilon))
        (hmap gamma) (hmap beta)
    rw [e]
    refine encloses_of_coord rfl fun ⟨i, j, ⟨⟩⟩ => ?_
    have hbi := Tensor.sequenceFinM_get_of_eq_some hbounds i
    have hrow := Graph.layerNormRow?_encloses hw _ _ gamma beta epsilon x i
      (fun k => by
        simpa only [Tensor.getScalar_eq_apply,
          Tensor.unstack_apply, Tensor.get2_eq_apply] using hin (i, k, ⟨⟩))
      hbi j
    simp only [Tensor.getAtOrZero_reshape_toFin,
      Tensor.dim_apply, Tensor.ofFn_apply]
    rcases hr : layerNormRange w (gamma.getScalar j) (beta.getScalar j) with _ | ⟨l, u⟩
    · simpa only [Tensor.get2_eq_apply] using hrow
    · have hlu := layerNormRange_encloses hw
        (Graph.layerNormRow?_epsilon_pos hbi).le x i j hr
      rw [e] at hlu
      simp only [Tensor.get2_eq_apply, LawfulBoundOps.toReal_maximum,
        LawfulBoundOps.toReal_minimum] at hrow hlu ⊢
      exact ⟨max_le hrow.1 hlu.1, le_min hrow.2 hlu.2⟩
  · cases h

end

end TorchLean.CROWN.Op.Directed

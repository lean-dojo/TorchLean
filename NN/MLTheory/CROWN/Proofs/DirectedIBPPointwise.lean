/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.DirectedIBPBasic
import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.Softplus
import NN.MLTheory.CROWN.Proofs.LayerNormEnclosure

/-!
# Rounded IBP for pointwise operations and mean squared error

The scalar softplus and safeLog laws follow from the existing directed arithmetic and nonlinear
contracts. Box transfers preserve coordinate enclosures, and mean squared error uses the directed
sum and count folds to divide by the exact number of coordinates.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean NN.IR
open TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph.Internal
open NN.MLTheory.CROWN.IntervalLemmas (value_min2 value_max2)
open scoped BigOperators

noncomputable section

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

/-- The coarse softplus interval encloses `log (1 + exp x)` using only directed addition. -/
theorem softplusBounds_enclosure :
    UnaryEnclosure (α := α) (fun x => Real.log (1 + Real.exp x))
      NonlinearBoundOps.softplusBounds := by
  intro lo hi outLo outHi x hout hxlo hxhi
  simp only [NonlinearBoundOps.softplusBounds, Option.some.injEq, Prod.mk.injEq] at hout
  obtain ⟨rfl, rfl⟩ := hout
  have hsoft := CertSoundness.softplus_envelope_real x
  rw [_root_.Proofs.softplus_spec_eq_log_one_add_exp] at hsoft
  constructor
  · simpa only [value_max2, (LawfulBoundOps.toReal_zero (α := α))] using
      (max_le_max hxlo le_rfl).trans hsoft.1
  · calc
      Real.log (1 + Real.exp x) ≤ max x 0 + 1 := hsoft.2
      _ ≤ max (value hi) 0 + 1 := add_le_add (max_le_max hxhi le_rfl) le_rfl
      _ = value (BoundOps.max2 hi 0) + value (1 : α) := by
        rw [value_max2, (LawfulBoundOps.toReal_zero (α := α)),
          (LawfulBoundOps.toReal_one (α := α))]
      _ ≤ value (BoundOps.addUp (BoundOps.max2 hi 0) 1) :=
        LawfulBoundOps.le_addUp _ _

/-- Both safeLog paths enclose `log (softplus x + epsilon)`, including its varying epsilon.

The executable positivity check supplies the logarithm's domain. If the logarithm transfer
declines, directed reciprocal bounds give `1 - 1/z ≤ log z ≤ z - 1`.
-/
theorem safeLogBounds_enclosure [NonlinearBoundOps α] [LawfulNonlinearBoundOps α] :
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

/-- Directed negation encloses absolute value without assuming that ordinary abs is exact. -/
theorem directed_abs_encloses {lo hi : α} {x : ℝ}
    (hlo : value lo ≤ x) (hhi : x ≤ value hi) :
    value (if lo < 0 then if 0 < hi then 0 else BoundOps.subDown 0 hi else lo) ≤ |x| ∧
      |x| ≤ value (BoundOps.max2 (BoundOps.subUp 0 lo) hi) := by
  constructor
  · split_ifs
    · simpa only [(LawfulBoundOps.toReal_zero (α := α))] using abs_nonneg x
    · calc
        value (BoundOps.subDown 0 hi) ≤ -value hi := by
          simpa only [(LawfulBoundOps.toReal_zero (α := α)), zero_sub] using
            LawfulBoundOps.subDown_le (0 : α) hi
        _ ≤ -x := neg_le_neg hhi
        _ ≤ |x| := neg_le_abs x
    · exact hlo.trans (le_abs_self x)
  · rw [value_max2]
    apply abs_le.mpr
    constructor
    · calc
        -max (value (BoundOps.subUp 0 lo)) (value hi) ≤
            -value (BoundOps.subUp 0 lo) := neg_le_neg (le_max_left _ _)
        _ ≤ value lo := by
          have h := LawfulBoundOps.le_subUp (0 : α) lo
          rw [(LawfulBoundOps.toReal_zero (α := α)), zero_sub] at h
          linarith
        _ ≤ x := hlo
    · exact hhi.trans (le_max_right _ _)

/-- The directed absolute-value box encloses the absolute value of every coordinate. -/
theorem boxAbs_encloses {x : FlatBox α} {n : Nat} {f : Nat → ℝ}
    (hx : RowEncloses x n f) :
    RowEncloses (boxAbs x) n (fun i => |f i|) := by
  obtain ⟨d, lo, hi⟩ := x
  have hd := hx.1
  change d = n at hd
  subst hd
  rw [rowEncloses_iff] at hx
  simp only [boxAbs]
  rw [rowEncloses_iff]
  intro i
  simpa only [getScalar_ofFn] using directed_abs_encloses (hx i).1 (hx i).2

/-- SafeLog uses the same enclosed scalar epsilon at every input coordinate. -/
theorem boxSafeLog?_encloses [NonlinearBoundOps α] [LawfulNonlinearBoundOps α]
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

/-- Componentwise maximum preserves enclosure in both operands. -/
theorem boxMaxElem_encloses {x y : FlatBox α} {n : Nat} {f g : Nat → ℝ}
    (hx : RowEncloses x n f) (hy : RowEncloses y n g) :
    RowEncloses (boxMaxElem x y) n (fun i => max (f i) (g i)) := by
  obtain ⟨dx, lx, ux⟩ := x
  obtain ⟨dy, ly, uy⟩ := y
  have hdx := hx.1
  have hdy := hy.1
  change dx = n at hdx
  change dy = n at hdy
  subst hdx hdy
  rw [rowEncloses_iff] at hx hy
  simp only [boxMaxElem, dite_true]
  rw [rowEncloses_iff]
  intro i
  simp only [Tensor.maxSpec, getScalar_map2Spec, LawfulBoundOps.toReal_max]
  exact ⟨max_le_max (hx i).1 (hy i).1, max_le_max (hx i).2 (hy i).2⟩

/-- Componentwise minimum preserves enclosure in both operands. -/
theorem boxMinElem_encloses [LawfulMinBoundOps α]
    {x y : FlatBox α} {n : Nat} {f g : Nat → ℝ}
    (hx : RowEncloses x n f) (hy : RowEncloses y n g) :
    RowEncloses (boxMinElem x y) n (fun i => min (f i) (g i)) := by
  obtain ⟨dx, lx, ux⟩ := x
  obtain ⟨dy, ly, uy⟩ := y
  have hdx := hx.1
  have hdy := hy.1
  change dx = n at hdx
  change dy = n at hdy
  subst hdx hdy
  rw [rowEncloses_iff] at hx hy
  simp only [boxMinElem, dite_true]
  rw [rowEncloses_iff]
  intro i
  simp only [Tensor.minSpec, getScalar_map2Spec, LawfulMinBoundOps.toReal_min]
  exact ⟨min_le_min (hx i).1 (hy i).1, min_le_min (hx i).2 (hy i).2⟩

private theorem value_min_lt (a b : α) :
    value (if a < b then a else b) = min (value a) (value b) := by
  simpa only [BoundOps.min2, Bool.decide_iff, min_comm] using value_min2 b a

private theorem value_max_gt (a b : α) :
    value (if a > b then a else b) = max (value a) (value b) := by
  simpa only [BoundOps.max2, Bool.decide_iff] using value_max2 a b

/-- Squaring uses directed endpoint products, with zero as a lower bound across zero. -/
theorem boxSquare_encloses {x : FlatBox α} {n : Nat} {f : Nat → ℝ}
    (hx : RowEncloses x n f) :
    RowEncloses (boxSquare x) n (fun i => (f i) ^ 2) := by
  obtain ⟨d, lo, hi⟩ := x
  have hd := hx.1
  change d = n at hd
  subst hd
  rw [rowEncloses_iff] at hx
  simp only [boxSquare]
  rw [rowEncloses_iff]
  intro i
  simp only [getScalar_ofFn, value_max_gt]
  constructor
  · by_cases hl : lo.getScalar i < (0 : α)
    · by_cases hh : (0 : α) < hi.getScalar i
      · simpa only [hl, hh, ↓reduceIte, (LawfulBoundOps.toReal_zero (α := α))] using
          sq_nonneg (f i.val)
      · have hhi : value (hi.getScalar i) ≤ 0 := by
          rw [← (LawfulBoundOps.toReal_zero (α := α))]
          exact le_of_not_gt fun h => hh ((LawfulBoundOps.lt_iff _ _).mpr h)
        have hsquare := mul_self_le_mul_self (neg_nonneg.mpr hhi) (neg_le_neg (hx i).2)
        simp only [hl, hh, ↓reduceIte, value_min_lt]
        exact (min_le_right _ _).trans ((LawfulBoundOps.mulDown_le _ _).trans
          (by simpa only [neg_mul_neg, pow_two] using hsquare))
    · have hlo : 0 ≤ value (lo.getScalar i) := by
        rw [← (LawfulBoundOps.toReal_zero (α := α))]
        exact le_of_not_gt fun h => hl ((LawfulBoundOps.lt_iff _ _).mpr h)
      simp only [hl, ↓reduceIte, value_min_lt]
      exact (min_le_left _ _).trans ((LawfulBoundOps.mulDown_le _ _).trans
        (by simpa only [pow_two] using mul_self_le_mul_self hlo (hx i).1))
  · simpa only [pow_two] using
      (LayerNormDirected.square_le_max (hx i).1 (hx i).2).trans
        (max_le_max (LawfulBoundOps.le_mulUp _ _) (LawfulBoundOps.le_mulUp _ _))

/-- A successful mean transfer divides the enclosed sum by the exact coordinate count. -/
theorem boxMean?_encloses [NonlinearBoundOps α] [LawfulNonlinearBoundOps α]
    {x box : FlatBox α} {n : Nat} {f g : Nat → ℝ}
    (hx : RowEncloses x n f) (hg : g 0 = (∑ i : Fin n, f i.val) / n)
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
    simp [directedRowMean?] at hmean
  obtain rfl := Option.some.inj hbox
  rw [rowEncloses_iff]
  intro i
  have hi : i = 0 := Subsingleton.elim _ _
  subst hi
  simpa only [getScalar_full, Fin.val_zero, hg] using
    LayerNormDirected.directedRowMean?_encloses hn _ (fun i => f i.val) hx hmean

/-- Node kinds handled by the remaining pointwise IBP transfers. -/
def ibpPointwiseSupportedNode (node : Node) : Bool :=
  match node.kind with
  | .abs | .maxElem | .minElem | .softplus | .safeLog | .mseLoss
  | .randUniform _ | .bernoulliMask _ => true
  | _ => false

/-- Real semantics missing from `NodeEquation` for this family.

SafeLog is `log (log (1 + exp x) + epsilon)` with one scalar epsilon parent. MSE is the exact
mean of squared differences; a zero-length mean cannot yield a successful IBP transfer.
Random realizations have the declared output shape and lie in `[0, 1]`. Other kinds use `True`
here and retain their existing `NodeEquation`.
-/
def PointwiseRealNodeEquation (nodes : Array Node) (dims : Nat → Nat)
    (v : Nat → Nat → ℝ) (id : Nat) : Prop :=
  let node := nodes[id]!
  match node.kind with
  | .safeLog =>
      ∀ p q, binaryParents? node.parents = some (p, q) →
        dims p = dims id ∧ dims q = 1 ∧
          ∀ i : Fin (dims id),
            v id i.val = Real.log (Real.log (1 + Real.exp (v p i.val)) + v q 0)
  | .mseLoss =>
      ∀ p q, binaryParents? node.parents = some (p, q) →
        dims p = dims q ∧ dims id = 1 ∧
          v id 0 = (∑ i : Fin (dims p), (v p i.val - v q i.val) ^ 2) / dims p
  | .randUniform _ | .bernoulliMask _ =>
      dims id = node.outShape.size ∧
        ∀ i : Fin node.outShape.size, 0 ≤ v id i.val ∧ v id i.val ≤ 1
  | _ => True

/-- A successful pointwise IBP step encloses the node's real value.

Parent boxes use the same agreement and enclosure interface as the basic step theorem. The
existing `NodeEquation` supplies abs, min, max, and softplus semantics; only safeLog, MSE, and
random support need the supplemental real equations.
-/
theorem ibpStepNodeAt?_pointwise_encloses
    [NonlinearBoundOps α] [LawfulNonlinearBoundOps α] [LawfulMinBoundOps α]
    {nodes : Array Node} {ps : ParamStore α} {boxes ibp : Array (Option (FlatBox α))}
    {dims : Nat → Nat} {v : Nat → Nat → ℝ} {id : Nat}
    (hsupported : ibpPointwiseSupportedNode nodes[id]! = true)
    (heq : NodeEquation nodes ps ibp dims v id)
    (hpointwise : PointwiseRealNodeEquation nodes dims v id)
    (hagree : ∀ p ∈ nodes[id]!.parents, (boxes[p]?).join = ibp[p]!)
    (henc : ∀ p ∈ nodes[id]!.parents, ∀ box, ibp[p]! = some box →
      RowEncloses box (dims p) (v p))
    {box : FlatBox α} (hstep : ibpStepNodeAt? nodes ps boxes id nodes[id]! = some box) :
    RowEncloses box (dims id) (v id) := by
  have hget : ∀ p ∈ nodes[id]!.parents, ∀ B, (boxes[p]?).join = some B →
      RowEncloses B (dims p) (v p) := fun p hp B hB =>
    henc p hp B ((hagree p hp).symm.trans hB)
  have hunary {p : Nat} (hp : unaryParent? nodes[id]!.parents = some p) :=
    hget p (mem_of_unaryParent?_eq_some hp)
  have hleft {p q : Nat} (hpq : binaryParents? nodes[id]!.parents = some (p, q)) :=
    hget p (fst_mem_of_binaryParents?_eq_some hpq)
  have hright {p q : Nat} (hpq : binaryParents? nodes[id]!.parents = some (p, q)) :=
    hget q (snd_mem_of_binaryParents?_eq_some hpq)
  unfold NodeEquation at heq
  unfold PointwiseRealNodeEquation at hpointwise
  rw [ibpStepNodeAt?] at hstep
  cases hk : (nodes[id]!).kind <;>
    simp only [ibpPointwiseSupportedNode, hk, Bool.false_eq_true] at hsupported <;>
    simp only [hk] at heq hpointwise <;>
    rw [hk] at hstep <;> dsimp only at hstep
  case abs =>
    obtain ⟨p, B, hp, hB, hbox⟩ := unary_lookup (k := fun B => some (boxAbs B)) hstep
    exact unary_step (transfer := fun B => some (boxAbs B))
      (fun hB hb => by obtain rfl := Option.some.inj hb; exact boxAbs_encloses hB)
      (heq p hp) (hunary hp B hB) hbox
  case maxElem =>
    obtain ⟨p, q, X, Y, hpq, hX, hY, hbox⟩ := binary_lookup
      (k := fun x y => if x.dim = y.dim then some (boxMaxElem x y) else none) hstep
    split at hbox
    · obtain rfl := Option.some.inj hbox
      obtain ⟨hdp, hdq, hv⟩ := heq p q hpq
      have hX' := hleft hpq X hX
      have hY' := hright hpq Y hY
      rw [hdp] at hX'
      rw [hdq] at hY'
      exact (boxMaxElem_encloses hX' hY').congr hv
    · simp at hbox
  case minElem =>
    obtain ⟨p, q, X, Y, hpq, hX, hY, hbox⟩ := binary_lookup
      (k := fun x y => if x.dim = y.dim then some (boxMinElem x y) else none) hstep
    split at hbox
    · obtain rfl := Option.some.inj hbox
      obtain ⟨hdp, hdq, hv⟩ := heq p q hpq
      have hX' := hleft hpq X hX
      have hY' := hright hpq Y hY
      rw [hdp] at hX'
      rw [hdq] at hY'
      exact (boxMinElem_encloses hX' hY').congr hv
    · simp at hbox
  case softplus =>
    obtain ⟨p, B, hp, hB, hbox⟩ := unary_lookup (k := boxSoftplus?) hstep
    exact unary_step (transfer := boxSoftplus?) (fun hB hb =>
      boxUnaryEnclosure?_encloses softplusBounds_enclosure hB hb)
      (heq p hp) (hunary hp B hB) hbox
  case safeLog =>
    obtain ⟨p, q, B, E, hpq, hB, hE, hbox⟩ := binary_lookup (k := boxSafeLog?) hstep
    obtain ⟨hdp, hdq, hv⟩ := hpointwise p q hpq
    have hB' := hleft hpq B hB
    have hE' := hright hpq E hE
    rw [hdp] at hB'
    rw [hdq] at hE'
    exact (boxSafeLog?_encloses hB' hE' hbox).congr hv
  case mseLoss =>
    obtain ⟨p, q, Y, T, hpq, hY, hT, hbox⟩ := binary_lookup
      (k := fun y t => if y.dim = t.dim then boxMean? (boxSquare (boxSub y t)) else none)
      hstep
    split at hbox
    · obtain ⟨hd, h1, hv⟩ := hpointwise p q hpq
      have hY' := hleft hpq Y hY
      have hT' := hright hpq T hT
      rw [← hd] at hT'
      rw [h1]
      exact boxMean?_encloses (boxSquare_encloses (boxSub_encloses hY' hT')) hv hbox
    · simp at hbox
  case randUniform | bernoulliMask =>
    obtain rfl := Option.some.inj hstep
    obtain ⟨hd, hv⟩ := hpointwise
    rw [hd, rowEncloses_iff]
    intro i
    simpa only [getScalar_full, (LawfulBoundOps.toReal_zero (α := α)),
      (LawfulBoundOps.toReal_one (α := α))] using hv i

end

end NN.MLTheory.CROWN.Graph.DirectedBackward

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.FDeriv.PullbackTensor
public import TorchLean.Tensor.Op.Table.Conv
public import TorchLean.NN.Op.Certificates
public import TorchLean.NN.Tactic.OpDeriv
public import TorchLean.Tensor.Op.Cert

/-!
# Derivatives of pooling

We prove that the stored pullback of `Op.pool` is the adjoint of the derivative of its forward,
for every kind of pool and every window configuration. We write each entry of `poolJVP op x u` as
a sum over the input coordinates `c` of a weight `poolWeight op x o c` times `u c`: a sum pool
counts the window cells that read `c`, an average pool divides that count by `poolCount`, and a
maximum or minimum pool puts weight one on the coordinate that `poolIndices` selects. Padding
cells read no coordinate. The stored backward is the same weighted sum taken over the outputs,
which gives the transpose identity `inner_poolJVP` at every input. At a tie `poolIndices` selects
the coordinate read by the first cell of the window in row-major order that reads a largest
(smallest) entry (`poolIndices_true_eq_some_iff`), so the stored backward sends the cotangent entry
there (`pool_backward_eq`).

Sum and average pools are linear, so `poolJVP op x` is their derivative at every input. For a
maximum or minimum pool we assume that each window takes its largest (smallest) entry at one input
coordinate only. A strict inequality between two entries persists near the input, so near it every
window selects the same coordinate, the pool agrees with the linear map `poolJVP op x`, and this
map is its derivative. A window of padding only selects nothing and gives zero.

The assumption is exact (`differentiableAt_pool_iff`). If a window of a maximum pool takes its
largest entry at two coordinates `c ≠ c'`, then for `t > 0` its entry at `x + t • (2 • e c - e c')`
exceeds its entry at `x + t • e c` by `t`, where `e c` is one at `c` and zero elsewhere. A
derivative `D` would then satisfy `D (e c) - D (e c') = 1` and, with `c` and `c'` swapped,
`D (e c') - D (e c) = 1`, which is impossible. A minimum pool is the same with the signs reversed.
-/

@[expose] public section

open scoped RealInnerProductSpace

open Filter Topology

namespace List

/-- A left fold over `finRange n` that adds `f i` exactly when `p i` holds adds the sum of the
guarded terms to the initial value. -/
theorem finRange_foldl_ite_add_eq_add_sum {β : Type} [AddCommMonoid β] {n : Nat}
    (p : Fin n → Prop) [DecidablePred p] (f : Fin n → β) (acc : β) :
    (List.finRange n).foldl (fun s i => if p i then s + f i else s) acc =
      acc + ∑ i, if p i then f i else 0 := by
  have h : (fun (s : β) (i : Fin n) => if p i then s + f i else s) =
      fun s i => s + if p i then f i else 0 := by
    funext s i
    by_cases hi : p i
    · rw [ite_eq_left hi, ite_eq_left hi]
    · rw [ite_eq_right hi, ite_eq_right hi, add_zero]
  rw [h, finRange_foldl_add_eq_add_sum]

end List

namespace TorchLean

namespace Tensor

variable {s : Shape} {dims : List (Fin s.rank)} {kernel stride padLow padHigh dilation : List Nat}
  {ceilMode : Bool}

/-- A double sum over the flat indices of two shapes, each read as its coordinate, is the double
sum over the coordinates. -/
theorem sum_unlinearize_sum_unlinearize {M : Type} [AddCommMonoid M] (u v : Shape)
    (f : u.Coord → v.Coord → M) :
    ∑ i : Fin (Shape.size u), ∑ j : Fin (Shape.size v),
        f (Shape.Coord.unlinearize i) (Shape.Coord.unlinearize j) = ∑ a, ∑ b, f a b :=
  (sum_unlinearize u fun a => ∑ j : Fin (Shape.size v), f a (Shape.Coord.unlinearize j)).trans
    (Finset.sum_congr rfl fun a _ => sum_unlinearize v (f a))

/-- With real entries, `v` replaces `b` in a maximum pool exactly when `b < v`. -/
theorem poolWins_true (v b : ℝ) : poolWins true v b = decide (b < v) := by
  unfold poolWins Context.gtBool
  rcases lt_trichotomy v b with h | rfl | h
  · simp [h, h.not_gt]
  · simp
  · simp [h]

/-- With real entries, `v` replaces `b` in a minimum pool exactly when `v < b`. -/
theorem poolWins_false (v b : ℝ) : poolWins false v b = decide (v < b) := by
  unfold poolWins Context.gtBool
  rcases lt_trichotomy v b with h | rfl | h
  · simp [h]
  · simp
  · simp [h, h.not_gt]

/-- The entry of `x` under the window cell `k` of `o`, or zero in the padding, is the sum over the
input coordinates `c` of `x c` when the cell reads `c`. -/
theorem poolInput_eq_sum (x : Tensor ℝ s)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord)
    (k : (s.poolWindow dims kernel).Coord) :
    poolInput x o k = ∑ c, if poolSource o k = some c then x c else 0 := by
  rw [poolInput_def]
  cases poolSource o k <;> simp

/-- The row-major sum over the window of `o` is the sum over the input coordinates `c` of the
number of window cells that read `c` times `x c`. -/
theorem foldl_poolInput_eq_sum (x : Tensor ℝ s)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) :
    (List.finRange (Shape.size (s.poolWindow dims kernel))).foldl
        (fun acc i => acc + poolInput x o (Shape.Coord.unlinearize i)) 0 =
      ∑ c, (∑ k, if poolSource o k = some c then (1 : ℝ) else 0) * x c := by
  rw [List.finRange_foldl_add_eq_add_sum, zero_add, sum_unlinearize _ fun k => poolInput x o k]
  simp only [poolInput_eq_sum, Finset.sum_mul, ite_mul, one_mul, zero_mul]
  exact Finset.sum_comm

/-- The weight with which `pool op` at `x` reads the input coordinate `c` into the output entry
`o`: the number of window cells over `c` for a sum, that number divided by `poolCount` for an
average, and one at the coordinate that `poolIndices` selects for a maximum or minimum. -/
noncomputable def poolWeight (op : PoolOp) (x : Tensor ℝ s)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) (c : s.Coord) : ℝ :=
  match op with
  | .sum => ∑ k, if poolSource o k = some c then 1 else 0
  | .max =>
      if poolIndices true x dims kernel stride padLow padHigh dilation ceilMode o = some c then 1
      else 0
  | .min =>
      if poolIndices false x dims kernel stride padLow padHigh dilation ceilMode o = some c then 1
      else 0
  | .avg countIncludePad =>
      (∑ k, if poolSource o k = some c then 1 else 0) / poolCount countIncludePad o

/-- An entry `o` of `poolJVP op x u` is the sum over the input coordinates `c` of the weight
`poolWeight op x o c` times `u c`. -/
theorem poolJVP_apply_eq_sum (op : PoolOp) (x u : Tensor ℝ s)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) :
    poolJVP op x u dims kernel stride padLow padHigh dilation ceilMode o =
      ∑ c, poolWeight op x o c * u c := by
  cases op with
  | sum => simp only [poolJVP, pool_sum_apply, foldl_poolInput_eq_sum, poolWeight]
  | max =>
    simp only [poolJVP, Rep.get_ofFn, poolWeight]
    cases poolIndices true x dims kernel stride padLow padHigh dilation ceilMode o <;> simp
  | min =>
    simp only [poolJVP, Rep.get_ofFn, poolWeight]
    cases poolIndices false x dims kernel stride padLow padHigh dilation ceilMode o <;> simp
  | avg _ =>
    simp only [poolJVP, pool_avg_apply, foldl_poolInput_eq_sum, Context.divNat_real, poolWeight]
    rw [Finset.sum_div]
    exact Finset.sum_congr rfl fun _ _ => (div_mul_eq_mul_div _ _ _).symm

/-- An entry `c` of `poolBackward op x δ` is the sum over the output coordinates `o` of the weight
`poolWeight op x o c` times `δ o`. -/
theorem poolBackward_apply_eq_sum (op : PoolOp) (x : Tensor ℝ s)
    (δ : Tensor ℝ (s.pool dims kernel stride padLow padHigh dilation ceilMode)) (c : s.Coord) :
    poolBackward op x dims kernel stride padLow padHigh dilation ceilMode δ c =
      ∑ o, poolWeight op x o c * δ o := by
  cases op with
  | sum =>
    simp only [poolBackward_sum_apply, List.finRange_foldl_ite_add_eq_add_sum,
      List.finRange_foldl_add_eq_add_sum, zero_add, poolWeight, Finset.sum_mul, ite_mul, one_mul,
      zero_mul]
    exact sum_unlinearize_sum_unlinearize _ _ fun o k =>
      if poolSource o k = some c then δ o else 0
  | max =>
    rw [poolBackward_max_apply, List.finRange_foldl_ite_add_eq_add_sum, zero_add]
    simp only [poolWeight, ite_mul, one_mul, zero_mul]
    exact sum_unlinearize _ fun o =>
      if poolIndices true x dims kernel stride padLow padHigh dilation ceilMode o = some c then δ o
      else 0
  | min =>
    rw [poolBackward_min_apply, List.finRange_foldl_ite_add_eq_add_sum, zero_add]
    simp only [poolWeight, ite_mul, one_mul, zero_mul]
    exact sum_unlinearize _ fun o =>
      if poolIndices false x dims kernel stride padLow padHigh dilation ceilMode o = some c then
        δ o
      else 0
  | avg countIncludePad =>
    simp only [poolBackward_avg_apply, Context.divNat_real, List.finRange_foldl_ite_add_eq_add_sum,
      List.finRange_foldl_add_eq_add_sum, zero_add, poolWeight, Finset.sum_div, ite_div, zero_div,
      Finset.sum_mul, ite_mul, div_mul_eq_mul_div, one_mul, zero_mul]
    exact sum_unlinearize_sum_unlinearize _ _ fun o k =>
      if poolSource o k = some c then δ o / (poolCount countIncludePad o : ℝ) else 0

/-- The map `u ↦ poolJVP op x u` of a pool at `x`, as a continuous linear map. -/
noncomputable def poolJVPCLM (op : PoolOp) (x : Tensor ℝ s) (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool) :
    Tensor ℝ s →L[ℝ] Tensor ℝ (s.pool dims kernel stride padLow padHigh dilation ceilMode) :=
  LinearMap.toContinuousLinearMap
    { toFun := fun u => poolJVP op x u dims kernel stride padLow padHigh dilation ceilMode
      map_add' := fun u v => Rep.ext fun o => by
        simp only [poolJVP_apply_eq_sum, Tensor.add_apply, mul_add, Finset.sum_add_distrib]
      map_smul' := fun a u => Rep.ext fun o => by
        simp only [poolJVP_apply_eq_sum, Rep.smul_apply, RingHom.id_apply, smul_eq_mul,
          Finset.mul_sum]
        exact Finset.sum_congr rfl fun _ _ => mul_left_comm _ _ _ }

/-- `poolJVPCLM op x dims kernel stride padLow padHigh dilation ceilMode` applied to `u` is
`poolJVP op x u dims kernel stride padLow padHigh dilation ceilMode`. -/
@[simp] theorem poolJVPCLM_apply (op : PoolOp) (x u : Tensor ℝ s) (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool) :
    poolJVPCLM op x dims kernel stride padLow padHigh dilation ceilMode u =
      poolJVP op x u dims kernel stride padLow padHigh dilation ceilMode :=
  rfl

/-- `poolBackward op x` is the transpose of `poolJVP op x` for the inner product of tensors, for
every kind of pool and every input `x`. -/
theorem inner_poolJVP (op : PoolOp) (x u : Tensor ℝ s)
    (δ : Tensor ℝ (s.pool dims kernel stride padLow padHigh dilation ceilMode)) :
    ⟪poolJVP op x u dims kernel stride padLow padHigh dilation ceilMode, δ⟫ =
      ⟪u, poolBackward op x dims kernel stride padLow padHigh dilation ceilMode δ⟫ := by
  simp only [inner_eq_sum, poolJVP_apply_eq_sum, poolBackward_apply_eq_sum, Finset.sum_mul,
    Finset.mul_sum]
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => by ring

/-- A left fold that keeps its selection when `src` gives nothing, takes the first value `src`
gives, and replaces its selection `b` by a value `d` exactly when `wins d b`, ends at `c` when `c`
beats every other value that `src` gives and is either selected at the start or given by `src`
along the list, provided the selection at the start loses to `c`. -/
theorem foldl_eq_some_of_wins {ι κ : Type*} {src : ι → Option κ} {wins : κ → κ → Bool}
    {step : Option κ → ι → Option κ} (hnone : ∀ best i, src i = none → step best i = best)
    (hfirst : ∀ i d, src i = some d → step none i = some d)
    (hnext : ∀ i d b, src i = some d → step (some b) i = if wins d b then some d else some b)
    {c : κ} (hc : ∀ i d, src i = some d → d ≠ c → wins c d = true ∧ wins d c = false) :
    ∀ (l : List ι) (best : Option κ), (∀ b, best = some b → b ≠ c → wins c b = true) →
      (best = some c ∨ ∃ i ∈ l, src i = some c) → l.foldl step best = some c := by
  intro l
  induction l with
  | nil =>
    intro best _ h
    simpa using h
  | cons i l ih =>
    intro best hbest h
    rw [List.foldl_cons]
    refine ih _ (fun b hb hbc => ?_) ?_
    · rcases hi : src i with _ | d
      · exact hbest b ((hnone best i hi).symm.trans hb) hbc
      · cases best with
        | none =>
          rw [hfirst i d hi] at hb
          cases hb
          exact (hc i _ hi hbc).1
        | some b0 =>
          rw [hnext i d b0 hi] at hb
          by_cases hw : wins d b0 = true
          · rw [ite_eq_left hw] at hb
            cases hb
            exact (hc i _ hi hbc).1
          · rw [ite_eq_right hw] at hb
            exact hbest b hb hbc
    · rcases h with rfl | ⟨j, hj, hjc⟩
      · left
        rcases hi : src i with _ | d
        · exact hnone _ i hi
        · rw [hnext i d c hi]
          by_cases hdc : d = c
          · rw [hdc, ite_self]
          · simp [(hc i d hi hdc).2]
      · rcases List.mem_cons.1 hj with rfl | hj
        · left
          cases best with
          | none => exact hfirst _ c hjc
          | some b0 =>
            rw [hnext _ c b0 hjc]
            by_cases hb0 : b0 = c
            · rw [hb0, ite_self]
            · simp [hbest b0 rfl hb0]
        · exact .inr ⟨j, hj, hjc⟩

/-- A maximum (`largest`) or minimum pool selects `c` in the window of `o` when the window reads
`c` and `c` strictly beats every other coordinate the window reads, in both directions of
`poolWins`. -/
theorem poolIndices_eq_some (largest : Bool) (y : Tensor ℝ s)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) (c : s.Coord)
    (hc : ∃ k, poolSource o k = some c)
    (hw : ∀ k d, poolSource o k = some d → d ≠ c →
      poolWins largest (y c) (y d) = true ∧ poolWins largest (y d) (y c) = false) :
    poolIndices largest y dims kernel stride padLow padHigh dilation ceilMode o = some c := by
  obtain ⟨k, hk⟩ := hc
  simp only [poolIndices, Rep.get_ofFn]
  refine foldl_eq_some_of_wins (src := fun i => poolSource o (Shape.Coord.unlinearize i))
    (wins := fun d b => poolWins largest (y d) (y b)) ?_ ?_ ?_ (fun _ d h => hw _ d h) _ none
    (fun _ hb => by cases hb) (.inr ⟨Shape.Coord.linearize k, List.mem_finRange _, ?_⟩)
  · rintro best i (h : poolSource o (Shape.Coord.unlinearize i) = none)
    simp only [h]
  · rintro i d (h : poolSource o (Shape.Coord.unlinearize i) = some d)
    simp only [h]
  · rintro i d b (h : poolSource o (Shape.Coord.unlinearize i) = some d)
    simp only [h]
  · simp only [Shape.Coord.unlinearize_linearize, hk]

/-- A maximum or minimum pool selects nothing in a window of padding only. -/
theorem poolIndices_eq_none (largest : Bool) (y : Tensor ℝ s)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord)
    (h : ∀ k, poolSource o k = none) :
    poolIndices largest y dims kernel stride padLow padHigh dilation ceilMode o = none := by
  simp only [poolIndices, Rep.get_ofFn, h, List.foldl_fixed]

/-- A left fold over a list sorted by `<` that skips the indices where `src` gives nothing, takes
the first value `src` gives, and replaces its selection `b` by a value `d` exactly when
`v b < v d`, either selects nothing on a list where `src` gives nothing, or selects a value `c`
that `src` gives at some `j` in the list. Every value `d` that `src` gives on the list then has
`v d ≤ v c`, and `v d < v c` when it is given before `j`. -/
theorem foldl_select_first_max {ι κ β : Type*} [Preorder ι] [LinearOrder β] {src : ι → Option κ}
    {v : κ → β} {step : Option κ → ι → Option κ}
    (hnone : ∀ best i, src i = none → step best i = best)
    (hfirst : ∀ i d, src i = some d → step none i = some d)
    (hwin : ∀ i d b, src i = some d → v b < v d → step (some b) i = some d)
    (hkeep : ∀ i d b, src i = some d → ¬v b < v d → step (some b) i = some b) :
    ∀ l : List ι, l.Pairwise (· < ·) →
      (l.foldl step none = none ∧ ∀ i ∈ l, src i = none) ∨
      ∃ c j, l.foldl step none = some c ∧ j ∈ l ∧ src j = some c ∧
        (∀ i ∈ l, ∀ d, src i = some d → v d ≤ v c) ∧
        ∀ i ∈ l, ∀ d, src i = some d → i < j → v d < v c := by
  intro l
  induction l using List.reverseRecOn with
  | nil => exact fun _ => .inl ⟨rfl, by simp⟩
  | append_singleton l a ih =>
    intro hl
    have hl' := List.pairwise_append.1 hl
    have hla : ∀ i ∈ l, i < a := fun i hi => hl'.2.2 i hi a (List.mem_singleton_self a)
    have hmem : ∀ i ∈ l ++ [a], i ∈ l ∨ i = a := fun i hi =>
      (List.mem_append.1 hi).imp_right List.mem_singleton.1
    rw [List.foldl_append, List.foldl_cons, List.foldl_nil]
    rcases ih hl'.1 with ⟨h0, hn⟩ | ⟨b, j, hb, hj, hsj, hle, hlt⟩
    · rw [h0]
      rcases ha : src a with _ | d
      · refine .inl ⟨hnone _ _ ha, fun i hi => ?_⟩
        rcases hmem i hi with hi | rfl
        · exact hn i hi
        · exact ha
      · refine .inr ⟨d, a, hfirst _ _ ha, List.mem_append_right _ (List.mem_singleton_self a), ha,
          fun i hi e he => ?_, fun i hi e he hia => ?_⟩
        · rcases hmem i hi with hi | rfl
          · rw [hn i hi] at he
            cases he
          · rw [ha] at he
            cases he
            exact le_rfl
        · rcases hmem i hi with hi | rfl
          · rw [hn i hi] at he
            cases he
          · exact absurd hia (lt_irrefl i)
    · rw [hb]
      rcases ha : src a with _ | d
      · refine .inr ⟨b, j, hnone _ _ ha, List.mem_append_left _ hj, hsj, fun i hi e he => ?_,
          fun i hi e he hij => ?_⟩
        · rcases hmem i hi with hi | rfl
          · exact hle i hi e he
          · rw [ha] at he
            cases he
        · rcases hmem i hi with hi | rfl
          · exact hlt i hi e he hij
          · rw [ha] at he
            cases he
      · by_cases hbd : v b < v d
        · refine .inr ⟨d, a, hwin _ _ _ ha hbd, List.mem_append_right _
            (List.mem_singleton_self a), ha, fun i hi e he => ?_, fun i hi e he hia => ?_⟩
          · rcases hmem i hi with hi | rfl
            · exact (hle i hi e he).trans hbd.le
            · rw [ha] at he
              cases he
              exact le_rfl
          · rcases hmem i hi with hi | rfl
            · exact (hle i hi e he).trans_lt hbd
            · exact absurd hia (lt_irrefl i)
        · refine .inr ⟨b, j, hkeep _ _ _ ha hbd, List.mem_append_left _ hj, hsj,
            fun i hi e he => ?_, fun i hi e he hij => ?_⟩
          · rcases hmem i hi with hi | rfl
            · exact hle i hi e he
            · rw [ha] at he
              cases he
              exact not_lt.1 hbd
          · rcases hmem i hi with hi | rfl
            · exact hlt i hi e he hij
            · exact absurd (hla j hj) (lt_asymm hij)

/-- A left fold over `finRange n` that skips the indices where `src` gives nothing, takes the
first value `src` gives, and replaces its selection `b` by a value `d` exactly when `v b < v d`
ends at `c` exactly when `src` gives `c` at some `j` such that every value `d` that `src` gives
has `v d ≤ v c` and every value `d` that `src` gives before `j` has `v d < v c`. -/
theorem foldl_finRange_eq_some_iff {κ β : Type*} [LinearOrder β] {n : Nat}
    {src : Fin n → Option κ} {v : κ → β} {step : Option κ → Fin n → Option κ}
    (hnone : ∀ best i, src i = none → step best i = best)
    (hfirst : ∀ i d, src i = some d → step none i = some d)
    (hwin : ∀ i d b, src i = some d → v b < v d → step (some b) i = some d)
    (hkeep : ∀ i d b, src i = some d → ¬v b < v d → step (some b) i = some b) (c : κ) :
    (List.finRange n).foldl step none = some c ↔ ∃ j, src j = some c ∧
      (∀ i d, src i = some d → v d ≤ v c) ∧ ∀ i d, src i = some d → i < j → v d < v c := by
  rcases foldl_select_first_max hnone hfirst hwin hkeep _ (List.pairwise_lt_finRange n) with
    ⟨h0, hn⟩ | ⟨b, j, hb, -, hsj, hle, hlt⟩
  · rw [h0]
    constructor
    · intro h
      cases h
    · rintro ⟨j, hj, -⟩
      rw [hn j (List.mem_finRange j)] at hj
      cases hj
  · rw [hb]
    constructor
    · rintro ⟨⟩
      exact ⟨j, hsj, fun i d h => hle i (List.mem_finRange i) d h,
        fun i d h hij => hlt i (List.mem_finRange i) d h hij⟩
    · rintro ⟨j', hj', hle', hlt'⟩
      rcases lt_trichotomy j j' with hjj | rfl | hjj
      · exact absurd (hle j' (List.mem_finRange j') c hj') (not_le.2 (hlt' j b hsj hjj))
      · exact hsj.symm.trans hj'
      · exact absurd (hle' j b hsj) (not_le.2 (hlt j' (List.mem_finRange j') c hj' hjj))

/-- Over the reals a maximum pool selects `c` in the window of `o` exactly when some cell `k` of
the window reads `c`, every entry the window reads is at most `x c`, and every entry read by a
cell before `k` in row-major order is below `x c`. -/
theorem poolIndices_true_eq_some_iff (x : Tensor ℝ s)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) (c : s.Coord) :
    poolIndices true x dims kernel stride padLow padHigh dilation ceilMode o = some c ↔
      ∃ k, poolSource o k = some c ∧ (∀ k' d, poolSource o k' = some d → x d ≤ x c) ∧
        ∀ k' d, poolSource o k' = some d →
          Shape.Coord.linearize k' < Shape.Coord.linearize k → x d < x c := by
  simp only [poolIndices, Rep.get_ofFn]
  rw [foldl_finRange_eq_some_iff (src := fun i => poolSource o (Shape.Coord.unlinearize i))
    (v := x) ?_ ?_ ?_ ?_]
  rotate_left
  · rintro best i (h : poolSource o (Shape.Coord.unlinearize i) = none)
    simp only [h]
  · rintro i d (h : poolSource o (Shape.Coord.unlinearize i) = some d)
    simp only [h]
  · rintro i d b (h : poolSource o (Shape.Coord.unlinearize i) = some d) hbd
    simp only [h, poolWins_true, hbd, decide_true, ite_true]
  · rintro i d b (h : poolSource o (Shape.Coord.unlinearize i) = some d) hbd
    simp only [h, poolWins_true, hbd, decide_false, Bool.false_eq_true, ite_false]
  constructor
  · rintro ⟨j, hj, hle, hlt⟩
    refine ⟨Shape.Coord.unlinearize j, hj, fun k' d h => hle (Shape.Coord.linearize k') d ?_,
      fun k' d h hk => hlt (Shape.Coord.linearize k') d ?_ ?_⟩
    · simpa only [Shape.Coord.unlinearize_linearize] using h
    · simpa only [Shape.Coord.unlinearize_linearize] using h
    · simpa only [Shape.Coord.linearize_unlinearize] using hk
  · rintro ⟨k, hk, hle, hlt⟩
    refine ⟨Shape.Coord.linearize k, ?_, fun i d h => hle _ d h, fun i d h hi => hlt _ d h ?_⟩
    · simpa only [Shape.Coord.unlinearize_linearize] using hk
    · simpa only [Shape.Coord.linearize_unlinearize] using hi

/-- Over the reals a minimum pool selects `c` in the window of `o` exactly when some cell `k` of
the window reads `c`, every entry the window reads is at least `x c`, and every entry read by a
cell before `k` in row-major order is above `x c`. -/
theorem poolIndices_false_eq_some_iff (x : Tensor ℝ s)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) (c : s.Coord) :
    poolIndices false x dims kernel stride padLow padHigh dilation ceilMode o = some c ↔
      ∃ k, poolSource o k = some c ∧ (∀ k' d, poolSource o k' = some d → x c ≤ x d) ∧
        ∀ k' d, poolSource o k' = some d →
          Shape.Coord.linearize k' < Shape.Coord.linearize k → x c < x d := by
  simp only [poolIndices, Rep.get_ofFn]
  rw [foldl_finRange_eq_some_iff (src := fun i => poolSource o (Shape.Coord.unlinearize i))
    (v := fun d => OrderDual.toDual (x d)) ?_ ?_ ?_ ?_]
  rotate_left
  · rintro best i (h : poolSource o (Shape.Coord.unlinearize i) = none)
    simp only [h]
  · rintro i d (h : poolSource o (Shape.Coord.unlinearize i) = some d)
    simp only [h]
  · rintro i d b (h : poolSource o (Shape.Coord.unlinearize i) = some d) hbd
    rw [OrderDual.toDual_lt_toDual] at hbd
    simp only [h, poolWins_false, hbd, decide_true, ite_true]
  · rintro i d b (h : poolSource o (Shape.Coord.unlinearize i) = some d) hbd
    rw [OrderDual.toDual_lt_toDual] at hbd
    simp only [h, poolWins_false, hbd, decide_false, Bool.false_eq_true, ite_false]
  simp only [OrderDual.toDual_le_toDual, OrderDual.toDual_lt_toDual]
  constructor
  · rintro ⟨j, hj, hle, hlt⟩
    refine ⟨Shape.Coord.unlinearize j, hj, fun k' d h => hle (Shape.Coord.linearize k') d ?_,
      fun k' d h hk => hlt (Shape.Coord.linearize k') d ?_ ?_⟩
    · simpa only [Shape.Coord.unlinearize_linearize] using h
    · simpa only [Shape.Coord.unlinearize_linearize] using h
    · simpa only [Shape.Coord.linearize_unlinearize] using hk
  · rintro ⟨k, hk, hle, hlt⟩
    refine ⟨Shape.Coord.linearize k, ?_, fun i d h => hle _ d h, fun i d h hi => hlt _ d h ?_⟩
    · simpa only [Shape.Coord.unlinearize_linearize] using hk
    · simpa only [Shape.Coord.linearize_unlinearize] using hi

/-- If any two coordinates where `v` takes its largest value over the window of `o` are equal,
then either the window is padding only or one coordinate `c` of the window has `v d < v c` for
every other coordinate `d` of the window. -/
theorem poolSource_eq_none_or_exists_lt {β : Type*} [LinearOrder β] (v : s.Coord → β)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord)
    (h : ∀ (k k' : (s.poolWindow dims kernel).Coord) (c c' : s.Coord), poolSource o k = some c →
      poolSource o k' = some c' →
      (∀ k'' d, poolSource o k'' = some d → v d ≤ v c ∧ v d ≤ v c') → c = c') :
    (∀ k, poolSource o k = none) ∨ ∃ c, (∃ k, poolSource o k = some c) ∧
      ∀ k d, poolSource o k = some d → d ≠ c → v d < v c := by
  classical
  by_cases hS : ∃ k c, poolSource o k = some c
  · right
    obtain ⟨c, hc, hmax⟩ := Finset.exists_max_image
      (Finset.univ.filter fun c => ∃ k, poolSource o k = some c) v
      (by obtain ⟨k, c, hkc⟩ := hS; exact ⟨c, Finset.mem_filter.2 ⟨Finset.mem_univ _, k, hkc⟩⟩)
    obtain ⟨k, hk⟩ := (Finset.mem_filter.1 hc).2
    have hle : ∀ k' d, poolSource o k' = some d → v d ≤ v c := fun k' d hd =>
      hmax d (Finset.mem_filter.2 ⟨Finset.mem_univ _, k', hd⟩)
    refine ⟨c, ⟨k, hk⟩, fun k' d hd hdc => lt_of_le_of_ne (hle k' d hd) fun hvd => hdc ?_⟩
    exact (h k k' c d hk hd fun k'' e he => ⟨hle k'' e he, (hle k'' e he).trans hvd.ge⟩).symm
  · left
    intro k
    by_contra hk
    exact hS ⟨k, Option.ne_none_iff_exists'.1 hk⟩

/-- A strict inequality between two entries of a real tensor holds on a neighbourhood of it. -/
theorem eventually_apply_lt {x : Tensor ℝ s} {a b : s.Coord} (h : x a < x b) :
    ∀ᶠ y in 𝓝 x, y a < y b :=
  (coordinateCLM a).continuous.continuousAt.eventually_lt
    (coordinateCLM b).continuous.continuousAt h

/-- If every window of a maximum pool takes its largest entry of `x` at one input coordinate only,
then every tensor near `x` has the same `poolIndices true` as `x`. -/
theorem eventually_poolIndices_true_eq {x : Tensor ℝ s}
    (h : ∀ (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord)
      (k k' : (s.poolWindow dims kernel).Coord) (c c' : s.Coord), poolSource o k = some c →
      poolSource o k' = some c' →
      (∀ k'' d, poolSource o k'' = some d → x d ≤ x c ∧ x d ≤ x c') → c = c') :
    ∀ᶠ y in 𝓝 x, ∀ o, poolIndices true y dims kernel stride padLow padHigh dilation ceilMode o =
      poolIndices true x dims kernel stride padLow padHigh dilation ceilMode o := by
  refine Filter.eventually_all.2 fun o => ?_
  rcases poolSource_eq_none_or_exists_lt (fun c => x c) o (h o) with hnone | ⟨c, hc, hlt⟩
  · exact Filter.Eventually.of_forall fun y => by
      rw [poolIndices_eq_none true y o hnone, poolIndices_eq_none true x o hnone]
  · have hsel : ∀ y : Tensor ℝ s, (∀ k d, poolSource o k = some d → d ≠ c → y d < y c) →
        poolIndices true y dims kernel stride padLow padHigh dilation ceilMode o = some c :=
      fun y hy => poolIndices_eq_some true y o c hc fun k d hd hdc => by
        rw [poolWins_true, poolWins_true]
        exact ⟨decide_eq_true (hy k d hd hdc), decide_eq_false (hy k d hd hdc).not_gt⟩
    have hev : ∀ᶠ y in 𝓝 x, ∀ k d, poolSource o k = some d → d ≠ c → y d < y c := by
      refine Filter.eventually_all.2 fun k => Filter.eventually_all.2 fun d => ?_
      by_cases hkd : poolSource o k = some d ∧ d ≠ c
      · filter_upwards [eventually_apply_lt (hlt k d hkd.1 hkd.2)] with y hy
        exact fun _ _ => hy
      · exact Filter.Eventually.of_forall fun _ hk hd => absurd ⟨hk, hd⟩ hkd
    filter_upwards [hev] with y hy
    rw [hsel y hy, hsel x hlt]

/-- If every window of a minimum pool takes its smallest entry of `x` at one input coordinate
only, then every tensor near `x` has the same `poolIndices false` as `x`. -/
theorem eventually_poolIndices_false_eq {x : Tensor ℝ s}
    (h : ∀ (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord)
      (k k' : (s.poolWindow dims kernel).Coord) (c c' : s.Coord), poolSource o k = some c →
      poolSource o k' = some c' →
      (∀ k'' d, poolSource o k'' = some d → x c ≤ x d ∧ x c' ≤ x d) → c = c') :
    ∀ᶠ y in 𝓝 x, ∀ o, poolIndices false y dims kernel stride padLow padHigh dilation ceilMode o =
      poolIndices false x dims kernel stride padLow padHigh dilation ceilMode o := by
  refine Filter.eventually_all.2 fun o => ?_
  rcases poolSource_eq_none_or_exists_lt (fun c => OrderDual.toDual (x c)) o
      (fun k k' c c' hk hk' hle => h o k k' c c' hk hk' fun k'' d hd =>
        ⟨OrderDual.toDual_le_toDual.1 (hle k'' d hd).1,
          OrderDual.toDual_le_toDual.1 (hle k'' d hd).2⟩) with hnone | ⟨c, hc, hlt⟩
  · exact Filter.Eventually.of_forall fun y => by
      rw [poolIndices_eq_none false y o hnone, poolIndices_eq_none false x o hnone]
  · have hlt' : ∀ k d, poolSource o k = some d → d ≠ c → x c < x d := fun k d hd hdc =>
      OrderDual.toDual_lt_toDual.1 (hlt k d hd hdc)
    have hsel : ∀ y : Tensor ℝ s, (∀ k d, poolSource o k = some d → d ≠ c → y c < y d) →
        poolIndices false y dims kernel stride padLow padHigh dilation ceilMode o = some c :=
      fun y hy => poolIndices_eq_some false y o c hc fun k d hd hdc => by
        rw [poolWins_false, poolWins_false]
        exact ⟨decide_eq_true (hy k d hd hdc), decide_eq_false (hy k d hd hdc).not_gt⟩
    have hev : ∀ᶠ y in 𝓝 x, ∀ k d, poolSource o k = some d → d ≠ c → y c < y d := by
      refine Filter.eventually_all.2 fun k => Filter.eventually_all.2 fun d => ?_
      by_cases hkd : poolSource o k = some d ∧ d ≠ c
      · filter_upwards [eventually_apply_lt (hlt' k d hkd.1 hkd.2)] with y hy
        exact fun _ _ => hy
      · exact Filter.Eventually.of_forall fun _ hk hd => absurd ⟨hk, hd⟩ hkd
    filter_upwards [hev] with y hy
    rw [hsel y hy, hsel x hlt']

/-- `pool op` has derivative `poolJVPCLM op x` at `x`. For a maximum (minimum) pool we assume that
every window takes its largest (smallest) entry of `x` at one input coordinate only, which
`differentiableAt_pool_iff` shows is also necessary for differentiability. -/
theorem hasFDerivAt_pool (op : PoolOp) (x : Tensor ℝ s)
    (hmax : op = .max →
      ∀ (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord)
        (k k' : (s.poolWindow dims kernel).Coord) (c c' : s.Coord),
        poolSource o k = some c → poolSource o k' = some c' →
        (∀ k'' d, poolSource o k'' = some d → x d ≤ x c ∧ x d ≤ x c') → c = c')
    (hmin : op = .min →
      ∀ (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord)
        (k k' : (s.poolWindow dims kernel).Coord) (c c' : s.Coord),
        poolSource o k = some c → poolSource o k' = some c' →
        (∀ k'' d, poolSource o k'' = some d → x c ≤ x d ∧ x c' ≤ x d) → c = c') :
    HasFDerivAt (fun y => pool op y dims kernel stride padLow padHigh dilation ceilMode)
      (poolJVPCLM op x dims kernel stride padLow padHigh dilation ceilMode) x := by
  refine (poolJVPCLM op x dims kernel stride padLow padHigh dilation
    ceilMode).hasFDerivAt.congr_of_eventuallyEq ?_
  cases op with
  | sum => exact Filter.Eventually.of_forall fun _ => rfl
  | max =>
    filter_upwards [eventually_poolIndices_true_eq (hmax rfl)] with y hy
    refine Rep.ext fun o => ?_
    cases h : poolIndices true x dims kernel stride padLow padHigh dilation ceilMode o <;>
      simp only [poolJVPCLM_apply, pool, poolJVP, Rep.get_ofFn, hy o, h]
  | min =>
    filter_upwards [eventually_poolIndices_false_eq (hmin rfl)] with y hy
    refine Rep.ext fun o => ?_
    cases h : poolIndices false x dims kernel stride padLow padHigh dilation ceilMode o <;>
      simp only [poolJVPCLM_apply, pool, poolJVP, Rep.get_ofFn, hy o, h]
  | avg _ => exact Filter.Eventually.of_forall fun _ => rfl

/-- A maximum pool reads `y c` in the window of `o` when the window reads `c` and every other
coordinate `d` that the window reads has `y d < y c`. -/
theorem pool_max_apply_of_lt (y : Tensor ℝ s)
    {o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord} {c : s.Coord}
    (hc : ∃ k, poolSource o k = some c)
    (hlt : ∀ k d, poolSource o k = some d → d ≠ c → y d < y c) :
    pool .max y dims kernel stride padLow padHigh dilation ceilMode o = y c := by
  have hsel : poolIndices true y dims kernel stride padLow padHigh dilation ceilMode o = some c :=
    poolIndices_eq_some true y o c hc fun k d hd hdc => by
      rw [poolWins_true, poolWins_true]
      exact ⟨decide_eq_true (hlt k d hd hdc), decide_eq_false (hlt k d hd hdc).not_gt⟩
  simp only [pool, Rep.get_ofFn, hsel]

/-- A minimum pool reads `y c` in the window of `o` when the window reads `c` and every other
coordinate `d` that the window reads has `y c < y d`. -/
theorem pool_min_apply_of_lt (y : Tensor ℝ s)
    {o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord} {c : s.Coord}
    (hc : ∃ k, poolSource o k = some c)
    (hlt : ∀ k d, poolSource o k = some d → d ≠ c → y c < y d) :
    pool .min y dims kernel stride padLow padHigh dilation ceilMode o = y c := by
  have hsel : poolIndices false y dims kernel stride padLow padHigh dilation ceilMode o = some c :=
    poolIndices_eq_some false y o c hc fun k d hd hdc => by
      rw [poolWins_false, poolWins_false]
      exact ⟨decide_eq_true (hlt k d hd hdc), decide_eq_false (hlt k d hd hdc).not_gt⟩
  simp only [pool, Rep.get_ofFn, hsel]

/-- A real function with derivative `D` at `x`, whose values on the rays `x + t • v` and
`x + t • w` differ by `a * t` for every `t > 0`, satisfies `D v - D w = a`. -/
theorem sub_eq_of_hasFDerivAt_of_ray {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    {F : E → ℝ} {D : E →L[ℝ] ℝ} {x : E} (hF : HasFDerivAt F D x) (v w : E) (a : ℝ)
    (h : ∀ t : ℝ, 0 < t → F (x + t • v) - F (x + t • w) = a * t) : D v - D w = a := by
  have hray : ∀ u : E, HasDerivAt (fun t : ℝ => F (x + t • u)) (D u) 0 := fun u => by
    have hl : HasDerivAt (fun t : ℝ => x + t • u) u 0 := by
      simpa using ((hasDerivAt_id' (0 : ℝ)).smul_const u).const_add x
    exact hF.comp_hasDerivAt_of_eq 0 hl (by simp)
  have hlin : HasDerivWithinAt (fun t : ℝ => a * t) a (Set.Ioi 0) 0 := by
    simpa using ((hasDerivAt_id' (0 : ℝ)).const_mul a).hasDerivWithinAt (s := Set.Ioi 0)
  exact (uniqueDiffWithinAt_Ioi 0).eq_deriv _ ((hray v).sub (hray w)).hasDerivWithinAt
    (hlin.congr (fun t ht => h t ht) (by simp))

/-- If a maximum pool is differentiable at `x`, then no window takes its largest entry of `x` at
two distinct input coordinates. -/
theorem eq_of_differentiableAt_pool_max {x : Tensor ℝ s}
    (h : DifferentiableAt ℝ
      (fun y => pool .max y dims kernel stride padLow padHigh dilation ceilMode) x)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord)
    (k k' : (s.poolWindow dims kernel).Coord) (c c' : s.Coord) (hk : poolSource o k = some c)
    (hk' : poolSource o k' = some c')
    (hle : ∀ k'' d, poolSource o k'' = some d → x d ≤ x c ∧ x d ≤ x c') : c = c' := by
  by_contra hcc
  obtain ⟨e, he⟩ : ∃ e : s.Coord → Tensor ℝ s, ∀ a d, e a d = if d = a then 1 else 0 :=
    ⟨fun a => Rep.ofFn fun d => if d = a then 1 else 0, fun a d => Rep.get_ofFn _ d⟩
  have he_self : ∀ a, e a a = 1 := fun a => (he a a).trans (ite_eq_left rfl)
  have he_ne : ∀ a d, d ≠ a → e a d = 0 := fun a d hda => (he a d).trans (ite_eq_right hda)
  have he_nonneg : ∀ a d, 0 ≤ e a d := fun a d => by
    by_cases hda : d = a
    · rw [hda, he_self]
      exact zero_le_one
    · rw [he_ne a d hda]
  have hray : ∀ a b : s.Coord, a ≠ b → (∃ k, poolSource o k = some a) →
      (∀ k d, poolSource o k = some d → x d ≤ x a) → ∀ t : ℝ, 0 < t →
        pool .max (x + t • ((2 : ℝ) • e a - e b)) dims kernel stride padLow padHigh dilation
            ceilMode o -
          pool .max (x + t • e a) dims kernel stride padLow padHigh dilation ceilMode o =
        1 * t := by
    intro a b hab ha hla t ht
    have h2 : pool .max (x + t • ((2 : ℝ) • e a - e b)) dims kernel stride padLow padHigh
        dilation ceilMode o = x a + t * (2 * 1 - 0) := by
      refine (pool_max_apply_of_lt (x + t • ((2 : ℝ) • e a - e b)) ha
        fun k d hd hda => ?_).trans ?_
      · simp only [Tensor.add_apply, Rep.smul_apply, Rep.hSub_apply, smul_eq_mul, he_self,
          he_ne a d hda, he_ne b a hab]
        linarith [hla k d hd, mul_nonneg ht.le (he_nonneg b d)]
      · simp only [Tensor.add_apply, Rep.smul_apply, Rep.hSub_apply, smul_eq_mul, he_self,
          he_ne b a hab]
    have h1 : pool .max (x + t • e a) dims kernel stride padLow padHigh dilation ceilMode o =
        x a + t * 1 := by
      refine (pool_max_apply_of_lt (x + t • e a) ha fun k d hd hda => ?_).trans ?_
      · simp only [Tensor.add_apply, Rep.smul_apply, smul_eq_mul, he_self, he_ne a d hda]
        linarith [hla k d hd]
      · simp only [Tensor.add_apply, Rep.smul_apply, smul_eq_mul, he_self]
    rw [h2, h1]
    ring
  have hF : HasFDerivAt
      (fun y => pool .max y dims kernel stride padLow padHigh dilation ceilMode o)
      ((coordinateCLM o).comp (fderiv ℝ
        (fun y => pool .max y dims kernel stride padLow padHigh dilation ceilMode) x)) x :=
    (coordinateCLM o).hasFDerivAt.comp x h.hasFDerivAt
  have h1 := sub_eq_of_hasFDerivAt_of_ray hF ((2 : ℝ) • e c - e c') (e c) 1
    (hray c c' hcc ⟨k, hk⟩ fun k'' d hd => (hle k'' d hd).1)
  have h2 := sub_eq_of_hasFDerivAt_of_ray hF ((2 : ℝ) • e c' - e c) (e c') 1
    (hray c' c (Ne.symm hcc) ⟨k', hk'⟩ fun k'' d hd => (hle k'' d hd).2)
  simp only [map_sub, map_smul, smul_eq_mul] at h1 h2
  linarith

/-- If a minimum pool is differentiable at `x`, then no window takes its smallest entry of `x` at
two distinct input coordinates. -/
theorem eq_of_differentiableAt_pool_min {x : Tensor ℝ s}
    (h : DifferentiableAt ℝ
      (fun y => pool .min y dims kernel stride padLow padHigh dilation ceilMode) x)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord)
    (k k' : (s.poolWindow dims kernel).Coord) (c c' : s.Coord) (hk : poolSource o k = some c)
    (hk' : poolSource o k' = some c')
    (hle : ∀ k'' d, poolSource o k'' = some d → x c ≤ x d ∧ x c' ≤ x d) : c = c' := by
  by_contra hcc
  obtain ⟨e, he⟩ : ∃ e : s.Coord → Tensor ℝ s, ∀ a d, e a d = if d = a then 1 else 0 :=
    ⟨fun a => Rep.ofFn fun d => if d = a then 1 else 0, fun a d => Rep.get_ofFn _ d⟩
  have he_self : ∀ a, e a a = 1 := fun a => (he a a).trans (ite_eq_left rfl)
  have he_ne : ∀ a d, d ≠ a → e a d = 0 := fun a d hda => (he a d).trans (ite_eq_right hda)
  have he_nonneg : ∀ a d, 0 ≤ e a d := fun a d => by
    by_cases hda : d = a
    · rw [hda, he_self]
      exact zero_le_one
    · rw [he_ne a d hda]
  have hray : ∀ a b : s.Coord, a ≠ b → (∃ k, poolSource o k = some a) →
      (∀ k d, poolSource o k = some d → x a ≤ x d) → ∀ t : ℝ, 0 < t →
        pool .min (x + t • (e b - (2 : ℝ) • e a)) dims kernel stride padLow padHigh dilation
            ceilMode o -
          pool .min (x + t • -e a) dims kernel stride padLow padHigh dilation ceilMode o =
        -1 * t := by
    intro a b hab ha hla t ht
    have h2 : pool .min (x + t • (e b - (2 : ℝ) • e a)) dims kernel stride padLow padHigh
        dilation ceilMode o = x a + t * (0 - 2 * 1) := by
      refine (pool_min_apply_of_lt (x + t • (e b - (2 : ℝ) • e a)) ha
        fun k d hd hda => ?_).trans ?_
      · simp only [Tensor.add_apply, Rep.smul_apply, Rep.hSub_apply, smul_eq_mul, he_self,
          he_ne a d hda, he_ne b a hab]
        linarith [hla k d hd, mul_nonneg ht.le (he_nonneg b d)]
      · simp only [Tensor.add_apply, Rep.smul_apply, Rep.hSub_apply, smul_eq_mul, he_self,
          he_ne b a hab]
    have h1 : pool .min (x + t • -e a) dims kernel stride padLow padHigh dilation ceilMode o =
        x a + t * -1 := by
      refine (pool_min_apply_of_lt (x + t • -e a) ha fun k d hd hda => ?_).trans ?_
      · simp only [Tensor.add_apply, Rep.smul_apply, Rep.neg_apply, smul_eq_mul, he_self,
          he_ne a d hda]
        linarith [hla k d hd]
      · simp only [Tensor.add_apply, Rep.smul_apply, Rep.neg_apply, smul_eq_mul, he_self]
    rw [h2, h1]
    ring
  have hF : HasFDerivAt
      (fun y => pool .min y dims kernel stride padLow padHigh dilation ceilMode o)
      ((coordinateCLM o).comp (fderiv ℝ
        (fun y => pool .min y dims kernel stride padLow padHigh dilation ceilMode) x)) x :=
    (coordinateCLM o).hasFDerivAt.comp x h.hasFDerivAt
  have h1 := sub_eq_of_hasFDerivAt_of_ray hF (e c' - (2 : ℝ) • e c) (-e c) (-1)
    (hray c c' hcc ⟨k, hk⟩ fun k'' d hd => (hle k'' d hd).1)
  have h2 := sub_eq_of_hasFDerivAt_of_ray hF (e c - (2 : ℝ) • e c') (-e c') (-1)
    (hray c' c (Ne.symm hcc) ⟨k', hk'⟩ fun k'' d hd => (hle k'' d hd).2)
  simp only [map_sub, map_smul, map_neg, smul_eq_mul] at h1 h2
  linarith

/-- A pool is differentiable at `x` exactly when, for a maximum (minimum) pool, every window takes
its largest (smallest) entry of `x` at one input coordinate only. -/
theorem differentiableAt_pool_iff (op : PoolOp) (x : Tensor ℝ s) :
    DifferentiableAt ℝ (fun y => pool op y dims kernel stride padLow padHigh dilation ceilMode)
        x ↔
      (op = .max →
        ∀ (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord)
          (k k' : (s.poolWindow dims kernel).Coord) (c c' : s.Coord),
          poolSource o k = some c → poolSource o k' = some c' →
          (∀ k'' d, poolSource o k'' = some d → x d ≤ x c ∧ x d ≤ x c') → c = c') ∧
      (op = .min →
        ∀ (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord)
          (k k' : (s.poolWindow dims kernel).Coord) (c c' : s.Coord),
          poolSource o k = some c → poolSource o k' = some c' →
          (∀ k'' d, poolSource o k'' = some d → x c ≤ x d ∧ x c' ≤ x d) → c = c') := by
  refine ⟨fun h => ⟨?_, ?_⟩, fun h => (hasFDerivAt_pool op x h.1 h.2).differentiableAt⟩
  · rintro rfl
    exact eq_of_differentiableAt_pool_max h
  · rintro rfl
    exact eq_of_differentiableAt_pool_min h

end Tensor

namespace HasPullbackAt

open Tensor

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] {s : Shape} {x : E}

/-- A pool after `f`, pulled back by `poolBackward` at `f x`. For a maximum (minimum) pool we
assume that every window takes its largest (smallest) entry of `f x` at one input coordinate
only. -/
@[fun_prop]
theorem pool (op : PoolOp) (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool) {f : E → Tensor ℝ s}
    {Df : E →L[ℝ] Tensor ℝ s} {bf : Tensor ℝ s → E}
    (hmax : op = .max →
      ∀ (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord)
        (k k' : (s.poolWindow dims kernel).Coord) (c c' : s.Coord),
        poolSource o k = some c → poolSource o k' = some c' →
        (∀ k'' d, poolSource o k'' = some d → f x d ≤ f x c ∧ f x d ≤ f x c') → c = c')
    (hmin : op = .min →
      ∀ (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord)
        (k k' : (s.poolWindow dims kernel).Coord) (c c' : s.Coord),
        poolSource o k = some c → poolSource o k' = some c' →
        (∀ k'' d, poolSource o k'' = some d → f x c ≤ f x d ∧ f x c' ≤ f x d) → c = c')
    (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt
      (fun y => Tensor.pool op (f y) dims kernel stride padLow padHigh dilation ceilMode)
      ((poolJVPCLM op (f x) dims kernel stride padLow padHigh dilation ceilMode).comp Df)
      (fun δ => bf (Tensor.poolBackward op (f x) dims kernel stride padLow padHigh dilation
        ceilMode δ)) x := by
  have hg : HasPullbackAt
      (fun y => Tensor.pool op y dims kernel stride padLow padHigh dilation ceilMode)
      (poolJVPCLM op (f x) dims kernel stride padLow padHigh dilation ceilMode)
      (fun δ => Tensor.poolBackward op (f x) dims kernel stride padLow padHigh dilation ceilMode
        δ) (f x) :=
    ⟨hasFDerivAt_pool op (f x) hmax hmin, fun u δ => inner_poolJVP op (f x) u δ⟩
  exact hg.comp hf

end HasPullbackAt

namespace Op

open Tensor

/-- The forward of `Op.pool` is differentiable at `x` and its backward is the adjoint of its
derivative, for every kind of pool and every window configuration, when for a maximum (minimum)
pool every window takes its largest (smallest) entry of `x` at one input coordinate only, which
`Tensor.differentiableAt_pool_iff` shows is the exact condition for differentiability. Where a
window takes it at two coordinates the forward is not differentiable, and the backward sends `δ o`
to the coordinate read by the first cell of the window of `o` in row-major order that reads a
largest (smallest) entry (`pool_backward_eq`). -/
@[op_cert TorchLean.Op.pool derivative]
theorem pool_backward_eq_adjoint {s : Shape} (op : PoolOp) (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool) (x : Tensor ℝ s)
    (δ : Tensor ℝ (s.pool dims kernel stride padLow padHigh dilation ceilMode))
    (hmax : op = .max →
      ∀ (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord)
        (k k' : (s.poolWindow dims kernel).Coord) (c c' : s.Coord),
        poolSource o k = some c → poolSource o k' = some c' →
        (∀ k'' d, poolSource o k'' = some d → x d ≤ x c ∧ x d ≤ x c') → c = c')
    (hmin : op = .min →
      ∀ (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord)
        (k k' : (s.poolWindow dims kernel).Coord) (c c' : s.Coord),
        poolSource o k = some c → poolSource o k' = some c' →
        (∀ k'' d, poolSource o k'' = some d → x c ≤ x d ∧ x c' ≤ x d) → c = c') :
    DifferentiableAt ℝ (fun x' =>
        (((Op.pool op dims kernel stride padLow padHigh dilation ceilMode).rule? (α := ℝ)
          |>.get rfl).forward (x', ())).1) x ∧
      (((Op.pool op dims kernel stride padLow padHigh dilation ceilMode).rule? (α := ℝ)
          |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun x' =>
          (((Op.pool op dims kernel stride padLow padHigh dilation ceilMode).rule? (α := ℝ)
            |>.get rfl).forward (x', ())).1) x).adjoint δ := by
  simp only [Op.rule?_pool, Option.get_some]
  apply HasPullbackAt.differentiableAt_and
  · exact HasPullbackAt.pool op dims kernel stride padLow padHigh dilation ceilMode hmax hmin
      (HasPullbackAt.id x)
  · rfl

/-- At every input, entry `c` of the stored backward of `Op.pool` is
`∑ o, poolWeight op x o c * δ o`, and for a maximum (minimum) pool it is the sum of `δ o` over
the outputs `o` whose window has a cell `k` over `c` such that `x c` is a largest (smallest) entry
the window reads and every entry read by a cell before `k` in row-major order is below (above)
`x c`. At a tie each `δ o` thus goes to the source of the first tied cell, which gets `δ o` times
the right (for a minimum, the left) partial derivative of the entry `o` of the forward, and every
other tied coordinate gets `δ o` times its left (right) partial derivative, which is zero. -/
@[op_cert TorchLean.Op.pool convention]
theorem pool_backward_eq {s : Shape} (op : PoolOp) (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool) (x : Tensor ℝ s)
    (δ : Tensor ℝ (s.pool dims kernel stride padLow padHigh dilation ceilMode)) (c : s.Coord) :
    (((Op.pool op dims kernel stride padLow padHigh dilation ceilMode).rule? (α := ℝ)
        |>.get rfl).backward (x, ()) (δ, ())).1 c = ∑ o, poolWeight op x o c * δ o ∧
      (op = .max →
        (((Op.pool op dims kernel stride padLow padHigh dilation ceilMode).rule? (α := ℝ)
          |>.get rfl).backward (x, ()) (δ, ())).1 c =
          ∑ o, if ∃ k, poolSource o k = some c ∧
              (∀ k' d, poolSource o k' = some d → x d ≤ x c) ∧
              ∀ k' d, poolSource o k' = some d →
                Shape.Coord.linearize k' < Shape.Coord.linearize k → x d < x c
            then δ o else 0) ∧
      (op = .min →
        (((Op.pool op dims kernel stride padLow padHigh dilation ceilMode).rule? (α := ℝ)
          |>.get rfl).backward (x, ()) (δ, ())).1 c =
          ∑ o, if ∃ k, poolSource o k = some c ∧
              (∀ k' d, poolSource o k' = some d → x c ≤ x d) ∧
              ∀ k' d, poolSource o k' = some d →
                Shape.Coord.linearize k' < Shape.Coord.linearize k → x c < x d
            then δ o else 0) := by
  simp only [Op.rule?_pool, Option.get_some]
  refine ⟨poolBackward_apply_eq_sum op x δ c, ?_, ?_⟩
  · rintro rfl
    simp only [poolBackward_apply_eq_sum, poolWeight, poolIndices_true_eq_some_iff, ite_mul,
      one_mul, zero_mul]
  · rintro rfl
    simp only [poolBackward_apply_eq_sum, poolWeight, poolIndices_false_eq_some_iff, ite_mul,
      one_mul, zero_mul]

end Op

end TorchLean

/-!
## Smooth maximum pooling

We prove that the stored backward of `Op.smoothMaxPool` is the adjoint of the Fréchet derivative
of its real forward. Over the reals no pivot is infinite, so the shift is the pivot, and for
`beta ≠ 0` an entry of the forward is the log-sum-exp `log (∑ k, exp (beta * v k)) / beta` of the
window entries `v k = poolInput x o k`, whatever the pivot (`smoothMaxPool_apply_real`). Its
derivative weighs each window cell with its softmax weight `smoothMaxPoolSoftmax`. We write each
entry of `smoothMaxPoolJVP` and of `smoothMaxPoolBackward` as a sum over the input coordinates
with the same weights `smoothMaxPoolWeight`, which gives the transpose identity
`inner_smoothMaxPoolJVP` at every input and every `beta`.

At `beta = 0` the logarithm is divided by zero, which gives zero, so an entry of the forward is
the minimum of its window, and the stored backward spreads each cotangent entry evenly over the
window (`smoothMaxPool_backward_eq`). When every window reads a single source (all its cells read
the same input coordinate, or all read padding), the minimum is linear and the even spread is its
transpose. The operator rejects `beta = 0` in `validate`.
-/

namespace TorchLean

namespace Tensor

variable {s : Shape} {dims : List (Fin s.rank)} {kernel stride padLow padHigh dilation : List Nat}
  {ceilMode : Bool}

/-- The window of an output coordinate `o` of a pool has a cell, since an output coordinate exists
only where every pooled axis has a nonzero kernel. -/
theorem nonempty_poolWindow_coord
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) :
    Nonempty (s.poolWindow dims kernel).Coord := by
  have hout : ∀ a : Fin s.rank, Shape.valueAt dims kernel 1 a ≠ 0 :=
    fun a => (Shape.kernel_ne_zero_and_mul_stride_lt_of_lt_poolExtent
      (Shape.Coord.index_lt_ofFn _ o a)).1
  have hpos : ∀ a, 0 < (s.poolWindow dims kernel).extent a := by
    intro a
    have ha : a.val < s.rank := by simpa [Shape.poolWindow, Shape.rank_eq_length] using a.isLt
    simpa [Shape.poolWindow, Shape.extent, Nat.pos_iff_ne_zero] using hout ⟨a.val, ha⟩
  exact ⟨Shape.Coord.ofFun fun a => ⟨0, hpos a⟩⟩

/-- Over the reals the shift of a smooth maximum pool is its pivot, since no real number is
infinite. -/
theorem smoothMaxPoolShift_real (beta : ℝ) (x : Tensor ℝ s)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) :
    smoothMaxPoolShift beta x o = smoothMaxPoolPivot beta x o := by
  simp only [smoothMaxPoolShift_def, Context.isInf_real, Bool.false_eq_true, ↓reduceIte]

/-- Over the reals the sum of shifted exponentials of a smooth maximum pool at `o` is the sum over
the window of `exp (beta * (poolInput x o k - shift))`. -/
theorem smoothMaxPoolSum_real (beta : ℝ) (x : Tensor ℝ s)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) (shift : ℝ) :
    smoothMaxPoolSum beta x o shift = ∑ k, Real.exp (beta * (poolInput x o k - shift)) := by
  rw [smoothMaxPoolSum_def, List.finRange_foldl_add_eq_add_sum, zero_add]
  exact sum_unlinearize _ fun k => Real.exp (beta * (poolInput x o k - shift))

/-- Over the reals the sum of shifted exponentials of a smooth maximum pool at `o` is the sum of
the unshifted exponentials divided by `exp (beta * shift)`. -/
theorem smoothMaxPoolSum_eq_div (beta : ℝ) (x : Tensor ℝ s)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) (shift : ℝ) :
    smoothMaxPoolSum beta x o shift =
      (∑ k, Real.exp (beta * poolInput x o k)) / Real.exp (beta * shift) := by
  rw [smoothMaxPoolSum_real, Finset.sum_div]
  exact Finset.sum_congr rfl fun k _ => by rw [mul_sub, Real.exp_sub]

/-- The sum over the window of `o` of `exp (beta * poolInput x o k)` is positive, since the window
has a cell. -/
theorem sum_exp_poolInput_pos (beta : ℝ) (x : Tensor ℝ s)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) :
    0 < ∑ k, Real.exp (beta * poolInput x o k) :=
  haveI := nonempty_poolWindow_coord o
  Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty

/-- For `beta ≠ 0` an entry of a real smooth maximum pool is the log-sum-exp
`log (∑ k, exp (beta * poolInput x o k)) / beta` over its window, whatever the pivot. -/
theorem smoothMaxPool_apply_real (beta : ℝ) (hbeta : beta ≠ 0) (x : Tensor ℝ s)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) :
    smoothMaxPool beta x dims kernel stride padLow padHigh dilation ceilMode o =
      Real.log (∑ k, Real.exp (beta * poolInput x o k)) / beta := by
  rw [smoothMaxPool_apply]
  simp only [smoothMaxPoolShift_real, smoothMaxPoolSum_eq_div]
  rw [show (FloatLib.Numerics.MathFunctions.log : ℝ → ℝ) = Real.log from rfl,
    Real.log_div (sum_exp_poolInput_pos beta x o).ne' (Real.exp_pos _).ne', Real.log_exp]
  field_simp
  ring

/-- The softmax weight `exp (beta * poolInput x o k) / ∑ j, exp (beta * poolInput x o j)` of the
cell `k` in the window of `o`. -/
noncomputable def smoothMaxPoolSoftmax (beta : ℝ) (x : Tensor ℝ s)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord)
    (k : (s.poolWindow dims kernel).Coord) : ℝ :=
  Real.exp (beta * poolInput x o k) / ∑ j, Real.exp (beta * poolInput x o j)

/-- The weight with which a smooth maximum pool at `x` reads the input coordinate `c` into the
output entry `o`: the sum of the softmax weights of the window cells that read `c`. -/
noncomputable def smoothMaxPoolWeight (beta : ℝ) (x : Tensor ℝ s)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) (c : s.Coord) : ℝ :=
  ∑ k, if poolSource o k = some c then smoothMaxPoolSoftmax beta x o k else 0

/-- The softmax weights of the window of `o` sum to one. -/
theorem sum_smoothMaxPoolSoftmax (beta : ℝ) (x : Tensor ℝ s)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) :
    ∑ k, smoothMaxPoolSoftmax beta x o k = 1 := by
  simp only [smoothMaxPoolSoftmax, ← Finset.sum_div]
  exact div_self (sum_exp_poolInput_pos beta x o).ne'

/-- At `beta = 0` every cell of the window of `o` has the softmax weight one over the number of
window cells. -/
theorem smoothMaxPoolSoftmax_zero (x : Tensor ℝ s)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord)
    (k : (s.poolWindow dims kernel).Coord) :
    smoothMaxPoolSoftmax 0 x o k = 1 / Fintype.card (s.poolWindow dims kernel).Coord := by
  simp only [smoothMaxPoolSoftmax, zero_mul, Real.exp_zero, Finset.sum_const, Finset.card_univ,
    nsmul_eq_mul, mul_one]

/-- Over the reals a shifted exponential of a smooth maximum pool divided by its sum of shifted
exponentials is the softmax weight of the cell, for every shift. -/
theorem exp_div_smoothMaxPoolSum (beta : ℝ) (x : Tensor ℝ s)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord)
    (k : (s.poolWindow dims kernel).Coord) (shift : ℝ) :
    Real.exp (beta * (poolInput x o k - shift)) / smoothMaxPoolSum beta x o shift =
      smoothMaxPoolSoftmax beta x o k := by
  rw [smoothMaxPoolSum_eq_div, mul_sub, Real.exp_sub,
    div_div_div_cancel_right₀ (Real.exp_pos _).ne']
  rfl

/-- Over the reals an entry `o` of `smoothMaxPoolJVP beta x u` is the sum over the window of the
softmax weight of each cell times the entry of `u` under it. -/
theorem smoothMaxPoolJVP_real (beta : ℝ) (x u : Tensor ℝ s)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) :
    smoothMaxPoolJVP beta x u dims kernel stride padLow padHigh dilation ceilMode o =
      ∑ k, smoothMaxPoolSoftmax beta x o k * poolInput u o k := by
  rw [smoothMaxPoolJVP_apply]
  simp only [MathFunctions.exp_real, exp_div_smoothMaxPoolSum, List.finRange_foldl_add_eq_add_sum,
    zero_add]
  exact sum_unlinearize _ fun k => smoothMaxPoolSoftmax beta x o k * poolInput u o k

/-- An entry `o` of `smoothMaxPoolJVP beta x u` is the sum over the input coordinates `c` of the
weight `smoothMaxPoolWeight beta x o c` times `u c`. -/
theorem smoothMaxPoolJVP_apply_eq_sum (beta : ℝ) (x u : Tensor ℝ s)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) :
    smoothMaxPoolJVP beta x u dims kernel stride padLow padHigh dilation ceilMode o =
      ∑ c, smoothMaxPoolWeight beta x o c * u c := by
  simp only [smoothMaxPoolJVP_real, smoothMaxPoolWeight, poolInput_eq_sum, Finset.mul_sum,
    Finset.sum_mul, ite_mul, mul_ite, mul_zero, zero_mul]
  exact Finset.sum_comm

/-- An entry `c` of `smoothMaxPoolBackward beta x δ` is the sum over the outputs `o` of the weight
`smoothMaxPoolWeight beta x o c` times `δ o`. -/
theorem smoothMaxPoolBackward_apply_eq_sum (beta : ℝ) (x : Tensor ℝ s)
    (δ : Tensor ℝ (s.pool dims kernel stride padLow padHigh dilation ceilMode)) (c : s.Coord) :
    smoothMaxPoolBackward beta x dims kernel stride padLow padHigh dilation ceilMode δ c =
      ∑ o, smoothMaxPoolWeight beta x o c * δ o := by
  rw [smoothMaxPoolBackward_apply]
  simp only [MathFunctions.exp_real, exp_div_smoothMaxPoolSum,
    List.finRange_foldl_ite_add_eq_add_sum, List.finRange_foldl_add_eq_add_sum, zero_add,
    smoothMaxPoolWeight, Finset.sum_mul, ite_mul, zero_mul]
  exact (sum_unlinearize_sum_unlinearize _ _ fun o k =>
    if poolSource o k = some c then δ o * smoothMaxPoolSoftmax beta x o k else 0).trans
    (Finset.sum_congr rfl fun o _ => Finset.sum_congr rfl fun k _ => by rw [mul_comm])

/-- The map `y ↦ poolInput y o k`, the entry of `y` under the window cell `k` of `o` or zero in the
padding, as a continuous linear functional. -/
noncomputable def poolInputCLM
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord)
    (k : (s.poolWindow dims kernel).Coord) : Tensor ℝ s →L[ℝ] ℝ :=
  LinearMap.toContinuousLinearMap
    { toFun := fun y => poolInput y o k
      map_add' := fun y z => by
        simp only [poolInput_eq_sum, Tensor.add_apply, ← Finset.sum_add_distrib, ite_add_zero]
      map_smul' := fun a y => by
        simp only [poolInput_eq_sum, Rep.smul_apply, RingHom.id_apply, smul_eq_mul, Finset.mul_sum,
          mul_ite, mul_zero] }

/-- `poolInputCLM o k` applied to `y` is `poolInput y o k`. -/
@[simp] theorem poolInputCLM_apply
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord)
    (k : (s.poolWindow dims kernel).Coord) (y : Tensor ℝ s) :
    poolInputCLM o k y = poolInput y o k :=
  rfl

/-- The map `u ↦ smoothMaxPoolJVP beta x u` of a smooth maximum pool at `x`, as a continuous linear
map. -/
noncomputable def smoothMaxPoolJVPCLM (beta : ℝ) (x : Tensor ℝ s) (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool) :
    Tensor ℝ s →L[ℝ] Tensor ℝ (s.pool dims kernel stride padLow padHigh dilation ceilMode) :=
  LinearMap.toContinuousLinearMap
    { toFun := fun u =>
        smoothMaxPoolJVP beta x u dims kernel stride padLow padHigh dilation ceilMode
      map_add' := fun u v => Rep.ext fun o => by
        simp only [smoothMaxPoolJVP_apply_eq_sum, Tensor.add_apply, mul_add, Finset.sum_add_distrib]
      map_smul' := fun a u => Rep.ext fun o => by
        simp only [smoothMaxPoolJVP_apply_eq_sum, Rep.smul_apply, RingHom.id_apply, smul_eq_mul,
          Finset.mul_sum]
        exact Finset.sum_congr rfl fun _ _ => mul_left_comm _ _ _ }

/-- `smoothMaxPoolJVPCLM beta x dims kernel stride padLow padHigh dilation ceilMode` applied to `u`
is `smoothMaxPoolJVP beta x u dims kernel stride padLow padHigh dilation ceilMode`. -/
@[simp] theorem smoothMaxPoolJVPCLM_apply (beta : ℝ) (x u : Tensor ℝ s)
    (dims : List (Fin s.rank)) (kernel stride padLow padHigh dilation : List Nat)
    (ceilMode : Bool) :
    smoothMaxPoolJVPCLM beta x dims kernel stride padLow padHigh dilation ceilMode u =
      smoothMaxPoolJVP beta x u dims kernel stride padLow padHigh dilation ceilMode :=
  rfl

/-- `smoothMaxPoolBackward beta x` is the transpose of `smoothMaxPoolJVP beta x` for the inner
product of tensors, for every `beta` and every input `x`. -/
theorem inner_smoothMaxPoolJVP (beta : ℝ) (x u : Tensor ℝ s)
    (δ : Tensor ℝ (s.pool dims kernel stride padLow padHigh dilation ceilMode)) :
    ⟪smoothMaxPoolJVP beta x u dims kernel stride padLow padHigh dilation ceilMode, δ⟫ =
      ⟪u, smoothMaxPoolBackward beta x dims kernel stride padLow padHigh dilation ceilMode δ⟫ := by
  simp only [inner_eq_sum, smoothMaxPoolJVP_apply_eq_sum, smoothMaxPoolBackward_apply_eq_sum,
    Finset.sum_mul, Finset.mul_sum]
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => by ring

/-- For `beta ≠ 0` the entry `o` of a smooth maximum pool has the entry `o` of
`smoothMaxPoolJVPCLM beta x` as derivative at `x`. -/
theorem hasFDerivAt_smoothMaxPool_apply (beta : ℝ) (hbeta : beta ≠ 0) (x : Tensor ℝ s)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) :
    HasFDerivAt
      (fun y => smoothMaxPool beta y dims kernel stride padLow padHigh dilation ceilMode o)
      ((coordinateCLM o).comp
        (smoothMaxPoolJVPCLM beta x dims kernel stride padLow padHigh dilation ceilMode)) x := by
  have hF : (fun y => smoothMaxPool beta y dims kernel stride padLow padHigh dilation ceilMode o) =
      fun y => Real.log (∑ k, Real.exp (beta * poolInputCLM o k y)) * beta⁻¹ :=
    funext fun y => by rw [smoothMaxPool_apply_real beta hbeta, div_eq_mul_inv]; rfl
  rw [hF]
  have hS := HasFDerivAt.fun_sum (u := Finset.univ) fun k _ =>
    (((poolInputCLM (dims := dims) (stride := stride) (padLow := padLow) (padHigh := padHigh)
      (dilation := dilation) (ceilMode := ceilMode) o k).hasFDerivAt (x := x)).const_mul beta).exp
  refine ((hS.log (sum_exp_poolInput_pos beta x o).ne').mul_const beta⁻¹).congr_fderiv ?_
  refine ContinuousLinearMap.ext fun u => ?_
  have hW := (sum_exp_poolInput_pos beta x o).ne'
  simp only [_root_.smul_apply, _root_.sum_apply, ContinuousLinearMap.comp_apply,
    coordinateCLM_apply, smoothMaxPoolJVPCLM_apply, smoothMaxPoolJVP_real, poolInputCLM_apply,
    smul_eq_mul, smoothMaxPoolSoftmax, Finset.mul_sum]
  refine Finset.sum_congr rfl fun k _ => ?_
  field_simp

/-- At `beta = 0`, when every cell of the window of `o` reads the same source, the entry `o` of a
real smooth maximum pool is the entry of `y` under any cell `k` of that window. -/
theorem smoothMaxPool_zero_apply_of_poolSource_eq (y : Tensor ℝ s)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord)
    (hsrc : ∃ σ, ∀ k, poolSource o k = σ) (k : (s.poolWindow dims kernel).Coord) :
    smoothMaxPool 0 y dims kernel stride padLow padHigh dilation ceilMode o = poolInput y o k := by
  obtain ⟨σ, hσ⟩ := hsrc
  have hv : ∀ k', poolInput y o k' = poolInput y o k := fun k' => by
    rw [poolInput_def, poolInput_def, hσ k', hσ k]
  have hpos := (Shape.Coord.linearize k).pos
  simp only [smoothMaxPool_apply, div_zero, add_zero, smoothMaxPoolShift_real,
    smoothMaxPoolPivot_def]
  rw [dite_eq_left hpos, hv (Shape.Coord.unlinearize ⟨0, hpos⟩)]
  exact List.foldl_fixed' (fun i => by
    rw [ite_eq_right (lt_irrefl _), hv (Shape.Coord.unlinearize i), min_self]) _

/-- When every cell of the window of `o` reads the same source, the entry `o` of
`smoothMaxPoolJVP beta x u` is the entry of `u` under any cell `k` of that window, for every
`beta`. -/
theorem smoothMaxPoolJVP_apply_of_poolSource_eq (beta : ℝ) (x u : Tensor ℝ s)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord)
    (hsrc : ∃ σ, ∀ k, poolSource o k = σ) (k : (s.poolWindow dims kernel).Coord) :
    smoothMaxPoolJVP beta x u dims kernel stride padLow padHigh dilation ceilMode o =
      poolInput u o k := by
  obtain ⟨σ, hσ⟩ := hsrc
  rw [smoothMaxPoolJVP_real]
  calc ∑ k', smoothMaxPoolSoftmax beta x o k' * poolInput u o k'
      = ∑ k', smoothMaxPoolSoftmax beta x o k' * poolInput u o k :=
        Finset.sum_congr rfl fun k' _ => by rw [poolInput_def, poolInput_def, hσ k', hσ k]
    _ = poolInput u o k := by rw [← Finset.sum_mul, sum_smoothMaxPoolSoftmax, one_mul]

/-- `smoothMaxPool beta` has derivative `smoothMaxPoolJVPCLM beta x` at `x` when `beta ≠ 0` or
every window reads a single source. At `beta = 0` the forward is the minimum of each window, which
is linear when the window reads a single source. -/
theorem hasFDerivAt_smoothMaxPool (beta : ℝ) (x : Tensor ℝ s)
    (h : beta ≠ 0 ∨ ∀ o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord,
      ∃ σ, ∀ k, poolSource o k = σ) :
    HasFDerivAt (fun y => smoothMaxPool beta y dims kernel stride padLow padHigh dilation ceilMode)
      (smoothMaxPoolJVPCLM beta x dims kernel stride padLow padHigh dilation ceilMode) x := by
  rcases eq_or_ne beta 0 with rfl | hbeta
  · have hsrc := h.resolve_left fun h0 => h0 rfl
    refine (smoothMaxPoolJVPCLM 0 x dims kernel stride padLow padHigh dilation
      ceilMode).hasFDerivAt.congr_of_eventuallyEq (Eventually.of_forall fun y => ?_)
    refine Rep.ext fun o => ?_
    obtain ⟨k⟩ := nonempty_poolWindow_coord o
    simp only [smoothMaxPoolJVPCLM_apply, smoothMaxPool_zero_apply_of_poolSource_eq y o (hsrc o) k,
      smoothMaxPoolJVP_apply_of_poolSource_eq 0 x y o (hsrc o) k]
  · refine (hasFDerivAt_of_coordinates _ _ x
      (hasFDerivAt_smoothMaxPool_apply (dims := dims) (kernel := kernel) (stride := stride)
        (padLow := padLow) (padHigh := padHigh) (dilation := dilation) (ceilMode := ceilMode)
        beta hbeta x)).congr_fderiv (ContinuousLinearMap.ext fun u => Rep.ext fun o => ?_)
    rw [assemble_apply]
    rfl

/-- A smooth maximum pool has `smoothMaxPoolJVPCLM beta x` as derivative and is pulled back by
`smoothMaxPoolBackward beta x` at `x`, when `beta ≠ 0` or every window reads a single source. -/
theorem hasPullbackAt_smoothMaxPool (beta : ℝ) (x : Tensor ℝ s)
    (h : beta ≠ 0 ∨ ∀ o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord,
      ∃ σ, ∀ k, poolSource o k = σ) :
    HasPullbackAt
      (fun y => smoothMaxPool beta y dims kernel stride padLow padHigh dilation ceilMode)
      (smoothMaxPoolJVPCLM beta x dims kernel stride padLow padHigh dilation ceilMode)
      (fun δ => smoothMaxPoolBackward beta x dims kernel stride padLow padHigh dilation ceilMode δ)
      x :=
  ⟨hasFDerivAt_smoothMaxPool beta x h, fun u δ => inner_smoothMaxPoolJVP beta x u δ⟩

end Tensor

namespace HasPullbackAt

open Tensor

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] {s : Shape} {x : E}

/-- A smooth maximum pool with `beta ≠ 0` after `f`, pulled back by `smoothMaxPoolBackward` at
`f x`. -/
@[fun_prop]
theorem smoothMaxPool (beta : ℝ) (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool) {f : E → Tensor ℝ s}
    {Df : E →L[ℝ] Tensor ℝ s} {bf : Tensor ℝ s → E} (hbeta : beta ≠ 0)
    (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt
      (fun y => Tensor.smoothMaxPool beta (f y) dims kernel stride padLow padHigh dilation ceilMode)
      ((smoothMaxPoolJVPCLM beta (f x) dims kernel stride padLow padHigh dilation ceilMode).comp Df)
      (fun δ => bf (Tensor.smoothMaxPoolBackward beta (f x) dims kernel stride padLow padHigh
        dilation ceilMode δ)) x :=
  (hasPullbackAt_smoothMaxPool beta (f x) (.inl hbeta)).comp hf

/-- A smooth maximum pool after `f` whose windows each read a single source, pulled back by
`smoothMaxPoolBackward` at `f x`, for every `beta`. -/
@[fun_prop]
theorem smoothMaxPool_of_poolSource_eq (beta : ℝ) (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool) {f : E → Tensor ℝ s}
    {Df : E →L[ℝ] Tensor ℝ s} {bf : Tensor ℝ s → E}
    (hsrc : ∀ o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord,
      ∃ σ, ∀ k, poolSource o k = σ)
    (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt
      (fun y => Tensor.smoothMaxPool beta (f y) dims kernel stride padLow padHigh dilation ceilMode)
      ((smoothMaxPoolJVPCLM beta (f x) dims kernel stride padLow padHigh dilation ceilMode).comp Df)
      (fun δ => bf (Tensor.smoothMaxPoolBackward beta (f x) dims kernel stride padLow padHigh
        dilation ceilMode δ)) x :=
  (hasPullbackAt_smoothMaxPool beta (f x) (.inr hsrc)).comp hf

end HasPullbackAt

namespace Op

open Tensor

/-- The rule of `Op.smoothMaxPool` over the reals, with the capabilities of `ℝ`. The arm equation
`Op.rule?_smoothMaxPool` takes every capability other than `DecidableEq` as absent, so it does not
rewrite the real rule. -/
@[op_deriv_simps]
theorem rule?_smoothMaxPool_real {s : Shape} (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool) :
    (Op.smoothMaxPool dims kernel stride padLow padHigh dilation ceilMode).rule? (α := ℝ) = some
      { forward := fun xs =>
          (Tensor.smoothMaxPool xs.2.1.item xs.1 dims kernel stride padLow padHigh dilation
            ceilMode, ())
        backward := fun xs g =>
          (Tensor.smoothMaxPoolBackward xs.2.1.item xs.1 dims kernel stride padLow padHigh dilation
            ceilMode g.1, (), ())
        validate := fun xs =>
          if xs.2.1.item == 0 || !(xs.2.1.item - xs.2.1.item == 0) then
            .error "autograd: smooth_max_pool requires finite nonzero beta"
          else .ok ()
        jvp? := some fun xs dxs =>
          (Tensor.smoothMaxPoolJVP xs.2.1.item xs.1 dxs.1 dims kernel stride padLow padHigh
            dilation ceilMode, ()) } :=
  rfl

/-- The forward of `Op.smoothMaxPool` with a fixed `beta` is differentiable at `x` and its backward
is the adjoint of its derivative, for `beta.item ≠ 0` and every window configuration, which holds
at every input that `validate` accepts. The exact set also contains the configurations with
`beta = 0` where every window reads a single source, which `validate` rejects and
`smoothMaxPool_backward_eq_adjoint_of_poolSource_eq` covers. -/
@[op_cert TorchLean.Op.smoothMaxPool derivative 0]
theorem smoothMaxPool_backward_eq_adjoint {s : Shape} (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool) (x : Tensor ℝ s)
    (beta : Tensor ℝ []) (h : beta.item ≠ 0)
    (δ : Tensor ℝ (s.pool dims kernel stride padLow padHigh dilation ceilMode)) :
    DifferentiableAt ℝ (fun x' =>
        (((Op.smoothMaxPool dims kernel stride padLow padHigh dilation ceilMode).rule? (α := ℝ)
          |>.get rfl).forward (x', beta, ())).1) x ∧
      (((Op.smoothMaxPool dims kernel stride padLow padHigh dilation ceilMode).rule? (α := ℝ)
          |>.get rfl).backward (x, beta, ()) (δ, ())).1 =
        (fderiv ℝ (fun x' =>
          (((Op.smoothMaxPool dims kernel stride padLow padHigh dilation ceilMode).rule? (α := ℝ)
            |>.get rfl).forward (x', beta, ())).1) x).adjoint δ := by
  op_deriv

/-- The forward of `Op.smoothMaxPool` with a fixed `beta` is differentiable at `x` and its backward
is the adjoint of its derivative, for every `beta`, when every window reads a single source (all
its cells read the same input coordinate, or all read padding). With `beta = 0` the forward is
then the entry of `x` under each window, or zero. -/
@[op_cert TorchLean.Op.smoothMaxPool derivative 0]
theorem smoothMaxPool_backward_eq_adjoint_of_poolSource_eq {s : Shape} (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool) (x : Tensor ℝ s)
    (beta : Tensor ℝ [])
    (hsrc : ∀ o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord,
      ∃ σ, ∀ k, poolSource o k = σ)
    (δ : Tensor ℝ (s.pool dims kernel stride padLow padHigh dilation ceilMode)) :
    DifferentiableAt ℝ (fun x' =>
        (((Op.smoothMaxPool dims kernel stride padLow padHigh dilation ceilMode).rule? (α := ℝ)
          |>.get rfl).forward (x', beta, ())).1) x ∧
      (((Op.smoothMaxPool dims kernel stride padLow padHigh dilation ceilMode).rule? (α := ℝ)
          |>.get rfl).backward (x, beta, ()) (δ, ())).1 =
        (fderiv ℝ (fun x' =>
          (((Op.smoothMaxPool dims kernel stride padLow padHigh dilation ceilMode).rule? (α := ℝ)
            |>.get rfl).forward (x', beta, ())).1) x).adjoint δ := by
  op_deriv

/-- At every input and every `beta`, entry `c` of the stored backward of `Op.smoothMaxPool` is
`∑ o, smoothMaxPoolWeight beta.item x o c * δ o`, where the weight adds the softmax weights at
`beta` of the cells of the window of `o` that read `c`. At `beta = 0`, which `validate` rejects,
every window cell has the weight one over the number of window cells, so each `δ o` is spread
evenly over the cells of its window. -/
@[op_cert TorchLean.Op.smoothMaxPool convention]
theorem smoothMaxPool_backward_eq {s : Shape} (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool) (x : Tensor ℝ s)
    (beta : Tensor ℝ [])
    (δ : Tensor ℝ (s.pool dims kernel stride padLow padHigh dilation ceilMode)) (c : s.Coord) :
    (((Op.smoothMaxPool dims kernel stride padLow padHigh dilation ceilMode).rule? (α := ℝ)
        |>.get rfl).backward (x, beta, ()) (δ, ())).1 c =
        ∑ o, smoothMaxPoolWeight beta.item x o c * δ o ∧
      (beta.item = 0 →
        (((Op.smoothMaxPool dims kernel stride padLow padHigh dilation ceilMode).rule? (α := ℝ)
          |>.get rfl).backward (x, beta, ()) (δ, ())).1 c =
          ∑ o, (∑ k, if poolSource o k = some c then (1 : ℝ) else 0) /
            Fintype.card (s.poolWindow dims kernel).Coord * δ o) := by
  simp only [rule?_smoothMaxPool_real, Option.get_some]
  refine ⟨smoothMaxPoolBackward_apply_eq_sum beta.item x δ c,
    fun h0 => (smoothMaxPoolBackward_apply_eq_sum beta.item x δ c).trans ?_⟩
  rw [h0]
  simp only [smoothMaxPoolWeight, smoothMaxPoolSoftmax_zero, Finset.sum_div, ite_div, zero_div]

end Op

end TorchLean

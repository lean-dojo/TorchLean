/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Context.Real
public import TorchLean.Tensor.Layers.Attention
public import TorchLean.NN.Softmax
public import TorchLean.NN.Attention.HardMask.FDeriv

/-!
# The online softmax schedule over the reals

`Softmax.onlineAverage` is FlashAttention's forward schedule for one query, the
`Softmax.Schedule.online` schedule of `Tensor.attention`: a pass over blocks of keys keeping a
running maximum `m`, a normalizer `l` and a weighted sum `o`, each block multiplying `l` and `o` by
`exp (m - m')` when it raises the maximum to `m'`.

Over the reals the online schedule computes the two-pass value: `attentionOnline_apply_real` reads
an entry of the online schedule on score matrices with any leading batch axes, a mask and a
multiplier, as the softmax average of its allowed keys, and `attention_online_eq` and
`softmax_online_eq` state that attention and softmax by the online schedule equal them by the
two-pass schedule, for every block size, mask and multiplier.

The module also holds the real algebra behind the rounding bounds of
`TorchLean.Verify.RuntimeApprox.OnlineSoftmax`. `TermwiseError` is one aggregate error inequality
whose allowance is a sum of per-term relative allowances. A rounding adds `u` to every term's
exponent (`TermwiseError.round`), a scaling by an approximate factor adds the factor's exponent
(`TermwiseError.scale`), and the shift error of the rescaling factor telescopes into the exponent of
every term it scales. `step_residual_le` grows the underflow residual by `residualGrowth` per block
and adds `blockResidual`. Gibbs' inequality `∑ w (M - s) ≤ S (log N - log S)`
(`sum_exp_mul_sub_le`) turns the termwise budgets into closed forms (`closed_form_le`,
`budget_closed_form`). `closed_form_le_of_bounds` and `stored_closed_form_le_of_bounds` evaluate
these from upper bounds on their exponentials and powers.
-/

@[expose] public section

namespace TorchLean.Softmax

/-- Over the reals, folding the blocks keeps the state at the running maximum `M`: the normalizer
and weighted sum are the starting state's, rescaled by `exp (m - M)` from its maximum `m`, plus the
terms of the blocks' keys. -/
theorem foldl_onlineStep_real :
    ∀ (blocks : List (List (ℝ × ℝ))) (state : ℝ × ℝ × ℝ),
      blocks.foldl (onlineStep Real.exp id id) state =
        (blocks.flatten.foldl (fun m e => max m e.1) state.1,
          Real.exp (state.1 - blocks.flatten.foldl (fun m e => max m e.1) state.1) * state.2.1 +
            (blocks.flatten.map fun e =>
              Real.exp (e.1 - blocks.flatten.foldl (fun m e => max m e.1) state.1)).sum,
          Real.exp (state.1 - blocks.flatten.foldl (fun m e => max m e.1) state.1) * state.2.2 +
            (blocks.flatten.map fun e =>
              Real.exp (e.1 - blocks.flatten.foldl (fun m e => max m e.1) state.1) * e.2).sum)
  | [], state => by simp
  | block :: blocks, state => by
      rw [List.foldl_cons, foldl_onlineStep_real blocks]
      obtain ⟨m, l, o⟩ := state
      simp only [onlineStep, id, List.flatten_cons, List.foldl_append, List.map_append,
        List.sum_append, foldl_add_eq_add_sum, zero_add, Context.maximum_real]
      set m₁ := block.foldl (fun m e => max m e.1) m
      set M := blocks.flatten.foldl (fun m e => max m e.1) m₁
      have hscale (x : ℝ) : Real.exp (m₁ - M) * Real.exp (x - m₁) = Real.exp (x - M) := by
        rw [← Real.exp_add]; ring_nf
      refine Prod.ext rfl (Prod.ext ?_ ?_)
      · simp only [mul_add, ← mul_assoc, hscale, ← List.sum_map_mul_left]
        ring
      · simp only [mul_add, ← mul_assoc, hscale, ← List.sum_map_mul_left]
        ring

/-- Over the reals, the online average is the softmax average `∑ exp s * v / ∑ exp s` of its keys
for every partition into blocks and every starting maximum. -/
theorem onlineAverage_eq_div (start : ℝ) (blocks : List (List (ℝ × ℝ))) :
    onlineAverage Real.exp id id start blocks =
      (blocks.flatten.map fun e => Real.exp e.1 * e.2).sum /
        (blocks.flatten.map fun e => Real.exp e.1).sum := by
  simp only [onlineAverage, foldl_onlineStep_real, mul_zero, zero_add]
  set M := blocks.flatten.foldl (fun m e => max m e.1) start
  have hshift (x : ℝ) : Real.exp (x - M) = Real.exp (-M) * Real.exp x := by
    rw [← Real.exp_add]; ring_nf
  simp only [hshift, mul_assoc, List.sum_map_mul_left]
  exact mul_div_mul_left _ _ (Real.exp_pos _).ne'

end TorchLean.Softmax

namespace TorchLean.OnlineSoftmax

open Softmax (onlineStep onlineAverage)

/-- `x` is within `∑ e ∈ P, w e * |g e| * (exp (a e) - 1) + ρ` of `∑ e ∈ P, w e * g e`: one
error inequality whose allowance is a relative allowance `exp (a e) - 1` for each term plus an
absolute residual `ρ`. -/
def TermwiseError {ι : Type*} (x : ℝ) (P : List ι) (w g a : ι → ℝ) (ρ : ℝ) : Prop :=
  |x - (P.map fun e => w e * g e).sum| ≤
    (P.map fun e => w e * |g e| * (Real.exp (a e) - 1)).sum + ρ

/-- Over the reals, the online attention output of a query is the softmax average of its allowed
keys, for every block size, and zero when no key is allowed. -/
theorem onlineAttend_real {n : Nat} (blocks : Softmax.BlockPartition) (score value : Fin n → ℝ)
    (allowed : Fin n → Bool) :
    Softmax.onlineAttend blocks score value allowed =
      (∑ j, if allowed j then Real.exp (score j) * value j else 0) /
        ∑ j, if allowed j then Real.exp (score j) else 0 := by
  simp only [Fin.sum_univ_def, List.sum_map_ite, Bool.decide_eq_true, List.sum_map_zero, add_zero]
  unfold Softmax.onlineAttend Softmax.onlineKeys
  split
  next h =>
    rw [List.map_eq_nil_iff] at h
    simp [h]
  next first rest h =>
    rw [Softmax.onlineStart_of_isInf_eq_false first rest (Context.isInf_real _)]
    change onlineAverage Real.exp id id first.1
      (List.toChunks blocks.size (first :: rest)) = _
    rw [Softmax.onlineAverage_eq_div, List.flatten_toChunks, ← h, List.map_map,
      List.map_map]
    rfl

/-- Over the reals, the online softmax weight of a key is its exponential over the sum of the
exponentials of the allowed keys, for every block size, and zero at a blocked key. -/
theorem onlineWeight_real {n : Nat} (blocks : Softmax.BlockPartition) (score : Fin n → ℝ)
    (allowed : Fin n → Bool) (j : Fin n) :
    Softmax.onlineWeight blocks score allowed j =
      (if allowed j then Real.exp (score j) else 0) /
        ∑ i, if allowed i then Real.exp (score i) else 0 := by
  unfold Softmax.onlineWeight Softmax.onlineKeys
  split
  next h =>
    rw [List.map_eq_nil_iff] at h
    have hj : allowed j = false := by
      by_contra hj
      have hmem : j ∈ (List.finRange n).filter allowed :=
        List.mem_filter.mpr ⟨List.mem_finRange j, by simpa using hj⟩
      simp [h] at hmem
    simp [hj]
  next first rest h =>
    rw [Softmax.onlineStart_of_isInf_eq_false first rest (Context.isInf_real _)]
    change (if allowed j then Real.exp (score j -
        ((List.toChunks blocks.size (first :: rest)).foldl
          (onlineStep Real.exp id id) (first.1, 0, 0)).1) /
        ((List.toChunks blocks.size (first :: rest)).foldl
          (onlineStep Real.exp id id) (first.1, 0, 0)).2.1 else 0) = _
    rw [Softmax.foldl_onlineStep_real, List.flatten_toChunks]
    simp only [mul_zero, zero_add]
    generalize (first :: rest).foldl (fun m e => max m e.1) first.1 = m
    rw [← h, List.map_map]
    have hshift (x : ℝ) : Real.exp (x - m) = Real.exp x * Real.exp (-m) := by
      rw [← Real.exp_add]; ring_nf
    have hsum : (((List.finRange n).filter allowed).map
        ((fun e : ℝ × ℝ => Real.exp (e.1 - m)) ∘ fun i => (score i, (0 : ℝ)))).sum =
        (∑ i, if allowed i then Real.exp (score i) else 0) * Real.exp (-m) := by
      simp only [Fin.sum_univ_def, List.sum_map_ite, Bool.decide_eq_true, List.sum_map_zero,
        add_zero, Function.comp_def, hshift, List.sum_map_mul_right]
    rw [hsum]
    split_ifs
    · rw [hshift]
      exact mul_div_mul_right _ _ (Real.exp_ne_zero _)
    · simp

/-- Over the reals and for every block size, an entry of the online schedule on score matrices with
any leading batch axes is the softmax average of the allowed entries of its score row against its
value column. The value column is multiplied entrywise by the multiplier row when one is given, and
the entry is zero when no key is allowed. -/
theorem attentionOnline_apply_real (blocks : Softmax.BlockPartition) {batch : Shape}
    {rows keys width : Nat} (scores : Tensor ℝ (batch ++ [rows, keys]))
    (value : Tensor ℝ (batch ++ [keys, width]))
    (multiplier : Option (Tensor ℝ (batch ++ [rows, keys])))
    (mask : Option (Tensor Bool [rows, keys])) (b : batch.Coord) (i : Fin rows) (d : Fin width) :
    (Attention.online blocks scores value multiplier mask).batchSlice b (i, d, ()) =
      (∑ j, if mask.elim true (fun mask => mask (i, j, ())) then
          Real.exp (scores.batchSlice b (i, j, ())) *
            multiplier.elim (value.batchSlice b (j, d, ())) (fun multiplier =>
              multiplier.batchSlice b (i, j, ()) * value.batchSlice b (j, d, ()))
        else 0) /
        ∑ j, if mask.elim true (fun mask => mask (i, j, ())) then
          Real.exp (scores.batchSlice b (i, j, ())) else 0 := by
  rw [Tensor.batchSlice_apply, Attention.online_apply]
  simp only [Equiv.apply_symm_apply]
  exact onlineAttend_real blocks _ _ _

/-- Over the reals, an entry of the two-pass attention on score matrices with any leading batch
axes is the softmax average of the allowed entries of its score row against its value column. The
value column is multiplied entrywise by the multiplier row when one is given, and the entry is zero
when no key is allowed. -/
theorem attentionValues_apply_real {batch : Shape} {rows keys width : Nat}
    (scores : Tensor ℝ (batch ++ [rows, keys]))
    (value : Tensor ℝ (batch ++ [keys, width]))
    (multiplier : Option (Tensor ℝ (batch ++ [rows, keys])))
    (mask : Option (Tensor Bool [rows, keys])) (b : batch.Coord) (i : Fin rows) (d : Fin width) :
    (Attention.values (Attention.weights scores mask) value
        multiplier).batchSlice b (i, d, ()) =
      (∑ j, if mask.elim true (fun mask => mask (i, j, ())) then
          Real.exp (scores.batchSlice b (i, j, ())) *
            multiplier.elim (value.batchSlice b (j, d, ())) (fun multiplier =>
              multiplier.batchSlice b (i, j, ()) * value.batchSlice b (j, d, ()))
        else 0) /
        ∑ j, if mask.elim true (fun mask => mask (i, j, ())) then
          Real.exp (scores.batchSlice b (i, j, ())) else 0 := by
  rw [Tensor.batchSlice_apply, Attention.values_apply]
  simp only [Equiv.apply_symm_apply]
  rw [Attention.weights_batchSlice, Attention.weightsSlice_eq_hardMasked]
  set fixed := mask.getD (Attention.allTrueMask rows keys)
  have hread {β : Type} [Storage β] (t : Tensor β [rows, keys]) (j : Fin keys) :
      (t.unstack i).getScalar j = t (i, j, ()) :=
    Tensor.get2_eq_apply t i j
  have hallowed (j : Fin keys) :
      (fixed.unstack i).getScalar j = mask.elim true (fun mask => mask (i, j, ())) := by
    cases mask with
    | none => simp [fixed, Attention.allTrueMask]
    | some mask => exact hread mask j
  have hweight (j : Fin keys) :
      Tensor.get2 (Softmax.hardMasked (scores.batchSlice b) fixed) i j =
        (if mask.elim true (fun mask => mask (i, j, ())) then
          Real.exp (scores.batchSlice b (i, j, ())) else 0) /
          ∑ k, if mask.elim true (fun mask => mask (i, k, ())) then
            Real.exp (scores.batchSlice b (i, k, ())) else 0 := by
    rw [Tensor.get2_eq_getScalar_unstack,
      Softmax.unstack_hardMasked,
      Softmax.getScalar_hardMaskedVec]
    simp only [HardMaskedSoftmax.numerator,
      HardMaskedSoftmax.denominator, Tensor.getScalarE_ofLp,
      hallowed, hread]
  rw [← Tensor.get2_eq_apply, Tensor.get2_matmul_eq_sum, Finset.sum_div]
  refine Finset.sum_congr rfl fun j _ => ?_
  cases multiplier with
  | none =>
    change Tensor.get2 (Softmax.hardMasked (scores.batchSlice b) fixed) i j *
      Tensor.get2 (value.batchSlice b) j d = _
    rw [hweight, Tensor.get2_eq_apply]
    split_ifs <;> simp [div_mul_eq_mul_div]
  | some multiplier =>
    change Tensor.get2 (Softmax.hardMasked (scores.batchSlice b) fixed *
      multiplier.batchSlice b) i j * Tensor.get2 (value.batchSlice b) j d = _
    rw [Tensor.get2_eq_apply (_ * _), Tensor.mul_def, Tensor.zipWith_apply,
      ← Tensor.get2_eq_apply, hweight]
    split_ifs <;> simp [div_mul_eq_mul_div, mul_assoc, Tensor.get2_eq_apply]

/-- Over the reals, the online schedule on score matrices computes the two-pass attention, for
every block size, mask and multiplier. -/
theorem attentionOnline_eq_real (blocks : Softmax.BlockPartition) {batch : Shape}
    {rows keys width : Nat} (scores : Tensor ℝ (batch ++ [rows, keys]))
    (value : Tensor ℝ (batch ++ [keys, width]))
    (multiplier : Option (Tensor ℝ (batch ++ [rows, keys])))
    (mask : Option (Tensor Bool [rows, keys])) :
    Attention.online blocks scores value multiplier mask =
      Attention.values (Attention.weights scores mask) value
        multiplier := by
  refine Tensor.Rep.ext fun c => ?_
  obtain ⟨⟨b, i, d, ⟨⟩⟩, rfl⟩ : ∃ q, ((Shape.Coord.appendEquiv batch [rows,
      width]).symm q) = c :=
    ⟨Shape.Coord.appendEquiv batch [rows, width]
      c, by simp⟩
  rw [← Tensor.batchSlice_apply, ← Tensor.batchSlice_apply, attentionOnline_apply_real,
    attentionValues_apply_real]

variable {ι : Type*}

/-- For nonnegative weights, `|∑ e ∈ P, w e * g e| ≤ ∑ e ∈ P, w e * |g e|`. -/
theorem abs_sum_le {P : List ι} (w g : ι → ℝ) (hw : ∀ e ∈ P, 0 ≤ w e) :
    |(P.map fun e => w e * g e).sum| ≤ (P.map fun e => w e * |g e|).sum := by
  refine (List.le_sum_of_subadditive abs abs_zero.le abs_add_le _).trans (le_of_eq ?_)
  rw [List.map_map]
  congr 1
  exact List.map_congr_left fun e he => by
    simp only [Function.comp_apply, abs_mul, abs_of_nonneg (hw e he)]

/-- For nonnegative weights, rounding `z` to `x` with `|x - z| ≤ u * |z| + η` keeps a termwise
error: every exponent grows by `u` and the residual `ρ` becomes `(1 + u) * ρ + η`. -/
theorem TermwiseError.round {x z u η ρ : ℝ} {P : List ι} {w g a : ι → ℝ}
    (h : TermwiseError z P w g a ρ) (hround : |x - z| ≤ u * |z| + η) (hu : 0 ≤ u)
    (hw : ∀ e ∈ P, 0 ≤ w e) :
    TermwiseError x P w g (fun e => a e + u) ((1 + u) * ρ + η) := by
  unfold TermwiseError at *
  set Z := (P.map fun e => w e * g e).sum
  have hZ : |Z| ≤ (P.map fun e => w e * |g e|).sum := abs_sum_le w g hw
  have hz : |z| ≤ |Z| + |z - Z| := by
    have := abs_add_le (z - Z) Z; simp only [sub_add_cancel] at this; linarith
  have key : (1 + u) * (P.map fun e => w e * |g e| * (Real.exp (a e) - 1)).sum +
      u * (P.map fun e => w e * |g e|).sum ≤
      (P.map fun e => w e * |g e| * (Real.exp (a e + u) - 1)).sum := by
    rw [← List.sum_map_mul_left, ← List.sum_map_mul_left, ← List.sum_map_add]
    refine List.sum_le_sum fun e he => ?_
    have hwg : 0 ≤ w e * |g e| := mul_nonneg (hw e he) (abs_nonneg _)
    have hexp : 1 + u ≤ Real.exp u := by linarith [Real.add_one_le_exp u]
    rw [Real.exp_add]
    have hea : 0 ≤ Real.exp (a e) := (Real.exp_pos _).le
    nlinarith [mul_le_mul_of_nonneg_left hexp hea]
  calc |x - Z| ≤ |x - z| + |z - Z| := abs_sub_le _ _ _
    _ ≤ u * (|Z| + |z - Z|) + η + |z - Z| := by nlinarith
    _ = (1 + u) * |z - Z| + u * |Z| + η := by ring
    _ ≤ (1 + u) * ((P.map fun e => w e * |g e| * (Real.exp (a e) - 1)).sum + ρ) +
        u * (P.map fun e => w e * |g e|).sum + η := by
      gcongr
    _ ≤ _ := by nlinarith [key]

/-- Termwise errors of `x` over `P` and of `y` over `Q` give one of `x + y` over `P ++ Q`,
with residual `ρ + σ`. -/
theorem TermwiseError.add {x y ρ σ : ℝ} {P Q : List ι} {w g a : ι → ℝ}
    (hx : TermwiseError x P w g a ρ) (hy : TermwiseError y Q w g a σ) :
    TermwiseError (x + y) (P ++ Q) w g a (ρ + σ) := by
  unfold TermwiseError at *
  simp only [List.map_append, List.sum_append]
  calc |x + y - ((P.map fun e => w e * g e).sum + (Q.map fun e => w e * g e).sum)| =
      |(x - (P.map fun e => w e * g e).sum) + (y - (Q.map fun e => w e * g e).sum)| := by
        ring_nf
    _ ≤ _ := (abs_add_le _ _).trans (by linarith)

/-- For nonnegative weights, a termwise error holds with larger exponents and a larger residual. -/
theorem TermwiseError.mono {x ρ ρ' : ℝ} {P : List ι} {w g a a' : ι → ℝ}
    (h : TermwiseError x P w g a ρ) (ha : ∀ e ∈ P, a e ≤ a' e) (hρ : ρ ≤ ρ')
    (hw : ∀ e ∈ P, 0 ≤ w e) : TermwiseError x P w g a' ρ' := by
  unfold TermwiseError at *
  refine h.trans (add_le_add (List.sum_le_sum fun e he => ?_) hρ)
  exact mul_le_mul_of_nonneg_left (by linarith [Real.exp_le_exp.mpr (ha e he)])
    (mul_nonneg (hw e he) (abs_nonneg _))

/-- A termwise error holds with weights and exponents that agree with `w` and `a` on `P`. -/
theorem TermwiseError.congr {x ρ : ℝ} {P : List ι} {w g a w' a' : ι → ℝ}
    (h : TermwiseError x P w g a ρ) (hw : ∀ e ∈ P, w e = w' e) (ha : ∀ e ∈ P, a e = a' e) :
    TermwiseError x P w' g a' ρ := by
  unfold TermwiseError at *
  have h1 : (P.map fun e => w e * g e) = P.map fun e => w' e * g e :=
    List.map_congr_left fun e he => by rw [hw e he]
  have h2 : (P.map fun e => w e * |g e| * (Real.exp (a e) - 1)) =
      P.map fun e => w' e * |g e| * (Real.exp (a' e) - 1) :=
    List.map_congr_left fun e he => by rw [hw e he, ha e he]
  rwa [h1, h2] at h

/-- For nonnegative weights and `0 ≤ y`, a factor `y'` with `|y' - y| ≤ y * (exp b - 1) + ζ`
gives a termwise error of `y' * x`, with weights `y * w e`, exponents `a e + b` and residual
`y * exp b * ρ + ζ * (∑ e ∈ P, w e * |g e| * exp (a e) + ρ)`. -/
theorem TermwiseError.scale {x y y' b ζ ρ : ℝ} {P : List ι} {w g a : ι → ℝ}
    (h : TermwiseError x P w g a ρ) (hy : |y' - y| ≤ y * (Real.exp b - 1) + ζ) (hy0 : 0 ≤ y)
    (hw : ∀ e ∈ P, 0 ≤ w e) :
    TermwiseError (y' * x) P (fun e => y * w e) g (fun e => a e + b)
      (y * Real.exp b * ρ + ζ * ((P.map fun e => w e * |g e| * Real.exp (a e)).sum + ρ)) := by
  unfold TermwiseError at *
  set X := (P.map fun e => w e * g e).sum
  set B := (P.map fun e => w e * |g e| * (Real.exp (a e) - 1)).sum
  set A := (P.map fun e => w e * |g e|).sum
  have hX : |X| ≤ A := abs_sum_le w g hw
  have hB0 : 0 ≤ B + ρ := (abs_nonneg _).trans h
  have hζ : 0 ≤ y * (Real.exp b - 1) + ζ := (abs_nonneg _).trans hy
  have hy' : |y'| ≤ y * Real.exp b + ζ := by
    have := abs_add_le (y' - y) y
    simp only [sub_add_cancel, abs_of_nonneg hy0] at this; linarith
  have hBA : B + A = (P.map fun e => w e * |g e| * Real.exp (a e)).sum := by
    simp only [B, A, ← List.sum_map_add]; congr 1; apply List.map_congr_left; intro e _; ring
  have hscaled : (P.map fun e => y * w e * g e).sum = y * X := by
    simp only [X, ← List.sum_map_mul_left]; congr 1; apply List.map_congr_left; intro e _; ring
  have hbudget : (P.map fun e => y * w e * |g e| * (Real.exp (a e + b) - 1)).sum =
      y * Real.exp b * B + y * (Real.exp b - 1) * A := by
    simp only [B, A, ← List.sum_map_mul_left, ← List.sum_map_add]
    congr 1; apply List.map_congr_left; intro e _; rw [Real.exp_add]; ring
  rw [hscaled, hbudget]
  have split : y' * x - y * X = y' * (x - X) + (y' - y) * X := by ring
  rw [split]
  calc |y' * (x - X) + (y' - y) * X| ≤ |y'| * |x - X| + |y' - y| * |X| := by
        rw [← abs_mul, ← abs_mul]; exact abs_add_le _ _
    _ ≤ (y * Real.exp b + ζ) * (B + ρ) + (y * (Real.exp b - 1) + ζ) * A := by
        have : 0 ≤ y * Real.exp b + ζ := (abs_nonneg _).trans hy'
        gcongr
    _ = y * Real.exp b * B + y * (Real.exp b - 1) * A +
        (y * Real.exp b * ρ + ζ * (B + A + ρ)) := by ring
    _ = _ := by rw [hBA]


/-- For nonnegative weights, a termwise error bounds `|x|` by
`∑ e ∈ P, w e * |g e| * exp (a e) + ρ`. -/
theorem TermwiseError.abs_le {x ρ : ℝ} {P : List ι} {w g a : ι → ℝ}
    (h : TermwiseError x P w g a ρ) (hw : ∀ e ∈ P, 0 ≤ w e) :
    |x| ≤ (P.map fun e => w e * |g e| * Real.exp (a e)).sum + ρ := by
  unfold TermwiseError at h
  have hX := abs_sum_le w g hw
  have hsplit : (P.map fun e => w e * |g e| * (Real.exp (a e) - 1)).sum +
      (P.map fun e => w e * |g e|).sum = (P.map fun e => w e * |g e| * Real.exp (a e)).sum := by
    rw [← List.sum_map_add]; congr 1; apply List.map_congr_left; intro e _; ring
  have := abs_add_le (x - (P.map fun e => w e * g e).sum) (P.map fun e => w e * g e).sum
  simp only [sub_add_cancel] at this
  linarith

/-- An exact factor `g e` of a single term moves into the weight of that term. -/
theorem TermwiseError.mul_single {x ρ : ℝ} {e : ι} {w a : ι → ℝ} (g : ι → ℝ)
    (h : TermwiseError x [e] w (fun _ => 1) a ρ) :
    TermwiseError (x * g e) [e] w g a (ρ * |g e|) := by
  unfold TermwiseError at *
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero, mul_one,
    abs_one] at *
  rw [show x * g e - w e * g e = (x - w e) * g e by ring, abs_mul]
  have := mul_le_mul_of_nonneg_right h (abs_nonneg (g e))
  nlinarith [abs_nonneg (g e)]

/-- The residual of one product of a probability, rounded to storage, with a value of magnitude
at most `V`. -/
def productResidual (u η uL ηL V : ℝ) : ℝ :=
  (1 + u) * (((1 + uL) * η + ηL) * V) + η

/-- The residual growth per block. -/
noncomputable def residualGrowth (u η : ℝ) : ℝ := (1 + u) ^ 2 * (Real.exp u + η)

/-- The residual added per block for `N` keys, blocks of at most `B` keys, a final exponent `C`,
values of magnitude at most `V` and a per-term residual `r`. -/
noncomputable def blockResidual (u η : ℝ) (N B : ℕ) (V C r : ℝ) : ℝ :=
  (1 + u) ^ 2 * η * N * V * Real.exp C + (1 + u) * η + (1 + u) ^ (B + 1) * B * r

/-- For nonnegative `u` and `η`, the residual growth is at least one. -/
theorem one_le_residualGrowth {u η : ℝ} (hu : 0 ≤ u) (hη : 0 ≤ η) : 1 ≤ residualGrowth u η := by
  unfold residualGrowth
  have h1 : 1 ≤ (1 + u) ^ 2 := one_le_pow₀ (by linarith)
  have h2 : 1 ≤ Real.exp u + η := by linarith [Real.one_le_exp hu]
  nlinarith

/-- One step's residual is at most the growth times the incoming residual `ρ` plus the block
residual, so the bound on `ρ` after `t` steps gives the bound after `t + 1`. -/
theorem step_residual_le {u η V c C r ρ : ℝ} {n N b B t : ℕ} (hu : 0 ≤ u) (hη : 0 ≤ η)
    (hV : 0 ≤ V) (hr : 0 ≤ r) (hn : n ≤ N) (hb : b ≤ B) (hc : c ≤ C)
    (hρ : ρ ≤ t * residualGrowth u η ^ t * blockResidual u η N B V C r) :
    (1 + u) * ((1 + u) * (Real.exp u * ρ + η * (n * V * Real.exp c + ρ)) + η +
        (1 + u) ^ b * (0 + b * r)) ≤
      (t + 1 : ℕ) * residualGrowth u η ^ (t + 1) * blockResidual u η N B V C r := by
  have hκ := one_le_residualGrowth hu hη
  have hQ : 0 ≤ blockResidual u η N B V C r := by unfold blockResidual; positivity
  have hnV : (n : ℝ) * V * Real.exp c ≤ N * V * Real.exp C := by
    gcongr
  have hpow : (1 + u) ^ b ≤ (1 + u) ^ B := pow_le_pow_right₀ (by linarith) hb
  have hbr : (b : ℝ) * r ≤ B * r := by gcongr
  have hstep : (1 + u) * ((1 + u) * (Real.exp u * ρ + η * (n * V * Real.exp c + ρ)) + η +
      (1 + u) ^ b * (0 + b * r)) ≤ residualGrowth u η * ρ + blockResidual u η N B V C r := by
    unfold residualGrowth blockResidual
    have h1 : (1 + u) ^ b * (0 + b * r) ≤ (1 + u) ^ B * (B * r) := by
      rw [zero_add]; exact mul_le_mul hpow hbr (by positivity) (by positivity)
    have h2 : η * (n * V * Real.exp c) ≤ η * (N * V * Real.exp C) :=
      mul_le_mul_of_nonneg_left hnV hη
    have h3 : 0 ≤ 1 + u := by linarith
    calc _ = (1 + u) ^ 2 * (Real.exp u + η) * ρ + (1 + u) ^ 2 * (η * (n * V * Real.exp c)) +
          (1 + u) * η + (1 + u) * ((1 + u) ^ b * (0 + b * r)) := by ring
      _ ≤ (1 + u) ^ 2 * (Real.exp u + η) * ρ + (1 + u) ^ 2 * (η * (N * V * Real.exp C)) +
          (1 + u) * η + (1 + u) * ((1 + u) ^ B * (B * r)) := by gcongr
      _ = _ := by ring
  refine hstep.trans ?_
  have : residualGrowth u η * ρ ≤ t * residualGrowth u η ^ (t + 1) *
      blockResidual u η N B V C r := by
    calc residualGrowth u η * ρ ≤ residualGrowth u η *
        (t * residualGrowth u η ^ t * blockResidual u η N B V C r) :=
          mul_le_mul_of_nonneg_left hρ (by linarith)
      _ = _ := by ring
  have hpowκ : 1 ≤ residualGrowth u η ^ (t + 1) := one_le_pow₀ hκ
  push_cast
  nlinarith

/-- A quotient of approximations: if `l` and `o` are within `El` and `Eo` of `S` and `O`, with
`El < S`, then `o / l` is within `(Eo + |O / S| * El) / (S - El)` of `O / S`. -/
theorem abs_div_sub_div_le {l o S O El Eo : ℝ} (hl : |l - S| ≤ El) (ho : |o - O| ≤ Eo)
    (hden : El < S) : S - El ≤ l ∧ |o / l - O / S| ≤ (Eo + |O / S| * El) / (S - El) := by
  have hEl : 0 ≤ El := (abs_nonneg _).trans hl
  have hS : 0 < S := by linarith
  have hlow : S - El ≤ l := by linarith [(abs_le.mp hl).1]
  have hpos : 0 < S - El := by linarith
  have hl0 : 0 < l := hpos.trans_le hlow
  refine ⟨hlow, ?_⟩
  have hEo : 0 ≤ Eo := (abs_nonneg _).trans ho
  have key : o / l - O / S = ((o - O) - O / S * (l - S)) / l := by
    field_simp; ring
  rw [key, abs_div, abs_of_pos hl0]
  have hnum : |(o - O) - O / S * (l - S)| ≤ Eo + |O / S| * El := by
    refine (abs_sub _ _).trans (add_le_add ho ?_)
    rw [abs_mul]; exact mul_le_mul_of_nonneg_left hl (abs_nonneg _)
  calc |(o - O) - O / S * (l - S)| / l ≤ (Eo + |O / S| * El) / l :=
        div_le_div_of_nonneg_right hnum hl0.le
    _ ≤ (Eo + |O / S| * El) / (S - El) :=
        div_le_div_of_nonneg_left (by positivity) hpos hlow

/-- Gibbs' inequality for the shifted exponentials `w = exp (s - M)` with total `S` over `N` keys:
`∑ w * (M - s) ≤ S * (log N - log S)`. The left side is `S` times the entropy of the softmax
weights minus `S * log S`. -/
theorem sum_exp_mul_sub_le (L : List ι) (s : ι → ℝ) (M : ℝ) (hL : L ≠ []) :
    (L.map fun e => Real.exp (s e - M) * (M - s e)).sum ≤
      (L.map fun e => Real.exp (s e - M)).sum *
        (Real.log L.length - Real.log (L.map fun e => Real.exp (s e - M)).sum) := by
  set S := (L.map fun e => Real.exp (s e - M)).sum
  have hN : (0 : ℝ) < L.length := by exact_mod_cast List.length_pos_iff.mpr hL
  have hS : 0 < S := by
    obtain ⟨e, he⟩ := List.exists_mem_of_ne_nil L hL
    refine lt_of_lt_of_le (Real.exp_pos (s e - M)) ?_
    exact List.single_le_sum (fun x hx => by
      obtain ⟨y, -, rfl⟩ := List.mem_map.mp hx; exact (Real.exp_pos _).le) _
      (List.mem_map_of_mem he)
  have hterm (e : ι) : Real.exp (s e - M) * (M - s e) + Real.exp (s e - M) *
      (Real.log S - Real.log L.length) ≤ S / L.length - Real.exp (s e - M) := by
    have hw := Real.exp_pos (s e - M)
    have hlog := Real.log_le_sub_one_of_pos (show 0 < S / (L.length * Real.exp (s e - M)) by
      positivity)
    rw [Real.log_div hS.ne' (by positivity), Real.log_mul hN.ne' hw.ne', Real.log_exp] at hlog
    have := mul_le_mul_of_nonneg_left hlog hw.le
    have hsimp : Real.exp (s e - M) * (S / (L.length * Real.exp (s e - M)) - 1) =
        S / L.length - Real.exp (s e - M) := by field_simp
    nlinarith
  have hsum := List.sum_le_sum (l := L) fun e _ => hterm e
  rw [List.sum_map_add, List.sum_hom₂ L (· - ·) (fun a b c d => add_sub_add_comm a b c d)
    (sub_zero (0 : ℝ)), List.map_const', List.sum_replicate, nsmul_eq_mul, List.sum_map_mul_right]
    at hsum
  have hconst : (L.length : ℝ) * (S / L.length) = S := by field_simp
  rw [hconst] at hsum
  change (L.map fun e => Real.exp (s e - M) * (M - s e)).sum + S * (Real.log S - Real.log L.length)
    ≤ S - S at hsum
  nlinarith

/-- With every scaled shift `u * (M - s)` in `[0, 1]`, a termwise budget with weights `|g| ≤ V` is
at most `V * S * ((exp c - 1) + 2 u exp c (log N - log S))`. -/
theorem sum_budget_le (L : List ι) (s g : ι → ℝ) {M u c V : ℝ} (hL : L ≠ []) (hu : 0 ≤ u)
    (hc : 0 ≤ c) (hs : ∀ e ∈ L, s e ≤ M ∧ u * (M - s e) ≤ 1) (hg : ∀ e ∈ L, |g e| ≤ V) :
    (L.map fun e => Real.exp (s e - M) * |g e| * (Real.exp (u * (M - s e) + c) - 1)).sum ≤
      V * (L.map fun e => Real.exp (s e - M)).sum * ((Real.exp c - 1) + 2 * u * Real.exp c *
        (Real.log L.length - Real.log (L.map fun e => Real.exp (s e - M)).sum)) := by
  have hV : 0 ≤ V := by
    obtain ⟨e, he⟩ := List.exists_mem_of_ne_nil L hL
    exact (abs_nonneg _).trans (hg e he)
  have hterm : ∀ e ∈ L, Real.exp (s e - M) * |g e| * (Real.exp (u * (M - s e) + c) - 1) ≤
      V * (Real.exp c - 1) * Real.exp (s e - M) +
        2 * u * Real.exp c * V * (Real.exp (s e - M) * (M - s e)) := by
    intro e he
    obtain ⟨hsM, hux⟩ := hs e he
    have hx : 0 ≤ u * (M - s e) := mul_nonneg hu (by linarith)
    have hexp := Real.abs_exp_sub_one_le (x := u * (M - s e)) (by rw [abs_of_nonneg hx]; exact hux)
    rw [abs_of_nonneg hx, abs_of_nonneg (by linarith [Real.one_le_exp hx])] at hexp
    have hpos : 0 ≤ Real.exp (u * (M - s e) + c) - 1 := by
      linarith [Real.one_le_exp (add_nonneg hx hc)]
    have hw := (Real.exp_pos (s e - M)).le
    have hsplit : Real.exp (u * (M - s e) + c) - 1 =
        Real.exp c * (Real.exp (u * (M - s e)) - 1) + (Real.exp c - 1) := by
      rw [Real.exp_add]; ring
    calc Real.exp (s e - M) * |g e| * (Real.exp (u * (M - s e) + c) - 1)
        ≤ Real.exp (s e - M) * V * (Real.exp (u * (M - s e) + c) - 1) := by
          gcongr; exact hg e he
      _ = Real.exp (s e - M) * V *
          (Real.exp c * (Real.exp (u * (M - s e)) - 1) + (Real.exp c - 1)) := by rw [hsplit]
      _ ≤ Real.exp (s e - M) * V * (Real.exp c * (2 * (u * (M - s e))) + (Real.exp c - 1)) := by
          gcongr
      _ = _ := by ring
  refine (List.sum_le_sum hterm).trans ?_
  rw [List.sum_map_add, List.sum_map_mul_left, List.sum_map_mul_left]
  have hent := sum_exp_mul_sub_le L s M hL
  have h0 : 0 ≤ 2 * u * Real.exp c * V := by positivity
  nlinarith [mul_le_mul_of_nonneg_left hent h0]

/-- The arithmetic that turns termwise budgets `El ≤ S (al + ρl)` and `Eo ≤ S (V ao + ρo)`, with a
normalizer `S ≥ 1` and an average bounded by `V`, into the closed-form output bound. -/
theorem closed_form_le {S O El Eo V al ao ρl ρo unit under : ℝ} (hS1 : 1 ≤ S)
    (hEl : El ≤ S * (al + ρl)) (hEo : Eo ≤ S * (V * ao + ρo)) (hO : |O| ≤ V * S)
    (hsmall : al + ρl < 1) (hal : 0 ≤ al) (hao : 0 ≤ ao) (hρl : 0 ≤ ρl) (hρo : 0 ≤ ρo)
    (hV : 0 ≤ V) (hu : 0 ≤ unit) (hEl0 : 0 ≤ El) :
    El < S ∧ (|O| + Eo) / (S - El) ≤ (V + V * ao + ρo) / (1 - al - ρl) ∧
      (1 + unit) * ((Eo + |O / S| * El) / (S - El)) + unit * |O / S| + under ≤
        (1 + unit) * ((V * (ao + al + ρl) + ρo) / (1 - al - ρl)) + unit * V + under := by
  have hS0 : 0 < S := by linarith
  have hgap : 0 < 1 - al - ρl := by linarith
  have hden : S * (1 - al - ρl) ≤ S - El := by nlinarith
  have hdenpos : 0 < S - El := lt_of_lt_of_le (mul_pos hS0 hgap) hden
  have hOS : |O / S| ≤ V := by
    rw [abs_div, abs_of_pos hS0, div_le_iff₀ hS0]; linarith
  refine ⟨by linarith, ?_, ?_⟩
  · rw [div_le_div_iff₀ hdenpos hgap]
    have : |O| + Eo ≤ S * (V + V * ao + ρo) := by nlinarith
    have hnum : 0 ≤ V + V * ao + ρo := by positivity
    nlinarith
  · have hD : (Eo + |O / S| * El) / (S - El) ≤ (V * (ao + al + ρl) + ρo) / (1 - al - ρl) := by
      rw [div_le_div_iff₀ hdenpos hgap]
      have h1 : |O / S| * El ≤ V * (S * (al + ρl)) := mul_le_mul hOS hEl hEl0 hV
      have hnum : Eo + |O / S| * El ≤ S * (V * (ao + al + ρl) + ρo) := by nlinarith
      have hnum0 : 0 ≤ V * (ao + al + ρl) + ρo := by positivity
      nlinarith
    have := mul_le_mul_of_nonneg_left hD (by linarith : (0 : ℝ) ≤ 1 + unit)
    nlinarith [mul_le_mul_of_nonneg_left hOS hu]

/-- The closed form of the termwise budgets. With the largest score `M` attained by a key, every
scaled shift `unit * (M - s)` at most one, values bounded by `V` and `al + ρl < 1`, the budgets
`El = ∑ w (exp (unit (M - s) + cl) - 1) + ρl` and `Eo = ∑ w |v| (exp (unit (M - s) + co) - 1) + ρo`
satisfy `El < S`, the quotient range `(|O| + Eo) / (S - El)` is at most
`(V + V ao + ρo) / (1 - al - ρl)`, and the bound of `div_error` is at most the closed form in
`al = (exp cl - 1) + 2 unit exp cl log N` and `ao` alike. -/
theorem budget_closed_form (L : List ι) (s v : ι → ℝ) {M V cl co ρl ρo unit under : ℝ}
    (hL : L ≠ []) (hu : 0 ≤ unit) (hcl : 0 ≤ cl) (hco : 0 ≤ co) (hρl : 0 ≤ ρl) (hρo : 0 ≤ ρo)
    (hV : 0 ≤ V) (hattained : ∃ e ∈ L, s e = M)
    (hshift : ∀ e ∈ L, s e ≤ M ∧ unit * (M - s e) ≤ 1) (hv : ∀ e ∈ L, |v e| ≤ V) :
    let S := (L.map fun e => Real.exp (s e - M)).sum
    let O := (L.map fun e => Real.exp (s e - M) * v e).sum
    let El := (L.map fun e => Real.exp (s e - M) * (Real.exp (unit * (M - s e) + cl) - 1)).sum +
      ρl
    let Eo := (L.map fun e => Real.exp (s e - M) * |v e| *
      (Real.exp (unit * (M - s e) + co) - 1)).sum + ρo
    let al := (Real.exp cl - 1) + 2 * unit * Real.exp cl * Real.log L.length
    let ao := (Real.exp co - 1) + 2 * unit * Real.exp co * Real.log L.length
    al + ρl < 1 →
      El < S ∧ (|O| + Eo) / (S - El) ≤ (V + V * ao + ρo) / (1 - al - ρl) ∧
        (1 + unit) * ((Eo + |O / S| * El) / (S - El)) + unit * |O / S| + under ≤
          (1 + unit) * ((V * (ao + al + ρl) + ρo) / (1 - al - ρl)) + unit * V + under := by
  intro S O El Eo al ao hsmall
  have hS1 : 1 ≤ S := by
    obtain ⟨e, he, hEq⟩ := hattained
    calc (1 : ℝ) = Real.exp (s e - M) := by rw [hEq, sub_self, Real.exp_zero]
      _ ≤ S := List.single_le_sum (fun x hx => by
          obtain ⟨y, -, rfl⟩ := List.mem_map.mp hx; exact (Real.exp_pos _).le) _
          (List.mem_map_of_mem he)
  have hS0 : 0 < S := by linarith
  have hlogS : 0 ≤ Real.log S := Real.log_nonneg hS1
  have hNlog : 0 ≤ Real.log L.length := Real.log_natCast_nonneg _
  have hbl : (L.map fun e => Real.exp (s e - M) * (Real.exp (unit * (M - s e) + cl) - 1)).sum ≤
      S * ((Real.exp cl - 1) + 2 * unit * Real.exp cl * (Real.log L.length - Real.log S)) := by
    have := sum_budget_le L s (fun _ => 1) (M := M) (V := 1) hL hu hcl hshift (fun _ _ => by simp)
    simpa only [abs_one, mul_one, one_mul] using this
  have hbo := sum_budget_le L s v (M := M) (V := V) hL hu hco hshift hv
  have hElS : El ≤ S * (al + ρl) := by
    have h1 : 2 * unit * Real.exp cl * (Real.log L.length - Real.log S) ≤
        2 * unit * Real.exp cl * Real.log L.length :=
      mul_le_mul_of_nonneg_left (by linarith) (by positivity)
    have h2 : ρl ≤ S * ρl := le_mul_of_one_le_left hρl hS1
    have h3 : S * ((Real.exp cl - 1) + 2 * unit * Real.exp cl * (Real.log L.length -
        Real.log S)) ≤ S * al := mul_le_mul_of_nonneg_left (by simp only [al]; linarith) hS0.le
    have : S * (al + ρl) = S * al + S * ρl := by ring
    simp only [El]; rw [this]; linarith
  have hEoS : Eo ≤ S * (V * ao + ρo) := by
    have h1 : 2 * unit * Real.exp co * (Real.log L.length - Real.log S) ≤
        2 * unit * Real.exp co * Real.log L.length :=
      mul_le_mul_of_nonneg_left (by linarith) (by positivity)
    have h2 : ρo ≤ S * ρo := le_mul_of_one_le_left hρo hS1
    have h3 : V * S * ((Real.exp co - 1) + 2 * unit * Real.exp co * (Real.log L.length -
        Real.log S)) ≤ V * S * ao :=
      mul_le_mul_of_nonneg_left (by simp only [ao]; linarith) (mul_nonneg hV hS0.le)
    have : S * (V * ao + ρo) = V * S * ao + S * ρo := by ring
    simp only [Eo]; rw [this]; linarith
  have hO : |O| ≤ V * S := by
    refine (abs_sum_le (P := L) (fun e => Real.exp (s e - M)) v
      (fun e _ => (Real.exp_pos _).le)).trans ?_
    rw [← List.sum_map_mul_left]
    refine List.sum_le_sum fun e he => ?_
    rw [mul_comm V]
    exact mul_le_mul_of_nonneg_left (hv e he) (Real.exp_pos _).le
  have hal : 0 ≤ al := by
    have : 0 ≤ Real.exp cl - 1 := by linarith [Real.one_le_exp hcl]
    positivity
  have hao : 0 ≤ ao := by
    have : 0 ≤ Real.exp co - 1 := by linarith [Real.one_le_exp hco]
    positivity
  have hEl0 : 0 ≤ El := by
    have : 0 ≤ (L.map fun e => Real.exp (s e - M) * (Real.exp (unit * (M - s e) + cl) - 1)).sum :=
      List.sum_nonneg fun x hx => by
        obtain ⟨e, he, rfl⟩ := List.mem_map.mp hx
        have := Real.one_le_exp (add_nonneg (mul_nonneg hu (sub_nonneg.mpr (hshift e he).1)) hcl)
        exact mul_nonneg (Real.exp_pos _).le (by linarith)
    simp only [El]; linarith
  exact closed_form_le hS1 hElS hEoS hO hsmall hal hao hρl hρo hV hu hEl0

/-- The closed form of `onlineAverage_blocks_error` grows with each of its budgets. -/
theorem closed_form_mono {u η V al ao ρl ρo al' ao' ρl' ρo' : ℝ} (hu : 0 ≤ u) (hV : 0 ≤ V)
    (hal : al ≤ al') (hao : ao ≤ ao') (hρl : ρl ≤ ρl') (hρo : ρo ≤ ρo') (hal0 : 0 ≤ al)
    (hao0 : 0 ≤ ao) (hρl0 : 0 ≤ ρl) (hρo0 : 0 ≤ ρo) (hsmall : al' + ρl' < 1) :
    (1 + u) * ((V * (ao + al + ρl) + ρo) / (1 - al - ρl)) + u * V + η ≤
      (1 + u) * ((V * (ao' + al' + ρl') + ρo') / (1 - al' - ρl')) + u * V + η := by
  have h1 : 0 < 1 - al' - ρl' := by linarith
  have h2 : 1 - al' - ρl' ≤ 1 - al - ρl := by linarith
  gcongr
  · nlinarith

/-- Upper bounds `Ecl ≥ exp cl`, `Eco ≥ exp co`, `Eu ≥ exp u`, `Lg ≥ log N` and
`Pw ≥ (1 + u) ^ (B + 1)` bound every quantity of the closed form by its value at those bounds. -/
theorem closed_form_le_of_bounds {u η V uL ηL cl co Ecl Eco Eu Lg Pw : ℝ} {N B T : ℕ}
    (hu : 0 ≤ u) (hη : 0 ≤ η) (hV : 0 ≤ V) (huL : 0 ≤ uL) (hηL : 0 ≤ ηL) (hcl : 0 ≤ cl)
    (hco : 0 ≤ co) (hEcl : Real.exp cl ≤ Ecl) (hEco : Real.exp co ≤ Eco) (hEu : Real.exp u ≤ Eu)
    (hLg : Real.log N ≤ Lg) (hPw : (1 + u) ^ (B + 1) ≤ Pw) :
    let al := (Real.exp cl - 1) + 2 * u * Real.exp cl * Real.log N
    let ao := (Real.exp co - 1) + 2 * u * Real.exp co * Real.log N
    let ρl := T * residualGrowth u η ^ T * blockResidual u η N B 1 cl η
    let ρo := T * residualGrowth u η ^ T * blockResidual u η N B V co (productResidual u η uL ηL V)
    let al' := (Ecl - 1) + 2 * u * Ecl * Lg
    let ao' := (Eco - 1) + 2 * u * Eco * Lg
    let κ := (1 + u) ^ 2 * (Eu + η)
    let ρl' := T * κ ^ T * ((1 + u) ^ 2 * η * N * 1 * Ecl + (1 + u) * η + Pw * B * η)
    let ρo' := T * κ ^ T * ((1 + u) ^ 2 * η * N * V * Eco + (1 + u) * η +
      Pw * B * productResidual u η uL ηL V)
    al' + ρl' < 1 →
      al + ρl < 1 ∧
      (N : ℝ) * 1 * Real.exp cl + ρl ≤ N * 1 * Ecl + ρl' ∧
      (N : ℝ) * V * Real.exp co + ρo ≤ N * V * Eco + ρo' ∧
      (V + V * ao + ρo) / (1 - al - ρl) ≤ (V + V * ao' + ρo') / (1 - al' - ρl') ∧
      (1 + u) * ((V * (ao + al + ρl) + ρo) / (1 - al - ρl)) + u * V + η ≤
        (1 + u) * ((V * (ao' + al' + ρl') + ρo') / (1 - al' - ρl')) + u * V + η := by
  intro al ao ρl ρo al' ao' κ ρl' ρo' hsmall
  have hlog : 0 ≤ Real.log N := Real.log_natCast_nonneg N
  have hecl := Real.one_le_exp hcl
  have heco := Real.one_le_exp hco
  have h2u : 0 ≤ 2 * u := by positivity
  have hEcl0 : 0 ≤ Ecl := (Real.exp_pos cl).le.trans hEcl
  have hEco0 : 0 ≤ Eco := (Real.exp_pos co).le.trans hEco
  have hEu0 : 0 ≤ Eu := (Real.exp_pos u).le.trans hEu
  have hal : al ≤ al' := by
    simp only [al, al']
    have : 2 * u * Real.exp cl * Real.log N ≤ 2 * u * Ecl * Lg :=
      mul_le_mul (mul_le_mul_of_nonneg_left hEcl h2u) hLg hlog
        (mul_nonneg h2u (by linarith))
    linarith
  have hao : ao ≤ ao' := by
    simp only [ao, ao']
    have : 2 * u * Real.exp co * Real.log N ≤ 2 * u * Eco * Lg :=
      mul_le_mul (mul_le_mul_of_nonneg_left hEco h2u) hLg hlog
        (mul_nonneg h2u (by linarith))
    linarith
  have hκ : residualGrowth u η ≤ κ := by
    simp only [residualGrowth, κ]; gcongr
  have hκ0 : 0 ≤ residualGrowth u η := by unfold residualGrowth; positivity
  have hpr : 0 ≤ productResidual u η uL ηL V := by unfold productResidual; positivity
  have hρl : ρl ≤ ρl' := by
    simp only [ρl, ρl', blockResidual]
    gcongr
  have hρo : ρo ≤ ρo' := by
    simp only [ρo, ρo', blockResidual]
    gcongr
  have hal0 : 0 ≤ al := by
    simp only [al]
    have := mul_nonneg (mul_nonneg h2u (Real.exp_pos cl).le) hlog
    linarith
  have hao0 : 0 ≤ ao := by
    simp only [ao]
    have := mul_nonneg (mul_nonneg h2u (Real.exp_pos co).le) hlog
    linarith
  have hρl0 : 0 ≤ ρl := by simp only [ρl, blockResidual]; positivity
  have hρo0 : 0 ≤ ρo := by simp only [ρo, blockResidual]; positivity
  refine ⟨by linarith, by gcongr, by gcongr, ?_, closed_form_mono hu hV hal hao hρl hρo hal0 hao0
    hρl0 hρo0 hsmall⟩
  have h1 : 0 < 1 - al' - ρl' := by linarith
  have hnum : V + V * ao + ρo ≤ V + V * ao' + ρo' := by
    have := mul_le_mul_of_nonneg_left hao hV
    linarith
  have hnum0 : 0 ≤ V + V * ao' + ρo' := by
    have := mul_nonneg hV (hao0.trans hao)
    linarith
  exact div_le_div₀ hnum0 hnum h1 (by linarith)

/-- `(1 + u) ^ n ≤ 1 / (1 - n u)` when `n u < 1`. -/
theorem one_add_pow_le_inv {u : ℝ} (n : ℕ) (hu : 0 ≤ u) (hn : n * u < 1) :
    (1 + u) ^ n ≤ 1 / (1 - n * u) := by
  calc (1 + u) ^ n ≤ Real.exp u ^ n := by
        gcongr; linarith [Real.add_one_le_exp u]
    _ = Real.exp (n * u) := by rw [← Real.exp_nat_mul]
    _ ≤ 1 / (1 - n * u) := Real.exp_bound_div_one_sub_of_interval (by positivity) hn

/-- With one exponent `c` for every term, the termwise allowance is the sum of magnitudes times
`exp c - 1`. -/
theorem TermwiseError.abs_sub_le {x ρ c : ℝ} {P : List ι} {w g a : ι → ℝ}
    (h : TermwiseError x P w g a ρ) (ha : ∀ e ∈ P, a e = c) :
    |x - (P.map fun e => w e * g e).sum| ≤
      (P.map fun e => w e * |g e|).sum * (Real.exp c - 1) + ρ := by
  have h' := h.congr (a' := fun _ => c) (fun _ _ => rfl) ha
  unfold TermwiseError at h'
  rwa [List.sum_map_mul_right] at h'

/-- A sum of terms in `[0, 1]` lies in `[0, L.length]`. -/
theorem sum_map_mem_Icc (L : List ι) (f : ι → ℝ) (h : ∀ e ∈ L, f e ∈ Set.Icc 0 1) :
    0 ≤ (L.map f).sum ∧ (L.map f).sum ≤ L.length := by
  constructor
  · have := List.sum_le_sum (l := L) (f := fun _ => (0 : ℝ)) (g := f) (fun e he => (h e he).1)
    rwa [List.map_const', List.sum_replicate, nsmul_eq_mul, mul_zero] at this
  · have := List.sum_le_sum (l := L) (f := f) (g := fun _ => (1 : ℝ)) (fun e he => (h e he).2)
    rwa [List.map_const', List.sum_replicate, nsmul_eq_mul, mul_one] at this

/-- With nonnegative weights `q` and values in `[lo, hi]`, the weighted sum lies between `lo` and
`hi` times the sum of the weights. -/
theorem sum_map_mul_bounds (L : List ι) (q v : ι → ℝ) {lo hi : ℝ}
    (hq : ∀ e ∈ L, 0 ≤ q e) (hv : ∀ e ∈ L, lo ≤ v e ∧ v e ≤ hi) :
    lo * (L.map q).sum ≤ (L.map fun e => q e * v e).sum ∧
      (L.map fun e => q e * v e).sum ≤ hi * (L.map q).sum := by
  rw [← List.sum_map_mul_left, ← List.sum_map_mul_left]
  exact ⟨List.sum_le_sum fun e he => by nlinarith [hq e he, (hv e he).1],
    List.sum_le_sum fun e he => by nlinarith [hq e he, (hv e he).2]⟩

/-- With nonnegative weights `q` and values bounded by `V`, the weighted sum of magnitudes is at
most `V` times the sum of the weights. -/
theorem sum_map_mul_abs_le (L : List ι) (q v : ι → ℝ) {V : ℝ}
    (hq : ∀ e ∈ L, 0 ≤ q e) (hv : ∀ e ∈ L, |v e| ≤ V) :
    (L.map fun e => q e * |v e|).sum ≤ V * (L.map q).sum := by
  rw [← List.sum_map_mul_left]
  exact List.sum_le_sum fun e he => by nlinarith [hq e he, hv e he]

/-- If termwise `|q - t| ≤ δ t + ηs`, the sums satisfy `|∑ q - ∑ t| ≤ δ ∑ t + L.length ηs`. -/
theorem abs_sum_map_sub_le (L : List ι) (t q : ι → ℝ) {δ ηs : ℝ}
    (h : ∀ e ∈ L, |q e - t e| ≤ δ * t e + ηs) :
    |(L.map q).sum - (L.map t).sum| ≤ δ * (L.map t).sum + L.length * ηs := by
  rw [← List.sum_hom₂ L (· - ·) (fun a b c d => add_sub_add_comm a b c d) (sub_zero 0) q t]
  refine (List.le_sum_of_subadditive abs abs_zero.le abs_add_le _).trans ?_
  rw [List.map_map]
  calc (L.map fun e => |q e - t e|).sum ≤ (L.map fun e => δ * t e + ηs).sum :=
        List.sum_le_sum h
    _ = δ * (L.map t).sum + L.length * ηs := by
        rw [List.sum_map_add, List.sum_map_mul_left, List.map_const', List.sum_replicate,
          nsmul_eq_mul]

/-- Rescaling exact sums `L, W, O` by `α ∈ [0, 1]` and adding block sums `Lb, Wb, Ob` with the same
bounds keeps every bound, with the lengths `n` and `b` adding. -/
theorem rescaled_sums_bounds {α L W O Lb Wb Ob V lo hi δ ηs n b : ℝ} (hα : α ∈ Set.Icc 0 1)
    (hηs : 0 ≤ ηs) (hn : 0 ≤ n) (hL : 0 ≤ L ∧ L ≤ n) (hW : 0 ≤ W ∧ W ≤ n) (hOV : |O| ≤ V * W)
    (hlo : lo * W ≤ O) (hhi : O ≤ hi * W) (hWL : |W - L| ≤ δ * L + n * ηs)
    (hLb : 0 ≤ Lb ∧ Lb ≤ b) (hWb : 0 ≤ Wb ∧ Wb ≤ b) (hOb : |Ob| ≤ V * Wb)
    (hlob : lo * Wb ≤ Ob) (hhib : Ob ≤ hi * Wb) (hWLb : |Wb - Lb| ≤ δ * Lb + b * ηs) :
    (0 ≤ α * L + Lb ∧ α * L + Lb ≤ n + b) ∧ (0 ≤ α * W + Wb ∧ α * W + Wb ≤ n + b) ∧
      |α * O + Ob| ≤ V * (α * W + Wb) ∧ lo * (α * W + Wb) ≤ α * O + Ob ∧
      α * O + Ob ≤ hi * (α * W + Wb) ∧
      |α * W + Wb - (α * L + Lb)| ≤ δ * (α * L + Lb) + (n + b) * ηs := by
  obtain ⟨hα0, hα1⟩ := hα
  have hαL0 : 0 ≤ α * L := mul_nonneg hα0 hL.1
  have hαW0 : 0 ≤ α * W := mul_nonneg hα0 hW.1
  have hαL : α * L ≤ L := mul_le_of_le_one_left hL.1 hα1
  have hαW : α * W ≤ W := mul_le_of_le_one_left hW.1 hα1
  have hαO : |α * O| ≤ α * (V * W) := by
    rw [abs_mul, abs_of_nonneg hα0]; exact mul_le_mul_of_nonneg_left hOV hα0
  have hαlo := mul_le_mul_of_nonneg_left hlo hα0
  have hαhi := mul_le_mul_of_nonneg_left hhi hα0
  have hαWL : |α * (W - L)| ≤ α * (δ * L + n * ηs) := by
    rw [abs_mul, abs_of_nonneg hα0]; exact mul_le_mul_of_nonneg_left hWL hα0
  have hαn : α * (n * ηs) ≤ n * ηs := mul_le_of_le_one_left (mul_nonneg hn hηs) hα1
  have hsum := abs_add_le (α * O) Ob
  have hdiff := abs_add_le (α * (W - L)) (Wb - Lb)
  have e1 : α * W + Wb - (α * L + Lb) = α * (W - L) + (Wb - Lb) := by ring
  refine ⟨⟨by linarith, by linarith⟩, ⟨by linarith, by linarith⟩, by linarith, by linarith,
    by linarith, ?_⟩
  rw [e1]
  linarith

/-- Upper bounds `Ecl ≥ exp (u (B + 2 T))`, `Eco ≥ exp (u (B + 1 + 2 T))`, `Pt ≥ (1 + u) ^ (2 T)`
and `PB ≥ (1 + u) ^ B` bound the budget, range and error terms of `onlineAverage_stored_error` by
the same expressions in the bounds. -/
theorem stored_closed_form_le_of_bounds {u η V V' Ecl Eco Pt PB : ℝ} {N B T : ℕ} (hu : 0 ≤ u)
    (hη : 0 ≤ η) (hV : 0 ≤ V) (hV' : 0 ≤ V') (hEcl : Real.exp (u * (B + 2 * T)) ≤ Ecl)
    (hEco : Real.exp (u * (B + 1 + 2 * T)) ≤ Eco) (hPt : (1 + u) ^ (2 * T) ≤ Pt)
    (hPB : (1 + u) ^ B ≤ PB) :
    let al := Real.exp (u * (B + 2 * T)) - 1
    let ao := Real.exp (u * (B + 1 + 2 * T)) - 1
    let ρl := T * (1 + u) ^ (2 * T) * η
    let ρo := T * (1 + u) ^ (2 * T) * (η + (1 + u) ^ B * (B * η))
    let ρl' := T * Pt * η
    let ρo' := T * Pt * (η + PB * (B * η))
    (Ecl - 1) + ρl' < 1 →
      al + ρl < 1 ∧
      (N : ℝ) * Real.exp (u * (B + 2 * T)) + ρl ≤ N * Ecl + ρl' ∧
      (N : ℝ) * V * Real.exp (u * (B + 1 + 2 * T)) + ρo ≤ N * V * Eco + ρo' ∧
      (V' + V' * ao + ρo) / (1 - al - ρl) ≤
        (V' + V' * (Eco - 1) + ρo') / (1 - (Ecl - 1) - ρl') ∧
      (1 + u) * ((V' * (ao + al + ρl) + ρo) / (1 - al - ρl)) + u * V' + η ≤
        (1 + u) * ((V' * ((Eco - 1) + (Ecl - 1) + ρl') + ρo') / (1 - (Ecl - 1) - ρl')) +
          u * V' + η := by
  intro al ao ρl ρo ρl' ρo' hsmall
  have hN0 : (0 : ℝ) ≤ N := Nat.cast_nonneg N
  have hPt0 : 0 ≤ Pt := (pow_nonneg (by linarith) _).trans hPt
  have hPB0 : 0 ≤ PB := (pow_nonneg (by linarith) _).trans hPB
  have hecl := Real.one_le_exp (x := u * (B + 2 * T)) (by positivity)
  have heco := Real.one_le_exp (x := u * (B + 1 + 2 * T)) (by positivity)
  have hal : al ≤ Ecl - 1 := by simp only [al]; linarith
  have hal0 : 0 ≤ al := by simp only [al]; linarith
  have hao : ao ≤ Eco - 1 := by simp only [ao]; linarith
  have hao0 : 0 ≤ ao := by simp only [ao]; linarith
  have hρl : ρl ≤ ρl' := by simp only [ρl, ρl']; gcongr
  have hρl0 : 0 ≤ ρl := by simp only [ρl]; positivity
  have hρo : ρo ≤ ρo' := by simp only [ρo, ρo']; gcongr
  have hρo0 : 0 ≤ ρo := by simp only [ρo]; positivity
  have hd : 0 < 1 - (Ecl - 1) - ρl' := by linarith
  have hdb : 1 - (Ecl - 1) - ρl' ≤ 1 - al - ρl := by linarith
  have hVao := mul_le_mul_of_nonneg_left hao hV'
  have hVEco : 0 ≤ V' * (Eco - 1) := mul_nonneg hV' (by linarith)
  have hsum : V' * (ao + al + ρl) ≤ V' * ((Eco - 1) + (Ecl - 1) + ρl') :=
    mul_le_mul_of_nonneg_left (by linarith) hV'
  have hsum0 : 0 ≤ V' * ((Eco - 1) + (Ecl - 1) + ρl') :=
    mul_nonneg hV' (by linarith)
  have hq := div_le_div₀ (add_nonneg hsum0 (hρo0.trans hρo)) (add_le_add hsum hρo) hd hdb
  refine ⟨by linarith, add_le_add (mul_le_mul_of_nonneg_left hEcl hN0) hρl,
    add_le_add (mul_le_mul_of_nonneg_left hEco (mul_nonneg hN0 hV)) hρo,
    div_le_div₀ (by linarith) (by linarith) hd hdb, ?_⟩
  linarith [mul_le_mul_of_nonneg_left hq (by linarith : (0 : ℝ) ≤ 1 + u)]

end TorchLean.OnlineSoftmax

namespace TorchLean.Tensor

/-- Over the reals, attention by the online schedule is attention by the two-pass schedule, for
every block size, mask and multiplier. -/
theorem attention_online_eq {batch : Shape} {rows keys depth width : Nat}
    (query : Tensor ℝ (batch ++ [rows, depth])) (key : Tensor ℝ (batch ++ [keys, depth]))
    (value : Tensor ℝ (batch ++ [keys, width])) (scale : ℝ)
    (mask : Option (Tensor Bool [rows, keys]))
    (weights : Option (Tensor ℝ (batch ++ [rows, keys]))) (blocks : Softmax.BlockPartition) :
    attention query key value scale mask weights (.online blocks) =
      attention query key value scale mask weights :=
  OnlineSoftmax.attentionOnline_eq_real blocks _ value weights mask

/-- Over the reals, softmax by the online schedule is softmax by the two-pass schedule, at every
axis and rank, for every block size and mask. -/
theorem softmax_online_eq {s : Shape} (x : Tensor ℝ s) (dim : Fin s.rank)
    (mask : Option (Tensor Bool s)) (blocks : Softmax.BlockPartition) :
    softmax x dim mask (.online blocks) = softmax x dim mask := by
  refine Rep.ext fun c => ?_
  rw [softmax_online_apply, OnlineSoftmax.onlineWeight_real]
  cases mask with
  | none =>
    rw [softmax_apply, Softmax.getScalar_vec_eq_exp_div]
    simp
  | some mask =>
    rw [softmax_mask_apply, Softmax.getScalar_hardMaskedVec]
    simp only [HardMaskedSoftmax.numerator,
      HardMaskedSoftmax.denominator, getScalarE_ofLp]
    rfl

end TorchLean.Tensor

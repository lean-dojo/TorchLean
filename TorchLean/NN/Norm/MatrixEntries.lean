/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Layers.Normalization
public import TorchLean.Tensor.Elementwise.Real
public import TorchLean.Tensor.Statistics
public import TorchLean.NN.Tape.RealNode.Matrix
public import TorchLean.NN.Norm.RowNorm
public import TorchLean.Tensor.List

/-!
# Entrywise formulas for row-normalization tensors

The normalization specs are written with sums and variances over one axis of an `[m, n]` matrix
(`reduce .sum` and `var` cast to the shape without that axis), broadcasts back along that axis,
column broadcasts and pointwise tensor arithmetic. The lemmas here work out what each of those
operations does to a single entry `get2 t i j`, so that the LayerNorm and BatchNorm proofs can
reason with plain real sums. Entries of filled tensors come from `TorchLean.Tensor.getScalar_full`
and `TorchLean.Tensor.get2_full`, and `Matmul.toVec_idxMN` reads `toVec` of a matrix at
`idxMN i j`.

`normalizedRows` divides each centered row by its scale `Tensor.normalizationScale` before
squaring it. Over the reals that scale cancels: for a matrix with `0 < n` columns, `get2_layerNorm`
gives each entry of `Tensor.layerNorm` over the last axis by the formula without the scale.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)
open TorchLean.Shape (rank size size_cons)

namespace TorchLean
namespace Normalization

open Tensor
open Shape.Coord

open scoped BigOperators

noncomputable section

variable {m n : Nat}

/-! ## Matrix vectorization -/

/-- Two flattened matrices agree once they agree at every `idxMN i j`. -/
theorem vec_ext_idxMN {u v : Vec (Shape.size [m, n])}
    (h : ∀ (i : Fin m) (j : Fin n),
      u (Matmul.idxMN (m := m) (n := n) i j) =
        v (Matmul.idxMN (m := m) (n := n) i j)) :
    u = v := by
  apply PiLp.ext
  intro ip
  rw [← RowNorm.idxMN_rowOf_colOf ip]
  exact h _ _

/-! ## Broadcasts and reductions along one axis -/

/-- Row broadcast of a length-`m` vector to an `m × n` matrix. -/
abbrev rowBroadcast (v : Tensor ℝ [m]) : Tensor ℝ [m, n] :=
  Tensor.broadcast v [m, n] _ (Shape.eraseIdx_broadcastDims [m, n] 1)

/-- Sums of the rows of an `m × n` matrix. -/
abbrev rowSum (x : Tensor ℝ [m, n]) : Tensor ℝ [m] :=
  castShape (reduce .sum x [⟨1, Nat.one_lt_two⟩] false) (Shape.reduce_singleton _ _)

/-- Sums of the columns of an `m × n` matrix. -/
abbrev colSum (x : Tensor ℝ [m, n]) : Tensor ℝ [n] :=
  castShape (reduce .sum x [⟨0, Nat.two_pos⟩] false) (Shape.reduce_singleton _ _)

/-- Entries of a row broadcast: every column of row `i` reads the vector entry `i`. -/
theorem get2_rowBroadcast (v : Tensor ℝ [m]) (i : Fin m) (j : Fin n) :
    Tensor.get2 (rowBroadcast (n := n) v) i j = getScalar v i := by
  rw [Tensor.get2_eq_apply, Tensor.getScalar_eq_apply, rowBroadcast]
  erw [broadcast_eraseIdx_apply (s := [m, n]) 1]
  rfl

/-- Entries of a column broadcast: every row reads the vector entry `j`. -/
theorem get2_broadcast_col (h : Shape.CanBroadcastTo (.dim n .scalar) (.dim m (.dim n .scalar)))
    (v : Tensor ℝ [n]) (i : Fin m) (j : Fin n) :
    Tensor.get2 (Tensor.broadcast v _ h.dims h.broadcastDims) i j = getScalar v j := by
  rw [Tensor.get2_eq_apply, Tensor.getScalar_eq_apply, Tensor.broadcast_apply]
  congr 1
  apply Shape.Coord.ext_val
  intro d
  fin_cases d
  simp only [Shape.Coord.get_ofFun]
  split_ifs with h1
  · have := j.isLt
    change n = 1 at h1
    show 0 = j.val
    omega
  · rfl

/-- Entry `i` of `rowSum x` is the sum of row `i`. -/
theorem getScalar_rowSum (x : Tensor ℝ [m, n]) (i : Fin m) :
    getScalar (rowSum x) i = ∑ j : Fin n, Tensor.get2 x i j := by
  rw [Tensor.getScalar_eq_apply]
  erw [castShape_reduce_sum_singleton_mk_apply x 1]
  rw [← Tensor.sum_eq_foldl, sum_vec]
  simp only [Tensor.getScalar_eq_apply, Tensor.ofFn, Rep.get_ofFn,
    Tensor.get2_eq_apply]
  rfl

/-- Entry `j` of `colSum x` is the sum of column `j`. -/
theorem getScalar_colSum (x : Tensor ℝ [m, n]) (j : Fin n) :
    getScalar (colSum x) j = ∑ i : Fin m, Tensor.get2 x i j := by
  rw [Tensor.getScalar_eq_apply]
  erw [castShape_reduce_sum_singleton_mk_apply x 0]
  rw [← Tensor.sum_eq_foldl, sum_vec]
  simp only [Tensor.getScalar_eq_apply, Tensor.ofFn, Rep.get_ofFn,
    Tensor.get2_eq_apply]
  rfl

/-- Entry `i` of the row mean is the sum of row `i` divided by `n`. -/
theorem getScalar_rowMean (x : Tensor ℝ [m, n]) (i : Fin m) :
    getScalar (Tensor.map (Context.divNat · n) (rowSum x)) i =
      (∑ j : Fin n, Tensor.get2 x i j) / n := by
  rw [getScalar_map, getScalar_rowSum, Context.divNat_real]

/-- For `0 < n`, entry `i` of `var x` over the last axis is the population variance of row `i`. -/
theorem getScalar_rowVar (hn : 0 < n) (x : Tensor ℝ [m, n]) (i : Fin m) :
    getScalar (castShape (var x [⟨1, Nat.one_lt_two⟩] false) (Shape.reduce_singleton _ _)) i =
      (∑ j : Fin n,
        (Tensor.get2 x i j - (∑ k : Fin n, Tensor.get2 x i k) / n) *
          (Tensor.get2 x i j - (∑ k : Fin n, Tensor.get2 x i k) / n)) / n := by
  rw [Tensor.getScalar_eq_apply]
  erw [castShape_var_singleton_apply x ⟨1, Nat.one_lt_two⟩]
  dsimp only
  have hcount : (if Shape.axisSize [m, n] 1 = 0 then 1 else Shape.axisSize [m, n] 1) = n := by
    rw [ite_eq_right (by simp; omega)]
    simp
  simp only [hcount, Context.divNat_real]
  rw [sum_vec, sum_vec]
  simp only [Tensor.getScalar_eq_apply, Tensor.ofFn, Rep.get_ofFn,
    Tensor.get2_eq_apply]
  rfl

/-! ## Vectors -/

/-- `toVec` of a rank-one tensor reads the corresponding scalar entry. -/
theorem _root_.TorchLean.Tensor.toVec_vec {k : Nat} (v : Tensor ℝ [k])
    (p : Fin (Shape.size (Shape.dim k .scalar))) :
    toVec (t := v) p = getScalar v (Fin.cast (Shape.size_vector k) p) := by
  have hp : p = finProdFinEquiv (Fin.cast (Shape.size_vector k) p,
      (⟨0, by simp⟩ : Fin (Shape.size Shape.scalar))) := by
    apply Fin.ext
    rw [finProdFinEquiv_apply_val, Fin.val_cast]
    -- simp leaves out `size_cons`. It would rewrite the size in the type of `Fin.cast` before
    -- the cast lemma matches.
    simp [-Shape.size_cons]
  conv_lhs => rw [← Tensor.dim_unstack v, hp]
  refine (toVec_dim_apply _ _).trans ?_
  rw [← Tensor.scalar_item (Tensor.unstack v _), toVec_scalar]
  rfl

/-! ## Pointwise operations on matrices and vectors -/

/-- Entries of a pointwise matrix product. -/
theorem _root_.TorchLean.Tensor.get2_mul (a b : Tensor ℝ [m, n]) (i : Fin m) (j : Fin n) :
    Tensor.get2 (a * b) i j = Tensor.get2 a i j * Tensor.get2 b i j := by
  simp [mul_def]

/-- Entries of a pointwise matrix quotient. -/
theorem _root_.TorchLean.Tensor.get2_div (a b : Tensor ℝ [m, n]) (i : Fin m) (j : Fin n) :
    Tensor.get2 (a / b) i j = Tensor.get2 a i j / Tensor.get2 b i j := by
  simp [div_def]

/-- Entries of a vector sum. -/
theorem _root_.TorchLean.Tensor.getScalar_add_apply {k : Nat} (a b : Tensor ℝ [k]) (i : Fin k) :
    getScalar (a + b) i = getScalar a i + getScalar b i := by
  simp [add_def]

/-- Entries of a pointwise vector quotient. -/
theorem _root_.TorchLean.Tensor.getScalar_div {k : Nat} (a b : Tensor ℝ [k]) (i : Fin k) :
    getScalar (a / b) i = getScalar a i / getScalar b i := by
  simp [div_def]

/-- Entries of a pointwise vector maximum. -/
theorem _root_.TorchLean.Tensor.getScalar_maximum {k : Nat} (a b : Tensor ℝ [k]) (i : Fin k) :
    getScalar (Tensor.maximum a b) i = max (getScalar a i) (getScalar b i) := by
  simp [Tensor.maximum, Context.maximum_real]

/-- Entries of the clamped square root. -/
theorem _root_.TorchLean.Tensor.getScalar_sqrt_clampNegative {k : Nat} (a : Tensor ℝ [k])
    (i : Fin k) :
    getScalar (Tensor.sqrt a .clampNegative) i = Real.sqrt (max (getScalar a i) 0) := by
  simp [Tensor.sqrt]
  rfl

/-- Entries of the clamped square root of a matrix. -/
theorem get2_sqrt_clampNegative (a : Tensor ℝ [m, n]) (i : Fin m) (j : Fin n) :
    Tensor.get2 (Tensor.sqrt a .clampNegative) i j = Real.sqrt (max (Tensor.get2 a i j) 0) := by
  simp [Tensor.sqrt]
  rfl

/-! ## Layer normalization of a matrix entrywise -/

/-- Column-broadcast evidence for the `gamma` and `beta` of a matrix layer normalization. -/
theorem canBroadcastTo_col : Shape.CanBroadcastTo (.dim n .scalar) (.dim m (.dim n .scalar)) :=
  Shape.CanBroadcastTo.expand_dims (Shape.CanBroadcastTo.dim_eq Shape.CanBroadcastTo.scalar)

/-- Row means of the LayerNorm input. -/
def lnMean (x : Tensor ℝ [m, n]) : Tensor ℝ [m] :=
  Tensor.map (Context.divNat · n) (rowSum x)

/-- Centered LayerNorm input. -/
def lnCentered (x : Tensor ℝ [m, n]) : Tensor ℝ [m, n] :=
  x - rowBroadcast (lnMean x)

/-- Clamped LayerNorm standard deviation `sqrt (max (max var 0 + ε) 0)`, where `var` is the mean
square of the centered input after centering it a second time. -/
def lnStd (x : Tensor ℝ [m, n]) (ε : ℝ) : Tensor ℝ [m] :=
  let deviation := lnCentered x - rowBroadcast (lnMean (lnCentered x))
  Tensor.sqrt ((Tensor.maximum (lnMean (deviation * deviation)) (Tensor.full (.dim m .scalar) 0)) +
    Tensor.full (.dim m .scalar) ε) .clampNegative

/-- Entries of the LayerNorm row means. -/
theorem getScalar_lnMean (x : Tensor ℝ [m, n]) (i : Fin m) :
    getScalar (lnMean x) i = 𝔼 j, Tensor.get2 x i j := by
  rw [lnMean, getScalar_rowMean, Fintype.expect_eq_sum_div_card, Fintype.card_fin]

/-- Entries of the centered LayerNorm input. -/
theorem get2_lnCentered (x : Tensor ℝ [m, n]) (i : Fin m) (j : Fin n) :
    Tensor.get2 (lnCentered x) i j = Tensor.get2 x i j - 𝔼 k, Tensor.get2 x i k := by
  rw [lnCentered, get2_sub, get2_rowBroadcast, getScalar_lnMean]

/-- Entry `i` of `lnStd x ε` is `sqrt (max (TorchLean.var (get2 x i) + ε) 0)`: the second centering
changes nothing and the clamp `max var 0` is inactive. -/
theorem getScalar_lnStd (x : Tensor ℝ [m, n]) (ε : ℝ) (i : Fin m) :
    getScalar (lnStd x ε) i = Real.sqrt (max (TorchLean.var (Tensor.get2 x i) + ε) 0) := by
  have h0 : 𝔼 k, (Tensor.get2 x i k - 𝔼 j, Tensor.get2 x i j) = 0 :=
    Fintype.expect_balance (Tensor.get2 x i)
  rw [lnStd, getScalar_sqrt_clampNegative, getScalar_add_apply, getScalar_maximum,
    getScalar_full, getScalar_full, getScalar_lnMean]
  simp only [get2_mul, get2_sub, get2_rowBroadcast, getScalar_lnMean, get2_lnCentered, h0,
    sub_zero]
  change Real.sqrt (max (max (TorchLean.var (Tensor.get2 x i)) 0 + ε) 0) = _
  rw [max_eq_left (var_nonneg (Tensor.get2 x i))]

/-! ## Dividing each row by its scale -/

/-- Dividing a row by `σ ≥ 0` and `ε` by `σ * σ` divides the clamped root of the row's mean square
plus `ε` by `σ`. -/
theorem sqrt_div_scale_eq {N : Nat} (c : Fin N → ℝ) {σ : ℝ} (hσ : 0 ≤ σ) (ε : ℝ) :
    Real.sqrt (max ((∑ k, c k / σ * (c k / σ)) / N + ε / σ / σ) 0) =
      Real.sqrt (max ((∑ k, c k * c k) / N + ε) 0) / σ := by
  have hsum : (∑ k, c k / σ * (c k / σ)) / N + ε / σ / σ =
      ((∑ k, c k * c k) / N + ε) / (σ * σ) := by
    rw [show (∑ k, c k / σ * (c k / σ)) = (∑ k, c k * c k) / (σ * σ) by
      rw [Finset.sum_div]
      exact Finset.sum_congr rfl fun k _ => by ring]
    ring
  have hmax : max (((∑ k, c k * c k) / N + ε) / (σ * σ)) 0 =
      max ((∑ k, c k * c k) / N + ε) 0 / (σ * σ) := by
    rw [← max_div_div_right (mul_self_nonneg σ), zero_div]
  rw [hsum, hmax, Real.sqrt_div' _ (mul_self_nonneg σ), Real.sqrt_mul_self hσ]

/-- If `σ` bounds every `|c k|` and `sqrt |ε|`, the scale cancels: `c j / σ` over the clamped root
of the scaled mean square plus `ε / σ / σ` is `c j` over the clamped root of the mean square plus
`ε`. -/
theorem div_scale_div_sqrt {N : Nat} (c : Fin N → ℝ) {σ : ℝ} (ε : ℝ) (hc : ∀ k, |c k| ≤ σ)
    (hε : Real.sqrt |ε| ≤ σ) (j : Fin N) :
    c j / σ / Real.sqrt (max ((∑ k, c k / σ * (c k / σ)) / N + ε / σ / σ) 0) =
      c j / Real.sqrt (max ((∑ k, c k * c k) / N + ε) 0) := by
  have hσ : 0 ≤ σ := (Real.sqrt_nonneg _).trans hε
  rw [sqrt_div_scale_eq c hσ ε]
  rcases hσ.eq_or_lt with h0 | hpos
  · have hcj : c j = 0 := abs_nonpos_iff.mp (by rw [h0]; exact hc j)
    rw [← h0, hcj]
    simp
  · exact div_div_div_cancel_right₀ hpos.ne' _ _

/-- If `σ` bounds every `|c k|` and `sqrt |ε|`, the scale cancels in the inverse standard deviation:
one over the clamped root of the scaled mean square plus `ε / σ / σ`, divided by `σ`, is one over
the clamped root of the mean square plus `ε`. -/
theorem one_div_sqrt_div_scale {N : Nat} (c : Fin N → ℝ) {σ : ℝ} (ε : ℝ) (hc : ∀ k, |c k| ≤ σ)
    (hε : Real.sqrt |ε| ≤ σ) :
    1 / Real.sqrt (max ((∑ k, c k / σ * (c k / σ)) / N + ε / σ / σ) 0) / σ =
      1 / Real.sqrt (max ((∑ k, c k * c k) / N + ε) 0) := by
  have hσ : 0 ≤ σ := (Real.sqrt_nonneg _).trans hε
  rw [sqrt_div_scale_eq c hσ ε]
  rcases hσ.eq_or_lt with h0 | hpos
  · have hck : ∀ k, c k = 0 := fun k => abs_nonpos_iff.mp (by rw [h0]; exact hc k)
    have hε0 : ε = 0 := by
      have h : Real.sqrt |ε| = 0 := le_antisymm (by rw [h0]; exact hε) (Real.sqrt_nonneg _)
      rwa [Real.sqrt_eq_zero (abs_nonneg ε), abs_eq_zero] at h
    rw [← h0]
    simp [hck, hε0]
  · rw [one_div_div, div_div_cancel_left' hpos.ne', one_div]

/-- `normalizedRows` computed with each row's scale taken from `scale`: the normalized rows,
and the inverse standard deviations divided by `scale`. -/
def rowsGiven (x : Tensor ℝ [m, n]) (scale : Tensor ℝ [m]) (ε : ℝ) (recenter : Bool) :
    Tensor ℝ [m, n] × Tensor ℝ [m] :=
  let scaled := lnCentered x / rowBroadcast scale
  let deviation := if recenter then scaled - rowBroadcast (lnMean scaled) else scaled
  let std := Tensor.sqrt (lnMean (deviation * deviation) +
    ((Tensor.full [m] ε / scale) / scale)) .clampNegative
  (scaled / rowBroadcast std, (Tensor.full [m] 1 / std) / scale)

/-- The scale `Tensor.normalizationScale` of each centered row of `x`. -/
def rowScale (x : Tensor ℝ [m, n]) (ε : ℝ) : Tensor ℝ [m] :=
  castShape (Tensor.normalizationScale (lnCentered x) [⟨1, Nat.one_lt_two⟩]
      (castShape (Tensor.full [m] ε) (Shape.reduce_singleton [m, n] ⟨1, Nat.one_lt_two⟩).symm))
    (Shape.reduce_singleton [m, n] ⟨1, Nat.one_lt_two⟩)

/-- `normalizedRows` is `rowsGiven` at each row's own scale `rowScale`. -/
theorem normalizedRows_eq_rowsGiven (x : Tensor ℝ [m, n]) (ε : ℝ) (recenter : Bool) :
    normalizedRows x ε recenter = rowsGiven x (rowScale x ε) ε recenter := rfl

/-- The recentered rows of `rowsGiven`, with the deviation spelled out. -/
theorem rowsGiven_fst_true (x : Tensor ℝ [m, n]) (scale : Tensor ℝ [m]) (ε : ℝ) :
    (rowsGiven x scale ε true).1 =
      let scaled := lnCentered x / rowBroadcast scale
      let deviation := scaled - rowBroadcast (lnMean scaled)
      scaled / rowBroadcast (Tensor.sqrt (lnMean (deviation * deviation) +
        ((Tensor.full [m] ε / scale) / scale)) .clampNegative) := rfl

/-- The rows of the centered input divided by a scale have mean zero. -/
theorem rowMeanE_lnCentered_div (x : Tensor ℝ [m, n]) (scale : Tensor ℝ [m]) (i : Fin m) :
    𝔼 j, Tensor.get2 (lnCentered x / rowBroadcast scale) i j = 0 := by
  have h0 : 𝔼 j, (Tensor.get2 x i j - 𝔼 k, Tensor.get2 x i k) = 0 :=
    Fintype.expect_balance (Tensor.get2 x i)
  simp only [get2_div, get2_rowBroadcast, get2_lnCentered]
  rw [← Finset.expect_div, h0, zero_div]

/-- Recentering the scaled rows a second time changes no entry. -/
theorem get2_rowsGiven_deviation (x : Tensor ℝ [m, n]) (scale : Tensor ℝ [m])
    (recenter : Bool) (i : Fin m) (j : Fin n) :
    Tensor.get2 (if recenter then lnCentered x / rowBroadcast scale -
          rowBroadcast (lnMean (lnCentered x / rowBroadcast scale))
        else lnCentered x / rowBroadcast scale) i j =
      (Tensor.get2 x i j - 𝔼 k, Tensor.get2 x i k) / getScalar scale i := by
  cases recenter
  · rw [ite_eq_right Bool.false_ne_true, get2_div, get2_rowBroadcast, get2_lnCentered]
  · rw [ite_eq_left rfl, get2_sub, get2_rowBroadcast, getScalar_lnMean, rowMeanE_lnCentered_div,
      sub_zero, get2_div, get2_rowBroadcast, get2_lnCentered]

/-- If row `i` of `x` reads `f`, entry `i` of `lnMean (x * x)` is the mean of `f j * f j`. -/
theorem getScalar_lnMean_mul_self (x : Tensor ℝ [m, n]) {i : Fin m} {f : Fin n → ℝ}
    (hf : ∀ j, Tensor.get2 x i j = f j) :
    getScalar (lnMean (x * x)) i = (∑ j, f j * f j) / n := by
  rw [lnMean, getScalar_rowMean]
  simp only [get2_mul, hf]

/-- Entries of the rows of `rowsGiven` when the scale bounds the centered row and `sqrt |ε|`: the
scale cancels. -/
theorem get2_rowsGiven_fst (x : Tensor ℝ [m, n]) (scale : Tensor ℝ [m]) (ε : ℝ)
    (recenter : Bool) (i : Fin m)
    (hc : ∀ j, |Tensor.get2 x i j - 𝔼 k, Tensor.get2 x i k| ≤ getScalar scale i)
    (hε : Real.sqrt |ε| ≤ getScalar scale i) (j : Fin n) :
    Tensor.get2 (rowsGiven x scale ε recenter).1 i j =
      (Tensor.get2 x i j - 𝔼 k, Tensor.get2 x i k) /
        Real.sqrt (max (TorchLean.var (Tensor.get2 x i) + ε) 0) := by
  simp only [rowsGiven]
  rw [get2_div, get2_div, get2_rowBroadcast, get2_rowBroadcast, get2_lnCentered,
    getScalar_sqrt_clampNegative, getScalar_add_apply, getScalar_div, getScalar_div, getScalar_full,
    getScalar_lnMean_mul_self _ (get2_rowsGiven_deviation x scale recenter i), TorchLean.var,
    Fintype.expect_eq_sum_div_card fun k =>
      (Tensor.get2 x i k - 𝔼 l, Tensor.get2 x i l) * (Tensor.get2 x i k - 𝔼 l, Tensor.get2 x i l),
    Fintype.card_fin]
  exact div_scale_div_sqrt (fun k => Tensor.get2 x i k - 𝔼 l, Tensor.get2 x i l) ε hc hε j

/-- Inverse standard deviations of `rowsGiven` when the scale bounds the centered row and
`sqrt |ε|`: the scale cancels. -/
theorem getScalar_rowsGiven_snd (x : Tensor ℝ [m, n]) (scale : Tensor ℝ [m])
    (ε : ℝ) (recenter : Bool) (i : Fin m)
    (hc : ∀ j, |Tensor.get2 x i j - 𝔼 k, Tensor.get2 x i k| ≤ getScalar scale i)
    (hε : Real.sqrt |ε| ≤ getScalar scale i) :
    getScalar (rowsGiven x scale ε recenter).2 i =
      1 / Real.sqrt (max (TorchLean.var (Tensor.get2 x i) + ε) 0) := by
  simp only [rowsGiven]
  rw [getScalar_div, getScalar_div, getScalar_full, getScalar_sqrt_clampNegative,
    getScalar_add_apply, getScalar_div, getScalar_div, getScalar_full,
    getScalar_lnMean_mul_self _ (get2_rowsGiven_deviation x scale recenter i), TorchLean.var,
    Fintype.expect_eq_sum_div_card fun k =>
      (Tensor.get2 x i k - 𝔼 l, Tensor.get2 x i l) * (Tensor.get2 x i k - 𝔼 l, Tensor.get2 x i l),
    Fintype.card_fin]
  exact one_div_sqrt_div_scale (fun k => Tensor.get2 x i k - 𝔼 l, Tensor.get2 x i l) ε hc hε

/-- The reduction over the last axis of `[m, n]` at an output coordinate of row `i` reads the entry
`(i, j)` at flat index `j`. -/
theorem reduceSource_row (o : (Shape.reduce [m, n] [⟨1, Nat.one_lt_two⟩] false).Coord)
    {i : Fin m} (ho : Shape.Coord.index o 0 = i.val) (j : Fin n)
    (hj : j.val < Shape.size (Shape.selectAxes [m, n] [⟨1, Nat.one_lt_two⟩])) :
    Shape.reduceSource [m, n] [⟨1, Nat.one_lt_two⟩] false o
        (Shape.Coord.unlinearize ⟨j.val, hj⟩) = ((i, j, ()) : Shape.Coord [m, n]) := by
  apply Shape.Coord.ext_index
  intro k hk
  rw [Shape.index_reduceSource, dite_eq_left hk]
  match k, hk with
  | 0, hk =>
    have hne : (⟨0, hk⟩ : Fin (Shape.rank [m, n])) ≠ ⟨1, Nat.one_lt_two⟩ :=
      Fin.ne_of_val_ne Nat.zero_ne_one
    have hlt : (⟨0, hk⟩ : Fin (Shape.rank [m, n])) < ⟨1, Nat.one_lt_two⟩ := Nat.zero_lt_one
    rw [ite_eq_right (mt List.mem_singleton.mp hne), Shape.keptIndex, Bool.cond_false,
      Shape.idxOf_otherAxes_singleton _ hne, ite_eq_left hlt, ho]
    rfl
  | 1, hk =>
    rw [ite_eq_left (List.mem_singleton_self _), Shape.sortedAxes_singleton, List.idxOf_cons_self,
      index_unlinearize_congr (Shape.selectAxes_singleton [m, n] _)]
    exact val_unlinearize_vector _
  | k + 2, hk => exact absurd hk (by show ¬ (k + 2 < 2); omega)

/-- The scale of each centered row of `x` bounds the row's magnitudes and `sqrt |ε|`. -/
theorem le_getScalar_rowScale (hn : 0 < n) (x : Tensor ℝ [m, n]) (ε : ℝ) (i : Fin m) :
    (∀ j, |Tensor.get2 x i j - 𝔼 k, Tensor.get2 x i k| ≤ getScalar (rowScale x ε) i) ∧
      Real.sqrt |ε| ≤ getScalar (rowScale x ε) i := by
  have hpos : ∀ d ∈ [(⟨1, Nat.one_lt_two⟩ : Fin (Shape.rank [m, n]))],
      0 < Shape.extent [m, n] d := by
    intro d hd
    rw [List.mem_singleton.mp hd]
    exact hn
  have habs : ∀ {s : Shape} (t : Tensor ℝ s) (c : s.Coord), Tensor.abs t c = |t c| :=
    fun t c => (Tensor.map_apply MathFunctions.abs t c).trans (LawfulContext.abs_eq (t c))
  have hstop : ∀ v : ℝ, Context.stopGradient v = v := fun _ => rfl
  have hsize : n = Shape.size (Shape.selectAxes [m, n] [⟨1, Nat.one_lt_two⟩]) := by
    rw [Shape.selectAxes_singleton, Shape.size_eq_prod, List.prod_singleton]
    rfl
  obtain ⟨o, ho, hval⟩ : ∃ o : (Shape.reduce [m, n] [⟨1, Nat.one_lt_two⟩] false).Coord,
      Shape.Coord.index o 0 = i.val ∧
        getScalar (rowScale x ε) i = Tensor.normalizationScale (lnCentered x)
          [⟨1, Nat.one_lt_two⟩] (castShape (Tensor.full [m] ε)
            (Shape.reduce_singleton [m, n] ⟨1, Nat.one_lt_two⟩).symm) o :=
    ⟨_, (Shape.Coord.index_cast (Shape.reduce_singleton [m, n] ⟨1, Nat.one_lt_two⟩).symm
        ((i, ()) : Shape.Coord [m]) 0).trans rfl, by
      rw [Tensor.getScalar_eq_apply]
      exact castShape_apply _ _ _⟩
  have hreg : Tensor.sqrt (Tensor.abs (castShape (Tensor.full [m] ε)
      (Shape.reduce_singleton [m, n] ⟨1, Nat.one_lt_two⟩).symm)) .clampNegative o =
      Real.sqrt |ε| := by
    rw [sqrt_clampNegative_apply, habs]
    erw [castShape_apply]
    rw [full_apply]
    exact congrArg Real.sqrt (max_eq_left (abs_nonneg ε))
  rw [hval]
  simp only [Tensor.normalizationScale]
  rw [detach_apply, hstop, maximum_apply, dite_eq_left hpos, reduce_max_apply]
  simp only [Context.maximum_real]
  refine ⟨fun j => le_max_of_le_left ?_, le_max_of_le_right hreg.ge⟩
  have hj : j.val < Shape.size (Shape.selectAxes [m, n] [⟨1, Nat.one_lt_two⟩]) :=
    hsize ▸ j.isLt
  calc |Tensor.get2 x i j - 𝔼 k, Tensor.get2 x i k|
      = Tensor.abs (lnCentered x) (Shape.reduceSource [m, n] [⟨1, Nat.one_lt_two⟩] false o
          (Shape.Coord.unlinearize ⟨j.val, hj⟩)) := by
        rw [reduceSource_row o ho j hj, habs, ← Tensor.get2_eq_apply (lnCentered x) i j,
          get2_lnCentered]
    _ ≤ _ := List.le_foldl_max_of_mem _
        (fun k => Tensor.abs (lnCentered x) (Shape.reduceSource [m, n] [⟨1, Nat.one_lt_two⟩]
          false o (Shape.Coord.unlinearize k))) (List.mem_finRange _)

/-- Layer normalization over the last axis of a matrix is `rowsGiven` at the scale `rowScale`,
scaled and shifted by the column parameters. -/
theorem layerNorm_eq_rowsGiven (hn : 0 < n) (ε : ℝ) (x : Tensor ℝ [m, n])
    (gamma beta : Tensor ℝ [n]) :
    Tensor.layerNorm x [⟨1, Nat.one_lt_two⟩] gamma beta ε =
      ((rowsGiven x (rowScale x ε) ε true).1 *
          Tensor.broadcast gamma _ (canBroadcastTo_col (m := m) (n := n)).dims
            (canBroadcastTo_col (m := m) (n := n)).broadcastDims) +
        Tensor.broadcast beta _ (canBroadcastTo_col (m := m) (n := n)).dims
          (canBroadcastTo_col (m := m) (n := n)).broadcastDims := by
  have hcount :
      ((if Shape.size (Shape.selectAxes [m, n] [⟨1, Nat.one_lt_two⟩]) = 0 then 1
        else Shape.size (Shape.selectAxes [m, n] [⟨1, Nat.one_lt_two⟩]) : Nat)) =
      n := by
    rw [Shape.selectAxes_singleton, Shape.size_eq_prod, List.prod_singleton]
    show (if n = 0 then 1 else n) = n
    rw [ite_eq_right (by omega)]
  have hmean : Tensor.map (Context.divNat · n) (rowSum x) =
      Tensor.map
        (Context.divNat · (if Shape.size (Shape.selectAxes [m, n]
          [⟨1, Nat.one_lt_two⟩]) = 0 then 1 else Shape.size (Shape.selectAxes [m, n]
            [⟨1, Nat.one_lt_two⟩])))
        (Tensor.reduce .sum x [⟨1, Nat.one_lt_two⟩] false) := by
    rw [hcount]
    rfl
  have hspread (t : Tensor ℝ [m]) :
      rowBroadcast (n := n) t =
        t.broadcast [m, n] (Shape.keptAxis [m, n] [⟨1, Nat.one_lt_two⟩])
          (Shape.broadcastDims_eraseAxes [m, n] [⟨1, Nat.one_lt_two⟩]) := by
    apply Tensor.Rep.ext
    rintro ⟨i, j, u⟩
    rw [rowBroadcast]
    erw [broadcast_eraseIdx_apply (s := [m, n]) 1]
  have hvar (c : Tensor ℝ [m, n]) :
      lnMean ((c - rowBroadcast (lnMean c)) * (c - rowBroadcast (lnMean c))) =
        castShape (var c [⟨1, Nat.one_lt_two⟩] false) (Shape.reduce_singleton _ _) := by
    apply Tensor.ext_vector
    intro i
    refine Eq.trans ?_ (getScalar_rowVar hn c i).symm
    rw [lnMean, getScalar_rowMean]
    simp only [get2_mul, get2_sub, get2_rowBroadcast, lnMean, getScalar_rowMean]
  have hscale (h5 : Shape.CanBroadcastTo [n] [m, n]) (g : Tensor ℝ [n]) :
      Tensor.broadcast g _ h5.dims h5.broadcastDims =
        g.broadcast [m, n] (Shape.listedAxis [m, n] [⟨1, Nat.one_lt_two⟩])
          (Shape.broadcastDims_selectAxes [m, n] [⟨1, Nat.one_lt_two⟩]) := by
    have key : ∀ (D : Fin (Shape.rank [n]) → Fin (Shape.rank [m, n]))
        (P : Shape.BroadcastDims [n] [m, n] D),
        D = Shape.listedAxis [m, n] [⟨1, Nat.one_lt_two⟩] →
        Tensor.broadcast g _ D P =
          g.broadcast [m, n] (Shape.listedAxis [m, n] [⟨1, Nat.one_lt_two⟩])
            (Shape.broadcastDims_selectAxes [m, n] [⟨1, Nat.one_lt_two⟩]) := by
      rintro D P rfl
      rfl
    refine key _ _ (funext fun d => ?_)
    fin_cases d
    rfl
  symm
  rw [rowsGiven_fst_true]
  simp only [rowScale, lnCentered]
  rw [hvar]
  simp only [lnMean]
  rw [hmean]
  erw [hspread, hspread, hspread]
  rw [hscale, hscale]
  rfl

/-- Entries of layer normalization over the last axis of a matrix: each row is centered, divided
by the clamped standard deviation of that row, then scaled and shifted by the per-column
parameters. -/
theorem get2_layerNorm (hn : 0 < n) (ε : ℝ) (x : Tensor ℝ [m, n])
    (gamma beta : Tensor ℝ [n]) (i : Fin m) (j : Fin n) :
    Tensor.get2 (Tensor.layerNorm x [⟨1, Nat.one_lt_two⟩] gamma beta ε) i j =
      (Tensor.get2 x i j - 𝔼 k, Tensor.get2 x i k) /
          Real.sqrt (max (TorchLean.var (Tensor.get2 x i) + ε) 0) * getScalar gamma j +
        getScalar beta j := by
  obtain ⟨hc, hε⟩ := le_getScalar_rowScale hn x ε i
  rw [layerNorm_eq_rowsGiven hn, get2_add, get2_mul, get2_rowsGiven_fst x _ ε true i hc hε,
    get2_broadcast_col, get2_broadcast_col]

/-- Over the reals the rows normalized by the spec are the centered rows divided by `lnStd`. -/
theorem normalizedRows_fst (hn : 0 < n) (x : Tensor ℝ [m, n]) (ε : ℝ) (recenter : Bool) :
    (normalizedRows x ε recenter).1 = lnCentered x / rowBroadcast (lnStd x ε) :=
  Tensor.ext_get2 fun i j => by
    obtain ⟨hc, hε⟩ := le_getScalar_rowScale hn x ε i
    rw [normalizedRows_eq_rowsGiven, get2_rowsGiven_fst x _ ε recenter i hc hε, get2_div,
      get2_rowBroadcast, get2_lnCentered, getScalar_lnStd]

/-- Over the reals the inverse standard deviations of the spec rows are `1 / lnStd`. -/
theorem normalizedRows_snd (hn : 0 < n) (x : Tensor ℝ [m, n]) (ε : ℝ) (recenter : Bool) :
    (normalizedRows x ε recenter).2 = Tensor.full [m] 1 / lnStd x ε :=
  Tensor.ext_vector fun i => by
    obtain ⟨hc, hε⟩ := le_getScalar_rowScale hn x ε i
    rw [normalizedRows_eq_rowsGiven, getScalar_rowsGiven_snd x _ ε recenter i hc hε,
      getScalar_div, getScalar_full, getScalar_lnStd]

end

end Normalization
end TorchLean

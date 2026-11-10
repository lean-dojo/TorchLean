/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import Mathlib.Algebra.BigOperators.Field
public import Mathlib.Analysis.InnerProductSpace.PiL2
public import Mathlib.Analysis.SpecialFunctions.Exp
public import TorchLean.Tensor.Numeric.Real
public import TorchLean.Tensor.Statistics.Real
import FloatLib.Numerics.MixedError.Elementary
import Mathlib.Tactic.Positivity.Finset

/-!
# Rounding error of a transformer block

`block` is a pre-normalization transformer block over the reals: layer normalization, attention of
every head over the keys `keys t` of each token `t`, an output projection and a residual sum, then
a second layer normalization, a two-layer perceptron with activation `act` and a second residual
sum. `Arithmetic.block` evaluates the same block with the functions of an `Arithmetic`: a rounding
after each scalar operation and after each product inside a sum, a rounding of each stored stage
output, a computed sum of each list, and computed exponential and activation.
`Arithmetic.Contract` bounds each of them by a mixed error `|f t - t| ≤ u * |t| + η`, so `u = 0`
gives absolute bounds and `η = 0` relative ones, and charges a sum of `n` terms `g n` times the
sum of their magnitudes plus `h n`. For a left fold of rounded additions,
`abs_foldl_round_add_sub_sum_le` gives `g n = (1 + u) ^ n - 1` and `h n = n * (1 + u) ^ n * η`.

`Arithmetic.sqrt_sum_sq_block_sub_le` bounds the Euclidean distance between each token of the
computed block and of the exact block. The model width `ι` (nonempty), the heads `ζ`, the head
width `ξ` and the hidden width `κ` are arbitrary finite types, and the tokens form an arbitrary
type in which each token `t` attends over a nonempty finite set `keys t`. The hypotheses are
Euclidean bounds on the exact stages, spectral and Frobenius bounds on the weights (`LinearBound`),
and one inequality per stage whose right side is the bound of its stage lemma at the errors of the
stages before it. Those lemmas are `Arithmetic.sqrt_sum_sq_linear_sub_le'` for affine maps,
`Arithmetic.sqrt_sum_sq_layerNorm_sub_le'` for layer normalization (`LayerNormCertificate`
collects its internal errors), `Arithmetic.sqrt_sum_sq_attention_sub_le'` for one head,
`sqrt_sum_sq_store_act_sub_le` for the activation and `sqrt_sum_sq_store_add_sub_le` for the
residual sums. They rest on two bounds of the exact operations. Layer normalization moves by at
most `2 * G / √(var x + ε)` times the change of its input
(`sqrt_sum_sq_layerNorm_sub_le`). Scores that move by at most `t` move the softmax by at
most `2 * (exp t - 1) / (exp t + 1) = 2 * tanh (t / 2)` in `ℓ¹` (`sum_abs_softmax_sub_le`), so an
attention output moves by at most that times the value bound plus the change of the values
(`sqrt_sum_sq_attention_sub_le`).

`TorchLean.Verify.RuntimeApprox.TransformerBlock.Ops` proves that `layerNorm`, `softmax`,
`attention`, `heads` and `block` agree at the reals, entry by entry, with forwards of `Op`. Only
these exact functions are tied to the op table. `Arithmetic.block` rounds its own schedule:
`Arithmetic.layerNorm` takes the variance of the unscaled centered entries and `Arithmetic.linear`
starts its sum from the bias, while `Tensor.layerNorm` rescales by `Tensor.normalizationScale` and
`Linear.forward` adds the bias last. The bound of `Arithmetic.sqrt_sum_sq_block_sub_le` therefore
does not cover the executed op-table block.
-/

@[expose] public section

open Finset FloatLib.Numerics
open scoped BigOperators

namespace TorchLean.TransformerBlock

open TorchLean (var var_nonneg)

variable {ι κ : Type*} [Fintype ι]

/-! ### Euclidean norms of finite vectors -/

/-- If `|x i| ≤ p i + q i + r i + s i` at every `i`, the Euclidean norm of `x` is at most the sum
of the Euclidean norms of `p`, `q`, `r` and `s`. -/
theorem sqrt_sum_sq_le_add_of_abs_le {x p q r s : ι → ℝ}
    (h : ∀ i, |x i| ≤ p i + q i + r i + s i) :
    √(∑ i, x i ^ 2) ≤
      √(∑ i, p i ^ 2) + √(∑ i, q i ^ 2) + √(∑ i, r i ^ 2) + √(∑ i, s i ^ 2) := by
  have htri : ∀ f g : ι → ℝ, √(∑ i, (f i + g i) ^ 2) ≤ √(∑ i, f i ^ 2) + √(∑ i, g i ^ 2) :=
    fun f g => by
      simpa [EuclideanSpace.norm_eq] using norm_add_le (WithLp.toLp 2 f) (WithLp.toLp 2 g)
  calc √(∑ i, x i ^ 2) ≤ √(∑ i, (p i + q i + r i + s i) ^ 2) :=
        Real.sqrt_le_sqrt (Finset.sum_le_sum fun i _ => by
          simpa only [sq_abs] using pow_le_pow_left₀ (abs_nonneg (x i)) (h i) 2)
    _ ≤ √(∑ i, (p i + q i + r i) ^ 2) + √(∑ i, s i ^ 2) := htri (fun i => p i + q i + r i) s
    _ ≤ √(∑ i, (p i + q i) ^ 2) + √(∑ i, r i ^ 2) + √(∑ i, s i ^ 2) := by
      gcongr; exact htri (fun i => p i + q i) r
    _ ≤ _ := by gcongr; exact htri p q

/-- The magnitudes `∑ j, |x j| * |W j c|` have Euclidean norm at most `‖x‖` times the
Frobenius norm of `W`. -/
theorem sqrt_sum_sq_sum_abs_mul_le [Fintype κ] (x : ι → ℝ) (W : ι → κ → ℝ) :
    √(∑ c, (∑ j, |x j| * |W j c|) ^ 2) ≤ √(∑ j, ∑ c, W j c ^ 2) * √(∑ j, x j ^ 2) := by
  have hc : ∀ c, |(∑ j, |x j| * |W j c|)| ≤ √(∑ j, x j ^ 2) * √(∑ j, W j c ^ 2) := fun c => by
    rw [abs_of_nonneg (Finset.sum_nonneg fun j _ => by positivity)]
    simpa [sq_abs] using Real.sum_mul_le_sqrt_mul_sqrt Finset.univ (|x ·|) fun j => |W j c|
  calc √(∑ c, (∑ j, |x j| * |W j c|) ^ 2)
      ≤ √(∑ c, (√(∑ j, x j ^ 2) * √(∑ j, W j c ^ 2)) ^ 2) :=
        Real.sqrt_le_sqrt (Finset.sum_le_sum fun c _ => by
          simpa only [sq_abs] using pow_le_pow_left₀ (abs_nonneg _) (hc c) 2)
    _ = √(∑ j, x j ^ 2) * √(∑ c, ∑ j, W j c ^ 2) := by
      simp_rw [mul_pow, ← Finset.mul_sum]
      rw [Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq (Real.sqrt_nonneg _)]
      congr 2
      exact Finset.sum_congr rfl fun c _ =>
        Real.sq_sqrt (Finset.sum_nonneg fun j _ => by positivity)
    _ = √(∑ j, ∑ c, W j c ^ 2) * √(∑ j, x j ^ 2) := by rw [Finset.sum_comm, mul_comm]

/-! ### Arithmetic of one evaluation -/

/-- The rounding and kernel functions of one evaluation: `round` after each scalar operation,
`roundMul` after each product inside a sum, `store` when a stage output is written, `sum` for
each reduction, `exp` for the softmax exponentials and `act` for the activation. -/
structure Arithmetic where
  /-- Rounding of a scalar operation in the working format. -/
  round : ℝ → ℝ
  /-- Rounding of a product inside a sum. -/
  roundMul : ℝ → ℝ
  /-- Rounding of a stored stage output. -/
  store : ℝ → ℝ
  /-- Computed sum of a list in its order. -/
  sum : List ℝ → ℝ
  /-- Computed exponential. -/
  exp : ℝ → ℝ
  /-- Computed activation. -/
  act : ℝ → ℝ

/-- Exact real arithmetic with activation `act`. -/
noncomputable def Arithmetic.exact (act : ℝ → ℝ) : Arithmetic where
  round := id
  roundMul := id
  store := id
  sum := List.sum
  exp := Real.exp
  act := act

/-- A list summation `S` whose error on a list `l` is at most
`g l.length * (l.map abs).sum + h l.length` sums the values of `f` on `s`, listed by
`Finset.toList`, within `g s.card * ∑ i ∈ s, |f i| + h s.card`. -/
theorem abs_sum_toList_sub_le {α : Type*} {S : List ℝ → ℝ} {g h : ℕ → ℝ}
    (hS : ∀ l : List ℝ, |S l - l.sum| ≤ g l.length * (l.map abs).sum + h l.length)
    (s : Finset α) (f : α → ℝ) :
    |S (s.toList.map f) - ∑ i ∈ s, f i| ≤ g s.card * ∑ i ∈ s, |f i| + h s.card := by
  simpa [List.map_map, Finset.sum_map_toList, Function.comp_def] using hS (s.toList.map f)

/-- A left fold that rounds each partial sum, started from an approximation `b` of `a` with
error `e`, has the error of a sum of `n` terms in the standard model with growth `(1 + u) ^ n`. -/
theorem abs_foldl_round_add_sub_le {fl : ℝ → ℝ} {u η : ℝ} (hu : 0 ≤ u) (hη : 0 ≤ η)
    (hfl : ∀ t, |fl t - t| ≤ u * |t| + η) (l : List ℝ) {a b e : ℝ} (hab : |b - a| ≤ e) :
    |l.foldl (fun s t => fl (s + t)) b - (a + l.sum)| ≤
      (1 + u) ^ l.length * e + ((1 + u) ^ l.length - 1) * (|a| + (l.map abs).sum) +
        l.length * (1 + u) ^ l.length * η := by
  induction l generalizing a b e with
  | nil => simpa using hab
  | cons t l ih =>
    have hstep : |fl (b + t) - (a + t)| ≤ (1 + u) * e + u * (|a| + |t|) + η := by
      have h2 : |b + t - (a + t)| ≤ e := by simpa using hab
      have := ((MixedError.mk hu hη (hfl _)).trans_of_abs_sub_le h2).bound
      have h3 : u * |a + t| ≤ u * (|a| + |t|) := by gcongr; exact abs_add_le a t
      linarith
    have hih := ih hstep
    have hR : 0 ≤ (l.map abs).sum := List.sum_nonneg (by simp)
    have hP : 1 ≤ (1 + u) ^ l.length := one_le_pow₀ (by linarith)
    have hat : (((1 + u) ^ l.length - 1) * |a + t|) ≤ ((1 + u) ^ l.length - 1) * (|a| + |t|) :=
      mul_le_mul_of_nonneg_left (abs_add_le a t) (by linarith)
    have h1 : 0 ≤ (1 + u) ^ l.length * u * (l.map abs).sum := by positivity
    have h4 : 0 ≤ (1 + u) ^ l.length * u * η * (l.length + 1) := by positivity
    simp only [List.foldl_cons, List.sum_cons, List.length_cons, List.map_cons, pow_succ]
    rw [show a + (t + l.sum) = a + t + l.sum by ring]
    push_cast
    nlinarith

/-- The left fold that rounds each partial sum, started from zero, satisfies the summation
contract with growth `(1 + u) ^ n - 1` and absolute term `n * (1 + u) ^ n * η`. -/
theorem abs_foldl_round_add_sub_sum_le {fl : ℝ → ℝ} {u η : ℝ} (hu : 0 ≤ u) (hη : 0 ≤ η)
    (hfl : ∀ t, |fl t - t| ≤ u * |t| + η) (l : List ℝ) :
    |l.foldl (fun s t => fl (s + t)) 0 - l.sum| ≤
      ((1 + u) ^ l.length - 1) * (l.map abs).sum + l.length * (1 + u) ^ l.length * η := by
  simpa using abs_foldl_round_add_sub_le hu hη hfl l (a := 0) (b := 0) (e := 0) (by simp)

/-- The linear map `x ↦ b + x W` evaluated in `A`: each product is rounded by `roundMul`, the
sum starts from the bias and its result is stored. -/
noncomputable def Arithmetic.linear (A : Arithmetic) (W : ι → κ → ℝ) (b : κ → ℝ) (x : ι → ℝ) :
    κ → ℝ :=
  fun c => A.store (A.sum (b c :: (Finset.univ : Finset ι).toList.map
    fun j => A.roundMul (x j * W j c)))

/-- In exact arithmetic, entry `c` of the linear map is `b c + ∑ j, x j * W j c`. -/
theorem Arithmetic.exact_linear (act : ℝ → ℝ) (W : ι → κ → ℝ) (b : κ → ℝ) (x : ι → ℝ) (c : κ) :
    (Arithmetic.exact act).linear W b x c = b c + ∑ j, x j * W j c := by
  simp [Arithmetic.linear, Arithmetic.exact, Finset.sum_map_toList]

/-- Each entry of a linear map evaluated in `A` has the summation error of its products and bias,
the product roundings and the storage rounding. -/
theorem Arithmetic.abs_linear_sub_le (A : Arithmetic) {g h : ℕ → ℝ} {um ηm us ηs : ℝ}
    (hS : ∀ l : List ℝ, |A.sum l - l.sum| ≤ g l.length * (l.map abs).sum + h l.length)
    (hg : 0 ≤ g (Fintype.card ι + 1)) (hus : 0 ≤ us)
    (hmul : ∀ t, |A.roundMul t - t| ≤ um * |t| + ηm)
    (hstore : ∀ t, |A.store t - t| ≤ us * |t| + ηs)
    (W : ι → κ → ℝ) (b : κ → ℝ) (x : ι → ℝ) (c : κ) :
    |A.linear W b x c - (b c + ∑ j, x j * W j c)| ≤
      us * |b c + ∑ j, x j * W j c| + (1 + us) * (g (Fintype.card ι + 1) * |b c| +
        (g (Fintype.card ι + 1) * (1 + um) + um) * ∑ j, |x j| * |W j c| +
        (1 + g (Fintype.card ι + 1)) * Fintype.card ι * ηm + h (Fintype.card ι + 1)) + ηs := by
  set g' := g (Fintype.card ι + 1)
  set T := ∑ j, |x j| * |W j c|
  set p : ι → ℝ := fun j => A.roundMul (x j * W j c)
  have hsum := hS (b c :: (Finset.univ : Finset ι).toList.map p)
  simp only [List.sum_cons, List.length_cons, List.map_cons, List.map_map, List.length_map,
    Finset.length_toList, Finset.card_univ, Finset.sum_map_toList, Function.comp_def] at hsum

  have hp : ∀ j, |p j - x j * W j c| ≤ um * (|x j| * |W j c|) + ηm := fun j => by
    simpa [p, abs_mul] using hmul (x j * W j c)
  have hpabs : ∑ j, |p j| ≤ (1 + um) * T + Fintype.card ι * ηm := by
    have : ∀ j, |p j| ≤ (1 + um) * (|x j| * |W j c|) + ηm := fun j => by
      have := hp j
      have h' : |p j| ≤ |x j * W j c| + |p j - x j * W j c| := by
        simpa using abs_add_le (x j * W j c) (p j - x j * W j c)
      rw [abs_mul] at h'
      linarith
    calc ∑ j, |p j| ≤ ∑ j, ((1 + um) * (|x j| * |W j c|) + ηm) :=
          Finset.sum_le_sum fun j _ => this j
      _ = (1 + um) * T + Fintype.card ι * ηm := by
        rw [Finset.sum_add_distrib, ← Finset.mul_sum]; simp [T]
  have hpsum : |∑ j, p j - ∑ j, x j * W j c| ≤ um * T + Fintype.card ι * ηm := by
    rw [← Finset.sum_sub_distrib]
    calc |∑ j, (p j - x j * W j c)| ≤ ∑ j, |p j - x j * W j c| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ j, (um * (|x j| * |W j c|) + ηm) := Finset.sum_le_sum fun j _ => hp j
      _ = um * T + Fintype.card ι * ηm := by
        rw [Finset.sum_add_distrib, ← Finset.mul_sum]; simp [T]
  have hpre : |A.sum (b c :: (Finset.univ : Finset ι).toList.map p) - (b c + ∑ j, x j * W j c)| ≤
      g' * |b c| + (g' * (1 + um) + um) * T + (1 + g') * Fintype.card ι * ηm +
        h (Fintype.card ι + 1) := by
    have htri : |A.sum (b c :: (Finset.univ : Finset ι).toList.map p) - (b c + ∑ j, x j * W j c)|
        ≤ |A.sum (b c :: (Finset.univ : Finset ι).toList.map p) - (b c + ∑ j, p j)| +
          |∑ j, p j - ∑ j, x j * W j c| := by
      have := abs_add_le (A.sum (b c :: (Finset.univ : Finset ι).toList.map p) - (b c + ∑ j, p j))
        (∑ j, p j - ∑ j, x j * W j c)
      ring_nf at this ⊢
      linarith
    have hT : 0 ≤ T := Finset.sum_nonneg fun j _ => by positivity
    have := mul_le_mul_of_nonneg_left hpabs hg
    nlinarith
  have hηs : 0 ≤ ηs := by simpa using (abs_nonneg _).trans (hstore 0)
  exact ((MixedError.mk hus hηs (hstore _)).trans_of_abs_sub_le hpre).bound.trans_eq
    (add_assoc _ _ _).symm

/-- A linear map evaluated in `A` has Euclidean error at most the storage error of the exact
result plus the summation and product errors, bounded through the Frobenius norm `F` of `W`. -/
theorem Arithmetic.sqrt_sum_sq_linear_sub_le [Fintype κ] (A : Arithmetic) {g h : ℕ → ℝ}
    {um ηm us ηs F : ℝ}
    (hS : ∀ l : List ℝ, |A.sum l - l.sum| ≤ g l.length * (l.map abs).sum + h l.length)
    (hg : 0 ≤ g (Fintype.card ι + 1)) (hh : 0 ≤ h (Fintype.card ι + 1)) (hum : 0 ≤ um)
    (hηm : 0 ≤ ηm) (hus : 0 ≤ us) (hηs : 0 ≤ ηs)
    (hmul : ∀ t, |A.roundMul t - t| ≤ um * |t| + ηm)
    (hstore : ∀ t, |A.store t - t| ≤ us * |t| + ηs)
    (W : ι → κ → ℝ) (b : κ → ℝ) (x : ι → ℝ) (hF : √(∑ j, ∑ c, W j c ^ 2) ≤ F) :
    √(∑ c, (A.linear W b x c - (b c + ∑ j, x j * W j c)) ^ 2) ≤
      us * √(∑ c, (b c + ∑ j, x j * W j c) ^ 2) + (1 + us) * (g (Fintype.card ι + 1) *
        √(∑ c, b c ^ 2) + (g (Fintype.card ι + 1) * (1 + um) + um) * F * √(∑ j, x j ^ 2) +
        √(Fintype.card κ) * ((1 + g (Fintype.card ι + 1)) * Fintype.card ι * ηm +
          h (Fintype.card ι + 1))) + √(Fintype.card κ) * ηs := by
  set g' := g (Fintype.card ι + 1)
  set K := (1 + us) * ((1 + g') * Fintype.card ι * ηm + h (Fintype.card ι + 1)) + ηs
  have hK : 0 ≤ K := by positivity
  have hcoef : 0 ≤ (1 + us) * (g' * (1 + um) + um) := by positivity
  have hterm : ∀ c, |A.linear W b x c - (b c + ∑ j, x j * W j c)| ≤
      us * |b c + ∑ j, x j * W j c| + (1 + us) * g' * |b c| +
        (1 + us) * (g' * (1 + um) + um) * |(∑ j, |x j| * |W j c|)| + K := fun c => by
    have := A.abs_linear_sub_le hS hg hus hmul hstore W b x c
    have hT0 : |(∑ j, |x j| * |W j c|)| = ∑ j, |x j| * |W j c| :=
      abs_of_nonneg (Finset.sum_nonneg fun j _ => by positivity)
    rw [hT0]
    simp only [K]
    nlinarith
  have hT := sqrt_sum_sq_sum_abs_mul_le x W
  have hm : ∀ {a : ℝ}, 0 ≤ a → ∀ y : κ → ℝ, √(∑ i, (a * |y i|) ^ 2) = a * √(∑ i, y i ^ 2) :=
    fun {a} ha y => by
      simp_rw [mul_pow, sq_abs, ← Finset.mul_sum]
      rw [Real.sqrt_mul (sq_nonneg a), Real.sqrt_sq ha]
  calc _ ≤ √(∑ c, (us * |b c + ∑ j, x j * W j c|) ^ 2) + √(∑ c, ((1 + us) * g' * |b c|) ^ 2) +
        √(∑ c, ((1 + us) * (g' * (1 + um) + um) * |(∑ j, |x j| * |W j c|)|) ^ 2) +
        √(∑ _c : κ, K ^ 2) := sqrt_sum_sq_le_add_of_abs_le hterm
    _ = us * √(∑ c, (b c + ∑ j, x j * W j c) ^ 2) + (1 + us) * g' * √(∑ c, b c ^ 2) +
        (1 + us) * (g' * (1 + um) + um) * √(∑ c, (∑ j, |x j| * |W j c|) ^ 2) +
        √(Fintype.card κ) * K := by
      rw [hm hus, hm (by positivity), hm hcoef, Finset.sum_const, Finset.card_univ, nsmul_eq_mul,
        Real.sqrt_mul (Nat.cast_nonneg _), Real.sqrt_sq hK]
    _ ≤ us * √(∑ c, (b c + ∑ j, x j * W j c) ^ 2) + (1 + us) * g' * √(∑ c, b c ^ 2) +
        (1 + us) * (g' * (1 + um) + um) * (F * √(∑ j, x j ^ 2)) + √(Fintype.card κ) * K := by
      gcongr
      exact hT.trans (mul_le_mul_of_nonneg_right hF (Real.sqrt_nonneg _))
    _ = _ := by simp only [K]; ring

/-- The exact linear map moves by at most `S` times the input change when `S` bounds the
spectral norm of `W`. -/
theorem sqrt_sum_sq_linear_sub_le [Fintype κ] {W : ι → κ → ℝ} {S : ℝ}
    (hW : ∀ v : ι → ℝ, √(∑ c, (∑ j, v j * W j c) ^ 2) ≤ S * √(∑ j, v j ^ 2))
    (b : κ → ℝ) (x x' : ι → ℝ) :
    √(∑ c, ((b c + ∑ j, x' j * W j c) - (b c + ∑ j, x j * W j c)) ^ 2) ≤
      S * √(∑ j, (x' j - x j) ^ 2) := by
  have : ∀ c, (b c + ∑ j, x' j * W j c) - (b c + ∑ j, x j * W j c) =
      ∑ j, (x' j - x j) * W j c := fun c => by
    rw [add_sub_add_left_eq_sub, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun j _ => by ring
  simp only [this]
  exact hW _

/-- If `v ↦ fun c => ∑ j, v j * W j c` scales Euclidean norms by at most `S`, the Euclidean norm of
`b c + ∑ j, x j * W j c` is at most `√(∑ c, b c ^ 2) + S * √(∑ j, x j ^ 2)`. -/
theorem sqrt_sum_sq_linear_le [Fintype κ] {W : ι → κ → ℝ} {S : ℝ}
    (hW : ∀ v : ι → ℝ, √(∑ c, (∑ j, v j * W j c) ^ 2) ≤ S * √(∑ j, v j ^ 2))
    (b : κ → ℝ) (x : ι → ℝ) :
    √(∑ c, (b c + ∑ j, x j * W j c) ^ 2) ≤ √(∑ c, b c ^ 2) + S * √(∑ j, x j ^ 2) := by
  have h : √(∑ c, (b c + ∑ j, x j * W j c) ^ 2) ≤
      √(∑ c, b c ^ 2) + √(∑ c, (∑ j, x j * W j c) ^ 2) := by
    simpa [EuclideanSpace.norm_eq] using
      norm_add_le (WithLp.toLp 2 b) (WithLp.toLp 2 fun c => ∑ j, x j * W j c)
  exact h.trans (by gcongr; exact hW x)

/-- `S`, `F` and `B` bound the spectral norm of `W`, the Frobenius norm of `W` and the norm
of `b`, with `0 ≤ S`. -/
structure LinearBound [Fintype κ] (W : ι → κ → ℝ) (b : κ → ℝ) (S F B : ℝ) : Prop where
  /-- `S` is nonnegative. -/
  nonneg : 0 ≤ S
  /-- `S` bounds the spectral norm of `W`. -/
  spectral : ∀ v : ι → ℝ, √(∑ c, (∑ j, v j * W j c) ^ 2) ≤ S * √(∑ j, v j ^ 2)
  /-- `F` bounds the Frobenius norm of `W`. -/
  frobenius : √(∑ j, ∑ c, W j c ^ 2) ≤ F
  /-- `B` bounds the norm of `b`. -/
  bias : √(∑ c, b c ^ 2) ≤ B

/-- A linear map evaluated in `A` at an input within `e` of `x`, where `‖x‖ ≤ X`, is within the
local error at the computed input plus `S * e` of the exact map at `x`. -/
theorem Arithmetic.sqrt_sum_sq_linear_sub_le' [Fintype κ] (A : Arithmetic) {g h : ℕ → ℝ}
    {um ηm us ηs S F B X e : ℝ}
    (hS : ∀ l : List ℝ, |A.sum l - l.sum| ≤ g l.length * (l.map abs).sum + h l.length)
    (hg : 0 ≤ g (Fintype.card ι + 1)) (hh : 0 ≤ h (Fintype.card ι + 1)) (hum : 0 ≤ um)
    (hηm : 0 ≤ ηm) (hus : 0 ≤ us) (hηs : 0 ≤ ηs)
    (hmul : ∀ t, |A.roundMul t - t| ≤ um * |t| + ηm)
    (hstore : ∀ t, |A.store t - t| ≤ us * |t| + ηs)
    {W : ι → κ → ℝ} {b : κ → ℝ} (hW : LinearBound W b S F B) {x x' : ι → ℝ}
    (hx : √(∑ j, x j ^ 2) ≤ X) (he : √(∑ j, (x' j - x j) ^ 2) ≤ e) :
    √(∑ c, (A.linear W b x' c - (b c + ∑ j, x j * W j c)) ^ 2) ≤
      S * e + us * (B + S * (X + e)) + (1 + us) * (g (Fintype.card ι + 1) * B +
        (g (Fintype.card ι + 1) * (1 + um) + um) * F * (X + e) +
        √(Fintype.card κ) * ((1 + g (Fintype.card ι + 1)) * Fintype.card ι * ηm +
          h (Fintype.card ι + 1))) + √(Fintype.card κ) * ηs := by
  have hx' : √(∑ j, x' j ^ 2) ≤ X + e := by
    have : √(∑ j, x' j ^ 2) ≤ √(∑ j, x j ^ 2) + √(∑ j, (x' j - x j) ^ 2) := by
      simpa [EuclideanSpace.norm_eq] using
        norm_le_norm_add_norm_sub' (WithLp.toLp 2 x') (WithLp.toLp 2 x)
    linarith
  have hloc := A.sqrt_sum_sq_linear_sub_le hS hg hh hum hηm hus hηs hmul hstore W b x'
    hW.frobenius
  have hlip := TransformerBlock.sqrt_sum_sq_linear_sub_le hW.spectral b x x'
  have hmag := sqrt_sum_sq_linear_le hW.spectral b x'
  have htri : √(∑ c, (A.linear W b x' c - (b c + ∑ j, x j * W j c)) ^ 2) ≤
      √(∑ c, (A.linear W b x' c - (b c + ∑ j, x' j * W j c)) ^ 2) +
        √(∑ c, ((b c + ∑ j, x' j * W j c) - (b c + ∑ j, x j * W j c)) ^ 2) := by
    simpa [EuclideanSpace.norm_eq] using norm_sub_le_norm_sub_add_norm_sub
      (WithLp.toLp 2 (A.linear W b x')) (WithLp.toLp 2 fun c => b c + ∑ j, x' j * W j c)
      (WithLp.toLp 2 fun c => b c + ∑ j, x j * W j c)
  have hS0 := hW.nonneg
  have h1 : us * √(∑ c, (b c + ∑ j, x' j * W j c) ^ 2) ≤ us * (B + S * (X + e)) := by
    have : √(∑ c, b c ^ 2) + S * √(∑ j, x' j ^ 2) ≤ B + S * (X + e) :=
      add_le_add hW.bias (mul_le_mul_of_nonneg_left hx' hS0)
    exact mul_le_mul_of_nonneg_left (hmag.trans this) hus
  have hF0 : 0 ≤ F := (Real.sqrt_nonneg _).trans hW.frobenius
  have h2 : g (Fintype.card ι + 1) * √(∑ c, b c ^ 2) +
      (g (Fintype.card ι + 1) * (1 + um) + um) * F * √(∑ j, x' j ^ 2) ≤
      g (Fintype.card ι + 1) * B + (g (Fintype.card ι + 1) * (1 + um) + um) * F * (X + e) :=
    add_le_add (mul_le_mul_of_nonneg_left hW.bias hg)
      (mul_le_mul_of_nonneg_left hx' (by positivity))
  have h3 : S * √(∑ j, (x' j - x j) ^ 2) ≤ S * e := mul_le_mul_of_nonneg_left he hS0
  have h4 := mul_le_mul_of_nonneg_left h2 (by linarith : (0 : ℝ) ≤ 1 + us)
  linarith

/-- A termwise bound `|x i| ≤ a * |y i| + b * |z i| + c` with nonnegative coefficients bounds
the Euclidean norm by `a * ‖y‖ + b * ‖z‖ + √card * c`. -/
theorem sqrt_sum_sq_le_mul_add_mul_add {x y z : ι → ℝ} {a b c : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b)
    (hc : 0 ≤ c) (h : ∀ i, |x i| ≤ a * |y i| + b * |z i| + c) :
    √(∑ i, x i ^ 2) ≤ a * √(∑ i, y i ^ 2) + b * √(∑ i, z i ^ 2) + √(Fintype.card ι) * c := by
  have h' : ∀ i, |x i| ≤ a * |y i| + b * |z i| + c + 0 := fun i => by linarith [h i]
  have hm : ∀ {a : ℝ}, 0 ≤ a → ∀ y : ι → ℝ, √(∑ i, (a * |y i|) ^ 2) = a * √(∑ i, y i ^ 2) :=
    fun {a} ha y => by
      simp_rw [mul_pow, sq_abs, ← Finset.mul_sum]
      rw [Real.sqrt_mul (sq_nonneg a), Real.sqrt_sq ha]
  have hc' : √(∑ _i : ι, c ^ 2) = √(Fintype.card ι) * c := by
    rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, Real.sqrt_mul (Nat.cast_nonneg _),
      Real.sqrt_sq hc]
  have := sqrt_sum_sq_le_add_of_abs_le h'
  rw [hm ha, hm hb, hc'] at this
  simpa using this

/-- Rounding each entry of `z` leaves the Euclidean error `u * ‖y‖ + (1 + u) * ‖z - y‖ + √card * η`
against `y`. -/
theorem sqrt_sum_sq_round_sub_le {f : ℝ → ℝ} {u η : ℝ} (hu : 0 ≤ u) (hη : 0 ≤ η)
    (hf : ∀ t, |f t - t| ≤ u * |t| + η) (z y : ι → ℝ) :
    √(∑ i, (f (z i) - y i) ^ 2) ≤
      u * √(∑ i, y i ^ 2) + (1 + u) * √(∑ i, (z i - y i) ^ 2) + √(Fintype.card ι) * η :=
  sqrt_sum_sq_le_mul_add_mul_add hu (by linarith) hη fun i =>
    ((MixedError.mk hu hη (hf (z i))).trans_of_abs_sub_le (le_refl |z i - y i|)).bound.trans_eq
      (add_assoc _ _ _).symm

/-- For `0 ≤ ε`, the map `p ↦ √(p ^ 2 + ε)` is `1`-Lipschitz. -/
theorem abs_sqrt_sq_add_sub_le {ε : ℝ} (hε : 0 ≤ ε) (p q : ℝ) :
    |√(p ^ 2 + ε) - √(q ^ 2 + ε)| ≤ |p - q| := by
  have key : ∀ p q : ℝ, √(p ^ 2 + ε) ≤ √(q ^ 2 + ε) + |p - q| := fun p q => by
    have hq : |q| ≤ √(q ^ 2 + ε) := Real.abs_le_sqrt (by linarith)
    have hs := Real.sq_sqrt (show 0 ≤ q ^ 2 + ε by positivity)
    rw [Real.sqrt_le_left (by positivity)]
    have h1 : q * (p - q) ≤ |q| * |p - q| := by rw [← abs_mul]; exact le_abs_self _
    have h2 := mul_le_mul_of_nonneg_right hq (abs_nonneg (p - q))
    nlinarith [sq_abs (p - q)]
  rw [abs_sub_le_iff]
  constructor
  · linarith [key p q]
  · linarith [key q p, abs_sub_comm p q]

/-! ### Layer normalization -/

private theorem var_eq_sum_sq_div (x : ι → ℝ) :
    var x = (∑ i, (x i - 𝔼 k, x k) ^ 2) / Fintype.card ι := by
  simp only [var, Fintype.expect_eq_sum_div_card, sq]

/-- Layer normalization of `x` with weight `γ`, bias `β` and variance offset `ε`. -/
noncomputable def layerNorm (γ β : ι → ℝ) (ε : ℝ) (x : ι → ℝ) (j : ι) : ℝ :=
  γ j * ((x j - 𝔼 k, x k) / √(var x + ε)) + β j

/-- Layer normalization evaluated in `A`: the mean, the centered entries, the variance, the
denominator and every output entry are rounded after each operation, and the output is stored. -/
noncomputable def Arithmetic.layerNorm (A : Arithmetic) (γ β : ι → ℝ) (ε : ℝ) (x : ι → ℝ) :
    ι → ℝ :=
  let μ := A.round (A.sum ((Finset.univ : Finset ι).toList.map x) / Fintype.card ι)
  let c := fun j => A.round (x j - μ)
  let σ := A.round (√(A.round (A.round (A.sum ((Finset.univ : Finset ι).toList.map
    fun j => A.round (c j * c j)) / Fintype.card ι) + ε)))
  fun j => A.store (A.round (A.round (γ j * A.round (c j / σ)) + β j))

/-- In exact arithmetic, `Arithmetic.layerNorm` is `TransformerBlock.layerNorm`. -/
theorem Arithmetic.exact_layerNorm (act : ℝ → ℝ) (γ β : ι → ℝ) (ε : ℝ) (x : ι → ℝ) :
    (Arithmetic.exact act).layerNorm γ β ε x = TransformerBlock.layerNorm γ β ε x := by
  funext j
  simp [Arithmetic.layerNorm, Arithmetic.exact, TransformerBlock.layerNorm, var_eq_sum_sq_div,
    Fintype.expect_eq_sum_div_card, Finset.sum_map_toList, sq]

/-- The sum of the squared deviations of `x` from its mean is
`∑ i, x i ^ 2 - Fintype.card ι * (𝔼 k, x k) ^ 2`. -/
theorem sum_sub_mean_sq (x : ι → ℝ) :
    ∑ i, (x i - 𝔼 k, x k) ^ 2 = ∑ i, x i ^ 2 - Fintype.card ι * (𝔼 k, x k) ^ 2 := by
  have hs : ∑ i, x i = Fintype.card ι * 𝔼 k, x k := (Fintype.card_mul_expect x).symm
  have : ∀ i, (x i - 𝔼 k, x k) ^ 2 = x i ^ 2 - 2 * (𝔼 k, x k) * x i + (𝔼 k, x k) ^ 2 :=
    fun i => by ring
  simp only [this, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, hs,
    Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  ring

/-- Subtracting the mean does not increase the Euclidean norm of `x`. -/
theorem sqrt_sum_sub_mean_sq_le (x : ι → ℝ) :
    √(∑ i, (x i - 𝔼 k, x k) ^ 2) ≤ √(∑ i, x i ^ 2) := by
  apply Real.sqrt_le_sqrt
  rw [sum_sub_mean_sq]
  have : 0 ≤ (Fintype.card ι : ℝ) * (𝔼 k, x k) ^ 2 := by positivity
  linarith

/-- Centering is a contraction in the Euclidean norm. -/
theorem sqrt_sum_sub_mean_sub_le (x x' : ι → ℝ) :
    √(∑ i, ((x' i - 𝔼 k, x' k) - (x i - 𝔼 k, x k)) ^ 2) ≤ √(∑ i, (x' i - x i) ^ 2) := by
  have h : √(∑ i, ((x' i - x i) - ((𝔼 k, x' k) - 𝔼 k, x k)) ^ 2) ≤
      √(∑ i, (x' i - x i) ^ 2) := by
    simpa only [Finset.expect_sub_distrib] using sqrt_sum_sub_mean_sq_le fun i => x' i - x i
  calc √(∑ i, ((x' i - 𝔼 k, x' k) - (x i - 𝔼 k, x k)) ^ 2)
      = √(∑ i, ((x' i - x i) - ((𝔼 k, x' k) - 𝔼 k, x k)) ^ 2) := by
        congr 1; exact Finset.sum_congr rfl fun i _ => by ring
    _ ≤ _ := h

/-- For `0 ≤ ε`, the Euclidean norm of `x` less its mean is at most
`√(Fintype.card ι) * √(var x + ε)`. -/
theorem sqrt_sum_sub_mean_sq_le_mul [Nonempty ι] {ε : ℝ} (hε : 0 ≤ ε) (x : ι → ℝ) :
    √(∑ i, (x i - 𝔼 k, x k) ^ 2) ≤ √(Fintype.card ι) * √(var x + ε) := by
  rw [← Real.sqrt_mul (Nat.cast_nonneg _)]
  apply Real.sqrt_le_sqrt
  have hd : (0 : ℝ) < Fintype.card ι := by exact_mod_cast Fintype.card_pos
  have : ∑ i, (x i - 𝔼 k, x k) ^ 2 = Fintype.card ι * var x := by
    rw [var_eq_sum_sq_div]; field_simp
  rw [this]; nlinarith

/-- The layer normalization denominator `√(var x + ε)` moves by at most the Euclidean input
change divided by `√card`. -/
theorem abs_sqrt_var_add_sub_le [Nonempty ι] {ε : ℝ} (hε : 0 ≤ ε) (x x' : ι → ℝ) :
    |√(var x' + ε) - √(var x + ε)| ≤ √(∑ i, (x' i - x i) ^ 2) / √(Fintype.card ι) := by
  have hd : 0 < √(Fintype.card ι : ℝ) := Real.sqrt_pos.2 (by exact_mod_cast Fintype.card_pos)
  have hv : ∀ x : ι → ℝ,
      var x = (√(∑ i, (x i - 𝔼 k, x k) ^ 2) / √(Fintype.card ι)) ^ 2 := fun x => by
    rw [div_pow, Real.sq_sqrt (Finset.sum_nonneg fun i _ => sq_nonneg _),
      Real.sq_sqrt (Nat.cast_nonneg _)]
    exact var_eq_sum_sq_div x
  rw [hv x, hv x']
  refine (abs_sqrt_sq_add_sub_le hε _ _).trans ?_
  rw [← sub_div, abs_div, abs_of_pos hd]
  gcongr
  refine le_trans ?_ (sqrt_sum_sub_mean_sub_le x x')
  simpa [EuclideanSpace.norm_eq] using abs_norm_sub_norm_le
    (WithLp.toLp 2 fun i => x' i - 𝔼 k, x' k) (WithLp.toLp 2 fun i => x i - 𝔼 k, x k)

/-- Layer normalization moves by at most `2 * G / √(var x + ε)` times the Euclidean input change
when `G` bounds the weight. -/
theorem sqrt_sum_sq_layerNorm_sub_le [Nonempty ι] {γ : ι → ℝ} {G ε : ℝ}
    (hγ : ∀ j, |γ j| ≤ G) (hε : 0 < ε) (β x x' : ι → ℝ) :
    √(∑ j, (layerNorm γ β ε x' j - layerNorm γ β ε x j) ^ 2) ≤
      2 * G * √(∑ j, (x' j - x j) ^ 2) / √(var x + ε) := by
  set σ := √(var x + ε) with hσ_def
  set σ' := √(var x' + ε) with hσ'_def
  set E := √(∑ j, (x' j - x j) ^ 2) with hE_def
  have hσ : 0 < σ := Real.sqrt_pos.2 (by linarith [var_nonneg x])
  have hσ' : 0 < σ' := Real.sqrt_pos.2 (by linarith [var_nonneg x'])
  have hd : 0 < √(Fintype.card ι : ℝ) := Real.sqrt_pos.2 (by exact_mod_cast Fintype.card_pos)
  have hG : 0 ≤ G := (abs_nonneg _).trans (hγ (Classical.arbitrary ι))
  have hdiff : |σ' - σ| ≤ E / √(Fintype.card ι) := abs_sqrt_var_add_sub_le hε.le x x'
  have hc' := sqrt_sum_sub_mean_sq_le_mul hε.le x'
  rw [← hσ'_def] at hc'
  have hcc : √(∑ i, ((x' i - 𝔼 k, x' k) - (x i - 𝔼 k, x k)) ^ 2) ≤ E :=
    sqrt_sum_sub_mean_sub_le x x'
  have hterm : ∀ j, |layerNorm γ β ε x' j - layerNorm γ β ε x j| ≤
      G / σ * |(x' j - 𝔼 k, x' k) - (x j - 𝔼 k, x k)| +
        G * |σ' - σ| / (σ' * σ) * |x' j - 𝔼 k, x' k| + 0 := fun j => by
    have hid : layerNorm γ β ε x' j - layerNorm γ β ε x j = γ j *
        (((x' j - 𝔼 k, x' k) - (x j - 𝔼 k, x k)) / σ +
          (x' j - 𝔼 k, x' k) * (σ - σ') / (σ' * σ)) := by
      simp only [layerNorm]
      rw [← hσ_def, ← hσ'_def]
      field_simp
      ring
    have h1 : |((x' j - 𝔼 k, x' k) - (x j - 𝔼 k, x k)) / σ +
        (x' j - 𝔼 k, x' k) * (σ - σ') / (σ' * σ)| ≤
          |(x' j - 𝔼 k, x' k) - (x j - 𝔼 k, x k)| / σ +
            |x' j - 𝔼 k, x' k| * |σ' - σ| / (σ' * σ) := by
      refine (abs_add_le _ _).trans (le_of_eq ?_)
      rw [abs_div, abs_div, abs_mul, abs_of_pos hσ, abs_of_pos (mul_pos hσ' hσ),
        abs_sub_comm σ σ']
    rw [hid, abs_mul]
    calc |γ j| * _ ≤ G * (|(x' j - 𝔼 k, x' k) - (x j - 𝔼 k, x k)| / σ +
          |x' j - 𝔼 k, x' k| * |σ' - σ| / (σ' * σ)) := mul_le_mul (hγ j) h1 (abs_nonneg _) hG
      _ = _ := by ring
  have hsum := sqrt_sum_sq_le_mul_add_mul_add (by positivity) (by positivity) le_rfl hterm
  have hprod : |σ' - σ| * √(∑ i, (x' i - 𝔼 k, x' k) ^ 2) ≤ E * σ' :=
    calc |σ' - σ| * √(∑ i, (x' i - 𝔼 k, x' k) ^ 2)
        ≤ E / √(Fintype.card ι) * (√(Fintype.card ι) * σ') :=
          mul_le_mul hdiff hc' (Real.sqrt_nonneg _) (by positivity)
      _ = E * σ' := by field_simp
  have h2 : G * |σ' - σ| / (σ' * σ) * √(∑ i, (x' i - 𝔼 k, x' k) ^ 2) ≤ G * E / σ := by
    rw [div_mul_eq_mul_div, div_le_div_iff₀ (by positivity) hσ]
    nlinarith [mul_le_mul_of_nonneg_left hprod (mul_nonneg hG hσ.le)]
  have h3 : G / σ * √(∑ i, ((x' i - 𝔼 k, x' k) - (x i - 𝔼 k, x k)) ^ 2) ≤ G / σ * E := by
    gcongr
  calc _ ≤ _ := hsum
    _ ≤ G / σ * E + G * E / σ + √(Fintype.card ι) * 0 := by gcongr
    _ = 2 * G * E / σ := by ring

/-- The Euclidean norm of layer normalization is at most `G * √card + ‖β‖`. -/
theorem sqrt_sum_sq_layerNorm_le [Nonempty ι] {γ : ι → ℝ} {G ε : ℝ} (hγ : ∀ j, |γ j| ≤ G)
    (hε : 0 < ε) (β x : ι → ℝ) :
    √(∑ j, layerNorm γ β ε x j ^ 2) ≤ G * √(Fintype.card ι) + √(∑ j, β j ^ 2) := by
  set σ := √(var x + ε) with hσ_def
  have hσ : 0 < σ := Real.sqrt_pos.2 (by linarith [var_nonneg x])
  have hG : 0 ≤ G := (abs_nonneg _).trans (hγ (Classical.arbitrary ι))
  have hc := sqrt_sum_sub_mean_sq_le_mul hε.le x
  rw [← hσ_def] at hc
  have hterm : ∀ j, |layerNorm γ β ε x j| ≤ G / σ * |x j - 𝔼 k, x k| + 1 * |β j| + 0 :=
    fun j => by
      have : |γ j * ((x j - 𝔼 k, x k) / σ)| ≤ G / σ * |x j - 𝔼 k, x k| := by
        rw [abs_mul, abs_div, abs_of_pos hσ, div_mul_eq_mul_div, mul_div_assoc']
        gcongr
        exact hγ j
      have h' := abs_add_le (γ j * ((x j - 𝔼 k, x k) / σ)) (β j)
      simp only [layerNorm, ← hσ_def] at h' ⊢
      linarith
  have hsum := sqrt_sum_sq_le_mul_add_mul_add (by positivity) zero_le_one le_rfl hterm
  have h1 : G / σ * √(∑ i, (x i - 𝔼 k, x k) ^ 2) ≤ G * √(Fintype.card ι) := by
    rw [div_mul_eq_mul_div, div_le_iff₀ hσ]
    nlinarith [mul_le_mul_of_nonneg_left hc hG]
  linarith

/-- For `0 < ε`, the vector `(x j - 𝔼 k, x k) / √(var x + ε)` has Euclidean norm at most
`√(Fintype.card ι)`. -/
theorem sqrt_sum_sq_sub_mean_div_le [Nonempty ι] {ε : ℝ} (hε : 0 < ε) (x : ι → ℝ) :
    √(∑ j, ((x j - 𝔼 k, x k) / √(var x + ε)) ^ 2) ≤ √(Fintype.card ι) := by
  have hσ : 0 < √(var x + ε) := Real.sqrt_pos.2 (by linarith [var_nonneg x])
  have hc := sqrt_sum_sub_mean_sq_le_mul hε.le x
  simp only [div_pow]
  rw [← Finset.sum_div, Real.sqrt_div' _ (sq_nonneg _), Real.sqrt_sq hσ.le, div_le_iff₀ hσ]
  linarith [mul_comm (√(var x + ε)) (√(Fintype.card ι : ℝ))]

private theorem abs_round_div_card_sub_mean_le [Nonempty ι] {f : ℝ → ℝ} {u η g h s M : ℝ}
    (hu : 0 ≤ u) (hf : ∀ t, |f t - t| ≤ u * |t| + η) (hg : 0 ≤ g) (x : ι → ℝ)
    (hs : |s - ∑ i, x i| ≤ g * ∑ i, |x i| + h) (hx : √(∑ j, x j ^ 2) ≤ M) :
    |f (s / Fintype.card ι) - 𝔼 k, x k| ≤ u * M / √(Fintype.card ι) +
      (1 + u) * (g * √(Fintype.card ι) * M + h) / Fintype.card ι + η := by
  have hd : (0 : ℝ) < Fintype.card ι := by exact_mod_cast Fintype.card_pos
  have hsd : 0 < √(Fintype.card ι : ℝ) := Real.sqrt_pos.2 hd
  have hsdd : √(Fintype.card ι : ℝ) * √(Fintype.card ι) = Fintype.card ι :=
    Real.mul_self_sqrt hd.le
  have hsx : ∑ i, |x i| ≤ √(Fintype.card ι : ℝ) * M :=
    (by simpa [sq_abs, Finset.card_univ] using
        Real.sum_mul_le_sqrt_mul_sqrt Finset.univ (fun _ => (1 : ℝ)) (|x ·|) :
      ∑ i, |x i| ≤ √(Fintype.card ι : ℝ) * √(∑ i, x i ^ 2)).trans
      (mul_le_mul_of_nonneg_left hx hsd.le)
  have hmean : |𝔼 k, x k| ≤ M / √(Fintype.card ι) := by
    rw [Fintype.expect_eq_sum_div_card, abs_div, abs_of_pos hd, div_le_div_iff₀ hd hsd]
    calc |∑ i, x i| * √(Fintype.card ι : ℝ) ≤ (√(Fintype.card ι) * M) * √(Fintype.card ι) :=
          mul_le_mul_of_nonneg_right ((Finset.abs_sum_le_sum_abs _ _).trans hsx) hsd.le
      _ = M * Fintype.card ι := by rw [mul_comm (√_) M, mul_assoc, hsdd]
  have h1 : |s / Fintype.card ι - 𝔼 k, x k| ≤
      (g * √(Fintype.card ι) * M + h) / Fintype.card ι := by
    rw [Fintype.expect_eq_sum_div_card, ← sub_div, abs_div, abs_of_pos hd]
    gcongr
    linarith [mul_le_mul_of_nonneg_left hsx hg]
  have hη : 0 ≤ η := by simpa using (abs_nonneg _).trans (hf 0)
  have h2 := ((MixedError.mk hu hη (hf _)).trans_of_abs_sub_le h1).bound
  have h3 := mul_le_mul_of_nonneg_left hmean hu
  have h4 : u * (M / √(Fintype.card ι)) = u * M / √(Fintype.card ι) := by ring
  have h5 : (1 + u) * ((g * √(Fintype.card ι) * M + h) / Fintype.card ι) =
      (1 + u) * (g * √(Fintype.card ι) * M + h) / Fintype.card ι := by ring
  linarith

private theorem sqrt_sum_sq_round_sub_mean_le {f : ℝ → ℝ} {u η μh eμ M : ℝ} (hu : 0 ≤ u)
    (hη : 0 ≤ η) (hf : ∀ t, |f t - t| ≤ u * |t| + η) (x : ι → ℝ)
    (hx : √(∑ j, x j ^ 2) ≤ M) (hμ : |μh - 𝔼 k, x k| ≤ eμ) :
    √(∑ j, (f (x j - μh) - (x j - 𝔼 k, x k)) ^ 2) ≤
      u * M + √(Fintype.card ι) * ((1 + u) * eμ + η) := by
  have heμ : 0 ≤ eμ := (abs_nonneg _).trans hμ
  have hterm : ∀ j, |f (x j - μh) - (x j - 𝔼 k, x k)| ≤
      u * |x j - 𝔼 k, x k| + 0 * |x j - 𝔼 k, x k| + ((1 + u) * eμ + η) := fun j => by
    have h1 : |(x j - μh) - (x j - 𝔼 k, x k)| ≤ eμ := by
      rw [show (x j - μh) - (x j - 𝔼 k, x k) = -(μh - 𝔼 k, x k) by ring, abs_neg]; exact hμ
    have := ((MixedError.mk hu hη (hf _)).trans_of_abs_sub_le h1).bound
    linarith
  have := sqrt_sum_sq_le_mul_add_mul_add hu le_rfl (by positivity) hterm
  have h2 := mul_le_mul_of_nonneg_left ((sqrt_sum_sub_mean_sq_le x).trans hx) hu
  linarith

private theorem abs_sum_round_mul_self_sub_le {f : ℝ → ℝ} {u η g h s M ec : ℝ} (hu : 0 ≤ u)
    (hf : ∀ t, |f t - t| ≤ u * |t| + η) (hg : 0 ≤ g) (x ch : ι → ℝ)
    (hx : √(∑ j, x j ^ 2) ≤ M) (hc : √(∑ j, (ch j - (x j - 𝔼 k, x k)) ^ 2) ≤ ec)
    (hs : |s - ∑ j, f (ch j * ch j)| ≤ g * ∑ j, |f (ch j * ch j)| + h) :
    |s - ∑ j, (x j - 𝔼 k, x k) ^ 2| ≤ g * ((1 + u) * (M + ec) ^ 2 + Fintype.card ι * η) + h +
      u * (M + ec) ^ 2 + Fintype.card ι * η + ec * (2 * M + ec) := by
  have hcM : √(∑ j, (x j - 𝔼 k, x k) ^ 2) ≤ M := (sqrt_sum_sub_mean_sq_le x).trans hx
  have hec0 : 0 ≤ ec := (Real.sqrt_nonneg _).trans hc
  have hchn : √(∑ j, ch j ^ 2) ≤ M + ec := by
    have : √(∑ j, ch j ^ 2) ≤
        √(∑ j, (x j - 𝔼 k, x k) ^ 2) + √(∑ j, (ch j - (x j - 𝔼 k, x k)) ^ 2) := by
      simpa [EuclideanSpace.norm_eq] using
        norm_le_norm_add_norm_sub' (WithLp.toLp 2 ch) (WithLp.toLp 2 fun j => x j - 𝔼 k, x k)
    linarith
  have hch2 : ∑ j, ch j ^ 2 ≤ (M + ec) ^ 2 := by
    have h0 : 0 ≤ ∑ j, ch j ^ 2 := Finset.sum_nonneg fun j _ => sq_nonneg _
    calc ∑ j, ch j ^ 2 = √(∑ j, ch j ^ 2) ^ 2 := (Real.sq_sqrt h0).symm
      _ ≤ (M + ec) ^ 2 := pow_le_pow_left₀ (Real.sqrt_nonneg _) hchn 2
  have hp : ∀ j, |f (ch j * ch j) - ch j ^ 2| ≤ u * ch j ^ 2 + η := fun j => by
    have := hf (ch j * ch j)
    rw [abs_mul_self] at this
    rw [sq]; exact this
  have hpabs : ∑ j, |f (ch j * ch j)| ≤ (1 + u) * ∑ j, ch j ^ 2 + Fintype.card ι * η := by
    have : ∀ j, |f (ch j * ch j)| ≤ (1 + u) * ch j ^ 2 + η := fun j => by
      have h1 := hp j
      have h2 := abs_sub_abs_le_abs_sub (f (ch j * ch j)) (ch j ^ 2)
      rw [abs_of_nonneg (sq_nonneg (ch j))] at h2
      linarith
    calc _ ≤ ∑ j, ((1 + u) * ch j ^ 2 + η) := Finset.sum_le_sum fun j _ => this j
      _ = _ := by rw [Finset.sum_add_distrib, ← Finset.mul_sum]; simp
  have hpsum : |∑ j, f (ch j * ch j) - ∑ j, ch j ^ 2| ≤
      u * ∑ j, ch j ^ 2 + Fintype.card ι * η := by
    rw [← Finset.sum_sub_distrib]
    calc _ ≤ ∑ j, |f (ch j * ch j) - ch j ^ 2| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ j, (u * ch j ^ 2 + η) := Finset.sum_le_sum fun j _ => hp j
      _ = _ := by rw [Finset.sum_add_distrib, ← Finset.mul_sum]; simp
  have hdiff2 : |∑ j, ch j ^ 2 - ∑ j, (x j - 𝔼 k, x k) ^ 2| ≤ ec * (2 * M + ec) := by
    have hfac : ∑ j, ch j ^ 2 - ∑ j, (x j - 𝔼 k, x k) ^ 2 =
        ∑ j, (ch j - (x j - 𝔼 k, x k)) * (ch j + (x j - 𝔼 k, x k)) := by
      rw [← Finset.sum_sub_distrib]; exact Finset.sum_congr rfl fun j _ => by ring
    have hplus : √(∑ j, (ch j + (x j - 𝔼 k, x k)) ^ 2) ≤ 2 * M + ec := by
      have : √(∑ j, (ch j + (x j - 𝔼 k, x k)) ^ 2) ≤
          √(∑ j, ch j ^ 2) + √(∑ j, (x j - 𝔼 k, x k) ^ 2) := by
        simpa [EuclideanSpace.norm_eq] using
          norm_add_le (WithLp.toLp 2 ch) (WithLp.toLp 2 fun j => x j - 𝔼 k, x k)
      linarith
    rw [hfac]
    calc _ ≤ ∑ j, |ch j - (x j - 𝔼 k, x k)| * |ch j + (x j - 𝔼 k, x k)| := by
          refine (Finset.abs_sum_le_sum_abs _ _).trans (le_of_eq ?_)
          exact Finset.sum_congr rfl fun j _ => abs_mul _ _
      _ ≤ √(∑ j, (ch j - (x j - 𝔼 k, x k)) ^ 2) * √(∑ j, (ch j + (x j - 𝔼 k, x k)) ^ 2) := by
          simpa [sq_abs] using Real.sum_mul_le_sqrt_mul_sqrt Finset.univ
            (fun j => |ch j - (x j - 𝔼 k, x k)|) fun j => |ch j + (x j - 𝔼 k, x k)|
      _ ≤ ec * (2 * M + ec) := mul_le_mul hc hplus (Real.sqrt_nonneg _) hec0
  have h1 : |s - ∑ j, f (ch j * ch j)| ≤
      g * ((1 + u) * (M + ec) ^ 2 + Fintype.card ι * η) + h := by
    refine hs.trans ?_
    have e1 := mul_le_mul_of_nonneg_left hch2 (by linarith : (0 : ℝ) ≤ 1 + u)
    have e2 : ∑ j, |f (ch j * ch j)| ≤ (1 + u) * (M + ec) ^ 2 + Fintype.card ι * η := by
      linarith
    have := mul_le_mul_of_nonneg_left e2 hg
    linarith
  have h2 := abs_sub_le s (∑ j, f (ch j * ch j)) (∑ j, ch j ^ 2)
  have h3 := abs_sub_le s (∑ j, ch j ^ 2) (∑ j, (x j - 𝔼 k, x k) ^ 2)
  have h4 := mul_le_mul_of_nonneg_left hch2 hu
  linarith

private theorem abs_round_sqrt_sub_le {f : ℝ → ℝ} {u η ε s eV ev ew σlo σhi : ℝ}
    (hu : 0 ≤ u) (hf : ∀ t, |f t - t| ≤ u * |t| + η) (hε : 0 < ε) (x : ι → ℝ)
    (hs : |s - ∑ j, (x j - 𝔼 k, x k) ^ 2| ≤ eV) (hσ0 : 0 < σlo) (hlo : σlo ≤ √(var x + ε))
    (hhi : √(var x + ε) ≤ σhi) (hv : u * σhi ^ 2 + (1 + u) * eV / Fintype.card ι + η ≤ ev)
    (hw : u * σhi ^ 2 + (1 + u) * ev + η ≤ ew) :
    |f (√(f (f (s / Fintype.card ι) + ε))) - √(var x + ε)| ≤
      u * σhi + (1 + u) * ew / σlo + η := by
  have hη : 0 ≤ η := by simpa using (abs_nonneg _).trans (hf 0)
  set σ := √(var x + ε) with hσ_def
  have hσpos : 0 < σ := hσ0.trans_le hlo
  have hσsq : σ ^ 2 = var x + ε := Real.sq_sqrt (by linarith [var_nonneg x])
  have hσhi2 : σ ^ 2 ≤ σhi ^ 2 := pow_le_pow_left₀ hσpos.le hhi 2
  have hverr : |f (s / Fintype.card ι) - var x| ≤ ev := by
    have h1 : |s / Fintype.card ι - var x| ≤ eV / Fintype.card ι := by
      rcases (Fintype.card ι).eq_zero_or_pos with h0 | h0
      · have he : (0 : ℝ) ≤ eV := (abs_nonneg _).trans hs
        simp [h0, var_eq_sum_sq_div]
      · have hd : (0 : ℝ) < Fintype.card ι := by exact_mod_cast h0
        rw [var_eq_sum_sq_div, ← sub_div, abs_div, abs_of_pos hd]; gcongr
    have h2 := ((MixedError.mk hu hη (hf _)).trans_of_abs_sub_le h1).bound
    rw [abs_of_nonneg (var_nonneg x)] at h2
    have h3 : u * var x ≤ u * σhi ^ 2 := mul_le_mul_of_nonneg_left (by linarith) hu
    have h4 : (1 + u) * (eV / Fintype.card ι) = (1 + u) * eV / Fintype.card ι := by ring
    linarith
  have hwerr : |f (f (s / Fintype.card ι) + ε) - σ ^ 2| ≤ ew := by
    have h1 : |f (s / Fintype.card ι) + ε - σ ^ 2| ≤ ev := by
      rw [hσsq, add_sub_add_right_eq_sub]; exact hverr
    have h2 := ((MixedError.mk hu hη (hf _)).trans_of_abs_sub_le h1).bound
    have h3 : u * |σ ^ 2| ≤ u * σhi ^ 2 := by
      rw [abs_of_nonneg (sq_nonneg _)]; exact mul_le_mul_of_nonneg_left hσhi2 hu
    linarith
  have hew0 : 0 ≤ ew := (abs_nonneg _).trans hwerr
  have h1 : |√(f (f (s / Fintype.card ι) + ε)) - σ| ≤ ew / σlo := by
    have := ((MixedError.of_abs_sub_le
      (le_refl |f (f (s / Fintype.card ι) + ε) - σ ^ 2|)).sqrt_of_le
        (show 0 < σ ^ 2 by positivity) le_rfl).bound
    rw [Real.sqrt_sq hσpos.le, zero_mul, zero_add] at this
    exact this.trans (div_le_div₀ hew0 hwerr hσ0 hlo)
  have h2 := ((MixedError.mk hu hη (hf _)).trans_of_abs_sub_le h1).bound
  rw [abs_of_pos hσpos] at h2
  have h3 := mul_le_mul_of_nonneg_left hhi hu
  have h4 : (1 + u) * (ew / σlo) = (1 + u) * ew / σlo := by ring
  linarith

private theorem sqrt_sum_sq_round_div_sub_le [Nonempty ι] {f : ℝ → ℝ} {u η ε σh ec eσ σlo : ℝ}
    (hu : 0 ≤ u) (hη : 0 ≤ η) (hf : ∀ t, |f t - t| ≤ u * |t| + η) (hε : 0 < ε)
    (x ch : ι → ℝ) (hc : √(∑ j, (ch j - (x j - 𝔼 k, x k)) ^ 2) ≤ ec)
    (hσh : |σh - √(var x + ε)| ≤ eσ) (hlo : σlo ≤ √(var x + ε)) (hσlo : eσ < σlo) :
    √(∑ j, (f (ch j / σh) - (x j - 𝔼 k, x k) / √(var x + ε)) ^ 2) ≤
      u * √(Fintype.card ι) + (1 + u) * (ec + √(Fintype.card ι) * eσ) / (σlo - eσ) +
        √(Fintype.card ι) * η := by
  have hzn := sqrt_sum_sq_sub_mean_div_le hε x
  have hcσ := sqrt_sum_sub_mean_sq_le_mul hε.le x
  set σ := √(var x + ε) with hσ_def
  have heσ0 : 0 ≤ eσ := (abs_nonneg _).trans hσh
  have hgap : 0 < σlo - eσ := by linarith
  have hσpos : 0 < σ := by linarith
  have hσh0 : σlo - eσ ≤ σh := by
    have := neg_abs_le (σh - σ); linarith
  have hσhpos : 0 < σh := hgap.trans_le hσh0
  have hdivterm : ∀ j, |ch j / σh - (x j - 𝔼 k, x k) / σ| ≤
      1 / (σlo - eσ) * |ch j - (x j - 𝔼 k, x k)| + eσ / (σ * (σlo - eσ)) * |x j - 𝔼 k, x k| + 0 :=
    fun j => by
    have hid : ch j / σh - (x j - 𝔼 k, x k) / σ =
        (ch j - (x j - 𝔼 k, x k)) / σh + (x j - 𝔼 k, x k) * (σ - σh) / (σ * σh) := by
      field_simp; ring
    rw [hid]
    refine (abs_add_le _ _).trans ?_
    rw [abs_div, abs_div, abs_mul, abs_of_pos hσhpos, abs_of_pos (mul_pos hσpos hσhpos),
      abs_sub_comm σ σh]
    have e1 : |ch j - (x j - 𝔼 k, x k)| / σh ≤ 1 / (σlo - eσ) * |ch j - (x j - 𝔼 k, x k)| := by
      rw [one_div_mul_eq_div]; exact div_le_div_of_nonneg_left (abs_nonneg _) hgap hσh0
    have e2 : |x j - 𝔼 k, x k| * |σh - σ| / (σ * σh) ≤
        eσ / (σ * (σlo - eσ)) * |x j - 𝔼 k, x k| := by
      rw [div_mul_eq_mul_div, mul_comm eσ]
      exact div_le_div₀ (mul_nonneg (abs_nonneg _) heσ0)
        (mul_le_mul_of_nonneg_left hσh (abs_nonneg _)) (mul_pos hσpos hgap)
        (mul_le_mul_of_nonneg_left hσh0 hσpos.le)
    linarith
  have hdiv : √(∑ j, (ch j / σh - (x j - 𝔼 k, x k) / σ) ^ 2) ≤
      (ec + √(Fintype.card ι) * eσ) / (σlo - eσ) := by
    have := sqrt_sum_sq_le_mul_add_mul_add (one_div_pos.2 hgap).le
      (div_nonneg heσ0 (mul_pos hσpos hgap).le) le_rfl hdivterm
    have e1 : 1 / (σlo - eσ) * √(∑ j, (ch j - (x j - 𝔼 k, x k)) ^ 2) ≤ 1 / (σlo - eσ) * ec :=
      mul_le_mul_of_nonneg_left hc (one_div_pos.2 hgap).le
    have e2 : eσ / (σ * (σlo - eσ)) * √(∑ j, (x j - 𝔼 k, x k) ^ 2) ≤
        √(Fintype.card ι) * eσ / (σlo - eσ) := by
      rw [div_mul_eq_mul_div, div_le_div_iff₀ (mul_pos hσpos hgap) hgap]
      calc eσ * √(∑ j, (x j - 𝔼 k, x k) ^ 2) * (σlo - eσ)
          ≤ eσ * (√(Fintype.card ι) * σ) * (σlo - eσ) := by gcongr
        _ = √(Fintype.card ι) * eσ * (σ * (σlo - eσ)) := by ring
    have e3 : 1 / (σlo - eσ) * ec + √(Fintype.card ι) * eσ / (σlo - eσ) =
        (ec + √(Fintype.card ι) * eσ) / (σlo - eσ) := by ring
    linarith
  have := sqrt_sum_sq_round_sub_le hu hη hf (fun j => ch j / σh) fun j => (x j - 𝔼 k, x k) / σ
  beta_reduce at this
  have e1 := mul_le_mul_of_nonneg_left hzn hu
  have e2 := mul_le_mul_of_nonneg_left hdiv (by linarith : (0 : ℝ) ≤ 1 + u)
  have e3 : (1 + u) * ((ec + √(Fintype.card ι) * eσ) / (σlo - eσ)) =
      (1 + u) * (ec + √(Fintype.card ι) * eσ) / (σlo - eσ) := by ring
  linarith

private theorem sqrt_sum_sq_affine_sub_le {f s : ℝ → ℝ} {u η us ηs G Y ez ep : ℝ}
    (hu : 0 ≤ u) (hη : 0 ≤ η) (hus : 0 ≤ us) (hηs : 0 ≤ ηs)
    (hf : ∀ t, |f t - t| ≤ u * |t| + η) (hs : ∀ t, |s t - t| ≤ us * |t| + ηs) {γ : ι → ℝ}
    (hγ : ∀ j, |γ j| ≤ G) (hG : 0 ≤ G) (β z zh : ι → ℝ)
    (hzn : √(∑ j, z j ^ 2) ≤ √(Fintype.card ι)) (hz : √(∑ j, (zh j - z j) ^ 2) ≤ ez)
    (hY : √(∑ j, (γ j * z j + β j) ^ 2) ≤ Y)
    (hp : u * G * √(Fintype.card ι) + (1 + u) * G * ez + √(Fintype.card ι) * η ≤ ep) :
    √(∑ j, (s (f (f (γ j * zh j) + β j)) - (γ j * z j + β j)) ^ 2) ≤
      (us + (1 + us) * u) * Y + (1 + us) * (1 + u) * ep +
        √(Fintype.card ι) * ((1 + us) * η + ηs) := by
  have hperr : √(∑ j, (f (γ j * zh j) - γ j * z j) ^ 2) ≤ ep := by
    have hterm : ∀ j, |f (γ j * zh j) - γ j * z j| ≤
        (u * G) * |z j| + ((1 + u) * G) * |zh j - z j| + η := fun j => by
      have h1 : |γ j * zh j - γ j * z j| ≤ G * |zh j - z j| := by
        rw [← mul_sub, abs_mul]; exact mul_le_mul_of_nonneg_right (hγ j) (abs_nonneg _)
      have h2 := ((MixedError.mk hu hη (hf _)).trans_of_abs_sub_le h1).bound
      have h3 : u * |γ j * z j| ≤ u * G * |z j| := by
        rw [abs_mul, ← mul_assoc]
        exact mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left (hγ j) hu) (abs_nonneg _)
      linarith
    have := sqrt_sum_sq_le_mul_add_mul_add (mul_nonneg hu hG)
      (mul_nonneg (by linarith) hG) hη hterm
    have e1 := mul_le_mul_of_nonneg_left hzn (mul_nonneg hu hG)
    have e2 := mul_le_mul_of_nonneg_left hz (mul_nonneg (by linarith : (0 : ℝ) ≤ 1 + u) hG)
    linarith
  have hfinal : ∀ j, |s (f (f (γ j * zh j) + β j)) - (γ j * z j + β j)| ≤
      (us + (1 + us) * u) * |γ j * z j + β j| +
        ((1 + us) * (1 + u)) * |f (γ j * zh j) - γ j * z j| + ((1 + us) * η + ηs) := fun j => by
    have h1 : |f (γ j * zh j) + β j - (γ j * z j + β j)| ≤ |f (γ j * zh j) - γ j * z j| := by
      rw [add_sub_add_right_eq_sub]
    have h2 := ((MixedError.mk hu hη (hf _)).trans_of_abs_sub_le h1).bound
    have h3 := ((MixedError.mk hus hηs (hs _)).trans_of_abs_sub_le h2).bound
    linarith
  have := sqrt_sum_sq_le_mul_add_mul_add (by positivity) (by positivity) (by positivity) hfinal
  have e1 := mul_le_mul_of_nonneg_left hY (by positivity : 0 ≤ us + (1 + us) * u)
  have e2 := mul_le_mul_of_nonneg_left hperr (by positivity : 0 ≤ (1 + us) * (1 + u))
  linarith

/-- Layer normalization evaluated in `A` at `x` is within the stated Euclidean distance of the
exact layer normalization at `x`, where `eμ, ec, eV, ev, ew, eσ, ez, ep` bound the errors of the
mean, the centered vector, the variance sum, the variance, `var + ε`, the denominator, the
normalized vector and its product with `γ`. -/
theorem Arithmetic.sqrt_sum_sq_layerNorm_sub_le [Nonempty ι] (A : Arithmetic) {g h : ℕ → ℝ}
    {u η us ηs : ℝ}
    (hS : ∀ l : List ℝ, |A.sum l - l.sum| ≤ g l.length * (l.map abs).sum + h l.length)
    (hg : 0 ≤ g (Fintype.card ι)) (hu : 0 ≤ u) (hη : 0 ≤ η) (hus : 0 ≤ us) (hηs : 0 ≤ ηs)
    (hround : ∀ t, |A.round t - t| ≤ u * |t| + η)
    (hstore : ∀ t, |A.store t - t| ≤ us * |t| + ηs) {γ : ι → ℝ} {G : ℝ} (hγ : ∀ j, |γ j| ≤ G)
    (β : ι → ℝ) {ε : ℝ} (hε : 0 < ε) {x : ι → ℝ} {M σlo σhi Y eμ ec eV ev ew eσ ez ep : ℝ}
    (hx : √(∑ j, x j ^ 2) ≤ M) (hσ0 : 0 < σlo) (hlo : σlo ≤ √(var x + ε))
    (hhi : √(var x + ε) ≤ σhi) (hY : √(∑ j, TransformerBlock.layerNorm γ β ε x j ^ 2) ≤ Y)
    (hμ : u * M / √(Fintype.card ι) + (1 + u) * (g (Fintype.card ι) * √(Fintype.card ι) * M +
      h (Fintype.card ι)) / Fintype.card ι + η ≤ eμ)
    (hc : u * M + √(Fintype.card ι) * ((1 + u) * eμ + η) ≤ ec)
    (hV : g (Fintype.card ι) * ((1 + u) * (M + ec) ^ 2 + Fintype.card ι * η) +
      h (Fintype.card ι) + u * (M + ec) ^ 2 + Fintype.card ι * η + ec * (2 * M + ec) ≤ eV)
    (hv : u * σhi ^ 2 + (1 + u) * eV / Fintype.card ι + η ≤ ev)
    (hw : u * σhi ^ 2 + (1 + u) * ev + η ≤ ew)
    (hσ : u * σhi + (1 + u) * ew / σlo + η ≤ eσ) (hσlo : eσ < σlo)
    (hz : u * √(Fintype.card ι) + (1 + u) * (ec + √(Fintype.card ι) * eσ) / (σlo - eσ) +
      √(Fintype.card ι) * η ≤ ez)
    (hp : u * G * √(Fintype.card ι) + (1 + u) * G * ez + √(Fintype.card ι) * η ≤ ep) :
    √(∑ j, (A.layerNorm γ β ε x j - TransformerBlock.layerNorm γ β ε x j) ^ 2) ≤
      (us + (1 + us) * u) * Y + (1 + us) * (1 + u) * ep +
        √(Fintype.card ι) * ((1 + us) * η + ηs) := by
  obtain ⟨μh, hμh⟩ : ∃ μh, μh = A.round (A.sum ((Finset.univ : Finset ι).toList.map x) /
      Fintype.card ι) := ⟨_, rfl⟩
  obtain ⟨ch, hch⟩ : ∃ ch : ι → ℝ, ch = fun j => A.round (x j - μh) := ⟨_, rfl⟩
  obtain ⟨Sv, hSv⟩ : ∃ Sv, Sv = A.sum ((Finset.univ : Finset ι).toList.map
      fun j => A.round (ch j * ch j)) := ⟨_, rfl⟩
  obtain ⟨σh, hσh⟩ : ∃ σh, σh = A.round (√(A.round (A.round (Sv / Fintype.card ι) + ε))) :=
    ⟨_, rfl⟩
  have hLN : ∀ j, A.layerNorm γ β ε x j =
      A.store (A.round (A.round (γ j * A.round (ch j / σh)) + β j)) := fun j => by
    subst hμh hch hSv hσh; rfl
  simp only [hLN]
  have hG : 0 ≤ G := (abs_nonneg _).trans (hγ (Classical.arbitrary ι))
  have hS1 := abs_sum_toList_sub_le hS Finset.univ x
  have hμerr : |μh - 𝔼 k, x k| ≤ eμ := by
    rw [hμh]; exact (abs_round_div_card_sub_mean_le hu hround hg x hS1 hx).trans hμ
  have hcerr : √(∑ j, (ch j - (x j - 𝔼 k, x k)) ^ 2) ≤ ec := by
    rw [hch]; exact (sqrt_sum_sq_round_sub_mean_le hu hη hround x hx hμerr).trans hc
  have hS2 := abs_sum_toList_sub_le hS Finset.univ fun j => A.round (ch j * ch j)
  have hVerr : |Sv - ∑ j, (x j - 𝔼 k, x k) ^ 2| ≤ eV := by
    rw [hSv]; exact (abs_sum_round_mul_self_sub_le hu hround hg x ch hx hcerr hS2).trans hV
  have hσerr : |σh - √(var x + ε)| ≤ eσ := by
    rw [hσh]; exact (abs_round_sqrt_sub_le hu hround hε x hVerr hσ0 hlo hhi hv hw).trans hσ
  have hzerr := (sqrt_sum_sq_round_div_sub_le hu hη hround hε x ch hcerr hσerr hlo hσlo).trans hz
  exact sqrt_sum_sq_affine_sub_le hu hη hus hηs hround hstore hγ hG β
    (fun j => (x j - 𝔼 k, x k) / √(var x + ε)) (fun j => A.round (ch j / σh))
    (sqrt_sum_sq_sub_mean_div_le hε x) hzerr hY hp

/-! ### Softmax and attention -/

variable {ν : Type*}

/-- The softmax weight of `j` among the logits `x` on the keys `s`. -/
noncomputable def softmax (s : Finset ν) (x : ν → ℝ) (j : ν) : ℝ :=
  Real.exp (x j) / ∑ i ∈ s, Real.exp (x i)

/-- Every entry of `softmax s x` is nonnegative. -/
theorem softmax_nonneg (s : Finset ν) (x : ν → ℝ) (j : ν) : 0 ≤ softmax s x j := by
  unfold softmax; positivity

/-- The entries of `softmax s x` over a nonempty `s` sum to `1`. -/
theorem sum_softmax {s : Finset ν} (hs : s.Nonempty) (x : ν → ℝ) :
    ∑ j ∈ s, softmax s x j = 1 := by
  unfold softmax
  rw [← Finset.sum_div]
  exact div_self (Finset.sum_pos (fun i _ => Real.exp_pos _) hs).ne'

/-- Subtracting `m` from every logit leaves the softmax unchanged. -/
theorem softmax_eq_exp_sub_div (s : Finset ν) (x : ν → ℝ) (m : ℝ) (j : ν) :
    softmax s x j = Real.exp (x j - m) / ∑ i ∈ s, Real.exp (x i - m) := by
  simp only [softmax, Real.exp_sub, ← Finset.sum_div]
  rw [div_div_div_cancel_right₀ (Real.exp_pos m).ne']

private theorem abs_sub_mul_sub_le {lo hi r a : ℝ} (hr : lo ≤ r ∧ r ≤ hi) (ha : lo ≤ a ∧ a ≤ hi) :
    |r - a| * (hi - lo) ≤ (hi - r) * (a - lo) + (r - lo) * (hi - a) := by
  rcases le_total a r with h | h
  · rw [abs_of_nonneg (by linarith)]; nlinarith
  · rw [abs_of_nonpos (by linarith)]; nlinarith

/-- Softmax distributions on `s` whose logits differ by at most `t` are within
`2 * (exp t - 1) / (exp t + 1)` of each other in `ℓ¹`. -/
theorem sum_abs_softmax_sub_le {s : Finset ν} (hs : s.Nonempty) {x x' : ν → ℝ} {t : ℝ}
    (ht : ∀ j ∈ s, |x' j - x j| ≤ t) :
    ∑ j ∈ s, |softmax s x' j - softmax s x j| ≤ 2 * (Real.exp t - 1) / (Real.exp t + 1) := by
  obtain ⟨j0, hj0⟩ := hs
  have ht0 : 0 ≤ t := (abs_nonneg _).trans (ht j0 hj0)
  obtain ⟨Z, hZ⟩ : ∃ Z, Z = ∑ i ∈ s, Real.exp (x i) := ⟨_, rfl⟩
  have hZ0 : 0 < Z := hZ ▸ Finset.sum_pos (fun i _ => Real.exp_pos _) ⟨j0, hj0⟩
  obtain ⟨r, hr⟩ : ∃ r : ν → ℝ, r = fun j => Real.exp (x' j - x j) := ⟨_, rfl⟩
  have hexp : ∀ j, Real.exp (x' j) = Real.exp (x j) * r j := fun j => by
    rw [hr, ← Real.exp_add]; ring_nf
  have hr0 : ∀ j, 0 < r j := fun j => hr ▸ Real.exp_pos _
  obtain ⟨A, hA⟩ : ∃ A, A = ∑ j ∈ s, softmax s x j * r j := ⟨_, rfl⟩
  have hp : ∀ j, softmax s x j = Real.exp (x j) / Z := fun j => by rw [softmax, hZ]
  have hAZ : A = (∑ i ∈ s, Real.exp (x i) * r i) / Z := by
    rw [hA, Finset.sum_div]; exact Finset.sum_congr rfl fun i _ => by rw [hp]; ring
  have hA0 : 0 < A := by
    rw [hAZ]
    exact div_pos (Finset.sum_pos (fun i _ => mul_pos (Real.exp_pos _) (hr0 i)) ⟨j0, hj0⟩) hZ0
  have hp' : ∀ j, softmax s x' j = softmax s x j * r j / A := fun j => by
    rw [softmax, hexp, Finset.sum_congr rfl fun i _ => hexp i, hAZ, hp]
    field_simp
  have hterm : ∀ j, |softmax s x' j - softmax s x j| = softmax s x j * |r j - A| / A :=
    fun j => by
    rw [hp', show softmax s x j * r j / A - softmax s x j = softmax s x j * (r j - A) / A by
      field_simp, abs_div, abs_mul, abs_of_nonneg (softmax_nonneg s x j), abs_of_pos hA0]
  simp only [hterm]
  rw [← Finset.sum_div]
  obtain ⟨D, hD⟩ : ∃ D, D = ∑ j ∈ s, softmax s x j * |r j - A| := ⟨_, rfl⟩
  rw [← hD]
  have hsum1 := sum_softmax ⟨j0, hj0⟩ x
  have hrb : ∀ j ∈ s, Real.exp (-t) ≤ r j ∧ r j ≤ Real.exp t := fun j hj => by
    have := abs_le.1 (ht j hj)
    rw [hr]
    exact ⟨Real.exp_le_exp.2 (by linarith), Real.exp_le_exp.2 (by linarith)⟩
  have hAb : Real.exp (-t) ≤ A ∧ A ≤ Real.exp t := by
    constructor
    · calc Real.exp (-t) = ∑ j ∈ s, softmax s x j * Real.exp (-t) := by
            rw [← Finset.sum_mul, hsum1, one_mul]
        _ ≤ A := hA ▸ Finset.sum_le_sum fun j hj =>
            mul_le_mul_of_nonneg_left (hrb j hj).1 (softmax_nonneg s x j)
    · calc A ≤ ∑ j ∈ s, softmax s x j * Real.exp t := hA ▸ Finset.sum_le_sum fun j hj =>
            mul_le_mul_of_nonneg_left (hrb j hj).2 (softmax_nonneg s x j)
        _ = Real.exp t := by rw [← Finset.sum_mul, hsum1, one_mul]
  have hD0 : 0 ≤ D := hD ▸ Finset.sum_nonneg fun j _ =>
    mul_nonneg (softmax_nonneg s x j) (abs_nonneg _)
  rcases ht0.eq_or_lt with h0 | hpos
  · subst h0
    have hr1 : ∀ j ∈ s, r j = 1 := fun j hj => by
      have := hrb j hj; simp only [neg_zero, Real.exp_zero] at this; linarith
    have hA1 : A = 1 := by simp only [neg_zero, Real.exp_zero] at hAb; linarith
    have : D = 0 := by
      rw [hD]; exact Finset.sum_eq_zero fun j hj => by rw [hr1 j hj, hA1]; simp
    rw [this]; simp
  set hi := Real.exp t with hhi
  set lo := Real.exp (-t) with hlo
  have hhilo : hi * lo = 1 := by rw [hhi, hlo, ← Real.exp_add]; simp
  have hhi1 : 1 < hi := Real.one_lt_exp_iff.2 hpos
  have hlo0 : 0 < lo := Real.exp_pos _
  have hchord : D * (hi - lo) ≤ 2 * (hi - A) * (A - lo) := by
    calc D * (hi - lo) = ∑ j ∈ s, softmax s x j * (|r j - A| * (hi - lo)) := by
          rw [hD, Finset.sum_mul]; exact Finset.sum_congr rfl fun j _ => by ring
      _ ≤ ∑ j ∈ s, softmax s x j * ((hi - r j) * (A - lo) + (r j - lo) * (hi - A)) :=
          Finset.sum_le_sum fun j hj => mul_le_mul_of_nonneg_left
            (abs_sub_mul_sub_le (hrb j hj) hAb) (softmax_nonneg s x j)
      _ = (hi * (A - lo) - lo * (hi - A)) * ∑ j ∈ s, softmax s x j +
            (hi - A - (A - lo)) * ∑ j ∈ s, softmax s x j * r j := by
          rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
          exact Finset.sum_congr rfl fun j _ => by ring
      _ = 2 * (hi - A) * (A - lo) := by rw [hsum1, ← hA]; ring
  have hamgm : 2 * (hi - A) * (A - lo) ≤ A * (2 * (hi + lo - 2)) := by
    nlinarith [sq_nonneg (A - 1)]
  have e1 : hi - lo = (hi - 1) * (1 + lo) := by linear_combination -hhilo
  have e2 : 2 * (hi + lo - 2) = (hi - 1) * (2 * (1 - lo)) := by linear_combination 2 * hhilo
  have hDA : D / A * (hi - lo) ≤ 2 * (hi + lo - 2) := by
    rw [div_mul_eq_mul_div, div_le_iff₀ hA0]; linarith
  have h1 : (hi - 1) * (D / A * (1 + lo)) ≤ (hi - 1) * (2 * (1 - lo)) := by
    calc (hi - 1) * (D / A * (1 + lo)) = D / A * (hi - lo) := by rw [e1]; ring
      _ ≤ 2 * (hi + lo - 2) := hDA
      _ = (hi - 1) * (2 * (1 - lo)) := e2
  have h2 := le_of_mul_le_mul_left h1 (by linarith)
  rw [le_div_iff₀ (by linarith)]
  have h3 := mul_le_mul_of_nonneg_left h2 (by linarith : (0 : ℝ) ≤ hi)
  have e3 : hi * (D / A * (1 + lo)) = D / A * (hi + 1) := by linear_combination D / A * hhilo
  have e4 : hi * (2 * (1 - lo)) = 2 * (hi - 1) := by linear_combination -2 * hhilo
  linarith

/-- Scaled dot-product attention of the query `q` over the keys `s` with scale `τ`. -/
noncomputable def attention (τ : ℝ) (s : Finset ν) (q : ι → ℝ) (k : ν → ι → ℝ) (v : ν → κ → ℝ)
    (c : κ) : ℝ :=
  ∑ j ∈ s, softmax s (fun j => τ * ∑ i, q i * k j i) j * v j c

/-- Exact attention moves by at most `ev + 2 * (exp t - 1) / (exp t + 1) * V` when the query,
keys and values move by at most `eq`, `ek` and `ev` and `t` bounds the change of the scores. -/
theorem sqrt_sum_sq_attention_sub_le [Fintype κ] {τ : ℝ} (hτ : 0 ≤ τ) {s : Finset ν}
    (hs : s.Nonempty) {q q' : ι → ℝ} {k k' : ν → ι → ℝ} {v v' : ν → κ → ℝ}
    {Q K V eq ek ev t : ℝ} (hq : √(∑ i, q i ^ 2) ≤ Q) (hk : ∀ j ∈ s, √(∑ i, k j i ^ 2) ≤ K)
    (hv : ∀ j ∈ s, √(∑ c, v j c ^ 2) ≤ V) (heq : √(∑ i, (q' i - q i) ^ 2) ≤ eq)
    (hek : ∀ j ∈ s, √(∑ i, (k' j i - k j i) ^ 2) ≤ ek)
    (hev : ∀ j ∈ s, √(∑ c, (v' j c - v j c) ^ 2) ≤ ev)
    (ht : τ * (eq * (K + ek) + Q * ek) ≤ t) :
    √(∑ c, (attention τ s q' k' v' c - attention τ s q k v c) ^ 2) ≤
      ev + 2 * (Real.exp t - 1) / (Real.exp t + 1) * V := by
  set p' := softmax s (fun j => τ * ∑ i, q' i * k' j i)
  set p := softmax s (fun j => τ * ∑ i, q i * k j i)
  have hscore : ∀ j ∈ s, |τ * ∑ i, q' i * k' j i - τ * ∑ i, q i * k j i| ≤ t := fun j hj => by
    have hsplit : τ * ∑ i, q' i * k' j i - τ * ∑ i, q i * k j i =
        τ * (∑ i, (q' i - q i) * k' j i + ∑ i, q i * (k' j i - k j i)) := by
      rw [← Finset.sum_add_distrib, ← mul_sub, ← Finset.sum_sub_distrib]
      exact congrArg _ (Finset.sum_congr rfl fun i _ => by ring)
    have hk' : √(∑ i, k' j i ^ 2) ≤ K + ek := by
      have : √(∑ i, k' j i ^ 2) ≤ √(∑ i, k j i ^ 2) + √(∑ i, (k' j i - k j i) ^ 2) := by
        simpa [EuclideanSpace.norm_eq] using
          norm_le_norm_add_norm_sub' (WithLp.toLp 2 (k' j)) (WithLp.toLp 2 (k j))
      linarith [hk j hj, hek j hj]
    have h1 : |∑ i, (q' i - q i) * k' j i| ≤ √(∑ i, (q' i - q i) ^ 2) * √(∑ i, k' j i ^ 2) :=
      (Finset.abs_sum_le_sum_abs _ _).trans (by
        simpa [abs_mul, sq_abs] using
          Real.sum_mul_le_sqrt_mul_sqrt Finset.univ (fun i => |q' i - q i|) fun i => |k' j i|)
    have h2 : |∑ i, q i * (k' j i - k j i)| ≤ √(∑ i, q i ^ 2) * √(∑ i, (k' j i - k j i) ^ 2) :=
      (Finset.abs_sum_le_sum_abs _ _).trans (by
        simpa [abs_mul, sq_abs] using
          Real.sum_mul_le_sqrt_mul_sqrt Finset.univ (fun i => |q i|) fun i => |k' j i - k j i|)
    have h3 : √(∑ i, (q' i - q i) ^ 2) * √(∑ i, k' j i ^ 2) ≤ eq * (K + ek) :=
      mul_le_mul heq hk' (Real.sqrt_nonneg _) ((Real.sqrt_nonneg _).trans heq)
    have h4 : √(∑ i, q i ^ 2) * √(∑ i, (k' j i - k j i) ^ 2) ≤ Q * ek :=
      mul_le_mul hq (hek j hj) (Real.sqrt_nonneg _) ((Real.sqrt_nonneg _).trans hq)
    rw [hsplit, abs_mul, abs_of_nonneg hτ]
    refine le_trans ?_ ht
    refine mul_le_mul_of_nonneg_left ?_ hτ
    refine (abs_add_le _ _).trans ?_
    linarith
  have hl1 := sum_abs_softmax_sub_le hs hscore
  have hdiff : ∀ c, attention τ s q' k' v' c - attention τ s q k v c =
      ∑ j ∈ s, p' j * (v' j c - v j c) + ∑ j ∈ s, (p' j - p j) * v j c := fun c => by
    simp only [attention, ← Finset.sum_add_distrib, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun j _ => by ring
  have hV0 : 0 ≤ V := (Real.sqrt_nonneg _).trans (hv _ hs.choose_spec)
  have hmink : ∀ (r : ν → ℝ) (y : ν → κ → ℝ),
      √(∑ c, (∑ j ∈ s, r j * y j c) ^ 2) ≤ ∑ j ∈ s, |r j| * √(∑ c, y j c ^ 2) := fun r y => by
    have hsum : WithLp.toLp 2 (fun c => ∑ j ∈ s, r j * y j c) =
        ∑ j ∈ s, r j • WithLp.toLp 2 (y j) := by
      ext c; simp [WithLp.ofLp_sum, smul_eq_mul]
    have h := norm_sum_le s fun j => r j • WithLp.toLp 2 (y j)
    rw [← hsum] at h
    simp only [norm_smul, Real.norm_eq_abs] at h
    simpa [EuclideanSpace.norm_eq] using h
  have hA : √(∑ c, (∑ j ∈ s, p' j * (v' j c - v j c)) ^ 2) ≤ ev := by
    refine (hmink p' fun j c => v' j c - v j c).trans ?_
    calc ∑ j ∈ s, |p' j| * √(∑ c, (v' j c - v j c) ^ 2) ≤ ∑ j ∈ s, p' j * ev :=
          Finset.sum_le_sum fun j hj => by
            rw [abs_of_nonneg (softmax_nonneg _ _ _)]
            exact mul_le_mul_of_nonneg_left (hev j hj) (softmax_nonneg _ _ _)
      _ = ev := by rw [← Finset.sum_mul, sum_softmax hs, one_mul]
  have hB : √(∑ c, (∑ j ∈ s, (p' j - p j) * v j c) ^ 2) ≤
      2 * (Real.exp t - 1) / (Real.exp t + 1) * V := by
    refine (hmink (fun j => p' j - p j) v).trans ?_
    calc ∑ j ∈ s, |p' j - p j| * √(∑ c, v j c ^ 2) ≤ ∑ j ∈ s, |p' j - p j| * V :=
          Finset.sum_le_sum fun j hj => mul_le_mul_of_nonneg_left (hv j hj) (abs_nonneg _)
      _ ≤ _ := by rw [← Finset.sum_mul]; exact mul_le_mul_of_nonneg_right hl1 hV0
  simp only [hdiff]
  have : √(∑ c, (∑ j ∈ s, p' j * (v' j c - v j c) + ∑ j ∈ s, (p' j - p j) * v j c) ^ 2) ≤
      √(∑ c, (∑ j ∈ s, p' j * (v' j c - v j c)) ^ 2) +
        √(∑ c, (∑ j ∈ s, (p' j - p j) * v j c) ^ 2) := by
    simpa [EuclideanSpace.norm_eq] using
      norm_add_le (WithLp.toLp 2 fun c => ∑ j ∈ s, p' j * (v' j c - v j c))
        (WithLp.toLp 2 fun c => ∑ j ∈ s, (p' j - p j) * v j c)
  linarith

/-- Attention evaluated in `A`: each score is a dot product with rounded products, scaled by `τ`
and rounded, and each weight is the computed exponential of the rounded difference from the
largest score. The weighted sum of the values is divided by the sum of the weights, rounded and
stored. -/
noncomputable def Arithmetic.attention (A : Arithmetic) (τ : ℝ) {s : Finset ν} (hs : s.Nonempty)
    (q : ι → ℝ) (k : ν → ι → ℝ) (v : ν → κ → ℝ) : κ → ℝ :=
  let a := fun j => A.round (τ * A.sum ((Finset.univ : Finset ι).toList.map
    fun i => A.roundMul (q i * k j i)))
  let w := fun j => A.exp (A.round (a j - s.sup' hs a))
  fun c => A.store (A.round (A.sum (s.toList.map fun j => A.roundMul (w j * v j c)) /
    A.sum (s.toList.map w)))

/-- In exact arithmetic, `Arithmetic.attention` is `TransformerBlock.attention`. -/
theorem Arithmetic.exact_attention (act : ℝ → ℝ) (τ : ℝ) {s : Finset ν} (hs : s.Nonempty)
    (q : ι → ℝ) (k : ν → ι → ℝ) (v : ν → κ → ℝ) :
    (Arithmetic.exact act).attention τ hs q k v = TransformerBlock.attention τ s q k v := by
  funext c
  simp only [Arithmetic.attention, Arithmetic.exact, id, Finset.sum_map_toList,
    TransformerBlock.attention]
  rw [Finset.sum_div]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [softmax_eq_exp_sub_div s _ (s.sup' hs fun j => τ * ∑ i, q i * k j i)]
  ring

/-- Summing rounded products in `S` has the summation error of the products plus their
rounding errors. -/
theorem abs_sum_roundMul_sub_le {α : Type*} {S : List ℝ → ℝ} {fm : ℝ → ℝ} {g h : ℕ → ℝ}
    {um ηm : ℝ} (hS : ∀ l : List ℝ, |S l - l.sum| ≤ g l.length * (l.map abs).sum + h l.length)
    (hmul : ∀ t, |fm t - t| ≤ um * |t| + ηm) (s : Finset α) (hg : 0 ≤ g s.card) (x y : α → ℝ) :
    |S (s.toList.map fun j => fm (x j * y j)) - ∑ j ∈ s, x j * y j| ≤
      (g s.card * (1 + um) + um) * ∑ j ∈ s, |x j| * |y j| + (1 + g s.card) * s.card * ηm +
        h s.card := by
  have hsum := abs_sum_toList_sub_le hS s fun j => fm (x j * y j)
  have hp : ∀ j, |fm (x j * y j) - x j * y j| ≤ um * (|x j| * |y j|) + ηm := fun j => by
    simpa [abs_mul] using hmul (x j * y j)
  have hpabs : ∑ j ∈ s, |fm (x j * y j)| ≤ (1 + um) * ∑ j ∈ s, |x j| * |y j| + s.card * ηm := by
    have : ∀ j, |fm (x j * y j)| ≤ (1 + um) * (|x j| * |y j|) + ηm := fun j => by
      have h1 := hp j
      have h2 := abs_sub_abs_le_abs_sub (fm (x j * y j)) (x j * y j)
      rw [abs_mul] at h2
      linarith
    calc _ ≤ ∑ j ∈ s, ((1 + um) * (|x j| * |y j|) + ηm) := Finset.sum_le_sum fun j _ => this j
      _ = _ := by rw [Finset.sum_add_distrib, ← Finset.mul_sum]; simp
  have hpsum : |∑ j ∈ s, fm (x j * y j) - ∑ j ∈ s, x j * y j| ≤
      um * ∑ j ∈ s, |x j| * |y j| + s.card * ηm := by
    rw [← Finset.sum_sub_distrib]
    calc _ ≤ ∑ j ∈ s, |fm (x j * y j) - x j * y j| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ j ∈ s, (um * (|x j| * |y j|) + ηm) := Finset.sum_le_sum fun j _ => hp j
      _ = _ := by rw [Finset.sum_add_distrib, ← Finset.mul_sum]; simp
  have h3 := mul_le_mul_of_nonneg_left hpabs hg
  have h4 := abs_sub_le (S (s.toList.map fun j => fm (x j * y j))) (∑ j ∈ s, fm (x j * y j))
    (∑ j ∈ s, x j * y j)
  nlinarith

/-- Rounding then storing each entry of `z` leaves the Euclidean error
`(us + (1 + us) * u) * ‖y‖ + (1 + us) * (1 + u) * ‖z - y‖ + √card * ((1 + us) * η + ηs)`. -/
theorem sqrt_sum_sq_store_round_sub_le {f fs : ℝ → ℝ} {u η us ηs : ℝ} (hu : 0 ≤ u) (hη : 0 ≤ η)
    (hus : 0 ≤ us) (hηs : 0 ≤ ηs) (hf : ∀ t, |f t - t| ≤ u * |t| + η)
    (hfs : ∀ t, |fs t - t| ≤ us * |t| + ηs) (z y : ι → ℝ) :
    √(∑ i, (fs (f (z i)) - y i) ^ 2) ≤ (us + (1 + us) * u) * √(∑ i, y i ^ 2) +
      (1 + us) * (1 + u) * √(∑ i, (z i - y i) ^ 2) + √(Fintype.card ι) * ((1 + us) * η + ηs) := by
  refine sqrt_sum_sq_le_mul_add_mul_add (by positivity) (by positivity) (by positivity) fun i => ?_
  have h1 := ((MixedError.mk hu hη (hf _)).trans_of_abs_sub_le (le_refl |z i - y i|)).bound
  have h2 := ((MixedError.mk hus hηs (hfs _)).trans_of_abs_sub_le h1).bound
  linarith

private theorem abs_score_sub_le (A : Arithmetic) {g h : ℕ → ℝ} {u η um ηm τ Q K es : ℝ}
    (hS : ∀ l : List ℝ, |A.sum l - l.sum| ≤ g l.length * (l.map abs).sum + h l.length)
    (hu : 0 ≤ u) (hround : ∀ t, |A.round t - t| ≤ u * |t| + η)
    (hmul : ∀ t, |A.roundMul t - t| ≤ um * |t| + ηm) (hum : 0 ≤ um)
    (hg : 0 ≤ g (Fintype.card ι)) (hτ : 0 ≤ τ) {q y : ι → ℝ} (hq : √(∑ i, q i ^ 2) ≤ Q)
    (hy : √(∑ i, y i ^ 2) ≤ K)
    (hes : u * τ * (Q * K) + (1 + u) * τ * ((g (Fintype.card ι) * (1 + um) + um) * (Q * K) +
      (1 + g (Fintype.card ι)) * Fintype.card ι * ηm + h (Fintype.card ι)) + η ≤ es) :
    |A.round (τ * A.sum ((Finset.univ : Finset ι).toList.map fun i => A.roundMul (q i * y i))) -
      τ * ∑ i, q i * y i| ≤ es ∧ |τ * ∑ i, q i * y i| ≤ τ * (Q * K) := by
  have hdot := abs_sum_roundMul_sub_le hS hmul Finset.univ (by rwa [Finset.card_univ]) q y
  rw [Finset.card_univ] at hdot
  have hQK : ∑ i, |q i| * |y i| ≤ Q * K :=
    (by simpa [sq_abs] using Real.sum_mul_le_sqrt_mul_sqrt Finset.univ (|q ·|) (|y ·|) :
      ∑ i, |q i| * |y i| ≤ √(∑ i, q i ^ 2) * √(∑ i, y i ^ 2)).trans
      (mul_le_mul hq hy (Real.sqrt_nonneg _) ((Real.sqrt_nonneg _).trans hq))
  have hσ : |τ * ∑ i, q i * y i| ≤ τ * (Q * K) := by
    rw [abs_mul, abs_of_nonneg hτ]
    have h : |∑ i, q i * y i| ≤ ∑ i, |q i| * |y i| :=
      (Finset.abs_sum_le_sum_abs _ _).trans_eq (Finset.sum_congr rfl fun i _ => abs_mul _ _)
    exact mul_le_mul_of_nonneg_left (h.trans hQK) hτ
  refine ⟨?_, hσ⟩
  have h1 : |τ * A.sum ((Finset.univ : Finset ι).toList.map fun i => A.roundMul (q i * y i)) -
      τ * ∑ i, q i * y i| ≤ τ * ((g (Fintype.card ι) * (1 + um) + um) * (Q * K) +
        (1 + g (Fintype.card ι)) * Fintype.card ι * ηm + h (Fintype.card ι)) := by
    rw [← mul_sub, abs_mul, abs_of_nonneg hτ]
    refine mul_le_mul_of_nonneg_left (hdot.trans ?_) hτ
    have := mul_le_mul_of_nonneg_left hQK (by positivity : 0 ≤ g (Fintype.card ι) * (1 + um) + um)
    linarith
  have hη : 0 ≤ η := by simpa using (abs_nonneg _).trans (hround 0)
  have h2 := ((MixedError.mk hu hη (hround _)).trans_of_abs_sub_le h1).bound
  have h3 := mul_le_mul_of_nonneg_left hσ hu
  nlinarith

private theorem abs_exp_round_sub_le {fr fe : ℝ → ℝ} {u η ue ηe a σ m σ0 es T δ θ : ℝ}
    (hu : 0 ≤ u) (hue : 0 ≤ ue) (hround : ∀ t, |fr t - t| ≤ u * |t| + η)
    (hexp : ∀ t, |fe t - Real.exp t| ≤ ue * Real.exp t + ηe) (ha : |a - σ| ≤ es)
    (hσ : |σ| ≤ T) (hm : |m - σ0| ≤ es) (hσ0 : |σ0| ≤ T) (hδ : u * (2 * T + 2 * es) + es + η ≤ δ)
    (hθ : (1 + ue) * Real.exp δ - 1 ≤ θ) :
    |fe (fr (a - m)) - Real.exp (σ - m)| ≤ θ * Real.exp (σ - m) + ηe := by
  have ham : |a - m| ≤ 2 * T + 2 * es := by
    have := abs_sub_le a σ m
    have := abs_sub_le σ σ0 m
    have := abs_sub (σ) σ0
    rw [abs_sub_comm m σ0] at hm
    linarith
  have hb : |fr (a - m) - (σ - m)| ≤ δ := by
    have h1 := hround (a - m)
    have h2 := mul_le_mul_of_nonneg_left ham hu
    have h3 : |fr (a - m) - (σ - m)| ≤ |fr (a - m) - (a - m)| + |a - σ| := by
      have := abs_add_le (fr (a - m) - (a - m)) (a - σ); ring_nf at this ⊢; linarith
    linarith
  set b := fr (a - m)
  set E := Real.exp (σ - m)
  have hE : 0 < E := Real.exp_pos _
  have hbE : Real.exp b = E * Real.exp (b - (σ - m)) := by rw [← Real.exp_add]; ring_nf
  have hd : |Real.exp (b - (σ - m)) - 1| ≤ Real.exp δ - 1 :=
    (FloatLib.Numerics.abs_exp_sub_one_le_exp_abs_sub_one _).trans
      (by linarith [Real.exp_le_exp.2 hb])
  have he : Real.exp (b - (σ - m)) ≤ Real.exp δ :=
    Real.exp_le_exp.2 ((le_abs_self _).trans hb)
  have h1 := hexp b
  have h2 : |Real.exp b - E| ≤ E * (Real.exp δ - 1) := by
    rw [hbE, show E * Real.exp (b - (σ - m)) - E = E * (Real.exp (b - (σ - m)) - 1) by ring,
      abs_mul, abs_of_pos hE]
    exact mul_le_mul_of_nonneg_left hd hE.le
  have h3 : ue * Real.exp b ≤ ue * (E * Real.exp δ) := by
    rw [hbE]; exact mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left he hE.le) hue
  have h4 := abs_sub_le (fe b) (Real.exp b) E
  have h5 : E * ((1 + ue) * Real.exp δ - 1) ≤ E * θ := mul_le_mul_of_nonneg_left hθ hE.le
  nlinarith

private theorem abs_sum_sub_sum_le_mul {α : Type*} {S : List ℝ → ℝ} {g h : ℕ → ℝ}
    (hS : ∀ l : List ℝ, |S l - l.sum| ≤ g l.length * (l.map abs).sum + h l.length)
    {s : Finset α} {w E : α → ℝ} {θ ηe gN hN N es eL : ℝ}
    (hg : 0 ≤ g s.card) (hgN : g s.card ≤ gN) (hhN : h s.card ≤ hN) (hcard : (s.card : ℝ) ≤ N)
    (hh : 0 ≤ h s.card) (hηe : 0 ≤ ηe) (hθ : 0 ≤ θ) (hE : ∀ j ∈ s, 0 ≤ E j)
    (hw : ∀ j ∈ s, |w j - E j| ≤ θ * E j + ηe) (hL : 1 ≤ Real.exp es * ∑ j ∈ s, E j)
    (heL : gN * (1 + θ) + θ + ((1 + gN) * N * ηe + hN) * Real.exp es ≤ eL) :
    |S (s.toList.map w) - ∑ j ∈ s, E j| ≤ eL * ∑ j ∈ s, E j := by
  have hsum := abs_sum_toList_sub_le hS s w
  have hwabs : ∑ j ∈ s, |w j| ≤ (1 + θ) * ∑ j ∈ s, E j + s.card * ηe := by
    calc _ ≤ ∑ j ∈ s, ((1 + θ) * E j + ηe) := Finset.sum_le_sum fun j hj => by
          have := abs_sub_abs_le_abs_sub (w j) (E j)
          rw [abs_of_nonneg (hE j hj)] at this
          linarith [hw j hj]
      _ = _ := by rw [Finset.sum_add_distrib, ← Finset.mul_sum]; simp
  have hdiff : |∑ j ∈ s, w j - ∑ j ∈ s, E j| ≤ θ * ∑ j ∈ s, E j + s.card * ηe := by
    rw [← Finset.sum_sub_distrib]
    calc _ ≤ ∑ j ∈ s, |w j - E j| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ j ∈ s, (θ * E j + ηe) := Finset.sum_le_sum hw
      _ = _ := by rw [Finset.sum_add_distrib, ← Finset.mul_sum]; simp
  set L := ∑ j ∈ s, E j
  have hL0 : 0 ≤ L := Finset.sum_nonneg hE
  have hgN0 : 0 ≤ gN := hg.trans hgN
  have hN0 : 0 ≤ N := (Nat.cast_nonneg _).trans hcard
  have e1 : g s.card * ((1 + θ) * L + s.card * ηe) ≤ gN * ((1 + θ) * L + N * ηe) :=
    mul_le_mul hgN (by gcongr) (by positivity) hgN0
  have e2 : (s.card : ℝ) * ηe ≤ N * ηe := mul_le_mul_of_nonneg_right hcard hηe
  have hC : 0 ≤ (1 + gN) * N * ηe + hN := by have := hh.trans hhN; positivity
  have e3 : (1 + gN) * N * ηe + hN ≤ ((1 + gN) * N * ηe + hN) * Real.exp es * L := by
    have := mul_le_mul_of_nonneg_left hL hC
    linarith
  have e4 := mul_le_mul_of_nonneg_right heL hL0
  have h1 := mul_le_mul_of_nonneg_left hwabs hg
  have h2 := abs_sub_le (S (s.toList.map w)) (∑ j ∈ s, w j) L
  nlinarith

private theorem sqrt_sum_sq_sum_mul_sub_le [Fintype κ] {α : Type*} {S : List ℝ → ℝ}
    {fm : ℝ → ℝ} {g h : ℕ → ℝ} {um ηm : ℝ}
    (hS : ∀ l : List ℝ, |S l - l.sum| ≤ g l.length * (l.map abs).sum + h l.length)
    (hum : 0 ≤ um) (hηm : 0 ≤ ηm) (hmul : ∀ t, |fm t - t| ≤ um * |t| + ηm)
    {s : Finset α} {w E : α → ℝ} {v : α → κ → ℝ} {θ ηe gN hN N es V eN : ℝ}
    (hg : 0 ≤ g s.card) (hgN : g s.card ≤ gN) (hhN : h s.card ≤ hN) (hcard : (s.card : ℝ) ≤ N)
    (hh : 0 ≤ h s.card) (hηe : 0 ≤ ηe) (hE : ∀ j ∈ s, 0 ≤ E j)
    (hw : ∀ j ∈ s, |w j - E j| ≤ θ * E j + ηe) (hL : 1 ≤ Real.exp es * ∑ j ∈ s, E j)
    (hv : ∀ j ∈ s, √(∑ c, v j c ^ 2) ≤ V) (hV : 0 ≤ V)
    (heN : ((gN * (1 + um) + um) * (1 + θ) + θ) * V + Real.exp es * ((gN * (1 + um) + um + 1) *
      V * N * ηe + √(Fintype.card κ) * ((1 + gN) * N * ηm + hN)) ≤ eN) :
    √(∑ c, (S (s.toList.map fun j => fm (w j * v j c)) - ∑ j ∈ s, E j * v j c) ^ 2) ≤
      eN * ∑ j ∈ s, E j := by
  set a := g s.card * (1 + um) + um
  set c0 := (1 + g s.card) * s.card * ηm + h s.card
  set L := ∑ j ∈ s, E j
  have hterm : ∀ c, |S (s.toList.map fun j => fm (w j * v j c)) - ∑ j ∈ s, E j * v j c| ≤
      a * |(∑ j ∈ s, |w j| * |v j c|)| + 1 * |(∑ j ∈ s, |w j - E j| * |v j c|)| + c0 :=
      fun c => by
    have h1 := abs_sum_roundMul_sub_le hS hmul s hg w fun j => v j c
    have h2 : |∑ j ∈ s, w j * v j c - ∑ j ∈ s, E j * v j c| ≤ ∑ j ∈ s, |w j - E j| * |v j c| := by
      rw [← Finset.sum_sub_distrib]
      refine (Finset.abs_sum_le_sum_abs _ _).trans (le_of_eq ?_)
      exact Finset.sum_congr rfl fun j _ => by rw [← sub_mul, abs_mul]
    have h3 := abs_sub_le (S (s.toList.map fun j => fm (w j * v j c))) (∑ j ∈ s, w j * v j c)
      (∑ j ∈ s, E j * v j c)
    have y0 : |(∑ j ∈ s, |w j| * |v j c|)| = ∑ j ∈ s, |w j| * |v j c| :=
      abs_of_nonneg (Finset.sum_nonneg fun j _ => by positivity)
    have z0 : |(∑ j ∈ s, |w j - E j| * |v j c|)| = ∑ j ∈ s, |w j - E j| * |v j c| :=
      abs_of_nonneg (Finset.sum_nonneg fun j _ => by positivity)
    rw [y0, z0]
    linarith
  have ha : 0 ≤ a := by positivity
  have hc0 : 0 ≤ c0 := by positivity
  have hmain := sqrt_sum_sq_le_mul_add_mul_add ha zero_le_one hc0 hterm
  have hwabs : ∑ j ∈ s, |w j| ≤ (1 + θ) * L + s.card * ηe := by
    calc _ ≤ ∑ j ∈ s, ((1 + θ) * E j + ηe) := Finset.sum_le_sum fun j hj => by
          have := abs_sub_abs_le_abs_sub (w j) (E j)
          rw [abs_of_nonneg (hE j hj)] at this
          linarith [hw j hj]
      _ = _ := by rw [Finset.sum_add_distrib, ← Finset.mul_sum]; simp [L]
  have hwE : ∑ j ∈ s, |w j - E j| ≤ θ * L + s.card * ηe := by
    calc _ ≤ ∑ j ∈ s, (θ * E j + ηe) := Finset.sum_le_sum hw
      _ = _ := by rw [Finset.sum_add_distrib, ← Finset.mul_sum]; simp [L]
  have hmink : ∀ (r : α → ℝ) (y : α → κ → ℝ),
      √(∑ c, (∑ j ∈ s, r j * y j c) ^ 2) ≤ ∑ j ∈ s, |r j| * √(∑ c, y j c ^ 2) := fun r y => by
    have hsum : WithLp.toLp 2 (fun c => ∑ j ∈ s, r j * y j c) =
        ∑ j ∈ s, r j • WithLp.toLp 2 (y j) := by
      ext c; simp [WithLp.ofLp_sum, smul_eq_mul]
    have h := norm_sum_le s fun j => r j • WithLp.toLp 2 (y j)
    rw [← hsum] at h
    simp only [norm_smul, Real.norm_eq_abs] at h
    simpa [EuclideanSpace.norm_eq] using h
  have hy : √(∑ c, (∑ j ∈ s, |w j| * |v j c|) ^ 2) ≤ V * ((1 + θ) * L + s.card * ηe) := by
    refine (hmink (fun j => |w j|) fun j c => |v j c|).trans ?_
    calc ∑ j ∈ s, |(|w j|)| * √(∑ c, |v j c| ^ 2) ≤ ∑ j ∈ s, |w j| * V :=
          Finset.sum_le_sum fun j hj => by
            simp only [abs_abs, sq_abs]
            exact mul_le_mul_of_nonneg_left (hv j hj) (abs_nonneg _)
      _ ≤ V * ((1 + θ) * L + s.card * ηe) := by
          rw [← Finset.sum_mul, mul_comm]; exact mul_le_mul_of_nonneg_left hwabs hV
  have hz : √(∑ c, (∑ j ∈ s, |w j - E j| * |v j c|) ^ 2) ≤ V * (θ * L + s.card * ηe) := by
    refine (hmink (fun j => |w j - E j|) fun j c => |v j c|).trans ?_
    calc ∑ j ∈ s, |(|w j - E j|)| * √(∑ c, |v j c| ^ 2) ≤ ∑ j ∈ s, |w j - E j| * V :=
          Finset.sum_le_sum fun j hj => by
            simp only [abs_abs, sq_abs]
            exact mul_le_mul_of_nonneg_left (hv j hj) (abs_nonneg _)
      _ ≤ V * (θ * L + s.card * ηe) := by
          rw [← Finset.sum_mul, mul_comm]; exact mul_le_mul_of_nonneg_left hwE hV
  have hL0 : 0 ≤ L := Finset.sum_nonneg hE
  have hgN0 : 0 ≤ gN := hg.trans hgN
  have hN0 : 0 ≤ N := (Nat.cast_nonneg _).trans hcard
  have haN : a ≤ gN * (1 + um) + um := by simp only [a]; gcongr
  have hcard : (s.card : ℝ) * ηe ≤ N * ηe := mul_le_mul_of_nonneg_right hcard hηe
  have t1 : a * √(∑ c, (∑ j ∈ s, |w j| * |v j c|) ^ 2) ≤
      (gN * (1 + um) + um) * (V * ((1 + θ) * L + N * ηe)) :=
    mul_le_mul haN (hy.trans (by gcongr)) (Real.sqrt_nonneg _) (by positivity)
  have t2 : √(∑ c, (∑ j ∈ s, |w j - E j| * |v j c|) ^ 2) ≤ V * (θ * L + N * ηe) :=
    hz.trans (by gcongr)
  have t3 : c0 ≤ (1 + gN) * N * ηm + hN := by
    simp only [c0]
    have : (1 + g s.card) * s.card * ηm ≤ (1 + gN) * N * ηm := by gcongr
    linarith
  set C := (gN * (1 + um) + um + 1) * V * N * ηe + √(Fintype.card κ) * ((1 + gN) * N * ηm + hN)
  have hC : 0 ≤ C := by have := hh.trans hhN; positivity
  have t4 : C ≤ Real.exp es * C * L := by
    have := mul_le_mul_of_nonneg_left hL hC; linarith
  have t5 := mul_le_mul_of_nonneg_right heN hL0
  have t6 := mul_le_mul_of_nonneg_left t3 (Real.sqrt_nonneg (Fintype.card κ : ℝ))
  have t7 : (gN * (1 + um) + um) * (V * ((1 + θ) * L + N * ηe)) + V * (θ * L + N * ηe) +
      √(Fintype.card κ) * ((1 + gN) * N * ηm + hN) =
      ((gN * (1 + um) + um) * (1 + θ) + θ) * V * L + C := by simp only [C]; ring
  have t8 : (((gN * (1 + um) + um) * (1 + θ) + θ) * V + Real.exp es * C) * L =
      ((gN * (1 + um) + um) * (1 + θ) + θ) * V * L + Real.exp es * C * L := by ring
  linarith

private theorem sqrt_sum_sq_div_sub_div_le [Fintype κ] {N N' : κ → ℝ} {L L' eN eL V : ℝ}
    (hL' : 0 < L') (hN : √(∑ c, (N c - N' c) ^ 2) ≤ eN * L') (hL : |L - L'| ≤ eL * L')
    (hN' : √(∑ c, N' c ^ 2) ≤ V * L') (heL : eL < 1) :
    √(∑ c, (N c / L - N' c / L') ^ 2) ≤ (eN + eL * V) / (1 - eL) := by
  have hLlo : (1 - eL) * L' ≤ L := by have := neg_abs_le (L - L'); nlinarith
  have hL0 : 0 < L := lt_of_lt_of_le (mul_pos (by linarith) hL') hLlo
  have heN : 0 ≤ eN := nonneg_of_mul_nonneg_left ((Real.sqrt_nonneg _).trans hN) hL'
  have heL0 : 0 ≤ eL := nonneg_of_mul_nonneg_left ((abs_nonneg _).trans hL) hL'
  have hV : 0 ≤ V := nonneg_of_mul_nonneg_left ((Real.sqrt_nonneg _).trans hN') hL'
  have hterm : ∀ c, |N c / L - N' c / L'| ≤
      1 / L * |N c - N' c| + |L' - L| / (L * L') * |N' c| + 0 := fun c => by
    have hid : N c / L - N' c / L' = (N c - N' c) / L + N' c * (L' - L) / (L * L') := by
      field_simp; ring
    rw [hid, add_zero]
    refine (abs_add_le _ _).trans (le_of_eq ?_)
    rw [abs_div, abs_of_pos hL0, abs_div, abs_mul, abs_of_pos (mul_pos hL0 hL')]
    ring
  have hmain := sqrt_sum_sq_le_mul_add_mul_add (by positivity) (by positivity) le_rfl hterm
  have e1 : 1 / L * √(∑ c, (N c - N' c) ^ 2) ≤ 1 / L * (eN * L') :=
    mul_le_mul_of_nonneg_left hN (by positivity)
  have e2 : |L' - L| / (L * L') * √(∑ c, N' c ^ 2) ≤ eL * L' / (L * L') * (V * L') := by
    rw [abs_sub_comm]
    exact mul_le_mul (div_le_div_of_nonneg_right hL (by positivity)) hN' (Real.sqrt_nonneg _)
      (by positivity)
  have e3 : 1 / L * (eN * L') + eL * L' / (L * L') * (V * L') = (eN + eL * V) * (L' / L) := by
    field_simp
  have e4 : (eN + eL * V) * (L' / L) ≤ (eN + eL * V) / (1 - eL) := by
    have hX : 0 ≤ eN + eL * V := by positivity
    rw [mul_div_assoc', div_le_div_iff₀ hL0 (by linarith)]
    nlinarith [mul_le_mul_of_nonneg_left hLlo hX]
  simp only [mul_zero, add_zero] at hmain
  linarith

/-- Attention evaluated in `A` on the query `q`, keys `k` and values `v` is within the stated
Euclidean distance of exact attention on the same inputs. Here `es`, `δ` and `θ` bound the errors
of the scores, of the shifted scores and of the weights, and `eL` and `eN` are the relative errors
of the normalizer and of the weighted sum. -/
theorem Arithmetic.sqrt_sum_sq_attention_sub_le [Fintype κ] (A : Arithmetic) {g h : ℕ → ℝ}
    {u η um ηm us ηs ue ηe : ℝ}
    (hS : ∀ l : List ℝ, |A.sum l - l.sum| ≤ g l.length * (l.map abs).sum + h l.length)
    (hu : 0 ≤ u) (hη : 0 ≤ η) (hum : 0 ≤ um) (hηm : 0 ≤ ηm) (hus : 0 ≤ us) (hηs : 0 ≤ ηs)
    (hue : 0 ≤ ue) (hηe : 0 ≤ ηe) (hround : ∀ t, |A.round t - t| ≤ u * |t| + η)
    (hmul : ∀ t, |A.roundMul t - t| ≤ um * |t| + ηm)
    (hstore : ∀ t, |A.store t - t| ≤ us * |t| + ηs)
    (hexp : ∀ t, |A.exp t - Real.exp t| ≤ ue * Real.exp t + ηe)
    {τ : ℝ} (hτ : 0 ≤ τ) {s : Finset ν} (hs : s.Nonempty) {gN hN N : ℝ}
    (hg : 0 ≤ g (Fintype.card ι)) (hgs : 0 ≤ g s.card) (hhs : 0 ≤ h s.card)
    (hgN : g s.card ≤ gN) (hhN : h s.card ≤ hN) (hcard : (s.card : ℝ) ≤ N)
    {q : ι → ℝ} {k : ν → ι → ℝ} {v : ν → κ → ℝ} {Q K V es δ θ eL eN : ℝ}
    (hq : √(∑ i, q i ^ 2) ≤ Q) (hk : ∀ j ∈ s, √(∑ i, k j i ^ 2) ≤ K)
    (hv : ∀ j ∈ s, √(∑ c, v j c ^ 2) ≤ V)
    (hes : u * τ * (Q * K) + (1 + u) * τ * ((g (Fintype.card ι) * (1 + um) + um) * (Q * K) +
      (1 + g (Fintype.card ι)) * Fintype.card ι * ηm + h (Fintype.card ι)) + η ≤ es)
    (hδ : u * (2 * (τ * (Q * K)) + 2 * es) + es + η ≤ δ)
    (hθ : (1 + ue) * Real.exp δ - 1 ≤ θ)
    (heL : gN * (1 + θ) + θ + ((1 + gN) * N * ηe + hN) * Real.exp es ≤ eL) (heL1 : eL < 1)
    (heN : ((gN * (1 + um) + um) * (1 + θ) + θ) * V + Real.exp es * ((gN * (1 + um) + um + 1) *
      V * N * ηe + √(Fintype.card κ) * ((1 + gN) * N * ηm + hN)) ≤ eN) :
    √(∑ c, (A.attention τ hs q k v c - TransformerBlock.attention τ s q k v c) ^ 2) ≤
      (us + (1 + us) * u) * V + (1 + us) * (1 + u) * ((eN + eL * V) / (1 - eL)) +
        √(Fintype.card κ) * ((1 + us) * η + ηs) := by
  obtain ⟨a, ha⟩ : ∃ a : ν → ℝ, a = fun j => A.round (τ * A.sum
      ((Finset.univ : Finset ι).toList.map fun i => A.roundMul (q i * k j i))) := ⟨_, rfl⟩
  obtain ⟨w, hw⟩ : ∃ w : ν → ℝ, w = fun j => A.exp (A.round (a j - s.sup' hs a)) := ⟨_, rfl⟩
  have hAtt : ∀ c, A.attention τ hs q k v c = A.store (A.round
      (A.sum (s.toList.map fun j => A.roundMul (w j * v j c)) / A.sum (s.toList.map w))) :=
    fun c => by subst ha hw; rfl
  simp only [hAtt]
  set σ : ν → ℝ := fun j => τ * ∑ i, q i * k j i with hσdef
  have hscore : ∀ j ∈ s, |a j - σ j| ≤ es ∧ |σ j| ≤ τ * (Q * K) := fun j hj => by
    rw [ha]; exact abs_score_sub_le A hS hu hround hmul hum hg hτ hq (hk j hj) hes
  obtain ⟨j0, hj0, hm⟩ := Finset.exists_mem_eq_sup' hs a
  set m := s.sup' hs a
  set E : ν → ℝ := fun j => Real.exp (σ j - m) with hEdef
  have hes0 : 0 ≤ es := (abs_nonneg _).trans (hscore j0 hj0).1
  have hδ0 : 0 ≤ δ := by nlinarith [abs_nonneg (σ j0), (hscore j0 hj0).2]
  have hθ0 : 0 ≤ θ := by
    have := Real.add_one_le_exp δ
    nlinarith
  have hwE : ∀ j ∈ s, |w j - E j| ≤ θ * E j + ηe := fun j hj => by
    rw [hw]
    exact abs_exp_round_sub_le hu hue hround hexp (hscore j hj).1 (hscore j hj).2
      (hm ▸ (hscore j0 hj0).1) (hscore j0 hj0).2 hδ hθ
  have hE0 : ∀ j ∈ s, 0 ≤ E j := fun j _ => (Real.exp_pos _).le
  have hL : 1 ≤ Real.exp es * ∑ j ∈ s, E j := by
    have h1 : E j0 ≤ ∑ j ∈ s, E j := Finset.single_le_sum hE0 hj0
    have h2 : -es ≤ σ j0 - m := by
      have := (abs_le.1 (hscore j0 hj0).1).2; rw [hm]; linarith
    have h3 : Real.exp (-es) ≤ E j0 := Real.exp_le_exp.2 h2
    have h4 : Real.exp es * Real.exp (-es) = 1 := by rw [← Real.exp_add]; simp
    nlinarith [Real.exp_pos es]
  have hV : 0 ≤ V := (Real.sqrt_nonneg _).trans (hv j0 hj0)
  have hLpos : 0 < ∑ j ∈ s, E j := Finset.sum_pos (fun j _ => Real.exp_pos _) hs
  have hden := abs_sum_sub_sum_le_mul hS hgs hgN hhN hcard hhs hηe hθ0 hE0 hwE hL heL
  have hnum := sqrt_sum_sq_sum_mul_sub_le hS hum hηm hmul hgs hgN hhN hcard hhs hηe hE0
    hwE hL hv hV heN
  have hmink : ∀ (r : ν → ℝ) (y : ν → κ → ℝ),
      √(∑ c, (∑ j ∈ s, r j * y j c) ^ 2) ≤ ∑ j ∈ s, |r j| * √(∑ c, y j c ^ 2) := fun r y => by
    have hsum : WithLp.toLp 2 (fun c => ∑ j ∈ s, r j * y j c) =
        ∑ j ∈ s, r j • WithLp.toLp 2 (y j) := by
      ext c; simp [WithLp.ofLp_sum, smul_eq_mul]
    have h := norm_sum_le s fun j => r j • WithLp.toLp 2 (y j)
    rw [← hsum] at h
    simp only [norm_smul, Real.norm_eq_abs] at h
    simpa [EuclideanSpace.norm_eq] using h
  have hN' : √(∑ c, (∑ j ∈ s, E j * v j c) ^ 2) ≤ V * ∑ j ∈ s, E j := by
    refine (hmink E v).trans ?_
    rw [Finset.mul_sum]
    exact Finset.sum_le_sum fun j hj => by
      rw [abs_of_nonneg (hE0 j hj), mul_comm V]
      exact mul_le_mul_of_nonneg_left (hv j hj) (hE0 j hj)
  have hratio := sqrt_sum_sq_div_sub_div_le hLpos hnum hden hN' heL1
  have hatt : ∀ c, TransformerBlock.attention τ s q k v c =
      (∑ j ∈ s, E j * v j c) / ∑ j ∈ s, E j := fun c => by
    rw [TransformerBlock.attention, Finset.sum_div]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [softmax_eq_exp_sub_div s σ m]
    ring
  simp only [hatt]
  have hy : √(∑ c, ((∑ j ∈ s, E j * v j c) / ∑ j ∈ s, E j) ^ 2) ≤ V := by
    simp only [div_pow]
    rw [← Finset.sum_div, Real.sqrt_div' _ (sq_nonneg _), Real.sqrt_sq hLpos.le,
      div_le_iff₀ hLpos]
    exact hN'
  have hfin := sqrt_sum_sq_store_round_sub_le hu hη hus hηs hround hstore
    (fun c => A.sum (s.toList.map fun j => A.roundMul (w j * v j c)) / A.sum (s.toList.map w))
    fun c => (∑ j ∈ s, E j * v j c) / ∑ j ∈ s, E j
  have e1 := mul_le_mul_of_nonneg_left hy (by positivity : 0 ≤ us + (1 + us) * u)
  have e2 := mul_le_mul_of_nonneg_left hratio (by positivity : 0 ≤ (1 + us) * (1 + u))
  linarith

/-! ### Stages of the block -/

/-- The error certificate of layer normalization over `d` entries at inputs with norm at most `M`
and `σlo ≤ √(var + ε) ≤ σhi`: errors of the mean, the centered vector, the variance sum, the
variance, `var + ε`, the denominator and the normalized vector, ending with the error `ep` of its
product with a weight bounded by `G`. -/
def LayerNormCertificate (g h : ℕ → ℝ) (d : ℕ) (u η G M σlo σhi ep : ℝ) : Prop :=
  ∃ eμ ec eV ev ew eσ ez : ℝ,
    u * M / √(d : ℝ) + (1 + u) * (g d * √(d : ℝ) * M + h d) / d + η ≤ eμ ∧
    u * M + √(d : ℝ) * ((1 + u) * eμ + η) ≤ ec ∧
    g d * ((1 + u) * (M + ec) ^ 2 + d * η) + h d + u * (M + ec) ^ 2 + d * η +
      ec * (2 * M + ec) ≤ eV ∧
    u * σhi ^ 2 + (1 + u) * eV / d + η ≤ ev ∧
    u * σhi ^ 2 + (1 + u) * ev + η ≤ ew ∧
    u * σhi + (1 + u) * ew / σlo + η ≤ eσ ∧ eσ < σlo ∧
    u * √(d : ℝ) + (1 + u) * (ec + √(d : ℝ) * eσ) / (σlo - eσ) + √(d : ℝ) * η ≤ ez ∧
    u * G * √(d : ℝ) + (1 + u) * G * ez + √(d : ℝ) * η ≤ ep

/-- A layer-normalization certificate for the standard-deviation range `[σlo', σhi']` is one for
every range `[σlo, σhi]` inside it with `0 ≤ σhi`. -/
theorem LayerNormCertificate.mono {g h : ℕ → ℝ} {d : ℕ} {u η G M σlo σhi σlo' σhi' ep : ℝ}
    (hc : LayerNormCertificate g h d u η G M σlo' σhi' ep) (hu : 0 ≤ u) (hη : 0 ≤ η)
    (hg : 0 ≤ g d) (hh : 0 ≤ h d) (hM : 0 ≤ M) (hσlo' : 0 < σlo') (hlo : σlo' ≤ σlo)
    (hσhi : 0 ≤ σhi) (hhi : σhi ≤ σhi') : LayerNormCertificate g h d u η G M σlo σhi ep := by
  obtain ⟨eμ, ec, eV, ev, ew, eσ, ez, h1, h2, h3, h4, h5, h6, h7, h8, h9⟩ := hc
  have hσhi' : 0 ≤ σhi' := hσhi.trans hhi
  have heμ : 0 ≤ eμ := le_trans (by positivity) h1
  have hec : 0 ≤ ec := le_trans (by positivity) h2
  have heV : 0 ≤ eV := le_trans (by positivity) h3
  have hev : 0 ≤ ev := le_trans (by positivity) h4
  have hew : 0 ≤ ew := le_trans (by positivity) h5
  have heσ : 0 ≤ eσ := le_trans (by positivity) h6
  refine ⟨eμ, ec, eV, ev, ew, eσ, ez, h1, h2, h3, le_trans ?_ h4, le_trans ?_ h5, le_trans ?_ h6,
    h7.trans_le hlo, le_trans ?_ h8, h9⟩
  · gcongr
  · gcongr
  · gcongr
  · have : 0 < σlo' - eσ := by linarith
    gcongr

/-- Layer normalization evaluated in `A` at an input within `e` of `x` is within the local error
at the computed input plus `2 * G * e / σlo` of exact layer normalization at `x`. -/
theorem Arithmetic.sqrt_sum_sq_layerNorm_sub_le' [Nonempty ι] (A : Arithmetic)
    {g h : ℕ → ℝ} {u η us ηs : ℝ}
    (hS : ∀ l : List ℝ, |A.sum l - l.sum| ≤ g l.length * (l.map abs).sum + h l.length)
    (hg : 0 ≤ g (Fintype.card ι)) (hu : 0 ≤ u) (hη : 0 ≤ η) (hus : 0 ≤ us) (hηs : 0 ≤ ηs)
    (hround : ∀ t, |A.round t - t| ≤ u * |t| + η)
    (hstore : ∀ t, |A.store t - t| ≤ us * |t| + ηs) {γ : ι → ℝ} {G : ℝ} (hγ : ∀ j, |γ j| ≤ G)
    (β : ι → ℝ) {ε : ℝ} (hε : 0 < ε) {x x' : ι → ℝ} {M σlo σhi Y e ep : ℝ}
    (hx : √(∑ j, x j ^ 2) ≤ M) (hlo : σlo ≤ √(var x + ε)) (hhi : √(var x + ε) ≤ σhi)
    (hY : √(∑ j, TransformerBlock.layerNorm γ β ε x j ^ 2) ≤ Y)
    (he : √(∑ j, (x' j - x j) ^ 2) ≤ e) (hσ0 : 0 < σlo - e / √(Fintype.card ι))
    (hcert : LayerNormCertificate g h (Fintype.card ι) u η G (M + e)
      (σlo - e / √(Fintype.card ι)) (σhi + e / √(Fintype.card ι)) ep) :
    √(∑ j, (A.layerNorm γ β ε x' j - TransformerBlock.layerNorm γ β ε x j) ^ 2) ≤
      (us + (1 + us) * u) * (Y + 2 * G * e / σlo) + (1 + us) * (1 + u) * ep +
        √(Fintype.card ι) * ((1 + us) * η + ηs) + 2 * G * e / σlo := by
  obtain ⟨eμ, ec, eV, ev, ew, eσ, ez, h1, h2, h3, h4, h5, h6, h7, h8, h9⟩ := hcert
  have hG : 0 ≤ G := (abs_nonneg _).trans (hγ (Classical.arbitrary ι))
  have he0 : 0 ≤ e := (Real.sqrt_nonneg _).trans he
  have hd : 0 < √(Fintype.card ι : ℝ) := Real.sqrt_pos.2 (by exact_mod_cast Fintype.card_pos)
  have hlo0 : 0 < σlo := by
    have : 0 ≤ e / √(Fintype.card ι : ℝ) := div_nonneg he0 hd.le
    linarith
  have hlip : √(∑ j, (TransformerBlock.layerNorm γ β ε x' j -
      TransformerBlock.layerNorm γ β ε x j) ^ 2) ≤ 2 * G * e / σlo := by
    refine (TransformerBlock.sqrt_sum_sq_layerNorm_sub_le hγ hε β x x').trans ?_
    calc 2 * G * √(∑ j, (x' j - x j) ^ 2) / √(var x + ε) ≤ 2 * G * e / √(var x + ε) := by
          gcongr
      _ ≤ 2 * G * e / σlo := div_le_div_of_nonneg_left (by positivity) hlo0 hlo
  have hx' : √(∑ j, x' j ^ 2) ≤ M + e := by
    have : √(∑ j, x' j ^ 2) ≤ √(∑ j, x j ^ 2) + √(∑ j, (x' j - x j) ^ 2) := by
      simpa [EuclideanSpace.norm_eq] using
        norm_le_norm_add_norm_sub' (WithLp.toLp 2 x') (WithLp.toLp 2 x)
    linarith
  have hσ := abs_sqrt_var_add_sub_le hε.le x x'
  have hσe : √(∑ i, (x' i - x i) ^ 2) / √(Fintype.card ι) ≤ e / √(Fintype.card ι) := by
    gcongr
  have hlo' : σlo - e / √(Fintype.card ι) ≤ √(var x' + ε) := by
    linarith [neg_abs_le (√(var x' + ε) - √(var x + ε))]
  have hhi' : √(var x' + ε) ≤ σhi + e / √(Fintype.card ι) := by
    linarith [le_abs_self (√(var x' + ε) - √(var x + ε))]
  have hY' : √(∑ j, TransformerBlock.layerNorm γ β ε x' j ^ 2) ≤ Y + 2 * G * e / σlo := by
    have : √(∑ j, TransformerBlock.layerNorm γ β ε x' j ^ 2) ≤
        √(∑ j, TransformerBlock.layerNorm γ β ε x j ^ 2) +
          √(∑ j, (TransformerBlock.layerNorm γ β ε x' j -
            TransformerBlock.layerNorm γ β ε x j) ^ 2) := by
      simpa [EuclideanSpace.norm_eq] using norm_le_norm_add_norm_sub'
        (WithLp.toLp 2 (TransformerBlock.layerNorm γ β ε x'))
        (WithLp.toLp 2 (TransformerBlock.layerNorm γ β ε x))
    linarith
  have hloc := A.sqrt_sum_sq_layerNorm_sub_le hS hg hu hη hus hηs hround hstore hγ β hε hx'
    hσ0 hlo' hhi' hY' h1 h2 h3 h4 h5 h6 h7 h8 h9
  have htri : √(∑ j, (A.layerNorm γ β ε x' j - TransformerBlock.layerNorm γ β ε x j) ^ 2) ≤
      √(∑ j, (A.layerNorm γ β ε x' j - TransformerBlock.layerNorm γ β ε x' j) ^ 2) +
        √(∑ j, (TransformerBlock.layerNorm γ β ε x' j -
          TransformerBlock.layerNorm γ β ε x j) ^ 2) := by
    simpa [EuclideanSpace.norm_eq] using norm_sub_le_norm_sub_add_norm_sub
      (WithLp.toLp 2 (A.layerNorm γ β ε x'))
      (WithLp.toLp 2 (TransformerBlock.layerNorm γ β ε x'))
      (WithLp.toLp 2 (TransformerBlock.layerNorm γ β ε x))
  linarith

/-- A vector over `ζ × ξ` whose slices `f (z, ·)` have norm at most `E` has norm at most
`√(card ζ) * E`. -/
theorem sqrt_sum_sq_le_sqrt_card_mul {ζ ξ : Type*} [Fintype ζ] [Fintype ξ] {f : ζ × ξ → ℝ}
    {E : ℝ} (h : ∀ z, √(∑ i, f (z, i) ^ 2) ≤ E) :
    √(∑ p, f p ^ 2) ≤ √(Fintype.card ζ) * E := by
  rcases isEmpty_or_nonempty ζ with hζ | ⟨⟨z0⟩⟩
  · simp
  have hE : 0 ≤ E := (Real.sqrt_nonneg _).trans (h z0)
  rw [show √(Fintype.card ζ : ℝ) * E = √(∑ _z : ζ, E ^ 2) by
    rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, Real.sqrt_mul (Nat.cast_nonneg _),
      Real.sqrt_sq hE]]
  apply Real.sqrt_le_sqrt
  rw [Fintype.sum_prod_type]
  refine Finset.sum_le_sum fun z _ => ?_
  have h0 : 0 ≤ ∑ i, f (z, i) ^ 2 := Finset.sum_nonneg fun i _ => sq_nonneg _
  calc ∑ i, f (z, i) ^ 2 = √(∑ i, f (z, i) ^ 2) ^ 2 := (Real.sq_sqrt h0).symm
    _ ≤ E ^ 2 := pow_le_pow_left₀ (Real.sqrt_nonneg _) (h z) 2

/-- Applying an activation computed with error `ua * |t| + ηa` to an input within `e` of `z` and
storing the result leaves the error `us * ‖act z‖ + (1 + us) * (ua * (Z + e) + La * e)` plus
`√card * ((1 + us) * ηa + ηs)` when `act` is `La`-Lipschitz and `‖z‖ ≤ Z`. -/
theorem sqrt_sum_sq_store_act_sub_le {fa fs act : ℝ → ℝ} {ua ηa us ηs La Z e : ℝ}
    (hua : 0 ≤ ua) (hηa : 0 ≤ ηa) (hus : 0 ≤ us) (hηs : 0 ≤ ηs) (hLa : 0 ≤ La)
    (hfa : ∀ t, |fa t - act t| ≤ ua * |t| + ηa) (hfs : ∀ t, |fs t - t| ≤ us * |t| + ηs)
    (hact : ∀ s t, |act s - act t| ≤ La * |s - t|) {z z' : ι → ℝ}
    (hz : √(∑ i, z i ^ 2) ≤ Z) (he : √(∑ i, (z' i - z i) ^ 2) ≤ e) :
    √(∑ i, (fs (fa (z' i)) - act (z i)) ^ 2) ≤ us * √(∑ i, act (z i) ^ 2) +
      (1 + us) * (ua * (Z + e) + La * e) + √(Fintype.card ι) * ((1 + us) * ηa + ηs) := by
  have hterm : ∀ i, |fs (fa (z' i)) - act (z i)| ≤ us * |act (z i)| + (1 + us) * ua * |z' i| +
      (1 + us) * La * |z' i - z i| + ((1 + us) * ηa + ηs) := fun i => by
    have h1 : |fa (z' i) - act (z i)| ≤ ua * |z' i| + ηa + La * |z' i - z i| := by
      calc |fa (z' i) - act (z i)| = |(fa (z' i) - act (z' i)) + (act (z' i) - act (z i))| := by
            ring_nf
        _ ≤ |fa (z' i) - act (z' i)| + |act (z' i) - act (z i)| := abs_add_le _ _
        _ ≤ ua * |z' i| + ηa + La * |z' i - z i| := add_le_add (hfa _) (hact _ _)
    have h2 := ((MixedError.mk hus hηs (hfs _)).trans_of_abs_sub_le h1).bound
    linarith
  have hsum := sqrt_sum_sq_le_add_of_abs_le hterm
  have hm : ∀ {a : ℝ}, 0 ≤ a → ∀ y : ι → ℝ, √(∑ i, (a * |y i|) ^ 2) = a * √(∑ i, y i ^ 2) :=
    fun {a} ha y => by
      simp_rw [mul_pow, sq_abs, ← Finset.mul_sum]
      rw [Real.sqrt_mul (sq_nonneg a), Real.sqrt_sq ha]
  rw [hm hus, hm (by positivity), hm (y := fun i => z' i - z i) (by positivity),
    Finset.sum_const, Finset.card_univ, nsmul_eq_mul, Real.sqrt_mul (Nat.cast_nonneg _),
    Real.sqrt_sq (by positivity)] at hsum
  have hz' : √(∑ i, z' i ^ 2) ≤ Z + e := by
    have : √(∑ i, z' i ^ 2) ≤ √(∑ i, z i ^ 2) + √(∑ i, (z' i - z i) ^ 2) := by
      simpa [EuclideanSpace.norm_eq] using
        norm_le_norm_add_norm_sub' (WithLp.toLp 2 z') (WithLp.toLp 2 z)
    linarith
  have h3 : (1 + us) * ua * √(∑ i, z' i ^ 2) ≤ (1 + us) * ua * (Z + e) := by gcongr
  have h4 : (1 + us) * La * √(∑ i, (z' i - z i) ^ 2) ≤ (1 + us) * La * e := by gcongr
  nlinarith

/-- Storing the rounded sum of approximations `x'` and `p'` of `x` and `p` with errors `ex` and
`ep` leaves the error `(us + (1 + us) * u) * ‖x + p‖ + (1 + us) * (1 + u) * (ex + ep)` plus
`√card * ((1 + us) * η + ηs)`. -/
theorem sqrt_sum_sq_store_add_sub_le {f fs : ℝ → ℝ} {u η us ηs ex ep : ℝ} (hu : 0 ≤ u)
    (hη : 0 ≤ η) (hus : 0 ≤ us) (hηs : 0 ≤ ηs) (hf : ∀ t, |f t - t| ≤ u * |t| + η)
    (hfs : ∀ t, |fs t - t| ≤ us * |t| + ηs) {x x' p p' : ι → ℝ}
    (hx : √(∑ i, (x' i - x i) ^ 2) ≤ ex) (hp : √(∑ i, (p' i - p i) ^ 2) ≤ ep) :
    √(∑ i, (fs (f (x' i + p' i)) - (x i + p i)) ^ 2) ≤
      (us + (1 + us) * u) * √(∑ i, (x i + p i) ^ 2) + (1 + us) * (1 + u) * (ex + ep) +
        √(Fintype.card ι) * ((1 + us) * η + ηs) := by
  have h1 := sqrt_sum_sq_store_round_sub_le hu hη hus hηs hf hfs (fun i => x' i + p' i)
    fun i => x i + p i
  have h2 : √(∑ i, ((x' i + p' i) - (x i + p i)) ^ 2) ≤ ex + ep := by
    have : √(∑ i, ((x' i + p' i) - (x i + p i)) ^ 2) ≤
        √(∑ i, (x' i - x i) ^ 2) + √(∑ i, (p' i - p i) ^ 2) := by
      simpa [EuclideanSpace.norm_eq, add_sub_add_comm] using
        norm_add_le (WithLp.toLp 2 fun i => x' i - x i) (WithLp.toLp 2 fun i => p' i - p i)
    linarith
  have := mul_le_mul_of_nonneg_left h2 (by positivity : (0 : ℝ) ≤ (1 + us) * (1 + u))
  linarith

/-! ### The block -/

variable {ζ ξ : Type*}

/-- Attention evaluated in `A` at queries, keys and values within `eq`, `ek` and `ev` of exact
ones is within the local error at the computed inputs plus `ev + 2 * tanh (ω / 2) * V` of exact
attention, where `ω` bounds the change of the scores. -/
theorem Arithmetic.sqrt_sum_sq_attention_sub_le' [Fintype κ] (A : Arithmetic)
    {g h : ℕ → ℝ} {u η um ηm us ηs ue ηe : ℝ}
    (hS : ∀ l : List ℝ, |A.sum l - l.sum| ≤ g l.length * (l.map abs).sum + h l.length)
    (hu : 0 ≤ u) (hη : 0 ≤ η) (hum : 0 ≤ um) (hηm : 0 ≤ ηm) (hus : 0 ≤ us) (hηs : 0 ≤ ηs)
    (hue : 0 ≤ ue) (hηe : 0 ≤ ηe) (hround : ∀ t, |A.round t - t| ≤ u * |t| + η)
    (hmul : ∀ t, |A.roundMul t - t| ≤ um * |t| + ηm)
    (hstore : ∀ t, |A.store t - t| ≤ us * |t| + ηs)
    (hexp : ∀ t, |A.exp t - Real.exp t| ≤ ue * Real.exp t + ηe)
    {τ : ℝ} (hτ : 0 ≤ τ) {s : Finset ν} (hs : s.Nonempty) {gN hN N : ℝ}
    (hg : 0 ≤ g (Fintype.card ι)) (hgs : 0 ≤ g s.card) (hhs : 0 ≤ h s.card)
    (hgN : g s.card ≤ gN) (hhN : h s.card ≤ hN) (hcard : (s.card : ℝ) ≤ N)
    {q q' : ι → ℝ} {k k' : ν → ι → ℝ} {v v' : ν → κ → ℝ} {Q K V eq ek ev ω es δ θ eL eN : ℝ}
    (hq : √(∑ i, q i ^ 2) ≤ Q) (hk : ∀ j ∈ s, √(∑ i, k j i ^ 2) ≤ K)
    (hv : ∀ j ∈ s, √(∑ c, v j c ^ 2) ≤ V) (heq : √(∑ i, (q' i - q i) ^ 2) ≤ eq)
    (hek : ∀ j ∈ s, √(∑ i, (k' j i - k j i) ^ 2) ≤ ek)
    (hev : ∀ j ∈ s, √(∑ c, (v' j c - v j c) ^ 2) ≤ ev)
    (hω : τ * (eq * (K + ek) + Q * ek) ≤ ω)
    (hes : u * τ * ((Q + eq) * (K + ek)) + (1 + u) * τ * ((g (Fintype.card ι) * (1 + um) + um) *
      ((Q + eq) * (K + ek)) + (1 + g (Fintype.card ι)) * Fintype.card ι * ηm +
        h (Fintype.card ι)) + η ≤ es)
    (hδ : u * (2 * (τ * ((Q + eq) * (K + ek))) + 2 * es) + es + η ≤ δ)
    (hθ : (1 + ue) * Real.exp δ - 1 ≤ θ)
    (heL : gN * (1 + θ) + θ + ((1 + gN) * N * ηe + hN) * Real.exp es ≤ eL) (heL1 : eL < 1)
    (heN : ((gN * (1 + um) + um) * (1 + θ) + θ) * (V + ev) + Real.exp es *
      ((gN * (1 + um) + um + 1) * (V + ev) * N * ηe +
        √(Fintype.card κ) * ((1 + gN) * N * ηm + hN)) ≤ eN) :
    √(∑ c, (A.attention τ hs q' k' v' c - TransformerBlock.attention τ s q k v c) ^ 2) ≤
      (us + (1 + us) * u) * (V + ev) + (1 + us) * (1 + u) * ((eN + eL * (V + ev)) / (1 - eL)) +
        √(Fintype.card κ) * ((1 + us) * η + ηs) +
          (ev + 2 * (Real.exp ω - 1) / (Real.exp ω + 1) * V) := by
  have hq' : √(∑ i, q' i ^ 2) ≤ Q + eq := by
    have : √(∑ i, q' i ^ 2) ≤ √(∑ i, q i ^ 2) + √(∑ i, (q' i - q i) ^ 2) := by
      simpa [EuclideanSpace.norm_eq] using
        norm_le_norm_add_norm_sub' (WithLp.toLp 2 q') (WithLp.toLp 2 q)
    linarith
  have hk' : ∀ j ∈ s, √(∑ i, k' j i ^ 2) ≤ K + ek := fun j hj => by
    have : √(∑ i, k' j i ^ 2) ≤ √(∑ i, k j i ^ 2) + √(∑ i, (k' j i - k j i) ^ 2) := by
      simpa [EuclideanSpace.norm_eq] using
        norm_le_norm_add_norm_sub' (WithLp.toLp 2 (k' j)) (WithLp.toLp 2 (k j))
    linarith [hk j hj, hek j hj]
  have hv' : ∀ j ∈ s, √(∑ c, v' j c ^ 2) ≤ V + ev := fun j hj => by
    have : √(∑ c, v' j c ^ 2) ≤ √(∑ c, v j c ^ 2) + √(∑ c, (v' j c - v j c) ^ 2) := by
      simpa [EuclideanSpace.norm_eq] using
        norm_le_norm_add_norm_sub' (WithLp.toLp 2 (v' j)) (WithLp.toLp 2 (v j))
    linarith [hv j hj, hev j hj]
  have hloc := A.sqrt_sum_sq_attention_sub_le hS hu hη hum hηm hus hηs hue hηe hround hmul
    hstore hexp hτ hs hg hgs hhs hgN hhN hcard hq' hk' hv' hes hδ hθ heL heL1 heN
  have hpert := TransformerBlock.sqrt_sum_sq_attention_sub_le hτ hs hq hk hv heq hek hev hω
  have htri : √(∑ c, (A.attention τ hs q' k' v' c - TransformerBlock.attention τ s q k v c) ^ 2) ≤
      √(∑ c, (A.attention τ hs q' k' v' c - TransformerBlock.attention τ s q' k' v' c) ^ 2) +
        √(∑ c, (TransformerBlock.attention τ s q' k' v' c -
          TransformerBlock.attention τ s q k v c) ^ 2) := by
    simpa [EuclideanSpace.norm_eq] using norm_sub_le_norm_sub_add_norm_sub
      (WithLp.toLp 2 (A.attention τ hs q' k' v'))
      (WithLp.toLp 2 (TransformerBlock.attention τ s q' k' v'))
      (WithLp.toLp 2 (TransformerBlock.attention τ s q k v))
  linarith

/-- Error bounds of the operations of `A` against exact arithmetic with activation `act`: mixed
relative and absolute errors of the roundings, the exponential and the activation, a summation
error `g n * ∑ |l i| + h n` for lists of length `n`, and a Lipschitz constant `la` of `act`. -/
structure Arithmetic.Contract (A : Arithmetic) (act : ℝ → ℝ) where
  /-- Relative error of `round`. -/
  u : ℝ
  /-- Absolute error of `round`. -/
  η : ℝ
  /-- Relative error of `roundMul`. -/
  um : ℝ
  /-- Absolute error of `roundMul`. -/
  ηm : ℝ
  /-- Relative error of `store`. -/
  us : ℝ
  /-- Absolute error of `store`. -/
  ηs : ℝ
  /-- Relative error of `exp`. -/
  ue : ℝ
  /-- Absolute error of `exp`. -/
  ηe : ℝ
  /-- Relative error of `act`, measured against the input. -/
  ua : ℝ
  /-- Absolute error of `act`. -/
  ηa : ℝ
  /-- Lipschitz constant of the exact activation. -/
  la : ℝ
  /-- Relative summation error for lists of each length. -/
  g : ℕ → ℝ
  /-- Absolute summation error for lists of each length. -/
  h : ℕ → ℝ
  /-- `u` is nonnegative. -/
  u_nonneg : 0 ≤ u
  /-- `η` is nonnegative. -/
  η_nonneg : 0 ≤ η
  /-- `um` is nonnegative. -/
  um_nonneg : 0 ≤ um
  /-- `ηm` is nonnegative. -/
  ηm_nonneg : 0 ≤ ηm
  /-- `us` is nonnegative. -/
  us_nonneg : 0 ≤ us
  /-- `ηs` is nonnegative. -/
  ηs_nonneg : 0 ≤ ηs
  /-- `ue` is nonnegative. -/
  ue_nonneg : 0 ≤ ue
  /-- `ηe` is nonnegative. -/
  ηe_nonneg : 0 ≤ ηe
  /-- `ua` is nonnegative. -/
  ua_nonneg : 0 ≤ ua
  /-- `ηa` is nonnegative. -/
  ηa_nonneg : 0 ≤ ηa
  /-- `la` is nonnegative. -/
  la_nonneg : 0 ≤ la
  /-- Every `g n` is nonnegative. -/
  g_nonneg : ∀ n, 0 ≤ g n
  /-- Every `h n` is nonnegative. -/
  h_nonneg : ∀ n, 0 ≤ h n
  /-- `round t` is within `u * |t| + η` of `t`. -/
  round_le : ∀ t, |A.round t - t| ≤ u * |t| + η
  /-- `roundMul t` is within `um * |t| + ηm` of `t`. -/
  roundMul_le : ∀ t, |A.roundMul t - t| ≤ um * |t| + ηm
  /-- `store t` is within `us * |t| + ηs` of `t`. -/
  store_le : ∀ t, |A.store t - t| ≤ us * |t| + ηs
  /-- `A.exp t` is within `ue * Real.exp t + ηe` of `Real.exp t`. -/
  exp_le : ∀ t, |A.exp t - Real.exp t| ≤ ue * Real.exp t + ηe
  /-- `A.sum l` is within `g n * ∑ |l i| + h n` of the sum of a list `l` of length `n`. -/
  sum_le : ∀ l : List ℝ, |A.sum l - l.sum| ≤ g l.length * (l.map abs).sum + h l.length
  /-- `A.act t` is within `ua * |t| + ηa` of `act t`. -/
  act_le : ∀ t, |A.act t - act t| ≤ ua * |t| + ηa
  /-- `act` is `la`-Lipschitz. -/
  act_lipschitz : ∀ s t, |act s - act t| ≤ la * |s - t|

/-- The parameters of a pre-normalization transformer block with model width `ι`, attention
columns `ζ × ξ` for heads `ζ` of width `ξ`, and hidden width `κ`. -/
structure Parameters (ι ζ ξ κ : Type*) where
  /-- Weight of the first layer normalization (`ln_1`). -/
  γ1 : ι → ℝ
  /-- Bias of the first layer normalization. -/
  β1 : ι → ℝ
  /-- Query projection. -/
  wq : ι → ζ × ξ → ℝ
  /-- Query bias. -/
  bq : ζ × ξ → ℝ
  /-- Key projection. -/
  wk : ι → ζ × ξ → ℝ
  /-- Key bias. -/
  bk : ζ × ξ → ℝ
  /-- Value projection. -/
  wv : ι → ζ × ξ → ℝ
  /-- Value bias. -/
  bv : ζ × ξ → ℝ
  /-- Output projection (`attn.c_proj`). -/
  wo : ζ × ξ → ι → ℝ
  /-- Output bias. -/
  bo : ι → ℝ
  /-- Weight of the second layer normalization (`ln_2`). -/
  γ2 : ι → ℝ
  /-- Bias of the second layer normalization. -/
  β2 : ι → ℝ
  /-- First MLP weight (`mlp.c_fc`). -/
  w1 : ι → κ → ℝ
  /-- First MLP bias. -/
  b1 : κ → ℝ
  /-- Second MLP weight (`mlp.c_proj`). -/
  w2 : κ → ι → ℝ
  /-- Second MLP bias. -/
  b2 : ι → ℝ

/-- Multi-head attention: head `z` of token `t` attends over `keys t` with the slices `(z, ·)` of
the queries, keys and values. -/
noncomputable def heads [Fintype ξ] (τ : ℝ) (keys : ν → Finset ν) (q k v : ν → ζ × ξ → ℝ) :
    ν → ζ × ξ → ℝ :=
  fun t p => attention τ (keys t) (fun i => q t (p.1, i)) (fun j i => k j (p.1, i))
    (fun j c => v j (p.1, c)) p.2

/-- Multi-head attention evaluated in `A`, head by head. -/
noncomputable def Arithmetic.heads [Fintype ξ] (A : Arithmetic) (τ : ℝ) {keys : ν → Finset ν}
    (hkeys : ∀ t, (keys t).Nonempty) (q k v : ν → ζ × ξ → ℝ) : ν → ζ × ξ → ℝ :=
  fun t p => A.attention τ (hkeys t) (fun i => q t (p.1, i)) (fun j i => k j (p.1, i))
    (fun j c => v j (p.1, c)) p.2

/-- The pre-normalization transformer block of GPT-2, `s = x + attn (ln_1 x)` and
`y = s + mlp (ln_2 s)`, with attention scale `τ` and token `t` attending over `keys t`. -/
noncomputable def block [Fintype ζ] [Fintype ξ] [Fintype κ] (act : ℝ → ℝ)
    (P : Parameters ι ζ ξ κ) (ε τ : ℝ) (keys : ν → Finset ν) (x : ν → ι → ℝ) : ν → ι → ℝ :=
  let a := fun t => layerNorm P.γ1 P.β1 ε (x t)
  let q := fun t c => P.bq c + ∑ j, a t j * P.wq j c
  let k := fun t c => P.bk c + ∑ j, a t j * P.wk j c
  let v := fun t c => P.bv c + ∑ j, a t j * P.wv j c
  let o := heads τ keys q k v
  let s := fun t i => x t i + (P.bo i + ∑ c, o t c * P.wo c i)
  let b := fun t => layerNorm P.γ2 P.β2 ε (s t)
  fun t i => s t i + (P.b2 i + ∑ c, act (P.b1 c + ∑ j, b t j * P.w1 j c) * P.w2 c i)

/-- The block evaluated in `A`: every layer normalization, linear map, attention head,
activation and residual sum is evaluated in `A` and stored. -/
noncomputable def Arithmetic.block [Fintype ζ] [Fintype ξ] [Fintype κ] (A : Arithmetic)
    (P : Parameters ι ζ ξ κ) (ε τ : ℝ) {keys : ν → Finset ν} (hkeys : ∀ t, (keys t).Nonempty)
    (x : ν → ι → ℝ) : ν → ι → ℝ :=
  let a := fun t => A.layerNorm P.γ1 P.β1 ε (x t)
  let q := fun t => A.linear P.wq P.bq (a t)
  let k := fun t => A.linear P.wk P.bk (a t)
  let v := fun t => A.linear P.wv P.bv (a t)
  let o := A.heads τ hkeys q k v
  let s := fun t i => A.store (A.round (x t i + A.linear P.wo P.bo (o t) i))
  let b := fun t => A.layerNorm P.γ2 P.β2 ε (s t)
  let m := fun t c => A.store (A.act (A.linear P.w1 P.b1 (b t) c))
  fun t i => A.store (A.round (s t i + A.linear P.w2 P.b2 (m t) i))

/-- The block evaluated in `Arithmetic.exact act` is the real block `block act`. -/
theorem Arithmetic.exact_block [Fintype ζ] [Fintype ξ] [Fintype κ] (act : ℝ → ℝ)
    (P : Parameters ι ζ ξ κ) (ε τ : ℝ) {keys : ν → Finset ν} (hkeys : ∀ t, (keys t).Nonempty)
    (x : ν → ι → ℝ) :
    (Arithmetic.exact act).block P ε τ hkeys x = TransformerBlock.block act P ε τ keys x := by
  funext t i
  simp only [Arithmetic.block, TransformerBlock.block, Arithmetic.heads, TransformerBlock.heads,
    Arithmetic.exact_layerNorm, Arithmetic.exact_linear, Arithmetic.exact_attention]
  rfl

/-- Every token of the block evaluated in `A` is within Euclidean distance `ey` of the exact
block. The exact stages `a, q, k, v, o, s, b, r` have the stated magnitudes, the weights satisfy
their `LinearBound`s, and each stage error is at least the bound of its stage lemma at the errors
of the stages before it. -/
theorem Arithmetic.sqrt_sum_sq_block_sub_le [Nonempty ι] [Fintype ζ] [Fintype ξ] [Fintype κ]
    {A : Arithmetic} {act : ℝ → ℝ} (C : A.Contract act) (P : Parameters ι ζ ξ κ) {ε τ : ℝ}
    (hε : 0 < ε) (hτ : 0 ≤ τ) {keys : ν → Finset ν} (hkeys : ∀ t, (keys t).Nonempty)
    (x : ν → ι → ℝ) {a : ν → ι → ℝ} (ha : a = fun t => TransformerBlock.layerNorm P.γ1 P.β1 ε (x t))
    {q k v : ν → ζ × ξ → ℝ} (hq : q = fun t c => P.bq c + ∑ j, a t j * P.wq j c)
    (hk : k = fun t c => P.bk c + ∑ j, a t j * P.wk j c)
    (hv : v = fun t c => P.bv c + ∑ j, a t j * P.wv j c)
    {o : ν → ζ × ξ → ℝ} (ho : o = TransformerBlock.heads τ keys q k v)
    {s : ν → ι → ℝ} (hs : s = fun t i => x t i + (P.bo i + ∑ c, o t c * P.wo c i))
    {b : ν → ι → ℝ} (hb : b = fun t => TransformerBlock.layerNorm P.γ2 P.β2 ε (s t))
    {r : ν → κ → ℝ} (hr : r = fun t c => P.b1 c + ∑ j, b t j * P.w1 j c)
    {X σlo1 σhi1 Ya Q K V O Z σlo2 σhi2 Yb R Ym Y G1 G2 : ℝ}
    (hX : ∀ t, √(∑ i, x t i ^ 2) ≤ X) (hσlo1 : ∀ t, σlo1 ≤ √(var (x t) + ε))
    (hσhi1 : ∀ t, √(var (x t) + ε) ≤ σhi1) (hYa : ∀ t, √(∑ i, a t i ^ 2) ≤ Ya)
    (hQ : ∀ t z, √(∑ i, q t (z, i) ^ 2) ≤ Q) (hK : ∀ t z, √(∑ i, k t (z, i) ^ 2) ≤ K)
    (hV : ∀ t z, √(∑ i, v t (z, i) ^ 2) ≤ V) (hO : ∀ t, √(∑ c, o t c ^ 2) ≤ O)
    (hZ : ∀ t, √(∑ i, s t i ^ 2) ≤ Z) (hσlo2 : ∀ t, σlo2 ≤ √(var (s t) + ε))
    (hσhi2 : ∀ t, √(var (s t) + ε) ≤ σhi2) (hYb : ∀ t, √(∑ i, b t i ^ 2) ≤ Yb)
    (hR : ∀ t, √(∑ c, r t c ^ 2) ≤ R) (hYm : ∀ t, √(∑ c, act (r t c) ^ 2) ≤ Ym)
    (hY : ∀ t, √(∑ i, TransformerBlock.block act P ε τ keys x t i ^ 2) ≤ Y)
    (hγ1 : ∀ j, |P.γ1 j| ≤ G1) (hγ2 : ∀ j, |P.γ2 j| ≤ G2)
    {Sq Fq Bq Sk Fk Bk Sv Fv Bv So Fo Bo S1 F1 B1 S2 F2 B2 : ℝ}
    (hWq : ∀ z, LinearBound (fun j i => P.wq j (z, i)) (fun i => P.bq (z, i)) Sq Fq Bq)
    (hWk : ∀ z, LinearBound (fun j i => P.wk j (z, i)) (fun i => P.bk (z, i)) Sk Fk Bk)
    (hWv : ∀ z, LinearBound (fun j i => P.wv j (z, i)) (fun i => P.bv (z, i)) Sv Fv Bv)
    (hWo : LinearBound P.wo P.bo So Fo Bo) (hW1 : LinearBound P.w1 P.b1 S1 F1 B1)
    (hW2 : LinearBound P.w2 P.b2 S2 F2 B2) {gN hN N : ℝ}
    (hgN : ∀ t, C.g (keys t).card ≤ gN) (hhN : ∀ t, C.h (keys t).card ≤ hN)
    (hcard : ∀ t, ((keys t).card : ℝ) ≤ N)
    {ep1 ea eq ek ev ω es δ θ eL eN eo ep ez ep2 eb er em ef ey : ℝ}
    (hσlo1pos : 0 < σlo1)
    (hLN1 : LayerNormCertificate C.g C.h (Fintype.card ι) C.u C.η G1 X σlo1 σhi1 ep1)
    (hea : (C.us + (1 + C.us) * C.u) * Ya + (1 + C.us) * (1 + C.u) * ep1 +
      √(Fintype.card ι) * ((1 + C.us) * C.η + C.ηs) ≤ ea)
    (heq : Sq * ea + C.us * (Bq + Sq * (Ya + ea)) + (1 + C.us) * (C.g (Fintype.card ι + 1) * Bq +
      (C.g (Fintype.card ι + 1) * (1 + C.um) + C.um) * Fq * (Ya + ea) +
      √(Fintype.card ξ) * ((1 + C.g (Fintype.card ι + 1)) * Fintype.card ι * C.ηm +
        C.h (Fintype.card ι + 1))) + √(Fintype.card ξ) * C.ηs ≤ eq)
    (hek : Sk * ea + C.us * (Bk + Sk * (Ya + ea)) + (1 + C.us) * (C.g (Fintype.card ι + 1) * Bk +
      (C.g (Fintype.card ι + 1) * (1 + C.um) + C.um) * Fk * (Ya + ea) +
      √(Fintype.card ξ) * ((1 + C.g (Fintype.card ι + 1)) * Fintype.card ι * C.ηm +
        C.h (Fintype.card ι + 1))) + √(Fintype.card ξ) * C.ηs ≤ ek)
    (hev : Sv * ea + C.us * (Bv + Sv * (Ya + ea)) + (1 + C.us) * (C.g (Fintype.card ι + 1) * Bv +
      (C.g (Fintype.card ι + 1) * (1 + C.um) + C.um) * Fv * (Ya + ea) +
      √(Fintype.card ξ) * ((1 + C.g (Fintype.card ι + 1)) * Fintype.card ι * C.ηm +
        C.h (Fintype.card ι + 1))) + √(Fintype.card ξ) * C.ηs ≤ ev)
    (hω : τ * (eq * (K + ek) + Q * ek) ≤ ω)
    (hes : C.u * τ * ((Q + eq) * (K + ek)) + (1 + C.u) * τ * ((C.g (Fintype.card ξ) *
      (1 + C.um) + C.um) * ((Q + eq) * (K + ek)) + (1 + C.g (Fintype.card ξ)) *
        Fintype.card ξ * C.ηm + C.h (Fintype.card ξ)) + C.η ≤ es)
    (hδ : C.u * (2 * (τ * ((Q + eq) * (K + ek))) + 2 * es) + es + C.η ≤ δ)
    (hθ : (1 + C.ue) * Real.exp δ - 1 ≤ θ)
    (heL : gN * (1 + θ) + θ + ((1 + gN) * N * C.ηe + hN) * Real.exp es ≤ eL) (heL1 : eL < 1)
    (heN : ((gN * (1 + C.um) + C.um) * (1 + θ) + θ) * (V + ev) + Real.exp es *
      ((gN * (1 + C.um) + C.um + 1) * (V + ev) * N * C.ηe +
        √(Fintype.card ξ) * ((1 + gN) * N * C.ηm + hN)) ≤ eN)
    (heo : (C.us + (1 + C.us) * C.u) * (V + ev) + (1 + C.us) * (1 + C.u) *
      ((eN + eL * (V + ev)) / (1 - eL)) + √(Fintype.card ξ) * ((1 + C.us) * C.η + C.ηs) +
        (ev + 2 * (Real.exp ω - 1) / (Real.exp ω + 1) * V) ≤ eo)
    (hep : So * (√(Fintype.card ζ) * eo) + C.us * (Bo + So * (O + √(Fintype.card ζ) * eo)) +
      (1 + C.us) * (C.g (Fintype.card (ζ × ξ) + 1) * Bo +
        (C.g (Fintype.card (ζ × ξ) + 1) * (1 + C.um) + C.um) * Fo *
          (O + √(Fintype.card ζ) * eo) + √(Fintype.card ι) *
            ((1 + C.g (Fintype.card (ζ × ξ) + 1)) * Fintype.card (ζ × ξ) * C.ηm +
              C.h (Fintype.card (ζ × ξ) + 1))) + √(Fintype.card ι) * C.ηs ≤ ep)
    (hez : (C.us + (1 + C.us) * C.u) * Z + (1 + C.us) * (1 + C.u) * ep +
      √(Fintype.card ι) * ((1 + C.us) * C.η + C.ηs) ≤ ez)
    (hσlo2pos : 0 < σlo2 - ez / √(Fintype.card ι))
    (hLN2 : LayerNormCertificate C.g C.h (Fintype.card ι) C.u C.η G2 (Z + ez)
      (σlo2 - ez / √(Fintype.card ι)) (σhi2 + ez / √(Fintype.card ι)) ep2)
    (heb : (C.us + (1 + C.us) * C.u) * (Yb + 2 * G2 * ez / σlo2) + (1 + C.us) * (1 + C.u) * ep2 +
      √(Fintype.card ι) * ((1 + C.us) * C.η + C.ηs) + 2 * G2 * ez / σlo2 ≤ eb)
    (her : S1 * eb + C.us * (B1 + S1 * (Yb + eb)) + (1 + C.us) * (C.g (Fintype.card ι + 1) * B1 +
      (C.g (Fintype.card ι + 1) * (1 + C.um) + C.um) * F1 * (Yb + eb) +
      √(Fintype.card κ) * ((1 + C.g (Fintype.card ι + 1)) * Fintype.card ι * C.ηm +
        C.h (Fintype.card ι + 1))) + √(Fintype.card κ) * C.ηs ≤ er)
    (hem : C.us * Ym + (1 + C.us) * (C.ua * (R + er) + C.la * er) +
      √(Fintype.card κ) * ((1 + C.us) * C.ηa + C.ηs) ≤ em)
    (hef : S2 * em + C.us * (B2 + S2 * (Ym + em)) + (1 + C.us) * (C.g (Fintype.card κ + 1) * B2 +
      (C.g (Fintype.card κ + 1) * (1 + C.um) + C.um) * F2 * (Ym + em) +
      √(Fintype.card ι) * ((1 + C.g (Fintype.card κ + 1)) * Fintype.card κ * C.ηm +
        C.h (Fintype.card κ + 1))) + √(Fintype.card ι) * C.ηs ≤ ef)
    (hey : (C.us + (1 + C.us) * C.u) * Y + (1 + C.us) * (1 + C.u) * (ez + ef) +
      √(Fintype.card ι) * ((1 + C.us) * C.η + C.ηs) ≤ ey) (t : ν) :
    √(∑ i, (A.block P ε τ hkeys x t i - TransformerBlock.block act P ε τ keys x t i) ^ 2) ≤
      ey := by
  have hu := C.u_nonneg
  have hη := C.η_nonneg
  have hus := C.us_nonneg
  have hηs := C.ηs_nonneg
  have hc : 0 ≤ C.us + (1 + C.us) * C.u := by positivity
  obtain ⟨a', ha'⟩ : ∃ a' : ν → ι → ℝ, a' = fun t => A.layerNorm P.γ1 P.β1 ε (x t) := ⟨_, rfl⟩
  obtain ⟨q', hq'⟩ : ∃ q' : ν → ζ × ξ → ℝ, q' = fun t => A.linear P.wq P.bq (a' t) := ⟨_, rfl⟩
  obtain ⟨k', hk'⟩ : ∃ k' : ν → ζ × ξ → ℝ, k' = fun t => A.linear P.wk P.bk (a' t) := ⟨_, rfl⟩
  obtain ⟨v', hv'⟩ : ∃ v' : ν → ζ × ξ → ℝ, v' = fun t => A.linear P.wv P.bv (a' t) := ⟨_, rfl⟩
  obtain ⟨o', ho'⟩ : ∃ o' : ν → ζ × ξ → ℝ, o' = A.heads τ hkeys q' k' v' := ⟨_, rfl⟩
  obtain ⟨s', hs'⟩ : ∃ s' : ν → ι → ℝ,
      s' = fun t i => A.store (A.round (x t i + A.linear P.wo P.bo (o' t) i)) := ⟨_, rfl⟩
  obtain ⟨b', hb'⟩ : ∃ b' : ν → ι → ℝ, b' = fun t => A.layerNorm P.γ2 P.β2 ε (s' t) :=
    ⟨_, rfl⟩
  obtain ⟨m', hm'⟩ : ∃ m' : ν → κ → ℝ,
      m' = fun t c => A.store (A.act (A.linear P.w1 P.b1 (b' t) c)) := ⟨_, rfl⟩
  have hA : ∀ i, A.block P ε τ hkeys x t i =
      A.store (A.round (s' t i + A.linear P.w2 P.b2 (m' t) i)) := by
    subst ha' hq' hk' hv' ho' hs' hb' hm'; intro i; rfl
  have hE : ∀ t i, TransformerBlock.block act P ε τ keys x t i =
      s t i + (P.b2 i + ∑ c, act (r t c) * P.w2 c i) := by
    subst ha hq hk hv ho hs hb hr; intro t i; rfl
  have e1 : ∀ t, √(∑ i, (a' t i - a t i) ^ 2) ≤ ea := fun t => by
    obtain ⟨eμ, ec, eV, ev1, ew, eσ, ez1, h1, h2, h3, h4, h5, h6, h7, h8, h9⟩ := hLN1
    have hYat := hYa t
    subst ha ha'
    exact (A.sqrt_sum_sq_layerNorm_sub_le C.sum_le (C.g_nonneg _) hu hη hus hηs C.round_le
      C.store_le hγ1 P.β1 hε (hX t) hσlo1pos (hσlo1 t) (hσhi1 t) hYat h1 h2 h3 h4 h5 h6 h7 h8
      h9).trans hea
  have e2q : ∀ t z, √(∑ i, (q' t (z, i) - q t (z, i)) ^ 2) ≤ eq := fun t z => by
    subst hq hq'
    exact (A.sqrt_sum_sq_linear_sub_le' C.sum_le (C.g_nonneg _) (C.h_nonneg _)
      C.um_nonneg C.ηm_nonneg hus hηs C.roundMul_le C.store_le (hWq z) (hYa t) (e1 t)).trans heq
  have e2k : ∀ t z, √(∑ i, (k' t (z, i) - k t (z, i)) ^ 2) ≤ ek := fun t z => by
    subst hk hk'
    exact (A.sqrt_sum_sq_linear_sub_le' C.sum_le (C.g_nonneg _) (C.h_nonneg _)
      C.um_nonneg C.ηm_nonneg hus hηs C.roundMul_le C.store_le (hWk z) (hYa t) (e1 t)).trans hek
  have e2v : ∀ t z, √(∑ i, (v' t (z, i) - v t (z, i)) ^ 2) ≤ ev := fun t z => by
    subst hv hv'
    exact (A.sqrt_sum_sq_linear_sub_le' C.sum_le (C.g_nonneg _) (C.h_nonneg _)
      C.um_nonneg C.ηm_nonneg hus hηs C.roundMul_le C.store_le (hWv z) (hYa t) (e1 t)).trans hev
  have e3 : ∀ t z, √(∑ c, (o' t (z, c) - o t (z, c)) ^ 2) ≤ eo := fun t z => by
    subst ho ho'
    exact (A.sqrt_sum_sq_attention_sub_le' C.sum_le hu hη C.um_nonneg C.ηm_nonneg hus hηs
      C.ue_nonneg C.ηe_nonneg C.round_le C.roundMul_le C.store_le C.exp_le hτ (hkeys t)
      (C.g_nonneg _) (C.g_nonneg _) (C.h_nonneg _) (hgN t) (hhN t) (hcard t) (hQ t z)
      (fun j _ => hK j z) (fun j _ => hV j z) (e2q t z) (fun j _ => e2k j z)
      (fun j _ => e2v j z) hω hes hδ hθ heL heL1 heN).trans heo
  have e4 : ∀ t, √(∑ c, (o' t c - o t c) ^ 2) ≤ √(Fintype.card ζ) * eo := fun t =>
    sqrt_sum_sq_le_sqrt_card_mul (f := fun c => o' t c - o t c) (e3 t)
  have e5 : ∀ t, √(∑ i, (A.linear P.wo P.bo (o' t) i - (P.bo i + ∑ c, o t c * P.wo c i)) ^ 2) ≤
      ep := fun t =>
    (A.sqrt_sum_sq_linear_sub_le' C.sum_le (C.g_nonneg _) (C.h_nonneg _) C.um_nonneg
      C.ηm_nonneg hus hηs C.roundMul_le C.store_le hWo (hO t) (e4 t)).trans hep
  have e6 : ∀ t, √(∑ i, (s' t i - s t i) ^ 2) ≤ ez := fun t => by
    have h0 : √(∑ i, (x t i - x t i) ^ 2) ≤ 0 := by simp
    have h1 := sqrt_sum_sq_store_add_sub_le hu hη hus hηs C.round_le C.store_le h0 (e5 t)
    have hZt : √(∑ i, (x t i + (P.bo i + ∑ c, o t c * P.wo c i)) ^ 2) ≤ Z := by
      have := hZ t; rw [hs] at this; exact this
    subst hs hs'
    refine h1.trans (le_trans ?_ hez)
    have := mul_le_mul_of_nonneg_left hZt hc
    linarith
  have e7 : ∀ t, √(∑ i, (b' t i - b t i) ^ 2) ≤ eb := fun t => by
    have hYbt := hYb t
    subst hb hb'
    exact (A.sqrt_sum_sq_layerNorm_sub_le' C.sum_le (C.g_nonneg _) hu hη hus hηs
      C.round_le C.store_le hγ2 P.β2 hε (hZ t) (hσlo2 t) (hσhi2 t) hYbt (e6 t) hσlo2pos
      hLN2).trans heb
  have e8 : ∀ t, √(∑ c, (A.linear P.w1 P.b1 (b' t) c - r t c) ^ 2) ≤ er := fun t => by
    subst hr
    exact (A.sqrt_sum_sq_linear_sub_le' C.sum_le (C.g_nonneg _) (C.h_nonneg _)
      C.um_nonneg C.ηm_nonneg hus hηs C.roundMul_le C.store_le hW1 (hYb t) (e7 t)).trans her
  have e9 : ∀ t, √(∑ c, (m' t c - act (r t c)) ^ 2) ≤ em := fun t => by
    subst hm'
    refine (sqrt_sum_sq_store_act_sub_le C.ua_nonneg C.ηa_nonneg hus hηs C.la_nonneg C.act_le
      C.store_le C.act_lipschitz (hR t) (e8 t)).trans (le_trans ?_ hem)
    have := mul_le_mul_of_nonneg_left (hYm t) hus
    linarith
  have e10 : ∀ t, √(∑ i, (A.linear P.w2 P.b2 (m' t) i -
      (P.b2 i + ∑ c, act (r t c) * P.w2 c i)) ^ 2) ≤ ef := fun t =>
    (A.sqrt_sum_sq_linear_sub_le' C.sum_le (C.g_nonneg _) (C.h_nonneg _) C.um_nonneg
      C.ηm_nonneg hus hηs C.roundMul_le C.store_le hW2 (x := fun c => act (r t c)) (hYm t)
      (e9 t)).trans hef
  simp only [hA, hE]
  have h1 := sqrt_sum_sq_store_add_sub_le hu hη hus hηs C.round_le C.store_le (e6 t)
    (e10 t)
  refine h1.trans (le_trans ?_ hey)
  have hYt : √(∑ i, (s t i + (P.b2 i + ∑ c, act (r t c) * P.w2 c i)) ^ 2) ≤ Y := by
    have := hY t; simp only [hE] at this; exact this
  have := mul_le_mul_of_nonneg_left hYt hc
  linarith

end TorchLean.TransformerBlock

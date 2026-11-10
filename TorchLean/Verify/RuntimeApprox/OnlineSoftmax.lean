/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import FloatLib.Floats.Formats.BinaryInterchange.Configured
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.CorrectlyRounded
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.MixedError.Rounding
public import TorchLean.Tensor.ConfiguredBinary
public import TorchLean.Tensor.List
public import TorchLean.Tensor.Numeric.Real
public import TorchLean.Verify.RuntimeApprox.OnlineSoftmax.Real
import FloatLib.Numerics.MixedError.Elementary
import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Elementwise -- shake: keep

/-!
# The rounding error of the online softmax schedule

`Softmax.onlineAverage` is FlashAttention's forward schedule for one query, a pass over
blocks of keys keeping a running maximum, a normalizer and a weighted sum. Over the reals it
computes the two-pass value (`attention_online_eq` and `softmax_online_eq` in
`TorchLean.Verify.RuntimeApprox.OnlineSoftmax.Real`), and here we bound its rounding error. The
running maximum is a fold of the IEEE `TorchLean.Context.maximum`, which on finite scores decodes to
the real maximum (`toReal_foldl_maximum`).

The rounding bounds are stated for the one-query iterator, whose inputs are the scores and values.
For `Tensor.attention` these are the computed scores and, with a multiplier, the rounded products
of the multiplier and the value. The online softmax weights of `Tensor.softmax` divide a fresh
exponential by the accumulated normalizer (`Softmax.onlineWeight`, an entry by
`Tensor.softmax_online_apply`). `onlineWeight_error` bounds such a weight against the softmax
weight of its key among the allowed keys (`onlineWeight_reference_eq`), with the normalizer
allowance of `onlineAverage_error` and one rounded shift, exponential and division for the key
itself. An entry of `Tensor.softmax x dim mask (.online blocks)` is such a weight for the scores
and mask of its fiber. `TorchLean.Verify.RuntimeApprox.Roundoff.Rules.OnlineSoftmax` registers
`onlineWeight_error`, and `onlineAverage_error` with `round = id`, as `roundoff` rules for
`onlineWeight` and `onlineAttend`.

Every operation is the configured IEEE operation with nearest-even rounding, and the exponential
is correctly rounded. The weights may pass through a rounding `round` before they multiply the
values and through `roundSum` before the normalizer adds them. FlashAttention, for example, stores
them in `bfloat16` for the matrix product while the normalizer sums them in the working format,
which is `roundSum = id` with `round = roundTo low` (`roundTo`, `roundTo_contract`).

`onlineAverage_error` (with `roundSum = id`) bounds the computed output against the exact softmax
average `O / S` for every IEEE format, every partition of the keys into blocks of at most `B` keys
and every number of keys, with gradual underflow. With `M` the final running maximum (the largest
score when the start is a key's score), each key's weight `exp (s - M)` contributes the relative
allowance `exp (u * (M - s) + c) - 1` to one aggregate bound. A key first seen against the maximum
`m_k` has its shift `s - m_k` and the later changes `m_t - m_{t-1}` rounded. Because the running
maximum only grows, the bounds on these roundings add up to `u * (M - s)`, the bound on the shift
of a single pass. For `T` blocks the count `c` is `B + 3 T - 1` unit roundoffs in the normalizer
and `B + 3 T` plus the weight rounding's `uL` in the weighted sum. Each later block charges the
sums accumulated so far three roundings (the exponential, the product and the sum), or only the
sum when it does not raise the maximum, since the scale and its product are then exact. Underflow
allowances make up the remaining terms. The theorem needs finite inputs, scores at most half the
largest finite value in magnitude, values bounded by `V`, a normalizer budget below the exact
normalizer, and range conditions on these input-only quantities.

Started at a value at least every score, as when the maximum is computed in a first pass, the same
schedule is the blocked two-pass softmax (`onlineAverage_error_of_le`), with `B + T + 1` and
`B + T + 2` units. The uniform bounds differ by `2 u (T - 1)`: a later block that raises the maximum
adds two unit roundoffs to the exponent budget of the accumulated sums, and the uniform online bound
charges this at every block. One block of all keys is the two-pass softmax with sequential sums.
`onlineAverage_blocks_error` and `onlineAverage_blocks_error_of_max` state both bounds in closed
form for blocks of `B` keys, through Gibbs' inequality `∑ w (M - s) ≤ S (log N - log S)`
(`sum_exp_mul_sub_le`): with every scaled shift `u * (M - s)` at most one, the normalizer's relative
error is at most `(exp c - 1) + 2 u exp c log N` plus its underflow residual.
`TorchLean.Verify.RuntimeApprox.OnlineSoftmax.Binary32` evaluates these bounds for `binary32` at
real sizes.

`onlineAverage_stored_error` compares the computation with the exact sums of the weights it
stores (`storedStep`, `StoredInvariant`) instead of the exact weights, for any `roundSum` and
`round` with `|round x - roundSum x| ≤ δ roundSum x + ηs`: the output is within a closed-form bound
of `β z`, with `z` in the range `[lo, hi]` of the values and `|β - 1| ≤ δ + N ηs`. Its exponents
are `u (B + 2 T)` and `u (B + 1 + 2 T)` for every partition, with no shift term, since the shifts
are those of the stored weights. Summing the stored weights gives `β = 1`
(`onlineAverage_stored_error_of_eq`, `onlineAverage_roundTo_error`): the output is within
the bound of a point of `[lo, hi]`, the values' convex hull when `lo` and `hi` are their extremes,
so constant values are kept up to that bound. FlashAttention's recipe meets the contract with the
unit roundoff and underflow allowance of `low` (`onlineAverage_id_roundTo_error`), so its output is
`β z` up to the bound, with `z ∈ [lo, hi]` and `|β - 1| ≤ uL + N ηL`.
`stored_closed_form_le_of_bounds` evaluates these bounds from upper bounds on their exponentials
and powers.

The proof is by hand, for closed forms. `roundoff` states the fold's bound as a recurrence over the
blocks (`TorchLean.Verify.RuntimeApprox.Roundoff.OnlineSoftmax`), and a closed form needs an
invariant over all the keys seen (`Invariant`, `StoredInvariant`), stated with the termwise
allowances of `TermwiseError`.
-/

@[expose] public section

namespace TorchLean.OnlineSoftmax

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange FloatLib.Numerics
open Softmax (onlineStep onlineAverage)
open TorchLean.Roundoff (toReal_foldl_maximum)

section Configured
variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)
local notation "u" => Model.unitRoundoffAt format
local notation "η" => Model.underflowErrorAt format
local notation "maxFinite" => Model.toReal (Model.posMaxFinite format)

/-- A left fold of `max` from a start bounded by `K` over values bounded by `K` is bounded by
`K`. -/
theorem abs_foldl_max_le {κ : Type} (f : κ → ℝ) {K : ℝ} (l : List κ) (start : ℝ)
    (h : |start| ≤ K)
    (hl : ∀ e ∈ l, |f e| ≤ K) : |l.foldl (fun acc e => max acc (f e)) start| ≤ K :=
  abs_le.mpr ⟨(neg_le_of_abs_le h).trans (List.le_foldl_max_init l f start),
    List.foldl_max_le_of_le l f (le_of_abs_le h) fun e he => le_of_abs_le (hl e he)⟩

/-- For finite `x ≤ m` with `|x - m| ≤ maxFinite`, the rounded exponential of the rounded
difference `x - m` is finite, lies in `[0, 1]` and differs from `exp (x - m)` by at most
`exp (x - m) * (exp (u * (m - x) + u) - 1) + η`. -/
theorem exp_sub_error
    (hformat : format.isIEEE = true) {x m : Value}
    (hx : ExecFloat.Binary.isFinite x = true) (hm : ExecFloat.Binary.isFinite m = true)
    (hle : ExecFloat.Binary.toReal x ≤ ExecFloat.Binary.toReal m)
    (hrange : |ExecFloat.Binary.toReal x - ExecFloat.Binary.toReal m| ≤ maxFinite) :
    ExecFloat.Binary.isFinite (ExecFloat.Binary.CorrectlyRounded.exp (x - m)) = true ∧
      ExecFloat.Binary.toReal (ExecFloat.Binary.CorrectlyRounded.exp (x - m)) ∈ Set.Icc 0 1 ∧
      |ExecFloat.Binary.toReal (ExecFloat.Binary.CorrectlyRounded.exp (x - m)) -
          Real.exp (ExecFloat.Binary.toReal x - ExecFloat.Binary.toReal m)| ≤
        Real.exp (ExecFloat.Binary.toReal x - ExecFloat.Binary.toReal m) *
          (Real.exp (u * (ExecFloat.Binary.toReal m - ExecFloat.Binary.toReal x) + u) - 1) + η := by
  set z := ExecFloat.Binary.toReal x - ExecFloat.Binary.toReal m with hz
  have hdfinite : ExecFloat.Binary.isFinite (x - m) = true :=
    ExecFloat.Binary.isFinite_sub_of_mixedError hx hm (.refl _) (by simpa using hrange)
  have hd := (ExecFloat.Binary.sub_rounding hdfinite).bound
  have hdval : ExecFloat.Binary.toReal (x - m) = Model.roundAt format z :=
    ExecFloat.Binary.toReal_sub_eq_roundAt hdfinite
  have hz0 : z ≤ 0 := by linarith
  have hd0 : ExecFloat.Binary.toReal (x - m) ≤ 0 := by
    rw [hdval]; exact Model.roundAt_nonpos format hz0
  have hexp1 : Real.exp (ExecFloat.Binary.toReal (x - m)) ≤ 1 := Real.exp_le_one_iff.mpr hd0
  have hthreshold : Real.exp (ExecFloat.Binary.toReal (x - m)) <
      Model.Transcendentals.Certified.overflowThreshold format :=
    hexp1.trans_lt (Model.Transcendentals.CorrectlyRounded.one_lt_overflowThreshold hformat)
  have hpfinite := ExecFloat.Binary.CorrectlyRounded.isFinite_exp hdfinite hthreshold
    (budget := {})
  have hpval := ExecFloat.Binary.CorrectlyRounded.toReal_exp_eq_roundAt hdfinite hthreshold
    (budget := {})
  refine ⟨hpfinite, ?_, ?_⟩
  · rw [hpval]
    exact Model.roundAt_mem_Icc_zero_one format ⟨(Real.exp_pos _).le, hexp1⟩
  · rw [hpval]
    set d := ExecFloat.Binary.toReal (x - m)
    have hround := (Model.mixedError_roundAt format (Real.exp d)).bound
    rw [abs_of_pos (Real.exp_pos _)] at hround
    have hdz : |d - z| ≤ u * |z| := by simpa using hd
    have habsz : |z| = ExecFloat.Binary.toReal m - ExecFloat.Binary.toReal x := by
      rw [abs_of_nonpos hz0]; ring
    rw [habsz] at hdz
    have hshift : |Real.exp d - Real.exp z| ≤
        Real.exp z *
          (Real.exp (u * (ExecFloat.Binary.toReal m - ExecFloat.Binary.toReal x)) - 1) := by
      have : Real.exp d - Real.exp z = Real.exp z * (Real.exp (d - z) - 1) := by
        rw [mul_sub, ← Real.exp_add]; ring_nf
      rw [this, abs_mul, abs_of_pos (Real.exp_pos _)]
      refine mul_le_mul_of_nonneg_left ((abs_exp_sub_one_le_exp_abs_sub_one _).trans ?_)
        (Real.exp_pos _).le
      linarith [Real.exp_le_exp.mpr hdz]
    have hu : 0 ≤ u := (Model.unitRoundoffAt_pos format).le
    have hed : Real.exp d ≤ Real.exp z *
        Real.exp (u * (ExecFloat.Binary.toReal m - ExecFloat.Binary.toReal x)) := by
      have := (abs_le.mp hshift).2; linarith
    have hexpu : 1 + u ≤ Real.exp u := by linarith [Real.add_one_le_exp u]
    rw [Real.exp_add]
    have hEz := (Real.exp_pos z).le
    have hEm := (Real.exp_pos (u * (ExecFloat.Binary.toReal m - ExecFloat.Binary.toReal x))).le
    calc |Model.roundAt format (Real.exp d) - Real.exp z|
        ≤ |Model.roundAt format (Real.exp d) - Real.exp d| + |Real.exp d - Real.exp z| :=
          abs_sub_le _ _ _
      _ ≤ u * Real.exp d + η + Real.exp z *
          (Real.exp (u * (ExecFloat.Binary.toReal m - ExecFloat.Binary.toReal x)) - 1) := by
          linarith
      _ ≤ _ := by nlinarith [mul_le_mul_of_nonneg_left hexpu (mul_nonneg hEz hEm)]


/-- The weight `exp (s - m)` of a key with score `s` against the running maximum `m`. -/
noncomputable def weight (m : ℝ) (e : Value × Value) : ℝ :=
  Real.exp (ExecFloat.Binary.toReal e.1 - m)

/-- The exponent `u * (m - s) + c` of the relative error allowed for a key with score `s`. -/
noncomputable def shift (c m : ℝ) (e : Value × Value) : ℝ :=
  u * (m - ExecFloat.Binary.toReal e.1) + c

/-- Every weight is nonnegative. -/
theorem weight_nonneg (m : ℝ) (e : Value × Value) : 0 ≤ weight m e := (Real.exp_pos _).le

/-- A key's weight times its allowed factor `exp (u * (m - s) + c)` is at most `exp c`, since the
score is at most `m` and `u ≤ 1`. -/
theorem weight_mul_exp_shift_le {m c : ℝ} {e : Value × Value}
    (he : ExecFloat.Binary.toReal e.1 ≤ m) :
    weight m e * Real.exp (shift c m e) ≤ Real.exp c := by
  unfold weight shift
  rw [← Real.exp_add, Real.exp_le_exp]
  have hu := (Model.unitRoundoffAt_lt_one format).le
  nlinarith

/-- Over keys with scores at most `m` and `|g e| ≤ V`, the terms
`weight m e * |g e| * exp (shift c m e)` sum to at most `P.length * V * exp c`. -/
theorem sum_magnitude_le {m c V : ℝ} {g : Value × Value → ℝ} :
    ∀ (P : List (Value × Value)), (∀ e ∈ P, ExecFloat.Binary.toReal e.1 ≤ m ∧ |g e| ≤ V) →
      (P.map fun e => weight m e * |g e| * Real.exp (shift c m e)).sum ≤
        P.length * V * Real.exp c
  | [], _ => by simp
  | e :: P, h => by
      have he := h e (List.mem_cons_self ..)
      have ih := sum_magnitude_le (c := c) P fun x hx => h x (List.mem_cons_of_mem _ hx)
      simp only [List.map_cons, List.sum_cons, List.length_cons]
      push_cast
      have h1 : weight m e * |g e| * Real.exp (shift c m e) ≤ V * Real.exp c := by
        calc weight m e * |g e| * Real.exp (shift c m e) =
            |g e| * (weight m e * Real.exp (shift c m e)) := by ring
          _ ≤ V * Real.exp c := mul_le_mul he.2 (weight_mul_exp_shift_le he.1)
              (mul_nonneg (weight_nonneg _ _) (Real.exp_pos _).le) ((abs_nonneg _).trans he.2)
      nlinarith

/-- Adding `d` to a shift adds `d` to its constant `c`. -/
theorem shift_add (c d m : ℝ) (e : Value × Value) : shift c m e + d = shift (c + d) m e := by
  unfold shift; ring

/-- Terms with weights `w` and exponents `a0 + c`, where `w * exp a0 ≤ 1` and `|g| ≤ V`, have
total magnitude at most `exp c` times `V` per term. -/
theorem sum_mul_exp_le {κ : Type} {w a0 g : κ → ℝ} {c V : ℝ} :
    ∀ (P : List κ), (∀ e ∈ P, 0 ≤ w e ∧ w e * Real.exp (a0 e) ≤ 1 ∧ |g e| ≤ V) →
      (P.map fun e => w e * |g e| * Real.exp (a0 e + c)).sum ≤ P.length * V * Real.exp c
  | [], _ => by simp
  | e :: P, h => by
      have he := h e (List.mem_cons_self ..)
      have ih := sum_mul_exp_le (c := c) P fun x hx => h x (List.mem_cons_of_mem _ hx)
      simp only [List.map_cons, List.sum_cons, List.length_cons]
      push_cast
      have h1 : w e * |g e| * Real.exp (a0 e + c) ≤ V * Real.exp c := by
        rw [Real.exp_add]
        have hV : 0 ≤ V := (abs_nonneg _).trans he.2.2
        have := mul_le_mul he.2.2 he.2.1 (mul_nonneg he.1 (Real.exp_pos _).le) hV
        calc w e * |g e| * (Real.exp (a0 e) * Real.exp c) =
            |g e| * (w e * Real.exp (a0 e)) * Real.exp c := by ring
          _ ≤ V * 1 * Real.exp c := mul_le_mul_of_nonneg_right this (Real.exp_pos _).le
          _ = V * Real.exp c := by ring
      nlinarith

/-- In a left-to-right sum of rounded additions of terms with weights `w` and exponents `a0 + c`,
each term's exponent grows by `u` per addition. -/
theorem foldl_add_error (hformat : format.isIEEE = true) (term : Value × Value → Value)
    (w a0 g : Value × Value → ℝ) (cterm V r : ℝ) (hr : 0 ≤ r) :
    ∀ (L P : List (Value × Value)) (acc : Value) (c ρ : ℝ), cterm ≤ c → 0 ≤ ρ →
      ExecFloat.Binary.isFinite acc = true →
      TermwiseError (ExecFloat.Binary.toReal acc) P w g (fun x => a0 x + c) ρ →
      (∀ e ∈ P ++ L, 0 ≤ w e ∧ w e * Real.exp (a0 e) ≤ 1 ∧ |g e| ≤ V) →
      (∀ e ∈ L, ExecFloat.Binary.isFinite (term e) = true ∧
        TermwiseError (ExecFloat.Binary.toReal (term e)) [e] w g (fun x => a0 x + cterm) r) →
      ((P.length + L.length : ℕ) : ℝ) * V * Real.exp (c + u * L.length) +
          (1 + u) ^ L.length * (ρ + L.length * r) ≤ maxFinite →
      ExecFloat.Binary.isFinite (L.foldl (fun acc e => acc + term e) acc) = true ∧
        TermwiseError (ExecFloat.Binary.toReal (L.foldl (fun acc e => acc + term e) acc))
          (P ++ L) w g (fun x => a0 x + (c + u * L.length))
          ((1 + u) ^ L.length * (ρ + L.length * r))
  | [], P, acc, c, ρ, _, _, hacc, h, _, _, _ => by simpa using And.intro hacc h
  | e :: L, P, acc, c, ρ, hc, hρ, hacc, h, hmem, hterms, hrange => by
      have hu : 0 ≤ u := (Model.unitRoundoffAt_pos format).le
      obtain ⟨htermFinite, hterm⟩ := hterms e (List.mem_cons_self ..)
      have hterm' : TermwiseError (ExecFloat.Binary.toReal (term e)) [e] w g
          (fun x => a0 x + c) r :=
        hterm.mono (fun x _ => by linarith) le_rfl
          (fun x hx => (hmem x (List.mem_append_right _
            (List.mem_cons.mpr (Or.inl (by simpa using hx))))).1)
      have hsum := h.add hterm'
      have hmemP : ∀ x ∈ P ++ [e], 0 ≤ w x ∧ w x * Real.exp (a0 x) ≤ 1 ∧ |g x| ≤ V := by
        intro x hx; apply hmem; simp only [List.mem_append, List.mem_cons] at hx ⊢; tauto
      have hV : 0 ≤ V := (abs_nonneg _).trans (hmemP e (by simp)).2.2
      -- the exact sum of the computed operands is in range
      have hmag := sum_mul_exp_le (c := c) (P ++ [e]) hmemP
      have hz : |ExecFloat.Binary.toReal acc + ExecFloat.Binary.toReal (term e)| ≤
          ((P ++ [e]).length : ℝ) * V * Real.exp c + (ρ + r) := by
        have hZ := abs_sum_le w g (P := P ++ [e]) (fun x hx => (hmemP x hx).1)
        have hsplit : ((P ++ [e]).map fun x => w x * |g x| *
            (Real.exp (a0 x + c) - 1)).sum + ((P ++ [e]).map fun x => w x * |g x|).sum =
            ((P ++ [e]).map fun x => w x * |g x| * Real.exp (a0 x + c)).sum := by
          rw [← List.sum_map_add]; congr 1; apply List.map_congr_left; intro x _; ring
        unfold TermwiseError at hsum
        have := abs_add_le (ExecFloat.Binary.toReal acc + ExecFloat.Binary.toReal (term e) -
          ((P ++ [e]).map fun x => w x * g x).sum)
          ((P ++ [e]).map fun x => w x * g x).sum
        simp only [sub_add_cancel] at this
        linarith
      have hgrow : ((P ++ [e]).length : ℝ) * V * Real.exp c + (ρ + r) ≤ maxFinite := by
        refine le_trans ?_ hrange
        simp only [List.length_append, List.length_cons, List.length_nil]
        push_cast
        have h1 : Real.exp c ≤ Real.exp (c + u * (L.length + 1)) :=
          Real.exp_le_exp.mpr (by nlinarith)
        have h2 : (1 : ℝ) ≤ (1 + u) ^ (L.length + 1) := one_le_pow₀ (by linarith)
        have h3 : ρ + r ≤ ρ + (L.length + 1) * r := by nlinarith
        have h4 : 0 ≤ ρ + r := by linarith
        have h5 : ((P.length : ℝ) + 1) * V ≤ ((P.length : ℝ) + (L.length + 1)) * V := by
          nlinarith
        have h6 : ((P.length : ℝ) + 1) * V * Real.exp c ≤
            ((P.length : ℝ) + (L.length + 1)) * V * Real.exp (c + u * (L.length + 1)) :=
          mul_le_mul h5 h1 (Real.exp_pos _).le (by positivity)
        nlinarith
      have hfinite : ExecFloat.Binary.isFinite (acc + term e) = true :=
        ExecFloat.Binary.isFinite_add_of_mixedError hacc htermFinite (.refl _)
          (by simpa using hz.trans hgrow)
      have hround := (ExecFloat.Binary.add_rounding hfinite).bound
      have hnext := (hsum.round hround hu (fun x hx => (hmemP x hx).1)).congr
        (a' := fun x => a0 x + (c + u)) (fun x _ => rfl) (fun x _ => by ring)
      have hmem' : ∀ x ∈ (P ++ [e]) ++ L, 0 ≤ w x ∧ w x * Real.exp (a0 x) ≤ 1 ∧ |g x| ≤ V := by
        intro x hx; apply hmem; simpa using hx
      have hrange' : (((P ++ [e]).length + L.length : ℕ) : ℝ) * V *
          Real.exp (c + u + u * L.length) +
          (1 + u) ^ L.length * ((1 + u) * (ρ + r) + 0 + L.length * r) ≤ maxFinite := by
        refine le_trans ?_ hrange
        simp only [List.length_append, List.length_cons, List.length_nil]
        push_cast
        have e1 : c + u + u * L.length = c + u * (L.length + 1) := by ring
        rw [e1]
        have hpow : (1 + u) ^ L.length ≤ (1 + u) ^ (L.length + 1) :=
          pow_le_pow_right₀ (by linarith) (Nat.le_succ _)
        have hLr : 0 ≤ (L.length : ℝ) * r := by positivity
        have : (1 + u) ^ L.length * ((1 + u) * (ρ + r) + 0 + L.length * r) ≤
            (1 + u) ^ (L.length + 1) * (ρ + (L.length + 1) * r) := by
          rw [pow_succ]
          nlinarith [mul_le_mul_of_nonneg_right hpow hLr, pow_succ (1 + u) L.length]
        have e2 : ((P.length : ℝ) + 1 + L.length) = (P.length + (L.length + 1)) := by ring
        rw [e2]
        linarith
      obtain ⟨hfin, herr⟩ := foldl_add_error hformat term w a0 g cterm V r hr L (P ++ [e])
        (acc + term e) (c + u) ((1 + u) * (ρ + r) + 0) (by linarith) (by positivity) hfinite
        hnext hmem' (fun x hx => hterms x (List.mem_cons_of_mem _ hx)) hrange'
      refine ⟨hfin, ?_⟩
      have happ : (P ++ [e]) ++ L = P ++ e :: L := by simp
      rw [happ] at herr
      refine herr.mono (fun x _ => ?_) ?_ (fun x hx => (hmem x hx).1) |>.congr
        (fun x _ => rfl) (fun x _ => rfl)
      · simp only [List.length_cons]; push_cast; linarith
      · simp only [List.length_cons]; push_cast
        have hpow : (1 + u) ^ L.length ≤ (1 + u) ^ (L.length + 1) :=
          pow_le_pow_right₀ (by linarith) (Nat.le_succ _)
        have hLr : 0 ≤ (L.length : ℝ) * r := by positivity
        rw [pow_succ]
        nlinarith [mul_le_mul_of_nonneg_right hpow hLr, pow_succ (1 + u) L.length]

/-- One block's update `scale * x + r` of a running sum, where `scale` is the rounded
`exp (m - m')`, `x` approximates the sum over the earlier keys `P` against the earlier maximum
`m`, and `r` the block's terms against the updated maximum `m'`. The block's terms gain `u` in their
exponents, and the earlier terms gain `3 u` beyond the shift, which telescopes, or `u` when the
maximum is unchanged and `scale` is exactly one. -/
theorem combine_error (hformat : format.isIEEE = true) {m m' : ℝ} {scale x r : Value}
    {P L : List (Value × Value)} {g : Value × Value → ℝ} {V c cblock C ρ ρblock : ℝ}
    (hscale : ExecFloat.Binary.isFinite scale = true ∧
      ExecFloat.Binary.toReal scale ∈ Set.Icc 0 1 ∧
      |ExecFloat.Binary.toReal scale - Real.exp (m - m')| ≤
        Real.exp (m - m') * (Real.exp (u * (m' - m) + u) - 1) + η)
    (hone : m = m' → ExecFloat.Binary.toReal scale = 1) (hmm : m ≤ m')
    (hx : ExecFloat.Binary.isFinite x = true)
    (hxe : TermwiseError (ExecFloat.Binary.toReal x) P (weight m) g (shift c m) ρ)
    (hP : ∀ e ∈ P, ExecFloat.Binary.toReal e.1 ≤ m ∧ |g e| ≤ V)
    (hr : ExecFloat.Binary.isFinite r = true)
    (hre : TermwiseError (ExecFloat.Binary.toReal r) L (weight m') g (shift cblock m') ρblock)
    (hL : ∀ e ∈ L, ExecFloat.Binary.toReal e.1 ≤ m' ∧ |g e| ≤ V)
    (hC : c + (if m = m' then u else 3 * u) ≤ C) (hCblock : cblock + u ≤ C) (hρ : 0 ≤ ρ)
    (hρblock : 0 ≤ ρblock) (hV : 0 ≤ V)
    (hrange : ((P.length + L.length : ℕ) : ℝ) * V * Real.exp C +
      (1 + u) * ((1 + u) * (Real.exp u * ρ + η * (P.length * V * Real.exp c + ρ)) + η +
        ρblock) ≤ maxFinite) :
    ExecFloat.Binary.isFinite (scale * x + r) = true ∧
      TermwiseError (ExecFloat.Binary.toReal (scale * x + r)) (P ++ L) (weight m') g
        (shift C m')
        ((1 + u) * ((1 + u) * (Real.exp u * ρ + η * (P.length * V * Real.exp c + ρ)) + η +
          ρblock)) := by
  have hu : 0 ≤ u := (Model.unitRoundoffAt_pos format).le
  have hu1 : u ≤ 1 := (Model.unitRoundoffAt_lt_one format).le
  have hη : 0 ≤ η := (Model.underflowErrorAt_pos format).le
  obtain ⟨hsf, hs01, hse⟩ := hscale
  have hxmax := ExecFloat.Binary.abs_toReal_le_posMaxFinite_of_isFinite hx
  have hmulFinite : ExecFloat.Binary.isFinite (scale * x) = true :=
    ExecFloat.Binary.isFinite_mul_of_mixedError hsf hx (.refl _) (by
      simp only [one_mul, add_zero, abs_mul]
      calc |ExecFloat.Binary.toReal scale| * |ExecFloat.Binary.toReal x|
          ≤ 1 * |ExecFloat.Binary.toReal x| := by
            gcongr; rw [abs_of_nonneg hs01.1]; exact hs01.2
        _ ≤ _ := by simpa using hxmax)
  have hmag := sum_magnitude_le (c := c) P hP
  have hρmul : ρ ≤ (1 + u) * (Real.exp u * ρ + η * (P.length * V * Real.exp c + ρ)) + η := by
    have h1 : ρ ≤ Real.exp u * ρ := le_mul_of_one_le_left hρ (Real.one_le_exp hu)
    have h2 : 0 ≤ η * (P.length * V * Real.exp c + ρ) := by positivity
    nlinarith
  -- the earlier sum after its product with the scale, against the updated maximum
  have hmul : TermwiseError (ExecFloat.Binary.toReal (scale * x)) P (weight m') g
      (shift (C - u) m')
      ((1 + u) * (Real.exp u * ρ + η * (P.length * V * Real.exp c + ρ)) + η) := by
    by_cases hmeq : m = m'
    · subst hmeq
      simp only [↓reduceIte] at hC
      have hexact : ExecFloat.Binary.toReal (scale * x) = ExecFloat.Binary.toReal x := by
        rw [ExecFloat.Binary.toReal_mul_of_abs_toReal_eq_one_left hsf hx
          (by rw [hone rfl, abs_one]), hone rfl, one_mul]
      rw [hexact]
      exact hxe.mono (fun e _ => by unfold shift; linarith) hρmul (fun e _ => weight_nonneg _ _)
    · simp only [hmeq, ↓reduceIte] at hC
      have hscaled := (hxe.scale hse (Real.exp_pos _).le (fun e _ => weight_nonneg _ _)).congr
        (w' := weight m') (a' := shift (c + u) m')
        (fun e _ => by unfold weight; rw [← Real.exp_add]; ring_nf)
        (fun e _ => by unfold shift; ring)
      have hgrowth : Real.exp (m - m') * Real.exp (u * (m' - m) + u) ≤ Real.exp u := by
        rw [← Real.exp_add, Real.exp_le_exp]; nlinarith
      have hscaled' : TermwiseError (ExecFloat.Binary.toReal scale * ExecFloat.Binary.toReal x) P
          (weight m') g (shift (c + u) m')
          (Real.exp u * ρ + η * (P.length * V * Real.exp c + ρ)) :=
        hscaled.mono (fun _ _ => le_rfl) (by gcongr) (fun e _ => weight_nonneg _ _)
      exact ((hscaled'.round (ExecFloat.Binary.mul_rounding hmulFinite).bound hu
        (fun e _ => weight_nonneg _ _)).congr (fun _ _ => rfl)
        (fun e _ => shift_add _ _ _ e)).mono
        (fun e _ => by unfold shift; linarith) le_rfl (fun e _ => weight_nonneg _ _)
  have hblock := hre.mono (a' := shift (C - u) m') (fun e _ => by unfold shift; linarith) le_rfl
    (fun e _ => weight_nonneg _ _)
  have hsum := hmul.add hblock
  have hPL : ∀ e ∈ P ++ L, ExecFloat.Binary.toReal e.1 ≤ m' ∧ |g e| ≤ V := by
    intro e he
    rcases List.mem_append.mp he with he | he
    · exact ⟨(hP e he).1.trans hmm, (hP e he).2⟩
    · exact hL e he
  have hzle := hsum.abs_le (fun e _ => weight_nonneg _ _)
  have hmagPL := sum_magnitude_le (c := C - u) (P ++ L) hPL
  have hlenV : ((P ++ L).length : ℝ) * V * Real.exp (C - u) ≤
      ((P.length + L.length : ℕ) : ℝ) * V * Real.exp C := by
    rw [List.length_append]
    exact mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr (by linarith)) (by positivity)
  have hres0 : 0 ≤ (1 + u) * (Real.exp u * ρ + η * (P.length * V * Real.exp c + ρ)) + η +
      ρblock := by positivity
  have haddFinite : ExecFloat.Binary.isFinite (scale * x + r) = true :=
    ExecFloat.Binary.isFinite_add_of_mixedError hmulFinite hr (.refl _) (by
      simp only [one_mul, add_zero]
      refine hzle.trans (le_trans ?_ hrange)
      nlinarith)
  refine ⟨haddFinite, ?_⟩
  have hfinal := (hsum.round (ExecFloat.Binary.add_rounding haddFinite).bound hu
    (fun e _ => weight_nonneg _ _)).congr (fun _ _ => rfl)
    (fun e _ => by rw [shift_add, sub_add_cancel])
  simpa using hfinal

/-- The state of online softmax after the keys `P`: the running maximum is finite, bounded by
`K` and at least every score seen, and the normalizer and weighted sum approximate their exact
values against it termwise. -/
def Invariant (P : List (Value × Value)) (state : Value × Value × Value) (K cl co ρl ρo : ℝ) :
    Prop :=
  ExecFloat.Binary.isFinite state.1 = true ∧ ExecFloat.Binary.isFinite state.2.1 = true ∧
    ExecFloat.Binary.isFinite state.2.2 = true ∧ |ExecFloat.Binary.toReal state.1| ≤ K ∧
    (∀ e ∈ P, ExecFloat.Binary.toReal e.1 ≤ ExecFloat.Binary.toReal state.1) ∧
    TermwiseError (ExecFloat.Binary.toReal state.2.1) P (weight (ExecFloat.Binary.toReal state.1))
      (fun _ => 1) (shift cl (ExecFloat.Binary.toReal state.1)) ρl ∧
    TermwiseError (ExecFloat.Binary.toReal state.2.2) P (weight (ExecFloat.Binary.toReal state.1))
      (fun e => ExecFloat.Binary.toReal e.2) (shift co (ExecFloat.Binary.toReal state.1)) ρo


/-- One block step preserves `Invariant`, with the exponents of the earlier terms growing by
`3 u`, or by `u` when the block does not raise the running maximum. The block's terms start at
`u * (b + 2)` in the normalizer and at `u * (b + 3) + uL` in the weighted sum, where `b` is the
block's length. -/
theorem onlineStep_invariant (hformat : format.isIEEE = true)
    (round : Value → Value) {uL ηL : ℝ} (huL : 0 ≤ uL) (hηL : 0 ≤ ηL)
    (hround : ∀ x : Value, ExecFloat.Binary.isFinite x = true →
      ExecFloat.Binary.toReal x ∈ Set.Icc 0 1 →
      ExecFloat.Binary.isFinite (round x) = true ∧
        ExecFloat.Binary.toReal (round x) ∈ Set.Icc 0 1 ∧
        |ExecFloat.Binary.toReal (round x) - ExecFloat.Binary.toReal x| ≤
          uL * |ExecFloat.Binary.toReal x| + ηL)
    {P block : List (Value × Value)} {state : Value × Value × Value} {K V cl co ρl ρo Cl Co : ℝ}
    (hK : 2 * K ≤ maxFinite) (hV : 0 ≤ V)
    (hblock : ∀ e ∈ block, ExecFloat.Binary.isFinite e.1 = true ∧
      ExecFloat.Binary.isFinite e.2 = true ∧ |ExecFloat.Binary.toReal e.1| ≤ K)
    (hvalues : ∀ e ∈ P ++ block, |ExecFloat.Binary.toReal e.2| ≤ V)
    (hinv : Invariant P state K cl co ρl ρo) (hρl : 0 ≤ ρl) (hρo : 0 ≤ ρo)
    (hCl : cl + (if ExecFloat.Binary.toReal state.1 = ExecFloat.Binary.toReal
      (block.foldl (fun m e => Context.maximum m e.1) state.1) then u else 3 * u) ≤ Cl)
    (hCl' : u * (block.length + 2) ≤ Cl)
    (hCo : co + (if ExecFloat.Binary.toReal state.1 = ExecFloat.Binary.toReal
      (block.foldl (fun m e => Context.maximum m e.1) state.1) then u else 3 * u) ≤ Co)
    (hCo' : u * (block.length + 3) + uL ≤ Co)
    (hrangel : ((P.length + block.length : ℕ) : ℝ) * 1 * Real.exp Cl +
      (1 + u) * ((1 + u) * (Real.exp u * ρl + η * (P.length * 1 * Real.exp cl + ρl)) + η +
        (1 + u) ^ block.length * (0 + block.length * η)) ≤ maxFinite)
    (hrangeo : ((P.length + block.length : ℕ) : ℝ) * V * Real.exp Co +
      (1 + u) * ((1 + u) * (Real.exp u * ρo + η * (P.length * V * Real.exp co + ρo)) + η +
        (1 + u) ^ block.length * (0 + block.length * productResidual u η uL ηL V)) ≤ maxFinite) :
    Invariant (P ++ block)
      (onlineStep (fun x => ExecFloat.Binary.CorrectlyRounded.exp x) id round state
        block)
      K Cl
      Co
      ((1 + u) * ((1 + u) * (Real.exp u * ρl + η * (P.length * 1 * Real.exp cl + ρl)) + η +
        (1 + u) ^ block.length * (0 + block.length * η)))
      ((1 + u) * ((1 + u) * (Real.exp u * ρo + η * (P.length * V * Real.exp co + ρo)) + η +
        (1 + u) ^ block.length * (0 + block.length * productResidual u η uL ηL V))) := by
  have hu : 0 ≤ u := (Model.unitRoundoffAt_pos format).le
  have hη : 0 ≤ η := (Model.underflowErrorAt_pos format).le
  obtain ⟨hmf, hlf, hof, hmK, hmP, hle, hoe⟩ := hinv
  obtain ⟨m, l, o⟩ := state
  simp only at hmf hlf hof hmK hmP hle hoe hCl hCo
  -- the running maximum
  obtain ⟨hm'f, hm'val⟩ := toReal_foldl_maximum Prod.fst block m hmf (fun e he => (hblock e he).1)
  set m' := block.foldl (fun m e => Context.maximum m e.1) m with hm'def
  set mR := ExecFloat.Binary.toReal m
  set mR' := ExecFloat.Binary.toReal m'
  have hmm' := List.le_foldl_max_init block (fun e : Value × Value => ExecFloat.Binary.toReal e.1)
    mR
  have hm'block : ∀ e ∈ block, ExecFloat.Binary.toReal e.1 ≤
      block.foldl (fun acc e => max acc (ExecFloat.Binary.toReal e.1)) mR := fun e he =>
    List.le_foldl_max_of_mem block (fun e : Value × Value => ExecFloat.Binary.toReal e.1) he
  rw [← hm'val] at hmm' hm'block
  have hm'K : |mR'| ≤ K := by
    rw [hm'val]; exact abs_foldl_max_le _ block mR hmK (fun e he => (hblock e he).2.2)
  have hsub (x : Value) (hx : |ExecFloat.Binary.toReal x| ≤ K) :
      |ExecFloat.Binary.toReal x - mR'| ≤ maxFinite := by
    have := abs_sub (ExecFloat.Binary.toReal x) mR'
    linarith
  -- the scale, exactly one when the maximum is unchanged
  have hscale := exp_sub_error hformat hmf hm'f hmm' (hsub m hmK)
  have hone : mR = mR' →
      ExecFloat.Binary.toReal (ExecFloat.Binary.CorrectlyRounded.exp (m - m')) = 1 := by
    intro hmeq
    have hdf : ExecFloat.Binary.isFinite (m - m') = true :=
      ExecFloat.Binary.isFinite_sub_of_mixedError hmf hm'f (.refl _)
        (by simpa using hsub m hmK)
    have hd0 : ExecFloat.Binary.toReal (m - m') = 0 := by
      rw [ExecFloat.Binary.toReal_sub_eq_roundAt hdf]
      change Model.roundAt format (mR - mR') = 0
      rw [hmeq, sub_self, Model.roundAt_zero]
    rw [ExecFloat.Binary.CorrectlyRounded.toReal_exp_eq_roundAt hdf (by
      rw [hd0, Real.exp_zero]
      exact Model.Transcendentals.CorrectlyRounded.one_lt_overflowThreshold hformat), hd0,
      Real.exp_zero, Model.roundAt_one]
  -- the normalizer's block sum
  have hlterms : ∀ e ∈ block,
      ExecFloat.Binary.isFinite (ExecFloat.Binary.CorrectlyRounded.exp (e.1 - m')) = true ∧
      TermwiseError (ExecFloat.Binary.toReal (ExecFloat.Binary.CorrectlyRounded.exp (e.1 - m')))
        [e] (weight mR') (fun _ => 1) (shift u mR') η := by
    intro e he
    obtain ⟨hf, -, hb⟩ := exp_sub_error hformat (hblock e he).1 hm'f (hm'block e he)
      (hsub e.1 (hblock e he).2.2)
    refine ⟨hf, ?_⟩
    unfold TermwiseError weight shift
    simpa using hb
  have hpow0 : 0 ≤ (1 + u) ^ block.length := by positivity
  have hlres0 : 0 ≤ (1 + u) ^ block.length * (0 + block.length * η) := by positivity
  have hlold0 : 0 ≤ (1 + u) * (Real.exp u * ρl + η * (P.length * 1 * Real.exp cl + ρl)) + η := by
    positivity
  have hweight (e : Value × Value) (he : ExecFloat.Binary.toReal e.1 ≤ mR') :
      0 ≤ weight mR' e ∧ weight mR' e * Real.exp (u * (mR' - ExecFloat.Binary.toReal e.1)) ≤ 1 :=
    ⟨weight_nonneg _ _, by simpa [shift] using weight_mul_exp_shift_le (c := 0) he⟩
  have hlblock := foldl_add_error hformat
    (fun e => ExecFloat.Binary.CorrectlyRounded.exp (e.1 - m'))
    (weight mR') (fun e => u * (mR' - ExecFloat.Binary.toReal e.1))
    (fun _ => 1) u 1 η hη block [] 0 u 0 le_rfl le_rfl
    ExecFloat.Binary.isFinite_zero (by simp [TermwiseError])
    (fun e he => ⟨(hweight e (hm'block e (by simpa using he))).1,
      (hweight e (hm'block e (by simpa using he))).2, by simp⟩) hlterms (by
      refine le_trans ?_ hrangel
      have h1 : Real.exp (u + u * block.length) ≤ Real.exp Cl := Real.exp_le_exp.mpr (by linarith)
      have h2 : ((([] : List (Value × Value)).length + block.length : ℕ) : ℝ) ≤
          ((P.length + block.length : ℕ) : ℝ) := by simp
      have h3 : ((([] : List (Value × Value)).length + block.length : ℕ) : ℝ) * 1 *
          Real.exp (u + u * block.length) ≤ ((P.length + block.length : ℕ) : ℝ) * 1 * Real.exp Cl :=
        mul_le_mul (by simp) h1 (Real.exp_pos _).le (by positivity)
      nlinarith)
  -- the weighted sum's block terms
  have hoterms : ∀ e ∈ block,
      ExecFloat.Binary.isFinite
        (round (ExecFloat.Binary.CorrectlyRounded.exp (e.1 - m')) * e.2) = true ∧
      TermwiseError (ExecFloat.Binary.toReal
          (round (ExecFloat.Binary.CorrectlyRounded.exp (e.1 - m')) * e.2))
        [e] (weight mR') (fun e => ExecFloat.Binary.toReal e.2) (shift (2 * u + uL) mR')
        (productResidual u η uL ηL V) := by
    intro e he
    obtain ⟨hpf, hp01, -⟩ := exp_sub_error hformat (hblock e he).1 hm'f (hm'block e he)
      (hsub e.1 (hblock e he).2.2)
    obtain ⟨hqf, hq01, hqb⟩ := hround _ hpf hp01
    have hq := ((hlterms e he).2.round hqb huL (by simp [weight_nonneg])).mul_single
      (fun e => ExecFloat.Binary.toReal e.2)
    have hvf := (hblock e he).2.1
    have hmulf : ExecFloat.Binary.isFinite
        (round (ExecFloat.Binary.CorrectlyRounded.exp (e.1 - m')) * e.2) = true :=
      ExecFloat.Binary.isFinite_mul_of_mixedError hqf hvf (.refl _) (by
        simp only [one_mul, add_zero, abs_mul]
        calc _ ≤ 1 * |ExecFloat.Binary.toReal e.2| := by
              gcongr; rw [abs_of_nonneg hq01.1]; exact hq01.2
          _ ≤ _ := by simpa using ExecFloat.Binary.abs_toReal_le_posMaxFinite_of_isFinite hvf)
    refine ⟨hmulf, ?_⟩
    have hr := (hq.round (ExecFloat.Binary.mul_rounding hmulf).bound hu
      (by simp [weight_nonneg])).congr (fun _ _ => rfl)
      (a' := shift (2 * u + uL) mR') (fun x _ => by unfold shift; ring)
    refine hr.mono (fun _ _ => le_rfl) ?_ (by simp [weight_nonneg])
    unfold productResidual
    have hv := hvalues e (List.mem_append_right _ he)
    gcongr
  have hores0 : 0 ≤ productResidual u η uL ηL V := by unfold productResidual; positivity
  have hoblock := foldl_add_error hformat
    (fun e => round (ExecFloat.Binary.CorrectlyRounded.exp (e.1 - m')) * e.2)
    (weight mR') (fun e => u * (mR' - ExecFloat.Binary.toReal e.1))
    (fun e => ExecFloat.Binary.toReal e.2) (2 * u + uL) V (productResidual u η uL ηL V)
    hores0 block [] 0 (2 * u + uL) 0 le_rfl le_rfl
    ExecFloat.Binary.isFinite_zero (by simp [TermwiseError])
    (fun e he => ⟨(hweight e (hm'block e (by simpa using he))).1,
      (hweight e (hm'block e (by simpa using he))).2,
      hvalues e (List.mem_append_right _ (by simpa using he))⟩) hoterms (by
      refine le_trans ?_ hrangeo
      have h1 : Real.exp (2 * u + uL + u * block.length) ≤ Real.exp Co :=
        Real.exp_le_exp.mpr (by linarith)
      have h3 : ((([] : List (Value × Value)).length + block.length : ℕ) : ℝ) * V *
          Real.exp (2 * u + uL + u * block.length) ≤
            ((P.length + block.length : ℕ) : ℝ) * V * Real.exp Co :=
        mul_le_mul (mul_le_mul_of_nonneg_right (by simp) hV) h1 (Real.exp_pos _).le
          (by positivity)
      have hoold0 : 0 ≤ (1 + u) * (Real.exp u * ρo + η * (P.length * V * Real.exp co + ρo)) +
          η := by positivity
      have : 0 ≤ (1 + u) ^ block.length * (0 + block.length * productResidual u η uL ηL V) := by
        positivity
      nlinarith)
  -- combine
  have hlblock2 : TermwiseError (ExecFloat.Binary.toReal (block.foldl
      (fun acc e => acc + ExecFloat.Binary.CorrectlyRounded.exp (e.1 - m')) 0))
      ([] ++ block) (weight mR') (fun _ => 1) (shift (u + u * block.length) mR')
      ((1 + u) ^ block.length * (0 + block.length * η)) := hlblock.2
  have hoblock2 : TermwiseError (ExecFloat.Binary.toReal (block.foldl
      (fun acc e => acc + round (ExecFloat.Binary.CorrectlyRounded.exp (e.1 - m')) * e.2) 0))
      ([] ++ block) (weight mR') (fun e => ExecFloat.Binary.toReal e.2)
      (shift (2 * u + uL + u * block.length) mR')
      ((1 + u) ^ block.length * (0 + block.length * productResidual u η uL ηL V)) := hoblock.2
  have hl' := combine_error hformat hscale hone hmm' hlf hle
    (fun e he => ⟨hmP e he, by simp⟩) hlblock.1 (by simpa using hlblock2)
    (fun e he => ⟨hm'block e he, by simp⟩) hCl (by linarith) hρl hlres0 zero_le_one hrangel
  have ho' := combine_error hformat hscale hone hmm' hof hoe
    (fun e he => ⟨hmP e he, hvalues e (List.mem_append_left _ he)⟩) hoblock.1
    (by simpa using hoblock2)
    (fun e he => ⟨hm'block e he, hvalues e (List.mem_append_right _ he)⟩) hCo (by linarith) hρo
    (by positivity) hV hrangeo
  refine ⟨hm'f, hl'.1, ho'.1, hm'K, fun e he => ?_, hl'.2, ho'.2⟩
  rcases List.mem_append.mp he with he | he
  · exact (hmP e he).trans hmm'
  · exact hm'block e he

/-- The first component of the folded state is the running maximum over all keys. -/
theorem foldl_onlineStep_fst {α : Type} [Context α]
    (exp roundSum round : α → α) : ∀ (blocks : List (List (α × α))) (state : α × α × α),
      (blocks.foldl (onlineStep exp roundSum round) state).1 =
        blocks.flatten.foldl (fun m e => Context.maximum m e.1) state.1
  | [], _ => rfl
  | block :: blocks, state => by
      rw [List.foldl_cons, foldl_onlineStep_fst exp roundSum round blocks, List.flatten_cons,
        List.foldl_append]
      rfl

/-- Folding the block steps preserves `Invariant`, with closed-form exponents `u * (B + 3 t - 1)`
and `u * (B + 3 t) + uL` and residuals `t * κ ^ t * Q` after `t` blocks, where `κ` is the
`residualGrowth` and `Q` the `blockResidual`. -/
theorem foldl_onlineStep_invariant (hformat : format.isIEEE = true)
    (round : Value → Value) {uL ηL : ℝ} (huL : 0 ≤ uL) (hηL : 0 ≤ ηL)
    (hround : ∀ x : Value, ExecFloat.Binary.isFinite x = true →
      ExecFloat.Binary.toReal x ∈ Set.Icc 0 1 →
      ExecFloat.Binary.isFinite (round x) = true ∧
        ExecFloat.Binary.toReal (round x) ∈ Set.Icc 0 1 ∧
        |ExecFloat.Binary.toReal (round x) - ExecFloat.Binary.toReal x| ≤
          uL * |ExecFloat.Binary.toReal x| + ηL)
    {K V : ℝ} {B N T : ℕ} (hK : 2 * K ≤ maxFinite) (hV : 0 ≤ V)
    (hrangel : (N : ℝ) * 1 * Real.exp (u * (B + 3 * T - 1)) +
      T * residualGrowth u η ^ T * blockResidual u η N B 1 (u * (B + 3 * T - 1)) η ≤ maxFinite)
    (hrangeo : (N : ℝ) * V * Real.exp (u * (B + 3 * T) + uL) +
      T * residualGrowth u η ^ T *
        blockResidual u η N B V (u * (B + 3 * T) + uL) (productResidual u η uL ηL V) ≤
      maxFinite) :
    ∀ (blocks : List (List (Value × Value))) (P : List (Value × Value))
      (state : Value × Value × Value) (t : ℕ), t + blocks.length = T →
      (P ++ blocks.flatten).length ≤ N → (∀ block ∈ blocks, block.length ≤ B) →
      (∀ e ∈ blocks.flatten, ExecFloat.Binary.isFinite e.1 = true ∧
        ExecFloat.Binary.isFinite e.2 = true ∧ |ExecFloat.Binary.toReal e.1| ≤ K) →
      (∀ e ∈ P ++ blocks.flatten, |ExecFloat.Binary.toReal e.2| ≤ V) →
      Invariant P state K (u * (B + 3 * t - 1)) (u * (B + 3 * t) + uL)
        (t * residualGrowth u η ^ t * blockResidual u η N B 1 (u * (B + 3 * T - 1)) η)
        (t * residualGrowth u η ^ t *
          blockResidual u η N B V (u * (B + 3 * T) + uL) (productResidual u η uL ηL V)) →
      Invariant (P ++ blocks.flatten)
        (blocks.foldl
          (onlineStep (fun x => ExecFloat.Binary.CorrectlyRounded.exp x) id round)
          state) K (u * (B + 3 * T - 1)) (u * (B + 3 * T) + uL)
        (T * residualGrowth u η ^ T * blockResidual u η N B 1 (u * (B + 3 * T - 1)) η)
        (T * residualGrowth u η ^ T *
          blockResidual u η N B V (u * (B + 3 * T) + uL) (productResidual u η uL ηL V))
  | [], P, state, t, ht, _, _, _, _, hinv => by
      simp only [List.length_nil, add_zero] at ht
      subst ht
      simpa using hinv
  | block :: blocks, P, state, t, ht, hN, hB, hkeys, hvalues, hinv => by
      have hu : 0 ≤ u := (Model.unitRoundoffAt_pos format).le
      have hη : 0 ≤ η := (Model.underflowErrorAt_pos format).le
      have htT : t + 1 ≤ T := by simp at ht; omega
      have hκ := one_le_residualGrowth hu hη
      have hbB := hB block (List.mem_cons_self ..)
      have hPN : P.length + block.length ≤ N := by
        have := hN; simp only [List.length_append, List.flatten_cons] at this; omega
      have hT1 : ((t : ℝ) + 1) ≤ T := by exact_mod_cast htT
      have hres0 (r V' C : ℝ) (hr : 0 ≤ r) (hV' : 0 ≤ V') :
          0 ≤ (t : ℝ) * residualGrowth u η ^ t * blockResidual u η N B V' C r := by
        unfold blockResidual; have : 0 ≤ residualGrowth u η := by linarith
        positivity
      have hroV : 0 ≤ productResidual u η uL ηL V := by unfold productResidual; positivity
      -- residual after the step, bounded by the closed form at `t + 1`
      have hlstep := step_residual_le (n := P.length) (N := N) (b := block.length) (B := B)
        (t := t) (V := 1) (c := u * (B + 3 * t - 1)) (C := u * (B + 3 * T - 1)) (r := η)
        hu hη zero_le_one hη (by omega) hbB (by nlinarith) le_rfl
      have hostep := step_residual_le (n := P.length) (N := N) (b := block.length) (B := B)
        (t := t) (V := V) (c := u * (B + 3 * t) + uL) (C := u * (B + 3 * T) + uL)
        (r := productResidual u η uL ηL V)
        hu hη hV hroV (by omega) hbB (by nlinarith) le_rfl
      have hclose (V' C r : ℝ) (hr : 0 ≤ r) (hV' : 0 ≤ V') :
          ((t + 1 : ℕ) : ℝ) * residualGrowth u η ^ (t + 1) * blockResidual u η N B V' C r ≤
            T * residualGrowth u η ^ T * blockResidual u η N B V' C r := by
        have hQ : 0 ≤ blockResidual u η N B V' C r := by unfold blockResidual; positivity
        have : residualGrowth u η ^ (t + 1) ≤ residualGrowth u η ^ T :=
          pow_le_pow_right₀ hκ htT
        push_cast
        gcongr
      have hNl : ((P.length + block.length : ℕ) : ℝ) ≤ N := by exact_mod_cast hPN
      have hexpl : Real.exp (u * (B + 3 * (t + 1 : ℕ) - 1)) ≤ Real.exp (u * (B + 3 * T - 1)) :=
        Real.exp_le_exp.mpr (by push_cast; nlinarith)
      have hexpo : Real.exp (u * (B + 3 * (t + 1 : ℕ)) + uL) ≤
          Real.exp (u * (B + 3 * T) + uL) :=
        Real.exp_le_exp.mpr (by push_cast; nlinarith)
      have hbBr : (block.length : ℝ) ≤ B := by exact_mod_cast hbB
      have hstep := onlineStep_invariant hformat round huL hηL hround (P := P)
        (block := block) (state := state) (Cl := u * (B + 3 * (t + 1 : ℕ) - 1))
        (Co := u * (B + 3 * (t + 1 : ℕ)) + uL) hK hV
        (fun e he => hkeys e (List.mem_flatten.mpr ⟨block, List.mem_cons_self .., he⟩))
        (fun e he => hvalues e (by
          rcases List.mem_append.mp he with he | he
          · exact List.mem_append_left _ he
          · exact List.mem_append_right _
              (List.mem_flatten.mpr ⟨block, List.mem_cons_self .., he⟩)))
        hinv (hres0 _ _ _ hη zero_le_one) (hres0 _ _ _ hroV hV)
        (by split_ifs <;> push_cast <;> linarith) (by push_cast; nlinarith)
        (by split_ifs <;> push_cast <;> linarith) (by push_cast; nlinarith)
        (by
          refine le_trans ?_ hrangel
          have := hlstep.trans (hclose 1 _ η hη zero_le_one)
          have h1 : ((P.length + block.length : ℕ) : ℝ) * 1 *
              Real.exp (u * (B + 3 * (t + 1 : ℕ) - 1))
              ≤ N * 1 * Real.exp (u * (B + 3 * T - 1)) := by gcongr
          linarith)
        (by
          refine le_trans ?_ hrangeo
          have := hostep.trans (hclose V _ _ hroV hV)
          have h1 : ((P.length + block.length : ℕ) : ℝ) * V *
              Real.exp (u * (B + 3 * (t + 1 : ℕ)) + uL)
              ≤ N * V * Real.exp (u * (B + 3 * T) + uL) := by gcongr
          linarith)
      obtain ⟨h1, h2, h3, h4, h5, h6, h7⟩ := hstep
      have hinv' : Invariant (P ++ block) (onlineStep
          (fun x => ExecFloat.Binary.CorrectlyRounded.exp x) id round state block) K
          (u * (B + 3 * (t + 1 : ℕ) - 1)) (u * (B + 3 * (t + 1 : ℕ)) + uL)
          ((t + 1 : ℕ) * residualGrowth u η ^ (t + 1) *
            blockResidual u η N B 1 (u * (B + 3 * T - 1)) η)
          ((t + 1 : ℕ) * residualGrowth u η ^ (t + 1) *
            blockResidual u η N B V (u * (B + 3 * T) + uL) (productResidual u η uL ηL V)) :=
        ⟨h1, h2, h3, h4, h5, h6.mono (fun _ _ => le_rfl) hlstep (fun _ _ => weight_nonneg _ _),
          h7.mono (fun _ _ => le_rfl) hostep (fun _ _ => weight_nonneg _ _)⟩
      have := foldl_onlineStep_invariant hformat round huL hηL hround hK hV hrangel hrangeo
        blocks (P ++ block) _ (t + 1) (by simp at ht ⊢; omega)
        (by simpa [List.flatten_cons, List.append_assoc] using hN)
        (fun x hx => hB x (List.mem_cons_of_mem _ hx))
        (fun e he => hkeys e (by simp only [List.flatten_cons, List.mem_append]; exact Or.inr he))
        (fun e he => hvalues e (by simpa [List.flatten_cons, List.append_assoc] using he)) hinv'
      simpa [List.flatten_cons, List.append_assoc] using this

/-- Started at a maximum at least as large as every score, the block steps never rescale, and the
exponents of the terms seen so far grow by `u` per block. After `t` blocks the exponents are
`u * (B + t + 1)` and `u * (B + t + 2) + uL` and the residuals `t * κ ^ t * Q`. -/
theorem foldl_onlineStep_invariant_of_le (hformat : format.isIEEE = true)
    (round : Value → Value) {uL ηL : ℝ} (huL : 0 ≤ uL) (hηL : 0 ≤ ηL)
    (hround : ∀ x : Value, ExecFloat.Binary.isFinite x = true →
      ExecFloat.Binary.toReal x ∈ Set.Icc 0 1 →
      ExecFloat.Binary.isFinite (round x) = true ∧
        ExecFloat.Binary.toReal (round x) ∈ Set.Icc 0 1 ∧
        |ExecFloat.Binary.toReal (round x) - ExecFloat.Binary.toReal x| ≤
          uL * |ExecFloat.Binary.toReal x| + ηL)
    {K V : ℝ} {B N T : ℕ} (hK : 2 * K ≤ maxFinite) (hV : 0 ≤ V)
    (hrangel : (N : ℝ) * 1 * Real.exp (u * (B + T + 1)) +
      T * residualGrowth u η ^ T * blockResidual u η N B 1 (u * (B + T + 1)) η ≤ maxFinite)
    (hrangeo : (N : ℝ) * V * Real.exp (u * (B + T + 2) + uL) +
      T * residualGrowth u η ^ T *
        blockResidual u η N B V (u * (B + T + 2) + uL) (productResidual u η uL ηL V) ≤
      maxFinite) :
    ∀ (blocks : List (List (Value × Value))) (P : List (Value × Value))
      (state : Value × Value × Value) (t : ℕ), t + blocks.length = T →
      (P ++ blocks.flatten).length ≤ N → (∀ block ∈ blocks, block.length ≤ B) →
      (∀ e ∈ blocks.flatten, ExecFloat.Binary.isFinite e.1 = true ∧
        ExecFloat.Binary.isFinite e.2 = true ∧ |ExecFloat.Binary.toReal e.1| ≤ K) →
      (∀ e ∈ P ++ blocks.flatten, |ExecFloat.Binary.toReal e.2| ≤ V) →
      (∀ e ∈ blocks.flatten, ExecFloat.Binary.toReal e.1 ≤ ExecFloat.Binary.toReal state.1) →
      Invariant P state K (u * (B + t + 1)) (u * (B + t + 2) + uL)
        (t * residualGrowth u η ^ t * blockResidual u η N B 1 (u * (B + T + 1)) η)
        (t * residualGrowth u η ^ t *
          blockResidual u η N B V (u * (B + T + 2) + uL) (productResidual u η uL ηL V)) →
      Invariant (P ++ blocks.flatten)
        (blocks.foldl
          (onlineStep (fun x => ExecFloat.Binary.CorrectlyRounded.exp x) id round)
          state) K (u * (B + T + 1)) (u * (B + T + 2) + uL)
        (T * residualGrowth u η ^ T * blockResidual u η N B 1 (u * (B + T + 1)) η)
        (T * residualGrowth u η ^ T *
          blockResidual u η N B V (u * (B + T + 2) + uL) (productResidual u η uL ηL V))
  | [], P, state, t, ht, _, _, _, _, _, hinv => by
      simp only [List.length_nil, add_zero] at ht
      subst ht
      simpa using hinv
  | block :: blocks, P, state, t, ht, hN, hB, hkeys, hvalues, hmax, hinv => by
      have hu : 0 ≤ u := (Model.unitRoundoffAt_pos format).le
      -- the block does not raise the maximum
      have hsame : ExecFloat.Binary.toReal state.1 =
          ExecFloat.Binary.toReal (block.foldl (fun m e => Context.maximum m e.1) state.1) := by
        rw [(toReal_foldl_maximum Prod.fst block state.1 hinv.1 (fun e he =>
          (hkeys e (List.mem_flatten.mpr ⟨block, List.mem_cons_self .., he⟩)).1)).2]
        exact le_antisymm (List.le_foldl_max_init block _ _)
          (List.foldl_max_le_of_le block _ le_rfl fun e he =>
            hmax e (List.mem_flatten.mpr ⟨block, List.mem_cons_self .., he⟩))
      have hη : 0 ≤ η := (Model.underflowErrorAt_pos format).le
      have htT : t + 1 ≤ T := by simp at ht; omega
      have hκ := one_le_residualGrowth hu hη
      have hbB := hB block (List.mem_cons_self ..)
      have hPN : P.length + block.length ≤ N := by
        have := hN; simp only [List.length_append, List.flatten_cons] at this; omega
      have hT1 : ((t : ℝ) + 1) ≤ T := by exact_mod_cast htT
      have hres0 (r V' C : ℝ) (hr : 0 ≤ r) (hV' : 0 ≤ V') :
          0 ≤ (t : ℝ) * residualGrowth u η ^ t * blockResidual u η N B V' C r := by
        unfold blockResidual; have : 0 ≤ residualGrowth u η := by linarith
        positivity
      have hroV : 0 ≤ productResidual u η uL ηL V := by unfold productResidual; positivity
      -- residual after the step, bounded by the closed form at `t + 1`
      have hlstep := step_residual_le (n := P.length) (N := N) (b := block.length) (B := B)
        (t := t) (V := 1) (c := u * (B + t + 1)) (C := u * (B + T + 1)) (r := η)
        hu hη zero_le_one hη (by omega) hbB (by nlinarith) le_rfl
      have hostep := step_residual_le (n := P.length) (N := N) (b := block.length) (B := B)
        (t := t) (V := V) (c := u * (B + t + 2) + uL) (C := u * (B + T + 2) + uL)
        (r := productResidual u η uL ηL V)
        hu hη hV hroV (by omega) hbB (by nlinarith) le_rfl
      have hclose (V' C r : ℝ) (hr : 0 ≤ r) (hV' : 0 ≤ V') :
          ((t + 1 : ℕ) : ℝ) * residualGrowth u η ^ (t + 1) * blockResidual u η N B V' C r ≤
            T * residualGrowth u η ^ T * blockResidual u η N B V' C r := by
        have hQ : 0 ≤ blockResidual u η N B V' C r := by unfold blockResidual; positivity
        have : residualGrowth u η ^ (t + 1) ≤ residualGrowth u η ^ T :=
          pow_le_pow_right₀ hκ htT
        push_cast
        gcongr
      have hNl : ((P.length + block.length : ℕ) : ℝ) ≤ N := by exact_mod_cast hPN
      have hexpl : Real.exp (u * (B + (t + 1 : ℕ) + 1)) ≤ Real.exp (u * (B + T + 1)) :=
        Real.exp_le_exp.mpr (by push_cast; nlinarith)
      have hexpo : Real.exp (u * (B + (t + 1 : ℕ) + 2) + uL) ≤
          Real.exp (u * (B + T + 2) + uL) :=
        Real.exp_le_exp.mpr (by push_cast; nlinarith)
      have hbBr : (block.length : ℝ) ≤ B := by exact_mod_cast hbB
      have hstep := onlineStep_invariant hformat round huL hηL hround (P := P)
        (block := block) (state := state) (Cl := u * (B + (t + 1 : ℕ) + 1))
        (Co := u * (B + (t + 1 : ℕ) + 2) + uL) hK hV
        (fun e he => hkeys e (List.mem_flatten.mpr ⟨block, List.mem_cons_self .., he⟩))
        (fun e he => hvalues e (by
          rcases List.mem_append.mp he with he | he
          · exact List.mem_append_left _ he
          · exact List.mem_append_right _
              (List.mem_flatten.mpr ⟨block, List.mem_cons_self .., he⟩)))
        hinv (hres0 _ _ _ hη zero_le_one) (hres0 _ _ _ hroV hV)
        (by simp only [hsame, ↓reduceIte]; push_cast; linarith) (by push_cast; nlinarith)
        (by simp only [hsame, ↓reduceIte]; push_cast; linarith) (by push_cast; nlinarith)
        (by
          refine le_trans ?_ hrangel
          have := hlstep.trans (hclose 1 _ η hη zero_le_one)
          have h1 : ((P.length + block.length : ℕ) : ℝ) * 1 *
              Real.exp (u * (B + (t + 1 : ℕ) + 1))
              ≤ N * 1 * Real.exp (u * (B + T + 1)) := by gcongr
          linarith)
        (by
          refine le_trans ?_ hrangeo
          have := hostep.trans (hclose V _ _ hroV hV)
          have h1 : ((P.length + block.length : ℕ) : ℝ) * V *
              Real.exp (u * (B + (t + 1 : ℕ) + 2) + uL)
              ≤ N * V * Real.exp (u * (B + T + 2) + uL) := by gcongr
          linarith)
      obtain ⟨h1, h2, h3, h4, h5, h6, h7⟩ := hstep
      have hinv' : Invariant (P ++ block) (onlineStep
          (fun x => ExecFloat.Binary.CorrectlyRounded.exp x) id round state block) K
          (u * (B + (t + 1 : ℕ) + 1)) (u * (B + (t + 1 : ℕ) + 2) + uL)
          ((t + 1 : ℕ) * residualGrowth u η ^ (t + 1) *
            blockResidual u η N B 1 (u * (B + T + 1)) η)
          ((t + 1 : ℕ) * residualGrowth u η ^ (t + 1) *
            blockResidual u η N B V (u * (B + T + 2) + uL) (productResidual u η uL ηL V)) :=
        ⟨h1, h2, h3, h4, h5, h6.mono (fun _ _ => le_rfl) hlstep (fun _ _ => weight_nonneg _ _),
          h7.mono (fun _ _ => le_rfl) hostep (fun _ _ => weight_nonneg _ _)⟩
      have hmax' : ∀ e ∈ blocks.flatten, ExecFloat.Binary.toReal e.1 ≤ ExecFloat.Binary.toReal
          (onlineStep (fun x => ExecFloat.Binary.CorrectlyRounded.exp x) id round state
            block).1 := fun e he =>
        (hmax e (by simp only [List.flatten_cons, List.mem_append]; exact Or.inr he)).trans
          (le_of_eq hsame)
      have := foldl_onlineStep_invariant_of_le hformat round huL hηL hround hK hV hrangel
        hrangeo blocks (P ++ block)
        (onlineStep (fun x => ExecFloat.Binary.CorrectlyRounded.exp x) id round state
          block) (t + 1) (by simp at ht ⊢; omega)
        (by simpa [List.flatten_cons, List.append_assoc] using hN)
        (fun x hx => hB x (List.mem_cons_of_mem _ hx))
        (fun e he => hkeys e (by simp only [List.flatten_cons, List.mem_append]; exact Or.inr he))
        (fun e he => hvalues e (by simpa [List.flatten_cons, List.append_assoc] using he))
        hmax' hinv'
      simpa [List.flatten_cons, List.append_assoc] using this

/-- If a finite normalizer and weighted sum are within `El` and `Eo` of `S` and `O`, with `El < S`
and `(|O| + Eo) / (S - El) ≤ maxFinite`, their rounded quotient is finite and within
`(1 + u) (Eo + |O / S| El) / (S - El) + u |O / S| + η` of `O / S`. -/
theorem div_error (hformat : format.isIEEE = true) {l o : Value} {S O El Eo : ℝ}
    (hlf : ExecFloat.Binary.isFinite l = true) (hof : ExecFloat.Binary.isFinite o = true)
    (hl : |ExecFloat.Binary.toReal l - S| ≤ El) (ho : |ExecFloat.Binary.toReal o - O| ≤ Eo)
    (hden : El < S) (hquot : (|O| + Eo) / (S - El) ≤ maxFinite) :
    ExecFloat.Binary.isFinite (o / l) = true ∧
      |ExecFloat.Binary.toReal (o / l) - O / S| ≤
        (1 + u) * ((Eo + |O / S| * El) / (S - El)) + u * |O / S| + η := by
  have hu : 0 ≤ u := (Model.unitRoundoffAt_pos format).le
  obtain ⟨hlow, hratio⟩ := abs_div_sub_div_le hl ho hden
  have hEl : 0 ≤ El := (abs_nonneg _).trans hl
  have hpos : 0 < ExecFloat.Binary.toReal l := by linarith
  have hzero : ExecFloat.Binary.isZero l = false := by
    rcases h : ExecFloat.Binary.isZero l
    · rfl
    · have := (ExecFloat.Binary.isZero_eq_true_iff_toReal_eq_zero hlf).mp h
      linarith
  have hquot' : |ExecFloat.Binary.toReal o / ExecFloat.Binary.toReal l| ≤
      maxFinite := by
    refine le_trans ?_ hquot
    rw [abs_div, abs_of_pos hpos]
    have ho' : |ExecFloat.Binary.toReal o| ≤ |O| + Eo := by
      have := abs_add_le (ExecFloat.Binary.toReal o - O) O
      simp only [sub_add_cancel] at this; linarith
    have hEo : 0 ≤ Eo := (abs_nonneg _).trans ho
    exact div_le_div₀ (by positivity) ho' (by linarith) hlow
  have hdivf : ExecFloat.Binary.isFinite (o / l) = true :=
    ExecFloat.Binary.isFinite_div_of_mixedError hof hlf hzero (.refl _)
      (by simpa using hquot')
  have hround' := (ExecFloat.Binary.div_rounding hlf hdivf).bound
  refine ⟨hdivf, ?_⟩
  set q := ExecFloat.Binary.toReal o / ExecFloat.Binary.toReal l
  have hq : |q| ≤ |O / S| + (Eo + |O / S| * El) / (S - El) := by
    have := abs_add_le (q - O / S) (O / S)
    simp only [sub_add_cancel] at this; linarith
  calc |ExecFloat.Binary.toReal (o / l) - O / S|
      ≤ |ExecFloat.Binary.toReal (o / l) - q| + |q - O / S| := abs_sub_le _ _ _
    _ ≤ u * |q| + η + (Eo + |O / S| * El) / (S - El) := add_le_add hround' hratio
    _ ≤ u * (|O / S| + (Eo + |O / S| * El) / (S - El)) + η + (Eo + |O / S| * El) / (S - El) := by
        gcongr
    _ = _ := by ring

/-- Online softmax over any partition of the keys into blocks of at most `B` keys, in any IEEE
format and with weights rounded by `round` with relative error `uL` and absolute error `ηL` on
`[0, 1]`, is finite and within `(1 + u) (Eo + |O / S| El) / (S - El) + u |O / S| + η` of the
softmax average `O / S`. It needs finite inputs, scores bounded by `K` with `2 K ≤ maxFinite`,
values bounded by `V`, `El < S` and the range conditions. `El` and `Eo` give each key's weight the
relative allowance `exp (u * (M - s) + c) - 1`, with `c` of `B + 3 T - 1` units in `El` and
`B + 3 T` units plus `uL` in `Eo` for `T` blocks, plus underflow residuals. -/
theorem onlineAverage_error (hformat : format.isIEEE = true) (round : Value → Value) {uL ηL : ℝ}
    (huL : 0 ≤ uL) (hηL : 0 ≤ ηL)
    (hround : ∀ x : Value, ExecFloat.Binary.isFinite x = true →
      ExecFloat.Binary.toReal x ∈ Set.Icc 0 1 →
      ExecFloat.Binary.isFinite (round x) = true ∧
        ExecFloat.Binary.toReal (round x) ∈ Set.Icc 0 1 ∧
        |ExecFloat.Binary.toReal (round x) - ExecFloat.Binary.toReal x| ≤
          uL * |ExecFloat.Binary.toReal x| + ηL)
    (start : Value) (blocks : List (List (Value × Value))) {K V : ℝ} {B : ℕ}
    (hK : 2 * K ≤ maxFinite) (hV : 0 ≤ V) (hstart : ExecFloat.Binary.isFinite start = true)
    (hstartK : |ExecFloat.Binary.toReal start| ≤ K) (hB : ∀ block ∈ blocks, block.length ≤ B)
    (hkeys : ∀ e ∈ blocks.flatten, ExecFloat.Binary.isFinite e.1 = true ∧
      ExecFloat.Binary.isFinite e.2 = true ∧ |ExecFloat.Binary.toReal e.1| ≤ K ∧
      |ExecFloat.Binary.toReal e.2| ≤ V) :
    let N := blocks.flatten.length
    let T := blocks.length
    let M := blocks.flatten.foldl (fun m e => max m (ExecFloat.Binary.toReal e.1))
      (ExecFloat.Binary.toReal start)
    let S := (blocks.flatten.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - M)).sum
    let O := (blocks.flatten.map fun e =>
      Real.exp (ExecFloat.Binary.toReal e.1 - M) * ExecFloat.Binary.toReal e.2).sum
    let El := (blocks.flatten.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - M) *
        (Real.exp (u * (M - ExecFloat.Binary.toReal e.1) + u * (B + 3 * T - 1)) - 1)).sum +
      T * residualGrowth u η ^ T * blockResidual u η N B 1 (u * (B + 3 * T - 1)) η
    let Eo := (blocks.flatten.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - M) *
        |ExecFloat.Binary.toReal e.2| *
        (Real.exp (u * (M - ExecFloat.Binary.toReal e.1) + (u * (B + 3 * T) + uL)) - 1)).sum +
      T * residualGrowth u η ^ T *
        blockResidual u η N B V (u * (B + 3 * T) + uL) (productResidual u η uL ηL V)
    (N : ℝ) * 1 * Real.exp (u * (B + 3 * T - 1)) +
        T * residualGrowth u η ^ T * blockResidual u η N B 1 (u * (B + 3 * T - 1)) η ≤
      maxFinite →
    (N : ℝ) * V * Real.exp (u * (B + 3 * T) + uL) +
        T * residualGrowth u η ^ T *
          blockResidual u η N B V (u * (B + 3 * T) + uL) (productResidual u η uL ηL V) ≤
      maxFinite →
    El < S → (|O| + Eo) / (S - El) ≤ maxFinite →
    ExecFloat.Binary.isFinite
        (onlineAverage (fun x => ExecFloat.Binary.CorrectlyRounded.exp x) id round start
          blocks) =
      true ∧
      |ExecFloat.Binary.toReal
          (onlineAverage (fun x => ExecFloat.Binary.CorrectlyRounded.exp x) id round start
            blocks) -
        O / S| ≤ (1 + u) * ((Eo + |O / S| * El) / (S - El)) + u * |O / S| + η := by
  intro N T M S O El Eo hrangel hrangeo hden hquot
  have hu : 0 ≤ u := (Model.unitRoundoffAt_pos format).le
  have hinv := foldl_onlineStep_invariant hformat round huL hηL hround (B := B) (N := N)
    (T := T) hK hV hrangel hrangeo blocks [] (start, 0, 0) 0 (by simp [T]) (by simp [N]) hB
    (fun e he => ⟨(hkeys e he).1, (hkeys e he).2.1, (hkeys e he).2.2.1⟩)
    (fun e he => (hkeys e (by simpa using he)).2.2.2)
    ⟨hstart, ExecFloat.Binary.isFinite_zero, ExecFloat.Binary.isFinite_zero, hstartK,
      by simp, by simp [TermwiseError], by simp [TermwiseError]⟩
  obtain ⟨hmf, hlf, hof, -, -, hle, hoe⟩ := hinv
  set final := blocks.foldl
    (onlineStep (fun x => ExecFloat.Binary.CorrectlyRounded.exp x) id round)
    (start, 0, 0)
  have hM : ExecFloat.Binary.toReal final.1 = M := by
    rw [foldl_onlineStep_fst]
    exact (toReal_foldl_maximum Prod.fst blocks.flatten start hstart
      (fun e he => (hkeys e he).1)).2
  rw [hM] at hle hoe
  simp only [List.nil_append] at hle hoe
  have hl : |ExecFloat.Binary.toReal final.2.1 - S| ≤ El := by
    unfold TermwiseError weight shift at hle
    simpa [S, El, M] using hle
  have ho : |ExecFloat.Binary.toReal final.2.2 - O| ≤ Eo := by
    unfold TermwiseError weight shift at hoe
    simpa [O, Eo, M] using hoe
  exact div_error hformat hlf hof hl ho hden hquot

/-- Started at a value at least as large as every score, online softmax is the blocked two-pass
softmax and satisfies the bound of `onlineAverage_error` with `M` the starting value and `c` of
`B + T + 1` units and `B + T + 2` units plus `uL`. -/
theorem onlineAverage_error_of_le (hformat : format.isIEEE = true) (round : Value → Value)
    {uL ηL : ℝ}
    (huL : 0 ≤ uL) (hηL : 0 ≤ ηL)
    (hround : ∀ x : Value, ExecFloat.Binary.isFinite x = true →
      ExecFloat.Binary.toReal x ∈ Set.Icc 0 1 →
      ExecFloat.Binary.isFinite (round x) = true ∧
        ExecFloat.Binary.toReal (round x) ∈ Set.Icc 0 1 ∧
        |ExecFloat.Binary.toReal (round x) - ExecFloat.Binary.toReal x| ≤
          uL * |ExecFloat.Binary.toReal x| + ηL)
    (start : Value) (blocks : List (List (Value × Value))) {K V : ℝ} {B : ℕ}
    (hK : 2 * K ≤ maxFinite) (hV : 0 ≤ V) (hstart : ExecFloat.Binary.isFinite start = true)
    (hstartK : |ExecFloat.Binary.toReal start| ≤ K) (hB : ∀ block ∈ blocks, block.length ≤ B)
    (hkeys : ∀ e ∈ blocks.flatten, ExecFloat.Binary.isFinite e.1 = true ∧
      ExecFloat.Binary.isFinite e.2 = true ∧ |ExecFloat.Binary.toReal e.1| ≤ K ∧
      |ExecFloat.Binary.toReal e.2| ≤ V)
    (hmax : ∀ e ∈ blocks.flatten, ExecFloat.Binary.toReal e.1 ≤ ExecFloat.Binary.toReal start) :
    let N := blocks.flatten.length
    let T := blocks.length
    let M := ExecFloat.Binary.toReal start
    let S := (blocks.flatten.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - M)).sum
    let O := (blocks.flatten.map fun e =>
      Real.exp (ExecFloat.Binary.toReal e.1 - M) * ExecFloat.Binary.toReal e.2).sum
    let El := (blocks.flatten.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - M) *
        (Real.exp (u * (M - ExecFloat.Binary.toReal e.1) + u * (B + T + 1)) - 1)).sum +
      T * residualGrowth u η ^ T * blockResidual u η N B 1 (u * (B + T + 1)) η
    let Eo := (blocks.flatten.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - M) *
        |ExecFloat.Binary.toReal e.2| *
        (Real.exp (u * (M - ExecFloat.Binary.toReal e.1) + (u * (B + T + 2) + uL)) - 1)).sum +
      T * residualGrowth u η ^ T *
        blockResidual u η N B V (u * (B + T + 2) + uL) (productResidual u η uL ηL V)
    (N : ℝ) * 1 * Real.exp (u * (B + T + 1)) +
        T * residualGrowth u η ^ T * blockResidual u η N B 1 (u * (B + T + 1)) η ≤
      maxFinite →
    (N : ℝ) * V * Real.exp (u * (B + T + 2) + uL) +
        T * residualGrowth u η ^ T *
          blockResidual u η N B V (u * (B + T + 2) + uL) (productResidual u η uL ηL V) ≤
      maxFinite →
    El < S → (|O| + Eo) / (S - El) ≤ maxFinite →
    ExecFloat.Binary.isFinite
        (onlineAverage (fun x => ExecFloat.Binary.CorrectlyRounded.exp x) id round start
          blocks) =
      true ∧
      |ExecFloat.Binary.toReal
          (onlineAverage (fun x => ExecFloat.Binary.CorrectlyRounded.exp x) id round start
            blocks) -
        O / S| ≤ (1 + u) * ((Eo + |O / S| * El) / (S - El)) + u * |O / S| + η := by
  intro N T M S O El Eo hrangel hrangeo hden hquot
  have hu : 0 ≤ u := (Model.unitRoundoffAt_pos format).le
  have hinv := foldl_onlineStep_invariant_of_le hformat round huL hηL hround (B := B)
    (N := N) (T := T) hK hV hrangel hrangeo blocks [] (start, 0, 0) 0 (by simp [T]) (by simp [N])
    hB (fun e he => ⟨(hkeys e he).1, (hkeys e he).2.1, (hkeys e he).2.2.1⟩)
    (fun e he => (hkeys e (by simpa using he)).2.2.2) hmax
    ⟨hstart, ExecFloat.Binary.isFinite_zero, ExecFloat.Binary.isFinite_zero, hstartK,
      by simp, by simp [TermwiseError], by simp [TermwiseError]⟩
  obtain ⟨hmf, hlf, hof, -, -, hle, hoe⟩ := hinv
  set final := blocks.foldl
    (onlineStep (fun x => ExecFloat.Binary.CorrectlyRounded.exp x) id round)
    (start, 0, 0)
  have hM : ExecFloat.Binary.toReal final.1 = M := by
    rw [foldl_onlineStep_fst]
    rw [(toReal_foldl_maximum Prod.fst blocks.flatten start hstart (fun e he => (hkeys e he).1)).2]
    exact le_antisymm (List.foldl_max_le_of_le _ _ le_rfl hmax) (List.le_foldl_max_init _ _ _)
  rw [hM] at hle hoe
  simp only [List.nil_append] at hle hoe
  have hl : |ExecFloat.Binary.toReal final.2.1 - S| ≤ El := by
    unfold TermwiseError weight shift at hle
    simpa [S, El, M] using hle
  have ho : |ExecFloat.Binary.toReal final.2.2 - O| ≤ Eo := by
    unfold TermwiseError weight shift at hoe
    simpa [O, Eo, M] using hoe
  exact div_error hformat hlf hof hl ho hden hquot

/-- Online softmax over blocks of `B` keys, started at the first key's score, is finite in any IEEE
format and within `(1 + u) (V (ao + al + ρl) + ρo) / (1 - al - ρl) + u V + η` of the softmax
average, given `2 K u ≤ 1`, `al + ρl < 1` and the range conditions. Here `T = ⌈N / B⌉`,
`al = (exp cl - 1) + 2 u exp cl log N` with `cl = u * (B + 3 T - 1)`, `ao` is the same expression
in `co = u * (B + 3 T) + uL`, and `ρl` and `ρo` are underflow residuals. -/
theorem onlineAverage_blocks_error (hformat : format.isIEEE = true) (round : Value → Value)
    {uL ηL : ℝ} (huL : 0 ≤ uL) (hηL : 0 ≤ ηL)
    (hround : ∀ x : Value, ExecFloat.Binary.isFinite x = true →
      ExecFloat.Binary.toReal x ∈ Set.Icc 0 1 →
      ExecFloat.Binary.isFinite (round x) = true ∧
        ExecFloat.Binary.toReal (round x) ∈ Set.Icc 0 1 ∧
        |ExecFloat.Binary.toReal (round x) - ExecFloat.Binary.toReal x| ≤
          uL * |ExecFloat.Binary.toReal x| + ηL)
    {B : ℕ} (hB : 0 < B) (first : Value × Value) (rest : List (Value × Value)) {K V : ℝ}
    (hK : 2 * K ≤ maxFinite) (hKu : 2 * K * u ≤ 1) (hV : 0 ≤ V)
    (hkeys : ∀ e ∈ first :: rest, ExecFloat.Binary.isFinite e.1 = true ∧
      ExecFloat.Binary.isFinite e.2 = true ∧ |ExecFloat.Binary.toReal e.1| ≤ K ∧
      |ExecFloat.Binary.toReal e.2| ≤ V) :
    let N := (first :: rest).length
    let T := (N + B - 1) / B
    let M := (first :: rest).foldl (fun m e => max m (ExecFloat.Binary.toReal e.1))
      (ExecFloat.Binary.toReal first.1)
    let S := ((first :: rest).map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - M)).sum
    let O := ((first :: rest).map fun e =>
      Real.exp (ExecFloat.Binary.toReal e.1 - M) * ExecFloat.Binary.toReal e.2).sum
    let al := (Real.exp (u * (B + 3 * T - 1)) - 1) +
      2 * u * Real.exp (u * (B + 3 * T - 1)) * Real.log N
    let ao := (Real.exp (u * (B + 3 * T) + uL) - 1) +
      2 * u * Real.exp (u * (B + 3 * T) + uL) * Real.log N
    let ρl := T * residualGrowth u η ^ T * blockResidual u η N B 1 (u * (B + 3 * T - 1)) η
    let ρo := T * residualGrowth u η ^ T *
      blockResidual u η N B V (u * (B + 3 * T) + uL) (productResidual u η uL ηL V)
    (N : ℝ) * 1 * Real.exp (u * (B + 3 * T - 1)) + ρl ≤ maxFinite →
    (N : ℝ) * V * Real.exp (u * (B + 3 * T) + uL) + ρo ≤ maxFinite →
    al + ρl < 1 → (V + V * ao + ρo) / (1 - al - ρl) ≤ maxFinite →
    ExecFloat.Binary.isFinite (onlineAverage (fun x => ExecFloat.Binary.CorrectlyRounded.exp x)
        id round first.1 (List.toChunks B (first :: rest))) = true ∧
      |ExecFloat.Binary.toReal (onlineAverage
          (fun x => ExecFloat.Binary.CorrectlyRounded.exp x) id round first.1
          (List.toChunks B (first :: rest))) - O / S| ≤
        (1 + u) * ((V * (ao + al + ρl) + ρo) / (1 - al - ρl)) + u * V + η := by
  intro N T M S O al ao ρl ρo hrangel hrangeo hsmall hquot
  have hu : 0 ≤ u := (Model.unitRoundoffAt_pos format).le
  have hη : 0 ≤ η := (Model.underflowErrorAt_pos format).le
  have hkeysne : (first :: rest) ≠ [] := List.cons_ne_nil _ _
  have main := onlineAverage_error hformat round huL hηL hround first.1
    (List.toChunks B (first :: rest)) hK hV (hkeys first (List.mem_cons_self ..)).1
    (hkeys first (List.mem_cons_self ..)).2.2.1 (List.length_le_of_mem_toChunks hB _)
    (by rw [List.flatten_toChunks]; exact hkeys)
  simp only [List.flatten_toChunks, List.length_toChunks hB] at main
  -- the largest score is a key's score
  have hattained : ∃ e ∈ first :: rest, ExecFloat.Binary.toReal e.1 = M := by
    rcases List.foldl_max_eq_init_or_mem (first :: rest) (fun e => ExecFloat.Binary.toReal e.1)
        (ExecFloat.Binary.toReal first.1) with h | ⟨e, he, h⟩
    · exact ⟨first, List.mem_cons_self .., h.symm⟩
    · exact ⟨e, he, h.symm⟩
  have hshift : ∀ e ∈ first :: rest, ExecFloat.Binary.toReal e.1 ≤ M ∧
      u * (M - ExecFloat.Binary.toReal e.1) ≤ 1 := by
    intro e he
    have hle := List.le_foldl_max_of_mem (first :: rest)
      (fun e : Value × Value => ExecFloat.Binary.toReal e.1)
      (acc := ExecFloat.Binary.toReal first.1)
      he
    refine ⟨hle, ?_⟩
    have hMK : |M| ≤ K := abs_foldl_max_le _ _ _ (hkeys first (List.mem_cons_self ..)).2.2.1
      (fun e he => (hkeys e he).2.2.1)
    have := (hkeys e he).2.2.1
    have hdiff : M - ExecFloat.Binary.toReal e.1 ≤ 2 * K := by
      have := abs_sub M (ExecFloat.Binary.toReal e.1)
      linarith [le_abs_self (M - ExecFloat.Binary.toReal e.1)]
    nlinarith
  have hT : 0 ≤ (B : ℝ) + 3 * T - 1 := by
    have : (1 : ℝ) ≤ B := by exact_mod_cast hB
    have : (0 : ℝ) ≤ T := by positivity
    linarith
  have hρ (V' C r : ℝ) (hV' : 0 ≤ V') (hr : 0 ≤ r) :
      0 ≤ (T : ℝ) * residualGrowth u η ^ T * blockResidual u η N B V' C r := by
    have := one_le_residualGrowth hu hη
    have : 0 ≤ residualGrowth u η := by linarith
    unfold blockResidual; positivity
  obtain ⟨hden, hquot', hfinal⟩ := budget_closed_form (first :: rest)
    (fun e => ExecFloat.Binary.toReal e.1) (fun e => ExecFloat.Binary.toReal e.2) (under := η)
    (cl := u * (B + 3 * T - 1)) (co := u * (B + 3 * T) + uL) (ρl := ρl) (ρo := ρo)
    (List.cons_ne_nil _ _) hu (mul_nonneg hu hT) (by positivity) (hρ _ _ _ zero_le_one hη)
    (hρ _ _ _ hV (by unfold productResidual; positivity)) hV hattained hshift
    (fun e he => (hkeys e he).2.2.2) hsmall
  obtain ⟨hfinite, hbound⟩ := main hrangel hrangeo hden (hquot'.trans hquot)
  exact ⟨hfinite, hbound.trans hfinal⟩


/-- Summed against a shift `start` equal to the largest score, the blocked two-pass softmax over
blocks of `B` keys satisfies the closed-form bound of `onlineAverage_blocks_error` with
`cl = u * (B + T + 1)` and `co = u * (B + T + 2) + uL`. -/
theorem onlineAverage_blocks_error_of_max (hformat : format.isIEEE = true) (round : Value → Value)
    {uL ηL : ℝ} (huL : 0 ≤ uL) (hηL : 0 ≤ ηL)
    (hround : ∀ x : Value, ExecFloat.Binary.isFinite x = true →
      ExecFloat.Binary.toReal x ∈ Set.Icc 0 1 →
      ExecFloat.Binary.isFinite (round x) = true ∧
        ExecFloat.Binary.toReal (round x) ∈ Set.Icc 0 1 ∧
        |ExecFloat.Binary.toReal (round x) - ExecFloat.Binary.toReal x| ≤
          uL * |ExecFloat.Binary.toReal x| + ηL)
    {B : ℕ} (hB : 0 < B) (first : Value × Value) (rest : List (Value × Value)) {K V : ℝ}
    (hK : 2 * K ≤ maxFinite) (hKu : 2 * K * u ≤ 1) (hV : 0 ≤ V)
    (hkeys : ∀ e ∈ first :: rest, ExecFloat.Binary.isFinite e.1 = true ∧
      ExecFloat.Binary.isFinite e.2 = true ∧ |ExecFloat.Binary.toReal e.1| ≤ K ∧
      |ExecFloat.Binary.toReal e.2| ≤ V)
    (start : Value) (hstart : ExecFloat.Binary.isFinite start = true)
    (hmax : ∀ e ∈ first :: rest, ExecFloat.Binary.toReal e.1 ≤ ExecFloat.Binary.toReal start)
    (hattained : ∃ e ∈ first :: rest, ExecFloat.Binary.toReal e.1 = ExecFloat.Binary.toReal start) :
    let N := (first :: rest).length
    let T := (N + B - 1) / B
    let M := ExecFloat.Binary.toReal start
    let S := ((first :: rest).map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - M)).sum
    let O := ((first :: rest).map fun e =>
      Real.exp (ExecFloat.Binary.toReal e.1 - M) * ExecFloat.Binary.toReal e.2).sum
    let al := (Real.exp (u * (B + T + 1)) - 1) +
      2 * u * Real.exp (u * (B + T + 1)) * Real.log N
    let ao := (Real.exp (u * (B + T + 2) + uL) - 1) +
      2 * u * Real.exp (u * (B + T + 2) + uL) * Real.log N
    let ρl := T * residualGrowth u η ^ T * blockResidual u η N B 1 (u * (B + T + 1)) η
    let ρo := T * residualGrowth u η ^ T *
      blockResidual u η N B V (u * (B + T + 2) + uL) (productResidual u η uL ηL V)
    (N : ℝ) * 1 * Real.exp (u * (B + T + 1)) + ρl ≤ maxFinite →
    (N : ℝ) * V * Real.exp (u * (B + T + 2) + uL) + ρo ≤ maxFinite →
    al + ρl < 1 → (V + V * ao + ρo) / (1 - al - ρl) ≤ maxFinite →
    ExecFloat.Binary.isFinite (onlineAverage (fun x => ExecFloat.Binary.CorrectlyRounded.exp x)
        id round start (List.toChunks B (first :: rest))) = true ∧
      |ExecFloat.Binary.toReal (onlineAverage
          (fun x => ExecFloat.Binary.CorrectlyRounded.exp x) id round start
          (List.toChunks B (first :: rest))) - O / S| ≤
        (1 + u) * ((V * (ao + al + ρl) + ρo) / (1 - al - ρl)) + u * V + η := by
  intro N T M S O al ao ρl ρo hrangel hrangeo hsmall hquot
  have hu : 0 ≤ u := (Model.unitRoundoffAt_pos format).le
  have hη : 0 ≤ η := (Model.underflowErrorAt_pos format).le
  have hkeysne : (first :: rest) ≠ [] := List.cons_ne_nil _ _
  have hMK : |M| ≤ K := by
    obtain ⟨e, he, hEq⟩ := hattained
    simp only [M]; rw [← hEq]; exact (hkeys e he).2.2.1
  have main := onlineAverage_error_of_le hformat round huL hηL hround start
    (List.toChunks B (first :: rest)) hK hV hstart hMK
    (List.length_le_of_mem_toChunks hB _)
    (by rw [List.flatten_toChunks]; exact hkeys)
    (by rw [List.flatten_toChunks]; exact hmax)
  simp only [List.flatten_toChunks, List.length_toChunks hB] at main
  have hshift : ∀ e ∈ first :: rest, ExecFloat.Binary.toReal e.1 ≤ M ∧
      u * (M - ExecFloat.Binary.toReal e.1) ≤ 1 := by
    intro e he
    refine ⟨hmax e he, ?_⟩
    have := (hkeys e he).2.2.1
    have hdiff : M - ExecFloat.Binary.toReal e.1 ≤ 2 * K := by
      have := abs_sub M (ExecFloat.Binary.toReal e.1)
      linarith [le_abs_self (M - ExecFloat.Binary.toReal e.1)]
    nlinarith
  have hT : 0 ≤ (B : ℝ) + T + 1 := by positivity
  have hρ (V' C r : ℝ) (hV' : 0 ≤ V') (hr : 0 ≤ r) :
      0 ≤ (T : ℝ) * residualGrowth u η ^ T * blockResidual u η N B V' C r := by
    have := one_le_residualGrowth hu hη
    have : 0 ≤ residualGrowth u η := by linarith
    unfold blockResidual; positivity
  obtain ⟨hden, hquot', hfinal⟩ := budget_closed_form (first :: rest)
    (fun e => ExecFloat.Binary.toReal e.1) (fun e => ExecFloat.Binary.toReal e.2) (under := η)
    (cl := u * (B + T + 1)) (co := u * (B + T + 2) + uL) (ρl := ρl) (ρo := ρo)
    (List.cons_ne_nil _ _) hu (mul_nonneg hu hT) (by positivity) (hρ _ _ _ zero_le_one hη)
    (hρ _ _ _ hV (by unfold productResidual; positivity)) hV hattained hshift
    (fun e he => (hkeys e he).2.2.2) hsmall
  obtain ⟨hfinite, hbound⟩ := main hrangel hrangeo hden (hquot'.trans hquot)
  exact ⟨hfinite, hbound.trans hfinal⟩

/-- Round a value to the format `low` and read it back in the working format: the value a weight
takes when it is stored in `low` for the matrix product, as FlashAttention stores its
probabilities in `bfloat16` or `binary16`. -/
def roundTo (low : FloatFormat) (x : Value) : Value :=
  Configured.Family.ofModel
    (Model.cast low format (Model.cast format low (ExecFloat.Binary.toModel x)))

/-- Storing a finite value in `[0, 1]` in an IEEE format `low` whose grid the working format
contains is finite and decodes to `Model.roundAt low` of the value. -/
theorem toReal_roundTo (hformat : format.isIEEE = true) {low : FloatFormat}
    (hlow : low.isIEEE = true) (hfrac : low.fracWidth ≤ format.fracWidth)
    (hmin : format.minSubnormalExponent ≤ low.minSubnormalExponent) {x : Value}
    (hx : ExecFloat.Binary.isFinite x = true) (h01 : ExecFloat.Binary.toReal x ∈ Set.Icc 0 1) :
    ExecFloat.Binary.isFinite (roundTo low x) = true ∧
      ExecFloat.Binary.toReal (roundTo low x) = Model.roundAt low (ExecFloat.Binary.toReal x) := by
  set m := ExecFloat.Binary.toModel x
  have hm : Model.isFinite m = true := hx
  have hreal : ExecFloat.Binary.toReal x = Model.toReal m := rfl
  rw [hreal] at h01 ⊢
  have habs : |Model.toReal m| ≤ 1 := by rw [abs_of_nonneg h01.1]; exact h01.2
  have hy : Model.isFinite (Model.cast format low m) = true :=
    Model.isFinite_cast_of_abs_toReal_le_posMaxFinite m hlow hm
      (habs.trans (Model.one_le_toReal_posMaxFinite low))
  have hyv := Model.cast_eq_roundAt hlow m hy
  have hy01 := Model.roundAt_mem_Icc_zero_one low h01
  rw [← hyv] at hy01
  have hz : Model.isFinite (Model.cast low format (Model.cast format low m)) = true :=
    Model.isFinite_cast_of_abs_toReal_le_posMaxFinite _ hformat hy
      ((by rw [abs_of_nonneg hy01.1]; exact hy01.2 : _ ≤ (1 : ℝ)).trans
        (Model.one_le_toReal_posMaxFinite format))
  have hzv := Model.cast_exact_of_gridExtension _ hformat hfrac hmin hz
  constructor
  · change Model.isFinite (Configured.Family.toModel (Configured.Family.ofModel _)) = true
    rw [Configured.Family.toModel_ofModel]; exact hz
  · change Model.toReal (Configured.Family.toModel (Configured.Family.ofModel _)) = _
    rw [Configured.Family.toModel_ofModel, hzv, hyv]

/-- Storing a weight in an IEEE format `low` whose grid the working format contains meets the
weight-rounding contract of `onlineAverage_error` with `uL` and `ηL` the unit roundoff and
underflow allowance of `low`. -/
theorem roundTo_contract (hformat : format.isIEEE = true) {low : FloatFormat}
    (hlow : low.isIEEE = true) (hfrac : low.fracWidth ≤ format.fracWidth)
    (hmin : format.minSubnormalExponent ≤ low.minSubnormalExponent) (x : Value)
    (hx : ExecFloat.Binary.isFinite x = true) (h01 : ExecFloat.Binary.toReal x ∈ Set.Icc 0 1) :
    ExecFloat.Binary.isFinite (roundTo low x) = true ∧
      ExecFloat.Binary.toReal (roundTo low x) ∈ Set.Icc 0 1 ∧
      |ExecFloat.Binary.toReal (roundTo low x) - ExecFloat.Binary.toReal x| ≤
        Model.unitRoundoffAt low * |ExecFloat.Binary.toReal x| + Model.underflowErrorAt low := by
  obtain ⟨hfin, hval⟩ := toReal_roundTo hformat hlow hfrac hmin hx h01
  refine ⟨hfin, ?_, ?_⟩
  · rw [hval]; exact Model.roundAt_mem_Icc_zero_one low h01
  · rw [hval]; exact Model.abs_roundAt_sub_le_unitRoundoffAt low _

/-- The decoded weight `f (exp (s - m))` the computation stores for a key with score `s` against
the running maximum `m`. -/
noncomputable def storedWeight (f : Value → Value) (m : Value) (e : Value × Value) : ℝ :=
  ExecFloat.Binary.toReal (f (ExecFloat.Binary.CorrectlyRounded.exp (e.1 - m)))

/-- The exact sums of the weights the computation stores: the running maximum and decoded scale
of `onlineStep`, the sums `L` of the weights `roundSum` and `W` of the weights `round`,
and the sum `O` of the weights `round` times the values, combined in exact arithmetic. -/
noncomputable def storedStep (roundSum round : Value → Value) (state : Value × ℝ × ℝ × ℝ)
    (block : List (Value × Value)) : Value × ℝ × ℝ × ℝ :=
  let m := block.foldl (fun m e => Context.maximum m e.1) state.1
  let scale := ExecFloat.Binary.toReal (ExecFloat.Binary.CorrectlyRounded.exp (state.1 - m))
  (m, scale * state.2.1 + (block.map (storedWeight roundSum m)).sum,
    scale * state.2.2.1 + (block.map (storedWeight round m)).sum,
    scale * state.2.2.2 +
      (block.map fun e => storedWeight round m e * ExecFloat.Binary.toReal e.2).sum)

/-- If `x` and `r` are within `Xa (exp c - 1) + ρ` and `Ra (exp cb - 1) + ρb` of exact sums `X`
and `R` with `|X| ≤ Xa` and `|R| ≤ Ra`, the rounded update `scale * x + r` is finite and within
`(scale Xa + Ra) (exp C - 1)` plus a residual of `scale * X + R`. Here `C` is any exponent at least
`c + 2 u` and `cb + u`, so the earlier allowance gains `2 u` in its exponent and the block's `u`. -/
theorem uniform_combine_error (hformat : format.isIEEE = true) {scale x r : Value}
    {X Xa R Ra c cb C ρ ρb : ℝ}
    (hsf : ExecFloat.Binary.isFinite scale = true)
    (hs01 : ExecFloat.Binary.toReal scale ∈ Set.Icc 0 1)
    (hx : ExecFloat.Binary.isFinite x = true)
    (hxe : |ExecFloat.Binary.toReal x - X| ≤ Xa * (Real.exp c - 1) + ρ) (hX : |X| ≤ Xa)
    (hr : ExecFloat.Binary.isFinite r = true)
    (hre : |ExecFloat.Binary.toReal r - R| ≤ Ra * (Real.exp cb - 1) + ρb) (hR : |R| ≤ Ra)
    (hc : 0 ≤ c) (hcb : 0 ≤ cb) (hC : c + 2 * u ≤ C) (hCb : cb + u ≤ C) (hρ : 0 ≤ ρ)
    (hρb : 0 ≤ ρb)
    (hrange : (ExecFloat.Binary.toReal scale * Xa + Ra) * Real.exp C +
      (1 + u) * ((1 + u) * ρ + η + ρb) ≤ maxFinite) :
    ExecFloat.Binary.isFinite (scale * x + r) = true ∧
      |ExecFloat.Binary.toReal (scale * x + r) - (ExecFloat.Binary.toReal scale * X + R)| ≤
        (ExecFloat.Binary.toReal scale * Xa + Ra) * (Real.exp C - 1) +
          (1 + u) * ((1 + u) * ρ + η + ρb) := by
  have hu : 0 ≤ u := (Model.unitRoundoffAt_pos format).le
  have hη : 0 ≤ η := (Model.underflowErrorAt_pos format).le
  set α := ExecFloat.Binary.toReal scale with hα
  have hXa : 0 ≤ Xa := (abs_nonneg _).trans hX
  have hRa : 0 ≤ Ra := (abs_nonneg _).trans hR
  have hexpu : 1 + u ≤ Real.exp u := by linarith [Real.add_one_le_exp u]
  have hxmax := ExecFloat.Binary.abs_toReal_le_posMaxFinite_of_isFinite hx
  have hmulFinite : ExecFloat.Binary.isFinite (scale * x) = true :=
    ExecFloat.Binary.isFinite_mul_of_mixedError hsf hx (.refl _) (by
      simp only [one_mul, add_zero, abs_mul]
      calc |α| * |ExecFloat.Binary.toReal x| ≤ 1 * |ExecFloat.Binary.toReal x| := by
            gcongr; rw [abs_of_nonneg hs01.1]; exact hs01.2
        _ ≤ _ := by simpa using hxmax)
  have hmul := (ExecFloat.Binary.mul_rounding hmulFinite).bound
  -- the product against the exact scaled sum
  have hz : |α * ExecFloat.Binary.toReal x - α * X| ≤ α * (Xa * (Real.exp c - 1) + ρ) := by
    rw [← mul_sub, abs_mul, abs_of_nonneg hs01.1]; exact mul_le_mul_of_nonneg_left hxe hs01.1
  have hαX : |α * X| ≤ α * Xa := by
    rw [abs_mul, abs_of_nonneg hs01.1]; exact mul_le_mul_of_nonneg_left hX hs01.1
  have hec : 0 ≤ Real.exp c - 1 := by linarith [Real.one_le_exp hc]
  have h2 : |α * ExecFloat.Binary.toReal x| ≤ α * Xa * Real.exp c + α * ρ := by
    have h := abs_add_le (α * ExecFloat.Binary.toReal x - α * X) (α * X)
    simp only [sub_add_cancel] at h
    have : α * (Xa * (Real.exp c - 1) + ρ) + α * Xa = α * Xa * Real.exp c + α * ρ := by ring
    linarith
  have hp : |ExecFloat.Binary.toReal (scale * x) - α * X| ≤
      α * Xa * (Real.exp (c + u) - 1) + (1 + u) * ρ + η := by
    have h3 : (1 + u) * Real.exp c ≤ Real.exp (c + u) := by
      rw [Real.exp_add, mul_comm]
      exact mul_le_mul_of_nonneg_left hexpu (Real.exp_pos c).le
    have h4 : 0 ≤ α * Xa := mul_nonneg hs01.1 hXa
    have h5 : α * ρ ≤ ρ := mul_le_of_le_one_left hρ hs01.2
    have h6 : u * |α * ExecFloat.Binary.toReal x| ≤ u * (α * Xa * Real.exp c + α * ρ) :=
      mul_le_mul_of_nonneg_left h2 hu
    have h7 : α * Xa * ((1 + u) * Real.exp c) ≤ α * Xa * Real.exp (c + u) :=
      mul_le_mul_of_nonneg_left h3 h4
    have h8 : u * (α * ρ) ≤ u * ρ := mul_le_mul_of_nonneg_left h5 hu
    calc |ExecFloat.Binary.toReal (scale * x) - α * X|
        ≤ |ExecFloat.Binary.toReal (scale * x) - α * ExecFloat.Binary.toReal x| +
          |α * ExecFloat.Binary.toReal x - α * X| := abs_sub_le _ _ _
      _ ≤ u * |α * ExecFloat.Binary.toReal x| + η + α * (Xa * (Real.exp c - 1) + ρ) :=
          add_le_add hmul hz
      _ ≤ _ := by nlinarith
  -- the sum before its rounding
  have hecb : 0 ≤ Real.exp cb - 1 := by linarith [Real.one_le_exp hcb]
  have hCu : Real.exp (c + u) ≤ Real.exp (C - u) := Real.exp_le_exp.mpr (by linarith)
  have hCb' : Real.exp cb ≤ Real.exp (C - u) := Real.exp_le_exp.mpr (by linarith)
  have hρm : 0 ≤ (1 + u) * ρ + η + ρb := by positivity
  have hsum : |(ExecFloat.Binary.toReal (scale * x) + ExecFloat.Binary.toReal r) - (α * X + R)| ≤
      (α * Xa + Ra) * (Real.exp (C - u) - 1) + ((1 + u) * ρ + η + ρb) := by
    have h1 := abs_add_le (ExecFloat.Binary.toReal (scale * x) - α * X)
      (ExecFloat.Binary.toReal r - R)
    have h2 : α * Xa * (Real.exp (c + u) - 1) ≤ α * Xa * (Real.exp (C - u) - 1) :=
      mul_le_mul_of_nonneg_left (by linarith) (mul_nonneg hs01.1 hXa)
    have h3 : Ra * (Real.exp cb - 1) ≤ Ra * (Real.exp (C - u) - 1) :=
      mul_le_mul_of_nonneg_left (by linarith) hRa
    have e1 : (ExecFloat.Binary.toReal (scale * x) + ExecFloat.Binary.toReal r) - (α * X + R) =
        (ExecFloat.Binary.toReal (scale * x) - α * X) + (ExecFloat.Binary.toReal r - R) := by ring
    have e2 : (α * Xa + Ra) * (Real.exp (C - u) - 1) =
        α * Xa * (Real.exp (C - u) - 1) + Ra * (Real.exp (C - u) - 1) := by ring
    rw [e1, e2]
    linarith
  have hZ : |α * X + R| ≤ α * Xa + Ra := (abs_add_le _ _).trans (add_le_add hαX hR)
  have hsumabs : |ExecFloat.Binary.toReal (scale * x) + ExecFloat.Binary.toReal r| ≤
      (α * Xa + Ra) * Real.exp (C - u) + ((1 + u) * ρ + η + ρb) := by
    have h := abs_add_le (ExecFloat.Binary.toReal (scale * x) + ExecFloat.Binary.toReal r -
      (α * X + R)) (α * X + R)
    simp only [sub_add_cancel] at h
    have : (α * Xa + Ra) * (Real.exp (C - u) - 1) + (α * Xa + Ra) =
        (α * Xa + Ra) * Real.exp (C - u) := by ring
    linarith
  have hαXR : 0 ≤ α * Xa + Ra := add_nonneg (mul_nonneg hs01.1 hXa) hRa
  have hrange' : (α * Xa + Ra) * Real.exp (C - u) + ((1 + u) * ρ + η + ρb) ≤ maxFinite := by
    refine le_trans ?_ hrange
    have h1 : (α * Xa + Ra) * Real.exp (C - u) ≤ (α * Xa + Ra) * Real.exp C :=
      mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr (by linarith)) hαXR
    have h2 : (1 + u) * ρ + η + ρb ≤ (1 + u) * ((1 + u) * ρ + η + ρb) :=
      le_mul_of_one_le_left hρm (by linarith)
    linarith
  have haddFinite : ExecFloat.Binary.isFinite (scale * x + r) = true :=
    ExecFloat.Binary.isFinite_add_of_mixedError hmulFinite hr (.refl _)
      (by simpa using hsumabs.trans hrange')
  refine ⟨haddFinite, ?_⟩
  have hadd := (ExecFloat.Binary.add_rounding haddFinite).bound
  have hround : |ExecFloat.Binary.toReal (scale * x + r) -
      (ExecFloat.Binary.toReal (scale * x) + ExecFloat.Binary.toReal r)| ≤
      u * ((α * Xa + Ra) * Real.exp (C - u) + ((1 + u) * ρ + η + ρb)) := by
    have := mul_le_mul_of_nonneg_left hsumabs hu
    simp only [add_zero] at hadd
    linarith
  have hfin : (1 + u) * Real.exp (C - u) ≤ Real.exp C := by
    calc (1 + u) * Real.exp (C - u) ≤ Real.exp u * Real.exp (C - u) :=
          mul_le_mul_of_nonneg_right hexpu (Real.exp_pos _).le
      _ = Real.exp C := by rw [← Real.exp_add]; ring_nf
  have hfin' : (α * Xa + Ra) * ((1 + u) * Real.exp (C - u)) ≤ (α * Xa + Ra) * Real.exp C :=
    mul_le_mul_of_nonneg_left hfin hαXR
  have := abs_sub_le (ExecFloat.Binary.toReal (scale * x + r))
    (ExecFloat.Binary.toReal (scale * x) + ExecFloat.Binary.toReal r) (α * X + R)
  have e3 : u * ((α * Xa + Ra) * Real.exp (C - u) + ((1 + u) * ρ + η + ρb)) +
      ((α * Xa + Ra) * (Real.exp (C - u) - 1) + ((1 + u) * ρ + η + ρb)) =
      (α * Xa + Ra) * ((1 + u) * Real.exp (C - u)) - (α * Xa + Ra) +
        (1 + u) * ((1 + u) * ρ + η + ρb) := by ring
  have e4 : (α * Xa + Ra) * (Real.exp C - 1) = (α * Xa + Ra) * Real.exp C - (α * Xa + Ra) := by
    ring
  linarith

/-- Facts about one block's stored weights: the running maximum stays finite and bounded, its
decoded scale lies in `[0, 1]`, and so do the decoded stored weights of both sums. -/
theorem storedStep_weights (hformat : format.isIEEE = true) (roundSum round : Value → Value)
    (hroundSum : ∀ x : Value, ExecFloat.Binary.isFinite x = true →
      ExecFloat.Binary.toReal x ∈ Set.Icc 0 1 → ExecFloat.Binary.isFinite (roundSum x) = true ∧
        ExecFloat.Binary.toReal (roundSum x) ∈ Set.Icc 0 1)
    (hround : ∀ x : Value, ExecFloat.Binary.isFinite x = true →
      ExecFloat.Binary.toReal x ∈ Set.Icc 0 1 → ExecFloat.Binary.isFinite (round x) = true ∧
        ExecFloat.Binary.toReal (round x) ∈ Set.Icc 0 1)
    {m : Value} {block : List (Value × Value)} {K : ℝ} (hK : 2 * K ≤ maxFinite)
    (hm : ExecFloat.Binary.isFinite m = true) (hmK : |ExecFloat.Binary.toReal m| ≤ K)
    (hblock : ∀ e ∈ block, ExecFloat.Binary.isFinite e.1 = true ∧
      |ExecFloat.Binary.toReal e.1| ≤ K) :
    let m' := block.foldl (fun m e => Context.maximum m e.1) m
    ExecFloat.Binary.isFinite m' = true ∧ |ExecFloat.Binary.toReal m'| ≤ K ∧
      ExecFloat.Binary.toReal m ≤ ExecFloat.Binary.toReal m' ∧
      (∀ e ∈ block, ExecFloat.Binary.toReal e.1 ≤ ExecFloat.Binary.toReal m') ∧
      ExecFloat.Binary.isFinite (ExecFloat.Binary.CorrectlyRounded.exp (m - m')) = true ∧
      ExecFloat.Binary.toReal (ExecFloat.Binary.CorrectlyRounded.exp (m - m')) ∈ Set.Icc 0 1 ∧
      ∀ e ∈ block,
        ExecFloat.Binary.isFinite (ExecFloat.Binary.CorrectlyRounded.exp (e.1 - m')) = true ∧
        ExecFloat.Binary.toReal (ExecFloat.Binary.CorrectlyRounded.exp (e.1 - m')) ∈
          Set.Icc 0 1 ∧
        ExecFloat.Binary.isFinite (roundSum (ExecFloat.Binary.CorrectlyRounded.exp (e.1 - m'))) =
          true ∧
        ExecFloat.Binary.toReal (roundSum (ExecFloat.Binary.CorrectlyRounded.exp (e.1 - m'))) ∈
          Set.Icc 0 1 ∧
        ExecFloat.Binary.isFinite (round (ExecFloat.Binary.CorrectlyRounded.exp (e.1 - m'))) =
          true ∧
        ExecFloat.Binary.toReal (round (ExecFloat.Binary.CorrectlyRounded.exp (e.1 - m'))) ∈
          Set.Icc 0 1 := by
  intro m'
  obtain ⟨hm'f, hm'val⟩ := toReal_foldl_maximum Prod.fst block m hm (fun e he => (hblock e he).1)
  have hmm' : ExecFloat.Binary.toReal m ≤ ExecFloat.Binary.toReal m' := by
    rw [hm'val]; exact List.le_foldl_max_init block _ _
  have hm'block : ∀ e ∈ block, ExecFloat.Binary.toReal e.1 ≤ ExecFloat.Binary.toReal m' := by
    intro e he; rw [hm'val]
    exact List.le_foldl_max_of_mem block (fun e : Value × Value => ExecFloat.Binary.toReal e.1) he
  have hm'K : |ExecFloat.Binary.toReal m'| ≤ K := by
    rw [hm'val]; exact abs_foldl_max_le _ block _ hmK (fun e he => (hblock e he).2)
  have hsub (x : Value) (hx : |ExecFloat.Binary.toReal x| ≤ K) :
      |ExecFloat.Binary.toReal x - ExecFloat.Binary.toReal m'| ≤ maxFinite := by
    have := abs_sub (ExecFloat.Binary.toReal x) (ExecFloat.Binary.toReal m')
    linarith
  obtain ⟨hsf, hs01, -⟩ := exp_sub_error hformat hm hm'f hmm' (hsub m hmK)
  refine ⟨hm'f, hm'K, hmm', hm'block, hsf, hs01, fun e he => ?_⟩
  obtain ⟨hpf, hp01, -⟩ := exp_sub_error hformat (hblock e he).1 hm'f (hm'block e he)
    (hsub e.1 (hblock e he).2)
  exact ⟨hpf, hp01, (hroundSum _ hpf hp01).1, (hroundSum _ hpf hp01).2, (hround _ hpf hp01).1,
    (hround _ hpf hp01).2⟩

/-- For finite `x` and `m` with equal real values, the rounded exponential of `x - m` is finite
and decodes to exactly one. -/
theorem toReal_exp_sub_eq_one (hformat : format.isIEEE = true) {x m : Value}
    (hx : ExecFloat.Binary.isFinite x = true) (hm : ExecFloat.Binary.isFinite m = true)
    (hrange : |ExecFloat.Binary.toReal x - ExecFloat.Binary.toReal m| ≤ maxFinite)
    (heq : ExecFloat.Binary.toReal x = ExecFloat.Binary.toReal m) :
    ExecFloat.Binary.isFinite (ExecFloat.Binary.CorrectlyRounded.exp (x - m)) = true ∧
      ExecFloat.Binary.toReal (ExecFloat.Binary.CorrectlyRounded.exp (x - m)) = 1 := by
  have hdf : ExecFloat.Binary.isFinite (x - m) = true :=
    ExecFloat.Binary.isFinite_sub_of_mixedError hx hm (.refl _) (by simpa using hrange)
  have hd0 : ExecFloat.Binary.toReal (x - m) = 0 := by
    rw [ExecFloat.Binary.toReal_sub_eq_roundAt hdf, heq, sub_self, Model.roundAt_zero]
  have hthr : Real.exp (ExecFloat.Binary.toReal (x - m)) <
      Model.Transcendentals.Certified.overflowThreshold format := by
    rw [hd0, Real.exp_zero]
    exact Model.Transcendentals.CorrectlyRounded.one_lt_overflowThreshold hformat
  refine ⟨ExecFloat.Binary.CorrectlyRounded.isFinite_exp hdf hthr, ?_⟩
  rw [ExecFloat.Binary.CorrectlyRounded.toReal_exp_eq_roundAt hdf hthr, hd0, Real.exp_zero,
    Model.roundAt_one]

/-- If each stored weight `f (exp (s - m))` of a block is finite with decoded value in `[0, 1]`,
their rounded left-to-right sum is finite and within relative error `exp (u * block.length) - 1`
of the exact sum of the stored weights. -/
theorem foldl_storedWeight_error (hformat : format.isIEEE = true) (f : Value → Value)
    {m : Value} {block : List (Value × Value)}
    (hw : ∀ e ∈ block, ExecFloat.Binary.isFinite
      (f (ExecFloat.Binary.CorrectlyRounded.exp (e.1 - m))) = true ∧
        storedWeight f m e ∈ Set.Icc 0 1)
    (hrange : (block.length : ℝ) * Real.exp (u * block.length) ≤ maxFinite) :
    ExecFloat.Binary.isFinite (block.foldl
      (fun acc e => acc + f (ExecFloat.Binary.CorrectlyRounded.exp (e.1 - m))) 0) = true ∧
      |ExecFloat.Binary.toReal (block.foldl
        (fun acc e => acc + f (ExecFloat.Binary.CorrectlyRounded.exp (e.1 - m))) 0) -
        (block.map (storedWeight f m)).sum| ≤
        (block.map (storedWeight f m)).sum * (Real.exp (u * block.length) - 1) := by
  have h := foldl_add_error hformat (fun e => f (ExecFloat.Binary.CorrectlyRounded.exp (e.1 - m)))
    (storedWeight f m) (fun _ => 0) (fun _ => 1) 0 1 0 le_rfl block [] 0 0 0 le_rfl le_rfl
    ExecFloat.Binary.isFinite_zero (by simp [TermwiseError])
    (fun e he => by
      have h := (hw e (by simpa using he)).2
      exact ⟨h.1, by simpa using h.2, by simp⟩)
    (fun e he => ⟨(hw e he).1, by simp [TermwiseError, storedWeight]⟩)
    (by simpa using hrange)
  refine ⟨h.1, ?_⟩
  have h' := h.2.abs_sub_le (c := u * block.length) (fun _ _ => by simp)
  simpa using h'

/-- With stored weights finite in `[0, 1]` and finite values bounded by `V`, the rounded
left-to-right sum of a block's products `f (exp (s - m)) * v` is finite and within
`V W (exp (u + u * block.length) - 1) + (1 + u) ^ block.length * (block.length * η)` of the exact
sum, where `W` is the exact sum of the stored weights. -/
theorem foldl_storedWeight_mul_error (hformat : format.isIEEE = true) (f : Value → Value)
    {m : Value} {block : List (Value × Value)} {V : ℝ}
    (hw : ∀ e ∈ block, ExecFloat.Binary.isFinite
      (f (ExecFloat.Binary.CorrectlyRounded.exp (e.1 - m))) = true ∧
        storedWeight f m e ∈ Set.Icc 0 1 ∧ ExecFloat.Binary.isFinite e.2 = true ∧
        |ExecFloat.Binary.toReal e.2| ≤ V)
    (hrange : (block.length : ℝ) * V * Real.exp (u + u * block.length) +
      (1 + u) ^ block.length * (block.length * η) ≤ maxFinite) :
    ExecFloat.Binary.isFinite (block.foldl
      (fun acc e => acc + f (ExecFloat.Binary.CorrectlyRounded.exp (e.1 - m)) * e.2) 0) = true ∧
      |ExecFloat.Binary.toReal (block.foldl
        (fun acc e => acc + f (ExecFloat.Binary.CorrectlyRounded.exp (e.1 - m)) * e.2) 0) -
        (block.map fun e => storedWeight f m e * ExecFloat.Binary.toReal e.2).sum| ≤
        V * (block.map (storedWeight f m)).sum * (Real.exp (u + u * block.length) - 1) +
          (1 + u) ^ block.length * (block.length * η) := by
  have hu : 0 ≤ u := (Model.unitRoundoffAt_pos format).le
  have hη : 0 ≤ η := (Model.underflowErrorAt_pos format).le
  have hterms : ∀ e ∈ block, ExecFloat.Binary.isFinite
      (f (ExecFloat.Binary.CorrectlyRounded.exp (e.1 - m)) * e.2) = true ∧
      TermwiseError (ExecFloat.Binary.toReal
        (f (ExecFloat.Binary.CorrectlyRounded.exp (e.1 - m)) * e.2)) [e]
        (storedWeight f m) (fun e => ExecFloat.Binary.toReal e.2)
        (fun x => (fun _ => (0 : ℝ)) x + u) η := by
    intro e he
    have hq : ExecFloat.Binary.toReal (f (ExecFloat.Binary.CorrectlyRounded.exp (e.1 - m))) ∈
        Set.Icc 0 1 := (hw e he).2.1
    have hqf := (hw e he).1
    have hvf := (hw e he).2.2.1
    have hmulf : ExecFloat.Binary.isFinite
        (f (ExecFloat.Binary.CorrectlyRounded.exp (e.1 - m)) * e.2) = true :=
      ExecFloat.Binary.isFinite_mul_of_mixedError hqf hvf (.refl _) (by
        simp only [one_mul, add_zero, abs_mul]
        calc _ ≤ 1 * |ExecFloat.Binary.toReal e.2| := by
              gcongr; rw [abs_of_nonneg hq.1]; exact hq.2
          _ ≤ _ := by simpa using ExecFloat.Binary.abs_toReal_le_posMaxFinite_of_isFinite hvf)
    refine ⟨hmulf, ?_⟩
    have hb := (ExecFloat.Binary.mul_rounding hmulf).bound
    have hexp : u ≤ Real.exp u - 1 := by linarith [Real.add_one_le_exp u]
    have h0 : 0 ≤ ExecFloat.Binary.toReal
        (f (ExecFloat.Binary.CorrectlyRounded.exp (e.1 - m))) *
        |ExecFloat.Binary.toReal e.2| := mul_nonneg hq.1 (abs_nonneg _)
    have h1 := mul_le_mul_of_nonneg_left hexp h0
    rw [abs_mul, abs_of_nonneg hq.1] at hb
    simp only [TermwiseError, storedWeight, List.map_cons, List.map_nil, List.sum_cons,
      List.sum_nil, add_zero, zero_add]
    linarith
  have h := foldl_add_error hformat
    (fun e => f (ExecFloat.Binary.CorrectlyRounded.exp (e.1 - m)) * e.2)
    (storedWeight f m) (fun _ => 0) (fun e => ExecFloat.Binary.toReal e.2) u V η hη
    block [] 0 u 0 le_rfl le_rfl ExecFloat.Binary.isFinite_zero (by simp [TermwiseError])
    (fun e he => by
      have h := hw e (by simpa using he)
      exact ⟨h.2.1.1, by simpa using h.2.1.2, h.2.2.2⟩)
    hterms (by simpa using hrange)
  refine ⟨h.1, ?_⟩
  have h' := h.2.abs_sub_le (c := u + u * block.length) (fun _ _ => by simp)
  simp only [List.nil_append, zero_add] at h'
  have hWV := sum_map_mul_abs_le block (storedWeight f m) (fun e => ExecFloat.Binary.toReal e.2)
    (fun e he => (hw e he).2.1.1) (fun e he => (hw e he).2.2.2)
  have hexp : 0 ≤ Real.exp (u + u * block.length) - 1 := by
    have := Real.add_one_le_exp (u + u * block.length)
    have : 0 ≤ u * (block.length : ℝ) := mul_nonneg hu (Nat.cast_nonneg _)
    linarith
  have := mul_le_mul_of_nonneg_right hWV hexp
  linarith

/-- The invariant of online softmax after the keys `P` against the exact sums
`shadow = (m, L, W, O)` of its stored weights. The state and `shadow` share a finite running
maximum `m` bounded by `K` and at least every score seen, `L` and `W` lie in `[0, P.length]` with
`|W - L| ≤ δ L + P.length ηs`, and `O` lies in `[lo W, hi W]` with `|O| ≤ V W`. The computed
normalizer and weighted sum are finite and within `L (exp cl - 1) + ρl` and
`V W (exp co - 1) + ρo` of `L` and `O`, and `L, W ≥ 1` once a score seen equals the maximum. -/
def StoredInvariant (P : List (Value × Value)) (state : Value × Value × Value)
    (shadow : Value × ℝ × ℝ × ℝ) (K V lo hi δ ηs cl co ρl ρo : ℝ) : Prop :=
  shadow.1 = state.1 ∧ ExecFloat.Binary.isFinite state.1 = true ∧
    ExecFloat.Binary.isFinite state.2.1 = true ∧ ExecFloat.Binary.isFinite state.2.2 = true ∧
    |ExecFloat.Binary.toReal state.1| ≤ K ∧
    (∀ e ∈ P, ExecFloat.Binary.toReal e.1 ≤ ExecFloat.Binary.toReal state.1) ∧
    0 ≤ shadow.2.1 ∧ shadow.2.1 ≤ P.length ∧ 0 ≤ shadow.2.2.1 ∧ shadow.2.2.1 ≤ P.length ∧
    |shadow.2.2.2| ≤ V * shadow.2.2.1 ∧ lo * shadow.2.2.1 ≤ shadow.2.2.2 ∧
    shadow.2.2.2 ≤ hi * shadow.2.2.1 ∧
    |shadow.2.2.1 - shadow.2.1| ≤ δ * shadow.2.1 + P.length * ηs ∧
    |ExecFloat.Binary.toReal state.2.1 - shadow.2.1| ≤ shadow.2.1 * (Real.exp cl - 1) + ρl ∧
    |ExecFloat.Binary.toReal state.2.2 - shadow.2.2.2| ≤
      V * shadow.2.2.1 * (Real.exp co - 1) + ρo ∧
    ((∃ e ∈ P, ExecFloat.Binary.toReal e.1 = ExecFloat.Binary.toReal state.1) →
      1 ≤ shadow.2.1 ∧ 1 ≤ shadow.2.2.1)

/-- One block step preserves `StoredInvariant`: the allowances of the earlier sums gain `2 u` in
their exponents, while the block's normalizer starts at exponent `u * block.length` and its
weighted sum at `u * (block.length + 1)`. -/
theorem storedStep_invariant (hformat : format.isIEEE = true) (roundSum round : Value → Value)
    {δ ηs : ℝ} (hηs : 0 ≤ ηs)
    (hroundSum : ∀ x : Value, ExecFloat.Binary.isFinite x = true →
      ExecFloat.Binary.toReal x ∈ Set.Icc 0 1 → ExecFloat.Binary.isFinite (roundSum x) = true ∧
        ExecFloat.Binary.toReal (roundSum x) ∈ Set.Icc 0 1)
    (hround : ∀ x : Value, ExecFloat.Binary.isFinite x = true →
      ExecFloat.Binary.toReal x ∈ Set.Icc 0 1 → ExecFloat.Binary.isFinite (round x) = true ∧
        ExecFloat.Binary.toReal (round x) ∈ Set.Icc 0 1)
    (hroundSum_one : ∀ x : Value, ExecFloat.Binary.isFinite x = true →
      ExecFloat.Binary.toReal x = 1 → ExecFloat.Binary.toReal (roundSum x) = 1)
    (hround_one : ∀ x : Value, ExecFloat.Binary.isFinite x = true →
      ExecFloat.Binary.toReal x = 1 → ExecFloat.Binary.toReal (round x) = 1)
    (hdiff : ∀ x : Value, ExecFloat.Binary.isFinite x = true →
      ExecFloat.Binary.toReal x ∈ Set.Icc 0 1 →
      |ExecFloat.Binary.toReal (round x) - ExecFloat.Binary.toReal (roundSum x)| ≤
        δ * ExecFloat.Binary.toReal (roundSum x) + ηs)
    {P block : List (Value × Value)} {state : Value × Value × Value}
    {shadow : Value × ℝ × ℝ × ℝ} {K V lo hi cl co ρl ρo Cl Co ρl' ρo' : ℝ}
    (hK : 2 * K ≤ maxFinite) (hV : 0 ≤ V)
    (hblock : ∀ e ∈ block, ExecFloat.Binary.isFinite e.1 = true ∧
      ExecFloat.Binary.isFinite e.2 = true ∧ |ExecFloat.Binary.toReal e.1| ≤ K ∧
      lo ≤ ExecFloat.Binary.toReal e.2 ∧ ExecFloat.Binary.toReal e.2 ≤ hi ∧
      |ExecFloat.Binary.toReal e.2| ≤ V)
    (hinv : StoredInvariant P state shadow K V lo hi δ ηs cl co ρl ρo) (hcl : 0 ≤ cl)
    (hco : 0 ≤ co) (hρl : 0 ≤ ρl) (hρo : 0 ≤ ρo) (hCl : cl + 2 * u ≤ Cl)
    (hCl' : u * block.length + u ≤ Cl) (hCo : co + 2 * u ≤ Co)
    (hCo' : u + u * block.length + u ≤ Co)
    (hρl' : (1 + u) * ((1 + u) * ρl + η) ≤ ρl')
    (hρo' : (1 + u) * ((1 + u) * ρo + η + (1 + u) ^ block.length * (block.length * η)) ≤ ρo')
    (hrangel : ((P.length + block.length : ℕ) : ℝ) * Real.exp Cl + ρl' ≤ maxFinite)
    (hrangeo : ((P.length + block.length : ℕ) : ℝ) * V * Real.exp Co + ρo' ≤ maxFinite) :
    StoredInvariant (P ++ block)
      (onlineStep (fun x => ExecFloat.Binary.CorrectlyRounded.exp x) roundSum round
        state block)
      (storedStep roundSum round shadow block) K V lo hi δ ηs Cl Co ρl' ρo' := by
  have hu : 0 ≤ u := (Model.unitRoundoffAt_pos format).le
  have hη : 0 ≤ η := (Model.underflowErrorAt_pos format).le
  obtain ⟨m, l, o⟩ := state
  obtain ⟨ms, L, W, O⟩ := shadow
  obtain ⟨hms, hmf, hlf, hof, hmK, hmP, hL0, hLP, hW0, hWP, hOV, hlo, hhi, hWL, hle, hoe, hatt⟩ :=
    hinv
  simp only at hms hmf hlf hof hmK hmP hL0 hLP hW0 hWP hOV hlo hhi hWL hle hoe hatt
  subst ms
  obtain ⟨hm'f, hm'K, hmm', hm'block, hsf, hs01, hw⟩ := storedStep_weights hformat roundSum round
    hroundSum hround hK hmf hmK (fun e he => ⟨(hblock e he).1, (hblock e he).2.2.1⟩)
  obtain ⟨-, hm'val⟩ := toReal_foldl_maximum Prod.fst block m hmf (fun e he => (hblock e he).1)
  set m' := block.foldl (fun m e => Context.maximum m e.1) m with hm'def
  have hcast : ((P.length + block.length : ℕ) : ℝ) = P.length + block.length := by push_cast; ring
  rw [hcast] at hrangel hrangeo
  have hn : (0 : ℝ) ≤ P.length := Nat.cast_nonneg _
  have hb : (0 : ℝ) ≤ block.length := Nat.cast_nonneg _
  have hub : 0 ≤ u * (block.length : ℝ) := mul_nonneg hu hb
  have hρl'0 : 0 ≤ ρl' := le_trans (by positivity) hρl'
  have hρo'b : (1 + u) ^ block.length * (block.length * η) ≤ ρo' := by
    refine le_trans ?_ hρo'
    have h0 : 0 ≤ (1 + u) * ρo + η := by positivity
    have h1 : 0 ≤ (1 + u) ^ block.length * (block.length * η) := by positivity
    have h2 := le_mul_of_one_le_left (add_nonneg h0 h1) (by linarith : (1 : ℝ) ≤ 1 + u)
    linarith
  -- the block sums
  have hlb := foldl_storedWeight_error hformat roundSum (m := m') (block := block)
    (fun e he => ⟨(hw e he).2.2.1, (hw e he).2.2.2.1⟩) (by
      have h1 : Real.exp (u * block.length) ≤ Real.exp Cl := Real.exp_le_exp.mpr (by linarith)
      have h2 := mul_le_mul (by linarith : (block.length : ℝ) ≤ P.length + block.length) h1
        (Real.exp_pos _).le (by linarith)
      linarith)
  have hob := foldl_storedWeight_mul_error hformat round (m := m') (block := block) (V := V)
    (fun e he => ⟨(hw e he).2.2.2.2.1, (hw e he).2.2.2.2.2, (hblock e he).2.1,
      (hblock e he).2.2.2.2.2⟩) (by
      have h1 : Real.exp (u + u * block.length) ≤ Real.exp Co := Real.exp_le_exp.mpr (by linarith)
      have h2 := mul_le_mul (mul_le_mul_of_nonneg_right
        (by linarith : (block.length : ℝ) ≤ P.length + block.length) hV) h1
        (Real.exp_pos _).le (mul_nonneg (by linarith) hV)
      linarith)
  have ht01 : ∀ e ∈ block, storedWeight roundSum m' e ∈ Set.Icc (0 : ℝ) 1 :=
    fun e he => (hw e he).2.2.2.1
  have hq01 : ∀ e ∈ block, storedWeight round m' e ∈ Set.Icc (0 : ℝ) 1 :=
    fun e he => (hw e he).2.2.2.2.2
  have hObV := sum_map_mul_abs_le block (storedWeight round m')
    (fun e => ExecFloat.Binary.toReal e.2) (fun e he => (hq01 e he).1)
    (fun e he => (hblock e he).2.2.2.2.2)
  have hObI := sum_map_mul_bounds block (storedWeight round m')
    (fun e => ExecFloat.Binary.toReal e.2) (fun e he => (hq01 e he).1)
    (fun e he => ⟨(hblock e he).2.2.2.1, (hblock e he).2.2.2.2.1⟩)
  have hR := rescaled_sums_bounds hs01 hηs hn ⟨hL0, hLP⟩ ⟨hW0, hWP⟩ hOV hlo hhi hWL
    (sum_map_mem_Icc block _ ht01) (sum_map_mem_Icc block _ hq01)
    ((abs_sum_le _ _ (fun e he => (hq01 e he).1)).trans hObV) hObI.1 hObI.2
    (abs_sum_map_sub_le block _ _ (fun e he => hdiff _ (hw e he).1 (hw e he).2.1))
  -- combine with the rescaled sums
  have hl' := uniform_combine_error hformat (ρb := 0) hsf hs01 hlf hle (abs_of_nonneg hL0).le
    hlb.1 (hlb.2.trans_eq (add_zero _).symm)
    (abs_of_nonneg (sum_map_mem_Icc block _ ht01).1).le hcl hub hCl hCl' hρl le_rfl (by
      have h1 := mul_le_mul_of_nonneg_right hR.1.2 (Real.exp_pos Cl).le
      have h2 : (1 + u) * ((1 + u) * ρl + η + 0) ≤ ρl' := by simpa using hρl'
      linarith)
  have ho' := uniform_combine_error hformat hsf hs01 hof hoe hOV hob.1 hob.2
    ((abs_sum_le _ _ (fun e he => (hq01 e he).1)).trans hObV) hco (by positivity) hCo hCo' hρo
    (by positivity) (by
      have h1 := mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hR.2.1.2 hV)
        (Real.exp_pos Co).le
      linarith)
  -- a block key at the updated maximum has stored weights one
  have hone_block : ∀ e ∈ block, ExecFloat.Binary.toReal e.1 = ExecFloat.Binary.toReal m' →
      1 ≤ (block.map (storedWeight roundSum m')).sum ∧
        1 ≤ (block.map (storedWeight round m')).sum := by
    intro e he heq
    obtain ⟨hpf, hp1⟩ := toReal_exp_sub_eq_one hformat (hblock e he).1 hm'f (by
      have := abs_sub (ExecFloat.Binary.toReal e.1) (ExecFloat.Binary.toReal m')
      linarith [(hblock e he).2.2.1]) heq
    have ht1 : storedWeight roundSum m' e = 1 := hroundSum_one _ hpf hp1
    have hq1 : storedWeight round m' e = 1 := hround_one _ hpf hp1
    exact ⟨ht1 ▸ List.single_le_sum (List.forall_mem_map.mpr fun x hx => (ht01 x hx).1) _
        (List.mem_map_of_mem he),
      hq1 ▸ List.single_le_sum (List.forall_mem_map.mpr fun x hx => (hq01 x hx).1) _
        (List.mem_map_of_mem he)⟩
  have hlen : ((P ++ block).length : ℝ) = P.length + block.length := by simp
  refine ⟨rfl, hm'f, hl'.1, ho'.1, hm'K, fun e he => ?_, hR.1.1, hlen ▸ hR.1.2, hR.2.1.1,
    hlen ▸ hR.2.1.2, hR.2.2.1, hR.2.2.2.1, hR.2.2.2.2.1, hlen ▸ hR.2.2.2.2.2, ?_, ?_, ?_⟩
  · rcases List.mem_append.mp he with he | he
    · exact (hmP e he).trans hmm'
    · exact hm'block e he
  · have h2 : (1 + u) * ((1 + u) * ρl + η + 0) ≤ ρl' := by simpa using hρl'
    exact le_trans hl'.2 (add_le_add le_rfl h2)
  · have e1 : ∀ a b : ℝ, (a * (V * W) + V * b) * (Real.exp Co - 1) =
        V * (a * W + b) * (Real.exp Co - 1) := fun a b => by ring
    exact le_trans ho'.2 (by rw [e1]; exact add_le_add le_rfl hρo')
  · rintro ⟨e, he, heq⟩
    have heq' : ExecFloat.Binary.toReal e.1 = ExecFloat.Binary.toReal m' := heq
    have hL1 := sum_map_mem_Icc block _ ht01
    have hW1 := sum_map_mem_Icc block _ hq01
    have hαL := mul_nonneg hs01.1 hL0
    have hαW := mul_nonneg hs01.1 hW0
    have key : 1 ≤ ExecFloat.Binary.toReal (ExecFloat.Binary.CorrectlyRounded.exp (m - m')) * L +
          (block.map (storedWeight roundSum m')).sum ∧
        1 ≤ ExecFloat.Binary.toReal (ExecFloat.Binary.CorrectlyRounded.exp (m - m')) * W +
          (block.map (storedWeight round m')).sum := by
      by_cases hmeq : ExecFloat.Binary.toReal m = ExecFloat.Binary.toReal m'
      · have hα1 : ExecFloat.Binary.toReal (ExecFloat.Binary.CorrectlyRounded.exp (m - m')) = 1 :=
          (toReal_exp_sub_eq_one hformat hmf hm'f (by
            have := abs_sub (ExecFloat.Binary.toReal m) (ExecFloat.Binary.toReal m')
            linarith) hmeq).2
        rw [hα1]
        rcases List.mem_append.mp he with he | he
        · obtain ⟨h1, h2⟩ := hatt ⟨e, he, heq'.trans hmeq.symm⟩
          constructor <;> linarith
        · obtain ⟨h1, h2⟩ := hone_block e he heq'
          constructor <;> linarith
      · rcases List.foldl_max_eq_init_or_mem block (fun e : Value × Value =>
            ExecFloat.Binary.toReal e.1) (ExecFloat.Binary.toReal m) with h | ⟨e', he', h⟩
        · exact absurd (hm'val.trans h).symm hmeq
        · obtain ⟨h1, h2⟩ := hone_block e' he' (hm'val.trans h).symm
          constructor <;> linarith
    exact key

/-- Folding the block steps from `t` blocks to `T` preserves `StoredInvariant` with the closed-form
allowances `u * (B + 2 t)` and `u * (B + 1 + 2 t)` and residuals `t (1 + u) ^ (2 t) η` and
`t (1 + u) ^ (2 t) (η + (1 + u) ^ B * (B * η))` after `t` blocks of at most `B` keys. -/
theorem foldl_storedStep_invariant (hformat : format.isIEEE = true)
    (roundSum round : Value → Value) {δ ηs : ℝ} (hηs : 0 ≤ ηs)
    (hroundSum : ∀ x : Value, ExecFloat.Binary.isFinite x = true →
      ExecFloat.Binary.toReal x ∈ Set.Icc 0 1 → ExecFloat.Binary.isFinite (roundSum x) = true ∧
        ExecFloat.Binary.toReal (roundSum x) ∈ Set.Icc 0 1)
    (hround : ∀ x : Value, ExecFloat.Binary.isFinite x = true →
      ExecFloat.Binary.toReal x ∈ Set.Icc 0 1 → ExecFloat.Binary.isFinite (round x) = true ∧
        ExecFloat.Binary.toReal (round x) ∈ Set.Icc 0 1)
    (hroundSum_one : ∀ x : Value, ExecFloat.Binary.isFinite x = true →
      ExecFloat.Binary.toReal x = 1 → ExecFloat.Binary.toReal (roundSum x) = 1)
    (hround_one : ∀ x : Value, ExecFloat.Binary.isFinite x = true →
      ExecFloat.Binary.toReal x = 1 → ExecFloat.Binary.toReal (round x) = 1)
    (hdiff : ∀ x : Value, ExecFloat.Binary.isFinite x = true →
      ExecFloat.Binary.toReal x ∈ Set.Icc 0 1 →
      |ExecFloat.Binary.toReal (round x) - ExecFloat.Binary.toReal (roundSum x)| ≤
        δ * ExecFloat.Binary.toReal (roundSum x) + ηs)
    {K V lo hi : ℝ} {B N T : ℕ} (hK : 2 * K ≤ maxFinite) (hV : 0 ≤ V)
    (hrangel : (N : ℝ) * Real.exp (u * (B + 2 * T)) + T * (1 + u) ^ (2 * T) * η ≤ maxFinite)
    (hrangeo : (N : ℝ) * V * Real.exp (u * (B + 1 + 2 * T)) +
      T * (1 + u) ^ (2 * T) * (η + (1 + u) ^ B * (B * η)) ≤ maxFinite) :
    ∀ (blocks : List (List (Value × Value))) (P : List (Value × Value))
      (state : Value × Value × Value) (shadow : Value × ℝ × ℝ × ℝ) (t : ℕ),
      t + blocks.length = T → (P ++ blocks.flatten).length ≤ N →
      (∀ block ∈ blocks, block.length ≤ B) →
      (∀ e ∈ blocks.flatten, ExecFloat.Binary.isFinite e.1 = true ∧
        ExecFloat.Binary.isFinite e.2 = true ∧ |ExecFloat.Binary.toReal e.1| ≤ K ∧
        lo ≤ ExecFloat.Binary.toReal e.2 ∧ ExecFloat.Binary.toReal e.2 ≤ hi ∧
        |ExecFloat.Binary.toReal e.2| ≤ V) →
      StoredInvariant P state shadow K V lo hi δ ηs (u * (B + 2 * t)) (u * (B + 1 + 2 * t))
        (t * (1 + u) ^ (2 * t) * η) (t * (1 + u) ^ (2 * t) * (η + (1 + u) ^ B * (B * η))) →
      StoredInvariant (P ++ blocks.flatten)
        (blocks.foldl
          (onlineStep (fun x => ExecFloat.Binary.CorrectlyRounded.exp x) roundSum
            round) state)
        (blocks.foldl (storedStep roundSum round) shadow) K V lo hi δ ηs (u * (B + 2 * T))
        (u * (B + 1 + 2 * T)) (T * (1 + u) ^ (2 * T) * η)
        (T * (1 + u) ^ (2 * T) * (η + (1 + u) ^ B * (B * η)))
  | [], P, state, shadow, t, ht, _, _, _, hinv => by
      simp only [List.length_nil, add_zero] at ht
      subst ht
      simpa using hinv
  | block :: blocks, P, state, shadow, t, ht, hN, hB, hkeys, hinv => by
      have hu : 0 ≤ u := (Model.unitRoundoffAt_pos format).le
      have hη : 0 ≤ η := (Model.underflowErrorAt_pos format).le
      have htT : t + 1 ≤ T := by simp at ht; omega
      have hbB := hB block (List.mem_cons_self ..)
      have hPN : P.length + block.length ≤ N := by
        have := hN; simp only [List.length_append, List.flatten_cons] at this; omega
      have hT1 : ((t + 1 : ℕ) : ℝ) ≤ T := by exact_mod_cast htT
      have ht0 : (0 : ℝ) ≤ t := Nat.cast_nonneg t
      have hbBr : (block.length : ℝ) ≤ B := by exact_mod_cast hbB
      have hNl : ((P.length + block.length : ℕ) : ℝ) ≤ N := by exact_mod_cast hPN
      have hg1 : (1 : ℝ) ≤ 1 + u := by linarith
      have hpow : (1 + u) ^ (2 * (t + 1)) = (1 + u) ^ 2 * (1 + u) ^ (2 * t) := by ring
      have hpowT : (1 + u) ^ (2 * (t + 1)) ≤ (1 + u) ^ (2 * T) := pow_le_pow_right₀ hg1 (by omega)
      have h1g : 1 + u ≤ (1 + u) ^ 2 * (1 + u) ^ (2 * t) := by
        rw [← hpow]; exact le_self_pow₀ hg1 (by omega)
      have hQb : (1 + u) ^ block.length * (block.length * η) ≤ (1 + u) ^ B * (B * η) :=
        mul_le_mul (pow_le_pow_right₀ hg1 hbB) (mul_le_mul_of_nonneg_right hbBr hη)
          (by positivity) (by positivity)
      have hQ0 : 0 ≤ η + (1 + u) ^ B * (B * η) := by positivity
      have hub := mul_le_mul_of_nonneg_left hbBr hu
      have hut := mul_nonneg hu ht0
      have hstep := storedStep_invariant hformat roundSum round hηs hroundSum hround hroundSum_one
        hround_one hdiff (P := P) (block := block) (state := state) (shadow := shadow)
        (Cl := u * (B + 2 * ((t + 1 : ℕ) : ℝ))) (Co := u * (B + 1 + 2 * ((t + 1 : ℕ) : ℝ)))
        (ρl' := ((t + 1 : ℕ) : ℝ) * (1 + u) ^ (2 * (t + 1)) * η)
        (ρo' := ((t + 1 : ℕ) : ℝ) * (1 + u) ^ (2 * (t + 1)) * (η + (1 + u) ^ B * (B * η)))
        hK hV (fun e he => hkeys e (List.mem_flatten.mpr ⟨block, List.mem_cons_self .., he⟩))
        hinv (by positivity) (by positivity) (by positivity) (by positivity)
        (by push_cast; linarith) (by push_cast; linarith) (by push_cast; linarith)
        (by push_cast; linarith)
        (by
          rw [hpow]; push_cast
          have := mul_le_mul_of_nonneg_right h1g hη
          linarith)
        (by
          rw [hpow]; push_cast
          have h2 := mul_le_mul_of_nonneg_left (by linarith : η + (1 + u) ^ block.length *
            (block.length * η) ≤ η + (1 + u) ^ B * (B * η)) (by linarith : (0 : ℝ) ≤ 1 + u)
          have h3 := mul_le_mul_of_nonneg_right h1g hQ0
          linarith)
        (by
          refine le_trans ?_ hrangel
          have h1 : Real.exp (u * (B + 2 * ((t + 1 : ℕ) : ℝ))) ≤ Real.exp (u * (B + 2 * T)) :=
            Real.exp_le_exp.mpr (mul_le_mul_of_nonneg_left (by linarith) hu)
          have h2 := mul_le_mul hNl h1 (Real.exp_pos _).le (Nat.cast_nonneg N)
          have h3 := mul_le_mul_of_nonneg_right (mul_le_mul hT1 hpowT (by positivity)
            (Nat.cast_nonneg T)) hη
          linarith)
        (by
          refine le_trans ?_ hrangeo
          have h1 : Real.exp (u * (B + 1 + 2 * ((t + 1 : ℕ) : ℝ))) ≤
              Real.exp (u * (B + 1 + 2 * T)) :=
            Real.exp_le_exp.mpr (mul_le_mul_of_nonneg_left (by linarith) hu)
          have h2 := mul_le_mul (mul_le_mul_of_nonneg_right hNl hV) h1 (Real.exp_pos _).le
            (mul_nonneg (Nat.cast_nonneg N) hV)
          have h3 := mul_le_mul_of_nonneg_right (mul_le_mul hT1 hpowT (by positivity)
            (Nat.cast_nonneg T)) hQ0
          linarith)
      have := foldl_storedStep_invariant hformat roundSum round hηs hroundSum hround hroundSum_one
        hround_one hdiff hK hV hrangel hrangeo blocks (P ++ block) _ _ (t + 1)
        (by simp at ht ⊢; omega) (by simpa [List.flatten_cons, List.append_assoc] using hN)
        (fun x hx => hB x (List.mem_cons_of_mem _ hx))
        (fun e he => hkeys e (by simp only [List.flatten_cons, List.mem_append]; exact Or.inr he))
        hstep
      simpa [List.flatten_cons, List.append_assoc] using this

/-- At the start of the fold, a finite maximum `start` bounded by `K` with empty sums satisfies
`StoredInvariant` for all nonnegative residuals. -/
theorem storedInvariant_nil {start : Value} {K V lo hi δ ηs cl co ρl ρo : ℝ}
    (hstart : ExecFloat.Binary.isFinite start = true)
    (hstartK : |ExecFloat.Binary.toReal start| ≤ K) (hρl : 0 ≤ ρl) (hρo : 0 ≤ ρo) :
    StoredInvariant [] (start, 0, 0) (start, 0, 0, 0) K V lo hi δ ηs cl co ρl ρo :=
  ⟨rfl, hstart, ExecFloat.Binary.isFinite_zero, ExecFloat.Binary.isFinite_zero, hstartK,
    by simp, le_rfl, by simp, le_rfl, by simp, by simp, by simp, by simp, by simp,
    by simpa using hρl, by simpa using hρo, by simp⟩

/-- If a finite normalizer `l` and weighted sum `o` are within `L al + ρl` and `V W ao + ρo` of
exact sums `L, W ≥ 1` and `O ∈ [lo W, hi W]`, with `|W - L| ≤ ε L` and `V W ≤ V' L`, then `o / l`
is finite and within the closed-form bound of `β z`. Here `β z = (W / L) (O / W)`, with
`|β - 1| ≤ ε` and `z ∈ [lo, hi]`. -/
theorem div_stored_error (hformat : format.isIEEE = true) {l o : Value}
    {L W O V V' lo hi ε al ao ρl ρo : ℝ} (hlf : ExecFloat.Binary.isFinite l = true)
    (hof : ExecFloat.Binary.isFinite o = true) (hL1 : 1 ≤ L) (hW1 : 1 ≤ W)
    (hWL : |W - L| ≤ ε * L) (hOV : |O| ≤ V * W) (hVW : V * W ≤ V' * L) (hlo : lo * W ≤ O)
    (hhi : O ≤ hi * W) (hle : |ExecFloat.Binary.toReal l - L| ≤ L * al + ρl)
    (hoe : |ExecFloat.Binary.toReal o - O| ≤ V * W * ao + ρo) (hV' : 0 ≤ V') (hal : 0 ≤ al)
    (hao : 0 ≤ ao) (hρl : 0 ≤ ρl) (hρo : 0 ≤ ρo) (hsmall : al + ρl < 1)
    (hquot : (V' + V' * ao + ρo) / (1 - al - ρl) ≤ maxFinite) :
    ∃ β z : ℝ, z ∈ Set.Icc lo hi ∧ |β - 1| ≤ ε ∧ ExecFloat.Binary.isFinite (o / l) = true ∧
      |ExecFloat.Binary.toReal (o / l) - β * z| ≤
        (1 + u) * ((V' * (ao + al + ρl) + ρo) / (1 - al - ρl)) + u * V' + η := by
  have hu : 0 ≤ u := (Model.unitRoundoffAt_pos format).le
  have hLpos : 0 < L := by linarith
  have hWpos : 0 < W := by linarith
  have hL0 : 0 ≤ L := hLpos.le
  have hcf := closed_form_le (O := O) (V := V') (unit := u) (under := η) hL1
    (show L * al + ρl ≤ L * (al + ρl) by linarith [le_mul_of_one_le_left hρl hL1])
    (show V * W * ao + ρo ≤ L * (V' * ao + ρo) by
      linarith [mul_le_mul_of_nonneg_right hVW hao, le_mul_of_one_le_left hρo hL1])
    (hOV.trans hVW) hsmall hal hao hρl hρo hV' hu (by linarith [mul_nonneg hL0 hal])
  have hdiv := div_error hformat hlf hof hle hoe hcf.1 (hcf.2.1.trans hquot)
  refine ⟨W / L, O / W, ⟨(le_div_iff₀ hWpos).mpr hlo, (div_le_iff₀ hWpos).mpr hhi⟩, ?_, hdiv.1,
    ?_⟩
  · have e1 : W / L - 1 = (W - L) / L := by field_simp
    rw [e1, abs_div, abs_of_pos hLpos, div_le_iff₀ hLpos]
    exact hWL
  · have e1 : W / L * (O / W) = O / L := by field_simp
    rw [e1]
    exact hdiv.2.trans hcf.2.2

/-- Online softmax over `T` blocks of at most `B` of `N` keys, whose stored weights meet
`|round x - roundSum x| ≤ δ roundSum x + ηs` on `[0, 1]`, is finite and within the closed-form
bound of `β z`, where `z ∈ [lo, hi]` is the average of the values with the stored weights `round`
and `|β - 1| ≤ δ + N ηs`. -/
theorem onlineAverage_stored_error (hformat : format.isIEEE = true)
    (roundSum round : Value → Value) {δ ηs : ℝ} (hδ : 0 ≤ δ) (hηs : 0 ≤ ηs)
    (hroundSum : ∀ x : Value, ExecFloat.Binary.isFinite x = true →
      ExecFloat.Binary.toReal x ∈ Set.Icc 0 1 → ExecFloat.Binary.isFinite (roundSum x) = true ∧
        ExecFloat.Binary.toReal (roundSum x) ∈ Set.Icc 0 1)
    (hround : ∀ x : Value, ExecFloat.Binary.isFinite x = true →
      ExecFloat.Binary.toReal x ∈ Set.Icc 0 1 → ExecFloat.Binary.isFinite (round x) = true ∧
        ExecFloat.Binary.toReal (round x) ∈ Set.Icc 0 1)
    (hroundSum_one : ∀ x : Value, ExecFloat.Binary.isFinite x = true →
      ExecFloat.Binary.toReal x = 1 → ExecFloat.Binary.toReal (roundSum x) = 1)
    (hround_one : ∀ x : Value, ExecFloat.Binary.isFinite x = true →
      ExecFloat.Binary.toReal x = 1 → ExecFloat.Binary.toReal (round x) = 1)
    (hdiff : ∀ x : Value, ExecFloat.Binary.isFinite x = true →
      ExecFloat.Binary.toReal x ∈ Set.Icc 0 1 →
      |ExecFloat.Binary.toReal (round x) - ExecFloat.Binary.toReal (roundSum x)| ≤
        δ * ExecFloat.Binary.toReal (roundSum x) + ηs)
    (start : Value) (blocks : List (List (Value × Value))) {K V lo hi : ℝ} {B N T : ℕ}
    (hK : 2 * K ≤ maxFinite) (hV : 0 ≤ V) (hstart : ExecFloat.Binary.isFinite start = true)
    (hstartK : |ExecFloat.Binary.toReal start| ≤ K)
    (hstart_le : ∃ e ∈ blocks.flatten, ExecFloat.Binary.toReal start ≤ ExecFloat.Binary.toReal e.1)
    (hN : blocks.flatten.length ≤ N) (hB : ∀ block ∈ blocks, block.length ≤ B)
    (hT : blocks.length = T)
    (hkeys : ∀ e ∈ blocks.flatten, ExecFloat.Binary.isFinite e.1 = true ∧
      ExecFloat.Binary.isFinite e.2 = true ∧ |ExecFloat.Binary.toReal e.1| ≤ K ∧
      lo ≤ ExecFloat.Binary.toReal e.2 ∧ ExecFloat.Binary.toReal e.2 ≤ hi ∧
      |ExecFloat.Binary.toReal e.2| ≤ V) :
    let al := Real.exp (u * (B + 2 * T)) - 1
    let ao := Real.exp (u * (B + 1 + 2 * T)) - 1
    let ρl := T * (1 + u) ^ (2 * T) * η
    let ρo := T * (1 + u) ^ (2 * T) * (η + (1 + u) ^ B * (B * η))
    let V' := V * (1 + δ + N * ηs)
    (N : ℝ) * Real.exp (u * (B + 2 * T)) + ρl ≤ maxFinite →
    (N : ℝ) * V * Real.exp (u * (B + 1 + 2 * T)) + ρo ≤ maxFinite →
    al + ρl < 1 → (V' + V' * ao + ρo) / (1 - al - ρl) ≤ maxFinite →
    ∃ β z : ℝ, z ∈ Set.Icc lo hi ∧ |β - 1| ≤ δ + N * ηs ∧
      ExecFloat.Binary.isFinite
        (onlineAverage (fun x => ExecFloat.Binary.CorrectlyRounded.exp x) roundSum round
          start blocks) = true ∧
      |ExecFloat.Binary.toReal
          (onlineAverage (fun x => ExecFloat.Binary.CorrectlyRounded.exp x) roundSum round
            start blocks) - β * z| ≤
        (1 + u) * ((V' * (ao + al + ρl) + ρo) / (1 - al - ρl)) + u * V' + η := by
  intro al ao ρl ρo V' hrangel hrangeo hsmall hquot
  have hu : 0 ≤ u := (Model.unitRoundoffAt_pos format).le
  have hη : 0 ≤ η := (Model.underflowErrorAt_pos format).le
  have hinv := foldl_storedStep_invariant hformat roundSum round hηs hroundSum hround
    hroundSum_one hround_one hdiff (B := B) (N := N) (T := T) (lo := lo) (hi := hi) hK hV
    hrangel hrangeo blocks [] (start, 0, 0) (start, 0, 0, 0) 0 (by simp [hT]) (by simpa using hN)
    hB hkeys (storedInvariant_nil hstart hstartK (by positivity) (by positivity))
  generalize blocks.foldl (storedStep roundSum round) (start, 0, 0, 0) = sh at hinv
  obtain ⟨ms, L, W, O⟩ := sh
  obtain ⟨-, hmf, hlf, hof, -, hmP, hL0, -, hW0, -, hOV, hlo, hhi, hWL, hle, hoe, hatt⟩ := hinv
  simp only at hL0 hW0 hOV hlo hhi hWL hle hoe hatt
  -- the final maximum is a score
  have hfst := foldl_onlineStep_fst (fun x => ExecFloat.Binary.CorrectlyRounded.exp x)
    roundSum round blocks (start, 0, 0)
  have hval := (toReal_foldl_maximum Prod.fst blocks.flatten start hstart
    (fun e he => (hkeys e he).1)).2
  rw [← hfst] at hval
  obtain ⟨hL1, hW1⟩ := hatt (by
    rcases List.foldl_max_eq_init_or_mem blocks.flatten (fun e : Value × Value =>
        ExecFloat.Binary.toReal e.1) (ExecFloat.Binary.toReal start) with h | ⟨e, he, h⟩
    · obtain ⟨e, he, hse⟩ := hstart_le
      refine ⟨e, by simpa using he, le_antisymm (hmP e (by simpa using he)) ?_⟩
      rw [hval, h]; exact hse
    · exact ⟨e, by simpa using he, (hval.trans h).symm⟩)
  have hlen := mul_le_mul_of_nonneg_right
    (by simpa using (Nat.cast_le (α := ℝ)).mpr hN :
      (([] ++ blocks.flatten : List (Value × Value)).length : ℝ) ≤ N) hηs
  have hNη := le_mul_of_one_le_left (mul_nonneg (Nat.cast_nonneg N) hηs) hL1
  have hal : 0 ≤ al := by
    have := Real.one_le_exp (by positivity : 0 ≤ u * (B + 2 * T))
    show 0 ≤ Real.exp (u * (B + 2 * T)) - 1; linarith
  have hao : 0 ≤ ao := by
    have := Real.one_le_exp (by positivity : 0 ≤ u * (B + 1 + 2 * T))
    show 0 ≤ Real.exp (u * (B + 1 + 2 * T)) - 1; linarith
  have hWL' : |W - L| ≤ (δ + N * ηs) * L := by linarith
  have hWle : W ≤ L * (1 + δ + N * ηs) := by linarith [(abs_le.mp hWL').2]
  have hVW : V * W ≤ V' * L := by
    have := mul_le_mul_of_nonneg_left hWle hV
    show V * W ≤ V * (1 + δ + N * ηs) * L
    linarith
  exact div_stored_error hformat hlf hof hL1 hW1 hWL' hOV hVW
    hlo hhi hle hoe (by positivity) hal hao (by positivity) (by positivity) hsmall hquot

/-- Online softmax that sums the weights it stores, `roundSum = round`, is finite and within the
closed-form bound of a value `z ∈ [lo, hi]`, the average of the values with the stored weights. -/
theorem onlineAverage_stored_error_of_eq (hformat : format.isIEEE = true) (round : Value → Value)
    (hround : ∀ x : Value, ExecFloat.Binary.isFinite x = true →
      ExecFloat.Binary.toReal x ∈ Set.Icc 0 1 → ExecFloat.Binary.isFinite (round x) = true ∧
        ExecFloat.Binary.toReal (round x) ∈ Set.Icc 0 1)
    (hround_one : ∀ x : Value, ExecFloat.Binary.isFinite x = true →
      ExecFloat.Binary.toReal x = 1 → ExecFloat.Binary.toReal (round x) = 1)
    (start : Value) (blocks : List (List (Value × Value))) {K V lo hi : ℝ} {B N T : ℕ}
    (hK : 2 * K ≤ maxFinite) (hV : 0 ≤ V) (hstart : ExecFloat.Binary.isFinite start = true)
    (hstartK : |ExecFloat.Binary.toReal start| ≤ K)
    (hstart_le : ∃ e ∈ blocks.flatten, ExecFloat.Binary.toReal start ≤ ExecFloat.Binary.toReal e.1)
    (hN : blocks.flatten.length ≤ N) (hB : ∀ block ∈ blocks, block.length ≤ B)
    (hT : blocks.length = T)
    (hkeys : ∀ e ∈ blocks.flatten, ExecFloat.Binary.isFinite e.1 = true ∧
      ExecFloat.Binary.isFinite e.2 = true ∧ |ExecFloat.Binary.toReal e.1| ≤ K ∧
      lo ≤ ExecFloat.Binary.toReal e.2 ∧ ExecFloat.Binary.toReal e.2 ≤ hi ∧
      |ExecFloat.Binary.toReal e.2| ≤ V) :
    let al := Real.exp (u * (B + 2 * T)) - 1
    let ao := Real.exp (u * (B + 1 + 2 * T)) - 1
    let ρl := T * (1 + u) ^ (2 * T) * η
    let ρo := T * (1 + u) ^ (2 * T) * (η + (1 + u) ^ B * (B * η))
    (N : ℝ) * Real.exp (u * (B + 2 * T)) + ρl ≤ maxFinite →
    (N : ℝ) * V * Real.exp (u * (B + 1 + 2 * T)) + ρo ≤ maxFinite →
    al + ρl < 1 → (V + V * ao + ρo) / (1 - al - ρl) ≤ maxFinite →
    ∃ z ∈ Set.Icc lo hi,
      ExecFloat.Binary.isFinite
        (onlineAverage (fun x => ExecFloat.Binary.CorrectlyRounded.exp x) round round
          start blocks) = true ∧
      |ExecFloat.Binary.toReal
          (onlineAverage (fun x => ExecFloat.Binary.CorrectlyRounded.exp x) round round
            start blocks) - z| ≤
        (1 + u) * ((V * (ao + al + ρl) + ρo) / (1 - al - ρl)) + u * V + η := by
  intro al ao ρl ρo hrangel hrangeo hsmall hquot
  have hV1 : V * (1 + 0 + N * 0) = V := by ring
  have h := onlineAverage_stored_error hformat round round le_rfl le_rfl hround hround hround_one
    hround_one (fun x _ _ => by simp) start blocks hK hV hstart hstartK hstart_le hN hB hT hkeys
    hrangel hrangeo hsmall (by rw [hV1]; exact hquot)
  obtain ⟨β, z, hz, hβ, hfin, hbound⟩ := h
  have hβ1 : β = 1 := by
    have : |β - 1| ≤ 0 := by simpa using hβ
    linarith [abs_nonpos_iff.mp this]
  refine ⟨z, hz, hfin, ?_⟩
  rw [hβ1, one_mul, hV1] at hbound
  exact hbound

/-- FlashAttention-2's recipe sums the normalizer from the unrounded weights and stores the
weights in an IEEE format `low` for the weighted sum. Its output is finite and within the
closed-form bound of `β z` with `z ∈ [lo, hi]` and `|β - 1| ≤ uL + N ηL`, where `uL` and `ηL` are
the unit roundoff and underflow allowance of `low`. -/
theorem onlineAverage_id_roundTo_error (hformat : format.isIEEE = true) {low : FloatFormat}
    (hlow : low.isIEEE = true) (hfrac : low.fracWidth ≤ format.fracWidth)
    (hmin : format.minSubnormalExponent ≤ low.minSubnormalExponent)
    (start : Value) (blocks : List (List (Value × Value))) {K V lo hi : ℝ} {B N T : ℕ}
    (hK : 2 * K ≤ maxFinite) (hV : 0 ≤ V) (hstart : ExecFloat.Binary.isFinite start = true)
    (hstartK : |ExecFloat.Binary.toReal start| ≤ K)
    (hstart_le : ∃ e ∈ blocks.flatten, ExecFloat.Binary.toReal start ≤ ExecFloat.Binary.toReal e.1)
    (hN : blocks.flatten.length ≤ N) (hB : ∀ block ∈ blocks, block.length ≤ B)
    (hT : blocks.length = T)
    (hkeys : ∀ e ∈ blocks.flatten, ExecFloat.Binary.isFinite e.1 = true ∧
      ExecFloat.Binary.isFinite e.2 = true ∧ |ExecFloat.Binary.toReal e.1| ≤ K ∧
      lo ≤ ExecFloat.Binary.toReal e.2 ∧ ExecFloat.Binary.toReal e.2 ≤ hi ∧
      |ExecFloat.Binary.toReal e.2| ≤ V) :
    let uL := Model.unitRoundoffAt low
    let ηL := Model.underflowErrorAt low
    let al := Real.exp (u * (B + 2 * T)) - 1
    let ao := Real.exp (u * (B + 1 + 2 * T)) - 1
    let ρl := T * (1 + u) ^ (2 * T) * η
    let ρo := T * (1 + u) ^ (2 * T) * (η + (1 + u) ^ B * (B * η))
    let V' := V * (1 + uL + N * ηL)
    (N : ℝ) * Real.exp (u * (B + 2 * T)) + ρl ≤ maxFinite →
    (N : ℝ) * V * Real.exp (u * (B + 1 + 2 * T)) + ρo ≤ maxFinite →
    al + ρl < 1 → (V' + V' * ao + ρo) / (1 - al - ρl) ≤ maxFinite →
    ∃ β z : ℝ, z ∈ Set.Icc lo hi ∧ |β - 1| ≤ uL + N * ηL ∧
      ExecFloat.Binary.isFinite
        (onlineAverage (fun x => ExecFloat.Binary.CorrectlyRounded.exp x) id (roundTo low)
          start blocks) = true ∧
      |ExecFloat.Binary.toReal
          (onlineAverage (fun x => ExecFloat.Binary.CorrectlyRounded.exp x) id (roundTo low)
            start blocks) - β * z| ≤
        (1 + u) * ((V' * (ao + al + ρl) + ρo) / (1 - al - ρl)) + u * V' + η :=
  onlineAverage_stored_error hformat id (roundTo low) (Model.unitRoundoffAt_pos low).le
    (Model.underflowErrorAt_pos low).le (fun _ hx h01 => ⟨hx, h01⟩)
    (fun x hx h01 => ⟨(roundTo_contract hformat hlow hfrac hmin x hx h01).1,
      (roundTo_contract hformat hlow hfrac hmin x hx h01).2.1⟩)
    (fun _ _ h => h)
    (fun x hx h1 => by
      rw [(toReal_roundTo hformat hlow hfrac hmin hx (by rw [h1]; simp)).2, h1, Model.roundAt_one])
    (fun x hx h01 => by
      have h := (roundTo_contract hformat hlow hfrac hmin x hx h01).2.2
      rw [abs_of_nonneg h01.1] at h
      exact h)
    start blocks hK hV hstart hstartK hstart_le hN hB hT hkeys

/-- Online softmax that stores its weights in an IEEE format `low` whose grid the working format
contains and sums the stored weights is finite and within the closed-form bound of some
`z ∈ [lo, hi]`. -/
theorem onlineAverage_roundTo_error (hformat : format.isIEEE = true) {low : FloatFormat}
    (hlow : low.isIEEE = true) (hfrac : low.fracWidth ≤ format.fracWidth)
    (hmin : format.minSubnormalExponent ≤ low.minSubnormalExponent)
    (start : Value) (blocks : List (List (Value × Value))) {K V lo hi : ℝ} {B N T : ℕ}
    (hK : 2 * K ≤ maxFinite) (hV : 0 ≤ V) (hstart : ExecFloat.Binary.isFinite start = true)
    (hstartK : |ExecFloat.Binary.toReal start| ≤ K)
    (hstart_le : ∃ e ∈ blocks.flatten, ExecFloat.Binary.toReal start ≤ ExecFloat.Binary.toReal e.1)
    (hN : blocks.flatten.length ≤ N) (hB : ∀ block ∈ blocks, block.length ≤ B)
    (hT : blocks.length = T)
    (hkeys : ∀ e ∈ blocks.flatten, ExecFloat.Binary.isFinite e.1 = true ∧
      ExecFloat.Binary.isFinite e.2 = true ∧ |ExecFloat.Binary.toReal e.1| ≤ K ∧
      lo ≤ ExecFloat.Binary.toReal e.2 ∧ ExecFloat.Binary.toReal e.2 ≤ hi ∧
      |ExecFloat.Binary.toReal e.2| ≤ V) :
    let al := Real.exp (u * (B + 2 * T)) - 1
    let ao := Real.exp (u * (B + 1 + 2 * T)) - 1
    let ρl := T * (1 + u) ^ (2 * T) * η
    let ρo := T * (1 + u) ^ (2 * T) * (η + (1 + u) ^ B * (B * η))
    (N : ℝ) * Real.exp (u * (B + 2 * T)) + ρl ≤ maxFinite →
    (N : ℝ) * V * Real.exp (u * (B + 1 + 2 * T)) + ρo ≤ maxFinite →
    al + ρl < 1 → (V + V * ao + ρo) / (1 - al - ρl) ≤ maxFinite →
    ∃ z ∈ Set.Icc lo hi,
      ExecFloat.Binary.isFinite
        (onlineAverage (fun x => ExecFloat.Binary.CorrectlyRounded.exp x) (roundTo low)
          (roundTo low) start blocks) = true ∧
      |ExecFloat.Binary.toReal
          (onlineAverage (fun x => ExecFloat.Binary.CorrectlyRounded.exp x) (roundTo low)
            (roundTo low) start blocks) - z| ≤
        (1 + u) * ((V * (ao + al + ρl) + ρo) / (1 - al - ρl)) + u * V + η :=
  onlineAverage_stored_error_of_eq hformat (roundTo low)
    (fun x hx h01 => ⟨(roundTo_contract hformat hlow hfrac hmin x hx h01).1,
      (roundTo_contract hformat hlow hfrac hmin x hx h01).2.1⟩)
    (fun x hx h1 => by
      rw [(toReal_roundTo hformat hlow hfrac hmin hx (by rw [h1]; simp)).2, h1, Model.roundAt_one])
    start blocks hK hV hstart hstartK hstart_le hN hB hT hkeys

/-! ## The online softmax weights -/

/-- The residual of the weighted sum of zero values is at most the normalizer's. -/
theorem blockResidual_zero_le {N B : ℕ} {C C' : ℝ} (hu : 0 ≤ u) (hη : 0 ≤ η) :
    blockResidual u η N B 0 C (productResidual u η 0 0 0) ≤ blockResidual u η N B 1 C' η := by
  unfold blockResidual productResidual
  have : 0 ≤ (1 + u) ^ 2 * η * N * 1 * Real.exp C' := by positivity
  simp only [mul_zero, zero_mul, add_zero, zero_add, mul_one] at this ⊢
  nlinarith [this]

/-- In an IEEE format the scalar dictionary's exponential is the correctly rounded one. -/
theorem mathFunctions_exp_eq (hformat : format.isIEEE = true) :
    (MathFunctions.exp : Value → Value) = fun x => ExecFloat.Binary.CorrectlyRounded.exp x :=
  funext fun x => ExecFloat.Binary.CorrectlyRounded.mathFunctions_exp_of_isIEEE x hformat

/-- For configured scores in an IEEE format, the online softmax weight of an allowed key is the
correctly rounded exponential of its score minus the running maximum, divided by the normalizer.
Both come from folding `onlineStep` over the blocks of the allowed keys, with the running maximum
starting at `TorchLean.Softmax.onlineStart`. -/
theorem onlineWeight_eq (hformat : format.isIEEE = true) {n : Nat}
    (blocks : Softmax.BlockPartition) (score : Fin n → Value) (allowed : Fin n → Bool) (j : Fin n)
    (hj : allowed j = true) (first : Value × Value) (rest : List (Value × Value))
    (hkeys : Softmax.onlineKeys score (fun _ => 0) allowed = first :: rest) :
    Softmax.onlineWeight blocks score allowed j =
      ExecFloat.Binary.CorrectlyRounded.exp (score j -
          ((List.toChunks blocks.size (first :: rest)).foldl
            (onlineStep (fun x => ExecFloat.Binary.CorrectlyRounded.exp x) id id)
            (Softmax.onlineStart first rest, 0, 0)).1) /
        ((List.toChunks blocks.size (first :: rest)).foldl
          (onlineStep (fun x => ExecFloat.Binary.CorrectlyRounded.exp x) id id)
          (Softmax.onlineStart first rest, 0, 0)).2.1 := by
  unfold Softmax.onlineWeight
  rw [hkeys]
  simp only [hj, ↓reduceIte, mathFunctions_exp_eq hformat]

/-- A left fold of `max` from the value at any member of a list gives the same maximum. -/
theorem foldl_max_start_eq {ι : Type} (l : List ι) (f : ι → ℝ) {a b : ι} (ha : a ∈ l)
    (hb : b ∈ l) :
    l.foldl (fun m e => max m (f e)) (f a) = l.foldl (fun m e => max m (f e)) (f b) :=
  le_antisymm
    (List.foldl_max_le_of_le l f (List.le_foldl_max_of_mem l f ha)
      fun _ hi => List.le_foldl_max_of_mem l f hi)
    (List.foldl_max_le_of_le l f (List.le_foldl_max_of_mem l f hb)
      fun _ hi => List.le_foldl_max_of_mem l f hi)

/-- `onlineWeight_error` for the list `keys` of the allowed keys. -/
private theorem onlineWeight_error_of_keys (hformat : format.isIEEE = true)
    (blocks : Softmax.BlockPartition) {n : Nat} (score : Fin n → Value) (allowed : Fin n → Bool)
    (j : Fin n) (hj : allowed j = true) (keys : List (Value × Value))
    (hkeys : Softmax.onlineKeys score (fun _ => 0) allowed = keys)
    {K : ℝ} {B : ℕ} (hK : 2 * K ≤ maxFinite)
    (hscore : ∀ i, allowed i = true →
      ExecFloat.Binary.isFinite (score i) = true ∧ |ExecFloat.Binary.toReal (score i)| ≤ K)
    (hB : ∀ block ∈ List.toChunks blocks.size keys, block.length ≤ B) :
    let N := keys.length
    let T := (List.toChunks blocks.size keys).length
    let M := keys.foldl (fun m e => max m (ExecFloat.Binary.toReal e.1))
      (ExecFloat.Binary.toReal (score j))
    let S := (keys.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - M)).sum
    let W := Real.exp (ExecFloat.Binary.toReal (score j) - M)
    let El := (keys.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - M) *
        (Real.exp (u * (M - ExecFloat.Binary.toReal e.1) + u * (B + 3 * T - 1)) - 1)).sum +
      T * residualGrowth u η ^ T * blockResidual u η N B 1 (u * (B + 3 * T - 1)) η
    let Eo := W * (Real.exp (u * (M - ExecFloat.Binary.toReal (score j)) + u) - 1) + η
    (N : ℝ) * 1 * Real.exp (u * (B + 3 * T - 1)) +
        T * residualGrowth u η ^ T * blockResidual u η N B 1 (u * (B + 3 * T - 1)) η ≤
      maxFinite →
    El < S → (W + Eo) / (S - El) ≤ maxFinite →
    ExecFloat.Binary.isFinite (Softmax.onlineWeight blocks score allowed j) = true ∧
      |ExecFloat.Binary.toReal (Softmax.onlineWeight blocks score allowed j) - W / S| ≤
        (1 + u) * ((Eo + W / S * El) / (S - El)) + u * (W / S) + η := by
  have hjkey : (score j, (0 : Value)) ∈ keys := by
    rw [← hkeys]
    simp only [Softmax.onlineKeys, List.mem_map, List.mem_filter, List.mem_finRange,
      true_and]
    exact ⟨j, hj, rfl⟩
  obtain ⟨first, rest, rfl⟩ := List.exists_cons_of_ne_nil (List.ne_nil_of_mem hjkey)
  intro N T M S W El Eo hrangel hden hquot
  have hu : 0 ≤ u := (Model.unitRoundoffAt_pos format).le
  have hη : 0 ≤ η := (Model.underflowErrorAt_pos format).le
  have hmem : ∀ e ∈ first :: rest, ∃ i, allowed i = true ∧ e = (score i, 0) := by
    intro e he
    rw [← hkeys] at he
    simp only [Softmax.onlineKeys, List.mem_map, List.mem_filter, List.mem_finRange,
      true_and] at he
    obtain ⟨i, hi, rfl⟩ := he
    exact ⟨i, hi, rfl⟩
  have hflat := List.flatten_toChunks blocks.size (first :: rest)
  have hkeyfacts : ∀ e ∈ (List.toChunks blocks.size (first :: rest)).flatten,
      ExecFloat.Binary.isFinite e.1 = true ∧ ExecFloat.Binary.isFinite e.2 = true ∧
        |ExecFloat.Binary.toReal e.1| ≤ K := by
    intro e he
    rw [hflat] at he
    obtain ⟨i, hi, rfl⟩ := hmem e he
    exact ⟨(hscore i hi).1, ExecFloat.Binary.isFinite_zero, (hscore i hi).2⟩
  have hfirst := hkeyfacts first (by rw [hflat]; exact List.mem_cons_self ..)
  have hrangeo : (N : ℝ) * 0 * Real.exp (u * (B + 3 * T) + 0) +
      T * residualGrowth u η ^ T *
        blockResidual u η N B 0 (u * (B + 3 * T) + 0) (productResidual u η 0 0 0) ≤
      maxFinite := by
    refine le_trans ?_ hrangel
    have h1 := blockResidual_zero_le (N := N) (B := B) (C := u * (B + 3 * T) + 0)
      (C' := u * (B + 3 * T - 1)) hu hη
    have h2 : 0 ≤ (T : ℝ) * residualGrowth u η ^ T := by
      have := one_le_residualGrowth hu hη; positivity
    have h3 : 0 ≤ (N : ℝ) * 1 * Real.exp (u * (B + 3 * T - 1)) := by positivity
    nlinarith [mul_le_mul_of_nonneg_left h1 h2]
  have hinv := foldl_onlineStep_invariant hformat id (uL := 0) (ηL := 0) le_rfl le_rfl
    (fun x hx h01 => ⟨hx, h01, by simp⟩) (B := B) (N := N) (T := T) (V := 0) hK le_rfl hrangel
    hrangeo (List.toChunks blocks.size (first :: rest)) [] (first.1, 0, 0) 0
    (by simp [T]) (by simp [N, hflat]) hB hkeyfacts
    (fun e he => by
      rw [List.nil_append, hflat] at he
      obtain ⟨i, -, rfl⟩ := hmem e he
      simp [ExecFloat.Binary.toReal_zero])
    ⟨hfirst.1, ExecFloat.Binary.isFinite_zero, ExecFloat.Binary.isFinite_zero, hfirst.2.2,
      by simp, by simp [TermwiseError], by simp [TermwiseError]⟩
  obtain ⟨hmf, hlf, -, hMK, hle, hlerr, -⟩ := hinv
  set final := (List.toChunks blocks.size (first :: rest)).foldl
    (onlineStep (fun x => ExecFloat.Binary.CorrectlyRounded.exp x) id id) (first.1, 0, 0)
    with hfinal
  have hM : ExecFloat.Binary.toReal final.1 = M := by
    rw [foldl_onlineStep_fst, hflat]
    rw [(toReal_foldl_maximum Prod.fst (first :: rest) first.1 hfirst.1
      (fun e he => (hkeyfacts e (by rw [hflat]; exact he)).1)).2]
    exact foldl_max_start_eq (a := first) (b := (score j, 0)) (first :: rest)
      (fun e => ExecFloat.Binary.toReal e.1) (List.mem_cons_self ..) hjkey
  rw [hM] at hlerr
  simp only [List.nil_append, hflat] at hlerr
  have hl : |ExecFloat.Binary.toReal final.2.1 - S| ≤ El := by
    unfold TermwiseError weight shift at hlerr
    simpa [S, El, M] using hlerr
  have hjmem := hjkey
  have hsj := hscore j hj
  have hjle : ExecFloat.Binary.toReal (score j) ≤ ExecFloat.Binary.toReal final.1 :=
    hle (score j, 0) (by rw [List.nil_append, hflat]; exact hjmem)
  have hrange : |ExecFloat.Binary.toReal (score j) - ExecFloat.Binary.toReal final.1| ≤
      maxFinite := by
    obtain ⟨h1, h2⟩ := abs_le.mp hsj.2
    obtain ⟨h3, h4⟩ := abs_le.mp hMK
    exact abs_le.mpr ⟨by linarith, by linarith⟩
  obtain ⟨hof, -, hoerr⟩ := exp_sub_error hformat hsj.1 hmf hjle hrange
  rw [hM] at hoerr
  have hW : 0 < W := Real.exp_pos _
  have hS : 0 < S := by
    simp only [S, List.map_cons, List.sum_cons]
    exact add_pos_of_pos_of_nonneg (Real.exp_pos _)
      (List.sum_nonneg fun x hx => by
        obtain ⟨e, -, rfl⟩ := List.mem_map.mp hx
        exact (Real.exp_pos _).le)
  have hdiv := div_error hformat hlf hof hl hoerr hden (by rwa [abs_of_pos hW])
  rw [abs_of_pos (div_pos hW hS)] at hdiv
  rw [onlineWeight_eq hformat blocks score allowed j hj first rest hkeys,
    Softmax.onlineStart_of_isInf_eq_false first rest (Roundoff.isInf_of_isFinite hfirst.1)]
  exact hdiv

/-- In any IEEE format, the online softmax weight of an allowed key `j` is finite and within
`(1 + u) (Eo + w El) / (S - El) + u w + η` of its softmax weight `w = exp (s_j - M) / S` among the
allowed keys (`onlineWeight_reference_eq`). The bound holds for every block size whose blocks of
the allowed keys have at most `B` keys, and size `0` makes one block of all of them. Here `M` is
the largest allowed score, `S` the sum of the shifted exponentials of the allowed keys, `El` the
normalizer's allowance of `onlineAverage_error` and `Eo` the allowance of the key's own rounded
shift and exponential. It needs finite allowed scores bounded by `K` with `2 K ≤ maxFinite`, the
normalizer's range condition, `El < S` and a range condition on the quotient. -/
theorem onlineWeight_error (hformat : format.isIEEE = true) (blocks : Softmax.BlockPartition)
    {n : Nat} (score : Fin n → Value) (allowed : Fin n → Bool) (j : Fin n) (hj : allowed j = true)
    {K : ℝ} {B : ℕ} (hK : 2 * K ≤ maxFinite)
    (hscore : ∀ i, allowed i = true →
      ExecFloat.Binary.isFinite (score i) = true ∧ |ExecFloat.Binary.toReal (score i)| ≤ K)
    (hB : ∀ block ∈ List.toChunks blocks.size
      (Softmax.onlineKeys score (fun _ => 0) allowed), block.length ≤ B) :
    let keys := Softmax.onlineKeys score (fun _ => 0) allowed
    let N := keys.length
    let T := (List.toChunks blocks.size keys).length
    let M := keys.foldl (fun m e => max m (ExecFloat.Binary.toReal e.1))
      (ExecFloat.Binary.toReal (score j))
    let S := (keys.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - M)).sum
    let W := Real.exp (ExecFloat.Binary.toReal (score j) - M)
    let El := (keys.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - M) *
        (Real.exp (u * (M - ExecFloat.Binary.toReal e.1) + u * (B + 3 * T - 1)) - 1)).sum +
      T * residualGrowth u η ^ T * blockResidual u η N B 1 (u * (B + 3 * T - 1)) η
    let Eo := W * (Real.exp (u * (M - ExecFloat.Binary.toReal (score j)) + u) - 1) + η
    (N : ℝ) * 1 * Real.exp (u * (B + 3 * T - 1)) +
        T * residualGrowth u η ^ T * blockResidual u η N B 1 (u * (B + 3 * T - 1)) η ≤
      maxFinite →
    El < S → (W + Eo) / (S - El) ≤ maxFinite →
    ExecFloat.Binary.isFinite (Softmax.onlineWeight blocks score allowed j) = true ∧
      |ExecFloat.Binary.toReal (Softmax.onlineWeight blocks score allowed j) - W / S| ≤
        (1 + u) * ((Eo + W / S * El) / (S - El)) + u * (W / S) + η :=
  onlineWeight_error_of_keys hformat blocks score allowed j hj _ rfl hK hscore hB

/-- The reference of `onlineWeight_error` is the softmax weight of the key among the allowed keys:
the shift by the running maximum cancels. -/
theorem onlineWeight_reference_eq {n : Nat} (score : Fin n → Value) (allowed : Fin n → Bool)
    (j : Fin n) :
    let keys := Softmax.onlineKeys score (fun _ => 0) allowed
    let M := keys.foldl (fun m e => max m (ExecFloat.Binary.toReal e.1))
      (ExecFloat.Binary.toReal (score j))
    Real.exp (ExecFloat.Binary.toReal (score j) - M) /
        (keys.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - M)).sum =
      Real.exp (ExecFloat.Binary.toReal (score j)) /
        ∑ i, if allowed i then Real.exp (ExecFloat.Binary.toReal (score i)) else 0 := by
  intro keys M
  have hshift (x : ℝ) : Real.exp (x - M) = Real.exp x * Real.exp (-M) := by
    rw [← Real.exp_add]; ring_nf
  simp only [keys, Softmax.onlineKeys, List.map_map, Fin.sum_univ_def, List.sum_map_ite,
    Bool.decide_eq_true, List.sum_map_zero, add_zero, Function.comp_def, hshift,
    List.sum_map_mul_right]
  exact mul_div_mul_right _ _ (Real.exp_ne_zero _)

end Configured

end TorchLean.OnlineSoftmax

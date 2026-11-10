/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Verify.RuntimeApprox.Constants
public import TorchLean.Verify.RuntimeApprox.Roundoff.Coverage
public import TorchLean.Verify.RuntimeApprox.NF.CROWN
public import TorchLean.Verify.NumericalCertificate
public import TorchLean.Verify.Floats.IEEE754.Expressions
public import TorchLean.Verify.RuntimeApprox.NF.Attention
public import TorchLean.Verify.RuntimeApprox.NF.Convolution
public import TorchLean.Verify.RuntimeApprox.NF.EndToEnd
public import TorchLean.Verify.RuntimeApprox.NF.Normalization
public import TorchLean.Verify.RuntimeApprox.NF.Optimizers
public import TorchLean.Verify.RuntimeApprox.NF.ReductionOps
public import TorchLean.Verify.Floats.IEEE754.Reductions
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Autograd -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.SoftmaxShift
public import TorchLean.Verify.RuntimeApprox.TransformerBlock
public import FloatLib.Floats.Formats.BinaryInterchange.Automation.Roundoff
public import FloatLib.Numerics.Automation.Attributes
public import TorchLean.Tensor.Literal
public import TorchLean.Tensor.Tactic.Init
public import TorchLean.Verify.RuntimeApprox.OnlineSoftmax
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Elementwise

/-!
# Blocked online softmax attention as a generated fold

`Softmax.onlineStep`, the step of the online schedule, is one block of online softmax
attention over any type with a `Context`: the `TorchLean.Context.maximum` of the running maximum and
the block's scores, then the normalizer and the weighted sum of the values, each rescaled to that
maximum and extended by the block's terms. For every IEEE format and every list of blocks of any
sizes, `roundoff mixed theorem` proves through `List.fold_relation` that the normalizer and the
output of one query, computed with the correctly rounded exponential, are finite and have mixed
bounds against the same schedule in real arithmetic on the decoded scores and values
(`blocked_normalizer_error`, `blocked_attention_error`). `blocked_attention_reference_eq` identifies
that reference with the output of `onlineStep Real.exp id id` on the decoded blocks. The routes of
`Tensor.softmax` and `Tensor.attention` by the online schedule fold this step over the blocks of the
allowed keys of a fiber, a list that the mask selects, which these loops do not read. `roundoff`
reads them through the hand bounds that
`TorchLean.Verify.RuntimeApprox.Roundoff.Rules.OnlineSoftmax` registers.

The budgets are recurrences over the blocks (`List.foldBudget` of the generated budget step). A
block charges the rescaled sums the rounding of the shift of the maximum, of its exponential, of
the product and of the sum, and each of its own terms the rounding of its shifted exponential and
of the additions after it. The range conditions are stated on the states the fold visits.

The budgets of the normalizer are at most the allowance `El` of `onlineAverage_error`
(`normalizerBudgets_le`). By induction over the blocks, after `k` blocks the relative budget times
the normalizer stays within the weights' allowances `exp (u * (m - s) + c) - 1` at the running
maximum `m`, with `c` of `B + 3 k - 1` units, and the absolute budget within that theorem's
residual recurrence. Under the conditions of `blocked_normalizer_error` the normalizer is therefore
within `El` of `S` (`blocked_normalizer_abs_sub_le`). Those conditions are not derived here from
the hypotheses of `onlineAverage_error`.
-/

public section

namespace TorchLean.OnlineSoftmax

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange FloatLib.Numerics
open Softmax (onlineStep)

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- The softmax normalizer of one query over blocks of (score, value) pairs, from the running
maximum `start`. -/
def blockedNormalizer (blocks : List (List (Value × Value))) (start : Value) : Value :=
  (blocks.foldl (onlineStep ExecFloat.Binary.CorrectlyRounded.exp id id) (start, 0, 0)).2.1

/-- Online softmax attention of one query over blocks of (score, value) pairs, from the running
maximum `start`. -/
def blockedAttention (blocks : List (List (Value × Value))) (start : Value) : Value :=
  let state := blocks.foldl (onlineStep ExecFloat.Binary.CorrectlyRounded.exp id id) (start, 0, 0)
  state.2.2 / state.2.1

/-- The blocked normalizer is finite and has a mixed bound against its real fold. -/
roundoff mixed theorem blocked_normalizer_error :=
  fun (blocks : List (List (Value × Value))) (start : Value) (_hformat : format.isIEEE = true) =>
    ExecFloat.Binary.toReal (blockedNormalizer blocks start)

/-- Blocked attention is finite and has a mixed bound against the quotient of its real folds. -/
roundoff mixed theorem blocked_attention_error :=
  fun (blocks : List (List (Value × Value))) (start : Value) (_hformat : format.isIEEE = true) =>
    ExecFloat.Binary.toReal (blockedAttention blocks start)

/-- One block of the real fold of the weighted sum and its maximum, on decoded entries. -/
noncomputable def realOutputStep (state : ℝ × ℝ) (block : List (Value × Value)) : ℝ × ℝ :=
  let m := block.foldl (fun m e => max m (ExecFloat.Binary.toReal e.1)) state.1
  (m, Real.exp (state.1 - m) * state.2 +
    block.foldl (fun o e => o + Real.exp (ExecFloat.Binary.toReal e.1 - m) *
      ExecFloat.Binary.toReal e.2) 0)

/-- One block of the real fold of the normalizer and its maximum, on decoded entries. -/
noncomputable def realNormalizerStep (state : ℝ × ℝ) (block : List (Value × Value)) : ℝ × ℝ :=
  let m := block.foldl (fun m e => max m (ExecFloat.Binary.toReal e.1)) state.1
  (m, Real.exp (state.1 - m) * state.2 +
    block.foldl (fun l e => l + Real.exp (ExecFloat.Binary.toReal e.1 - m)) 0)

/-- The decoded (score, value) pairs of a block. -/
noncomputable def decodeBlock (block : List (Value × Value)) : List (ℝ × ℝ) :=
  block.map fun e => (ExecFloat.Binary.toReal e.1, ExecFloat.Binary.toReal e.2)

/-- The real folds `realOutputStep` and `realNormalizerStep` carry the maximum with the weighted
sum and with the normalizer of `onlineStep Real.exp id id` on the decoded blocks. -/
theorem foldl_realSteps (blocks : List (List (Value × Value))) (state : ℝ × ℝ × ℝ) :
    blocks.foldl realOutputStep (state.1, state.2.2) =
        (((blocks.map decodeBlock).foldl (onlineStep Real.exp id id) state).1,
          ((blocks.map decodeBlock).foldl (onlineStep Real.exp id id) state).2.2) ∧
      blocks.foldl realNormalizerStep (state.1, state.2.1) =
        (((blocks.map decodeBlock).foldl (onlineStep Real.exp id id) state).1,
          ((blocks.map decodeBlock).foldl (onlineStep Real.exp id id) state).2.1) := by
  induction blocks generalizing state with
  | nil => simp
  | cons block blocks ih =>
    obtain ⟨houtput, hnormalizer⟩ := ih (onlineStep Real.exp id id state (decodeBlock block))
    simp only [List.foldl_cons, List.map_cons]
    constructor
    · rw [← houtput]
      congr 1
      simp [realOutputStep, onlineStep, decodeBlock, List.foldl_map, Context.maximum_real]
    · rw [← hnormalizer]
      congr 1
      simp [realNormalizerStep, onlineStep, decodeBlock, List.foldl_map, Context.maximum_real]

/-- The reference of `blocked_attention_error` is the output of `onlineStep Real.exp id id` on
the decoded blocks. -/
theorem blocked_attention_reference_eq (blocks : List (List (Value × Value))) (start : Value) :
    (blocks.foldl realOutputStep (ExecFloat.Binary.toReal start, 0)).2 /
        (blocks.foldl realNormalizerStep (ExecFloat.Binary.toReal start, 0)).2 =
      ((blocks.map decodeBlock).foldl (onlineStep Real.exp id id)
          (ExecFloat.Binary.toReal start, 0, 0)).2.2 /
        ((blocks.map decodeBlock).foldl (onlineStep Real.exp id id)
          (ExecFloat.Binary.toReal start, 0, 0)).2.1 := by
  obtain ⟨houtput, hnormalizer⟩ := foldl_realSteps blocks (ExecFloat.Binary.toReal start, 0, 0)
  rw [houtput, hnormalizer]

section Budgets

open OnlineSoftmax (residualGrowth blockResidual step_residual_le)

local notation "u" => Model.unitRoundoffAt format
local notation "η" => Model.underflowErrorAt format

/-- One addition of a weight against the maximum `m` to a block's real sum. -/
noncomputable def weightStep (m reference : ℝ) (e : Value × Value) : ℝ :=
  reference + Real.exp (ExecFloat.Binary.toReal e.1 - m)

/-- The relative budget of one addition of a weight against `m` to a block's sum: the rounding of
the shift, of its exponential and of the sum. -/
noncomputable def weightRelative (m : ℝ) (_ : Value) (reference relative _ : ℝ)
    (e : Value × Value) : ℝ :=
  (1 + u) * ((relative * |reference| +
      ((1 + u) * (Real.exp (u * |ExecFloat.Binary.toReal e.1 - m|) - 1) + u) *
        Real.exp (ExecFloat.Binary.toReal e.1 - m)) /
    |reference + Real.exp (ExecFloat.Binary.toReal e.1 - m)|) + u

/-- The absolute budget of one addition of a weight to a block's sum: the underflow of the
exponential, rounded with the sum. -/
noncomputable def weightAbsolute (_ : Value) (_ _ absolute : ℝ) (_ : Value × Value) : ℝ :=
  (1 + u) * (absolute + η)

/-- The rounded step of a block's sum of weights from the state `state`. -/
def weightSumStep (state : Value × Value × Value) (block : List (Value × Value)) (l : Value)
    (e : Value × Value) : Value :=
  l + id (ExecFloat.Binary.CorrectlyRounded.exp
    (e.1 - block.foldl (fun m e => Context.maximum m e.1) state.1))

/-- The relative and absolute budgets that `blocked_normalizer_error` charges a block, from the
running maximum and normalizer `realState` and the budgets before the block. -/
noncomputable def normalizerBudget (state : Value × Value × Value) (realState budgets : ℝ × ℝ)
    (block : List (Value × Value)) : ℝ × ℝ :=
  let m := block.foldl (fun reference value => max reference (ExecFloat.Binary.toReal value.1))
    realState.1
  let σ := (1 + u) * (Real.exp (u * |realState.1 - m|) - 1) + u
  let scaled := Real.exp (realState.1 - m) * realState.2
  let inner := block.foldl (fun reference e => reference +
    Real.exp (ExecFloat.Binary.toReal e.1 - m)) 0
  ((1 + u) * ((((1 + u) * (σ + budgets.1 + σ * budgets.1) + u) * |scaled| +
        List.foldRelative (weightSumStep state block) (weightStep m) (weightRelative m)
          weightAbsolute 0 0 0 0 block * |inner|) / |scaled + inner|) + u,
    (1 + u) * ((1 + u) * ((1 + budgets.1) * η * |realState.2| +
        (1 + σ) * budgets.2 * Real.exp (realState.1 - m) + η * budgets.2) + η +
      List.foldAbsolute (weightSumStep state block) (weightStep m) (weightRelative m)
        weightAbsolute 0 0 0 0 block))

/-- The state of a block's weight sum after `k` entries, for unit roundoff `unit` and underflow
error `under`: the sum is `V`, its relative budget is at most what the allowance `W` gives after
`k` additions, and its absolute budget is at most `k` underflows grown by the later additions. -/
def WeightBound {α : Type} (unit under : ℝ) (k : ℕ) (V W : ℝ) (state : α × ℝ × ℝ × ℝ) : Prop :=
  state.2.1 = V ∧ 0 ≤ state.2.2.1 ∧ (1 + state.2.2.1) * V ≤ W * Real.exp (unit * (k + 1)) ∧
    0 ≤ state.2.2.2 ∧ state.2.2.2 ≤ (1 + unit) ^ k * (k * under)

/-- One addition of a weight against a maximum `m` at least its score keeps `WeightBound`. -/
theorem weightBound_step (step : Value → Value × Value → Value) {m V W : ℝ} {k : ℕ}
    {state : Value × ℝ × ℝ × ℝ} {e : Value × Value} (he : ExecFloat.Binary.toReal e.1 ≤ m)
    (hV : 0 ≤ V) (hW : 0 ≤ W) (h : WeightBound u η k V W state) :
    WeightBound u η (k + 1) (V + Real.exp (ExecFloat.Binary.toReal e.1 - m))
      (W + Real.exp (ExecFloat.Binary.toReal e.1 - m) *
        Real.exp (u * (m - ExecFloat.Binary.toReal e.1)))
      (List.mixedFoldStep step (weightStep m) (weightRelative m) weightAbsolute state e) := by
  obtain ⟨hstate, hr, hbound, ha, habs⟩ := h
  obtain ⟨acc, V', r, a⟩ := state
  simp only at hstate hr hbound ha habs
  subst hstate
  have hu : 0 ≤ u := (Model.unitRoundoffAt_pos format).le
  have hη : 0 ≤ η := (Model.underflowErrorAt_pos format).le
  have hexpu : 1 + u ≤ Real.exp u := (add_comm 1 u).trans_le (Real.add_one_le_exp u)
  have hw : 0 < Real.exp (ExecFloat.Binary.toReal e.1 - m) := Real.exp_pos _
  have hE : 1 ≤ Real.exp (u * (m - ExecFloat.Binary.toReal e.1)) :=
    Real.one_le_exp (mul_nonneg hu (by linarith))
  have hdist : |ExecFloat.Binary.toReal e.1 - m| = m - ExecFloat.Binary.toReal e.1 := by
    rw [abs_sub_comm, abs_of_nonneg (by linarith)]
  have hX : 0 < V' + Real.exp (ExecFloat.Binary.toReal e.1 - m) := by linarith
  simp only [WeightBound, List.mixedFoldStep, weightStep, weightRelative, weightAbsolute]
  rw [hdist, abs_of_nonneg hV, abs_of_pos hX]
  generalize Real.exp (ExecFloat.Binary.toReal e.1 - m) = w at hw hX ⊢
  generalize Real.exp (u * (m - ExecFloat.Binary.toReal e.1)) = E at hE ⊢
  have hτ : 0 ≤ (1 + u) * (E - 1) + u := by nlinarith
  refine ⟨trivial, ?_, ?_, by positivity, ?_⟩
  · have := div_nonneg (add_nonneg (mul_nonneg hr hV) (mul_nonneg hτ hw.le)) hX.le
    positivity
  · have key : (1 + ((1 + u) * ((r * V' + ((1 + u) * (E - 1) + u) * w) / (V' + w)) + u)) *
        (V' + w) = (1 + u) * ((1 + r) * V') + (1 + u) ^ 2 * (E * w) := by
      field_simp
      ring
    rw [key]
    push_cast
    have g1 : (1 + u) * Real.exp (u * (k + 1)) ≤ Real.exp (u * (k + 1 + 1)) := by
      rw [show u * (k + 1 + 1) = u + u * (k + 1) by ring, Real.exp_add]
      exact mul_le_mul_of_nonneg_right hexpu (Real.exp_pos _).le
    have g2 : (1 + u) ^ 2 ≤ Real.exp (u * (k + 1 + 1)) := by
      calc (1 + u) ^ 2 ≤ Real.exp u ^ 2 := pow_le_pow_left₀ (by linarith) hexpu 2
        _ = Real.exp (u + u) := by rw [sq, Real.exp_add]
        _ ≤ Real.exp (u * (k + 1 + 1)) := by
          apply Real.exp_le_exp.mpr
          have : (0 : ℝ) ≤ k := Nat.cast_nonneg k
          nlinarith
    have h1 : (1 + u) * ((1 + r) * V') ≤ W * Real.exp (u * (k + 1 + 1)) := by
      calc (1 + u) * ((1 + r) * V') ≤ (1 + u) * (W * Real.exp (u * (k + 1))) :=
            mul_le_mul_of_nonneg_left hbound (by linarith)
        _ = W * ((1 + u) * Real.exp (u * (k + 1))) := by ring
        _ ≤ W * Real.exp (u * (k + 1 + 1)) := mul_le_mul_of_nonneg_left g1 hW
    have h2 : (1 + u) ^ 2 * (E * w) ≤ w * E * Real.exp (u * (k + 1 + 1)) := by
      calc (1 + u) ^ 2 * (E * w) ≤ Real.exp (u * (k + 1 + 1)) * (E * w) :=
            mul_le_mul_of_nonneg_right g2 (by positivity)
        _ = w * E * Real.exp (u * (k + 1 + 1)) := by ring
    nlinarith
  · push_cast
    have hpow : 1 ≤ (1 + u) ^ k := one_le_pow₀ (by linarith)
    calc (1 + u) * (a + η) ≤ (1 + u) * ((1 + u) ^ k * (k * η) + (1 + u) ^ k * η) := by
          gcongr
          nlinarith
      _ = (1 + u) ^ (k + 1) * ((k + 1) * η) := by ring

/-- `WeightBound` after a block of entries whose scores are at most `m`. -/
theorem weightBound_foldl (step : Value → Value × Value → Value) {m : ℝ} :
    ∀ (Q : List (Value × Value)) {state : Value × ℝ × ℝ × ℝ} {k : ℕ} {V W : ℝ},
      (∀ e ∈ Q, ExecFloat.Binary.toReal e.1 ≤ m) → 0 ≤ V → 0 ≤ W → WeightBound u η k V W state →
      WeightBound u η (k + Q.length)
        (V + (Q.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - m)).sum)
        (W + (Q.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - m) *
          Real.exp (u * (m - ExecFloat.Binary.toReal e.1))).sum)
        (Q.foldl (List.mixedFoldStep step (weightStep m) (weightRelative m) weightAbsolute) state)
  | [], _, _, _, _, _, _, _, h => by simpa using h
  | e :: Q, state, k, V, W, hQ, hV, hW, h => by
      have h' := weightBound_step step (hQ e (List.mem_cons_self ..)) hV hW h
      have := weightBound_foldl step Q (fun x hx => hQ x (List.mem_cons_of_mem _ hx))
        (by positivity) (by positivity) h'
      simp only [List.foldl_cons, List.length_cons, List.map_cons, List.sum_cons]
      rw [show k + (Q.length + 1) = k + 1 + Q.length by omega, ← add_assoc V, ← add_assoc W]
      exact this

/-- A sum of nonnegative terms is nonnegative. -/
theorem sum_map_nonneg {α : Type} (Q : List α) {f : α → ℝ} (h : ∀ e, 0 ≤ f e) :
    0 ≤ (Q.map f).sum :=
  List.sum_nonneg fun x hx => by
    obtain ⟨e, -, rfl⟩ := List.mem_map.mp hx
    exact h e

/-- Rescaling weights from the maximum `m` to `m'`. -/
theorem sum_exp_shift (Q : List (Value × Value)) (m m' : ℝ) :
    Real.exp (m - m') * (Q.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - m)).sum =
      (Q.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - m')).sum := by
  rw [← List.sum_map_mul_left]
  congr 1
  apply List.map_congr_left
  intro e _
  rw [← Real.exp_add]
  congr 1
  ring

/-- Rescaling weights and their allowances from the maximum `m` to `m'`, with the allowance
grown by `x`. -/
theorem sum_allowance_shift (Q : List (Value × Value)) (m m' c x : ℝ) :
    Real.exp (u * (m' - m)) * Real.exp (m - m') * Real.exp x * (Q.map fun e =>
        Real.exp (ExecFloat.Binary.toReal e.1 - m) *
          Real.exp (u * (m - ExecFloat.Binary.toReal e.1) + c)).sum =
      (Q.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - m') *
        Real.exp (u * (m' - ExecFloat.Binary.toReal e.1) + (c + x))).sum := by
  rw [← List.sum_map_mul_left]
  congr 1
  apply List.map_congr_left
  intro e _
  simp only [← Real.exp_add]
  congr 1
  ring

/-- Growing every allowance of a sum of weights. -/
theorem sum_exp_mul_exp_le {α : Type} (Q : List α) (f g : α → ℝ) {x y : ℝ} (h : x ≤ y) :
    (Q.map fun e => Real.exp (f e) * Real.exp (g e)).sum * Real.exp x ≤
      (Q.map fun e => Real.exp (f e) * Real.exp (g e + y)).sum := by
  rw [← List.sum_map_mul_right]
  apply List.sum_le_sum
  intro e _
  rw [mul_assoc, ← Real.exp_add]
  gcongr

/-- Each weight times its allowance is at most `exp c`, since the score is at most `m`. -/
theorem sum_allowance_le (Q : List (Value × Value)) {m : ℝ} (c : ℝ)
    (hQ : ∀ e ∈ Q, ExecFloat.Binary.toReal e.1 ≤ m) :
    (Q.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - m) *
      Real.exp (u * (m - ExecFloat.Binary.toReal e.1) + c)).sum ≤ Q.length * Real.exp c := by
  induction Q with
  | nil => simp
  | cons e Q ih =>
    simp only [List.map_cons, List.sum_cons, List.length_cons]
    have he := hQ e (List.mem_cons_self ..)
    have h1 : Real.exp (ExecFloat.Binary.toReal e.1 - m) *
        Real.exp (u * (m - ExecFloat.Binary.toReal e.1) + c) ≤ Real.exp c := by
      rw [← Real.exp_add, Real.exp_le_exp]
      have hu1 : u < 1 := Model.unitRoundoffAt_lt_one format
      nlinarith
    have := ih fun x hx => hQ x (List.mem_cons_of_mem _ hx)
    push_cast
    linarith

/-- The relative budget of a block step against its allowance: `E` grows the allowance of the
weights before the block under the maximum after it, `D` rescales their sum `L`, and `I` is the
block's sum. -/
theorem normalizer_relative_le {v r ρ E D L I RP W ε e3 e1 QP Qb : ℝ} (hu : 0 ≤ v)
    (hr : 0 ≤ r) (hρ : 0 ≤ ρ) (hE : 1 ≤ E) (hD : 0 ≤ D) (hL : 0 ≤ L) (hI : 0 ≤ I)
    (hW : 0 ≤ W) (hε : 0 ≤ ε) (hrel : (1 + r) * L ≤ RP) (hρI : (1 + ρ) * I ≤ W * ε)
    (h3 : (1 + v) ^ 3 ≤ e3) (h1 : 1 + v ≤ e1) (hQP : E * D * e3 * RP = QP)
    (hQb : W * ε * e1 ≤ Qb) :
    0 ≤ (1 + v) * ((((1 + v) * ((1 + v) * (E - 1) + v + r + ((1 + v) * (E - 1) + v) * r) +
        v) * (D * L) + ρ * I) / (D * L + I)) + v ∧
      (1 + ((1 + v) * ((((1 + v) * ((1 + v) * (E - 1) + v + r + ((1 + v) * (E - 1) + v) * r) +
        v) * (D * L) + ρ * I) / (D * L + I)) + v)) * (D * L + I) ≤ QP + Qb := by
  have h1u : 0 ≤ 1 + v := by linarith
  have hδ : 0 ≤ (1 + v) * (E - 1) + v := by nlinarith
  have hr1 : 0 ≤ (1 + v) * ((1 + v) * (E - 1) + v + r + ((1 + v) * (E - 1) + v) * r) + v :=
    add_nonneg (mul_nonneg h1u (add_nonneg (add_nonneg hδ hr) (mul_nonneg hδ hr))) hu
  have hX : 0 ≤ D * L + I := add_nonneg (mul_nonneg hD hL) hI
  refine ⟨add_nonneg (mul_nonneg h1u (div_nonneg (add_nonneg (mul_nonneg hr1
    (mul_nonneg hD hL)) (mul_nonneg hρ hI)) hX)) hu, ?_⟩
  have hRP : 0 ≤ RP := le_trans (mul_nonneg (by linarith) hL) hrel
  have he3 : 0 ≤ e3 := le_trans (pow_nonneg h1u 3) h3
  have he1 : 0 ≤ e1 := le_trans h1u h1
  have hQP0 : 0 ≤ QP := hQP ▸ mul_nonneg (mul_nonneg (mul_nonneg (by linarith) hD) he3) hRP
  have hQb0 : 0 ≤ Qb := le_trans (mul_nonneg (mul_nonneg hW hε) he1) hQb
  rcases hX.eq_or_lt with hX0 | hX0
  · rw [← hX0, mul_zero]
    linarith
  have key : ∀ N : ℝ, (1 + ((1 + v) * (N / (D * L + I)) + v)) * (D * L + I) =
      (1 + v) * (D * L + I) + (1 + v) * N := fun N => by
    field_simp
    ring
  rw [key]
  have t1 : (1 + v) ^ 3 * (E * D) * ((1 + r) * L) ≤ QP := by
    rw [← hQP]
    calc (1 + v) ^ 3 * (E * D) * ((1 + r) * L) ≤ e3 * (E * D) * RP :=
          mul_le_mul (mul_le_mul_of_nonneg_right h3 (mul_nonneg (by linarith) hD)) hrel
            (mul_nonneg (by linarith) hL) (mul_nonneg he3 (mul_nonneg (by linarith) hD))
      _ = E * D * e3 * RP := by ring
  have t2 : (1 + v) * ((1 + ρ) * I) ≤ Qb :=
    calc (1 + v) * ((1 + ρ) * I) ≤ e1 * (W * ε) :=
          mul_le_mul h1 hρI (mul_nonneg (by linarith) hI) he1
      _ = W * ε * e1 := by ring
      _ ≤ Qb := hQb
  linarith

/-- The absolute budget of a block step against the residual recurrence: the weights before the
block weigh `(1 + r) L ≤ n c`, and their rescaled residual grows by at most `eu ≥ 1 + v`. -/
theorem normalizer_absolute_le {v w r a α L E D eu n c q Z T : ℝ} (hu : 0 ≤ v) (hw : 0 ≤ w)
    (hr : 0 ≤ r) (ha : 0 ≤ a) (hα : 0 ≤ α) (hL : 0 ≤ L) (hE : 1 ≤ E) (hD : 0 ≤ D)
    (hED : E * D ≤ 1) (heu : 1 + v ≤ eu) (hrel : (1 + r) * L ≤ n * c) (hαq : α ≤ q * Z)
    (h : (1 + v) * ((1 + v) * (eu * a + w * (n * 1 * c + a)) + w + q * (0 + Z)) ≤ T) :
    0 ≤ (1 + v) * ((1 + v) * ((1 + r) * w * L + (1 + ((1 + v) * (E - 1) + v)) * a * D +
        w * a) + w + α) ∧
      (1 + v) * ((1 + v) * ((1 + r) * w * L + (1 + ((1 + v) * (E - 1) + v)) * a * D +
        w * a) + w + α) ≤ T := by
  have h1u : 0 ≤ 1 + v := by linarith
  have hδ : 0 ≤ 1 + ((1 + v) * (E - 1) + v) := by nlinarith
  refine ⟨mul_nonneg h1u (add_nonneg (add_nonneg (mul_nonneg h1u (add_nonneg (add_nonneg
    (mul_nonneg (mul_nonneg (by linarith) hw) hL) (mul_nonneg (mul_nonneg hδ ha) hD))
      (mul_nonneg hw ha))) hw) hα), le_trans ?_ h⟩
  have t1 := mul_le_mul_of_nonneg_left hrel hw
  have t2 := mul_le_mul_of_nonneg_left
    ((mul_le_mul_of_nonneg_left hED h1u).trans (by linarith : (1 + v) * 1 ≤ eu)) ha
  have hx : (1 + r) * w * L + (1 + ((1 + v) * (E - 1) + v)) * a * D + w * a ≤
      eu * a + w * (n * 1 * c + a) := by
    linarith
  have := mul_le_mul_of_nonneg_left hx h1u
  apply mul_le_mul_of_nonneg_left _ h1u
  linarith

/-- `realState` and `budgets`, the real state and the budgets of the blocked normalizer after `k`
of `T` blocks with entries `P` from the maximum `start`, are within the allowance for unit
roundoff `unit`, underflow error `under`, `N` keys and blocks of at most `B` keys. -/
def NormalizerBound (unit under : ℝ) (B N T k : ℕ) (start : ℝ) (P : List (Value × Value))
    (realState budgets : ℝ × ℝ) : Prop :=
  realState.1 = P.foldl (fun m e => max m (ExecFloat.Binary.toReal e.1)) start ∧
    realState.2 = (P.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - realState.1)).sum ∧
    0 ≤ budgets.1 ∧
    (1 + budgets.1) * realState.2 ≤ (P.map fun e =>
      Real.exp (ExecFloat.Binary.toReal e.1 - realState.1) *
        Real.exp (unit * (realState.1 - ExecFloat.Binary.toReal e.1) +
          unit * (B + 3 * k - 1))).sum ∧
    0 ≤ budgets.2 ∧
    budgets.2 ≤ k * residualGrowth unit under ^ k *
      blockResidual unit under N B 1 (unit * (B + 3 * T - 1)) under

/-- One block keeps `NormalizerBound`, with one more block counted. -/
theorem normalizerBound_step {B N T k : ℕ} {start : ℝ} {P block : List (Value × Value)}
    {state : Value × Value × Value} {realState budgets : ℝ × ℝ} (hB : block.length ≤ B)
    (hk : k + 1 ≤ T) (hN : (P ++ block).length ≤ N)
    (h : NormalizerBound u η B N T k start P realState budgets) :
    NormalizerBound u η B N T (k + 1) start (P ++ block) (realNormalizerStep realState block)
      (normalizerBudget state realState budgets block) := by
  obtain ⟨m, L⟩ := realState
  obtain ⟨r, a⟩ := budgets
  obtain ⟨hm, hL, hr, hrel, ha, habs⟩ := h
  simp only at hm hL hr hrel ha habs
  have hu : 0 ≤ u := (Model.unitRoundoffAt_pos format).le
  have hu1 : u < 1 := Model.unitRoundoffAt_lt_one format
  have hη : 0 ≤ η := (Model.underflowErrorAt_pos format).le
  have hexpu : 1 + u ≤ Real.exp u := (add_comm 1 u).trans_le (Real.add_one_le_exp u)
  have h1u : 0 ≤ 1 + u := by linarith
  have hP : ∀ e ∈ P, ExecFloat.Binary.toReal e.1 ≤ m := by
    intro e he
    rw [hm]
    exact List.le_foldl_max_of_mem P (fun e : Value × Value => ExecFloat.Binary.toReal e.1) he
  have hmm' : m ≤ block.foldl (fun m e => max m (ExecFloat.Binary.toReal e.1)) m :=
    List.le_foldl_max_init block (fun e : Value × Value => ExecFloat.Binary.toReal e.1) m
  have hblock : ∀ e ∈ block, ExecFloat.Binary.toReal e.1 ≤
      block.foldl (fun m e => max m (ExecFloat.Binary.toReal e.1)) m := fun e he =>
    List.le_foldl_max_of_mem block (fun e : Value × Value => ExecFloat.Binary.toReal e.1) he
  have hmax : (P ++ block).foldl (fun m e => max m (ExecFloat.Binary.toReal e.1)) start =
      block.foldl (fun m e => max m (ExecFloat.Binary.toReal e.1)) m := by
    rw [List.foldl_append, ← hm]
  simp only [NormalizerBound, normalizerBudget, realNormalizerStep, List.foldRelative,
    List.foldAbsolute, hmax, Softmax.foldl_add_eq_add_sum, zero_add]
  generalize block.foldl (fun m e => max m (ExecFloat.Binary.toReal e.1)) m = m' at hmm' hblock ⊢
  obtain ⟨-, hρ, hρbound, hα, hαbound⟩ := weightBound_foldl (weightSumStep state block) block
    (state := ((0 : Value), (0 : ℝ), (0 : ℝ), (0 : ℝ))) (k := 0) (V := 0) (W := 0) hblock
    le_rfl le_rfl ⟨rfl, le_rfl, by simp, le_rfl, by simp⟩
  simp only [zero_add] at hρbound hαbound
  generalize (List.foldl (List.mixedFoldStep (weightSumStep state block) (weightStep m')
    (weightRelative m') weightAbsolute) ((0 : Value), (0 : ℝ), (0 : ℝ), (0 : ℝ)) block).2.2.1 = ρ
    at hρ hρbound ⊢
  generalize (List.foldl (List.mixedFoldStep (weightSumStep state block) (weightStep m')
    (weightRelative m') weightAbsolute) ((0 : Value), (0 : ℝ), (0 : ℝ), (0 : ℝ)) block).2.2.2 = α
    at hα hαbound ⊢
  rw [abs_sub_comm m m', abs_of_nonneg (sub_nonneg.mpr hmm')]
  have hLnn : 0 ≤ L := hL ▸ sum_map_nonneg P fun e => (Real.exp_pos _).le
  have hI : 0 ≤ (block.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - m')).sum :=
    sum_map_nonneg block fun e => (Real.exp_pos _).le
  have hD := Real.exp_pos (m - m')
  rw [abs_of_nonneg hLnn, abs_of_nonneg (mul_nonneg hD.le hLnn), abs_of_nonneg hI,
    abs_of_nonneg (add_nonneg (mul_nonneg hD.le hLnn) hI)]
  simp only [List.map_append, List.sum_append]
  have hc : u * ((B : ℝ) + 3 * ((k + 1 : ℕ) : ℝ) - 1) = u * (B + 3 * k - 1) + 3 * u := by
    push_cast
    ring
  rw [hc]
  have hb : Real.exp (m - m') * L =
      (P.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - m')).sum := by
    rw [hL]
    exact sum_exp_shift P m m'
  have hE : 1 ≤ Real.exp (u * (m' - m)) :=
    Real.one_le_exp (mul_nonneg hu (sub_nonneg.mpr hmm'))
  have hED : Real.exp (u * (m' - m)) * Real.exp (m - m') ≤ 1 := by
    rw [← Real.exp_add, Real.exp_le_one_iff]
    nlinarith
  have h3 : (1 + u) ^ 3 ≤ Real.exp (3 * u) :=
    calc (1 + u) ^ 3 ≤ Real.exp u ^ 3 := pow_le_pow_left₀ h1u hexpu 3
      _ = Real.exp (3 * u) := by
        rw [show 3 * u = u + u + u by ring, Real.exp_add, Real.exp_add]
        ring
  have hRP := sum_allowance_le P (u * (B + 3 * k - 1)) hP
  have hQb := sum_exp_mul_exp_le block (fun e => ExecFloat.Binary.toReal e.1 - m')
    (fun e => u * (m' - ExecFloat.Binary.toReal e.1))
    (x := u * (block.length + 1) + u) (y := u * (B + 3 * k - 1) + 3 * u) (by
      have : (block.length : ℝ) ≤ B := by exact_mod_cast hB
      have : (0 : ℝ) ≤ k := Nat.cast_nonneg k
      nlinarith)
  rw [Real.exp_add, ← mul_assoc] at hQb
  have hrelative := normalizer_relative_le hu hr hρ hE hD.le hLnn hI
    (sum_map_nonneg block fun e => (mul_pos (Real.exp_pos _) (Real.exp_pos _)).le)
    (Real.exp_pos _).le hrel hρbound h3 hexpu
    (sum_allowance_shift P m m' (u * (B + 3 * k - 1)) (3 * u)) hQb
  have hn : P.length ≤ N := by
    rw [List.length_append] at hN
    omega
  have hcT : u * ((B : ℝ) + 3 * k - 1) ≤ u * (B + 3 * T - 1) := by
    have : (k : ℝ) + 1 ≤ T := by exact_mod_cast hk
    nlinarith
  have habsolute := normalizer_absolute_le hu hη hr ha hα hLnn hE hD.le hED hexpu
    (hrel.trans hRP) hαbound (step_residual_le hu hη zero_le_one hη hn hB hcT habs)
  exact ⟨trivial, by rw [hb], hrelative.1, hrelative.2, habsolute.1, habsolute.2⟩

/-- `NormalizerBound` holds before the first block. -/
theorem normalizerBound_nil (B N T : ℕ) (start : ℝ) :
    NormalizerBound u η B N T 0 start ([] : List (Value × Value)) (start, 0) (0, 0) := by
  simp [NormalizerBound]

/-- The budgets of `blocked_normalizer_error` stay within the allowance over any blocks. -/
theorem normalizerBound_foldl {B N T : ℕ} {start : ℝ} :
    ∀ (blocks : List (List (Value × Value))) {k : ℕ} {P : List (Value × Value)}
      {state : Value × Value × Value} {realState budgets : ℝ × ℝ},
      (∀ block ∈ blocks, block.length ≤ B) → k + blocks.length ≤ T →
      (P ++ blocks.flatten).length ≤ N →
      NormalizerBound u η B N T k start P realState budgets →
      NormalizerBound u η B N T (k + blocks.length) start (P ++ blocks.flatten)
        (blocks.foldl realNormalizerStep realState)
        (List.foldBudget (onlineStep (fun value => ExecFloat.Binary.CorrectlyRounded.exp value)
          id id) realNormalizerStep normalizerBudget state realState budgets blocks)
  | [], _, _, _, _, _, _, _, _, h => by simpa using h
  | block :: blocks, k, P, state, realState, budgets, hB, hk, hN, h => by
    simp only [List.length_cons, List.flatten_cons, ← List.append_assoc] at hk hN
    rw [List.foldBudget_cons, List.foldl_cons, List.flatten_cons, ← List.append_assoc,
      List.length_cons, show k + (blocks.length + 1) = k + 1 + blocks.length by omega]
    refine normalizerBound_foldl blocks (fun b hb => hB b (List.mem_cons_of_mem _ hb))
      (by omega) hN (normalizerBound_step (hB block (List.mem_cons_self ..)) (by omega) ?_ h)
    have := hN
    rw [List.length_append] at this
    omega

/-- The budgets that `blocked_normalizer_error` charges, the fold of `normalizerBudget`. -/
noncomputable def normalizerBudgets (blocks : List (List (Value × Value))) (start : Value) :
    ℝ × ℝ :=
  List.foldBudget (onlineStep (fun value => ExecFloat.Binary.CorrectlyRounded.exp value) id id)
    realNormalizerStep normalizerBudget (start, 0, 0) (ExecFloat.Binary.toReal start, 0) (0, 0)
    blocks

/-- The reference of `blocked_normalizer_error` is the normalizer `S` of `onlineAverage_error`,
and its budgets are at most that theorem's allowance `El` on `S`. -/
theorem normalizerBudgets_le (blocks : List (List (Value × Value))) (start : Value) {B : ℕ}
    (hB : ∀ block ∈ blocks, block.length ≤ B) :
    let N := blocks.flatten.length
    let T := blocks.length
    let M := blocks.flatten.foldl (fun m e => max m (ExecFloat.Binary.toReal e.1))
      (ExecFloat.Binary.toReal start)
    let S := (blocks.flatten.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - M)).sum
    let El := (blocks.flatten.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - M) *
        (Real.exp (u * (M - ExecFloat.Binary.toReal e.1) + u * (B + 3 * T - 1)) - 1)).sum +
      T * residualGrowth u η ^ T * blockResidual u η N B 1 (u * (B + 3 * T - 1)) η
    (blocks.foldl realNormalizerStep (ExecFloat.Binary.toReal start, 0)).2 = S ∧
      (normalizerBudgets blocks start).1 * |S| + (normalizerBudgets blocks start).2 ≤ El := by
  intro N T M S El
  obtain ⟨hm, hL, -, hrel, -, habs⟩ := normalizerBound_foldl (N := N) (T := T) blocks
    (k := 0) (P := []) (state := (start, 0, 0)) hB (by simp [T]) (by simp [N])
    (normalizerBound_nil B N T (ExecFloat.Binary.toReal start))
  simp only [zero_add, List.nil_append] at hm hL hrel habs
  rw [hm] at hL hrel
  have hS : 0 ≤ S := sum_map_nonneg _ fun e => (Real.exp_pos _).le
  refine ⟨hL, ?_⟩
  rw [hL] at hrel
  rw [abs_of_nonneg hS]
  have hsplit : El = (blocks.flatten.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - M) *
        Real.exp (u * (M - ExecFloat.Binary.toReal e.1) + u * (B + 3 * T - 1))).sum - S +
      T * residualGrowth u η ^ T * blockResidual u η N B 1 (u * (B + 3 * T - 1)) η := by
    simp only [El, S, mul_sub, mul_one]
    rw [List.sum_hom₂ _ (· - ·) (fun a b c d => add_sub_add_comm a b c d) (sub_zero (0 : ℝ))]
  rw [hsplit]
  unfold normalizerBudgets
  linarith

/-- Under the generated bound, the blocked normalizer is within the allowance `El` of
`onlineAverage_error` of the normalizer `S`, with `M` the maximum of `start` and the scores. -/
theorem blocked_normalizer_abs_sub_le (blocks : List (List (Value × Value))) (start : Value)
    {B : ℕ} (hB : ∀ block ∈ blocks, block.length ≤ B)
    (h : MixedError (ExecFloat.Binary.toReal (blockedNormalizer blocks start))
      (blocks.foldl realNormalizerStep (ExecFloat.Binary.toReal start, 0)).2
      (normalizerBudgets blocks start).1 (normalizerBudgets blocks start).2) :
    let N := blocks.flatten.length
    let T := blocks.length
    let M := blocks.flatten.foldl (fun m e => max m (ExecFloat.Binary.toReal e.1))
      (ExecFloat.Binary.toReal start)
    let S := (blocks.flatten.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - M)).sum
    let El := (blocks.flatten.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - M) *
        (Real.exp (u * (M - ExecFloat.Binary.toReal e.1) + u * (B + 3 * T - 1)) - 1)).sum +
      T * residualGrowth u η ^ T * blockResidual u η N B 1 (u * (B + 3 * T - 1)) η
    |ExecFloat.Binary.toReal (blockedNormalizer blocks start) - S| ≤ El := by
  intro N T M S El
  obtain ⟨hS, hle⟩ := normalizerBudgets_le blocks start hB
  have := h.bound
  rw [hS] at this
  exact this.trans hle

end Budgets

end TorchLean.OnlineSoftmax

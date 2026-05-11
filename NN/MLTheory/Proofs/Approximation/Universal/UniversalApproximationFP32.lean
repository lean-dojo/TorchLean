/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.FP32
public import NN.MLTheory.Proofs.Approximation.Universal.UniversalApproximation
import Mathlib.Tactic.Linarith

/-!
# Exact ReLU interpolation and FP32 error lifting

This file has two jobs.

First, it proves an exact interpolation lemma on a uniform grid: given arbitrary target values at
the grid points, we construct a two-layer ReLU MLP that matches those values exactly under the
ideal `ℝ` semantics.  This is the familiar hinge-basis construction behind one-dimensional
piecewise-linear approximation; see Pinkus for the approximation-theory background and Yarotsky
for quantitative ReLU-network rates.

Second, it states the FP32 lifting layer: once a real construction is fixed, explicit
rounding-error bounds turn it into an `FP32` approximation theorem.  This keeps the architecture
clean: real approximation, parameter rounding, and executable IEEE semantics are proved in
separate modules, matching the numerical-analysis separation used in Goldberg and Higham.
-/

@[expose] public section


namespace NN.MLTheory.Proofs.UniversalApproximation

open _root_.Spec
open _root_.Spec.Tensor
open Spec.Tensor
open Examples

noncomputable section

/--
Exact interpolation on a uniform grid by a 2-layer ReLU MLP (over `ℝ` semantics).

Given arbitrary target values `y₀,…,y_N` at the uniform grid points
`grid k = a + k * ((b-a)/N)`, this constructs a width-`N` hinge network that matches them at the
grid points.
-/
theorem relu_mlp_exact_on_uniform_grid {a b : ℝ} (h_ab : a < b) :
    ∀ {N : ℕ}, 0 < N → ∀ y : Fin (N + 1) → ℝ,
      ∃ (l1 : LinearSpec ℝ 1 N) (l2 : LinearSpec ℝ N 1),
        ∀ k : Fin (N + 1),
          mlpEval1d N l1 l2 (a + (k.1 : ℝ) * ((b - a) / (N : ℝ))) = y k := by
  intro N hN y
  classical
  have hba : 0 < b - a := sub_pos.mpr h_ab
  have hNpos : 0 < (N : ℝ) := by exact_mod_cast hN
  let δ : ℝ := (b - a) / (N : ℝ)
  have hδpos : 0 < δ := by
    dsimp [δ]
    exact div_pos hba hNpos
  have hδne : δ ≠ 0 := ne_of_gt hδpos
  have hδnonneg : 0 ≤ δ := le_of_lt hδpos
  let grid : ℕ → ℝ := fun k => a + (k : ℝ) * δ

  -- `yAt k` reads the `k`-th target value for `k ≤ N` and is `0` outside that range.
  let yAt : ℕ → ℝ := fun k =>
    if hk : k ≤ N then y ⟨k, Nat.lt_succ_of_le hk⟩ else 0
  let mNat : ℕ → ℝ := fun k => (yAt (k + 1) - yAt k) / δ
  let cNat : ℕ → ℝ
    | 0 => mNat 0
    | k + 1 => mNat (k + 1) - mNat k
  let g : ℝ → ℝ := fun x => yAt 0 + ∑ i ∈ Finset.range N, cNat i * relu (x - grid i)

  have yAt_0 : yAt 0 = y 0 := by
    -- `0 ≤ N` always holds.
    simp [yAt]

  have prefix_sum_cNat_eq_mNat : ∀ k : ℕ, (∑ i ∈ Finset.range (k + 1), cNat i) = mNat k := by
    intro k
    induction k with
    | zero =>
        simp [cNat, mNat]
    | succ k ih =>
        calc
          (∑ i ∈ Finset.range (k + 2), cNat i)
              = (∑ i ∈ Finset.range (k + 1), cNat i) + cNat (k + 1) := by
                  simpa using (Finset.sum_range_succ (f := fun i => cNat i) (n := k + 1))
          _ = mNat k + cNat (k + 1) := by simp [ih]
          _ = mNat k + (mNat (k + 1) - mNat k) := by simp [cNat]
          _ = mNat (k + 1) := by ring

  have grid_mono : Monotone grid := by
    intro m n hmn
    dsimp [grid]
    have : (m : ℝ) ≤ (n : ℝ) := by exact_mod_cast hmn
    -- `add_le_add_right` adds the same constant on the left: `a + _ ≤ a + _`.
    exact add_le_add_right (mul_le_mul_of_nonneg_right this hδnonneg) a

  have g_grid_eq_yAt : ∀ k : ℕ, k ≤ N → g (grid k) = yAt k := by
    intro k hk
    induction k with
    | zero =>
        -- At the left endpoint, all hinges vanish because `grid i ≥ grid 0 = a`.
        have ha_le : ∀ i ∈ Finset.range N, a ≤ grid i := by
          intro i hi
          have : grid 0 ≤ grid i := grid_mono (Nat.zero_le i)
          simpa [grid] using this
        have hsum :
            (∑ i ∈ Finset.range N, cNat i * relu (a - grid i)) = 0 := by
          refine Finset.sum_eq_zero ?_
          intro i hi
          have : a ≤ grid i := ha_le i hi
          simp [relu_sub_eq_zero_of_le (x := a) (t := grid i) this]
        have hgrid0 : grid 0 = a := by simp [grid]
        simp [g, hgrid0, hsum, yAt_0]
    | succ k ih =>
        have hk_le : k ≤ N := le_trans (Nat.le_succ k) hk
        have hsub : g (grid (k + 1)) - g (grid k) =
            ∑ i ∈ Finset.range N,
              cNat i * (relu (grid (k + 1) - grid i) - relu (grid k - grid i)) := by
          -- Expand both sides and cancel the shared `yAt 0` term.
          have hsum :
                (∑ i ∈ Finset.range N, cNat i * relu (grid (k + 1) - grid i)) -
                    (∑ i ∈ Finset.range N, cNat i * relu (grid k - grid i)) =
                  ∑ i ∈ Finset.range N,
                    (cNat i * relu (grid (k + 1) - grid i) - cNat i * relu (grid k - grid i)) := by
              exact
                (Finset.sum_sub_distrib (s := Finset.range N)
                  (f := fun i => cNat i * relu (grid (k + 1) - grid i))
                  (g := fun i => cNat i * relu (grid k - grid i))).symm
          calc
            g (grid (k + 1)) - g (grid k)
                =
                  (yAt 0 + ∑ i ∈ Finset.range N, cNat i * relu (grid (k + 1) - grid i)) -
                    (yAt 0 + ∑ i ∈ Finset.range N, cNat i * relu (grid k - grid i)) := by
                      rfl
            _ =
                  (∑ i ∈ Finset.range N, cNat i * relu (grid (k + 1) - grid i)) -
                    (∑ i ∈ Finset.range N, cNat i * relu (grid k - grid i)) := by
                      ring
            _ =
                  ∑ i ∈ Finset.range N,
                    (cNat i * relu (grid (k + 1) - grid i) - cNat i * relu (grid k - grid i)) := by
                      exact hsum
            _ =
                  ∑ i ∈ Finset.range N,
                    cNat i * (relu (grid (k + 1) - grid i) - relu (grid k - grid i)) := by
                      apply Finset.sum_congr rfl
                      intro i hi
                      simpa using
                        (mul_sub (cNat i) (relu (grid (k + 1) - grid i)) (relu (grid k - grid
                          i))).symm
        -- Reduce the RHS sum to indices `i < k+1` (others contribute `0`).
        let F : ℕ → ℝ := fun i =>
          cNat i * (relu (grid (k + 1) - grid i) - relu (grid k - grid i))
        have hsubRange : Finset.range (k + 1) ⊆ Finset.range N := by
          intro i hi
          have hi' : i < k + 1 := Finset.mem_range.mp hi
          have : i < N := lt_of_lt_of_le hi' hk
          exact Finset.mem_range.mpr this
        have hFzero : ∀ i ∈ Finset.range N, i ∉ Finset.range (k + 1) → F i = 0 := by
          intro i hiN hik
          have hik' : k + 1 ≤ i := by
            have : ¬ i < k + 1 := fun hlt => hik (Finset.mem_range.mpr hlt)
            exact Nat.le_of_not_gt this
          have hgi1 : grid (k + 1) ≤ grid i := grid_mono hik'
          have hgi0 : grid k ≤ grid i := by
            have : k ≤ i := le_trans (Nat.le_succ k) hik'
            exact grid_mono this
          simp [F, relu_sub_eq_zero_of_le (x := grid (k + 1)) (t := grid i) hgi1,
            relu_sub_eq_zero_of_le (x := grid k) (t := grid i) hgi0]
        have sumF :
            (∑ i ∈ Finset.range N, F i) = (∑ i ∈ Finset.range (k + 1), F i) := by
          symm
          exact Finset.sum_subset hsubRange hFzero
        have hFconst : ∀ i ∈ Finset.range (k + 1), F i = cNat i * δ := by
          intro i hi
          have hi_le : i ≤ k := Nat.le_of_lt_succ (Finset.mem_range.mp hi)
          have hgi : grid i ≤ grid k := grid_mono hi_le
          have hgi1 : grid i ≤ grid (k + 1) := by
            have : i ≤ k + 1 := le_trans hi_le (Nat.le_succ k)
            exact grid_mono this
          have hrelu1 : relu (grid (k + 1) - grid i) = grid (k + 1) - grid i := by
            exact relu_sub_eq_of_le (x := grid (k + 1)) (t := grid i) hgi1
          have hrelu0 : relu (grid k - grid i) = grid k - grid i := by
            exact relu_sub_eq_of_le (x := grid k) (t := grid i) hgi
          have hstep : grid (k + 1) - grid k = δ := by
            dsimp [grid]
            simp [Nat.cast_add, Nat.cast_one, add_assoc, add_comm, mul_add, sub_eq_add_neg,
              mul_comm]
          have hdiff : relu (grid (k + 1) - grid i) - relu (grid k - grid i) = δ := by
            calc
              relu (grid (k + 1) - grid i) - relu (grid k - grid i)
                  = (grid (k + 1) - grid i) - (grid k - grid i) := by simp [hrelu1, hrelu0]
              _ = grid (k + 1) - grid k := by ring
              _ = δ := hstep
          simp [F, hdiff]
        have sumF' :
            (∑ i ∈ Finset.range (k + 1), F i) = (∑ i ∈ Finset.range (k + 1), cNat i * δ) := by
          exact Finset.sum_congr rfl hFconst
        have hdiff : g (grid (k + 1)) - g (grid k) = mNat k * δ := by
          calc
            g (grid (k + 1)) - g (grid k)
                = ∑ i ∈ Finset.range N, F i := by simpa [F] using hsub
            _ = ∑ i ∈ Finset.range (k + 1), F i := by simp [sumF]
            _ = ∑ i ∈ Finset.range (k + 1), cNat i * δ := sumF'
            _ = (∑ i ∈ Finset.range (k + 1), cNat i) * δ := by
                  simpa using
                    (Finset.sum_mul (s := Finset.range (k + 1))
                      (f := fun i => cNat i) (a := δ)).symm
            _ = mNat k * δ := by simp [prefix_sum_cNat_eq_mNat k]
        have hm : mNat k * δ = yAt (k + 1) - yAt k := by
          dsimp [mNat]
          field_simp [hδne]
        -- Finish by induction, using `g(grid(k+1)) = g(grid k) + (yAt(k+1)-yAt k)`.
        have hkval : g (grid k) = yAt k := ih hk_le
        calc
          g (grid (k + 1))
              = g (grid k) + mNat k * δ := by linarith [hdiff]
          _ = yAt k + (yAt (k + 1) - yAt k) := by simp [hkval, hm]
          _ = yAt (k + 1) := by ring

  -- Package as an MLP via the hinge construction from `universal_approximation.lean`.
  let t : Fin N → ℝ := fun i => grid i.1
  let c : Fin N → ℝ := fun i => cNat i.1
  refine ⟨hingeLayer1 N t, hingeLayer2 N c (yAt 0), ?_⟩
  intro k
  have hnet :
      mlpEval1d N (hingeLayer1 N t) (hingeLayer2 N c (yAt 0)) (grid k.1) =
        hingeFun N t c (yAt 0) (grid k.1) := by
    simpa using (mlp_eval_1d_hinge N t c (yAt 0) (grid k.1))
  have hhinge : hingeFun N t c (yAt 0) (grid k.1) = g (grid k.1) := by
    -- Convert the `Fin` sum in `hinge_fun` to a `Finset.range N` sum.
    unfold hingeFun g t c
    congr 1
    simpa using
      (Fin.sum_univ_eq_sum_range
        (f := fun i : ℕ => cNat i * relu (grid k.1 - grid i)) (n := N))
  have hk_leN : k.1 ≤ N := Nat.le_of_lt_succ k.2
  have hyAt : yAt k.1 = y k := by
    simp [yAt, hk_leN]
  calc
    mlpEval1d N (hingeLayer1 N t) (hingeLayer2 N c (yAt 0)) (grid k.1)
        = hingeFun N t c (yAt 0) (grid k.1) := hnet
    _ = g (grid k.1) := hhinge
    _ = yAt k.1 := g_grid_eq_yAt k.1 hk_leN
    _ = y k := hyAt

end

/-!
## FP32 (rounding-on-ℝ) error propagation for the hinge construction

This section does *not* claim that FP32 is equivalent to hardware float32.
It proves a pointwise bound for evaluating the same hinge network with rounded operations
(`TorchLean.Floats.FP32`) versus exact real arithmetic on the rounded inputs/weights.
-/

noncomputable section

open TorchLean.Floats

/-- Alias for the “round-after-each-primitive” real-valued FP32 model (`TorchLean.Floats.FP32`). -/
abbrev FP32 : Type := TorchLean.Floats.FP32

/-- Extract scalar from a length-1 tensor (FP32). -/
def extractScalarOutputFp32 (t : Tensor FP32 (.dim 1 .scalar)) : FP32 :=
  match t with
  | .dim f => toScalar (f ⟨0, by norm_num⟩)

/-- Evaluate a 2-layer ReLU MLP on a scalar FP32 input (returns an FP32 scalar). -/
noncomputable def mlpEval1dFp32 (hidDim : ℕ)
    (l1 : LinearSpec FP32 1 hidDim) (l2 : LinearSpec FP32 hidDim 1) (x : FP32) : FP32 :=
  extractScalarOutputFp32 (Examples.mlpForward l1 l2 (Tensor.singleton x))

/-- Scalar ReLU on `FP32`. (No rounding: it is `max x 0` at the `FP32` level.) -/
@[inline] def reluFp32 (x : FP32) : FP32 := Activation.Math.reluSpec x

@[simp] lemma fp32_zero_val : (0 : FP32).val = 0 := by
  -- `0 : FP32` is `NF.ofReal 0`, i.e. rounding `0`.
  have hne : TorchLean.Floats.neuralNearestEven (0 : ℝ) = 0 := by
    simp [TorchLean.Floats.neuralNearestEven]
  change
    (TorchLean.Floats.NF.ofReal (β := TorchLean.Floats.binaryRadix) (fexp :=
      TorchLean.Floats.fexp32)
          (rnd := TorchLean.Floats.rnd32) (0 : ℝ)).val = 0
  -- Reduce to the mantissa being `0`; the exponent is irrelevant.
  simp [TorchLean.Floats.NF.ofReal, TorchLean.Floats.NF.roundR, TorchLean.Floats.neuralRound,
    TorchLean.Floats.neuralToReal, TorchLean.Floats.neuralScaledMantissa,
      TorchLean.Floats.neuralCexp,
    TorchLean.Floats.neuralMagnitude, TorchLean.Floats.rnd32, hne]

@[simp] lemma relu_fp32_val (x : FP32) : (reluFp32 x).val = relu x.val := by
  -- `reluFp32` is `max x 0` at the `FP32` level, and order is via `.val`.
  by_cases hx0 : 0 ≤ x.val
  · have hx0' : (0 : FP32) ≤ x := by
      change (0 : FP32).val ≤ x.val
      simpa [fp32_zero_val] using hx0
    -- Reduce to `x.val = x.val ⊔ 0`.
    simpa [reluFp32, relu, Activation.Math.reluSpec, Max.max, hx0', fp32_zero_val] using
      (sup_of_le_left (a := x.val) (b := (0 : ℝ)) hx0).symm
  · have hx0' : ¬(0 : FP32) ≤ x := by
      change ¬((0 : FP32).val ≤ x.val)
      simpa [fp32_zero_val] using hx0
    have hxle : x.val ≤ 0 := le_of_lt (lt_of_not_ge hx0)
    -- Reduce to `0 = x.val ⊔ 0`.
    simpa [reluFp32, relu, Activation.Math.reluSpec, Max.max, hx0', fp32_zero_val] using
      (sup_of_le_right (a := x.val) (b := (0 : ℝ)) hxle).symm

/--
Real ReLU is 1-Lipschitz.

This elementary analytic fact is what lets the FP32 error analysis pass a subtraction-rounding
error through the ReLU nonlinearity without amplifying it.
-/
lemma relu_lipschitz (u v : ℝ) : |relu u - relu v| ≤ |u - v| := by
  by_cases hu : 0 ≤ u <;> by_cases hv : 0 ≤ v
  · -- u ≥ 0, v ≥ 0
    simp [relu, Activation.Math.reluSpec, max_eq_left hu, max_eq_left hv]
  · -- u ≥ 0, v < 0
    have hv' : v ≤ 0 := le_of_not_ge hv
    have huv : 0 ≤ u - v := by linarith
    have hu_le : u ≤ u - v := by linarith
    -- `relu u = u`, `relu v = 0`, and `|u - v| = u - v`.
    simp [relu, Activation.Math.reluSpec, max_eq_left hu, max_eq_right hv',
      abs_of_nonneg hu, abs_of_nonneg huv, hu_le]
  · -- u < 0, v ≥ 0
    have hu' : u ≤ 0 := le_of_not_ge hu
    have huv : u - v ≤ 0 := by linarith
    have hvu : v ≤ v - u := by linarith
    -- `relu u = 0`, `relu v = v`, and `|u - v| = v - u`.
    simp [relu, Activation.Math.reluSpec, max_eq_right hu', max_eq_left hv,
      abs_of_nonneg hv, abs_of_nonpos huv, hvu]
  · -- u < 0, v < 0
    have hu' : u ≤ 0 := le_of_not_ge hu
    have hv' : v ≤ 0 := le_of_not_ge hv
    simp [relu, Activation.Math.reluSpec, max_eq_right hu', max_eq_right hv']

/-! ### FP32 hinge layers and a pointwise error bound -/

/-- First FP32 hinge layer: hidden unit `i` computes `x - tᵢ` before ReLU. -/
  noncomputable def hingeLayer1Fp32 (n : ℕ) (t : Fin n → FP32) : LinearSpec FP32 1 n :=
  { weights := matrixMN n 1 (fun _ _ => (1 : FP32))
    bias := vectorN n (fun i => -t i) }

/-- Second FP32 hinge layer: sum hidden activations with coefficients `cᵢ` and bias `b`. -/
  noncomputable def hingeLayer2Fp32 (n : ℕ) (c : Fin n → FP32) (b : FP32) : LinearSpec FP32 n 1 :=
  { weights := matrixMN 1 n (fun _ j => c j)
    bias := vectorN 1 (fun _ => b) }

/-- One rounded hinge term `cᵢ * reluFp32 (x - tᵢ)` in the FP32 model. -/
  noncomputable def hingeTermFp32 {n : ℕ} (c t : Fin n → FP32) (x : FP32) (i : Fin n) : FP32 :=
  c i * reluFp32 (x - t i)

/-- Real reference for the same hinge term, using the `.val` denotation of FP32 parameters. -/
  noncomputable def hingeTermReal {n : ℕ} (c t : Fin n → FP32) (x : FP32) (i : Fin n) : ℝ :=
  (c i).val * relu (x.val - (t i).val)

/--
Per-neuron FP32 hinge-term error bound.

The bound has two pieces: one half-ulp term for the final multiplication and one subtraction
rounding term propagated through the 1-Lipschitz ReLU and scaled by `|cᵢ|`.
-/
  lemma hinge_term_abs_error {n : ℕ} (c t : Fin n → FP32) (x : FP32) (i : Fin n) :
    let term32 := hingeTermFp32 c t x i
    let termR := hingeTermReal c t x i
    |term32.val - termR| ≤
      neuralUlp binaryRadix fexp32 ((c i).val * (reluFp32 (x - t i)).val) TrainingPhase.forward /
        2
      + |(c i).val| *
          (neuralUlp binaryRadix fexp32 (x.val - (t i).val) TrainingPhase.forward / 2) := by
  intro term32 termR
  -- Notation for the intermediate subtraction and ReLU.
  let u32 : FP32 := x - t i
  let r32 : FP32 := reluFp32 u32
  have hsub :
      |u32.val - (x.val - (t i).val)| ≤
        neuralUlp binaryRadix fexp32 (x.val - (t i).val) TrainingPhase.forward / 2 := by
    -- `sub_abs_error` is exactly this statement.
    simpa [u32] using (TorchLean.Floats.FP32.sub_abs_error (a := x) (b := t i))
  have hrelu :
      |r32.val - relu (x.val - (t i).val)| ≤
        |u32.val - (x.val - (t i).val)| := by
    -- `reluFp32` is exact on `.val`, so this is just the Lipschitz property of real ReLU.
    simpa [r32, u32] using (relu_lipschitz u32.val (x.val - (t i).val))
  have hmul :
      |term32.val - ((c i).val * r32.val)| ≤
        neuralUlp binaryRadix fexp32 ((c i).val * r32.val) TrainingPhase.forward / 2 := by
    -- `mul_abs_error` compares `(a*b).val` to `a.val*b.val`.
    have := TorchLean.Floats.FP32.mul_abs_error (a := c i) (b := r32)
    simpa [term32, hingeTermFp32, r32] using this
  have hlin :
      |(c i).val * r32.val - (c i).val * relu (x.val - (t i).val)| =
        |(c i).val| * |r32.val - relu (x.val - (t i).val)| := by
    have hfactor :
        (c i).val * r32.val - (c i).val * relu (x.val - (t i).val) =
          (c i).val * (r32.val - relu (x.val - (t i).val)) := by ring
    simp [hfactor, abs_mul]
  calc
    |term32.val - termR|
        ≤ |term32.val - ((c i).val * r32.val)| +
            |(c i).val * r32.val - (c i).val * relu (x.val - (t i).val)| := by
              -- use `abs_sub_le` directly
              simpa [termR] using (abs_sub_le (term32.val) ((c i).val * r32.val) termR)
    _ ≤ neuralUlp binaryRadix fexp32 ((c i).val * r32.val) TrainingPhase.forward / 2 +
          (|(c i).val| * |r32.val - relu (x.val - (t i).val)|) := by
          -- Apply the two bounds.
          have hmul' := hmul
          have hlin' : |(c i).val * r32.val - (c i).val * relu (x.val - (t i).val)| ≤
              |(c i).val| * |r32.val - relu (x.val - (t i).val)| := by
            simp [hlin]
          exact add_le_add hmul' hlin'
    _ ≤ neuralUlp binaryRadix fexp32 ((c i).val * r32.val) TrainingPhase.forward / 2 +
          (|(c i).val| * |u32.val - (x.val - (t i).val)|) := by
          gcongr
    _ ≤ neuralUlp binaryRadix fexp32 ((c i).val * r32.val) TrainingPhase.forward / 2 +
          (|(c i).val| * (neuralUlp binaryRadix fexp32 (x.val - (t i).val) TrainingPhase.forward /
            2)) := by
          gcongr
    _ = neuralUlp binaryRadix fexp32 ((c i).val * (reluFp32 (x - t i)).val) TrainingPhase.forward
      / 2
        + |(c i).val| * (neuralUlp binaryRadix fexp32 (x.val - (t i).val) TrainingPhase.forward /
          2) := by
          simp [u32, r32]

/-- The per-hinge error bound used by `hinge_term_abs_error`. -/
noncomputable def hingeTermErrorBound {n : ℕ} (c t : Fin n → FP32) (x : FP32) (i : Fin n) : ℝ :=
  neuralUlp binaryRadix fexp32 ((c i).val * (reluFp32 (x - t i)).val) TrainingPhase.forward / 2
  + |(c i).val| *
      (neuralUlp binaryRadix fexp32 (x.val - (t i).val) TrainingPhase.forward / 2)

/-- Named version of `hinge_term_abs_error` using `hingeTermErrorBound`. -/
  @[simp] lemma hinge_term_abs_error' {n : ℕ} (c t : Fin n → FP32) (x : FP32) (i : Fin n) :
    |(hingeTermFp32 c t x i).val - hingeTermReal c t x i| ≤ hingeTermErrorBound c t x i := by
  simpa [hingeTermErrorBound] using (hinge_term_abs_error (c := c) (t := t) (x := x) (i := i))

/-! ### Summation error propagation (FP32 hinge network) -/

/--
Fold state for summing hinge terms in `FP32`, while tracking:
- a real reference sum (computed from `.val`),
- and a provable error bound on the difference between them.
-/
abbrev HingeSumState : Type := FP32 × ℝ × ℝ

/-- One summation step: add a hinge term, and accumulate rounding+term error bounds. -/
noncomputable def hingeSumStateStep {n : ℕ} (c t : Fin n → FP32) (x : FP32) :
    HingeSumState → Fin n → HingeSumState
  | (acc32, accR, err), i =>
      let term32 : FP32 := hingeTermFp32 c t x i
      let termR : ℝ := hingeTermReal c t x i
      let termErr : ℝ := hingeTermErrorBound c t x i
      let addErr : ℝ :=
        neuralUlp binaryRadix fexp32 (acc32.val + term32.val) TrainingPhase.forward / 2
      (acc32 + term32, accR + termR, err + termErr + addErr)

/-- Compute the hinge-term sum state over all `Fin n` in a fixed order (`List.finRange`). -/
noncomputable def hingeSumState {n : ℕ} (c t : Fin n → FP32) (x : FP32) : HingeSumState :=
  (List.finRange n).foldl (hingeSumStateStep c t x) (0, 0, 0)

/-- FP32 value produced by folding all hinge terms in the fixed `List.finRange` order. -/
  noncomputable def hingeSumFp32 {n : ℕ} (c t : Fin n → FP32) (x : FP32) : FP32 :=
  (hingeSumState c t x).1

/-- Real reference sum accumulated alongside `hingeSumFp32`. -/
  noncomputable def hingeSumReal {n : ℕ} (c t : Fin n → FP32) (x : FP32) : ℝ :=
  (hingeSumState c t x).2.1

/-- Accumulated certified absolute-error budget for `hingeSumFp32`. -/
  noncomputable def hingeSumErrorBound {n : ℕ} (c t : Fin n → FP32) (x : FP32) : ℝ :=
  (hingeSumState c t x).2.2

/--
Fold invariant for FP32 hinge summation.

At every prefix of the fold, the rounded accumulator is within the tracked error budget of the
real accumulator.  The proof is deliberately order-sensitive because floating-point addition is
not associative.
-/
  lemma hinge_sum_state_invariant_aux {n : ℕ} (c t : Fin n → FP32) (x : FP32) :
    ∀ (xs : List (Fin n)) (acc32 : FP32) (accR err : ℝ),
      |acc32.val - accR| ≤ err →
      let st := xs.foldl (hingeSumStateStep c t x) (acc32, accR, err)
      |st.1.val - st.2.1| ≤ st.2.2 := by
  intro xs
  induction xs with
  | nil =>
      intro acc32 accR err herr
      simp [List.foldl, herr]
  | cons i xs ih =>
      intro acc32 accR err herr
      -- Abbreviations for this step.
      let term32 : FP32 := hingeTermFp32 c t x i
      let termR : ℝ := hingeTermReal c t x i
      let termErr : ℝ := hingeTermErrorBound c t x i
      let addErr : ℝ :=
        neuralUlp binaryRadix fexp32 (acc32.val + term32.val) TrainingPhase.forward / 2
      have hterm : |term32.val - termR| ≤ termErr := by
        dsimp [term32, termR, termErr]
        exact hinge_term_abs_error' (c := c) (t := t) (x := x) (i := i)
      have hadd :
          |(acc32 + term32).val - (acc32.val + term32.val)| ≤ addErr := by
        simpa [addErr] using (TorchLean.Floats.FP32.add_abs_error (a := acc32) (b := term32))
      have hstep :
          |(acc32 + term32).val - (accR + termR)| ≤ err + termErr + addErr := by
        -- Triangle: rounding of addition + linearization error.
        have htri :
            |(acc32 + term32).val - (accR + termR)| ≤
              |(acc32 + term32).val - (acc32.val + term32.val)| +
              |(acc32.val + term32.val) - (accR + termR)| := by
          simpa using
            (abs_sub_le (a := (acc32 + term32).val) (b := acc32.val + term32.val) (c := accR +
              termR))
        have hlin :
            |(acc32.val + term32.val) - (accR + termR)| ≤
              |acc32.val - accR| + |term32.val - termR| := by
          have hdecomp :
              (acc32.val + term32.val) - (accR + termR) = (acc32.val - accR) + (term32.val - termR)
                := by
            ring
          simpa [hdecomp] using (abs_add_le (acc32.val - accR) (term32.val - termR))
        have hlin' : |(acc32.val + term32.val) - (accR + termR)| ≤ err + termErr := by
          have : |acc32.val - accR| + |term32.val - termR| ≤ err + termErr :=
            add_le_add herr hterm
          exact le_trans hlin this
        have hsum :
            |(acc32 + term32).val - (accR + termR)| ≤ addErr + (err + termErr) := by
          exact le_trans htri (add_le_add hadd hlin')
        -- Reassociate to match the invariant's bound.
        linarith
      -- Apply IH on the tail, starting from the updated state.
      have := ih (acc32 + term32) (accR + termR) (err + termErr + addErr) hstep
      simpa [List.foldl, hingeSumStateStep, term32, termR, termErr, addErr] using this

/-- Certified absolute-error bound for the whole FP32 hinge-term sum. -/
  lemma hinge_sum_abs_error {n : ℕ} (c t : Fin n → FP32) (x : FP32) :
    |(hingeSumFp32 c t x).val - hingeSumReal c t x| ≤ hingeSumErrorBound c t x := by
  -- Instantiate the generic invariant with `xs = finRange` and the zero initial state.
  have h0 : |(0 : FP32).val - (0 : ℝ)| ≤ (0 : ℝ) := by simp
  have h :=
    hinge_sum_state_invariant_aux (c := c) (t := t) (x := x)
      (xs := List.finRange n) (acc32 := (0 : FP32)) (accR := (0 : ℝ)) (err := (0 : ℝ)) h0
  simpa [hingeSumFp32, hingeSumReal, hingeSumErrorBound, hingeSumState] using h

/-- FP32 hinge-network output: sum of hinge terms, then add the bias. -/
noncomputable def hingeFunFp32 {n : ℕ} (t c : Fin n → FP32) (b x : FP32) : FP32 :=
  hingeSumFp32 c t x + b

/-- Real reference for `hinge_fun_fp32`: evaluate over `ℝ` on the `.val` parameters/inputs. -/
noncomputable def hingeFunReal {n : ℕ} (t c : Fin n → FP32) (b : FP32) (x : FP32) : ℝ :=
  hingeSumReal c t x + b.val

/-- Total FP32 hinge-network error budget, including the final rounded bias addition. -/
  noncomputable def hingeFunErrorBound {n : ℕ} (t c : Fin n → FP32) (b : FP32) (x : FP32) : ℝ :=
  hingeSumErrorBound c t x
    + neuralUlp binaryRadix fexp32 ((hingeSumFp32 c t x).val + b.val) TrainingPhase.forward / 2

/--
Certified absolute-error bound for the complete FP32 hinge network.

This composes the fold invariant with the final rounded `+ b`, giving the main FP32 rounding term
used by the executable approximation theorems.
-/
  lemma hinge_fun_abs_error {n : ℕ} (t c : Fin n → FP32) (b x : FP32) :
    |(hingeFunFp32 t c b x).val - hingeFunReal t c b x| ≤ hingeFunErrorBound t c b x := by
  -- First use the hinge-sum bound, then account for the final rounded `+ b`.
  have hsum : |(hingeSumFp32 c t x).val - hingeSumReal c t x| ≤ hingeSumErrorBound c t x :=
    hinge_sum_abs_error (c := c) (t := t) (x := x)
  set s32 : FP32 := hingeSumFp32 c t x
  set sR : ℝ := hingeSumReal c t x
  set eS : ℝ := hingeSumErrorBound c t x
  have hadd :
      |(s32 + b).val - (s32.val + b.val)| ≤
        neuralUlp binaryRadix fexp32 (s32.val + b.val) TrainingPhase.forward / 2 := by
    simpa using (TorchLean.Floats.FP32.add_abs_error (a := s32) (b := b))
  have htri :
      |(s32 + b).val - (sR + b.val)| ≤
        |(s32 + b).val - (s32.val + b.val)| + |(s32.val + b.val) - (sR + b.val)| := by
    simpa using (abs_sub_le (a := (s32 + b).val) (b := s32.val + b.val) (c := sR + b.val))
  have hcancel : |(s32.val + b.val) - (sR + b.val)| = |s32.val - sR| := by ring_nf
  -- Combine and finish.
  have hbound :
      |(s32 + b).val - (sR + b.val)| ≤
        (neuralUlp binaryRadix fexp32 (s32.val + b.val) TrainingPhase.forward / 2) + eS := by
    have : |(s32 + b).val - (sR + b.val)| ≤
        (neuralUlp binaryRadix fexp32 (s32.val + b.val) TrainingPhase.forward / 2) + |s32.val -
          sR| := by
      simpa [hcancel] using le_trans htri (add_le_add hadd (le_rfl))
    exact le_trans this (by gcongr)
  -- Unfold the goal's definitions.
  simpa [hingeFunFp32, hingeFunReal, hingeFunErrorBound, s32, sR, eS, add_assoc,
    add_left_comm, add_comm] using
    hbound

/-! ### Real approximation + rounding combination (pointwise) -/

/--
Triangle bound combining real approximation error with FP32 rounding error.

The theorem is pointwise: it does not construct the approximating hinge parameters, it simply says
that once a real hinge network is close to `f`, the rounded FP32 network is close up to the
certified rounding budget.
-/
  lemma hinge_fun_total_abs_error_le {n : ℕ} (f : ℝ → ℝ) (t c : Fin n → FP32) (b x : FP32) :
    |f x.val - (hingeFunFp32 t c b x).val| ≤
      |f x.val - hingeFunReal t c b x| + hingeFunErrorBound t c b x := by
  have hround : |hingeFunReal t c b x - (hingeFunFp32 t c b x).val| ≤ hingeFunErrorBound t c
    b x := by
    simpa [abs_sub_comm] using (hinge_fun_abs_error (t := t) (c := c) (b := b) (x := x))
  have htri :
      |f x.val - (hingeFunFp32 t c b x).val| ≤
        |f x.val - hingeFunReal t c b x| +
          |hingeFunReal t c b x - (hingeFunFp32 t c b x).val| := by
    simpa using
      (abs_sub_le (a := f x.val) (b := hingeFunReal t c b x) (c := (hingeFunFp32 t c b x).val))
  -- Add the same `|f - hinge_real|` term on the left of the rounding bound.
  exact le_trans htri (add_le_add_right hround _)

/-- Strict version of `hinge_fun_total_abs_error_le` for use with `< ε` approximation statements. -/
  lemma hinge_fun_total_abs_error_lt {n : ℕ} (f : ℝ → ℝ) (t c : Fin n → FP32) (b x : FP32)
    {ε : ℝ} (hε : |f x.val - hingeFunReal t c b x| < ε) :
    |f x.val - (hingeFunFp32 t c b x).val| < ε + hingeFunErrorBound t c b x := by
  have hle :=
    hinge_fun_total_abs_error_le (f := f) (t := t) (c := c) (b := b) (x := x)
  have hlt : |f x.val - hingeFunReal t c b x| + hingeFunErrorBound t c b x < ε +
    hingeFunErrorBound t c b x :=
    add_lt_add_of_lt_of_le hε (le_rfl)
  exact lt_of_le_of_lt hle hlt

/-- The real reference accumulator is the ordinary finite sum of real hinge terms. -/
  lemma hinge_sum_real_eq_sum {n : ℕ} (c t : Fin n → FP32) (x : FP32) :
    hingeSumReal c t x = ∑ i : Fin n, hingeTermReal c t x i := by
  classical
  -- First, show `hinge_sum_real` is the plain `foldl` sum of `hinge_term_real`.
  have hfold :
      hingeSumReal c t x =
        (List.finRange n).foldl (fun acc i => acc + hingeTermReal c t x i) 0 := by
    -- Peel off the irrelevant components of the fold state by induction over the list.
    have :
        ∀ (xs : List (Fin n)) (acc32 : FP32) (accR err : ℝ),
          (xs.foldl (hingeSumStateStep c t x) (acc32, accR, err)).2.1 =
            xs.foldl (fun acc i => acc + hingeTermReal c t x i) accR := by
      intro xs
      induction xs with
      | nil =>
          intro acc32 accR err
          simp [List.foldl]
      | cons i xs ih =>
          intro acc32 accR err
          simp [List.foldl, hingeSumStateStep, ih]
    simpa [hingeSumReal, hingeSumState] using
      (this (xs := List.finRange n) (acc32 := (0 : FP32)) (accR := (0 : ℝ)) (err := (0 : ℝ)))
  -- Then use the existing `finRange_foldl_add` lemma.
  simpa [hfold] using (finRange_foldl_add n (fun i : Fin n => hingeTermReal c t x i))

/--
1D ReLU approximation with a pointwise FP32 rounding bound.

This combines:
- the constructive real-valued hinge-network approximation theorem
  (`relu_universal_approximation_Icc_hinge`), and
- the FP32 hinge-network rounding bound (`hinge_fun_total_abs_error_lt`).

The output network is evaluated on the `FP32` model (`NF` rounding-on-`ℝ`); the additional
rounding error is given by `hinge_fun_error_bound`.
-/
theorem relu_universal_approximation_Icc_fp32 {f : ℝ → ℝ} {a b L : ℝ}
    (h_ab : a < b) (hL : 0 < L)
    (h_lip : ∀ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, |f x - f y| ≤ L * |x - y|) :
    ∀ ε > 0,
      ∃ (hidDim : ℕ) (t : Fin hidDim → FP32) (c : Fin hidDim → FP32) (b0 : FP32),
        ∀ x ∈ Set.Icc a b,
          |f x - (hingeFunFp32 t c b0 (⟨x⟩ : FP32)).val| <
            ε + hingeFunErrorBound t c b0 (⟨x⟩ : FP32) := by
  intro ε hε
  classical
  rcases
      relu_universal_approximation_Icc_hinge (f := f) (a := a) (b := b) (L := L)
        h_ab hL h_lip ε hε with
    ⟨hidDim, tR, cR, happx⟩
  let t : Fin hidDim → FP32 := fun i => ⟨tR i⟩
  let c : Fin hidDim → FP32 := fun i => ⟨cR i⟩
  let b0 : FP32 := ⟨f a⟩
  refine ⟨hidDim, t, c, b0, ?_⟩
  intro x hx
  let x32 : FP32 := ⟨x⟩
  have hreal :
      hingeFunReal t c b0 x32 = hingeFun hidDim tR cR (f a) x := by
    -- Expand `hinge_fun_real` as a `Fin` sum; then it matches `hinge_fun` by commutativity.
    have hsum :
        hingeSumReal c t x32 = ∑ i : Fin hidDim, cR i * relu (x - tR i) := by
      -- `hinge_sum_real` is a sum of `hinge_term_real`; `.val` of `⟨r⟩` is `r`.
      simpa [hingeTermReal, t, c, x32] using (hinge_sum_real_eq_sum (c := c) (t := t) (x := x32))
    -- Now rewrite both sides into the same shape.
    simp [hingeFunReal, hingeFun, hsum, b0, add_comm, x32]
  have happx' : |f x32.val - hingeFunReal t c b0 x32| < ε := by
    -- From the real-valued hinge approximation bound.
    have := happx x hx
    simpa [x32, hreal] using this
  -- Combine approximation error with the FP32 rounding bound.
  simpa [x32] using
    (hinge_fun_total_abs_error_lt (f := f) (t := t) (c := c) (b := b0) (x := x32) (ε := ε) happx')

end

end NN.MLTheory.Proofs.UniversalApproximation

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Floats.Formats.BinaryInterchange.Analysis.Error
public import NN.MLTheory.Proofs.Approximation.Universal.UniversalApproximation
public import NN.Spec.Core.FloatInstances.NF

/-!
# Exact ReLU interpolation and rounded-real approximation

The real interpolation theorem constructs a two-layer ReLU MLP matching arbitrary values on a
uniform grid. `RoundedReLUApprox` then bounds each rounded hinge term, its ordered sum, and the
final bias addition for an explicit `FloatFormat`, using FloatLib's half-ULP error theorem.

The scalar carrier is FloatLib's `NF`: its stored real value need not be representable, while
arithmetic operations round to the selected precision and gradual-underflow grid. There is no
upper exponent bound. The approximation theorem embeds exact real parameters and inputs in this
carrier; parameter quantization and finite executable arithmetic require separate bounds.
-/

@[expose] public section

open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq

namespace NN.MLTheory.Proofs.UniversalApproximation

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor
open Examples

noncomputable section

/--
Exact interpolation on a uniform grid by a two-layer ReLU MLP over $\mathbb{R}$ semantics.

Given arbitrary target values $y_0,\ldots,y_N$ at the uniform grid points
$\operatorname{grid}(k)=a+k(b-a)/N$, this constructs a width-$N$ hinge network that matches them
at the grid points.
-/
theorem relu_mlp_exact_on_uniform_grid {a b : ℝ} (h_ab : a < b) :
    ∀ {N : ℕ}, 0 < N → ∀ y : Fin (N + 1) → ℝ,
      ∃ (l1 : LinearSpec ℝ 1 N) (l2 : LinearSpec ℝ N 1),
        ∀ k : Fin (N + 1),
          mlpEvalScalar N l1 l2 (a + (k.1 : ℝ) * ((b - a) / (N : ℝ))) = y k := by
  intro N hN y
  classical
  have hba : 0 < b - a := sub_pos.mpr h_ab
  have hNpos : 0 < (N : ℝ) := by exact_mod_cast hN
  let δ : ℝ := (b - a) / (N : ℝ)
  have hδpos : 0 < δ := by
    dsimp [δ]
    exact div_pos hba hNpos
  let yAt : ℕ → ℝ := fun k =>
    if hk : k ≤ N then y ⟨k, Nat.lt_succ_of_le hk⟩ else 0
  let t : Fin N → ℝ := fun i => Internal.UniformInterpolation.grid a δ i.1
  let c : Fin N → ℝ := fun i => Internal.UniformInterpolation.coefficient δ yAt i.1
  refine ⟨hingeLayer1 N t, hingeLayer2 N c (yAt 0), ?_⟩
  intro k
  have hk_leN : k.1 ≤ N := Nat.le_of_lt_succ k.2
  calc
    mlpEvalScalar N (hingeLayer1 N t) (hingeLayer2 N c (yAt 0))
        (a + (k.1 : ℝ) * δ) = hingeFun N t c (yAt 0) (a + (k.1 : ℝ) * δ) :=
      mlp_eval_scalar_hinge N t c (yAt 0) _
    _ = Internal.UniformInterpolation.interpolant N a δ yAt
        (Internal.UniformInterpolation.grid a δ k.1) :=
      Internal.UniformInterpolation.hinge_fun_eq_interpolant N a δ yAt _
    _ = yAt k.1 := Internal.UniformInterpolation.interpolant_grid a yAt hδpos k.1 hk_leN
    _ = y k := by simp [yAt, hk_leN]

end

namespace RoundedReLUApprox

noncomputable section

/-- Real-valued scalars whose arithmetic rounds to the selected format's nearest-even grid.
The stored value is unrestricted; `NF.ofReal` additionally rounds a real input. -/
abbrev Rounded (format : FloatFormat) : Type :=
  NF binaryRadix (Model.fexpOf format) nearestEven

variable (format : FloatFormat)

/-- The `Rounded format` zero denotes the real `0`: rounding `0` is exact. -/
@[simp] theorem rounded_zero_val : (0 : Rounded format).val = 0 := by
  change (NF.ofReal (β := binaryRadix) (fexp := Model.fexpOf format)
    (rnd := nearestEven) (0 : ℝ)).val = 0
  -- Reduce to the mantissa being `0`; the exponent is irrelevant.
  simp [NF.ofReal, NF.roundR, Flocq.round, Flocq.toReal, scaledMantissa, cexp,
    magnitude, nearestEven]

/-- The `Rounded format` ReLU agrees with the real ReLU on the underlying value.

The Boolean zero test compares real values. A zero-valued input therefore returns a scalar whose
value is zero, and the remaining branch selects the input or zero according to the same real order.
Both cases give $\max(x.\mathrm{val},0)$ exactly, with no rounding error. -/
@[simp] theorem relu_rounded_val (x : Rounded format) :
    (Activation.Math.reluSpec x).val = relu x.val := by
  rw [Activation.Math.reluSpec_eq_max, relu, Activation.Math.reluSpec_eq_max]
  by_cases hx : (0 : Rounded format) ≤ x
  · have hReal : 0 ≤ x.val := by
      change (0 : Rounded format).val ≤ x.val at hx
      simpa only [rounded_zero_val format] using hx
    rw [max_eq_left hx, max_eq_left hReal]
  · have hx0 : x ≤ (0 : Rounded format) := le_of_not_ge hx
    have hReal : x.val ≤ 0 := by
      change x.val ≤ (0 : Rounded format).val at hx0
      simpa only [rounded_zero_val format] using hx0
    rw [max_eq_right hx0, rounded_zero_val format, max_eq_right hReal]

/--
Real ReLU is 1-Lipschitz.

The subtraction rounding error passes through the nonlinearity without amplification.
-/
theorem relu_lipschitz (u v : ℝ) : |relu u - relu v| ≤ |u - v| := by
  simpa only [relu, Activation.Math.reluSpec_eq_max] using abs_max_sub_max_le_abs u v 0

/-! ### Rounded hinge terms and a pointwise error bound -/

/-- One rounded hinge term $c_i\operatorname{ReLU}(x-t_i)$ in the rounded-real model. -/
def hingeTerm {n : ℕ} (c t : Fin n → Rounded format) (x : Rounded format) (i : Fin n) :
    Rounded format :=
  c i * Activation.Math.reluSpec (x - t i)

/-- Real reference for the same hinge term, using the parameters’ real values. -/
def hingeTermReal {n : ℕ} (c t : Fin n → Rounded format) (x : Rounded format) (i : Fin n) : ℝ :=
  (c i).val * relu (x.val - (t i).val)

/-- Certified error budget for one rounded hinge term: a half-ulp for the final multiplication plus
the subtraction rounding error, propagated through the `1`-Lipschitz ReLU and scaled by `|cᵢ|`. -/
def hingeTermErrorBound {n : ℕ} (c t : Fin n → Rounded format) (x : Rounded format)
    (i : Fin n) : ℝ :=
  Model.epsilonAt format ((c i).val * (Activation.Math.reluSpec (x - t i)).val)
  + |(c i).val| *
      (Model.epsilonAt format (x.val - (t i).val))

/--
Per-neuron rounded hinge-term error bound.

The bound has two pieces: one half-ulp term for the final multiplication and one subtraction
rounding term propagated through the $1$-Lipschitz ReLU and scaled by $|c_i|$.
-/
theorem hinge_term_abs_error {n : ℕ} (c t : Fin n → Rounded format) (x : Rounded format)
    (i : Fin n) :
    |(hingeTerm format c t x i).val - hingeTermReal format c t x i| ≤
      hingeTermErrorBound format c t x i := by
  set termQ := hingeTerm format c t x i
  set termR := hingeTermReal format c t x i
  let u : Rounded format := x - t i
  let r : Rounded format := Activation.Math.reluSpec u
  have hsub :
      |u.val - (x.val - (t i).val)| ≤
        Model.epsilonAt format (x.val - (t i).val) := by
    exact Model.abs_roundAt_sub_le format (x.val - (t i).val)
  have hrelu :
      |r.val - relu (x.val - (t i).val)| ≤
        |u.val - (x.val - (t i).val)| := by
    -- `Activation.Math.reluSpec` is exact on `.val`, so this is just the Lipschitz property of real
    -- ReLU.
    simpa only [r, relu_rounded_val format] using (relu_lipschitz u.val (x.val - (t i).val))
  have hmul :
      |termQ.val - ((c i).val * r.val)| ≤
        Model.epsilonAt format ((c i).val * r.val) := by
    exact Model.abs_roundAt_sub_le format ((c i).val * r.val)
  have hlin :
      |(c i).val * r.val - (c i).val * relu (x.val - (t i).val)| =
        |(c i).val| * |r.val - relu (x.val - (t i).val)| := by
    have hfactor :
        (c i).val * r.val - (c i).val * relu (x.val - (t i).val) =
          (c i).val * (r.val - relu (x.val - (t i).val)) := by ring
    rw [hfactor, abs_mul]
  calc
    |termQ.val - termR|
        ≤ |termQ.val - ((c i).val * r.val)| +
            |(c i).val * r.val - (c i).val * relu (x.val - (t i).val)| := by
              simpa [termR, hingeTermReal] using
                (abs_sub_le (termQ.val) ((c i).val * r.val) termR)
    _ ≤ Model.epsilonAt format ((c i).val * r.val) +
          (|(c i).val| * |r.val - relu (x.val - (t i).val)|) := by
          exact add_le_add hmul hlin.le
    _ ≤ Model.epsilonAt format ((c i).val * r.val) +
          (|(c i).val| * |u.val - (x.val - (t i).val)|) := by
          gcongr
    _ ≤ Model.epsilonAt format ((c i).val * r.val) +
          (|(c i).val| * (Model.epsilonAt format (x.val - (t i).val))) := by
          gcongr
    _ = hingeTermErrorBound format c t x i := by
          simp [hingeTermErrorBound, u, r]

/-! ### Summation error propagation (rounded hinge network) -/

/--
Fold state for summing hinge terms in `Rounded format`, while tracking:
- a real reference sum (computed from `.val`),
- and a provable error bound on the difference between them.
-/
abbrev HingeSumState : Type := Rounded format × ℝ × ℝ

/-- One summation step: add a hinge term, and accumulate rounding+term error bounds. -/
def hingeSumStateStep {n : ℕ} (c t : Fin n → Rounded format) (x : Rounded format) :
    HingeSumState format → Fin n → HingeSumState format
  | (accQ, accR, err), i =>
      let termQ : Rounded format := hingeTerm format c t x i
      let termR : ℝ := hingeTermReal format c t x i
      let termErr : ℝ := hingeTermErrorBound format c t x i
      let addErr : ℝ :=
        Model.epsilonAt format (accQ.val + termQ.val)
      (accQ + termQ, accR + termR, err + termErr + addErr)

/-- Compute the hinge-term sum state in ascending index order. -/
def hingeSumState {n : ℕ} (c t : Fin n → Rounded format) (x : Rounded format) :
    HingeSumState format :=
  Fin.foldl n (hingeSumStateStep format c t x) (0, 0, 0)

/-- Rounded value produced by folding all hinge terms in ascending index order. -/
def hingeSum {n : ℕ} (c t : Fin n → Rounded format) (x : Rounded format) : Rounded format :=
  (hingeSumState format c t x).1

/-- Real reference sum accumulated alongside `hingeSum`. -/
def hingeSumReal {n : ℕ} (c t : Fin n → Rounded format) (x : Rounded format) : ℝ :=
  (hingeSumState format c t x).2.1

/-- Accumulated certified absolute-error budget for `hingeSum`. -/
def hingeSumErrorBound {n : ℕ} (c t : Fin n → Rounded format) (x : Rounded format) : ℝ :=
  (hingeSumState format c t x).2.2

/--
Fold invariant for rounded hinge summation.

At every prefix of the fold, the rounded accumulator is within the tracked error budget of the
real accumulator.  The proof is deliberately order-sensitive because floating-point addition is
not associative.
-/
theorem hinge_sum_state_invariant {n : ℕ} (c t : Fin n → Rounded format) (x : Rounded format) :
    ∀ (xs : List (Fin n)) (accQ : Rounded format) (accR err : ℝ),
      |accQ.val - accR| ≤ err →
      let st := xs.foldl (hingeSumStateStep format c t x) (accQ, accR, err)
      |st.1.val - st.2.1| ≤ st.2.2 := by
  intro xs
  induction xs with
  | nil =>
      intro accQ accR err herr
      simp [List.foldl, herr]
  | cons i xs ih =>
      intro accQ accR err herr
      let termQ : Rounded format := hingeTerm format c t x i
      let termR : ℝ := hingeTermReal format c t x i
      let termErr : ℝ := hingeTermErrorBound format c t x i
      let addErr : ℝ :=
        Model.epsilonAt format (accQ.val + termQ.val)
      have hterm : |termQ.val - termR| ≤ termErr := hinge_term_abs_error format c t x i
      have hadd :
          |(accQ + termQ).val - (accQ.val + termQ.val)| ≤ addErr := by
        exact Model.abs_roundAt_sub_le format (accQ.val + termQ.val)
      have hstep :
          |(accQ + termQ).val - (accR + termR)| ≤ err + termErr + addErr := by
        -- Triangle: rounding of addition + linearization error.
        have htri :
            |(accQ + termQ).val - (accR + termR)| ≤
              |(accQ + termQ).val - (accQ.val + termQ.val)| +
              |(accQ.val + termQ.val) - (accR + termR)| := by
          simpa using
            (abs_sub_le (a := (accQ + termQ).val) (b := accQ.val + termQ.val)
              (c := accR + termR))
        have hlin :
            |(accQ.val + termQ.val) - (accR + termR)| ≤
              |accQ.val - accR| + |termQ.val - termR| := by
          have hdecomp :
              (accQ.val + termQ.val) - (accR + termR) =
                (accQ.val - accR) + (termQ.val - termR) := by
            ring
          simpa [hdecomp] using (abs_add_le (accQ.val - accR) (termQ.val - termR))
        have hlin' : |(accQ.val + termQ.val) - (accR + termR)| ≤ err + termErr := by
          have : |accQ.val - accR| + |termQ.val - termR| ≤ err + termErr :=
            add_le_add herr hterm
          exact le_trans hlin this
        have hsum :
            |(accQ + termQ).val - (accR + termR)| ≤ addErr + (err + termErr) := by
          exact le_trans htri (add_le_add hadd hlin')
        -- Reassociate to match the invariant's bound.
        linarith
      -- Apply IH on the tail, starting from the updated state.
      have := ih (accQ + termQ) (accR + termR) (err + termErr + addErr) hstep
      simpa [List.foldl, hingeSumStateStep, termQ, termR, termErr, addErr] using this

/-- Certified absolute-error bound for the whole rounded hinge-term sum. -/
theorem hinge_sum_abs_error {n : ℕ} (c t : Fin n → Rounded format) (x : Rounded format) :
    |(hingeSum format c t x).val - hingeSumReal format c t x| ≤
      hingeSumErrorBound format c t x := by
  have h0 : |(0 : Rounded format).val - (0 : ℝ)| ≤ (0 : ℝ) := by simp
  have h :=
    hinge_sum_state_invariant format (c := c) (t := t) (x := x)
      (xs := List.finRange n) (accQ := (0 : Rounded format)) (accR := (0 : ℝ)) (err := (0 : ℝ)) h0
  simpa [hingeSum, hingeSumReal, hingeSumErrorBound, hingeSumState,
    Fin.foldl_eq_foldl_finRange] using h

/-- Rounded hinge-network output: sum of hinge terms, then add the bias. -/
def hingeFun {n : ℕ} (t c : Fin n → Rounded format) (b x : Rounded format) : Rounded format :=
  hingeSum format c t x + b

/-- Real reference for `hingeFun`: evaluate over $\mathbb{R}$ on the `.val` parameters and
inputs. -/
def hingeFunReal {n : ℕ} (t c : Fin n → Rounded format) (b x : Rounded format) : ℝ :=
  hingeSumReal format c t x + b.val

/-- Total rounded hinge-network error budget, including the final rounded bias addition. -/
def hingeFunErrorBound {n : ℕ} (t c : Fin n → Rounded format) (b x : Rounded format) : ℝ :=
  hingeSumErrorBound format c t x
    + Model.epsilonAt format ((hingeSum format c t x).val + b.val)

/--
Certified absolute-error bound for the complete rounded hinge network.

This composes the fold invariant with the final rounded bias addition.
-/
theorem hinge_fun_abs_error {n : ℕ} (t c : Fin n → Rounded format) (b x : Rounded format) :
    |(hingeFun format t c b x).val - hingeFunReal format t c b x| ≤
      hingeFunErrorBound format t c b x := by
  have hsum := hinge_sum_abs_error format c t x
  set sQ : Rounded format := hingeSum format c t x
  set sR : ℝ := hingeSumReal format c t x
  set eS : ℝ := hingeSumErrorBound format c t x
  have hadd :
      |(sQ + b).val - (sQ.val + b.val)| ≤
        Model.epsilonAt format (sQ.val + b.val) := by
    exact Model.abs_roundAt_sub_le format (sQ.val + b.val)
  have htri :
      |(sQ + b).val - (sR + b.val)| ≤
        |(sQ + b).val - (sQ.val + b.val)| + |(sQ.val + b.val) - (sR + b.val)| := by
    simpa using (abs_sub_le (a := (sQ + b).val) (b := sQ.val + b.val) (c := sR + b.val))
  have hcancel : |(sQ.val + b.val) - (sR + b.val)| = |sQ.val - sR| := by
    rw [add_sub_add_right_eq_sub]
  have hbound :
      |(sQ + b).val - (sR + b.val)| ≤
        (Model.epsilonAt format (sQ.val + b.val)) + eS := by
    have : |(sQ + b).val - (sR + b.val)| ≤
        (Model.epsilonAt format (sQ.val + b.val)) +
          |sQ.val - sR| := by
      simpa [hcancel] using le_trans htri (add_le_add hadd (le_rfl))
    exact le_trans this (by gcongr)
  simpa [hingeFun, hingeFunReal, hingeFunErrorBound, sQ, sR, eS, add_assoc,
    add_left_comm, add_comm] using
    hbound

/-! ### Real approximation + rounding combination (pointwise) -/

/--
Triangle bound combining real approximation error with arithmetic rounding error.

The theorem is pointwise. Given a real hinge network close to `f`, it bounds the rounded-real
network by the real approximation error plus the certified rounding budget; it does not construct
the hinge parameters.
-/
theorem hinge_fun_total_abs_error_le {n : ℕ} (f : ℝ → ℝ) (t c : Fin n → Rounded format)
    (b x : Rounded format) :
    |f x.val - (hingeFun format t c b x).val| ≤
      |f x.val - hingeFunReal format t c b x| + hingeFunErrorBound format t c b x := by
  have hround :
      |hingeFunReal format t c b x - (hingeFun format t c b x).val| ≤
        hingeFunErrorBound format t c b x := by
    simpa [abs_sub_comm] using (hinge_fun_abs_error format (t := t) (c := c) (b := b) (x := x))
  have htri :
      |f x.val - (hingeFun format t c b x).val| ≤
        |f x.val - hingeFunReal format t c b x| +
          |hingeFunReal format t c b x - (hingeFun format t c b x).val| := by
    simpa using
      (abs_sub_le (a := f x.val) (b := hingeFunReal format t c b x)
        (c := (hingeFun format t c b x).val))
  exact le_trans htri (add_le_add_right hround _)

/-- Strict version of `hinge_fun_total_abs_error_le` for use with `< ε` approximation statements. -/
theorem hinge_fun_total_abs_error_lt {n : ℕ} (f : ℝ → ℝ) (t c : Fin n → Rounded format)
    (b x : Rounded format) {ε : ℝ} (hε : |f x.val - hingeFunReal format t c b x| < ε) :
    |f x.val - (hingeFun format t c b x).val| < ε + hingeFunErrorBound format t c b x :=
  lt_of_le_of_lt (hinge_fun_total_abs_error_le format f t c b x) (add_lt_add_of_lt_of_le hε le_rfl)

/-- The real reference accumulator is the ordinary finite sum of real hinge terms. -/
theorem hinge_sum_real_eq_sum {n : ℕ} (c t : Fin n → Rounded format) (x : Rounded format) :
    hingeSumReal format c t x = ∑ i : Fin n, hingeTermReal format c t x i := by
  classical
  -- First, show `hingeSumReal` is the plain `foldl` sum of `hingeTermReal`.
  have hfold :
      hingeSumReal format c t x =
        (List.finRange n).foldl (fun acc i => acc + hingeTermReal format c t x i) 0 := by
    -- Peel off the irrelevant components of the fold state by induction over the list.
    have :
        ∀ (xs : List (Fin n)) (accQ : Rounded format) (accR err : ℝ),
          (xs.foldl (hingeSumStateStep format c t x) (accQ, accR, err)).2.1 =
            xs.foldl (fun acc i => acc + hingeTermReal format c t x i) accR := by
      intro xs
      induction xs with
      | nil =>
          intro accQ accR err
          simp [List.foldl]
      | cons i xs ih =>
          intro accQ accR err
          simp [List.foldl, hingeSumStateStep, ih]
    simpa [hingeSumReal, hingeSumState, Fin.foldl_eq_foldl_finRange] using
      (this (List.finRange n) 0 0 0)
  -- Rewrite the executable fold as the corresponding finite sum.
  simpa [hfold] using
    (List.finRange_foldl_add_eq_finset_sum (n := n)
      (fun i : Fin n => hingeTermReal format c t x i))

/--
One-dimensional ReLU approximation with a pointwise rounding budget for any `FloatFormat`.

The real hinge construction supplies exact real parameters, embedded in `Rounded format` without
quantization. Each arithmetic operation rounds to the chosen grid. Thus the bound combines the
real approximation error with `hingeFunErrorBound`; it does not assert representability of the
parameters or inputs, or finiteness of a corresponding executable computation.
-/
theorem relu_universal_approximation_Icc {f : ℝ → ℝ} {a b L : ℝ}
    (h_ab : a < b) (hL : 0 < L)
    (h_lip : ∀ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, |f x - f y| ≤ L * |x - y|) :
    ∀ ε > 0,
      ∃ (hidDim : ℕ) (t c : Fin hidDim → Rounded format) (b0 : Rounded format),
        ∀ x ∈ Set.Icc a b,
          |f x - (hingeFun format t c b0 (⟨x⟩ : Rounded format)).val| <
            ε + hingeFunErrorBound format t c b0 (⟨x⟩ : Rounded format) := by
  intro ε hε
  classical
  rcases
      relu_universal_approximation_Icc_hinge (f := f) (a := a) (b := b) (L := L)
        h_ab hL h_lip ε hε with
    ⟨hidDim, tR, cR, happx⟩
  let t : Fin hidDim → Rounded format := fun i => ⟨tR i⟩
  let c : Fin hidDim → Rounded format := fun i => ⟨cR i⟩
  let b0 : Rounded format := ⟨f a⟩
  refine ⟨hidDim, t, c, b0, ?_⟩
  intro x hx
  let xQ : Rounded format := ⟨x⟩
  have hreal :
      hingeFunReal format t c b0 xQ = UniversalApproximation.hingeFun hidDim tR cR (f a) x := by
    -- The reference sum on exact embeddings equals the real hinge construction.
    have hsum :
        hingeSumReal format c t xQ = ∑ i : Fin hidDim, cR i * relu (x - tR i) := by
      -- `hingeSumReal` is a sum of `hingeTermReal`; `.val` of `⟨r⟩` is `r`.
      simpa [hingeTermReal, t, c, xQ] using hinge_sum_real_eq_sum format c t xQ
    simp [hingeFunReal, UniversalApproximation.hingeFun, hsum, b0, add_comm, xQ]
  have happx' : |f xQ.val - hingeFunReal format t c b0 xQ| < ε := by
    have := happx x hx
    simpa [xQ, hreal] using this
  simpa [xQ] using
    hinge_fun_total_abs_error_lt format f t c b0 xQ happx'

end

end RoundedReLUApprox

end NN.MLTheory.Proofs.UniversalApproximation

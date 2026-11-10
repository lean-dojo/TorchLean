/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Floats.Formats.BinaryInterchange.Analysis.Error
public import TorchLean.NN.UniversalApproximation
public import TorchLean.Tensor.NF

/-!
# Exact ReLU interpolation and rounded-real approximation

The real interpolation theorem gives a two-layer ReLU MLP matching arbitrary values on a uniform
grid. The `UniversalApproximation.Rounded` lemmas then bound each rounded hinge term, its ordered
sum, and the final bias addition for any `FloatFormat`, using FloatLib's half-ULP error theorem
`Model.abs_roundAt_sub_le`.

The scalar carrier is FloatLib's `NF`. Its stored real value need not be representable, while
arithmetic operations round to the selected precision and gradual-underflow grid, which has no
upper exponent bound. The approximation theorem stores exact real parameters and inputs in this
carrier, so parameter quantization and finite executable arithmetic need separate bounds.

The rounding term of `relu_Icc`, `hingeFunErrorBound format t c b0 x`, depends on the input `x`
and on parameters chosen after `ε`, and no theorem bounds it uniformly on `[a, b]`.
-/

@[expose] public section

open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq

namespace TorchLean.UniversalApproximation

open _root_.TorchLean.Tensor
open Activation (relu)

noncomputable section

/--
Exact interpolation on a uniform grid by a two-layer ReLU MLP over $\mathbb{R}$ semantics.

Given arbitrary target values $y_0,\ldots,y_N$ at the uniform grid points
$\operatorname{grid}(k)=a+k(b-a)/N$, some width-$N$ hinge network takes the value $y_k$ at
$\operatorname{grid}(k)$ for every $k$.
-/
theorem relu_mlp_exact_on_grid {a b : ℝ} (h_ab : a < b) :
    ∀ {N : ℕ}, 0 < N → ∀ y : Fin (N + 1) → ℝ,
      ∃ (l1 : Linear ℝ 1 N) (l2 : Linear ℝ N 1),
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
  let t : Fin N → ℝ := fun i => grid a δ i.1
  let c : Fin N → ℝ := fun i => hingeCoeff δ yAt i.1
  refine ⟨hingeLayer1 N t, hingeLayer2 N c (yAt 0), ?_⟩
  intro k
  have hk_leN : k.1 ≤ N := Nat.le_of_lt_succ k.2
  calc
    mlpEvalScalar N (hingeLayer1 N t) (hingeLayer2 N c (yAt 0))
        (a + (k.1 : ℝ) * δ) = hingeFun t c (yAt 0) (a + (k.1 : ℝ) * δ) :=
      mlpEvalScalar_hinge N t c (yAt 0) _
    _ = interpolant N a δ yAt
        (grid a δ k.1) :=
      hingeFun_eq_interpolant N a δ yAt _
    _ = yAt k.1 := interpolant_grid a yAt hδpos k.1 hk_leN
    _ = y k := by simp [yAt, hk_leN]

end

namespace Rounded

noncomputable section

/-- Real-valued scalars whose arithmetic rounds to nearest, ties to even, on the grid of the
selected format. The stored value is unrestricted, and `NF.ofReal` rounds a real input onto that
grid. -/
abbrev _root_.TorchLean.UniversalApproximation.Rounded (format : FloatFormat) : Type :=
  NF binaryRadix (Model.fexpOf format) nearestEven

variable (format : FloatFormat)

/-- The `Rounded format` zero denotes the real `0`: rounding `0` is exact. -/
@[simp] theorem zero_val : (0 : Rounded format).val = 0 := by
  change (NF.ofReal (β := binaryRadix) (fexp := Model.fexpOf format)
    (rnd := nearestEven) (0 : ℝ)).val = 0
  -- The scaled mantissa of `0` rounds to `0`, whatever the canonical exponent is.
  simp [NF.ofReal, NF.roundR, Flocq.round, Flocq.toReal, scaledMantissa, cexp,
    magnitude, nearestEven]

/-- The `Rounded format` ReLU agrees with the real ReLU on the underlying value.

The Boolean zero test compares real values. A zero-valued input therefore returns a scalar whose
value is zero, and the remaining branch selects the input or zero according to the same real order.
Both cases give $\max(x.\mathrm{val},0)$ exactly, with no rounding error. -/
@[simp] theorem relu_val (x : Rounded format) :
    (Activation.relu x).val = Activation.relu x.val := by
  rw [Activation.relu_eq_max, Activation.relu_eq_max]
  by_cases hx : (0 : Rounded format) ≤ x
  · have hReal : 0 ≤ x.val := by
      change (0 : Rounded format).val ≤ x.val at hx
      simpa only [zero_val format] using hx
    rw [max_eq_left hx, max_eq_left hReal]
  · have hx0 : x ≤ (0 : Rounded format) := le_of_not_ge hx
    have hReal : x.val ≤ 0 := by
      change x.val ≤ (0 : Rounded format).val at hx0
      simpa only [zero_val format] using hx0
    rw [max_eq_right hx0, zero_val format, max_eq_right hReal]

/-! ### Rounded hinge terms and a pointwise error bound -/

/-- Real reference for the same hinge term, computed from the real values of the parameters and
the input. -/
def hingeTermReal {n : ℕ} (c t : Fin n → Rounded format) (x : Rounded format) (i : Fin n) : ℝ :=
  (c i).val * Activation.relu (x.val - (t i).val)

/-- Certified error budget for one rounded hinge term: half an ulp at the exact product for the
final multiplication, plus $|c_i|$ times half an ulp at the exact difference $x-t_i$. The second
term bounds the subtraction rounding error, which the $1$-Lipschitz ReLU does not amplify. -/
def hingeTermErrorBound {n : ℕ} (c t : Fin n → Rounded format) (x : Rounded format)
    (i : Fin n) : ℝ :=
  Model.halfUlpAt format ((c i).val * (Activation.relu (x - t i)).val)
  + |(c i).val| *
      (Model.halfUlpAt format (x.val - (t i).val))

/--
Per-neuron rounded hinge-term error bound.

The rounded term lies within `hingeTermErrorBound` of its real reference. The proof bounds both
roundings by the half-ulp theorem, carries the subtraction error through ReLU with
`relu_val` and `abs_max_sub_max_le_abs`, and scales it by $|c_i|$.
-/
theorem hingeTerm_abs_error {n : ℕ} (c t : Fin n → Rounded format) (x : Rounded format)
    (i : Fin n) :
    |(hingeTerm c t x i).val - hingeTermReal format c t x i| ≤
      hingeTermErrorBound format c t x i := by
  set termQ := hingeTerm c t x i
  set termR := hingeTermReal format c t x i
  let u : Rounded format := x - t i
  let r : Rounded format := Activation.relu u
  have hsub :
      |u.val - (x.val - (t i).val)| ≤
        Model.halfUlpAt format (x.val - (t i).val) := by
    exact Model.abs_roundAt_sub_le format (x.val - (t i).val)
  have hrelu :
      |r.val - Activation.relu (x.val - (t i).val)| ≤
        |u.val - (x.val - (t i).val)| := by
    -- `Activation.relu` is exact on `.val`, so real ReLU's Lipschitz bound applies.
    rw [show r.val = Activation.relu u.val from relu_val format u]
    simpa only [Activation.relu_eq_max] using
      abs_max_sub_max_le_abs u.val (x.val - (t i).val) 0
  have hmul :
      |termQ.val - ((c i).val * r.val)| ≤
        Model.halfUlpAt format ((c i).val * r.val) := by
    exact Model.abs_roundAt_sub_le format ((c i).val * r.val)
  have hlin :
      |(c i).val * r.val - (c i).val * Activation.relu (x.val - (t i).val)| =
        |(c i).val| * |r.val - Activation.relu (x.val - (t i).val)| := by
    have hfactor :
        (c i).val * r.val - (c i).val * Activation.relu (x.val - (t i).val) =
          (c i).val * (r.val - Activation.relu (x.val - (t i).val)) := by ring
    rw [hfactor, abs_mul]
  calc
    |termQ.val - termR|
        ≤ |termQ.val - ((c i).val * r.val)| +
            |(c i).val * r.val - (c i).val * Activation.relu (x.val - (t i).val)| := by
              simpa [termR, hingeTermReal] using
                (abs_sub_le (termQ.val) ((c i).val * r.val) termR)
    _ ≤ Model.halfUlpAt format ((c i).val * r.val) +
          (|(c i).val| * |r.val - Activation.relu (x.val - (t i).val)|) := by
          exact add_le_add hmul hlin.le
    _ ≤ Model.halfUlpAt format ((c i).val * r.val) +
          (|(c i).val| * |u.val - (x.val - (t i).val)|) := by
          gcongr
    _ ≤ Model.halfUlpAt format ((c i).val * r.val) +
          (|(c i).val| * (Model.halfUlpAt format (x.val - (t i).val))) := by
          gcongr
    _ = hingeTermErrorBound format c t x i := by
          simp [hingeTermErrorBound, u, r]

/-! ### Summation error propagation (rounded hinge network) -/

/--
Fold state for summing hinge terms: the rounded accumulator in `Rounded format`, the real
reference sum computed from `.val`, and an error budget for the distance between the two.
-/
abbrev HingeSumState : Type := Rounded format × ℝ × ℝ

/-- One summation step adds the next rounded hinge term to the accumulator and its real reference
to the reference sum. The budget grows by the term's bound and by half an ulp at the exact sum of
the accumulator and the rounded term. -/
def hingeStep {n : ℕ} (c t : Fin n → Rounded format) (x : Rounded format) :
    HingeSumState format → Fin n → HingeSumState format
  | (accQ, accR, err), i =>
      let termQ : Rounded format := hingeTerm c t x i
      let termR : ℝ := hingeTermReal format c t x i
      let termErr : ℝ := hingeTermErrorBound format c t x i
      let addErr : ℝ :=
        Model.halfUlpAt format (accQ.val + termQ.val)
      (accQ + termQ, accR + termR, err + termErr + addErr)

/-- Compute the hinge-term sum state in ascending index order. -/
def hingeSumFold {n : ℕ} (c t : Fin n → Rounded format) (x : Rounded format) :
    HingeSumState format :=
  Fin.foldl n (hingeStep format c t x) (0, 0, 0)

/-- The rounded accumulator of the hinge-sum state is `hingeSum`. -/
theorem hingeSumFold_fst {n : ℕ} (c t : Fin n → Rounded format) (x : Rounded format) :
    (hingeSumFold format c t x).1 = hingeSum c t x := by
  have h : ∀ (xs : List (Fin n)) (accQ : Rounded format) (accR err : ℝ),
      (xs.foldl (hingeStep format c t x) (accQ, accR, err)).1 =
        xs.foldl (fun acc i => acc + hingeTerm c t x i) accQ := by
    intro xs
    induction xs with
    | nil => intro accQ accR err; rfl
    | cons i xs ih => intro accQ accR err; exact ih _ _ _
  simpa only [hingeSumFold, hingeSum, Fin.foldl_eq_foldl_finRange] using
    h (List.finRange n) 0 0 0

/-- Real reference sum accumulated alongside `hingeSum`. -/
def hingeSumReal {n : ℕ} (c t : Fin n → Rounded format) (x : Rounded format) : ℝ :=
  (hingeSumFold format c t x).2.1

/-- Accumulated certified absolute-error budget for `hingeSum`. -/
def hingeSumErrorBound {n : ℕ} (c t : Fin n → Rounded format) (x : Rounded format) : ℝ :=
  (hingeSumFold format c t x).2.2

/--
Fold invariant for rounded hinge summation.

If the rounded accumulator starts within the tracked budget of the real accumulator, it ends
within the final budget after a fold of `hingeStep` over any list of indices. The budget
depends on the order of the list because rounded addition is not associative.
-/
theorem hingeStep_invariant {n : ℕ} (c t : Fin n → Rounded format) (x : Rounded format) :
    ∀ (xs : List (Fin n)) (accQ : Rounded format) (accR err : ℝ),
      |accQ.val - accR| ≤ err →
      let st := xs.foldl (hingeStep format c t x) (accQ, accR, err)
      |st.1.val - st.2.1| ≤ st.2.2 := by
  intro xs
  induction xs with
  | nil =>
      intro accQ accR err herr
      simp [List.foldl, herr]
  | cons i xs ih =>
      intro accQ accR err herr
      let termQ : Rounded format := hingeTerm c t x i
      let termR : ℝ := hingeTermReal format c t x i
      let termErr : ℝ := hingeTermErrorBound format c t x i
      let addErr : ℝ :=
        Model.halfUlpAt format (accQ.val + termQ.val)
      have hterm : |termQ.val - termR| ≤ termErr := hingeTerm_abs_error format c t x i
      have hadd :
          |(accQ + termQ).val - (accQ.val + termQ.val)| ≤ addErr := by
        exact Model.abs_roundAt_sub_le format (accQ.val + termQ.val)
      have hstep :
          |(accQ + termQ).val - (accR + termR)| ≤ err + termErr + addErr := by
        -- Triangle inequality at the exact sum `accQ.val + termQ.val`.
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
      simpa [List.foldl, hingeStep, termQ, termR, termErr, addErr] using this

/-- Certified absolute-error bound for the whole rounded hinge-term sum. -/
theorem hingeSum_abs_error {n : ℕ} (c t : Fin n → Rounded format) (x : Rounded format) :
    |(hingeSum c t x).val - hingeSumReal format c t x| ≤
      hingeSumErrorBound format c t x := by
  rw [← hingeSumFold_fst]
  have h0 : |(0 : Rounded format).val - (0 : ℝ)| ≤ (0 : ℝ) := by simp
  have h :=
    hingeStep_invariant format (c := c) (t := t) (x := x)
      (xs := List.finRange n) (accQ := (0 : Rounded format)) (accR := (0 : ℝ)) (err := (0 : ℝ)) h0
  simpa [hingeSumReal, hingeSumErrorBound, hingeSumFold, Fin.foldl_eq_foldl_finRange] using h

/-- Real reference for `hingeFun`: evaluate over $\mathbb{R}$ on the `.val` parameters and
inputs. -/
def hingeFunReal {n : ℕ} (t c : Fin n → Rounded format) (b x : Rounded format) : ℝ :=
  hingeSumReal format c t x + b.val

/-- Total rounded hinge-network error budget, including the final rounded bias addition. -/
def hingeFunErrorBound {n : ℕ} (t c : Fin n → Rounded format) (b x : Rounded format) : ℝ :=
  hingeSumErrorBound format c t x
    + Model.halfUlpAt format ((hingeSum c t x).val + b.val)

/--
Certified absolute-error bound for the complete rounded hinge network.

The proof extends `hingeSum_abs_error` by half an ulp at the exact sum for the final bias
addition.
-/
theorem hingeFun_abs_error {n : ℕ} (t c : Fin n → Rounded format) (b x : Rounded format) :
    |(hingeFun t c b x).val - hingeFunReal format t c b x| ≤
      hingeFunErrorBound format t c b x := by
  have hsum := hingeSum_abs_error format c t x
  set sQ : Rounded format := hingeSum c t x
  set sR : ℝ := hingeSumReal format c t x
  set eS : ℝ := hingeSumErrorBound format c t x
  have hadd :
      |(sQ + b).val - (sQ.val + b.val)| ≤
        Model.halfUlpAt format (sQ.val + b.val) := by
    exact Model.abs_roundAt_sub_le format (sQ.val + b.val)
  have htri :
      |(sQ + b).val - (sR + b.val)| ≤
        |(sQ + b).val - (sQ.val + b.val)| + |(sQ.val + b.val) - (sR + b.val)| := by
    simpa using (abs_sub_le (a := (sQ + b).val) (b := sQ.val + b.val) (c := sR + b.val))
  have hcancel : |(sQ.val + b.val) - (sR + b.val)| = |sQ.val - sR| := by
    rw [add_sub_add_right_eq_sub]
  have hbound :
      |(sQ + b).val - (sR + b.val)| ≤
        (Model.halfUlpAt format (sQ.val + b.val)) + eS := by
    have : |(sQ + b).val - (sR + b.val)| ≤
        (Model.halfUlpAt format (sQ.val + b.val)) +
          |sQ.val - sR| := by
      simpa [hcancel] using le_trans htri (add_le_add hadd (le_rfl))
    exact le_trans this (by gcongr)
  simpa [hingeFun, hingeFunReal, hingeFunErrorBound, sQ, sR, eS, add_assoc,
    add_left_comm, add_comm] using
    hbound

/-! ### Real approximation + rounding combination (pointwise) -/

/--
Triangle bound combining real approximation error with arithmetic rounding error.

The bound is pointwise. At any input and for any parameters, the distance from `f` to the
rounded-real network is at most the distance from `f` to the real reference network plus
`hingeFunErrorBound`.
-/
theorem abs_sub_hingeFun_le {n : ℕ} (f : ℝ → ℝ) (t c : Fin n → Rounded format)
    (b x : Rounded format) :
    |f x.val - (hingeFun t c b x).val| ≤
      |f x.val - hingeFunReal format t c b x| + hingeFunErrorBound format t c b x := by
  have hround :
      |hingeFunReal format t c b x - (hingeFun t c b x).val| ≤
        hingeFunErrorBound format t c b x := by
    simpa [abs_sub_comm] using (hingeFun_abs_error format (t := t) (c := c) (b := b) (x := x))
  have htri :
      |f x.val - (hingeFun t c b x).val| ≤
        |f x.val - hingeFunReal format t c b x| +
          |hingeFunReal format t c b x - (hingeFun t c b x).val| := by
    simpa using
      (abs_sub_le (a := f x.val) (b := hingeFunReal format t c b x)
        (c := (hingeFun t c b x).val))
  exact le_trans htri (add_le_add_right hround _)

/-- Strict version of `abs_sub_hingeFun_le` for use with `< ε` approximation statements. -/
theorem abs_sub_hingeFun_lt {n : ℕ} (f : ℝ → ℝ) (t c : Fin n → Rounded format)
    (b x : Rounded format) {ε : ℝ} (hε : |f x.val - hingeFunReal format t c b x| < ε) :
    |f x.val - (hingeFun t c b x).val| < ε + hingeFunErrorBound format t c b x :=
  lt_of_le_of_lt (abs_sub_hingeFun_le format f t c b x) (add_lt_add_of_lt_of_le hε le_rfl)

/-- The real reference accumulator is the ordinary finite sum of real hinge terms. -/
theorem hingeSumReal_eq_sum {n : ℕ} (c t : Fin n → Rounded format) (x : Rounded format) :
    hingeSumReal format c t x = ∑ i : Fin n, hingeTermReal format c t x i := by
  classical
  -- First, show `hingeSumReal` is the plain `foldl` sum of `hingeTermReal`.
  have hfold :
      hingeSumReal format c t x =
        (List.finRange n).foldl (fun acc i => acc + hingeTermReal format c t x i) 0 := by
    -- Peel off the irrelevant components of the fold state by induction over the list.
    have :
        ∀ (xs : List (Fin n)) (accQ : Rounded format) (accR err : ℝ),
          (xs.foldl (hingeStep format c t x) (accQ, accR, err)).2.1 =
            xs.foldl (fun acc i => acc + hingeTermReal format c t x i) accR := by
      intro xs
      induction xs with
      | nil =>
          intro accQ accR err
          simp [List.foldl]
      | cons i xs ih =>
          intro accQ accR err
          simp [List.foldl, hingeStep, ih]
    simpa [hingeSumReal, hingeSumFold, Fin.foldl_eq_foldl_finRange] using
      (this (List.finRange n) 0 0 0)
  -- Rewrite the `List.finRange` fold as the corresponding finite sum.
  simpa [hfold] using
    ((Fin.sum_univ_def fun i : Fin n => hingeTermReal format c t x i).trans
      (List.sum_eq_foldl.trans List.foldl_map)).symm

/--
One-dimensional ReLU approximation with a pointwise rounding budget for any `FloatFormat`.

For an $L$-Lipschitz target on $[a,b]$ and any $\varepsilon > 0$, the real hinge construction
`relu_Icc_hinge` supplies exact parameters and the bias $f(a)$. They are
stored in `Rounded format` without rounding, like each input. Each arithmetic operation then
rounds to the chosen grid, and the rounded network lies within $\varepsilon$ plus
`hingeFunErrorBound` of $f$ at every point of $[a,b]$.
-/
theorem relu_Icc {f : ℝ → ℝ} {a b L : ℝ}
    (h_ab : a < b) (hL : 0 < L)
    (h_lip : ∀ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, |f x - f y| ≤ L * |x - y|) :
    ∀ ε > 0,
      ∃ (hidDim : ℕ) (t c : Fin hidDim → Rounded format) (b0 : Rounded format),
        ∀ x ∈ Set.Icc a b,
          |f x - (hingeFun t c b0 (⟨x⟩ : Rounded format)).val| <
            ε + hingeFunErrorBound format t c b0 (⟨x⟩ : Rounded format) := by
  intro ε hε
  classical
  rcases
      relu_Icc_hinge (f := f) (a := a) (b := b) (L := L)
        h_ab hL h_lip ε hε with
    ⟨hidDim, tR, cR, happx⟩
  let t : Fin hidDim → Rounded format := fun i => ⟨tR i⟩
  let c : Fin hidDim → Rounded format := fun i => ⟨cR i⟩
  let b0 : Rounded format := ⟨f a⟩
  refine ⟨hidDim, t, c, b0, ?_⟩
  intro x hx
  let xQ : Rounded format := ⟨x⟩
  have hreal :
      hingeFunReal format t c b0 xQ = hingeFun tR cR (f a) x := by
    -- The reference sum on exact embeddings equals the real hinge construction.
    have hsum :
        hingeSumReal format c t xQ = ∑ i : Fin hidDim, cR i * Activation.relu (x - tR i) := by
      -- `hingeSumReal` is a sum of `hingeTermReal` terms, and `.val` of `⟨r⟩` is `r`.
      simpa [hingeTermReal, t, c, xQ] using hingeSumReal_eq_sum format c t xQ
    simp [hingeFunReal, hingeFun_eq_add_sum, hsum, b0, add_comm, xQ]
  have happx' : |f xQ.val - hingeFunReal format t c b0 xQ| < ε := by
    have := happx x hx
    simpa [xQ, hreal] using this
  simpa [xQ] using
    abs_sub_hingeFun_lt format f t c b0 xQ happx'

end

end Rounded

end TorchLean.UniversalApproximation

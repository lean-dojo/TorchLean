/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# NeuralFloat core (Flocq-style rounded arithmetic)

TorchLean frequently reasons about floating-point behavior using a classical "rounded arithmetic
on $\mathbb{R}$" approach rather than a bit-level IEEE-754 model:

- represent a value with integer mantissa $m$ and exponent $e$,
- interpret it as the real number $m\beta^e$,
- describe the *format* via an exponent-selection function `fexp : ℤ → ℤ` and a rounding operator.

This decomposition is the same one used by the Coq library **Flocq**. It makes many theorems
reusable across formats (fixed-point, unbounded floats, and lower-exponent-bounded floats) and aligns well
with ULP-style error bounds from numerical analysis.

For executable, bit-level IEEE-754 semantics (NaN/Inf/signed zero), see `NN/Floats/IEEEExec/`.
Training annotations and named precision policies are deliberately kept out of this mathematical
core; see `NN/Floats/NeuralFloat/Metadata.lean`.

## References

- Flocq project (documentation + sources): https://flocq.gitlabpages.inria.fr/flocq/
- S. Boldo, G. Melquiond, “Flocq: a unified Coq library for proving floating-point algorithms
  correct”
  (ARITH 2011), DOI: 10.1109/ARITH.2011.40
- IEEE Standard for Floating-Point Arithmetic (IEEE 754-2019)
- N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002
-/

@[expose] public section


namespace TorchLean.Floats

/--
Radix (base) for "floating-point-like" representations.

In practice we almost always use base $2$ (`binaryRadix`) because that's what hardware implements,
but keeping the base explicit helps make the model match the literature (and it keeps some proofs
parametric in the radix).
-/
structure NeuralRadix where
  /-- Radix base (e.g. `2` for binary). -/
  base : ℕ
  /-- Validity condition: the base is at least 2. -/
  base_valid : 2 ≤ base

/-- Standard binary radix ($\beta=2$). -/
def binaryRadix : NeuralRadix := ⟨2, by norm_num⟩

namespace NeuralRadix

variable (r : NeuralRadix)

/-- Coerce the radix base to $\mathbb{R}$ (used by `bpow` and logarithms). -/
def toReal : ℝ := r.base

/-- The radix base is positive when viewed as a real number. -/
lemma pos : 0 < r.toReal := by
  have hb : 0 < r.base := lt_of_lt_of_le (by norm_num) r.base_valid
  have hbR : (0 : ℝ) < (r.base : ℝ) := by exact_mod_cast hb
  simpa [toReal] using hbR

/-- The radix base is nonzero when viewed as a real number. -/
lemma ne_zero : r.toReal ≠ 0 := ne_of_gt (pos r)

/-- The radix base is strictly greater than $1$ (for a valid radix,
$2\le\mathtt{base}$). -/
lemma gt_one : 1 < r.toReal := by
  simp [toReal]
  exact Nat.one_lt_cast.mpr (Nat.succ_le_iff.mp r.base_valid)

end NeuralRadix

/-- A radix-$\beta$ floating-point representation with integer mantissa and exponent. -/
structure NeuralFloat (β : NeuralRadix) where
  /-- Integer mantissa `m`. -/
  mantissa : ℤ
  /-- Integer exponent `e`. -/
  exponent : ℤ

namespace NeuralFloat

variable {β : NeuralRadix} (f : NeuralFloat β)

/-- Structural zero test (mantissa is exactly `0`). -/
def isZero : Prop := f.mantissa = 0

/-- Sign of the mantissa (matches the sign of `neuralToReal` since $\beta^e>0$). -/
def sign : ℤ := Int.sign f.mantissa

end NeuralFloat

/--
Base power: $\beta^e$ as a real number.

This is Flocq's `bpow` concept: the scaling factor used to interpret mantissa/exponent pairs.
-/
noncomputable def neuralBpow (β : NeuralRadix) (e : ℤ) : ℝ := β.toReal ^ e

namespace neuralBpow

variable (β : NeuralRadix)

/-- Base powers are positive: $\beta^e>0$ for any exponent $e$. -/
lemma pos (e : ℤ) : 0 < neuralBpow β e := zpow_pos (NeuralRadix.pos β) e

/-- Base powers are nonnegative: $\beta^e\ge0$ for any exponent $e$. -/
lemma nonneg (e : ℤ) : 0 ≤ neuralBpow β e := le_of_lt (pos β e)

/-- Base powers are never zero. -/
lemma ne_zero (e : ℤ) : neuralBpow β e ≠ 0 := ne_of_gt (pos β e)

/-- Exponent addition law: $\beta^{e_1+e_2}=\beta^{e_1}\beta^{e_2}$. -/
lemma add_exp (e1 e2 : ℤ) : neuralBpow β (e1 + e2) = neuralBpow β e1 * neuralBpow β e2 := by
  simp [neuralBpow, zpow_add₀ (NeuralRadix.ne_zero β)]

/-- Negating the exponent inverts the base power:
$\beta^{-e}=(\beta^e)^{-1}$. -/
lemma neg_exp (e : ℤ) : neuralBpow β (-e) = (neuralBpow β e)⁻¹ := by
  simp [neuralBpow, zpow_neg]

/-- Exponent subtraction corresponds to division of radix powers. -/
lemma sub_exp (e₁ e₂ : ℤ) :
    neuralBpow β (e₁ - e₂) = neuralBpow β e₁ / neuralBpow β e₂ := by
  simp [neuralBpow, zpow_sub₀ (NeuralRadix.ne_zero β)]

end neuralBpow

/--
Interpret a `NeuralFloat` as the real number $m\beta^e$.
-/
noncomputable def neuralToReal {β : NeuralRadix} (f : NeuralFloat β) : ℝ :=
  f.mantissa * neuralBpow β f.exponent

namespace neuralToReal

variable {β : NeuralRadix} (f : NeuralFloat β)

/-- `neuralToReal` is zero iff the mantissa is zero (since $\beta^e\ne0$). -/
@[simp] lemma zero_iff : neuralToReal f = 0 ↔ f.mantissa = 0 := by
  simp only [neuralToReal]
  -- We need to show: f.mantissa * neural_bpow β f.exponent = 0 ↔ f.mantissa = 0
  have bpow_pos : 0 < neuralBpow β f.exponent := neuralBpow.pos β f.exponent
  have bpow_ne_zero : neuralBpow β f.exponent ≠ 0 := ne_of_gt bpow_pos
  -- Use mul_eq_zero and the fact that bpow_ne_zero
  rw [mul_eq_zero]
  simp only [bpow_ne_zero, or_false]
  -- Now we just need to show the equivalence for the integer cast
  simp only [Int.cast_eq_zero]

end neuralToReal

/--
Magnitude in base $\beta$ of a real number.

This matches the usual definition
$\operatorname{mag}(x)=\lfloor\log_\beta|x|\rfloor+1$ for $x\ne0$, and $0$ for $x=0$.
It is the bridge between a real input $x$ and the exponent-selection function `fexp`.
-/
noncomputable def neuralMagnitude (β : NeuralRadix) (x : ℝ) : ℤ :=
  if x = 0 then 0
  else
    let base_mag := ⌊Real.log (abs x) / Real.log β.toReal⌋ + 1
    base_mag

/--
Validity predicate for exponent-selection functions.

This is the exponent-validity condition used by Flocq. Properties that are not consequences of
validity, such as monotonicity, are separate classes so generic results do not acquire unnecessary
hypotheses.
-/
class NeuralValidExp (fexp : ℤ → ℤ) : Prop where
  flocq_valid : ∀ k : ℤ,
    (fexp k < k → fexp (k + 1) ≤ k) ∧
    (k ≤ fexp k → fexp (fexp k + 1) ≤ fexp k ∧ ∀ l, l ≤ fexp k → fexp l = fexp k)

/-- Exponent-selection functions that preserve order. -/
class NeuralMonotoneExp (fexp : ℤ → ℤ) : Prop where
  monotone : ∀ k1 k2 : ℤ, k1 ≤ k2 → fexp k1 ≤ fexp k2

/-- A witness that the format has a lower exponent region, in Flocq's sense. -/
def IsNeuralNegligibleExp (fexp : ℤ → ℤ) (n : ℤ) : Prop :=
  n ≤ fexp n

/--
Select a witness $n\le\mathtt{fexp}(n)$ when the format has one. Unbounded formats such as FLX return
`none`; lower-bounded formats such as FLT return `some n`.
-/
noncomputable def neuralNegligibleExp (fexp : ℤ → ℤ) : Option ℤ := by
  classical
  exact if h : ∃ n, IsNeuralNegligibleExp fexp n then some (Classical.choose h) else none

/-- A selected negligible exponent satisfies $n\le\mathtt{fexp}(n)$. -/
theorem neuralNegligibleExp_spec {fexp : ℤ → ℤ} {n : ℤ}
    (h : neuralNegligibleExp fexp = some n) : IsNeuralNegligibleExp fexp n := by
  classical
  unfold neuralNegligibleExp at h
  split at h
  next hex =>
    have hn : Classical.choose hex = n := Option.some.inj h
    rw [← hn]
    exact Classical.choose_spec hex
  next _ => simp at h

/-- A format has no selected negligible exponent exactly when no negligible exponent exists. -/
theorem neuralNegligibleExp_eq_none_iff (fexp : ℤ → ℤ) :
    neuralNegligibleExp fexp = none ↔ ¬∃ n, IsNeuralNegligibleExp fexp n := by
  classical
  unfold neuralNegligibleExp
  split <;> simp_all

/--
Canonical exponent (`cexp` in Flocq terminology).

Given a nonzero `x`, we first compute its magnitude `mag(x)` and then apply `fexp` to pick the
exponent used for scaling/rounding.
-/
noncomputable def neuralCexp (β : NeuralRadix) (fexp : ℤ → ℤ) [NeuralValidExp fexp] (x : ℝ) : ℤ :=
  fexp (neuralMagnitude β x)

/-- A float representation is canonical when its stored exponent is the exponent selected for its value. -/
def NeuralCanonical (β : NeuralRadix) (fexp : ℤ → ℤ) [NeuralValidExp fexp]
    (f : NeuralFloat β) : Prop :=
  f.exponent = neuralCexp β fexp (neuralToReal f)

/--
Scaled mantissa $x\beta^{-\operatorname{cexp}(x)}$.

Intuitively: rescale `x` so that rounding “happens around exponent 0”, which is where `rnd` acts.
-/
noncomputable def neuralScaledMantissa (β : NeuralRadix) (fexp : ℤ → ℤ) [NeuralValidExp fexp] (x :
  ℝ) : ℝ :=
  x * neuralBpow β (-neuralCexp β fexp x)

/--
Generic format predicate (Flocq-style).

This says that $x$ is exactly representable in the format picked out by $\beta$ and `fexp`.
One way to read it is: the scaled mantissa is an integer (so there is no rounding error).
-/
def neuralGenericFormat (β : NeuralRadix) (fexp : ℤ → ℤ) [NeuralValidExp fexp] (x : ℝ) : Prop :=
  x =
    neuralToReal (β := β)
      { mantissa := ⌊neuralScaledMantissa β fexp x⌋
      , exponent := neuralCexp β fexp x
      }

/--
Unit in the last place (ULP) associated with $x$ and the format selected by `fexp`.

This is the scale of the “one ulp” step at the exponent selected by `cexp`. For round-to-nearest,
many standard bounds have the shape
$|\operatorname{round}(x)-x|\le\operatorname{ulp}(x)/2$.

An ULP is a property of the format and does not depend on runtime provenance annotations.
-/
noncomputable def neuralUlp (β : NeuralRadix) (fexp : ℤ → ℤ) [NeuralValidExp fexp] (x : ℝ) : ℝ :=
  if x = 0 then
    match neuralNegligibleExp fexp with
    | some n => neuralBpow β (fexp n)
    | none => 0
  else neuralBpow β (neuralCexp β fexp x)

namespace neuralUlp

variable (β : NeuralRadix) (fexp : ℤ → ℤ) [NeuralValidExp fexp]

/--
`neuralUlp` is always nonnegative.

Informally: an ulp is a step size on a real grid, so it cannot be negative.
-/
lemma nonneg (x : ℝ) : 0 ≤ neuralUlp β fexp x := by
  by_cases hx : x = 0
  · simp [neuralUlp, hx]
    split <;> simp [neuralBpow.nonneg]
  · simp [neuralUlp, hx, neuralBpow.nonneg]

/--
`neuralUlp` is strictly positive away from zero.

If $x\ne0$, the exponent selection $\operatorname{cexp}(x)$ picks a power of $\beta$, which is
strictly positive.
-/
lemma pos_of_ne_zero (x : ℝ) (hx : x ≠ 0) : 0 < neuralUlp β fexp x := by
  simp [neuralUlp, hx, neuralBpow.pos]

/-- The ULP at zero is determined by the format's negligible exponent, when one exists. -/
@[simp] lemma zero :
    neuralUlp β fexp (0 : ℝ) =
      match neuralNegligibleExp fexp with
      | some n => neuralBpow β (fexp n)
      | none => 0 := by
  simp [neuralUlp]

/--
Away from zero, `neuralUlp` is the base grid step $\beta^{\operatorname{cexp}(x)}$.
-/
@[simp] lemma of_ne_zero (x : ℝ) (hx : x ≠ 0) :
    neuralUlp β fexp x = neuralBpow β (neuralCexp β fexp x) := by
  simp [neuralUlp, hx]

end neuralUlp

end TorchLean.Floats

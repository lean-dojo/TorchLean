/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Numerics.Enclosure.Elementary.BinaryGridRuntime
public import FloatLib.Numerics.Enclosure.Interval.Hyperbolic
public import FloatLib.Numerics.Enclosure.Interval.Real
public import NN.Core.Numeric.Quotient
import FloatLib.Numerics.Enclosure.Elementary.BinaryGridProof
import Mathlib.Algebra.Ring.Int.Parity
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# Finite enclosures for weighted scalar coefficients

The bounded replay used by activation and power pullbacks evaluates all nested
coefficients over partial rational intervals. FloatLib proves the endpoint operations and
enclosures. Structural lifting preserves their real meaning at every coefficient depth. A successful
collection contains no failed leaf.

Each candidate carries a proof that it encloses the real coefficient expression before the
codec rounds it. This does not by itself prove the final codec's rounding law, identify that
expression with every caller's derivative, or verify a native GPU implementation.

`enclosedSlope?_eq_some` links a successful replay to its input and seed encodings, certified
bounds, and returned endpoint. The proof fields are erased from execution.
-/

namespace TorchLean.Numeric.Internal

open TorchLean.Numeric
open QuotientCoefficients

@[expose] public def Related {α β : Type} (R : α → β → Prop) :
    {n : Nat} → QuotientCoefficients n α → QuotientCoefficients n β → Prop
  | _, .scalar a, .scalar b => R a b
  | _, .dual a da, .dual b db => Related R a b ∧ Related R da db

theorem related_add {α β : Type} [Add α] [Add β] {R : α → β → Prop}
    (hadd : ∀ {a b : α} {c d : β}, R a c → R b d → R (a + b) (c + d))
    {n : Nat} {a b : QuotientCoefficients n α} {c d : QuotientCoefficients n β}
    (ha : Related R a c) (hb : Related R b d) : Related R (add a b) (add c d) := by
  induction n with
  | zero => cases a; cases b; cases c; cases d; exact hadd ha hb
  | succ n ih =>
    cases a; cases b; cases c; cases d
    exact ⟨ih ha.1 hb.1, ih ha.2 hb.2⟩

theorem related_sub {α β : Type} [Sub α] [Sub β] {R : α → β → Prop}
    (hsub : ∀ {a b : α} {c d : β}, R a c → R b d → R (a - b) (c - d))
    {n : Nat} {a b : QuotientCoefficients n α} {c d : QuotientCoefficients n β}
    (ha : Related R a c) (hb : Related R b d) : Related R (sub a b) (sub c d) := by
  induction n with
  | zero => cases a; cases b; cases c; cases d; exact hsub ha hb
  | succ n ih =>
    cases a; cases b; cases c; cases d
    exact ⟨ih ha.1 hb.1, ih ha.2 hb.2⟩

theorem related_mul {α β : Type} [Add α] [Add β] [Mul α] [Mul β] {R : α → β → Prop}
    (hadd : ∀ {a b : α} {c d : β}, R a c → R b d → R (a + b) (c + d))
    (hmul : ∀ {a b : α} {c d : β}, R a c → R b d → R (a * b) (c * d))
    {n : Nat} {a b : QuotientCoefficients n α} {c d : QuotientCoefficients n β}
    (ha : Related R a c) (hb : Related R b d) : Related R (mul a b) (mul c d) := by
  induction n with
  | zero => cases a; cases b; cases c; cases d; exact hmul ha hb
  | succ n ih =>
    cases a; cases b; cases c; cases d
    exact ⟨ih ha.1 hb.1, related_add hadd (ih ha.2 hb.1) (ih ha.1 hb.2)⟩

theorem related_div {α β : Type} [Add α] [Add β] [Sub α] [Sub β]
    [Mul α] [Mul β] [Div α] [Div β] {R : α → β → Prop}
    (hadd : ∀ {a b : α} {c d : β}, R a c → R b d → R (a + b) (c + d))
    (hsub : ∀ {a b : α} {c d : β}, R a c → R b d → R (a - b) (c - d))
    (hmul : ∀ {a b : α} {c d : β}, R a c → R b d → R (a * b) (c * d))
    (hdiv : ∀ {a b : α} {c d : β}, R a c → R b d → R (a / b) (c / d))
    {n : Nat} {a b : QuotientCoefficients n α} {c d : QuotientCoefficients n β}
    (ha : Related R a c) (hb : Related R b d) : Related R (div a b) (div c d) := by
  induction n with
  | zero => cases a; cases b; cases c; cases d; exact hdiv ha hb
  | succ n ih =>
    cases a; cases b; cases c; cases d
    exact ⟨ih ha.1 hb.1,
      ih (related_sub hsub ha.2 (related_mul hadd hmul (ih ha.1 hb.1) hb.2)) hb.1⟩

theorem related_exp {α β : Type} [Add α] [Add β] [Mul α] [Mul β] {R : α → β → Prop}
    (hadd : ∀ {a b : α} {c d : β}, R a c → R b d → R (a + b) (c + d))
    (hmul : ∀ {a b : α} {c d : β}, R a c → R b d → R (a * b) (c * d))
    (f : α → α) (g : β → β) (hf : ∀ {a b}, R a b → R (f a) (g b))
    {n : Nat} {a : QuotientCoefficients n α} {b : QuotientCoefficients n β}
    (ha : Related R a b) : Related R (exp f a) (exp g b) := by
  induction n with
  | zero => cases a; cases b; exact hf ha
  | succ n ih =>
    cases a; cases b
    exact ⟨ih ha.1, related_mul hadd hmul (ih ha.1) ha.2⟩

/-- Lift a logarithm through the existing coefficient quotient rule. -/
@[expose] public def logarithm {α : Type} [Add α] [Sub α] [Mul α] [Div α]
    (evaluate : α → α) :
    {n : Nat} → QuotientCoefficients n α → QuotientCoefficients n α
  | _, .scalar x => .scalar (evaluate x)
  | _, .dual x dx => .dual (logarithm evaluate x) (div dx x)

theorem related_logarithm {α β : Type} [Add α] [Add β] [Sub α] [Sub β]
    [Mul α] [Mul β] [Div α] [Div β] {R : α → β → Prop}
    (hadd : ∀ {a b : α} {c d : β}, R a c → R b d → R (a + b) (c + d))
    (hsub : ∀ {a b : α} {c d : β}, R a c → R b d → R (a - b) (c - d))
    (hmul : ∀ {a b : α} {c d : β}, R a c → R b d → R (a * b) (c * d))
    (hdiv : ∀ {a b : α} {c d : β}, R a c → R b d → R (a / b) (c / d))
    (f : α → α) (g : β → β) (hf : ∀ {a b}, R a b → R (f a) (g b))
    {n : Nat} {a : QuotientCoefficients n α} {b : QuotientCoefficients n β}
    (ha : Related R a b) : Related R (logarithm f a) (logarithm g b) := by
  induction n with
  | zero => cases a; cases b; exact hf ha
  | succ n ih =>
    cases a; cases b
    exact ⟨ih ha.1, related_div hadd hsub hmul hdiv ha.2 ha.1⟩

theorem related_ofScalar {α β : Type} [Zero α] [Zero β] {R : α → β → Prop}
    (hzero : R 0 0) (n : Nat) {a : α} {b : β} (h : R a b) :
    Related R (ofScalar n a) (ofScalar n b) := by
  induction n generalizing a b with
  | zero => exact h
  | succ n ih => exact ⟨ih h, ih hzero⟩

theorem related_map {α β γ : Type} {R : β → γ → Prop}
    (f : α → β) (g : α → γ) (h : ∀ a, R (f a) (g a)) {n : Nat}
    (a : QuotientCoefficients n α) : Related R (map f a) (map g a) := by
  induction n with
  | zero => cases a; exact h _
  | succ n ih => cases a; exact ⟨ih _, ih _⟩

/-- Lift a constant rational exponent through the existing coefficient product rule. -/
def constantPower {α : Type} [Zero α] [Add α] [Mul α]
    (constant : Rat → α) (evaluate : α → Rat → α) :
    {n : Nat} → QuotientCoefficients n α → Rat → QuotientCoefficients n α
  | n, input, exponent =>
    if exponent = 0 then ofScalar n (constant 1) else
      match input with
      | .scalar value => .scalar (evaluate value exponent)
      | .dual value tangent =>
          .dual (constantPower constant evaluate value exponent)
            (mul (ofScalar _ (constant exponent))
              (mul (constantPower constant evaluate value (exponent - 1)) tangent))

theorem related_constantPower {α β : Type} [Zero α] [Zero β]
    [Add α] [Add β] [Mul α] [Mul β] {R : α → β → Prop}
    (hzero : R 0 0) (constant : Rat → α) (constant' : Rat → β)
    (hconstant : ∀ q, R (constant q) (constant' q))
    (hadd : ∀ {a b : α} {c d : β}, R a c → R b d → R (a + b) (c + d))
    (hmul : ∀ {a b : α} {c d : β}, R a c → R b d → R (a * b) (c * d))
    (evaluate : α → Rat → α) (evaluate' : β → Rat → β)
    (hevaluate : ∀ {a b} (q : Rat), R a b → R (evaluate a q) (evaluate' b q))
    {n : Nat} {a : QuotientCoefficients n α} {b : QuotientCoefficients n β}
    (ha : Related R a b) (exponent : Rat) :
    Related R (constantPower constant evaluate a exponent)
      (constantPower constant' evaluate' b exponent) := by
  induction n generalizing exponent with
  | zero =>
    cases a; cases b
    by_cases hz : exponent = 0
    · simpa only [constantPower, ite_eq_left hz] using
        related_ofScalar hzero 0 (hconstant 1)
    · simpa only [constantPower, ite_eq_right hz, Related] using hevaluate exponent ha
  | succ n ih =>
    cases a; cases b
    by_cases hz : exponent = 0
    · simpa only [constantPower, ite_eq_left hz] using
        related_ofScalar hzero (n + 1) (hconstant 1)
    · simp only [constantPower, ite_eq_right hz]
      exact ⟨ih (exponent := exponent) ha.1,
        related_mul hadd hmul (related_ofScalar hzero n (hconstant exponent))
          (related_mul hadd hmul (ih (exponent := exponent - 1) ha.1) ha.2)⟩

open FloatLib.Numerics

abbrev Bounds := Option (Interval Rat)

def Holds (candidate : Bounds) (value : ℝ) : Prop :=
  ∀ interval, candidate = some interval → interval.ContainsReal some value

def binary (operation : Interval Rat → Interval Rat → Bounds) (x y : Bounds) : Bounds := do
  operation (← x) (← y)

def rounding : OutwardRounding Rat Rat := Interval.Internal.rationalOutwardRounding

local instance : Zero Bounds := ⟨some (Interval.point 0)⟩
local instance : Add Bounds := ⟨binary (Interval.add? rounding)⟩
local instance : Sub Bounds := ⟨binary (Interval.sub? rounding)⟩
local instance : Mul Bounds := ⟨binary (Interval.mul? rounding)⟩
local instance : Div Bounds := ⟨binary (Interval.div? rounding)⟩

theorem holds_point (x : Rat) : Holds (some (Interval.point x)) (x : ℝ) := by
  intro interval h
  cases Option.some.inj h
  exact Interval.containsReal_some_iff _ _ |>.2 ⟨le_rfl, le_rfl⟩

theorem holds_binary (operation : Interval Rat → Interval Rat → Bounds) (f : ℝ → ℝ → ℝ)
    (hsound : ∀ {a b c : Interval Rat} {x y : ℝ}, operation a b = some c →
      a.ContainsReal some x → b.ContainsReal some y → c.ContainsReal some (f x y))
    {a b : Bounds} {x y : ℝ} (ha : Holds a x) (hb : Holds b y) :
    Holds (binary operation a b) (f x y) := by
  intro interval h
  cases a with
  | none => simp [binary] at h
  | some a =>
    cases b with
    | none => simp [binary] at h
    | some b => exact hsound h (ha a rfl) (hb b rfl)

theorem holds_add {a b : Bounds} {x y : ℝ} (ha : Holds a x) (hb : Holds b y) :
    Holds (a + b) (x + y) :=
  holds_binary _ _ (fun h hx hy => Interval.containsReal_add? rounding h hx hy) ha hb

theorem holds_sub {a b : Bounds} {x y : ℝ} (ha : Holds a x) (hb : Holds b y) :
    Holds (a - b) (x - y) :=
  holds_binary _ _ (fun h hx hy => Interval.containsReal_sub? rounding h hx hy) ha hb

theorem holds_mul {a b : Bounds} {x y : ℝ} (ha : Holds a x) (hb : Holds b y) :
    Holds (a * b) (x * y) :=
  holds_binary _ _ (fun h hx hy => Interval.containsReal_mul? rounding h hx hy) ha hb

theorem holds_div {a b : Bounds} {x y : ℝ} (ha : Holds a x) (hb : Holds b y) :
    Holds (a / b) (x / y) :=
  holds_binary _ _ (fun h hx hy => Interval.containsReal_div? rounding h hx hy) ha hb

theorem holds_zero : Holds (Zero.zero : Bounds) (0 : ℝ) := by
  change Holds (some (Interval.point 0)) 0
  simpa only [Rat.cast_zero] using holds_point 0

abbrev CertifiedExp := (input : Interval Rat) → Option { bounds : Interval Rat //
  ∀ x : ℝ, input.ContainsReal some x → bounds.ContainsReal some (Real.exp x) }

def gridExp (degree precision : Nat) : CertifiedExp := fun input =>
  let lower := Enclosure.BinaryGrid.exp input.lo degree precision
  let upper := Enclosure.BinaryGrid.exp input.hi degree precision
  some ⟨⟨lower.lo, upper.hi⟩, by
    intro x hx
    rw [Interval.containsReal_some_iff] at hx ⊢
    exact ⟨(Enclosure.BinaryGrid.contains_exp input.lo degree precision).1.trans
        (Real.exp_le_exp.mpr hx.1),
      (Real.exp_le_exp.mpr hx.2).trans
        (Enclosure.BinaryGrid.contains_exp input.hi degree precision).2⟩⟩

private def gridLog (degree precision : Nat) (input : Interval Rat) :
    Option { bounds : Interval Rat //
      ∀ x : ℝ, input.ContainsReal some x → bounds.ContainsReal some (Real.log x) } :=
  if hpos : 0 < input.lo then
    let lower := Enclosure.BinaryGrid.log input.lo degree precision
    let upper := Enclosure.BinaryGrid.log input.hi degree precision
    some ⟨⟨lower.lo, upper.hi⟩, by
      intro x hx
      have hx' := (Interval.containsReal_some_iff input x).mp hx
      have hpositive : 0 < x :=
        (show (0 : ℝ) < (input.lo : ℝ) by exact_mod_cast hpos).trans_le hx'.1
      have hhi : 0 < input.hi := by
        have horder : (input.lo : ℝ) ≤ (input.hi : ℝ) := hx'.1.trans hx'.2
        have : input.lo ≤ input.hi := by exact_mod_cast horder
        exact hpos.trans_le this
      apply (Interval.containsReal_some_iff _ _).mpr
      exact ⟨(Enclosure.BinaryGrid.contains_log input.lo degree precision hpos).1.trans
          (Real.log_le_log (by exact_mod_cast hpos) hx'.1),
        (Real.log_le_log hpositive hx'.2).trans
          (Enclosure.BinaryGrid.contains_log input.hi degree precision hhi).2⟩⟩
  else none

private def boundedExp (degree precision budget : Nat) : CertifiedExp := fun input =>
  if -(budget : Rat) ≤ input.lo ∧ input.hi ≤ (budget : Rat) then
    gridExp degree precision input
  else none

private def positivePower (degree precision budget : Nat)
    (input : Interval Rat) (exponent : Rat) : Bounds := do
  let logarithm ← (gridLog degree precision input).map Subtype.val
  let argument ← Interval.mul? rounding logarithm (Interval.point exponent)
  (boundedExp degree precision budget argument).map Subtype.val

private theorem positivePower_contains {degree precision budget : Nat}
    {input output : Interval Rat} {exponent : Rat}
    (h : positivePower degree precision budget input exponent = some output)
    {x : ℝ} (hx : input.ContainsReal some x) :
    output.ContainsReal some (x ^ (exponent : ℝ)) := by
  cases hl : gridLog degree precision input with
  | none => simp [positivePower, hl] at h
  | some logarithm =>
    have hpos : 0 < input.lo := by
      by_contra hn
      simp [gridLog, hn] at hl
    have hpositive : 0 < x :=
      (show (0 : ℝ) < (input.lo : ℝ) by exact_mod_cast hpos).trans_le
        ((Interval.containsReal_some_iff input x).mp hx).1
    cases hm : Interval.mul? rounding logarithm.val (Interval.point exponent) with
    | none => simp [positivePower, hl, hm] at h
    | some argument =>
      have hproduct := Interval.containsReal_mul? rounding hm (logarithm.property x hx)
        (holds_point exponent _ rfl)
      change argument.ContainsReal some (Real.log x * (exponent : ℝ)) at hproduct
      cases he : boundedExp degree precision budget argument with
      | none => simp [positivePower, hl, hm, he] at h
      | some result =>
        have hout : result.val = output := by simpa [positivePower, hl, hm, he] using h
        subst output
        rw [Real.rpow_def_of_pos hpositive]
        exact result.property _ hproduct

/-- Enclose a real power on positive intervals or supported nonpositive point bases. -/
def powerLeaf (degree precision budget : Nat) (input : Bounds) (exponent : Rat) : Bounds := do
  let input ← input
  if 0 < input.lo then
    positivePower degree precision budget input exponent
  else if input.lo != input.hi then none
  else if input.lo == 0 then
    if exponent == 0 then some (Interval.point 1)
    else if exponent > 0 then some (Interval.point 0) else none
  else if exponent.den != 1 then none
  else
    let magnitude ← positivePower degree precision budget (Interval.point (-input.lo)) exponent
    if exponent.num % 2 == 0 then some magnitude
    else Interval.neg? rounding magnitude

private theorem rational_point_contains (q : Rat) :
    (Interval.point q).ContainsReal some (q : ℝ) :=
  holds_point q _ rfl

private theorem rpow_integer_sign (x : ℝ) (exponent : Rat)
    (hden : exponent.den = 1) :
    x ^ (exponent : ℝ) =
      if exponent.num % 2 = 0 then (-x) ^ (exponent : ℝ)
      else -((-x) ^ (exponent : ℝ)) := by
  have hcast : (exponent : ℝ) = (exponent.num : ℝ) := by
    exact_mod_cast (Rat.coe_int_num_of_den_eq_one hden).symm
  rw [hcast, Real.rpow_intCast, Real.rpow_intCast]
  by_cases heven : exponent.num % 2 = 0
  · rw [ite_eq_left heven]
    have h : Even exponent.num := ⟨exponent.num / 2, by omega⟩
    simpa only [neg_neg] using h.neg_zpow (-x)
  · rw [ite_eq_right heven]
    have h : Odd exponent.num := Int.odd_iff.mpr (by omega)
    simpa only [neg_neg] using h.neg_zpow (-x)

theorem powerLeaf_contains {degree precision budget : Nat}
    {input output : Interval Rat} {exponent : Rat}
    (h : powerLeaf degree precision budget (some input) exponent = some output)
    {x : ℝ} (hx : input.ContainsReal some x) :
    output.ContainsReal some (x ^ (exponent : ℝ)) := by
  by_cases hpositive : 0 < input.lo
  · exact positivePower_contains (by simpa [powerLeaf, hpositive] using h) hx
  by_cases hpoint : input.lo = input.hi
  · have hvalue : x = (input.lo : ℝ) := by
      obtain ⟨hlo, hhi⟩ := (Interval.containsReal_some_iff input x).mp hx
      apply le_antisymm
      · simpa [← hpoint] using hhi
      · exact hlo
    subst x
    by_cases hzero : input.lo = 0
    · by_cases hexponent : exponent = 0
      · have hout : Interval.point 1 = output := by
          have hs := h
          simp [powerLeaf, hzero, hexponent] at hs
          exact hs.2
        subst output
        simpa [hexponent] using rational_point_contains 1
      · by_cases hepositive : 0 < exponent
        · have hout : Interval.point 0 = output := by
            have hs := h
            simp [powerLeaf, hzero, hexponent, hepositive] at hs
            exact hs.2
          subst output
          have hcast : (exponent : ℝ) ≠ 0 := by exact_mod_cast hexponent
          simpa [hzero, Real.zero_rpow hcast] using rational_point_contains 0
        · simp [powerLeaf, hzero, hexponent, hepositive] at h
    · by_cases hden : exponent.den = 1
      · cases hmagnitude : positivePower degree precision budget
            (Interval.point (-input.lo)) exponent with
        | none => simp [powerLeaf, hpositive, hzero, hden, hmagnitude] at h
        | some magnitude =>
          have hbound := positivePower_contains hmagnitude (rational_point_contains (-input.lo))
          have hsign := rpow_integer_sign (input.lo : ℝ) exponent hden
          by_cases heven : exponent.num % 2 = 0
          · have hout : magnitude = output := by
              have hs := h
              simp [powerLeaf, hpositive, hzero, hden, hmagnitude, heven] at hs
              exact hs.2
            subst output
            rw [hsign, ite_eq_left heven]
            simpa only [Rat.cast_neg] using hbound
          · have hneg : Interval.neg? rounding magnitude = some output := by
              have hs := h
              simp [powerLeaf, hpositive, hzero, hden, hmagnitude, heven] at hs
              exact hs.2
            rw [hsign, ite_eq_right heven]
            have hn := Interval.containsReal_neg? rounding hneg hbound
            change output.ContainsReal some (-(((-input.lo : Rat) : ℝ) ^ (exponent : ℝ))) at hn
            simpa only [Rat.cast_neg] using hn
      · simp [powerLeaf, hpositive, hzero, hden] at h
  · simp [powerLeaf, hpositive, hpoint] at h

theorem holds_powerLeaf (degree precision budget : Nat) {input : Bounds} {x : ℝ}
    (hinput : Holds input x) (exponent : Rat) :
    Holds (powerLeaf degree precision budget input exponent) (x ^ (exponent : ℝ)) := by
  intro output h
  cases input with
  | none => simp [powerLeaf] at h
  | some input => exact powerLeaf_contains h (hinput input rfl)

theorem holds_exp (evaluate : Interval Rat → Bounds)
    (hsound : ∀ {a b : Interval Rat} {x : ℝ}, evaluate a = some b →
      a.ContainsReal some x → b.ContainsReal some (Real.exp x))
    {a : Bounds} {x : ℝ} (ha : Holds a x) :
    Holds (a.bind evaluate) (Real.exp x) := by
  intro interval h
  cases a with
  | none => simp at h
  | some a => exact hsound h (ha a rfl)

theorem related_collect {n : Nat} {bounds : QuotientCoefficients n Bounds}
    {values : QuotientCoefficients n ℝ} (hvalues : Related Holds bounds values)
    {result : QuotientCoefficients n (Interval Rat)} (hresult : mapM id bounds = some result) :
    Related (Interval.ContainsReal some) result values := by
  induction n with
  | zero =>
    cases bounds with | scalar a =>
      cases values with | scalar x =>
        cases a with
        | none => simp [mapM] at hresult
        | some interval =>
          change some (scalar interval) = some result at hresult
          cases Option.some.inj hresult
          exact hvalues interval rfl
  | succ n ih =>
    cases bounds with | dual a b =>
      cases values with | dual x y =>
        cases ha : mapM id a with
        | none => simp [mapM, ha] at hresult
        | some a' =>
          cases hb : mapM id b with
          | none => simp [mapM, ha, hb] at hresult
          | some b' =>
            simp [mapM, ha, hb] at hresult
            subst result
            exact ⟨ih hvalues.1 ha, ih hvalues.2 hb⟩

def weighted {α : Type} [Zero α] [Add α] [Sub α] [Mul α] [Div α]
    {n : Nat} (one : α) (seed z : QuotientCoefficients n α) (isTanh : Bool) :
    QuotientCoefficients n α :=
  let denominator := add (ofScalar n one) z
  let numerator := mul seed z
  let numerator := if isTanh then
    let twice := add numerator numerator
    add twice twice
    else numerator
  div numerator (mul denominator denominator)

theorem related_weighted {α β : Type} [Zero α] [Zero β] [Add α] [Add β]
    [Sub α] [Sub β] [Mul α] [Mul β] [Div α] [Div β] {R : α → β → Prop}
    (hzero : R 0 0) {one : α} {one' : β} (hone : R one one')
    (hadd : ∀ {a b : α} {c d : β}, R a c → R b d → R (a + b) (c + d))
    (hsub : ∀ {a b : α} {c d : β}, R a c → R b d → R (a - b) (c - d))
    (hmul : ∀ {a b : α} {c d : β}, R a c → R b d → R (a * b) (c * d))
    (hdiv : ∀ {a b : α} {c d : β}, R a c → R b d → R (a / b) (c / d))
    {n : Nat} {seed z : QuotientCoefficients n α} {seed' z' : QuotientCoefficients n β}
    (hseed : Related R seed seed') (hz : Related R z z') (isTanh : Bool) :
    Related R (weighted one seed z isTanh) (weighted one' seed' z' isTanh) := by
  have hden := related_add hadd (related_ofScalar hzero n hone) hz
  have hnum := related_mul hadd hmul hseed hz
  have hden2 := related_mul hadd hmul hden hden
  cases isTanh
  · exact related_div hadd hsub hmul hdiv hnum hden2
  · have htwice := related_add hadd hnum hnum
    exact related_div hadd hsub hmul hdiv (related_add hadd htwice htwice) hden2

def argument {n : Nat} (x : QuotientCoefficients n) (isTanh : Bool) :
    QuotientCoefficients n :=
  let sign : Rat := if x.primal > 0 then -1 else 1
  let rate := if isTanh then 2 * sign else sign
  mul (ofScalar n rate) x

def slopeRaw {n : Nat} (x seed : QuotientCoefficients n) (isTanh : Bool)
    (evaluate : Interval Rat → Bounds) : QuotientCoefficients n Bounds :=
  let point := fun value : Rat => some (Interval.point value)
  let input := map point (argument x isTanh)
  let z := exp (fun input => input.bind evaluate) input
  weighted (point 1) (map point seed) z isTanh

public noncomputable def idealSlope {n : Nat}
    (x seed : QuotientCoefficients n) (isTanh : Bool) :
    QuotientCoefficients n ℝ :=
  let z := exp Real.exp (map (fun q : Rat => (q : ℝ)) (argument x isTanh))
  weighted 1 (map (fun q : Rat => (q : ℝ)) seed) z isTanh

/-- The scalar coefficient has the same decaying-exponential formula on either sign branch. -/
public theorem idealSlope_scalar (input seed : Rat) (isTanh : Bool) :
    (idealSlope (.scalar input) (.scalar seed) isTanh).primal =
      let tail := Real.exp (-(if isTanh then (2 : ℝ) else 1) * |(input : ℝ)|)
      let numerator := (seed : ℝ) * tail
      (if isTanh then (numerator + numerator) + (numerator + numerator) else numerator) /
        ((1 + tail) * (1 + tail)) := by
  by_cases positive : (0 : Rat) < input
  · have realPositive : (0 : ℝ) < (input : ℝ) := by exact_mod_cast positive
    cases isTanh <;>
      simp [idealSlope, argument, weighted, QuotientCoefficients.exp,
        QuotientCoefficients.map, QuotientCoefficients.mul, QuotientCoefficients.add,
        QuotientCoefficients.div, QuotientCoefficients.ofScalar,
        QuotientCoefficients.primal, positive, abs_of_pos realPositive]
  · have realNonpositive : (input : ℝ) ≤ 0 := by exact_mod_cast le_of_not_gt positive
    cases isTanh <;>
      simp [idealSlope, argument, weighted, QuotientCoefficients.exp,
        QuotientCoefficients.map, QuotientCoefficients.mul, QuotientCoefficients.add,
        QuotientCoefficients.div, QuotientCoefficients.ofScalar,
        QuotientCoefficients.primal, positive, abs_of_nonpos realNonpositive]

theorem related_slopeRaw {n : Nat} (x seed : QuotientCoefficients n) (isTanh : Bool)
    (evaluate : Interval Rat → Bounds)
    (hsound : ∀ {a b : Interval Rat} {x : ℝ}, evaluate a = some b →
      a.ContainsReal some x → b.ContainsReal some (Real.exp x)) :
    Related Holds (slopeRaw x seed isTanh evaluate) (idealSlope x seed isTanh) := by
  apply related_weighted holds_zero (by simpa only [Rat.cast_one] using holds_point 1)
    holds_add holds_sub holds_mul holds_div
  · exact related_map _ _ holds_point seed
  · exact related_exp holds_add holds_mul _ _ (holds_exp evaluate hsound)
      (related_map _ _ holds_point (argument x isTanh))

theorem certified_exp_sound (evaluate : CertifiedExp) {a b : Interval Rat} {x : ℝ}
    (h : (evaluate a).map Subtype.val = some b) (hx : a.ContainsReal some x) :
    b.ContainsReal some (Real.exp x) := by
  cases he : evaluate a with
  | none => simp [he] at h
  | some result =>
    simp only [he, Option.map_some, Option.some.injEq] at h
    subst b
    exact result.property x hx

def slopeBounds {n : Nat} (x seed : QuotientCoefficients n) (isTanh : Bool)
    (evaluate : CertifiedExp) :
    Option { bounds : QuotientCoefficients n (Interval Rat) //
      Related (Interval.ContainsReal some) bounds (idealSlope x seed isTanh) } :=
  match h : mapM id (slopeRaw x seed isTanh
      (fun input => (evaluate input).map Subtype.val)) with
  | none => none
  | some bounds => some ⟨bounds,
      related_collect (related_slopeRaw x seed isTanh _ (certified_exp_sound evaluate)) h⟩

/-- The complete constant-exponent expression `exponent * x^(exponent-1) * seed`. -/
def powerRaw {n : Nat} (x seed : QuotientCoefficients n) (exponent : Rat)
    (degree precision budget : Nat) :
    QuotientCoefficients n Bounds :=
  let point := fun value : Rat => some (Interval.point value)
  mul (ofScalar n (point exponent))
    (mul (constantPower point (powerLeaf degree precision budget) (map point x) (exponent - 1))
      (map point seed))

noncomputable def idealPower {n : Nat} (x seed : QuotientCoefficients n) (exponent : Rat) :
    QuotientCoefficients n ℝ :=
  let constant := fun value : Rat => (value : ℝ)
  mul (ofScalar n (constant exponent))
    (mul (constantPower constant (fun value q => value ^ (q : ℝ))
      (map constant x) (exponent - 1)) (map constant seed))

theorem related_powerRaw {n : Nat} (x seed : QuotientCoefficients n)
    (exponent : Rat) (degree precision budget : Nat) :
    Related Holds (powerRaw x seed exponent degree precision budget)
      (idealPower x seed exponent) := by
  apply related_mul holds_add holds_mul (related_ofScalar holds_zero n (holds_point exponent))
  apply related_mul holds_add holds_mul
  · exact related_constantPower holds_zero _ _ holds_point holds_add holds_mul _ _
      (fun q hx => holds_powerLeaf degree precision budget hx q)
      (related_map _ _ holds_point x) (exponent - 1)
  · exact related_map _ _ holds_point seed

def powerBounds {n : Nat} (x seed : QuotientCoefficients n) (exponent : Rat)
    (degree precision budget : Nat) :
    Option { bounds : QuotientCoefficients n (Interval Rat) //
      Related (Interval.ContainsReal some) bounds (idealPower x seed exponent) } :=
  match h : mapM id (powerRaw x seed exponent degree precision budget) with
  | none => none
  | some bounds => some ⟨bounds,
      related_collect (related_powerRaw x seed exponent degree precision budget) h⟩

private def logLeaf (degree precision : Nat) (input : Bounds) : Bounds := do
  let input ← input
  (gridLog degree precision input).map Subtype.val

private theorem holds_logLeaf (degree precision : Nat) {input : Bounds} {x : ℝ}
    (hx : Holds input x) : Holds (logLeaf degree precision input) (Real.log x) := by
  intro output h
  cases input with
  | none => simp [logLeaf] at h
  | some input =>
    cases hl : gridLog degree precision input with
    | none => simp [logLeaf, hl] at h
    | some result =>
      have hout : result.val = output := by simpa [logLeaf, hl] using h
      subst output
      exact result.property x (hx input rfl)

/-- Enclose a logarithm and all of its nested quotient-rule coefficients. -/
def logarithmBounds {n : Nat} (input : QuotientCoefficients n) (degree precision : Nat) :
    Option { bounds : QuotientCoefficients n (Interval Rat) //
      Related (Interval.ContainsReal some) bounds
        (logarithm Real.log (map (fun q : Rat => (q : ℝ)) input)) } :=
  let raw := logarithm (logLeaf degree precision)
    (map (fun q => some (Interval.point q)) input)
  match h : mapM id raw with
  | none => none
  | some bounds => some ⟨bounds, related_collect
      (related_logarithm holds_add holds_sub holds_mul holds_div _ _
        (holds_logLeaf degree precision) (related_map _ _ holds_point input)) h⟩

/-- Joint power tangent formed from one shared logarithm and the full exponential expression. -/
def powerDirectional {α : Type} [Add α] [Sub α] [Mul α] [Div α]
    (evaluateLog evaluateExp : α → α) {n : Nat}
    (x y dx dy : QuotientCoefficients n α) : QuotientCoefficients n α :=
  let logarithm := logarithm evaluateLog x
  let power := exp evaluateExp (mul y logarithm)
  mul power (add (mul dy logarithm) (mul y (div dx x)))

def directionalRaw {n : Nat} (x y dx dy : QuotientCoefficients n)
    (degree precision budget : Nat) : QuotientCoefficients n Bounds :=
  let point := fun value : Rat => some (Interval.point value)
  powerDirectional (logLeaf degree precision)
    (fun input => input.bind fun interval =>
      (boundedExp degree precision budget interval).map Subtype.val)
    (map point x) (map point y) (map point dx) (map point dy)

noncomputable def idealPowerTangent {n : Nat} (x y dx dy : QuotientCoefficients n) :
    QuotientCoefficients n ℝ :=
  let real := fun value : Rat => (value : ℝ)
  powerDirectional Real.log Real.exp (map real x) (map real y) (map real dx) (map real dy)

theorem related_directionalRaw {n : Nat} (x y dx dy : QuotientCoefficients n)
    (degree precision budget : Nat) :
    Related Holds (directionalRaw x y dx dy degree precision budget)
      (idealPowerTangent x y dx dy) := by
  have hx := related_map _ _ holds_point x
  have hy := related_map _ _ holds_point y
  have hdx := related_map _ _ holds_point dx
  have hdy := related_map _ _ holds_point dy
  have hlog := related_logarithm holds_add holds_sub holds_mul holds_div _ _
    (holds_logLeaf degree precision) hx
  have hexp := related_exp holds_add holds_mul _ _
    (holds_exp _ (certified_exp_sound (boundedExp degree precision budget)))
    (related_mul holds_add holds_mul hy hlog)
  exact related_mul holds_add holds_mul hexp
    (related_add holds_add (related_mul holds_add holds_mul hdy hlog)
      (related_mul holds_add holds_mul hy
        (related_div holds_add holds_sub holds_mul holds_div hdx hx)))

def directionalBounds {n : Nat} (x y dx dy : QuotientCoefficients n)
    (degree precision budget : Nat) :
    Option { bounds : QuotientCoefficients n (Interval Rat) //
      Related (Interval.ContainsReal some) bounds (idealPowerTangent x y dx dy) } :=
  match h : mapM id (directionalRaw x y dx dy degree precision budget) with
  | none => none
  | some bounds => some ⟨bounds,
      related_collect (related_directionalRaw x y dx dy degree precision budget) h⟩


/-- Enclose a sufficiently negative exponential without evaluating a Taylor polynomial. -/
def tailExp? (bits : Nat) : CertifiedExp := fun input =>
  if h : (bits : Rat) ≤ -input.hi then
    some ⟨⟨0, (1 / 2 : Rat) ^ bits⟩, by
      intro y hy
      rw [Interval.containsReal_some_iff] at hy ⊢
      refine ⟨by simpa using Real.exp_nonneg y, ?_⟩
      have hh : (bits : ℝ) ≤ -(input.hi : ℝ) := by exact_mod_cast h
      have hhalf : Real.exp (-1) ≤ (1 / 2 : ℝ) := by
        rw [Real.exp_neg]
        simpa only [one_div] using one_div_le_one_div_of_le (by norm_num : (0 : ℝ) < 2)
          (by linarith [Real.add_one_le_exp (1 : ℝ)])
      have hbound : Real.exp y ≤ (1 / 2 : ℝ) ^ bits := calc
        Real.exp y ≤ Real.exp (-(bits : ℝ)) := Real.exp_le_exp.mpr (by linarith)
        _ = Real.exp (-1) ^ bits := by rw [← Real.exp_nat_mul]; congr 1; ring
        _ ≤ (1 / 2 : ℝ) ^ bits := pow_le_pow_left₀ (Real.exp_nonneg _) hhalf bits
      simpa using hbound⟩
  else none

private def coefficientBits : {n : Nat} → QuotientCoefficients n → Nat
  | _, .scalar x => x.num.natAbs.log2 + 1 - x.den.log2
  | _, .dual x dx => max (coefficientBits x) (coefficientBits dx)

private def coefficientSize : {n : Nat} → QuotientCoefficients n → Nat
  | _, .scalar x => x.num.natAbs.log2 + x.den.log2 + 2
  | _, .dual x dx => max (coefficientSize x) (coefficientSize dx)

private def boundsWidth : {n : Nat} →
    QuotientCoefficients n (FloatLib.Numerics.Interval Rat) → Rat
  | _, .scalar bounds => bounds.hi - bounds.lo
  | _, .dual x dx => max (boundsWidth x) (boundsWidth dx)

private def roundBounds? {α : Type} [ops : QuotientArithmetic α]
    (bounds : QuotientCoefficients ops.depth (Interval Rat)) : Option α := do
  let lower ← ops.decode (QuotientCoefficients.map (·.lo) bounds)
  let upper ← ops.decode (QuotientCoefficients.map (·.hi) bounds)
  let lowerCoefficients ← ops.encode lower
  let upperCoefficients ← ops.encode upper
  if lowerCoefficients == upperCoefficients then some lower else none

private theorem coefficients_eq_of_beq {n : Nat} {left right : QuotientCoefficients n}
    (equal : (left == right) = true) : left = right := by
  induction n with
  | zero =>
      cases left with | scalar left =>
        cases right with | scalar right =>
          change (left == right) = true at equal
          exact congrArg Coefficients.scalar (beq_iff_eq.mp equal)
  | succ n ih =>
      cases left with | dual left leftDirection =>
        cases right with | dual right rightDirection =>
          change ((left == right) && (leftDirection == rightDirection)) = true at equal
          simp only [Bool.and_eq_true] at equal
          obtain ⟨valueEqual, directionEqual⟩ := equal
          rw [ih valueEqual, ih directionEqual]

/-- Acceptance returns the decoded lower endpoints and agrees on every encoded coefficient. -/
private theorem roundBounds?_eq_some {α : Type} [ops : QuotientArithmetic α]
    {bounds : QuotientCoefficients ops.depth (Interval Rat)} {result : α}
    (accepted : roundBounds? bounds = some result) :
    ops.decode (map (·.lo) bounds) = some result ∧
      ∃ upper shared, ops.decode (map (·.hi) bounds) = some upper ∧
        ops.encode result = some shared ∧ ops.encode upper = some shared := by
  unfold roundBounds? at accepted
  obtain ⟨lower, lowerDecoded, accepted⟩ := Option.bind_eq_some_iff.mp accepted
  obtain ⟨upper, upperDecoded, accepted⟩ := Option.bind_eq_some_iff.mp accepted
  obtain ⟨lowerCoefficients, lowerEncoded, accepted⟩ := Option.bind_eq_some_iff.mp accepted
  obtain ⟨upperCoefficients, upperEncoded, accepted⟩ := Option.bind_eq_some_iff.mp accepted
  split at accepted
  · rename_i agree
    have coefficientsEqual := coefficients_eq_of_beq agree
    have resultEqual := Option.some.inj accepted
    subst result
    exact ⟨lowerDecoded, upper, lowerCoefficients, upperDecoded, lowerEncoded,
      coefficientsEqual.symm ▸ upperEncoded⟩
  · cases accepted

private def refineBounds? {α : Type} [ops : QuotientArithmetic α] (budget : Nat)
    (enclose : Nat → Nat → Option (QuotientCoefficients ops.depth (Interval Rat))) : Option α :=
  let rec search (degree precision : Nat) (previousWidth : Option Rat) : Nat → Option α
    | 0 => none
    | steps + 1 => do
      let bounds ← enclose degree precision
      match roundBounds? bounds with
      | some value => some value
      | none =>
        let width := boundsWidth bounds
        if previousWidth.any (fun previous => decide (previous ≤ 4 * width)) then
          search degree (2 * precision) none steps
        else
          search (2 * degree) (precision + 1) (some width) steps
  search 8 budget none 8

private theorem refineBounds_search_eq_some {α : Type} [ops : QuotientArithmetic α]
    (enclose : Nat → Nat → Option (QuotientCoefficients ops.depth (Interval Rat)))
    (degree precision : Nat) (previousWidth : Option Rat) (steps : Nat) {result : α}
    (accepted : refineBounds?.search enclose degree precision previousWidth steps = some result) :
    ∃ degree precision bounds,
      enclose degree precision = some bounds ∧ roundBounds? bounds = some result := by
  induction steps generalizing degree precision previousWidth with
  | zero =>
      simp only [refineBounds?.search] at accepted
      cases accepted
  | succ steps ih =>
      simp only [refineBounds?.search] at accepted
      obtain ⟨bounds, enclosed, accepted⟩ := Option.bind_eq_some_iff.mp accepted
      cases rounded : roundBounds? bounds with
      | some value =>
          simp only [rounded] at accepted
          cases Option.some.inj accepted
          exact ⟨degree, precision, bounds, enclosed, rounded⟩
      | none =>
          simp only [rounded] at accepted
          split at accepted <;> exact ih _ _ _ accepted

/-- Refine a logarithm from finite coefficients before rounding through the selected codec.

This accepts rational arguments outside the scalar format's range. The domain is positive and
all nested coefficients must have agreeing rounded enclosure endpoints. -/
@[noinline] public def enclosedLogarithm? {α : Type} [ops : QuotientArithmetic α]
    (input : QuotientCoefficients ops.depth) : Option α :=
  if input.primal ≤ 0 then none else
    let budget := 64 + (ops.depth + 1) * coefficientSize input
    refineBounds? budget fun degree precision =>
      (logarithmBounds input degree precision).map Subtype.val

/-- A successful logarithm replay retains its enclosure and both endpoint roundings.
No finiteness or rounding law for an arbitrary codec is inferred from mere decoding success. -/
public theorem enclosedLogarithm?_eq_some {α : Type} [ops : QuotientArithmetic α]
    (input : QuotientCoefficients ops.depth) {result : α}
    (accepted : enclosedLogarithm? input = some result) :
    ∃ bounds, Related (Interval.ContainsReal some) bounds
        (logarithm Real.log (map (fun q : Rat => (q : ℝ)) input)) ∧
      ops.decode (map (·.lo) bounds) = some result ∧
      ∃ upper shared, ops.decode (map (·.hi) bounds) = some upper ∧
        ops.encode result = some shared ∧ ops.encode upper = some shared := by
  unfold enclosedLogarithm? at accepted
  split at accepted
  · cases accepted
  · obtain ⟨degree, precision, bounds, enclosed, rounded⟩ :=
      refineBounds_search_eq_some _ _ _ _ _ accepted
    obtain ⟨certified, certifiedAt, same⟩ := Option.map_eq_some_iff.mp enclosed
    cases same
    exact ⟨certified.val, certified.property, roundBounds?_eq_some rounded⟩

private def directionalReplay? {α : Type} [ops : QuotientArithmetic α]
    (base exponent baseDirection exponentDirection : QuotientCoefficients ops.depth) : Option α :=
  if base.primal ≤ 0 then none else
    let budget := 64 + (ops.depth + 1) *
      (coefficientSize base + coefficientSize exponent +
        coefficientSize baseDirection + coefficientSize exponentDirection)
    refineBounds? budget fun degree precision =>
      (directionalBounds base exponent baseDirection exponentDirection
        degree precision budget).map Subtype.val

/-- Refine finite coefficient enclosures until the original codec rounds every pair equally.

A coarse exponential bound first checks for unavoidable underflow of the complete coefficient
tree. Otherwise the degree/precision choice follows FloatLib's enclosure refinement policy,
applied to the widest coefficient. Both searches are bounded and may be inconclusive. A missing
finite encoding is not a certificate; primal-only equality of a dual number never accepts
differing derivative bounds.
-/
@[noinline] public def enclosedSlope? {α : Type} [ops : QuotientArithmetic α]
    (x : α) (seed : Option α) (isTanh : Bool) : Option α := do
  let input ← ops.encode x
  let upstream ← match seed with
    | none => some (QuotientCoefficients.ofScalar ops.depth (1 : Rat))
    | some value => ops.encode value
  -- Cap the power by a bit budget, not the potentially enormous magnitude of a finite input.
  let magnitude := if input.primal > 0 then input.primal else -input.primal
  let limit := Nat.floor ((if isTanh then 2 else 1) * magnitude)
  let rec underflow (bits : Nat) : Nat → Option α
    | 0 => none
    | steps + 1 => do
      let certified ← slopeBounds input upstream isTanh (tailExp? bits)
      let bounds := certified.val
      match roundBounds? bounds with
      | some value => some value
      | none => if bits < limit then underflow (min limit (2 * bits)) steps else none
  let rec search (degree precision : Nat) (previousWidth : Option Rat) : Nat → Option α
    | 0 => none
    | steps + 1 => do
      let certified ← slopeBounds input upstream isTanh (gridExp degree precision)
      let bounds := certified.val
      match roundBounds? bounds with
      | some value => some value
      | none =>
        let width := boundsWidth bounds
        if previousWidth.any (fun previous => decide (previous ≤ 4 * width)) then
          search degree (max 1 (2 * precision)) none steps
        else
          search (2 * degree) (precision + 1) (some width) steps
  -- Start above the seed's scale, rather than discarding its eventual rescaling effect.
  let precision := 64 + coefficientBits upstream + ops.depth * coefficientBits input
  (underflow (min limit precision) 16).orElse fun _ => search 8 precision none 16

private theorem enclosedSlope_underflow_eq_some {α : Type} [ops : QuotientArithmetic α]
    (isTanh : Bool) (input upstream : QuotientCoefficients ops.depth)
    (limit bits steps : Nat) {result : α}
    (accepted : enclosedSlope?.underflow isTanh input upstream limit bits steps = some result) :
    ∃ bounds, Related (Interval.ContainsReal some) bounds (idealSlope input upstream isTanh) ∧
      roundBounds? bounds = some result := by
  induction steps generalizing bits with
  | zero =>
      simp only [enclosedSlope?.underflow] at accepted
      cases accepted
  | succ steps ih =>
      simp only [enclosedSlope?.underflow] at accepted
      obtain ⟨certified, _, accepted⟩ := Option.bind_eq_some_iff.mp accepted
      cases rounded : roundBounds? certified.val with
      | some value =>
          simp only [rounded] at accepted
          have resultEqual := Option.some.inj accepted
          subst result
          exact ⟨certified.val, certified.property, rounded⟩
      | none =>
          simp only [rounded] at accepted
          split at accepted
          · exact ih _ accepted
          · cases accepted

private theorem enclosedSlope_search_eq_some {α : Type} [ops : QuotientArithmetic α]
    (isTanh : Bool) (input upstream : QuotientCoefficients ops.depth)
    (degree precision : Nat) (previousWidth : Option Rat) (steps : Nat) {result : α}
    (accepted : enclosedSlope?.search isTanh input upstream degree precision previousWidth steps =
      some result) :
    ∃ bounds, Related (Interval.ContainsReal some) bounds (idealSlope input upstream isTanh) ∧
      roundBounds? bounds = some result := by
  induction steps generalizing degree precision previousWidth with
  | zero =>
      simp only [enclosedSlope?.search] at accepted
      cases accepted
  | succ steps ih =>
      simp only [enclosedSlope?.search] at accepted
      obtain ⟨certified, _, accepted⟩ := Option.bind_eq_some_iff.mp accepted
      cases rounded : roundBounds? certified.val with
      | some value =>
          simp only [rounded] at accepted
          have resultEqual := Option.some.inj accepted
          subst result
          exact ⟨certified.val, certified.property, rounded⟩
      | none =>
          simp only [rounded] at accepted
          split at accepted <;> exact ih _ _ _ accepted

/-- A successful recovery returns a rounded lower endpoint of a certified enclosure.
Both endpoint roundings have the same complete coefficient encoding. The witnesses retain the
original input and seed conversions; numerical interpretation uses the selected codec's laws. -/
public theorem enclosedSlope?_eq_some {α : Type} [ops : QuotientArithmetic α]
    (x : α) (seed : Option α) (isTanh : Bool) {result : α}
    (accepted : enclosedSlope? x seed isTanh = some result) :
    ∃ (input upstream : QuotientCoefficients ops.depth)
        (bounds : QuotientCoefficients ops.depth (Interval Rat)),
      ops.encode x = some input ∧
      (match seed with
        | none => some (ofScalar ops.depth (1 : Rat))
        | some value => ops.encode value) = some upstream ∧
      Related (Interval.ContainsReal some) bounds (idealSlope input upstream isTanh) ∧
      ops.decode (map (·.lo) bounds) = some result ∧
      ∃ upper shared, ops.decode (map (·.hi) bounds) = some upper ∧
        ops.encode result = some shared ∧ ops.encode upper = some shared := by
  unfold enclosedSlope? at accepted
  obtain ⟨input, inputEncoded, accepted⟩ := Option.bind_eq_some_iff.mp accepted
  let replay := fun upstream : QuotientCoefficients ops.depth =>
    let magnitude := if input.primal > 0 then input.primal else -input.primal
    let limit := Nat.floor ((if isTanh then 2 else 1) * magnitude)
    let precision := 64 + coefficientBits upstream + ops.depth * coefficientBits input
    (enclosedSlope?.underflow isTanh input upstream limit (min limit precision) 16).orElse
      fun _ => enclosedSlope?.search isTanh input upstream 8 precision none 16
  have replayed : (match seed with
      | none => some (ofScalar ops.depth (1 : Rat))
      | some value => ops.encode value).bind replay = some result := by
    cases seed <;> exact accepted
  obtain ⟨upstream, upstreamEncoded, accepted⟩ := Option.bind_eq_some_iff.mp replayed
  have enclosed : ∃ bounds,
      Related (Interval.ContainsReal some) bounds (idealSlope input upstream isTanh) ∧
        roundBounds? bounds = some result := by
    rcases (Option.orElse_eq_some _ _ _).mp accepted with returned | ⟨_, returned⟩
    · exact enclosedSlope_underflow_eq_some _ _ _ _ _ _ returned
    · exact enclosedSlope_search_eq_some _ _ _ _ _ _ _ returned
  obtain ⟨bounds, contained, rounded⟩ := enclosed
  obtain ⟨decoded, upper, shared, upperDecoded, resultEncoded, upperEncoded⟩ :=
    roundBounds?_eq_some rounded
  refine ⟨input, upstream, bounds, inputEncoded, ?_, contained, decoded,
    upper, shared, upperDecoded, resultEncoded, upperEncoded⟩
  cases seed <;> exact upstreamEncoded

/-- Refine the complete weighted power coefficient tree for a fixed outer exponent direction.

An exponent constant in every encoded direction uses the polynomial rule, including supported
nonpositive bases. An exponent varying in inner directions uses the joint positive-base rule with
zero outer exponent direction. Every lower and upper coefficient must round alike through the codec.
The search is bounded; unsupported bases, failed finite conversion, or inconclusive bounds return
`none`. The interval proofs concern `idealPower` and `idealPowerTangent`, not the codec or a
native backend.
Logarithms and exponentials use rounded binary-grid arithmetic. The exponential argument must fit
a work budget derived from the original coefficient sizes before its enclosure is evaluated.
-/
@[noinline] public def enclosedPowerSlope? {α : Type} [ops : QuotientArithmetic α]
    (x exponent seed : α) : Option α := do
  let input ← ops.encode x
  let exponent ← ops.encode exponent
  let upstream ← ops.encode seed
  if exponent != QuotientCoefficients.ofScalar ops.depth exponent.primal then
    directionalReplay? input exponent upstream (QuotientCoefficients.ofScalar ops.depth 0)
  else
    let budget := 64 + (ops.depth + 1) *
      (coefficientSize input + coefficientSize exponent + coefficientSize upstream)
    refineBounds? budget fun degree precision =>
      (powerBounds input upstream exponent.primal degree precision budget).map Subtype.val

/-- Enclose a joint power tangent from the original positive base, exponent, and both directions.

The exponential, logarithm, and complete product/quotient expression use the existing coefficient
tree. Finite conversion and a positive base primal are required. Every coefficient's rounded
endpoints must agree; bounded work or an unsupported domain returns `none`, not a certificate.
This retains all varying-exponent mixed coefficients without treating them as constants.
-/
@[noinline] public def enclosedPowerTangent? {α : Type} [ops : QuotientArithmetic α]
    (x y dx dy : α) : Option α := do
  let base ← ops.encode x
  if base.primal ≤ 0 then none else do
    let exponent ← ops.encode y
    let baseDirection ← ops.encode dx
    let exponentDirection ← ops.encode dy
    directionalReplay? base exponent baseDirection exponentDirection

end TorchLean.Numeric.Internal

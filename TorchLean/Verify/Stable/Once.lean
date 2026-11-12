/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.Stable.Certificate
import TorchLean.Verify.RuntimeApprox.InverseTrigonometric

/-!
# Programs that round once

A target (`Target n`) is a real function `F` of `n` inputs on a domain `D`, with a comparator that
compares `F` at every rational point of `D` with every rational. In a configured binary format,
`t.once d x` rounds the target once at the finite inputs `x` of `D` through `Elementary.once`, and
`t.prog` is the program that returns it. A program computes the target (`Target.Computes`) when it
reads as `F` over ℝ on `D` and, at every finite input of `D` in an IEEE format, its result has the
real value of `t.once d x` where that is finite and is `t.once d x` elsewhere. `t.prog` computes
`t` (`computes_prog`).

In every IEEE binary format, a program that computes a target
* at the finite inputs of `D`, returns the nearest-even rounding of `F` wherever `|F| < Θ`
  (`Computes.roundsOnce`), so a finite zero wherever `F` is zero (`Computes.zeroExact`), and the
  infinity of the sign of `F` wherever `Θ ≤ |F|` (`Computes.overflows`);
* covers `D` with no gap: every input is a zero, below, above or band input with the outcome of
  its part (`Computes.covers`);
* has the certificate of `Once.claim` on `D`, with budget `(u, max 0 (η - u * |F|))` and band
  `(η, Θ)`, given a point of `D` with coordinates in `{-1, 0, 1}` where `1/4 < |F| ≤ 1`
  (`Computes.cert`). Such a point is a finite input of every IEEE format
  (`Binary.ofSign_spec`), and its answer lies in the band (`mem_band`).

In every configured binary format, `t.prog` returns `+0` at the finite inputs of `D` where `F` is
zero (`prog_zero`).

The repair `t.repair p` of a program `p` returns the result of `p` where it is finite and not zero
(`repair_eq`), and the result of `t.prog` where it is zero or not finite (`repair_eq_prog`). In
every IEEE binary format, its result is finite and not zero at the finite inputs of `D` where `|F|`
lies in the band `(η, Θ)` (`repair_finite`), and at every finite input of `D` it is rejected
exactly where `p` is rejected with a result that is finite and not zero (`rejected_repair_iff`).

Targets come from functions that are rational at the rational points of the domain (`ofRat`) and
from ratios `(α * exp r + β) / (γ * exp r + δ)` whose coefficients and exponent are rational at the
rational points (`ofExp`). The reciprocal square root `(√x)⁻¹` on `x > 0` is a target (`rsqrt`), and
FloatLib's `rsqrt` computes it (`computes_rsqrt`), with a certificate (`cert_rsqrt`). The floor of a
quotient `⌊a / b⌋` on `b ≠ 0` is a rational target (`floorDiv`). Floor division rounded once
(`floorDivProg`) returns `floorDiv.once`, except that a zero result is the zero signed by the
exclusive or of the sign bits of `a` and `b`, as IEEE 754 signs `a / b`. It computes the target
(`computes_floorDiv`), with a certificate (`cert_floorDiv`), and in every configured binary format
it returns that signed zero at the finite inputs where `b ≠ 0` and `⌊a / b⌋ = 0`
(`floorDivProg_zero`).

The results hold for every target. Those on `Computes` hold for every program that computes it,
and those on `repair` for every program.
-/

@[expose] public section

namespace TorchLean.Stable

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange
open FloatLib.Floats.Formats.BinaryInterchange.Model.Transcendentals.Certified (overflowThreshold)
open Arith (Val isFinite toReal valid)

variable {n : ℕ}

/-- The value `-1`, `0` or `1` of a configured binary format, by a sign. -/
def Binary.ofSign (d : Binary.Desc) : SignType → Val d
  | .zero => (0 : ExecFloat (Configured.Family d.format d.code d.plan))
  | .neg => (-1 : ExecFloat (Configured.Family d.format d.code d.plan))
  | .pos => (1 : ExecFloat (Configured.Family d.format d.code d.plan))

/-- In an IEEE format the value of a sign is finite and denotes the sign. -/
theorem Binary.ofSign_spec {d : Binary.Desc} (hd : valid d) (s : SignType) :
    isFinite (Binary.ofSign d s) = true ∧ toReal (Binary.ofSign d s) = s := by
  have h1 := ExecFloat.Binary.isFinite_one (format := d.format) (plan := d.plan) (code := d.code) hd
  have h2 := ExecFloat.Binary.toReal_one (format := d.format) (plan := d.plan) (code := d.code) hd
  cases s
  · exact ⟨ExecFloat.Binary.isFinite_zero, ExecFloat.Binary.toReal_zero⟩
  · refine ⟨(ExecFloat.Binary.isFinite_neg _).trans h1, ?_⟩
    show Model.toReal (ExecFloat.Binary.toModel
      (-1 : ExecFloat (Configured.Family d.format d.code d.plan))) = -1
    rw [ExecFloat.Binary.toModel_neg, Model.toReal_neg _ h1]
    exact congrArg Neg.neg h2
  · exact ⟨h1, h2⟩

/-- In an IEEE format every magnitude in `(1/4, 1]` lies in the band `(η, Θ)`. -/
theorem mem_band {d : Binary.Desc} (hd : valid d) {y : ℝ} (h1 : 1 / 4 < |y|) (h3 : |y| ≤ 1) :
    |y| ∈ Once.band d := by
  have hη : Model.underflowErrorAt d.format ≤ (2 : ℝ) ^ (-2 : ℤ) :=
    Model.underflowErrorAt_le_zpow d.format (by
      have := d.format.exponentBias_pos
      have := d.format.fracWidth_pos
      simp only [FloatFormat.minSubnormalExponent, FloatFormat.minNormalExponent,
        Int.ofNat_eq_natCast]
      omega)
  have hΘ := Model.Transcendentals.CorrectlyRounded.one_lt_overflowThreshold (fmt := d.format) hd
  have h4 : (2 : ℝ) ^ (-2 : ℤ) = 1 / 4 := by norm_num
  exact ⟨(hη.trans_eq h4).trans_lt h1, h3.trans_lt hΘ⟩

/-- A real function of `n` inputs on a domain, with a comparator of its values at the rational
points of the domain. -/
structure Target (n : ℕ) where
  /-- The real function. -/
  F : (Fin n → ℝ) → ℝ
  /-- The domain. -/
  D : Set (Fin n → ℝ)
  /-- The comparator: at a rational point `q` of the domain, `compare q` compares `F q` with
  rationals. -/
  compare : (Fin n → ℚ) → ℚ → Ordering
  /-- At the rational points of the domain, `compare` compares `F` with rationals. -/
  compare_eq : ∀ q : Fin n → ℚ, (fun i => (q i : ℝ)) ∈ D →
    ∀ b : ℚ, compare q b = cmp (F fun i => (q i : ℝ)) (b : ℝ)

namespace Target

variable (t : Target n)

/-- `Elementary.once` of the comparator at the inputs `x` of format `d`: at finite inputs whose
real values lie in the domain, the target rounded once to nearest even (`once_eq`), and the invalid
result when an input is not finite. -/
def once (d : Binary.Desc) (x : Fin n → Val d) : Model d.format :=
  Elementary.once t.compare d.format fun i => Binary.toModel (x i)

/-- The program that returns `t.once`, the target rounded once at the finite inputs of the
domain. -/
noncomputable def prog : Prog Binary.Desc n where
  real := t.F
  float d x := Binary.ofModel (t.once d x)

/-- `p` computes the target: it reads as `F` over ℝ on the domain, and at every finite input of
the domain in an IEEE format its result has the real value of `t.once` where that is finite and
is `t.once` elsewhere. -/
structure Computes (p : Prog Binary.Desc n) : Prop where
  /-- The program reads as `F` over ℝ on the domain. -/
  real : ∀ v ∈ t.D, p.real v = t.F v
  /-- Where `t.once` is finite, the result is finite with its real value. -/
  finite : ∀ d : Binary.Desc, valid d → ∀ x ∈ Inputs (fun _ => t.D) d,
    Model.isFinite (t.once d x) = true →
      isFinite (p.float d x) = true ∧ toReal (p.float d x) = Model.toReal (t.once d x)
  /-- Where `t.once` is not finite, the result is `t.once`. -/
  infinite : ∀ d : Binary.Desc, valid d → ∀ x ∈ Inputs (fun _ => t.D) d,
    Model.isFinite (t.once d x) = false → Binary.toModel (p.float d x) = t.once d x

/-- The program that returns `t.once` computes the target. -/
theorem computes_prog : t.Computes t.prog where
  real _ _ := rfl
  finite d _ x _ h := by
    have he : Binary.toModel (t.prog.float d x) = t.once d x := ExecFloat.Binary.toModel_ofModel _
    show Model.isFinite (Binary.toModel (t.prog.float d x)) = true ∧
      Model.toReal (Binary.toModel (t.prog.float d x)) = _
    rw [he]
    exact ⟨h, rfl⟩
  infinite _ _ _ _ _ := ExecFloat.Binary.toModel_ofModel _

/-- At a finite input of the domain, `t.once` is `roundSigned` of a comparator of the answer. -/
theorem once_eq {d : Binary.Desc} {x : Fin n → Val d} (hx : x ∈ Inputs (fun _ => t.D) d) :
    ∃ c : ℚ → Ordering, (∀ b : ℚ, c b = cmp (t.F (toReal ∘ x)) (b : ℝ)) ∧
      t.once d x = Elementary.roundSigned d.format c := by
  have hq : (fun i => (Elementary.rats (fun i => Binary.toModel (x i)) i : ℝ)) = toReal ∘ x :=
    funext (Elementary.rats_cast (x := fun i => Binary.toModel (x i)) hx.1)
  refine ⟨t.compare (Elementary.rats fun i => Binary.toModel (x i)), fun b => ?_,
    Elementary.once_eq t.compare hx.1⟩
  rw [t.compare_eq (Elementary.rats fun i => Binary.toModel (x i)) (hq ▸ hx.2) b, hq]

/-- At a finite input of the domain where `F` is zero, `t.once` is `+0`. -/
theorem once_zero {d : Binary.Desc} {x : Fin n → Val d} (hx : x ∈ Inputs (fun _ => t.D) d)
    (h0 : t.F (toReal ∘ x) = 0) : t.once d x = Model.zero d.format false := by
  obtain ⟨c, hc, he⟩ := t.once_eq hx
  have hs : decide ((0 : ℝ) < 0) = false := decide_eq_false (lt_irrefl 0)
  rw [he, Elementary.roundSigned_eq_round hc, h0, abs_zero, hs]
  exact InverseTrigonometric.round_cmp_zero false

/-- At a finite input of the domain where `F` is zero, the program that rounds the target once
returns `+0`. -/
theorem prog_zero {d : Binary.Desc} {x : Fin n → Val d} (hx : x ∈ Inputs (fun _ => t.D) d)
    (h0 : t.F (toReal ∘ x) = 0) : Binary.toModel (t.prog.float d x) = Model.zero d.format false :=
  (ExecFloat.Binary.toModel_ofModel _).trans (t.once_zero hx h0)

variable {t} {p : Prog Binary.Desc n}

/-- In every IEEE format, at the finite inputs of the domain where `|F| < Θ`, a program that
computes a target returns the nearest-even rounding of `F`. -/
theorem Computes.roundsOnce (h : t.Computes p) : RoundsOnce p t.F {d | valid d} fun _ => t.D := by
  intro d hd x hx hfin
  have hd' : d.format.isIEEE = true := hd
  obtain ⟨c, hc, he⟩ := t.once_eq hx
  obtain ⟨h1, h2⟩ := Elementary.roundSigned_eq_roundAt hd' hc hfin
  rw [← he] at h1 h2
  obtain ⟨h3, h4⟩ := h.finite d hd x hx h1
  exact ⟨h3, h4.trans h2⟩

/-- In every IEEE format, at the finite inputs of the domain where `Θ ≤ |F|`, a program that
computes a target returns the infinity of the sign of `F`. -/
theorem Computes.overflows (h : t.Computes p) : Overflows p t.F {d | valid d} fun _ => t.D := by
  intro d hd x hx hfin
  have hd' : d.format.isIEEE = true := hd
  obtain ⟨c, hc, he⟩ := t.once_eq hx
  have hΘ : (overflowThreshold d.format : ℝ) ≤ |t.F (toReal ∘ x)| := not_lt.mp hfin
  have ho := Elementary.once_above hd' hc hΘ
  rw [← he] at ho
  have hinf : Model.isFinite (t.once d x) = false := by
    rw [ho]
    exact Model.isFinite_nativeOverflow hd' _
  exact (h.infinite d hd x hx hinf).trans ho

/-- In every IEEE format, at the finite inputs of the domain where `F` is zero, a program that
computes a target returns a finite zero. -/
theorem Computes.zeroExact (h : t.Computes p) : ZeroExact p t.F {d | valid d} fun _ => t.D :=
  h.roundsOnce.zeroExact fun _ hd => hd

/-- In an IEEE format, a program that computes a target covers its domain with no gap. -/
theorem Computes.covers (h : t.Computes p) {d : Binary.Desc} (hd : valid d) :
    Covers p t.F d t.D (Once.claim t.F fun _ => t.D) ∅ := by
  have hc := Covers.ofRoundsOnce hd
    (fun g hg => h.roundsOnce g (by rw [Set.mem_singleton_iff.mp hg]; exact hd))
    (fun g hg => h.overflows g (by rw [Set.mem_singleton_iff.mp hg]; exact hd)) h.real
  have h0 : {v ∈ t.D | |t.F v| ∈ Once.holes d} = ∅ :=
    Set.eq_empty_of_forall_notMem fun v hv => Set.notMem_empty _ hv.2
  rwa [h0] at hc

/-- A program that computes a target has the certificate of `Once.claim` on the domain in every
IEEE binary format, given a point of the domain with coordinates in `{-1, 0, 1}` where
`1/4 < |F| ≤ 1`. -/
theorem Computes.cert (h : t.Computes p) (s : Fin n → SignType)
    (hD : (fun i => (s i : ℝ)) ∈ t.D) (h1 : 1 / 4 < |t.F fun i => (s i : ℝ)|)
    (h3 : |t.F fun i => (s i : ℝ)| ≤ 1) :
    Cert p t.F {d | valid d} (Once.claim t.F fun _ => t.D) :=
  Cert.ofRoundsOnce h.roundsOnce (fun _ _ => h.real)
    ⟨⟨Binary.binary32, FloatFormat.isIEEE_binary32⟩, fun _ hd => hd⟩ fun d hd => by
      have hx : toReal ∘ (fun i => Binary.ofSign d (s i)) = fun i => (s i : ℝ) :=
        funext fun i => (Binary.ofSign_spec hd (s i)).2
      refine ⟨fun i => Binary.ofSign d (s i), ⟨fun i => (Binary.ofSign_spec hd (s i)).1, ?_⟩, ?_⟩
      · rw [hx]
        exact hD
      · rw [hx]
        exact mem_band hd h1 h3

/-! ## Repair -/

variable (t)

/-- The program that returns the result of `p` where it is finite and not zero, and the result of
`t.prog`, the target rounded once, where it is zero or not finite. -/
noncomputable def repair (p : Prog Binary.Desc n) : Prog Binary.Desc n where
  real := p.real
  float d x :=
    if isFinite (p.float d x) && !Model.isZero (Binary.toModel (p.float d x)) then p.float d x
    else t.prog.float d x

/-- Where the result of `p` is finite and not zero, `t.repair p` returns it. -/
theorem repair_eq {d : Binary.Desc} {x : Fin n → Val d} (hf : isFinite (p.float d x) = true)
    (h0 : toReal (p.float d x) ≠ 0) : (t.repair p).float d x = p.float d x := by
  have hz : Model.isZero (Binary.toModel (p.float d x)) = false :=
    Bool.eq_false_iff.mpr fun hz => h0 (Model.toReal_eq_zero_of_isZero _ hz)
  simp [repair, hf, hz]

/-- Where the result of `p` is zero or not finite, `t.repair p` returns the target rounded once. -/
theorem repair_eq_prog {d : Binary.Desc} {x : Fin n → Val d}
    (h : isFinite (p.float d x) = true → toReal (p.float d x) = 0) :
    (t.repair p).float d x = t.prog.float d x := by
  cases hf : isFinite (p.float d x)
  · simp [repair, hf]
  · have hz : Model.isZero (Binary.toModel (p.float d x)) = true :=
      (Model.isZero_eq_true_iff_toReal_eq_zero (Binary.toModel (p.float d x)) hf).mpr (h hf)
    simp [repair, hf, hz]

/-- In every IEEE binary format, at the finite inputs of the domain where `|F|` lies in the band
`(η, Θ)`, the result of `t.repair p` is finite and not zero, for every program `p`. -/
theorem repair_finite (p : Prog Binary.Desc n) {d : Binary.Desc} (hd : valid d)
    {x : Fin n → Val d} (hx : x ∈ Inputs (fun _ => t.D) d)
    (hb : |t.F (toReal ∘ x)| ∈ Once.band d) :
    isFinite ((t.repair p).float d x) = true ∧ toReal ((t.repair p).float d x) ≠ 0 := by
  by_cases h : isFinite (p.float d x) = true ∧ toReal (p.float d x) ≠ 0
  · rw [t.repair_eq h.1 h.2]
    exact h
  · rw [t.repair_eq_prog fun hf => not_not.mp fun h0 => h ⟨hf, h0⟩]
    obtain ⟨h1, h2, -⟩ := (t.computes_prog.covers hd).band.excludes (Set.mem_singleton d) hx hb
    exact ⟨h1, left_ne_zero_of_mul h2.ne'⟩

/-- In every IEEE binary format, at the finite inputs of the domain, `t.repair p` is rejected
exactly where `p` is rejected with a result that is finite and not zero. -/
theorem rejected_repair_iff {d : Binary.Desc} (hd : valid d) {x : Fin n → Val d}
    (hx : x ∈ Inputs (fun _ => t.D) d) :
    Rejected (t.repair p) t.F d x ↔
      Rejected p t.F d x ∧ isFinite (p.float d x) = true ∧ toReal (p.float d x) ≠ 0 := by
  by_cases h : isFinite (p.float d x) = true ∧ toReal (p.float d x) ≠ 0
  · have he : ∀ q : Prog Binary.Desc n, q.real = p.real → q.float d x = p.float d x →
        (Rejected q t.F d x ↔ Rejected p t.F d x) := fun q hr hf => by
      unfold Rejected
      rw [hr, hf]
    rw [he (t.repair p) rfl (t.repair_eq h.1 h.2)]
    exact ⟨fun hr => ⟨hr, h⟩, And.left⟩
  · refine iff_of_false (fun hr => ?_) fun hr => h hr.2
    obtain ⟨h1, -, h3, h4⟩ := hr
    rw [t.repair_eq_prog fun hf => not_not.mp fun h0 => h ⟨hf, h0⟩] at h4
    exact (t.computes_prog.covers hd).band.not_rejected (Set.mem_singleton d) hx h3
      ⟨h1, rfl, h3, h4⟩

end Target

/-! ## Targets -/

namespace Target

/-- The target of a function whose value at each rational point `q` of `D` is the rational
`val q`. -/
def ofRat (F : (Fin n → ℝ) → ℝ) (D : Set (Fin n → ℝ)) (val : (Fin n → ℚ) → ℚ)
    (hF : ∀ q : Fin n → ℚ, (fun i => (q i : ℝ)) ∈ D → F (fun i => (q i : ℝ)) = val q) :
    Target n where
  F := F
  D := D
  compare q b := cmp (val q) b
  compare_eq q hq b := by
    rw [hF q hq]
    exact (Rat.cast_strictMono.cmp_map_eq _ _).symm

/-- The target of a function that is `(α * exp r + β) / (γ * exp r + δ)` at the rational points
of `D`, with coefficients and exponent that are rational at each rational point. -/
def ofExp (F : (Fin n → ℝ) → ℝ) (D : Set (Fin n → ℝ)) (α β γ δ r : (Fin n → ℚ) → ℚ)
    (hF : ∀ q : Fin n → ℚ, (fun i => (q i : ℝ)) ∈ D → F (fun i => (q i : ℝ)) =
      ((α q : ℝ) * Real.exp (r q) + β q) / ((γ q : ℝ) * Real.exp (r q) + δ q)) :
    Target n where
  F := F
  D := D
  compare q := Elementary.expRatioCompare (α q) (β q) (γ q) (δ q) (r q)
  compare_eq q hq b := by
    rw [hF q hq]
    exact Elementary.expRatioCompare_eq _ _ _ _ _ b

/-- The reciprocal square root `(√x)⁻¹` on `x > 0`, compared through exact squares. -/
noncomputable def rsqrt : Target 1 where
  F v := (Real.sqrt (v 0))⁻¹
  D := {v | 0 < v 0}
  compare q := Formats.Posit.Model.RootRounding.compareRoot (q 0)⁻¹ 2
  compare_eq q hq b := by
    have h0 : (0 : ℝ) < q 0 := hq
    show Formats.Posit.Model.RootRounding.compareRoot (q 0)⁻¹ 2 b = cmp (Real.sqrt (q 0))⁻¹ (b : ℝ)
    refine Formats.Posit.Model.RootRounding.compareRoot_eq_real _ 2 _
      (inv_nonneg.mpr (Real.sqrt_nonneg _)) two_ne_zero ?_ b
    rw [inv_pow, Real.sq_sqrt h0.le, Rat.cast_inv]

/-- FloatLib's reciprocal square root, as a program of one input. -/
noncomputable def rsqrtProg : Prog Binary.Desc 1 where
  real v := (Real.sqrt (v 0))⁻¹
  float d x := ExecFloat.Binary.rsqrt (format := d.format) (plan := d.plan) (code := d.code) (x 0)

/-- FloatLib's `rsqrt` computes the reciprocal square root on `x > 0`. -/
theorem computes_rsqrt : rsqrt.Computes rsqrtProg := by
  have he : ∀ d : Binary.Desc, ∀ x ∈ Inputs (fun _ => rsqrt.D) d,
      Binary.toModel (rsqrtProg.float d x) = rsqrt.once d x := by
    intro d x hx
    have hpos : 0 < Model.toReal (Binary.toModel (x 0)) := hx.2
    refine (ExecFloat.Binary.toModel_rsqrt _).trans ?_
    rw [Elementary.rsqrt_eq_roundSigned (hx.1 0) hpos]
    exact (Elementary.once_eq rsqrt.compare hx.1).symm
  refine ⟨fun _ _ => rfl, fun d _ x hx h => ?_, fun d _ x hx _ => he d x hx⟩
  show Model.isFinite (Binary.toModel (rsqrtProg.float d x)) = true ∧
    Model.toReal (Binary.toModel (rsqrtProg.float d x)) = _
  rw [he d x hx]
  exact ⟨h, rfl⟩

/-- FloatLib's `rsqrt` has the certificate of `Once.claim` on `x > 0` in every IEEE binary
format. -/
theorem cert_rsqrt :
    Cert rsqrtProg rsqrt.F {d | valid d} (Once.claim rsqrt.F fun _ => rsqrt.D) := by
  refine computes_rsqrt.cert (fun _ => .pos) ?_ ?_ ?_ <;> norm_num [rsqrt]

/-- The floor of a quotient `⌊a / b⌋` on `b ≠ 0`, a rational target. -/
noncomputable def floorDiv : Target 2 :=
  ofRat (fun v => (⌊v 0 / v 1⌋ : ℝ)) {v | v 1 ≠ 0} (fun q => (⌊q 0 / q 1⌋ : ℚ)) fun q _ => by
    show (⌊(q 0 : ℝ) / q 1⌋ : ℝ) = ((⌊q 0 / q 1⌋ : ℚ) : ℝ)
    rw [← Rat.cast_div, Rat.floor_cast, Rat.cast_intCast]

/-- Floor division rounded once: `floorDiv.once`, except that a zero result is the zero signed by
the exclusive or of the sign bits of `a` and `b`, as IEEE 754 signs `a / b`. A format without a
signed zero returns its one zero. -/
noncomputable def floorDivProg : Prog Binary.Desc 2 where
  real := floorDiv.F
  float d x := Binary.ofModel <|
    if Model.isZero (floorDiv.once d x) then
      Model.zero d.format
        (Bool.xor (Model.signBit (Binary.toModel (x 0))) (Model.signBit (Binary.toModel (x 1))))
    else floorDiv.once d x

/-- Floor division rounded once computes the floor of a quotient on `b ≠ 0`. -/
theorem computes_floorDiv : floorDiv.Computes floorDivProg := by
  have he : ∀ d x, Binary.toModel (floorDivProg.float d x) =
      if Model.isZero (floorDiv.once d x) then
        Model.zero d.format
          (Bool.xor (Model.signBit (Binary.toModel (x 0))) (Model.signBit (Binary.toModel (x 1))))
      else floorDiv.once d x :=
    fun _ _ => ExecFloat.Binary.toModel_ofModel _
  refine ⟨fun _ _ => rfl, fun d _ x _ h => ?_, fun d _ x _ h => ?_⟩
  · show Model.isFinite (Binary.toModel (floorDivProg.float d x)) = true ∧
      Model.toReal (Binary.toModel (floorDivProg.float d x)) = _
    by_cases hz : Model.isZero (floorDiv.once d x) = true
    · rw [he, ite_eq_left hz]
      exact ⟨Model.isFinite_eq_true_of_isZero_eq_true _ (Model.isZero_zero _ _),
        (Model.toReal_zero _ _).trans (Model.toReal_eq_zero_of_isZero _ hz).symm⟩
    · rw [he, ite_eq_right hz]
      exact ⟨h, rfl⟩
  · have hz : ¬Model.isZero (floorDiv.once d x) = true := fun hz =>
      absurd (Model.isFinite_eq_true_of_isZero_eq_true _ hz) (by rw [h]; decide)
    exact (he d x).trans (ite_eq_right hz)

/-- In every configured binary format, at finite inputs `a` and `b` with `b ≠ 0` and
`⌊a / b⌋ = 0`, floor division rounded once returns the zero signed by the exclusive or of the sign
bits of `a` and `b`. -/
theorem floorDivProg_zero {d : Binary.Desc} {x : Fin 2 → Val d} (hx : ∀ i, isFinite (x i) = true)
    (hb : toReal (x 1) ≠ 0) (h0 : ⌊toReal (x 0) / toReal (x 1)⌋ = 0) :
    Binary.toModel (floorDivProg.float d x) = Model.zero d.format
      (Bool.xor (Model.signBit (Binary.toModel (x 0))) (Model.signBit (Binary.toModel (x 1)))) := by
  have hz : Model.isZero (floorDiv.once d x) = true := by
    rw [floorDiv.once_zero ⟨hx, hb⟩ (by
      show ((⌊toReal (x 0) / toReal (x 1)⌋ : ℤ) : ℝ) = 0
      rw [h0, Int.cast_zero])]
    exact Model.isZero_zero _ _
  exact (ExecFloat.Binary.toModel_ofModel _).trans (ite_eq_left hz)

/-- Floor division rounded once has the certificate of `Once.claim` on `b ≠ 0` in every IEEE
binary format. -/
theorem cert_floorDiv :
    Cert floorDivProg floorDiv.F {d | valid d} (Once.claim floorDiv.F fun _ => floorDiv.D) := by
  refine computes_floorDiv.cert (fun _ => .pos) ?_ ?_ ?_ <;> norm_num [floorDiv, ofRat]

end Target

end TorchLean.Stable

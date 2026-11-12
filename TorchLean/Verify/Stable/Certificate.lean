/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.Stable.Criterion
public import TorchLean.Tensor.ConfiguredBinary
public import TorchLean.Tensor.Context.Real
public import Mathlib.Basic.Real.Sign

/-!
# Certificates of informative programs

A number system (`Arith`) gives each format a value type, finiteness, a real reading and a
validity condition. A program (`Prog`) has a reading over ℝ and one in every format. A claim
(`Claim`) gives each format a domain of real inputs, a set `fin` of answers with finite results, a
band of scale values, a mixed budget `(r, a)` and a grade `κ`.

`Bounds p F m S c` says that in every format of `S` the program `p` computes `F` over ℝ on the
domain, that its result is finite and within the mixed budget of `F` wherever the answer lies in
`fin`, and that the budget is at most `κ * m` with `κ < 1` wherever the scale `m` lies in the
positive band, whose answers lie in `fin`. `CertTo` adds that `S` is a nonempty set of valid
formats, each with a finite input whose scale lies in the band. `Cert p F` measures band and grade
against `|F|`, and at an input whose answer lies in the band its result is finite, has the sign of
the answer and is within `κ * |F|` of it (`Bounds.excludes`).

A number system with a once-rounding (`Once`) carries its error theory: a mixed budget wherever the
rounding is finite, informative on a positive band, and no informative finite value below the band.
`RoundsOnce` says that a program returns the rounding of its answer wherever that rounding is
finite. With the real equality and a band witness it gives the certificate of `Once.claim`, whose
budget and grade are the rounding's (`Cert.ofRoundsOnce`). `Covers` splits the inputs of a domain
into zero, below, above, band and gap parts, with the outcomes of the zero, above and band parts,
and `RoundsOnce` with `Overflows` gives it with the gap of the inputs whose answers lie in a hole
(`Covers.ofRoundsOnce`).
A program is rejected at a finite input (`Rejected`) when it reads as the answer over ℝ there, the
answer's magnitude lies in the band, and the result is not finite or is no closer to the answer
than its magnitude. A result whose real value is zero or has the sign opposite to the answer is no
closer to it than its magnitude (`Rejected.of_mul_nonpos`). No input of a certificate's band is
rejected (`Bounds.not_rejected`).

The configured binary formats form a number system (`Binary.Desc`), valid at the IEEE
descriptors, with a once-rounding: nearest-even rounding, whose answers with finite rounding are
those below the overflow threshold `Θ` in magnitude, with the budget of `once_mixedError`, the band
`(η, Θ)` and no holes. In an IEEE format a program that reads as the
answer over ℝ is rejected at a finite input iff some finite value is closer to the answer than its
magnitude, the answer is below `Θ` in magnitude, and the result is no closer to the answer than
its magnitude (`Binary.rejected_iff`).
An answer at or above `Θ` in magnitude lies outside the band, so it is never rejected.
`Prog.ofContext` reads a program written against TorchLean's `Context` over ℝ and in every
configured binary format.

The generic results hold for every number system, program and claim, and the binary results for
every configured binary format, IEEE where stated.
-/

@[expose] public section

namespace TorchLean.Stable

open FloatLib.Numerics

universe u

/-- A number system: a value type per format, finiteness, a real reading, and the formats where
its theorems hold. -/
class Arith (A : Type u) where
  /-- The values of format `f`. -/
  Val : A → Type
  /-- Whether a value is finite. -/
  isFinite : {f : A} → Val f → Bool
  /-- The real number that a value denotes. -/
  toReal : {f : A} → Val f → ℝ
  /-- The formats where the theorems of the number system hold. -/
  valid : A → Prop

open Arith (Val isFinite toReal valid)

variable {A : Type u} {n : ℕ}

/-- A program with `n` inputs, read over ℝ and in every format of a number system. -/
structure Prog (A : Type u) [Arith A] (n : ℕ) where
  /-- The reading over ℝ. -/
  real : (Fin n → ℝ) → ℝ
  /-- The reading in format `f`. -/
  float : (f : A) → (Fin n → Val f) → Val f

/-- The finite inputs of format `f` whose real values lie in `D f`. -/
def Inputs [Arith A] (D : A → Set (Fin n → ℝ)) (f : A) : Set (Fin n → Val f) :=
  {x | (∀ i, isFinite (x i) = true) ∧ toReal ∘ x ∈ D f}

/-- What a certificate claims in each format: a domain, the answers whose results are finite, a
band of scale values, a mixed budget and a grade. -/
structure Claim (A : Type u) (n : ℕ) where
  /-- The domain of real inputs. -/
  D : A → Set (Fin n → ℝ)
  /-- The answers whose results are finite. -/
  fin : A → Set ℝ
  /-- The scale values where the budget is graded. -/
  band : A → Set ℝ
  /-- The relative part of the mixed budget. -/
  r : A → (Fin n → ℝ) → ℝ
  /-- The absolute part of the mixed budget. -/
  a : A → (Fin n → ℝ) → ℝ
  /-- The grade. -/
  κ : A → (Fin n → ℝ) → ℝ

/-- On the formats `S`, `p` computes `F` over ℝ on the domain, its result is finite and within the
mixed budget `(r, a)` of `F` wherever the answer lies in `fin`, and that budget is at most `κ * m`
with `κ < 1` wherever `m` lies in the positive set `band`, whose answers lie in `fin`. -/
structure Bounds [Arith A] (p : Prog A n) (F m : (Fin n → ℝ) → ℝ) (S : Set A) (c : Claim A n) :
    Prop where
  /-- The program computes `F` over ℝ on the domain. -/
  real : ∀ f ∈ S, ∀ v ∈ c.D f, p.real v = F v
  /-- The band is positive. -/
  pos : ∀ f ∈ S, ∀ y ∈ c.band f, 0 < y
  /-- An answer whose scale lies in the band lies in `fin`. -/
  sub : ∀ f ∈ S, ∀ v ∈ c.D f, m v ∈ c.band f → F v ∈ c.fin f
  /-- Where the answer lies in `fin`, the result is finite and within the mixed budget. -/
  bound : ∀ f ∈ S, ∀ x ∈ Inputs c.D f, F (toReal ∘ x) ∈ c.fin f →
    isFinite (p.float f x) = true ∧
      MixedError (toReal (p.float f x)) (F (toReal ∘ x)) (c.r f (toReal ∘ x)) (c.a f (toReal ∘ x))
  /-- Where the scale lies in the band, the budget is at most `κ * m`, with `κ < 1`. -/
  informative : ∀ f ∈ S, ∀ v ∈ c.D f, m v ∈ c.band f →
    c.r f v * |F v| + c.a f v ≤ c.κ f v * m v ∧ c.κ f v < 1

/-- `Bounds` on a nonempty set of valid formats, each with a finite input in the band. -/
structure CertTo [Arith A] (p : Prog A n) (F m : (Fin n → ℝ) → ℝ) (S : Set A) (c : Claim A n) :
    Prop extends Bounds p F m S c where
  /-- The formats form a nonempty set of valid formats. -/
  covers : S.Nonempty ∧ ∀ f ∈ S, valid f
  /-- Each format has a finite input of the domain whose scale lies in the band. -/
  witness : ∀ f ∈ S, ∃ x ∈ Inputs c.D f, m (toReal ∘ x) ∈ c.band f

/-- The informative certificate: band and grade are measured against `|F|`. -/
abbrev Cert [Arith A] (p : Prog A n) (F : (Fin n → ℝ) → ℝ) : Set A → Claim A n → Prop :=
  CertTo p F fun v => |F v|

section Arith

variable [Arith A] {p : Prog A n} {F m : (Fin n → ℝ) → ℝ} {S : Set A} {c : Claim A n}

/-- With the scale `|F|`, at an input whose answer lies in the band, the result is finite, has the
sign of the answer, and is within `κ * |F|` of it. -/
theorem Bounds.excludes (h : Bounds p F (fun v => |F v|) S c) {f : A} (hf : f ∈ S)
    {x : Fin n → Val f} (hx : x ∈ Inputs c.D f) (hb : |F (toReal ∘ x)| ∈ c.band f) :
    isFinite (p.float f x) = true ∧ 0 < toReal (p.float f x) * F (toReal ∘ x) ∧
      |toReal (p.float f x) - F (toReal ∘ x)| ≤ c.κ f (toReal ∘ x) * |F (toReal ∘ x)| := by
  obtain ⟨hfin, hmix⟩ := h.bound f hf x hx (h.sub f hf _ hx.2 hb)
  obtain ⟨ht, hκ⟩ := h.informative f hf _ hx.2 hb
  have ht' : Tight (c.κ f (toReal ∘ x)) (F (toReal ∘ x)) (c.r f (toReal ∘ x))
      (c.a f (toReal ∘ x)) := ht
  have hF : F (toReal ∘ x) ≠ 0 := abs_pos.mp (h.pos f hf _ hb)
  exact ⟨hfin, (informative_iff_excludes_sign hmix.r_nonneg hmix.a_nonneg).mp
    (ht'.informative hκ hF) _ hmix, ht'.abs_sub_le hmix⟩

end Arith

/-- The once-rounding of a number system with its error theory: a mixed budget wherever the
rounding is finite, informative on the band, and no informative finite value below it. -/
class Once (A : Type u) extends Arith A where
  /-- The once-rounding of a real answer, read as a real. -/
  round : A → ℝ → ℝ
  /-- The answers whose rounding is finite. -/
  fin : A → Set ℝ
  /-- The magnitudes where the rounding is informative. -/
  band : A → Set ℝ
  /-- The magnitudes where no finite value is informative. -/
  below : A → Set ℝ
  /-- The magnitudes outside zero, the band, `below` and the answers without finite rounding. -/
  holes : A → Set ℝ
  /-- The relative part of the rounding's mixed budget. -/
  r : A → ℝ → ℝ
  /-- The absolute part of the rounding's mixed budget. -/
  a : A → ℝ → ℝ
  /-- The largest finite magnitude. -/
  Ω : A → ℝ
  /-- The result for an answer without finite rounding, by the sign of the answer. -/
  overflow : (f : A) → Bool → Val f → Prop
  /-- The rounding has the mixed budget wherever it is finite. -/
  mixed : ∀ {f : A} {y : ℝ}, valid f → y ∈ fin f → MixedError (round f y) y (r f y) (a f y)
  /-- On the band the rounding is finite and its budget is informative. -/
  informative : ∀ {f : A} {y : ℝ}, valid f → |y| ∈ band f →
    y ∈ fin f ∧ Informative y (r f y) (a f y)
  /-- The band is positive. -/
  pos : ∀ {f : A} {y : ℝ}, valid f → y ∈ band f → 0 < y
  /-- Zero has a finite rounding, and it is zero. -/
  zero : ∀ {f : A}, valid f → 0 ∈ fin f ∧ round f 0 = 0
  /-- Every answer is zero, below, without finite rounding, in the band or in a hole. -/
  split : ∀ {f : A} {y : ℝ}, valid f →
    y = 0 ∨ |y| ∈ below f ∨ y ∉ fin f ∨ |y| ∈ band f ∨ |y| ∈ holes f
  /-- Below the band, no finite value is closer to the answer than its magnitude. -/
  no_informative : ∀ {f : A} {y : ℝ}, valid f → |y| ∈ below f → ∀ v : Val f,
    isFinite v = true → |y| ≤ |toReal v - y|
  /-- A finite value is at most `Ω` in magnitude. -/
  le_Ω : ∀ {f : A}, valid f → ∀ v : Val f, isFinite v = true → |toReal v| ≤ Ω f

section Once

variable [Once A]

/-- The claim of the once-rounding on the domain `D`: the format's `fin` and band, the rounding's
budget at the answer, and the grade `(r * |F| + a) / |F|`. -/
noncomputable def Once.claim (F : (Fin n → ℝ) → ℝ) (D : A → Set (Fin n → ℝ)) : Claim A n where
  D := D
  fin := Once.fin
  band := Once.band
  r f v := Once.r f (F v)
  a f v := Once.a f (F v)
  κ f v := (Once.r f (F v) * |F v| + Once.a f (F v)) / |F v|

/-- `p` is rejected at the input `x` of format `f`: the inputs are finite, `p` reads as `F` over ℝ
there, the answer's magnitude lies in the band, on which the once-rounding of a valid format is
finite and informative, and the result of `p` is not finite or is no closer to the answer than its
magnitude. -/
def Rejected (p : Prog A n) (F : (Fin n → ℝ) → ℝ) (f : A) (x : Fin n → Val f) : Prop :=
  (∀ i, isFinite (x i) = true) ∧ p.real (toReal ∘ x) = F (toReal ∘ x) ∧
    |F (toReal ∘ x)| ∈ Once.band f ∧ (isFinite (p.float f x) = false ∨
      |F (toReal ∘ x)| ≤ |toReal (p.float f x) - F (toReal ∘ x)|)

variable (p : Prog A n) (F : (Fin n → ℝ) → ℝ) (S : Set A) (D : A → Set (Fin n → ℝ))

/-- Where the answer has a finite rounding, the result is finite and is that rounding. -/
def RoundsOnce : Prop :=
  ∀ f ∈ S, ∀ x ∈ Inputs D f, F (toReal ∘ x) ∈ Once.fin f →
    isFinite (p.float f x) = true ∧ toReal (p.float f x) = Once.round f (F (toReal ∘ x))

/-- Where the answer is zero, the result is a finite zero. -/
def ZeroExact : Prop :=
  ∀ f ∈ S, ∀ x ∈ Inputs D f, F (toReal ∘ x) = 0 →
    isFinite (p.float f x) = true ∧ toReal (p.float f x) = 0

/-- Where the answer has no finite rounding, the result is the overflow value of its sign. -/
def Overflows : Prop :=
  ∀ f ∈ S, ∀ x ∈ Inputs D f, F (toReal ∘ x) ∉ Once.fin f →
    Once.overflow f (decide (F (toReal ∘ x) < 0)) (p.float f x)

/-- Where the answer has no finite rounding, the result is the largest finite value with the
sign of the answer. -/
def Saturates : Prop :=
  ∀ f ∈ S, ∀ x ∈ Inputs D f, F (toReal ∘ x) ∉ Once.fin f →
    isFinite (p.float f x) = true ∧ toReal (p.float f x) = Real.sign (F (toReal ∘ x)) * Once.Ω f

/-- In format `f`, `p` computes `F` over ℝ on `D₀`, every input of `D₀` outside `gap` is a zero,
below, above or band input, and the zero, above and band inputs have the outcomes of their parts.
-/
structure Covers (f : A) (D₀ : Set (Fin n → ℝ)) (c : Claim A n) (gap : Set (Fin n → ℝ)) :
    Prop where
  /-- The format is valid. -/
  valid : valid f
  /-- The program computes `F` over ℝ on `D₀`. -/
  real : ∀ v ∈ D₀, p.real v = F v
  /-- Every input of `D₀` outside the gap is a zero, below, above or band input. -/
  split : ∀ v ∈ D₀ \ gap,
    F v = 0 ∨ |F v| ∈ Once.below f ∨ F v ∉ Once.fin f ∨ (v ∈ c.D f ∧ |F v| ∈ c.band f)
  /-- At a zero answer the result is a finite zero. -/
  zero : ZeroExact p F {f} fun _ => D₀ \ gap
  /-- At an answer without finite rounding the result overflows or saturates. -/
  above : Overflows p F {f} (fun _ => D₀ \ gap) ∨ Saturates p F {f} fun _ => D₀ \ gap
  /-- The bounds of the claim hold, with the scale `|F|`. -/
  band : Bounds p F (fun v => |F v|) {f} c

variable {p F S D}

/-- A program that rounds once, with the real equality on `D` in valid formats, has the bounds of
`Once.claim`. -/
theorem Bounds.ofRoundsOnce (h : RoundsOnce p F S D)
    (hreal : ∀ f ∈ S, ∀ v ∈ D f, p.real v = F v) (hS : ∀ f ∈ S, valid f) :
    Bounds p F (fun v => |F v|) S (Once.claim F D) where
  real := hreal
  pos f hf _ hy := Once.pos (hS f hf) hy
  sub f hf _ _ hb := (Once.informative (hS f hf) hb).1
  bound f hf x hx hfin := by
    obtain ⟨h1, h2⟩ := h f hf x hx hfin
    refine ⟨h1, ?_⟩
    rw [h2]
    exact Once.mixed (hS f hf) hfin
  informative f hf v _ hb := by
    dsimp only [Once.claim] at hb ⊢
    have hF : 0 < |F v| := Once.pos (hS f hf) hb
    refine ⟨(div_mul_cancel₀ _ hF.ne').ge, ?_⟩
    rw [div_lt_one hF]
    exact (Once.informative (hS f hf) hb).2

/-- A program that rounds once, with the real equality on `D`, on a nonempty set of valid formats
each with a finite input whose answer lies in the band, has the certificate of `Once.claim`. -/
theorem Cert.ofRoundsOnce (h : RoundsOnce p F S D)
    (hreal : ∀ f ∈ S, ∀ v ∈ D f, p.real v = F v) (hS : S.Nonempty ∧ ∀ f ∈ S, valid f)
    (hw : ∀ f ∈ S, ∃ x ∈ Inputs D f, |F (toReal ∘ x)| ∈ Once.band f) :
    Cert p F S (Once.claim F D) :=
  { Bounds.ofRoundsOnce h hreal hS.2 with covers := hS, witness := hw }

/-- In valid formats a program that rounds once returns a finite zero at a zero answer. -/
theorem RoundsOnce.zeroExact (h : RoundsOnce p F S D) (hS : ∀ f ∈ S, valid f) :
    ZeroExact p F S D := by
  intro f hf x hx h0
  obtain ⟨hfin, hround⟩ := Once.zero (hS f hf)
  obtain ⟨h1, h2⟩ := h f hf x hx (by rw [h0]; exact hfin)
  exact ⟨h1, by rw [h2, h0, hround]⟩

/-- In a valid format a program that rounds once on `D₀`, overflows at the answers of `D₀` without
finite rounding and computes `F` over ℝ on `D₀` covers `D₀`, with the claim `Once.claim` on `D₀`
and the gap of the inputs whose answers lie in a hole. -/
theorem Covers.ofRoundsOnce {f : A} {D₀ : Set (Fin n → ℝ)} (hf : Arith.valid f)
    (h : RoundsOnce p F {f} fun _ => D₀) (ho : Overflows p F {f} fun _ => D₀)
    (hreal : ∀ v ∈ D₀, p.real v = F v) :
    Covers p F f D₀ (Once.claim F fun _ => D₀) {v ∈ D₀ | |F v| ∈ Once.holes f} where
  valid := hf
  real := hreal
  split v hv := by
    rcases Once.split (y := F v) hf with h0 | hb | hfin | hband | hh
    · exact Or.inl h0
    · exact Or.inr (Or.inl hb)
    · exact Or.inr (Or.inr (Or.inl hfin))
    · exact Or.inr (Or.inr (Or.inr ⟨hv.1, hband⟩))
    · exact absurd ⟨hv.1, hh⟩ hv.2
  zero g hg x hx :=
    h.zeroExact (fun _ hg => Set.mem_singleton_iff.mp hg ▸ hf) g hg x ⟨hx.1, hx.2.1⟩
  above := Or.inl fun g hg x hx => ho g hg x ⟨hx.1, hx.2.1⟩
  band := Bounds.ofRoundsOnce h (fun _ _ => hreal) fun _ hg => Set.mem_singleton_iff.mp hg ▸ hf

/-- With the scale `|F|`, a program is not rejected at an input whose answer lies in the band of
its bounds. -/
theorem Bounds.not_rejected {c : Claim A n} (h : Bounds p F (fun v => |F v|) S c) {f : A}
    (hf : f ∈ S) {x : Fin n → Val f} (hx : x ∈ Inputs c.D f) (hb : |F (toReal ∘ x)| ∈ c.band f) :
    ¬Rejected p F f x := fun hr => by
  obtain ⟨h1, -, h3⟩ := h.excludes hf hx hb
  have h4 := mul_lt_mul_of_pos_right (h.informative f hf _ hx.2 hb).2 (h.pos f hf _ hb)
  rcases hr.2.2.2 with h5 | h5
  · exact absurd (h5.symm.trans h1) (by decide)
  · linarith

/-- A program is rejected at finite inputs where it reads as `F` over ℝ, the answer's magnitude
lies in the band, and the product of the result and the answer is at most zero. -/
theorem Rejected.of_mul_nonpos {f : A} {x : Fin n → Val f} (hx : ∀ i, isFinite (x i) = true)
    (hp : p.real (toReal ∘ x) = F (toReal ∘ x)) (hb : |F (toReal ∘ x)| ∈ Once.band f)
    (hv : toReal (p.float f x) * F (toReal ∘ x) ≤ 0) : Rejected p F f x :=
  ⟨hx, hp, hb, Or.inr (sq_le_sq.mp (by nlinarith [sq_nonneg (toReal (p.float f x))]))⟩

end Once

/-! ## Configured binary formats -/

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange
open FloatLib.Floats.Formats.BinaryInterchange.Model.Transcendentals.Certified (overflowThreshold)

/-- A configured binary format: its descriptor, storage plan and code type, with their codec. -/
structure Binary.Desc where
  /-- The format descriptor. -/
  format : FloatFormat
  /-- The storage plan. -/
  plan : Configured.StoragePlan format
  /-- The type that stores the values. -/
  code : Type
  /-- The codec between the stored values and the model of the format. -/
  [codec : ExecFloat.ModelCodec plan (Model format) code]

attribute [instance] Binary.Desc.codec

/-- The model of a value of a configured binary format. -/
abbrev Binary.toModel {d : Binary.Desc} (v : ExecFloat (Configured.Family d.format d.code d.plan)) :
    Model d.format :=
  ExecFloat.Binary.toModel (format := d.format) (plan := d.plan) (code := d.code) v

/-- The value of a configured binary format with a given model. -/
abbrev Binary.ofModel {d : Binary.Desc} (m : Model d.format) :
    ExecFloat (Configured.Family d.format d.code d.plan) :=
  ExecFloat.Binary.ofModel (format := d.format) (plan := d.plan) (code := d.code) m

/-- The configured binary formats as a number system, valid at the IEEE descriptors. -/
noncomputable instance : Arith Binary.Desc where
  Val d := ExecFloat (Configured.Family d.format d.code d.plan)
  isFinite x := ExecFloat.Binary.isFinite x
  toReal x := ExecFloat.Binary.toReal x
  valid d := d.format.isIEEE = true

/-- A program written once against `Storage` and `Context`, read over ℝ and in every configured
binary format. -/
noncomputable def Prog.ofContext (p : ∀ {α : Type} [Storage α] [Context α], (Fin n → α) → α) :
    Prog Binary.Desc n :=
  ⟨p, fun d => p (α := ExecFloat (Configured.Family d.format d.code d.plan))⟩

/-- Nearest-even rounding in a configured binary format, with the laws of `Once` at the IEEE
descriptors: the answers with finite rounding are those below the overflow threshold `Θ` in
magnitude, the mixed budget is `(u, max 0 (η - u * |y|))`, informative on the band `(η, Θ)`, no
finite value is closer than its magnitude to an answer on `(0, η]`, and there are no holes. The
overflow value of a sign is `nativeOverflow` of that sign, the infinity of that sign in an IEEE
format. -/
noncomputable instance : Once Binary.Desc where
  toArith := inferInstance
  round d y := Model.roundAt d.format y
  fin d := {y | |y| < overflowThreshold d.format}
  band d := Set.Ioo (Model.underflowErrorAt d.format) (overflowThreshold d.format)
  below d := Set.Ioc 0 (Model.underflowErrorAt d.format)
  holes _ := ∅
  r d _ := Model.unitRoundoffAt d.format
  a d y := max 0 (Model.underflowErrorAt d.format - Model.unitRoundoffAt d.format * |y|)
  Ω d := Model.toReal (Model.posMaxFinite d.format)
  overflow d s v := Binary.toModel v = Model.nativeOverflow d.format s
  mixed _ _ := once_mixedError _ _
  informative _ hy := ⟨hy.2, (once_informative_iff _ _).mpr hy.1⟩
  pos _ hy := (Model.underflowErrorAt_pos _).trans hy.1
  zero hd := by
    refine ⟨?_, Model.roundAt_zero _⟩
    show |(0 : ℝ)| < _
    rw [abs_zero]
    exact (Model.toReal_posMaxFinite_pos _).trans (posMaxFinite_lt_overflowThreshold hd)
  split := by
    intro d y _
    rcases eq_or_ne y 0 with h0 | h0
    · exact Or.inl h0
    rcases le_or_gt |y| (Model.underflowErrorAt d.format) with hη | hη
    · exact Or.inr (Or.inl ⟨abs_pos.mpr h0, hη⟩)
    rcases le_or_gt (overflowThreshold d.format : ℝ) |y| with hΘ | hΘ
    · exact Or.inr (Or.inr (Or.inl (not_lt.mpr hΘ)))
    · exact Or.inr (Or.inr (Or.inr (Or.inl ⟨hη, hΘ⟩)))
  no_informative := by
    intro d y _ hy v hv
    exact no_informative_below hy.2 (Binary.toModel v) hv
  le_Ω := by
    intro d _ v hv
    exact Model.abs_toReal_le_posMaxFinite_of_isFinite (Binary.toModel v) hv

/-- In an IEEE format, `p` is rejected at `x` iff the inputs are finite, `p` reads as `F` over ℝ
there, some finite value is closer to the answer than its magnitude, the answer is below `Θ` in
magnitude, and the result is no closer to the answer than its magnitude. -/
theorem Binary.rejected_iff {n : ℕ} {p : Prog Binary.Desc n} {F : (Fin n → ℝ) → ℝ}
    {d : Binary.Desc} (hd : valid d) {x : Fin n → Val d} :
    Rejected p F d x ↔ (∀ i, isFinite (x i) = true) ∧ p.real (toReal ∘ x) = F (toReal ∘ x) ∧
      (∃ v : Val d, isFinite v = true ∧ |toReal v - F (toReal ∘ x)| < |F (toReal ∘ x)|) ∧
        |F (toReal ∘ x)| < overflowThreshold d.format ∧
          |F (toReal ∘ x)| ≤ |toReal (p.float d x) - F (toReal ∘ x)| := by
  have hd' : d.format.isIEEE = true := hd
  have hv : (∃ v : Val d, isFinite v = true ∧ |toReal v - F (toReal ∘ x)| < |F (toReal ∘ x)|) ↔
      Model.underflowErrorAt d.format < |F (toReal ∘ x)| := by
    refine ⟨fun ⟨v, hv, hlt⟩ => (exists_informative_iff hd').mp ⟨Binary.toModel v, hv, hlt⟩,
      fun h => ?_⟩
    obtain ⟨m, hm, hlt⟩ := (exists_informative_iff hd').mpr h
    have he : Binary.toModel (Binary.ofModel m) = m := ExecFloat.Binary.toModel_ofModel _
    refine ⟨Binary.ofModel m, ?_, ?_⟩
    · show Model.isFinite (Binary.toModel (Binary.ofModel m)) = true
      rw [he]
      exact hm
    · show |Model.toReal (Binary.toModel (Binary.ofModel m)) - _| < _
      rw [he]
      exact hlt
  unfold Rejected
  refine and_congr_right fun _ => and_congr_right fun _ => ?_
  rw [hv, ← and_assoc]
  refine and_congr_right fun _ => or_iff_right_of_imp fun h => ?_
  have h0 : toReal (p.float d x) = 0 := Model.toReal_eq_zero_of_isFinite_eq_false h
  rw [h0, zero_sub, abs_neg]

/-- binary32 with its default storage plan and code. -/
def Binary.binary32 : Binary.Desc where
  format := FloatFormat.binary32
  plan := Configured.StoragePlan.forKnownWidth _ 32 rfl
  code := Configured.Code (Configured.StoragePlan.forKnownWidth FloatFormat.binary32 32 rfl)

end TorchLean.Stable

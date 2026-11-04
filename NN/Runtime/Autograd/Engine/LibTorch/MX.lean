/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

LibTorch FFI: tensors in the OCP Microscaling (MX) 1.0 formats on the CUDA device.
-/

module

public import FloatLib.Floats.Formats.OCP.MX.Standard.Core
public import Mathlib.Algebra.Order.Field.Power
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Positivity

/-!
# OCP MX tensors

Shapes, byte layout and the exactness fact behind decoding for tensors stored in the six concrete
formats of the OCP *Microscaling Formats (MX) Specification* 1.0 (E5M2, E4M3, E3M2, E2M3, E2M1 and
INT8). FloatLib's `FloatLib.Floats.Formats.OCP.MX.Standard` is their semantics: a block is 32
elements sharing one E8M0 scale.

A tensor here is a row-major matrix of `rows × columns` elements, blocked by 32 consecutive
elements along each row, so `columns` is a positive multiple of 32 (the specification pads a
shorter row, §6.2). Codes are stored one to a byte, except E2M1, two to a byte with element 0 in
the low four bits; an FP6 code occupies a byte's low six bits. Scales take one byte per block.

Decoding to binary32 multiplies each element's binary32 value by the block's power of two. The
theorem `decodeLane_binary32` shows that product is exact whenever the decoded value lies in
binary32's finite range: every finite lane value is `m · 2^e` with a significand of at most 8 bits
and `e ≥ -143`, inside binary32's 24 bits and minimal exponent `-149`.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace LibTorch
namespace MX

open FloatLib.Floats.Formats.OCP.MX.Standard
open FloatLib.Numerics (SignedRat)

/-- A row-major `rows × columns` matrix of MX elements of one profile, blocked by 32 along each
row. A shape whose column count is zero or not a multiple of 32 cannot be built. -/
structure Shape where
  /-- The element encoding of every block. -/
  profile : Profile
  /-- The number of rows; zero gives an empty tensor. -/
  rows : Nat
  /-- The number of elements per row. -/
  columns : Nat
  columns_pos : 0 < columns
  columns_dvd : 32 ∣ columns

namespace Shape

/-- The number of elements. -/
def elements (s : Shape) : Nat := s.rows * s.columns

/-- The number of blocks, one scale byte each. -/
def blocks (s : Shape) : Nat := s.elements / 32

/-- The bytes holding the element codes: two E2M1 codes to a byte, one code of any other
profile. -/
def codeBytes (s : Shape) : Nat :=
  match s.profile with
  | .e2m1 => s.elements / 2
  | _ => s.elements

theorem dvd_elements (s : Shape) : 32 ∣ s.elements :=
  Nat.dvd_trans s.columns_dvd (Nat.dvd_mul_left _ _)

/-- The scales cover the elements exactly: no element is left without a block. -/
theorem blocks_mul (s : Shape) : s.blocks * 32 = s.elements :=
  Nat.div_mul_cancel s.dvd_elements

/-- The E2M1 codes fill their bytes exactly: no half byte is left over. -/
theorem codeBytes_mul_of_e2m1 (s : Shape) (h : s.profile = .e2m1) :
    s.codeBytes * 2 = s.elements := by
  have : 2 ∣ s.elements := Nat.dvd_trans (by decide) s.dvd_elements
  simp [codeBytes, h, Nat.div_mul_cancel this]

/-- Any other profile takes one byte per code. -/
theorem codeBytes_of_ne_e2m1 (s : Shape) (h : s.profile ≠ .e2m1) :
    s.codeBytes = s.elements := by
  unfold codeBytes
  split <;> simp_all

end Shape

/-- binary32's largest finite value, `(2 - 2^-23) · 2^127`. -/
def binary32Max : Rat := (2 ^ 24 - 1) * 2 ^ (104 : Int)

/-- The finite binary32 values: a significand of at most 24 bits, an exponent of at least the
smallest subnormal's, `-149`, and a magnitude at most the largest finite value. -/
def IsBinary32 (x : Rat) : Prop :=
  (∃ m e : Int, x = m * 2 ^ e ∧ m.natAbs < 2 ^ 24 ∧ -149 ≤ e) ∧ |x| ≤ binary32Max

/-- The largest `e` in `[-16, 16]` with `q / 2^e` an integer, paired with that integer. -/
def elementWitness (q : Rat) : Int × Int :=
  match ((List.range 33).map (fun i : Nat => (16 : Int) - (i : Int))).find?
      (fun e => (q / 2 ^ e).den == 1) with
  | some e => ((q / 2 ^ e).num, e)
  | none => (0, 0)

/-- Whether a decoded element, if finite, is `m · 2^e` with `|m| < 2^8` and `e ≥ -16`, by the
witness `elementWitness`. -/
def elementFits (q? : Option Rat) : Bool :=
  match q? with
  | none => true
  | some q =>
    q == ((elementWitness q).1 : Rat) * 2 ^ (elementWitness q).2 &&
      decide ((elementWitness q).1.natAbs < 2 ^ 8) && decide (-16 ≤ (elementWitness q).2)

theorem elementFits_all (p : Profile) (w : Element p) : elementFits w.toRat? = true := by
  have key : ∀ n : Fin (2 ^ p.width),
      elementFits (Element.toRat? (profile := p) (BitVec.ofNat p.width n)) = true := by
    cases p <;> decide +kernel
  simpa [BitVec.ofNat_toNat] using key ⟨w.toNat, w.isLt⟩

/-- Every finite element of every profile is `m · 2^e` with `|m| < 2^8` and `e ≥ -16`. -/
theorem element_dyadic {p : Profile} (w : Element p) {q : Rat} (h : w.toRat? = some q) :
    ∃ m e : Int, q = m * 2 ^ e ∧ m.natAbs < 2 ^ 8 ∧ -16 ≤ e := by
  have fits := elementFits_all p w
  simp only [h, elementFits, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at fits
  exact ⟨_, _, fits.1.1, fits.1.2, fits.2⟩

/-- Decoding is exact in binary32: a finite lane value of any block, whose magnitude lies in
binary32's finite range, is a binary32 value. -/
theorem decodeLane_binary32 {p : Profile} (b : Block p) (lane : Fin 32) {v : SignedRat}
    (h : (b.decodeLane lane).finite? = some v) (range : |v.value| ≤ binary32Max) :
    IsBinary32 v.value := by
  refine ⟨?_, range⟩
  unfold Block.decodeLane at h
  cases hs : b.scale.exponent? with
  | none => simp [hs, FloatLib.Numerics.NumericalValue.finite?] at h
  | some s =>
    have s_ge : -127 ≤ s := by
      unfold FloatLib.Floats.Formats.OCP.MX.E8M0.exponent? at hs
      split at hs <;> simp_all [FloatLib.Floats.Formats.OCP.MX.E8M0.bias]
      omega
    cases hd : Element.decode b.values[lane.val] with
    | finite u =>
      simp [hs, hd, FloatLib.Numerics.NumericalValue.map,
        FloatLib.Numerics.NumericalValue.finite?] at h
      subst h
      have hq : b.values[lane.val].toRat? = some u.value := by
        simp [Element.toRat?, hd, FloatLib.Numerics.NumericalValue.finite?]
      obtain ⟨m, e, hm, hbits, he⟩ := element_dyadic _ hq
      refine ⟨m, e + s, ?_, by omega, by omega⟩
      simp only [scaleFinite, SignedRat.value_mul, SignedRat.value_ofRat, hm]
      rw [zpow_add₀ (by norm_num : (2 : Rat) ≠ 0)]
      ring
    | infinity sign => simp [hs, hd, FloatLib.Numerics.NumericalValue.map,
        FloatLib.Numerics.NumericalValue.finite?] at h
    | exceptional e => simp [hs, hd, FloatLib.Numerics.NumericalValue.map,
        FloatLib.Numerics.NumericalValue.finite?] at h

/-- One bit finer fails: `2^-150`, half binary32's smallest subnormal, is no binary32 value, so
the bound `-149` in `IsBinary32` is the one binary32 has and the theorem above is no vacuous
statement about every rational. -/
theorem not_isBinary32_half_smallest : ¬ IsBinary32 (2 ^ (-150 : Int)) := by
  rintro ⟨⟨m, e, h, -, he⟩, -⟩
  have hm : (m : Rat) = 2 ^ (-150 - e) := by
    rw [zpow_sub₀ (by norm_num : (2 : Rat) ≠ 0), h, mul_div_assoc,
      div_self (zpow_ne_zero _ (by norm_num : (2 : Rat) ≠ 0)), mul_one]
  have pos : (0 : Rat) < m := by rw [hm]; positivity
  have lt : (m : Rat) < 1 := by
    rw [hm]
    exact zpow_lt_one_of_neg₀ (by norm_num) (by omega)
  have : (0 : Int) < m := by exact_mod_cast pos
  have : m < (1 : Int) := by exact_mod_cast lt
  omega

end MX
end LibTorch
end Autograd
end Runtime

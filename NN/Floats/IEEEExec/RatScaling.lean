/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Algebra.Order.Field.Basic
public import Mathlib.Data.Nat.Bitwise
public import NN.Floats.IEEEExec.Exec32
public import NN.Floats.NeuralFloat.Core

/-!
# RatScaling

Small algebraic helper lemmas about powers of two and `Nat.shiftLeft` used by the IEEE32Exec
kernel.

Several proofs (notably the bridge theorems and division soundness) need to normalize expressions
of the form `dyadicToReal dx / dyadicToReal dy` into the same *signed rational* shape that the
executable implementation uses:

- align dyadic exponents by shifting either the numerator or denominator,
- extract a sign bit via `Bool.xor`,
- and express the quotient as `±(num/den)`.

This module provides the shared normalization lemmas used by the bridge theorems without creating
import cycles.
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754

open TorchLean.Floats

namespace IEEE32Exec

/-- Interpret an exact dyadic `(-1)^sign * mant * 2^exp` as a real. -/
noncomputable def dyadicToReal (d : Dyadic) : ℝ :=
  let s : ℝ := if d.sign then (-1 : ℝ) else (1 : ℝ)
  s * (d.mant : ℝ) * neuralBpow binaryRadix d.exp

/-- Scale a rational by a nonnegative exponent difference by shifting the numerator. -/
lemma scaleRat_ofNat (num den sh : Nat) :
    ((num : ℝ) / (den : ℝ)) * neuralBpow binaryRadix (Int.ofNat sh) =
      ((Nat.shiftLeft num sh : Nat) : ℝ) / (den : ℝ) := by
  have hb : neuralBpow binaryRadix (Int.ofNat sh) = (2 : ℝ) ^ sh := by
    simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal]
  rw [hb]
  have hnumShift : (num : ℝ) * ((2 : ℝ) ^ sh) = ((Nat.shiftLeft num sh : Nat) : ℝ) := by
    have hp : (2 : ℝ) ^ sh = ((2 ^ sh : Nat) : ℝ) := by
      simp
    rw [hp]
    simp [Nat.shiftLeft_eq, Nat.cast_mul]
  calc
    ((num : ℝ) / (den : ℝ)) * (2 : ℝ) ^ sh = ((num : ℝ) * ((2 : ℝ) ^ sh)) / (den : ℝ) := by
      simp [div_mul_eq_mul_div]
    _ = ((Nat.shiftLeft num sh : Nat) : ℝ) / (den : ℝ) := by
      rw [hnumShift]

/-- Scale a rational by a negative exponent difference by shifting the denominator. -/
lemma scaleRat_negSucc (num den sh : Nat) :
    ((num : ℝ) / (den : ℝ)) * neuralBpow binaryRadix (Int.negSucc sh) =
      (num : ℝ) / ((Nat.shiftLeft den (sh + 1) : Nat) : ℝ) := by
  have hb : neuralBpow binaryRadix (Int.negSucc sh) = (1 : ℝ) / (2 : ℝ) ^ (sh + 1) := by
    simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal, zpow_negSucc,
      div_eq_mul_inv]
  rw [hb]
  calc
    ((num : ℝ) / (den : ℝ)) * ((1 : ℝ) / (2 : ℝ) ^ (sh + 1)) =
        (num : ℝ) / ((den : ℝ) * ((2 : ℝ) ^ (sh + 1))) := by
      field_simp
    _ = (num : ℝ) / ((Nat.shiftLeft den (sh + 1) : Nat) : ℝ) := by
      have hp : (2 : ℝ) ^ (sh + 1) = ((2 ^ (sh + 1) : Nat) : ℝ) := by
        simp
      rw [hp]
      have hdenShift :
          (den : ℝ) * ((2 ^ (sh + 1) : Nat) : ℝ) = ((Nat.shiftLeft den (sh + 1) : Nat) : ℝ) := by
        simp [Nat.shiftLeft_eq, Nat.cast_mul]
      rw [hdenShift]

/-- Exponent subtraction law for `neuralBpow`, packaged as a division identity. -/
lemma neural_bpow_div (e1 e2 : Int) :
    neuralBpow binaryRadix e1 / neuralBpow binaryRadix e2 = neuralBpow binaryRadix (e1 - e2) := by
  -- `bpow e1 / bpow e2 = bpow (e1 + (-e2))`.
  simp [div_eq_mul_inv]
  have hneg : neuralBpow binaryRadix (-e2) = (neuralBpow binaryRadix e2)⁻¹ := by
    simpa using (neuralBpow.neg_exp binaryRadix e2)
  have hadd :
      neuralBpow binaryRadix (e1 + (-e2)) =
        neuralBpow binaryRadix e1 * neuralBpow binaryRadix (-e2) := by
    simpa using (neuralBpow.add_exp binaryRadix e1 (-e2))
  calc
    neuralBpow binaryRadix e1 * (neuralBpow binaryRadix e2)⁻¹
        = neuralBpow binaryRadix e1 * neuralBpow binaryRadix (-e2) := by simp [hneg]
    _ = neuralBpow binaryRadix (e1 + (-e2)) := by simpa using hadd.symm
    _ = neuralBpow binaryRadix (e1 - e2) := by simp [sub_eq_add_neg]

/--
Rewrite dyadic division into the same signed rational shape used by `div`/`divDown`/`divUp`.

This is the "multiplicative" form: `(-1)^sign * (num/den)`.
-/
lemma dyadicToReal_div_eq_signedRat_mul (dx dy : Dyadic) (hy0 : dy.mant ≠ 0) :
    let sign : Bool := Bool.xor dx.sign dy.sign
    let eDiff : Int := dx.exp - dy.exp
    let (num, den) :=
      match eDiff with
      | .ofNat sh => (Nat.shiftLeft dx.mant sh, dy.mant)
      | .negSucc sh => (dx.mant, Nat.shiftLeft dy.mant (sh + 1))
    dyadicToReal dx / dyadicToReal dy =
      (if sign then (-1 : ℝ) else 1) * ((num : ℝ) / (den : ℝ)) := by
  classical
  set sign : Bool := Bool.xor dx.sign dy.sign
  set eDiff : Int := dx.exp - dy.exp
  have hsy : (if dy.sign then (-1 : ℝ) else 1) ≠ 0 := by
    cases dy.sign <;> norm_num
  have hmy : (dy.mant : ℝ) ≠ 0 := by exact_mod_cast hy0
  have hby : neuralBpow binaryRadix dy.exp ≠ 0 := neuralBpow.ne_zero binaryRadix dy.exp
  have hden :
      (if dy.sign then (-1 : ℝ) else 1) * (dy.mant : ℝ) * neuralBpow binaryRadix dy.exp ≠ 0 :=
    mul_ne_zero (mul_ne_zero hsy hmy) hby

  -- Split the dyadic quotient into sign, mantissa ratio, and exponent ratio.
  have hsplit :
      dyadicToReal dx / dyadicToReal dy =
        ((if dx.sign then (-1 : ℝ) else 1) / (if dy.sign then (-1 : ℝ) else 1)) *
          ((dx.mant : ℝ) / (dy.mant : ℝ)) *
            (neuralBpow binaryRadix dx.exp / neuralBpow binaryRadix dy.exp) := by
    by_cases hx : dx.sign <;> by_cases hy : dy.sign <;>
      (field_simp [dyadicToReal, hden, hmy, hby, hx, hy]
       ; simp [dyadicToReal, hx, hy, mul_assoc, mul_left_comm, mul_comm]
       ; try ring_nf)

  have hsign :
      ((if dx.sign then (-1 : ℝ) else 1) / (if dy.sign then (-1 : ℝ) else 1)) =
        (if sign then (-1 : ℝ) else 1) := by
    by_cases hx : dx.sign <;> by_cases hy : dy.sign <;>
      simp [sign, Bool.xor, hx, hy]

  have hbpow :
      neuralBpow binaryRadix dx.exp / neuralBpow binaryRadix dy.exp =
        neuralBpow binaryRadix (dx.exp - dy.exp) := by
    simpa using (neural_bpow_div (e1 := dx.exp) (e2 := dy.exp))

  -- Turn the mantissa/exponent ratio into the same rational `num/den` used by `div`.
  cases hE : eDiff with
  | ofNat sh =>
      have he : dx.exp - dy.exp = (Int.ofNat sh) := by simpa [eDiff] using hE
      have hscale :
          ((dx.mant : ℝ) / (dy.mant : ℝ)) * neuralBpow binaryRadix (Int.ofNat sh) =
            ((Nat.shiftLeft dx.mant sh : Nat) : ℝ) / (dy.mant : ℝ) := by
        simpa using (scaleRat_ofNat (num := dx.mant) (den := dy.mant) (sh := sh))
      have : dyadicToReal dx / dyadicToReal dy =
          (if sign then (-1 : ℝ) else 1) *
            (((Nat.shiftLeft dx.mant sh : Nat) : ℝ) / (dy.mant : ℝ)) := by
        rw [hsplit, hsign, hbpow]
        have hscale' :
            (if sign then (-1 : ℝ) else 1) *
                ((dx.mant : ℝ) / (dy.mant : ℝ) * neuralBpow binaryRadix (Int.ofNat sh)) =
              (if sign then (-1 : ℝ) else 1) * (((Nat.shiftLeft dx.mant sh : Nat) : ℝ) / (dy.mant :
                ℝ)) := by
          simpa using congrArg (fun t : ℝ => (if sign then (-1 : ℝ) else 1) * t) hscale
        simpa [he, mul_assoc, mul_left_comm, mul_comm] using hscale'
      simp [sign, this, mul_comm]
  | negSucc sh =>
      have he : dx.exp - dy.exp = (Int.negSucc sh) := by simpa [eDiff] using hE
      have hscale :
          ((dx.mant : ℝ) / (dy.mant : ℝ)) * neuralBpow binaryRadix (Int.negSucc sh) =
            (dx.mant : ℝ) / ((Nat.shiftLeft dy.mant (sh + 1) : Nat) : ℝ) := by
        simpa using (scaleRat_negSucc (num := dx.mant) (den := dy.mant) (sh := sh))
      have : dyadicToReal dx / dyadicToReal dy =
          (if sign then (-1 : ℝ) else 1) *
            ((dx.mant : ℝ) / ((Nat.shiftLeft dy.mant (sh + 1) : Nat) : ℝ)) := by
        rw [hsplit, hsign, hbpow]
        simp [he, mul_comm, hscale]
      simp [sign, this, mul_comm]

/--
Rewrite dyadic division into a signed rational `±(num/den)`.

This is the "conditional negation" form used by the directed rounding soundness proofs.
-/
lemma dyadicToReal_div_eq_signedRat (dx dy : Dyadic) (hy0 : dy.mant ≠ 0) :
    let sign : Bool := Bool.xor dx.sign dy.sign
    let eDiff : Int := dx.exp - dy.exp
    let (num, den) :=
      match eDiff with
      | .ofNat sh => (Nat.shiftLeft dx.mant sh, dy.mant)
      | .negSucc sh => (dx.mant, Nat.shiftLeft dy.mant (sh + 1))
    dyadicToReal dx / dyadicToReal dy =
      if sign then -((num : ℝ) / (den : ℝ)) else (num : ℝ) / (den : ℝ) := by
  classical
  cases hs : Bool.xor dx.sign dy.sign <;>
    simpa (config := { zeta := true }) [hs] using
      (dyadicToReal_div_eq_signedRat_mul (dx := dx) (dy := dy) hy0)

end IEEE32Exec
end TorchLean.Floats.IEEE754

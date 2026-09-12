/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Nicolas Rouquette, TorchLean Team
-/

module

public import NN.Floats.FP32.Notation
public import NN.Floats.IEEEExec.Bridge.FP32.Ops

/-!
# Executable Binary32 ULPs and Absorption

`ulp32` specifies the unit in the last place as a real number. This file supplies the executable
counterpart for `IEEE32Exec`: `ulpExp?` reads a finite binary32 payload and returns the exponent
`k` for which the ULP is `2^k`. It returns `none` for NaNs and infinities rather than assigning
them an artificial spacing.

The absorption test checks whether adding one executable value to another leaves the first value
unchanged. Its soundness theorem transports that observation through the finite-operation bridge
to the rounded-real binary32 model.
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754

open TorchLean.Floats

namespace IEEE32Exec

/--
Compute the binary32 ULP exponent from an executable bit pattern.

Finite zero has exponent `-149`. A finite nonzero value with dyadic mantissa `m` and exponent `e`
has magnitude `⌊log₂ m⌋ + e + 1`; applying `fexp32` gives its ULP exponent. NaNs and infinities
return `none`.
-/
def ulpExp? (x : IEEE32Exec) : Option Int :=
  match toDyadic? x with
  | some d =>
      some (if d.mant = 0 then -149 else fexp32 (Int.ofNat (Nat.log 2 d.mant) + d.exp + 1))
  | none => none

/-- `ulpExp?` succeeds exactly on finite executable binary32 values. -/
@[simp] theorem ulpExp?_isSome (x : IEEE32Exec) :
    (ulpExp? x).isSome = isFinite x := by
  rw [← toDyadic?_isSome_eq_isFinite]
  cases hdy : toDyadic? x <;> simp [ulpExp?, hdy]

/-- NaNs and infinities are precisely the values for which `ulpExp?` returns `none`. -/
theorem ulpExp?_eq_none_iff (x : IEEE32Exec) :
    ulpExp? x = none ↔ isFinite x = false := by
  rw [← toDyadic?_isSome_eq_isFinite]
  cases hdy : toDyadic? x <;> simp [ulpExp?, hdy]

/--
For every finite executable binary32 value, exponentiating the result of `ulpExp?` gives its
rounded-real ULP exactly.
-/
theorem neuralBpow_ulpExp?_eq_ulp32 (x : IEEE32Exec) {d : Dyadic}
    (hx : toDyadic? x = some d) :
    (ulpExp? x).map (neuralBpow binaryRadix) = some (ulp32 (toReal x)) := by
  by_cases hm : d.mant = 0
  · have hz : dyadicToReal d = 0 := by
      have h := abs_dyadicToReal d
      rw [hm, Nat.cast_zero, zero_mul] at h
      exact abs_eq_zero.mp h
    have hxreal : toReal x = 0 := by
      rw [toReal_eq, hx]
      exact hz
    have hue : ulpExp? x = some (-149) := by simp [ulpExp?, hx, hm]
    rw [hue, hxreal]
    simp only [Option.map_some, Option.some.injEq]
    show neuralBpow binaryRadix (-149) = ulp32 0
    rw [ulp32_zero]
  · have hxreal : toReal x = dyadicToReal d := by rw [toReal_eq, hx]
    have hpos : 0 < _root_.abs (dyadicToReal d) := by
      rw [abs_dyadicToReal]
      exact mul_pos (by exact_mod_cast Nat.pos_of_ne_zero hm) (neuralBpow.pos binaryRadix d.exp)
    have hne : dyadicToReal d ≠ 0 := abs_pos.mp hpos
    simp only [ulpExp?, hx, hm, ↓reduceIte, Option.map_some, Option.some.injEq]
    rw [hxreal]
    show neuralBpow binaryRadix (fexp32 (Int.ofNat (Nat.log 2 d.mant) + d.exp + 1)) =
        neuralUlp binaryRadix fexp32 (dyadicToReal d)
    rw [neuralUlp.of_ne_zero binaryRadix fexp32 (dyadicToReal d) hne]
    simp only [neuralCexp]
    rw [neural_magnitude_dyadic d hm]

/--
Direct soundness theorem for the executable query: if `ulpExp? x` returns `k`, then `2^k` is
exactly the rounded-real binary32 ULP at `x`.
-/
theorem neuralBpow_eq_ulp32_of_ulpExp?_eq_some {x : IEEE32Exec} {k : Int}
    (hx : ulpExp? x = some k) :
    neuralBpow binaryRadix k = ulp32 (toReal x) := by
  cases hdy : toDyadic? x with
  | none =>
      simp [ulpExp?, hdy] at hx
  | some d =>
      have hsound := neuralBpow_ulpExp?_eq_ulp32 x hdy
      rw [hx] at hsound
      exact Option.some.inj hsound

/-- Whether executable binary32 addition leaves its left operand unchanged. -/
def absorbs (a b : IEEE32Exec) : Bool :=
  decide (add a b = a)

/--
If the executable absorption test succeeds on a finite addition, the rounded-real model agrees:
rounding the exact real sum returns the left operand.
-/
theorem round32_add_eq_left_of_absorbs {a b : IEEE32Exec} {da db : Dyadic}
    (ha : toDyadic? a = some da) (hb : toDyadic? b = some db)
    (hfin : isFinite (add a b) = true) (habs : absorbs a b = true) :
    round32 (toReal a + toReal b) = toReal a := by
  have h := toReal_add_eq_fp32Round a b ha hb hfin
  have habsEq : add a b = a := of_decide_eq_true habs
  rw [habsEq] at h
  exact h.symm

/--
Public finite-value form of absorption soundness. Dyadic decoding witnesses are recovered from the
three executable finiteness checks.
-/
theorem round32_add_eq_left_of_absorbs_of_isFinite {a b : IEEE32Exec}
    (ha : isFinite a = true) (hb : isFinite b = true)
    (hadd : isFinite (add a b) = true) (habs : absorbs a b = true) :
    round32 (toReal a + toReal b) = toReal a := by
  obtain ⟨da, hda⟩ := exists_toDyadic?_of_isFinite ha
  obtain ⟨db, hdb⟩ := exists_toDyadic?_of_isFinite hb
  exact round32_add_eq_left_of_absorbs hda hdb hadd habs

end IEEE32Exec

end TorchLean.Floats.IEEE754

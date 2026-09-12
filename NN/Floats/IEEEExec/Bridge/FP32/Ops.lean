/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.FP32.Sterbenz
public import NN.Floats.IEEEExec.Bridge.FP32.Core
public import NN.Floats.IEEEExec.Bridge.FP32.RoundRat
public import NN.Floats.IEEEExec.Exec32.Arithmetic

/-!
# IEEE32Exec and FP32: Arithmetic Operation Refinement
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754

open TorchLean.Floats

namespace IEEE32Exec

/-!
## Op-level refinement theorems (finite/no-overflow)

These are the results that the rest of TorchLean typically consumes: statements that each arithmetic
operation in `IEEE32Exec` refines its `FP32` real-rounded counterpart.

If you are coming from PyTorch: this is the “float32 math model” that underlies many informal
numerical arguments (“the kernel computes the exact real result, then rounds to float32”),
but made explicit and proved for our executable kernel.
-/

/-- Finite refinement for addition: `IEEE32Exec.add` = exact real add + float32 rounding. -/
theorem toReal_add_eq_fp32Round (x y : IEEE32Exec) {dx dy : Dyadic}
    (hx : toDyadic? x = some dx) (hy : toDyadic? y = some dy)
    (hfin : isFinite (add x y) = true) :
    toReal (add x y) = fp32Round (toReal x + toReal y) := by
  have hadd : add x y = roundDyadicToIEEE32 (addDyadic dx dy) :=
    add_eq_roundDyadicToIEEE32_of_toDyadic? (hx := hx) (hy := hy)
  have hfin' : isFinite (roundDyadicToIEEE32 (addDyadic dx dy)) = true := by
    simpa [hadd] using hfin
  calc
    toReal (add x y) = toReal (roundDyadicToIEEE32 (addDyadic dx dy)) := by
      simp [hadd]
    _ = fp32Round (dyadicToReal (addDyadic dx dy)) := by
      simpa using (toReal_roundDyadicToIEEE32_eq_fp32Round (d := addDyadic dx dy) hfin')
    _ = fp32Round (dyadicToReal dx + dyadicToReal dy) := by
      rw [dyadicToReal_addDyadic_exact (a := dx) (b := dy)]
    _ = fp32Round (toReal x + toReal y) := by
      simp [toReal_eq, hx, hy]

/-- Finite refinement for subtraction, reduced to addition + negation. -/
theorem toReal_sub_eq_fp32Round (x y : IEEE32Exec) {dx dy : Dyadic}
    (hx : toDyadic? x = some dx) (hy : toDyadic? y = some dy)
    (hfin : isFinite (sub x y) = true) :
    toReal (sub x y) = fp32Round (toReal x - toReal y) := by
  classical
  -- `sub x y` is defined as `add x (neg y)`.
  let dyNeg : Dyadic := { sign := (!dy.sign), mant := dy.mant, exp := dy.exp }
  have hyNeg : toDyadic? (neg y) = some dyNeg := by
    simpa [dyNeg] using (toDyadic?_neg_of_toDyadic?_some (x := y) (d := dy) hy)
  have hfin' : isFinite (add x (neg y)) = true := by simpa [sub] using hfin
  have hadd :
      toReal (add x (neg y)) = fp32Round (toReal x + toReal (neg y)) := by
    simpa [dyNeg] using
      (toReal_add_eq_fp32Round (x := x) (y := neg y) (dx := dx) (dy := dyNeg) hx hyNeg hfin')
  have hnegReal : toReal (neg y) = -toReal y := toReal_neg_eq_neg (x := y) (d := dy) hy
  calc
    toReal (sub x y) = fp32Round (toReal x + toReal (neg y)) := by
      simpa [sub] using hadd
    _ = fp32Round (toReal x - toReal y) := by
      simp [hnegReal, sub_eq_add_neg]

/--
Subtraction of two finite, nonnegative binary32 values cannot overflow.

Both operands lie between zero and the largest finite binary32 value, so their exact difference
lies in the symmetric interval bounded by that value. The executable dyadic rounder therefore
returns a finite result.
-/
theorem isFinite_sub_of_isFinite_of_nonneg (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true)
    (hx0 : 0 ≤ toReal x) (hy0 : 0 ≤ toReal y) :
    isFinite (sub x y) = true := by
  obtain ⟨dx, hdx⟩ := exists_toDyadic?_of_isFinite hx
  obtain ⟨dy, hdy⟩ := exists_toDyadic?_of_isFinite hy
  let dyNeg : Dyadic := { sign := (!dy.sign), mant := dy.mant, exp := dy.exp }
  have hdyNeg : toDyadic? (neg y) = some dyNeg := by
    simpa [dyNeg] using (toDyadic?_neg_of_toDyadic?_some (x := y) (d := dy) hdy)
  have hsub :
      sub x y = roundDyadicToIEEE32 (addDyadic dx dyNeg) := by
    simpa [sub] using
      (add_eq_roundDyadicToIEEE32_of_toDyadic? (x := x) (y := neg y)
        (dx := dx) (dy := dyNeg) hdx hdyNeg)
  have hnegReal : toReal (neg y) = -toReal y :=
    toReal_neg_eq_neg (x := y) (d := dy) hdy
  have hreal :
      dyadicToReal (addDyadic dx dyNeg) = toReal x - toReal y := by
    calc
      dyadicToReal (addDyadic dx dyNeg) =
          dyadicToReal dx + dyadicToReal dyNeg :=
        dyadicToReal_addDyadic_exact (a := dx) (b := dyNeg)
      _ = toReal x + toReal (neg y) := by simp [toReal_eq, hdx, hdyNeg]
      _ = toReal x - toReal y := by rw [hnegReal]; ring
  have hxMax : toReal x ≤ FP32.ieeeMaxFinite :=
    (le_abs_self (toReal x)).trans (abs_toReal_le_ieeeMaxFinite_of_isFinite x hx)
  have hyMax : toReal y ≤ FP32.ieeeMaxFinite :=
    (le_abs_self (toReal y)).trans (abs_toReal_le_ieeeMaxFinite_of_isFinite y hy)
  have hbound : |dyadicToReal (addDyadic dx dyNeg)| ≤ FP32.ieeeMaxFinite := by
    rw [hreal, abs_le]
    constructor <;> linarith
  rw [hsub]
  exact isFinite_roundDyadicToIEEE32_of_abs_le_ieeeMaxFinite _ hbound

/--
Executable Sterbenz theorem for finite binary32 values.

If two positive operands are within a factor of two, then bit-level subtraction is finite and
denotes the exact real difference. The result combines executable-operation refinement,
representability of finite bit patterns, and the rounded-real Sterbenz theorem.
-/
theorem toReal_sub_eq_sub_of_sterbenz (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true)
    (hxpos : 0 < toReal x) (hypos : 0 < toReal y)
    (hxy : toReal x ≤ 2 * toReal y) (hyx : toReal y ≤ 2 * toReal x) :
    toReal (sub x y) = toReal x - toReal y := by
  obtain ⟨dx, hdx⟩ := exists_toDyadic?_of_isFinite hx
  obtain ⟨dy, hdy⟩ := exists_toDyadic?_of_isFinite hy
  have hsub : isFinite (sub x y) = true :=
    isFinite_sub_of_isFinite_of_nonneg x y hx hy hxpos.le hypos.le
  rw [toReal_sub_eq_fp32Round x y hdx hdy hsub]
  exact round32_sub_exact_of_sterbenz
    (toReal_neuralGenericFormat_of_isFinite x hx)
    (toReal_neuralGenericFormat_of_isFinite y hy)
    hxpos hypos hxy hyx

/-- Finite refinement for multiplication: `IEEE32Exec.mul` = exact real mul + float32 rounding. -/
theorem toReal_mul_eq_fp32Round (x y : IEEE32Exec) {dx dy : Dyadic}
    (hx : toDyadic? x = some dx) (hy : toDyadic? y = some dy)
    (hfin : isFinite (mul x y) = true) :
    toReal (mul x y) = fp32Round (toReal x * toReal y) := by
  let prod : Dyadic :=
    { sign := Bool.xor dx.sign dy.sign, mant := dx.mant * dy.mant, exp := dx.exp + dy.exp }
  have hmul : mul x y = roundDyadicToIEEE32 prod :=
    mul_eq_roundDyadicToIEEE32_of_toDyadic? (hx := hx) (hy := hy)
  have hfin' : isFinite (roundDyadicToIEEE32 prod) = true := by
    simpa [hmul] using hfin
  calc
    toReal (mul x y) = toReal (roundDyadicToIEEE32 prod) := by
      simp [hmul]
    _ = fp32Round (dyadicToReal prod) := by
      simpa using (toReal_roundDyadicToIEEE32_eq_fp32Round (d := prod) hfin')
    _ = fp32Round (dyadicToReal dx * dyadicToReal dy) := by
      -- exact dyadic product semantics
      simpa [prod] using congrArg fp32Round (dyadicToReal_mul_exact (a := dx) (b := dy))
    _ = fp32Round (toReal x * toReal y) := by
      simp [toReal_eq, hx, hy]

/-- Finite refinement for fused multiply-add: `fma x y z` rounds `x*y + z` once at the end. -/
theorem toReal_fma_eq_fp32Round (x y z : IEEE32Exec) {dx dy dz : Dyadic}
    (hx : toDyadic? x = some dx) (hy : toDyadic? y = some dy) (hz : toDyadic? z = some dz)
    (hfin : isFinite (fma x y z) = true) :
    toReal (fma x y z) = fp32Round (toReal x * toReal y + toReal z) := by
  let prod : Dyadic :=
    { sign := Bool.xor dx.sign dy.sign, mant := dx.mant * dy.mant, exp := dx.exp + dy.exp }
  have hfma : fma x y z = roundDyadicToIEEE32 (addDyadic prod dz) :=
    fma_eq_roundDyadicToIEEE32_of_toDyadic? (hx := hx) (hy := hy) (hz := hz)
  have hfin' : isFinite (roundDyadicToIEEE32 (addDyadic prod dz)) = true := by
    simpa [hfma] using hfin
  calc
    toReal (fma x y z) = toReal (roundDyadicToIEEE32 (addDyadic prod dz)) := by
      simp [hfma]
    _ = fp32Round (dyadicToReal (addDyadic prod dz)) := by
      simpa using (toReal_roundDyadicToIEEE32_eq_fp32Round (d := addDyadic prod dz) hfin')
    _ = fp32Round (dyadicToReal prod + dyadicToReal dz) := by
      rw [dyadicToReal_addDyadic_exact (a := prod) (b := dz)]
    _ = fp32Round (dyadicToReal dx * dyadicToReal dy + dyadicToReal dz) := by
      -- dyadic product semantics inside the sum
      simpa [prod] using
        congrArg fp32Round
          (congrArg (fun r : ℝ => r + dyadicToReal dz) (dyadicToReal_mul_exact (a := dx) (b := dy)))
    _ = fp32Round (toReal x * toReal y + toReal z) := by
      simp [toReal_eq, hx, hy, hz]

/--
Finite refinement for division.

At the executable level, division is implemented by forming an exact rational quotient `num/den`
(after aligning dyadic exponents) and then rounding that rational to float32. This theorem states
that the overall real meaning is `FP32` rounding of real division.
-/
theorem toReal_div_eq_fp32Round (x y : IEEE32Exec) {dx dy : Dyadic}
    (hx : toDyadic? x = some dx) (hy : toDyadic? y = some dy) (hy0 : dy.mant ≠ 0)
    (hfin : isFinite (div x y) = true) :
    toReal (div x y) = fp32Round (toReal x / toReal y) := by
  -- Reduce IEEE32 division to rounding an exact rational quotient.
  have hdiv :
      div x y =
        let sign : Bool := Bool.xor dx.sign dy.sign
        let eDiff : Int := dx.exp - dy.exp
        let (num, den) :=
          match eDiff with
          | .ofNat sh => (Nat.shiftLeft dx.mant sh, dy.mant)
          | .negSucc sh => (dx.mant, Nat.shiftLeft dy.mant (sh + 1))
        roundRatToIEEE32 sign num den :=
    div_eq_roundRatToIEEE32_of_toDyadic? (hx := hx) (hy := hy) hy0
  have hfin' :
      isFinite
          (let sign : Bool := Bool.xor dx.sign dy.sign
            let eDiff : Int := dx.exp - dy.exp
            let (num, den) :=
              match eDiff with
              | .ofNat sh => (Nat.shiftLeft dx.mant sh, dy.mant)
              | .negSucc sh => (dx.mant, Nat.shiftLeft dy.mant (sh + 1))
            roundRatToIEEE32 sign num den) = true := by
    simpa [hdiv] using hfin
  -- Real semantics of the exact quotient is `toReal x / toReal y`.
  let sign : Bool := Bool.xor dx.sign dy.sign
  cases hE : (dx.exp - dy.exp) with
  | ofNat sh =>
      let num : Nat := Nat.shiftLeft dx.mant sh
      let den : Nat := dy.mant
      have hden0 : den ≠ 0 := by
        simpa [den] using hy0
      have htoRat :
          toReal x / toReal y = (if sign then (-1 : ℝ) else 1) * ((num : ℝ) / (den : ℝ)) := by
        have hrat := dyadicToReal_div_eq_signedRat_mul (dx := dx) (dy := dy) hy0
        have hrat' :
            dyadicToReal dx / dyadicToReal dy =
              (if sign then (-1 : ℝ) else 1) * ((num : ℝ) / (den : ℝ)) := by
          simpa (config := { zeta := true }) [sign, hE, num, den] using hrat
        simpa [toReal_eq, hx, hy] using hrat'
      have hfinCase : isFinite (roundRatToIEEE32 sign num den) = true := by
        simpa (config := { zeta := true }) [sign, hE, num, den] using hfin'
      -- Apply the rounding refinement theorem.
      calc
        toReal (div x y) = toReal (roundRatToIEEE32 sign num den) := by
          simp (config := { zeta := true }) [hdiv, sign, hE, num, den]
        _ = fp32Round ((if sign then (-1 : ℝ) else 1) * ((num : ℝ) / (den : ℝ))) := by
          simpa using (toReal_roundRatToIEEE32_eq_fp32Round (sign := sign) (num := num) (den := den)
            hden0 hfinCase)
        _ = fp32Round (toReal x / toReal y) := by
          rw [htoRat]
  | negSucc sh =>
      let num : Nat := dx.mant
      let den : Nat := Nat.shiftLeft dy.mant (sh + 1)
      have hden0 : den ≠ 0 := by
        intro h0
        have hmul : dy.mant * 2 ^ (sh + 1) = 0 := by
          simpa [den, Nat.shiftLeft_eq] using h0
        have : dy.mant = 0 := by
          have : dy.mant = 0 ∨ 2 ^ (sh + 1) = 0 := Nat.mul_eq_zero.mp hmul
          cases this with
          | inl h => exact h
          | inr hpow =>
              have hpos : 0 < 2 ^ (sh + 1) :=
                Nat.pow_pos (a := 2) (n := sh + 1) (by decide : 0 < (2 : Nat))
              exact False.elim ((Nat.ne_of_gt hpos) hpow)
        exact (hy0 this).elim
      have htoRat :
          toReal x / toReal y = (if sign then (-1 : ℝ) else 1) * ((num : ℝ) / (den : ℝ)) := by
        have hrat := dyadicToReal_div_eq_signedRat_mul (dx := dx) (dy := dy) hy0
        have hrat' :
            dyadicToReal dx / dyadicToReal dy =
              (if sign then (-1 : ℝ) else 1) * ((num : ℝ) / (den : ℝ)) := by
          simpa (config := { zeta := true }) [sign, hE, num, den] using hrat
        simpa [toReal_eq, hx, hy] using hrat'
      have hfinCase : isFinite (roundRatToIEEE32 sign num den) = true := by
        simpa (config := { zeta := true }) [sign, hE, num, den] using hfin'
      calc
        toReal (div x y) = toReal (roundRatToIEEE32 sign num den) := by
          simp (config := { zeta := true }) [hdiv, sign, hE, num, den]
        _ = fp32Round ((if sign then (-1 : ℝ) else 1) * ((num : ℝ) / (den : ℝ))) := by
          simpa using (toReal_roundRatToIEEE32_eq_fp32Round (sign := sign) (num := num) (den := den)
            hden0 hfinCase)
        _ = fp32Round (toReal x / toReal y) := by
          rw [htoRat]

end IEEE32Exec

end TorchLean.Floats.IEEE754

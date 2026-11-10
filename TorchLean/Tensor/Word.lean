/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity

/-!
# Integer words

An integer of `w` bits is a word `BitVec w`, read as a two's-complement integer when its type is
signed and as its word otherwise (`toInteger`), and `fits` says whether an integer is a value of
the type. Where `BitVec` does not provide an integer operator in the form the frontends define, we
define it on words: shifts by an amount that is itself a word, the upper half of a product,
PyTorch's integer division, remainder and power, and the order of a type (`divTrunc`, `divFloor`,
`remainder`, `pow`, `lt`, `le`, `maximum`, `minimum`).

A shift amount is its word read as an unsigned integer, whatever the signedness of its type, so an
encoded `-1` is a large amount. Below the width the shifts are `BitVec`'s. At an amount of the
width or more every bit is shifted out: the left and logical right shifts return zero and the
arithmetic right shift returns the top bit in every position (`shiftLeft_of_le`,
`shiftRightLogical_of_le`, `shiftRightArithmetic_of_le`). StableHLO leaves those amounts open and
PyTorch documents them as undefined, so the values there are TorchLean's choice, the ones XLA's
CPU code and PyTorch's CPU kernels compute (measured, jax 0.12 and torch 2.13 on x86-64). The
arithmetic shift fills with the top bit on an unsigned type too. Capping the amount at the width
before shifting also keeps the shift from building a number of `2 ^ 64` bits. Torch 2.8.0, the
version the parity checks pin, gives those values too: in int32 `1 << 32` is `0`, `-8 >> 32` is
`-1` and `-5 >> -1` is `-1`. The test `test_integer_bitwise_operators_and_shifts` in
`scripts/checks/test_ir_capture.py` compares the shifts with Python integer arithmetic.

We took the division, remainder and power by hand from the PyTorch 2.8 docs and ATen at v2.8.0.
`divTrunc` and `divFloor` are the `"trunc"` and `"floor"` rounding modes of `torch.div`, and
`remainder` is the formula `a - a.div(b, rounding_mode="floor") * b` of `torch.remainder`, so
`[-7, 7, -7, 7]` by `[3, -3, -3, 3]` leaves `[2, -2, -1, 1]` in both. `pow` is `powi` of
`aten/src/ATen/native/Pow.h`, wrapping (`3 ^ 25` is `1180052131` in int32) and with its answers at
a negative exponent. At a zero divisor we keep the `BitVec` values, `0` for both quotients and `x`
for the remainder, where the `torch.fmod` page says torch "raises RuntimeError for integer
division by zero on CPU". The capture test `test_integer_division_negation_and_power` compares
both divisions and a cube with eager PyTorch in int32.

`mulhi` is the floor of the exact product over `2 ^ w`, of the signed values for a signed type
(`toInt_mulhi_true`) and of the words otherwise (`toNat_mulhi_false`). The other operators are
`BitVec`'s: `&&&`, `|||`, `^^^`, `~~~`, `BitVec.cpop`, the number of one bits (`cpop_eq`), and
`BitVec.clz`, the number of zero bits above the highest one bit, the width at zero (`clz_eq`).
-/

@[expose] public section

namespace TorchLean.Word

variable {w : Nat}

/-- The value of a word of a signed or an unsigned integer type: its two's-complement value when
`signed`, the natural number it writes otherwise. -/
def toInteger (signed : Bool) (x : BitVec w) : Int :=
  if signed then x.toInt else x.toNat

/-- A word is the word of its value. -/
@[simp] theorem ofInt_toInteger (signed : Bool) (x : BitVec w) :
    BitVec.ofInt w (toInteger signed x) = x := by
  cases signed <;> simp [toInteger]

/-- Whether `n` is a value of the integer type of `width` bits: in `[-2 ^ (width - 1),
2 ^ (width - 1))` for a signed type and in `[0, 2 ^ width)` otherwise. -/
def fits (signed : Bool) (width : Nat) (n : Int) : Bool :=
  if signed then decide (-(2 ^ (width - 1) : Int) ≤ n ∧ n < 2 ^ (width - 1))
  else decide (0 ≤ n ∧ n < 2 ^ width)

/-- `x` shifted left by the amount `n` read unsigned, zero once it reaches the width. -/
def shiftLeft (x n : BitVec w) : BitVec w := x <<< min n.toNat w

/-- `x` shifted right by the amount `n` read unsigned, filled with the top bit of `x`. -/
def shiftRightArithmetic (x n : BitVec w) : BitVec w := x.sshiftRight (min n.toNat w)

/-- `x` shifted right by the amount `n` read unsigned, filled with zeros. -/
def shiftRightLogical (x n : BitVec w) : BitVec w := x >>> min n.toNat w

/-- The upper half of the exact product of `x` and `y`: the floor of the product of their signed
values over `2 ^ w` when `signed`, of their words otherwise. -/
def mulhi (signed : Bool) (x y : BitVec w) : BitVec w :=
  if signed then BitVec.ofInt w ((x.toInt * y.toInt).fdiv (2 ^ w))
  else BitVec.ofNat w (x.toNat * y.toNat / 2 ^ w)

/-- Below the width, `shiftLeft` is `BitVec`'s left shift. -/
theorem shiftLeft_of_lt {x n : BitVec w} (h : n.toNat < w) : shiftLeft x n = x <<< n.toNat := by
  simp [shiftLeft, Nat.min_eq_left h.le]

/-- At the width or more, `shiftLeft` shifts every bit out. -/
theorem shiftLeft_of_le {x n : BitVec w} (h : w ≤ n.toNat) : shiftLeft x n = 0 := by
  apply BitVec.eq_of_toNat_eq
  simp [shiftLeft, Nat.min_eq_right h]

/-- Below the width, `shiftRightLogical` is `BitVec`'s logical right shift. -/
theorem shiftRightLogical_of_lt {x n : BitVec w} (h : n.toNat < w) :
    shiftRightLogical x n = x >>> n.toNat := by
  simp [shiftRightLogical, Nat.min_eq_left h.le]

/-- At the width or more, `shiftRightLogical` shifts every bit out. -/
theorem shiftRightLogical_of_le {x n : BitVec w} (h : w ≤ n.toNat) :
    shiftRightLogical x n = 0 := by
  apply BitVec.eq_of_toNat_eq
  simp only [shiftRightLogical, Nat.min_eq_right h, BitVec.toNat_ushiftRight,
    Nat.shiftRight_eq_div_pow]
  exact Nat.div_eq_of_lt x.isLt

/-- Below the width, `shiftRightArithmetic` is `BitVec`'s arithmetic right shift. -/
theorem shiftRightArithmetic_of_lt {x n : BitVec w} (h : n.toNat < w) :
    shiftRightArithmetic x n = x.sshiftRight n.toNat := by
  simp [shiftRightArithmetic, Nat.min_eq_left h.le]

/-- At the width or more, `shiftRightArithmetic` repeats the top bit in every position. -/
theorem shiftRightArithmetic_of_le {x n : BitVec w} (h : w ≤ n.toNat) :
    shiftRightArithmetic x n = if x.msb then BitVec.allOnes w else 0 := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [shiftRightArithmetic, Nat.min_eq_right h, BitVec.getLsbD_sshiftRight]
  cases x.msb <;> simp [hi, Nat.not_le.mpr hi]

/-- The unsigned `mulhi` is the floor of the product of the words over `2 ^ w`. -/
theorem toNat_mulhi_false (x y : BitVec w) :
    (mulhi false x y).toNat = x.toNat * y.toNat / 2 ^ w := by
  simp only [mulhi, Bool.false_eq_true, ↓reduceIte, BitVec.toNat_ofNat]
  apply Nat.mod_eq_of_lt
  rw [Nat.div_lt_iff_lt_mul (Nat.two_pow_pos w)]
  exact Nat.mul_lt_mul'' x.isLt y.isLt

/-- At a positive width, the signed `mulhi` is the floor of the product of the signed values
over `2 ^ w`. -/
theorem toInt_mulhi_true {x y : BitVec w} (hw : 0 < w) :
    (mulhi true x y).toInt = (x.toInt * y.toInt).fdiv (2 ^ w) := by
  have hx1 := BitVec.le_toInt x
  have hx2 := BitVec.toInt_lt (x := x)
  have hy1 := BitVec.le_toInt y
  have hy2 := BitVec.toInt_lt (x := y)
  have hp : ((2 ^ w : Nat) : Int) = 2 * 2 ^ (w - 1) := by
    rw [Nat.cast_pow, Nat.cast_ofNat, ← pow_succ']; congr 1; omega
  have hq : (0 : Int) < 2 ^ (w - 1) := by positivity
  simp only [mulhi, ↓reduceIte, BitVec.toInt_ofInt]
  have hpow : (2 : Int) ^ w = 2 * 2 ^ (w - 1) := by rw [← pow_succ']; congr 1; omega
  rw [hpow]
  set q : Int := 2 ^ (w - 1)
  set a := x.toInt
  set b := y.toInt
  have hd : (0 : Int) < 2 * q := by omega
  rw [Int.fdiv_eq_ediv_of_nonneg _ hd.le]
  have hc1 := Int.ediv_mul_le (a * b) (ne_of_gt hd)
  have hc2 := Int.lt_ediv_add_one_mul_self (a * b) hd
  set c := a * b / (2 * q)
  have hlo : -(q * q) ≤ a * b := by nlinarith
  have hhi : a * b ≤ q * q := by nlinarith
  have hc3 : c * 2 < 2 * q := by nlinarith
  have hc4 : -(2 * q) ≤ c * 2 := by nlinarith
  exact Int.bmod_eq_of_le_mul_two (by rw [hp]; omega) (by rw [hp]; omega)

/-- `BitVec.cpopNatRec` adds the number of one bits below `n` to its accumulator. -/
theorem cpopNatRec_eq (x : BitVec w) (n acc : Nat) :
    x.cpopNatRec n acc = acc + (List.range n).countP fun i => x.getLsbD i := by
  induction n generalizing acc with
  | zero => simp [BitVec.cpopNatRec]
  | succ n ih =>
    simp only [BitVec.cpopNatRec, ih, List.range_succ, List.countP_append]
    by_cases h : x.getLsbD n <;> simp [h]
    omega

/-- `BitVec.cpop` counts the one bits. -/
theorem cpop_eq (x : BitVec w) :
    x.cpop = BitVec.ofNat w ((List.range w).countP fun i => x.getLsbD i) := by
  simp [BitVec.cpop, cpopNatRec_eq]

/-- For `n < w`, `BitVec.clzAuxRec x n` counts the zero bits from bit `n` down to the highest one
bit, plus the `w - 1 - n` bits above `n`. -/
theorem clzAuxRec_eq (x : BitVec w) (n : Nat) (hn : n < w) :
    x.clzAuxRec n = BitVec.ofNat w
      (w - 1 - n + ((List.range (n + 1)).reverse.takeWhile fun i => !x.getLsbD i).length) := by
  induction n with
  | zero =>
    simp only [BitVec.clzAuxRec]
    cases h : x.getLsbD 0 <;> simp [h]
    congr 1
    omega
  | succ n ih =>
    rw [BitVec.clzAuxRec, List.range_succ, List.reverse_append, List.reverse_singleton,
      List.singleton_append, List.takeWhile_cons]
    cases h : x.getLsbD (n + 1)
    · simp only [Bool.false_eq_true, ↓reduceIte, Bool.not_false, List.length_cons]
      rw [ih (by omega)]
      congr 1; omega
    · simp

/-- `BitVec.clz` counts the zero bits above the highest one bit, all of them at zero. -/
theorem clz_eq (x : BitVec w) :
    x.clz = BitVec.ofNat w ((List.range w).reverse.takeWhile fun i => !x.getLsbD i).length := by
  rcases Nat.eq_zero_or_pos w with rfl | hw
  · exact Subsingleton.elim _ _
  · rw [BitVec.clz, clzAuxRec_eq x (w - 1) (by omega), Nat.sub_add_cancel hw]
    simp

/-- Two's-complement power by repeated squaring, the loop of PyTorch's `powi_impl`. -/
def powWrapping (base : BitVec w) (exponent : Nat) : BitVec w :=
  if _h : exponent = 0 then 1
  else
    let rest := powWrapping (base * base) (exponent / 2)
    if exponent % 2 = 1 then base * rest else rest
termination_by exponent
decreasing_by omega

/-- The quotient rounded toward zero: `sdiv` of the signed values, `udiv` of the words. -/
def divTrunc (signed : Bool) (x y : BitVec w) : BitVec w :=
  if signed then x.sdiv y else x.udiv y

/-- The quotient rounded toward negative infinity: `Int.fdiv` of the signed values wrapped to
`w` bits, `udiv` of the words. -/
def divFloor (signed : Bool) (x y : BitVec w) : BitVec w :=
  if signed then BitVec.ofInt w (x.toInt.fdiv y.toInt) else x.udiv y

/-- PyTorch's `remainder`, with the divisor's sign: `Int.fmod` of the signed values wrapped to
`w` bits, `umod` of the words. -/
def remainder (signed : Bool) (x y : BitVec w) : BitVec w :=
  if signed then BitVec.ofInt w (x.toInt.fmod y.toInt) else x.umod y

/-- PyTorch's integer power: `powWrapping`, and at a negative signed exponent `1` for base `1`,
`±1` by the exponent's parity for base `-1` and `0` otherwise. -/
def pow (signed : Bool) (x y : BitVec w) : BitVec w :=
  if signed && y.toInt < 0 then
    if x = 1 then 1 else if x = -1 then (if y.toInt % 2 = 0 then 1 else -1) else 0
  else powWrapping x y.toNat

/-- Whether `x` is below `y` in the type's order. -/
def lt (signed : Bool) (x y : BitVec w) : Bool :=
  if signed then x.slt y else x.ult y

/-- Whether `x` is at most `y` in the type's order. -/
def le (signed : Bool) (x y : BitVec w) : Bool :=
  if signed then x.sle y else x.ule y

/-- The larger of two words in the type's order, `x` at a tie. -/
def maximum (signed : Bool) (x y : BitVec w) : BitVec w :=
  if lt signed x y then y else x

/-- The smaller of two words in the type's order, `x` at a tie. -/
def minimum (signed : Bool) (x y : BitVec w) : BitVec w :=
  if lt signed y x then y else x

/-- The order of words is the order of their values. -/
theorem lt_eq (signed : Bool) (x y : BitVec w) :
    lt signed x y = decide (toInteger signed x < toInteger signed y) := by
  cases signed <;> simp [lt, toInteger] <;> rfl

/-- The non-strict order of words is the order of their values. -/
theorem le_eq (signed : Bool) (x y : BitVec w) :
    le signed x y = decide (toInteger signed x ≤ toInteger signed y) := by
  cases signed <;> simp [le, toInteger] <;> rfl

/-- Two words are equal exactly when their values are. -/
theorem toInteger_inj (signed : Bool) {x y : BitVec w} :
    toInteger signed x = toInteger signed y ↔ x = y := by
  refine ⟨fun h => ?_, congrArg _⟩
  rw [← ofInt_toInteger signed x, h, ofInt_toInteger]

/-- The maximum does not depend on the order of its operands. -/
theorem maximum_comm (signed : Bool) (x y : BitVec w) :
    maximum signed x y = maximum signed y x := by
  simp only [maximum, lt_eq, decide_eq_true_eq]
  split_ifs with h₁ h₂ h₂
  · omega
  · rfl
  · rfl
  · exact (toInteger_inj signed).1 (by omega)

/-- The minimum does not depend on the order of its operands. -/
theorem minimum_comm (signed : Bool) (x y : BitVec w) :
    minimum signed x y = minimum signed y x := by
  simp only [minimum, lt_eq, decide_eq_true_eq]
  split_ifs with h₁ h₂ h₂
  · omega
  · rfl
  · rfl
  · exact (toInteger_inj signed).1 (by omega)

end TorchLean.Word

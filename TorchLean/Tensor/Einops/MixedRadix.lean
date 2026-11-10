/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

/-!
# Mixed-radix arithmetic

For an encoded value `remainder + radix * digit` with `remainder < radix`, quotient and
remainder by `radix` recover the two digits. The compiler uses these lemmas to cancel
coordinate decoding after a checked transform has been composed with a tensor index.
-/

@[expose] public section

namespace TorchLean.Einops.MixedRadix

/--
Dividing an encoded mixed-radix value by its radix recovers the higher digit.
-/
theorem div_encode (remainder digit radix : Nat) (h : remainder < radix) :
    (remainder + radix * digit) / radix = digit := by
  rw [Nat.add_mul_div_left _ _ (Nat.zero_lt_of_lt h)]
  simp only [Nat.div_eq_of_lt h, Nat.zero_add]

/--
Taking the remainder of an encoded mixed-radix value recovers its lower digit.
-/
theorem mod_encode (remainder digit radix : Nat) (h : remainder < radix) :
    (remainder + radix * digit) % radix = remainder := by
  rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt h]

/--
After removing one lower radix, quotient and remainder at the next radix
reconstruct the remaining mixed-radix index.
-/
theorem div_eq_mod_add_mul_div
    (value lowerRadix upperRadix : Nat) :
    (value / lowerRadix) % upperRadix +
        upperRadix * (value / (lowerRadix * upperRadix)) =
      value / lowerRadix := by
  rw [← Nat.div_div_eq_div_mul]
  exact Nat.mod_add_div (value / lowerRadix) upperRadix

end TorchLean.Einops.MixedRadix

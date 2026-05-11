/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.MkBitsToDyadic
public import NN.Floats.IEEEExec.RealSemantics

/-!
# Decoding `mkBits` back to reals

Many IEEE32Exec proofs need to reason about floats that are constructed directly from their
bitfields:

```
ofBits (mkBits sign exp frac)
```

When `exp < 255` (i.e. not all-ones) and `frac < 2^23`, this is a *finite* float32 value
(normal/subnormal/zero) and we can compute its real meaning explicitly.

This module exposes the finite-case decoding lemma as part of the IEEEExec proof interface, so
the bridge, interval, and runtime-approximation developments share the same closed-form real
expression.
-/

@[expose] public section


namespace TorchLean.Floats.IEEE754
namespace IEEE32Exec

open TorchLean.Floats

/-!
## Main decoding lemma

Recall:
- if `exp = 0` and `frac = 0` we have signed zero (both map to real `0`),
- if `exp = 0` and `frac ≠ 0` we have a subnormal with value `frac * 2^-149`,
- if `exp ≠ 0` we have a normal with value `(2^23 + frac) * 2^(exp - 150)`.

All of this is encoded in `toDyadic?` and exposed here via `toReal`.
-/

/--
Decode `ofBits (mkBits sign exp frac)` to a closed-form real expression (finite case).

Assumptions:
- `exp < 255` so the exponent field is not all ones (excluding NaN/Inf),
- `frac < 2^23` so the fraction fits in the binary32 layout.

Conclusion: `toReal` returns the standard IEEE-754 normal/subnormal/zero formulas.
-/
theorem toReal_ofBits_mkBits_fin (sign : Bool) (exp frac : Nat)
    (hexp : exp < 255) (hfrac : frac < 2 ^ 23) :
    toReal (ofBits (mkBits sign exp frac)) =
      if exp = 0 then
        if frac = 0 then 0
        else if sign then -((frac : ℝ) * neuralBpow binaryRadix (-149))
        else (frac : ℝ) * neuralBpow binaryRadix (-149)
      else
        if sign then
          -(((pow2 23 + frac : Nat) : ℝ) * neuralBpow binaryRadix ((Int.ofNat exp) - 150))
        else
          ((pow2 23 + frac : Nat) : ℝ) * neuralBpow binaryRadix ((Int.ofNat exp) - 150) := by
  -- Decode the bits to a dyadic.
  have hdy :=
    toDyadic?_ofBits_mkBits_fin (sign := sign) (exp := exp) (frac := frac) hexp hfrac
  -- Compute `toReal` by rewriting `toDyadic?` using `hdy`.
  simp [toReal_eq, dyadicToReal]
  -- Replace the `toDyadic?` call with the explicit `if exp=0 then ...` expression.
  rw [hdy]
  -- Now split the `exp=0` / `frac=0` / `sign` cases and finish by unfolding `neural_bpow`.
  by_cases hexp0 : exp = 0 <;> by_cases hfrac0 : frac = 0 <;> cases hsign : sign <;>
    simp [hexp0, hfrac0, neuralBpow, binaryRadix, NeuralRadix.toReal, pow2, Nat.shiftLeft_eq]

end IEEE32Exec
end TorchLean.Floats.IEEE754

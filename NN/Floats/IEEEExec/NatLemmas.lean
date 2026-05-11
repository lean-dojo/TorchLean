/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.Nat.Bitwise
public import NN.Floats.IEEEExec.Exec32

/-!
# Small `Nat` lemmas for the IEEE32Exec float kernel

The executable IEEE-754 kernel (`NN/Floats/IEEEExec/Exec32.lean`) defines a handful of low-level
`Nat` helpers such as:

* `pow2 k = 2^k` (as a `Nat`), implemented via `Nat.shiftLeft`.

The IEEEExec bridge and soundness proofs use the same facts about these helpers, so the shared
lemmas live here.
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754
namespace IEEE32Exec

/--
`pow2 k` is just `2^k`.

Informal: `pow2` is defined as `Nat.shiftLeft 1 k`, i.e. shifting the bit `1` left by `k` places,
which equals the power of two `2^k`.
-/
theorem pow2_eq_two_pow (k : Nat) : pow2 k = 2 ^ k := by
  simp [pow2, Nat.shiftLeft_eq]

/--
`pow2 k` is strictly positive.

Informal: `2^k > 0` for all `k`, hence the same holds for `pow2 k`.
-/
theorem pow2_pos (k : Nat) : 0 < pow2 k := by
  simp [pow2_eq_two_pow]

end IEEE32Exec
end TorchLean.Floats.IEEE754

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Nicolas Rouquette, TorchLean Team
-/

module


public import NN.Floats.FP32.Notation

/-!
# Exact Binary32 Subtraction

Sterbenz's lemma specialized to the rounded-real binary32 configuration
`fexp32 = fltExp (-149) 24`. Two positive representable values within a factor of two have an
exactly representable difference, so rounding that difference does nothing.

## Reference

- P. H. Sterbenz, *Floating-Point Computation*, Prentice-Hall, 1974.
-/

@[expose] public section

open FloatLib.Numerics FloatLib.Floats.Formats.Flocq

namespace TorchLean.Floats

/--
If two positive binary32-representable reals are within a factor of two, rounding their exact
difference is the identity.
-/
theorem round32_sub_exact_of_sterbenz {u v : ℝ}
    (hu : genericFormat binaryRadix fexp32 u)
    (hv : genericFormat binaryRadix fexp32 v)
    (hupos : 0 < u) (hvpos : 0 < v) (huv : u ≤ 2 * v) (hvu : v ≤ 2 * u) :
    round32 (u - v) = u - v := by
  have hfmt : genericFormat binaryRadix fexp32 (u - v) :=
    generic_format_FLT_sterbenz (-149) 24 (by decide) hupos hvpos huv hvu hu hv
  exact round_preserves_generic (β := binaryRadix) (fexp := fexp32) rnd32 (u - v) hfmt

namespace FP32

/--
Subtraction of positive representable `FP32` values within a factor of two has no rounding error.
-/
theorem sub_exact_of_sterbenz {a b : FP32}
    (ha : a.IsRepresentable) (hb : b.IsRepresentable)
    (hapos : 0 < a.val) (hbpos : 0 < b.val)
    (hab : a.val ≤ 2 * b.val) (hba : b.val ≤ 2 * a.val) :
    (a - b).val = a.val - b.val := by
  change round32 (a.val - b.val) = a.val - b.val
  exact round32_sub_exact_of_sterbenz ha hb hapos hbpos hab hba

end FP32

end TorchLean.Floats

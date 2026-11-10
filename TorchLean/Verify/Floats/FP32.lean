/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Floats.Formats.BinaryInterchange.Analysis.Error
public import FloatLib.Floats.Formats.Flocq

/-!
# Rounded-real binary32 (`FP32`)

`FP32` specializes FloatLib's `NF` to binary32 precision with gradual underflow and no upper
exponent bound. Each arithmetic operation rounds its real result to this grid. The type stores a
real value with no representability invariant, so round an input with `NF.ofReal`.

We built it by hand from Flocq's FLT grid, the `Model.fexpOf FloatFormat.binary32` of 24
significant bits down to the subnormal spacing `2^-149`. `nearestEven` makes each step
IEEE 754-2019's roundTiesToEven (4.3.1). The missing upper bound shows in `2^127 + 2^127`,
which is `2^128` here and overflows to `+∞` in binary32.

The rounding theorems come straight from FloatLib: `Model.abs_roundAt_sub_le fmt` bounds one
rounding step, and `round_nearestEven_computed` gives its mantissa/exponent form. Bit-level
encodings and exceptional values (infinities, NaNs) belong to `ExecFloat.Binary 8 23`.
-/

@[expose] public section

open FloatLib.Numerics FloatLib.Floats.Formats.Flocq
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

namespace TorchLean

/-- Rounded-real binary32 arithmetic, without an upper exponent bound. -/
abbrev FP32 : Type := NF binaryRadix (Model.fexpOf FloatFormat.binary32) nearestEven

end TorchLean

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.Floats.FP32
public import TorchLean.Tensor.NF
public import TorchLean.Verify.CROWN.BoundOps.Interval

/-!
# Bound-propagation instances for `FP32`

`BoundOps`, `LawfulBoundOps`, `NonlinearBoundOps` and `LawfulNonlinearBoundOps` instances for the
rounded-real `FP32` model. Each applies the outward-rounding construction of
`TorchLean.Verify.CROWN.BoundOps.Interval` at the binary32 rounder: the exact real operation
or transfer, with lower endpoints rounded down and upper endpoints rounded up to the binary32 grid.

Because `FP32` holds a real number, the arithmetic instances are `noncomputable` and serve proofs
only. They let a proof state soundness for float32 semantics without mentioning Lean's `Float`.

`FP32` models binary32 with an unbounded exponent: 24-bit significands and gradual underflow, but
no overflow, infinities or NaNs. The nonlinear bounds use `Real.exp` and other real functions, so
results here hold for this proof model and say nothing about overflow on binary32 hardware.
-/

@[expose] public section

open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)
open FloatLib.Floats.Interval (formatRounder)

open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq
open TorchLean (FP32)

namespace TorchLean.CROWN

/-! ## FP32 instances -/

/-- Exact real endpoint arithmetic rounded outward to the binary32 grid. -/
noncomputable instance : BoundOps FP32 :=
  .outward (formatRounder binaryRadix (Model.fexpOf FloatFormat.binary32))

/--
The FP32 endpoint operations enclose exact real arithmetic, and `1` is a binary32 value. Sound
rounded CROWN statements over `FP32` use these laws.
-/
noncomputable instance : LawfulBoundOps FP32 :=
  .outward (formatRounder binaryRadix (Model.fexpOf FloatFormat.binary32)) <| by
    simpa using generic_format_bpow (β := binaryRadix)
      (fexp := Model.fexpOf FloatFormat.binary32) 0 (by decide)

/-- Exact real nonlinear transfers rounded outward to the binary32 grid. -/
noncomputable instance : NonlinearBoundOps FP32 :=
  .outward (formatRounder binaryRadix (Model.fexpOf FloatFormat.binary32))

/-- The FP32 nonlinear transfers enclose the real functions they bound. -/
instance : LawfulNonlinearBoundOps FP32 :=
  .outward (formatRounder binaryRadix (Model.fexpOf FloatFormat.binary32)) _

end TorchLean.CROWN

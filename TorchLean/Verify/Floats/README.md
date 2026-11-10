# Floating-point bridges

These modules connect FloatLib's executable IEEE arithmetic to the rounded-real model that
TorchLean's numerical proofs use. The runtime-approximation, CROWN, approximation, and LibTorch
contract proofs in `TorchLean/Verify` import them.

- `FP32.lean`: the rounded-real binary32 type `TorchLean.FP32`.
- `IEEE754.lean`: finite rounding equalities for configured IEEE arithmetic, in
  `TorchLean.IEEE754`.

```lean
import TorchLean.Verify.Floats.IEEE754
open TorchLean.IEEE754
```

## Rounded reals and encoded values

`FP32` abbreviates `NF binaryRadix (Model.fexpOf FloatFormat.binary32) nearestEven`, and its
exponent function unfolds to `fltExp (-149) 24`. It rounds each real operation to a grid with 24
significant bits and gradual underflow down to the binary32 subnormal scale.
**Its exponent has no upper bound.** NaNs, infinities, and signed zeros are absent, so `FP32`
has no largest element. Transferring an executable IEEE result into this model needs a
finiteness hypothesis on that result.

FloatLib's descriptor-generic rounding theorems apply to `FP32` with `FloatFormat.binary32`:
`Model.abs_roundAt_sub_le` and `Model.roundAt_mem_Icc` (half-ULP error and enclosure) and
`Model.relativeError_roundAt_le_of_normal` (relative error in the normal range) in
`FloatLib/Floats/Formats/BinaryInterchange/Analysis/Error.lean`. In `.../Analysis/Sterbenz.lean`,
`Model.roundAt_sub_eq_of_sterbenz` gives exact subtraction for nonnegative representable values
within a factor of two, and `Model.roundAt_sub_eq_of_sterbenz_of_same_sign` gives it for operands
of one sign.

For executable encoded arithmetic, use `ExecFloat.Binary 8 23` for binary32. The two parameters
count exponent and stored fraction bits. FloatLib owns arithmetic, special-value behavior,
and word/model conversions. Its `ofBits32` and `toBits32` preserve every encoded bit. Native
`Float32` conversions have the separate native NaN boundary documented in the source.

FloatLib's `IEEE754.Native` modules supply exact interchange-word round trips and
configured/native conversion equations. `Native.AddSub` proves equality for finite inputs,
including both signed zeros and results that overflow to infinity. `Native.Sqrt` covers every
input after NaN canonicalization. These theorems concern Lean's logical float model. They do not
verify a compiler's floating-point instructions or the host's floating-point environment.

Other precisions use `ExecFloat.Binary` with other parameters, and FloatLib provides their
configured arithmetic, rational conversion, rounding modes, and status APIs. A wider carrier does
not make a conversion through native `Float` exact, so code that needs more than binary64
precision must keep values in the configured carrier and use its exact conversion interfaces.

`TorchLean.Tensor.ConfiguredBinary` imports
`FloatLib.Floats.Formats.BinaryInterchange.Configured.Transcendentals`. The
`ExecFloat.Binary.exp`, `log`, and other elementary functions defined there are deterministic
approximations with no real-error or correct-rounding theorem. For IEEE formats,
`ExecFloat.Binary.CorrectlyRounded.exp`, `log`, `expMinus1`, `logPlus1`, and `expMul` are
correctly rounded. TorchLean's configured `Context` uses all five on IEEE formats, as
`ConfiguredBinary.exp_of_isIEEE`, `log_of_isIEEE`, `log1p_of_isIEEE`, `expm1_of_isIEEE` and
`expMul_of_isIEEE` state.

In `TorchLean/Tensor/ConfiguredBinary/InverseTrigonometric.lean`, TorchLean's own
`InverseTrigonometric.atan`, `asin`, `acos`, and `atan2` round the real function once in every
IEEE format, and their `ConfiguredBinary` versions form the `InverseTrig` instance of every
configured format. `TorchLean/Verify/RuntimeApprox/InverseTrigonometric.lean` proves the rounding,
the finiteness of results at finite inputs in the domain, and the special values. For proved square
root without the transcendental approximations, import FloatLib's configured rounding modules.
`TorchLean.Tensor.Numeric` supplies the native scalar instances, including a `NatCast Float32` that
rounds naturals to binary32 without an intermediate binary64 rounding.

## Proofs at the TorchLean boundary

FloatLib's `ExecFloat.Binary.toReal_add_eq_roundAt`, `toReal_sub_eq_roundAt`,
`toReal_mul_eq_roundAt` and `toReal_fma_eq_roundAt` identify a finite configured result with one
rounding of the exact operation. `IEEE754.lean` adds the division and square-root
theorems, which require only a finite result, and the decoding facts that the reduction proofs
use. FloatLib's `ieee_format` tactic discharges the IEEE format hypothesis by default, so binary32
code applies the same theorems. Subnormal results need no global relative-error assumption.

Native proofs import FloatLib's native bridge, and the tensor error bounds use `IEEE754.lean`.

For native addition, the finite-input premise is part of the theorem. Square root compares
canonicalized results because Lean's model has one NaN encoding while the configured carrier
can retain payloads:

```lean
import FloatLib.Floats.Formats.IEEE754.Native.AddSub
import FloatLib.Floats.Formats.IEEE754.Native.Sqrt

open FloatLib.Floats
open FloatLib.Floats.ExecFloat.Binary
open FloatLib.Floats.Formats.BinaryInterchange

example (x : Float32) :
    toFloat32 (ofFloat32 x : Binary32) = x :=
  toFloat32_ofFloat32 x

example (x y : Float32) (hx : x.isFinite = true) (hy : y.isFinite = true) :
    (ofFloat32 (x + y) : Binary32) = ofFloat32 x + ofFloat32 y :=
  ofFloat32_add_of_isFinite x y hx hy

example (x : Float32) :
    toModel (ofFloat32 x.sqrt : Binary32) =
      Model.canonicalizeModel (Model.sqrt (toModel (ofFloat32 x : Binary32))) :=
  toModel_ofFloat32_sqrt x
```

Configured arithmetic has total software-model refinements. Native CPU instructions and LibTorch
have separate execution contracts recorded in `docs/TRUST_BOUNDARIES.md`.

`TorchLean.Verify.Floats.IEEE754.Expressions` defines the scalar expression AST over any configured
IEEE format and proves its finite-evaluation refinement `Expr.toReal_eval` from the
FloatLib and `IEEE754.lean` rounding equalities. The ridge-regression example in
`TorchLeanExamples/DeepDives/Stability/RidgeRegression1D.lean` uses it in any configured format,
with a binary32 dataset.

`TorchLean.Verify.Floats.IEEE754.Reductions` evaluates FloatLib's `ReductionTree` schedules with
the executable operations of a configured IEEE format and applies FloatLib's reduction-tree error
bounds. These bounds account for rounding at each node of the chosen tree, whereas FloatLib's exact
sum and dot-product accumulators round once, at the end.

The enclosures `evalSum_enclosure` and `SumResult.enclosure` use the
format's unit roundoff `Model.unitRoundoffAt format` (`2⁻²⁴` for binary32) on the tree's
representable operands. They require finite leaves and finite intermediate sums, and they cover
subnormal sums without a global relative-error bound on all real inputs.
`DotResult.enclosure` bounds the accumulation relative to the exact sum of the
already-rounded products. A bound on the real dot product must also account for the
multiplications. An external reduction needs evidence that it follows one of these tree
schedules.

The nested derivative quotient repair is TorchLean code, in `TorchLean.Tensor.Numeric.Quotient` and
`TorchLean.NN.Dual`. FloatLib supplies exact finite rational decoding, scalar
rounding, and configured operation status. TorchLean detects unsafe intermediates and, when
needed, replays the complete nested coefficient tree. A finite exact derivative can overflow its
destination format, and neither the scalar library nor the replay can produce a finite encoded
result for a derivative the format cannot represent.

## FloatLib APIs used directly

A rounding-error proof compares real values, while a claim about the sign of zero or a NaN payload
needs equality of encodings. `FloatLib.Floats.Formats.Flocq` supplies `NF` and the
`Calculation.Round`, `.Operations`, `.Arithmetic` and `.Bracket` proof modules. Its
`round_nearestEven_computed` writes nearest-even rounding as an integer selection on the scaled
mantissa at the canonical exponent. Encoded-format consumers import
`FloatLib.Floats.Formats.BinaryInterchange` and use `ExecFloat.Binary.toModel` for the exact
configured-to-model conversion.

On eligible formats, configured division and square root run FloatLib's certified word-accumulator
kernels. A finite cast between formats with the same exponent width, bias and encoding into a
fraction at least as wide compiles to one shift of the stored word. FloatLib's exact sums and dot
products also determine the sign of a zero total and keep an invalid flag raised by any term. None
of these changes the accumulation order of TorchLean's tensor sums or of native GPU kernels.

For real inequalities, `interval` from `FloatLib.Numerics.Automation.Interval` proves bounds
directly. It handles real powers on a strictly positive base interval, `Real.logb` and the inverse
hyperbolic functions, checking each operation's domain. `@[interval_extension]` registers a
function's executable enclosure with its containment theorem, and the tactic evaluates that
enclosure again on each subdivided input box. The floating-point guide in
`home_page/blueprint/TorchLeanBlueprint/Guide/Ch3_Backend/Floats.lean` proves one such bound.
These are bounds on real expressions, not checks of native GPU output.

`BinaryInterchange.Model.Interval` gives format-generic outward-rounded interval arithmetic, and
`FloatLib.Floats.Interval` holds real-valued enclosure proofs whose endpoints move outward through
directed rounders. Binary32 interval endpoints are
`FloatLib.Numerics.Interval (ExecFloat.Binary 8 23)`, with arithmetic and conversion proofs in
`ExecFloat.Binary.Interval`. TorchLean's numerical graph certificates use these endpoints, and
selecting another scalar format does not supply a graph certificate for it. FloatLib's
`RealAffineQuantizer` acts on a shaped tensor through `Tensor.map`, and the facts next to
`Tensor.forall_map` carry its code-range, monotonicity, round-trip and error theorems to every
coordinate. Its integer code bounds `qmin` and `qmax` cover int8, uint8, int4 and custom code
ranges.

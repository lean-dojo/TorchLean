# Floating-point models used by TorchLean

`NN.Floats` connects FloatLib's scalar arithmetic to TorchLean's tensor and runtime proofs.
FloatLib supplies the generic formats, exact decoding, rounding algorithms, arithmetic kernels,
error theory, and interval operations. TorchLean retains rounded-real specializations,
interval endpoint contracts, quantization, and the proofs needed by tensor and runtime consumers.

The dependency follows FloatLib's `main` branch, with its resolved revision locked in
`lake-manifest.json`, using Lean and mathlib 4.34.0. Run `scripts/lake.sh update floatlib` and
rebuild to adopt upstream changes; ordinary builds use the locked revision.
Import `NN.Floats` for the adapters, or import FloatLib directly when a program needs a generic
scalar API.

The current dependency includes certified word-accumulator implementations for division and
square root, and a whole-word shift for compatible format widening. Existing configured scalar
operations select these implementations when their format is eligible; TorchLean does not need
another arithmetic wrapper. FloatLib also supplies the new exact-reduction results for signed
zeros and exception flags. These are available through the upstream imports; they do not change
the accumulation order of TorchLean's ordinary tensor sums or native GPU kernels.

```lean
import NN.Floats
open TorchLean.Floats
```

A sum can require different models depending on the claim. A rounding-error proof may compare
its real value with an exact real sum. A program that inspects the sign of zero or a NaN payload
needs an equality of encodings instead. These models keep that distinction explicit.

| Purpose | Scalar API | TorchLean adapter |
| --- | --- | --- |
| Rounded reals and error bounds | `Flocq.NF` | `FP32/` |
| Configurable encoded formats | `ExecFloat.Binary` | Direct FloatLib API |
| Computed real rounding | Flocq calculation theorems | `FP32.round_eq_computed` |
| Outward intervals | FloatLib interval models | `Interval/FP32.lean`, external Arb adapters |
| Affine quantization | FloatLib rounding policies | `Quantization.lean` |
| Native Lean float expressions | FloatLib `IEEE754.Native` proofs | Direct upstream imports |

Here `Flocq` is `FloatLib.Floats.Formats.Flocq`, and `ExecFloat` is
`FloatLib.Floats.ExecFloat`. `FP32/Core.lean` supplies the specialization,
`FP32/Error.lean` its error corollaries, and `FP32/Sterbenz.lean` exact-subtraction results.
The interval APIs are `FloatLib.Floats.Interval` and `BinaryInterchange.Model.Interval`.

Format widths and exponent bounds come from FloatLib's `FloatFormat` descriptors, including
`FloatFormat.binary32` for the `FP32` bridge constants. Use FloatLib's namespaces directly for
formats, rounding, scalar arithmetic, and analysis.

Calculation proofs use `FloatLib.Floats.Formats.Flocq.Calculation.Round`, `.Operations`,
`.Arithmetic`, and `.Bracket`.
Encoded-format consumers import `FloatLib.Floats.Formats.BinaryInterchange` and
`FloatLib.Floats.Formats.IEEE754.Native` directly. Consumer proofs use
`ExecFloat.Binary.toModel` for the exact
configured-to-model conversion.

## Rounded reals and encoded values

`FP32` abbreviates `NF binaryRadix (fltExp (-149) 24) nearestEven`. It rounds each real operation
to a grid with 24 significant bits and gradual underflow down to the binary32 subnormal scale.
**Its exponent has no upper bound.** NaNs, infinities, and signed zeros are absent. Consequently,
`FP32.ieeeMaxFinite` is a guard for transferring an IEEE result, not a largest element of `FP32`.

For executable encoded arithmetic, use `ExecFloat.Binary 8 23` for binary32. The two parameters
count exponent and stored fraction bits. FloatLib owns arithmetic, special-value behavior,
and word/model conversions. Its `ofBits32` and `toBits32` preserve every encoded bit. Native
`Float32` conversions have the separate native NaN boundary documented in the source.

FloatLib's `IEEE754.Native` modules supply exact interchange-word round trips and
configured/native conversion equations. `Native.AddSub` proves equality for finite inputs,
including both signed zeros and results that overflow to infinity. `Native.Sqrt` covers every
input after NaN canonicalization. These are logical-model theorems; they do not verify a
compiler's floating-point instructions or the host's floating-point environment.

Use `ExecFloat.Binary` directly for other precisions. Its configured arithmetic, rational
conversion, rounding modes, and status APIs remain in FloatLib. A wider carrier does not make a
conversion through native `Float` exact: consumers that need more than binary64 precision must
keep values in the configured carrier and use its exact conversion interfaces.

`NN.Floats` imports `FloatLib.Floats.Formats.BinaryInterchange.Configured.Transcendentals`
directly. Use `ExecFloat.Binary.exp`, `log`, and the other elementary functions from FloatLib.
Those are deterministic approximations; the import provides no universal real-error or
correct-rounding theorem. Import FloatLib's configured rounding modules for proved square root
without the transcendental approximations. The standalone umbrella
also imports `NN.Core.Numeric` for native scalar instances, including direct binary32 natural casts.

## Proofs retained at the TorchLean boundary

`IEEEExec/Bridge/Finite.lean` transfers FloatLib's arithmetic refinement to configured binary32.
Its add and multiply theorems have the original result-finiteness premise and conclude that the
encoded real value equals one `fp32Round` of the exact operation. Subnormal cases do not acquire
a global relative-error assumption.

Native proofs import FloatLib's native bridge directly. The CUDA contract reads native bits through
`ExecFloat.Binary.ofFloat32`; the tensor error bounds still use `Bridge/Finite.lean`.

For native addition, the finite-input premise is part of the theorem. Square root compares
canonicalized results because Lean's model has one NaN encoding while the configured carrier
can retain payloads:

```lean
import FloatLib.Floats.Formats.IEEE754.Native.AddSub
import FloatLib.Floats.Formats.IEEE754.Native.Sqrt

open FloatLib.Floats.ExecFloat.Binary
open FloatLib.Floats.Formats.BinaryInterchange

example (x : Float32) :
    toFloat32 (ofFloat32 x) = x :=
  toFloat32_ofFloat32 x

example (x y : Float32) (hx : x.isFinite = true) (hy : y.isFinite = true) :
    ofFloat32 (x + y) = ofFloat32 x + ofFloat32 y :=
  ofFloat32_add_of_isFinite x y hx hy

example (x : Float32) :
    toModel (ofFloat32 x.sqrt) =
      Model.canonicalizeModel (Model.sqrt (toModel (ofFloat32 x))) :=
  toModel_ofFloat32_sqrt x
```

These examples also appear in `NN/Examples/BugZoo/FloatBoundary.lean`. Configured arithmetic
has total software-model refinements. Native CPU instructions and LibTorch have separate execution
contracts recorded in `docs/TRUST_BOUNDARIES.md`.

`NN.Proofs.RuntimeApprox.IEEE32.Expressions` defines the scalar expression AST and proves its
finite-evaluation refinement. Its operation adapters live in
`NN.Proofs.RuntimeApprox.IEEE32.Arithmetic`; `IEEEExec/Bridge/Finite.lean` specializes FloatLib's
generic arithmetic refinements to the binary32 model used by TorchLean.

`NN.Proofs.RuntimeApprox.Reductions.Tree` keeps the array-facing reduction API and uses FloatLib's
generic reduction-tree error bound. The binary32 execution refinement remains in
`NN.Proofs.RuntimeApprox.Reductions.IEEE32`. These bounds account for rounding at each chosen
tree node; FloatLib's exact sum and dot-product accumulators round only once at the end.

The nested derivative quotient repair also remains TorchLean code, in
`NN.Core.Numeric.Quotient` and `NN.Runtime.Autograd.Model.Dual`. FloatLib supplies exact finite
rational decoding, scalar rounding, and configured operation status. TorchLean detects unsafe
intermediates and replays the complete nested coefficient tree when needed. A finite exact
derivative can still overflow its destination format; neither the scalar library nor replay can
promise a finite encoded result for an unrepresentable derivative.

## Intervals, quantization, and external enclosures

Use `FloatLib.Numerics.Interval (FloatLib.Floats.ExecFloat.Binary 8 23)` directly for binary32
endpoints, with arithmetic and conversion proofs from `ExecFloat.Binary.Interval`.
TorchLean's numerical graph certificates use these binary32 endpoints; selecting another scalar
format does not supply a graph certificate for that format. `Interval/FP32.lean` specializes
rounded-real enclosures. `NN.Spec.Quantization` lifts FloatLib's `RealAffineQuantizer` to shaped
tensors; integer code ranges specify int8, uint8, int4, or custom quantizers independently of
tensor layout.

`NN.Floats` imports the Arb interface, but importing it does not run an oracle. Actual requests
cross the Python/Arb/FLINT process boundary. `Interval/IEEEExec32ArbTrans.lean` is a separate
adapter for those external enclosures; parsing an oracle result is not a Lean proof of it.

The worked tensor example remains in `NN/Examples/DeepDives/Floats/EffectiveRounding.lean`.
It connects shaped operations to the rounded-real and encoded models used by their proofs.

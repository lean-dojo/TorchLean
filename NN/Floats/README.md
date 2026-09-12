# Floating-Point Arithmetic

`NN.Floats` is TorchLean's numerical library for floating-point formats, rounding, executable
binary32 arithmetic, interval enclosures, and error analysis. It can be imported without the tensor,
model, autograd, CUDA, or certificate-checking code:

```lean
import NN.Floats
open TorchLean.Floats
```

Smaller imports are available when only one part of the library is needed:

```lean
import NN.Floats.NeuralFloat
import NN.Floats.FP32
import NN.Floats.IEEEExec
import NN.Floats.Interval
```

Tensor quantization is defined separately in `NN.Spec.Quantization`. Proofs comparing rounded
execution with ideal tensor operations live in `NN.Proofs.RuntimeApprox.FP32`. The optional Arb
adapter under `NN.Floats.Arb` is not imported by `NN.Floats`.

## Numerical Models

The library has several representations because rounded-real error analysis and special-value-aware
binary32 execution answer different mathematical questions.

### `NeuralFloat`

`NeuralFloat` defines floating-point formats over `ℝ` with configurable radix, precision, exponent
range, and rounding policy. It contains:

- representability and neighboring-value results;
- directed, nearest-even, round-to-odd, and fixed-grid rounding;
- ULP and absolute or relative error bounds;
- double-rounding theorems;
- Sterbenz subtraction with gradual underflow;
- generic addition, multiplication, division, and square-root results.

The main modules are `NeuralFloat/Format.lean`, `NeuralFloat/Rounding.lean`,
`NeuralFloat/Scalar.lean`, `NeuralFloat/Analysis.lean`, and `NeuralFloat/Error.lean`.
`NN.Floats.Quantization` uses the fixed-grid rounding operation for affine scalar quantization.

This development is written in Lean. Flocq influenced its organization and several results, but the
directory is not a translation of every Flocq module.

### `FP32`

`FP32` specializes rounded-real arithmetic to the finite binary32 grid. An operation is modeled as
an exact real operation followed by binary32 rounding. NaNs and infinities are absent, which makes
this model convenient for numerical error proofs.

- `FP32/Core.lean` defines the format and rounded operations.
- `FP32/Error.lean` proves error bounds.
- `FP32/Sterbenz.lean` proves exact subtraction for nearby representable operands.
- `Interval/FP32.lean` derives interval enclosures.

### `IEEE32Exec`

`IEEE32Exec` is an independent executable binary32 implementation over 32-bit encodings. Its Lean
definitions cover normal and subnormal values, signed zeros, infinities, NaNs, overflow, underflow,
nearest-even rounding, and the five IEEE exception indicators.

The core implementation is in `IEEEExec/Exec32.lean`. Special-value rules are in
`IEEEExec/Rules/SpecialRules.lean`; status-bearing operations are defined in
`IEEEExec/Exec32/Arithmetic.lean`; and order-sensitive reductions are in
`IEEEExec/Reductions.lean`.
Executable directed rounding is also available for addition, subtraction, multiplication,
division, fused multiply-add, and square root.

### Lean `Float32`

Lean versions before 4.33 exposed the bits of a `Float32` but kept its arithmetic opaque to the
kernel. Lean 4.33 introduced `Float32.Model`, so proofs can inspect the logical definitions of core
`Float32` operations. TorchLean removed its former assumption-based bridge and now proves that
`Float32.Model` and `IEEE32Exec` agree.

The bridge in `IEEEExec/Bridge/LeanFloat32.lean` covers:

- bit conversion and canonical representation;
- finite, infinite, and NaN classification;
- comparison, `<`, `≤`, and Boolean equality;
- addition, subtraction, multiplication, and division;
- negation, absolute value, and square root.

The arithmetic proofs include normal and subnormal operands, signed zeros, infinities, NaNs,
underflow, overflow, and nearest-even ties. Lean uses one canonical NaN encoding, while
`IEEE32Exec` preserves NaN sign and payload bits. Arithmetic results are therefore compared after
the explicit `canonicalize` operation.

```lean
import NN.Floats.IEEEExec.Bridge.LeanFloat32

open TorchLean.Floats.IEEE754
open TorchLean.Floats.IEEE754.Float32Bridge

example (x : Float32) :
    Float32.isFinite x = IEEE32Exec.isFinite (toIEEE32Exec x) := by
  exact float32_isFinite_eq_ieee32 x

example (x y : Float32) :
    toIEEE32Exec (x + y) =
      canonicalize (IEEE32Exec.add (toIEEE32Exec x) (toIEEE32Exec y)) := by
  exact toIEEE32Exec_add x y

example (x : Float32) :
    toIEEE32Exec (Float32.sqrt x) =
      canonicalize (IEEE32Exec.sqrt (toIEEE32Exec x)) := by
  exact toIEEE32Exec_sqrt x
```

Lean defines the language model in
[`Init.Data.Float.Model.Float32`](https://github.com/leanprover/lean4/blob/v4.33.0/src/lean/Init/Data/Float/Model/Float32.lean).

## Rounding Calculations

`Calc` exposes the integer calculation behind a rounded-real result. Starting from a bracket around
an exact value, it determines the adjacent representable numbers, applies a rounding policy, and
returns a mantissa and exponent. Theorems in `FP32/Core.lean` identify the real value of the result
with `fp32Round`.

This is useful when a proof needs both the mathematical rounding theorem and the mantissa/exponent
that an implementation should produce. Calculations from arbitrary real numbers are noncomputable;
executable programs use `IEEE32Exec` or a named runtime provider.

## Bridges

The bridge modules connect the rounded-real and executable representations:

- `IEEEExec/Bridge/FP32.lean` proves finite, no-overflow refinement results of the form
  `toReal (opExec ...) = fp32Round (opReal ...)`.
- `IEEEExec/Bridge/FP32Total.lean` combines finite refinement with NaN and infinity rules using
  `toReal?`.
- `IEEEExec/Bridge/Expressions.lean` proves refinement for a scalar expression language.
- `IEEEExec/Bridge/ERealTotal.lean` gives a total `EReal` semantics that distinguishes positive
  and negative infinity.
- `IEEEExec/Bridge/LeanFloat32.lean` proves agreement with Lean's logical `Float32` operations.

These theorems concern Lean definitions. Compiled CPU instructions, CUDA kernels, cuBLAS, and
LibTorch are runtime providers with contracts recorded in `docs/TRUST_BOUNDARIES.md`.

## Intervals And Quantization

`Interval` contains outward rounders and endpoint enclosures over both real-valued and executable
representations. `IEEEExec32Interval` evaluates endpoint arithmetic with directed binary32
operations.

`neuralRoundAtScale` provides fixed-grid rounding with a positive grid step. It is used by affine
quantization and fixed-point proofs. The round-to-odd theorem in
`NeuralFloat/Rounding/Odd.lean` shows when a fine binary intermediate prevents nearest-even double
rounding on a coarser grid.

The tensor adapter in `NN.Spec.Quantization` is rank-polymorphic. Integer code ranges determine
int8, uint8, int4, or custom quantizers; tensor layout is not part of the scalar arithmetic.

## Examples

`NN/Examples/DeepDives/Floats/EffectiveRounding.lean` follows a shaped tensor addition through the
rounded-real and executable binary32 models. `NN/Examples/BugZoo/FloatBoundary.lean` shows how a
real-valued theorem can fail to describe a rounded computation until the required bridge has been
proved.

## References

- IEEE 754-2019, [IEEE Standard for Floating-Point Arithmetic](https://doi.org/10.1109/IEEESTD.2019.8766229).
- David Goldberg, [What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://doi.org/10.1145/103162.103163), 1991.
- Nicholas J. Higham, *Accuracy and Stability of Numerical Algorithms*, second edition, 2002.
- Sylvie Boldo and Guillaume Melquiond, [Flocq: A Unified Library for Proving Floating-Point Algorithms in Coq](https://doi.org/10.1109/ARITH.2011.40), 2011.

# `NN/Floats/IEEEExec`: Executable IEEE-754 Binary32 Semantics

This directory contains TorchLean's Lean-defined executable model of IEEE-754 binary32. It also
holds bridge theorems that connect bit-level execution to rounded-real models over `ℝ`.

## Directory Layout

- `Exec32.lean` and `Exec32/`: the executable kernel (`IEEE32Exec`), bit layout, arithmetic,
  comparison, directed rounding, dyadic conversion, exception-status outcomes, instances, and
  deterministic transcendental wrappers.
- `Encoding/`: interpretation of bit patterns as exact dyadics and reals, together with sign-bit
  facts.
- `Rounding/`: integer and rational lemmas used to prove the executable rounding algorithms.
- `Semantics/`: real and `EReal` interpretations, error bounds, and operation sandwiches.
- `Rules/`: proved rules for special values and the deterministic transcendental wrappers.
- `Reductions.lean`: reduction semantics for sums/dot products.
- `Bridge/`: refinement from executable bits to rounded-real, extended-real, expression, and Lean
  `Float32.Model` semantics.

## Directed Rounding And Intervals

The directed rounding kernels provide lower and upper binary32 endpoints with soundness proofs:

- `DirectedRoundingSoundness/`: soundness of directed dyadic and rational rounding and the
  endpoint operations for addition, multiplication, fused multiply-add, division, and square
  root. The statements use `EReal`, so endpoint overflow to $\pm\infty$ remains a valid enclosure.
- `Semantics/MinMaxERealSoundness.lean`: order lemmas used by endpoint min/max rules.

The interval API built from these results lives in `NN/Floats/Interval/`.

## Rounded-Real Bridges

The bridge files connect executable bit patterns with rounded-real and language-level semantics:

- `Bridge/FP32.lean` and `Bridge/FP32/`: per-operation refinement lemmas on the finite branch,
  including dyadic/rational rounding infrastructure, exact subtraction under Sterbenz's
  hypotheses, and executable ULP exponents. The ULP query succeeds exactly for finite bit patterns
  and returns `none` exactly for NaNs and infinities.
- `Bridge/Expressions.lean`: expression-level refinement that composes operation lemmas once.
- `Bridge/FP32Total.lean`: packages finite refinement and proved special-value rules using
  `toReal?`.
- `Bridge/ERealTotal.lean`: an `EReal` interpretation that distinguishes $+\infty$ and $-\infty$ while
  representing NaN as `none`.
- `Bridge/LeanFloat32.lean`: agreement with Lean's logical `Float32.Model` for classification,
  comparison, addition, subtraction, multiplication, division, square root, negation, and absolute
  value. Arithmetic results are compared after NaN canonicalization because `IEEE32Exec` retains
  payload and sign bits that `Float32.Model` discards.

## Using The Executor

Use `IEEE32Exec` for executable binary32 calculations inside Lean and `FP32` for rounded-real error
bounds. Native `Float32`, CUDA, and external libraries have separate provider contracts in
`docs/TRUST_BOUNDARIES.md`.

The value-only operations (`add`, `mul`, `div`, `fma`, and `sqrt`) are accompanied by
status-bearing operations (`addWithStatus`, `mulWithStatus`, `divWithStatus`, `fmaWithStatus`, and
`sqrtWithStatus`). An `IEEEOutcome` contains the computed bits and an `IEEEStatus` containing the
invalid, divide-by-zero, overflow, underflow, and inexact indicators. Tininess is detected after
rounding, and underflow implies inexactness by theorem. `Rules/SpecialRules.lean` proves the main
special-value status cases.

An executable error proof commonly follows this sequence:

```text
IEEE32Exec operation
  -> finite/no-overflow bridge
  -> FP32 rounded-real statement
  -> real-valued error envelope or interval claim
```

## Transcendental Functions

IEEE 754 does not specify bit-level results for functions such as `expf` and `logf`, and platform
libm implementations can differ. The library separates:

- core arithmetic (add/mul/div/sqrt, specials) as the proved executable kernel, and
- transcendentals as deterministic but not IEEE specified unless you use a separate rigorous
  backend (e.g. the Arb oracle under `NN/Floats/Arb/` and the interval glue in `NN/Floats/Interval/`).

`Rules/TranscendentalRules.lean` defines `UnaryApproximationContract` for connecting a finite
executable implementation to a real function on a stated domain with a stated error tolerance.
This contract does not assert that the current deterministic wrappers satisfy a particular
accuracy bound; such a theorem needs evidence for the chosen implementation.

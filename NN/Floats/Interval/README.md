# Interval arithmetic (`NN/Floats/Interval`)

This folder collects interval and enclosure utilities that sit between:

- proof-oriented rounded-real models (`NN/Floats/NeuralFloat`, `NN/Floats/FP32`), and
- executable bit-level kernels (`NN/Floats/IEEEExec`), and
- external validated numerics backends (`NN/Floats/Arb`).

The goal is to make bound propagation (IBP/CROWN style) and numerical soundness envelopes
reusable across the codebase, while keeping the trust boundary explicit.

## Files

- `Rounders.lean`: a minimal `Rounder` interface (`down`/`up`) plus a Flocq-style implementation
  using floor/ceil rounding on `ℝ` (directed rounding on a discrete grid).
- `Quantized.lean`: a proof oriented quantized interval arithmetic layer over `ℝ` and `EReal`,
  with outward rounding at endpoints and overflow awareness for division by zero (returns
  `[-∞,+∞]`).
- `FP32.lean`: interval-style enclosure corollaries for the `FP32` model.
- `IEEEExec32.lean`: executable endpoint intervals for `IEEE32Exec`.
- `RealBounds.lean`: a shared real analysis lemma (`mul_bounds_Icc`) giving the classical
  four corner enclosure for multiplication on real intervals.
- `IEEEExec32NoNaN.lean`: shared non NaN helper lemmas for `IEEE32Exec` (`minimum`/`maximum`,
  dyadic rounding), used by the interval soundness proofs.
- `IEEEExec32AddSoundness.lean`: enclosure theorems for executable interval `add/sub/neg` in the
  finite regime.
- `IEEEExec32MinMaxSoundness.lean`: small lemmas computing `toReal` semantics of the `min4/max4`
  helpers (finite regime).
- `IEEEExec32MulSoundness.lean`: enclosure theorem for executable interval multiplication
  (`Interval32.mul`) via the 4-corner rule.
- `IEEEExec32DivSoundness.lean`: enclosure theorem for executable interval division / reciprocal
  (`Interval32.div`/`inv`), with a `whole = [-∞,+∞]` fallback when the denominator interval contains
  `0`.
- `IEEEExec32Soundness.lean`: umbrella import re-exporting the main `IEEEExec32` interval enclosure
  theorems (and small helper lemmas) in one place.
- `IEEEExec32ArbTrans.lean`: Arb-backed interval endpoints for `tanh/exp/log/sqrt` on `Interval32`,
  returning outward-rounded float32 endpoints while keeping the transcendental soundness boundary
  explicit (oracle).
- Arb-backed comparison workflows live under `NN/Examples/Advanced/Floats/`.

## Running an example workflow

```bash
lake exe torchlean floats_arb_ieee_compare
```

## References

- IEEE 1788-2015: interval arithmetic standard.
- Moore, Kearfott, Cloud, *Introduction to Interval Analysis* (2009).
- Higham, *Accuracy and Stability of Numerical Algorithms* (2002) (directed rounding discussion).
- Flocq (Boldo–Melquiond, 2011): rounding-on-`ℝ` model used by `Rounders.lean`.

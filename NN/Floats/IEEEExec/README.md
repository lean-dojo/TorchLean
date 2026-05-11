# `NN/Floats/IEEEExec`: Executable IEEE-754 Float32 Semantics

This directory contains TorchLean's Lean defined executable model of IEEE-754 binary32. It also holds
bridge theorems that connect runtime execution to proof oriented rounding models over `ℝ`.

## What lives here

- `Exec32.lean`: the executable kernel (`IEEE32Exec`) and its operations.
- `ERealSemantics.lean`: small shared helpers for interpreting `IEEE32Exec` values in `EReal`
  (including a total `toEReal` used in endpoint/enclosure proofs).
- `SpecialRules.lean`: proved rewrite rules for NaN/Inf/±0 propagation, so proofs do not need to
  repeatedly unfold the executable definitions.
- `TranscendentalRules.lean`: deterministic, non IEEE specified transcendental functions
  (`exp`, `log`, `tanh`, …) and their special-value rules.
- `RoundShiftRightEven.lean`: small order facts about round to nearest even on shifted mantissas
  (used in later rounding / interval arguments).

## Interval / Directed Rounding Semantics

If your goal is interval arithmetic over float32, this folder provides the directed rounding
kernels and their soundness proofs:

- `DirectedRoundingSoundness.lean`: soundness of `roundDyadicDown/Up` and endpoint ops
  (`addDown/addUp/mulDown/mulUp`) in `EReal` (so overflow to `±∞` is handled cleanly).
- `DivDirectedRoundingSoundness.lean`: analogous soundness for rational rounding and
  `divDown/divUp` (finite / nonzero-divisor regime).
- `MinMaxERealSoundness.lean`: basic order lemmas used by endpoint min/max rules.

The interval *API layer* that uses these results lives in `NN/Floats/Interval/`.

## Bridge To Proof Oriented Float Models

TorchLean keeps two float32 views:

- Executable (`IEEE32Exec`): what we can run inside Lean, with bit level semantics.
- Proof oriented (`FP32`): round on real semantics used for numerical error envelopes.

Bridge files connect these on the finite, no overflow path:

- `BridgeFP32.lean`: per-operation refinement lemmas (finite branch).
- `BridgeFP32Expr.lean`: expression-level refinement (compose op lemmas once).
- `BridgeFP32Total.lean`: packages finite refinement + proved special-value rules using `toReal?`.
- `BridgeERealTotal.lean`: a slightly richer `EReal` interpretation (`+∞` vs `-∞`, NaN as `none`).
- `BridgeInitFloat32.lean`: an assumption based interface relating Lean runtime `Float32` to
  `IEEE32Exec`. The runtime is opaque to the kernel, so this cannot be proved internally.

## Trust boundary (important)

IEEE-754 does not specify bit level results for transcendentals (`expf`, `logf`, ...), and platform
libm implementations can differ. TorchLean therefore treats:

- core arithmetic (add/mul/div/sqrt, specials) as the proved executable kernel, and
- transcendentals as deterministic but not IEEE specified unless you use a separate rigorous
  backend (e.g. the Arb oracle under `NN/Floats/Arb/` and the interval glue in `NN/Floats/Interval/`).

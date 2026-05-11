# Floats (`NN/Floats`)

This directory contains TorchLean's floating point backends and the theory that connects them.

- `FP32/`: a proof oriented, finite float32 model based on rounding over `ℝ`.
- `NeuralFloat/`: generic rounding over `ℝ` (`NeuralRadix`, `NF`, rounding, ULPs, and error bounds).
- `IEEEExec/`: an executable IEEE-754 binary32 kernel (`IEEE32Exec`) plus bridge theorems to `FP32`.
- `Interval/`: interval and enclosure utilities, including quantized intervals over `ℝ` and executable endpoint intervals.
- `Arb/`: an external Arb/FLINT oracle backend (python-flint) for ball and interval enclosures.

If you want the whole float chapter, import `NN.Entrypoint.Floats`. If you only want a single
float32 name to depend on, import `NN.Floats.Float32`.

Executable examples that exercise this infrastructure live under `NN/Examples/`.

## The Three Float32 Views

TorchLean uses three complementary notions of float32. They have different
strengths, and the bridge files let us move between them without blurring the
trust boundary.

### 1. `IEEE32Exec` (Executable IEEE-754 Binary32)

Folder: `NN/Floats/IEEEExec/`

This is the executable backend. Values are 32-bit encodings, and operations are implemented as pure
Lean code over those bits.

Why we keep it:
- we can execute models inside Lean without relying on opaque runtime float calls,
- we can state and prove theorems about the kernel itself (NaN/Inf propagation, signed zeros, etc.),
- and it provides a stable target for connecting to the mathematical models below.

Where to look:
- `NN/Floats/IEEEExec/Exec32.lean`: the core executable kernel (`IEEE32Exec`),
- `NN/Floats/IEEEExec/SpecialRules.lean`: NaN/Inf propagation rules,
- `NN/Floats/IEEEExec/Reductions.lean`: reduction semantics (sums/dot products) that match deployment realities.
  (For executable endpoint-interval arithmetic, see `NN/Floats/Interval/IEEEExec32.lean`.)

### 2. `FP32` (Finite Real Rounded Float32 Model)

Folder: `NN/Floats/FP32/`

This is the mathematical view: float32 computation is modeled as real arithmetic followed by a
float32 rounding operator. It is finite only, with no NaNs or infinities, which keeps many error
analysis statements manageable.

Why we keep it:
- most error-bound theorems are naturally stated over reals + rounding,
- it composes cleanly with `NeuralFloat`/`NF` error bounds,
- it gives a clear semantic target when we want to describe what a float32 computation means.

Where to look:
- `NN/Floats/FP32/Core.lean`: the main definitions,
- `NN/Floats/FP32/Error.lean`: error bounds,
- `NN/Floats/Interval/FP32.lean`: interval-style enclosure corollaries.

### 3. `NeuralFloat` / `NF` (Generic Format And Error Analysis)

Folder: `NN/Floats/NeuralFloat/`

This layer abstracts over radix, precision, and exponent range. It is where we develop reusable facts
about rounding, ULPs, and relative or absolute error bounds. `FP32` is the
binary32-specialized instance of this generic layer.

Why we keep it:
- we want one set of theorems that applies to multiple precisions and training phases,
- it lets us reuse the same reasoning patterns for float32 and other formats.

Where to look:
- `NN/Floats/NeuralFloat/Core.lean`, `NN/Floats/NeuralFloat/Formats.lean`, `NN/Floats/NeuralFloat/Rounding.lean`,
- `NN/Floats/NeuralFloat/NF.lean` and `NN/Floats/NeuralFloat/ErrorBounds.lean`.

## How The Bridge Layers Fit Together

`IEEE32Exec` is executable; `FP32` and `NF` are aimed at proofs. The bridge files connect them:

- `NN/Floats/IEEEExec/BridgeFP32.lean`: core refinement theorems on the **finite/no-overflow** path:
  `toReal (op_exec …) = fp32Round (op_real …)`.
- `NN/Floats/IEEEExec/BridgeFP32Total.lean`: total wrappers that combine NaN/Inf propagation rules
  with the finite refinement theorems, phrased using `toReal?`.
- `NN/Floats/IEEEExec/BridgeFP32Expr.lean`: a compact scalar AST + a whole-expression refinement theorem.
- `NN/Floats/IEEEExec/BridgeERealTotal.lean`: an `EReal`-valued semantics that distinguishes `+∞` and `-∞`.
- `NN/Floats/IEEEExec/BridgeInitFloat32.lean`: an assumption based bridge from Lean's runtime
  `Init.Float32` to `IEEE32Exec`, at the bit level.

Background references that informed the design:
- IEEE 754-2019: https://doi.org/10.1109/IEEESTD.2019.8766229
- Goldberg (1991): https://doi.org/10.1145/103162.103163
- Higham (2002): *Accuracy and Stability of Numerical Algorithms* (2nd ed.), ISBN 0-89871-521-0
- Flocq (Boldo–Melquiond, 2011): https://doi.org/10.1109/ARITH.2011.40

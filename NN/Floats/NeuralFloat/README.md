# NeuralFloat (`NN/Floats/NeuralFloat`)

This folder provides the generic rounded arithmetic over `ℝ` that much of TorchLean builds on. It
holds the shared definitions for precision, rounding, and error without committing to a particular
executable kernel.

- `Core.lean`: core datatypes (radix plus Flocq style rounding scaffolding) and the `NeuralFloat` record.
- `Metadata.lean`: TorchLean metadata (training phase, named precisions, mixed precision presets).
- `Formats.lean`: Flocq style exponent functions and format predicates (FIX/FLX/FLT).
- `Rounding.lean`: rounding modes and `neural_round` validation classes.
- `NF.lean`: `NF` ("neural float") as a rounded scalar type with `Context` instances.
- `NNOps.lean`: a small library of rounded scalar functions used by NN specs, built on `neural_round`.
- `Conversion.lean`: conversion helpers with lightweight error metadata.
- `ErrorBounds.lean`: small reusable error bound lemmas for composing proofs.

For executable, bit-level IEEE-754 float32 behavior (NaN/Inf/signed zero), see `NN/Floats/IEEEExec/`.

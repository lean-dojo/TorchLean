# `NN.Floats.FP32`

`FP32` is TorchLean's proof-oriented rounded-real model of float32 arithmetic. It is used for
theorem statements and compositional error arguments when the proof should look like finite
precision arithmetic but does not need NaNs, infinities, signed zero, or bit-level payloads.

The typical proof has the form:

```text
real spec value
  -> rounded FP32 operation
  -> bounded error or interval enclosure
```

Bit-level binary32 behavior, including special values, is defined by FloatLib's configured
`ExecFloat.Binary 8 23` format.

## Files

- `Core.lean`: canonical binary32 configuration (`fexp32`, `rnd32`) and the `FP32` type alias.
- `Notation.lean`: aliases over `ℝ` for the model, including `round32`, `ulp32`, and `eps32`.
- `Error.lean`: per-operation absolute error bounds.
- `NN/Proofs/RuntimeApprox/FP32.lean`: error bounds restated through the generic tolerance relation
  `≈[t]`.
- `Sterbenz.lean`: exact subtraction for nearby representable binary32 values.

For the half-ulp enclosure of a rounded value, use FloatLib's `Model.roundAt_mem_Icc` at
`FloatFormat.binary32`.

## Relationship To Runtime

Bridges and provider contracts outside this directory connect `FP32` results with Lean `Float`,
C/CUDA `float`, and external kernels.

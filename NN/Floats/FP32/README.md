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

## Proof APIs

`NN/Floats/FP32.lean` defines `FP32` by specializing `NF` with
`Model.fexpOf FloatFormat.binary32` and `nearestEven`.

FloatLib supplies the descriptor-generic rounding theorems:

- `Model.abs_roundAt_sub_le` and `Model.roundAt_mem_Icc` in
  `FloatLib/Floats/Formats/BinaryInterchange/Analysis/Error.lean` give the half-ULP error bound
  and enclosure. Instantiate them with `FloatFormat.binary32`.
- `Model.relativeError_roundAt_le_of_normal` in the same module gives the relative error bound
  under its normal-range hypothesis.
- `Model.roundAt_sub_eq_of_sterbenz` in
  `FloatLib/Floats/Formats/BinaryInterchange/Analysis/Sterbenz.lean` gives exact subtraction
  for positive representable values within a factor of two.

`NN/Proofs/RuntimeApprox/FP32.lean` expresses the per-operation bounds through the generic
tolerance relation `≈[t]`.

## Relationship To Runtime

Applying these bounds to Lean `Float`, C/CUDA `float`, or external kernels requires a separate
agreement theorem or provider contract. See [trust boundaries](../../../docs/TRUST_BOUNDARIES.md).

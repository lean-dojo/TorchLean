# `NN.Floats.FP32`

`FP32` is TorchLean's proof oriented model of float32 rounding. We use it for theorem
statements and compositional error arguments.

- The scalar type is `TorchLean.Floats.FP32` (an `NF ...` rounded-real value).
- It is finite only, with no NaNs or infinities. Use it when you want clean error bounds and stability proofs.
- For an executable IEEE-754 binary32 backend, use `TorchLean.Floats.IEEE754.IEEE32Exec`
  (re-exported as `TorchLean.Floats.IEEE32Exec`) via `NN/Floats/Float32.lean`.

Files:

- `Core.lean`: the canonical binary32 configuration (`fexp32`, `rnd32`) and the `FP32` type alias.
- `Notation.lean`: aliases over `ℝ` for the model (`round32`, `ulp32`, `eps32`).
- `Error.lean`: per-op absolute error bounds (ULP/2 style statements).
- `RuntimeApprox.lean`: restates the error bounds using the generic tolerance relation `≈[t]`.
- Interval enclosures live in `NN/Floats/Interval/FP32.lean` (imported by `NN.Floats.FP32`).

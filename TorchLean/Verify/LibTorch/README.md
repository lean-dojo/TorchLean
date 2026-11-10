# `TorchLean/Verify/LibTorch`

Contracts that connect native LibTorch results to FloatLib's arithmetic, in any format of
FloatLib's configured binary family, with the rounding theorems in its IEEE formats (binary32 and
binary64 among them). A native result that agrees with FloatLib's, up to the encoding of NaNs,
carries FloatLib's rounding theorems. The agreement itself is a hypothesis that tests check for a
particular build, not a theorem about the native code.

- `Contract.lean`: `Agreement`, the agreement of a native function with FloatLib's
  addition, subtraction, multiplication, division, square root or fused multiply-add. A finite
  agreeing result is FloatLib's result and, on the operation's domain, lies within half an ulp of
  the exact result. `scripts/checks/cuda.sh parity` compares binary32 addition, multiplication,
  division, square root and fused multiply-add bit for bit, except NaN encodings.
- `TreeSum.lean`: the schedule `treeSum` of the deterministic CUDA sum, its evaluation
  `reduceSumTree`, and the unit-roundoff enclosure of a finite native sum that agrees with it.
  `TorchLeanTest/CUDA/ReduceTree.lean` compares native binary32 sums with it on fixtures.

The buffers and FFI declarations these contracts describe are in
[`TorchLean/CUDA/LibTorch`](../../CUDA/LibTorch/README.md).

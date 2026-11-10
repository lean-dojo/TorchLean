# `TorchLean/CUDA/LibTorch`

The Lean side of the LibTorch CUDA backend: the opaque buffer type, the foreign-function
declarations of the ATen operations TorchLean calls, and the process-wide runtime controls.
TorchLean keeps its own tape and selected local VJPs, so native calls never record a LibTorch
autograd graph. `scripts/lake.sh -Kcuda=true build` links `csrc/libtorch/torchlean.cpp`,
`csrc/libtorch/conv_pool.cpp` and `csrc/libtorch/tree.cpp`. The default build links
`csrc/libtorch/unavailable.c`, which reports `.notLinked`, reads settings as off and memory
statistics as zero, and fails every buffer operation. Lean does not prove the compiled native
code correct, and [`docs/TRUST_BOUNDARIES.md`](../../../docs/TRUST_BOUNDARIES.md) lists what is
trusted.

- `Trusted.lean`: the opaque `Buffer` type and the map from native source files to Lean modules.
- `Buffer.lean`: `RuntimeStatus.query`, `Memory.read`, and dtype-preserving allocation, copy,
  random, elementwise, loss and Adam operations.
- `Controls.lean`: device selection, TF32, determinism, SDP, cuDNN and memory-fraction settings,
  reads of the device, determinism, cuDNN benchmarking, SDP and memory-fraction settings,
  `synchronize`, `emptyCache` and `version`.
- `Kernels.lean`: ATen softmax, cross-entropy, grouped convolution, reductions, indexing,
  matmul/BMM, normalization, FFT, scans and broadcast.
- `ConvPool.lean`: float32 and float64 transposed convolution and pooling.
- `Convert.lean`: row-major conversion of Boolean tensor masks to `FloatArray`.
- `Broadcast.lean`: the `axisMap` of a broadcast (`Shape.CanBroadcastTo`).

The CUDA tape over these buffers is in `TorchLean/NN/LibTorch`, the binary32
contracts for native bits are in [`TorchLean/Verify/LibTorch`](../../Verify/LibTorch/README.md), and
the native sources are described in [`csrc/libtorch`](../../../csrc/libtorch/README.md).

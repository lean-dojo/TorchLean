# `TorchLean/CUDA`

This layer is TorchLean's GPU boundary: the `@[extern]` declarations of the LibTorch bridge, the
compiler that turns supported Lean scalar functions into CUDA source, and the folds that run
recursive functions over trees level by level. It depends on the tensor layer, FloatLib and Mathlib.
The C++ side of the bridge lives in [`csrc/libtorch`](../../csrc/libtorch/README.md).

- [`Kernel/`](Kernel/README.md): the certified custom-computation compiler and its NVRTC runtime.
- [`LibTorch/`](LibTorch/README.md): CUDA buffers, runtime controls and the LibTorch kernel FFI.
- [`Tree/`](Tree/README.md): recursive functions over inductive types as folds, hylomorphisms and
  unfolds over leveled arenas, on the GPU or the host.

`scripts/lake.sh -Kcuda=true build` links the LibTorch CUDA backend, which
`scripts/libtorch_build.py` builds. Without `-Kcuda=true` the build links the stub compiled from
`csrc/libtorch/unavailable.c`, which reports the backend as not linked and rejects GPU
computation. Status and memory queries work in both builds.

Import the module a file uses, such as `TorchLean.CUDA.Kernel.Elab` for `Program.of` and
`TorchLean.CUDA.Tree.Elab` for `Fold.of`.

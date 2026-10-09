/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Kernel.Function
public import NN.Kernel.Graph
public import NN.Tensor.Internal.Elab.TensorLiteral -- shake: keep

/-!
# Custom tensor computations

Write an ordinary scalar function and use `f.run input (device := cpu)` or `gpu` to apply it to
a tensor of the same shape. Binary functions use `f.zip left right` with equally shaped tensors.
CPU evaluates Lean code; GPU generates CUDA and executes it through NVRTC. Captured scalars and
unsigned loop bounds may be supplied at runtime. Indexed calculations use `Program.of` with
explicit input readers and output shapes.
Named scalar recurrences are recognized from their zero/successor equations and lowered to loops,
with fixed parameters and a final natural-number argument. Their depths may vary by output entry.
Programs retain their reference calculation and a checked semantic correspondence proof.

GPU execution supports native binary32/binary64 and configured binary arithmetic through
an integer-limb software backend, including binary16, bfloat16 and wider-than-binary64 formats.
Decimal and posit formats remain CPU-only. Indexed reads
are checked, branches are lazy and bounded folds preserve their accumulation order. Unsupported
source constructs are rejected. NVIDIA compilation, GPU execution and LibTorch remain external
trust boundaries. Ordinary whole-tensor functions use the same `f.run` call. Set `grad := true`
to record supported arithmetic, reductions and matrix products through the existing autograd
tape and derivative rules; the returned value has `backward` and `close`.
Arbitrary indexed custom bodies do not acquire a backward rule from source equivalence alone.
-/

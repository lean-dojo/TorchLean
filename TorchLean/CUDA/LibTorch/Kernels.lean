/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

LibTorch FFI: ATen operations over dtype-carrying `LibTorch.Buffer` values.

Notes:
- `LibTorch.Buffer` is an opaque, contiguous buffer in CUDA device memory.
- Row and column counts, axes and other sizes cross the FFI as `UInt32`, and shapes, per-axis
  parameters and row indices as `Array Nat`.
- Build with `scripts/lake.sh -Kcuda=true build` and a LibTorch SDK. The default build links
  placeholders for these symbols that fail when called.
- TorchLean owns the differentiation tape. The LibTorch bridge disables graph recording and calls
  the ATen forward and backward operators.
-/

module

public import TorchLean.CUDA.LibTorch.Trusted

/-!
# CUDA Buffer Kernels FFI

Foreign-function declarations for ATen operations on TorchLean's dtype-carrying
`LibTorch.Buffer`: cross-entropy and distillation losses, grouped convolution, row reductions,
masks and hard-masked softmax, indexing, matmul/BMM, normalization, Fourier transforms, scans,
and broadcast/view helpers. Lean composes attention from these primitives in
`TorchLean.NN.LibTorch.Attention`. The declarations here are the Lean side of the LibTorch CUDA
trust boundary documented in `docs/TRUST_BOUNDARIES.md`.
-/

@[expose] public section

namespace TorchLean
namespace LibTorch

namespace Buffer

/-- Weighted logits cross-entropy along `axis`, divided by the sample count when `mean` is true.
A weighted term whose unweighted log-probability overflows is computed from the original logits.
The foreign routine does not record autograd. -/
@[never_extract, extern "torchlean_cuda_buffer_cross_entropy"]
opaque crossEntropy (logits target : @& Buffer) (dims : @& Array Nat) (axis : UInt32)
    (mean : Bool := true) : Buffer

/-- Local logits and target-weight pullbacks, in that order, multiplied by a scalar seed. -/
@[never_extract, extern "torchlean_cuda_buffer_cross_entropy_bwd"]
opaque crossEntropyBackward (logits target seed : @& Buffer) (dims : @& Array Nat)
    (axis : UInt32) (mean : Bool := true) : Buffer × Buffer

/-- Fixed-teacher mean cross-entropy, or its student gradient when `gradient` is true.
ATen evaluates the shared logit-difference formula without recording an autograd graph. The
result is scalar for the loss and has the input shape for the gradient. -/
@[never_extract, extern "torchlean_cuda_buffer_distill"]
opaque distill (student teacher : @& Buffer) (dims : @& Array Nat) (axis : UInt32)
    (gradient : Bool := false) : Buffer

/-- Grouped convolution over a spatial suffix, with dilation and zero padding before and after
each axis.

Axes before the channel axis are folded into a batch. `kernelDims` is PyTorch's packed
`(outChannels, inChannels / groups, spatial...)` layout. LibTorch supports one to three spatial
dimensions. Input lengths and all geometry are checked again at the foreign boundary. No gradient
is recorded.
-/
@[never_extract, extern "torchlean_cuda_buffer_conv_io"]
opaque conv (input kernel bias : @& Buffer)
    (inputDims kernelDims stride paddingBefore paddingAfter dilation : @& Array Nat)
    (groups : UInt32) : IO Buffer

/-- The kernel, bias and input gradients of `conv` for the output gradient `gradient`, with the
same dimensions and geometry. The bias gradient sums `gradient` over every axis but the feature
axis. With an empty output every gradient is zero, where `Op.conv`'s rule multiplies the absent
cotangent entries, read as zero, by the weights (a NaN input gradient for an infinite weight). The
foreign routine does not record autograd. -/
@[never_extract, extern "torchlean_cuda_buffer_conv_bwd"]
opaque convBackward (input kernel gradient : @& Buffer)
    (inputDims kernelDims stride paddingBefore paddingAfter dilation : @& Array Nat)
    (groups : UInt32) : Buffer × Buffer × Buffer

/--
Sum across the columns of a 2D row-major buffer.

Input `b` has shape `(rows, cols)` and is stored as length `rows*cols`. The output has length
`rows` and holds the sum of each row.
-/
@[never_extract, extern "torchlean_cuda_buffer_reduce_sum_by_row"]
opaque reduceSumByRow (b : @& Buffer) (rows cols : UInt32) : Buffer

/-- Maximum of each row of a row-major `(rows, cols)` buffer over its non-NaN entries, so a row of
NaNs gives `-∞`. When `cols = 0`, all `rows` outputs are `+0`. -/
@[never_extract, extern "torchlean_cuda_buffer_reduce_max_by_row"]
opaque reduceMaxByRow (b : @& Buffer) (rows cols : UInt32) : Buffer

/-- One at the first maximal coordinate in each nonempty row and zero elsewhere.
Empty rows return an empty buffer. The mask retains the input dtype. -/
@[never_extract, extern "torchlean_cuda_buffer_argmax_mask_by_row"]
opaque argmaxMaskByRow (b : @& Buffer) (rows cols : UInt32) : Buffer

/--
One for the `k` entries of each row of a flat `(rows, cols)` buffer that fewer than `k` entries of
the row beat, and zero for the others (`TorchLean.MoE.kept`). Entry `j` beats entry `i` when it is
larger, or when neither is larger and `j < i`. The kernel compares and counts, so it is exact.
-/
@[never_extract, extern "torchlean_cuda_buffer_top_k_mask_by_row"]
opaque topKMaskByRow (b : @& Buffer) (rows cols k : UInt32) : Buffer

/--
Stable row-wise hard-masked softmax for flat `(rows, cols)` buffers.

The row maximum and denominator are computed only from entries whose mask value is nonzero. Blocked
entries are exactly zero in the output. A row with no allowed entries is defined to be all zeros.
-/
@[never_extract, extern "torchlean_cuda_buffer_hard_masked_softmax_by_row"]
opaque hardMaskedSoftmaxByRow (scores mask : @& Buffer) (rows cols : UInt32) : Buffer

/-- Concatenate two 1D buffers `a` (length `n`) and `b` (length `m`). -/
@[never_extract, extern "torchlean_cuda_buffer_concat1d"]
opaque concat (a b : @& Buffer) (n m : UInt32) : Buffer

/--
Slice a 1D buffer `b` (length `n`) starting at `start` for `len` elements.

Requires `start + len ≤ n`.
-/
@[never_extract, extern "torchlean_cuda_buffer_slice1d"]
opaque slice (b : @& Buffer) (n start len : UInt32) : Buffer

/--
Broadcast a column-vector (length `rows`) to a `(rows, cols)` matrix.

Output is row-major of length `rows*cols`, with `out[i, j] = vec[i]`.
-/
@[never_extract, extern "torchlean_cuda_buffer_broadcast_vec_to_cols"]
opaque broadcastVecToCols (vec : @& Buffer) (rows cols : UInt32) : Buffer

/--
Layer normalization over the columns of a row-major `(rows, cols)` buffer, in double precision
with `TorchLean.Normalization.normalizedRows`'s formula: each centered row is divided by its scale
(the maximum of its largest magnitude and `sqrt |epsilon|`) before it is squared, then by its
standard deviation.

Both dimensions must be positive, and `gamma` and `beta` each have length `cols`. The result is
`(output, normalized, invStd)`, where `normalized` has shape `(rows, cols)` and
`invStd = 1 / std / scale` has length `rows`. These two saved values are all that TorchLean's
layer-normalization VJP needs, and the native kernel creates no autograd graph.
-/
@[never_extract, extern "torchlean_cuda_buffer_layer_norm_fwd"]
opaque layerNorm
    (x gamma beta : @& Buffer) (rows cols : UInt32) (epsilon : Float) :
    Buffer × Buffer × Buffer

/--
TorchLean's layer-normalization VJP evaluated by LibTorch tensor operations.

Given the upstream derivative, the cached normalized values and inverse standard deviations, and
`gamma`, it returns `(dX, dGamma, dBeta)` with `TorchLean.LayerNorm.backward`'s formula: each row
mean divides its sum once by `cols`. Both dimensions must be positive. The tape records the parents
that receive these gradients, and this primitive only evaluates the formula.
-/
@[never_extract, extern "torchlean_cuda_buffer_layer_norm_bwd"]
opaque layerNormBackward
    (dOut normalized invStd gamma : @& Buffer) (rows cols : UInt32) : Buffer × Buffer × Buffer

/-- Batched product `A B` of logical shapes `(batch, m, n)` and `(batch, n, p)` over row-major
buffers. Flag `transposeA = 1` reads `A` stored as `(batch, n, m)` and `transposeB = 1` reads `B`
stored as `(batch, p, n)`, with no transposed copy, and the native boundary rejects other flags. -/
@[never_extract, extern "torchlean_cuda_buffer_bmm_with_transpose"]
opaque bmm (A B : @& Buffer) (batch m n p : UInt32) (transposeA transposeB : UInt32 := 0) : Buffer

/-- Scaled batched product and a range-loss flag for its original operands.

For a nonzero `Float64` scale of magnitude below one, the native path multiplies the left operand
by a power of two near the square root of the scale's magnitude and the right operand by the
remainder. No square root is rounded, and a power-of-two scale adds no rounding while both factors
and the scaled operands are representable in the buffer dtype. Product-range checks, tiny or zero
results and nonfinite intermediates set the flag, which requests replay from the original buffers.
The checks are conservative (an exactly zero contraction can set the flag too), and the flag
certifies no error bound. It stays false when the scale or an operand entry is nonfinite.
Computing it costs at most one device synchronization. The caller owns the returned buffer. -/
@[never_extract, extern "torchlean_cuda_buffer_scaled_bmm"]
opaque scaledBmm (A B : @& Buffer) (batch m n p transposeA transposeB : UInt32)
    (scale : Float) : Buffer × Bool

/--
Real-valued 1D FFT over row-major batches, returning a packed half-spectrum.

Input:
- `x`: length `batch*n`, interpreted as shape `(batch, n)`.

Output:
- length `batch*(n/2+1)*2`, interpreted as shape `(batch, n/2+1, 2)`, whose last axis stores
  `[real, imag]` for each nonredundant frequency bin.

The CUDA path calls LibTorch's real FFT without normalization, and `n` must be positive.
Differentiable wrappers should state their backward convention separately, because half-spectrum
packing has normalization and conjugate-symmetry edge cases.
-/
@[never_extract, extern "torchlean_cuda_buffer_rfft1d_packed"]
opaque rfftPacked (x : @& Buffer) (batch n : UInt32) : Buffer

/--
Inverse of `rfftPacked` for packed half-spectra.

Input:
- `spec`: length `batch*(n/2+1)*2`, interpreted as `(batch, n/2+1, 2)`.

Output:
- length `batch*n`, interpreted as `(batch, n)`.

The CUDA implementation calls LibTorch's inverse real FFT with `1/n` normalization, matching
the CPU reference. It reads the imaginary parts of bin `0` and, for even `n`, of bin `n/2` as
zero, so a NaN or an infinity there does not reach the output, where the CPU reference multiplies
it by zero and gives NaN. `n` must be positive.
-/
@[never_extract, extern "torchlean_cuda_buffer_irfft1d_packed"]
opaque irfftPacked (spec : @& Buffer) (batch n : UInt32) : Buffer

/--
Unnormalized inverse real FFT with the same packed layout and endpoint handling as `irfftPacked`.

This numerical primitive omits the `1/n` factor inside the transform. Lean uses it for the real
FFT adjoint, avoiding both overflow from scaling the spectrum by `n` beforehand and underflow
from normalizing a tiny result before scaling it back up.
-/
@[never_extract, extern "torchlean_cuda_buffer_irfft1d_packed_unnormalized"]
opaque irfftPackedUnnormalized (spec : @& Buffer) (batch n : UInt32) : Buffer

/--
Diagonal selective scan for state-space models.

Inputs:
- `A`, `B`, `h0`: length `state`, representing per-channel recurrence parameters and initial state,
- `X`: length `seqLen*state`, row-major token/state inputs.

Output:
- length `seqLen*state`, row-major hidden states, with
  `h[t,j] = A[j] * h[t-1,j] + B[j] * X[t,j]`, starting from `h0[j]`.

This is the runtime primitive corresponding to the proof layer affine scan contract in
`TorchLean.Tensor.Layers.StateSpace` and `TorchLean.NN.StateSpace.Scan`.
The adapter evaluates the recurrence left to right in the order of
`TorchLean.StateSpace.scanStates`, rounding `B[j] * X[t,j]`, `A[j] * h[t-1,j]` and their sum
once each.
-/
@[never_extract, extern "torchlean_cuda_buffer_selective_scan_diag_fwd"]
opaque selectiveScanDiag (A B X h0 : @& Buffer) (seqLen state : UInt32) : Buffer

/--
Backward kernel for `selectiveScanDiag`.

Given `out = selectiveScanDiag A B X h0` and an upstream gradient `dY` with the same
`seqLen*state` layout as `out`, returns `(dA, dB, dX, dH0)`. The shared `dA[j]` and `dB[j]` add
their per-step terms from `+0` in increasing `t`, the order of
`TorchLean.Tensor.scanCoefficientGrad`.
-/
@[never_extract, extern "torchlean_cuda_buffer_selective_scan_diag_bwd"]
opaque selectiveScanDiagBackward (A B X h0 out dY : @& Buffer) (seqLen state : UInt32) :
    Buffer × Buffer × Buffer × Buffer

/--
Diagonal selective scan with token-dependent coefficients.

Inputs:
- `A`, `B`, `X`: length `seqLen*state`, row-major by `(time, flattened_state_channel)`,
- `h0`: length `state`.

Output:
- length `seqLen*state`, with
  `h[t,j] = A[t,j] * h[t-1,j] + B[t,j] * X[t,j]`.

This is the runtime primitive corresponding to full Mamba-style selective scans where the token
controls the affine transition coefficients. It evaluates the recurrence left to right like
`selectiveScanDiag`.
-/
@[never_extract, extern "torchlean_cuda_buffer_selective_scan_diag_var_fwd"]
opaque selectiveScanDiagVar (A B X h0 : @& Buffer) (seqLen state : UInt32) : Buffer

/--
Reverse accumulation for token-dependent diagonal coefficients.

With `g[t] = dY[t] + A[t+1] * g[t+1]`, the returned arrays are
`dA[t] = g[t] * h[t-1]`, `dB[t] = g[t] * X[t]`, `dX[t] = g[t] * B[t]`, and
`dH0 = A[0] * g[0]`. Empty sequences return an empty coefficient/input gradient and zero `dH0`.
The adapter applies the affine scan to the reversed sequence, then forms the pointwise gradients.
-/
@[never_extract, extern "torchlean_cuda_buffer_selective_scan_diag_var_bwd"]
opaque selectiveScanDiagVarBackward (A B X h0 out dY : @& Buffer) (seqLen state : UInt32) :
    Buffer × Buffer × Buffer × Buffer

/--
Broadcast a buffer to the shape `outDims`, with the semantics of `Shape.CanBroadcastTo`.

Arguments:
- `x`: input buffer
- `inDims`: input dimension list (outermost-first)
- `outDims`: output dimension list (outermost-first)
- `axisMap`: length `outDims.size`. `axisMap[j] = 0` marks output axis `j` as an inserted axis
  (input coordinate `0`), and otherwise `axisMap[j] = inAxis+1` names the input axis to read.

The native boundary requires every input axis to appear once in `axisMap`, with extent `1` or the
extent of its output axis. Lean generates the map from a `Shape.CanBroadcastTo` proof, so the
kernel never interprets the proof object.
-/
@[never_extract, extern "torchlean_cuda_buffer_broadcast_to"]
opaque broadcastTo (x : @& Buffer) (inDims outDims axisMap : @& Array Nat) : Buffer

/--
Adjoint of `broadcastTo` for sum-accumulation: reduce a broadcasted gradient back to the input
shape by summing over broadcasted axes.

The arguments `inDims`, `outDims` and `axisMap` follow the conventions of `broadcastTo`.
-/
@[never_extract, extern "torchlean_cuda_buffer_reduce_from_broadcast"]
opaque reduceFromBroadcastTo (dOut : @& Buffer) (inDims outDims axisMap : @& Array Nat) : Buffer

/--
Swap adjacent axes at `depth` for a contiguous buffer described by `dims`.

The call swaps axes `depth` and `depth + 1`, so `depth = 0` swaps the first two axes. It requires
`depth + 1 < dims.size`.
-/
@[never_extract, extern "torchlean_cuda_buffer_swap_adjacent_at_depth"]
opaque swapAdjacentAtDepth (x : @& Buffer) (dims : @& Array Nat) (depth : UInt32) : Buffer

/--
Reduce-sum along `axis` for an N-D contiguous buffer described by `dims` (outermost-first).

For nonempty `dims` it requires `axis < dims.size`, and the row-major result has shape `dims` with
entry `axis` removed. Empty `dims` (a scalar) gives a copy of the input.
-/
@[never_extract, extern "torchlean_cuda_buffer_reduce_sum_axis"]
opaque reduceSumAxis (x : @& Buffer) (dims : @& Array Nat) (axis : UInt32) : Buffer

/--
Gather `k` rows from a row-major matrix.

Input:
- `mat`: shape `(rows, cols)` stored row-major as length `rows*cols`
- `indices`: host `Array Nat` of length `k`

Output:
- shape `(k, cols)` stored row-major as length `k*cols`

An index that fits in `UInt32` but is out of bounds gives a row of zeros.
Large `Nat` values outside the FFI index range are rejected by the runtime.
-/
@[never_extract, extern "torchlean_cuda_buffer_gather_rows"]
opaque gatherRows (mat : @& Buffer) (rows cols : UInt32) (indices : @& Array Nat) (k : UInt32) :
  Buffer

/--
Scatter-add `k` rows given host indices.

Semantics: `out = mat` with `out[indices[r], j] += values[r, j]` for each `r < k`, `j < cols`.
Indices that fit in `UInt32` but are out of bounds are ignored, and repeated indices accumulate
(scatter-add). Large `Nat` values outside the FFI index range are rejected by the runtime.
-/
@[never_extract, extern "torchlean_cuda_buffer_scatter_add_rows"]
opaque scatterAddRows (mat values : @& Buffer) (rows cols : UInt32) (indices : @& Array Nat)
  (k : UInt32) : Buffer

end Buffer

end LibTorch
end TorchLean

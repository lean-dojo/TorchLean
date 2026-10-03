/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

LibTorch FFI: additional ATen operations over `LibTorch.Buffer` (float32).

Notes:
- `LibTorch.Buffer` is an opaque contiguous float32 buffer in CUDA device memory.
- These operations keep their shape APIs explicit: dimensions are passed as `UInt32`.
- Build with `scripts/lake.sh -Kcuda=true build` and a LibTorch SDK; the default build links failing
  placeholders for these symbols.
- TorchLean owns the differentiation tape. The LibTorch bridge disables graph recording and
  calls ATen forward/backward operators directly.
-/

module

public import NN.Runtime.Autograd.Engine.LibTorch.Trusted

/-!
# CUDA Buffer Kernels FFI

Foreign-function declarations for ATen operations on TorchLean's float32
`LibTorch.Buffer`: reductions, indexing, matmul/BMM, normalization, Fourier transforms, scans,
and broadcast/view helpers. Lean composes attention from these primitives in `Ops.Attention`.
The declarations here are the Lean side of the LibTorch CUDA trust boundary documented in
`docs/TRUST_BOUNDARIES.md`.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace LibTorch

namespace Buffer

/--
Sum across the columns of a 2D row-major buffer.

Input `b` has shape `(rows, cols)` and is stored as length `rows*cols`.
Output is length `rows` (sum across the columns for each row).
-/
@[never_extract, extern "torchlean_cuda_buffer_reduce_sum_by_row"]
opaque reduceSumByRow (b : @& Buffer) (rows cols : UInt32) : Buffer

/--
Maximum down the rows of a 2D row-major buffer.

Input `b` has shape `(rows, cols)` and is stored as length `rows*cols`.
Output is length `cols` (max down the rows for each column).
-/
@[never_extract, extern "torchlean_cuda_buffer_reduce_max_by_column"]
opaque reduceMaxByColumn (b : @& Buffer) (rows cols : UInt32) : Buffer

/--
Maximum across the columns of a 2D row-major buffer.

Input `b` has shape `(rows, cols)` and is stored as length `rows*cols`.
Output is length `rows` (max across the columns for each row).
-/
@[never_extract, extern "torchlean_cuda_buffer_reduce_max_by_row"]
opaque reduceMaxByRow (b : @& Buffer) (rows cols : UInt32) : Buffer

/--
Stable row-wise hard-masked softmax for flat `(rows, cols)` buffers.

The row maximum and denominator are computed only from entries whose mask value is nonzero. Blocked
entries are exactly zero in the output. A row with no allowed entries is defined to be all zeros.
-/
@[never_extract, extern "torchlean_cuda_buffer_hard_masked_softmax_by_row"]
opaque hardMaskedSoftmaxByRow (scores mask : @& Buffer) (rows cols : UInt32) : Buffer

/-- Concatenate two 1D buffers `a` (length `n`) and `b` (length `m`). -/
@[never_extract, extern "torchlean_cuda_buffer_concat1d"]
opaque concatBuffers (a b : @& Buffer) (n m : UInt32) : Buffer

/--
Slice a 1D buffer `b` (length `n`) starting at `start` for `len` elements.

Requires `start + len ≤ n`.
-/
@[never_extract, extern "torchlean_cuda_buffer_slice1d"]
opaque sliceBuffer (b : @& Buffer) (n start len : UInt32) : Buffer

/--
Broadcast a column-vector (length `rows`) to a `(rows, cols)` matrix.

Output is row-major of length `rows*cols`, with `out[i, j] = vec[i]`.
-/
@[never_extract, extern "torchlean_cuda_buffer_broadcast_vec_to_cols"]
opaque broadcastVecToCols (vec : @& Buffer) (rows cols : UInt32) : Buffer

/--
Layer normalization over the columns of a row-major `(rows, cols)` buffer.

`gamma` and `beta` each have length `cols`. The result is `(output, normalized, invStd)`, where
`normalized` has shape `(rows, cols)` and `invStd` has length `rows`. Keeping these two values is
enough for TorchLean's layer-normalization VJP; the native kernel does not create or own an
autograd graph.
-/
@[never_extract, extern "torchlean_cuda_buffer_layer_norm_fwd"]
opaque layerNormFwd
    (x gamma beta : @& Buffer) (rows cols : UInt32) (invCols epsilon : Float) :
    Buffer × Buffer × Buffer

/--
TorchLean's layer-normalization VJP evaluated by LibTorch tensor operations.

Given the upstream derivative, cached normalized values and inverse standard deviations, and
`gamma`, returns `(dX, dGamma, dBeta)`. The formula and parent association remain part of the
TorchLean tape; this primitive only evaluates that formula.
-/
@[never_extract, extern "torchlean_cuda_buffer_layer_norm_bwd"]
opaque layerNormBwd
    (dOut normalized invStd gamma : @& Buffer) (rows cols : UInt32)
    (colsScale invCols : Float) : Buffer × Buffer × Buffer

/--
Batched matrix multiply over row-major buffers, optionally transposing either logical operand.

The logical multiplication always has shape `(batch, m, n) × (batch, n, p)`. When
`transposeA = 1`, the stored shape of `A` is `(batch, n, m)`; when `transposeB = 1`, the stored
shape of `B` is `(batch, p, n)`. Other flag values are rejected by the native boundary.

LibTorch accepts these views. In particular, backward rules can request `Aᵀ B` or
`A Bᵀ` without first allocating a transposed buffer.
-/
@[never_extract, extern "torchlean_cuda_buffer_bmm_with_transpose"]
opaque bmmWithTranspose (A B : @& Buffer) (batch m n p transposeA transposeB : UInt32) : Buffer

/-- Batched matrix multiplication `A B` for ordinary row-major operands. -/
def bmm (A B : Buffer) (batch m n p : UInt32) : Buffer :=
  bmmWithTranspose A B batch m n p 0 0

/--
Batched multiplication `A Bᵀ`.

`A` is stored as `(batch, m, n)` and `B` as `(batch, p, n)`; the result has shape
`(batch, m, p)`.
-/
def bmmRightTranspose (A B : Buffer) (batch m n p : UInt32) : Buffer :=
  bmmWithTranspose A B batch m n p 0 1

/--
Batched multiplication `Aᵀ B`.

`A` is stored as `(batch, n, m)` and `B` as `(batch, n, p)`; the result has shape
`(batch, m, p)`.
-/
def bmmLeftTranspose (A B : Buffer) (batch m n p : UInt32) : Buffer :=
  bmmWithTranspose A B batch m n p 1 0

/--
Real-valued 1D FFT over row-major batches, returning a packed half-spectrum.

Input:
- `x`: length `batch*n`, interpreted as shape `(batch, n)`.

Output:
- length `batch*(n/2+1)*2`, interpreted as shape `(batch, n/2+1, 2)`;
- the last channel stores `[real, imag]` for each nonredundant frequency bin.

CUDA calls LibTorch's real FFT. This is a low-level runtime primitive; differentiable
tensor/autograd wrappers should spell out their backward convention separately because half-spectrum
packing has normalization and conjugate-symmetry edge cases.
-/
@[never_extract, extern "torchlean_cuda_buffer_rfft1d_packed"]
opaque rfft1dPacked (x : @& Buffer) (batch n : UInt32) : Buffer

/--
Inverse of `rfft1dPacked` for packed half-spectra.

Input:
- `spec`: length `batch*(n/2+1)*2`, interpreted as `(batch, n/2+1, 2)`.

Output:
- length `batch*n`, interpreted as `(batch, n)`.

The CUDA implementation calls LibTorch's inverse real FFT with `1/n` normalization, matching
the CPU reference.
-/
@[never_extract, extern "torchlean_cuda_buffer_irfft1d_packed"]
opaque irfft1dPacked (spec : @& Buffer) (batch n : UInt32) : Buffer

/--
Unnormalized inverse real FFT with the same packed layout and endpoint handling as `irfft1dPacked`.

This numerical primitive omits the `1/n` factor inside the transform. Lean uses it for the real
FFT adjoint, avoiding both overflow from scaling the spectrum by `n` beforehand and underflow
from normalizing a tiny result before scaling it back up.
-/
@[never_extract, extern "torchlean_cuda_buffer_irfft1d_packed_unnormalized"]
opaque irfft1dPackedUnnormalized (spec : @& Buffer) (batch n : UInt32) : Buffer

/--
Diagonal selective scan for state-space models.

Inputs:
- `A`, `B`, `h0`: length `state`, representing per-channel recurrence parameters and initial state,
- `X`: length `seqLen*state`, row-major token/state inputs.

Output:
- length `seqLen*state`, row-major hidden states, with
  `h[t,j] = A[j] * h[t-1,j] + B[j] * X[t,j]`, starting from `h0[j]`.

This is the runtime primitive corresponding to the proof layer affine scan contract in
`NN.Spec.Layers.SelectiveScan` and `NN.MLTheory.Proofs.StateSpace.Scan`.
The adapter composes ATen operations in a parallel affine scan for finite inputs whose
coefficients have magnitude at most one. Other inputs, or a nonfinite parallel result, use the
sequential recurrence. The parallel schedule can round differently from a left-to-right evaluation.
-/
@[never_extract, extern "torchlean_cuda_buffer_selective_scan_diag_fwd"]
opaque selectiveScanDiagFwd (A B X h0 : @& Buffer) (seqLen state : UInt32) : Buffer

/--
Backward kernel for `selectiveScanDiagFwd`.

Given `out = selectiveScanDiagFwd A B X h0` and an upstream gradient `dY` with the same
`seqLen*state` layout as `out`, returns `(dA, dB, dX, dH0)`.
-/
@[never_extract, extern "torchlean_cuda_buffer_selective_scan_diag_bwd"]
opaque selectiveScanDiagBwd (A B X h0 out dY : @& Buffer) (seqLen state : UInt32) :
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
controls the affine transition coefficients. It uses the same parallel or sequential scheduling
as `selectiveScanDiagFwd`.
-/
@[never_extract, extern "torchlean_cuda_buffer_selective_scan_diag_var_fwd"]
opaque selectiveScanDiagVarFwd (A B X h0 : @& Buffer) (seqLen state : UInt32) : Buffer

/--
Reverse accumulation for token-dependent diagonal coefficients.

With `g[t] = dY[t] + A[t+1] * g[t+1]`, the returned arrays are
`dA[t] = g[t] * h[t-1]`, `dB[t] = g[t] * X[t]`, `dX[t] = g[t] * B[t]`, and
`dH0 = A[0] * g[0]`. Empty sequences return an empty coefficient/input gradient and zero `dH0`.
The adapter applies the affine scan to the reversed sequence, then forms the pointwise gradients.
-/
@[never_extract, extern "torchlean_cuda_buffer_selective_scan_diag_var_bwd"]
opaque selectiveScanDiagVarBwd (A B X h0 out dY : @& Buffer) (seqLen state : UInt32) :
    Buffer × Buffer × Buffer × Buffer

/--
Scatter-add into a 1D vector using host indices.

Input:
- `x`: length `n`
- `values`: length `k`
- `indices`: `Array Nat` of length `k`

Semantics:
- returns a copy of `x` with `out[indices[j]] += values[j]` for each `j`,
- indices that fit in `UInt32` but are out of bounds are ignored,
- large `Nat` values outside the FFI index range are rejected by the runtime,
- repeated indices accumulate (scatter-add semantics).
-/
@[never_extract, extern "torchlean_cuda_buffer_scatter_add"]
opaque scatterAdd (x values : @& Buffer) (n : UInt32) (indices : @& Array Nat) (k : UInt32) : Buffer

/--
Broadcast a buffer to a new shape (TorchLean `Shape.CanBroadcastTo` semantics).

Arguments:
- `x`: input buffer
- `inDims`: input dimension list (outermost-first)
- `outDims`: output dimension list (outermost-first)
- `axisMap`: length `outDims.size`; `axisMap[j] = 0` means the output axis `j` is an
  inserted/broadcast axis (input coordinate is `0`), otherwise `axisMap[j] = inAxis+1` tells which
  input axis to read.

This shape-driven mapping is generated in Lean from a `Shape.CanBroadcastTo` proof so the kernel
does not need to interpret the proof object.
-/
@[never_extract, extern "torchlean_cuda_buffer_broadcast_to"]
opaque broadcastTo (x : @& Buffer) (inDims outDims axisMap : @& Array Nat) : Buffer

/--
Adjoint of `broadcastTo` for sum-accumulation: reduce a broadcasted gradient back to the input
shape by summing over broadcasted axes.

This uses the same `(inDims,outDims,axisMap)` convention as `broadcastTo`.
-/
@[never_extract, extern "torchlean_cuda_buffer_reduce_from_broadcast"]
opaque reduceFromBroadcastTo (dOut : @& Buffer) (inDims outDims axisMap : @& Array Nat) : Buffer

/--
Swap adjacent axes at `depth` for a contiguous buffer described by `dims`.

`depth = 0` swaps the first two axes; `depth = 1` swaps axes 1 and 2; etc.
-/
@[never_extract, extern "torchlean_cuda_buffer_swap_adjacent_at_depth"]
opaque swapAdjacentAtDepth (x : @& Buffer) (dims : @& Array Nat) (depth : UInt32) : Buffer

/--
Reduce-sum along `axis` for an N-D contiguous buffer described by `dims` (outermost-first).

The returned buffer is laid out row-major with shape `dims` with the `axis` dimension removed.
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

Indices that fit in `UInt32` but are out of bounds are totalized to `0` rows.
Large `Nat` values outside the FFI index range are rejected by the runtime.
-/
@[never_extract, extern "torchlean_cuda_buffer_gather_rows"]
opaque gatherRows (mat : @& Buffer) (rows cols : UInt32) (indices : @& Array Nat) (k : UInt32) :
  Buffer

/--
Scatter-add `k` rows given host indices.

Semantics: `out = mat` with `out[indices[r], j] += values[r, j]` for each `r < k`, `j < cols`.
Indices that fit in `UInt32` but are out of bounds are ignored; repeated indices accumulate
(scatter-add). Large `Nat` values outside the FFI index range are rejected by the runtime.
-/
@[never_extract, extern "torchlean_cuda_buffer_scatter_add_rows"]
opaque scatterAddRows (mat values : @& Buffer) (rows cols : UInt32) (indices : @& Array Nat)
  (k : UInt32) : Buffer

/--
Gather `table[positions[i]]` for `count` device-resident positions.

`table` has `size > 0` entries. Positions are float32 values computed on device: a NaN reads as
`0`, every position is clamped to `[0, size - 1]` and then truncated, so a non-integral position
floors and no selection leaves the table. The result has `count` entries.
-/
@[never_extract, extern "torchlean_cuda_buffer_gather_at"]
opaque gatherAt (table : @& Buffer) (size : UInt32) (positions : @& Buffer) (count : UInt32) :
  Buffer

/--
Scatter-add `count` values into a fresh zero buffer of `size > 0` entries at device-resident
positions, addressed exactly as in `gatherAt`: the VJP of `gatherAt` with respect to its table.
Repeated positions accumulate; the native accumulation order is unspecified unless deterministic
algorithms are selected.
-/
@[never_extract, extern "torchlean_cuda_buffer_scatter_add_at"]
opaque scatterAddAt (size : UInt32) (positions values : @& Buffer) (count : UInt32) : Buffer

end Buffer

end LibTorch
end Autograd
end Runtime

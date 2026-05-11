/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

CUDA FFI: additional kernels over `Cuda.Buffer` (float32) to support composite ops.

Notes:
- `Cuda.Buffer` is an opaque contiguous float32 buffer (device memory when built with `-K cuda=true`,
  otherwise a CPU stub buffer).
- These kernels keep their shape APIs explicit: dimensions are passed as `UInt32`.
- Build with `lake build -K cuda=true` to use real CUDA kernels at runtime; otherwise the stub
  implementation runs on CPU for portability.
-/

module


public import NN.Runtime.Autograd.Engine.Cuda.Buffer

/-!
# CUDA Buffer Kernels FFI

Foreign-function declarations for TorchLean's float32 `Cuda.Buffer` kernels: reductions, indexing,
matmul/BMM, attention, broadcast/view helpers, and related tensor operations. The declarations here
are the Lean side of the explicit CUDA trust boundary documented in `TRUST_BOUNDARIES.md`.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Cuda

namespace Buffer

/--
Sum over axis 0 of a 2D buffer in row-major order.

Input `b` has shape `(rows, cols)` and is stored as length `rows*cols`.
Output is length `cols` (sum down the rows for each column).
-/
@[extern "torchlean_cuda_buffer_reduce_sum_axis0"]
opaque reduceSumAxis0 (b : Buffer) (rows cols : UInt32) : Buffer

/--
Sum over axis 1 of a 2D buffer in row-major order.

Input `b` has shape `(rows, cols)` and is stored as length `rows*cols`.
Output is length `rows` (sum across the columns for each row).
-/
@[extern "torchlean_cuda_buffer_reduce_sum_axis1"]
opaque reduceSumAxis1 (b : Buffer) (rows cols : UInt32) : Buffer

/--
Max over axis 0 of a 2D buffer in row-major order.

Input `b` has shape `(rows, cols)` and is stored as length `rows*cols`.
Output is length `cols` (max down the rows for each column).
-/
@[extern "torchlean_cuda_buffer_reduce_max_axis0"]
opaque reduceMaxAxis0 (b : Buffer) (rows cols : UInt32) : Buffer

/--
Max over axis 1 of a 2D buffer in row-major order.

Input `b` has shape `(rows, cols)` and is stored as length `rows*cols`.
Output is length `rows` (max across the columns for each row).
-/
@[extern "torchlean_cuda_buffer_reduce_max_axis1"]
opaque reduceMaxAxis1 (b : Buffer) (rows cols : UInt32) : Buffer

/-- Concatenate two 1D buffers `a` (length `n`) and `b` (length `m`). -/
@[extern "torchlean_cuda_buffer_concat1d"]
opaque concat1d (a b : Buffer) (n m : UInt32) : Buffer

/--
Slice a 1D buffer `b` (length `n`) starting at `start` for `len` elements.

Requires `start + len ≤ n`.
-/
@[extern "torchlean_cuda_buffer_slice1d"]
opaque slice1d (b : Buffer) (n start len : UInt32) : Buffer

/--
Broadcast a row-vector (length `cols`) to a `(rows, cols)` matrix.

Output is row-major of length `rows*cols`, with `out[i, j] = vec[j]`.
-/
@[extern "torchlean_cuda_buffer_broadcast_vec_to_rows"]
opaque broadcastVecToRows (vec : Buffer) (rows cols : UInt32) : Buffer

/--
Broadcast a column-vector (length `rows`) to a `(rows, cols)` matrix.

Output is row-major of length `rows*cols`, with `out[i, j] = vec[i]`.
-/
@[extern "torchlean_cuda_buffer_broadcast_vec_to_cols"]
opaque broadcastVecToCols (vec : Buffer) (rows cols : UInt32) : Buffer

/--
Batched matrix multiply over row-major buffers.

Input:
- `A`: length `batch*m*n` representing `batch` matrices of shape `(m, n)` (row-major)
- `B`: length `batch*n*p` representing `batch` matrices of shape `(n, p)` (row-major)
Output:
- length `batch*m*p` representing `batch` matrices of shape `(m, p)` (row-major)
-/
@[extern "torchlean_cuda_buffer_bmm"]
opaque bmm (A B : Buffer) (batch m n p : UInt32) : Buffer

/--
Real-valued 1D FFT over row-major batches, returning a packed half-spectrum.

Input:
- `x`: length `batch*n`, interpreted as shape `(batch, n)`.

Output:
- length `batch*(n/2+1)*2`, interpreted as shape `(batch, n/2+1, 2)`;
- the last channel stores `[real, imag]` for each nonredundant frequency bin.

CUDA uses cuFFT `R2C` under the hood. The CPU stub uses a direct reference DFT, so this primitive
remains available in non-CUDA builds for tests and portability. This is a low-level runtime
primitive; differentiable tensor/autograd wrappers should spell out their backward convention
separately because half-spectrum packing has normalization and conjugate-symmetry edge cases.
-/
@[extern "torchlean_cuda_buffer_rfft1d_packed"]
opaque rfft1dPacked (x : Buffer) (batch n : UInt32) : Buffer

/--
Inverse of `rfft1dPacked` for packed half-spectra.

Input:
- `spec`: length `batch*(n/2+1)*2`, interpreted as `(batch, n/2+1, 2)`.

Output:
- length `batch*n`, interpreted as `(batch, n)`.

The CUDA implementation uses cuFFT `C2R` and explicitly scales by `1/n`, matching the CPU reference
and the usual normalized inverse FFT convention used by high-level ML APIs.
-/
@[extern "torchlean_cuda_buffer_irfft1d_packed"]
opaque irfft1dPacked (spec : Buffer) (batch n : UInt32) : Buffer

/--
Fused real-FFT spectral convolution for one FNO1D block.

Input:
- `x`: length `grid*width`, row-major shape `(grid, width)`;
- `wRe`, `wIm`: length `modes*width*width`, row-major shape `(modes, width, width)`.

Semantics:
1. apply an unnormalized real FFT along the grid axis for each input channel,
2. keep frequency bins `0 ≤ k < modes`,
3. multiply each retained complex vector by `wRe[k] + i*wIm[k]`,
4. zero all other bins,
5. apply the normalized inverse real FFT.

This is the CUDA/cuFFT-backed runtime primitive intended to replace dense DFT matrix multiplies in
float32 FNO examples. The three backward primitives below are its explicit VJP components.
-/
@[extern "torchlean_cuda_buffer_spectral_conv1d_rfft_fwd"]
opaque spectralConv1dRfftFwd
    (x wRe wIm : Buffer) (grid width modes : UInt32) : Buffer

/-- VJP component `∂L/∂x` for `spectralConv1dRfftFwd`. -/
@[extern "torchlean_cuda_buffer_spectral_conv1d_rfft_bwd_x"]
opaque spectralConv1dRfftBwdX
    (x wRe wIm dY : Buffer) (grid width modes : UInt32) : Buffer

/-- VJP component `∂L/∂wRe` for `spectralConv1dRfftFwd`. -/
@[extern "torchlean_cuda_buffer_spectral_conv1d_rfft_bwd_wre"]
opaque spectralConv1dRfftBwdWRe
    (x wRe wIm dY : Buffer) (grid width modes : UInt32) : Buffer

/-- VJP component `∂L/∂wIm` for `spectralConv1dRfftFwd`. -/
@[extern "torchlean_cuda_buffer_spectral_conv1d_rfft_bwd_wim"]
opaque spectralConv1dRfftBwdWIm
    (x wRe wIm dY : Buffer) (grid width modes : UInt32) : Buffer

/--
Diagonal selective-scan forward kernel for state-space models.

Inputs:
- `A`, `B`, `h0`: length `state`, representing per-channel recurrence parameters and initial state,
- `X`: length `seqLen*state`, row-major token/state inputs.

Output:
- length `seqLen*state`, row-major hidden states, with
  `h[t,j] = A[j] * h[t-1,j] + B[j] * X[t,j]`, starting from `h0[j]`.

This is the runtime primitive corresponding to the proof-facing affine scan contract in
`NN.Spec.Layers.SelectiveScan` and `NN.MLTheory.Proofs.StateSpace.Scan`.
-/
@[extern "torchlean_cuda_buffer_selective_scan_diag_fwd"]
opaque selectiveScanDiagFwd (A B X h0 : Buffer) (seqLen state : UInt32) : Buffer

/--
Backward kernel for `selectiveScanDiagFwd`.

Given `out = selectiveScanDiagFwd A B X h0` and an upstream gradient `dY` with the same
`seqLen*state` layout as `out`, returns `(dA, dB, dX, dH0)`.
-/
@[extern "torchlean_cuda_buffer_selective_scan_diag_bwd"]
opaque selectiveScanDiagBwd (A B X h0 out dY : Buffer) (seqLen state : UInt32) :
    Buffer × Buffer × Buffer × Buffer

/--
Diagonal selective-scan forward kernel with token-dependent coefficients.

Inputs:
- `A`, `B`, `X`: length `seqLen*state`, row-major by `(time, flattened_state_channel)`,
- `h0`: length `state`.

Output:
- length `seqLen*state`, with
  `h[t,j] = A[t,j] * h[t-1,j] + B[t,j] * X[t,j]`.

This is the runtime primitive corresponding to full Mamba-style selective scans where the token
controls the affine transition coefficients.
-/
@[extern "torchlean_cuda_buffer_selective_scan_diag_var_fwd"]
opaque selectiveScanDiagVarFwd (A B X h0 : Buffer) (seqLen state : UInt32) : Buffer

/--
Native fused scaled dot-product attention forward over split attention heads.

Inputs are row-major buffers with shapes:
- `Q`, `K`, `V`: `(batch, n, d)`, where `batch` is usually the number of heads,
- `mask`: `(batch, n, n)` encoded as `0.0/1.0` when `hasMask != 0`; otherwise ignored.

Output has shape `(batch, n, d)` and computes the same no-dropout masked attention semantics as:
`softmax((Q Kᵀ) * scale + maskFill) V`, where blocked mask entries use TorchLean's
`-1000.0` fill convention.

This is a fused native runtime primitive, not a proof object. The proof-facing contract is
`Spec.flashAttention` in `NN/Spec/Layers/FlashAttention.lean`.
-/
@[extern "torchlean_cuda_buffer_flash_attention_fwd"]
opaque flashAttentionFwd
    (Q K V mask : Buffer) (hasMask batch n d : UInt32) (scale : Float) : Buffer

/-- Fused VJP component `∂L/∂Q` for `flashAttentionFwd`. -/
@[extern "torchlean_cuda_buffer_flash_attention_bwd_q"]
opaque flashAttentionBwdQ
    (Q K V mask dOut : Buffer) (hasMask batch n d : UInt32) (scale : Float) : Buffer

/-- Fused VJP component `∂L/∂K` for `flashAttentionFwd`. -/
@[extern "torchlean_cuda_buffer_flash_attention_bwd_k"]
opaque flashAttentionBwdK
    (Q K V mask dOut : Buffer) (hasMask batch n d : UInt32) (scale : Float) : Buffer

/-- Fused VJP component `∂L/∂V` for `flashAttentionFwd`. -/
@[extern "torchlean_cuda_buffer_flash_attention_bwd_v"]
opaque flashAttentionBwdV
    (Q K V mask dOut : Buffer) (hasMask batch n d : UInt32) (scale : Float) : Buffer

/--
Row-major transpose of a 2D buffer.

Input `b` has shape `(rows, cols)` and is stored as length `rows*cols`.
Output has shape `(cols, rows)` and is stored as length `rows*cols` (row-major).
-/
@[extern "torchlean_cuda_buffer_transpose2d"]
opaque transpose2d (b : Buffer) (rows cols : UInt32) : Buffer

/--
Gather `k` scalars from a 1D vector using host indices.

Input:
- `vec`: length `n`
- `indices`: `Array Nat` of length `k`

Indices that fit in `UInt32` but are out of bounds are totalized to `0`.
Large `Nat` values outside the FFI index range are rejected by the runtime.
-/
@[extern "torchlean_cuda_buffer_gather_vec"]
opaque gatherVec (vec : Buffer) (n : UInt32) (indices : Array Nat) (k : UInt32) : Buffer

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
@[extern "torchlean_cuda_buffer_scatter_add"]
opaque scatterAdd (x values : Buffer) (n : UInt32) (indices : Array Nat) (k : UInt32) : Buffer

/--
Broadcast a buffer to a new shape (TorchLean `Shape.CanBroadcastTo` semantics).

Arguments:
- `x`: input buffer
- `inDims`: input dimension list (outermost-first)
- `outDims`: output dimension list (outermost-first)
- `axisMap`: length `outDims.size`; `axisMap[j] = 0` means the output axis `j` is an inserted/broadcast
  axis (input coordinate is `0`), otherwise `axisMap[j] = inAxis+1` tells which input axis to read.

This shape-driven mapping is generated in Lean from a `Shape.CanBroadcastTo` proof so the kernel
does not need to interpret the proof object.
-/
@[extern "torchlean_cuda_buffer_broadcast_to"]
opaque broadcastTo (x : Buffer) (inDims outDims axisMap : Array Nat) : Buffer

/--
Adjoint of `broadcastTo` for sum-accumulation: reduce a broadcasted gradient back to the input
shape by summing over broadcasted axes.

This uses the same `(inDims,outDims,axisMap)` convention as `broadcastTo`.
-/
@[extern "torchlean_cuda_buffer_reduce_from_broadcast"]
opaque reduceFromBroadcastTo (dOut : Buffer) (inDims outDims axisMap : Array Nat) : Buffer

/--
Swap adjacent axes at `depth` for a contiguous buffer described by `dims`.

`depth = 0` swaps the first two axes; `depth = 1` swaps axes 1 and 2; etc.
-/
@[extern "torchlean_cuda_buffer_swap_adjacent_at_depth"]
opaque swapAdjacentAtDepth (x : Buffer) (dims : Array Nat) (depth : UInt32) : Buffer

/--
Reduce-sum along `axis` for an N-D contiguous buffer described by `dims` (outermost-first).

The returned buffer is laid out row-major with shape `dims` with the `axis` dimension removed.
-/
@[extern "torchlean_cuda_buffer_reduce_sum_axis"]
opaque reduceSumAxis (x : Buffer) (dims : Array Nat) (axis : UInt32) : Buffer

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
@[extern "torchlean_cuda_buffer_gather_rows"]
opaque gatherRows (mat : Buffer) (rows cols : UInt32) (indices : Array Nat) (k : UInt32) : Buffer

/--
Scatter-add into a single matrix row.

Returns a copy of `mat` with `out[i,:] += rowVec`.
-/
@[extern "torchlean_cuda_buffer_scatter_add_row"]
opaque scatterAddRow (mat rowVec : Buffer) (rows cols : UInt32) (i : UInt32) : Buffer

/--
Scatter-add `k` rows given host indices.

Semantics: `out = mat` with `out[indices[r], j] += values[r, j]` for each `r < k`, `j < cols`.
Indices that fit in `UInt32` but are out of bounds are ignored; repeated indices accumulate
(scatter-add). Large `Nat` values outside the FFI index range are rejected by the runtime.
-/
@[extern "torchlean_cuda_buffer_scatter_add_rows"]
opaque scatterAddRows (mat values : Buffer) (rows cols : UInt32) (indices : Array Nat) (k : UInt32) :
  Buffer

end Buffer

end Cuda
end Autograd
end Runtime

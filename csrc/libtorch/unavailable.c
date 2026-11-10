#include <lean/lean.h>
#include <lean/mimalloc.h>

#include <stdint.h>

// Backend linked without LibTorch (the default `scripts/lake.sh build`).
//
// Every Lean extern of the GPU buffer ABI resolves here so the default build links, but no buffer
// can ever be created: `torchlean_cuda_runtime_status` reports 0, CUDA sessions are rejected
// before they start, IO constructors return an error, and the pure operations panic. Build with
// `-K cuda=true` to link `csrc/libtorch` instead.

#define TORCHLEAN_UNAVAILABLE_MESSAGE \
  "TorchLean was built without LibTorch; rebuild with `scripts/lake.sh -R -K cuda=true build`"

static lean_obj_res unavailable_io(void) {
  return lean_io_result_mk_error(
      lean_mk_io_user_error(lean_mk_string(TORCHLEAN_UNAVAILABLE_MESSAGE)));
}

LEAN_EXPORT lean_obj_res torchlean_kernel_run_buffer(
    b_lean_obj_arg source, uint8_t dtype, b_lean_obj_arg inputs, uint64_t count) {
  (void)source; (void)dtype; (void)inputs; (void)count;
  return unavailable_io();
}

LEAN_EXPORT lean_obj_res torchlean_kernel_run_bytes(
    b_lean_obj_arg source, uint32_t format, uint64_t width,
    b_lean_obj_arg inputs, uint64_t count) {
  (void)source; (void)format; (void)width; (void)inputs; (void)count;
  return unavailable_io();
}

LEAN_EXPORT lean_obj_res torchlean_tree_upload(
    uint32_t format, b_lean_obj_arg descriptor, uint64_t width, uint64_t count,
    uint64_t index_slots, uint64_t scalar_slots, b_lean_obj_arg words,
    b_lean_obj_arg scalars) {
  (void)format; (void)descriptor; (void)width; (void)count; (void)index_slots;
  (void)scalar_slots; (void)words; (void)scalars;
  return unavailable_io();
}

LEAN_EXPORT lean_obj_res torchlean_tree_words(
    b_lean_obj_arg tree, uint32_t format, b_lean_obj_arg descriptor, uint64_t width) {
  (void)tree; (void)format; (void)descriptor; (void)width;
  return unavailable_io();
}

LEAN_EXPORT lean_obj_res torchlean_tree_scalars(
    b_lean_obj_arg tree, uint32_t format, b_lean_obj_arg descriptor, uint64_t width) {
  (void)tree; (void)format; (void)descriptor; (void)width;
  return unavailable_io();
}

LEAN_EXPORT lean_obj_res torchlean_tree_fold(
    b_lean_obj_arg tree, uint32_t format, b_lean_obj_arg descriptor, uint64_t width,
    uint64_t index_slots, uint64_t scalar_slots, b_lean_obj_arg source, b_lean_obj_arg sizes,
    b_lean_obj_arg inputs) {
  (void)tree; (void)format; (void)descriptor; (void)width; (void)index_slots;
  (void)scalar_slots; (void)source; (void)sizes; (void)inputs;
  return unavailable_io();
}

LEAN_EXPORT lean_obj_res torchlean_tree_expand(
    uint32_t format, b_lean_obj_arg descriptor, uint64_t width, uint64_t index_slots,
    uint64_t scalar_slots, uint64_t count, b_lean_obj_arg words, b_lean_obj_arg scalars,
    b_lean_obj_arg calls, b_lean_obj_arg arguments, b_lean_obj_arg nodes,
    uint64_t node_index_slots, uint64_t node_scalar_slots, b_lean_obj_arg inputs,
    uint64_t capacity) {
  (void)format; (void)descriptor; (void)width; (void)index_slots; (void)scalar_slots;
  (void)count; (void)words; (void)scalars; (void)calls; (void)arguments; (void)nodes;
  (void)node_index_slots; (void)node_scalar_slots; (void)inputs; (void)capacity;
  return unavailable_io();
}

// 0 = built without LibTorch, 1 = LibTorch with a visible device, 2 = LibTorch without one.
LEAN_EXPORT lean_obj_res torchlean_cuda_runtime_status(void) {
  return lean_io_result_mk_ok(lean_box_uint32(0u));
}

LEAN_EXPORT lean_obj_res torchlean_libtorch_version(void) {
  return lean_io_result_mk_ok(lean_mk_string("unavailable"));
}

// Returning unused Lean heap pages to the OS is host work and does not need LibTorch.
LEAN_EXPORT lean_obj_res torchlean_runtime_collect_allocator(void) {
  mi_collect(false);
  return lean_io_result_mk_ok(lean_box(0));
}

// Settings read as off and memory statistics read as zero, so reports work in either build.
LEAN_EXPORT lean_obj_res torchlean_libtorch_get_setting(uint32_t setting) {
  if (setting > 8u) {
    return lean_io_result_mk_error(
        lean_mk_io_user_error(lean_mk_string("LibTorch: unknown runtime setting")));
  }
  return lean_io_result_mk_ok(lean_box_uint32(0u));
}

LEAN_EXPORT lean_obj_res torchlean_libtorch_get_memory_fraction(void) {
  return lean_io_result_mk_ok(lean_box_float(0.0));
}

#define ZERO(BOX, NAME) \
  LEAN_EXPORT lean_obj_res NAME(void) { return lean_io_result_mk_ok(BOX(0u)); }

ZERO(lean_box_uint32, torchlean_libtorch_device_count)
ZERO(lean_box_uint32, torchlean_libtorch_get_device)
ZERO(lean_box_uint64, torchlean_libtorch_allocated_bytes)
ZERO(lean_box_uint64, torchlean_libtorch_reserved_bytes)
ZERO(lean_box_uint64, torchlean_libtorch_peak_allocated_bytes)
ZERO(lean_box_uint64, torchlean_libtorch_peak_reserved_bytes)
ZERO(lean_box_uint64, torchlean_cuda_allocator_live_bytes)
ZERO(lean_box_uint64, torchlean_cuda_allocator_peak_bytes)
ZERO(lean_box_uint64, torchlean_cuda_allocator_alloc_count)
ZERO(lean_box_uint64, torchlean_cuda_allocator_free_count)
ZERO(lean_box_uint64, torchlean_cuda_wrapper_live_count)
ZERO(lean_box_uint64, torchlean_cuda_wrapper_peak_count)
ZERO(lean_box_uint64, torchlean_cuda_wrapper_alloc_count)
ZERO(lean_box_uint64, torchlean_cuda_wrapper_finalize_count)
ZERO(lean_box_uint64, torchlean_cuda_allocator_device_free_bytes)
ZERO(lean_box_uint64, torchlean_cuda_allocator_device_total_bytes)
#undef ZERO

#define UNAVAILABLE_IO(NAME, ...) \
  LEAN_EXPORT lean_obj_res NAME(__VA_ARGS__) { return unavailable_io(); }

UNAVAILABLE_IO(torchlean_libtorch_set_device, uint32_t index)
UNAVAILABLE_IO(torchlean_libtorch_set_setting, uint32_t setting, uint32_t enabled)
UNAVAILABLE_IO(torchlean_libtorch_set_memory_fraction, double fraction)
UNAVAILABLE_IO(torchlean_libtorch_synchronize, void)
UNAVAILABLE_IO(torchlean_libtorch_empty_cache, void)
UNAVAILABLE_IO(torchlean_cuda_buffer_zeros_io, uint32_t n, uint8_t dtype)
UNAVAILABLE_IO(torchlean_cuda_buffer_full_io, uint32_t n, double v, uint8_t dtype)
UNAVAILABLE_IO(torchlean_cuda_buffer_rand_uniform_io, uint32_t n, uint64_t key, uint8_t dtype)
UNAVAILABLE_IO(torchlean_cuda_buffer_bernoulli_mask_io,
    uint32_t n, double keepProb, uint64_t key, uint8_t dtype)
UNAVAILABLE_IO(torchlean_cuda_buffer_of_float_array_io, b_lean_obj_arg AObj, uint8_t dtype)
UNAVAILABLE_IO(torchlean_cuda_buffer_of_encoded_io,
    b_lean_obj_arg bytes, b_lean_obj_arg format, uint64_t width)
UNAVAILABLE_IO(torchlean_cuda_buffer_to_encoded_io, b_lean_obj_arg buffer)
UNAVAILABLE_IO(torchlean_cuda_buffer_to_float_array_io, b_lean_obj_arg BObj)
UNAVAILABLE_IO(torchlean_cuda_buffer_size_io, b_lean_obj_arg BObj)
UNAVAILABLE_IO(torchlean_cuda_buffer_release_io, b_lean_obj_arg BObj)
UNAVAILABLE_IO(torchlean_cuda_buffer_to_bytes_io, b_lean_obj_arg BObj, uint8_t dtype)
UNAVAILABLE_IO(torchlean_cuda_buffer_of_bytes_io,
    b_lean_obj_arg BytesObj, uint8_t dtype, uint8_t source)
UNAVAILABLE_IO(torchlean_cuda_buffer_conv_io,
    b_lean_obj_arg input, b_lean_obj_arg kernel, b_lean_obj_arg bias,
    b_lean_obj_arg input_dims, b_lean_obj_arg kernel_dims, b_lean_obj_arg stride,
    b_lean_obj_arg padding_before, b_lean_obj_arg padding_after, b_lean_obj_arg dilation,
    uint32_t groups)
#undef UNAVAILABLE_IO

// Reached only through a buffer, and no buffer exists in this build.
#define UNAVAILABLE(TYPE, NAME, ...) \
  LEAN_EXPORT TYPE NAME(__VA_ARGS__) { lean_internal_panic(TORCHLEAN_UNAVAILABLE_MESSAGE); }

UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_conv_bwd,
    b_lean_obj_arg input, b_lean_obj_arg kernel, b_lean_obj_arg gradient,
    b_lean_obj_arg input_dims, b_lean_obj_arg kernel_dims, b_lean_obj_arg stride,
    b_lean_obj_arg padding_before, b_lean_obj_arg padding_after, b_lean_obj_arg dilation,
    uint32_t groups)
UNAVAILABLE(lean_obj_res, torchlean_cuda_convtranspose_fwd,
    b_lean_obj_arg inputObj, b_lean_obj_arg kernelObj, b_lean_obj_arg biasObj,
    b_lean_obj_arg inSpatialObj, b_lean_obj_arg kernelSpatialObj, b_lean_obj_arg strideObj,
    b_lean_obj_arg paddingObj, uint32_t inC, uint32_t outC)
UNAVAILABLE(lean_obj_res, torchlean_cuda_convtranspose_bwd,
    b_lean_obj_arg inputObj, b_lean_obj_arg kernelObj, b_lean_obj_arg gradObj,
    b_lean_obj_arg inSpatialObj, b_lean_obj_arg kernelSpatialObj, b_lean_obj_arg strideObj,
    b_lean_obj_arg paddingObj, uint32_t inC, uint32_t outC)
UNAVAILABLE(lean_obj_res, torchlean_cuda_maxpool_fwd,
    b_lean_obj_arg inputObj, b_lean_obj_arg inSpatialObj, b_lean_obj_arg kernelObj,
    b_lean_obj_arg strideObj, b_lean_obj_arg paddingObj, uint32_t inC)
UNAVAILABLE(lean_obj_res, torchlean_cuda_maxpool_bwd,
    b_lean_obj_arg inputObj, b_lean_obj_arg gradObj, b_lean_obj_arg inSpatialObj,
    b_lean_obj_arg kernelObj, b_lean_obj_arg strideObj, b_lean_obj_arg paddingObj, uint32_t inC)
UNAVAILABLE(lean_obj_res, torchlean_cuda_avgpool_fwd,
    b_lean_obj_arg inputObj, b_lean_obj_arg inSpatialObj, b_lean_obj_arg kernelObj,
    b_lean_obj_arg strideObj, b_lean_obj_arg paddingObj, uint32_t inC)
UNAVAILABLE(lean_obj_res, torchlean_cuda_avgpool_bwd,
    b_lean_obj_arg gradObj, b_lean_obj_arg inSpatialObj, b_lean_obj_arg kernelObj,
    b_lean_obj_arg strideObj, b_lean_obj_arg paddingObj, uint32_t inC)
UNAVAILABLE(lean_obj_res, torchlean_cuda_smooth_maxpool_fwd,
    b_lean_obj_arg inputObj, double beta, b_lean_obj_arg inSpatialObj, b_lean_obj_arg kernelObj,
    b_lean_obj_arg strideObj, b_lean_obj_arg paddingObj, uint32_t inC)
UNAVAILABLE(lean_obj_res, torchlean_cuda_smooth_maxpool_bwd,
    b_lean_obj_arg inputObj, b_lean_obj_arg gradObj, double beta, b_lean_obj_arg inSpatialObj,
    b_lean_obj_arg kernelObj, b_lean_obj_arg strideObj, b_lean_obj_arg paddingObj, uint32_t inC)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_reduce_sum_by_row,
    b_lean_obj_arg BObj, uint32_t rows, uint32_t cols)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_reduce_max_by_row,
    b_lean_obj_arg BObj, uint32_t rows, uint32_t cols)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_argmax_mask_by_row,
    b_lean_obj_arg BObj, uint32_t rows, uint32_t cols)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_distill,
    b_lean_obj_arg student, b_lean_obj_arg teacher, b_lean_obj_arg dimensions,
    uint32_t axis, uint8_t gradient)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_cross_entropy,
    b_lean_obj_arg logits, b_lean_obj_arg target, b_lean_obj_arg dimensions,
    uint32_t axis, uint8_t mean)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_cross_entropy_bwd,
    b_lean_obj_arg logits, b_lean_obj_arg target, b_lean_obj_arg seed,
    b_lean_obj_arg dimensions, uint32_t axis, uint8_t mean)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_top_k_mask_by_row,
    b_lean_obj_arg BObj, uint32_t rows, uint32_t cols, uint32_t k)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_hard_masked_softmax_by_row,
    b_lean_obj_arg ScoresObj, b_lean_obj_arg MaskObj, uint32_t rows, uint32_t cols)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_concat1d,
    b_lean_obj_arg AObj, b_lean_obj_arg BObj, uint32_t n, uint32_t m)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_slice1d,
    b_lean_obj_arg BObj, uint32_t n, uint32_t start, uint32_t len)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_broadcast_vec_to_cols,
    b_lean_obj_arg VObj, uint32_t rows, uint32_t cols)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_layer_norm_fwd,
    b_lean_obj_arg XObj, b_lean_obj_arg GammaObj, b_lean_obj_arg BetaObj, uint32_t rows,
    uint32_t cols, double epsilonArg)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_layer_norm_bwd,
    b_lean_obj_arg DOutObj, b_lean_obj_arg NormalizedObj, b_lean_obj_arg InvStdObj,
    b_lean_obj_arg GammaObj, uint32_t rows, uint32_t cols)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_div_count, b_lean_obj_arg XObj, uint64_t count)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_mse_mean, b_lean_obj_arg XObj, uint64_t count)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_mse_slope, b_lean_obj_arg XObj, uint64_t count)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_bmm_with_transpose,
    b_lean_obj_arg AObj, b_lean_obj_arg BObj, uint32_t batch, uint32_t m, uint32_t n, uint32_t p,
    uint32_t transposeA, uint32_t transposeB)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_scaled_bmm,
    b_lean_obj_arg AObj, b_lean_obj_arg BObj, uint32_t batch, uint32_t m, uint32_t n, uint32_t p,
    uint32_t transposeA, uint32_t transposeB, double scale)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_rfft1d_packed,
    b_lean_obj_arg XObj, uint32_t batch, uint32_t n)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_irfft1d_packed,
    b_lean_obj_arg SpecObj, uint32_t batch, uint32_t n)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_irfft1d_packed_unnormalized,
    b_lean_obj_arg SpecObj, uint32_t batch, uint32_t n)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_selective_scan_diag_fwd,
    b_lean_obj_arg AObj, b_lean_obj_arg BObj, b_lean_obj_arg XObj, b_lean_obj_arg H0Obj,
    uint32_t seqLen, uint32_t stateDim)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_selective_scan_diag_bwd,
    b_lean_obj_arg AObj, b_lean_obj_arg BObj, b_lean_obj_arg XObj, b_lean_obj_arg H0Obj,
    b_lean_obj_arg OutObj, b_lean_obj_arg DYObj, uint32_t seqLen, uint32_t stateDim)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_selective_scan_diag_var_fwd,
    b_lean_obj_arg AObj, b_lean_obj_arg BObj, b_lean_obj_arg XObj, b_lean_obj_arg H0Obj,
    uint32_t seqLen, uint32_t stateDim)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_selective_scan_diag_var_bwd,
    b_lean_obj_arg AObj, b_lean_obj_arg BObj, b_lean_obj_arg XObj, b_lean_obj_arg H0Obj,
    b_lean_obj_arg OutObj, b_lean_obj_arg DYObj, uint32_t seqLen, uint32_t stateDim)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_broadcast_to,
    b_lean_obj_arg XObj, b_lean_obj_arg InDimsObj, b_lean_obj_arg OutDimsObj,
    b_lean_obj_arg AxisMapObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_reduce_from_broadcast,
    b_lean_obj_arg DOutObj, b_lean_obj_arg InDimsObj, b_lean_obj_arg OutDimsObj,
    b_lean_obj_arg AxisMapObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_swap_adjacent_at_depth,
    b_lean_obj_arg XObj, b_lean_obj_arg DimsObj, uint32_t depth)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_reduce_sum_axis,
    b_lean_obj_arg XObj, b_lean_obj_arg DimsObj, uint32_t axis)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_gather_rows,
    b_lean_obj_arg MObj, uint32_t rows, uint32_t cols, b_lean_obj_arg IdxObj, uint32_t k)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_scatter_add_rows,
    b_lean_obj_arg MObj, b_lean_obj_arg ValuesObj, uint32_t rows, uint32_t cols,
    b_lean_obj_arg IdxObj, uint32_t k)
UNAVAILABLE(uint32_t, torchlean_cuda_buffer_size, b_lean_obj_arg BObj)
UNAVAILABLE(uint8_t, torchlean_cuda_buffer_dtype, b_lean_obj_arg BObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_format, b_lean_obj_arg buffer)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_fail, b_lean_obj_arg message)
UNAVAILABLE(lean_obj_res, torchlean_kernel_run_encoded,
    b_lean_obj_arg source, b_lean_obj_arg inputs, b_lean_obj_arg format,
    uint64_t width, uint64_t count)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_release_then,
    b_lean_obj_arg scratchObj, b_lean_obj_arg keepObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_zeros, uint32_t n, uint8_t dtype)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_zeros_like, b_lean_obj_arg reference, uint32_t n)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_copy, b_lean_obj_arg reference)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_duplicate, b_lean_obj_arg reference)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_full, uint32_t n, double v, uint8_t dtype)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_of_float_array, b_lean_obj_arg AObj, uint8_t dtype)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_to_float_array, b_lean_obj_arg BObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_abs_bwd, b_lean_obj_arg XObj, b_lean_obj_arg GObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_sqrt_bwd, b_lean_obj_arg XObj, b_lean_obj_arg GObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_clamp, b_lean_obj_arg BObj, double lo, double hi)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_clamp_bwd,
    b_lean_obj_arg XObj, b_lean_obj_arg GObj, double lo, double hi)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_dropout,
    b_lean_obj_arg XObj, b_lean_obj_arg UObj, double p)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_max_bwd,
    b_lean_obj_arg AObj, b_lean_obj_arg BObj, b_lean_obj_arg GObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_min_bwd,
    b_lean_obj_arg AObj, b_lean_obj_arg BObj, b_lean_obj_arg GObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_copy_and_release, b_lean_obj_arg BObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_adam_step,
    b_lean_obj_arg ParametersObj, b_lean_obj_arg GradientObj, b_lean_obj_arg FirstMomentObj,
    b_lean_obj_arg SecondMomentObj, double beta1, double oneMinusBeta1, double beta2,
    double oneMinusBeta2, double firstMomentDenominator, double secondMomentDenominator,
    double epsilon, b_lean_obj_arg DecayObj, double updateScale)
// The same list generates the signatures for both backend configurations.
#define TORCHLEAN_UNARY_EXPORT(NAME, EXPRESSION) \
  UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_##NAME, b_lean_obj_arg x)
#define TORCHLEAN_BINARY_EXPORT(NAME, EXPRESSION) \
  UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_##NAME, b_lean_obj_arg a, b_lean_obj_arg b)
#define TORCHLEAN_UNARY_SCALAR_EXPORT(NAME, EXPRESSION) \
  UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_##NAME, b_lean_obj_arg x, double scalar)
#define TORCHLEAN_BINARY_SCALAR_EXPORT(NAME, EXPRESSION) \
  UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_##NAME, \
      b_lean_obj_arg a, b_lean_obj_arg b, double scalar)
#define TORCHLEAN_VJP_EXPORT(NAME, EXPRESSION) \
  TORCHLEAN_BINARY_EXPORT(NAME##_bwd, EXPRESSION)

#include "operations.h"

#undef TORCHLEAN_VJP_EXPORT
#undef TORCHLEAN_BINARY_SCALAR_EXPORT
#undef TORCHLEAN_UNARY_SCALAR_EXPORT
#undef TORCHLEAN_BINARY_EXPORT
#undef TORCHLEAN_UNARY_EXPORT
#undef UNAVAILABLE

#include <lean/lean.h>
#include <lean/mimalloc.h>

#include <stdint.h>

// Backend linked when TorchLean is built without LibTorch (the default `lake build`).
//
// Every Lean extern of the GPU buffer ABI resolves here so the default build links, but no buffer
// can ever be created: `torchlean_cuda_runtime_status` reports 0, CUDA sessions are rejected
// before they start, IO constructors return an error, and the pure operations panic. Build with
// `-K cuda=true` to link `csrc/libtorch` instead.

#define TORCHLEAN_UNAVAILABLE_MESSAGE \
  "TorchLean was built without LibTorch; rebuild with `lake -R -K cuda=true build`"

static lean_obj_res unavailable_io(void) {
  return lean_io_result_mk_error(
      lean_mk_io_user_error(lean_mk_string(TORCHLEAN_UNAVAILABLE_MESSAGE)));
}

// 0 = built without LibTorch, 1 = LibTorch with a visible device, 2 = LibTorch without one.
LEAN_EXPORT uint32_t torchlean_cuda_runtime_status(uint32_t token) {
  (void)token;
  return 0u;
}

LEAN_EXPORT lean_obj_res torchlean_libtorch_version(uint32_t token) {
  (void)token;
  return lean_mk_string("unavailable");
}

// Returning unused Lean heap pages to the OS is host work and does not need LibTorch.
LEAN_EXPORT uint32_t torchlean_runtime_collect_allocator(uint32_t token) {
  (void)token;
  mi_collect(false);
  return 1u;
}

// Settings read as off and memory statistics read as zero, so reports work in either build.
LEAN_EXPORT lean_obj_res torchlean_libtorch_get_setting(uint32_t setting) {
  (void)setting;
  return lean_io_result_mk_ok(lean_box_uint32(0u));
}

LEAN_EXPORT lean_obj_res torchlean_libtorch_get_memory_fraction(uint32_t token) {
  (void)token;
  return lean_io_result_mk_ok(lean_box_float(0.0));
}

#define ZERO(TYPE, NAME) \
  LEAN_EXPORT TYPE NAME(uint32_t token) { (void)token; return 0u; }

ZERO(uint32_t, torchlean_libtorch_device_count)
ZERO(uint32_t, torchlean_libtorch_get_device)
ZERO(uint64_t, torchlean_libtorch_allocated_bytes)
ZERO(uint64_t, torchlean_libtorch_reserved_bytes)
ZERO(uint64_t, torchlean_libtorch_peak_allocated_bytes)
ZERO(uint64_t, torchlean_libtorch_peak_reserved_bytes)
ZERO(uint64_t, torchlean_cuda_allocator_live_bytes)
ZERO(uint64_t, torchlean_cuda_allocator_peak_bytes)
ZERO(uint64_t, torchlean_cuda_allocator_alloc_count)
ZERO(uint64_t, torchlean_cuda_allocator_free_count)
ZERO(uint64_t, torchlean_cuda_wrapper_live_count)
ZERO(uint64_t, torchlean_cuda_wrapper_peak_count)
ZERO(uint64_t, torchlean_cuda_wrapper_alloc_count)
ZERO(uint64_t, torchlean_cuda_wrapper_finalize_count)
ZERO(uint64_t, torchlean_cuda_allocator_device_free_bytes)
ZERO(uint64_t, torchlean_cuda_allocator_device_total_bytes)
#undef ZERO

#define UNAVAILABLE_IO(NAME, ...) \
  LEAN_EXPORT lean_obj_res NAME(__VA_ARGS__) { return unavailable_io(); }

UNAVAILABLE_IO(torchlean_libtorch_set_device, uint32_t index)
UNAVAILABLE_IO(torchlean_libtorch_set_setting, uint32_t setting, uint32_t enabled)
UNAVAILABLE_IO(torchlean_libtorch_set_memory_fraction, double fraction)
UNAVAILABLE_IO(torchlean_libtorch_synchronize, uint32_t token)
UNAVAILABLE_IO(torchlean_libtorch_empty_cache, uint32_t token)
UNAVAILABLE_IO(torchlean_cuda_buffer_zeros_io, uint32_t n)
UNAVAILABLE_IO(torchlean_cuda_buffer_full_io, uint32_t n, double v)
UNAVAILABLE_IO(torchlean_cuda_buffer_rand_uniform_io, uint32_t n, uint64_t key)
UNAVAILABLE_IO(torchlean_cuda_buffer_bernoulli_mask_io, uint32_t n, double keepProb, uint64_t key)
UNAVAILABLE_IO(torchlean_cuda_buffer_of_float_array_io, b_lean_obj_arg AObj)
UNAVAILABLE_IO(torchlean_cuda_buffer_to_float_array_io, b_lean_obj_arg BObj)
UNAVAILABLE_IO(torchlean_cuda_buffer_to_float32_bytes_io, b_lean_obj_arg BObj)
UNAVAILABLE_IO(torchlean_cuda_buffer_of_float32_bytes_io, b_lean_obj_arg BytesObj)
#undef UNAVAILABLE_IO

// Reached only through a buffer, and no buffer exists in this build.
#define UNAVAILABLE(TYPE, NAME, ...) \
  LEAN_EXPORT TYPE NAME(__VA_ARGS__) { lean_internal_panic(TORCHLEAN_UNAVAILABLE_MESSAGE); }

UNAVAILABLE(lean_obj_res, torchlean_dgemm_cuda,
    b_lean_obj_arg AObj, b_lean_obj_arg BObj, uint32_t m, uint32_t n, uint32_t p)
UNAVAILABLE(lean_obj_res, torchlean_cuda_conv_fwd,
    b_lean_obj_arg inputObj, b_lean_obj_arg kernelObj, b_lean_obj_arg biasObj,
    b_lean_obj_arg inSpatialObj, b_lean_obj_arg kernelSpatialObj, b_lean_obj_arg strideObj,
    b_lean_obj_arg paddingObj, uint32_t inC, uint32_t outC)
UNAVAILABLE(lean_obj_res, torchlean_cuda_conv_bwd,
    b_lean_obj_arg inputObj, b_lean_obj_arg kernelObj, b_lean_obj_arg gradObj,
    b_lean_obj_arg inSpatialObj, b_lean_obj_arg kernelSpatialObj, b_lean_obj_arg strideObj,
    b_lean_obj_arg paddingObj, uint32_t inC, uint32_t outC)
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
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_sigmoid, b_lean_obj_arg BObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_tanh, b_lean_obj_arg BObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_reduce_sum_by_row,
    b_lean_obj_arg BObj, uint32_t rows, uint32_t cols)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_reduce_max_by_column,
    b_lean_obj_arg BObj, uint32_t rows, uint32_t cols)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_reduce_max_by_row,
    b_lean_obj_arg BObj, uint32_t rows, uint32_t cols)
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
    uint32_t cols, double invColsArg, double epsilonArg)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_layer_norm_bwd,
    b_lean_obj_arg DOutObj, b_lean_obj_arg NormalizedObj, b_lean_obj_arg InvStdObj,
    b_lean_obj_arg GammaObj, uint32_t rows, uint32_t cols, double colsScaleArg, double invColsArg)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_bmm_with_transpose,
    b_lean_obj_arg AObj, b_lean_obj_arg BObj, uint32_t batch, uint32_t m, uint32_t n, uint32_t p,
    uint32_t transposeA, uint32_t transposeB)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_rfft1d_packed,
    b_lean_obj_arg XObj, uint32_t batch, uint32_t n)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_irfft1d_packed,
    b_lean_obj_arg SpecObj, uint32_t batch, uint32_t n)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_spectral_conv1d_rfft_fwd,
    b_lean_obj_arg XObj, b_lean_obj_arg WReObj, b_lean_obj_arg WImObj, uint32_t grid,
    uint32_t width, uint32_t modes)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_spectral_conv1d_rfft_bwd,
    b_lean_obj_arg XObj, b_lean_obj_arg WReObj, b_lean_obj_arg WImObj, b_lean_obj_arg DYObj,
    uint32_t grid, uint32_t width, uint32_t modes)
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
UNAVAILABLE(lean_obj_res, torchlean_libtorch_attention_fwd,
    b_lean_obj_arg QObj, b_lean_obj_arg KObj, b_lean_obj_arg VObj, b_lean_obj_arg MaskObj,
    uint32_t hasMask, uint32_t batch, uint32_t n, uint32_t d, double scaleHost)
UNAVAILABLE(lean_obj_res, torchlean_libtorch_attention_bwd,
    b_lean_obj_arg OutObj, b_lean_obj_arg DOutObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_scatter_add,
    b_lean_obj_arg XObj, b_lean_obj_arg ValuesObj, uint32_t n, b_lean_obj_arg IdxObj, uint32_t k)
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
UNAVAILABLE(uint32_t, torchlean_cuda_buffer_size_with_token, b_lean_obj_arg BObj, uint32_t token)
UNAVAILABLE(uint32_t, torchlean_cuda_buffer_release_with_token, b_lean_obj_arg BObj, uint32_t token)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_release_then,
    b_lean_obj_arg scratchObj, b_lean_obj_arg keepObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_zeros, uint32_t n)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_full, uint32_t n, double v)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_rand_uniform, uint32_t n, uint64_t key)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_rand_normal,
    uint32_t n, double mean, double std, uint64_t key)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_bernoulli_mask,
    uint32_t n, double keepProb, uint64_t key)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_of_float_array, b_lean_obj_arg AObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_to_float_array, b_lean_obj_arg BObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_abs, b_lean_obj_arg BObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_abs_bwd, b_lean_obj_arg XObj, b_lean_obj_arg GObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_sqrt, b_lean_obj_arg BObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_sqrt_bwd, b_lean_obj_arg XObj, b_lean_obj_arg GObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_exp, b_lean_obj_arg BObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_sin, b_lean_obj_arg BObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_cos, b_lean_obj_arg BObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_log, b_lean_obj_arg BObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_inv, b_lean_obj_arg BObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_clamp, b_lean_obj_arg BObj, double lo, double hi)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_clamp_bwd,
    b_lean_obj_arg XObj, b_lean_obj_arg GObj, double lo, double hi)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_max, b_lean_obj_arg AObj, b_lean_obj_arg BObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_max_bwd,
    b_lean_obj_arg AObj, b_lean_obj_arg BObj, b_lean_obj_arg GObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_min, b_lean_obj_arg AObj, b_lean_obj_arg BObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_min_bwd,
    b_lean_obj_arg AObj, b_lean_obj_arg BObj, b_lean_obj_arg GObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_div, b_lean_obj_arg AObj, b_lean_obj_arg BObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_relu, b_lean_obj_arg BObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_relu_bwd, b_lean_obj_arg XObj, b_lean_obj_arg GObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_gelu, b_lean_obj_arg XObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_gelu_bwd, b_lean_obj_arg XObj, b_lean_obj_arg GObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_add, b_lean_obj_arg AObj, b_lean_obj_arg BObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_sub, b_lean_obj_arg AObj, b_lean_obj_arg BObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_mul, b_lean_obj_arg AObj, b_lean_obj_arg BObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_scale, b_lean_obj_arg BObj, double c)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_copy_and_release, b_lean_obj_arg BObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_axpy,
    b_lean_obj_arg AObj, b_lean_obj_arg BObj, double c)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_adam_step,
    b_lean_obj_arg ParametersObj, b_lean_obj_arg GradientObj, b_lean_obj_arg FirstMomentObj,
    b_lean_obj_arg SecondMomentObj, double beta1, double oneMinusBeta1, double beta2,
    double oneMinusBeta2, double firstMomentCorrection, double secondMomentCorrection,
    double epsilon, double decay, double updateScale)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_scaled_prod_exp,
    b_lean_obj_arg AObj, b_lean_obj_arg BObj, double c)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_reduce_sum, b_lean_obj_arg BObj)
UNAVAILABLE(lean_obj_res, torchlean_cuda_buffer_reduce_mean, b_lean_obj_arg BObj)
#undef UNAVAILABLE

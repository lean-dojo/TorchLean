// Paired ATen attention operators. TorchLean owns the only differentiation tape.
//
// Operator schemas checked against PyTorch 2.12.0a0 (0291f960b6) and pip torch 2.13.0+cu130.
// Upstream preprocessing and derivative references:
//   aten/src/ATen/native/native_functions.yaml
//   aten/src/ATen/native/transformers/attention.cpp
//   aten/src/ATen/native/transformers/cuda/attention_backward.cu
#include "torchlean_libtorch.h"

#include <ATen/Context.h>
#include <ATen/SDPBackend.h>
#include <ATen/native/transformers/cuda/sdp_utils.h>
#include <ATen/ops/_fused_sdp_choice.h>
#include <ATen/ops/_scaled_dot_product_attention_math.h>
#include <ATen/ops/_softmax_backward_data.h>

#if __has_include(<ATen/ops/_scaled_dot_product_flash_attention.h>) && \
    __has_include(<ATen/ops/_scaled_dot_product_flash_attention_backward.h>)
#include <ATen/ops/_scaled_dot_product_flash_attention.h>
#include <ATen/ops/_scaled_dot_product_flash_attention_backward.h>
#define TORCHLEAN_HAS_FLASH_ATTENTION_PAIR 1
#endif

#if __has_include(<ATen/ops/_scaled_dot_product_efficient_attention.h>) && \
    __has_include(<ATen/ops/_scaled_dot_product_efficient_attention_backward.h>)
#include <ATen/ops/_scaled_dot_product_efficient_attention.h>
#include <ATen/ops/_scaled_dot_product_efficient_attention_backward.h>
#define TORCHLEAN_HAS_EFFICIENT_ATTENTION_PAIR 1
#endif

#if __has_include(<ATen/ops/_scaled_dot_product_cudnn_attention.h>) && \
    __has_include(<ATen/ops/_scaled_dot_product_cudnn_attention_backward.h>)
#include <ATen/ops/_scaled_dot_product_cudnn_attention.h>
#include <ATen/ops/_scaled_dot_product_cudnn_attention_backward.h>
#define TORCHLEAN_HAS_CUDNN_ATTENTION_PAIR 1
#endif

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>

namespace {

using Gradients = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

struct AttentionContext {
  at::SDPBackend backend = at::SDPBackend::math;
  at::Tensor query;
  at::Tensor key;
  at::Tensor value;
  at::Tensor output;
  at::Tensor bias;
  at::Tensor blocked_rows;
  at::Tensor logsumexp;
  at::Tensor cum_seq_q;
  at::Tensor cum_seq_k;
  at::Tensor rng_state;
  at::Tensor rng_offset;
  at::Tensor probabilities;
  int64_t max_q = 0;
  int64_t max_k = 0;
  int64_t head_dim = 0;
  double scale = 1.0;
  bool empty = false;
  bool deterministic = false;
  bool deterministic_warn_only = false;
};

// A deleter also identifies the erased context without reading another operator's state.
struct DeleteAttentionContext {
  void operator()(AttentionContext* context) const { delete context; }
};

size_t attention_elements(uint32_t batch, uint32_t rows, uint32_t cols) {
  uint64_t count = batch;
  for (const uint32_t factor : {rows, cols}) {
    TORCH_CHECK(factor == 0 || count <= uint64_t(INT64_MAX) / factor,
                "LibTorch attention: shape exceeds ATen's signed element count");
    count *= factor;
  }
  TORCH_CHECK(count <= SIZE_MAX / sizeof(float),
              "LibTorch attention: buffer byte size overflow");
  return static_cast<size_t>(count);
}

void check_buffer(b_lean_obj_arg object, size_t elements, const char* name) {
  const auto* buffer = torchlean_cuda_buffer_unbox(object);
  TORCH_CHECK(buffer->size == elements,
              "LibTorch attention: ", name, " buffer size mismatch (expected ",
              elements, ", got ", buffer->size, ")");
  TORCH_CHECK(buffer->tensor.defined(), "LibTorch attention: ", name, " was released");
  TORCH_CHECK(buffer->tensor.numel() == static_cast<int64_t>(elements),
              "LibTorch attention: ", name, " tensor size disagrees with its buffer");
  TORCH_CHECK(buffer->tensor.is_cuda() && buffer->tensor.scalar_type() == at::kFloat,
              "LibTorch attention: ", name, " must be a CUDA float32 tensor");
  TORCH_CHECK(!buffer->tensor.requires_grad(),
              "LibTorch attention: ", name, " must not be an autograd tensor");
}

std::optional<at::Tensor> optional_bias(const AttentionContext& context) {
  if (context.bias.defined()) return context.bias;
  return std::nullopt;
}

at::Tensor pad_head(const at::Tensor& value) {
  const int64_t padding = (8 - value.size(-1) % 8) % 8;
  return padding == 0 ? value : at::constant_pad_nd(value, {0, padding}, 0.0);
}

void use_math(AttentionContext& context) {
  TORCH_CHECK(at::globalContext().userEnabledMathSDP(),
              "LibTorch attention: this request needs the math SDPA pair, "
              "but the ATen math backend is disabled");
  context.backend = at::SDPBackend::math;
}

bool has_pair(at::SDPBackend backend) {
  switch (backend) {
    case at::SDPBackend::math:
      return true;
#ifdef TORCHLEAN_HAS_FLASH_ATTENTION_PAIR
    case at::SDPBackend::flash_attention:
      return true;
#endif
#ifdef TORCHLEAN_HAS_EFFICIENT_ATTENTION_PAIR
    case at::SDPBackend::efficient_attention:
      return true;
#endif
#ifdef TORCHLEAN_HAS_CUDNN_ATTENTION_PAIR
    case at::SDPBackend::cudnn_attention:
      return true;
#endif
    default:
      return false;
  }
}

void require_eligible_provider(
    const AttentionContext& context, const std::optional<at::Tensor>& allowed) {
  auto& settings = at::globalContext();
  if (settings.userEnabledMathSDP()) return;
  const sdp::sdp_params params{
      context.query, context.key, context.value, allowed, 0.0, false, false};
  const bool flash = settings.userEnabledFlashSDP() &&
      sdp::can_use_flash_attention(params, false);
  const bool efficient = settings.userEnabledMemEfficientSDP() &&
      sdp::can_use_mem_efficient_attention(params, false);
  const bool cudnn = settings.userEnabledCuDNNSDP() &&
      sdp::can_use_cudnn_attention(params, false);
  // The SDK selector's enabled overrideable entry raises "Invalid backend" after
  // these three predicates fail. Establish ineligibility without changing global
  // settings; eligible requests still use ATen's configured priority order.
  TORCH_CHECK(flash || efficient || cudnn,
              "LibTorch attention: No available kernel. ATen eligibility checks "
              "found no enabled provider for CUDA float32 inputs (math=disabled, flash=",
              settings.userEnabledFlashSDP(), ", efficient=",
              settings.userEnabledMemEfficientSDP(), ", cuDNN=",
              settings.userEnabledCuDNNSDP(), ", hasMask=", allowed.has_value(),
              "); each enabled fused provider failed ATen eligibility checks");
}

void run_forward(AttentionContext& context) {
  auto& q = context.query;
  auto& k = context.key;
  auto& v = context.value;
  switch (context.backend) {
#ifdef TORCHLEAN_HAS_FLASH_ATTENTION_PAIR
    case at::SDPBackend::flash_attention: {
      q = pad_head(q);
      k = pad_head(k);
      v = pad_head(v);
      auto result = at::_scaled_dot_product_flash_attention(
          q, k, v, 0.0, false, false, context.scale);
      context.output = std::get<0>(result);
      context.logsumexp = std::get<1>(result);
      context.cum_seq_q = std::get<2>(result);
      context.cum_seq_k = std::get<3>(result);
      context.max_q = std::get<4>(result).expect_int();
      context.max_k = std::get<5>(result).expect_int();
      context.rng_state = std::get<6>(result);
      context.rng_offset = std::get<7>(result);
      return;
    }
#endif
#ifdef TORCHLEAN_HAS_EFFICIENT_ATTENTION_PAIR
    case at::SDPBackend::efficient_attention: {
      // Upstream's efficient pair requires aligned bias strides, including in backward.
      if (context.bias.defined()) {
        const int64_t width = context.bias.size(-1);
        const int64_t padding = (8 - width % 8) % 8;
        if (padding != 0) {
          context.bias = at::constant_pad_nd(context.bias, {0, padding}, 0.0)
                             .slice(-1, 0, width);
        }
      }
      auto result = at::_scaled_dot_product_efficient_attention(
          q, k, v, optional_bias(context), true, 0.0, false, context.scale);
      context.output = std::get<0>(result);
      context.logsumexp = std::get<1>(result);
      context.rng_state = std::get<2>(result);
      context.rng_offset = std::get<3>(result);
      return;
    }
#endif
#ifdef TORCHLEAN_HAS_CUDNN_ATTENTION_PAIR
    case at::SDPBackend::cudnn_attention: {
      auto result = at::_scaled_dot_product_cudnn_attention(
          q, k, v, optional_bias(context), true, 0.0, false, false, context.scale);
      context.output = std::get<0>(result);
      context.logsumexp = std::get<1>(result);
      context.cum_seq_q = std::get<2>(result);
      context.cum_seq_k = std::get<3>(result);
      context.max_q = std::get<4>(result).expect_int();
      context.max_k = std::get<5>(result).expect_int();
      context.rng_state = std::get<6>(result);
      context.rng_offset = std::get<7>(result);
      return;
    }
#endif
    case at::SDPBackend::math: {
      auto result = at::_scaled_dot_product_attention_math(
          q, k, v, optional_bias(context), 0.0, false, std::nullopt, context.scale, false);
      context.output = std::get<0>(result);
      context.probabilities = std::get<1>(result);
      return;
    }
    default:
      TORCH_CHECK(false, "LibTorch attention: unsupported saved SDPA provider");
  }
}

lean_obj_res attention_forward(
    b_lean_obj_arg query, b_lean_obj_arg key, b_lean_obj_arg value, b_lean_obj_arg mask,
    uint32_t has_mask, uint32_t batch, uint32_t n, uint32_t d, double scale) {
  TORCH_CHECK(has_mask <= 1, "LibTorch attention: hasMask must be 0 or 1");
  TORCH_CHECK(std::isfinite(scale), "LibTorch attention: scale must be finite");
  TORCH_CHECK(std::abs(scale) <= std::numeric_limits<float>::max(),
              "LibTorch attention: scale exceeds the float32 range");
  const size_t elements = attention_elements(batch, n, d);
  check_buffer(query, elements, "Q");
  check_buffer(key, elements, "K");
  check_buffer(value, elements, "V");
  if (has_mask) check_buffer(mask, attention_elements(batch, n, n), "mask");

  auto context = std::shared_ptr<AttentionContext>(
      new AttentionContext(), DeleteAttentionContext{});
  // The ABI folds (sample, head) into batch. SDPA expects four dimensions.
  const std::array<int64_t, 4> shape{batch, 1, n, d};
  context->query = torchlean::shaped(query, shape);
  context->key = torchlean::shaped(key, shape);
  context->value = torchlean::shaped(value, shape);
  TORCH_CHECK(context->query.device() == context->key.device() &&
                  context->query.device() == context->value.device(),
              "LibTorch attention: Q/K/V must be on the same CUDA device");
  const c10::DeviceGuard device_guard(context->query.device());
  context->head_dim = d;
  context->scale = scale;
  auto& settings = at::globalContext();
  context->deterministic = settings.deterministicAlgorithms();
  context->deterministic_warn_only = settings.deterministicAlgorithmsWarnOnly();
  const bool require_deterministic = context->deterministic;
  TORCH_CHECK(!require_deterministic || !context->deterministic_warn_only,
              "LibTorch attention: deterministic reductions require strict ATen determinism");

  std::optional<at::Tensor> allowed;
  if (has_mask) {
    TORCH_CHECK(torchlean::tensor(mask).device() == context->query.device(),
                "LibTorch attention: mask must be on the Q/K/V device");
    allowed = torchlean::shaped(mask, {batch, 1, n, n}).to(at::kBool);
  }
  context->empty = elements == 0;
  if (context->empty) {
    context->output = at::zeros_like(context->query);
  } else {
    require_eligible_provider(*context, allowed);
    context->backend = static_cast<at::SDPBackend>(at::_fused_sdp_choice(
        context->query, context->key, context->value, allowed, 0.0, false, scale, false));
    if (!has_pair(context->backend)) use_math(*context);
    // Explicit hard masks use the math or efficient pair with an exact -infinity bias.
    if (allowed.has_value() &&
        (context->backend == at::SDPBackend::cudnn_attention ||
         context->backend == at::SDPBackend::flash_attention)) {
      use_math(*context);
    }
    // The no-grad selector does not check flash's training-only head-dimension restriction.
    // This conservative guard also covers SDKs using the same paired CUDA operators.
    if (context->backend == at::SDPBackend::flash_attention && d > 192 && d <= 224) {
      use_math(*context);
    }
    if (context->backend == at::SDPBackend::cudnn_attention &&
        context->deterministic) {
      use_math(*context);
    }
    if (allowed.has_value()) {
      const auto blocked = allowed->logical_not();
      context->blocked_rows = blocked.all(-1, true);
      context->bias = at::zeros(allowed->sizes(), context->query.options())
                          .masked_fill(blocked, -std::numeric_limits<float>::infinity());
      // Give fully blocked rows a finite internal softmax. Their output and incoming
      // cotangent are masked to zero, so they contribute nothing to Q/K/V gradients.
      context->bias.masked_fill_(context->blocked_rows, 0.0);
      context->query = context->query.masked_fill(context->blocked_rows, 0.0);
    }
    run_forward(*context);
  }

  auto result = context->output.slice(-1, 0, context->head_dim);
  if (context->blocked_rows.defined()) {
    result = result.masked_fill(context->blocked_rows, 0.0);
  }
  auto* buffer = torchlean::owned(result);
  buffer->context = std::move(context);
  return torchlean_cuda_buffer_box(buffer);
}

Gradients run_backward(const AttentionContext& context, const at::Tensor& grad) {
  const auto& q = context.query;
  const auto& k = context.key;
  const auto& v = context.value;
  if (context.empty) {
    return {at::zeros_like(q), at::zeros_like(k), at::zeros_like(v)};
  }
  switch (context.backend) {
#ifdef TORCHLEAN_HAS_FLASH_ATTENTION_PAIR
    case at::SDPBackend::flash_attention:
      return at::_scaled_dot_product_flash_attention_backward(
          pad_head(grad), q, k, v, context.output, context.logsumexp,
          context.cum_seq_q, context.cum_seq_k, context.max_q, context.max_k,
          0.0, false, context.rng_state, context.rng_offset, context.scale);
#endif
#ifdef TORCHLEAN_HAS_EFFICIENT_ATTENTION_PAIR
    case at::SDPBackend::efficient_attention: {
      auto result = at::_scaled_dot_product_efficient_attention_backward(
          grad, q, k, v, context.bias, context.output, context.logsumexp,
          context.rng_state, context.rng_offset, 0.0, {true, true, true, false},
          false, context.scale);
      return {std::get<0>(result), std::get<1>(result), std::get<2>(result)};
    }
#endif
#ifdef TORCHLEAN_HAS_CUDNN_ATTENTION_PAIR
    case at::SDPBackend::cudnn_attention:
      return at::_scaled_dot_product_cudnn_attention_backward(
          grad, q, k, v, context.output, context.logsumexp, context.rng_state,
          context.rng_offset, context.bias, context.cum_seq_q, context.cum_seq_k,
          context.max_q, context.max_k, 0.0, false, context.scale);
#endif
    case at::SDPBackend::math: {
      // The SDK has no monolithic math SDPA backward. Reuse the forward probabilities
      // with ATen's softmax backward and contractions; never recompute Q Kᵀ or softmax.
      const auto& p = context.probabilities;
      const auto dp = at::matmul(grad, v.transpose(-2, -1));
      const auto ds = at::_softmax_backward_data(dp, p, -1, p.scalar_type());
      // Match upstream's split scaling, including negative and zero scale.
      const double magnitude = std::sqrt(std::abs(context.scale));
      const double query_scale = context.scale < 0.0 ? -magnitude : magnitude;
      auto dq = at::matmul(ds, k * magnitude) * query_scale;
      auto dk = at::matmul(ds.transpose(-2, -1), q * query_scale) * magnitude;
      auto dv = at::matmul(p.transpose(-2, -1), grad);
      return {std::move(dq), std::move(dk), std::move(dv)};
    }
    default:
      TORCH_CHECK(false, "LibTorch attention: unsupported saved SDPA provider");
  }
}

lean_obj_res attention_backward(b_lean_obj_arg output, b_lean_obj_arg grad_output) {
  const auto* buffer = torchlean_cuda_buffer_unbox(output);
  TORCH_CHECK(std::get_deleter<DeleteAttentionContext>(buffer->context) != nullptr,
              "LibTorch attention: backward requires the original, unreleased forward buffer");
  const auto context = std::static_pointer_cast<AttentionContext>(buffer->context);
  const auto& settings = at::globalContext();
  TORCH_CHECK(context->deterministic == settings.deterministicAlgorithms() &&
                  context->deterministic_warn_only == settings.deterministicAlgorithmsWarnOnly(),
              "LibTorch attention: deterministic policy changed between forward and backward");
  const size_t elements = static_cast<size_t>(
      context->query.size(0) * context->query.size(2) * context->head_dim);
  check_buffer(output, elements, "forward output");
  check_buffer(grad_output, elements, "dOut");
  TORCH_CHECK(torchlean::tensor(grad_output).device() == context->query.device(),
              "LibTorch attention: dOut must be on the forward device");
  const c10::DeviceGuard device_guard(context->query.device());
  auto grad = torchlean::shaped(
      grad_output, {context->query.size(0), 1, context->query.size(2), context->head_dim});
  if (context->blocked_rows.defined()) {
    grad = grad.masked_fill(context->blocked_rows, 0.0);
  }
  auto [dq, dk, dv] = run_backward(*context, grad);
  if (context->blocked_rows.defined()) {
    dq = dq.masked_fill(context->blocked_rows, 0.0);
  }
  return torchlean::triple(dq.slice(-1, 0, context->head_dim),
                           dk.slice(-1, 0, context->head_dim),
                           dv.slice(-1, 0, context->head_dim));
}

// Both tape execution and effectful callers receive the same checked result.
template <typename F>
lean_obj_res attention_result(F&& body) {
  try {
    torchlean::initialize();
    const at::NoGradGuard no_grad;
    const c10::DeviceGuard device_guard(torchlean::device());
    lean_object* value = std::forward<F>(body)();
    lean_object* result = lean_alloc_ctor(1, 1, 0);
    lean_ctor_set(result, 0, value);
    return result;
  } catch (const std::exception& error) {
    lean_object* result = lean_alloc_ctor(0, 1, 0);
    lean_ctor_set(result, 0, lean_mk_string(error.what()));
    return result;
  }
}

}  // namespace

extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_attention_fwd(
    b_lean_obj_arg query, b_lean_obj_arg key, b_lean_obj_arg value, b_lean_obj_arg mask,
    uint32_t has_mask, uint32_t batch, uint32_t n, uint32_t d, double scale) {
  return attention_result([&] {
    return attention_forward(query, key, value, mask, has_mask, batch, n, d, scale);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_attention_bwd(
    b_lean_obj_arg output, b_lean_obj_arg grad_output) {
  return attention_result([&] { return attention_backward(output, grad_output); });
}

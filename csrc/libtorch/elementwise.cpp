#include "torchlean_libtorch.h"

#include <ATen/Context.h>

#include <algorithm>
#include <limits>

// Lifetime and telemetry remain in runtime.cpp, including explicit source retirement.
extern "C" LEAN_EXPORT uint32_t torchlean_cuda_buffer_release(b_lean_obj_arg object);

namespace {

using torchlean::box;
using torchlean::invoke;
using torchlean::pair;
using torchlean::require;
using torchlean::tensor;
using torchlean::triple;

at::Tensor flat(b_lean_obj_arg object) {
  return tensor(object).reshape({-1});
}

void same_size(const at::Tensor& a, const at::Tensor& b, const char* message) {
  require(a.numel() == b.numel(), message);
}

at::Tensor selected_sqrt(const at::Tensor& x) {
  // Ordered comparison preserves NaN and selects positive zero for both signed zeros.
  return at::sqrt(at::where(at::le(x, 0.0f), 0.0f, x));
}

at::Tensor selected_relu(const at::Tensor& x) {
  // TorchLean selects zero for NaN as well as for nonpositive inputs.
  return at::where(at::gt(x, 0.0f), x, 0.0f);
}

at::Tensor axpy(const at::Tensor& a, const at::Tensor& b, float c) {
  // PyTorch 0291f960b6: CUDA DeviceAddCmulCdiv.cuh explicitly calls
  // std::fma(tensor1, tensor2, input) when addcmul's value is exactly one.
  // Place c in tensor2 (the supported CPU-scalar operand), NOT in value:
  // value != 1 first rounds tensor1*tensor2 before the final FMA.
  // This avoids depending on implicit contraction in the add(alpha) ufunc.
  const auto coefficient = at::scalar_tensor(
      c, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  return at::addcmul(a, b, coefficient, 1.0f);
}

constexpr float kGeluCoeff = 0.044715f;
constexpr float kSqrtTwoOverPi = 0.79788456080286535588f;

at::Tensor gelu_tanh_term(const at::Tensor& x) {
  // Each ATen call materializes one specified rounded float32 stage.
  const auto cubic0 = at::mul(x, kGeluCoeff);
  const auto cubic1 = at::mul(cubic0, x);
  const auto cubic = at::mul(cubic1, x);
  const auto inner = at::add(x, cubic);
  return at::tanh(at::mul(inner, kSqrtTwoOverPi));
}

at::Tensor staged_gelu(const at::Tensor& x) {
  const auto tanh_term = gelu_tanh_term(x);
  const auto scaled_input = at::mul(x, at::add(tanh_term, 1.0f));
  // Multiplication by the exactly representable 1/2 has the same rounding as /2.
  return at::mul(scaled_input, 0.5f);
}

at::Tensor staged_gelu_backward(const at::Tensor& x, const at::Tensor& g) {
  const auto tanh_term = gelu_tanh_term(x);
  const auto sech_term = at::sub(at::ones_like(x), at::mul(tanh_term, tanh_term));
  constexpr float scaled_coeff = 3.0f * kGeluCoeff;
  const auto quadratic0 = at::mul(x, scaled_coeff);
  const auto quadratic = at::mul(quadratic0, x);
  const auto inner_deriv = at::mul(at::add(quadratic, 1.0f), kSqrtTwoOverPi);
  const auto derivative_term = at::mul(at::mul(x, sech_term), inner_deriv);
  const auto numerator = at::add(at::add(tanh_term, 1.0f), derivative_term);
  return at::mul(at::mul(numerator, 0.5f), g);
}

lean_obj_res minmax_backward(
    const at::Tensor& a, const at::Tensor& b, const at::Tensor& g, bool maximum) {
  const auto a_wins = maximum ? at::gt(a, b) : at::gt(b, a);
  const auto b_wins = maximum ? at::gt(b, a) : at::gt(a, b);
  const auto half = at::mul(g, 0.5f);
  // Unordered comparisons follow the tie branch too. Selecting zero, rather
  // than multiplying by a zero mask, also matters when the cotangent is NaN/Inf.
  auto da = at::where(a_wins, g, at::where(b_wins, 0.0f, half));
  auto db = at::where(b_wins, g, at::where(a_wins, 0.0f, half));
  return pair(std::move(da), std::move(db));
}

at::Tensor deterministic_sum(const at::Tensor& input) {
  // Preserve the specified 256-lane tree and capped grid-stride accumulation
  // order using upstream tensor operations. A different reduction tree can
  // change finite results even when it is itself deterministic.
  constexpr int64_t block_size = 256;
  constexpr int64_t max_blocks = 65535;
  if (input.numel() == 0) {
    return at::zeros({1}, input.options());
  }
  auto current = input;
  do {
    const int64_t n = current.numel();
    const int64_t blocks = std::min((n + block_size - 1) / block_size, max_blocks);
    const int64_t stride = blocks * block_size;
    auto lanes = at::zeros({stride}, input.options());
    for (int64_t offset = 0; offset < n; offset += stride) {
      const int64_t count = std::min(stride, n - offset);
      // Only participating lanes add an input; inactive lanes keep their bits.
      // Each lane's first addition starts at +0.
      lanes.narrow(0, 0, count).add_(current.narrow(0, offset, count));
    }
    auto tree = lanes.reshape({blocks, block_size});
    for (int64_t width = block_size / 2; width != 0; width /= 2) {
      tree = at::add(tree.narrow(1, 0, width), tree.narrow(1, width, width));
    }
    current = tree.reshape({blocks});
  } while (current.numel() > 1);
  return current;
}

at::Tensor reduce_sum(const at::Tensor& x) {
  if (at::globalContext().deterministicAlgorithms()) {
    return deterministic_sum(x);
  }
  return at::sum(x).reshape({1});
}

}  // namespace

#define TORCHLEAN_UNARY_EXPORT(NAME, EXPRESSION)                                \
  extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_##NAME(             \
      b_lean_obj_arg BObj) {                                                   \
    return invoke([&]() {                                                     \
      const auto x = flat(BObj);                                              \
      return box(EXPRESSION);                                                 \
    });                                                                       \
  }

#define TORCHLEAN_BINARY_EXPORT(NAME, EXPRESSION)                               \
  extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_##NAME(             \
      b_lean_obj_arg AObj, b_lean_obj_arg BObj) {                               \
    return invoke([&]() {                                                     \
      const auto a = flat(AObj);                                              \
      const auto b = flat(BObj);                                              \
      same_size(a, b, "torchlean_cuda_buffer_" #NAME ": size mismatch");        \
      return box(EXPRESSION);                                                 \
    });                                                                       \
  }

#define TORCHLEAN_UNARY_SCALAR_EXPORT(NAME, EXPRESSION)                         \
  extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_##NAME(             \
      b_lean_obj_arg BObj, double c) {                                         \
    return invoke([&]() {                                                     \
      const auto x = flat(BObj);                                              \
      const float scalar = static_cast<float>(c);                              \
      return box(EXPRESSION);                                                 \
    });                                                                       \
  }

#define TORCHLEAN_BINARY_SCALAR_EXPORT(NAME, EXPRESSION)                        \
  extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_##NAME(             \
      b_lean_obj_arg AObj, b_lean_obj_arg BObj, double c) {                     \
    return invoke([&]() {                                                     \
      const auto a = flat(AObj);                                              \
      const auto b = flat(BObj);                                              \
      same_size(a, b, "torchlean_cuda_buffer_" #NAME ": size mismatch");        \
      const float scalar = static_cast<float>(c);                              \
      return box(EXPRESSION);                                                 \
    });                                                                       \
  }

#define TORCHLEAN_VJP_EXPORT(NAME, EXPRESSION)                                  \
  extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_##NAME##_bwd(       \
      b_lean_obj_arg XObj, b_lean_obj_arg GObj) {                               \
    return invoke([&]() {                                                     \
      const auto x = flat(XObj);                                              \
      const auto g = flat(GObj);                                              \
      same_size(x, g, "torchlean_cuda_buffer_" #NAME "_bwd: size mismatch");    \
      return box(EXPRESSION);                                                 \
    });                                                                       \
  }

TORCHLEAN_UNARY_EXPORT(abs, at::abs(x))
TORCHLEAN_UNARY_EXPORT(sqrt, selected_sqrt(x))
TORCHLEAN_UNARY_EXPORT(exp, at::exp(x))
TORCHLEAN_UNARY_EXPORT(sin, at::sin(x))
TORCHLEAN_UNARY_EXPORT(cos, at::cos(x))
TORCHLEAN_UNARY_EXPORT(log, at::log(x))
TORCHLEAN_UNARY_EXPORT(inv, at::reciprocal(x))
TORCHLEAN_UNARY_EXPORT(relu, selected_relu(x))
TORCHLEAN_UNARY_EXPORT(sigmoid, at::sigmoid(x))
TORCHLEAN_UNARY_EXPORT(tanh, at::tanh(x))
TORCHLEAN_UNARY_EXPORT(gelu, staged_gelu(x))

TORCHLEAN_BINARY_EXPORT(max, at::fmax(a, b))
TORCHLEAN_BINARY_EXPORT(min, at::fmin(a, b))
TORCHLEAN_BINARY_EXPORT(div, at::div(a, b))
TORCHLEAN_BINARY_EXPORT(add, at::add(a, b))
TORCHLEAN_BINARY_EXPORT(sub, at::sub(a, b))
TORCHLEAN_BINARY_EXPORT(mul, at::mul(a, b))

TORCHLEAN_UNARY_SCALAR_EXPORT(scale, at::mul(x, scalar))
TORCHLEAN_BINARY_SCALAR_EXPORT(axpy, axpy(a, b, scalar))
// Preserve left association and use the same exp operation as the composed path.
TORCHLEAN_BINARY_SCALAR_EXPORT(scaled_prod_exp, at::exp(at::mul(at::mul(a, scalar), b)))

TORCHLEAN_VJP_EXPORT(relu, at::where(at::gt(x, 0.0f), g, 0.0f))
TORCHLEAN_VJP_EXPORT(gelu, staged_gelu_backward(x, g))

#undef TORCHLEAN_VJP_EXPORT
#undef TORCHLEAN_BINARY_SCALAR_EXPORT
#undef TORCHLEAN_UNARY_SCALAR_EXPORT
#undef TORCHLEAN_BINARY_EXPORT
#undef TORCHLEAN_UNARY_EXPORT

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_abs_bwd(
    b_lean_obj_arg XObj, b_lean_obj_arg GObj) {
  return invoke([&]() {
    const auto x = flat(XObj);
    const auto g = flat(GObj);
    same_size(x, g, "torchlean_cuda_buffer_abs_bwd: size mismatch");
    const auto sign = at::where(
        at::gt(x, 0.0f), 1.0f, at::where(at::lt(x, 0.0f), -1.0f, at::zeros_like(x)));
    // abs uses sign(x)*g even at zero/NaN, including 0*Inf and signed-zero results.
    return box(at::mul(sign, g));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_sqrt_bwd(
    b_lean_obj_arg XObj, b_lean_obj_arg GObj) {
  return invoke([&]() {
    const auto x = flat(XObj);
    const auto g = flat(GObj);
    same_size(x, g, "torchlean_cuda_buffer_sqrt_bwd: size mismatch");
    const auto positive = at::gt(x, 0.0f);
    const auto root = at::sqrt(at::where(positive, x, 1.0f));
    const auto factor = at::reciprocal(at::mul(root, 2.0f));
    return box(at::where(positive, at::mul(g, factor), 0.0f));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_clamp(
    b_lean_obj_arg BObj, double lo, double hi) {
  return invoke([&]() {
    const auto x = flat(BObj);
    const auto lower = at::scalar_tensor(static_cast<float>(lo), x.options());
    const auto upper = at::scalar_tensor(static_cast<float>(hi), x.options());
    // fmin/fmax implement the selected NaN behavior, including NaN bounds and lo > hi.
    return box(at::fmin(at::fmax(x, lower), upper));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_clamp_bwd(
    b_lean_obj_arg XObj, b_lean_obj_arg GObj, double lo, double hi) {
  return invoke([&]() {
    const auto x = flat(XObj);
    const auto g = flat(GObj);
    same_size(x, g, "torchlean_cuda_buffer_clamp_bwd: size mismatch");
    const auto interior = at::logical_and(
        at::gt(x, static_cast<float>(lo)), at::lt(x, static_cast<float>(hi)));
    return box(at::where(interior, g, 0.0f));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_max_bwd(
    b_lean_obj_arg AObj, b_lean_obj_arg BObj, b_lean_obj_arg GObj) {
  return invoke([&]() {
    const auto a = flat(AObj);
    const auto b = flat(BObj);
    const auto g = flat(GObj);
    same_size(a, b, "torchlean_cuda_buffer_max_bwd: size mismatch");
    same_size(a, g, "torchlean_cuda_buffer_max_bwd: size mismatch");
    return minmax_backward(a, b, g, true);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_min_bwd(
    b_lean_obj_arg AObj, b_lean_obj_arg BObj, b_lean_obj_arg GObj) {
  return invoke([&]() {
    const auto a = flat(AObj);
    const auto b = flat(BObj);
    const auto g = flat(GObj);
    same_size(a, b, "torchlean_cuda_buffer_min_bwd: size mismatch");
    same_size(a, g, "torchlean_cuda_buffer_min_bwd: size mismatch");
    return minmax_backward(a, b, g, false);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_copy_and_release(
    b_lean_obj_arg BObj) {
  return invoke([&]() {
    // Complete allocation and enqueue the scale-by-one copy before retiring the
    // source through the runtime's allocator and telemetry path.
    auto result = box(at::mul(flat(BObj), 1.0f));
    (void)torchlean_cuda_buffer_release(BObj);
    return result;
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_adam_step(
    b_lean_obj_arg ParametersObj,
    b_lean_obj_arg GradientObj,
    b_lean_obj_arg FirstMomentObj,
    b_lean_obj_arg SecondMomentObj,
    double beta1,
    double oneMinusBeta1,
    double beta2,
    double oneMinusBeta2,
    double firstMomentCorrection,
    double secondMomentCorrection,
    double epsilon,
    double decay,
    double updateScale) {
  return invoke([&]() {
    const auto parameters = flat(ParametersObj);
    const auto gradient = flat(GradientObj);
    const auto first_moment = flat(FirstMomentObj);
    const auto second_moment = flat(SecondMomentObj);
    same_size(parameters, gradient, "torchlean_cuda_buffer_adam_step: size mismatch");
    same_size(parameters, first_moment, "torchlean_cuda_buffer_adam_step: size mismatch");
    same_size(parameters, second_moment, "torchlean_cuda_buffer_adam_step: size mismatch");

    // Keep all nine host-to-float32 conversions, including the independently
    // supplied one-minus-beta values. Do not recompute them from rounded betas.
    const float beta1_f = static_cast<float>(beta1);
    const float one_minus_beta1_f = static_cast<float>(oneMinusBeta1);
    const float beta2_f = static_cast<float>(beta2);
    const float one_minus_beta2_f = static_cast<float>(oneMinusBeta2);
    const float first_correction = static_cast<float>(firstMomentCorrection);
    const float second_correction = static_cast<float>(secondMomentCorrection);
    const float epsilon_f = static_cast<float>(epsilon);
    const float decay_f = static_cast<float>(decay);
    const float update_scale = static_cast<float>(updateScale);

    const auto m_scaled = at::mul(first_moment, beta1_f);
    auto m = axpy(m_scaled, gradient, one_minus_beta1_f);
    const auto g2 = at::mul(gradient, gradient);
    const auto v_scaled = at::mul(second_moment, beta2_f);
    auto v = axpy(v_scaled, g2, one_minus_beta2_f);
    const auto m_hat = at::mul(m, first_correction);
    const auto v_hat = at::mul(v, second_correction);
    // Adam specifies the raw IEEE square root; Buffer.sqrt has a selected nonpositive branch.
    // A negative v_hat must therefore remain NaN rather than being clamped.
    const auto denominator = at::add(at::sqrt(v_hat), epsilon_f);
    // Keep a tensor denominator: ATen's scalar-divisor optimization uses a
    // rounded reciprocal followed by multiplication, which can change bits.
    const auto normalized_update = at::div(m_hat, denominator);
    const auto decayed_parameters = axpy(parameters, parameters, decay_f);
    auto updated = axpy(decayed_parameters, normalized_update, update_scale);
    return triple(std::move(updated), std::move(m), std::move(v));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_reduce_sum(b_lean_obj_arg BObj) {
  return invoke([&]() { return box(reduce_sum(flat(BObj))); });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_reduce_mean(b_lean_obj_arg BObj) {
  return invoke([&]() {
    const auto x = flat(BObj);
    if (x.numel() == 0) {
      return box(at::full({1}, std::numeric_limits<float>::quiet_NaN(), x.options()));
    }
    // Mean first rounds the complete sum, then multiplies by a host
    // float32 reciprocal. ATen mean may distribute the scale across its tree.
    const float scale = 1.0f / static_cast<float>(x.numel());
    return box(at::mul(reduce_sum(x), scale));
  });
}

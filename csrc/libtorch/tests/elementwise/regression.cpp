// No CUDA code: this executable calls the production C ABI and upstream ATen.
// Build and run only in the cluster; see README.md for the Lean reference input.
#include "../../torchlean_libtorch.h"

#include <ATen/record_function.h>
#include <torch/version.h>

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void lean_initialize_runtime_module();
extern "C" lean_obj_res torchlean_libtorch_get_setting(uint32_t);
extern "C" lean_obj_res torchlean_libtorch_set_setting(uint32_t, uint32_t);
extern "C" lean_obj_res torchlean_libtorch_get_memory_fraction(uint32_t);

#define DECLARE_UNARY(name) \
  extern "C" lean_obj_res torchlean_cuda_buffer_##name(b_lean_obj_arg);
#define DECLARE_BINARY(name) \
  extern "C" lean_obj_res torchlean_cuda_buffer_##name(b_lean_obj_arg, b_lean_obj_arg);
DECLARE_UNARY(abs)
DECLARE_UNARY(sqrt)
DECLARE_UNARY(exp)
DECLARE_UNARY(sin)
DECLARE_UNARY(cos)
DECLARE_UNARY(log)
DECLARE_UNARY(inv)
DECLARE_UNARY(relu)
DECLARE_UNARY(gelu)
DECLARE_UNARY(copy_and_release)
DECLARE_UNARY(reduce_sum)
DECLARE_UNARY(reduce_mean)
DECLARE_BINARY(max)
DECLARE_BINARY(min)
DECLARE_BINARY(div)
DECLARE_BINARY(add)
DECLARE_BINARY(sub)
DECLARE_BINARY(mul)
DECLARE_BINARY(abs_bwd)
DECLARE_BINARY(sqrt_bwd)
DECLARE_BINARY(relu_bwd)
DECLARE_BINARY(gelu_bwd)
#undef DECLARE_BINARY
#undef DECLARE_UNARY
extern "C" lean_obj_res torchlean_cuda_buffer_scale(b_lean_obj_arg, double);
extern "C" lean_obj_res torchlean_cuda_buffer_axpy(b_lean_obj_arg, b_lean_obj_arg, double);
extern "C" lean_obj_res torchlean_cuda_buffer_scaled_prod_exp(
    b_lean_obj_arg, b_lean_obj_arg, double);
extern "C" lean_obj_res torchlean_cuda_buffer_clamp(b_lean_obj_arg, double, double);
extern "C" lean_obj_res torchlean_cuda_buffer_clamp_bwd(
    b_lean_obj_arg, b_lean_obj_arg, double, double);
extern "C" lean_obj_res torchlean_cuda_buffer_max_bwd(
    b_lean_obj_arg, b_lean_obj_arg, b_lean_obj_arg);
extern "C" lean_obj_res torchlean_cuda_buffer_min_bwd(
    b_lean_obj_arg, b_lean_obj_arg, b_lean_obj_arg);
extern "C" lean_obj_res torchlean_cuda_buffer_adam_step(
    b_lean_obj_arg, b_lean_obj_arg, b_lean_obj_arg, b_lean_obj_arg,
    double, double, double, double, double, double, double, double, double);

namespace {

void check(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

float from_bits(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

uint32_t to_bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

bool is_nan(uint32_t bits) {
  return (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0;
}

std::string hex(uint32_t bits) {
  std::ostringstream out;
  out << "0x" << std::hex << std::setfill('0') << std::setw(8) << bits;
  return out.str();
}

size_t comparisons = 0;
size_t nan_payload_differences = 0;

void equal_bits(uint32_t actual, uint32_t expected, const std::string& label) {
  ++comparisons;
  if (actual == expected) return;
  // This is exactly AgreeUpToNaN: never use a tolerance for finite values or signed zeros.
  if (is_nan(actual) && is_nan(expected)) {
    ++nan_payload_differences;
    return;
  }
  throw std::runtime_error(label + ": got " + hex(actual) + ", expected " + hex(expected));
}

at::Tensor on_device(const std::vector<uint32_t>& bits) {
  auto cpu = at::empty({static_cast<int64_t>(bits.size())},
                       at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  if (!bits.empty()) std::memcpy(cpu.data_ptr<float>(), bits.data(), bits.size() * sizeof(float));
  return cpu.to(at::kCUDA);
}

std::vector<uint32_t> tensor_bits(const at::Tensor& value) {
  check(value.scalar_type() == at::kFloat, "result is not float32");
  const auto cpu = value.to(at::kCPU).contiguous();
  std::vector<uint32_t> bits(cpu.numel());
  if (!bits.empty()) std::memcpy(bits.data(), cpu.data_ptr<float>(), bits.size() * sizeof(float));
  return bits;
}

void equal_tensor(const at::Tensor& value, const std::vector<uint32_t>& expected,
                  const std::string& label) {
  const auto actual = tensor_bits(value);
  check(actual.size() == expected.size(), label + ": wrong length");
  for (size_t i = 0; i < actual.size(); ++i)
    equal_bits(actual[i], expected[i], label + "[" + std::to_string(i) + "]");
}

struct Object {
  lean_object* value;
  explicit Object(lean_object* object) : value(object) {}
  explicit Object(const at::Tensor& tensor) : value(torchlean::box(tensor)) {}
  ~Object() { lean_dec(value); }
  Object(const Object&) = delete;
  Object& operator=(const Object&) = delete;
  operator lean_object*() const { return value; }
  const at::Tensor& tensor() const { return torchlean::tensor(value); }
};

// Setting IDs are the current LibTorch control ABI, shared with Cuda.LibTorch.
constexpr uint32_t kDeterministic = 2;
constexpr uint32_t kCuDNNBenchmark = 3;

uint32_t get_setting(uint32_t setting) {
  const Object result(torchlean_libtorch_get_setting(setting));
  check(lean_io_result_is_ok(result), "LibTorch setting read failed: " +
                                    std::to_string(setting));
  const auto value = lean_unbox_uint32(lean_ctor_get(result, 0));
  check(value <= 1, "LibTorch setting returned an invalid boolean");
  return value;
}

void set_setting(uint32_t setting, uint32_t value) {
  const Object result(torchlean_libtorch_set_setting(setting, value));
  check(lean_io_result_is_ok(result), "LibTorch setting request failed: " +
                                    std::to_string(setting) + "=" + std::to_string(value));
  check(get_setting(setting) == value, "LibTorch setting readback disagrees with request");
}

void control_getter_regressions() {
  const auto deterministic = get_setting(kDeterministic);
  const Object invalid(torchlean_libtorch_get_setting(9));
  check(!lean_io_result_is_ok(invalid), "unknown LibTorch setting must return an IO error");
  check(get_setting(kDeterministic) == deterministic,
        "failed LibTorch setting read changed determinism");
  const Object memory_fraction(torchlean_libtorch_get_memory_fraction(0));
  check(lean_io_result_is_ok(memory_fraction), "LibTorch memory fraction read failed");
  const auto fraction = lean_unbox_float(lean_ctor_get(memory_fraction, 0));
  check(std::isfinite(fraction) && fraction >= 0.0 && fraction <= 1.0,
        "LibTorch memory fraction returned an invalid Float");
}

// Volatile materializes each binary32 stage independently; CMake forbids contraction/fast math.
float rounded_mul(float a, float b) { volatile float r = a * b; return r; }
float rounded_add(float a, float b) { volatile float r = a + b; return r; }
float rounded_div(float a, float b) { volatile float r = a / b; return r; }
float rounded_sqrt(float a) { volatile float r = std::sqrt(a); return r; }

struct FmaCase {
  uint32_t x, y, z, expected;
  const char* label;
};

void fused_discriminators() {
  const std::vector<FmaCase> cases{
      {0x3f800001, 0x3f800001, 0xbf800002, 0x28800000, "cancellation: 2^-46"},
      // The product is halfway between 0x3fc00001 and 0x3fc00002. Subtracting 2^-80
      // puts the exact sum just below halfway; binary64 addition loses that perturbation.
      {0x3f800001, 0x3fc00000, 0x97800000, 0x3fc00001, "binary64 double rounding"},
      {0x7f7fffff, 0x40000000, 0xff7fffff, 0x7f7fffff, "overflow cancellation"},
      {0x00800000, 0x3f000000, 0x00000000, 0x00400000, "subnormal result"},
      {0x00000001, 0x3f000000, 0x00000000, 0x00000000, "underflow tie"},
      {0x80000000, 0x3f800000, 0x80000000, 0x80000000, "two negative zeros"},
      {0x80000000, 0x3f800000, 0x00000000, 0x00000000, "opposite zeros"},
      {0x00000000, 0x7f800000, 0x3f800000, 0x7fc00000, "zero times infinity"},
      {0x7fc00123, 0x00000000, 0x3f800000, 0x7fc00000, "NaN times zero"},
      {0x3f800000, 0x3f800000, 0x7fc00123, 0x7fc00000, "NaN addend"},
  };
  std::vector<uint32_t> xs, ys, zs, expected;
  for (const auto& c : cases) {
    xs.push_back(c.x); ys.push_back(c.y); zs.push_back(c.z); expected.push_back(c.expected);
    equal_bits(to_bits(std::fma(from_bits(c.x), from_bits(c.y), from_bits(c.z))),
               c.expected, std::string("host FMA: ") + c.label);
    const Object a(on_device({c.z})), b(on_device({c.y}));
    const Object actual(torchlean_cuda_buffer_axpy(a, b, static_cast<double>(from_bits(c.x))));
    equal_tensor(actual.tensor(), {c.expected}, std::string("C ABI AXPY: ") + c.label);
    const auto coefficient = at::scalar_tensor(
        from_bits(c.x), at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
    equal_tensor(at::addcmul(a.tensor(), b.tensor(), coefficient, 1.0f),
                 {c.expected}, std::string("CPU scalar tensor2: ") + c.label);
    equal_tensor(at::addcmul(a.tensor(), b.tensor(), coefficient.to(at::kCUDA), 1.0f),
                 {c.expected}, std::string("CUDA scalar tensor2: ") + c.label);
  }
  const auto x = on_device(xs), y = on_device(ys), z = on_device(zs);
  equal_tensor(at::addcmul(z, x, y, 1.0f), expected, "CUDA tensor FMA");
  // Negative controls demonstrate that the fixtures reject these plausible replacements.
  equal_tensor(at::add(at::mul(x, y), z).slice(0, 0, 1), {0}, "split negative control");
  const auto widened = at::add(at::mul(x.to(at::kDouble), y.to(at::kDouble)),
                               z.to(at::kDouble)).to(at::kFloat);
  equal_tensor(widened.slice(0, 1, 2), {0x3fc00002}, "binary64 negative control");

  // add(alpha) may happen to fuse in this SDK build. Report it, never infer a guarantee.
  const auto implicit = at::add(z.slice(0, 0, 1), y.slice(0, 0, 1), from_bits(xs[0]));
  std::cout << "add(alpha) cancellation bits (diagnostic only): "
            << hex(tensor_bits(implicit)[0]) << '\n';

  // The C ABI first rounds its double coefficient to float32, even at a halfway value.
  const Object zero(on_device({0})), one(on_device({0x3f800000}));
  const Object rounded(torchlean_cuda_buffer_axpy(zero, one, 1.0 + 0x1p-24));
  equal_tensor(rounded.tensor(), {0x3f800000}, "AXPY scalar rounds to float32");
}

using AdamScalars = std::array<double, 9>;

std::array<float, 3> adam_oracle(float p, float g, float m0, float v0, const AdamScalars& s) {
  const float b1 = static_cast<float>(s[0]), omb1 = static_cast<float>(s[1]);
  const float b2 = static_cast<float>(s[2]), omb2 = static_cast<float>(s[3]);
  const float c1 = static_cast<float>(s[4]), c2 = static_cast<float>(s[5]);
  const float eps = static_cast<float>(s[6]), decay = static_cast<float>(s[7]);
  const float update = static_cast<float>(s[8]);
  const float m = std::fma(omb1, g, rounded_mul(m0, b1));
  const float g2 = rounded_mul(g, g);
  const float v = std::fma(omb2, g2, rounded_mul(v0, b2));
  const float m_hat = rounded_mul(m, c1), v_hat = rounded_mul(v, c2);
  const float denom = rounded_add(rounded_sqrt(v_hat), eps);
  const float normalized = rounded_div(m_hat, denom);
  const float decayed = std::fma(decay, p, p);
  return {std::fma(update, normalized, decayed), m, v};
}

lean_obj_res adam(const Object& p, const Object& g, const Object& m, const Object& v,
                  const AdamScalars& s) {
  return torchlean_cuda_buffer_adam_step(
      p, g, m, v, s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8]);
}

std::array<lean_object*, 3> triple_fields(lean_object* result) {
  const auto tail = lean_ctor_get(result, 1);
  return {lean_ctor_get(result, 0), lean_ctor_get(tail, 0), lean_ctor_get(tail, 1)};
}

void adam_fixture(const std::string& label, const std::array<uint32_t, 4>& input,
                  const AdamScalars& scalars, const std::array<uint32_t, 3>& expected) {
  const Object p(on_device({input[0]})), g(on_device({input[1]}));
  const Object m(on_device({input[2]})), v(on_device({input[3]}));
  const Object result(adam(p, g, m, v, scalars));
  const auto fields = triple_fields(result);
  const auto oracle = adam_oracle(from_bits(input[0]), from_bits(input[1]),
                                  from_bits(input[2]), from_bits(input[3]), scalars);
  for (size_t i = 0; i < 3; ++i) {
    equal_bits(to_bits(oracle[i]), expected[i], label + " independent fixture/oracle");
    equal_tensor(torchlean::tensor(fields[i]), {expected[i]},
                 label + " output " + std::to_string(i));
  }
}

void adam_regressions() {
  const double t = static_cast<double>(from_bits(0x3f800001));
  const double s = static_cast<double>(from_bits(0x3f800002));
  // Both moment residuals vanish if either explicit FMA is replaced by mul then add.
  adam_fixture("moment FMA residuals", {0, 0x3f800001, 0xbf800002, 0xbf800004},
               {1, t, 1, s, 1, 1, 0, 0, 0}, {0, 0x28800000, 0x29800000});
  adam_fixture("parameter update FMA", {0xbf800002, 0, 0x3f800001, 0x3f800000},
               {1, 0, 1, 0, 1, 1, 0, 0, t}, {0x28800000, 0x3f800001, 0x3f800000});
  // (1+2^-23)*(-1+2^-23) + (1+2^-23) = 2^-23 + 2^-46.
  adam_fixture("decay FMA", {0x3f800001, 0, 0, 0x3f800000},
               {1, 0, 1, 0, 1, 1, 0, from_bits(0xbf7ffffe), 0},
               {0x34000001, 0, 0x3f800000});
  // The initial multiplications are rounded BEFORE the moment FMAs.
  adam_fixture("first moment multiplication boundary", {0, 0xbf800000, 0x3f800001, 0x3f800000},
               {t, s, 1, 0, 1, 1, 0, 0, 0}, {0, 0, 0x3f800000});
  adam_fixture("second moment multiplication boundary", {0, 0x3f800000, 0, 0x3f800001},
               {1, 0, t, -s, 1, 1, 1, 0, 0}, {0, 0, 0});
  // Squaring the gradient must be a separate rounded stage too.
  adam_fixture("gradient square boundary", {0, 0x3f800001, 0, 0xbf800002},
               {1, 0, 1, 1, 1, 1, 1, 0, 0}, {0, 0, 0});
  adam_fixture("Adam raw sqrt remains NaN", {0, 0, 0x3f800000, 0xbf800000},
               {1, 0, 1, 0, 1, 1, 0, 0, 1}, {0x7fc00000, 0x3f800000, 0xbf800000});

  constexpr size_t count = 1024;
  std::array<std::vector<uint32_t>, 4> input;
  uint32_t random = 0x72ef941a;
  for (size_t i = 0; i < count; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      random ^= random << 13; random ^= random >> 17; random ^= random << 5;
      // Finite magnitudes [0.5, 1), signed except the second moment.
      input[j].push_back(0x3f000000u | (random & 0x007fffffu) |
                         (j == 3 ? 0u : (random & 0x80000000u)));
    }
  }
  const Object p(on_device(input[0])), g(on_device(input[1]));
  const Object m(on_device(input[2])), v(on_device(input[3]));
  const AdamScalars ordinary{0.9, 0.1, 0.999, 0.001, 1.25, 1.5, 1e-8, -0.01, -0.001};
  // Independently perturb each double scalar across a binary32 midpoint. This also prevents
  // recomputing oneMinusBeta from beta or moving the host conversion past arithmetic.
  for (size_t variant = 0; variant <= ordinary.size(); ++variant) {
    auto scalars = ordinary;
    if (variant != 0) scalars[variant - 1] = 1.0 + 0x1p-24;
    std::array<std::vector<uint32_t>, 3> expected;
    for (size_t i = 0; i < count; ++i) {
      const auto row = adam_oracle(from_bits(input[0][i]), from_bits(input[1][i]),
                                   from_bits(input[2][i]), from_bits(input[3][i]), scalars);
      for (size_t j = 0; j < 3; ++j) expected[j].push_back(to_bits(row[j]));
    }
    const Object result(adam(p, g, m, v, scalars));
    const auto fields = triple_fields(result);
    for (size_t j = 0; j < 3; ++j)
      equal_tensor(torchlean::tensor(fields[j]), expected[j],
                   "Adam staged oracle variant " + std::to_string(variant) +
                   " output " + std::to_string(j));
  }
}

void deterministic_reduction_regressions() {
  const auto old_deterministic = get_setting(kDeterministic);
  const auto old_benchmark = get_setting(kCuDNNBenchmark);
  const auto restore_settings = [&] {
    set_setting(kDeterministic, old_deterministic);
    set_setting(kCuDNNBenchmark, old_benchmark);
  };
  try {
    set_setting(kDeterministic, 1);
    const auto sum_case = [](const std::vector<uint32_t>& input, uint32_t expected,
                             const std::string& label) {
      const Object x(on_device(input));
      const Object first(torchlean_cuda_buffer_reduce_sum(x));
      const Object second(torchlean_cuda_buffer_reduce_sum(x));
      equal_tensor(first.tensor(), {expected}, label);
      equal_tensor(second.tensor(), {expected}, label + " repeat");
    };
    sum_case({}, 0, "empty sum");
    sum_case({0x80000000}, 0, "sum starts with positive zero");
    sum_case({0x00000001, 0x00000001}, 0x00000002, "sum retains subnormals");
    sum_case({0x7f800000, 0xff800000}, 0x7fc00000, "sum opposite infinities");
    // The 256-lane tree first combines lanes 0 and 2, then adds lane 1.
    // These two cases distinguish it from both a float32 left fold and a double sum.
    constexpr uint32_t large = 0x4cbebc20;      // 100000000
    constexpr uint32_t negative = 0xccbebc20;  // -100000000
    sum_case({large, 0x3f800000, negative}, 0x3f800000,
             "tree cancellation retains middle unit");
    sum_case({large, negative, 0x3f800000}, 0,
             "tree cancellation rounds lane pair before final add");
    check(to_bits(rounded_add(rounded_add(from_bits(large), from_bits(negative)), 1.0f)) ==
              0x3f800000,
          "left-fold negative control changed");
    check(to_bits(static_cast<float>(
              static_cast<double>(from_bits(large)) + from_bits(negative) + 1.0)) ==
              0x3f800000,
          "double-sum negative control changed");

    std::vector<uint32_t> boundary(257, 0);
    boundary[0] = large;
    boundary[128] = negative;
    boundary[255] = 0x3f800000;
    boundary[256] = 0x40000000;
    sum_case(boundary, 0x40400000, "partial final block");
    std::vector<uint32_t> recursive(513, 0);
    recursive[0] = large;
    recursive[256] = negative;
    recursive[512] = 0x3f800000;
    sum_case(recursive, 0, "recursive partial sums use the same tree");

    // Above the grid cap, lane 0 adds the trailing unit BEFORE cancellation with lane 128.
    // Removing the cap would place that unit in a separate block and produce one instead.
    constexpr size_t capped_stride = 65535u * 256u;
    std::vector<uint32_t> capped(capped_stride + 1, 0);
    capped[0] = large;
    capped[128] = negative;
    capped[capped_stride] = 0x3f800000;
    sum_case(capped, 0, "capped grid-stride accumulation precedes block tree");

    const Object mean_input(on_device({0x40e00000, 0, 0}));  // [7, 0, 0]
    const Object mean(torchlean_cuda_buffer_reduce_mean(mean_input));
    equal_tensor(mean.tensor(), {0x40155556}, "mean rounds reciprocal before multiply");
    equal_bits(to_bits(rounded_div(7.0f, 3.0f)), 0x40155555,
               "direct-division negative control");
    const Object overflowing(on_device({0x7f7fffff, 0x7f7fffff}));
    const Object overflow_mean(torchlean_cuda_buffer_reduce_mean(overflowing));
    equal_tensor(overflow_mean.tensor(), {0x7f800000}, "mean rounds sum before scaling");
    const Object empty(on_device({}));
    const Object empty_mean(torchlean_cuda_buffer_reduce_mean(empty));
    equal_tensor(empty_mean.tensor(), {0x7fc00000}, "empty mean is NaN");
  } catch (...) {
    restore_settings();
    throw;
  }
  restore_settings();
  std::cout << "deterministic reductions: exact tree/grid-cap and mean stages\n";
}

// Observe GradMode *inside* ATen dispatch: merely checking detached outputs could conceal a
// graph that was constructed and then detached by the runtime's boxing helper.
thread_local bool observing_export = false;
thread_local size_t observed_ops = 0;
thread_local size_t observed_grad_enabled = 0;

std::unique_ptr<at::ObserverContext> observe(const at::RecordFunction&) {
  if (observing_export) {
    ++observed_ops;
    if (at::GradMode::is_enabled()) ++observed_grad_enabled;
  }
  return nullptr;
}

struct Observer {
  at::CallbackHandle handle = at::addThreadLocalCallback(at::RecordFunctionCallback(&observe));
  ~Observer() { at::removeCallback(handle); }
};

void no_graph(lean_object* object, const std::string& label) {
  const auto& value = torchlean::tensor(object);
  check(!value.requires_grad(), label + ": result requires gradients");
  check(value.grad_fn() == nullptr, label + ": result has an autograd node");
}

void no_autograd_regressions() {
  Observer observer;
  for (const bool ambient_grad : {true, false}) {
    at::AutoGradMode ambient(ambient_grad);
    const Object x(on_device({0x3f000000, 0x3f800000}));
    const Object y(on_device({0x3fc00000, 0x40000000}));
    const Object g(on_device({0x3f800000, 0x3f800000}));
    // The runtime rejects gradient-bearing Buffer inputs. Use a separate ATen leaf to
    // calibrate observation and exercise invoke with gradients, then valid Buffer inputs
    // for all exports. Observation detects a missing guard even with no-grad inputs.
    auto leaf = on_device({0x3f800000});
    leaf.set_requires_grad(true);
    observed_ops = observed_grad_enabled = 0;
    observing_export = true;
    const auto calibration = at::mul(leaf, leaf);
    observing_export = false;
    check(observed_ops > 0 && (observed_grad_enabled > 0) == ambient_grad,
          "ATen observer calibration failed");
    check(calibration.requires_grad() == ambient_grad, "autograd calibration failed");
    const auto guarded = torchlean::invoke([&] {
      check(!at::GradMode::is_enabled(), "invoke did not disable gradients");
      return at::mul(leaf, leaf);
    });
    check(!guarded.requires_grad() && guarded.grad_fn() == nullptr,
          "invoke recorded a graph from a gradient-bearing tensor");
    check(at::GradMode::is_enabled() == ambient_grad && leaf.requires_grad(),
          "invoke changed caller state");

    size_t export_count = 0;
    auto run = [&](const std::string& name, const std::function<lean_object*()>& call,
                   size_t fields = 1) {
      observed_ops = observed_grad_enabled = 0;
      observing_export = true;
      const Object result(call());
      observing_export = false;
      check(at::GradMode::is_enabled() == ambient_grad, name + ": ambient GradMode changed");
      check(observed_ops > 0, name + ": observer saw no ATen operations");
      check(observed_grad_enabled == 0, name + ": ATen executed with gradients enabled");
      if (fields == 1) no_graph(result, name);
      if (fields == 2) {
        no_graph(lean_ctor_get(result, 0), name + " first");
        no_graph(lean_ctor_get(result, 1), name + " second");
      }
      if (fields == 3)
        for (auto* field : triple_fields(result)) no_graph(field, name);
      ++export_count;
    };
#define CHECK_UNARY(name) run(#name, [&] { return torchlean_cuda_buffer_##name(x); });
    CHECK_UNARY(abs)
    CHECK_UNARY(sqrt)
    CHECK_UNARY(exp)
    CHECK_UNARY(sin)
    CHECK_UNARY(cos)
    CHECK_UNARY(log)
    CHECK_UNARY(inv)
    CHECK_UNARY(relu)
    CHECK_UNARY(gelu)
    CHECK_UNARY(reduce_sum)
    CHECK_UNARY(reduce_mean)
#undef CHECK_UNARY
#define CHECK_BINARY(name) run(#name, [&] { return torchlean_cuda_buffer_##name(x, y); });
    CHECK_BINARY(max)
    CHECK_BINARY(min)
    CHECK_BINARY(div)
    CHECK_BINARY(add)
    CHECK_BINARY(sub)
    CHECK_BINARY(mul)
#undef CHECK_BINARY
#define CHECK_VJP(name) run(#name "_bwd", [&] { return torchlean_cuda_buffer_##name##_bwd(x, g); });
    CHECK_VJP(abs)
    CHECK_VJP(sqrt)
    CHECK_VJP(relu)
    CHECK_VJP(gelu)
#undef CHECK_VJP
    run("scale", [&] { return torchlean_cuda_buffer_scale(x, 0.25); });
    run("axpy", [&] { return torchlean_cuda_buffer_axpy(x, y, 0.25); });
    run("scaled_prod_exp", [&] { return torchlean_cuda_buffer_scaled_prod_exp(x, y, 0.25); });
    run("clamp", [&] { return torchlean_cuda_buffer_clamp(x, 0.0, 1.0); });
    run("clamp_bwd", [&] { return torchlean_cuda_buffer_clamp_bwd(x, g, 0.0, 1.0); });
    run("max_bwd", [&] { return torchlean_cuda_buffer_max_bwd(x, y, g); }, 2);
    run("min_bwd", [&] { return torchlean_cuda_buffer_min_bwd(x, y, g); }, 2);
    run("adam_step", [&] { return adam(x, g, x, y, {1, 0, 1, 0, 1, 1, 0, 0, 1}); }, 3);
    const Object copied(on_device({0x3f800000}));
    run("copy_and_release", [&] { return torchlean_cuda_buffer_copy_and_release(copied); });
    check(export_count == 30, "noAutograd export coverage changed");
    check(!x.tensor().requires_grad() && !y.tensor().requires_grad() && !g.tensor().requires_grad(),
          "export mutated an input's requires_grad flag");
    const auto old_deterministic = get_setting(kDeterministic);
    const auto old_benchmark = get_setting(kCuDNNBenchmark);
    const auto restore_settings = [&] {
      set_setting(kDeterministic, old_deterministic);
      set_setting(kCuDNNBenchmark, old_benchmark);
    };
    try {
      set_setting(kDeterministic, !old_deterministic);
      if (!old_deterministic)
        check(get_setting(kCuDNNBenchmark) == 0,
              "strict determinism did not disable cuDNN benchmarking");
      run("other reduction mode: sum", [&] { return torchlean_cuda_buffer_reduce_sum(x); });
      run("other reduction mode: mean", [&] { return torchlean_cuda_buffer_reduce_mean(x); });
    } catch (...) {
      restore_settings();
      throw;
    }
    restore_settings();
    check(at::GradMode::is_enabled() == ambient_grad,
          "LibTorch settings changed caller GradMode");
    std::cout << "noAutograd: 30 exports plus both reduction paths, ambient="
              << ambient_grad << '\n';
  }
}

struct PrimitiveCase { uint32_t x, y = 0, z = 0, expected = 0; };

uint32_t parse_bits(const std::string& text) {
  size_t consumed = 0;
  const auto value = std::stoull(text, &consumed, 0);
  check(consumed == text.size() && value <= UINT32_MAX, "invalid reference bits: " + text);
  return static_cast<uint32_t>(value);
}

void lean_primitive_parity(const char* path) {
  std::ifstream file(path);
  check(file.good(), "cannot open Lean reference case file");
  std::map<std::string, std::vector<PrimitiveCase>> groups;
  std::string line;
  while (std::getline(file, line)) {
    std::istringstream fields(line);
    std::string op, word;
    if (!(fields >> op)) continue;
    check(op == "add" || op == "mul" || op == "div" || op == "fma" || op == "sqrt",
          "unknown reference primitive: " + op);
    std::vector<uint32_t> bits;
    while (fields >> word) bits.push_back(parse_bits(word));
    const size_t arity = op == "sqrt" ? 1 : (op == "fma" ? 3 : 2);
    check(bits.size() == arity + 1, "malformed reference row: " + line);
    PrimitiveCase c{bits[0], arity > 1 ? bits[1] : 0, arity > 2 ? bits[2] : 0, bits.back()};
    groups[op].push_back(c);
  }
  for (const std::string op : {"add", "mul", "div", "sqrt", "fma"}) {
    const auto& cases = groups[op];
    check(!cases.empty(), "reference file omitted " + op);
    std::vector<uint32_t> xs, ys, zs, expected, selected;
    for (const auto& c : cases) {
      xs.push_back(c.x); ys.push_back(c.y); zs.push_back(c.z); expected.push_back(c.expected);
      const float x = from_bits(c.x), y = from_bits(c.y), z = from_bits(c.z);
      const float host = op == "add" ? rounded_add(x, y) :
                         op == "mul" ? rounded_mul(x, y) :
                         op == "div" ? rounded_div(x, y) :
                         op == "sqrt" ? rounded_sqrt(x) : std::fma(x, y, z);
      equal_bits(to_bits(host), c.expected, "host oracle against Lean " + op);
      // Buffer.sqrt is selected_sqrt, whereas NativePrimitiveAgreement names raw IEEE sqrt.
      selected.push_back(op == "sqrt" && x <= 0.0f ? 0 : c.expected);
    }
    const Object x(on_device(xs)), y(on_device(ys)), z(on_device(zs));
    if (op == "fma") {
      equal_tensor(at::addcmul(z.tensor(), x.tensor(), y.tensor(), 1.0f), expected,
                   "Lean FMA / CUDA tensor addcmul");
      // Bound temporary scalar allocations for the default 200000-case sweep. Every case
      // still traverses production AXPY; only the device-to-host comparison is batched.
      constexpr size_t chunk_size = 4096;
      for (size_t begin = 0; begin < cases.size(); begin += chunk_size) {
        const size_t end = std::min(begin + chunk_size, cases.size());
        std::vector<at::Tensor> results;
        results.reserve(end - begin);
        for (size_t i = begin; i < end; ++i) {
          const Object a(z.tensor().slice(0, i, i + 1)), b(y.tensor().slice(0, i, i + 1));
          const Object result(torchlean_cuda_buffer_axpy(
              a, b, static_cast<double>(from_bits(xs[i]))));
          results.push_back(result.tensor());
        }
        equal_tensor(at::cat(results),
                     std::vector<uint32_t>(expected.begin() + begin, expected.begin() + end),
                     "Lean FMA / production scalar AXPY chunk " + std::to_string(begin));
      }
    } else if (op == "sqrt") {
      equal_tensor(at::sqrt(x.tensor()), expected, "Lean raw IEEE sqrt / ATen");
      const Object result(torchlean_cuda_buffer_sqrt(x));
      equal_tensor(result.tensor(), selected, "Lean sqrt / selected Buffer.sqrt");
    } else {
      const Object result(op == "add" ? torchlean_cuda_buffer_add(x, y) :
                          op == "mul" ? torchlean_cuda_buffer_mul(x, y) :
                                        torchlean_cuda_buffer_div(x, y));
      equal_tensor(result.tensor(), expected, "Lean " + op + " / production C ABI");
    }
    std::cout << "Lean primitive " << op << ": " << cases.size() << " cases\n";
  }
}

}  // namespace

// Shared entrypoint for the migrated parity CLI; its frontend is ordinary C++.
int torchlean_elementwise_regressions(const char* reference_path) {
  try {
    check(std::fesetround(FE_TONEAREST) == 0, "cannot set host round-to-nearest");
    lean_initialize_runtime_module();
    std::cout << "LibTorch headers: " << TORCH_VERSION << '\n';
    control_getter_regressions();
    {
      at::NoGradGuard no_grad;
      fused_discriminators();
      adam_regressions();
      deterministic_reduction_regressions();
      lean_primitive_parity(reference_path);
    }
    no_autograd_regressions();
    std::cout << "PASS: " << comparisons << " exact/AgreeUpToNaN comparisons; "
              << nan_payload_differences << " NaN encoding differences\n";
    return 0;
  } catch (const std::exception& error) {
    observing_export = false;
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}

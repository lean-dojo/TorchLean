// Standalone regression executable: link ATen/LibTorch and the Lean runtime.
// Pass the optional "cuda" argument on a CUDA-capable host to exercise the CUDA dispatcher.
// Include the implementation to compare its general-rank composition against upstream kernels.
#define TORCHLEAN_CONV_POOL_TEST
#include "../conv_pool.cpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using namespace torchlean::conv_pool;

void close(const at::Tensor& actual, const at::Tensor& expected, const char* label,
           double tolerance = 2e-5) {
  if (actual.numel() != expected.numel() ||
      !at::allclose(actual.reshape({-1}), expected.reshape({-1}), tolerance, tolerance, true))
    throw std::runtime_error(label);
}

at::Tensor values(std::initializer_list<float> entries, const at::TensorOptions& options) {
  return at::tensor(std::vector<float>(entries), options.device(at::kCPU)).to(options.device());
}

void ordinary_convolutions(const at::TensorOptions& options) {
  for (size_t rank = 1; rank <= 3; ++rank) {
    Shape input(rank, 4), kernel(rank, 2), stride(rank, 1), padding(rank, 0);
    stride.back() = 2;
    padding.front() = 1;
    for (const bool transposed : {false, true}) {
      const Geometry g(input, kernel, stride, padding,
                       transposed ? Kind::transposed : Kind::convolution);
      const auto x = (at::arange(2 * g.input_volume, options) / 17 - 1)
          .reshape(with_channels(2, g.input));
      const auto w = (at::arange(6 * g.kernel_volume, options) / 23 - 1)
          .reshape(g.kernel_shape(2, 3, transposed));
      const auto bias = at::arange(3, options) / 5;
      const auto upstream = convolution_forward(x, w, bias, g, 2, 3, transposed);
      close(convolution_composed(x, w, bias, g, 2, 3, transposed), upstream,
            "composed convolution differs from upstream");
      const auto dy = (at::arange(upstream.numel(), options) / 29 - 1)
          .reshape(upstream.sizes());
      const auto expected = convolution_backward(x, w, dy, g, 2, 3, transposed);
      const auto actual = convolution_backward_composed(x, w, dy, g, 2, 3, transposed);
      close(std::get<0>(actual), std::get<0>(expected), "convolution weight cotangent");
      close(std::get<1>(actual), std::get<1>(expected), "convolution bias cotangent");
      close(std::get<2>(actual), std::get<2>(expected), "convolution input cotangent");
    }
  }
}

void higher_rank_convolutions(const at::TensorOptions& options) {
  // Independent hand-computed values exercise every supported rank above the upstream limit.
  for (size_t rank = 4; rank <= 8; ++rank) {
    Shape input(rank, 1), kernel(rank, 1), stride(rank, 1), padding(rank, 0);
    input[0] = 3;
    input[1] = 2;
    kernel[0] = 2;
    const Geometry g(input, kernel, stride, padding, Kind::convolution);
    const auto x = at::arange(1, 7, options);
    const auto w = values({2, -1}, options);
    const auto bias = at::zeros({1}, options);
    close(convolution_forward(x, w, bias, g, 1, 1, false),
          values({-1, 0, 1, 2}, options), "higher-rank convolution");
    const auto grads = convolution_backward(x, w, at::ones({4}, options), g, 1, 1, false);
    close(std::get<0>(grads), values({10, 18}, options), "higher-rank kernel gradient");
    close(std::get<1>(grads), values({4}, options), "higher-rank bias gradient");
    close(std::get<2>(grads), values({2, 2, 1, 1, -1, -1}, options),
          "higher-rank input gradient");

    input[0] = 2;
    const Geometry gt(input, kernel, stride, padding, Kind::transposed);
    const auto xt = at::arange(1, 5, options);
    close(convolution_forward(xt, w, bias, gt, 1, 1, true),
          values({2, 4, 5, 6, -3, -4}, options), "higher-rank transposed convolution");
    const auto grads_t = convolution_backward(xt, w, at::ones({6}, options), gt, 1, 1, true);
    close(std::get<0>(grads_t), values({10, 10}, options), "transposed kernel gradient");
    close(std::get<1>(grads_t), values({6}, options), "transposed bias gradient");
    close(std::get<2>(grads_t), at::ones({4}, options), "transposed input gradient");
  }
}

void ordinary_pooling(const at::TensorOptions& options) {
  for (size_t rank = 1; rank <= 3; ++rank) {
    Shape input(rank, 4), kernel(rank, 2), stride(rank, 1), padding(rank, 0);
    stride.back() = 2;
    padding.front() = 1;
    const Geometry g(input, kernel, stride, padding, Kind::pooling);
    // Singleton axes embed the same windows into rank four, selecting the independent composition.
    input.resize(4, 1);
    kernel.resize(4, 1);
    stride.resize(4, 1);
    padding.resize(4, 0);
    const Geometry general(input, kernel, stride, padding, Kind::pooling);
    const auto x = at::arange(2 * g.input_volume, options).remainder(5) - 2;
    const auto dy = at::arange(1, 2 * g.output_volume + 1, options) / 7;
    close(max_pool_forward(x, g, 2), max_pool_forward(x, general, 2),
          "ordinary max-pool versus composition");
    close(max_pool_backward(x, dy, g, 2), max_pool_backward(x, dy, general, 2),
          "ordinary max-pool selected gradient");
    close(avg_pool_forward(x, g, 2), avg_pool_forward(x, general, 2),
          "ordinary average-pool including padding");
    close(avg_pool_backward(dy, g, 2), avg_pool_backward(dy, general, 2),
          "ordinary average-pool gradient");
  }
}

void exceptional_max_pool(const at::TensorOptions& options) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  const auto x = values({-inf, -inf, nan, 3, 3, nan}, options);
  for (size_t rank = 1; rank <= 8; ++rank) {
    Shape input(rank, 1), kernel(rank, 1), stride(rank, 1), padding(rank, 0);
    input[0] = 6;
    kernel[0] = 2;
    const Geometry g(input, kernel, stride, padding, Kind::pooling);
    close(max_pool_forward(x, g, 1), values({-inf, -inf, nan, 3, 3}, options),
          "max-pool NaN, negative infinity and finite tie convention", 0);
    close(max_pool_backward(x, at::ones({5}, options), g, 1),
          values({1, 1, 1, 1, 1, 0}, options), "max-pool exceptional selected gradient", 0);
    close(max_pool_backward(x, values({nan, 1, 1, 1, 1}, options), g, 1),
          values({nan, 1, 1, 1, 1, 0}, options),
          "max-pool NaN cotangent only reaches the selected input", 0);

    input[0] = 2;
    kernel[0] = 3;
    padding[0] = 1;
    const Geometry padded(input, kernel, stride, padding, Kind::pooling);
    close(max_pool_forward(values({nan, 3}, options), padded, 1),
          values({nan, nan}, options), "first valid NaN after padding is retained", 0);
    close(max_pool_backward(values({nan, 3}, options), values({1, 2}, options), padded, 1),
          values({3, 0}, options), "padding does not change the selected first NaN", 0);
    close(max_pool_backward(values({3, nan}, options), values({1, 2}, options), padded, 1),
          values({3, 0}, options), "later NaN after padding is ignored", 0);
  }
}

void small_padded_average(const at::TensorOptions& options) {
  // Upstream avg_pool3d rejects this input smaller than the kernel, despite valid windows.
  const Geometry g({2, 1, 1}, {3, 1, 1}, {1, 1, 1}, {1, 0, 0}, Kind::pooling);
  close(avg_pool_forward(values({2, 3}, options), g, 1),
        values({5.0f / 3, 5.0f / 3}, options), "small padded 3-D average-pool");
  close(avg_pool_backward(values({1, 2}, options), g, 1),
        values({1, 1}, options), "small padded 3-D average-pool cotangent");
}

void deterministic_pooling(const at::TensorOptions& options) {
  auto& context = at::globalContext();
  struct Restore {
    at::Context& context;
    bool enabled, warn_only;
    ~Restore() { context.setDeterministicAlgorithms(enabled, warn_only); }
  } restore{context, context.deterministicAlgorithms(), context.deterministicAlgorithmsWarnOnly()};
  context.setDeterministicAlgorithms(true, false);
  // This also exercises the 3-D CUDA fallback, which upstream backward would reject.
  ordinary_pooling(options);
  exceptional_max_pool(options);
  small_padded_average(options);
}

void smooth_pool(const at::TensorOptions& options) {
  const Geometry g({2}, {2}, {1}, {0}, Kind::pooling);
  const auto x = values({1e20f, -1e20f}, options);
  for (const double beta : {1e20, -1e20}) {
    const auto y = smooth_pool_forward(x, g, 1, beta);
    close(y / 1e20, values({beta > 0 ? 1.0f : -1.0f}, options),
          "smooth-pool pivot before beta multiplication");
    close(smooth_pool_backward(x, at::ones({1}, options), g, 1, beta),
          beta > 0 ? values({1, 0}, options) : values({0, 1}, options),
          "smooth-pool extreme gradient");
  }
  for (size_t rank : {size_t(1), size_t(4), size_t(8)}) {
    Shape input(rank, 1), kernel(rank, 1), stride(rank, 1), padding(rank, 0);
    input[0] = 2;
    kernel[0] = 3;
    padding[0] = 1;
    const Geometry padded(input, kernel, stride, padding, Kind::pooling);
    const auto source = values({2, 3}, options);
    for (const double beta : {0.5, -0.5}) {
      const double denominator = 1 + std::exp(2 * beta) + std::exp(3 * beta);
      const float expected = static_cast<float>(std::log(denominator) / beta);
      close(smooth_pool_forward(source, padded, 1, beta),
            values({expected, expected}, options), "smooth-pool includes zero padding");
      const auto gradient = smooth_pool_backward(
          source, values({1, 2}, options), padded, 1, beta);
      close(gradient,
            values({static_cast<float>(3 * std::exp(2 * beta) / denominator),
                    static_cast<float>(3 * std::exp(3 * beta) / denominator)}, options),
            "smooth-pool padded selected gradient");
    }
  }
}

void empty_and_wide_geometry(const at::TensorOptions& options) {
  const auto empty = at::empty({0}, options);
  const Geometry g({1, 1}, {3, 3}, {1, 1}, {0, 0}, Kind::convolution);
  const auto x = values({2}, options), w = at::ones({9}, options), bias = values({3}, options);
  close(convolution_forward(x, w, bias, g, 1, 1, false), empty, "empty convolution");
  const auto grads = convolution_backward(x, w, empty, g, 1, 1, false);
  close(std::get<0>(grads), at::zeros_like(w), "empty convolution kernel cotangent");
  close(std::get<1>(grads), at::zeros_like(bias), "empty convolution bias cotangent");
  close(std::get<2>(grads), at::zeros_like(x), "empty convolution input cotangent");
  const Geometry invalid({1}, {1}, {1}, {32768}, Kind::pooling);
  close(max_pool_forward(x, invalid, 1), empty, "excessive pool padding totalizes to empty");
  close(max_pool_backward(x, empty, invalid, 1), at::zeros_like(x), "empty max-pool gradient");
  close(avg_pool_backward(empty, invalid, 1), at::zeros_like(x), "empty average-pool gradient");
  close(smooth_pool_backward(x, empty, invalid, 1, -0.5), at::zeros_like(x),
        "empty smooth-pool gradient");
  const Geometry wide({2}, {1}, {UINT32_MAX}, {0}, Kind::pooling);
  close(max_pool_forward(values({4, 7}, options), wide, 1), values({4}, options),
        "UInt32 stride remains supported");
  close(max_pool_backward(values({4, 7}, options), values({3}, options), wide, 1),
        values({3, 0}, options), "UInt32 stride gradient");
  const Geometry zero_channels({2}, {1}, {1}, {0}, Kind::convolution);
  close(convolution_forward(empty, empty, bias, zero_channels, 0, 1, false),
        values({3, 3}, options), "zero input channels preserve bias");
  const Geometry empty_spatial({0}, {1}, {1}, {1}, Kind::convolution);
  close(convolution_forward(empty, values({2}, options), bias, empty_spatial, 1, 1, false),
        values({3, 3}, options), "padded empty input preserves bias");
  const auto empty_spatial_grads = convolution_backward(
      empty, values({2}, options), at::ones({2}, options), empty_spatial, 1, 1, false);
  close(std::get<0>(empty_spatial_grads), values({0}, options), "empty spatial kernel gradient");
  close(std::get<1>(empty_spatial_grads), values({2}, options), "empty spatial bias gradient");
  close(std::get<2>(empty_spatial_grads), empty, "empty spatial input gradient");
  const Geometry large_empty(
      {UINT32_MAX, UINT32_MAX}, {1, 1}, {1, 1}, {0, 0}, Kind::pooling);
  close(max_pool_forward(empty, large_empty, 0), empty, "zero channels with wide spatial product");
  close(avg_pool_backward(empty, large_empty, 0), empty, "wide empty average-pool gradient");
  close(smooth_pool_forward(empty, large_empty, 0, 1), empty, "wide empty smooth-pool");
  for (const int64_t padding : {int64_t(UINT32_MAX), int64_t(UINT32_MAX - 1)}) {
    const Geometry wide_conv({3}, {1}, {UINT32_MAX}, {padding}, Kind::convolution);
    const auto source = values({2, 7, 11}, options);
    const auto weight = values({1}, options), zero_bias = values({0}, options);
    const auto middle = padding == UINT32_MAX ? 2.0f : 7.0f;
    close(convolution_forward(source, weight, zero_bias, wide_conv, 1, 1, false),
          values({0, middle, 0}, options), "UInt32 padded convolution", 0);
    const auto gradient = convolution_backward(
        source, weight, at::ones({3}, options), wide_conv, 1, 1, false);
    close(std::get<0>(gradient), values({middle}, options), "UInt32 kernel cotangent", 0);
    close(std::get<2>(gradient),
          padding == UINT32_MAX ? values({1, 0, 0}, options) : values({0, 1, 0}, options),
          "UInt32 input cotangent", 0);
  }
  const Geometry wide_transpose(
      {3}, {1}, {UINT32_MAX}, {UINT32_MAX}, Kind::transposed);
  close(convolution_forward(values({2, 7, 11}, options), values({1}, options),
                            values({0}, options), wide_transpose, 1, 1, true),
        values({7}, options), "UInt32 transposed convolution", 0);
  const auto transposed_grads = convolution_backward(
      values({2, 7, 11}, options), values({1}, options), values({1}, options),
      wide_transpose, 1, 1, true);
  close(std::get<0>(transposed_grads), values({7}, options), "UInt32 transpose kernel gradient", 0);
  close(std::get<2>(transposed_grads), values({0, 1, 0}, options),
        "UInt32 transpose input gradient", 0);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    at::NoGradGuard no_grad;
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
    const auto device = argc > 1 ? at::Device(argv[1]) : at::Device(at::kCPU);
    const auto options = at::TensorOptions().dtype(at::kFloat).device(device);
    ordinary_convolutions(options);
    higher_rank_convolutions(options);
    ordinary_pooling(options);
    exceptional_max_pool(options);
    small_padded_average(options);
    deterministic_pooling(options);
    smooth_pool(options);
    empty_and_wide_geometry(options);
    std::cout << "LibTorch conv/pool regression checks passed on " << device << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

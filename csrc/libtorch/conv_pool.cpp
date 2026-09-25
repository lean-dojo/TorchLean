#include "torchlean_libtorch.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <tuple>
#include <vector>

// ATen schemas checked against the PyTorch 2.12 nightly 0291f960b6 and pip torch 2.13.0+cu130:
// https://github.com/pytorch/pytorch/blob/0291f960b6/aten/src/ATen/native/native_functions.yaml
// Backward calls return explicit cotangents to the Lean tape; they do not record autograd graphs.
namespace torchlean::conv_pool {

// TorchLean shapes use Nat subtraction, so negative intermediate lengths clamp at zero.
constexpr size_t kMaxRank = 8;

static uint32_t outDim(uint32_t in, uint32_t k, uint32_t stride, uint32_t padding) {
  torchlean::require(stride != 0, "LibTorch conv/pool: stride must be > 0");
  if (k == 0) return 0;
  // Invalid geometry has no windows; do not turn saturated subtraction into a phantom window.
  const uint64_t inPad = uint64_t{in} + 2 * uint64_t{padding};
  if (inPad < k) return 0;
  const uint64_t out = (inPad - k) / stride + 1;
  torchlean::require(out <= UINT32_MAX, "LibTorch conv/pool: outDim overflow");
  return static_cast<uint32_t>(out);
}

// N-D pooling follows `Spec.poolOutSpatialPad`: empty inputs and padding beyond half the kernel
// are invalid axes, even when the generic sliding-window formula would produce a positive length.
static uint32_t poolOutDim(uint32_t in, uint32_t k, uint32_t stride, uint32_t padding) {
  if (in == 0 || k == 0 || padding > k / 2) return 0;
  return outDim(in, k, stride, padding);
}

// Spec: ((in - 1) * stride + k) - 2 * padding in Nat, so the addition precedes the subtraction.
static uint32_t outDimTranspose(uint32_t in, uint32_t k, uint32_t stride, uint32_t padding) {
  torchlean::require(stride != 0, "LibTorch conv/pool: stride must be > 0");
  if (in == 0 || k == 0) return 0;
  const uint64_t t = uint64_t{in - 1} * stride + k;
  const uint64_t sub = 2 * uint64_t{padding};
  const uint64_t out = t >= sub ? t - sub : 0;
  torchlean::require(out <= UINT32_MAX, "LibTorch conv/pool: outDimTranspose overflow");
  return static_cast<uint32_t>(out);
}

// Validate after the ABI's binary64-to-binary32 conversion: a finite nonzero Lean `Float` may
// overflow to infinity or underflow to zero as a float32.
static float checked_smoothmax_beta(double beta, const char* msg) {
  const float betaF = static_cast<float>(beta);
  torchlean::require(std::isfinite(betaF) && betaF != 0.0f, msg);
  return betaF;
}

// Floor and ceiling division for b > 0.
static int64_t floor_div_i64(int64_t a, int64_t b) {
  const int64_t q = a / b;
  return (a % b != 0 && a < 0) ? q - 1 : q;
}

static int64_t ceil_div_i64(int64_t a, int64_t b) { return -floor_div_i64(-a, b); }

using Shape = std::vector<int64_t>;
using Gradients = std::tuple<at::Tensor, at::Tensor, at::Tensor>;
enum class Kind { convolution, transposed, pooling };

static size_t spatial_volume(const Shape& shape) {
  if (std::find(shape.begin(), shape.end(), 0) != shape.end()) return 0;
  size_t result = 1;
  for (const auto dim : shape) {
    result = checked_mul_size(result, static_cast<size_t>(dim),
                              "torchlean conv/pool: dimension product overflow");
  }
  return result;
}

static int64_t volume(const Shape& shape) {
  const size_t result = spatial_volume(shape);
  require(result <= static_cast<size_t>(INT64_MAX),
          "torchlean conv/pool: tensor size exceeds ATen indexing");
  return static_cast<int64_t>(result);
}

static Shape with_channels(int64_t channels, const Shape& spatial) {
  Shape shape{channels};
  shape.insert(shape.end(), spatial.begin(), spatial.end());
  return shape;
}

struct Geometry {
  Shape input, kernel, stride, padding, output;
  size_t input_volume, kernel_volume, output_volume;

  Geometry(Shape in, Shape k, Shape s, Shape p, Kind kind)
      : input(std::move(in)), kernel(std::move(k)),
        stride(std::move(s)), padding(std::move(p)) {
    require(!input.empty() && input.size() <= kMaxRank,
            "torchlean conv/pool: spatial rank must be between one and eight");
    require(kernel.size() == input.size() && stride.size() == input.size() &&
                padding.size() == input.size(),
            "torchlean conv/pool: array rank mismatch");
    for (size_t axis = 0; axis < input.size(); ++axis) {
      require(input[axis] >= 0 && input[axis] <= UINT32_MAX &&
                  kernel[axis] > 0 && kernel[axis] <= UINT32_MAX &&
                  stride[axis] > 0 && stride[axis] <= UINT32_MAX &&
                  padding[axis] >= 0 && padding[axis] <= UINT32_MAX,
              "torchlean conv/pool: invalid spatial dimension, kernel, stride or padding");
      const auto i = static_cast<uint32_t>(input[axis]);
      const auto w = static_cast<uint32_t>(kernel[axis]);
      const auto s0 = static_cast<uint32_t>(stride[axis]);
      const auto p0 = static_cast<uint32_t>(padding[axis]);
      output.push_back(kind == Kind::transposed ? outDimTranspose(i, w, s0, p0)
                       : kind == Kind::pooling ? poolOutDim(i, w, s0, p0)
                                               : outDim(i, w, s0, p0));
    }
    // Validate spatial products even when a channel count subsequently makes the buffer empty.
    // Keep these unsigned: zero-channel buffers can carry larger spatial metadata than
    // a nonempty ATen tensor can represent. Validate actual buffer sizes separately.
    input_volume = spatial_volume(input);
    kernel_volume = spatial_volume(kernel);
    output_volume = spatial_volume(output);
  }

  bool ordinary() const {
    if (input.size() > 3) return false;
    // Several upstream pooling implementations narrow geometry to signed int.
    // Large but valid UInt32 geometry remains supported by the view composition.
    for (const auto* shape : {&input, &kernel, &stride, &padding, &output})
      for (const auto dim : *shape)
        if (dim > INT32_MAX) return false;
    return kernel_volume <= INT32_MAX;
  }

  Shape kernel_shape(int64_t in_channels, int64_t out_channels, bool transposed) const {
    Shape shape = transposed ? Shape{in_channels, out_channels}
                             : Shape{out_channels, in_channels};
    shape.insert(shape.end(), kernel.begin(), kernel.end());
    return shape;
  }
};

// For one kernel offset, valid input/output pairs form a Cartesian product of intervals.
// Strided views express that product without allocating an im2col tensor or device index grid.
struct Window {
  Shape first, last, base_first, base_last, step;
  int64_t elements = 1;

  Window(const Shape& base, const Shape& windows, const Geometry& g, size_t offset)
      : step(g.stride) {
    Shape coordinate(g.kernel.size());
    for (size_t a = g.kernel.size(); a-- > 0;) {
      coordinate[a] = offset % g.kernel[a];
      offset /= g.kernel[a];
    }
    for (size_t a = 0; a < base.size(); ++a) {
      const int64_t shift = coordinate[a] - g.padding[a];
      const int64_t lo = std::max<int64_t>(0, ceil_div_i64(-shift, g.stride[a]));
      const int64_t hi = std::min<int64_t>(
          windows[a], floor_div_i64(base[a] - 1 - shift, g.stride[a]) + 1);
      if (hi <= lo) {
        elements = 0;
        return;
      }
      first.push_back(lo);
      last.push_back(hi);
      base_first.push_back(lo * g.stride[a] + shift);
      base_last.push_back((hi - 1) * g.stride[a] + shift + 1);
      elements *= hi - lo;
    }
  }

  at::Tensor base_view(at::Tensor value) const {
    for (size_t a = 0; a < first.size(); ++a)
      value = value.slice(a + 1, base_first[a], base_last[a], step[a]);
    return value;
  }

  at::Tensor window_view(at::Tensor value) const {
    for (size_t a = 0; a < first.size(); ++a)
      value = value.slice(a + 1, first[a], last[a]);
    return value;
  }

  at::Tensor input_indices(const Geometry& g, const at::TensorOptions& options) const {
    const auto index_options = options.dtype(at::kLong);
    auto indices = at::zeros({}, index_options);
    for (size_t a = 0; a < first.size(); ++a) {
      Shape axis_shape(first.size() + 1, 1);
      axis_shape[a + 1] = last[a] - first[a];
      const auto position = (at::arange(last[a] - first[a], index_options) * step[a] +
                             base_first[a]).reshape(axis_shape);
      indices = indices * g.input[a] + position;
    }
    return indices;
  }
};

static void add_matrix(at::Tensor destination, const at::Tensor& matrix) {
  destination.add_(matrix.reshape(destination.sizes()));
}

static at::Tensor bias_output(const at::Tensor& bias, const Geometry& g, int64_t channels) {
  Shape singleton(g.input.size() + 1, 1);
  singleton[0] = channels;
  return bias.reshape(singleton).expand(with_channels(channels, g.output)).clone();
}

static at::Tensor convolution_composed(
    const at::Tensor& x, const at::Tensor& weight, const at::Tensor& bias,
    const Geometry& g, int64_t in_channels, int64_t out_channels, bool transposed) {
  auto y = bias_output(bias, g, out_channels);
  const auto w = weight.reshape(
      {transposed ? in_channels : out_channels,
       transposed ? out_channels : in_channels, static_cast<int64_t>(g.kernel_volume)});
  for (size_t k = 0; k < g.kernel_volume; ++k) {
    const Window window(transposed ? g.output : g.input,
                        transposed ? g.input : g.output, g, k);
    if (window.elements == 0) continue;
    const auto wk = w.select(2, k);
    if (transposed) {
      const auto input = window.window_view(x).reshape({in_channels, window.elements});
      add_matrix(window.base_view(y), at::mm(wk.t(), input));
    } else {
      const auto input = window.base_view(x).reshape({in_channels, window.elements});
      add_matrix(window.window_view(y), at::mm(wk, input));
    }
  }
  return y;
}

static Gradients convolution_backward_composed(
    const at::Tensor& x, const at::Tensor& weight, const at::Tensor& grad,
    const Geometry& g, int64_t in_channels, int64_t out_channels, bool transposed) {
  auto dx = at::zeros_like(x);
  auto dw = at::zeros_like(weight);
  const auto w = weight.reshape(
      {transposed ? in_channels : out_channels,
       transposed ? out_channels : in_channels, static_cast<int64_t>(g.kernel_volume)});
  auto dw_flat = dw.reshape(w.sizes());
  for (size_t k = 0; k < g.kernel_volume; ++k) {
    const Window window(transposed ? g.output : g.input,
                        transposed ? g.input : g.output, g, k);
    if (window.elements == 0) continue;
    if (transposed) {
      const auto input = window.window_view(x).reshape({in_channels, window.elements});
      const auto dy = window.base_view(grad).reshape({out_channels, window.elements});
      dw_flat.select(2, k).copy_(at::mm(input, dy.t()));
      add_matrix(window.window_view(dx), at::mm(w.select(2, k), dy));
    } else {
      const auto input = window.base_view(x).reshape({in_channels, window.elements});
      const auto dy = window.window_view(grad).reshape({out_channels, window.elements});
      dw_flat.select(2, k).copy_(at::mm(dy, input.t()));
      add_matrix(window.base_view(dx), at::mm(w.select(2, k).t(), dy));
    }
  }
  return {dw, grad.reshape({out_channels, static_cast<int64_t>(g.output_volume)}).sum(1), dx};
}

static at::Tensor convolution_forward(
    const at::Tensor& input, const at::Tensor& kernel, const at::Tensor& bias,
    const Geometry& g, int64_t in_channels, int64_t out_channels, bool transposed) {
  if (out_channels == 0 || g.output_volume == 0) return at::empty({0}, input.options());
  if (input.numel() == 0) return bias_output(bias, g, out_channels);
  const auto x = input.reshape(with_channels(in_channels, g.input));
  const auto w = kernel.reshape(g.kernel_shape(in_channels, out_channels, transposed));
  if (!g.ordinary())
    return convolution_composed(x, w, bias, g, in_channels, out_channels, transposed);
  const Shape dilation(g.input.size(), 1), output_padding(g.input.size(), 0);
  return at::convolution(x.unsqueeze(0), w, bias, g.stride, g.padding, dilation,
                         transposed, output_padding, 1).squeeze(0);
}

static Gradients convolution_backward(
    const at::Tensor& input, const at::Tensor& kernel, const at::Tensor& gradient,
    const Geometry& g, int64_t in_channels, int64_t out_channels, bool transposed) {
  if (gradient.numel() == 0 || input.numel() == 0 || kernel.numel() == 0) {
    auto db = gradient.numel() == 0 ? at::zeros({out_channels}, gradient.options())
        : gradient.reshape({out_channels, static_cast<int64_t>(g.output_volume)}).sum(1);
    return {at::zeros_like(kernel), db, at::zeros_like(input)};
  }
  const auto x = input.reshape(with_channels(in_channels, g.input));
  const auto w = kernel.reshape(g.kernel_shape(in_channels, out_channels, transposed));
  const auto grad = gradient.reshape(with_channels(out_channels, g.output));
  if (!g.ordinary())
    return convolution_backward_composed(x, w, grad, g, in_channels, out_channels, transposed);
  const Shape dilation(g.input.size(), 1), output_padding(g.input.size(), 0);
  const Shape bias_shape{out_channels};
  const auto result = at::convolution_backward(
      grad.unsqueeze(0), x.unsqueeze(0), w, at::IntArrayRef(bias_shape), g.stride, g.padding,
      dilation, transposed, output_padding, 1, std::array<bool, 3>{true, true, true});
  // ATen returns (input, weight, bias); the existing Lean ABI expects (weight, bias, input).
  return {std::get<1>(result), std::get<2>(result), std::get<0>(result).squeeze(0)};
}

struct PoolArguments {
  Shape kernel, stride, padding, dilation;
  explicit PoolArguments(const Geometry& g)
      : kernel(g.kernel), stride(g.stride), padding(g.padding), dilation(g.input.size(), 1) {
    // ATen exposes 1-D pooling backward through its 2-D operators.
    if (kernel.size() == 1) {
      kernel.insert(kernel.begin(), 1);
      stride.insert(stride.begin(), 1);
      padding.insert(padding.begin(), 0);
      dilation.insert(dilation.begin(), 1);
    }
  }
  at::Tensor batched(const at::Tensor& x, const Geometry& g) const {
    return g.input.size() == 1 ? x.unsqueeze(0).unsqueeze(2) : x.unsqueeze(0);
  }
  at::Tensor unbatched(const at::Tensor& x, const Geometry& g) const {
    return g.input.size() == 1 ? x.squeeze(0).squeeze(1) : x.squeeze(0);
  }
};

static bool ordinary_pool(const Geometry& g, bool average) {
  if (!g.ordinary()) return false;
  for (size_t a = 0; a < g.input.size(); ++a) {
    // The upstream kernels still use int for some padded window endpoints.
    if (g.input[a] + 2 * g.padding[a] > INT32_MAX) return false;
    // avg_pool3d's shape check excludes valid padded windows accepted by TorchLean.
    if (average && g.input.size() == 3 && g.input[a] < g.kernel[a]) return false;
  }
  // max_pool3d's flattened spatial index is int in the pinned CUDA implementation.
  return g.input.size() != 3 ||
         (g.input_volume <= INT32_MAX && g.output_volume <= INT32_MAX);
}

static bool ordinary_pool_backward(
    const Geometry& g, bool average, const at::Tensor& grad) {
  if (!ordinary_pool(g, average)) return false;
  // Both upstream CUDA 3-D backward operators reject deterministic-algorithms mode.
  // The offset composition updates disjoint strided views within each step, in fixed order.
  return g.input.size() != 3 || !grad.is_cuda() ||
         !at::globalContext().deterministicAlgorithms();
}

// First valid source position in each window, flattened over the spatial axes only.
static at::Tensor first_pool_indices(const Geometry& g, const at::TensorOptions& options) {
  auto indices = at::zeros(g.output, options.dtype(at::kLong));
  for (size_t a = 0; a < g.input.size(); ++a) {
    Shape axis_shape(g.input.size(), 1);
    axis_shape[a] = g.output[a];
    const auto position = (at::arange(g.output[a], options.dtype(at::kLong)) *
                           g.stride[a] - g.padding[a]).clamp_min(0).reshape(axis_shape);
    indices = indices * g.input[a] + position;
  }
  return indices;
}

static std::tuple<at::Tensor, at::Tensor> max_pool_ordinary(
    const at::Tensor& x, const Geometry& g) {
  const PoolArguments args(g);
  // Later NaNs are ignored by TorchLean's strict comparison. The first valid NaN is retained.
  const auto cleaned = at::where(x.isnan(), -std::numeric_limits<float>::infinity(), x);
  const auto input = args.batched(cleaned, g);
  const auto result = g.input.size() == 3
      ? at::max_pool3d_with_indices(input, args.kernel, args.stride, args.padding,
                                   args.dilation, false)
      : at::max_pool2d_with_indices(input, args.kernel, args.stride, args.padding,
                                   args.dilation, false);
  const auto values = args.unbatched(std::get<0>(result), g);
  const auto indices = args.unbatched(std::get<1>(result), g);
  const auto first = first_pool_indices(g, x.options());
  const auto first_values = x.reshape({x.size(0), static_cast<int64_t>(g.input_volume)})
      .index_select(1, first.reshape({-1})).reshape(values.sizes());
  // Also repair all-negative-infinity windows, whose upstream index can be a sentinel.
  const auto use_first = first_values.isnan().logical_or(
      values.eq(-std::numeric_limits<float>::infinity()));
  return {at::where(use_first, first_values, values),
          at::where(use_first, first.unsqueeze(0).expand_as(indices), indices)};
}

static std::tuple<at::Tensor, at::Tensor> max_pool_composed(
    const at::Tensor& x, const Geometry& g) {
  auto output = at::zeros(with_channels(x.size(0), g.output), x.options());
  auto selected = at::full(output.sizes(), -1, x.options().dtype(at::kLong));
  for (size_t k = 0; k < g.kernel_volume; ++k) {
    const Window window(g.input, g.output, g, k);
    if (window.elements == 0) continue;
    auto best = window.window_view(output);
    auto choice = window.window_view(selected);
    const auto value = window.base_view(x);
    const auto take = choice.lt(0).logical_or(value.gt(best));
    best.copy_(at::where(take, value, best));
    // Store source indices, as upstream does. A valid source index fits Int64 even when
    // a mostly padded kernel's volume exceeds that range.
    choice.copy_(at::where(take, window.input_indices(g, x.options()), choice));
  }
  return {output, selected};
}

static at::Tensor max_pool_forward(const at::Tensor& input, const Geometry& g, int64_t channels) {
  if (channels == 0 || g.output_volume == 0) return at::empty({0}, input.options());
  const auto x = input.reshape(with_channels(channels, g.input));
  return std::get<0>(ordinary_pool(g, false) ? max_pool_ordinary(x, g) : max_pool_composed(x, g));
}

static at::Tensor max_pool_backward(
    const at::Tensor& input, const at::Tensor& gradient, const Geometry& g, int64_t channels) {
  if (gradient.numel() == 0) return at::zeros_like(input);
  const auto x = input.reshape(with_channels(channels, g.input));
  const auto grad = gradient.reshape(with_channels(channels, g.output));
  if (ordinary_pool_backward(g, false, grad)) {
    const PoolArguments args(g);
    const auto indices = args.batched(std::get<1>(max_pool_ordinary(x, g)), g);
    const auto dy = args.batched(grad, g), source = args.batched(x, g);
    const auto result = g.input.size() == 3
        ? at::max_pool3d_with_indices_backward(
              dy, source, args.kernel, args.stride, args.padding, args.dilation, false, indices)
        : at::max_pool2d_with_indices_backward(
              dy, source, args.kernel, args.stride, args.padding, args.dilation, false, indices);
    return args.unbatched(result, g);
  }
  const auto selected = std::get<1>(
      ordinary_pool(g, false) ? max_pool_ordinary(x, g) : max_pool_composed(x, g));
  auto dx = at::zeros_like(x);
  for (size_t k = 0; k < g.kernel_volume; ++k) {
    const Window window(g.input, g.output, g, k);
    if (window.elements == 0) continue;
    window.base_view(dx).add_(at::where(
        window.window_view(selected).eq(window.input_indices(g, x.options())),
        window.window_view(grad), 0.0));
  }
  return dx;
}

static at::Tensor avg_pool_forward(const at::Tensor& input, const Geometry& g, int64_t channels) {
  if (channels == 0 || g.output_volume == 0) return at::empty({0}, input.options());
  const auto x = input.reshape(with_channels(channels, g.input));
  if (ordinary_pool(g, true)) {
    const PoolArguments args(g);
    const auto source = args.batched(x, g);
    const auto result = g.input.size() == 3
        ? at::avg_pool3d(source, args.kernel, args.stride, args.padding, false, true, std::nullopt)
        : at::avg_pool2d(source, args.kernel, args.stride, args.padding, false, true, std::nullopt);
    return args.unbatched(result, g);
  }
  auto output = at::zeros(with_channels(channels, g.output), x.options());
  for (size_t k = 0; k < g.kernel_volume; ++k) {
    const Window window(g.input, g.output, g, k);
    if (window.elements != 0) window.window_view(output).add_(window.base_view(x));
  }
  return output / static_cast<double>(g.kernel_volume);
}

static at::Tensor avg_pool_backward(
    const at::Tensor& gradient, const Geometry& g, int64_t channels) {
  const auto count = volume(with_channels(channels, g.input));
  if (gradient.numel() == 0) return at::zeros({count}, gradient.options());
  const auto grad = gradient.reshape(with_channels(channels, g.output));
  if (ordinary_pool_backward(g, true, grad)) {
    const PoolArguments args(g);
    // The upstream backward needs the input's shape, but never its values.
    const auto source = args.batched(at::empty(with_channels(channels, g.input), grad.options()), g);
    const auto dy = args.batched(grad, g);
    const auto result = g.input.size() == 3
        ? at::avg_pool3d_backward(
              dy, source, args.kernel, args.stride, args.padding, false, true, std::nullopt)
        : at::avg_pool2d_backward(
              dy, source, args.kernel, args.stride, args.padding, false, true, std::nullopt);
    return args.unbatched(result, g);
  }
  auto dx = at::zeros(with_channels(channels, g.input), gradient.options());
  const auto scaled = grad / static_cast<double>(g.kernel_volume);
  for (size_t k = 0; k < g.kernel_volume; ++k) {
    const Window window(g.input, g.output, g, k);
    if (window.elements != 0) window.base_view(dx).add_(window.window_view(scaled));
  }
  return dx;
}

static std::tuple<at::Tensor, at::Tensor> smooth_pool_statistics(
    const at::Tensor& x, const Geometry& g, float beta) {
  const auto output_shape = with_channels(x.size(0), g.output);
  auto pivot = at::full(output_shape, beta > 0 ? -std::numeric_limits<float>::infinity()
                                             : std::numeric_limits<float>::infinity(), x.options());
  auto values = at::zeros(output_shape, x.options());
  // Padding contributes the literal value zero, including to the pivot and denominator.
  for (size_t k = 0; k < g.kernel_volume; ++k) {
    values.zero_();
    const Window window(g.input, g.output, g, k);
    if (window.elements != 0) window.window_view(values).copy_(window.base_view(x));
    pivot = beta > 0 ? at::fmax(pivot, values) : at::fmin(pivot, values);
  }
  auto denominator = at::zeros_like(pivot);
  for (size_t k = 0; k < g.kernel_volume; ++k) {
    values.zero_();
    const Window window(g.input, g.output, g, k);
    if (window.elements != 0) window.window_view(values).copy_(window.base_view(x));
    // Subtract in input space first: beta*x can overflow even for a finite smooth maximum.
    denominator.add_(((values - pivot) * beta).exp());
  }
  return {pivot, denominator};
}

static at::Tensor smooth_pool_forward(
    const at::Tensor& input, const Geometry& g, int64_t channels, double beta) {
  // Match the existing forward ABI: empty output does not evaluate beta.
  if (channels == 0 || g.output_volume == 0) return at::empty({0}, input.options());
  const float b = checked_smoothmax_beta(beta, "torchlean smooth pool: beta must be finite and nonzero");
  const auto x = input.reshape(with_channels(channels, g.input));
  const auto statistics = smooth_pool_statistics(x, g, b);
  return std::get<0>(statistics) + std::get<1>(statistics).log() / b;
}

static at::Tensor smooth_pool_backward(
    const at::Tensor& input, const at::Tensor& gradient,
    const Geometry& g, int64_t channels, double beta) {
  const float b = checked_smoothmax_beta(beta, "torchlean smooth pool: beta must be finite and nonzero");
  if (gradient.numel() == 0) return at::zeros_like(input);
  const auto x = input.reshape(with_channels(channels, g.input));
  const auto grad = gradient.reshape(with_channels(channels, g.output));
  const auto statistics = smooth_pool_statistics(x, g, b);
  auto dx = at::zeros_like(x);
  for (size_t k = 0; k < g.kernel_volume; ++k) {
    const Window window(g.input, g.output, g, k);
    if (window.elements == 0) continue;
    const auto numerator = ((window.base_view(x) -
        window.window_view(std::get<0>(statistics))) * b).exp();
    const auto weight = numerator / window.window_view(std::get<1>(statistics));
    window.base_view(dx).add_(window.window_view(grad) * weight);
  }
  return dx;
}

#ifndef TORCHLEAN_CONV_POOL_TEST
static Geometry read_geometry(b_lean_obj_arg input, b_lean_obj_arg kernel,
                              b_lean_obj_arg stride, b_lean_obj_arg padding, Kind kind) {
  for (auto object : {input, kernel, stride, padding})
    require(lean_is_array(object), "torchlean conv/pool: expected shape arrays");
  return Geometry(dimensions(input, "torchlean conv/pool: bad input dimension"),
                  dimensions(kernel, "torchlean conv/pool: bad kernel dimension"),
                  dimensions(stride, "torchlean conv/pool: bad stride"),
                  dimensions(padding, "torchlean conv/pool: bad padding"), kind);
}

static const at::Tensor& checked_tensor(b_lean_obj_arg object, const Shape& shape) {
  const auto& value = tensor(object);
  require(value.numel() == volume(shape), "torchlean conv/pool: buffer size mismatch");
  return value;
}

static lean_obj_res convolution_ffi(
    b_lean_obj_arg input, b_lean_obj_arg kernel, b_lean_obj_arg bias_or_gradient,
    b_lean_obj_arg spatial, b_lean_obj_arg kernel_spatial,
    b_lean_obj_arg stride, b_lean_obj_arg padding, uint32_t in_channels, uint32_t out_channels,
    bool transposed, bool backward) {
  return invoke([&]() -> lean_obj_res {
    const auto g = read_geometry(spatial, kernel_spatial, stride, padding,
                                 transposed ? Kind::transposed : Kind::convolution);
    const auto& x = checked_tensor(input, with_channels(in_channels, g.input));
    const auto& w = checked_tensor(kernel, g.kernel_shape(in_channels, out_channels, transposed));
    const auto& other = checked_tensor(
        bias_or_gradient, backward ? with_channels(out_channels, g.output) : Shape{out_channels});
    // Check the output size even when this is a forward call with no output buffer yet.
    volume(with_channels(out_channels, g.output));
    if (!backward)
      return box(convolution_forward(x, w, other, g, in_channels, out_channels, transposed));
    const auto result = convolution_backward(x, w, other, g, in_channels, out_channels, transposed);
    return triple(std::get<0>(result), std::get<1>(result), std::get<2>(result));
  });
}

enum class Pool { maximum, average, smooth };

static lean_obj_res pooling_ffi(
    b_lean_obj_arg input, b_lean_obj_arg gradient, double beta,
    b_lean_obj_arg spatial, b_lean_obj_arg kernel, b_lean_obj_arg stride, b_lean_obj_arg padding,
    uint32_t channels, Pool pool, bool backward) {
  return invoke([&]() -> lean_obj_res {
    const auto g = read_geometry(spatial, kernel, stride, padding, Kind::pooling);
    volume(with_channels(channels, g.input));
    volume(with_channels(channels, g.output));
    if (backward) {
      const auto& grad = checked_tensor(gradient, with_channels(channels, g.output));
      if (pool == Pool::average) return box(avg_pool_backward(grad, g, channels));
      const auto& x = checked_tensor(input, with_channels(channels, g.input));
      return box(pool == Pool::maximum ? max_pool_backward(x, grad, g, channels)
                                      : smooth_pool_backward(x, grad, g, channels, beta));
    }
    const auto& x = checked_tensor(input, with_channels(channels, g.input));
    if (pool == Pool::maximum) return box(max_pool_forward(x, g, channels));
    if (pool == Pool::average) return box(avg_pool_forward(x, g, channels));
    return box(smooth_pool_forward(x, g, channels, beta));
  });
}
#endif

}  // namespace torchlean::conv_pool

#ifndef TORCHLEAN_CONV_POOL_TEST
extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_conv_fwd(
    b_lean_obj_arg input, b_lean_obj_arg kernel, b_lean_obj_arg bias,
    b_lean_obj_arg spatial, b_lean_obj_arg kernel_spatial,
    b_lean_obj_arg stride, b_lean_obj_arg padding, uint32_t in_channels, uint32_t out_channels) {
  return torchlean::conv_pool::convolution_ffi(
      input, kernel, bias, spatial, kernel_spatial, stride, padding,
      in_channels, out_channels, false, false);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_conv_bwd(
    b_lean_obj_arg input, b_lean_obj_arg kernel, b_lean_obj_arg gradient,
    b_lean_obj_arg spatial, b_lean_obj_arg kernel_spatial,
    b_lean_obj_arg stride, b_lean_obj_arg padding, uint32_t in_channels, uint32_t out_channels) {
  return torchlean::conv_pool::convolution_ffi(
      input, kernel, gradient, spatial, kernel_spatial, stride, padding,
      in_channels, out_channels, false, true);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_convtranspose_fwd(
    b_lean_obj_arg input, b_lean_obj_arg kernel, b_lean_obj_arg bias,
    b_lean_obj_arg spatial, b_lean_obj_arg kernel_spatial,
    b_lean_obj_arg stride, b_lean_obj_arg padding, uint32_t in_channels, uint32_t out_channels) {
  return torchlean::conv_pool::convolution_ffi(
      input, kernel, bias, spatial, kernel_spatial, stride, padding,
      in_channels, out_channels, true, false);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_convtranspose_bwd(
    b_lean_obj_arg input, b_lean_obj_arg kernel, b_lean_obj_arg gradient,
    b_lean_obj_arg spatial, b_lean_obj_arg kernel_spatial,
    b_lean_obj_arg stride, b_lean_obj_arg padding, uint32_t in_channels, uint32_t out_channels) {
  return torchlean::conv_pool::convolution_ffi(
      input, kernel, gradient, spatial, kernel_spatial, stride, padding,
      in_channels, out_channels, true, true);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_maxpool_fwd(
    b_lean_obj_arg input, b_lean_obj_arg spatial, b_lean_obj_arg kernel,
    b_lean_obj_arg stride, b_lean_obj_arg padding, uint32_t channels) {
  return torchlean::conv_pool::pooling_ffi(
      input, nullptr, 0, spatial, kernel, stride, padding, channels,
      torchlean::conv_pool::Pool::maximum, false);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_maxpool_bwd(
    b_lean_obj_arg input, b_lean_obj_arg gradient, b_lean_obj_arg spatial, b_lean_obj_arg kernel,
    b_lean_obj_arg stride, b_lean_obj_arg padding, uint32_t channels) {
  return torchlean::conv_pool::pooling_ffi(
      input, gradient, 0, spatial, kernel, stride, padding, channels,
      torchlean::conv_pool::Pool::maximum, true);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_avgpool_fwd(
    b_lean_obj_arg input, b_lean_obj_arg spatial, b_lean_obj_arg kernel,
    b_lean_obj_arg stride, b_lean_obj_arg padding, uint32_t channels) {
  return torchlean::conv_pool::pooling_ffi(
      input, nullptr, 0, spatial, kernel, stride, padding, channels,
      torchlean::conv_pool::Pool::average, false);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_avgpool_bwd(
    b_lean_obj_arg gradient, b_lean_obj_arg spatial, b_lean_obj_arg kernel,
    b_lean_obj_arg stride, b_lean_obj_arg padding, uint32_t channels) {
  return torchlean::conv_pool::pooling_ffi(
      nullptr, gradient, 0, spatial, kernel, stride, padding, channels,
      torchlean::conv_pool::Pool::average, true);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_smooth_maxpool_fwd(
    b_lean_obj_arg input, double beta, b_lean_obj_arg spatial, b_lean_obj_arg kernel,
    b_lean_obj_arg stride, b_lean_obj_arg padding, uint32_t channels) {
  return torchlean::conv_pool::pooling_ffi(
      input, nullptr, beta, spatial, kernel, stride, padding, channels,
      torchlean::conv_pool::Pool::smooth, false);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_smooth_maxpool_bwd(
    b_lean_obj_arg input, b_lean_obj_arg gradient, double beta,
    b_lean_obj_arg spatial, b_lean_obj_arg kernel,
    b_lean_obj_arg stride, b_lean_obj_arg padding, uint32_t channels) {
  return torchlean::conv_pool::pooling_ffi(
      input, gradient, beta, spatial, kernel, stride, padding, channels,
      torchlean::conv_pool::Pool::smooth, true);
}
#endif

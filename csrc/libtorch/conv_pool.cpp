// Convolution and pooling over TorchLean CUDA buffers: grouped and transposed convolutions and
// max, average and smooth max pools, each with its explicit backward call.
#include "torchlean_libtorch.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <tuple>
#include <vector>

// The ATen schemas we call are those of the SDKs under "Tested SDK versions" in README.md.
// Backward calls return explicit cotangents to the Lean tape and record no autograd graph.
namespace torchlean::conv_pool {

// The LibTorch backend follows PyTorch's convolution and pooling families: one, two, or three
// spatial dimensions. Lean specifies higher ranks too, and the native backend does not emulate
// kernels that LibTorch does not provide.
constexpr size_t kMaxSpatialRank = 3;

static uint32_t outDim(uint32_t in, uint32_t k, uint32_t stride, uint32_t padding) {
  torchlean::require(stride != 0, "LibTorch conv/pool: stride must be > 0");
  if (k == 0) return 0;
  // Invalid geometry has no windows, so a saturated subtraction must not become a phantom window.
  const uint64_t inPad = uint64_t{in} + 2 * uint64_t{padding};
  if (inPad < k) return 0;
  const uint64_t out = (inPad - k) / stride + 1;
  torchlean::require(out <= UINT32_MAX, "LibTorch conv/pool: outDim overflow");
  return static_cast<uint32_t>(out);
}

// N-D pooling follows `Pool.outSpatial`: empty inputs and padding beyond half the kernel
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

// Validate in the input dtype: a finite nonzero host coefficient can overflow or
// underflow when a binary32 buffer requires conversion.
static double checked_smoothmax_beta(double beta, at::ScalarType dtype, const char* msg) {
  const double betaF = rounded(beta, dtype);
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
enum class Kind { transposed, pooling };

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
  size_t kernel_volume, output_volume;

  Geometry(Shape in, Shape k, Shape s, Shape p, Kind kind)
      : input(std::move(in)), kernel(std::move(k)),
        stride(std::move(s)), padding(std::move(p)) {
    require(!input.empty() && input.size() <= kMaxSpatialRank,
            "torchlean conv/pool: spatial rank must be one, two, or three");
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
                                                : poolOutDim(i, w, s0, p0));
    }
    // Validate spatial products even when a channel count subsequently makes the buffer empty.
    // Keep these unsigned: zero-channel buffers can carry larger spatial metadata than
    // a nonempty ATen tensor can represent. Validate actual buffer sizes separately.
    spatial_volume(input);
    kernel_volume = spatial_volume(kernel);
    output_volume = spatial_volume(output);
  }

  // ATen's transposed-convolution weight layout (in_channels, out_channels, kernel...).
  Shape kernel_shape(int64_t in_channels, int64_t out_channels) const {
    Shape shape{in_channels, out_channels};
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

};

// Without input channels every window is empty: each output is the empty sum `+0` plus the bias,
// which is `+0` for a `-0` bias.
static at::Tensor bias_output(const at::Tensor& bias, const Geometry& g, int64_t channels) {
  Shape singleton(g.input.size() + 1, 1);
  singleton[0] = channels;
  return at::zeros(with_channels(channels, g.output), bias.options()) + bias.reshape(singleton);
}

static at::Tensor transposed_convolution_forward(
    const at::Tensor& input, const at::Tensor& kernel, const at::Tensor& bias,
    const Geometry& g, int64_t in_channels, int64_t out_channels) {
  if (out_channels == 0 || g.output_volume == 0) return at::empty({0}, input.options());
  if (input.numel() == 0) return bias_output(bias, g, out_channels);
  const auto x = input.reshape(with_channels(in_channels, g.input));
  const auto w = kernel.reshape(g.kernel_shape(in_channels, out_channels));
  const Shape dilation(g.input.size(), 1), output_padding(g.input.size(), 0);
  return at::convolution(x.unsqueeze(0), w, bias, g.stride, g.padding, dilation,
                         /*transposed=*/true, output_padding, 1).squeeze(0);
}

static Gradients transposed_convolution_backward(
    const at::Tensor& input, const at::Tensor& kernel, const at::Tensor& gradient,
    const Geometry& g, int64_t in_channels, int64_t out_channels) {
  if (gradient.numel() == 0 || input.numel() == 0 || kernel.numel() == 0) {
    auto db = gradient.numel() == 0 ? at::zeros({out_channels}, gradient.options())
        : gradient.reshape({out_channels, static_cast<int64_t>(g.output_volume)}).sum(1);
    return {at::zeros_like(kernel), db, at::zeros_like(input)};
  }
  const auto x = input.reshape(with_channels(in_channels, g.input));
  const auto w = kernel.reshape(g.kernel_shape(in_channels, out_channels));
  const auto grad = gradient.reshape(with_channels(out_channels, g.output));
  const Shape dilation(g.input.size(), 1), output_padding(g.input.size(), 0);
  const Shape bias_shape{out_channels};
  const auto result = at::convolution_backward(
      grad.unsqueeze(0), x.unsqueeze(0), w, at::IntArrayRef(bias_shape), g.stride, g.padding,
      dilation, /*transposed=*/true, output_padding, 1, std::array<bool, 3>{true, true, true});
  // ATen returns (input, weight, bias), and the Lean ABI is (weight, bias, input).
  return {std::get<1>(result), std::get<2>(result), std::get<0>(result).squeeze(0)};
}

struct PoolArguments {
  Shape kernel, stride, padding, dilation, explicit_padding;
  explicit PoolArguments(const Geometry& g, bool average = false)
      : kernel(g.kernel), stride(g.stride), padding(g.padding), dilation(g.input.size(), 1) {
    // avg_pool3d rejects a small unpadded input even when its padded windows are valid.
    // Materializing those zeros keeps the full-window divisor and uses the same ATen operator.
    if (average && kernel.size() == 3) {
      bool small = false;
      for (size_t axis = 0; axis < kernel.size(); ++axis)
        small = small || g.input[axis] < kernel[axis];
      if (small) {
        for (size_t axis = kernel.size(); axis-- > 0;) {
          explicit_padding.push_back(padding[axis]);
          explicit_padding.push_back(padding[axis]);
        }
        padding.assign(kernel.size(), 0);
      }
    }
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

static std::tuple<at::Tensor, at::Tensor> max_pool_with_indices(
    const at::Tensor& x, const Geometry& g) {
  const PoolArguments args(g);
  const auto pool = [&](const at::Tensor& t) {
    const auto input = args.batched(t, g);
    const auto result = g.input.size() == 3
        ? at::max_pool3d_with_indices(input, args.kernel, args.stride, args.padding,
                                     args.dilation, false)
        : at::max_pool2d_with_indices(input, args.kernel, args.stride, args.padding,
                                     args.dilation, false);
    return std::make_tuple(args.unbatched(std::get<0>(result), g),
                           args.unbatched(std::get<1>(result), g));
  };
  const auto [value, index] = pool(x);
  // ATen keeps the first of a `-0` and `0` tie. The IEEE 754-2019 `maximum` is `0`, as in
  // `poolWins`, so a window whose maximum is zero and that holds a `0` selects its first `0`.
  const auto positive_zero = at::logical_and(at::eq(x, 0), at::logical_not(at::signbit(x)));
  const auto [has_positive_zero, positive_index] = pool(positive_zero.to(x.scalar_type()));
  const auto take = at::logical_and(at::eq(value, 0), at::eq(has_positive_zero, 1));
  return {at::where(take, at::zeros_like(value), value), at::where(take, positive_index, index)};
}

static at::Tensor max_pool_forward(const at::Tensor& input, const Geometry& g, int64_t channels) {
  if (channels == 0 || g.output_volume == 0) return at::empty({0}, input.options());
  const auto x = input.reshape(with_channels(channels, g.input));
  return std::get<0>(max_pool_with_indices(x, g));
}

static at::Tensor max_pool_backward(
    const at::Tensor& input, const at::Tensor& gradient, const Geometry& g, int64_t channels) {
  if (gradient.numel() == 0) return at::zeros_like(input);
  const auto x = input.reshape(with_channels(channels, g.input));
  const auto grad = gradient.reshape(with_channels(channels, g.output));
  const PoolArguments args(g);
  const auto indices = args.batched(std::get<1>(max_pool_with_indices(x, g)), g);
  const auto dy = args.batched(grad, g), source = args.batched(x, g);
  const auto result = g.input.size() == 3
      ? at::max_pool3d_with_indices_backward(
            dy, source, args.kernel, args.stride, args.padding, args.dilation, false, indices)
      : at::max_pool2d_with_indices_backward(
            dy, source, args.kernel, args.stride, args.padding, args.dilation, false, indices);
  return args.unbatched(result, g);
}

static at::Tensor avg_pool_forward(const at::Tensor& input, const Geometry& g, int64_t channels) {
  if (channels == 0 || g.output_volume == 0) return at::empty({0}, input.options());
  const auto x = input.reshape(with_channels(channels, g.input));
  const PoolArguments args(g, true);
  auto source = args.batched(x, g);
  if (!args.explicit_padding.empty())
    source = at::constant_pad_nd(source, args.explicit_padding, 0);
  const auto result = g.input.size() == 3
      ? at::avg_pool3d(source, args.kernel, args.stride, args.padding, false, true, std::nullopt)
      : at::avg_pool2d(source, args.kernel, args.stride, args.padding, false, true, std::nullopt);
  return args.unbatched(result, g);
}

static at::Tensor avg_pool_backward(
    const at::Tensor& gradient, const Geometry& g, int64_t channels) {
  const auto count = volume(with_channels(channels, g.input));
  if (gradient.numel() == 0) return at::zeros({count}, gradient.options());
  const auto grad = gradient.reshape(with_channels(channels, g.output));
  const PoolArguments args(g, true);
  // The upstream backward needs the input's shape, but never its values.
  auto input_shape = with_channels(channels, g.input);
  if (!args.explicit_padding.empty())
    for (size_t axis = 0; axis < g.input.size(); ++axis)
      input_shape[axis + 1] += 2 * g.padding[axis];
  const auto source = args.batched(at::empty(input_shape, grad.options()), g);
  const auto dy = args.batched(grad, g);
  auto result = g.input.size() == 3
      ? at::avg_pool3d_backward(
            dy, source, args.kernel, args.stride, args.padding, false, true, std::nullopt)
      : at::avg_pool2d_backward(
            dy, source, args.kernel, args.stride, args.padding, false, true, std::nullopt);
  if (!args.explicit_padding.empty())
    for (size_t axis = 0; axis < g.input.size(); ++axis)
      result = result.narrow(axis + 2, g.padding[axis], g.input[axis]);
  return args.unbatched(result, g);
}

// Widen before subtraction, not before scaling each operand. The latter loses nearby
// differences, and a narrow subtraction can overflow for finite inputs of opposite signs.
static at::Tensor smooth_pool_exp(const at::Tensor& values, const at::Tensor& shift,
                                  double beta) {
  const auto dtype = values.scalar_type();
  if (dtype == at::kHalf || dtype == at::kBFloat16 || dtype == at::kFloat)
    return ((values.to(at::kDouble) - shift.to(at::kDouble)) * beta).exp().to(dtype);
  return ((values - shift) * beta).exp();
}

// The window pivots (maxima for positive beta, minima otherwise) and the sums of exponentials
// shifted by them. With zero_infinite_pivot, as Tensor.smoothMaxPoolShift, an infinite pivot
// shifts by zero instead, where inf - inf would be NaN, and the first result is that shift.
static std::tuple<at::Tensor, at::Tensor> smooth_pool_statistics(
    const at::Tensor& x, const Geometry& g, double beta, bool zero_infinite_pivot) {
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
  const auto shift = zero_infinite_pivot
      ? at::where(at::isinf(pivot), at::zeros_like(pivot), pivot) : pivot;
  auto denominator = at::zeros_like(shift);
  for (size_t k = 0; k < g.kernel_volume; ++k) {
    values.zero_();
    const Window window(g.input, g.output, g, k);
    if (window.elements != 0) window.window_view(values).copy_(window.base_view(x));
    denominator.add_(smooth_pool_exp(values, shift, beta));
  }
  return {shift, denominator};
}

static at::Tensor smooth_pool_forward(
    const at::Tensor& input, const Geometry& g, int64_t channels, double beta) {
  // The forward ABI does not check beta when the output is empty.
  if (channels == 0 || g.output_volume == 0) return at::empty({0}, input.options());
  const double b = checked_smoothmax_beta(beta, input.scalar_type(),
                                         "torchlean smooth pool: beta must be finite and nonzero");
  const auto x = input.reshape(with_channels(channels, g.input));
  const auto statistics = smooth_pool_statistics(x, g, b, true);
  // ATen's CUDA division by a CPU scalar multiplies by the scalar's rounded reciprocal, which
  // overflows at a subnormal `beta` such as binary32 `2^-149`, so a window whose sum is 1 would
  // give `0 * inf`, NaN. A device tensor divisor divides once, as `Tensor.smoothMaxPool` does.
  return std::get<0>(statistics) +
      at::div(std::get<1>(statistics).log(), at::scalar_tensor(b, x.options()));
}

static at::Tensor smooth_pool_backward(
    const at::Tensor& input, const at::Tensor& gradient,
    const Geometry& g, int64_t channels, double beta) {
  const double b = checked_smoothmax_beta(beta, input.scalar_type(),
                                         "torchlean smooth pool: beta must be finite and nonzero");
  if (gradient.numel() == 0) return at::zeros_like(input);
  const auto x = input.reshape(with_channels(channels, g.input));
  const auto grad = gradient.reshape(with_channels(channels, g.output));
  const auto statistics = smooth_pool_statistics(x, g, b, false);
  auto dx = at::zeros_like(x);
  for (size_t k = 0; k < g.kernel_volume; ++k) {
    const Window window(g.input, g.output, g, k);
    if (window.elements == 0) continue;
    const auto numerator = smooth_pool_exp(window.base_view(x),
        window.window_view(std::get<0>(statistics)), b);
    const auto weight = numerator / window.window_view(std::get<1>(statistics));
    window.base_view(dx).add_(window.window_view(grad) * weight);
  }
  return dx;
}

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

static lean_obj_res transposed_convolution_ffi(
    b_lean_obj_arg input, b_lean_obj_arg kernel, b_lean_obj_arg bias_or_gradient,
    b_lean_obj_arg spatial, b_lean_obj_arg kernel_spatial,
    b_lean_obj_arg stride, b_lean_obj_arg padding, uint32_t in_channels, uint32_t out_channels,
    bool backward) {
  return invoke([&]() -> lean_obj_res {
    const auto g = read_geometry(spatial, kernel_spatial, stride, padding, Kind::transposed);
    const auto& x = checked_tensor(input, with_channels(in_channels, g.input));
    const auto& w = checked_tensor(kernel, g.kernel_shape(in_channels, out_channels));
    const auto& other = checked_tensor(
        bias_or_gradient, backward ? with_channels(out_channels, g.output) : Shape{out_channels});
    // Check the output size even when this is a forward call with no output buffer yet.
    volume(with_channels(out_channels, g.output));
    if (!backward)
      return box(transposed_convolution_forward(x, w, other, g, in_channels, out_channels));
    const auto result = transposed_convolution_backward(x, w, other, g, in_channels, out_channels);
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

}  // namespace torchlean::conv_pool

namespace {

using torchlean::io_dimensions;
using torchlean::io_volume;

// A grouped convolution over the spatial suffix of an input `(batch..., C, spatial...)` with
// PyTorch's packed kernel `(O, C / groups, window...)`, its padding before and after each axis and
// its dilation. The batch axes are folded into one.
struct GroupedConvolution {
  std::vector<int64_t> input, weight, stride, before, after, dilation, batched, output;
  int64_t groups = 1;
  bool symmetric = true;
};

GroupedConvolution grouped_convolution(
    b_lean_obj_arg input_dims, b_lean_obj_arg kernel_dims, b_lean_obj_arg strides,
    b_lean_obj_arg padding_before, b_lean_obj_arg padding_after, b_lean_obj_arg dilations,
    uint32_t groups) {
  GroupedConvolution c;
  c.input = io_dimensions(input_dims);
  c.weight = io_dimensions(kernel_dims);
  c.stride = io_dimensions(strides);
  c.before = io_dimensions(padding_before);
  c.after = io_dimensions(padding_after);
  c.dilation = io_dimensions(dilations);
  c.groups = groups;
  const size_t rank = c.stride.size();
  TORCH_CHECK(rank >= 1 && rank <= torchlean::conv_pool::kMaxSpatialRank,
              "LibTorch conv: spatial rank must be one, two, or three");
  TORCH_CHECK(c.input.size() >= rank + 1 && c.weight.size() == rank + 2 &&
              c.before.size() == rank && c.after.size() == rank && c.dilation.size() == rank,
              "LibTorch conv: rank mismatch");
  const size_t channel_axis = c.input.size() - rank - 1;
  const int64_t in_channels = c.input[channel_axis], out_channels = c.weight[0];
  TORCH_CHECK(groups > 0 && in_channels % groups == 0 && out_channels % groups == 0 &&
              c.weight[1] == in_channels / groups, "LibTorch conv: invalid channel groups");
  const int64_t batch = io_volume(std::vector<int64_t>(c.input.begin(),
                                                       c.input.begin() + channel_axis));
  TORCH_CHECK(batch <= UINT32_MAX, "LibTorch conv: batch exceeds ABI");
  c.batched = {batch};
  c.batched.insert(c.batched.end(), c.input.begin() + channel_axis, c.input.end());
  c.output = {batch, out_channels};
  for (size_t axis = 0; axis < rank; ++axis) {
    const auto k = c.weight[axis + 2];
    TORCH_CHECK(k > 0 && c.stride[axis] > 0 && c.dilation[axis] > 0,
                "LibTorch conv: kernel, stride and dilation must be positive");
    const uint64_t effective = static_cast<uint64_t>(k - 1) * c.dilation[axis] + 1;
    const uint64_t padded = static_cast<uint64_t>(c.batched[2 + axis]) + c.before[axis] +
                            c.after[axis];
    const uint64_t extent = padded < effective ? 0 : (padded - effective) / c.stride[axis] + 1;
    TORCH_CHECK(extent <= UINT32_MAX, "LibTorch conv: output dimension exceeds ABI");
    c.output.push_back(static_cast<int64_t>(extent));
    c.symmetric = c.symmetric && c.before[axis] == c.after[axis];
  }
  TORCH_CHECK(io_volume(c.output) <= UINT32_MAX, "LibTorch conv: output exceeds ABI");
  return c;
}

// The input with its zero padding: ATen pads symmetrically itself, so only an asymmetric padding,
// or the padding of an input without entries, is materialized, in `constant_pad_nd`'s
// last-axis-first order.
at::Tensor padded_input(const GroupedConvolution& c, const at::Tensor& x,
                        std::vector<int64_t>& padding) {
  if (c.symmetric && x.numel() != 0) {
    padding = c.before;
    return x;
  }
  std::vector<int64_t> pads;
  for (size_t axis = c.stride.size(); axis-- > 0;) {
    pads.push_back(c.before[axis]);
    pads.push_back(c.after[axis]);
  }
  padding.assign(c.stride.size(), 0);
  return at::constant_pad_nd(x, pads, 0);
}

// The bias gradient: the output gradient summed over every axis but the feature axis.
at::Tensor bias_gradient(const at::Tensor& gradient, const GroupedConvolution& c) {
  if (gradient.numel() == 0) return at::zeros({c.weight[0]}, gradient.options());
  std::vector<int64_t> axes{0};
  for (size_t axis = 2; axis < c.output.size(); ++axis) axes.push_back(static_cast<int64_t>(axis));
  return gradient.reshape(c.output).sum(axes);
}

}  // namespace

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_conv_io(
    b_lean_obj_arg input, b_lean_obj_arg kernel, b_lean_obj_arg bias,
    b_lean_obj_arg input_dims, b_lean_obj_arg kernel_dims, b_lean_obj_arg strides,
    b_lean_obj_arg padding_before, b_lean_obj_arg padding_after, b_lean_obj_arg dilations,
    uint32_t groups) {
  return torchlean::io([&] {
    using namespace torchlean;
    const auto c = grouped_convolution(input_dims, kernel_dims, strides, padding_before,
                                       padding_after, dilations, groups);
    const auto& x_flat = io_tensor(input, c.input);
    const auto& w_flat = io_tensor(kernel, c.weight);
    const auto& b = io_tensor(bias, std::vector<int64_t>{c.weight[0]});
    if (io_volume(c.output) == 0) return box(at::empty({0}, x_flat.options()));
    // Without input channels every window is empty: each output is the empty sum `+0` plus the
    // bias, which is `+0` for a `-0` bias.
    if (c.weight[1] == 0) {
      std::vector<int64_t> singleton(c.output.size(), 1);
      singleton[1] = c.weight[0];
      return box(at::zeros(c.output, b.options()) + b.reshape(singleton));
    }
    std::vector<int64_t> padding;
    const auto x = padded_input(c, x_flat.reshape(c.batched), padding);
    const std::vector<int64_t> zeros(c.stride.size(), 0);
    const auto result = at::convolution(x, w_flat.reshape(c.weight), b, c.stride, padding,
                                        c.dilation, false, zeros, c.groups);
    TORCH_CHECK(result.sizes().vec() == c.output, "LibTorch conv: unexpected output geometry");
    return box(result);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_conv_bwd(
    b_lean_obj_arg input, b_lean_obj_arg kernel, b_lean_obj_arg gradient,
    b_lean_obj_arg input_dims, b_lean_obj_arg kernel_dims, b_lean_obj_arg strides,
    b_lean_obj_arg padding_before, b_lean_obj_arg padding_after, b_lean_obj_arg dilations,
    uint32_t groups) {
  return torchlean::invoke([&]() -> lean_obj_res {
    using namespace torchlean;
    const auto c = grouped_convolution(input_dims, kernel_dims, strides, padding_before,
                                       padding_after, dilations, groups);
    const auto& x_flat = io_tensor(input, c.input);
    const auto& w_flat = io_tensor(kernel, c.weight);
    const auto& g_flat = io_tensor(gradient, c.output);
    if (g_flat.numel() == 0 || c.weight[1] == 0)
      return triple(at::zeros_like(w_flat), bias_gradient(g_flat, c), at::zeros_like(x_flat));
    std::vector<int64_t> padding;
    const auto x = padded_input(c, x_flat.reshape(c.batched), padding);
    const std::vector<int64_t> zeros(c.stride.size(), 0), bias_shape{c.weight[0]};
    const auto result = at::convolution_backward(
        g_flat.reshape(c.output), x, w_flat.reshape(c.weight), at::IntArrayRef(bias_shape),
        c.stride, padding, c.dilation, false, zeros, c.groups,
        std::array<bool, 3>{true, true, true});
    auto dx = std::get<0>(result);
    if (!c.symmetric || x_flat.numel() == 0)
      for (size_t axis = 0; axis < c.stride.size(); ++axis)
        dx = dx.narrow(static_cast<int64_t>(axis + 2), c.before[axis], c.batched[axis + 2]);
    // ATen returns (input, weight, bias), and the Lean ABI is (weight, bias, input).
    return triple(std::get<1>(result), std::get<2>(result), dx);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_convtranspose_fwd(
    b_lean_obj_arg input, b_lean_obj_arg kernel, b_lean_obj_arg bias,
    b_lean_obj_arg spatial, b_lean_obj_arg kernel_spatial,
    b_lean_obj_arg stride, b_lean_obj_arg padding, uint32_t in_channels, uint32_t out_channels) {
  return torchlean::conv_pool::transposed_convolution_ffi(
      input, kernel, bias, spatial, kernel_spatial, stride, padding,
      in_channels, out_channels, false);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_convtranspose_bwd(
    b_lean_obj_arg input, b_lean_obj_arg kernel, b_lean_obj_arg gradient,
    b_lean_obj_arg spatial, b_lean_obj_arg kernel_spatial,
    b_lean_obj_arg stride, b_lean_obj_arg padding, uint32_t in_channels, uint32_t out_channels) {
  return torchlean::conv_pool::transposed_convolution_ffi(
      input, kernel, gradient, spatial, kernel_spatial, stride, padding,
      in_channels, out_channels, true);
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

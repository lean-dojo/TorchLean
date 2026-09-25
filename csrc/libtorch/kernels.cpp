#include "torchlean_libtorch.h"

#include <ATen/Context.h>
#include <algorithm>
#include <limits>
#include <optional>

// These functions evaluate TorchLean's buffer operations and selected VJPs. The
// Lean tape owns differentiation; invoke() disables native graph recording.
namespace {

using namespace torchlean;
using at::Tensor;

constexpr int64_t kMaxRank = 8;
constexpr double kNegativeInfinity = -std::numeric_limits<double>::infinity();

int64_t element_count(at::IntArrayRef shape) {
  int64_t count = 1;
  for (int64_t dim : shape) {
    require(dim >= 0, "LibTorch kernels: negative dimension");
    require(dim == 0 || count <= std::numeric_limits<int64_t>::max() / dim,
            "LibTorch kernels: shape size overflow");
    count *= dim;
  }
  return count;
}

Tensor checked(b_lean_obj_arg object, at::IntArrayRef shape) {
  require(tensor(object).numel() == element_count(shape),
          "LibTorch kernels: buffer size does not match shape");
  return shaped(object, shape);
}

std::vector<int64_t> shape_array(b_lean_obj_arg object) {
  require(lean_is_array(const_cast<lean_object*>(object)),
          "LibTorch kernels: expected Array Nat");
  auto shape = dimensions(object, "LibTorch kernels: dimension exceeds UInt32");
  require(shape.size() <= kMaxRank, "LibTorch kernels: rank exceeds eight");
  element_count(shape);
  return shape;
}

std::vector<int64_t> index_array(b_lean_obj_arg object, uint32_t count) {
  require(lean_is_array(const_cast<lean_object*>(object)),
          "LibTorch kernels: expected Array Nat indices");
  require(lean_array_size(object) == count,
          "LibTorch kernels: indices.size mismatch");
  return dimensions(object, "LibTorch kernels: index exceeds UInt32");
}

Tensor device_indices(const std::vector<int64_t>& values, const Tensor& like) {
  // The transfer is blocking: no GPU operation may outlive this host vector.
  return at::tensor(values, at::TensorOptions().dtype(at::kLong))
      .to(like.device(), at::kLong, false, true);
}

// At PyTorch revision 0291f960b6 (a 2.12 nightly), segment_reduce on a rank-two input
// uses a sequential fold within each segment/column on both CPU and CUDA
// (aten/src/ATen/native/{cuda/SegmentReduce.cu,SegmentReduce.cpp}). Keep the
// trailing column dimension even for vectors: rank one selects a different
// CUDA reduction. SDK upgrades must recheck this implementation detail and run
// the base-order regression; ATen's public API does not promise this ordering.
Tensor ordered_segments(const Tensor& data, const std::vector<int64_t>& lengths,
                        double initial = 0.0) {
  require(data.dim() == 2, "LibTorch kernels: ordered segments need rank two");
  return at::segment_reduce(data, "sum", device_indices(lengths, data),
                            std::nullopt, std::nullopt, 0, true, initial);
}

Tensor ordered_columns(const Tensor& matrix) {
  if (matrix.size(0) == 0 || matrix.size(1) == 0) {
    return at::zeros({matrix.size(1)}, matrix.options());
  }
  return ordered_segments(matrix, {matrix.size(0)}).reshape({matrix.size(1)});
}

Tensor column_sum(const Tensor& matrix, bool ordered) {
  return ordered ? ordered_columns(matrix) : matrix.sum(0);
}

Tensor max_ignoring_nan(const Tensor& input, int64_t axis, bool keepdim = false) {
  return input.masked_fill(input.isnan(), kNegativeInfinity).amax({axis}, keepdim);
}

struct BroadcastShape {
  std::vector<int64_t> input;
  std::vector<int64_t> output;
  std::vector<int64_t> map;
  std::vector<int64_t> input_permutation;
  std::vector<int64_t> expanded_input;
};

BroadcastShape broadcast_shape(b_lean_obj_arg input, b_lean_obj_arg output,
                               b_lean_obj_arg map) {
  BroadcastShape result;
  result.input = shape_array(input);
  result.output = shape_array(output);
  require(lean_is_array(const_cast<lean_object*>(map)),
          "LibTorch kernels: expected Array Nat axis map");
  result.map = dimensions(map, "LibTorch kernels: axis map exceeds UInt32");
  require(result.map.size() == result.output.size(),
          "LibTorch kernels: axis map rank mismatch");
  std::vector<bool> seen(result.input.size(), false);
  for (size_t axis = 0; axis < result.output.size(); ++axis) {
    const int64_t mapped = result.map[axis];
    require(mapped <= static_cast<int64_t>(result.input.size()),
            "LibTorch kernels: broadcast axis out of range");
    if (mapped == 0) {
      result.expanded_input.push_back(1);
      continue;
    }
    const int64_t source = mapped - 1;
    require(!seen[source], "LibTorch kernels: repeated broadcast input axis");
    seen[source] = true;
    const int64_t dim = result.input[source];
    require(dim == 1 || dim == result.output[axis],
            "LibTorch kernels: incompatible broadcast dimension");
    result.input_permutation.push_back(source);
    result.expanded_input.push_back(dim);
  }
  require(std::all_of(seen.begin(), seen.end(), [](bool value) { return value; }),
          "LibTorch kernels: broadcast omits an input axis");
  return result;
}

Tensor gather_rows(const Tensor& input, const std::vector<int64_t>& indices) {
  const int64_t count = static_cast<int64_t>(indices.size());
  if (input.size(0) == 0 || input.size(1) == 0 || count == 0) {
    return at::zeros({count, input.size(1)}, input.options());
  }
  auto index = device_indices(indices, input);
  auto valid = index.lt(input.size(0));
  auto gathered = input.index_select(0, index.clamp_max(input.size(0) - 1));
  // Multiplication by zero would leak NaNs from the clamped source row.
  return gathered.masked_fill(valid.logical_not().unsqueeze(1), 0);
}

Tensor scatter_rows(const Tensor& base, const Tensor& values,
                    const std::vector<int64_t>& indices) {
  auto out = base.clone();
  if (base.numel() == 0 || indices.empty()) return out;
  std::vector<int64_t> source;
  source.reserve(indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    if (indices[i] < base.size(0)) source.push_back(static_cast<int64_t>(i));
  }
  if (source.empty()) return out;

  if (!at::globalContext().deterministicAlgorithms()) {
    std::vector<int64_t> destination;
    destination.reserve(source.size());
    for (int64_t i : source) destination.push_back(indices[i]);
    out.index_add_(0, device_indices(destination, base),
                   values.index_select(0, device_indices(source, values)));
    return out;
  }

  // The indices already live on the host. A stable sort groups updates while
  // preserving their original order; only touched rows enter the reduction.
  std::stable_sort(source.begin(), source.end(),
                   [&](int64_t a, int64_t b) { return indices[a] < indices[b]; });
  std::vector<int64_t> touched;
  std::vector<int64_t> lengths;
  for (int64_t i : source) {
    if (touched.empty() || touched.back() != indices[i]) {
      touched.push_back(indices[i]);
      lengths.push_back(1);  // Each segment begins with its base value.
    }
    ++lengths.back();
  }
  std::vector<int64_t> order;
  order.reserve(source.size() + touched.size());
  int64_t offset = 0;
  for (size_t group = 0; group < touched.size(); ++group) {
    order.push_back(static_cast<int64_t>(group));
    for (int64_t j = 1; j < lengths[group]; ++j) {
      order.push_back(static_cast<int64_t>(touched.size()) + offset++);
    }
  }
  auto destination = device_indices(touched, base);
  auto grouped = at::cat(
      {base.index_select(0, destination),
       values.index_select(0, device_indices(source, values))}, 0);
  grouped = grouped.index_select(0, device_indices(order, base));
  // -0 + base preserves either sign of zero. Starting from +0 would change a
  // negative-zero base before the first update. Untouched rows remain cloned.
  auto reduced = ordered_segments(grouped, lengths, -0.0);
  out.index_copy_(0, destination, reduced);
  return out;
}

// Mutate only freshly allocated spectra, never a borrowed Lean buffer.
void real_endpoints(Tensor& spectrum, int64_t length, int64_t axis) {
  auto imaginary = at::imag(spectrum);
  imaginary.select(axis, 0).zero_();
  if (length % 2 == 0) imaginary.select(axis, length / 2).zero_();
}

void spectral_shapes(uint32_t grid, uint32_t width, uint32_t modes) {
  require(grid > 0 && width > 0,
          "spectralConv1dRfft: grid and width must be positive");
  require(modes <= grid / 2 + 1,
          "spectralConv1dRfft: modes exceeds rfft frequency count");
  element_count({grid, width});
  element_count({modes, width, width});
}

// Evaluate h[t] = a[t] h[t-1] + b[t]. The parallel affine prefix has logarithmic
// launch depth, O(T*D) live workspace, and O(T*D*log(T)) arithmetic.
//
// Products use double precision so contractive coefficients do not underflow
// in float32 before multiplying a large state. Bias/state results are rounded
// to float32 at each composition. Reassociation can change rounding; it is not
// a bitwise reproduction of the old sequential CUDA recurrence.
//
// Expansive/nonfinite inputs, and a nonfinite parallel result, use the original
// recurrence order. That fallback avoids introducing product overflow or a
// different nonfinite propagation rule merely to obtain a parallel schedule.
Tensor affine_scan(const Tensor& a, const Tensor& b, const Tensor& initial) {
  const int64_t steps = b.size(0);
  if (steps == 0 || b.size(1) == 0) return at::empty_like(b);
  auto sequential = [&]() {
    auto result = at::empty_like(b);
    auto state = initial;
    for (int64_t t = 0; t < steps; ++t) {
      state = a.select(0, t) * state + b.select(0, t);
      result.select(0, t).copy_(state);
    }
    return result;
  };
  auto eligible = a.abs().le(1).all()
      .logical_and(b.isfinite().all()).logical_and(initial.isfinite().all());
  if (!eligible.item<bool>()) return sequential();

  auto coefficients = a.to(at::kDouble).clone();
  auto states = b.clone();
  states.select(0, 0).copy_(a.select(0, 0) * initial + b.select(0, 0));
  for (int64_t stride = 1; stride < steps; stride *= 2) {
    const int64_t count = steps - stride;
    auto right = coefficients.narrow(0, stride, count);
    auto next_states =
        (right * states.narrow(0, 0, count).to(at::kDouble)
         + states.narrow(0, stride, count).to(at::kDouble)).to(at::kFloat);
    auto next_coefficients = right * coefficients.narrow(0, 0, count);
    // Both right-hand sides are materialized before either source is changed.
    states.narrow(0, stride, count).copy_(next_states);
    coefficients.narrow(0, stride, count).copy_(next_coefficients);
  }
  return states.isfinite().all().item<bool>() ? states : sequential();
}

struct ScanInputs {
  Tensor a;
  Tensor b;
  Tensor x;
  Tensor initial;
};

ScanInputs scan_inputs(b_lean_obj_arg a, b_lean_obj_arg b, b_lean_obj_arg x,
                       b_lean_obj_arg initial, uint32_t steps, uint32_t state,
                       bool variable) {
  auto coefficient_a = variable ? checked(a, {steps, state}) : checked(a, {state});
  auto coefficient_b = variable ? checked(b, {steps, state}) : checked(b, {state});
  if (!variable) {
    coefficient_a = coefficient_a.unsqueeze(0).expand({steps, state});
    coefficient_b = coefficient_b.unsqueeze(0).expand({steps, state});
  }
  return {coefficient_a, coefficient_b, checked(x, {steps, state}),
          checked(initial, {state})};
}

lean_obj_res scan_backward(const ScanInputs& inputs, const Tensor& output,
                           const Tensor& dy, bool variable) {
  const int64_t steps = inputs.x.size(0);
  const int64_t state = inputs.x.size(1);
  if (steps == 0 || state == 0) {
    auto parameter_shape = variable ? std::vector<int64_t>{steps, state}
                                    : std::vector<int64_t>{state};
    return quadruple(at::zeros(parameter_shape, inputs.x.options()),
                     at::zeros(parameter_shape, inputs.x.options()),
                     at::zeros_like(inputs.x),
                     at::zeros_like(inputs.initial));
  }
  // Reverse recurrence: g[t] = dy[t] + a[t+1] * g[t+1].
  // The final step adds +0 without reading a coefficient beyond the sequence.
  auto shifted = at::cat(
      {inputs.a.narrow(0, 1, steps - 1),
       at::zeros({1, state}, inputs.a.options())}, 0);
  auto g = affine_scan(shifted.flip({0}), dy.flip({0}),
                       at::zeros_like(inputs.initial)).flip({0});
  auto previous = at::cat(
      {inputs.initial.unsqueeze(0), output.narrow(0, 0, steps - 1)}, 0);
  auto da = g * previous;
  auto db = g * inputs.x;
  if (!variable) {
    // Constant coefficients accumulate their cotangents in reverse time.
    da = ordered_columns(da.flip({0}));
    db = ordered_columns(db.flip({0}));
  }
  return quadruple(da, db, g * inputs.b,
                   g.select(0, 0) * inputs.a.select(0, 0));
}

}  // namespace

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_reduce_sum_by_row(
    b_lean_obj_arg input, uint32_t rows, uint32_t cols) {
  return invoke([&] { return box(checked(input, {rows, cols}).sum(1)); });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_reduce_max_by_column(
    b_lean_obj_arg input, uint32_t rows, uint32_t cols) {
  return invoke([&] {
    auto x = checked(input, {rows, cols});
    return box(rows == 0 || cols == 0 ? at::zeros({cols}, x.options())
                                     : max_ignoring_nan(x, 0));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_reduce_max_by_row(
    b_lean_obj_arg input, uint32_t rows, uint32_t cols) {
  return invoke([&] {
    auto x = checked(input, {rows, cols});
    return box(rows == 0 || cols == 0 ? at::zeros({rows}, x.options())
                                     : max_ignoring_nan(x, 1));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_hard_masked_softmax_by_row(
    b_lean_obj_arg scores, b_lean_obj_arg mask, uint32_t rows, uint32_t cols) {
  return invoke([&] {
    auto x = checked(scores, {rows, cols});
    auto allowed = checked(mask, {rows, cols}).ne(0);
    if (rows == 0 || cols == 0) return box(at::empty_like(x));
    auto masked = x.masked_fill(allowed.logical_not(), kNegativeInfinity);
    auto empty = max_ignoring_nan(masked, 1, true).eq(kNegativeInfinity);
    auto probabilities = at::softmax(masked.masked_fill(empty, 0), 1);
    return box(probabilities.masked_fill(allowed.logical_not().logical_or(empty), 0));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_concat1d(
    b_lean_obj_arg first, b_lean_obj_arg second, uint32_t n, uint32_t m) {
  return invoke([&] { return box(at::cat({checked(first, {n}), checked(second, {m})})); });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_slice1d(
    b_lean_obj_arg input, uint32_t n, uint32_t start, uint32_t length) {
  return invoke([&] {
    auto x = checked(input, {n});
    require(start <= n && length <= n - start, "slice1d: slice out of bounds");
    return box(x.narrow(0, start, length).clone());
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_broadcast_vec_to_cols(
    b_lean_obj_arg input, uint32_t rows, uint32_t cols) {
  return invoke([&] {
    element_count({rows, cols});
    return box(checked(input, {rows}).unsqueeze(1).expand({rows, cols}).clone());
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_layer_norm_fwd(
    b_lean_obj_arg input, b_lean_obj_arg gamma, b_lean_obj_arg beta,
    uint32_t rows, uint32_t cols, double invCols, double epsilon) {
  return invoke([&] {
    (void)invCols;  // Derive the divisor from the exact integer column count.
    require(rows > 0 && cols > 0, "layerNormFwd: dimensions must be positive");
    auto x = checked(input, {rows, cols}).to(at::kDouble);
    auto weight = checked(gamma, {cols});
    auto bias = checked(beta, {cols});
    // Ordinary float32 native_layer_norm loses the double-centered intermediate
    // required by TorchLean's large-common-offset regression.
    // A device tensor divisor also avoids ATen's CPU-scalar division shortcut,
    // which multiplies by a rounded reciprocal instead of dividing the sum.
    auto divisor = at::full({1, 1}, static_cast<double>(cols), x.options());
    auto centered = x - x.sum({1}, true) / divisor;
    auto standard_deviation =
        (centered.square().sum({1}, true) / divisor + epsilon).sqrt();
    auto normalized = (centered / standard_deviation).to(at::kFloat);
    auto inverse = standard_deviation.reciprocal().to(at::kFloat);
    return triple(normalized * weight + bias, normalized, inverse);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_layer_norm_bwd(
    b_lean_obj_arg dout, b_lean_obj_arg normalized, b_lean_obj_arg invstd,
    b_lean_obj_arg gamma, uint32_t rows, uint32_t cols,
    double colsScale, double invCols) {
  return invoke([&] {
    require(rows > 0 && cols > 0, "layerNormBwd: dimensions must be positive");
    auto dy = checked(dout, {rows, cols});
    auto xhat = checked(normalized, {rows, cols});
    auto inverse = checked(invstd, {rows, 1});
    auto weight = checked(gamma, {cols});
    auto dxhat = dy * weight;
    // Keep the supplied, float32-rounded scale parameters and saved xhat/rstd.
    // Reconstructing native_layer_norm inputs would change this selected VJP.
    auto centered = dxhat * static_cast<float>(colsScale) - dxhat.sum({1}, true);
    auto term = centered - xhat * (dxhat * xhat).sum({1}, true);
    auto dx = (term * inverse) * static_cast<float>(invCols);
    return triple(dx, (dy * xhat).sum(0), dy.sum(0));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_bmm_with_transpose(
    b_lean_obj_arg first, b_lean_obj_arg second, uint32_t batch,
    uint32_t m, uint32_t n, uint32_t p, uint32_t transposeA, uint32_t transposeB) {
  return invoke([&] {
    require(transposeA <= 1 && transposeB <= 1, "bmm: transpose flag must be zero or one");
    auto a = transposeA ? checked(first, {batch, n, m}).transpose(1, 2)
                        : checked(first, {batch, m, n});
    auto b = transposeB ? checked(second, {batch, p, n}).transpose(1, 2)
                        : checked(second, {batch, n, p});
    element_count({batch, m, p});
    return box(at::bmm(a, b));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_rfft1d_packed(
    b_lean_obj_arg input, uint32_t batch, uint32_t n) {
  return invoke([&] {
    require(n > 0, "rfft1dPacked: length must be positive");
    auto x = checked(input, {batch, n});
    element_count({batch, n / 2 + 1, 2});
    if (batch == 0) return box(at::empty({0}, x.options()));
    auto spectrum = at::fft_rfft(x, n, 1, "backward");
    return box(at::view_as_real(spectrum));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_irfft1d_packed(
    b_lean_obj_arg input, uint32_t batch, uint32_t n) {
  return invoke([&] {
    require(n > 0, "irfft1dPacked: length must be positive");
    auto packed = checked(input, {batch, n / 2 + 1, 2});
    element_count({batch, n});
    if (batch == 0) return box(at::empty({0}, packed.options()));
    auto spectrum = at::view_as_complex(packed.contiguous()).clone();
    real_endpoints(spectrum, n, 1);
    // Explicit n is essential: the same half-spectrum shape also admits n-1.
    return box(at::fft_irfft(spectrum, n, 1, "backward"));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_spectral_conv1d_rfft_fwd(
    b_lean_obj_arg input, b_lean_obj_arg real, b_lean_obj_arg imaginary,
    uint32_t grid, uint32_t width, uint32_t modes) {
  return invoke([&] {
    spectral_shapes(grid, width, modes);
    auto x = checked(input, {grid, width});
    auto wr = checked(real, {modes, width, width});
    auto wi = checked(imaginary, {modes, width, width});
    if (modes == 0) return box(at::zeros_like(x));
    auto spectrum = at::fft_rfft(x, grid, 0, "backward");
    auto product = at::zeros_like(spectrum);
    product.narrow(0, 0, modes).copy_(
        at::bmm(spectrum.narrow(0, 0, modes).unsqueeze(1),
                at::complex(wr, wi)).squeeze(1));
    real_endpoints(product, grid, 0);
    return box(at::fft_irfft(product, grid, 0, "backward"));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_spectral_conv1d_rfft_bwd(
    b_lean_obj_arg input, b_lean_obj_arg real, b_lean_obj_arg imaginary,
    b_lean_obj_arg dout, uint32_t grid, uint32_t width, uint32_t modes) {
  return invoke([&] {
    spectral_shapes(grid, width, modes);
    auto x = checked(input, {grid, width});
    auto wr = checked(real, {modes, width, width});
    auto wi = checked(imaginary, {modes, width, width});
    auto dy = checked(dout, {grid, width});
    if (modes == 0) {
      return triple(at::zeros_like(x), at::zeros_like(wr), at::zeros_like(wi));
    }
    const int64_t frequencies = grid / 2 + 1;
    auto spectrum = at::fft_rfft(x, grid, 0, "backward");
    auto dz = at::fft_rfft(dy, grid, 0, "backward");
    auto factors = at::full({frequencies, 1}, 2.0f / static_cast<float>(grid),
                            x.options());
    factors.select(0, 0).fill_(1.0f / static_cast<float>(grid));
    if (grid % 2 == 0) factors.select(0, grid / 2).fill_(1.0f / static_cast<float>(grid));
    dz = dz * factors;
    real_endpoints(dz, grid, 0);
    auto kept_dz = dz.narrow(0, 0, modes);
    auto kept_x = spectrum.narrow(0, 0, modes);
    auto weights = at::complex(wr, wi);
    auto dx_spectrum = at::zeros_like(spectrum);
    dx_spectrum.narrow(0, 0, modes).copy_(
        at::bmm(weights.conj(), kept_dz.unsqueeze(2)).squeeze(2));
    // The adjoint of packed RFFT counts an interior coordinate once, whereas
    // C2R reconstructs both conjugate partners. Its inverse is unnormalized.
    auto adjoint_factors = at::full({frequencies, 1}, 0.5, x.options());
    adjoint_factors.select(0, 0).fill_(1);
    if (grid % 2 == 0) adjoint_factors.select(0, grid / 2).fill_(1);
    dx_spectrum = dx_spectrum * adjoint_factors;
    real_endpoints(dx_spectrum, grid, 0);
    auto dx = at::fft_irfft(dx_spectrum, grid, 0, "forward");
    auto dw = kept_x.conj().unsqueeze(2) * kept_dz.unsqueeze(1);
    return triple(dx, at::real(dw), at::imag(dw));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_selective_scan_diag_fwd(
    b_lean_obj_arg a, b_lean_obj_arg b, b_lean_obj_arg x, b_lean_obj_arg initial,
    uint32_t steps, uint32_t state) {
  return invoke([&] {
    auto inputs = scan_inputs(a, b, x, initial, steps, state, false);
    return box(affine_scan(inputs.a, inputs.b * inputs.x, inputs.initial));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_selective_scan_diag_bwd(
    b_lean_obj_arg a, b_lean_obj_arg b, b_lean_obj_arg x, b_lean_obj_arg initial,
    b_lean_obj_arg output, b_lean_obj_arg dout, uint32_t steps, uint32_t state) {
  return invoke([&] {
    auto inputs = scan_inputs(a, b, x, initial, steps, state, false);
    return scan_backward(inputs, checked(output, {steps, state}),
                         checked(dout, {steps, state}), false);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_selective_scan_diag_var_fwd(
    b_lean_obj_arg a, b_lean_obj_arg b, b_lean_obj_arg x, b_lean_obj_arg initial,
    uint32_t steps, uint32_t state) {
  return invoke([&] {
    auto inputs = scan_inputs(a, b, x, initial, steps, state, true);
    return box(affine_scan(inputs.a, inputs.b * inputs.x, inputs.initial));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_selective_scan_diag_var_bwd(
    b_lean_obj_arg a, b_lean_obj_arg b, b_lean_obj_arg x, b_lean_obj_arg initial,
    b_lean_obj_arg output, b_lean_obj_arg dout, uint32_t steps, uint32_t state) {
  return invoke([&] {
    auto inputs = scan_inputs(a, b, x, initial, steps, state, true);
    return scan_backward(inputs, checked(output, {steps, state}),
                         checked(dout, {steps, state}), true);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_scatter_add(
    b_lean_obj_arg input, b_lean_obj_arg values, uint32_t n,
    b_lean_obj_arg indices, uint32_t k) {
  return invoke([&] {
    return box(scatter_rows(checked(input, {n, 1}), checked(values, {k, 1}),
                            index_array(indices, k)));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_broadcast_to(
    b_lean_obj_arg input, b_lean_obj_arg input_dims,
    b_lean_obj_arg output_dims, b_lean_obj_arg axis_map) {
  return invoke([&] {
    auto shape = broadcast_shape(input_dims, output_dims, axis_map);
    auto x = checked(input, shape.input);
    return box(x.permute(shape.input_permutation).reshape(shape.expanded_input)
                   .expand(shape.output).clone());
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_reduce_from_broadcast(
    b_lean_obj_arg dout, b_lean_obj_arg input_dims,
    b_lean_obj_arg output_dims, b_lean_obj_arg axis_map) {
  return invoke([&] {
    auto shape = broadcast_shape(input_dims, output_dims, axis_map);
    auto dy = checked(dout, shape.output);
    const int64_t input_size = element_count(shape.input);
    if (input_size == 0 || dy.numel() == 0) {
      return box(at::zeros(shape.input, dy.options()));
    }
    std::vector<int64_t> permutation;
    std::vector<int64_t> kept(shape.input.size(), -1);
    for (size_t axis = 0; axis < shape.map.size(); ++axis) {
      const int64_t mapped = shape.map[axis];
      if (mapped == 0 || shape.input[mapped - 1] == 1) {
        permutation.push_back(static_cast<int64_t>(axis));
      } else {
        kept[mapped - 1] = static_cast<int64_t>(axis);
      }
    }
    for (int64_t axis : kept) {
      if (axis >= 0) permutation.push_back(axis);
    }
    // Reduced coordinates come first in their original row-major order;
    // surviving axes are then arranged in input order.
    auto matrix = dy.permute(permutation).reshape({dy.numel() / input_size, input_size});
    return box(column_sum(matrix, at::globalContext().deterministicAlgorithms()));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_swap_adjacent_at_depth(
    b_lean_obj_arg input, b_lean_obj_arg dims, uint32_t depth) {
  return invoke([&] {
    auto shape = shape_array(dims);
    require(static_cast<uint64_t>(depth) + 1 < shape.size(),
            "swapAdjacentAtDepth: invalid depth");
    return box(checked(input, shape).transpose(depth, static_cast<int64_t>(depth) + 1).clone());
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_reduce_sum_axis(
    b_lean_obj_arg input, b_lean_obj_arg dims, uint32_t axis) {
  return invoke([&] {
    auto shape = shape_array(dims);
    auto x = checked(input, shape);
    if (shape.empty()) return box(x.clone());
    require(axis < shape.size(), "reduceSumAxis: invalid axis");
    if (!at::globalContext().deterministicAlgorithms()) return box(x.sum(axis));
    std::vector<int64_t> permutation{axis};
    std::vector<int64_t> output_shape;
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i != axis) {
        permutation.push_back(static_cast<int64_t>(i));
        output_shape.push_back(shape[i]);
      }
    }
    auto matrix = x.permute(permutation).reshape({shape[axis], element_count(output_shape)});
    return box(ordered_columns(matrix));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_gather_rows(
    b_lean_obj_arg input, uint32_t rows, uint32_t cols,
    b_lean_obj_arg indices, uint32_t k) {
  return invoke([&] {
    element_count({k, cols});
    return box(gather_rows(checked(input, {rows, cols}), index_array(indices, k)));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_scatter_add_rows(
    b_lean_obj_arg input, b_lean_obj_arg values, uint32_t rows, uint32_t cols,
    b_lean_obj_arg indices, uint32_t k) {
  return invoke([&] {
    return box(scatter_rows(checked(input, {rows, cols}), checked(values, {k, cols}),
                            index_array(indices, k)));
  });
}

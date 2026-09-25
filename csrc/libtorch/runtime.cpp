#include "torchlean_libtorch.h"

#include "../cuda/common/torchlean_cuda_deterministic_reductions_env.h"
#include <ATen/Context.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <lean/mimalloc.h>
#include <torch/version.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>

namespace {

struct Counter {
  std::atomic<uint64_t> live{0}, peak{0}, created{0}, retired{0};

  void add(uint64_t amount) {
    created.fetch_add(1, std::memory_order_relaxed);
    const uint64_t next = live.fetch_add(amount, std::memory_order_relaxed) + amount;
    uint64_t previous = peak.load(std::memory_order_relaxed);
    while (previous < next &&
           !peak.compare_exchange_weak(previous, next, std::memory_order_relaxed)) {}
  }

  void remove(uint64_t amount) {
    retired.fetch_add(1, std::memory_order_relaxed);
    live.fetch_sub(amount, std::memory_order_relaxed);
  }
};

// Logical ownership counters deliberately exclude ATen temporaries and shared-storage aliases.
// The separate upstream allocator counters include storage retained by attention contexts.
Counter payloads;
Counter wrappers;
std::once_flag initialization;
std::atomic<int> selected_device{0};

bool release_data(torchlean_cuda_buffer* buffer) {
  if (!buffer || !buffer->tensor.defined()) return false;
  const size_t size = buffer->size;
  buffer->context.reset();
  buffer->tensor = at::Tensor();
  buffer->size = 0;
  if (size != 0) payloads.remove(size * sizeof(float));
  return size != 0;
}

void finalize(void* pointer) {
  auto* buffer = static_cast<torchlean_cuda_buffer*>(pointer);
  if (!buffer) return;
  release_data(buffer);
  delete buffer;
  wrappers.remove(1);
}

void foreach_reference(void*, b_lean_obj_arg) {}

lean_external_class* buffer_class() {
  static lean_external_class* result =
      lean_register_external_class(finalize, foreach_reference);
  return result;
}

template <typename F>
lean_obj_res io(F&& body) {
  try {
    torchlean::initialize();
    at::NoGradGuard no_grad;
    c10::DeviceGuard guard(torchlean::device());
    return lean_io_result_mk_ok(std::forward<F>(body)());
  } catch (const c10::OutOfMemoryError& error) {
    return lean_io_result_mk_error(
        lean_mk_io_error_resource_exhausted(0, lean_mk_string(error.what())));
  } catch (const std::bad_alloc& error) {
    return lean_io_result_mk_error(
        lean_mk_io_error_resource_exhausted(0, lean_mk_string(error.what())));
  } catch (const std::exception& error) {
    return lean_io_result_mk_error(
        lean_mk_io_error_other_error(0, lean_mk_string(error.what())));
  }
}

int64_t signed_bits(uint64_t value) {
  int64_t result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

at::Tensor shift_right(const at::Tensor& value, int64_t amount) {
  // ATen int64 shifts are arithmetic; masking recovers the unsigned SplitMix shift.
  const auto mask = static_cast<int64_t>(UINT64_MAX >> amount);
  return at::bitwise_and(at::bitwise_right_shift(value, amount), mask);
}

at::Tensor splitmix_draws(uint32_t n, uint64_t key, int64_t step = 1, int64_t offset = 0) {
  auto counter = at::arange(static_cast<int64_t>(n), torchlean::options().dtype(at::kLong));
  if (step != 1) counter = at::mul(counter, step);
  auto value = at::add(counter, signed_bits(key + static_cast<uint64_t>(offset) +
                                           UINT64_C(0x9e3779b97f4a7c15)));
  value = at::mul(at::bitwise_xor(value, shift_right(value, 30)),
                  signed_bits(UINT64_C(0xbf58476d1ce4e5b9)));
  value = at::mul(at::bitwise_xor(value, shift_right(value, 27)),
                  signed_bits(UINT64_C(0x94d049bb133111eb)));
  return at::bitwise_and(at::bitwise_xor(value, shift_right(value, 31)), INT64_C(0xffffffff));
}

at::Tensor uniform(uint32_t n, uint64_t key) {
  return at::div(splitmix_draws(n, key).to(at::kDouble), 4294967296.0).to(at::kFloat);
}

at::Tensor normal(uint32_t n, double mean, double deviation, uint64_t key) {
  const auto first = splitmix_draws(n, key, 2, 0).to(at::kFloat);
  const auto second = splitmix_draws(n, key, 2, 1);
  const auto u1 = at::div(at::add(first, 1.0f), static_cast<float>(4294967297.0));
  const auto u2 = at::div(second.to(at::kDouble), 4294967296.0).to(at::kFloat);
  const auto radius = at::sqrt(at::mul(at::log(u1), -2.0f));
  const auto angle = at::cos(at::mul(u2, static_cast<float>(6.2831853071795864769)));
  const auto sample = at::mul(radius, angle);
  return at::add(at::full_like(sample, static_cast<float>(mean)),
                 sample, static_cast<float>(deviation));
}

at::Tensor bernoulli(uint32_t n, double probability, uint64_t key) {
  const float p = static_cast<float>(probability);
  if (!(p > 0.0f)) return at::zeros({n}, torchlean::options());
  if (!(1.0f > p)) return at::ones({n}, torchlean::options());
  return at::lt(uniform(n, key), p).to(at::kFloat);
}

at::Tensor upload(b_lean_obj_arg object) {
  const size_t n = lean_sarray_size(object);
  TORCH_CHECK(n <= INT64_MAX, "LibTorch upload: input is too large");
  if (n == 0) return at::empty({0}, torchlean::options());
  // The blocking copy ends before Lean can release or mutate the borrowed host array.
  auto host = at::from_blob(lean_float_array_cptr(object), {static_cast<int64_t>(n)},
                            at::TensorOptions().dtype(at::kDouble).device(at::kCPU));
  return host.to(torchlean::options(), false, true);
}

lean_obj_res download(b_lean_obj_arg object) {
  const auto* buffer = torchlean_cuda_buffer_unbox(object);
  at::Tensor host;
  if (buffer->size != 0)
    host = torchlean::tensor(object).to(at::TensorOptions().device(at::kCPU).dtype(at::kDouble))
               .contiguous();
  lean_object* out = lean_mk_empty_float_array(lean_box(buffer->size));
  lean_sarray_set_size(out, buffer->size);
  if (buffer->size != 0)
    std::memcpy(lean_float_array_cptr(out), host.const_data_ptr<double>(),
                checked_bytes_size(buffer->size, sizeof(double), "FloatArray size overflow"));
  return out;
}

uint32_t read_bits(const uint8_t* source) {
  return uint32_t(source[0]) | (uint32_t(source[1]) << 8) |
         (uint32_t(source[2]) << 16) | (uint32_t(source[3]) << 24);
}

void write_bits(uint8_t* destination, float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  for (size_t i = 0; i != 4; ++i) destination[i] = static_cast<uint8_t>(bits >> (8 * i));
}

at::Tensor upload_bytes(b_lean_obj_arg object) {
  const size_t bytes = lean_sarray_size(object);
  TORCH_CHECK(bytes % sizeof(float) == 0,
              "float32 checkpoint payload is not a multiple of four bytes");
  const size_t n = bytes / sizeof(float);
  auto host = at::empty({static_cast<int64_t>(n)}, at::TensorOptions().dtype(at::kFloat));
  const auto* source = reinterpret_cast<const uint8_t*>(lean_sarray_cptr(object));
  float* values = host.data_ptr<float>();
  for (size_t i = 0; i != n; ++i) {
    const uint32_t bits = read_bits(source + 4 * i);
    std::memcpy(values + i, &bits, sizeof(bits));
  }
  return host.to(torchlean::options(), false, true);
}

lean_obj_res download_bytes(b_lean_obj_arg object) {
  const auto* buffer = torchlean_cuda_buffer_unbox(object);
  const size_t bytes =
      checked_bytes_size(buffer->size, sizeof(float), "float32 checkpoint size overflow");
  at::Tensor host;
  if (bytes != 0) host = torchlean::tensor(object).to(at::kCPU).contiguous();
  lean_object* out = lean_alloc_sarray(1, bytes, bytes);
  auto* destination = reinterpret_cast<uint8_t*>(lean_sarray_cptr(out));
  for (size_t i = 0; i != buffer->size; ++i)
    write_bits(destination + 4 * i, host.const_data_ptr<float>()[i]);
  return out;
}

auto allocator_stats() {
  return c10::cuda::CUDACachingAllocator::getDeviceStats(torchlean::device().index());
}

uint64_t memory_info(bool total) {
  return torchlean::invoke([&]() -> uint64_t {
    size_t free_bytes = 0, total_bytes = 0;
    C10_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return total ? total_bytes : free_bytes;
  });
}

}  // namespace

namespace torchlean {

void initialize() {
  std::call_once(initialization, [] {
    auto& context = at::globalContext();
    // Keep float32 input precision unless an application explicitly chooses TF32.
    context.setFloat32MatmulPrecision("highest");
    context.setAllowTF32CuBLAS(false);
    context.setAllowTF32CuDNN(false);
    context.setAllowFP16ReductionCuBLAS(false);
    context.setAllowBF16ReductionCuBLAS(false);
    context.setAllowFP16BF16ReductionMathSDP(false);
    context.setBenchmarkCuDNN(false);
    const bool deterministic = torchlean_read_deterministic_reductions_env() != 0;
    context.setDeterministicAlgorithms(deterministic, false);
    context.setDeterministicCuDNN(deterministic);
    context.lazyInitDevice(c10::kCUDA);
  });
}

c10::Device device() {
  return c10::Device(c10::kCUDA, static_cast<c10::DeviceIndex>(selected_device.load()));
}

at::TensorOptions options() {
  return at::TensorOptions().device(device()).dtype(at::kFloat).requires_grad(false);
}

const at::Tensor& tensor(b_lean_obj_arg object) {
  const auto* buffer = torchlean_cuda_buffer_unbox(object);
  TORCH_CHECK(buffer->tensor.defined(), "LibTorch: buffer has been released");
  TORCH_CHECK(buffer->tensor.is_cuda() && buffer->tensor.scalar_type() == at::kFloat,
              "LibTorch: expected a CUDA float32 buffer");
  TORCH_CHECK(!buffer->tensor.requires_grad(), "LibTorch: unexpected autograd tensor");
  TORCH_CHECK(buffer->tensor.numel() == static_cast<int64_t>(buffer->size),
              "LibTorch: buffer size disagrees with storage");
  return buffer->tensor;
}

torchlean_cuda_buffer* owned(at::Tensor value) {
  TORCH_CHECK(value.defined() && value.is_cuda() && value.scalar_type() == at::kFloat,
              "LibTorch: expected a CUDA float32 result");
  TORCH_CHECK(!value.requires_grad(), "LibTorch: native operations must not record autograd");
  auto buffer = std::make_unique<torchlean_cuda_buffer>();
  buffer->tensor = value.reshape({-1}).contiguous();
  buffer->size = static_cast<size_t>(buffer->tensor.numel());
  checked_bytes_size(buffer->size, sizeof(float), "LibTorch buffer byte size overflow");
  if (buffer->size != 0) payloads.add(buffer->size * sizeof(float));
  return buffer.release();
}

lean_obj_res box(at::Tensor value) {
  return torchlean_cuda_buffer_box(owned(std::move(value)));
}

}  // namespace torchlean

extern "C" torchlean_cuda_buffer* torchlean_cuda_buffer_unbox(b_lean_obj_arg object) {
  torchlean::require(lean_is_external(object), "LibTorch: expected an external buffer");
  return static_cast<torchlean_cuda_buffer*>(lean_get_external_data(object));
}

extern "C" lean_obj_res torchlean_cuda_buffer_box(torchlean_cuda_buffer* buffer) {
  auto* result = lean_alloc_external(buffer_class(), buffer);
  wrappers.add(1);
  return result;
}

extern "C" torchlean_cuda_buffer* torchlean_cuda_buffer_alloc(size_t n) {
  return torchlean::invoke([&] {
    TORCH_CHECK(n <= INT64_MAX, "LibTorch: buffer exceeds signed tensor size range");
    return torchlean::owned(at::empty({static_cast<int64_t>(n)}, torchlean::options()));
  });
}

extern "C" void torchlean_cuda_buffer_drop_unboxed(torchlean_cuda_buffer* buffer) {
  if (!buffer) return;
  release_data(buffer);
  delete buffer;
}

extern "C" LEAN_EXPORT uint32_t torchlean_cuda_runtime_status(uint32_t) {
  return c10::cuda::device_count() > 0 ? 1 : 2;
}

#define TORCHLEAN_COUNTER(NAME, VALUE)                                             \
  extern "C" LEAN_EXPORT uint64_t torchlean_cuda_##NAME(uint32_t) {                 \
    return (VALUE).load(std::memory_order_relaxed);                                \
  }
TORCHLEAN_COUNTER(allocator_live_bytes, payloads.live)
TORCHLEAN_COUNTER(allocator_peak_bytes, payloads.peak)
TORCHLEAN_COUNTER(allocator_alloc_count, payloads.created)
TORCHLEAN_COUNTER(allocator_free_count, payloads.retired)
TORCHLEAN_COUNTER(wrapper_live_count, wrappers.live)
TORCHLEAN_COUNTER(wrapper_peak_count, wrappers.peak)
TORCHLEAN_COUNTER(wrapper_alloc_count, wrappers.created)
TORCHLEAN_COUNTER(wrapper_finalize_count, wrappers.retired)
#undef TORCHLEAN_COUNTER

extern "C" LEAN_EXPORT uint64_t torchlean_cuda_allocator_device_free_bytes(uint32_t) {
  return memory_info(false);
}

extern "C" LEAN_EXPORT uint64_t torchlean_cuda_allocator_device_total_bytes(uint32_t) {
  return memory_info(true);
}

#define TORCHLEAN_ALLOCATOR_STAT(NAME, FIELD, WHICH)                                \
  extern "C" LEAN_EXPORT uint64_t torchlean_libtorch_##NAME(uint32_t) {             \
    return torchlean::invoke([]() -> uint64_t { return allocator_stats().FIELD[0].WHICH; }); \
  }
TORCHLEAN_ALLOCATOR_STAT(allocated_bytes, allocated_bytes, current)
TORCHLEAN_ALLOCATOR_STAT(reserved_bytes, reserved_bytes, current)
TORCHLEAN_ALLOCATOR_STAT(peak_allocated_bytes, allocated_bytes, peak)
TORCHLEAN_ALLOCATOR_STAT(peak_reserved_bytes, reserved_bytes, peak)
#undef TORCHLEAN_ALLOCATOR_STAT

extern "C" LEAN_EXPORT uint32_t torchlean_cuda_buffer_size(b_lean_obj_arg object) {
  const auto* buffer = torchlean_cuda_buffer_unbox(object);
  torchlean::require(buffer->size <= UINT32_MAX, "LibTorch: buffer size exceeds UInt32");
  return static_cast<uint32_t>(buffer->size);
}

extern "C" LEAN_EXPORT uint32_t torchlean_cuda_buffer_size_with_token(
    b_lean_obj_arg object, uint32_t) {
  return torchlean_cuda_buffer_size(object);
}

extern "C" LEAN_EXPORT uint32_t torchlean_cuda_buffer_release(b_lean_obj_arg object) {
  return release_data(torchlean_cuda_buffer_unbox(object)) ? 1 : 0;
}

extern "C" LEAN_EXPORT uint32_t torchlean_cuda_buffer_release_with_token(
    b_lean_obj_arg object, uint32_t) {
  return torchlean_cuda_buffer_release(object);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_release_then(
    b_lean_obj_arg scratch, b_lean_obj_arg keep) {
  torchlean_cuda_buffer_release(scratch);
  lean_inc(keep);
  return keep;
}

extern "C" LEAN_EXPORT uint32_t torchlean_runtime_collect_allocator(uint32_t) {
  mi_collect(false);
  return 1;
}

#define TORCHLEAN_CONSTRUCTOR(NAME, PARAMETERS, EXPRESSION)                         \
  extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_##NAME PARAMETERS {     \
    return torchlean::invoke([&] { return torchlean::box(EXPRESSION); });             \
  }                                                                               \
  extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_##NAME##_io PARAMETERS { \
    return io([&] { return torchlean::box(EXPRESSION); });                            \
  }
TORCHLEAN_CONSTRUCTOR(zeros, (uint32_t n), at::zeros({n}, torchlean::options()))
TORCHLEAN_CONSTRUCTOR(full, (uint32_t n, double v),
                     at::full({n}, static_cast<float>(v), torchlean::options()))
TORCHLEAN_CONSTRUCTOR(rand_uniform, (uint32_t n, uint64_t key), uniform(n, key))
TORCHLEAN_CONSTRUCTOR(bernoulli_mask, (uint32_t n, double p, uint64_t key), bernoulli(n, p, key))
TORCHLEAN_CONSTRUCTOR(of_float_array, (b_lean_obj_arg object), upload(object))
#undef TORCHLEAN_CONSTRUCTOR

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_rand_normal(
    uint32_t n, double mean, double deviation, uint64_t key) {
  return torchlean::invoke([&] { return torchlean::box(normal(n, mean, deviation, key)); });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_to_float_array(b_lean_obj_arg object) {
  return torchlean::invoke([&] { return download(object); });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_to_float_array_io(b_lean_obj_arg object) {
  return io([&] { return download(object); });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_to_float32_bytes_io(b_lean_obj_arg object) {
  return io([&] { return download_bytes(object); });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_of_float32_bytes_io(b_lean_obj_arg object) {
  return io([&] { return torchlean::box(upload_bytes(object)); });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_float_array_to_float32_bytes(b_lean_obj_arg object) {
  const size_t n = lean_sarray_size(object);
  const size_t bytes = checked_bytes_size(n, sizeof(float), "float32 checkpoint size overflow");
  lean_object* out = lean_alloc_sarray(1, bytes, bytes);
  const double* source = lean_float_array_cptr(object);
  auto* destination = reinterpret_cast<uint8_t*>(lean_sarray_cptr(out));
  for (size_t i = 0; i != n; ++i) write_bits(destination + 4 * i, static_cast<float>(source[i]));
  return out;
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_float32_bytes_to_float_array(b_lean_obj_arg object) {
  const size_t bytes = lean_sarray_size(object);
  torchlean::require(bytes % sizeof(float) == 0,
                     "float32 checkpoint payload is not a multiple of four bytes");
  const size_t n = bytes / sizeof(float);
  lean_object* out = lean_mk_empty_float_array(lean_box(n));
  lean_sarray_set_size(out, n);
  const auto* source = reinterpret_cast<const uint8_t*>(lean_sarray_cptr(object));
  double* destination = lean_float_array_cptr(out);
  for (size_t i = 0; i != n; ++i) {
    const uint32_t bits = read_bits(source + 4 * i);
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    destination[i] = value;
  }
  return out;
}

// These controls are process-wide, except for ATen's current device. Applications configure
// them before concurrent execution; each native call installs the selected device on its thread.
extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_version(uint32_t) {
  return lean_mk_string(TORCH_VERSION);
}

extern "C" LEAN_EXPORT uint32_t torchlean_libtorch_device_count(uint32_t) {
  return static_cast<uint32_t>(c10::cuda::device_count());
}

extern "C" LEAN_EXPORT uint32_t torchlean_libtorch_get_device(uint32_t) {
  return static_cast<uint32_t>(selected_device.load());
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_set_device(uint32_t index) {
  return io([&] {
    TORCH_CHECK(index < static_cast<uint32_t>(c10::cuda::device_count()),
                "LibTorch: selected CUDA device does not exist");
    TORCH_CHECK(wrappers.live.load() == 0,
                "LibTorch: select the device before creating tensor buffers");
    selected_device.store(static_cast<int>(index));
    return lean_box(0);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_get_setting(uint32_t setting) {
  return io([&] {
    auto& context = at::globalContext();
    uint32_t value = 0;
    switch (setting) {
      case 0: value = context.allowTF32CuBLAS(); break;
      case 1: value = context.allowTF32CuDNN(); break;
      case 2:
        value = context.deterministicAlgorithms() && !context.deterministicAlgorithmsWarnOnly();
        break;
      case 3: value = context.benchmarkCuDNN(); break;
      case 4: value = context.userEnabledFlashSDP(); break;
      case 5: value = context.userEnabledMemEfficientSDP(); break;
      case 6: value = context.userEnabledMathSDP(); break;
      case 7: value = context.userEnabledCuDNNSDP(); break;
      case 8: value = context.userEnabledCuDNN(); break;
      default: TORCH_CHECK(false, "LibTorch: unknown runtime setting");
    }
    return lean_box_uint32(value);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_set_setting(
    uint32_t setting, uint32_t enabled) {
  return io([&] {
    TORCH_CHECK(enabled <= 1, "LibTorch: runtime boolean must be zero or one");
    const bool value = enabled != 0;
    auto& context = at::globalContext();
    switch (setting) {
      case 0: context.setAllowTF32CuBLAS(value); break;
      case 1: context.setAllowTF32CuDNN(value); break;
      case 2:
        context.setDeterministicAlgorithms(value, false);
        context.setDeterministicCuDNN(value);
        if (value) context.setBenchmarkCuDNN(false);
        break;
      case 3:
        TORCH_CHECK(!value || !context.deterministicAlgorithms(),
                    "LibTorch: disable deterministic mode before enabling cuDNN benchmarking");
        context.setBenchmarkCuDNN(value);
        break;
      case 4: context.setSDPUseFlash(value); break;
      case 5: context.setSDPUseMemEfficient(value); break;
      case 6: context.setSDPUseMath(value); break;
      case 7: context.setSDPUseCuDNN(value); break;
      case 8: context.setUserEnabledCuDNN(value); break;
      default: TORCH_CHECK(false, "LibTorch: unknown runtime setting");
    }
    return lean_box(0);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_get_memory_fraction(uint32_t) {
  return io([] {
    return lean_box_float(
        c10::cuda::CUDACachingAllocator::getMemoryFraction(torchlean::device().index()));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_set_memory_fraction(double fraction) {
  return io([&] {
    TORCH_CHECK(std::isfinite(fraction) && fraction > 0.0 && fraction <= 1.0,
                "LibTorch: memory fraction must be finite and in (0, 1]");
    c10::cuda::CUDACachingAllocator::setMemoryFraction(fraction, torchlean::device().index());
    return lean_box(0);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_synchronize(uint32_t) {
  return io([] {
    c10::cuda::device_synchronize();
    return lean_box(0);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_empty_cache(uint32_t) {
  return io([] {
    c10::cuda::CUDACachingAllocator::emptyCache();
    return lean_box(0);
  });
}

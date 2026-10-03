// Lean–LibTorch interface: ownership, operations, and explicit backward calls.
#include "torchlean_libtorch.h"

// Buffer ownership and runtime controls

#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
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
// The separate upstream allocator counters include storage allocated inside ATen kernels.
Counter payloads;
Counter wrappers;
std::once_flag initialization;
std::atomic<int> selected_device{0};

bool release_data(torchlean_cuda_buffer* buffer) {
  if (!buffer || !buffer->tensor.defined()) return false;
  const size_t size = buffer->size;
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

// Initial policy from TORCHLEAN_CUDA_DETERMINISTIC_REDUCTIONS: unset, empty, or "0" means off.
bool deterministic_reductions_requested() {
  const char* value = std::getenv("TORCHLEAN_CUDA_DETERMINISTIC_REDUCTIONS");
  return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0;
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
    const bool deterministic = deterministic_reductions_requested();
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
  TORCH_CHECK(lean_get_external_class(object) == buffer_class(),
              "LibTorch: external object is not a tensor buffer");
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

// Device identity. Every query takes an explicit index and reads the SDK's per-device cached
// properties, so the answer never depends on the selected device and a multi-GPU host never
// combines fields from two cards in one record.
namespace {

const cudaDeviceProp& device_properties(uint32_t index) {
  TORCH_CHECK(index < static_cast<uint32_t>(c10::cuda::device_count()),
              "LibTorch: CUDA device index is outside the visible count");
  return *at::cuda::getDeviceProperties(static_cast<c10::DeviceIndex>(index));
}

// Clocks and bus width come from `cudaDeviceGetAttribute`: CUDA 13 removed `clockRate` and
// `memoryClockRate` from `cudaDeviceProp`, and the attribute enums exist in 12.x and 13.x.
uint32_t device_attribute(uint32_t index, cudaDeviceAttr attribute) {
  device_properties(index);
  int value = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&value, attribute, static_cast<int>(index)));
  return static_cast<uint32_t>(value);
}

}  // namespace

extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_device_name(uint32_t index) {
  return io([&] { return lean_mk_string(device_properties(index).name); });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_device_capability(uint32_t index) {
  return io([&] {
    const auto& properties = device_properties(index);
    return lean_box_uint32(static_cast<uint32_t>(properties.major * 10 + properties.minor));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_device_sm_count(uint32_t index) {
  return io([&] {
    return lean_box_uint32(static_cast<uint32_t>(device_properties(index).multiProcessorCount));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_device_total_bytes(uint32_t index) {
  return io([&] {
    return lean_box_uint64(static_cast<uint64_t>(device_properties(index).totalGlobalMem));
  });
}

#define TORCHLEAN_DEVICE_ATTRIBUTE(NAME, ATTRIBUTE)                                      \
  extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_device_##NAME(uint32_t index) { \
    return io([&] { return lean_box_uint32(device_attribute(index, ATTRIBUTE)); });      \
  }
TORCHLEAN_DEVICE_ATTRIBUTE(clock_khz, cudaDevAttrClockRate)
TORCHLEAN_DEVICE_ATTRIBUTE(mem_clock_khz, cudaDevAttrMemoryClockRate)
TORCHLEAN_DEVICE_ATTRIBUTE(mem_bus_width, cudaDevAttrGlobalMemoryBusWidth)
#undef TORCHLEAN_DEVICE_ATTRIBUTE

extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_driver_version(uint32_t) {
  return io([] {
    int version = 0;
    C10_CUDA_CHECK(cudaDriverGetVersion(&version));
    return lean_box_uint32(static_cast<uint32_t>(version));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_runtime_version(uint32_t) {
  return io([] {
    int version = 0;
    C10_CUDA_CHECK(cudaRuntimeGetVersion(&version));
    return lean_box_uint32(static_cast<uint32_t>(version));
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

// Elementwise operations


#include <algorithm>
#include <tuple>

namespace torchlean::elementwise {

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
  // PyTorch 0291f960b6 (a 2.12 nightly): CUDA DeviceAddCmulCdiv.cuh explicitly calls
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

at::Tensor reduce_mean(const at::Tensor& x) {
  if (x.numel() == 0)
    return at::full({1}, std::numeric_limits<float>::quiet_NaN(), x.options());
  // Preserve rounding of the complete sum before multiplication by the float32 reciprocal.
  const float scale = 1.0f / static_cast<float>(x.numel());
  return at::mul(reduce_sum(x), scale);
}

// All generated calls share conversion, validation, device selection, and result ownership.
at::Tensor argument(b_lean_obj_arg object) { return flat(object); }
float argument(double scalar) { return static_cast<float>(scalar); }

void check_argument(int64_t& size, const at::Tensor& value) {
  if (size < 0) size = value.numel();
  require(value.numel() == size, "LibTorch buffer operation: size mismatch");
}
void check_argument(int64_t&, float) {}

template <typename Function, typename... Args>
lean_obj_res call_buffer(Function&& function, Args... args) {
  return invoke([&]() {
    auto converted = std::make_tuple(argument(args)...);
    return std::apply([&](const auto&... values) {
      int64_t size = -1;
      (check_argument(size, values), ...);
      return box(function(values...));
    }, converted);
  });
}

}  // namespace

#define TORCHLEAN_BUFFER_EXPORT(NAME, PARAMETERS, ARGUMENTS, FUNCTION) \
  extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_##NAME PARAMETERS { \
    return call_buffer(FUNCTION, TORCHLEAN_ARGUMENTS ARGUMENTS); \
  }

// Parentheses keep argument lists intact while expanding the signature macros.
#define TORCHLEAN_ARGUMENTS(...) __VA_ARGS__
#define TORCHLEAN_UNARY_EXPORT(NAME, EXPRESSION) \
  TORCHLEAN_BUFFER_EXPORT(NAME, (b_lean_obj_arg x), (x), \
      ([](const auto& x) { return EXPRESSION; }))
#define TORCHLEAN_BINARY_EXPORT(NAME, EXPRESSION) \
  TORCHLEAN_BUFFER_EXPORT(NAME, (b_lean_obj_arg a, b_lean_obj_arg b), \
      (a, b), ([](const auto& a, const auto& b) { return EXPRESSION; }))
#define TORCHLEAN_UNARY_SCALAR_EXPORT(NAME, EXPRESSION) \
  TORCHLEAN_BUFFER_EXPORT(NAME, (b_lean_obj_arg x, double scalar), \
      (x, scalar), ([](const auto& x, float scalar) { return EXPRESSION; }))
#define TORCHLEAN_BINARY_SCALAR_EXPORT(NAME, EXPRESSION) \
  TORCHLEAN_BUFFER_EXPORT(NAME, (b_lean_obj_arg a, b_lean_obj_arg b, double scalar), \
      (a, b, scalar), \
      ([](const auto& a, const auto& b, float scalar) { return EXPRESSION; }))
#define TORCHLEAN_VJP_EXPORT(NAME, EXPRESSION) \
  TORCHLEAN_BUFFER_EXPORT(NAME##_bwd, (b_lean_obj_arg x, b_lean_obj_arg g), \
      (x, g), ([](const auto& x, const auto& g) { return EXPRESSION; }))

#include "operations.h"

#undef TORCHLEAN_VJP_EXPORT
#undef TORCHLEAN_BINARY_SCALAR_EXPORT
#undef TORCHLEAN_UNARY_SCALAR_EXPORT
#undef TORCHLEAN_BINARY_EXPORT
#undef TORCHLEAN_UNARY_EXPORT
#undef TORCHLEAN_ARGUMENTS
#undef TORCHLEAN_BUFFER_EXPORT

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

}  // namespace torchlean::elementwise

// Tensor operations

#include <optional>

// These functions evaluate TorchLean's buffer operations and selected VJPs. The
// Lean tape owns differentiation; invoke() disables native graph recording.
namespace torchlean::operators {

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

Tensor inverse_real_fft(b_lean_obj_arg input, uint32_t batch, uint32_t n, bool normalized) {
  require(n > 0, "irfft1dPacked: length must be positive");
  auto packed = checked(input, {batch, n / 2 + 1, 2});
  element_count({batch, n});
  if (batch == 0) return at::empty({0}, packed.options());
  auto spectrum = at::view_as_complex(packed.contiguous()).clone();
  real_endpoints(spectrum, n, 1);
  // Explicit n distinguishes odd/even lengths with the same half-spectrum shape.
  // The unnormalized transform avoids scaling large spectra up or tiny outputs down.
  return at::fft_irfft(spectrum, n, 1, normalized ? "backward" : "forward");
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
    // Only the mask determines whether a row is empty; allowed nonfinite scores
    // must retain their softmax arithmetic rather than being silently zeroed.
    auto empty = allowed.any(1, true).logical_not();
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
  return invoke([&] { return box(inverse_real_fft(input, batch, n, true)); });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_irfft1d_packed_unnormalized(
    b_lean_obj_arg input, uint32_t batch, uint32_t n) {
  return invoke([&] { return box(inverse_real_fft(input, batch, n, false)); });
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

}  // namespace torchlean::operators

// Attention and saved backward state

// Host double-precision matrix multiplication


// FloatArray stores binary64; this adapter keeps its dtype when using ATen matmul.
extern "C" LEAN_EXPORT lean_obj_res torchlean_dgemm_cuda(
    b_lean_obj_arg a_object, b_lean_obj_arg b_object, uint32_t m, uint32_t n, uint32_t p) {
  return torchlean::invoke([&]() {
    const size_t a_size = checked_mul_size(m, n, "dgemm: A size overflow");
    const size_t b_size = checked_mul_size(n, p, "dgemm: B size overflow");
    const size_t c_size = checked_mul_size(m, p, "dgemm: output size overflow");
    TORCH_CHECK(lean_sarray_size(a_object) == a_size, "dgemm: A size mismatch");
    TORCH_CHECK(lean_sarray_size(b_object) == b_size, "dgemm: B size mismatch");
    TORCH_CHECK(c_size <= INT64_MAX, "dgemm: output exceeds ATen element count");
    const auto cpu = at::TensorOptions().dtype(at::kDouble).device(at::kCPU);
    at::Tensor result;
    if (c_size == 0 || n == 0) {
      result = at::zeros({m, p}, cpu);
    } else {
      const auto gpu = cpu.device(torchlean::device());
      const auto a = at::from_blob(lean_float_array_cptr(a_object), {m, n}, cpu).to(gpu);
      const auto b = at::from_blob(lean_float_array_cptr(b_object), {n, p}, cpu).to(gpu);
      result = at::matmul(a, b).to(cpu).contiguous();
    }
    auto* out = lean_mk_empty_float_array(lean_box(c_size));
    lean_sarray_set_size(out, c_size);
    if (c_size != 0)
      std::memcpy(lean_float_array_cptr(out), result.const_data_ptr<double>(),
                  checked_bytes_size(c_size, sizeof(double), "dgemm: output byte size overflow"));
    return out;
  });
}

// Convolution and pooling

#include <vector>

// ATen schemas checked against the PyTorch 2.12 nightly 0291f960b6 and pip torch 2.13.0+cu130:
// https://github.com/pytorch/pytorch/blob/0291f960b6/aten/src/ATen/native/native_functions.yaml
// Backward calls return explicit cotangents to the Lean tape; they do not record autograd graphs.
namespace torchlean::conv_pool {

// The LibTorch backend follows PyTorch's convolution and pooling families: one, two, or three
// spatial dimensions. Higher-dimensional mathematical specifications remain available in Lean,
// but the native backend does not emulate kernels that LibTorch does not provide.
constexpr size_t kMaxSpatialRank = 3;

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

};

static at::Tensor bias_output(const at::Tensor& bias, const Geometry& g, int64_t channels) {
  Shape singleton(g.input.size() + 1, 1);
  singleton[0] = channels;
  return bias.reshape(singleton).expand(with_channels(channels, g.output)).clone();
}

static at::Tensor convolution_forward(
    const at::Tensor& input, const at::Tensor& kernel, const at::Tensor& bias,
    const Geometry& g, int64_t in_channels, int64_t out_channels, bool transposed) {
  if (out_channels == 0 || g.output_volume == 0) return at::empty({0}, input.options());
  if (input.numel() == 0) return bias_output(bias, g, out_channels);
  const auto x = input.reshape(with_channels(in_channels, g.input));
  const auto w = kernel.reshape(g.kernel_shape(in_channels, out_channels, transposed));
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

static std::tuple<at::Tensor, at::Tensor> max_pool_with_indices(
    const at::Tensor& x, const Geometry& g) {
  const PoolArguments args(g);
  const auto input = args.batched(x, g);
  const auto result = g.input.size() == 3
      ? at::max_pool3d_with_indices(input, args.kernel, args.stride, args.padding,
                                   args.dilation, false)
      : at::max_pool2d_with_indices(input, args.kernel, args.stride, args.padding,
                                   args.dilation, false);
  return {args.unbatched(std::get<0>(result), g),
          args.unbatched(std::get<1>(result), g)};
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
  const PoolArguments args(g);
  const auto source = args.batched(x, g);
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

}  // namespace torchlean::conv_pool

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

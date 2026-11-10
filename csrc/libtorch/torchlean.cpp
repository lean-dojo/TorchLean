// Lean–LibTorch interface: ownership, operations, and explicit backward calls.
#include "torchlean_libtorch.h"
#include "binary.h"

// Buffer ownership and runtime controls

#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <lean/mimalloc.h>
#include <torch/version.h>
#include <cuda.h>
#include <nvrtc.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <list>
#include <mutex>
#include <new>
#include <string>

namespace {

using torchlean::io;
using torchlean::rounded;

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
  const size_t bytes = buffer->tensor.nbytes();
  buffer->tensor = at::Tensor();
  buffer->size = 0;
  if (size != 0) payloads.remove(bytes);
  return size != 0;
}

void finalize(void* pointer) {
  auto* buffer = static_cast<torchlean_cuda_buffer*>(pointer);
  if (!buffer) return;
  release_data(buffer);
  if (buffer->format) lean_dec(buffer->format);
  delete buffer;
  wrappers.remove(1);
}

void foreach_reference(void* pointer, b_lean_obj_arg visitor) {
  auto* buffer = static_cast<torchlean_cuda_buffer*>(pointer);
  if (buffer && buffer->format) {
    lean_inc(visitor);
    lean_inc(buffer->format);
    lean_dec(lean_apply_1(visitor, buffer->format));
  }
}

lean_external_class* buffer_class() {
  static lean_external_class* result =
      lean_register_external_class(finalize, foreach_reference);
  return result;
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

at::Tensor splitmix_draws(uint32_t n, uint64_t key) {
  auto counter = at::arange(static_cast<int64_t>(n), torchlean::options().dtype(at::kLong));
  auto value = at::add(counter, signed_bits(key + UINT64_C(0x9e3779b97f4a7c15)));
  value = at::mul(at::bitwise_xor(value, shift_right(value, 30)),
                  signed_bits(UINT64_C(0xbf58476d1ce4e5b9)));
  value = at::mul(at::bitwise_xor(value, shift_right(value, 27)),
                  signed_bits(UINT64_C(0x94d049bb133111eb)));
  return at::bitwise_and(at::bitwise_xor(value, shift_right(value, 31)), INT64_C(0xffffffff));
}

at::ScalarType scalar_type(uint8_t dtype) {
  TORCH_CHECK(dtype <= 1, "LibTorch: unsupported buffer dtype");
  return dtype == 0 ? at::kFloat : at::kDouble;
}

at::Tensor uniform(uint32_t n, uint64_t key, at::ScalarType dtype) {
  return at::div(splitmix_draws(n, key).to(at::kDouble), 4294967296.0).to(dtype);
}

at::Tensor bernoulli(uint32_t n, double probability, uint64_t key, at::ScalarType dtype) {
  const double p = rounded(probability, dtype);
  if (!(p > 0.0)) return at::zeros({n}, torchlean::options().dtype(dtype));
  if (!(1.0 > p)) return at::ones({n}, torchlean::options().dtype(dtype));
  return at::lt(uniform(n, key, dtype), p).to(dtype);
}

at::Tensor upload(b_lean_obj_arg object, at::ScalarType dtype) {
  const size_t n = lean_sarray_size(object);
  TORCH_CHECK(n <= INT64_MAX, "LibTorch upload: input is too large");
  if (n == 0) return at::empty({0}, torchlean::options().dtype(dtype));
  // The blocking copy ends before Lean can release or mutate the borrowed host array.
  auto host = at::from_blob(lean_float_array_cptr(object), {static_cast<int64_t>(n)},
                            at::TensorOptions().dtype(at::kDouble).device(at::kCPU));
  return host.to(torchlean::options().dtype(dtype), false, true);
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

void check_driver(CUresult result) {
  if (result == CUDA_SUCCESS) return;
  const char* message = nullptr;
  cuGetErrorString(result, &message);
  TORCH_CHECK(false, "kernel CUDA driver: ", message ? message : "unknown error");
}

void check_nvrtc(nvrtcResult result) {
  TORCH_CHECK(result == NVRTC_SUCCESS, "kernel NVRTC: ", nvrtcGetErrorString(result));
}

struct CompiledKernel {
  CUcontext context = nullptr;
  CUmodule module = nullptr;
  CUfunction function = nullptr;

  ~CompiledKernel() {
    // Eviction can occur on a different host thread or after its device guard has changed.
    // If CUDA has already torn down this context, it has also reclaimed the module.
    if (module && cuCtxPushCurrent(context) == CUDA_SUCCESS) {
      cuModuleUnload(module);
      CUcontext previous = nullptr;
      cuCtxPopCurrent(&previous);
    }
  }
};

std::shared_ptr<CompiledKernel> compile_kernel(const std::string& source) {
  TORCH_CHECK(source.size() <= 16 * 1024 * 1024, "kernel: generated source exceeds 16 MiB");
  C10_CUDA_CHECK(cudaFree(nullptr));  // Establish the selected device's primary context.
  CUcontext context = nullptr;
  check_driver(cuCtxGetCurrent(&context));
  TORCH_CHECK(context, "kernel: no current CUDA context");
  using Entry = std::pair<std::pair<CUcontext, std::string>, std::shared_ptr<CompiledKernel>>;
  static std::mutex lock;
  static std::list<Entry> cache;
  std::lock_guard<std::mutex> guard(lock);
  for (auto it = cache.begin(); it != cache.end(); ++it) {
    if (it->first.first == context && it->first.second == source) {
      auto result = it->second;
      cache.splice(cache.begin(), cache, it);
      return result;
    }
  }

  nvrtcProgram program = nullptr;
  const char* headers[] = {torchlean::binary_cuda};
  const char* header_names[] = {"torchlean_binary.cuh"};
  check_nvrtc(nvrtcCreateProgram(&program, source.c_str(), "torchlean_kernel.cu", 1,
                               headers, header_names));
  struct ProgramGuard {
    nvrtcProgram* program;
    ~ProgramGuard() { nvrtcDestroyProgram(program); }
  } program_guard{&program};
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  const std::string architecture = "--gpu-architecture=compute_" +
      std::to_string(properties->major) + std::to_string(properties->minor);
  const char* options[] = {architecture.c_str(), "--std=c++17", "--fmad=false",
                          "--ftz=false", "--prec-div=true", "--prec-sqrt=true"};
  const auto status = nvrtcCompileProgram(program, 6, options);
  size_t log_size = 0;
  check_nvrtc(nvrtcGetProgramLogSize(program, &log_size));
  std::string log(log_size, '\0');
  if (log_size) check_nvrtc(nvrtcGetProgramLog(program, log.data()));
  TORCH_CHECK(status == NVRTC_SUCCESS, "kernel NVRTC compilation failed: ",
              nvrtcGetErrorString(status), "\n", log);
  size_t ptx_size = 0;
  check_nvrtc(nvrtcGetPTXSize(program, &ptx_size));
  std::string ptx(ptx_size, '\0');
  check_nvrtc(nvrtcGetPTX(program, ptx.data()));
  auto result = std::make_shared<CompiledKernel>();
  result->context = context;
  check_driver(cuModuleLoadData(&result->module, ptx.c_str()));
  check_driver(cuModuleGetFunction(&result->function, result->module, "torchlean_kernel"));
  cache.push_front({{context, source}, result});
  // Shared ownership retains evicted modules until any launch using them has finished.
  if (cache.size() > 16) cache.pop_back();
  return result;
}

at::Tensor run_kernel(b_lean_obj_arg source, const std::vector<at::Tensor>& inputs,
                      at::ScalarType scalar, uint64_t count, uint64_t stride = 1) {
  TORCH_CHECK(stride > 0 && stride <= INT64_MAX && count <= INT64_MAX / stride,
              "kernel: invalid scalar width or output extent");
  TORCH_CHECK(count <= INT64_MAX, "kernel: output size exceeds signed tensor size range");
  TORCH_CHECK(count * stride <= SIZE_MAX / c10::elementSize(scalar),
              "kernel: output byte size overflow");
  const auto kernel = compile_kernel(lean_string_cstr(source));
  auto output = at::empty({static_cast<int64_t>(count * stride)},
                          torchlean::options().dtype(scalar));
  auto error = at::zeros({4}, torchlean::options().dtype(at::kLong));
  if (count == 0) return output;
  constexpr uint64_t threads = 256;
  const uint64_t blocks = (count + threads - 1) / threads;
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(blocks <= static_cast<uint64_t>(properties->maxGridSize[0]),
              "kernel: output exceeds CUDA grid extent");
  std::vector<CUdeviceptr> addresses(inputs.size());
  std::vector<uint64_t> sizes(inputs.size());
  std::vector<void*> arguments;
  arguments.reserve(inputs.size() * 2 + 3);
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& input = inputs[i];
    TORCH_CHECK(input.device() == output.device() && input.scalar_type() == scalar &&
                input.is_contiguous(), "kernel: input device, dtype or layout mismatch");
    addresses[i] = reinterpret_cast<CUdeviceptr>(input.const_data_ptr());
    TORCH_CHECK(input.numel() % stride == 0, "kernel: truncated input scalar encoding");
    sizes[i] = static_cast<uint64_t>(input.numel()) / stride;
    arguments.push_back(&addresses[i]);
    arguments.push_back(&sizes[i]);
  }
  CUdeviceptr destination = reinterpret_cast<CUdeviceptr>(output.data_ptr());
  CUdeviceptr error_address = reinterpret_cast<CUdeviceptr>(error.data_ptr());
  arguments.push_back(&destination);
  arguments.push_back(&error_address);
  arguments.push_back(&count);
  const auto stream = c10::cuda::getCurrentCUDAStream(output.device().index());
  check_driver(cuLaunchKernel(kernel->function, static_cast<unsigned int>(blocks), 1, 1,
                             threads, 1, 1, 0, stream.stream(), arguments.data(), nullptr));
  // Keep inputs and the module alive until execution and error reporting have completed.
  // The blocking copy also sequences ATen's error initialization on the same stream.
  auto host_error = error.to(at::kCPU).contiguous();
  const auto* record = host_error.const_data_ptr<int64_t>();
  TORCH_CHECK(record[0] == 0, "kernel: input ", static_cast<uint64_t>(record[1]),
              " index ", static_cast<uint64_t>(record[2]), " is outside size ",
              static_cast<uint64_t>(record[3]));
  return output;
}

at::Tensor upload_bytes(b_lean_obj_arg object, at::ScalarType dtype, at::ScalarType source_dtype) {
  const size_t bytes = lean_sarray_size(object);
  const size_t width = c10::elementSize(source_dtype);
  TORCH_CHECK(bytes % width == 0, "checkpoint payload length does not match its dtype");
  const size_t n = bytes / width;
  auto host = at::empty({static_cast<int64_t>(n)}, at::TensorOptions().dtype(source_dtype));
  const auto* source = reinterpret_cast<const uint8_t*>(lean_sarray_cptr(object));
  auto* values = static_cast<uint8_t*>(host.data_ptr());
  // Encoded arithmetic already supplies byte-ordered words, independent of host endianness.
  if (width == 1) {
    if (bytes) std::memcpy(values, source, bytes);
  } else for (size_t i = 0; i != n; ++i) {
    uint64_t bits = 0;
    for (size_t j = 0; j != width; ++j) bits |= uint64_t(source[width * i + j]) << (8 * j);
    if (width == 4) {
      const uint32_t word = static_cast<uint32_t>(bits);
      std::memcpy(values + width * i, &word, width);
    } else std::memcpy(values + width * i, &bits, width);
  }
  return host.to(torchlean::options().dtype(dtype), false, true);
}

lean_obj_res download_bytes(b_lean_obj_arg object, at::ScalarType dtype) {
  const auto* buffer = torchlean_cuda_buffer_unbox(object);
  const size_t bytes =
      checked_bytes_size(buffer->size, c10::elementSize(dtype), "checkpoint size overflow");
  at::Tensor host;
  if (bytes != 0)
    host = torchlean::tensor(object).to(at::TensorOptions().device(at::kCPU).dtype(dtype))
                                    .contiguous();
  lean_object* out = lean_alloc_sarray(1, bytes, bytes);
  auto* destination = reinterpret_cast<uint8_t*>(lean_sarray_cptr(out));
  const size_t width = c10::elementSize(dtype);
  for (size_t i = 0; i != buffer->size; ++i) {
    uint64_t bits = 0;
    const auto* values = static_cast<const uint8_t*>(host.const_data_ptr());
    if (width == 4) {
      uint32_t word;
      std::memcpy(&word, values + width * i, width);
      bits = word;
    } else std::memcpy(&bits, values + width * i, width);
    for (size_t j = 0; j != width; ++j)
      destination[width * i + j] = static_cast<uint8_t>(bits >> (8 * j));
  }
  return out;
}

auto allocator_stats() {
  return c10::cuda::CUDACachingAllocator::getDeviceStats(torchlean::device().index());
}

uint64_t memory_info(bool total) {
  size_t free_bytes = 0, total_bytes = 0;
  C10_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  return total ? total_bytes : free_bytes;
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
  TORCH_CHECK(buffer->tensor.is_cuda() &&
              (buffer->tensor.scalar_type() == at::kFloat ||
               buffer->tensor.scalar_type() == at::kDouble),
              "LibTorch: expected a CUDA real buffer");
  TORCH_CHECK(!buffer->tensor.requires_grad(), "LibTorch: unexpected autograd tensor");
  TORCH_CHECK(buffer->tensor.numel() == static_cast<int64_t>(buffer->size),
              "LibTorch: buffer size disagrees with storage");
  return buffer->tensor;
}

torchlean_cuda_buffer* owned(at::Tensor value) {
  TORCH_CHECK(value.defined() && value.is_cuda() &&
              (value.scalar_type() == at::kFloat || value.scalar_type() == at::kDouble),
              "LibTorch: expected a CUDA real result");
  TORCH_CHECK(!value.requires_grad(), "LibTorch: native operations must not record autograd");
  auto buffer = std::make_unique<torchlean_cuda_buffer>();
  buffer->tensor = value.reshape({-1}).contiguous();
  buffer->size = static_cast<size_t>(buffer->tensor.numel());
  const auto bytes = checked_bytes_size(buffer->size, buffer->tensor.element_size(),
                                         "LibTorch buffer byte size overflow");
  if (buffer->size != 0) payloads.add(bytes);
  return buffer.release();
}

lean_obj_res box(at::Tensor value) {
  return torchlean_cuda_buffer_box(owned(std::move(value)));
}

lean_obj_res encoded(at::Tensor value, b_lean_obj_arg format, size_t width) {
  TORCH_CHECK(value.is_cuda() && value.scalar_type() == at::kByte && width > 0 &&
              value.numel() % width == 0, "encoded buffer: invalid storage extent");
  TORCH_CHECK(static_cast<uint64_t>(value.numel()) / width <= UINT32_MAX,
              "encoded buffer: scalar count exceeds the tape ABI");
  auto buffer = std::make_unique<torchlean_cuda_buffer>();
  buffer->tensor = value.reshape({-1}).contiguous();
  buffer->size = buffer->tensor.numel() / width;
  buffer->width = width;
  buffer->format = format;
  lean_inc(format);
  if (buffer->size) payloads.add(buffer->tensor.nbytes());
  return torchlean_cuda_buffer_box(buffer.release());
}

// Context.divNat on the device: each entry divided once by the exact count, through a device
// tensor divisor. ATen turns a CPU scalar divisor into a product with its rounded reciprocal, which
// changes bits: binary32 147 * 2^-149 / 98 rounds to 2^-148, the product to 2^-149. Below 2^53 a
// binary64 count is exact and the quotient is one IEEE division. binary32 divides in binary64 and
// then rounds to binary32; for finite s and 1 <= n < 2^29, RN32(RN64(s / n)) = RN32(s / n). Since
// |s| >= |s / n|, s is a multiple of the binary32 spacing D at s / n and n * M of D / 2 for a
// binary32 midpoint M, so a quotient other than M lies at least D / (2n) > D * 2^-30 from M, more
// than binary64's half spacing there: the binary64 rounding cannot reach or cross M, and a midpoint
// quotient is exact in binary64 and takes binary32's tie rule. Subnormal quotients have
// D = 2^-149, and a nonzero quotient exceeds 2^-178, a normal binary64 number. Zeros keep their
// sign; infinities and NaNs pass both roundings; the count zero is the IEEE division by zero.
at::Tensor divide_by_count(const at::Tensor& x, uint64_t count) {
  const auto dtype = x.scalar_type();
  if (dtype == at::kDouble) {
    require(count < (uint64_t{1} << 53), "divCount: a binary64 count must be below 2^53");
    return at::div(x, at::scalar_tensor(static_cast<double>(count), x.options()));
  }
  require(dtype == at::kFloat, "divCount: a binary32 or binary64 buffer is required");
  require(count < (uint64_t{1} << 29), "divCount: a binary32 count must be below 2^29");
  const auto wide = x.to(at::kDouble);
  return at::div(wide, at::scalar_tensor(static_cast<double>(count), wide.options()))
      .to(at::kFloat);
}

rows_kernel compile_rows(const std::string& source) {
  return compile_kernel(source);
}

void launch_rows(const rows_kernel& handle, std::vector<launch>& launches) {
  const auto* kernel = static_cast<const CompiledKernel*>(handle.get());
  constexpr uint64_t threads = 256;
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  const auto stream = c10::cuda::getCurrentCUDAStream(device().index());
  std::vector<void*> pointers;
  for (auto& entry : launches) {
    if (entry.count == 0) continue;
    const uint64_t blocks = (entry.count + threads - 1) / threads;
    TORCH_CHECK(blocks <= static_cast<uint64_t>(properties->maxGridSize[0]),
                "kernel: rows exceed CUDA grid extent");
    pointers.clear();
    for (auto& argument : entry.arguments) pointers.push_back(&argument);
    check_driver(cuLaunchKernel(kernel->function, static_cast<unsigned int>(blocks), 1, 1,
                                threads, 1, 1, 0, stream.stream(), pointers.data(), nullptr));
  }
  // The caller holds the module until the launches have finished.
  C10_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
}

void launch_rows(const std::string& source, std::vector<launch>& launches) {
  launch_rows(compile_rows(source), launches);
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

extern "C" void torchlean_cuda_buffer_drop_unboxed(torchlean_cuda_buffer* buffer) {
  if (!buffer) return;
  release_data(buffer);
  if (buffer->format) lean_dec(buffer->format);
  delete buffer;
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_runtime_status() {
  return lean_io_result_mk_ok(lean_box_uint32(c10::cuda::device_count() > 0 ? 1u : 2u));
}

// Ownership counters are relaxed atomic loads, so reading them never initializes CUDA.
#define TORCHLEAN_COUNTER(NAME, VALUE)                                             \
  extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_##NAME() {                     \
    return lean_io_result_mk_ok(                                                   \
        lean_box_uint64((VALUE).load(std::memory_order_relaxed)));                 \
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

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_allocator_device_free_bytes() {
  return io([] { return lean_box_uint64(memory_info(false)); });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_allocator_device_total_bytes() {
  return io([] { return lean_box_uint64(memory_info(true)); });
}

#define TORCHLEAN_ALLOCATOR_STAT(NAME, FIELD, WHICH)                                \
  extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_##NAME() {                 \
    return io([] { return lean_box_uint64(allocator_stats().FIELD[0].WHICH); });    \
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

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_size_io(b_lean_obj_arg object) {
  return lean_io_result_mk_ok(lean_box_uint32(torchlean_cuda_buffer_size(object)));
}

extern "C" LEAN_EXPORT uint32_t torchlean_cuda_buffer_release(b_lean_obj_arg object) {
  return release_data(torchlean_cuda_buffer_unbox(object)) ? 1 : 0;
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_release_io(
    b_lean_obj_arg object) {
  return lean_io_result_mk_ok(lean_box_uint32(torchlean_cuda_buffer_release(object)));
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_release_then(
    b_lean_obj_arg scratch, b_lean_obj_arg keep) {
  torchlean_cuda_buffer_release(scratch);
  lean_inc(keep);
  return keep;
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_runtime_collect_allocator() {
  mi_collect(false);
  return lean_io_result_mk_ok(lean_box(0));
}

#define TORCHLEAN_CONSTRUCTOR(NAME, PARAMETERS, EXPRESSION)                         \
  extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_##NAME PARAMETERS {     \
    return torchlean::invoke([&] { return torchlean::box(EXPRESSION); });             \
  }                                                                               \
  extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_##NAME##_io PARAMETERS { \
    return io([&] { return torchlean::box(EXPRESSION); });                            \
  }
TORCHLEAN_CONSTRUCTOR(zeros, (uint32_t n, uint8_t dtype),
                     at::zeros({n}, torchlean::options().dtype(scalar_type(dtype))))
TORCHLEAN_CONSTRUCTOR(full, (uint32_t n, double v, uint8_t dtype),
                     at::full({n}, rounded(v, scalar_type(dtype)),
                              torchlean::options().dtype(scalar_type(dtype))))
TORCHLEAN_CONSTRUCTOR(of_float_array, (b_lean_obj_arg object, uint8_t dtype),
                     upload(object, scalar_type(dtype)))
#undef TORCHLEAN_CONSTRUCTOR

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_rand_uniform_io(
    uint32_t n, uint64_t key, uint8_t dtype) {
  return io([&] { return torchlean::box(uniform(n, key, scalar_type(dtype))); });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_bernoulli_mask_io(
    uint32_t n, double p, uint64_t key, uint8_t dtype) {
  return io([&] { return torchlean::box(bernoulli(n, p, key, scalar_type(dtype))); });
}

extern "C" LEAN_EXPORT uint8_t torchlean_cuda_buffer_dtype(b_lean_obj_arg object) {
  return torchlean::invoke([&]() -> uint8_t {
    const auto* buffer = torchlean_cuda_buffer_unbox(object);
    if (buffer->format) {
      TORCH_CHECK(buffer->tensor.defined(), "encoded buffer: released handle");
      return 2;
    }
    return torchlean::tensor(object).scalar_type() == at::kFloat ? 0 : 1;
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_fail(b_lean_obj_arg message) {
  return torchlean::invoke([&]() -> lean_obj_res {
    TORCH_CHECK(false, lean_string_cstr(message));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_zeros_like(
    b_lean_obj_arg reference, uint32_t count) {
  return torchlean::invoke([&] {
    auto* buffer = torchlean_cuda_buffer_unbox(reference);
    TORCH_CHECK(buffer->tensor.defined(), "zeros: released buffer");
    const uint64_t width = buffer->format ? buffer->width : 1;
    TORCH_CHECK(width > 0 && count <= INT64_MAX / width, "zeros: invalid extent");
    auto value = at::zeros({static_cast<int64_t>(count * width)}, buffer->tensor.options());
    return buffer->format ? torchlean::encoded(value, buffer->format, width) : torchlean::box(value);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_copy(b_lean_obj_arg reference) {
  return torchlean::invoke([&] {
    auto* buffer = torchlean_cuda_buffer_unbox(reference);
    TORCH_CHECK(buffer->tensor.defined(), "copy: released buffer");
    auto value = buffer->tensor.clone();
    return buffer->format ? torchlean::encoded(value, buffer->format, buffer->width)
                          : torchlean::box(value);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_duplicate(b_lean_obj_arg reference) {
  return torchlean::invoke([&] {
    auto* buffer = torchlean_cuda_buffer_unbox(reference);
    TORCH_CHECK(buffer->tensor.defined(), "duplicate: released buffer");
    auto first = buffer->tensor.clone();
    auto second = buffer->tensor.clone();
    auto* result = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(result, 0, buffer->format
      ? torchlean::encoded(first, buffer->format, buffer->width) : torchlean::box(first));
    lean_ctor_set(result, 1, buffer->format
      ? torchlean::encoded(second, buffer->format, buffer->width) : torchlean::box(second));
    return result;
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_to_float_array(b_lean_obj_arg object) {
  return torchlean::invoke([&] { return download(object); });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_to_float_array_io(b_lean_obj_arg object) {
  return io([&] { return download(object); });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_kernel_run_buffer(
    b_lean_obj_arg source, uint8_t dtype, b_lean_obj_arg objects, uint64_t count) {
  return io([&] {
    std::vector<at::Tensor> inputs;
    inputs.reserve(lean_array_size(objects));
    for (size_t i = 0; i < lean_array_size(objects); ++i)
      inputs.push_back(torchlean::tensor(lean_array_uget(objects, i)));
    return torchlean::box(run_kernel(source, inputs, scalar_type(dtype), count));
  });
}

// The source's scalar layout and stride are generated together by Lean. Byte tensors
// preserve wide words, signed zeros and NaN payloads without numerical conversion.
extern "C" LEAN_EXPORT lean_obj_res torchlean_kernel_run_bytes(
    b_lean_obj_arg source, uint32_t format, uint64_t width,
    b_lean_obj_arg arrays, uint64_t count) {
  return io([&] {
    TORCH_CHECK(format <= 2 && width > 0, "kernel: invalid encoded scalar layout");
    const auto scalar = format == 0 ? at::kFloat : format == 1 ? at::kDouble : at::kByte;
    TORCH_CHECK(format == 2 || width == c10::elementSize(scalar),
                "kernel: native scalar width mismatch");
    std::vector<at::Tensor> inputs;
    inputs.reserve(lean_array_size(arrays));
    for (size_t i = 0; i < lean_array_size(arrays); ++i) {
      const auto object = lean_array_uget(arrays, i);
      const auto bytes = lean_sarray_size(object);
      TORCH_CHECK(bytes <= INT64_MAX && bytes % width == 0,
                  "kernel: invalid input byte extent");
      const auto elements = bytes / c10::elementSize(scalar);
      if (!bytes) inputs.push_back(at::empty({0}, torchlean::options().dtype(scalar)));
      else {
        const auto host = at::from_blob(lean_sarray_cptr(object),
            {static_cast<int64_t>(elements)}, at::TensorOptions().dtype(scalar).device(at::kCPU));
        inputs.push_back(host.to(torchlean::options().dtype(scalar), false, true));
      }
    }
    const auto result = run_kernel(source, inputs, scalar, count, format == 2 ? width : 1);
    const auto host = result.to(at::kCPU).contiguous();
    TORCH_CHECK(count <= SIZE_MAX / width, "kernel: host byte size overflow");
    const auto bytes = static_cast<size_t>(count * width);
    auto output = lean_alloc_sarray(1, bytes, bytes);
    if (bytes) std::memcpy(lean_sarray_cptr(output), host.const_data_ptr(), bytes);
    return output;
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_format(b_lean_obj_arg object) {
  auto* buffer = torchlean_cuda_buffer_unbox(object);
  if (!buffer->format) return lean_box(0);
  auto* result = lean_alloc_ctor(1, 1, 0);
  lean_inc(buffer->format);
  lean_ctor_set(result, 0, buffer->format);
  return result;
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_of_encoded_io(
    b_lean_obj_arg bytes, b_lean_obj_arg format, uint64_t width) {
  return io([&] {
    TORCH_CHECK(width && lean_sarray_size(bytes) % width == 0,
                "encoded buffer: truncated scalar word");
    return torchlean::encoded(upload_bytes(bytes, at::kByte, at::kByte), format, width);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_to_encoded_io(
    b_lean_obj_arg object) {
  return io([&] {
    auto* buffer = torchlean_cuda_buffer_unbox(object);
    TORCH_CHECK(buffer->format && buffer->tensor.defined(), "encoded buffer: invalid handle");
    const auto host = buffer->tensor.to(at::kCPU).contiguous();
    const size_t bytes = host.nbytes();
    auto* result = lean_alloc_sarray(1, bytes, bytes);
    if (bytes) std::memcpy(lean_sarray_cptr(result), host.const_data_ptr(), bytes);
    return result;
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_kernel_run_encoded(
    b_lean_obj_arg source, b_lean_obj_arg objects, b_lean_obj_arg format,
    uint64_t width, uint64_t count) {
  return torchlean::invoke([&] {
    std::vector<at::Tensor> inputs;
    for (size_t i = 0; i < lean_array_size(objects); ++i) {
      auto* buffer = torchlean_cuda_buffer_unbox(lean_array_uget(objects, i));
      TORCH_CHECK(buffer->format && buffer->width == width && buffer->tensor.defined(),
                  "encoded buffer: incompatible input storage");
      inputs.push_back(buffer->tensor);
    }
    return torchlean::encoded(run_kernel(source, inputs, at::kByte, count, width), format, width);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_to_bytes_io(
    b_lean_obj_arg object, uint8_t dtype) {
  return io([&] { return download_bytes(object, scalar_type(dtype)); });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_of_bytes_io(
    b_lean_obj_arg object, uint8_t dtype, uint8_t source) {
  return io([&] { return torchlean::box(upload_bytes(object, scalar_type(dtype), scalar_type(source))); });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_version() {
  return lean_io_result_mk_ok(lean_mk_string(TORCH_VERSION));
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_device_count() {
  return lean_io_result_mk_ok(
      lean_box_uint32(static_cast<uint32_t>(c10::cuda::device_count())));
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_get_device() {
  return lean_io_result_mk_ok(lean_box_uint32(static_cast<uint32_t>(selected_device.load())));
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

extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_get_memory_fraction() {
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

extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_synchronize() {
  return io([] {
    c10::cuda::device_synchronize();
    return lean_box(0);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_empty_cache() {
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
using torchlean::divide_by_count;
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
  require(a.scalar_type() == b.scalar_type(), "LibTorch buffer operation: dtype mismatch");
}

at::Tensor selected_sqrt(const at::Tensor& x) {
  // Ordered comparison preserves NaN and selects positive zero for both signed zeros.
  return at::sqrt(at::where(at::le(x, 0.0f), 0.0f, x));
}

at::Tensor selected_relu(const at::Tensor& x) {
  // Match Activation.relu's ordered maximum: preserve NaN, but normalize both zeros to +0.
  return at::where(at::le(x, 0.0f), 0.0f, x);
}

at::Tensor axpy(const at::Tensor& a, const at::Tensor& b, double c) {
  // CUDA addcmul evaluates std::fma(tensor1, tensor2, input) when its value is exactly one
  // (DeviceAddCmulCdiv.cuh), while add with alpha fuses only if the SDK's compiler contracted it.
  // We pass c as tensor2, the CPU-scalar operand, since a value other than one rounds
  // tensor1 * tensor2 before the fma. TorchLeanTest/CUDA/Numerics.lean checks the fma.
  const auto coefficient = at::scalar_tensor(
      rounded(c, a.scalar_type()), at::TensorOptions().dtype(a.scalar_type()).device(at::kCPU));
  return at::addcmul(a, b, coefficient, 1.0f);
}

// Native buffers are real: an ordered magnitude bound rejects both infinities and NaNs.
at::Tensor finite(const at::Tensor& x) {
  return at::lt(at::abs(x), std::numeric_limits<double>::infinity());
}

at::Tensor mean_update(const at::Tensor& mean, const at::Tensor& sample, double weight) {
  require(std::isfinite(weight) && weight > 0.0 && weight <= 1.0,
          "LibTorch running mean: weight must be finite and in (0, 1]");
  const double previous_weight = rounded(1.0 - weight, mean.scalar_type());
  auto weighted = at::mul(mean, previous_weight);
  weighted.add_(at::mul(sample, weight));
  // Subtraction is safe for same-sign finite values; opposite-sign weighted terms cannot
  // overflow their sum. Bound the finite result by its operands to contain rounding overshoot.
  at::Tensor candidate;
  {
    auto shifted = at::sub(sample, mean);
    shifted.mul_(weight).add_(mean);
    candidate = at::where(at::eq(at::signbit(mean), at::signbit(sample)), shifted, weighted);
  }
  // Only fresh workspaces are mutated; keep peak storage bounded for large parameter tensors.
  at::maximum_out(candidate, candidate, at::minimum(mean, sample));
  at::minimum_out(candidate, candidate, at::maximum(mean, sample));
  return at::where(at::logical_and(at::isfinite(mean), at::isfinite(sample)), candidate, weighted);
}

// The constants of `Activation.geluTanh`: `0.044715`, rounded to the dtype by `rounded`
// (binary32 0x3d372713, binary64 the value nearest 0.044715), and `sqrt(2 / pi)` computed in the
// dtype, the square root of the rounded quotient of 2 by the rounded pi: binary32 0x3f4c4229, one
// unit in the last place below PyTorch's kBeta rounded to binary32; binary64 the value nearest.
constexpr double kGeluKappa = 0.044715;

double gelu_beta(at::ScalarType dtype) {
  if (dtype == at::kFloat) {
    const float pi = static_cast<float>(M_PI);
    return static_cast<double>(std::sqrt(2.0f / pi));
  }
  require(dtype == at::kDouble, "gelu: the LibTorch buffers compute binary32 or binary64 only");
  return std::sqrt(2.0 / M_PI);
}

at::Tensor stable_sigmoid(const at::Tensor& x) {
  const auto z = at::exp(at::neg(at::abs(x)));
  const auto denominator = at::add(z, 1.0f);
  return at::where(at::gt(x, 0.0f), at::reciprocal(denominator), at::div(z, denominator));
}

at::Tensor activation_backward(const at::Tensor& x, const at::Tensor& g, bool is_tanh) {
  const auto magnitude = at::abs(x);
  const auto z = at::exp(at::neg(magnitude));
  const auto denominator = at::add(is_tanh ? at::mul(z, z) : z, 1.0f);
  at::Tensor slope;
  if (is_tanh) {
    const auto reciprocal_cosh = at::div(at::add(z, z), denominator);
    slope = at::mul(reciprocal_cosh, reciprocal_cosh);
  } else {
    slope = at::div(z, at::mul(denominator, denominator));
  }
  const auto ordinary = at::mul(slope, g);

  const auto half_argument = at::mul(magnitude, -0.5f);
  const auto q = at::exp(half_argument);
  const auto gradient_magnitude = at::abs(g);
  const double min_normal = x.scalar_type() == at::kFloat
      ? std::numeric_limits<float>::min() : std::numeric_limits<double>::min();
  at::Tensor repaired;
  if (x.scalar_type() == at::kFloat) {
    // The replay of TorchLean.Numeric.applySlope, q^2 g / (1 + q^2)^2 or
    // 4 q^4 g / (1 + q^4)^2, evaluated in binary64 and rounded once to binary32. Its binary64
    // error, below 8 units of 2^-53 relative, decides that rounding only next to a midpoint.
    // A binary32 q below the least normal sends the rule to its enclosure of the exact slope,
    // for which the binary64 exp stands in.
    const auto wide_q = at::where(at::lt(q, min_normal),
                                  at::exp(at::mul(magnitude.to(at::kDouble), -0.5)),
                                  q.to(at::kDouble));
    auto power = at::mul(wide_q, wide_q);
    if (is_tanh) power = at::mul(power, power);
    const auto wide_denominator = at::add(power, 1.0);
    auto weighted = at::div(at::mul(power, g.to(at::kDouble)),
                            at::mul(wide_denominator, wide_denominator));
    if (is_tanh) {
      weighted = at::mul(weighted, 4.0);
    } else {
      // At a nonzero x whose half argument rounds below the least normal, the rule encloses the
      // exact slope, which lies below the replay's 1/4 by a relative x^2/4 < 2^-250. One binary64
      // step toward zero rounds as the exact value does: both lie below g/4 and above every
      // binary32 number or midpoint below it. weighted * 0 keeps the sign of a zero.
      const auto enclosed = at::logical_and(at::gt(magnitude, 0.0f),
                                            at::gt(half_argument, -min_normal));
      weighted = at::where(enclosed, at::nextafter(weighted, at::mul(weighted, 0.0)), weighted);
    }
    repaired = weighted.to(at::kFloat);
  } else {
    // Without a wider device type, share the incoming magnitude between the exponential
    // factors before squaring, so the weighted gradient fits where a rounded slope is zero.
    auto amplitude = at::mul(at::sqrt(gradient_magnitude), q);
    if (is_tanh) amplitude = at::mul(at::mul(amplitude, q), 2.0f);
    amplitude = at::div(amplitude, denominator);
    repaired = at::copysign(at::mul(amplitude, amplitude), g);
  }
  const auto lost_range = at::logical_or(at::lt(at::abs(slope), min_normal),
                                       at::lt(at::abs(ordinary), min_normal));
  const auto finite_gradient = at::lt(gradient_magnitude, std::numeric_limits<double>::infinity());
  const auto repair = at::logical_and(lost_range, at::logical_and(finite(x), finite_gradient));
  return at::where(repair, repaired, ordinary);
}

// The limits that the Lean definitions of GELU, SiLU and their slopes return at the infinities
// before their formulas, which compute `inf * 0` there: `at_positive` at `+inf`, `-0` at `-inf`.
at::Tensor infinity_limits(const at::Tensor& x, const at::Tensor& at_positive,
                           const at::Tensor& formula) {
  const auto limit = at::where(at::gt(x, 0.0f), at_positive, at::neg(at::zeros_like(x)));
  return at::where(at::isinf(x), limit, formula);
}

at::Tensor staged_silu(const at::Tensor& x) {
  // Bound the inactive exponential before evaluating either ATen branch.
  const auto negative = at::minimum(x, at::zeros_like(x));
  const auto r = at::exp(at::mul(negative, 0.5f));
  const auto denominator = at::add(at::mul(r, r), 1.0f);
  const auto tail = at::div(at::mul(at::mul(x, r), r), denominator);
  return infinity_limits(x, x, at::where(at::gt(x, 0.0f), at::mul(x, stable_sigmoid(x)), tail));
}

at::Tensor staged_silu_backward(const at::Tensor& x, const at::Tensor& g) {
  const auto negative = at::minimum(x, at::zeros_like(x));
  const auto r = at::exp(at::mul(negative, 0.5f));
  const auto denominator = at::add(at::mul(r, r), 1.0f);
  const auto tail = at::div(at::mul(at::mul(at::add(x, denominator), r), r),
                            at::mul(denominator, denominator));
  const auto z = at::exp(at::neg(at::abs(x)));
  const auto positive_denominator = at::add(z, 1.0f);
  const auto positive_slope = at::div(z, at::mul(positive_denominator, positive_denominator));
  const auto positive = at::add(at::reciprocal(positive_denominator), at::mul(x, positive_slope));
  return at::mul(infinity_limits(x, at::ones_like(x), at::where(at::gt(x, 0.0f), positive, tail)),
                 g);
}

at::Tensor gelu_inner(const at::Tensor& x) {
  // Each ATen call materializes one rounded stage in the input dtype: beta (x + kappa x^3).
  const auto cube = at::mul(at::mul(x, x), x);
  const auto inner = at::add(x, at::mul(cube, rounded(kGeluKappa, x.scalar_type())));
  return at::mul(inner, gelu_beta(x.scalar_type()));
}

// `Activation.geluTanh` in its order, `(0.5 x) (1 + tanh (beta (x + kappa x^3)))`, one rounded
// ATen stage per operation, with its limits at the infinities; the device's `tanh` is its own
// approximation.
at::Tensor staged_gelu(const at::Tensor& x) {
  return infinity_limits(x, x, at::mul(at::mul(x, 0.5), at::add(at::tanh(gelu_inner(x)), 1.0)));
}

at::Tensor staged_gelu_backward(const at::Tensor& x, const at::Tensor& g) {
  const auto inner = gelu_inner(x);
  const auto gate = stable_sigmoid(at::add(inner, inner));
  const auto z = at::exp(at::neg(at::abs(inner)));
  const auto reciprocal_cosh = at::div(at::add(z, z), at::add(at::mul(z, z), 1.0f));
  const auto sech_term = at::mul(reciprocal_cosh, reciprocal_cosh);
  const double scaled_coeff = rounded(3.0 * rounded(kGeluKappa, x.scalar_type()), x.scalar_type());
  const auto quadratic0 = at::mul(x, scaled_coeff);
  const auto quadratic = at::mul(quadratic0, x);
  const auto inner_deriv = at::mul(at::add(quadratic, 1.0f),
                                  gelu_beta(x.scalar_type()));
  const auto derivative_term = at::where(at::eq(sech_term, 0.0f), at::zeros_like(x),
                                       at::mul(at::mul(x, sech_term), inner_deriv));
  const auto numerator = at::add(at::add(gate, gate), derivative_term);
  return at::mul(infinity_limits(x, at::ones_like(x), at::mul(numerator, 0.5f)), g);
}

// IEEE 754-2019 9.6 `maximum` from ATen pieces: `at::maximum` returns a NaN when an operand is
// one, and at equal zeros `a + b` is the larger zero in round to nearest, `+0` unless both are
// `-0`.
at::Tensor ieee_maximum(const at::Tensor& a, const at::Tensor& b) {
  return at::where(at::logical_and(at::eq(a, b), at::eq(a, 0)), at::add(a, b), at::maximum(a, b));
}

// 9.6 `minimum`: at equal zeros `-((-a) + (-b))` is the smaller zero, `-0` unless both are `+0`.
at::Tensor ieee_minimum(const at::Tensor& a, const at::Tensor& b) {
  return at::where(at::logical_and(at::eq(a, b), at::eq(a, 0)),
                   at::neg(at::add(at::neg(a), at::neg(b))), at::minimum(a, b));
}

lean_obj_res minmax_backward(
    const at::Tensor& a, const at::Tensor& b, const at::Tensor& g, bool maximum) {
  const auto a_wins = maximum ? at::gt(a, b) : at::gt(b, a);
  const auto b_wins = maximum ? at::gt(b, a) : at::gt(a, b);
  // The CPU rule, by selection: the strict winner receives g bit for bit, the strict loser +0 at
  // every g, and a tie or an unordered pair (a NaN operand) half of g each.
  const auto zero = at::zeros_like(g);
  const auto half = at::mul(at::full_like(g, 0.5), g);
  auto da = at::where(a_wins, g, at::where(b_wins, zero, half));
  auto db = at::where(b_wins, g, at::where(a_wins, zero, half));
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
  // Op.mean over every axis: the sum divided once by the element count, by one for an empty
  // buffer, whose sum is zero.
  return divide_by_count(reduce_sum(x), std::max<uint64_t>(static_cast<uint64_t>(x.numel()), 1));
}

// Tensor.meanSquaredError of the differences d: the sum of the squares divided once by the count
// n, or, where that sum overflows while every difference is finite, the sum of d * (d / n), whose
// exact terms do not exceed the exact mean. Both sums are reduce_sum's.
at::Tensor mse_mean(const at::Tensor& d, uint64_t count) {
  const auto total = reduce_sum(at::mul(d, d));
  const auto repair = at::logical_and(at::logical_not(finite(total)), at::all(finite(d)));
  return at::where(repair, reduce_sum(at::mul(d, divide_by_count(d, count))),
                   divide_by_count(total, count));
}

// Tensor.meanSquaredErrorBackward's slope 2d / n at each difference d: a finite d + d divided once
// by n, and otherwise d / n doubled. Both are 2d / n rounded once: a finite d + d is exact, and
// for n >= 2 so is the doubling of d / n, which stays normal at every count divide_by_count
// accepts (the quotient of a d with an overflowing double is subnormal only past 2^(emax - emin)).
at::Tensor mse_slope(const at::Tensor& d, uint64_t count) {
  const auto twice = at::add(d, d);
  const auto quotient = divide_by_count(d, count);
  return at::where(finite(twice), divide_by_count(twice, count), at::add(quotient, quotient));
}

// Huber.term at each prediction p and target t with the threshold delta, every operation in
// the buffers' dtype and in the rule's order: (|d| / 2) |d| below delta, else delta (|d| - delta /
// 2), or |delta p - delta t| - (delta / 2) delta where d = p - t is not finite, delta > 0 and that
// weighted difference is finite.
at::Tensor huber_terms(const at::Tensor& p, const at::Tensor& t, double delta) {
  const auto threshold = at::scalar_tensor(delta, p.options());
  const auto half = at::mul(threshold, 0.5);
  const auto difference = at::sub(p, t);
  const auto x = at::abs(difference);
  const auto direct = at::mul(threshold, at::sub(x, half));
  const auto weighted = at::abs(at::sub(at::mul(threshold, p), at::mul(threshold, t)));
  const auto keep = at::logical_or(
      at::logical_or(finite(difference), at::logical_not(at::gt(threshold, 0.0))),
      at::logical_not(finite(weighted)));
  const auto linear = at::where(keep, direct, at::sub(weighted, at::mul(half, threshold)));
  return at::where(at::lt(x, threshold), at::mul(at::mul(x, 0.5), x), linear);
}

// Tensor.huberLossBackward's slope at each difference d before the mean's division: d
// where |d| < delta, else delta, -delta or +0 by the sign of d (+0 at a NaN).
at::Tensor huber_slope(const at::Tensor& d, double delta) {
  const auto threshold = at::scalar_tensor(delta, d.options());
  const auto positive = at::gt(d, 0.0);
  const auto magnitude = at::where(positive, d, at::neg(d));
  const auto outer = at::where(positive, threshold,
      at::where(at::lt(d, 0.0), at::neg(threshold), at::zeros_like(d)));
  return at::where(at::lt(magnitude, threshold), d, outer);
}

// All generated calls share conversion, validation, device selection, and result ownership.
at::Tensor argument(b_lean_obj_arg object) { return flat(object); }
double argument(double scalar) { return scalar; }

void check_argument(int64_t& size, at::ScalarType& dtype, const at::Tensor& value) {
  if (size < 0) { size = value.numel(); dtype = value.scalar_type(); }
  require(value.numel() == size, "LibTorch buffer operation: size mismatch");
  require(value.scalar_type() == dtype, "LibTorch buffer operation: dtype mismatch");
}
void check_argument(int64_t&, at::ScalarType&, double) {}

const at::Tensor& rounded_argument(const at::Tensor& value, at::ScalarType) { return value; }
double rounded_argument(double value, at::ScalarType dtype) { return rounded(value, dtype); }

template <typename Function, typename... Args>
lean_obj_res call_buffer(Function&& function, Args... args) {
  return invoke([&]() {
    auto converted = std::make_tuple(argument(args)...);
    return std::apply([&](const auto&... values) {
      int64_t size = -1;
      at::ScalarType dtype = at::kFloat;
      (check_argument(size, dtype, values), ...);
      return box(function(rounded_argument(values, dtype)...));
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
      (x, scalar), ([](const auto& x, double scalar) { return EXPRESSION; }))
#define TORCHLEAN_BINARY_SCALAR_EXPORT(NAME, EXPRESSION) \
  TORCHLEAN_BUFFER_EXPORT(NAME, (b_lean_obj_arg a, b_lean_obj_arg b, double scalar), \
      (a, b, scalar), \
      ([](const auto& a, const auto& b, double scalar) { return EXPRESSION; }))
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

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_mse_mean(
    b_lean_obj_arg input, uint64_t count) {
  return invoke([&] { return box(mse_mean(tensor(input), count)); });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_mse_slope(
    b_lean_obj_arg input, uint64_t count) {
  return invoke([&] { return box(mse_slope(tensor(input), count)); });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_abs_bwd(
    b_lean_obj_arg XObj, b_lean_obj_arg GObj) {
  return invoke([&]() {
    const auto x = flat(XObj);
    const auto g = flat(GObj);
    same_size(x, g, "torchlean_cuda_buffer_abs_bwd: size mismatch");
    const auto sign = at::where(
        at::gt(x, 0.0f), 1.0f, at::where(at::lt(x, 0.0f), -1.0f, x));
    // Context.sign: x itself at a zero or NaN, so a NaN propagates and a zero keeps its sign.
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
    // The rule of Op.sqrt .clampNegative rounds g / (2 root) once and multiplies a zero factor
    // where x > 0 fails, which gives -0 at a negative g and a NaN at a nonfinite one.
    return box(at::where(positive, at::div(g, at::mul(root, 2.0f)),
                         at::mul(at::zeros_like(g), g)));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_clamp(
    b_lean_obj_arg BObj, double lo, double hi) {
  return invoke([&]() {
    const auto x = flat(BObj);
    const auto lower = at::scalar_tensor(rounded(lo, x.scalar_type()), x.options());
    const auto upper = at::scalar_tensor(rounded(hi, x.scalar_type()), x.options());
    // `Context.clamp`: minimum (maximum x lo) hi with 9.6's operations, so a NaN operand or bound
    // gives a NaN and, for a number x, lo > hi gives hi.
    return box(ieee_minimum(ieee_maximum(x, lower), upper));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_clamp_bwd(
    b_lean_obj_arg XObj, b_lean_obj_arg GObj, double lo, double hi) {
  return invoke([&]() {
    const auto x = flat(XObj);
    const auto g = flat(GObj);
    same_size(x, g, "torchlean_cuda_buffer_clamp_bwd: size mismatch");
    const auto interior = at::logical_and(
        at::gt(x, rounded(lo, x.scalar_type())), at::lt(x, rounded(hi, x.scalar_type())));
    // `clampDerivative x lo hi * g`: the coefficient 1 or 0 times the cotangent.
    return box(at::mul(at::where(interior, at::ones_like(g), at::zeros_like(g)), g));
  });
}

// Op.dropout in both passes: an entry is kept iff its uniform sample u is at least p (IEEE >=, so a
// NaN p keeps none); a kept entry is one IEEE division x / (1 - p), a dropped one x * 0 (NaN at an
// infinite or NaN x, -0 at a negative one), and every entry is +0 when p >= 1. The divisor is a
// device tensor, since ATen turns a CPU scalar divisor into a product with its rounded reciprocal.
// p is rounded to the dtype first; binary32 forms 1 - p in binary64 and rounds it once more, which
// is the binary32 difference (53 >= 2 * 24 + 2 bits).
extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_dropout(
    b_lean_obj_arg XObj, b_lean_obj_arg UObj, double probability) {
  return invoke([&]() {
    const auto x = flat(XObj);
    const auto u = flat(UObj);
    same_size(x, u, "torchlean_cuda_buffer_dropout: size mismatch");
    const auto dtype = x.scalar_type();
    const double p = rounded(probability, dtype);
    if (p >= 1.0) return box(at::zeros_like(x));
    const auto divisor = at::scalar_tensor(rounded(1.0 - p, dtype), x.options());
    return box(at::where(at::ge(u, p), at::div(x, divisor), at::mul(x, 0.0)));
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
    // Complete allocation and enqueue the copy before retiring the source
    // through the runtime's allocator and telemetry path.
    auto result = box(flat(BObj).clone());
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
    double firstMomentDenominator,
    double secondMomentDenominator,
    double epsilon,
    b_lean_obj_arg DecayObj,
    double updateScale) {
  return invoke([&]() {
    const auto parameters = flat(ParametersObj);
    const auto gradient = flat(GradientObj);
    const auto first_moment = flat(FirstMomentObj);
    const auto second_moment = flat(SecondMomentObj);
    same_size(parameters, gradient, "torchlean_cuda_buffer_adam_step: size mismatch");
    same_size(parameters, first_moment, "torchlean_cuda_buffer_adam_step: size mismatch");
    same_size(parameters, second_moment, "torchlean_cuda_buffer_adam_step: size mismatch");

    // Preserve independent coefficients and round each in the parameter dtype.
    const auto dtype = parameters.scalar_type();
    const double beta1_f = rounded(beta1, dtype);
    const double one_minus_beta1_f = rounded(oneMinusBeta1, dtype);
    const double beta2_f = rounded(beta2, dtype);
    const double one_minus_beta2_f = rounded(oneMinusBeta2, dtype);
    const double first_correction = rounded(1.0 / firstMomentDenominator, dtype);
    const double second_correction = rounded(1.0 / secondMomentDenominator, dtype);
    const double epsilon_f = rounded(epsilon, dtype);
    const double update_scale = rounded(updateScale, dtype);

    const auto m_scaled = at::mul(first_moment, beta1_f);
    auto m = axpy(m_scaled, gradient, one_minus_beta1_f);
    const auto g2 = at::mul(gradient, gradient);
    const auto v_scaled = at::mul(second_moment, beta2_f);
    const auto weighted_square = at::mul(at::mul(gradient, one_minus_beta2_f), gradient);
    auto v = at::where(at::lt(g2, std::numeric_limits<double>::infinity()),
                       axpy(v_scaled, g2, one_minus_beta2_f),
                       at::add(v_scaled, weighted_square));
    // Retain the ordinary host-computed reciprocal, but do not require it to fit the dtype.
    const auto correct_moment = [&](const at::Tensor& moment, double inverse, double divisor) {
      if (std::isfinite(inverse)) return at::mul(moment, inverse);
      return at::div(moment, at::full_like(moment, rounded(divisor, dtype)));
    };
    const auto m_hat = correct_moment(m, first_correction, firstMomentDenominator);
    const auto v_hat = correct_moment(v, second_correction, secondMomentDenominator);
    // The raw IEEE root of Buffer.sqrt, not Buffer.sqrtClampNegative: a negative v_hat stays
    // NaN, where the CPU's Optimizer.correctedRMS clamps it to zero.
    auto corrected_root = at::sqrt(v_hat);
    const double second_divisor = rounded(secondMomentDenominator, dtype);
    if (second_divisor > 0) {
      const auto correction_root = at::sqrt(at::full({}, second_divisor, v.options()));
      const auto scaled_root = at::div(at::sqrt(v),
          correction_root);
      corrected_root = at::where(finite(v_hat), corrected_root, scaled_root);
    }
    const auto denominator = at::add(corrected_root, epsilon_f);
    // Keep a tensor denominator: ATen's scalar-divisor optimization uses a
    // rounded reciprocal followed by multiplication, which can change bits.
    const auto normalized_update = at::div(m_hat, denominator);
    // Adam passes none: even a zero coefficient turns an infinite parameter into NaN.
    const auto decayed_parameters = lean_is_scalar(DecayObj) ? parameters
        : axpy(parameters, parameters,
               rounded(lean_unbox_float(lean_ctor_get(DecayObj, 0)), dtype));
    auto updated = axpy(decayed_parameters, normalized_update, update_scale);
    // A finite weighted update can survive an unrepresentable unscaled quotient.
    // Keep the ordinary FMA above; rescale exceptional entries by exact powers of two.
    if (std::isfinite(update_scale)) {
      const double min_normal = dtype == at::kFloat
          ? std::numeric_limits<float>::min() : std::numeric_limits<double>::min();
      const auto quotient_magnitude = at::abs(normalized_update);
      const auto quotient_finite = at::lt(quotient_magnitude,
          std::numeric_limits<double>::infinity());
      const auto range_loss = at::logical_or(at::logical_not(quotient_finite),
          at::logical_and(at::lt(quotient_magnitude, min_normal), at::ne(m_hat, 0)));
      const auto largest_input = at::maximum(at::maximum(at::abs(decayed_parameters),
          at::abs(m_hat)), at::abs(denominator));
      const auto finite_inputs = at::logical_and(
          at::lt(largest_input, std::numeric_limits<double>::infinity()), at::ne(denominator, 0));
      const auto repair = at::logical_and(range_loss, finite_inputs);
      if (update_scale == 0) {
        // Multiplying finite inputs first also preserves the sign of a zero update.
        const auto zero_update = at::div(at::mul(m_hat, update_scale), denominator);
        updated = at::where(repair, at::add(decayed_parameters, zero_update), updated);
      } else {
        const auto [m_fraction, m_exponent] = at::frexp(m_hat);
        const auto [d_fraction, d_exponent] = at::frexp(denominator);
        int scale_exponent = 0;
        const double scale_fraction = std::frexp(update_scale, &scale_exponent);
        const auto fraction = at::mul(at::div(m_fraction, d_fraction), scale_fraction);
        const auto exponent = at::add(at::sub(m_exponent, d_exponent), scale_exponent);
        const auto scaled_parameters = at::ldexp(decayed_parameters, at::neg(exponent));
        const auto rescaled = at::ldexp(at::add(scaled_parameters, fraction), exponent);
        // If this scaling overflows, the parameter dwarfs the update at its own precision.
        const auto recovered = at::where(finite(scaled_parameters),
            rescaled, decayed_parameters);
        updated = at::where(repair, recovered, updated);
      }
    }
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

constexpr double kNegativeInfinity = -std::numeric_limits<double>::infinity();

int64_t element_count(at::IntArrayRef shape) {
  for (int64_t dim : shape) require(dim >= 0, "LibTorch kernels: negative dimension");
  // A zero extent empties the shape whatever its other extents are.
  if (std::find(shape.begin(), shape.end(), 0) != shape.end()) return 0;
  int64_t count = 1;
  for (int64_t dim : shape) {
    require(count <= std::numeric_limits<int64_t>::max() / dim,
            "LibTorch kernels: shape size overflow");
    count *= dim;
  }
  return count;
}

void require_size(b_lean_obj_arg object, at::IntArrayRef shape) {
  require(tensor(object).numel() == element_count(shape),
          "LibTorch kernels: buffer size does not match shape");
}

Tensor checked(b_lean_obj_arg object, at::IntArrayRef shape) {
  require_size(object, shape);
  return shaped(object, shape);
}

// ATen's contiguous strides multiply the extents after the first, reading a zero extent as one,
// so they can overflow for a shape with a zero extent, which has no entries. A kernel returns the
// result of such a shape before it builds a tensor of that shape.
Tensor no_entries(b_lean_obj_arg like) {
  return at::empty({0}, tensor(like).options());
}

std::vector<int64_t> shape_array(b_lean_obj_arg object) {
  require(lean_is_array(const_cast<lean_object*>(object)),
          "LibTorch kernels: expected Array Nat");
  auto shape = dimensions(object, "LibTorch kernels: dimension exceeds UInt32");
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

// segment_reduce on a rank-two input folds each segment column in order from the initial value,
// on CPU and CUDA (aten/src/ATen/native/{cuda/SegmentReduce.cu,SegmentReduce.cpp}). ATen does not
// promise this order, so TorchLeanTest/CUDA/Numerics.lean checks it. We keep the
// trailing column dimension even for vectors because rank one selects a different CUDA reduction.
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

Tensor log_normalizer(const Tensor& shifted, const Tensor& pivot, int64_t axis) {
  return at::log1p(shifted.exp().scatter(axis, pivot, 0).sum(axis, true));
}

Tensor weighted_log_softmax(const Tensor& x, const Tensor& weights, int64_t axis) {
  const auto pivot = x.argmax(axis, true);
  const auto largest = x.gather(axis, pivot);
  const auto shifted = x - largest;
  const auto normalizer = log_normalizer(shifted, pivot, axis);
  const auto logp = shifted - normalizer;
  const auto finite = at::isfinite(logp);
  const auto direct = weights * at::where(finite, logp, at::zeros_like(logp));
  const auto difference = largest - x;
  const auto finite_difference = at::isfinite(difference);
  // Weight opposite-sign finite operands before an overflowing subtraction. This also avoids
  // halving a small derivative contribution. Nonfinite original inputs still propagate NaN.
  const auto gap = at::where(finite_difference,
      weights * at::where(finite_difference, difference, at::zeros_like(difference)),
      weights * largest - weights * x);
  return at::where(finite, direct, -(weights * normalizer + gap));
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

// Evaluate h[t] = a[t] h[t-1] + b[t] left to right with one rounded product and one rounded sum
// per step, the order of TorchLean.StateSpace.scanStates and of its adjoint recurrence.
Tensor affine_scan(const Tensor& a, const Tensor& b, const Tensor& initial) {
  const int64_t steps = b.size(0);
  if (steps == 0 || b.size(1) == 0) return at::empty_like(b);
  auto result = at::empty_like(b);
  auto state = initial;
  for (int64_t t = 0; t < steps; ++t) {
    state = a.select(0, t) * state + b.select(0, t);
    result.select(0, t).copy_(state);
  }
  return result;
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
  // The final step adds (+0) * (-0) = -0, the identity of rounded addition, so g[T-1] = dy[T-1]
  // exactly as in StateSpace.scanAdjoints, without reading a coefficient beyond the sequence.
  auto shifted = at::cat(
      {inputs.a.narrow(0, 1, steps - 1),
       at::zeros({1, state}, inputs.a.options())}, 0);
  auto g = affine_scan(shifted.flip({0}), dy.flip({0}),
                       at::full_like(inputs.initial, -0.0)).flip({0});
  auto previous = at::cat(
      {inputs.initial.unsqueeze(0), output.narrow(0, 0, steps - 1)}, 0);
  auto da = g * previous;
  auto db = g * inputs.x;
  if (!variable) {
    // Shared coefficients add their cotangents from +0 in increasing t, the order of
    // TorchLean.Tensor.scanCoefficientGrad.
    da = ordered_columns(da);
    db = ordered_columns(db);
  }
  return quadruple(da, db, g * inputs.b,
                   g.select(0, 0) * inputs.a.select(0, 0));
}

}  // namespace

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_reduce_sum_by_row(
    b_lean_obj_arg input, uint32_t rows, uint32_t cols) {
  return invoke([&] { return box(checked(input, {rows, cols}).sum(1)); });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_reduce_max_by_row(
    b_lean_obj_arg input, uint32_t rows, uint32_t cols) {
  return invoke([&] {
    auto x = checked(input, {rows, cols});
    return box(rows == 0 || cols == 0 ? at::zeros({rows}, x.options())
                                     : max_ignoring_nan(x, 1));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_argmax_mask_by_row(
    b_lean_obj_arg input, uint32_t rows, uint32_t cols) {
  return invoke([&] {
    const auto x = checked(input, {rows, cols});
    auto mask = at::zeros_like(x);
    if (rows != 0 && cols != 0) mask.scatter_(1, x.argmax(1, true), 1);
    return box(mask);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_distill(
    b_lean_obj_arg student, b_lean_obj_arg teacher, b_lean_obj_arg dimensions,
    uint32_t axis, uint8_t gradient) {
  return invoke([&] {
    const auto shape = shape_array(dimensions);
    require(axis < shape.size(), "distill: class axis out of range");
    require_size(student, shape);
    require_size(teacher, shape);
    require(tensor(student).scalar_type() == tensor(teacher).scalar_type(),
            "distill: student/teacher dtype mismatch");
    // The empty loss is the negated empty sum, `-0`, and the empty gradient is empty.
    if (element_count(shape) == 0) {
      return box(gradient ? no_entries(student) : -at::zeros({}, tensor(student).options()));
    }
    const auto x = shaped(student, shape);
    const auto t = shaped(teacher, shape);
    const auto samples = x.numel() / shape[axis];
    const auto p = at::softmax(t, axis, t.scalar_type());
    if (!gradient) {
      const auto pivot = x.argmax(axis, true);
      const auto shifted = x - x.gather(axis, pivot);
      const auto logq = shifted - log_normalizer(shifted, pivot, axis);
      // A zero teacher weight must not multiply an infinite student log-probability.
      const auto safe = at::where(p == 0, at::zeros_like(logq), logq);
      return box(divide_by_count(-(p * safe).sum(), samples));
    }
    const auto q = at::softmax(x, axis, x.scalar_type());
    const auto delta = x - t;
    const auto centered = delta - (p * delta).sum(axis, true);
    const auto ratio = centered - at::log1p((p * at::expm1(centered)).sum(axis, true));
    const auto close = at::abs(ratio) < 1;
    // Mask before expm1 as well: the inactive near-field branch must not overflow.
    const auto safe = at::where(close, ratio, at::zeros_like(ratio));
    const auto difference = at::where(close, p * at::expm1(safe), q - p);
    return box(divide_by_count(difference - q * difference.sum(axis, true), samples));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_cross_entropy(
    b_lean_obj_arg logits, b_lean_obj_arg target, b_lean_obj_arg dimensions,
    uint32_t axis, uint8_t mean) {
  return invoke([&] {
    const auto shape = shape_array(dimensions);
    require(axis < shape.size(), "cross_entropy: class axis out of range");
    require_size(logits, shape);
    require_size(target, shape);
    require(tensor(logits).scalar_type() == tensor(target).scalar_type(),
            "cross_entropy: dtype mismatch");
    // The empty loss is the negated empty sum, `-0`.
    if (element_count(shape) == 0) return box(-at::zeros({}, tensor(logits).options()));
    const auto x = shaped(logits, shape);
    const auto weights = shaped(target, shape);
    const auto total = -weighted_log_softmax(x, weights, axis).sum();
    return box(mean ? divide_by_count(total, x.numel() / shape[axis]) : total);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_cross_entropy_bwd(
    b_lean_obj_arg logits, b_lean_obj_arg target, b_lean_obj_arg seed,
    b_lean_obj_arg dimensions, uint32_t axis, uint8_t mean) {
  return invoke([&] {
    const auto shape = shape_array(dimensions);
    require(axis < shape.size(), "cross_entropy: class axis out of range");
    require_size(logits, shape);
    require_size(target, shape);
    const auto g = checked(seed, {});
    require(tensor(logits).scalar_type() == tensor(target).scalar_type() &&
                tensor(logits).scalar_type() == g.scalar_type(),
            "cross_entropy: dtype mismatch");
    if (element_count(shape) == 0) return pair(no_entries(logits), no_entries(target));
    const auto x = shaped(logits, shape);
    const auto weights = shaped(target, shape);
    const auto pivot = x.argmax(axis, true);
    const auto shifted = x - x.gather(axis, pivot);
    const auto logp = shifted - log_normalizer(shifted, pivot, axis);
    const auto probabilities = logp.exp();
    const auto upstream = -weights;
    const auto others = upstream.scatter(axis, pivot, 0);
    const auto regular = upstream - probabilities * upstream.sum(axis, true);
    const auto diagonal = -at::expm1(logp.gather(axis, pivot)) * upstream.gather(axis, pivot) -
                          probabilities.gather(axis, pivot) * others.sum(axis, true);
    // Each entry divided once by the number of samples when `mean`, as the spec's pullbacks.
    const auto scattered = regular.scatter(axis, pivot, diagonal);
    const auto targets = -weighted_log_softmax(x, g.expand_as(x), axis);
    const auto samples = x.numel() / shape[axis];
    const auto dx = (mean ? divide_by_count(scattered, samples) : scattered) * g;
    const auto dt = mean ? divide_by_count(targets, samples) : targets;
    return pair(dx, dt);
  });
}

// One for the entries of each row that fewer than k entries of the row beat, zero elsewhere
// (`MoE.kept`): entry j beats entry i when x_j > x_i, or when neither is larger and j < i.
// Without NaN that keeps min(k, cols) entries. Comparisons and integer counts only, so every
// device and run gives the same mask.
extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_top_k_mask_by_row(
    b_lean_obj_arg input, uint32_t rows, uint32_t cols, uint32_t k) {
  return invoke([&] {
    auto x = checked(input, {rows, cols});
    if (rows == 0 || cols == 0) return box(at::zeros_like(x));
    // Entry [r, i, j] compares entry j of row r with entry i.
    auto xi = x.unsqueeze(2);
    auto xj = x.unsqueeze(1);
    auto index = at::arange(static_cast<int64_t>(cols), x.options().dtype(at::kLong));
    auto earlier = index.unsqueeze(0).lt(index.unsqueeze(1));
    auto beats = xj.gt(xi).logical_or(earlier.logical_and(xi.gt(xj).logical_not()));
    auto rank = beats.sum(2);
    return box(rank.lt(static_cast<int64_t>(k)).to(x.scalar_type()));
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

// Whether two `FloatFormat` descriptors are equal. A descriptor holds `expWidth`, `fracWidth` and
// `exponentBias` as `Nat` objects and then its `encoding` byte. Its proofs are erased.
static bool same_format(b_lean_obj_arg x, b_lean_obj_arg y) {
  const unsigned fields = lean_ctor_num_objs(x);
  if (fields != lean_ctor_num_objs(y)) return false;
  for (unsigned i = 0; i < fields; ++i)
    if (!lean_nat_eq(lean_ctor_get(x, i), lean_ctor_get(y, i))) return false;
  const unsigned encoding = fields * sizeof(void*);
  return lean_ctor_get_uint8(x, encoding) == lean_ctor_get_uint8(y, encoding);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_concat1d(
    b_lean_obj_arg first, b_lean_obj_arg second, uint32_t n, uint32_t m) {
  return invoke([&] {
    auto* a = torchlean_cuda_buffer_unbox(first);
    auto* b = torchlean_cuda_buffer_unbox(second);
    if (a->format || b->format) {
      require(a->format && b->format && same_format(a->format, b->format) &&
                a->width == b->width && a->tensor.defined() && b->tensor.defined() &&
                a->size == n && b->size == m,
              "concat: incompatible encoded buffers");
      return encoded(at::cat({a->tensor, b->tensor}), a->format, a->width);
    }
    return box(at::cat({checked(first, {n}), checked(second, {m})}));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_slice1d(
    b_lean_obj_arg input, uint32_t n, uint32_t start, uint32_t length) {
  return invoke([&] {
    auto* buffer = torchlean_cuda_buffer_unbox(input);
    if (buffer->format) {
      require(buffer->tensor.defined() && buffer->size == n && start <= n && length <= n - start,
                "slice: invalid encoded buffer or extent");
      return encoded(buffer->tensor.narrow(0, static_cast<int64_t>(start * buffer->width),
                        static_cast<int64_t>(length * buffer->width)).clone(),
                     buffer->format, buffer->width);
    }
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

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_div_count(
    b_lean_obj_arg input, uint64_t count) {
  return invoke([&] { return box(divide_by_count(tensor(input), count)); });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_layer_norm_fwd(
    b_lean_obj_arg input, b_lean_obj_arg gamma, b_lean_obj_arg beta,
    uint32_t rows, uint32_t cols, double epsilon) {
  return invoke([&] {
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
    // Normalization.normalizedRows: each centered row is divided by Tensor.normalizationScale,
    // the IEEE maximum of its largest magnitude and sqrt |epsilon|, before it is squared. The
    // variance of the scaled row centers it again, and the clamp keeps a NaN.
    auto regularizer = at::full({1, 1}, epsilon, x.options());
    auto scale = at::maximum(centered.abs().amax({1}, true), regularizer.abs().sqrt());
    auto scaled = centered / scale;
    auto deviation = scaled - scaled.sum({1}, true) / divisor;
    auto standard_deviation = (deviation.square().sum({1}, true) / divisor +
        regularizer / scale / scale).clamp_min(0).sqrt();
    auto normalized = (scaled / standard_deviation).to(weight.scalar_type());
    auto inverse = (standard_deviation.reciprocal() / scale).to(weight.scalar_type());
    return triple(normalized * weight + bias, normalized, inverse);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_layer_norm_bwd(
    b_lean_obj_arg dout, b_lean_obj_arg normalized, b_lean_obj_arg invstd,
    b_lean_obj_arg gamma, uint32_t rows, uint32_t cols) {
  return invoke([&] {
    require(rows > 0 && cols > 0, "layerNormBwd: dimensions must be positive");
    auto dy = checked(dout, {rows, cols});
    auto xhat = checked(normalized, {rows, cols});
    auto inverse = checked(invstd, {rows, 1});
    auto weight = checked(gamma, {cols});
    auto dxhat = dy * weight;
    // LayerNorm.backward on the saved xhat and rstd (reconstructing native_layer_norm inputs
    // would change this VJP): rstd * ((dxhat - mean dxhat) - xhat * mean (dxhat * xhat)), each
    // mean a row sum divided once by the column count.
    auto mean = divide_by_count(dxhat.sum({1}, true), cols);
    auto mean_xhat = divide_by_count((dxhat * xhat).sum({1}, true), cols);
    auto dx = inverse * ((dxhat - mean) - xhat * mean_xhat);
    return triple(dx, (dy * xhat).sum(0), dy.sum(0));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_bmm_with_transpose(
    b_lean_obj_arg first, b_lean_obj_arg second, uint32_t batch,
    uint32_t m, uint32_t n, uint32_t p, uint32_t transposeA, uint32_t transposeB) {
  return invoke([&] {
    require(transposeA <= 1 && transposeB <= 1, "bmm: transpose flag must be zero or one");
    require_size(first, {batch, m, n});
    require_size(second, {batch, n, p});
    element_count({batch, m, p});
    if (batch == 0) return box(no_entries(first));
    auto a = transposeA ? shaped(first, {batch, n, m}).transpose(1, 2)
                        : shaped(first, {batch, m, n});
    auto b = transposeB ? shaped(second, {batch, p, n}).transpose(1, 2)
                        : shaped(second, {batch, n, p});
    return box(at::bmm(a, b));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_scaled_bmm(
    b_lean_obj_arg first, b_lean_obj_arg second, uint32_t batch,
    uint32_t m, uint32_t n, uint32_t p, uint32_t transposeA, uint32_t transposeB,
    double scale) {
  return invoke([&] {
    require(transposeA <= 1 && transposeB <= 1, "scaledBmm: invalid transpose flag");
    require_size(first, {batch, m, n});
    require_size(second, {batch, n, p});
    element_count({batch, m, p});
    require(tensor(first).scalar_type() == tensor(second).scalar_type(),
            "scaledBmm: dtype mismatch");
    if (batch == 0) {
      // An empty batch has no product to replay.
      auto* result = lean_alloc_ctor(0, 2, 0);
      lean_ctor_set(result, 0, box(no_entries(first)));
      lean_ctor_set(result, 1, lean_box(false));
      return result;
    }
    auto a = transposeA ? shaped(first, {batch, n, m}).transpose(1, 2)
                        : shaped(first, {batch, m, n});
    auto b = transposeB ? shaped(second, {batch, p, n}).transpose(1, 2)
                        : shaped(second, {batch, n, p});
    const auto minimum = a.scalar_type() == at::kFloat
        ? static_cast<double>(std::numeric_limits<float>::min())
        : std::numeric_limits<double>::min();
    const auto maximum = a.scalar_type() == at::kFloat
        ? static_cast<double>(std::numeric_limits<float>::max())
        : std::numeric_limits<double>::max();
    auto risky = at::zeros({}, a.options().dtype(at::kBool));
    if (a.numel() != 0 && b.numel() != 0) {
      const auto abs_a = a.abs();
      const auto abs_b = b.abs();
      const auto least_a = at::where(abs_a != 0, abs_a,
          at::full_like(abs_a, std::numeric_limits<double>::infinity())).amin();
      const auto least_b = at::where(abs_b != 0, abs_b,
          at::full_like(abs_b, std::numeric_limits<double>::infinity())).amin();
      // Conservative bounds may request replay for products from different contraction rows.
      // They also catch products rounded to zero before a growing scale can recover them.
      risky = at::logical_or(least_a * least_b < minimum,
                            abs_a.amax() * abs_b.amax() > maximum);
    }
    // A zero original row or column has no nonzero finite product to recover. Padding must not
    // alone force a host replay of the matrix; nonzero orthogonal rows remain conservative.
    const auto contributes = at::logical_and((a != 0).any(2, true), (b != 0).any(1, true));
    at::Tensor value;
    if (scale != 0.0 && std::abs(scale) < 1.0) {
      // A power of two near sqrt(|scale|) on the left and the remainder on the right: no rounded
      // square root, and no rounding for a power-of-two scale while the factors and the scaled
      // operands are representable in the dtype.
      int exponent = 0;
      std::frexp(scale, &exponent);
      const auto power = std::ldexp(1.0, exponent / 2);
      const auto left = at::mul(a, power);
      const auto right = at::mul(b, scale / power);
      // Zero is included: a nonzero original operand can disappear before the contraction.
      risky = at::logical_or(risky,
          at::logical_or(at::logical_and(a != 0, left.abs() < minimum).any(),
                         at::logical_and(b != 0, right.abs() < minimum).any()));
      value = at::bmm(left, right);
    } else {
      const auto product = at::bmm(a, b);
      // Cancellation can leave a tiny or zero sum even when every constituent product is normal.
      // A growing scale must see that original contraction, not its already rounded result.
      risky = at::logical_or(risky,
          at::logical_or(at::logical_not(at::isfinite(product)),
              at::logical_and(contributes, product.abs() < minimum)).any());
      value = scale == 1.0 ? product : at::mul(product, scale);
    }
    risky = at::logical_or(risky,
        at::logical_or(at::logical_not(at::isfinite(value)),
            at::logical_and(contributes, value.abs() < minimum)).any());
    // Keep the native buffer itself for nonfinite originals. Downloading a NaN through a
    // widened host FloatArray could change its payload even when no coefficient can be replayed.
    risky = at::logical_and(risky,
        at::logical_and(at::isfinite(a).all(), at::isfinite(b).all()));
    // One scalar read gates host recovery. The conservative guard also includes exact zeros.
    const bool replay = std::isfinite(scale) && risky.item<bool>();
    auto* result = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(result, 0, box(value));
    lean_ctor_set(result, 1, lean_box(replay));
    return result;
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

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_broadcast_to(
    b_lean_obj_arg input, b_lean_obj_arg input_dims,
    b_lean_obj_arg output_dims, b_lean_obj_arg axis_map) {
  return invoke([&] {
    auto shape = broadcast_shape(input_dims, output_dims, axis_map);
    require_size(input, shape.input);
    // An input with a zero extent broadcasts to an output with that extent.
    if (element_count(shape.output) == 0) return box(no_entries(input));
    auto x = shaped(input, shape.input);
    return box(x.permute(shape.input_permutation).reshape(shape.expanded_input)
                   .expand(shape.output).clone());
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_reduce_from_broadcast(
    b_lean_obj_arg dout, b_lean_obj_arg input_dims,
    b_lean_obj_arg output_dims, b_lean_obj_arg axis_map) {
  return invoke([&] {
    auto shape = broadcast_shape(input_dims, output_dims, axis_map);
    require_size(dout, shape.output);
    const int64_t input_size = element_count(shape.input);
    if (input_size == 0 || element_count(shape.output) == 0) {
      return box(at::zeros({input_size}, tensor(dout).options()));
    }
    auto dy = shaped(dout, shape.output);
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
    require_size(input, shape);
    if (element_count(shape) == 0) return box(no_entries(input));
    return box(shaped(input, shape).transpose(depth, static_cast<int64_t>(depth) + 1).clone());
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_reduce_sum_axis(
    b_lean_obj_arg input, b_lean_obj_arg dims, uint32_t axis) {
  return invoke([&] {
    auto shape = shape_array(dims);
    require_size(input, shape);
    if (shape.empty()) return box(shaped(input, shape).clone());
    require(axis < shape.size(), "reduceSumAxis: invalid axis");
    std::vector<int64_t> permutation{axis};
    std::vector<int64_t> output_shape;
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i != axis) {
        permutation.push_back(static_cast<int64_t>(i));
        output_shape.push_back(shape[i]);
      }
    }
    const int64_t output_size = element_count(output_shape);
    // The sums of an empty input are empty sums, `+0`.
    if (element_count(shape) == 0) return box(at::zeros({output_size}, tensor(input).options()));
    auto x = shaped(input, shape);
    if (!at::globalContext().deterministicAlgorithms()) return box(x.sum(axis));
    auto matrix = x.permute(permutation).reshape({shape[axis], output_size});
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

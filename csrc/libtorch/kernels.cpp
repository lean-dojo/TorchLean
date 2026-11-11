// Generated CUDA kernels and the CUDA driver. NVRTC compiles a kernel's source for the device's
// own architecture, and the driver loads it in the device's primary context, torch's context,
// and launches it on torch's current stream. The device count, memory and synchronization the
// bridge reports come from the driver too. No other file of the bridge calls the CUDA driver API
// or NVRTC.
#include "torchlean_libtorch.h"
#include "binary.h"

#include <cuda.h>
#include <nvrtc.h>

#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace torchlean {
namespace {

void check_driver(CUresult result) {
  if (result == CUDA_SUCCESS) return;
  const char* message = nullptr;
  cuGetErrorString(result, &message);
  fail("kernel CUDA driver: ", message ? message : "unknown error");
}

void check_nvrtc(nvrtcResult result) {
  TORCHLEAN_CHECK(result == NVRTC_SUCCESS, "kernel NVRTC: ", nvrtcGetErrorString(result));
}

// The driver handle of CUDA device `index`, with the driver initialized.
CUdevice cuda_device(int32_t index) {
  check_driver(cuInit(0));
  CUdevice device = 0;
  check_driver(cuDeviceGet(&device, index));
  return device;
}

// Makes the primary context of CUDA device `index`, the context of torch's CUDA work on it,
// current on this thread and returns it. Each primary context is retained once and never
// released, so it lives as long as the process.
CUcontext primary_context(int32_t index) {
  static std::mutex lock;
  static std::vector<CUcontext> contexts;
  CUcontext context = nullptr;
  {
    std::lock_guard<std::mutex> guard(lock);
    if (static_cast<size_t>(index) >= contexts.size()) contexts.resize(index + 1, nullptr);
    if (!contexts[index]) check_driver(cuDevicePrimaryCtxRetain(&contexts[index],
                                                                cuda_device(index)));
    context = contexts[index];
  }
  check_driver(cuCtxSetCurrent(context));
  return context;
}

// The value of attribute `which` of CUDA device `index`.
int attribute(CUdevice_attribute which, int32_t index) {
  int value = 0;
  check_driver(cuDeviceGetAttribute(&value, which, cuda_device(index)));
  return value;
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
  TORCHLEAN_CHECK(source.size() <= 16 * 1024 * 1024, "kernel: generated source exceeds 16 MiB");
  const int32_t index = device_index();
  const CUcontext context = primary_context(index);
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
  // NVRTC compiles for the device's own architecture, so the driver loads machine code and never
  // translates PTX, which a driver older than the NVRTC of the torch runtime cannot read.
  const std::string architecture = "--gpu-architecture=sm_" +
      std::to_string(attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, index)) +
      std::to_string(attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, index));
  const char* options[] = {architecture.c_str(), "--std=c++17", "--fmad=false",
                          "--ftz=false", "--prec-div=true", "--prec-sqrt=true"};
  const auto status = nvrtcCompileProgram(program, 6, options);
  size_t log_size = 0;
  check_nvrtc(nvrtcGetProgramLogSize(program, &log_size));
  std::string log(log_size, '\0');
  if (log_size) check_nvrtc(nvrtcGetProgramLog(program, log.data()));
  TORCHLEAN_CHECK(status == NVRTC_SUCCESS, "kernel NVRTC compilation failed: ",
                  nvrtcGetErrorString(status), "\n", log);
  size_t cubin_size = 0;
  check_nvrtc(nvrtcGetCUBINSize(program, &cubin_size));
  std::string cubin(cubin_size, '\0');
  check_nvrtc(nvrtcGetCUBIN(program, cubin.data()));
  auto result = std::make_shared<CompiledKernel>();
  result->context = context;
  check_driver(cuModuleLoadData(&result->module, cubin.data()));
  check_driver(cuModuleGetFunction(&result->function, result->module, "torchlean_kernel"));
  cache.push_front({{context, source}, result});
  // Shared ownership retains evicted modules until any launch using them has finished.
  if (cache.size() > 16) cache.pop_back();
  return result;
}

}  // namespace

int device_count() {
  int count = 0;
  if (cuInit(0) != CUDA_SUCCESS || cuDeviceGetCount(&count) != CUDA_SUCCESS) return 0;
  return count;
}

uint64_t memory_info(bool total) {
  size_t free_bytes = 0, total_bytes = 0;
  primary_context(device_index());
  check_driver(cuMemGetInfo(&free_bytes, &total_bytes));
  return total ? total_bytes : free_bytes;
}

void synchronize() {
  primary_context(device_index());
  check_driver(cuCtxSynchronize());
}

Tensor run_kernel(b_lean_obj_arg source, const std::vector<Tensor>& inputs, ScalarType scalar,
                  uint64_t count, uint64_t stride) {
  TORCHLEAN_CHECK(stride > 0 && stride <= INT64_MAX && count <= INT64_MAX / stride,
                  "kernel: invalid scalar width or output extent");
  TORCHLEAN_CHECK(count <= INT64_MAX, "kernel: output size exceeds signed tensor size range");
  TORCHLEAN_CHECK(count * stride <= SIZE_MAX / element_size(scalar),
                  "kernel: output byte size overflow");
  const auto kernel = compile_kernel(lean_string_cstr(source));
  const auto output = aten<"empty.memory_format">({static_cast<int64_t>(count * stride)},
                                                  options().dtype(scalar));
  const auto error = aten<"zeros">({4}, options().dtype(kLong));
  if (count == 0) return output;
  constexpr uint64_t threads = 256;
  const uint64_t blocks = (count + threads - 1) / threads;
  const int32_t index = output.get_device_index();
  TORCHLEAN_CHECK(blocks <= static_cast<uint64_t>(attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
                                                            index)),
                  "kernel: output exceeds CUDA grid extent");
  std::vector<CUdeviceptr> addresses(inputs.size());
  std::vector<uint64_t> sizes(inputs.size());
  std::vector<void*> arguments;
  arguments.reserve(inputs.size() * 2 + 3);
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& input = inputs[i];
    TORCHLEAN_CHECK(input.device() == output.device() && input.scalar_type() == scalar &&
                    input.is_contiguous(), "kernel: input device, dtype or layout mismatch");
    addresses[i] = reinterpret_cast<CUdeviceptr>(input.const_data_ptr());
    TORCHLEAN_CHECK(input.numel() % stride == 0, "kernel: truncated input scalar encoding");
    sizes[i] = static_cast<uint64_t>(input.numel()) / stride;
    arguments.push_back(&addresses[i]);
    arguments.push_back(&sizes[i]);
  }
  CUdeviceptr destination = reinterpret_cast<CUdeviceptr>(output.mutable_data_ptr());
  CUdeviceptr error_address = reinterpret_cast<CUdeviceptr>(error.mutable_data_ptr());
  arguments.push_back(&destination);
  arguments.push_back(&error_address);
  arguments.push_back(&count);
  const auto stream = static_cast<CUstream>(current_stream(index));
  check_driver(cuLaunchKernel(kernel->function, static_cast<unsigned int>(blocks), 1, 1,
                             threads, 1, 1, 0, stream, arguments.data(), nullptr));
  // Keep inputs and the module alive until execution and error reporting have completed.
  // The blocking copy also sequences ATen's error initialization on the same stream.
  const auto host_error = host_copy(error);
  const auto* record = host_error.const_data_ptr<int64_t>();
  TORCHLEAN_CHECK(record[0] == 0, "kernel: input ", static_cast<uint64_t>(record[1]),
                  " index ", static_cast<uint64_t>(record[2]), " is outside size ",
                  static_cast<uint64_t>(record[3]));
  return output;
}

rows_kernel compile_rows(const std::string& source) {
  return compile_kernel(source);
}

void launch_rows(const rows_kernel& handle, std::vector<launch>& launches) {
  const auto* kernel = static_cast<const CompiledKernel*>(handle.get());
  constexpr uint64_t threads = 256;
  const int32_t index = device_index();
  const auto grid = static_cast<uint64_t>(attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, index));
  const auto stream = static_cast<CUstream>(current_stream(index));
  std::vector<void*> pointers;
  for (auto& entry : launches) {
    if (entry.count == 0) continue;
    const uint64_t blocks = (entry.count + threads - 1) / threads;
    TORCHLEAN_CHECK(blocks <= grid, "kernel: rows exceed CUDA grid extent");
    pointers.clear();
    for (auto& argument : entry.arguments) pointers.push_back(&argument);
    check_driver(cuLaunchKernel(kernel->function, static_cast<unsigned int>(blocks), 1, 1,
                                threads, 1, 1, 0, stream, pointers.data(), nullptr));
  }
  // The caller holds the module until the launches have finished.
  check_driver(cuStreamSynchronize(stream));
}

void launch_rows(const std::string& source, std::vector<launch>& launches) {
  launch_rows(compile_rows(source), launches);
}

}  // namespace torchlean

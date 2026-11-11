// The unstable boundary of the LibTorch bridge (see unstable.h). Each `at::Context` function used
// here is defined out of line in libtorch_cpu, so it binds by symbol name; torch 2.13, 2.14 and
// 2.15 export each with the same signature. The caching allocator is reached through inline
// functions that call virtual functions of the exported
// `c10::cuda::CUDACachingAllocator::allocator`, which depends on the layout of its classes, so
// those calls run only on the releases checked. Exceptions propagate unchanged; the bridge's `io`
// reports them as Lean errors.
#include "unstable.h"

#include <ATen/Context.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace torchlean __attribute__((visibility("hidden"))) {
namespace unstable {
namespace {

namespace allocator = c10::cuda::CUDACachingAllocator;

// The torch releases in which the four allocator functions called below have the vtable slots,
// and the `DeviceStats` fields read below the offsets, that the bridge was checked against. A
// runtime of another release fails the call.
void check_allocator_abi() {
  const std::uint64_t version = aoti_torch_abi_version();
  const std::uint64_t major = version >> 56, minor = (version >> 48) & 0xff;
  if (major != 2 || minor < 13 || minor > 15) {
    throw std::runtime_error(
        "LibTorch: the CUDA caching allocator interface is checked for torch 2.13 to 2.15, not " +
        std::to_string(major) + "." + std::to_string(minor));
  }
}

}  // namespace

void initialize(bool deterministic) {
  auto& context = at::globalContext();
  // Keep float32 input precision unless an application explicitly chooses TF32.
  context.setFloat32MatmulPrecision("highest");
  context.setAllowTF32CuBLAS(false);
  context.setAllowTF32CuDNN(false);
  context.setAllowFP16ReductionCuBLAS(false);
  context.setAllowBF16ReductionCuBLAS(false);
  context.setAllowFP16BF16ReductionMathSDP(false);
  context.setBenchmarkCuDNN(false);
  context.setDeterministicAlgorithms(deterministic, false);
  context.setDeterministicCuDNN(deterministic);
}

bool setting(std::uint32_t id) {
  auto& context = at::globalContext();
  switch (id) {
    case 0: return context.allowTF32CuBLAS();
    case 1: return context.allowTF32CuDNN();
    case 2: return context.deterministicAlgorithms() && !context.deterministicAlgorithmsWarnOnly();
    case 3: return context.benchmarkCuDNN();
    case 4: return context.userEnabledFlashSDP();
    case 5: return context.userEnabledMemEfficientSDP();
    case 6: return context.userEnabledMathSDP();
    case 7: return context.userEnabledCuDNNSDP();
    case 8: return context.userEnabledCuDNN();
    default: throw std::runtime_error("LibTorch: unknown runtime setting");
  }
}

void set_setting(std::uint32_t id, bool value) {
  auto& context = at::globalContext();
  switch (id) {
    case 0: context.setAllowTF32CuBLAS(value); break;
    case 1: context.setAllowTF32CuDNN(value); break;
    case 2:
      context.setDeterministicAlgorithms(value, false);
      context.setDeterministicCuDNN(value);
      if (value) context.setBenchmarkCuDNN(false);
      break;
    case 3:
      if (value && context.deterministicAlgorithms()) {
        throw std::runtime_error(
            "LibTorch: disable deterministic mode before enabling cuDNN benchmarking");
      }
      context.setBenchmarkCuDNN(value);
      break;
    case 4: context.setSDPUseFlash(value); break;
    case 5: context.setSDPUseMemEfficient(value); break;
    case 6: context.setSDPUseMath(value); break;
    case 7: context.setSDPUseCuDNN(value); break;
    case 8: context.setUserEnabledCuDNN(value); break;
    default: throw std::runtime_error("LibTorch: unknown runtime setting");
  }
}

bool deterministic() { return at::globalContext().deterministicAlgorithms(); }

device_stats allocator_stats(std::int32_t device) {
  check_allocator_abi();
  const auto stats = allocator::getDeviceStats(static_cast<c10::DeviceIndex>(device));
  constexpr auto all = static_cast<std::size_t>(c10::CachingAllocator::StatType::AGGREGATE);
  return {static_cast<std::uint64_t>(stats.allocated_bytes[all].current),
          static_cast<std::uint64_t>(stats.reserved_bytes[all].current),
          static_cast<std::uint64_t>(stats.allocated_bytes[all].peak),
          static_cast<std::uint64_t>(stats.reserved_bytes[all].peak)};
}

void empty_cache() {
  check_allocator_abi();
  allocator::emptyCache();
}

double memory_fraction(std::int32_t device) {
  check_allocator_abi();
  return allocator::getMemoryFraction(static_cast<c10::DeviceIndex>(device));
}

void set_memory_fraction(double fraction, std::int32_t device) {
  check_allocator_abi();
  allocator::setMemoryFraction(fraction, static_cast<c10::DeviceIndex>(device));
}

}  // namespace unstable
}  // namespace torchlean

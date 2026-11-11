// The unstable boundary of the LibTorch bridge: the only source compiled against torch's C++ API
// instead of the LibTorch Stable ABI. It holds what the Stable ABI cannot express, the process-wide
// backend settings of `at::globalContext()` and the statistics and controls of the CUDA caching
// allocator. No torch type crosses this interface. "Unstable Remainder" in
// docs/TRUST_BOUNDARIES.md lists each torch symbol it uses, with the reason.
#pragma once

#include <cstdint>

namespace torchlean __attribute__((visibility("hidden"))) {
namespace unstable {

/// The bridge's initial backend policy: float32 matmuls and cuDNN convolutions at full precision
/// (no TF32), no reduced-precision reductions, no cuDNN benchmarking, and deterministic algorithms
/// and cuDNN exactly when `deterministic`.
void initialize(bool deterministic);

/// Runtime setting `id` of `TorchLean.LibTorch`'s controls (0 to 8) and its update. An unknown
/// `id` throws.
bool setting(std::uint32_t id);
void set_setting(std::uint32_t id, bool value);

/// Whether torch requires deterministic algorithms, as
/// `at::globalContext().deterministicAlgorithms()`.
bool deterministic();

/// The caching allocator's aggregate byte counts on CUDA device `device`.
struct device_stats {
  std::uint64_t allocated, reserved, peak_allocated, peak_reserved;
};

// The allocator calls below throw on a torch runtime other than 2.13, 2.14 and 2.15, the releases
// whose allocator interface the bridge was checked against.
device_stats allocator_stats(std::int32_t device);
void empty_cache();
double memory_fraction(std::int32_t device);
void set_memory_fraction(double fraction, std::int32_t device);

}  // namespace unstable
}  // namespace torchlean

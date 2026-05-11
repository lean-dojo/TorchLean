#pragma once

#include <stdint.h>

// Shared deterministic RNG primitive for native random buffers.
//
// TorchLean uses SplitMix64 as a small, reproducible counter-based stream: element `i` is sampled
// from `splitmix64(key + i)`. This is not a cryptographic RNG and is not meant to match PyTorch's
// generator. Its job is simpler: CPU stubs and CUDA kernels should produce the same deterministic
// pseudo-random sequence for tests and examples.

#if defined(__CUDACC__)
// `__forceinline__` already expands to an inline attribute in CUDA, so don't spell `inline` twice.
#define TORCHLEAN_CUDA_RNG_INLINE __device__ __forceinline__ static
#else
#define TORCHLEAN_CUDA_RNG_INLINE static inline
#endif

TORCHLEAN_CUDA_RNG_INLINE uint64_t torchlean_splitmix64(uint64_t x) {
  uint64_t z1 = x + 0x9e3779b97f4a7c15ULL;
  uint64_t z2 = (z1 ^ (z1 >> 30)) * 0xbf58476d1ce4e5b9ULL;
  uint64_t z3 = (z2 ^ (z2 >> 27)) * 0x94d049bb133111ebULL;
  return z3 ^ (z3 >> 31);
}

#undef TORCHLEAN_CUDA_RNG_INLINE

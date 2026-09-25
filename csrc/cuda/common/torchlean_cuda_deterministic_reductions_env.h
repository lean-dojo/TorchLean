#pragma once

#include <stdint.h>
#include <stdlib.h>

// Initial policy from TORCHLEAN_CUDA_DETERMINISTIC_REDUCTIONS.
// The native backend requests strict ATen determinism and supports LibTorch runtime controls.
// CPU stubs capture this policy on first use to select their deterministic accumulation paths.
static inline uint32_t torchlean_read_deterministic_reductions_env(void) {
  const char* v = getenv("TORCHLEAN_CUDA_DETERMINISTIC_REDUCTIONS");
  if (!v || !*v) {
    return 0u;
  }
  if (v[0] == '0' && v[1] == '\0') {
    return 0u;
  }
  return 1u;
}

#pragma once

#include <stdint.h>
#include <stdlib.h>

// Env-var default for deterministic reductions mode.
//
// This is intentionally shared by CUDA and CPU stubs so the user-facing toggle has the same meaning
// in both builds. The runtime setter can override the env default after startup; the env parser only
// answers the initial policy.
//
// - `TORCHLEAN_CUDA_DETERMINISTIC_REDUCTIONS=1` (preferred)
// - `TORCHLEAN_DETERMINISTIC_REDUCTIONS=1` (alias)
static inline uint32_t torchlean_read_deterministic_reductions_env() {
  const char* v = getenv("TORCHLEAN_CUDA_DETERMINISTIC_REDUCTIONS");
  if (!v || !*v) {
    v = getenv("TORCHLEAN_DETERMINISTIC_REDUCTIONS");
  }
  if (!v || !*v) {
    return 0u;
  }
  if (v[0] == '0' && v[1] == '\0') {
    return 0u;
  }
  return 1u;
}

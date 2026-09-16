#pragma once

#include <lean/lean.h>

#include "torchlean_size_common.h"

#include <cuda_runtime.h>
#include <stddef.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

// Shared CUDA runtime bits.
// Wrappers validate shapes before launch; `checkCuda` is only for CUDA driver/runtime failures.

static inline void checkCuda(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) {
    char detail[512];
    snprintf(detail, sizeof(detail), "%s: %s (%d)", msg, cudaGetErrorString(e), (int)e);
    lean_internal_panic(detail);
  }
}

// CUDA records the last asynchronous launch error per host thread. Clear it immediately before a
// launch when the next check is meant to diagnose that launch, not an older unchecked kernel.
static inline void torchlean_cuda_clear_pending_error() {
  (void)cudaGetLastError();
}

// Check the launch error recorded by the kernel that was just submitted.
static inline void torchlean_cuda_check_launch(const char* msg) {
  checkCuda(cudaGetLastError(), msg);
}

static inline void torchlean_cuda_free_checked(void** ptr, const char* msg) {
  if (ptr && *ptr) {
    checkCuda(cudaFree(*ptr), msg);
    *ptr = nullptr;
  }
}

static inline void torchlean_cuda_lock(pthread_mutex_t* mutex, const char* msg) {
  if (pthread_mutex_lock(mutex) != 0) {
    lean_internal_panic(msg);
  }
}

static inline void torchlean_cuda_unlock(pthread_mutex_t* mutex, const char* msg) {
  if (pthread_mutex_unlock(mutex) != 0) {
    lean_internal_panic(msg);
  }
}

// Tensor buffers and the scratch pools below share one retention budget. Each translation unit
// still owns its scratch list, but reserving bytes here charges the same process-wide counter.
// Global reclamation is called only after releasing the local pool mutex.
extern "C" bool torchlean_cuda_cache_reserve_bytes(size_t bytes);
extern "C" void torchlean_cuda_cache_release_bytes(size_t bytes);
extern "C" void torchlean_cuda_flush_all_caches(void);

struct torchlean_cuda_scratch_block {
  size_t bytes;
  void* ptr;
  cudaEvent_t ready;
};

static pthread_mutex_t g_torchlean_cuda_scratch_mutex = PTHREAD_MUTEX_INITIALIZER;
static torchlean_cuda_scratch_block* g_torchlean_cuda_scratch_cache = nullptr;
static size_t g_torchlean_cuda_scratch_count = 0;
static size_t g_torchlean_cuda_scratch_cap = 0;

static inline void torchlean_cuda_scratch_push(torchlean_cuda_scratch_block block) {
  if (g_torchlean_cuda_scratch_count == g_torchlean_cuda_scratch_cap) {
    size_t new_cap = g_torchlean_cuda_scratch_cap == 0 ? 16 : g_torchlean_cuda_scratch_cap * 2;
    if (new_cap < g_torchlean_cuda_scratch_cap) {
      lean_internal_panic("torchlean_cuda_scratch_push: capacity overflow");
    }
    const size_t bytes = checked_bytes_size(
        new_cap, sizeof(torchlean_cuda_scratch_block),
        "torchlean_cuda_scratch_push: cache byte size overflow");
    void* next = realloc(g_torchlean_cuda_scratch_cache,
                         bytes);
    if (!next) {
      lean_internal_panic_out_of_memory();
    }
    g_torchlean_cuda_scratch_cache = (torchlean_cuda_scratch_block*)next;
    g_torchlean_cuda_scratch_cap = new_cap;
  }
  g_torchlean_cuda_scratch_cache[g_torchlean_cuda_scratch_count++] = block;
}

static inline void torchlean_cuda_scratch_flush(void) {
  torchlean_cuda_lock(&g_torchlean_cuda_scratch_mutex, "pthread_mutex_lock scratch flush failed");
  torchlean_cuda_scratch_block* blocks = g_torchlean_cuda_scratch_cache;
  size_t count = g_torchlean_cuda_scratch_count;
  g_torchlean_cuda_scratch_cache = nullptr;
  g_torchlean_cuda_scratch_count = 0;
  g_torchlean_cuda_scratch_cap = 0;
  torchlean_cuda_unlock(&g_torchlean_cuda_scratch_mutex, "pthread_mutex_unlock scratch flush failed");

  for (size_t i = 0; i < count; ++i) {
    torchlean_cuda_scratch_block block = blocks[i];
    if (block.ready) {
      checkCuda(cudaEventSynchronize(block.ready), "cudaEventSynchronize cached scratch block failed");
      checkCuda(cudaEventDestroy(block.ready), "cudaEventDestroy cached scratch block failed");
    }
    torchlean_cuda_free_checked(&block.ptr, "cudaFree cached scratch block failed");
    torchlean_cuda_cache_release_bytes(block.bytes);
  }
  free(blocks);
}

static inline void* torchlean_cuda_scratch_alloc_bytes(size_t bytes, const char* msg) {
  if (bytes == 0) {
    return nullptr;
  }
  torchlean_cuda_lock(&g_torchlean_cuda_scratch_mutex, "pthread_mutex_lock scratch alloc failed");
  for (size_t i = 0; i < g_torchlean_cuda_scratch_count; ++i) {
    if (g_torchlean_cuda_scratch_cache[i].bytes != bytes) {
      continue;
    }
    cudaError_t ready = cudaEventQuery(g_torchlean_cuda_scratch_cache[i].ready);
    if (ready == cudaSuccess) {
      void* ptr = g_torchlean_cuda_scratch_cache[i].ptr;
      checkCuda(cudaEventDestroy(g_torchlean_cuda_scratch_cache[i].ready),
                "cudaEventDestroy scratch reuse event failed");
      g_torchlean_cuda_scratch_cache[i] =
        g_torchlean_cuda_scratch_cache[g_torchlean_cuda_scratch_count - 1];
      g_torchlean_cuda_scratch_count--;
      torchlean_cuda_unlock(&g_torchlean_cuda_scratch_mutex,
                            "pthread_mutex_unlock scratch alloc failed");
      torchlean_cuda_cache_release_bytes(bytes);
      return ptr;
    }
    if (ready != cudaErrorNotReady) {
      torchlean_cuda_unlock(&g_torchlean_cuda_scratch_mutex,
                            "pthread_mutex_unlock scratch alloc error failed");
      checkCuda(ready, "cudaEventQuery scratch reuse event failed");
    }
  }
  torchlean_cuda_unlock(&g_torchlean_cuda_scratch_mutex, "pthread_mutex_unlock scratch alloc failed");
  void* ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, bytes);
  if (err == cudaErrorMemoryAllocation) {
    // An allocation failure can be caused by cached blocks in a different pool. Reclaim all
    // reusable storage once, clearing the recovered OOM before any later launch-error check.
    (void)cudaGetLastError();
    torchlean_cuda_flush_all_caches();
    err = cudaMalloc(&ptr, bytes);
  }
  checkCuda(err, msg);
  return ptr;
}

static inline void torchlean_cuda_scratch_free_bytes(void** ptr, size_t bytes, const char* msg) {
  (void)msg;
  if (!ptr || !*ptr) {
    return;
  }
  if (bytes == 0) {
    torchlean_cuda_free_checked(ptr, msg);
    return;
  }
  cudaEvent_t ready = nullptr;
  checkCuda(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming),
            "cudaEventCreate cached scratch block failed");
  checkCuda(cudaEventRecord(ready, 0), "cudaEventRecord cached scratch block failed");
  if (!torchlean_cuda_cache_reserve_bytes(bytes)) {
    // A pending kernel may still read this workspace. Wait for its recorded use to finish
    // before returning an over-budget block to the driver.
    checkCuda(cudaEventSynchronize(ready), "cudaEventSynchronize uncached scratch block failed");
    checkCuda(cudaEventDestroy(ready), "cudaEventDestroy uncached scratch block failed");
    torchlean_cuda_free_checked(ptr, msg);
    return;
  }
  torchlean_cuda_lock(&g_torchlean_cuda_scratch_mutex, "pthread_mutex_lock scratch free failed");
  torchlean_cuda_scratch_push({bytes, *ptr, ready});
  torchlean_cuda_unlock(&g_torchlean_cuda_scratch_mutex, "pthread_mutex_unlock scratch free failed");
  *ptr = nullptr;
}

template <typename T>
static inline T* torchlean_cuda_scratch_alloc(size_t count, const char* msg) {
  return (T*)torchlean_cuda_scratch_alloc_bytes(checked_bytes_size(count, sizeof(T), msg), msg);
}

template <typename T>
static inline void torchlean_cuda_scratch_free(T** ptr, size_t count, const char* msg) {
  torchlean_cuda_scratch_free_bytes((void**)ptr, checked_bytes_size(count, sizeof(T), msg), msg);
}

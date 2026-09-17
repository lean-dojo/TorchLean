#pragma once

#include <lean/lean.h>

#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

#include "torchlean_cuda_buffer.h"

// Scoped device-memory arena ("withCudaArena").
//
// Why this exists.  TorchLean wraps each device allocation in a Lean external object whose finalizer
// returns the memory.  In a long *pure* eager loop (a `foldl` of `Buffer -> Buffer` ops with no IO
// sequencing) every intermediate `Buffer` stays GC-*reachable* until the final readback, so the
// finalizer never runs and device memory grows without bound.  Explicit `release` cannot help from
// inside a pure carrier: there is no IO point at which to call it.
//
// A scoped arena sidesteps GC reachability entirely.  `arena_enter` opens an allocation epoch; every
// buffer allocated while it is open is registered to the epoch.  `arena_exit` frees the device data of
// *every* buffer in the epoch -- whether or not it is still reachable -- except a small set of `keep`
// buffers (the step's results), which are promoted to the parent epoch (or untracked at the outermost
// level).  This matches the natural phase boundary of a training loop: one LM step / one fold.
//
// Registration model.  Each tracked buffer points at a heap-allocated `torchlean_arena_reg` (and the
// reg points back).  When a buffer's struct is freed early (its finalizer runs mid-scope), it flips
// the reg's `alive` flag instead of leaving a dangling pointer, so the exit walk skips it.  All
// registry mutation is serialized by `g_arena_mutex`; the fast paths (`arena_register` when no epoch
// is open, `arena_unlink` when the buffer is untracked) take no lock.
//
// Threading contract.  `arena_enter`/`arena_exit` and the allocations between them run on one driver
// thread; finalizers (which only ever *unlink*) may run on any thread and are mutex-safe.  Concurrent
// arenas on multiple threads are not supported in this first cut.

typedef struct torchlean_arena_reg {
  torchlean_cuda_buffer* b;  // owning struct; valid only while `alive`
  size_t depth;              // epoch index this reg belongs to (stack position)
  bool alive;                // false once the buffer struct has been freed or the reg neutralized
} torchlean_arena_reg;

typedef struct torchlean_arena_epoch {
  torchlean_arena_reg** regs;
  size_t count;
  size_t cap;
} torchlean_arena_epoch;

static pthread_mutex_t g_torchlean_arena_mutex = PTHREAD_MUTEX_INITIALIZER;
static torchlean_arena_epoch* g_torchlean_arena_stack = NULL;
static size_t g_torchlean_arena_depth = 0;  // number of open epochs
static size_t g_torchlean_arena_cap = 0;

static inline void torchlean_arena_epoch_push(torchlean_arena_epoch* e, torchlean_arena_reg* r) {
  if (e->count == e->cap) {
    size_t ncap = e->cap == 0 ? 16 : e->cap * 2;
    void* p = realloc(e->regs, ncap * sizeof(torchlean_arena_reg*));
    if (!p) {
      lean_internal_panic_out_of_memory();
    }
    e->regs = (torchlean_arena_reg**)p;
    e->cap = ncap;
  }
  e->regs[e->count++] = r;
}

// Open a new allocation epoch.  Subsequent allocations on this thread are registered to it until the
// matching `torchlean_arena_exit`.
static inline void torchlean_arena_enter(void) {
  pthread_mutex_lock(&g_torchlean_arena_mutex);
  if (g_torchlean_arena_depth == g_torchlean_arena_cap) {
    size_t ncap = g_torchlean_arena_cap == 0 ? 4 : g_torchlean_arena_cap * 2;
    void* p = realloc(g_torchlean_arena_stack, ncap * sizeof(torchlean_arena_epoch));
    if (!p) {
      pthread_mutex_unlock(&g_torchlean_arena_mutex);
      lean_internal_panic_out_of_memory();
    }
    g_torchlean_arena_stack = (torchlean_arena_epoch*)p;
    g_torchlean_arena_cap = ncap;
  }
  torchlean_arena_epoch* e = &g_torchlean_arena_stack[g_torchlean_arena_depth++];
  e->regs = NULL;
  e->count = 0;
  e->cap = 0;
  pthread_mutex_unlock(&g_torchlean_arena_mutex);
}

// Register a freshly allocated buffer to the current epoch (no-op when no arena is open or the buffer
// holds no device memory).  Sets `b->arena_reg`.
static inline void torchlean_arena_register(torchlean_cuda_buffer* b) {
  if (g_torchlean_arena_depth == 0 || !b || !b->data) {
    return;  // fast path: nothing to track
  }
  pthread_mutex_lock(&g_torchlean_arena_mutex);
  if (g_torchlean_arena_depth == 0) {
    pthread_mutex_unlock(&g_torchlean_arena_mutex);
    return;
  }
  torchlean_arena_reg* r = (torchlean_arena_reg*)malloc(sizeof(torchlean_arena_reg));
  if (!r) {
    pthread_mutex_unlock(&g_torchlean_arena_mutex);
    lean_internal_panic_out_of_memory();
  }
  r->b = b;
  r->depth = g_torchlean_arena_depth - 1;
  r->alive = true;
  b->arena_reg = r;
  torchlean_arena_epoch_push(&g_torchlean_arena_stack[g_torchlean_arena_depth - 1], r);
  pthread_mutex_unlock(&g_torchlean_arena_mutex);
}

// Detach a buffer from the registry because its struct is about to be freed.  Flips the reg `alive`
// flag (so a concurrent exit walk skips it) and clears the back-pointer.  No-op for untracked buffers.
static inline void torchlean_arena_unlink(torchlean_cuda_buffer* b) {
  if (!b || !b->arena_reg) {
    return;  // fast path: not arena-tracked
  }
  pthread_mutex_lock(&g_torchlean_arena_mutex);
  torchlean_arena_reg* r = (torchlean_arena_reg*)b->arena_reg;
  if (r) {
    r->alive = false;
  }
  b->arena_reg = NULL;
  pthread_mutex_unlock(&g_torchlean_arena_mutex);
}

// Close the current epoch.  Releases the device data of every still-live buffer allocated in it,
// except the `nkeep` buffers in `keep`, which are promoted to the parent epoch (or untracked at the
// outermost level).  `release_data` is the backend's per-buffer reclaim (cache-return on CUDA, free on
// the CPU stub).  An unbalanced exit (no open epoch) is ignored.
static inline void torchlean_arena_exit(torchlean_cuda_buffer** keep, size_t nkeep,
                                        bool (*release_data)(torchlean_cuda_buffer*)) {
  pthread_mutex_lock(&g_torchlean_arena_mutex);
  if (g_torchlean_arena_depth == 0) {
    pthread_mutex_unlock(&g_torchlean_arena_mutex);
    return;
  }
  size_t exited = --g_torchlean_arena_depth;
  torchlean_arena_epoch e = g_torchlean_arena_stack[exited];
  bool has_parent = g_torchlean_arena_depth > 0;

  // Promote kept buffers out of the exiting epoch so the walk below leaves their data alone.  A kept
  // buffer is recognized by its reg living at the exiting depth; one tracked elsewhere (or not at all)
  // is ignored.
  for (size_t k = 0; k < nkeep; ++k) {
    torchlean_cuda_buffer* kb = keep[k];
    if (!kb || !kb->arena_reg) {
      continue;
    }
    torchlean_arena_reg* old = (torchlean_arena_reg*)kb->arena_reg;
    if (old->depth != exited) {
      continue;  // belongs to an ancestor epoch; not ours to promote
    }
    old->alive = false;  // neutralize: the walk frees this reg without releasing data
    if (has_parent) {
      torchlean_arena_reg* nr = (torchlean_arena_reg*)malloc(sizeof(torchlean_arena_reg));
      if (!nr) {
        pthread_mutex_unlock(&g_torchlean_arena_mutex);
        lean_internal_panic_out_of_memory();
      }
      nr->b = kb;
      nr->depth = g_torchlean_arena_depth - 1;
      nr->alive = true;
      kb->arena_reg = nr;
      torchlean_arena_epoch_push(&g_torchlean_arena_stack[g_torchlean_arena_depth - 1], nr);
    } else {
      kb->arena_reg = NULL;  // untrack: managed by RC / finalizer from here on
    }
  }

  for (size_t i = 0; i < e.count; ++i) {
    torchlean_arena_reg* r = e.regs[i];
    if (r->alive) {
      torchlean_cuda_buffer* b = r->b;  // struct still valid while alive
      b->arena_reg = NULL;              // detach before freeing the reg (finalizer-safe)
      (void)release_data(b);
      if (torchlean_arena_debug_enabled()) {
        b->arena_freed_depth = exited + 1;  // record the reclaiming epoch for the UAF detector
      }
    }
    free(r);
  }
  free(e.regs);
  pthread_mutex_unlock(&g_torchlean_arena_mutex);
}

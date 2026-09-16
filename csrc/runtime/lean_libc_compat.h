// Lean's Linux linker sysroot predates glibc's C23 strtol entry points.
// Keep the allocator's option parser on the same pre-C23 ABI as Lean.
#include <features.h>
#ifdef __GLIBC_USE_C2X_STRTOL
#undef __GLIBC_USE_C2X_STRTOL
#define __GLIBC_USE_C2X_STRTOL 0
#endif
#ifdef __GLIBC_USE_C23_STRTOL
#undef __GLIBC_USE_C23_STRTOL
#define __GLIBC_USE_C23_STRTOL 0
#endif

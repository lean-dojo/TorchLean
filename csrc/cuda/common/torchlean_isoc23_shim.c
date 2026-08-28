/*
 * glibc >= 2.38 isoc23 link shim for the CUDA build.
 *
 * On glibc >= 2.38 the `strto*` family is redirected to `__isoc23_strto*` under
 * `_GNU_SOURCE` (which nvcc's host pass forces). nvcc-compiled host objects then
 * reference `__isoc23_strtoull`/etc. When the final executable is linked against an
 * *older* glibc than the one nvcc compiled with -- which is the case whenever the Lean
 * toolchain bundles a pre-2.38 glibc -- those symbols are undefined and the CUDA link
 * fails (`undefined reference to __isoc23_strtoull`).
 *
 * This object *defines* the referenced names as thin wrappers over the plain, still
 * resolvable `strto*` (declared directly below so this file itself does not go through
 * the C23 redirect). The definitions are `weak`, so a link-time glibc that already
 * exports the real `__isoc23_strto*` wins and this shim is ignored; when it does not,
 * these satisfy the references. It is compiled and linked only into the CUDA build, and
 * is inert on any host whose objects never reference these names (older glibc, non-CUDA).
 *
 * A `-Wl,--defsym=__isoc23_strtoull=strtoull` alias does NOT work here: lld resolves the
 * defsym target among *defined* symbols, but with the C23 redirect nothing references the
 * plain `strtoull`, so lld reports `symbol not found: strtoull`. A real definition (this
 * file) is required.
 */

extern unsigned long long strtoull(const char *, char **, int);
extern long long          strtoll (const char *, char **, int);
extern unsigned long      strtoul (const char *, char **, int);
extern long               strtol  (const char *, char **, int);

__attribute__((weak)) unsigned long long __isoc23_strtoull(const char *p, char **e, int b) {
  return strtoull(p, e, b);
}
__attribute__((weak)) long long __isoc23_strtoll(const char *p, char **e, int b) {
  return strtoll(p, e, b);
}
__attribute__((weak)) unsigned long __isoc23_strtoul(const char *p, char **e, int b) {
  return strtoul(p, e, b);
}
__attribute__((weak)) long __isoc23_strtol(const char *p, char **e, int b) {
  return strtol(p, e, b);
}

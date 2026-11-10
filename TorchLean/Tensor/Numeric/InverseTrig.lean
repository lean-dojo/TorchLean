/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# Inverse trigonometric functions of real coordinates

A complex logarithm needs the argument of a point as well as real logarithms and square roots.
`InverseTrig` supplies that angle as an operation on real coordinates, so complex numbers need no
ordering or angle operation of their own. It also carries the arctangent, arcsine and arccosine,
each a separate field so that a scalar type supplies the implementation it has. The host `Float`
and `Float32` instances are the C library's functions, with its signed-zero and branch-cut
conventions.

We took the `y, x` order and the ranges by hand from C99 (7.12.4.1 to 7.12.4.4 of WG14 N1256).
C closes the `atan2` range at `-π`, and that end is the `-0` case the `atan2` field describes.
Lean 4.35.0 binds `Float.atan2` to C's `atan2` with `@[extern "atan2"]`, and `Float32.atan2` to
`atan2f`.
-/

@[expose] public section

namespace TorchLean

/-- The principal polar angle of a point and the inverse trigonometric functions, in radians. The
ranges below hold over the reals, and a rounded result can leave them: in binary32, `atan ∞` is
above `π/2` and `acos (-1)` is above `π`. -/
class InverseTrig (α : Type) where
  /-- The polar angle of the point `(x, y)`. The argument order is `y, x`, as in the standard
  two-argument arctangent. Over the reals it is the principal argument in `(-π, π]`. IEEE formats
  give the rounded `-π` for a `y` of `-0` against a negative `x`. -/
  atan2 : α → α → α
  /-- The arctangent, in `[-π/2, π/2]` over the reals. -/
  atan : α → α
  /-- The arcsine, in `[-π/2, π/2]` on `[-1, 1]` over the reals. -/
  asin : α → α
  /-- The arccosine, in `[0, π]` on `[-1, 1]` over the reals. -/
  acos : α → α

/-- The C library's `atan2`, `atan`, `asin` and `acos` in binary64. -/
instance : InverseTrig Float := ⟨Float.atan2, Float.atan, Float.asin, Float.acos⟩

/-- The C library's `atan2f`, `atanf`, `asinf` and `acosf` in binary32. -/
instance : InverseTrig Float32 := ⟨Float32.atan2, Float32.atan, Float32.asin, Float32.acos⟩

end TorchLean

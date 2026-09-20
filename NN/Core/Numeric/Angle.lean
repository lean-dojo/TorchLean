/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# Polar angle of real coordinates

Complex logarithms need the argument of a point, in addition to real logarithms and square roots.
`Atan2` supplies this real-coordinate operation without requiring an ordering or an angle operation
on complex numbers themselves. Floating-point instances follow the host library's signed-zero and
branch-cut conventions.
-/

@[expose] public section

/-- Principal polar angle of the point `(x, y)`, in radians. -/
class Atan2 (α : Type) where
  /-- The argument order is `y, x`, as in the standard two-argument arctangent. -/
  atan2 : α → α → α

instance : Atan2 Float := ⟨Float.atan2⟩
instance : Atan2 Float32 := ⟨Float32.atan2⟩

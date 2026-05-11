/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Init.Data.Float32
public import NN.Floats.IEEEExec.Exec32

/-!
# BridgeInitFloat32

External (assumption-based) bridge: Lean's `Init.Float32` ↔ `IEEE32Exec`.

Why assumptions are necessary:
`Init.Float32` arithmetic in Lean is implemented by *external* runtime calls. Those calls are opaque
to the Lean kernel, so (inside Lean) we cannot prove that the runtime implementation coincides
bit-for-bit with any particular float32 specification.

We package the intended connection as a typeclass interface. If you (or your trusted runtime) can
discharge the assumptions that the runtime `Float32` primitives match the executable kernel
`IEEE32Exec` bit-for-bit, then you can:

1) execute with `Float32` (runtime),
2) rewrite the result to `IEEE32Exec`,
3) reuse the internal refinement theorems (`BridgeFP32.lean` / `BridgeFP32Expr.lean`) to connect
   execution to the `FP32` rounding-on-`ℝ` model on finite/no-overflow inputs.

This keeps the trust boundary explicit: the only unproved part is the external/runtime correctness
assumption, which is unavoidable in a pure Lean development.

Background:
- IEEE 754-2019 (what it means to “match float32 semantics”):
  https://doi.org/10.1109/IEEESTD.2019.8766229
-/

@[expose] public section


namespace TorchLean.Floats.IEEE754

open IEEE32Exec

namespace Float32Bridge

/-- Local alias for Lean's runtime `Float32` type. -/
abbrev F32 := Float32

/-- Reinterpret a runtime `Float32` as the executable bit-level float32 (`IEEE32Exec`). -/
@[inline] def toIEEE32Exec (x : F32) : IEEE32Exec :=
  IEEE32Exec.ofBits x.toBits

/-- Reinterpret an executable float32 bit-pattern as a runtime `Float32`. -/
@[inline] def ofIEEE32Exec (x : IEEE32Exec) : F32 :=
  Float32.ofBits x.bits

/-!
## What this bridge gives us

This file does *not* prove anything about the runtime semantics of `Float32`. Instead, it gives a
clean *interface* that you can assume/provide:

- If the runtime `Float32` primitives produce the same result bits as `IEEE32Exec`,
  then runtime evaluation can be rewritten into `IEEE32Exec` evaluation.
- Once we are in `IEEE32Exec`, we can use the internal bridge theorems to connect execution to the
  `FP32` rounding-on-`ℝ` model on the finite/no-overflow path (`BridgeFP32.lean`,
  `BridgeFP32Total.lean`, `BridgeFP32Expr.lean`).

We keep this separation because it makes the trust boundary explicit: the only “axioms” are the
bit-level runtime correctness assumptions below.
-/

/-! ## External correctness assumptions -/

/-- Assumption package relating Lean's runtime `Float32` primitives to `IEEE32Exec`. -/
class RuntimeFloat32MatchesIEEE32Exec : Prop where
  toBits_ofBits : ∀ b : UInt32, (Float32.ofBits b).toBits = b
  ofBits_toBits : ∀ x : F32, Float32.ofBits x.toBits = x

  add_bits : ∀ a b : F32,
    (Float32.add a b).toBits = (IEEE32Exec.add (toIEEE32Exec a) (toIEEE32Exec b)).bits
  sub_bits : ∀ a b : F32,
    (Float32.sub a b).toBits = (IEEE32Exec.sub (toIEEE32Exec a) (toIEEE32Exec b)).bits
  mul_bits : ∀ a b : F32,
    (Float32.mul a b).toBits = (IEEE32Exec.mul (toIEEE32Exec a) (toIEEE32Exec b)).bits
  div_bits : ∀ a b : F32,
    (Float32.div a b).toBits = (IEEE32Exec.div (toIEEE32Exec a) (toIEEE32Exec b)).bits
  neg_bits : ∀ a : F32,
    (Float32.neg a).toBits = (IEEE32Exec.neg (toIEEE32Exec a)).bits
  sqrt_bits : ∀ a : F32,
    (Float32.sqrt a).toBits = (IEEE32Exec.sqrt (toIEEE32Exec a)).bits

  isNaN_bits : ∀ a : F32,
    Float32.isNaN a = IEEE32Exec.isNaN (toIEEE32Exec a)
  isInf_bits : ∀ a : F32,
    Float32.isInf a = IEEE32Exec.isInf (toIEEE32Exec a)
  isFinite_bits : ∀ a : F32,
    Float32.isFinite a = IEEE32Exec.isFinite (toIEEE32Exec a)

/-! ## Derived bit-level refinement lemmas -/

namespace RuntimeFloat32MatchesIEEE32Exec

variable [RuntimeFloat32MatchesIEEE32Exec]

/-!
## Derived lemmas (rewriting runtime to executable)

The assumptions above are phrased as bit equalities. In practice we almost always want the more
convenient *value-level* rewriting lemmas below:

`toIEEE32Exec (a + b) = IEEE32Exec.add (toIEEE32Exec a) (toIEEE32Exec b)`, etc.

These are the lemmas you use to “turn a runtime evaluation into an `IEEE32Exec` evaluation”.
-/

-- `IEEE32Exec` is a thin wrapper around a `UInt32` bit pattern; equality is extensional on `.bits`.
omit [RuntimeFloat32MatchesIEEE32Exec] in
private theorem bits_inj {x y : IEEE32Exec} (h : x.bits = y.bits) : x = y := by
  cases x
  cases y
  cases h
  rfl

-- We write these theorems using the usual notation (`a + b`, `a * b`) because that is how most
-- downstream code is written; the assumptions are stated in terms of `Float32.add`, etc.
/-- Rewrite runtime float32 addition into executable `IEEE32Exec.add`. -/
theorem toIEEE32Exec_add (a b : F32) :
    toIEEE32Exec (a + b) = IEEE32Exec.add (toIEEE32Exec a) (toIEEE32Exec b) := by
  apply bits_inj
  simpa [toIEEE32Exec, IEEE32Exec.ofBits] using
    (RuntimeFloat32MatchesIEEE32Exec.add_bits (a := a) (b := b))

/-- Rewrite runtime float32 subtraction into executable `IEEE32Exec.sub` (value-level form). -/
theorem toIEEE32Exec_sub (a b : F32) :
    toIEEE32Exec (a - b) = IEEE32Exec.sub (toIEEE32Exec a) (toIEEE32Exec b) := by
  apply bits_inj
  simpa [toIEEE32Exec, IEEE32Exec.ofBits] using
    (RuntimeFloat32MatchesIEEE32Exec.sub_bits (a := a) (b := b))

/-- Rewrite runtime float32 multiplication into executable `IEEE32Exec.mul` (value-level form). -/
theorem toIEEE32Exec_mul (a b : F32) :
    toIEEE32Exec (a * b) = IEEE32Exec.mul (toIEEE32Exec a) (toIEEE32Exec b) := by
  apply bits_inj
  simpa [toIEEE32Exec, IEEE32Exec.ofBits] using
    (RuntimeFloat32MatchesIEEE32Exec.mul_bits (a := a) (b := b))

/-- Rewrite runtime float32 division into executable `IEEE32Exec.div` (value-level form). -/
theorem toIEEE32Exec_div (a b : F32) :
    toIEEE32Exec (a / b) = IEEE32Exec.div (toIEEE32Exec a) (toIEEE32Exec b) := by
  apply bits_inj
  simpa [toIEEE32Exec, IEEE32Exec.ofBits] using
    (RuntimeFloat32MatchesIEEE32Exec.div_bits (a := a) (b := b))

/-- Rewrite runtime float32 negation into executable `IEEE32Exec.neg` (value-level form). -/
theorem toIEEE32Exec_neg (a : F32) :
    toIEEE32Exec (-a) = IEEE32Exec.neg (toIEEE32Exec a) := by
  apply bits_inj
  simpa [toIEEE32Exec, IEEE32Exec.ofBits] using
    (RuntimeFloat32MatchesIEEE32Exec.neg_bits (a := a))

/-- Rewrite runtime float32 square root into executable `IEEE32Exec.sqrt` (value-level form). -/
theorem toIEEE32Exec_sqrt (a : F32) :
    toIEEE32Exec (Float32.sqrt a) = IEEE32Exec.sqrt (toIEEE32Exec a) := by
  apply bits_inj
  simpa [toIEEE32Exec, IEEE32Exec.ofBits] using
    (RuntimeFloat32MatchesIEEE32Exec.sqrt_bits (a := a))

/-- Converting an `IEEE32Exec` value to runtime `Float32` and back returns the original bits. -/
theorem toIEEE32Exec_ofIEEE32Exec (x : IEEE32Exec) :
    toIEEE32Exec (ofIEEE32Exec x) = x := by
  apply bits_inj
  simpa [toIEEE32Exec, ofIEEE32Exec, IEEE32Exec.ofBits] using
    (RuntimeFloat32MatchesIEEE32Exec.toBits_ofBits (b := x.bits))

end RuntimeFloat32MatchesIEEE32Exec

end Float32Bridge

end TorchLean.Floats.IEEE754

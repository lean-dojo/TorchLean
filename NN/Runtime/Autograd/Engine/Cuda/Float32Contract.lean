/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.ErrorBounds
public import NN.Floats.IEEEExec.BridgeInitFloat32

/-!
# CUDA float32 contract

TorchLean's CUDA eager backend stores native `float` values in an opaque FFI buffer. Lean cannot
look inside CUDA kernels, C casts, libdevice calls, or cuBLAS, so the native backend is necessarily a
trusted/validated implementation boundary.

This module keeps that boundary precise:

- `IEEE32Exec` is the executable, bit-level reference model for scalar binary32.
- host `Float` inputs enter the float32 world through `IEEE32Exec.ofFloat`, matching the intended
  "round binary64 host literals to binary32" contract;
- external/native CUDA scalar results are represented only by their raw 32-bit result bits;
- if those native bits agree with the `IEEE32Exec` reference op, then the existing proved
  `IEEE32Exec → FP32-on-ℝ` theorems apply immediately.

In other words, the proof stack is:

`native CUDA bits` --(explicit agreement assumption / tests / toolchain contract)-->
`IEEE32Exec` --(proved in Lean)--> `FP32` rounding-on-`ℝ` error bounds.

What is *not* proved here:

- that a particular compiled CUDA kernel, C compiler, device, libdevice implementation, or cuBLAS
  version produces the reference bits;
- deterministic ordering for atomic reductions unless the backend uses a fixed reduction tree;
- correct-rounding for transcendental functions that IEEE-754 itself does not specify.

Those are runtime/toolchain assumptions, and the CUDA stress tests are intended to validate them
against this reference contract.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Cuda
namespace Float32Contract

open TorchLean.Floats
open TorchLean.Floats.IEEE754

noncomputable section

/-! ## Reference scalar and host conversion -/

/--
The scalar reference for CUDA float32 reasoning.

CUDA buffers are opaque to Lean; this is the scalar model we compare their 32-bit elements against.
-/
abbrev RefScalar := IEEE32Exec

/-- Interpret raw native binary32 bits as the `IEEE32Exec` reference scalar. -/
@[inline] def fromNativeBits (bits : UInt32) : RefScalar :=
  IEEE32Exec.ofBits bits

/-- Extract the binary32 bit pattern used for native/reference comparisons. -/
@[inline] def toNativeBits (x : RefScalar) : UInt32 :=
  IEEE32Exec.toBits x

/--
Reference meaning of uploading a Lean `Float` into a CUDA float32 buffer.

Lean `Float` is binary64. The CUDA buffer path casts host doubles to native `float`; the reference
contract for that cast is round-to-nearest-even binary32, implemented by `IEEE32Exec.ofFloat`.
-/
@[inline] def fromLeanFloat (x : Float) : RefScalar :=
  IEEE32Exec.ofFloat x

/--
Reference meaning of downloading a finite CUDA float32 element into Lean `Float`.

Every finite binary32 value embeds exactly in binary64. NaN/Inf are mapped to the canonical Lean
`Float` NaN/Inf values chosen by `IEEE32Exec.toFloat`.
-/
@[inline] def toLeanFloat (x : RefScalar) : Float :=
  IEEE32Exec.toFloat x

@[simp] theorem toNativeBits_fromNativeBits (bits : UInt32) :
    toNativeBits (fromNativeBits bits) = bits := by
  rfl

@[simp] theorem fromNativeBits_toNativeBits (x : RefScalar) :
    fromNativeBits (toNativeBits x) = x := by
  exact IEEE32Exec.ofBits_toBits x

/-- Host `Float` upload is exactly the `IEEE32Exec.ofFloat` conversion. -/
theorem fromLeanFloat_eq_ieee32_ofFloat (x : Float) :
    fromLeanFloat x = IEEE32Exec.ofFloat x := by
  rfl

/-- Host `Float` upload bits are exactly the reference binary32 conversion bits. -/
theorem fromLeanFloat_bits_eq_ieee32_ofFloat_bits (x : Float) :
    toNativeBits (fromLeanFloat x) = IEEE32Exec.toBits (IEEE32Exec.ofFloat x) := by
  rfl

/-- Runtime Lean `Float32` values can also be reinterpreted as the same bit-level reference scalar. -/
theorem runtimeFloat32_toRef_eq_bridge (x : Float32) :
    fromNativeBits x.toBits = Float32Bridge.toIEEE32Exec x := by
  rfl

/-! ## Abstract native scalar semantics -/

/--
Abstract result bits for native CUDA scalar primitives.

This deliberately does not claim that CUDA has been proved correct in Lean. It provides an explicit comparison point where
the FFI/runtime implementation can be compared against the `IEEE32Exec` reference, one result bit
pattern at a time. Vector/tensor kernels lift this elementwise, except reductions whose order must
also be specified.
-/
structure NativePrimitiveBits where
  addBits : RefScalar → RefScalar → UInt32
  mulBits : RefScalar → RefScalar → UInt32
  divBits : RefScalar → RefScalar → UInt32
  fmaBits : RefScalar → RefScalar → RefScalar → UInt32
  sqrtBits : RefScalar → UInt32

/--
The trusted/validated CUDA scalar agreement assumption.

For a concrete CUDA build, these fields are what parity tests, compiler flags, and backend policy are
checking: native result bits match the executable `IEEE32Exec` reference for primitive float32 ops.
-/
structure NativePrimitiveAgreement (native : NativePrimitiveBits) : Prop where
  add_bits : ∀ x y, native.addBits x y = toNativeBits (IEEE32Exec.add x y)
  mul_bits : ∀ x y, native.mulBits x y = toNativeBits (IEEE32Exec.mul x y)
  div_bits : ∀ x y, native.divBits x y = toNativeBits (IEEE32Exec.div x y)
  fma_bits : ∀ x y z, native.fmaBits x y z = toNativeBits (IEEE32Exec.fma x y z)
  sqrt_bits : ∀ x, native.sqrtBits x = toNativeBits (IEEE32Exec.sqrt x)

private theorem ref_ext {x y : RefScalar} (h : toNativeBits x = toNativeBits y) : x = y := by
  cases x
  cases y
  cases h
  rfl

variable {native : NativePrimitiveBits}

/-- Native addition agrees with the reference value when its result bits satisfy the contract. -/
theorem native_add_eq_ieee32 (h : NativePrimitiveAgreement native) (x y : RefScalar) :
    fromNativeBits (native.addBits x y) = IEEE32Exec.add x y := by
  apply ref_ext
  simp [fromNativeBits, toNativeBits, h.add_bits x y]

/-- Native multiplication agrees with the reference value when its result bits satisfy the contract. -/
theorem native_mul_eq_ieee32 (h : NativePrimitiveAgreement native) (x y : RefScalar) :
    fromNativeBits (native.mulBits x y) = IEEE32Exec.mul x y := by
  apply ref_ext
  simp [fromNativeBits, toNativeBits, h.mul_bits x y]

/-- Native division agrees with the reference value when its result bits satisfy the contract. -/
theorem native_div_eq_ieee32 (h : NativePrimitiveAgreement native) (x y : RefScalar) :
    fromNativeBits (native.divBits x y) = IEEE32Exec.div x y := by
  apply ref_ext
  simp [fromNativeBits, toNativeBits, h.div_bits x y]

/-- Native fused multiply-add agrees with the reference value when its result bits satisfy the contract. -/
theorem native_fma_eq_ieee32 (h : NativePrimitiveAgreement native) (x y z : RefScalar) :
    fromNativeBits (native.fmaBits x y z) = IEEE32Exec.fma x y z := by
  apply ref_ext
  simp [fromNativeBits, toNativeBits, h.fma_bits x y z]

/-- Native square root agrees with the reference value when its result bits satisfy the contract. -/
theorem native_sqrt_eq_ieee32 (h : NativePrimitiveAgreement native) (x : RefScalar) :
    fromNativeBits (native.sqrtBits x) = IEEE32Exec.sqrt x := by
  apply ref_ext
  simp [fromNativeBits, toNativeBits, h.sqrt_bits x]

/-! ## Inheriting the proved `IEEE32Exec → FP32` bounds -/

/--
If native CUDA addition matches the `IEEE32Exec` result bits and the result is finite, then it has
the standard binary32 half-ULP absolute error bound against real addition.
-/
theorem native_add_abs_error_of_isFinite
    (h : NativePrimitiveAgreement native) (x y : RefScalar)
    (hfin : IEEE32Exec.isFinite (fromNativeBits (native.addBits x y)) = true) :
    _root_.abs
        (IEEE32Exec.toReal (fromNativeBits (native.addBits x y)) -
          (IEEE32Exec.toReal x + IEEE32Exec.toReal y)) ≤
      eps₃₂ (IEEE32Exec.toReal x + IEEE32Exec.toReal y) := by
  have hx : fromNativeBits (native.addBits x y) = IEEE32Exec.add x y :=
    native_add_eq_ieee32 h x y
  rw [hx] at hfin ⊢
  exact IEEE32Exec.toReal_add_abs_error_of_isFinite x y hfin

/--
If native CUDA multiplication matches the `IEEE32Exec` result bits and the result is finite, then it
has the standard binary32 half-ULP absolute error bound against real multiplication.
-/
theorem native_mul_abs_error_of_isFinite
    (h : NativePrimitiveAgreement native) (x y : RefScalar)
    (hfin : IEEE32Exec.isFinite (fromNativeBits (native.mulBits x y)) = true) :
    _root_.abs
        (IEEE32Exec.toReal (fromNativeBits (native.mulBits x y)) -
          (IEEE32Exec.toReal x * IEEE32Exec.toReal y)) ≤
      eps₃₂ (IEEE32Exec.toReal x * IEEE32Exec.toReal y) := by
  have hx : fromNativeBits (native.mulBits x y) = IEEE32Exec.mul x y :=
    native_mul_eq_ieee32 h x y
  rw [hx] at hfin ⊢
  exact IEEE32Exec.toReal_mul_abs_error_of_isFinite x y hfin

/--
If native CUDA division matches the `IEEE32Exec` result bits and the result is finite, then it has
the standard binary32 half-ULP absolute error bound against real division.
-/
theorem native_div_abs_error_of_isFinite
    (h : NativePrimitiveAgreement native) (x y : RefScalar)
    (hfin : IEEE32Exec.isFinite (fromNativeBits (native.divBits x y)) = true) :
    _root_.abs
        (IEEE32Exec.toReal (fromNativeBits (native.divBits x y)) -
          (IEEE32Exec.toReal x / IEEE32Exec.toReal y)) ≤
      eps₃₂ (IEEE32Exec.toReal x / IEEE32Exec.toReal y) := by
  have hx : fromNativeBits (native.divBits x y) = IEEE32Exec.div x y :=
    native_div_eq_ieee32 h x y
  rw [hx] at hfin ⊢
  exact IEEE32Exec.toReal_div_abs_error_of_isFinite x y hfin

/--
If native CUDA FMA matches the `IEEE32Exec` result bits and the result is finite, then it has the
standard binary32 half-ULP absolute error bound against real `x*y+z`.
-/
theorem native_fma_abs_error_of_isFinite
    (h : NativePrimitiveAgreement native) (x y z : RefScalar)
    (hfin : IEEE32Exec.isFinite (fromNativeBits (native.fmaBits x y z)) = true) :
    _root_.abs
        (IEEE32Exec.toReal (fromNativeBits (native.fmaBits x y z)) -
          (IEEE32Exec.toReal x * IEEE32Exec.toReal y + IEEE32Exec.toReal z)) ≤
      eps₃₂ (IEEE32Exec.toReal x * IEEE32Exec.toReal y + IEEE32Exec.toReal z) := by
  have hx : fromNativeBits (native.fmaBits x y z) = IEEE32Exec.fma x y z :=
    native_fma_eq_ieee32 h x y z
  rw [hx] at hfin ⊢
  exact IEEE32Exec.toReal_fma_abs_error_of_isFinite x y z hfin

/--
If native CUDA square root matches the `IEEE32Exec` result bits and the result is finite, then it
has the standard binary32 half-ULP absolute error bound against real square root.
-/
theorem native_sqrt_abs_error_of_isFinite
    (h : NativePrimitiveAgreement native) (x : RefScalar)
    (hfin : IEEE32Exec.isFinite (fromNativeBits (native.sqrtBits x)) = true) :
    _root_.abs
        (IEEE32Exec.toReal (fromNativeBits (native.sqrtBits x)) -
          Real.sqrt (IEEE32Exec.toReal x)) ≤
      eps₃₂ (Real.sqrt (IEEE32Exec.toReal x)) := by
  have hx : fromNativeBits (native.sqrtBits x) = IEEE32Exec.sqrt x :=
    native_sqrt_eq_ieee32 h x
  rw [hx] at hfin ⊢
  exact IEEE32Exec.toReal_sqrt_abs_error_of_isFinite x hfin

end

end Float32Contract
end Cuda
end Autograd
end Runtime

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Floats.Formats.Flocq

public import NN.Floats.FP32.Notation
import Mathlib.Algebra.Order.Algebra
import NN.Floats.FP32.Error

/-!
# `FP32` interval enclosures

This module packages per-op absolute error bounds into the convenient enclosure form:

`exact ∈ Icc (approx - eps) (approx + eps)`.
-/

@[expose] public section

open FloatLib.Numerics FloatLib.Floats.Formats.Flocq


namespace TorchLean.Floats
namespace FP32

/-! ## Interval enclosures (soundness-friendly form) -/

/--
Enclosure for `exp` in the `FP32` proof model.

The exact real value `Real.exp a.val` lies in the interval
`[exp(a) - ulp/2, exp(a) + ulp/2]`, where `exp(a)` is the `FP32`-rounded value.
-/
theorem exp_mem_Icc (a : FP32) :
    Real.exp a.val ∈
      Set.Icc
        ((FloatLib.Numerics.MathFunctions.exp a).val - eps32 (Real.exp a.val))
        ((FloatLib.Numerics.MathFunctions.exp a).val + eps32 (Real.exp a.val)) := by
  have h := (abs_le).1 (exp_abs_error (a := a))
  constructor
  · linarith [h.2]
  · linarith [h.1]

/-- Enclosure for `tanh` in the `FP32` proof model (`Real.tanh` rounded once). -/
theorem tanh_mem_Icc (a : FP32) :
    Real.tanh a.val ∈
      Set.Icc
        ((FloatLib.Numerics.MathFunctions.tanh a).val - eps32 (Real.tanh a.val))
        ((FloatLib.Numerics.MathFunctions.tanh a).val + eps32 (Real.tanh a.val)) := by
  have h := (abs_le).1 (tanh_abs_error (a := a))
  constructor
  · linarith [h.2]
  · linarith [h.1]

/-- Enclosure for `log` in the `FP32` proof model (`Real.log` rounded once). -/
theorem log_mem_Icc (a : FP32) :
    Real.log a.val ∈
      Set.Icc
        ((FloatLib.Numerics.MathFunctions.log a).val - eps32 (Real.log a.val))
        ((FloatLib.Numerics.MathFunctions.log a).val + eps32 (Real.log a.val)) := by
  have h := (abs_le).1 (log_abs_error (a := a))
  constructor
  · linarith [h.2]
  · linarith [h.1]

/-- Enclosure for `cos` in the `FP32` proof model (`Real.cos` rounded once). -/
theorem cos_mem_Icc (a : FP32) :
    Real.cos a.val ∈
      Set.Icc
        ((FloatLib.Numerics.MathFunctions.cos a).val - eps32 (Real.cos a.val))
        ((FloatLib.Numerics.MathFunctions.cos a).val + eps32 (Real.cos a.val)) := by
  have h := (abs_le).1 (cos_abs_error (a := a))
  constructor <;> linarith [h.2, h.1]

/-- Enclosure for `sin` in the `FP32` proof model (`Real.sin` rounded once). -/
theorem sin_mem_Icc (a : FP32) :
    Real.sin a.val ∈
      Set.Icc
        ((FloatLib.Numerics.MathFunctions.sin a).val - eps32 (Real.sin a.val))
        ((FloatLib.Numerics.MathFunctions.sin a).val + eps32 (Real.sin a.val)) := by
  have h := (abs_le).1 (sin_abs_error (a := a))
  constructor <;> linarith [h.2, h.1]

/-- Enclosure for `sinh` in the `FP32` proof model (`Real.sinh` rounded once). -/
theorem sinh_mem_Icc (a : FP32) :
    Real.sinh a.val ∈
      Set.Icc
        ((FloatLib.Numerics.MathFunctions.sinh a).val - eps32 (Real.sinh a.val))
        ((FloatLib.Numerics.MathFunctions.sinh a).val + eps32 (Real.sinh a.val)) := by
  have h := (abs_le).1 (sinh_abs_error (a := a))
  constructor <;> linarith [h.2, h.1]

/-- Enclosure for `cosh` in the `FP32` proof model (`Real.cosh` rounded once). -/
theorem cosh_mem_Icc (a : FP32) :
    Real.cosh a.val ∈
      Set.Icc
        ((FloatLib.Numerics.MathFunctions.cosh a).val - eps32 (Real.cosh a.val))
        ((FloatLib.Numerics.MathFunctions.cosh a).val + eps32 (Real.cosh a.val)) := by
  have h := (abs_le).1 (cosh_abs_error (a := a))
  constructor <;> linarith [h.2, h.1]

/-- Enclosure for `sqrt` in the `FP32` proof model (`Real.sqrt` rounded once). -/
theorem sqrt_mem_Icc (a : FP32) :
    Real.sqrt a.val ∈
      Set.Icc
        ((FloatLib.Numerics.MathFunctions.sqrt a).val - eps32 (Real.sqrt a.val))
        ((FloatLib.Numerics.MathFunctions.sqrt a).val + eps32 (Real.sqrt a.val)) := by
  have h := (abs_le).1 (sqrt_abs_error (a := a))
  constructor <;> linarith [h.2, h.1]

/--
Enclosure for `abs` in the `FP32` proof model.

Even though `|x|` is exact over `Real`, the `FP32` proof semantics rounds after each primitive.
-/
theorem abs_mem_Icc (a : FP32) :
    |a.val| ∈
      Set.Icc
        ((FloatLib.Numerics.MathFunctions.abs a).val - eps32 (|a.val|))
        ((FloatLib.Numerics.MathFunctions.abs a).val + eps32 (|a.val|)) := by
  have h := (abs_le).1 (abs_abs_error (a := a))
  constructor <;> linarith [h.2, h.1]

end FP32
end TorchLean.Floats

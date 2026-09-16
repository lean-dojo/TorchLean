/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Floats.Formats.BinaryInterchange.Configured
public import FloatLib.Floats.Formats.IEEE754.Native
public import FloatLib.Floats.Formats.BinaryInterchange.Interval
public import FloatLib.Floats.Formats.BinaryInterchange.IntervalSemantics
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

/-!
# Configured endpoints for FloatLib intervals

TorchLean certificates store the same configured values as the runtime. This adapter transports
FloatLib's format-generic `Model.Interval` to configured binary32 endpoints. All constructors and
operations delegate to the library, including its conservative whole-range fallback for unordered
or indeterminate endpoint results. The `toModel_*` lemmas expose the library operations to proofs.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


namespace TorchLean.Floats.IEEE754.IEEE32Exec

open FloatLib.Floats.Formats.BinaryInterchange

/-- Configured binary32 endpoints for FloatLib's executable interval semantics. -/
structure Interval32 where
  lo : ExecFloat.Binary 8 23
  hi : ExecFloat.Binary 8 23
  deriving Repr

namespace Interval32

@[inline] def toModel (x : Interval32) : Model.Interval FloatFormat.binary32 :=
  ⟨ExecFloat.Binary.toModel x.lo, ExecFloat.Binary.toModel x.hi⟩

@[inline] def ofModel (x : Model.Interval FloatFormat.binary32) : Interval32 :=
  ⟨ExecFloat.Binary.ofModel x.lo, ExecFloat.Binary.ofModel x.hi⟩

@[simp] theorem toModel_ofModel (x : Model.Interval FloatFormat.binary32) :
    toModel (ofModel x) = x := by
  cases x with
  | mk lo hi =>
    exact congrArg₂ Model.Interval.mk
      (ExecFloat.Binary.toModel_ofModel lo) (ExecFloat.Binary.toModel_ofModel hi)

@[simp] theorem ofModel_toModel (x : Interval32) : ofModel (toModel x) = x := by
  cases x with
  | mk lo hi =>
    exact congrArg₂ Interval32.mk
      (ExecFloat.Binary.ofModel_toModel lo) (ExecFloat.Binary.ofModel_toModel hi)

def mem (x : Interval32) (value : ExecFloat.Binary 8 23) : Prop :=
  Model.Interval.mem (toModel x) (ExecFloat.Binary.toModel value)

instance : Membership (ExecFloat.Binary 8 23) Interval32 where
  mem := Interval32.mem

abbrev Valid (x : Interval32) : Prop := Model.Interval.Valid (toModel x)
abbrev ValidExtended (x : Interval32) : Prop := Model.Interval.ValidExtended (toModel x)

@[inline] def point (x : ExecFloat.Binary 8 23) : Interval32 :=
  ofModel (Model.Interval.point (ExecFloat.Binary.toModel x))

@[simp] theorem toModel_point (x : ExecFloat.Binary 8 23) :
    toModel (point x) = Model.Interval.point (ExecFloat.Binary.toModel x) :=
  toModel_ofModel _

@[inline] def whole : Interval32 := ofModel (Model.Interval.whole FloatFormat.binary32)

@[simp] theorem toModel_whole :
    toModel whole = Model.Interval.whole FloatFormat.binary32 := by
  simp [whole]

@[inline] def containsZero (x : Interval32) : Bool := Model.Interval.containsZero (toModel x)
@[inline] def leB (x y : ExecFloat.Binary 8 23) : Bool :=
  Model.Interval.leB (ExecFloat.Binary.toModel x) (ExecFloat.Binary.toModel y)

@[inline] def hull (x y : Interval32) : Interval32 :=
  ofModel (Model.Interval.hull (toModel x) (toModel y))

@[simp] theorem toModel_hull (x y : Interval32) :
    toModel (hull x y) = Model.Interval.hull (toModel x) (toModel y) := by
  simp [hull]

@[inline] def add (x y : Interval32) : Interval32 :=
  ofModel (Model.Interval.add (toModel x) (toModel y))

@[simp] theorem toModel_add (x y : Interval32) :
    toModel (add x y) = Model.Interval.add (toModel x) (toModel y) := by
  simp [add]

@[inline] def sub (x y : Interval32) : Interval32 :=
  ofModel (Model.Interval.sub (toModel x) (toModel y))

@[simp] theorem toModel_sub (x y : Interval32) :
    toModel (sub x y) = Model.Interval.sub (toModel x) (toModel y) := by
  simp [sub]

@[inline] def mul (x y : Interval32) : Interval32 :=
  ofModel (Model.Interval.mul (toModel x) (toModel y))

@[simp] theorem toModel_mul (x y : Interval32) :
    toModel (mul x y) = Model.Interval.mul (toModel x) (toModel y) := by
  simp [mul]

@[inline] def div (x y : Interval32) : Interval32 :=
  ofModel (Model.Interval.div (toModel x) (toModel y))

@[simp] theorem toModel_div (x y : Interval32) :
    toModel (div x y) = Model.Interval.div (toModel x) (toModel y) := by
  simp [div]

@[inline] def neg (x : Interval32) : Interval32 :=
  ofModel (Model.Interval.neg (toModel x))

@[simp] theorem toModel_neg (x : Interval32) :
    toModel (neg x) = Model.Interval.neg (toModel x) := by
  simp [neg]

@[inline] def inv (x : Interval32) : Interval32 :=
  ofModel (Model.Interval.inv (toModel x))

@[simp] theorem toModel_inv (x : Interval32) :
    toModel (inv x) = Model.Interval.inv (toModel x) := by
  simp [inv]

@[inline] def relu (x : Interval32) : Interval32 :=
  ofModel (Model.Interval.relu (toModel x))

@[simp] theorem toModel_relu (x : Interval32) :
    toModel (relu x) = Model.Interval.relu (toModel x) := by
  simp [relu]

@[inline] def abs (x : Interval32) : Interval32 :=
  ofModel (Model.Interval.abs (toModel x))

@[simp] theorem toModel_abs (x : Interval32) :
    toModel (abs x) = Model.Interval.abs (toModel x) := by
  simp [abs]

@[inline] def sqrt (x : Interval32) : Interval32 :=
  ofModel (Model.Interval.sqrt (toModel x))

@[simp] theorem toModel_sqrt (x : Interval32) :
    toModel (sqrt x) = Model.Interval.sqrt (toModel x) := by
  simp [sqrt]

end Interval32
end TorchLean.Floats.IEEE754.IEEE32Exec

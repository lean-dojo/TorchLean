/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/

module

public import NN.IR.BinaryProofs
public import FloatLib.Floats.Formats.BinaryInterchange.Analysis.Error
import Mathlib.Tactic.Linarith

/-!
# Bounds for concrete typed graph executions

The checker below runs the existing binary interpreter, including the captured casts and
reduction order. Its soundness theorem concerns those executions, not supplied output vectors.
Every compared coordinate must decode to a finite rational; NaNs and infinities are rejected.

These are input-specific certificates. Acceptance does not prove a bound for other arguments,
frontend faithfulness, or external CPU/GPU kernels. In particular, this checker does not replace
an operation-level roundoff analysis over an input region.
-/

@[expose] public section

namespace NN.IR

open TorchLean FloatLib.Floats
open FloatLib.Floats.Formats.BinaryInterchange

namespace Binary

/-- The actual interpreter cast has FloatLib's half-ULP error on finite IEEE destinations.
The premise on the converted value excludes overflow; subnormal results are allowed. -/
theorem cast_abs_error_le {source target : FloatFormat}
    (value : Scalar source) (converted : Scalar target) (flags : ExecFloat.ConversionStatus)
    (hcast : cast (.binary source) (.binary target) .nearest value = .ok (converted, flags))
    (htarget : target.isIEEE = true)
    (hvalue : ExecFloat.Binary.isFinite value = true)
    (hconverted : ExecFloat.Binary.isFinite converted = true) :
    |ExecFloat.Binary.toReal converted - ExecFloat.Binary.toReal value| ≤
      Model.epsilonAt target (ExecFloat.Binary.toReal value) := by
  have hmodel := cast_value source target .nearest value converted flags hcast
  change |(ExecFloat.Binary.toModel converted).toReal -
      (ExecFloat.Binary.toModel value).toReal| ≤ _
  rw [hmodel]
  apply Model.abs_toReal_cast_sub_le_of_isIEEE_destination _ htarget hvalue
  rw [← hmodel]
  exact hconverted

end Binary

namespace Interpreter.Value

/-- Decode a row-major coordinate exactly. Unsupported types, missing coordinates and nonfinite
values have no rational observation. -/
def rationalAt? (value : Value Interpreter.binary) (index : Nat) : Option Rat :=
  match value with
  | ⟨.binary format, tensor⟩ =>
      letI := Interpreter.binary.storage (.binary format)
      tensor.tensor.data[index]? >>= ExecFloat.Binary.toRat?
  | _ => none

/-- Absolute error at every coordinate, with the tensor shape preserved. -/
def ErrorLe (bound : Rat) (left right : Value Interpreter.binary) : Prop :=
  left.shape = right.shape ∧ ∀ i : Fin (Spec.Shape.size left.shape),
    match left.rationalAt? i.val, right.rationalAt? i.val with
    | some a, some b => |a - b| ≤ bound
    | _, _ => False

instance (bound : Rat) (left right : Value Interpreter.binary) :
    Decidable (ErrorLe bound left right) := by
  have coordinate (i : Fin (Spec.Shape.size left.shape)) : Decidable
      (match left.rationalAt? i.val, right.rationalAt? i.val with
       | some a, some b => |a - b| ≤ bound
       | _, _ => False) := by
    cases left.rationalAt? i.val <;> cases right.rationalAt? i.val <;> infer_instance
  letI := coordinate
  unfold ErrorLe
  infer_instance

/-- Recover the finite coordinate values bounded by an accepted tensor relation. -/
theorem ErrorLe.coordinate {bound : Rat} {left right : Value Interpreter.binary}
    (h : ErrorLe bound left right) (i : Fin (Spec.Shape.size left.shape)) :
    ∃ a b : Rat, left.rationalAt? i.val = some a ∧ right.rationalAt? i.val = some b ∧
      |a - b| ≤ bound := by
  have hi := h.2 i
  cases ha : left.rationalAt? i.val with
  | none => simp [ha] at hi
  | some a =>
      cases hb : right.rationalAt? i.val with
      | none => simp [ha, hb] at hi
      | some b => exact ⟨a, b, rfl, rfl, by simpa [ha, hb] using hi⟩

/-- A positive reference margin and a coordinate error below half that margin preserve a strict
comparison. The values come from the executions, not from decimal observations. -/
theorem ErrorLe.preserves_comparison {bound margin : Rat}
    {left right : Value Interpreter.binary} (h : ErrorLe bound left right)
    (winner competitor : Fin (Spec.Shape.size left.shape))
    {a b : Rat} (hw : left.rationalAt? winner.val = some a)
    (hk : left.rationalAt? competitor.val = some b)
    (hmargin : margin ≤ a - b) (hbound : 2 * bound < margin) :
    ∃ x y : Rat, right.rationalAt? winner.val = some x ∧
      right.rationalAt? competitor.val = some y ∧ y < x := by
  obtain ⟨a', x, ha, hx, hax⟩ := h.coordinate winner
  obtain ⟨b', y, hb, hy, hby⟩ := h.coordinate competitor
  have ea : a' = a := Option.some.inj (ha.symm.trans hw)
  have eb : b' = b := Option.some.inj (hb.symm.trans hk)
  subst a' b'
  exact ⟨x, y, hx, hy, by
    obtain ⟨_, haw⟩ := abs_le.mp hax
    obtain ⟨hbk, _⟩ := abs_le.mp hby
    linarith⟩

/-- Every competing finite score is below the chosen coordinate by more than `gap`.
An invalid winner, a missing coordinate or a nonfinite score does not satisfy the predicate. -/
def MarginAbove (value : Value Interpreter.binary) (winner : Nat) (gap : Rat) : Prop :=
  winner < Spec.Shape.size value.shape ∧
    match value.rationalAt? winner with
    | none => False
    | some score => ∀ i : Fin (Spec.Shape.size value.shape), i.val ≠ winner →
        match value.rationalAt? i.val with
        | none => False
        | some other => gap < score - other

instance (value : Value Interpreter.binary) (winner : Nat) (gap : Rat) :
    Decidable (MarginAbove value winner gap) := by
  have coordinate (score : Rat) (i : Fin (Spec.Shape.size value.shape)) : Decidable
      (i.val ≠ winner → match value.rationalAt? i.val with
        | none => False | some other => gap < score - other) := by
    cases value.rationalAt? i.val <;> infer_instance
  letI := coordinate
  unfold MarginAbove
  cases value.rationalAt? winner <;> infer_instance

/-- A reference margin larger than twice the proved error makes the candidate's winner strict. -/
theorem ErrorLe.preserves_winner {bound : Rat} {left right : Value Interpreter.binary}
    (h : ErrorLe bound left right) {winner : Nat}
    (hm : MarginAbove left winner (2 * bound)) : MarginAbove right winner 0 := by
  have hw : winner < Spec.Shape.size left.shape := hm.1
  refine ⟨by simpa only [← h.1] using hw, ?_⟩
  obtain ⟨score, candidate, hs, hc, _⟩ := h.coordinate ⟨winner, hw⟩
  simp only [MarginAbove, hs] at hm
  simp only [hc]
  intro i hi
  let j : Fin (Spec.Shape.size left.shape) := ⟨i.val, by simpa only [h.1] using i.isLt⟩
  obtain ⟨other, alternative, ho, ha, _⟩ := h.coordinate j
  have hgap : 2 * bound < score - other := by
    simpa only [ho] using hm.2 j hi
  obtain ⟨x, y, hx, hy, hxy⟩ := h.preserves_comparison
    ⟨winner, hw⟩ j hs ho (le_refl (score - other)) hgap
  have ex : x = candidate := Option.some.inj (hx.symm.trans hc)
  have ey : y = alternative := Option.some.inj (hy.symm.trans ha)
  change right.rationalAt? i.val = some alternative at ha
  simp only [ha]
  linarith

end Interpreter.Value

namespace Numerical

open Interpreter

/-- Error bounds retain output count, return order, shapes and all finite coordinates. -/
def OutputsErrorLe (bound : Rat) (left right : Array (Value binary)) : Prop :=
  0 ≤ bound ∧ left.size = right.size ∧ ∀ i : Fin left.size,
    match right[i.val]? with
    | some value => Value.ErrorLe bound left[i] value
    | none => False

instance (bound : Rat) (left right : Array (Value binary)) :
    Decidable (OutputsErrorLe bound left right) := by
  have coordinate (i : Fin left.size) : Decidable
      (match right[i.val]? with
       | some value => Value.ErrorLe bound left[i] value
       | none => False) := by
    cases right[i.val]? <;> infer_instance
  letI := coordinate
  unfold OutputsErrorLe
  infer_instance

/-- Read the corresponding output relation without dropping return order or shape checks. -/
theorem OutputsErrorLe.at {bound : Rat} {left right : Array (Value binary)}
    (h : OutputsErrorLe bound left right) (i : Fin left.size) :
    ∃ value, right[i.val]? = some value ∧ Value.ErrorLe bound left[i] value := by
  have hi := h.2.2 i
  cases hr : right[i.val]? with
  | none => simp [hr] at hi
  | some value => exact ⟨value, rfl, by simpa [hr] using hi⟩

end Numerical

end NN.IR

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.IR.Loaded
public import FloatLib.Floats.Formats.BinaryInterchange.Analysis.Error
import Mathlib.Tactic.Linarith

/-!
# Concrete numerical certificates for captured programs

`checkError` re-executes both artifacts on the reference runtime of the operator IR, including the
captured casts and reduction order, before checking a rational absolute error budget. It reads
the programs, constants and arguments and ignores any reference outputs the artifacts store, so
the soundness theorem concerns the captured graph executions themselves. Every compared coordinate
must decode to a finite rational, which rejects NaNs and infinities.

A certificate bounds the two executions at the supplied arguments. Other inputs, the faithfulness
of the frontend capture and external CPU/GPU kernels lie outside it, and a bound over an input
region needs an operation-level roundoff analysis. A closed acceptance proof can be checked by
kernel reduction, which is practical only for small artifacts.
-/

@[expose] public section

open TorchLean.Shape (size)

namespace TorchLean.IR

open FloatLib.Floats
open FloatLib.Floats.Formats.BinaryInterchange

namespace Binary

/-- A round-to-nearest binary cast to an IEEE format has at most FloatLib's half-ULP error when its
result is finite. The finiteness premise excludes overflow but admits subnormal results. -/
theorem cast_abs_error_le {source target : FloatFormat}
    (value : Scalar source) (htarget : target.isIEEE = true)
    (hconverted : ExecFloat.Binary.isFinite (cast source target .nearest value).1 = true) :
    |ExecFloat.Binary.toReal (cast source target .nearest value).1 -
        ExecFloat.Binary.toReal value| ≤
      Model.halfUlpAt target (ExecFloat.Binary.toReal value) := by
  have hmodel := cast_value source target .nearest value
  change |(ExecFloat.Binary.toModel (cast source target .nearest value).1).toReal -
      (ExecFloat.Binary.toModel value).toReal| ≤ _
  rw [hmodel]
  apply Model.abs_toReal_cast_sub_le _ htarget
  rw [← hmodel]
  exact hconverted

end Binary

namespace Program.Value

/-- Decode a row-major coordinate of a binary tensor exactly. Other ports, missing coordinates and
nonfinite values have no rational observation. -/
def rationalAt? (value : Value Machine.reference) (index : Nat) : Option Rat :=
  match value with
  | ⟨⟨⟨.value, s⟩, some (.binary format)⟩, x⟩ | ⟨⟨⟨.frozen, s⟩, some (.binary format)⟩, x⟩
  | ⟨⟨⟨.detached, s⟩, some (.binary format)⟩, x⟩ =>
      (@Tensor.Rep.data _ (Reference.storage (.binary format)) s x)[index]? >>=
        ExecFloat.Binary.toRat?
  | _ => none

/-- Absolute error at every coordinate, with the tensor shape preserved. -/
def ErrorLe (bound : Rat) (left right : Value Machine.reference) : Prop :=
  left.typed.port.shape = right.typed.port.shape ∧
    ∀ i : Fin (Shape.size left.typed.port.shape),
      match left.rationalAt? i.val, right.rationalAt? i.val with
      | some a, some b => |a - b| ≤ bound
      | _, _ => False

/-- `ErrorLe` is decided by comparing the shapes and checking every coordinate. -/
instance (bound : Rat) (left right : Value Machine.reference) :
    Decidable (ErrorLe bound left right) := by
  have coordinate (i : Fin (Shape.size left.typed.port.shape)) : Decidable
      (match left.rationalAt? i.val, right.rationalAt? i.val with
       | some a, some b => |a - b| ≤ bound
       | _, _ => False) := by
    cases left.rationalAt? i.val <;> cases right.rationalAt? i.val <;> infer_instance
  letI := coordinate
  unfold ErrorLe
  infer_instance

/-- Recover the finite coordinate values bounded by an accepted tensor relation. -/
theorem ErrorLe.coordinate {bound : Rat} {left right : Value Machine.reference}
    (h : ErrorLe bound left right) (i : Fin (Shape.size left.typed.port.shape)) :
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
comparison. The compared values are the exact rational decodings (`rationalAt?`) of the tensors. -/
theorem ErrorLe.preserves_comparison {bound margin : Rat}
    {left right : Value Machine.reference} (h : ErrorLe bound left right)
    (winner competitor : Fin (Shape.size left.typed.port.shape))
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
def MarginAbove (value : Value Machine.reference) (winner : Nat) (gap : Rat) : Prop :=
  winner < Shape.size value.typed.port.shape ∧
    match value.rationalAt? winner with
    | none => False
    | some score => ∀ i : Fin (Shape.size value.typed.port.shape), i.val ≠ winner →
        match value.rationalAt? i.val with
        | none => False
        | some other => gap < score - other

/-- `MarginAbove` is decided by checking the winner's index and every competing coordinate. -/
instance (value : Value Machine.reference) (winner : Nat) (gap : Rat) :
    Decidable (MarginAbove value winner gap) := by
  have coordinate (score : Rat) (i : Fin (Shape.size value.typed.port.shape)) : Decidable
      (i.val ≠ winner → match value.rationalAt? i.val with
        | none => False | some other => gap < score - other) := by
    cases value.rationalAt? i.val <;> infer_instance
  letI := coordinate
  unfold MarginAbove
  cases value.rationalAt? winner <;> infer_instance

/-- A reference margin larger than twice the proved error makes the candidate's winner strict. -/
theorem ErrorLe.preserves_winner {bound : Rat} {left right : Value Machine.reference}
    (h : ErrorLe bound left right) {winner : Nat}
    (hm : MarginAbove left winner (2 * bound)) : MarginAbove right winner 0 := by
  have hw : winner < Shape.size left.typed.port.shape := hm.1
  refine ⟨by simpa only [← h.1] using hw, ?_⟩
  obtain ⟨score, candidate, hs, hc, _⟩ := h.coordinate ⟨winner, hw⟩
  simp only [MarginAbove, hs] at hm
  simp only [hc]
  intro i hi
  let j : Fin (Shape.size left.typed.port.shape) :=
    ⟨i.val, by simpa only [h.1] using i.isLt⟩
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

end Program.Value

end TorchLean.IR

namespace TorchLean.Capture

open Lean IR.Program

/-- Both output arrays have the same length, the bound is nonnegative, and outputs at the same
return position satisfy `Value.ErrorLe`, so their shapes match and every coordinate is finite. -/
def OutputsErrorLe (bound : Rat) (left right : Array (Value Machine.reference)) : Prop :=
  0 ≤ bound ∧ left.size = right.size ∧ ∀ i : Fin left.size,
    match right[i.val]? with
    | some value => Value.ErrorLe bound left[i] value
    | none => False

/-- `OutputsErrorLe` is decided by checking the bound, the lengths and every return position. -/
instance (bound : Rat) (left right : Array (Value Machine.reference)) :
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
theorem OutputsErrorLe.at {bound : Rat} {left right : Array (Value Machine.reference)}
    (h : OutputsErrorLe bound left right) (i : Fin left.size) :
    ∃ value, right[i.val]? = some value ∧ Value.ErrorLe bound left[i] value := by
  have hi := h.2.2 i
  cases hr : right[i.val]? with
  | none => simp [hr] at hi
  | some value => exact ⟨value, rfl, by simpa [hr] using hi⟩

/-- Execute a capture document's own program and arguments on the reference runtime. No saved
output is accepted as its result. -/
def execute (document : Json) : Except String (Array (Value Machine.reference)) := do
  let (program, payload) ← IR.load document
  let arguments ← IR.loadArguments Machine.reference document
  let (outputs, _) ← program.eval Machine.reference payload arguments
  pure outputs.toArray

/-- Check a concrete execution-error budget, rejecting failed execution and nonfinite outputs. -/
def checkError (left right : Json) (bound : Rat) : Bool :=
  match execute left, execute right with
  | .ok xs, .ok ys => decide (OutputsErrorLe bound xs ys)
  | _, _ => false

/-- A bound on the finite outputs of the two artifacts at their supplied arguments. -/
def ErrorBound (left right : Json) (bound : Rat) : Prop :=
  ∃ xs ys, execute left = .ok xs ∧ execute right = .ok ys ∧ OutputsErrorLe bound xs ys

/-- An accepted budget bounds the actual finite outputs of the two imported programs. -/
theorem checkError_sound (left right : Json) (bound : Rat)
    (h : checkError left right bound = true) : ErrorBound left right bound := by
  unfold ErrorBound
  unfold checkError at h
  cases hl : execute left with
  | error message => simp [hl] at h
  | ok xs =>
      cases hr : execute right with
      | error message => simp [hl, hr] at h
      | ok ys =>
          exact ⟨xs, ys, rfl, rfl, of_decide_eq_true (by simpa only [hl, hr] using h)⟩

/-- Check that the error budget preserves a chosen winner in one returned tensor.
`winner` is a row-major coordinate of the whole tensor and competes with every other coordinate. -/
def checkPrediction (left right : Json) (bound : Rat) (output winner : Nat) : Bool :=
  match execute left, execute right with
  | .ok xs, .ok ys =>
      if h : output < xs.size then
        decide (OutputsErrorLe bound xs ys) &&
          decide (Value.MarginAbove xs[output] winner (2 * bound))
      else false
  | _, _ => false

/-- Both artifact executions are within budget, and their selected tensor has the same strict
winner. The reference margin is retained as part of the conclusion. -/
def PredictionStable (left right : Json) (bound : Rat) (output winner : Nat) : Prop :=
  ∃ xs ys reference candidate,
    execute left = .ok xs ∧ execute right = .ok ys ∧
    xs[output]? = some reference ∧ ys[output]? = some candidate ∧
    OutputsErrorLe bound xs ys ∧ Value.MarginAbove reference winner (2 * bound) ∧
      Value.MarginAbove candidate winner 0

/-- A successful stability check supplies actual executions and a strictly winning candidate
coordinate. The comparison follows from the error budget and reference margin. -/
theorem checkPrediction_sound (left right : Json) (bound : Rat) (output winner : Nat)
    (h : checkPrediction left right bound output winner = true) :
    PredictionStable left right bound output winner := by
  unfold PredictionStable
  unfold checkPrediction at h
  cases hl : execute left with
  | error message => simp [hl] at h
  | ok xs =>
      cases hr : execute right with
      | error message => simp [hl, hr] at h
      | ok ys =>
          simp only [hl, hr] at h
          split at h
          next hout =>
            simp only [Bool.and_eq_true, decide_eq_true_eq] at h
            obtain ⟨he, hm⟩ := h
            obtain ⟨candidate, hc, related⟩ := he.at ⟨output, hout⟩
            exact ⟨xs, ys, xs[output], candidate, rfl, rfl,
              by simp, hc, he, hm, related.preserves_winner hm⟩
          next => simp at h

end TorchLean.Capture

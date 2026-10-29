/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/

module

public import NN.IR.Numerical
public import NN.Runtime.IR

/-!
# Concrete numerical certificates for captured programs

`checkError` re-executes both artifacts using the binary interpreter before checking a rational
absolute error budget. It reads the programs, constants and arguments; reference output fields
are not used. The soundness theorem therefore concerns the captured graph executions themselves.

A closed acceptance proof can be checked by reduction, or by `native_decide` with its additional
compiler trust. Neither route proves a guarantee for unexamined inputs or a frontend translation.
-/

@[expose] public section

namespace NN.Verification.Numerical

open Lean NN.IR NN.IR.Interpreter

/-- Execute the artifact's own graph and arguments; no saved output is accepted as its result. -/
def execute (artifact : Json) : Except String (Array (Value binary)) := do
  let loadBinary (dtype : DType) (shape : Spec.Shape) (words : Array Nat) :
      Except String (Value binary) := do
    let .binary format := dtype
      | throw s!"numerical certificate: {dtype} is not a binary format"
    Value.binary format shape words
  let program ← Import.IR.parseProgram artifact
  let payload ← Import.IR.parsePayload binary loadBinary artifact
  let arguments ← Import.IR.parseArguments binary loadBinary artifact
  let (outputs, _) ← binary.eval program payload arguments
  pure outputs

/-- Check a concrete execution-error budget, rejecting failed execution and nonfinite outputs. -/
def checkError (left right : Json) (bound : Rat) : Bool :=
  match execute left, execute right with
  | .ok xs, .ok ys => decide (IR.Numerical.OutputsErrorLe bound xs ys)
  | _, _ => false

/-- A bound on the finite outputs of the two artifacts at their supplied arguments. -/
def ErrorBound (left right : Json) (bound : Rat) : Prop :=
  ∃ xs ys, execute left = .ok xs ∧ execute right = .ok ys ∧
    IR.Numerical.OutputsErrorLe bound xs ys

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
`winner` is a row-major coordinate of that tensor, not a batch-axis convention. -/
def checkPrediction (left right : Json) (bound : Rat) (output winner : Nat) : Bool :=
  match execute left, execute right with
  | .ok xs, .ok ys =>
      if h : output < xs.size then
        decide (IR.Numerical.OutputsErrorLe bound xs ys) &&
          decide (Value.MarginAbove xs[output] winner (2 * bound))
      else false
  | _, _ => false

/-- Both artifact executions are within budget, and their selected tensor has the same strict
winner. The reference margin is retained as part of the conclusion. -/
def PredictionStable (left right : Json) (bound : Rat) (output winner : Nat) : Prop :=
  ∃ xs ys reference candidate,
    execute left = .ok xs ∧ execute right = .ok ys ∧
    xs[output]? = some reference ∧ ys[output]? = some candidate ∧
    IR.Numerical.OutputsErrorLe bound xs ys ∧ Value.MarginAbove reference winner (2 * bound) ∧
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

end NN.Verification.Numerical

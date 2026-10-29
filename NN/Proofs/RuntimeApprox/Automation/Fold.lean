/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Proofs.RuntimeApprox.Rounding
public import NN.Spec.Core.TensorReductionShape.Reductions

/-!
# Error bounds through tensor loops

The numerical analyzer derives a step's real reference, error expression, and side conditions.
These lemmas carry that proof through the existing tensor fold. Conditions are required only at
states visited by the fold, and the accumulation order is preserved.
-/

@[expose] public section

namespace Proofs.RuntimeApprox.Roundoff

open TorchLean

variable {α β γ : Type} [Storage β] {shape : Spec.Shape}

/-- Finite index loops have the same increasing order as a one-dimensional tensor traversal.
This equality needs no algebraic laws for the accumulator. -/
theorem fin_foldl_eq_coordinates {δ : Type} (n : Nat) (step : δ → Fin n → δ) (initial : δ) :
    Fin.foldl n step initial =
      (Tensor.Internal.Rep.coordinates [n]).foldl
        (fun state coordinate => step state coordinate.1) initial := by
  rw [Tensor.Internal.Rep.foldl_eq_fin_foldl]
  simp only [Tensor.Internal.Rep.coordinates, Tensor.Internal.Rep.ofFn,
    Tensor.Internal.Rep.getFlat_ofFlatFn]
  have size : Tensor.Internal.Shape.size [n] = n := by simp [Tensor.Internal.Shape.size]
  rw [Fin.foldl_congr size]
  congr 1
  funext state index
  congr 1
  apply Fin.ext
  simp [Tensor.Internal.Coord.unlinearize_eq_unlinearizeAux, Tensor.Internal.Coord.unlinearizeAux]
  rfl

/-- A condition holds at every step of this execution, before advancing its accumulator. -/
def AllSteps (step : α → β → α) (condition : α → β → Prop)
    (initial : α) (input : Tensor β shape) : Prop :=
  (input.foldl (fun state value =>
    (step state.1 value, state.2 ∧ condition state.1 value)) (initial, True)).2

/-- Check the visited states in the same order as the original fold. The condition's decision
procedure is supplied once; no condition is imposed on unreachable accumulator values. -/
def AllSteps.check (step : α → β → α) (condition : α → β → Prop)
    [∀ state value, Decidable (condition state value)]
    (initial : α) (input : Tensor β shape) : Bool :=
  (input.foldl (fun state value =>
    (step state.1 value, state.2 && decide (condition state.1 value))) (initial, true)).2

/-- A successful Boolean check supplies precisely the fold's range evidence. -/
theorem AllSteps.check_eq_true_iff (step : α → β → α) (condition : α → β → Prop)
    [∀ state value, Decidable (condition state value)]
    (initial : α) (input : Tensor β shape) :
    AllSteps.check step condition initial input = true ↔ AllSteps step condition initial input := by
  have related := Tensor.Internal.Rep.foldl_rel input input
    (fun state value => (step state.1 value, state.2 && decide (condition state.1 value)))
    (fun state value => (step state.1 value, state.2 ∧ condition state.1 value))
    (fun (checked : α × Bool) (specified : α × Prop) =>
      checked.1 = specified.1 ∧ (checked.2 = true ↔ specified.2))
    (initial, true) (initial, True) ⟨rfl, by simp⟩ (by
      rintro index ⟨state, ok⟩ ⟨state', valid⟩ ⟨rfl, previous⟩
      refine ⟨rfl, ?_⟩
      simpa only [Bool.and_eq_true, decide_eq_true_eq] using and_congr previous Iff.rfl)
  exact related.2

instance AllSteps.instDecidable (step : α → β → α) (condition : α → β → Prop)
    [∀ state value, Decidable (condition state value)]
    (initial : α) (input : Tensor β shape) : Decidable (AllSteps step condition initial input) :=
  decidable_of_iff _ (AllSteps.check_eq_true_iff step condition initial input)

/-- An invariant need only be preserved at steps whose conditions hold. -/
theorem AllSteps.invariant {step : α → β → α} {condition : α → β → Prop}
    {initial : α} {input : Tensor β shape} {invariant : α → Prop}
    (hvalid : AllSteps step condition initial input) (hinit : invariant initial)
    (hstep : ∀ state value, condition state value → invariant state →
      invariant (step state value)) : invariant (input.foldl step initial) := by
  have related := Tensor.Internal.Rep.foldl_rel input input step
    (fun state value => (step state.1 value, state.2 ∧ condition state.1 value))
    (fun state checked => state = checked.1 ∧ (checked.2 → invariant state))
    initial (initial, True) ⟨rfl, fun _ => hinit⟩ (by
      intro index state checked h
      rcases h with ⟨rfl, hbefore⟩
      exact ⟨rfl, fun h => hstep _ _ h.2 (hbefore h.1)⟩)
  exact related.2 hvalid

/-- Conditions independent of auxiliary state can be checked on the original accumulator. -/
theorem AllSteps.of_map (step : α → β → α) (condition : α → β → Prop)
    (initial : α) (input : Tensor β shape) (interpret : α → γ)
    (step' : γ → β → γ) (condition' : γ → β → Prop)
    (hstep : ∀ state value, interpret (step state value) = step' (interpret state) value)
    (hcondition : ∀ state value, condition state value = condition' (interpret state) value)
    (hvalid : AllSteps step' condition' (interpret initial) input) :
    AllSteps step condition initial input := by
  have mapped := Tensor.Internal.Rep.map_foldl input
    (fun state value => (step state.1 value, state.2 ∧ condition state.1 value))
    (initial, True) id (fun state : α × Prop => (interpret state.1, state.2))
    (fun state value => (step' state.1 value, state.2 ∧ condition' state.1 value)) (by
      intro state value
      simp only [hstep, hcondition, id_eq])
  have conditions := congrArg Prod.snd mapped
  simp only [Tensor.Internal.Rep.map_id] at conditions
  exact conditions.mpr hvalid

/-- A uniform condition is sufficient, although execution-specific evidence is more general. -/
theorem AllSteps.of_forall (step : α → β → α) (condition : α → β → Prop)
    (initial : α) (input : Tensor β shape) (hcondition : ∀ state value, condition state value) :
    AllSteps step condition initial input := by
  unfold AllSteps
  rw [Tensor.Internal.Rep.foldl_eq_fin_foldl, Fin.foldl_eq_foldl_finRange]
  apply List.foldlRecOn (motive := fun state : α × Prop => state.2)
  · trivial
  · intro state hstate index _
    exact ⟨hstate, hcondition _ _⟩

/-- Fuse a source map into its range conditions, preserving the visited states. -/
theorem AllSteps.map [Storage γ] (step : α → γ → α) (condition : α → γ → Prop)
    (initial : α) (input : Tensor β shape) (f : β → γ) :
    AllSteps step condition initial (Tensor.Internal.Rep.map f input) =
      AllSteps (fun state value => step state (f value))
        (fun state value => condition state (f value)) initial input := by
  unfold AllSteps
  rw [Tensor.Internal.Rep.foldl_map]

/-- Proof bookkeeping for one iteration; the first component is the original scalar step. -/
def foldStep (step : α → β → α) (reference : ℝ → β → ℝ)
    (error : α → ℝ → ℝ → β → ℝ) (state : α × ℝ × ℝ) (value : β) : α × ℝ × ℝ :=
  (step state.1 value, reference state.2.1 value, error state.1 state.2.1 state.2.2 value)

/-- Error accumulated in the source program's traversal order. This is proof bookkeeping;
it does not replace the executed tensor program. -/
def foldError (step : α → β → α) (reference : ℝ → β → ℝ)
    (error : α → ℝ → ℝ → β → ℝ) (initial : α) (initialReference initialError : ℝ)
    (input : Tensor β shape) : ℝ :=
  (input.foldl (foldStep step reference error) (initial, initialReference, initialError)).2.2

/-- Fuse a source map into its error recurrence without changing the reduction order. -/
theorem foldError_map [Storage γ] (step : α → γ → α) (reference : ℝ → γ → ℝ)
    (error : α → ℝ → ℝ → γ → ℝ) (initial : α) (initialReference initialError : ℝ)
    (input : Tensor β shape) (f : β → γ) :
    foldError step reference error initial initialReference initialError
        (Tensor.Internal.Rep.map f input) =
      foldError (fun state value => step state (f value))
        (fun state value => reference state (f value))
        (fun state realValue budget value => error state realValue budget (f value))
        initial initialReference initialError input := by
  unfold foldError
  rw [Tensor.Internal.Rep.foldl_map]
  rfl

/-- An exact scalar step preserves the incoming error budget, independently of tensor size. -/
@[simp] theorem foldError_identity (step : α → β → α) (reference : ℝ → β → ℝ)
    (initial : α) (initialReference initialError : ℝ) (input : Tensor β shape) :
    foldError step reference (fun _ _ error _ => error)
      initial initialReference initialError input = initialError := by
  unfold foldError
  rw [Tensor.Internal.Rep.foldl_eq_fin_foldl, Fin.foldl_eq_foldl_finRange]
  exact List.foldlRecOn (motive := fun state : α × ℝ × ℝ => state.2.2 = initialError)
    _ _ rfl (fun _ h _ _ => h)

/-- Lift a generated scalar error proof through a tensor fold, including its initial error. -/
theorem fold_error (observe : α → ℝ) (step : α → β → α) (reference : ℝ → β → ℝ)
    (error : α → ℝ → ℝ → β → ℝ) (condition : α → ℝ → ℝ → β → Prop)
    (initial : α) (initialReference initialError : ℝ) (input : Tensor β shape)
    (hinit : |observe initial - initialReference| ≤ initialError)
    (hstep : ∀ state realValue budget value, condition state realValue budget value →
      |observe state - realValue| ≤ budget →
      |observe (step state value) - reference realValue value| ≤
        error state realValue budget value)
    (hvalid : AllSteps (foldStep step reference error)
      (fun state value => condition state.1 state.2.1 state.2.2 value)
      (initial, initialReference, initialError) input) :
    |observe (input.foldl step initial) - input.foldl reference initialReference| ≤
      foldError step reference error initial initialReference initialError input := by
  have bound := hvalid.invariant (invariant := fun state =>
    |observe state.1 - state.2.1| ≤ state.2.2) hinit
    (fun state value hcondition herror => hstep _ _ _ _ hcondition herror)
  have actual := Tensor.Internal.Rep.map_foldl input (foldStep step reference error)
    (initial, initialReference, initialError) id Prod.fst step (fun _ _ => rfl)
  have referenceValue := Tensor.Internal.Rep.map_foldl input (foldStep step reference error)
    (initial, initialReference, initialError) id (fun state : α × ℝ × ℝ => state.2.1)
    reference (fun _ _ => rfl)
  simp only [Tensor.Internal.Rep.map_id] at actual referenceValue
  rw [actual, referenceValue] at bound
  exact bound

end Proofs.RuntimeApprox.Roundoff

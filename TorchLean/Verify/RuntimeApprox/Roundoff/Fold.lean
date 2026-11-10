/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import FloatLib.Numerics.ErrorBound.Loop
public import TorchLean.Verify.RuntimeApprox.Rounding
public import TorchLean.Tensor.List
public import TorchLean.Tensor.Reduce

/-!
# Error bounds through tensor loops

`roundoff` derives the real reference, the error budget and the conditions of one step of a loop.
These lemmas carry that step bound through a tensor fold in the fold's own order: the conditions
need to hold only at the states the fold visits (`AllSteps`), and the budgets accumulate along the
same traversal (`foldError`, `foldRelative`, `foldAbsolute`).
-/

@[expose] public section

namespace TorchLean.Roundoff

open FloatLib.Numerics

variable {α β γ : Type} [Storage β] {shape : Shape}

/-- A sum over the coordinates of a vector is the finite sum over its indices. -/
theorem _root_.TorchLean.Tensor.Rep.foldl_coordinates_add {M : Type} [AddCommMonoid M] {n : Nat}
    (f : Fin n → M) :
    (Tensor.Rep.coordinates [n]).foldl (fun sum coordinate => sum + f coordinate.1) 0 =
      ∑ index, f index := by
  rw [← Tensor.fin_foldl_eq_coordinates n (fun sum index => sum + f index),
    Fin.foldl_eq_foldl_finRange, Fin.sum_univ_def, List.sum_eq_foldl, List.foldl_map]

/-- A sum over the coordinates of a nonempty vector is positive when each of its terms is. -/
theorem _root_.TorchLean.Tensor.Rep.foldl_coordinates_add_pos {n : Nat}
    {f : Shape.Coord [n + 1] → ℝ} (h : ∀ coordinate, 0 < f coordinate) :
    0 < (Tensor.Rep.coordinates [n + 1]).foldl
      (fun sum coordinate => sum + f coordinate) 0 :=
  (Finset.sum_pos (fun _ _ => h _) Finset.univ_nonempty).trans_eq
    (Tensor.Rep.foldl_coordinates_add fun index => f (index, ())).symm

/-- A vector fold visits the coordinates in increasing order. -/
theorem foldl_eq_fin_foldl {n : Nat} (step : α → β → α) (initial : α) (input : Tensor β [n]) :
    Tensor.foldl step initial input =
      Fin.foldl n (fun state i => step state (input.getScalar i)) initial := by
  have coordinates : Tensor.map (Tensor.Rep.get input)
      (Tensor.Rep.coordinates [n]) = input :=
    Tensor.Rep.ext fun coordinate => by simp [Tensor.Rep.coordinates]
  rw [Tensor.fin_foldl_eq_coordinates]
  conv_lhs => rw [← coordinates, Tensor.Rep.foldl_map]
  simp only [Tensor.getScalar_eq_apply]

/-- The first component of a tensor fold whose step computes it from the accumulator's first
component alone is the fold of that component. -/
theorem _root_.TorchLean.Tensor.Rep.foldl_fst {δ : Type} (step : α × δ → β → α × δ)
    (first : α → β → α)
    (h : ∀ state value, (step state value).1 = first state.1 value) (initial : α × δ)
    (input : Tensor β shape) :
    (Tensor.foldl step initial input).1 =
      Tensor.foldl first initial.1 input :=
  Tensor.Rep.foldl_rel input input step first (fun state value => state.1 = value)
    initial initial.1 rfl fun _ _ _ previous => by rw [h, previous]

/-- A property of the accumulator that every step keeps holds of the result of a fold. -/
theorem _root_.TorchLean.Tensor.foldl_induction {step : α → β → α} {initial : α}
    {input : Tensor β shape} (motive : α → Prop) (hinit : motive initial)
    (hstep : ∀ state value, motive state → motive (step state value)) :
    motive (Tensor.foldl step initial input) :=
  Tensor.Rep.foldl_rel input input step step (fun state _ => motive state) initial initial
    hinit fun _ state _ h => hstep state _ h

/-- `condition` holds at the accumulator and the element of every step of the fold of `step` over
`input`, before that step. -/
def AllSteps (step : α → β → α) (condition : α → β → Prop)
    (initial : α) (input : Tensor β shape) : Prop :=
  (Tensor.foldl (fun state value =>
    (step state.1 value, state.2 ∧ condition state.1 value)) (initial, True) input).2

/-- Decide `AllSteps` by running the fold with a Boolean flag that records whether every step met
`condition`. -/
def AllSteps.check (step : α → β → α) (condition : α → β → Prop)
    [∀ state value, Decidable (condition state value)]
    (initial : α) (input : Tensor β shape) : Bool :=
  (Tensor.foldl (fun state value =>
    (step state.1 value, state.2 && decide (condition state.1 value))) (initial, true) input).2

/-- `AllSteps.check` returns `true` exactly when `AllSteps` holds. -/
theorem AllSteps.check_eq_true_iff (step : α → β → α) (condition : α → β → Prop)
    [∀ state value, Decidable (condition state value)]
    (initial : α) (input : Tensor β shape) :
    AllSteps.check step condition initial input = true ↔ AllSteps step condition initial input := by
  have related := Tensor.Rep.foldl_rel input input
    (fun state value => (step state.1 value, state.2 && decide (condition state.1 value)))
    (fun state value => (step state.1 value, state.2 ∧ condition state.1 value))
    (fun (checked : α × Bool) (specified : α × Prop) =>
      checked.1 = specified.1 ∧ (checked.2 = true ↔ specified.2))
    (initial, true) (initial, True) ⟨rfl, by simp⟩ (by
      rintro index ⟨state, ok⟩ ⟨state', valid⟩ ⟨rfl, previous⟩
      refine ⟨rfl, ?_⟩
      simpa only [Bool.and_eq_true, decide_eq_true_eq] using and_congr previous Iff.rfl)
  exact related.2

/-- `AllSteps` is decided by running the fold with `AllSteps.check`
(`AllSteps.check_eq_true_iff`). -/
instance AllSteps.instDecidable (step : α → β → α) (condition : α → β → Prop)
    [∀ state value, Decidable (condition state value)]
    (initial : α) (input : Tensor β shape) : Decidable (AllSteps step condition initial input) :=
  decidable_of_iff _ (AllSteps.check_eq_true_iff step condition initial input)

/-- An invariant of the initial accumulator that every step meeting `condition` preserves holds of
the result of a fold whose steps all meet `condition`. -/
theorem AllSteps.invariant {step : α → β → α} {condition : α → β → Prop}
    {initial : α} {input : Tensor β shape} {invariant : α → Prop}
    (hvalid : AllSteps step condition initial input) (hinit : invariant initial)
    (hstep : ∀ state value, condition state value → invariant state →
      invariant (step state value)) :
    invariant (Tensor.foldl step initial input) := by
  have related := Tensor.Rep.foldl_rel input input step
    (fun state value => (step state.1 value, state.2 ∧ condition state.1 value))
    (fun state checked => state = checked.1 ∧ (checked.2 → invariant state))
    initial (initial, True) ⟨rfl, fun _ => hinit⟩ (by
      intro index state checked h
      rcases h with ⟨rfl, hbefore⟩
      exact ⟨rfl, fun h => hstep _ _ h.2 (hbefore h.1)⟩)
  exact related.2 hvalid

/-- If `interpret` commutes with the steps and `condition` depends on the accumulator only through
`interpret`, then `AllSteps` follows from the same statement for the interpreted fold. -/
theorem AllSteps.of_map (step : α → β → α) (condition : α → β → Prop)
    (initial : α) (input : Tensor β shape) (interpret : α → γ)
    (step' : γ → β → γ) (condition' : γ → β → Prop)
    (hstep : ∀ state value, interpret (step state value) = step' (interpret state) value)
    (hcondition : ∀ state value, condition state value = condition' (interpret state) value)
    (hvalid : AllSteps step' condition' (interpret initial) input) :
    AllSteps step condition initial input := by
  have mapped := Tensor.Rep.map_foldl input
    (fun state value => (step state.1 value, state.2 ∧ condition state.1 value))
    (initial, True) id (fun state : α × Prop => (interpret state.1, state.2))
    (fun state value => (step' state.1 value, state.2 ∧ condition' state.1 value)) (by
      intro state value
      simp only [hstep, hcondition, id_eq])
  have conditions := congrArg Prod.snd mapped
  simp only [Tensor.Rep.map_id] at conditions
  exact conditions.mpr hvalid

/-- A condition that holds at every accumulator and element holds at every step. -/
theorem AllSteps.of_forall (step : α → β → α) (condition : α → β → Prop)
    (initial : α) (input : Tensor β shape) (hcondition : ∀ state value, condition state value) :
    AllSteps step condition initial input := by
  unfold AllSteps
  rw [Tensor.Rep.foldl_eq_fin_foldl, Fin.foldl_eq_foldl_finRange]
  apply List.foldlRecOn (motive := fun state : α × Prop => state.2)
  · trivial
  · intro state hstate index _
    exact ⟨hstate, hcondition _ _⟩

/-- `AllSteps` over a mapped tensor is `AllSteps` over the original tensor with the map composed
into `step` and `condition`. -/
theorem AllSteps.map [Storage γ] (step : α → γ → α) (condition : α → γ → Prop)
    (initial : α) (input : Tensor β shape) (f : β → γ) :
    AllSteps step condition initial (Tensor.map f input) =
      AllSteps (fun state value => step state (f value))
        (fun state value => condition state (f value)) initial input := by
  unfold AllSteps
  rw [Tensor.Rep.foldl_map]

/-- The error budget that `List.foldStep` accumulates over `input`, in the fold's traversal
order. -/
def foldError (step : α → β → α) (reference : ℝ → β → ℝ)
    (error : α → ℝ → ℝ → β → ℝ) (initial : α) (initialReference initialError : ℝ)
    (input : Tensor β shape) : ℝ :=
  (Tensor.foldl (List.foldStep step reference error)
    (initial, initialReference, initialError) input).2.2

/-- The error budget of a fold over a mapped tensor is that of the fold over the original tensor
with the map composed into its steps. -/
theorem foldError_map [Storage γ] (step : α → γ → α) (reference : ℝ → γ → ℝ)
    (error : α → ℝ → ℝ → γ → ℝ) (initial : α) (initialReference initialError : ℝ)
    (input : Tensor β shape) (f : β → γ) :
    foldError step reference error initial initialReference initialError
        (Tensor.map f input) =
      foldError (fun state value => step state (f value))
        (fun state value => reference state (f value))
        (fun state realValue budget value => error state realValue budget (f value))
        initial initialReference initialError input := by
  unfold foldError
  rw [Tensor.Rep.foldl_map]
  rfl

/-- If the error step returns its incoming budget, `foldError` is the initial budget, at every
shape. -/
@[simp] theorem foldError_identity (step : α → β → α) (reference : ℝ → β → ℝ)
    (initial : α) (initialReference initialError : ℝ) (input : Tensor β shape) :
    foldError step reference (fun _ _ error _ => error)
      initial initialReference initialError input = initialError := by
  unfold foldError
  rw [Tensor.Rep.foldl_eq_fin_foldl, Fin.foldl_eq_foldl_finRange]
  exact List.foldlRecOn (motive := fun state : α × ℝ × ℝ => state.2.2 = initialError)
    _ _ rfl (fun _ h _ _ => h)

/-- If the initial error is at most `initialError`, the step bound holds under its condition and
every step of the fold meets that condition (`AllSteps`), then the error of the whole fold is at
most `foldError`. -/
theorem fold_error (observe : α → ℝ) (step : α → β → α) (reference : ℝ → β → ℝ)
    (error : α → ℝ → ℝ → β → ℝ) (condition : α → ℝ → ℝ → β → Prop)
    (initial : α) (initialReference initialError : ℝ) (input : Tensor β shape)
    (hinit : |observe initial - initialReference| ≤ initialError)
    (hstep : ∀ state realValue budget value, condition state realValue budget value →
      |observe state - realValue| ≤ budget →
      |observe (step state value) - reference realValue value| ≤
        error state realValue budget value)
    (hvalid : AllSteps (List.foldStep step reference error)
      (fun state value => condition state.1 state.2.1 state.2.2 value)
      (initial, initialReference, initialError) input) :
    |observe (Tensor.foldl step initial input) -
        Tensor.foldl reference initialReference input| ≤
      foldError step reference error initial initialReference initialError input := by
  have bound := hvalid.invariant (invariant := fun state =>
    |observe state.1 - state.2.1| ≤ state.2.2) hinit
    (fun state value hcondition herror => hstep _ _ _ _ hcondition herror)
  have actual := Tensor.Rep.map_foldl input (List.foldStep step reference error)
    (initial, initialReference, initialError) id Prod.fst step (fun _ _ => rfl)
  have referenceValue := Tensor.Rep.map_foldl input (List.foldStep step reference error)
    (initial, initialReference, initialError) id (fun state : α × ℝ × ℝ => state.2.1)
    reference (fun _ _ => rfl)
  simp only [Tensor.Rep.map_id] at actual referenceValue
  rw [actual, referenceValue] at bound
  exact bound

/-- The relative part of the mixed error that `List.mixedFoldStep` accumulates over `input`,
in the fold's traversal order. -/
def foldRelative (step : α → β → α) (reference : ℝ → β → ℝ)
    (relative absolute : α → ℝ → ℝ → ℝ → β → ℝ) (initial : α)
    (initialReference initialRelative initialAbsolute : ℝ) (input : Tensor β shape) : ℝ :=
  (Tensor.foldl (List.mixedFoldStep step reference relative absolute)
    (initial, initialReference, initialRelative, initialAbsolute) input).2.2.1

/-- The absolute part of the mixed error that `List.mixedFoldStep` accumulates over `input`,
in the fold's traversal order. -/
def foldAbsolute (step : α → β → α) (reference : ℝ → β → ℝ)
    (relative absolute : α → ℝ → ℝ → ℝ → β → ℝ) (initial : α)
    (initialReference initialRelative initialAbsolute : ℝ) (input : Tensor β shape) : ℝ :=
  (Tensor.foldl (List.mixedFoldStep step reference relative absolute)
    (initial, initialReference, initialRelative, initialAbsolute) input).2.2.2

/-- A mixed-error step bound carries through a tensor fold with the budgets `foldRelative` and
`foldAbsolute`, together with a relation `invariant` between the accumulator and its reference,
such as finiteness or a sign, that holds initially and that each step preserves. -/
theorem fold_mixedError (observe : α → ℝ) (step : α → β → α) (reference : ℝ → β → ℝ)
    (relative absolute : α → ℝ → ℝ → ℝ → β → ℝ) (invariant : α → ℝ → Prop)
    (condition : α → ℝ → ℝ → ℝ → β → Prop) (initial : α)
    (initialReference initialRelative initialAbsolute : ℝ) (input : Tensor β shape)
    (hinit : MixedError (observe initial) initialReference initialRelative initialAbsolute ∧
      invariant initial initialReference)
    (hstep : ∀ state realValue r a value, condition state realValue r a value →
      MixedError (observe state) realValue r a ∧ invariant state realValue →
      MixedError (observe (step state value)) (reference realValue value)
          (relative state realValue r a value) (absolute state realValue r a value) ∧
        invariant (step state value) (reference realValue value))
    (hvalid : AllSteps (List.mixedFoldStep step reference relative absolute)
      (fun state value => condition state.1 state.2.1 state.2.2.1 state.2.2.2 value)
      (initial, initialReference, initialRelative, initialAbsolute) input) :
    MixedError (observe (Tensor.foldl step initial input))
        (Tensor.foldl reference initialReference input)
        (foldRelative step reference relative absolute initial initialReference initialRelative
          initialAbsolute input)
        (foldAbsolute step reference relative absolute initial initialReference initialRelative
          initialAbsolute input) ∧
      invariant (Tensor.foldl step initial input)
        (Tensor.foldl reference initialReference input) := by
  have bound := hvalid.invariant (invariant := fun state =>
    MixedError (observe state.1) state.2.1 state.2.2.1 state.2.2.2 ∧ invariant state.1 state.2.1)
    hinit (fun state value hcondition hstate => hstep _ _ _ _ _ hcondition hstate)
  have actual := Tensor.Rep.map_foldl input
    (List.mixedFoldStep step reference relative absolute)
    (initial, initialReference, initialRelative, initialAbsolute) id Prod.fst step (fun _ _ => rfl)
  have referenceValue := Tensor.Rep.map_foldl input
    (List.mixedFoldStep step reference relative absolute)
    (initial, initialReference, initialRelative, initialAbsolute) id
    (fun state : α × ℝ × ℝ × ℝ => state.2.1) reference (fun _ _ => rfl)
  simp only [Tensor.Rep.map_id] at actual referenceValue
  rw [actual, referenceValue] at bound
  exact bound

/-- In a left fold over distinct elements, exactly one of which satisfies `guard`, `before`
holds until that element and `after` from it on. -/
private theorem foldl_after_single {ι δ : Type} (step : δ → ι → δ) (guard : ι → Prop)
    (before after : δ → Prop) (hunique : ∀ i j, guard i → guard j → i = j)
    (hbefore : ∀ state i, ¬guard i → before state → before (step state i))
    (hguard : ∀ state i, guard i → before state → after (step state i))
    (hafter : ∀ state i, ¬guard i → after state → after (step state i)) :
    ∀ (elements : List ι), elements.Nodup → (∃ i ∈ elements, guard i) → ∀ state, before state →
      after (elements.foldl step state)
  | [], _, ⟨_, hmember, _⟩, _, _ => absurd hmember List.not_mem_nil
  | i :: rest, hdistinct, ⟨j, hmember, hj⟩, state, hstate => by
      rw [List.nodup_cons] at hdistinct
      by_cases hi : guard i
      · exact List.foldlRecOn (motive := after) (b := step state i) rest step
          (hguard state i hi hstate)
          fun b hb k hk => hafter b k (fun hk' => hdistinct.1 ((hunique i k hi hk') ▸ hk)) hb
      · have hrest : j ∈ rest := by
          rcases List.mem_cons.mp hmember with rfl | hrest
          · exact absurd hj hi
          · exact hrest
        exact foldl_after_single step guard before after hunique hbefore hguard hafter rest
          hdistinct.2 ⟨j, hrest, hj⟩ _ (hbefore state i hi hstate)

/-- A fold over the coordinates of a shape from an observed zero ends at `value` when the step at
`index` takes zero to `value` and the steps at every other coordinate keep both zero and `value`,
whatever the size of the shape. One-hot reductions are the typical case. The fold carries `valid`
along, for example finiteness. -/
theorem fold_eq_of_single (observe : α → ℝ) (step : α → shape.Coord → α) (valid : α → Prop)
    (index : shape.Coord) (value : ℝ) (initial : α)
    (hinit : valid initial ∧ observe initial = 0)
    (hzero : ∀ state coordinate, coordinate ≠ index → valid state ∧ observe state = 0 →
      valid (step state coordinate) ∧ observe (step state coordinate) = 0)
    (hindex : ∀ state, valid state ∧ observe state = 0 →
      valid (step state index) ∧ observe (step state index) = value)
    (hrest : ∀ state coordinate, coordinate ≠ index → valid state ∧ observe state = value →
      valid (step state coordinate) ∧ observe (step state coordinate) = value) :
    valid ((Tensor.Rep.coordinates shape).foldl step initial) ∧
      observe ((Tensor.Rep.coordinates shape).foldl step initial) = value := by
  have traversal : (Tensor.Rep.coordinates shape).foldl step initial =
      (List.finRange (Shape.size shape)).foldl
        (fun state flat => step state (Shape.Coord.unlinearize flat)) initial := by
    rw [Tensor.Rep.foldl_eq_fin_foldl, Fin.foldl_eq_foldl_finRange]
    simp only [Tensor.Rep.coordinates, Tensor.Rep.ofFn,
      Tensor.Rep.getFlat_ofFlatFn, id_eq]
  have hindexed (flat : Fin (Shape.size shape)) :
      Shape.Coord.unlinearize flat = index ↔
        flat = Shape.Coord.linearize index := by
    constructor
    · intro hflat
      rw [← hflat, Shape.Coord.linearize_unlinearize]
    · intro hflat
      rw [hflat, Shape.Coord.unlinearize_linearize]
  rw [traversal]
  exact foldl_after_single
    (fun state flat => step state (Shape.Coord.unlinearize flat))
    (fun flat => Shape.Coord.unlinearize flat = index)
    (fun state => valid state ∧ observe state = 0)
    (fun state => valid state ∧ observe state = value)
    (fun i j hi hj => ((hindexed i).mp hi).trans ((hindexed j).mp hj).symm)
    (fun state flat hflat hstate => hzero state _ hflat hstate)
    (fun state flat hflat hstate => by
      rw [hflat]
      exact hindex state hstate)
    (fun state flat hflat hstate => hrest state _ hflat hstate)
    _ (List.nodup_finRange _)
    ⟨_, List.mem_finRange _, (hindexed _).mpr rfl⟩ initial hinit

end TorchLean.Roundoff

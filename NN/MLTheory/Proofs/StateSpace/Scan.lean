/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.SelectiveScan

/-!
# Proofs for affine selective scan

The Mamba/S4 scan theorem is an algebra theorem about affine maps.  A sequential recurrent update
and a parallel prefix scan are equivalent because affine transition composition is associative:

$$
(a_2,b_2)\circ(a_1,b_1)=(a_2a_1,a_2b_1+b_2).
$$

The tensor/CUDA implementation is allowed to choose an efficient scan schedule, but the mathematical
contract is this file: prefix summaries denote the same state as the left-to-right recurrence.
-/

@[expose] public section

namespace NN
namespace MLTheory
namespace StateSpace

open Spec TorchLean
namespace ScalarAffineTransition

variable {α : Type}

/-- Composing scalar affine transitions agrees with function composition. -/
@[simp] theorem compose_apply [Semiring α] (t₂ t₁ : Spec.ScalarAffineTransition α)
    (h : α) :
    (Spec.ScalarAffineTransition.compose t₂ t₁).apply h = t₂.apply (t₁.apply h) := by
  cases t₁
  cases t₂
  simp [Spec.ScalarAffineTransition.compose, Spec.ScalarAffineTransition.apply,
    mul_add, mul_assoc, add_assoc]

/-- The identity transition is a left identity for composition. -/
@[simp] theorem compose_id_left [Semiring α] (t : Spec.ScalarAffineTransition α) :
    Spec.ScalarAffineTransition.compose Spec.ScalarAffineTransition.id t = t := by
  cases t
  simp [Spec.ScalarAffineTransition.compose, Spec.ScalarAffineTransition.id]

/-- The identity transition is a right identity for composition. -/
@[simp] theorem compose_id_right [Semiring α] (t : Spec.ScalarAffineTransition α) :
    Spec.ScalarAffineTransition.compose t Spec.ScalarAffineTransition.id = t := by
  cases t
  simp [Spec.ScalarAffineTransition.compose, Spec.ScalarAffineTransition.id]

/-- Scalar affine transition composition is associative. -/
@[simp] theorem compose_assoc [Semiring α]
    (t₃ t₂ t₁ : Spec.ScalarAffineTransition α) :
    Spec.ScalarAffineTransition.compose
        (Spec.ScalarAffineTransition.compose t₃ t₂) t₁ =
      Spec.ScalarAffineTransition.compose t₃
        (Spec.ScalarAffineTransition.compose t₂ t₁) := by
  cases t₁
  cases t₂
  cases t₃
  simp [Spec.ScalarAffineTransition.compose, mul_add, mul_assoc, add_assoc]

/-- Zero is a fixed point of a homogeneous scalar transition. -/
@[simp] theorem homogeneous_zero_fixed [Semiring α] (a : α) :
    (Spec.ScalarAffineTransition.apply { a := a, b := 0 } 0) = 0 := by
  simp [Spec.ScalarAffineTransition.apply]

end ScalarAffineTransition

namespace DiagonalTransition

variable {α : Type}

/-- Applying a diagonal transition is exactly the scalar affine update in each channel. -/
@[simp] theorem apply_getScalar [Add α] [Mul α] {stateDim : Nat}
    (tr : Spec.DiagonalTransition α stateDim)
    (h : TorchLean.Tensor α [stateDim])
    (i : Fin stateDim) :
    TorchLean.Tensor.getScalar (tr.apply h) i =
      TorchLean.Tensor.getScalar tr.a i * TorchLean.Tensor.getScalar h i +
        TorchLean.Tensor.getScalar tr.b i := by
  simp [Spec.DiagonalTransition.apply,
    TorchLean.Tensor.getScalar_eq_apply,
    TorchLean.Tensor.addSpec, TorchLean.Tensor.mulSpec]

/--
Composing diagonal transitions agrees channelwise with composing the corresponding scalar affine
maps.  This is the exact algebraic invariant used by the variable-coefficient selective-scan
kernel: each flattened state lane is an independent affine scan.
-/
@[simp] theorem compose_apply_getScalar [Semiring α] {stateDim : Nat}
    (t₂ t₁ : Spec.DiagonalTransition α stateDim)
    (h : TorchLean.Tensor α [stateDim])
    (i : Fin stateDim) :
    TorchLean.Tensor.getScalar ((Spec.DiagonalTransition.compose t₂ t₁).apply h) i =
      TorchLean.Tensor.getScalar (t₂.apply (t₁.apply h)) i := by
  simp only [apply_getScalar]
  simp [Spec.DiagonalTransition.compose,
    TorchLean.Tensor.getScalar_eq_apply,
    TorchLean.Tensor.addSpec, TorchLean.Tensor.mulSpec,
    TorchLean.Tensor.map2Spec, mul_add, mul_assoc, add_assoc]

end DiagonalTransition

/-- Seeding a scan with existing output preserves its state evolution and prepends that output. -/
theorem scanArrayFrom_eq {State Input Output : Type}
    (step : State → Input → State × Output) (initial : State)
    (initialOutputs : Array Output) (xs : Array Input) :
    Spec.scanArrayFrom step initial initialOutputs xs =
      let result := Spec.scanArray step initial xs
      (result.1, initialOutputs ++ result.2) := by
  unfold Spec.scanArray Spec.scanArrayFrom
  rw [← Array.foldl_toList, ← Array.foldl_toList]
  generalize xs.toList = items
  induction items generalizing initial initialOutputs with
  | nil => simp
  | cons input rest ih =>
      simp only [List.foldl_cons]
      rcases step initial input with ⟨nextState, output⟩
      rw [ih nextState (initialOutputs.push output)]
      simp only [show (#[] : Array Output).push output = #[output] from rfl]
      rw [ih nextState #[output]]
      simp

/-- The output buffer carried by a scan does not affect its final state. -/
theorem scanArrayFrom_state_eq_foldl {State Input Output : Type}
    (step : State → Input → State × Output) (initial : State)
    (initialOutputs : Array Output) (xs : Array Input) :
    (Spec.scanArrayFrom step initial initialOutputs xs).1 =
      xs.foldl (fun state input => (step state input).1) initial := by
  unfold Spec.scanArrayFrom
  rw [← Array.foldl_toList, ← Array.foldl_toList]
  generalize xs.toList = items
  induction items generalizing initial initialOutputs with
  | nil => rfl
  | cons input rest ih =>
      simp only [List.foldl_cons]
      rcases step initial input with ⟨nextState, output⟩
      exact ih nextState (initialOutputs.push output)

/-- The state component of `scanArray` is the ordinary state-only left fold. -/
theorem scanArray_state_eq_foldl {State Input Output : Type}
    (step : State → Input → State × Output) (initial : State) (xs : Array Input) :
    (Spec.scanArray step initial xs).1 =
      xs.foldl (fun state input => (step state input).1) initial := by
  exact scanArrayFrom_state_eq_foldl step initial #[] xs

/-- A scan over appended inputs is the prefix scan followed by the state-dependent suffix scan. -/
theorem scanArray_append {State Input Output : Type}
    (step : State → Input → State × Output) (initial : State) (xs ys : Array Input) :
    Spec.scanArray step initial (xs ++ ys) =
      let first := Spec.scanArray step initial xs
      let second := Spec.scanArray step first.1 ys
      (second.1, first.2 ++ second.2) := by
  unfold Spec.scanArray Spec.scanArrayFrom
  rw [Array.foldl_append]
  generalize hfirst :
    Array.foldl
      (fun stateAndOutputs input =>
        let (state, outputs) := stateAndOutputs
        let (nextState, output) := step state input
        (nextState, outputs.push output))
      (initial, #[]) xs = first
  rcases first with ⟨firstState, firstOutputs⟩
  exact scanArrayFrom_eq step firstState firstOutputs ys

private theorem Array.take_append_left {α : Type} (xs ys : Array α) :
    (xs ++ ys).take xs.size = xs := by
  apply Array.ext
  · simp
  · intro i h₁ h₂
    simp

/-- Appending future inputs cannot change outputs already emitted by a stateful scan. -/
theorem scanArray_append_outputs_take {State Input Output : Type}
    (step : State → Input → State × Output) (initial : State) (xs ys : Array Input) :
    (Spec.scanArray step initial (xs ++ ys)).2.take xs.size =
      (Spec.scanArray step initial xs).2 := by
  rw [show (Spec.scanArray step initial (xs ++ ys)).2 =
      (Spec.scanArray step initial xs).2 ++
        (Spec.scanArray step
          (Spec.scanArray step initial xs).1 ys).2 by
    simpa using congrArg Prod.snd (scanArray_append step initial xs ys)]
  rw [← Spec.scanArray_outputs_size step initial xs]
  exact Array.take_append_left _ _

/-- A stateful scan emits exactly one value for every input. -/
@[simp] theorem scanArray_outputs_size {State Input Output : Type}
    (step : State → Input → State × Output) (initial : State) (xs : Array Input) :
    (Spec.scanArray step initial xs).2.size = xs.size :=
  Spec.scanArray_outputs_size step initial xs

/-- Running appended scalar transitions factors through the state reached after the prefix. -/
theorem runScalarAffine_append {α : Type} [Mul α] [Add α] (h0 : α)
    (xs ys : Array (Spec.ScalarAffineTransition α)) :
    Spec.runScalarAffine h0 (xs ++ ys) =
      Spec.runScalarAffine (Spec.runScalarAffine h0 xs) ys := by
  simpa [Spec.runScalarAffine] using
    congrArg Prod.fst (scanArray_append
      (fun state transition =>
        let nextState := transition.apply state
        (nextState, nextState)) h0 xs ys)

/-- Running one scalar transition is the same as applying it. -/
@[simp] theorem runScalarAffine_singleton {α : Type} [Mul α] [Add α] (h0 : α)
    (tr : Spec.ScalarAffineTransition α) :
    Spec.runScalarAffine h0 #[tr] = tr.apply h0 := by
  rfl

/-- The affine summary denotes the same state as the sequential recurrence. -/
theorem summarizeScalarAffine_apply_eq_run {α : Type} [Semiring α] (h0 : α)
    (transitions : Array (Spec.ScalarAffineTransition α)) :
    (Spec.summarizeScalarAffine transitions).apply h0 =
      Spec.runScalarAffine h0 transitions := by
  unfold Spec.runScalarAffine
  rw [scanArray_state_eq_foldl]
  unfold Spec.summarizeScalarAffine
  rw [← Array.foldr_toList, ← Array.foldl_toList]
  generalize transitions.toList = items
  induction items generalizing h0 with
  | nil =>
      simp [Spec.ScalarAffineTransition.id,
        Spec.ScalarAffineTransition.apply]
  | cons transition rest ih =>
      simp only [List.foldr_cons, List.foldl_cons]
      rw [ScalarAffineTransition.compose_apply, ih]

private theorem foldr_compose_summary {α : Type} [Semiring α]
    (initial : Spec.ScalarAffineTransition α)
    (transitions : Array (Spec.ScalarAffineTransition α)) :
    transitions.foldr
        (fun transition summary =>
          Spec.ScalarAffineTransition.compose summary transition) initial =
      Spec.ScalarAffineTransition.compose initial
        (Spec.summarizeScalarAffine transitions) := by
  unfold Spec.summarizeScalarAffine
  rw [← Array.foldr_toList, ← Array.foldr_toList]
  generalize transitions.toList = items
  induction items with
  | nil => simp
  | cons transition rest ih =>
      simp only [List.foldr_cons]
      rw [ih]
      exact ScalarAffineTransition.compose_assoc initial
        (List.foldr
          (fun transition summary =>
            Spec.ScalarAffineTransition.compose summary transition)
          Spec.ScalarAffineTransition.id rest) transition

/-- Prefix summaries compose across array append in execution order. -/
theorem summarizeScalarAffine_append {α : Type} [Semiring α]
    (xs ys : Array (Spec.ScalarAffineTransition α)) :
    Spec.summarizeScalarAffine (xs ++ ys) =
      Spec.ScalarAffineTransition.compose
        (Spec.summarizeScalarAffine ys)
        (Spec.summarizeScalarAffine xs) := by
  unfold Spec.summarizeScalarAffine
  rw [Array.foldr_append]
  exact foldr_compose_summary
    (Array.foldr
      (fun transition summary =>
        Spec.ScalarAffineTransition.compose summary transition)
      Spec.ScalarAffineTransition.id ys) xs

/-- Prefix summaries composed across append have the expected denotation. -/
theorem summarizeScalarAffine_append_apply {α : Type} [Semiring α] (h0 : α)
    (xs ys : Array (Spec.ScalarAffineTransition α)) :
    (Spec.summarizeScalarAffine (xs ++ ys)).apply h0 =
      (Spec.ScalarAffineTransition.compose
        (Spec.summarizeScalarAffine ys)
        (Spec.summarizeScalarAffine xs)).apply h0 := by
  rw [summarizeScalarAffine_append]

/-- The scalar affine scan has one state per transition. -/
@[simp] theorem scalarAffineScan_size {α : Type} [Mul α] [Add α] (h0 : α)
    (transitions : Array (Spec.ScalarAffineTransition α)) :
    (Spec.scalarAffineScan h0 transitions).size = transitions.size := by
  exact scanArray_outputs_size _ h0 transitions

/-- Scanning appended scalar transitions is the prefix scan followed by the suffix scan. -/
theorem scalarAffineScan_append {α : Type} [Mul α] [Add α] (h0 : α)
    (xs ys : Array (Spec.ScalarAffineTransition α)) :
    Spec.scalarAffineScan h0 (xs ++ ys) =
      Spec.scalarAffineScan h0 xs ++
        Spec.scalarAffineScan (Spec.runScalarAffine h0 xs) ys := by
  simpa [Spec.scalarAffineScan, Spec.runScalarAffine] using
    congrArg Prod.snd (scanArray_append
      (fun state transition =>
        let nextState := transition.apply state
        (nextState, nextState)) h0 xs ys)

/-- The diagonal tensor scan has one state per transition. -/
@[simp] theorem diagonalSelectiveScan_size {α : Type} [Add α] [Mul α] {stateDim : Nat}
    (h0 : TorchLean.Tensor α [stateDim])
    (transitions : Array (Spec.DiagonalTransition α stateDim)) :
    (Spec.diagonalSelectiveScan h0 transitions).size = transitions.size := by
  exact scanArray_outputs_size _ h0 transitions

/-- Running appended diagonal transitions factors through the state after the prefix. -/
theorem runDiagonalTransitions_append {α : Type} [Add α] [Mul α] {stateDim : Nat}
    (h0 : TorchLean.Tensor α [stateDim])
    (xs ys : Array (Spec.DiagonalTransition α stateDim)) :
    Spec.runDiagonalTransitions h0 (xs ++ ys) =
      Spec.runDiagonalTransitions (Spec.runDiagonalTransitions h0 xs) ys := by
  simpa [Spec.runDiagonalTransitions] using
    congrArg Prod.fst (scanArray_append
      (fun state transition =>
        let nextState := transition.apply state
        (nextState, nextState)) h0 xs ys)

/-- The diagonal scan of an append is the prefix scan followed by the state-dependent suffix
scan. -/
theorem diagonalSelectiveScan_append {α : Type} [Add α] [Mul α] {stateDim : Nat}
    (h0 : TorchLean.Tensor α [stateDim])
    (xs ys : Array (Spec.DiagonalTransition α stateDim)) :
    Spec.diagonalSelectiveScan h0 (xs ++ ys) =
      Spec.diagonalSelectiveScan h0 xs ++
        Spec.diagonalSelectiveScan (Spec.runDiagonalTransitions h0 xs) ys := by
  simpa [Spec.diagonalSelectiveScan, Spec.runDiagonalTransitions] using
    congrArg Prod.snd (scanArray_append
      (fun state transition =>
        let nextState := transition.apply state
        (nextState, nextState)) h0 xs ys)

/--
A homogeneous affine transition over $\mathbb{R}$ is Lipschitz with factor $\rho$ whenever
$|a|\leq\rho$.

This is the one-channel stability lemma used to lift diagonal SSMs into contraction proofs.
-/
theorem abs_homogeneous_apply_le (a ρ h : ℝ) (ha : |a| ≤ ρ) :
    |(Spec.ScalarAffineTransition.apply { a := a, b := 0 } h)| ≤ ρ * |h| := by
  calc
    |(Spec.ScalarAffineTransition.apply { a := a, b := 0 } h)|
        = |a * h| := by simp [Spec.ScalarAffineTransition.apply]
    _ = |a| * |h| := by rw [abs_mul]
    _ ≤ ρ * |h| := mul_le_mul_of_nonneg_right ha (abs_nonneg h)

/-- A homogeneous scalar transition with $|a|\leq 1$ is non-expansive. -/
theorem abs_homogeneous_apply_le_self (a h : ℝ) (ha : |a| ≤ 1) :
    |(Spec.ScalarAffineTransition.apply { a := a, b := 0 } h)| ≤ |h| := by
  calc
    |(Spec.ScalarAffineTransition.apply { a := a, b := 0 } h)| ≤ 1 * |h| :=
      abs_homogeneous_apply_le a 1 h ha
    _ = |h| := by simp

end StateSpace
end MLTheory
end NN

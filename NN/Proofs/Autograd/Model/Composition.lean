/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Layers.Seq

/-!
# Execution of composed models

Sequential composition preserves the order of model state, forward calls, and buffer updates.
These equations keep the execution monad and scalar backend abstract. They therefore apply to
graph recording as well as eager execution, without expanding either backend's operation instance.
-/

public section

open Spec TorchLean Runtime.Autograd.Model Runtime.Autograd.Torch

namespace Runtime.Autograd.Model.Layers.Seq

/-- Composed models retain the state of the first model followed by the second model's state. -/
@[simp] theorem stateShapes_comp {σ τ υ : Shape} (first : Seq σ τ) (second : Seq τ υ) :
    stateShapes (comp first second) = stateShapes first ++ stateShapes second := by
  induction first with
  | id => rfl
  | cons layer rest ih => simp only [comp, stateShapes, ih, List.append_assoc]

/-- Applying a model's curried program passes its state and input to the sequential evaluator. -/
theorem forward_apply {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops m α] {σ τ : Shape}
    (model : Seq σ τ) (mode : Mode)
    (state : RefList (RefTy (m := m) (α := α)) (stateShapes model))
    (x : RefTy (m := m) (α := α) σ) :
    CurriedRef.uncurry (forward model mode (α := α) (m := m))
      (state.append (.cons x .nil)) = forwardState model mode state x := by
  simp only [forward]
  rw [CurriedRef.uncurry_curry, RefList.splitLast_append]

/-- A single-layer model in evaluation mode runs that layer without a buffer-update callback. -/
theorem forwardState_fromLayer_eval {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [LawfulMonad m] [Ops m α] {σ τ : Shape}
    (layer : Layer σ τ) (state : RefList (RefTy (m := m) (α := α)) layer.stateShapes)
    (x : RefTy (m := m) (α := α) σ) :
    forwardState (fromLayer layer) .eval
      (state.append .nil) x = layer.forwardRef .eval state x := by
  simp [fromLayer, forwardState]
  congr 1
  exact congrArg Prod.fst (RefList.split_append state (.nil))

private theorem forwardState_cons {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [LawfulMonad m] [Ops m α] {σ τ υ : Shape}
    (layer : Layer σ τ) (rest : Seq τ υ) (mode : Mode)
    (state : RefList (RefTy (m := m) (α := α)) layer.stateShapes)
    (restState : RefList (RefTy (m := m) (α := α)) (stateShapes rest))
    (x : RefTy (m := m) (α := α) σ) :
    forwardState (.cons layer rest) mode (state.append restState) x = (do
      let y ← forwardState (fromLayer layer) mode (state.append .nil) x
      forwardState rest mode restState y) := by
  dsimp only [fromLayer, forwardState, stateShapes]
  rw [RefList.split_append state restState, RefList.split_append state (.nil)]
  simp only [bind_assoc]
  congr 1
  funext y
  split
  · split
    · split <;> simp
    · simp
  · simp

private theorem append_assoc_cast {Ref : Shape → Type} {a b c d : List Shape}
    (h : b ++ c = d) (xs : RefList Ref a) (ys : RefList Ref b) (zs : RefList Ref c) :
    ((List.append_assoc a b c).trans (congrArg (a ++ ·) h) ▸
      RefList.append (RefList.append xs ys) zs) = RefList.append xs (h ▸ RefList.append ys zs) := by
  cases h
  exact RefList.append_assoc xs ys zs

/-- Composed model execution is monadic composition, with the original state and effect order.

In training mode this includes buffer-update callbacks after each layer. The proof uses only
monad laws; it neither commutes effects nor assumes algebraic laws on numerical operations.
-/
theorem forwardState_comp {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [LawfulMonad m] [Ops m α] {σ τ υ : Shape}
    (first : Seq σ τ) (second : Seq τ υ) (mode : Mode)
    (firstState : RefList (RefTy (m := m) (α := α)) (stateShapes first))
    (secondState : RefList (RefTy (m := m) (α := α)) (stateShapes second))
    (x : RefTy (m := m) (α := α) σ) :
    forwardState (comp first second) mode
      ((stateShapes_comp first second).symm ▸ firstState.append secondState) x =
      (forwardState first mode firstState x >>= forwardState second mode secondState) := by
  induction first with
  | id =>
      cases firstState
      simp [comp, forwardState]
      rfl
  | @cons σ τ υ layer rest ih =>
      obtain ⟨state, restState, hstate⟩ : ∃ state restState,
          RefList.append state restState = firstState :=
        ⟨(RefList.split (ss₁ := layer.stateShapes) firstState).1,
          (RefList.split (ss₁ := layer.stateShapes) firstState).2,
          RefList.append_split firstState⟩
      subst firstState
      have hcast : ((stateShapes_comp (.cons layer rest) second).symm ▸
          (state.append restState).append secondState) =
          state.append ((stateShapes_comp rest second).symm ▸ restState.append secondState) := by
        exact append_assoc_cast (stateShapes_comp rest second).symm state restState secondState
      dsimp only [comp, stateShapes] at hcast ⊢
      rw [hcast, forwardState_cons, forwardState_cons]
      simp only [bind_assoc]
      congr 1
      funext y
      exact ih second restState secondState y

end Runtime.Autograd.Model.Layers.Seq

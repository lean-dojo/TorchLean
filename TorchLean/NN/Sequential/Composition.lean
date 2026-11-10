/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Layer.Activation
public import TorchLean.NN.Sequential

/-!
# Execution of composed models

Sequential composition preserves the order of model state, forward calls, and buffer updates.
The `forwardState` equations keep the execution monad and scalar backend abstract, so they apply to
graph recording as well as eager execution without expanding either backend's operation instance.
`eval_id`, `eval_comp` and `eval_fromLayer` are their specializations to the pure interpreter at
`Id`, the denotation `nn.Sequential.eval`. `eval_fromLayer_op` evaluates a one-operator layer
`Layer.op` by the rule of its operator.
-/

public section

open TorchLean.Ops

namespace TorchLean.nn.Sequential

/-- Composed models retain the state of the first model followed by the second model's state. -/
@[simp] theorem stateShapes_comp {σ τ υ : Shape} (first : Sequential σ τ)
    (second : Sequential τ υ) :
    stateShapes (comp first second) = stateShapes first ++ stateShapes second := by
  induction first with
  | id => rfl
  | cons layer rest ih => simp only [comp, stateShapes, ih, List.append_assoc]

/-- Applying a model's curried program passes its state and input to the sequential evaluator. -/
theorem forward_apply {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops m α] {σ τ : Shape}
    (model : Sequential σ τ) (mode : Mode)
    (state : Pack (Ref (m := m) (α := α)) (stateShapes model))
    (x : Ref (m := m) (α := α) σ) :
    CurriedRef.uncurry (forward model mode (α := α) (m := m))
      (state.append (.cons x .nil)) = forwardState model mode state x := by
  simp only [forward]
  rw [CurriedRef.uncurry_curry, ← Pack.snoc_eq_append, Pack.unsnoc_snoc]

/-- A single-layer model in evaluation mode runs that layer without a buffer-update callback. -/
theorem forwardState_fromLayer_eval {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [LawfulMonad m] [Ops m α] {σ τ : Shape}
    (layer : Layer σ τ) (state : Pack (Ref (m := m) (α := α)) layer.stateShapes)
    (x : Ref (m := m) (α := α) σ) :
    forwardState (fromLayer layer) .eval
      (state.append .nil) x = layer.forwardRef .eval state x := by
  simp [fromLayer, forwardState]
  congr 1
  exact congrArg Prod.fst (Pack.split_append state (.nil))

private theorem forwardState_cons {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [LawfulMonad m] [Ops m α] {σ τ υ : Shape}
    (layer : Layer σ τ) (rest : Sequential τ υ) (mode : Mode)
    (state : Pack (Ref (m := m) (α := α)) layer.stateShapes)
    (restState : Pack (Ref (m := m) (α := α)) (stateShapes rest))
    (x : Ref (m := m) (α := α) σ) :
    forwardState (.cons layer rest) mode (state.append restState) x = (do
      let y ← forwardState (fromLayer layer) mode (state.append .nil) x
      forwardState rest mode restState y) := by
  dsimp only [fromLayer, forwardState, stateShapes]
  rw [Pack.split_append state restState, Pack.split_append state (.nil)]
  simp only [bind_assoc]
  congr 1
  funext y
  split
  · split
    · split <;> simp
    · simp
  · simp

private theorem append_assoc_cast {Ref : Shape → Type} {a b c d : List Shape}
    (h : b ++ c = d) (xs : Pack Ref a) (ys : Pack Ref b) (zs : Pack Ref c) :
    ((List.append_assoc a b c).trans (congrArg (a ++ ·) h) ▸
      Pack.append (Pack.append xs ys) zs) = Pack.append xs (h ▸ Pack.append ys zs) := by
  cases h
  exact eq_of_heq ((eqRec_heq _ _).trans (Pack.append_assoc_heq xs ys zs))

/-- Composed model execution is monadic composition, with the original state and effect order.

In training mode this includes buffer-update callbacks after each layer. The proof uses only
monad laws and neither commutes effects nor assumes algebraic laws on numerical operations.
-/
theorem forwardState_comp {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [LawfulMonad m] [Ops m α] {σ τ υ : Shape}
    (first : Sequential σ τ) (second : Sequential τ υ) (mode : Mode)
    (firstState : Pack (Ref (m := m) (α := α)) (stateShapes first))
    (secondState : Pack (Ref (m := m) (α := α)) (stateShapes second))
    (x : Ref (m := m) (α := α) σ) :
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
          Pack.append state restState = firstState :=
        ⟨(Pack.split (ss₁ := layer.stateShapes) firstState).1,
          (Pack.split (ss₁ := layer.stateShapes) firstState).2,
          Pack.append_split firstState⟩
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

/-- The identity model evaluates to its input. -/
@[simp] theorem eval_id {α : Type} [Storage α] [Context α] {σ : Shape} (x : Tensor α σ) :
    eval (Sequential.id σ) .nil x = x := rfl

/-- Evaluating a composed model evaluates the second model on the first model's output. -/
theorem eval_comp {α : Type} [Storage α] [Context α] {σ τ υ : Shape}
    (first : Sequential σ τ) (second : Sequential τ υ)
    (firstParams : TensorPack α (stateShapes first))
    (secondParams : TensorPack α (stateShapes second)) (x : Tensor α σ) :
    eval (comp first second) ((stateShapes_comp first second).symm ▸
        firstParams.append secondParams) x =
      eval second secondParams (eval first firstParams x) := by
  simp only [eval]
  exact forwardState_comp (m := Id) first second .eval firstParams secondParams x

/-- Evaluating a single-layer model runs that layer's forward program under the pure
interpreter. -/
theorem eval_fromLayer {α : Type} [Storage α] [Context α] {σ τ : Shape} (layer : Layer σ τ)
    (params : TensorPack α layer.stateShapes) (x : Tensor α σ) :
    eval (fromLayer layer) (params.append .nil) x =
      layer.forwardRef (m := Id) .eval params x := by
  simp only [eval]
  exact forwardState_fromLayer_eval (m := Id) layer params x

/-- A one-operator layer evaluates to the first output of the operator's rule at the empty
capability record, or to zeros when the table has no rule for it. -/
theorem eval_fromLayer_op {α : Type} [Storage α] [Context α] {σ τ : Shape} {rest : List Port}
    (o : Op [.value σ] (.value τ :: rest)) (h : o.need = .none) (x : Tensor α σ) :
    eval (fromLayer (Layer.op o h)) .nil x =
      match @Op.rule? α _ _ Op.Need.Supplied.empty.caps _ _ o with
      | some rule => (rule.forward (x, ())).1
      | none => Tensor.full τ 0 := by
  change Ops.pureApply o Op.Need.Supplied.empty.caps (x, ()) = _
  unfold Ops.pureApply
  cases @Op.rule? α _ _ Op.Need.Supplied.empty.caps _ _ o <;> rfl

end TorchLean.nn.Sequential

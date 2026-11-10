/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Context.Real
public import TorchLean.NN.Optimizer.Update

/-!
# First-Order Optimizer Relations

Relations between distinct first-order optimizer updates, phrased over `TorchLean.Tensor`.

Projected SGD with the identity projector updates the parameters as SGD does, at every scalar type
(`GaLore.update_params_eq_sgd`, by definition). Over `ℝ`, a zero weight-decay
AdamW update is the Adam update (`AdamW.update_params_eq_adam`).
-/

@[expose] public section


namespace TorchLean
open TorchLean.Tensor

namespace AdamW

/--
AdamW reduces to Adam on the Adam state it extends when `weightDecay = 0` (parameter-update
equality), over `ℝ`.

`Context α` gives executable operators but no algebraic laws such as `x * 0 = 0`, so the statement
is made over `ℝ`, where Mathlib supplies them.
-/
theorem update_params_eq_adam {s : Shape}
    (state : State ℝ s) (parameters gradients : Tensor ℝ s) (hwd : state.weightDecay = 0) :
    (update state parameters gradients).parameters =
      (Adam.update state.toState parameters gradients).parameters := by
  apply Tensor.Rep.ext
  intro coordinate
  simp [AdamW.update, Adam.update, hwd, Tensor.sub_def, Tensor.scale]

end AdamW

namespace GaLore

/-- Projected SGD with the identity projector gives the parameters of SGD with the same learning
rate, at every scalar type. -/
theorem update_params_eq_sgd {α : Type} [Storage α] [Sub α] [Mul α]
    {s : Shape} (lr : α) (parameters gradients : Tensor α s) :
    (update { lr := lr, projector := identityProjector } parameters gradients).parameters =
      (SGD.update { lr := lr } parameters gradients).parameters :=
  rfl

end GaLore

end TorchLean

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Optim.Optimizers

/-!
# Runtime Optimizer Equations

Small executable theorems about TorchLean's optimizer equations.

These are intentionally modest. They prove properties of the update rules that TorchLean actually
executes, rather than broad convergence claims that would require assumptions about convexity,
smoothness, stochastic gradients, and floating-point error. Larger optimization theory can build on
these equations.

This is the tensor-facing layer: the statements are phrased over `Spec.Tensor` and the executable
operator dictionary `Spec.Context`. When we want ordinary algebraic simplification, such as proving
that a zero weight-decay AdamW update is the same parameter update as Adam, we specialize to `ℝ`,
where mathlib provides the ring laws.
-/

@[expose] public section


namespace Optim
open Spec
open Tensor

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

namespace SGD

/-- SGD is definitionally `p - lr * g`. -/
theorem update_eq {s : Shape} (state : State α s) (params grads : Tensor α s) :
    update state params grads = subSpec params (scaleSpec grads state.lr) := by
  rfl

end SGD

namespace MomentumSGD

/-- Momentum SGD updates its buffer to `momentum * old_buffer + gradient`. -/
theorem update_buffer_eq {s : Shape} (state : State α s) (params grads : Tensor α s) :
    (update state params grads).1.buf =
      addSpec (scaleSpec state.buf state.momentum) grads := by
  rfl

/-- Momentum SGD updates parameters using the freshly updated buffer. -/
theorem update_params_eq {s : Shape} (state : State α s) (params grads : Tensor α s) :
    (update state params grads).2 =
      subSpec params
        (scaleSpec (addSpec (scaleSpec state.buf state.momentum) grads) state.lr) := by
  rfl

end MomentumSGD

namespace Adam

/-- Adam increments its bias-correction counter by exactly one every step. -/
theorem update_time_eq {s : Shape} (state : State α s) (params grads : Tensor α s) :
    (update state params grads).1.t = state.t + 1 := by
  rfl

end Adam

namespace AdamW

/-- AdamW increments its bias-correction counter by exactly one every step. -/
theorem update_time_eq {s : Shape} (state : State α s) (params grads : Tensor α s) :
    (update state params grads).1.t = state.t + 1 := by
  rfl

/-!
`Context α` gives us executable operators, but not algebraic laws like `x * 0 = 0`.

Optimizer-relationship theorems (AdamW → Adam, L2 vs weight decay, etc.) therefore live most
naturally over proof backends like `ℝ` where the laws are available from Mathlib.
-/

/--
AdamW reduces to Adam when `weight_decay = 0` (parameter-update equality), over `ℝ`.
-/
theorem update_weight_decay_zero_params_eq_adam_real {s : Shape}
    (state : State ℝ s) (params grads : Tensor ℝ s) (hwd : state.weight_decay = 0) :
    (update state params grads).2 =
      (Adam.update
          ({ lr := state.lr, beta1 := state.beta1, beta2 := state.beta2, epsilon := state.epsilon,
             m := state.m, v := state.v, t := state.t } : Adam.State ℝ s)
          params grads).2 := by
  -- `Context` gives operators; to simplify the decoupled decay term we use `ℝ`'s algebraic laws
  -- together with structural recursion on TorchLean spec tensors.
  --
  -- Helper lemmas: scaling by `0` yields the all-zero tensor, and subtracting that yields identity.
  have scaleSpec_zero : ∀ {s : Shape} (t : Tensor ℝ s), scaleSpec t (0 : ℝ) = fill 0 s := by
    intro s t
    induction t with
    | scalar x =>
        simp [Tensor.scaleSpec, Tensor.mapSpec, Spec.fill]
    | dim g ih =>
        -- `fill` and `mapSpec` are definitional: function ext + IH is enough.
        apply congrArg Tensor.dim
        funext i
        simpa [Tensor.scaleSpec, Tensor.mapSpec, Spec.fill] using ih i
  have subSpec_fill_zero : ∀ {s : Shape} (t : Tensor ℝ s), subSpec t (fill 0 s) = t := by
    intro s t
    induction t with
    | scalar x =>
        simp [Tensor.subSpec, Tensor.map2Spec, Spec.fill]
    | dim g ih =>
        apply congrArg Tensor.dim
        funext i
        simpa [Tensor.subSpec, Tensor.map2Spec, Spec.fill] using ih i
  cases state with
  | mk lr beta1 beta2 epsilon weight_decay m v t =>
      -- Now `weight_decay = 0` makes the decayed-parameter term a no-op.
      -- We rewrite `weight_decay` first, then discharge the tensor equalities via the helpers.
      have hwd' : weight_decay = 0 := by simpa using hwd
      -- `simp` doesn't know tensor algebra, but it will reduce the remaining goal to our helpers.
      simp [AdamW.update, Adam.update, hwd', scaleSpec_zero, subSpec_fill_zero]

end AdamW

namespace Adadelta

/-- Adadelta's gradient accumulator follows the documented EMA equation. -/
theorem update_v_eq {s : Shape} (state : State α s) (params grads : Tensor α s) :
    (update state params grads).1.v =
      addSpec (scaleSpec state.v state.rho)
        (scaleSpec (squareSpec grads) (1 - state.rho)) := by
  rfl

end Adadelta

end Optim

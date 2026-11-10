/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Builder.Layer

/-!
# Model Automatic Differentiation

Derivatives of a model with respect to its state and its input. `autograd.model.State` is the
state of a model, and `autograd.model.Loss` is a scalar loss on the model output and a target.
`Loss.mse` and `Loss.oneHotCrossEntropy` build losses, and `Loss.detach` stops the gradient
through the model output.

* `autograd.model.grad` differentiates a loss with respect to the state.
* `autograd.model.vjp` pulls an output cotangent back to the state and the input.
* `autograd.model.jacrev` gives the Jacobian of the output with respect to the state.
* `autograd.model.jvp` and `autograd.model.hvp` give the directional derivative and the
  Hessian-vector product of a loss along a direction in the state.

The modules under `TorchLean/NN/Autograd/` make up the public autograd API.
-/

@[expose] public section

namespace TorchLean

namespace autograd

namespace model

/-- The complete state of `model`, parameters and persistent buffers, indexed by its statically
known tensor shapes. PyTorch analogue: `model.state_dict()`. -/
abbrev State {σ τ : Shape}
    (model : nn.Sequential σ τ) (α : Type) [TorchLean.Storage α] :=
  TensorPack α (nn.Sequential.stateShapes model)

/-- A scalar loss on a model output of shape `τ` and a target of shape `υ`, together with the
configuration checks that `grad`, `jvp` and `hvp` run before evaluating it. -/
structure Loss (τ υ : Shape) : Type 1 where
  /-- The loss as a program over the differentiable operations, generic in the scalar type and the
  execution monad. -/
  forward : ∀ {α : Type}, [TorchLean.Storage α] → [Context α] → {m : Type → Type} → [Monad m] →
      [Ops (m := m) (α := α)] →
      Ops.Ref (m := m) (α := α) τ →
      Ops.Ref (m := m) (α := α) υ →
      m (Ops.Ref
        (m := m) (α := α) [])
  /-- Configuration checks, run before the loss is lowered or executed. The default accepts every
  configuration. -/
  validate : Except String Unit := pure ()

namespace Loss

/-- Mean squared error between a model output and its target. The default `.mean` reduction
averages over all entries, and `.sum` adds the squared differences.
PyTorch analogue: `torch.nn.MSELoss`. -/
def mse {τ : Shape}
    (reduction : TorchLean.Loss.Reduction := .mean) :
    model.Loss τ τ :=
  { forward := fun {α} _ _ => fun {m} _ _ output target =>
      TorchLean.Loss.mse
        (m := m) (α := α) (s := τ) output target
        (reduction := reduction) }

/-- Cross-entropy between logits and one-hot targets along the class dimension `axis`. If `axis`
is not below `τ.rank`, `validate` fails with an error naming the axis. -/
def oneHotCrossEntropy {τ : Shape}
    (axis : Nat)
    (reduction : TorchLean.Loss.Reduction := .mean) :
    model.Loss τ τ :=
  if axisInBounds : axis < τ.rank then
    letI : Shape.AxisInBounds axis τ :=
      Shape.AxisInBounds.mk axisInBounds
    { forward := fun {elementType} _ _ => fun {m} _ _ logits target =>
        TorchLean.Loss.oneHotCrossEntropy
          (m := m) (α := elementType) (s := τ) axis logits target
          (reduction := reduction) }
  else
    { forward := fun {elementType} _ _ => fun {m} _ _ logits _target =>
        TorchLean.Loss.mse
          (m := m) (α := elementType) (s := τ) logits logits
          (reduction := .sum)
      validate :=
        .error s!"OneHotCrossEntropy: axis {axis} is out of bounds for rank {τ.rank}" }

/-- Evaluate `loss` on a detached copy of the model output, so no gradient flows back into the
model. The checks of `loss` carry over. PyTorch analogue: `loss_fn(output.detach(), target)`. -/
def detach {τ υ : Shape}
    (loss : model.Loss τ υ) :
    model.Loss τ υ :=
  { forward := fun {α} _ _ => fun {m} _ _ output target => do
      let detachedOutput ← Ops.apply (m := m) (α := α) (.detach (s := τ))
          (output, ())
      loss.forward (α := α) (m := m) detachedOutput target
    validate := loss.validate }

end Loss

/-- Lower `loss (model state input) target` to the typed scalar program used by autograd. Its
arguments are the state tensors followed by the input and the target. -/
def program {σ τ υ : Shape}
    (model : nn.Sequential σ τ)
    (loss : Loss τ υ) :
    ∀ {α : Type}, [TorchLean.Storage α] → [Context α] → Program α
        (nn.Sequential.stateShapes model ++
          [σ, υ])
        [] :=
  fun {α} _ _ => fun {m} _ _ =>
    CurriedRef.curry
      (Ref := fun s => Ops.Ref (m := m) (α := α) s)
      (ss := nn.Sequential.stateShapes model ++
        [σ, υ])
      (β := m (Ops.Ref
        (m := m) (α := α) []))
      (fun arguments => do
        let (state, inputs) :=
          TorchLean.Pack.split
            (F := fun s => Ops.Ref
              (m := m) (α := α) s)
            (ss₁ := nn.Sequential.stateShapes model)
            (ss₂ := [σ, υ]) arguments
        let (input, target) := match inputs with
          | .cons input (.cons target .nil) => (input, target)
        let output ←
          CurriedRef.uncurry
            (Ref := fun s => Ops.Ref
              (m := m) (α := α) s)
            (ss := nn.Sequential.stateShapes model ++
              [σ])
            (β := m (Ops.Ref
              (m := m) (α := α) τ))
            (nn.Sequential.forward model (α := α))
            (TorchLean.Pack.append state (.cons input .nil))
        loss.forward (α := α) (m := m) output target)

/-- Reject an invalid model or loss before lowering an autograd program. -/
def validate {σ τ υ : Shape}
    (model : nn.Sequential σ τ) (loss : Loss τ υ) : IO Unit := do
  IO.ofExcept (nn.Sequential.validate model)
  IO.ofExcept loss.validate

/--
Differentiate a model loss with respect to every tensor in the model state.

The result has the same shape-indexed layout as `State model α`. Set `value := true` to return
`(grad, lossValue)` from the same evaluation. Persistent buffers receive their mathematical
sensitivities too, except where an operator reads a buffer as a frozen input: BatchNorm's
running statistics in evaluation mode get zero. Which state tensors an optimizer updates is set
separately, by `nn.Sequential.requiresGrad`.

PyTorch analogue: `torch.func.grad` of the loss as a function of the model state, and
`torch.func.grad_and_value` for `value := true`.
-/
def grad {σ τ υ : Shape}
    (model : nn.Sequential σ τ) (loss : Loss τ υ)
    {α : Type} [TorchLean.Storage α] [Context α]
    (state : State model α)
    (input : Tensor α σ) (target : Tensor α υ)
    (value : Bool := false) :
    IO (match value with
      | false => State model α
      | true => State model α × Tensor α []) := by
  cases value with
  | false =>
      exact do
        validate model loss
        let (grad, _) ← Autodiff.gradients
          (α := α)
          (paramShapes := nn.Sequential.stateShapes model)
          (inputShapes := [σ, υ])
          (program model loss)
          state
          (TorchLean.Pack.pair input target)
        pure grad
  | true =>
      exact do
        validate model loss
        let stateShapes := nn.Sequential.stateShapes model
        let graph ←
          Autodiff.lowerToTypedGraph (α := α)
            (paramShapes := stateShapes)
            (inputShapes := [σ, υ])
            (program model loss)

        let arguments : TorchLean.TensorPack α
            (stateShapes ++ [σ, υ]) :=
          TorchLean.Pack.append (ss₁ := stateShapes)
            (ss₂ := [σ, υ])
            state
            (TorchLean.Pack.pair input target)

        let (allGradients, lossValue) ←
          Autodiff.pullback
            graph arguments (Tensor.scalar (1 : α))

        let (grad, _) :=
          TorchLean.Pack.split (ss₁ := stateShapes)
            (ss₂ := [σ, υ]) allGradients
        pure (grad, lossValue)

/--
Vector-Jacobian product of the model output with respect to the state and the input.

Pull `outputGradient` back through `model` and return `(stateGrad, inputGrad)`. Both values come
from one reverse pass. PyTorch analogue: the pullback that `torch.func.vjp` returns for the model
as a function of its state and input, applied to `outputGradient`.
-/
def vjp {σ τ : Shape} (model : nn.Sequential σ τ)
    {α : Type} [TorchLean.Storage α] [Context α]
    (state : State model α)
    (input : Tensor α σ) (outputGradient : Tensor α τ) :
    IO (State model α × Tensor α σ) := do
  IO.ofExcept (nn.Sequential.validate model)
  let (stateGrad, inputGrads) ← Autodiff.vjp
    (α := α)
    (paramShapes := nn.Sequential.stateShapes model)
    (inputShapes := [σ]) (τ := τ)
    (fun {β} _ _ =>
      nn.Sequential.forward model (α := β))
    state
    (TorchLean.Pack.singleton input) outputGradient
  pure
    (stateGrad,
      TorchLean.Pack.head inputGrads)

/--
Reverse-mode Jacobian (`jacrev`) of the model output with respect to model state.

Returns one Jacobian tensor per state tensor, with the output axes followed by that state
tensor's axes. Each output entry costs one reverse pass. PyTorch analogue: `torch.func.jacrev`
of the model as a function of its state, which returns one Jacobian per tensor in this layout.
-/
def jacrev {σ τ : Shape}
    (model : nn.Sequential σ τ)
    {α : Type} [TorchLean.Storage α] [Context α]
    (state : State model α)
    (input : Tensor α σ) :
    IO (TensorPack α ((nn.Sequential.stateShapes model).map (τ ++ ·))) := do
  IO.ofExcept (nn.Sequential.validate model)
  let rows ← Autodiff.jacrev
    (α := α)
    (paramShapes := nn.Sequential.stateShapes model)
    (inputShapes := [σ]) (τ := τ)
    (fun {β} _ _ =>
      nn.Sequential.forward model (α := β))
    state
    (TorchLean.Pack.singleton input) Prod.fst
  pure rows

/--
Jacobian-vector product (JVP) of a scalar loss with respect to model state.

The result is the directional derivative of the loss along `stateDirection`, computed in one
forward-mode pass. With $x$ for `input`, and with `input` and `target` held fixed, it is

$$
\left.\frac{d}{dt}
\operatorname{loss}(\mathrm{state}+t\,\mathrm{stateDirection},x,\mathrm{target})
\right|_{t=0}.
$$

PyTorch analogue: the tangent that `torch.func.jvp` returns for the loss as a function of the
model state.
-/
def jvp {σ τ υ : Shape}
    (model : nn.Sequential σ τ) (loss : Loss τ υ)
    {α : Type} [TorchLean.Storage α] [Context α]
    (state : State model α)
    (input : Tensor α σ) (target : Tensor α υ)
    (stateDirection : State model α) :
    IO (Tensor α []) := do
  validate model loss
  Autodiff.jvpLossParams
    (α := α)
    (paramShapes := nn.Sequential.stateShapes model)
    (inputShapes := [σ, υ])
    (program model loss)
    state
    (TorchLean.Pack.pair input target)
    stateDirection

/--
Hessian-vector product (HVP) of a scalar loss with respect to model state.

The Hessian of the loss in the state is applied to `stateDirection`, with `input` and `target`
held fixed. The result has the same shape layout as `state`. It comes from one reverse pass over
dual numbers whose tangents are `stateDirection`.
PyTorch analogue: `torch.autograd.functional.hvp`.
-/
def hvp {σ τ υ : Shape}
    (model : nn.Sequential σ τ) (loss : Loss τ υ)
    {α : Type} [TorchLean.Storage α] [Context α]
    (state : State model α)
    (input : Tensor α σ) (target : Tensor α υ)
    (stateDirection : State model α) :
    IO (State model α) := do
  validate model loss
  let (result, _) ← Autodiff.hvp
    (α := α)
    (paramShapes := nn.Sequential.stateShapes model)
    (inputShapes := [σ, υ])
    (program model loss)
    state
    (TorchLean.Pack.pair input target)
    stateDirection
    (TorchLean.TensorPack.zero (ss := [σ, υ]))
  pure result
end model

end autograd

end TorchLean

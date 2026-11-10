/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Session.Selection
public import TorchLean.NN.Autodiff

/-!
# Function Transforms

Automatic differentiation transforms for one-argument tensor programs, in the style of
`torch.func`: `autograd.grad`, `autograd.vjp`, `autograd.jacrev`, `autograd.jacfwd` and
`autograd.hessian`. Programs use the differentiable operations throughout, since a value computed
on the host falls outside the recorded graph. The same program can be interpreted with different
scalar types. The transforms differentiate programs of type
`autograd.Function`, so an ordinary Lean function has to be written in that form first.
The modules under `TorchLean/NN/Autograd/` make up the public autograd API.
-/

@[expose] public section

namespace TorchLean

namespace autograd

/-- A scalar-polymorphic tensor function written against TorchLean's differentiable operations. -/
abbrev Function (σ τ : Shape) :=
  ∀ {α : Type}, [TorchLean.Storage α] → [Context α] → {m : Type → Type} → [Monad m] →
      [Ops (m := m) (α := α)] →
      Ops.Ref (m := m) (α := α) σ →
      m (Ops.Ref (m := m) (α := α) τ)

/--
Present a `Function` as the one-argument `Program` the autograd runtime consumes.

`curry` presents the runtime's single tensor argument as the `Ops.Ref` a
`Function` expects.
-/
def program {σ τ : Shape}
    (f : Function σ τ) :
    ∀ {α : Type}, [TorchLean.Storage α] → [Context α] → Program α [σ] τ :=
  fun {α} _ _ => fun {m} _ _ =>
    CurriedRef.curry
      (Ref := fun s => Ops.Ref (m := m) (α := α) s)
      (ss := [σ])
      (β := m (Ops.Ref
        (m := m) (α := α) τ))
      (fun arguments =>
        match arguments with
        | .cons input .nil => f (α := α) (m := m) input)

/--
Forward-mode Jacobian with output axes followed by input axes, matching `jacrev`. PyTorch
analogue: `torch.func.jacfwd`.
-/
def jacfwd {σ τ : Shape} (f : Function σ τ)
    {α : Type} [TorchLean.Storage α] [Context α]
    (input : Tensor α σ) : IO (Tensor α (τ ++ σ)) :=
  Autodiff.jacfwdInput
    (α := α) (σ := σ) (τ := τ) (program f) input

/--
Hessian of a scalar function, with one copy of the input axes for each derivative. PyTorch
analogue: `torch.func.hessian`.
-/
def hessian {σ : Shape} (f : Function σ [])
    {α : Type} [TorchLean.Storage α] [Context α]
    (input : Tensor α σ) : IO (Tensor α (σ ++ σ)) :=
  Autodiff.hessianInput
    (α := α) (σ := σ) (program f) input

/--
Vector-Jacobian product: pull `outputGradient` back through `f` at `input`. PyTorch analogue:
`torch.func.vjp`.
-/
def vjp {σ τ : Shape} (f : Function σ τ)
    {α : Type} [TorchLean.Storage α] [Context α]
    (input : Tensor α σ) (outputGradient : Tensor α τ) :
    IO (Tensor α σ) := do
  let emptyState : TorchLean.TensorPack α ([] : List Shape) := TorchLean.Pack.nil
  let (_, inputGradients) ←
    Autodiff.vjp (α := α)
      (paramShapes := ([] : List Shape)) (inputShapes := [σ])
      (τ := τ)
      (program (σ := σ) (τ := τ) f)
      emptyState (TorchLean.Pack.singleton input) outputGradient
  pure (TorchLean.Pack.head inputGradients)

/--
Reverse-mode Jacobian (`jacrev`) of a pure tensor function.

Returns a tensor with output axes followed by input axes, matching `jacfwd`. PyTorch analogue:
`torch.func.jacrev`.
-/
def jacrev {σ τ : Shape} (f : Function σ τ)
    {α : Type} [TorchLean.Storage α] [Context α]
    (input : Tensor α σ) :
    IO (Tensor α (τ ++ σ)) := do
  let emptyState : TorchLean.TensorPack α ([] : List Shape) := TorchLean.Pack.nil
  let rows ←
    Autodiff.jacrev (α := α)
      (paramShapes := ([] : List Shape)) (inputShapes := [σ])
      (τ := τ)
      (program (σ := σ) (τ := τ) f)
      emptyState (TorchLean.Pack.singleton input) Prod.snd
  pure (TorchLean.Pack.head rows)

/--
Differentiate a scalar-valued function with respect to its input.

By default this returns only the gradient, as `torch.func.grad` does. Set `value := true` to
return `(gradient, functionValue)` from the same evaluation, as `torch.func.grad_and_value` does.
-/
def grad {σ : Shape} (f : Function σ [])
    {α : Type} [TorchLean.Storage α] [Context α]
    (input : Tensor α σ) (value : Bool := false) :
    IO (match value with
      | false => Tensor α σ
      | true => Tensor α σ × Tensor α []) := by
  cases value with
  | false =>
      exact do
        let emptyState : TorchLean.TensorPack α ([] : List Shape) :=
          TorchLean.Pack.nil
        let (_, inputGradients) ←
          Autodiff.gradients (α := α)
            (paramShapes := ([] : List Shape)) (inputShapes := [σ])
            (program (σ := σ) (τ := []) f)
            emptyState (TorchLean.Pack.singleton input)
        pure (TorchLean.Pack.head inputGradients)
  | true =>
      exact do
        let graph ←
          Autodiff.lowerToTypedGraph (α := α)
            (paramShapes := ([] : List Shape)) (inputShapes := [σ])
            (program (σ := σ) (τ := []) f)
        let arguments : TorchLean.TensorPack α [σ] :=
          TorchLean.Pack.singleton input
        let (gradients, functionValue) ←
          Autodiff.pullback
            graph arguments (Tensor.scalar (1 : α))
        pure (TorchLean.Pack.head gradients, functionValue)

end autograd

end TorchLean

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Runtime
public import NN.Runtime.Autograd.Model.Autodiff
public import NN.Tensor -- shake: keep

/-!
# Function Transforms

Automatic differentiation transforms for pure one-argument tensor functions.
Import `NN.API.Autograd` for the complete public autograd API.
-/

@[expose] public section

namespace TorchLean

namespace autograd

/-
Pure-function autograd: treat a pure function `f : Tensor σ -> Tensor τ` as the object of
differentiation (no parameters).
-/

/-!
In PyTorch terms, this is the "functorch" style: differentiate plain functions, not modules.
-/

/-- A scalar-polymorphic tensor function written against TorchLean's differentiable operations. -/
abbrev Function (σ τ : Shape) :=
  ∀ {α : Type}, [TorchLean.Storage α] → [Context α] → {m : Type → Type} → [Monad m] →
      [Runtime.Autograd.Torch.Ops (m := m) (α := α)] →
      TorchLean.Runtime.ValueRef (m := m) (α := α) σ →
      m (TorchLean.Runtime.ValueRef (m := m) (α := α) τ)

/--
Present a `Function` as the one-argument `Program` the autograd runtime consumes.

The wrapping is pure plumbing: `curry` turns the runtime's heterogeneous argument list into the
single `ValueRef` a `Function` expects. Every differentiation entry point below goes through here,
so exactly one place knows the arity convention.
-/
def Internal.functionProgram {σ τ : Shape}
    (f : Function σ τ) :
    ∀ {α : Type}, [TorchLean.Storage α] → [Context α] → Runtime.Autograd.Model.Program α [σ] τ :=
  fun {α} _ _ => fun {m} _ _ =>
    Runtime.Autograd.Torch.CurriedRef.curry
      (Ref := fun s => TorchLean.Runtime.ValueRef (m := m) (α := α) s)
      (ss := [σ])
      (β := m (TorchLean.Runtime.ValueRef
        (m := m) (α := α) τ))
      (fun arguments =>
        match arguments with
        | .cons input .nil => f (α := α) (m := m) input)

/-- Forward-mode Jacobian with output axes followed by input axes, matching `jacrev`. -/
def jacfwd {σ τ : Shape} (f : Function σ τ)
    {α : Type} [TorchLean.Storage α] [Context α]
    (input : Tensor α σ) : IO (Tensor α (τ.concat σ)) :=
  Runtime.Autograd.Model.Autodiff.jacfwdInput
    (α := α) (σ := σ) (τ := τ) (Internal.functionProgram f) input

/-- Hessian of a scalar function, with one copy of the input axes for each derivative. -/
def hessian {σ : Shape} (f : Function σ [])
    {α : Type} [TorchLean.Storage α] [Context α]
    (input : Tensor α σ) : IO (Tensor α (σ.concat σ)) :=
  Runtime.Autograd.Model.Autodiff.hessianInput
    (α := α) (σ := σ) (Internal.functionProgram f) input

/-- Vector-Jacobian product (VJP) for a pure function. -/
def vjp {σ τ : Shape} (f : Function σ τ)
    {α : Type} [TorchLean.Storage α] [Context α]
    (input : Tensor α σ) (outputGradient : Tensor α τ) :
    IO (Tensor α σ) := do
  let emptyState : TorchLean.TensorPack α ([] : List Shape) := TorchLean.TensorPack.empty
  let (_, inputGradients) ←
    Runtime.Autograd.Model.Autodiff.vjp (α := α)
      (paramShapes := ([] : List Shape)) (inputShapes := [σ])
      (τ := τ)
      (Internal.functionProgram (σ := σ) (τ := τ) f)
      emptyState (TorchLean.TensorPack.singleton input) outputGradient
  pure (TorchLean.TensorPack.head inputGradients)

/--
Reverse-mode Jacobian (`jacrev`) of a pure tensor function.

Returns a tensor with output axes followed by input axes, matching `jacfwd`.
-/
def jacrev {σ τ : Shape} (f : Function σ τ)
    {α : Type} [TorchLean.Storage α] [Context α]
    (input : Tensor α σ) :
    IO (Tensor α (τ.concat σ)) := do
  let emptyState : TorchLean.TensorPack α ([] : List Shape) := TorchLean.TensorPack.empty
  let rows ←
    Runtime.Autograd.Model.Autodiff.jacrevOutInputs (α := α)
      (paramShapes := ([] : List Shape)) (inputShapes := [σ])
      (τ := τ)
      (Internal.functionProgram (σ := σ) (τ := τ) f)
      emptyState (TorchLean.TensorPack.singleton input)
  pure (TorchLean.TensorPack.head rows)

/--
Differentiate a scalar-valued function with respect to its input.

By default this returns only the gradient. Set `value := true` to return
`(gradient, functionValue)` from the same evaluation.
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
          TorchLean.TensorPack.empty
        let (_, inputGradients) ←
          Runtime.Autograd.Model.Autodiff.gradients (α := α)
            (paramShapes := ([] : List Shape)) (inputShapes := [σ])
            (Internal.functionProgram (σ := σ) (τ := []) f)
            emptyState (TorchLean.TensorPack.singleton input)
        pure (TorchLean.TensorPack.head inputGradients)
  | true =>
      exact do
        let graph ←
          Runtime.Autograd.Model.Autodiff.lowerScalarToTypedGraph (α := α)
            (paramShapes := ([] : List Shape)) (inputShapes := [σ])
            (Internal.functionProgram (σ := σ) (τ := []) f)
        let arguments : TorchLean.TensorPack α [σ] :=
          TorchLean.TensorPack.singleton input
        let (gradients, functionValue) ←
          Runtime.Autograd.Model.Autodiff.Impl.vjpWithValue
            graph arguments (Tensor.scalar (1 : α))
        pure (TorchLean.TensorPack.head gradients, functionValue)

end autograd

end TorchLean

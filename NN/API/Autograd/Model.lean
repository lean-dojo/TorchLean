/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Neural.Builders
public import NN.API.Sample -- shake: keep

/-!
# Model Automatic Differentiation

Model state, output losses, and differentiation with respect to model state and inputs.
Import `NN.API.Autograd` for the complete public autograd API.
-/

@[expose] public section

namespace TorchLean

namespace autograd

namespace model

/-
Model-shaped autograd: a TorchLean `NN.Seq` plus a `Loss` over its output.

This covers the common training use case.
-/

/-- Complete model state, indexed by its statically known tensor shapes. -/
abbrev State {σ τ : Shape}
    (model : nn.Sequential σ τ) (α : Type) [TorchLean.Storage α] :=
  nn.State α (Runtime.Autograd.Model.Layers.Seq.stateShapes model)

/-- Construct a model-shaped state whose every tensor contains `value`. -/
def fullState {σ τ : Shape}
    (model : nn.Sequential σ τ)
    {α : Type} [TorchLean.Storage α] (value : α) : State model α :=
  nn.State.full value

/-- A checked scalar loss computed from a model output and its target. -/
structure Loss (τ υ : Shape) : Type 1 where
  /-- Operation-polymorphic loss program. -/
  forward : ∀ {α : Type}, [TorchLean.Storage α] → [Context α] → {m : Type → Type} → [Monad m] →
      [Runtime.Autograd.Torch.Ops (m := m) (α := α)] →
      TorchLean.Runtime.ValueRef (m := m) (α := α) τ →
      TorchLean.Runtime.ValueRef (m := m) (α := α) υ →
      m (TorchLean.Runtime.ValueRef
        (m := m) (α := α) [])
  /-- Configuration checks performed before lowering or execution. -/
  validate : Except String Unit := pure ()

/-- Initialize model state in an element type that accepts host `Float` values. -/
def initialState {σ τ : Shape}
    (model : nn.Sequential σ τ)
    {α : Type} [TorchLean.Storage α] [Runtime.FromFloat α] : State model α :=
  nn.State.Internal.fromTensorPack <|
    Runtime.Autograd.Model.Module.castPack (Runtime.ofFloat (α := α))
      (Runtime.Autograd.Model.Layers.Seq.initState model)

namespace Loss

/-- Mean-squared error between a model output and its target. -/
def meanSquaredError {τ : Shape}
    (reduction : TorchLean.Loss.Reduction := .mean) :
    model.Loss τ τ :=
  { forward := fun {α} _ _ => fun {m} _ _ output target =>
      TorchLean.Loss.mse
        (m := m) (α := α) (s := τ) output target
        (reduction := reduction) }

/-- Cross-entropy between logits and one-hot targets along the selected class dimension. -/
def oneHotCrossEntropy {τ : Shape}
    (axis : Nat)
    (reduction : TorchLean.Loss.Reduction := .mean) :
    model.Loss τ τ :=
  if axisInBounds : axis < τ.rank then
    letI : Spec.Shape.AxisInBounds axis τ :=
      Spec.Shape.AxisInBounds.ofRank axisInBounds
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

/-- Stop gradients through the model output before evaluating `loss`. -/
def detach {τ υ : Shape}
    (loss : model.Loss τ υ) :
    model.Loss τ υ :=
  { forward := fun {α} _ _ => fun {m} _ _ output target => do
      let detachedOutput ← Runtime.Autograd.Model.F.detach
        (m := m) (α := α) (s := τ) output
      loss.forward (α := α) (m := m) detachedOutput target
    validate := loss.validate }

end Loss

/-- Lower `loss (model state input) target` to the typed scalar program used by autograd. -/
def Internal.lossProgram {σ τ υ : Shape}
    (model : nn.Sequential σ τ)
    (loss : Loss τ υ) :
    ∀ {α : Type}, [TorchLean.Storage α] → [Context α] → Runtime.Autograd.Model.Program α
        (Runtime.Autograd.Model.Layers.Seq.stateShapes model ++
          [σ, υ])
        [] :=
  fun {α} _ _ => fun {m} _ _ =>
    Runtime.Autograd.Torch.CurriedRef.curry
      (Ref := fun s => TorchLean.Runtime.ValueRef (m := m) (α := α) s)
      (ss := Runtime.Autograd.Model.Layers.Seq.stateShapes model ++
        [σ, υ])
      (β := m (TorchLean.Runtime.ValueRef
        (m := m) (α := α) []))
      (fun arguments => do
        let (state, inputs) :=
          Runtime.Autograd.Torch.RefList.split
            (Ref := fun s => TorchLean.Runtime.ValueRef
              (m := m) (α := α) s)
            (ss₁ := Runtime.Autograd.Model.Layers.Seq.stateShapes model)
            (ss₂ := [σ, υ]) arguments
        let (input, target) := match inputs with
          | .cons input (.cons target .nil) => (input, target)
        let output ←
          Runtime.Autograd.Torch.CurriedRef.uncurry
            (Ref := fun s => TorchLean.Runtime.ValueRef
              (m := m) (α := α) s)
            (ss := Runtime.Autograd.Model.Layers.Seq.stateShapes model ++
              [σ])
            (β := m (TorchLean.Runtime.ValueRef
              (m := m) (α := α) τ))
            (Runtime.Autograd.Model.Layers.Seq.forward model (α := α))
            (Runtime.Autograd.Torch.RefList.append state (.cons input .nil))
        loss.forward (α := α) (m := m) output target)

/-- Reject an invalid model or loss before lowering an autograd program. -/
def Internal.validateLoss {σ τ υ : Shape}
    (model : nn.Sequential σ τ) (loss : Loss τ υ) : IO Unit := do
  match nn.validate model with
  | .ok () => pure ()
  | .error message => throw <| IO.userError message
  match loss.validate with
  | .ok () => pure ()
  | .error message => throw <| IO.userError message

/--
Differentiate a model loss with respect to every tensor in the model state.

The result has the same shape-indexed layout as `State model α`. Set `value := true` to return
`(grad, lossValue)` from the same evaluation. For models with persistent buffers, this
computes mathematical sensitivities for those entries as well; `nn.requiresGrad` separately
controls which state tensors an optimizer updates.
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
        Internal.validateLoss model loss
        let (grad, _) ← Runtime.Autograd.Model.Autodiff.gradients
          (α := α)
          (paramShapes := Runtime.Autograd.Model.Layers.Seq.stateShapes model)
          (inputShapes := [σ, υ])
          (Internal.lossProgram model loss)
          (nn.State.Internal.toTensorPack state)
          (TorchLean.TensorPack.pair input target)
        pure (nn.State.Internal.fromTensorPack grad)
  | true =>
      exact do
        Internal.validateLoss model loss
        let stateShapes := Runtime.Autograd.Model.Layers.Seq.stateShapes model
        let graph ←
          Runtime.Autograd.Model.Autodiff.lowerScalarToTypedGraph (α := α)
            (paramShapes := stateShapes)
            (inputShapes := [σ, υ])
            (Internal.lossProgram model loss)

        let arguments : TorchLean.TensorPack α
            (stateShapes ++ [σ, υ]) :=
          TorchLean.TensorPack.append (ss₁ := stateShapes)
            (ss₂ := [σ, υ])
            (nn.State.Internal.toTensorPack state)
            (TorchLean.TensorPack.pair input target)

        let (allGradients, lossValue) ←
          Runtime.Autograd.Model.Autodiff.Impl.vjpWithValue
            graph arguments (Tensor.scalar (1 : α))

        let (grad, _) :=
          TorchLean.TensorPack.split (α := α) (ss₁ := stateShapes)
            (ss₂ := [σ, υ]) allGradients
        pure (nn.State.Internal.fromTensorPack grad, lossValue)

/--
Vector-Jacobian product with respect to the model and its input.

The returned pair is `(stateGrad, inputGrad)`. Both values come from one reverse pass.
-/
def vjp {σ τ : Shape} (model : nn.Sequential σ τ)
    {α : Type} [TorchLean.Storage α] [Context α]
    (state : State model α)
    (input : Tensor α σ) (outputGradient : Tensor α τ) :
    IO (State model α × Tensor α σ) := do
  IO.ofExcept (nn.validate model)
  let (stateGrad, inputGrads) ← Runtime.Autograd.Model.Autodiff.vjp
    (α := α)
    (paramShapes := Runtime.Autograd.Model.Layers.Seq.stateShapes model)
    (inputShapes := [σ]) (τ := τ)
    (fun {β} _ _ =>
      Runtime.Autograd.Model.Layers.Seq.forward model (α := β))
    (nn.State.Internal.toTensorPack state)
    (TorchLean.TensorPack.singleton input) outputGradient
  pure
    (nn.State.Internal.fromTensorPack stateGrad,
      TorchLean.TensorPack.head inputGrads)

/--
Reverse-mode Jacobian (`jacrev`) of the model output with respect to model state.

Returns one Jacobian tensor per state tensor. Each has the output axes followed by that state
tensor's axes, so parameter shapes stay distinct without an outer array of gradient states.
-/
def jacrev {σ τ : Shape}
    (model : nn.Sequential σ τ)
    {α : Type} [TorchLean.Storage α] [Context α]
    (state : State model α)
    (input : Tensor α σ) :
    IO (nn.State α ((nn.stateShapes model).map τ.concat)) := do
  IO.ofExcept (nn.validate model)
  let rows ← Runtime.Autograd.Model.Autodiff.jacrevOutParams
    (α := α)
    (paramShapes := Runtime.Autograd.Model.Layers.Seq.stateShapes model)
    (inputShapes := [σ]) (τ := τ)
    (fun {β} _ _ =>
      Runtime.Autograd.Model.Layers.Seq.forward model (α := β))
    (nn.State.Internal.toTensorPack state)
    (TorchLean.TensorPack.singleton input)
  pure (nn.State.Internal.fromTensorPack rows)

/--
Jacobian-vector product (JVP) of a scalar loss with respect to model state.

Directional derivative in the direction `stateDirection`. Conceptually:

$$
\left.\frac{d}{dt}
\operatorname{loss}(\mathrm{state}+t\,\mathrm{stateDirection},x,\mathrm{target})
\right|_{t=0}.
$$
-/
def jvp {σ τ υ : Shape}
    (model : nn.Sequential σ τ) (loss : Loss τ υ)
    {α : Type} [TorchLean.Storage α] [Context α]
    (state : State model α)
    (input : Tensor α σ) (target : Tensor α υ)
    (stateDirection : State model α) :
    IO (Tensor α []) := do
  Internal.validateLoss model loss
  Runtime.Autograd.Model.Autodiff.jvpLossParams
    (α := α)
    (paramShapes := Runtime.Autograd.Model.Layers.Seq.stateShapes model)
    (inputShapes := [σ, υ])
    (Internal.lossProgram model loss)
    (nn.State.Internal.toTensorPack state)
    (TorchLean.TensorPack.pair input target)
    (nn.State.Internal.toTensorPack stateDirection)

/--
Hessian-vector product (HVP) of a scalar loss with respect to model state.

Returns model state with the same shape layout as `state`.
-/
def hvp {σ τ υ : Shape}
    (model : nn.Sequential σ τ) (loss : Loss τ υ)
    {α : Type} [TorchLean.Storage α] [Context α]
    (state : State model α)
    (input : Tensor α σ) (target : Tensor α υ)
    (stateDirection : State model α) :
    IO (State model α) := do
  Internal.validateLoss model loss
  let result ← Runtime.Autograd.Model.Autodiff.hvpParams
    (α := α)
    (paramShapes := Runtime.Autograd.Model.Layers.Seq.stateShapes model)
    (inputShapes := [σ, υ])
    (Internal.lossProgram model loss)
    (nn.State.Internal.toTensorPack state)
    (TorchLean.TensorPack.pair input target)
    (nn.State.Internal.toTensorPack stateDirection)
  pure (nn.State.Internal.fromTensorPack result)
end model

end autograd

end TorchLean

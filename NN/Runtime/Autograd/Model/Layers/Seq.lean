/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Layers.Core
public import NN.Runtime.Autograd.Model.Loss
public import NN.Runtime.Autograd.Model.Module.Objective

/-!
# TorchLean NN: Sequential Models
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Model

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Proofs.Autograd.Algebra

namespace Layers

/-! ## Sequential models -/

/--
Sequential composition of `Layer`s, indexed by input/output shape.

This is the builder-layer analogue of `torch.nn.Sequential`: a `Seq σ τ` represents a model that
takes an input of shape `σ` and produces an output of shape `τ` by running layers left-to-right.

`Seq` lives in `Type 1`, and no lower. Each `Layer` stores its forward pass as an
execution-polymorphic `Program`, which quantifies over the scalar type `α : Type` and the
interpreter monad `m : Type → Type`, so `Layer σ τ : Type 1` and any type that stores a `Layer`
is at least `Type 1`. Consequently `IO (Seq σ τ)` is ill-typed; build models purely with
`nn.build seed builder`, draw a seed in `IO` with `nn.buildIO`, or pass the model to a
continuation with `nn.withModel`.
-/
inductive Seq : Shape → Shape → Type 1 where
  /-- The empty sequence, which leaves a tensor unchanged. -/
  | id (s : Shape) : Seq s s
  /-- Run one layer, then the remaining sequence. -/
  | cons {σ τ υ : Shape} : Layer σ τ → Seq τ υ → Seq σ υ

namespace Seq

/-- Lift one layer into a sequential model. -/
def fromLayer {σ τ : Shape} (layer : Layer σ τ) : Seq σ τ :=
  .cons layer (.id τ)

/--
Collect the parameter and persistent-buffer shapes owned by a sequential model.

This concatenates each layer's `stateShapes` in order.
-/
def stateShapes : {σ τ : Shape} → Seq σ τ → List Shape
  | _, _, .id _ => []
  | _, _, .cons l rest => l.stateShapes ++ stateShapes rest

/--
Collect the gradient flags for all parameters and buffers in a sequential model.

This concatenates each layer's `requiresGrad` in order. Persistent buffers carry `false`.
-/
def requiresGrad : {σ τ : Shape} → Seq σ τ → Array Bool
  | _, _, .id _ => #[]
  | _, _, .cons l rest => l.requiresGrad ++ requiresGrad rest

/-- Validate every layer's static value-level configuration. -/
def validate : {σ τ : Shape} → Seq σ τ → Except String Unit
  | _, _, .id _ => pure ()
  | _, _, .cons layer rest => do
      layer.validate
      validate rest

/--
Initial parameter and persistent-buffer values for a sequential model.

This concatenates each layer's `initState` into the flat state list expected by `forward` and
the supervised module constructors.
-/
def initState : {σ τ : Shape} → (m : Seq σ τ) → TorchLean.TensorPack Float (stateShapes m)
  | _, _, .id _ => .nil
  | _, _, .cons l rest =>
      let xs := l.initState
      let ys := initState rest
      TorchLean.TensorPack.append (α := Float)
        (ss₁ := l.stateShapes) (ss₂ := stateShapes rest) xs ys

/--
Collect a storage-first initializer plan when every parameterized layer supplies one.

Parameter-free layers need no annotation and contribute the empty plan. If any parameterized layer
has only tensor-valued initializers, the whole model falls back to the ordinary initialization path.
-/
def runtimeInit? : {σ τ : Shape} → (m : Seq σ τ) →
    Option (Runtime.Autograd.Model.Module.RuntimeInit.Plan (stateShapes m))
  | _, _, .id _ => some .nil
  | _, _, .cons l rest =>
      match l.runtimeInit, runtimeInit? rest with
      | some xs, some ys => some (Runtime.Autograd.Model.Module.RuntimeInit.Plan.append xs ys)
      | _, _ => none

/-- Whether any layer in the sequence owns mode-dependent mutable buffers. -/
def hasBufferUpdates : {σ τ : Shape} → Seq σ τ → Bool
  | _, _, .id _ => false
  | _, _, .cons l rest => l.updateBuffers.isSome || hasBufferUpdates rest

/--
Sequential composition for `Seq` models.

`comp f g` runs `f` then `g`. We also provide the infix `>>>` operator.
-/
def comp {σ τ υ : Shape} : Seq σ τ → Seq τ υ → Seq σ υ
  | .id _, g => g
  | .cons l rest, g => .cons l (comp rest g)

infixr:80 " >>> " => comp

/--
Internal evaluator that splits the flat model state as it walks the model.

This is the reference-level forward pass used to implement `forward`.
-/
def forwardState {σ τ : Shape} (model : Seq σ τ) {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Torch.Ops (m := m) (α := α)]
    (mode : Mode)
    (ps : Torch.RefList (RefTy (m := m) (α := α)) (stateShapes model))
    (x : RefTy (m := m) (α := α) σ) : m (RefTy (m := m) (α := α) τ) :=
  match model with
  | .id _ => pure x
  | .cons l rest =>
      let (psL, psR) :=
        Torch.RefList.split (Ref := RefTy (m := m) (α := α))
          (ss₁ := l.stateShapes) (ss₂ := stateShapes rest) ps
      do
        let y ← l.forwardRef (α := α) (m := m) mode psL x
        if mode == .train && !l.updatesBuffersInForward then
          if let some update := l.updateBuffers then
            if let some observe := Torch.Ops.updateBuffers? (m := m) (α := α) then
              observe psL x (update mode)
        forwardState (model := rest) (α := α) (m := m) mode psR y

/--
The differentiable forward computation of a sequential model.

The result is operation-polymorphic: eager execution records an autograd tape, while typed-graph
execution records shape-indexed SSA data. `mode` controls layers such as dropout and batch
normalization; it does not enable or disable gradient tracking.
-/
def forward {σ τ : Shape} (model : Seq σ τ) (mode : Mode := .eval)
    {α : Type} [TorchLean.Storage α] [Context α] :
    Runtime.Autograd.Model.Program α (stateShapes model ++ [σ]) τ :=
  fun {m} _ _ =>
    Torch.CurriedRef.curry (Ref := RefTy (m := m) (α := α))
      (ss := stateShapes model ++ [σ]) (β := m (RefTy (m := m) (α := α) τ)) (fun args => do
        let (ps, x) := Torch.RefList.splitLast (Ref := RefTy (m := m) (α := α)) (ss := stateShapes
          model) (τ := σ) args
        forwardState (model := model) (α := α) (m := m) mode ps x)

  /-!
  ## Forward and inference helpers

  `Mode.train` and `Mode.eval` choose how layers such as dropout and BatchNorm behave.
  `forwardNoGrad` takes live parameters and a concrete input, runs eagerly without recording
  gradients, and returns the output tensor. `predict` selects evaluation mode for that same
  operation. Decoding and sampling loops can use these helpers to inspect logits directly.

  For repeated graph execution, call `lowerToTypedGraph` once and evaluate the result with
  `TypedGraph.forward`. The recorded graph keeps the layer mode selected during lowering.
  -/

  /--
  Run an eager forward pass for one concrete input under an explicit mode.

  This uses the eager runtime so CUDA kernels stay available, reads back the concrete output, and
  then releases ephemeral CUDA tape buffers because no backward pass will follow. Use this for
  validation, decoding, diffusion sampling, and other inference loops.
  -/
  def forwardNoGrad {σ τ : Shape}
      (options : Runtime.Autograd.Torch.Config)
      (model : Seq σ τ)
      {α : Type} [TorchLean.Storage α] [Context α]
      [tensorTransfer : Runtime.Autograd.Torch.TensorTransfer α]
      (params : Runtime.Autograd.Torch.ParamList α (stateShapes model))
      (x : TorchLean.Tensor α σ) (mode : Mode := .eval)
      (rngCounter : Option (IO.Ref Nat) := none) : IO (TorchLean.Tensor α τ) := do
    match validate model with
    | .error message => throw <| IO.userError message
    | .ok () => pure ()
    -- Inference still uses the eager session machinery so it can select native kernels, but its
    -- leaves are deliberately non-differentiable and the transient tape is released before return.
    let options := { options with gradEnabled := false }
    let sess ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := α) options
    let sess := match rngCounter with
      | some counter => { sess with rngCounter := counter }
      | none => sess
    sess.resetTape
    try
      let outRef ← (do
        let pRefs ← Runtime.Autograd.Torch.Internal.useParams (α := α)
          (ss := stateShapes model) params
        let xRefs ← Runtime.Autograd.Torch.Internal.useInputs (α := α)
          (ss := [σ]) (.cons x .nil)
        let allRefs := Runtime.Autograd.Torch.RefList.append
          (ss₁ := stateShapes model) (ss₂ := [σ]) pRefs xRefs
        Runtime.Autograd.Torch.CurriedRef.uncurry
          (ss := stateShapes model ++ [σ])
          (forward model (mode := mode) (α := α)) allRefs) |>.run sess
      Runtime.Autograd.Torch.Internal.EagerSession.getValue (α := α) sess outRef
    finally
      -- The result has its own host storage. Retire the tape even if execution or readback throws,
      -- preserving shared parameter snapshots and reusable device blocks for the next call.
      sess.resetTape
      if options.usesCuda then
        Runtime.Autograd.Cuda.Buffer.collectGarbage

  /--
  Run eval-mode eager inference for one concrete input.

  This is the eval-mode convenience wrapper around `forwardNoGrad`.
  -/
  def predict {σ τ : Shape}
      (options : Runtime.Autograd.Torch.Config)
      (model : Seq σ τ)
      {α : Type} [TorchLean.Storage α] [Context α]
      [tensorTransfer : Runtime.Autograd.Torch.TensorTransfer α]
      (params : Runtime.Autograd.Torch.ParamList α (stateShapes model))
      (x : TorchLean.Tensor α σ) : IO (TorchLean.Tensor α τ) :=
    forwardNoGrad (α := α) (tensorTransfer := tensorTransfer) options model params x

  /--
  Lower a sequential model into a reusable `TypedGraph`.

  The model is recorded once as a typed SSA graph and can then be evaluated repeatedly.
  -/
  def lowerToTypedGraph {σ τ : Shape}
      (model : Seq σ τ)
      (mode : Mode := .eval)
      {α : Type} [TorchLean.Storage α] [Context α] :
      IO (Runtime.Autograd.Torch.TypedGraph α (stateShapes model ++ [σ]) τ) :=
    match validate model with
    | .error message => throw <| IO.userError message
    | .ok () =>
        Runtime.Autograd.Model.Autodiff.lowerToTypedGraph (α := α)
          (paramShapes := stateShapes model) (inputShapes := [σ]) (τ := τ)
          (fun {β} _ _ _ => forward model (mode := mode) (α := β))

  /--
  Update per-layer buffers across a sequential model.

This explicit reference replay walks the model left-to-right, updating each layer's state from
its reference activation. It does not observe a previous runtime execution or reproduce an eager
session's random stream. Live modules and trainers instead use the buffer hooks in their actual
forward pass.

PyTorch analogy: updating `running_mean` / `running_var` buffers during a forward pass in train
  mode.
-/
def updateBuffers {σ τ : Shape} (mode : Mode) (model : Seq σ τ)
    {α : Type} [TorchLean.Storage α] [Context α]
    (ps : TorchLean.TensorPack α (stateShapes model)) (x : Tensor α σ) :
    IO (TorchLean.TensorPack α (stateShapes model)) :=
  match model with
  | .id _ => pure .nil
  | .cons l rest => do
      let (psL, psR) :=
        TorchLean.TensorPack.split
          (α := α) (ss₁ := l.stateShapes) (ss₂ := stateShapes rest) ps
      let psL' ←
        match l.updateBuffers with
        | some f => f mode psL x
        | none => pure psL
      let y ← Layer.forwardTensor l mode psL' x
      let psR' ← updateBuffers mode rest psR y
      pure <| TorchLean.TensorPack.append
        (α := α) (ss₁ := l.stateShapes) (ss₂ := stateShapes rest) psL' psR'

/-! ## Scalar objectives -/

namespace Objective

/--
Pair an immutable sequential model with a scalar loss.

The resulting definition initializes the model's complete parameter-and-buffer state and computes
the loss from one `(input, target)` pair. Training mode is the default; pass
`mode := .eval` when evaluating a mode-sensitive model.
-/
def fromLoss {σ τ : Shape} (model : Seq σ τ)
    (loss : ∀ {α : Type}, [TorchLean.Storage α] → [Context α] →
      Runtime.Autograd.Model.Program α [τ, τ] [])
    (mode : Mode := .train) :
    Runtime.Autograd.Model.Module.ObjectiveDef Unit (stateShapes model) [σ, τ] :=
  { initState := initState model
    runtimeInit := runtimeInit? model
    requiresGrad := requiresGrad model
    validate := validate model
    loss := fun {α} => by
      intro _ _; exact
        (fun {m} _ _ =>
          Torch.CurriedRef.curry (Ref := RefTy (m := m) (α := α))
            (ss := stateShapes model ++ [σ, τ])
            (β := m (RefTy (m := m) (α := α) [])) (fun args => do
              let (ps, xy) :=
                Torch.RefList.split (Ref := RefTy (m := m) (α := α))
                  (ss₁ := stateShapes model) (ss₂ := [σ, τ]) args
              let .cons x (.cons y .nil) := xy
              let yhat ← forwardState (model := model) (α := α) (m := m) mode ps x
              Torch.CurriedRef.uncurry (Ref := RefTy (m := m) (α := α)) (ss := [τ, τ])
                (loss (α := α) (m := m)) (.cons yhat (.cons y .nil))
          ))
  }

/-- Pair a model with mean-squared error. -/
def mse {σ τ : Shape} (model : Seq σ τ)
    (reduction : TorchLean.Loss.Reduction :=
  .mean) (mode : Mode := .train) :
    Runtime.Autograd.Model.Module.ObjectiveDef Unit (stateShapes model) [σ, τ] :=
  fromLoss (model := model) (mode := mode) (loss := fun {α} _ _ =>
    fun {m} _ _ =>
      fun yhat y => TorchLean.Loss.mse (m := m) (α := α) (s := τ) yhat y
        (reduction := reduction))

/-- Pair a model with one-hot cross entropy. -/
def oneHotCrossEntropy {σ τ : Shape} (model : Seq σ τ)
    (axis : Nat)
    (reduction : TorchLean.Loss.Reduction := .mean)
    (mode : Mode := .train) :
    Runtime.Autograd.Model.Module.ObjectiveDef Unit (stateShapes model) [σ, τ] :=
  if axisInBounds : axis < τ.rank then
    letI : Shape.AxisInBounds axis τ :=
      Shape.AxisInBounds.ofRank axisInBounds
    fromLoss (model := model) (mode := mode) (loss := fun {α} _ _ =>
      fun {m} _ _ =>
        fun logits targetOneHot =>
          TorchLean.Loss.oneHotCrossEntropy (m := m) (α := α) (s := τ) axis
            logits targetOneHot
            (reduction := reduction))
  else
    let fallback :=
      fromLoss (model := model) (mode := mode) (loss := fun {α} _ _ =>
        fun {m} _ _ =>
          fun _ _ =>
            Torch.const (m := m) (α := α) (Tensor.zeros []))
    { fallback with
      validate := do
        fallback.validate
        throw s!"OneHotCrossEntropy: axis {axis} is out of bounds for rank {τ.rank}" }

end Objective

end Seq
end Layers

end Model
end Autograd
end Runtime

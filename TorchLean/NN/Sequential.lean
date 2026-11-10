/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Layer
public import TorchLean.NN.Ops.Pure
public import TorchLean.NN.Loss
public import TorchLean.NN.Module.Objective

/-!
# TorchLean NN: Sequential Models

`TorchLean.nn.Sequential σ τ` runs `Layer`s left to right (PyTorch analogy: `torch.nn.Sequential`),
and `f >>> g` composes two models. `nn.Objective.fromLoss`, `nn.Objective.mse` and
`nn.Objective.oneHotCrossEntropy` pair a model with a scalar loss as a
`TorchLean.ObjectiveDef`, which the runtime training code executes.
-/

@[expose] public section

namespace TorchLean.nn

open TorchLean.Tensor
open Ops

/-! ## Sequential models -/

/--
`Sequential σ τ` runs `Layer`s left to right, from an input of shape `σ` to an output of shape `τ`
(PyTorch analogy: `torch.nn.Sequential`). It lives in `Type 1`, since a `Layer` stores a `Program`
quantified over the scalar type and the interpreter monad, so `IO (Sequential σ τ)` is ill-typed.
Models come from `nn.build seed builder`, or from the `model` of `nn.buildIO`, which draws a seed
in `IO`.
-/
inductive Sequential : Shape → Shape → Type 1 where
  /-- The empty sequence, which leaves a tensor unchanged. -/
  | id (s : Shape) : Sequential s s
  /-- Run one layer, then the remaining sequence. -/
  | cons {σ τ υ : Shape} : Layer σ τ → Sequential τ υ → Sequential σ υ

namespace Sequential

/-- Lift one layer into a sequential model. -/
def fromLayer {σ τ : Shape} (layer : Layer σ τ) : Sequential σ τ :=
  .cons layer (.id τ)

/--
Collect the parameter and persistent-buffer shapes owned by a sequential model, by concatenating
each layer's `stateShapes` in order.
-/
def stateShapes : {σ τ : Shape} → Sequential σ τ → List Shape
  | _, _, .id _ => []
  | _, _, .cons l rest => l.stateShapes ++ stateShapes rest

/-- Append each layer's gradient flags in sequential state order. -/
def requiresGradFlags : {σ τ : Shape} → Sequential σ τ → Array Bool → Array Bool
  | _, _, .id _, flags => flags
  | _, _, .cons layer rest, flags => requiresGradFlags rest (flags ++ layer.requiresGrad)

/--
Collect the gradient flags for all parameters and buffers in layer order.

Persistent buffers carry `false`. The accumulator appends each layer's flags once.
-/
def requiresGrad {σ τ : Shape} (model : Sequential σ τ) : Array Bool :=
  requiresGradFlags model #[]

/-- Validate every layer's static value-level configuration. -/
def validate : {σ τ : Shape} → Sequential σ τ → Except String Unit
  | _, _, .id _ => pure ()
  | _, _, .cons layer rest => do
      layer.validate
      validate rest

/--
Initial parameter and persistent-buffer values for a sequential model.

Each layer's `initState` is appended in order, which gives the flat state list that `forward` and
the supervised module constructors expect.
-/
def initState : {σ τ : Shape} → (m : Sequential σ τ) → TorchLean.TensorPack Float (stateShapes m)
  | _, _, .id _ => .nil
  | _, _, .cons l rest =>
      let xs := l.initState
      let ys := initState rest
      TorchLean.Pack.append
        (ss₁ := l.stateShapes) (ss₂ := stateShapes rest) xs ys

/--
Collect a storage-first initializer plan when every parameterized layer supplies one.

Parameter-free layers need no annotation and contribute the empty plan. If any parameterized layer
has only tensor-valued initializers, the whole model falls back to the ordinary initialization path.
-/
def runtimeInit? : {σ τ : Shape} → (m : Sequential σ τ) →
    Option (Init.Plan (stateShapes m))
  | _, _, .id _ => some .nil
  | _, _, .cons l rest =>
      match l.runtimeInit, runtimeInit? rest with
      | some xs, some ys => some (xs.append ys)
      | _, _ => none

/-- Whether any layer in the sequence updates mutable buffers through an `updateBuffers` hook. -/
def hasBufferUpdates : {σ τ : Shape} → Sequential σ τ → Bool
  | _, _, .id _ => false
  | _, _, .cons l rest => l.updateBuffers.isSome || hasBufferUpdates rest

/--
`comp f g` runs `f`, then `g`. The infix `f >>> g` comes with `open scoped TorchLean.nn.Sequential`,
scoped so that it does not overload Lean's shift `>>>` elsewhere.
-/
def comp {σ τ υ : Shape} : Sequential σ τ → Sequential τ υ → Sequential σ υ
  | .id _, g => g
  | .cons l rest, g => .cons l (comp rest g)

@[inherit_doc comp] scoped infixr:80 " >>> " => comp

/--
Internal evaluator that splits the flat model state as it walks the model. It is the
reference-level forward pass behind `forward`.

In `Mode.train`, a layer that has an `updateBuffers` hook and leaves `updatesBuffersInForward`
false passes that hook, with its state and its input, to the interpreter's
`Ops.updateBuffers?` observer when the interpreter provides one.
-/
def forwardState {σ τ : Shape} (model : Sequential σ τ) {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    (mode : Mode)
    (ps : Pack (Ref (m := m) (α := α)) (stateShapes model))
    (x : Ref (m := m) (α := α) σ) : m (Ref (m := m) (α := α) τ) :=
  match model with
  | .id _ => pure x
  | .cons l rest =>
      let (psL, psR) :=
        Pack.split (F := Ref (m := m) (α := α))
          (ss₁ := l.stateShapes) (ss₂ := stateShapes rest) ps
      do
        let y ← l.forwardRef (α := α) (m := m) mode psL x
        if mode == .train && !l.updatesBuffersInForward then
          if let some update := l.updateBuffers then
            if let some observe := Ops.updateBuffers? (m := m) (α := α) then
              observe psL x (update mode)
        forwardState (model := rest) (α := α) (m := m) mode psR y

/--
The differentiable forward computation of a sequential model.

The result is operation-polymorphic: eager execution records an autograd tape, while typed-graph
execution records shape-indexed SSA data. `mode` selects the behavior of layers such as dropout and
batch normalization, and it does not switch gradient tracking on or off.
-/
def forward {σ τ : Shape} (model : Sequential σ τ) (mode : Mode := .eval)
    {α : Type} [TorchLean.Storage α] [Context α] :
    Program α (stateShapes model ++ [σ]) τ :=
  fun {m} _ _ =>
    CurriedRef.curry (Ref := Ref (m := m) (α := α))
      (ss := stateShapes model ++ [σ]) (β := m (Ref (m := m) (α := α) τ)) (fun args => do
        let (ps, x) := Pack.unsnoc (F := Ref (m := m) (α := α)) (ss := stateShapes model)
          (τ := σ) args
        forwardState (model := model) (α := α) (m := m) mode ps x)

/--
The evaluation-mode meaning of `model` at any scalar, with `params` the values of all of its state
(parameters and buffers): `forwardState` under the total pure interpreter `Ops Id α`.
-/
def eval {σ τ : Shape} (model : Sequential σ τ) {α : Type} [TorchLean.Storage α] [Context α]
    (params : TorchLean.TensorPack α (stateShapes model)) (x : Tensor α σ) : Tensor α τ :=
  forwardState (m := Id) model .eval params x

  /-!
  ## Forward and inference helpers

  `Mode.train` and `Mode.eval` choose how layers such as dropout and BatchNorm behave.
  `forwardNoGrad` takes live parameters and a concrete input, runs eagerly without recording
  gradients, and returns the output tensor. Evaluation mode is the default. Decoding and sampling
  loops can use it to read their logits.

  For repeated graph execution, call `lowerToTypedGraph` once and evaluate the result with
  `nn.TypedGraphModel.forward`. The recorded graph keeps the layer mode selected during lowering.
  -/

  /--
  Run an eager forward pass for one concrete input under the given mode.

  The pass validates `model`, runs on the eager runtime so that CUDA kernels stay available, and
  reads back the concrete output. No backward pass follows, so it then releases the tape and, on
  CUDA, the ephemeral device buffers. Use it in validation, decoding, diffusion sampling and other
  inference loops.
  -/
  def forwardNoGrad {σ τ : Shape}
      (options : Session.Config)
      (model : Sequential σ τ)
      {α : Type} [TorchLean.Storage α] [Context α]
      [tensorTransfer : TensorTransfer α]
      (params : ParamList α (stateShapes model))
      (x : TorchLean.Tensor α σ) (mode : Mode := .eval)
      (rngCounter : Option (IO.Ref Nat) := none) : IO (TorchLean.Tensor α τ) := do
    IO.ofExcept (validate model)
    -- Inference runs on the eager session machinery so that it can select native kernels. Its
    -- leaves are non-differentiable, and the transient tape is released before the call returns.
    let sess ← Session.create (α := α) options
      (gradEnabled := false)
    let sess := match rngCounter with
      | some counter => { sess with rngCounter := counter }
      | none => sess
    sess.resetTape
    try
      let outRef ← (do
        let pRefs ← ParamList.use (α := α)
          (ss := stateShapes model) params
        let xRefs ← Session.useInputs (α := α)
          (ss := [σ]) (.cons x .nil)
        let allRefs := Pack.append
          (ss₁ := stateShapes model) (ss₂ := [σ]) pRefs xRefs
        CurriedRef.uncurry
          (ss := stateShapes model ++ [σ])
          (forward model (mode := mode) (α := α)) allRefs) |>.run sess
      Session.getValue (α := α) sess outRef
    finally
      -- The result has its own host storage. Retire the tape even if execution or readback throws,
      -- preserving shared parameter snapshots and reusable device blocks for the next call.
      sess.resetTape
      if sess.options.usesCUDA then
        LibTorch.Buffer.collectGarbage

  /--
  Lower a sequential model into a reusable `TypedGraph`.

  The model is recorded once as a typed SSA graph and can then be evaluated repeatedly.
  -/
  def lowerToTypedGraph {σ τ : Shape}
      (model : Sequential σ τ)
      (mode : Mode := .eval)
      {α : Type} [TorchLean.Storage α] [Context α] :
      IO (TypedGraph α (stateShapes model ++ [σ]) τ) :=
    match validate model with
    | .error message => throw <| IO.userError message
    | .ok () =>
        Autodiff.lowerToTypedGraph (α := α)
          (paramShapes := stateShapes model) (inputShapes := [σ]) (τ := τ)
          (fun {β} _ _ _ => forward model (mode := mode) (α := β))

  /--
Update per-layer buffers across a sequential model.

A reference replay walks the model left to right. Each layer with an `updateBuffers` hook updates
its state from its input in the replay, and `nn.Layer.forwardTensor` under the updated state gives
the next layer's input. The replay does not observe a previous runtime execution or reproduce an
eager session's random stream. Live modules and trainers update buffers through the hooks of their
own forward pass.

PyTorch analogy: updating `running_mean` / `running_var` buffers during a forward pass in train
mode.
-/
def updateBuffers {σ τ : Shape} (mode : Mode) (model : Sequential σ τ)
    {α : Type} [TorchLean.Storage α] [Context α]
    (ps : TorchLean.TensorPack α (stateShapes model)) (x : Tensor α σ) :
    IO (TorchLean.TensorPack α (stateShapes model)) :=
  match model with
  | .id _ => pure .nil
  | .cons l rest => do
      let (psL, psR) :=
        TorchLean.Pack.split
          (ss₁ := l.stateShapes) (ss₂ := stateShapes rest) ps
      let psL' ←
        match l.updateBuffers with
        | some f => f mode psL x
        | none => pure psL
      let y ← Layer.forwardTensor l mode psL' x
      let psR' ← updateBuffers mode rest psR y
      pure <| TorchLean.Pack.append
        (ss₁ := l.stateShapes) (ss₂ := stateShapes rest) psL' psR'

end Sequential

/-! ## Scalar objectives -/

namespace Objective

open Sequential

/--
Pair an immutable sequential model with a scalar loss.

The resulting definition initializes the model's complete parameter-and-buffer state and computes
the loss from one `(input, target)` pair. Training mode is the default, so pass
`mode := nn.Mode.eval` when evaluating a mode-sensitive model.
-/
def fromLoss {σ τ : Shape} (model : Sequential σ τ)
    (loss : ∀ {α : Type}, [TorchLean.Storage α] → [Context α] →
      Program α [τ, τ] [])
    (mode : Mode := .train) :
    ObjectiveDef Unit (stateShapes model) [σ, τ] :=
  { initState := initState model
    runtimeInit := runtimeInit? model
    requiresGrad := requiresGrad model
    validate := validate model
    loss := fun {α} => by
      intro _ _; exact
        (fun {m} _ _ =>
          CurriedRef.curry (Ref := Ref (m := m) (α := α))
            (ss := stateShapes model ++ [σ, τ])
            (β := m (Ref (m := m) (α := α) [])) (fun args => do
              let (ps, xy) :=
                Pack.split (F := Ref (m := m) (α := α))
                  (ss₁ := stateShapes model) (ss₂ := [σ, τ]) args
              let .cons x (.cons y .nil) := xy
              let yhat ← forwardState (model := model) (α := α) (m := m) mode ps x
              CurriedRef.uncurry (Ref := Ref (m := m) (α := α)) (ss := [τ, τ])
                (loss (α := α) (m := m)) (.cons yhat (.cons y .nil))
          ))
  }

/-- Pair a model with mean-squared error. -/
def mse {σ τ : Shape} (model : Sequential σ τ)
    (reduction : TorchLean.Loss.Reduction :=
  .mean) (mode : Mode := .train) :
    ObjectiveDef Unit (stateShapes model) [σ, τ] :=
  fromLoss (model := model) (mode := mode) (loss := fun {α} _ _ =>
    fun {m} _ _ =>
      fun yhat y => TorchLean.Loss.mse (m := m) (α := α) (s := τ) yhat y
        (reduction := reduction))

/-- Pair a model with one-hot cross entropy along `axis`. An `axis` outside the rank of `τ` gives an
objective whose `validate` fails. -/
def oneHotCrossEntropy {σ τ : Shape} (model : Sequential σ τ)
    (axis : Nat)
    (reduction : TorchLean.Loss.Reduction := .mean)
    (mode : Mode := .train) :
    ObjectiveDef Unit (stateShapes model) [σ, τ] :=
  if axisInBounds : axis < τ.rank then
    letI : Shape.AxisInBounds axis τ :=
      Shape.AxisInBounds.mk axisInBounds
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
            Ops.const (m := m) (α := α) (Tensor.zeros []))
    { fallback with
      validate := do
        fallback.validate
        throw s!"OneHotCrossEntropy: axis {axis} is out of bounds for rank {τ.rank}" }

end Objective

end TorchLean.nn

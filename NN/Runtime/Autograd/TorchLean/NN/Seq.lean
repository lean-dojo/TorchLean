/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.NN.Core

/-!
# TorchLean NN: Sequential Models
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace TorchLean

open Spec
open Tensor
open Proofs.Autograd.Algebra

namespace NN

/-! ## Sequential models -/

/--
Sequential composition of `LayerDef`s, indexed by input/output shape.

This is the builder-layer analogue of `torch.nn.Sequential`: a `Seq σ τ` represents a model that
takes an input of shape `σ` and produces an output of shape `τ` by running layers left-to-right.
-/
inductive Seq : Shape → Shape → Type 2 where
  | id (s : Shape) : Seq s s
  | cons {σ τ υ : Shape} : LayerDef σ τ → Seq τ υ → Seq σ υ

namespace Seq

/--
Collect the parameter shapes required by a sequential model.

This concatenates each layer’s `paramShapes` in order.
-/
def paramShapes : {σ τ : Shape} → Seq σ τ → List Shape
  | _, _, .id _ => []
  | _, _, .cons l rest => l.paramShapes ++ paramShapes rest

/--
Collect the `requires_grad` flags for all parameters in a sequential model.

This concatenates each layer’s `paramRequiresGrad` in order.
-/
def paramRequiresGrad : {σ τ : Shape} → Seq σ τ → List Bool
  | _, _, .id _ => []
  | _, _, .cons l rest => l.paramRequiresGrad ++ paramRequiresGrad rest

/--
Initial parameter values for a sequential model.

This concatenates each layer’s `initParams` into the flat parameter list expected by
`programWithMode` / `scalarModuleDefWithMode`.
-/
def initParams : {σ τ : Shape} → (m : Seq σ τ) → Torch.TList Float (paramShapes m)
  | _, _, .id _ => .nil
  | _, _, .cons l rest =>
      let xs := l.initParams
      let ys := initParams rest
      Torch.Proofs.Autograd.Algebra.TList.append (α := Float)
        (ss₁ := l.paramShapes) (ss₂ := paramShapes rest) xs ys

/--
Sequential composition for `Seq` models.

`comp f g` runs `f` then `g`. We also provide the infix `>>>` operator.
-/
def comp {σ τ υ : Shape} : Seq σ τ → Seq τ υ → Seq σ υ
  | .id _, g => g
  | .cons l rest, g => .cons l (comp rest g)

infixr:80 " >>> " => comp

/--
Backend reference type used while evaluating a sequential model.

This is the `Torch.Ops.Ref` type provided by the chosen runtime backend.
-/
abbrev RefT (m : Type → Type) (α : Type) [Context α] [DecidableEq Shape]
    [Torch.Ops (m := m) (α := α)] (s : Shape) : Type :=
  Torch.Ops.Ref (m := m) (α := α) s

/--
Internal evaluator that splits the flat parameter list as it walks the model.

This is the reference-level forward pass used to implement `programWithMode`.
-/
def evalParams {σ τ : Shape} (model : Seq σ τ) {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Torch.Ops (m := m) (α := α)]
    (mode : Mode)
    (ps : Torch.RefList (RefT (m := m) (α := α)) (paramShapes model))
    (x : RefT (m := m) (α := α) σ) : m (RefT (m := m) (α := α) τ) :=
  match model with
  | .id _ => pure x
  | .cons l rest =>
      let (psL, psR) :=
        Torch.RefList.split (Ref := RefT (m := m) (α := α))
          (ss₁ := l.paramShapes) (ss₂ := paramShapes rest) ps
      do
        let y ← l.eval (α := α) (m := m) mode psL x
        evalParams (model := rest) (α := α) (m := m) mode psR y

/-- Turn a sequential model into a backend-generic `Program` (forward pass only). -/
def programWithMode {σ τ : Shape} (mode : Mode) (model : Seq σ τ)
    {α : Type} [Context α] [DecidableEq Shape] :
    TorchLean.Program α (paramShapes model ++ [σ]) τ :=
  fun {m} _ _ =>
    Torch.CurriedRef.curry (Ref := RefT (m := m) (α := α))
      (ss := paramShapes model ++ [σ]) (β := m (RefT (m := m) (α := α) τ)) (fun args => do
        let (ps, x) := Torch.RefList.splitAppend1 (Ref := RefT (m := m) (α := α)) (ss := paramShapes
          model) (τ := σ) args
        evalParams (model := model) (α := α) (m := m) mode ps x)

  /-- Default inference/eval forward pass for a sequential model. -/
  def program {σ τ : Shape} (model : Seq σ τ) {α : Type} [Context α] [DecidableEq Shape] :
      TorchLean.Program α (paramShapes model ++ [σ]) τ :=
    programWithMode .eval model

  /-!
  ## Compiled inference helpers

  These helpers are used by runtime code that wants to:

  - compile a `Seq` once (`compileOut*`), and
  - run it repeatedly on concrete tensors (`predict1*`).

  The public API (`NN.API.nn`) re-exports these helpers for model code written in a PyTorch-like
  style.
  -/

  /-!
  ## Eager inference helpers

  These helpers run a `Seq` directly through the eager runtime, given a *live* `ParamList`.

  Why this exists: several runnable examples want to inspect logits (argmax decoding, probes,
  interactive loops) without re-implementing the `useParams/useInputs` boilerplate.

  Note: this is eager-only. If you want "compile once, run many", use `compileOut`
  + `predict1` instead.
  -/

  /--
  Run one inference/evaluation forward without keeping an autograd tape.

  PyTorch analogy:

  ```python
  model.eval()
  with torch.no_grad():
      y = model(x)
  ```

  This uses the eager runtime so CUDA fast kernels stay available, reads back the concrete output,
  and then releases ephemeral tape buffers because no backward pass will follow. Use this for
  validation, decoding, diffusion sampling, and other inference loops.
  -/
  def eval1NoGrad {σ τ : Shape}
      (opts : _root_.Runtime.Autograd.Torch.Options)
      (model : Seq σ τ)
      {α : Type} [Context α] [DecidableEq Shape]
      [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
      (params : _root_.Runtime.Autograd.Torch.ParamList α (paramShapes model))
      (x : Spec.Tensor α σ) : IO (Spec.Tensor α τ) := do
    let opts := { opts with noGrad := true }
    let sess ← _root_.Runtime.Autograd.Torch.Internal.EagerSession.new (α := α) opts
    sess.resetTape
    let outRef ← (do
      let pRefs ← _root_.Runtime.Autograd.Torch.Internal.useParams (α := α)
        (ss := paramShapes model) params
      let xRefs ← _root_.Runtime.Autograd.Torch.Internal.useInputs (α := α)
        (ss := [σ]) (.cons x .nil)
      let allRefs := _root_.Runtime.Autograd.Torch.RefList.append
        (ss₁ := paramShapes model) (ss₂ := [σ]) pRefs xRefs
      _root_.Runtime.Autograd.Torch.CurriedRef.uncurry
        (ss := paramShapes model ++ [σ])
        (programWithMode .eval (model := model) (α := α)) allRefs) |>.run sess
    let y ← _root_.Runtime.Autograd.Torch.Internal.EagerSession.getValue (α := α) sess outRef
    if opts.useGpu then
      _root_.Runtime.Autograd.Torch.Internal.EagerSession.releaseCudaTapeNonParamValues sess
      sess.cudaTape.set _root_.Runtime.Autograd.Cuda.Tape.empty
      sess.paramsByLeaf.set (Std.HashMap.emptyWithCapacity)
      sess.nats.set #[]
      _root_.Runtime.Autograd.Torch.Internal.EagerSession.collectCudaAllocator
    else
      pure ()
    pure y

  /-- Preferred single-input evaluation helper. This is an alias for `eval1NoGrad`. -/
  def eval1 {σ τ : Shape}
      (opts : _root_.Runtime.Autograd.Torch.Options)
      (model : Seq σ τ)
      {α : Type} [Context α] [DecidableEq Shape]
      [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
      (params : _root_.Runtime.Autograd.Torch.ParamList α (paramShapes model))
      (x : Spec.Tensor α σ) : IO (Spec.Tensor α τ) :=
    eval1NoGrad (α := α) opts model params x

  /--
  Compile a sequential model into a reusable `CompiledOut`.

  This is the "compile once, run many times" entrypoint for inference.
  -/
  def compileOutWithMode {σ τ : Shape}
      (mode : Mode)
      (model : Seq σ τ)
      {α : Type} [Context α] [DecidableEq Shape] :
      IO (_root_.Runtime.Autograd.Torch.CompiledOut α (paramShapes model ++ [σ]) τ) :=
    _root_.Runtime.Autograd.TorchLean.Autodiff.compileOut (α := α)
      (paramShapes := paramShapes model) (inputShapes := [σ]) (τ := τ)
      (fun {β} _ _ => programWithMode mode (model := model) (α := β))

  /--
  Compile a sequential model in evaluation mode (`Mode.eval`).
  -/
  def compileOut {σ τ : Shape} (model : Seq σ τ)
      {α : Type} [Context α] [DecidableEq Shape] :
      IO (_root_.Runtime.Autograd.Torch.CompiledOut α (paramShapes model ++ [σ]) τ) :=
    compileOutWithMode (α := α) .eval model

  /--
  Run a compiled sequential model on a single input tensor.

  This helper calls `CompiledOut.forward` and handles packing the
  argument list `params ++ [x]`.
  -/
  def predict1WithMode {σ τ : Shape}
      (_mode : Mode)
      (model : Seq σ τ)
      {α : Type} [Context α] [DecidableEq Shape]
      (compiled : _root_.Runtime.Autograd.Torch.CompiledOut α (paramShapes model ++ [σ]) τ)
      (params : _root_.Runtime.Autograd.Torch.TList α (paramShapes model))
      (x : Spec.Tensor α σ) : Spec.Tensor α τ :=
    let args : _root_.Runtime.Autograd.Torch.TList α (paramShapes model ++ [σ]) :=
      _root_.Runtime.Autograd.Torch.Proofs.Autograd.Algebra.TList.append
        (α := α) (ss₁ := paramShapes model) (ss₂ := [σ]) params (.cons x .nil)
    _root_.Runtime.Autograd.Torch.CompiledOut.forward compiled args

  /-- Run a compiled sequential model once in evaluation mode (`Mode.eval`). -/
  def predict1 {σ τ : Shape} (model : Seq σ τ)
      {α : Type} [Context α] [DecidableEq Shape]
      (compiled : _root_.Runtime.Autograd.Torch.CompiledOut α (paramShapes model ++ [σ]) τ)
      (params : _root_.Runtime.Autograd.Torch.TList α (paramShapes model))
      (x : Spec.Tensor α σ) : Spec.Tensor α τ :=
    predict1WithMode (α := α) .eval model compiled params x

  /--
  Run a sequential model once in evaluation mode without building an autograd training graph.

  PyTorch analogy:

  ```python
  model.eval()
  with torch.no_grad():
      y = model(x)
  ```

  This snapshots the live `ParamList`, synchronizing CUDA-resident parameter mirrors when needed,
  and evaluates the model through the compiled tensor-output path. Use this for validation,
  decoding, sampling, and probes where gradients are not needed.
  -/
  def eval1CompiledNoGrad {σ τ : Shape}
      (model : Seq σ τ)
      {α : Type} [Context α] [DecidableEq Shape]
      [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
      (params : _root_.Runtime.Autograd.Torch.ParamList α (paramShapes model))
      (x : Spec.Tensor α σ) : IO (Spec.Tensor α τ) := do
    let ps ← _root_.Runtime.Autograd.Torch.ParamList.valuesSynced (α := α)
      (ss := paramShapes model) params
    let compiled ← compileOutWithMode (α := α) .eval model
    pure <| predict1WithMode (α := α) .eval model compiled ps x

  /--
  Update per-layer buffers across a sequential model.

This walks the model left-to-right and, for each layer that defines `LayerDef.updateBuffers`,
updates that layer’s parameter/buffer slice using the current activation. This is used to implement
BatchNorm-style running statistics (and similar stateful layers) in a pure, explicit way.

PyTorch analogy: updating `running_mean` / `running_var` buffers during a forward pass in train
  mode.
-/
def updateBuffers {σ τ : Shape} (mode : Mode) (model : Seq σ τ)
    {α : Type} [Context α] [DecidableEq Shape]
    (ps : Torch.TList α (paramShapes model)) (x : Tensor α σ) :
    IO (Torch.TList α (paramShapes model)) := do
  match model with
  | .id _ => pure .nil
  | .cons l rest =>
      let (psL, psR) :=
        Torch.Proofs.Autograd.Algebra.TList.splitAppend
          (α := α) (ss₁ := l.paramShapes) (ss₂ := paramShapes rest) ps
      let psL' ←
        match l.updateBuffers with
        | some f => f mode psL x
        | none => pure psL
      let y ← LayerDef.evalTensor l mode psL' x
      let psR' ← updateBuffers mode rest psR y
      pure <| Torch.Proofs.Autograd.Algebra.TList.append
        (α := α) (ss₁ := l.paramShapes) (ss₂ := paramShapes rest) psL' psR'

/-! ## Build a runnable `ScalarModuleDef` -/

/--
Bundle a sequential model and a supervised loss into a `ScalarModuleDef`.

The resulting `ScalarModuleDef` can be handed to TorchLean’s runtime training code: it knows how to
initialize parameters and compute a scalar loss given `(x, y)` pairs.

PyTorch analogy: an `nn.Module` paired with a loss function, evaluated under `mode` (`train` vs
  `eval`).
-/
def scalarModuleDefWithMode {σ τ : Shape} (mode : Mode) (model : Seq σ τ)
    (loss : ∀ {α : Type}, [Context α] → [DecidableEq Shape] → TorchLean.Program α [τ, τ]
      Shape.scalar) :
    TorchLean.Module.ScalarModuleDef (paramShapes model) [σ, τ] :=
  { initParams := initParams model
    initRequiresGrad := paramRequiresGrad model
    loss := fun {α} => by
      intro _ _; exact
        (fun {m} _ _ =>
          Torch.CurriedRef.curry (Ref := RefT (m := m) (α := α))
            (ss := paramShapes model ++ [σ, τ])
            (β := m (RefT (m := m) (α := α) Shape.scalar)) (fun args => do
              let (ps, xy) :=
                Torch.RefList.split (Ref := RefT (m := m) (α := α))
                  (ss₁ := paramShapes model) (ss₂ := [σ, τ]) args
              let .cons x (.cons y .nil) := xy
              let yhat ← evalParams (model := model) (α := α) (m := m) mode ps x
              Torch.CurriedRef.uncurry (Ref := RefT (m := m) (α := α)) (ss := [τ, τ])
                (loss (α := α) (m := m)) (.cons yhat (.cons y .nil))
          ))
  }

/-- Training-mode scalar-loss wrapper. -/
def scalarModuleDef {σ τ : Shape} (model : Seq σ τ)
    (loss : ∀ {α : Type}, [Context α] → [DecidableEq Shape] → TorchLean.Program α [τ, τ]
      Shape.scalar) :
    TorchLean.Module.ScalarModuleDef (paramShapes model) [σ, τ] :=
  scalarModuleDefWithMode .train model loss

/-- Common supervised regression wrapper: `loss := Loss.mse` with a chosen reduction. -/
def mseScalarModuleDefWithMode {σ τ : Shape} (mode : Mode) (model : Seq σ τ)
    (reduction : TorchLean.Loss.Reduction := .mean) :
    TorchLean.Module.ScalarModuleDef (paramShapes model) [σ, τ] :=
  scalarModuleDefWithMode mode (model := model) (loss := fun {α} _ _ =>
    fun {m} _ _ =>
      fun yhat y => TorchLean.Loss.mse (m := m) (α := α) (s := τ) yhat y (reduction := reduction))

/-- Training-mode MSE wrapper. -/
def mseScalarModuleDef {σ τ : Shape} (model : Seq σ τ) (reduction : TorchLean.Loss.Reduction :=
  .mean) :
    TorchLean.Module.ScalarModuleDef (paramShapes model) [σ, τ] :=
  mseScalarModuleDefWithMode .train model reduction

/-- Common supervised classification wrapper: `loss := Loss.crossEntropyOneHot` with a chosen
  reduction. -/
def crossEntropyOneHotScalarModuleDefWithMode {σ τ : Shape} (mode : Mode) (model : Seq σ τ)
    (reduction : TorchLean.Loss.Reduction := .mean) :
    TorchLean.Module.ScalarModuleDef (paramShapes model) [σ, τ] :=
  scalarModuleDefWithMode mode (model := model) (loss := fun {α} _ _ =>
    fun {m} _ _ =>
      fun logits targetOneHot =>
        TorchLean.Loss.crossEntropyOneHot (m := m) (α := α) (s := τ) logits targetOneHot
          (reduction := reduction))

/-- Training-mode cross-entropy wrapper. -/
def crossEntropyOneHotScalarModuleDef {σ τ : Shape} (model : Seq σ τ)
    (reduction : TorchLean.Loss.Reduction := .mean) :
    TorchLean.Module.ScalarModuleDef (paramShapes model) [σ, τ] :=
  crossEntropyOneHotScalarModuleDefWithMode .train model reduction

end Seq
end NN

end TorchLean
end Autograd
end Runtime

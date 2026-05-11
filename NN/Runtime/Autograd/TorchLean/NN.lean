/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.Autodiff
public import NN.Runtime.Autograd.TorchLean.Loss
public import NN.Runtime.Autograd.TorchLean.Module
public import NN.Runtime.Autograd.TorchLean.Norm

import Mathlib.Algebra.Order.Algebra

/-!
# NN

`TorchLean.NN`: a compact `torch.nn`-style builder layer.

This module defines a small `torch.nn`-style builder layer for constructing shape-typed models.
It packages parameter shapes/initial values together with a backend-polymorphic forward program, so
example code does not have to spell `paramShapes := [...]` / `inputShapes := [...]` everywhere.

## Main definitions

- `LayerDef σ τ` packages a shape-typed layer with explicit parameters (shapes + initial values) and
  a polymorphic `forward` program.
- `Seq σ τ` composes layers sequentially (PyTorch analogy: `torch.nn.Sequential`), written `f >>>
  g`.
- `scalarModuleDef*` helpers bundle a `Seq` model together with a scalar loss, producing a
  `TorchLean.Module.ScalarModuleDef` that the runtime training code can execute.

## PyTorch analogies

- `LayerDef` is like a small `nn.Module` definition, except parameters are an explicit list instead
  of fields, and the forward pass is a typed TorchLean program.
- `Mode` is like `module.train()` vs `module.eval()` (dropout and batchnorm-like layers branch on
  it).
- The `updateBuffers` mechanism is like updating non-gradient buffers (e.g. BatchNorm running
  stats).

The surface here is intentionally narrow: it supports TorchLean's executable model constructors and
training helpers without trying to mirror the full `torch.nn` API.

## References

- PyTorch `torch.nn`: https://pytorch.org/docs/stable/nn.html
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace TorchLean

open Spec
open Tensor
open Proofs.Autograd.Algebra

namespace NN

/-!
### Mode

TorchLean keeps "train vs eval" behavior explicit. This affects layers like dropout and
batch-normalization that behave differently during training vs inference.
-/

/--
Execution mode for layers that branch between training-time and inference-time behavior.

PyTorch analogy: `model.train()` / `model.eval()` (affects dropout, batchnorm, etc.).
-/
inductive Mode where
  | train
  | eval
deriving Repr, DecidableEq

/-! ## Layer definitions -/

/--
A shape-typed layer definition with explicit parameters and a backend-polymorphic forward program.

`LayerDef σ τ` is the core building block used by `Seq` (sequential composition). It stores:
- a list of parameter shapes,
- initial values for those parameters/buffers (as `Float` tensors, for demo-friendly
  initialization),
- per-parameter `requires_grad` flags, and
- a `forward` program that is polymorphic over the backend monad and scalar type.

PyTorch analogy: a small `nn.Module`, where:
- `paramShapes`/`initParams` correspond to parameters (and possibly buffers),
- `forward` corresponds to `Module.forward`,
- `updateBuffers` corresponds to updating things like `running_mean`/`running_var` in BatchNorm.
-/
structure LayerDef (σ τ : Shape) where
  /-- Shapes of the layer's parameter tensors, in the order expected by `forward`. -/
  paramShapes : List Shape
  /-- Initial parameter values (stored as `Float` tensors for convenient seeding/init schemes). -/
  initParams : Torch.TList Float paramShapes
  /--
  Per-parameter `requires_grad` flags (defaults to all `true`).

  PyTorch analogy: `tensor.requires_grad_(...)` on parameters/buffers.
  -/
  paramRequiresGrad : List Bool := List.replicate paramShapes.length true
  /--
  Optional buffer update function (used for running-statistics style layers).

  This is called during a forward pass (typically in `Mode.train`) to produce updated
    parameter/buffer
  values. A canonical example is BatchNorm updating its `running_mean` / `running_var` buffers.
  -/
  updateBuffers :
    Option (
      Mode → ∀ {α : Type}, [Context α] → [DecidableEq Shape] →
        Torch.TList α paramShapes → Tensor α σ → IO (Torch.TList α paramShapes)
    ) := none
  /--
  Forward pass as a typed TorchLean program.

  The program expects `(paramShapes ++ [σ])` inputs (the parameters, then the layer input) and
  produces an output of shape `τ`.
  -/
  forward :
    Mode → ∀ {α : Type}, [Context α] → [DecidableEq Shape] →
      TorchLean.Program α (paramShapes ++ [σ]) τ

/--
Update rule for a running statistics vector using momentum.

This implements an exponential moving average:

`next = (1 - momentum) * running + momentum * batch`.

PyTorch analogy: the update performed for `running_mean` / `running_var` in BatchNorm.
-/
def updateRunningVec {α : Type} [Context α] {c : Nat}
    (running batch : Tensor α (.dim c .scalar)) (momentum : Tensor α Shape.scalar) :
    Tensor α (.dim c .scalar) :=
  match running, batch, momentum with
  | .dim runningF, .dim batchF, .scalar mom =>
      let keep : Tensor α Shape.scalar := Tensor.scalar ((1 : α) - mom)
      Tensor.dim (fun i =>
        addSpec
          (mulSpec (runningF i) keep)
          (mulSpec (batchF i) (Tensor.scalar mom)))

/--
Compute per-channel mean and variance for a CHW tensor (no batch dimension).

This reduces over the spatial axes `(H, W)` and returns `(mean, var)` vectors of length `channels`.

PyTorch analogy: the statistics used by `torch.nn.BatchNorm2d` in training mode (but here for an
unbatched `C×H×W` input).
-/
def chwBatchStats {α : Type} [Context α]
    {channels height width : Nat}
    (x : Tensor α (NN.Tensor.Shape.CHW channels height width)) :
    Tensor α (.dim channels .scalar) × Tensor α (.dim channels .scalar) :=
  let means : Tensor α (.dim channels .scalar) :=
    Tensor.dim (fun c =>
      let channelData := getAtSpec x c
      let channelSum :=
        (List.finRange height).foldl (fun accH i =>
          (List.finRange width).foldl (fun accW j =>
            if hI : i < height then
              if hJ : j < width then
                addSpec accW (getAtSpec (getAtSpec channelData ⟨i, hI⟩) ⟨j, hJ⟩)
              else accW
            else accW
          ) accH
        ) (Tensor.scalar 0)
      divSpec channelSum (Tensor.scalar ((height * width : Nat) : α)))
  let vars : Tensor α (.dim channels .scalar) :=
    Tensor.dim (fun c =>
      let channelData := getAtSpec x c
      let mean := getAtSpec means c
      let varianceSum :=
        (List.finRange height).foldl (fun accH i =>
          (List.finRange width).foldl (fun accW j =>
            if hI : i < height then
              if hJ : j < width then
                let v := getAtSpec (getAtSpec channelData ⟨i, hI⟩) ⟨j, hJ⟩
                let d := subSpec v mean
                addSpec accW (mulSpec d d)
              else accW
            else accW
          ) accH
        ) (Tensor.scalar 0)
      divSpec varianceSum (Tensor.scalar ((height * width : Nat) : α)))
  (means, vars)

/--
Compute per-channel mean and variance for an NCHW tensor.

This reduces over `(N, H, W)` and returns `(mean, var)` vectors of length `c`.

PyTorch analogy: the batch statistics computed by `torch.nn.BatchNorm2d` in training mode.
-/
def nchwBatchStats {α : Type} [Context α]
    {n c h w : Nat}
    (x : Tensor α (NN.Tensor.Shape.NCHW n c h w)) :
    Tensor α (.dim c .scalar) × Tensor α (.dim c .scalar) :=
  let means : Tensor α (.dim c .scalar) :=
    Tensor.dim (fun ch =>
      let total :=
        (List.finRange n).foldl (fun accN ni =>
          (List.finRange h).foldl (fun accH i =>
            (List.finRange w).foldl (fun accW j =>
              if hN : ni < n then
                if hI : i < h then
                  if hJ : j < w then
                    let sample := getAtSpec x ⟨ni, hN⟩
                    let channel := getAtSpec sample ch
                    addSpec accW (getAtSpec (getAtSpec channel ⟨i, hI⟩) ⟨j, hJ⟩)
                  else accW
                else accW
              else accW
            ) accH
          ) accN
        ) (Tensor.scalar 0)
      divSpec total (Tensor.scalar ((n * h * w : Nat) : α)))
  let vars : Tensor α (.dim c .scalar) :=
    Tensor.dim (fun ch =>
      let mean := getAtSpec means ch
      let total :=
        (List.finRange n).foldl (fun accN ni =>
          (List.finRange h).foldl (fun accH i =>
            (List.finRange w).foldl (fun accW j =>
              if hN : ni < n then
                if hI : i < h then
                  if hJ : j < w then
                    let sample := getAtSpec x ⟨ni, hN⟩
                    let channel := getAtSpec sample ch
                    let v := getAtSpec (getAtSpec channel ⟨i, hI⟩) ⟨j, hJ⟩
                    let d := subSpec v mean
                    addSpec accW (mulSpec d d)
                  else accW
                else accW
              else accW
            ) accH
          ) accN
        ) (Tensor.scalar 0)
      divSpec total (Tensor.scalar ((n * h * w : Nat) : α)))
  (means, vars)

namespace LayerDef

/--
Backend reference type used when evaluating a `LayerDef`.

This is the `Ref` type provided by the current `Torch.Ops` backend instance (eager tape, compiled
  IR,
etc.).
-/
abbrev RefT (m : Type → Type) (α : Type) [Context α] [DecidableEq Shape]
    [Torch.Ops (m := m) (α := α)] (s : Shape) : Type :=
  Torch.Ops.Ref (m := m) (α := α) s

/--
Evaluate a `LayerDef` given parameter refs and an input ref.

This is the "module forward" operation at the reference level.

PyTorch analogy: calling `layer(x)` where the layer's parameters are already allocated.
-/
def eval {σ τ : Shape} (l : LayerDef σ τ) {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Torch.Ops (m := m) (α := α)]
    (mode : Mode)
    (ps : Torch.RefList (RefT (m := m) (α := α)) l.paramShapes)
    (x : RefT (m := m) (α := α) σ) : m (RefT (m := m) (α := α) τ) :=
  Torch.CurriedRef.uncurry (ss := l.paramShapes ++ [σ]) (Ref := RefT (m := m) (α := α))
    (l.forward mode (α := α) (m := m)) (Torch.RefList.append ps (.cons x .nil))

/--
Evaluate a `LayerDef` on concrete tensors by compiling its forward program.

This is primarily used by runtime utilities (e.g. sequential `updateBuffers`) where we want to run
forward to obtain intermediate activations.

PyTorch analogy: running a forward pass eagerly on concrete tensors.
-/
def evalTensor {σ τ : Shape} (l : LayerDef σ τ) (mode : Mode)
    {α : Type} [Context α] [DecidableEq Shape]
    (ps : Torch.TList α l.paramShapes) (x : Tensor α σ) : IO (Tensor α τ) := do
  let compiled ← _root_.Runtime.Autograd.TorchLean.Autodiff.compileOut (α := α)
    (paramShapes := l.paramShapes) (inputShapes := [σ]) (τ := τ)
    (l.forward mode)
  let args : Torch.TList α (l.paramShapes ++ [σ]) :=
    Torch.Proofs.Autograd.Algebra.TList.append (α := α) (ss₁ := l.paramShapes) (ss₂ := [σ]) ps
      (.cons x .nil)
  pure <| _root_.Runtime.Autograd.Torch.CompiledOut.forward compiled args

end LayerDef

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

  These are small convenience wrappers used by runtime code that wants to:

  - compile a `Seq` once (`compileOut*`), and
  - run it repeatedly on concrete tensors (`predict1*`).

  The public API facade (`NN.API.nn`) re-exports these so examples can remain "PyTorch-like".
  -/

  /-!
  ## Eager inference helpers

  These helpers run a `Seq` directly through the eager runtime, given a *live* `ParamList`.

  Why this exists: several runnable examples want to inspect logits (argmax decoding, probes,
  interactive loops) without re-implementing the `useParams/useInputs` boilerplate.

  Note: this is intentionally eager-only. If you want "compile once, run many", use `compileOut`
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
  and then releases temporary tape buffers because no backward pass will follow. Use this for
  validation, decoding, diffusion sampling, and other inference loops.
  -/
  def eval1NoGrad {σ τ : Shape}
      (opts : _root_.Runtime.Autograd.Torch.Options)
      (model : Seq σ τ)
      {α : Type} [Context α] [DecidableEq Shape]
      [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
      (params : _root_.Runtime.Autograd.Torch.ParamList α (paramShapes model))
      (x : Spec.Tensor α σ) : IO (Spec.Tensor α τ) := do
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
  Compatibility alias. Prefer `eval1`.
  -/
  def predict1Eager {σ τ : Shape}
      (opts : _root_.Runtime.Autograd.Torch.Options)
      (model : Seq σ τ)
      {α : Type} [Context α] [DecidableEq Shape]
      [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
      (params : _root_.Runtime.Autograd.Torch.ParamList α (paramShapes model))
      (x : Spec.Tensor α σ) : IO (Spec.Tensor α τ) :=
    eval1NoGrad (α := α) opts model params x

  /--
  Compatibility alias. Prefer `eval1NoGrad` (or simply `eval1`).
  -/
  def predict1EagerNoGrad {σ τ : Shape}
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

  This is a small convenience wrapper around `CompiledOut.forward` that also handles packing the
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
  Compatibility alias for the compiled no-grad path. For CUDA inference loops, prefer `eval1`.
  -/
  def predict1NoGrad {σ τ : Shape}
      (model : Seq σ τ)
      {α : Type} [Context α] [DecidableEq Shape]
      [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
      (params : _root_.Runtime.Autograd.Torch.ParamList α (paramShapes model))
      (x : Spec.Tensor α σ) : IO (Spec.Tensor α τ) :=
    eval1CompiledNoGrad (α := α) model params x

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

/-! ## Convenience constructors (layers) -/

/--
Fully-connected affine layer on vectors: `y = W x + b`.

Parameters:
- `W : (outDim × inDim)` initialized with Xavier initialization,
- `b : (outDim)` initialized to zeros.

PyTorch analogy: `torch.nn.Linear(inDim, outDim)`.
-/
def linear (inDim outDim : Nat) (seedW seedB : Nat := 0) :
    LayerDef (NN.Tensor.Shape.Vec inDim) (NN.Tensor.Shape.Vec outDim) :=
  let WShape : Shape := NN.Tensor.Shape.Mat outDim inDim
  let bShape : Shape := NN.Tensor.Shape.Vec outDim
  let w0 : Tensor Float WShape := Torch.Init.xavierW (outDim := outDim) (inDim := inDim) (seed :=
    seedW)
  let b0 : Tensor Float bShape := Torch.Init.tensor (s := bShape) (sch := .zeros) (seed := seedB)
  { paramShapes := [WShape, bShape]
    initParams := Torch.tlist2 w0 b0
    paramRequiresGrad := [true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun w b x =>
          TorchLean.linear (m := m) (α := α) (inDim := inDim) (outDim := outDim) w b x
  }

/--
Batched / matrix-valued affine layer: `y = x @ Wᵀ + b`.

Input shape: `(batch × inDim)`. Output shape: `(batch × outDim)`.

PyTorch analogy: `torch.nn.Linear(inDim, outDim)` applied to a 2D tensor.
-/
def linear2d (batch inDim outDim : Nat) (seedW seedB : Nat := 0) :
    LayerDef (.dim batch (.dim inDim .scalar)) (.dim batch (.dim outDim .scalar)) :=
  let WShape : Shape := NN.Tensor.Shape.Mat outDim inDim
  let bShape : Shape := NN.Tensor.Shape.Vec outDim
  let w0 : Tensor Float WShape := Torch.Init.xavierW (outDim := outDim) (inDim := inDim) (seed :=
    seedW)
  let b0 : Tensor Float bShape := Torch.Init.tensor (s := bShape) (sch := .zeros) (seed := seedB)
  { paramShapes := [WShape, bShape]
    initParams := Torch.tlist2 w0 b0
    paramRequiresGrad := [true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun w b x =>
          TorchLean.linear2d (m := m) (α := α)
            (batch := batch) (inDim := inDim) (outDim := outDim)
            w b x
  }

/--
Vanilla RNN layer (time-major sequence, no batch axis).

Semantics:
`h_t = tanh(W [x_t; h_{t-1}] + b)`, with `h_{-1} = 0`.

This is implemented by unrolling a fixed number of steps (`seqLen`) using existing TorchLean ops,
so it works on both CPU and CUDA backends.

PyTorch analogy: `torch.nn.RNN(inputSize, hiddenSize, nonlinearity="tanh")` with
`batch_first=false`, specialized to a single batch element.
Docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html
-/
def rnn (seqLen inputSize hiddenSize : Nat) (seedW seedB : Nat := 0) :
    LayerDef (.dim seqLen (.dim inputSize .scalar)) (.dim seqLen (.dim hiddenSize .scalar)) :=
  let WShape : Shape := NN.Tensor.Shape.Mat hiddenSize (inputSize + hiddenSize)
  let bShape : Shape := NN.Tensor.Shape.Vec hiddenSize
  let w0 : Tensor Float WShape := Torch.Init.xavierW (outDim := hiddenSize) (inDim := inputSize +
    hiddenSize) (seed := seedW)
  let b0 : Tensor Float bShape := Torch.Init.tensor (s := bShape) (sch := .zeros) (seed := seedB)
  { paramShapes := [WShape, bShape]
    initParams := Torch.tlist2 w0 b0
    paramRequiresGrad := [true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun w b xs => show m (Ref (.dim seqLen (.dim hiddenSize .scalar))) from do
          let h0T : Tensor α (.dim hiddenSize .scalar) :=
            Spec.fill (α := α) (0 : α) (.dim hiddenSize .scalar)
          let out0T : Tensor α (.dim seqLen (.dim hiddenSize .scalar)) :=
            Spec.fill (α := α) (0 : α) (.dim seqLen (.dim hiddenSize .scalar))
          let h0 ← TorchLean.const (m := m) (α := α) (s := .dim hiddenSize .scalar) h0T
          let out0 ← TorchLean.const (m := m) (α := α) (s := .dim seqLen (.dim hiddenSize .scalar))
            out0T
          let (_, out) ← (List.finRange seqLen).foldlM (init := (h0, out0)) (fun st t => do
            let (hPrev, outPrev) := st
            let x_t ← TorchLean.gatherRow (m := m) (α := α) (rows := seqLen) (cols := inputSize) xs t
            let concat ← TorchLean.concatVectors (m := m) (α := α)
              (nDim := inputSize) (mDim := hiddenSize) x_t hPrev
            let pre ← TorchLean.linear (m := m) (α := α)
              (inDim := inputSize + hiddenSize) (outDim := hiddenSize)
              w b concat
            let h_t ← TorchLean.tanh (m := m) (α := α) (s := .dim hiddenSize .scalar) pre
            let outNext ← TorchLean.scatterAddRow (m := m) (α := α)
              (rows := seqLen) (cols := hiddenSize) outPrev h_t t
            pure (h_t, outNext))
          pure out
  }

/--
GRU layer (time-major sequence, no batch axis).

This is an unrolled GRU using the standard gate equations (reset/update/candidate), with
`h_{-1} = 0`.

PyTorch analogy: `torch.nn.GRU(inputSize, hiddenSize)` with `batch_first=false`, specialized to a
single batch element.
Docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html
-/
def gru (seqLen inputSize hiddenSize : Nat) (seedW seedB : Nat := 0) :
    LayerDef (.dim seqLen (.dim inputSize .scalar)) (.dim seqLen (.dim hiddenSize .scalar)) :=
  let WShape : Shape := NN.Tensor.Shape.Mat hiddenSize (inputSize + hiddenSize)
  let bShape : Shape := NN.Tensor.Shape.Vec hiddenSize
  let wReset0 : Tensor Float WShape := Torch.Init.xavierW (outDim := hiddenSize) (inDim := inputSize +
    hiddenSize) (seed := seedW + 0)
  let bReset0 : Tensor Float bShape := Torch.Init.tensor (s := bShape) (sch := .zeros) (seed :=
    seedB + 0)
  let wUpdate0 : Tensor Float WShape := Torch.Init.xavierW (outDim := hiddenSize) (inDim := inputSize +
    hiddenSize) (seed := seedW + 1)
  let bUpdate0 : Tensor Float bShape := Torch.Init.tensor (s := bShape) (sch := .zeros) (seed :=
    seedB + 1)
  let wNew0 : Tensor Float WShape := Torch.Init.xavierW (outDim := hiddenSize) (inDim := inputSize +
    hiddenSize) (seed := seedW + 2)
  let bNew0 : Tensor Float bShape := Torch.Init.tensor (s := bShape) (sch := .zeros) (seed :=
    seedB + 2)
  { paramShapes := [WShape, bShape, WShape, bShape, WShape, bShape]
    initParams := .cons wReset0 (.cons bReset0 (.cons wUpdate0 (.cons bUpdate0 (.cons wNew0 (.cons bNew0 .nil)))))
    paramRequiresGrad := [true, true, true, true, true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun wReset bReset wUpdate bUpdate wNew bNew xs =>
          show m (Ref (.dim seqLen (.dim hiddenSize .scalar))) from do
          let h0T : Tensor α (.dim hiddenSize .scalar) :=
            Spec.fill (α := α) (0 : α) (.dim hiddenSize .scalar)
          let out0T : Tensor α (.dim seqLen (.dim hiddenSize .scalar)) :=
            Spec.fill (α := α) (0 : α) (.dim seqLen (.dim hiddenSize .scalar))
          let onesT : Tensor α (.dim hiddenSize .scalar) :=
            Spec.fill (α := α) (1 : α) (.dim hiddenSize .scalar)
          let h0 ← TorchLean.const (m := m) (α := α) (s := .dim hiddenSize .scalar) h0T
          let out0 ← TorchLean.const (m := m) (α := α) (s := .dim seqLen (.dim hiddenSize .scalar))
            out0T
          let ones ← TorchLean.const (m := m) (α := α) (s := .dim hiddenSize .scalar) onesT
          let (_, out) ← (List.finRange seqLen).foldlM (init := (h0, out0)) (fun st t => do
            let (hPrev, outPrev) := st
            let x_t ← TorchLean.gatherRow (m := m) (α := α) (rows := seqLen) (cols := inputSize) xs t
            let concat ← TorchLean.concatVectors (m := m) (α := α)
              (nDim := inputSize) (mDim := hiddenSize) x_t hPrev
            let r_pre ← TorchLean.linear (m := m) (α := α)
              (inDim := inputSize + hiddenSize) (outDim := hiddenSize)
              wReset bReset concat
            let r ← TorchLean.sigmoid (m := m) (α := α) (s := .dim hiddenSize .scalar) r_pre
            let z_pre ← TorchLean.linear (m := m) (α := α)
              (inDim := inputSize + hiddenSize) (outDim := hiddenSize)
              wUpdate bUpdate concat
            let z ← TorchLean.sigmoid (m := m) (α := α) (s := .dim hiddenSize .scalar) z_pre
            let r_hPrev ← TorchLean.mul (m := m) (α := α) (s := .dim hiddenSize .scalar) r hPrev
            let concat2 ← TorchLean.concatVectors (m := m) (α := α)
              (nDim := inputSize) (mDim := hiddenSize) x_t r_hPrev
            let n_pre ← TorchLean.linear (m := m) (α := α)
              (inDim := inputSize + hiddenSize) (outDim := hiddenSize)
              wNew bNew concat2
            let n ← TorchLean.tanh (m := m) (α := α) (s := .dim hiddenSize .scalar) n_pre
            let oneMinusZ ← TorchLean.sub (m := m) (α := α) (s := .dim hiddenSize .scalar) ones z
            let newContrib ← TorchLean.mul (m := m) (α := α) (s := .dim hiddenSize .scalar) oneMinusZ n
            let hiddenContrib ← TorchLean.mul (m := m) (α := α) (s := .dim hiddenSize .scalar) z hPrev
            let h_t ← TorchLean.add (m := m) (α := α) (s := .dim hiddenSize .scalar) newContrib hiddenContrib
            let outNext ← TorchLean.scatterAddRow (m := m) (α := α)
              (rows := seqLen) (cols := hiddenSize) outPrev h_t t
            pure (h_t, outNext))
          pure out
  }

/--
Mamba-style gated diagonal state-space layer (time-major sequence, no batch axis).

This is the trainable recurrent core used by the runnable Mamba text example.  At each time step it
learns an input candidate, a token/state-dependent retention gate, and an output gate:

`u_t = silu(Wᵤ x_t + bᵤ)`

`δ_t = sigmoid(Wδ [x_t; h_{t-1}] + bδ)`

`h_t = δ_t * h_{t-1} + (1 - δ_t) * u_t`

`y_t = h_t * silu(Wz x_t + bz)`

The recurrence is unrolled with ordinary TorchLean differentiable ops, so the same definition trains
on the CPU backend and on the CUDA backend.  The lower-level selective-scan CUDA kernels are still
available for forward experiments, but this layer is intentionally built from autograd-covered ops so
all projections and gates train correctly.
-/
def mamba (seqLen inputSize hiddenSize : Nat) (seedW seedB : Nat := 0) :
    LayerDef (.dim seqLen (.dim inputSize .scalar)) (.dim seqLen (.dim hiddenSize .scalar)) :=
  let WInShape : Shape := NN.Tensor.Shape.Mat hiddenSize inputSize
  let WDeltaShape : Shape := NN.Tensor.Shape.Mat hiddenSize (inputSize + hiddenSize)
  let bShape : Shape := NN.Tensor.Shape.Vec hiddenSize
  let wIn0 : Tensor Float WInShape := Torch.Init.xavierW
    (outDim := hiddenSize) (inDim := inputSize) (seed := seedW + 0)
  let bIn0 : Tensor Float bShape := Torch.Init.tensor (s := bShape) (sch := .zeros)
    (seed := seedB + 0)
  let wDelta0 : Tensor Float WDeltaShape := Torch.Init.xavierW
    (outDim := hiddenSize) (inDim := inputSize + hiddenSize) (seed := seedW + 1)
  let bDelta0 : Tensor Float bShape := Torch.Init.tensor (s := bShape) (sch := .zeros)
    (seed := seedB + 1)
  let wGate0 : Tensor Float WInShape := Torch.Init.xavierW
    (outDim := hiddenSize) (inDim := inputSize) (seed := seedW + 2)
  let bGate0 : Tensor Float bShape := Torch.Init.tensor (s := bShape) (sch := .zeros)
    (seed := seedB + 2)
  { paramShapes := [WInShape, bShape, WDeltaShape, bShape, WInShape, bShape]
    initParams := .cons wIn0 (.cons bIn0 (.cons wDelta0 (.cons bDelta0
      (.cons wGate0 (.cons bGate0 .nil)))))
    paramRequiresGrad := [true, true, true, true, true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun wIn bIn wDelta bDelta wGate bGate xs =>
          show m (Ref (.dim seqLen (.dim hiddenSize .scalar))) from do
          let h0T : Tensor α (.dim hiddenSize .scalar) :=
            Spec.fill (α := α) (0 : α) (.dim hiddenSize .scalar)
          let out0T : Tensor α (.dim seqLen (.dim hiddenSize .scalar)) :=
            Spec.fill (α := α) (0 : α) (.dim seqLen (.dim hiddenSize .scalar))
          let onesT : Tensor α (.dim hiddenSize .scalar) :=
            Spec.fill (α := α) (1 : α) (.dim hiddenSize .scalar)
          let h0 ← TorchLean.const (m := m) (α := α) (s := .dim hiddenSize .scalar) h0T
          let out0 ← TorchLean.const (m := m) (α := α) (s := .dim seqLen (.dim hiddenSize .scalar))
            out0T
          let ones ← TorchLean.const (m := m) (α := α) (s := .dim hiddenSize .scalar) onesT
          let (_, out) ← (List.finRange seqLen).foldlM (init := (h0, out0)) (fun st t => do
            let (hPrev, outPrev) := st
            let x_t ← TorchLean.gatherRow (m := m) (α := α) (rows := seqLen) (cols := inputSize) xs t
            let uPre ← TorchLean.linear (m := m) (α := α)
              (inDim := inputSize) (outDim := hiddenSize) wIn bIn x_t
            let u ← _root_.Runtime.Autograd.Torch.silu
              (m := m) (α := α) (s := .dim hiddenSize .scalar) uPre
            let concat ← TorchLean.concatVectors (m := m) (α := α)
              (nDim := inputSize) (mDim := hiddenSize) x_t hPrev
            let deltaPre ← TorchLean.linear (m := m) (α := α)
              (inDim := inputSize + hiddenSize) (outDim := hiddenSize)
              wDelta bDelta concat
            let delta ← TorchLean.sigmoid (m := m) (α := α) (s := .dim hiddenSize .scalar) deltaPre
            let oneMinusDelta ← TorchLean.sub (m := m) (α := α) (s := .dim hiddenSize .scalar)
              ones delta
            let keep ← TorchLean.mul (m := m) (α := α) (s := .dim hiddenSize .scalar)
              delta hPrev
            let write ← TorchLean.mul (m := m) (α := α) (s := .dim hiddenSize .scalar)
              oneMinusDelta u
            let h_t ← TorchLean.add (m := m) (α := α) (s := .dim hiddenSize .scalar)
              keep write
            let gatePre ← TorchLean.linear (m := m) (α := α)
              (inDim := inputSize) (outDim := hiddenSize) wGate bGate x_t
            let gate ← _root_.Runtime.Autograd.Torch.silu
              (m := m) (α := α) (s := .dim hiddenSize .scalar) gatePre
            let y_t ← TorchLean.mul (m := m) (α := α) (s := .dim hiddenSize .scalar)
              h_t gate
            let outNext ← TorchLean.scatterAddRow (m := m) (α := α)
              (rows := seqLen) (cols := hiddenSize) outPrev y_t t
            pure (h_t, outNext))
          pure out
  }

/--
LSTM layer (time-major sequence, no batch axis).

This is an unrolled LSTM using the standard four gates, with `(h_{-1}, c_{-1}) = (0, 0)`.

PyTorch analogy: `torch.nn.LSTM(inputSize, hiddenSize)` with `batch_first=false`, specialized to a
single batch element.
Docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html
-/
def lstm (seqLen inputSize hiddenSize : Nat) (seedW seedB : Nat := 0) :
    LayerDef (.dim seqLen (.dim inputSize .scalar)) (.dim seqLen (.dim hiddenSize .scalar)) :=
  let WShape : Shape := NN.Tensor.Shape.Mat hiddenSize (inputSize + hiddenSize)
  let bShape : Shape := NN.Tensor.Shape.Vec hiddenSize
  let wF0 : Tensor Float WShape := Torch.Init.xavierW (outDim := hiddenSize) (inDim := inputSize +
    hiddenSize) (seed := seedW + 0)
  let bF0 : Tensor Float bShape := Torch.Init.tensor (s := bShape) (sch := .zeros) (seed := seedB + 0)
  let wI0 : Tensor Float WShape := Torch.Init.xavierW (outDim := hiddenSize) (inDim := inputSize +
    hiddenSize) (seed := seedW + 1)
  let bI0 : Tensor Float bShape := Torch.Init.tensor (s := bShape) (sch := .zeros) (seed := seedB + 1)
  let wC0 : Tensor Float WShape := Torch.Init.xavierW (outDim := hiddenSize) (inDim := inputSize +
    hiddenSize) (seed := seedW + 2)
  let bC0 : Tensor Float bShape := Torch.Init.tensor (s := bShape) (sch := .zeros) (seed := seedB + 2)
  let wO0 : Tensor Float WShape := Torch.Init.xavierW (outDim := hiddenSize) (inDim := inputSize +
    hiddenSize) (seed := seedW + 3)
  let bO0 : Tensor Float bShape := Torch.Init.tensor (s := bShape) (sch := .zeros) (seed := seedB + 3)
  { paramShapes := [WShape, bShape, WShape, bShape, WShape, bShape, WShape, bShape]
    initParams :=
      .cons wF0 (.cons bF0 (.cons wI0 (.cons bI0 (.cons wC0 (.cons bC0 (.cons wO0 (.cons bO0 .nil)))))))
    paramRequiresGrad := [true, true, true, true, true, true, true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun wF bF wI bI wC bC wO bO xs =>
          show m (Ref (.dim seqLen (.dim hiddenSize .scalar))) from do
          let h0T : Tensor α (.dim hiddenSize .scalar) :=
            Spec.fill (α := α) (0 : α) (.dim hiddenSize .scalar)
          let out0T : Tensor α (.dim seqLen (.dim hiddenSize .scalar)) :=
            Spec.fill (α := α) (0 : α) (.dim seqLen (.dim hiddenSize .scalar))
          let h0 ← TorchLean.const (m := m) (α := α) (s := .dim hiddenSize .scalar) h0T
          let c0 ← TorchLean.const (m := m) (α := α) (s := .dim hiddenSize .scalar) h0T
          let out0 ← TorchLean.const (m := m) (α := α) (s := .dim seqLen (.dim hiddenSize .scalar))
            out0T
          let (_, _, out) ← (List.finRange seqLen).foldlM (init := (h0, c0, out0)) (fun st t => do
            let (hPrev, cPrev, outPrev) := st
            let x_t ← TorchLean.gatherRow (m := m) (α := α) (rows := seqLen) (cols := inputSize) xs t
            let concat ← TorchLean.concatVectors (m := m) (α := α)
              (nDim := inputSize) (mDim := hiddenSize) x_t hPrev
            let f_pre ← TorchLean.linear (m := m) (α := α)
              (inDim := inputSize + hiddenSize) (outDim := hiddenSize)
              wF bF concat
            let f ← TorchLean.sigmoid (m := m) (α := α) (s := .dim hiddenSize .scalar) f_pre
            let i_pre ← TorchLean.linear (m := m) (α := α)
              (inDim := inputSize + hiddenSize) (outDim := hiddenSize)
              wI bI concat
            let i ← TorchLean.sigmoid (m := m) (α := α) (s := .dim hiddenSize .scalar) i_pre
            let g_pre ← TorchLean.linear (m := m) (α := α)
              (inDim := inputSize + hiddenSize) (outDim := hiddenSize)
              wC bC concat
            let g ← TorchLean.tanh (m := m) (α := α) (s := .dim hiddenSize .scalar) g_pre
            let o_pre ← TorchLean.linear (m := m) (α := α)
              (inDim := inputSize + hiddenSize) (outDim := hiddenSize)
              wO bO concat
            let o ← TorchLean.sigmoid (m := m) (α := α) (s := .dim hiddenSize .scalar) o_pre
            let fc ← TorchLean.mul (m := m) (α := α) (s := .dim hiddenSize .scalar) f cPrev
            let ig ← TorchLean.mul (m := m) (α := α) (s := .dim hiddenSize .scalar) i g
            let c_t ← TorchLean.add (m := m) (α := α) (s := .dim hiddenSize .scalar) fc ig
            let tanhC ← TorchLean.tanh (m := m) (α := α) (s := .dim hiddenSize .scalar) c_t
            let h_t ← TorchLean.mul (m := m) (α := α) (s := .dim hiddenSize .scalar) o tanhC
            let outNext ← TorchLean.scatterAddRow (m := m) (α := α)
              (rows := seqLen) (cols := hiddenSize) outPrev h_t t
            pure (h_t, c_t, outNext))
          pure out
  }

/--
ReLU activation layer (no parameters).

PyTorch analogy: `torch.nn.ReLU` / `torch.nn.functional.relu`.
-/
def relu {s : Shape} : LayerDef s s :=
  { paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => TorchLean.relu (m := m) (α := α) (s := s) x
  }

/--
SiLU (a.k.a. swish) activation layer (no parameters).

PyTorch analogy: `torch.nn.SiLU` / `torch.nn.functional.silu`.
-/
def silu {s : Shape} : LayerDef s s :=
  { paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => _root_.Runtime.Autograd.Torch.silu (m := m) (α := α) (s := s) x
  }

/--
GELU activation layer (no parameters).

PyTorch analogy: `torch.nn.GELU` / `torch.nn.functional.gelu`.
-/
def gelu {s : Shape} : LayerDef s s :=
  { paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => _root_.Runtime.Autograd.Torch.gelu (m := m) (α := α) (s := s) x
  }

/--
Sigmoid activation layer (no parameters).

PyTorch analogy: `torch.sigmoid`.
-/
def sigmoid {s : Shape} : LayerDef s s :=
  { paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => TorchLean.sigmoid (m := m) (α := α) (s := s) x
  }

/--
Hyperbolic tangent activation layer (no parameters).

PyTorch analogy: `torch.tanh`.
-/
def tanh {s : Shape} : LayerDef s s :=
  { paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => TorchLean.tanh (m := m) (α := α) (s := s) x
  }

/--
Softmax layer along the last axis (shape-preserving, no parameters).

PyTorch analogy: `torch.softmax(x, dim=-1)`.
-/
def softmax {s : Shape} : LayerDef s s :=
  { paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => TorchLean.softmax (m := m) (α := α) (s := s) x
  }

/--
Pointwise square `x ↦ x^2` (no parameters).

PyTorch analogy: `torch.square(x)` / `x.square()`.
-/
def square {s : Shape} : LayerDef s s :=
  { paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => TorchLean.F.square (m := m) (α := α) (s := s) x
  }

/--
Sum-reduce all elements of the input to a scalar (no parameters).

PyTorch analogy: `x.sum()`.
-/
def sum {s : Shape} : LayerDef s Shape.scalar :=
  { paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => TorchLean.sum (m := m) (α := α) (s := s) x
  }

/--
Flatten any tensor to a 1D vector of length `Shape.size s` (no parameters).

PyTorch analogy: `torch.flatten(x)` or `x.reshape(-1)`.
-/
def flatten {s : Shape} : LayerDef s (.dim (Shape.size s) .scalar) :=
  { paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => TorchLean.flatten (m := m) (α := α) (s := s) x
  }

/--
Flatten everything except the leading batch axis.

Input shape: `N × s`. Output shape: `N × (size s)`.

PyTorch analogy: `torch.flatten(x, start_dim=1)` for an `N×…` tensor.
-/
def flattenKeep0 {batch : Nat} {s : Shape} :
    LayerDef (.dim batch s) (.dim batch (.dim (Shape.size s) .scalar)) :=
  { paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => TorchLean.flattenKeep0 (m := m) (α := α) (batch := batch) (s := s) x
  }

/--
Dropout layer controlled by `Mode`.

- In `Mode.train`, randomly zeroes entries with probability `p`.
- In `Mode.eval`, it is the identity.

We store `p` as a scalar parameter tensor (with `requires_grad := false`) so it can be threaded
through the unified parameter list without being optimized.

PyTorch analogy: `torch.nn.Dropout(p)` / `torch.nn.functional.dropout(x, p, training=...)`.
-/
def dropout {s : Shape} (p : Float) (seed : Nat := 0) : LayerDef s s :=
  let pShape : Shape := Shape.scalar
  let p0 : Tensor Float pShape := Tensor.scalar p
  { paramShapes := [pShape]
    initParams := Torch.tlist1 p0
    paramRequiresGrad := [false]
    forward := fun mode {α} _ _ =>
      fun {m} _ _ =>
        fun pRef x =>
          _root_.Runtime.Autograd.TorchLean.F.dropoutRefSeeded (m := m) (α := α) (s := s) x pRef
            seed
            (training := mode == .train)
  }

/--
Layer normalization over the last axis of a `(seqLen × embedDim)` activation.

This learns `gamma` and `beta` vectors of shape `(embedDim)`, applied per token position.

PyTorch analogy: `torch.nn.LayerNorm(embedDim)` applied to a sequence tensor.
-/
def layerNorm
    (batch seqLen embedDim : Nat)
    {h_seq_pos : seqLen > 0} {h_embed_pos : embedDim > 0}
    (seedGamma seedBeta : Nat := 0) :
    LayerDef (.dim batch (.dim seqLen (.dim embedDim .scalar)))
      (.dim batch (.dim seqLen (.dim embedDim .scalar))) :=
  let gammaShape : Shape := NN.Tensor.Shape.Vec embedDim
  let betaShape : Shape := NN.Tensor.Shape.Vec embedDim
  let gamma0 : Tensor Float gammaShape := Torch.Init.tensor (s := gammaShape) (sch := .ones) (seed
    := seedGamma)
  let beta0 : Tensor Float betaShape := Torch.Init.tensor (s := betaShape) (sch := .zeros) (seed :=
    seedBeta)
  { paramShapes := [gammaShape, betaShape]
    initParams := Torch.tlist2 gamma0 beta0
    paramRequiresGrad := [true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun gamma beta x =>
          TorchLean.layerNorm (m := m) (α := α)
            (batch := batch) (seqLen := seqLen) (embedDim := embedDim)
            (h_seq_pos := h_seq_pos) (h_embed_pos := h_embed_pos)
            x gamma beta
  }

/--
RMS normalization over the last axis of a `(seqLen × embedDim)` activation.

This learns a `gamma` vector of shape `(embedDim)` and is commonly used in transformer models.

PyTorch analogy: a typical RMSNorm implementation in `torch.nn`-style code (often a small custom
`nn.Module`).
-/
def rmsNorm
    (batch seqLen embedDim : Nat)
    {h_seq_pos : seqLen > 0} {h_embed_pos : embedDim > 0}
    (seedGamma : Nat := 0) :
    LayerDef (.dim batch (.dim seqLen (.dim embedDim .scalar)))
      (.dim batch (.dim seqLen (.dim embedDim .scalar))) :=
  let gammaShape : Shape := NN.Tensor.Shape.Vec embedDim
  let gamma0 : Tensor Float gammaShape := Torch.Init.tensor (s := gammaShape) (sch := .ones) (seed
    := seedGamma)
  { paramShapes := [gammaShape]
    initParams := Torch.tlist1 gamma0
    paramRequiresGrad := [true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun gamma x =>
          TorchLean.Norm.rmsNormLastBatched (m := m) (α := α)
            (batch := batch) (seqLen := seqLen) (embedDim := embedDim)
            (h_seq_pos := h_seq_pos) (h_embed_pos := h_embed_pos)
            x gamma
  }

/--
Batch normalization for an unbatched `C×H×W` tensor (channel-first).

This uses the current activation’s per-channel statistics (over spatial axes) and applies learnable
affine parameters `gamma`/`beta`.

PyTorch analogy: `torch.nn.BatchNorm2d(channels)` in training mode (applied to a single sample).
-/
def batchnormChannelFirst
    (channels height width : Nat)
    {h_c : channels > 0} {h_h : height > 0} {h_w : width > 0}
    (seedGamma seedBeta : Nat := 0) :
    LayerDef (NN.Tensor.Shape.CHW channels height width) (NN.Tensor.Shape.CHW channels height width)
      :=
  let gammaShape : Shape := NN.Tensor.Shape.Vec channels
  let betaShape : Shape := NN.Tensor.Shape.Vec channels
  let gamma0 : Tensor Float gammaShape := Torch.Init.tensor (s := gammaShape) (sch := .ones) (seed
    := seedGamma)
  let beta0 : Tensor Float betaShape := Torch.Init.tensor (s := betaShape) (sch := .zeros) (seed :=
    seedBeta)
  { paramShapes := [gammaShape, betaShape]
    initParams := Torch.tlist2 gamma0 beta0
    paramRequiresGrad := [true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun gamma beta x =>
          TorchLean.batchnormChannelFirst (m := m) (α := α)
            (channels := channels) (height := height) (width := width) (h_c := h_c) (h_h := h_h)
              (h_w := h_w)
            x gamma beta
  }

/--
Batch normalization for a `C×H×W` tensor in eval mode using provided running statistics.

Parameters include `gamma`/`beta` plus fixed `mean`/`var` buffers.

PyTorch analogy: `torch.nn.BatchNorm2d` in eval mode (uses `running_mean` / `running_var`).
-/
def batchnormChannelFirstEval
    (channels height width : Nat)
    {h_c : channels > 0} {h_h : height > 0} {h_w : width > 0}
    (seedGamma seedBeta seedMean seedVar : Nat := 0) :
    LayerDef (NN.Tensor.Shape.CHW channels height width) (NN.Tensor.Shape.CHW channels height width)
      :=
  let gammaShape : Shape := NN.Tensor.Shape.Vec channels
  let betaShape : Shape := NN.Tensor.Shape.Vec channels
  let meanShape : Shape := NN.Tensor.Shape.Vec channels
  let varShape : Shape := NN.Tensor.Shape.Vec channels
  let gamma0 : Tensor Float gammaShape := Torch.Init.tensor (s := gammaShape) (sch := .ones) (seed
    := seedGamma)
  let beta0 : Tensor Float betaShape := Torch.Init.tensor (s := betaShape) (sch := .zeros) (seed :=
    seedBeta)
  let mean0 : Tensor Float meanShape := Torch.Init.tensor (s := meanShape) (sch := .zeros) (seed :=
    seedMean)
  let var0 : Tensor Float varShape := Torch.Init.tensor (s := varShape) (sch := .ones) (seed :=
    seedVar)
  { paramShapes := [gammaShape, betaShape, meanShape, varShape]
    initParams := Torch.tlist4 gamma0 beta0 mean0 var0
    paramRequiresGrad := [true, true, false, false]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun gamma beta mean var x =>
          TorchLean.Norm.batchNorm2dChwEval (m := m) (α := α)
            (c := channels) (h := height) (w := width)
            h_c h_h h_w x gamma beta mean var
  }

/--
Batch normalization for `C×H×W` with explicit `Mode` and running-statistics buffers.

- In `Mode.train`, computes per-channel batch stats and updates `(runningMean, runningVar)` using
  `momentum`.
- In `Mode.eval`, normalizes using the stored running buffers.

PyTorch analogy: `torch.nn.BatchNorm2d(channels, momentum=...)` with `.train()` / `.eval()`
  behavior.
-/
def batchnormChannelFirstMode
    (channels height width : Nat)
    {h_c : channels > 0} {h_h : height > 0} {h_w : width > 0}
    (seedGamma seedBeta seedMean seedVar : Nat := 0)
    (momentum : Float := 0.1) :
    LayerDef (NN.Tensor.Shape.CHW channels height width) (NN.Tensor.Shape.CHW channels height width)
      :=
  let gammaShape : Shape := NN.Tensor.Shape.Vec channels
  let betaShape : Shape := NN.Tensor.Shape.Vec channels
  let meanShape : Shape := NN.Tensor.Shape.Vec channels
  let varShape : Shape := NN.Tensor.Shape.Vec channels
  let momentumShape : Shape := Shape.scalar
  let gamma0 : Tensor Float gammaShape := Torch.Init.tensor (s := gammaShape) (sch := .ones) (seed
    := seedGamma)
  let beta0 : Tensor Float betaShape := Torch.Init.tensor (s := betaShape) (sch := .zeros) (seed :=
    seedBeta)
  let mean0 : Tensor Float meanShape := Torch.Init.tensor (s := meanShape) (sch := .zeros) (seed :=
    seedMean)
  let var0 : Tensor Float varShape := Torch.Init.tensor (s := varShape) (sch := .ones) (seed :=
    seedVar)
  let momentum0 : Tensor Float momentumShape := Tensor.scalar momentum
  { paramShapes := [gammaShape, betaShape, meanShape, varShape, momentumShape]
    initParams := .cons gamma0 (.cons beta0 (.cons mean0 (.cons var0 (.cons momentum0 .nil))))
    paramRequiresGrad := [true, true, false, false, false]
    updateBuffers := some (fun mode {_α} _ _ ps x => do
      match mode, ps with
      | .eval, _ => pure ps
      | .train, .cons gamma (.cons beta (.cons runningMean (.cons runningVar (.cons momentumT
        .nil)))) =>
          let (batchMean, batchVar) := chwBatchStats x
          let nextMean := updateRunningVec runningMean batchMean momentumT
          let nextVar := updateRunningVec runningVar batchVar momentumT
          pure (.cons gamma (.cons beta (.cons nextMean (.cons nextVar (.cons momentumT .nil)))))
      | .train, _ => pure ps
    )
    forward := fun mode {α} _ _ =>
      fun {m} _ _ =>
        fun gamma beta mean var _momentum x =>
          match mode with
          | .train =>
              TorchLean.batchnormChannelFirst (m := m) (α := α)
                (channels := channels) (height := height) (width := width)
                (h_c := h_c) (h_h := h_h) (h_w := h_w) x gamma beta
          | .eval =>
              TorchLean.Norm.batchNorm2dChwEval (m := m) (α := α)
                (c := channels) (h := height) (w := width)
                h_c h_h h_w x gamma beta mean var
  }

/--
Instance normalization for `N×C×H×W` tensors.

This normalizes each sample independently (per-channel), then applies learnable affine parameters
`gamma`/`beta`.

PyTorch analogy: `torch.nn.InstanceNorm2d(c, affine=True)` (with `NCHW` layout).
-/
def instanceNorm2dNchw
    (n c h w : Nat)
    {h_n_pos : n > 0} {h_c_pos : c > 0} {h_h_pos : h > 0} {h_w_pos : w > 0}
    (seedGamma seedBeta : Nat := 0) :
    LayerDef (NN.Tensor.Shape.NCHW n c h w) (NN.Tensor.Shape.NCHW n c h w) :=
  let gammaShape : Shape := NN.Tensor.Shape.Vec c
  let betaShape : Shape := NN.Tensor.Shape.Vec c
  let gamma0 : Tensor Float gammaShape := Torch.Init.tensor (s := gammaShape) (sch := .ones) (seed
    := seedGamma)
  let beta0 : Tensor Float betaShape := Torch.Init.tensor (s := betaShape) (sch := .zeros) (seed :=
    seedBeta)
  { paramShapes := [gammaShape, betaShape]
    initParams := Torch.tlist2 gamma0 beta0
    paramRequiresGrad := [true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun gamma beta x =>
          TorchLean.Norm.instanceNorm2dNchw (m := m) (α := α)
            (n := n) (c := c) (h := h) (w := w)
            h_n_pos h_c_pos h_h_pos h_w_pos
            x gamma beta
  }

/--
Group normalization for `N×C×H×W` tensors.

Channels are split into `groups` groups (requiring `c % groups = 0`), normalization is performed per
group, then learnable affine parameters `gamma`/`beta` are applied.

PyTorch analogy: `torch.nn.GroupNorm(groups, c)` (with `NCHW` layout).
-/
def groupNorm2dNchw
    (n c h w groups : Nat)
    {h_n_pos : n > 0} {h_c_pos : c > 0} {h_h_pos : h > 0} {h_w_pos : w > 0} {h_g_pos : groups > 0}
    (h_ge : c ≥ groups) (h_div : c % groups = 0)
    (seedGamma seedBeta : Nat := 0) :
    LayerDef (NN.Tensor.Shape.NCHW n c h w) (NN.Tensor.Shape.NCHW n c h w) :=
  let gammaShape : Shape := NN.Tensor.Shape.Vec c
  let betaShape : Shape := NN.Tensor.Shape.Vec c
  let gamma0 : Tensor Float gammaShape := Torch.Init.tensor (s := gammaShape) (sch := .ones) (seed
    := seedGamma)
  let beta0 : Tensor Float betaShape := Torch.Init.tensor (s := betaShape) (sch := .zeros) (seed :=
    seedBeta)
  { paramShapes := [gammaShape, betaShape]
    initParams := Torch.tlist2 gamma0 beta0
    paramRequiresGrad := [true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun gamma beta x =>
          TorchLean.Norm.groupNorm2dNchw (m := m) (α := α)
            (n := n) (c := c) (h := h) (w := w) (groups := groups)
            h_n_pos h_c_pos h_h_pos h_w_pos h_g_pos h_ge h_div
            x gamma beta
  }

/--
Batch normalization training behavior for `N×C×H×W` tensors (no running buffers).

This computes batch statistics across `(N, H, W)` and applies learnable affine parameters
`gamma`/`beta`.

PyTorch analogy: `torch.nn.BatchNorm2d(c)` in training mode (stat computation).
-/
def batchNorm2dNchw
    (n c h w : Nat)
    {h_n_pos : n > 0} {h_c_pos : c > 0} {h_h_pos : h > 0} {h_w_pos : w > 0}
    (seedGamma seedBeta : Nat := 0) :
    LayerDef (NN.Tensor.Shape.NCHW n c h w) (NN.Tensor.Shape.NCHW n c h w) :=
  let gammaShape : Shape := NN.Tensor.Shape.Vec c
  let betaShape : Shape := NN.Tensor.Shape.Vec c
  let gamma0 : Tensor Float gammaShape := Torch.Init.tensor (s := gammaShape) (sch := .ones) (seed
    := seedGamma)
  let beta0 : Tensor Float betaShape := Torch.Init.tensor (s := betaShape) (sch := .zeros) (seed :=
    seedBeta)
  { paramShapes := [gammaShape, betaShape]
    initParams := Torch.tlist2 gamma0 beta0
    paramRequiresGrad := [true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun gamma beta x =>
          TorchLean.Norm.batchNorm2dNchwTrain (m := m) (α := α)
            (n := n) (c := c) (h := h) (w := w)
            h_n_pos h_c_pos h_h_pos h_w_pos
            x gamma beta
  }

/--
Batch normalization for `N×C×H×W` with explicit `Mode` and running-statistics buffers.

Parameters include `gamma`, `beta`, running `mean`/`var` buffers, and a momentum scalar:
- in `Mode.train`, compute batch stats and update running buffers,
- in `Mode.eval`, normalize using the running buffers.

PyTorch analogy: `torch.nn.BatchNorm2d(c, momentum=...)` with `.train()` / `.eval()` behavior.
-/
def batchNorm2dNchwMode
    (n c h w : Nat)
    {h_n_pos : n > 0} {h_c_pos : c > 0} {h_h_pos : h > 0} {h_w_pos : w > 0}
    (seedGamma seedBeta seedMean seedVar : Nat := 0)
    (momentum : Float := 0.1) :
    LayerDef (NN.Tensor.Shape.NCHW n c h w) (NN.Tensor.Shape.NCHW n c h w) :=
  let gammaShape : Shape := NN.Tensor.Shape.Vec c
  let betaShape : Shape := NN.Tensor.Shape.Vec c
  let meanShape : Shape := NN.Tensor.Shape.Vec c
  let varShape : Shape := NN.Tensor.Shape.Vec c
  let momentumShape : Shape := Shape.scalar
  let gamma0 : Tensor Float gammaShape := Torch.Init.tensor (s := gammaShape) (sch := .ones) (seed
    := seedGamma)
  let beta0 : Tensor Float betaShape := Torch.Init.tensor (s := betaShape) (sch := .zeros) (seed :=
    seedBeta)
  let mean0 : Tensor Float meanShape := Torch.Init.tensor (s := meanShape) (sch := .zeros) (seed :=
    seedMean)
  let var0 : Tensor Float varShape := Torch.Init.tensor (s := varShape) (sch := .ones) (seed :=
    seedVar)
  let momentum0 : Tensor Float momentumShape := Tensor.scalar momentum
  { paramShapes := [gammaShape, betaShape, meanShape, varShape, momentumShape]
    initParams := .cons gamma0 (.cons beta0 (.cons mean0 (.cons var0 (.cons momentum0 .nil))))
    paramRequiresGrad := [true, true, false, false, false]
    updateBuffers := some (fun mode {_α} _ _ ps x => do
      match mode, ps with
      | .eval, _ => pure ps
      | .train, .cons gamma (.cons beta (.cons runningMean (.cons runningVar (.cons momentumT
        .nil)))) =>
          let (batchMean, batchVar) := nchwBatchStats x
          let nextMean := updateRunningVec runningMean batchMean momentumT
          let nextVar := updateRunningVec runningVar batchVar momentumT
          pure (.cons gamma (.cons beta (.cons nextMean (.cons nextVar (.cons momentumT .nil)))))
      | .train, _ => pure ps
    )
    forward := fun mode {α} _ _ =>
      fun {m} _ _ =>
        fun gamma beta mean var _momentum x =>
          match mode with
          | .train =>
              TorchLean.Norm.batchNorm2dNchwTrain (m := m) (α := α)
                (n := n) (c := c) (h := h) (w := w)
                h_n_pos h_c_pos h_h_pos h_w_pos
                x gamma beta
          | .eval =>
              TorchLean.Norm.batchNorm2dNchwEval (m := m) (α := α)
                (n := n) (c := c) (h := h) (w := w)
                h_n_pos h_c_pos h_h_pos h_w_pos
                x gamma beta mean var
  }

/--
N-D convolution layer for a channels-first tensor `(inC, spatial...)` (no batch axis).

Parameters:
- kernel `K : (outC × inC × kernel[0] × ... × kernel[d-1])`,
- bias `b : (outC)`.

The output spatial shape is computed from `(stride, padding, kernel)`.

PyTorch analogy: `torch.nn.Conv{d}d` / `torch.nn.functional.conv{d}d` specialized to a single
sample (no batch axis), with `groups=1` and `dilation=1`.
-/
def conv
    (batch d inC outC : Nat)
    (kernel stride padding : Vector Nat d)
    (inSpatial : Vector Nat d)
    {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    (seedK seedB : Nat := 0)
    (kInit : Torch.Init.Scheme := .uniform (-0.1) 0.1) :
    LayerDef (.dim batch (Shape.ofList (inC :: inSpatial.toList)))
      (.dim batch (Shape.ofList (outC :: (Spec.convOutSpatial inSpatial kernel stride padding).toList))) :=
  let KShape : Shape := Shape.ofList (outC :: inC :: kernel.toList)
  let bShape : Shape := NN.Tensor.Shape.Vec outC
  let k0 : Tensor Float KShape := Torch.Init.tensor (s := KShape) (sch := kInit) (seed := seedK)
  let b0 : Tensor Float bShape := Torch.Init.tensor (s := bShape) (sch := .zeros) (seed := seedB)
  { paramShapes := [KShape, bShape]
    initParams := Torch.tlist2 k0 b0
    paramRequiresGrad := [true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun k b x =>
          TorchLean.conv (m := m) (α := α)
            (batch := batch) (d := d) (inC := inC) (outC := outC)
            (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
            (hInC := hInC) (hKernel := hKernel)
            k b x
  }

/--
N-D transpose convolution layer for a channels-first tensor `(inC, spatial...)` (no batch axis).

Parameters:
- kernel `K : (inC × outC × kernel[0] × ... × kernel[d-1])` (PyTorch layout),
- bias `b : (outC)`.

The output spatial shape uses:
`out[a] = (in[a] - 1) * stride[a] - 2*padding[a] + kernel[a]` (with `output_padding = 0`).

PyTorch analogy: `torch.nn.ConvTranspose{d}d` / `torch.nn.functional.conv_transpose{d}d`
specialized to a single sample (no batch axis), with `groups=1`, `dilation=1`, `output_padding=0`.
-/
def convTranspose
    (batch d inC outC : Nat)
    (kernel stride padding : Vector Nat d)
    (inSpatial : Vector Nat d)
    {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    (seedK seedB : Nat := 0)
    (kInit : Torch.Init.Scheme := .uniform (-0.1) 0.1) :
    LayerDef (.dim batch (Shape.ofList (inC :: inSpatial.toList)))
      (.dim batch (Shape.ofList (outC :: (Spec.convTransposeOutSpatial inSpatial kernel stride padding).toList)))
      :=
  let KShape : Shape := Shape.ofList (inC :: outC :: kernel.toList)
  let bShape : Shape := NN.Tensor.Shape.Vec outC
  let k0 : Tensor Float KShape := Torch.Init.tensor (s := KShape) (sch := kInit) (seed := seedK)
  let b0 : Tensor Float bShape := Torch.Init.tensor (s := bShape) (sch := .zeros) (seed := seedB)
  { paramShapes := [KShape, bShape]
    initParams := Torch.tlist2 k0 b0
    paramRequiresGrad := [true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun k b x =>
          TorchLean.convTranspose (m := m) (α := α)
            (batch := batch) (d := d) (inC := inC) (outC := outC)
            (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
            (hInC := hInC) (hKernel := hKernel)
            k b x
  }

/--
N-D max pooling layer for a channels-first tensor `(batch, C, spatial...)` (no parameters).

Output spatial dims follow `Spec.pool_out_spatial_pad`.

PyTorch analogy: `torch.nn.functional.max_pool{d}d` on an `N×C×...` tensor.
-/
def maxPool
    (batch d C : Nat)
    (kernel stride padding : Vector Nat d)
    (inSpatial : Vector Nat d)
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0} :
    LayerDef (.dim batch (Shape.ofList (C :: inSpatial.toList)))
      (.dim batch (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList)))
      :=
  { paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x =>
          TorchLean.maxPool (m := m) (α := α)
            (batch := batch) (d := d) (C := C)
            (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
            (hKernel := hKernel)
            x
  }

/--
N-D average pooling layer for a channels-first tensor `(batch, C, spatial...)` (no parameters).

PyTorch analogy: `torch.nn.functional.avg_pool{d}d` on an `N×C×...` tensor.
-/
def avgPool
    (batch d C : Nat)
    (kernel stride padding : Vector Nat d)
    (inSpatial : Vector Nat d)
    (hKernel : ∀ i : Fin d, kernel.get i ≠ 0) :
    LayerDef (.dim batch (Shape.ofList (C :: inSpatial.toList)))
      (.dim batch (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList)))
      :=
  { paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x =>
          TorchLean.avgPool (m := m) (α := α)
            (batch := batch) (d := d) (C := C)
            (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
            (hKernel := hKernel)
            x
  }

/--
2D convolution layer for a `C×H×W` (channel-first) input.

Parameters:
- kernel `K : (outC × inC × kH × kW)` (OIHW layout),
- bias `b : (outC)`.

The output spatial shape is computed from `(stride, padding, kH, kW)`.

PyTorch analogy: `torch.nn.Conv2d(inC, outC, (kH, kW), stride=stride, padding=padding)`.
-/
def conv2d
    (inC outC kH kW stride padding inH inW : Nat)
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    (seedK seedB : Nat := 0)
    (kInit : Torch.Init.Scheme := .uniform (-0.1) 0.1) :
    LayerDef (NN.Tensor.Shape.CHW inC inH inW)
      (NN.Tensor.Shape.CHW outC ((inH + 2 * padding - kH) / stride + 1) ((inW + 2 * padding - kW) /
        stride + 1)) :=
  let _outH : Nat := (inH + 2 * padding - kH) / stride + 1
  let _outW : Nat := (inW + 2 * padding - kW) / stride + 1
  let KShape : Shape := NN.Tensor.Shape.OIHW outC inC kH kW
  let bShape : Shape := NN.Tensor.Shape.Vec outC
  let k0 : Tensor Float KShape := Torch.Init.tensor (s := KShape) (sch := kInit) (seed := seedK)
  let b0 : Tensor Float bShape := Torch.Init.tensor (s := bShape) (sch := .zeros) (seed := seedB)
  { paramShapes := [KShape, bShape]
    initParams := Torch.tlist2 k0 b0
    paramRequiresGrad := [true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun k b x =>
          TorchLean.conv2d (m := m) (α := α)
            (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding :=
              padding)
            (inH := inH) (inW := inW) (h1 := h1) (h2 := h2) (h3 := h3)
            k b x
  }

/-- Alias for `conv2d` (compat shorthand). -/
abbrev conv2dCompat
    (inC outC kH kW stride padding inH inW : Nat)
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    (seedK seedB : Nat := 0)
    (kInit : Torch.Init.Scheme := .uniform (-0.1) 0.1) :=
  conv2d (inC := inC) (outC := outC) (kH := kH) (kW := kW)
    (stride := stride) (padding := padding) (inH := inH) (inW := inW)
    (h1 := h1) (h2 := h2) (h3 := h3)
    (seedK := seedK) (seedB := seedB) (kInit := kInit)

/--
2D transpose convolution layer for a `C×H×W` (channel-first) input.

Parameters:
- kernel `K : (inC × outC × kH × kW)` (PyTorch layout),
- bias `b : (outC)`.

PyTorch analogy: `torch.nn.ConvTranspose2d(inC, outC, (kH, kW), stride=stride, padding=padding)`
(single-sample CHW specialization).
-/
def convTranspose2d
    (inC outC kH kW stride padding inH inW : Nat)
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    (seedK seedB : Nat := 0)
    (kInit : Torch.Init.Scheme := .uniform (-0.1) 0.1) :
    LayerDef (NN.Tensor.Shape.CHW inC inH inW)
      (NN.Tensor.Shape.CHW outC
        ((inH - 1) * stride - 2 * padding + kH)
        ((inW - 1) * stride - 2 * padding + kW)) :=
  let KShape : Shape := .dim inC (.dim outC (.dim kH (.dim kW .scalar)))
  let bShape : Shape := NN.Tensor.Shape.Vec outC
  let k0 : Tensor Float KShape := Torch.Init.tensor (s := KShape) (sch := kInit) (seed := seedK)
  let b0 : Tensor Float bShape := Torch.Init.tensor (s := bShape) (sch := .zeros) (seed := seedB)
  { paramShapes := [KShape, bShape]
    initParams := Torch.tlist2 k0 b0
    paramRequiresGrad := [true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun k b x =>
          TorchLean.convTranspose2d (m := m) (α := α)
            (inC := inC) (outC := outC) (kH := kH) (kW := kW)
            (stride := stride) (padding := padding)
            (inH := inH) (inW := inW)
            (h1 := h1) (h2 := h2) (h3 := h3)
            k b x
  }

/--
2D max pooling on a `C×H×W` tensor.

PyTorch analogy: `torch.nn.functional.max_pool2d` (channel-first layout).
-/
def maxPool2d
    (kH kW inH inW inC stride : Nat)
    {h1 : kH ≠ 0} {h2 : kW ≠ 0} :
    LayerDef (NN.Tensor.Shape.CHW inC inH inW)
      (NN.Tensor.Shape.CHW inC ((inH - kH) / stride + 1) ((inW - kW) / stride + 1)) :=
  { paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x =>
          TorchLean.maxPool2d (m := m) (α := α)
            (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
            (h1 := h1) (h2 := h2) x
  }

/--
2D max pooling with padding on a `C×H×W` tensor.

PyTorch analogy: `torch.nn.functional.max_pool2d(..., padding=padding)`.
-/
def maxPool2dPad
    (kH kW inH inW inC stride padding : Nat)
    {h1 : kH ≠ 0} {h2 : kW ≠ 0} :
    LayerDef (NN.Tensor.Shape.CHW inC inH inW)
      (NN.Tensor.Shape.CHW inC ((inH + 2 * padding - kH) / stride + 1) ((inW + 2 * padding - kW) /
        stride + 1)) :=
  { paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x =>
          TorchLean.maxPool2dPad (m := m) (α := α)
            (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC)
            (stride := stride) (padding := padding) (h1 := h1) (h2 := h2) x
  }

/--
2D average pooling on a `C×H×W` tensor.

PyTorch analogy: `torch.nn.functional.avg_pool2d` (channel-first layout).
-/
def avgPool2d
    (kH kW inH inW inC stride : Nat)
    (h1 : kH ≠ 0) (h2 : kW ≠ 0) :
    LayerDef (NN.Tensor.Shape.CHW inC inH inW)
      (NN.Tensor.Shape.CHW inC ((inH - kH) / stride + 1) ((inW - kW) / stride + 1)) :=
  { paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x =>
          TorchLean.avgPool2d (m := m) (α := α)
            (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
            h1 h2 x
  }

/--
2D average pooling with padding on a `C×H×W` tensor.

PyTorch analogy: `torch.nn.functional.avg_pool2d(..., padding=padding)`.
-/
def avgPool2dPad
    (kH kW inH inW inC stride padding : Nat)
    (h1 : kH ≠ 0) (h2 : kW ≠ 0) :
    LayerDef (NN.Tensor.Shape.CHW inC inH inW)
      (NN.Tensor.Shape.CHW inC ((inH + 2 * padding - kH) / stride + 1) ((inW + 2 * padding - kW) /
        stride + 1)) :=
  { paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x =>
          TorchLean.avgPool2dPad (m := m) (α := α)
            (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC)
            (stride := stride) (padding := padding) h1 h2 x
  }

/-- Compat alias for `max_pool2d` (unbatched CHW pooling). -/
abbrev maxPool2dCompat := maxPool2d

/-- Alias for `max_pool2d_pad` (PyTorch-style shorthand). -/
abbrev maxPoolPad := maxPool2dPad

/-- Compat alias for `avg_pool2d` (unbatched CHW pooling). -/
abbrev avgPool2dCompat := avgPool2d

/-- Alias for `avg_pool2d_pad` (PyTorch-style shorthand). -/
abbrev avgPoolPad := avgPool2dPad

/--
Global average pooling over spatial axes of a `C×H×W` tensor, producing a `C` vector.

PyTorch analogy: `torch.nn.functional.adaptive_avg_pool2d(x, output_size=1)` followed by flattening.
-/
def globalAvgPool2dChw
    (c h w : Nat)
    {h_c_pos : c > 0} {h_h_pos : h > 0} {h_w_pos : w > 0} :
    LayerDef (NN.Tensor.Shape.CHW c h w) (NN.Tensor.Shape.Vec c) :=
  { paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x =>
          _root_.Runtime.Autograd.Torch.globalAvgPool2dChw (m := m) (α := α) (c := c) (h := h) (w
            := w)
            h_c_pos h_h_pos h_w_pos x
  }

/--
Global average pooling over spatial axes of an `N×C×H×W` tensor, producing an `N×C` tensor.

PyTorch analogy: `torch.nn.functional.adaptive_avg_pool2d(x, output_size=1)` then reshape to `(N,
  C)`.
-/
def globalAvgPool2dNchw
    (n c h w : Nat)
    {h_n_pos : n > 0} {h_c_pos : c > 0} {h_h_pos : h > 0} {h_w_pos : w > 0} :
    LayerDef (NN.Tensor.Shape.NCHW n c h w) (.dim n (.dim c .scalar)) :=
  { paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x =>
          _root_.Runtime.Autograd.Torch.globalAvgPool2dNchw (m := m) (α := α) (n := n) (c := c)
            (h := h) (w := w)
            h_n_pos h_c_pos h_h_pos h_w_pos x
  }

/--
Multi-head self-attention layer for a sequence `(n × dModel) → (n × dModel)`.

This layer packs the four projection matrices `(Wq, Wk, Wv, Wo)` and calls the TorchLean attention
primitive. An optional boolean mask of shape `(n × n)` can be provided (e.g. causal masking).

PyTorch analogy: `torch.nn.MultiheadAttention(embed_dim=dModel, num_heads=numHeads)` in
  self-attention
mode (shape conventions differ; TorchLean uses explicit `n × dModel` tensors).
-/
def multiHeadAttention
    (batch n dModel numHeads headDim : Nat)
    {h1 : n ≠ 0}
    (seedW : Nat := 0)
    (mask : Option (Tensor Bool (.dim n (.dim n .scalar))) := none) :
    LayerDef (.dim batch (.dim n (.dim dModel .scalar)))
      (.dim batch (.dim n (.dim dModel .scalar))) :=
  let projDim := numHeads * headDim
  let wProjShape : Shape := .dim dModel (.dim projDim .scalar)
  let wOShape : Shape := .dim projDim (.dim dModel .scalar)
  let wq0 : Tensor Float wProjShape := Torch.Init.xavierW (outDim := dModel) (inDim := projDim)
    (seed := seedW)
  let wk0 : Tensor Float wProjShape := Torch.Init.xavierW (outDim := dModel) (inDim := projDim)
    (seed := seedW + 1)
  let wv0 : Tensor Float wProjShape := Torch.Init.xavierW (outDim := dModel) (inDim := projDim)
    (seed := seedW + 2)
  let wo0 : Tensor Float wOShape := Torch.Init.xavierW (outDim := projDim) (inDim := dModel) (seed
    := seedW + 3)
  { paramShapes := [wProjShape, wProjShape, wProjShape, wOShape]
    initParams := Torch.tlist4 wq0 wk0 wv0 wo0
    paramRequiresGrad := [true, true, true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun wq wk wv wo x =>
          TorchLean.multiHeadAttention (m := m) (α := α)
            (batch := batch) (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
            (h1 := h1)
            wq wk wv wo x (mask := mask)
  }

/-- Lift a single layer into a 1-layer sequential model. -/
def seq1 {σ τ : Shape} (l : LayerDef σ τ) : Seq σ τ :=
  .cons l (.id τ)

/-!
## Sequential model literal (`tlseq[...]`)

When writing small models, chaining `seq1` with `>>>` is a bit verbose:

```lean
TorchLean.NN.seq1 (TorchLean.NN.linear inDim hidDim) >>>
TorchLean.NN.seq1 TorchLean.NN.tanh >>>
TorchLean.NN.seq1 (TorchLean.NN.linear hidDim outDim)
```

This macro provides a compact, explicit alternative:

```lean
tlseq[
  TorchLean.NN.linear inDim hidDim,
  TorchLean.NN.tanh,
  TorchLean.NN.linear hidDim outDim
]
```

It expands to `seq1 ... >>> seq1 ... >>> ...`.
The syntax is intentionally namespaced to avoid colliding with other libraries.
-/

syntax (name := torchLeanSeqLit) "tlseq" "[" term,+ "]" : term

macro_rules
  | `(tlseq[$l]) => `(TorchLean.NN.seq1 $l)
  | `(tlseq[$l, $ls,*]) => `(TorchLean.NN.seq1 $l >>> tlseq[$ls,*])

end NN

end TorchLean
end Autograd
end Runtime

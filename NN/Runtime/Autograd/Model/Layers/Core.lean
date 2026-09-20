/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Autodiff
public import NN.Runtime.Autograd.Model.Module.RuntimeInit

/-!
# NN

`TorchLean.NN`: a compact `torch.nn`-style builder layer.

This module defines a small `torch.nn`-style builder layer for constructing shape-typed models.
It packages model-state shapes and initial values together with an execution-polymorphic forward
program, so example code does not have to spell `paramShapes := [...]` / `inputShapes := [...]`
everywhere.

## Main definitions

- `Layer σ τ` packages a shape-typed layer with explicit state (parameters and buffers) and
  a polymorphic `forward` program.
- `Seq σ τ` composes layers sequentially (PyTorch analogy: `torch.nn.Sequential`), written `f >>>
  g`.
- `Seq.Objective` bundles a `Seq` model together with a scalar loss, producing a
  `TorchLean.Module.ObjectiveDef` that the runtime training code can execute.

## PyTorch analogies

- `Layer` is like a small `nn.Module` definition, except parameters are an explicit list instead
  of fields, and the forward pass is a typed TorchLean program.
- `Mode` is like `module.train()` vs `module.eval()` (dropout and batchnorm-like layers branch on
  it).
- The `updateBuffers` mechanism is like updating non-gradient buffers (e.g. BatchNorm running
  stats).

The surface here is narrow by design: it supports TorchLean's executable model constructors and
training helpers without trying to mirror the full `torch.nn` API.

## References

- PyTorch `torch.nn`: https://pytorch.org/docs/stable/nn.html
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace Model

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Proofs.Autograd.Algebra

namespace Layers

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
A shape-typed layer definition with explicit model state and an execution-polymorphic forward
program.

`Layer σ τ` is the core building block used by `Seq` (sequential composition). It stores:
- the shapes of parameters and persistent buffers,
- initial values for that state (as `Float` tensors, for reproducible
  initialization),
- per-parameter `requiresGrad` flags, and
- a `forward` program that is polymorphic over the backend monad and scalar type.

PyTorch analogy: a small `nn.Module`, where:
- `stateShapes`/`initState` contain parameters and persistent buffers,
- `forward` corresponds to `Module.forward`,
- `updateBuffers` corresponds to updating things like `running_mean`/`running_var` in BatchNorm.
-/
structure Layer (σ τ : Shape) where
  /-- Layer label used by public model summaries. -/
  kind : String := "Layer"
  /-- Shapes of parameters and persistent buffers, in the order expected by `forward`. -/
  stateShapes : List Shape
  /-- Initial model state, stored as `Float` tensors for convenient initialization. -/
  initState : TorchLean.TensorPack Float stateShapes
  /--
  Optional storage-first initialization plan for executable `Float` backends.

  This does not replace `initState`: the tensor-valued initializers remain available to the
  specification and proof layers. The plan lets a runtime create equivalent parameter storage
  without first enumerating those tensors on the host.
  -/
  runtimeInit : Option (Runtime.Autograd.Model.Module.RuntimeInit.Plan stateShapes) :=
    if h : stateShapes = [] then some (h ▸ .nil) else none
  /--
  Gradient flags for model state (defaults to all `true`). Buffers use `false`.

  PyTorch analogy: `tensor.requires_grad_(...)` on parameters/buffers.
  -/
  requiresGrad : Array Bool := Array.replicate stateShapes.length true
  /--
  Validate static layer configuration before allocating runtime state or lowering a graph.

  Shape compatibility remains enforced by the type. This check is for value-level configuration
  such as a dropout probability that must belong to a finite numeric interval.
  -/
  validateConfig : Except String Unit := pure ()
  /--
  Optional buffer update function (used for running-statistics style layers).

  This is called during a forward pass (typically in `Mode.train`) to produce updated
    parameter/buffer
  state values. A canonical example is BatchNorm updating its `running_mean` / `running_var`
  buffers.
  -/
  updateBuffers :
    Option (
      Mode → ∀ {α : Type}, [TorchLean.Storage α] → [Context α] →
        TorchLean.TensorPack α stateShapes → Tensor α σ → IO (TorchLean.TensorPack α stateShapes)
    ) := none
  /-- Composite layers delegate runtime buffer updates to their nested forward programs. -/
  updatesBuffersInForward : Bool := false
  /--
  Forward pass as a typed TorchLean program.

  The program expects `(stateShapes ++ [σ])` inputs (model state, then the layer input) and
  produces an output of shape `τ`.
  -/
  forward :
    Mode → ∀ {α : Type}, [TorchLean.Storage α] → [Context α] →
      Runtime.Autograd.Model.Program α (stateShapes ++ [σ]) τ

/--
Update running statistics of any shape using momentum.

This implements an exponential moving average:

`next = (1 - momentum) * running + momentum * batch`.

PyTorch analogy: the update performed for `running_mean` / `running_var` in BatchNorm.
-/
def updateRunning {α : Type} [TorchLean.Storage α] [Context α] {s : Shape}
    (running batch : Tensor α s) (momentum : Tensor α .scalar) : Tensor α s :=
  let mom := momentum.item
  addSpec (scaleSpec running ((1 : α) - mom)) (scaleSpec batch mom)

/--
Convert the biased variance used by BatchNorm's training forward pass into the unbiased estimate
stored in its running buffer. For a singleton sample set there is no unbiased estimate; TorchLean
keeps the finite biased value rather than dividing by zero.
-/
def unbiasedRunningVariance {α : Type} [TorchLean.Storage α] [Context α] {s : Shape}
    (biased : Tensor α s) (sampleCount : Nat) : Tensor α s :=
  if sampleCount > 1 then
    scaleSpec biased ((sampleCount : α) / (sampleCount - 1 : Nat))
  else
    biased

/--
Compute per-channel mean and biased variance for a batched tensor.

The first two axes are batch and channel; every remaining axis is reduced. The result is a pair of
vectors indexed by channel. A running-variance update uses
`unbiasedRunningVariance vars (batch * spatial.size)` instead.
-/
def batchChannelStats {α : Type} [TorchLean.Storage α] [Context α]
    {batch channels : Nat} {spatial : Shape}
    (x : Tensor α (.dim batch (.dim channels spatial))) :
    Tensor α [channels] × Tensor α [channels] :=
  let spatialSize := Shape.size spatial
  let flatShape : Shape := .dim batch (.dim channels (.dim spatialSize .scalar))
  let xFlat : Tensor α flatShape := reshapeSpec x (by
    simp only [flatShape, spatialSize, Shape.size, Nat.mul_one])
  let sampleCount := batch * spatialSize
  let means : Tensor α [channels] :=
    Tensor.dim (fun ch =>
      let total :=
        (List.finRange batch).foldl (fun accBatch ni =>
          (List.finRange spatialSize).foldl (fun accSpatial i =>
            if hN : ni < batch then
              if hI : i < spatialSize then
                let channel := get (get xFlat ⟨ni, hN⟩) ch
                addSpec accSpatial (get channel ⟨i, hI⟩)
              else accSpatial
            else accSpatial
          ) accBatch
        ) (Tensor.scalar 0)
      divSpec total (Tensor.scalar (sampleCount : α)))
  let vars : Tensor α [channels] :=
    Tensor.dim (fun ch =>
      let mean := get means ch
      let total :=
        (List.finRange batch).foldl (fun accBatch ni =>
          (List.finRange spatialSize).foldl (fun accSpatial i =>
            if hN : ni < batch then
              if hI : i < spatialSize then
                let channel := get (get xFlat ⟨ni, hN⟩) ch
                let d := subSpec (get channel ⟨i, hI⟩) mean
                addSpec accSpatial (mulSpec d d)
              else accSpatial
            else accSpatial
          ) accBatch
        ) (Tensor.scalar 0)
      divSpec total (Tensor.scalar (sampleCount : α)))
  (means, vars)

namespace Layer

/--
Validate a layer's complete static contract.

Configuration checks supplied by the layer are combined with generic state-metadata and runtime
initializer checks so every execution path rejects malformed custom layers consistently.
-/
def validate {σ τ : Shape} (layer : Layer σ τ) : Except String Unit := do
  unless layer.requiresGrad.size = layer.stateShapes.length do
    throw s!"{layer.kind}: expected {layer.stateShapes.length} requiresGrad flags, \
      got {layer.requiresGrad.size}"
  layer.validateConfig
  match layer.runtimeInit with
  | some plan => plan.validate
  | none => pure ()

/--
Run a `Layer` forward given parameter refs and an input ref.

This is the "module forward" operation at the reference level.

PyTorch analogy: calling `layer(x)` where the layer's parameters are already allocated.
-/
def forwardRef {σ τ : Shape} (l : Layer σ τ) {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Torch.Ops (m := m) (α := α)]
    (mode : Mode)
    (ps : Torch.RefList (RefTy (m := m) (α := α)) l.stateShapes)
    (x : RefTy (m := m) (α := α) σ) : m (RefTy (m := m) (α := α) τ) :=
  Torch.CurriedRef.uncurry (ss := l.stateShapes ++ [σ]) (Ref := RefTy (m := m) (α := α))
    (l.forward mode (α := α) (m := m)) (Torch.RefList.append ps (.cons x .nil))

/--
Run a `Layer` on concrete tensors by lowering its forward program to a typed graph.

This is primarily used by runtime utilities (e.g. sequential `updateBuffers`) where we want to run
forward to obtain intermediate activations.

PyTorch analogy: running a forward pass eagerly on concrete tensors.
-/
def forwardTensor {σ τ : Shape} (l : Layer σ τ) (mode : Mode)
    {α : Type} [TorchLean.Storage α] [Context α]
    (ps : TorchLean.TensorPack α l.stateShapes) (x : Tensor α σ) : IO (Tensor α τ) := do
  match l.validate with
  | .error message => throw <| IO.userError message
  | .ok () => pure ()
  let graph ← Runtime.Autograd.Model.Autodiff.lowerToTypedGraph (α := α)
    (paramShapes := l.stateShapes) (inputShapes := [σ]) (τ := τ)
    (l.forward mode)
  let args : TorchLean.TensorPack α (l.stateShapes ++ [σ]) :=
    TorchLean.TensorPack.append (α := α) (ss₁ := l.stateShapes) (ss₂ := [σ]) ps
      (.cons x .nil)
  pure <| Runtime.Autograd.Torch.TypedGraph.forward graph args

end Layer
end Layers

end Model
end Autograd
end Runtime

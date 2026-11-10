/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Autodiff
public import TorchLean.NN.Init

/-!
# Layers

`TorchLean.nn.Layer` is the `torch.nn`-style builder layer for shape-typed models, and
`TorchLean.nn.Mode` is its execution mode. A layer packages model-state shapes and initial values
with an execution-polymorphic forward program, so example code does not have to spell
`paramShapes := [...]` / `inputShapes := [...]` everywhere.

## Main definitions

- `TorchLean.nn.Mode` selects training or inference behavior.
- `Layer σ τ` packages a shape-typed layer with its state (parameters and buffers) and a
  polymorphic `forward` program.
- `unbiasedRunningVariance` and `updateRunning` update batch normalization's running buffers
  from its batch statistics.

Sequential models (`TorchLean.nn.Sequential`) and their scalar objectives (`TorchLean.nn.Objective`)
are in `TorchLean.NN.Sequential`.

## PyTorch analogies

- `Layer` is like a small `nn.Module` definition whose parameters form a shape-indexed list and
  whose forward pass is a typed TorchLean program.
- `Mode` plays the role of `module.train()` and `module.eval()`: dropout and batch-normalization
  layers branch on it.
- `updateBuffers` updates buffers that take no gradient, such as BatchNorm's running statistics.

The surface covers what TorchLean's executable model constructors and training helpers need and
leaves out most of the `torch.nn` API.

## References

- PyTorch `torch.nn`: https://pytorch.org/docs/stable/nn.html
-/

@[expose] public section

namespace TorchLean.nn

/-!
### Mode

Layers whose training and inference behavior differ, such as dropout and batch normalization, read
the `Mode` argument that every forward pass receives.
-/

/--
Execution mode for layers that branch between training-time and inference-time behavior.

PyTorch analogy: `model.train()` / `model.eval()` (affects dropout, batchnorm, etc.).
-/
inductive Mode where
  /-- Training behavior: dropout is active and batch normalization uses batch statistics. -/
  | train
  /-- Inference behavior: dropout is off and batch normalization uses its running statistics. -/
  | eval
deriving Repr, DecidableEq

end TorchLean.nn

namespace TorchLean
namespace nn

open TorchLean.Tensor
open Ops

/-! ## Layer definitions -/

/--
A shape-typed layer definition: its model state and an execution-polymorphic forward program.

`Layer σ τ` is the core building block used by `nn.Sequential`. It stores:
- the shapes of parameters and persistent buffers,
- initial values for that state, as `Float` tensors for reproducible initialization,
- per-parameter `requiresGrad` flags, and
- a `forward` program that is polymorphic over the backend monad and scalar type.

PyTorch analogy: a small `nn.Module`, where:
- `stateShapes`/`initState` contain parameters and persistent buffers,
- `forward` corresponds to `Module.forward`,
- `updateBuffers` corresponds to updating things like `running_mean`/`running_var` in BatchNorm.
-/
structure Layer (σ τ : Shape) where
  /-- Layer label, shown in model summaries and error messages. -/
  kind : String := "Layer"
  /-- Shapes of parameters and persistent buffers, in the order expected by `forward`. -/
  stateShapes : List Shape
  /-- Initial model state, stored as `Float` tensors for convenient initialization. -/
  initState : TensorPack Float stateShapes
  /--
  Optional storage-first initialization plan for executable `Float` backends.

  The plan leaves `initState` in place for the specification and proof layers, and it lets a
  runtime create the parameter storage without first enumerating those tensors on the host.
  -/
  runtimeInit : Option (Init.Plan stateShapes) :=
    if h : stateShapes = [] then some (h ▸ .nil) else none
  /--
  Gradient flags for model state (defaults to all `true`). Buffers use `false`.

  PyTorch analogy: `tensor.requires_grad_(...)` on parameters/buffers.
  -/
  requiresGrad : Array Bool := Array.replicate stateShapes.length true
  /--
  Validate the static layer configuration before allocating runtime state or lowering a graph.

  The type enforces shape compatibility, and this check covers value-level configuration such as
  a dropout probability that must be finite and in `[0, 1]`.
  -/
  validateConfig : Except String Unit := pure ()
  /--
  Optional buffer update, for layers with running statistics such as BatchNorm's `running_mean`
  and `running_var` buffers. It maps the mode, the layer state and the layer input to the updated
  state. In `Mode.train`, `nn.Sequential.forwardState` calls it through the interpreter's
  `updateBuffers?` hook, and `nn.Sequential.updateBuffers` calls it during its reference replay.
  -/
  updateBuffers :
    Option (
      nn.Mode → ∀ {α : Type}, [Storage α] → [Context α] →
        TensorPack α stateShapes → Tensor α σ → IO (TensorPack α stateShapes)
    ) := none
  /--
  Whether the forward program updates the buffers itself, as a composite layer does through its
  nested layers. `nn.Sequential.forwardState` then skips the layer's `updateBuffers` hook.
  -/
  updatesBuffersInForward : Bool := false
  /--
  Forward pass as a typed TorchLean program.

  The program expects `(stateShapes ++ [σ])` inputs (model state, then the layer input) and
  produces an output of shape `τ`.
  -/
  forward :
    nn.Mode → ∀ {α : Type}, [Storage α] → [Context α] →
      Program α (stateShapes ++ [σ]) τ

/--
Exponential moving average with momentum `momentum`, for running statistics of any shape:

`next = (1 - momentum) * running + momentum * batch`.

PyTorch analogy: the update performed for `running_mean` / `running_var` in BatchNorm.
-/
def Layer.updateRunning {α : Type} [Storage α] [Add α] [Sub α] [Mul α] [One α]
    {s : Shape}
    (running batch : Tensor α s) (momentum : Tensor α .scalar) : Tensor α s :=
  let mom := momentum.item
  Tensor.scale running ((1 : α) - mom) + Tensor.scale batch mom

/--
Convert the biased variance used by BatchNorm's training forward pass into the unbiased estimate
stored in its running buffer. The factor `n / (n - 1)` is the exact ratio cast once into the
scalar. It is at most `2` and so finite, where a cast of the count `n` itself could overflow. With
fewer than two samples there is no unbiased estimate, and the biased value is returned unchanged,
with no division by zero.
-/
def Layer.unbiasedRunningVariance {α : Type} [Storage α] [Mul α] [RatCast α]
    {s : Shape}
    (biased : Tensor α s) (sampleCount : Nat) : Tensor α s :=
  if sampleCount > 1 then
    Tensor.scale biased ((sampleCount / (sampleCount - 1 : Nat) : Rat) : α)
  else
    biased

namespace Layer

/--
Validate a layer's complete static contract: one `requiresGrad` flag per state shape, the layer's
own `validateConfig`, and its runtime initialization plan, if any. Every execution path runs
these checks, so all of them reject a malformed custom layer alike.
-/
def validate {σ τ : Shape} (layer : Layer σ τ) : Except String Unit :=
  validateStatic layer.kind layer.requiresGrad layer.validateConfig layer.runtimeInit

/--
Run the forward program of a `Layer` on its state references and an input reference, the module
call at the reference level.

PyTorch analogy: calling `layer(x)` where the layer's parameters are already allocated.
-/
def forwardRef {σ τ : Shape} (l : Layer σ τ) {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    (mode : nn.Mode)
    (ps : Pack (Ref (m := m) (α := α)) l.stateShapes)
    (x : Ref (m := m) (α := α) σ) : m (Ref (m := m) (α := α) τ) :=
  CurriedRef.uncurry (ss := l.stateShapes ++ [σ]) (Ref := Ref (m := m) (α := α))
    (l.forward mode (α := α) (m := m)) (Pack.append ps (.cons x .nil))

/--
Run a `Layer` on concrete tensors: validate it, lower its forward program to a typed graph and
evaluate the graph. `nn.Sequential.updateBuffers` uses it for the intermediate activations of
its reference replay.

PyTorch analogy: running a forward pass eagerly on concrete tensors.
-/
def forwardTensor {σ τ : Shape} (l : Layer σ τ) (mode : nn.Mode)
    {α : Type} [Storage α] [Context α]
    (ps : TensorPack α l.stateShapes) (x : Tensor α σ) : IO (Tensor α τ) := do
  IO.ofExcept (l.validate)
  let graph ← Autodiff.lowerToTypedGraph (α := α)
    (paramShapes := l.stateShapes) (inputShapes := [σ]) (τ := τ)
    (l.forward mode)
  let args : TensorPack α (l.stateShapes ++ [σ]) :=
    Pack.append (ss₁ := l.stateShapes) (ss₂ := [σ]) ps
      (.cons x .nil)
  pure <| TypedGraph.WithData.forward graph args ()

end Layer

end nn
end TorchLean

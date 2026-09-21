/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/


module

public import NN.API.Macros
public import NN.API.Neural.Layers.Pooling
public import NN.API.Neural.Layers.Attention -- shake: keep
public import NN.API.Sample -- shake: keep

/-!
# Reusable Neural-Network Blocks

Configuration records for MLP and convolution blocks, plus residual and branching combinators over
already-built sequential models. The seeded block constructors live in `NN.API.Seeded`.
-/

@[expose] public section

namespace TorchLean
namespace nn

/--
MLP (multi-layer perceptron) configuration.

This builder produces a sequential stack of linear layers with activations and optional dropout.

PyTorch analogue: a hand-written `nn.Sequential(Linear(...), ReLU(), ..., Linear(...))`.
-/
structure MLP.Config where
  /-- Hidden layer widths (each entry creates a `Linear -> Activation` stage). -/
  hiddenWidths : List Nat := []
  /-- Activation used after each hidden linear layer. -/
  activation : Activation.Kind := .relu
  /-- Optional dropout probability after each activation. -/
  dropout? : Option Float := none

namespace MLP.Config

/--
Validate the input, output, hidden widths, and optional dropout probability before allocating
parameter seeds.
-/
def validate (config : MLP.Config) (inputWidth outputWidth : Nat) : Except String Unit := do
  if inputWidth = 0 then
    throw "MLP: input width must be positive"
  if outputWidth = 0 then
    throw "MLP: output width must be positive"
  if config.hiddenWidths.any (· = 0) then
    throw "MLP: hidden widths must be positive"
  match config.dropout? with
  | none => pure ()
  | some probability =>
      unless probability.isFinite && 0.0 <= probability && probability <= 1.0 do
        throw s!"MLP: dropout probability must be finite and in [0, 1], got {probability}"

end MLP.Config

/-- Convolution followed by an activation and optional dropout. -/
structure ConvBlock.Config (d : Nat) where
  /-- Convolution configuration. -/
  convolution : Convolution.Config d
  /-- Activation applied after convolution. -/
  activation : Activation.Kind := .relu
  /-- Optional dropout probability applied after activation. -/
  dropout? : Option Float := none

namespace ConvBlock.Config

/-- Validate convolution and dropout settings before allocating any parameter seeds. -/
def validate {d : Nat} (config : ConvBlock.Config d)
    (inputChannels : Nat) (input : Tensor Nat [d])
    (kind : String := "ConvBlock") : Except String Unit := do
  config.convolution.validate inputChannels input (kind := kind)
  match config.dropout? with
  | none => pure ()
  | some probability =>
      unless probability.isFinite && 0.0 <= probability && probability <= 1.0 do
        throw s!"{kind}: dropout probability must be finite and in [0, 1], got {probability}"

end ConvBlock.Config

/-- Convolution/activation followed by max pooling. -/
structure ConvPoolBlock.Config (d : Nat) where
  /-- Convolution and activation stage. -/
  block : ConvBlock.Config d
  /-- Pooling stage. -/
  pooling : Pooling.Config d

namespace ConvPoolBlock.Config

/-- Validate the convolution/dropout stage and the following pooling geometry. -/
def validate {d : Nat} (config : ConvPoolBlock.Config d)
    (inputChannels : Nat) (input : Tensor Nat [d])
    (kind : String := "ConvPoolBlock") : Except String Unit := do
  config.block.validate inputChannels input (kind := kind)
  config.pooling.validate config.block.convolution.outChannels
    (config.block.convolution.outputSpatial input) (kind := kind)

end ConvPoolBlock.Config

/-- Interpret an activation kind as an elementwise sequential model. -/
def activation {s : Spec.Shape} : Activation.Kind → Sequential s s
  | .relu => Sequential.fromLayer <| Runtime.Autograd.Model.Layers.relu (s := s)
  | .gelu => Sequential.fromLayer <| Runtime.Autograd.Model.Layers.gelu (s := s)
  | .silu => Sequential.fromLayer <| Runtime.Autograd.Model.Layers.silu (s := s)
  | .tanh => Sequential.fromLayer <| Runtime.Autograd.Model.Layers.tanh (s := s)
  | .sigmoid => Sequential.fromLayer <| Runtime.Autograd.Model.Layers.sigmoid (s := s)

/--
Residual (skip) connection as a single `Layer`.

Given `inner : Seq s s` this computes $x \mapsto \operatorname{inner}(x) + x$, the shape that
appears in ResNet (He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016) and in
every Transformer sublayer since Vaswani et al., *Attention Is All You Need* (NeurIPS 2017).

The `Layer` namespace holds the raw single-layer constructors; the same name without the namespace
returns a `Sequential`. That split is why there is no `residualLayer`: the return type belongs in
the namespace, not in the identifier.
-/
def Layer.residual {s : Spec.Shape} (inner : Sequential s s) : Layer s s :=
  let stateShapes := Runtime.Autograd.Model.Layers.Seq.stateShapes inner
  { kind := "Residual"
    updatesBuffersInForward := true
    stateShapes
    initState := Runtime.Autograd.Model.Layers.Seq.initState inner
    runtimeInit := Runtime.Autograd.Model.Layers.Seq.runtimeInit? inner
    requiresGrad := Runtime.Autograd.Model.Layers.Seq.requiresGrad inner
    validateConfig := Runtime.Autograd.Model.Layers.Seq.validate inner
    updateBuffers :=
      if Runtime.Autograd.Model.Layers.Seq.hasBufferUpdates inner then
        some (fun mode {α} _ _ state input =>
          Runtime.Autograd.Model.Layers.Seq.updateBuffers
            (α := α) (model := inner) mode state input)
      else
        none
    forward := fun mode {α} _ _ =>
      fun {m} _ _ =>
        Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := fun sh => TorchLean.Runtime.ValueRef (m := m) (α := α) sh)
          (ss := stateShapes ++ [s])
          (β := m (TorchLean.Runtime.ValueRef (m := m) (α := α) s))
          (fun arguments => do
            let (_state, input) :=
              Runtime.Autograd.Torch.RefList.splitLast
                (Ref := fun sh => TorchLean.Runtime.ValueRef (m := m) (α := α) sh)
                (ss := stateShapes) (τ := s) arguments
            let output ←
              Runtime.Autograd.Torch.CurriedRef.uncurry
                (Ref := fun sh => TorchLean.Runtime.ValueRef (m := m) (α := α) sh)
                (ss := stateShapes ++ [s])
                (β := m (TorchLean.Runtime.ValueRef (m := m) (α := α) s))
                (Runtime.Autograd.Model.Layers.Seq.forward inner (mode := mode) (α := α))
                arguments
            Runtime.Autograd.Torch.add (m := m) (α := α) (s := s) output input)
  }

/--
The same residual connection, wrapped as a one-element `Sequential` model.

Example:
```lean
def inner : nn.Builder (nn.Sequential [64] [64]) :=
  nn.Sequential![nn.linear 64 64, nn.relu]

-- `x + inner x`, the connection that made deep stacks trainable.
def model : nn.Builder (nn.Sequential [64] [64]) := do
  pure (nn.residual (← inner))
```
-/
def residual {s : Spec.Shape} (inner : Sequential s s) : Sequential s s :=
  nn.Sequential.fromLayer (Layer.residual inner)

/-!
## Branching (skip connections)

`Seq` is linear, but skip connections require two computations to consume the same input. The
generic constructor below owns the shared state plumbing; public blocks choose how to combine the
two outputs.
-/

/--
Run two sequential branches on the same input and combine their outputs.

Parameters and persistent buffers are stored as `state(f) ++ state(g)`. The combining operation is
polymorphic in the runtime, so eager execution and graph lowering share the same branch structure.
-/
def Layer.combineBranches {σ τ₁ τ₂ υ : Spec.Shape} (kind : String)
    (f : Sequential σ τ₁) (g : Sequential σ τ₂)
    (combine : ∀ {α : Type}, [TorchLean.Storage α] → [Context α] →
      ∀ {m : Type → Type}, [Monad m] →
        [Runtime.Autograd.Torch.Ops (m := m) (α := α)] →
        TorchLean.Runtime.ValueRef (m := m) (α := α) τ₁ →
        TorchLean.Runtime.ValueRef (m := m) (α := α) τ₂ →
        m (TorchLean.Runtime.ValueRef (m := m) (α := α) υ)) : Layer σ υ :=
  let firstStateShapes := Runtime.Autograd.Model.Layers.Seq.stateShapes f
  let secondStateShapes := Runtime.Autograd.Model.Layers.Seq.stateShapes g
  { kind := kind
    updatesBuffersInForward := true
    stateShapes := firstStateShapes ++ secondStateShapes
    initState :=
      TorchLean.TensorPack.append
        (α := Float) (ss₁ := firstStateShapes) (ss₂ := secondStateShapes)
        (Runtime.Autograd.Model.Layers.Seq.initState f)
        (Runtime.Autograd.Model.Layers.Seq.initState g)
    runtimeInit :=
      match Runtime.Autograd.Model.Layers.Seq.runtimeInit? f,
          Runtime.Autograd.Model.Layers.Seq.runtimeInit? g with
      | some fPlan, some gPlan => some (fPlan.append gPlan)
      | _, _ => none
    requiresGrad :=
      Runtime.Autograd.Model.Layers.Seq.requiresGrad f ++
        Runtime.Autograd.Model.Layers.Seq.requiresGrad g
    validateConfig := do
      Runtime.Autograd.Model.Layers.Seq.validate f
      Runtime.Autograd.Model.Layers.Seq.validate g
    updateBuffers :=
      if Runtime.Autograd.Model.Layers.Seq.hasBufferUpdates f ||
          Runtime.Autograd.Model.Layers.Seq.hasBufferUpdates g then
        some (fun mode {α} _ _ state input => do
          let (firstState, secondState) :=
            TorchLean.TensorPack.split
              (α := α) (ss₁ := firstStateShapes) (ss₂ := secondStateShapes) state
          let nextFirstState ←
            Runtime.Autograd.Model.Layers.Seq.updateBuffers
              (α := α) (model := f) mode firstState input
          let nextSecondState ←
            Runtime.Autograd.Model.Layers.Seq.updateBuffers
              (α := α) (model := g) mode secondState input
          pure <| TorchLean.TensorPack.append
            (α := α) (ss₁ := firstStateShapes) (ss₂ := secondStateShapes)
            nextFirstState nextSecondState)
      else
        none
    forward := fun mode {α} _ _ =>
      fun {m} _ _ =>
        Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := fun sh => TorchLean.Runtime.ValueRef (m := m) (α := α) sh)
          (ss := firstStateShapes ++ secondStateShapes ++ [σ])
          (β := m (TorchLean.Runtime.ValueRef (m := m) (α := α) υ))
          (fun arguments => do
            let (state, input) :=
              Runtime.Autograd.Torch.RefList.splitLast
                (Ref := fun sh => TorchLean.Runtime.ValueRef (m := m) (α := α) sh)
                (ss := firstStateShapes ++ secondStateShapes) (τ := σ) arguments
            let (firstState, secondState) :=
              Runtime.Autograd.Torch.RefList.split
                (Ref := fun sh => TorchLean.Runtime.ValueRef (m := m) (α := α) sh)
                (ss₁ := firstStateShapes) (ss₂ := secondStateShapes) state
            let firstOutput ←
              Runtime.Autograd.Torch.CurriedRef.uncurry
                (Ref := fun sh => TorchLean.Runtime.ValueRef (m := m) (α := α) sh)
                (ss := firstStateShapes ++ [σ])
                (β := m (TorchLean.Runtime.ValueRef (m := m) (α := α) τ₁))
                (Runtime.Autograd.Model.Layers.Seq.forward f (mode := mode) (α := α))
                (Runtime.Autograd.Torch.RefList.append firstState (.cons input .nil))
            let secondOutput ←
              Runtime.Autograd.Torch.CurriedRef.uncurry
                (Ref := fun sh => TorchLean.Runtime.ValueRef (m := m) (α := α) sh)
                (ss := secondStateShapes ++ [σ])
                (β := m (TorchLean.Runtime.ValueRef (m := m) (α := α) τ₂))
                (Runtime.Autograd.Model.Layers.Seq.forward g (mode := mode) (α := α))
                (Runtime.Autograd.Torch.RefList.append secondState (.cons input .nil))
            combine firstOutput secondOutput)
  }

/-- Two branches over one input, outputs added. The `Sequential` form is `nn.addBranches`. -/
def Layer.addBranches {σ τ : Spec.Shape} (f g : Sequential σ τ) : Layer σ τ :=
  Layer.combineBranches "AddBranches" f g fun {α} _ _ {m} _ _ yF yG =>
    Runtime.Autograd.Torch.add (m := m) (α := α) (s := τ) yF yG

/--
Combine two models with the same input/output shapes by summing their outputs.

This is a typed residual-add block: `addBranches f g` represents the model
$x \mapsto f(x) + g(x)$,
and its parameter list is the concatenation of the two branches’ parameter lists.

Example:
```lean
-- Two branches over the same input, outputs summed. Parameters are stored as the first branch's
-- list followed by the second's.
def model : nn.Builder (nn.Sequential [32] [8]) := do
  let wide ← nn.Sequential![nn.linear 32 8, nn.relu]
  let shortcut ← nn.linear 32 8
  pure (nn.addBranches wide shortcut)
```
-/
def addBranches {σ τ : Spec.Shape} (f g : Sequential σ τ) : Sequential σ τ :=
  nn.Sequential.fromLayer (Layer.addBranches f g)

/--
Concatenate two branch outputs along their first axis.

Both branches consume the same input. Their outputs must agree on every remaining dimension, and
the result records the sum of their first-axis extents in its type. This is the general typed skip
connection needed by encoder-decoder models; arbitrary outer batch axes can be added with
`mapLeading`.
-/
def Layer.concatBranches {σ s : Spec.Shape} {n m : Nat}
    (f : Sequential σ (s.prependDim n)) (g : Sequential σ (s.prependDim m)) :
    Layer σ (s.prependDim (n + m)) :=
  Layer.combineBranches "ConcatBranches" f g fun {α} _ _ {mRuntime} _ _ yF yG =>
    Runtime.Autograd.Torch.concatLeadingAxis
      (m := mRuntime) (α := α) (nDim := n) (mDim := m) (s := s) yF yG

/--
Concatenate two typed branches along their first output axis.

Example:
```lean
-- Concatenation along the first output axis: `[4, 8]` next to `[6, 8]` gives `[10, 8]`, and the
-- addition happens in the type rather than in a runtime shape check.
def model : nn.Builder (nn.Sequential [16] [10, 8]) := do
  let left ← nn.Sequential![nn.linear 16 (4 * 8), nn.reshape [4 * 8] [4, 8]]
  let right ← nn.Sequential![nn.linear 16 (6 * 8), nn.reshape [6 * 8] [6, 8]]
  pure (nn.concatBranches left right)
```
-/
def concatBranches {σ s : Spec.Shape} {n m : Nat}
    (f : Sequential σ (s.prependDim n)) (g : Sequential σ (s.prependDim m)) :
    Sequential σ (s.prependDim (n + m)) :=
  nn.Sequential.fromLayer (Layer.concatBranches f g)

end nn
end TorchLean

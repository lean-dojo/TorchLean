/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/


module

public import TorchLean.NN.Sequential.Literal
public import TorchLean.NN.Builder.Pooling
public import TorchLean.NN.Builder.Attention -- shake: keep

/-!
# Reusable Neural-Network Blocks

Configuration records for MLP and convolution blocks (`MLP.Config`, `ConvBlock.Config`,
`ConvPoolBlock.Config`), each with a `validate` check, and three combinators over already-built
sequential models: `residual`, `addBranches` and `concatBranches`. The seeded constructors that
read the records, such as `nn.mlp` and `nn.convPoolBlock`, live in `TorchLean.NN.Builder`.
-/

@[expose] public section

namespace TorchLean
namespace nn

/--
Configuration of a multi-layer perceptron. `nn.mlp` turns it into a sequential stack in which each
hidden width contributes a linear layer, the activation and optional dropout, and a final linear
layer maps to the output width.

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
Check that the input, output and hidden widths are positive and that the dropout probability, if
set, is finite and in $[0, 1]$. `nn.mlp` runs this check before it draws any parameter seed.
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
      unless functional.inUnitInterval probability do
        throw s!"MLP: dropout probability must be finite and in [0, 1], got {probability}"

end MLP.Config

/-- Convolution followed by an activation and optional dropout. -/
structure ConvBlock.Config (d : Nat) where
  /-- Convolution configuration. -/
  convolution : Conv.Config d
  /-- Activation applied after convolution. -/
  activation : Activation.Kind := .relu
  /-- Optional dropout probability applied after activation. -/
  dropout? : Option Float := none

namespace ConvBlock.Config

/--
Check the convolution against `inputChannels` and the spatial grid `input`, and check that the
dropout probability, if set, is finite and in $[0, 1]$.
-/
def validate {d : Nat} (config : ConvBlock.Config d)
    (inputChannels : Nat) (input : Tensor Nat [d])
    (kind : String := "ConvBlock") : Except String Unit := do
  config.convolution.validate inputChannels input (kind := kind)
  match config.dropout? with
  | none => pure ()
  | some probability =>
      unless functional.inUnitInterval probability do
        throw s!"{kind}: dropout probability must be finite and in [0, 1], got {probability}"

end ConvBlock.Config

/-- Convolution/activation followed by max pooling. -/
structure ConvPoolBlock.Config (d : Nat) where
  /-- Convolution and activation stage. -/
  block : ConvBlock.Config d
  /-- Pooling stage. -/
  pooling : Window d

namespace ConvPoolBlock.Config

/--
Validate the convolution stage with `ConvBlock.Config.validate`, then the pooling geometry on the
grid the convolution produces.
-/
def validate {d : Nat} (config : ConvPoolBlock.Config d)
    (inputChannels : Nat) (input : Tensor Nat [d])
    (kind : String := "ConvPoolBlock") : Except String Unit := do
  config.block.validate inputChannels input (kind := kind)
  config.pooling.validatePool config.block.convolution.outChannels
    (config.block.convolution.convOutputSpatial input) (kind := kind)

end ConvPoolBlock.Config

/--
Residual (skip) connection as a single `Layer`.

For `inner : Sequential s s` it computes $x \mapsto \operatorname{inner}(x) + x$, the shape that
appears in ResNet (He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016) and in
every Transformer sublayer since Vaswani et al., *Attention Is All You Need* (NeurIPS 2017).

Constructors in the `Layer` namespace return one raw `Layer`, and the same name outside the
namespace returns a `Sequential`: `residual` is the `Sequential` form of this one.
-/
def Layer.residual {s : Shape} (inner : Sequential s s) : Layer s s :=
  let stateShapes := Sequential.stateShapes inner
  { kind := "Residual"
    updatesBuffersInForward := true
    stateShapes
    initState := Sequential.initState inner
    runtimeInit := Sequential.runtimeInit? inner
    requiresGrad := Sequential.requiresGrad inner
    validateConfig := Sequential.validate inner
    updateBuffers :=
      if Sequential.hasBufferUpdates inner then
        some (fun mode {α} _ _ state input =>
          Sequential.updateBuffers
            (α := α) (model := inner) mode state input)
      else
        none
    forward := fun mode {α} _ _ =>
      fun {m} _ _ =>
        CurriedRef.curry
          (Ref := fun sh => Ops.Ref (m := m) (α := α) sh)
          (ss := stateShapes ++ [s])
          (β := m (Ops.Ref (m := m) (α := α) s))
          (fun arguments => do
            let (state, input) :=
              TorchLean.Pack.unsnoc
                (F := fun sh => Ops.Ref (m := m) (α := α) sh)
                (ss := stateShapes) (τ := s) arguments
            let output ←
              Sequential.forwardState inner mode state input
            Ops.apply (m := m) (α := α) (.add (s := s)) (output, input, ()))
  }

/--
The residual connection $x \mapsto \operatorname{inner}(x) + x$ as a one-layer `Sequential` model.

Example:
```lean
def inner : nn.Builder (nn.Sequential [64] [64]) :=
  nn.Sequential![nn.linear 64 64, nn.op .relu]

-- `inner x + x`, the connection that made deep stacks trainable.
def model : nn.Builder (nn.Sequential [64] [64]) := do
  pure (nn.residual (← inner))
```
-/
def residual {s : Shape} (inner : Sequential s s) : Sequential s s :=
  nn.Sequential.fromLayer (Layer.residual inner)

/-!
## Branching (skip connections)

A `Sequential` model is a linear chain, but a skip connection feeds one input to two computations.
`Layer.combineBranches` handles the shared state, and `addBranches` and `concatBranches` choose how
the two outputs combine.
-/

/--
Run two sequential branches on the same input and combine their outputs.

Parameters and persistent buffers are stored as `state(f) ++ state(g)`. The combining operation is
polymorphic in the runtime, so eager execution and graph lowering share the same branch structure.
-/
def Layer.combineBranches {σ τ₁ τ₂ υ : Shape} (kind : String)
    (f : Sequential σ τ₁) (g : Sequential σ τ₂)
    (combine : ∀ {α : Type}, [TorchLean.Storage α] → [Context α] →
      ∀ {m : Type → Type}, [Monad m] →
        [Ops (m := m) (α := α)] →
        Ops.Ref (m := m) (α := α) τ₁ →
        Ops.Ref (m := m) (α := α) τ₂ →
        m (Ops.Ref (m := m) (α := α) υ)) : Layer σ υ :=
  let firstStateShapes := Sequential.stateShapes f
  let secondStateShapes := Sequential.stateShapes g
  { kind := kind
    updatesBuffersInForward := true
    stateShapes := firstStateShapes ++ secondStateShapes
    initState :=
      TorchLean.Pack.append
        (ss₁ := firstStateShapes) (ss₂ := secondStateShapes)
        (Sequential.initState f)
        (Sequential.initState g)
    runtimeInit :=
      match Sequential.runtimeInit? f,
          Sequential.runtimeInit? g with
      | some fPlan, some gPlan => some (fPlan.append gPlan)
      | _, _ => none
    requiresGrad :=
      Sequential.requiresGrad f ++
        Sequential.requiresGrad g
    validateConfig := do
      Sequential.validate f
      Sequential.validate g
    updateBuffers :=
      if Sequential.hasBufferUpdates f ||
          Sequential.hasBufferUpdates g then
        some (fun mode {α} _ _ state input => do
          let (firstState, secondState) :=
            TorchLean.Pack.split
              (ss₁ := firstStateShapes) (ss₂ := secondStateShapes) state
          let nextFirstState ←
            Sequential.updateBuffers
              (α := α) (model := f) mode firstState input
          let nextSecondState ←
            Sequential.updateBuffers
              (α := α) (model := g) mode secondState input
          pure <| TorchLean.Pack.append
            (ss₁ := firstStateShapes) (ss₂ := secondStateShapes)
            nextFirstState nextSecondState)
      else
        none
    forward := fun mode {α} _ _ =>
      fun {m} _ _ =>
        CurriedRef.curry
          (Ref := fun sh => Ops.Ref (m := m) (α := α) sh)
          (ss := firstStateShapes ++ secondStateShapes ++ [σ])
          (β := m (Ops.Ref (m := m) (α := α) υ))
          (fun arguments => do
            let (state, input) :=
              TorchLean.Pack.unsnoc
                (F := fun sh => Ops.Ref (m := m) (α := α) sh)
                (ss := firstStateShapes ++ secondStateShapes) (τ := σ) arguments
            let (firstState, secondState) :=
              TorchLean.Pack.split
                (F := fun sh => Ops.Ref (m := m) (α := α) sh)
                (ss₁ := firstStateShapes) (ss₂ := secondStateShapes) state
            let firstOutput ←
              Sequential.forwardState f mode firstState input
            let secondOutput ←
              Sequential.forwardState g mode secondState input
            combine firstOutput secondOutput)
  }

/-- Two branches over one input, outputs added. The `Sequential` form is `nn.addBranches`. -/
def Layer.addBranches {σ τ : Shape} (f g : Sequential σ τ) : Layer σ τ :=
  Layer.combineBranches "AddBranches" f g fun {α} _ _ {m} _ _ yF yG =>
    Ops.apply (m := m) (α := α) (.add (s := τ)) (yF, yG, ())

/--
Sum of two models with the same input and output shapes: `addBranches f g` is the model
$x \mapsto f(x) + g(x)$, and its parameter list holds the parameters of `f` followed by those of
`g`.

Example:
```lean
-- A ReLU branch plus a linear shortcut, both from `[32]` to `[8]`.
def model : nn.Builder (nn.Sequential [32] [8]) := do
  let wide ← nn.Sequential![nn.linear 32 8, nn.op .relu]
  let shortcut ← nn.linear 32 8
  pure (nn.addBranches wide shortcut)
```
-/
def addBranches {σ τ : Shape} (f g : Sequential σ τ) : Sequential σ τ :=
  nn.Sequential.fromLayer (Layer.addBranches f g)

/--
Run two branches on the same input and concatenate their outputs along the first axis.

The outputs must agree on every other dimension, and the result type records the sum `n + m` of
their first-axis extents. Encoder-decoder models use this typed skip connection, and `mapLeading`
adds outer batch axes around it.
-/
def Layer.concatBranches {σ s : Shape} {n m : Nat}
    (f : Sequential σ (Shape.dim n s)) (g : Sequential σ (Shape.dim m s)) :
    Layer σ (Shape.dim (n + m) s) :=
  Layer.combineBranches "ConcatBranches" f g fun {α} _ _ {mRuntime} _ _ yF yG =>
    Ops.apply (m := mRuntime) (α := α)
        (.cat (s := .dim 0 s) ⟨0, by simp⟩ [n, m]) (yF, yG, ())

/--
Concatenate two typed branches along their first output axis.

Example:
```lean
-- `[4, 8]` next to `[6, 8]` gives `[10, 8]`, and the type checker does the addition.
def model : nn.Builder (nn.Sequential [16] [10, 8]) := do
  let left ← nn.Sequential![nn.linear 16 (4 * 8),
    nn.op (.reshape (s := [4 * 8]) (t := [4, 8]) rfl)]
  let right ← nn.Sequential![nn.linear 16 (6 * 8),
    nn.op (.reshape (s := [6 * 8]) (t := [6, 8]) rfl)]
  pure (nn.concatBranches left right)
```
-/
def concatBranches {σ s : Shape} {n m : Nat}
    (f : Sequential σ (Shape.dim n s)) (g : Sequential σ (Shape.dim m s)) :
    Sequential σ (Shape.dim (n + m) s) :=
  nn.Sequential.fromLayer (Layer.concatBranches f g)

end nn

end TorchLean

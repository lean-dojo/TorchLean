/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Activation
public import NN.Spec.Layers.Gnn

/-!
# Graph Neural Network Models

Spec-layer graph neural-network model definitions.

The layer-level message-passing math lives in `NN.Spec.Layers.Gnn`, in particular
`Spec.GCNLayerSpec`, `Spec.gcnLayerSpec`, and `Spec.gcnLayerBackwardSpec`. This module wires
those layers into a two-layer GCN with graph-level mean pooling and records the end-to-end shapes.

Reference (GCN):

- Kipf and Welling, "Semi-Supervised Classification with Graph Convolutional Networks" (2017):
  https://arxiv.org/abs/1609.02907

PyTorch ecosystem analogies:

- `torch_geometric.nn.GCNConv` for the layer-level GCN operator,
- global mean pooling as in `torch_geometric.nn.global_mean_pool`.

## Implementation status

No API builder implements this model (`nn.*` has no graph layers), no runtime code imports it,
and no theorem relates it to anything else. It is a standalone reference specification.
-/

@[expose] public section


namespace Models

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Shape
open Activation

variable {α : Type} [TorchLean.Storage α] [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-!
## 2-layer GCN with gradients

Forward is:

1. `Z₁ = GCN₁(X)` (linear message passing)
2. `H₁ = ReLU(Z₁)`
3. `H₂ = GCN₂(H₁)`
4. `y  = mean_nodes(H₂)` (graph-level readout)

Diagram (single graph, no batching):

```
X : (n × inDim)
   └─ GCN₁ ─→ Z₁ : (n × hidDim) ─ ReLU ─→ H₁ : (n × hidDim)
                                 └─ GCN₂ ─→ H₂ : (n × outDim)
                                              └─ mean over nodes ─→ y : (outDim)
```

Backward mirrors this structure:
- "mean nodes" broadcasts the `outDim` gradient back to `n×outDim` and scales by `1/n`,
- each GCN layer uses the matrix-calculus rules in `Spec.gcnLayerBackwardSpec`,
- ReLU gates gradients by `ReLU'`.
-/

/-- A 2-layer GCN "model spec" for a fixed graph with `n` nodes.

`GCNLayerSpec` packages the per-layer parameters (including the adjacency/normalization choice),
so the model here is just two such layers composed with a nonlinearity and a readout.
-/
structure GCN2Spec (n inDim hidDim outDim : Nat) (α : Type) [TorchLean.Storage α] where
  /-- First GCN layer: `inDim → hidDim`. -/
  inputLayer : Spec.GCNLayerSpec n inDim hidDim α
  /-- Second GCN layer: `hidDim → outDim`. -/
  outputLayer : Spec.GCNLayerSpec n hidDim outDim α

/-- Forward pass for the 2-layer GCN with a graph-level mean pooling readout.

Input:
- `x : n × inDim` node features.

Output:
- `y : outDim` graph embedding produced by averaging node embeddings.

The `h_n : n > 0` assumption is only used to make the mean pooling well-defined (division by `n`).
-/
def GCN2Spec.forward
  {n inDim hidDim outDim : Nat}
  (m : GCN2Spec n inDim hidDim outDim α)
  (x : Tensor α [n, inDim])
  (h_n : n > 0) :
  Tensor α [outDim] :=
  let hiddenFeatures : Tensor α [n, hidDim] :=
    reluSpec (Spec.gcnLayerSpec (α := α) (n := n) (inDim := inDim) (outDim := hidDim)
      m.inputLayer x)
  let outputFeatures : Tensor α [n, outDim] :=
    Spec.gcnLayerSpec (α := α) (n := n) (inDim := hidDim) (outDim := outDim)
      m.outputLayer hiddenFeatures
  have hn0 : n ≠ 0 := Nat.ne_of_gt h_n
  have hLeadingAxis : Shape.HasNonemptyAxis 0 (Shape.dim n (Shape.dim outDim Shape.scalar)) :=
    Shape.hasNonemptyAxisZeroOfNe hn0
  reduceMean 0 outputFeatures hLeadingAxis.proof

/-- Gradients for both layers of `GCN2Spec`. -/
structure GCN2Grads (n inDim hidDim outDim : Nat) (α : Type) [TorchLean.Storage α] where
  /-- Gradients for the `inDim → hidDim` input GCN layer. -/
  inputLayer : Spec.GCNLayerParameterGradients n inDim hidDim α
  /-- Gradients for the `hidDim → outDim` output GCN layer. -/
  outputLayer : Spec.GCNLayerParameterGradients n hidDim outDim α

/-- Backward/VJP for `GCN2Spec.forward`.

This is written in the same "spec style" as the rest of TorchLean:
- recompute small intermediates instead of depending on runtime caches,
- apply VJPs in reverse order,
- keep shapes explicit.

PyTorch analogy: this corresponds to what autograd would do for a graph built from
`GCNConv → ReLU → GCNConv → global_mean_pool`, but expressed as a pure function.
-/
def GCN2Spec.backward
  {n inDim hidDim outDim : Nat}
  (m : GCN2Spec n inDim hidDim outDim α)
  (x : Tensor α [n, inDim])
  (gradOutput : Tensor α [outDim])
  (h_n : n > 0) :
  (GCN2Grads n inDim hidDim outDim α ×
   Tensor α [n, inDim]) :=

  have hn0 : n ≠ 0 := Nat.ne_of_gt h_n

  -- Recompute forward intermediates (small and keeps the backward spec self-contained).
  let hiddenPreActivation : Tensor α [n, hidDim] :=
    Spec.gcnLayerSpec (α := α) (n := n) (inDim := inDim) (outDim := hidDim)
      m.inputLayer x
  let hiddenFeatures : Tensor α [n, hidDim] :=
    reluSpec hiddenPreActivation

  -- Backprop through node-mean pooling:
  -- y = (1/n) * Σᵢ outputFeatures[i]
  have hB : Shape.CanBroadcastTo (.dim outDim .scalar) (.dim n (.dim outDim .scalar)) := by
    apply Shape.CanBroadcastTo.expand_dims
    apply Shape.CanBroadcastTo.dim_eq
    exact Shape.CanBroadcastTo.scalar

  let outputFeatureGrad : Tensor α [n, outDim] :=
    scaleSpec (broadcastTo hB gradOutput) (1 / (n : α))

  -- Output layer backward.
  let outputLayerGrads :=
    Spec.gcnLayerBackwardSpec (α := α) (n := n) (inDim := hidDim) (outDim := outDim)
      m.outputLayer hiddenFeatures outputFeatureGrad hn0

  -- ReLU backward: dZ = dH ⊙ ReLU'(Z).
  let hiddenPreActivationGrad : Tensor α [n, hidDim] :=
    mulSpec outputLayerGrads.inputGradient (reluDerivSpec hiddenPreActivation)

  -- Input layer backward.
  let inputLayerGrads :=
    Spec.gcnLayerBackwardSpec (α := α) (n := n) (inDim := inDim) (outDim := hidDim)
      m.inputLayer x hiddenPreActivationGrad hn0

  let grads : GCN2Grads n inDim hidDim outDim α :=
    { inputLayer := inputLayerGrads.parameters
      outputLayer := outputLayerGrads.parameters }

  (grads, inputLayerGrads.inputGradient)

end Models

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Layers.Activation
public import TorchLean.Tensor.Layers.GCN

/-!
# Graph Neural Network Models

A two-layer graph convolutional network (GCN) with graph-level mean pooling, built from the layer
specs in `TorchLean.Tensor.Layers.GCN`: `GCN.Layer` and `GCN.Layer.forward`. The types
fix the end-to-end shapes.

The shape is Kipf and Welling's two-layer model, eq. (9), `Â ReLU(Â X W₀) W₁`, with three changes
we made for a graph-level spec. Each layer adds a bias `b`. There's no softmax at the end. And a
mean over the nodes turns the node features into one vector per graph, where the paper
classifies nodes. We don't build the renormalized `Â = D̃^{-1/2} (A + I) D̃^{-1/2}` of Section 2.2
here: each layer stores whatever `A` the caller passes.

Reference (GCN):

- Thomas N. Kipf and Max Welling, "Semi-Supervised Classification with Graph Convolutional
  Networks", ICLR 2017, arXiv:1609.02907, eq. (2) for the layer and eq. (9) for the two-layer
  model: https://arxiv.org/abs/1609.02907

PyTorch ecosystem analogies:

- `torch_geometric.nn.GCNConv` for the layer-level GCN operator,
- global mean pooling as in `torch_geometric.nn.global_mean_pool`.

## Implementation status

No API builder implements this model (`nn.*` has no graph layers), no runtime code imports it,
and no theorem relates it to anything else. It is a standalone reference specification.
-/

@[expose] public section


namespace TorchLean

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-!
## Two-layer GCN

The forward pass:

1. `Z₁ = GCN₁(X)` (message passing, `A · X · W + b`)
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
-/

/-- A two-layer GCN for graphs with `n` nodes.

Each `GCN.Layer` holds its own adjacency operator `A` next to `W` and `b`. The caller picks the
normalization and, for one fixed graph, passes the same `A` to both layers.
-/
structure GCN (n inDim hidDim outDim : Nat) (α : Type) [Storage α] where
  /-- First GCN layer: `inDim → hidDim`. -/
  inputLayer : GCN.Layer n inDim hidDim α
  /-- Second GCN layer: `hidDim → outDim`. -/
  outputLayer : GCN.Layer n hidDim outDim α

/-- Forward pass of the two-layer GCN: node features `x : n × inDim` go through both layers, and
mean pooling over the nodes gives a graph embedding of size `outDim`.

The readout is `Tensor.mean` over the node axis: the sum over the nodes divided once by `n`, or
by one for an empty graph.
-/
def GCN.forward
  {n inDim hidDim outDim : Nat}
  (m : GCN n inDim hidDim outDim α)
  (x : Tensor α [n, inDim]) :
  Tensor α [outDim] :=
  let hiddenFeatures : Tensor α [n, hidDim] :=
    Tensor.relu (Layer.forward (α := α) (n := n) (inDim := inDim) (outDim := hidDim)
      m.inputLayer x)
  let outputFeatures : Tensor α [n, outDim] :=
    Layer.forward (α := α) (n := n) (inDim := hidDim) (outDim := outDim)
      m.outputLayer hiddenFeatures
  castShape (Tensor.mean outputFeatures [⟨0, by simp⟩] false) (Shape.reduce_singleton _ _)

end TorchLean

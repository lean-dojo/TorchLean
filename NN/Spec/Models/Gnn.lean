/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Activation
public import NN.Spec.Layers.Gnn

/-!
# Gnn

GNN models (spec layer).

We provide a compact 2-layer GCN with a graph-level mean pooling readout.

This file is intentionally "model glue": the actual message-passing math lives in
`NN.Spec.Layers.Gnn` (in particular `Spec.GCNLayerSpec`, `Spec.gcn_layer_spec`,
and `Spec.gcn_layer_backward_spec`).  Here we focus on wiring layers together and documenting
the end-to-end shapes.

Reference (GCN):

- Kipf and Welling, "Semi-Supervised Classification with Graph Convolutional Networks" (2017):
  https://arxiv.org/abs/1609.02907

PyTorch ecosystem analogies:

- `torch_geometric.nn.GCNConv` for the layer-level GCN operator,
- global mean pooling as in `torch_geometric.nn.global_mean_pool`.
-/

@[expose] public section


namespace Models

open Spec
open Tensor
open Shape
open Activation

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

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

Backward follows this structure literally:
- "mean nodes" broadcasts the `outDim` gradient back to `n×outDim` and scales by `1/n`,
- each GCN layer uses the matrix-calculus rules in `Spec.gcn_layer_backward_spec`,
- ReLU gates gradients by `ReLU'`.
-/

/-- A 2-layer GCN "model spec" for a fixed graph with `n` nodes.

`GCNLayerSpec` packages the per-layer parameters (including the adjacency/normalization choice),
so the model here is just two such layers composed with a nonlinearity and a readout.
-/
structure GCN2Spec (n inDim hidDim outDim : Nat) (α : Type) where
  /-- First GCN layer: `inDim → hidDim`. -/
  l1 : Spec.GCNLayerSpec n inDim hidDim α
  /-- Second GCN layer: `hidDim → outDim`. -/
  l2 : Spec.GCNLayerSpec n hidDim outDim α

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
  (x : Tensor α (.dim n (.dim inDim .scalar)))
  (h_n : n > 0) :
  Tensor α (.dim outDim .scalar) :=
  let h1 : Tensor α (.dim n (.dim hidDim .scalar)) :=
    reluSpec (Spec.gcnLayerSpec (α := α) m.l1 x)
  let h2 : Tensor α (.dim n (.dim outDim .scalar)) :=
    Spec.gcnLayerSpec (α := α) m.l2 h1
  have hn0 : n ≠ 0 := Nat.ne_of_gt h_n
  have h_axis0 : Shape.valid_axis_inst 0 (Shape.dim n (Shape.dim outDim Shape.scalar)) :=
    Shape.validAxisInstZeroAlt hn0
  reduceMeanAuto 0 h_axis0 h2

/-- Per-layer gradients returned by `GCNLayerSpec` backward.

This mirrors the tuple returned by `Spec.gcn_layer_backward_spec`:
- `dA`: gradient w.r.t. the adjacency-like operator used by the layer,
- `dW`: gradient w.r.t. the weight matrix,
- `db`: gradient w.r.t. the bias vector.
-/
structure GCNLayerGrads (n inDim outDim : Nat) (α : Type) where
  /-- d A. -/
  dA : Tensor α (.dim n (.dim n .scalar))
  /-- d W. -/
  dW : Tensor α (.dim inDim (.dim outDim .scalar))
  /-- db. -/
  db : Tensor α (.dim outDim .scalar)

/-- Gradients for both layers of `GCN2Spec`. -/
structure GCN2Grads (n inDim hidDim outDim : Nat) (α : Type) where
  /-- l 1. -/
  l1 : GCNLayerGrads n inDim hidDim α
  /-- l 2. -/
  l2 : GCNLayerGrads n hidDim outDim α

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
  (x : Tensor α (.dim n (.dim inDim .scalar)))
  (grad_output : Tensor α (.dim outDim .scalar))
  (h_n : n > 0) :
  (GCN2Grads n inDim hidDim outDim α ×
   Tensor α (.dim n (.dim inDim .scalar))) :=

  have hn0 : n ≠ 0 := Nat.ne_of_gt h_n

  -- Recompute forward intermediates (small and keeps the backward spec self-contained).
  let z1 : Tensor α (.dim n (.dim hidDim .scalar)) := Spec.gcnLayerSpec (α := α) m.l1 x
  let h1 : Tensor α (.dim n (.dim hidDim .scalar)) := reluSpec z1
  let h2 : Tensor α (.dim n (.dim outDim .scalar)) := Spec.gcnLayerSpec (α := α) m.l2 h1

  -- Backprop through node-mean pooling:
  -- y = (1/n) * Σᵢ h2[i]
  have hB : Shape.CanBroadcastTo (.dim outDim .scalar) (.dim n (.dim outDim .scalar)) := by
    apply Shape.CanBroadcastTo.expand_dims
    apply Shape.CanBroadcastTo.dim_eq
    apply Shape.CanBroadcastTo.scalar_to_any .scalar

  let grad_h2 : Tensor α (.dim n (.dim outDim .scalar)) :=
    scaleSpec (broadcastTo hB grad_output) (1 / (n : α))

  -- Layer 2 backward.
  let (dA2, dW2, db2, grad_h1) :=
    Spec.gcnLayerBackwardSpec (α := α) m.l2 h1 grad_h2 hn0

  -- ReLU backward: dZ1 = dH1 ⊙ ReLU'(Z1)
  let grad_z1 : Tensor α (.dim n (.dim hidDim .scalar)) :=
    mulSpec grad_h1 (reluDerivSpec z1)

  -- Layer 1 backward.
  let (dA1, dW1, db1, grad_x) :=
    Spec.gcnLayerBackwardSpec (α := α) m.l1 x grad_z1 hn0

  let grads : GCN2Grads n inDim hidDim outDim α :=
    { l1 := { dA := dA1, dW := dW1, db := db1 }
      l2 := { dA := dA2, dW := dW2, db := db2 } }

  (grads, grad_x)

end Models

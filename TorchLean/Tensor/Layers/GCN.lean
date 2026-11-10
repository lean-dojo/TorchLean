/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.AdjacentSwap
public import TorchLean.Tensor.Contract

/-!
# Graph neural network layers (spec layer)

This file defines a GCN-style graph convolution layer. Neighbor aggregation is the shared tensor
matrix multiplication `Tensor.matmul`.

## Message passing (the common core idea)

Most GNN layers aggregate neighbor features along the graph structure and then may apply a learned
transformation and a nonlinearity. The aggregation here is a product with a matrix `A : (n×n)`:

`Agg(A, H) = A · H`.

One formula covers the common conventions:

- if `A` is the raw adjacency, the layer sums neighbors,
- if `A` is normalized (e.g. `D^{-1/2} (A + I) D^{-1/2}`), it applies the "GCN normalization",
- if `A` includes edge weights, it takes a weighted sum.

## GCN layer (one very common choice)

A GCN-style layer here is:

  `H' = A · H · W + b`

where:
- `A : (n×n)` is an adjacency-like matrix (often normalized, and often with self-loops),
- `H : (n×inDim)` are node features,
- `W : (inDim×outDim)` and `b : outDim` are trainable parameters.

The same formula is the algebraic core of PyTorch Geometric's `GCNConv` once a concrete `A` (raw
adjacency, `D^{-1/2} (A + I) D^{-1/2}`, and so on) and a batching convention are fixed.

The layer is eq. (2) of Kipf and Welling, "Semi-Supervised Classification with Graph
Convolutional Networks", ICLR 2017, arXiv:1609.02907, Sec. 2, entered by hand. The activation `σ`
of that equation belongs to the model (`TorchLean.GCN.forward` puts a ReLU between its two
layers), and the renormalized `D̃^{-1/2} (A + I) D̃^{-1/2}` of Sec. 2.2 is something the caller
passes as `A`. Eq. (2) has no bias. We keep one because `GCNConv` adds it by default
(`bias=True`, PyG docs:
https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html).
The product is `(A · X) · W`. Over the reals that equals `A · (X · W)`, but the two associations
can round differently in a binary format, and the spec fixes the first.
-/

@[expose] public section


namespace TorchLean

open TorchLean.Tensor

variable {α : Type} [Storage α] [Add α] [Mul α] [Zero α]

/-- The adjacency operator and the trainable parameters of one GCN-style layer.

`A` sits in the record because some uses fix it per graph while others learn it (a learned
normalization, for example), and one record serves both.
-/
structure GCN.Layer (n inDim outDim : Nat) (α : Type) [Storage α] where
  /-- The (possibly normalized) adjacency operator acting on the `n` nodes. -/
  adjacency : Tensor α [n, n]
  /-- The weight matrix mapping input features to output features. -/
  weight : Tensor α [inDim, outDim]
  /-- The bias vector, broadcast across nodes. -/
  bias : Tensor α [outDim]

/-- Forward pass of a GCN-style layer, `Y = (A · X) · W + b`, with the bias `b` added to the row of
every node. Any normalization or self-loop convention belongs in the choice of `A`.
-/
def GCN.Layer.forward {n inDim outDim : Nat}
  (layer : Layer n inDim outDim α)
  (x : Tensor α [n, inDim]) :
  Tensor α [n, outDim] :=
  let ax : Tensor α [n, inDim] :=
    Tensor.matmul layer.adjacency x
  let axw : Tensor α [n, outDim] :=
    Tensor.matmul ax layer.weight
  let hB : Shape.CanBroadcastTo (.dim outDim .scalar) (.dim n (.dim outDim .scalar)) := by
    apply Shape.CanBroadcastTo.expand_dims
    apply Shape.CanBroadcastTo.dim_eq
    exact Shape.CanBroadcastTo.scalar
  axw + Tensor.broadcast layer.bias _ hB.dims hB.broadcastDims

end TorchLean

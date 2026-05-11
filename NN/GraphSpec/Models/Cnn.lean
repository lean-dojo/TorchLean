/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.Primitives.Vision

/-!
# GraphSpec model: CNN (2 convs)

This is a GraphSpec version of the classic small CNN:

`Conv2D → ReLU → MaxPool2D → Conv2D → ReLU → MaxPool2D → Flatten → Linear`

Notes:
- This is still a **chain** model (no skip connections), so it fits the sequential GraphSpec core
  (`NN.GraphSpec.Core`) and can be built using `>>>`.
- Parameter shapes are derived in the type:
  `(K1,b1,K2,b2,W,b)` where `K` are Conv kernels and `W` is the linear head matrix.

## Shape bookkeeping (why all the Nat arithmetic?)

The point of GraphSpec is that the *shape interface is part of the type*. Convolution and pooling
therefore bake their output shapes into the type, using the standard formulas:

- Conv2D:
  `outH = (inH + 2*padding - kH) / stride + 1` (and similarly for `outW`)
- MaxPool2D:
  `outH = (inH - kH) / stride + 1` (and similarly for `outW`)

This file defines small helper abbreviations (`outH/outW/poolH/poolW`) so that the overall
classifier head shape (the input dimension to the final `Linear`) is computed once and reused.

## Why not put this under `GraphSpec.DAG`?

You *can* express a chain model as a DAG term, but the sequential DSL is the simpler interface for
pure pipelines. The DAG language is reserved for models that fundamentally need sharing or
multi-input nodes (e.g. residual adds).
-/

@[expose] public section


namespace NN
namespace GraphSpec
namespace Models

open Spec
open NN.Tensor

/-! ### Conv/pool output size helpers -/

/-- Convolution output height formula (standard DL convention). -/
abbrev outH (inH kH stride padding : Nat) : Nat :=
  (inH + 2 * padding - kH) / stride + 1

/-- Convolution output width formula (standard DL convention). -/
abbrev outW (inW kW stride padding : Nat) : Nat :=
  (inW + 2 * padding - kW) / stride + 1

/-- MaxPool output height formula. -/
abbrev poolH (inH kH stride : Nat) : Nat :=
  (inH - kH) / stride + 1

/-- MaxPool output width formula. -/
abbrev poolW (inW kW stride : Nat) : Nat :=
  (inW - kW) / stride + 1

/--
Final feature-map height after:

`conv1 → pool1 → conv2 → pool2`.
 -/
abbrev featH
    (inH kH stride1 padding1 poolKH poolStride1 stride2 padding2 poolStride2 : Nat) : Nat :=
  poolH (outH (poolH (outH inH kH stride1 padding1) poolKH poolStride1) kH stride2 padding2) poolKH
    poolStride2

/--
Final feature-map width after:

`conv1 → pool1 → conv2 → pool2`.
 -/
abbrev featW
    (inW kW stride1 padding1 poolKW poolStride1 stride2 padding2 poolStride2 : Nat) : Nat :=
  poolW (outW (poolW (outW inW kW stride1 padding1) poolKW poolStride1) kW stride2 padding2) poolKW
    poolStride2

/--
Total flattened feature size for the classifier head.

If the second conv produces `c2` channels and the final spatial size is `featH × featW`,
then the flattened vector length is `(CHW c2 featH featW).size`.
 -/
abbrev featSize
    (c2 inH inW kH kW stride1 padding1 stride2 padding2 poolKH poolKW poolStride1 poolStride2 : Nat)
      : Nat :=
  (Shape.CHW c2
    (featH inH kH stride1 padding1 poolKH poolStride1 stride2 padding2 poolStride2)
    (featW inW kW stride1 padding1 poolKW poolStride1 stride2 padding2 poolStride2)).size

/--
2-conv CNN GraphSpec model.

Conventions:
- both conv layers share `(kH,kW)` but can differ in stride/padding.
- pooling uses kernel `(poolKH,poolKW)` and can differ in stride per pooling site.

Parameter layout (type-level):

- `(K1,b1)` for conv1,
- `(K2,b2)` for conv2,
- `(W,b)` for the final linear head.

Input/output:

- input `x : CHW inC inH inW` (no batch dimension),
- output logits `: Vec outDim`.
-/
def cnn2
    (inC c1 c2 outDim inH inW kH kW stride1 padding1 stride2 padding2 poolKH poolKW poolStride1
      poolStride2 : Nat)
    {h_inC : inC ≠ 0} {h_c1 : c1 ≠ 0} {_h_c2 : c2 ≠ 0}
    {h_kH : kH ≠ 0} {h_kW : kW ≠ 0}
    {h_poolKH : poolKH ≠ 0} {h_poolKW : poolKW ≠ 0}
    {h_poolStride1 : poolStride1 ≠ 0} {h_poolStride2 : poolStride2 ≠ 0} :
    Graph
      [ Shape.OIHW c1 inC kH kW, Shape.Vec c1
      , Shape.OIHW c2 c1 kH kW, Shape.Vec c2
      , Shape.Mat outDim (featSize c2 inH inW kH kW stride1 padding1 stride2 padding2 poolKH poolKW
        poolStride1 poolStride2)
      , Shape.Vec outDim ]
      (Shape.CHW inC inH inW) (Shape.Vec outDim) :=
  Graph.conv2d (inC := inC) (outC := c1) (kH := kH) (kW := kW) (stride := stride1) (padding :=
    padding1)
    (inH := inH) (inW := inW) (h_inC := h_inC) (h_kH := h_kH) (h_kW := h_kW)
  >>>
  Graph.relu
    (Shape.CHW c1
      (outH inH kH stride1 padding1)
      (outW inW kW stride1 padding1))
  >>>
  Graph.maxPool2d
    (kH := poolKH) (kW := poolKW)
    (inH := outH inH kH stride1 padding1)
    (inW := outW inW kW stride1 padding1)
    (inC := c1) (stride := poolStride1)
    (h_kH := h_poolKH) (h_kW := h_poolKW) (hStride := h_poolStride1)
  >>>
  Graph.conv2d
    (inC := c1) (outC := c2) (kH := kH) (kW := kW) (stride := stride2) (padding := padding2)
    (inH := poolH (outH inH kH stride1 padding1) poolKH poolStride1)
    (inW := poolW (outW inW kW stride1 padding1) poolKW poolStride1)
    (h_inC := h_c1) (h_kH := h_kH) (h_kW := h_kW)
  >>>
  Graph.relu
    (Shape.CHW c2
      (outH (poolH (outH inH kH stride1 padding1) poolKH poolStride1) kH stride2 padding2)
      (outW (poolW (outW inW kW stride1 padding1) poolKW poolStride1) kW stride2 padding2))
  >>>
  Graph.maxPool2d
    (kH := poolKH) (kW := poolKW)
    (inH := outH (poolH (outH inH kH stride1 padding1) poolKH poolStride1) kH stride2 padding2)
    (inW := outW (poolW (outW inW kW stride1 padding1) poolKW poolStride1) kW stride2 padding2)
    (inC := c2) (stride := poolStride2)
    (h_kH := h_poolKH) (h_kW := h_poolKW) (hStride := h_poolStride2)
  >>>
  Graph.flatten
    (Shape.CHW c2
      (featH inH kH stride1 padding1 poolKH poolStride1 stride2 padding2 poolStride2)
      (featW inW kW stride1 padding1 poolKW poolStride1 stride2 padding2 poolStride2))
  >>>
  -- Linear head: interpret the flattened feature vector as `Vec (featSize ...)`.
  Graph.linear
    (inDim := featSize c2 inH inW kH kW stride1 padding1 stride2 padding2 poolKH poolKW poolStride1
      poolStride2)
    (outDim := outDim)

/--
The same 2-conv CNN, but exposed as a DAG `Model` via the structural lowering
`LowerToDAG.Graph.toDAGModelZeroInit`.

This lets “DAG-only” downstream tooling consume this architecture even though it is authored as a
sequential `Graph` pipeline.

Initialization: all-zero parameters (see `LowerToDAG.Graph.toDAGModelZeroInit`).
 -/
def cnn2DAGModelZeroInit
    (inC c1 c2 outDim inH inW kH kW stride1 padding1 stride2 padding2 poolKH poolKW poolStride1
      poolStride2 : Nat)
    {h_inC : inC ≠ 0} {h_c1 : c1 ≠ 0} {h_c2 : c2 ≠ 0}
    {h_kH : kH ≠ 0} {h_kW : kW ≠ 0}
    {h_poolKH : poolKH ≠ 0} {h_poolKW : poolKW ≠ 0}
    {h_poolStride1 : poolStride1 ≠ 0} {h_poolStride2 : poolStride2 ≠ 0} :
    DAG.Model
      [ Shape.OIHW c1 inC kH kW, Shape.Vec c1
      , Shape.OIHW c2 c1 kH kW, Shape.Vec c2
      , Shape.Mat outDim (featSize c2 inH inW kH kW stride1 padding1 stride2 padding2 poolKH poolKW
        poolStride1 poolStride2)
      , Shape.Vec outDim ]
      [Shape.CHW inC inH inW]
      (Shape.Vec outDim) :=
  LowerToDAG.Graph.toDAGModelZeroInit <|
    cnn2
      (inC := inC) (c1 := c1) (c2 := c2) (outDim := outDim)
      (inH := inH) (inW := inW)
      (kH := kH) (kW := kW)
      (stride1 := stride1) (padding1 := padding1)
      (stride2 := stride2) (padding2 := padding2)
      (poolKH := poolKH) (poolKW := poolKW) (poolStride1 := poolStride1) (poolStride2 :=
        poolStride2)
      (h_inC := h_inC) (h_c1 := h_c1) (_h_c2 := h_c2)
      (h_kH := h_kH) (h_kW := h_kW)
      (h_poolKH := h_poolKH) (h_poolKW := h_poolKW)
      (h_poolStride1 := h_poolStride1) (h_poolStride2 := h_poolStride2)

end Models
end GraphSpec
end NN

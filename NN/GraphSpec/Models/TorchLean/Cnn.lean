/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.NN
import Mathlib.Algebra.Order.Algebra

/-!
# TorchLean-executable model: CNN

This file provides a small CNN constructor using the TorchLean `Seq` builder.

We keep this model “PyTorch-shaped”: it is a literal chain of conv/pool/flatten/linear.

For residual / DAG-style CNNs, see GraphSpec-backed models like `resnet18`.
-/

@[expose] public section


namespace NN
namespace GraphSpec
namespace Models
namespace TorchLean

open NN.Tensor

/--
2-conv CNN in TorchLean:

`Conv2D → ReLU → MaxPool2D → Conv2D → ReLU → MaxPool2D → Flatten → Linear`.

Initialization is deterministic and matches the current GraphSpec primitive convention:
- each “parameterized layer occurrence” gets an index `i = 0,1,2,...`,
- and seeds are `seedW/seedK = 2*i`, `seedB = 2*i + 1`.

So for this CNN:
- Conv1 uses `(seedK,seedB) = (0,1)`
- Conv2 uses `(2,3)`
- Linear head uses `(seedW,seedB) = (4,5)`
-/
def cnn2
    (inC c1 c2 outDim inH inW kH kW stride1 padding1 stride2 padding2 poolKH poolKW poolStride1
      poolStride2 : Nat)
    {h_inC : inC ≠ 0} {h_c1 : c1 ≠ 0} {_h_c2 : c2 ≠ 0}
    {h_kH : kH ≠ 0} {h_kW : kW ≠ 0}
    {h_poolKH : poolKH ≠ 0} {h_poolKW : poolKW ≠ 0} :
    _root_.Runtime.Autograd.TorchLean.NN.Seq (Shape.CHW inC inH inW) (Shape.Vec outDim) :=
  let outH1 : Nat := (inH + 2 * padding1 - kH) / stride1 + 1
  let outW1 : Nat := (inW + 2 * padding1 - kW) / stride1 + 1
  let poolH1 : Nat := (outH1 - poolKH) / poolStride1 + 1
  let poolW1 : Nat := (outW1 - poolKW) / poolStride1 + 1
  let outH2 : Nat := (poolH1 + 2 * padding2 - kH) / stride2 + 1
  let outW2 : Nat := (poolW1 + 2 * padding2 - kW) / stride2 + 1
  let poolH2 : Nat := (outH2 - poolKH) / poolStride2 + 1
  let poolW2 : Nat := (outW2 - poolKW) / poolStride2 + 1
  let featSize : Nat := (Shape.CHW c2 poolH2 poolW2).size
  _root_.Runtime.Autograd.TorchLean.NN.seq1
      (_root_.Runtime.Autograd.TorchLean.NN.conv2d
        (inC := inC) (outC := c1) (kH := kH) (kW := kW) (stride := stride1) (padding := padding1)
        (inH := inH) (inW := inW) (h1 := h_inC) (h2 := h_kH) (h3 := h_kW)
        (seedK := 0) (seedB := 1))
    >>>
    _root_.Runtime.Autograd.TorchLean.NN.seq1 _root_.Runtime.Autograd.TorchLean.NN.relu
    >>>
    _root_.Runtime.Autograd.TorchLean.NN.seq1
      (_root_.Runtime.Autograd.TorchLean.NN.maxPool2d
        (kH := poolKH) (kW := poolKW) (inH := outH1) (inW := outW1) (inC := c1) (stride :=
          poolStride1)
        (h1 := h_poolKH) (h2 := h_poolKW))
    >>>
    _root_.Runtime.Autograd.TorchLean.NN.seq1
      (_root_.Runtime.Autograd.TorchLean.NN.conv2d
        (inC := c1) (outC := c2) (kH := kH) (kW := kW) (stride := stride2) (padding := padding2)
        (inH := poolH1) (inW := poolW1) (h1 := h_c1) (h2 := h_kH) (h3 := h_kW)
        (seedK := 2) (seedB := 3))
    >>>
    _root_.Runtime.Autograd.TorchLean.NN.seq1 _root_.Runtime.Autograd.TorchLean.NN.relu
    >>>
    _root_.Runtime.Autograd.TorchLean.NN.seq1
      (_root_.Runtime.Autograd.TorchLean.NN.maxPool2d
        (kH := poolKH) (kW := poolKW) (inH := outH2) (inW := outW2) (inC := c2) (stride :=
          poolStride2)
        (h1 := h_poolKH) (h2 := h_poolKW))
    >>>
    _root_.Runtime.Autograd.TorchLean.NN.seq1
      (_root_.Runtime.Autograd.TorchLean.NN.flatten (s := Shape.CHW c2 poolH2 poolW2))
    >>>
    _root_.Runtime.Autograd.TorchLean.NN.seq1
      (_root_.Runtime.Autograd.TorchLean.NN.linear featSize outDim (seedW := 4) (seedB := 5))

end TorchLean
end Models
end GraphSpec
end NN

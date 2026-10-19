/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Ops.Convolution
public import NN.Runtime.Autograd.Torch.Core.Ops.Elementwise
public import NN.Runtime.Autograd.Torch.Core.Ops.Indexing
public import NN.Runtime.Autograd.Torch.Core.Ops.Layers
public import NN.Runtime.Autograd.Torch.Core.Ops.LinearAlgebra
public import NN.Runtime.Autograd.Torch.Core.Ops.Pooling
public import NN.Runtime.Autograd.Torch.Core.Ops.ShapeReduction

/-!
# Eager Tensor Operations

Typed eager operations record nodes on the session's tape through a shared dispatch layer, which
runs the CPU or the CUDA kernel of each operation. The focused
modules group operations by behavior; this umbrella preserves the complete eager operation API.
-/

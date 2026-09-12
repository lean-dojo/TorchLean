/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Core.ActivationsLoss
public import NN.Runtime.Autograd.Engine.Core.Backward
public import NN.Runtime.Autograd.Engine.Core.Base
public import NN.Runtime.Autograd.Engine.Core.ConvPool
public import NN.Runtime.Autograd.Engine.Core.Elementwise
public import NN.Runtime.Autograd.Engine.Core.Indexing
public import NN.Runtime.Autograd.Engine.Core.Linear
public import NN.Runtime.Autograd.Engine.Core.Neural
public import NN.Runtime.Autograd.Engine.Core.Shape

/-!
Core eager-engine operations.

The split submodules keep shape manipulation, elementwise kernels, neural-network layers, linear
algebra, and convolution/pooling logic in separate files while preserving one import point.
-/

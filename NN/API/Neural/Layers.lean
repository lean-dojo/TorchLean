/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/


module

public import NN.API.Neural.Layers.Attention
public import NN.API.Neural.Layers.Convolution
public import NN.API.Neural.Layers.Pooling

/-!
# Neural Layers

Named configurations for attention, convolution, and pooling. The seeded constructors that consume
them live in `NN.API.Seeded`; spatial operators state the trailing axes they consume, while
`batchShape` records any axes mapped pointwise by the layer.
-/

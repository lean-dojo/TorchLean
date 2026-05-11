/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.Models.TorchLean

/-!
# TorchLean executable model zoo

This module re-exports TorchLean’s small runnable model constructors (MLP/CNN/Transformer/etc.).
It is the executable model-zoo counterpart to the pure specs in `NN.Spec.Models.*`.

The implementations live under `NN.GraphSpec.Models.TorchLean.*`, because they are architecture
constructors. The runtime namespace stays focused on execution machinery: ops, backends, sessions,
losses, optimizers, and training loops.

We keep this short `NN.TorchLeanModels` facade so examples can write `TorchLeanModels.mlp` without
threading a long namespace path everywhere.
-/

@[expose] public section


namespace NN
namespace TorchLeanModels

export _root_.NN.GraphSpec.Models.TorchLean
  (mlp autoencoder cnn2 softmaxRegression mlpClassifier transformerBlock
   fno1d fno1dParamShapes
   resnet18Model resnet18Program resnet18InitParams
  )

end TorchLeanModels
end NN

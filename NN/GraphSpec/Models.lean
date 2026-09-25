/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.Models.Mlp
public import NN.GraphSpec.Models.Cnn
public import NN.GraphSpec.Models.ResidualLinear

/-!
# GraphSpec Model Catalog

Import for the graph-authored example models in `NN.GraphSpec.Models.*`. Their structure is
itself a typed `Chain` or `DAG.Model`, so the same architecture can be lowered to TorchLean and
its graph shape reasoned about. Pure reference specifications (Transformer, ViT, Mamba, S4, UNet,
VAE/VQ-VAE/GAN, classical baselines) live in `NN.Spec.Models` and are not imported here; runnable
scripts live in `NN.Examples.Models`.

The set is a coverage ladder, not a catalog:

1. `mlp`: the smallest sequential `Chain` with a typed parameter ABI.
2. `cnn`: a caller-supplied feature chain (for example convolutions) followed by flattening and a
   linear classifier.
3. `residualLinear`: a minimal `DAG.Model` with a real skip connection.

The first two are sequential chains and the third is DAG-native. To turn a chain into a
zero-initialized `DAG.Model`, import `NN.GraphSpec.Chain.ToDAG.Model` and use
`LowerToDAG.Chain.toDAGModelZeroInit`.

See also:
- `NN/GraphSpec/README.md` for the overall layout and motivation.
- `NN.GraphSpec.Chain.Lowering` for the sequential DSL and lowering helpers.
- `NN.GraphSpec.DAG` for the DAG term language and its primitives.
-/

@[expose] public section

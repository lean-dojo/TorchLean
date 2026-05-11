/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Models
public import NN.GraphSpec.Models.Mlp
public import NN.GraphSpec.Models.Cnn
public import NN.GraphSpec.Models.ResidualLinear
public import NN.GraphSpec.Models.Resnet18
public import NN.GraphSpec.Models.TorchLean

/-!
# GraphSpec Model Catalog

Curated architecture import for GraphSpec users.

This file is the place a user should import when they are thinking “models as architectures.” It
re-exports the pure model specifications from `NN.Spec.Models` and the graph-native examples in
this directory.

We still keep the source files split by semantic layer:

- `NN.Spec.Models.*` contains pure mathematical/reference specifications such as Transformer, ViT,
  Mamba, S4, UNet, VAE/VQ-VAE/GAN, and classical baselines.
- `NN.GraphSpec.Models.*` contains graph-authored models whose structure is itself a typed
  `Graph`/`DAG.Model`, so we can compile the same architecture to TorchLean and reason about the
  graph shape.
- `NN.GraphSpec.Models.TorchLean.*` contains executable TorchLean constructors for models that are
  already useful as reusable autograd programs.
- `NN.Examples.Models.*` contains runnable scripts and training demos.

That split avoids circular dependencies and keeps proofs honest, while this umbrella gives users a
single architecture-facing import that includes both the broad spec catalog and the graph-authored
coverage ladder.

The current set is intentionally a coverage ladder, not an exhaustive catalog:

1. `mlp`: smallest sequential typed parameter ABI.
2. `cnn2`: sequential vision pipeline with convolution/pooling shape arithmetic.
3. `residualLinear`: minimal DAG model with a real skip connection.
4. `ResNet18.model`: larger DAG model with repeated residual blocks and projection shortcuts.

The examples intentionally mix two authoring styles, but they have one conceptual endpoint:
`DAG.Model`.

- sequential `Graph` models for simple pipelines,
- DAG-native `Model` terms for residual / shared-structure examples.

`NN.GraphSpec.Models` is the single import for these GraphSpec-specific examples, regardless of
which GraphSpec surface syntax they were authored in.

Included examples:
- `NN.GraphSpec.Models.mlp` (minimal sequential MLP) and
  `NN.GraphSpec.Models.mlpDAGModelZeroInit` (the same chain lowered to DAG),
- `NN.GraphSpec.Models.cnn2` (sequential chain) and `NN.GraphSpec.Models.cnn2DAGModelZeroInit`
  (the same model, lowered to DAG),
- DAG-native models such as `NN.GraphSpec.Models.residualLinear` and
  `NN.GraphSpec.Models.ResNet18.model`.

See also:
- `NN.GraphSpec/README.md` for the overall layout and motivation.
- `NN.GraphSpec.Core` for the sequential DSL and lowering helpers.
- `NN.GraphSpec.DAG` for the canonical DAG IR and semantics.

If you are new to this directory, a good order is:

1. `Models.mlp`,
2. `Models.cnn2`,
3. `Models.residualLinear` as the minimal DAG/skip-connection example,
4. `Models.ResNet18.model` as the larger residual architecture.
-/

@[expose] public section

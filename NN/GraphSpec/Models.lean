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

Curated architecture import for GraphSpec users.

This file is the place a user should import when they are thinking “models as architectures.” It
re-exports the pure model specifications from `NN.Spec.Models` and the graph-native examples in
this directory.

We still keep the source files split by semantic layer:

- `NN.Spec.Models.*` contains pure mathematical/reference specifications such as Transformer, ViT,
  Mamba, S4, UNet, VAE/VQ-VAE/GAN, and classical baselines.
- `NN.GraphSpec.Models.*` contains graph-authored models whose structure is itself a typed
  `Chain`/`DAG.Model`, so we can lower the same architecture to TorchLean and reason about the
  graph shape.
- `NN.Examples.Models.*` contains runnable scripts and training examples.

That split avoids circular dependencies. This umbrella is the architecture-facing import that
includes both the broad spec catalog and the graph-authored coverage ladder.

The current set is intentionally a coverage ladder, not an exhaustive catalog:

1. `mlp`: smallest sequential typed parameter ABI.
2. `cnn`: a typed feature chain followed by a linear classifier.
3. `residualLinear`: minimal DAG model with a real skip connection.

The examples intentionally mix two authoring styles, but they have one conceptual endpoint:
`DAG.Model`.

- sequential `Chain` models for simple pipelines,
- DAG-native `Model` terms for residual / shared-structure examples.

`NN.GraphSpec.Models` is the single import for these GraphSpec-specific examples, regardless of
which GraphSpec surface syntax they were authored in.

Included examples:
- `NN.GraphSpec.Models.mlp` (minimal sequential MLP),
- `NN.GraphSpec.Models.cnn` (classifier over a feature chain),
- the DAG-native `NN.GraphSpec.Models.residualLinear` model.

Use `LowerToDAG.Chain.toDAGModelZeroInit` to lower any sequential model to a zero-initialized DAG.

See also:
- `NN.GraphSpec/README.md` for the overall layout and motivation.
- `NN.GraphSpec.Core` for the sequential DSL and lowering helpers.
- `NN.GraphSpec.DAG` for the canonical DAG IR and semantics.

If you are new to this directory, a good order is:

1. `Models.mlp`,
2. `Models.cnn`,
3. `Models.residualLinear` as the minimal DAG/skip-connection example,
-/

@[expose] public section

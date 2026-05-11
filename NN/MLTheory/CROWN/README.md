# CROWN / LiRPA in TorchLean (folder guide)

This folder contains TorchLean's CROWN/LiRPA style bound propagation code and related certificate
checkers.

## Where To Start

1. `core.lean`: interval boxes (`Box`) and the basic affine form container (`AffineVec`). These are
   the data structures used by the rest of the folder.

2. `mlp.lean`: a small vector in, vector out CROWN development for MLP style networks. Includes ReLU
   relaxations, IBP bounds, and affine propagation.

3. `graph.lean`: graph-based LiRPA development over the project's op-tagged IR graphs. It implements
   an engine that can run IBP and parts of affine bound propagation over graphs.

4. `operators/`: transfer rules used by the graph engine for activations, arithmetic, reductions, and
   slicing. `operators.lean` imports the standard operator set.

5. `soundness_proofs.lean`: soundness lemmas for parts of the CROWN development, mainly over `ℝ`.

6. `runtime_ops.lean`: runtime definitions used by the graph engine, kept separate to avoid Mathlib
   overhead.

## Subfolders

- `operators/`: op specific bound transfer rules (IBP and/or affine relaxations).
- `propagation/`: experimental or specialized propagation routines (e.g. backward passes).
- `Extras/`: optional helpers and proof toolboxes that are not required by the main engine.

## Optional modules (`Extras/`)

- `Extras/interval_lemmas.lean`: basic interval-arithmetic lemmas over `ℝ`.
- `Extras/alpha_config.lean`: data structures for alpha optimized relaxations (alpha CROWN style).
- `Extras/FloatIntegration.lean`: experiments integrating bound propagation with explicit rounding.
- `Extras/fp32.lean`: convenience wrappers specializing the graph engine to `TorchLean.Floats.FP32`.

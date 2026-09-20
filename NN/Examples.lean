/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Quickstart
public import NN.Examples.Data.Loaders
public import NN.Examples.Models
public import NN.Examples.Interop.PyTorch
public import NN.Examples.DeepDives
public import NN.Examples.Factorization
public import NN.Examples.Functional.Transcendentals
public import NN.Examples.Optimization
public import NN.Examples.Verification
public import NN.Examples.BugZoo.All

/-!
# `NN.Examples`

Single umbrella for TorchLean examples.

Import this module to compile every library-style example: introductory examples, models, data
loaders, interop, widgets, mathematical deep dives, and verification workflows.

Typical usage:

* Build the full example surface: `lake build NNExamples`
* Run model examples through the CLI:
  `lake exe torchlean mlp --steps 10`
* Run CUDA-only model examples with both build-time and runtime CUDA selection:
  `lake -R -K cuda=true exe torchlean gpt_adder --device cuda --steps 1`

The heavier examples remain in their subdirectories, so each can still be built independently. The
standalone CLI root `NN.Examples.RunnerMain` is intentionally excluded because importing an
executable root would introduce a global `main`; Lake builds that target separately.
-/

@[expose] public section

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Core
public import NN.MLTheory.CROWN.Graph.Engine
public import NN.MLTheory.CROWN.Graph.Backward
public import NN.MLTheory.CROWN.Graph.Theorems

/-!
# CROWN Graph

Public façade for the graph-based LiRPA/CROWN engine.

The implementation is split into focused modules:
- `Graph/Core`: graph aliases, flattened affine-bound state, and propagation state records,
- `Graph/Engine`: the executable engine: IBP, derivative IBP, forward affine CROWN, and
  objective-dependent backward CROWN,
- `Graph/Backward`: a narrow import façade for users who want the backward-CROWN chapter name,
- `Graph/Theorems`: shape, dimension, and enclosure lemmas used by proof modules.

Import this module when you want the complete graph engine API.
-/

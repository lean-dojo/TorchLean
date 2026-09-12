/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine.Base
public import NN.MLTheory.CROWN.Graph.Engine.IBP
public import NN.MLTheory.CROWN.Graph.Engine.Refinement
public import NN.MLTheory.CROWN.Graph.Engine.Derivatives
public import NN.MLTheory.CROWN.Graph.Engine.Affine
public import NN.MLTheory.CROWN.Graph.Engine.CROWN.Run
public import NN.MLTheory.CROWN.Graph.Engine.BackwardObjective

/-!
# Flat LiRPA Engine

Executable graph engine for interval propagation, affine forms, and CROWN-style transfer rules.

- `Engine.Base`: flat vectors, boxes, parameter stores, and shared tensor helpers.
- `Engine.IBP`: interval bound propagation.
- `Engine.Refinement`: optional input subdivision with a shared work budget.
- `Engine.Derivatives`: first- and second-derivative interval passes.
- `Engine.Affine`: affine-form propagation.
- `Engine.CROWN`: forward CROWN/DeepPoly bounds.
- `Engine.BackwardObjective`: objective-dependent backward CROWN.
-/

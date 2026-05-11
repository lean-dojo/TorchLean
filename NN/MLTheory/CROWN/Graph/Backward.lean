/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine

/-!
# CROWN Graph Backward Objective API

Compatibility façade for the objective-dependent backward CROWN functions. The implementation lives
with the forward affine engine in `Graph/Engine`, because the backward pass reuses the same private
affine-transfer helpers. Keeping this module lets downstream code import the backward chapter by
name without exposing those helpers as public API.
-/

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.Chain.ToDAG.Model
public import NN.GraphSpec.Chain.ToDAG.Semantics

/-!
# Conversion of sequential GraphSpec chains to DAGs

This is the canonical import for structural chain-to-DAG conversion and the associated DAG model
constructors. Term construction lives in `Chain.ToDAG.Core`; parameter initialization and model
packaging live in `Chain.ToDAG.Model`. `Chain.ToDAG.Semantics` proves that the converted term has
the direct chain interpretation for every parameter pack and input, including custom primitives.
This is a pure-semantics theorem; executable primitive programs require separate agreement proofs.
-/

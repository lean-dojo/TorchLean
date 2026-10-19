/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TypedGraph.Core
public import NN.Runtime.Autograd.TypedGraph.Compiled
public import NN.Runtime.Autograd.TypedGraph.GraphM

/-!
# Typed Graph Autograd Runtime

Typed graphs are the representation TorchLean's autograd transforms and proofs work on: a
`GraphData` value whose nodes are indexed by their shapes. The trainer does not use them; it
records on the one eager tape in `NN.Runtime.Autograd.Engine`. Hand-written loops can, as the
explicit-precision quickstart does with a typed VJP and `nn.sgdStep`.

Their maintained execution interface is `compileChecked`: one checked forward evaluation saves
local reverse programs and an indexed tensor context. `Compiled.backwardDenseAllFrom` seeds
one input or intermediate node and visits its certified dependencies when storage equality is
available. `Compiled.backwardDenseFrom` accepts a complete seed pack and retains dense semantics.

The modules are:

- `TypedGraph.Core`: packages executable `GraphData`, lowers it to a tape, and exposes dense
  reverse-mode entry points;
- `TypedGraph.Compiled`: indexed checked execution, dependency-aware single-output backward,
  and an explicit all-seeds dense backward;
- `TypedGraph.GraphM`: a typed builder DSL for authoring `GraphData` without manually threading
  dependent node indices.

`Compiled.toTape` and `lowerToTapeChecked` lower a typed graph onto the ordinary tape.
The raw Tape engine materializes a full prefix contribution per node, so its dense backward
can take quadratic time and allocation. Ordinary checked TypedGraph differentiation uses
the compiled interface.
-/

@[expose] public section

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

This is the runtime umbrella for TorchLean's typed graph execution path.

The typed graph path is the middle layer between:

- the low-level dynamic tape engine in `NN.Runtime.Autograd.Engine`, and
- the user-facing TorchLean session/model API in `NN.Runtime.Autograd.Model`.

Its maintained execution interface is `compileChecked`: one checked forward evaluation saves
local reverse programs and an indexed tensor context. `Compiled.backwardDenseAllFrom` seeds
any input or intermediate node; `Compiled.backwardDenseFrom` accepts a complete seed pack.

The modules are:

- `TypedGraph.Core`: packages executable `GraphData`, lowers it to a tape, and exposes dense
  reverse-mode entry points;
- `TypedGraph.Compiled`: indexed checked execution and ordered dense-gradient results;
- `TypedGraph.GraphM`: a typed builder DSL for authoring `GraphData` without manually threading
  dependent node indices.

`Compiled.toTape` and `lowerToTapeChecked` retain the raw Tape compatibility interface.
The raw Tape engine materializes a full prefix contribution per node, so its dense backward
can take quadratic time and allocation. Ordinary checked TypedGraph differentiation uses
the compiled interface.

Forward-only execution of the separate operation-tagged `NN.IR.Graph` representation lives under
`NN.Runtime.Autograd.IRExec`. Its `ForwardGraph` stores only forward operations, so it cannot be
mistaken for the differentiable typed graphs exposed here.
-/

@[expose] public section

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Compiled.Core
public import NN.Runtime.Autograd.Compiled.GraphM
public import NN.Runtime.Autograd.Compiled.IRExec

/-!
# Compiled Autograd Runtime

This is the runtime umbrella for TorchLean's compiled execution path.

The compiled path is the middle layer between:

- the low-level dynamic tape engine in `NN.Runtime.Autograd.Engine`, and
- the user-facing TorchLean session/model API in `NN.Runtime.Autograd.TorchLean`.

It has three pieces:

- `Compiled.Core`: packages executable `GraphData`, compiles it to a tape, and exposes dense
  reverse-mode entry points;
- `Compiled.GraphM`: a typed builder DSL for authoring `GraphData` without manually threading
  dependent node indices; and
- `Compiled.IRExec`: lowers shared `NN.IR.Graph` programs into executable `GraphData` for
  forward execution.

Correctness proofs for the IR bridge live under
`NN.Runtime.Autograd.Compiled.IRExec.Correctness`. They are intentionally kept in their own
umbrella so ordinary runtime imports do not have to elaborate the full semantic-equivalence proof.

## Trust Boundary

`IRExec` compiles **forward** semantics. Its generated node payloads intentionally use
sentinel JVP/VJP implementations, so training-style gradients should continue to use the
autograd-capable `GraphM` / TorchLean compiled backend rather than the shared-IR execution bridge.
-/

@[expose] public section

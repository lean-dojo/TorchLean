/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Entrypoint.API
public import NN.Entrypoint.Floats
public import NN.Entrypoint.GraphSpec
public import NN.Entrypoint.IR
public import NN.Entrypoint.Proofs
public import NN.Entrypoint.Runtime
public import NN.Entrypoint.Spec
public import NN.Entrypoint.TorchLeanModels
public import NN.Entrypoint.Verification
public import NN.Entrypoint.MLTheory
public import NN.Entrypoint.Tensor
public import NN.Entrypoint.Widgets

/-!
# Library

`NN.Library` is the curated umbrella import for TorchLean's broad reusable library surface area.

It includes:
- the public user facade (`NN.API.Public`),
- subsystem entrypoints (`NN.Entrypoint.*`),
- the shared op-tagged IR through `NN.Entrypoint.IR`,
- ML-theory and tensor APIs,
- infoview tooling through `NN.Entrypoint.Widgets`,
- and proof-backed verification APIs through `NN.Entrypoint.Verification`.

It excludes:
- executables / demos that define `main`
- test suites and benchmark runners

Trust-boundary documentation lives in `TRUST_BOUNDARIES.md`, where it can cover both Lean
declarations and external CUDA / Python / Julia / Arb producers without treating those prose notes
are part of the Lean API.

If you’re writing ordinary model/training code, prefer `import NN`. If you want to make the broad
umbrella explicit in downstream code, use `import NN.Library`. For lighter imports, use
`NN.Entrypoint.API`, `NN.Entrypoint.Tensor`, `NN.Entrypoint.IR`, or another subsystem entrypoint.
-/

@[expose] public section

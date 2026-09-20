/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN
public import NN.API.Trainer.FixedSample
public import NN.Examples
public import NN.Examples.Runner
public import NN.Verification.CLI
public import NN.CI.SlowProofs
public import NN.MLTheory.CROWN.Proofs.GraphRefinement

/-!
# TorchLean documentation surface

This import-only module collects the maintained reusable library, examples, command dispatchers,
and slow proof developments for API documentation. Executable-only dispatcher wrappers such as
`NN.Verification.Main` remain outside this surface; established example modules may still expose
their own runnable `main` alongside the declarations being documented.

Build the complete API documentation with:

```text
lake build TorchLeanDocs:docs
```
-/

@[expose] public section

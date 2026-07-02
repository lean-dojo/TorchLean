/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public
public import NN.API.Public.Facade
public import NN.API.Adapters
public import NN.API.Models.Generative
public import NN.API.SelfSupervised

/-!
# API Import

This import collects the public names used by `NN.Entrypoint.*` callers, including the
`TorchLean.*` namespaces.

Most model and training files can just use `import NN`. Use this one when you want the same public
names under the `NN.Entrypoint.*` tree without pulling in the broader umbrella.
-/

@[expose] public section

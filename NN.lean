/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Library

/-!
# TorchLean

Root umbrella import.

This re-exports `NN.Library`, the curated umbrella for TorchLean's reusable library surface.
Examples and CLI registries are documented as additional `NN:docs` roots, but they do not sit under
`import NN` because many examples intentionally import `NN`.

For subsystem-specific imports, use the `NN/Entrypoint/*` modules.
-/

@[expose] public section

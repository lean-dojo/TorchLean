/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.External.Process
public import NN.Runtime.External.Julia

/-!
# Runtime External

`NN.Runtime.External` is the umbrella for optional subprocess integrations.

TorchLean uses external programs in a narrow, explicit way: the external process may produce an
artifact, but Lean-side code must still parse, validate, or check that artifact before it becomes
trusted. This is the same “untrusted producer, trusted checker” boundary used by the Arb oracle,
Julia demos, PyTorch export smoke tests, and future certificate-producing tools.

This umbrella re-exports:
- `NN.Runtime.External.Process`, the generic subprocess/JSON/availability utilities; and
- `NN.Runtime.External.Julia`, the optional Julia wrapper.

Importing this file does not require Python, Julia, or any other external executable to be
installed. Those tools are only needed when a caller actually runs the corresponding IO action.
-/

@[expose] public section

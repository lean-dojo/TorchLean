/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.External.Julia

/-!
# Runtime External

`NN.Runtime.External` is the umbrella for optional subprocess integrations.

External programs produce data and artifacts. The adapters handle process failures and parse
outputs; mathematical claims about those outputs require a separate proof or certified checker.
Parsing JSON alone does not establish numerical correctness. This boundary applies to the Arb
oracle, Julia examples, and PyTorch export checks.

This umbrella re-exports:
- `NN.Core.ExternalProcess`, the generic subprocess/JSON/availability utilities; and
- `NN.Runtime.External.Julia`, the optional Julia wrapper.

Importing this file does not require Python, Julia, or any other external executable to be
installed. Those tools are only needed when a caller actually runs the corresponding IO action.
-/

@[expose] public section

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import NN.CI.All

/-!
# Comparator entrypoint

`leanprover/comparator` is a sandboxed checker for *untrusted* Lean submissions. It wants:
- a “challenge module” to compile, and
- a small set of declaration names to export and kernel-check.

For TorchLean we use this as a convenient “compile everything under `NN/`” target:
`NN.CI.ComparatorAll` imports `NN.CI.All`, which in turn imports the full library plus examples.

The marker theorem below is intentionally trivial; it just gives Comparator a stable declaration
name to export once compilation finishes.
-/

@[expose] public section

/-- Marker theorem for Comparator-based sandbox runs over the full `NN/` import surface. -/
theorem _root_.TorchLean.ciComparatorMarker : True := by
  trivial

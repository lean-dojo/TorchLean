/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public meta import NN.Tensor.Internal.Tactic.Report.Analysis.Render

/-!
# Static analysis for verified tensor transformations

Operation-specific analyzers decode reflected certificates into reports of
checked types, shapes, proof obligations, logical stages, generated execution,
static work estimates, and semantic correctness. The renderer preserves the
same account for symbolic dimensions without inventing concrete values.
-/

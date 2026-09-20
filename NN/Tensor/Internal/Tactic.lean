/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public meta import NN.Tensor.Internal.Tactic.Report
public import NN.Tensor.Internal.Tactic.Proof
import Mathlib.Algebra.Order.Field.Basic

/-!
# Verified tensor proof automation

This module is the public entry point for the `einops` proof tactic and the
`einops?` InfoView report. The proof engine and report renderer live in
separate modules so clients that only need proof automation do not depend on
the widget implementation.
-/

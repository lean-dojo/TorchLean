/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import Mathlib.Data.Finset.Attr
import Mathlib.Tactic.Finiteness.Attr
import Mathlib.Tactic.SetLike
public import NN.Tensor.Internal.Check.Einsum -- shake: keep
public import NN.Tensor.Internal.Check.Pack -- shake: keep
public import NN.Tensor.Internal.Check.ParseShape -- shake: keep
public import NN.Tensor.Internal.Check.Transform -- shake: keep
public import NN.Tensor.Internal.Syntax.Parser -- shake: keep
public import NN.Tensor.Internal.Syntax.Render -- shake: keep

/-!
# Tensor Pattern Language

Parsing, diagnostics, normalization, and static checking for TorchLean's
einops-compatible literal patterns. This facade is independent of elaborator
and native-kernel implementation details.
-/

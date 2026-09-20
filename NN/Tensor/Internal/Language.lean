/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Check.Einsum -- shake: keep
public import NN.Tensor.Internal.Check.Pack -- shake: keep
public import NN.Tensor.Internal.Check.ParseShape -- shake: keep

/-!
# Tensor Pattern Language

Parsing, diagnostics, normalization, and static checking for TorchLean's
einops-compatible literal patterns. This facade is independent of elaborator
and native-kernel implementation details.
-/

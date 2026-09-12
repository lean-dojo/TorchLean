/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Syntax.Parser.Expression.Config
public import NN.Tensor.Internal.Syntax.Parser.Expression.Parse
public import NN.Tensor.Internal.Syntax.Parser.Expression.Roundtrip
public import NN.Tensor.Internal.Syntax.Parser.Expression.Split
public import NN.Tensor.Internal.Syntax.Parser.Expression.State

/-!
# Shared einops expression parsing

This facade exposes expression configuration, token-state invariants, parsing,
canonical rendering, and transformation-arrow decomposition.
-/

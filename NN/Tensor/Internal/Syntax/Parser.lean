/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Syntax.Parser.Einsum
public import NN.Tensor.Internal.Syntax.Parser.Pack
public import NN.Tensor.Internal.Syntax.Parser.Transform

/-!
# Parsers for the einops functional language

This facade exports the expression, transformation, packing, and einsum
parsers while keeping their implementations in focused modules.
-/

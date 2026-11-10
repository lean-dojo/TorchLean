/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Tactic.Verify
public import TorchLean.Tensor.Einops.Lowering.Einsum

/-!
# Verification rules for program lowering

Import this module to prove with `verify` that an einsum plan preserves semantics. It tags
`Lowering.einsum_correct` and `Lowering.einsumKernel_correct` as `@[verify]` rules,
which equate the lowering of an already checked plan with its tensor semantics. The parser,
native compilation and external kernels need evidence of their own.
-/

public meta section

attribute [verify]
  TorchLean.Einops.Lowering.einsum_correct
  TorchLean.Einops.Lowering.einsumKernel_correct

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tactic.Verify
public import NN.Tensor.Internal.Lowering.Einsum

/-!
# Verification rules for program lowering

Import this module explicitly when proving that an einsum plan preserves semantics. The rules
compare an already checked plan with its tensor semantics. They do not certify the parser, native
compilation, or an external kernel; those require separate evidence.
-/

public meta section

attribute [verify]
  TorchLean.Tensor.Internal.Lowering.einsumTensor_correct
  TorchLean.Tensor.Internal.Lowering.einsumTensorKernel_correct

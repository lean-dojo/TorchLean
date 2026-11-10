/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

-- Downstream verification modules reach the trainer and the lowering through this module, so keep
-- its imports.
module -- shake: keep-downstream

public import TorchLean.NN.Sequential.Execution -- shake: keep
public import TorchLean.Verify.CROWN.Arithmetic -- shake: keep

/-!
# Lowering models to the verifier graph

Low-level access to TorchLean's verifier graph, for constructing or inspecting graph-level
IBP/CROWN workflows directly, starting from `lowerForwardToIR` for a sequential model or from
`IR.lowerForward` for a custom forward program whose last argument is the input. A trained model
is verified with `trained.verify center (radius := r)` from `TorchLean.Verify.Trainer`.

Both lowerings check the generated graph's structure and shapes. Slicing and axis selection
copy source entries through native slice and reshape operations, which keep their floating-point
words, and local evaluator theorems prove that correspondence. The complete model has no such
theorem: applying its bounds to the original finite-precision execution needs a source
correspondence argument of its own.
-/

@[expose] public section

namespace TorchLean.Verification

/-- Lower a sequential model into structurally checked verifier IR with one distinguished input.
Numerical equivalence with the complete source model requires a separate proof. -/
def lowerForwardToIR {α : Type} [TorchLean.Storage α] [Context α]
    {σ τ : Shape}
    (model : TorchLean.nn.Sequential σ τ)
    (state : TorchLean.TensorPack α (TorchLean.nn.Sequential.stateShapes model)) :
    Except String (IR.LoweredModel α) :=
  IR.lowerForward
    (α := α) (paramShapes := TorchLean.nn.Sequential.stateShapes model)
    (inShape := σ) (outShape := τ)
    (TorchLean.nn.Sequential.forward model (α := α))
    state

end TorchLean.Verification

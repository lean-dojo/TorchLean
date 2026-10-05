/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Kernel.Frontend
public import NN.Kernel.Tensor

/-!
# Scalar functions on tensors

`f.run input (device := cpu)` applies an ordinary Lean scalar function to a tensor.
`f.run input (device := gpu)` uses the checked custom-operation frontend and generated CUDA.
The input determines the output shape. CPU execution does not require GPU-compatible arithmetic;
unsupported GPU source is reported as an error, never replaced with CPU execution.
-/

public meta section

namespace NN.Kernel.Frontend

open Lean Lean.Meta Lean.Elab.Term Lean.Elab.Tactic

/-- Prepare the optional GPU representation of an ordinary scalar function at elaboration time. -/
elab "prepare_tensor_function" : tactic => withMainContext do
  let goal ← getMainGoal
  let target ← goal.getType
  unless target.isAppOf ``Except && target.getAppArgs.size == 2 do
    throwError "expected an optional elementwise program"
  let elementwise := target.getAppArgs.back!
  unless elementwise.isAppOf ``NN.Kernel.Elementwise && elementwise.getAppArgs.size == 2 do
    throwError "expected an elementwise source function"
  let function := elementwise.getAppArgs.back!
  let result ← (show TermElabM Lean.Expr from do
    let saved ← saveState
    try
      let program ← elaborate (← exprToSyntax function) none
      let source ← exprToSyntax program
      elabTermEnsuringType (← `(Except.ok ⟨$source, by intro read index; rfl⟩)) (some target)
    catch error =>
      let message ← error.toMessageData.toString
      saved.restore
      elabTermEnsuringType (← `(Except.error $(quote message))) (some target))
  goal.assign result
  replaceMainGoal []

end NN.Kernel.Frontend

end

public section

namespace Function

open TorchLean

/-- Apply a scalar function to every tensor entry on the selected device, preserving its shape.

CPU evaluates the original Lean function. GPU supports the custom-operation frontend's FP32/FP64
subset; unsupported functions, unavailable CUDA and other devices produce IO errors. GPU source
recognition happens during Lean elaboration; native compilation and execution remain external
trust boundaries. The compilation argument is supplied automatically.
-/
def run {α : Type} [Storage α]
    [Runtime.Autograd.Torch.TensorTransfer α] {shape : Spec.Shape}
    (f : α → α) (input : Tensor α shape) (device : NN.Backend.Device := cpu)
    (compilation : Except String (NN.Kernel.Elementwise f) := by prepare_tensor_function) :
    IO (Tensor α shape) := do
  if device == cpu then
    return Tensor.map f input
  if device != gpu then
    throw (IO.userError s!"custom operation: device {device.cliName} is unsupported")
  let compiled ← IO.ofExcept compilation
  compiled.program.run (TensorPack.singleton input) shape (device := device)

end Function

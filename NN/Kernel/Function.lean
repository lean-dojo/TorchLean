/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Kernel.Frontend
public import NN.Kernel.Tensor
public import NN.API.Autograd.Frontend

/-!
# Ordinary functions on tensors

`f.run input (device := cpu)` applies an ordinary Lean scalar function to a tensor.
`f.run input (device := gpu)` uses generated CUDA for supported scalar functions, and the existing
LibTorch operations for supported whole-tensor functions. `grad := true` records either kind
through the existing operation interface and returns its forward value and input pullback.
`f.zip left right` combines two equally shaped tensors with a binary scalar function.
Captured parameters are constants. Scalar mapping preserves the input shape; whole-tensor functions
can reduce or change shape. The call site infers the element type, storage and result shape.
Named numerical recurrences can use the same API when their recursive argument is last and their
parameters stay fixed. GPU lowering derives a loop and checks agreement with their Lean equations.
CPU execution does not require GPU-compatible arithmetic. A scalar type without a GPU encoding
runs on CPU with a diagnostic, retaining its type and precision. Unsupported source for an
otherwise supported GPU type, compilation failures and execution failures remain errors.
-/

public meta section

namespace NN.Kernel.Frontend

open Lean Lean.Meta Lean.Elab.Term Lean.Elab.Tactic

/-- Prepare a certified GPU representation of a unary or binary scalar function. -/
elab "prepare_tensor_function" : tactic => withMainContext do
  let goal ← getMainGoal
  let target ← goal.getType
  unless target.isAppOf ``Except && target.getAppArgs.size == 2 do
    throwError "expected a tensor-function compilation result"
  let contract := target.getAppArgs.back!
  let binary := contract.isAppOf ``NN.Kernel.Zip
  unless (contract.isAppOf ``NN.Kernel.Elementwise || binary) && contract.getAppArgs.size == 2 do
    throwError "expected a scalar tensor function"
  let function := contract.getAppArgs.back!
  let result ← (show TermElabM Lean.Expr from do
    let saved ← saveState
    try
      let source ← exprToSyntax (← whnf function)
      let scalar ← exprToSyntax contract.getAppArgs[0]!
      let source ← if binary then
          `(fun (read : NN.Kernel.Reader $scalar) (i : UInt64) => do
            let x ← read 0 i
            let y ← read 1 i
            pure ($source x y))
        else pure source
      let program ← elaborate source none
      -- Build the wrapper directly, retaining the checked unsigned-depth certificate.
      let constructor := if binary then ``NN.Kernel.Zip.mk else ``NN.Kernel.Elementwise.mk
      let partialConstructor := mkAppN (← mkConstWithFreshMVarLevels constructor)
        #[contract.getAppArgs[0]!, function, program]
      let .forallE _ evidenceType _ _ ← whnf (← inferType partialConstructor) |
        throwError "expected a tensor-function reference certificate"
      let evidence ← forallTelescope evidenceType fun locals body => do
        let evidence ← mkFreshExprSyntheticOpaqueMVar body
        let remaining ← Lean.Elab.Tactic.run evidence.mvarId! do
          evalTactic (← `(tactic|
            dsimp only [] <;>
            first
            | simp (config := { implicitDefEqProofs := false }) only
                [UInt64.toNat_ofNat', Nat.reduceMod] <;> rfl
            | rfl))
        unless remaining.isEmpty do
          let diagnostic ← Lean.Meta.ppGoal remaining.head!
          throwError "could not identify the tensor function's reference; {diagnostic}"
        mkLambdaFVars locals (← instantiateMVars evidence)
      let compiled ← instantiateMVars (mkApp partialConstructor evidence)
      mkAppOptM ``Except.ok #[some (mkConst ``String), some contract, some compiled]
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

namespace Internal

/-- Determine the result shape from the ordinary function's signature, before call-site recording.
This lets indexing and other consumers infer their types without waiting for source elaboration. -/
class Signature (α ι ο : Type) [Storage α] (input : Spec.Shape)
    (output : outParam Spec.Shape) : Prop where

@[default_instance] instance (priority := high) {α : Type} [Storage α] {input : Spec.Shape} :
    Signature α α α input input := ⟨⟩

instance {α : Type} [Storage α] {input output : Spec.Shape} :
    Signature α (Tensor α input) (Tensor α output) input output := ⟨⟩

instance {α : Type} [Storage α] {input : Spec.Shape} :
    Signature α (Tensor α input) α input Spec.Shape.scalar := ⟨⟩

/-- Call-site execution selected from the function's input and output types. -/
structure Execution {ι ο : Type} (f : ι → ο) (α : Type) [Storage α]
    (input output : Spec.Shape) (grad : Bool) where
  run : Tensor α input → NN.Backend.Device →
    IO (match grad with | false => Tensor α output | true => autograd.Result α input output)

/-- Apply a scalar function to every tensor entry on the selected device, preserving its shape.

CPU evaluates the original Lean function. GPU supports native FP32/FP64 and configured binary
arithmetic. Types without a GPU encoding run on CPU with a diagnostic, without narrowing values.
Unsupported functions, unavailable CUDA and other devices still produce IO errors.
GPU source recognition happens during Lean elaboration; native compilation and execution remain
external trust boundaries. The compilation argument is supplied automatically.
-/
def scalar {α : Type} [Storage α]
    {shape : Spec.Shape}
    (f : α → α) (input : Tensor α shape) (device : NN.Backend.Device := cpu)
    (gpuScalar? : Option (NN.Kernel.Scalar α) := by
      first | exact some inferInstance | exact none)
    (compilation : Except String (NN.Kernel.Elementwise f) := by prepare_tensor_function) :
    IO (Tensor α shape) := do
  if device == cpu then
    return Tensor.map f input
  if device != gpu then
    throw (IO.userError s!"custom operation: device {device.cliName} is unsupported")
  if !(gpuScalar?.any fun scalar => scalar.precision.supportsGpu) then
    IO.eprintln "custom computation: this scalar type is CPU-only; \
      keeping its type and precision on CPU"
    return Tensor.map f input
  let compiled ← IO.ofExcept compilation
  compiled.program.run (Arguments.empty.push input) shape (device := device)

private def tapeDevice {α : Type} [Storage α] [Runtime.Autograd.Torch.TensorTransfer α]
    (device : NN.Backend.Device) : IO NN.Backend.Device := do
  if device == cpu then return cpu
  if device != gpu then
    throw <| IO.userError s!"custom operation: device {device.cliName} is unsupported"
  if Runtime.Autograd.Torch.TensorTransfer.supportsGpu (α := α) then return gpu
  if Runtime.Autograd.Torch.TensorTransfer.supportsEncodedGpu (α := α) then return gpu
  IO.eprintln "autograd: this scalar uses the CPU tape; keeping its type and precision"
  return cpu

/-- Record with the existing tape without narrowing a custom function's element carrier. -/
def record {α : Type} [Storage α] [Runtime.Autograd.Torch.TensorTransfer α]
    {input output : Spec.Shape} (recording : autograd.Internal.Recording α input output)
    (value : Tensor α input) (device : NN.Backend.Device) :
    IO (autograd.Result α input output) := do
  recording.run value (← tapeDevice (α := α) device)

/-- Whole-tensor GPU execution shares the recorded runtime's precision policy. -/
def tensor {α : Type} [Storage α] [Runtime.Autograd.Torch.TensorTransfer α]
    {input output : Spec.Shape} (f : Tensor α input → Tensor α output)
    (recording : Except String (autograd.Internal.Recording α input output))
    (value : Tensor α input) (device : NN.Backend.Device) : IO (Tensor α output) := do
  if device == cpu then return f value
  let device ← tapeDevice (α := α) device
  if device == cpu then return f value
  match recording with
  | .error message => throw <| IO.userError message
  | .ok recording =>
      let result ← recording.run value device (grad := false)
      try return result.value finally result.close

end Internal

end Function

public meta section

namespace NN.Kernel.Frontend

open Lean Meta Elab Term Tactic

/-- Infer scalar mapping or whole-tensor execution and prepare the existing operation recording. -/
elab "prepare_function_execution" : tactic => withMainContext do
  let goal ← getMainGoal
  let target ← goal.getType
  let args := target.getAppArgs
  unless target.isAppOf ``Function.Internal.Execution do
    throwError "expected tensor function execution"
  let function := args[2]!
  let scalar := args[3]!
  let input := args[5]!
  let output := args[6]!
  let grad ← whnf args[7]!
  let domain := args[0]!
  let range := args[1]!
  let beforeScalar ← saveState
  let elementwise ← isDefEq domain scalar
  if !elementwise then beforeScalar.restore
  let tensor := ← mkConstWithFreshMVarLevels ``TorchLean.Tensor
  let tensorType := mkAppN tensor #[scalar, input, args[4]!]
  let scalarOutput ← if elementwise then do
      unless ← isDefEq range scalar do
        throwError "scalar tensor functions must preserve their element type"
      unless ← isDefEq output input do throwError "scalar mapping preserves its input shape"
      pure false
    else do
      unless ← isDefEq domain tensorType do
          throwError
            "expected a scalar function or tensor function; got {domain}, expected {tensorType}"
      if ← isDefEq range scalar then
        unless ← isDefEq output (mkConst ``Spec.Shape.scalar) do
          throwError "scalar tensor reductions return shape []"
        pure true
      else
        unless ← isDefEq range (mkAppN tensor #[scalar, output, args[4]!]) do
          throwError "tensor functions must preserve their element type"
        pure false
  let result ← (show TermElabM Lean.Expr from do
    let source ← exprToSyntax (← instantiateMVars function)
    let recordingType ← mkAppM ``TorchLean.autograd.Internal.Recording #[scalar, input, output]
    let recordingType ← mkAppM ``Except #[mkConst ``String, recordingType]
    let saved ← saveState
    let recording ← if elementwise && grad.isConstOf ``Bool.false then
        elabTermEnsuringType (← `(Except.error "recording was not requested")) (some recordingType)
      else try
        let result ← try
          elabTermEnsuringType
            (← `(Except.ok {
              context := inferInstance
              program := record_function% $source })) (some recordingType)
        catch error =>
          if !elementwise then throw error
          saved.restore
          elabTermEnsuringType (← `(Except.ok
            (TorchLean.autograd.Internal.Recording.elementwise {
              context := inferInstance
              program := record_function% $source }))) (some recordingType)
        synthesizeSyntheticMVarsNoPostponing
        instantiateMVars result
      catch error =>
        let message ← error.toMessageData.toString
        saved.restore
        elabTermEnsuringType (← `(Except.error $(quote message))) (some recordingType)
    let recording ← exprToSyntax recording
    let forward ← if elementwise then
        `(fun value device => Function.Internal.scalar $source value (device := device))
      else if scalarOutput then
        `(fun value device => Function.Internal.tensor
          (fun input => TorchLean.Tensor.scalar ($source input)) $recording value device)
      else
        `(fun value device => Function.Internal.tensor $source $recording value device)
    let backward ← `(match ($recording) with
      | .error message => throw (IO.userError message)
      | .ok recording => Function.Internal.record recording value device)
    let body ← if grad.isConstOf ``Bool.true then pure backward
      else if grad.isConstOf ``Bool.false then `($forward value device)
      else `(match ($(← exprToSyntax grad)) with
        | false => $forward value device
        | true => $backward)
    elabTermEnsuringType (← `(⟨fun value device => $body⟩)) (some target))
  goal.assign (← instantiateMVars result)
  replaceMainGoal []

end NN.Kernel.Frontend

end

public section

namespace Function

open TorchLean

/-- Execute an ordinary scalar or tensor function with inferred shapes and element type.

Scalar functions map over entries; tensor functions receive the whole input. CPU forward-only
calls evaluate the original function. Set `grad := true` to record supported operations on the
existing tape and return a value with `backward` and `close`. Unsupported recordings fail rather
than receive invented derivatives. Native `Float32` and `Float` keep binary32 and binary64 on the
same GPU tape. Configured binary arithmetic retains complete words on that tape; its operator
subset is narrower than native LibTorch's. CPU-only carriers keep their precision through a
reported fallback. Scalar branch decisions synchronize to Lean and record only the chosen path.
The execution argument is prepared automatically at the call site.
-/
def run {α ι ο : Type} [Storage α] {input output : Spec.Shape}
    (f : ι → ο) (value : Tensor α input) [Internal.Signature α ι ο input output]
    (device : NN.Backend.Device := cpu)
    (grad : Bool := false)
    (execution : Internal.Execution f α input output grad := by prepare_function_execution) :
    IO (match (generalizing := false) grad with
      | false => Tensor α output
      | true => autograd.Result α input output) :=
  execution.run value device

/-- Combine corresponding entries of two tensors on the selected device.

The shared shape prevents truncation or accidental broadcasting. CPU uses the packed binary map;
GPU compiles the binary scalar function into one custom operation. Captured scalar parameters
are allowed, with the same source-equivalence and foreign-code boundaries as `Function.run`.
-/
def zip {α : Type} [Storage α]
    {shape : Spec.Shape}
    (f : α → α → α) (left right : Tensor α shape) (device : NN.Backend.Device := cpu)
    (gpuScalar? : Option (NN.Kernel.Scalar α) := by
      first | exact some inferInstance | exact none)
    (compilation : Except String (NN.Kernel.Zip f) := by prepare_tensor_function) :
    IO (Tensor α shape) := do
  if device == cpu then
    return Tensor.map2Spec f left right
  if device != gpu then
    throw (IO.userError s!"custom operation: device {device.cliName} is unsupported")
  if !(gpuScalar?.any fun scalar => scalar.precision.supportsGpu) then
    IO.eprintln "custom computation: this scalar type is CPU-only; \
      keeping its type and precision on CPU"
    return Tensor.map2Spec f left right
  let compiled ← IO.ofExcept compilation
  compiled.program.run ((Arguments.empty.push left).push right) shape (device := device)

end Function

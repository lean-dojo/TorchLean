/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.CUDA.Kernel.Elab
public import TorchLean.NN.Kernel.Tensor
public import TorchLean.NN.Autograd.Frontend

/-!
# Ordinary functions on tensors

`f.run input (device := .cpu)` applies an ordinary Lean scalar function to a tensor.
`f.run input (device := .cuda)` uses generated CUDA for supported scalar functions, and LibTorch
operations for supported whole-tensor functions. With `grad := true`, `run` records either kind
with the autograd operations and returns the forward value and the input pullback. A function that
the recorder cannot record (an unsupported operation, or an instance that computes something other
than the element context's operation) is then an elaboration error.
`f.zip left right` combines two equally shaped tensors with a binary scalar function.

Captured parameters are constants. Scalar mapping preserves the input shape, and a whole-tensor
function can reduce or change it. The call site infers the element type, storage and result shape.
Named numerical recurrences can use the same API when their recursive argument is last and their
parameters stay fixed, and GPU lowering derives a loop and checks agreement with their Lean
equations.

CPU execution does not require GPU-compatible arithmetic, and a scalar type without a GPU encoding
runs on CPU with a diagnostic, keeping its type and precision. Unsupported source for an otherwise
supported GPU type, compilation failures and execution failures are errors.
-/

public meta section

namespace TorchLean.Kernel.Elab

open Lean Lean.Meta Lean.Elab.Term Lean.Elab.Tactic

/-- Prepare a certified GPU representation of a unary or binary scalar function. -/
elab "prepare_tensor_function" : tactic => withMainContext do
  let goal ← getMainGoal
  let target ← goal.getType
  unless target.isAppOf ``Except && target.getAppArgs.size == 2 do
    throwError "expected a tensor-function compilation result"
  let contract := target.getAppArgs.back!
  let binary := contract.isAppOf ``Kernel.Zip
  unless (contract.isAppOf ``Kernel.Elementwise || binary) && contract.getAppArgs.size == 2 do
    throwError "expected a scalar tensor function"
  let function := contract.getAppArgs.back!
  let result ← (show TermElabM Lean.Expr from do
    let saved ← saveState
    try
      let source ← exprToSyntax (← whnf function)
      let scalar ← exprToSyntax contract.getAppArgs[0]!
      let source ← if binary then
          `(fun (read : Kernel.Reader $scalar) (i : UInt64) => do
            let x ← read 0 i
            let y ← read 1 i
            pure ($source x y))
        else pure source
      let program ← checkedProgram source none
      -- Apply the wrapper's constructor, retaining the checked unsigned-depth certificate.
      let constructor := if binary then ``Kernel.Zip.mk else ``Kernel.Elementwise.mk
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

end TorchLean.Kernel.Elab

end

public section

namespace TorchLean.Kernel

open Backend.Device

/-- Determine the result shape from the ordinary function's signature, before call-site recording,
so indexing and other consumers can infer their types without waiting for source elaboration. A
scalar function keeps its input shape, a tensor function has the shape of its result type, and a
function from a tensor to a scalar has the shape `[]`. -/
class Signature (α ι ο : Type) [Storage α] (input : Shape)
    (output : outParam Shape) : Prop where

@[default_instance] instance (priority := high) {α : Type} [Storage α] {input : Shape} :
    Signature α α α input input := ⟨⟩

instance {α : Type} [Storage α] {input output : Shape} :
    Signature α (Tensor α input) (Tensor α output) input output := ⟨⟩

instance {α : Type} [Storage α] {input : Shape} :
    Signature α (Tensor α input) α input Shape.scalar := ⟨⟩

/-- Call-site execution selected from the function's input and output types. -/
structure Execution {ι ο : Type} (f : ι → ο) (α : Type) [Storage α]
    (input output : Shape) (grad : Bool) where
  /-- Run `f` on an input tensor on a device, returning its output, or a recording of it
  (`autograd.Result`) with its pullback when `grad` is `true`. -/
  run : Tensor α input → Backend.Device →
    IO (match grad with | false => Tensor α output | true => autograd.Result α input output)

/-- Apply a scalar function to every tensor entry on the selected device, preserving its shape.

CPU maps the Lean function `f` over the entries. GPU supports native FP32/FP64 and configured
binary arithmetic, and a type without a GPU encoding runs on CPU with a diagnostic, without
narrowing values. Unsupported functions and unavailable CUDA produce IO errors. GPU source
recognition happens during Lean elaboration, and native compilation and execution are external
trust boundaries. The `prepare_tensor_function` tactic supplies the compilation argument.
-/
def map {α : Type} [Storage α]
    {shape : Shape}
    (f : α → α) (input : Tensor α shape) (device : Backend.Device := cpu)
    (gpuScalar? : Option (Kernel.Scalar α) := by
      first | exact some inferInstance | exact none)
    (compilation : Except String (Kernel.Elementwise f) := by prepare_tensor_function) :
    IO (Tensor α shape) := do
  if device == cpu then
    return Tensor.map f input
  if !(gpuScalar?.any fun scalar => scalar.precision.supportsGPU) then
    IO.eprintln "custom computation: this scalar type is CPU-only; \
      keeping its type and precision on CPU"
    return Tensor.map f input
  let compiled ← IO.ofExcept compilation
  compiled.program.run (.singleton input) shape (device := device)

private def tapeDevice {α : Type} [Storage α] [TensorTransfer α]
    (device : Backend.Device) : IO Backend.Device := do
  if device == cpu then return cpu
  if TensorTransfer.supportsGPU (α := α) then return .cuda
  if TensorTransfer.supportsEncodedGPU (α := α) then return .cuda
  IO.eprintln "autograd: this scalar uses the CPU tape; keeping its type and precision"
  return cpu

/-- Record on the autograd tape without narrowing a custom function's element carrier. On `.cuda`,
a carrier that the GPU tape holds neither natively nor in encoded form is recorded on the CPU
tape with a diagnostic. -/
def record {α : Type} [Storage α] [TensorTransfer α]
    {input output : Shape} (recording : autograd.Recording α input output)
    (value : Tensor α input) (device : Backend.Device) :
    IO (autograd.Result α input output) := do
  recording.run value (← tapeDevice (α := α) device)

/-- Whole-tensor GPU execution shares the recorded runtime's precision policy. -/
def apply {α : Type} [Storage α] [TensorTransfer α]
    {input output : Shape} (f : Tensor α input → Tensor α output)
    (recording : Except String (autograd.Recording α input output))
    (value : Tensor α input) (device : Backend.Device) : IO (Tensor α output) := do
  if device == cpu then return f value
  let device ← tapeDevice (α := α) device
  if device == cpu then return f value
  match recording with
  | .error message => throw <| IO.userError message
  | .ok recording =>
      let result ← recording.run value device (grad := false)
      try return result.value finally result.close

end TorchLean.Kernel

public meta section

namespace TorchLean.Kernel.Elab

open Lean Meta Lean.Elab Term Tactic

/-- Infer scalar mapping or whole-tensor execution from the function's type and prepare its
recording with the autograd operations. A scalar function with `grad := false` has no recording.
When `grad` is literally `true`, a failed recording is an elaboration error. Otherwise the failure
is kept as an `Except.error` message, which the GPU forward path and a `grad` known only at run
time raise as an IO error. -/
elab "prepare_function_execution" : tactic => withMainContext do
  let goal ← getMainGoal
  let target ← goal.getType
  let args := target.getAppArgs
  unless target.isAppOf ``Execution do
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
  let tensor := ← mkConstWithFreshMVarLevels ``Tensor
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
        unless ← isDefEq output (mkConst ``Shape.scalar) do
          throwError "scalar tensor reductions return shape []"
        pure true
      else
        unless ← isDefEq range (mkAppN tensor #[scalar, output, args[4]!]) do
          throwError "tensor functions must preserve their element type"
        pure false
  let result ← (show TermElabM Lean.Expr from do
    let source ← exprToSyntax (← instantiateMVars function)
    let recordingType ← mkAppM ``autograd.Recording #[scalar, input, output]
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
            (autograd.Recording.elementwise {
              context := inferInstance
              program := record_function% $source }))) (some recordingType)
        synthesizeSyntheticMVarsNoPostponing
        instantiateMVars result
      catch error =>
        saved.restore
        -- With `grad := true` the recording is the result, so its failure is an elaboration error.
        if grad.isConstOf ``Bool.true then throw error
        let message ← error.toMessageData.toString
        elabTermEnsuringType (← `(Except.error $(quote message))) (some recordingType)
    let recording ← exprToSyntax recording
    let forward ← if elementwise then
        `(fun value device => map $source value (device := device))
      else if scalarOutput then
        `(fun value device => apply
          (fun input => TorchLean.Tensor.scalar ($source input)) $recording value device)
      else
        `(fun value device => apply $source $recording value device)
    let backward ← `(match ($recording) with
      | .error message => throw (IO.userError message)
      | .ok recording => record recording value device)
    let body ← if grad.isConstOf ``Bool.true then pure backward
      else if grad.isConstOf ``Bool.false then `($forward value device)
      else `(match ($(← exprToSyntax grad)) with
        | false => $forward value device
        | true => $backward)
    elabTermEnsuringType (← `(⟨fun value device => $body⟩)) (some target))
  goal.assign (← instantiateMVars result)
  replaceMainGoal []

end TorchLean.Kernel.Elab

end

public section

namespace Function

open TorchLean Backend.Device

/-- Execute an ordinary scalar or tensor function with inferred shapes and element type.

A scalar function maps over the entries, and a tensor function receives the whole input. A
forward-only call on CPU evaluates the Lean function itself. With `grad := true`, `run` records
the supported operations on the autograd tape and returns a value with `backward` and `close`. A
function the recorder cannot record faithfully (an unsupported operation, or an instance that
differs from the element context's) is an elaboration error when `grad` is literally `true`, and an
IO error when `grad` is known only at run time. Native `Float32` and `Float` keep binary32 and
binary64 on the same GPU tape. Configured binary arithmetic keeps complete words on that tape, with
a narrower operator subset than native LibTorch's. CPU-only carriers keep their precision through a
reported fallback. Scalar branch decisions synchronize to Lean and record only the chosen path. The
`prepare_function_execution` tactic prepares the execution argument at the call site.
-/
def run {α ι ο : Type} [Storage α] {input output : Shape}
    (f : ι → ο) (value : Tensor α input) [Kernel.Signature α ι ο input output]
    (device : Backend.Device := cpu)
    (grad : Bool := false)
    (execution : Kernel.Execution f α input output grad := by prepare_function_execution) :
    IO (match (generalizing := false) grad with
      | false => Tensor α output
      | true => autograd.Result α input output) :=
  execution.run value device

/-- Combine corresponding entries of two tensors on the selected device.

The shared shape prevents truncation or accidental broadcasting. CPU uses the packed binary map,
and GPU compiles the binary scalar function into one custom operation. A scalar type without a GPU
encoding runs on CPU with a diagnostic. Captured scalar parameters are allowed, with the same
source-equivalence and foreign-code boundaries as `Function.run`.
-/
def zip {α : Type} [Storage α]
    {shape : Shape}
    (f : α → α → α) (left right : Tensor α shape) (device : Backend.Device := cpu)
    (gpuScalar? : Option (Kernel.Scalar α) := by
      first | exact some inferInstance | exact none)
    (compilation : Except String (Kernel.Zip f) := by prepare_tensor_function) :
    IO (Tensor α shape) := do
  if device == cpu then
    return Tensor.zipWith f left right
  if !(gpuScalar?.any fun scalar => scalar.precision.supportsGPU) then
    IO.eprintln "custom computation: this scalar type is CPU-only; \
      keeping its type and precision on CPU"
    return Tensor.zipWith f left right
  let compiled ← IO.ofExcept compilation
  compiled.program.run (.pair left right) shape (device := device)

end Function

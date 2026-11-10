/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.NN.Backend
public import TorchLean.CUDA.Kernel.Program
public import TorchLean.CUDA.Kernel.Launch
public import TorchLean.Tensor.Constructors
public import TorchLean.Tensor.Pack

/-!
# Custom operations on tensors

A custom operation reads its inputs from a `TensorPack`, the public container of shape-indexed
tensors, and fills an output tensor of the requested shape. `Program.eval` evaluates it in Lean
with checked row-major reads and stops at the first failure, and `Program.eval_eq_reference`
proves that it agrees with the operation's source function. `Program.run` uses that evaluator on
the CPU and generated CUDA on the GPU. Native execution transfers complete scalar words in byte
arrays, which callers never handle.
-/

@[expose] public section

namespace TorchLean.Kernel

open Backend.Device

/-- Read tensor arguments by operand number and row-major unsigned index.

Each tensor's shape bounds its indices, and no intermediate array is built. An operand number past
the last tensor is an `.input` error, and an index past its tensor's size is a `.bounds` error.
-/
def Reader.ofState {α : Type} [Storage α] {shapes : List Shape}
    (inputs : TensorPack α shapes) : Reader α := fun operand index =>
  if hOperand : operand < shapes.length then
    let tensor := inputs.get ⟨operand, hOperand⟩
    if hIndex : index.toNat < (shapes.get ⟨operand, hOperand⟩).size then
      .ok (Tensor.Rep.getFlat tensor ⟨index.toNat, hIndex⟩)
    else
      .error (.bounds operand index (shapes.get ⟨operand, hOperand⟩).size)
  else
    .error (.input operand)

/-- Evaluate a custom operation once per output entry, in increasing row-major order.

Every output index must fit the expression language's unsigned addressing, so an output of more
than `UInt64.size` entries is a `.size` error. The first failed read stops the evaluation, and an
empty output makes no reads.
-/
def Program.eval {α : Type} [Storage α] {shapes : List Shape} (program : Program α)
    (inputs : TensorPack α shapes) (shape : Shape) : Except Error (Tensor α shape) :=
  if shape.size ≤ UInt64.size then
    let block := Target.lower program.expression
    Tensor.generateFlatM shape fun index =>
      block.eval program.arithmetic (Reader.ofState inputs) (Env.output index.val.toUInt64)
  else
    .error (.size shape.size)

/-- Tensor evaluation agrees with the operation's source function `program.reference` at every
output index, read failures and the output-size error included. -/
theorem Program.eval_eq_reference {α : Type} [Storage α] {shapes : List Shape}
    (program : Program α) (inputs : TensorPack α shapes) (shape : Shape) :
    program.eval inputs shape =
      if shape.size ≤ UInt64.size then
        Tensor.generateFlatM shape fun index =>
          program.reference (Reader.ofState inputs) index.val.toUInt64
      else .error (.size shape.size) := by
  unfold Program.eval
  split
  · dsimp only
    congr 1
    funext index
    exact program.lower_correct (Reader.ofState inputs) index.val.toUInt64
  · rfl

/-- Serialize complete words in little-endian order, including exceptional-value metadata. -/
private def upload {α : Type} [Storage α] (scalar : Scalar α) :
    {shapes : List Shape} → TensorPack α shapes → Array ByteArray → Except String (Array ByteArray)
  | [], .nil, values => .ok values
  | _ :: _, .cons first rest, values => do
      upload scalar rest (values.push (← scalar.pack first))

/-- Execute a custom operation on `device`, returning an output of the requested shape.

The Lean evaluator `Program.eval` runs it on the CPU and generated CUDA on the GPU. A scalar format
outside the GPU arithmetic's limits falls back to the Lean evaluator at the same precision and
prints a diagnostic. A GPU output of more than `2 ^ 63 - 1` entries is an error. The GPU path is
foreign code that no theorem here covers.
-/
@[no_expose] def Program.run {α : Type} [Storage α]
    {shapes : List Shape} (program : Program α)
    (inputs : TensorPack α shapes) (shape : Shape)
    (device : Backend.Device := cpu) : IO (Tensor α shape) := do
  if device == cpu then
    return ← IO.ofExcept (program.eval inputs shape |>.mapError reprStr)
  if !program.scalar.precision.supportsGPU then
    IO.eprintln "custom computation: this scalar format is CPU-only; keeping its precision on CPU"
    return ← IO.ofExcept (program.eval inputs shape |>.mapError reprStr)
  if shape.size > 2 ^ 63 - 1 then
    throw (IO.userError s!"kernel: output size {shape.size} exceeds native tensor addressing")
  let inputs ← IO.ofExcept
    (upload program.scalar inputs #[])
  let values ← runBytes program.scalar program.expression inputs shape.size.toUInt64
  IO.ofExcept (program.scalar.unpack shape values)

end TorchLean.Kernel

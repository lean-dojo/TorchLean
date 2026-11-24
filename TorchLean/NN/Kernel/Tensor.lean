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

`Program.evalBinary32` and `Program.runBinary32` run a binary64 operation on binary32 tensors: each
element read is widened exactly to binary64 and each result is rounded once to binary32.
`Program.evalBinary32_eq_reference` states that result in terms of the source function.
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

/-- Read binary32 tensor arguments as binary64 values, by `Reader.ofState` widened exactly with
`Float32.toFloat`. -/
def Reader.ofBinary32 {shapes : List Shape} (inputs : TensorPack Float32 shapes) : Reader Float :=
  fun operand index => (Reader.ofState inputs operand index).map Float32.toFloat

/-- Evaluate a binary64 operation on binary32 tensors, once per output entry in increasing
row-major order: the reads are widened with `Float32.toFloat` and each result is rounded with
`Float.toFloat32`. Failures are those of `Program.eval`. -/
def Program.evalBinary32 {shapes : List Shape} (program : Program Float)
    (inputs : TensorPack Float32 shapes) (shape : Shape) : Except Error (Tensor Float32 shape) :=
  if shape.size ≤ UInt64.size then
    let block := Target.lower program.expression
    Tensor.generateFlatM shape fun index =>
      (block.eval program.arithmetic (Reader.ofBinary32 inputs)
        (Env.output index.val.toUInt64)).map Float.toFloat32
  else
    .error (.size shape.size)

/-- Evaluation on binary32 tensors is the source function on the widened inputs, rounded to
binary32 at every output index, read failures and the output-size error included. -/
theorem Program.evalBinary32_eq_reference {shapes : List Shape} (program : Program Float)
    (inputs : TensorPack Float32 shapes) (shape : Shape) :
    program.evalBinary32 inputs shape =
      if shape.size ≤ UInt64.size then
        Tensor.generateFlatM shape fun index =>
          (program.reference (Reader.ofBinary32 inputs) index.val.toUInt64).map Float.toFloat32
      else .error (.size shape.size) := by
  unfold Program.evalBinary32
  split
  · dsimp only
    congr 1
    funext index
    rw [program.lower_correct (Reader.ofBinary32 inputs) index.val.toUInt64]
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

/-- Execute a binary64 operation on binary32 tensors on `device`, returning a binary32 output of
the requested shape.

On the CPU this is `Program.evalBinary32`. On the GPU one generated kernel reads `float` elements,
computes in `double` and writes `float` results (`CUDA.emitStored .binary32 .binary64`), so no
binary64 copy of an input or of the output is made. A program whose scalar is not native binary64
and a GPU output of more than `2 ^ 63 - 1` entries are errors. The GPU path, the two conversions
included, is foreign code that no theorem here covers.
-/
@[no_expose] def Program.runBinary32 {shapes : List Shape} (program : Program Float)
    (inputs : TensorPack Float32 shapes) (shape : Shape)
    (device : Backend.Device := cpu) : IO (Tensor Float32 shape) := do
  if device == cpu then
    return ← IO.ofExcept (program.evalBinary32 inputs shape |>.mapError reprStr)
  let .native .binary64 := program.scalar.precision |
    throw (IO.userError "kernel: binary32 storage needs a native binary64 operation")
  if shape.size > 2 ^ 63 - 1 then
    throw (IO.userError s!"kernel: output size {shape.size} exceeds native tensor addressing")
  let words ← IO.ofExcept (upload instScalarFloat32 inputs #[])
  let buffers ← words.mapM fun bytes => LibTorch.Buffer.ofBytesIO bytes .float32
  let output ← runBuffers .binary64 (fun x => (program.scalar.toBits x).toNat.toUInt64)
    program.expression buffers shape.size.toUInt64 (storage := .binary32)
  IO.ofExcept (instScalarFloat32.unpack shape (← LibTorch.Buffer.toBytesIO output .float32))

end TorchLean.Kernel

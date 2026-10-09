/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Backend.Types
public import NN.Kernel.Program
public import NN.Kernel.Runtime
public import NN.Tensor.Constructors
public import NN.API.Arguments

/-!
# Custom operations on tensors

Inputs use the public `Arguments` container for shape-indexed tensors. The pure evaluator uses
checked row-major reads and stops at the first failure. Native execution transfers complete scalar
words; byte arrays remain an implementation detail of the LibTorch bridge.
-/

@[expose] public section

namespace NN.Kernel

open Spec TorchLean

/-- Read tensor arguments by operand number and row-major unsigned index.

The shape determines the bounds, without materializing an intermediate array. Invalid operand
numbers and invalid addresses retain their distinct error diagnostics.
-/
def Reader.ofArguments {α : Type} [Storage α] {shapes : List Shape}
    (inputs : Arguments α shapes) : Reader α := fun operand index =>
  if hOperand : operand < shapes.length then
    let tensor := inputs.get ⟨operand, hOperand⟩
    if hIndex : index.toNat < (shapes.get ⟨operand, hOperand⟩).size then
      .ok (Tensor.Internal.Rep.getFlat tensor
        ⟨index.toNat, by simpa only [Shape.internalSize_eq] using hIndex⟩)
    else
      .error (.bounds operand index (shapes.get ⟨operand, hOperand⟩).size)
  else
    .error (.input operand)

/-- Tensor argument reads use the emitted unsigned bounds condition when the native length agrees
with the tensor shape and in-bounds native loads agree with its entries.

The premises identify the buffer metadata and contents; they do not assert that a foreign memory
implementation satisfies them. Out-of-bounds loads need no value because that branch never loads.
-/
theorem Reader.ofArguments_eq_native {α : Type} [Storage α] {shapes : List Shape}
    (inputs : Arguments α shapes) (operand : Nat) (hOperand : operand < shapes.length)
    (size : UInt64) (hSize : size.toNat = (shapes.get ⟨operand, hOperand⟩).size)
    (load : UInt64 → α)
    (hLoad : ∀ index (hIndex : index.toNat < (shapes.get ⟨operand, hOperand⟩).size),
      load index = Tensor.Internal.Rep.getFlat (inputs.get ⟨operand, hOperand⟩)
        ⟨index.toNat, by simpa only [Shape.internalSize_eq] using hIndex⟩)
    (index : UInt64) :
    Reader.ofArguments inputs operand index =
      if index ≥ size then .error (.bounds operand index size.toNat) else .ok (load index) := by
  unfold Reader.ofArguments
  simp only [hOperand, ↓reduceDIte]
  by_cases h : index ≥ size
  · have hIndex : ¬index.toNat < (shapes.get ⟨operand, hOperand⟩).size := by
      rw [← hSize]
      exact Nat.not_lt.mpr (UInt64.le_iff_toNat_le.mp h)
    simp only [hIndex, ↓reduceDIte, h, ↓reduceIte, hSize]
  · have hIndex : index.toNat < (shapes.get ⟨operand, hOperand⟩).size := by
      rw [← hSize]
      exact Nat.lt_of_not_ge fun hn => h (UInt64.le_iff_toNat_le.mpr hn)
    simp only [hIndex, ↓reduceDIte, h, ↓reduceIte]
    exact congrArg Except.ok (hLoad index hIndex).symm

/-- Evaluate a custom operation once per output entry, retaining the requested tensor shape.

All output indices must fit the expression language's unsigned addressing. An empty output makes
no reads. Input failures stop construction in increasing output-index order.
-/
def Program.eval {α : Type} [Storage α] {shapes : List Shape} (program : Program α)
    (inputs : Arguments α shapes) (shape : Shape) : Except Error (Tensor α shape) :=
  if shape.size ≤ UInt64.size then
    let block := Target.lower program.expression
    Tensor.generateFlatM shape fun index =>
      block.eval program.arithmetic (Reader.ofArguments inputs) (Env.output index.val.toUInt64)
  else
    .error (.size shape.size)

/-- Tensor evaluation agrees with the original source function at every output index.

The equality includes read failures and output-size rejection, not just successful tensors.
-/
theorem Program.eval_eq_reference {α : Type} [Storage α] {shapes : List Shape}
    (program : Program α) (inputs : Arguments α shapes) (shape : Shape) :
    program.eval inputs shape =
      if shape.size ≤ UInt64.size then
        Tensor.generateFlatM shape fun index =>
          program.reference (Reader.ofArguments inputs) index.val.toUInt64
      else .error (.size shape.size) := by
  unfold Program.eval
  split
  · dsimp only
    congr 1
    funext index
    exact program.lower_correct (Reader.ofArguments inputs) index.val.toUInt64
  · rfl

namespace Internal

/-- Serialize complete words in little-endian order, including exceptional-value metadata. -/
private def upload {α : Type} [Storage α] (scalar : Scalar α) :
    {shapes : List Shape} → TensorPack α shapes → Array ByteArray → Except String (Array ByteArray)
  | [], .nil, values => .ok values
  | _ :: _, .cons first rest, values => do
      upload scalar rest (values.push (← scalar.pack first))

end Internal

/-- Execute a custom operation on the selected device, retaining the requested output shape.

CPU uses the Lean evaluator; GPU uses generated CUDA. A format outside the GPU implementation's
limits uses the same Lean evaluator with a diagnostic, without changing precision. Other devices
and operational failures are rejected. Native execution remains an external trust boundary.
-/
@[no_expose] def Program.run {α : Type} [Storage α]
    {shapes : List Shape} (program : Program α)
    (inputs : Arguments α shapes) (shape : Shape)
    (device : NN.Backend.Device := cpu) : IO (Tensor α shape) := do
  if device == cpu then
    return ← IO.ofExcept (program.eval inputs shape |>.mapError reprStr)
  if device != gpu then
    throw (IO.userError s!"custom operation: device {device.cliName} is unsupported")
  if !program.scalar.precision.supportsGpu then
    IO.eprintln "custom computation: this scalar format is CPU-only; keeping its precision on CPU"
    return ← IO.ofExcept (program.eval inputs shape |>.mapError reprStr)
  if shape.size > 2 ^ 63 - 1 then
    throw (IO.userError s!"kernel: output size {shape.size} exceeds native tensor addressing")
  let inputs ← IO.ofExcept
    (Internal.upload program.scalar (Arguments.Internal.toTensorPack inputs) #[])
  let values ← Internal.runBytes program.scalar program.expression inputs shape.size.toUInt64
  IO.ofExcept (program.scalar.unpack shape values)

end NN.Kernel

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Program
public import NN.IR.Semantics
public import FloatLib.Floats.ExecFloat.Conversion.Core

/-!
# Heterogeneous graph interpretation

An interpreter supplies a scalar family indexed by `DType`, its storage, available
arithmetic, and explicit conversions. Values still contain ordinary TorchLean
tensors. Shape erasure reuses `Spec.SomeTensor`; only the element-type index is
added at the interchange boundary.

Ordinary numeric nodes reuse `Graph.evalNodeRaw`. Typed casts and registered
custom operations are dispatched separately, with the result's scalar type fixed
by the program's type assignment. Unsupported types and operations return errors.
No cast is implemented by relabeling a tensor or converting through host `Float`.
Layout operations reuse `Graph.evalTransform` and need only storage, not arithmetic.
Pointwise operators reuse `Graph.evalPointwise`, which also needs only storage: a type with an
arithmetic dictionary takes their meanings from `Spec.PointwiseOp.meaning`, and a type without
one, such as an integer type, from the interpreter's own `pointwise` table.
-/

@[expose] public section

namespace NN.IR

open TorchLean
open FloatLib.Floats.ExecFloat (ConversionStatus)
open FloatLib.Floats.Formats.BinaryInterchange.Model (IEEERoundingMode)

/-- Scalar representations and operations supplied to the shared typed graph interpreter. -/
structure Interpreter where
  /-- Actual Lean carrier for each element descriptor. -/
  Scalar : DType → Type
  /-- Physical tensor representation for that carrier. -/
  storage : (dtype : DType) → Storage (Scalar dtype)
  /-- Whether this interpreter implements the described element representation. -/
  supported : DType → Bool
  /-- Optional full arithmetic dictionary, used by the existing numerical operator evaluator. -/
  arithmetic : (dtype : DType) → Option (Context (Scalar dtype))
  /-- Convert one element, retaining flags; failures are explicit. -/
  cast : (source target : DType) → IEEERoundingMode → Scalar source →
    Except String (Scalar target × ConversionStatus)
  /-- Pointwise operators for element types without an arithmetic dictionary, such as integers.
  Types with a dictionary use `Spec.PointwiseOp.meaning` instead; the default lists none. -/
  pointwise : (dtype : DType) → Spec.PointwiseTable (Scalar dtype) := fun _ _ => none

namespace Interpreter

/-- A tensor with both its element descriptor and its shape preserved. -/
structure Value (interpreter : Interpreter) where
  dtype : DType
  tensor : @Spec.SomeTensor (interpreter.Scalar dtype) (interpreter.storage dtype)

namespace Value

/-- Runtime shape retained by the shared tensor package. -/
def shape {interpreter : Interpreter} (value : Value interpreter) : Spec.Shape :=
  @Spec.SomeTensor.shape (interpreter.Scalar value.dtype) (interpreter.storage value.dtype)
    value.tensor

/-- Recover a tensor only after checking its element-type index. -/
def get {interpreter : Interpreter} (value : Value interpreter) (dtype : DType) :
    Except String (@Spec.SomeTensor (interpreter.Scalar dtype) (interpreter.storage dtype)) := by
  letI := interpreter.storage value.dtype
  exact
  match value with
  | ⟨actual, tensor⟩ =>
      if h : actual = dtype then by
        cases h
        exact .ok tensor
      else .error s!"IR value: expected {dtype}, got {actual}"

end Value

/-- Typed constants, existing homogeneous parameter stores, and extension implementations. -/
structure Payload (interpreter : Interpreter) where
  /-- Constants retain their actual scalar carrier, including Boolean and integer tensors. -/
  constant? : Nat → Option (Value interpreter) := fun _ => none
  /-- Existing numerical payloads, selected at the operation's scalar type. -/
  homogeneous : (dtype : DType) →
    @NN.IR.Payload (interpreter.Scalar dtype) (interpreter.storage dtype)
  /-- Implement a registered operation at its declared output type; names are not guessed. -/
  call : (name : Lean.Name) → (output : DType) → Array (Value interpreter) →
    Except String
      (@Spec.SomeTensor (interpreter.Scalar output) (interpreter.storage output) × ConversionStatus)

/-- Convert a tensor element by element, accumulating FloatLib's conversion flags. -/
def castTensor (interpreter : Interpreter) (source target : DType) (shape : Spec.Shape)
    (rounding : IEEERoundingMode)
    (tensor : @Tensor (interpreter.Scalar source) shape (interpreter.storage source)) :
    Except String
      (@Tensor (interpreter.Scalar target) shape (interpreter.storage target) × ConversionStatus) :=
  letI := interpreter.storage source
  letI := interpreter.storage target
  let compute : StateT ConversionStatus (Except String)
      (Tensor (interpreter.Scalar target) shape) :=
    Tensor.generateFlatM shape fun index => do
      let value := Tensor.Internal.Rep.getFlat tensor
        (index.cast (Spec.Shape.internalSize_eq shape).symm)
      let (converted, flags) ← interpreter.cast source target rounding value
      modify fun accumulated => accumulated.merge flags
      pure converted
  compute.run {}

/-- Evaluate one node using already computed values and a complete argument table. -/
def evalNode (interpreter : Interpreter) (program : Program) (payload : Payload interpreter)
    (arguments : Array (Value interpreter)) (values : Array (Value interpreter)) (node : Node) :
    Except String (Value interpreter × ConversionStatus) := do
  let dtype ← program.dtype node.id
  unless interpreter.supported dtype do
    throw s!"IR interpreter: no implementation for element type {dtype}"
  letI : Storage (interpreter.Scalar dtype) := interpreter.storage dtype
  do
  let parents ← node.parents.mapM fun id => do
    let some value := values[id]?
      | throw s!"IR interpreter: missing parent {id} for node {node.id}"
    let expected ← program.dtype id
    unless value.dtype = expected do
      throw s!"IR interpreter: parent {id} has {value.dtype}, expected {expected}"
    pure value
  let (result, flags) ←
    match node.kind, program.primitive? node.id with
    | .input, none => do
        let some position := program.inputs.findIdx? (· == node.id)
          | throw s!"IR interpreter: input node {node.id} is not in the argument list"
        let some argument := arguments[position]?
          | throw s!"IR interpreter: missing argument {position}"
        pure (← argument.get dtype, {})
    | .const _, none => do
        let some value := payload.constant? node.id
          | throw s!"IR interpreter: missing typed constant for node {node.id}"
        pure (← value.get dtype, {})
    | .custom _ _ _, some (.cast rounding) => do
        let some parent := parents[0]?
          | throw s!"IR interpreter: cast node {node.id} has no operand"
        letI := interpreter.storage parent.dtype
        do
        let source ← Graph.expectShape node.outShape parent.tensor
        let (converted, flags) ←
          interpreter.castTensor parent.dtype dtype node.outShape rounding source
        pure (Spec.SomeTensor.ofTensor converted, flags)
    | .custom _ _ _, some (.call name _ _) =>
        payload.call name dtype parents
    | .pointwise op, none => do
        let meaning : Option (Spec.PointwiseFn (interpreter.Scalar dtype) op.arity) :=
          match interpreter.arithmetic dtype with
          | some arithmetic =>
              letI := arithmetic
              Spec.PointwiseOp.meaning op
          | none => interpreter.pointwise dtype op
        if meaning.isNone then
          throw s!"IR interpreter: {op.describe} has no implementation for {dtype}"
        let typedParents ← parents.mapM (fun value => value.get dtype)
        let localNode : Node :=
          { id := node.id, parents := Array.range typedParents.size,
            kind := node.kind, outShape := node.outShape }
        pure (← Graph.evalPointwise typedParents node.id localNode op meaning, {})
    | .permute _, none | .broadcastTo _ _, none | .reshape _ _, none |
        .flatten _, none | .concat _, none | .transpose _ _, none | .slice _ _ _, none => do
        let typedParents ← parents.mapM (fun value => value.get dtype)
        let localNode : Node :=
          { id := node.id, parents := Array.range typedParents.size,
            kind := node.kind, outShape := node.outShape }
        pure (← Graph.evalTransform typedParents node.id localNode, {})
    | _, _ => do
        let some arithmetic := interpreter.arithmetic dtype
          | throw s!"IR interpreter: {node.kind.tag} has no arithmetic implementation for {dtype}"
        letI := arithmetic
        do
        let typedParents ← parents.mapM (fun value => value.get dtype)
        -- The numerical evaluator expects global ids; the local view has exactly these parents.
        let localNode : Node :=
          { id := node.id, parents := Array.range typedParents.size,
            kind := node.kind, outShape := node.outShape }
        let input := Spec.SomeTensor.ofTensor
          (Tensor.scalar (0 : interpreter.Scalar dtype))
        let result ←
          Graph.evalNodeRaw (payload.homogeneous dtype) input typedParents node.id localNode
        pure (result, {})
  let tensor ← Graph.expectShape node.outShape result
  pure (⟨dtype, Spec.SomeTensor.ofTensor tensor⟩, flags)

/-- Evaluate a typed program, returning every intermediate tensor and accumulated cast flags. -/
def run (interpreter : Interpreter) (program : Program) (payload : Payload interpreter)
    (arguments : Array (Value interpreter)) :
    Except String (Array (Value interpreter) × ConversionStatus) := do
  program.check
  unless arguments.size == program.inputs.size do
    throw "IR interpreter: argument count does not match the program interface"
  let mut values : Array (Value interpreter) := #[]
  let mut flags : ConversionStatus := {}
  for node in program.graph.nodes do
    let (value, status) ← interpreter.evalNode program payload arguments values node
    values := values.push value
    flags := flags.merge status
  pure (values, flags)

/-- Evaluate the designated results in return order, retaining conversion flags. -/
def eval (interpreter : Interpreter) (program : Program) (payload : Payload interpreter)
    (arguments : Array (Value interpreter)) :
    Except String (Array (Value interpreter) × ConversionStatus) := do
  let (values, flags) ← interpreter.run program payload arguments
  let results ← program.outputs.mapM fun id =>
    match values[id]? with
    | some value => pure value
    | none => throw s!"IR interpreter: missing result node {id}"
  pure (results, flags)

end Interpreter

end NN.IR

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Reference
public import NN.Runtime.PyTorch.Import.TorchExport

/-!
# Typed graph artifacts

Frontends share the tensor graph schema and its shape checker. Each tensor node
also declares `dtype`; a custom node with `primitive = "cast"` declares an explicit
rounding mode. Constants for binary execution carry complete unsigned words in
`bits`, so parsing never loses precision through JSON floating-point numbers.

An artifact records all runtime inputs in `input_ids`, in argument order. Each
input has its own shape and element type. Typed container lowering still needs explicit
container type rules.
-/

@[expose] public section

namespace Import.IR

open Lean NN.IR
open FloatLib.Floats.Formats.BinaryInterchange

namespace Internal

open PyTorch.TorchExport.Internal

/-- Decode one explicit rounding direction; unknown policies are not silently defaulted. -/
def rounding (value : Json) : Except String Model.IEEERoundingMode := do
  match ← value.getStr? with
  | "nearestEven" => pure .nearestEven
  | "towardZero" => pure .towardZero
  | "towardPositiveInfinity" => pure .towardPositiveInfinity
  | "towardNegativeInfinity" => pure .towardNegativeInfinity
  | name => throw s!"IR import: unsupported rounding direction {name}"

/-- Read a parameter at its declared element type through the selected tensor decoder. -/
def parameter (interpreter : Interpreter)
    (load : DType → Spec.Shape → Array Nat → Except String (Interpreter.Value interpreter))
    (dtype : DType) (shape : Spec.Shape) (name : String)
    (object : Import.PyTorch.StateDict) :
    Except String (@TorchLean.Tensor (interpreter.Scalar dtype) shape
      (interpreter.storage dtype)) := do
  letI := interpreter.storage dtype
  do
  let actual ← DType.fromJson (← field "parameter" (name ++ "_dtype") object)
  unless actual = dtype do
    throw s!"IR import: parameter {name} has {actual}, expected {dtype}"
  let words ← nats name (← field "parameter" (name ++ "_bits") object)
  let value ← load dtype shape words
  Graph.expectShape shape (← value.get dtype)

end Internal

namespace Internal

/-- Parse once so the type assignment and payload use the same raw-to-tensor mapping. -/
def parse (artifact : Json) : Except String
    (PyTorch.TorchExport.Internal.CapturedValueGraph × PyTorch.TorchExport.CapturedGraph ×
      Program) := do
  let raw ← PyTorch.TorchExport.Internal.graph artifact
  let rawInputs ← PyTorch.TorchExport.Internal.inputs artifact
  let captured ← PyTorch.TorchExport.Internal.lower raw rawInputs
  let inputs ← rawInputs.mapM fun id =>
    match captured.rawToTensor[id]? with
    | some (some id) => pure id
    | _ => throw s!"IR import: input {id} has no tensor result"
  let mut dtypes : Array DType := #[]
  let mut primitives : Array (Option Primitive) := #[]
  for node in raw.nodes do
    let .tensor _ := node.valueShape
      | throw "IR import: typed container lowering is not supported"
    let some (some id) := captured.rawToTensor[node.id]?
      | throw s!"IR import: node {node.id} has no tensor result"
    unless dtypes.size ≤ id do
      throw "IR import: typed aliases need an explicit container type rule"
    let dtype ← DType.fromJson (← PyTorch.TorchExport.Internal.field
      s!"node[{node.id}]" "dtype" node.raw)
    -- Parameter-backed builtins use the same scalar context as their tensor edges.
    -- A different parameter type needs an explicit lowering, not an implicit relabeling.
    let parameterTypes :=
      #["weight_dtype", "bias_dtype", "gamma_dtype", "beta_dtype", "mean_dtype", "var_dtype"]
    for key in parameterTypes do
      if let some value := node.raw.get? key then
        let parameterType ← DType.fromJson value
        unless parameterType = dtype do
          throw s!"IR types: node {node.id}: {key} has {parameterType}, expected {dtype}"
    let primitive ← match node.raw.get? "primitive" with
      | some (.str "cast") => pure (some (.cast (← Internal.rounding
          (← PyTorch.TorchExport.Internal.field "cast" "rounding" node.raw))))
      | some (.str "call") =>
          let name ← DType.Internal.name
            (← PyTorch.TorchExport.Internal.field "call" "name" node.raw)
          let signature ← (← (← PyTorch.TorchExport.Internal.field
            "call" "input_dtypes" node.raw).getArr?).mapM DType.fromJson
          let result ← DType.fromJson
            (← PyTorch.TorchExport.Internal.field "call" "output_dtype" node.raw)
          pure (some (.call name signature result))
      | none => pure none
      | some other => throw s!"IR import: unsupported primitive registration {other}"
    -- Axis reductions may expand to several same-type nodes in the shared lowerer.
    for i in [dtypes.size:id + 1] do
      dtypes := dtypes.push dtype
      primitives := primitives.push (if i == id then primitive else none)
  let program : Program :=
    { graph := captured.graph, dtypes := dtypes, primitives := primitives
      inputs := inputs, outputs := captured.outputIds }
  pure (raw, captured, program)

end Internal

/-- Import element types and explicit casts using the existing tensor graph lowering. -/
def parseProgram (artifact : Json) : Except String Program := do
  let (_, _, program) ← Internal.parse artifact
  program.check
  pure program

/-- The returned program passed the shared checker, including its complete input interface. -/
theorem parseProgram_checked {artifact : Json} {program : Program}
    (h : parseProgram artifact = .ok program) : program.check = .ok () := by
  unfold parseProgram at h
  cases hParse : Internal.parse artifact with
  | error message => simp [hParse, Bind.bind, Except.bind] at h
  | ok data =>
      obtain ⟨raw, captured, parsed⟩ := data
      simp only [hParse, Bind.bind, Except.bind] at h
      cases hCheck : parsed.check with
      | error message => simp [hCheck] at h
      | ok result =>
          cases result
          have hProgram : parsed = program := by
            simpa [hCheck, Pure.pure, Except.pure] using h
          cases hProgram
          exact hCheck

/-- Every accepted typed artifact has a well-shaped graph, regardless of its frontend. -/
theorem parseProgram_wellShaped {artifact : Json} {program : Program}
    (h : parseProgram artifact = .ok program) : program.graph.WellShaped :=
  Program.check_wellShaped (parseProgram_checked h)

/-- Load typed constants and linear parameters using one explicitly selected word decoder.

The decoder's returned type and shape are checked before accepting its tensor. Loading a
parameter does not establish that the selected interpreter implements that operation's arithmetic.
-/
def parsePayload (interpreter : Interpreter)
    (load : DType → Spec.Shape → Array Nat → Except String (Interpreter.Value interpreter))
    (artifact : Json) : Except String (Interpreter.Payload interpreter) := do
  let (raw, captured, program) ← Internal.parse artifact
  program.check
  let mut constants : Array (Option (Interpreter.Value interpreter)) :=
    Array.replicate program.graph.nodes.size none
  let mut linears : Array (Option (Σ dtype : DType,
      @LinearWB (interpreter.Scalar dtype) (interpreter.storage dtype))) :=
    Array.replicate program.graph.nodes.size none
  for node in raw.nodes do
    if node.kind == "const" then
      let some (some id) := captured.rawToTensor[node.id]?
        | throw "IR import: constant has no tensor result"
      let dtype ← program.dtype id
      let constant ←
        letI : TorchLean.Storage (interpreter.Scalar dtype) := interpreter.storage dtype
        do
          let lowered ← program.graph.getNode id
          let words ← PyTorch.TorchExport.Internal.nats "constant.bits"
            (← PyTorch.TorchExport.Internal.field "constant" "bits" node.raw)
          let value ← load dtype lowered.outShape words
          let tensor ← Graph.expectShape lowered.outShape (← value.get dtype)
          pure (⟨dtype, Spec.SomeTensor.ofTensor tensor⟩ : Interpreter.Value interpreter)
      constants := constants.set! id (some constant)
    else if node.kind == "linear" then
      let some (some id) := captured.rawToTensor[node.id]?
        | throw "IR import: linear operation has no tensor result"
      let dtype ← program.dtype id
      let parameters ←
        letI : TorchLean.Storage (interpreter.Scalar dtype) := interpreter.storage dtype
        do
          let outDim ← PyTorch.TorchExport.Internal.natField "linear" "out_dim" node.raw
          let inDim ← PyTorch.TorchExport.Internal.natField "linear" "in_dim" node.raw
          let parameters : LinearWB (interpreter.Scalar dtype) :=
            { outDim := outDim, inDim := inDim
              W := ← Internal.parameter interpreter load dtype [outDim, inDim] "weight" node.raw
              b := ← Internal.parameter interpreter load dtype [outDim] "bias" node.raw }
          pure parameters
      linears := linears.set! id (some ⟨dtype, parameters⟩)
  pure
    { constant? := fun id => (constants[id]?).join
      homogeneous := fun dtype =>
        letI := interpreter.storage dtype
        { linear? := fun id => do
            let some ⟨source, parameters⟩ ← linears[id]? | none
            if h : source = dtype then some (h ▸ parameters) else none }
      call := fun name _ _ => .error s!"IR interpreter: no implementation registered for {name}" }

end Import.IR

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Check
public import NN.IR.Payload
public import NN.Runtime.PyTorch.Import.Core
public import NN.Runtime.PyTorch.Wire

/-!
# `torch.export` / FX Graph JSON Import

This module is the Lean half of the PyTorch-module import pipeline.

PyTorch has two different artifacts that people often blur together:

- `state_dict`: tensor values keyed by names; great for weights, but it does **not** describe the
  architecture.
- `torch.export` / FX graph: a captured tensor program; this is the piece we need before TorchLean
  can run native semantics, verification, or proof-oriented analyses.

The runtime bridge therefore uses a small, explicit JSON format:

```json
{
  "format": "torchlean.ir.v1",
  "input_id": 0,
  "output_ids": [4],
  "nodes": [
    {"id": 0, "kind": "input", "parents": [], "shape": [1, 4]},
    {"id": 1, "kind": "relu", "parents": [0], "shape": [1, 4]}
  ]
}
```

The parser below is deliberately conservative. It accepts only the current `NN.IR.OpKind` subset and
then runs TorchLean's executable graph validators (`checkWellFormed` and `checkShapes`). That gives
downstream tools a useful guarantee: if `parseGraph` succeeds, the resulting graph is structurally
acyclic, id-disciplined, arity-correct, and shape-consistent according to the shared IR inference
rules.

Op kind strings are not spelled here. They come from the shared table in
`NN.Runtime.PyTorch.Wire`, which the Python-emitting exporters also use.
The tag round trip is proved there.

The matching Python-side emitter lives in `NN.Runtime.PyTorch.Export.TorchExport`.
-/

@[expose] public section

namespace Import
namespace PyTorch
namespace TorchExport

open Lean
open Json
open Spec TorchLean
open NN.IR
open Interop.PyTorch

/--
Node ids whose payloads are synthesized while lowering one raw single-head MHA tuple node.

Keeping this binding explicit prevents payload parsing from depending on incidental node-number
offsets or reverse-engineering the generated subgraph.
-/
structure MhaPayloadBinding where
  /-- Raw value-graph node that owns the serialized MHA parameters. -/
  rawId : Nat
  /-- Generated query projection node. -/
  queryId : Nat
  /-- Generated key projection node. -/
  keyId : Nat
  /-- Generated value projection node. -/
  valueId : Nat
  /-- Generated attention-score scale constant node. -/
  scaleId : Nat
  /-- Generated output projection node. -/
  outputId : Nat
  deriving Repr

/-- A captured PyTorch graph lowered into TorchLean IR plus its designated interface node ids. -/
structure CapturedGraph where
  /-- TorchLean's checked op-tagged graph. -/
  graph : Graph
  /-- Designated runtime input node id. -/
  inputId : Nat
  /-- Designated graph output node ids, in PyTorch return order. -/
  outputIds : Array Nat
  /-- Map from serialized FX value ids to lowered tensor IR ids. -/
  rawToTensor : Array (Option Nat)
  /-- Payload ownership for subgraphs synthesized from raw MHA tuple nodes. -/
  mhaPayloadBindings : Array MhaPayloadBinding
  deriving Repr

namespace Internal

/--
Shape metadata for a PyTorch/FX value before it has been lowered to TorchLean's tensor-only IR.

PyTorch FX nodes may produce non-tensor containers. The most common example is
`nn.MultiheadAttention`, whose forward result is `(attn_output, attn_weights)`. We keep this
container structure in the import layer instead of treating every FX node as a tensor
node. Only tensor-valued nodes can be lowered into `NN.IR.Graph`.
-/
inductive ValueShape where
  | tensor (shape : Shape)
  | tuple (items : Array Shape)
  deriving Repr, BEq

/-- One raw PyTorch/FX value node from `torchlean.ir.v1` JSON. -/
structure CapturedValueNode where
  /-- Raw FX node id from the JSON artifact. -/
  id : Nat
  /-- Raw parent ids. These may refer to tensor values or tuple/container values. -/
  parents : Array Nat
  /-- Stable TorchLean/PyTorch import tag, e.g. `relu`, `matmul`, or `tuple_getitem`. -/
  kind : String
  /-- Tensor or tuple shape metadata. -/
  valueShape : ValueShape
  /-- Original object, retained so tensor nodes can be parsed through `parseOpKind`. -/
  raw : StateDict

/-- Captured PyTorch graph before tuple/container values are lowered away. -/
structure CapturedValueGraph where
  /-- Raw FX/value nodes. -/
  nodes : Array CapturedValueNode
  /-- Raw designated input id. -/
  inputId : Nat
  /-- Raw designated output ids. -/
  outputIds : Array Nat

/-! ## Small JSON helpers -/

/-- A uniform "wrong JSON shape" error, so import failures all read the same way. -/
def typeError {α : Type} (ctx expected : String) (j : Json) : Except String α :=
  .error s!"PyTorch graph import: {ctx}: expected {expected}, got {j}"

/-- Interpret a JSON number as a natural number. -/
def jsonNat (ctx : String) : Json → Except String Nat
  | .num n =>
      match n.toString.toNat? with
      | some k => .ok k
      | none => .error s!"PyTorch graph import: {ctx}: expected natural number, got {n}"
  | j => typeError ctx "natural number" j

/-- Decode a JSON value as a string, reporting the importing context on mismatch. -/
def jsonString (ctx : String) : Json → Except String String
  | .str s => .ok s
  | j => typeError ctx "string" j

/-- Decode a JSON value as an array, reporting the importing context on mismatch. -/
def jsonArray (ctx : String) : Json → Except String (Array Json)
  | .arr xs => .ok xs
  | j => typeError ctx "array" j

/-- Decode a JSON value as an object/state dictionary. -/
def jsonObject (ctx : String) : Json → Except String StateDict
  | .obj o => .ok o
  | j => typeError ctx "object" j

/-- Read a required object field. -/
def field (ctx key : String) (o : StateDict) : Except String Json :=
  match o.get? key with
  | some j => .ok j
  | none => .error s!"PyTorch graph import: {ctx}: missing field `{key}`"

/-- Read an optional object field. -/
def field? (key : String) (o : StateDict) : Option Json :=
  o.get? key

/-- Parse a JSON array of natural numbers. -/
def parseNatArray (ctx : String) (j : Json) : Except String (Array Nat) := do
  let xs ← jsonArray ctx j
  xs.mapM (jsonNat ctx)

/--
Parse a shape encoded as a dimension list.

Examples:
- `[]` means scalar;
- `[4]` means `Shape.dim 4 Shape.scalar`;
- `[2, 3]` means `Shape.dim 2 (Shape.dim 3 Shape.scalar)`.
-/
def parseShape (ctx : String) (j : Json) : Except String Shape := do
  pure (Shape.ofArray (← parseNatArray ctx j))

/-- Parse a node's parent ids. -/
def parseParents (ctx : String) (j : Json) : Except String (Array Nat) :=
  parseNatArray ctx j

/-- Read a natural-number field from a parsed Torch export JSON object. -/
def natField (ctx key : String) (o : StateDict) : Except String Nat := do
  jsonNat s!"{ctx}.{key}" (← field ctx key o)

/-- Read a Boolean field from a parsed Torch export JSON object. -/
def boolField (ctx key : String) (o : StateDict) : Except String Bool := do
  match ← field ctx key o with
  | .bool b => pure b
  | bad => typeError s!"{ctx}.{key}" "boolean" bad

/-- Read a shape-valued field from a parsed Torch export JSON object. -/
def shapeField (ctx key : String) (o : StateDict) : Except String Shape := do
  parseShape s!"{ctx}.{key}" (← field ctx key o)

/-- Read a natural-number array field from a parsed Torch export JSON object. -/
def natArrayField (ctx key : String) (o : StateDict) : Except String (Array Nat) := do
  parseNatArray s!"{ctx}.{key}" (← field ctx key o)

/-- Read a floating-point field used by numerical operator payloads. -/
def floatField (ctx key : String) (o : StateDict) : Except String Float := do
  match ← field ctx key o with
  | .num value => pure value.toFloat
  | bad => typeError s!"{ctx}.{key}" "number" bad

/-- Read a shape-indexed floating-point tensor field. -/
def tensorField (ctx key : String) (shape : Shape) (o : StateDict) :
    Except String (Tensor Float shape) := do
  match Import.PyTorch.parseTensor shape (← field ctx key o) with
  | some value => pure value
  | none => throw s!"PyTorch graph import: {ctx}.{key}: tensor does not match {repr shape}"

/-- Read fixed-length tensor metadata, rejecting inconsistent serialized geometry. -/
def natTensorField (ctx key : String) (rank : Nat) (o : StateDict) :
    Except String (TorchLean.Tensor Nat [rank]) := do
  let values ← natArrayField ctx key o
  if h : values.size = rank then
    pure (h ▸ TorchLean.Tensor.from values)
  else
    throw s!"PyTorch graph import: {ctx}.{key}: expected {rank} entries, got {values.size}"

/-- Parse generic pooling window metadata. -/
def windowConfig (ctx : String) (o : StateDict) : Except String WindowConfig := do
  let spatialRank ← natField ctx "spatial_rank" o
  pure
    { spatialRank := spatialRank
      kernel := ← natTensorField ctx "kernel" spatialRank o
      stride := ← natTensorField ctx "stride" spatialRank o
      padding := ← natTensorField ctx "padding" spatialRank o }

/-- Parse a JSON array whose elements are shape arrays. -/
def parseShapeArray (ctx : String) (j : Json) : Except String (Array Shape) := do
  let xs ← jsonArray ctx j
  xs.mapM (parseShape ctx)

/--
Parse the value-level shape metadata emitted by the Python bridge.

`torchlean.ir.v1` artifacts without `value_kind` are interpreted as tensor-valued nodes with a
required `shape` field. The explicit form uses `value_kind = "tensor"` or
`value_kind = "tuple"` explicitly.
-/
def parseValueShape (ctx : String) (o : StateDict) : Except String ValueShape := do
  match field? "value_kind" o with
  | none => pure (.tensor (← shapeField ctx "shape" o))
  | some (.str "tensor") => pure (.tensor (← shapeField ctx "shape" o))
  | some (.str "tuple") =>
      pure (.tuple (← parseShapeArray s!"{ctx}.tuple_shapes" (← field ctx "tuple_shapes" o)))
  | some (.str other) =>
      throw s!"PyTorch graph import: {ctx}: unsupported value_kind `{other}`"
  | some bad => typeError s!"{ctx}.value_kind" "string" bad

/--
Parse a TorchLean IR op kind.

The schema uses a stable string tag plus op-specific scalar fields. The tag is resolved through
`Wire.parseOpTag?`, which recognizes the fixed v1 spellings. This function then reads the
attributes each constructor needs. We avoid trying to parse raw PyTorch operator
names here; the Python adapter is responsible for translating `torch.ops.aten.*` / FX targets into
these TorchLean tags.
-/
def parseOpKind (ctx : String) (outShape : Shape) (o : StateDict) : Except String OpKind := do
  let tagString ← jsonString s!"{ctx}.kind" (← field ctx "kind" o)
  let some tag := Wire.parseOpTag? tagString
    | throw s!"PyTorch graph import: {ctx}: unsupported TorchLean IR op kind `{tagString}`"
  match tag with
  | .const =>
      let valueShape ←
        match field? "value_shape" o with
        | some j => parseShape s!"{ctx}.value_shape" j
        | none => pure outShape
      pure (.const valueShape)
  | .permute => pure (.permute (← natArrayField ctx "perm" o))
  | .transpose => pure (.transpose (← natField ctx "axis1" o) (← natField ctx "axis2" o))
  | .randUniform => pure (.randUniform (← natField ctx "seed" o))
  | .bernoulliMask => pure (.bernoulliMask (← natField ctx "seed" o))
  | .maxPool => pure (.maxPool (← windowConfig ctx o))
  | .avgPool => pure (.avgPool (← windowConfig ctx o))
  | .broadcastTo =>
      pure (.broadcastTo (← shapeField ctx "from_shape" o) (← shapeField ctx "to_shape" o))
  | .reduceSum | .reduceMean =>
      throw s!"PyTorch graph import: {ctx}: reduction must be lowered from its axes configuration"
  | .conv =>
      let window ← windowConfig ctx o
      pure (.conv
        { spatialRank := window.spatialRank
          kernel := window.kernel
          stride := window.stride
          padding := window.padding
          dilation := ← natTensorField ctx "dilation" window.spatialRank o
          paddingAfter := ← natTensorField ctx "padding_after" window.spatialRank o
          groups := ← natField ctx "groups" o
          channelAxis := ← natField ctx "channel_axis" o
          inChannels := ← natField ctx "in_channels" o
          outChannels := ← natField ctx "out_channels" o })
  | .batchNormEval =>
      pure (.batchNormEval (← natField ctx "channel_axis" o) (← natField ctx "channels" o))
  | .softmax => pure (.softmax (← natField ctx "axis" o))
  | .hardMaskedSoftmax =>
      throw s!"PyTorch graph import: {ctx}: `{tagString}` has no artifact encoding for its mask yet"
  | .layernorm => pure (.layernorm (← natField ctx "axis" o))
  | .reshape => pure (.reshape (← shapeField ctx "in_shape" o) (← shapeField ctx "out_shape" o))
  | .flatten => pure (.flatten (← shapeField ctx "value_shape" o))
  | .concat => pure (.concat (← natField ctx "axis" o))
  | .input | .detach | .add | .sub | .mulElem | .abs | .sqrt | .inv | .maxElem | .minElem | .sum
  | .matmul | .linear | .relu | .tanh | .sigmoid | .softplus | .safeLog
  | .exp | .log | .sin | .cos | .mseLoss =>
      match tag.toKind? with
      | some kind => pure kind
      | none => throw s!"PyTorch graph import: {ctx}: internal error: `{tagString}` needs a payload"

/-- Parse one raw PyTorch/FX value node. -/
def parseValueNode (j : Json) : Except String CapturedValueNode := do
  let o ← jsonObject "node" j
  let id ← natField "node" "id" o
  let ctx := s!"node[{id}]"
  let parents ← parseParents s!"{ctx}.parents" (← field ctx "parents" o)
  let kind ← jsonString s!"{ctx}.kind" (← field ctx "kind" o)
  let valueShape ← parseValueShape ctx o
  pure { id := id, parents := parents, kind := kind, valueShape := valueShape, raw := o }

/-- Parse the graph object into the PyTorch/FX value-level graph, before tensor lowering. -/
def parseValueGraph (j : Json) : Except String CapturedValueGraph := do
  let o ← jsonObject "root" j
  match field? "format" o with
  | some (.str format) =>
      if format != Wire.format then
        throw s!"PyTorch graph import: unsupported format `{format}` (expected `{Wire.format}`)"
  | some bad => typeError "root.format" "string" bad
  | none => throw "PyTorch graph import: root: missing field `format`"
  let inputId ← natField "root" "input_id" o
  let outputIds ← natArrayField "root" "output_ids" o
  if outputIds.isEmpty then
    throw "PyTorch graph import: root.output_ids must contain at least one output"
  let nodeVals ← jsonArray "root.nodes" (← field "root" "nodes" o)
  let nodes ← nodeVals.mapM parseValueNode
  pure { nodes := nodes, inputId := inputId, outputIds := outputIds }

/-- Look up a raw value node by id. -/
def getValueNode (vg : CapturedValueGraph) (id : Nat) : Except String CapturedValueNode :=
  match vg.nodes[id]? with
  | some n =>
      if n.id = id then pure n
      else throw s!"PyTorch graph import: raw node id mismatch at index {id}: found node {n.id}"
  | none => throw s!"PyTorch graph import: raw node id {id} out of bounds"

/--
Validate the value graph before raw ids are used as array indices during tensor lowering.

The tensor IR checker performs the corresponding validation after lowering. This earlier check is
still necessary because tuple/container nodes do not enter the tensor IR, and because `rawToTensor`
is indexed by raw node id while lowering is in progress.
-/
def checkValueGraph (vg : CapturedValueGraph) : Except String Unit := do
  if vg.nodes.isEmpty then
    throw "PyTorch graph import: graph contains no nodes"
  let mut inputCount := 0
  for i in [0:vg.nodes.size] do
    let raw ←
      match vg.nodes[i]? with
      | some raw => pure raw
      | none => throw s!"PyTorch graph import: internal error: missing raw node at index {i}"
    if raw.id != i then
      throw <|
        s!"PyTorch graph import: raw id discipline violated at index {i}: " ++
          s!"nodes[{i}].id = {raw.id}"
    for parentId in raw.parents do
      if parentId ≥ raw.id then
        throw <|
          s!"PyTorch graph import: node[{raw.id}]: parent id {parentId} is not < {raw.id}"
    if raw.kind = Wire.opTag .input then
      inputCount := inputCount + 1
  if inputCount != 1 then
    throw s!"PyTorch graph import: expected exactly one input node, got {inputCount}"
  let input ← getValueNode vg vg.inputId
  if input.kind != Wire.opTag .input then
    throw <|
      s!"PyTorch graph import: input_id {vg.inputId} designates `{input.kind}`, not `input`"
  for outputId in vg.outputIds do
    let _output ← getValueNode vg outputId
  pure ()

/--
Lower a PyTorch/FX value graph to TorchLean's tensor-only IR.

Tuple/container nodes are preserved in `CapturedValueGraph`, but they do not become `NN.IR.Node`s.
TorchLean's verification and execution passes consume the clean tensor DAG, while the import layer
can explain container-valued PyTorch failures without changing their semantics.
-/
def lowerValueGraph (vg : CapturedValueGraph) : Except String CapturedGraph := do
  checkValueGraph vg
  let mut rawToTensor : Array (Option Nat) := Array.replicate vg.nodes.size none
  let mut tensorNodes : Array Node := #[]
  let mut mhaPayloadBindings : Array MhaPayloadBinding := #[]
  for raw in vg.nodes do
    let ctx := s!"node[{raw.id}]"
    match raw.valueShape with
    | .tuple _items =>
        rawToTensor := rawToTensor.set! raw.id none
    | .tensor outShape =>
        if raw.kind = Wire.tupleGetItem then
          let index ← natField ctx "index" raw.raw
          let parentId ←
            match raw.parents with
            | #[p] => pure p
            | _ => throw s!"PyTorch graph import: {ctx}: tuple_getitem expects one tuple parent"
          let parent ← getValueNode vg parentId
          match parent.valueShape with
          | .tuple items =>
              let some selectedShape := items[index]?
                | throw <|
                    s!"PyTorch graph import: {ctx}: tuple index {index} is out of bounds for " ++
                      s!"{items.size} components"
              if selectedShape != outShape then
                throw <|
                  s!"PyTorch graph import: {ctx}: tuple component {index} has shape " ++
                    s!"{repr selectedShape}, but the projection declares {repr outShape}"
              if parent.kind = Wire.mhaTuple then
                if index != 0 then
                  throw <|
                    s!"PyTorch graph import: {ctx}: `nn.MultiheadAttention` attention weights " ++
                    "are represented in the value graph but are not lowered to tensor IR yet"
                let numHeads ← natField s!"node[{parent.id}]" "num_heads" parent.raw
                let embedDim ← natField s!"node[{parent.id}]" "embed_dim" parent.raw
                let batchFirst ← boolField s!"node[{parent.id}]" "batch_first" parent.raw
                let dropoutZero ← boolField s!"node[{parent.id}]" "dropout_zero" parent.raw
                if numHeads != 1 then
                  throw <|
                    s!"PyTorch graph import: {ctx}: `nn.MultiheadAttention` lowering " ++
                    s!"supports only num_heads=1, got {numHeads}"
                if !batchFirst then
                  throw <|
                    s!"PyTorch graph import: {ctx}: `nn.MultiheadAttention` lowering " ++
                    "supports only batch_first=True"
                if !dropoutZero then
                  throw <|
                    s!"PyTorch graph import: {ctx}: `nn.MultiheadAttention` lowering requires " ++
                    "dropout=0/eval deterministic semantics"
                let xRawId ←
                  match parent.parents with
                  | #[q, k, v] =>
                      if q = k ∧ k = v then pure q
                      else
                        throw <|
                          s!"PyTorch graph import: {ctx}: `nn.MultiheadAttention` lowering " ++
                          "supports self-attention only (query/key/value must be the same value)"
                  | _ =>
                      throw <|
                        s!"PyTorch graph import: {ctx}: expected query/key/value parents for " ++
                        "`nn.MultiheadAttention`"
                let xId ←
                  match rawToTensor[xRawId]? with
                  | some (some tid) => pure tid
                  | some none =>
                      throw <|
                        s!"PyTorch graph import: {ctx}: MHA input raw node {xRawId} is not " ++
                          "tensor-lowerable"
                  | none =>
                      throw <|
                        s!"PyTorch graph import: {ctx}: MHA input raw node {xRawId} out of bounds"
                let xNode ←
                  match tensorNodes[xId]? with
                  | some n => pure n
                  | none =>
                      throw <|
                        s!"PyTorch graph import: {ctx}: internal tensor node {xId} out of bounds"
                let (batch, seqLen, actualEmbed) ←
                  match xNode.outShape with
                  | .dim b (.dim n (.dim d .scalar)) => pure (b, n, d)
                  | s =>
                      throw <|
                        s!"PyTorch graph import: {ctx}: single-head MHA lowering expects " ++
                        s!"batch-first rank-3 input `(batch, seq, embed)`, got {repr s}"
                if actualEmbed != embedDim then
                  throw <|
                    s!"PyTorch graph import: {ctx}: MHA metadata embed_dim={embedDim} but " ++
                    s!"input last dimension is {actualEmbed}"
                if outShape != xNode.outShape then
                  throw <|
                    s!"PyTorch graph import: {ctx}: MHA output shape mismatch: expected " ++
                    s!"{repr xNode.outShape}, got {repr outShape}"

                -- Decompose the deterministic single-head self-attention output into existing IR
                -- primitives:
                --   q/k/v = F.linear(x, ...)
                --   scores = q @ kᵀ
                --   probs = softmax(scores * scale, dim=-1)
                --   out = F.linear(probs @ v, ...)
                --
                -- Projection weights and the scale constant remain external payloads keyed by the
                -- generated node ids, exactly like ordinary `linear`/`const` IR nodes.
                let qId := tensorNodes.size
                tensorNodes := tensorNodes.push
                  { id := qId, parents := #[xId], kind := .linear, outShape := xNode.outShape }
                let kId := tensorNodes.size
                tensorNodes := tensorNodes.push
                  { id := kId, parents := #[xId], kind := .linear, outShape := xNode.outShape }
                let vId := tensorNodes.size
                tensorNodes := tensorNodes.push
                  { id := vId, parents := #[xId], kind := .linear, outShape := xNode.outShape }
                let ktShape : Shape := .dim batch (.dim actualEmbed (.dim seqLen .scalar))
                let ktId := tensorNodes.size
                tensorNodes := tensorNodes.push
                  { id := ktId, parents := #[kId], kind := .transpose 1 2, outShape := ktShape }
                let scoresShape : Shape := .dim batch (.dim seqLen (.dim seqLen .scalar))
                let scoresId := tensorNodes.size
                tensorNodes := tensorNodes.push
                  { id := scoresId, parents := #[qId, ktId], kind := .matmul,
                    outShape := scoresShape }
                let scaleId := tensorNodes.size
                tensorNodes := tensorNodes.push
                  { id := scaleId, parents := #[], kind := .const scoresShape,
                    outShape := scoresShape }
                let scaledId := tensorNodes.size
                tensorNodes := tensorNodes.push
                  { id := scaledId, parents := #[scoresId, scaleId], kind := .mulElem,
                    outShape := scoresShape }
                let probsId := tensorNodes.size
                tensorNodes := tensorNodes.push
                  { id := probsId, parents := #[scaledId], kind := .softmax 2,
                    outShape := scoresShape }
                let ctxId := tensorNodes.size
                tensorNodes := tensorNodes.push
                  { id := ctxId, parents := #[probsId, vId], kind := .matmul,
                    outShape := xNode.outShape }
                let outId := tensorNodes.size
                tensorNodes := tensorNodes.push
                  { id := outId, parents := #[ctxId], kind := .linear, outShape := outShape }
                mhaPayloadBindings := mhaPayloadBindings.push
                  { rawId := parent.id
                    queryId := qId
                    keyId := kId
                    valueId := vId
                    scaleId := scaleId
                    outputId := outId }
                rawToTensor := rawToTensor.set! raw.id (some outId)
              else
                throw <|
                  s!"PyTorch graph import: {ctx}: selected tensor from tuple-producing parent " ++
                  s!"`{parent.kind}`. Tuple/getitem is represented in the value graph, but this " ++
                  "tuple producer has no tensor-lowering rule yet. Decompose the producer into " ++
                  "supported tensor ops, or add a real semantic lowering for that operation."
          | .tensor _ =>
              throw s!"PyTorch graph import: {ctx}: getitem on a tensor value is not tensor-lowered"
        else if raw.kind = Wire.opTag .reduceSum || raw.kind = Wire.opTag .reduceMean then
          let parentRawId ←
            match raw.parents with
            | #[p] => pure p
            | _ => throw s!"PyTorch graph import: {ctx}: reduction expects one tensor parent"
          let parentId ←
            match rawToTensor[parentRawId]? with
            | some (some id) => pure id
            | some none => throw s!"PyTorch graph import: {ctx}: reduction parent is not a tensor"
            | none => throw s!"PyTorch graph import: {ctx}: reduction parent is out of bounds"
          let parentNode ←
            match tensorNodes[parentId]? with
            | some node => pure node
            | none => throw s!"PyTorch graph import: {ctx}: internal reduction parent missing"
          let axes ← natArrayField ctx "axes" raw.raw
          let keepDim ← boolField ctx "keepdim" raw.raw
          let rank := parentNode.outShape.rank
          if axes.any fun axis => axis >= rank then
            throw s!"PyTorch graph import: {ctx}: reduction axis is outside rank {rank}"
          if axes.any fun axis => (axes.filter fun candidate => candidate = axis).size > 1 then
            throw s!"PyTorch graph import: {ctx}: reduction axes contain duplicates"
          let mut remaining := axes
          let mut currentId := parentId
          let mut currentShape := parentNode.outShape
          while !remaining.isEmpty do
            let axis := remaining.foldl Nat.max 0
            remaining := remaining.filter fun candidate => candidate != axis
            let dims := currentShape.toList
            let reducedShape := Shape.ofList (dims.take axis ++ dims.drop (axis + 1))
            let reducedId := tensorNodes.size
            let kind :=
              if raw.kind = Wire.opTag .reduceSum then .reduceSum axis else .reduceMean axis
            tensorNodes := tensorNodes.push
              { id := reducedId, parents := #[currentId], kind := kind, outShape := reducedShape }
            if keepDim then
              let keptShape := Shape.ofList (dims.take axis ++ 1 :: dims.drop (axis + 1))
              let reshapeId := tensorNodes.size
              tensorNodes := tensorNodes.push
                { id := reshapeId, parents := #[reducedId],
                  kind := .reshape reducedShape keptShape, outShape := keptShape }
              currentId := reshapeId
              currentShape := keptShape
            else
              currentId := reducedId
              currentShape := reducedShape
          if currentShape != outShape then
            throw <|
              s!"PyTorch graph import: {ctx}: reduction shape mismatch: computed " ++
                s!"{repr currentShape}, declared {repr outShape}"
          rawToTensor := rawToTensor.set! raw.id (some currentId)
        else
          let mut parents : Array Nat := #[]
          for p in raw.parents do
            match rawToTensor[p]? with
            | some (some tid) => parents := parents.push tid
            | some none =>
                throw <|
                  s!"PyTorch graph import: {ctx}: parent raw node {p} is not a tensor value " ++
                  "lowerable to NN.IR.Graph"
            | none => throw s!"PyTorch graph import: {ctx}: parent raw node {p} out of bounds"
          let kind ← parseOpKind ctx outShape raw.raw
          let tensorId := tensorNodes.size
          tensorNodes := tensorNodes.push
            { id := tensorId, parents := parents, kind := kind, outShape := outShape }
          rawToTensor := rawToTensor.set! raw.id (some tensorId)
  let inputId ←
    match rawToTensor[vg.inputId]? with
    | some (some id) => pure id
    | some none => throw "PyTorch graph import: graph input is not tensor-lowerable"
    | none => throw s!"PyTorch graph import: input id {vg.inputId} out of bounds"
  let outputIds ← vg.outputIds.mapM fun outputId =>
    match rawToTensor[outputId]? with
    | some (some id) => pure id
    | some none => throw s!"PyTorch graph import: graph output {outputId} is not tensor-lowerable"
    | none => throw s!"PyTorch graph import: output id {outputId} out of bounds"
  pure
    { graph := { nodes := tensorNodes }
      inputId := inputId
      outputIds := outputIds
      rawToTensor := rawToTensor
      mhaPayloadBindings := mhaPayloadBindings }

/-- Parse the graph object and lower PyTorch/FX values to the tensor IR. -/
def parseGraph (j : Json) : Except String CapturedGraph := do
  lowerValueGraph (← parseValueGraph j)

end Internal

/--
Parse and validate a captured PyTorch graph.

Success means:
- the JSON uses the TorchLean graph-artifact schema,
- every op is in the supported TorchLean IR subset,
- node ids are disciplined and topologically ordered,
- arities are valid, and
- declared output shapes match `NN.IR.Infer`.
-/
def parseGraph (j : Json) : Except String CapturedGraph := do
  match Internal.parseGraph j with
  | .error e => .error e
  | .ok cg =>
      match cg.graph.checkShapes with
      | .error e => .error e
      | .ok _ =>
          match cg.graph.getNode cg.inputId with
          | .error e => .error e
          | .ok _ =>
              match cg.outputIds.mapM cg.graph.getNode with
              | .error e => .error e
              | .ok _ => .ok cg

/-- Parse serialized parameters into the node-keyed Float payload. -/
def parsePayload (j : Json) : Except String (Payload Float) := do
  let valueGraph ← Internal.parseValueGraph j
  let captured ← parseGraph j
  let mut constParams : Array (Option (ConstFlat Float)) :=
    Array.replicate captured.graph.nodes.size none
  let mut linearParams : Array (Option (LinearWB Float)) :=
    Array.replicate captured.graph.nodes.size none
  let mut convolutionParams : Array (Option (ConvParams Float)) :=
    Array.replicate captured.graph.nodes.size none
  let mut layerNormParams : Array (Option (LayerNormParams Float)) :=
    Array.replicate captured.graph.nodes.size none
  let mut batchNormParams : Array (Option (BatchNormEvalParams Float)) :=
    Array.replicate captured.graph.nodes.size none
  for raw in valueGraph.nodes do
    let some tensorId? := captured.rawToTensor[raw.id]?
      | throw s!"PyTorch graph import: missing raw-to-tensor entry for node {raw.id}"
    match tensorId? with
    | none => pure ()
    | some tensorId =>
        let ctx := s!"node[{raw.id}]"
        if raw.kind = Wire.opTag .linear then
          let outDim ← Internal.natField ctx "out_dim" raw.raw
          let inDim ← Internal.natField ctx "in_dim" raw.raw
          let params : LinearWB Float :=
            { outDim := outDim
              inDim := inDim
              W := ← Internal.tensorField ctx "weight" [outDim, inDim] raw.raw
              b := ← Internal.tensorField ctx "bias" [outDim] raw.raw }
          linearParams := linearParams.set! tensorId (some params)
        else if raw.kind = Wire.opTag .conv then
          let spatialRank ← Internal.natField ctx "spatial_rank" raw.raw
          let inChannels ← Internal.natField ctx "in_channels" raw.raw
          let outChannels ← Internal.natField ctx "out_channels" raw.raw
          let kernel ← Internal.natTensorField ctx "kernel" spatialRank raw.raw
          let stride ← Internal.natTensorField ctx "stride" spatialRank raw.raw
          let padding ← Internal.natTensorField ctx "padding" spatialRank raw.raw
          let dilation ← Internal.natTensorField ctx "dilation" spatialRank raw.raw
          let paddingAfter ← Internal.natTensorField ctx "padding_after" spatialRank raw.raw
          let inputSpatial ← Internal.natTensorField ctx "input_spatial" spatialRank raw.raw
          let groups ← Internal.natField ctx "groups" raw.raw
          if _hInChannels : inChannels ≠ 0 then
            if hKernel : ∀ axis : Fin spatialRank, kernel.getScalar axis ≠ 0 then
              if hStride : ∀ axis : Fin spatialRank, stride.getScalar axis ≠ 0 then
                let spec :
                    Spec.ConvSpec spatialRank inChannels outChannels kernel stride padding Float :=
                  { kernel := ← Internal.tensorField ctx "weight"
                      (Shape.ofList (outChannels :: inChannels :: kernel.data.toList)) raw.raw
                    bias := ← Internal.tensorField ctx "bias" [outChannels] raw.raw }
                let params : ConvParams Float :=
                  { spatialRank := spatialRank
                    inChannels := inChannels
                    outChannels := outChannels
                    kernel := kernel
                    stride := stride
                    padding := padding
                    dilation := dilation
                    paddingAfter := paddingAfter
                    groups := groups
                    inputSpatial := inputSpatial
                    kernelNonzero := hKernel
                    strideNonzero := hStride
                    spec := spec }
                convolutionParams := convolutionParams.set! tensorId (some params)
              else
                throw s!"PyTorch graph import: {ctx}: convolution stride extents must be nonzero"
            else
              throw s!"PyTorch graph import: {ctx}: convolution kernel extents must be nonzero"
          else
            throw s!"PyTorch graph import: {ctx}: convolution input channels must be nonzero"
        else if raw.kind = Wire.opTag .layernorm then
          let outShape ←
            match raw.valueShape with
            | .tensor shape => pure shape
            | .tuple _ => throw s!"PyTorch graph import: {ctx}: LayerNorm output must be a tensor"
          let axis ← Internal.natField ctx "axis" raw.raw
          let normalizedShape := Shape.ofList (outShape.toList.drop axis)
          let params : LayerNormParams Float :=
            { normalizedShape := normalizedShape
              gamma := ← Internal.tensorField ctx "gamma" normalizedShape raw.raw
              beta := ← Internal.tensorField ctx "beta" normalizedShape raw.raw
              eps := ← Internal.floatField ctx "eps" raw.raw }
          layerNormParams := layerNormParams.set! tensorId (some params)
        else if raw.kind = Wire.opTag .batchNormEval then
          let channels ← Internal.natField ctx "channels" raw.raw
          let channelShape : Shape := [channels]
          let params : BatchNormEvalParams Float :=
            { c := channels
              gamma := ← Internal.tensorField ctx "gamma" channelShape raw.raw
              beta := ← Internal.tensorField ctx "beta" channelShape raw.raw
              mean := ← Internal.tensorField ctx "mean" channelShape raw.raw
              var := ← Internal.tensorField ctx "var" channelShape raw.raw
              eps := ← Internal.floatField ctx "eps" raw.raw }
          batchNormParams := batchNormParams.set! tensorId (some params)
  for binding in captured.mhaPayloadBindings do
    let raw ←
      match valueGraph.nodes[binding.rawId]? with
      | some value => pure value
      | none => throw s!"PyTorch graph import: missing raw MHA payload node {binding.rawId}"
    let ctx := s!"node[{binding.rawId}]"
    if raw.kind != Wire.mhaTuple then
      throw s!"PyTorch graph import: {ctx}: MHA payload binding designates `{raw.kind}`"
    let embedDim ← Internal.natField ctx "embed_dim" raw.raw
    let linearParamsFor (fieldPrefix : String) : Except String (LinearWB Float) := do
      pure
        { outDim := embedDim
          inDim := embedDim
          W := ← Internal.tensorField ctx s!"{fieldPrefix}_weight" [embedDim, embedDim] raw.raw
          b := ← Internal.tensorField ctx s!"{fieldPrefix}_bias" [embedDim] raw.raw }
    linearParams := linearParams.set! binding.queryId
      (some (← linearParamsFor "q"))
    linearParams := linearParams.set! binding.keyId
      (some (← linearParamsFor "k"))
    linearParams := linearParams.set! binding.valueId
      (some (← linearParamsFor "v"))
    linearParams := linearParams.set! binding.outputId
      (some (← linearParamsFor "out"))
    let scaleNode ←
      match captured.graph.nodes[binding.scaleId]? with
      | some node => pure node
      | none => throw s!"PyTorch graph import: missing generated MHA scale node {binding.scaleId}"
    let scale ← Internal.floatField ctx "scale" raw.raw
    let n := scaleNode.outShape.size
    constParams := constParams.set! binding.scaleId
      (some { n := n, v := Tensor.full (α := Float) [n] scale })
  pure
    { const? := fun id => (constParams[id]?).join
      linear? := fun id => (linearParams[id]?).join
      conv? := fun id => (convolutionParams[id]?).join
      layerNorm? := fun id => (layerNormParams[id]?).join
      batchNormEval? := fun id => (batchNormParams[id]?).join }

/--
Guarantee exposed by the parser: a successfully parsed graph is well-shaped.

This theorem is compact but important. It is the theorem downstream verification/export code can
quote when it receives a graph artifact through this importer.
-/
theorem parseGraph_wellShaped {j : Json} {cg : CapturedGraph}
    (h : parseGraph j = .ok cg) : cg.graph.WellShaped := by
  cases hparse : Internal.parseGraph j with
  | error e =>
      have hbad : Except.error e = Except.ok cg := by
        simp [parseGraph, hparse] at h
      cases hbad
  | ok cg0 =>
      cases hshape : cg0.graph.checkShapes with
      | error e =>
          have hbad : Except.error e = Except.ok cg := by
            simp [parseGraph, hparse, hshape] at h
          cases hbad
      | ok u =>
          cases hin : cg0.graph.getNode cg0.inputId with
          | error e =>
              have hbad : Except.error e = Except.ok cg := by
                simp [parseGraph, hparse, hshape, hin] at h
              cases hbad
          | ok inNode =>
              cases hout : cg0.outputIds.mapM cg0.graph.getNode with
              | error e =>
                  have hbad : Except.error e = Except.ok cg := by
                    simp [parseGraph, hparse, hshape, hin, hout] at h
                  cases hbad
              | ok outNode =>
                  have hok : cg0 = cg := by
                    simpa [parseGraph, hparse, hshape, hin, hout] using h
                  cases hok
                  unfold Graph.WellShaped Graph.checkShapes
                  exact hshape

end TorchExport
end PyTorch
end Import

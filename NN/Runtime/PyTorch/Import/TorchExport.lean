/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Check
public import NN.Runtime.PyTorch.Import.Core

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
  "output_id": 4,
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

The matching Python-side emitter lives in `NN.Runtime.PyTorch.Export.TorchExport`.
-/

@[expose] public section

namespace Import
namespace PyTorch
namespace TorchExport

open Lean
open Json
open Spec
open NN.IR

/-- A captured PyTorch graph lowered into TorchLean IR plus the designated input/output node ids. -/
structure CapturedGraph where
  /-- TorchLean's checked op-tagged graph. -/
  graph : Graph
  /-- Designated runtime input node id. -/
  inputId : Nat
  /-- Designated graph output node id. -/
  outputId : Nat
  deriving Repr

/--
Shape metadata for a PyTorch/FX value before it has been lowered to TorchLean's tensor-only IR.

PyTorch FX nodes may produce non-tensor containers. The most common example is
`nn.MultiheadAttention`, whose forward result is `(attn_output, attn_weights)`. We keep this
container structure in the import layer instead of treating every FX node as a tensor
node. Only tensor-valued nodes can be lowered into `NN.IR.Graph`.
-/
inductive ValueShape where
  | tensor (shape : Shape)
  | tuple (items : List Shape)
  deriving Repr, BEq

/-- One raw PyTorch/FX value node from `torchlean.ir.v1` JSON. -/
structure CapturedValueNode where
  /-- Raw FX node id from the JSON artifact. -/
  id : Nat
  /-- Raw parent ids. These may refer to tensor values or tuple/container values. -/
  parents : List Nat
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
  /-- Raw designated output id. -/
  outputId : Nat

/-! ## Small JSON helpers -/

def typeError {α : Type} (ctx expected : String) (j : Json) : Except String α :=
  .error s!"PyTorch graph import: {ctx}: expected {expected}, got {j}"

/-- Interpret a JSON number as a natural number. -/
def jsonNat (ctx : String) : Json → Except String Nat
  | .num n =>
      match n.toString.toNat? with
      | some k => .ok k
      | none => .error s!"PyTorch graph import: {ctx}: expected natural number, got {n}"
  | j => typeError ctx "natural number" j

/-- Interpret a JSON string. -/
def jsonString (ctx : String) : Json → Except String String
  | .str s => .ok s
  | j => typeError ctx "string" j

/-- Interpret a JSON array. -/
def jsonArray (ctx : String) : Json → Except String (Array Json)
  | .arr xs => .ok xs
  | j => typeError ctx "array" j

/-- Interpret a JSON object. -/
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
def parseNatList (ctx : String) (j : Json) : Except String (List Nat) := do
  let xs ← jsonArray ctx j
  xs.toList.mapM (jsonNat ctx)

/-- Convert a list of dimensions into TorchLean's nested `Shape` representation. -/
def shapeOfDims (dims : List Nat) : Shape :=
  dims.foldr Shape.dim Shape.scalar

/--
Parse a shape encoded as a dimension list.

Examples:
- `[]` means scalar;
- `[4]` means `Shape.dim 4 Shape.scalar`;
- `[2, 3]` means `Shape.dim 2 (Shape.dim 3 Shape.scalar)`.
-/
def parseShape (ctx : String) (j : Json) : Except String Shape := do
  pure (shapeOfDims (← parseNatList ctx j))

/-- Parse a node's parent id list. -/
def parseParents (ctx : String) (j : Json) : Except String (List Nat) :=
  parseNatList ctx j

def natField (ctx key : String) (o : StateDict) : Except String Nat := do
  jsonNat s!"{ctx}.{key}" (← field ctx key o)

def boolField (ctx key : String) (o : StateDict) : Except String Bool := do
  match ← field ctx key o with
  | .bool b => pure b
  | bad => typeError s!"{ctx}.{key}" "boolean" bad

def shapeField (ctx key : String) (o : StateDict) : Except String Shape := do
  parseShape s!"{ctx}.{key}" (← field ctx key o)

def natListField (ctx key : String) (o : StateDict) : Except String (List Nat) := do
  parseNatList s!"{ctx}.{key}" (← field ctx key o)

/-- Parse a JSON array whose elements are shape arrays. -/
def parseShapeList (ctx : String) (j : Json) : Except String (List Shape) := do
  let xs ← jsonArray ctx j
  xs.toList.mapM (parseShape ctx)

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
      pure (.tuple (← parseShapeList s!"{ctx}.tuple_shapes" (← field ctx "tuple_shapes" o)))
  | some (.str other) =>
      throw s!"PyTorch graph import: {ctx}: unsupported value_kind `{other}`"
  | some bad => typeError s!"{ctx}.value_kind" "string" bad

/--
Parse a TorchLean IR op kind.

The schema uses a stable string tag plus op-specific scalar fields. We intentionally avoid trying
to parse raw PyTorch operator names here; the Python adapter is responsible for translating
`torch.ops.aten.*` / FX targets into these TorchLean tags.
-/
def parseOpKind (ctx : String) (outShape : Shape) (o : StateDict) : Except String OpKind := do
  let tag ← jsonString s!"{ctx}.kind" (← field ctx "kind" o)
  match tag with
  | "input" => pure .input
  | "const" =>
      let valueShape ←
        match field? "value_shape" o with
        | some j => parseShape s!"{ctx}.value_shape" j
        | none => pure outShape
      pure (.const valueShape)
  | "permute" => pure (.permute (← natListField ctx "perm" o))
  | "detach" => pure .detach
  | "rand_uniform" => pure (.randUniform (← natField ctx "seed" o))
  | "bernoulli_mask" => pure (.bernoulliMask (← natField ctx "seed" o))
  | "add" => pure .add
  | "sub" => pure .sub
  | "mul_elem" => pure .mul_elem
  | "abs" => pure .abs
  | "sqrt" => pure .sqrt
  | "inv" => pure .inv
  | "max_elem" => pure .maxElem
  | "min_elem" => pure .minElem
  | "max_pool2d" =>
      pure (.maxPool2d (← natField ctx "kH" o) (← natField ctx "kW" o)
        (← natField ctx "stride" o))
  | "max_pool2d_pad" =>
      pure (.maxPool2dPad (← natField ctx "kH" o) (← natField ctx "kW" o)
        (← natField ctx "stride" o) (← natField ctx "padding" o))
  | "avg_pool2d" =>
      pure (.avgPool2d (← natField ctx "kH" o) (← natField ctx "kW" o)
        (← natField ctx "stride" o))
  | "avg_pool2d_pad" =>
      pure (.avgPool2dPad (← natField ctx "kH" o) (← natField ctx "kW" o)
        (← natField ctx "stride" o) (← natField ctx "padding" o))
  | "broadcast_to" =>
      pure (.broadcastTo (← shapeField ctx "from_shape" o) (← shapeField ctx "to_shape" o))
  | "reduce_sum" => pure (.reduceSum (← natField ctx "axis" o))
  | "reduce_mean" => pure (.reduceMean (← natField ctx "axis" o))
  | "sum" => pure .sum
  | "matmul" => pure .matmul
  | "linear" => pure .linear
  | "conv2d" =>
      pure (.conv2d
        (← natField ctx "inC" o) (← natField ctx "outC" o)
        (← natField ctx "kH" o) (← natField ctx "kW" o)
        (← natField ctx "stride" o) (← natField ctx "padding" o))
  | "relu" => pure .relu
  | "tanh" => pure .tanh
  | "sigmoid" => pure .sigmoid
  | "exp" => pure .exp
  | "log" => pure .log
  | "sin" => pure .sin
  | "cos" => pure .cos
  | "softmax" => pure (.softmax (← natField ctx "axis" o))
  | "layernorm" => pure (.layernorm (← natField ctx "axis" o))
  | "reshape" => pure (.reshape (← shapeField ctx "in_shape" o) (← shapeField ctx "out_shape" o))
  | "flatten" => pure (.flatten (← shapeField ctx "value_shape" o))
  | "concat" => pure (.concat (← natField ctx "axis" o))
  | "swap_first_two" => pure .swap_first_two
  | "transpose3d_last_two" => pure .transpose3dLastTwo
  | "mse_loss" => pure .mseLoss
  | other => throw s!"PyTorch graph import: {ctx}: unsupported TorchLean IR op kind `{other}`"

/-- Parse one node object from the graph JSON format. -/
def parseNode (j : Json) : Except String Node := do
  let o ← jsonObject "node" j
  let id ← natField "node" "id" o
  let ctx := s!"node[{id}]"
  let outShape ← shapeField ctx "shape" o
  let parents ← parseParents s!"{ctx}.parents" (← field ctx "parents" o)
  let kind ← parseOpKind ctx outShape o
  pure { id := id, parents := parents, kind := kind, outShape := outShape }

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
def parseValueGraphUnchecked (j : Json) : Except String CapturedValueGraph := do
  let o ← jsonObject "root" j
  match field? "format" o with
  | some (.str "torchlean.ir.v1") => pure ()
  | some (.str other) =>
      throw s!"PyTorch graph import: unsupported format `{other}` (expected `torchlean.ir.v1`)"
  | some bad => typeError "root.format" "string" bad
  | none => pure ()
  let inputId ← natField "root" "input_id" o
  let outputId ← natField "root" "output_id" o
  let nodeVals ← jsonArray "root.nodes" (← field "root" "nodes" o)
  let nodes ← nodeVals.mapM parseValueNode
  pure { nodes := nodes, inputId := inputId, outputId := outputId }

/-- Look up a raw value node by id. -/
def getValueNode (vg : CapturedValueGraph) (id : Nat) : Except String CapturedValueNode :=
  match vg.nodes[id]? with
  | some n =>
      if n.id = id then pure n
      else throw s!"PyTorch graph import: raw node id mismatch at index {id}: found node {n.id}"
  | none => throw s!"PyTorch graph import: raw node id {id} out of bounds"

/--
Lower a PyTorch/FX value graph to TorchLean's tensor-only IR.

Tuple/container nodes are preserved in `CapturedValueGraph`, but they do not become `NN.IR.Node`s.
TorchLean's verification and execution passes consume the clean tensor DAG, while the import layer
can explain container-valued PyTorch failures without changing their semantics.
-/
def lowerValueGraph (vg : CapturedValueGraph) : Except String CapturedGraph := do
  let mut rawToTensor : Array (Option Nat) := Array.replicate vg.nodes.size none
  let mut tensorNodes : Array Node := #[]
  for raw in vg.nodes do
    let ctx := s!"node[{raw.id}]"
    match raw.valueShape with
    | .tuple _items =>
        rawToTensor := rawToTensor.set! raw.id none
    | .tensor outShape =>
        if raw.kind = "tuple_getitem" then
          let index ← natField ctx "index" raw.raw
          let parentId ←
            match raw.parents with
            | [p] => pure p
            | _ => throw s!"PyTorch graph import: {ctx}: tuple_getitem expects one tuple parent"
          let parent ← getValueNode vg parentId
          match parent.valueShape with
          | .tuple _ =>
              if parent.kind = "multihead_attention" then
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
                  | [q, k, v] =>
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
                      throw s!"PyTorch graph import: {ctx}: MHA input raw node {xRawId} is not tensor-lowerable"
                  | none => throw s!"PyTorch graph import: {ctx}: MHA input raw node {xRawId} out of bounds"
                let xNode ←
                  match tensorNodes[xId]? with
                  | some n => pure n
                  | none => throw s!"PyTorch graph import: {ctx}: internal tensor node {xId} out of bounds"
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
                  { id := qId, parents := [xId], kind := .linear, outShape := xNode.outShape }
                let kId := tensorNodes.size
                tensorNodes := tensorNodes.push
                  { id := kId, parents := [xId], kind := .linear, outShape := xNode.outShape }
                let vId := tensorNodes.size
                tensorNodes := tensorNodes.push
                  { id := vId, parents := [xId], kind := .linear, outShape := xNode.outShape }
                let ktShape : Shape := .dim batch (.dim actualEmbed (.dim seqLen .scalar))
                let ktId := tensorNodes.size
                tensorNodes := tensorNodes.push
                  { id := ktId, parents := [kId], kind := .transpose3dLastTwo, outShape := ktShape }
                let scoresShape : Shape := .dim batch (.dim seqLen (.dim seqLen .scalar))
                let scoresId := tensorNodes.size
                tensorNodes := tensorNodes.push
                  { id := scoresId, parents := [qId, ktId], kind := .matmul, outShape := scoresShape }
                let scaleId := tensorNodes.size
                tensorNodes := tensorNodes.push
                  { id := scaleId, parents := [], kind := .const scoresShape, outShape := scoresShape }
                let scaledId := tensorNodes.size
                tensorNodes := tensorNodes.push
                  { id := scaledId, parents := [scoresId, scaleId], kind := .mul_elem,
                    outShape := scoresShape }
                let probsId := tensorNodes.size
                tensorNodes := tensorNodes.push
                  { id := probsId, parents := [scaledId], kind := .softmax 2, outShape := scoresShape }
                let ctxId := tensorNodes.size
                tensorNodes := tensorNodes.push
                  { id := ctxId, parents := [probsId, vId], kind := .matmul, outShape := xNode.outShape }
                let outId := tensorNodes.size
                tensorNodes := tensorNodes.push
                  { id := outId, parents := [ctxId], kind := .linear, outShape := outShape }
                rawToTensor := rawToTensor.set! raw.id (some outId)
              else
                throw <|
                  s!"PyTorch graph import: {ctx}: selected tensor from tuple-producing parent " ++
                  s!"`{parent.kind}`. Tuple/getitem is represented in the value graph, but this " ++
                  "tuple producer has no tensor-lowering rule yet. Decompose the producer into " ++
                  "supported tensor ops, or add a real semantic lowering for that operation."
          | .tensor _ =>
              throw s!"PyTorch graph import: {ctx}: getitem on a tensor value is not tensor-lowered"
        else
          let mut parents : List Nat := []
          for p in raw.parents do
            match rawToTensor[p]? with
            | some (some tid) => parents := parents ++ [tid]
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
  let outputId ←
    match rawToTensor[vg.outputId]? with
    | some (some id) => pure id
    | some none => throw "PyTorch graph import: graph output is not tensor-lowerable"
    | none => throw s!"PyTorch graph import: output id {vg.outputId} out of bounds"
  pure { graph := { nodes := tensorNodes }, inputId := inputId, outputId := outputId }

/-- Parse the graph object and lower PyTorch/FX values to the tensor IR without validation. -/
def parseGraphUnchecked (j : Json) : Except String CapturedGraph := do
  lowerValueGraph (← parseValueGraphUnchecked j)

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
  match parseGraphUnchecked j with
  | .error e => .error e
  | .ok cg =>
      match cg.graph.checkShapes with
      | .error e => .error e
      | .ok _ =>
          match cg.graph.getNode cg.inputId with
          | .error e => .error e
          | .ok _ =>
              match cg.graph.getNode cg.outputId with
              | .error e => .error e
              | .ok _ => .ok cg

/--
Guarantee exposed by the parser: a successfully parsed graph is well-shaped.

This is intentionally small but important. It is the theorem downstream verification/export code can
quote when it receives a graph artifact through this importer.
-/
theorem parseGraph_wellShaped {j : Json} {cg : CapturedGraph}
    (h : parseGraph j = .ok cg) : cg.graph.WellShaped := by
  cases hparse : parseGraphUnchecked j with
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
              cases hout : cg0.graph.getNode cg0.outputId with
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

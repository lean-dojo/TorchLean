/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Operator

/-!
# Graph wire format

`NN.IR.OpTag` identifies semantic operators. This module assigns each identity its fixed spelling in
`torchlean.ir.v1`: for example, `.mulElem` is written as `"mul_elem"`. Keeping this table separate
from diagnostic names lets us improve an error message while keeping existing graphs readable.
Exporters and importers use the same table.

The tag round trip is proved for every constructor. Static attributes are parsed separately;
`parseOpKind?` constructs only operations that need none. Tuple projections and supported container
nodes belong to the external value graph, not the tensor graph's semantic operator vocabulary.
-/

@[expose] public section

namespace Interop.PyTorch.Wire

open NN.IR

/-- Format marker written to the root `format` field. -/
def format : String := "torchlean.ir.v1"

/-- Projection of one component out of a tuple-valued FX node. -/
def tupleGetItem : String := "tuple_getitem"

/-- Container-valued `nn.MultiheadAttention` call kept in the value graph. -/
def mhaTuple : String := "multihead_attention"

/-- Legacy marker for a tuple producer without a lowering rule; the importer reports its limits. -/
def pyTuple : String := "py_tuple"

/-- Fixed constructor spelling in the `torchlean.ir.v1` artifact. -/
def opTag : OpTag → String
  | .input => "input"
  | .const => "const"
  | .permute => "permute"
  | .transpose => "transpose"
  | .detach => "detach"
  | .randUniform => "rand_uniform"
  | .bernoulliMask => "bernoulli_mask"
  | .add => "add"
  | .sub => "sub"
  | .mulElem => "mul_elem"
  | .abs => "abs"
  | .sqrt => "sqrt"
  | .inv => "inv"
  | .maxElem => "max_elem"
  | .minElem => "min_elem"
  | .maxPool => "max_pool"
  | .avgPool => "avg_pool"
  | .broadcastTo => "broadcast_to"
  | .reduceSum => "reduce_sum"
  | .reduceMean => "reduce_mean"
  | .sum => "sum"
  | .matmul => "matmul"
  | .linear => "linear"
  | .conv => "conv"
  | .batchNormEval => "batch_norm_eval"
  | .relu => "relu"
  | .tanh => "tanh"
  | .sigmoid => "sigmoid"
  | .softplus => "softplus"
  | .safeLog => "safe_log"
  | .exp => "exp"
  | .log => "log"
  | .sin => "sin"
  | .cos => "cos"
  | .softmax => "softmax"
  | .hardMaskedSoftmax => "hard_masked_softmax"
  | .layernorm => "layernorm"
  | .reshape => "reshape"
  | .flatten => "flatten"
  | .concat => "concat"
  | .mseLoss => "mse_loss"

/-- Parse a v1 constructor spelling into its semantic identity. -/
def parseOpTag? (s : String) : Option OpTag :=
  OpTag.all.find? fun tag => opTag tag == s

/-- Parse an attribute-free operation; attributed operators need their separate fields. -/
def parseOpKind? (s : String) : Option OpKind :=
  (parseOpTag? s).bind OpTag.toKind?

/-- JSON or Python `"kind": "<wire>"` field for an operator identity. -/
def kindField (tag : OpTag) : String := "\"kind\": \"" ++ opTag tag ++ "\""

/-- JSON or Python object holding only the kind field. -/
def kindObject (tag : OpTag) : String := "{" ++ kindField tag ++ "}"

/-- The wire string as a quoted JSON or Python literal. -/
def quotedOpTag (tag : OpTag) : String := "\"" ++ opTag tag ++ "\""

/-- Python set literal of quoted wire strings. -/
def quotedOpTags (tags : List OpTag) : String :=
  "{" ++ String.intercalate ", " (tags.map quotedOpTag) ++ "}"

/-- Every v1 tag parses back to its identity: the codec is complete and collision free. -/
theorem parse_op_tag (tag : OpTag) : parseOpTag? (opTag tag) = some tag := by
  cases tag <;> rfl

/-- Serializing and parsing an operation preserves its identity, regardless of its attributes. -/
theorem parse_op_kind_tag (kind : OpKind) :
    parseOpTag? (opTag kind.opTag) = some kind.opTag :=
  parse_op_tag kind.opTag

/-- The attribute-free v1 round trip reconstructs the original operation. -/
theorem parse_op_kind (kind : OpKind) (h : kind.opTag.hasAttributes = false) :
    parseOpKind? (opTag kind.opTag) = some kind := by
  rw [parseOpKind?, parse_op_tag]
  exact kind.to_kind_op_tag h

end Interop.PyTorch.Wire

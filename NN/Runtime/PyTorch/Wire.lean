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
`parse?` constructs only operations that need none. Tuple projections and supported container
nodes belong to the external value graph, not the tensor graph's semantic operator vocabulary.
-/

@[expose] public section

namespace Interop.PyTorch.Wire

open NN.IR

/-- Format marker written to the root `format` field. -/
def format : String := "torchlean.ir.v1"

/-- Projection of one component out of a tuple-valued FX node. -/
def projection : String := "tuple_getitem"

/-- Container-valued `nn.MultiheadAttention` call kept in the value graph. -/
def attention : String := "multihead_attention"

/-- Fixed constructor spelling in the `torchlean.ir.v1` artifact. -/
def tag : OpTag → String
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
  | .custom => "custom"

/-- Parse a v1 constructor spelling into its semantic identity. -/
def parseTag? (s : String) : Option OpTag :=
  OpTag.all.find? fun op => tag op == s

/-- Parse an attribute-free operation; attributed operators need their separate fields. -/
def parse? (s : String) : Option OpKind :=
  (parseTag? s).bind OpTag.toKind?

/-- JSON or Python `"kind": "<wire>"` field for an operator identity. -/
def field (op : OpTag) : String := "\"kind\": \"" ++ tag op ++ "\""

/-- JSON or Python object holding only the kind field. -/
def object (op : OpTag) : String := "{" ++ field op ++ "}"

/-- The wire string as a quoted JSON or Python literal. -/
def quoted (op : OpTag) : String := "\"" ++ tag op ++ "\""

/-- Python set expression of quoted wire strings. Empty input uses `set()`, not a dictionary. -/
def set : List OpTag → String
  | [] => "set()"
  | tags@(_ :: _) => "{" ++ String.intercalate ", " (tags.map quoted) ++ "}"

/-- Every v1 tag parses back to its identity: the codec is complete and collision free. -/
theorem parse_op_tag (tag : OpTag) : parseTag? (Wire.tag tag) = some tag := by
  cases tag <;> rfl

/-- Serializing and parsing an operation preserves its identity, regardless of its attributes. -/
theorem parse_op_kind_tag (kind : OpKind) :
    parseTag? (tag kind.opTag) = some kind.opTag :=
  parse_op_tag kind.opTag

/-- The attribute-free v1 round trip reconstructs the original operation. -/
theorem parse_op_kind (kind : OpKind) (h : kind.opTag.hasAttributes = false) :
    parse? (tag kind.opTag) = some kind := by
  rw [parse?, parse_op_tag]
  exact kind.to_kind_op_tag h

end Interop.PyTorch.Wire

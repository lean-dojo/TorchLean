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
`torchlean.ir.v2`: for example, `.mulElem` is written as `"mul_elem"`. Keeping this table separate
from diagnostic names lets us improve an error message while keeping existing graphs readable.
Exporters and importers use the same table.

The tag round trip is proved for every constructor. Static attributes are parsed separately;
`parse?` constructs only operations that need none. Tuple projections and supported container
nodes belong to the external value graph, not the tensor graph's semantic operator vocabulary.

Pointwise operators are spelled by operator rather than by their shared constructor: `"div"`,
`"pow"`, `"neg"`, `"rsqrt"`, `"erf"`, `"gelu"` and `"clamp_min"`. GELU writes PyTorch's
`"approximate"` string, and a rounded division writes its `"rounding_mode"`; true division has no
such field. The family spelling `"pointwise"` names the constructor in diagnostics and is not an
artifact kind.
-/

@[expose] public section

namespace Interop.PyTorch.Wire

open NN.IR

/-- Format marker written to the root `format` field. -/
def format : String := "torchlean.ir.v2"

/-- Projection of one component out of a tuple-valued FX node. -/
def projection : String := "tuple_getitem"

/-- Container-valued `nn.MultiheadAttention` call kept in the value graph. -/
def attention : String := "multihead_attention"

/-- Fixed constructor spelling in the `torchlean.ir.v2` artifact. -/
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
  | .topKMask => "top_k_mask"
  | .layernorm => "layernorm"
  | .reshape => "reshape"
  | .flatten => "flatten"
  | .concat => "concat"
  | .slice => "slice"
  | .mseLoss => "mse_loss"
  | .custom => "custom"
  | .pointwise => "pointwise"

/-- Parse a constructor spelling into its semantic identity. -/
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

/-- Fixed wire kind of a pointwise operator; its attribute is a separate field. -/
def pointwiseTag : Spec.PointwiseOp → String
  | .neg => "neg"
  | .rsqrt => "rsqrt"
  | .erf => "erf"
  | .gelu _ => "gelu"
  | .div _ => "div"
  | .pow => "pow"
  | .clampMin => "clamp_min"

/-- Attribute field of `gelu`: PyTorch's `approximate` argument. -/
def geluApproximateField : String := "approximate"

/-- Attribute field of a rounded `div`: PyTorch's `rounding_mode` argument. -/
def divRoundingField : String := "rounding_mode"

/-- Name of the attribute field written with a pointwise kind, if that kind has one. -/
def pointwiseAttributeField (kind : String) : Option String :=
  if kind == pointwiseTag (.gelu .none) then some geluApproximateField
  else if kind == pointwiseTag (.div .none) then some divRoundingField
  else none

/-- Value of a pointwise operator's attribute field, using PyTorch's strings. -/
def pointwiseAttribute : Spec.PointwiseOp → Option String
  | .gelu .none => some "none"
  | .gelu .tanh => some "tanh"
  | .div .trunc => some "trunc"
  | .div .floor => some "floor"
  | .neg | .rsqrt | .erf | .div .none | .pow | .clampMin => none

/-- Parse a pointwise operator from its kind and the value of its attribute field. -/
def parsePointwise? (kind : String) (value : Option String) : Option Spec.PointwiseOp :=
  Spec.PointwiseOp.all.find? fun op =>
    pointwiseTag op == kind && pointwiseAttribute op == value

/-- JSON or Python object holding a pointwise operator's kind and attribute fields. -/
def pointwiseObject (op : Spec.PointwiseOp) : String :=
  let kind := "\"kind\": \"" ++ pointwiseTag op ++ "\""
  match pointwiseAttributeField (pointwiseTag op), pointwiseAttribute op with
  | some key, some value => "{" ++ kind ++ ", \"" ++ key ++ "\": \"" ++ value ++ "\"}"
  | _, _ => "{" ++ kind ++ "}"

/-- No pointwise kind collides with an ordinary constructor spelling. -/
theorem parseTag_pointwiseTag (op : Spec.PointwiseOp) : parseTag? (pointwiseTag op) = none := by
  cases op <;> rfl

/-- Every pointwise operator parses back from its kind and attribute: the codec is complete and
collision free. -/
theorem parse_pointwise (op : Spec.PointwiseOp) :
    parsePointwise? (pointwiseTag op) (pointwiseAttribute op) = some op := by
  rcases op with _ | _ | _ | approximate | rounding | _ | _
  · rfl
  · rfl
  · rfl
  · cases approximate <;> rfl
  · cases rounding <;> rfl
  · rfl
  · rfl

/-- Every tag parses back to its identity: the codec is complete and collision free. -/
theorem parse_op_tag (tag : OpTag) : parseTag? (Wire.tag tag) = some tag := by
  cases tag <;> rfl

/-- Serializing and parsing an operation preserves its identity, regardless of its attributes. -/
theorem parse_op_kind_tag (kind : OpKind) :
    parseTag? (tag kind.opTag) = some kind.opTag :=
  parse_op_tag kind.opTag

/-- The attribute-free round trip reconstructs the original operation. -/
theorem parse_op_kind (kind : OpKind) (h : kind.opTag.hasAttributes = false) :
    parse? (tag kind.opTag) = some kind := by
  rw [parse?, parse_op_tag]
  exact kind.to_kind_op_tag h

end Interop.PyTorch.Wire

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Core

/-!
# Graph operators

An operation such as `.softmax 1` has a constructor, `.softmax`, and a static attribute: axis `1`.
`OpKind` keeps both. `OpTag` keeps just the constructor, so code that needs an operation's name or
parent count can ask for it without inventing an axis, shape, or convolution configuration.

Tensor parameters stay in the graph's external payload store. For example, `.linear` has no static
attributes here, but its weight and bias still come from that store. The parent count describes
dataflow: a linear node has one parent for its input tensor.

`NN.IR.Graph` adds node identities, dependency edges, and declared output shapes. The fixed
`torchlean.ir.v1` spelling of each constructor is defined separately in `NN.Runtime.PyTorch.Wire`.
-/

@[expose] public section

namespace NN.IR

open Spec TorchLean

/-- A row-major Boolean mask carried by an IR operation.

The payload records its logical tensor shape separately from the flat array so graph validation can
reject malformed serialized or programmatically constructed masks before evaluation. `true` means
that the corresponding entry is allowed.
-/
structure HardMask where
  shape : Shape
  allowed : Array Bool
  deriving Repr, DecidableEq

/-- Per-axis geometry for pooling and convolution operators.

All three tensors describe the same spatial suffix of the input tensor. Keeping the geometry in one
record prevents frontends from silently imposing a common stride or padding on every axis.
-/
structure WindowConfig where
  /-- Number of spatial axes. -/
  spatialRank : Nat
  /-- Window extent along each spatial axis. -/
  kernel : TorchLean.Tensor Nat [spatialRank]
  /-- Step along each spatial axis. -/
  stride : TorchLean.Tensor Nat [spatialRank]
  /-- Symmetric padding along each spatial axis. -/
  padding : TorchLean.Tensor Nat [spatialRank]
  deriving Repr, DecidableEq

/-- Shape metadata for an arbitrary-dimensional convolution.

Axes before `channelAxis` are preserved and mapped independently. The channel axis is replaced by
`outChannels`; every following axis is spatial and is governed by `window`.
-/
structure ConvConfig extends WindowConfig where
  /-- Spacing between kernel elements along each spatial axis. -/
  dilation : TorchLean.Tensor Nat [spatialRank] :=
    TorchLean.Tensor.dim fun _ => TorchLean.Tensor.scalar 1
  /-- Zero padding after the input along each spatial axis.

  The inherited `padding` field is the padding before the input. Keeping both sides explicit
  represents asymmetric padding without introducing rank-specific convolution variants. -/
  paddingAfter : TorchLean.Tensor Nat [spatialRank] := padding
  /-- Number of channel groups. `1` is an ordinary dense convolution. -/
  groups : Nat := 1
  /-- Axis containing the input channels. -/
  channelAxis : Nat
  /-- Expected extent of the input-channel axis. -/
  inChannels : Nat
  /-- Extent of the output-channel axis. -/
  outChannels : Nat
  deriving Repr, DecidableEq

/-- Operation kinds in an op-tagged computation graph.

`DecidableEq` is derived (and therefore `==` is available) so passes can compare operation tags
including their geometry payloads; the tensor-valued fields compare by row-major data. -/
inductive OpKind where
  | input
      -- Designated graph input (analogous to a PyTorch FX graph input).
  | const (valueShape : Shape)
      -- Constant tensor. We record the shape here, but keep the *value* in an external store
      -- (e.g. verifier parameters, exporter initializers).
  | permute (perm : Array Nat)
      -- Permute axes (0-based). Similar to `torch.permute`.
  | transpose (axis₁ axis₂ : Nat)
      -- Swap two arbitrary axes (0-based). Similar to `torch.transpose`.
  | detach
      -- Identity in the forward pass; marks a gradient stop at runtime (analogous to
      -- `Tensor.detach()`).
  | randUniform (seed : Nat)
      -- Deterministic U[0,1) tensor (seeded). We keep RNG explicit because verification needs a
      -- stable, replayable source of “randomness”.
  | bernoulliMask (seed : Nat)
      -- Deterministic {0,1} mask (seeded); parent is keepProb : scalar.
      -- This is the IR-level representation we use for dropout-style masks.
  | add
      -- Elementwise addition (broadcasting is explicit via `broadcastTo`).
  | sub
      -- Elementwise subtraction.
  | mulElem
      -- Elementwise multiplication.
  | abs
      -- Elementwise absolute value.
  | sqrt
      -- Elementwise square root (ties to the scalar backend semantics).
  | inv
      -- Elementwise reciprocal (1/x).
  | maxElem
      -- Elementwise max.
  | minElem
      -- Elementwise min.
  | maxPool (config : WindowConfig)
      -- Max pool over the final `config.spatialRank` axes. Leading axes are preserved.
  | avgPool (config : WindowConfig)
      -- Average pool over the same spatial suffix, including zero padding in the divisor.
  | broadcastTo (s₁ s₂ : Shape)
      -- Broadcast parent from s₁→s₂ (analogous to `torch.broadcast_to` / `Tensor.expand`).
  | reduceSum (axis : Nat)
      -- Sum along an axis (axis must be valid).
  | reduceMean (axis : Nat)
      -- Mean along an axis (axis must be valid).
  | sum
      -- Sum reduction to scalar (convenience op used by some loss/verification code paths).
  | matmul
      -- Matrix multiply over the final two axes, broadcasting leading batch axes and promoting
      -- 1D operands as `torch.matmul` does.
  | linear
      -- Affine layer `y = W x + b`. Parameters live in an external store keyed by node id;
      -- the sole parent is the activation input `x`.
  | conv (config : ConvConfig)
      -- Arbitrary-dimensional grouped convolution. Parameters live in an external store keyed by
      -- node id.
  | batchNormEval (channelAxis channels : Nat)
      -- Eval-mode BatchNorm along an explicit channel axis. Affine parameters and running
      -- statistics live in an external store keyed by node id.
  | relu | tanh | sigmoid | exp | log | sin | cos
      -- Common elementwise activations / nonlinearities.
  | softplus
      -- Stable pointwise softplus, using the scalar specification's sign branch.
  | safeLog
      -- Smooth log surrogate: log(softplus(x) + epsilon). The first parent has the output
      -- shape; the second is a scalar epsilon tensor, preserving the complete scalar value.
  | softmax (axis : Nat)
      -- Softmax along an axis.
  | hardMaskedSoftmax (mask : HardMask)
      -- Stable last-axis softmax with exactly zero mass at blocked entries.
  | layernorm (axis : Nat)
      -- LayerNorm over the suffix of dimensions starting at `axis`.
      -- PyTorch analogue: `F.layer_norm(x, normalized_shape=x.shape[axis:])`.
      -- An optional LayerNorm payload supplies gamma, beta, and epsilon. Without it, evaluation
      -- uses unit scale, zero bias, and the default epsilon. Lowered programs may instead express
      -- the affine transformation with surrounding `mulElem`/`add` nodes.
  | reshape (inShape outShape : Shape)
      -- Pure reshape (no data movement).
  | flatten (s : Shape)
      -- Flatten to a vector of length `Spec.Shape.size s`.
  | concat (axis : Nat)
      -- Concatenate along an axis (verifier/export may allow an arbitrary number of parents ≥ 2).
  | mseLoss
      -- Scalar mean squared error (used in some training/verification examples).
  deriving Repr, DecidableEq

/-- Permitted parent-count interval for an IR operation. -/
structure ParentArity where
  /-- Minimum number of parents required by the operation. -/
  min : Nat
  /-- Maximum number of parents, or `none` when no finite upper bound is imposed. -/
  max? : Option Nat
  deriving DecidableEq, Repr

/-- Structural metadata shared by all instances of an IR operation kind. -/
structure OpMetadata where
  /-- Short tag used in diagnostics; artifact codecs define their own spelling. -/
  tag : String
  /-- Permitted number of parent nodes. -/
  arity : ParentArity
  deriving DecidableEq, Repr

/-- Constructor identity of an `NN.IR.OpKind`, with the payload forgotten. -/
inductive OpTag where
  | input | const | permute | transpose | detach | randUniform | bernoulliMask
  | add | sub | mulElem | abs | sqrt | inv | maxElem | minElem
  | maxPool | avgPool | broadcastTo | reduceSum | reduceMean | sum
  | matmul | linear | conv | batchNormEval
  | relu | tanh | sigmoid | exp | log | sin | cos
  | softplus | safeLog
  | softmax | hardMaskedSoftmax | layernorm | reshape | flatten | concat | mseLoss
  deriving DecidableEq, Repr, Inhabited

namespace OpTag

/-- Every semantic operator identity, in declaration order. -/
def all : List OpTag :=
  [ .input, .const, .permute, .transpose, .detach, .randUniform, .bernoulliMask
  , .add, .sub, .mulElem, .abs, .sqrt, .inv, .maxElem, .minElem
  , .maxPool, .avgPool, .broadcastTo, .reduceSum, .reduceMean, .sum
  , .matmul, .linear, .conv, .batchNormEval
  , .relu, .tanh, .sigmoid, .exp, .log, .sin, .cos
  , .softplus, .safeLog
  , .softmax, .hardMaskedSoftmax, .layernorm, .reshape, .flatten, .concat, .mseLoss ]

/-- Every operator identity occurs in the enumeration. -/
theorem mem_all (tag : OpTag) : tag ∈ all := by
  cases tag <;> simp [all]

/-- Construct an operation when its tag supplies all the static information.

For example, `.relu` becomes `some .relu`, while `.softmax` returns `none` because its axis is still
missing. Tensor parameters such as linear weights are read separately from the payload store. -/
def toKind? : OpTag → Option OpKind
  | .input => some .input
  | .const => none
  | .permute => none
  | .transpose => none
  | .detach => some .detach
  | .randUniform => none
  | .bernoulliMask => none
  | .add => some .add
  | .sub => some .sub
  | .mulElem => some .mulElem
  | .abs => some .abs
  | .sqrt => some .sqrt
  | .inv => some .inv
  | .maxElem => some .maxElem
  | .minElem => some .minElem
  | .maxPool => none
  | .avgPool => none
  | .broadcastTo => none
  | .reduceSum => none
  | .reduceMean => none
  | .sum => some .sum
  | .matmul => some .matmul
  | .linear => some .linear
  | .conv => none
  | .batchNormEval => none
  | .relu => some .relu
  | .tanh => some .tanh
  | .sigmoid => some .sigmoid
  | .softplus => some .softplus
  | .safeLog => some .safeLog
  | .exp => some .exp
  | .log => some .log
  | .sin => some .sin
  | .cos => some .cos
  | .softmax => none
  | .hardMaskedSoftmax => none
  | .layernorm => none
  | .reshape => none
  | .flatten => none
  | .concat => none
  | .mseLoss => some .mseLoss

/-- Whether constructing the operation needs an axis, shape, or other static attributes. -/
def hasAttributes (tag : OpTag) : Bool := tag.toKind?.isNone

/-- Structural metadata shared by every instance of an operation.

Parameters such as linear weights live outside the parent edges, so `linear` has arity one. -/
def metadata : OpTag → OpMetadata
  | .input => ⟨"input", ⟨0, some 0⟩⟩
  | .const => ⟨"const", ⟨0, some 0⟩⟩
  | .permute => ⟨"permute", ⟨1, some 1⟩⟩
  | .transpose => ⟨"transpose", ⟨1, some 1⟩⟩
  | .detach => ⟨"detach", ⟨1, some 1⟩⟩
  | .randUniform => ⟨"rand_uniform", ⟨0, some 0⟩⟩
  | .bernoulliMask => ⟨"bernoulli_mask", ⟨1, some 1⟩⟩
  | .add => ⟨"add", ⟨2, some 2⟩⟩
  | .sub => ⟨"sub", ⟨2, some 2⟩⟩
  | .mulElem => ⟨"mul_elem", ⟨2, some 2⟩⟩
  | .abs => ⟨"abs", ⟨1, some 1⟩⟩
  | .sqrt => ⟨"sqrt", ⟨1, some 1⟩⟩
  | .inv => ⟨"inv", ⟨1, some 1⟩⟩
  | .maxElem => ⟨"max_elem", ⟨2, some 2⟩⟩
  | .minElem => ⟨"min_elem", ⟨2, some 2⟩⟩
  | .maxPool => ⟨"max_pool", ⟨1, some 1⟩⟩
  | .avgPool => ⟨"avg_pool", ⟨1, some 1⟩⟩
  | .broadcastTo => ⟨"broadcast_to", ⟨1, some 1⟩⟩
  | .reduceSum => ⟨"reduce_sum", ⟨1, some 1⟩⟩
  | .reduceMean => ⟨"reduce_mean", ⟨1, some 1⟩⟩
  | .sum => ⟨"sum", ⟨1, some 1⟩⟩
  | .matmul => ⟨"matmul", ⟨2, some 2⟩⟩
  | .linear => ⟨"linear", ⟨1, some 1⟩⟩
  | .conv => ⟨"conv", ⟨1, some 1⟩⟩
  | .batchNormEval => ⟨"batch_norm_eval", ⟨1, some 1⟩⟩
  | .relu => ⟨"relu", ⟨1, some 1⟩⟩
  | .tanh => ⟨"tanh", ⟨1, some 1⟩⟩
  | .sigmoid => ⟨"sigmoid", ⟨1, some 1⟩⟩
  | .softplus => ⟨"softplus", ⟨1, some 1⟩⟩
  | .safeLog => ⟨"safe_log", ⟨2, some 2⟩⟩
  | .exp => ⟨"exp", ⟨1, some 1⟩⟩
  | .log => ⟨"log", ⟨1, some 1⟩⟩
  | .sin => ⟨"sin", ⟨1, some 1⟩⟩
  | .cos => ⟨"cos", ⟨1, some 1⟩⟩
  | .softmax => ⟨"softmax", ⟨1, some 1⟩⟩
  | .hardMaskedSoftmax => ⟨"hard_masked_softmax", ⟨1, some 1⟩⟩
  | .layernorm => ⟨"layernorm", ⟨1, some 1⟩⟩
  | .reshape => ⟨"reshape", ⟨1, some 1⟩⟩
  | .flatten => ⟨"flatten", ⟨1, some 1⟩⟩
  | .concat => ⟨"concat", ⟨2, none⟩⟩
  | .mseLoss => ⟨"mse_loss", ⟨2, some 2⟩⟩

end OpTag

namespace OpKind

/-- Forget the static attributes of an operation. -/
def opTag : OpKind → OpTag
  | .input => .input
  | .const .. => .const
  | .permute .. => .permute
  | .transpose .. => .transpose
  | .detach => .detach
  | .randUniform .. => .randUniform
  | .bernoulliMask .. => .bernoulliMask
  | .add => .add
  | .sub => .sub
  | .mulElem => .mulElem
  | .abs => .abs
  | .sqrt => .sqrt
  | .inv => .inv
  | .maxElem => .maxElem
  | .minElem => .minElem
  | .maxPool .. => .maxPool
  | .avgPool .. => .avgPool
  | .broadcastTo .. => .broadcastTo
  | .reduceSum .. => .reduceSum
  | .reduceMean .. => .reduceMean
  | .sum => .sum
  | .matmul => .matmul
  | .linear => .linear
  | .conv .. => .conv
  | .batchNormEval .. => .batchNormEval
  | .relu => .relu
  | .tanh => .tanh
  | .sigmoid => .sigmoid
  | .softplus => .softplus
  | .safeLog => .safeLog
  | .exp => .exp
  | .log => .log
  | .sin => .sin
  | .cos => .cos
  | .softmax .. => .softmax
  | .hardMaskedSoftmax .. => .hardMaskedSoftmax
  | .layernorm .. => .layernorm
  | .reshape .. => .reshape
  | .flatten .. => .flatten
  | .concat .. => .concat
  | .mseLoss => .mseLoss

/-- Structural metadata, obtained from the operation's constructor identity. -/
def metadata (kind : OpKind) : OpMetadata := kind.opTag.metadata

/-- Forgetting attributes and reconstructing an attribute-free operation preserves it. -/
theorem to_kind_op_tag (kind : OpKind) (h : kind.opTag.hasAttributes = false) :
    kind.opTag.toKind? = some kind := by
  cases kind <;> first | rfl | simp [opTag, OpTag.hasAttributes, OpTag.toKind?] at h

/--
The minimum number of parent nodes expected by an `OpKind`.
-/
def minParents (kind : OpKind) : Nat := kind.metadata.arity.min

/--
An optional maximum number of parent nodes expected by an `OpKind`.

For `concat`, the verifier permits an arbitrary number of inputs (at least 2), so this returns
`none`.
-/
def maxParents? (kind : OpKind) : Option Nat := kind.metadata.arity.max?

/-- A short tag for error messages and debugging output. -/
def tag (kind : OpKind) : String := kind.metadata.tag

/--
Human-facing operation description including operation-local parameters.

`tag` is short and stable for grouping/log filtering. `describe` is for diagnostics:
it prints axes, shapes, seeds, and convolution/pooling metadata so malformed graph dumps are useful
without cross-referencing the original builder.
-/
def describe : OpKind → String
  | .input => "input"
  | .const valueShape => s!"const(shape={repr valueShape})"
  | .permute perm => s!"permute(perm={repr perm})"
  | .transpose axis₁ axis₂ => s!"transpose(axis1={axis₁}, axis2={axis₂})"
  | .detach => "detach"
  | .randUniform seed => s!"rand_uniform(seed={seed})"
  | .bernoulliMask seed => s!"bernoulli_mask(seed={seed})"
  | .add => "add"
  | .sub => "sub"
  | .mulElem => "mul_elem"
  | .abs => "abs"
  | .sqrt => "sqrt"
  | .inv => "inv"
  | .maxElem => "max_elem"
  | .minElem => "min_elem"
  | .maxPool config => s!"max_pool(config={repr config})"
  | .avgPool config => s!"avg_pool(config={repr config})"
  | .broadcastTo s₁ s₂ => s!"broadcastTo(from={repr s₁}, to={repr s₂})"
  | .reduceSum axis => s!"reduce_sum(axis={axis})"
  | .reduceMean axis => s!"reduce_mean(axis={axis})"
  | .sum => "sum"
  | .matmul => "matmul"
  | .linear => "linear(payload=node_id)"
  | .conv config => s!"conv(config={repr config}, payload=node_id)"
  | .batchNormEval channelAxis channels =>
      s!"batch_norm_eval(channelAxis={channelAxis}, channels={channels}, payload=node_id)"
  | .relu => "relu"
  | .tanh => "tanh"
  | .sigmoid => "sigmoid"
  | .softplus => "softplus"
  | .safeLog => "safe_log(epsilon=scalar_parent)"
  | .exp => "exp"
  | .log => "log"
  | .sin => "sin"
  | .cos => "cos"
  | .softmax axis => s!"softmax(axis={axis})"
  | .hardMaskedSoftmax mask =>
      s!"hard_masked_softmax(maskShape={repr mask.shape})"
  | .layernorm axis => s!"layernorm(axis={axis})"
  | .reshape inShape outShape => s!"reshape(from={repr inShape}, to={repr outShape})"
  | .flatten s => s!"flatten(shape={repr s})"
  | .concat axis => s!"concat(axis={axis})"
  | .mseLoss => "mse_loss"

end OpKind

end NN.IR

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Op.Capability
public import TorchLean.Tensor.Op.Rule
public import TorchLean.Tensor.Elementwise
public import TorchLean.Tensor.Random
public import TorchLean.Tensor.Arithmetic
public import TorchLean.Tensor.Word
public import TorchLean.Tensor.Cumulative
public import TorchLean.Tensor.Scatter
public import TorchLean.Tensor.Sort
public import TorchLean.Tensor.Layers.Attention
public import TorchLean.Tensor.Layers.Linear
public import TorchLean.Tensor.Layers.RealFFT
public import TorchLean.Tensor.Layers.StateSpace
public import TorchLean.Tensor.Layers.MoE
public import TorchLean.Tensor.Layers.Loss
public import TorchLean.Tensor.Layers.Loss.Axis
public import TorchLean.Tensor.Layers.Activation
public import TorchLean.Tensor.Layers.Dropout
public import TorchLean.Tensor.Layers.Embedding
public import TorchLean.Tensor.Layers.Normalization.LayerNorm
public import TorchLean.Tensor.Layers.Normalization.BatchNorm
public import TorchLean.Tensor.Layers.Normalization
public import TorchLean.Tensor.Window
public import TorchLean.Tensor.Contract
public import TorchLean.Tensor.Movement
public import TorchLean.Tensor.Reduce
public import TorchLean.Tensor.Op.Pointwise

/-!
# Registered operators

`TorchLean.Op ins outs` names a registered operator together with its attributes, indexed by its
input and output ports, one constructor per operator. The table holds the elementwise operators and
activations, `detach` (a detached input), the comparisons and predicates (Boolean outputs, frozen
inputs), `select` (a Boolean input) and the cumulative operators. `sort`, `topk`, `cummax`,
`cummin`, `argmax` and `argmin` have index outputs, and `gather` (over any axes, with batch
dimensions) and `scatter` take an index-tuple input. The data movements are `indices`, `reshape`,
`permute`, `broadcast`, `flip`, `slice`, `indexAt` (one index of an axis, the axis removed) and
`cat` (any number of inputs). `matmul` and `scaledMatmul` broadcast their batch prefixes. The layer
operators are `linear` (over leading axes), `conv`, `convTranspose`, `pool`, `smoothMaxPool`,
`layerNorm` (over the last axis, the leading axes folded into the rows of the matrix rule) and the
batch normalizations. `batchNorm` uses batch statistics along the first axis and `batchNormRunning`
frozen running statistics along any axis. The table also holds the losses, `dropout` with a uniform
input, `attention` (scaled dot-product attention), `attentionFromScores` (its weighted stage after
the score product) and the fused `multiheadAttention` (over any leading axes). The reductions are
`reduce` (a sum, product, maximum or minimum over any axes) and `mean`, and `maskedSoftmax` is the
softmax of the entries a Boolean mask input allows. `topKMask` and `bernoulliMask` are masks of the
scalar with no gradient, `rfft` and `irfft` are real Fourier transforms along any axis, and
`selectiveScan` is a diagonal scan along any axis. `randUniform` and `randn` are keyed, and the rule
of `custom` is the checked program of its node. Operators whose per-frontend definitions differ in
their numerical policy are separate constructors, named for the policy (`max` and `fmax`,
`crossEntropy` and `crossEntropyProbabilities`). Scalar attributes and captured operands are frozen
inputs.

The logical operators `logicalAnd`, `logicalOr` and `logicalXor` take and return Boolean tensors,
`cast` reads Boolean or index entries as scalars, `pad` pads with a scalar input along every axis,
and `contract` adds products over any paired axes, with batch pairs. The bitwise operators, the
shifts, `populationCount`, `countLeadingZeros` and `mulhi` take and return integer words, computed
with `BitVec`'s operators and the functions of `Word`. `bitcast` reads each word at another
sort of the same width, and `cbrt` and `nextafter` read the scalar's `Cbrt` and `NextAfter`.
`encodeIndex` and `decodeIndex` convert index tuples to and from integer words in the IR's layout
(`Port.irShape`). Encoding fails at a component the integer type cannot hold and decoding at a
component outside its bound. `decodeIndex` has a rule only where its output has a value (every bound
positive, or no entry).

`Op.check` checks the attributes a type does not (a pooling window, distinct reduction axes). A
recorder runs it before reading any input, and the wire decoder does not.

The element argument of `reshape`, `permute`, `broadcast`, `flip`, `slice`, `pad`, `cat`,
`gather`, `scatter` and `select` (`Op.Element`) and of the arithmetic operators and comparisons
(`Op.NumericElement`) defaults to `.scalar`, the model's scalar. At `.bool` or
`.integer signed width` the operator moves Booleans or integer words as it moves scalars, or
computes PyTorch's integer meaning with `BitVec` and `Word` (wrapping `+`, `-`, `*` and
negation, the order of the type), without gradients. An integer division or remainder fails its
value check at a zero divisor, an integer true division (`div .none`) has no rule, and a Boolean
scatter only replaces.

Elementwise operators take equal shapes and leave broadcasting to the separate operator
`broadcast`.

`Op.rule?` in `TorchLean.Tensor.Op.Table.Rule` gives each operator its rule. The equation lemmas of
its arms are in one module per operator family: `TorchLean.Tensor.Op.Table.Elementwise`,
`TorchLean.Tensor.Op.Table.Compare`, `TorchLean.Tensor.Op.Table.Word`,
`TorchLean.Tensor.Op.Table.Reduce`, `TorchLean.Tensor.Op.Table.Movement`,
`TorchLean.Tensor.Op.Table.Linear`, `TorchLean.Tensor.Op.Table.Conv`,
`TorchLean.Tensor.Op.Table.Normalization`, `TorchLean.Tensor.Op.Table.Attention`,
`TorchLean.Tensor.Op.Table.Loss` and `TorchLean.Tensor.Op.Table.Random`.

We wrote each arm of `Op.rule?` by hand: the forward from the operator's `Tensor` function, the
pullback and the tangent from its derivative, and a scalar function such as `Activation.reluDeriv`
or `Context.sign` to fix the value at a kink or a NaN. The notes on the activations, `abs`, `clamp`,
`cumsum`, `mseLoss` and `scatter` compare those choices with torch 2.8.0, the version the parity
checks pin (`scripts/checks/requirements-interop.txt`): its rules in
`tools/autograd/derivatives.yaml` and `torch/csrc/autograd/FunctionsManual.cpp`, and its CPU results
in binary64. Over ℝ, the certificates of `TorchLean.NN.Op.Pointwise` and `TorchLean.NN.Op.Softmax`
prove that the pullbacks of those activations, `abs` and `clamp` are the adjoints of their
derivatives (`Op.softplus_backward_eq_adjoint`, `Op.relu_backward_eq_adjoint` away from zero,
`Op.softmax_backward_eq_adjoint`), except at the kinks that `TorchLean.NN.Op.Pointwise` lists.
`scatter`'s torch test is named in its docstring. The certificates in `TorchLean.NN.Op.Reduce`
and `TorchLean.NN.Op.Loss` also prove the pullbacks of `cumsum` and `mseLoss` over the reals.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)
open TorchLean.Shape (size size_nil)

namespace TorchLean

open TorchLean.Tensor

/-- What `Op.cast` converts to the scalar: Booleans, or indices below `bound`. -/
inductive Op.CastSource where
  /-- Booleans: `true` is `1` and `false` is `0`. -/
  | bool
  /-- Indices below `bound`, each as its natural number. -/
  | index (bound : Nat)
  deriving DecidableEq, Repr, Inhabited

/-- The input port of `Op.cast` from `source` at the shape `s`. -/
abbrev Op.CastSource.port : Op.CastSource → Shape → Port
  | .bool, s => .bool s
  | .index bound, s => .index [bound] s

/-- What `Op.bitcast` reads or writes: the scalar's interchange word, or an integer word. -/
inductive Op.Word where
  /-- The scalar's word, through its `Bits`. -/
  | scalar
  /-- A word of an integer type, two's complement when `signed`. -/
  | integer (signed : Bool)
  deriving DecidableEq, Repr, Inhabited

/-- The input port of `Op.bitcast` from `word` at `width` bits and the shape `s`: a frozen scalar,
as the input receives no gradient, or an integer. -/
abbrev Op.Word.input : Op.Word → Nat → Shape → Port
  | .scalar, _, s => .frozen s
  | .integer signed, width, s => .integer signed width s

/-- The output port of `Op.bitcast` to `word` at `width` bits and the shape `s`. -/
abbrev Op.Word.output : Op.Word → Nat → Shape → Port
  | .scalar, _, s => .value s
  | .integer signed, width, s => .integer signed width s

/-- The element type of the data ports of a generalized operator (the movements, `select`,
`gather`, `scatter` and `pad`): the model's scalar, Booleans, or words of `width` bits of an integer
type, two's complement when `signed`. -/
inductive Op.Element where
  /-- The model's scalar, at the port's role (a value or a frozen scalar). -/
  | scalar
  /-- Booleans. -/
  | bool
  /-- Words of `width` bits of an integer type, two's complement when `signed`. -/
  | integer (signed : Bool) (width : Nat)
  deriving DecidableEq, Repr, Inhabited

/-- The element type of an arithmetic operator or a comparison: the model's scalar, or words of
`width` bits of an integer type, two's complement when `signed`. -/
inductive Op.NumericElement where
  /-- The model's scalar, at the port's role. -/
  | scalar
  /-- Words of `width` bits of an integer type, two's complement when `signed`. -/
  | integer (signed : Bool) (width : Nat)
  deriving DecidableEq, Repr, Inhabited

/-- The element of a numeric element. -/
abbrev Op.NumericElement.toElement : Op.NumericElement → Op.Element
  | .scalar => .scalar
  | .integer signed width => .integer signed width

/-- The numeric element of an element: the scalar or the integer type, and `none` at Booleans. -/
def Op.Element.numeric? : Op.Element → Option Op.NumericElement
  | .scalar => some .scalar
  | .bool => none
  | .integer signed width => some (.integer signed width)

/-- The port of an operand of element `e` at the shape `s`: the scalar at the port's `role`
(`⟨role, s⟩`), or the native Boolean or integer port, which has no gradient. -/
abbrev Op.Element.port (role : Port.Kind) : Op.Element → Shape → Port
  | .scalar, s => ⟨role, s⟩
  | .bool, s => .bool s
  | .integer signed width, s => .integer signed width s

/-- The port of an operand of numeric element `e` at the shape `s`: the scalar at the port's
`role`, or the native integer port. -/
abbrev Op.NumericElement.port (role : Port.Kind) : Op.NumericElement → Shape → Port
  | .scalar, s => ⟨role, s⟩
  | .integer signed width, s => .integer signed width s

/-- The operators of the table, indexed by their input and output ports. -/
inductive Op : List Port → List Port → Type where
  /-- Elementwise ReLU. We take torch's subgradient `0` at the kink, but at a NaN torch's
  `threshold_backward` passes `g` through, where our slope `Activation.reluDeriv` is `0` and we send
  `0 * g`. At `-0` we return `+0` and `torch.relu` keeps `-0`, so `TorchLean.PyTorch.emit` writes
  `torch.relu(x) + 0.0`, whose words `TorchLeanTest.Interop.IRPyTorch` compares with ours at `±0`,
  `±1`, `±∞` and a NaN. -/
  | relu {s : Shape} : Op [.value s] [.value s]
  /-- Elementwise logistic sigmoid. -/
  | sigmoid {s : Shape} : Op [.value s] [.value s]
  /-- Elementwise hyperbolic tangent. -/
  | tanh {s : Shape} : Op [.value s] [.value s]
  /-- Elementwise exponential. -/
  | exp {s : Shape} : Op [.value s] [.value s]
  /-- Elementwise absolute value, with subgradient zero at the origin. The pullback is
  `Tensor.sign x * g`, and `Context.sign` keeps a zero's sign and a NaN, so seed `1` sends `-0` at
  `x = -0` and a NaN at a NaN, where torch's `grad * self.sgn()` sends `+0` at both. -/
  | abs {s : Shape} : Op [.value s] [.value s]
  /-- Elementwise square root by `policy`: at `.raw`, the default, IEEE 754 squareRoot (`-0` at
  `-0`, a NaN below zero) with pullback `g / (2 |y|)` at the root `y`, so `+∞ · g` at `±0`. At
  `.clampNegative` it is the root of `max x 0`, with pullback `0 * g` where `x > 0` is false. At a
  NaN `x` that root is a NaN and the pullback `0 * g` is a zero with the sign of a finite `g` and a
  NaN at an infinite or NaN `g`. -/
  | sqrt {s : Shape} (policy : Tensor.SqrtPolicy := .raw) : Op [.value s] [.value s]
  /-- Elementwise sine. -/
  | sin {s : Shape} : Op [.value s] [.value s]
  /-- Elementwise cosine. -/
  | cos {s : Shape} : Op [.value s] [.value s]
  /-- Elementwise negation. -/
  | neg {s : Shape} (e : Op.NumericElement := .scalar) : Op [e.port .value s] [e.port .value s]
  /-- Elementwise addition. -/
  | add {s : Shape} (e : Op.NumericElement := .scalar) :
      Op [e.port .value s, e.port .value s] [e.port .value s]
  /-- Elementwise multiplication. -/
  | mul {s : Shape} (e : Op.NumericElement := .scalar) :
      Op [e.port .value s, e.port .value s] [e.port .value s]
  /-- Elementwise IEEE 754-2019 `maximum`: a NaN where either operand is one, and `+0` above `-0`.
  The larger operand receives `g` and gives the tangent, and the other receives `+0` at every `g`.
  Where they are equal (`+0` and `-0` included) each receives half of `g` and the tangent is the
  average of theirs, by convention also where one is a NaN (`torch.maximum` gives `g` to both
  there). `max x x` receives `g` once, by the alias rule, which is enabled where
  `Numeric.QuotientArithmetic` is supported. -/
  | max {s : Shape} (e : Op.NumericElement := .scalar) :
      Op [e.port .value s, e.port .value s] [e.port .value s]
  /-- Elementwise natural logarithm, with pullback `g / x`. An IEEE format gives `log 0 = -∞` and
  a NaN below zero, while over ℝ it is `Real.log`, with `log 0 = 0` and `log x = log |x|`
  (`Real.log_zero`, `Real.log_abs`). -/
  | log {s : Shape} : Op [.value s] [.value s]
  /-- `Tensor.detach`: the input, with the differentiation metadata of a scalar that carries any
  cleared (`Context.stopGradient?`). The input is detached, so it receives no gradient, and the
  tangent is zero. -/
  | detach {s : Shape} : Op [.detached s] [.value s]
  /-- Elementwise softplus. This is `torch.nn.functional.softplus` (PyTorch 2.8 docs) at `beta = 1`
  without its `threshold = 20`, so at `21` we return `21.000000000758256` with slope
  `0.9999999992417439` in binary64 where torch returns `21` with slope `1`. The PyTorch capture
  lowers `F.softplus` to torch's formula, threshold included (`python/torchlean/_pointwise.py`). -/
  | softplus {s : Shape} : Op [.value s] [.value s]
  /-- Elementwise SiLU, `x * sigmoid x`. It's the Sigmoid Linear Unit of Hendrycks and Gimpel (2016,
  arXiv:1606.08415, Section 2) and of `torch.nn.functional.silu` (PyTorch 2.8 docs), except at `±∞`,
  where we return the limits and torch gives a NaN at `-∞` and NaN slopes at both.
  `TorchLean.PyTorch.emit` selects the limits with `torch.where`, and
  `TorchLeanTest.Interop.IRPyTorch` checks the emitted values and slopes at both infinities. -/
  | silu {s : Shape} : Op [.value s] [.value s]
  /-- Elementwise square. The pullback and the tangent multiply by the rounded `2 * x`, which in a
  binary format overflows for `|x|` above half the largest finite value, so they return an infinity
  or a NaN where the exact product with the seed is finite. -/
  | square {s : Shape} : Op [.value s] [.value s]
  /-- Elementwise reciprocal. -/
  | reciprocal {s : Shape} : Op [.value s] [.value s]
  /-- Elementwise subtraction. -/
  | sub {s : Shape} (e : Op.NumericElement := .scalar) :
      Op [e.port .value s, e.port .value s] [e.port .value s]
  /-- Elementwise division, with the quotient rounded as `torch.div`'s `rounding_mode`: true
  division by default, flooring division (`Tensor.floorDivide`) at `.floor`, and at `.trunc` the
  flooring division of `|a|` by `|b|` with the sign of `a / b` (`trunc (a / b)` over the reals,
  `Tensor.copysign_floorDivide_abs_real`). In a binary format `.trunc` gives a NaN
  where `a / b` is one, `a / b` where that is infinite, and a zero with the sign of `a / b` where
  `|a| < |b|`, as in PyTorch, and `9` at `(1, 0.1)`, where PyTorch truncates the rounded quotient
  to `10`. -/
  | div {s : Shape} (rounding : DivRounding := .none) (e : Op.NumericElement := .scalar) :
      Op [e.port .value s, e.port .value s] [e.port .value s]
  /-- Elementwise IEEE 754-2019 `minimum`: a NaN where either operand is one, and `-0` below `+0`.
  The smaller operand receives `g` and gives the tangent, and the other receives `+0` at every `g`.
  Where they are equal (`+0` and `-0` included) each receives half of `g` and the tangent is the
  average of theirs, by convention also where one is a NaN (`torch.minimum` gives `g` to both
  there). `min x x` receives `g` once, by the alias rule, which is enabled where
  `Numeric.QuotientArithmetic` is supported. -/
  | min {s : Shape} (e : Op.NumericElement := .scalar) :
      Op [e.port .value s, e.port .value s] [e.port .value s]
  /-- Cumulative sum along `dim`. The pullback is torch's `cumsum_backward`, the cumulative sum of
  `g` from the last index down (`Tensor.cumsumBackward`), so `cumsum [1, 2, 3]` with upstream
  `[0.5, -1, 2]` sends `[1.5, 1, 2]`. The tangent is the cumulative sum of the tangent, torch's
  linear forward rule. -/
  | cumsum {s : Shape} (dim : Fin s.rank) : Op [.value s] [.value s]
  /-- `log (softplus x + ε)`, with `ε` a frozen scalar input. -/
  | safeLog {s : Shape} : Op [.value s, .frozen []] [.value s]
  /-- Clamp to the frozen scalar bounds `[lo, hi]`, with pullback `g` where `lo < x < hi` and
  `0 * g` elsewhere. Torch's `clamp_backward` takes the subgradient `1` at a bound, keeping `g` on
  the closed interval, so at `x = lo` with seed `1` torch sends `1` and we send `0`. -/
  | clamp {s : Shape} : Op [.value s, .frozen [], .frozen []] [.value s]
  /-- Multiply by a frozen scalar input. -/
  | scale {s : Shape} : Op [.value s, .frozen []] [.value s]
  /-- ELU with a frozen scalar `alpha`. At the kink we take the left slope `alpha * exp 0`, as
  torch's `elu_backward` does, so with `alpha = 0.5` and seed `1` both send `0.5` at `x = 0`. -/
  | elu {s : Shape} : Op [.value s, .frozen []] [.value s]
  /-- Leaky ReLU with a frozen scalar negative slope. As in torch's `leaky_relu_backward`, the
  pullback takes the negative slope wherever `x > 0` fails, `x = 0` and a NaN included
  (`Activation.leakyReLUDeriv`). -/
  | leakyReLU {s : Shape} : Op [.value s, .frozen []] [.value s]
  /-- Softmax along `dim`. Torch's `_softmax_backward_data` reads the saved output, and
  `Tensor.softmaxBackward` recomputes it from the logits and centers `g` at a maximal logit first
  (`Softmax.pivot`). The Jacobian `diag y - y yᵀ` is symmetric, so one map serves as the pullback
  and the forward-mode rule. -/
  | softmax {s : Shape} (dim : Fin s.rank) : Op [.value s] [.value s]
  /-- Log-softmax along `dim`. The pullback and the tangent read the output `y`, as torch's
  `_log_softmax_backward_data` reads its saved result, and the prepared node computes `y` once for
  its value and its pullback. -/
  | logSoftmax {s : Shape} (dim : Fin s.rank) : Op [.value s] [.value s]
  /-- The affine map `W x + b` at every index of the leading shape `lead`, with the weight, bias
  and input as ports, in that order. At `lead = []` it is the vector rule. At a leading shape of
  positive rank (possibly with zero extents) the weight and bias gradients sum over the flattened
  leading indices from zero, so an empty leading axis gives zero parameter gradients. -/
  | linear {lead : Shape} {inDim outDim : Nat} :
      Op [.value [outDim, inDim], .value [outDim], .value (lead ++ [inDim])]
        [.value (lead ++ [outDim])]
  /-- `matmul (softmax scores * multiplier) value` over the last two axes, with an optional mask of
  allowed score entries. The inputs are the scores, the value and the multiplier. -/
  | attentionFromScores {batch : Shape} {rows keys width : Nat}
      (mask : Option (Tensor Bool [rows, keys])) :
      Op [.value (batch ++ [rows, keys]), .value (batch ++ [keys, width]),
        .value (batch ++ [rows, keys])] [.value (batch ++ [rows, width])]
  /-- Stable sort along `dim`: the sorted values and the index each came from. -/
  | sort {s : Shape} (dim : Fin s.rank) (descending : Bool) :
      Op [.value s] [.value s, .index [s.extent dim] s]
  /-- The `k` largest, or smallest, entries along `dim` and their indices. -/
  | topk {s : Shape} (dim : Fin s.rank) (k : Nat) (largest : Bool) (h : k ≤ s.extent dim) :
      Op [.value s]
        [.value (s.slice [dim] [0] [k] [1]), .index [s.extent dim] (s.slice [dim] [0] [k] [1])]
  /-- Read `x` at the index tuples of the second input along the axes `dims`, and along each
  batch pair `(a, b)` of `batchDims` at the index tensor's coordinate along `b`. The result has the
  index tensor's axes followed by the other axes of `x`. The pullback adds each upstream entry into
  its source entry of `x`, in row-major order of the index tensor. -/
  | gather {s t : Shape} (dims : List (Fin s.rank)) (batchDims : List (Fin s.rank × Fin t.rank))
      (h : Shape.GatherValid s t dims batchDims) (e : Op.Element := .scalar) :
      Op [e.port .value s, .index (dims.map s.extent) t]
        [e.port .value (t ++ s.eraseAxes (dims ++ batchDims.map Prod.fst))]
  /-- `Random.uniform` from `key`: each entry is the exact `u / 2^32` of a 32-bit `u` rounded
  once into the scalar, in `[0, 1)` over ℝ (`TorchLean.Random.uniform_mem_Ico`) and in binary64,
  which holds it exactly. At a binary precision `p ≤ 31` rounding to nearest gives `1` for the
  `2^(31 - p)` largest `u`, with probability `2^-25` in binary32, `2^-12` in binary16 and `2^-9`
  in bfloat16. -/
  | randUniform {s : Shape} (key : UInt64) : Op [] [.value s]
  /-- Deterministic approximate standard-normal tensor from `key`, evaluated in the scalar's
  arithmetic (`Random.normal`). -/
  | randn {s : Shape} (key : UInt64) : Op [] [.value s]
  /-- `1` above zero, `-1` below, and the input at a zero or a NaN: a zero keeps its sign and a
  NaN stays a NaN, as `jax.lax.sign` computes it (`torch.sign` gives `0` at `-0` and at a NaN).
  The pullback is zero, as in PyTorch. -/
  | sign {s : Shape} : Op [.value s] [.value s]
  /-- The largest integer not above the input, with a zero pullback. Needs `Rounding`. -/
  | floor {s : Shape} : Op [.value s] [.value s]
  /-- The smallest integer not below the input, with a zero pullback. Needs `Rounding`. -/
  | ceil {s : Shape} : Op [.value s] [.value s]
  /-- The nearest integer (ties to even), with a zero pullback. Needs `Rounding`. -/
  | round {s : Shape} : Op [.value s] [.value s]
  /-- The input rounded toward zero, with a zero pullback. Needs `Rounding`. -/
  | trunc {s : Shape} : Op [.value s] [.value s]
  /-- The input minus its truncation, with pullback `g`. Needs `Rounding`. -/
  | frac {s : Shape} : Op [.value s] [.value s]
  /-- `1 / √x`, rounded twice in a binary format, so binary64's `rsqrt 3` is one ulp above the
  correctly rounded `rSqrt 3`. The pullback is `-(1/2) g y³` at the result `y`. -/
  | rsqrt {s : Shape} : Op [.value s] [.value s]
  /-- `2ˣ`, with pullback `g · 2ˣ · log 2`. -/
  | exp2 {s : Shape} : Op [.value s] [.value s]
  /-- `exp x - 1`, which `Float` and `Float32` round twice at `|x| ≥ 1`, so `Float`'s `expm1 1` is
  one ulp below the correctly rounded value. The pullback is `g exp x`. -/
  | expm1 {s : Shape} : Op [.value s] [.value s]
  /-- The base-2 logarithm, with pullback `g / (x log 2)`. -/
  | log2 {s : Shape} : Op [.value s] [.value s]
  /-- The base-10 logarithm, with pullback `g / (x log 10)`. -/
  | log10 {s : Shape} : Op [.value s] [.value s]
  /-- `log (1 + x)`, with pullback `g / (x + 1)`. -/
  | log1p {s : Shape} : Op [.value s] [.value s]
  /-- The tangent, with pullback `g (1 + y²)` at the result `y`. -/
  | tan {s : Shape} : Op [.value s] [.value s]
  /-- The inverse hyperbolic sine, with pullback `g / √(x² + 1)`. -/
  | asinh {s : Shape} : Op [.value s] [.value s]
  /-- The inverse hyperbolic cosine, with pullback `g / (√(x - 1) √(x + 1))`, which is
  `g / √(x² - 1)` above one. -/
  | acosh {s : Shape} : Op [.value s] [.value s]
  /-- The inverse hyperbolic tangent, with pullback `g / (1 - x²)`. -/
  | atanh {s : Shape} : Op [.value s] [.value s]
  /-- The arctangent, with pullback `g / (x² + 1)`. Needs `InverseTrig`. -/
  | atan {s : Shape} : Op [.value s] [.value s]
  /-- The arcsine, with pullback `g / √(1 - x²)`. Needs `InverseTrig`. -/
  | asin {s : Shape} : Op [.value s] [.value s]
  /-- The arccosine, with pullback `-g / √(1 - x²)`. Needs `InverseTrig`. -/
  | acos {s : Shape} : Op [.value s] [.value s]
  /-- The error function, with pullback `2 / √π · exp (-x²) · g`. Needs `Erf`. -/
  | erf {s : Shape} : Op [.value s] [.value s]
  /-- The angle of `(x, y)` for inputs `y` and `x`, with pullbacks `g x / (y² + x²)` and
  `-g y / (y² + x²)`. At the origin and at an infinite coordinate each pullback is the zero slope
  times the seed. Needs `InverseTrig`. -/
  | atan2 {s : Shape} : Op [.value s, .value s] [.value s]
  /-- C's truncating remainder, with pullbacks `g` and `-g n` for `n = Context.fmodQuotient a b`,
  the truncated exact quotient while `|n| < 2 ^ (p - 2)` at precision `p` (PyTorch truncates the
  rounded one). Needs `Rounding`. -/
  | fmod {s : Shape} : Op [.value s, .value s] [.value s]
  /-- Python's `%`, with pullbacks `g` and `-g q` for `q = Context.remainderQuotient a b`, the
  floored exact quotient where `Context.fmodQuotient` is exact and a NaN where `fmod a b` is one.
  Needs `Rounding`. -/
  | remainder {s : Shape} (e : Op.NumericElement := .scalar) :
      Op [e.port .value s, e.port .value s] [e.port .value s]
  /-- `√(a² + b²)`, with pullbacks `g a / h` and `g b / h` at the result `h`. At the origin both
  pullbacks are `0 * g`. -/
  | hypot {s : Shape} : Op [.value s, .value s] [.value s]
  /-- The magnitude of `a` with the sign of `b`. As in PyTorch, `b` receives a zero gradient. -/
  | copysign {s : Shape} : Op [.value s, .value s] [.value s]
  /-- The NaN-ignoring maximum. The gradient goes to the operand it returns
  (`Context.fmaxTakesFirst`): to `a` where `b` is a NaN, or neither is a NaN and `a` is not below
  `b` with `+0` above `-0`, and to `b` elsewhere. -/
  | fmax {s : Shape} : Op [.value s, .value s] [.value s]
  /-- The NaN-ignoring minimum. The gradient goes to the operand it returns
  (`Context.fminTakesFirst`): to `a` where `b` is a NaN, or neither is a NaN and `a` is not above
  `b` with `-0` below `+0`, and to `b` elsewhere. -/
  | fmin {s : Shape} : Op [.value s, .value s] [.value s]
  /-- NaN, `+∞` and `-∞` replaced by three frozen scalar inputs. As in PyTorch, the pullback is the
  upstream gradient at a finite entry and `0` elsewhere. -/
  | nanToNum {s : Shape} : Op [.value s, .frozen [], .frozen [], .frozen []] [.value s]
  /-- `select cond x y` (PyTorch's `where`) with a Boolean condition input: `x` receives the
  gradient where the condition holds and `y` elsewhere, zeros at the other entries. -/
  | select {s : Shape} (e : Op.Element := .scalar) :
      Op [.bool s, e.port .value s, e.port .value s] [e.port .value s]
  /-- `Context.lerp a b w`: `a + w (b - a)` when `|w| < 1/2`, else `b - (b - a) (1 - w)`, with
  pullbacks `g (1 - w)`, `g w` and `g (b - a)`. -/
  | lerp {s : Shape} : Op [.value s, .value s, .value s] [.value s]
  /-- `x + value · a · b` for a frozen scalar input `value`, with pullbacks `g`, `g b value` and
  `g a value`. -/
  | addcmul {s : Shape} : Op [.value s, .value s, .value s, .frozen []] [.value s]
  /-- `x + value · a / b` for a frozen scalar input `value`, with pullbacks `g`, `g value / b` and
  `-g value a / b²`. -/
  | addcdiv {s : Shape} : Op [.value s, .value s, .value s, .frozen []] [.value s]
  /-- GELU with `torch.nn.functional.gelu`'s `approximate`: the tanh formula by default, the exact
  `x Φ(x)` at `.none`, each with the pullback of its derivative. The exact form needs `Erf`. Our
  default differs from torch's, which is `'none'` (PyTorch 2.8 docs), so the PyTorch capture reads a
  missing `approximate` as `'none'` (`python/torchlean/_pointwise.py`).
  `TorchLeanTest.Interop.IRPyTorch` compares both forms, as `TorchLean.PyTorch.emit` writes them,
  with ours at `±∞`, `±0`, a NaN and four finite points. -/
  | gelu {s : Shape} (approximate : GeluApproximation := .tanh) : Op [.value s] [.value s]
  /-- The hyperbolic sine, whose pullback multiplies by `cosh`. -/
  | sinh {s : Shape} : Op [.value s] [.value s]
  /-- The hyperbolic cosine, whose pullback multiplies by `sinh`. -/
  | cosh {s : Shape} : Op [.value s] [.value s]
  /-- `1 / (x + ε)` with the scalar's default `ε`. The pullback and the tangent multiply the seed
  by the rounded `-(1 / (x + ε))²`, so a finite seed gives a zero where that square underflows. -/
  | safeInv {s : Shape} : Op [.value s] [.value s]
  /-- `Activation.smoothAbs x ε` with a frozen scalar `ε`, which is `√(x² + ε)` over ℝ
  (`TorchLean.Activation.smoothAbs_eq_sqrt`). The pullback multiplies by
  `x / Activation.smoothAbs x ε`. -/
  | smoothAbs {s : Shape} : Op [.value s, .frozen []] [.value s]
  /-- `x / (d + ε)` for a frozen divisor `d` and the scalar's default `ε`, with pullback
  `g / (d + ε)`. -/
  | safeDiv {s : Shape} : Op [.value s, .frozen s] [.value s]
  /-- The power `x ^ e` with a frozen exponent tensor. The pullback to the base is
  `e x ^ (e - 1) g`, the weighted derivative that `Numeric.powerBackward` computes with
  range checks. -/
  | pow {s : Shape} (e : Op.NumericElement := .scalar) :
      Op [e.port .value s, e.port .frozen s] [e.port .value s]
  /-- Cumulative product along `dim`. The pullback at an entry adds, over the prefixes that contain
  it, the upstream gradient at the prefix's last entry times the product of the prefix without the
  entry. -/
  | cumprod {s : Shape} (dim : Fin s.rank) : Op [.value s] [.value s]
  /-- Running maxima along `dim`, each with the index it comes from. The pullback adds the upstream
  gradient at that index. -/
  | cummax {s : Shape} (dim : Fin s.rank) : Op [.value s] [.value s, .index [s.extent dim] s]
  /-- Running minima along `dim`, each with the index it comes from. The pullback adds the upstream
  gradient at that index. -/
  | cummin {s : Shape} (dim : Fin s.rank) : Op [.value s] [.value s, .index [s.extent dim] s]
  /-- The mean squared error of a prediction and a target, both differentiable: the target receives
  the negated prediction gradient. That's `torch.nn.functional.mse_loss` at its default
  `reduction='mean'` (PyTorch 2.8 docs), and torch's `derivatives.yaml` gives the target the same
  negation, `mse_loss_backward` with the arguments swapped. On an empty input the loss is `0`
  (`meanDenominator` divides by one there) and torch's is a NaN. -/
  | mseLoss {s : Shape} : Op [.value s, .value s] [.value .scalar]
  /-- The mean absolute error against a frozen target. Its gradient, `Tensor.l1LossBackward`, is
  `+0` at a zero or a NaN difference, so a NaN target gives `+0` where the loss is NaN. -/
  | maeLoss {s : Shape} : Op [.value s, .frozen s] [.value .scalar]
  /-- The Huber loss against a frozen target, with its threshold a frozen scalar input: the mean of
  the elementwise losses when `mean`, else their sum. -/
  | huberLoss {s : Shape} (mean : Bool) : Op [.value s, .frozen s, .frozen []] [.value .scalar]
  /-- The mean hinge loss against a frozen target of signs. -/
  | hingeLoss {s : Shape} : Op [.value s, .frozen s] [.value .scalar]
  /-- The Poisson negative log-likelihood of log-rates against a frozen target. -/
  | poissonLoss {s : Shape} : Op [.value s, .frozen s] [.value .scalar]
  /-- The mean `log cosh` of the difference from a frozen target. -/
  | logCoshLoss {s : Shape} : Op [.value s, .frozen s] [.value .scalar]
  /-- The mean binary cross-entropy of probabilities against a frozen target, clamped by a frozen
  scalar epsilon. -/
  | binaryCrossEntropy {s : Shape} : Op [.value s, .frozen s, .frozen []] [.value .scalar]
  /-- One minus the cosine similarity over every axis against a frozen target, with a frozen scalar
  epsilon guarding the norms. -/
  | cosineLoss {s : Shape} : Op [.value s, .frozen s, .frozen []] [.value .scalar]
  /-- The cross-entropy of logits against a target along `dim`, averaged over the other axes when
  `mean`. Each fiber of the logits receives `softmax x * sum target - target` (`softmax x - target`
  for a distribution), divided by the fiber count when `mean`, and the target its own gradient.
  The probabilities form is `crossEntropyProbabilities`. -/
  | crossEntropy {s : Shape} (dim : Fin s.rank) (mean : Bool) :
      Op [.value s, .value s] [.value .scalar]
  /-- The cross-entropy of probabilities against a frozen target distribution along `dim`, averaged
  over the other axes (divided by the number of fibers along `dim`), with the probabilities clamped
  by a frozen scalar epsilon. The logits form is `crossEntropy`. -/
  | crossEntropyProbabilities {s : Shape} (dim : Fin s.rank) :
      Op [.value s, .frozen s, .frozen []] [.value .scalar]
  /-- The cross-entropy of a student's logits against a teacher's along `dim`
  (`Tensor.distillLoss`). The teacher is a detached input, a recorded parent read without a
  gradient. -/
  | distill {s : Shape} (dim : Fin s.rank) : Op [.value s, .detached s] [.value .scalar]
  /-- The same entries at another shape of equal size, with a pullback that reshapes the gradient
  back. -/
  | reshape {s t : Shape} (h : s.size = t.size) (e : Op.Element := .scalar) :
      Op [e.port .value s] [e.port .value t]
  /-- The axes reordered into `out`, axis `i` of the result being axis `perm i` of the input. The
  pullback permutes the gradient back. -/
  | permute {s : Shape} (perm : Equiv.Perm (Fin s.rank)) (out : Shape)
      (hout : out = s.permute perm) (e : Op.Element := .scalar) :
      Op [e.port .value s] [e.port .value out]
  /-- The input broadcast to `t`, axis `i` of the input placed at axis `dims i`. The pullback sums
  the gradient over the other axes and over each axis of extent one that is repeated
  (`Tensor.reduceBroadcast`). PyTorch's `broadcast_to` is the right-aligned placement
  `Shape.CanBroadcastTo.dims`. -/
  | broadcast {s t : Shape} (dims : Fin s.rank → Fin t.rank) (h : Shape.BroadcastDims s t dims)
      (e : Op.Element := .scalar) : Op [e.port .value s] [e.port .value t]
  /-- A tensor of the input's shape filled with a frozen scalar input, so the input receives a zero
  gradient. -/
  | fullLike {s : Shape} : Op [.value s, .frozen []] [.value s]
  /-- Inverted dropout with a detached uniform sample `u` on `[0, 1]` (rounding in binary32 can
  produce the endpoint `1`) and a frozen drop probability `p`. An entry is kept iff `u ≥ p`
  (`Op.ge`'s function), with probability `1 - p` for a uniform `u`. Every entry is kept at `p = 0`
  and none at a NaN `p`. A kept entry is `x / (1 - p)` with one division, a dropped one is `x * 0`
  (NaN at `±∞` and NaN, `-0` at a negative `x`), and every entry is `+0` when `p ≥ 1`. The pullback
  and the forward-mode rule are the same map. The layers refuse a NaN `p` and one outside
  `[0, 1]`. -/
  | dropout {s : Shape} : Op [.value s, .detached s, .frozen []] [.value s]
  /-- The rows of a table at an index input, with the row width appended to the index shape. The
  pullback adds each upstream row into its table row, in row-major index order. -/
  | embedding {num dim : Nat} {t : Shape} :
      Op [.value [num, dim], .index [num] t] [.value (t ++ [dim])]
  /-- The matrix product over the last two axes with both batch prefixes broadcast to `batch`. The
  pullbacks are the products with the transposed other operand, summed over broadcast axes. With
  `lv` the left operand is a vector, read as a one-row matrix, and with `rv` the right operand is a
  vector, read as a one-column matrix. The result leaves out the axis of each (`Shape.matmulLeft`,
  `Shape.matmulRight`, `Shape.matmulOut`), and a vector's gradient is that of its matrix view. -/
  | matmul {batchA batchB batch : Shape} {m n p : Nat} (lv rv : Bool) (hl : lv = true → m = 1)
      (hr : rv = true → p = 1) (broadcastA : Shape.CanBroadcastTo batchA batch)
      (broadcastB : Shape.CanBroadcastTo batchB batch) :
      Op [.value (Shape.matmulLeft lv batchA m n), .value (Shape.matmulRight rv batchB n p)]
        [.value (Shape.matmulOut lv rv batch m p)]
  /-- The matrix product with a frozen scalar scale kept inside each contraction, broadcast and with
  vector operands as in `matmul`. The pullbacks keep the same scale in each product. -/
  | scaledMatmul {batchA batchB batch : Shape} {m n p : Nat} (lv rv : Bool)
      (hl : lv = true → m = 1) (hr : rv = true → p = 1)
      (broadcastA : Shape.CanBroadcastTo batchA batch)
      (broadcastB : Shape.CanBroadcastTo batchB batch) :
      Op [.value (Shape.matmulLeft lv batchA m n), .value (Shape.matmulRight rv batchB n p),
          .frozen []] [.value (Shape.matmulOut lv rv batch m p)]
  /-- Convolution with a kernel input, a bias of one entry per output feature and the input, with
  the dimension numbers `dims`, per-axis stride, padding before and after, kernel dilation and
  feature groups, which `Op.check` requires to be positive and to divide the kernel's
  output-feature extent `k.extent (dims.kernel 0)`, and the input's feature extent
  `s.extent (dims.input 1)` to be `groups * k.extent (dims.kernel 1)`. The batch axis `dims.input 0`
  of `s` stands for the axes of `lead`, folded row-major. The input `x` is `s` with that axis
  replaced in place by `lead`, and the result `y` the output with its batch axis replaced likewise
  (`Shape.expandAxis`). With `lead` the batch extent alone, `x` is `s`. The rule is the one-batch
  rule on the folded tensors. Its bias gradient sums the folded output gradient over every axis but
  the feature axis, in that rule's order, cast to the bias shape by
  `Shape.reduce_conv_otherAxes`. -/
  | conv {s k : Shape} {r : Nat} (dims : Shape.ConvDims s k r)
      (stride padLow padHigh dilation : Fin r → Nat) (groups : Nat) {lead x y : Shape}
      (hlead : s.extent (dims.input ⟨0, by omega⟩) = lead.size)
      (hx : x = s.expandAxis (dims.input ⟨0, by omega⟩) lead)
      (hy : y = (s.conv k dims stride padLow padHigh (fun _ => 1) dilation 1).expandAxis
        (dims.output stride padLow padHigh (fun _ => 1) dilation 1 ⟨0, by omega⟩) lead) :
      Op [.value k, .value [k.extent (dims.kernel ⟨0, by omega⟩)], .value x] [.value y]
  /-- Pooling over the axes `dims` with the reduction `op` (sum, maximum, minimum or average), with
  per-axis kernel, stride, padding before and after, dilation and ceil mode. `Op.check` rejects an
  ill-formed window before any input is read. -/
  | pool {s : Shape} (op : PoolOp) (dims : List (Fin s.rank))
      (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool) :
      Op [.value s] [.value (s.pool dims kernel stride padLow padHigh dilation ceilMode)]
  /-- The log-sum-exp surrogate of max pooling with a frozen finite nonzero `beta`, over the axes
  `dims` with per-axis window parameters. `validate` rejects a zero or nonfinite `beta`. A recorder
  reports an empty axis list or an ill-formed window first, then a failed read of the input, then
  the value of `beta`. -/
  | smoothMaxPool {s : Shape} (dims : List (Fin s.rank))
      (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool) :
      Op [.value s,
          .frozen []] [.value (s.pool dims kernel stride padLow padHigh dilation ceilMode)]
  /-- Layer normalization over the last axis of `x`, the axes `lead` followed by an axis of
  `embedDim` entries, with scale and bias inputs and a frozen scalar epsilon. The axes of `lead`
  fold row-major into the `seqLen = lead.size` rows of a `(seqLen, embedDim)` matrix, and the rule
  is the matrix rule on that view of the input and of the output gradient (`rule?_layerNorm`). With
  `lead` the rows alone, `x` is that matrix and the rule is the matrix rule itself
  (`rule?_layerNorm_matrix`). The matrix has positive extents. -/
  | layerNorm {seqLen embedDim : Nat} (hSeq : seqLen > 0) (hEmbed : embedDim > 0) {lead x : Shape}
      (hlead : seqLen = lead.size) (hx : x = lead ++ [embedDim]) :
      Op [.value x, .value [embedDim], .value [embedDim], .frozen []] [.value x]
  /-- Batch normalization of a channel-first tensor over every spatial axis with batch statistics,
  scale and bias inputs and a frozen scalar epsilon. -/
  | batchNorm {channels : Nat} {spatial : Shape}
      (hWellFormed : (Shape.dim channels spatial).Positive) :
      Op [.value (.dim channels spatial), .value [channels], .value [channels],
          .frozen []] [.value (.dim channels spatial)]
  /-- Batch normalization along `dim` with scale and bias inputs and frozen running mean, running
  variance and scalar epsilon, `Tensor.batchNorm x dim gamma beta (.running mean var) epsilon`. The
  statistics and epsilon receive no gradient (`BatchNorm.runningBackward`) and no tangent. -/
  | batchNormRunning {s : Shape} (dim : Fin s.rank) :
      Op [.value s, .value [s.extent dim], .value [s.extent dim], .frozen [s.extent dim],
          .frozen [s.extent dim], .frozen []] [.value s]
  /-- Multi-head self-attention of every `(n, dModel)` sequence along the leading axes `batch`, with
  query, key, value and output projection inputs shared by the sequences and an optional mask,
  recorded as one fused node. Each sequence's value and input gradient are those of the sequence
  alone, and the projection gradients of the sequences are added in row-major order of `batch`
  (`rule?_multiheadAttention_single`, `multiheadAttention_forward_dim`,
  `multiheadAttention_backward_dim`). It emits the input's gradient before the four projections', as
  the tape node does (`order`). -/
  | multiheadAttention {batch : Shape} {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
      (mask : Option (Tensor Bool [n, n])) :
      Op [.value [dModel, numHeads * headDim], .value [dModel, numHeads * headDim],
        .value [dModel, numHeads * headDim], .value [numHeads * headDim, dModel],
        .value (batch ++ [n, dModel])] [.value (batch ++ [n, dModel])]
  /-- Transposed convolution with a kernel input in PyTorch's `(inC, outC / groups, kernel...)`
  layout, a bias of one entry per output feature and the input, with per-axis stride, padding,
  output padding, kernel dilation and groups, which `Op.check` requires to be positive and to divide
  the input's feature extent `inC = s.extent (dims.input 1)`, with `k.extent (dims.kernel 1) = inC`.
  The bias gradient is cast to the bias shape by
  `Shape.reduce_convTranspose_otherAxes`. -/
  | convTranspose {s k : Shape} {r : Nat} (dims : Shape.ConvDims s k r)
      (stride padLow padHigh outputPadding dilation : Fin r → Nat) (groups : Nat) :
      Op [.value k, .value [groups * k.extent (dims.kernel ⟨0, by omega⟩)],
          .value s] [.value
          (s.convTranspose k dims stride padLow padHigh outputPadding dilation groups)]
  /-- Whether corresponding entries are equal, as the scalar's `==` decides. The inputs are frozen,
  since a comparison has no gradient. -/
  | eq {s : Shape} (e : Op.NumericElement := .scalar) :
      Op [e.port .frozen s, e.port .frozen s] [.bool s]
  /-- Whether corresponding entries differ, with frozen inputs. -/
  | ne {s : Shape} (e : Op.NumericElement := .scalar) :
      Op [e.port .frozen s, e.port .frozen s] [.bool s]
  /-- Whether each entry of the first input is below the second's, with frozen inputs. -/
  | lt {s : Shape} (e : Op.NumericElement := .scalar) :
      Op [e.port .frozen s, e.port .frozen s] [.bool s]
  /-- Whether each entry of the first input is at most the second's, with frozen inputs. -/
  | le {s : Shape} (e : Op.NumericElement := .scalar) :
      Op [e.port .frozen s, e.port .frozen s] [.bool s]
  /-- Whether each entry of the first input is above the second's, with frozen inputs. -/
  | gt {s : Shape} (e : Op.NumericElement := .scalar) :
      Op [e.port .frozen s, e.port .frozen s] [.bool s]
  /-- Whether each entry of the first input is at least the second's, with frozen inputs. -/
  | ge {s : Shape} (e : Op.NumericElement := .scalar) :
      Op [e.port .frozen s, e.port .frozen s] [.bool s]
  /-- Whether each entry of the frozen input is a NaN. -/
  | isNaN {s : Shape} : Op [.frozen s] [.bool s]
  /-- Whether each entry of the frozen input is an infinity. -/
  | isInf {s : Shape} : Op [.frozen s] [.bool s]
  /-- Whether each entry of the frozen input is neither a NaN nor an infinity. -/
  | isFinite {s : Shape} : Op [.frozen s] [.bool s]
  /-- The negation of a Boolean tensor. -/
  | logicalNot {s : Shape} : Op [.bool s] [.bool s]
  /-- The conjunction of two Boolean tensors. -/
  | logicalAnd {s : Shape} : Op [.bool s, .bool s] [.bool s]
  /-- The disjunction of two Boolean tensors. -/
  | logicalOr {s : Shape} : Op [.bool s, .bool s] [.bool s]
  /-- The exclusive disjunction of two Boolean tensors. -/
  | logicalXor {s : Shape} : Op [.bool s, .bool s] [.bool s]
  /-- Bitwise and of integer words (`torch.bitwise_and`, `lax.bitwise_and`). -/
  | bitwiseAnd {signed : Bool} {width : Nat} {s : Shape} :
      Op [.integer signed width s, .integer signed width s] [.integer signed width s]
  /-- Bitwise or of integer words (`torch.bitwise_or`, `lax.bitwise_or`). -/
  | bitwiseOr {signed : Bool} {width : Nat} {s : Shape} :
      Op [.integer signed width s, .integer signed width s] [.integer signed width s]
  /-- Bitwise exclusive or of integer words (`torch.bitwise_xor`, `lax.bitwise_xor`). -/
  | bitwiseXor {signed : Bool} {width : Nat} {s : Shape} :
      Op [.integer signed width s, .integer signed width s] [.integer signed width s]
  /-- Bitwise complement of integer words (`torch.bitwise_not`, `lax.bitwise_not`). -/
  | bitwiseNot {signed : Bool} {width : Nat} {s : Shape} :
      Op [.integer signed width s] [.integer signed width s]
  /-- The first input shifted left by the second (`Word.shiftLeft`,
  `torch.bitwise_left_shift`, `lax.shift_left`). -/
  | bitwiseLeftShift {signed : Bool} {width : Nat} {s : Shape} :
      Op [.integer signed width s, .integer signed width s] [.integer signed width s]
  /-- The first input shifted right by the second, filled with its top bit
  (`Word.shiftRightArithmetic`, `lax.shift_right_arithmetic`). -/
  | shiftRightArithmetic {signed : Bool} {width : Nat} {s : Shape} :
      Op [.integer signed width s, .integer signed width s] [.integer signed width s]
  /-- The first input shifted right by the second, filled with zeros
  (`Word.shiftRightLogical`, `lax.shift_right_logical`). -/
  | shiftRightLogical {signed : Bool} {width : Nat} {s : Shape} :
      Op [.integer signed width s, .integer signed width s] [.integer signed width s]
  /-- The number of one bits of each word, encoded as a word of the same type (`BitVec.cpop`,
  `lax.population_count`). At a signed width of one, a count of one reads as `-1`. -/
  | populationCount {signed : Bool} {width : Nat} {s : Shape} :
      Op [.integer signed width s] [.integer signed width s]
  /-- The number of zero bits above the highest one bit of each word, the width at zero, encoded
  as a word of the same type (`BitVec.clz`, `lax.clz`). At a signed width of two, `clz 0` reads
  as `-2`. -/
  | countLeadingZeros {signed : Bool} {width : Nat} {s : Shape} :
      Op [.integer signed width s] [.integer signed width s]
  /-- The upper half of the exact product of two words, of their signed values when `signed`
  (`Word.mulhi`, `lax.mulhi`). -/
  | mulhi {signed : Bool} {width : Nat} {s : Shape} :
      Op [.integer signed width s, .integer signed width s] [.integer signed width s]
  /-- Each entry's word of `width` bits read at another sort (`lax.bitcast_convert_type` and
  `Tensor.view(dtype)` at one width), every bit kept. A scalar side reads the scalar's `Bits` and
  has a rule only where its words have `width` bits. The input receives no gradient. -/
  | bitcast {s : Shape} (width : Nat) (source target : Op.Word) :
      Op [source.input width s] [target.output width s]
  /-- The cube root (`lax.cbrt`), the scalar's `Cbrt`. The pullback computes
  `g * (1/3 * (1 / (y * y)))` at the output `y`, JAX's order, with `1/3` rounded once. Over the
  reals that is the derivative at `x ≠ 0`, though the `Cbrt` class carries no proof of it. -/
  | cbrt {s : Shape} : Op [.value s] [.value s]
  /-- C's `nextafter` (`torch.nextafter`, `lax.nextafter`), the scalar's `NextAfter`. The inputs are
  frozen, since TorchLean gives it no derivative (PyTorch implements none). -/
  | nextafter {s : Shape} : Op [.frozen s, .frozen s] [.value s]
  /-- The index of the first largest entry along `dim`, or of the first NaN where there is one
  (`Tensor.argmax`), the axis kept with extent one when `keepdim`. The input is frozen, since an
  index has no gradient. -/
  | argmax {s : Shape} (dim : Fin s.rank) (keepdim : Bool) (h : 0 < s.extent dim) :
      Op [.frozen s] [.index [s.extent dim] (s.reduce [dim] keepdim)]
  /-- The index of the first smallest entry along `dim`, or of the first NaN where there is one
  (`Tensor.argmin`), the axis kept with extent one when `keepdim`. The input is frozen. -/
  | argmin {s : Shape} (dim : Fin s.rank) (keepdim : Bool) (h : 0 < s.extent dim) :
      Op [.frozen s] [.index [s.extent dim] (s.reduce [dim] keepdim)]
  /-- The entries reversed along each axis of `dims`. The pullback flips the upstream gradient
  along the same axes, and the forward-mode rule flips the input tangent. -/
  | flip {s : Shape} (dims : List (Fin s.rank)) (e : Op.Element := .scalar) :
      Op [e.port .value s] [e.port .value s]
  /-- The index of each coordinate along `dim`, a tensor of shape `s` with no inputs
  (JAX's `iota`). -/
  | indices (s : Shape) (dim : Fin s.rank) : Op [] [.index [s.extent dim] s]
  /-- The entries of a Boolean or index input as scalars (PyTorch's `Tensor.to`): `1` for `true`,
  `0` for `false` and an index as its natural number. The input has no gradient. -/
  | cast {s : Shape} (source : Op.CastSource) : Op [source.port s] [.value s]
  /-- The index tuples of `s` as words of `width` bits of an integer type, signed when `signed`, in
  the layout of `Port.irShape`, the IR's form of an index result. Component `k` of the tuple at flat
  position `j` is the word at `j * bounds.length + k`. Its value check fails at a component the type
  cannot hold, though a bound the type cannot hold is no failure by itself. -/
  | encodeIndex {s : Shape} (bounds : List Nat) (signed : Bool) (width : Nat) :
      Op [.index bounds s] [.integer signed width (Port.irShape ⟨.index bounds, s⟩)]
  /-- The index tuples of `s` held by words of `width` bits in the layout of `Port.irShape`, each
  word read as its value at the integer type, signed when `signed`: the IR's form of an index
  operand. Its value check fails at a component outside `[0, bound)`. It has a rule where every
  bound is positive or `s` has no entry, the condition for its output to have a value. -/
  | decodeIndex {s : Shape} (bounds : List Nat) (signed : Bool) (width : Nat) :
      Op [.integer signed width (Port.irShape ⟨.index bounds, s⟩)] [.index bounds s]
  /-- Scaled dot-product attention `softmax (scale · q kᵀ) v` over any leading batch axes, with an
  optional mask of allowed score entries, a schedule and a frozen scalar scale. The pullbacks to
  the query, key and value are those of `Tensor.attentionBackward`. The weighted form with a
  multiplier is `attentionFromScores` after the score product. -/
  | attention {batch : Shape} {rows keys depth width : Nat} (mask : Option (Tensor Bool [rows,
      keys]))
      (schedule : Softmax.Schedule) :
      Op [.value (batch ++ [rows, depth]), .value (batch ++ [keys, depth]),
        .value (batch ++ [keys, width]), .frozen []] [.value (batch ++ [rows, width])]
  /-- The reduction `op` over the axes `dims` (`Tensor.reduce`: `torch.sum`, `torch.prod`,
  `torch.amax`, `torch.amin`), the axes kept with extent one when `keepdim`. A maximum or minimum
  needs every listed axis nonempty, and `Op.check` rejects a repeated axis. The pullback of a
  maximum or minimum splits the upstream gradient evenly among the entries equal to the extremum
  (by the scalar's `==`, a NaN equal to a NaN extremum), and its tangent is the mean of their
  tangents (`Tensor.reduceJVP`). -/
  | reduce {s : Shape} (op : Tensor.ReduceOp) (dims : List (Fin s.rank)) (keepdim : Bool)
      (h : op = .max ∨ op = .min → ∀ d ∈ dims, 0 < s.extent d) :
      Op [.value s] [.value (s.reduce dims keepdim)]
  /-- The mean over the axes `dims` (`torch.mean`): each fiber's sum divided by its number of
  entries, and by one for an empty fiber (TorchLean's totalized mean). The pullback divides each
  entry's upstream gradient once by the same count, and `Op.check` rejects a repeated axis. -/
  | mean {s : Shape} (dims : List (Fin s.rank)) (keepdim : Bool) :
      Op [.value s] [.value (s.reduce dims keepdim)]
  /-- The contraction of the paired axes `dims` with the batch pairs `batchDims`
  (`Tensor.contract`): each output entry adds the products of the paired entries to zero in
  row-major order of the contracted axes. Each pullback contracts the upstream gradient with the
  other input over that input's other axes (`Tensor.contractBackwardLeft`,
  `Tensor.contractBackwardRight`). -/
  | contract {s t : Shape} (dims batchDims : List (Fin s.rank × Fin t.rank))
      (h : Shape.ContractValid s t dims batchDims) :
      Op [.value s, .value t] [.value (s.contract t dims batchDims)]
  /-- Softmax along `dim` of the entries a Boolean mask input allows (`Tensor.softmax` with the
  mask): a blocked entry has a zero numerator, so weight zero where the fiber's allowed scores are
  finite, and a fiber without an allowed entry is zero. The pullback is the hard-masked vector
  pullback of each fiber, zero on a fiber without an allowed entry, and the mask receives no
  gradient. -/
  | maskedSoftmax {s : Shape} (dim : Fin s.rank) : Op [.value s, .bool s] [.value s]
  /-- Along `dims[k]` the indices `starts[k], starts[k] + steps[k], ...` below `stops[k]`
  (`x[a:b:c]`), the other axes whole. The pullback writes the upstream gradient back where the
  slice read it and zero elsewhere. -/
  | slice {s : Shape} (dims : List (Fin s.rank)) (starts stops steps : List Nat)
      (h : Shape.SliceValid s dims starts stops steps) (e : Op.Element := .scalar) :
      Op [e.port .value s] [e.port .value (s.slice dims starts stops steps)]
  /-- The input padded with a scalar input `value` (`Tensor.pad`): along axis `d`, `low d` entries
  before the input, `high d` after it and `interior d` between consecutive entries. The input's
  pullback reads the upstream gradient at the entries that hold the input (`Tensor.padBackward`).
  The pullback to `value` is `Tensor.sum` of the upstream gradient with those entries replaced by
  zero (`Tensor.select`), added from zero in row-major order. -/
  | pad {s : Shape} (low high interior : Fin s.rank → Nat) (e : Op.Element := .scalar) :
      Op [e.port .value s, e.port .value []] [e.port .value (s.pad low high interior)]
  /-- The entries at `index` along `dim`, the axis removed (`torch.select(x, dim, index)`). The
  pullback copies the upstream gradient to `index` along `dim` and writes zero at the other
  indices. -/
  | indexAt {s : Shape} (dim : Fin s.rank) (index : Fin (s.extent dim)) :
      Op [.value s] [.value (s.eraseIdx dim)]
  /-- The inputs concatenated along `dim` in order (`torch.cat`), input `i` of extent `sizes[i]`
  along `dim` and of the extents of `s` elsewhere. Each input's gradient is its block of the
  upstream gradient. -/
  | cat {s : Shape} (dim : Fin s.rank) (sizes : List Nat) (e : Op.Element := .scalar) :
      Op (sizes.map fun n => e.port .value (s.set dim n))
        [e.port .value (s.set dim sizes.sum)]
  /-- The updates written into `base` at the index tuples of the second input along `dims`, the
  other axes whole, by `rule` in row-major order of the index tensor (`Tensor.scatter`). A sum adds,
  a replacement keeps the last, a product multiplies, and `max` and `min` take the scalar's maximum
  and minimum, IEEE's in a binary format. In the pullback of a sum or a replacement, `base` receives
  the upstream gradient, zero where a replacement overwrote it, and an update reads the upstream
  gradient where it lands, for a replacement only the last update to land there
  (`Tensor.scatterUpdatesBackward`). For a product, each participant receives the upstream gradient
  times the product of the others, with no division (`Tensor.scatterProdBackward`). For a maximum or
  minimum, the participants that tie the result (equal, or both NaN) share the upstream gradient
  evenly (`Tensor.scatterExtremumBackward`), a convention, since the extremum has no derivative at a
  tie. `TorchLeanTest.OpTable` checks the product, maximum and minimum pullbacks against torch
  autograd's gradients of `scatter_reduce`, where a three-way tie at a maximum with upstream `1`
  sends `1/3` to each participant. Torch finds ties with `==`, so at a NaN result it sends a NaN to
  every participant, where we split `g` among the NaN ones. -/
  | scatter {s t : Shape} (rule : Tensor.ScatterReduce) (dims : List (Fin s.rank))
      (h : dims.Nodup) (e : Op.Element := .scalar) :
      Op [e.port .value s, .index (dims.map s.extent) t, e.port .value (t ++ s.eraseAxes dims)]
        [e.port .value s]
  /-- One for the entries of each fiber along `dim` that the pairwise comparison of
  `MoE.topKMask` ranks among the first `k`, zero for the others: on a fiber without NaN the `k`
  largest, ties to the lower index. The input is detached, a recorded parent that receives no
  contribution, and the mask has no gradient. -/
  | topKMask {s : Shape} (dim : Fin s.rank) (k : Nat) : Op [.detached s] [.value s]
  /-- A keep mask of `s` drawn from `key`: one with the frozen scalar keep probability and zero
  otherwise (`Random.mask`). The mask has no gradient. -/
  | bernoulliMask {s : Shape} (key : UInt64) : Op [.frozen []] [.value s]
  /-- The real Fourier transform along `dim` of length `n` (`Tensor.rfft`, unnormalized): the
  `n / 2 + 1` stored frequencies along `dim` and their (real, imaginary) parts on an appended last
  axis. The pullback applies the transposed matrix (`Tensor.rfftBackward`). -/
  | rfft {s : Shape} (dim : Fin s.rank) (n : Nat) (hn : 0 < n) :
      Op [.value (s.set dim n)] [.value (s.set dim (n / 2 + 1) ++ [2])]
  /-- The normalized inverse real Fourier transform along `dim` with output length `n`
  (`Tensor.irfft`). The pullback applies the transposed matrix (`Tensor.irfftBackward`). -/
  | irfft {s : Shape} (dim : Fin s.rank) (n : Nat) (hn : 0 < n) :
      Op [.value (s.set dim (n / 2 + 1) ++ [2])] [.value (s.set dim n)]
  /-- The diagonal selective scan along `dim` (`Tensor.selectiveScan`): every state
  `h t = a t * h (t - 1) + b t * x t` from the initial state, with the coefficients `a` and `b`
  each shared over the steps or one per step. The inputs are `a`, `b`, `x` and the initial state.
  The pullback is the adjoint recurrence (`Tensor.selectiveScanBackward`) and the tangent the
  tangent recurrence (`Tensor.selectiveScanJVP`). -/
  | selectiveScan {s : Shape} (dim : Fin s.rank) (sharedA sharedB : Bool) :
      Op [.value (Tensor.scanCoefficients s dim sharedA),
        .value (Tensor.scanCoefficients s dim sharedB), .value s, .value (s.eraseIdx dim)]
        [.value s]
  /-- A checked scalar program named `name`, one evaluation per output entry. -/
  | custom (name : String) (inputs : List Shape) (output : Shape) :
      Op (inputs.map .value) [.value output]

namespace Op

/-- The operator `o`, written at the scalar, at the element `e`: `o` itself at the scalar
(`Op.atElement_scalar`), and `o` unchanged if it has no value, frozen or detached port (the logical,
word and index operators). Otherwise it is `o` at `e` if `o` takes an element (`Op.Element`, or
`Op.NumericElement` for the arithmetic operators and comparisons, which have no Boolean form), and
`none` for every other operator. -/
def atElement (e : Op.Element) (o : Σ ins outs, Op ins outs) :
    Option (Σ ins outs, Op ins outs) :=
  if e = .scalar then some o else
  match o with
  | ⟨_, _, .neg (s := s) .scalar⟩ => e.numeric?.map fun e => ⟨_, _, .neg (s := s) e⟩
  | ⟨_, _, .add (s := s) .scalar⟩ => e.numeric?.map fun e => ⟨_, _, .add (s := s) e⟩
  | ⟨_, _, .mul (s := s) .scalar⟩ => e.numeric?.map fun e => ⟨_, _, .mul (s := s) e⟩
  | ⟨_, _, .max (s := s) .scalar⟩ => e.numeric?.map fun e => ⟨_, _, .max (s := s) e⟩
  | ⟨_, _, .sub (s := s) .scalar⟩ => e.numeric?.map fun e => ⟨_, _, .sub (s := s) e⟩
  | ⟨_, _, .div (s := s) r .scalar⟩ => e.numeric?.map fun e => ⟨_, _, .div (s := s) r e⟩
  | ⟨_, _, .min (s := s) .scalar⟩ => e.numeric?.map fun e => ⟨_, _, .min (s := s) e⟩
  | ⟨_, _, .remainder (s := s) .scalar⟩ =>
      e.numeric?.map fun e => ⟨_, _, .remainder (s := s) e⟩
  | ⟨_, _, .pow (s := s) .scalar⟩ => e.numeric?.map fun e => ⟨_, _, .pow (s := s) e⟩
  | ⟨_, _, .eq (s := s) .scalar⟩ => e.numeric?.map fun e => ⟨_, _, .eq (s := s) e⟩
  | ⟨_, _, .ne (s := s) .scalar⟩ => e.numeric?.map fun e => ⟨_, _, .ne (s := s) e⟩
  | ⟨_, _, .lt (s := s) .scalar⟩ => e.numeric?.map fun e => ⟨_, _, .lt (s := s) e⟩
  | ⟨_, _, .le (s := s) .scalar⟩ => e.numeric?.map fun e => ⟨_, _, .le (s := s) e⟩
  | ⟨_, _, .gt (s := s) .scalar⟩ => e.numeric?.map fun e => ⟨_, _, .gt (s := s) e⟩
  | ⟨_, _, .ge (s := s) .scalar⟩ => e.numeric?.map fun e => ⟨_, _, .ge (s := s) e⟩
  | ⟨_, _, .gather dims batchDims h .scalar⟩ => some ⟨_, _, .gather dims batchDims h e⟩
  | ⟨_, _, .select (s := s) .scalar⟩ => some ⟨_, _, .select (s := s) e⟩
  | ⟨_, _, .reshape h .scalar⟩ => some ⟨_, _, .reshape h e⟩
  | ⟨_, _, .permute perm out h .scalar⟩ => some ⟨_, _, .permute perm out h e⟩
  | ⟨_, _, .broadcast dims h .scalar⟩ => some ⟨_, _, .broadcast dims h e⟩
  | ⟨_, _, .flip dims .scalar⟩ => some ⟨_, _, .flip dims e⟩
  | ⟨_, _, .slice dims starts stops steps h .scalar⟩ =>
      some ⟨_, _, .slice dims starts stops steps h e⟩
  | ⟨_, _, .pad low high interior .scalar⟩ => some ⟨_, _, .pad low high interior e⟩
  | ⟨_, _, .cat dim sizes .scalar⟩ => some ⟨_, _, .cat dim sizes e⟩
  | ⟨_, _, .scatter (t := t) rule dims h .scalar⟩ => some ⟨_, _, .scatter (t := t) rule dims h e⟩
  | o => if (o.1 ++ o.2.1).any (·.kind.isScalar) then none else some o

/-- At the scalar element, `atElement` returns every operator unchanged. -/
theorem atElement_scalar (o : Σ ins outs, Op ins outs) : atElement .scalar o = some o :=
  ite_eq_left rfl

end Op

/-- The window of a pooling over `dims`: one kernel extent, stride, padding pair and dilation per
pooled axis, the axes distinct, every kernel extent and stride positive. -/
def Op.checkWindow {s : Shape} (label : String) (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) : Except String Unit := do
  unless dims.Nodup do throw s!"autograd: {label} requires distinct pooled axes"
  unless kernel.length = dims.length && stride.length = dims.length &&
      padLow.length = dims.length && padHigh.length = dims.length &&
      dilation.length = dims.length do
    throw s!"autograd: {label} requires one window parameter per pooled axis"
  if kernel.any (· = 0) then
    throw s!"autograd: {label} requires kernel_size > 0 on every pooled axis"
  if stride.any (· = 0) then throw s!"autograd: {label} requires stride > 0 on every pooled axis"

/-- The check of an operator's attributes that their types do not make. It checks the window of a
pooling (and at least one pooled axis for the smooth max pool) and, for a convolution, positive
groups dividing the kernel's output-feature extent and an input feature extent of `groups` times the
kernel's input-feature extent. For a transposed convolution it checks positive groups dividing the
input's feature extent and a kernel with that input-feature extent. It also checks distinct axes for
a reduction or a mean, a positive word width for `encodeIndex` and `decodeIndex`, a rounding for an
integer division and the replacement rule for a Boolean scatter. A recorder runs it before reading
any input, and decoding a wire record does not. -/
def Op.check : {ins outs : List Port} → Op ins outs → Except String Unit
  | _, _, .pool op dims kernel stride padLow padHigh dilation _ =>
      Op.checkWindow (match op with
        | .sum => "sum_pool"
        | .max => "max_pool"
        | .min => "min_pool"
        | .avg _ => "avg_pool") dims kernel stride padLow padHigh dilation
  | _, _, .smoothMaxPool dims kernel stride padLow padHigh dilation _ => do
      if dims.isEmpty then throw "autograd: smooth_max_pool requires at least one pooled axis"
      Op.checkWindow "smooth_max_pool" dims kernel stride padLow padHigh dilation
  | _, _, .conv (s := s) (k := k) dims _ _ _ _ groups _ _ _ => do
      unless 0 < groups do throw "conv: groups must be positive"
      unless groups ∣ k.extent (dims.kernel ⟨0, by omega⟩) do
        throw "conv: groups must divide the kernel's output features"
      unless groups * k.extent (dims.kernel ⟨1, by omega⟩) =
          s.extent (dims.input ⟨1, by omega⟩) do
        throw "conv: the input's features must be groups times the kernel's input features"
  | _, _, .convTranspose (s := s) (k := k) dims _ _ _ _ _ groups => do
      unless 0 < groups do throw "conv_transpose: groups must be positive"
      unless groups ∣ s.extent (dims.input ⟨1, by omega⟩) do
        throw "conv_transpose: groups must divide the input's features"
      unless k.extent (dims.kernel ⟨1, by omega⟩) = s.extent (dims.input ⟨1, by omega⟩) do
        throw "conv_transpose: the kernel's input features must equal the input's features"
  | _, _, .reduce _ dims _ _ => unless dims.Nodup do throw "reduce requires distinct axes"
  | _, _, .mean dims _ => unless dims.Nodup do throw "mean requires distinct axes"
  | _, _, .encodeIndex _ _ width | _, _, .decodeIndex _ _ width =>
      unless 0 < width do throw "an index adapter requires a positive width"
  | _, _, .div .none (.integer _ _) =>
      throw "div: true division of integers has no integer result; insert casts"
  | _, _, .scatter rule _ _ .bool =>
      unless rule = .replace do throw "scatter: Boolean tensors scatter only with replace"
  -- The other elements of `div` and `scatter` are named, so that the match has equation lemmas.
  | _, _, .div _ .scalar | _, _, .div _ (.integer _ _) | _, _, .scatter _ _ _ .scalar
  | _, _, .scatter _ _ _ (.integer _ _) => .ok ()
  | _, _, _ => .ok ()

/-- The sum over every axis passes the attribute check of `Op.reduce`. -/
theorem Op.reduce_check (shape : Shape) :
    (Op.reduce (s := shape) .sum (List.finRange shape.rank) false
      (fun h => absurd h (by decide))).check = .ok () :=
  ite_eq_left (List.nodup_finRange _)

end TorchLean

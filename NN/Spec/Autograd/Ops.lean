/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Autograd.AutogradSpec
public import NN.Spec.Layers.Dropout
public import NN.Spec.Layers.Embedding
public import NN.Spec.Layers.Activation
public import NN.Spec.Layers.Linear
public import NN.Spec.Layers.Loss
public import NN.Spec.Layers.Normalization

/-!
# Autograd OpSpecs (spec layer)

This file defines small `OpSpec` building blocks (forward + VJP) for common tensor operations.
The definitions are intentionally direct mathematical contracts and live purely in the spec layer.

How to read this file:

- Each operation below is an `OpSpec`: a pure `forward` plus a pure VJP `backward`.
- Most ops here are thin wrappers around `*Spec` and derivative-spec definitions from `NN/Spec/*`.

Where this sits in TorchLean:

- `NN.Spec.*` files define pure denotational semantics: what tensors/layers mean.
- This file packages some of those pure definitions as unary `OpSpec`s: `forward` plus VJP.
- `NN.Runtime.Autograd.*` executes programs, tracks parameters, manages tapes/sessions, dispatches
  CUDA kernels, handles RNG, and compiles graphs.

This file does not mirror every runtime method one-for-one. It is the reusable adapter layer for
operations whose input-gradient VJP is naturally expressed as a single
`OpSpec`. Larger multi-input/parameterized layers (convolution, attention, batchnorm, pooling, RNG)
still have precise specs and runtime implementations, but their full backward state usually belongs
in layer/runtime code rather than in this compact unary interface.

PyTorch analogy (roughly):

- A spec `OpSpec` is like a compact `torch.autograd.Function` where we write down the VJP directly.
- We do not model PyTorch's mutable `ctx`; the spec layer receives the input tensor `x` directly.
-/

@[expose] public section


namespace Spec

open Tensor
open Activation

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-! ## Elementwise lifting helpers -/

/-- Lift a scalar function to a tensor by pointwise map.

PyTorch analogy: most `torch.*` pointwise ops are vectorized elementwise maps. -/
def liftElementwise {s : Shape}
  (f : α → α) : Tensor α s → Tensor α s :=
  mapSpec f

/-- Lift an elementwise backward using the chain rule: `dL/dx = df(x) * dL/dy` pointwise.

This is the standard VJP pattern for elementwise ops.

PyTorch analogy: the "local backward" rule for a pointwise op multiplies by the derivative mask. -/
def liftElementwiseBackward [Mul α] {s : Shape}
  (df : α → α) : Tensor α s → Tensor α s → Tensor α s :=
  fun x dLdy =>
    let gx := mapSpec df x
    mulSpec gx dLdy

/-- Elementwise ReLU OpSpec on any shape.

PyTorch analogy: `torch.relu(x)` / `torch.nn.functional.relu(x)`. -/
def reluOp [Mul α] [One α] [Zero α] [Max α] [LT α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {s : Shape} : OpSpec α s s :=
{ forward      := liftElementwise (α:=α) (s:=s) Activation.Math.reluSpec
, backward     := liftElementwiseBackward (α:=α) (s:=s) Activation.Math.reluDerivSpec }

/-- Elementwise sigmoid OpSpec on any shape.

PyTorch analogy: `torch.sigmoid(x)`. -/
def sigmoidOp {s : Shape} : OpSpec α s s :=
{ forward      := liftElementwise (α:=α) (s:=s) Activation.Math.sigmoidSpec
, backward     := liftElementwiseBackward (α:=α) (s:=s) Activation.Math.sigmoidDerivSpec }

/-- Elementwise tanh OpSpec on any shape.

PyTorch analogy: `torch.tanh(x)`. -/
def tanhOp {s : Shape} : OpSpec α s s :=
{ forward      := liftElementwise (α:=α) (s:=s) Activation.Math.tanhSpec
, backward     := liftElementwiseBackward (α:=α) (s:=s) Activation.Math.tanhDerivSpec }

/-- Elementwise softplus OpSpec on any shape.

PyTorch analogy: `torch.nn.functional.softplus(x)`. -/
def softplusOp {s : Shape} : OpSpec α s s :=
{ forward      := liftElementwise (α:=α) (s:=s) Activation.Math.softplusSpec
, backward     := liftElementwiseBackward (α:=α) (s:=s) Activation.Math.softplusDerivSpec }

/-- Elementwise Swish / SiLU OpSpec on any shape.

PyTorch analogy: `torch.nn.functional.silu(x)`. -/
def swishOp {s : Shape} : OpSpec α s s :=
{ forward      := fun x => Activation.swishSpec (α := α) (s := s) x
, backward     := fun x dLdy => mulSpec (Activation.swishDerivSpec (α := α) (s := s) x) dLdy }

/-- Alias for `swishOp`, using the common SiLU name. -/
abbrev siluOp {s : Shape} : OpSpec α s s := swishOp (α := α) (s := s)

/-- Elementwise ELU OpSpec on any shape. -/
def eluOp {s : Shape} (eluAlpha : α) : OpSpec α s s :=
{ forward      := fun x => Activation.eluSpec (α := α) (s := s) x eluAlpha
, backward     := fun x dLdy =>
    mulSpec (Activation.eluDerivSpec (α := α) (s := s) x eluAlpha) dLdy }

/-- Elementwise tanh-approximate GELU OpSpec on any shape.

PyTorch analogy: `torch.nn.functional.gelu(x, approximate="tanh")`. -/
def geluOp [OfScientific α] {s : Shape} : OpSpec α s s :=
{ forward      := fun x => Activation.geluSpec (α := α) (s := s) x
, backward     := fun x dLdy => mulSpec (Activation.geluDerivSpec (α := α) (s := s) x) dLdy }

/-- Elementwise hyperbolic sine OpSpec. -/
def sinhOp {s : Shape} : OpSpec α s s :=
{ forward      := fun x => sinhSpec (α := α) (s := s) x
, backward     := liftElementwiseBackward (α:=α) (s:=s) MathFunctions.cosh }

/-- Elementwise hyperbolic cosine OpSpec. -/
def coshOp {s : Shape} : OpSpec α s s :=
{ forward      := fun x => coshSpec (α := α) (s := s) x
, backward     := liftElementwiseBackward (α:=α) (s:=s) MathFunctions.sinh }

/-- Elementwise "softmax" OpSpec on any shape.

This is a true softmax along the last axis (applied independently over all outer slices).

PyTorch analogy: `torch.softmax(x, dim=-1)`. -/
def softmaxOp {s : Shape} : OpSpec α s s :=
{ forward      := fun x => Activation.softmaxSpec (α := α) (s := s) x
, backward     := fun x dLdy => Activation.softmaxBackwardSpec (α := α) (s := s) x dLdy }

/-- Stable last-axis log-softmax OpSpec.

Backward recomputes the forward output so the VJP matches `logSoftmaxBackwardSpec`. Runtime engines
may cache that output instead. -/
def logSoftmaxOp {s : Shape} : OpSpec α s s :=
{ forward      := fun x => Activation.logSoftmaxSpec (α := α) (s := s) x
, backward     := fun x dLdy =>
    Activation.logSoftmaxBackwardSpec (α := α) (s := s)
      (Activation.logSoftmaxSpec (α := α) (s := s) x) dLdy }

/-! ## Linear layers -/

/-- Linear layer as an OpSpec: `y = W x + b`.

This `OpSpec` only returns the input gradient `dL/dx`. Parameter gradients for `W` and `b`
are intentionally not part of `OpSpec` (those live at the graph/runtime level).

PyTorch analogy: `torch.nn.Linear` forward, with autograd producing grads for `x/W/b`. -/
def linearOp [Add α] [Mul α] [Zero α] [One α] {inDim outDim : Nat}
  (m : LinearSpec α inDim outDim) :
  OpSpec α (.dim inDim .scalar) (.dim outDim .scalar) :=
{ forward      := fun x => linearSpec (α:=α) m x
, backward     := fun _x dLdy => linearInputDerivSpec (α:=α) m.weights dLdy }

/-- Extract scalar value from a scalar tensor.

We use this when an upstream gradient is a scalar (e.g. for reduced losses). In PyTorch this is
the common pattern "loss is scalar, so `grad_output` is a scalar too". -/
def scalarOf : Tensor α Shape.scalar → α
  | Tensor.scalar a => a

/-- Alias for `scalarOf` (clarifies intent at call sites). -/
abbrev scalarValue {α : Type} : Tensor α Shape.scalar → α := scalarOf

/-- Generic elementwise binary OpSpec with captured right-hand tensor and `d/dx`.

This is a "closure style" op: we treat the RHS tensor as a captured constant and only return
the VJP with respect to the LHS input.

PyTorch analogy: in a tape/graph, `rhs` is typically another node; here we are writing the
"lhs-only" derivative for convenience. -/
def binaryElemOp
  {s : Shape}
  (rhs : Tensor α s)
  (f : α → α → α)
  (dfdx : α → α → α) : OpSpec α s s :=
{ forward      := fun x => map2Spec f x rhs
, backward     := fun x dLdy =>
    let gx := map2Spec dfdx x rhs
    mulSpec gx dLdy
}

/-- Scale by constant scalar.

PyTorch analogy: `x * c` where `c` is a scalar constant. -/
def scaleOp {s : Shape} (c : α) : OpSpec α s s :=
{ forward      := fun x => scaleSpec x c
, backward     := fun _x dLdy => scaleSpec dLdy c }

/-! ## Unary elementwise ops -/

/-- Negation (`-x`). -/
def negOp   {s : Shape} : OpSpec α s s :=
{ forward := fun x => negSpec x
, backward := fun _x dLdy => negSpec dLdy }

/-- Absolute value (uses `signSpec` for the subgradient).

PyTorch analogy: `torch.abs(x)`. At `x = 0` we pick the subgradient `0`. -/
def absOp   {s : Shape} : OpSpec α s s :=
{ forward := fun x => absSpec x
, backward := fun x dLdy =>
    let sgn := signSpec x
    mulSpec sgn dLdy
}

/-- Smooth absolute value (a differentiable surrogate for `abs`).

This is useful when you want to avoid a kink at 0 in optimization.
PyTorch analogy: there is no single canonical `smooth_abs`, but it is similar in spirit to
`sqrt(x^2 + eps)`-style smoothings. -/
def smoothAbsOp {s : Shape} (ε : α := Numbers.epsilon) : OpSpec α s s :=
{ forward      := liftElementwise (α:=α) (s:=s) (fun x => Activation.Math.smoothAbsSpec (α := α)
  x ε)
, backward     := liftElementwiseBackward (α:=α) (s:=s) (fun x =>
  Activation.Math.smoothAbsDerivSpec (α := α) x ε) }

/-- Elementwise exp.

PyTorch analogy: `torch.exp(x)`. -/
def expOp   {s : Shape} : OpSpec α s s :=
{ forward := fun x => expSpec x
, backward := fun x dLdy => mulSpec (expSpec x) dLdy }

/--
Elementwise natural logarithm.

Domain discipline: this is the raw mathematical/PyTorch-style rule. The VJP multiplies by `1/x`,
so callers should use it only when the input is strictly positive. Runtime backends are allowed to
reject nonpositive inputs rather than silently manufacture a gradient. Use `safeLogOp` when the
intended model is `log(x + ε)`.

PyTorch analogy: `torch.log(x)`. -/
def logOp   {s : Shape} : OpSpec α s s :=
{ forward := fun x => logSpec x
, backward := fun x dLdy => mulSpec (invSpec x) dLdy }

/--
Elementwise log with epsilon shift, `log(x + ε)`.

This is the default facade-safe logarithm: it is total as a spec expression and its VJP uses
`1/(x+ε)` pointwise.

PyTorch analogy: often written manually as `torch.log(x + eps)`. -/
def safeLogOp {s : Shape} (ε : α := Numbers.epsilon) : OpSpec α s s :=
{ forward      := liftElementwise (α:=α) (s:=s) (fun x => Activation.Math.safeLogSpec (α := α) x
  ε)
, backward     := liftElementwiseBackward (α:=α) (s:=s) (fun x =>
  Activation.Math.safeLogDerivSpec (α := α) x ε) }

/--
Elementwise square root.

Domain discipline: TorchLean's spec-level `sqrtSpec` is total by clamping the forward value on
nonpositive inputs. The VJP follows that convention and returns zero where `x <= 0`, rather than
introducing an artificial `1/ε` spike.

PyTorch analogy: `torch.sqrt(x)` on the positive region, with an explicit TorchLean subgradient
choice outside the classical domain.
-/
def sqrtOp  {s : Shape} : OpSpec α s s :=
{ forward := fun x => sqrtSpec x
, backward := fun x dLdy =>
    -- `sqrtSpec` clamps negative inputs to 0 (so the forward is constant on `x <= 0`).
    -- We reflect that in the VJP: for `x <= 0` we return `0` rather than a "safe divide"
    -- that would introduce an artificial `1/epsilon` spike.
    --
    -- This also matches the runtime autograd rule used in `NN/Runtime/Autograd/*`.
    let dsqrt : Tensor α s :=
      mapSpec (α := α) (s := s) (fun v =>
        if v > 0 then
          (1 : α) / (Numbers.two * MathFunctions.sqrt v)
        else
          (0 : α)) x
    mulSpec dsqrt dLdy
}

/-- Elementwise square (`x^2`). -/
def squareOp {s : Shape} : OpSpec α s s :=
{ forward := fun x => squareSpec x
, backward := fun x dLdy => mulSpec (mulSpec (fill (Numbers.two) _) x) dLdy }

/-- Elementwise power with a captured RHS exponent tensor.

This is the VJP with respect to the base `x` for `x ^ rhs`. Domain restrictions are the usual
ones for the scalar backend's power operation. -/
def powOp {s : Shape} (rhs : Tensor α s) : OpSpec α s s :=
{ forward := fun x => powSpec x rhs
, backward := fun x dLdy =>
    let exponentMinusOne := subSpec rhs (fill (1 : α) s)
    let localGrad := mulSpec rhs (powSpec x exponentMinusOne)
    mulSpec localGrad dLdy }

/--
Elementwise reciprocal, `1/x`.

Domain discipline: this is the raw reciprocal. Its VJP is `-1/x^2`, so callers should use it only
when zero is excluded by the surrounding invariant. Use `safeInvOp` when the intended model is
`1/(x+ε)`.

PyTorch analogy: `torch.reciprocal(x)` or `1 / x`. -/
def invOp   {s : Shape} : OpSpec α s s :=
{ forward := fun x => invSpec x
, backward := fun x dLdy =>
    -- d/dx (1/x) = -1/x^2
    let x2 := squareSpec x
    let g := mulSpec (fill (-(1 : α)) _) (invSpec x2)
    mulSpec g dLdy
}

/--
Elementwise epsilon-shifted reciprocal, `1/(x+ε)`.

This is the safe facade counterpart to `invOp`: the forward pass delegates to `safedivSpec` with
unit numerator, and the VJP is the derivative of the same shifted expression.

PyTorch analogy: usually written manually as `1.0 / (x + eps)`.
-/
def safeInvOp {s : Shape} : OpSpec α s s :=
{ forward := fun x => safedivSpec (fill (1 : α) _) x
, backward := fun x dLdy =>
    let denomInv := mapSpec (fun y => (1 : α) / (y + Numbers.epsilon)) x
    let g := mulSpec (fill (-(1 : α)) _) (squareSpec denomInv)
    mulSpec g dLdy
}

/-! ## Binary ops capturing a right-hand tensor -/

/-- Add a captured RHS tensor (`x + rhs`). -/
def addOp  {s : Shape} (rhs : Tensor α s) : OpSpec α s s :=
  binaryElemOp (α:=α) rhs (· + ·) (fun _ _ => (1 : α))

/-- Subtract a captured RHS tensor (`x - rhs`). -/
def subOp  {s : Shape} (rhs : Tensor α s) : OpSpec α s s :=
  binaryElemOp (α:=α) rhs (· - ·) (fun _ _ => (1 : α))

/-- Elementwise multiply by a captured RHS tensor. -/
def mulOp  {s : Shape} (rhs : Tensor α s) : OpSpec α s s :=
  binaryElemOp (α:=α) rhs (· * ·) (fun _ y => y)

/--
Elementwise divide by a captured RHS tensor.

Domain discipline: this is the raw division rule. The VJP multiplies by `1/rhs`, so callers should
only use it when the captured denominator is known nonzero. Use `safeDivOp` when the
intended model is `x/(rhs+ε)`.
-/
def divOp  {s : Shape} (rhs : Tensor α s) : OpSpec α s s :=
  binaryElemOp (α:=α) rhs (· / ·) (fun _ y => (1 : α) / y)

/--
Elementwise safe division by a captured RHS tensor, `x/(rhs+ε)`.

PyTorch analogy: usually written manually as `x / (rhs + eps)`.
-/
def safeDivOp {s : Shape} (rhs : Tensor α s) : OpSpec α s s :=
{ forward      := fun x => safedivSpec x rhs
, backward     := fun _x dLdy =>
    let denomInv := mapSpec (fun y => (1 : α) / (y + Numbers.epsilon)) rhs
    mulSpec denomInv dLdy
}

/-- Elementwise min with captured RHS.

We pick a subgradient via a `<=` mask (ties go to the left input).

PyTorch analogy: `torch.minimum(x, rhs)` (subgradient convention is an implementation detail). -/
def minOp {s : Shape} (rhs : Tensor α s) : OpSpec α s s :=
{ forward      := fun x => minSpec x rhs
, backward     := fun x dLdy =>
    let mask := lessEqualSpec x rhs
    mulBoolMaskSpec dLdy mask
}

/-- Elementwise max with captured RHS.

We pick a subgradient via a `>=` mask (ties go to the left input).

PyTorch analogy: `torch.maximum(x, rhs)` (subgradient convention is an implementation detail). -/
def maxOp {s : Shape} (rhs : Tensor α s) : OpSpec α s s :=
{ forward      := fun x => maxSpec x rhs
, backward     := fun x dLdy =>
    let mask := greaterEqualSpec x rhs
    mulBoolMaskSpec dLdy mask
}

/-- Leaky ReLU with slope parameter.

PyTorch analogy: `torch.nn.functional.leaky_relu(x, negative_slope=alpha_l)`. -/
def leakyReluOp {s : Shape} (αₗ : α) : OpSpec α s s :=
{ forward      := fun x => mapSpec (fun v => if v > 0 then v else αₗ * v) x
, backward     := fun x dLdy =>
    let df := mapSpec (fun v => if v > 0 then (1 : α) else αₗ) x
    mulSpec df dLdy
}

/-- Clamp OpSpec with a fixed interval.

We choose the standard subgradient `1` strictly inside the interval and `0` at/outside the
boundaries, matching `clampDerivativeSpec`. -/
def clampOp {s : Shape} (minVal maxVal : α) : OpSpec α s s :=
{ forward      := fun x => clampSpec x minVal maxVal
, backward     := fun x dLdy => mulSpec (clampDerivativeSpec x minVal maxVal) dLdy }

/-! ## Loss OpSpecs -/

/-
These loss ops:

- capture the `target` tensor (so the op is a function of predictions only),
- return a scalar tensor (`Shape.scalar`),
- scale the per-element gradient by the upstream scalar `dL/dy`.

PyTorch analogy: `torch.nn.functional.*_loss(reduction="mean")` returns a scalar, and the upstream
gradient is a scalar too. (Our exact loss semantics live in `NN/Spec/Layers/Loss.lean`.)
-/

/-- MSE loss (returns a scalar), capturing the target. -/
def mseLossOp {s : Shape} (target : Tensor α s) : OpSpec α s Shape.scalar :=
{ forward      := fun yhat => Tensor.scalar (mseSpec (α:=α) (s:=s) yhat target)
, backward     := fun yhat dLdy =>
    let g := scalarOf dLdy
    scaleSpec (mseDerivSpec (α:=α) (s:=s) yhat target) g
}

/-- MAE loss (returns a scalar), capturing the target. -/
def maeLossOp {s : Shape} (target : Tensor α s) : OpSpec α s Shape.scalar :=
{ forward      := fun yhat => Tensor.scalar (maeSpec (α:=α) (s:=s) yhat target)
, backward     := fun yhat dLdy =>
    let g := scalarOf dLdy
    scaleSpec (maeDerivSpec (α:=α) (s:=s) yhat target) g
}

/-- Huber loss (returns a scalar), capturing the target. -/
def huberLossOp {s : Shape} (target : Tensor α s) (delta : α := (1 : α)) : OpSpec α s Shape.scalar
  :=
{ forward      := fun yhat => Tensor.scalar (huberSpec (α:=α) (s:=s) yhat target delta)
, backward     := fun yhat dLdy =>
    let g := scalarOf dLdy
    scaleSpec (huberDerivSpec (α:=α) (s:=s) yhat target delta) g
}

/-- Cross-entropy loss (returns a scalar), capturing the target distribution.

This is "cross-entropy between distributions": `target` is `p`, `yhat` is `q`.
PyTorch analogy: closer to `-(p * log(q)).mean()` than to the logits-based
`torch.nn.functional.cross_entropy` default. -/
def crossEntropyLossOp {s : Shape} (target : Tensor α s) (epsilon : α := Numbers.epsilon) :
  OpSpec α s Shape.scalar :=
{ forward      := fun yhat => Tensor.scalar (crossEntropySpec (α:=α) (s:=s) yhat target epsilon)
, backward     := fun yhat dLdy =>
    let g := scalarOf dLdy
    scaleSpec (crossEntropyDerivSpec (α:=α) (s:=s) yhat target epsilon) g
}

/-- Logits-based cross-entropy loss, capturing the target distribution. -/
def crossEntropyLogitsLossOp {s : Shape} (target : Tensor α s) : OpSpec α s Shape.scalar :=
{ forward      := fun logits => Tensor.scalar (crossEntropyLogitsSpec (α:=α) (s:=s) logits target)
, backward     := fun logits dLdy =>
    let g := scalarOf dLdy
    scaleSpec (crossEntropyLogitsDerivSpec (α:=α) (s:=s) logits target) g }

/-- Binary cross-entropy loss on probability tensors, capturing the target tensor. -/
def binaryCrossEntropyLossOp {s : Shape} (target : Tensor α s) (epsilon : α := Numbers.epsilon) :
    OpSpec α s Shape.scalar :=
{ forward      := fun yhat =>
    Tensor.scalar (binaryCrossEntropyTensorSpec (α:=α) (s:=s) yhat target epsilon)
, backward     := fun yhat dLdy =>
    let g := scalarOf dLdy
    scaleSpec (binaryCrossEntropyTensorDerivSpec (α:=α) (s:=s) yhat target epsilon) g }

/-- Cosine-similarity loss, capturing the target tensor. -/
def cosineSimilarityLossOp {s : Shape} (target : Tensor α s) (epsilon : α := Numbers.epsilon) :
    OpSpec α s Shape.scalar :=
{ forward      := fun yhat => Tensor.scalar (cosineSimilaritySpec (α:=α) (s:=s) yhat target epsilon)
, backward     := fun yhat dLdy =>
    let g := scalarOf dLdy
    scaleSpec (cosineSimilarityDerivSpec (α:=α) (s:=s) yhat target epsilon) g }

/-- Hinge loss (returns a scalar), capturing the target. -/
def hingeLossOp {s : Shape} (target : Tensor α s) : OpSpec α s Shape.scalar :=
{ forward      := fun yhat => Tensor.scalar (hingeSpec (α:=α) (s:=s) yhat target)
, backward     := fun yhat dLdy =>
    let g := scalarOf dLdy
    scaleSpec (hingeDerivSpec (α:=α) (s:=s) yhat target) g
}

/-- Poisson loss (returns a scalar), capturing the target. -/
def poissonLossOp {s : Shape} (target : Tensor α s) : OpSpec α s Shape.scalar :=
{ forward      := fun yhat => Tensor.scalar (poissonSpec (α:=α) (s:=s) yhat target)
, backward     := fun yhat dLdy =>
    let g := scalarOf dLdy
    scaleSpec (poissonDerivSpec (α:=α) (s:=s) yhat target) g
}

/-- Log-cosh loss (returns a scalar), capturing the target. -/
def logCoshLossOp {s : Shape} (target : Tensor α s) : OpSpec α s Shape.scalar :=
{ forward      := fun yhat => Tensor.scalar (logCoshSpec (α:=α) (s:=s) yhat target)
, backward     := fun yhat dLdy =>
    let g := scalarOf dLdy
    scaleSpec (logCoshDerivSpec (α:=α) (s:=s) yhat target) g
}

/-! ## Normalization -/

/-- LayerNorm OpSpec over (seqLen × embedDim). Parameters gamma/beta are captured.
Backward returns only ∂L/∂x (parameter grads are not returned at this level). -/
def layerNormOp (seqLen embedDim : Nat)
  (gamma : Tensor α (.dim embedDim .scalar))
  (beta  : Tensor α (.dim embedDim .scalar))
  (h_seq_pos : seqLen > 0 := by decide)
  (h_embed_pos : embedDim > 0 := by decide) :
  OpSpec α (.dim seqLen (.dim embedDim .scalar)) (.dim seqLen (.dim embedDim .scalar)) :=
{ forward      := fun x => layerNorm (α:=α) (seqLen:=seqLen) (embedDim:=embedDim) x gamma beta
    h_seq_pos h_embed_pos
, backward     := fun x dLdy =>
    let (dx, _dg, _db) := layerNormBackward (α:=α) (seqLen:=seqLen) (embedDim:=embedDim) h_seq_pos
      h_embed_pos x gamma beta dLdy
    dx
}

/-- Identity op: pass-through forward and backward. -/
def identityOp {s : Shape} : OpSpec α s s :=
{ forward      := fun x => x
, backward     := fun _x dLdy => dLdy }

/-! ## Shape/structure ops -/

/-- Reshape op (requires a size-equality proof).

PyTorch analogy: `x.reshape(...)` (or `view`), but here the shape relationship is explicit. -/
def reshapeOp {s t : Shape} [Inhabited α]
  (h : s.size = t.size) : OpSpec α s t :=
{ forward      := fun x => reshapeSpec (α:=α) (s₁:=s) (s₂:=t) x h
, backward     := fun _x dLdz => reshapeSpec (α:=α) (s₁:=t) (s₂:=s) dLdz h.symm }

/-- Matrix transpose (2D) op.

PyTorch analogy: `x.transpose(0, 1)` for a matrix. -/
def matrixTransposeOp {m n : Nat} :
  OpSpec α (.dim m (.dim n .scalar)) (.dim n (.dim m .scalar)) :=
{ forward      := fun x => matrixTransposeSpec (α:=α) x
, backward     := fun _x dLdz => matrixTransposeSpec (α:=α) dLdz }

/-- Fill a tensor with a constant (ignores input).

PyTorch analogy: `torch.full_like(x, value)` (but here we keep the input only to fit the `OpSpec`
shape, and ignore its content). -/
def constantOp {s : Shape} (value : α) [Inhabited α] : OpSpec α s s :=
{ forward      := fun _x => broadcastFill (α:=α) s value
, backward     := fun x _d => mapSpec (fun _ => 0) x }

/-- Replicate a scalar to any shape; backward sums gradients back to a scalar.

PyTorch analogy: broadcasting a scalar in arithmetic, and in backward accumulating by sum. -/
def scalarToShapeOp {s : Shape} [Inhabited α] :
  OpSpec α Shape.scalar s :=
{ forward      := fun a => replicate (α:=α) (s:=s) a
, backward     := fun _a dLdy => Tensor.scalar (sumSpec (α:=α) dLdy) }

/-- Apply boolean mask: keep where mask true, else set 0.

PyTorch analogy: `torch.where(mask, x, 0)`. -/
def applyMaskOp {s : Shape} (mask : Tensor Bool s) : OpSpec α s s :=
{ forward      := fun x => map2Spec (fun v b => if b then v else 0) x mask
, backward     := fun _x dLdy => map2Spec (fun g b => if b then g else 0) dLdy mask }

/-- Deterministic inference-style dropout scaling. -/
def dropoutInferenceOp {s : Shape} (p : α) : OpSpec α s s :=
{ forward      := fun x => dropoutInferenceSpec (α:=α) p x
, backward     := fun _x dLdy => dropoutInferenceBackwardSpec (α:=α) p dLdy }

/-- Masked inverted-dropout OpSpec with an explicit mask. -/
def dropoutMaskedOp {s : Shape} (p : α) (mask : Tensor Bool s) : OpSpec α s s :=
{ forward      := fun x => dropoutMaskedSpec (α:=α) p mask x
, backward     := fun _x dLdy => dropoutMaskedBackwardSpec (α:=α) p mask dLdy }

/-- Right-multiply by fixed matrix: X (m×n) ↦ X·B (m×p). -/
def matmulRightOp {m n p : Nat}
  (B : Tensor α (.dim n (.dim p .scalar))) :
  OpSpec α (.dim m (.dim n .scalar)) (.dim m (.dim p .scalar)) :=
{ forward      := fun X => matMulSpec (α:=α) X B
, backward     := fun _X dLdz => matMulSpec (α:=α) dLdz (matrixTransposeSpec B) }

/-- Left-multiply by fixed matrix: X (n×p) ↦ A·X (m×p). -/
def matmulLeftOp {m n p : Nat}
  (A : Tensor α (.dim m (.dim n .scalar))) :
  OpSpec α (.dim n (.dim p .scalar)) (.dim m (.dim p .scalar)) :=
{ forward      := fun X => matMulSpec (α:=α) A X
, backward     := fun _X dLdz => matMulSpec (α:=α) (matrixTransposeSpec A) dLdz }

/-- Batched matrix multiply with captured RHS: `A ↦ A @ B`. -/
def bmmRightOp {batch m n p : Nat}
  (B : Tensor α (.dim batch (.dim n (.dim p .scalar)))) :
  OpSpec α (.dim batch (.dim m (.dim n .scalar))) (.dim batch (.dim m (.dim p .scalar))) :=
{ forward      := fun A => bmmSpec (α:=α) A B
, backward     := fun A dLdz => (bmmBackwardSpec (α:=α) A B dLdz).1 }

/-- Batched matrix multiply with captured LHS: `B ↦ A @ B`. -/
def bmmLeftOp {batch m n p : Nat}
  (A : Tensor α (.dim batch (.dim m (.dim n .scalar)))) :
  OpSpec α (.dim batch (.dim n (.dim p .scalar))) (.dim batch (.dim m (.dim p .scalar))) :=
{ forward      := fun B => bmmSpec (α:=α) A B
, backward     := fun B dLdz => (bmmBackwardSpec (α:=α) A B dLdz).2 }

/-- One-hot embedding as an OpSpec over the one-hot input. Parameter gradients stay outside
`OpSpec`; this wrapper returns only `dOneHot`. -/
def embeddingOnehotOp {vocab embedDim seqLen : Nat}
  (emb : EmbeddingSpec vocab embedDim α) :
  OpSpec α (.dim seqLen (.dim vocab .scalar)) (.dim seqLen (.dim embedDim .scalar)) :=
{ forward      := fun oneHot => embeddingOnehotSpec (α:=α) emb oneHot
, backward     := fun oneHot dLdy => (embeddingOnehotBackwardSpec (α:=α) emb oneHot dLdy).1 }

/-- Expand vector to column (unsqueeze last dim size 1) and the inverse. -/
def expandToColOp {n : Nat} {s : Shape} :
  OpSpec α (.dim n s) (.dim n (.dim 1 s)) :=
{ forward      := fun t => expandToColSpec (α:=α) t
, backward     := fun _t dLdz => squeezeColSpec (α:=α) dLdz }

/-- Squeeze a trailing singleton dim `(n,1,...)` to `(n,...)` (adjoint unsqueezes). -/
def squeezeColOp {n : Nat} {s : Shape} :
  OpSpec α (.dim n (.dim 1 s)) (.dim n s) :=
{ forward      := fun t => squeezeColSpec (α:=α) t
, backward     := fun _t dLdz => expandToColSpec (α:=α) dLdz }

/-- Concatenate along the leading dimension with a captured RHS, returning the gradient slice for
the LHS input. -/
def concatDim0LeftOp {n m : Nat} {s : Shape}
  (rhs : Tensor α (.dim m s)) :
  OpSpec α (.dim n s) (.dim (n + m) s) :=
{ forward      := fun lhs => concatDim0Spec lhs rhs
, backward     := fun _lhs dLdz =>
    sliceRange0Spec (α:=α) (n := n + m) (s := s) 0 n (by
      simp) dLdz }

/-- Concatenate along the leading dimension with a captured LHS, returning the gradient slice for
the RHS input. -/
def concatDim0RightOp {n m : Nat} {s : Shape}
  (lhs : Tensor α (.dim n s)) :
  OpSpec α (.dim m s) (.dim (n + m) s) :=
{ forward      := fun rhs => concatDim0Spec lhs rhs
, backward     := fun _rhs dLdz =>
    sliceRange0Spec (α:=α) (n := n + m) (s := s) n m (by
      rw [Nat.add_comm m n]) dLdz }

/-- Slice a leading-axis range; backward inserts the upstream gradient into the original shape. -/
def sliceRange0Op {n : Nat} {s : Shape} [Zero α]
  (start len : Nat) (h : len + start ≤ n) :
  OpSpec α (.dim n s) (.dim len s) :=
{ forward      := fun x => sliceRange0Spec (α:=α) (n := n) (s := s) start len h x
, backward     := fun _x dLdy => sliceRange0BackwardSpec (α:=α) (n := n) (s := s) start len h dLdy }

/-! ## Reductions and broadcasting -/

/-- Reduce-sum along axis using a `valid_axis` proof; backward broadcasts back.

PyTorch analogy: `torch.sum(x, dim=axis)` (with `keepdim=false`). -/
def reduceSumOp {s : Shape} (axis : Nat)
  [valid : Shape.valid_axis_inst axis s]
  [wf : Shape.WellFormed s]
  [Inhabited α] :
  OpSpec α s (shapeAfterSum s axis) :=
{ forward      := fun x => reduceSumAuto (α:=α) axis x
, backward     := fun _x dLdz =>
    let cb := shapeAfterSumBroadcastBack axis valid wf
    broadcastTo cb dLdz
}

/-- Generic broadcasting-aware binary OpSpec.

The caller supplies:

- explicit broadcast proofs (`CanBroadcastTo`) for both sides, and
- a `reduce_back` map that takes a gradient in the broadcasted shape `t` and reduces it back to
  the left shape `s1`.

PyTorch analogy: this is where PyTorch's implicit broadcasting rules and reduction-of-broadcasted
gradients ("sum over broadcasted dimensions") happen. In TorchLean we keep those shape relations
explicit. -/
def binaryBroadcastOp {s1 s2 t : Shape}
  [Inhabited α]
  (rhs : Tensor α s2)
  (cbx : Shape.CanBroadcastTo s1 t)
  (cby : Shape.CanBroadcastTo s2 t)
  (f : α → α → α)
  (dfdx : α → α → α)
  (reduce_back : Tensor α t → Tensor α s1) :
  OpSpec α s1 t :=
{ forward      := fun x =>
    let xb := broadcastTo (α:=α) cbx x
    let yb := broadcastTo (α:=α) cby rhs
    map2Spec f xb yb
, backward     := fun x dLdz =>
    let xb := broadcastTo (α:=α) cbx x
    let yb := broadcastTo (α:=α) cby rhs
    let gx := map2Spec dfdx xb yb
    let g  := mulSpec gx dLdz
    reduce_back g
}

/-- Convenience: broadcasting-aware add with caller-provided reduction. -/
def addBroadcastOp {s1 s2 t : Shape}
  [Inhabited α]
  (rhs : Tensor α s2) (cbx : Shape.CanBroadcastTo s1 t) (cby : Shape.CanBroadcastTo s2 t)
  (reduce_back : Tensor α t → Tensor α s1) :
  OpSpec α s1 t :=
  binaryBroadcastOp (α:=α) rhs cbx cby (· + ·) (fun _ _ => (1 : α)) reduce_back

/-- Convenience: broadcasting-aware mul with caller-provided reduction. -/
def mulBroadcastOp {s1 s2 t : Shape}
  [Inhabited α]
  (rhs : Tensor α s2) (cbx : Shape.CanBroadcastTo s1 t) (cby : Shape.CanBroadcastTo s2 t)
  (reduce_back : Tensor α t → Tensor α s1) :
  OpSpec α s1 t :=
  binaryBroadcastOp (α:=α) rhs cbx cby (· * ·) (fun _ y => y) reduce_back

end Spec

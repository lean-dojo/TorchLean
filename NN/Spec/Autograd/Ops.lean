/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Autograd.AutogradSpec
public import NN.Spec.Autograd.Trigonometric
public import NN.Spec.Layers.Dropout
public import NN.Spec.Layers.Embedding
public import NN.Spec.Layers.Linear
public import NN.Spec.Layers.Loss

/-!
# Autograd OpSpecs (spec layer)

This file defines small `OpSpec` building blocks (forward + VJP) for common tensor operations.
The definitions are intentionally direct mathematical contracts and live purely in the spec layer.

The declarations follow a uniform pattern:

- Each operation below is an `OpSpec`: a pure `forward` plus a pure VJP `backward`.
- Most ops here package `*Spec` and derivative-spec definitions from `NN/Spec/*`.

Where this sits in TorchLean:

- `NN.Spec.*` files define pure denotational semantics: what tensors/layers mean.
- This file packages some of those pure definitions as unary `OpSpec`s: `forward` plus VJP.
- `NN.Runtime.Autograd.*` executes programs, tracks parameters, manages tapes/sessions, dispatches
  CUDA kernels, handles RNG, and lowers graphs for reusable typed execution.

This file adapts operations whose input-gradient VJP is naturally expressed as a single `OpSpec`.
Larger multi-input or parameterized layers (convolution, attention, batchnorm, pooling, RNG)
still have precise specs and runtime implementations, but their full backward state usually belongs
in layer/runtime code rather than in this compact unary interface.

PyTorch analogy (approximately):

- A spec `OpSpec` is like a compact `torch.autograd.Function` where we write down the VJP directly.
- We do not model PyTorch's mutable `ctx`; the spec layer receives the input tensor `x` directly.
-/

@[expose] public section


open TorchLean

namespace Spec

open TorchLean TorchLean.Tensor
open Activation

variable {α : Type} [TorchLean.Storage α]

/-! ## Elementwise lifting helpers -/

/-- Lift a scalar function to a tensor by pointwise map.

PyTorch analogy: most `torch.*` pointwise ops are vectorized elementwise maps. -/
def liftElementwise {s : Shape}
  (f : α → α) : Tensor α s → Tensor α s :=
  mapSpec f

/-- Lift an elementwise backward using the chain rule:
$\frac{\partial L}{\partial x}=f'(x)\frac{\partial L}{\partial y}$ pointwise.

This is the standard VJP pattern for elementwise ops.

PyTorch analogy: the "local backward" rule for a pointwise op multiplies by the derivative mask. -/
def liftElementwiseBackward [Mul α] {s : Shape}
  (df : α → α) : Tensor α s → Tensor α s → Tensor α s :=
  fun x dLdy =>
    let gx := mapSpec df x
    mulSpec gx dLdy

/-- Elementwise ReLU OpSpec on any shape.

PyTorch analogy: `torch.relu(x)` / `torch.nn.functional.relu(x)`. -/
def reluOp [Mul α] [One α] [Zero α] [Max α] [BEq α] [LT α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {s : Shape} : OpSpec α s s :=
{ forward      := liftElementwise (α:=α) (s:=s) Activation.Math.reluSpec
, backward     := liftElementwiseBackward (α:=α) (s:=s) Activation.Math.reluDerivSpec }

variable [Context α]

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

/-- Elementwise SiLU (also called Swish) OpSpec on any shape.

PyTorch analogy: `torch.nn.functional.silu(x)`. -/
def siluOp {s : Shape} : OpSpec α s s :=
{ forward      := fun x => Activation.swishSpec (α := α) (s := s) x
, backward     := fun x dLdy => mulSpec (Activation.swishDerivSpec (α := α) (s := s) x) dLdy }

/-- Elementwise ELU OpSpec on any shape. -/
def eluOp {s : Shape} (eluAlpha : α) : OpSpec α s s :=
{ forward      := fun x => Activation.eluSpec (α := α) (s := s) x eluAlpha
, backward     := fun x dLdy =>
    mulSpec (Activation.eluDerivSpec (α := α) (s := s) x eluAlpha) dLdy }

/-- Elementwise tanh-approximate GELU OpSpec on any shape.

PyTorch analogy: `torch.nn.functional.gelu(x, approximate="tanh")`. -/
def geluOp {s : Shape} : OpSpec α s s :=
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

/-- Softmax `OpSpec` along an explicitly selected tensor dimension.

PyTorch analogy: `torch.softmax(x, dim=axis)`. -/
def softmaxOp {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s] : OpSpec α s s :=
{ forward      := fun x => Activation.softmaxSpec (α := α) (s := s) axis x
, backward     := fun x dLdy =>
    Activation.softmaxBackwardSpec (α := α) (s := s) axis x dLdy }

/-- Stable log-softmax `OpSpec` along an explicitly selected tensor dimension.

Backward recomputes the forward output so the VJP uses the same axis-parametric semantics. Runtime
engines may cache that output instead. -/
def logSoftmaxOp {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s] : OpSpec α s s :=
{ forward      := fun x => Activation.logSoftmaxSpec (α := α) (s := s) axis x
, backward     := fun x dLdy =>
    Activation.logSoftmaxBackwardSpec (α := α) (s := s) axis
      (Activation.logSoftmaxSpec (α := α) (s := s) axis x) dLdy }

/-! ## Linear layers -/

/-- Linear layer as an OpSpec: $y=Wx+b$.

This `OpSpec` only returns the input gradient $\partial L/\partial x$. Parameter gradients for
$W$ and $b$
are not part of `OpSpec` (those live at the graph/runtime level).

PyTorch analogy: `torch.nn.functional.linear` forward, with autograd producing gradients for
`x`, `W`, and `b`. -/
def linearOp {α : Type} [TorchLean.Storage α]
    [Add α] [Mul α] [Zero α] [One α] {inDim outDim : Nat}
  (m : LinearSpec α inDim outDim) :
  OpSpec α ([inDim]) ([outDim]) :=
{ forward      := fun x => linearSpec (α:=α) m x
, backward     := fun _x dLdy => linearInputDerivSpec (α:=α) m.weights dLdy }

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

PyTorch analogy: `torch.abs(x)`. At $x=0$ we pick the subgradient $0$. -/
def absOp   {s : Shape} : OpSpec α s s :=
{ forward := fun x => absSpec x
, backward := fun x dLdy =>
    let sgn := signSpec x
    mulSpec sgn dLdy
}

/-- Smooth absolute value (a differentiable surrogate for `abs`).

This is useful when you want to avoid a kink at 0 in optimization.
PyTorch analogy: there is no single canonical `smoothAbs`, but it is similar in spirit to
$\sqrt{x^2+\varepsilon}$-style smoothings. -/
def smoothAbsOp {s : Shape} (ε : α := Context.defaultEpsilon) : OpSpec α s s :=
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
intended model is the smooth surrogate $\log(\operatorname{softplus}(x)+\varepsilon)$.

PyTorch analogy: `torch.log(x)`. -/
def logOp   {s : Shape} : OpSpec α s s :=
{ forward := fun x => logSpec x
, backward := fun x dLdy => mulSpec (invSpec x) dLdy }

/--
Elementwise smooth logarithm surrogate, $\log(\operatorname{softplus}(x)+\varepsilon)$.

For $\varepsilon>0$ this is defined on every real input. Its VJP multiplies by
$\operatorname{sigmoid}(x)/(\operatorname{softplus}(x)+\varepsilon)$.

PyTorch expression: `torch.log(torch.nn.functional.softplus(x) + eps)`.
-/
def safeLogOp {s : Shape} (ε : α := Context.defaultEpsilon) : OpSpec α s s :=
{ forward      := liftElementwise (α:=α) (s:=s) (fun x => Activation.Math.safeLogSpec (α := α) x
  ε)
, backward     := liftElementwiseBackward (α:=α) (s:=s) (fun x =>
  Activation.Math.safeLogDerivSpec (α := α) x ε) }

/--
Elementwise square root.

Domain discipline: TorchLean's spec-level `sqrtSpec` is total by clamping the forward value on
nonpositive inputs. The VJP follows that convention and returns zero where $x\le0$, rather than
introducing an artificial $1/\varepsilon$ spike.

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
          (1 : α) / (2 * MathFunctions.sqrt v)
        else
          (0 : α)) x
    mulSpec dsqrt dLdy
}

/-- Elementwise square, $x^2$. -/
def squareOp {s : Shape} : OpSpec α s s :=
{ forward := fun x => squareSpec x
, backward := fun x dLdy => mulSpec (mulSpec (Tensor.full _ (2)) x) dLdy }

/-- Elementwise power with a captured RHS exponent tensor.

This is the VJP with respect to the base $x$ for $x^{\mathtt{rhs}}$. Domain restrictions are the
usual ones for the scalar backend's power operation. -/
def powOp {s : Shape} (rhs : Tensor α s) : OpSpec α s s :=
{ forward := fun x => powSpec x rhs
, backward := fun x dLdy =>
    let exponentMinusOne := subSpec rhs (Tensor.full s (1 : α))
    let localGrad := mulSpec rhs (powSpec x exponentMinusOne)
    mulSpec localGrad dLdy }

/--
Elementwise reciprocal, $1/x$.

Domain discipline: this is the raw reciprocal. Its VJP is $-1/x^2$, so callers should use it only
when zero is excluded by the surrounding invariant. Use `safeInvOp` when the intended model is
$1/(x+\varepsilon)$.

PyTorch analogy: `torch.reciprocal(x)` or `1 / x`. -/
def invOp   {s : Shape} : OpSpec α s s :=
{ forward := fun x => invSpec x
, backward := fun x dLdy =>
    -- d/dx (1/x) = -1/x^2
    let x2 := squareSpec x
    let g := mulSpec (Tensor.full _ (-(1 : α))) (invSpec x2)
    mulSpec g dLdy
}

/--
Elementwise epsilon-shifted reciprocal, $1/(x+\varepsilon)$.

This is the safe API counterpart to `invOp`: the forward pass delegates to `safedivSpec` with
unit numerator, and the VJP is the derivative of the same shifted expression.

PyTorch analogy: usually written manually as `1.0 / (x + eps)`.
-/
def safeInvOp {s : Shape} : OpSpec α s s :=
{ forward := fun x => safedivSpec (Tensor.full _ (1 : α)) x
, backward := fun x dLdy =>
    let denomInv := mapSpec (fun y => (1 : α) / (y + Context.defaultEpsilon)) x
    let g := mulSpec (Tensor.full _ (-(1 : α))) (squareSpec denomInv)
    mulSpec g dLdy
}

/-! ## Binary ops capturing a right-hand tensor -/

/-- Add a captured RHS tensor, $x+\mathtt{rhs}$. -/
def addOp  {s : Shape} (rhs : Tensor α s) : OpSpec α s s :=
  binaryElemOp (α:=α) rhs (· + ·) (fun _ _ => (1 : α))

/-- Subtract a captured RHS tensor, $x-\mathtt{rhs}$. -/
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
Elementwise safe division by a captured RHS tensor, $x/(\mathtt{rhs}+\varepsilon)$.

PyTorch analogy: usually written manually as `x / (rhs + eps)`.
-/
def safeDivOp {s : Shape} (rhs : Tensor α s) : OpSpec α s s :=
{ forward      := fun x => safedivSpec x rhs
, backward     := fun _x dLdy =>
    let denomInv := mapSpec (fun y => (1 : α) / (y + Context.defaultEpsilon)) rhs
    mulSpec denomInv dLdy
}

/-- Elementwise minimum with a captured right-hand tensor.

The backward pass gives the input the full upstream gradient where it is strictly smaller than
`rhs`, zero where it is strictly larger, and half at a tie. This is the same selected gradient as
the two-input tape operation. Capturing `rhs` removes its gradient output; it does not transfer
its half of a tied gradient to the remaining input.

Away from ties this is the usual derivative. At a tie, minimum is not differentiable, and the
half-gradient is the convention used by the runtime.
-/
def minOp {s : Shape} (rhs : Tensor α s) : OpSpec α s s :=
{ forward      := fun x => minSpec x rhs
, backward     := fun x dLdy =>
    let half : α := (1 : α) / ((2 : Nat) : α)
    let mask := map2Spec (fun a b =>
      if b > a then (1 : α) else if a > b then (0 : α) else half) x rhs
    mulSpec mask dLdy
}

/-- Elementwise maximum with a captured right-hand tensor.

The backward pass gives the input the full upstream gradient where it is strictly larger than
`rhs`, zero where it is strictly smaller, and half at a tie. As in `minOp`, the captured tensor
keeps its share of the selected gradient even though this operation returns only the gradient
with respect to the input.

The strict comparisons and their order match the two-input tape operation, including its
fallback when neither comparison holds.
-/
def maxOp {s : Shape} (rhs : Tensor α s) : OpSpec α s s :=
{ forward      := fun x => maxSpec x rhs
, backward     := fun x dLdy =>
    let half : α := (1 : α) / ((2 : Nat) : α)
    let mask := map2Spec (fun a b =>
      if a > b then (1 : α) else if b > a then (0 : α) else half) x rhs
    mulSpec mask dLdy
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
    let g := dLdy.item
    scaleSpec (mseDerivSpec (α:=α) (s:=s) yhat target) g
}

/-- MAE loss (returns a scalar), capturing the target. -/
def maeLossOp {s : Shape} (target : Tensor α s) : OpSpec α s Shape.scalar :=
{ forward      := fun yhat => Tensor.scalar (maeSpec (α:=α) (s:=s) yhat target)
, backward     := fun yhat dLdy =>
    let g := dLdy.item
    scaleSpec (maeDerivSpec (α:=α) (s:=s) yhat target) g
}

/-- Huber loss (returns a scalar), capturing the target. -/
def huberLossOp {s : Shape} (target : Tensor α s) (delta : α := (1 : α)) : OpSpec α s Shape.scalar
  :=
{ forward      := fun yhat => Tensor.scalar (huberSpec (α:=α) (s:=s) yhat target delta)
, backward     := fun yhat dLdy =>
    let g := dLdy.item
    scaleSpec (huberDerivSpec (α:=α) (s:=s) yhat target delta) g
}

/-- Cross-entropy loss (returns a scalar), capturing the target distribution.

This is "cross-entropy between distributions": `target` is $p$, `yhat` is $q$.
PyTorch analogy: closer to `-(p * log(q)).mean()` than to the logits-based
`torch.nn.functional.cross_entropy` default. -/
def crossEntropyLossOp {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (target : Tensor α s) (epsilon : α := Context.defaultEpsilon) :
  OpSpec α s Shape.scalar :=
{ forward      := fun yhat =>
    Tensor.scalar (crossEntropySpec (α:=α) (s:=s) axis yhat target epsilon)
, backward     := fun yhat dLdy =>
    let g := dLdy.item
    scaleSpec (crossEntropyDerivSpec (α:=α) (s:=s) axis yhat target epsilon) g
}

/-- Logits-based cross-entropy loss, capturing the target distribution. -/
def crossEntropyLogitsLossOp {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (target : Tensor α s) : OpSpec α s Shape.scalar :=
{ forward      := fun logits =>
    Tensor.scalar (crossEntropyLogitsSpec (α:=α) (s:=s) axis logits target)
, backward     := fun logits dLdy =>
    let g := dLdy.item
    scaleSpec (crossEntropyLogitsDerivSpec (α:=α) (s:=s) axis logits target) g }

/-- Binary cross-entropy loss on probability tensors, capturing the target tensor. -/
def binaryCrossEntropyLossOp {s : Shape} (target : Tensor α s)
    (epsilon : α := Context.defaultEpsilon) :
    OpSpec α s Shape.scalar :=
{ forward      := fun yhat =>
    Tensor.scalar (binaryCrossEntropyTensorSpec (α:=α) (s:=s) yhat target epsilon)
, backward     := fun yhat dLdy =>
    let g := dLdy.item
    scaleSpec (binaryCrossEntropyTensorDerivSpec (α:=α) (s:=s) yhat target epsilon) g }

/-- Cosine-similarity loss, capturing the target tensor. -/
def cosineSimilarityLossOp {s : Shape} (target : Tensor α s)
    (epsilon : α := Context.defaultEpsilon) :
    OpSpec α s Shape.scalar :=
{ forward      := fun yhat => Tensor.scalar (cosineSimilaritySpec (α:=α) (s:=s) yhat target epsilon)
, backward     := fun yhat dLdy =>
    let g := dLdy.item
    scaleSpec (cosineSimilarityDerivSpec (α:=α) (s:=s) yhat target epsilon) g }

/-- Hinge loss (returns a scalar), capturing the target. -/
def hingeLossOp {s : Shape} (target : Tensor α s) : OpSpec α s Shape.scalar :=
{ forward      := fun yhat => Tensor.scalar (hingeSpec (α:=α) (s:=s) yhat target)
, backward     := fun yhat dLdy =>
    let g := dLdy.item
    scaleSpec (hingeDerivSpec (α:=α) (s:=s) yhat target) g
}

/-- Poisson loss (returns a scalar), capturing the target. -/
def poissonLossOp {s : Shape} (target : Tensor α s) : OpSpec α s Shape.scalar :=
{ forward      := fun yhat => Tensor.scalar (poissonSpec (α:=α) (s:=s) yhat target)
, backward     := fun yhat dLdy =>
    let g := dLdy.item
    scaleSpec (poissonDerivSpec (α:=α) (s:=s) yhat target) g
}

/-- Log-cosh loss (returns a scalar), capturing the target. -/
def logCoshLossOp {s : Shape} (target : Tensor α s) : OpSpec α s Shape.scalar :=
{ forward      := fun yhat => Tensor.scalar (logCoshSpec (α:=α) (s:=s) yhat target)
, backward     := fun yhat dLdy =>
    let g := dLdy.item
    scaleSpec (logCoshDerivSpec (α:=α) (s:=s) yhat target) g
}

/-! ## Shape/structure ops -/

/-- Reshape op (requires a size-equality proof).

PyTorch analogy: `x.reshape(...)` (or `view`), but here the shape relationship is explicit. -/
def reshapeOp {s t : Shape}
  (h : s.size = t.size) : OpSpec α s t :=
{ forward      := fun x => reshapeSpec (α := α) (source := s) (target := t) x h
, backward     := fun _x dLdz =>
    reshapeSpec (α := α) (source := t) (target := s) dLdz h.symm }

/-- Swap adjacent axes at an arbitrary depth. -/
def swapAdjacentAxesOp {s : Shape} (depth : Nat) :
    OpSpec α s (s.swapAdjacentAtDepth depth) :=
{ forward := fun x => swapAdjacentAxes x depth
, backward := fun _x dLdz =>
    Tensor.castShape (swapAdjacentAxes dLdz depth)
      (by simp) }

/-- Fill a tensor with a constant (ignores input).

PyTorch analogy: `torch.full_like(x, value)` (but here we keep the input only to fit the `OpSpec`
shape, and ignore its content). -/
def constantOp {s : Shape} (value : α) : OpSpec α s s :=
{ forward      := fun _x => Tensor.full s value
, backward     := fun x _d => mapSpec (fun _ => 0) x }

/-- Replicate a scalar to any shape; backward sums gradients back to a scalar.

PyTorch analogy: broadcasting a scalar in arithmetic, and in backward accumulating by sum. -/
def scalarToShapeOp {s : Shape} :
  OpSpec α Shape.scalar s :=
{ forward      := fun a => replicate (α := α) (shape := s) a
, backward     := fun _a dLdy => Tensor.scalar (sumSpec (α:=α) dLdy) }

/-- Apply boolean mask: keep where mask true, else set 0.

PyTorch analogy: `torch.where(mask, x, 0)`. -/
def applyMaskOp {s : Shape} (mask : Tensor Bool s) : OpSpec α s s :=
{ forward      := fun x => map2Spec (fun v b => if b then v else 0) x mask
, backward     := fun _x dLdy => map2Spec (fun g b => if b then g else 0) dLdy mask }

/-- Evaluation-mode dropout, which is the identity in both the forward and backward maps. -/
def dropoutInferenceOp {s : Shape} (p : α) : OpSpec α s s :=
{ forward      := fun x => dropoutInferenceSpec (α:=α) p x
, backward     := fun _x dLdy => dropoutInferenceBackwardSpec (α:=α) p dLdy }

/-- Masked inverted-dropout OpSpec with an explicit mask. -/
def dropoutMaskedOp {s : Shape} (p : α) (mask : Tensor Bool s) : OpSpec α s s :=
{ forward      := fun x => dropoutMaskedSpec (α:=α) p mask x
, backward     := fun _x dLdy => dropoutMaskedBackwardSpec (α:=α) p mask dLdy }

/-- Matrix-rank multiplication with a captured right operand and broadcasted batch prefixes. -/
def matmulRightOp {batchA batchB batch : Shape} {m n p : Nat}
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch)
    (B : Tensor α (batchB.concat [n, p])) :
    OpSpec α (batchA.concat [m, n]) (batch.concat [m, p]) :=
{ forward := fun A => matmulSpec broadcastA broadcastB A B
, backward := fun A dLdz => (matmulBackwardSpec broadcastA broadcastB A B dLdz).1 }

/-- Matrix-rank multiplication with a captured left operand and broadcasted batch prefixes. -/
def matmulLeftOp {batchA batchB batch : Shape} {m n p : Nat}
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch)
    (A : Tensor α (batchA.concat [m, n])) :
    OpSpec α (batchB.concat [n, p]) (batch.concat [m, p]) :=
{ forward := fun B => matmulSpec broadcastA broadcastB A B
, backward := fun B dLdz => (matmulBackwardSpec broadcastA broadcastB A B dLdz).2 }

/-- One-hot embedding as an OpSpec over the one-hot input. Parameter gradients stay outside
`OpSpec`; this wrapper returns only `dOneHot`. -/
def oneHotEmbeddingOp {vocab embedDim seqLen : Nat}
  (embedding : Embedding vocab embedDim α) :
  OpSpec α ([seqLen, vocab]) ([seqLen, embedDim]) :=
{ forward      := fun oneHot => embedding.oneHot oneHot
, backward     := fun oneHot dLdy => (embedding.oneHotVjp oneHot dLdy).1 }

/-- Slice a contiguous range along an arbitrary tensor axis. -/
def sliceAxisRangeOp {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (start count : Nat) (hRange : start + count ≤ Shape.axisSize s axis) :
    OpSpec α s (s.replaceAxis axis count) :=
{ forward := fun x => sliceAxisRangeSpec axis x start count hRange
, backward := fun _x dLdy => sliceAxisRangeBackwardSpec axis start count hRange dLdy }

/-! ## Reductions and broadcasting -/

/-- Reduce-sum along axis using a `NonemptyAxis` proof; backward broadcasts back.

PyTorch analogy: `torch.sum(x, dim=axis)` (with `keepdim=false`). -/
def reduceSumOp {s : Shape} (axis : Nat)
  [_valid : Shape.HasNonemptyAxis axis s]
  [_wf : Shape.WellFormed s]
  :
  OpSpec α s (shapeAfterSum s axis) :=
{ forward      := fun x => reduceSum (α:=α) axis x _valid.proof
, backward     := fun _x dLdz => broadcastAfterSum s axis dLdz
}

/-- Generic broadcasting-aware binary OpSpec.

The caller supplies:

- explicit broadcast proofs (`CanBroadcastTo`) for both sides, and
- a `reduceBack` map that takes a gradient in the broadcasted shape `t` and reduces it back to
  the left shape `s1`.

PyTorch analogy: this is where PyTorch's implicit broadcasting rules and reduction-of-broadcasted
gradients ("sum over broadcasted dimensions") happen. In TorchLean we keep those shape relations
explicit. -/
def binaryBroadcastOp {s1 s2 t : Shape}
  (rhs : Tensor α s2)
  (cbx : Shape.CanBroadcastTo s1 t)
  (cby : Shape.CanBroadcastTo s2 t)
  (f : α → α → α)
  (dfdx : α → α → α)
  (reduceBack : Tensor α t → Tensor α s1) :
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
    reduceBack g
}

/-- Convenience: broadcasting-aware add with caller-provided reduction. -/
def addBroadcastOp {s1 s2 t : Shape}
  (rhs : Tensor α s2) (cbx : Shape.CanBroadcastTo s1 t) (cby : Shape.CanBroadcastTo s2 t)
  (reduceBack : Tensor α t → Tensor α s1) :
  OpSpec α s1 t :=
  binaryBroadcastOp (α:=α) rhs cbx cby (· + ·) (fun _ _ => (1 : α)) reduceBack

/-- Convenience: broadcasting-aware mul with caller-provided reduction. -/
def mulBroadcastOp {s1 s2 t : Shape}
  (rhs : Tensor α s2) (cbx : Shape.CanBroadcastTo s1 t) (cby : Shape.CanBroadcastTo s2 t)
  (reduceBack : Tensor α t → Tensor α s1) :
  OpSpec α s1 t :=
  binaryBroadcastOp (α:=α) rhs cbx cby (· * ·) (fun _ y => y) reduceBack

end Spec

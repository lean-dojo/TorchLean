/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Autograd.Trigonometric
public import NN.Spec.Layers.Dropout
public import NN.Spec.Layers.Embedding
public import NN.Spec.Layers.Linear
public import NN.Spec.Layers.Loss

/-!
# Tensor operations and pullbacks

An `OpSpec` pairs a pure forward computation with an input-gradient pullback. Both functions use
the tensor's scalar type and storage. Derivative and numerical-error theorems establish properties
of these functions under their stated assumptions.

This unary interface can capture a right-hand tensor as a constant; its gradient is then omitted.
Operations with several differentiable inputs or mutable state use the typed graph and runtime
interfaces. `NN.Runtime.Autograd` handles recording, sessions, and execution.
-/

@[expose] public section


open TorchLean

namespace Spec

open TorchLean TorchLean.Tensor
open Activation

variable {α : Type} [TorchLean.Storage α]

/-! ## Elementwise lifting helpers -/

/-- Multiply the incoming gradient by `df` at each input coordinate. -/
def liftElementwiseBackward [Mul α] {s : Shape}
  (df : α → α) : Tensor α s → Tensor α s → Tensor α s :=
  fun x dLdy =>
    let gx := Tensor.map df x
    mulSpec gx dLdy

/-- Elementwise ReLU and its input-gradient pullback. -/
def reluOp [Mul α] [One α] [Zero α] [Max α] [BEq α] [LT α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {s : Shape} : OpSpec α s s :=
{ forward      := Tensor.map (α:=α) (shape :=s) Activation.Math.reluSpec
, backward     := liftElementwiseBackward (α:=α) (s:=s) Activation.Math.reluDerivSpec }

variable [Context α]

/-- Elementwise sigmoid and its input-gradient pullback. -/
def sigmoidOp {s : Shape} : OpSpec α s s :=
{ forward      := Tensor.map (α:=α) (shape :=s) Activation.Math.sigmoidSpec
, backward     := Activation.sigmoidBackwardSpec }

/-- Elementwise tanh and its input-gradient pullback. -/
def tanhOp {s : Shape} : OpSpec α s s :=
{ forward      := Tensor.map (α:=α) (shape :=s) Activation.Math.tanhSpec
, backward     := Activation.tanhBackwardSpec }

/-- Elementwise softplus and its input-gradient pullback. -/
def softplusOp {s : Shape} : OpSpec α s s :=
{ forward      := Tensor.map (α:=α) (shape :=s) Activation.Math.softplusSpec
, backward     := liftElementwiseBackward (α:=α) (s:=s) Activation.Math.softplusDerivSpec }

/-- Elementwise SiLU (Swish) and its input-gradient pullback. -/
def siluOp {s : Shape} : OpSpec α s s :=
{ forward      := fun x => Activation.swishSpec (α := α) (s := s) x
, backward     := fun x dLdy => mulSpec (Activation.swishDerivSpec (α := α) (s := s) x) dLdy }

/-- Elementwise ELU OpSpec on any shape. -/
def eluOp {s : Shape} (eluAlpha : α) : OpSpec α s s :=
{ forward      := fun x => Activation.eluSpec (α := α) (s := s) x eluAlpha
, backward     := fun x dLdy =>
    mulSpec (Activation.eluDerivSpec (α := α) (s := s) x eluAlpha) dLdy }

/-- Elementwise tanh-approximate GELU and its input-gradient pullback. -/
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

/-- Softmax and its input-gradient pullback along `axis`. -/
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

/-- The affine map $y=Wx+b$ with captured parameters and an input-gradient pullback.
Parameter gradients are handled by the graph and runtime interfaces. -/
def linearOp {α : Type} [TorchLean.Storage α]
    [Add α] [Mul α] [Zero α] [One α] {inDim outDim : Nat}
  (m : LinearSpec α inDim outDim) :
  OpSpec α ([inDim]) ([outDim]) :=
{ forward      := fun x => linearSpec (α:=α) m x
, backward     := fun _x dLdy => linearInputDerivSpec (α:=α) m.weights dLdy }

/-- Apply `f` with a captured right operand. The pullback multiplies the incoming gradient
by `dfdx` at each coordinate and returns only the left input's gradient. -/
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

/-- Scale by a captured scalar. -/
def scaleOp {s : Shape} (c : α) : OpSpec α s s :=
{ forward      := fun x => scaleSpec x c
, backward     := fun _x dLdy => scaleSpec dLdy c }

/-! ## Unary elementwise ops -/

/-- Negation (`-x`). -/
def negOp   {s : Shape} : OpSpec α s s :=
{ forward := fun x => negSpec x
, backward := fun _x dLdy => negSpec dLdy }

/-- Absolute value, choosing subgradient zero at the origin. -/
def absOp   {s : Shape} : OpSpec α s s :=
{ forward := fun x => absSpec x
, backward := fun x dLdy =>
    let sgn := signSpec x
    mulSpec sgn dLdy
}

/-- Smooth absolute value with the derivative of the same scalar formula. -/
def smoothAbsOp {s : Shape} (ε : α := Context.defaultEpsilon) : OpSpec α s s :=
{ forward      := Tensor.map (α:=α) (shape :=s) (fun x => Activation.Math.smoothAbsSpec (α := α)
  x ε)
, backward     := liftElementwiseBackward (α:=α) (s:=s) (fun x =>
  Activation.Math.smoothAbsDerivSpec (α := α) x ε) }

/-- Elementwise exponential and its input-gradient pullback. -/
def expOp   {s : Shape} : OpSpec α s s :=
{ forward := fun x => expSpec x
, backward := fun x dLdy => mulSpec (expSpec x) dLdy }

/--
Elementwise natural logarithm.

Domain discipline: this is the raw mathematical/PyTorch-style rule. The VJP divides by `x`,
so callers should use it only when the input is strictly positive. Runtime backends are allowed to
reject nonpositive inputs rather than silently manufacture a gradient. Use `safeLogOp` when the
intended model is the smooth surrogate $\log(\operatorname{softplus}(x)+\varepsilon)$. -/
def logOp   {s : Shape} : OpSpec α s s :=
{ forward := fun x => logSpec x
, backward := fun x dLdy => divSpec dLdy x }

/--
Elementwise smooth logarithm surrogate, $\log(\operatorname{softplus}(x)+\varepsilon)$.

For $\varepsilon>0$ this is defined on every real input. Its VJP multiplies by
$\operatorname{sigmoid}(x)/(\operatorname{softplus}(x)+\varepsilon)$.

PyTorch expression: `torch.log(torch.nn.functional.softplus(x) + eps)`.
-/
def safeLogOp {s : Shape} (ε : α := Context.defaultEpsilon) : OpSpec α s s :=
{ forward      := Tensor.map (α:=α) (shape :=s) (fun x => Activation.Math.safeLogSpec (α := α) x
  ε)
, backward     := liftElementwiseBackward (α:=α) (s:=s) (fun x =>
  Activation.Math.safeLogDerivSpec (α := α) x ε) }

/--
Elementwise square root.

The forward pass evaluates `sqrt (max x 0)`. The backward pass uses the local factor
$1/(2\sqrt{x})$ when `x > 0`, and zero otherwise, then multiplies that factor by the upstream
value. Checked scalar backends replay the complete weighted quotient if an intermediate loses
range, using the root rounded by that backend. Over the reals this gives a zero VJP for
$x\le0$. On IEEE backends, a zero factor
multiplied by an infinite or NaN upstream value need not produce zero.

Positive inputs agree with `torch.sqrt`; nonpositive inputs use the stated subgradient.
-/
def sqrtOp  {s : Shape} : OpSpec α s s :=
{ forward := fun x => sqrtSpec x
, backward := fun x dLdy =>
    map2Spec (fun value seed =>
      if value > 0 then
        TorchLean.Numeric.Internal.sqrtBackward (MathFunctions.sqrt value) seed
          (factorFirst := true)
      else
        -- Preserve zero times a nonfinite upstream value, including unordered inputs.
        (0 : α) * seed) x dLdy
}

/-- Elementwise square, $x^2$. -/
def squareOp {s : Shape} : OpSpec α s s :=
{ forward := fun x => squareSpec x
, backward := fun x dLdy => mulSpec (mulSpec (Tensor.full _ (2)) x) dLdy }

/-- Elementwise power with a captured RHS exponent tensor.

This is the VJP with respect to the base $x$ for $x^{\mathtt{rhs}}$. Domain restrictions are the
usual ones for the scalar backend's power operation. The zero-exponent branch tests all
differentiation coefficients, so a zero exponent primal does not erase mixed derivatives.
For a constant exponent, checked range loss refines the complete weighted derivative instead of
rounding an unrepresentable local factor before applying the incoming gradient. -/
def powOp {s : Shape} (rhs : Tensor α s) : OpSpec α s s :=
{ forward := fun x => powSpec x rhs
, backward := fun x dLdy =>
    Tensor.Internal.Rep.ofFlatFn fun index =>
      TorchLean.Numeric.Internal.powerBackward Context.isZero
        (Tensor.Internal.Rep.getFlat x index) (Tensor.Internal.Rep.getFlat rhs index)
        (Tensor.Internal.Rep.getFlat dLdy index) }

/--
Elementwise reciprocal, $1/x$, with pullback `(-seed / x) / x`.
The successive divisions avoid a separate reciprocal or squared denominator. A numerical
comparison with real division requires nonzero `x`. `safeInvOp` instead uses $1/(x+\varepsilon)$. -/
def invOp   {s : Shape} : OpSpec α s s :=
{ forward := fun x => invSpec x
, backward := fun x dLdy =>
    divSpec (divSpec (negSpec dLdy) x) x
}

/--
Elementwise epsilon-shifted reciprocal, $1/(x+\varepsilon)$.

The forward pass delegates to `safedivSpec` with unit numerator, and the VJP uses the derivative
of the same shifted expression. The shift alone does not exclude a zero denominator: over
the reals, `x = -Context.defaultEpsilon` makes it zero.
-/
def safeInvOp {s : Shape} : OpSpec α s s :=
{ forward := fun x => safedivSpec (Tensor.full _ (1 : α)) x
, backward := fun x dLdy =>
    let denomInv := Tensor.map (fun y => (1 : α) / (y + Context.defaultEpsilon)) x
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

The VJP divides the upstream gradient by `rhs` directly, avoiding an overflowing intermediate
reciprocal when the weighted result is finite. Comparison with real division requires nonzero `rhs`.
Use `safeDivOp` for the shifted formula `x/(rhs+ε)`.
-/
def divOp  {s : Shape} (rhs : Tensor α s) : OpSpec α s s :=
{ forward := fun x => divSpec x rhs
, backward := fun _ seed => divSpec seed rhs }

/--
Elementwise epsilon-shifted division by a captured RHS tensor,
$x/(\mathtt{rhs}+\varepsilon)$. The shift does not guarantee a nonzero denominator or finite output.
-/
def safeDivOp {s : Shape} (rhs : Tensor α s) : OpSpec α s s :=
{ forward := fun x => safedivSpec x rhs
, backward := fun _ seed => safedivSpec seed rhs }

/-- Elementwise minimum with a captured right-hand tensor.

The backward pass multiplies the upstream value by a mask: one where the input is strictly smaller
than `rhs`, zero where it is strictly larger, and half when neither comparison holds. The last
case includes ties and unordered IEEE comparisons. This is the same mask as the two-input tape
operation. Capturing `rhs` removes its gradient output; it does not transfer its half of a tied
gradient to the remaining input. A zero mask can still produce NaN with a nonfinite upstream value.

Away from ties this is the usual derivative. At a tie, minimum is not differentiable, and the
half-gradient is the convention used by the runtime.
The half is a rational constant, so constructing it does not require a representable `2`.
-/
def minOp {s : Shape} (rhs : Tensor α s) : OpSpec α s s :=
{ forward      := fun x => minSpec x rhs
, backward     := fun x dLdy =>
    let half : α := Rat.cast (1 / 2 : Rat)
    let mask := map2Spec (fun a b =>
      if b > a then (1 : α) else if a > b then (0 : α) else half) x rhs
    mulSpec mask dLdy
}

/-- Elementwise maximum with a captured right-hand tensor.

The backward pass multiplies the upstream value by a mask: one where the input is strictly larger
than `rhs`, zero where it is strictly smaller, and half when neither comparison holds. As in
`minOp`, the captured tensor keeps its share of the selected gradient even though this operation
returns only the gradient with respect to the input.

The strict comparisons and their order match the two-input tape operation, including its
fallback when neither comparison holds.
-/
def maxOp {s : Shape} (rhs : Tensor α s) : OpSpec α s s :=
{ forward      := fun x => maxSpec x rhs
, backward     := fun x dLdy =>
    let half : α := Rat.cast (1 / 2 : Rat)
    let mask := map2Spec (fun a b =>
      if a > b then (1 : α) else if b > a then (0 : α) else half) x rhs
    mulSpec mask dLdy
}

/-- Leaky ReLU with a captured negative slope. -/
def leakyReluOp {s : Shape} (αₗ : α) : OpSpec α s s :=
{ forward      := fun x => Activation.leakyReluSpec x αₗ
, backward     := fun x dLdy =>
    let df := Activation.leakyReluDerivSpec x αₗ
    mulSpec df dLdy
}

/-- Clamp OpSpec with a fixed interval.

We choose the standard subgradient `1` strictly inside the interval and `0` at/outside the
boundaries, matching `clampDerivativeSpec`. -/
def clampOp {s : Shape} (minVal maxVal : α) : OpSpec α s s :=
{ forward      := fun x => clampSpec x minVal maxVal
, backward     := fun x dLdy => mulSpec (clampDerivativeSpec x minVal maxVal) dLdy }

/-! ## Loss OpSpecs -/

/- The losses capture their targets, return scalar tensors, and scale the prediction gradient
by the incoming scalar. Their formulas and reduction conventions are defined in `Layers.Loss`. -/

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

/-- Reshape between equal-size tensors; the pullback restores the input shape. -/
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

/-- Ignore the input, return a constant tensor, and return a zero input gradient. -/
def constantOp {s : Shape} (value : α) : OpSpec α s s :=
{ forward      := fun _x => Tensor.full s value
, backward     := fun x _d => Tensor.map (fun _ => 0) x }

/-- Replicate a scalar; the pullback sums the incoming tensor gradient. -/
def scalarToShapeOp {s : Shape} :
  OpSpec α Shape.scalar s :=
{ forward      := fun a => replicate (α := α) (shape := s) a
, backward     := fun _a dLdy => Tensor.scalar (sumSpec (α:=α) dLdy) }

/-- Keep values where the mask is true and set the others to zero, in both passes. -/
def applyMaskOp {s : Shape} (mask : Tensor Bool s) : OpSpec α s s :=
{ forward      := fun x => map2Spec (fun v b => if b then v else 0) x mask
, backward     := fun _x dLdy => map2Spec (fun g b => if b then g else 0) dLdy mask }

/-- Masked inverted-dropout OpSpec with an explicit mask. -/
def dropoutMaskedOp {s : Shape} (p : α) (mask : Tensor Bool s) : OpSpec α s s :=
{ forward      := fun x => dropoutMaskedSpec (α:=α) p mask x
, backward     := fun _x dLdy => dropoutMaskedBackwardSpec (α:=α) p mask dLdy }

/-- Matrix-rank multiplication with a captured right operand and broadcasted batch prefixes. -/
def matmulRightOp {α : Type} [TorchLean.Storage α] [Add α] [Mul α] [Zero α]
    {batchA batchB batch : Shape} {m n p : Nat}
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch)
    (B : Tensor α (batchB.concat [n, p])) :
    OpSpec α (batchA.concat [m, n]) (batch.concat [m, p]) :=
{ forward := fun A => batchMatmulSpec broadcastA broadcastB A B
, backward := fun A dLdz => (matmulBackwardSpec broadcastA broadcastB A B dLdz).1 }

/-- Matrix-rank multiplication with a captured left operand and broadcasted batch prefixes. -/
def matmulLeftOp {α : Type} [TorchLean.Storage α] [Add α] [Mul α] [Zero α]
    {batchA batchB batch : Shape} {m n p : Nat}
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch)
    (A : Tensor α (batchA.concat [m, n])) :
    OpSpec α (batchB.concat [n, p]) (batch.concat [m, p]) :=
{ forward := fun B => batchMatmulSpec broadcastA broadcastB A B
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

/-- Sum along `axis`; the pullback broadcasts the incoming gradient to the input shape. -/
def reduceSumOp {s : Shape} (axis : Nat)
  [_valid : Shape.HasNonemptyAxis axis s]
  [_wf : Shape.WellFormed s]
  :
  OpSpec α s (shapeAfterSum s axis) :=
{ forward      := fun x => reduceSum (α:=α) axis x _valid.proof
, backward     := fun _x dLdz => broadcastAfterSum s axis dLdz
}

/-- Broadcast both operands to `t` and apply `f` pointwise. The pullback multiplies by `dfdx`
and uses the supplied `reduceBack` to accumulate the left input's gradient over broadcast axes. -/
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


end Spec

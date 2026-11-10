import VersoManual
import TorchLean.NN.LoRA
import TorchLean.NN.Autograd.Function
import TorchLean.NN.Autograd.Frontend
import TorchLean.NN.Autograd.Differential
import TorchLean.NN.Autograd.Complex
import TorchLean.NN.CLI.Training.Command
import TorchLean.NN.Data.Synthetic
import TorchLean.NN.Data.CausalLM
import TorchLean.NN.Models.KAN
import TorchLean.NN.Models.CNN
import TorchLean.NN.Models.ResNet
import TorchLean.NN.Models.UNet
import TorchLean.NN.Models.Recurrent
import TorchLean.NN.Models.CausalTransformer.LanguageModel
import TorchLean.NN.Models.Mamba
import TorchLean.NN.Models.MaskedPatchReconstructor
import TorchLean.NN.Models.Diffusion.NoisePredictor
import TorchLean.NN.Models.FNO
import TorchLean.NN.Models.PPO
import TorchLean.NN.Builder.MoE
import TorchLean.NN.Sequential.Training
import TorchLean.NN.RL.CLI
import TorchLean.NN.RL.PPO.ActorCritic
import TorchLean.NN.SelfSupervised
import TorchLean.NN.Text.Vocabulary
import TorchLean.NN.Text.GPT2BPE
import TorchLean.NN.Text.Generation
import TorchLean.NN.Trainer.Constructor
import TorchLean.NN.Trainer.Train
import TorchLean.NN.Data.Image
import TorchLean.Tensor.Layers.Linear
import TorchLean.Tensor.Layers.Activation
import TorchLean.Tensor.Layers.Attention
import TorchLean.Tensor.Layers.Conv
import TorchLean.Tensor.Layers.Dropout
import TorchLean.NN.Models.MLP
import TorchLean.Tensor.Context.Rational
import TorchLeanExamples.DeepDives.Sequential.MLP
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles
import TorchLean.NN.Session.Random
import TorchLean.Verify.Trainer

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean Softmax Dropout Attention Conv
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)
open FloatLib.Numerics (MathFunctions)

#doc (Manual) "Mathematical Specification" =>
%%%
tag := "spec-layer"
file := "The-Mathematical-Specification"
%%%

We can write the model we trained earlier on one line:

$$`F_\theta(x)=W_2\,\operatorname{ReLU}(W_1x+b_1)+b_2.`

Here $`\theta` consists of the four parameter tensors $`W_1`, $`b_1`, $`W_2`, and $`b_2`. The first
affine layer produces the hidden preactivations, ReLU clamps their negative entries to zero, and
the second affine layer produces the output. TorchLean's spec layer expresses this sequence as Lean
definitions, independently of the buffers and kernels used to evaluate it.

A reference Lean loop, a native CPU loop, cuBLAS, and an ATen kernel can all evaluate a linear
layer. They store and combine the values differently, but the specification gives them one
statement to satisfy.

We can run the algebraic part of this specification with exact rational arithmetic. Most of the
spec layer is parametric in the scalar type, and `ℚ` supplies the operations this MLP needs. That
lets us inspect each intermediate value as a fraction before comparing it with floating-point
execution. Each named Lean block below is checked while the page is built, and the Python examples
record the corresponding framework behavior.

# Tensors

TorchLean uses one tensor type for specifications and execution:

```
-- The scalar type and shape are independent indices of the
-- same tensor interface.
Tensor α shape
```

A tensor's runtime representation is one row-major buffer whose length is certified by `shape`.
Through `Storage`, the scalar type chooses the physical buffer: executable scalar types can use
packed native storage, while general scalar types use arrays that may contain boxed values. `Float`
and `UInt8` have packed storage, but that doesn't imply an unboxed buffer for every executable
scalar type.

Proofs don't reason about raw buffer offsets. A tensor coerces to a total function on the typed
coordinate space of its shape, and extensionality says that tensors with equal values at every
coordinate are equal. A vector coordinate contains `Fin n`, so out-of-range lookup can't be
expressed. Shape induction remains available through rank-zero observation and leading-axis
slicing, but those are proof views and smart constructors over the same stored tensor (not a second
recursive physical datatype). Public shapes still use `[]`, `[n]`, and `[m, n]`.

Model code uses ordinary shape lists, and a linear layer is a structure with two fields:

```lean (name := slTypes)
-- Check shaped vectors, matrices, and linear-layer
-- parameter records over exact rationals.
#check Tensor ℚ [2]
#check Tensor ℚ [3, 2]
#check Linear ℚ 2 3
```

```leanOutput slTypes (whitespace := lax)
Tensor ℚ [2] : Type
```

```leanOutput slTypes (whitespace := lax)
Tensor ℚ [3, 2] : Type
```

```leanOutput slTypes (whitespace := lax)
Linear ℚ 2 3 : Type
```

For a linear map from two inputs to three outputs, `Linear ℚ 2 3` contains

$$`W\in\mathbb{Q}^{3\times 2},
\qquad b\in\mathbb{Q}^{3}.`

Output features index the rows of the weight matrix (following PyTorch's `nn.Linear` convention),
so each row holds the coefficients for one output coordinate. Keeping that layout in the
specification lets an imported PyTorch weight matrix retain its interpretation.
{Informal.citet pytorch2019}[] documents that layout.

All three `#check` results describe types, not computed tensors. `Tensor ℚ [2]` fixes two rational
coordinates, while `Tensor ℚ [3, 2]` fixes a matrix with three rows and two columns. And the layer
type packages a matrix and bias with compatible dimensions. These indices stop a caller from
silently passing a tensor of a different shape, but they don't constrain the numerical values
inside it. Positivity, normalization, and closeness to another tensor are separate propositions.

# Scalar Operations And Algebraic Assumptions

A tensor's shape fixes where its values live, and the scalar context fixes which formulas are
available. A linear layer needs zero, addition, and multiplication, and softmax also needs
exponentiation and division. Square root, comparison, trigonometric operations, and Boolean
decisions require their own pieces of structure.

Here the signature separates the storage instance from the three arithmetic operations
`Linear.forward` uses:

```lean (name := slLinearSig)
-- Only the arithmetic used by the linear formula appears in
-- its constraints.
#check @Linear.forward
```

```leanOutput slLinearSig (whitespace := lax)
@Linear.forward : {α : Type} → [inst : Storage α] → [Add α] → [Mul α] → [Zero α] →
  {inDim outDim : ℕ} → Linear α inDim outDim → Tensor α [inDim] → Tensor α [outDim]
```

`Tensor.relu` asks for zero, a maximum operation, and Boolean equality:

```lean (name := slReluSig)
-- The zero comparison fixes the selected ReLU behavior for
-- AD scalar types.
#check @Tensor.relu
```

```leanOutput slReluSig (whitespace := lax)
@Tensor.relu : {α : Type} → [inst : Storage α] → [Zero α] → [Max α] → [BEq α] →
  {s : Shape} → Tensor α s → Tensor α s
```

At zero, ReLU returns the scalar context's zero. For a dual number this also sets its tangent to
zero (matching TorchLean's selected derivative at the kink). Boolean equality selects that branch,
and away from zero the definition uses the maximum operation.

Neither signature asserts algebraic or order laws for those operations. Changing the scalar
instances changes how those operations are evaluated, so the same linear-layer definition can be
used over integers, intervals, or binary32 bit patterns. Algebraic laws about those operations need
their own hypotheses.

We package the common operations in `Context α`, so the same tensor definition can be interpreted
over reals, rounded formats, interval endpoints, or executable binary32. This polymorphism requires
care when stating hypotheses: a proof of a softmax identity over `ℝ` may use properties of the real
exponential that an arbitrary `Context α` doesn't provide.

Which hypotheses we need depends on what we want to prove:

:::table +header
*
  * Level
  * Typical statement
*
  * generic scalar context
  * the operation is well shaped and computes the declared recursion
*
  * algebraic structure
  * addition, multiplication, linearity, or order laws
*
  * real analysis
  * derivatives, continuity, convexity, and analytic bounds
*
  * rounded arithmetic
  * representability, rounding error, and finite-path conditions
*
  * native execution
  * agreement with one selected provider and reduction policy
:::

Requiring a field just to define the linear formula would be unnecessarily strong. A derivative
theorem over the reals can ask for the analytic structure it needs later. Reusing the definition at
another scalar type preserves the operation sequence, while the laws available for reasoning about
it depend on the chosen instances. Native execution also needs a contract relating that sequence to
the provider's implementation.

# Linear Layer Forward And Backward Rules

We kept the forward definition short:

```
-- The definition exposes the matrix product and the
-- subsequent bias addition.
def Linear.forward {α : Type} [Storage α] [Add α] [Mul α] [Zero α]
    {inDim outDim : Nat}
    (m : Linear α inDim outDim)
    (input : Tensor α [inDim]) :
    Tensor α [outDim] :=
  Tensor.contract m.weights input _ _ [outDim] .matVec + m.bias
```

At coordinate $`i`, this means

$$`y_i=\left(\sum_{j=0}^{\mathrm{inDim}-1}W_{ij}x_j\right)+b_i.`

Take $`W=\begin{bmatrix}0.8&-0.4\end{bmatrix}`, $`b=0.2`, and $`x=(0.1,-1)`. By hand,

$$`y_0=0.2+0.8\cdot\tfrac{1}{10}-0.4\cdot(-1)
     =\tfrac15+\tfrac{2}{25}+\tfrac25=\tfrac{17}{25}.`

Over `ℚ` the definition returns that fraction, exactly:

```lean (name := slLinearEval)
-- Evaluate an affine layer exactly, so the fraction reveals
-- the full result.
#eval Linear.forward
  { weights := ([[4/5, -2/5]] : Tensor ℚ [1, 2]),
    bias := ([1/5] : Tensor ℚ [1]) }
  ([1/10, -1] : Tensor ℚ [2])
```

```leanOutput slLinearEval (whitespace := lax)
[(17 : Rat)/25]
```

The backward specification receives an upstream cotangent $`g=\partial L/\partial y` and returns

$$`
\frac{\partial L}{\partial W}=g\,x^\mathsf{T},\qquad
\frac{\partial L}{\partial b}=g,\qquad
\frac{\partial L}{\partial x}=W^\mathsf{T}g.
`

In Lean these three tensors have shapes `[outDim, inDim]`, `[outDim]`, and `[inDim]`. So a
transposed weight gradient has the wrong type when the input and output dimensions differ, but for
a square weight matrix the shape alone can't detect the transpose.

`Linear.backward` returns all three in a `Linear.Gradients` record: the weight and bias gradients as
a `Linear` in its `parameters` field, beside the input gradient. The gradient of a parameter has the
parameter's shape, so the layer's own record holds it, and the named fields keep the bias and input
gradients apart even in a square layer, where their shapes agree. Convolution uses `Conv.Gradients`
for the same reason. For the second layer of our MLP, take a hidden activation of $`(\tfrac1{10},0)`
and a cotangent of $`1`:

```lean (name := slBackward)
-- Seed the scalar output with one and inspect parameter and
-- input cotangents separately.
def slSecond : Linear ℚ 2 1 :=
  { weights := [[4/5, -2/5]], bias := [1/5] }

#eval Linear.backward slSecond
  ([1/10, 0] : Tensor ℚ [2]) ([1] : Tensor ℚ [1])
```

```leanOutput slBackward (whitespace := lax)
{ parameters := { weights := [[(1 : Rat)/10, 0]], bias := [1] },
  inputGradient := [(4 : Rat)/5, (-2 : Rat)/5] }
```

Check each gradient against the formula above. The weight gradient $`g\,x^\mathsf{T}` is the hidden
activation written as a row. Since $`\partial y/\partial b` is the identity, the bias gradient is
$`g` unchanged. And the input gradient $`W^\mathsf{T}g` is the weight row transposed. All three are
exact rationals that can be compared directly with the three expressions.

You'll find the definitions `Linear.forward`, `Linear.weightsDeriv`, `Linear.inputDeriv`, and
`Linear.backward` in
{src "TorchLean/Tensor/Layers/Linear.lean"}[`TorchLean/Tensor/Layers/Linear.lean`].
Later, the autograd proofs compare executable VJP rules with these definitions, and
{ref "autograd-proofs"}[the autograd proofs chapter] states the agreement theorems and their
assumptions.

We chose seed one to inspect the derivative of this scalar output. With seed two, every returned
cotangent would double over the rationals. In a composed model, an earlier layer receives the
cotangent produced by a later layer, so it can't assume that its own output is seeded with one.

# MLP Forward Semantics

Our running model composes two linear specifications and a pointwise activation:

```lean (name := slMlpSig)
-- The model composes the same layer specifications used by
-- the individual examples.
#check @MLP.forward
```

```leanOutput slMlpSig (whitespace := lax)
@MLP.forward : {α : Type} → [inst : Storage α] → [Context α] → {inDim hidDim outDim : ℕ} →
  Linear α inDim hidDim → Linear α hidDim outDim → Tensor α [inDim] → Tensor α [outDim]
```

`MLP.forward` runs it on one input vector. Give it the weights from the formula at
the top of the chapter,

$$`
W_1=\begin{bmatrix}1&1\\-1&1\end{bmatrix},
\quad b_1=0,\quad
W_2=\begin{bmatrix}0.8&-0.4\end{bmatrix},
\quad b_2=0.2,
`

and evaluate it at $`x=(0.25,-0.75)`:

```lean (name := slMlp)
-- Choose rational parameters whose hidden preactivations
-- can be inspected exactly.
def slFirst : Linear ℚ 2 2 :=
  { weights := [[1, 1], [-1, 1]], bias := [0, 0] }

def slPoint : Tensor ℚ [2] := [1/4, -3/4]

open scoped RationalAlgebraic in
#eval MLP.forward slFirst slSecond slPoint
```

```leanOutput slMlp (whitespace := lax)
[(1 : Rat)/5]
```

We can also evaluate the three stages separately:

```lean (name := slStages)
-- Display the linear result, the selected ReLU branch, and
-- the resulting model output.
#eval Linear.forward slFirst slPoint
#eval Tensor.relu (Linear.forward slFirst slPoint)
#eval Linear.forward slSecond
  (Tensor.relu (Linear.forward slFirst slPoint))
```

```leanOutput slStages (whitespace := lax)
[(-1 : Rat)/2, -1]
```

```leanOutput slStages (whitespace := lax)
[0, 0]
```

```leanOutput slStages (whitespace := lax)
[(1 : Rat)/5]
```

So $`W_1x=(-\tfrac12,-1)`, both preactivations are negative, ReLU sends the whole hidden layer to
zero, and the output is the second bias alone: exactly $`\tfrac15`. As an `nn.Sequential`, the same
model is `Examples.Sequential.MLP.model`, and `Examples.Sequential.MLP.eval_model` proves that it
evaluates to `MLP.forward` at every scalar.

These rational evaluations also predict which parameter changes can affect the output. As long as
both hidden preactivations remain negative, changing the first layer or the second weight matrix
leaves the output unchanged. Crossing an activation boundary opens a path through the hidden layer
(the bias perturbation below shows this).

## The selected backward rule at zero

ReLU's definition commits to two choices:

$$`\operatorname{ReLU}(z)=\max(z,0),`

and for the selected derivative,

$$`\operatorname{ReLU}'(z)=
  \begin{cases}
    1 & z>0,\\
    0 & z\le 0.
  \end{cases}`

What we pick at the kink matters, because other subgradients are mathematically defensible:
anything in $`[0,1]` is a valid subgradient of $`\max(z,0)` at zero. A forward and backward
correctness theorem needs a single concrete rule, and TorchLean chooses zero at the kink.

Choosing zero agrees with TorchLean's runtime path and with PyTorch:

```
>>> # Inspect the gradient convention chosen at the ReLU
>>> # kink.
>>> x = torch.tensor(0.0, requires_grad=True)
>>> y = torch.relu(x); y.backward()
>>> x.grad.item()
0.0
```

Choosing $`\tfrac12` would make the selected backward rule disagree with the eager ReLU rule at
zero. Forward values wouldn't change (and a surrounding model might mask the gradient difference),
but the local backward contract would still need to record the discrepancy.

## Difference Quotients

Exact arithmetic lets us separate branch behavior from rounding in a derivative check. A
finite-difference check in floating point has to pick a step size and live with the error curve
that comes with it (as {ref "running-example"}[the running example] shows in detail). Over `ℚ` the
second layer is affine, so the step cancels algebraically and the quotient is the derivative for
every nonzero step:

```lean (name := slQuot)
-- Affine difference quotients are exact at every nonzero
-- step used here.
/-- Forward difference quotients of the second layer in
both input coordinates, at step `h`, over `ℚ`. -/
def slQuot (h : ℚ) : ℚ × ℚ :=
  let y := (Linear.forward slSecond [1/10, 0])[0]
  let y₀ := (Linear.forward slSecond [1/10 + h, 0])[0]
  let y₁ := (Linear.forward slSecond [1/10, h])[0]
  ((y₀ - y) / h, (y₁ - y) / h)

#eval (slQuot (1/1000), slQuot (10^6))
#eval slQuot (1/1000) == slQuot (-10^6)
```

```leanOutput slQuot (whitespace := lax)
((4 / 5, -2 / 5), 4 / 5, -2 / 5)
```

```leanOutput slQuot (whitespace := lax)
true
```

Both coordinates return the weight row $`(\tfrac45,-\tfrac25)` for the small positive step and
the large steps of either sign. Here $`e_j` selects input coordinate $`j`. The identity
$`W(x+he_j)+b-(Wx+b)=hW e_j` explains the result: subtracting the original output cancels the
bias and the unchanged input coordinates, and division by nonzero $`h` leaves column $`j` of
$`W`. At the ReLU kink, the two signs of the step select different branches:

```lean (name := slKink)
-- Opposite sides of the kink give different quotients, so
-- no two-sided derivative exists.
def slReluQuot (h : ℚ) : ℚ :=
  ((Tensor.relu ([h] : Tensor ℚ [1]))[0]
    - (Tensor.relu ([0] : Tensor ℚ [1]))[0]) / h

#eval (slReluQuot (1/1000), slReluQuot (-1/1000))
```

```leanOutput slKink (whitespace := lax)
(1, 0)
```

For every positive step the quotient is one, and for every negative step it's zero. That
disagreement persists as the step approaches zero, so ReLU has no derivative there. TorchLean's
backward rule selects the subgradient `0` from $`[0,1]`. Exact arithmetic makes the source of the
disagreement visible: it comes from the two branches of ReLU.

# Runtime And Specification Comparison

We can evaluate the same inputs at `Float`. `Tensor.linear` is the runtime operation and
`Linear.forward` is its pure specification. Here we compare their results for this input:

```lean (name := slRuntime)
-- Compare the runtime linear route with the same formula at
-- scalar type Float.
def slWeight : Tensor Float [1, 2] := [[0.8, -0.4]]
def slBias : Tensor Float [1] := [0.2]
def slInput : Tensor Float [2] := [0.1, -1.0]

#eval Tensor.linear slInput slWeight slBias
#eval Linear.forward
  { weights := slWeight, bias := slBias } slInput
#eval (Tensor.linear slInput slWeight slBias)[0] ==
  (Linear.forward
    { weights := slWeight, bias := slBias } slInput)[0]
```

```leanOutput slRuntime (whitespace := lax)
[0.680000]
```

```leanOutput slRuntime (whitespace := lax)
[0.680000]
```

```leanOutput slRuntime (whitespace := lax)
true
```

Both routes agree here, and PyTorch agrees with both:

```
>>> # Match dtype and parameters before comparing the affine
>>> # result.
>>> import torch
>>> import torch.nn.functional as F
>>> from torch import nn
>>> F.linear(torch.tensor([0.1, -1.0], dtype=torch.float64),
...          torch.tensor([[0.8, -0.4]], dtype=torch.float64),
...          torch.tensor([0.2], dtype=torch.float64)).tolist()
[0.68]
```

So far we've compared a single vector input. But `Tensor.linear` also applies along the final axis
of a tensor with leading dimensions, so a batch of rows comes back as a batch of rows:

```lean (name := slBatch)
-- Apply the shared affine map independently to each row of
-- a batch.
def slRows : Tensor Float [3, 2] :=
  [[0.1, -1.0], [0.0, 0.0], [1.0, 1.0]]

def slProjected := Tensor.linear slRows slWeight slBias

#eval slProjected
```

```leanOutput slBatch (whitespace := lax)
[[0.680000], [0.200000], [0.600000]]
```

PyTorch, same call, same three rows:

```
>>> # Use binary64 here to match Lean Float in the preceding
>>> # calculation.
>>> W = torch.tensor([[0.8, -0.4]], dtype=torch.float64)
>>> b = torch.tensor([0.2], dtype=torch.float64)
>>> F.linear(torch.tensor([[0.1, -1.], [0., 0.], [1., 1.]], dtype=torch.float64),
...          W, b).tolist()
[[0.68], [0.2], [0.6000000000000001]]
```

Look closer at the third row. PyTorch prints `0.6000000000000001` while TorchLean prints
`0.600000`, and comparing the stored value with the literal `0.6` reveals a difference that
TorchLean's printer hides:

```lean (name := slNotSix)
-- A short decimal display can hide a nonzero difference
-- between stored values.
#eval slProjected[2][0] == 0.6
#eval slProjected[2][0] - 0.6
```

```leanOutput slNotSix (whitespace := lax)
false
```

```leanOutput slNotSix (whitespace := lax)
0.000000
```

So the equality check establishes that TorchLean's stored result differs from the literal $`0.6`,
although the subtraction still prints as `0.000000`. PyTorch's longer decimal output exposes a
difference that six decimal places hide. These displays alone don't establish that the two
libraries stored identical bit patterns (that would take a bit-level comparison). Even this one-row
affine calculation shows why formatted output is insufficient for an exact numerical comparison.
{Informal.citet goldberg1991}[] is the standard account, and {ref "runtime-approximation"}[the
runtime approximation chapter] states what TorchLean actually proves about the gap.

Our tensor examples above evaluate forward operations and don't construct an autograd tape. In
model execution, the `nn.linear` builder fixes the parameter shapes and emits the runtime linear
operation, while the graph interpreter assigns `.linear` its `Linear.forward` denotation. And the
VJP proof relates the selected backward rule to `Linear.backward`. These connections use the same
parameter layout and backward convention.

Native execution adds another boundary: a capsule records the provider selected at planning time
and its declared obligations, and execution checks establish which provider actually ran. A
capsule's evidence determines what is known about that provider's agreement with the specification.

Batch axes play a separate role from the feature axis: `Tensor.linear` applies the affine map
independently to each row. It introduces no reduction across rows, so the first row's value doesn't
depend on the second row's features.

# Bias Perturbation And ReLU

Change only the first bias, from $`0` to $`0.6`. That moves the first hidden preactivation from
$`-\tfrac12` to $`\tfrac1{10}`, so it survives ReLU, and the output becomes

$$`0.8\cdot0.1+0.2=0.28=\tfrac{7}{25}.`

Both claims check out:

```lean (name := slPerturb)
-- Move one hidden preactivation across zero to expose the
-- newly active path.
def slShifted : Linear ℚ 2 2 :=
  { slFirst with bias := ([3/5, 0] : Tensor ℚ [2]) }

#eval Linear.forward slShifted slPoint
#eval Linear.forward slSecond
  (Tensor.relu (Linear.forward slShifted slPoint))
```

```leanOutput slPerturb (whitespace := lax)
[(1 : Rat)/10, -1]
```

```leanOutput slPerturb (whitespace := lax)
[(7 : Rat)/25]
```

Interval propagation extends this calculation from one point to a set of possible inputs, bounding
each intermediate tensor. What matters here is the sign change: the first hidden unit contributed
zero before the bias changed, then contributed $`1/10` afterwards. For an input region, we must
determine which preactivation intervals stay on one side of zero and which can cross it. That's
part of the analysis that {ref "verification"}[the verification chapter] formalizes.

Within either fixed activation region the model is affine, and an analytic proof can use that
formula once it has established the branch conditions. At the boundary, though, the proof must
distinguish the selected backward rule from ordinary differentiability.

# Scalar Polymorphism

The type parameter $`\alpha` determines what the symbols `+`, `*`, `max`, `exp`, and division mean.
TorchLean reuses the tensor structure at several scalar interpretations:

:::table +header
*
  * Scalar
  * Meaning
*
  * `ℝ`
  * exact real arithmetic used for mathematical statements
*
  * `ℚ`
  * exact rational arithmetic, executable, algebraic fragments only
*
  * `FP32`
  * binary32-precision, gradual-underflow rounded-real values with no upper exponent cutoff
*
  * FloatLib binary32
  * executable binary32 bit patterns, including signed zero, infinity, and NaN
*
  * runtime `Float`
  * Lean's native executable floating-point value
:::

`Context ℚ` is a *scoped* instance in
{src "TorchLean/Tensor/Context/Rational.lean"}[`Context/Rational.lean`], which is why the
MLP evaluation above had to write `open scoped RationalAlgebraic` to get at it. Since the real
exponential doesn't return a rational in general, this algebraic context supplies zero placeholders
for the transcendental operations:

```lean (name := slQExp)
-- The scoped rational context supplies placeholders for
-- transcendental operations.
open scoped RationalAlgebraic in
#eval (MathFunctions.exp (1 : ℚ), MathFunctions.log (1 : ℚ))
```

```leanOutput slQExp (whitespace := lax)
(0, 0)
```

Evaluating softmax with these placeholder transcendental operations produces the result below. For
`Tensor.softmax`, the second argument is the axis (a `Fin` of the tensor's rank), so
`⟨0, Nat.one_pos⟩` is axis `0` and an axis out of range doesn't typecheck:

```lean (name := slQSoftmax)
-- A well-typed call can still lack the intended analytic
-- interpretation.
open scoped RationalAlgebraic in
#eval Tensor.softmax ([0, 1, 2] : Tensor ℚ [3])
  ⟨0, Nat.one_pos⟩
```

```leanOutput slQSoftmax (whitespace := lax)
[0, 0, 0]
```

Here the entries sum to zero, so this result can't be interpreted as a probability distribution.
The tensor calculation typechecks, but the real-exponential laws used to justify softmax don't hold
for these placeholder operations. Because the instance is scoped, a file has to opt in to the
rational algebraic interpretation, but callers still need to restrict its use to the supported
algebraic fragment.

With the real exponential, softmax has the formula

$$`\operatorname{softmax}(x)_i=
\frac{\exp(x_i-m)}{\sum_j\exp(x_j-m)},\qquad m=\max_jx_j`

For finite binary32 inputs, `max` selects an existing value without rounding. Subtraction,
exponential approximation, summation, and division can introduce error (and a CUDA reduction may
also choose a different summation tree). Both {ref "floats"}[the floating-point chapter] and
{ref "runtime-approximation"}[the runtime-approximation chapter] state the conditions under which
one interpretation encloses or approximates another. In Rocq, {Informal.citet flocq2011}[] and
{Informal.citet boldo2015}[] develop related formal accounts of rounding and numerical error. An
executable softmax needs a relation between its scalar operations and the real operations in this
formula.

`Tensor.softmax x dim mask schedule` also takes a `Softmax.Schedule`, the order in which a fiber is
traversed. Its default `.twoPass` finds the maximum and the denominator before the weights, as the
formula reads. `.online blocks` makes one pass over blocks of entries, keeps a running maximum, and
multiplies the partial denominator by $`\exp(m-m')` whenever a block raises the running maximum
from $`m` to $`m'`. That's FlashAttention's schedule. Over the reals the two schedules are the same
function (`Tensor.softmax_online_eq`), but in floating point they round differently, and
{ref "attention-roundoff"}[Rounding Error In Attention] bounds the online one.

## Summation Order

To see how the reduction order enters, keep the four inputs fixed and change only the grouping.
Left-to-right and paired addition agree whenever addition is associative. Two definitions let us
compare both orders over `ℚ` and `Float`:

```lean (name := slSumOrder)
-- Keep the association explicit when comparing exact and
-- rounded scalar arithmetic.
def slLeft (α : Type) [Add α] (a b c d : α) : α :=
  a + b + c + d

def slPaired (α : Type) [Add α] (a b c d : α) : α :=
  (a + b) + (c + d)

#eval (slLeft ℚ (10^16) 1 1 1,
  slPaired ℚ (10^16) 1 1 1)
#eval (slLeft Float 1e16 1.0 1.0 1.0,
  slPaired Float 1e16 1.0 1.0 1.0)
```

```leanOutput slSumOrder (whitespace := lax)
(10000000000000003, 10000000000000003)
```

```leanOutput slSumOrder (whitespace := lax)
(10000000000000000.000000,
 10000000000000002.000000)
```

Over `ℚ` the two orders agree, as they must. In `Float` they differ by two, because the spacing of
representable numbers near $`10^{16}` is `2`: adding `1` to $`10^{16}` rounds back to $`10^{16}`
three times in the left-to-right order, while the paired order adds `1 + 1` first and gets a step
it can represent. Both results follow from rounding each addition in its specified order. A
specification stated over `ℚ` or `ℝ` doesn't determine which of these two numbers a kernel returns.
So a claim about a fused reduction, a multi-threaded partial-sum tree, or a tensor-core
accumulation needs a rounding-aware statement (an appeal to the formula isn't enough). We make
those statements in {ref "runtime-approximation"}[the runtime approximation chapter], and Higham's
*Accuracy and Stability of Numerical Algorithms* is the standard reference for how summation error
depends on the order chosen.

# The Implemented Specifications

TorchLean's spec layer covers more than the running MLP.
{srcDir "TorchLean/Tensor/Layers"}[`TorchLean/Tensor/Layers`] contains typed meanings for
linear algebra, rank-polymorphic convolution, transposed convolution, and pooling, activations and
losses, normalization, dropout, embeddings, recurrent cells, selective scan, and scaled dot-product
attention. Model definitions under {srcDir "TorchLean/NN/Models"}[`TorchLean/NN/Models`]
compose these operations into families such as CNNs, transformers, recurrent networks, and
state-space models.

An operation can have a pure definition before it has an eager tape rule, typed graph lowering, a
CUDA kernel, or an end-to-end correctness theorem. Each route's supported operations are listed in
the runtime and lowering chapters.

The operators that the runtimes record are the constructors of `TorchLean.Op` in
{src "TorchLean/Tensor/Op/Table.lean"}[the operator table], one per registered operator with
its attributes, indexed by its input and output ports. `Op.rule?` gives each one its rule at a
scalar type, built from the definitions of this layer: the forward map, the pullback, and the node
a recorder prepares. TorchLean's CPU tape records an operator through this rule
(`Tape.apply`), and the CUDA path and the operator graph `IR.Graph` name
operators of the same table. An operator whose scalar lacks a capability (such as the error
function on `Float`) has no rule at that scalar.

# Shapes And Element Types

A tensor's shape index $`s` and element type $`\alpha` answer different questions. `Tensor α s`
fixes both for one value, and `IR.Graph.eval` follows the homogeneous-element contract from
*Tensors And Shapes*: it evaluates a graph at one scalar type `α`. A value, frozen or detached port
holds a tensor over `α`, while an index, Boolean or integer port holds a tensor of index tuples,
Booleans or fixed-width words (`TorchLean.Port.Carrier`). In the IR chapter, “heterogeneous” refers
only to a table of values at different ports, not to a graph over several scalar types.

# Specification Conventions

Shapes are only one source of ambiguity, and the spec layer also fixes choices that a model name
doesn't determine. Each subsection below is a place where two reasonable libraries could differ, so
the definition has to say which one TorchLean means.

## Loss reductions

`Tensor.meanSquaredError` takes a global mean over all entries. `Tensor.crossEntropy .logits`
applies log-softmax
along the selected class axis and, with `mean := true`, divides by the number of remaining slices.
Summing instead changes both the loss and every gradient by a scale factor.

Between our output $`\tfrac{17}{25}` and a target of $`\tfrac15`, the mean-squared error is
$`\left(\tfrac{12}{25}\right)^2=\tfrac{144}{625}`, and again the fraction is the answer:

```lean (name := slMse)
-- For one prediction, mean squared error is simply the
-- squared residual. The mean divides through
-- `Context.divNat`, so ℚ takes the scoped rational context.
open scoped RationalAlgebraic in
#eval Tensor.meanSquaredError ([17/25] : Tensor ℚ [1])
  ([1/5] : Tensor ℚ [1])
```

```leanOutput slMse (whitespace := lax)
144 / 625
```

Because the objective is a mean, adding a fifth example to a four-example batch rescales the
contributions from the existing four. A sum-reduced loss would not rescale those contributions.
Neither convention is wrong, and a learning rate
tuned under one may need rescaling under the other, which is why the reduction lives in the
definition and not
in a training script.

Our one-coordinate example can't distinguish mean from sum (dividing by the number of entries
divides by one). A comparison intended to check the reduction needs several entries, and the
specification must identify which axes participate.

## Attention masks

A boolean attention mask is a hard support constraint:

- `true` allows the key
- `false` gives the key exactly zero softmax numerator
- if every key in a row is blocked, the output row is zero

This is the finite formulation of a negative-infinity mask. `causalMask n` is the standard lower
triangle from {Informal.citet transformer2017}[]:

```lean (name := slCausal)
-- Inspect both the Boolean causal support and the
-- normalized rows it permits.
#eval causalMask 3
#eval hardMasked
  ([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]] :
    Tensor Float [3, 3])
  (causalMask 3)
```

```leanOutput slCausal (whitespace := lax)
[[true, false, false], [true, true, false], [true, true, true]]
```

```leanOutput slCausal (whitespace := lax)
[[1.000000, 0.000000, 0.000000],
 [0.268941, 0.731059, 0.000000],
 [0.090031, 0.244728, 0.665241]]
```

Read the rows as the causal structure they encode. Row 0 may attend only to itself, so it gets all
the mass. Row 1 sees positions 0 and 1 with scores $`0` and $`1`, and
$`e/(1+e)\approx0.731059` is exactly the logistic function at $`1` . Row 2 sees all three. Every
blocked
entry is a hard zero, not a small number. PyTorch's `masked_fill` with $`-\infty` agrees at the
six-decimal precision displayed above:

```
>>> # Apply the same causal support to the comparison
>>> # tensor.
>>> scores = torch.tensor([[0., 1., 2.]] * 3, dtype=torch.float64)
>>> causal = torch.ones(3, 3, dtype=torch.bool).tril()
>>> F.softmax(scores.masked_fill(~causal, float('-inf')), dim=-1).tolist()
[[1.0, 0.0, 0.0],
 [0.26894142136999516, 0.7310585786300049, 0.0],
 [0.09003057317038045, 0.2447284710547976, 0.6652409557748218]]
```

A finite additive bias can leave mass on a blocked key. Take two positions with scores $`0` and
$`1000`, block the second, and compare a hard mask with adding $`-1000` to its score:

```lean (name := slLeak)
-- Hard exclusion and a finite negative bias define
-- different forward functions.
#eval hardMaskedVec
  ([0.0, 1000.0] : Tensor Float [2])
  ([true, false] : Tensor Bool [2])
#eval Tensor.softmax
  (Tensor.add ([0.0, 1000.0] : Tensor Float [2])
    ([0.0, -1000.0] : Tensor Float [2])) ⟨0, Nat.one_pos⟩
```

```leanOutput slLeak (whitespace := lax)
[1.000000, 0.000000]
```

```leanOutput slLeak (whitespace := lax)
[0.500000, 0.500000]
```

So the sentinel gave the blocked position half the attention mass. Push the scores a little further
and the blocked position takes essentially all of it:

```lean (name := slLeak2)
-- Increase the blocked score past the finite penalty to
-- expose the leaked probability.
#eval Tensor.softmax
  (Tensor.add ([1.0, 1200.0] : Tensor Float [2])
    ([0.0, -1000.0] : Tensor Float [2])) ⟨0, Nat.one_pos⟩
#eval hardMaskedVec
  ([1.0, 1200.0] : Tensor Float [2])
  ([true, false] : Tensor Bool [2])
```

```leanOutput slLeak2 (whitespace := lax)
[0.000000, 1.000000]
```

```leanOutput slLeak2 (whitespace := lax)
[1.000000, 0.000000]
```

In this two-key example, a $`-1000` bias leaves the blocked score above the allowed score, so
attention concentrates on the blocked key. What matters is the score difference, not whether either
unshifted score exceeds $`1000`. PyTorch reproduces the same failure, because it's a property of
the arithmetic, not of either library:

```
>>> # The comparison uses the same two score choices as the
>>> # Lean example.
>>> F.softmax(torch.tensor([0., 1000.]) + torch.tensor([0., -1000.]), dim=-1)
tensor([0.5000, 0.5000])
>>> F.softmax(torch.tensor([1., 1200.], dtype=torch.float64)
...           + torch.tensor([0., -1000.], dtype=torch.float64), dim=-1)
tensor([3.7618e-87, 1.0000e+00], dtype=torch.float64)
```

`hardMasked` computes the row maximum and the denominator over the allowed entries only,
so a blocked numerator is exactly zero. With real exponential and at least one allowed entry, the
denominator is positive: the maximum contributes $`\exp(0)=1`. This argument needs laws for the
scalar operations, which the generic `Context` interface alone doesn't provide. Floating-point
execution also needs a contract for exponentiation and non-finite inputs. We define a fully blocked
row to be zero, not $`0/0`, matching PyTorch's scaled dot-product attention and TorchLean's CUDA
providers:

```lean (name := slAllBlocked)
-- The declared all-blocked convention returns zero weights
-- rather than dividing by zero.
#eval hardMaskedVec
  ([0.0, 1000.0] : Tensor Float [2])
  ([false, false] : Tensor Bool [2])
```

```leanOutput slAllBlocked (whitespace := lax)
[0.000000, 0.000000]
```

Tiled attention algorithms such as {Informal.citet flashattention2022}[] also need to preserve the
mask support while accumulating the softmax denominator. Excluding a key and approximating its
exclusion with a finite penalty remain different operations. In `Tensor.softmax` and
`Tensor.attention`, the online schedule folds over the allowed keys only, so a blocked key enters
neither the running maximum nor the denominator. And a row without an allowed key is zero under
both schedules.

Here the mask is fixed data describing which coordinates may interact. A derivative theorem with
respect to scores differentiates the function with that mask held fixed. It doesn't define a
derivative with respect to a Boolean choice, and a theorem about a finite additive score bias
describes another function. You need these distinctions when comparing an attention proof with the
mask convention of an execution backend.

## Dropout

Randomness is explicit. A masked dropout specification receives the mask as an argument. Runtime
training code may generate that mask from a seed and tape state, but the semantic function does not
consult hidden global randomness.

Ordinary inverted dropout scales the kept entries by $`1/(1-p)` {Informal.citet dropout2014}[].
TorchLean divides each kept entry by $`1-p` once, so a kept entry is the correctly rounded
quotient of $`x` by the rounded $`1-p`, and a dropped entry is $`x\cdot 0`. At $`p\ge 1` every
entry is $`+0`. `Op.dropout` keeps an entry iff its uniform sample $`u` satisfies $`u\ge p`. At
$`p=\tfrac12` the kept entries are doubled:

```lean (name := slDropout)
-- A fixed mask selects the coordinates that are kept and
-- divided by 1 - p.
open scoped RationalAlgebraic in
#eval masked ((1 : ℚ)/2)
  ([true, false, true, false] : Tensor Bool [4])
  ([1, 2, 3, 4] : Tensor ℚ [4])
```

```leanOutput slDropout (whitespace := lax)
[2, 0, 6, 0]
```

If each mask entry is Bernoulli with keep probability $`1/2`, this preserves each entry in
expectation over that distribution. A dropped entry is the product $`x\cdot 0`, not a literal
zero, so a negative entry gives $`-0` and an infinite or NaN entry gives NaN, as in PyTorch:

```lean (name := slDropoutSigned)
-- Dropped entries are x * 0 in IEEE arithmetic.
#eval masked (0.5 : Float)
  ([false, false, true] : Tensor Bool [3])
  ([-1.0, 1.0 / 0.0, 3.0] : Tensor Float [3])
```

```leanOutput slDropoutSigned (whitespace := lax)
[-0.000000, NaN, 6.000000]
```

Evaluation mode is the identity on values: in `Mode.eval` the dropout layer returns its input, as
PyTorch does:

```
>>> # Use inference mode explicitly when checking the
>>> # identity behavior.
>>> x = torch.tensor([1., 2., 3., 4.])
>>> F.dropout(x, p=0.5, training=False).tolist()
[1.0, 2.0, 3.0, 4.0]
```

An expectation theorem needs a distributional assumption on the mask, and a stateful specification
must also include the generator state in its contract.

## Invalid windows

Some mathematical operations are total where the runtime is partial. For unit dilation, input
extent $`n`, positive kernel extent $`k`, symmetric padding $`p`, and positive stride $`s`, the
usual
convolution output extent, when $`k \le n+2p`, is

$$`\left\lfloor\frac{n+2p-k}{s}\right\rfloor+1.`

`Shape.slidingWindowOutDimDilated` computes it. Its answers match PyTorch on the ordinary cases:

```lean (name := slConv)
-- Compare spatial output lengths under different strides
-- and then a full shape.
#eval Shape.slidingWindowOutDimDilated 28 3 1 1 1 1
#eval Shape.slidingWindowOutDimDilated 28 3 2 1 1 1
#eval Tensor.to (outSpatialDilated
  ([28, 28] : Tensor Nat [2])
  ([5, 5] : Tensor Nat [2]) ([2, 2] : Tensor Nat [2])
  ([1, 1] : Tensor Nat [2]) ([0, 0] : Tensor Nat [2])
  ([0, 0] : Tensor Nat [2])) (List Nat)
```

```leanOutput slConv (whitespace := lax)
28
```

```leanOutput slConv (whitespace := lax)
14
```

```leanOutput slConv (whitespace := lax)
[12, 12]
```

```
>>> # Check the corresponding convolution shapes in the
>>> # comparison API.
>>> nn.Conv2d(1, 1, 3, stride=1, padding=1)(torch.zeros(1, 1, 28, 28)).shape
torch.Size([1, 1, 28, 28])
>>> nn.Conv2d(1, 1, 5, stride=2, padding=0)(torch.zeros(1, 1, 28, 28)).shape
torch.Size([1, 1, 12, 12])
```

For a zero stride or a kernel wider than the padded input, the usual window calculation is invalid.
Even so, the shape helper returns a `Nat`:

```lean (name := slConvBad)
-- Total shape arithmetic returns zero for these invalid or
-- empty windows.
#eval Shape.slidingWindowOutDimDilated 28 3 0 1 1 1
#eval Shape.slidingWindowOutDimDilated 3 5 1 1 0 0
```

```leanOutput slConvBad (whitespace := lax)
0
```

```leanOutput slConvBad (whitespace := lax)
0
```

PyTorch rejects both, at different moments:

```
>>> # The execution API may reject a configuration whose
>>> # total shape formula returns zero.
>>> nn.Conv2d(1, 1, 5, stride=1, padding=0)(torch.zeros(1, 1, 3, 3))
RuntimeError: Calculated padded input size per channel: (3 x 3). Kernel size: (5 x 5).
Kernel size can't be greater than actual input size
>>> nn.Conv2d(1, 1, 3, stride=0, padding=1)(torch.zeros(1, 1, 8, 8))
RuntimeError: non-positive stride is not supported
```

`Shape.slidingWindowOutDimDilated` is a total function used inside type indices, so it returns a
value for every configuration (here `0` represents no output positions). Runtime validation
separately determines whether the configuration may be executed and rejects these cases before
native code is reached. So a computed shape can be used in a type without implying that every
configuration producing it is admitted by the runtime.

When you apply a convolution theorem, check its well-formedness and positive-stride hypotheses
alongside the computed output length. Those hypotheses identify the configurations the theorem
applies to.

# MLP Specification Equivalence

`Examples.Sequential.MLP.model`, the sequential MLP, uses the same four parameter tensors:

```
-- The ordered parameter roles matter even when two entries
-- have equal shapes.
[W₁ : [hidden, input],
 b₁ : [hidden],
 W₂ : [output, hidden],
 b₂ : [output]]
```

`Examples.Sequential.MLP.eval_model` proves that evaluating that model under the pure interpreter is
equal to the hand-written two-layer specification, for every two linear layers supplying the
parameter pack and every input, at every scalar type with `Storage` and `Context` instances:

```lean (name := slAxioms)
-- Inspect dependencies of the pure model-equivalence
-- theorem, not a native execution claim.
#print axioms
  Examples.Sequential.MLP.eval_model
```

```leanOutput slAxioms (whitespace := lax)
'TorchLean.Examples.Sequential.MLP.eval_model' depends on axioms:
[propext, Classical.choice, Quot.sound]
```

Those three axioms are Lean's standard classical foundations, also used by
{Informal.citet mathlib2020}[]. So the theorem adds no TorchLean-specific axiom or unproved
placeholder.

Its conclusion identifies the pure interpretation of the sequential model with
$`W_2\operatorname{ReLU}(W_1x+b_1)+b_2`. So a bound on that interpretation can be read as a bound
on this formula, with the same parameters and input. Connecting this result to an imported
checkpoint, an eager tape, or a CUDA execution requires the corresponding import and runtime
arguments.

# Sources

The definitions above, in the order they appear:

- {src "TorchLean/Tensor/Layers/Linear.lean"}[`Layers/Linear.lean`], `Linear` and the four
  forward and backward definitions
- {src "TorchLean/Tensor/Layers/Activation.lean"}[`Layers/Activation.lean`], `Tensor.relu`
- {src "TorchLean/Tensor/Layers/Softmax.lean"}[`Layers/Softmax.lean`], `Tensor.softmax` and
  `Tensor.logSoftmax` along any axis, the `Softmax.Schedule` that softmax takes, `hardMaskedMax?`,
  and `hardMasked`
- {src "TorchLean/Tensor/Layers/Attention.lean"}[`Layers/Attention.lean`], `causalMask`
- {src "TorchLean/Tensor/Layers/Dropout.lean"}[`Layers/Dropout.lean`], `Dropout.masked`
- {src "TorchLean/Tensor/Layers/Conv.lean"}[`Layers/Conv.lean`], the window geometry
- {src "TorchLean/NN/Models/MLP.lean"}[`Models/MLP.lean`], `MLP.forward`
- {src "TorchLean/Tensor/Context/Rational.lean"}[`Context/Rational.lean`], the
  scoped rational backend
- {src "TorchLeanExamples/DeepDives/Sequential/MLP.lean"}[`DeepDives/Sequential/MLP.lean`], the
  sequential MLP and its evaluation theorem

{ref "sequential"}[The sequential models chapter] turns these formulas into typed architectures
with explicit parameter layouts.

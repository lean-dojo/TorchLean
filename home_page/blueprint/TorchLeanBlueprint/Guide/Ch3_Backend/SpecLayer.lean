import VersoManual
import NN.API
import NN.Spec.Layers.Linear
import NN.Spec.Layers.Activation
import NN.Spec.Layers.Attention
import NN.Spec.Layers.Conv
import NN.Spec.Layers.Dropout
import NN.Spec.Models.Mlp
import NN.Spec.Core.Context.Rational
import NN.GraphSpec.Models.MlpSpecEquivalence
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean Spec
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Mathematical Specification" =>
%%%
tag := "spec-layer"
file := "The-Mathematical-Specification"
%%%

The model we trained earlier can be written on one line:

$$`F_\theta(x)=W_2\,\operatorname{ReLU}(W_1x+b_1)+b_2.`

Here $`\theta` consists of the four parameter tensors $`W_1`, $`b_1`, $`W_2`, and $`b_2`.
The first affine layer produces the hidden preactivations, ReLU clamps their negative entries to
zero, and the second affine layer produces the output. TorchLean's `NN.Spec` library expresses
this sequence as Lean definitions, independently of the buffers and kernels used to evaluate it.

The distinction is important because executable ML code changes form constantly. A linear layer may
be evaluated by a reference Lean loop, a native CPU loop, cuBLAS, or an ATen kernel. The
specification gives all of those implementations one statement to satisfy.

We can run the algebraic part of this specification with exact rational arithmetic. Most of the
spec layer is parametric in the scalar type, and `ℚ` supplies the operations needed by this MLP.
That lets us inspect each intermediate value as a fraction before comparing it with floating-point
execution. The named Lean blocks below are checked while the page is built; the Python examples
record the corresponding framework behavior.

# Tensors

TorchLean uses one tensor type for specifications and execution:

```
-- The scalar type and shape are independent indices of the
-- same tensor interface.
Tensor α shape
```

Its runtime representation is one contiguous row-major buffer whose length is certified by
`shape`. The scalar type chooses the physical buffer through `Storage`: executable scalar types
can use packed native storage, while general proof scalar types use the generic representation.

Proofs do not reason about raw buffer offsets. A tensor coerces to a total function on the typed
coordinate space of its shape, and extensionality says that tensors with equal values at every
coordinate are equal. A vector coordinate contains `Fin n`, so out-of-range lookup cannot be
expressed. Shape induction remains available through rank-zero observation and leading-axis slicing,
but those are proof views and smart constructors over the same contiguous tensor, not a second
recursive physical datatype. Public shapes still use `[]`, `[n]`, and `[m, n]`.

Model code uses ordinary shape lists, and a linear layer is a structure with two fields:

```lean (name := slTypes)
-- Check shaped vectors, matrices, and linear-layer
-- parameter records over exact rationals.
#check Tensor ℚ [2]
#check Tensor ℚ [3, 2]
#check LinearSpec ℚ 2 3
```

```leanOutput slTypes (whitespace := lax)
Tensor ℚ [2] : Type
```

```leanOutput slTypes (whitespace := lax)
Tensor ℚ [3, 2] : Type
```

```leanOutput slTypes (whitespace := lax)
LinearSpec ℚ 2 3 : Type
```

For a linear map from two inputs to three outputs, `LinearSpec ℚ 2 3` contains

$$`W\in\mathbb{Q}^{3\times 2},
\qquad b\in\mathbb{Q}^{3}.`

Output features index the rows of the weight matrix, following PyTorch's `nn.Linear` convention.
Each row therefore contains the coefficients for one output coordinate. Keeping that layout in the
specification lets an imported PyTorch weight matrix retain its interpretation.
{Informal.citet pytorch2019}[] documents that layout.

The three `#check` results describe types, not computed tensors. `Tensor ℚ [2]` fixes two rational
coordinates, while `Tensor ℚ [3, 2]` fixes a matrix with three rows and two columns. The layer type
then packages a matrix and bias with compatible dimensions. These indices prevent a caller from
silently passing a tensor of a different shape, but they do not constrain the numerical values
inside it. Positivity, normalization, and closeness to another tensor are separate propositions.

# Scalar Operations And Algebraic Assumptions

The shape fixes where values live. The scalar context fixes which formulas are available. A linear
layer needs zero, addition, and multiplication. Softmax also needs exponentiation and division.
Square root, comparison, trigonometric operations, and Boolean decisions require their own pieces of
structure.

The signature separates the storage instance from the three arithmetic operations `linearSpec`
uses:

```lean (name := slLinearSig)
-- Only the arithmetic used by the linear formula appears in
-- its constraints.
#check @Spec.linearSpec
```

```leanOutput slLinearSig (whitespace := lax)
@linearSpec : {α : Type} → [inst : Storage α] → [Add α] → [Mul α] → [Zero α] →
  {inDim outDim : ℕ} → LinearSpec α inDim outDim → Tensor α [inDim] → Tensor α [outDim]
```

`reluSpec` asks for zero, a maximum operation, and Boolean equality:

```lean (name := slReluSig)
-- The zero comparison fixes the selected ReLU behavior for
-- AD scalar types.
#check @Activation.reluSpec
```

```leanOutput slReluSig (whitespace := lax)
@Activation.reluSpec : {α : Type} → [inst : Storage α] → [Zero α] → [Max α] → [BEq α] →
  {s : Shape} → Tensor α s → Tensor α s
```

At zero, ReLU returns the scalar context's zero. For a dual number, this also sets its tangent to
zero, matching TorchLean's selected derivative at the kink. Boolean equality selects that branch;
away from zero, the definition uses the maximum operation.

Neither signature asserts algebraic or order laws for those operations. Changing the scalar
instances
changes how those operations are evaluated, so the same linear-layer definition can be used over
integers, intervals, or binary32 bit patterns. Algebraic laws about those operations need their own
hypotheses.

TorchLean packages the common operations in `Context α`, so the same tensor definition can be
interpreted over reals, rounded formats, interval endpoints, or executable binary32. This
polymorphism requires care when stating hypotheses. A proof of a softmax identity over
`ℝ` may use properties of the real exponential that an arbitrary `Context α` does not provide.

The usual progression is:

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

These levels require different hypotheses; native execution also introduces provider contracts. A
theorem should not inherit a stronger numerical meaning
because the same operation name appears at every level.

The operation constraints explain why this specification can serve more than one purpose.
Addition, multiplication, and zero suffice to evaluate a linear layer; a field or an ordered
field would be unnecessarily strong requirements for that definition. A derivative theorem over
the reals can later use the stronger analytic structure of that particular scalar system. Thus
reusing a definition at several scalar types does not transport all theorems between them. It
transports the operation sequence, and the laws available for interpreting that sequence depend
on the chosen instances.

# Linear Layer Forward And Backward Rules

The forward definition is deliberately short:

```
-- The definition exposes the matrix product and the
-- subsequent bias addition.
def linearSpec {α : Type} [Storage α] [Add α] [Mul α] [Zero α]
    {inDim outDim : Nat}
    (m : LinearSpec α inDim outDim)
    (input : Tensor α [inDim]) :
    Tensor α [outDim] :=
  addSpec (matVecMulSpec m.weights input) m.bias
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
#eval Spec.linearSpec
  { weights := ([[4/5, -2/5]] : Tensor ℚ [1, 2]),
    bias := ([1/5] : Tensor ℚ [1]) }
  ([1/10, -1] : Tensor ℚ [2])
```

```leanOutput slLinearEval (whitespace := lax)
[(17 : Rat)/25]
```

The stored result is the fraction $`17/25`. Both products and the bias addition were evaluated in
`ℚ`, so there is no rounding error to separate from the formula in this example.

The backward specification receives an upstream cotangent $`g=\partial L/\partial y` and returns

$$`
\frac{\partial L}{\partial W}=g\,x^\mathsf{T},\qquad
\frac{\partial L}{\partial b}=g,\qquad
\frac{\partial L}{\partial x}=W^\mathsf{T}g.
`

In Lean these three tensors have shapes `[outDim, inDim]` , `[outDim]` , and `[inDim]` . A
transposed weight gradient has the wrong type when the input and output dimensions differ.
For a square weight matrix, its shape alone cannot detect the transpose.

`linearBackwardSpec` returns all three in a `Spec.LinearGradients` record. Its named fields
distinguish the weight, bias, and input gradients even when two of them have the same shape, as
the bias and input gradients do in a square layer. Convolution uses `Spec.ConvGradients` for the
same reason. For the second layer of our MLP, take a hidden activation of $`(\tfrac1{10},0)` and
a cotangent of $`1`:

```lean (name := slBackward)
-- Seed the scalar output with one and inspect parameter and
-- input cotangents separately.
def slSecond : Spec.LinearSpec ℚ 2 1 :=
  { weights := [[4/5, -2/5]], bias := [1/5] }

#eval Spec.linearBackwardSpec slSecond
  ([1/10, 0] : Tensor ℚ [2]) ([1] : Tensor ℚ [1])
```

```leanOutput slBackward (whitespace := lax)
{ weightGradient := [[(1 : Rat)/10, 0]], biasGradient := [1],
  inputGradient := [(4 : Rat)/5, (-2 : Rat)/5] }
```

Check each against the formula above. The weight gradient $`g\,x^\mathsf{T}` is the hidden
activation written as a row. The bias gradient is $`g` unchanged, because $`\partial y/\partial b`
is the identity. The input gradient $`W^\mathsf{T}g` is the weight row transposed. All three are
exact rationals that can be compared directly with the three expressions.

The source definitions are `linearSpec`, `linearWeightsDerivSpec`, `linearBiasDerivSpec`,
`linearInputDerivSpec`, and `linearBackwardSpec` in
{src "NN/Spec/Layers/Linear.lean"}[`NN/Spec/Layers/Linear.lean`]. Later, the autograd proofs compare
executable VJP rules with these definitions.
{ref "autograd-proofs"}[The autograd proofs chapter] states the agreement theorems and their
assumptions.

The backward output separates three roles. The weight cotangent is the upstream seed multiplied
by each input feature, the bias cotangent is the seed itself, and the input cotangent uses the
transpose of the weight map. Here the seed is one because the layer has one scalar output. With
seed two, every returned cotangent would double over the rationals. This dependence on the seed
is what makes a VJP compositional: an earlier layer receives the cotangent produced by a later
layer, rather than assuming that every layer's output is independently seeded with one.

# MLP Forward Semantics

The running model is the composition of two linear specifications and a pointwise activation. In the
library it is assembled as a chain of module specs rather than as one nested expression:

```lean (name := slMlpSig)
-- The model composes the same layer specifications used by
-- the individual examples.
#check @Examples.mlpSpec
```

```leanOutput slMlpSig (whitespace := lax)
@Examples.mlpSpec : {α : Type} → [inst : Storage α] → [Context α] → {inDim hidDim outDim : ℕ} →
  LinearSpec α inDim hidDim → LinearSpec α hidDim outDim → Module.Chain α [inDim] [outDim]
```

`Examples.mlpForward` runs that chain on one input vector. Give it the weights from the formula at
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
open scoped Spec.RationalAlgebraic in
def slFirst : Spec.LinearSpec ℚ 2 2 :=
  { weights := [[1, 1], [-1, 1]], bias := [0, 0] }

def slPoint : Tensor ℚ [2] := [1/4, -3/4]

open scoped Spec.RationalAlgebraic in
#eval Examples.mlpForward slFirst slSecond slPoint
```

```leanOutput slMlp (whitespace := lax)
[(1 : Rat)/5]
```

Expanding the chain gives the formula at the start of the chapter, and here it is expanded by hand,
one stage at a time, so the intermediate values are visible:

```lean (name := slStages)
-- Display the linear result, the selected ReLU branch, and
-- the resulting model output.
#eval Spec.linearSpec slFirst slPoint
#eval Activation.reluSpec (Spec.linearSpec slFirst slPoint)
#eval Spec.linearSpec slSecond
  (Activation.reluSpec (Spec.linearSpec slFirst slPoint))
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
zero, and the output is the second bias alone: exactly $`\tfrac15`. That the composed chain and the
hand expansion print the same fraction is a small but real check that `Module.Chain` wiring means
what the formula says.

No graph traversal or runtime state is hidden inside any of this. The values are rationals and the
evaluation is Lean's, not a backend's.

The intermediate outputs explain the MLP result more clearly than the final fraction alone. Both
hidden preactivations are negative, so the activation sends both to zero. The second matrix
therefore contributes nothing at this input, and its bias is the whole output. This also predicts
which local parameter changes can affect the result: a sufficiently small perturbation that leaves
both hidden values negative cannot expose the hidden matrix to the output. Crossing an activation
boundary changes that conclusion, which the later bias perturbation demonstrates explicitly.

## The selected backward rule at zero

The ReLU definition commits to two choices:

$$`\operatorname{ReLU}(z)=\max(z,0),`

and for the selected derivative,

$$`\operatorname{ReLU}'(z)=
  \begin{cases}
    1 & z>0,\\
    0 & z\le 0.
  \end{cases}`

The value at the kink matters. Other subgradients are mathematically defensible; anything in
$`[0,1]` is a valid subgradient of $`\max(z,0)` at zero, and a forward and backward correctness
theorem needs one concrete rule rather than a set. TorchLean chooses zero at the kink.

The zero choice agrees with TorchLean's runtime path and with PyTorch:

```
>>> # Inspect the gradient convention chosen at the ReLU
>>> # kink.
>>> x = torch.tensor(0.0, requires_grad=True)
>>> y = torch.relu(x); y.backward()
>>> x.grad.item()
0.0
```

Choosing $`\tfrac12` would make the selected backward rule disagree with the eager ReLU rule at
zero. Forward values would remain unchanged, and a surrounding model might mask the gradient
difference; the local backward contract would still need to record the discrepancy.

## Difference Quotients

Exact arithmetic also changes what a derivative check *is*. A finite-difference check in floating
point has to pick a step size and live with the error curve that comes with it, as
{ref "running-example"}[the running example] shows in detail. Over `ℚ` the second layer is affine,
so the step cancels algebraically and the quotient is the derivative for every nonzero step
whatsoever:

```lean (name := slQuot)
-- Affine difference quotients are exact at every nonzero
-- step used here.
/-- Forward difference quotients of the second layer in
both input coordinates, at step `h`, over `ℚ`. -/
def slQuot (h : ℚ) : ℚ × ℚ :=
  let y := (Spec.linearSpec slSecond [1/10, 0])[0]
  let y₀ := (Spec.linearSpec slSecond [1/10 + h, 0])[0]
  let y₁ := (Spec.linearSpec slSecond [1/10, h])[0]
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
  ((Activation.reluSpec ([h] : Tensor ℚ [1]))[0]
    - (Activation.reluSpec ([0] : Tensor ℚ [1]))[0]) / h

#eval (slReluQuot (1/1000), slReluQuot (-1/1000))
```

```leanOutput slKink (whitespace := lax)
(1, 0)
```

For every positive step the quotient is one; for every negative step it is zero. The disagreement
persists as the step approaches zero, so ReLU has no derivative there. The library's backward rule
selects the subgradient `0` from $`[0,1]`. Exact arithmetic makes the source of the disagreement
visible: it comes from the two branches of ReLU.

The exact difference quotients should be read according to the function being tested. For an
affine function, subtracting the value at the base point cancels the constant term, and division
by any nonzero step recovers the coefficient exactly. That explains why changing the step by many
orders of magnitude leaves these rational outputs unchanged. ReLU at zero has a different
structure: positive and negative steps approach through different affine pieces. The two quotient
values expose the failure of a two-sided derivative rather than a numerical failure to estimate
one. Choosing zero for the runtime backward rule is a convention at that point.

# Runtime And Specification Comparison

The same inputs can now be evaluated at `Float`. `Tensor.linear` is the runtime operation and
`Spec.linearSpec` is its pure specification; the following comparison checks their results for
this input:

```lean (name := slRuntime)
-- Compare the runtime linear route with the same formula at
-- scalar type Float.
def slWeight : Tensor Float [1, 2] := [[0.8, -0.4]]
def slBias : Tensor Float [1] := [0.2]
def slInput : Tensor Float [2] := [0.1, -1.0]

#eval Tensor.linear slInput slWeight slBias
#eval Spec.linearSpec
  { weights := slWeight, bias := slBias } slInput
#eval (Tensor.linear slInput slWeight slBias)[0] ==
  (Spec.linearSpec
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

They agree here, and PyTorch agrees with both:

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

The comparison covers one vector input. `Tensor.linear` also applies along the final axis of a
tensor with leading dimensions, so a batch of rows comes back as a batch of rows:

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

The third row needs a closer comparison. PyTorch prints `0.6000000000000001` while TorchLean
prints `0.600000`. Comparing the stored TorchLean value with the literal `0.6` reveals a difference
that its printer hides:

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

The equality check establishes that TorchLean's stored result differs from the literal $`0.6`,
although the subtraction still prints as `0.000000`. PyTorch's longer decimal output exposes a
difference that six decimal places hide. These displays alone do not establish that the two
libraries stored identical bit patterns; that would require a bit-level comparison. Even this
one-row affine calculation shows why formatted output is insufficient for an exact numerical
comparison.
{Informal.citet goldberg1991}[] is the standard account, and
{ref "runtime-approximation"}[the runtime approximation chapter] states what TorchLean actually
proves about the gap.

The executable examples run runtime tensors and an autograd tape. The `nn.linear` builder fixes
the parameter shapes and emits the runtime linear operation, while the graph interpreter assigns
`.linear` its `linearSpec` denotation. The VJP proof relates the selected backward rule to
`linearBackwardSpec`. These connections use the same parameter layout and backward convention.

Native execution adds another boundary: a capsule records the planned provider and its declared
obligations, and execution checks establish which provider actually ran. The capsule's evidence
determines what is known about that provider's agreement with the specification.

The batch output contains one application of the same affine map per row. No reduction across
rows is implied by `projectLast`: the first row's value does not depend on the second row's
features. The equality check against `0.6` then asks a stronger question than whether the printed
row looks like six tenths. A formatter can display both nearby floating-point values identically
and even print their small difference as zero. The Boolean comparison operates on the stored
values, so the two outputs are consistent rather than contradictory.

# Bias Perturbation And ReLU

The formula can be inspected without training, and this is where exact arithmetic earns its keep.
Change only the first bias, from $`0` to $`0.6`. The first hidden preactivation moves from
$`-\tfrac12` to $`\tfrac1{10}`, so it survives ReLU, and the output becomes

$$`0.8\cdot0.1+0.2=0.28=\tfrac{7}{25}.`

Both claims check out:

```lean (name := slPerturb)
-- Move one hidden preactivation across zero to expose the
-- newly active path.
def slShifted : Spec.LinearSpec ℚ 2 2 :=
  { slFirst with bias := ([3/5, 0] : Tensor ℚ [2]) }

#eval Spec.linearSpec slShifted slPoint
#eval Spec.linearSpec slSecond
  (Activation.reluSpec (Spec.linearSpec slShifted slPoint))
```

```leanOutput slPerturb (whitespace := lax)
[(1 : Rat)/10, -1]
```

```leanOutput slPerturb (whitespace := lax)
[(7 : Rat)/25]
```

$`\tfrac{7}{25}` is exactly $`0.28`. A `Float` result printed as `0.280000` would need a separate
comparison of its stored value, as the preceding example showed.

Interval propagation extends this calculation from one point to a set of possible inputs, bounding
each intermediate tensor. The sign change is the important event here: the first hidden unit
contributed zero before the bias changed, then contributed $`1/10` afterwards. For an input region,
we must determine which preactivation intervals stay on one side of zero and which can cross it.
That is part of the analysis that
{ref "verification"}[the verification chapter] formalizes.

Changing the first bias makes one hidden value positive while leaving the other path inactive.
The new output is consequently the old output bias plus one active hidden contribution. This is
a useful way to inspect a model specification: keep the parameter change small and local, then
account for the result at each stage. The calculation also identifies where an analytic proof
would need a branch condition. Within either fixed activation region the model is affine; at the
boundary the selected backward rule needs to be distinguished from ordinary differentiability.

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
  * `IEEE32Exec`
  * executable binary32 bit patterns, including signed zero, infinity, and NaN
*
  * interval contexts
  * sets of possible values, with outward enclosure operations
*
  * runtime `Float`
  * Lean's native executable floating-point value
:::

The rational interpretation illustrates why scalar polymorphism needs care. `Context ℚ` is a
*scoped* instance in {src "NN/Spec/Core/Context/Rational.lean"}[`Context/Rational.lean`], which is
why the MLP evaluation above had to write `open scoped Spec.RationalAlgebraic` to get at it. The
reason is that `ℚ` has no rational exponential, so the scoped backend defines the transcendentals
as zero:

```lean (name := slQExp)
-- The scoped rational context supplies placeholders for
-- transcendental operations.
open scoped Spec.RationalAlgebraic in
#eval (MathFunctions.exp (1 : ℚ), MathFunctions.log (1 : ℚ))
```

```leanOutput slQExp (whitespace := lax)
(0, 0)
```

Evaluating softmax with these placeholder transcendental operations produces:

```lean (name := slQSoftmax)
-- A well-typed call can still lack the intended analytic
-- interpretation.
open scoped Spec.RationalAlgebraic in
#eval Activation.softmaxVecSpec ([0, 1, 2] : Tensor ℚ [3])
```

```leanOutput slQSoftmax (whitespace := lax)
[0, 0, 0]
```

The entries sum to zero, so this result cannot be interpreted as a probability distribution.
The tensor calculation typechecks, but the real-exponential laws used to justify softmax do not
hold for these placeholder operations. The scoped instance requires a file to opt in to the
rational algebraic interpretation; callers still need to restrict its use to the supported
algebraic fragment.

Softmax is the standard example:

$$`\operatorname{softmax}(x)_i=
\frac{\exp(x_i-m)}{\sum_j\exp(x_j-m)},\qquad m=\max_jx_j`

is a clean real-valued formula. For finite binary32 inputs, `max` selects an existing value without
rounding.
Subtraction, exponential approximation, summation, and division can introduce error. A CUDA
reduction may also choose
a different summation tree. {ref "floats"}[The floating-point chapter] and
{ref "runtime-approximation"}[the runtime-approximation chapter] state the conditions under which
one interpretation encloses or approximates another; {Informal.citet flocq2011}[] and
{Informal.citet boldo2015}[] are the Rocq-side precedent for that style of formalization.

The rational exponential and softmax outputs show why typeclass availability is not a mathematical
correctness theorem. The scoped context lets algebraic examples use a broad interface, including
names for operations that are not meaningful transcendental implementations over rationals. Its
zero outputs are therefore evidence about that supplied context, not values of the real
exponential or a probability distribution. For analytic softmax claims, the real context and its
exponential laws matter. For an executable approximation, the chosen scalar implementation needs
its own relation to those real operations.

## Summation Order

To see how the reduction order enters, keep the four inputs fixed and change only the grouping.
Left-to-right and paired addition agree whenever addition is associative. The definitions below
let us compare both orders over `ℚ` and `Float`:

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
three times in the left-to-right order, while the paired order adds `1 + 1` first and gets a step it
can represent. Both results follow from rounding each addition in its specified order. A
specification stated over `ℚ` or `ℝ` does not determine which of these two numbers a kernel returns,
so a claim about a fused reduction, a multi-threaded partial-sum tree, or a tensor-core accumulation
needs a rounding-aware statement rather than an appeal to the formula.
{ref "runtime-approximation"}[The runtime approximation chapter] is where those statements are made,
and Higham's *Accuracy and Stability of Numerical Algorithms* is the standard reference for how
summation error depends on the order chosen.

# The Implemented Specifications

The spec layer is broader than the running MLP. `NN.Spec.Layers` contains typed meanings for linear
algebra, rank-polymorphic convolution, transposed convolution, and pooling, activations and losses,
normalization, dropout, embeddings, recurrent cells, selective scan, and scaled dot-product
attention. Model definitions under `NN.Spec.Models` compose these operations into families such as
CNNs, transformers, recurrent networks, and state-space models.

That inventory is a semantics inventory, not a runtime support matrix. An operation can have a pure
definition before it has an eager tape rule, typed graph lowering, a CUDA kernel, or an end-to-end
correctness theorem. The relevant runtime and lowering chapters name those smaller supported
fragments explicitly.

# Shapes And Element Types

The shape index $`s` and element type $`\alpha` answer different questions. `Tensor α s` fixes
both for one value, and `Graph.denote` follows the homogeneous-element contract from *Tensors And
Shapes*. The later IR chapter uses “heterogeneous” only for a table of differently shaped values,
not for a mixed-dtype graph.

# Specification Conventions

Shapes are only one source of ambiguity. The spec layer also fixes choices that a model name does
not determine. Each subsection below is a place where two reasonable libraries could differ, so the
definition has to say which one TorchLean means.

## Loss reductions

`mseSpec` takes a global mean over all entries. Cross-entropy over logits applies log-softmax along
the selected class dimension and averages over the remaining slices. Changing `mean` to `sum`
changes both the loss and every gradient by a scale factor.

The mean-squared error between our output $`\tfrac{17}{25}` and a target of $`\tfrac15` is
$`\left(\tfrac{12}{25}\right)^2=\tfrac{144}{625}`, and again the fraction is the answer:

```lean (name := slMse)
-- For one prediction, mean squared error is simply the
-- squared residual.
#eval Spec.mseSpec ([17/25] : Tensor ℚ [1])
  ([1/5] : Tensor ℚ [1])
```

```leanOutput slMse (whitespace := lax)
144 / 625
```

Because the objective is a mean, adding a fifth example to a four-example batch changes the gradient
of the existing four. A sum-reduced loss would not. Neither convention is wrong, and a learning rate
tuned under one may need rescaling under the other, which is why the reduction lives in the
definition and not
in a training script.

The MSE output is exact because this example uses a single rational residual. With several
coordinates, the choice of mean rather than sum introduces a factor equal to the reciprocal of
the number of entries. That factor also appears in the gradient. A forward comparison can look
plausible despite a reduction mismatch on one specially chosen input, so the loss specification
must identify which axes and how many coordinates participate. The name of a loss alone does not
settle its scaling in a training step.

## Attention masks

A boolean attention mask is a hard support constraint:

- `true` allows the key;
- `false` gives the key exactly zero softmax numerator;
- if every key in a row is blocked, the output row is zero.

This is the finite formulation of a negative-infinity mask. `causalMask n` is the standard lower
triangle from {Informal.citet transformer2017}[]:

```lean (name := slCausal)
-- Inspect both the Boolean causal support and the
-- normalized rows it permits.
#eval causalMask 3
#eval hardMaskedSoftmaxSpec
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
#eval hardMaskedSoftmaxVecSpec
  ([0.0, 1000.0] : Tensor Float [2])
  ([true, false] : Tensor Bool [2])
#eval Activation.softmaxVecSpec
  (Tensor.add ([0.0, 1000.0] : Tensor Float [2])
    ([0.0, -1000.0] : Tensor Float [2]))
```

```leanOutput slLeak (whitespace := lax)
[1.000000, 0.000000]
```

```leanOutput slLeak (whitespace := lax)
[0.500000, 0.500000]
```

The sentinel gave the blocked position half the attention mass. Push the scores a little further and
it takes essentially all of it:

```lean (name := slLeak2)
-- Increase the blocked score past the finite penalty to
-- expose the leaked probability.
#eval Activation.softmaxVecSpec
  (Tensor.add ([1.0, 1200.0] : Tensor Float [2])
    ([0.0, -1000.0] : Tensor Float [2]))
#eval hardMaskedSoftmaxVecSpec
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
attention concentrates on the blocked key. The relevant quantity is the score difference, not
whether either unshifted score exceeds $`1000`. PyTorch reproduces the same failure, because it is a
property of the arithmetic rather than of either library:

```
>>> # The comparison uses the same two score choices as the
>>> # Lean example.
>>> F.softmax(torch.tensor([0., 1000.]) + torch.tensor([0., -1000.]), dim=-1)
tensor([0.5000, 0.5000])
>>> F.softmax(torch.tensor([1., 1200.], dtype=torch.float64)
...           + torch.tensor([0., -1000.], dtype=torch.float64), dim=-1)
tensor([3.7618e-87, 1.0000e+00], dtype=torch.float64)
```

`hardMaskedSoftmaxSpec` computes the row maximum and the denominator over the allowed entries
only, so a blocked numerator is exactly zero. With real exponential and at least one allowed
entry, the denominator is positive: the maximum contributes $`\exp(0)=1`. This argument needs
laws for the scalar operations; the generic `Context` interface alone does not provide them.
Floating-point execution also needs a contract for exponentiation and non-finite inputs. A fully
blocked row is defined to be zero rather than
$`0/0`, matching PyTorch's scaled dot-product attention and TorchLean's CUDA providers:

```lean (name := slAllBlocked)
-- The declared all-blocked convention returns zero weights
-- rather than dividing by zero.
#eval hardMaskedSoftmaxVecSpec
  ([0.0, 1000.0] : Tensor Float [2])
  ([false, false] : Tensor Bool [2])
```

```leanOutput slAllBlocked (whitespace := lax)
[0.000000, 0.000000]
```

{Informal.citet flashattention2022}[] relies on exactly this structure when it reorders the softmax
denominator across tiles, since a hard zero is invariant under any summation order.

The causal example normalizes each row over its permitted columns. In the first row there is only
one allowed entry, so its probability is one regardless of that entry's score. Later rows have
more allowed entries and therefore a different denominator. A finite score penalty changes that
denominator instead of deleting a term from it, which is why a sufficiently large blocked score
can reclaim probability mass in the leakage examples. The all-blocked case needs its own rule
because there is no allowed denominator to normalize; the declared zero row makes that behavior
explicit.

The mask here is fixed data describing which coordinates may interact. A derivative theorem with
respect to scores differentiates the function with that mask held fixed. It does not define a
derivative with respect to a Boolean choice, and a theorem about a finite additive score bias
describes another function. These distinctions are necessary when comparing an attention proof
with the mask convention of an execution backend.

## Dropout

Randomness is explicit. A masked dropout specification receives the mask as an argument. Runtime
training code may generate that mask from a seed and tape state, but the semantic function does not
consult hidden global randomness.

The inverted-dropout scale of {Informal.citet dropout2014}[] is $`1/(1-p)`, which at $`p=\tfrac12`
is exactly $`2`:

```lean (name := slDropout)
-- Separate a fixed dropout mask from the scale applied to
-- its retained coordinates.
open scoped Spec.RationalAlgebraic in
#eval dropoutKeepScale ((1 : ℚ)/2)

open scoped Spec.RationalAlgebraic in
#eval dropoutMaskedSpec ((1 : ℚ)/2)
  ([true, false, true, false] : Tensor Bool [4])
  ([1, 2, 3, 4] : Tensor ℚ [4])
```

```leanOutput slDropout (whitespace := lax)
2
```

```leanOutput slDropout (whitespace := lax)
[2, 0, 6, 0]
```

Kept entries are doubled and dropped entries are zero. If each mask entry is Bernoulli with
keep probability $`1/2` , this preserves each entry in expectation over that distribution.
Evaluation mode is the identity on values, which is where PyTorch and TorchLean agree
trivially:

```lean (name := slDropoutEval)
-- Inference dropout leaves every input coordinate
-- unchanged.
#eval dropoutInferenceSpec (0.5 : Float)
  ([1.0, 2.0, 3.0, 4.0] : Tensor Float [4])
```

```leanOutput slDropoutEval (whitespace := lax)
[1.000000, 2.000000, 3.000000, 4.000000]
```

```
>>> # Use inference mode explicitly when checking the
>>> # identity behavior.
>>> x = torch.tensor([1., 2., 3., 4.])
>>> F.dropout(x, p=0.5, training=False).tolist()
[1.0, 2.0, 3.0, 4.0]
```

With an explicit mask, the output is determined by the arguments shown in the definition. A theorem
can first reason about that fixed mask, then add a distributional assumption when an expectation is
needed. A stateful specification can also be formalized, but must include the generator state in
its contract.

The dropout example conditions on one particular mask. A kept entry is multiplied by the keep
scale and a dropped entry becomes zero, so the displayed vector can be checked coordinate by
coordinate. A probabilistic claim about expectations would additionally need a distribution over
masks and the matching keep probability. The deterministic tensor equation alone does not supply
that distribution. Inference mode takes a different route altogether: it is the identity, which
is why the inference example returns the original four entries without rescaling them.

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
#eval convOutShape ([28, 28] : Tensor Nat [2])
  ([5, 5] : Tensor Nat [2]) ([2, 2] : Tensor Nat [2])
  ([0, 0] : Tensor Nat [2])
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

For a zero stride or a kernel wider than the padded input, the usual window calculation is
invalid. The shape helper still returns a `Nat`:

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
value for every configuration. Here `0` represents no output positions. Runtime validation
separately determines whether the configuration may be executed and rejects these cases before
native code is reached. A computed shape can thus be used in a type without implying that every
configuration producing it is admitted by the runtime.

Returning zero from the spatial size formula makes it a total function on natural-number inputs.
It does not certify that every such configuration is acceptable to a convolution runtime. A
stride of zero and a kernel larger than the available padded extent can therefore be described by
shape arithmetic while still being rejected by an execution API. When using a convolution theorem,
the accompanying well-formedness or positive-stride hypotheses identify the intended operating
region; the computed output length is only one part of that contract.

# MLP Specification Equivalence

GraphSpec's checked MLP uses the same four parameter tensors:

```
-- The ordered parameter roles matter even when two entries
-- have equal shapes.
[W₁ : [hidden, input],
 b₁ : [hidden],
 W₂ : [output, hidden],
 b₂ : [output]]
```

The theorem `NN.GraphSpec.Models.mlp_interp_eq_spec_mlp_forward` proves that interpreting that
GraphSpec model is equal to the hand-written two-layer specification, for every parameter pack and
every input, at every scalar type with `Storage` and `Context` instances:

```lean (name := slAxioms)
-- Inspect dependencies of the pure model-equivalence
-- theorem, not a native execution claim.
#print axioms
  NN.GraphSpec.Models.mlp_interp_eq_spec_mlp_forward
```

```leanOutput slAxioms (whitespace := lax)
'NN.GraphSpec.Models.mlp_interp_eq_spec_mlp_forward' depends on axioms:
[propext, Classical.choice, Quot.sound]
```

Those three are Lean's standard classical foundations, also used by
{Informal.citet mathlib2020}[]. The theorem adds no TorchLean-specific axiom or unproved
placeholder.

The conclusion identifies the pure GraphSpec interpretation with
$`W_2\operatorname{ReLU}(W_1x+b_1)+b_2`. A bound on that interpretation can therefore be read as
a bound on this formula, with the same parameters and input. Connecting it to an imported
checkpoint, an eager tape, or a CUDA execution requires the corresponding import and runtime
arguments.

# Sources

The definitions used in this chapter, in the order they appeared:

- {src "NN/Spec/Layers/Linear.lean"}[`Layers/Linear.lean`], `LinearSpec` and the four
  forward and backward definitions;
- {src "NN/Spec/Layers/Activation.lean"}[`Layers/Activation.lean`], `reluSpec`,
  `softmaxVecSpec`, and the log-softmax family;
- {src "NN/Spec/Layers/Attention.lean"}[`Layers/Attention.lean`], `causalMask`,
  `hardMaskedMax?`, and `hardMaskedSoftmaxSpec`;
- {src "NN/Spec/Layers/Dropout.lean"}[`Layers/Dropout.lean`], `dropoutKeepScale` and the
  masked and inference specifications;
- {src "NN/Spec/Layers/Conv.lean"}[`Layers/Conv.lean`], the window geometry;
- {src "NN/Spec/Models/Mlp.lean"}[`Models/Mlp.lean`], `mlpSpec` and `mlpForward`;
- {src "NN/Spec/Core/Context/Rational.lean"}[`Core/Context/Rational.lean`], the scoped
  rational backend;
- {src "NN/GraphSpec/Models/MlpSpecEquivalence.lean"}[`MlpSpecEquivalence.lean`], the
  equivalence theorem.

The next chapter turns these formulas into typed architectures with explicit parameter layouts.

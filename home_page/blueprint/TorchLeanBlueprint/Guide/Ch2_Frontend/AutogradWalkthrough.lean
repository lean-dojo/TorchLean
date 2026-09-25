import VersoManual
import NN.API
import NN.Proofs.Autograd.Core.RealCorrectness
import NN.Proofs.Autograd.FDeriv.MlpMse
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open Runtime.Autograd.Torch (Ops)
open Proofs.Autograd
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Differentiation" =>
%%%
tag := "autograd-walkthrough"
file := "Differentiation-By-Example"
%%%

Squaring two coordinates gives two outputs. To differentiate their sum, reverse mode starts
with weight one on each output; to give the second output ten times the influence, we change
that weight to ten. Those weights form the output cotangent. Together with the function and its
input, they determine the vector-Jacobian product.

I'll keep the input fixed while changing the derivative query, then apply the same reasoning to
an affine model with every parameter stated explicitly. We can calculate its loss, parameter
gradient, and curvature by hand before comparing the returned tensors. Named Lean blocks and
their outputs are checked when this page is built; shell and PyTorch transcripts record separate
runs.

After the command-line examples, the tensor calculations use:

$$`x=(0.5,\,-1.2).`

# Runnable Differentiation Examples

The two runnable examples are:

```terminal
# Compare tensor differentiation with differentiation of a
# model loss.
lake exe torchlean quickstart_autograd
```

```terminal +output
== Differentiate a tensor function ==
mean(x^2) = 4.666667
d/dx       = [0.666667, 1.333333, 2.000000]

== Differentiate a model loss ==
loss     = 0.769887
gradient = [[1, 2]: [[-0.877432, 1.754865]], [1]: [-1.754865]]
```

and

```terminal
# Request directional derivatives and curvature without an
# optimizer update.
lake exe torchlean autograd_transforms
```

```terminal +output
Jacobian of x^2: [[1.000000, 0.000000], [0.000000, -2.400000]]
Hessian of mean(x^2): [[1.000000, 0.000000], [0.000000, 1.000000]]
loss directional derivative = 0.011629
loss Hessian-vector product = [[3, 2]: [[0.010000, -0.024000],
  [0.010000, -0.024000], [0.010000, -0.024000]], [3]: [0.020000, 0.020000, 0.020000]]
state gradient after detaching the model output = [[3, 2]: [[0.000000, 0.000000],
  [0.000000, 0.000000], [0.000000, 0.000000]], [3]: [0.000000, 0.000000, 0.000000]]
mixed input derivative of tanh at zero: [0.000000, 0.000000] (expected [0, 0])
third input derivative along [1, 0]: [-2.000000, 0.000000] (expected [-2, 0])
```

Their sources are {src "NN/Examples/Quickstart/AutogradBasics.lean"}[`AutogradBasics.lean`] and
{src "NN/Examples/DeepDives/AutogradTransforms.lean"}[`AutogradTransforms.lean`].

The first transcript uses $`x=(1,2,3)`: `mean(x^2) = 4.666667` is $`(1+4+9)/3`, and the gradient
is $`2x/3`. Its model result also depends on the weights drawn by `nn.build 0 (nn.linear 2 1)`.
Reproducing `-0.877432` by hand would require those weights, the sample, and the loss. In the model
calculations below, we set every parameter to a stated constant so the output and each derivative
can be derived directly.

In the second transcript, the Jacobian has one row per output of the square function. Its first
row says that changing the first input changes only the first output; the second row says the
same for the second coordinate, with a negative slope at $`-1.2`. The Hessian columns describe a
different function, the scalar mean of those squares. The identity matrix means its gradient
changes by exactly the perturbation applied to the input. The final zero pack belongs to the
detached loss, so it should be compared with the ordinary model gradient, not with a zero forward
loss. These distinctions matter when several derivative objects appear in one log.

The last two calls differentiate model inputs, not parameters. The model applies $`\tanh` to
each coordinate independently, so its mixed derivative in the two coordinate directions is zero.
Repeating the first direction three times gives $`\tanh'''(0)=-2` in the first output and zero
in the second. `autograd.model.derivative` takes the directions as a list; its length chooses
the derivative order.

The affine model's initial state is seeded, whereas `agState` below replaces that state with
constants. This explains why its directional derivative is `0.011629` and the later fixed-state
calculation gives `-0.004200`.

# The Differentiable Program Type

The type being differentiated is not `Tensor α σ → Tensor α τ`:

```lean (name := agFnType)
-- The two shapes describe the input and output; the
-- interpreter is quantified inside.
#check @autograd.Function
```

```leanOutput agFnType (whitespace := lax)
autograd.Function : Shape → Shape → Type 1
```

It abbreviates this:

```
-- Scalar and interpreter choices remain arguments of the
-- program.
abbrev Function (σ τ : Shape) :=
  ∀ {α : Type}, [TorchLean.Storage α] → [Context α] → {m : Type → Type} → [Monad m] →
      [Runtime.Autograd.Torch.Ops (m := m) (α := α)] →
      TorchLean.Runtime.ValueRef (m := m) (α := α) σ →
      m (TorchLean.Runtime.ValueRef (m := m) (α := α) τ)
```

The quantifiers separate the program from the scalar representation and execution machinery.

`{α : Type}` with `[Storage α]` and `[Context α]` means the function does not know its scalar type.
The same source text runs at `Float`, at software binary32, and at `ℝ` for proofs.

`{m : Type → Type}` with `[Monad m]` means it does not know how operations are being recorded. It
could be building an eager tape, or a typed graph, or nothing at all.

`[Ops (m := m) (α := α)]` is the operation vocabulary, a class with one field per primitive: `add`,
`mul`, `matmul`, `relu`, `reduceSum`, `detach`, and so on. A `Function` may only call those fields.

`ValueRef … σ` is a handle with a shape in its type, not a tensor. The body can pass handles around
but cannot look inside one.

A `Function` is a shape-correct program written against an abstract interpreter: a typed
tagless-final encoding ({Informal.citet kiselyov2012}[]). This means
that `autograd.grad` does not inspect Lean source syntax. It applies the function to a
graph-building
interpreter, recording supported operations for differentiation. That is the same reason JAX made
differentiation a
function on functions rather than a method on a tensor ({Informal.citep jax2018}[]), reached by a
different route: JAX traces a Python function into a jaxpr, whereas here the polymorphism is in the
type and Lean's elaborator does the work.

Because a `Function` is universally quantified over `α`, it
lives in `Type 1`, so it cannot be stored in an `IO` reference or returned from `IO` without care,
and it cannot branch on the numeric value of an intermediate tensor. Data-dependent control flow has
to be expressed as an operation (a `clamp`, a mask, a `max`) rather than as an `if`. That
restriction is what makes the same text differentiable, provable, and executable at three scalar
types.

A shape index tells the interpreter which kind of handle it must return. For `agSumSq`, the
input handle has shape `[2]`, squaring preserves that shape, and reduction returns shape `[]`.
That empty shape denotes one scalar value; it does not denote an empty tensor. The body never
needs to extract a host array to make the reduction. Consequently, a recorder can retain the
connection from the scalar result to both input coordinates. Extracting numbers and computing
with ordinary host arithmetic would require a different interface and would lose that recorded
connection unless a derivative rule were supplied for the new operation.

# Scalar Gradients

Take

$$`f(x_0,x_1)=\frac{x_0^2+x_1^2}{2},`

which in TorchLean is a two-line `do` block, and differentiate it:

```lean (name := agGrad)
-- Keep the input fixed so the mean reduction and its
-- gradient can be checked together.
def agX : Tensor Float [2] := [0.5, -1.2]

def agSumSq : autograd.Function [2] [] := fun x => do
  let squared ← nn.functional.square x
  nn.functional.mean squared

#eval show IO Unit from do
  let (gradient, value) ←
    autograd.grad agSumSq agX (value := true)
  IO.println s!"value    = {value}"
  IO.println s!"gradient = {reprStr gradient}"
```

```leanOutput agGrad (whitespace := lax)
value    = 0.845000
gradient = [0.500000, -1.200000]
```

By hand, $`f(x)=(0.25+1.44)/2=0.845` and $`\nabla f(x)=x=(0.5,-1.2)`. Both agree.

The reduction divides by two, cancelling the factor of two from differentiating each square. Thus
the gradient of $`\tfrac12\sum x_i^2` is $`x`. Replacing `mean` with `sum` would leave the
elementwise derivative rule unchanged but double the final gradient.

The signature explains the `(value := true)`:

```lean (name := agGradSig)
-- The value option changes the result type, so callers
-- cannot confuse a pair with a gradient.
#check @autograd.grad
```

```leanOutput agGradSig (whitespace := lax)
@autograd.grad : {σ : Shape} →
  autograd.Function σ [] →
    {α : Type} →
      [inst : Storage α] →
        [Context α] →
          Tensor α σ →
            (value : optParam Bool false) →
              IO
                (match value with
                | false => Tensor α σ
                | true => Tensor α σ × Tensor α [])
```

The return type is computed from `value`. With its default `false`, the result is just the
input-shaped gradient. With `true`, it is a pair containing that gradient and the scalar forward
value, often called the primal. The caller can request both results from one evaluation, and the
type records which form was requested.

The order of the pair is worth following at the call site: `(gradient, value)`, with the
input-shaped object first. The `IO` in the signature allows the helper to construct and execute
its recorded computation and report failures. It does not make the tensor itself a mutable
gradient container. Nor does the polymorphic `Function` signature promise that every backend
implements every primitive: the selected interpreter still has to support the operations the
function calls.

## PyTorch Gradients

```
# Keep PyTorch's default float32 visible when comparing
# printed derivatives.
import torch
x = torch.tensor([0.5, -1.2], requires_grad=True)
y = (x ** 2).mean()
(g,) = torch.autograd.grad(y, x)
print("%.6f" % y.item(), g.tolist())
```

```
0.845000 [0.5, -1.2000000476837158]
```

The value matches. The gradient's second component does not print the same way, and the reason is
not automatic differentiation at all: `torch.tensor([0.5, -1.2])` is float32, and `-1.2` is not a
binary fraction, so the stored number is `-1.2000000476837158`. TorchLean's `Float` is binary64
here, where the same literal rounds to something that prints as `-1.200000`. Neither library made an
arithmetic mistake; they were asked to differentiate at slightly different points
({Informal.citep goldberg1991}[]). {ref "floats"}[Floating-Point Semantics] is where this stops
being a printing curiosity, and {ref "tensors-shapes"}[Tensors And Shapes] states which scalar
contract each entry point uses.

For a numerical comparison, first align the input dtype and then compare the stored values.
Printing both answers to six places hides the distinction in this example, while printing many
digits exposes it. Neither choice changes the derivative program. A tolerance comparison asks
whether the numerical answers are sufficiently close for a stated purpose; a bit comparison
asks whether the representations agree exactly. Those are useful but different questions even
before a reduction tree or a GPU kernel enters the calculation.

# Vector-Jacobian Products

For

$$`f:\mathbb R^n\to\mathbb R^m`

with Jacobian $`J_f(x)\in\mathbb R^{m\times n}`, reverse mode accepts an output cotangent
$`\bar y\in\mathbb R^m` and returns

$$`\bar x=J_f(x)^{\mathsf T}\bar y.`

The cotangent assigns a weight to each output component. Holding those weights fixed, the VJP is
the input gradient of their weighted sum. No Jacobian is built. One reverse traversal costs about
what one forward evaluation costs, whatever
$`n` is, which is the fact that makes training large models possible at all
({Informal.citep baydin2018}[]).

Take the elementwise square

$$`g(x_0,x_1)=(x_0^2,x_1^2),
\qquad
J_g(x)=\begin{bmatrix}2x_0&0\\0&2x_1\end{bmatrix},`

and pull two different cotangents back through it:

```lean (name := agVjp)
-- Changing only the output cotangent isolates the weighting
-- performed by reverse mode.
def agSquare : autograd.Function [2] [2] := fun x =>
  nn.functional.square x

#eval show IO Unit from do
  let ones : Tensor Float [2] := [1.0, 1.0]
  let tilted : Tensor Float [2] := [1.0, 10.0]
  let dx1 ← autograd.vjp agSquare agX ones
  let dx2 ← autograd.vjp agSquare agX tilted
  IO.println s!"seed (1, 1)  -> {reprStr dx1}"
  IO.println s!"seed (1, 10) -> {reprStr dx2}"
```

```leanOutput agVjp (whitespace := lax)
seed (1, 1)  -> [1.000000, -2.400000]
seed (1, 10) -> [1.000000, -24.000000]
```

$`J_g(x)^{\mathsf T}(1,1)=(2\cdot0.5,\,2\cdot(-1.2))=(1,-2.4)`, and scaling the second seed
component by ten scales the second output by ten. PyTorch reports the same two vectors from
`torch.autograd.grad(out, x, grad_outputs=seed)`.

The two seeds ask different questions about the same function: the first weights both outputs
equally, while the second gives the second output ten times the weight. `autograd.vjp` requires
this choice explicitly, with a cotangent of the output shape:

```lean (name := agVjpSig)
-- The seed has output shape, while the returned sensitivity
-- has input shape.
#check @autograd.vjp
```

```leanOutput agVjpSig (whitespace := lax)
@autograd.vjp : {σ τ : Shape} →
  autograd.Function σ τ →
  {α : Type} → [inst : Storage α] → [Context α] →
  Tensor α σ → Tensor α τ → IO (Tensor α σ)
```

Input, output cotangent, and result: `Tensor α σ`, `Tensor α τ`, `Tensor α σ`. Handing in a seed of
the wrong shape is a compile error, not a runtime broadcast.

A scalar gradient is the special case in which the output has one coordinate and its cotangent
is one. A zero cotangent would ask for the derivative of a zero-weighted output and return zero,
even when the function itself is sensitive to its inputs. This is why the seed belongs in the
meaning of a VJP rather than being an incidental implementation argument. It also explains why
an output-shaped seed is enough: the reverse traversal combines its weights with local rules
without allocating a matrix with one entry for every input/output pair.

# Full Jacobians And Signed Zero

Sometimes the whole Jacobian is what you want. TorchLean builds it both ways:

- `jacfwd` runs forward mode once per input coordinate;
- `jacrev` runs reverse mode once per output coordinate.

Both return a tensor whose output axes come first, followed by the input axes.

```lean (name := agJac)
-- Compute both orientations of the same Jacobian to expose
-- their numerical conventions.
#eval show IO Unit from do
  let forward ← autograd.jacfwd agSquare agX
  let reverse ← autograd.jacrev agSquare agX
  IO.println s!"jacfwd {forward}"
  IO.println s!"jacrev {reverse}"
```

```leanOutput agJac (whitespace := lax)
jacfwd [[1.000000, 0.000000], [-0.000000, -2.400000]]
jacrev [[1.000000, 0.000000], [0.000000, -2.400000]]
```

Both recover $`\operatorname{diag}(1,-2.4)` as a `Tensor Float [2, 2]`. For higher-rank inputs
and outputs, the concatenated shape preserves their axes.

Now look at the first column: `-0.000000`. The off-diagonal entry is negative zero, and it is
negative because forward mode computed $`2x_1\cdot 0 = 2\cdot(-1.2)\cdot 0`, whose IEEE 754 result
is $`-0` . Reverse-mode accumulation can add a $`-0` contribution to an initial $`+0` , producing
$`+0` .
Reversing the order of multiplication alone would not change its sign. PyTorch does the same thing
with the following signed-zero entries:

```
jacfwd = [[1.0, 0.0], [-0.0, -2.4000000953674316]]
jacrev = [[1.0, -0.0], [0.0, -2.4000000953674316]]
```

IEEE equality treats `-0.0` and `0.0` as equal, but their bits differ. Operations such as
reciprocal,
copy-sign, and some elementary functions expose that distinction; even adding a signed zero can
change a zero result's sign. The real-valued Jacobian does not record it.
{ref "floats"}[Floating-Point Semantics] explains why numerical equality and bit equality must be
reported separately.

Choosing between the two modes is a shape question. Forward mode costs one pass per input
coordinate, reverse mode one pass per output coordinate, so a function from many inputs to one
scalar wants `jacrev`, and one from a few inputs to many outputs wants `jacfwd`. Either way,
choosing the cheaper mode takes $`\min(n,m)` directional passes, with storage for $`nm` entries.
Most scalar-loss training needs only one VJP instead.

For this two-coordinate function, the full Jacobian occupies four entries although only two are
nonzero. A function from an image tensor to ten scores would retain the image axes after a
leading score axis, so materializing the Jacobian would store ten image-sized arrays. If the
next computation only needs a weighted sum of those scores, one VJP computes precisely that
combination. The choice is about the derivative information the caller needs, as well as which
mode takes fewer passes.

# Gradient Return Values And Accumulation

PyTorch's usual training-loop idiom stores gradients on tensors. Two backward calls without
clearing that storage show its accumulation behavior:

```
# Rebuild the forward expression but retain the same leaf
# gradient buffer.
xa = torch.tensor([0.5, -1.2], requires_grad=True)
(xa ** 2).mean().backward()
print("after one backward :", xa.grad.tolist())
(xa ** 2).mean().backward()
print("after two backwards:", xa.grad.tolist())
xa.grad = None
(xa ** 2).mean().backward()
print("after zeroing      :", xa.grad.tolist())
```

```
after one backward : [0.5, -1.2000000476837158]
after two backwards: [1.0, -2.4000000953674316]
after zeroing      : [0.5, -1.2000000476837158]
```

The second reverse pass did not report the second gradient. It added to the first one, because
`.grad` is a mutable buffer that `backward` accumulates into. That behaviour is deliberate and
useful in PyTorch: it is how gradient accumulation over microbatches works, and how a parameter
shared between two subgraphs collects both contributions ({Informal.citep pytorch2019}[]). It is
also a source of training mistakes, such as a forgotten
`optimizer.zero_grad()`, which does not crash and does not warn. It quietly trains a different
model.

TorchLean's transforms return the derivative as a value. There is no `.grad` field to clear, and
running the same query twice returns the same answer twice, because nothing was mutated.
Accumulation across microbatches is an explicit addition of returned packs.

The caller supplies the function to differentiate and passes the resulting gradient to its
consumer. In a language where values are immutable and
sharing is managed by reference counting ({Informal.citep immutablebeans2019}[]), returning the pack
can avoid a copy when ownership permits buffer reuse; actual allocation cost depends on the
operations that build and consume the pack.
{ref "runtime-autograd"}[Runtime and Autograd] shows the tape underneath, where accumulation across
several consumers of one value is still exactly the addition PyTorch performs, just at a different
level.

# Parameter VJPs

Now the affine model

$$`y=Wx+b,\qquad W\in\mathbb R^{3\times2},\ b\in\mathbb R^3,`

with every entry of $`W` and $`b` set to $`0.1`. These fixed parameters let us calculate the
forward value and all three parts of the pullback.

For an output cotangent $`\bar y`, calculus gives three derivatives:

$$`
\bar W=\bar y\,x^{\mathsf T},
\qquad
\bar b=\bar y,
\qquad
\bar x=W^{\mathsf T}\bar y.
`

With $`\bar y=(1,1,1)`, that predicts every row of $`\bar W` equal to $`x=(0.5,-1.2)`, every
component of $`\bar b` equal to one, and $`\bar x=(0.3,0.3)` because each column of $`W` sums to
$`0.3`. One reverse pass, all three:

```lean (name := agModelVjp)
-- Build the parameter layout once, then replace
-- initialization with known values.
def agModel : nn.Sequential [2] [3] :=
  nn.build 0 (nn.linear 2 3)

/-- Every weight and bias pinned to `0.1`, so the numbers
below can be checked by hand. -/
def agState : autograd.model.State agModel Float :=
  autograd.model.fullState agModel 0.1

def agIn : Tensor Float [2] := [0.5, -1.2]

#eval show IO Unit from do
  let ones : Tensor Float [3] := [1.0, 1.0, 1.0]
  let (stateGrad, inputGrad) ←
    autograd.model.vjp agModel agState agIn ones
  IO.println s!"state gradient = {reprStr stateGrad}"
  IO.println s!"input gradient = {reprStr inputGrad}"
```

```leanOutput agModelVjp (whitespace := lax)
state gradient = [[3, 2]: [[0.500000, -1.200000],
  [0.500000, -1.200000], [0.500000, -1.200000]],
  [3]: [1.000000, 1.000000, 1.000000]]
input gradient = [0.300000, 0.300000]
```

All three predictions, from one call. PyTorch's
`torch.autograd.grad(out, [lin.weight, lin.bias, xi], grad_outputs=torch.ones(3))` returns the same
three objects with the same numbers.

The printed shape tags correspond to the return type:

```lean (name := agModelVjpSig)
-- One reverse pass returns both model-state and input
-- sensitivities.
#check @autograd.model.vjp
```

```leanOutput agModelVjpSig (whitespace := lax)
@autograd.model.vjp : {σ τ : Shape} →
  (model : nn.Sequential σ τ) →
  {α : Type} → [inst : Storage α] → [Context α] →
  autograd.model.State model α → Tensor α σ →
  Tensor α τ →
  IO (autograd.model.State model α × Tensor α σ)
```

The returned state gradient has type `autograd.model.State model α`, the same type as the
parameters that went in. The pack checks their ordered shape list, but it cannot detect a
semantic swap between gradients for two parameters with equal shapes. The input gradient is
returned alongside
the state gradient, so both results come from the same reverse pass.

The type protects structural correspondence: a `[3,2]` weight gradient cannot occupy the `[3]`
bias slot. For this affine model there are six weight entries and three biases, so the returned
pack contains nine parameter sensitivities, plus the separate two-entry input sensitivity.
For a model with persistent state, `State` can also include buffers. The model transform computes
sensitivities for the state entries; the optimizer's trainability mask separately determines
which entries it updates. A derivative result by itself is not an instruction to modify every
state tensor.

# Loss And Parameter Gradients

A loss determines the output cotangent. Write the model output as $`\widehat y=F_\theta(x)` and
the target as $`y`, with the same shape. The Jacobian $`J_{F_\theta}^{\theta}(x)` measures how
that output changes with the parameters $`\theta`. The chain rule pulls the loss's output
gradient back through this Jacobian:

$$`
\nabla_\theta L
=
J_{F_\theta}^{\theta}(x)^{\mathsf T}\,
\nabla_{\widehat y}L(\widehat y,y).
`

With all parameters at $`0.1` and $`x=(0.5,-1.2)`, the model output is
$`0.1\cdot0.5+0.1\cdot(-1.2)+0.1=0.03` in each of the three components. Against the target
$`(0.7,0.1,-0.5)` the residuals are $`(-0.67,-0.07,0.53)`, so the mean squared error is

$$`\frac{0.4489+0.0049+0.2809}{3}=0.2449,`

and the output gradient of the mean squared error is $`\tfrac23(\widehat y-y)`, that is
$`(-0.446\overline6,\,-0.046\overline6,\,0.353\overline3)`:

```lean (name := agModelGrad)
-- The loss supplies the output cotangent from prediction
-- errors against this target.
def agTarget : Tensor Float [3] := [0.7, 0.1, -0.5]

def agMse : autograd.model.Loss [3] [3] :=
  autograd.model.Loss.mse

#eval show IO Unit from do
  let (gradient, lossValue) ←
    autograd.model.grad agModel agMse agState agIn agTarget
      (value := true)
  IO.println s!"loss     = {lossValue}"
  IO.println s!"gradient = {reprStr gradient}"
```

```leanOutput agModelGrad (whitespace := lax)
loss     = 0.244900
gradient = [[3, 2]: [[-0.223333, 0.536000],
  [-0.023333, 0.056000], [0.176667, -0.424000]],
  [3]: [-0.446667, -0.046667, 0.353333]]
```

PyTorch's `mse_loss` and `torch.autograd.grad` return `0.244900` and the same two arrays.

The bias gradient is the output gradient. Each weight row is the
corresponding bias gradient times $`x`: $`-0.446\overline6\cdot0.5=-0.223\overline3` and
$`-0.446\overline6\cdot(-1.2)=0.536`. That is $`\bar W=\bar y\,x^{\mathsf T}` again, now with
$`\bar y` supplied by the loss instead of by us.

## The Outer-Product Gradient Theorem

The same outer-product rule describes the final layer of a two-layer ReLU network.
Write $`a_1=\operatorname{ReLU}(W_1x+b_1)` for the hidden activation and
$`y=W_2a_1+b_2` for the output. For the derivative with respect to $`W_2`,
$`a_1` is fixed: it plays the role that $`x` played in our affine example. The
mean squared error supplies the output gradient $`\bar y`, giving
$`\bar W_2=\bar y\,a_1^{\mathsf T}`.

{src "NN/Proofs/Autograd/FDeriv/MlpMse.lean"}[`MlpMse.lean`] proves this identity
using Mathlib's Fréchet derivative and adjoint ({Informal.citep mathlib2020}[]):

```lean (name := agFDeriv)
-- This is the real-valued outer-product theorem,
-- independent of a floating-point run.
#check @Proofs.Autograd.grad_W2_mse
```

```leanOutput agFDeriv (whitespace := lax)
@grad_W2_mse : ∀ {inDim hidDim outDim : ℕ}
  (W1 : Mat hidDim inDim) (b1 : Vec hidDim)
  (W2 : Mat outDim hidDim) (b2 : Vec outDim)
  (x : Vec inDim) (t : Vec outDim),
  (ContinuousLinearMap.adjoint
      (fderiv ℝ
        (fun W2 => mse t (mlpVecMat W1 b1 W2 b2 x)) W2))
      1 =
    outer (mseGrad (y W1 b1 W2 b2 x) t) (a1 W1 b1 x)
```

Read the right-hand side: `outer` of the output gradient with the hidden activation, which is
$`\bar y\,a_1^{\mathsf T}`. The left-hand side is the adjoint of the Fréchet derivative of
the scalar loss, applied to the seed $`1`. The theorem is stated over real-valued vectors and
matrices, independently of a tape or floating-point evaluation.

The runtime printed `-0.223333` for one input; the theorem establishes the outer-product formula
for all inputs over `ℝ`. Connecting a floating-point execution to this real-valued statement also
requires a refinement argument about rounding, developed in
{ref "fp32-soundness"}[Float32 Soundness].

The seed `1` on the theorem's left is the same scalar cotangent used by `grad`. The adjoint
converts a linear map from weight perturbations to loss perturbations into a gradient in weight
space. That is why the result has matrix shape even though the loss is scalar.

# ReLU Derivatives At Zero

ReLU has no derivative at zero, but autograd still has to return something. TorchLean returns zero:

```lean (name := agKink)
-- Include a zero input to distinguish the selected ReLU
-- rule from its smooth branches.
def agRelu : autograd.Function [3] [3] := fun x =>
  Ops.relu x

#eval show IO Unit from do
  let z : Tensor Float [3] := [-1.0, 0.0, 1.0]
  let seed : Tensor Float [3] := [1.0, 1.0, 1.0]
  let dz ← autograd.vjp agRelu z seed
  IO.println s!"relu vjp at (-1, 0, 1) = {reprStr dz}"
```

```leanOutput agKink (whitespace := lax)
relu vjp at (-1, 0, 1) = [0.000000, 0.000000, 1.000000]
```

PyTorch prints `[0.0, 0.0, 1.0]` for the same input, so the convention agrees, and it is a
convention: $`0`, $`1`, and $`\tfrac12` are all defensible choices at the kink, and different
libraries have shipped different ones. The modern justification for treating the returned value as a
real derivative object rather than as an error is Bolte and Pauwels' conservative-field model
({Informal.citet bolte2020}[]), which provides a framework for nonsmooth differentiation and
convergence results under explicit
hypotheses; it is not an unconditional convergence guarantee for arbitrary selections or training.

The theorem for a first-layer bias makes the smoothness requirement explicit:

```lean (name := agKinkThm)
-- The nonzero-preactivation hypothesis is needed when the
-- derivative crosses ReLU.
#check @Proofs.Autograd.grad_b1_mse
```

```leanOutput agKinkThm (whitespace := lax)
@grad_b1_mse : ∀ {inDim hidDim outDim : ℕ}
  (W1 : Mat hidDim inDim) (b1 : Vec hidDim)
  (W2 : Mat outDim hidDim) (b2 : Vec outDim)
  (x : Vec inDim) (t : Vec outDim),
  (∀ (i : Fin hidDim), (z1 W1 b1 x).ofLp i ≠ 0) →
    (ContinuousLinearMap.adjoint
        (fderiv ℝ
          (fun b1 => mse t (mlpVecMat W1 b1 W2 b2 x)) b1))
        1 =
      (reluDerivCLM (z1 W1 b1 x))
        ((ContinuousLinearMap.adjoint
            (matCLM (toMatrix W2)))
          (mseGrad (y W1 b1 W2 b2 x) t))
```

The hypothesis `∀ i, (z1 W1 b1 x).ofLp i ≠ 0` says no hidden pre-activation sits exactly on the
kink. The gradient with respect to the first-layer bias passes through the ReLU derivative, so the
theorem is stated only where that derivative exists. Compare it with `grad_W2_mse` above, which
needs no such hypothesis: the second-layer weight sits after the activation, so it never sees the
nonsmooth point.

The gradient's shape does not establish differentiability. Here the theorem excludes hidden
pre-activations at zero; a theorem covering those points would need to state which generalized
derivative it describes.

In the three-entry ReLU output, the negative input contributes zero because changing it slightly
still leaves the activation at zero. The positive input contributes one because ReLU is locally
the identity there. The middle zero comes from the selected rule at the boundary. A finite
difference centered on that boundary can average behavior from both sides and report a different
number. Such a comparison must account for nonsmoothness before diagnosing the local rule as an
implementation error.

# Local Rules And The Global Reverse Pass

At runtime each primitive contributes a forward value, references to its parents, and a local VJP
rule. Reverse traversal seeds the output, applies the local rules in reverse topological order, and
adds cotangents wherever several paths meet. To see why addition is necessary, consider

$$`z=\tfrac12\!\left(x^2+x^2\right),`

the two paths each contribute $`x`, so the gradient must be $`2x`:

```lean (name := agAccum)
-- Both square nodes depend on x, so their cotangents must
-- be added at that shared input.
def agTwice : autograd.Function [2] [] := fun x => do
  let a ← nn.functional.square x
  let b ← nn.functional.square x
  let s ← Ops.add a b
  nn.functional.mean s

#eval show IO Unit from do
  let g ← autograd.grad agTwice agX
  IO.println s!"grad of mean(x^2 + x^2) = {reprStr g}"
```

```leanOutput agAccum (whitespace := lax)
grad of mean(x^2 + x^2) = [1.000000, -2.400000]
```

Exactly twice the `[0.500000, -1.200000]` from the single-path version. A tape that overwrote one
parent contribution instead of adding would print `[0.500000, -1.200000]` here and would be wrong
while every local rule was right. This accumulation occurs within one reverse pass, whenever a
value has several consumers. It is separate from retaining gradients across successive backward
calls, as PyTorch's `.grad` buffers do.

The split into local rules plus a global traversal is also how the proofs are organized. The
algebraic property each local rule must satisfy is adjointness of its forward and reverse
derivatives:

```lean (name := agVjpCorrect)
-- Read the three functions as the forward map, its chosen
-- JVP, and its chosen VJP.
#check @Proofs.Autograd.VJPCorrect
```

```leanOutput agVjpCorrect (whitespace := lax)
@VJPCorrect : {σ τ : Shape} →
  (Tensor ℝ σ → Tensor ℝ τ) →
  (Tensor ℝ σ → Tensor ℝ σ → Tensor ℝ τ) →
  (Tensor ℝ σ → Tensor ℝ τ → Tensor ℝ σ) → Prop
```

Unfolded, it is

$$`
\bigl\langle \operatorname{JVP}(x,dx),\,\delta\bigr\rangle
=
\bigl\langle dx,\,\operatorname{VJP}(x,\delta)\bigr\rangle
\quad\text{for all }x,\ dx,\ \delta,
`

with $`\langle\cdot,\cdot\rangle` the tensor dot product. This says forward mode and reverse mode
are adjoint to each other. Identifying the chosen JVP with the analytic derivative requires
a separate result; once that is established, this is the Jacobian-transpose law
$`\langle J\,dx,\delta\rangle=\langle dx,J^{\mathsf T}\delta\rangle`. It is chosen deliberately in
preference to "the VJP equals the analytic Jacobian transpose": it is provable by algebra for
elementwise rules, it is exactly what the reverse pass needs in order to compose, and it makes sense
for rules whose local derivative is a chosen mask rather than a classical derivative
({Informal.citet pearlmutter2008}[]).

A proved-correct operation bundles the forward map, the reverse rule, the forward rule, and the
proof:

```lean (name := agOpSpec)
-- This bundle stores local derivative rules together with
-- their adjointness proof.
#check @Proofs.Autograd.OpSpecCorrect
```

```leanOutput agOpSpec (whitespace := lax)
OpSpecCorrect : Shape → Shape → Type
```

```lean (name := agSquareCorrect)
-- The square rule preserves shape and supplies an
-- adjointness proof at every shape.
#check @Proofs.Autograd.squareCorrect
```

```leanOutput agSquareCorrect (whitespace := lax)
@squareCorrect : {s : Shape} → OpSpecCorrect s s
```

The bundled proof specializes to the shape used in this example:

```lean (name := agAdjointThm)
-- Specialize the bundled identity without introducing a new
-- derivative assumption.
/-- Adjointness for `square`, at the shape we use. -/
theorem agSquareAdjoint (x dx delta : Tensor ℝ [2]) :
    Spec.dot ((squareCorrect (s := [2])).jvp x dx) delta =
      Spec.dot dx
        ((squareCorrect (s := [2])).op.backward x delta) :=
  (squareCorrect (s := [2])).correct x dx delta
```

And the reverse-mode chain rule is a function on those bundles:

```lean (name := agCompose)
-- The intermediate shape is shared by the first output and
-- the second input.
#check @Proofs.Autograd.OpSpecCorrect.compose
```

```leanOutput agCompose (whitespace := lax)
@OpSpecCorrect.compose : {σ τ υ : Shape} →
  OpSpecCorrect σ τ → OpSpecCorrect τ υ →
  OpSpecCorrect σ υ
```

Composing two proved operations produces a proved operation, with no new obligation for the caller:

```lean (name := agChain)
-- Composition carries the adjointness evidence through both
-- chosen local rules.
/-- ReLU followed by squaring, carrying its own proof. -/
noncomputable def agReluSquare : OpSpecCorrect [2] [2] :=
  (reluCorrect (s := [2])).compose
    (squareCorrect (s := [2]))

/-- The composite obeys the same law. Nothing to prove here:
`compose` already did it. -/
theorem agChainAdjoint (x dx delta : Tensor ℝ [2]) :
    Spec.dot (agReluSquare.jvp x dx) delta =
      Spec.dot dx (agReluSquare.op.backward x delta) :=
  agReluSquare.correct x dx delta
```

Primitive rules establish adjointness, `compose` establishes that composition preserves it, and
the tape soundness results in
{ref "autograd-proofs"}[Autograd Correctness] connect a recorded traversal to that composition.
{src "NN/Proofs/Autograd/Core/RealCorrectness.lean"}[`RealCorrectness.lean`] carries the bundles for
ReLU, sigmoid, tanh, softplus, SiLU, GELU, ELU, exp, log, the epsilon-protected logarithm, smooth
absolute value, sinh, cosh, square, linear layers, and reductions. Its companion
`SemiringCorrectness.lean` keeps the algebraic ones generic over a commutative semiring so exact
backends such as `ℚ` can use them without dragging in real analysis.

The `OpSpecCorrect s s` output says that one bundle works at any shape `s`; it contains no
runtime gradient numbers. In the two proof blocks, `.correct` extracts its adjointness field at
the supplied `x`, `dx`, and `delta`. Composition feeds the first operation's forward value to the
second and sends the second operation's cotangent back through the first. The intermediate
shape must match in both directions. This is the structural reason the same bundle interface
can organize many local rules without reproving the chain of dot-product equalities each time.

# `detach` And Gradient Flow

`autograd.model.Loss.detach` leaves the forward value alone and replaces the backward map by zero:

```lean (name := agDetach)
-- Stop the model-output path while retaining the same
-- prediction and target in the loss.
#eval show IO Unit from do
  let detached := autograd.model.Loss.detach agMse
  let (gradient, lossValue) ←
    autograd.model.grad agModel detached agState agIn
      agTarget (value := true)
  IO.println s!"detached loss     = {lossValue}"
  IO.println s!"detached gradient = {reprStr gradient}"
```

```leanOutput agDetach (whitespace := lax)
detached loss     = 0.244900
detached gradient = [[3, 2]: [[0.000000, 0.000000],
  [0.000000, 0.000000], [0.000000, 0.000000]],
  [3]: [0.000000, 0.000000, 0.000000]]
```

The loss remains `0.244900`, while every parameter gradient is zero. Both results matter: detach
must preserve the forward value and remove the derivative contribution along this path.

Detach is used in target networks, stop-gradient estimators, contrastive objectives, and
statistics that must not receive gradients. It changes the update rule without changing the forward
scalar value. The resulting vector field
need not be the gradient of any scalar objective, so adding a detach requires mathematical intent.
{ref "runtime-autograd"}[Runtime and Autograd] shows the same idea one level down, where detach is a
tape node that the reverse traversal simply never crosses.

Detaching the model output also leaves the target available to the loss. The operation cuts the
path back into the model state; it does not replace the target or erase the residual calculation.
For example, the three nonzero residuals above still produce `0.244900`, but none can send a
cotangent through the stopped output. When reading a zero gradient, inspect this connectivity
before concluding that a parameter is at a stationary point of the original, undetached loss.

# JVPs And Hessian-Vector Products

Forward mode computes $`J_f(x)v`, and composing it with reverse mode gives a Hessian-vector product
$`H_f(x)v` without ever forming $`H_f(x)`, which is Pearlmutter's trick
({Informal.citep pearlmutter2008}[]). For model parameters:

```lean (name := agHvp)
-- Use one state-shaped direction for both the loss slope
-- and the change in its gradient.
#eval show IO Unit from do
  let direction : autograd.model.State agModel Float :=
    autograd.model.fullState agModel 0.1
  let d ←
    autograd.model.jvp agModel agMse agState agIn agTarget
      direction
  let c ←
    autograd.model.hvp agModel agMse agState agIn agTarget
      direction
  IO.println s!"directional derivative = {d}"
  IO.println s!"Hessian-vector product = {reprStr c}"
```

```leanOutput agHvp (whitespace := lax)
directional derivative = -0.004200
Hessian-vector product = [[3, 2]: [[0.010000, -0.024000],
  [0.010000, -0.024000], [0.010000, -0.024000]],
  [3]: [0.020000, 0.020000, 0.020000]]
```

Both numbers are checkable, which is the reason for pinning the parameters. The direction sets every
entry to $`0.1`, so its effect on each output component is
$`0.1\cdot0.5+0.1\cdot(-1.2)+0.1=0.03`. The mean squared error contributes a factor $`\tfrac23`, so
the bias block of $`Hv` is $`\tfrac23\cdot0.03=0.02` and each weight row is
$`0.02\,x=(0.01,-0.024)`. The directional derivative is the dot product of the earlier gradient with
the direction, which is $`0.1\cdot0.098-0.1\cdot0.14=-0.0042` in exact arithmetic. For this model
the Hessian is
exactly $`\tfrac23 J^{\mathsf T}J` , because the output is linear in the parameters. Nonlinear
dependence on parameters can add second-order
terms; the transform must also support the operations and derivative rules used.

For a scalar function of a small input, the full Hessian is also available, and for
$`f(x)=\tfrac12(x_0^2+x_1^2)` it should be the identity:

```lean (name := agHessian)
-- The mean of two squares has identity Hessian because its
-- gradient is x.
#eval show IO Unit from do
  let hessian ← autograd.hessian agSumSq agX
  IO.println s!"hessian {hessian}"
```

```leanOutput agHessian (whitespace := lax)
hessian [[1.000000, 0.000000], [0.000000, 1.000000]]
```

`torch.func.hessian(lambda v: (v ** 2).mean())` prints the same matrix. Change `mean` to `sum` and
it becomes $`2I`, which is the reduction convention showing up in the second derivative as well.

Cost is the reason `hvp` exists next to `hessian`. A Hessian-vector product is a constant number of
passes. A full Hessian for a model with $`p` parameters is $`p` of them, and $`p^2` numbers to
store.

The Hessian-vector product has the same two blocks as the parameter direction, so it can be
paired with that direction or passed to an iterative curvature method without flattening the
state. Its positive and negative entries reflect the signs of the two input coordinates; they
do not by themselves say the Hessian is indefinite. For this affine squared-loss example the
quadratic form is nonnegative over the reals because it is a scaled squared norm of `Jv`.
That conclusion uses the stated model and loss, rather than the signs of a printed vector.

# Runtime Derivatives And Mathematical Derivatives

Differentiation connects three distinct objects:

1. the calculus derivative of the ideal real-valued operation;
2. the derivative program TorchLean's transform builds;
3. the numbers a CPU, a CUDA kernel, or an external provider produces when it runs.

Claims 1 and 2 are related by Lean theorems for supported primitives and well-formed graphs; this
chapter showed both halves of that, the adjointness bundles and the `fderiv` results. Claim 3 needs
a rounding argument or an explicit backend boundary. A CUDA kernel may implement the same formal VJP
equation with a different reduction tree and produce different last bits, which is not a bug in
either the kernel or the theorem, and is precisely why they are counted as different claims.

First-order `autograd` helpers use typed graph lowering, including dual-number evaluation for
forward mode. Higher-order helpers also use the tape lowering described in the runtime chapter. A
trainer's choice between
eager and typed-graph execution, and its selection of native kernels, are separate decisions
described in {ref "execution-modes"}[Execution Modes] and
{ref "backend-selection"}[Backend Selection]. Primals, cotangents, parameters, and returned
derivatives all follow the scalar contract from {ref "tensors-shapes"}[Tensors And Shapes].

# Tape Inspection In VS Code

Place the cursor on one of:

```
-- Inspect connectivity and cotangent accumulation in an
-- existing recorded example.
#tape_view ...
#tape_grads_view ...
#tape_trace_view ...
```

For `agTwice`, inspect the two square nodes and the cotangents that meet at their shared input.
For the detached loss, inspect where the reverse path stops. The views also identify parameter
leaves and show the accumulated gradient at each reached node.
{ref "widgets"}[Interactive Widgets] documents the controls.

# API Summary

The transforms differ in what they seed and what they return:

:::table +header
*
  * Query
  * Seed
  * Result
  * Best use
*
  * `grad`
  * scalar output cotangent `1`
  * one gradient with the input shape
  * scalar objectives
*
  * `grad (value := true)`
  * scalar output cotangent `1`
  * gradient and primal value
  * losses that should be evaluated only once
*
  * `vjp`
  * caller-supplied output cotangent
  * pullback into the input shape
  * vector-valued functions
*
  * `jacfwd`
  * input basis tangents
  * Jacobian tensor, output axes then input axes
  * small input dimension
*
  * `jacrev`
  * output basis cotangents
  * Jacobian tensor, output axes then input axes
  * small output dimension
*
  * `hessian`
  * nested forward and reverse queries
  * Hessian tensor, input axes repeated
  * small scalar problems
*
  * `hvp`
  * one parameter-shaped direction
  * Hessian-vector product
  * curvature without a full Hessian
:::

For one-input tensor functions:

```
-- These queries differentiate a single-input tensor
-- function.
autograd.grad
autograd.vjp
autograd.jacfwd
autograd.jacrev
autograd.hessian
```

For checked models:

```
-- These queries retain the model's state layout in their
-- derivative results.
autograd.model.grad
autograd.model.vjp
autograd.model.jacrev
autograd.model.jvp
autograd.model.hvp
```

Namespace completion after `autograd.` lists the function transforms, and after `autograd.model.`
the model transforms and the loss namespace. Named options such as `value` live in the signature
rather than spawning a second declaration per variant.

# Differentiation Experiments

The examples isolate several aspects of the derivative rules. Small changes let us check each
one independently:

1. Replacing `nn.functional.mean` with a sum in `agSumSq` doubles the gradient by removing the
   division by two.
2. The seed $`(0,1)` in `autograd.vjp agSquare` selects the second Jacobian row.
3. Setting `agState` to `autograd.model.fullState agModel 0.2` changes the loss through the
   affine model output. The same residual calculation predicts the new value.
4. Setting one coordinate of `agIn` to $`0` makes the corresponding weight-gradient column zero.
5. Evaluating `agRelu` at `[-1.0, 1.0e-30, 1.0]` and then at `[-1.0, -1.0e-30, 1.0]` crosses the
   kink.
   The reported derivative jumps between $`0` and $`1` across a distance of $`2\cdot10^{-30}`.
6. Replacing `reluCorrect` in `agReluSquare` with `tanhCorrect` preserves the adjointness proof
   through `compose`. Operations whose intermediate shapes disagree cannot be composed.
7. Adding a third `nn.functional.square` path to `agTwice` adds one more copy of the
   single-path gradient.

The next page applies these transforms to a scientific equation, using input derivatives to build
a differential-equation residual.
{ref "runtime-autograd"}[Runtime and Autograd] then opens the runtime and shows which graph or tape
each of these calls actually builds, and why the canonical verification IR is a third artifact.

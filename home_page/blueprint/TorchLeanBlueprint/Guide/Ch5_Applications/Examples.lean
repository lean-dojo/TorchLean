import VersoManual
import NN.API
import NN.MLTheory.CROWN.BoundOps
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Examples" =>
%%%
tag := "examples"
file := "Worked-Examples"
%%%

Two tensors can print the same decimal digits and still contain different numbers. We will start
there, with four entries whose sums we can check by hand. Keeping the calculations small also lets
us follow a gradient back to its factors and an interval bound back to its endpoints. When the
examples reach training and imported graphs, those calculations give us something concrete to
compare with the command output.

Every Lean block on this page is elaborated when the site is built, and its output is checked
against the accompanying expected output. Shell and Python commands let you compare the
calculations with your executable and external dependencies.

# Tensor Shapes And Scalar Types

A `Tensor α s` carries its shape `s` in its type and its arithmetic in its element type `α`. Both
choices are visible in the first quickstart. The shape controls which operations can compose; the
element type determines how their arithmetic is evaluated.

```lean (name := exBasics)
-- Hold the values comparable while changing rank and scalar
-- arithmetic.
#eval show IO Unit from do
  let threshold : Tensor Float [] := 0.5
  let floats : Tensor Float [4] := [0.1, 0.2, 0.3, 0.4]
  let rationals : Tensor ℚ [4] := [0.1, 0.2, 0.3, 0.4]
  let integers : Tensor Int [4] := [1, 2, 3, 4]
  IO.println s!"rank zero = {threshold}"
  IO.println s!"Float     = {floats}"
  IO.println s!"Rat       = {rationals}"
  IO.println s!"Int       = {integers}"
```
```leanOutput exBasics
rank zero = 0.500000
Float     = [0.100000, 0.200000, 0.300000, 0.400000]
Rat       = [(1 : Rat)/10, (1 : Rat)/5, (3 : Rat)/10, (2 : Rat)/5]
Int       = [1, 2, 3, 4]
```

The rank-zero tensor has shape `[]`, not a special scalar type. That is a deliberate choice: a loss
value and a batch of images then have the same type constructor, so `Trainer` does not need a
separate code path for the scalar it minimizes.

Writing `0.1` at `ℚ` produces the exact number
$`\tfrac1{10}`, and the display says so. Writing `0.1` at `Float` produces the nearest binary64
value, which is not $`\tfrac1{10}` {Informal.citep goldberg1991}[]. The two rows look almost
identical at this display precision, despite representing different numbers.

A scalar loss still has one stored value: an empty list of axes has size one. By contrast,
a shape such as `[0]` has an axis with no entries. Keeping those cases separate matters when
reductions or empty batches are involved. The rank-zero threshold above is therefore an ordinary
input to scalar tensor operations, while the four-element examples let us inspect how a reduction
combines several stored values into that same scalar shape.

## Floating-Point And Rational Arithmetic

```lean (name := exExact)
-- Test equality separately from the rounded tensor display.
#eval show IO Unit from do
  let floats : Tensor Float [4] := [0.1, 0.2, 0.3, 0.4]
  let rationals : Tensor ℚ [4] := [0.1, 0.2, 0.3, 0.4]
  let floatSum := Tensor.sumSpec floats
  IO.println s!"Float sum   = {floatSum}"
  IO.println s!"Rat sum     = {Tensor.sumSpec rationals}"
  IO.println s!"Float sum 1 = {floatSum == 1.0}"
  IO.println s!"0.1+0.2=0.3 = {(0.1 : Float) + 0.2 == 0.3}"
  IO.println s!"exact       = {(0.1 : Rat) + 0.2 == 0.3}"
```
```leanOutput exExact
Float sum   = 1.000000
Rat sum     = 1
Float sum 1 = true
0.1+0.2=0.3 = false
exact       = true
```

The equality tests distinguish values that the six-decimal display hides. `0.1 + 0.2` is not
`0.3` in binary64.
But summing all four tenths *does* give exactly `1.0`, because the rounding errors happen to cancel
along that particular left fold. Python binary64 arithmetic gives the same answers for this
explicitly ordered expression:

```
# Preserve the addition order used by the four-element
# reduction.
print(0.1 + 0.2 == 0.3, 0.1 + 0.2 + 0.3 + 0.4 == 1.0)
```

```
False True
```

The printed digits alone cannot establish equality. TorchLean keeps `Float`, `Float32`,
FloatLib binary32, `ℚ`, and $`\mathbb R` as separate element types so that definitions and theorems
identify the arithmetic they use. See {ref "floats"}[the floating-point chapter] and
{Informal.citet boldo2015}[] for the same separation in Coq.

The rational calculation is a useful reference because its result does not depend on a
rounding policy. It is not a replacement for checking the requested runtime arithmetic. If the
application stores binary32 weights, a theorem about the corresponding rational expression
still needs a connection to conversion and rounding. I use these decimals because we can keep
their exact meanings in view while comparing the two arithmetic paths.

## Rank, Slices, And The Reshape Guard

```lean (name := exCube)
-- Reinterpret four stored elements as two rows without
-- changing their order.
#eval show IO Unit from do
  let cube : Tensor Float [2, 2, 2] :=
    [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
  let digits : Array Float := #[1.0, 2.0, 3.0, 4.0]
  let flat := Tensor.from digits
  IO.println s!"cube = {cube}"
  IO.println s!"mat  = {flat.reshape [2, 2]}"
```
```leanOutput exCube (whitespace := lax)
cube = [[[1.000000, 2.000000], [3.000000, 4.000000]],
  [[5.000000, 6.000000], [7.000000, 8.000000]]]
mat  = [[1.000000, 2.000000], [3.000000, 4.000000]]
```

`Tensor.reshape` takes the size equality as an argument with `by decide` as its default, so a
reshape that does not preserve the element count is rejected while the file is being elaborated:

```lean +error (name := exBadReshape)
-- Four source elements cannot supply the nine entries
-- required by this result type.
def exDigits : Array Float := #[1.0, 2.0, 3.0, 4.0]

def exBroken : Tensor Float [3, 3] :=
  (Tensor.from exDigits).reshape [3, 3]
```
```leanOutput exBadReshape (whitespace := lax)
could not synthesize default value for parameter 'hSize' using tactics
```

```leanOutput exBadReshape (whitespace := lax)
Tactic `decide` proved that the proposition
  (Tensor.SourceShape.shape exDigits).size = Shape.size [3, 3]
is false
```

The second half of the message identifies the failed proposition and how it was decided.

The equality is the entire safety condition: reshape does not pad, truncate, or move data, so if the
counts agree the operation is a reinterpretation of the same buffer. Stating it as a proof
obligation erases the size check at runtime. It prevents a mismatched element count; callers still
need to choose the intended axis order, and native storage remains a separate implementation
boundary.

A PyTorch operation can report incompatible shapes at execution time. This addition example
illustrates that timing:

```
# These non-singleton dimensions cannot be made equal by
# broadcasting.
torch.zeros(2, 2) + torch.zeros(3, 3)
```

```
RuntimeError: The size of tensor a (2) must match the size of tensor b (3)
at non-singleton dimension 1
```

The addition fails when executed {Informal.citep pytorch2019}[]. The Lean reshape above fails
during elaboration because its size-equality obligation is false. These checks establish specific
shape requirements; neither establishes that the program implements the intended model.

The command that prints the tensors above is
{src "NN/Examples/Quickstart/TensorBasics.lean"}[`TensorBasics.lean`]:

```terminal
# Print the standalone tensor quickstart, including its
# explicit Float32 cast.
lake exe torchlean quickstart_tensors
```

```terminal +output
== Quickstart: tensor basics ==
Rank-zero tensor: 0.500000
Float tensor:    [0.100000, 0.200000, 0.300000, 0.400000]
Rational tensor: [(1 : Rat)/10, (1 : Rat)/5, (3 : Rat)/10, (2 : Rat)/5]
Integer tensor:  [1, 2, 3, 4]
Float32 cast:    [0.100000, 0.200000, 0.300000, 0.400000]
Rank-3 tensor:   [[[1.000000, 2.000000], [3.000000, 4.000000]], ...
Array -> matrix: [[1.000000, 2.000000], [3.000000, 4.000000]]
```

The rank-3 line is truncated here to fit the page; it is the same value the block above printed.
Read the source beside the output, change one element type at a time, and let Lean show which
operations need a different algebraic context.

The cube and the matrix also separate shape from meaning. The same four values can represent
two points with two features each, or a two-by-two image. Lean checks the chosen dimensions,
but the program supplies their interpretation. In later examples a leading axis means a batch,
a trailing axis means features or classes, and transposing them changes the computation even
when their sizes happen to agree. Naming those roles beside the values is part of specifying
the model.

# Function And Model Differentiation

To compute a gradient, we first choose what may vary. `autograd.grad` takes the input tensor as
that variable; `autograd.model.grad` takes the model's typed state. We can see the difference by
differentiating a mean of squares, then a loss for one affine layer.

```lean (name := exGrad)
-- Differentiate a scalar mean with respect to all three
-- input coordinates.
def exMeanSquare : autograd.Function [3] [] :=
  fun x => do
    let squared ← nn.functional.square x
    nn.functional.mean squared

#eval show IO Unit from do
  let x : Tensor Float [3] := [1.0, 2.0, 3.0]
  let (g, v) ← autograd.grad exMeanSquare x (value := true)
  IO.println s!"mean(x^2) = {v}"
  IO.println s!"d/dx      = {g}"
```
```leanOutput exGrad
mean(x^2) = 4.666667
d/dx      = [0.666667, 1.333333, 2.000000]
```

For $`f(x)=\tfrac13\sum_i x_i^2` we have
$`\partial f/\partial x_i = \tfrac23 x_i`, so the gradient at $`(1,2,3)` is
$`(\tfrac23,\tfrac43,2)`, and $`f(1,2,3)=\tfrac{14}3=4.6\overline6`. Every digit above matches.
PyTorch computes the same thing by the same rules {Informal.citep baydin2018}[]:

```
# Match binary64 arithmetic and seed reverse mode with the
# scalar output cotangent.
x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
y = (x ** 2).mean()
y.backward()
print(y.item(), x.grad.tolist())
```

```
4.666666666666667 [0.6666666666666666, 1.3333333333333333, 2.0]
```

The extra digits are the display, not the arithmetic: PyTorch prints a shortest round-trip
representation of binary64, and TorchLean's tensor display rounds to six decimals.

The input gradient has shape `[3]` because it answers one question per input coordinate:
how would the scalar mean change if that coordinate moved? The division by three belongs to
the objective, so it must appear in every coordinate derivative. Replacing the mean with a sum
would leave the intermediate squared tensor unchanged and triple the returned gradient. This
small example makes a missing reduction factor visible without depending on parameter
initialization or a training loop.

## Differentiating A Model

The second path takes a model and a loss, and returns a gradient shaped like the model's state:

```lean (name := exModelGrad)
-- The differentiation variable is the typed weight-and-bias
-- state.
def exModel : nn.Sequential [2] [1] :=
  nn.build 0 (nn.linear 2 1)

#eval show IO Unit from do
  let state : autograd.model.State exModel Float :=
    autograd.model.initialState exModel
  let input : Tensor Float [2] := [0.5, -1.0]
  let target : Tensor Float [1] := [0.25]
  let (g, loss) ← autograd.model.grad exModel
    autograd.model.Loss.mse state input target
    (value := true)
  IO.println s!"loss     = {loss}"
  IO.println s!"gradient = {reprStr g}"
```
```leanOutput exModelGrad (whitespace := lax)
loss     = 0.769887
gradient = [[1, 2]: [[-0.877432, 1.754865]], [1]: [-1.754865]]
```

The gradient prints with its shapes, `[1, 2]` for the weight and `[1]` for the bias, which is the
same list `nn.stateShapes` reports for this model. Their values follow from the chain rule. For
$`y=Wx+b` and a squared-error loss on one output,

$$`\frac{\partial L}{\partial b}=2(y-t),
\qquad
\frac{\partial L}{\partial W}=2(y-t)\,x^{\top},
\qquad
L=(y-t)^2.`

So the bias gradient should be twice the residual, and the weight gradient should be that same
number times the input $`(0.5,-1)`. The loss determines the residual magnitude. Its sign also
requires the prediction or the signed
gradient; the following calculation uses the negative sign from this example's bias gradient:

```lean (name := exGradCheck)
-- Recover the residual magnitude and use the observed
-- gradient sign.
#eval show IO Unit from do
  let state : autograd.model.State exModel Float :=
    autograd.model.initialState exModel
  let input : Tensor Float [2] := [0.5, -1.0]
  let target : Tensor Float [1] := [0.25]
  let (_, loss) ← autograd.model.grad exModel
    autograd.model.Loss.mse state input target
    (value := true)
  let residual := Float.sqrt (Tensor.at loss ())
  IO.println s!"|y - t|   = {residual}"
  IO.println s!"2 (y - t) = {-2.0 * residual}"
  IO.println s!"times 0.5 = {-2.0 * residual * 0.5}"
```
```leanOutput exGradCheck
|y - t|   = 0.877432
2 (y - t) = -1.754865
times 0.5 = -0.877432
```

`-1.754865` is the printed bias gradient and `-0.877432` is the printed first weight gradient, to
every digit shown. The sign tells us the untrained prediction is below the target. This calculation
checks the chain rule and the squared-error convention against the displayed gradient.

The same identity can be checked in PyTorch with explicitly chosen parameters:

```
# Fix the parameters explicitly so the chain-rule arithmetic
# can be checked by hand.
lin = torch.nn.Linear(2, 1).double()
with torch.no_grad():
    lin.weight.copy_(torch.tensor([[0.1, 0.2]], dtype=torch.float64))
    lin.bias.copy_(torch.tensor([0.0], dtype=torch.float64))
x = torch.tensor([0.5, -1.0], dtype=torch.float64)
t = torch.tensor([0.25], dtype=torch.float64)
loss = ((lin(x) - t) ** 2).sum()
loss.backward()
```

```
loss     : 0.16000000000000003
grad W   : [[-0.4, 0.8]]
grad b   : [-0.8]
2(y - t) : -0.8
```

Here $`y = 0.1\cdot0.5 + 0.2\cdot(-1) = -0.15`, so $`y-t=-0.4`, the loss is $`0.16`, the bias
gradient is $`-0.8`, and the weight gradient is $`-0.8\cdot(0.5,-1)=(-0.4,0.8)`. Same three
relations, different initialization. The trailing `3` in `0.16000000000000003` is the binary64
rounding of $`(-0.4)^2` showing through PyTorch's round-trip display.

The two gradient entry points differ in what they differentiate:

- `autograd.grad` differentiates a scalar tensor function with respect to its input tensor;
- `autograd.model.grad` differentiates a loss with respect to the model state;
- `(value := true)` also returns the scalar value from the same evaluation, so a training loop does
  not pay for a second forward pass to report the loss it just differentiated.

Training normally calls the second path through `Trainer`, which manages the differentiation
invocation for each update.

A state gradient is useful because an optimizer must update the same collection of tensors
that produced the prediction. The weight row and bias vector have different shapes and different
roles; concatenating their displayed numbers would lose that information. The typed state keeps
the pairing available through differentiation and the subsequent update. Input differentiation
instead treats those parameters as fixed and asks how the model responds to a changed example.
Both are chain-rule computations, but they answer different application questions.

## Higher-Order And Directional Transforms

```terminal
# Inspect Jacobian, Hessian, directional, and detached-state
# results together.
lake exe torchlean autograd_transforms
```

```terminal +output
Jacobian rows of x^2:
  [1.000000, 0.000000]
  [0.000000, -2.400000]
Hessian columns of mean(x^2):
  [1.000000, 0.000000]
  [0.000000, 1.000000]
loss directional derivative = 0.011629
loss Hessian-vector product =
  [[3, 2]: [[0.010000, -0.024000], [0.010000, -0.024000],
    [0.010000, -0.024000]], [3]: [0.020000, 0.020000, 0.020000]]
state gradient after detaching the model output =
  [[3, 2]: [[0.000000, 0.000000], [0.000000, 0.000000],
    [0.000000, 0.000000]], [3]: [0.000000, 0.000000, 0.000000]]
```

The last two entries are printed on one line each by the command and were rewrapped to fit here.

The Jacobian of $`x\mapsto x^2` is diagonal with entries $`2x_i`, so the rows above say the input
was $`(0.5,-1.2)`. The Hessian of $`\tfrac1n\sum x_i^2` is $`\tfrac2n I`, which at $`n=2` is the
identity at every input, so these columns do not depend on the evaluation point.

After detaching the model output, every component of the state gradient
is `0.000000`, not merely small. Stop-gradient is not an approximation: the detached subgraph
contributes nothing to the reverse pass, so the result is a structural zero rather than a
cancellation that happened to work out. That is the mechanism behind target networks in
reinforcement learning and behind the stop-gradient branches in self-supervised objectives.

The model is a single affine layer with three outputs and mean squared error. Its probe sets every
weight and bias perturbation to 0.1. With input $`x=(0.5,-1.2)`, each output perturbation is
$`0.1(0.5-1.2)+0.1=0.03`. The loss averages three squared residuals, so the bias component of the
Hessian-vector product is $`(2/3)\cdot0.03=0.02`; multiplying by $`x` gives the weight row
$`(0.01,-0.024)`. All three rows agree because the affine outputs use the same input, loss
weighting, and probe. Sharing an input alone would not imply this for a different loss or model.

Sources:
{src "NN/Examples/Quickstart/AutogradBasics.lean"}[`AutogradBasics.lean`]
and
{src "NN/Examples/DeepDives/AutogradTransforms.lean"}[`AutogradTransforms.lean`].
These executions are useful checks, not theorems that every runtime derivative agrees with real
calculus; {ref "autograd-walkthrough"}[the autograd chapter] is where that question is taken up.

The directional derivative and Hessian-vector product answer questions that a full matrix dump
can obscure. The first gives the local loss change along one chosen parameter perturbation; the
second gives how the gradient changes along that perturbation. Their tensor layouts still match
the parameter state, so the three weight rows and three biases remain distinguishable. A zero
Hessian-vector entry would concern curvature in that direction, whereas the detached gradients
above are zero because a computation path has been cut. Reading the operation that produced a
zero is as important as reading its printed value.

# MLP Training

The training quickstart generates 25 regression samples for a two-input, one-output function and
uses a hidden layer of width eight.

```terminal
# Keep the seed fixed for a short initial training trace.
lake exe torchlean quickstart_mlp \
  --device cpu --steps 20 --seed 2026
```

The command reports two statistics that measure different things. The step loss concerns
the current streamed sample; `mean_loss` covers all 25. A per-sample loss can be far below the
dataset mean simply because that sample was easy, which is why both are labelled rather than
reported as one number called "loss".

Compare the held-out prediction before and after training with its target, $`0.2`. This shows
the direction and size of the change at one point. It does not establish whether the optimizer
has converged.

To examine a larger update budget, keep the seed and increase the step count:

```terminal
# Change only the update budget to compare with the
# twenty-step run.
lake exe torchlean quickstart_mlp \
  --device cpu --steps 200 --seed 2026
```

Each step loss is measured on the sample used at that step, so this column does not track a fixed
objective on a fixed input. The before-and-after `mean_loss` values evaluate the whole dataset.
Even that average is not guaranteed to decrease without assumptions on the objective, sampling,
and step size. A lower mean loss would show improvement on this grid; a prediction closer to
$`0.2` would show improvement at the held-out point. Neither observation supplies a convergence
theorem or a generalization bound.

## Training With Executable Binary32

The quickstart is arithmetic-polymorphic, so the same code runs on the executable binary32 model:

```terminal
# Compare native and configured binary32 on the same two-update workload.
lake exe torchlean quickstart_mlp \
  --device cpu --arithmetic native --steps 2 --seed 2026
lake exe torchlean quickstart_mlp \
  --device cpu --arithmetic ieee --steps 2 --seed 2026
```

The second command selects FloatLib's `ExecFloat.Binary 8 23` for the forward pass, gradient
calculation, and optimizer update. Compare the initial loss, final loss, and held-out prediction
with the native run. Equality of decimal displays alone does not establish bitwise agreement.
The {ref "floats"}[floating-point chapter] follows the rounding steps behind these comparisons.

Not every command is arithmetic-polymorphic. Command-specific validation rejects an unsupported
combination with an error, preserving the arithmetic requested by the caller.

Source:
{src "NN/Examples/Quickstart/SimpleMlpTrain.lean"}[`SimpleMlpTrain.lean`].

The synthetic target is chosen to fit the model family: it is a weighted combination of two
ReLU features, $`\operatorname{relu}(x_1+x_2)` and
$`\operatorname{relu}(x_2-x_1)`, plus a constant. A width-eight hidden layer can represent those
features, so this exercise does not force the optimizer to approximate a smooth curve with an
undersized piecewise-linear model. The twenty-five training points form a five-by-five grid.
The held-out point lies between grid locations, making its prediction a separate observation
from the mean training loss.

# Convolutional Model Training

Prepare the small CIFAR-10 fixture and run one CPU optimizer step:

```terminal
# Prepare CIFAR arrays before running the cropped-image
# classifier.
python3 scripts/datasets/download_example_data.py --cifar10
lake exe torchlean cnn --device cpu --n-total 1 --steps 1 --seed 2026
```

The starting loss can be compared with a reference value. A uniform prediction on ten classes
has cross-entropy $`\ln 10`:

```lean (name := exLn10)
-- A uniform prediction assigns probability one tenth to the
-- target class.
#eval Float.log 10.0
```
```leanOutput exLn10
2.302585
```

A loss close to $`\ln 10` on one sample does not establish that the full prediction vector is near
uniform. Very small or large losses are
reasons to inspect the logits, labels, and objective, rather than proof of a particular bug.

`dataset size = 1` counts the training dataset's minibatches. The current CNN command fixes the
batch size at one, loads a shuffled epoch of full batches, and crops each image to $`8\times8`.
Thus this particular run has one image in one batch. Increasing `--n-total` supplies more rows
and therefore more one-image batches; `--steps` independently sets how many optimizer updates
run. The input contract is `[1, 3, 8, 8]`, and the output `[1, 10]` contains one class-logit row.

The command defines one public `nn.models.CNN.Config`, derives the checked input and output shapes
from that value, and passes it to `nn.models.cnn`. It then uses the shared NPY data boundary and
`Trainer.new` with `trainer.train`. `--n-total 1` keeps this a check of that training path; it does
not establish useful CIFAR accuracy. Compare the reported loss before and after the update to
check its effect on this sample.

Source:
{src "NN/Examples/Models/Vision/Cnn.lean"}[`Cnn.lean`].

Convolution changes the parameter-sharing pattern from the regression MLP. A small kernel is
reused at spatial positions within each image, so nearby patches are processed by the same learned
weights. The cropped input is chosen to exercise this spatial path and its gradients with a small
workload. Image classification quality would need held-out images and an accuracy evaluation
beyond this one-step training check.

# Model Lowering And Interval Bounds

The predictions and gradients above concern particular inputs. To ask whether an output stays
within a bound throughout a region, we need to account for every input in that region.
Interval bound propagation does this by carrying an enclosure through the model. Start from
an input box

$$`x\in[\ell,u]`

and push the box through the network, one operation at a time, keeping an enclosure at every step
{Informal.citep gowal2018}[].

For an affine layer $`y=Wx+b`, the rule separates positive and negative weights, because a negative
coefficient maps the lower endpoint to the upper one:

$$`\begin{aligned}
\ell'_i
&=\sum_j
  \left(\max(W_{ij},0)\ell_j+\min(W_{ij},0)u_j\right)+b_i,\\
u'_i
&=\sum_j
  \left(\max(W_{ij},0)u_j+\min(W_{ij},0)\ell_j\right)+b_i.
\end{aligned}`

For ReLU the rule is simpler, because the function is monotone:

$$`[\ell_i,u_i]
\longmapsto
[\max(0,\ell_i),\max(0,u_i)].`

## Interval Bound Calculation

The registered workflow uses a fixed two-layer model, and its constants are small enough to write
out in full. They come straight from
{src "NN/Verification/Builtin/IBPWorkflow.lean"}[`IBPWorkflow.lean`]:

```lean (name := exIbpModel)
-- Use the workflow’s fixed parameters and a radius around
-- its two-dimensional center.
def exW1 : Tensor Float [3, 2] :=
  [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
def exB1 : Tensor Float [3] := [0.1, 0.2, 0.3]
def exW2 : Tensor Float [1, 3] := [[0.7, 0.8, 0.9]]
def exB2 : Float := 0.4
def exCenter : Tensor Float [2] := [0.5, 0.8]
def exEps : Float := 0.1
```

One function implements the affine rule above, once, for any width:

```lean (name := exIbpRow)
/-- One affine row under interval arithmetic. A positive
weight pairs each endpoint with itself; a negative weight
swaps them. -/
def exRow {n : Nat} (w : Fin n → Float) (bias : Float)
    (lo hi : Fin n → Float) : Float × Float :=
  (List.finRange n).foldl
    (fun acc j =>
      let wj := w j
      if wj ≥ 0.0 then
        (acc.fst + wj * lo j, acc.snd + wj * hi j)
      else
        (acc.fst + wj * hi j, acc.snd + wj * lo j))
    (bias, bias)
```

Every weight in this model is positive and every hidden lower bound will turn out to be positive
too, so the ReLU layer acts as the identity and we can chain the two affine layers directly:

```lean (name := exIbpRun)
-- Propagate each hidden interval before combining the three
-- output contributions.
#eval show IO Unit from do
  let xLo := fun j => Tensor.at exCenter (j, ()) - exEps
  let xHi := fun j => Tensor.at exCenter (j, ()) + exEps
  let hidden := (List.finRange 3).map fun i =>
    exRow (fun j => Tensor.at exW1 (i, (j, ())))
      (Tensor.at exB1 (i, ())) xLo xHi
  for (box, i) in hidden.zipIdx do
    IO.println s!"h{i + 1} = ({box.fst}, {box.snd})"
  let y := exRow (fun i => Tensor.at exW2 (0, (i, ()))) exB2
    (fun i => (hidden[i]!).fst) (fun i => (hidden[i]!).snd)
  IO.println s!"y   = ({y.fst}, {y.snd})"
  IO.println s!"mid = {0.5 * (y.fst + y.snd)}"
```
```leanOutput exIbpRun
h1 = (0.280000, 0.340000)
h2 = (0.600000, 0.740000)
h3 = (0.920000, 1.140000)
y   = (1.904000, 2.256000)
mid = 2.080000
```

All three hidden lower bounds are positive, which retroactively justifies dropping the ReLU. That
also permits a stronger calculation here: all weights are nonnegative, so the lower and upper
input corners attain the extrema. With these decimal constants interpreted as exact reals,
$`[1.904, 2.256]` is the exact range. Stable ReLUs alone would not make multilayer IBP exact: mixed
signs can still lose dependencies. The binary64 calculation above illustrates the formula; it does
not use outward rounding and is not itself a certified enclosure. IBP is usually conservative;
here the nonnegative weights let the same pair of input corners attain all the intermediate
extrema.

For the first hidden unit, the lower input corner is $`(0.4,0.7)`, giving
$`0.1\cdot0.4+0.2\cdot0.7+0.1=0.28`. The upper corner gives $`0.34`.
Repeating that calculation for all three rows produces the displayed hidden intervals. Their
midpoints are the hidden activations at the center because this whole region stays within one
linear piece of the ReLU network. If a hidden interval crossed zero, the active linear expression
could change within the box and that midpoint argument would need to be reconsidered.

## IBP Workflow

```terminal
# Lower the fixed model and propagate its input box with
# native bound arithmetic.
lake exe verify -- torchlean-ibp
```

Compare the reported endpoints with the hand calculation. The workflow uses outward rounding,
so a wider result can reflect the numerical enclosure policy.

A lower endpoint rounded upward or an upper endpoint rounded downward can exclude the exact
result. Interval operations must account for that error. The native path uses
`HostFloat32.nextUp` and `HostFloat32.nextDown` from
{srcDir "NN/MLTheory/CROWN"}[`NN/MLTheory/CROWN`] to widen ordinary binary32 results by one
representable step. The following calculation measures the size of such a step near the final
upper endpoint; it does not count the widenings or derive their accumulated effect:

```lean (name := exUlp)
-- Measure representable spacing near the upper endpoint, in
-- billionths of a unit.
open NN.MLTheory.CROWN in
#eval show IO Unit from do
  let top : Float32 := 2.256
  let step := (HostFloat32.nextUp top).toFloat - top.toFloat
  IO.println s!"one ulp   = {1000000000.0 * step} nano"
  IO.println s!"four ulps = {4000000000.0 * step} nano"
```
```leanOutput exUlp
one ulp   = 238.418579 nano
four ulps = 953.674316 nano
```

Four ulps at this magnitude are just under one millionth. To compare a different rounding
implementation, select FloatLib arithmetic for the same model and box:

```terminal
# Select directed reference arithmetic for the same lowered
# model and box.
lake exe verify -- torchlean-ibp --arithmetic ieee
```

FloatLib exposes directed addition and multiplication through `ExecFloat.Binary.addWithRounding`
and `ExecFloat.Binary.mulWithRounding` with `.towardPositiveInfinity`. On the finite real branch,
these choose the smallest representable upper result, rather than always widening a nearest-rounded
result. The interval rules also account for exceptional endpoints. The native path widens a
nearest-rounded result, while the reference path computes directed rounding in Lean. The arithmetic
banner identifies which policy produced the box.

The lowering path is:

```
TorchLean model
  -> canonical IR
  -> supported-operation check
  -> interval propagation
  -> output box
```

`lowered IR nodes: 18` counts the canonical IR after lowering, not the layers of the model. A
two-layer MLP expands into more nodes than one might guess because reshapes, broadcasts, and the
bias additions are all explicit at that level. The supported-operation check inspects these IR
operations.

The result applies to this supported IR fragment and numerical policy. A different model command
needs its own lowering and supported-operation checks. IBP propagates intervals without auxiliary
linear coefficients, losing dependencies between intermediate values. CROWN-style relaxations
{Informal.citep crown2018}[] can retain more of that information and produce tighter bounds for
suitable networks and relaxations; there is no unconditional precision ordering for arbitrary
implementations. Related methods appear in {Informal.citet autolirpa2020}[] and
{ref "verification"}[the verification chapter].

The other registered workflows, from `lirpa-cnn` through the two-stage Lyapunov
refinements, are listed by:

```terminal
# Ask the verification dispatcher which workflows it
# currently registers.
lake exe verify -- list
```

The dispatcher is reachable from {src "NN/Verification/CLI.lean"}[`NN/Verification/CLI.lean`].

An output box becomes useful only after it is compared with the application's property.
For this scalar example, the lower endpoint can establish a threshold claim that lies below it,
while a threshold inside the interval remains undecided by this enclosure alone. A wider box
can therefore be numerically sound but insufficient for the requested property. This is the
practical reason to inspect both supported operations and bound precision before treating
successful propagation as successful verification.

# Numerical Certificate Replay

We can also record intermediate ranges in a numerical certificate and replay it with a checker
{Informal.citep necula1997}[].

First, evaluate the same small MLP at the center of its input box and compare its native and
reference gradients. We already have the weights and interval calculation needed to interpret
the result:

```terminal
# Compare the fixed MLP’s native and reference outputs and
# reverse-mode gradients.
lake exe torchlean float32_semantics
```

Over the exact reals, $`y=2.08` is the midpoint of the interval calculated above, because the same
parameters are evaluated at the center of the same input region. The gradients are also checkable
by hand. The scalar being differentiated is the output itself, so the cotangent is one, and the
reverse pass reduces to three products:

$$`\frac{\partial y}{\partial W_2}=h,
\qquad
\frac{\partial y}{\partial b_1}=W_2^{\top},
\qquad
\frac{\partial y}{\partial x}=W_2W_1.`

```lean (name := exFpIdent)
-- Reconstruct the three gradient factors using the already
-- declared model constants.
#eval show IO Unit from do
  let x := fun j => Tensor.at exCenter (j, ())
  let w2 := fun i => Tensor.at exW2 (0, (i, ()))
  let hidden := (List.finRange 3).map fun i =>
    (List.finRange 2).foldl
      (fun acc j => acc + Tensor.at exW1 (i, (j, ())) * x j)
      (Tensor.at exB1 (i, ()))
  let dInput := (List.finRange 2).map fun j =>
    (List.finRange 3).foldl
      (fun acc i =>
        acc + w2 i * Tensor.at exW1 (i, (j, ())))
      0.0
  IO.println s!"h      = {hidden}"
  IO.println s!"W2     = {(List.finRange 3).map w2}"
  IO.println s!"W2 W1  = {dInput}"
```
```leanOutput exFpIdent
h      = [0.310000, 0.670000, 1.030000]
W2     = [0.700000, 0.800000, 0.900000]
W2 W1  = [0.760000, 1.000000]
```

These three calculations correspond to `outputWeightGrad`, `hiddenBiasGrad`, and `inputGrad`
in the command's output. The `hiddenWeightGrad` row is the outer product $`W_2^{\top}x^{\top}`,
whose first entry is $`0.7\times0.5=0.35`. Comparing these factors with the computed gradients
can catch transposed matrices and dropped chain-rule factors; the checked calculation here uses
`Float`, so it does not prescribe every bit of a binary32 result.

The command reports the maximum absolute difference across the output and five gradient tensors,
after converting both implementations' results to `Float`. Inspect that number: completion alone
does not assert that the difference is zero. For finite values, zero would establish numerical
agreement on these tensors, without distinguishing the signs of zero or inspecting intermediate
rounding decisions.
The command also contrasts the proof-only rounded-real model, which is selected in theorem
statements and never executed, with those two executable paths; the distinction is developed in
{ref "floats"}[the floating-point chapter] and has the same shape as the separation between
the real and the float models in {Informal.citet flocq2011}[].

The input-gradient row offers another connection to the interval calculation. Within this
stable ReLU region, changing the two inputs by $`(\delta_1,\delta_2)` changes the exact-real
output by $`0.76\delta_1+\delta_2`. A radius of $`0.1` in each coordinate therefore gives a
maximum deviation of $`0.176`, yielding $`2.08\pm0.176`. The derivative calculation and the
interval calculation meet here because this example is affine throughout the region; a local
gradient would not bound a region crossing an activation change.

## Certificate Generation And Replay

```terminal
# Generate certificates, replay concrete inputs, and reject
# a tampered interval.
lake exe torchlean numerical_certificate
```

The command generates and replays a small scalar graph, checks that certificate validation rejects
an altered addition interval, and applies the same machinery to a ten-node MLP. A failed check
returns a nonzero exit code. These checks exercise artifact generation and execution; they do not
supply a proof of enclosure for every real input.

The checker and its proof layer are:

- {src "NN/Examples/DeepDives/Floats/GraphNumericalCertificate.lean"}[
  `GraphNumericalCertificate.lean`];
- {src "NN/Proofs/RuntimeApprox/Graph/NumericalCertificate.lean"}[
  `NN.Proofs.RuntimeApprox.Graph.NumericalCertificate`].

Certificate ranges are outward-rounded FloatLib binary32, and concrete inputs are replayed in the
bit-level interpreter. A native runtime is a separate provider. Comparing its outputs and
gradients with FloatLib on one trajectory does not prove agreement on every execution.

Replaying a certificate also separates the producer's work from the checker's decision. The
producer can supply proposed intermediate ranges, but acceptance depends on checking them against
the graph and numerical operations. Altering a range is therefore a useful negative probe: the
checker must reject an artifact that no longer satisfies its conditions even if its format still
parses. The rejection check concerns that altered artifact; the proof layer states the general
conditions under which the checker can justify a numerical claim.

# PyTorch Graph Import

The last lab imports PyTorch models into canonical IR and compares their outputs on deterministic
inputs:

```terminal
# Exercise external graph capture, parsing, and numerical
# comparison on fixed probes.
lake exe pytorch_export_check
```

```terminal +output
== PyTorch nn.Module → TorchLean IR runtime check ==
generated Python float bit patterns and tiny epsilon: ok
generated reference code and state-dict round trip: ok
  numerical parity: ok (TinyMLP)
  numerical parity: ok (TinyAffineLayerNorm)
  numerical parity: ok (TinyLayerNormEpsilon)
  numerical parity: ok (TinyBatchNormEpsilon)
pytorch_export_check: ok
```

The four numerical probes cover an MLP, affine LayerNorm, and small-epsilon LayerNorm and
BatchNorm. The generated-reference checks also exercise scalar expression emission and state-dict
naming and orientation. These are small interoperability checks, not an exhaustive importer suite.

The import path is:

```
Python/PyTorch producer
  -> JSON value graph
  -> Lean parser
  -> canonical TorchLean IR
  -> WellShaped result or explicit rejection
```

`parseGraph_wellShaped` establishes the structural property of a successful parse. Numerical
agreement is checked separately: a graph with correct shapes can still compute the wrong function.
Neither result proves the Python exporter correct or establishes agreement on untested inputs.
Unsupported operations and malformed artifacts must be rejected at the capture or import boundary;
see {ref "pytorch-roundtrip"}[the round-trip chapter] for the supported schema and its limits.

Source: {src "NN/Tests/Interop/PyTorch.lean"}[`PyTorch.lean`].

The normalization probes were chosen because seemingly small import details change the
function: an affine scale and bias must retain their orientation, and epsilon belongs in a
particular place in the normalization denominator. An imported graph can preserve every tensor
shape while losing one of those details. Comparing deterministic outputs complements structural
validation by making such semantic changes observable on the selected inputs. A new supported
operator needs the same kind of account of its attributes, state, and numerical convention.

# Building The Example Suite

Before changing maintained examples, build the curated umbrella:

```terminal
# Elaborate the curated example targets before running their
# entry points.
lake build NNExamples
```

That checks the maintained Lean example targets. Elaboration is
not execution, so runtime behavior is checked separately:

```terminal
# Execute the retained suite; the final command runs the
# CUDA-enabled build.
lake exe nn_tests_suite
lake -R -K cuda=true build nn_tests_suite
lake env .lake/build/bin/nn_tests_suite
```

The maintained suite concentrates on numerical and native-boundary behavior: attention,
convolution, pooling, reductions, gather and scatter, FFT, matrix multiplication, and buffer
lifetime. For an example change, run that command with a small input and check its output;
routine command and API refactors use temporary checks rather than a permanent all-command suite.
Optional ALE/Pong and documentation rendering require their own external environment.

Runnable examples use public executable APIs such as `Tensor.qr`, `Tensor.cholesky`, `nn.linear`,
and `Trainer.train`. `Spec.*` references remain only where an example is explicitly demonstrating a
mathematical specification or theorem; no runtime example fabricates a result by evaluating a
placeholder in place of the public execution path. Many `Spec` definitions are executable reference
functions, but running them does not test the selected backend implementation.

Command implementations keep their local configuration vocabulary short because the namespace
already supplies the command name: `Options` for parsed flags and `Preset` for a named model
setup. Data prepared for one command likewise uses role names such as `Splits` or `Evaluation`,
rather than repeating the command name in every type.

# Example Coverage

When reusing an example, carry its evidence with it. The value and gradient transcripts record
particular executions: `mean(x^2)` having the right gradient at $`(1,2,3)` is a useful check of
autodiff, but a theorem must cover its stated domain. The reshape example illustrates a different
condition: the API requires a proof that the element counts agree, and Lean rejects the attempted
four-to-nine reshape because that proposition is false.

The interval workflow covers an input region within its supported IR fragment and numerical
policy. Certificate replay checks an artifact against the checker's conditions, including a
deliberately altered range that it rejects. The import checks combine a structural guarantee with
output comparisons on fixed probes; correctness of the Python exporter remains a separate
obligation.

The chapters that carry those claims further are {ref "verification"}[verification],
{ref "floats"}[floating point], and {ref "autograd-walkthrough"}[autograd].

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
import TorchLean.Verify.CROWN.BoundOps
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles
import TorchLean.NN.Loss
import TorchLean.NN.Session.Random
import TorchLean.Verify.Trainer

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Examples" =>
%%%
tag := "examples"
file := "Worked-Examples"
%%%

Two tensors can print the same decimal digits and still contain different numbers. We start there,
with four entries whose sums we can check by hand. Keeping the calculations small also lets us
follow a gradient back to its factors and an interval bound back to its endpoints, and when the
examples reach training and imported graphs, those calculations give us something concrete to
compare with the command output.

Every Lean block on this page is elaborated when the site is built, and its output is checked
against the accompanying expected output. Shell and Python commands let you compare the
calculations with your executable and external dependencies.

# Tensor Shapes And Scalar Types

A `Tensor α s` carries its shape `s` in its type and its arithmetic in its element type `α`. Both
choices are visible in the first quickstart. A tensor's shape controls which operations can
compose, while its element type determines how their arithmetic is evaluated.

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

A rank-zero tensor has shape `[]`, not a special scalar type. We made that choice on purpose: a
loss value and a batch of images then have the same type constructor, so `Trainer` doesn't need a
separate code path for the scalar it minimizes.

Writing `0.1` at `ℚ` produces the exact number $`\tfrac1{10}`, and the display says so. Writing
`0.1` at `Float` produces the nearest binary64 value, which is not $`\tfrac1{10}`
{Informal.citep goldberg1991}[]. The Float row still prints `0.100000`, because six decimal places
can't show the difference.

A scalar loss still has one stored value: an empty list of axes has size one. By contrast, a shape
such as `[0]` has an axis with no entries. Keeping those cases separate matters when reductions or
empty batches are involved. So the rank-zero threshold above is an ordinary input to scalar tensor
operations, while the four-element examples let us inspect how a reduction combines several stored
values into that same scalar shape.

## Floating-Point And Rational Arithmetic

```lean (name := exExact)
-- Test equality separately from the rounded tensor display.
#eval show IO Unit from do
  let floats : Tensor Float [4] := [0.1, 0.2, 0.3, 0.4]
  let rationals : Tensor ℚ [4] := [0.1, 0.2, 0.3, 0.4]
  let floatSum := Tensor.sum floats
  IO.println s!"Float sum   = {floatSum}"
  IO.println s!"Rat sum     = {Tensor.sum rationals}"
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

Those equality tests distinguish values that the six-decimal display hides. `0.1 + 0.2` isn't `0.3`
in binary64. But summing all four tenths *does* give exactly `1.0`, because the rounding errors
happen to cancel along that particular left fold. Python binary64 arithmetic gives the same answers
for this explicitly ordered expression:

```
# Preserve the addition order used by the four-element
# reduction.
print(0.1 + 0.2 == 0.3, 0.1 + 0.2 + 0.3 + 0.4 == 1.0)
```

```
False True
```

Printed digits alone can't establish equality. TorchLean keeps `Float`, `Float32`, FloatLib
binary32, `ℚ`, and $`\mathbb R` as separate element types so that definitions and theorems identify
the arithmetic they use. See {ref "floats"}[the floating-point chapter] and
{Informal.citet boldo2015}[] for the same separation in Coq.

The rational calculation is a useful reference because its result doesn't depend on a rounding
policy, but it doesn't replace checking the requested runtime arithmetic. If the application stores
binary32 weights, a theorem about the corresponding rational expression still needs a connection to
conversion and rounding. I use these decimals because we can keep their exact meanings in view
while comparing the two arithmetic paths.

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

In that error message, the second half identifies the failed proposition and how it was decided.

That equality ensures that reshape preserves the element count and linear element order, and
stating it as a proof obligation erases the size check at runtime. Callers still need to choose the
intended axis order. Whether storage is shared or copied belongs to the implementation boundary.

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

In PyTorch, the addition fails when executed {Informal.citep pytorch2019}[], while the Lean reshape
above fails during elaboration because its size-equality obligation is false. These checks
establish specific shape requirements, but neither establishes that the program implements the
intended model.

{src "TorchLeanExamples/Quickstart/Tensors.lean"}[`Tensors.lean`] is the command that
prints the tensors above:

```terminal
# Print the standalone tensor quickstart, including its
# explicit Float32 cast.
scripts/lake.sh exe torchlean quickstart_tensors
```

```terminal +output
== Quickstart: tensor basics ==
Rank-zero tensor: 0.500000
Float tensor:    [0.100000, 0.200000, 0.300000, 0.400000]
Rational tensor: [(1 : Rat)/10, (1 : Rat)/5, (3 : Rat)/10, (2 : Rat)/5]
Integer tensor:  [1, 2, 3, 4]
Float32 cast:    [0.100000, 0.200000, 0.300000, 0.400000]
Rank-3 tensor:   [[[1.000000, 2.000000], [3.000000, 4.000000]], ...
Reshaped matrix: [[0.100000, 0.200000], [0.300000, 0.400000]]
```

We truncated the rank-3 line here to fit the page, but it's the same value the block above printed.
Read the source beside the output, change one element type at a time, and let Lean show which
operations need a different algebraic context.

In the cube and the matrix, shape is also separate from meaning. The same four values can represent
two points with two features each, or a two-by-two image. Lean checks the chosen dimensions, but
the program supplies their interpretation: in later examples a leading axis means a batch, a
trailing axis means features or classes, and transposing them changes the computation even when
their sizes happen to agree. Naming those roles beside the values is part of specifying the model.

# Function And Model Differentiation

To compute a gradient, we first choose what may vary. `autograd.grad` takes the input tensor as
that variable, while `autograd.model.grad` takes the model's typed state. We can see the difference
by differentiating a mean of squares, then a loss for one affine layer.

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

Those extra digits come from the display, not the arithmetic: PyTorch prints a shortest round-trip
representation of binary64, and TorchLean's tensor display rounds to six decimals.

Here the input gradient has shape `[3]` because it answers one question per input coordinate: how
the scalar mean would change if that coordinate moved. Since the division by three belongs to the
objective, it must appear in every coordinate derivative. Replacing the mean with a sum would leave
the intermediate squared tensor unchanged and triple the returned gradient. This small example
makes a missing reduction factor visible without depending on parameter initialization or a
training loop.

## Differentiating A Model

The second path takes a model and a loss, and returns a gradient shaped like the model's state:

```lean (name := exModelGrad)
-- The differentiation variable is the typed weight-and-bias
-- state.
def exModel : nn.Sequential [2] [1] :=
  nn.build 0 (nn.linear 2 1)

#eval show IO Unit from do
  let state : autograd.model.State exModel Float :=
    nn.initialState exModel
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

That gradient prints with its shapes, `[1, 2]` for the weight and `[1]` for the bias, which is the
same list `nn.Sequential.stateShapes` reports for this model. Their values follow from the chain
rule. For $`y=Wx+b` and a squared-error loss on one output,

$$`\frac{\partial L}{\partial b}=2(y-t),
\qquad
\frac{\partial L}{\partial W}=2(y-t)\,x^{\top},
\qquad
L=(y-t)^2.`

So the bias gradient should be twice the residual, and the weight gradient should be that same
number times the input $`(0.5,-1)`. Dividing the printed bias gradient by two gives a residual
of about $`-0.8774325`: the untrained prediction is below the target. Its square is about
$`0.769887`, matching the displayed loss. Multiplying the bias gradient by $`0.5` and $`-1`
recovers the two weight entries, to the displayed precision.

You can check the same identity in PyTorch with explicitly chosen parameters:

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
print("loss     :", loss.item())
print("grad W   :", lin.weight.grad.tolist())
print("grad b   :", lin.bias.grad.tolist())
print("2(y - t) :", (2 * (lin(x) - t)).item())
```

```
loss     : 0.16000000000000003
grad W   : [[-0.4, 0.8]]
grad b   : [-0.8]
2(y - t) : -0.8
```

Here $`y = 0.1\cdot0.5 + 0.2\cdot(-1) = -0.15`, so $`y-t=-0.4`, the loss is $`0.16`, the bias
gradient is $`-0.8`, and the weight gradient is $`-0.8\cdot(0.5,-1)=(-0.4,0.8)`. Same three
relations, different initialization. In `0.16000000000000003`, the trailing `3` is the binary64
rounding of $`(-0.4)^2` showing through PyTorch's round-trip display.

TorchLean's two gradient entry points differ in what they differentiate:

- `autograd.grad` differentiates a scalar tensor function with respect to its input tensor.
- `autograd.model.grad` differentiates a loss with respect to the model state.
- `(value := true)` also returns the scalar value from the same evaluation, so a training loop
  doesn't pay for a second forward pass to report the loss it just differentiated.

Training normally calls the second path through `Trainer`, which manages the differentiation
invocation for each update.

In the state gradient, each tensor stays paired with its parameter, including its shape, so an
optimizer can update the weights and biases without flattening them into one untyped collection.

## Higher-Order And Directional Transforms

```terminal
# Inspect Jacobian, Hessian, directional, and detached-state
# results together.
scripts/lake.sh exe torchlean autograd_transforms
```

```terminal +output
Jacobian of x^2: [[1.000000, 0.000000], [0.000000, -2.400000]]
Hessian of mean(x^2): [[1.000000, 0.000000], [0.000000, 1.000000]]
loss directional derivative = 0.011629
loss Hessian-vector product =
  [[3, 2]: [[0.010000, -0.024000], [0.010000, -0.024000],
    [0.010000, -0.024000]], [3]: [0.020000, 0.020000, 0.020000]]
state gradient after detaching the model output =
  [[3, 2]: [[0.000000, 0.000000], [0.000000, 0.000000],
    [0.000000, 0.000000]], [3]: [0.000000, 0.000000, 0.000000]]
mixed input derivative of tanh at zero: [0.000000, 0.000000] (expected [0, 0])
third input derivative along [1, 0]: [-2.000000, 0.000000] (expected [-2, 0])
```

We rewrapped the Hessian-vector product and the detached state gradient to fit here (the command
prints each on one line).

For $`x\mapsto x^2` the Jacobian is diagonal with entries $`2x_i`, so the rows above say the input
was $`(0.5,-1.2)`. And the Hessian of $`\tfrac1n\sum x_i^2` is $`\tfrac2n I`, which at $`n=2` is
the identity at every input, so these columns don't depend on the evaluation point.

After detaching the model output, every component of the state gradient
is `0.000000`, not merely small. Stop-gradient is not an approximation: the detached subgraph
contributes nothing to the reverse pass, so the result is a structural zero rather than a
cancellation that happened to work out. That is the mechanism behind target networks in
reinforcement learning and behind the stop-gradient branches in self-supervised objectives.

Here the model is a single affine layer with three outputs and mean squared error. Its probe sets
every weight and bias perturbation to 0.1. With input $`x=(0.5,-1.2)`, each output perturbation is
$`0.1(0.5-1.2)+0.1=0.03`. Since the loss averages three squared residuals, the bias component of
the Hessian-vector product is $`(2/3)\cdot0.03=0.02`, and multiplying by $`x` gives the weight row
$`(0.01,-0.024)`. All three rows agree because the affine outputs use the same input, loss
weighting, and probe. Sharing an input alone wouldn't imply this for a different loss or model.

Sources: {src "TorchLeanExamples/Quickstart/Autograd.lean"}[`Autograd.lean`] and
{src "TorchLeanExamples/DeepDives/AutogradTransforms.lean"}[`AutogradTransforms.lean`]. These
executions are useful checks, not theorems that every runtime derivative agrees with real calculus
({ref "autograd-walkthrough"}[the autograd chapter] takes up that question).

The directional derivative measures the local loss change along the chosen perturbation, while the
Hessian-vector product measures how the gradient changes. A zero in either result concerns that
direction, whereas detaching cuts a computation out of the reverse pass.

# MLP Training

The training quickstart generates 25 regression samples for a two-input, one-output function and
uses a hidden layer of width eight.

```terminal
# Keep the seed fixed for a short initial training trace.
scripts/lake.sh exe torchlean quickstart_mlp \
  --device cpu --steps 20 --seed 2026
```

That command reports two statistics that measure different things. Its step loss concerns the
sample streamed at that step, while `mean_loss` covers all 25. A per-sample loss can be far below
the dataset mean just because that sample was easy, which is why the command labels both instead of
reporting one number called "loss".

Compare the held-out prediction before and after training with its target, $`0.2`. That comparison
shows the direction and size of the change at one point. It doesn't establish whether the optimizer
has converged.

To examine a larger update budget, keep the seed and increase the step count:

```terminal
# Change only the update budget to compare with the
# twenty-step run.
scripts/lake.sh exe torchlean quickstart_mlp \
  --device cpu --steps 200 --seed 2026
```

Each step loss is measured on the sample used at that step, so this column doesn't track a fixed
objective on a fixed input. The before-and-after `mean_loss` values evaluate the whole dataset, but
even that average isn't guaranteed to decrease without assumptions on the objective, sampling, and
step size. A lower mean loss would show improvement on this grid and a prediction closer to $`0.2`
would show improvement at the held-out point, but neither observation supplies a convergence
theorem or a generalization bound.

## Training With Executable Binary32

Because the quickstart is arithmetic-polymorphic, the same code runs on the executable binary32
model:

```terminal
# Compare native and configured binary32 on the same two-update workload.
scripts/lake.sh exe torchlean quickstart_mlp \
  --device cpu --arithmetic native --steps 2 --seed 2026
scripts/lake.sh exe torchlean quickstart_mlp \
  --device cpu --arithmetic ieee --steps 2 --seed 2026
```

Of the two commands, the second selects FloatLib's `ExecFloat.Binary 8 23` for the forward pass,
gradient calculation, and optimizer update. Compare the initial loss, final loss, and held-out
prediction with the native run. Equality of decimal displays alone doesn't establish bitwise
agreement. For the rounding steps behind these comparisons, see the {ref "floats"}[floating-point
chapter].

Not every command is arithmetic-polymorphic. Command-specific validation rejects an unsupported
combination with an error, preserving the arithmetic requested by the caller.

Source:
{src "TorchLeanExamples/Quickstart/MLP.lean"}[`MLP.lean`].

We chose the synthetic target to fit the model family: it's a weighted combination of two ReLU
features, $`\operatorname{relu}(x_1+x_2)` and $`\operatorname{relu}(x_2-x_1)`, plus a constant. A
width-eight hidden layer can represent those features, so this exercise doesn't force the optimizer
to approximate a smooth curve with an undersized piecewise-linear model. The twenty-five training
points form a five-by-five grid. Since the held-out point lies between grid locations, its
prediction is a separate observation from the mean training loss.

# Convolutional Model Training

Prepare the small CIFAR-10 fixture and run one CPU optimizer step:

```terminal
# Prepare CIFAR arrays before running the cropped-image
# classifier.
python3 scripts/datasets/download_example_data.py --cifar10
scripts/lake.sh exe torchlean cnn --device cpu --n-total 1 --steps 1 --seed 2026
```

You can compare the starting loss with a reference value. A uniform prediction on ten classes has
cross-entropy $`\ln 10`:

```lean (name := exLn10)
-- A uniform prediction assigns probability one tenth to the
-- target class.
#eval Float.log 10.0
```
```leanOutput exLn10
2.302585
```

A loss close to $`\ln 10` on one sample doesn't establish that the full prediction vector is near
uniform. Very small or large losses are reasons to inspect the logits, labels, and objective, not
proof of a particular bug.

`dataset size = 1` counts the training dataset's minibatches. The CNN command fixes the batch size
at one, loads a shuffled epoch of full batches, and crops each image to $`8\times8`, so this
particular run has one image in one batch. Increasing `--n-total` supplies more rows and therefore
more one-image batches, while `--steps` independently sets how many optimizer updates run. Here the
input contract is `[1, 3, 8, 8]`, and the output `[1, 10]` contains one class-logit row.

Internally, the command defines one public `nn.CNN.Config`, derives the checked input and
output shapes from that value, and passes it to `nn.cnn`. It then uses the shared NPY data
boundary and `Trainer.create` with `trainer.train`. `--n-total 1` keeps this a check of that
training path, so the run doesn't establish useful CIFAR accuracy. Compare the reported loss before
and after the update to check its effect on this sample.

Source:
{src "TorchLeanExamples/Models/Vision/CNN.lean"}[`CNN.lean`].

Convolution changes the parameter-sharing pattern from the regression MLP. A small kernel is reused
at spatial positions within each image, so nearby patches are processed by the same learned
weights. We chose the cropped input to exercise this spatial path and its gradients with a small
workload. Image classification quality would need held-out images and an accuracy evaluation beyond
this one-step training check.

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
{src "TorchLeanExamples/Verification/IBPWorkflow.lean"}[`IBPWorkflow.lean`]:

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
weight pairs each endpoint with itself, and a negative
weight swaps them. -/
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
  let intervals := (List.finRange 3).map fun i =>
    exRow (fun j => Tensor.at exW1 (i, (j, ())))
      (Tensor.at exB1 (i, ())) xLo xHi
  for (box, i) in intervals.zipIdx do
    IO.println s!"h{i + 1} = ({box.fst}, {box.snd})"
  let y := exRow (fun i => Tensor.at exW2 (0, (i, ()))) exB2
    (fun i => (intervals[i]!).fst)
    (fun i => (intervals[i]!).snd)
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
also permits a stronger calculation here: all weights are nonnegative, so the lower and upper input
corners attain the extrema. With these decimal constants interpreted as exact reals,
$`[1.904, 2.256]` is the exact range. Stable ReLUs alone wouldn't make multilayer IBP exact: mixed
signs can still lose dependencies. The binary64 calculation above illustrates the formula. It
doesn't use outward rounding and isn't itself a certified enclosure. IBP is usually conservative.
Here the nonnegative weights let the same pair of input corners attain all the intermediate
extrema.

For the first hidden unit, the lower input corner is $`(0.4,0.7)`, giving
$`0.1\cdot0.4+0.2\cdot0.7+0.1=0.28`, and the upper corner gives $`0.34`. Repeating that calculation
for all three rows produces the displayed hidden intervals, whose midpoints are the hidden
activations at the center because this whole region stays within one linear piece of the ReLU
network. If a hidden interval crossed zero, the active linear expression could change within the
box and that midpoint argument would need to be reconsidered.

## IBP Workflow

```terminal
# Lower the fixed model and propagate its input box with
# native bound arithmetic.
scripts/lake.sh exe verify -- torchlean-ibp
```

Compare the reported endpoints with the hand calculation. Because the workflow uses outward
rounding, a wider result can reflect the numerical enclosure policy.

A lower endpoint rounded upward or an upper endpoint rounded downward can exclude the exact
result. Interval operations must account for that error. The native path uses
`HostFloat.nextUp` and `HostFloat.nextDown` from
{srcDir "TorchLean/Verify/CROWN"}[`TorchLean/Verify/CROWN`] to widen ordinary binary32 results by
one representable step. The following calculation measures the size of such a step near the final
upper endpoint, without counting the widenings or their accumulated effect:

```lean (name := exUlp)
-- Measure representable spacing near the upper endpoint, in
-- billionths of a unit.
open CROWN in
#eval show IO Unit from do
  let top : Float32 := 2.256
  let step := (HostFloat.nextUp top).toFloat - top.toFloat
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
scripts/lake.sh exe verify -- torchlean-ibp --arithmetic ieee
```

FloatLib exposes directed addition and multiplication through `ExecFloat.Binary.add` and
`ExecFloat.Binary.mul` with `.up`. On the finite real branch, these choose the smallest
representable upper result instead of always widening a nearest-rounded result. The interval rules
also account for exceptional endpoints. Of the two paths, the native one widens a nearest-rounded
result, while the reference path computes directed rounding in Lean. In the command output, the
arithmetic banner identifies which policy produced the box.

Lowering follows this path:

```
TorchLean model
  -> canonical IR
  -> graph of registered operators
  -> interval propagation
  -> output box
```

`lowered IR nodes: 18` counts the canonical IR after lowering, not the layers of the model. A
two-layer MLP expands into more nodes than one might guess because reshapes, broadcasts, and the
bias additions are all explicit at that level. Here the import keeps one node per IR node, and each
operator has its own interval rule: a node whose operator has none gets no box, and a rule gives no
box when a box or constant it needs is missing.

The result applies to this supported IR fragment and numerical policy, and a different model
command needs its own lowering and supported-operation checks. IBP propagates intervals without
auxiliary linear coefficients, losing dependencies between intermediate values. CROWN-style
relaxations {Informal.citep crown2018}[] can retain more of that information and produce tighter
bounds for suitable networks and relaxations, but there's no unconditional precision ordering for
arbitrary implementations. Related methods appear in {Informal.citet autolirpa2020}[] and
{ref "verification"}[the verification chapter].

You can list the other registered workflows, from `ibp-cert` through the two-stage Lyapunov
refinement, with:

```terminal
# Ask the verification dispatcher which workflows it
# registers.
scripts/lake.sh exe verify -- list
```

The dispatcher is reachable from {src "TorchLeanExamples/Verification/CLI.lean"}[
`TorchLeanExamples/Verification/CLI.lean`].

An output box becomes useful only after it is compared with the application's property. For this
scalar example, the lower endpoint can establish a threshold claim that lies below it, while a
threshold inside the interval remains undecided by this enclosure alone, so a wider box can be
numerically sound but insufficient for the requested property. This is the practical reason to
inspect both supported operations and bound precision before treating successful propagation as
successful verification.

# Numerical Certificate Replay

We can also record intermediate ranges in a numerical certificate and replay it with a checker
{Informal.citep necula1997}[].

First, evaluate the same small MLP at the center of its input box and compare its native and
reference gradients. We already have the weights and interval calculation needed to interpret
the result:

```terminal
# Compare the fixed MLP’s native and reference outputs and
# reverse-mode gradients.
scripts/lake.sh exe torchlean float32_semantics
```

Over the exact reals, $`y=2.08` is the midpoint of the interval calculated above, because the same
parameters are evaluated at the center of the same input region. You can also check the gradients
by hand. Since the scalar being differentiated is the output itself, the cotangent is one, and the
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
  let activations := (List.finRange 3).map fun i =>
    (List.finRange 2).foldl
      (fun acc j => acc + Tensor.at exW1 (i, (j, ())) * x j)
      (Tensor.at exB1 (i, ()))
  let dInput := (List.finRange 2).map fun j =>
    (List.finRange 3).foldl
      (fun acc i =>
        acc + w2 i * Tensor.at exW1 (i, (j, ())))
      0.0
  IO.println s!"h      = {activations}"
  IO.println s!"W2     = {(List.finRange 3).map w2}"
  IO.println s!"W2 W1  = {dInput}"
```
```leanOutput exFpIdent
h      = [0.310000, 0.670000, 1.030000]
W2     = [0.700000, 0.800000, 0.900000]
W2 W1  = [0.760000, 1.000000]
```

These three calculations correspond to `outputWeightGrad`, `hiddenBiasGrad`, and `inputGrad` in the
command's output. The `hiddenWeightGrad` row is the outer product $`W_2^{\top}x^{\top}`, whose
first entry is $`0.7\times0.5=0.35`. Comparing these factors with the computed gradients can catch
transposed matrices and dropped chain-rule factors. But the checked calculation here uses `Float`,
so it doesn't prescribe every bit of a binary32 result.

After converting both implementations' results to `Float`, the command reports the maximum absolute
difference across the output and five gradient tensors. Inspect that number: completion alone
doesn't assert that the difference is zero. For finite values, zero would establish numerical
agreement on these tensors, without distinguishing the signs of zero or inspecting intermediate
rounding decisions. The command also contrasts the proof-only rounded-real model (selected in
theorem statements and never executed) with those two executable paths, a distinction developed in
{ref "floats"}[the floating-point chapter] that has the same shape as the separation between the
real and the float models in {Informal.citet flocq2011}[].

Another connection to the interval calculation comes from the input-gradient row. Within this
stable ReLU region, changing the two inputs by $`(\delta_1,\delta_2)` changes the exact-real output
by $`0.76\delta_1+\delta_2`. So a radius of $`0.1` in each coordinate gives a maximum deviation of
$`0.176`, yielding $`2.08\pm0.176`. Here the derivative calculation and the interval calculation
meet because this example is affine throughout the region, but a local gradient wouldn't bound a
region crossing an activation change.

## Certificate Generation And Replay

The example certifies and replays a small scalar graph and a ten-node MLP, and alters an addition
interval of the scalar certificate so that validation rejects it. These definitions exercise
artifact generation and execution. They do not supply a proof of enclosure for every real input.

Here are the checker and its proof layer:

- {src "TorchLeanExamples/DeepDives/Floats/GraphNumericalCertificate.lean"}[
  `GraphNumericalCertificate.lean`].
- {srcDir "TorchLean/Verify/NumericalCertificate"}[
  `TorchLean/Verify/NumericalCertificate`].

Certificate ranges are outward-rounded FloatLib binary32, and concrete inputs are replayed in the
bit-level interpreter. A native runtime is a separate provider. Comparing its outputs and
gradients with FloatLib on one trajectory does not prove agreement on every execution.

# PyTorch Graph Import

Supported PyTorch graphs enter Lean as typed programs of registered operators
(`TorchLean.IR.Program`), which the Lean evaluators run. Import follows this path:

```
Python/PyTorch producer
  -> JSON capture document with a torchlean.ir artifact
  -> Lean loader (TorchLean.IR.load): references, ports and constants checked
  -> program of registered operators or explicit rejection
```

`TorchLean.IR.load_ok` establishes the structural property of a successful import: the program
passes the shape, type and interface check and its payload the payload check. Numerical agreement
needs a separate comparison or semantic proof: a graph with correct shapes can still compute the
wrong function. Structural validation doesn't prove the Python exporter correct. Unsupported
operations and malformed artifacts must be rejected at the capture or import boundary (see
{ref "pytorch-roundtrip"}[the round-trip chapter] for the supported schema and its limits).

You'll find the implementation in {src "TorchLean/NN/IR/Loaded.lean"}[`TorchLean.IR.load`]. An
affine scale and bias must retain their orientation, and epsilon belongs in a particular place in
the normalization denominator. Each supported operator needs an account of its attributes, state,
and numerical convention, not only its output shape.

# Building The Example Suite

Before changing maintained examples, build the curated umbrella:

```terminal
# Elaborate the curated example targets before running their
# entry points.
scripts/lake.sh build TorchLeanExamples
```

That checks the maintained Lean example targets. Elaboration is
not execution, so runtime behavior is checked separately:

```terminal
# Execute the retained suite; the final command runs the
# CUDA-enabled build.
scripts/lake.sh exe nn_tests_suite
scripts/lake.sh -R -K cuda=true build nn_tests_suite
scripts/lake.sh -K cuda=true env .lake/build/bin/nn_tests_suite
```

The maintained suite concentrates on numerical and native-boundary behavior: attention, convolution,
pooling, reductions, gather and scatter, FFT, matrix multiplication, and buffer lifetime. For an
example change, run that command with a small input and check its output. Routine command and API
refactors use temporary checks, not a permanent all-command suite. Documentation rendering requires
its own external environment.

Runnable examples use public executable APIs such as `Tensor.qr`, `Tensor.cholesky`, `nn.linear`,
and `Trainer.train`. Other examples deliberately evaluate mathematical specifications or
demonstrate theorems. Many specifications are executable reference functions, but running
them does not test the selected backend implementation.

Command implementations keep their local configuration vocabulary short because the namespace
already supplies the command name: `Options` for parsed flags and `Preset` for a named model
setup. Data prepared for one command likewise uses role names such as `Splits` or `Evaluation`,
rather than repeating the command name in every type.

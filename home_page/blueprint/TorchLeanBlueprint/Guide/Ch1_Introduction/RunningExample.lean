import VersoManual
import NN.API
import NN.API.Seeded
import NN.API.Verification.Lowering
import NN.Runtime.Autograd.Model.Layers.Seq
import NN.Spec.Models.Mlp
import NN.Verification.Builtin.Lowering.API
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open NN.Verification.Builtin (lInfBall)
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Running Example" =>
%%%
tag := "running-example"
file := "A-Running-Example"
%%%

The `quickstart_mlp` program fits a two-layer network to a known function. Knowing the target lets
us do more than watch the loss fall: we can write down weights that represent it exactly and compare
them with the weights Adam finds. We can also perturb individual weights to check the backward pass
and enlarge the input region to see how the output bounds change. The source
is {src "NN/Examples/Quickstart/SimpleMlpTrain.lean"}[`NN/Examples/Quickstart/SimpleMlpTrain.lean`],
and the chapter uses its definitions alongside commands for comparing training configurations.

The task is to learn the piecewise-linear function

$$`y(x_1,x_2)
  =0.8\,\operatorname{ReLU}(x_1+x_2)
   -0.4\,\operatorname{ReLU}(x_2-x_1)+0.2`

on a grid in $`[-1,1]^2`. Here $`x_1` and $`x_2` are the two input coordinates, and ReLU replaces
negative values with zero. Each term changes slope along a line through the input plane. A single
affine layer cannot represent those changes, but two ReLU units suffice. This gives us a way to
separate the network's ability to represent the target from the optimizer's success in finding it.

# The Source Program

The model is a two-layer MLP:

```lean (name := reSource)
-- Keep the architecture fixed while we inspect
-- initialization, gradients, and input bounds.
def reModel : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 8 1
  ]
```

Read the type before the body. `nn.Builder` says that model construction consumes a deterministic
seed stream. `nn.Sequential [2] [1]` says that the initialized model will map a
length-two tensor to a length-one tensor. The hidden width `8` is checked through composition: the
first linear layer produces length eight, ReLU preserves the shape, and the second linear layer
expects length eight.

Changing the second layer's input width to `7` breaks that composition:

```lean +error (name := reBad)
-- The final layer asks for seven inputs, but the preceding
-- layer produces eight.
def reBadModel : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 7 1
  ]
```

```leanOutput reBad (whitespace := lax)
nn.Sequential!: layer 2 expects input shape [7], but layer 1 outputs [8].
Change layer 2's input shape or insert a layer that converts [8] to [7].
```

The final layer constrains ReLU's shape to `[7]`, so the diagnostic points to layer 2: that ReLU
cannot consume the first layer's `[8]` output. Lean rejects the composition before a forward
pass runs.

The model returns a one-element tensor rather than a scalar because the shape-indexed layer API
treats the output feature dimension uniformly. A dataset target must therefore have shape `[1]`, not
shape `[]` and not shape `[batch]`.

# Seeded Initialization

Linear layers need initial weights and biases. Hiding randomness in a global generator would make a
model definition depend on ambient state. TorchLean instead represents initialization as a pure
state computation:

```lean (name := reInitDef)
-- Materialize the builder with seed 2026; later examples
-- refer to this initial payload.
def reInit : nn.Sequential [2] [1] := nn.build 2026 reModel
```

Running the same builder with the same seed produces the same initialization stream.
The resulting {lean}`reInit` records both the architecture and its initial parameter values.
Its parameter shapes are:

```lean (name := reShapes)
-- Inspect the four state tensors in the order consumed by
-- the forward program.
#eval nn.stateShapes reInit
```

```leanOutput reShapes
[[8, 2], [8], [1, 8], [1]]
```

Four tensors, in layer order: first-layer weights `[8, 2]`, first-layer bias `[8]`, second-layer
weights `[1, 8]`, second-layer bias `[1]`. The ReLU contributes nothing because it owns no
parameters. The initial values themselves live in a payload of exactly that shape list:

```lean (name := reState)
-- The state type depends on this model's complete internal
-- shape list.
#check nn.initialState reInit
```

```leanOutput reState (whitespace := lax)
nn.initialState reInit Float : nn.State Float
  (Runtime.Autograd.Model.Layers.Seq.stateShapes reInit)
```

The shape expression printed after `nn.State Float` depends on `reInit`, so it includes hidden
state layout as well as the public input and output shapes. The first weight tensor contributes
sixteen entries, its bias eight, the second weight eight, and its bias one. That is thirty-three
numbers grouped into four tensors. A function consuming this state can pattern-match those groups
without looking up string keys. It still needs the correct numerical payload: the type cannot
identify which training run produced two states of the same layout.

The order is part of the forward-program interface. It is not a PyTorch-style dictionary of names.
Checkpoint adapters may use names at an external boundary, but they must eventually construct this
typed ordered payload.

## PyTorch State Dictionaries

PyTorch's `state_dict` gives names to the corresponding tensors
{Informal.citep pytorch2019}[]:

```
>>> # Compare state names and shapes; the parameterless ReLU
>>> # contributes no dictionary entries.
>>> model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
>>> list(model.state_dict().keys())
['0.weight', '0.bias', '2.weight', '2.bias']
>>> [tuple(v.shape) for v in model.state_dict().values()]
[(8, 2), (8,), (1, 8), (1,)]
```

The four shapes are identical to the four Lean shapes above, in the same order. The difference is
what names them. PyTorch names them with strings that encode a position in the module list, so
index `1` is missing because the parameterless ReLU sits there. Lean names them by position in a
type-level list. Reordering entries with different shapes is a type error; swapping equally shaped
entries still requires a value-level check. The {ref "pytorch-roundtrip"}[round-trip chapter] is
about translating between the two conventions without losing that guarantee.

# The Dataset

The target in the source example is:

```lean (name := reTargetDef)
-- Two ReLU terms create sloped regions separated by a flat
-- region with value 0.2.
def reTarget (x : Tensor Float [2]) : Tensor Float [1] :=
  let relu (v : Float) := if v < 0.0 then 0.0 else v
  [0.8 * relu (x[0] + x[1])
    - 0.4 * relu (x[1] - x[0]) + 0.2]
```

Evaluating three corners and a held-out point shows both sloped and flat parts of the target:

```lean (name := reTargetEval)
-- Evaluate three grid corners and one point between grid
-- coordinates to locate those regions.
#eval reTarget [1.0, 1.0]
#eval reTarget [-1.0, 1.0]
#eval reTarget [-1.0, -1.0]
#eval reTarget [0.25, -0.75]
```

```leanOutput reTargetEval
[1.800000]
```

```leanOutput reTargetEval
[-0.600000]
```

```leanOutput reTargetEval
[0.200000]
```

```leanOutput reTargetEval
[0.200000]
```

At `[1, 1]`, the first ReLU receives two and the second receives zero, giving
`0.8 * 2 + 0.2 = 1.8`. At `[-1, 1]`, those roles switch, giving `-0.4 * 2 + 0.2 = -0.6`.
At the remaining two points both terms vanish. The target therefore contains more structure than
a single affine map: a learner must discover where each contribution turns on. The held-out input
is useful because a small training loss on the grid does not determine what the fitted function
does between its rows.

Both ReLUs are off when $`x_1+x_2\le0` and $`x_2\le x_1`, so that region is flat at `0.2`.
The held-out point `[0.25, -0.75]` lies inside this region and between the grid coordinates used
for training. Its prediction tests the learned function away from the sampled inputs.

The example samples five positions on each input axis, giving 25 examples:

```lean (name := reGrid)
-- Five positions on each axis produce 25 rows, each
-- containing two input coordinates.
#check Data.Synthetic.squareGrid (-1.0 : Float) 1.0 5
```

```leanOutput reGrid (whitespace := lax)
Data.Synthetic.squareGrid (-1.0) 1.0 5 : Tensor Float [5 * 5, 2]
```

The first dimension retains the expression `5 * 5` from the generator's type. Lean can reduce
that expression to `25`, so `[5 * 5, 2]` and `[25, 2]` are definitionally equal shapes.

`Tensor.mapLeading` applies the target to each row without fixing the operation to vectors or
regression:

```lean (name := reData)
-- Apply the same target to each input row before pairing
-- inputs and length-one labels.
def reData : Trainer.Dataset [2] [1] :=
  let inputs := Data.Synthetic.squareGrid (-1.0) 1.0 5
  Data.fromTensors inputs
    (Tensor.mapLeading [5 * 5] reTarget inputs)
```

`mapLeading [5 * 5]` treats the first axis as a collection of samples and passes each remaining
length-two tensor to `reTarget`. Since that function returns shape `[1]`, the resulting target
tensor has shape `[25, 1]`. `Data.fromTensors` then pairs rows with the same sample index.
No reshuffling or independent target generation occurs in this construction, so the pairing is
visible directly in the expression. This matters when debugging training: correct shapes alone
would still permit labels to be attached to the wrong input rows.

# Training

The trainer combines the model, objective, optimizer, seed, and runtime choices:

```
-- Preserve the command-line runtime choices while setting
-- the training objective and optimizer.
def trainer := Trainer.new model
  { flags.runtime with
      objective := .mse
      optimizer := optim.adam { learningRate := 0.03 }
      seed := flags.seed }
```

Run the complete checked-in command from the repository root:

```terminal
# Train the checked-in MLP example for 200 updates from
# initialization seed 2026.
lake exe torchlean quickstart_mlp \
  --device cpu \
  --steps 200 \
  --seed 2026
```

The command reports the dataset size, mean loss before and after training, periodic step losses,
and a prediction at the held-out input. Read the two loss reports separately: `mean_loss` evaluates
all 25 examples, while a step loss measures the single example used for that update. A low step
loss can therefore coexist with a much larger dataset mean.

Compare `trained(heldout)` with the target `0.2` as well. That prediction tests one point between
grid locations; it answers a different question from the average error on the training grid.

Whenever we return to `quickstart_mlp`, we keep this seed and configuration fixed; later chapters
also introduce smaller purpose-built variants.

## IEEE Binary32 Reference Execution

The same command accepts `--arithmetic ieee`, which selects FloatLib's executable binary32
arithmetic in place of the host's native operations:

```terminal
lake exe torchlean quickstart_mlp --device cpu --steps 200 \
  --seed 2026 --arithmetic ieee
```

The model, seed, samples, and Adam settings match the native command above. The selected scalar
changes to FloatLib's `ExecFloat.Binary 8 23`, so forward evaluation, the backward pass, and the
optimizer all use the reference arithmetic. Compare the starting loss, final loss, and held-out
prediction. Matching printed decimals would establish agreement only at that display precision,
for these runs. The
{ref "floats"}[floating-point chapter] explains how rounding can separate the results, and
{ref "motivation"}[the motivation chapter] shows a bound-propagation example where the two do
disagree.

## PyTorch Training Comparison

The following PyTorch program uses the same architecture, 25-point grid, target, mean squared error,
and Adam learning rate. Its loop uses a full batch at each step; that difference matters when
comparing the results.

```terminal
python3 - <<'PY'
# Compare a full-batch PyTorch run; its initial weights and
# update schedule differ.
import torch
from torch import nn

torch.manual_seed(2026)
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
axis = torch.linspace(-1, 1, 5)
grid = torch.cartesian_prod(axis, axis)
ys = (0.8 * torch.relu(grid[:, 0] + grid[:, 1])
      - 0.4 * torch.relu(grid[:, 1] - grid[:, 0]) + 0.2).unsqueeze(1)
heldout = torch.tensor([0.25, -0.75])
opt = torch.optim.Adam(model.parameters(), lr=0.03)
with torch.no_grad():
    print("mean_loss(before) =", nn.MSELoss()(model(grid), ys).item())
for step in range(200):
    opt.zero_grad()
    loss = nn.MSELoss()(model(grid), ys)
    loss.backward()
    opt.step()
with torch.no_grad():
    print("mean_loss(after) =", nn.MSELoss()(model(grid), ys).item())
    print("trained(heldout) =", model(heldout).item())
PY
```

This loop evaluates 25 examples per update: 5000 example gradients across 200 steps, compared with
200 in the Lean command. The libraries also use different initialization generators. Comparing
their final losses therefore compares two training procedures; it does not isolate the arithmetic
implementation. To make that comparison, supply the same parameter tensors and use the same samples
in the same update order.

# Training Step

For one sampled pair $`(x,y)`, the runtime performs:

1. read the current four parameter tensors;
2. compute `Linear -> ReLU -> Linear`;
3. compute the regression loss;
4. seed the loss cotangent with one;
5. traverse the autograd tape backward to obtain parameter gradients;
6. update Adam's first and second moments;
7. replace each parameter with its updated value;
8. release or retain runtime buffers according to ownership.

Write the four parameter tensors as $`\theta=(W_1,b_1,W_2,b_2)`. The weight matrices map two
input coordinates to eight hidden coordinates, then eight hidden coordinates to one output;
the biases have lengths eight and one. The forward map is

$$`f_\theta(x)
  =W_2\,\operatorname{ReLU}(W_1x+b_1)+b_2`.

Step two has a specification-level twin we can run directly. `Examples.mlpForward` is that equation
as a pure function of two weight-and-bias records, without an autograd tape. Its tensor operations
can still allocate intermediate buffers:

```lean (name := reForward)
-- Unpack the initial state in layer order and evaluate it
-- at the held-out input.
def reHeldout : Tensor Float [2] := [0.25, -0.75]

def rePred : Tensor Float [1] :=
  match Runtime.Autograd.Model.Layers.Seq.initState
      (m := reInit) with
  | .cons w1 (.cons b1 (.cons w2 (.cons b2 .nil))) =>
      Examples.mlpForward (α := Float)
        { weights := w1, bias := b1 }
        { weights := w2, bias := b2 } reHeldout

#eval rePred
```

```leanOutput reForward
[-0.088261]
```

The pattern match reads the four tensors from the initial state. Compare this result with
`untrained(heldout)` from the command above: both start from the seed-2026 payload, but this
calculation uses host `Float` while the trainer selects its configured arithmetic. If they disagree,
this small forward calculation helps separate an error in the MLP equations from an error in
loading or executing
the state, or a difference in rounding. It also gives proofs a computation to refer to without
including tape management.
The comparison checks one input; equality for all inputs requires a theorem. The derivative and
interval examples below continue to use this initial payload.

The loss for a single example is a scalar function of $`\theta`, even though the prediction has
shape `[1]`. Reverse mode computes vector-Jacobian products from that scalar back to all four
tensors {Informal.citep baydin2018}[]. Adam then uses those gradients and its optimizer state to
construct the next payload.

To reason about that update, we need to connect the eager runtime's saved tensors and tape nodes
to an ideal VJP. The autograd proofs state the derivative rule; rounded VJP error transformers
bound the numerical discrepancy. Optimizer contracts then describe how those gradients enter SGD,
momentum, or AdamW updates {Informal.citep adamw2019}[].

# Exact Representation Of The Target

The target does not require all eight hidden units. Its
definition supplies parameters for a two-unit network directly.
The term $`0.8\operatorname{ReLU}(x_1+x_2)` is one hidden unit with
incoming row $`(1,1)` and outgoing weight $`0.8`; $`-0.4\operatorname{ReLU}(x_2-x_1)` is a second
with row $`(-1,1)` and outgoing weight $`-0.4`; the $`+0.2` is the output bias.
Setting those values explicitly gives:

```lean (name := reExact)
-- Encode the target directly with two hidden units, then
-- check its finite-grid loss.
def reExactHidden : Spec.LinearSpec Float 2 2 :=
  { weights := [[1.0, 1.0], [-1.0, 1.0]]
    bias := [0.0, 0.0] }

def reExactOut : Spec.LinearSpec Float 2 1 :=
  { weights := [[0.8, -0.4]], bias := [0.2] }

def reExact (x : Tensor Float [2]) : Tensor Float [1] :=
  Examples.mlpForward (α := Float)
    reExactHidden reExactOut x

def reGridInputs : Tensor Float [5 * 5, 2] :=
  Data.Synthetic.squareGrid (-1.0) 1.0 5

/-- Mean squared error of any candidate forward map over
the 25 training points, so competing parameter settings can
be compared on the same footing. -/
def reGridLoss
    (f : Tensor Float [2] → Tensor Float [1]) : Float :=
  Tensor.meanSquaredError
    (Tensor.mapLeading [5 * 5] f reGridInputs)
    (Tensor.mapLeading [5 * 5] reTarget reGridInputs)

#eval (reExact [0.25, -0.75], reExact [1.0, 1.0])
#eval reGridLoss reExact
-- Printing six decimals would hide a small nonzero, so
-- ask the question directly.
#eval reGridLoss reExact == 0.0
```

```leanOutput reExact (whitespace := lax)
([0.200000], [1.800000])
```

```leanOutput reExact (whitespace := lax)
0.000000
```

```leanOutput reExact (whitespace := lax)
true
```

The hidden rows compute the two linear forms in `reTarget`, and the readout applies their
coefficients. We obtained these weights directly from the target equation. The final Boolean
confirms that their computed grid loss is zero, even beyond the six printed decimals, and their
held-out prediction is `0.2`.

Over the reals, the construction represents the target by definition. The floating-point test
covers this finite grid; it does not establish equality for every input, including overflow and
non-finite cases. If a training run leaves a nonzero error, insufficient width alone cannot explain
it: the architecture can contain this two-unit construction.
That distinction separates {ref "approximation-theory"}[approximation theory] from
{ref "optimization-theory"}[optimization theory]. Even a zero-loss run need not recover these
particular weights, as the following symmetries show.

## Parameter Symmetries

The exact parameters above are not *the* exact parameters. ReLU is positively homogeneous, so
scaling a hidden row by $`c>0` and its outgoing weight by $`1/c` leaves the function alone, and
relabelling the hidden units does too:

```lean (name := reSymmetry)
-- Row scaled up by four, outgoing weight down by four.
def reScaledHidden : Spec.LinearSpec Float 2 2 :=
  { weights := [[4.0, 4.0], [-1.0, 1.0]]
    bias := [0.0, 0.0] }

def reScaledOut : Spec.LinearSpec Float 2 1 :=
  { weights := [[0.2, -0.4]], bias := [0.2] }

-- The same two units, listed in the other order.
def rePermHidden : Spec.LinearSpec Float 2 2 :=
  { weights := [[-1.0, 1.0], [1.0, 1.0]]
    bias := [0.0, 0.0] }

def rePermOut : Spec.LinearSpec Float 2 1 :=
  { weights := [[-0.4, 0.8]], bias := [0.2] }

def reScaled (x : Tensor Float [2]) : Tensor Float [1] :=
  Examples.mlpForward (α := Float)
    reScaledHidden reScaledOut x

def rePerm (x : Tensor Float [2]) : Tensor Float [1] :=
  Examples.mlpForward (α := Float)
    rePermHidden rePermOut x

#eval (reGridLoss reScaled, reGridLoss rePerm)
#eval (reGridLoss reScaled == 0.0,
  reGridLoss rePerm == 0.0)
```

```leanOutput reSymmetry (whitespace := lax)
(0.000000, 0.000000)
```

```leanOutput reSymmetry (whitespace := lax)
(true, true)
```

Both alternatives have zero computed loss on this finite grid. The floating-point agreement for
this scale also depends on its representation: multiplying by four only shifts an exponent, and
$`4\cdot\mathtt{fl}(0.2)` happens to be $`\mathtt{fl}(0.8)`. Pick $`c=3` and the two programs would
agree over $`ℝ` and could differ in the last bit.

Over the reals, positive rescaling and hidden-unit permutations can give distinct parameters for
the same function; repeated or zero units can make some such transformations coincide. Weight
recovery therefore needs an identifiability convention. The eight-unit model can embed the two-unit
construction with six inactive units. What can be stated, and what {ref "verification"}[the
verification chapter] states, are
properties of the *function* a parameter vector denotes: its value at a point, its bound over a
region, its distance from another function. These are unchanged by the real-valued symmetries above,
even when the individual weights change.

# Checking The Gradient Against A Difference Quotient

Decreasing loss is not enough to validate a backward pass: some incorrect updates can still
reduce the objective. A separate derivative check compares the backward implementation with
changes in the forward computation
{Informal.citep griewank2000}[].

We can estimate a derivative by perturbing a parameter and taking a difference quotient, as
`torch.autograd.gradcheck` does. Here `Examples.mlpBackward` returns five tensors in order:
gradients for the first weights, first bias, second weights, second bias, and finally the input.
To compare them with the forward computation, return to
the seeded eight-unit model and repackage its payload as two weight-and-bias records. Its
first-layer
pre-activations at the held-out point are:

```lean (name := reGradPre)
-- Find which initial hidden units can transmit a derivative
-- at the held-out point.
def reL1 : Spec.LinearSpec Float 2 8 :=
  match Runtime.Autograd.Model.Layers.Seq.initState
      (m := reInit) with
  | .cons w1 (.cons b1 (.cons _ (.cons _ .nil))) =>
      { weights := w1, bias := b1 }

def reL2 : Spec.LinearSpec Float 8 1 :=
  match Runtime.Autograd.Model.Layers.Seq.initState
      (m := reInit) with
  | .cons _ (.cons _ (.cons w2 (.cons b2 .nil))) =>
      { weights := w2, bias := b2 }

def rePre : Tensor Float [8] :=
  Spec.linearSpec (α := Float) reL1 reHeldout

#eval (List.finRange 8).map
  fun i => decide (0.0 < rePre[i])
#eval (rePre[0], rePre[5])
```

```leanOutput reGradPre
[false, false, false, false, false, true, false, false]
```

```leanOutput reGradPre
(-0.479660, 0.154824)
```

Only hidden unit five has a positive pre-activation at `[0.25, -0.75]`; the other seven output
zero after ReLU. This is the activation pattern of the randomly initialized model, separate from
the two-unit construction of the target. It lets us inspect the gradient through one active path.

Hold the other parameters fixed and view the loss as a function of the first-layer weights.
At the held-out point, let $`y` be the prediction and $`t` the target. The loss is their squared
difference, so the cotangent supplied to the backward pass is $`\partial L/\partial y = 2(y-t)`:

```lean (name := reGradAn)
-- Hold three parameter tensors fixed and differentiate the
-- loss with respect to W1.
def reLossAt (w1 : Tensor Float [8, 2]) : Float :=
  let y := Examples.mlpForward (α := Float)
    { weights := w1, bias := reL1.bias } reL2 reHeldout
  let d := y[0] - (reTarget reHeldout)[0]
  d * d

def reAnalyticW1 : Tensor Float [8, 2] :=
  let y := Examples.mlpForward (α := Float)
    reL1 reL2 reHeldout
  let dLdy : Tensor Float [1] :=
    [2.0 * (y[0] - (reTarget reHeldout)[0])]
  let (dW1, _, _, _, _) :=
    Examples.mlpBackward (α := Float)
      reL1 reL2 reHeldout dLdy
  dW1

#eval reLossAt reL1.weights
#eval reAnalyticW1
```

```leanOutput reGradAn
0.083094
```

```leanOutput reGradAn
[[0.000000, -0.000000],
 [-0.000000, 0.000000],
 [-0.000000, 0.000000],
 [0.000000, -0.000000],
 [-0.000000, 0.000000],
 [0.082165, -0.246494],
 [0.000000, -0.000000],
 [0.000000, -0.000000]]
```

The first printed value is the squared error of the initial network at one input; it is not the
mean training loss reported by the command. The matrix that follows has exactly the shape of `W1`,
so each entry answers how changing that weight locally changes this scalar loss. The upstream
cotangent `2 * (y - t)` already includes differentiation of the loss. Supplying one instead would
ask for the derivative of the prediction. This distinction is easy to miss when calling a backward
routine directly, because both cotangents have the same tensor shape.

Seven rows are zero because their ReLU units are inactive. Signed zeros reflect the arithmetic
path; their signs alone would not distinguish exact zeros from underflowed values. Only row five
survives, and its two entries stand in
ratio $`-3`, which is $`x_2/x_1=-0.75/0.25`. If $`\delta_i` denotes the loss derivative with
respect to unit $`i`'s pre-activation, then
$`\partial L/\partial (W_1)_{ij}=\delta_i x_j`: each row is a scalar multiple of the input vector.

The claim to test is that `0.082165` is really a derivative. Perturb that one entry up and down and
take a central difference of the loss:

```lean (name := reGradFd)
-- Change only one weight in each direction; invalid
-- coordinates propagate as none.
def reCentral (row col : Nat) (delta : Float) :
    Option Float := do
  let w1 := reL1.weights
  let up   ← w1.modify? #[row, col] (· + delta)
  let down ← w1.modify? #[row, col] (· - delta)
  pure ((reLossAt up - reLossAt down) / (2.0 * delta))

#eval reCentral 5 0 1.0e-5
#eval reCentral 5 1 1.0e-5
```

```leanOutput reGradFd
some 0.082165
```

```leanOutput reGradFd
some (-0.246494)
```

Both entries agree with the backward pass to every printed digit. Each comparison changes one
weight while holding the other weights, biases, input, and target fixed. Dividing by `2 * delta`
measures the change across the symmetric interval around that weight.

`Tensor.modify?` returns an `Option` because the coordinates are runtime naturals. The `do` block
propagates a failed lookup, and `some` confirms that these coordinates existed. Step size needs a
separate check: all calls here use nonzero `delta`, but the helper does not enforce that condition
or decide whether the perturbation is small enough.

## Finite-Difference Step Size

A central difference often has an error curve shaped like a shallow U. Truncation error
typically scales as $`\delta^2` for a smooth function, while rounding in the subtracted values is
amplified by $`1/\delta`. The best scale depends on the function and the evaluation error. Here
is the relative
error against the backward pass, in units of $`10^{-12}`:

```lean (name := reGradSweep)
-- Report relative disagreement in units of 1e-12 while
-- reducing the perturbation size.
def reRelErr (delta : Float) : Float :=
  match reCentral 5 0 delta, reAnalyticW1.at? #[5, 0] with
  | some fd, some ad =>
      Float.abs (fd - ad) / Float.abs ad * 1.0e12
  | _, _ => 1.0e12

#eval [1.0e-2, 1.0e-3, 1.0e-4].map reRelErr
#eval [1.0e-5, 1.0e-6, 1.0e-8].map reRelErr
```

```leanOutput reGradSweep
[0.012668, 0.004223, 0.080228]
```

```leanOutput reGradSweep
[0.080228, 177.427282, 2035.348794]
```

The scale on this output is easy to misread. A displayed error of about 177 means approximately
`177 * 1e-12` relative disagreement, not a factor of 177 between the derivatives. The denominator
is the absolute analytic entry, which is nonzero for this chosen active weight. A zero entry would
need a different comparison, such as an absolute error criterion. The fallback value in `reRelErr`
marks a failed coordinate lookup; it is not a measured relative error. For the fixed valid indices
used here, the displayed values come from the successful branch.

Here the error is already at rounding level for $`\delta=10^{-2}` and grows as the step shrinks.
As long as the ReLU pattern does not change, the real-valued loss is
*exactly* quadratic in this weight. Unit five is affine in $`(W_1)_{50}`, the second layer is affine
in that unit, and the squared error turns the composition into a parabola. A central difference is
exact on a parabola, so there is no truncation error left to trade against cancellation, and
shrinking $`\delta` only destroys significant digits.

This finite-difference test checks local agreement in a fixed activation region. It does not
certify every input or settle the derivative convention at a kink.

## Crossing An Activation Boundary

Nudge a dead row instead of the live one:

```lean (name := reGradDead)
-- A large perturbation can activate a unit whose derivative
-- at the original point is zero.
#eval reCentral 0 0 1.0e-5
#eval reCentral 0 0 1.0
#eval reCentral 0 0 3.0
```

```leanOutput reGradDead
some 0.000000
```

```leanOutput reGradDead
some 0.000000
```

```leanOutput reGradDead
some 0.002521
```

A nudge of $`10^{-5}` leaves the loss bit-identical, and so does a nudge of $`1`. Unit zero has
pre-activation $`-0.4797` and the entry we are moving enters it multiplied by $`x_1=0.25`, so it
takes $`\delta>1.92` to wake the unit up. At $`\delta=3` the upward perturbation activates it while
the downward one does not, the two loss values stop being equal, and the quotient reports `0.002521`
where the derivative is zero.

The quotient with this larger step spans two activation regions, so it need not approximate the
local derivative at the original weights. At an activation boundary, the ordinary derivative may
not exist; generalized derivatives provide a set-valued description
{Informal.citep bolte2020}[]. A finite-difference check alone does not establish that both
perturbations remain in one region. {ref "autograd-proofs"}[The autograd proofs chapter] makes
the activation hypotheses explicit in derivative theorems, while
{ref "verification"}[the verification chapters] use relaxations valid over an entire input region.

## PyTorch Gradient Checks

PyTorch's `gradcheck` packages the perturbations and comparisons into one call:

```
# Copy the Lean payload into double tensors before checking
# derivatives by finite differences.
# Python / PyTorch
import torch
torch.manual_seed(2026)

net = torch.nn.Sequential(
    torch.nn.Linear(2, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 1),
).double()

x = torch.tensor([0.25, -0.75], dtype=torch.float64)
t = torch.tensor([0.2], dtype=torch.float64)

def loss(w1):
    hidden = torch.relu(torch.nn.functional.linear(x, w1, net[0].bias))
    prediction = torch.nn.functional.linear(hidden, net[2].weight, net[2].bias)
    return ((prediction - t) ** 2).sum()

w1 = net[0].weight.detach().clone()
w1.requires_grad_(True)
print(torch.autograd.gradcheck(loss, (w1,), eps=1e-6))
```

This reports `True`. Double precision suits `gradcheck`'s default tolerances; the Lean version
above already uses binary64 `Float`. The Python loss uses its `w1` argument directly, preserving the
autograd connection that wrapping it in a new `Parameter` would break. `gradcheck` compares all
coordinates and returns a Boolean on success, which is convenient for a test suite but hides the
structure we spent this section looking at: which rows are dead, which entries are proportional to
the input, and how the error behaves as the step shrinks.

# Device Selection

The source model does not mention CPU or CUDA. Device selection belongs to the runtime
configuration:

```terminal
# Run the training example with a CPU execution request.
lake exe torchlean quickstart_mlp --device cpu --steps 200
```

or, in a CUDA-enabled build:

```terminal
# Build with CUDA support and request CUDA execution for the
# same example.
lake -R -K cuda=true exe torchlean \
  quickstart_mlp --device cuda --steps 200
```

The model type and parameter layout stay put. The selected backend profile plans the operations
using the capsules available in that build, and `--show-backend` prints the resulting plan. Each
line names the provider, the trust level, the VJP source, and how the operation is justified:

```
  matmul: reference.matmul provider=reference trust=checked vjp=torchlean-tape
    shape: shape safety for matmul; guarded at runtime by portable runtime shape checks
    layout: canonical-tensor layout compatibility for matmul
```

The complete report includes every capsule used by the run. These entries connect the device
choice to the implementations of individual operations; {ref "backend-selection"}[the backend
chapter] explains how the planner selects them and what each trust level means.

# Lowering The Forward Map

Verification starts from an initialized model and a concrete parameter payload:

```lean (name := reLower)
-- Ask for the lowering type without claiming that a
-- particular call has succeeded.
#check Verification.lowerForwardToIR (α := Float) reInit
  (nn.initialState reInit)
```

```leanOutput reLower (whitespace := lax)
Verification.lowerForwardToIR reInit
  (nn.initialState reInit Float) : Except String (NN.Verification.Builtin.LoweredIR Float)
```

On success, `LoweredIR Float` pairs an `NN.IR.Graph` with its parameter payload store and the
distinguished input and output nodes. The graph records shapes and operation tags; the payload's
scalar type fixes how its entries are represented. The `Except` also admits a lowering error,
which the examples below propagate through a `do` block.

The lowered graph expands the two linear layers and ReLU into primitive operations, parameter
constants, and shape adjustments. Its node count is:

```lean (name := reNodes)
-- Evaluate the lowering and extract its node count only
-- from a successful result.
#eval (Verification.lowerForwardToIR (α := Float) reInit
    (nn.initialState reInit)).map
  fun l => l.graph.nodes.size
```

```leanOutput reNodes
Except.ok 18
```

Eighteen nodes for a three-layer network, because lowering is explicit: parameters become constant
nodes, a linear layer becomes a reshape, a matmul, and an add, and each node records its parents and
output shape. {ref "graphs-and-ir"}[The IR chapter] walks the node list.

Using `nn.initialState reInit` analyzes the initial model. To analyze the trained model, the
lowering must receive the trained runtime parameters through the lower-level manual interface or a
saved exact-bits payload. Reusing the initial payload after training would verify another function.
Everything below therefore describes the network at seed `2026`, before any Adam step.

# Input Regions

A forward prediction answers "what did the model return at $`x`?" Verification usually asks a
quantified question. Around the held-out point

$$`c=(0.25,-0.75)`,

an $`\ell_\infty` ball of radius $`\varepsilon` is the box

$$`B_\varepsilon(c)
 =\{x\mid |x_i-c_i|\leq\varepsilon\text{ for }i=1,2\}`.

`FlatBox.lInfBall` builds that box, `LoweredIR.seedInputBox` places it at the graph's input node,
and `runIBP` propagates one interval per coordinate through the graph
{Informal.citep gowal2018}[]. The output shape is checked before returning the bounds:

```lean (name := reBoxDef)
-- Place the input region at the lowered input node and
-- check the output dimension.
def reBox (eps : Float) :
    Except String
      (Tensor Float [1] × Tensor Float [1]) := do
  let lowered ← Verification.lowerForwardToIR (α := Float)
    reInit (nn.initialState reInit)
  let ps := lowered.seedInputBox (lInfBall reHeldout eps)
  let out ← lowered.outputBox? (lowered.runIBP ps)
  if h : out.dim = 1 then
    return (h ▸ out.lo, h ▸ out.hi)
  else
    throw "expected one output coordinate"
```

The `if h : out.dim = 1` branch supplies evidence that the flat output box has one coordinate.
The expressions `h ▸ out.lo` and `h ▸ out.hi` use it to return tensors of shape `[1]`, preserving
their entries without rounding them again. If the dimension differs, the function returns an error.

At radius zero, the input box contains only the held-out point, so we can compare its output
bounds with the earlier prediction:

```lean (name := reBoxZero)
-- Collapse the region to the held-out input to compare with
-- the earlier forward value.
#eval reBox 0.0
```

```leanOutput reBoxZero
Except.ok ([-0.088261], [-0.088261])
```

Both endpoints print as `-0.088261`, matching {lean}`rePred` at this point and precision. This is
a useful regression check of lowering and payload handling, not a proof that the functions agree
on all inputs. Outward rounding can also give a nonzero-width enclosure at a single point.

Increasing the radius to `0.05` includes a neighborhood of that point:

```lean (name := reBoxEps)
-- Allow each input coordinate to move by 0.05 in either
-- direction.
#eval reBox 0.05
```

```leanOutput reBoxEps
Except.ok ([-0.111691], [-0.064831])
```

These are the computed candidate endpoints, rounded for display. Interpreting them as an enclosure
for every point in the input box requires a soundness theorem for the chosen graph and arithmetic.
The printed decimals themselves must not be substituted for outward-rounded exact endpoint bits.

## Output Width And Input Radius

To see how the output interval changes with the input region, divide its width by the input
radius at several scales:

```lean (name := reWidths)
-- Compare width per unit input radius; these calls use
-- positive radii and the initial state.
def reWidth (eps : Float) : Float :=
  match reBox eps with
  | .ok (lo, hi) => hi[0]! - lo[0]!
  | .error _ => 0.0

#eval [0.01, 0.05, 0.1].map fun e => reWidth e / e
#eval [0.5, 1.0].map fun e => reWidth e / e
```

```leanOutput reWidths
[0.937198, 0.937198, 0.937198]
```

```leanOutput reWidths
[0.730910, 1.276392]
```

The helper returns zero on an error to keep this display short. A verification tool must instead
preserve the `Except` result: failure is not an interval of zero width. Here we are inspecting the
successful propagation path for the fixed graph.

The first three ratios agree at the displayed precision. In the ideal real computation on these
small boxes, only one hidden unit is active. Its affine output has width
$`2\varepsilon\lVert a\rVert_1` for effective row $`a`, and this single active path avoids the
dependency loss that can remain even in a multi-path affine network.

The last two ratios show that output width no longer scales linearly with radius. ReLU crossings
change the piecewise affine behavior, and interval dependency loss may also contribute. Width
divided by radius is not a direct measure of looseness: measuring looseness would require an
independent bound on the true range. This is exactly the gap that affine relaxations attack by
keeping a linear function of the input instead of a pair of numbers
{Informal.citep crown2018}[]{Informal.citep autolirpa2020}[].

## Output Properties

For this one-output regression model, choose an allowed error $`\delta` from the target value
`0.2`. The desired property is

$$`\forall x\in B_\varepsilon(c),\qquad
  |f_\theta(x)-0.2|\leq\delta`.

A computed output interval $`[\ell,u]` supports this claim when

$$`0.2-\delta\leq \ell
  \quad\text{and}\quad
  u\leq 0.2+\delta`,

provided a soundness theorem covers the graph operations, payload, input bounds, and scalar
semantics used by the pass.

For $`\varepsilon=0.05`, the computed outer interval would require $`\delta` of about `0.312`
to pass this sufficient interval test. A loose outer interval cannot establish the smallest
possible tolerance or refute a property by itself. The prediction at the center already supplies
a numerical counterexample to tolerances much smaller than `0.288`. Conditional on a sound
enclosure bridge, the upper endpoint would also establish that every output in this region is
below `-0.06`.

# Verification Parameters

The interval calculation encloses negative outputs because `reBox` lowers `nn.initialState reInit`.
Training updates a separate parameter payload. A changed held-out prediction from the trainer
therefore cannot be compared with this box as though the two described the same fitted function.
The architecture and input point agree; the weights need not.

To bound the trained network, we need its current parameter values, then we must repeat lowering
and bound propagation with that payload. A saved verification result must identify those values
along with its input region and arithmetic. The tensor shapes let the forward program accept the
new payload; they cannot tell us whether it is the payload a particular claim was proved about.

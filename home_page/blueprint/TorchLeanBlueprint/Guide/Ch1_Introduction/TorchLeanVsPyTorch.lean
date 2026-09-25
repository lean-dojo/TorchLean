import VersoManual
import NN.API
import NN.API.Runtime
import NN.API.Verification.Lowering
import NN.IR
import NN.Spec.Models.Mlp
import NN.Examples.BugZoo.BatchInvariance
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "TorchLean and PyTorch" =>
%%%
tag := "torchlean_vs_pytorch"
%%%

TorchLean and PyTorch both organize training around tensors, modules, parameters, autograd, and
optimizers. The comparison becomes more precise when we follow the same model through those
interfaces. Where is its shape recorded? How do we obtain its current weights? Which function
does a backward rule differentiate, and what arithmetic evaluates that rule?

Named Lean blocks are elaborated when this page is built, and their recorded outputs are checked.
The Python examples record runs against PyTorch 2.13.0; the guide build does not execute them.

# MLP Definitions

A PyTorch model commonly owns its parameters through `nn.Module`:

```
# Build a four-feature, two-output MLP and call it on one
# input vector.
# Python / PyTorch
import torch
net = torch.nn.Sequential(
    torch.nn.Linear(4, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 2),
)

x = torch.zeros(4)
logits = net(x)
```

The corresponding TorchLean builder is:

```lean (name := tpModelDef)
-- Match the Python architecture while retaining its input
-- and output shapes in the type.
def tpModel :
    nn.Builder (nn.Sequential [4] [2]) :=
  nn.Sequential![
    nn.linear 4 8,
    nn.relu,
    nn.linear 8 2
  ]
```

In PyTorch, an eager tensor carries its shape as runtime metadata. Calling a layer inspects those
dimensions while the program executes. PyTorch's export and compilation systems can add symbolic
shape constraints later, but an ordinary annotation such as `torch.Tensor` does not distinguish a
vector of length four from a matrix with four columns.

In TorchLean, the input and output shapes index the `nn.Sequential` type. Layer composition is
checked while Lean elaborates the definition. `nn.Builder` also records that `tpModel` is a
deterministic seed-state computation waiting to initialize its parameters.

## Layer And Input Shape Errors

Suppose the second linear layer is typed with the wrong input width. In TorchLean the definition
does not survive elaboration:

```lean +error (name := tpWidth)
-- A seven-input readout cannot follow an eight-output
-- hidden layer.
def tpBroken :
    nn.Builder (nn.Sequential [4] [2]) :=
  nn.Sequential![
    nn.linear 4 8,
    nn.relu,
    nn.linear 7 2
  ]
```
```leanOutput tpWidth (whitespace := lax)
nn.Sequential!: layer 2 expects input shape [7], but layer 1 outputs [8].
Change layer 2's input shape or insert a layer that converts [8] to [7].
```

The message identifies a layer expecting width `7` where composition requires width `8`.
Lean rejects this model before execution.

A related PyTorch mismatch occurs when the valid four-input model receives a width-three input:

```
# Python / PyTorch
>>> net(torch.zeros(3))
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x3 and 4x8)
```

Here PyTorch checks the dimensions during the call. An invalid operation in a conditional
branch is therefore detected when that branch executes. Lean checks the shape contracts of both
branches while elaborating a shape-indexed program.

Dynamic shapes are convenient during exploration and for data-dependent programs. Shape-indexed
types require more information up front, but they make layer composition and later theorem
statements much cleaner. TorchLean still accepts runtime-loaded data; it checks the dimensions once
at the boundary and then works with the resulting typed tensor.

These contracts also describe higher-rank inputs. `matmul` handles matrices, batches of
matrices, and higher-rank collections, broadcasting compatible leading dimensions.

There is also no special batch tensor type. Operations that preserve outer axes accept a leading
shape explicitly. Thus `nn.flattenAfter [batch, time]` keeps the batch and time axes and
flattens everything after them. Classification and regression heads use the same argument, so one
definition works for single examples, batches, sequences of examples, and higher-rank collections.
The resulting input and output shapes are still checked while the model is built.

# Parameters: Object Fields Versus Explicit Payloads

A PyTorch `nn.Module` registers parameter objects. Calling `model(x)` reads the module's current
fields. An optimizer mutates those parameters, usually through gradient fields populated by
autograd.

TorchLean's model description contains parameter shapes, initialization tensors, gradient
flags, and a forward program. The forward program receives the *live parameter payload* explicitly.
Initialization and execution are therefore related but distinguishable:

```lean (name := tpInitDef)
-- Initialize once, then inspect the type of the state
-- associated with this model.
def tpInit : nn.Sequential [4] [2] :=
  nn.build 2026 tpModel

#check nn.initialState tpInit
```
```leanOutput tpInitDef (whitespace := lax)
nn.initialState tpInit Float : nn.State Float
  (Runtime.Autograd.Model.Layers.Seq.stateShapes tpInit)
```

The initialized description exposes its state shape list:

```lean (name := tpShapes)
-- List weights and biases in the order required by the
-- forward program.
#eval nn.stateShapes tpInit
```
```leanOutput tpShapes
[[8, 4], [8], [2, 8], [2]]
```

The four groups contain 32 weights, eight hidden biases, sixteen readout weights, and two
readout biases. The input and output interface `[4] -> [2]` alone does not reveal those groups:
another model could use the same interface with a different hidden width. The shape list exposes
that internal layout. The PyTorch names additionally identify where each tensor came from in the
module tree. An adapter must connect names to positions and validate their shapes before it can
compare or execute the resulting payload.

PyTorch answers the same question through the module's `state_dict`:

```
# Inspect the named state tensors that must be matched by
# any checkpoint adapter.
# Python / PyTorch
for name, t in net.state_dict().items():
    print(f"{name:10s} {tuple(t.shape)}")
```

```
0.weight   (8, 4)
0.bias     (8,)
2.weight   (2, 8)
2.bias     (2,)
```

Both lists contain the same four tensors in the same order. PyTorch attaches a string key to
each tensor; loading must account for missing, unexpected, or
renamed keys. TorchLean attaches a position and shape computed from the model. This rejects
incompatible state shape lists, but another architecture with the same state shapes can supply
a payload of the same type. Neither convention alone proves that the intended weights were loaded.

The initial tensors are part of the initialized description. Execution instantiates a mutable
`nn.Module` from them, and the whole lifecycle is ordinary `IO`:

```lean (name := tpRun)
-- Instantiate on CPU, choose evaluation mode, and predict
-- from the initialized weights.
#eval do
  let module ← nn.Module.instantiate tpInit
    { device := .cpu }
  module.eval
  let output ← module.forward [0.1, 0.2, 0.3, 0.4]
  IO.println s!"{output}"
```
```leanOutput tpRun
[-0.284787, -0.230906]
```

The module owns the live parameters, persistent buffers, and train/eval mode, and `module.state`
reads the current payload back out. The immutable `nn.Sequential` remains the value used by graph
lowering and proofs. A theorem about `nn.initialState tpInit` is therefore not a theorem about the
parameters after 10,000 optimizer steps; a claim about trained weights must name the saved payload
it analyzes.

Lowering receives the initialized model and the tensors to analyze. This fixes the values in
the resulting graph artifact independently of later module updates. A PyTorch checkpoint importer
must map the external names, shapes, order, and layout into this payload.

The payload distinction also applies to layers beyond this MLP. For integer-indexed embeddings,
`nn.embedding vocab embedDim` constructs a trainable table,
`nn.Embedding.fromWeight weight` preserves an existing table exactly, and `freeze := true` keeps
that table in module state without requesting a parameter gradient. If the index tensor has
dimensions `dims`, its output has dimensions `dims ++ [embedDim]`; repeated ids accumulate into the
same gradient row.

# Autograd Implementations And Specifications

PyTorch dynamically constructs an autograd graph while tensor operations run. During backward,
saved tensors and derivative rules propagate vector-Jacobian products to leaves through its
dispatcher and autograd machinery
({Informal.citep pytorch2019 baydin2018}[]).

TorchLean's eager runtime records its own tape, including when its CUDA backend calls LibTorch.
The native bridge disables LibTorch autograd recording. ATen computes values and local gradients;
TorchLean decides which local rule to invoke and accumulates its results while traversing the tape.
To reason about a backward step, its proofs refer
to an ideal derivative or VJP definition. For covered operations, correctness theorems show that
this rule is the mathematical derivative; numerical results relate it to the rule evaluated with
the runtime node's saved values.

For a scalar loss $`L(\theta)`, reverse mode starts with derivative $`1` at the loss and
propagates derivatives backward. At a node $`y=f(x)`, let $`J_f(x)` be the Jacobian and
$`\bar{y}` the derivative of the loss with respect to the node's output, represented as a column
vector. The corresponding input derivative $`\bar{x}` is obtained by the local VJP:

$$`\bar{x}=J_f(x)^\mathsf{T}\bar{y}`.

Each node computes this product without materializing the full network Jacobian. The ideal VJP
states the intended rule; correctness and numerical error results describe its relationship to
the runtime implementation under their hypotheses.

External kernels make the three layers particularly important because forward and backward may
cross different boundaries. The backend chapter records that ownership per operation instead of
letting it hide inside the training loop.

## Gradient Comparison

To inspect a whole gradient, use a smaller network with two inputs, two hidden units, and one
output. Explicit weights make one hidden unit inactive and the other active at the chosen input,
so we can follow which derivatives survive ReLU in both implementations.

```lean (name := tpWeights)
-- Fix asymmetric weights and nonzero biases so the two
-- implementations see identical data.
def tpW1 : Tensor Float [2, 2] :=
  [[0.5, -0.5], [1.0, 0.25]]
def tpB1 : Tensor Float [2] := [0.1, -0.2]
def tpW2 : Tensor Float [1, 2] := [[1.5, -0.75]]
def tpB2 : Tensor Float [1] := [0.05]

def tpX : Tensor Float [2] := [1.0, 2.0]
def tpY : Tensor Float [1] := [0.5]

def tpNet : nn.Sequential [2] [1] :=
  nn.build 0 nn.Sequential![
    nn.linear 2 2,
    nn.relu,
    nn.linear 2 1
  ]

def tpState : autograd.model.State tpNet Float :=
  nn.State.empty
    |>.push tpW1 |>.push tpB1
    |>.push tpW2 |>.push tpB2
```

The seed `0` in `nn.build` only supplies the architecture and its shape list here, because the
payload we pass to autograd is `tpState`, replacing the model's initial values for this call.
The explicit payload lets us use exactly the weights in the hand calculation.

Write the first layer's weights and bias as $`W_1` and $`b_1`, and its pre-activation vector as
$`z`. For the input $`x` above,

$$`z=W_1x+b_1=(0.5-1.0+0.1,\; 1.0+0.5-0.2)=(-0.4,\;1.3)`,

so the hidden activation $`h=\operatorname{ReLU}(z)` has first coordinate zero. With the second
layer's weights $`W_2` and bias $`b_2`, the prediction is

$$`\hat{y}=1.5\cdot 0+(-0.75)\cdot 1.3+0.05=-0.925`,

giving squared error $`(-0.925-0.5)^2=2.030625`. The specification-level forward agrees:

```lean (name := tpSpecForward)
-- Evaluate the explicit payload before comparing its loss
-- derivatives.
#eval Examples.mlpForward (α := Float)
  { weights := tpW1, bias := tpB1 }
  { weights := tpW2, bias := tpB2 } tpX
```
```leanOutput tpSpecForward
[-0.925000]
```

Only the second hidden unit can pass a nonzero derivative. That gives us a local check before
comparing full parameter arrays: a nonzero gradient through the first unit would disagree with its
inactive ReLU.

For this calculation, $`y=0.5` is the target and $`\bar{y}` denotes the loss derivative with
respect to the prediction $`\hat{y}`. Bars on other quantities denote their loss derivatives.
The symbol $`\odot` means elementwise multiplication, and
$`\mathbf{1}[z>0]` is the vector of indicators for active ReLU units. Applying the chain rule
from the output toward the input, with $`\bar{y}=2(\hat{y}-y)=-2.85`, gives:

$$`\overline{W_2}=\bar{y}\,h^\mathsf{T}=(0,\;-3.705),\qquad \overline{b_2}=-2.85`,

$$`\bar{z}=\bar{y}\,W_2^\mathsf{T}\odot\mathbf{1}[z>0]=(0,\;2.1375)`,

$$`\overline{W_1}=\bar{z}\,x^\mathsf{T}=\begin{pmatrix}0&0\\2.1375&4.275\end{pmatrix},\qquad
  \overline{b_1}=(0,\;2.1375)`.

Now the runtime:

```lean (name := tpGradRun)
-- Request parameter cotangents and the scalar loss from the
-- same forward/backward run.
#eval do
  let (grads, loss) ← autograd.model.grad tpNet
    autograd.model.Loss.mse
    tpState tpX tpY (value := true)
  IO.println s!"loss  = {loss}"
  IO.println s!"grads = {reprStr grads}"
```
```leanOutput tpGradRun (whitespace := lax)
loss  = 2.030625
grads = [[2, 2]:
 [[0.000000, 0.000000], [2.137500, 4.275000]],
 [2]:
 [0.000000, 2.137500],
 [1, 2]:
 [[0.000000, -3.705000]],
 [1]:
 [-2.850000]]
```

Read the output groups in parameter order: first weights, first bias, second weights, second bias.
The first weight row is zero, and the second row is the hidden cotangent `2.1375` multiplied by
the input `[1, 2]`. The remaining groups match the bias and readout derivatives above.

And PyTorch, with the same four tensors copied in:

```
# Copy all four parameter tensors before comparing the loss
# and parameter gradients.
# Python / PyTorch
grad_net = torch.nn.Sequential(
    torch.nn.Linear(2, 2), torch.nn.ReLU(), torch.nn.Linear(2, 1))
with torch.no_grad():
    grad_net[0].weight.copy_(torch.tensor([[0.5, -0.5], [1.0, 0.25]]))
    grad_net[0].bias.copy_(torch.tensor([0.1, -0.2]))
    grad_net[2].weight.copy_(torch.tensor([[1.5, -0.75]]))
    grad_net[2].bias.copy_(torch.tensor([0.05]))

loss = torch.nn.functional.mse_loss(grad_net(torch.tensor([1.0, 2.0])),
                                    torch.tensor([0.5]))
loss.backward()
```

```
loss = 2.0306248664855957
0.weight grad = [0.0, 0.0, 2.1374998092651367, 4.274999618530273]
0.bias grad = [0.0, 2.1374998092651367]
2.weight grad = [0.0, -3.7049996852874756]
2.bias grad = [-2.8499999046325684]
```

The printed entries approximate the hand-derived values, including the zeros from the inactive
ReLU. The examples use different formats: TorchLean's `Float` is binary64, while PyTorch's
default dtype is binary32. Running the TorchLean example in binary32 lets us compare the stored
bits directly:

```lean (name := tpCastDefs)
/-- The same four tensors, cast into any runtime scalar. -/
def tpStateIn (α : Type) [Storage α] [Context α]
    [Runtime.FromFloat α] :
    autograd.model.State tpNet α :=
  let c : Float → α := Runtime.ofFloat
  nn.State.empty
    |>.push (Tensor.map c tpW1)
    |>.push (Tensor.map c tpB1)
    |>.push (Tensor.map c tpW2)
    |>.push (Tensor.map c tpB2)

/-- Gradient payload flattened in state order. -/
def tpFlat {α : Type} [Storage α] [Context α]
    (g : autograd.model.State tpNet α) : Array α :=
  match nn.State.Internal.toTensorPack g with
  | .cons w1 (.cons b1 (.cons w2 (.cons b2 .nil))) =>
      Tensor.to w1 (Array α) ++ Tensor.to b1 (Array α)
        ++ Tensor.to w2 (Array α)
        ++ Tensor.to b2 (Array α)
```

```lean (name := tpBits)
-- Flatten Float32 gradients in state order and compare
-- their exact stored words.
#eval do
  let g ← autograd.model.grad (α := Float32) tpNet
    autograd.model.Loss.mse
    (tpStateIn Float32)
    (Tensor.map Runtime.ofFloat tpX)
    (Tensor.map Runtime.ofFloat tpY)
  IO.println s!"{(tpFlat g).map Float32.toBits}"
```
```leanOutput tpBits
#[0, 0, 1074318540, 1082707148, 0, 1074318540, 0, 3228376759, 3224790630]
```

Flattening is part of this comparison's contract. Each tensor is flattened internally, then the
four groups are concatenated in the same weight, bias, weight, bias order. Reordering groups could
make a correct backward pass look wrong, or hide a swap if the chosen entries happened to agree.
The Python `view(torch.int32)` reads the storage words rather than rounding gradients to integers;
masking with `0xffffffff` displays those words as unsigned values. The comparison therefore checks
both this layout convention and all bits of these nine gradient entries for the chosen input.

```
# Reinterpret gradient storage as 32-bit words; do not
# numerically convert the values to ints.
# Python / PyTorch
bits = []
for _, p in grad_net.named_parameters():
    bits += p.grad.view(torch.int32).flatten().tolist()
print([b & 0xffffffff for b in bits])
```

```
[0, 0, 1074318540, 1082707148, 0, 1074318540, 0, 3228376759, 3224790630]
```

All nine 32-bit words match. Agreement between implementations does not imply correct rounding
of the exact real derivative, however. One entry has exact value $`2.85\times 0.75=2.1375`, whose
nearest binary32 representation differs from the shared word `1074318540`:

```lean (name := tpRound)
-- Compare rounding the final real derivative with the
-- recorded multistep Float32 result.
#eval (2.1375 : Float).toFloat32.toBits
#eval ((Float32.ofBits 1074318540).toFloat - 2.1375)
  * 1000000000.0
```
```leanOutput tpRound
1074318541
```
```leanOutput tpRound
-190.734863
```

Rounding the exact derivative gives `1074318541`. Both libraries returned `1074318540`, one ULP
lower, or about $`1.9\times 10^{-7}` below the exact value. The computation rounds intermediate
results, so it need not equal a single rounding of the final real expression. Matching those
intermediate operations can reproduce the same bits. Later in this chapter, a batched matrix
product illustrates how changing the evaluation order can break such agreement.

`Float.toString` prints six decimals, which would hide this discrepancy. Comparing stored words
establishes more than comparing those displays, but still only for this payload and input. The
guide reruns the Lean half; repeating the Python half is a separate cross-framework check.
The proof chapters instead state quantified results over specification VJPs for covered operations.

# Training Loops

A conventional PyTorch loop is explicit Python mutation:

```
# Clear old gradients before differentiating this batch and
# updating the parameters.
for x, y in loader:
    optimizer.zero_grad()
    prediction = model(x)
    loss = loss_fn(prediction, y)
    loss.backward()
    optimizer.step()
```

TorchLean's trainer packages the same lifecycle:

```lean (name := tpTrainer)
-- The output is [2], so its class axis is zero; this
-- definition configures a trainer.
def tpTrainer :=
  Trainer.new tpModel
    { objective := .oneHotCrossEntropy 0
      optimizer := optim.adam
        { learningRate := 0.001 }
      seed := 2026 }
```

The output is a vector of two class logits, so its class axis is zero. In
`oneHotCrossEntropy 0`, that argument selects the axis; the one-hot target of shape `[2]` selects
the desired class. The objective reduces the logits to a scalar loss. `tpTrainer` configures this
objective and Adam; a subsequent training call supplies the dataset and performs the updates.

Internally, the runtime owns mutable parameters, gradients, optimizer moments, tape state, and
possibly device buffers. TorchLean does not force a large GPU training loop to allocate a new pure
tensor tree at every step. The semantic interfaces remain explicit while the execution engine uses
mutation and ownership where performance requires it.

The `train` call returns a trained result whose prediction closures refer to the trained runner.
Lower-level manual APIs expose parameter tensors and individual forward, backward, and optimizer
steps when verification or research code needs them. The {ref "running-example"}[running example]
runs 200 steps and compares the results with a PyTorch loop, explaining the different batch sizes
and initializations.

## Momentum SGD

Reproducing a training run requires matching the optimizer's recurrence as well as its name.
Let $`p` be a parameter, $`g` its gradient, $`\eta` the learning rate, and $`\mu` the momentum
coefficient. The momentum form associated with Polyak {Informal.citep polyak1964}[] stores the
learning rate inside a buffer: $`v\gets\mu v-\eta g`, then $`p\gets p+v`. PyTorch uses a buffer
$`b` with the learning rate outside: $`b\gets\mu b+g`, then $`p\gets p-\eta b`.
For constant $`\eta`, scaling the buffers appropriately relates the recurrences; a changing
learning rate generally changes that relationship. TorchLean's `updateMomentumBuffer` follows
the PyTorch convention, as its docstring specifies.

Here are three steps on the same parameter and the same repeated gradient:

```lean (name := tpOptSgd)
-- Thread the returned momentum state through each update
-- with the same fixed gradient.
def tpTheta : Tensor Float [3] := [1.0, -2.0, 0.5]
def tpGradient : Tensor Float [3] := [0.1, 0.3, -0.2]

def tpSgdSteps (n : Nat) : Tensor Float [3] :=
  let rec go (k : Nat)
      (st : Optim.MomentumSGD.State Float [3])
      (p : Tensor Float [3]) : Tensor Float [3] :=
    match k with
    | 0 => p
    | k + 1 =>
      let step := Optim.MomentumSGD.update st p tpGradient
      go k step.optimizerState step.parameters
  go n (Optim.MomentumSGD.init 0.1 0.9 tpTheta) tpTheta

#eval tpSgdSteps 1
#eval tpSgdSteps 3
```
```leanOutput tpOptSgd
[0.990000, -2.030000, 0.520000]
```
```leanOutput tpOptSgd
[0.943900, -2.168300, 0.612200]
```

Starting from a zero buffer, the first three momentum buffers are `g`, `1.9 * g`, and
`2.71 * g`. The total displacement after three updates is therefore `0.1 * 5.61 * g` in exact
arithmetic. For the first coordinate that subtracts 0.0561 from one; for the third it adds 0.1122
to 0.5 because the gradient is negative. This explains the direction and scale of every coordinate
in the printed vector. Threading `step.optimizerState` is essential: reinitializing the buffer on
each call would erase this history and produce plain SGD behavior instead.

Use binary64 in PyTorch too, so the comparison holds the scalar format fixed:

```
# Reuse one fixed gradient for three momentum steps in
# binary64 arithmetic.
# Python / PyTorch
import torch
p = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64,
                 requires_grad=True)
g = torch.tensor([0.1, 0.3, -0.2], dtype=torch.float64)
opt = torch.optim.SGD([p], lr=0.1, momentum=0.9)
for _ in range(3):
    p.grad = g.clone()
    opt.step()
print(p.detach().tolist())
```

```
[0.9439, -2.1683, 0.6122000000000001]
```

For the third coordinate, PyTorch prints `0.6122000000000001` because that is
the binary64 number it holds; TorchLean prints `0.612200` because its printer stops at six decimals.
Lean confirms the stored value agrees when asked to compare against the full literal:

```lean (name := tpOptBits)
-- Check the full binary64 value behind the shorter
-- six-decimal display.
#eval (tpSgdSteps 3)[2] == 0.6122000000000001
```
```leanOutput tpOptBits
true
```

## Adam's Epsilon

Adam adds a second convention to compare: where the small positive constant $`\varepsilon`
enters the denominator {Informal.citep adam2015}[]. TorchLean adds it after the square root,
matching Kingma and Ba and PyTorch's `denom`. The constant keeps the denominator nonzero but
also changes the update for nonzero gradients. With a constant gradient, bias correction removes
the initial attenuation of the moment estimates, making that effect easy to isolate:

```lean (name := tpOptAdam)
-- Repeated identical gradients make the denominator's
-- epsilon effect easier to inspect.
def tpAdamSteps (n : Nat) : Tensor Float [3] :=
  let rec go (k : Nat)
      (st : Optim.Adam.State Float [3])
      (p : Tensor Float [3]) : Tensor Float [3] :=
    match k with
    | 0 => p
    | k + 1 =>
      let step := Optim.Adam.update st p tpGradient
      go k step.optimizerState step.parameters
  go n
    (Optim.Adam.init 0.1 0.9 0.999 1.0e-8 tpTheta) tpTheta

#eval tpAdamSteps 1
#eval tpAdamSteps 3
```
```leanOutput tpOptAdam
[0.900000, -2.100000, 0.600000]
```
```leanOutput tpOptAdam
[0.700000, -2.300000, 0.800000]
```

For a constant gradient and initially zero moments, bias correction makes the ideal first
moment equal to `g` and the ideal second moment equal to `g^2`. The normalized update is therefore
`learningRate * g / (abs g + epsilon)`. Coordinates with different nonzero gradient magnitudes
move by nearly the same amount. This cancellation depends on holding the gradient fixed.

With this constant nonzero gradient, each coordinate moves approximately $`0.1` per step, in the
opposite direction to its gradient. Epsilon and rounding change the magnitude slightly; changing
gradients need not produce the same step size. PyTorch prints the same
run, after one step and after three, with more digits:

```
# Hold the gradient fixed to isolate Adam's bias correction
# and epsilon placement.
# Python / PyTorch
def adam_run(steps):
    p = torch.tensor([1.0, -2.0, 0.5],
                     dtype=torch.float64,
                     requires_grad=True)
    g = torch.tensor([0.1, 0.3, -0.2],
                     dtype=torch.float64)
    opt = torch.optim.Adam([p], lr=0.1,
                           betas=(0.9, 0.999), eps=1e-8)
    for _ in range(steps):
        p.grad = g.clone()
        opt.step()
    return p.detach().tolist()

print(adam_run(1))
print(adam_run(3))
```

```
[0.900000009999999, -2.099999996666667, 0.5999999950000002]
[0.7000000299999976, -2.29999999, 0.799999985]
```

Scaling the deviation from the nominal step size reveals the effect hidden by six-decimal
printing:

```lean (name := tpOptEps)
-- Magnify the residual relative to an idealized step of
-- exactly 0.1 per update.
#eval ((tpAdamSteps 1)[0] - 0.9) * 1.0e9
#eval ((tpAdamSteps 3)[0] - 0.7) * 1.0e9
```
```leanOutput tpOptEps
9.999999
```
```leanOutput tpOptEps
29.999998
```

Write the bias-corrected first and second moments as $`\hat m` and $`\hat v`. For the first
coordinate the residual is approximately $`10^{-8}` per step. The update is
$`\eta\hat m/(\sqrt{\hat v}+\varepsilon)` rather than $`\eta\hat m/\sqrt{\hat v}`, so it falls short
of $`\eta` by the relative factor $`\varepsilon/(|g|+\varepsilon)` in exact arithmetic.
PyTorch's `0.900000009999999` is the same shortfall.
Three steps accumulate three times as much, which the second number confirms.

The results check the placement of epsilon for this trajectory. Double precision makes its small
effect visible; with these parameter values, Float32 can hide it below one ULP.
{ref "fp32-soundness"}[The soundness chapter] separates the update equation from the format in
which it is evaluated so that both sources of numerical behavior can be accounted for.

# Dispatch And Backend Selection

PyTorch routes an operation through its dispatcher. Device, dtype, layout, compilation state, and
available libraries determine the eventual implementation. A CUDA matrix multiplication may use
cuBLAS, convolution may use cuDNN, and attention may select one of several fused kernels.

TorchLean records the corresponding choice as a device, operation, provider, and accepted kernel
capsule. Reports name the implementation selected for each operation, and unavailable requests
fail instead of becoming an unreported CPU run. This keeps the model, parameter layout, graph, and
tape stable while selected operations cross a native boundary.

{ref "backend-selection"}[Backend Selection]
explains capsules, provider preference, local VJP ownership, and assurance policies in full.
{ref "gpu-and-cuda"}[GPU and CUDA] then follows
the native CUDA boundary.

## Backend Dispatch Reports

For the four-input model, we can inspect how each system selects a `linear` implementation.
PyTorch's dispatcher uses a key set derived from the tensors and execution context:

```
>>> # Inspect CPU and CUDA dispatch keys, then count the
>>> # recorded linear registrations.
>>> torch._C._dispatch_key_set(torch.zeros(4))
DispatchKeySet(CPU, ADInplaceOrView, AutogradCPU, AutocastCPU)
>>> torch._C._dispatch_key_set(torch.zeros(4, device='cuda'))
DispatchKeySet(CUDA, ADInplaceOrView, AutogradCUDA, AutocastCUDA)
>>> d = torch._C._dispatch_dump("aten::linear")
>>> len([l for l in d.splitlines() if ': registered at' in l])
15
```

Fifteen registrations for one operator, and the key set picks among them. TorchLean asks the same
question of a profile, before anything executes, and the answer is a value you can print:

```lean (name := tpDispatch)
-- Ask each declared backend profile for a plan without
-- executing its kernels.
section
open NN.Backend

/-- A CUDA request against a build that declares only CPU
availability. The model, the graph, and the parameters are
untouched; the policy is the only thing that changed. -/
def tpCudaOnCpuBuild : BackendProfile :=
  { BackendProfile.checkedCpu with
    name := "cuda_request_cpu_build"
    policy :=
      { BackendProfile.checkedCpu.policy with
        device := .cuda } }

#eval show IO Unit from do
  let l ← IO.ofExcept
    (Verification.lowerForwardToIR (α := Float)
      tpInit (nn.initialState tpInit))
  for p in [BackendProfile.checkedCpu,
      BackendProfile.checkedCuda, tpCudaOnCpuBuild] do
    match p.planGraphNodes l.graph with
    | .ok plan =>
        let names := (plan.kernels.map
          (·.capsule.name)).toList.eraseDups
        IO.println s!"{p.name}: {names.length} capsules"
        for n in names do IO.println s!"  {n}"
    | .error e =>
        IO.println s!"{p.name}: rejected"
        IO.println s!"  {e}"

end
```

```leanOutput tpDispatch (whitespace := lax)
checked_cpu: 6 capsules
  reference.reshape
  reference.permute
  reference.matmul
  reference.broadcast
  reference.add
  reference.relu
checked_cuda: 6 capsules
  libtorch.reshape
  libtorch.permute
  libtorch.matmul
  libtorch.broadcast
  libtorch.add
  libtorch.relu
cuda_request_cpu_build: rejected
  no admissible kernel capsule for op reshape on device cuda
```

The three profiles describe different plans for the same eighteen-node graph. The CPU profile
selects six reference capsules; the CUDA profile selects LibTorch capsules for those operations.
The third request asks for CUDA while declaring only CPU availability. Planning rejects it and
names the unsupported operation before any allocation or launch.

The failure is in the runtime configuration: the graph's shapes are valid, but the declared
providers cannot implement the requested operation on CUDA. Conversely, a successful plan only
shows that the graph fits the profile's declarations. This example does not execute the kernels.

PyTorch also rejects incompatible device placement:

```
>>> # The layer retains CPU parameters while its input is
>>> # allocated on CUDA.
>>> torch.nn.Linear(4, 2)(torch.zeros(4, device='cuda'))
RuntimeError: Expected all tensors to be on the same device, but got mat1 is
on cuda:0, different from other tensors on cpu (when checking argument in
method wrapper_CUDA_addmm)
```

The difference is when the decision happens and what carries it. PyTorch's answer is a function of
ambient tensor state at the moment of the call, spread across device, dtype, autocast mode, and grad
mode. Dispatch inspection can expose parts of that choice before execution, as the key-set
example above shows. TorchLean's answer is a function of one record
you can name, print, compare against another profile, and hand to a proof. The `availability` field
is what the build declares rather than a probe of the driver. Planning stays a
pure function of the profile and the graph, so the same plan comes out on a machine with no GPU at
all, and a host that cannot honour it fails at the capsule boundary instead of quietly producing a
different plan. {ref "backend-selection"}[The backend planner chapter] is where that record, and the
limits of what a plan currently controls, are laid out.

# Graphs, Lowering, And Compilation

PyTorch offers FX, `torch.export`, AOTAutograd, and compiler stacks such as `torch.compile`.
Their graphs support transformation and deployment inside the PyTorch ecosystem.

TorchLean has two graph-facing layers with different purposes:

- `GraphSpec` describes structured architectures and lowers them to TorchLean programs;
- `NN.IR.Graph` is the canonical operation DAG used by lower-level evaluation and verification.

An IR node contains an operation tag, parent ids, and an output shape. Parameter and constant values
live in payload stores. `NN.IR.Semantics` defines how supported nodes are interpreted over a scalar
domain.

The two graphs for our four-input MLP are a good way to see the difference in granularity.
`torch.export` produces nine nodes:

```
# Export the existing model and inspect how this graph
# represents its two affine layers.
# Python / PyTorch
exported = torch.export.export(net, (torch.zeros(4),))
for node in exported.graph_module.graph.nodes:
    print(f"  {node.op:15s} {node.target}")
```

```
placeholder     p_0_weight
placeholder     p_0_bias
placeholder     p_2_weight
placeholder     p_2_bias
placeholder     input
call_function   aten.linear.default
call_function   aten.relu.default
call_function   aten.linear.default
output          output
```

TorchLean's verification lowering produces eighteen:

```lean (name := tpNodes)
-- Inspect the successful lowering's node count and boundary
-- ids, not a timing measurement.
#eval (Verification.lowerForwardToIR (α := Float)
    tpInit (nn.initialState tpInit)).map fun l =>
  (l.graph.nodes.size, l.inputId, l.outputId)
```
```leanOutput tpNodes
Except.ok (18, 0, 17)
```

`aten.linear.default` retains the affine layer as one graph operation; its execution can depend
on later dispatch and compilation choices. TorchLean's verification graph expands that layer into
reshapes, a matrix product, a broadcast, and an addition. Each primitive has a semantic equation
used when reasoning through the graph one node at a time. The
{ref "graphs-and-ir"}[graph chapter] lists the node kinds.

The counts describe this difference in granularity. Eighteen nodes versus nine gives no direct
comparison of memory or speed; those depend on how the operations execute.

The verification lowering pass can lower supported initialized models and parameter payloads to
this IR. A separate first-order source language under `NN.Verification.Builtin.Proved` has an
end-to-end lowering-correctness theorem. They share an IR target, but the theorem applies to the
proved source fragment, not automatically to every model accepted by the broader lowering pass.

TorchLean calls the conversion to `NN.IR.Graph` *verification lowering*. Typed graph lowering is a
separate path: it records a shape-indexed SSA graph whose nodes store forward and differentiation
functions. Derivative correctness is established by separate theorems over proof-carrying nodes and
graphs. Neither lowering path performs optimization, fusion, scheduling, or native code generation,
so selecting typed graph execution does not imply a `torch.compile`-style compiler.

# Checkpoints And Graph Import

PyTorch checkpoints are Python-oriented zip/pickle artifacts. TorchLean does not duplicate that
loader inside Lean. Its adapter generates Python that asks PyTorch to load a `state_dict` and emit
named tensor data as plain JSON. Lean then parses each requested tensor into a statically known
shape.

Graph import is separate. A generated adapter uses `torch.export` or FX to emit the
`torchlean.ir.v1` format, which Lean parses into `NN.IR.Graph`. The ONNX adapter lowers supported
static nodes to the same format.

Parsing checks the JSON schema and requested tensor shapes; graph checks validate node references
and shape contracts. These catch a malformed export, but a transposed square weight matrix can
pass them. A value comparison can expose that error. To prove equivalence with the source graph,
we would need refinement for each operator and a bridge from graph semantics to the deployed
kernels. The {ref "pytorch-roundtrip"}[round trip chapter] exercises the import and comparison
on a real checkpoint.

# Floating Point

PyTorch's numerical behavior depends on dtype, device, library versions, compiler transformations,
and reduction algorithms. Its numerical-accuracy documentation explicitly warns that mathematically
identical computations are not guaranteed to be bitwise identical across batched, sliced, device,
or backend paths.

Comparing a batched product with the product of one selected slice exposes this issue:

```
# Compare a row computed as part of a batch with the same
# row computed on its own.
# Python / PyTorch
a, b = torch.randn(64, 128), torch.randn(128, 128)
full, sliced = a @ b, a[:1] @ b
print(torch.equal(full[:1], sliced))
print((full[:1] - sliced).abs().max().item())
print(full[0, 0].view(torch.int32).item() & 0xffffffff,
      sliced[0, 0].view(torch.int32).item() & 0xffffffff)
```

```
False
8.58306884765625e-06
3206383400 3206383430
```

`False` says at least one entry differs; the next line gives the maximum absolute difference
across the row. The integer words describe only the first entry, which need not attain that
maximum. The snippet does not fix a random seed, so these are results from the recorded run.

On the recorded platform, these two mathematically equal products differed in their computed
binary32 values. Blocking and reduction choices can change rounding; the displayed first entry
differs by thirty ULPs. Other platforms or inputs may give equal results. A claim about the
executed output may therefore need to include the batching policy: grouping requests can change
the numerical computation even when the ideal function acts independently on each row.

TorchLean names several numerical meanings so that a statement can pick one:

- real-valued specifications for ideal mathematics;
- `NF`, a configurable rounded-real arithmetic;
- `FP32`, a rounded-real specialization with binary32 precision and gradual underflow, but without
  an upper exponent bound or IEEE special values;
- FloatLib binary32, an executable bit-level binary32 model;
- runtime CPU representations and LibTorch CUDA tensors.

For batching, the library states a property of the reference semantics. If a batched run is
defined by applying the same function independently to each row, then
reading one row back out is that row's own result:

```lean (name := tpBatch)
-- Read the precise rowwise equation guaranteed for the pure
-- mapLeading construction.
open TorchLean.Tensor in
#check @unstack_mapLeading
```
```leanOutput tpBatch (whitespace := lax)
@unstack_mapLeading : ∀ {α : Type} [inst : Storage α] {batch : ℕ}
  {inShape outShape : Shape} (f : Tensor α inShape → Tensor α outShape)
  (xs : Tensor α (inShape.prependDim batch)) (i : Fin batch),
  (mapLeading [batch] f xs).unstack i = f (xs.unstack i)
```

The quantifiers range over an element type, a batch length, input and output shapes, a function
`f`, the batch tensor `xs`, and a valid row index `i`. The left side applies `f` independently
through `mapLeading` and then selects row `i`; the right side first selects that row and applies
`f` once. Equality says those two constructions agree for every such argument. The bounded index
`Fin batch` ensures the selected row exists. The hypothesis is built into the construction: each
row is processed by the same pure function, without consulting the other rows.

This theorem establishes batch invariance for the reference semantics in
{src "NN/Examples/BugZoo/BatchInvariance.lean"}[`BatchInvariance.lean`]. Applying it to a cuBLAS
or other native kernel requires a contract relating that kernel to the specification. The named
property states exactly what such a contract would need to preserve.

The generic layer was influenced by Flocq's separation of formats from rounding operators
({Informal.citep flocq2011}[]), and the CompCert float work
({Informal.citep boldo2015}[]) addresses connections between floating-point semantics and
compiled code. FloatLib supplies executable binary formats, with binary32 as the familiar example
here. The
{ref "floats"}[floating-point chapters] derive these layers from examples and show how they
reconnect to a runtime.

# Verification Artifacts

For this MLP, TorchLean exposes a sequence of objects connecting the model to a verification
claim:

```
shape-typed model
  -> initialized parameter layout
  -> runtime tape and trained payload
  -> canonical operation graph
  -> exact or rounded arithmetic
  -> bound or certificate
  -> theorem with named assumptions
```

Each arrow is an interface that can be tested, checked, or proved independently as coverage grows.

# PyTorch Integration

Training can remain in PyTorch while TorchLean checks a supported exported artifact, or run in
TorchLean while selected operations use PyTorch kernels. In either arrangement, a claim about the
executed model must identify the imported weights, graph semantics, and kernel contracts.

# References

The system compared against throughout this chapter is described in
{Informal.citet pytorch2019}[], and the reverse-mode machinery both libraries implement is surveyed
in {Informal.citet baydin2018}[]. The floating-point design references are
{Informal.citet flocq2011}[] and {Informal.citet boldo2015}[]. The PyTorch documentation pages used
for the transcripts above are:

- [`nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html);
- [`nn.Embedding`](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html);
- [`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html);
- [autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html);
- [`torch.export`](https://docs.pytorch.org/docs/stable/export.html);
- [numerical accuracy notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html).

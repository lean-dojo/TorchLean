import VersoManual
import NN.API
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Functional Programming" =>
%%%
tag := "why_functional"
file := "Why-The-Model-Is-Written-As-A-Function"
%%%

For an input space $`X`, an output space $`Y`, and parameters $`\theta`, a neural network is
usually introduced as a function

$$`f_\theta : X\to Y`.

The notation already tells us something useful: the output depends on an input $`x` and on
parameters $`\theta`. Real training code depends on more than that. Batch normalization reads and
updates running statistics {Informal.citep batchnorm2015}[]. Dropout depends on a mode and on a
random mask {Informal.citep dropout2014}[]. An optimizer carries momentum or moment estimates
{Informal.citep polyak1964}[]. A checkpoint loader reads bytes from a file. A GPU execution owns
buffers whose lifetime matters.

TorchLean makes these dependencies explicit in arguments, results, and effectful interfaces.
That gives a theorem access to the same parameters and state that determine a computation.
For example, an optimizer update receives its old momentum buffer and returns the new one;
a forward pass receives the mode that determines whether dropout applies. The corresponding
PyTorch interfaces {Informal.citep pytorch2019}[] expose those dependencies through mutable objects.

# An Affine Model With Explicit Parameters

An affine function isolates the relationship between a model's parameters and its input.
The record stores a weight and bias; `forward` receives both the record and the input:

```lean (name := wfAffineEval)
-- Store the coefficients as a value and apply the same
-- forward equation at input three.
structure WfAffine where
  weight : Float
  bias : Float
deriving Repr

def WfAffine.forward (p : WfAffine) (x : Float) :
    Float :=
  p.weight * x + p.bias

def wfSmall : WfAffine :=
  { weight := 2.0, bias := 0.5 }

#eval WfAffine.forward wfSmall 3.0
```
```leanOutput wfAffineEval
6.500000
```

The result of {lean}`WfAffine.forward` depends on its {lean}`WfAffine` and {lean}`Float`
arguments. Fixing them determines the computation, so a proposition can refer to that same
application:

```lean (name := wfAffineProof)
-- This particular closed Float computation reduces to the
-- exact displayed result.
example : WfAffine.forward wfSmall 3.0 = 6.5 := by
  rfl
```

The coefficients and intermediate values are exactly representable, so `rfl` can unfold the
definitions to the same Float value on each side. The proposition names the parameter record and
input directly; those arguments identify the entire computation. This proof works by reduction.
It does not give Float arithmetic the algebraic laws of the reals.

## Floating-Point Equality

The same tactic fails for an equality that holds over the reals but is false in binary64:

```lean +error (name := wfFloatFail)
-- Decimal notation does not make binary floating-point
-- addition obey rational arithmetic.
example : (0.1 + 0.2 : Float) = 0.3 := by rfl
```
```leanOutput wfFloatFail (whitespace := lax)
Tactic `rfl` failed: The left-hand side
  0.1 + 0.2
is not definitionally equal to the right-hand side
  0.3

⊢ 0.1 + 0.2 = 0.3
```

The two sides are different Float values. We can state the equality that does hold, then magnify
the difference from `0.3`:

```lean (name := wfFloatTrue)
-- State the actual Float equality, then magnify its
-- difference from the decimal 0.3.
example :
    (0.1 + 0.2 : Float) = 0.30000000000000004 := by
  rfl

#eval ((0.1 + 0.2 : Float) - 0.3) * 1e18
```
```leanOutput wfFloatTrue
55.511151
```

The sum sits about $`5.55\times 10^{-17}` above the nearest double to $`0.3`, which is one unit in
the last place at that magnitude. Notice that `#eval` on the sum by itself would have printed
`0.300000`, because {lean}`Float`'s printing shows six decimals. Of the two tools in this section,
the proof checks equality directly, while the scaled numerical difference exposes what decimal
printing hides.

Purity fixes which computation we mean; its arithmetic determines which equalities hold.
TorchLean can execute {lean}`Float` and {lean}`Float32` computations, describe IEEE-754 rounding
explicitly, and state specifications over $`\mathbb{R}`. Connecting these interpretations requires
choosing a format and a rounding rule, a distinction also developed in Flocq
{Informal.citet flocq2011}[]. Goldberg's survey explains the effects of representation and
rounding {Informal.citep goldberg1991}[]. The float chapter,
{ref "floats"}[Floats And Rounded Arithmetic], works through the machinery, and
{ref "runtime-approximation"}[Runtime Approximation] connects a rounded run back to a real-valued
specification.

# Parameter Payloads

TorchLean's layer representation takes the same idea to tensor scale. A
{src "NN/Runtime/Autograd/Model/Layers/Core.lean"}[`Layer σ τ`] records:

- `stateShapes`, the shapes of its parameters and persistent buffers, in the order `forward`
  expects them;
- `initState`, initial values for that state, as `Float` tensors;
- `requiresGrad`, one flag per state entry, so buffers can be marked as not differentiated;
- `updateBuffers`, an optional function for running statistics;
- `forward`, a program that takes a mode and then the state followed by the input.

Every one of those is a field of an ordinary structure. Training replaces the parameter values many
times without touching the architecture, because the architecture is the value that holds the
shapes and the forward program.

For a two-layer MLP:

```lean (name := wfShapes)
-- Initialize a fixed architecture and inspect its ordered
-- state shapes.
def wfMlp : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![
    nn.linear 2 4,
    nn.relu,
    nn.linear 4 1
  ]

def wfInit := nn.build 2026 wfMlp

#eval (nn.stateShapes wfInit).map Shape.toList
```
```leanOutput wfShapes
[[4, 2], [4], [1, 4], [1]]
```

Those are the first weight matrix, the first bias, the second weight matrix, and the second bias.
The order is part of the forward program's type rather than a convention to remember, and
{lean}`nn.initialState wfInit` is the matching initial payload:

```lean (name := wfPayload)
-- The state's type records this architecture's complete
-- parameter and buffer layout.
#check nn.initialState wfInit
```
```leanOutput wfPayload (whitespace := lax)
nn.initialState wfInit Float : nn.State Float
  (Runtime.Autograd.Model.Layers.Seq.stateShapes wfInit)
```

The state is indexed by a list of shapes, not just by the number of entries in that list. Four
tensors of the wrong extents therefore cannot stand in for these four tensors. Within a layer,
`requiresGrad` identifies which state entries are trainable, while persistent buffers can remain
part of the same state interface. A forward pass may read both. This distinction is needed for
layers such as batch normalization: an optimizer updates learned coefficients, whereas a buffer
update records running statistics under a separate rule.

A trained runtime owns a later payload with that same ordered shape list. Models with incompatible
state shapes cannot exchange payloads directly; models with identical state shapes still need a
value-level check to establish which parameters they use.

## Checking Incompatible State Shapes

Input and output shapes alone do not determine the parameter layout. A second model can have
the same interface but a different hidden width. Trying to use its initial payload for the first
model exposes the difference:

```lean +error (name := wfPayloadClash)
-- Identical public input/output shapes do not make
-- different hidden-state layouts compatible.
def wfWide : nn.Sequential [2] [1] :=
  nn.build 2026 <| nn.Sequential![
    nn.linear 2 6,
    nn.relu,
    nn.linear 6 1
  ]

example : nn.State Float
    (Runtime.Autograd.Model.Layers.Seq.stateShapes
      wfInit) :=
  nn.initialState wfWide
```
```leanOutput wfPayloadClash (whitespace := lax)
Type mismatch
  nn.initialState wfWide Float
has type
  nn.State Float
    (Runtime.Autograd.Model.Layers.Seq.stateShapes wfWide)
but is expected to have type
  nn.State Float
    (Runtime.Autograd.Model.Layers.Seq.stateShapes wfInit)
```

Both models map $`\mathbb{R}^2` to $`\mathbb{R}^1`, both have four state entries, and both were
built from the same seed. The one thing that differs is a hidden width, and that is enough: a
payload is a {src "NN/Tensor/Pack.lean"}[`TensorPack`], an inductive family indexed by the list of
shapes it holds, so `[[4, 2], [4], [1, 4], [1]]` and `[[6, 2], [6], [1, 6], [1]]` name two
different types. A loader must check external data before constructing either payload, and a
theorem about one payload cannot be applied directly to a value of the other type. Equal shape
lists would still leave the separate obligation to identify the parameter values.

The same distinction appears in the verification interfaces. Soundness statements in
{ref "verification"}[the verification chapter] mention their parameter payloads, and
{ref "graphs-and-ir"}[the IR chapter]'s `denote` takes a payload as an argument. The types
establish that the payload fits; the explicit argument identifies which values the claim concerns.

Try changing the hidden width from `4` to `6` in both linear layers. The printed shapes become
`[[6, 2], [6], [1, 6], [1]]`. Now change only the second layer, to `nn.linear 6 1`, and the model
stops elaborating:

```lean +error (name := wfWidthClash)
-- Change only the final layer's input width to expose the
-- broken composition.
def wfBroken : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![
    nn.linear 2 4,
    nn.relu,
    nn.linear 6 1
  ]
```
```leanOutput wfWidthClash (whitespace := lax)
nn.Sequential!: layer 2 expects input shape [6], but layer 1 outputs [4].
Change layer 2's input shape or insert a layer that converts [4] to [6].
```

The ReLU hands on a length-four vector and the final layer demands length six, so composition has
no type. PyTorch reports the same disagreement, but as a `RuntimeError` from the first batch that
reaches the offending matrix multiply;
{ref "torchlean_vs_pytorch"}[TorchLean and PyTorch] puts the two messages side by side.

## Counting From State Shapes

Because state shapes are part of the architecture value, the count itself is ordinary arithmetic
on a list of naturals and does not inspect tensor entries. The initialized model also carries its
initial tensor values:

```lean (name := wfCount)
-- Count stored scalars from shapes, independently of the
-- numerical initialization values.
def wfParamCount {σ τ : Shape}
    (m : nn.Sequential σ τ) : Nat :=
  ((nn.stateShapes m).map Shape.size).sum

#eval wfParamCount wfInit
#eval (wfParamCount (nn.build 7 wfMlp),
  wfParamCount (nn.build 2026 wfMlp))
```
```leanOutput wfCount
17
```
```leanOutput wfCount
(17, 17)
```

Seventeen is $`4\times 2` weights plus $`4` biases plus $`1\times 4` weights plus one bias, and the
second line checks the same count for another seed. The shapes come from the builder, and the
seed only decides what goes into those shapes, so
changing the seed of this builder does not change the parameter budget. An arbitrary custom
builder can inspect the seed and choose a different architecture, so this is a property of the
fixed layer construction, not of the `Builder` type alone.

The count is not merely computable, it is checkable by the kernel:

```lean (name := wfCountProof)
-- Natural-number shape arithmetic can establish this fixed
-- count by reduction.
example : wfParamCount wfInit = 17 := by rfl
```

Lean unfolds `wfParamCount` and sums the fixed model's shape sizes to seventeen. The theorem is
about `wfInit`; a result for arbitrary builders or seeds would need to quantify over them.
PyTorch's corresponding
expression is `sum(p.numel() for p in model.parameters())`, which obtains the counts from the
module's parameter tensors. The Lean helper computes from the state shape list, though this
initialized model also contains tensor values. It counts all state entries, including persistent
buffers in models that have them; for this MLP, that equals the trainable parameter count.

This proof works by reducing arithmetic on a list of natural numbers. Later, an optimizer
example will need storage lookup laws to reason about packed tensor entries; those computations
do not all reduce in the same way.

# Seeded Initialization

The linear layers need random initial weights. Rather than drawing them from an unnamed global
generator, layer constructors return an {lean}`nn.Builder`, which is a deterministic computation
over a seed stream. `nn.build` runs that computation with an initial seed:

```lean (name := wfBuildType)
-- Expose the pure function from a seed and builder to the
-- value they construct.
#check @nn.build
```
```leanOutput wfBuildType
@nn.build : {α : Type u_1} → ℕ → nn.Builder α → α
```

Here `α` is the type of value the builder produces, a sequential model in this example.
`nn.build` takes the seed and builder and returns that value directly, without `IO`. Repeating
the same arguments gives the same value. A custom builder could use the seed to choose an
architecture too; our fixed builder uses it to draw the weights:

```lean (name := wfSeeds)
-- Repeat one seed and compare another while holding the
-- builder fixed.
def wfModel : nn.Builder (nn.Sequential [2] [2]) :=
  nn.Sequential![nn.linear 2 2]

def wfFirst := nn.build 2026 wfModel
def wfSecond := nn.build 2026 wfModel
def wfOther := nn.build 7 wfModel

#eval do
  let same := reprStr (nn.initialState wfFirst) ==
    reprStr (nn.initialState wfSecond)
  IO.println s!"2026: {reprStr (nn.initialState wfFirst)}"
  IO.println s!"   7: {reprStr (nn.initialState wfOther)}"
  IO.println s!"2026 twice equal: {same}"
```
```leanOutput wfSeeds (whitespace := lax)
2026: [[2, 2]: [[-0.534110, 0.833175], [-1.213642, 0.820955]], [2]: [0.000000, 0.000000]]
   7: [[2, 2]: [[0.964935, -0.659584], [0.073548, -1.193925]], [2]: [0.000000, 0.000000]]
2026 twice equal: true
```

The two seeds give different weight matrices and the same zero bias. The final Boolean compares
formatted state strings, so it checks only what the printer exposes. Exact reproducibility follows
from applying the pure `nn.build` definition to the same seed and builder.

The builder's seed stream is independent of unrelated random draws. Repeating the build after
drawing randomness and initializing another model gives:

```lean (name := wfNoise)
-- An unrelated IO draw and another builder do not advance
-- this explicit seed stream.
#eval do
  let _ ← (IO.rand 0 1000 : IO Nat)
  let noise := nn.build 99 (nn.Sequential![nn.linear 8 8])
  let again := nn.build 2026 wfModel
  let same := reprStr (nn.initialState again) ==
    reprStr (nn.initialState wfSecond)
  IO.println s!"unrelated build: {nn.stateShapes noise}"
  IO.println s!"still equal: {same}"
```
```leanOutput wfNoise
unrelated build: [[8, 8], [8]]
still equal: true
```

## PyTorch Initialization

A fresh PyTorch module normally draws its weights from the default generator. Resetting that
generator reproduces the initialization only if the intervening draws also agree:

```
# Compare initialization before, after, and with an extra
# draw following a seed reset.
import torch, torch.nn as nn

a = nn.Linear(2, 2)
b = nn.Linear(2, 2)
print(torch.equal(a.weight, b.weight))

torch.manual_seed(2026); c = nn.Linear(2, 2)
torch.manual_seed(2026); d = nn.Linear(2, 2)
print(torch.equal(c.weight, d.weight))

torch.manual_seed(2026); e = nn.Linear(2, 2)
torch.manual_seed(2026); _ = torch.randn(1); f = nn.Linear(2, 2)
print(torch.equal(e.weight, f.weight))
```

The recorded PyTorch 2.13 run prints:

```
False
True
False
```

The middle line confirms that resetting the seed reproduces the weights. In the third line,
one extra draw advances the generator before construction, so the same architecture receives
different weights. Passing a seed directly to the TorchLean builder fixes the starting stream for
that construction. Reproducing an entire training run also requires fixing the other choices
described in PyTorch's reproducibility notes, linked below.

## Initial And Trained Parameters

Passing a seed to a trainer has a different meaning depending on whether its model is already
initialized. Compare these two calls:

```lean (name := wfTrainers)
-- Contrast a builder that consumes the config seed with a
-- model that is already initialized.
def wfFromBuilder :=
  Trainer.new wfMlp
    { objective := .mse, seed := 2026 }

def wfFromValue :=
  Trainer.new wfInit
    { objective := .mse, seed := 999 }

example : Tensor Float [2] → IO (Tensor Float [1]) :=
  wfFromBuilder.predict
example : Tensor Float [2] → IO (Tensor Float [1]) :=
  wfFromValue.predict
```

Both prediction methods take a length-two Float tensor and return an `IO` action producing
length one. The first trainer uses `2026` to run the builder. The second receives `wfInit`,
which already has its weights, so `999` does not reinitialize them. This allows a trainer to
preserve a deliberately constructed or loaded payload.

The `Trainer.ToModel` instance chosen during elaboration determines which construction applies.
The identical prediction types do not identify the weights: after training, an analysis must
still read the updated payload.

# Optimizer State Transitions

Let $`\theta_t` be the parameter tensor at step $`t`, $`g_t` its loss gradient, and $`\eta` the
learning rate. Plain gradient descent updates the parameters by

$$`\theta_{t+1}=\theta_t-\eta g_t`,

and momentum {Informal.citep polyak1964}[] adds a velocity $`v_t` whose previous value is
weighted by the coefficient $`\mu`:

$$`
\begin{aligned}
v_{t+1} &= \mu v_t+g_t,\\
\theta_{t+1} &= \theta_t-\eta v_{t+1}.
\end{aligned}
`

The step returns both the updated parameters and the updated velocity. TorchLean packages these
as parameters and optimizer state:

```lean (name := wfStepType)
-- An optimizer step packages new parameters together with
-- the state needed by the next step.
#check @Optim.Step
```
```leanOutput wfStepType
Optim.Step : (α : Type) → [Storage α] → Shape → Type → Type
```

`Optim.Step α s σ` pairs the next parameters, of shape `s`, with the next optimizer state, of type
`σ`. The momentum optimizer is then a function from the old triple to that pair:

```lean (name := wfMomentumType)
-- Read the state, parameter, and gradient arguments before
-- the returned Step type.
#check @Optim.MomentumSGD.update
```
```leanOutput wfMomentumType (whitespace := lax)
@Optim.MomentumSGD.update : {α : Type} →
  [inst : Storage α] →
    [inst_1 : Context α] →
      [DecidableRel fun x1 x2 => x1 > x2] →
        {s : Shape} →
          Optim.MomentumSGD.State α s → Tensor α s →
            Tensor α s →
              Optim.Step α s (Optim.MomentumSGD.State α s)
```

The next call needs both fields of `Step`. Passing new parameters with an old momentum buffer
changes the recurrence. The shared shape variable aligns parameters, gradients, and buffer entries
coordinate by coordinate; the storage and arithmetic instances supply their representation and
operations.

Let us run two steps of it from $`\theta_0 = 2`, with $`\eta = 0.1`, $`\mu = 0.9`, and a constant
gradient $`g = 0.25`. By hand: $`v_1 = 0.25`, so $`\theta_1 = 2 - 0.025 = 1.975`; then
$`v_2 = 0.9\cdot 0.25 + 0.25 = 0.475`, so $`\theta_2 = 1.975 - 0.0475 = 1.9275`.

```lean (name := wfMomentum)
-- Carry the momentum buffer forward so the second update
-- includes the first gradient.
def wfTheta : Tensor Float [1] := [2.0]
def wfGrad : Tensor Float [1] := [0.25]

#eval do
  let s0 := Optim.MomentumSGD.init (α := Float)
    0.1 0.9 wfTheta
  let step1 := Optim.MomentumSGD.update s0 wfTheta wfGrad
  let step2 := Optim.MomentumSGD.update
    step1.optimizerState step1.parameters wfGrad
  IO.println s!"step 1: {step1.parameters}, \
    buffer {step1.optimizerState.momentumBuffer}"
  IO.println s!"step 2: {step2.parameters}, \
    buffer {step2.optimizerState.momentumBuffer}"
```
```leanOutput wfMomentum (whitespace := lax)
step 1: [1.975000], buffer [0.250000]
step 2: [1.927500], buffer [0.475000]
```

The gradient stays fixed, but the second update grows because the buffer retains the first
gradient's contribution. Resuming this trajectory therefore needs the momentum buffer as well as
the parameter. Saving only the parameter is enough to reuse the forward function, but loses the
state needed for the next training step.

PyTorch, asked for the same two steps:

```
# Feed the same explicit gradient twice and inspect the
# stored momentum buffer.
p = torch.nn.Parameter(torch.tensor([2.0]))
opt = torch.optim.SGD([p], lr=0.1, momentum=0.9)
for _ in range(2):
    opt.zero_grad()
    p.grad = torch.tensor([0.25])
    opt.step()
    print(p.detach().tolist(), opt.state[p]["momentum_buffer"].tolist())
```

```
[1.975000023841858] [0.25]
[1.9275000095367432] [0.4749999940395355]
```

The values follow the same recurrence at the displayed precision; the PyTorch example uses
binary32 and the Lean example binary64. `MomentumSGD.State`'s docstring specifies PyTorch's SGD
convention with `dampening = 0` and `nesterov = false`. These two steps provide a numerical check
of that convention.

Nothing in the Lean version mutates the old state; it returns the new one. Adam and AdamW carry
more fields {Informal.citep adamw2019}[], and the picture is unchanged.

## Comparing Exact And Rounded Updates

`Optim.MomentumSGD.update` takes its state as an argument and is polymorphic in the scalar
type. The same definition can therefore be instantiated twice: once over $`\mathbb{R}`, where the
recurrence above holds exactly, and once over a rounded arithmetic that models what the machine
does. `momentumSGDContract` in
{src "NN/Proofs/RuntimeApprox/NF/Optimizers.lean"}[`NF/Optimizers.lean`] does exactly that, setting
both `updateExact` and `updateRuntime` to `Optim.MomentumSGD.update`, and
`approxTensor_momentumSGD_update` then bounds how far the two can drift apart after one step. An
optimizer with hidden state would need a model of that state before the same comparison could be
stated. Explicit arguments make that correspondence easier to express.

Returning a new state describes the update's meaning without prescribing a buffer copy. Lean 4
uses deterministic reference counting, so
compiled code can reuse the storage of a uniquely owned value instead of copying it
{Informal.citep immutablebeans2019}[]; TorchLean also updates native device buffers in place where
the workload calls for it. The functional rule says what the update means, and the implementation
decides how to realize it.

Parameters, gradients, and optimizer state all follow the scalar choice already made for the run.
Optimizer state does not introduce a second, hidden dtype policy.

## Gradient Accumulation

The update also needs a precise choice of gradient. In PyTorch a gradient is a mutable field on
the parameter, and `backward()` adds into it. Repeated backward calls without `zero_grad`
accumulate:

```
# Deliberately keep earlier gradients to show how
# accumulation changes the update.
p = torch.nn.Parameter(torch.tensor([2.0]))
opt = torch.optim.SGD([p], lr=0.1)
for i in range(3):
    (0.25 * p).sum().backward()   # no opt.zero_grad()
    opt.step()
    print(i, p.grad.tolist(), p.detach().tolist())
```

The gradient of this loss is the constant $`0.25`, so an update using only that gradient would move
the parameter by $`0.025`. The recorded PyTorch 2.13 run shows increasing steps:

```
0 [0.25] [1.975000023841858]
1 [0.5]  [1.9250000715255737]
2 [0.75] [1.850000023841858]
```

Adding `opt.zero_grad()` at the top of the loop gives `1.975`, `1.95`, `1.925`, using only the
current gradient at each step. Without that reset, each update includes earlier gradients too.
Accumulation is useful when intended, but changes this loop's update rule.

In TorchLean the gradient is an argument of the update, so both trajectories are writable and
neither is the default:

```lean (name := wfAccum)
-- Compare fixed per-step gradients with explicitly
-- accumulated copies of that gradient.
#eval do
  let mut p := wfTheta
  let mut s := Optim.SGD.init (α := Float) 0.1 wfTheta
  for _ in [0:3] do
    let st := Optim.SGD.update s p wfGrad
    p := st.parameters
    s := st.optimizerState
    IO.print s!"{p} "
  IO.println ""
  -- The un-zeroed loop, spelled out: after k backward
  -- calls the field holds k copies of the same gradient.
  let mut q := wfTheta
  let mut t := Optim.SGD.init (α := Float) 0.1 wfTheta
  for k in [0:3] do
    let acc : Tensor Float [1] :=
      Tensor.full [1] (0.25 * Nat.toFloat (k + 1))
    let st := Optim.SGD.update t q acc
    q := st.parameters
    t := st.optimizerState
    IO.print s!"{q} "
  IO.println ""
```
```leanOutput wfAccum (whitespace := lax)
[1.975000] [1.950000] [1.925000]
[1.975000] [1.925000] [1.850000]
```

The first trajectory subtracts 0.025 three times, matching PyTorch with `zero_grad`. The second
subtracts 0.025, then 0.05, then 0.075, matching the accumulating loop. The expression passed to
{name}`Optim.SGD.update` makes that choice visible. Both programs have the same tensor shapes;
the caller still has to decide whether the objective calls for a current gradient, a sum over
samples, or a mean with the appropriate sample count.

For accumulation across micro-batches, a TorchLean loop adds the gradient tensors and calls the
update after forming the intended batch gradient. The accumulation state is then another explicit
value carried by the loop.

## Definitional Equality And Tensor Storage

Setting $`\mu=0` in the momentum recurrence gives $`v_{t+1}=g_t`, so momentum SGD with zero momentum
should be plain SGD. Because both updates are functions of their arguments, that sentence is a claim
about two definitions rather than about two library behaviours, and we can put the two side by side:

```lean (name := wfTwoOptims)
-- Compare zero-momentum SGD with plain SGD on the same
-- parameter and gradient.
def wfMomStep : Tensor Float [1] :=
  (Optim.MomentumSGD.update
      (Optim.MomentumSGD.init (α := Float) 0.1 0.0 wfTheta)
      wfTheta wfGrad).parameters

def wfSgdStep : Tensor Float [1] :=
  (Optim.SGD.update
      (Optim.SGD.init (α := Float) 0.1 wfTheta)
      wfTheta wfGrad).parameters

#eval (wfMomStep, wfSgdStep, wfMomStep[0] == wfSgdStep[0])
```
```leanOutput wfTwoOptims
([1.975000], [1.975000], true)
```

They agree, and `==` on the entries confirms it is not a printing coincidence. Now try to promote
that observation to a proof the same way the affine example at the top of this chapter was proved:

```lean +error (name := wfTensorRfl)
-- A true executable comparison need not be an equality the
-- kernel can close by unfolding.
example : wfMomStep[0] = wfSgdStep[0] := by rfl
```
```leanOutput wfTensorRfl (whitespace := lax)
Tactic `rfl` failed: The left-hand side
  wfMomStep[0]
is not definitionally equal to the right-hand side
  wfSgdStep[0]

⊢ wfMomStep[0] = wfSgdStep[0]
```

Both sides are closed terms, so the equality is well posed. But the kernel cannot reduce this
particular packed `FloatArray` computation by `rfl`. `Tensor` uses a
scalar-dependent storage interface with lookup laws; the packed float primitives do not all reduce
by definitional unfolding. `#eval` uses their compiled implementation.

Notice that `wfParamCount wfInit = 17` did go through by `rfl` earlier. The difference is not that
one statement is harder than the other. It is that shapes are `List Nat` values the kernel can
compute with, while this packed payload requires reasoning through storage and tensor lemmas.

To prove the algebraic identity uniformly, we can work at a scalar type carrying the needed
laws. Over
$`\mathbb{R}` the same equation follows from two pointwise facts, `scaleSpec t 0 = 0` and
`addSpec 0 g = g`, which is how the optimizer file under
{src "NN/Proofs/RuntimeApprox/NF/Optimizers.lean"}[`NF/Optimizers.lean`] reasons about the very same
`Optim.MomentumSGD.update` definition. The real-valued result holds for every parameter tensor
and learning rate. It is a general algebraic statement, distinct from the finite floating-point
comparison above. {ref "runtime-approximation"}[Runtime Approximation] connects exact and rounded
updates with an error bound.

# Training And Evaluation Modes

Some layers denote two different functions. Dropout samples a mask during training and is the
identity during evaluation {Informal.citep dropout2014}[]. Batch normalization uses batch statistics
and updates running buffers during training, then reads the saved statistics during evaluation
{Informal.citep batchnorm2015}[]. In TorchLean, `Mode` is an
argument of `Layer.forward`, ahead of the state and the input.

```lean (name := wfMode)
-- train is a mode value; checking its type does not switch
-- a live module's mode.
#check Runtime.Autograd.Model.Layers.Mode.train
```
```leanOutput wfMode (whitespace := lax)
Runtime.Autograd.Model.Layers.Mode.train :
  Runtime.Autograd.Model.Layers.Mode
```

The running-statistics update is a function too, with the batch value and the old running value
both named:

```lean (name := wfRunning)
-- Blend the old running values with the new batch values
-- using momentum 0.1.
open Runtime.Autograd.Model.Layers (updateRunning) in
#eval updateRunning (α := Float) (s := [2])
  [1.0, 2.0] [3.0, 4.0] (Tensor.full [] 0.1)
```
```leanOutput wfRunning
[1.200000, 2.200000]
```

The first vector holds the old running values and the second holds the batch statistics.
The rank-zero tensor weights the new batch by one tenth. Optimizer momentum instead weights the
old buffer, so the shared name hides a difference in the equations.

That is $`0.9\cdot(1,2) + 0.1\cdot(3,4)`, the exponential moving average with momentum $`0.1`.
The corresponding PyTorch `running_mean` update is:

```
# Use identical rows so the batch mean is exactly the vector
# being blended into the buffer.
bn = nn.BatchNorm1d(2, momentum=0.1, affine=False)
with torch.no_grad():
    bn.running_mean.copy_(torch.tensor([1.0, 2.0]))
bn.train()
bn(torch.tensor([[3.0, 4.0], [3.0, 4.0]]))
print(bn.running_mean.tolist())
```

```
[1.2000000476837158, 2.200000047683716]
```

For dropout, let $`p` be the probability of dropping an entry. In evaluation the layer is the
identity; in training this construction uses a seeded mask and scales the surviving entries by
$`1/(1-p)`. For ideal independent
Bernoulli masks this scaling preserves each entry in expectation; that probabilistic statement
is distinct from inspecting one deterministic mask:

```lean (name := wfDropout)
-- Run one module in both modes to expose identity versus a
-- scaled seeded mask.
def wfDrop : nn.Sequential [4] [4] :=
  nn.build 2026 (nn.dropout 0.5)

#eval do
  let m ← nn.Module.instantiate wfDrop { device := .cpu }
  m.eval
  let a ← m.forward [1.0, 1.0, 1.0, 1.0]
  m.train
  let b ← m.forward [1.0, 1.0, 1.0, 1.0]
  IO.println s!"eval  = {a}"
  IO.println s!"train = {b}"
```
```leanOutput wfDropout (whitespace := lax)
eval  = [1.000000, 1.000000, 1.000000, 1.000000]
train = [2.000000, 0.000000, 2.000000, 0.000000]
```

Evaluation leaves the four ones unchanged. Training keeps two entries and doubles them. This
particular mask preserves the input sum, but another mask need not: the expectation statement
concerns repeated ideal draws at each coordinate.

PyTorch, same layer and same input:

```
# Observe evaluation identity and one training mask from
# PyTorch's generator.
torch.manual_seed(2026)
drop = nn.Dropout(0.5)
x = torch.ones(4)
drop.eval();  print(drop(x).tolist())
drop.train(); print(drop(x).tolist())
```

```
[1.0, 1.0, 1.0, 1.0]
[2.0, 0.0, 2.0, 2.0]
```

The convention agrees: identity in evaluation, and survivors scaled by $`2` in training, which is
the inverted-dropout scaling from the original paper's section 10. The masks differ because the two
masks came from different generators. Ours came from the seed the builder consumed, so
`nn.build 2026 (nn.dropout 0.5)` reproduces that seeded construction, while PyTorch's came
from the global generator's state at the moment of the call.

Making the mode an argument is also what lets an exported graph or a theorem say which function it
is about. A statement about a model containing dropout is ambiguous until the mode is fixed, and
here it cannot be left unfixed, because `forward` will not run without it.

# Runtime Effects And Pure Interfaces

Reading a dataset and launching a kernel are effects. The call that hands a model to a runtime
therefore has a different result type from the pure builder:

```lean (name := wfEffect)
-- Switching a live module to evaluation mode is an IO
-- action returning no data.
#check @nn.Module.eval
```
```leanOutput wfEffect (whitespace := lax)
@nn.Module.eval : {σ τ : Shape} →
  {α : Type} → [inst : Storage α] → [inst_1 : Context α] →
    {model : nn.Sequential σ τ} → nn.Module α model →
      IO Unit
```

Read the final ordinary argument as the live module whose mode will change. The implicit
`model` argument ties that module to its architecture, and `σ` and `τ` record its input and
output shapes. None of those arguments is an input tensor to predict on. After the returned action
runs, a subsequent forward call reads the selected mode. This explains why `m.eval` and
`#eval expression` have unrelated roles despite their similar spelling: one changes runtime state,
while the other asks Lean to execute an expression during the example.

The `IO Unit` result says that switching a live module into evaluation mode is an action with
no returned data. For a `Float` model with output shape `[1]`, `nn.Module.forward` returns
{lean}`IO (Tensor Float [1])`: the action produces a tensor.

The same separation works for a saved certificate. `IO.FS.readFile` reads its bytes; a pure parser
inspects the returned string; a pure checker inspects the parsed certificate. A theorem can then
state what acceptance implies without modelling the file system. Training likewise performs its
parameter updates, tape operations, and device activity in `IO`, while the specification layer
reasons about pure interpretations of the model.

# Storage Reuse In The Runtime

Lean 4's reference counting allows storage reuse when an immutable value has a unique owner: the
compiled code can update it in place {Informal.citep immutablebeans2019}[]. Above that, TorchLean
uses explicitly mutable runtime objects and foreign buffers wherever the workload calls for them;
{ref "backend-selection"}[Backend Selection] is where that machinery lives.

The source-level interface controls dependencies. Storage ownership and mutation belong to the
execution strategy. That is also what lets the CPU evaluator and LibTorch CUDA backend implement
one model: both receive the same explicit inputs, so changing device does not mean rewriting the
model around a different collection of hidden fields.

# Function Inputs, State, And Effects

When you meet a TorchLean definition, three questions place it quickly:

1. Which values determine the result?
2. Is it pure, a deterministic computation over a seed stream such as {lean}`nn.Builder`, or an `IO`
   action?
3. If state changes, where is the old state named and where is the new state named?

Try them on `Optim.MomentumSGD.update`. Its result depends on the optimizer state, the
parameters, and the gradients, and on nothing else. It is pure. The old state is the first
argument, and the new state is the `optimizerState` field of the returned `Optim.Step`. Now try them
on `autograd.model.grad`, which appears in {ref "torchlean_vs_pytorch"}[TorchLean and PyTorch]: it
takes a model, a loss, a state, an input, and a target, and it returns its answer in `IO`, because
it runs a tape on a real device.

# References

Reference counting for a purely functional language is {Informal.citet immutablebeans2019}[].
The two mode-dependent layers used as examples are {Informal.citet dropout2014}[] and
{Informal.citet batchnorm2015}[]; momentum is {Informal.citet polyak1964}[] and the weight-decay
variant is {Informal.citet adamw2019}[]. The floating-point discussion follows
{Informal.citet goldberg1991}[] and {Informal.citet flocq2011}[]. The framework we compare against
throughout is {Informal.citet pytorch2019}[].

- Lean 4 language reference:
  [Functions](https://lean-lang.org/doc/reference/latest/Terms/Functions/) and
  [Do notation](https://lean-lang.org/doc/reference/latest/Terms/do--Notation/).
- PyTorch reproducibility notes:
  [Randomness](https://pytorch.org/docs/stable/notes/randomness.html).
- PyTorch API pages:
  [`torch.optim.SGD`](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html),
  [`nn.Dropout`](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html), and
  [`nn.BatchNorm1d`](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html).
- TorchLean sources:
  {src "NN/Runtime/Autograd/Model/Layers/Core.lean"}[`Layer`],
  {src "NN/Runtime/Autograd/Model/Layers/Seq.lean"}[`Seq`], and
  {src "NN/Runtime/Optim/Optimizers.lean"}[`Optim`].

import VersoManual
import NN.API
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Execution Modes" =>
%%%
tag := "execution-modes"
file := "Choosing-How-A-Model-Runs"
%%%

The MLP trained in {ref "api_tour"}[A Tour Of The API] maps a two-component input to a
one-component output, with parameters $`\theta`:

$$`F_\theta:[2]\to[1]`

Its type stays `[2] → [1]` whether it runs eagerly on the CPU, through a typed graph, or with
LibTorch's CUDA operations. Holding the architecture, seed, and data fixed lets us compare these
execution
paths on the same training problem. The transcripts below record the binary's output on the
machine used for this guide; the embedded Lean comparisons also run when the page is built.

# Runtime Configuration

TorchLean configures arithmetic, execution, providers, and model mode separately. The runtime
validates their compatibility; not every combination is supported. Each setting controls a
different part of the computation:

:::table +header
*
  * Dial
  * Values
  * What it decides
*
  * arithmetic
  * `.native`, `.ieee`
  * which binary32 semantics every scalar operation obeys
*
  * execution
  * `.eager`, `.typedGraph`
  * whether operations run as they are met or are recorded once and replayed
*
  * device and providers
  * `.cpu`, `.cuda`, a full profile
  * which implementation of each operation is selected, and what evidence backs it
*
  * train or evaluation mode
  * `.train`, `.eval`
  * how mode-sensitive layers such as dropout behave
:::

Changing the device selects implementations of the model's operations. Changing train or evaluation
mode can change the operations' behavior, as dropout will show below. Recording both choices is
necessary to distinguish a provider comparison from a comparison of different forward computations.

`ExecutionMode` offers two ways to run the model:

```lean (name := emModes)
-- These constructors select runtime representations, not
-- optimization levels.
#print Runtime.ExecutionMode
```

```leanOutput emModes (whitespace := lax)
inductive Runtime.Autograd.Torch.ExecutionMode : Type
number of parameters: 0
constructors:
Runtime.Autograd.Torch.ExecutionMode.eager :
  Runtime.Autograd.Torch.ExecutionMode
Runtime.Autograd.Torch.ExecutionMode.typedGraph :
  Runtime.Autograd.Torch.ExecutionMode
```

The two constructors select eager or typed graph execution. Kernel planning and code generation
are separate mechanisms; neither is an additional value of this execution setting.

```lean (name := emArith)
-- The dispatcher vocabulary is broader than the trainer's
-- accepted arithmetic.
#print Runtime.Arithmetic
```

```leanOutput emArith (whitespace := lax)
inductive TorchLean.Runtime.Arithmetic : Type
number of parameters: 0
constructors:
TorchLean.Runtime.Arithmetic.native : Runtime.Arithmetic
TorchLean.Runtime.Arithmetic.ieee : Runtime.Arithmetic
TorchLean.Runtime.Arithmetic.complex : Runtime.Arithmetic
```

The supervised trainer accepts two of these three constructors. As the errors below show, the lower
dispatcher can execute complex binary32, but the supervised interface exchanges real data and
results. Complex training instead uses an explicit real loss with `autograd.complex.grad`, followed
by `nn.sgdStep` on complex state. The `complex_regression` command demonstrates this separate path;
it preserves both components in predictions and checkpoints.

The device vocabulary is the widest of the four:

```lean (name := emDevices)
-- A device constructor names a target without asserting
-- that a handler is linked.
#print Runtime.Device
```

```leanOutput emDevices (whitespace := lax)
inductive NN.Backend.Device : Type
number of parameters: 0
constructors:
NN.Backend.Device.cpu : NN.Backend.Device
NN.Backend.Device.cuda : NN.Backend.Device
NN.Backend.Device.rocm : NN.Backend.Device
NN.Backend.Device.metal : NN.Backend.Device
NN.Backend.Device.wasm : NN.Backend.Device
NN.Backend.Device.tpu : NN.Backend.Device
NN.Backend.Device.trainium : NN.Backend.Device
NN.Backend.Device.custom : NN.Backend.Device
NN.Backend.Device.external : NN.Backend.Device
```

Two of these nine devices currently have maintained runtime profiles. Device names also index
profiles, capsules, and reports, so the vocabulary includes targets such as `rocm` and `metal`
without claiming that an executable provider is available for them.

`withDevice` checks for a maintained profile before execution. `validateForExecution` additionally
checks whether the linked binary can provide the requested runtime:

```lean (name := emValidate)
-- Pure profile lookup and effectful execution validation
-- answer different questions.
open Runtime.Autograd.Torch in
#check @Config.withDevice

open Runtime.Autograd.Torch in
#check @Config.validateForExecution
```

```leanOutput emValidate (whitespace := lax)
Config.withDevice : Runtime.Autograd.Torch.Config →
  NN.Backend.Device →
  Except String Runtime.Autograd.Torch.Config
```

```leanOutput emValidate (whitespace := lax)
Config.validateForExecution :
  Runtime.Autograd.Torch.Config → IO Unit
```

The first returns `Except String Config`, so the failure is a value and can be inspected without
running anything. Checking the four named devices takes four lines:

```lean (name := emProfiles)
-- Check maintained defaults without requiring the named
-- devices to be present.
open Runtime.Autograd.Torch in
#eval do
  let devices : List Runtime.Device :=
    [.cpu, .cuda, .metal, .tpu]
  devices.forM fun d => do
    let name := d.cliName
    match Config.withDevice {} d with
    | .ok _ => IO.println s!"{name}: maintained profile"
    | .error msg => IO.println s!"{name}: {msg}"
```

```leanOutput emProfiles (whitespace := lax)
cpu: maintained profile
cuda: maintained profile
metal: device `metal` has no maintained runtime profile;
  provide a backend profile with executable capsules
tpu: device `tpu` has no maintained runtime profile;
  provide a backend profile with executable capsules
```

A custom backend needs both a profile describing admissible kernels and runtime handlers that
execute them. Supplying capsule metadata alone does not implement Metal. The maintained profiles
reject unsupported devices at configuration time.

The `Except String Config` result lets a caller handle a missing profile before entering `IO`.
Execution validation then checks the linked runtime and device availability. Individual operations
still have their own shape, layout, arithmetic, and gradient requirements.

# Runtime Flags

The runner lists its examples and runtime flags:

```terminal
# List the runnable examples and the shared entry-point
# options.
lake exe torchlean --help
```

The help lists the runnable examples, followed by their shared runtime flags:

```
Runtime flags:
  --choose                         ask for runtime choices before running
  --device auto|cpu|cuda|rocm|metal|wasm|tpu|trainium|custom|external
  --arithmetic native|ieee|complex
      arithmetic availability depends on the example; check its --help
  --execution eager|typed-graph
  --seed N
  --show-backend

Verification commands live under `lake exe verify -- list`.
Use `lake exe torchlean <example> --help` for command-specific flags.
```

The top-level parser knows three arithmetics; each
example states which ones it supports. For our example:

```terminal
# Inspect the controls accepted by this particular training
# example.
lake exe torchlean quickstart_mlp --help
```

```
TorchLean simple MLP quickstart

Usage:
  lake exe torchlean quickstart_mlp [--steps N] [--seed S]
    [--arithmetic native|ieee] [--execution eager|typed-graph] [--device cpu|cuda]
```

This example supports `native|ieee` and `cpu|cuda`. An opt-in interactive chooser can also select
the device:

```terminal +output
$ # Supply the CPU choice to the interactive device
$ # selector.
$ printf 'cpu\n' | lake exe torchlean --choose quickstart_mlp --steps 2 --seed 2026
TorchLean runtime chooser
Runtime device:
  1) CPU    portable default
  2) CUDA   GPU runtime, requires `lake -R -K cuda=true exe ...`
Select device [1]: == Quickstart: simple MLP training (seed=2026, steps=2) ==
```

The prompt is opt-in precisely so that scripts and CI never block waiting on input. The transcript
above pipes an answer in, which is also how it gets tested.

The menu and flags both request a configuration. The banner echoes that request; the backend
report below names the implementations selected for the model's operations.

# CPU Eager Execution

The baseline uses native binary32 arithmetic, eager CPU execution, and twenty training steps:

```terminal
# Run the short CPU baseline and print the selected backend
# capsules.
lake exe torchlean quickstart_mlp \
  --execution eager \
  --device cpu \
  --steps 20 \
  --seed 2026 \
  --show-backend
```

Leaving out `--show-backend` for a moment, the run prints:

```terminal +output
== Quickstart: simple MLP training (seed=2026, steps=20) ==
target(heldout)    = [0.200000]
untrained(heldout) = [-0.088261]
dataset size = 25
mean_loss(before training) = 0.495227
step 0: loss=0.250488
mean_loss(after training) = 0.401184
steps=20 arithmetic=native scalar=Float32 loss=0.495227 -> 0.401184
trained(heldout) = [0.365380]
```

Eager execution creates a session and records operations as the model runs. Every operation asks the
active profile for an admissible capsule, executes that capsule's provider, and appends a local VJP
rule when gradients are required. A capsule bundles a
provider, a trust level, a VJP owner, and a numerical policy, and `--show-backend` prints one the
first time each operation is selected. Here is the entry for addition, wrapped to fit the page (in
the terminal each of these is a single line):

```
  add: reference.add provider=reference trust=checked
    vjp=torchlean-tape reduction=n/a
    shape: shape safety for add; guarded at runtime by
      portable runtime shape checks
    layout: canonical-tensor layout compatibility for add;
      guarded at runtime by typed tensor layout
    value: add forward refines its TorchLean semantics;
      covered by test suite NN.Tests.Runtime.Floats.Suite
    vjp: add torchlean-tape VJP refines its TorchLean
      semantics; covered by test suite
      NN.Tests.Runtime.Floats.Suite
```

The four evidence lines distinguish shape, layout, forward-value, and VJP claims. Two cite runtime
guards and two cite a test suite. These labels identify the declared evidence for the selected
capsule; they do not assert a theorem. {ref "verification"}[Verification And Certificates]
develops the proof interfaces.

Our twenty-step CPU run selects seven capsules. Tabulating the first line of each entry:

:::table +header
*
  * Operation
  * Capsule
  * VJP owner
  * Reduction policy
*
  * `reshape`
  * `reference.reshape`
  * `torchlean-tape`
  * `n/a`
*
  * `permute`
  * `reference.permute`
  * `torchlean-tape`
  * `n/a`
*
  * `matmul`
  * `reference.matmul`
  * `torchlean-tape`
  * `fixed-left`
*
  * `broadcast`
  * `reference.broadcast`
  * `torchlean-tape`
  * `n/a`
*
  * `add`
  * `reference.add`
  * `torchlean-tape`
  * `n/a`
*
  * `relu`
  * `reference.relu`
  * `torchlean-tape`
  * `n/a`
*
  * `mse_loss`
  * `reference.mse_loss`
  * `torchlean-tape`
  * `fixed-left`
:::

Every provider is `reference`, every trust level is `checked`, and every VJP is owned by TorchLean's
own tape. The `fixed-left` policy on the two reducing operations specifies their summation order,
one of the conditions needed to reproduce floating-point results. The CUDA profile below changes
that policy.

With `--show-backend`, the full run prints three banners reading
`[TorchLean] backend capsules used:`,
and the first two have nothing under them. The banner is printed when an eager session is created
({srcDir "NN/Runtime/Autograd/Torch/Core"}[Core]`/Session.lean`), so it counts sessions rather than
operations. The first two sessions here evaluate `target` and `untrained`, and CPU prediction is
evaluated through a lowered graph without a tape, so those sessions select no capsules at all and
their banners are empty. The banner therefore identifies a session opening; the entries beneath
it identify capsule selections.

Eager mode is the natural starting point when operation structure depends on runtime values, when
you want to inspect the tape or the provider choices, or when you are using the maintained CUDA
runtime. It also accepts more dynamic frontend programs than the fixed typed graph recorder, which
is the subject of *Dynamic Control Flow And Typed Graphs* below.

There are three different loss observations in this short run. The initial `0.495227` is an
average over the 25 training examples. The `step 0` value, `0.250488`, belongs to the example used
for that update. The final `0.401184` is another evaluation over the dataset after 20 updates.
Comparing the first and last values is meaningful as a change in dataset loss; comparing either
one directly with the single-example step loss mixes two different measurements. Twenty updates
also need not visit every example equally often.

The held-out prediction is a fourth observation. Its target is `0.2`, and its trained value here is
`0.365380`; a lower training loss does not make that particular prediction exact. Keeping it
separate makes the log useful even when individual update losses fluctuate.

The seven capsule kinds describe the operations needed by the model and loss. They are not a
count of kernel launches or arithmetic instructions. Repeated operations may choose the same
capsule, and a report with no new entries during prediction does not mean prediction performed no
work. It means that report contains no additional selections to display for that session.

# CPU Typed Graph Execution

One flag changes:

```terminal
# Keep the short training setup while selecting typed graph
# execution.
lake exe torchlean quickstart_mlp \
  --execution typed-graph \
  --device cpu \
  --steps 20 \
  --seed 2026
```

```terminal +output
== Quickstart: simple MLP training (seed=2026, steps=20) ==
target(heldout)    = [0.200000]
untrained(heldout) = [-0.088261]
dataset size = 25
mean_loss(before training) = 0.495227
step 0: loss=0.250488
mean_loss(after training) = 0.401184
steps=20 arithmetic=native scalar=Float32 loss=0.495227 -> 0.401184
trained(heldout) = [0.365380]
```

The displayed output matches the eager baseline character for character. A longer run compares
the paths over two hundred updates, during which the sampled losses span several orders of
magnitude:

```terminal +output
$ # Compare complete printed logs from fresh runs with the
$ # same seed and update count.
$ lake exe torchlean quickstart_mlp --execution eager --steps 200 --seed 2026 > eager.txt
$ lake exe torchlean quickstart_mlp --execution typed-graph --steps 200 --seed 2026 > graph.txt
$ diff eager.txt graph.txt && echo identical
identical
```

For reference, that curve is not monotone:

```terminal +output
step 0: loss=0.250488
step 25: loss=0.586318
step 50: loss=0.847444
step 75: loss=0.003933
step 100: loss=0.023397
step 125: loss=0.069799
step 150: loss=0.000061
step 175: loss=0.029500
mean_loss(after training) = 0.002402
```

These are losses on successive samples, not repeated measurements of the full dataset loss.
The rise around step 50 therefore does not by itself diagnose optimizer overshoot. Reproducing
this transcript checks several intermediate observations as well as the endpoint.

For this trainer path, typed graph execution records the fixed scalar-loss program once as a typed
SSA graph, including its forward, JVP, and VJP behavior, then reuses that graph with the current
parameters and data. The graph stores its output as a typed reference to an input or a recorded
node, rather than assuming that the last node created is the result, and the shape-indexed builder
rejects ill-shaped connections and ill-shaped output references.

Graph reuse has a narrower scope than compilation or derivative verification:

- It does not prove the derivative rules correct. The graph stores executable derivative rules;
  selecting `.typedGraph` runs them. A derivative theorem needs the corresponding proof-carrying
  nodes from the autograd proof layer. TorchLean does prove that lowering `GraphData` to a runtime
  tape preserves the stored backpropagation program, and that implementation theorem is a different
  statement from mathematical derivative correctness.
- It does not optimize, fuse, schedule, or generate native code. It records.
- It does not consume an `AcceptedGraphKernelPlan` and it is not CUDA Graph capture.

The current typed graph trainer is CPU only, and a CUDA request fails explicitly rather than falling
back.

Reusing a typed graph means reusing the program's structure while supplying the current parameter
values. It does not mean caching the prediction from the first training step. In the maintained
graph trainer, the stored graph is lowered to a fresh tape for a call, so the operation closures
capture the values for that call. This is a concrete form of structural reuse with a remaining
execution cost. The matching loss sequence below establishes agreement for this workload; it
does not measure allocations, graph-lowering overhead, or a speedup over eager execution.

# Eager And Typed Graph Training Comparison

The comparison can also run inside the guide, so the page build checks it directly. The setup
fixes the quickstart model, target, dataset, optimizer, and seed:

```lean (name := emSetup)
-- Hold model, target, data, and seed fixed while exposing
-- the runtime choices.
/-- The `2 -> 8 -> 1` regression MLP from the quickstart. -/
def emModel : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![nn.linear 2 8, nn.relu, nn.linear 8 1]

/-- Target `0.8 relu(x₁+x₂) - 0.4 relu(x₂-x₁) + 0.2`. -/
def emTarget (x : Tensor Float [2]) : Tensor Float [1] :=
  let relu (v : Float) := if v < 0.0 then 0.0 else v
  let rise := 0.8 * relu (x[0] + x[1])
  let fall := 0.4 * relu (x[1] - x[0])
  [rise - fall + 0.2]

def emInputs : Tensor Float [25, 2] :=
  Data.Synthetic.squareGrid (-1.0) 1.0 5

def emData : Trainer.Dataset [2] [1] :=
  Data.fromTensors emInputs
    (Tensor.mapLeading [25] emTarget emInputs)

/-- A grid point that is not in the training set. -/
def emPoint : Tensor Float [2] := [0.25, -0.75]

/-- One trainer per choice of the three run dials. -/
def emRun (arithmetic : Runtime.Arithmetic)
    (execution : Runtime.ExecutionMode)
    (device : Runtime.Device) : Trainer [2] [1] :=
  Trainer.new emModel
    { objective := .mse
      optimizer := optim.adam { learningRate := 0.03 }
      arithmetic := arithmetic
      execution := execution
      device := device
      seed := 2026 }
```

`emRun` exposes arithmetic, execution, and device as arguments while holding the training problem
fixed. The following block changes only execution mode, then compares both the printed summaries
and the held-out predictions:

```lean (name := emEagerVsGraph)
-- Train two fresh models so only the execution
-- representation changes.
#eval do
  let opts : Trainer.TrainOptions :=
    { steps := 200, logEvery := 0 }
  let run := emRun .native
  let eager ← (run .eager .cpu).train emData opts
  let graph ← (run .typedGraph .cpu).train emData opts
  IO.println s!"eager = {eager.summary}"
  IO.println s!"graph = {graph.summary}"
  let a ← eager.predict emPoint
  let b ← graph.predict emPoint
  IO.println s!"eager prediction = {a}"
  IO.println s!"graph prediction = {b}"
  IO.println s!"predictions equal = {a == b}"
```

```leanOutput emEagerVsGraph (whitespace := lax)
dataset size = 25
mean_loss(before training) = 0.495227
mean_loss(after training) = 0.002402
dataset size = 25
mean_loss(before training) = 0.495227
mean_loss(after training) = 0.002402
eager = steps=200 arithmetic=native scalar=Float32
  loss=0.495227 -> 0.002402
graph = steps=200 arithmetic=native scalar=Float32
  loss=0.495227 -> 0.002402
eager prediction = [0.228325]
graph prediction = [0.228325]
predictions equal = true
```

The final line compares prediction values with `Float` equality rather than comparing their
six-decimal formatting. It can detect differences hidden by the displayed loss. Numerical equality
still differs from bit equality, notably for signed zero, and this check covers one prediction
rather than all parameter buffers or possible inputs.

The comparison opens two fresh training runs. Both start from seed `2026`, receive the same data
and optimizer configuration, and perform 200 updates. That setup matters: switching the execution
mode after one model had already trained would compare different parameter states as well as
different representations. Equality of the reported summaries is evidence about these runs at
the displayed precision, not a theorem covering all models and all intermediate tensors.

The held-out target can be checked without running either trainer. At `[0.25, -0.75]`, the sum of
the coordinates is `-0.5` and their difference `x₁ - x₀` is `-1`. Both ReLU terms in the target
function therefore vanish, leaving its constant `0.2`. The common prediction `0.228325` is about
`0.028325` above that target. The small final dataset loss and this remaining prediction error
answer different questions, even though both came from the same trained parameters.

# CUDA Execution Through LibTorch

CUDA support is a build-time choice. Set `TORCHLEAN_LIBTORCH_HOME` to a CUDA-enabled SDK as
described in the installation guide, then select it with `-K cuda=true`:

```terminal
# Select the CUDA build profile before requesting a CUDA
# training session.
lake -R -K cuda=true exe torchlean quickstart_mlp \
  --execution eager \
  --device cuda \
  --steps 20 \
  --seed 2026 \
  --show-backend
```

The report now names a different provider for every operation. Comparing it against the CPU table:

:::table +header
*
  * Operation
  * CPU capsule
  * CUDA capsule
  * CUDA reduction policy
*
  * `reshape`
  * `reference.reshape`
  * `libtorch.reshape`
  * `n/a`
*
  * `permute`
  * `reference.permute`
  * `libtorch.permute`
  * `n/a`
*
  * `matmul`
  * `reference.matmul`
  * `libtorch.matmul`
  * `implementation-defined`
*
  * `broadcast`
  * `reference.broadcast`
  * `libtorch.broadcast`
  * `n/a`
*
  * `add`
  * `reference.add`
  * `libtorch.add`
  * `n/a`
*
  * `relu`
  * `reference.relu`
  * `libtorch.relu`
  * `n/a`
*
  * `mse_loss`
  * `reference.mse_loss`
  * `libtorch.mse_loss`
  * `implementation-defined`
:::

The provider label changes from `reference` to `libtorch`, the VJP label changes from
`torchlean-tape` to `backend-vjp`, the test suite named in the evidence lines changes from
`NN.Tests.Runtime.Floats.Suite` to `NN.Tests.Runtime.Cuda.Suite`, and the guard for shapes changes
from portable runtime checks to CUDA FFI size and rank checks at the Lean and native boundary. Trust
stays `checked` throughout. `backend-vjp` means that an ATen operation evaluates the local
gradient; TorchLean still records and traverses the tape. LibTorch autograd recording is disabled.
For numerical comparison, the last column is especially relevant:
the two
reducing operations move from `fixed-left` to `implementation-defined`.

To compare the two devices, hold the model, dataset, seed, optimizer, and step count fixed.
A provider with `reduction=implementation-defined` may use a different summation order, and
floating point addition is not associative {Informal.citep goldberg1991}[]. Compare the artifact
relevant to the experiment, such as predictions or parameter buffers, and record its dtype,
provider, and comparison tolerance. Agreement in printed decimal output does not establish equal
parameter bits or equal behavior on other inputs.

The module and CUDA buffer both use binary32 values, but upload is not conversion-free. The current
bridge widens host `Float32` elements to `Float` staging values before packing them into a CUDA
float32 buffer. This preserves finite binary32 values but adds transfer work. A capsule identifies
the trusted native provider; it does not prove the kernel, compiler, driver, or device correct.

A build without CUDA support rejects the request. The default build still compiles and tests
everything else without a GPU.

A common seed fixes initialization and the example sequence; it does not fix the order in which
a backend adds partial sums. Updates can carry numerical differences into later steps, and a ReLU
can change its active branch near zero. To investigate a discrepancy, compare the inputs and
outputs of the first differing operation before attributing the final loss difference to the
optimizer.

# Executable Binary32 Arithmetic

```terminal
# Use executable binary32 arithmetic for the same short CPU
# workload.
lake exe torchlean quickstart_mlp \
  --arithmetic ieee \
  --execution eager \
  --device cpu \
  --steps 2 \
  --seed 2026
```

The `.ieee` path evaluates addition and multiplication with FloatLib binary32. Comparing the two
implementations can expose errors
in rounding or exceptional-value handling that a comparison of decimal output alone may miss.
Formal floating-point models make these rules explicit
{Informal.citep goldberg1991}[]. Flocq made such a model reusable inside a proof assistant
{Informal.citep flocq2011}[], and the same group's later work connects such a model to the
arithmetic a compiler actually emits {Informal.citep boldo2015}[]. FloatLib supplies the executable
format and arithmetic used here. For higher precision, select a valid FloatLib binary format in
typed CPU tensors and graphs; the trainer flag shown above remains fixed to binary32.

The following embedded comparison runs two hundred steps with each arithmetic choice, holding
the seed and dataset fixed. It also checks prediction equality:

```lean (name := emNativeVsIeee)
-- Keep device and execution fixed to compare the two
-- binary32 implementations.
#eval do
  let opts : Trainer.TrainOptions :=
    { steps := 200, logEvery := 0 }
  let native ← (emRun .native .eager .cpu).train emData opts
  let ieee ← (emRun .ieee .eager .cpu).train emData opts
  IO.println s!"native = {native.summary}"
  IO.println s!"ieee   = {ieee.summary}"
  let a ← native.predict emPoint
  let b ← ieee.predict emPoint
  IO.println s!"native prediction = {a}"
  IO.println s!"ieee   prediction = {b}"
  IO.println s!"predictions equal = {a == b}"
```

```leanOutput emNativeVsIeee (whitespace := lax)
dataset size = 25
mean_loss(before training) = 0.495227
mean_loss(after training) = 0.002402
dataset size = 25
mean_loss(before training) = 0.495227
mean_loss(after training) = 0.002402
native = steps=200 arithmetic=native scalar=Float32
  loss=0.495227 -> 0.002402
ieee   = steps=200 arithmetic=ieee scalar=ExecFloat.Binary 8 23
  loss=0.495227 -> 0.002402
native prediction = [0.228325]
ieee   prediction = [0.228325]
predictions equal = true
```

Agreement on one model, one seed, and one dataset is a test result. A differential
test between a hardware instruction and a bit-level implementation can expose rounding, tie,
or sign errors exercised by its inputs and assertions. It says nothing about inputs we did not
try. The theorem-side story lives in {ref "floats"}[Floating Point Semantics], where the
proof-oriented `FP32` model and exact `Real` appear in statements rather than in `IO`, and where the
finite and no-overflow bridge from `FP32` to FloatLib binary32 is stated. `FP32` rounds reals at
binary32
precision with gradual underflow, and it does not model overflow, NaN, infinity, or signed zero,
which is why the bridge has hypotheses.

The bit-level implementation can cost more than native arithmetic. Measure that cost on the
workload and hardware of interest, separating process startup from repeated execution. Record
commands, software versions, thread settings, and the actual provider alongside timings so that
others can reproduce the comparison.

Keep the training horizon fixed when comparing arithmetic choices. A two-step demonstration and
a two-hundred-step run answer different questions about the same model.

There is also an interface boundary in these examples: the trainer accepts and reports public
`Float` values while its selected implementation works with a binary32 scalar representation.
An explicit `Float` autograd program elsewhere in the guide is a separate arithmetic choice.
Matching the public input type alone does not establish matching internal arithmetic.

# Runtime Configuration In Lean

`Trainer.RunConfig` exposes the runtime choices as typed fields:

```lean (name := emRunCfg)
-- Record updates isolate runtime changes while retaining
-- the optimizer settings.
def emEagerCpu : Trainer.RunConfig :=
  { arithmetic := .native
    execution := .eager
    device := .cpu
    optimizer := optim.adam { learningRate := 0.03 } }

def emTypedGraphCpu : Trainer.RunConfig :=
  { emEagerCpu with execution := .typedGraph }

def emEagerCuda : Trainer.RunConfig :=
  { emEagerCpu with device := .cuda }

/-- Bind a run configuration to an objective and a seed. -/
def emTrainerFromRun (run : Trainer.RunConfig) :
    Trainer [2] [1] :=
  Trainer.new emModel
    (run.forObjective .mse 2026)

#eval do
  let runs := [emEagerCpu, emTypedGraphCpu, emEagerCuda]
  runs.forM fun (run : Trainer.RunConfig) => do
    let settings := run.executionSettings
    let mode := repr settings.execution
    IO.println s!"{settings.deviceName} {mode}"
```

```leanOutput emRunCfg (whitespace := lax)
cpu Runtime.Autograd.Torch.ExecutionMode.eager
cpu Runtime.Autograd.Torch.ExecutionMode.typedGraph
cuda Runtime.Autograd.Torch.ExecutionMode.eager
```

The record update syntax makes `emTypedGraphCpu` differ from `emEagerCpu` in exactly one field.
`RunConfig` carries the runtime choices and optimizer; `forObjective` adds the objective and seed
to produce a full `Trainer.Config`. This permits reuse of one runtime configuration across
several objectives. `executionSettings` projects out the
`Runtime.Config` that the session layer actually validates, which is the value that
`validateForExecution` inspects.

`RunConfig.withRuntime` goes the other way: it takes a `Runtime.Config` and overwrites exactly the
execution, device, backend-profile and reporting fields of a configuration, leaving the optimizer
and the rest alone. A sweep therefore declares the model once, keeps one base `RunConfig`, and
derives each cell of the sweep from it.

Constructing `emEagerCuda` does not open CUDA. These labels print only the device and execution
fields of each record; they omit arithmetic, optimizer settings, and provider overrides. Save those
fields too when comparing runs.

# Graph Lowering And Direct Execution

The trainer retains its graph for repeated execution but does not expose that graph through the
training result. When a program needs the graph itself, lower it directly:

```lean (name := emGraphTypes)
-- Graph leaves contain state followed by input; the wrapper
-- separates them for callers.
#check @nn.lowerToTypedGraph
#check @nn.TypedGraphModel.forward
```

```leanOutput emGraphTypes (whitespace := lax)
@Runtime.Autograd.Model.Layers.Seq.lowerToTypedGraph :
  {σ τ : Shape} → (model : Runtime.Autograd.Model.Layers.Seq σ τ) →
  optParam Runtime.Autograd.Model.Layers.Mode
    Runtime.Autograd.Model.Layers.Mode.eval →
  {α : Type} → [inst : Storage α] → [Context α] →
  IO (Runtime.Autograd.Torch.TypedGraph α
    (model.stateShapes ++ [σ]) τ)
```

```leanOutput emGraphTypes (whitespace := lax)
@nn.TypedGraphModel.forward : {σ τ : Shape} → {α : Type} →
  [inst : Storage α] → {stateShapes : List Shape} →
  nn.TypedGraphModel stateShapes σ τ α →
  nn.State α stateShapes → Tensor α σ → Tensor α τ
```

`lowerToTypedGraph` returns a
`TypedGraph α (model.stateShapes ++ [σ]) τ`, whose index is a flat list of every graph leaf:
parameters first, then the model input. `nn.TypedGraphModel` is the model-facing view of exactly
that type, and its `forward` splits the leaves back into `nn.State α stateShapes` and
`Tensor α σ`. Nothing is converted; the second signature is a way of talking about the first that
keeps the parameters and the input apart.

`nn.TypedGraphModel` is an `abbrev`, so it is transparent to dot
notation. Writing `g.forward state input` resolves to the raw `TypedGraph.forward`, which wants one
`TensorPack` over the concatenated leaves and will reject the two arguments. Call the model-facing
wrappers by name, as below.

```lean (name := emGraph)
-- Zero parameter tangents isolate derivatives with respect
-- to the input coordinates.
#eval do
  let built := nn.build 2026 emModel
  let g ← nn.lowerToTypedGraph built (α := Float)
  let state := nn.initialState built
  let shapes := nn.stateShapes built
  let y := nn.TypedGraphModel.forward g state emPoint
  IO.println s!"forward = {y}"
  let (_, inputGradient) :=
    nn.TypedGraphModel.vjp g state emPoint [1.0]
  IO.println s!"input gradient = {inputGradient}"
  IO.println s!"parameter shapes = {shapes}"
  let zero : nn.State Float shapes := nn.State.zeros
  let jvp := nn.TypedGraphModel.jvp g state zero emPoint
  let d0 := jvp [1.0, 0.0]
  let d1 := jvp [0.0, 1.0]
  IO.println s!"jvp along e₀ = {d0}, jvp along e₁ = {d1}"
```

```leanOutput emGraph (whitespace := lax)
forward = [-0.088261]
input gradient = [0.263189, 0.205411]
parameter shapes = [[8, 2], [8], [1, 8], [1]]
jvp along e₀ = [0.263189], jvp along e₁ = [0.205411]
```

`forward = [-0.088261]` is the same number the runner printed as
`untrained(heldout)` in every experiment above, which is a small consistency check that the lowered
graph and the trainer agree on the model. The parameter shapes `[[8, 2], [8], [1, 8], [1]]` are the
two weight matrices and two bias vectors of a `2 → 8 → 1` MLP, in declaration order.

The reverse mode pass, seeded with the output
cotangent `[1.0]`, returns the input gradient `[0.263189, 0.205411]`. The two forward mode passes,
seeded with the basis tangents `[1.0, 0.0]` and `[0.0, 1.0]` and a zero parameter tangent, return
`[0.263189]` and `[0.205411]`. For a function with a one-dimensional output these must coincide,
because both are computing the same $`1 \times 2` Jacobian, one row at a time from the left and one
column at a time from the right. They coincide to the last printed digit. Comparing forward and
reverse directional derivatives checks their agreement without requiring a separate symbolic
derivative {Informal.citep baydin2018}[]. Retaining the graph lets all three calls use the same
recorded operations.

Two lifetimes to keep straight. `nn.lowerToTypedGraph` hands you a graph that lives as long as the
reference does. The imperative `Session` API instead records one graph per recording phase, and
`resetTape` starts a fresh one. When the graph itself must survive across calls, lower it or use the
high level trainer.

The direct graph calculation isolates derivatives with respect to the input. Its parameter
tangents are zero, so the two unit input directions ask how the prediction changes when one input
coordinate moves and the parameters stay fixed. The resulting JVPs, `0.263189` and `0.205411`,
match the two coordinates of the input VJP seeded by `1`. This scalar-output case makes the
relationship especially visible: each basis direction selects one entry of the same gradient.
The example explicitly uses `Float`; it should not be read as another execution of the trainer's
binary32 comparison, despite the similar printed initial prediction.

# Device And Provider Profiles

Ordinary code selects a device and lets TorchLean resolve the maintained profile for it. Advanced
code can select a complete profile with `withBackendProfile`, choosing provider preference,
assurance policy, VJP ownership, target availability, and capsule modules together. Calling
`withDevice` afterwards clears that override, which is deliberate: a device name and a hand-built
profile are two ways to answer the same question, and the last answer wins rather than silently
merging.

{ref "backend-selection"}[Backend Selection] follows an operation request through provider
selection, contract checks, and binding to a runtime handler.

# Train And Evaluation Mode

The high-level `Trainer.Session` selects mode from the operation you call. Lower-level module APIs
also provide explicit mode setters:

:::table +header
*
  * Call
  * Mode
*
  * `session.step sample`
  * training
*
  * `session.step batch (batch := true)`
  * training
*
  * `session.predict input`
  * evaluation
*
  * `session.loss sample`
  * evaluation
*
  * `session.eval data`
  * evaluation
:::

`trainer.train` uses the training path for updates and the evaluation path for the losses it
reports, for the summary prediction, and for later calls to `Trainer.Result.predict`. Mode is
independent of device and execution choice: a CUDA session switches mode without changing the model
architecture or the provider profile.

`step` returns `Unit` by default, so an update does not read the loss back to the host. Add
`(loss := true)` when the caller needs the loss from that update. The `batch` flag selects a
nonempty array of samples; it does not change training mode.

Dropout {Informal.citep dropout2014}[] makes the effect of mode visible. During training it samples
a mask; during evaluation it applies its deterministic inference behavior. The next block inserts
dropout between the MLP's hidden activation and output layer, then evaluates the same dataset
through both session paths:

```lean (name := emDropout)
-- Evaluate before the batch update so the mode comparison
-- starts at the same state.
/-- The same MLP with one dropout layer. -/
def emDropoutModel (p : Float) :
    nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![nn.linear 2 8, nn.relu, nn.dropout p,
    nn.linear 8 1]

def emDropoutTrainer (p : Float) : Trainer [2] [1] :=
  Trainer.new (emDropoutModel p)
    { objective := .mse
      optimizer := optim.adam { learningRate := 0.03 }
      seed := 2026 }

#eval do
  let batch := (← emData.materialize (α := Float)).toArray
  for p in [0.0, 0.5] do
    let session ← (emDropoutTrainer p).open
    let evaluation ← session.eval emData
    let training ← session.step batch
      (batch := true) (loss := true)
    IO.println s!"p = {p}"
    IO.println s!"  evaluation mode {evaluation}"
    IO.println s!"  training mode   {training}"
    let gap := (training - evaluation) * 1e9
    IO.println s!"  gap * 1e9 = {gap}"
```

```leanOutput emDropout (whitespace := lax)
p = 0.000000
  evaluation mode 0.495227
  training mode   0.495227
  gap * 1e9 = -29.355288
p = 0.500000
  evaluation mode 0.761530
  training mode   0.307745
  gap * 1e9 = -453785276.710987
```

Read the gaps, not the printed losses. At `p = 0.0` the layer is the identity in both modes and the
two numbers print the same to six decimals, yet the gap is `-2.9e-8` rather than zero.
`eval` converts per-sample losses to `Float` and averages them there; `step` with `(batch := true)`
uses the runtime scalar for its batch reduction. This comparison changes both reduction structure
and the precision of the final average, so it does not isolate summation order alone.

At `p = 0.5` the much larger gap illustrates the effect of training-mode dropout in this example.
The `p = 0` result is a useful control, not a universal bound on floating point discrepancies.

Repeating the `p = 0.5` measurement in freshly opened sessions gives the same result for this seed.
The builder derives the dropout layer's seed from the model build seed. This demonstrates seeded
reproducibility; it does not establish independence of masks across calls. `Session.save` saves
model state, not a complete optimizer, loader, and random-stream snapshot for exact training resume.

Compare PyTorch, where the same distinction exists but is carried by a mutable attribute
{Informal.citep pytorch2019}[]:

```
>>> # Compare repeated calls before and after disabling
>>> # training-mode dropout.
>>> torch.manual_seed(2026)
>>> x = torch.tensor([0.25, -0.75])
>>> net = torch.nn.Sequential(
...     torch.nn.Linear(2, 8), torch.nn.ReLU(),
...     torch.nn.Dropout(0.5), torch.nn.Linear(8, 1))
>>> net.training
True
>>> [net(x).item() for _ in range(3)]
[-0.8670920133590698, 0.04916824400424957, -0.2292545586824417]
>>> net.eval()
>>> [net(x).item() for _ in range(3)]
[-0.1853325217962265, -0.1853325217962265, -0.1853325217962265]
```

Three calls to the same object on the same input give three different answers, then the same call
becomes deterministic after `net.eval()`. In PyTorch, mode is a flag on the module and defaults to
training. In TorchLean,
`Trainer.Session` chooses the mode for these operations, while lower-level APIs require the caller
to select it. The example derives training-mode randomness from the build seed.

The mode also determines which random state the reverse rule must retain. Consider the operation

$$`y=\operatorname{Dropout}_{p}(x)`.

During training a mask is realized and retained for the backward rule. Re-running the backward pass
with a freshly sampled mask would not differentiate the forward value that was actually computed; it
would differentiate a different function with the same input and output shapes. The reverse rule
must therefore retain the realized mask or enough state to reproduce that exact draw.

Each dropout measurement starts a fresh session and evaluates before performing the batch update.
Within that session, the evaluation and the training loss therefore refer to the same initial
parameter state. The training loss is produced as part of the update; it is not a second evaluation
of the updated model. Evaluating after the update would mix the effect of mode with the effect of
changed parameters.

# Dynamic Control Flow And Typed Graphs

A fixed typed graph needs its operation structure and its shapes to be known when it is recorded. A
program that reads a token value and then decides which operations to perform cannot be represented
as one fixed graph, and TorchLean's recorder does not try.

Tracing and export illustrate why this restriction matters. Consider a function whose branch
depends on the input values:

```
# A value-dependent branch must remain observable for later
# inputs.
def f(x):
    if x.sum() > 0:
        return x * 2
    else:
        return x - 100
```

Tracing on a positive-sum input records only the first branch. PyTorch warns that the trace may
not generalize:

```
>>> # Trace the positive branch, then probe a negative input
>>> # with the same shape.
>>> traced = torch.jit.trace(f, torch.tensor([1.0, 1.0]))
TracerWarning: Converting a tensor to a Python boolean might cause the
trace to be incorrect. ... the trace might not generalize to other inputs!
>>> f(torch.tensor([-1.0, -1.0])).tolist()
[-101.0, -101.0]
>>> traced(torch.tensor([-1.0, -1.0])).tolist()
[-2.0, -2.0]
>>> traced.graph
graph(%x : Float(2, strides=[1], requires_grad=0, device=cpu)):
  %5 : Long(requires_grad=0, device=cpu) = prim::Constant[value={2}]()
  %6 : Float(2, strides=[1], requires_grad=0, device=cpu) = aten::mul(%x, %5)
  return (%6)
```

The recorded graph contains one `aten::mul` and no comparison. It therefore returns `[-2.0, -2.0]`
on the negative input, where the original function takes the unrecorded `else` branch and returns
`[-101.0, -101.0]`. Matching the input shape did not preserve the input-dependent choice.

The modern export path refuses instead:

```
>>> # Ask export to retain a branch whose decision depends
>>> # on a tensor value.
>>> class F(torch.nn.Module):
...     def forward(self, x):
...         return f(x)
>>> torch.export.export(F(), (torch.tensor([1.0, 1.0]),))
GuardOnDataDependentSymNode: Could not guard on data-dependent expression
Eq(u0, 1) (unhinted: Eq(u0, 1)).
```

The partial graph ends at `torch.ops.aten.item.default`: evaluating the branch condition requires
a tensor value that is not determined by the input shape.

TorchLean's typed graph recorder supports a fixed operation structure and rejects unsupported
recording operations. Its shape indices check tensor dimensions and references; they do not prove
that an arbitrary frontend program preserves data-dependent branching. A fixed graph must represent
the intended choice explicitly rather than specialize it to one sample's values.

Data-dependent structure can remain in an eager frontend that supports it. Other options are to
express the choice as a supported tensor operation, such as a mask or select, or record separate
static branches and choose between them at runtime. In each case, the implementation must preserve
the choice on subsequent inputs.

Agreement on several positive-sum inputs would not expose this trace's defect: all of them follow
the branch that recording retained. A useful probe crosses the decision boundary while keeping
the tensor shape fixed. The negative input above does exactly that. More generally, validating a
captured program requires inputs that distinguish its possible behaviors, not just repeated
examples from the region in which it was captured.

A branch on a fixed model configuration can be resolved while constructing a graph. A branch on
a future tensor value needs a representation of that decision in the executable program. The
polymorphic TorchLean function interface uses opaque tensor references even with an eager handler;
it does not expose their elements as ordinary Lean booleans. Supporting that style of dynamic
control flow would require an appropriate frontend and runtime interface, beyond choosing eager
execution for the current model API.

# Unsupported Runtime Configurations

The following requests fail at different configuration boundaries. The
messages are wrapped here to fit the page; each is a single line in the terminal.

An unimplemented device:

```terminal +output
$ # Request a recognized device that has no maintained
$ # runtime profile.
$ lake exe torchlean quickstart_mlp --device metal --steps 1
error: quickstart_mlp: device `metal` has no
maintained runtime profile; use a programmatic backend profile
```

An execution mode that has no profile for the requested device:

```terminal +output
$ # Exercise validation of the graph execution and device
$ # combination.
$ lake exe torchlean quickstart_mlp --execution typed-graph --device cuda --steps 1
error: typed graph execution currently supports
device `cpu`; requested `cuda`
```

An arithmetic the lower dispatcher can execute but the supervised trainer cannot use:

```terminal +output
$ # Check that this supervised trainer rejects unsupported
$ # complex arithmetic.
$ lake exe torchlean quickstart_mlp --arithmetic complex --steps 1
error: quickstart_mlp: TorchLean.Trainer: supervised
training supports native or IEEE arithmetic; complex arithmetic
requires an explicit complex-valued training API
```

A CUDA build with no visible device:

```terminal +output
$ # Hide visible GPUs while keeping the native CUDA build to
$ # isolate runtime availability.
$ CUDA_VISIBLE_DEVICES="" lake -R -K cuda=true exe torchlean quickstart_mlp --device cuda
error: torch eager session: CUDA was requested and
this is a CUDA build, but no usable CUDA device is visible
```

That last message distinguishes the two failure modes it could be reporting. A build without
LibTorch says so and tells you to rebuild with `-K cuda=true`; a real CUDA build with no
device says that instead. Both refuse. Notice also where the second message came from: the typed
graph rejection happened after the banner line had already printed, because it is raised when the
session opens rather than when the flags are parsed. Session validation checks programmatic
configurations as well as command-line requests.

The rejection is a value in Lean, so this page can exhibit it without a terminal:

```lean (name := emReject)
-- Catch the session-opening failure without executing an
-- unsupported CUDA graph.
#eval do
  try
    let _ ← (emRun .native .typedGraph .cuda).train emData
      { steps := 1, logEvery := 0 }
    IO.println "the run was accepted"
  catch e => IO.println s!"rejected: {e.toString}"
```

```leanOutput emReject (whitespace := lax)
rejected: typed graph execution currently supports device `cpu`;
  requested `cuda`
```

The general rule covers more than these four. CUDA in a CPU-only build, typed graph execution with a
non-CPU profile, proof-only arithmetic in `IO`, and an operation with no admissible capsule all fail
rather than quietly changing the requested configuration behind the caller's back. These failures
help protect benchmark provenance. Record the accepted configuration and actual provider so that
a requested device is not mistaken for evidence of where an operation ran.

# Execution Mode Selection

:::table +header
*
  * Goal
  * Arithmetic
  * Mode
  * Profile
*
  * inspect ordinary training
  * native `Float32`
  * eager
  * CPU
*
  * replay a supported fixed graph
  * native `Float32`
  * typed graph
  * CPU
*
  * run native GPU training
  * native `Float32`
  * eager
  * CUDA
*
  * inspect binary32 reference behavior
  * FloatLib binary32
  * eager
  * CPU
*
  * use native attention forward and its matched backward
  * native `Float32`
  * eager
  * CUDA through LibTorch
*
  * verify or export an operation graph
  * semantic context
  * IR evaluator
  * no trainer profile
:::

The final row lowers a model to `NN.IR.Graph` for semantic inspection and proof. See
{ref "spec-layer"}[The Specification Layer] for how to reason about that graph.

# Execution Reproducibility

To make an execution comparison reproducible, record:

```
model architecture and parameter count
dataset identity and preprocessing
seed and optimizer
arithmetic
eager or typed graph execution
device and provider capsules
train or evaluation mode
checkpoint and code revision
```

Without those, two loss curves may be incomparable even when both are labelled "TorchLean float32".
The batching comparison has a nonzero gap that a six-decimal summary hides. Provider and
reduction-policy metadata help investigate such differences; exact buffer comparisons and
controlled experiments are needed to explain them.

Sources:

- {src "NN/API/Trainer/Core.lean"}[Trainer/Core.lean], the `RunConfig` record;
- {src "NN/API/Trainer/Run.lean"}[Run.lean], `forObjective` and `executionSettings`;
- {src "NN/API/Trainer/Session.lean"}[Session.lean], the single session choke point;
- {src "NN/Runtime/Autograd/Torch/Core/Types.lean"}[Types.lean], `validateForExecution`;
- {src "NN/Runtime/Autograd/Torch/Core/Session.lean"}[Session.lean], capsule reporting;
- {src "NN/Backend/Report.lean"}[Report.lean], the evidence lines;
- {src "NN/Examples/Quickstart/SimpleMlpTrain.lean"}[SimpleMlpTrain.lean], the runnable example.

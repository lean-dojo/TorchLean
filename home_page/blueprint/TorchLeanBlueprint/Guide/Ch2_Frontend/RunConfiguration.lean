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
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles
import TorchLean.NN.Loss
import TorchLean.NN.Session.Random
import TorchLean.Verify.Trainer

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean Arithmetic Backend.Device Trainer.Objective
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Choosing How A Model Runs" =>
%%%
tag := "run-configuration"
file := "Choosing-How-A-Model-Runs"
%%%

In {ref "api_tour"}[A Tour Of The API] we trained an MLP that maps a two-component input to a
one-component output, with parameters $`\theta`:

$$`F_\theta:[2]\to[1]`

Its type stays `[2] → [1]` whether it runs on the CPU, with FloatLib's bit-level binary32, or with
LibTorch's CUDA operations. Every run records the same autograd tape, and the device only chooses
who executes each operation. Holding the architecture, seed, and data fixed lets us compare these
choices on the same training problem. Each transcript below records the binary's output on the
machine used for this guide, and the embedded Lean comparisons also run when the page is built.

# Runtime Configuration

TorchLean configures arithmetic, providers, and model mode separately. Not every combination is
supported, so the runtime validates their compatibility. Each setting controls a different part of
the computation:

:::table +header
*
  * Dial
  * Values
  * What it decides
*
  * arithmetic
  * `native`, `ieee`
  * which binary32 semantics every scalar operation obeys
*
  * device and providers
  * `cpu`, `cuda`, a full profile
  * which implementation of each operation is selected, and what evidence backs it
*
  * train or evaluation mode
  * `nn.Mode.train`, `nn.Mode.eval`
  * how mode-sensitive layers such as dropout behave
:::

Changing the device selects implementations of the model's operations. Changing train or evaluation
mode can change the operations' behavior, as dropout will show below. Recording both choices is
necessary to distinguish a provider comparison from a comparison of different forward computations.

```lean (name := runArith)
-- The dispatcher vocabulary is broader than the trainer's
-- accepted arithmetic.
#print Arithmetic
```

```leanOutput runArith (whitespace := lax)
inductive TorchLean.Arithmetic : Type
number of parameters: 0
constructors:
TorchLean.Arithmetic.native : Arithmetic
TorchLean.Arithmetic.ieee : Arithmetic
TorchLean.Arithmetic.complex : Arithmetic
```

Of these three constructors, the supervised trainer accepts two. As the errors below show, the
lower dispatcher can execute complex binary32, but the supervised interface exchanges real data and
results. Complex training instead uses a hand-written real loss with `autograd.complex.grad`,
followed by `nn.sgdStep` on complex state. You can see this separate path in the
`complex_regression` command, which preserves both components in predictions and checkpoints.

The device type has two constructors:

```lean (name := runDevices)
-- A device constructor names a target without asserting
-- that an implementation is linked.
#print Backend.Device
```

```leanOutput runDevices (whitespace := lax)
inductive TorchLean.Backend.Device : Type
number of parameters: 0
constructors:
TorchLean.Backend.Device.cpu : Backend.Device
TorchLean.Backend.Device.cuda : Backend.Device
```

Both have maintained runtime profiles, `checked_cpu` and `libtorch_cuda`. A CPU-only build still
names `cuda`, so naming a device doesn't mean an executable provider is linked for it.

`withDevice` checks for a maintained profile before execution. `validateForExecution` also checks
whether the linked binary can provide the requested runtime:

```lean (name := runValidate)
-- Pure profile lookup and effectful execution validation
-- answer different questions.
#check @Session.Config.withDevice

#check @Session.Config.validateForExecution
```

```leanOutput runValidate (whitespace := lax)
Session.Config.withDevice : Session.Config →
  Backend.Device →
  Except String Session.Config
```

```leanOutput runValidate (whitespace := lax)
Session.Config.validateForExecution :
  Session.Config → IO Unit
```

Because the first returns `Except String Config`, its failure is a value that you can inspect
without running anything. Checking both devices takes a few lines:

```lean (name := runProfiles)
-- Check maintained defaults without requiring the named
-- devices to be present.
#eval do
  let devices : List Backend.Device :=
    [.cpu, .cuda]
  devices.forM fun d => do
    let name := d.cliName
    match Session.Config.withDevice {} d with
    | .ok _ => IO.println s!"{name}: maintained profile"
    | .error msg => IO.println s!"{name}: {msg}"
```

```leanOutput runProfiles (whitespace := lax)
cpu: maintained profile
cuda: maintained profile
```

A custom backend needs both a profile describing admissible kernels and runtime implementations
that execute them. Supplying capsule metadata alone doesn't implement a provider.

With an `Except String Config` result, a caller can handle a missing profile before entering `IO`,
and execution validation then checks the linked runtime and device availability. Beyond these
checks, individual operations have their own shape, layout, arithmetic, and gradient requirements.

# Runtime Flags

Ask the runner to list its examples and runtime flags:

```terminal
# List the runnable examples and the shared entry-point
# options.
scripts/lake.sh exe torchlean --help
```

Its output lists the runnable examples first, then the runtime flags they share:

```
Runtime flags:
  --choose                         ask for a device when the example supports --device
  --device auto|cpu|gpu|cuda
  --arithmetic native|ieee|complex
      arithmetic availability depends on the example; check its --help
  --seed N
  --show-backend

Verification commands live under `scripts/lake.sh exe verify -- list`.
Use `scripts/lake.sh exe torchlean <example> --help` for command-specific flags.
```

While the top-level parser knows three arithmetics, each example states which ones it supports. For
our example:

```terminal
# Inspect the controls accepted by this particular training
# example.
scripts/lake.sh exe torchlean quickstart_mlp --help
```

```
TorchLean simple MLP quickstart

Usage:
  scripts/lake.sh exe torchlean quickstart_mlp [--steps N] [--seed S]
    [--arithmetic native|ieee] [--device cpu|cuda]
    [--show-backend]
```

This example supports `native|ieee` and `cpu|cuda`. An opt-in interactive chooser, `--choose`, can
also select the device (the {ref "cli"}[Command-Line Reference] shows its prompt).

Both the chooser and the flags request a configuration. The banner echoes that request, while the
backend report below names the implementations selected for the model's operations.

# CPU Execution

Our baseline uses native binary32 arithmetic on the CPU and twenty training steps:

```terminal
# Run the short CPU baseline and print the selected backend
# capsules.
scripts/lake.sh exe torchlean quickstart_mlp \
  --device cpu \
  --steps 20 \
  --seed 2026 \
  --show-backend
```

Leaving out `--show-backend` for a moment, the run prints:

```terminal +output
== Quickstart: simple MLP training (seed=2026, steps=20) ==
target(heldout)    = [0.200000]
untrained(heldout) = [-0.047640]
dataset size = 25
mean_loss(before training) = 0.629263
mean_loss(after training) = 0.471118
steps=20 arithmetic=native scalar=Float32 loss=0.629263 -> 0.471118
trained(heldout) = [0.369533]
```

Training opens one session and records each sample's operations on a fresh tape as the model
runs. Every registered operator the model records asks the active profile for an admissible
capsule of that operator, and the session records it with the implementation the capsule names:
on the CPU, the operator's rule in `Op.rule?`, which supplies its value and its local VJP. A
capsule bundles a provider, a trust level, a VJP owner, and a numerical policy, and
`--show-backend` prints each capsule name the first time the session selects it. Here is the
entry for addition, wrapped to fit the page (in the terminal each of these is a single line):

```
  add: reference.add provider=reference trust=checked
    vjp=torchlean-tape reduction=n/a
    shape: shape safety for add; guarded at runtime by
      portable runtime shape checks
    layout: canonical-tensor layout compatibility for add;
      guarded at runtime by typed tensor layout
    value: add forward refines its TorchLean semantics;
      uses canonical definition TorchLean.Op.rule? directly
    vjp: add torchlean-tape VJP refines its TorchLean
      semantics; uses canonical definition TorchLean.Op.rule? directly
```

That entry's four evidence lines distinguish shape, layout, forward-value, and VJP claims. Two cite
runtime guards and two cite the canonical definition that the CPU tape runs. These labels
identify the declared evidence for the selected capsule, but they don't assert a theorem.
{ref "verification"}[Verification And Certificates] develops the proof interfaces.

Our twenty-step CPU run selects seven capsules. Each entry's first word is the operator's tag, its
constructor name in `TorchLean.Op`, so the loss reads `mseLoss` while its capsule is named
`reference.mse_loss`. Tabulating the first line of each entry:

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
  * `mseLoss`
  * `reference.mse_loss`
  * `torchlean-tape`
  * `fixed-left`
:::

Every provider is `reference`, every trust level is `checked`, and every VJP is owned by
TorchLean's own tape. On the two reducing operations, the `fixed-left` policy specifies their
summation order, one of the conditions needed to reproduce floating-point results. That policy
changes under the CUDA profile below.

With `--show-backend`, the run prints the banner `[TorchLean] backend capsules used:` several
times. An eager session prints it when it is created
({src "TorchLean/NN/Session/Lifecycle.lean"}[`Session/Lifecycle.lean`]),
so the banners count sessions, not operations. `trainer.predict` and `trainer.train` each open a
trainer session, which creates one eager session that records a fresh tape for each training
sample, and further eager sessions without gradients for prediction and evaluation loss. Their
banners print together when the trainer session opens, so a banner can have nothing under it. A
banner therefore identifies a session opening, while the entries beneath it identify capsule
selections.

With the tape you can inspect provider choices and use the maintained CUDA runtime, and a manually
assembled eager session can record a different operation sequence on each call. But the model
frontend hides tensor elements behind `Function`, so the tape alone doesn't enable arbitrary
tensor-dependent Lean branches. *Dynamic Control Flow And Typed Graphs* below explains this
frontend boundary.

There are three different loss observations in this short run. The initial `0.629263` is an average
over the 25 training examples. At `step 0`, the loss `0.330237` belongs to the example used for
that update (this run doesn't log it, and the {ref "training-from-scratch"}[training chapter]
prints it). After 20 updates, the final `0.471118` is another evaluation over the dataset.
Comparing the first and last values is meaningful as a change in dataset loss, but comparing either
one with the single-example step loss mixes two different measurements. Twenty updates also need
not visit every example equally often.

A fourth observation is the held-out prediction. Its target is `0.2`, and its trained value here is
`0.369533`. A lower training loss doesn't make that particular prediction exact. Keeping it
separate makes the log useful even when individual update losses fluctuate.

These seven capsule kinds describe the operations needed by the model and loss, and they aren't a
count of kernel launches or arithmetic instructions. Repeated operations may choose the same
capsule, and a report with no new entries during prediction doesn't mean prediction performed no
work. It means that report contains no additional selections to display for that session.

# Training Inside The Guide

Comparisons can also run inside the guide, so building the page checks them. Our setup fixes the
quickstart model, target, dataset, optimizer, and seed:

```lean (name := runSetup)
-- Hold model, target, data, and seed fixed while exposing
-- the runtime choices.
/-- The `2 -> 8 -> 1` regression MLP from the quickstart. -/
def runModel : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![nn.linear 2 8, nn.op .relu, nn.linear 8 1]

/-- Target `0.8 relu(x₁+x₂) - 0.4 relu(x₂-x₁) + 0.2`. -/
def runTarget (x : Tensor Float [2]) : Tensor Float [1] :=
  let relu (v : Float) := if v < 0.0 then 0.0 else v
  let rise := 0.8 * relu (x[0] + x[1])
  let fall := 0.4 * relu (x[1] - x[0])
  [rise - fall + 0.2]

def runInputs : Tensor Float [25, 2] :=
  Data.Synthetic.squareGrid (-1.0) 1.0 5

def runData : Trainer.Dataset [2] [1] :=
  Data.fromTensors runInputs
    (Tensor.mapLeading [25] runTarget runInputs)

/-- A grid point that is not in the training set. -/
def runPoint : Tensor Float [2] := [0.25, -0.75]

/-- One trainer per choice of arithmetic and device. -/
def trainerOn (arithmetic : Arithmetic)
    (device : Backend.Device) : Trainer [2] [1] :=
  Trainer.create runModel
    { objective := mse
      optimizer := optim.adam { lr := 0.03 }
      arithmetic := arithmetic
      device := device
      seed := 2026 }
```

`trainerOn` exposes arithmetic and device as arguments while holding the training problem fixed.
Each call opens a fresh run from seed `2026`, so two runs differ only in the argument that changed,
not in parameters left over from an earlier run.

# CUDA Execution Through LibTorch

CUDA support is a build-time choice. Set `TORCHLEAN_LIBTORCH_HOME` to a CUDA-enabled SDK as
described in the installation guide, then select it with `-K cuda=true`:

```terminal
# Select the CUDA build profile before requesting a CUDA
# training session.
scripts/lake.sh -K cuda=true exe torchlean quickstart_mlp \
  --device cuda \
  --steps 20 \
  --seed 2026 \
  --show-backend
```

This time the report names a different provider for every operation. Comparing it against the CPU
table:

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
  * `mseLoss`
  * `reference.mse_loss`
  * `libtorch.mse_loss`
  * `implementation-defined`
:::

Provider labels change from `reference` to `libtorch`. Evidence names the retained CUDA comparison
for each tested operation, and operations without one record a LibTorch trust boundary. Shape
guards move to CUDA FFI size and rank checks at the Lean and native boundary. The default GPU
profile admits those external implementation assumptions, but the strict checked profile doesn't.

Local VJP ownership depends on the operation. ReLU and broadcast use `backend-vjp`: their
derivatives call native backward primitives. Matmul, addition, MSE, reshape, and permutation retain
`torchlean-tape`: Lean composes their derivatives from LibTorch numerical operations. In both cases
TorchLean records and traverses the tape, with LibTorch autograd recording disabled. For numerical
comparison, the two reducing operations move from `fixed-left` to `implementation-defined`.

To compare the two devices, hold the model, dataset, seed, optimizer, and step count fixed.
A provider with `reduction=implementation-defined` may use a different summation order, and
floating point addition is not associative {Informal.citep goldberg1991}[]. Compare the artifact
relevant to the experiment, such as predictions or parameter buffers, and record its dtype,
provider, and comparison tolerance. Agreement in printed decimal output does not establish equal
parameter bits or equal behavior on other inputs.

Both the module and the CUDA buffer use binary32 values, but upload isn't conversion-free: the
bridge widens host `Float32` elements to `Float` staging values before packing them into a CUDA
float32 buffer. This preserves finite binary32 values but adds transfer work. A capsule identifies
the trusted native provider, but it doesn't prove the kernel, compiler, driver, or device correct.

A build without CUDA support rejects the request. Without a GPU, the default CPU profile supports
the portable runtime, but compiling it doesn't exercise the CUDA provider.

A common seed fixes initialization and the example sequence, but not the order in which a backend
adds partial sums. Updates can carry numerical differences into later steps, and a ReLU can change
its active branch near zero. To investigate a discrepancy, compare the inputs and outputs of the
first differing operation before attributing the final loss difference to the optimizer.

# Executable Binary32 Arithmetic

```terminal
# Use executable binary32 arithmetic for the same short CPU
# workload.
scripts/lake.sh exe torchlean quickstart_mlp \
  --arithmetic ieee \
  --device cpu \
  --steps 2 \
  --seed 2026
```

The `ieee` path evaluates addition and multiplication with FloatLib binary32. Comparing the two
implementations can expose errors in rounding or exceptional-value handling that a comparison of
decimal output alone may miss. Formal floating-point models make these rules explicit
{Informal.citep goldberg1991}[]. Flocq made such a model reusable inside a proof assistant
{Informal.citep flocq2011}[], and the same group's later work connects such a model to the
arithmetic a compiler actually emits {Informal.citep boldo2015}[]. FloatLib supplies the executable
format and arithmetic used here. For higher precision, select a valid FloatLib binary format in
typed CPU tensors and graphs, because the trainer flag shown above remains fixed to binary32.

For typed training, `trainer.openTyped (α := α)` keeps inputs, state, predictions, and losses in
the chosen scalar type on CPU. It rejects GPU devices and custom backend profiles. Supply
`initialState?` when the parameters need precision beyond the seeded `Float` initializers, because
converting those initializers into a wider format doesn't add precision. Even in typed training,
optimizer and scheduler coefficients are configured in `Float` and converted into the selected
scalar.

Next, an embedded comparison runs two hundred steps with each arithmetic choice, holding the seed
and dataset fixed. It also checks prediction equality:

```lean (name := runNativeVsIeee)
-- Keep the device fixed to compare the two binary32
-- implementations.
#eval do
  let opts : Trainer.TrainOptions :=
    { steps := 200, logEvery := 0 }
  let native ← (trainerOn native cpu).train runData opts
  let ieee ← (trainerOn ieee cpu).train runData opts
  IO.println s!"native = {native.summary}"
  IO.println s!"ieee   = {ieee.summary}"
  let a ← native.predict runPoint
  let b ← ieee.predict runPoint
  IO.println s!"native prediction = {a}"
  IO.println s!"ieee   prediction = {b}"
  IO.println s!"predictions equal = {a == b}"
```

```leanOutput runNativeVsIeee (whitespace := lax)
dataset size = 25
mean_loss(before training) = 0.629263
mean_loss(after training) = 0.002308
dataset size = 25
mean_loss(before training) = 0.629263
mean_loss(after training) = 0.002308
native = steps=200 arithmetic=native scalar=Float32
  loss=0.629263 -> 0.002308
ieee   = steps=200 arithmetic=ieee scalar=ExecFloat.Binary 8 23
  loss=0.629263 -> 0.002308
native prediction = [0.189033]
ieee   prediction = [0.189033]
predictions equal = true
```

You can check the held-out target without running either trainer. At `[0.25, -0.75]`, the sum of
the coordinates is `-0.5` and their difference `x₁ - x₀` is `-1`. Both ReLU terms in the target
function therefore vanish, leaving its constant `0.2`. The common prediction `0.189033` is about
`0.010967` below that target. Even though both came from the same trained parameters, the small
final dataset loss and this remaining prediction error answer different questions.

Agreement on one model, one seed, and one dataset is a test result. A differential test between a
hardware instruction and a bit-level implementation can expose rounding, tie, or sign errors
exercised by its inputs and assertions. It says nothing about inputs we didn't try.
{ref "floats"}[Floating Point Semantics] tells the theorem-side story: there the proof-oriented
`FP32` model and exact `Real` appear in statements, not in `IO`, and the finite and no-overflow
bridge from `FP32` to FloatLib binary32 is stated. `FP32` rounds reals at binary32 precision with
gradual underflow, and it doesn't model overflow, NaN, infinity, or signed zero, which is why the
bridge has hypotheses.

FloatLib's bit-level implementation can cost more than native arithmetic. Measure that cost on the
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

```lean (name := runConfigs)
-- Record updates isolate runtime changes while retaining
-- the optimizer settings.
def runCpu : Trainer.RunConfig :=
  { arithmetic := native
    device := cpu
    optimizer := optim.adam { lr := 0.03 } }

def runIeeeCpu : Trainer.RunConfig :=
  { runCpu with arithmetic := ieee }

def runCuda : Trainer.RunConfig :=
  { runCpu with device := cuda }

/-- Bind a run configuration to an objective and a seed. -/
def trainerFromRun (run : Trainer.RunConfig) :
    Trainer [2] [1] :=
  Trainer.create runModel
    { run with objective := mse, seed := 2026 }

#eval do
  let runs := [runCpu, runIeeeCpu, runCuda]
  runs.forM fun (run : Trainer.RunConfig) => do
    let device := run.toConfig.device.cliName
    IO.println s!"{device} {run.arithmetic.cliName}"
```

```leanOutput runConfigs (whitespace := lax)
cpu native
cpu ieee
cuda native
```

Record update syntax makes `runIeeeCpu` and `runCuda` each differ from `runCpu` in exactly one
field. `RunConfig` extends `Session.Config` with the optimizer and arithmetic, and `Trainer.Config`
extends `RunConfig` with the objective, so one record update such as
`{ run with objective := mse, seed := 2026 }` produces a full `Trainer.Config`. That lets you reuse
one runtime configuration across several objectives. `run.toConfig` is the `Session.Config` that
the session layer actually validates, which is the value that `validateForExecution` inspects.

Record updates go the other way too: `{ runtime with optimizer := ... }` starts a `RunConfig` from
a `Session.Config` parsed from the command line. A sweep therefore declares the model once, keeps
one base `RunConfig`, and derives each cell of the sweep from it.

Constructing `runCuda` doesn't open CUDA. These labels print only the device and arithmetic fields
of each record and omit optimizer settings and provider overrides. Save those fields too when
comparing runs.

# Graph Lowering And Direct Execution

The trainer runs on the tape and keeps no graph. When a program needs a typed graph, for example to
take forward-mode derivatives, lower the model directly:

```lean (name := runGraphTypes)
-- Graph leaves contain state followed by input, and the
-- wrapper separates them for callers.
#check @nn.Sequential.lowerToTypedGraph
#check @nn.TypedGraphModel.forward
```

```leanOutput runGraphTypes (whitespace := lax)
@nn.Sequential.lowerToTypedGraph :
  {σ τ : Shape} → (model : nn.Sequential σ τ) →
  optParam nn.Mode
    nn.Mode.eval →
  {α : Type} → [inst : Storage α] → [Context α] →
  IO (TypedGraph α
    (model.stateShapes ++ [σ]) τ)
```

```leanOutput runGraphTypes (whitespace := lax)
@nn.TypedGraphModel.forward : {σ τ : Shape} → {α : Type} →
  [inst : Storage α] → {stateShapes : List Shape} →
  nn.TypedGraphModel stateShapes σ τ α →
  TensorPack α stateShapes → Tensor α σ → Tensor α τ
```

`lowerToTypedGraph` returns a `TypedGraph α (model.stateShapes ++ [σ]) τ`, whose index is a flat
list of every graph leaf: parameters first, then the model input. `nn.TypedGraphModel` is the
model-facing view of that same type, and its `forward` splits the leaves back into
`TensorPack α stateShapes` and `Tensor α σ`. Nothing is converted: the second signature is a way of
talking about the first that keeps the parameters and the input apart.

`nn.TypedGraphModel` is an `abbrev`, so it is transparent to dot
notation. Writing `g.forward state input` resolves to the raw `TypedGraph.WithData.forward`, which
wants one `TensorPack` over the concatenated leaves and the unit data `()`, and rejects the two
arguments. Call the model-facing wrappers by name, as below.

```lean (name := runGraph)
-- Zero parameter tangents isolate derivatives with respect
-- to the input coordinates.
#eval do
  let built := nn.build 2026 runModel
  let g ← nn.Sequential.lowerToTypedGraph built (α := Float)
  let state := nn.initialState built
  let shapes := nn.Sequential.stateShapes built
  let y := nn.TypedGraphModel.forward g state runPoint
  IO.println s!"forward = {y}"
  let (_, inputGradient) :=
    nn.TypedGraphModel.vjp g state runPoint [1.0]
  IO.println s!"input gradient = {inputGradient}"
  IO.println s!"parameter shapes = {shapes}"
  let zero : TensorPack Float shapes := .zero
  let jvp := nn.TypedGraphModel.jvp g state zero runPoint
  let d0 := jvp [1.0, 0.0]
  let d1 := jvp [0.0, 1.0]
  IO.println s!"jvp along e₀ = {d0}, jvp along e₁ = {d1}"
```

```leanOutput runGraph (whitespace := lax)
forward = [-0.047640]
input gradient = [0.142060, 0.110873]
parameter shapes = [[8, 2], [8], [1, 8], [1]]
jvp along e₀ = [0.142060], jvp along e₁ = [0.110873]
```

`forward = [-0.047640]` is the same number the runner printed as `untrained(heldout)` in every
experiment above, which is a small consistency check that the lowered graph and the trainer agree
on the model. And the parameter shapes `[[8, 2], [8], [1, 8], [1]]` are the two weight matrices and
two bias vectors of a `2 → 8 → 1` MLP, in declaration order.

Seeded with the output cotangent `[1.0]`, the reverse mode pass returns the input gradient
`[0.142060, 0.110873]`. The two forward mode passes, seeded with the basis tangents `[1.0, 0.0]`
and `[0.0, 1.0]` and a zero parameter tangent, return `[0.142060]` and `[0.110873]`. For a function
with a one-dimensional output these must coincide, because both are computing the same
$`1 \times 2` Jacobian, one row at a time from the left and one column at a time from the right.
They coincide to the last printed digit. Comparing forward and reverse directional derivatives
checks their agreement without requiring a separate symbolic derivative
{Informal.citep baydin2018}[]. Retaining the graph lets all three calls use the same recorded
operations.

Two lifetimes to keep straight. `nn.Sequential.lowerToTypedGraph` hands you a graph that lives as
long as the reference does. An eager session instead records one tape per recording phase, and
`resetTape` starts a fresh one. When the graph itself must survive across calls, lower it.

This example runs in `Float`, so it isn't another execution of the trainer's binary32 comparison,
despite the similar printed initial prediction.

# Device And Provider Profiles

Ordinary code selects a device and lets TorchLean resolve the maintained profile for it. Advanced
code can select a complete profile with `withBackendProfile`, choosing provider preference,
assurance policy, VJP ownership, target availability, and implementation tables together. Calling
`withDevice` afterwards clears that override, which is deliberate: a device name and a hand-built
profile are two ways to answer the same question, and the last answer wins rather than silently
merging.

{ref "backend-selection"}[Backend Selection] follows an operation request through provider
selection, contract checks, and the implementation that runs it.

# Train And Evaluation Mode

At the high level, `Trainer.Session` selects mode from the operation you call. Lower-level module
APIs also provide explicit mode setters:

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

`step` returns `Unit` by default, so an update doesn't read the loss back to the host. Add
`(loss := true)` when the caller needs the loss from that update. Setting the `batch` flag selects
a nonempty array of samples, but it doesn't change training mode.

Dropout {Informal.citep dropout2014}[] makes the effect of mode visible. During training it samples
a mask, while during evaluation it applies its deterministic inference behavior. In the next block
we insert dropout between the MLP's hidden activation and output layer, then evaluate the same
dataset through both session paths:

```lean (name := runDropout)
-- Evaluate before the batch update so the mode comparison
-- starts at the same state.
/-- The same MLP with one dropout layer. -/
def runDropoutModel (p : Float) :
    nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![nn.linear 2 8, nn.op .relu, nn.dropout p,
    nn.linear 8 1]

def runDropoutTrainer (p : Float) : Trainer [2] [1] :=
  Trainer.create (runDropoutModel p)
    { objective := mse
      optimizer := optim.adam { lr := 0.03 }
      seed := 2026 }

#eval do
  let batch := (← runData.materialize (α := Float)).toArray
  for p in [0.0, 0.5] do
    let session ← (runDropoutTrainer p).open
    let evaluation ← session.eval runData
    let training ← session.step batch
      (batch := true) (loss := true)
    IO.println s!"p = {p}"
    IO.println s!"  evaluation mode {evaluation}"
    IO.println s!"  training mode   {training}"
    let gap := (training - evaluation) * 1e9
    IO.println s!"  gap * 1e9 = {gap}"
```

```leanOutput runDropout (whitespace := lax)
p = 0.000000
  evaluation mode 0.629263
  training mode   0.629263
  gap * 1e9 = -57.220459
p = 0.500000
  evaluation mode 0.360462
  training mode   0.495756
  gap * 1e9 = 135293566.733599
```

Read the gaps, not the printed losses. At `p = 0.0` the layer is the identity in both modes and the
two numbers print the same to six decimals, yet the gap is a nonzero `-5.7e-8`. `eval` converts
per-sample losses to `Float` and averages them there, while `step` with `(batch := true)` uses the
runtime scalar for its batch reduction. This comparison changes both reduction structure and the
precision of the final average, so it doesn't isolate summation order alone.

At `p = 0.5` the much larger gap illustrates the effect of training-mode dropout in this example,
while the `p = 0` result is a useful control, not a universal bound on floating point
discrepancies.

Repeating the `p = 0.5` measurement in freshly opened sessions gives the same result for this seed:
the builder derives the dropout layer's seed from the model build seed. This demonstrates seeded
reproducibility, but it doesn't establish independence of masks across calls. `Session.save` saves
model state, but not the optimizer history, loader position, or random stream needed for exact
training resume. At a lower level, `Checkpoint.Optimizer.save` separately supports backend-owned
CUDA Adam/AdamW state and its eager random counter.

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
training. In TorchLean, `Trainer.Session` chooses the mode for these operations, while lower-level
APIs require the caller to select it. Our example derives its training-mode randomness from the
build seed.

Mode also determines which random state the reverse rule must retain. Consider the operation

$$`y=\operatorname{Dropout}_{p}(x)`.

During training a mask is realized and retained for the backward rule. Re-running the backward pass
with a freshly sampled mask wouldn't differentiate the forward value that was actually computed. It
would differentiate a different function with the same input and output shapes. So the reverse rule
must retain the realized mask, or enough state to reproduce that exact draw.

Each dropout measurement starts a fresh session and evaluates before performing the batch update,
so within that session the evaluation and the training loss refer to the same initial parameter
state. The training loss is produced as part of the update and isn't a second evaluation of the
updated model. Evaluating after the update would mix the effect of mode with the effect of changed
parameters.

# Dynamic Control Flow And Typed Graphs

A fixed typed model graph needs its operation structure and its shapes to be known when it is
recorded. A program that reads a token value and then changes that structure cannot be represented
by this recorder as one fixed graph.

A scalar conditional inside a tensor computation is a different matter. TorchLean's
{ref "custom-computations"}[custom computation frontend] supports `if` expressions on tensor
elements on CPU and GPU, and such a conditional stays in the scalar program: it doesn't change the
model graph's nodes or shapes.

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

The recorded graph contains one `aten::mul` and no comparison, so on the negative input it returns
`[-2.0, -2.0]`, where the original function takes the unrecorded `else` branch and returns
`[-101.0, -101.0]`. Matching the input shape didn't preserve the input-dependent choice.

PyTorch's modern export path refuses instead:

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
recording operations. Its shape indices check tensor dimensions and references, but they don't
prove that an arbitrary frontend program preserves data-dependent branching. A fixed graph must
represent the intended choice explicitly, not specialize it to one sample's values.

Data-dependent structure can remain in an eager frontend that supports it. Other options are to
express the choice as a supported tensor operation, such as a mask or select, or record separate
static branches and choose between them at runtime. In each case, the implementation must preserve
the choice on subsequent inputs.

Agreement on several positive-sum inputs wouldn't expose this trace's defect: all of them follow
the branch that recording retained. A useful probe crosses the decision boundary while keeping the
tensor shape fixed, which is what the negative input above does. More generally, validating a
captured program requires inputs that distinguish its possible behaviors, not just repeated
examples from the region in which it was captured.

A branch on a fixed model configuration can be resolved while constructing a graph, but a branch on
a future tensor value needs a representation of that decision in the executable program.
TorchLean's polymorphic function interface uses opaque tensor references even when run eagerly, and
it doesn't expose their elements as ordinary Lean booleans. Supporting that style of dynamic
control flow would require an appropriate frontend and runtime interface, which the model API
doesn't provide.

# Unsupported Runtime Configurations

The requests below fail at different configuration boundaries. Their messages are wrapped here to
fit the page, and each is a single line in the terminal.

An unknown device:

```terminal +output
$ # Request a device name the parser doesn't know.
$ scripts/lake.sh exe torchlean quickstart_mlp --device metal --steps 1
error: quickstart_mlp: unknown device metal (known devices:
cpu|gpu|cuda)
```

An arithmetic the lower dispatcher can execute but the supervised trainer can't use,
`--arithmetic complex`, is refused when the flags are parsed (the {ref "cli"}[Command-Line
Reference] shows the message).

A CUDA build with no visible device:

```terminal +output
$ # Hide visible GPUs while keeping the native CUDA build to
$ # isolate runtime availability.
$ CUDA_VISIBLE_DEVICES="" scripts/lake.sh -K cuda=true exe torchlean quickstart_mlp --device cuda
error: torch eager session: CUDA was requested and
this is a CUDA build, but no usable CUDA device is visible
```

That last message distinguishes the two failure modes it could be reporting. A build without
LibTorch says so and tells you to rebuild with `-K cuda=true`, while a real CUDA build with no
device gives the message above. Both refuse, and both are raised when the session opens, not when
the flags are parsed, so session validation checks programmatic configurations as well as
command-line requests.

Because the rejection is a value in Lean, we can exhibit one here without a terminal. A
configuration built in code passes no flag parser, so the same complex-arithmetic request fails
when the session opens:

```lean (name := runReject)
-- Ask the supervised trainer for complex arithmetic and
-- catch the session-opening failure.
#eval do
  try
    let trainer := Trainer.create runModel
      { objective := mse, arithmetic := complex,
        seed := 2026 }
    let _ ← trainer.train runData
      { steps := 1, logEvery := 0 }
    IO.println "the run was accepted"
  catch e => IO.println s!"rejected: {e.toString}"
```

```leanOutput runReject (whitespace := lax)
rejected: TorchLean.Trainer: supervised training supports real
  arithmetic; complex arithmetic requires an explicit complex-valued
  training API
```

The general rule covers more than these three. CUDA in a CPU-only build, proof-only arithmetic in
`IO`, and an operation with no admissible capsule all fail instead of quietly changing the
requested configuration behind the caller's back. These failures help protect benchmark provenance.
Record the accepted configuration and actual provider so that a requested device isn't mistaken for
evidence of where an operation ran.

# Choosing A Configuration

:::table +header
*
  * Goal
  * Arithmetic
  * Profile
*
  * inspect ordinary training
  * native `Float32`
  * CPU
*
  * run native GPU training
  * native `Float32`
  * CUDA
*
  * inspect binary32 reference behavior
  * FloatLib binary32
  * CPU
*
  * run Lean-composed attention on GPU primitives
  * native `Float32`
  * CUDA through LibTorch
*
  * verify or export an operation graph
  * semantic context
  * the IR semantics, no trainer profile
:::

For the final row, TorchLean lowers a model to a graph of registered operators (`IR.Graph`)
for semantic inspection and proof. See {ref "spec-layer"}[The Specification Layer] for how to
reason about that graph.

# Reproducible Comparisons

To make a run comparison reproducible, record:

```
model architecture and parameter count
dataset identity and preprocessing
seed and optimizer
arithmetic
device and provider capsules
train or evaluation mode
checkpoint and code revision
```

Without those, two loss curves may be incomparable even when both are labelled "TorchLean float32".
The batching comparison, for instance, has a nonzero gap that a six-decimal summary hides. Provider
and reduction-policy metadata help investigate such differences, but explaining them takes exact
buffer comparisons and controlled experiments.

Sources:

- {src "TorchLean/NN/Trainer.lean"}[NN/Trainer.lean], the `RunConfig` record
- {src "TorchLean/NN/Trainer/Session.lean"}[Session.lean], the single session choke point
- {src "TorchLean/NN/Session/Config.lean"}[Config.lean], `validateForExecution`
- {srcDir "TorchLean/NN/Session"}[Session/], capsule reporting
- {src "TorchLean/NN/Backend/Report.lean"}[Report.lean], the evidence lines
- {src "TorchLeanExamples/Quickstart/MLP.lean"}[MLP.lean], the runnable
  example.

import VersoManual
import TorchLean.NN.Sequential.Literal
import TorchLean.NN.Builder.MoE
import TorchLean.NN.Sequential.Training
import TorchLean.NN.Optimizer.Constructor
import TorchLean.NN.Session.Selection
import TorchLean.NN.Trainer.Constructor
import TorchLean.Verify.Lowering
import TorchLean.NN.Backend.Graph
import TorchLean.NN.Backend.LibTorch
import TorchLean.CUDA.LibTorch.Controls
import TorchLean.NN.IR.CustomRule
import TorchLean.NN.IR.Tangent
import TorchLean.NN.IR.Program
import TorchLean.Verify.RuntimeApprox.OnlineSoftmax
import TorchLean.NN.Attention.CausalMask
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles
import TorchLean.NN.IR.OpGraph
import TorchLean.NN.Loss
import TorchLean.NN.Session.Random
import TorchLean.Tensor.Op.Pointwise.Real

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean Backend.Device Trainer.Objective
open Backend Contract
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Backend Selection" =>
%%%
tag := "backend-selection"
file := "Inside-The-Backend-Planner"
%%%

Selecting CPU or CUDA through the runtime API, as on the previous page, leaves a less visible
decision: when a graph asks for matrix multiplication, attention, or a reduction, TorchLean has to
decide which implementation is allowed to answer.

A device name isn't enough. TorchLean's CUDA attention, for instance, composes matrix products and
softmax in Lean over ATen numerical primitives, and its intermediate storage, backward rule, and
supporting evidence need to be identified. So the backend planner keeps those contracts in data,
and it either returns an accepted plan or explains why it couldn't make one.

The path is:

$$`
\text{operation}+\text{profile}+\text{available providers}
\longrightarrow \text{capsule}
\longrightarrow \text{audit}
\longrightarrow \text{contract check}
\longrightarrow \text{accepted kernel}
\longrightarrow \text{implementation}.
`

In the Lean examples the planner runs during elaboration. We can ask it to select an attention
kernel, misfile a contract deliberately, and inspect the rejection before any kernel executes.

Device selection first requires a profile. Both device constructors have maintained profiles:

```lean (name := bsDevices)
-- Inspect maintained policy defaults without probing a
-- device or launching a kernel.
#eval do
  for d in [Device.cpu, .cuda] do
    match Profile.maintainedForDevice? d with
    | some p => IO.println s!"{d.cliName}: {p.name}"
    | none =>
      IO.println s!"{d.cliName}: no maintained profile"
```
```leanOutput bsDevices
cpu: checked_cpu
cuda: libtorch_cuda
```

Returning an `Option` distinguishes a maintained profile from an unsupported request. When the
result is `none`, the caller must supply a profile describing admissible implementations and their
evidence before execution can proceed.

The `cpu` and `cuda` lines report names of maintained policies, not results of hardware probes. On
the CPU the default uses checked assurance, while the CUDA default admits the explicitly recorded
LibTorch boundary for operations without retained numerical comparisons
(`Profile.checkedCUDA` remains available when we want planning to reject those operations
instead). That distinction lets a tool inspect a CUDA plan on a machine that can't execute it. A
custom profile must describe the provider it intends to use. An eager session runs reference CPU
and LibTorch CUDA capsules only, and rejects a capsule of any other provider or device before it
runs, so another provider also needs an implementation in the runtime. Keeping planning pure makes
missing coverage visible before data transfer or training begins.

# Kernel Capsules

Suppose a graph reaches scaled dot-product attention. Its CUDA capsule describes a Lean composition
of matrix products and softmax, with an explicit local VJP over saved probabilities. LibTorch
supplies the numerical primitives while TorchLean owns the tape, saved buffers, and parameter
gradients. The capsule's contract covers this composition. It doesn't select a fused attention
implementation or record a LibTorch autograd graph.

A `KernelCapsule` records those differences:

```
-- These fields describe an implementation contract before
-- any implementation runs.
structure KernelCapsule where
  name : String
  op : Op.Wire
  provider : Provider
  device : Device
  trustLevel : TrustLevel
  supportsForward : Bool
  vjpMode : VJPMode
  shapeContract : Contract.Descriptor
  layoutContract : Contract.Descriptor
  valueContract : Contract.Descriptor
  vjpContract : Contract.Descriptor
  numericalPolicy : NumericalPolicy
```

The `op` field is the wire record of a registered operator of `TorchLean.Op`: the constructor name
in `tag`, such as `matmul`, and the shapes and attributes the operator was built with. A capsule
therefore describes one operator at one set of attributes.

A `KernelContract` holds the rest of a capsule without its operator, provider, and device: name,
trust level, forward and VJP support, layout, the explanation and evidence of the four contract
fields, and numerical policy. `KernelContract.capsule` builds the capsule of one operator for a
provider and a device, and every claim in it names that operator's record. `Reference.contract op`
is a contract of this kind, and so is the `contract?` of a `LibTorch.Kernel.ofOp` entry. The
planner may select a capsule only when its device, provider, gradient mode, and trust level fit the
requested profile. If no capsule fits, planning stops with an error.

Capsules are offered by named implementation tables. An `Implementation` maps each registered
operator to the capsule it offers, or to `none`. The maintained registry holds two tables: the
LibTorch CUDA table offers the capsule of the contract of each `LibTorch.Kernel.ofOp` entry (every
entry but `detach`'s carries one), and the reference table offers the capsule of each operator with
a reference contract. Attention has no table of its own: `multiheadAttention` is an operator in
both. A downstream provider can prepend another table with `Profile.withImplementations` without
adding a new model class or a branch to the graph walker. With or without it, the model lowers to
registered operators, and the planner either finds an admissible capsule for each operator or
reports the missing one.

Registry validation rejects duplicate implementation names:

```lean (name := bsModules)
-- Duplicate implementation names are an error even if each
-- table offers valid capsules.
#eval Registry.validate
  #[Reference.implementation, Reference.implementation]

#eval Registry.validate Registry.maintained
```
```leanOutput bsModules
Except.error "duplicate backend implementation `reference`"
```
```leanOutput bsModules
Except.ok ()
```

A table answers for an operator with its attributes, so it can offer a capsule at some attributes
and decline others. For example, the LibTorch table offers `cat` only along the first axis of two
parts and offers no capsule for masked softmax. Both maintained profiles share the two tables, and
the device filter happens when a plan is made, not when a table is registered.

For the record and the tables, see {src "TorchLean/NN/Backend/Capsule.lean"}[`Capsule.lean`], and
for the maintained list, {src "TorchLean/NN/Backend/Registry.lean"}[`Registry.lean`]. The
installation guide has an [installation and kernel
overview](https://lean-dojo.github.io/TorchLean/installation/#from-a-model-to-a-kernel).

Distinct tables can still offer the same operator, and the profile decides which one to use.

# Capsule Contracts And Evidence

Each capsule has four obligations: shape, layout, forward value, and VJP. Each is a
`Descriptor`: a structured `Claim`, a one-line summary for readers, and the evidence
behind the claim.

Each claim is one of shape safety, compatibility with a named tensor layout, refinement of a
forward specification, refinement of a VJP specification, or `vjpUnavailable` for a forward-only
capsule. Before planning, TorchLean checks that each descriptor has the right obligation kind and
operation, and that a VJP descriptor names the capsule's declared VJP mode.

For example, moving a value test into the portable attention capsule's shape field makes the
descriptor disagree with its assigned obligation:

```lean (name := bsMisfiled)
-- Keep the value claim intact but deliberately assign it to
-- the wrong obligation. Scores [3, 3], values [3, 2].
def bsAttend :=
  Op.attentionFromScores (batch := []) (rows := 3)
    (keys := 3) (width := 2) none

def bsPortable : KernelCapsule :=
  (Reference.contract bsAttend).capsule .reference .cpu
    bsAttend

def bsMisfiled : KernelCapsule :=
  { bsPortable with
    name := "demo.misfiled"
    shapeContract :=
      Descriptor.tested
        (.valueRefinement bsPortable.op)
        "value test filed as shape evidence" "demo suite" }

#eval (bsPortable.contractsAligned,
  bsMisfiled.contractsAligned)
```
```leanOutput bsMisfiled
(true, false)
```

A misaligned capsule isn't merely reported: `KernelCapsule.admissible` requires `contractsAligned`,
so the planner won't select it at all. Here the table built by `Implementation.single` offers that
one capsule for its operator:

```lean (name := bsMisfiledPlan)
-- The malformed descriptor must prevent selection, not
-- merely appear in a warning.
#eval planOp Profile.checkedCPU.policy
    #[Implementation.single "demo" bsMisfiled] bsAttend
  |>.map fun k => k.capsule.name
```
```leanOutput bsMisfiledPlan (whitespace := lax)
Except.error "no admissible kernel capsule for op
  attentionFromScores on device cpu"
```

A value test placed in the shape field is therefore rejected rather than counted as shape evidence,
and a forward-only capsule cannot describe `.none` as though it were a VJP refinement.

`Evidence` classifies the evidence in five ways:

- `sharedDefinition name` records that the reference implementation calls the named canonical
  definition directly, which is shared code rather than a separate floating-point accuracy guarantee
- `runtimeGuard name` records validation performed at an execution boundary
- `testSuite name` records a regression or differential test suite
- `trustedBoundary reason` names code whose correctness is assumed for the claim
- `notApplicable` records an obligation that the capsule intentionally does not provide.

There is no "proved" evidence variant: this descriptor doesn't carry a Lean proof of native
implementation correctness. What a capsule can say is which canonical definition it calls, which
guards and tests stand behind it, or that it delegates to code TorchLean doesn't check. In the
related idea of proof-carrying code {Informal.citep necula1997}[], a producer has to ship a
checkable certificate with its binary. A capsule carries a weaker record: a classified claim and
its evidence source. So the checker verifies that classification and the caller's policy, without
verifying the kernel.

`NumericalPolicy` has one field, `reduction : ReductionPolicy`, with values `fixedLeft`,
`fixedLeftOrExact`, `implementationDefined`, and `notApplicable`. Of the portable capsules, the
matrix product records the fixed left fold used by the tensor semantics, and the scaled matrix
product records `fixedLeftOrExact`: the fixed left fold unless its checked arithmetic selects exact
replay. Each LibTorch CUDA capsule of a reduction or accumulation records an implementation-defined
reduction, so a fixed-left range trace can't be reused for an ATen schedule merely because both
capsules implement `matmul`. There are precision and deterministic-algorithm controls in the
runtime, but the capsule doesn't audit their readbacks or specify every rounding, subnormal, and
multiply-add choice.

Trust has two levels, `checked` and `trustedExternal`. Both maintained profiles accept `checked`
capsules whose evidence is shared definitions, guards and tests, while a capsule whose value
contract rests on a trusted boundary is admitted only under the explicit `external` policy, even if
the capsule itself is labelled `checked`: the contract check reads the evidence, not the label.

Alignment checks that each claim belongs in its assigned field. Whether a named test adequately
covers that claim remains a separate question about the test and implementation.

For a matrix product, layout and numerical policy answer different questions. Layout determines
which stored elements are interpreted as each row and column. Reduction policy determines the
order in which products are added. Two implementations may agree on every address and still
produce different last bits through their summation orders. A shape check can catch incompatible
matrix dimensions but cannot establish either of those numerical relationships. This is why a
capsule carries several obligations instead of one broad correctness label.

# Kernel Selection Plans

Planning is a pure function from a profile and a list of operators to a plan or an error message,
so a plan can be printed. Each operator is passed with its ports, as `⟨_, _, op⟩`. This helper
prints one block per selected capsule:

```lean (name := bsPlan)
-- Print the selected implementation and the evidence
-- relevant to its forward values.
def bsRows (profile : Profile)
    (ops : Array (Σ ins outs, Op ins outs)) : IO Unit := do
  match profile.planOps ops with
  | .error e => IO.println s!"planning failed: {e}"
  | .ok plan =>
    for k in plan.kernels do
      IO.println s!"{k.op.tag} -> {k.capsule.name}"
      IO.println s!"  provider {k.capsule.provider.label}, \
        trust {k.capsule.trustLevel.label}"
      IO.println s!"  reduction \
        {k.capsule.numericalPolicy.reduction.label}"
      let value := k.capsule.valueContract.evidence.label
      IO.println s!"  value: {value}"

-- A [2, 3] by [3, 4] matrix product.
def bsMatmul :
    Op [.value [2, 3], .value [3, 4]] [.value [2, 4]] :=
  Op.matmul (m := 2) (n := 3) (p := 4) false false
    (fun h => nomatch h) (fun h => nomatch h)
    (Shape.CanBroadcastTo.refl [])
    (Shape.CanBroadcastTo.refl [])

#eval bsRows Profile.checkedCPU
  #[⟨_, _, bsMatmul⟩, ⟨_, _, Op.relu (s := [2])⟩]
```
```leanOutput bsPlan
matmul -> reference.matmul
  provider reference, trust checked
  reduction fixed-left
  value: uses canonical definition TorchLean.Op.rule? directly
relu -> reference.relu
  provider reference, trust checked
  reduction n/a
  value: uses canonical definition TorchLean.Op.rule? directly
```

Here the portable matrix product commits to a fixed left fold, and `relu` reports `n/a` because a
pointwise operation performs no reduction at all. Those are the labels a range trace or a benchmark
record should quote, and {ref "runtime-approximation"}[Runtime Approximation] explains why a
summation order is not a detail.

A missing implementation is an ordinary value, available before any kernel is launched:

```lean (name := bsGap)
-- A scan along the first axis of [5, 2] and a real FFT of
-- length 8. The reference table has a capsule for neither.
#eval do
  let ops : List (Σ ins outs, Op ins outs) :=
    [⟨_, _, Op.selectiveScan (s := [5, 2])
        ⟨0, by decide⟩ false false⟩,
     ⟨_, _, Op.rfft (s := [2, 8]) ⟨1, by decide⟩ 8
        (by decide)⟩]
  for op in ops do
    match Profile.checkedCPU.planOps #[op] with
    | .ok plan => IO.println s!"{plan.capsuleNames}"
    | .error e => IO.println e
```
```leanOutput bsGap
#[reference.selectiveScan]
#[reference.rfft]
```

Only the LibTorch CUDA table offers capsules for `selectiveScan` and `rfft`, so the CPU profile
can't plan either operator. Generic differentiable scan and real-FFT reference paths can instead
express their computation through ordinary operators that the reference table covers. Each error
identifies a missing capsule for the requested operator, not the absence of every CPU
implementation of the underlying algorithm.

Those `reference.matmul` and `reference.relu` rows also explain how to read a successful plan: the
name identifies the implementation family, the provider identifies its owner, and the
value-evidence line names the retained test source. While printing the report the planner doesn't
run that suite: its input is the registered evidence record. A benchmark that stores only
`checked_cpu` loses these per-operation choices, but storing the selected capsules makes later
changes in preference or registry order inspectable.

## The Implementation Inside An Attention Capsule

Selecting `libtorch.multihead_attention` identifies TorchLean's Lean-composed attention route. ATen
chooses the implementations of the route's numerical primitives, but there is no fused attention
provider selection. Even so, the public controls in
{src "TorchLean/CUDA/LibTorch/Controls.lean"}[`LibTorch`] expose the SDK's flash, efficient, math,
and cuDNN attention permissions.

For example, this definition changes the SDK's flash permission when called, but it doesn't change
TorchLean's attention composition:

```lean (name := bsSDPControl)
def bsDisableFlash : IO Unit :=
  LibTorch.setSDPEnabled .flash false
```

`getSDPEnabled` reads that SDK permission. TorchLean's attention calls matrix-product and softmax
primitives directly, so these permissions do not govern its forward or backward execution.

Before data transfer the planner can inspect and reject the capsule, but its accepted plan doesn't
establish that the request fits device memory: attention materializes the full score matrix and
saves probabilities, using quadratic memory in sequence length. {ref "gpu-and-cuda"}[GPU and CUDA]
follows those buffers through forward and backward.

# The Attention Specification Theorem

TorchLean's FlashAttention specification is the online schedule of `Tensor.attention`, and its
theorem gives a precise target for refinement:

```lean (name := bsFlash)
-- The tiled schedule against the two-pass schedule,
-- over the reals.
#check @Tensor.attention_online_eq
```
```leanOutput bsFlash (whitespace := lax)
@Tensor.attention_online_eq : ∀ {batch : Shape} {rows keys depth width : ℕ}
  (query : Tensor ℝ (batch ++ [rows, depth])) (key : Tensor ℝ (batch ++ [keys, depth]))
  (value : Tensor ℝ (batch ++ [keys, width])) (scale : ℝ)
  (mask : Option (Tensor Bool [rows, keys]))
  (weights : Option (Tensor ℝ (batch ++ [rows, keys]))) (blocks : Softmax.BlockPartition),
  query.attention key value scale mask weights (Softmax.Schedule.online blocks) =
    query.attention key value scale mask weights
```

With `schedule := .online blocks`, `Tensor.attention` folds the blocks of each score row from a
running maximum, keeping a normalizer and a weighted sum that are rescaled whenever a block raises
the maximum, and divides them at the end. According to the theorem, over the reals the result is
the two-pass attention for every block size, mask and multiplier
{Informal.citep flashattention2022}[], so tile sizes don't change the denotation. That is what
licenses a semantic graph rewrite, and it is what a fused implementation can be asked to refine.
`OnlineSoftmax.onlineAverage_error` bounds the rounding error of the one-query
iterator. No memory-traffic model is defined. An axiom audit of the specification theorem gives:

```lean (name := bsAxioms)
-- Inspect logical dependencies of the specification
-- theorem, not native-kernel evidence.
#print axioms
  Tensor.attention_online_eq
```
```leanOutput bsAxioms (whitespace := lax)
'TorchLean.Tensor.attention_online_eq' depends on
  axioms: [propext, Classical.choice, Quot.sound]
```

Just the three standard Lean axioms, and no `sorryAx`. You'll find the theorem in
{src "TorchLean/Verify/RuntimeApprox/OnlineSoftmax/Real.lean"}[`OnlineSoftmax/Real.lean`].

That theorem is about a Lean specification over the reals. It doesn't compare CPU and CUDA
arithmetic. The LibTorch attention capsule records the runtime guards, regression tests, and
`checked` trust level of the actual kernel, which composes matrix products and a softmax rather
than the online schedule. Connecting PTX or a library call all the way to the specification would
require another refinement argument over Float32, layout, compiler, and hardware behavior, and no
capsule field can stand in for it.

# Forward And Backward Ownership

Inference asks for a forward value. Training asks for more: the value must remain connected to the
derivative rule used by the optimizer.

TorchLean distinguishes three VJP modes:

- `none`: a capsule supplies no reverse rule, and a profile requests forward execution only
- `torchLeanTape`: a capsule composes its local VJP from TorchLean runtime operations
- `backendVJP`: a capsule supplies its local VJP through a native bridge routine.

A profile requesting `torchLeanTape` admits either local VJP implementation while retaining
TorchLean's tape and backward traversal. A profile requesting `backendVJP` specifically requires
the native local rule. Attention's capsule uses `torchLeanTape`: Lean computes its matrix products
and explicit softmax VJP over tape-owned probabilities, then adds the input cotangents to the
surrounding graph's contributions. There is no LibTorch autograd graph to traverse. Reverse-mode
accumulation remains the classical construction {Informal.citep baydin2018}[].

GPU attention uses LibTorch automatically. There's no separate composed-attention provider to
select. Its registered `LibTorch.multiheadAttention` contract records the shape checks, hard-mask
convention, local VJP, and implementation-defined reduction order.

No-grad sessions request `none` automatically. During training, a differentiable operation cannot
select a forward-only capsule. Seeded random sources are the deliberate exception: they create
non-differentiable values, so they do not need a local VJP of their own.

Keep the selected capsule names with an experiment. A preference alone does not identify what
ran, and `libtorch.multihead_attention` leaves the numerical primitives' implementation choices
to the SDK.

# Boolean Attention Masks

TorchLean gives boolean attention masks one semantics across specifications and runtimes. A blocked
entry contributes exactly zero to the softmax numerator, as if its score were negative infinity. On
CUDA, the composition forms zero numerators for blocked entries and handles fully blocked rows
explicitly. Additive score biases remain a separate operation. {ref "modern-models"}[Modern Models]
runs that mask against PyTorch and shows the theorem which makes “exactly zero” exact over the
reals rather than approximate.

Replacing blocked scores by a large negative sentinel and calling an ordinary softmax changes the
function. The original transformer paper writes the mask as $`-\infty` before the softmax
{Informal.citep transformer2017}[], and $`\exp(-\infty)=0` exactly, but a sentinel is a finite
number and $`\exp` of it isn't zero. TorchLean therefore builds the softmax numerators directly: an
allowed entry contributes $`\exp(\text{score}-\text{rowMax})` and a blocked entry contributes
literally `0`. In the three-token causal example below, each row is a query and each column is a
key. A query may attend to its own position and earlier positions, so entries above the diagonal
are blocked:

```lean (name := bsMaskHard)
-- Each row is a query, and the causal mask excludes keys
-- strictly to its right.
def bsScores : Tensor Float [3, 3] :=
  [[1.0, 2.0, 3.0],
   [0.5, 0.0, -0.5],
   [2.0, 2.0, 2.0]]

def bsHard : Tensor Float [3, 3] :=
  Softmax.hardMasked bsScores (Attention.causalMask 3)

#eval bsHard
```
```leanOutput bsMaskHard (whitespace := lax)
[[1.000000, 0.000000, 0.000000], [0.622459, 0.377541, 0.000000], [0.333333, 0.333333, 0.333333]]
```

Every entry above the diagonal prints `0.000000`. In the real specification each nonempty row sums
to one, and the displayed Float entries approximate those normalized weights. For the first query
only one key is allowed, and that key receives all its mass. The second row normalizes its first
two scores, and the third is uniform because all three keys are allowed and their scores are equal.

In row two, the ratio of the allowed weights is $`\exp(0.5)/\exp(0)`, so the first key receives
about `0.622459` and the second `0.377541`. The third key receives no mass even though its score
is a finite number. This is the contract a provider must preserve when it accepts a boolean
mask. Multiplying the resulting weights by value vectors then prevents the blocked value from
contributing through that attention edge. Causality of a whole model also requires checking its
other paths, such as convolutions and any unmasked attention layers.

## Finite Sentinels And Attention Leakage

Run the same scores through an ordinary softmax after substituting a sentinel, and look at the
weight the model gives to a position it isn't allowed to see. We scale the first number by
$`10^{12}` because `Repr Float` prints six decimals and the leak is smaller than that:

```lean (name := bsMaskLeak)
-- Scale the tiny blocked weight so six-decimal formatting
-- cannot hide the leak.
def bsSentinel (m : Float) : Tensor Float [3, 3] :=
  Tensor.softmax
    (Tensor.zipWith (fun s ok => if ok then s else m)
      bsScores (Attention.causalMask 3)) ⟨1, Nat.one_lt_two⟩

#eval (Tensor.get2 (bsSentinel (-30.0)) 0 2 * 1.0e12,
  Tensor.get2 (bsSentinel (-1.0e30)) 0 2,
  Tensor.get2 bsHard 0 2)
```
```leanOutput bsMaskLeak
(0.034425, 0.000000, 0.000000)
```

With a sentinel of $`-30` the future token still receives about $`3.4\times10^{-14}` of the
attention mass. A comparison with a tolerance larger than this leak would miss it. That nonzero
weight can violate exact independence from a future value, so a theorem claiming that independence
would need additional hypotheses or a different masking definition. With a sentinel of $`-10^{30}`
the exponential underflows and the leak becomes exactly zero in this evaluation, but that result
depends on the floating-point format and scores, while the hard-mask definition sets blocked
numerators to zero directly.

## Fully Blocked Rows

A padded sequence can produce a query row whose every key is blocked, and in that case subtracting
the row maximum from equal sentinel values leaves zeros, so their exponentials don't underflow.
Hard masking defines the row to be zero:

```lean (name := bsMaskEmpty)
-- A fully blocked row distinguishes hard masking from equal
-- finite sentinels.
#eval Softmax.hardMasked bsScores
  (Attention.allFalseMask 3 3)

#eval Tensor.softmax
  (Tensor.zipWith (fun s ok => if ok then s else -1.0e30)
    bsScores (Attention.allFalseMask 3 3))
  ⟨1, Nat.one_lt_two⟩
```
```leanOutput bsMaskEmpty (whitespace := lax)
[[0.000000, 0.000000, 0.000000], [0.000000, 0.000000, 0.000000], [0.000000, 0.000000, 0.000000]]
```
```leanOutput bsMaskEmpty (whitespace := lax)
[[0.333333, 0.333333, 0.333333], [0.333333, 0.333333, 0.333333], [0.333333, 0.333333, 0.333333]]
```

The second result distributes attention over three forbidden keys. Every sentinel equals the row
maximum, every shifted exponent is $`\exp(0)=1`, and normalization divides by three. Its weights
are ordinary finite numbers, so a numerical-validity check wouldn't detect the masking error.

The all-false case tests a different part of the definition from the ordinary causal triangle,
because a causal row has at least its diagonal entry available while a fully blocked row has no
valid normalization mass. Defining its result as zero gives the caller a specific behavior for
padding and empty visibility. It also avoids treating three equally forbidden keys as three equally
plausible keys. A parity fixture should include this case because ordinary triangular examples
wouldn't expose the sentinel implementation's uniform-row behavior.

## Exact Zero Over The Reals

All the evaluations above use one score matrix and one dtype, while the general specification
states that every strict-future coordinate of a causal attention matrix is zero, over the reals
where $`\exp` has its usual laws:

```lean (name := bsMaskThm)
-- Instantiate the general future-zero theorem at three
-- tokens and a zero score tensor.
open Softmax in
example (i j : Fin 3) (hij : i.val < j.val) :
    Tensor.get2 (Softmax.hardMasked
      (α := ℝ) 0 (Attention.causalMask 3)) i j = 0 :=
  hardMasked_causalMask_eq_zero 0 i j hij
```

`hardMasked_causalMask_eq_zero` in
{src "TorchLean/NN/Attention/CausalMask.lean"}[`CausalMask.lean`] is stated for every
sequence length, every score tensor, and every strict-future pair, so the instance above needs no
tactic block beyond naming it. Its proof applies the general blocked-coordinate lemma to the causal
mask. That lemma handles the empty-row branch and simplifies the zero numerator through real
division, without relying on an exponential underflowing. A capsule on any provider is then
measured against this specification rather than against another implementation's tolerance.

# Contract Checks

Planning and acceptance are separate steps. Planning finds capsules for graph operations. Auditing
turns their contract fields into obligation reports. Then the evidence check classifies those
reports under an `AssurancePolicy`, and execution acceptance also rechecks the full `KernelPolicy`.

One path runs through the implementation:

1. `Availability` states the devices and providers declared for planning. Eager execution performs
   the separate linked-library and hardware probes before launching a kernel.
2. `Registry` supplies the implementation tables, and `Planner` chooses one `PlannedKernel` for
   each operator among the capsules the tables offer for it. Together these choices form a
   `KernelPlan`. For a graph, `IR.Graph.plan` plans one capsule for the operator of each
   runtime-relevant node, and `Grouping` coalesces adjacent nodes served by the same capsule.
3. `Audit` reads the four contract descriptors of each selected capsule off the plan
   (`PlannedKernel.contracts`) and reports whether any of them is trusted external
   (`KernelPlan.hasTrustedExternal`).
4. `KernelPlan.checkContracts policy` filters the obligations whose evidence the policy does not
   accept. Its result is a `Check`: `accepted`, or `rejected` with the failing
   `ObligationReport`s. The complete acceptance gate also checks policy compatibility. Eager
   execution receives an `AcceptedKernel` for each operation, and graph planning produces an
   `AcceptedGraphKernelPlan` that a graph executor could consume. Nothing in the runtime executes
   from this graph metadata.
5. An eager session runs the selected capsule with the implementation of its provider and device:
   the operator's rule on the CPU tape for a reference CPU capsule, or the operator's
   `LibTorch.Kernel.ofOp` entry on the CUDA tape for a LibTorch CUDA capsule, whose own capsule must
   name the same operator, provider, and device (`KernelCapsule.implements`). For any other
   capsule, execution fails before entering a different provider's code.
6. `Report` renders providers, trust levels, VJP modes, reduction policies, and per-obligation
   evidence for logs and benchmark records.

To isolate the evidence check, make a teaching capsule whose forward-value evidence is an
assumption. Its other descriptors come from the LibTorch capsule of `bsMatmul`. This is a
deliberately constructed diagnostic plan, not a new provider or an executable selection:

```lean (name := bsCheck)
-- Change one evidence field, then hold the plan fixed.
#eval do
  let cuda := LibTorch.implementation
  let some base := cuda.capsule? bsMatmul
    | IO.println "no CUDA capsule"
  let cap : KernelCapsule :=
    { base with
      name := "demo.assumed_matmul"
      valueContract :=
        { base.valueContract with
          evidence :=
            .trustedBoundary "demonstration assumption" } }
  let plan : KernelPlan :=
    { kernels := #[{ op := cap.op, capsule := cap }] }
  let policies : List AssurancePolicy :=
    [AssurancePolicy.checked, AssurancePolicy.external]
  policies.forM fun (policy : AssurancePolicy) => do
    match KernelPlan.checkContracts policy plan with
    | .accepted => IO.println s!"{policy.label}: accepted"
    | .rejected reports =>
      for r in reports do
        IO.println s!"{policy.label}: rejected \
          {r.capsuleName} {r.obligation.label}"
        IO.println s!"  {r.evidence.label}"
```
```leanOutput bsCheck
checked: rejected demo.assumed_matmul value
  trusted boundary: demonstration assumption
external: accepted
```

One obligation out of four fails, and the report names which one: the forward value. Shape and
layout still have runtime guards, and the VJP still names its regression suite. Accepting
`external` permits the demonstration assumption while retaining those other obligations.
The contract of the matmul entry of `LibTorch.Kernel.ofOp` uses test-suite evidence for its forward
value. The example changes that field to expose the policy boundary.

Before it runs an implementation, the eager runtime selects an accepted capsule for each operator
it records and keeps the first acceptance of each capsule name for its report. It then runs the
implementation of the selected provider and device, or fails if none is linked. A reported
selection is therefore not a record of completed calls. Inspection tools can also retain rejected
graph plans and explain why they failed.

`AcceptedKernel` carries a proof that `PlannedKernel.acceptable policy = true`. This gate checks
operation identity, forward support, contract alignment, trust, provider, device, VJP mode, and
evidence. `AcceptedGraphKernelPlan` carries the corresponding result for the groups derived from
its stored graph plan. A diagnostic plan alone does not authorize execution: these accepted values
require the complete gate to succeed.

`AssurancePolicy` has one field, `allowTrustedExternal`. The `checked` policy leaves it false and
admits shared definitions, runtime guards, test suites, and `notApplicable`, while the `external`
policy sets it true and also admits trusted boundaries. That one record decides which capsules the
planner may select (by trust level) and which evidence the selected capsules may rely on (by
evidence kind), and `IR.Graph.accept` uses the profile's configured policy for both steps.
Diagnostic evidence checks, such as `bsCheck`, can deliberately compare policies without producing
an executable accepted value. Provider preference chooses between admissible implementations, but
the evidence check is a separate filter: asking for a provider can't override an obligation's
rejected evidence.

The evidence check accepts exactly when the filtered list of rejected obligations is empty. What it
establishes is that every selected capsule rests on the evidence classes the caller agreed to. It
doesn't establish the numerical correctness of any kernel, and no registered capsule claims a
proof. Its sources are {src "TorchLean/NN/Backend/Audit.lean"}[`Audit.lean`] and
{src "TorchLean/NN/Backend/ContractCheck.lean"}[`Check.lean`].

At the FFI boundary, the maintained CUDA wrappers also perform concrete checks. Convolution and
pooling validate rank and dimension conversion, nonzero strides, representable element counts,
buffer lengths, and operation-specific domains such as finite nonzero smooth-max $`\beta` after
conversion to Float32 (the C++ adapter repeats critical size and overflow checks). These guards
prevent malformed calls, and they complement a mathematical value-refinement argument without
replacing it.

Capsules record reduction order and layout claims but have no scalar-type field. Which arithmetic
and buffer format actually execute is determined by the runtime configuration and native conversion
checks, and a capsule label alone doesn't certify their relationship.

Neither `bsCheck` result reports a numerical comparison: that example only checks which evidence
classes the policy permits.

# Runtime Configuration

The model API stays independent of these implementation details. Every device uses the same
builder, and the device is an ordinary argument:

```lean (name := bsRuntime)
-- Changing the device does not alter the builder's input
-- and output shapes.
def bsModel : nn.Builder (nn.Sequential [4] [1]) :=
  nn.Sequential![nn.linear 4 8, nn.op .relu, nn.linear 8 1]

def bsTrainer (device : Backend.Device) :=
  Trainer.create bsModel
    { objective := mse
      optimizer := optim.adam { lr := 0.01 }
      device := device }

#check bsTrainer cuda
```
```leanOutput bsRuntime (whitespace := lax)
bsTrainer cuda : Trainer [4] [1]
```

On every device the trainer's type is `Trainer [4] [1]`: the device isn't part of the model's
interface, so switching it can't change which inputs the model accepts. It's the same for a bare
runtime configuration:

```lean (name := bsConfig)
-- Projecting a configuration field performs no runtime
-- compatibility check.
#eval ({ device := cuda : Session.Config }).device.cliName
```
```leanOutput bsConfig
"cuda"
```

This block only projects a field from the configuration and doesn't validate or execute it. A build
without LibTorch rejects this request when the session opens.

# Graph Kernel Planning

Everything above planned a list of operators. Graph-level planning (`IR.Graph.plan`) plans
the registered operator of each node and also records source node identities: some nodes need no
kernel, and adjacent selections with the same capsule can share one audit row. Each planned kernel
names its node and the wire record of its operator (printed below by its tag). Lower the model to
the canonical verification IR and inspect its plan. This plan is diagnostic: eager execution
selects an implementation per operator.

```lean (name := bsGraphPlan)
-- Retain source node ids so each selected capsule can be
-- traced to the lowered model.
def bsBuilt : nn.Sequential [4] [1] :=
  nn.build 2026 bsModel

def bsGraph : Except String IR.Graph :=
  (Verification.lowerForwardToIR (α := Float) bsBuilt
    (nn.initialState bsBuilt)).map (·.graph)

#eval show IO Unit from do
  let g ← IO.ofExcept bsGraph
  let plan ← IO.ofExcept
    (g.plan Profile.checkedCPU.policy
      Profile.checkedCPU.availability
      Profile.checkedCPU.implementations)
  IO.println s!"nodes {g.nodes.size}, \
    planned {plan.kernels.size}"
  for k in plan.kernels do
    IO.println s!"  {k.nodeId} {k.op.tag} \
      -> {k.capsule.name}"
```
```leanOutput bsGraphPlan (whitespace := lax)
nodes 18, planned 13
  1 reshape -> reference.reshape
  3 permute -> reference.permute
  4 matmul -> reference.matmul
  6 broadcast -> reference.broadcast
  7 add -> reference.add
  8 reshape -> reference.reshape
  9 relu -> reference.relu
  10 reshape -> reference.reshape
  12 permute -> reference.permute
  13 matmul -> reference.matmul
  15 broadcast -> reference.broadcast
  16 add -> reference.add
  17 reshape -> reference.reshape
```

The plan has thirteen kernels for eighteen nodes. Its five missing nodes are the input and the four
weight and bias constants: `input` and `const` nodes plan nothing (nor does a `detach` node, which
`Graph.check` has checked against its parent). A constant weight tensor is data the executor
already holds and needs no operation kernel.

Every kernel here came from the `reference.*` family, because `Profile.checkedCPU` declares CPU
availability and the reference table is the only one offering CPU capsules. Under a CUDA profile,
the same graph selects LibTorch capsules for the same operations, and the audit rows change
accordingly while the graph does not.

In the node list, a vector linear layer turns into ordinary tensor operations. A reshape introduces
a batch axis, a transpose presents the stored weights in matrix-product orientation, and a
broadcast aligns the bias with the result. The next reshape removes that batch axis. Those
operations are part of the plan even though the user wrote one `nn.linear`. Four constants retain
the model's two weight matrices and two biases. Reading the plan at this level helps locate where a
provider gap arises without changing the model's public shape interface.

## Kernel Groups In Audit Reports

Adjacent nodes that selected the same capsule for the same operation are collected into one
`KernelGroup`. On the graph above it never fires:

```lean (name := bsGroups)
-- Count audit groups separately from the kernels
-- represented by those groups.
#eval show IO Unit from do
  let g ← IO.ofExcept bsGraph
  let plan ← IO.ofExcept
    (g.plan Profile.checkedCPU.policy
      Profile.checkedCPU.availability
      Profile.checkedCPU.implementations)
  let grouped := plan.toCoalescedGroups
  IO.println s!"kernels {plan.kernels.size}, \
    groups {grouped.groups.size}"
  for gr in grouped.groups do
    IO.println s!"  {gr.nodeIds} {gr.op.tag}"
```
```leanOutput bsGroups (whitespace := lax)
kernels 13, groups 13
  #[1] reshape
  #[3] permute
  #[4] matmul
  #[6] broadcast
  #[7] add
  #[8] reshape
  #[9] relu
  #[10] reshape
  #[12] permute
  #[13] matmul
  #[15] broadcast
  #[16] add
  #[17] reshape
```

Thirteen kernels, thirteen groups. Lowering interleaves reshapes and transposes between the
arithmetic, so no two neighbours ever agree on an operation. What grouping exists for is a graph
that repeats one operation, and that's easy enough to write by hand:

```lean (name := bsChain)
-- Consecutive identical ReLU contracts can share an audit
-- row while retaining three nodes.
def bsChain : IR.Graph :=
  { nodes :=
      #[.input 0 (.value [4]),
        .op ⟨_, _, TorchLean.Op.relu (s := [4])⟩ #[⟨0, 0⟩],
        .op ⟨_, _, TorchLean.Op.relu (s := [4])⟩ #[⟨1, 0⟩],
        .op ⟨_, _, TorchLean.Op.relu (s := [4])⟩ #[⟨2, 0⟩]]
    outputs := #[⟨3, 0⟩] }

#eval show IO Unit from do
  let plan ← IO.ofExcept
    (bsChain.plan Profile.checkedCPU.policy
      Profile.checkedCPU.availability
      Profile.checkedCPU.implementations)
  let grouped := plan.toCoalescedGroups
  IO.println s!"kernels {plan.kernels.size}, \
    groups {grouped.groups.size}"
  for gr in grouped.groups do
    IO.println s!"  {gr.nodeIds} {gr.op.tag} \
      {gr.capsule.name}"
```
```leanOutput bsChain (whitespace := lax)
kernels 3, groups 1
  #[1, 2, 3] relu reference.relu
```

Three ReLU nodes, three planned kernels, one audit row. What that row says is that nodes `1`, `2`,
and `3` all rest on the same evidence. The module docstring in
{src "TorchLean/NN/Backend/Grouping.lean"}[`Grouping.lean`] says it first: grouping "does not
translate graph semantics, fuse operations, or claim that a group executes as one kernel launch."
Three ReLU nodes still run as three operations. Fusing them would be a semantic rewrite, and a
semantic rewrite needs a theorem of the kind {name}`Tensor.attention_online_eq` provides, not a
grouping pass.

## Accepted Plans And Policy Checks

Contract checking a graph plan returns the accepted record, including its policy check:

```lean (name := bsAcceptGraph)
-- Only the accepted branch carries evidence of the complete
-- policy gate.
#eval show IO Unit from do
  let g ← IO.ofExcept bsGraph
  match g.accept Profile.checkedCPU with
  | .error e => IO.println s!"planning failed: {e}"
  | .ok (.accepted accepted) =>
    IO.println s!"accepted under \
      {accepted.policy.assurance.label}"
    IO.println s!"{accepted.capsuleNames.size} rows, \
      distinct capsules:"
    for n in accepted.capsuleNames.toList.eraseDups do
      IO.println s!"  {n}"
  | .ok (.rejected _ failures) =>
    for failure in failures do
      IO.println s!"rejected: {failure.message}"
```
```leanOutput bsAcceptGraph (whitespace := lax)
accepted under checked
13 rows, distinct capsules:
  reference.reshape
  reference.permute
  reference.matmul
  reference.broadcast
  reference.add
  reference.relu
```

The `.accepted` constructor holds a proof of `groupedPlan.acceptable policy = true`. `acceptable`
rechecks operation identity, forward support, contract alignment, trust, provider, device, VJP
ownership, and contract evidence for every grouped kernel. A graph executor can therefore require
an `AcceptedGraphKernelPlan` and obtain that check from the argument, but the runtime doesn't
execute from this record: graph planning and auditing are diagnostic, while eager execution selects
and runs an implementation per operation.

The accepted report has thirteen rows but only six distinct capsule names. Reuse of a name means
the same implementation contract serves several nodes, not that those nodes share a result or a
launch. Grouping compares complete capsule records, including evidence and numerical policy, so
identically named capsules with different obligations can't silently share an audit group. Because
the source node identifiers are stored, a diagnostic can connect a rejected contract back to the
particular graph operations that depended on it.

# Backend Evidence And Guarantees

These statements have different strengths:

- "the example ran on CUDA" reports an execution path
- "CUDA matched the CPU reference on this test suite" reports finite parity evidence
- "the fused attention spec equals standard attention" cites a Lean semantic theorem
- "the Lean composition over ATen primitives implements the attention spec" requires an
  implementation refinement argument
- "this LibTorch build passed the attention tests" reports the tested cases and configuration,
  while the foreign implementation remains outside the Lean proof.

A backend report records the selected provider and its declared evidence. Keep it beside a
benchmark to identify the intended implementations, together with execution results, SDK version
and runtime settings. On its own, the report doesn't establish that those implementations ran.

For example, a shape guard cannot justify a claim about attention's numerical result. Follow the
selected capsule's value obligation to its cited test or trusted boundary, then check what that
evidence covers. The retained CUDA comparison of `max` and `min`, for instance, checks their
pullbacks bit for bit at ties, infinities, signed zeros and NaN operands, but their values on
finite nonzero inputs only. At a NaN or a signed zero, the CUDA values rest on the adapter's
construction of the IEEE 754-2019 maximum and minimum, not on that comparison.

# Related Chapters

Read {ref "run-configuration"}[Choosing How A Model Runs] for the runtime API, and
{ref "gpu-and-cuda"}[GPU and CUDA] for the native implementation details.
{ref "runtime-approximation"}[Runtime Approximation] is where reduction order and Float32 rounding
are related back to real-valued specifications. For platform commands and the profiles wired into
the repository, see the [Installation page](https://lean-dojo.github.io/TorchLean/installation/).

# References

The framework whose tensor library supplies CUDA execution is {Informal.citet pytorch2019}[], the
fused attention algorithm is {Informal.citet flashattention2022}[], reverse-mode accumulation is
surveyed by {Informal.citet baydin2018}[], and the proof-carrying-code approach used for comparison
is {Informal.citet necula1997}[].

- PyTorch, [C++ and LibTorch API](https://docs.pytorch.org/cppdocs/), and the
  [`torch.nn.attention`](https://docs.pytorch.org/docs/stable/nn.attention.html) backend selector
  for the SDK permissions exposed by the controls above.
- NVIDIA, [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/).
- Lean, [validating proofs](https://lean-lang.org/doc/reference/latest/ValidatingProofs/), for what
  `#print axioms` is checking.

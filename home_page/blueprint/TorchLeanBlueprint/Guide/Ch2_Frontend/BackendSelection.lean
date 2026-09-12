import VersoManual
import NN.API
import NN.API.Runtime
import NN.API.Verification.Lowering
import NN.IR
import NN.Spec.Layers.FlashAttention
import NN.Proofs.Models.Attention.CausalMask
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open NN.Backend
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Backend Selection" =>
%%%
tag := "backend-selection"
file := "Inside-The-Backend-Planner"
%%%

The previous page selected CPU, CUDA, or an optional provider through the runtime API. A less
visible question remains: when a graph asks for matrix multiplication, attention, or a
reduction, how does TorchLean decide which implementation is allowed to answer?

A device name is not enough. One CUDA build may contain a hand-written kernel, a cuBLAS call, and a
LibTorch bridge for different operations. Their layouts, numerical behavior, backward support, and
supporting evidence differ. The backend planner keeps those differences in data and either returns
an accepted plan or explains why it could not make one.

The path is:

$$`
\text{operation}+\text{profile}+\text{available providers}
\longrightarrow \text{capsule}
\longrightarrow \text{audit}
\longrightarrow \text{contract check}
\longrightarrow \text{accepted kernel}
\longrightarrow \text{typed handler}.
`

The sections below trace that path from an operation request to the handler that executes it. Every
plan, rejection, and report on this page is produced by running the planner during elaboration, so
what you read is what the current code decides.

Device selection first requires a profile. There are nine device constructors, and maintained
profiles exist for two of them:

```lean (name := bsDevices)
-- Inspect maintained policy defaults without probing a
-- device or launching a kernel.
#eval do
  for d in [Device.cpu, .cuda, .rocm, .metal, .tpu] do
    match BackendProfile.maintainedForDevice? d with
    | some p => IO.println s!"{d.cliName}: {p.name}"
    | none =>
      IO.println s!"{d.cliName}: no maintained profile"
```
```leanOutput bsDevices
cpu: checked_cpu
cuda: checked_cuda
rocm: no maintained profile
metal: no maintained profile
tpu: no maintained profile
```

The `Option` result distinguishes a maintained profile from an unsupported request. When the result
is `none`, the caller must supply a profile describing admissible implementations and their
evidence before execution can proceed.

The `cpu` and `cuda` lines report names of maintained policies, not results of hardware probes.
This distinction lets a tool inspect a CUDA plan on a machine that cannot execute it. The three
`none` cases mean no default policy has been supplied for those devices; adding a constructor to
`Device` would not change that result. A custom profile must describe the provider it intends to
use, and the eventual runtime must bind that description to callable implementations. Keeping
planning pure makes missing coverage visible before data transfer or training begins.

# Kernel Capsules

Suppose a graph reaches scaled dot-product attention. TorchLean currently knows three maintained
ways to compute it: a composed TorchLean expression, a native fused CUDA implementation, and a
LibTorch forward bridge with a TorchLean-owned backward pass. The operation is the same; the
implementation contract is not.

A `KernelCapsule` records those differences:

```
-- These fields describe an implementation contract before
-- any handler is called.
structure KernelCapsule where
  name : String
  op : BackendOp
  provider : Provider
  device : Device
  trustLevel : TrustLevel
  supportsForward : Bool
  vjpMode : VJPMode
  shapeContract : ContractDescriptor
  layoutContract : ContractDescriptor
  valueContract : ContractDescriptor
  vjpContract : ContractDescriptor
  numericalPolicy : NumericalPolicy
```

Capsules are declared before the run and registered with the backend. The planner may select one
only when its device, provider, gradient mode, and trust level fit the requested profile. If no
capsule fits, planning stops with an error.

Capsules are collected in named `CapsuleModule`s. Built-in attention, native CUDA, portable
reference, and optional LibTorch code contribute modules to the same registry. A downstream
provider can prepend another module with `BackendProfile.withCapsuleModules`; it does not add a new
model class or a branch to the graph walker. The model still lowers to ordinary `BackendOp`s, and
the planner either finds an admissible capsule for each operation or reports the missing operation.

The registry rejects duplicate module names:

```lean (name := bsModules)
-- Duplicate module identity is an error even if each module
-- contains valid capsules.
#eval Registry.validateModules
  #[Registry.libTorchModule, Registry.libTorchModule]

#eval Registry.validateModules Registry.maintainedModules
```
```leanOutput bsModules
Except.error "duplicate backend capsule module `libtorch`"
```
```leanOutput bsModules
Except.ok ()
```

`BackendOp` names semantic operation families such as matrix multiplication, reduction, pooling, or
convolution. Rank, axes, padding, strides, and index tensors remain in the graph payload. This keeps
capability discovery general without erasing the information needed to state the operation
correctly. The catalog is one flat list; both maintained profiles share it, and the device filter
happens when a plan is made, not when a module is registered.

The sources are {src "NN/Backend/Capsule.lean"}[`Capsule.lean`] for the record and
{src "NN/Backend/Registry.lean"}[`Registry.lean`] for the modules. The installation guide has an
[installation and kernel
overview](https://lean-dojo.github.io/TorchLean/installation/#from-a-model-to-a-kernel).

The first registry output rejects the second occurrence of the name `libtorch`; it is checking
the identity of contributed modules. The second accepts the maintained module collection. Neither
result launches attention or compares two numerical answers. This check matters when an extension
adds capsules: silently registering the same module twice could make catalog order and fallback
hard to understand. Once modules have distinct names, the planner can still consider several
implementations of one semantic operation and select among them using the profile.

# Capsule Contracts And Evidence

Each capsule has four obligations: shape, layout, forward value, and VJP. Each is a
`ContractDescriptor`: a structured `ContractClaim`, a one-line summary for readers, the evidence
behind the claim, and optional provenance (source paths and native symbols). Provenance identifies
the implementation being discussed; it is not evidence of correctness.

The claim is one of shape safety, compatibility with a named tensor layout, refinement of a forward
specification, refinement of a VJP specification, or `vjpUnavailable` for a forward-only capsule.
Before planning, TorchLean checks that each descriptor has the right obligation kind and operation,
and that a VJP descriptor names the capsule's declared VJP mode.

For example, moving a value test into the portable attention capsule's shape field makes the
descriptor disagree with its assigned obligation:

```lean (name := bsMisfiled)
-- Keep the value claim intact but deliberately assign it to
-- the wrong obligation.
def bsMisfiled : KernelCapsule :=
  { Reference.attention with
    name := "demo.misfiled"
    shapeContract :=
      ContractDescriptor.tested
        (.valueRefinement .scaledDotProductAttention)
        "value test filed as shape evidence" "demo suite" }

#eval (Reference.attention.contractsAligned,
  bsMisfiled.contractsAligned)
```
```leanOutput bsMisfiled
(true, false)
```

A misaligned capsule is not merely reported; `KernelCapsule.admissible` requires
`contractsAligned`, so the planner will not select it at all:

```lean (name := bsMisfiledPlan)
-- The malformed descriptor must prevent selection, not
-- merely appear in a warning.
#eval planOp BackendProfile.checkedCpu.policy
    #[bsMisfiled] .scaledDotProductAttention
  |>.map fun k => k.capsule.name
```
```leanOutput bsMisfiledPlan (whitespace := lax)
Except.error "no admissible kernel capsule for op
  scaled_dot_product_attention on device cpu"
```

A value test placed in the shape field is therefore rejected rather than counted as shape evidence,
and a forward-only capsule cannot describe `.none` as though it were a VJP refinement.

`ContractEvidence` classifies the evidence in four ways:

- `runtimeGuard name` records validation performed at an execution boundary;
- `testSuite name` records a regression or differential test suite;
- `trustedBoundary reason` names code whose correctness is assumed for the claim;
- `notApplicable` records an obligation that the capsule intentionally does not provide.

There is no "proved" evidence variant. A capsule cannot claim that a Lean theorem covers its native
implementation, because no such theorem exists for any registered kernel. What a capsule can say is
which guards and tests stand behind it, or that it delegates to code TorchLean does not check. The
related idea of proof-carrying code {Informal.citep necula1997}[] requires a producer to ship a
checkable certificate with its binary. A capsule carries a weaker record: a classified claim and
its evidence source. The checker verifies that classification and the caller's policy, without
verifying the kernel.

`NumericalPolicy` currently has one field,
`reduction : ReductionPolicy`, with values `fixedLeft`, `implementationDefined`, and
`notApplicable`. The portable matrix-product capsule records the fixed left fold used by the tensor
semantics; the CUDA and LibTorch capsules record an implementation-defined reduction. A fixed-left
range trace therefore cannot be reused for a cuBLAS schedule merely because both capsules implement
`matmul`. Rounding mode, subnormal handling, and multiply-add contraction are not recorded in the
capsule, and nothing audits them.

Trust has two levels, `checked` and `trustedExternal`. The maintained profiles accept `checked`
capsules whose evidence is guards and tests. A capsule whose value contract rests on a trusted
boundary is admitted only under the explicit `external` policy, even if the capsule itself is
labelled `checked`: the contract check reads the evidence, not the label.

The pair `(true, false)` isolates a bookkeeping invariant: the original capsule places each
claim in its matching field, while the edited capsule puts a value claim where a shape claim is
required. The subsequent error demonstrates that this mismatch affects selection. It would not
be enough merely to print a warning and then execute the capsule. Conversely, alignment alone
cannot establish that the named tests cover the implementation well. The descriptor separates
what is claimed from what kind of evidence is offered, and the policy checks both at their
respective boundaries.

For a matrix product, layout and numerical policy answer different questions. Layout determines
which stored elements are interpreted as each row and column. Reduction policy determines the
order in which products are added. Two implementations may agree on every address and still
produce different last bits through their summation orders. A shape check can catch incompatible
matrix dimensions but cannot establish either of those numerical relationships. This is why a
capsule carries several obligations instead of one broad correctness label.

# Kernel Selection Plans

Planning is a pure function from a profile and a list of operations to a plan or an error message,
so a plan can be printed. This helper prints one block per selected capsule:

```lean (name := bsPlan)
-- Print the selected implementation and the evidence
-- relevant to its forward values.
def bsRows (profile : BackendProfile)
    (ops : Array BackendOp) : IO Unit := do
  match profile.planOps ops with
  | .error e => IO.println s!"planning failed: {e}"
  | .ok plan =>
    for a in plan.audit.kernels do
      IO.println s!"{a.op.name} -> {a.capsuleName}"
      IO.println s!"  provider {a.provider.label}, \
        trust {a.trustLevel.label}"
      IO.println s!"  reduction \
        {a.numericalPolicy.reduction.label}"
      let value := a.valueContract.evidence.label
      IO.println s!"  value: {value}"

#eval bsRows BackendProfile.checkedCpu #[.matmul, .relu]
```
```leanOutput bsPlan
matmul -> reference.matmul
  provider reference, trust checked
  reduction fixed-left
  value: covered by test suite NN.Tests.Runtime.Floats.Suite
relu -> reference.relu
  provider reference, trust checked
  reduction n/a
  value: covered by test suite NN.Tests.Runtime.Floats.Suite
```

The portable matrix product commits to a fixed left fold, and
`relu` reports `n/a` because a pointwise operation performs no reduction at all. Those are the
labels a range trace or a benchmark record should quote, and
{ref "runtime-approximation"}[Runtime Approximation] explains why a summation order is not a detail.

A missing implementation is an ordinary value, available before any kernel is launched:

```lean (name := bsGap)
-- These family-level requests require dedicated capsules in
-- the selected CPU catalog.
#eval do
  for op in [BackendOp.selectiveScan, BackendOp.fftFno] do
    match BackendProfile.checkedCpu.planOps #[op] with
    | .ok plan => IO.println s!"{plan.capsuleNames}"
    | .error e => IO.println e
```
```leanOutput bsGap
no admissible kernel capsule for op selective_scan on device cpu
no admissible kernel capsule for op fft_fno on device cpu
```

The registered `selective_scan` and `fft_fno` capsules are CUDA-only, so the CPU profile cannot
plan those operation-family requests. Generic differentiable scan and real-FFT reference paths
can instead express their computation through supported ordinary operations. The error identifies
a missing capsule for the requested family, not the absence of every CPU implementation of the
underlying algorithm.

The `reference.matmul` and `reference.relu` rows also explain how to read a successful plan:
the name identifies the implementation family, the provider identifies its owner, and the
value-evidence line names the retained test source. The planner does not run that suite while
printing the report. Its input is the registered evidence record. A benchmark that stores only
`checked_cpu` loses these per-operation choices; storing the selected capsules makes later
changes in preference or registry order inspectable.

## PyTorch Attention Kernel Selection

PyTorch also chooses among several attention kernels, through a context manager that expresses a
preference. The difference is when the choice is visible. Requesting the flash kernel for CPU
tensors succeeds, and the returned tensor carries no record of which kernel produced it:

```
# The returned shape does not reveal which attention
# implementation was selected.
q = torch.zeros(1, 1, 4, 8)
with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
    out = F.scaled_dot_product_attention(q, q, q)
print(tuple(out.shape))
```

```
(1, 1, 4, 8)
```

When no kernel satisfies the request, the call raises an exception and reports the rejected
conditions:

```
# This request changes both the device and dtype and may
# have no eligible kernel.
q64 = torch.zeros(1, 1, 4, 8, device="cuda", dtype=torch.float64)
with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
    F.scaled_dot_product_attention(q64, q64, q64)
```

```
UserWarning: Expected query, key and value to all be of dtype:
  {Half, BFloat16}. Got Query dtype: double, ...
RuntimeError: No available kernel. Aborting execution.
```

These examples expose kernel selection in different forms. PyTorch reports the failure at the
tensor call site, including the unsupported dtype. TorchLean's planner returns a value before
execution that tools can store, compare across builds, or attach to a benchmark. Runtime binding
and hardware checks still follow; a successful plan alone does not establish that a kernel ran.

The successful PyTorch output is only an output shape. It establishes that the call returned a
tensor with the expected batch, head, sequence, and feature dimensions; it does not identify a
kernel from that tensor alone. The failure transcript concerns a different request with CUDA
double-precision inputs. Such retained messages describe the tested build and invocation, not a
universal rule for every PyTorch version. For either system, a useful provider comparison needs
matching dtypes and operation settings in addition to matching output shapes.

# The Attention Specification Theorem

TorchLean's FlashAttention specification gives a precise target for refinement:

```lean (name := bsFlash)
-- Both sides share one scalar context; this statement
-- compares Lean specifications.
#check @Spec.flashAttention_eq_scaledDotProductAttention
```
```leanOutput bsFlash (whitespace := lax)
@Spec.flashAttention_eq_scaledDotProductAttention :
  ∀ {α : Type} [inst : Storage α] [inst_1 : Context α]
    [inst_2 : DecidableRel fun x1 x2 => x1 > x2]
    (config : Spec.FlashAttentionConfig) {nQ nK dModel : ℕ}
    {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (ctx : Spec.AttentionContext α nQ nK dModel h1 h2),
    Spec.flashAttention config ctx =
      Spec.scaledDotProductAttention ctx
```

Both sides are `Spec` functions. The theorem says that the fused *Lean specification* has the same
denotation as TorchLean's standard scaled-dot-product-attention specification, for every element
type with the required operation instances and for every block configuration
{Informal.citep flashattention2022}[]. That is what licenses a semantic graph rewrite, and it is
what a fused implementation can be asked to refine. The Lean definition computes the standard
attention stages and ignores the tile sizes; this theorem does not verify an online tiled
algorithm or its memory traffic. Its axiom audit is:

```lean (name := bsAxioms)
-- Inspect logical dependencies of the specification
-- theorem, not native-kernel evidence.
#print axioms
  Spec.flashAttention_eq_scaledDotProductAttention
```
```leanOutput bsAxioms (whitespace := lax)
'Spec.flashAttention_eq_scaledDotProductAttention' depends on
  axioms: [propext, Classical.choice, Quot.sound]
```

The three standard Lean axioms, and no `sorryAx`. The companion theorem
`flashAttentionBackward_eq_scaledDotProductAttentionBackward` has the same axiom set for the
backward pass; both live in {src "NN/Spec/Layers/FlashAttention.lean"}[`FlashAttention.lean`].

Those theorems compare two Lean specifications. The native CUDA capsule records the runtime
guards, regression tests, source provenance, and `checked` trust level of the actual kernel.
Connecting PTX or a library call all the way to the specification would require another refinement
argument over Float32, layout, compiler, and hardware behavior, and no capsule field can stand in
for it.

Read the theorem's `Context α` as the operations used to give both sides their meaning. The
statement quantifies one context shared by the two expressions; it does not compare CPU
arithmetic with CUDA arithmetic. The nonzero query and key dimensions are part of the attention
context's contract. The axiom output reports logical dependencies of this Lean theorem and says
nothing about a compiled CUDA binary. It is therefore useful evidence about the specification
rewrite, while the capsule audit remains the place to inspect the implementation boundary.

# Forward And Backward Ownership

Inference asks for a forward value. Training asks for more: the value must remain connected to the
derivative rule used by the optimizer.

TorchLean distinguishes three VJP modes:

- `none`: no gradient is requested;
- `torchLeanTape`: TorchLean owns the tape and backward traversal; each capsule declares whether its
  local VJP is expressed through TorchLean operations or a named backend kernel;
- `backendVJP`: require capsules whose local VJP is computed by a backend kernel.

The preferred external-forward design is therefore precise: a provider may compute a fast forward
value, TorchLean records the same operation on its tape, and TorchLean applies the backward rule.
This requires enough forward information to be retained for that rule. If the bridge cannot provide
it, the implementation must fall back or expose a larger trust boundary. Reverse-mode accumulation
itself is the classical construction {Informal.citep baydin2018}[]; what is being negotiated here is
only who owns each local rule.

The maintained LibTorch-forward profile implements this design for scaled-dot-product attention.
Selection is per operation. The following profiles differ in provider preference and assurance:

```lean (name := bsPrefer)
-- Change assurance while preserving preference to expose
-- the admissibility filter.
def bsStrictLibTorch : BackendProfile :=
  { BackendProfile.libTorchForwardCuda with
    name := "libtorch_forward_strict"
    policy :=
      { BackendProfile.libTorchForwardCuda.policy with
        assurance := .checked } }

#eval do
  let attention := #[BackendOp.scaledDotProductAttention]
  let profiles : List BackendProfile :=
    [BackendProfile.checkedCuda,
      BackendProfile.libTorchForwardCuda, bsStrictLibTorch]
  profiles.forM fun (p : BackendProfile) => do
    match p.planOps attention with
    | .error e => IO.println s!"{p.name}: {e}"
    | .ok plan =>
      IO.println s!"{p.name}: {plan.capsuleNames}"
```
```leanOutput bsPrefer
checked_cuda: #[torchlean.composed_attention]
libtorch_forward_cuda: #[libtorch.sdpa_forward]
libtorch_forward_strict: #[native_cuda.direct_attention]
```

The third profile asks for LibTorch by preference but keeps the `checked`
assurance policy, so the LibTorch capsule is not admissible at all, and `chooseCapsuleFor?` falls
back to the first admissible capsule in catalog order. Preference is a request, and the trust level
is a filter that a request cannot override. Fallback follows catalog order. A profile that requires
a particular provider must use `only` rather than a preference.

No-grad sessions request `none` automatically. During training, a differentiable operation cannot
select a forward-only capsule. Seeded random sources are the deliberate exception: they create
non-differentiable values, so they do not need a local VJP of their own.

The three selected capsule names make the interaction between preference and assurance visible.
`checked_cuda` prefers the composed TorchLean route for this request. The external profile admits
the LibTorch forward boundary and selects it. Tightening assurance while retaining that preference
forces selection to another admissible implementation. A preference therefore cannot be used as
a measurement label without checking the result: a run requesting LibTorch may legally select a
different provider. If the experiment requires LibTorch specifically, provider restriction and
failure are more informative than fallback.

# Boolean Attention Masks

TorchLean gives boolean attention masks one semantics across specifications and runtimes. A blocked
entry contributes exactly zero to the softmax numerator, as if its score were negative infinity.
Native CUDA skips blocked entries, while the LibTorch bridge passes a boolean mask directly to
scaled dot-product attention. Additive score biases remain a separate operation.
{ref "modern-models"}[Modern Models] runs that mask against PyTorch and shows the theorem which
makes “exactly zero” exact over the reals rather than approximate.

Replacing blocked scores by a large negative sentinel and calling an ordinary softmax changes
the function. The original transformer paper writes the mask as $`-\infty` before the
softmax {Informal.citep transformer2017}[], and $`\exp(-\infty)=0` exactly; a sentinel is a finite
number and $`\exp` of it is not zero. TorchLean therefore builds the softmax numerators directly:
an allowed entry contributes $`\exp(\text{score}-\text{rowMax})` and a blocked entry contributes
literally `0`. In the three-token causal example below, each row is a query and each column is
a key. A query may attend to its own position and earlier positions, so entries above the diagonal
are blocked:

```lean (name := bsMaskHard)
-- Each row is a query; the causal mask excludes keys
-- strictly to its right.
def bsScores : Tensor Float [3, 3] :=
  [[1.0, 2.0, 3.0],
   [0.5, 0.0, -0.5],
   [2.0, 2.0, 2.0]]

def bsHard : Tensor Float [3, 3] :=
  Spec.hardMaskedSoftmaxSpec bsScores (Spec.causalMask 3)

#eval bsHard
```
```leanOutput bsMaskHard (whitespace := lax)
[[1.000000, 0.000000, 0.000000], [0.622459, 0.377541, 0.000000], [0.333333, 0.333333, 0.333333]]
```

Every entry above the diagonal is `0.000000` and every row still sums to one. The first query has
only one allowed key, which receives all its mass. The second row normalizes its first two scores.
The third row is uniform because all three keys are allowed and their scores are equal.

In row two, the ratio of the allowed weights is $`\exp(0.5)/\exp(0)`, so the first key receives
about `0.622459` and the second `0.377541`. The third key receives no mass even though its score
is a finite number. This is the contract a provider must preserve when it accepts a boolean
mask. Multiplying the resulting weights by value vectors then prevents the blocked value from
contributing through that attention edge. Causality of a whole model also requires checking its
other paths, such as convolutions and any unmasked attention layers.

## Finite Sentinels And Attention Leakage

Now run the same scores through an ordinary softmax after substituting a sentinel, and look at the
weight the model gives to a position it is not allowed to see. The first number is scaled by
$`10^{12}` because `Repr Float` prints six decimals and the leak is smaller than that:

```lean (name := bsMaskLeak)
-- Scale the tiny blocked weight so six-decimal formatting
-- cannot hide the leak.
def bsSentinel (m : Float) : Tensor Float [3, 3] :=
  Activation.softmaxSpec 1
    (Tensor.map2Spec (fun s ok => if ok then s else m)
      bsScores (Spec.causalMask 3))

#eval (Spec.get2 (bsSentinel (-30.0)) 0 2 * 1.0e12,
  Spec.get2 (bsSentinel (-1.0e30)) 0 2,
  Spec.get2 bsHard 0 2)
```
```leanOutput bsMaskLeak
(0.034425, 0.000000, 0.000000)
```

With a sentinel of $`-30` the future token still receives about $`3.4\times10^{-14}` of the
attention mass. A comparison with a tolerance larger than this leak would miss it. The nonzero
weight can violate exact independence from a future value; a theorem claiming that independence
would need additional hypotheses or a different masking definition. With a sentinel of $`-10^{30}`
the exponential underflows and the leak becomes
exactly zero in this evaluation. That result depends on the floating-point format and scores;
the hard-mask definition sets blocked numerators to zero directly.

## Fully Blocked Rows

A padded sequence can produce a query row whose every key is blocked. In that case, subtracting
the row maximum from equal sentinel values leaves zeros, so their exponentials do not underflow.
Hard masking defines the row to be zero:

```lean (name := bsMaskEmpty)
-- A fully blocked row distinguishes hard masking from equal
-- finite sentinels.
#eval Spec.hardMaskedSoftmaxSpec bsScores
  (Spec.allFalseMask 3 3)

#eval Activation.softmaxSpec 1
  (Tensor.map2Spec (fun s ok => if ok then s else -1.0e30)
    bsScores (Spec.allFalseMask 3 3))
```
```leanOutput bsMaskEmpty (whitespace := lax)
[[0.000000, 0.000000, 0.000000], [0.000000, 0.000000, 0.000000], [0.000000, 0.000000, 0.000000]]
```
```leanOutput bsMaskEmpty (whitespace := lax)
[[0.333333, 0.333333, 0.333333], [0.333333, 0.333333, 0.333333], [0.333333, 0.333333, 0.333333]]
```

The second result distributes attention over three forbidden keys. Every sentinel equals the row
maximum, every shifted exponent is $`\exp(0)=1`, and normalization divides by three. The result
contains ordinary finite weights, so a numerical-validity check would not detect the masking error.

The all-false case tests a different part of the definition from the ordinary causal triangle.
A causal row has at least its diagonal entry available, while a fully blocked row has no valid
normalization mass. Defining its result as zero gives the caller a specific behavior for padding
and empty visibility. It also avoids treating three equally forbidden keys as three equally
plausible keys. A parity fixture should include this case because ordinary triangular examples
would not expose the sentinel implementation's uniform-row behavior.

## Exact Zero Over The Reals

The evaluations above use one score matrix and one dtype. The general specification states that
every strict-future coordinate of a causal attention matrix is zero, over the reals where
$`\exp` has its usual laws:

```lean (name := bsMaskThm)
-- Instantiate the general future-zero theorem at three
-- tokens and a zero score tensor.
open NN.Proofs.Models.Attention in
example (i j : Fin 3) (hij : i.val < j.val) :
    Spec.get2 (Spec.hardMaskedSoftmaxSpec
      (α := ℝ) 0 (Spec.causalMask 3)) i j = 0 :=
  hardMaskedSoftmaxSpec_causal_future_zero 0 i j hij
```

`hardMaskedSoftmaxSpec_causal_future_zero` in
{src "NN/Proofs/Models/Attention/CausalMask.lean"}[`CausalMask.lean`] is stated for every sequence
length, every score tensor, and every strict-future pair, so the instance above needs no tactic
block beyond naming it. Its proof is short for a reason: with the numerator defined as `0` rather
than as $`\exp` of a sentinel, the blocked coordinate reduces to zero definitionally, and the
lemma is one `unfold` and one `simp`. A
capsule on any provider is then measured against this specification rather than against another
implementation's tolerance.

# Contract Checks

Planning and acceptance are separate steps. Planning finds capsules for graph operations. Auditing
turns their contract fields into obligation reports. The evidence check classifies those reports
under an `AssurancePolicy`; execution acceptance also rechecks the full `KernelPolicy`.

The implementation follows one explicit path:

1. `Availability` states the devices and providers declared for planning. Eager execution performs
   the separate linked-library and hardware probes before launching a kernel.
2. `Registry` supplies the capsule modules, and `Planner` chooses one `PlannedKernel` for each
   operation. Together these choices form a `KernelPlan`; for a graph, `NN.Backend.IR` plans one
   capsule per runtime-relevant node and `Grouping` coalesces adjacent nodes served by the same
   capsule.
3. `Audit` turns the plan into a `KernelPlanAudit`: one `KernelAudit` per selected capsule with
   its four contract descriptors and reduction policy.
4. `KernelPlanAudit.checkContracts policy` filters the obligations whose evidence the policy does
   not accept. Its result is a `ContractCheck`: `accepted`, or `rejected` with the failing
   `ObligationReport`s. The complete acceptance gate also checks policy compatibility. Eager
   execution receives an `AcceptedKernel` for each operation; graph
   planning produces an `AcceptedGraphKernelPlan` that a later graph executor could consume. The
   current runtime does not execute from this graph metadata.
5. The eager session binds the selected capsule to a `KernelHandler` with the same operation,
   provider, and device through `KernelCapsule.bind`. If this build has no such handler, execution
   fails before entering a different provider's code.
6. `Report` renders providers, trust levels, VJP modes, reduction policies, and per-obligation
   evidence for logs and benchmark records.

The LibTorch-forward plan illustrates the evidence check. Holding the selected capsule fixed,
check its obligations under both assurance policies:

```lean (name := bsCheck)
-- Hold the selected capsule fixed and inspect which
-- evidence each policy accepts.
#eval do
  let attention := #[BackendOp.scaledDotProductAttention]
  let profile := BackendProfile.libTorchForwardCuda
  match profile.planOps attention with
  | .error e => IO.println s!"planning failed: {e}"
  | .ok plan =>
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
checked: rejected libtorch.sdpa_forward value
  trusted boundary: LibTorch/CUDA SDPA implementation
external: accepted
```

One obligation out of four fails, and the report names which one: the forward value. The capsule's
shape and layout obligations are runtime guards, its VJP obligation is TorchLean's own rule, and
only the forward value relies on the named LibTorch boundary. Accepting `external` permits that
assumption while retaining the other recorded obligations.

These are Lean data structures rather than an informal convention between command-line flags. The
eager runtime consumes the accepted per-operation value, binds it to the implementation it will
call, and records the capsule it actually used. Inspection tools can retain rejected graph plans
and explain why they failed.

`AcceptedKernel` carries a proof that `PlannedKernel.acceptable policy = true`. This gate checks
operation identity, forward support, contract alignment, trust, provider, device, VJP mode, and
evidence. `AcceptedGraphKernelPlan` carries the corresponding result for the groups derived from
its stored graph plan. A diagnostic plan alone does not authorize execution: these accepted values
require the complete gate to succeed.

`AssurancePolicy` has one field, `allowTrustedExternal`. The `checked` policy leaves it false and
admits runtime guards, test suites, and `notApplicable`; the `external` policy sets it true and
also admits trusted boundaries. The same record decides which capsules the planner may select (by
trust level) and which evidence the selected capsules may rely on (by evidence kind). A profile's
`acceptGraph` uses its configured policy for both steps. Diagnostic evidence checks, such as
`bsCheck`, can deliberately compare policies without producing an executable accepted value.
The `bsPrefer` block above shows both halves
of that sentence at once: under `checked` the LibTorch capsule was never selected, and under
`external` it was selected and then accepted.

The evidence check accepts exactly when the filtered
list of rejected obligations is empty. What it establishes is that every selected capsule rests on
the evidence classes the caller agreed to. It does not establish the numerical correctness of any
kernel, and no registered capsule claims a proof. The sources are
{src "NN/Backend/Audit.lean"}[`Audit.lean`] and
{src "NN/Backend/ContractCheck.lean"}[`ContractCheck.lean`].

The maintained CUDA wrappers also perform concrete checks at the FFI boundary. Convolution and
pooling validate rank and dimension conversion, nonzero strides, representable element counts,
buffer lengths, and operation-specific domains such as finite nonzero smooth-max $`\beta` after
conversion to Float32. The C/CUDA boundary repeats critical size and overflow checks. These guards
prevent malformed launches; they complement rather than replace a mathematical value-refinement
argument.

Capsules record reduction order and layout claims but have no scalar-type field. The runtime
configuration and native conversion checks determine which arithmetic and buffer format actually
execute; a capsule label alone does not certify their relationship.

The rejected value obligation does not say that the LibTorch answer was numerically wrong.
It says that the chosen policy does not permit that obligation to rest on an external assumption.
Likewise, `external: accepted` does not report a numerical comparison; it records agreement
between the evidence class and the caller's policy. This gives an application a precise way to
allow an external forward implementation while continuing to require shape guards and a
TorchLean-owned reverse path. A single unchecked/checked switch for the whole application would
hide which part of the computation actually relies on that assumption.

# Runtime Configuration

The model API stays independent of these implementation details. The same builder is used for both
execution modes, and the mode is an ordinary argument:

```lean (name := bsRuntime)
-- Changing execution does not alter the builder's input and
-- output shapes.
def bsModel : nn.Builder (nn.Sequential [4] [1]) :=
  nn.Sequential![nn.linear 4 8, nn.relu, nn.linear 8 1]

def bsTrainer (execution : Runtime.ExecutionMode) :=
  Trainer.new bsModel
    { objective := .meanSquaredError
      optimizer := optim.adam { learningRate := 0.01 }
      execution := execution }

#check bsTrainer .eager
```
```leanOutput bsRuntime (whitespace := lax)
bsTrainer Runtime.Autograd.Torch.ExecutionMode.eager :
  Trainer [4] [1]
```

The trainer's type is `Trainer [4] [1]` either way: the execution mode is not part of the model's
interface, so switching it cannot change which inputs the model accepts. Device selection lives in
the same configuration record:

```lean (name := bsConfig)
-- Projecting a configuration field performs no runtime
-- compatibility check.
#eval ({ execution := .typedGraph, device := .cuda :
  Runtime.Config }).device.cliName
```
```leanOutput bsConfig
"cuda"
```

This block only projects a field from the configuration; it does not validate or execute it.
The current typed graph trainer rejects this CUDA combination when the session opens.

Typed graph execution records and reuses a typed SSA graph. The execution chapter describes this
lowering and its proof boundary; backend selection here determines which registered implementation
may execute each operation. Graph construction alone is not an optimizing compiler and does not
prove the stored derivative rules correct, which is why the two chapters stay separate.

# Graph Kernel Planning

Everything above planned a list of operations. Graph-level planning additionally records source
node identities: some nodes need no kernel, and adjacent selections with the same capsule can share
one audit row. Lower the model to the canonical verification IR and inspect its plan. This is a
diagnostic graph plan; eager execution selects handlers per operation.

```lean (name := bsGraphPlan)
-- Retain source node ids so each selected capsule can be
-- traced to the lowered model.
def bsBuilt : nn.Sequential [4] [1] :=
  nn.build 2026 bsModel

def bsGraph : Except String NN.IR.Graph :=
  (Verification.lowerForwardToIR (α := Float) bsBuilt
    (nn.initialState bsBuilt)).map (·.graph)

#eval show IO Unit from do
  let g ← IO.ofExcept bsGraph
  let plan ← IO.ofExcept
    (BackendProfile.checkedCpu.planGraphNodes g)
  IO.println s!"nodes {g.nodes.size}, \
    planned {plan.kernels.size}"
  for k in plan.kernels do
    IO.println s!"  {k.nodeId} {repr k.kind} \
      -> {k.capsule.name}"
```
```leanOutput bsGraphPlan (whitespace := lax)
nodes 18, planned 13
  1 NN.IR.OpKind.reshape [4] [1, 4] -> reference.reshape
  3 NN.IR.OpKind.transpose 0 1 -> reference.permute
  4 NN.IR.OpKind.matmul -> reference.matmul
  6 NN.IR.OpKind.broadcastTo [8] [1, 8] -> reference.broadcast
  7 NN.IR.OpKind.add -> reference.add
  8 NN.IR.OpKind.reshape [1, 8] [8] -> reference.reshape
  9 NN.IR.OpKind.relu -> reference.relu
  10 NN.IR.OpKind.reshape [8] [1, 8] -> reference.reshape
  12 NN.IR.OpKind.transpose 0 1 -> reference.permute
  13 NN.IR.OpKind.matmul -> reference.matmul
  15 NN.IR.OpKind.broadcastTo [1] [1, 1] -> reference.broadcast
  16 NN.IR.OpKind.add -> reference.add
  17 NN.IR.OpKind.reshape [1, 1] [1] -> reference.reshape
```

The plan has thirteen kernels for eighteen nodes. The five missing
ones are the input and the four weight and bias constants, because `NN.Backend.IR.op?` maps
`input`, `const`, and `detach` to `none`. A constant
weight tensor is data the executor already holds and needs no operation kernel.

Every kernel here came from the `reference.*` family, because `checkedCpu` declares CPU
availability and the portable module is the only one offering CPU capsules. The same graph under a
CUDA profile selects native capsules for the same operations, and the audit rows change accordingly
while the graph does not.

The node list exposes how a vector linear layer becomes ordinary tensor operations. A reshape
introduces a batch axis, a transpose presents the stored weights in matrix-product orientation,
and a broadcast aligns the bias with the result. The next reshape removes that batch axis.
Those operations are part of the plan even though the user wrote one `nn.linear`. The four
constants retain the model's two weight matrices and two biases. Reading the plan at this level
helps locate where a provider gap arises without changing the model's public shape interface.

## Kernel Groups In Audit Reports

Adjacent nodes that selected the same capsule for the same operation are collected into one
`KernelGroup`. On the graph above it never fires:

```lean (name := bsGroups)
-- Count audit groups separately from the kernels
-- represented by those groups.
#eval show IO Unit from do
  let g ← IO.ofExcept bsGraph
  let plan ← IO.ofExcept
    (BackendProfile.checkedCpu.planGraphNodes g)
  let grouped := plan.toCoalescedGroups
  IO.println s!"kernels {plan.kernels.size}, \
    groups {grouped.groups.size}"
  for gr in grouped.groups do
    IO.println s!"  {gr.nodeIds} {gr.op.name}"
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
arithmetic, so no two neighbours ever agree on an operation. The case grouping exists for is the
graph that repeats one operation, which is easy enough to write by hand:

```lean (name := bsChain)
-- Consecutive identical ReLU contracts can share an audit
-- row while retaining three nodes.
def bsChain : NN.IR.Graph :=
  { nodes :=
      #[{ id := 0, parents := #[], kind := .input
          outShape := [4] },
        { id := 1, parents := #[0], kind := .relu
          outShape := [4] },
        { id := 2, parents := #[1], kind := .relu
          outShape := [4] },
        { id := 3, parents := #[2], kind := .relu
          outShape := [4] }] }

#eval show IO Unit from do
  let plan ← IO.ofExcept
    (BackendProfile.checkedCpu.planGraphNodes bsChain)
  let grouped := plan.toCoalescedGroups
  IO.println s!"kernels {plan.kernels.size}, \
    groups {grouped.groups.size}"
  for gr in grouped.groups do
    IO.println s!"  {gr.nodeIds} {gr.op.name} \
      {gr.capsule.name}"
```
```leanOutput bsChain (whitespace := lax)
kernels 3, groups 1
  #[1, 2, 3] relu reference.relu
```

Three ReLU nodes, three planned kernels, one audit row. What that row says is that nodes `1`, `2`,
and `3` all rest on the same evidence. The module docstring in
{src "NN/Backend/Grouping.lean"}[`Grouping.lean`] says it first: grouping "does not translate graph
semantics, fuse operations, or claim that a group executes as one kernel launch." Three ReLU nodes
still run as three operations. Fusing them would be a semantic rewrite, and a semantic rewrite needs
a theorem of the kind {name}`Spec.flashAttention_eq_scaledDotProductAttention` provides, not a
grouping pass.

## Accepted Plans And Policy Checks

Contract checking a graph plan returns the accepted record, including its policy check:

```lean (name := bsAcceptGraph)
-- Only the accepted branch carries evidence of the complete
-- policy gate.
#eval show IO Unit from do
  let g ← IO.ofExcept bsGraph
  match BackendProfile.checkedCpu.acceptGraph g with
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

The `.accepted` constructor holds a proof of `groupedPlan.acceptable policy = true`.
`acceptable` rechecks operation identity, forward support, contract alignment, trust, provider,
device, VJP ownership, and contract evidence for every grouped kernel. A later graph executor can
therefore require an `AcceptedGraphKernelPlan` and obtain that check from the argument. The
current runtime does not execute from this record: graph planning and auditing are diagnostic,
while eager execution selects and binds handlers per operation.

The accepted report has thirteen rows but only six distinct capsule names. Reuse of a name
means the same implementation contract serves several nodes; it does not mean those nodes share
a result or a launch. Grouping compares complete capsule records, including evidence and
numerical policy, so identically named capsules with different obligations cannot silently share
an audit group. The stored source node identifiers let a diagnostic connect a rejected contract
back to the particular graph operations that depended on it.

# Backend Evidence And Guarantees

These statements have different strengths:

- "the example ran on CUDA" reports an execution path;
- "CUDA matched the CPU reference on this test suite" reports finite parity evidence;
- "the fused attention spec equals standard attention" cites a Lean semantic theorem;
- "the native attention kernel implements the fused spec" requires a native refinement argument;
- "the LibTorch result is correct" depends on the explicitly named LibTorch boundary unless a
  stronger checker or theorem covers it.

A backend report records the selected
provider and the evidence attached to it. Keeping that report beside a benchmark makes “CUDA”
concrete: readers can see which operations were native or external and which guards and tests stand
behind each one.

Assessing a claim about a run requires identifying:

1. which capsule answered the operation, and under which profile;
2. which of the four obligations the claim depends on;
3. what the evidence label for that obligation actually says.

These answers identify the implementation and the scope of its recorded evidence. Assessing the
claim still requires checking that the cited guard, test, or trusted boundary supports it.

# Related Chapters

Read {ref "execution-modes"}[Execution Modes] for the runtime API, and
{ref "gpu-and-cuda"}[GPU and CUDA] for the native implementation
details. {ref "runtime-approximation"}[Runtime Approximation] is where reduction order and
Float32 rounding are related back to real-valued specifications. The
[Installation page](https://lean-dojo.github.io/TorchLean/installation/) lists platform commands and
the profiles currently wired into the repository.

# References

The framework whose kernel-selection behavior we compare against is
{Informal.citet pytorch2019}[]; the fused attention algorithm is
{Informal.citet flashattention2022}[]; reverse-mode accumulation is surveyed by
{Informal.citet baydin2018}[]; and the proof-carrying-code approach used for comparison is
{Informal.citet necula1997}[].

- PyTorch,
  [`torch.compile` reference](https://docs.pytorch.org/docs/stable/generated/torch.compile.html).
- PyTorch, [C++ and LibTorch API](https://docs.pytorch.org/cppdocs/), and the
  [`torch.nn.attention`](https://docs.pytorch.org/docs/stable/nn.attention.html) backend selector
  used in the transcripts above.
- NVIDIA, [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/).
- Lean, [validating proofs](https://lean-lang.org/doc/reference/latest/ValidatingProofs/), for what
  `#print axioms` is checking.

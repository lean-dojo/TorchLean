---
title: Updates
---

<nav class="timeline-nav" aria-label="TorchLean update timeline">
  <a href="#september-2026-libtorch">LibTorch backend</a>
  <a href="#september-2026-floatlib">FloatLib and precision</a>
  <a href="#september-2026-proof-refactor">Proof refactor</a>
  <a href="#august-2026-tensor-overhaul">Tensor overhaul</a>
  <a href="#august-2026-lean-433">Lean 4.33</a>
  <a href="#august-2026-autograd-cuda">August 2026</a>
  <a href="#july-2026-refactor">July 2026</a>
  <a href="#june-2026-reliability">June 2026 reliability</a>
  <a href="#june-2026-lean-431">Lean 4.31</a>
  <a href="#may-2026-cleanup">Comment cleanup</a>
  <a href="#may-2026-runtime-api">Runtime API</a>
  <a href="#may-2026-cuda-stability">CUDA stability</a>
  <a href="#may-2026-data-note">Data note</a>
  <a href="#may-2026-release">Initial release</a>
</nav>

<div class="updates-timeline">

<article class="update-card" id="september-2026-libtorch" markdown="1">
  <div class="update-date">September 2026</div>
  <div class="update-body" markdown="1">

## Running CUDA Through LibTorch

The CUDA backend now calls ATen through a LibTorch SDK instead of TorchLean's own kernels. The
Lean side did not change: the same buffer API, the same extern symbols, and the same runtime tape
that owns differentiation. What changed is the code behind those symbols. About 7,000 lines of
hand-written CUDA (elementwise, reduction, convolution, pooling, tensor, attention and DGEMM
kernels) are gone, replaced by six C++ files under `csrc/libtorch/` that call the corresponding
ATen operations under a no-grad guard. The per-kernel CPU stub files went with them. A build
without LibTorch now links one file, `unavailable.c`, which exports every symbol and fails each
call with a message that says how to rebuild.

Select the SDK with `-Klibtorch_home=PATH` or `TORCHLEAN_LIBTORCH_HOME`; a CUDA-enabled pip
PyTorch installation works as the SDK root. This tree was tested locally against pip torch
2.13.0+cu130 with CUDA 13.0 on A100, and previously against a PyTorch 2.12 nightly. The attention
path uses internal ATen entry points, so other SDK versions should be treated as untested until
the CUDA suite passes on them. The default CPU build needs no SDK or toolkit.

Several native features were removed rather than ported:

| Removed | Use instead |
| --- | --- |
| `Buffer.setDeterministicReductions` | `LibTorch.setDeterministic` |
| `TORCHLEAN_CUDA_CACHE_CAP_BYTES`, `AllocatorStats.cacheBytes` and `cacheCapBytes` | `AllocatorStats.allocatedBytes` and `reservedBytes`, with `LibTorch.setMemoryFraction` and `LibTorch.emptyCache`. The limit is now a fraction of device memory, not a byte cap. |
| `flashAttentionFwd`/`Bwd`, `broadcastRowToRows`, `gatherVec`, `reduceSumByColumn` | The corresponding ATen operations behind the existing buffer API |
| The three fused FNO backward externs | One spectral convolution backward call |
| `matmulCublas`, `matmulCublas32`, `matmulCublas64`, `CublasPrecision` | Ordinary `matmul`; LibTorch chooses the BLAS call |
| `nn.geluTanh` | `nn.gelu`, which is the same tanh approximation |

The fused LayerNorm and GELU kernels and the transpose-free matrix backward described in the August
notes below are part of what was removed. ATen supplies its own implementations. We have not
measured the end-to-end effect on training speed, so this entry makes no performance claim either
way.

Convolution layers built through GraphSpec now start from different initial weights. They take
their initialization seed from the layer's position, the same way linear layers do, where they
used to take twice that value. A model with a convolution therefore trains from a different
starting point than before, even with the same seed. Linear layers and the zero or one
initializations of biases and normalization parameters are unchanged.

  </div>
</article>

<article class="update-card" id="september-2026-floatlib" markdown="1">
  <div class="update-date">September 2026</div>
  <div class="update-body" markdown="1">

## Choosing Scalar Precision with FloatLib

The checkout selects Lean 4.34.0 and imports its reusable scalar arithmetic and generic
rounding theory from a pinned FloatLib dependency. Choose a FloatLib binary format, including
custom precision, directly in typed tensors and models. `ExecFloat.Binary 8 23` gives the familiar
binary32 format; `ExecFloat.Binary 15 112` gives binary128. Valid custom widths use the same API.

A wider type matters only if the extra digits reach the computation. For example, binary128
can retain $1+2^{-100}$, while binary64 rounds it to 1. The
[tensor chapter]({{ '/blueprint/Building-Models/Tensors-That-Remember-Their-Shapes/' | relative_url }})
constructs that parameter directly from a rational, keeps it through a typed affine model,
and compares its forward value, input derivatives and `nn.sgdStep` update with an elementary
calculation. This uses software arithmetic on the typed CPU path. The supervised trainer retains
`Float` data, report and checkpoint boundaries; its `.ieee` option remains fixed to binary32.
The eager LibTorch CUDA backend uses binary32 buffers; the separate matrix-product interface uses
binary64. Configurable FloatLib precision runs on the typed CPU path.

The numerical proofs distinguish rounded trees from exact accumulation followed by one final
rounding. Their intermediate values can differ, so an error theorem for one does not justify the
other. Native Float32 import/export and arithmetic proofs now come directly from FloatLib:
addition and subtraction require finite operands, while square root covers every input through
NaN-canonicalizing native export. The old local bridge files have been removed. These logical
results remain separate from compiler and hardware conformance.

We also moved the shared interval, affine-quantization, and reduction-tree mathematics into
FloatLib, then replaced TorchLean's copies with imports and adapters. Users can choose the grouping
and input order of a reduction tree, use a real quantization scale with their own rounding rule,
or construct interval bounds with binary, decimal, or posit endpoints. TorchLean keeps the tensor
lifts and backend contracts. Its numerical graph certificates still use binary32 endpoints; the
broader scalar interval API does not change that certificate format.

The [installation page]({{ '/installation/' | relative_url }}) records the current dependency pin,
and the [floating-point chapter]({{ '/blueprint/Floating-Point-and-Native-Boundaries/Floating-Point-Semantics/' | relative_url }})
explains the public scalar API and its proof boundaries. Earlier timeline entries describe the
older versions, including their numerical module names and recorded validation runs.

  </div>
</article>

<article class="update-card" id="september-2026-proof-refactor" markdown="1">
  <div class="update-date">September 2026</div>
  <div class="update-body" markdown="1">

## Derivative Theorems for the Executed Backward Sweep

This round of work was about closing gaps between what the runtime executes and what the theorems
talk about, and about cutting code that no longer earned its place.

### Proofs about the code that runs

The eager runtime's dense backward sweep now has a Frechet-derivative theorem. Lowering a graph to
a tape and running `backwardDenseAll` over it computes the adjoint of the derivative of the graph's
real semantics (`backwardDenseAll_lowerGraphToTape_adjoint_fderiv`), with supporting results that
the sweep agrees with `backpropAllCtx` and that lowering preserves zero seeds. Before, the
statements were about an abstract backpropagation function and the executed sweep had to be
trusted to match it.

IR shape inference is proved sound against the semantics. `checkShapes_sound` and
`denoteAll_shape` say that when the shape checker accepts a graph, every value produced by
denotation has the inferred shape. The lowering theorem `denoteAll_eq_of_lowerToForwardGraph` now
needs only a `NoRawLog` hypothesis; the `NoMSELoss` and `NoConcat` side conditions are gone
because lowering covers concatenation, rank four and higher matrix multiplication, and batched
linear layers. `expectResultShape` and every `panic!` were removed from the IR executor.

The real-valued CROWN transfer proofs now compose with IBP soundness. The transfer theorems used to
assume `IBPEnclosesVals`, that the IBP boxes enclose the semantic values. `EndToEnd.lean` derives
that premise from the IBP soundness theorem. `alphaCrown_cert_encloses_semantics` and the alpha-beta
variant retain the graph support, topological ordering, input, local replay, and relaxation
parameter hypotheses. Their conclusions concern nodes with both a certificate entry and a semantic
value; the all-node variants require coverage. `runIBP_encloses_evalGraphRec` gives the corresponding
result for the runtime IBP algorithm over real scalars, under its engine-core and coverage
hypotheses. These results do not establish soundness of the rounded CLI passes.

Softmax, attention, LayerNorm, and BatchNorm derivative proofs are now stated about the `Spec`
definitions rather than about auxiliary functions that happened to agree with them.
`hasFDerivAt_softmaxSpec_vec` and `softmaxBackwardSpec_eq_vjp` cover softmax and log-softmax;
`backpropVec_eq_adjoint_fderiv_scaledDotProductAttention` connects the attention tape rule to the
derivative of the spec; `hasFDerivAt_batchNorm`, `fderiv_batchNorm_eq_batchNormJvp`, and
`layerNormJvp_layerNormBackward_adjoint` do the same for normalization. The LayerNorm theorems
take `0 < ε` explicitly.

Two division bounds in the rounded-real approximation library were vacuous: their hypotheses could
not be met at the same time, so the theorems proved nothing about any real run. They were deleted,
and the sigmoid, logistic, and mean bounds were rebuilt on `divPosErrorBound`, with small
regression theorems (`reciprocal_sigmoid_bound_scalar_le_one`, `mean_row_bound_of_exact`) so the
same mistake would be caught again. The strongly convex gradient descent bound now keeps its
`q^k` factor.

The native LayerNorm kernel now keeps its mean and centered variance in binary64 before storing
binary32 normalized values. This fixes large-offset rows whose small differences were lost by
rounding the mean too early. Its input, scale, and bias gradients were checked against the Float
specification using the saved forward values.

LayerNorm derivative certification now returns an unsupported result. The old flattened interval
rule could understate a derivative bound; the real-valued row theorems do not justify that rule.
The complete transfer proof for the directed row payload is still needed before this certificate
path can be enabled.

### Smaller pieces

Several monolithic proofs were split into files that can be read one case at a time.
`alphaCrown_transfer_sound` went from 1,857 lines to a 44-line dispatch over per-operation files,
`cert_encloses_semantics` from 1,283 lines to 10, and the two dense backward lowering proofs from
about 800 lines each to 64 over shared leaf and snoc lemmas.

Fully qualified `_root_` escapes dropped from about 4,400 to a few hundred. Most were proofs
reaching around their own namespace for names that a proper `open` or a local `abbrev` supplies.
The `lemma` keyword is replaced by `theorem` outside `NN/Floats`.

The backend capsule layer was cut to what the runtime uses. `ProofCarryingKernel`,
`VerifiedPlannedKernel`, `Target`, `Gate`, `Recheck`, and the `TrustLevel.verified` and `fuzzed`
labels are gone; none of the maintained capsules carried a refinement theorem, so the vocabulary
promised more than the code delivered. What remains is `KernelCapsule` with `bind`,
`checkContracts` producing a `ContractCheck`, and a numerical policy with one field, `reduction`,
which the numerical certificate registry reads. The
[installation page]({{ '/installation/#from-a-model-to-a-kernel' | relative_url }}) describes the
current selection path.

The generated `UnicodeData.lean` table went from 5,767 lines of per-character entries to a
376-line interval table with the same lookups.

### API names

`TrainOptions.batchSize` is now `samplesPerStep`, and `steps` is required rather than defaulted.
The `seq!` macro is gone: `nn.compose![...]` composes built layers and `nn.Sequential![...]` is the
monadic builder, both scoped so files need `open TorchLean`. The einops term syntax is scoped under
`TorchLean.Tensor`, and `repeat` is renamed `expand`. `Runtime.Autograd.TorchLean` is
`Runtime.Autograd.Model`, and `NN.Verification.TorchLean` is `NN.Verification.Builtin`.
`Trainer.Result` exposes `state`, `save`, and `stateShapes`, `Trainer.load` restores a saved
result, and `nn.buildIO` instantiates a builder outside a trainer.

Then we went looking for names that existed only so that older call sites would keep working, and
deleted them. `Spec.fill`, `Spec.zeros`, and `Spec.ones` were thin wrappers whose own docstrings
described them as older spellings of `Tensor.full`, `Tensor.zeros`, and `Tensor.ones`, so the
canonical names are now the only ones, and the value argument follows the shape the way
`torch.full(shape, value)` reads. The `Runtime.Autograd.Model.Random` module was the same idea at
module scale, fourteen wrappers around `Spec.Random`; it is gone, and the RNG helpers now come from
`NN/Spec/Core/Random.lean` as `Spec.Random.splitmix64` and friends. Trainer runtime flag parsing
now lives in `NN.API.CLI.Trainer`: use `TorchLean.CLI.Trainer.parse`, `parseCommandLine`, and
`cliArguments`. The older trainer helpers and `NN.Examples.Support.TrainerFlags` are gone.

Smaller renames from the same pass: `Tensor.tensorFoldlSpec` lost its stuttering prefix and is
`Tensor.foldlSpec`, the ONNX helpers dropped their `onnx` prefix now that they live in
`namespace ONNX`, `Opts` is spelled `Options` in the verification command lines,
`inferNodeOutShape` is `nodeOutShape`, and `TextCorpusOptions` is `CorpusFileOptions`. Two naming
rules came out of this and are written down in `docs/CONTRIBUTING.md` so the next reader does not have to
guess: the smart constructor of a sealed structure is `create` inside that structure's own
`Internal` namespace, and a batched variant is the unbatched name with a `batch` prefix, which is
why `logitScoresAt` has `batchLogitScoresAt` beside it.

The repository lint now enforces 100-column lines and prose without em-dashes outside
`NN/Floats`, and `omega` is allowed again.

  </div>
</article>

<article class="update-card" id="august-2026-tensor-overhaul" markdown="1">
  <div class="update-date">August 2026</div>
  <div class="update-body" markdown="1">

## Cleaning Up the Tensor API

We wanted to improve the naming and cut out repeated code. Too many public names said `2d`, `3d`,
`NCHW`, or `batch` even when the operation itself did not depend on those choices. That made the
API harder to guess and left us maintaining several routes to the same tensor operation.

Convolution was the biggest example. `nn.conv`, `nn.convTranspose`, `nn.maxPool`, and `nn.avgPool`
now take a spatial rank `d` together with `Tensor Nat [d]` values for their geometry. One definition
covers lines, images, volumes, and higher-dimensional grids. Batch, sequence, and other outer axes
are passed in `leading : List Nat` and preserved by the operation. `nn.ConvGeometry.samePadding`
works at any spatial rank and comes with a proof that it preserves positive spatial extents.

We made the same change to normalization and shape operations. `nn.batchNorm` and
`nn.instanceNorm` preserve any leading and spatial axes. `permute` and `transpose` take explicit
axes instead of putting a layout in the function name. `matmul` handles compatible leading
dimensions, and `flattenAfter` says exactly which prefix of the shape should be kept.

The main name changes are:

| Old name | Use now |
| --- | --- |
| `conv2d`, `Conv2dSpec` | `nn.conv`, `Spec.convSpec` |
| `convTranspose2d`, `ConvTranspose2dSpec` | `nn.convTranspose`, `Spec.convTransposeSpec` |
| `maxPool2d`, `avgPool2d`, adaptive `*2d` specs | rank-polymorphic pooling operations and specs |
| `batchNorm2d`, `batchNormChannelFirst`, `instanceNorm2dNchw` | axis-general normalization |
| `transpose2d`, `transpose3d*`, `nchwToNhwc`, `nhwcToNchw` | `permute` or `transpose` with explicit axes |
| `mm`, `bmm` | `matmul` |
| `flattenBatch`, `flattenLeading` | `flattenAfter` |

We removed the old names and migrated their callers instead of keeping aliases around. PyTorch's
rank-specific names still appear inside import and export code because that is how PyTorch spells
those operations, but they are no longer TorchLean APIs.

While doing that, we got rid of vague names like `General` and `ND` too. The ordinary MLP evaluator
is now just `mlpEval`; its scalar-input version is `mlpEvalScalar`. Grouped convolution and dilated
geometry say so in their names, the pooling implementation lives under `Pooling.Spatial`, and the
tensor-input Stone–Weierstrass result lives in `Universal.StoneWeierstrass`.

We followed the change all the way down through the specification, typed graph, IR evaluator, shape
inference, eager runtime, CUDA dispatch, and reverse-mode rules. Convolution now has general input,
kernel, bias, and transpose-convolution derivatives. BatchNorm has the matching adjointness theorem
for its input, scale, and bias gradients at any spatial rank. The rounded-real proofs follow the
implementation's actual accumulation order.

The smaller tensor details got cleaned up too. Proofs can use `Tensor.map_scalar`, `Tensor.map_dim`,
and `Tensor.getScalar_map` instead of reopening the shape recursion themselves. Classifier and
regressor heads keep their leading axes. Indexed embeddings append their width to the token shape
and scatter-add repeated token IDs during backward execution.

This let us delete the old tensor forwarding modules, packed/vector wrappers, fixed-rank pooling
files, 2D padding helpers, and the channel-first BatchNorm proof wrapper along with their callers.

  </div>
</article>

<article class="update-card" id="august-2026-lean-433" markdown="1">
  <div class="update-date">August 2026</div>
  <div class="update-body" markdown="1">

## Lean 4.33

TorchLean now builds on Lean and mathlib 4.33. The migration touched dependent tensor casts,
autograd derivative proofs, graph evaluation, CROWN certificates, and the documentation toolchain.

The execution API now separates execution mode from device selection:

```lean
{ execution := .typedGraph
  device := .cpu }
```

Eager execution runs each operation immediately and builds a dynamic tape. Typed graph execution
records a shape-indexed executable graph once and reuses its forward, JVP, and VJP programs. The
graph stores an exact typed output reference, so a model may return an input or intermediate rather
than only its final recorded node. Call `nn.lowerToTypedGraph model` when the graph itself is needed.
The executable graph stores derivative rules; the separate proof-carrying graph establishes when
those rules are mathematically correct. TorchLean now uses *lowering* for this conversion and
reserves *compilation* for future optimization, fusion, scheduling, and native code generation. The
previous misleading execution terminology was removed rather than retained as aliases.

The model API also distinguishes definitions from live state. `nn.Sequential` and `nn.Layer` are
immutable values used by lowering and proofs. Instantiating a checked model produces an
`nn.Module` with mutable parameters, persistent buffers, and train/eval mode. A lower-level
`Module.Objective` pairs that state with a scalar training objective. Its name makes clear that the
object owns model state as well as the loss calculation.

Training now has one public lifecycle for regression, classification, and custom losses. A trainer
selects the loss once, then `predict`, `train`, checkpointing, streams, and verification reuse the
same model state and runtime dispatch. The duplicate task-specific runners and definitional state
layout lemmas are gone.

Lean's `Float32.Model` exposes the logical definitions of core binary32 operations. TorchLean now
proves agreement between that model and its independent raw-bit binary32 implementation for
classification, comparison, addition, subtraction, multiplication, division, square root,
negation, and absolute value. The arithmetic proofs cover normal and subnormal inputs, signed zeros,
infinities, NaNs, underflow, overflow, and nearest-even rounding. NaN results are canonicalized
before their bits are compared because the models retain different payload information.

Native compiler output, processors, CUDA, cuBLAS, and LibTorch retain their own backend contracts.
Transcendental `Float32` functions also remain outside the core logical model.

The logical model is part of
[Lean 4.33](https://lean-lang.org/doc/reference/latest/releases/v4.33.0/), and its binary32
implementation is available in
[`Init.Data.Float.Model.Float32`](https://github.com/leanprover/lean4/blob/v4.33.0/src/lean/Init/Data/Float/Model/Float32.lean).

  </div>
</article>

<article class="update-card" id="august-2026-autograd-cuda" markdown="1">
  <div class="update-date">August 2026</div>
  <div class="update-body" markdown="1">

## Exact Tape Derivatives and CUDA Cache Limits

The exact autograd proof reaches the lowered tape. For a real algebraic graph, dense reverse
accumulation returns the graph's full cotangent context, and its input block is the adjoint Fréchet
derivative of forward evaluation applied to the output seed. The pointwise theorem permits
piecewise-smooth operators when the chosen execution point satisfies their differentiability and
domain hypotheses.

The connection uses an exact correspondence between the analytic graph and the real algebraic graph
with a trivial environment. Conversion preserves forward evaluation, JVPs, and reverse
accumulation, and both graph conversions round-trip. The theorem can be inspected without importing
an executable backend:

```lean
import NN.Proofs.Autograd.Runtime.Link.FDeriv

#check Proofs.Autograd.Algebra.Graph.backwardDenseFrom_lowerGraphToTape_adjoint_fderiv
#check Proofs.Autograd.Algebra.Graph.backwardDenseFrom_lowerGraphToTape_adjoint_fderiv_at
```

This result concerns the exact tape over `Real`. Native `Float32` and CUDA executions still require
the numerical-refinement assumptions described in the runtime-approximation chapter.

The CUDA allocator also has an optional byte limit for released buffers retained for reuse:

```bash
TORCHLEAN_CUDA_CACHE_CAP_BYTES=$((512 * 1024 * 1024)) \
  lake -K cuda=true exe torchlean gpt2 --device cuda --steps 100
```

`AllocatorStats.cacheBytes` reports reusable device memory separately from live tensors, while
`cacheCapBytes` reports the parsed limit. A block that would exceed the limit is synchronized and
freed. The cap is fixed when the allocator first reads it; `0` or an unset variable means
unbounded.

The CUDA stress suite runs the allocator in fresh subprocesses and checks a finite limit, an
explicit unbounded control, malformed input, and integer overflow. Its assertion uses the limit
reported by the native allocator rather than reproducing the native parser in Lean.

  </div>
</article>

<article class="update-card" id="july-2026-refactor" markdown="1">
  <div class="update-date">July 2026</div>
  <div class="update-body" markdown="1">

## Lean 4.32, Numerical Proofs, and Runtime Tests

<p class="update-kicker">Imports, floating point, training, CUDA, and documentation</p>
<p class="update-summary">
The July work touched most of TorchLean. We removed duplicate entry points, generalized tensor
operations, reorganized the floating-point files, and tested the training runtime on models large
enough to expose bugs that the small examples never reached.
</p>

### Imports and File Layout

Most model code starts with `import NN.API`. The old `NN.Library` and `NN.Entrypoint.*` forwarding
modules are gone. Focused imports such as `NN.Spec`, `NN.Runtime`, `NN.Floats`, and
`NN.Verification` still lead directly to their declarations. The model examples remain part of
TorchLean.

We also broke up several files that had become difficult to navigate. Training, data handling,
schedulers, CROWN propagation, graph lowering, runtime operations, normalization, Muon, and
floating-point semantics moved into smaller modules with narrower imports. The API tree is about
300 lines smaller and the guide is more than 5,000 lines shorter. The proof tree grew to include
numerical certificates, rounded backpropagation, optimizer contracts, and new floating-point
results.

The public neural-network API has one owner. Seeded layer builders and model-zoo constructors are
declared under `TorchLean.nn`; `NN.API` gathers them without redeclaring their names.
Fixed-sample training lives under `TorchLean.Trainer.FixedSample`. Inside the runtime, mutable
parameter storage and optimizer checkpoint schemas have their own modules, separate from trainer
execution and CUDA Adam serialization.

<div class="update-grid">
  <section>
    <h3>General tensors</h3>
    <p>
      A batch is an axis of a tensor. Permutation, reduction, reshape, and global average pooling
      work over declared axes. Channel normalization takes an explicit channel axis and preserves
      every other axis.
    </p>
  </section>
  <section>
    <h3>Models</h3>
    <p>
      CNNs, ResNets, ViTs, FNOs, transformers, GPT, Mamba, recurrent models, generative models,
      reinforcement learning, and self-supervised examples all remain available. They use the
      same tensor and layer API instead of carrying model-specific forwarding stacks.
    </p>
  </section>
  <section>
    <h3>Training</h3>
    <p>
      Optimizers share one stateful tensor interface. We kept laws that say something useful
      about update rules and stream composition, and removed generated tables and `rfl` theorems
      that only repeated a definition.
    </p>
  </section>
</div>

The trainer treats `samplesPerStep` (called `batchSize` at the time of this note) as the number
of dataset items per optimizer update. For an ordinary dataset those items are samples. For
`Data.batch`, each item is already a typed tensor minibatch, so `samplesPerStep := 1` keeps one
vectorized pass per update. Larger values accumulate
gradients across several items. Logged pre-update loss comes from the same forward tapes as the
gradients; training no longer runs a second forward pass just for logging.

Transformer batches use one attention tape node instead of asking the host to run the layer once
per sample. The public model and typed graph preserve the declared leading shape. The eager CUDA
path may fold those axes together with the attention heads for batched matrix multiplication, while
TorchLean still applies the hard mask and computes the local VJP. A regression compares the
vectorized forward value, input gradient, and shared weight gradients with repeated single-sample
execution.

Layer normalization and tanh-approximate GELU follow the same rule: one TorchLean operation,
one local VJP, and fused CUDA kernels for the numerical work. In a two-step GPT-2-small trace with
batch 6 and context 1024, local profiling showed fewer kernel launches while leaving the
matrix-multiplication schedule unchanged. The CUDA parity suite checks forward values and gradients
against the CPU path; the native kernels remain inside the documented runtime boundary. Timing
claims belong with a retained benchmark configuration and trace, so this update records the
implementation and regression coverage rather than presenting one workstation run as a general
speedup.

Matrix backward no longer materializes transposed copies before calling cuBLAS. The runtime
passes logical transpose flags to the same batched-matrix primitive used by linear layers,
projection weights, and ordinary `matmul`. Focused traces confirm that these temporary transpose
kernels disappear, and parity tests compare the resulting forward values and gradients with the
existing path.

CUDA Adam and AdamW state can be saved independently of model checkpoints. The binary
format records optimizer hyperparameters, parameter shapes, mutability flags, moment tensors, and
step counters using explicit little-endian fields. Loading rejects mismatched models, changed
moment parameters, duplicate state entries, truncation, and trailing data. Writes close and flush a
fresh sibling file before renaming it over the destination. The parameter-schema codec is shared by
backend-owned optimizer checkpoints; the CUDA Adam-family codec supplies the format-specific
configuration and moment payload.

Discrete model inputs have their own typed path through programs, modules, evaluators,
trainers, and checkpoints. CharGPT passes token ids and targets as bounded
`Tensor (Fin vocab) [batch, seqLen]` values; raw `Tensor Nat [batch, seqLen]` values are admitted
only after `Tensor.checkIndices` validates the tokenizer boundary. The old
floating-point transport and conversion step are gone. The causal Transformer API supports either
an independent vocabulary head or an output projection tied to the embedding table. In the tied
form, lookup and output gradients accumulate into the same parameter.

### Floating-Point Semantics

`import NN.Floats` provides formats, rounding, finite binary32 semantics, executable IEEE binary32
operations, interval rounders, and scalar quantization. It does not pull in tensors, models,
autograd, CUDA, certificate checkers, or external tools. Tensor and proof integrations sit above
that import, while optional Arb checks require an explicit import.

The generic development under `NN.Floats.NeuralFloat` is organized by format, rounding, scalar
operations, analysis, error bounds, and execution policy. It covers radix and exponent formats,
directed and nearest rounding, round-to-odd, ULPs and neighboring values, double rounding, Sterbenz
subtraction, and absolute and relative error bounds. Flocq influenced the layout. TorchLean's
definitions and proofs are written in Lean.

Sterbenz subtraction covers gradual underflow and has a binary32 specialization. Every finite
executable binary32 bit pattern is proved representable in that specification, so the executable Sterbenz
theorem can identify nearby subtraction with the exact real difference. Finite executable values
also expose a checked ULP exponent, and an absorption theorem connects an unchanged binary32
accumulator to the rounded-real specification.

We use the following distinction throughout TorchLean:

- `NeuralFloat` and `NF` describe configurable rounded-real arithmetic used in proofs;
- `FP32` specializes the rounded-real model to binary32-sized parameters;
- The executable binary32 model includes IEEE-754-style behavior, including special values;
- runtime bridges state how native values are interpreted by those models.

The effective-rounding example shows the whole argument on one value: choose a format and rounding
mode, perform the rounding, and derive the resulting error bound. The runtime-approximation proofs
then start from the ideal autograd theorems and make every extra hypothesis about rounded execution
explicit.

### Whole-Graph Numerical Certificates

TorchLean can build a numerical trace over the canonical `NN.IR.Graph`. Source intervals use
exact binary32 endpoints. The checker reconstructs
outward-rounded ranges for supported arithmetic, activations, directed square root, reductions,
matrix multiplication, pooling, MSE, and stable softmax; malformed domains and non-finite ranges
fail at the node that produced them.

Range rules live in an operation registry. The same traversal handles any architecture after
lowering. Before propagation, a coverage pass lists
the exact nodes whose primitives lack a range contract. Custom registries are named and the name is
stored in the certificate, so an artifact cannot be replayed under a different set of rules.

The same certificate contains the kernel-selection audit. Each kernel capsule records its
reduction order in its numerical policy. Portable accumulations use the fixed left fold from the
tensor semantics. CUDA and LibTorch accumulations are marked
implementation-dependent, so their matrix products, convolutions, normalizations, FFT/FNO paths,
scans, and attention kernels cannot accidentally inherit a proof for a different reduction order.

The bit-level replay evaluates every graph intermediate with executable binary32 arithmetic, checks its shape and
range, and rejects NaN or infinity. A checked certificate stores the exact graph it was checked
against, so replay cannot substitute a different graph. A separate proved real execution supplies
the semantic enclosure; combining it with the bit-level replay yields an entrywise error trace for
every node. The deep-dive example includes successful arithmetic, reduction, matmul, LayerNorm,
`abs -> sqrt`, and softmax traces, together with deliberately tampered, invalid-domain, and
wrong-reduction-policy cases. It ends with a complete two-layer MLP: ten graph nodes pass
coverage, range generation, backend-capsule audit, and bit-level replay. The
[numerical-runtime walkthrough]({{ '/examples/numerical-runtime/' | relative_url }}) follows that
run from source enclosures to its checked output.

### Rounded Backpropagation and Optimizers

The numerical proof continues past the forward graph. Proof-bearing reverse nodes carry both
their ideal VJP and their rounded VJP error transformer. The global reverse theorem composes those
local bounds through gradient accumulation and connects the result to executable autograd
`GraphData`.

One optimizer contract carries the gradient error through parameter updates. SGD and momentum SGD
have no extra domain condition. AdamW uses the same interface, with step data recording errors from
both moments, bias correction, square root, adaptive division, decoupled weight decay, and the final
subtraction; explicit margins keep the rounded denominator away from zero. The end-to-end theorem
therefore works unchanged for all three optimizers and for every model represented by a `RevGraph`.
A model-wide update applies it at each typed parameter index.

The canonical `NN.IR.Graph` certificate remains a forward certificate. Its current lowering does not
attach proved VJPs, so backward claims use the proof-bearing reverse graph path instead of silently
attributing autograd semantics to a forward-only lowering.

### Tensor Quantization

Uniform affine quantization uses FloatLib's scalar definitions, re-exported by
`NN.Floats.Quantization`, with a rank-polymorphic tensor adapter under `NN.Spec.Quantization`.
The proofs cover code-range
preservation, monotonicity, exact dequantize/quantize round trips for in-range integer tensors, and
the half-step reconstruction bound when saturation is inactive. Layout and storage width are not
part of the arithmetic: int8, uint8, int4, and custom code sets differ through their integer bounds
rather than separate image-specific APIs.

### Backend Contracts

Backend planning now records device, provider, operation, contracts, and evidence separately, then
binds each accepted capsule to a matching runtime handler. Unavailable providers fail explicitly,
and the contract check accepts a capsule's evidence only when the profile's assurance policy admits
it, rather than relying on a trust label. The [backend chapter]({{ '/blueprint/Runtime___-Autograd___-and-Interop/Inside-The-Backend-Planner/' | relative_url }})
contains the maintained profiles and full contract model; the
[GPU chapter]({{ '/blueprint/Floating-Point-and-Native-Boundaries/From-A-Tensor-Operation-To-A-GPU-Kernel/' | relative_url }})
covers native execution and platform boundaries.

### Mathematical and Verification Corrections

<div class="update-grid">
  <section>
    <h3>Losses and masks</h3>
    <p>
      Huber loss and Smooth L1 have their intended, distinct scaling. Hard attention masks use
      exact exclusion in the softmax semantics: a blocked entry contributes zero numerator. The old
      finite <code>-1000</code> masking convention was removed from attention paths and examples.
    </p>
  </section>
  <section>
    <h3>Bounds</h3>
    <p>
      Leaky-ReLU interval propagation handles a negative slope across the kink at zero. Logarithm
      interval checks reject nonpositive domains. Unsound or undocumented GELU and ELU candidate
      bounds were removed instead of being exposed under names that suggested certified enclosure.
    </p>
  </section>
  <section>
    <h3>Certificates</h3>
    <p>
      JSON certificate readers reject non-finite claims before array comparisons. IBP certificates
      are checked by recomputing the complete binary32 trace from the trusted graph,
      parameters, and input box; an artifact may widen that trace but may not shrink it. CROWN and
      $\alpha,\beta$-CROWN affine entries are compared exactly with a sequential replay instead of being
      propagated from certificate-supplied parents. A theorem turns successful exact replay into
      the local-consistency proposition used by the generic CROWN soundness development. Relating that
      binary32 replay to real-valued enclosure still requires the stated finite-precision refinement
      assumptions.
    </p>
  </section>
  <section>
    <h3>Classical models</h3>
    <p>
      HMM normalization records zero probability for an impossible observation, and log-likelihood
      is partial at that boundary. GMM covariance matrices must be symmetric positive definite,
      mixture weights must be positive and normalized, and singular inversion fails rather
      than returning the identity. The covariance gradients use the transpose-correct formulas.
    </p>
  </section>
  <section>
    <h3>Attention and diffusion</h3>
    <p>
      Multi-head attention reshapes sequence data to
      <code>(sequence, heads, head-dimension)</code> before exchanging the sequence and head axes.
      The probability-flow ODE uses the required one-half score coefficient, and its Euler sampler
      visits time points in descending order from the noisy endpoint.
    </p>
  </section>
  <section>
    <h3>Layer and model edges</h3>
    <p>
      Dropout is the identity in evaluation mode. Max pooling rejects padding configurations that
      would create windows containing no input values. PCA requires at least two samples for its
      unbiased covariance and exports the centering term as a linear bias. Linear SVM fitting calls
      its regularization coefficient <code>lambda</code>, leaving <code>C</code> for the standard
      inverse-strength convention.
    </p>
  </section>
  <section>
    <h3>Formats and smooth pooling</h3>
    <p>
      A radix carries a proof that its base is at least two, and a format precision carries a
      proof that it is positive. Checked constructors reject bad integers at configuration
      boundaries. Smooth max pooling uses a sign-aware pivot for both positive and negative
      inverse temperatures on CPU and CUDA. Zero, non-finite, or unrepresentable inverse
      temperatures are rejected before a native kernel runs.
    </p>
  </section>
</div>

Rounded CROWN no longer discards every backward objective to a constant interval. For algebraic
nodes it carries lower and upper coefficient vectors with directed arithmetic, including the sign
of each input interval when the final affine form is evaluated. Nonlinear or unsupported nodes
still fall back to their checked IBP boxes. This improves the executable bound without pretending
that an unproved floating-point transfer is exact.

The arithmetic interface separates implementation from proof. `BoundOps` provides executable
lower and upper operations; `LawfulBoundOps` proves that those operations enclose exact real
addition, subtraction, and multiplication. Real and `FP32` endpoints have lawful instances. Host
`Float` remains an explicitly trusted execution boundary, while IEEE special values are handled by
finite-path theorems instead of a blanket ordered instance.

The Lyapunov workflow no longer contains a repository-wide oracle axiom. Python output records a
region and numerical margins, and generated Lean files may prove arithmetic facts about those
numbers. A stability theorem additionally requires `LyapunovCert.ValidFor`, whose fields prove that
the reported intervals enclose the named Lyapunov function and orbital derivative throughout that
region.

The graph evaluator and verifier lowering are split by operation. Their coverage theorems still
range over the full operation vocabulary, so adding a new file does not weaken the statement being
proved. We removed theorems tied to incidental list lengths and retained small definitional lemmas
only when later correctness proofs actually use them.

Convolution, transposed convolution, fixed-window pooling, and adaptive pooling now share
channel-first contracts parameterized by spatial rank. Their public APIs take vectors of kernel,
stride, padding, and output dimensions; the same definitions therefore cover lines, images,
volumes, and higher-dimensional grids. The IR, eager runtime, CUDA path, and shape inference use
these contracts without separate rank-named wrappers. A semantic-preservation theorem connects
typed convolution lowering to forward IR evaluation. The exact derivative theorem covers the same
rank-polymorphic operation and proves the input, kernel, and bias reverse rules. Rounded-real
theorems bound every forward and backward coordinate using the implementation's actual accumulation
order. BatchNorm now has the corresponding arbitrary-spatial-rank adjointness theorem for its input,
scale, and bias gradients.

PyTorch graph import now preserves every leading dimension of a linear layer. In particular, a
batched input of shape `[3, 4]` passed through `Linear(4, 1)` is imported with output shape `[3, 1]`
rather than `[3]`. The runtime check includes this case alongside arbitrary-axis reductions,
permutations, normalization, convolution, pooling, and attention, while continuing to reject
unsupported operator semantics explicitly.

Verification artifact readers share one finite box-region parser. It rejects non-finite
coordinates, negative radii, mismatched dimensions, reversed intervals, incomplete field pairs,
and mixed endpoint/center schemas. Format-specific checkers can require the exact endpoint schema;
the alpha-beta-CROWN leaf checker does so before checking nesting and threshold witnesses.

### Lean 4.32

TorchLean builds with Lean, mathlib, DocGen, and Verso 4.32. During the upgrade we replaced the
deprecated `Lean.RBMap` with `Std.TreeMap`, used mathlib's stronger sine remainder estimate, and made
several dependent casts explicit. Proof-valued runtime helpers are theorems when they serve as
opaque evidence; constructors that must compute remain reducible `abbrev`s. We fixed the new
linters rather than suppressing them.

### Runtime Scaling and CUDA Ownership

The small examples had hidden an expensive habit: large parameters were first expanded into nested
Lean values and only then copied into the execution engine. Parameters and gradients are
materialized directly where they will run. We also stopped generic convolution backward from
rebuilding the same derivative structure, and taught CUDA attention and fused FNO paths to release
temporary buffers as soon as their contribution is consumed.

The most useful failure came from sparse reverse mode. A pure expression allocating a one-element
CUDA seed could be shared by Lean, even though backward consumed and released the native buffer.
The next use then referred to storage that was no longer alive. Seeds that cross an ownership
boundary come from an effectful constructor, and transfers between gradient maps use explicit
copy-and-release operations. A stress test repeats this path and fails if live CUDA allocation
grows or a supposedly fresh seed is reused.

A second lifetime problem was in the FFI signatures themselves. Buffer and array inputs were being
passed as owned Lean objects to native functions that treated them as borrowed, so neither side
released the wrapper reference. The declarations mark those inputs as borrowed. Separate
payload and wrapper counters make the distinction visible, and the stress suite checks thousands
of allocations for matching finalization counts.

Shape-erased CUDA values compare the native buffer length with the recorded tensor shape before
an operation runs. Dense and sparse backward also reject output seeds or initial gradients with the
wrong length. The stress suite covers each rejected case.

We exercised 21 CPU workflows and 24 CUDA workflows, including dense, convolutional, attention,
recurrent, operator-learning, generative, and reinforcement-learning models. On the machine used
for this release, a roughly 100-million-parameter MLP completed ten CUDA optimizer steps in about
15.2 seconds. The fused Burgers FNO ran for 100 steps with no growth in live buffers; training MSE
fell from 0.3260 to 0.0172 and test MSE ended at 0.0220. These numbers record what we tested on one
machine. They are not a general performance promise.

### Documentation and Validation

The Guide and API reference follow the new module layout. Installation has separate notes for
Linux, macOS, WSL2, native Windows, CUDA, and optional LibTorch support, and the floating-point and
backend chapters explain where a theorem ends and a runtime assumption begins. Repository checks
build `NN` directly; `NN.Library` no longer exists.

Lyapunov results consume an explicit `LyapunovCert.ValidFor` proof, so a producer's JSON flags
cannot become a stability theorem by themselves.

The Graphs page contains the module-import explorer and build-performance link. The Tools page links
LeanProfiler and TorchLean Verified Examples. LeanProfiler includes a TorchLean model run, Perfetto
trace output, and JSON comparisons. The verified examples cover batch-invariant inference and a
verifiable transformer checkpoint.

The import explorer ignores fenced guide examples, so an `import` shown in a tutorial is not
mistaken for a source-module dependency.

Wide tables are wrapped during the documentation build, and the Guide's equations render with
KaTeX.

<div class="validation-list" markdown="1">
  <h3>Validation</h3>

- `lake lint`
- `lake build`
- `lake build NN NN.CI.All`
- `lake exe nn_tests_suite`
- `lake -R -K cuda=true exe nn_tests_suite`
- temporary audits across registered commands and selected CPU/CUDA examples
- sustained 20-update CPU runs across 21 model workflows
- sustained 100-update CUDA runs across 24 model workflows
- repeated sparse-backward ownership and allocator-drift regression
- external-wrapper allocation/finalization regression on both the CUDA and CPU-stub builds
- NVIDIA Compute Sanitizer memcheck (`ERROR SUMMARY: 0 errors`)
- DocGen API generation
- Verso Guide generation
- dependency audit and interactive import-graph generation
- Jekyll production build
- `git diff --check`

All of these checks passed on the Linux machine used for the release. That gives us evidence for the
paths we exercised, but it does not turn CUDA machine code or LibTorch into Lean proofs. Their trust
levels remain explicit in the backend contracts and in `docs/TRUST_BOUNDARIES.md`.
</div>

  </div>
</article>

<article class="update-card" id="june-2026-reliability" markdown="1">
  <div class="update-date">June 2026</div>
  <div class="update-body" markdown="1">

## CUDA, CROWN, and PINN Reliability

<p class="update-kicker">Native runtime, certificates, scientific ML</p>
<p class="update-summary">
The June reliability pass tightened CUDA allocation checks, made CROWN certificate statements more
direct, and moved duplicated PINN training code into shared helpers.
</p>

<div class="update-grid">
  <section>
    <h3>Runtime checks</h3>
    <p>
      CUDA and CPU stubs use shared checked size arithmetic for products,
      byte counts, and additions at the Lean FFI boundary. Broadcast, reduction,
      swap, gather/scatter, attention, convolution/pooling, tensor-copy, and
      spectral-convolution paths reject impossible sizes before allocation or
      kernel launch.
    </p>
  </section>
  <section>
    <h3>Proof API</h3>
    <p>
      The graph CROWN certificate theorem returns the enclosure for the
      node being checked. The IEEE32 version records the no-self-dependency
      condition on the evaluator trace, and the two-layer MLP CROWN code exposes
      the affine forms used by <code>boundAffineCrown</code>.
    </p>
  </section>
  <section>
    <h3>PINNs</h3>
    <p>
      Python PINN trainers share dataset loading, MLP construction,
      expression evaluation, gradients, constant parsing, and export helpers
      through <code>scripts/verification/pinn/pinn_common.py</code>.
    </p>
  </section>
</div>

The focused API import exposes the short `TorchLean.*` namespaces directly.
With `import NN.API`, users get `TorchLean.nn`, `TorchLean.optim`,
`TorchLean.Trainer`, `TorchLean.Data`, `TorchLean.Loss`, and
`TorchLean.Metrics` without importing the broader `NN` umbrella.

<div class="validation-list" markdown="1">
  <h3>Validation</h3>

- `lake test`
- `lake build NN.CI.All`
- `lake lint -R -K cuda=true -K cuda_home=/usr/local/cuda-13.0`
- `scripts/checks/check.sh --cuda --cuda-home /usr/local/cuda-13.0`
- `scripts/checks/cuda_sanitize_tests.sh --cuda-home /usr/local/cuda-13.0 --all-tools`
- focused Lean checks for the CROWN MLP and graph CROWN certificate modules
- PyTorch CUDA regression runs for the PINN trainers on an A100 GPU

CUDA sanitizer reported zero memcheck/initcheck/synccheck errors and no
racecheck hazards on the exercised runtime suite.
</div>

  </div>
</article>

<article class="update-card" id="june-2026-lean-431" markdown="1">
  <div class="update-date">June 2026</div>
  <div class="update-body" markdown="1">

## Lean 4.31 Migration

<p class="update-kicker">Toolchain alignment</p>
<p class="update-summary">
TorchLean builds with <code>leanprover/lean4:v4.31.0</code>. The root Lake
manifest, Mathlib pin, documentation generator pin, Verso blueprint toolchain,
website metadata, README, and formalization metadata were moved together.
</p>

The migration fixed proof-term breakages in differentiability and autograd
composition files where Lean 4.31 became stricter about composed functions and
eventual equality.

  </div>
</article>

<article class="update-card" id="may-2026-cleanup" markdown="1">
  <div class="update-date">May 2026</div>
  <div class="update-body" markdown="1">

## Repository Modularization and Comment Cleanup

<p class="update-kicker">Source organization</p>
<p class="update-summary">
The public source tree is easier to read, easier to review, and easier to extend without changing
TorchLean's intended behavior.
</p>

<div class="update-grid">
  <section>
    <h3>Structure</h3>
    <ul>
      <li>Large proof and runtime files were split along conceptual boundaries.</li>
      <li>Umbrella modules were kept only where they make imports clearer.</li>
      <li>Obsolete import shells and example names were removed.</li>
    </ul>
  </section>
  <section>
    <h3>Documentation</h3>
    <p>
      Comments were rewritten in a more mathlib-style voice: definitions state
      mathematical intent, runtime boundaries name assumptions, and examples
      avoid stale narration.
    </p>
  </section>
  <section>
    <h3>Scope</h3>
    <p>
      Model semantics, verification claims, CUDA behavior, and trusted boundaries are unchanged.
    </p>
  </section>
</div>

The examples and website pages were rebuilt against the new module layout.

  </div>
</article>

<article class="update-card" id="may-2026-runtime-api" markdown="1">
  <div class="update-date">May 2026</div>
  <div class="update-body" markdown="1">

## Runtime API Update

<p class="update-kicker">Training loops, streams, initialization</p>
<p class="update-summary">
Longer examples use the same public runtime API for initialization, minibatches, optimizer
steps, logging, and checkpoint-style parameter files.
</p>

<div class="update-grid">
  <section>
    <h3>Initialization</h3>
    <p>
      Runtime-side Float initializers and shape-indexed initializer plans make parameter construction
      explicit: the model declares the parameter shape, the initializer plan selects
      the distribution or constant, and the runtime builds each tensor once.
    </p>
  </section>
  <section>
    <h3>Streams</h3>
    <p>
      Step-indexed training streams make minibatches explicit. A batch may come from a rule, a
      simulator, a replay buffer, or a file-backed window source, but the training loop sees a typed
      stream of inputs, targets, and metadata.
    </p>
  </section>
  <section>
    <h3>Language models</h3>
    <p>
      Integer-token embedding and row-wise cross-entropy helpers let GPT-style examples train on
      token ids directly, instead of expanding every target into a one-hot vector.
    </p>
  </section>
</div>

The runtime documentation follows the same path as the examples: initialize parameters, produce
batches, run forward/autograd, update parameters, save reports, and state the native or external
boundary when a backend is selected.

  </div>
</article>

<article class="update-card" id="may-2026-cuda-stability" markdown="1">
  <div class="update-date">May 2026</div>
  <div class="update-body" markdown="1">

## CUDA Training Stability

<p class="update-kicker">Memory lifetime, allocator diagnostics</p>
<p class="update-summary">
Longer CUDA training runs exposed allocator pressure from intermediate values
that could stay attached to a run longer than needed.
</p>

The issue was not model size. Some intermediate values created during training
-- tape entries, gradient buffers, and kernel workspace buffers -- could remain
alive across many optimizer steps. The fix made the training loop and CUDA eager
backend explicit about which values are returned to the caller and which buffers
can be released after the step finishes.

<div class="update-grid">
  <section>
    <h3>Step counts</h3>
    <p>Loader-based model commands treat <code>--steps</code> as optimizer updates.</p>
  </section>
  <section>
    <h3>Buffer lifetime</h3>
    <p>CUDA eager/autograd releases ephemeral tape, gradient, and workspace buffers after each step.</p>
  </section>
  <section>
    <h3>Diagnostics</h3>
    <p>Longer CUDA examples report allocator telemetry by default, with <code>--cuda-mem-watch N</code> for exact sampling.</p>
  </section>
</div>

<div class="validation-list" markdown="1">
  <h3>Validation</h3>

- `lake build`
- `lake -R -K cuda=true build`
- `lake exe torchlean mlp --device cpu --steps 10 --log false`
- `lake exe torchlean mlp --steps 10 --scalar float32 --execution eager --log false`
- `lake -R -K cuda=true exe torchlean mlp --device cuda --steps 1000 --log false`
- `lake -R -K cuda=true exe torchlean cnn --device cuda --steps 1000 --log false`
- `lake -R -K cuda=true exe torchlean gpt2 --device cuda --steps 1200 --generate 0 --log false`
- `lake -R -K cuda=true exe torchlean fno1d_burgers --device cuda --steps 50 --log false`

The validation checked representative losses or MSE values going down and the
CUDA allocator staying bounded on the exercised runs.
</div>

  </div>
</article>

<article class="update-card" id="may-2026-data-note" markdown="1">
  <div class="update-date">May 2026</div>
  <div class="update-body" markdown="1">

## Introductory Data Note

<p class="update-kicker">Reproducible builds</p>
<p class="update-summary">
Some examples use public datasets and do not download them during
<code>lake build</code>. Keeping data downloads explicit makes ordinary builds
deterministic and avoids silently committing large external data.
</p>

For the model-zoo MLP example:

```bash
python3 scripts/datasets/download_example_data.py --auto-mpg
```

For the text and vision examples:

```bash
python3 scripts/datasets/download_example_data.py --tiny-shakespeare --cifar10
```

  </div>
</article>

<article class="update-card" id="may-2026-release" markdown="1">
  <div class="update-date">May 2026</div>
  <div class="update-body" markdown="1">

## TorchLean Released

<p class="update-kicker">Initial public release</p>
<p class="update-summary">
TorchLean became public as a Lean 4 framework for writing, running, inspecting,
and verifying neural-network programs.
</p>

<div class="update-grid">
  <section>
    <h3>Core system</h3>
    <p>
      Typed tensors, layers, model APIs, loaders, training loops, examples, and
      a shared graph IR for execution, inspection, verification, and
      import/export.
    </p>
  </section>
  <section>
    <h3>Semantics</h3>
    <p>
      Finite-precision semantics, executable IEEE-style Float32 models,
      autograd, optimizer support, and explicit runtime agreement boundaries.
    </p>
  </section>
  <section>
    <h3>Verification</h3>
    <p>
      IBP/CROWN-style bounds, certificate checkers, VNN-COMP-style bundles, Bug
      Zoo contracts, 3D geometry certificates, and optional CUDA/native runtime
      paths with documented trust boundaries.
    </p>
  </section>
</div>

The README example is the shortest entry point; the Guide and Examples pages carry the longer
walkthroughs.

  </div>
</article>

</div>

---
title: Updates
---

<nav class="timeline-nav" aria-label="TorchLean update timeline">
  <a href="#november-2026-proved-gradients">Proved gradients and trees</a>
  <a href="#november-2026-operator-table">One operator table</a>
  <a href="#october-2026-one-tape">One tape and MoE</a>
  <a href="#october-2026-lean-435">Lean 4.35 and numerical fixes</a>
  <a href="#october-2026-custom-computations">Custom computations</a>
  <a href="#september-2026-libtorch">LibTorch backend</a>
  <a href="#september-2026-floatlib">FloatLib and precision</a>
</nav>

<div class="updates-timeline">

<article class="update-card" id="november-2026-proved-gradients" markdown="1">
  <div class="update-date">November 2026</div>
  <div class="update-body" markdown="1">

## Proved Operator Gradients and Recursion over Trees

Over the reals, the operators with a fixed number of inputs have 166 value inputs, and the gradient
of each has a certificate. At the points its hypotheses name, the stored backward is the adjoint of
the Fréchet derivative of the forward in all the value inputs together. `cat` has
`cat_backward_eq_adjoint`, and at kinks, such as `relu` at zero or `max` at a tie, a `convention`
certificate states the stored backward. The `op_deriv` tactic proves the same statement for a
backward pass written by hand: `fun_prop` composes the pullbacks of the operators, and a wrong
backward fails with the equations the tactic couldn't close. For a custom computation,
`Program.tangent` and `Program.gradient` derive tangent and gradient programs that run on the CPU
and the GPU, and over the reals they compute the derivative and its adjoint where the evaluation is
regular. None of these theorems bounds rounding. The [autograd proofs
chapter]({{ '/blueprint/Verification-and-Certificates/Autograd-Proofs/' | relative_url }}) runs the
tactic on softplus.

Recursive functions over our own inductive types run on the GPU. A type that derives
`Sig.Inductive` is stored by levels, and `Fold.of` turns a structurally recursive function into a
fold that launches one kernel per level, with a proof that the fold computes the function.
Hylomorphisms expand their calls on the GPU, and unfolds build trees in device memory. Inside
kernels, checked tables give FloatLib's correctly rounded `exp`, `expMinus1`, `log` and `logPlus1`
in formats of at most 20 bits. The [custom computations
example]({{ '/examples/custom-computations/' | relative_url }}) folds a tree and gives our
measurements with their conditions.

  </div>
</article>

<article class="update-card" id="november-2026-operator-table" markdown="1">
  <div class="update-date">November 2026</div>
  <div class="update-body" markdown="1">

## One Operator Table, Attention Rounding, and Library Bugs

CPU tensors and CUDA buffers share one autograd tape. Both record operators from `TorchLean.Op`,
the same table used by the verifier's graph lowering. It covers elementwise functions, gathers,
scatters, sorts, convolution, normalization, attention and losses. Every operator carries its
attributes and typed ports. For the supported scalar capabilities, `Op.rule?` supplies the forward
map, pullback and prepared node. Each recorder has its own entry point (`Tape.apply` for CPU tensors,
`LibTorch.Tape.apply` through the operator's entry in `LibTorch.Kernel.ofOp` for CUDA buffers, and
`IR.record` in the lowering). On the CPU, a node's pullback computes
contributions only for the parents that need a gradient. A CUDA entry calls LibTorch kernels, and
no theorem relates them to the rule. No theorem relates the recorded graph to the program it was
recorded from either.

Ports also carry indices and Booleans, so `sort`, `topk`, `argmax` and the comparisons return them,
and `gather`, `scatter` and `select` take them. Configured formats compute the inverse
trigonometric functions correctly rounded. Both importers (the `torch.export` capture and the Jaxpr
importer) translate a framework operator onto a registered one for the attributes, element types
and inputs the capture accepts, and a configuration the capture refuses fails at capture time. The
[Registered Operators
chapter]({{ '/blueprint/Semantics-and-Graphs/The-Operator-Table/' | relative_url }}) lists the
operators by family, along with the framework operators the importers refuse, and the [PyTorch
interop
notes](https://github.com/lean-dojo/TorchLean/blob/main/TorchLean/NN/PyTorch/README.md)
record, operator by operator, where the import follows a framework kernel and where it differs from
eager PyTorch.

Every operation node of `IR.Graph` holds a registered operator, and a node can have several outputs.
The graph also records its inputs and constants. The `torch.export` capture and the Jaxpr importer
write typed programs over these operators in one JSON format, `torchlean.ir`, which `IR.load`
checks and loads. Interval propagation runs on the graphs. For scalars with lawful endpoint operations,
`TorchLean.CROWN.Op.runIBP_encloses` proves that its boxes enclose a successful real run
of the graph (not a native run). The [IR
chapter]({{ '/blueprint/Semantics-and-Graphs/The-Canonical-Graph-IR/' | relative_url }}) explains
the graph and how it runs.

FloatLib's `roundoff` tactic applies an operator's proved bound wherever that operator occurs, so a
composite route such as two-pass attention gets its bound as a chain of its operators' bounds. It
reads tensor entries through checked storage laws. Native `Float32` has no numerical rule in that
analysis.

We also bounded attention by hand. Over the reals, FlashAttention's online softmax gives the
two-pass result for every block partition and mask, and `onlineAverage_error` bounds its rounding
in every IEEE format, with gradual underflow. Since the bound takes the scores as inputs, it leaves
out the rounding of the score product. At 4096 keys in blocks of 128, with binary32 precision,
scores of magnitude at most $2^{23}$ and values of magnitude at most one, the output is within
`2.9e-5` of the softmax average. For a pre-normalization transformer block at GPT-2-small sizes, a
theorem bounds the binary32 block within `0.538` of the exact block per token in Euclidean norm.
Its hypotheses are bounds on the exact stages and the weights, its stores are unrounded, and its
rounding has no upper exponent bound, so it describes executable binary32 only when nothing
overflows. It bounds the rounding schedule `Arithmetic.block`. The executed op-table block
normalizes and adds biases differently, and the bound does not cover it. See the
[numerical runtime example]({{ '/examples/numerical-runtime/' | relative_url }}) for the
`roundoff` theorems.

We wrote fifteen formulas from PyTorch, JAX, Transformers, TRL, Stable-Baselines3, diffusers,
PyTorch Geometric, TensorFlow Probability, torchvision and xLSTM with TorchLean's scalar
operations. Each comes with an input on which the kernel checks that it fails in an IEEE format,
and with a stable formula that doesn't fail there. The [Bug
Zoo]({{ '/examples/bug-zoo/' | relative_url }}) points to them. They check the formulas as written
in Lean, not the installed libraries.

  </div>
</article>

<article class="update-card" id="october-2026-one-tape" markdown="1">
  <div class="update-date">October 20, 2026</div>
  <div class="update-body" markdown="1">

## One Tape, One Training Path, and Mixture of Experts

CPU and CUDA use the same autograd tape structure and dense backward traversal. CUDA training
also uses a sparse traversal that releases intermediate gradients as it goes. Trainers record on
the device selected by `device`; the `autograd` transforms record typed graphs for gradients,
JVPs, VJPs and HVPs.

`Trainer.RunConfig` extends `Session.Config`, so a parsed runtime configuration can supply the
device and seed when we create a trainer:

```lean
let trainer := Trainer.create model
  { runtime with optimizer := optim.adam { lr := 0.03 }, objective := .mse }
```

Learning-rate schedules live in `Trainer.Scheduler`, and training logs record the run's settings
as an `ExperimentLog`. The [run-configuration chapter]({{ '/blueprint/Runtime___-Autograd___-and-Interop/Choosing-How-A-Model-Runs/' | relative_url }})
explains device selection, arithmetic and train/evaluation mode.

Mixture of experts is new: `nn.moe` and `nn.moeMLP`, a `topKMask` IR operation with CPU and native
CUDA implementations, and PyTorch MoE import. Configured-precision GPU arithmetic does not
support this mask operation. The spec, `TorchLean.MoE`, comes with a proof that over `ℝ` the
dense layer equals Mixtral's sparse block.

Routing follows the same backend selection and audit as other operations. PyTorch captures
retain finite-score and strict-cutoff-gap assumptions as unchecked metadata for partial top-k
selection; a sample comparison does not prove those assumptions for other inputs.

`CROWNNode.check` checks node certificates with optional ReLU phases. The phased α-CROWN
runner uses those phases without β multipliers or branch-and-bound search. The verifier's
default method is `crown`, which runs directed CROWN on a rounded scalar. See the
[verification example]({{ '/examples/verification/' | relative_url }}) for the commands and
certificate assumptions.

  </div>
</article>

<article class="update-card" id="october-2026-lean-435" markdown="1">
  <div class="update-date">October 2026</div>
  <div class="update-body" markdown="1">

## Lean 4.35 and Numerical Fixes

TorchLean, mathlib and the documentation tools use Lean 4.35. We checked calculations near zero,
at very large magnitudes and across the CPU/GPU boundary. Softplus keeps small negative-tail
values, log-cosh stays finite for large residuals, and the log and reciprocal gradients avoid
unnecessary overflow in intermediate expressions. Integer-to-binary casts report overflow even
when directed rounding returns a finite value, using FloatLib's status classifier.

Power derivatives keep the original inputs when an intermediate overflows or underflows. For a
positive base, this includes derivatives where both the base and exponent vary. The recovery uses
FloatLib enclosures and accepts a result only when both bounds round to the same value, including
every coefficient of a nested dual. It has a bounded refinement budget; unsupported domains and
inconclusive bounds keep the ordinary result.

Attention backward retains the original contractions through softmax, including the biased and
dropout paths, rather than trying to recover a finite gradient from an already infinite tensor.
The compound operation has real-valued derivative and JVP/VJP adjoint proofs. Native recovery can
transfer finite operands to CPU; these proofs do not verify the external GPU kernels.

Numerical certificates now cover the reference scaled matrix product, including recovery when the
unscaled contraction overflows. The scale is a checked scalar operand. The range rule has a
real-enclosure proof, and binary32 replay checks the concrete execution separately. Native
implementation-defined reduction policies are still rejected by this rule.

Typed graphs now count random draws rather than tensor nodes, matching the eager seed stream.
This changes masks from earlier seeded typed-graph runs. Verifier IR lowering currently rejects
random draws instead of substituting a different mask; eager and typed-graph dropout remain available.

Rounded CROWN uses directed bounds and starts propagation from the requested input box.
ODE corridor endpoints and time bounds round outward. Eager single-output backward passes skip
unreached nodes, so an unused singular calculation cannot contaminate a gradient. Reached nodes still
run their backward rules, even when their cotangent is zero.

The proof updates include binary32 reduction bounds over the tree's own representable operands,
an AdamW denominator margin that admits a zero-gradient first step, and replay-buffer theorems
about stored contents and FIFO order. ODE corridor theorems require a real-valued interpretation
with stated enclosure and differential-equation hypotheses; discrete binary32 samples do not
establish those hypotheses.

The dense backward sweep over a tape lowered from a proof-carrying graph (`Tape.GraphData.toTape`)
has a Fréchet-derivative theorem over ℝ: `Tape.Graph.backwardDenseAll_eq_fderiv` connects the
lowered tape's result to the derivative of the graph's real semantics. No runtime path sweeps a
lowered tape. The CPU eager tape is linked node by node
(`TorchLean/NN/Tape/Apply/Correctness.lean`). IR shape soundness and CROWN enclosure theorems
describe their respective Lean models. Applying these results to rounded arithmetic or LibTorch
execution additionally requires the numerical and native-runtime assumptions in the
[proof-systems chapter]({{ '/blueprint/Verification-and-Certificates/Proof-Systems-Beyond-Bounds/' | relative_url }}).

  </div>
</article>

<article class="update-card" id="october-2026-custom-computations" markdown="1">
  <div class="update-date">October 2026</div>
  <div class="update-body" markdown="1">

## Running Our Own Tensor Calculations

We wanted to write our own tensor calculations in Lean without writing them again in CUDA.
We can give an ordinary Lean function a tensor and choose the device when we run it:

```lean
import TorchLean.NN.Kernel.Function
open TorchLean

def square := fun (x : Float32) => x * x
def input : Tensor Float32 [3] := [1, 2, 3]

#eval square.run input
-- [1.000000, 4.000000, 9.000000]

def onGpu := square.run input (device := .cuda)
```

CPU is the default and evaluates the original Lean function. GPU supports native FP32/FP64 and
FloatLib's configured binary formats, including binary128. Where hardware arithmetic matches the
format, we use it. For wider or custom formats, we store each number in several integers and do
the arithmetic on the GPU, keeping the extra precision.

We can pass parameters, branch on a value, or combine two tensors with `f.zip`. For calculations
such as row sums and repeated updates, `Program.of` lets us read tensor entries and write ordered
loops. It also recognizes recursive functions whose parameters stay fixed and whose step count
decreases by one. CPU-only scalar types or formats keep their precision on CPU with a diagnostic.
Unsupported code for a GPU-capable type and execution failures return errors.

Lean checks that the expression we compile agrees with the function we wrote. For native FP32/FP64,
we also prove that lowering and source generation preserve that expression, assuming the stated
arithmetic and memory behavior. That proof doesn't verify NVIDIA's compiler or hardware. For
configured formats, we compare GPU results against FloatLib; we haven't proved the CUDA arithmetic
implementation correct. Standard model operations use LibTorch.

### Ask for a gradient

We keep the function definition and turn on recording when we call it:

```lean
#eval do
  let result ← square.run input (grad := true)
  IO.println result.value
  IO.println (← result.backward)
-- [1.000000, 4.000000, 9.000000]
-- [2.000000, 4.000000, 6.000000]
```

This uses TorchLean's tape. Whole-tensor functions such as sums and rank-two matrix products use
the same call. Shapes and storage come from the input; tensors captured by the function stay
constant. The frontend checks that the recorded expression matches the function.

The GPU tape keeps binary32 for `Float32` and binary64 for `Float`, including saved values,
gradients, optimizer state and checkpoints. Configured binary arithmetic records on that same
tape, retaining complete words for saved values and gradients. We can record addition,
subtraction, multiplication, division, negation and scalar traversal, including binary128.
Native model operators and optimizer checkpoints require their native dtypes.

Scalar conditionals record the chosen branch, and fixed-parameter recurrences record each step.
GPU branch decisions synchronize to Lean; arithmetic and backward calculations stay on GPU.
Recording keeps the intermediate values needed for backward and launches operations per step;
forward-only execution can use one compiled loop instead. The count must be independent of the
input we're differentiating. At a branch boundary, the recorded gradient need not be a derivative.
Arbitrary indexed programs need their own backward calculation.

Try the [custom computations example]({{ '/examples/custom-computations/' | relative_url }})
for gradients, parameters, row sums, recursive updates and binary128 arithmetic. The
[guide chapter]({{ '/blueprint/Runtime___-Autograd___-and-Interop/Custom-Tensor-Computations/' | relative_url }})
explains the precision choices and proofs.

  </div>
</article>

<article class="update-card" id="september-2026-libtorch" markdown="1">
  <div class="update-date">September 2026</div>
  <div class="update-body" markdown="1">

## Running CUDA Through LibTorch

LibTorch gives us tensor operations from C++, while we keep the model definitions and autograd
tape in Lean. Matrix products, convolutions, pooling and reductions call ATen through
`csrc/libtorch/torchlean.cpp` and `csrc/libtorch/conv_pool.cpp` under a no-grad guard. TorchLean
owns differentiation and the saved buffers used by its local reverse rules.

Attention composes matrix products and softmax in Lean, including its local VJP, and the tape owns
Q/K/V and saved probabilities. Full score matrices use quadratic sequence memory. Spectral models
likewise compose FFT, frequency mixing and inverse FFT through the numerical primitives.
LibTorch, its compiler, CUDA libraries and hardware remain outside Lean's proof kernel.

Select the SDK with `-Klibtorch_home=PATH` or `TORCHLEAN_LIBTORCH_HOME`; a CUDA-enabled pip
PyTorch installation works as the SDK root. The [SDK notes](https://github.com/lean-dojo/TorchLean/blob/main/csrc/libtorch/README.md#torch-versions)
record the tested versions and GPU checks; other SDK versions need their own CUDA suite results.
Build with `-Kcuda=true`, then select `--device cuda` or `--device gpu`. An unavailable GPU is an
error. The default CPU build needs no SDK or toolkit.

The backend reports name the CUDA comparisons we keep in the test suite. For other GPU operations,
we rely on LibTorch's implementation. The default GPU profile allows that assumption; the strict
checked profile rejects it. See the [installation guide]({{ '/installation/' | relative_url }})
for platform requirements and backend selection.

  </div>
</article>

<article class="update-card" id="september-2026-floatlib" markdown="1">
  <div class="update-date">September 2026</div>
  <div class="update-body" markdown="1">

## Choosing Scalar Precision with FloatLib

We wanted precision to be a choice users make in their tensors and models. FloatLib supplies
the scalar arithmetic and rounding theory. `ExecFloat.Binary 8 23` gives the familiar
binary32 format; `ExecFloat.Binary 15 112` gives binary128. Valid custom widths use the same API.

A wider type matters only if the extra digits reach the computation. For example, binary128
can retain $1+2^{-100}$, while binary64 rounds it to 1. The
[tensor chapter]({{ '/blueprint/Building-Models/Tensors-That-Remember-Their-Shapes/' | relative_url }})
constructs that parameter directly from a rational, keeps it through a typed affine model,
and compares its forward value, input derivatives and `nn.sgdStep` update with an elementary
calculation. This uses software arithmetic on the typed CPU path. `trainer.openTyped` also retains
the chosen scalar in supervised samples, predictions, loss, reports and checkpoints. These typed
sessions run on CPU and reject custom backend profiles. Their results carry no lowered model, so
`Result.verify` on them raises an error.
Initialization and optimizer settings enter through `Float`. The ordinary trainer's `ieee`
option selects binary32.

The numerical proofs distinguish rounded trees from exact accumulation followed by one final
rounding. Their intermediate values can differ, so an error theorem for one does not justify the
other. FloatLib's native Float32 addition and subtraction theorems require finite operands;
its square-root theorem covers every input through NaN-canonicalizing native export. These
logical results remain separate from compiler and hardware conformance.

FloatLib also supplies interval, affine-quantization and reduction-tree mathematics. Users can
choose the grouping and input order of a reduction tree, use a real quantization scale with their
own rounding rule, or construct interval bounds with binary, decimal or posit endpoints.
TorchLean supplies the tensor lifts and backend contracts. Its numerical graph certificates use
binary32 endpoints.

The [floating-point chapter]({{ '/blueprint/Floating-Point-and-Native-Boundaries/Floating-Point-Semantics/' | relative_url }})
explains the scalar API and numerical proofs, including `interval` bounds for real powers,
`Real.logb` and inverse hyperbolic functions.

  </div>
</article>

</div>

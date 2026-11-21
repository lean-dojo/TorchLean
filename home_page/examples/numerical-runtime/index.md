---
title: Numerical Runtime Certificates
---

Pick one input to a two-layer MLP. Before we run it, can we give an interval for each result,
including the values in the hidden layer? We'll compute those intervals from the graph, run
the same operations in binary32, and check each intermediate value against its interval.

## Build the Example

From the TorchLean checkout, run the certificate example.

```bash
scripts/lake.sh build TorchLeanExamples
```

The example also defines a corrupted range for the checker to reject.

## The Model

Here's the model we'll check:

```text
input [1,2]
  -> matmul [2,3]
  -> add bias [1,3]
  -> ReLU
  -> matmul [3,1]
  -> add bias [1,1]
```

Weights and biases are ordinary constant nodes. No `MLP` tag reaches the checker, which sees ten IR
nodes: input, constants, matrix multiplications, additions, and ReLU. So the same path works for a
larger architecture assembled from supported operations.

You'll find the source in `TorchLeanExamples/DeepDives/Floats/GraphNumericalCertificate.lean`,
where `MLP.graph`, `MLP.sources`, `MLP.payload`, `MLP.certificate`, and `MLP.replay` define the
graph, inputs, certificate, and replay used below.

## What Happens During the Run

### 1. Check operation coverage

`rangeTransfer?` gives each supported `TorchLean.Op` constructor a rule for computing its output
range from its input ranges. TorchLean checks coverage before propagation starts. If a rule is
missing, it reports the node id and operation name and stops.

### 2. Propagate source ranges

We supply binary32 intervals for the input and each parameter tensor. These intervals are our
assumed ranges for their values. Matrix multiplication uses outward-rounded products and a
declared accumulation order. Addition uses directed endpoints. ReLU restores the nonnegative
lower bound of the hidden activation.

The resulting range trace contains one row for every node. A row records the node id, operation,
input enclosures, and derived output enclosure.

### 3. Select kernel capsules

For each operation, the kernel planner selects a kernel capsule (a description of an
implementation) and records the provider, device, layout requirements, forward and reverse-mode
ownership, reduction policy, and trust classification. We'll use the checked portable CPU profile
because its fixed-left matrix accumulation matches the canonical tensor semantics.

A range rule must support the selected numerical policy. For example, a rule for left-to-right
accumulation can't be used for a provider that advertises a different reduction order. The audit
records the selected plan, but it doesn't prove that native kernels executed it.

Scaled matrix products have a separate rule because multiplying only after the dot product can
overflow before a small scale brings the result back into range. During replay, the scale (a scalar
graph operand) is checked against its source range. The reference policy allows the ordinary left
fold or recovery from the original finite coefficients, but an implementation-defined native
reduction doesn't satisfy that policy. As with the other rules, real enclosure evidence and the
concrete binary32 replay are separate checks.

### 4. Check the certificate

`generateChecked` generates a certificate and checks it immediately. That check compares the
profile name, recomputes the ranges from the supplied graph and source assumptions, and compares
the resulting ranges and kernel audit with the certificate. A mismatch is rejected. The returned
checked value also stores the graph used for these calculations.

These are consistency checks, not a unique fingerprint of the graph or of the range transfers.
A change that leaves all the comparisons unchanged need not be rejected.

### 5. Execute binary32 and replay every node

`executeIEEE32` evaluates the stored graph with TorchLean's bit-level binary32 semantics. Every
intermediate tensor must remain finite and lie in the enclosure regenerated for its node. The run
rejects NaN, infinity, malformed payloads, missing constants, and out-of-range intermediates.

A successful replay means that this concrete binary32 execution passed the stored graph
certificate. It doesn't, by itself, turn an interval endpoint calculation into a theorem about
every real input. A `NumericalCertificate.RealEnclosure` supplies inclusion evidence for one real
execution of the same graph, with its own input, payload, and node trace. Paired with the checked
replay, it yields pointwise error bounds from the shared interval widths. A guarantee over an entire
source region would need this evidence for every admissible real input. This example doesn't
construct that evidence.

## Error Bounds from the Code

A small gradient makes the choice of guarantee matter. If its value is about $2\times10^{-9}$,
returning zero can still pass an absolute-error tolerance of $10^{-7}$.

`roundoff` builds numerical-error proofs from the existing Lean computation. Its analysis is
FloatLib's, and `TorchLean.Verify.RuntimeApprox.Roundoff` registers TorchLean's tensors with it:
it follows tensor operations and reductions into their scalar arithmetic, and uses FloatLib's
proved numerical rules to compose the errors. We can apply it to an autograd operation's backward
function:

```lean
import TorchLean.Verify.RuntimeApprox.Roundoff
import TorchLean.NN.VJPOp
import TorchLean.Tensor.ConfiguredBinary

open TorchLean
open FloatLib.Floats

roundoff theorem gradient_error :=
  fun {shape : TorchLean.Shape} (input upstream : Tensor Binary32 shape) (index : shape.Coord) =>
    ExecFloat.Binary.toReal
      (((Tensor.full shape (2 : Binary32) * input) * upstream : Tensor Binary32 shape) index)
```

That declaration generates a conditional theorem for every coordinate of the tensor. Its statement
contains the real reference, the error budget and the conditions the proof needs: computed values
that are finite and inputs within range. Applying it requires proving those conditions. The
operation code stays polymorphic over the scalar type, and the numerical laws retain their format
parameters. Here we choose binary32, and the underlying IEEE rules state their format and
finiteness assumptions.

On an IEEE format the exponential, the logarithm, `expm1`, `log1p` and `Context.expMul` are
FloatLib's correctly rounded functions. Each call rounds its real result once (half an ulp), and
the conditions keep that result below the overflow threshold. Sine, cosine and the hyperbolic
functions use FloatLib's rational enclosures instead, so their budget contains the distance from
the computed value to the enclosure and is known only once the value is computed. The configured
`asin`, `acos`, `atan` and `atan2` are correctly rounded too, but the analysis has no rule for the
`quietNaN` their code applies, so their forward routes are unsupported.

### Absolute and mixed error

`roundoff theorem` states an absolute bound, `|computed - reference| ≤ budget`.
`roundoff mixed theorem` states instead that the computed value is finite and has a mixed error
`MixedError computed reference r a` (that is, `|computed - reference| ≤ r * |reference| + a`)
against the exact computation on the decoded inputs. Where it can, the analysis derives the
finiteness of intermediate values from range conditions on the inputs and budgets, and those
conditions become the theorem's premises. A mixed bound keeps the leading digits of a small exact
value whenever `a` is much smaller than `|reference|`. Sums of terms of one sign keep a relative
error weighted by their magnitudes, and exact operations (such as a finite `x - x` or a dot product
with a one-hot vector) add no error at all. But cancellation leaves an absolute error.

In `TorchLean/Verify/RuntimeApprox/Roundoff/Softmax.lean`, `roundoff` infers mixed bounds for the
stable softmax, the stable log-softmax and its VJP, for every IEEE format and every vector length,
and the theorems there discharge the conditions of the last two in binary32 and bfloat16. At the
logits $[9,-9,-9]$ the VJP keeps the small target gradient with a relative error of a few units of
roundoff, so the computed gradient is not zero.
`TorchLeanExamples/DeepDives/Floats/RoundingBugs.lean` proves, for every length and under
conditions on the forward computation, that the cross-entropy gradient of a one-hot target is
negative at the target and nonnegative elsewhere.
Cross-entropy also uses a second shift when the individual tail exponentials underflow but their
combined mass can still be represented. For example, binary16 logits $[9,-9,-9]$ retain a target
gradient of $-2^{-24}$; the one-level formula returns zero. The CPU result is checked by a Lean
theorem, and the CUDA implementation has separate runtime checks in binary32 and binary64.

If the inner sum in the configured two-level formula overflows, it retries with the terms divided
by powers of two. `Tensor.var` uses the same helper when its two-pass result is not finite. For
binary16 inputs $[181,-1447/8]$, this recovers a variance of $32768$ from an overflowing sum of
squares. Both operations keep every already finite result unchanged. Their roundoff rules
currently carry the bounds of the unscaled computation under its finiteness conditions; the
recovered cases have separate checked examples in `RoundingBugs.lean`.

[`LibraryBugs.lean`](https://github.com/lean-dojo/TorchLean/blob/main/TorchLeanExamples/DeepDives/Floats/LibraryBugs.lean)
applies the same analysis to formulas from other libraries, including soft-margin loss, normal KL
divergence, approximate GELU and label smoothing. For each it checks a failing input with the
kernel and states mixed bounds for a stable formula with the same real value. The
[Bug Zoo]({{ '/examples/bug-zoo/' | relative_url }}) links the further examples in probability,
losses and recurrent models. These checks concern the formulas written in Lean; they don't prove
that an installed library executes those formulas.

### Functions, shapes and layers

An operator may take a function, such as the fields of the specifications that
`TorchLean.VJPOp.compose` composes. Its routes are analyzed under a contract for that function, a
hypothesis stated like a generated theorem: at finite arguments in a domain, the function's
result is finite with mixed error `(u, η)` against a real model. `roundoff` uses such a hypothesis
as it uses a generated theorem, and the result holds for every function that meets its contract.

Softmax, log-softmax and the log-softmax VJP along any axis are read as their vector kernels on the
fiber through a coordinate, so their theorems hold at every shape and axis
(`TorchLean/Verify/RuntimeApprox/Roundoff/SoftmaxAxes.lean`). The online schedule of softmax and
attention, FlashAttention's pass over blocks of keys, is read through the hand-proved bounds that
`TorchLean/Verify/RuntimeApprox/Roundoff/Rules/OnlineSoftmax.lean` registers as rules. The guide
chapter [Rounding Error In
Attention]({{ '/blueprint/Verification-and-Certificates/Rounding-Error-In-Attention/' | relative_url }})
explains the schedule and its bounds. The general data-movement operators
`broadcast`, `permute`, `reshape`, `slice`, `pad`, `cat`, `flip` and `gather` are read through the
input coordinate each output coordinate selects, at any axes, and add no error. Reductions and
contractions use the order recorded by the operator. Scatter updates follow their specified fold.
Log-sum-exp, the variance and the
cosine loss over any axes, cross-entropy of probabilities and of logits, and layer, RMS and batch
normalization over their axes are analyzed the same way, by the coverage theorems of
`TorchLean/Verify/Coverage/TensorComposites/` (`Softmax.lean`, `Loss.lean` and
`Normalization.lean`). Convolution and pooling are analyzed
for any spatial rank, kernel, stride, padding and dilation, and max pooling is exact
(`TorchLean/Verify/RuntimeApprox/Roundoff/Convolution.lean`). The cross-entropy gradient with mean
reduction has a mixed bound against the exact gradient, the softmax minus the one-hot target
(`TorchLean/Verify/RuntimeApprox/Roundoff/CrossEntropy.lean`). The distillation gradient is exactly
zero at equal student and teacher logits, for every IEEE format, shape and axis, when the logits,
the softmax weights and the mean scale are finite
(`TorchLean/Verify/RuntimeApprox/Roundoff/Distillation.lean`). `roundoff` also derives conditional
bounds for scaled vector normalization and the public RMSNorm and LayerNorm forward functions,
retaining their reduction order and precision conversions.

In `TorchLean/Verify/RuntimeApprox/Roundoff/Operators.lean`, the log-softmax theorems show why the
choice matters at a single input. For logits $[0,-20]$ and upstream gradient $[1,0]$, the stable
backward code retains a gradient of about $2.06\times10^{-9}$ where the conventional subtraction
returns zero. We compare the public backward operation with the real gradient at the original
logits, $\exp(-20)/(1+\exp(-20))$. In binary32 the stable result has absolute error at most
$10^{-16}$ and relative error at most $10^{-7}$, and is strictly positive. Lean evaluates the
binary32 result exactly, and FloatLib's exponential enclosure supplies the real-number bounds.

### Choosing a reduction order

In binary32, there is room for every integer through $2^{24}$. Above that, adjacent values
are two apart. Add three ones to $2^{24}$ and the result depends on when those ones meet.

```lean
import TorchLean.Tensor.ConfiguredBinary
import TorchLean.Tensor.Reduce.Order

open TorchLean FloatLib.Floats

def samples : Tensor Binary32 [4] :=
  Tensor.Rep.ofArray #[16777216, 1, 1, 1] (by decide)

#eval ExecFloat.Binary.toRat? (Tensor.sumIn .leftFold samples)    -- some 16777216
#eval ExecFloat.Binary.toRat? (Tensor.sumIn (.blocked 2) samples) -- some 16777218
#eval ExecFloat.Binary.toRat? (Tensor.sumIn .pairwise samples)    -- some 16777220
```

The exact sum is $16{,}777{,}219$. The left fold loses each one to rounding. The blocked
sum keeps the last two ones together. The pairwise tree adds the three ones before
combining them with $2^{24}$, so the final tie rounds to $16{,}777{,}220$. The ordered-sum
bounds use the tree's depth, along with premises on the IEEE format and finite results.
See [the ordered-sum theorems](https://github.com/lean-dojo/TorchLean/blob/main/TorchLean/Verify/RuntimeApprox/Roundoff/Rules/ReduceOrder.lean).

This choice also lives in the IR. A version 5 reduction record can end in
`{"order": "pairwise"}`; an explicit tree has to use the start value and every reduced
entry exactly once. Imports from PyTorch, JAX, and ONNX use left folds for these
operators. Check the backend's arithmetic schedule before applying a bound for a
particular order to a native run. See [the IR operator reference](https://github.com/lean-dojo/TorchLean/blob/main/TorchLean/NN/IR/README.md#one-shape-rule-per-operation).

### Checks and assumptions

`scripts/lake.sh lint` fails unless every theorem of the modules under `TorchLean/`,
`TorchLeanExamples/` and `TorchLeanTest/`, the runtime-approximation proofs among them, depends on
at most `propext`, `Classical.choice` and `Quot.sound`. The file that runs the check,
`scripts/checks/Library.lean`, leaves out the root modules of the executables.

`#roundoff_coverage` records analysis attempts for the forward values, tangents and pullbacks of
registered operators, together with CPU tape builders and tensor composites. Each route is analyzed
for configured IEEE arithmetic in absolute and mixed modes, and for native `Float32` and the rounded
reals `FP32`. A route is `proved` when its theorem has no premise and its reference and budget decode
only inputs, `conditional` when it has premises or caveats, `unsupported` when it meets an operation
without a numerical rule, and `failed` when analysis stops otherwise. Failed and unsupported routes
supply no error bound.

A caveat marks a budget or reference that decodes computed values, such as a reference that
branches on a computed comparison, or a proof that uses an enclosure in place of a rounding theorem.
The theorem holds, but it is not an a priori bound. Graph lowerings that build whole tapes, and
LibTorch tape builders that run device kernels, are outside this analysis.

The coverage check in `scripts/lake.sh lint` requires registration of public operator specifications
and CPU tape constructors under `TorchLean/`, equation lemmas and configured-scalar analyses for the
operator table, and certificates that speak about their operators' routes. It compares outcomes,
references, budgets, premises and caveats with `scripts/checks/roundoff_baseline.jsonl`; both
regressions and unrecorded changes fail the check. The
[runtime-approximation README](https://github.com/lean-dojo/TorchLean/blob/main/TorchLean/Verify/RuntimeApprox/README.md)
explains coverage and lists the remaining failed analyses.

A route whose analysis exceeds the heartbeat budget of one declaration is analyzed once more with
the generated theorems of the operations it calls as rules, each stated against real references
for its inputs, so that the route's bound is a chain of its parts' bounds. Its outcome names the
theorems it was composed from. The two-pass forward of `Op.attention` gets its theorem this way.
The guide chapter
[Registered Operators]({{ '/blueprint/Semantics-and-Graphs/The-Operator-Table/' | relative_url }})
lists the operators of the table by family, with their CUDA entries, certificates and bound
propagations.

These are proofs about the Lean computations and their selected scalar implementations. The
analysis reads a tensor entry through TorchLean's `Storage` class only with laws that give each
method of the class a defining equality through the array observation `Storage.toArray`, such as
`Storage.get_eq_toArray_getElem`. The laws do not show that a generated reference agrees with a
separately written specification, and assumptions about storage types and index data remain.
FloatLib states them in the trust assumptions of
`FloatLib/Tactic/Roundoff.lean`. Real derivative theorems
establish what the backward formula differentiates. Correspondence with a native kernel needs its
own evidence. A missing numerical rule stops the analyzer, and its error names the expression it
stopped at.

## Forward, Backward, and the Optimizer

To compose forward, backward, and optimizer bounds across a graph, we use the theorems for
`RevGraph`. Each node carries:

- exact and rounded forward functions
- exact and rounded vector–Jacobian products (VJPs), which propagate output gradients to inputs
- a forward error transformer and proof
- a VJP error transformer and proof

`RevGraph.eval_approx` composes the forward bounds. `RevGraph.backprop_approx` traverses the same
graph in reverse and includes rounding from gradient accumulation. A parameter gradient can then be
passed to any `OptimizerContract`:

```lean
import TorchLean.Verify.RuntimeApprox.NF.EndToEnd

#check TorchLean.NF.eval_approx_graphData
#check TorchLean.NF.backprop_approx_graphData
#check TorchLean.NF.update_approx_graphData
#check TorchLean.NF.traceTrainingStep
```

SGD, momentum SGD, and AdamW use that one optimizer interface. AdamW contributes error budgets for
bias correction and a lower bound for the complete denominator, including epsilon. Its square-root
estimate remains valid when the second moment is zero. `traceTrainingStep` collects the forward,
backward, parameter-gradient, parameter-update, and optimizer-state budgets in a noncomputable
report, while the preceding theorem proves the error bounds.

The MLP example checks the forward calculation. `roundoff` can also analyze typed-graph forward and
VJP computations from their Lean code (the regressions check this on a graph with a single addition
node). The end-to-end optimizer theorems above use a supplied `RevGraph` with proved local VJPs and
error bounds. A `TorchLean.IR.Graph` is checked by its forward replay, but nothing constructs that
proof-bearing reverse graph from it automatically.

## Adding Another Architecture

To try this with another model, we first build it as a graph of registered operators. We don't
need a new architecture wrapper. `TorchLean.NumericalCertificate.requireCoverage graph` checks
whether all its operations have range rules, and its error names the nodes that don't.

Covered graphs use the same certificate and replay APIs, subject to the shape, finite-range,
payload, and backend-policy checks described above. If one operation is missing, add its executable
transfer to `rangeTransfer?` and prove the corresponding exact-real enclosure rule. The transfer
extends range generation, and the enclosure rule is needed before the generated range can support
`NumericalCertificate.RealEnclosure`. If execution needs another provider, give it an
`Implementation` that offers a `KernelCapsule` for each operator it runs (for example with
`Implementation.ofContracts` and a `KernelContract` per operator), and add it to the profile with
`Profile.withImplementations`. Then the planner offers that implementation, under the profile's
availability and assurance policy, to every architecture that uses those operators.

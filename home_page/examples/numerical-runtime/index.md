---
title: Numerical Runtime Certificates
---

So now, let's take a small two-layer MLP and check its intermediate results as
we run it. We'll compute a range for each result, then use bit-level binary32 arithmetic
to check that the computed values stay within those ranges.

We'll represent our model with `NN.IR.Graph`, a graph of tensor operations and their dependencies.
The checker uses a range rule for each supported operation, such as matrix multiplication or ReLU.
Those rules can also be used in other models built from the same operations.

Later, we'll look at a separate graph representation, `RevGraph`, whose nodes carry proofs
of forward and gradient error bounds. The runnable certificate example does not automatically
construct those proofs.

## Run the Example

Let's run it first. From the TorchLean repository:

```bash
scripts/lake.sh exe torchlean numerical_certificate
```

The last two checks should report:

```text
  ok  two-layer MLP certificate
  ok  two-layer MLP IEEE replay
```

The example also demonstrates rejection of a corrupted range. The regression suite in
`NN/Tests/Verification/GraphNumericalCertificate.lean` checks duplicate contracts, changed registry
identities, invalid square-root inputs, and incompatible CUDA reduction policies. Run the complete
suite with `scripts/lake.sh test`.

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

Weights and biases are ordinary constant nodes. The checker is not given an `MLP` tag. It sees ten
IR nodes: input, constants, matrix multiplications, additions, and ReLU. This is why the same path
works for a larger architecture assembled from supported operations.

The source is
[`GraphNumericalCertificate.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/DeepDives/Floats/GraphNumericalCertificate.lean).
The definitions `Mlp.graph`, `Mlp.sources`, `Mlp.payload`, `Mlp.certificate`, and `Mlp.replay`
define the graph, inputs, certificate, and replay used below.

## What Happens During the Run

### 1. Check operation coverage

`GraphRangeRegistry` associates each supported operation with a rule for computing its output
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

The kernel planner selects a kernel capsule, a description of an implementation, for each operation.
It records the provider, device, layout requirements, forward and reverse-mode ownership,
reduction policy, and trust classification. We'll use the checked portable CPU profile because
its fixed-left matrix accumulation matches the canonical tensor semantics.

The selected numerical policy must be supported by the range rule. For example, a rule for
left-to-right accumulation cannot be used for a provider that advertises a different reduction
order. The audit records the selected plan; it does not prove that native kernels executed it.

Scaled matrix products have a separate rule because multiplying only after the dot product can
overflow before a small scale brings the result back into range. The scale is a scalar graph
operand, checked against its source range during replay. The reference policy allows the ordinary
left fold or recovery from the original finite coefficients; an implementation-defined native
reduction does not satisfy that policy. As with the other rules, real enclosure evidence and the
concrete binary32 replay are separate checks.

### 4. Check the certificate

`generateChecked` generates a certificate and checks it immediately. The checker compares the
profile and registry names, recomputes the ranges from the supplied graph and source assumptions,
and compares the resulting ranges and kernel audit with the certificate. A mismatch is rejected.
The returned checked value also stores the graph used for these calculations.

These are consistency checks, not a unique fingerprint of the graph or registry implementation.
A change that leaves all the comparisons unchanged need not be rejected.

### 5. Execute binary32 and replay every node

`executeIEEE32` evaluates the stored graph with TorchLean's bit-level binary32 semantics. Every
intermediate tensor must remain finite and lie in the enclosure regenerated for its node. The run
rejects NaN, infinity, malformed payloads, missing constants, and out-of-range intermediates.

A successful replay means that this concrete binary32 execution passed the stored graph
certificate. It does not, by itself, turn an interval endpoint calculation into a theorem about
every real input. A `ProvedRealEnclosure` supplies inclusion evidence for one real execution of the
same graph, with its own input, payload, and node trace. Paired with the checked replay, it yields
pointwise error bounds from the shared interval widths. A guarantee over an entire source region
would need this evidence for every admissible real input; this example does not construct it.

## Error Bounds from the Code

A small gradient makes the choice of guarantee matter. If its value is about $2\times10^{-9}$,
returning zero can still pass an absolute-error tolerance of $10^{-7}$.

`roundoff` builds numerical-error proofs from the existing Lean computation. It follows tensor
operations and reductions into their scalar arithmetic, and uses FloatLib's proved numerical
rules to compose the errors. We can apply it directly to an autograd operation's backward function:

```lean
import NN.Proofs.RuntimeApprox.Automation
import NN.Spec.Autograd.Ops
import NN.Spec.Core.FloatInstances

open TorchLean
open FloatLib.Floats

roundoff theorem gradient_error :=
  fun {shape : Spec.Shape} (input upstream : Tensor Binary32 shape) (index : shape.Coord) =>
    ExecFloat.Binary.toReal (Spec.squareOp.backward input upstream index)
```

This generates a conditional theorem for every coordinate of the tensor. Its statement contains
the real reference, error budget, and required range conditions. Applying it requires proving
those conditions. The operation code stays polymorphic over the scalar type, and the numerical
laws retain their format parameters. Here we choose binary32; the underlying IEEE rules state
their format and finiteness assumptions explicitly.

The regressions also derive conditional bounds for scaled vector normalization and the public
RMSNorm and LayerNorm forward functions, retaining their reduction order and precision conversions.

The [log-softmax proofs](https://github.com/lean-dojo/TorchLean/blob/main/NN/Tests/Verification/Roundoff.lean)
check why this matters. For logits $[0,-20]$ and upstream gradient $[1,0]$, the stable backward
code retains a gradient of about $2.06\times10^{-9}$ where the conventional subtraction returns
zero. We compare the public backward operation with the real gradient at the original logits,
$\exp(-20)/(1+\exp(-20))$. The stable result has absolute error at most $10^{-16}$ and relative
error at most $10^{-7}$, and is strictly positive. FloatLib's exponential enclosure supplies
the real-number bounds; Lean checks the concrete floating-point result and the rational comparisons.

These are proofs about the Lean computations and their selected scalar implementations. Real
derivative theorems establish what the backward formula differentiates; correspondence with a
native kernel needs its own evidence. Operator coverage is still growing, and a missing numerical
rule stops the analyzer.

## Forward, Backward, and the Optimizer

To compose forward, backward, and optimizer bounds across a graph, we use the theorems for
`RevGraph`. Each node carries:

- exact and rounded forward functions;
- exact and rounded vector–Jacobian products (VJPs), which propagate output gradients to inputs;
- a forward error transformer and proof;
- a VJP error transformer and proof.

`RevGraph.eval_approx` composes the forward bounds. `RevGraph.backprop_approx` traverses the same
graph in reverse and includes rounding from gradient accumulation. A parameter gradient can then be
passed to any `NumericalStepContract`:

```lean
import NN.Proofs.RuntimeApprox.NF.EndToEnd

#check Proofs.RuntimeApprox.NFBackend.eval_approx_graphData
#check Proofs.RuntimeApprox.NFBackend.backprop_approx_graphData
#check Proofs.RuntimeApprox.NFBackend.backprop_optimizer_update_approx_graphData
#check Proofs.RuntimeApprox.NFBackend.trainingStepTrace
```

SGD, momentum SGD, and AdamW use that one optimizer interface. AdamW contributes error budgets
for bias correction and a lower bound for the complete denominator, including epsilon. Its
square-root estimate remains valid when the second moment is zero.
`trainingStepTrace` collects the forward, backward, parameter-gradient, parameter-update, and
optimizer-state budgets in a noncomputable report. The preceding theorem proves the error bounds.

The MLP example checks the forward calculation. `roundoff` can also analyze supported typed-graph
forward and VJP computations directly from their Lean code. The end-to-end optimizer theorems above
use a supplied `RevGraph` with proved local VJPs and error bounds. TorchLean's current `NN.IR.Graph`
lowering preserves forward semantics, but does not automatically construct that proof-bearing
reverse graph for every model.

## Adding Another Architecture

To try this with another model, we first lower it to the canonical graph and check whether all
its operations have range rules. We don't need a new architecture wrapper:

```lean
let registry <- Proofs.RuntimeApprox.NumericalCertificate.defaultRegistry
let _ <- Proofs.RuntimeApprox.NumericalCertificate.requireNumericalCoverage registry graph
```

Covered graphs use the same certificate and replay APIs, subject to the shape, finite-range,
payload, and backend-policy checks described above. If one operation
is missing, add its executable `GraphRangeContract` and prove the corresponding exact-real
enclosure rule. The former extends range generation; the latter is needed before the generated
range can support `ProvedRealEnclosure`. If execution needs a new provider, add a `KernelCapsule`
and place it in a `CapsuleModule`. The registry and planner then make that implementation available
to every architecture that uses the operation.

---
title: Numerical Runtime Certificates
---

TorchLean can follow a model through two numerical views. The canonical IR records the operations
that will run. The runtime-approximation graph records the local forward and reverse theorems used
to compare rounded execution with exact real semantics. Both are assembled from operation-level
contracts, so neither checker needs a separate case for an MLP, transformer, convolutional model,
or neural operator.

The example runs the canonical-IR path on a two-layer MLP and identifies the corresponding backward
and optimizer theorems.

## Run the Example

From the TorchLean repository:

```bash
lake exe torchlean numerical_certificate
```

The final two rows should be:

```text
  ok  two-layer MLP certificate
  ok  two-layer MLP IEEE replay
```

The example also demonstrates rejection of a corrupted range. The regression suite in
`NN/Tests/Verification/GraphNumericalCertificate.lean` checks duplicate contracts, changed registry
identities, invalid square-root inputs, and incompatible CUDA reduction policies. Run the complete
suite with `lake test`.

## The Model

The complete model pass is:

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
The definitions `mlpGraph`, `mlpSources`, `mlpPayload`, `mlpCertificate`, and `mlpReplay` are the
complete runnable path.

## What Happens During the Run

### 1. Check operation coverage

`GraphRangeRegistry` maps an operation key to its numerical transfer. Before propagation starts,
TorchLean checks that every node has a registered transfer. An unknown primitive reports its node id
and operation name. It never receives a placeholder interval.

### 2. Propagate source ranges

The example supplies binary32 enclosures for the input and each parameter tensor. Matrix
multiplication uses outward-rounded products and a declared accumulation order. Addition uses
directed endpoints. ReLU restores the nonnegative lower bound of the hidden activation.

The resulting range trace contains one row for every node. A row records the node id, operation,
input enclosures, and derived output enclosure.

### 3. Select kernel capsules

The kernel planner independently selects a capsule for each operation. A capsule records the
provider, device, layout requirements, forward and VJP ownership, reduction policy, and trust
classification. The example uses the checked portable CPU profile because its fixed-left matrix
accumulation matches the canonical tensor semantics.

Changing the profile is not cosmetic. If a selected provider advertises a different numerical
policy, certificate generation fails until that provider has a matching contract.

### 4. Bind the artifact

`generateChecked` stores the graph, range-registry name, source assumptions, derived ranges, and
kernel audit together. Replay checks all of them. Replacing the graph, registry, or kernel plan
therefore invalidates the artifact.

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

## Forward, Backward, and the Optimizer

The proof-bearing training path uses `RevGraph`. Each node carries:

- exact and rounded forward functions;
- exact and rounded VJPs;
- a forward error transformer and proof;
- a VJP error transformer and proof.

`RevGraph.eval_approx` composes the forward bounds. `RevGraph.backprop_approx` traverses the same
graph in reverse and includes rounding from gradient accumulation. A parameter gradient can then be
passed to any `NumericalStepContract`:

```lean
#check Proofs.RuntimeApprox.NFBackend.eval_approx_graphData
#check Proofs.RuntimeApprox.NFBackend.backprop_approx_graphData
#check Proofs.RuntimeApprox.NFBackend.backprop_optimizer_update_approx_graphData
#check Proofs.RuntimeApprox.NFBackend.trainingStepTrace
```

SGD, momentum SGD, and AdamW use that one optimizer interface. AdamW contributes additional
step data because square root, bias correction, and division need explicit positivity margins.
`trainingStepTrace` returns forward bounds, backward bounds, the selected parameter-gradient bound,
the next parameter bound, and optimizer-state bounds. A future InfoView or training dashboard can
render this record without changing the proof.

There is one boundary to keep clear. Canonical `NN.IR.Graph` lowering currently proves forward
semantic preservation. It does not yet lower every typed graph node to a proof-bearing VJP. Therefore:

- the runnable MLP above is a complete canonical-IR **forward** certificate and IEEE replay;
- rounded backward and optimizer theorems are complete for a supplied proof-bearing `RevGraph`;
- automatically producing that `RevGraph` from every canonical-IR model remains a separate lowering
  theorem.

TorchLean keeps these statements separate so a forward artifact is never presented as a proof of a
backward implementation.

## Adding Another Architecture

No architecture wrapper is required. Lower the model to the canonical graph and check coverage:

```lean
let registry <- Proofs.RuntimeApprox.NumericalCertificate.defaultRegistry
let _ <- Proofs.RuntimeApprox.NumericalCertificate.requireNumericalCoverage registry graph
```

If all primitives are covered, certificate generation and replay work unchanged. If one operation
is missing, add its executable `GraphRangeContract` and prove the corresponding exact-real
enclosure rule. The former extends range generation; the latter is needed before the generated
range can support `ProvedRealEnclosure`. If execution needs a new provider, add a `KernelCapsule`
and place it in a `CapsuleModule`. The registry and planner then make that implementation available
to every architecture that uses the operation.

The design scales by operations, not by model names.

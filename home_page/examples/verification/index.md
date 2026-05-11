---
title: Verification Bounds
usemathjax: true
---

This page is the tutorial entry point for TorchLean’s verification demos. It explains what interval
bound propagation computes, what CROWN-style affine bounds add, how a TorchLean model becomes a
verification graph, and what it means to check an external certificate.

<div class="media-slab">
  <img src="{{ '/assets/media/examples/showcase/verification-bounds.png' | relative_url }}" alt="IBP and alpha-CROWN verification example"/>
</div>

## The Question

Most neural-network verification examples start with a robustness question:

> For every input inside a small box around this example, can the model’s output still satisfy the
> desired margin or safety condition?

TorchLean represents that question with four concrete objects:

- a model, written in the same API used for training examples;
- a compiled `NN.IR.Graph`, so verifier code can traverse named nodes;
- an input box, with lower and upper bounds for every input coordinate;
- an output property, usually a margin such as `logit_true - logit_other ≥ 0`.

The common path is therefore:

1. write or import a TorchLean model,
2. compile it to `NN.IR.Graph`,
3. attach an input box,
4. run a bound engine,
5. inspect or check the output bounds.

The examples under `NN/Examples/Verification/TorchLean/` are compact enough that the whole bound
path can be read without opening a large benchmark harness.

The seed box is built explicitly. For the small MLP example, `x0` is the center point, `eps` is the
radius, and `xB` is the flattened input box inserted at the compiled input node:

```lean
let x0 : Tensor α xShape :=
  NN.Tensor.tensorNDOfLenEq (α := α) [2] [cast 0.5, cast 0.8] (by rfl)
let eps : α := Runtime.ofFloat 0.1
let rad : Tensor α xShape := Spec.fill eps xShape

let xB : FlatBox α :=
  { dim := inDim
    lo := Tensor.subSpec x0 rad
    hi := Tensor.addSpec x0 rad }

let ps : ParamStore α :=
  { compiled.ps with inputBoxes := compiled.ps.inputBoxes.insert compiled.inputId xB }
```

So the verifier is not checking one input point. It is checking all inputs in the box
`[x0 - eps, x0 + eps]`.

```lean
let ibp := runIBP (α := α) compiled.graph ps
let some outB := ibp[compiled.outputId]! |
  throw <| IO.userError "IBP produced no output box"
```

The bound engine returns node-indexed `FlatBox` values. Each box stores a flattened dimension plus
lower and upper tensors. If the output box satisfies a margin condition such as
`lo[label] > max hi[other]`, every input in the seed box is certified for that label.

## IBP: Propagate Boxes Through The Graph

Interval bound propagation is the simplest sound bound engine in this part of TorchLean. Each node
gets a lower and upper bound. The transformer for each operation must enclose all possible outputs
of that operation when its inputs range over their current boxes.

A scalar example captures the idea. Suppose

```text
x ∈ [-1, 2]
y = 3 * x + 0.5
```

IBP computes

```text
y ∈ [3 * (-1) + 0.5, 3 * 2 + 0.5] = [-2.5, 6.5]
```

For a ReLU node, the transformer is monotone:

```text
z ∈ [l, u]
relu(z) ∈ [max(0, l), max(0, u)]
```

For a linear layer, the implementation splits positive and negative weights so each input interval
is used in the direction that gives the worst case. The result is conservative by design: every true
activation is inside the box, but the box may include values that cannot occur together.

That tradeoff explains both why IBP is useful and why it can fail to certify true properties. It is
fast, local, and easy to compose over graphs; it loses correlations between coordinates.

## A Margin Example

Consider a two-logit classifier. To certify class `0` against class `1`, we want:

```text
logit_0 - logit_1 ≥ 0
```

If IBP gives

```text
logit_0 ∈ [1.2, 1.8]
logit_1 ∈ [0.1, 0.7]
```

then the margin is at least `1.2 - 0.7 = 0.5`, so the box certifies the property. If instead IBP
gives

```text
logit_0 ∈ [0.8, 1.4]
logit_1 ∈ [0.2, 1.0]
```

then the lower margin bound is `0.8 - 1.0 = -0.2`. That does not prove the property. It also does
not mean the model is unsafe. It means this abstraction was not tight enough for this input box.

## CROWN-Style Affine Bounds

CROWN-style passes keep affine upper and lower forms instead of only interval boxes. In other words,
the bound is not just “this node lies between two numbers.” It can be “this node is bounded by a
linear expression over the input variables.” That preserves more correlation information.

CROWN still uses IBP intervals, because nonlinear relaxations need pre-activation ranges. Forward
CROWN stores affine lower and upper forms for nodes with respect to the chosen input node. Backward
CROWN starts from one scalar objective, such as `logit_0 - logit_1`, and propagates that objective
back to an input-box bound.

For ReLU, the affine relaxation depends on the pre-activation interval:

- if the interval is entirely nonnegative, ReLU is exactly the identity;
- if the interval is entirely nonpositive, ReLU is exactly zero;
- if the interval crosses zero, CROWN uses a sound linear envelope.

The example builds a context for the input node and runs CROWN after IBP:

```lean
let ctx : AffineCtx :=
  { inputId := compiled.inputId, inputDim := softmaxInDim }

let crown := runCROWN (α := α) compiled.graph ps ctx ibp
```

For a margin objective, the backward pass asks for a bound on one scalar expression, such as
`logit_0 - logit_1`. Instead of bounding every output independently, this lets the verifier push a
single objective backward through the graph:

```lean
let objV : Tensor α (.dim softmaxOutDim .scalar) :=
  NN.Tensor.tensorNDOfLenEq
    (α := α) [3] [cast 1.0, cast (-1.0), cast 0.0] (by rfl)

let obj : FlatVec α := { n := softmaxOutDim, v := objV }

match runCROWNBackwardObjective
    (α := α) compiled.graph ps ctx ibp compiled.outputId obj with
| none => IO.println "[CROWN-backward] no affine bounds"
| some objAff => IO.println s!"[CROWN-backward] objective dim = {objAff.outDim}"
```

This is the reason the example is useful: the model, graph, bounds, and certificate checks all
refer to the same node ids and tensor shapes.

## What Each Command Shows

Start with the small TorchLean-native examples:

```bash
lake exe verify -- torchlean-crown-ops --dtype float
lake exe verify -- torchlean-robustness --dtype float
lake exe verify -- margin-cert
```

`torchlean-crown-ops` compiles a compact graph, seeds an input box, runs IBP, then runs forward and
backward CROWN-style affine passes over supported operations. `torchlean-robustness` prints IBP,
CROWN, and backward-CROWN certification booleans for a small robustness workflow. `margin-cert`
checks an exported margin JSON artifact by recomputing the margin predicate.

Typical output from the native CROWN example includes softmax bounds, an MSE-loss bound, a margin
lower bound, and the backward objective bound. The exact numbers depend on dtype and runtime flags,
but the shape of the output should look like this:

```text
[IBP] logits lo = ...
[IBP] logits hi = ...
[CROWN] logits lo = ...
[CROWN] logits hi = ...
[CROWN-backward] objective bound = ...
```

Then check the bundled alpha-beta-CROWN leaf certificate:

```bash
lake exe verify -- abcrown-leaf
```

To run the fast non-interactive checker suite:

```bash
lake exe verify -- all
```

## External Certificates

The alpha-beta-CROWN leaf checker is a structural checker for a declared leaf artifact. The external
tool writes the JSON; Lean checks the part represented in that file: box nesting, compatible array
sizes, and the witness lower-bound test.

```lean
def leafVerifiedAt (lb thr : Array Float) (witnessIdx : Nat) : Bool :=
  if witnessIdx < lb.size ∧ witnessIdx < thr.size then
    ltBool thr[witnessIdx]! lb[witnessIdx]!
  else
    false
```

The CLI entry point defaults to a small bundled fixture:

```bash
lake exe verify -- abcrown-leaf \
  NN/Examples/Verification/AbCrown/sample_abcrown_leaf_cert_v0_1.json
```

## Where To Read The Source

- TorchLean-native graph and IBP demo:
  [`NN/Examples/Verification/TorchLean/TorchLeanIBP.lean`]({{ '/docs/NN/Examples/Verification/TorchLean/TorchLeanIBP.html' | relative_url }})
- CROWN operation demo:
  [`NN/Examples/Verification/TorchLean/TorchLeanCrownOps.lean`]({{ '/docs/NN/Examples/Verification/TorchLean/TorchLeanCrownOps.html' | relative_url }})
- α,β-CROWN leaf certificate checker:
  [`NN.Verification.Cert.AbCrownLeafCert`]({{ '/docs/NN/Verification/Cert/AbCrownLeafCert.html' | relative_url }})
- Verification guide chapter:
  [Verification and Certificates]({{ '/blueprint/Verification-and-Certificates/Verification/' | relative_url }})

The examples provide commands that check the relevant graph, bound, or certificate artifact in
Lean, rather than relying only on plots or external Python objects.

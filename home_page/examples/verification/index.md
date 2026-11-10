---
title: Verification Bounds
---

Let's look at what happens when we perturb a model's input. We'll compute bounds on its outputs
over an input region, starting with interval bound propagation (IBP). Then we'll try CROWN, which
can retain more information about how the outputs depend on the inputs.

We'll also check certificates exported by other tools. Each checker has a specific job:
some recompute bounds, while others only check the consistency of supplied bounds and witnesses.

<div class="media-slab">
  <img src="{{ '/assets/media/examples/showcase/verification-bounds.png' | relative_url }}" alt="IBP and alpha-CROWN verification example"/>
</div>

## The Question

Here's the robustness question we'll work with:

> For every input inside a small box around the example point, can the model’s output still satisfy the
> desired margin or safety condition?

To ask it in TorchLean, we need four things:

- a model, written in the same API used for training examples
- a lowered graph of registered operators (`TorchLean.IR.Graph`), so verifier code can traverse its
  nodes by id
- an input box, with lower and upper bounds for every input coordinate
- an output property, usually a margin such as
  $\operatorname{logit}\_{\mathrm{true}}-\operatorname{logit}\_{\mathrm{other}}>0$ for every other
  class

So the common path is:

1. write or import a TorchLean model,
2. lower it to a graph of registered operators (`TorchLean.IR.Graph`),
3. attach an input box,
4. run a bound engine,
5. inspect or check the output bounds.

Our reusable workflows under `TorchLean/Verify/CROWN/` keep the model, input box, output
property, and bound pass together. Imported weights need to be attached to a supported graph.
External verifier leaves go through a separate artifact reader, which doesn't recompute their
bounds.

Let's build an input box for a small MLP. We'll use `inputCenter` as its center and `eps` as its
radius, and take the lowered graph with its payload as an import (`LoweredModel.toImported`), whose
input `0` takes the flattened `inputBox`:

```lean
let inputCenter : Tensor α [2] :=
  Tensor.map cast ([0.5, 0.8] : Tensor Float [2])
let eps : α := ofFloat 0.1
let inputBox : FlatBox α := FlatBox.lInfBall (α := α) inputCenter eps
let imported := lowered.toImported
```

IBP propagates the input box

$$[\mathtt{inputCenter}-\mathtt{eps},\mathtt{inputCenter}+\mathtt{eps}].$$

```lean
let ibp := imported.runIBP inputBox
let outB ← IO.ofExcept (imported.outputBox? ibp)
```

IBP returns node-indexed `FlatBox` values, each storing a flattened dimension plus lower and upper
tensors. A sound output enclosure satisfying
$\mathrm{lo}[\mathrm{label}]>\max_{\mathrm{other}}\mathrm{hi}[\mathrm{other}]$ establishes that
label for every input in the seed box. The rounded CLI passes below calculate candidate enclosures,
and applying this argument to them requires a soundness theorem covering the graph, parameters, and
arithmetic. Their printed bounds alone don't supply that proof.

## IBP: Propagate Boxes Through The Graph

Interval bound propagation assigns a lower and upper bound to each node. A sound transformer for
an operation must enclose all possible outputs of that operation when its inputs range over their
current boxes.

We can see how this works with one scalar. Suppose

$$
x \in [-1,2],
\qquad
y=3x+0.5.
$$

IBP computes

$$
y \in [3(-1)+0.5,\;3(2)+0.5]=[-2.5,6.5].
$$

For a ReLU node, the transformer is monotone:

$$
z \in [\ell,u]
\quad\Longrightarrow\quad
\operatorname{ReLU}(z) \in [\max(0,\ell),\max(0,u)].
$$

For a linear layer, the implementation bounds each coefficient-times-input product at both
endpoints and accumulates lower and upper sums with directed rounding. With a sound transfer,
every true activation is inside the box, but the box may include values that cannot occur together.

That tradeoff explains both why IBP works well as a first verifier and why it can fail to certify
true properties. It's fast, local, and easy to compose over graphs, but it loses correlations
between coordinates.

## A Margin Example

Let's apply this to a classifier with two logits. To certify class $0$ against class $1$, we want:

$$
\operatorname{logit}_0-\operatorname{logit}_1 \geq 0.
$$

If IBP gives

$$
\operatorname{logit}_0 \in [1.2,1.8],
\qquad
\operatorname{logit}_1 \in [0.1,0.7],
$$

then the margin is at least $1.2-0.7=0.5$, so the box certifies the property. If instead IBP
gives

$$
\operatorname{logit}_0 \in [0.8,1.4],
\qquad
\operatorname{logit}_1 \in [0.2,1.0],
$$

then the lower margin bound is $0.8-1.0=-0.2$. At the IBP-box level the property is undecided. The
model may still be safe, but this abstraction wasn't tight enough for this input box.

## CROWN-Style Affine Bounds

CROWN-style passes keep affine upper and lower forms instead of only interval boxes. So a bound can
say more than “this node lies between two numbers.” It can say “this node is bounded by a linear
expression over the input variables,” which preserves more correlation information.

CROWN still uses IBP intervals, because nonlinear relaxations need pre-activation ranges. Forward
CROWN stores affine lower and upper forms for nodes with respect to the chosen input node. Backward
CROWN starts from one scalar objective, such as
$\operatorname{logit}_0-\operatorname{logit}_1$, and propagates that objective
back to an input-box bound.

For ReLU, the affine relaxation depends on the pre-activation interval:

- if the interval is entirely nonnegative, ReLU is exactly the identity
- if the interval is entirely nonpositive, ReLU is exactly zero
- if the interval crosses zero, CROWN uses a sound linear envelope

We can ask the imported graph for a CROWN output box. That request runs the backward pass on every
backend (interval coefficients swept from the output to the input with outward rounding), and a
node without an affine transfer keeps its IBP enclosure:

```lean
let crown ← IO.ofExcept (imported.outputBoxCROWN? inputBox)
```

The result is another `FlatBox` for the output node. It need not improve on IBP. The forward α-CROWN
pass `TorchLean.CROWN.Op.outputBoxAlphaCROWN?` on `imported.graph` runs on backends supporting exact
affine reassociation and falls back to this box on the others. When the intermediate objects are
needed, `TorchLean.CROWN.Op.crownBoxes` on `imported.graph` returns the box of every node from full
CROWN, and `TorchLean.CROWN.Op.runCROWN` on `imported.graph` returns nodewise affine bounds, not
only the output box.

For a margin objective, the backward pass asks for a bound on one scalar expression, such as
$\operatorname{logit}_0-\operatorname{logit}_1$, so the verifier can push a single objective
backward through the graph instead of bounding every output independently:

```lean
let objV : Tensor α [3] :=
  Tensor.map cast ([1.0, -1.0, 0.0] : Tensor Float [3])
let obj : FlatTensor α := { n := 3, v := objV }

let objectiveBox ← IO.ofExcept (imported.backwardObjectiveBox? ibp inputBox obj)
let margin := Tensor.getAtOrZero objectiveBox.lo [0]
```

`margin` is the reported candidate lower bound on $p_0 - p_1$ over the input box. Its semantic
guarantee has the same soundness obligations as the output enclosure above.

`backwardObjectiveBox?` reads the affine context of the import's input `0` and output itself, so
the caller does not build that context by hand.

The model, graph, bounds, and certificate checks all refer to the same node ids and tensor shapes.

## Tensor Shapes In The Bound Pass

Flattening a tensor into a box doesn't discard its graph shape. Matrix products promote vector
operands and broadcast compatible leading batch axes: `[2]` times `[2,3]` gives `[3]`, while
`[2,1,2,2]` times `[1,3,2,2]` gives `[2,3,2,2]` (the contraction dimensions must match). IR
evaluation, IBP, and CROWN share this shape contract. The backward pass bounds a binary product by
McCormick planes of its operand boxes, and that relaxation can lose correlations even when the
shape is supported.

Concatenation accepts any valid axis and at least two parents, including empty dimensions and
repeated parent ids. Dimensions off the selected axis must match. The backward pass splits
coefficients by parent occurrence and adds the contributions when a parent appears more than once.

Convolution transfers accept groups, dilation, asymmetric padding, and arbitrary leading batch
dimensions, with the configuration and payload checked together. A convolution node in
PyTorch's layout reads its kernel as `(outC, inC / groups, k...)`, keeping for each output
channel the input channels of its group (`TorchLean.Conv.packedKernel`). The backward pass sends
coefficients through a convolution by the transposed convolution, so it builds no dense matrix
over the flattened coordinates.

LayerNorm normalizes the whole suffix beginning at its chosen axis. On `[2,2,2]`, axis one means
two independent rows with normalized shape `[2,2]` (flattening all eight entries into one row would
compute a different function). LayerNorm's scale and bias must have that exact suffix shape.
Softmax value bounds likewise use the selected axis: an extent of one gives `[1,1]`, while other
extents use `[0,1]`.

These rules describe executable bound propagation. For the rounded pass, the soundness theorem
`TorchLean.CROWN.Op.Directed.encloses_of_run` covers every operator the pass boxes at scalars
with lawful endpoint operations, but it doesn't discharge the rounded-execution assumptions of a
native run.

## What Each Command Shows

Let's run the TorchLean examples first:

```bash
scripts/lake.sh exe verify -- torchlean-ibp
scripts/lake.sh exe verify -- torchlean-transformer-ibp
scripts/lake.sh exe verify -- torchlean-crown-ops
scripts/lake.sh exe verify -- torchlean-mlp-workflow
scripts/lake.sh exe verify -- camera-box3d-cert
```

`torchlean-ibp` is the smallest graph-bound check: lower a TorchLean model, attach an input box,
and propagate interval bounds to the output. `torchlean-transformer-ibp` applies attention and
LayerNorm, then bounds their MSE against a fixed target. With `--with-crown` it also requests CROWN
output bounds and objective-dependent backward CROWN bounds on the loss. `torchlean-crown-ops` uses
the same graph style but adds forward and backward CROWN-style affine passes over softmax and
MSE-loss operations. `torchlean-mlp-workflow` trains a classifier and then checks robustness with
`trained.verify`, which runs CROWN (`outputBoxCROWN?`) on the resulting graph from the requested
input box. The arithmetic used by these commands follows the same `--arithmetic native|ieee` flag
as the examples.

`vnncomp-mnistfc` exercises a compact VNN-COMP-style fully connected MNIST network/property pair.
`camera-box3d-cert` checks a camera projection certificate for a supplied finite point set by
recomputing its projections and the claimed 2D envelope expanded by the artifact's nonnegative
tolerance. Eight cuboid corners are one choice of point set.

The MNIST runner labels its result `numerically_refuted`, not `safe`. It uses outward-widened host
`Float` operations to refute the unsafe output region, but that executable result is not itself a
Lean theorem about real-valued network semantics.

For the MNIST workflow you need externally prepared weights and suite files, which aren't bundled.
After preparing the JSON artifacts that
[`TorchLeanExamples.Verification.VNNComp.MNISTFC`](https://github.com/lean-dojo/TorchLean/blob/main/TorchLeanExamples/Verification/VNNComp/MNISTFC.lean)
reads from `_external/vnncomp/mnist_fc/`, run:

```bash
scripts/lake.sh exe verify -- vnncomp-mnistfc \
  --weights=_external/vnncomp/mnist_fc/model_weights.json \
  --suite=_external/vnncomp/mnist_fc/suite.json
```

Typical output from the native CROWN example includes softmax bounds, an MSE-loss bound, a margin
lower bound, and the backward objective bound. Exact values depend on dtype and runtime flags, but
the shape of the output should look like this:

```text
[IBP] p lo = ...
[IBP] p hi = ...
[CROWN] p lo = ...
[CROWN] p hi = ...
[CROWN-backward] margin lo = ...
[CROWN-backward] margin hi = ...
```

Then check the bundled alpha-beta-CROWN-style leaf artifact:

```bash
scripts/lake.sh exe verify -- abcrown-leaf
```

Run the bundled IBP certificate check:

```bash
scripts/lake.sh exe verify -- ibp-cert
```

This command is a good check when you change certificate parsing or bound-replay utilities. Its
bundled certificate is for a 3 → 4 → 2 ReLU MLP, and Lean accepts it only when the serialized
bounds enclose the bounds it recomputes. The fixture is evidence for the checker format and replay
predicate, and it doesn't cover every possible IBP producer.

To run the fast non-interactive checker suite:

```bash
scripts/lake.sh exe verify -- all
```

## External Artifacts

The alpha-beta-CROWN leaf checker is a consistency checker for a declared leaf artifact. Vanilla
alpha-beta-CROWN doesn't emit TorchLean's JSON schema. Instead, an external verifier exposes or
dumps terminal leaf data, TorchLean's exporter converts that data to `abcrown_leaf_artifact_v0_1`,
and Lean checks the represented part of the artifact (box nesting, coverage of the root box by the
leaves, compatible tensor dimensions, and the witness lower-bound test). The checker doesn't
recompute the lower bounds.

So the checker relies on the producer for the correctness of the supplied lower bounds, and it
can't establish whether the JSON faithfully records the external verifier's output.

The witness selects one output coordinate, and the checker requires its lower bound to exceed the
corresponding unsafe threshold. Mismatched dimensions and out-of-range witness indices are
rejected. We implement the checker in `TorchLean.Verify.Certificate.AbCROWNLeaf` and its shared
verification utilities.

We can try the leaf checker with the small bundled certificate, which is also its default input:

```bash
scripts/lake.sh exe verify -- abcrown-leaf \
  TorchLeanExamples/Verification/AbCROWN/sample_abcrown_leaf_artifact_v0_1.json
```

```text
[artifact] Checked 1 leaves: ok=1, bad=0
[artifact] consistent: the leaves cover the root and every leaf clears its threshold.
[artifact] The lower bounds are the producer's claims; TorchLean did not recompute them.
```

To create that schema from a raw terminal-domain dump, use the TorchLean exporter:

```bash
python3 scripts/verification/abcrown/export_leaf_artifact.py \
  --input TorchLeanExamples/Verification/AbCROWN/example_raw_leaf_dump.json \
  --out _external/abcrown/leaf_artifact.json \
  --check
```

`ABCROWN_ARTIFACT_OUT` is the TorchLean side exporter hook. Use it from a small wrapper or
instrumented external verifier to write the schema that Lean checks. The external search owns the
branch-and-bound run, while TorchLean owns the exported artifact schema and the local witness
predicate replayed by the checker.

## Verification Results

Let's distinguish what the results of these commands tell us:

- IBP and CROWN commands lower a TorchLean model, attach an input box, propagate bounds over the
  supported graph operations, and report the resulting margin predicate.
- `ibp-cert` replays a stored IBP bound for a small 3-4-2 ReLU MLP. It is a regression fixture
  for the checker API and an example of the finite objects Lean can reload.
- $\alpha,\beta$-CROWN-style leaf artifacts carry one terminal external-verifier claim into Lean. The checker
  validates the schema, box nesting and root coverage, tensor dimensions, and witness lower-bound comparison represented in
  that artifact.
- VNN-COMP-style examples show how a benchmark-shaped network/property pair can enter TorchLean
  while the benchmark runner remains an external producer.
- PINN, ODE, spline, and geometry examples use the same pattern outside image classification: a
  producer exports an artifact, and Lean recomputes the residual, enclosure, interval, or projection
  predicate being checked.

When a command succeeds, cite the object and predicate it checked. For example, say that Lean
accepted the `abcrown_leaf_artifact_v0_1` witness predicate for a particular JSON file, or that the
TorchLean graph IBP pass reported a margin computed for the stated input box and arithmetic. That
phrasing is more precise than saying only that a verifier ran.

## Where To Read The Source

- TorchLean-native graph and IBP entry point:
  [`TorchLeanExamples/Verification/IBPWorkflow.lean`](https://github.com/lean-dojo/TorchLean/blob/main/TorchLeanExamples/Verification/IBPWorkflow.lean)
- CROWN operation entry point:
  [`TorchLeanExamples/Verification/CROWNOpsWorkflow.lean`](https://github.com/lean-dojo/TorchLean/blob/main/TorchLeanExamples/Verification/CROWNOpsWorkflow.lean)
- $\alpha,\beta$-CROWN-style leaf artifact checker:
  [`TorchLean.Verify.Certificate.AbCROWNLeaf`](https://github.com/lean-dojo/TorchLean/blob/main/TorchLean/Verify/Certificate/AbCROWNLeaf.lean)
- VNN-COMP-style MNIST entry point:
  [`TorchLeanExamples.Verification.VNNComp.MNISTFC`](https://github.com/lean-dojo/TorchLean/blob/main/TorchLeanExamples/Verification/VNNComp/MNISTFC.lean)
- 3D geometry certificate checker:
  [`TorchLeanExamples/Verification/Geometry3D`](https://github.com/lean-dojo/TorchLean/blob/main/TorchLeanExamples/Verification/Geometry3D/)
- Verification guide chapter:
  [Neural Network Verification]({{ '/blueprint/Verification-and-Certificates/Neural-Network-Verification/' | relative_url }})

# `NN/Verification`

This directory contains the reusable verification machinery for TorchLean. It is where examples and
CLI commands go when they need to turn a model, graph, certificate, dataset, or PDE residual into a
Lean-checkable object.

The runnable verification commands reuse the same pieces as the Bug Zoo, robustness examples,
PINN/scientific ML checks, ODE and spline certificates, and graph-level IBP/CROWN workflows.

## What Lives Here

- `Builtin/`: the built-in verifier workflows (`IBPWorkflow`, `CrownOpsWorkflow`,
  `TransformerIBPWorkflow`, `MlpTrainVerifyWorkflow`), the lowering from TorchLean model fragments
  to the verifier graph IR, and the `Proved/` subtree of theorem-backed lowering and evaluator
  fragments for supported forward programs.
- `Cert/`: JSON certificate formats and executable recomputation checkers, including IBP, CROWN,
  alpha-CROWN, and alpha-beta-CROWN-style local node artifacts. The alpha-beta-CROWN node format
  adds ReLU phases that must already follow from IBP, so it replays alpha-CROWN bounds; it has no
  beta multipliers or branch splits. `Cert/FiniteArtifact` checks binary32 affine bounds for
  linear/ReLU chains against exact rational transfers, and `FiniteArtifact.accepts_graph_sound`
  proves the requested output inequalities for the original graph under the exact real values of
  its binary32 parameters. It is a library API with no `lake exe verify` command.
- `Monotonicity`: exact nonnegative-weight certificates for linear/ReLU chains, with a theorem
  from JSON text acceptance to global componentwise monotonicity over real inputs.
- `Robustness/`: dataset-backed robustness workflows, including certified accuracy for small
  sklearn digits models and VNN-COMP-style MNIST fully connected examples.
- `PINN/`: PDE expression parsing, graph builders, residual interval helpers, certificate replay,
  CLI commands, and dataset containment checks for PINN-style artifacts.
- `ODE/`: corridor certificate checking for learned subsolution and supersolution enclosures.
- `Geometry3D/`: camera/projection certificate checks for 3D bounding-box workflows.
- `LiRPA/`: small exported artifacts for attention, CNN, GRU, MLP, and transformer encoder
  certificates.
- `VNNComp/`: compact VNN-COMP-style network/property loaders and MNIST fully connected checks.
- `Util/`: shared parsing, tolerance, and numeric helpers.

## Main Workflows

TorchLean verification work has five recurring shapes.

1. A Lean-native model is lowered to `NN.IR.Graph`; interval or affine bounds are propagated over
   the graph; the checker reports whether the requested output relation follows from those bounds.

2. An external tool produces an artifact, such as an alpha-beta-CROWN-style certificate or exported
   JSON bounds. Lean parses the artifact and checks the predicate the format actually claims.

3. A scientific ML script exports small PDE or dataset artifacts. Lean reloads the residual
   expression, boxes, weights, samples, or intervals and checks the stated containment or residual
   condition.

4. A theorem-backed fragment connects the executable checker to a mathematical statement. This is
   where a checked artifact becomes part of a formal claim, with the Lean hypotheses and remaining
   producer assumptions visible at the call site.

5. A domain-specific checker tests a smaller artifact predicate: projected-corner containment,
   ODE corridor inequalities, PINN residual bounds, or a classifier margin derived from computed
   bounds. A guarantee about a trajectory or every input additionally needs the corresponding
   semantic soundness theorem.

The common pattern is intentionally plain: parse a small artifact, check shape and schema fields,
recompute the claim in Lean, and name the assumptions that remain outside Lean.

## What Is Proved Versus Checked

External JSON is always treated as untrusted input. A checker may accept a file, but acceptance only
means the file satisfied the predicate implemented by that checker. Decimal tolerances are explicit
format tolerances. Soundness of the external producer is a separate claim and should be represented
by an exporter/provenance argument, a replayable certificate, or a theorem about the producer.

The theorem-level graph IBP/CROWN-family soundness results live under
`NN.MLTheory.CROWN.Proofs.*` and are imported by `NN.Verification`.

The TorchLean-to-IR proof API is separate. It covers the proved lowering fragment and local
evaluator lemmas for supported imported operations. Current coverage includes elementwise
arithmetic and activations, reshape/flatten/broadcast/sum, concat, axis reductions, supported
transpose and permutation forms, matrix and batched matrix multiplication, arbitrary-axis softmax,
payload-backed constants, `linear`, arbitrary-rank no-dilation convolution, eval-mode
channel-first BatchNorm, arbitrary-rank pooling, suffix-axis LayerNorm, graph input/detach, and scalar MSE formulas used
by the PyTorch/ONNX path.

For payload-backed imported ops, the bridge records both the helper evaluator contract and the
actual one-step `Graph.evalAt` success path. The
`NN.Verification.Builtin.Proved.Correctness.Eval` import collects the concrete evaluator modules
for elementwise, shape, reduction, permutation, linear-algebra, softmax, payload, and lowering
facts. The imported theorems, rather than a separate coverage declaration, are the current record
of proved evaluator support.

The CROWN engine's executable transfer coverage is a separate boundary from these lowering
theorems. Its matrix products promote vector operands and broadcast compatible leading batch
axes. Convolution validates its groups, dilation and both padding sides, and LayerNorm uses the
full trailing normalized shape. See
[the CROWN geometry contracts](../MLTheory/CROWN/README.md#tensor-geometry) for the operator limits
and their relation to the graph soundness theorems.

## Public Imports

- `NN.Verification`: reusable verification APIs and public handles to proof-backed
  certificate soundness statements.
- `NN.Verification.Cert`: executable certificate checker API.
- `NN.Verification.Cert.CROWNQuery.Json`: exact dense/ReLU CROWN output-query checker and
  acceptance-to-safety theorems.
- `NN.Verification.Monotonicity.Json`: exact monotonicity decoder, checker, and soundness theorems.
- `NN.Verification.LiRPA`: compact LiRPA-style artifact checkers and shared certificate utilities.
- `NN.Verification.ODE`: ODE corridor verifier API.
- `NN.Verification.Splines`: spline and piecewise-polynomial certificate checker API.
- `NN.Verification.Robustness`: reusable robustness workflows and loaders.
- `NN.Verification.PINN`: reusable PINN verification support.
- `NN.Verification.Geometry3D`: 3D projection and box-certificate checking.
- `NN.Verification.VNNComp`: compact VNN-COMP-style network and property workflows.
- `NN.Verification.CLI`: runnable CLI registry used by `lake exe verify`.

## Commands To Try

### Exact CROWN output queries

Run `lake exe verify -- crown-query <query.json>`, or import `NN.Verification.Cert.CROWNQuery.Json`
and call `NN.Verification.CROWNQuery.acceptsText`, on a document such as:

```json
{
  "format": "crown_query_v1",
  "input_dim": 1,
  "input": {"lo": ["-1"], "hi": ["1"]},
  "layers": [
    {"kind": "linear", "weights": [["1"], ["-1"]], "bias": ["0", "0"]},
    {"kind": "relu", "alpha": ["1/2", "1/2"]},
    {"kind": "linear", "weights": [["1", "1"]], "bias": ["0"]}
  ],
  "query": {"weights": [["1"]], "bias": ["-3/2"], "strict": true}
}
```

This asks whether `relu(x) + relu(-x) < 3/2` for every real `x` in `[-1, 1]`.
Each query row means `weights * output + bias < 0`, or `≤ 0` when `strict` is false;
all rows must hold. The checker retains affine dependence on the original input, recomputes
linear sign-split and α-ReLU transfers using existing CROWN operations, and bounds the final
query over the input box. It checks each proposed α lies in `[0, 1]`.

Parameters and endpoints are exact integer/fraction strings. Unsupported operations, malformed
dimensions, invalid fractions, reversed boxes, empty input/output/query dimensions, and invalid
α values reject. No external affine bounds are trusted. Failure means the checker did not prove
the query; it need not mean the property is false.

`Query.check_sound`, `acceptsJson_sound`, and `acceptsText_sound` prove that acceptance implies
every decoded output inequality for every real input in the decoded box. The model uses existing
`Spec.Module.linear`, `Spec.Module.relu`, and `Spec.Module.Chain` semantics; its evaluation is
total. These theorems require no producer-soundness, local-transfer, or graph-coverage hypothesis.
`NN.Tests.MLTheory.CROWNQuery` includes a concrete kernel-checked safety theorem and runtime
acceptance/rejection tests in `nn_tests_suite`.

This format is separate from the binary32 node-replay JSON format. It does not certify ONNX
translation, floating-point deployment, β dual variables, cuts, or branch-tree coverage. A model
producer still has to establish that the decoded network is the model it intends to verify.

### Exact monotonicity certificates

Import `NN.Verification.Monotonicity.Json` and use
`NN.Verification.Monotonicity.acceptsText` on a document such as:

```json
{
  "format": "monotonicity_v1",
  "input_dim": 2,
  "layers": [
    {"kind": "linear", "weights": [["1/3", "0"]], "bias": ["-7"]},
    {"kind": "relu"}
  ]
}
```

Each weight and bias is an exact integer or fraction string. The decoder checks all layer
dimensions; malformed rational strings, numeric JSON parameters, and unsupported operations reject.
The weight check accepts nonnegative entries and allows arbitrary biases. It is sufficient but
incomplete: some monotone networks contain negative weights and will be rejected.

`acceptsText_sound` proves that every accepted text decodes to a `Spec.Module.Chain ℝ` whose
outputs preserve componentwise order for every ordered pair of real inputs. `check_sound` is the
corresponding theorem for typed certificates. Neither requires a producer-soundness or local-transfer
hypothesis. This is a theorem about the model recorded in the document; correspondence to an ONNX
file, deployment arithmetic, and CROWN/branch-and-bound certificates are outside this format.

`NN.Tests.MLTheory.Monotonicity` contains a concrete kernel-checked model theorem and parser tests,
included in `nn_tests_suite`.

### Other verification workflows

Run the curated verification suite:

```bash
lake exe verify -- all
```

Run TorchLean-native graph workflows:

```bash
lake exe verify -- torchlean-ibp
lake exe verify -- torchlean-crown-ops
lake exe verify -- torchlean-mlp-workflow
```

These commands lower or build small TorchLean models, attach input regions, and run native bound
passes. The MLP workflow trains a classifier and calls
`trained.verify center (radius := 0.10) (norm := .inf) (property := .topLabel 0)`; its graph and
bound state stay internal. The default algorithm, `.alphaBetaCrown`, is fixed-relaxation
alpha-CROWN with ReLU phases inferred from IBP; it has no beta multipliers or branch splits. Use these commands when changing graph lowering, bound
propagation, typed graph execution, or the verification API.

Run a compact PINN certificate and residual-expression check:

```bash
lake exe verify -- pinn-cert
lake exe verify -- pinn-cli -- "u_t + u*u_x - 0.01*u_xx" 0.0 0.5 0.01
```

Run a dataset containment diagnostic for PINN artifacts:

```bash
lake exe verify -- pinn-dataset-check
```

Use `--strict` on diagnostic commands when misses should turn into command failure.

Run a compact consistency check of a converted alpha-beta-CROWN leaf artifact:

```bash
lake exe verify -- abcrown-leaf
```

The checker confirms that the leaf boxes lie inside the root and together cover it, and that each
leaf's reported lower bound clears its threshold. The lower bounds are the producer's numbers;
TorchLean does not recompute them, so a pass means the artifact is consistent, not that the
network is verified.

Run compact LiRPA-style fixture checks:

```bash
lake exe verify -- lirpa-mlp
lake exe verify -- lirpa-cnn
lake exe verify -- lirpa-attention
lake exe verify -- lirpa-gru
lake exe verify -- lirpa-encoder
```

These fixtures are small JSON artifacts for supported network fragments. They are replay fixtures:
the Python exporters in `scripts/verification/lirpa` reproduce Lean's own host-Float IBP pass
operation for operation, so each certificate matches Lean's recomputation bit for bit and carries
no independent evidence. A bound computed by auto_LiRPA itself would generally not pass. The checks
exercise the artifact parser and replay predicate without depending on a live external verifier,
and they rely on host `Float` arithmetic; no theorem covers them.

The VNN-COMP-style MNIST workflow requires externally prepared weights and suite files; they are
not bundled. Follow [the artifact setup](../Examples/Verification/VNNComp/README.md), then run:

```bash
lake exe verify -- vnncomp-mnistfc \
  --weights=_external/vnncomp/mnist_fc/model_weights.json \
  --suite=_external/vnncomp/mnist_fc/suite.json
```

Run a 3D projection certificate check:

```bash
lake exe verify -- camera-box3d-cert
```

`BoxCameraCert.pointCount` fixes the number of supplied points; its tensor has shape
`[pointCount, 3]`. A `torchlean.camera.box3d.v1` artifact carries exactly eight corners, with
`point_count` omitted or set to `8`. A `torchlean.camera.box3d.v2` artifact must declare a positive
`point_count` that matches the number of triples in `corners3d`. The parser rejects an artifact
with no points in either format. Acceptance checks projection, positive depth, and image/box
containment for each supplied point. It does not establish that those points form a cuboid or
correspond to an object in an image.

The checker runs on `BoxCameraCert Float`: the projection and every comparison use host binary64
with round-to-nearest. The soundness theorems are generic in the scalar type, so for `Float` they
say that the accepted Boolean checks held for the rounded values. Nothing is proved about the
exact real projection; the artifact's `tol` is the only slack for rounding.

Run the spline certificate checker:

```bash
lake exe verify -- spline-cert
```

The exact command registry lives in `NN.Verification.CLI`; `lake exe verify --help` prints the
commands available in the current checkout.

## What A Checker Should Say

A new checker should answer five questions in the code or README:

1. What object is being checked?
2. What file format or runtime object is allowed to provide candidate data?
3. Which predicate is recomputed inside Lean?
4. Which theorem, if any, lifts the check to a mathematical statement?
5. Which producer, solver, runtime, or floating-point assumptions remain outside Lean?

That convention makes checker output precise: a successful run identifies the object, the predicate,
the theorem layer when one applies, and the remaining producer or runtime assumptions.

## References

- IBP: Gowal et al., 2018, "On the Effectiveness of Interval Bound Propagation for Training
  Verifiably Robust Models".
- CROWN: Zhang et al., 2018, "Efficient Neural Network Verification with CROWN".
- LiRPA: Xu et al., 2020, "Automatic Perturbation Analysis for Scalable Certified Robustness and
  Beyond".
- ODE learn-and-verify corridors: Tanaka and Yatabe, 2026, arXiv:2601.19818.

# `TorchLean/Verify`

The verification layer holds the checkers and workflows that turn a model, graph, certificate,
dataset or PDE residual into something Lean can check. The `verify` commands, the Bug Zoo, the
robustness and PINN examples, the ODE and spline certificates and the graph IBP/CROWN workflows all
use it.

The PINN, ODE and spline checkers are in the
[verification examples](../../TorchLeanExamples/Verification/README.md).

External JSON is untrusted input. Acceptance means the file satisfied the predicate its checker
implements, and a stronger claim needs one of the theorems named below.

## What Lives Here

- [`CROWN/`](CROWN/README.md) and `CROWN.lean`: CROWN/LiRPA-style bound propagation over graphs of
  registered operators (`IR.Graph`), its certificate data and its proofs. `CROWN.lean` holds the
  interval boxes (`Box`) and the evaluation of affine forms on them (`TorchLean.Linear.evalOnBox`).
  The passes on imported graphs (`Imported.runIBP`, `Imported.outputBoxCROWN?`) are in
  `CROWN/Op/Imported.lean`, and `CROWN/Arithmetic.lean` holds `withArithmetic`, which runs a bound
  computation under native `Float32` or the bit-level `ExecFloat.Binary 8 23`. The infoview panels
  for IBP boxes and CROWN affine bounds (`#crown_view`) are in `CROWN/Widget.lean`.
- `Capture.lean` and `Capture/`: the checks of the `ir` command, which check a captured typed IR
  artifact and can run supplied arguments or compare reference outputs. `checkError`
  (`Capture/Error.lean`) re-executes two captured programs on the reference runtime of the
  operator IR and checks a rational absolute error budget at their supplied arguments.
- `Certificate/`: JSON certificate formats and executable recomputation checkers, including IBP and
  alpha-CROWN local node artifacts. `CROWNNode.check` decides a CROWN node certificate with
  `FiniteArtifact.accepts` (`Certificate/FiniteArtifact.lean`), which checks its binary32 affine
  bounds for linear/ReLU chains against exact rational transfers, and
  `FiniteArtifact.accepts_graph_sound` proves the requested output inequalities for the original
  graph under the exact real values of its binary32 parameters. `CROWNNode.replayDiagnostic` lists
  where the certificate differs from Lean's binary32 alpha-CROWN replay, with ReLU phases that IBP
  already implies and no beta multipliers or branch splits. It is a diagnostic, not a safety
  verdict. Both are library APIs with no `scripts/lake.sh exe verify` command.
- `Coverage/`: the `#roundoff_coverage` runs over the operator specifications, by family
  (elementwise, indexing, layers, losses, ordering, pointwise, tensor composites and tensor
  operations) and for the structure specifications and tape operations. Each run declares a
  theorem for every route it proves.
- [`Floats/`](Floats/README.md): the rounded-real binary32 model `FP32` and finite configured
  arithmetic for reduction proofs (`Floats/IEEE754.lean`).
- [`LibTorch/`](LibTorch/README.md): the agreement contract for native LibTorch arithmetic in the
  formats of FloatLib's configured binary family, and the deterministic CUDA sum `reduceSumTree`.
- `Monotonicity.lean` and `Monotonicity/`: exact nonnegative-weight certificates for linear/ReLU
  chains, with a theorem from JSON text acceptance to global componentwise monotonicity over real
  inputs.
- `Numeric/`: absolute-tolerance comparisons for artifact replay (`Numeric/ApproxEq.lean`) and
  exact endpoint interval addition (`Numeric/Interval.lean`).
- `NumericalCertificate.lean` and `NumericalCertificate/`: numerical certificates of a graph,
  linked to the backend kernel plan. `NumericalCertificate.check` rebuilds a certificate's range
  rows and kernel-plan audit from the graph and accepts it only when both match.
- [`Robustness/`](Robustness/README.md): Lipschitz bounds for linear layers and two-layer ReLU MLPs,
  and margin lemmas that keep an `argmax` classifier stable under small ℓ∞ perturbations. A CROWN
  margin proof for a small MLP is in the
  [Robustness examples](../../TorchLeanExamples/Verification/README.md).
- `RuntimeApprox.lean` and [`RuntimeApprox/`](RuntimeApprox/README.md): the approximation
  predicates `ApproxWith` and `ApproxWithTol`, and rounding-error theorems for runtime operators,
  reductions, graphs and optimizers.
- `UniversalApproximation/` and [`Approximation/`](Approximation/README.md): ReLU approximation
  bounds under rounded and configured binary arithmetic, and the interval semantics of binary32
  ReLU networks (`Approximation/FloatInterval/`).
- `Report.lean`: the request vocabulary of `trained.verify` (`Algorithm`, `Property`) and the
  `Report` it returns.
- `Trainer.lean`: `trained.verify` (`Trainer.Result.verify`), which lowers its named arguments to
  a verifier input region and returns a validated report.
- `Lowering.lean`: direct construction and inspection of the verifier graph for graph-level
  IBP/CROWN workflows (`Verification.lowerForwardToIR`). A custom forward program is lowered by
  `IR.lowerForward`, and the lowered model `IR.LoweredModel` is defined in
  `TorchLean.NN.IR.Lowering`.
- `DecimalRounding.lean`, `JSON.lean` and `Tensor.lean`: the helpers the checkers share. They round
  an exact JSON decimal once, in a chosen direction, into a binary format, read the rationals,
  boxes and `format` fields of certificates, and load tensors from JSON payloads of checked shape.

## Main Workflows

TorchLean verification work has five recurring shapes.

1. A Lean-native model is lowered to a graph of registered operators (`IR.Graph`), interval
   or affine bounds are propagated over the graph, and the checker reports whether the requested
   output relation follows from those bounds.

2. An external tool produces an artifact, such as an alpha-beta-CROWN-style certificate or exported
   JSON bounds. Lean parses the artifact and checks the predicate the format claims.

3. A scientific ML script exports small PDE or dataset artifacts. Lean reloads the residual
   expression, boxes, weights, samples, or intervals and checks the stated containment or residual
   condition.

4. A theorem-backed fragment connects the executable checker to a mathematical statement. A checked
   artifact then becomes part of a formal claim, with the Lean hypotheses and the producer
   assumptions left open visible at the call site.

5. A domain-specific checker tests a smaller artifact predicate: projected-corner containment,
   ODE corridor inequalities, PINN residual bounds, or a classifier margin derived from computed
   bounds. A guarantee about a trajectory or every input also needs the corresponding semantic
   soundness theorem.

The common pattern is plain: parse a small artifact, check its shape and schema fields, recompute
the claim in Lean, and name the assumptions left outside Lean.

## Verifying A Trained Model

A trained model is checked over the L∞ ball of radius `r` around `center` with
`trained.verify center (radius := r)`, which returns a `Verification.Report` holding the bounds and
the choices that produced them. The bounds concern the recorded verifier graph, so applying a
report to the original finite-precision execution needs a source correspondence argument for the
complete model.

The defaults are the `crown` algorithm and the `outputBounds` property. `(algorithm := .ibp)` and
`(algorithm := .alphaCROWN)` select the other bound passes, and `(property := .topLabel label)`
checks that output `label` stays strictly above every other output:

```lean
-- Inside a `do` block, with a `trainer`, a `dataset` and an input `center`:
let trained ← trainer.train dataset { steps := 120 }
let report ← trained.verify center (radius := 0.10) (property := .topLabel 0)
report.printSummary
```

`TorchLeanExamples/Verification/MLPTrainVerifyWorkflow.lean` runs this workflow with
`scripts/lake.sh exe verify -- torchlean-mlp-workflow`.

## What Is Proved Versus Checked

Any decimal tolerance a checker allows is part of its format. Soundness of the external producer
is a separate claim, which needs an exporter or provenance argument, a replayable certificate, or
a theorem about the producer.

The soundness theorems of the graph passes are in `CROWN/Op/`, and the convolution, LayerNorm and
relaxation lemmas they use are in `CROWN/Conv/`, `CROWN/LayerNorm/`, `CROWN/Activation/` and
`CROWN/FlatBox/`. Among them, `Op.Directed.encloses_of_run`, `CROWN.Op.outputBoxCROWN?_of_run` and
`Op.Alpha.encloses_of_run` are in `TorchLean.Verify.CROWN.Op.Directed`,
`TorchLean.Verify.CROWN.Op.BackwardRun` and `TorchLean.Verify.CROWN.Op.Alpha`, and the IBP
certificate soundness `Op.certLocalOK_encloses` is in `TorchLean.Verify.CROWN.Op.CertSoundness`.

`lowerForwardToIR` runs a TorchLean program under the recording interpreter `RecordM`, builds the
graph of registered operators of the recorded typed graph (`IR.OpGraph`) and checks its
structure. On every input it returns the recorded graph's value (`IR.lowerForward_eq_ok`,
`IR.OpGraph.eval_toOp`), and no theorem relates the recorded graph to the source program. It
lowers stable log-softmax to the IR's `logSoftmax` node, which evaluates `Tensor.logSoftmax` itself.
The logarithm of computed softmax probabilities would round differently, even without underflow.
GELU is refused, because the verifier fragment cannot lower its gate division, and no numerically
different expansion takes its place. The IR has an elementwise division, and the operator-graph
IBP pass bounds its true and truncating forms (`TorchLean.Verify.CROWN.Op.Exec`), but the lowering
has no division case. Slicing uses the general slice, and selecting one coordinate along an axis
(`indexAt`) adds a reshape that drops the length-one dimension. Clamp puts each bound before the
value in its maximum and minimum, while `Context.clamp` puts the value first, so when the value
and a bound are both NaN the graph carries the bound's NaN. Clamp in a scalar context with
differentiation metadata (`Context.stopGradient?` is `some`) is refused, because the graph cannot
clear its tangents conditionally at the bounds.

The CROWN engine's executable transfer coverage is a separate boundary from the lowering. Its matrix
products promote vector operands and broadcast compatible leading batch axes. Convolution validates
its groups, dilation and both padding sides, and LayerNorm uses the full trailing normalized shape.
The [CROWN geometry contracts](CROWN/README.md#tensor-geometry) give each operator's limits and the
soundness theorems that cover them.

## Safety Verdicts

A public API that returns a safety verdict decides it in one of three ways: (A) a checker with a
soundness theorem about the real (`ℝ`) semantics, (B) agreement with the rounded binary32 affine
replay, which can fail to enclose the real graph, or (C) host `Float` or binary32 evaluation with no
theorem about the real graph. No verdict uses (B). `CROWNNode.replayDiagnostic` and
`CROWNNode.replayAgrees` compare a CROWN node certificate with the replay, and they are
diagnostics.

| Verdict (`verify` command) | Path | Theorem or reason |
| --- | --- | --- |
| `CROWNQuery.Query.check`, `CROWNQuery.acceptsJson`, `CROWNQuery.acceptsText` (`crown-query`) | A | `Query.check_sound`, `acceptsJson_sound`, `acceptsText_sound` |
| `CROWNNode.check`, `CROWNNode.verdict`, `FiniteArtifact.accepts` | A | `CROWNNode.verdict_ok`, `FiniteArtifact.accepts_sound`, `FiniteArtifact.accepts_graph_sound` |
| `Monotonicity.Certificate.check`, `Monotonicity.acceptsJson`, `Monotonicity.acceptsText` | A | `Certificate.check_sound`, `acceptsJson_sound`, `acceptsText_sound` |
| `Json.accepts`, `Json.acceptsText` | A, given a sound check | `Json.accepts_sound`, `Json.acceptsText_sound` |
| `Distillation.checkEquivalence` | A | `checkEquivalence_sound` |
| `Capture.checkError`, `Capture.checkPrediction` | A, about the binary reference runtime, not `ℝ` | `checkError_sound`, `checkPrediction_sound` |
| `IBPNode.check` | C | binary32 has no `LawfulBoundOps` instance, so `Op.Directed.encloses_of_run` does not apply |
| `IBP.check` (`ibp-cert`) | C | host `Float` |
| `AbCROWNLeaf.check` (`abcrown-leaf`) | C | exact rational consistency of the leaves only; the format has no graph |
| `Capture.check` (`ir`, Python `torchlean.verify`) | C | structure, types and runs on the reference runtime |
| `Trainer.Result.verify`, `Verification.forLowered`, `Verification.Result.fromBounds` (`torchlean-mlp-workflow`) | C | host `Float` bounds |
| `NumericalCertificate.check` | C | binary32 execution with no real enclosure |
| `camera-box3d-cert`, `pinn-cert`, `pinn-cli`, `pinn-dataset-check`, `spline-cert`, `ode`, `vnncomp-mnistfc` | C | host `Float`, binary32 or unproved exact rational checks |

## Public Imports

- `TorchLean.Verify.Trainer`: `trained.verify`, imported next to the trainer modules.
- `TorchLean.Verify.Certificate.IBP`, `TorchLean.Verify.Certificate.IBPNode` and
  `TorchLean.Verify.Certificate.CROWNNode`: output-box IBP certificates, per-node IBP certificates,
  and the verdict and replay diagnostic of per-node α-CROWN certificates.
- `TorchLean.Verify.Certificate.FiniteArtifact.Semantics`: the finite binary32 affine artifact
  checker and its soundness theorem `FiniteArtifact.accepts_graph_sound`.
- `TorchLean.Verify.Certificate.AbCROWNLeaf`: consistency checks of converted alpha-beta-CROWN leaf
  artifacts.
- `TorchLean.Verify.Certificate.CROWNQuery.JSON`: exact dense/ReLU CROWN output-query checker and
  acceptance-to-safety theorems.
- `TorchLean.Verify.Monotonicity.JSON`: exact monotonicity decoder, checker, and soundness theorems.

## Commands To Try

### Exact CROWN output queries

Run `scripts/lake.sh exe verify -- crown-query <query.json>`, or import
`TorchLean.Verify.Certificate.CROWNQuery.JSON` and call
`TorchLean.Certificate.CROWNQuery.acceptsText`, on a document such as:

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

The query asks whether `relu(x) + relu(-x) < 3/2` for every real `x` in `[-1, 1]`.
Each query row means `weights * output + bias < 0`, or `≤ 0` when `strict` is false, and all rows
must hold. The checker keeps the affine dependence on the original input, recomputes linear
sign-split and α-ReLU transfers with the CROWN operations, and bounds the final query over the
input box. It checks that each proposed α lies in `[0, 1]`.

Parameters and endpoints are exact integer/fraction strings. Unsupported operations, malformed
dimensions, invalid fractions, reversed boxes, empty input/output/query dimensions, repeated
object keys and invalid α values reject. No external affine bounds are trusted. Failure means the
checker didn't prove the query, which need not mean the property is false.

`Query.check_sound`, `acceptsJson_sound`, and `acceptsText_sound` prove that acceptance implies
every decoded output inequality for every real input in the decoded box. The model is an
`nn.Sequential` of `nn.Layer.linear` and `nn.Layer.op .relu` layers, evaluated over the reals by
`nn.Sequential.eval`, which is total. These theorems require no producer-soundness,
local-transfer, or graph-coverage hypothesis.

The format is separate from the binary32 node-replay JSON format, and it covers no ONNX
translation, floating-point deployment, β dual variables, cuts or branch-tree coverage. A model
producer has to establish that the decoded network is the model it intends to verify.

### Exact monotonicity certificates

Import `TorchLean.Verify.Monotonicity.JSON` and use `TorchLean.Monotonicity.acceptsText` on a
document such as:

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
dimensions and rejects malformed rational strings, numeric JSON parameters, repeated object keys
and unsupported operations. The weight check accepts nonnegative entries and allows arbitrary
biases. It's sufficient but incomplete: some monotone networks contain negative weights and are
rejected.

`check_sound` is one theorem over any linearly ordered field with a lawful `Context`, such as `ℝ`,
or `ℚ` with `TorchLean.RationalAlgebraic` open: the model of an accepted certificate preserves
componentwise order (`PreservesOrder`). `acceptsJson_sound` and `acceptsText_sound` apply it at
`ℝ`, so every accepted document decodes to an `nn.Sequential` model whose real evaluation
preserves componentwise order for every ordered pair of real inputs. None of the three needs a
producer-soundness or local-transfer hypothesis. They concern the model recorded in the document,
and its correspondence to an ONNX file, deployment arithmetic or CROWN/branch-and-bound
certificates is outside this format.

### Other verification workflows

Run the curated verification suite:

```bash
scripts/lake.sh exe verify -- all
```

The commands of the TorchLean-native graph workflows are in the
[verification examples](../../TorchLeanExamples/Verification/README.md#graph-workflows).

Run a compact consistency check of a converted alpha-beta-CROWN leaf artifact:

```bash
scripts/lake.sh exe verify -- abcrown-leaf
```

The checker confirms that the leaf boxes lie inside the root and together cover it, and that each
leaf has a coordinate whose reported lower bound exceeds its threshold. Those lower bounds are the
producer's numbers and TorchLean doesn't recompute them, so a pass means the artifact is
consistent and says nothing about the network.

The IBP certificate fixture and what its check establishes are in the
[verification examples](../../TorchLeanExamples/Verification/README.md#ibp-certificate).

The application checkers `pinn-cert`, `pinn-cli`, `pinn-dataset-check`, `ode`, `spline-cert`,
`camera-box3d-cert` and `vnncomp-mnistfc` are documented in the example READMEs listed under
[Example Groups](../../TorchLeanExamples/Verification/README.md#example-groups).

The command registry is `TorchLeanExamples.Verification.CLI`. `scripts/lake.sh exe verify --help`
prints the available commands.

## What A Checker Should Say

A checker names five things in its code or README:

1. the object being checked,
2. the file format or runtime object that may provide candidate data,
3. the predicate recomputed inside Lean,
4. the theorem, if any, that lifts the check to a mathematical statement,
5. the producer, solver, runtime or floating-point assumptions left outside Lean.

A successful run then identifies the object, the predicate, the theorem layer when one applies,
and the producer or runtime assumptions it leaves open.

## References

- IBP: Gowal et al., 2018, "On the Effectiveness of Interval Bound Propagation for Training
  Verifiably Robust Models".
- CROWN: Zhang et al., 2018, "Efficient Neural Network Robustness Certification with General
  Activation Functions".
- DeepPoly: Singh et al., 2019, "An Abstract Domain for Certifying Neural Networks".
- LiRPA: Xu et al., 2020, "Automatic Perturbation Analysis for Scalable Certified Robustness and
  Beyond".

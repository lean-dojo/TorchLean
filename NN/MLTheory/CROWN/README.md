# CROWN / LiRPA in TorchLean

This folder contains TorchLean's CROWN/LiRPA-style bound propagation code, certificate data
structures, and proof files. It is the mathematical engine behind several verification workflows:
TorchLean-native IBP/CROWN examples, external certificate checks, VNN-COMP-style exported suites,
and Lyapunov/controller experiments.

## Main Files

1. `Core.lean`: interval boxes (`Box`) and the basic affine form container (`AffineVec`).
   `Flatbox.lean` holds `FlatBox`, the interval container over flattened tensor values that the
   graph engine and the flattened transfer rules share.
2. `Models/Mlp.lean`: vector-in/vector-out CROWN development for small MLP-style networks. It uses
   the canonical executable ReLU relaxations from `Runtime/Ops.lean` together with its IBP bounds
   and model-specific affine composition.
3. `Graph.lean` and `Graph/`: graph-based LiRPA over TorchLean's op-tagged IR graphs. The graph
   engine stores per-node interval boxes, affine forms, parameter stores, and transfer state.
4. `Operators.lean` and `Operators/`: transfer rules for ReLU-family activations, arithmetic,
   convolution, pooling, batch normalization, reductions, slicing, and trigonometric operations.
5. `Cert/`: alpha-CROWN and alpha-beta-CROWN certificate data structures.
6. `Proofs/`: theorem-backed pieces of the CROWN development. `GraphCertSoundness/` proves IBP
   certificate soundness, with `cert_encloses_semantics` dispatching over the operator files in
   `Main/`. `GraphAlphaCrownTransferSoundness/` proves `alphaCrown_transfer_sound` and
   `alphaBetaCrown_transfer_sound` as short dispatches over `Alpha/` and `AlphaBeta/`; its
   `EndToEnd.lean` discharges their `IBPEnclosesVals` hypothesis from the IBP theorem
   (`ibp_encloses_vals_of_cert_local_ok`) and states the composed corollaries
   `alphaCrown_cert_encloses_semantics`, `alphaBetaCrown_cert_encloses_semantics'`, and
   `alphaCrown_cert_encloses_evalGraphRec`. `GraphRunibpEndToEnd.lean` connects the proof-side
   IBP pass to the engine's executable `runIBP` (`runIBP_eq_runIBP?`, `runIBP_encloses_evalGraphRec`)
   and `GraphRuntimeBridge.lean` proves per-node agreement between the runtime evaluator and the
   semantics (`evalNode_bridge`), then composes it over a whole topologically ordered graph
   (`denoteAll_semLocalOK`).
   `AlphaReLULowerBound.lean` is the shared scalar lower-bound theorem used by both alpha-CROWN and
   alpha/beta-CROWN proofs.
7. `Runtime/Ops.lean`: the canonical executable ReLU relaxation definitions used by both the graph
   engine and MLP development, kept separate from heavier proof imports.

## Bound Computation And Proofs

CROWN-style code has three distinct jobs:

- represent bounds and affine relaxations;
- compute or check transfer rules over supported operators;
- prove that accepted bounds imply a semantic property of the graph or model.

Not every executable bound pass has the same theorem coverage. The proof files name the fragments
that currently have Lean support, while executable workflows and JSON checkers can still be useful
as diagnostics or artifact checks. When writing a claim, cite the strongest available support:
runtime report, checked certificate, transfer-rule assumption, or theorem.

Support depends on the IR operation, tensor geometry, parameter payload, and scalar backend.
Successful model construction or lowering does not guarantee that a bound pass can handle it.
The [verification guide](../../../home_page/blueprint/TorchLeanBlueprint/Guide/Ch4_Verification/Verification.lean)
states the theorem fragments and their premises; runtime support is broader.

## Claim Shapes

The same graph can support several levels of claim, and the wording should identify which one is
being used.

| Claim | Evidence to cite |
| --- | --- |
| A bound pass ran on a graph | the runtime command, graph id/output id, input box, and printed bound result |
| A JSON artifact was accepted | the checker module, schema name, artifact path, and recomputed predicate |
| A graph certificate is sound | the theorem in `Proofs/`, the graph semantics, and the hypotheses discharged by the checker |
| An external verifier found the leaf | the external producer/provenance plus the Lean-checked leaf artifact |
| A finite-precision bound is being used | the scalar format and numerical correspondence required by the caller |

For the checked exact-real result on finite binary32 artifacts, see
[certificate acceptance](#certificate-acceptance) below.

For example, an alpha-beta-CROWN leaf artifact represents one exported terminal leaf: boxes, lower
bounds, thresholds, labels, and the witness comparison represented by the schema. A full producer
claim additionally needs provenance for the external branch-and-bound run that generated the leaf.

## Graph Engine And Proofs

The graph engine works over `NN.IR.Graph` node ids and payload stores. A typical workflow creates or
imports a graph, attaches an input box, computes per-node IBP boxes, and then propagates affine
forms for a selected output or objective. Operator files provide the transfer rules; proof files say
which rules have soundness statements or which assumptions remain.

### Tensor geometry

The graph stores tensor shapes even though its boxes and affine coefficients use flat coordinates.
The shape determines which coordinates participate in each operation:

| Operation | Executable transfer contract |
| --- | --- |
| Concatenation | Any valid axis; at least two parents; matching dimensions off that axis. |
| Binary matmul | Broadcastable leading axes, vector promotion, and matching inner dimensions. |
| Convolution | Checked groups, dilation, stride, asymmetric padding, and leading batch axes. |
| Axis sum/mean | A nonempty reduced axis; other axes, including empty leading axes, are preserved. |
| Average pooling | Positive per-axis kernels and strides over any spatial suffix; padded zeros count in the divisor. |
| Eval BatchNorm | A valid channel axis with matching parameter lengths; the input shape is preserved. |
| LayerNorm | `axis` begins the complete normalized suffix; payload shape equals that suffix. |
| Softmax value range | Any valid axis; extent one gives `[1,1]`, other extents give `[0,1]`. |
| Elementwise interval maps | Preserve arbitrary shapes, including scalar and empty shapes. |

Axis sums use directed accumulation; means and average pooling also enclose the count before
division. Eval BatchNorm propagates intervals through stabilization, square root, normalization,
scale, and bias. It returns no bound when required finite arithmetic or a positive denominator
cannot be established. These executable transfers do not extend the `EngineCore` theorem fragment
described below.

Concatenation preserves parent order, including empty dimensions and repeated parent ids.
Values, interval endpoints, and affine rows follow the same coordinate map. Backward propagation
splits the output coefficients along that map and accumulates every occurrence of a repeated
parent. For `concat(x, x)`, the two coefficient slices both contribute to `x`.
The proof-side certificate evaluator also uses this layout. Both `Supported` and the executable
`EngineCore` theorem include concatenation. `GraphRuntimeBridge` proves agreement with the IR
evaluator along every valid axis: moving that axis to the front, concatenating, and undoing the
permutation gives the same coordinates used by the certificate.

For matrix multiplication, the leading axes broadcast before the final two axes contract.
For example, `[1,4,5]` times `[2,5,6]` produces `[2,4,6]`. A vector is promoted for the
contraction and its introduced axis is removed afterward: `[5]` times `[2,5,6]` produces
`[2,6]`, while `[5]` times `[5]` produces a scalar.
The directed backward path bounds a binary product's active objective against its output IBP box.
It does not propagate coefficients through both operands, so this fallback can lose correlations.

A LayerNorm on `[2,2,2]` beginning at axis one normalizes two rows of four coordinates;
a payload shaped `[4]` is rejected because the normalized suffix is `[2,2]`. A leading batch
may be empty, but the normalized suffix must contain at least one coordinate. The elementwise
maps in `Runtime.Ops.IBP` (`mapMinmax`, `sigmoid`, `tanh`, `sin`, and `cos`) apply the same
scalar formula at every coordinate while preserving the input shape.

Convolution payloads store a dense input-channel axis and select the channels belonging to each
group. They do not use a packed `inChannels / groups` weight axis. Affine convolution transfers
construct a matrix with one row per output scalar and one column per input scalar, including
leading batch coordinates. This dense representation can be expensive even when the convolution
it represents is sparse; general shape support is not a memory or speed guarantee.

In `Proofs/Conv.lean`, `ConvProof.conv_linear_matrix_add_bias_eq_grouped_conv` proves that this
matrix applied to the flattened input, plus the broadcast bias, equals the flattened grouped
convolution over real tensors. It covers dilation, asymmetric padding, and leading batches.
Graph execution additionally validates the configuration; rounded execution needs its own
numerical correspondence.

`Proofs/ConvDerivatives.lean` differentiates the actual grouped convolution over arbitrary leading
batch dimensions and proves its input adjoint. The derivative pass evaluates the same geometry
with zero bias, including when an upstream nonlinear operation supplies a nonzero mixed derivative.
`Proofs/ConvEnclosure.lean` proves enclosure for the directed convolution's channel and kernel
folds over real endpoints, and relates their result to the same grouped convolution.
The checked graph transfer uses that result in `Proofs/ConvGraphEnclosure.lean`; convolution
is included in the real certificate induction and the executable `EngineCore` theorem.
`Proofs/GraphConvBridge.lean` also proves that successful IR convolution agrees with the flat
certificate evaluator, deriving the checked geometry from the actual execution.

LayerNorm's first and mixed derivative transfers operate on independent normalization rows, with
the stored scale and positive epsilon. `Graph/Proofs/LayerNormDerivativeEnclosure.lean` proves that
the actual directed row calculation encloses the first and mixed derivatives of the real
normalization. It requires enclosed upstream values and derivatives, the scalar operation laws,
and exact interpretations of the literals two through four (zero and one are laws of
`LawfulBoundOps`); a corollary specializes to real endpoints. A full graph derivative-pass induction remains separate.
For values, `Proofs/LayerNormEnclosure.lean` proves the directed row sequence encloses
`Spec.layerNorm` under the endpoint and nonlinear operation laws, with zero and one interpreted
exactly. This includes the actual directed sum and count used to compute a mean.
Softmax derivative transfers also use independent rows along any valid axis. They require the
scalar backend's ideal coupled-derivative contract; finite backends that do not satisfy it still
leave the derivative unresolved.
The `EngineCore` end-to-end IBP theorem does not cover LayerNorm, softmax, or MSE loss.
The proof-side payload-backed unary matrix map also has a different interface from binary graph
matrix multiplication. Read the theorem's graph fragment and hypotheses before applying it to
one of these computations.

Endpoint arithmetic is organized by four interfaces in `BoundOps.lean`. `BoundOps` contains the
executable lower and upper operations. `LawfulBoundOps` interprets their endpoints as real numbers
and proves that each lower result is below the exact operation and each upper result is above it.
Sound arithmetic lemmas require both interfaces; defining executable operations alone does not
establish an enclosure theorem. `NonlinearBoundOps` contains interval transfers for division,
square root, transcendental functions, bounded activations, and layer normalization. A backend
returns `none` when it has no justified enclosure. The graph pass then leaves that node unresolved;
it does not substitute an ordinary host-library value. `LawfulNonlinearBoundOps` proves that every
successful nonlinear transfer encloses the corresponding real operation and relates the
layer-normalization and coupled-derivative flags to explicit mathematical obligations.

The current instances make the numerical boundary visible:

- real endpoints use exact arithmetic and satisfy `LawfulBoundOps` definitionally;
- `FP32` rounds exact-real endpoints outward to the binary32 grid and has a proved
  `LawfulBoundOps` instance;
- FloatLib binary32 uses proved directed division and square root, while exponential and
  logarithmic transfers remain unavailable. Finite-path soundness is stated in the IEEE semantics
  modules rather than as a global ordered instance over NaNs and infinities;
- host `Float` and `Float32` widen basic operations by one adjacent value and do not claim directed
  transcendental-library results or provide a global `LawfulBoundOps` instance.

`runIBP` records an unavailable transfer as `none`; the output query fails if its requested box is
missing. For example, direct `exp` and `log` bounds are unavailable on both CLI backends
(`native`/`ieee`), while softmax can use the coarse coordinate ranges above. Backend support also
differs: native `Float32` leaves sigmoid and tanh unresolved, whereas FloatLib binary32 supplies
their global codomain bounds.

`outputBoxCROWN?` uses a forward affine sweep on exact-reassociation backends. Rounded backends
instead request directed backward bounds for the output coordinates. When a node has only an IBP
enclosure, the directed pass can bound its active objective against that box without propagating
coefficients to its parents. Binary matmul and MSE loss take this fallback, so the optional
attention/LayerNorm/MSE example can finish without propagating CROWN coefficients through attention.
A CROWN result need not tighten IBP. The rounded backward soundness theorem in
`Proofs/DirectedBackwardEvaluation.lean` covers coefficient propagation, interval fallbacks, and
final affine-bound evaluation. It assumes lawful directed arithmetic, the real graph equations for
the stored parameters, and enclosing IBP bounds.

The rounded forward pass itself is proved sound in `Proofs/DirectedIBPSoundness.lean`.
`runIBP_encloses` shows that every box `runIBP` returns encloses its node's real value, for any
backend with `LawfulBoundOps` and `LawfulNonlinearBoundOps`, provided parents precede their
consumers, the input boxes contain the real inputs, the real point satisfies `NodeEquation`, and
every node kind passes `ibpForwardSupported`. The supported kinds are inputs, constants, `detach`,
`reshape`, `flatten`, `add`, `sub`, `mulElem`, `relu`, `linear`, unary `matmul`, `sum`, `exp`,
`log`, `sqrt`, `inv`, `tanh`, `sigmoid`, `sin`, and `cos`. On such graphs `GraphPoint.ofRunIBP`
discharges the IBP hypothesis, and `runCROWNBackwardObjective_encloses_runIBP`,
`directedNodeBounds_encloses_runIBP`, and `backwardObjectiveBox_encloses_runIBP` are end-to-end
rounded statements.

The rounded forward transfers without a proof are convolution, `concat`, `transpose`, `permute`,
binary `matmul`, `abs`, `maxElem`, `minElem`, `softplus`, `safeLog`, the pools, `broadcastTo`,
`reduceSum`, `reduceMean`, `mseLoss`, `batchNormEval`, `layernorm`, `softmax`,
`hardMaskedSoftmax`, and the random nodes. For these the IBP enclosure is proved only over real
endpoints (`runIBP_encloses_evalGraphRec`, where covered) or remains a hypothesis. The rounded
backward sweep does not use per-neuron ReLU slopes: the rounded branch of
`runCROWNBackwardObjectiveLowerWithReluAlpha` returns `none` when any slope is supplied.
Relating a rounded result to a separate native execution also needs the finite-precision bridge
described in `NN/Proofs/RuntimeApprox`.

Use this split when adding operators:

1. Add the executable interval/affine transfer rule.
2. State the shape and payload assumptions it needs.
3. Add or extend the proof layer soundness theorem when the operator supports formal
   graph-certificate claims.
4. Add a small verifier example or fixture if the rule is exposed through `lake exe verify`.

That keeps runtime diagnostics, accepted certificates, and theorem-backed graph claims connected
while preserving the distinction between execution evidence, checker acceptance, and theorem-backed
graph claims.

### Certificate acceptance

Node certificates are checked by the library functions `checkCROWNNodeCertificate` and
`checkAlphaBetaCROWNNodeCertificate`. `lake exe verify` has no command for them. Both finish with a
pure complete replay. `certificateAccepts_eq_true` and
`AlphaBetaCROWNNodeCertificate.accepts_eq_true` turn a successful binary32 replay into
`CrownCertLocalOK`. The result is a theorem about the imported
binary32 transcript. Applying a real-semantic enclosure theorem additionally requires the
appropriate transfer and finite-precision refinement hypotheses.

For nonempty vector linear/ReLU chains, `NN.Verification.Cert.FiniteArtifact.accepts` checks
the supplied bounds against exact rational transfers instead of binary32 replay. It rejects
inward-rounded affine coefficients even when they agree with binary32 replay, and it accepts
bounds from any producer that dominate the exact transfers. `FiniteArtifact.accepts_graph_sound`
then proves that the original graph, interpreted with the exact real values of those binary32
parameters, satisfies every requested output inequality throughout the input box. The checker
requires bounds for the whole chain and at least one inequality. Native execution error remains
a separate obligation.

## Subfolders

- `Graph/`: graph engine, backward propagation, and graph-level theorem statements.
- `Operators/`: op-specific IBP and affine transfer rules.
- `Propagation/`: specialized propagation routines such as backward or sign-split passes. The
  canonical `IBP.matPos`/`IBP.matNeg` weight decomposition lives in `Core.lean` and is shared by
  interval and graph-CROWN linear rules.
- `Cert/`: alpha/alpha-beta certificate structures.
- `Lyapunov/`: controller and Lyapunov-oriented CROWN workflows. Imported numerical bounds support
  a theorem only after the caller proves `LyapunovCert.ValidFor`.
  The two Lean-executed pipelines share lowered gradient search and loss-box verification through
  `Lyapunov/TwoStage/LossAnalysis.lean` (`projectedGradientStep` and `checkLossBox`).
- `Proofs/`: soundness theorems and proof layer overviews; `Proofs/Overview.lean` is the map.
- `Extras/`: optional helpers and proof toolboxes.
- `Tactics/`: diagnostic commands for running an external producer and inspecting its certificates.

## Optional Modules

- `Extras/IntervalLemmas.lean`: interval-arithmetic lemmas over `ℝ`.
- `Extras/AlphaConfig.lean`: data structures for alpha-optimized relaxations.
- `Extras/FP32.lean` and `Extras/BoundOpsIEEE32Exec.lean`: finite-precision specializations and
  executable IEEE32 connections.

## Optional input subdivision

`Graph.refinedIBPOutput? g ps inputId outId splitBudget` refines an output box by splitting one
selected graph input. It uses the existing transfers for every supported layer; there is no
Transformer-specific path. Other graph inputs and parameters retain their original boxes/values.

For example, over `x ∈ [-1, 1]`, independent interval evaluation of
`ReLU(x) + ReLU(-x)` gives approximately `[0, 2]`. Splitting at zero and taking the hull of both
results gives approximately `[0, 1]`. Likewise, splitting can improve the lower bound on `x * x`.
A dense layer with independent input coordinates may already have tight bounds and see no gain.
Input-independent activation fallbacks such as `[0, 1]` also need not improve.

The budget counts binary splits, not depth: there are at most `2 * splitBudget + 1` IBP calls.
Zero leaves the ordinary enclosure calculation unchanged for a valid input/output selection.
No representable interior midpoint means no split. Both child results are required; a failed branch
retains the parent result. Child results are combined by a hull, then intersected with the parent
bound so a successful refinement does not widen it. Invalid boxes or incompatible output dimensions
are rejected before they can become graph output certificates.

The artifact checker accepts the same option:
`NN.Verification.IBPCert.check g ps outId path (refinement := some (inputId, splitBudget))`.
Its default performs a single pass. A split budget is an explicit runtime/precision
tradeoff, rather than a hidden cost added to every verification request.

In `NN.MLTheory.CROWN.Proofs.GraphRefinement`, `Graph.Refinement.splitAt_covers` proves that every real input in a parent box belongs to at least
one child. The enclosure procedure still needs sound transfer rules: subdivision does not prove
universal soundness of rounded LayerNorm or other backend operations. The maintained tests cover
containment, actual tightening, Float/Float32/IEEE32Exec, multiple inputs, failed branches, invalid
endpoints, and the artifact-checker option.
